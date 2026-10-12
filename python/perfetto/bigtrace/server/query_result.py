# Copyright (C) 2026 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fast, incremental decoding of trace processor query results.

TraceProcessor.query() decodes the whole QueryResult proto up front, which is
slow with the pure-Python protobuf runtime (~20us per row of `slice`) and holds
the GIL, stalling every other query and HTTP request. This splits the proto
into its encoded batches, which callers decode only when they need the rows.
"""

import base64
import os
import struct
import threading
from typing import Any, Iterator, List, Tuple

from perfetto.trace_processor.api import TraceProcessor, TraceProcessorException

# Field numbers from protos/perfetto/trace_processor/trace_processor.proto.
_QUERY_ARGS_SQL = 1
_RESULT_COLUMN_NAMES = 1
_RESULT_ERROR = 2
_RESULT_BATCH = 3
_BATCH_CELLS = 1
_BATCH_VARINTS = 2
_BATCH_FLOAT64S = 3
_BATCH_BLOBS = 4
_BATCH_STRINGS = 5

# CellsBatch.CellType values.
_CELL_NULL = 1
_CELL_VARINT = 2
_CELL_FLOAT64 = 3
_CELL_STRING = 4
_CELL_BLOB = 5


class QueryResult:
  __slots__ = ("columns", "batches", "raw_bytes")

  def __init__(self,
               columns: List[str],
               batches: List[memoryview],
               raw_bytes: bytes = b""):
    self.columns = columns
    # Encoded CellsBatch protos; see decode_batch().
    self.batches = batches
    self.raw_bytes = raw_bytes

  def __iter__(self) -> Iterator[Any]:
    yield self.columns
    yield self.batches


def query(tp: TraceProcessor, sql: str) -> QueryResult:
  """Runs `sql` and returns its result with the batches still encoded.

  Decode batches with decode_batch(), so callers only pay for the rows they
  need. Raises TraceProcessorException if the query fails.
  """
  sql_bytes = sql.encode("utf-8")
  body = bytes([_QUERY_ARGS_SQL << 3 | 2]) + _varint(len(sql_bytes)) + sql_bytes
  conn = tp.http.conn
  conn.request("POST", "/query", body=body)
  with conn.getresponse() as response:
    data = response.read()

  columns: List[str] = []
  batches: List[memoryview] = []
  for field, value in _fields(memoryview(data)):
    if field == _RESULT_COLUMN_NAMES:
      columns.append(bytes(value).decode("utf-8", "replace"))
    elif field == _RESULT_ERROR and len(value):
      raise TraceProcessorException(bytes(value).decode("utf-8", "replace"))
    elif field == _RESULT_BATCH:
      batches.append(value)
  return QueryResult(columns, batches, data)


_TLS = threading.local()


def _parse_header_into(mv: memoryview, columns: List[str], sample: List[bytes],
                       need_sample: bool) -> None:
  pos = 0
  end = len(mv)
  while pos < end:
    try:
      tag, pos = _read_varint(mv, pos)
      field = tag >> 3
      wire = tag & 7
      if wire == 2:
        length, pos = _read_varint(mv, pos)
        if pos + length > end:
          break
        val = mv[pos:pos + length]
        pos += length
        if field == _RESULT_COLUMN_NAMES:
          columns.append(bytes(val).decode("utf-8", "replace"))
        elif field == _RESULT_ERROR and length:
          raise TraceProcessorException(bytes(val).decode("utf-8", "replace"))
        elif field == _RESULT_BATCH:
          if need_sample and not sample:
            sample.append(bytes(val))
          break
      elif wire == 0:
        _, pos = _read_varint(mv, pos)
      elif wire == 1:
        pos += 8
      elif wire == 5:
        pos += 4
      else:
        break
    except IndexError:
      break


def query_to_spool_file(
    tp: TraceProcessor,
    sql: str,
    spool_path: str,
    need_sample: bool = False,
) -> Tuple[List[str], List[bytes], int]:
  """Streams TP HTTP `/query` socket directly to `spool_path` without 100MB allocs."""
  buf: bytearray = getattr(_TLS, "io_buf", None)
  if buf is None:
    buf = bytearray(1 << 20)
    _TLS.io_buf = buf
  mv = memoryview(buf)

  sql_bytes = sql.encode("utf-8")
  body = bytes([_QUERY_ARGS_SQL << 3 | 2]) + _varint(len(sql_bytes)) + sql_bytes
  conn = tp.http.conn
  conn.request("POST", "/query", body=body)

  columns: List[str] = []
  sample: List[bytes] = []
  total_bytes = 0
  header_parsed = False
  hdr_buf = bytearray()
  tail = b""

  fd = os.open(spool_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
  try:
    with conn.getresponse() as response:
      while True:
        n = response.readinto(mv)
        if not n:
          break
        chunk = mv[:n]
        os.write(fd, chunk)
        total_bytes += n
        if not header_parsed:
          hdr_buf.extend(chunk)
          if len(hdr_buf) >= (1 << 18):
            header_parsed = True
            _parse_header_into(
                memoryview(hdr_buf), columns, sample, need_sample)
            hdr_buf.clear()
        if n >= 512:
          tail = bytes(chunk[n - 512:])
        else:
          tail = (tail + bytes(chunk))[-512:]
  except Exception:
    os.close(fd)
    try:
      os.unlink(spool_path)
    except OSError:
      pass
    raise
  os.close(fd)

  if not header_parsed and hdr_buf:
    _parse_header_into(memoryview(hdr_buf), columns, sample, need_sample)

  if b"\x12" in tail and os.path.isfile(spool_path):
    with open(spool_path, "rb") as f:
      full_data = f.read()
    for field, value in _fields(memoryview(full_data)):
      if field == _RESULT_ERROR and len(value):
        try:
          os.unlink(spool_path)
        except OSError:
          pass
        raise TraceProcessorException(bytes(value).decode("utf-8", "replace"))

  return columns, sample, total_bytes


def row_count(batch: memoryview, column_count: int) -> int:
  """Returns the number of rows in a batch without decoding it."""
  if column_count == 0:
    return 0
  for field, value in _fields(batch):
    if field == _BATCH_CELLS:
      # Cell types are all below 128, so each packed varint is one byte.
      return len(value) // column_count
  return 0


def decode_batch_flat(data: bytes) -> List[Any]:
  """Decodes a batch into a flat row-major list. Blobs are base64 strings."""
  cells = b""
  varints: List[int] = []
  float64s: Tuple[float, ...] = ()
  blobs: List[str] = []
  strings: List[str] = []
  for field, value in _fields(memoryview(data)):
    if field == _BATCH_CELLS:
      cells = bytes(value)
    elif field == _BATCH_VARINTS:
      varints = _packed_varints(value)
    elif field == _BATCH_FLOAT64S:
      float64s = struct.unpack(f"<{len(value) // 8}d", value)
    elif field == _BATCH_BLOBS:
      blobs.append(base64.b64encode(value).decode("ascii"))
    elif field == _BATCH_STRINGS:
      # NUL-terminated strings, concatenated.
      strings = bytes(value).decode("utf-8", "replace").split("\0")[:-1]

  sources = {
      _CELL_VARINT: iter(varints).__next__,
      _CELL_FLOAT64: iter(float64s).__next__,
      _CELL_STRING: iter(strings).__next__,
      _CELL_BLOB: iter(blobs).__next__,
  }
  null = lambda: None
  next_value = [sources.get(t, null) for t in range(_CELL_BLOB + 1)]
  return [next_value[t]() for t in cells]


def decode_batch(data: bytes, column_count: int) -> List[List[Any]]:
  """Decodes a batch into rows. Blobs are base64 so values are JSON-safe."""
  if column_count == 0:
    return []
  values = decode_batch_flat(data)
  return [
      values[i:i + column_count] for i in range(0, len(values), column_count)
  ]


def _fields(data: memoryview) -> Iterator[Tuple[int, Any]]:
  """Yields (field number, value) for each field of a proto message.

  Length-delimited values are memoryviews; others are ints (fixed-width ones
  are left undecoded, as none of the fields read here use them).
  """
  pos, end = 0, len(data)
  while pos < end:
    tag, pos = _read_varint(data, pos)
    wire_type = tag & 7
    if wire_type == 0:
      value, pos = _read_varint(data, pos)
    elif wire_type == 2:
      size, pos = _read_varint(data, pos)
      value, pos = data[pos:pos + size], pos + size
    elif wire_type == 1:
      value, pos = 0, pos + 8
    elif wire_type == 5:
      value, pos = 0, pos + 4
    else:
      raise TraceProcessorException(
          f"Malformed query result (wire type {wire_type})")
    yield tag >> 3, value


def _read_varint(data: memoryview, pos: int) -> Tuple[int, int]:
  value = shift = 0
  while True:
    b = data[pos]
    pos += 1
    value |= (b & 0x7F) << shift
    if b < 0x80:
      return value, pos
    shift += 7


def _packed_varints(data: memoryview) -> List[int]:
  """Decodes packed int64s, the hot loop when decoding results."""
  out: List[int] = []
  append = out.append
  value = shift = 0
  for b in bytes(data):
    value |= (b & 0x7F) << shift
    if b < 0x80:
      # int64 is two's complement over 64 bits.
      append(value - (1 << 64) if value >= 1 << 63 else value)
      value = shift = 0
    else:
      shift += 7
  return out


def _varint(value: int) -> bytes:
  out = bytearray()
  while value >= 0x80:
    out.append(value & 0x7F | 0x80)
    value >>= 7
  out.append(value)
  return bytes(out)
