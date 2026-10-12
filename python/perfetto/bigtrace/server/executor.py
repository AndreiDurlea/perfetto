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
"""Runs queries over traces or DuckDB tables and streams results via DuckDB."""

import collections
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import itertools
import logging
import os
import re
import shutil
import tempfile
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import uuid

from perfetto.trace_processor.api import (
    TraceProcessorConfig,
    TraceProcessorException,
)
from perfetto.bigtrace.server import query_result
from perfetto.bigtrace.server.catalog import (
    SettingsOverrides,
    TraceCatalog,
    TraceEntry,
)
from perfetto.bigtrace.server.duckdb_engine import (
    DuckDBEngine,
    STATUS_FIELDS,
    TERMINAL_STATUSES,
    public_view,
)
from perfetto.bigtrace.server.filters import parse_experiment_filter
from perfetto.bigtrace.server.snapshot import SnapshotStore
from perfetto.bigtrace.server.trace_processor_pool import (
    PoolQuery,
    TraceProcessorPool,
)
from perfetto.bigtrace.server.util import (
    as_int,
    get_param,
    infer_column_type,
    iso_now,
    to_wire,
)

logger = logging.getLogger(__name__)

_DUCKDB_DIRECTIVE = re.compile(r"^\s*--\s*(?:@duckdb|engine\s*:\s*duckdb)\b",
                               re.IGNORECASE)
_STREAM_FLUSH_ROWS = 500_000
_SPOOL_FLUSH_TRACES = 1024
_SPOOL_FLUSH_ROWS = 2_000_000_000


def _spool_root() -> str:
  shm = "/dev/shm"
  try:
    if os.path.isdir(shm) and os.access(shm, os.W_OK | os.X_OK):
      st = os.statvfs(shm)
      if st.f_bavail * st.f_frsize > 4 * (1 << 30):
        return shm
  except OSError:
    pass
  return tempfile.gettempdir()


class QueryFailedError(Exception):
  pass


class _StreamBuffer:
  """Streams per-trace query batches into DuckDB via C++ `pf_scan_tp_proto`."""

  def __init__(self, duckdb: DuckDBEngine, table_name: str, query_uuid: str,
               sql: str):
    self.duckdb = duckdb
    self.table_name = table_name
    self.query_uuid = query_uuid
    self.sql = sql
    self.sql_columns: List[str] = []
    self.meta_columns: List[str] = []
    self.columns: List[str] = []
    self.schema_map: Dict[str, str] = {}
    self.all_uuids: List[str] = []
    self.created = False
    self.cols: List[List[Any]] = []
    self.spools: List[Tuple[str, str]] = []
    self.buffered_rows = 0
    self._seq = itertools.count(1)
    self._flush_lock = threading.Lock()
    self.spool_dir = os.path.join(_spool_root(),
                                  f"bigtrace_spool_{os.getpid()}_{query_uuid}")

  def next_spool_path(self) -> str:
    os.makedirs(self.spool_dir, exist_ok=True)
    return os.path.join(self.spool_dir, f"{next(self._seq):06d}.pb")

  def write_raw_spool(self, raw_bytes: bytes,
                      total_rows: int) -> Tuple[Optional[str], int]:
    """Writes raw `QueryResult` socket bytes directly to `/dev/shm` zero-copy."""
    if not raw_bytes or total_rows <= 0:
      return None, 0
    os.makedirs(self.spool_dir, exist_ok=True)
    path = os.path.join(self.spool_dir, f"{next(self._seq):06d}.pb")
    with open(path, "wb") as f:
      f.write(raw_bytes)
    return path, total_rows

  def write_trace_spool(self, batches: Sequence[Tuple[bytes, int]],
                        sql_cols: int) -> Tuple[Optional[str], int]:
    """Writes kept `CellsBatch` messages to `.pb` on worker thread without GIL/decode."""
    if not batches:
      return None, 0
    total_rows = sum(count for _, count in batches if count > 0)
    if total_rows <= 0:
      return None, 0
    os.makedirs(self.spool_dir, exist_ok=True)
    path = os.path.join(self.spool_dir, f"{next(self._seq):06d}.pb")
    parts: List[bytes] = []
    for batch, count in batches:
      if count <= 0:
        continue
      full = query_result.row_count(memoryview(batch), sql_cols)
      if sql_cols > 0 and count < full:
        # Truncate cell-type array length varint so C++ reads exact `count` rows
        trimmed = bytearray()
        for fnum, val in query_result._fields(memoryview(batch)):  # pylint: disable=protected-access
          if fnum == 1:
            sub = bytes(val)[:count * sql_cols]
          else:
            sub = bytes(val)
          trimmed.append((fnum << 3) | 2)
          trimmed.extend(query_result._varint(len(sub)))  # pylint: disable=protected-access
          trimmed.extend(sub)
        b_out = bytes(trimmed)
      else:
        b_out = batch
      parts.append(b"\x1a" + query_result._varint(len(b_out)) + b_out)  # pylint: disable=protected-access
    with open(path, "wb") as f:
      f.writelines(parts)
    return path, total_rows

  def ensure_initialized(self,
                         columns: Sequence[str],
                         schema_map: Dict[str, str],
                         sql_columns: Optional[Sequence[str]] = None,
                         meta_columns: Optional[Sequence[str]] = None) -> None:
    if not self.columns:
      self.columns = list(columns)
      self.cols = [[] for _ in self.columns]
    if sql_columns is not None and not self.sql_columns:
      self.sql_columns = list(sql_columns)
    if meta_columns is not None and not self.meta_columns:
      self.meta_columns = list(meta_columns)
    self.schema_map.update(schema_map)

  def add_spool(self, spool_path: Optional[str], trace_uuid: str,
                row_count: int) -> bool:
    if spool_path and row_count > 0:
      self.spools.append((spool_path, trace_uuid))
      self.buffered_rows += row_count
    return (len(self.spools) >= _SPOOL_FLUSH_TRACES or
            self.buffered_rows >= _SPOOL_FLUSH_ROWS)

  def add_batches(self, batches: Sequence[Tuple[bytes, int]], sql_cols: int,
                  meta_vals: Sequence[Any]) -> int:
    added = 0
    for batch, count in batches:
      if count <= 0:
        continue
      vals = query_result.decode_batch_flat(batch) if sql_cols > 0 else []
      end = count * sql_cols
      for col_idx in range(len(self.columns)):
        if col_idx < sql_cols:
          self.cols[col_idx].extend(vals[col_idx:end:sql_cols])
        else:
          meta_idx = col_idx - sql_cols
          mval = meta_vals[meta_idx] if meta_idx < len(meta_vals) else None
          self.cols[col_idx].extend([mval] * count)
      self.buffered_rows += count
      added += count
      if self.buffered_rows >= _STREAM_FLUSH_ROWS:
        self.flush()
    return added

  def flush(self) -> None:
    with self._flush_lock:
      if not self.columns:
        return
      if not self.created:
        self.table_name, _ = self.duckdb.create_streaming_table(
            self.table_name,
            self.columns,
            self.schema_map,
            query_uuid=self.query_uuid,
            sql=self.sql,
        )
        self.created = True
      if self.spools:
        to_spools = self.spools
        rows = self.buffered_rows
        self.spools = []
        self.buffered_rows = 0
        self.duckdb.ingest_proto_spools(
            self.table_name,
            self.sql_columns,
            self.meta_columns,
            self.schema_map,
            to_spools,
            rows,
            all_uuids=self.all_uuids,
        )
      elif self.buffered_rows > 0:
        to_insert = self.cols
        self.cols = [[] for _ in self.columns]
        self.buffered_rows = 0
        self.duckdb.append_columns(self.table_name, self.columns,
                                   self.schema_map, to_insert)

  def cleanup(self) -> None:
    shutil.rmtree(self.spool_dir, ignore_errors=True)


class BigtraceQueryExecutor:
  """Runs PerfettoSQL over traces or DuckDB SQL over materialized tables.

  Every query streams its rows directly into DuckDB (`result_1`, `result_2`,
  or a custom `table_name`, plus `latest`) and serves UI pages directly from
  DuckDB.
  """

  def __init__(
      self,
      catalog: TraceCatalog,
      tp_config: Optional[TraceProcessorConfig] = None,
      max_workers: int = 4,
      cache_size: int = 16,
      snapshots: Optional[SnapshotStore] = None,
      duckdb_engine: Optional[DuckDBEngine] = None,
  ):
    self.catalog = catalog
    self.cache_size = cache_size
    self.max_workers = max_workers
    self.pool = TraceProcessorPool(tp_config, snapshots)
    self.duckdb = duckdb_engine or DuckDBEngine()
    self.duckdb.sync_catalog(self.catalog)
    self._workers = ThreadPoolExecutor(max_workers=max_workers)
    self._lock = threading.RLock()
    self._executions: "collections.OrderedDict[str, Dict[str, Any]]" = (
        collections.OrderedDict())
    self._runs: List[threading.Thread] = []

  def close(self) -> None:
    with self._lock:
      exs = list(self._executions.values())
      runs = list(self._runs)
    for ex in exs:
      self._stop(ex)
    self.pool.close()
    self._workers.shutdown(wait=True, cancel_futures=True)
    for run in runs:
      run.join(timeout=10)
    self.duckdb.close()

  def _persist(self, write: Callable[..., Any], *args: Any) -> None:
    try:
      write(*args)
    except Exception as e:  # pylint: disable=broad-except
      logger.warning("Could not save query history: %s", e)

  def _remember(self, ex: Dict[str, Any]) -> None:
    with self._lock:
      self._executions[ex["queryUuid"]] = ex
      self._executions.move_to_end(ex["queryUuid"])
      finished = [
          k for k, v in self._executions.items()
          if v["status"] in TERMINAL_STATUSES
      ]
      for k in finished[:max(0, len(finished) - self.cache_size)]:
        del self._executions[k]

  def _load(self, query_uuid: str) -> Dict[str, Any]:
    with self._lock:
      ex = self._executions.get(query_uuid)
      if ex:
        self._executions.move_to_end(query_uuid)
        return ex
    ex = self.duckdb.get_execution(query_uuid)
    if ex is None:
      raise KeyError(query_uuid)
    ex["_cancel"] = threading.Event()
    ex["_active"] = set()
    self._remember(ex)
    return ex

  def list_executions(self) -> List[Dict[str, Any]]:
    by_uuid = {ex["queryUuid"]: ex for ex in self.duckdb.list_executions()}
    with self._lock:
      for query_uuid, ex in self._executions.items():
        by_uuid[query_uuid] = public_view(ex)
    return sorted(
        by_uuid.values(), key=lambda e: e.get("startTime") or "", reverse=True)

  def get_execution(self, query_uuid: str) -> Optional[Dict[str, Any]]:
    with self._lock:
      if query_uuid in self._executions:
        return public_view(self._executions[query_uuid])
    ex = self.duckdb.get_execution(query_uuid)
    return public_view(ex) if ex else None

  def get_status(self, query_uuid: str) -> Optional[Dict[str, Any]]:
    with self._lock:
      if query_uuid in self._executions:
        return public_view(self._executions[query_uuid], STATUS_FIELDS)
    return self.duckdb.get_status(query_uuid)

  def check_table_exists(self, table_name: str) -> Dict[str, Any]:
    exists_duck, resolved = self.duckdb.table_exists(table_name)
    all_exs = self.list_executions()
    in_flight = any(
        ex.get("status") == "IN_PROGRESS" and ex.get("tableName") in (
            table_name, resolved) for ex in all_exs)
    hist_names = {
        ex["tableName"]
        for ex in all_exs
        if ex.get("tableName") and (ex.get("status") == "IN_PROGRESS" or
                                    (ex.get("processedRows") or 0) > 0)
    }
    base = resolved or table_name
    suggested = self.duckdb.suggest_table_name(base, reserved=hist_names)
    return {
        "exists": exists_duck or in_flight,
        "resolvedTableName": base,
        "suggestedTableName": suggested,
    }

  def _stop(self, ex: Dict[str, Any]) -> None:
    """Stops a running query, killing the trace processors running it."""
    with self._lock:
      cancel = ex.get("_cancel")
      if cancel:
        cancel.set()
      batch_cancel = ex.get("_batch_cancel")
      if isinstance(batch_cancel, threading.Event):
        if all(
            e.get("_cancel", threading.Event()).is_set()
            for e in self._executions.values()
            if e.get("_batch_cancel") is batch_cancel):
          batch_cancel.set()
      active = list(ex.get("_active", ()))
      if "_active" in ex:
        ex["_active"].clear()
      shared_tps = {
          tp for other in self._executions.values()
          if other is not ex and self._wants_rows(other)
          for tp in other.get("_active", ())
      }
      to_close = [tp for tp in active if tp not in shared_tps]
      duck_cursor = ex.pop("_duck_cursor", None)
    for tp in to_close:
      tp.close()
    if duck_cursor is not None:
      try:
        duck_cursor.interrupt()
      except Exception:  # pylint: disable=broad-except
        pass

  def cancel_query(self, query_uuid: str) -> None:
    with self._lock:
      ex = self._executions.get(query_uuid)
      if ex and ex["status"] == "IN_PROGRESS":
        ex["status"] = "CANCELLED"
        ex["endTime"] = iso_now()
    if ex:
      self._stop(ex)
    self._persist(self.duckdb.mark_cancelled, query_uuid)

  def delete_execution(self, query_uuid: str) -> bool:
    table_name = None
    with self._lock:
      ex = self._executions.pop(query_uuid, None)
      if ex:
        table_name = ex.get("tableName")
        ex["_deleted"] = True
    if ex:
      self._stop(ex)
    if not table_name:
      stored = self.duckdb.get_execution(query_uuid)
      if stored:
        table_name = stored.get("tableName")
    self.duckdb.drop_table(table_name, query_uuid)
    return self.duckdb.delete_execution_record(query_uuid) or ex is not None

  @staticmethod
  def _parse_query_specs(
      params: Dict[str, Any],
      default_sql: str,
      default_table: Optional[str],
      default_limit: int,
  ) -> List[Tuple[str, Optional[str], int]]:
    raw_queries = get_param(params, "queries")
    specs: List[Tuple[str, Optional[str], int]] = []
    if isinstance(raw_queries, list) and raw_queries:
      for item in raw_queries:
        if isinstance(item, dict):
          q_sql = str(
              get_param(item, "perfetto_sql", "") or get_param(item, "sql", ""))
          q_tbl = get_param(item, "table_name")
          q_lim = max(0, as_int(get_param(item, "limit"), default_limit))
          if q_sql.strip():
            specs.append((q_sql, str(q_tbl) if q_tbl else None, q_lim))
        elif isinstance(item, str) and item.strip():
          specs.append((item, None, default_limit))
    if not specs and default_sql.strip():
      specs.append((default_sql, str(default_table) if default_table else None,
                    default_limit))
    return specs

  def execute(self,
              params: Dict[str, Any],
              is_async: bool = False) -> Dict[str, Any]:
    """Starts one or more queries. Sync calls block and return the first page."""
    sql = str(get_param(params, "perfetto_sql", ""))
    req_table = get_param(params, "table_name")
    limit = max(0, as_int(get_param(params, "limit"), 10000))
    specs = self._parse_query_specs(params, sql,
                                    str(req_table) if req_table else None,
                                    limit)
    if not specs:
      raise ValueError("perfetto_sql is required")

    first_sql = specs[0][0]
    engine = str(
        get_param(params, "engine", "") or
        get_param(params, "query_engine", "")).lower().strip()
    if not engine and _DUCKDB_DIRECTIVE.match(first_sql):
      engine = "duckdb"
    engine = "duckdb" if engine == "duckdb" else "traces"

    trace_limit = as_int(get_param(params, "trace_limit"), 0) or None
    settings = get_param(params, "settings", [])
    trace_filters = get_param(params, "trace_filters", [])
    trace_order_by = get_param(params, "trace_order_by")
    experiment_filter = parse_experiment_filter(
        get_param(params, "experiment_filter"))

    traces: List[TraceEntry] = []
    if engine == "traces":
      overrides = SettingsOverrides.parse(settings)
      if self.catalog.apply_overrides(overrides):
        self.duckdb.sync_catalog(self.catalog)
      traces, _ = self.catalog.filter_and_sort(
          filters=trace_filters,
          order_by=trace_order_by,
          experiment_filter=experiment_filter,
          trace_uuids=overrides.trace_uuids,
          limit=trace_limit,
      )

    req_meta = list(get_param(params, "trace_metadata_columns", []))
    if engine == "traces" and "_trace_uuid" not in req_meta:
      req_meta = req_meta + ["_trace_uuid"]

    norm_settings = [{
        "settingId": get_param(s, "setting_id", ""),
        "values": [str(v) for v in s.get("values") or []],
        "category": s.get("category", "TRACE_ADDRESS"),
    } for s in settings if isinstance(s, dict)]

    batch_cancel = threading.Event()
    executions: List[Dict[str, Any]] = []
    with self._lock:
      reserved_tables = {
          e["tableName"]
          for e in self._executions.values()
          if e.get("status") == "IN_PROGRESS" and e.get("tableName")
      }
      for q_sql, q_table, q_lim in specs:
        q_uuid = str(uuid.uuid4())
        t_name = self.duckdb.next_table_name(q_table, reserved=reserved_tables)
        reserved_tables.add(t_name)
        ex: Dict[str, Any] = dict(
            queryUuid=q_uuid,
            status="IN_PROGRESS",
            startTime=iso_now(),
            endTime=None,
            processedRows=0,
            processedTraces=0,
            totalTraces=len(traces) if engine == "traces" else 1,
            perfettoSql=q_sql,
            queryEngine=engine,
            limit=q_lim,
            traceLimit=trace_limit,
            materialized=True,
            tableName=t_name,
            tableLink=None,
            schema=[],
            settings=norm_settings,
            traceFilters=trace_filters,
            traceMetadataColumns=req_meta,
            traceOrderBy=trace_order_by,
            experimentFilter=experiment_filter,
            error=None,
            errorMessage=None,
            _columns=[],
            _schema_map={},
            _cancel=threading.Event(),
            _batch_cancel=batch_cancel,
            _active=set(),
            _claimed_rows=0,
            _stream_buf=_StreamBuffer(self.duckdb, t_name, q_uuid, q_sql),
        )
        self._remember(ex)
        executions.append(ex)
    for ex in executions:
      self._persist(self.duckdb.save_execution, ex)

    primary = executions[0]
    if engine == "duckdb":
      target = self._run_duckdb_batch
      args: Tuple[Any, ...] = (executions,)
    else:
      target = self._run_batch
      args = (executions, traces)

    if is_async:
      run = threading.Thread(target=target, args=args, daemon=True)
      with self._lock:
        self._runs = [r for r in self._runs if r.is_alive()] + [run]
      run.start()
      return {
          "queryUuid": primary["queryUuid"],
          "tableName": primary["tableName"],
          "executions": [{
              "queryUuid": e["queryUuid"],
              "tableName": e["tableName"],
          } for e in executions],
          "schema": [],
          "rows": [],
          "totalFilteredRows": 0,
      }
    target(*args)
    successful = next((e for e in executions if e["status"] == "SUCCESS"), None)
    if not successful and primary["status"] == "FAILED":
      raise QueryFailedError(primary["errorMessage"])
    active_primary = successful or primary
    sync_limit = (
        active_primary["limit"]
        if active_primary["limit"] > 0 else active_primary["processedRows"])
    res = self.fetch_results(
        active_primary["queryUuid"], offset=0, limit=sync_limit)
    res["executions"] = [{
        "queryUuid": e["queryUuid"],
        "tableName": e["tableName"],
    } for e in executions]
    return res

  def _run_duckdb_batch(self, executions: List[Dict[str, Any]]) -> None:
    for ex in executions:
      self._run_duckdb(ex)

  def _flush_active_stream_buffers(self) -> None:
    with self._lock:
      live = [
          e.get("_stream_buf")
          for e in self._executions.values()
          if e.get("queryEngine") == "traces" and e.get("_stream_buf")
      ]
    for buf in live:
      if buf and buf.columns:
        with self._lock:
          buf.flush()

  def _run_duckdb(self, ex: Dict[str, Any]) -> None:
    """Executes a DuckDB query and writes its results directly into a DuckDB table."""
    self._flush_active_stream_buffers()

    def on_cursor(cur: Any) -> None:
      with self._lock:
        if ex["_cancel"].is_set():
          raise QueryFailedError("cancelled")
        ex["_duck_cursor"] = cur

    try:
      columns, schema_map, row_count = self.duckdb.execute_into_table(
          ex["perfettoSql"],
          ex["tableName"],
          ex["limit"],
          query_uuid=ex["queryUuid"],
          on_cursor=on_cursor,
      )
      with self._lock:
        ex.pop("_duck_cursor", None)
        ex["_columns"] = columns
        ex["_schema_map"] = schema_map
        ex["schema"] = [{
            "name": c,
            "type": schema_map.get(c, "STRING")
        } for c in columns]
        ex["processedTraces"] = 1
        ex["processedRows"] = row_count
        if ex["status"] != "CANCELLED":
          ex["status"] = "SUCCESS"
    except Exception as e:  # pylint: disable=broad-except
      has_table, _ = self.duckdb.table_exists(ex.get("tableName") or "")
      with self._lock:
        ex.pop("_duck_cursor", None)
        if not has_table:
          ex["tableName"] = None
        if ex["status"] != "CANCELLED":
          ex["status"] = "FAILED"
          ex["errorMessage"] = f"DuckDB error: {e}"
    with self._lock:
      ex["endTime"] = iso_now()
      deleted = ex.get("_deleted")
    if deleted:
      self.duckdb.drop_table(ex.get("tableName"), ex["queryUuid"])
      return
    self._persist(self.duckdb.save_execution, ex)
    self._remember(ex)

  @staticmethod
  def _wants_rows(ex: Dict[str, Any]) -> bool:
    return not ex["_cancel"].is_set() and (ex["limit"] <= 0 or
                                           ex["_claimed_rows"] < ex["limit"])

  def _claim_rows(self, ex: Dict[str, Any], count: int) -> int:
    with self._lock:
      if ex["_cancel"].is_set():
        return 0
      if ex["limit"] <= 0:
        ex["_claimed_rows"] += count
        return count
      granted = max(0, min(count, ex["limit"] - ex["_claimed_rows"]))
      ex["_claimed_rows"] += granted
      return granted

  def _query_trace_batch(
      self,
      executions: List[Dict[str, Any]],
      trace: TraceEntry,
      pool_query: PoolQuery,
      batch_cancel: Optional[threading.Event] = None,
  ) -> List[Tuple[Optional[Tuple[List[str], List[Tuple[bytes, int]]]],
                  Optional[Exception]]]:
    """Loads `trace` once and runs all `executions` sequentially on that TP."""
    if not any(self._wants_rows(e) for e in executions):
      return [(None, None) for _ in executions]
    if not os.path.isfile(trace.file_path):
      err = FileNotFoundError("trace file not found")
      return [(None, err) for _ in executions]
    cancel_event = batch_cancel or executions[0]["_cancel"]
    try:
      tp = self.pool.acquire(trace.file_path, cancel_event, pool_query)
    except TraceProcessorException as e:
      if all(ex["_cancel"].is_set() for ex in executions):
        return [(None, None) for _ in executions]
      return [(None, e) for _ in executions]
    if not any(self._wants_rows(e) for e in executions):
      self.pool.release(trace.file_path, tp, discard=True)
      return [(None, None) for _ in executions]
    with self._lock:
      for ex in executions:
        if self._wants_rows(ex):
          ex["_active"].add(tp)
    healthy = False
    outcomes: List[Tuple[Optional[Tuple[List[str], List[Tuple[bytes, int]]]],
                         Optional[Exception]]] = []
    try:
      for ex in executions:
        if not self._wants_rows(ex):
          outcomes.append((None, None))
          continue
        try:
          buf: _StreamBuffer = ex["_stream_buf"]
          if self.duckdb.extension_loaded and ex["limit"] <= 0:
            need_sample = not ex["_columns"] or any(
                c not in ex["_schema_map"] for c in ex["_columns"])
            spool_path = buf.next_spool_path()
            cols, sample_bytes, total_bytes = query_result.query_to_spool_file(
                tp, ex["perfettoSql"], spool_path, need_sample=need_sample)
            healthy = True
            sample = ([(
                sample_bytes[0],
                min(
                    256,
                    query_result.row_count(
                        memoryview(sample_bytes[0]), len(cols))),
            )] if sample_bytes else [])
            outcomes.append(((cols, sample, spool_path if total_bytes else None,
                              1 if total_bytes else 0), None))
            continue

          result = query_result.query(tp, ex["perfettoSql"])
          healthy = True

          kept: List[Tuple[bytes, int]] = []
          for batch in result.batches:
            if not self._wants_rows(ex):
              break
            granted = self._claim_rows(
                ex, query_result.row_count(batch, len(result.columns)))
            if granted:
              kept.append((bytes(batch), granted))
          if self.duckdb.extension_loaded:
            spool_path, total_rows = buf.write_trace_spool(
                kept, len(result.columns))
            sample = kept[:1] if kept else []
            outcomes.append(
                ((result.columns, sample, spool_path, total_rows), None))
          else:
            outcomes.append(((result.columns, kept, None, 0), None))
        except TraceProcessorException as e:
          if ex["_cancel"].is_set():
            outcomes.append((None, None))
          else:
            healthy = True
            outcomes.append((None, e))
        except Exception as e:  # pylint: disable=broad-except
          if ex["_cancel"].is_set():
            outcomes.append((None, None))
          else:
            healthy = False
            outcomes.append((None, e))
            for _ in range(len(executions) - len(outcomes)):
              outcomes.append((None, e))
            break
      return outcomes
    finally:
      with self._lock:
        for ex in executions:
          ex["_active"].discard(tp)
        killed = getattr(tp, "subprocess", None) is None
      self.pool.release(trace.file_path, tp, discard=killed or not healthy)

  def _run_batch(self, executions: List[Dict[str, Any]],
                 traces: List[TraceEntry]) -> None:
    """Runs 1..N queries per trace in a single pass, streaming into DuckDB."""
    self.duckdb.sync_catalog(self.catalog)
    catalog_types = {c["name"]: c["type"] for c in self.catalog.get_schema()}
    catalog_types["_trace_uuid"] = "STRING"
    n = len(executions)
    sql_columns: List[Optional[List[str]]] = [None] * n
    meta_columns: List[List[str]] = [[] for _ in range(n)]
    errors: List[List[str]] = [[] for _ in range(n)]
    succeeded: List[int] = [0] * n
    combined_sql = "\n".join(e["perfettoSql"] for e in executions)
    pool_query = self.pool.start_query(combined_sql)
    all_trace_uuids = [t.trace_uuid for t in traces]
    with self._lock:
      batch_cancel = (executions[0].get("_batch_cancel")
                      if executions else None) or threading.Event()
      for ex in executions:
        ex["_batch_cancel"] = batch_cancel
        ex["_stream_buf"].all_uuids = all_trace_uuids

    def _bg_flush(fbuf: _StreamBuffer) -> None:
      fbuf.flush()
      for e in executions:
        if e.get("_stream_buf") is fbuf:
          real_rows = int((self.duckdb._table_meta.get(fbuf.table_name) or
                           {}).get("rowCount", 0))
          with self._lock:
            e["processedRows"] = real_rows

    def any_wants_more() -> bool:
      return any(not e["_cancel"].is_set() and
                 (e["limit"] <= 0 or e["processedRows"] < e["limit"])
                 for e in executions)

    flush_pool = ThreadPoolExecutor(max_workers=1)
    try:
      max_active = self.max_workers
      active_futures: Dict[Any, TraceEntry] = {}
      trace_iter = iter(traces)

      def submit_next() -> bool:
        if not any_wants_more():
          return False
        t = next(trace_iter, None)
        if t is None:
          return False
        active_futures[self._workers.submit(self._query_trace_batch, executions,
                                            t, pool_query, batch_cancel)] = t
        return True

      for _ in range(max_active):
        if not submit_next():
          break

      while active_futures:
        done, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)
        for future in done:
          trace = active_futures.pop(future)
          try:
            outcomes = future.result()
          except Exception as e:  # pylint: disable=broad-except
            outcomes = [(None, e) for _ in executions]

          to_flush: List[_StreamBuffer] = []
          for i, ex in enumerate(executions):
            res_tuple, err = outcomes[i]
            buf: _StreamBuffer = ex["_stream_buf"]
            added_rows = 0
            if err is not None:
              if not ex["_cancel"].is_set():
                logger.info("Query failed on %s: %s", trace.file_name, err)
                errors[i].append(f"{trace.file_name}: {err}")
            elif res_tuple is not None:
              columns, batches, spool_path, spool_rows = res_tuple
              if sql_columns[i] is None:
                sql_columns[i] = columns
                meta_order = ["_trace_uuid"] + [
                    c for c in catalog_types if c != "_trace_uuid"
                ]
                meta_columns[i] = [c for c in meta_order if c not in columns]
                with self._lock:
                  ex["_columns"] = sql_columns[i] + meta_columns[i]
                  ex["_schema_map"] = {
                      c: catalog_types[c] for c in meta_columns[i]
                  }
              if columns == sql_columns[i]:
                succeeded[i] += 1
                self._update_schema(ex, sql_columns[i], batches)
                if self.duckdb.extension_loaded:
                  added_rows = spool_rows
                  with self._lock:
                    buf.ensure_initialized(
                        ex["_columns"],
                        ex["_schema_map"],
                        sql_columns=sql_columns[i],
                        meta_columns=meta_columns[i],
                    )
                    if buf.add_spool(spool_path, trace.trace_uuid, spool_rows):
                      to_flush.append(buf)
                else:
                  with self._lock:
                    buf.ensure_initialized(ex["_columns"], ex["_schema_map"])
                    added_rows = buf.add_batches(
                        batches, len(columns),
                        [trace.get_value(c) for c in meta_columns[i]])
              else:
                if spool_path:
                  try:
                    os.remove(spool_path)
                  except OSError:
                    pass
                errors[i].append(
                    f"{trace.file_name}: returned different columns")

            with self._lock:
              ex["processedTraces"] += 1
              ex["processedRows"] += added_rows
              if ex["limit"] > 0 and ex["processedRows"] >= ex["limit"]:
                ex["_cancel"].set()

          submit_next()

          for fbuf in to_flush:
            flush_pool.submit(_bg_flush, fbuf)

          if not any_wants_more():
            batch_cancel.set()
            for ex in executions:
              ex["_cancel"].set()
            for f in active_futures:
              f.cancel()
            break

    finally:
      self.pool.end_query()
      flush_pool.shutdown(wait=True)

    for i, ex in enumerate(executions):
      buf = ex["_stream_buf"]
      try:
        with self._lock:
          status = ex["status"]
          if status != "CANCELLED":
            if errors[i]:
              ex["errorMessage"] = (f"{len(errors[i])} trace(s) failed: " +
                                    "; ".join(errors[i][:3]))
            status = "FAILED" if errors[i] and succeeded[i] == 0 else "SUCCESS"
          deleted = ex.get("_deleted")
          should_materialize = not deleted and (status == "SUCCESS" or
                                                (status == "CANCELLED" and
                                                 ex["processedRows"] > 0))
        if should_materialize and buf.columns:
          buf.flush()
          real_rows = int((self.duckdb._table_meta.get(buf.table_name) or
                           {}).get("rowCount", 0))
          self.duckdb.finalize_streaming_table(
              buf.table_name,
              query_uuid=ex["queryUuid"],
              sql=ex.get("perfettoSql"))
          with self._lock:
            ex["processedRows"] = real_rows
            ex["tableName"] = buf.table_name
        else:
          with self._lock:
            ex["tableName"] = None
      finally:
        buf.cleanup()
        with self._lock:
          ex.pop("_stream_buf", None)
          if ex["status"] != "CANCELLED":
            ex["status"] = status
          ex["endTime"] = iso_now()
      if deleted:
        self.duckdb.drop_table(ex.get("tableName"), ex["queryUuid"])
        continue
      self._persist(self.duckdb.save_execution, ex)
      self._remember(ex)

  def _update_schema(self, ex: Dict[str, Any], sql_columns: List[str],
                     batches: List[Tuple[bytes, int]]) -> None:
    """Types each SQL column by the first non-null value any trace returns."""
    schema_map = ex["_schema_map"]
    untyped = [i for i, c in enumerate(sql_columns) if c not in schema_map]
    types: Dict[str, str] = {}
    if untyped and batches:
      sample = query_result.decode_batch(batches[0][0], len(sql_columns))
      for i in untyped:
        value = next((r[i] for r in sample if r[i] is not None), None)
        if value is not None:
          types[sql_columns[i]] = infer_column_type([value])
    with self._lock:
      schema_map.update(types)
      visible = sql_columns + [
          c for c in ex["traceMetadataColumns"] if c not in sql_columns
      ]
      ex["schema"] = [{
          "name": c,
          "type": schema_map.get(c, "STRING")
      } for c in visible]

  def fetch_results(
      self,
      query_uuid: str,
      offset: int = 0,
      limit: int = 50,
      order_by: Optional[str] = None,
      filters: Optional[List[Dict[str, Any]]] = None,
      columns: Optional[List[str]] = None,
  ) -> Dict[str, Any]:
    """Streams a filtered, sorted and projected page directly from DuckDB."""
    ex = self._load(query_uuid)
    with self._lock:
      buf: Optional[_StreamBuffer] = ex.get("_stream_buf")
      if buf and buf.columns:
        buf.flush()
      table_name = ex.get("tableName") or ""
      all_columns = list(ex.get("_columns") or [])
      schema_map = dict(ex.get("_schema_map") or {})
      visible = [s["name"] for s in ex.get("schema") or []]
      if not all_columns and visible:
        all_columns = list(visible)
      for s in ex.get("schema") or []:
        schema_map.setdefault(s["name"], s.get("type", "STRING"))

    offset, limit = max(0, offset), max(0, limit)
    out_indices = _column_indices(all_columns, columns or visible or
                                  all_columns)
    out_columns = [all_columns[i] for i in out_indices]

    total, page = self.duckdb.query_page(
        table_name=table_name,
        all_columns=all_columns,
        out_indices=out_indices,
        offset=offset,
        limit=limit,
        order_by=order_by,
        filters=filters,
    )

    return {
        "queryUuid": query_uuid,
        "tableName": table_name,
        "schema": [{
            "name": c,
            "type": schema_map.get(c, "STRING")
        } for c in out_columns],
        "rows": [{
            "values": [to_wire(v) for v in r]
        } for r in page],
        "totalFilteredRows": total,
        "availableColumnNames": all_columns,
    }


def _column_indices(all_columns: List[str], wanted: List[str]) -> List[int]:
  used: set = set()
  out = []
  for name in wanted:
    idx = next(
        (i for i, c in enumerate(all_columns) if c == name and i not in used),
        None)
    if idx is not None:
      used.add(idx)
      out.append(idx)
  return out
