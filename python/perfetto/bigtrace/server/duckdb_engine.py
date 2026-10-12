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
"""DuckDB engine for storing trace query tables, execution history, and paging."""

import datetime
import decimal
import glob
import json
import logging
import os
import re
import subprocess
import sys
import threading
from typing import (Any, Callable, Collection, Dict, Iterator, List, Optional,
                    Sequence, Tuple)

from perfetto.bigtrace.server import query_result
from perfetto.bigtrace.server.catalog import TraceCatalog
from perfetto.bigtrace.server.filters import is_null, parse_order_by
from perfetto.bigtrace.server.util import iso_now

logger = logging.getLogger(__name__)

_EXT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "duckdb_ext"))
_EXT_PATH = os.path.join(_EXT_DIR, "perfetto.duckdb_extension")
_BATCH_ROWS = 2048

TERMINAL_STATUSES = ("SUCCESS", "FAILED", "CANCELLED")

STATUS_FIELDS = (
    "queryUuid",
    "status",
    "startTime",
    "endTime",
    "processedRows",
    "processedTraces",
    "totalTraces",
    "error",
    "errorMessage",
)

PUBLIC_FIELDS = STATUS_FIELDS + (
    "perfettoSql",
    "queryEngine",
    "limit",
    "traceLimit",
    "materialized",
    "tableName",
    "tableLink",
    "schema",
    "settings",
    "traceFilters",
    "traceMetadataColumns",
    "traceOrderBy",
    "experimentFilter",
)

_RUNTIME_KEYS = ("_cancel", "_batch_cancel", "_active", "_claimed_rows",
                 "_duck_cursor", "_deleted", "_stream_buf")

_BIGTRACE_TO_DUCKDB_TYPE = {
    "INT64": "BIGINT",
    "DOUBLE": "DOUBLE",
    "FLOAT64": "DOUBLE",
    "BOOLEAN": "BOOLEAN",
    "TIMESTAMP": "VARCHAR",
    "STRING": "VARCHAR",
}


def public_view(ex: Dict[str, Any],
                fields: Tuple[str, ...] = PUBLIC_FIELDS) -> Dict[str, Any]:
  """Returns the wire form of an execution, omitting unset fields."""
  return {k: ex[k] for k in fields if ex.get(k) not in (None, [], {})}


def _dedup_column_names(columns: Sequence[str]) -> List[str]:
  seen: Dict[str, int] = {}
  unique_cols: List[str] = []
  for c in columns:
    base = c or "col"
    if base in seen:
      seen[base] += 1
      unique_cols.append(f"{base}_{seen[base]}")
    else:
      seen[base] = 0
      unique_cols.append(base)
  return unique_cols


def _coerce_column(col: Sequence[Any], btype: str) -> List[Any]:
  if btype == "INT64":
    out: List[Any] = []
    for val in col:
      if val is None:
        out.append(None)
      else:
        try:
          out.append(int(val))
        except (ValueError, TypeError):
          out.append(None)
    return out
  if btype in ("DOUBLE", "FLOAT64"):
    out = []
    for val in col:
      if val is None:
        out.append(None)
      else:
        try:
          out.append(float(val))
        except (ValueError, TypeError):
          out.append(None)
    return out
  if btype == "BOOLEAN":
    return [
        None if val is None else
        (val.lower() in ("true", "1") if isinstance(val, str) else bool(val))
        for val in col
    ]
  return [None if val is None else str(val) for val in col]


def sanitize_table_name(name: str) -> str:
  cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name.strip())
  if not cleaned:
    return ""
  if cleaned[0].isdigit():
    cleaned = "t_" + cleaned
  return cleaned


def _quote_ident(ident: str) -> str:
  return '"' + ident.replace('"', '""') + '"'


def _quote_str(val: str) -> str:
  return "'" + val.replace("'", "''") + "'"


def _load_duckdb_module():
  if hasattr(os, "RTLD_GLOBAL") and hasattr(os, "RTLD_NOW"):
    sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
  import duckdb  # pylint: disable=import-outside-toplevel
  return duckdb


def _extension_is_stale() -> bool:
  if not os.path.isfile(_EXT_PATH):
    return True
  built = os.path.getmtime(_EXT_PATH)
  sources = [os.path.join(_EXT_DIR, "build.sh")]
  for pattern in ("src/*.cc", "src/*.h", "sql/*.sql"):
    sources += glob.glob(os.path.join(_EXT_DIR, pattern))
  return any(os.path.getmtime(s) > built for s in sources if os.path.isfile(s))


def _ensure_extension_built() -> Optional[str]:
  """Returns the extension path, (re)building it when missing or stale."""
  build_script = os.path.join(_EXT_DIR, "build.sh")
  if not _extension_is_stale() or not os.path.isfile(build_script):
    return _EXT_PATH if os.path.isfile(_EXT_PATH) else None
  try:
    subprocess.run([build_script],
                   cwd=_EXT_DIR,
                   check=True,
                   capture_output=True,
                   text=True,
                   timeout=300)
  except (OSError, subprocess.SubprocessError) as e:
    logger.warning("Could not build perfetto.duckdb_extension: %s", e)
  return _EXT_PATH if os.path.isfile(_EXT_PATH) else None


_PF_INTERVAL_INTERSECTION_FULL_ROW_MACRO = """
CREATE OR REPLACE MACRO pf_interval_intersection(
    rels, aliases, per := [], full_row := true) AS TABLE
FROM query(
  CASE WHEN NOT full_row THEN
    'SELECT f.ts AS ts, f.dur AS dur, ' || pf_interval__keys(per, 'k.')
    || array_to_string(list_transform(aliases, (a, i) ->
         '{''id'': ' || CASE WHEN len(rels) IN (2, 3) THEN 'f.r' || (i - 1) ELSE 'f.rows[' || i || ']' END || '} AS ' || a), ', ')
    || ' FROM (SELECT ' || pf_interval__keys(per, '') || 'unnest(__pf_a) AS f FROM (SELECT '
    || pf_interval__keys(per, '')
    || CASE
         WHEN len(rels) = 2 THEN 'pf_interval_intersect2_agg(side, row, ts, dur)'
         WHEN len(rels) = 3 THEN 'pf_interval_intersect3_agg(side, row, ts, dur)'
         ELSE 'pf_interval_intersect_agg(side, row, ts, dur, ' || len(rels) || ')'
       END
    || ' AS __pf_a FROM ('
    || array_to_string(list_transform(rels, (r, i) ->
         'SELECT ' || pf_interval__keys(per, '') || (i - 1) || '::INTEGER AS side, '
         || 'id::BIGINT AS row, ts::BIGINT AS ts, dur::BIGINT AS dur FROM ' || r || ' __pf_t' || i), ' UNION ALL ')
    || ')' || pf_interval__groupby(per) || ')) k'
  ELSE
    'WITH ' || array_to_string(list_transform(rels, (r, i) -> 'o' || i || ' AS MATERIALIZED ' || pf_interval__num(r)), ', ')
    || ', k AS (SELECT ' || pf_interval__keys(per, '') || 'unnest(__pf_a) AS f FROM (SELECT '
    || pf_interval__keys(per, '')
    || CASE
         WHEN len(rels) = 2 THEN 'pf_interval_intersect2_agg(side, row, ts, dur)'
         WHEN len(rels) = 3 THEN 'pf_interval_intersect3_agg(side, row, ts, dur)'
         ELSE 'pf_interval_intersect_agg(side, row, ts, dur, ' || len(rels) || ')'
       END
    || ' AS __pf_a FROM ('
    || array_to_string(list_transform(rels, (r, i) -> pf_interval__feed(per, i - 1, 'o' || i)), ' UNION ALL ')
    || ')' || pf_interval__groupby(per) || '))'
    || ' SELECT f.ts AS ts, f.dur AS dur, ' || pf_interval__keys(per, 'k.')
    || array_to_string(list_transform(aliases, (a, i) -> 'o' || i || '.__pf_s AS ' || a), ', ')
    || ' FROM k ' || array_to_string(list_transform(rels, (r, i) ->
         'JOIN o' || i || ' ON o' || i || '.__pf_r = '
         || CASE WHEN len(rels) IN (2, 3) THEN 'f.r' || (i - 1) ELSE 'f.rows[' || i || ']' END), ' ')
  END
)
"""


class DuckDBEngine:
  """Manages the embedded DuckDB instance, query tables, history, and paging."""

  def __init__(self,
               db_path: Optional[str] = None,
               parquet_dir: Optional[str] = None):
    self.db_path = db_path or ":memory:"
    if self.db_path != ":memory:":
      self.db_path = os.path.abspath(os.path.expanduser(self.db_path))
      os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    self.parquet_dir = (
        os.path.abspath(os.path.expanduser(parquet_dir))
        if parquet_dir else None)
    if self.parquet_dir:
      os.makedirs(self.parquet_dir, exist_ok=True)
    self._lock = threading.RLock()
    self._counter = 0
    self._table_meta: Dict[str, Dict[str, Any]] = {}
    self._latest_table: Optional[str] = None
    self._duckdb = _load_duckdb_module()
    self._conn = self._duckdb.connect(
        self.db_path, config={"allow_unsigned_extensions": "true"})
    self._conn.execute("SET preserve_insertion_order = false")
    ext_path = _ensure_extension_built()
    self.extension_loaded = False
    if ext_path:
      try:
        self._conn.execute(f"LOAD {_quote_str(ext_path)}")
        self._conn.execute(_PF_INTERVAL_INTERSECTION_FULL_ROW_MACRO)
        self.extension_loaded = True
      except Exception as e:  # pylint: disable=broad-except
        logger.warning("Failed to load perfetto.duckdb_extension: %s", e)
    self._init_history_schema()
    if self.parquet_dir:
      self._load_existing_parquet()

  def _init_history_schema(self) -> None:
    with self._lock:
      self._conn.execute("""
          CREATE TABLE IF NOT EXISTS _bigtrace_executions (
            query_uuid VARCHAR PRIMARY KEY,
            status VARCHAR NOT NULL,
            start_time VARCHAR,
            data VARCHAR NOT NULL
          )""")
      stale = self._conn.execute(
          "SELECT data FROM _bigtrace_executions WHERE status = 'IN_PROGRESS'"
      ).fetchall()
      for (data,) in stale:
        ex = json.loads(data)
        ex.update(
            status="CANCELLED",
            endTime=iso_now(),
            errorMessage="Interrupted by server restart")
        self.save_execution(ex)
      for row in self.list_executions():
        tname = row.get("tableName")
        uid = row.get("queryUuid")
        if tname and not tname.startswith("_bigtrace"):
          m = re.fullmatch(r"result_(\d+)", tname)
          if m:
            self._counter = max(self._counter, int(m.group(1)))
          if self._physical_table_exists_locked(tname):
            self._record_table_meta_locked(
                tname, query_uuid=uid, sql=row.get("perfettoSql"))

  def _physical_table_exists_locked(self, table_name: str) -> bool:
    row = self._conn.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'main' AND table_name = ?",
        [table_name]).fetchone()
    return row is not None

  def _load_existing_parquet(self) -> None:
    if not self.parquet_dir or not os.path.isdir(self.parquet_dir):
      return
    with self._lock:
      for fname in sorted(os.listdir(self.parquet_dir)):
        if not fname.endswith(".parquet") or fname.startswith("."):
          continue
        tname = sanitize_table_name(fname[:-len(".parquet")])
        if not tname or tname.startswith("_bigtrace"):
          continue
        fpath = os.path.join(self.parquet_dir, fname)
        try:
          self._conn.execute(
              f"CREATE OR REPLACE TABLE {_quote_ident(tname)} "
              f"AS SELECT * FROM read_parquet({_quote_str(fpath)})")
          self._record_table_meta_locked(tname)
          m = re.fullmatch(r"result_(\d+)", tname)
          if m:
            self._counter = max(self._counter, int(m.group(1)))
        except Exception as e:  # pylint: disable=broad-except
          logger.warning("Could not load parquet %s: %s", fpath, e)

  # Execution metadata persistence in DuckDB --------------------------------

  def save_execution(self, ex: Dict[str, Any]) -> None:
    doc = {k: v for k, v in ex.items() if k not in _RUNTIME_KEYS}
    with self._lock:
      self._conn.execute(
          "INSERT OR REPLACE INTO _bigtrace_executions VALUES (?, ?, ?, ?)", [
              doc["queryUuid"],
              doc.get("status", "UNKNOWN"),
              doc.get("startTime"),
              json.dumps(doc, default=str),
          ])

  def get_execution(self, query_uuid: str) -> Optional[Dict[str, Any]]:
    with self._lock:
      row = self._conn.execute(
          "SELECT data FROM _bigtrace_executions WHERE query_uuid = ?",
          [query_uuid]).fetchone()
    return json.loads(row[0]) if row else None

  def list_executions(self) -> List[Dict[str, Any]]:
    with self._lock:
      rows = self._conn.execute("SELECT data FROM _bigtrace_executions "
                                "ORDER BY start_time DESC").fetchall()
    return [public_view(json.loads(data)) for (data,) in rows]

  def get_status(self, query_uuid: str) -> Optional[Dict[str, Any]]:
    ex = self.get_execution(query_uuid)
    return public_view(ex, STATUS_FIELDS) if ex else None

  def mark_cancelled(self, query_uuid: str) -> None:
    ex = self.get_execution(query_uuid)
    if ex and ex.get("status") == "IN_PROGRESS":
      ex.update(status="CANCELLED", endTime=iso_now())
      self.save_execution(ex)

  def delete_execution_record(self, query_uuid: str) -> bool:
    with self._lock:
      exists = self._conn.execute(
          "SELECT 1 FROM _bigtrace_executions WHERE query_uuid = ?",
          [query_uuid]).fetchone()
      if exists:
        self._conn.execute(
            "DELETE FROM _bigtrace_executions WHERE query_uuid = ?",
            [query_uuid])
      return exists is not None

  # Catalog & Table naming --------------------------------------------------

  def sync_catalog(self, catalog: TraceCatalog) -> None:
    """Materializes `trace_metadata` with `_trace_uuid` and catalog columns."""
    traces = catalog.get_traces()
    schema = catalog.get_schema()
    col_names = ["_trace_uuid"
                ] + [c["name"] for c in schema if c["name"] != "_trace_uuid"]
    col_types = {c["name"]: c["type"] for c in schema}
    col_types["_trace_uuid"] = "STRING"
    col_data: List[List[Any]] = [[t.trace_uuid for t in traces]]
    for c in col_names[1:]:
      col_data.append([t.get_value(c) for t in traces])
    with self._lock:
      self.create_streaming_table("trace_metadata", col_names, col_types)
      self.append_columns("trace_metadata", col_names, col_types, col_data)
      self.finalize_streaming_table(
          "trace_metadata", sql="-- Trace catalog metadata", export=False)

  def next_table_name(self,
                      preferred: Optional[str] = None,
                      reserved: Optional[Collection[str]] = None) -> str:
    with self._lock:
      reserved_set = set(reserved or ())
      if preferred:
        cleaned = sanitize_table_name(preferred)
        if cleaned:
          candidate = cleaned
          idx = 2
          while candidate in reserved_set:
            candidate = f"{cleaned}_{idx}"
            idx += 1
          return candidate
      while True:
        self._counter += 1
        candidate = f"result_{self._counter}"
        if (candidate not in self._table_meta and
            candidate not in reserved_set and
            not self._physical_table_exists_locked(candidate)):
          return candidate

  def suggest_table_name(self,
                         base_name: str,
                         reserved: Optional[Collection[str]] = None) -> str:
    cleaned = sanitize_table_name(base_name) or "tbl"
    with self._lock:
      reserved_set = set(reserved or ())
      idx = 1
      while True:
        candidate = f"{cleaned}_{idx}"
        if (candidate not in self._table_meta and
            candidate not in reserved_set and
            not self._physical_table_exists_locked(candidate)):
          return candidate
        idx += 1

  def table_exists(self, table_name: str) -> Tuple[bool, str]:
    resolved = sanitize_table_name(table_name)
    if not resolved or resolved.startswith("_bigtrace"):
      return False, ""
    with self._lock:
      if resolved in self._table_meta or self._physical_table_exists_locked(
          resolved):
        return True, resolved
      return False, resolved

  # Direct streaming into DuckDB tables -------------------------------------

  def create_streaming_table(
      self,
      table_name: str,
      columns: Sequence[str],
      schema_map: Dict[str, str],
      query_uuid: Optional[str] = None,
      sql: Optional[str] = None) -> Tuple[str, List[str]]:
    """Creates or replaces an empty DuckDB table ready for streaming appends."""
    resolved = sanitize_table_name(table_name) or self.next_table_name()
    col_names = list(columns) or ["_trace_uuid"]
    unique_cols = _dedup_column_names(col_names)
    col_defs = []
    col_meta = []
    for orig, uniq in zip(col_names, unique_cols):
      btype = schema_map.get(orig, "STRING")
      dtype = _BIGTRACE_TO_DUCKDB_TYPE.get(btype, "VARCHAR")
      col_defs.append(f"{_quote_ident(uniq)} {dtype}")
      col_meta.append({"name": uniq, "type": _duckdb_type_to_bigtrace(dtype)})

    qtable = _quote_ident(resolved)
    with self._lock:
      m = re.fullmatch(r"result_(\d+)", resolved)
      if m:
        self._counter = max(self._counter, int(m.group(1)))
      try:
        self._conn.execute(f"DROP VIEW IF EXISTS {qtable}")
      except Exception:  # pylint: disable=broad-except
        pass
      self._conn.execute(
          f"CREATE OR REPLACE TABLE {qtable} ({', '.join(col_defs)})")
      prev = self._table_meta.get(resolved, {})
      self._table_meta[resolved] = {
          "name": resolved,
          "rowCount": 0,
          "columns": col_meta,
          "queryUuid": query_uuid or prev.get("queryUuid"),
          "sql": sql or prev.get("sql"),
      }
      self._latest_table = resolved
    return resolved, unique_cols

  def append_columns(self, table_name: str, columns: Sequence[str],
                     schema_map: Dict[str, str],
                     col_data: Sequence[Sequence[Any]]) -> int:
    """Appends a chunk of per-column arrays directly into `table_name`."""
    if not col_data or not col_data[0]:
      return 0
    row_count = len(col_data[0])
    resolved = sanitize_table_name(table_name)
    unique_cols = _dedup_column_names(columns)
    types = [schema_map.get(c, "STRING") for c in columns]
    qtable = _quote_ident(resolved)

    arrow_tbl = None
    try:
      import pyarrow as pa  # pylint: disable=import-outside-toplevel
      pa_types = {
          "INT64": pa.int64(),
          "DOUBLE": pa.float64(),
          "FLOAT64": pa.float64(),
          "BOOLEAN": pa.bool_(),
      }
      arrays = []
      for col, btype in zip(col_data, types):
        target_pa = pa_types.get(btype, pa.string())
        try:
          arrays.append(pa.array(col, type=target_pa))
        except Exception:  # pylint: disable=broad-except
          arrays.append(pa.array(_coerce_column(col, btype), type=target_pa))
      arrow_tbl = pa.Table.from_arrays(arrays, names=unique_cols)
    except Exception:  # pylint: disable=broad-except
      arrow_tbl = None

    with self._lock:
      inserted = False
      if arrow_tbl is not None:
        try:
          self._conn.register("_bt_staging_arrow", arrow_tbl)
          try:
            self._conn.execute(
                f"INSERT INTO {qtable} SELECT * FROM _bt_staging_arrow")
            inserted = True
          finally:
            self._conn.unregister("_bt_staging_arrow")
        except Exception:  # pylint: disable=broad-except
          inserted = False
      if not inserted:
        coerced_cols = [
            _coerce_column(col, btype) for col, btype in zip(col_data, types)
        ]
        placeholders = ", ".join(["?"] * len(unique_cols))
        coerced_rows = list(zip(*coerced_cols)) if coerced_cols else []
        self._conn.executemany(f"INSERT INTO {qtable} VALUES ({placeholders})",
                               coerced_rows)
      meta = self._table_meta.get(resolved)
      if meta is not None:
        meta["rowCount"] = int(meta.get("rowCount", 0)) + row_count
    return row_count

  def ingest_proto_spools(
      self,
      table_name: str,
      sql_columns: Sequence[str],
      meta_columns: Sequence[str],
      schema_map: Dict[str, str],
      spools: Sequence[Tuple[str, str]],
      row_count: int,
      all_uuids: Optional[Sequence[str]] = None,
  ) -> int:
    """Ingests `.pb` trace spool files in parallel C++ via `pf_scan_tp_proto`."""
    if not spools:
      return 0
    resolved = sanitize_table_name(table_name)
    qtable = _quote_ident(resolved)
    sql_uniq = _dedup_column_names(sql_columns)
    sql_types = [schema_map.get(c, "STRING") for c in sql_columns]
    paths = [p for p, _ in spools]
    uuids = [u for _, u in spools]

    proj_exprs = [f"p.{_quote_ident(u)}" for u in sql_uniq]
    extra_meta = [mc for mc in meta_columns if mc != "_trace_uuid"]
    meta_map: Dict[str, Tuple[Any, ...]] = {}
    if extra_meta:
      qcols = ", ".join(['"_trace_uuid"'] +
                        [_quote_ident(mc) for mc in extra_meta])
      with self._lock:
        rows = self._conn.execute(
            f'SELECT {qcols} FROM "trace_metadata"').fetchall()
      meta_map = {str(r[0]): r[1:] for r in rows}

    enum_defs: List[Tuple[str, List[str]]] = []
    params: List[Any] = []
    for mi, mc in enumerate(meta_columns):
      qen = _quote_ident(f"_bt_meta_{resolved}_{mi}")
      if mc == "_trace_uuid":
        domain = list(
            dict.fromkeys(
                list(all_uuids or ()) + list(meta_map.keys()) + uuids))
        params.append(uuids)
        if 0 < len(domain) <= 65535:
          enum_defs.append((qen, domain))
          proj_exprs.append(
              f'array_extract(?::{qen}[], p."_trace_ord" + 1) AS "_trace_uuid"')
        else:
          proj_exprs.append(
              'array_extract(?::VARCHAR[], p."_trace_ord" + 1) AS "_trace_uuid"'
          )
      else:
        idx = extra_meta.index(mc)
        col_vals = [
            None if u not in meta_map or meta_map[u][idx] is None else str(
                meta_map[u][idx]) for u in uuids
        ]
        params.append(col_vals)
        dtype = _BIGTRACE_TO_DUCKDB_TYPE.get(
            schema_map.get(mc, "STRING"), "VARCHAR")
        if dtype == "VARCHAR":
          domain = list(
              dict.fromkeys([
                  str(v[idx]) for v in meta_map.values() if v[idx] is not None
              ] + [v for v in col_vals if v is not None]))
          if 0 < len(domain) <= 65535:
            enum_defs.append((qen, domain))
            proj_exprs.append(
                f'array_extract(?::{qen}[], p."_trace_ord" + 1) AS {_quote_ident(mc)}'
            )
          else:
            proj_exprs.append(
                f'array_extract(?::VARCHAR[], p."_trace_ord" + 1) AS {_quote_ident(mc)}'
            )
        else:
          proj_exprs.append(
              f'CAST(array_extract(?::VARCHAR[], p."_trace_ord" + 1) AS {dtype}) AS {_quote_ident(mc)}'
          )
    proj_sql = ", ".join(proj_exprs) if proj_exprs else 'p."_trace_ord"'
    params.extend([paths, sql_uniq, sql_types])

    select_sql = f"SELECT {proj_sql} FROM pf_scan_tp_proto(?, ?, ?, true) p"

    with self._lock:
      meta = self._table_meta.get(resolved)
      existing_rows = int(meta.get("rowCount", 0)) if meta is not None else 0
      if existing_rows == 0:
        self._conn.execute(f"DROP TABLE IF EXISTS {qtable}")
        for qen, domain in enum_defs:
          self._conn.execute(f"DROP TYPE IF EXISTS {qen}")
          vals_sql = ", ".join(_quote_str(u) for u in domain)
          self._conn.execute(f"CREATE TYPE {qen} AS ENUM ({vals_sql})")
        self._conn.execute(f"CREATE TABLE {qtable} AS {select_sql}", params)
      else:
        self._conn.execute(f"INSERT INTO {qtable} {select_sql}", params)
      actual_total = int(
          self._conn.execute(f"SELECT count(*) FROM {qtable}").fetchone()[0])
      if meta is not None:
        meta["rowCount"] = actual_total
    for p in paths:
      try:
        os.remove(p)
      except OSError:
        pass
    return row_count

  def finalize_streaming_table(self,
                               table_name: str,
                               query_uuid: Optional[str] = None,
                               sql: Optional[str] = None,
                               export: bool = True) -> None:
    resolved = sanitize_table_name(table_name)
    if not resolved:
      return
    with self._lock:
      if not self._physical_table_exists_locked(resolved):
        return
      if self._latest_table == resolved and resolved != "trace_metadata":
        self._conn.execute(
            f"CREATE OR REPLACE VIEW latest AS SELECT * FROM {_quote_ident(resolved)}"
        )
      if query_uuid:
        alias = f"bigtrace_{query_uuid.replace('-', '_')}"
        if alias != resolved:
          self._conn.execute(f"CREATE OR REPLACE VIEW {_quote_ident(alias)} "
                             f"AS SELECT * FROM {_quote_ident(resolved)}")
      if export and self.parquet_dir:
        pq_path = os.path.join(self.parquet_dir, f"{resolved}.parquet")
        tmp_path = os.path.join(self.parquet_dir, f".{resolved}.parquet.tmp")
        try:
          self._conn.execute(
              f"COPY {_quote_ident(resolved)} TO {_quote_str(tmp_path)} "
              "(FORMAT PARQUET, COMPRESSION ZSTD)")
          os.replace(tmp_path, pq_path)
        except Exception as e:  # pylint: disable=broad-except
          logger.warning("Parquet export for %s failed: %s", resolved, e)
          try:
            os.remove(tmp_path)
          except OSError:
            pass
      self._record_table_meta_locked(resolved, query_uuid=query_uuid, sql=sql)

  def _record_table_meta_locked(self,
                                table_name: str,
                                query_uuid: Optional[str] = None,
                                sql: Optional[str] = None) -> None:
    qtable = _quote_ident(table_name)
    count = self._conn.execute(f"SELECT count(*) FROM {qtable}").fetchone()[0]
    desc = self._conn.execute(f"DESCRIBE {qtable}").fetchall()
    cols = [{
        "name": r[0],
        "type": _duckdb_type_to_bigtrace(str(r[1]))
    } for r in desc]
    prev = self._table_meta.get(table_name, {})
    self._table_meta[table_name] = {
        "name": table_name,
        "rowCount": int(count),
        "columns": cols,
        "queryUuid": query_uuid or prev.get("queryUuid"),
        "sql": sql or prev.get("sql"),
    }

  def drop_table(self,
                 table_name: Optional[str],
                 query_uuid: Optional[str] = None) -> None:
    if not table_name:
      return
    resolved = sanitize_table_name(table_name)
    with self._lock:
      owner = self._table_meta.get(resolved, {}).get("queryUuid")
      if query_uuid and owner and owner != query_uuid:
        resolved = ""
      if resolved:
        try:
          self._conn.execute(f"DROP TABLE IF EXISTS {_quote_ident(resolved)}")
        except Exception:  # pylint: disable=broad-except
          pass
        self._table_meta.pop(resolved, None)
      if query_uuid:
        alias = f"bigtrace_{query_uuid.replace('-', '_')}"
        try:
          self._conn.execute(f"DROP VIEW IF EXISTS {_quote_ident(alias)}")
        except Exception:  # pylint: disable=broad-except
          pass
      if self.parquet_dir and resolved:
        pq_path = os.path.join(self.parquet_dir, f"{resolved}.parquet")
        try:
          os.remove(pq_path)
        except OSError:
          pass

  def list_tables(self) -> List[Dict[str, Any]]:
    with self._lock:
      return [
          dict(meta)
          for name, meta in self._table_meta.items()
          if not name.startswith("_bigtrace")
      ]

  def execute_into_table(
      self,
      sql: str,
      table_name: str,
      limit: int,
      query_uuid: Optional[str] = None,
      on_cursor: Optional[Callable[[Any], None]] = None,
  ) -> Tuple[List[str], Dict[str, str], int]:
    """Runs DuckDB SQL natively in C++ and materializes directly into `table_name`."""
    resolved = sanitize_table_name(table_name) or self.next_table_name()
    qtable = _quote_ident(resolved)
    with self._lock:
      cur = self._conn.cursor()
    try:
      if on_cursor:
        on_cursor(cur)
      cur.execute(sql)
      if not cur.description:
        return [], {}, 0
      raw_columns = [d[0] for d in cur.description]
      unique_cols = _dedup_column_names(raw_columns)
      schema_map = {
          u: _duckdb_type_to_bigtrace(str(d[1]))
          for u, d in zip(unique_cols, cur.description)
      }
      select_exprs = []
      for orig, uniq, desc in zip(raw_columns, unique_cols, cur.description):
        dtype = str(desc[1]).upper()
        qorig = _quote_ident(orig)
        quniq = _quote_ident(uniq)
        if any(t in dtype for t in ("TIMESTAMP", "DATE", "TIME", "INTERVAL")):
          select_exprs.append(f"CAST({qorig} AS VARCHAR) AS {quniq}")
        elif orig != uniq:
          select_exprs.append(f"{qorig} AS {quniq}")
        else:
          select_exprs.append(qorig)
      select_cols = ", ".join(select_exprs)
      limit_sql = f" LIMIT {int(limit)}" if limit > 0 else ""
      stream_name = f"_bt_stream_{os.getpid()}_{threading.get_ident()}"
      qstream = _quote_ident(stream_name)
      with self._lock:
        m = re.fullmatch(r"result_(\d+)", resolved)
        if m:
          self._counter = max(self._counter, int(m.group(1)))
        try:
          self._conn.execute(f"DROP VIEW IF EXISTS {qtable}")
        except Exception:  # pylint: disable=broad-except
          pass
        write_cur = self._conn.cursor()
      # Stream on `cur` -> `write_cur` outside `self._lock` so status polling
      # (`/query_executions:status` & `get_execution`) never blocks!
      reader = cur.to_arrow_reader(65536)
      write_cur.register(stream_name, reader)
      try:
        write_cur.execute(f"CREATE OR REPLACE TABLE {qtable} AS "
                          f"SELECT {select_cols} FROM {qstream}{limit_sql}")
      finally:
        try:
          write_cur.unregister(stream_name)
        finally:
          write_cur.close()
      with self._lock:
        self._latest_table = resolved
      self.finalize_streaming_table(resolved, query_uuid=query_uuid, sql=sql)
      total_rows = int(self._table_meta.get(resolved, {}).get("rowCount", 0))
    finally:
      cur.close()
    return unique_cols, schema_map, total_rows

  def query_page(
      self,
      table_name: str,
      all_columns: Sequence[str],
      out_indices: Sequence[int],
      offset: int = 0,
      limit: int = 50,
      order_by: Optional[str] = None,
      filters: Optional[List[Dict[str, Any]]] = None,
  ) -> Tuple[int, List[List[Any]]]:
    """Streams a filtered, sorted, projected page directly from DuckDB."""
    resolved = sanitize_table_name(table_name)
    unique_cols = _dedup_column_names(all_columns)
    if not resolved or not unique_cols:
      return 0, []
    with self._lock:
      if not self._physical_table_exists_locked(resolved):
        return 0, []
      cur = self._conn.cursor()
    try:
      qtable = _quote_ident(resolved)
      col_Set = set(unique_cols)
      # Map original column names (or deduped names) for filters/sorting
      orig_to_uniq: Dict[str, str] = {}
      for orig, uniq in zip(all_columns, unique_cols):
        orig_to_uniq.setdefault(orig, uniq)
        orig_to_uniq[uniq] = uniq

      where_parts: List[str] = []
      params: List[Any] = []
      for f in filters or ():
        field = f.get("field")
        if not field:
          continue
        op = str(f.get("op", "=")).lower()
        target = f.get("value")
        col_ident = (
            _quote_ident(orig_to_uniq[field])
            if field in orig_to_uniq else "NULL")
        cond, cond_params = _build_filter_sql(col_ident, op, target)
        where_parts.append(cond)
        params.extend(cond_params)

      where_sql = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
      total = cur.execute(f"SELECT count(*) FROM {qtable}{where_sql}",
                          params).fetchone()[0]
      if limit <= 0 or offset >= total:
        return int(total), []

      order_parts: List[str] = []
      for field, desc in parse_order_by(order_by):
        uniq = orig_to_uniq.get(field)
        if not uniq or uniq not in col_Set:
          continue
        c = _quote_ident(uniq)
        direction = "DESC" if desc else "ASC"
        is_empty = f"({c} IS NULL OR CAST({c} AS VARCHAR) = '' OR UPPER(CAST({c} AS VARCHAR)) = 'NULL')"
        num_val = f"TRY_CAST({c} AS DOUBLE)"
        if desc:
          # Descending reverses the whole numbers-then-strings-then-nulls order.
          bucket = f"CASE WHEN {is_empty} THEN 0 WHEN {num_val} IS NOT NULL THEN 2 ELSE 1 END"
        else:
          bucket = f"CASE WHEN {is_empty} THEN 2 WHEN {num_val} IS NOT NULL THEN 0 ELSE 1 END"
        order_parts.append(f"{bucket} ASC")
        order_parts.append(f"{num_val} {direction}")
        order_parts.append(f"LOWER(CAST({c} AS VARCHAR)) {direction}")
      if order_parts:
        order_parts.append("rowid ASC")
        order_sql = f" ORDER BY {', '.join(order_parts)}"
      else:
        order_sql = ""

      proj = ", ".join(f"{_quote_ident(unique_cols[i])} AS c_{k}"
                       for k, i in enumerate(out_indices))
      if not proj:
        proj = "1"
      sql = (f"SELECT {proj} FROM {qtable}{where_sql}{order_sql} "
             f"LIMIT {int(limit)} OFFSET {int(offset)}")
      rows = cur.execute(sql, params).fetchall()
      return int(total), [
          [_to_cell(v) for v in r] if out_indices else [] for r in rows
      ]
    finally:
      cur.close()

  def close(self) -> None:
    with self._lock:
      try:
        self._conn.close()
      except Exception:  # pylint: disable=broad-except
        pass


_SUPPORTED_FILTER_OPS = {
    "=", "!=", "<", "<=", ">", ">=", "glob", "not glob", "in", "not in",
    "is null", "is not null"
}


def _sql_equals_expr(col_ident: str, target: Any) -> Tuple[str, List[Any]]:
  s_target = "" if target is None else str(target)
  try:
    f_target = float(target)
    return (f"(LOWER(CAST({col_ident} AS VARCHAR)) = LOWER(?) OR "
            f"TRY_CAST({col_ident} AS DOUBLE) = ?)", [s_target, f_target])
  except (TypeError, ValueError):
    return f"LOWER(CAST({col_ident} AS VARCHAR)) = LOWER(?)", [s_target]


def _build_filter_sql(col_ident: str, op: str,
                      target: Any) -> Tuple[str, List[Any]]:
  if op not in _SUPPORTED_FILTER_OPS:
    raise ValueError(f"Unsupported filter op: {op}")
  is_empty = (f"({col_ident} IS NULL OR CAST({col_ident} AS VARCHAR) = '' "
              f"OR UPPER(CAST({col_ident} AS VARCHAR)) = 'NULL')")
  if op == "is null":
    return is_empty, []
  if op == "is not null":
    return f"NOT {is_empty}", []
  if col_ident == "NULL":
    return "FALSE", []
  if op == "=":
    eq, p = _sql_equals_expr(col_ident, target)
    return f"({col_ident} IS NOT NULL AND {eq})", p
  if op == "!=":
    eq, p = _sql_equals_expr(col_ident, target)
    return f"({col_ident} IS NOT NULL AND NOT {eq})", p
  if op in ("<", "<=", ">", ">="):
    s_target = "" if target is None else str(target)
    try:
      f_target = float(target)
      return (
          f"({col_ident} IS NOT NULL AND CASE "
          f"WHEN TRY_CAST({col_ident} AS DOUBLE) IS NOT NULL "
          f"THEN TRY_CAST({col_ident} AS DOUBLE) {op} ? "
          f"ELSE CAST({col_ident} AS VARCHAR) {op} ? END)",
          [f_target, s_target],
      )
    except (TypeError, ValueError):
      return f"({col_ident} IS NOT NULL AND CAST({col_ident} AS VARCHAR) {op} ?)", [
          s_target
      ]
  if op in ("glob", "not glob"):
    pattern = "" if target is None else str(target)
    neg = "NOT " if op == "not glob" else ""
    return f"({col_ident} IS NOT NULL AND {neg}(CAST({col_ident} AS VARCHAR) GLOB ?))", [
        pattern
    ]
  targets = target if isinstance(target, (list, tuple)) else [target]
  if not targets:
    return ("FALSE" if op == "in" else f"({col_ident} IS NOT NULL)"), []
  parts = []
  params: List[Any] = []
  for t in targets:
    eq, p = _sql_equals_expr(col_ident, t)
    parts.append(eq)
    params.extend(p)
  joined = " OR ".join(parts)
  if op == "in":
    return f"({col_ident} IS NOT NULL AND ({joined}))", params
  return f"({col_ident} IS NOT NULL AND NOT ({joined}))", params


def _duckdb_type_to_bigtrace(dtype: str) -> str:
  u = dtype.upper()
  if "INT" in u and "INTERVAL" not in u and "[" not in u and "(" not in u:
    return "INT64"
  if u.startswith(("DOUBLE", "FLOAT", "DECIMAL", "REAL", "NUMERIC")):
    return "DOUBLE"
  if u == "BOOLEAN":
    return "BOOLEAN"
  return "STRING"


def _to_cell(v: Any) -> Any:
  if v is None or isinstance(v, (bool, int, float, str)):
    return v
  if isinstance(v, (bytes, bytearray, memoryview)):
    import base64  # pylint: disable=import-outside-toplevel
    return base64.b64encode(bytes(v)).decode("ascii")
  if isinstance(v, decimal.Decimal):
    return float(v)
  if isinstance(v, (datetime.datetime, datetime.date, datetime.time)):
    return v.isoformat()
  if isinstance(v, (list, tuple, dict)):
    return json.dumps(v, default=str)
  return str(v)


def _fetch_arrow_columns(
    cur: Any, limit: int) -> Tuple[List[str], Iterator[List[List[Any]]]]:
  import pyarrow as pa  # pylint: disable=import-outside-toplevel
  reader = cur.to_arrow_reader(50_000)
  types = [_arrow_type_to_bigtrace(pa, f.type) for f in reader.schema]

  def chunks() -> Iterator[List[List[Any]]]:
    remaining = limit if limit > 0 else (1 << 62)
    for batch in reader:
      if remaining <= 0:
        return
      if batch.num_rows > remaining:
        batch = batch.slice(0, remaining)
      if batch.num_rows == 0:
        continue
      remaining -= batch.num_rows
      yield [[_to_cell(v) for v in c.to_pylist()] for c in batch.columns]

  return types, chunks()


def _fetch_row_columns(
    cur: Any, limit: int,
    num_cols: int) -> Tuple[List[str], Iterator[List[List[Any]]]]:
  types = [_duckdb_type_to_bigtrace(str(d[1])) for d in cur.description]

  def chunks() -> Iterator[List[List[Any]]]:
    remaining = limit if limit > 0 else (1 << 62)
    while remaining > 0:
      batch = cur.fetchmany(min(remaining, 50_000))
      if not batch:
        break
      remaining -= len(batch)
      cols: List[List[Any]] = [[] for _ in range(num_cols)]
      for r in batch:
        for i, v in enumerate(r):
          cols[i].append(_to_cell(v))
      yield cols

  return types, chunks()


def _arrow_type_to_bigtrace(pa: Any, t: Any) -> str:
  if pa.types.is_boolean(t):
    return "BOOLEAN"
  if pa.types.is_integer(t):
    return "INT64"
  if pa.types.is_floating(t) or pa.types.is_decimal(t):
    return "DOUBLE"
  return "STRING"
