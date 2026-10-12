# Bigtrace local server

`tools/bigtrace_server.py` serves the Bigtrace UI API over traces on local
disk, so the Bigtrace UI can query many traces without a cloud backend. It needs
only Python 3 and the `perfetto` package in this repo.

The manifests in this directory are format examples and test data for
`python/test/bigtrace_server_*_unittest.py`.

## Quick start

```bash
tools/bigtrace_server.py /path/to/traces --generate-manifest
ui/run-dev-server --bigtrace    # in another shell
```

Open `http://localhost:10000/bigtrace.html?server=http://127.0.0.1:5001`.
`?server=` sets the backend endpoint, same as editing it in the connection
panel.

`/path/to/traces` is a directory or an archive. Zip and tar files (plain, gz,
bz2 or xz) are recognised by content, whatever their name, and unpacked once
to `~/.cache/perfetto/bigtrace/`; the unpacked copy, with its manifest and
query history, is reused until the archive changes.

Agents can do all of this with the [`bigtrace` skill](../../ai/skills/bigtrace/SKILL.md).

## Options

| Flag | Default | Description |
| --- | --- | --- |
| `TRACES` | `.` | Directory or archive of traces. |
| `-r`, `--recursive` | off (on for archives) | Scan subdirectories. |
| `-m`, `--manifest PATH` | `bigtrace_manifest.json` or `metadata.db` in the trace directory, if present | Manifest to load; CSV if it ends in `.csv`, a trace_fetch SQLite index if it ends in `.db`. |
| `-p`, `--port` | `5001` | Port to listen on. |
| `--host` | `127.0.0.1` | Address to bind. |
| `-w`, `--workers` | CPU count, at most one per 2 GB of RAM | Traces queried in parallel. |
| `--shell-path PATH` | auto | `trace_processor_shell` to use. |
| `--snapshot-dir DIR` | off | Optional directory for parse-once `export perfetto` snapshots. |
| `--snapshot-budget-gb N` | `50` | Disk budget for `--snapshot-dir` snapshots; 0 disables them. |
| `--parquet-dir DIR` | off | Optional directory to export each query's DuckDB table as `<name>.parquet` and reload existing `.parquet` files on startup. |
| `--duckdb-path PATH` | `:memory:` | Optional file path for persistent DuckDB storage (tables and query history). |
| `--origin URL` | `https://ui.perfetto.dev` | Perfetto UI that trace links open in. |
| `--cors-origin ORIGIN` | | Extra origin allowed to call the server; repeatable. |
| `--generate-manifest [OUT]` | `bigtrace_manifest.json` in the trace directory | Update the manifest before serving. |
| `-f`, `--force` | off | With `--generate-manifest`: re-extract every trace. |
| `--metadata-query SQL` | | With `--generate-manifest`: extra columns per trace. |

## Manifests

Without a manifest the trace list only has file columns (name, size, link).
`--generate-manifest` loads every trace in parallel before serving and records
its bounds, duration, process, thread and slice counts, and the build/OS rows
of the `metadata` table, so the list can be filtered and sorted by them. Later
runs pick up `bigtrace_manifest.json` from the trace directory without the
flag.

With the flag, only new or changed traces (by size and mtime) are loaded and
deleted ones are dropped, so passing it every time is cheap; `--force`
re-extracts everything. A running server re-reads its source on
`POST /refresh`.

`--metadata-query` adds the first row of a query to each trace:

```bash
tools/bigtrace_server.py /path/to/traces --generate-manifest \
  --metadata-query "SELECT count(1) AS jank_slices FROM slice WHERE name GLOB '*Choreographer*'"
tools/bigtrace_server.py /path/to/traces --generate-manifest \
  --metadata-query "$(cat metadata.sql)"
```

## Query execution and DuckDB tables

Each trace query runs in parallel across all selected traces (`--workers` at a
time), attaches `_trace_uuid` to every row, and streams its results directly
into a DuckDB table (`result_1`, `result_2`, ... or an alternate name set via
the `Table:` field in the UI toolbar). UI paging, sorting, and filtering stream
pages directly from DuckDB.

Switching an editor tab from **Traces** to **DuckDB** (or clicking **Query in
DuckDB** on any history item) executes DuckDB SQL directly against the shared
DuckDB instance, with the PerfettoSQL Next DuckDB extension
(`python/perfetto/bigtrace/duckdb_ext/`) pre-loaded (`pf_interval_*`,
`pf_tree_*`, and `pf_graph_*` operators). Because every query result table and
`trace_metadata` share `_trace_uuid`, you can query a single run or join across
any number of historical query tables in milliseconds:

```sql
SELECT
  r1._trace_uuid,
  m.device_name,
  r1.startup_dur_ns,
  r2.binder_count
FROM result_1 r1
JOIN result_2 r2 USING (_trace_uuid)
JOIN trace_metadata m USING (_trace_uuid)
ORDER BY r1.startup_dur_ns DESC;
```

Passing `--duckdb-path PATH` persists both DuckDB tables and query execution
history to a DuckDB file across restarts. Passing `--parquet-dir DIR` exports
each query's DuckDB table to `<DIR>/<table_name>.parquet` and reloads existing
`.parquet` files into DuckDB on startup. Passing `--snapshot-dir DIR` caches
parse-once `export perfetto` snapshots (up to `--snapshot-budget-gb` GB) so
subsequent trace queries load about 4x faster than parsing raw traces.

## Opening traces

Clicking a trace's `link` opens it in the Perfetto UI, which fetches it from
this server:

```
https://ui.perfetto.dev/#!/?url=http://127.0.0.1:5001/trace/<uuid>/<name>&referrer=bigtrace_server
```

Like `tools/open_trace_in_ui`, `--origin` selects another UI, e.g.
`--origin http://localhost:10000`.

## Access

Browsers may only call the server from localhost origins, `https://ui.perfetto.dev`
and `--origin`; add others with `--cors-origin`. Anyone who can reach the port
can read every trace it serves, so keep the default `--host 127.0.0.1` unless you
need otherwise.

## Manifest formats

JSON (see `bigtrace_manifest.json`):

```json
{
  "trace_dir": ".",
  "schema": [
    {"name": "duration_ms", "displayName": "Duration (ms)", "type": "FLOAT64",
     "defaultVisible": true, "description": "Trace duration in milliseconds"}
  ],
  "traces": [
    {"file_path": "startup_cold.perfetto-trace", "duration_ms": 2997.996,
     "experiment_id": 101, "control_id": 102, "is_treatment": true}
  ],
  "presets": [
    {"id": "builtin:slice_count", "category": "General",
     "name": "Slice count per trace",
     "perfettoSql": "SELECT count(1) AS slice_count FROM slice;"}
  ],
  "experiments": [
    {"experimentId": 101, "experimentName": "PGO", "controlId": 102,
     "controlName": "Baseline"}
  ]
}
```

Only `traces[].file_path` is required. Paths are relative to `trace_dir`, which
is relative to the manifest. Other trace keys become metadata columns. `schema`
sets display names, types and default visibility; unlisted columns are
inferred. `experiment_id`, `control_id` and `is_treatment` drive the UI's
experiment filter.

CSV (see `bigtrace_manifest.csv`) has one row per trace with the same keys:

```csv
file_path,device_name,experiment_id,control_id,is_treatment
startup_cold.perfetto-trace,Pixel 8,101,102,true
```
