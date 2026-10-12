"""Extracts interval-relevant tables from raw traces with TP main into Parquet,
so DuckDB and TP (PerfettoSQL Next oracle) see exactly the same data.

usage: extract_raw.py <n_traces> <out_dir> [workers]
Writes <out_dir>/<table>/part-<trace_idx>.parquet and <out_dir>/traces.txt
("<trace_idx> <path>" for every trace that loaded).
"""
import csv
import io
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq

TP = os.path.expanduser('~/bt_bench/tp_arrow/out/rel/trace_processor_shell')
RAW = os.path.expanduser('~/bigtrace_10k/traces')

TABLES = {
    'slice': 'SELECT id, ts, dur, track_id, depth, parent_id, name FROM slice',
    'thread_track': 'SELECT id, utid FROM thread_track',
    'thread': 'SELECT utid, tid, upid, is_main_thread, name FROM thread',
    'thread_state': 'SELECT id, ts, dur, utid, state, cpu FROM thread_state',
    'sched': 'SELECT id, ts, dur, cpu, utid FROM sched',
    'counter': 'SELECT id, ts, track_id, value FROM counter',
    'trace_bounds': 'SELECT start_ts, end_ts FROM trace_bounds',
}
INT_COLS = {
    'id', 'ts', 'dur', 'track_id', 'depth', 'parent_id', 'utid', 'cpu', 'tid',
    'upid', 'is_main_thread', 'start_ts', 'end_ts'
}


def conv(name, vals):
  if name in INT_COLS:
    return pa.array([None if v in ('', '[NULL]') else int(v) for v in vals],
                    pa.int64())
  if name == 'value':
    return pa.array([None if v in ('', '[NULL]') else float(v) for v in vals],
                    pa.float64())
  return pa.array([None if v == '[NULL]' else v for v in vals], pa.string())


def extract(job):
  idx, path, out = job
  sql = ';\n'.join(TABLES.values()) + ';\n'
  p = subprocess.run([TP, 'query', '--quiet', '-f', '-', path],
                     input=sql,
                     capture_output=True,
                     text=True)
  if p.returncode != 0:
    return idx, path, False
  sections, cur = [], []
  for row in csv.reader(io.StringIO(p.stdout)):
    if not row:
      if cur:
        sections.append(cur)
      cur = []
      continue
    cur.append(row)
  if cur:
    sections.append(cur)
  if len(sections) != len(TABLES):
    return idx, path, False
  for (tname, _), sec in zip(TABLES.items(), sections):
    hdr, rows = sec[0], sec[1:]
    cols = {h: [r[i] for r in rows] for i, h in enumerate(hdr)}
    t = pa.table({
        'trace_idx': pa.array([idx] * len(rows), pa.int64()),
        **{
            h: conv(h, v) for h, v in cols.items()
        }
    })
    d = os.path.join(out, tname)
    os.makedirs(d, exist_ok=True)
    pq.write_table(t, os.path.join(d, f'part-{idx:05d}.parquet'))
  return idx, path, True


def main():
  n, out = int(sys.argv[1]), sys.argv[2]
  workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16
  files = sorted(
      os.path.join(RAW, f)
      for f in os.listdir(RAW)
      if f.endswith('.pftrace.zst'))[:n]
  with ThreadPoolExecutor(workers) as ex:
    res = list(ex.map(extract, [(i, f, out) for i, f in enumerate(files)]))
  with open(os.path.join(out, 'traces.txt'), 'w') as f:
    for idx, path, ok in res:
      if ok:
        f.write(f'{idx} {path}\n')
  print(f'{sum(r[2] for r in res)}/{len(res)} traces extracted')


if __name__ == '__main__':
  main()
