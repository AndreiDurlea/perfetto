"""Benchmarks the interval operators vs TP main (PerfettoSQL Next / stdlib).

usage: python bench/bench_intervals.py [--traces 100] [--tp-par 32] [--threads 48]
         [--skip-tp] [--out bench/results_intervals.json]

Data: ~/bt_bench/duck_intervals/raw600 (tables extracted from raw traces by
TP main with bench/intervals_extract_raw.py), so both engines see the same
rows. For every workload with a TP equivalent, TP runs one process per trace
(sequential, and N in parallel; load time reported separately), DuckDB runs
all traces in one query (PER trace_idx, ...), and per-trace checksums are
compared. DuckDB-only operators are timed on the same traces and on all 600.
Run heavy invocations under:
  systemd-run --user --scope -q -p MemoryMax=30G -p MemorySwapMax=0 ...
"""
import argparse
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_EXT = os.path.join(HERE, '..', 'perfetto.duckdb_extension')


def _private_ext():
  # Load a private copy: other agents rebuild the shared file in place.
  import shutil
  import tempfile
  d = tempfile.mkdtemp(prefix='pf_intervals_')
  dst = os.path.join(d, 'perfetto.duckdb_extension')
  shutil.copy(_SRC_EXT, dst)
  return dst


EXT = _private_ext()
TP = os.path.expanduser('~/bt_bench/tp_arrow/out/rel/trace_processor_shell')
RAW = os.path.expanduser('~/bt_bench/duck_intervals/raw600')

SLICE_THREAD = (
    'SELECT s.id, s.ts, s.dur, t.utid FROM slice s JOIN thread_track t '
    'ON s.track_id = t.id WHERE s.dur >= 0')

# name -> (TP SQL producing one checksum row, DuckDB checksum query).
WORKLOADS = {
    'flatten (COUNT, SUM) PER track': (
        """PERFETTO PRAGMA pipelines = 1;
CREATE PERFETTO TABLE x AS FROM (SELECT ts, dur, track_id, depth FROM slice WHERE dur >= 0)
  |> INTERVAL FLATTEN PER track_id AGGREGATE COUNT(*) AS c, SUM(depth) AS sd;
SELECT count(*) AS n, sum(dur) AS a, sum(c) AS b, sum(sd) AS c FROM x;""",
        """SELECT trace_idx, count(*) AS n, sum(dur) AS a, sum(count) AS b, sum(sum_depth) AS c
FROM pf_interval_flatten_sum('(SELECT trace_idx, ts, dur, track_id, depth FROM d_slice WHERE dur >= 0)',
                             ['depth'], per := ['trace_idx', 'track_id']) GROUP BY 1"""
    ),
    'flatten generic aggs PER track': (
        None,
        """SELECT trace_idx, count(*) AS n, sum(dur) AS a, sum(c) AS b, sum(sd) AS c
FROM pf_interval_flatten('(SELECT trace_idx, ts, dur, track_id, depth FROM d_slice WHERE dur >= 0)',
                         'count(*) AS c, sum(depth) AS sd', per := ['trace_idx', 'track_id']) GROUP BY 1"""
    ),
    'intersection 2-way PER utid': (
        f"""PERFETTO PRAGMA pipelines = 1;
CREATE PERFETTO TABLE x AS INTERVAL INTERSECTION OF (
  (SELECT id, ts, dur, utid FROM thread_state WHERE dur >= 0) AS a,
  ({SLICE_THREAD}) AS b) PER utid |> SELECT ts, dur, utid, a.id AS a_id, b.id AS b_id;
SELECT count(*) AS n, sum(dur) AS a, sum(a_id) AS b, sum(b_id) AS c FROM x;""",
        """SELECT trace_idx, count(*) AS n, sum(dur) AS a, sum(a.id) AS b, sum(b.id) AS c
FROM pf_interval_intersection(['d_ts', 'd_st'], ['a', 'b'], per := ['trace_idx', 'utid']) GROUP BY 1"""
    ),
    'intersection 3-way PER utid': (
        f"""PERFETTO PRAGMA pipelines = 1;
CREATE PERFETTO TABLE x AS INTERVAL INTERSECTION OF (
  (SELECT id, ts, dur, utid FROM thread_state WHERE dur >= 0) AS a,
  ({SLICE_THREAD}) AS b,
  (SELECT id, ts, dur, utid FROM sched WHERE dur >= 0) AS c)
  PER utid |> SELECT ts, dur, utid, a.id AS a_id, b.id AS b_id, c.id AS c_id;
SELECT count(*) AS n, sum(dur) AS a, sum(a_id) AS b, sum(c_id) AS c FROM x;""",
        """SELECT trace_idx, count(*) AS n, sum(dur) AS a, sum(a.id) AS b, sum(c.id) AS c
FROM pf_interval_intersection(['d_ts', 'd_st', 'd_sched'], ['a', 'b', 'c'],
                              per := ['trace_idx', 'utid']) GROUP BY 1"""),
    'merge overlapping PER track (stdlib)': (
        """INCLUDE PERFETTO MODULE intervals.overlap;
SELECT count(*) AS n, sum(dur) AS a, sum(track_id) AS b, sum(ts % 1000003) AS c FROM
  interval_merge_overlapping_partitioned!((SELECT ts, dur, track_id FROM slice WHERE dur > 0), (track_id));""",
        """SELECT trace_idx, count(*) AS n, sum(dur) AS a, sum(track_id) AS b, sum(ts % 1000003) AS c
FROM pf_interval_merge_overlapping('(SELECT trace_idx, ts, dur, track_id FROM d_slice WHERE dur > 0)',
                                   'count(*) AS cnt', per := ['trace_idx', 'track_id']) GROUP BY 1"""
    ),
    'from events: counter intervals PER track': (
        """SELECT count(*) AS n, sum(dur) AS a, sum(id) AS b, sum(track_id) AS c FROM (
  SELECT id, ts, lead(ts, 1, trace_end()) OVER (PARTITION BY track_id ORDER BY ts, id) - ts AS dur, track_id
  FROM counter);""",
        """SELECT trace_idx, count(*) AS n, sum(dur) AS a, sum(id) AS b, sum(track_id) AS c
FROM pf_intervals_from_events('(SELECT c.*, b.end_ts FROM d_counter c JOIN d_bounds b USING (trace_idx))',
                              per := ['trace_idx', 'track_id'], close_last_at := 'end_ts') GROUP BY 1"""
    ),
    # DuckDB-only operators (no TP equivalent yet).
    'union 2-way PER utid':
        (None, """SELECT trace_idx, count(*) AS n, sum(dur) AS a FROM
pf_interval_union(['d_ts', 'd_sched'], ['a', 'b'], per := ['trace_idx', 'utid']) GROUP BY 1"""
        ),
    'split slices by thread_state PER utid':
        (None, """SELECT trace_idx, count(*) AS n, sum(dur) AS a FROM
pf_interval_split('d_st', 'd_ts', 'st', per := ['trace_idx', 'utid']) GROUP BY 1"""
        ),
    'subtract sched from thread_state PER utid':
        (None, """SELECT trace_idx, count(*) AS n, sum(dur) AS a FROM
pf_interval_subtract('d_ts', 'd_sched', per := ['trace_idx', 'utid']) GROUP BY 1"""
        ),
    'left join covering begin (state at slice start)':
        (None, """SELECT trace_idx, count(*) AS n, count(st) AS a FROM
pf_interval_join('d_st', 'd_ts', 'st', 'covering begin', per := ['trace_idx', 'utid'], left_join := true) GROUP BY 1"""
        ),
    'keep slices overlapping sched':
        (None, """SELECT trace_idx, count(*) AS n FROM
pf_interval_keep_if('d_st', 'd_sched', 'overlapping', per := ['trace_idx', 'utid']) GROUP BY 1"""
        ),
    'find next slice starting after end PER track':
        (None, """SELECT trace_idx, count(*) AS n, count(nxt) AS a FROM
pf_interval_find('(SELECT trace_idx, id, ts, dur, track_id FROM d_slice WHERE dur >= 0)',
                 '(SELECT trace_idx, id, ts, dur, track_id FROM d_slice WHERE dur >= 0)', 'nxt',
                 'starting', 'after', 'end', per := ['trace_idx', 'track_id'], left_join := true) GROUP BY 1"""
        ),
    'quantize thread_state 1ms':
        (None, """SELECT trace_idx, count(*) AS n, sum(dur) AS a FROM
pf_interval_quantize('d_ts', 1000000, 'trace_start', bounds := '(SELECT trace_idx, start_ts FROM d_bounds)') GROUP BY 1"""
        ),
    'merge consecutive thread_state BY state':
        (None, """SELECT trace_idx, count(*) AS n, sum(dur) AS a FROM
pf_interval_merge_consecutive('d_ts', ['state'], 'count(*) AS cnt', per := ['trace_idx', 'utid']) GROUP BY 1"""
        ),
}


def tp_one(sql, path):
  with tempfile.NamedTemporaryFile('r', suffix='.perf') as perf:
    t0 = time.time()
    p = subprocess.run(
        [TP, 'query', '--quiet', '-f', '-', '--perf-file', perf.name, path],
        input=sql,
        capture_output=True,
        text=True)
    wall = time.time() - t0
    if p.returncode != 0:
      return dict(error=p.stderr[-300:], wall=wall)
    load_ns, query_ns = map(int, perf.read().strip().split(','))
  rows = [r for r in csv.reader(io.StringIO(p.stdout.strip())) if r]
  hdr, vals = rows[-2], rows[-1]
  chk = {h: (None if v in ('', '[NULL]') else int(v)) for h, v in zip(hdr, vals)}
  return dict(load=load_ns / 1e9, query=query_ns / 1e9, wall=wall, chk=chk)


def run_tp(sql, traces, par):
  t0 = time.time()
  with ThreadPoolExecutor(par) as ex:
    res = list(ex.map(lambda t: tp_one(sql, t[1]), traces))
  wall = time.time() - t0
  ok = [r for r in res if 'chk' in r]
  return dict(
      wall=wall,
      query=sum(r['query'] for r in ok),
      load=sum(r['load'] for r in ok),
      errors=len(res) - len(ok),
      per={
          t[0]: r.get('chk') for t, r in zip(traces, res)
      })


def duck_init(trace_ids):
  c = duckdb.connect(config={'allow_unsigned_extensions': 'true'})
  c.execute(f"LOAD '{EXT}'")
  c.execute(
      "SET threads = 32; SET memory_limit = '20GB'; SET preserve_insertion_order = false"
  )
  for t in [
      'slice', 'thread_track', 'thread_state', 'sched', 'counter',
      'trace_bounds'
  ]:
    c.execute(
        f"CREATE TEMP TABLE d_{ {'trace_bounds': 'bounds', 'sched': 'sched_raw'}.get(t, t)} AS "
        f"SELECT * FROM read_parquet('{RAW}/{t}/*.parquet')")
  c.execute(
      "CREATE TEMP VIEW d_ts AS SELECT trace_idx, id, ts, dur, utid, state FROM d_thread_state WHERE dur >= 0"
  )
  c.execute(
      "CREATE TEMP VIEW d_st AS SELECT s.trace_idx, s.id, s.ts, s.dur, t.utid FROM d_slice s "
      "JOIN d_thread_track t USING (trace_idx) WHERE s.track_id = t.id AND s.dur >= 0"
  )
  c.execute(
      "CREATE TEMP VIEW d_sched AS SELECT trace_idx, id, ts, dur, utid FROM d_sched_raw WHERE dur >= 0"
  )
  c.execute('CREATE TEMP TABLE sel AS SELECT unnest(?::BIGINT[]) AS trace_idx',
            [trace_ids])
  c.execute('CREATE SCHEMA IF NOT EXISTS sub')
  for t in [
      'slice', 'thread_track', 'thread_state', 'sched_raw', 'counter', 'bounds'
  ]:
    c.execute(f'CREATE OR REPLACE TABLE sub.d_{t} AS SELECT * FROM temp.d_{t} '
              f'WHERE trace_idx IN (SELECT trace_idx FROM sel)')
  c.execute(
      "CREATE OR REPLACE VIEW sub.d_ts AS SELECT trace_idx, id, ts, dur, utid, state FROM sub.d_thread_state WHERE dur >= 0"
  )
  c.execute(
      "CREATE OR REPLACE VIEW sub.d_st AS SELECT s.trace_idx, s.id, s.ts, s.dur, t.utid FROM sub.d_slice s "
      "JOIN sub.d_thread_track t USING (trace_idx) WHERE s.track_id = t.id AND s.dur >= 0"
  )
  c.execute(
      "CREATE OR REPLACE VIEW sub.d_sched AS SELECT trace_idx, id, ts, dur, utid FROM sub.d_sched_raw WHERE dur >= 0"
  )
  return c


def _sub_exec(c, dq):
  return c.sql(dq.replace("'d_", "'sub.d_").replace(' d_',
                                                    ' sub.d_')).fetchall()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--traces', type=int, default=100)
  ap.add_argument('--tp-par', type=int, default=32)
  ap.add_argument('--threads', default='1,32')
  ap.add_argument('--skip-tp', action='store_true')
  ap.add_argument('--skip-all', action='store_true')
  ap.add_argument('--only')
  ap.add_argument('--out', default=os.path.join(HERE, 'results_intervals.json'))
  a = ap.parse_args()
  thread_list = [int(x) for x in str(a.threads).split(',')]

  paths = dict(l.split() for l in open(os.path.join(RAW, 'traces.txt')))
  probe = duckdb.sql(
      f"SELECT DISTINCT trace_idx FROM read_parquet('{RAW}/slice/*.parquet') "
      f"ORDER BY 1 LIMIT {a.traces}").fetchall()
  traces = [(i, paths[str(i)]) for (i,) in probe]
  ids = [t[0] for t in traces]
  # Phase 1: TP (no DuckDB memory held).
  tp = {}
  if not a.skip_tp:
    for name, (tp_sql, _) in WORKLOADS.items():
      if tp_sql and (not a.only or a.only in name):
        tp[name] = (run_tp(tp_sql, traces, 1), run_tp(tp_sql, traces, a.tp_par))
        print(f'tp done: {name}', flush=True)
  # Phase 2: DuckDB, one database; the subset is pre-materialized in schema `sub`.
  c = duck_init(ids)
  rows_all = c.sql(
      'SELECT count(*), count(DISTINCT trace_idx) FROM d_slice').fetchone()
  print(
      f'subset: {len(ids)} traces; all: {rows_all[1]} traces with slices, {rows_all[0]} slices',
      flush=True)
  results = []
  for name, (tp_sql, dq) in WORKLOADS.items():
    if a.only and a.only not in name:
      continue
    r = dict(workload=name)
    duck_per = None
    for th in thread_list:
      c.execute(f'SET threads = {th}')
      os.environ['PF_THREADS'] = str(th)
      tags = ['sub'] if a.skip_all else ['sub', 'all']
      for tag in tags:
        times = []
        for _ in range(3):
          t0 = time.time()
          rows = _sub_exec(c, dq) if tag == 'sub' else c.sql(dq).fetchall()
          times.append(time.time() - t0)
        r[f'duck_t{th}_{tag}_s'] = round(min(times), 3)
        r[f'duck_{tag}_rows'] = sum(x[1] for x in rows)
        if tag == 'sub' and duck_per is None:
          cols = [d[0] for d in c.sql(dq + ' LIMIT 0').description]
          duck_per = {x[0]: dict(zip(cols, x)) for x in rows}
    if name in tp:
      seq, par = tp[name]
      r.update(
          tp_seq_query_s=round(seq['query'], 2),
          tp_seq_load_s=round(seq['load'], 2),
          tp_seq_wall_s=round(seq['wall'], 2),
          tp_par_wall_s=round(par['wall'], 2),
          tp_errors=seq['errors'],
          tp_rows=sum((v or {}).get('n') or 0 for v in seq['per'].values()))
      mism = 0
      for idx, chk in seq['per'].items():
        if chk is None:
          continue
        d = duck_per.get(idx)
        if not chk.get('n'):
          mism += bool(d and d['n'])
          continue
        mism += d is None or any(chk[k] != d[k] for k in chk)
      r['checksum_mismatch_traces'] = mism
      for th in thread_list:
        r[f'speedup_t{th}_vs_tp_seq_query'] = round(
            seq['query'] / r[f'duck_t{th}_sub_s'], 2)
        r[f'speedup_t{th}_vs_tp_par_wall'] = round(
            par['wall'] / r[f'duck_t{th}_sub_s'], 2)
    print(json.dumps(r), flush=True)
    results.append(r)
  with open(a.out, 'w') as f:
    json.dump(results, f, indent=1)


if __name__ == '__main__':
  main()
