"""Benchmarks the tree operators vs trace processor (TP main), same traces.

usage: python bench/bench_trees.py [N=100] [PAR=24] [THREADS=1,8,16,32] [MODE=all|duck|tp]

Data: the raw traces listed in ~/bt_bench/raw1k.txt; DuckDB reads the slices
TP main extracted from exactly those traces (~/bt_bench/ex1k/slice/<idx>.parquet).
Every op returns per trace (n, s) checksums which must match TP's.

TP side, one TP main process per trace (raw trace, heap graphs skipped):
  * TREE ACCUMULATE UP/DOWN: PerfettoSQL Next pipe syntax (native operator).
  * every other op: the closest PerfettoSQL formulation (ancestor_slice /
    correlated subqueries); TP has no native operator for them.
Reports TP sequential (sum of per-trace query time; load reported apart), TP
parallel (PAR processes, wall incl. load, and the load-only wall), DuckDB at
each thread count (best of 3, data already in a DuckDB table).
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

B = os.path.expanduser('~/bt_bench')
sys.path.insert(0, f'{B}/tp_arrow/python')
import duckdb  # noqa: E402
from perfetto.trace_processor import TraceProcessor  # noqa: E402
from perfetto.trace_processor import TraceProcessorConfig  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.environ.get('PF_EXT', os.path.join(ROOT, 'perfetto.duckdb_extension'))
TP_BIN = f'{B}/tp_arrow/out/rel/trace_processor_shell'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
PAR = int(sys.argv[2]) if len(sys.argv) > 2 else 24
THREADS = [
    int(x)
    for x in (sys.argv[3] if len(sys.argv) > 3 else '1,8,16,32').split(',')
]
MODE = sys.argv[4] if len(sys.argv) > 4 else 'all'
OPS = os.environ.get('TREE_OPS')

paths = [l.strip() for l in open(f'{B}/raw1k.txt')]
have = sorted(int(f[:-8]) for f in os.listdir(f'{B}/ex1k/slice'))
traces = have[:N]

MK = 'depth IN (1, 2) AND dur > 5000000'
MKa = 'a.depth IN (1, 2) AND a.dur > 5000000'

# op -> (TP query returning one row (n, s), DuckDB query returning (trace_idx, n, s))
Q = {
    'accumulate_up':
        ("""CREATE OR REPLACE PERFETTO TABLE r AS
FROM (SELECT id, parent_id, dur FROM slice) AS s |> TREE ACCUMULATE UP SUM(dur) AS total;
SELECT count(*) AS n, sum(total) AS s FROM r""",
         """SELECT trace_idx, count(*), sum(total) FROM pf_tree_accumulate_up(
  '(SELECT trace_idx, id, parent_id, dur FROM s)', 'dur', per := ['trace_idx']) GROUP BY 1"""
        ),
    'accumulate_down':
        ("""CREATE OR REPLACE PERFETTO TABLE r AS
FROM (SELECT id, parent_id, dur FROM slice) AS s |> TREE ACCUMULATE DOWN SUM(dur) AS total;
SELECT count(*) AS n, sum(total) AS s FROM r""",
         """SELECT trace_idx, count(*), sum(total) FROM pf_tree_accumulate_down(
  '(SELECT trace_idx, id, parent_id, dur FROM s)', 'dur', per := ['trace_idx']) GROUP BY 1"""
        ),
    'keep_if_descendant': (
        f"""SELECT count(*) AS n, sum(s.id) AS s FROM slice s
WHERE EXISTS (SELECT 1 FROM ancestor_slice(s.id) a WHERE {MKa})""",
        f"""SELECT trace_idx, count(*), sum(id) FROM pf_tree_keep_if('s', 'descendant',
  '(SELECT trace_idx, id FROM s WHERE {MK})', per := ['trace_idx']) GROUP BY 1"""
    ),
    'keep_if_ancestor': (
        f"""SELECT count(*) AS n, sum(id) AS s FROM slice WHERE id IN (
  SELECT a.id FROM (SELECT * FROM slice WHERE {MK}) m, ancestor_slice(m.id) a)""",
        f"""SELECT trace_idx, count(*), sum(id) FROM pf_tree_keep_if('s', 'ancestor',
  '(SELECT trace_idx, id FROM s WHERE {MK})', per := ['trace_idx']) GROUP BY 1"""
    ),
    'prune': (
        f"""SELECT count(*) AS n, sum(s.id) AS s FROM slice s WHERE NOT ({MK.replace('depth', 's.depth').replace('dur', 's.dur')})
AND NOT EXISTS (SELECT 1 FROM ancestor_slice(s.id) a WHERE {MKa})""",
        f"""SELECT trace_idx, count(*), sum(id) FROM pf_tree_prune('s',
  '(SELECT trace_idx, id FROM s WHERE {MK})', per := ['trace_idx']) GROUP BY 1"""
    ),
    'reparent': (
        f"""SELECT count(*) AS n, sum(coalesce((SELECT a.id FROM ancestor_slice(s.id) a WHERE {MKa}
  ORDER BY a.depth DESC LIMIT 1), -1)) AS s FROM slice s""",
        f"""SELECT trace_idx, count(*), sum(coalesce(parent_id, -1)) FROM pf_tree_reparent('s',
  '(SELECT trace_idx, id FROM s WHERE {MK})', per := ['trace_idx']) GROUP BY 1"""
    ),
    'absorb': (
        f"""SELECT count(*) AS n, sum(sd + owner) AS s FROM (
  SELECT owner, sum(dur) AS sd FROM (
    SELECT coalesce((SELECT a.id FROM ancestor_slice(s.id) a WHERE {MKa} ORDER BY a.depth LIMIT 1),
      s.id) AS owner, s.dur FROM slice s) GROUP BY owner)""",
        f"""SELECT trace_idx, count(*), sum(sd + id) FROM pf_tree_absorb('(SELECT trace_idx, id, parent_id, dur FROM s)',
  '(SELECT trace_idx, id FROM s WHERE {MK})', ['sum(dur) AS sd'], per := ['trace_idx']) GROUP BY 1"""
    ),
    'contract': (
        f"""SELECT count(*) AS n, sum(sd + owner) AS s FROM (
  SELECT owner, sum(dur) AS sd FROM (
    SELECT CASE WHEN {MK.replace('depth', 's.depth').replace('dur', 's.dur')} THEN (SELECT a.id FROM ancestor_slice(s.id) a
      WHERE NOT ({MKa}) ORDER BY a.depth DESC LIMIT 1) ELSE s.id END AS owner, s.dur FROM slice s)
  WHERE owner IS NOT NULL GROUP BY owner)""",
        f"""SELECT trace_idx, count(*), sum(sd + id) FROM pf_tree_contract('(SELECT trace_idx, id, parent_id, dur FROM s)',
  '(SELECT trace_idx, id FROM s WHERE {MK})', ['sum(dur) AS sd'], per := ['trace_idx']) GROUP BY 1"""
    ),
    'merge_siblings': (
        """SELECT count(*) AS n, sum(sd) AS s FROM (
  SELECT path, sum(dur) AS sd FROM (
    SELECT coalesce((SELECT group_concat(coalesce(name, '?'), char(31)) FROM (
      SELECT name FROM ancestor_slice(s.id) ORDER BY depth)), '') || char(31) || coalesce(s.name, '?') AS path,
      s.dur FROM slice s) GROUP BY path)""",
        """SELECT trace_idx, count(*), sum(sd) FROM pf_tree_merge_siblings('(SELECT trace_idx, id, parent_id, name, dur FROM s)',
  ['name'], ['sum(dur) AS sd'], per := ['trace_idx']) GROUP BY 1"""),
    'merge_into_parent': (
        """SELECT count(*) AS n, sum(sd + owner) AS s FROM (
  SELECT owner, sum(dur) AS sd FROM (
    SELECT CASE WHEN od = s.depth THEN s.id ELSE (SELECT a.id FROM ancestor_slice(s.id) a WHERE a.depth = od) END
      AS owner, s.dur FROM (
      SELECT s.*, coalesce((SELECT max(a.depth) FROM ancestor_slice(s.id) a WHERE a.name IS NOT s.name) + 1, 0) AS od
      FROM slice s) s) GROUP BY owner)""",
        """SELECT trace_idx, count(*), sum(sd + id) FROM pf_tree_merge_into_parent('(SELECT trace_idx, id, parent_id, name, dur FROM s)',
  ['name'], ['sum(dur) AS sd'], per := ['trace_idx']) GROUP BY 1"""),
    'invert': (
        """SELECT count(*) AS n, sum(x) AS s FROM (
  SELECT l.id AS x FROM slice l WHERE l.id NOT IN (SELECT parent_id FROM slice WHERE parent_id IS NOT NULL)
  UNION ALL
  SELECT a.id FROM (SELECT * FROM slice WHERE id NOT IN (SELECT parent_id FROM slice WHERE parent_id IS NOT NULL)) l,
    ancestor_slice(l.id) a)""",
        """SELECT trace_idx, count(*), sum(node_id) FROM pf_tree_invert('(SELECT trace_idx, id, parent_id FROM s)',
  per := ['trace_idx']) GROUP BY 1"""),
    # DuckDB only (no TP equivalent):
    'accumulate_multi': (
        None,
        """SELECT trace_idx, count(*), sum(n) + sum(m) + sum(length(a)) FROM pf_tree_accumulate('s', 'up',
  ['count(*) AS n', 'max(dur) AS m', 'arg_max(name, dur) AS a'], per := ['trace_idx']) GROUP BY 1"""
    ),
    'merge_into_parent_indirect': (
        None,
        """SELECT trace_idx, count(*), sum(sd + id) FROM pf_tree_merge_into_parent('(SELECT trace_idx, id, parent_id, name, dur FROM s)',
  ['name'], ['sum(dur) AS sd'], per := ['trace_idx'], indirect := true) GROUP BY 1"""
    ),
    'merge_siblings_ordered': (
        None,
        """SELECT trace_idx, count(*), sum(sd) FROM pf_tree_merge_siblings('(SELECT trace_idx, id, parent_id, name, dur FROM s)',
  ['name'], ['sum(dur) AS sd'], per := ['trace_idx'], ordered := true) GROUP BY 1"""
    ),
    'expand_down': (
        None,
        """SELECT trace_idx, count(*), sum(id) FROM pf_tree_expand('(SELECT trace_idx, id, parent_id, name, dur FROM s)',
  'down', '(SELECT ''binder transaction'' AS name, range AS seq FROM range(3))', ['name'],
  ordered_by := ['seq'], charge := ['dur'], charge_to := 'leaf', per := ['trace_idx']) GROUP BY 1"""
    ),
}
if OPS:
  Q = {k: v for k, v in Q.items() if k in OPS.split(',')}


def tp_open(idx):
  return TraceProcessor(
      trace=paths[idx],
      config=TraceProcessorConfig(
          bin_path=TP_BIN,
          extra_flags=['--dev', '--dev-flag', 'skip-heap-graph=true'],
          load_timeout=300))


def tp_run(idx, ops=None):
  t0 = time.time()
  tp = tp_open(idx)
  out = {'load': time.time() - t0}
  tp.query('PERFETTO PRAGMA pipelines = 1')
  for name, (q, _) in Q.items():
    if q is None or (ops is not None and name not in ops):
      continue
    t = time.time()
    try:
      r = tp.query_arrow(q).to_pylist()[0]
      out[name] = (time.time() - t, r['n'], r['s'] or 0)
    except Exception as e:  # pylint: disable=broad-except
      out[name] = (time.time() - t, 'err', str(e)[:200])
  tp.close()
  return idx, out


def duck_con(threads):
  con = duckdb.connect(config={'allow_unsigned_extensions': 'true'})
  con.execute(f'SET threads={threads}')
  con.execute("SET memory_limit='20GB'")
  con.execute('SET preserve_insertion_order=false')
  con.execute(f"LOAD '{EXT}'")
  return con


def main():
  report = {'traces': len(traces)}
  tp_ck = {}
  if MODE in ('all', 'tp'):
    t0 = time.time()
    seq = [tp_run(i) for i in traces]
    report['tp_seq_wall'] = time.time() - t0
    report['tp_load_sum'] = sum(o['load'] for _, o in seq)
    for name, (q, _) in Q.items():
      if q is not None:
        report[f'tp_seq_{name}'] = sum(o[name][0] for _, o in seq)
    tp_ck = {i: o for i, o in seq}
    t0 = time.time()
    with ThreadPoolExecutor(PAR) as ex:
      list(ex.map(lambda i: tp_run(i, ops=set()), traces))
    report[f'tp_par{PAR}_load_only'] = time.time() - t0
    for name, (q, _) in Q.items():
      if q is None:
        continue
      t0 = time.time()
      with ThreadPoolExecutor(PAR) as ex:
        list(ex.map(lambda i: tp_run(i, ops={name}), traces))
      report[f'tp_par{PAR}_{name}_incl_load'] = time.time() - t0
    json.dump({
        str(k): v for k, v in tp_ck.items()
    }, open('/tmp/bench_trees_tp.json', 'w'))
  elif os.path.exists('/tmp/bench_trees_tp.json'):
    tp_ck = {
        int(k): v
        for k, v in json.load(open('/tmp/bench_trees_tp.json')).items()
    }
  if MODE in ('all', 'duck'):
    files = [f'{B}/ex1k/slice/{i:06d}.parquet' for i in traces]
    con = duck_con(max(max(THREADS), 32))
    t0 = time.time()
    con.execute(
        f'CREATE TABLE s AS SELECT * FROM read_parquet({files!r}, union_by_name=true)'
    )
    report['duck_load'] = time.time() - t0
    report['slices'] = con.execute('SELECT count(*) FROM s').fetchone()[0]
    for th in THREADS:
      con.execute(f'SET threads={th}')
      os.environ['PF_TREE_THREADS'] = str(th)
      for name, (_, dq) in Q.items():
        best = None
        try:
          for _ in range(3):
            t = time.time()
            rows = con.execute(dq).fetchall()
            dt = time.time() - t
            best = dt if best is None else min(best, dt)
        except duckdb.Error as e:
          report[f'duck_t{th}_{name}'] = 'err: ' + str(e)[:200]
          print(f't={th} {name}: ERR {e}', flush=True)
          continue
        report[f'duck_t{th}_{name}'] = best
        print(f't={th} {name}: {best:.3f}s', flush=True)
        got = {r[0]: (r[1], r[2]) for r in rows}
        if tp_ck and Q[name][0] is not None:
          bad = []
          for i in traces:
            o = tp_ck[i].get(name)
            if o is None:
              continue
            want = (o[1], o[2]) if o[1] else (0, 0)
            if got.get(
                i,
                (0, 0)) != tuple(want) and not (want[0] == 0 and i not in got):
              bad.append(i)
          report[f'mismatch_t{th}_{name}'] = len(bad)
          if bad:
            report[f'example_{name}'] = [
                bad[0], tp_ck[bad[0]].get(name),
                got.get(bad[0])
            ]
    con.close()
  print(
      json.dumps(
          {
              k: round(v, 3) if isinstance(v, float) else v
              for k, v in report.items()
          },
          indent=1),
      flush=True)


if __name__ == '__main__':
  main()
