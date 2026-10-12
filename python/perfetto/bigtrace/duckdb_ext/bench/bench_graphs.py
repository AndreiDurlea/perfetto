"""Graph operator benchmarks vs trace processor, on real Java heap dumps.

Subcommands (run from the repo root; wrap heavy runs in
`systemd-run --user --scope -q -p MemoryMax=30G -p MemorySwapMax=0`):

  scan-select --scan F [--n 20]
      Pick N heap-dump traces of varied size (always incl. the largest) from a
      `bench/graphs_scan_heap.sh` output; writes DATA/traces.txt.
  extract [--par 4]
      `trace_processor_shell export arrow_tar` per trace (TP main), keeps the
      raw heap_graph_object / heap_graph_reference / heap_graph_class columns
      as Parquet with a trace_idx column (DATA/raw/<table>/t<idx>.parquet).
      DuckDB derives everything else itself (excluded refs, edges, roots).
  oracle [--par 4]
      TP main outputs used by tests/test_graphs.py: graph_reachable_dfs /
      graph_reachable_bfs on the dominator edge set (edges ordered by
      (src, dst), starts ordered by id DESC for DFS, ASC for BFS) and
      android.memory.heap_graph.dominator_tree's heap_graph_dominator_tree.
  tp [--bin main|legacy] [--par P] [--ops a,b]
      TP timings, one process per (trace, op), `--perf-file` load/query split.
      P=1 is sequential; P>1 measures batch wall time.
  duck [--threads 1,48]
      DuckDB timings over all traces at once (per := ['trace_idx', 'upid']):
      load excluded (inputs in DuckDB tables) and load included (from Parquet).
  report
      Markdown tables from DATA/results/*.json.

Pipelines (same semantics on both engines):
  dom_retained : edges(TP excluded_refs filter) -> dominator tree (virtual
                 super-root over GC roots) -> retained = TREE ACCUMULATE UP
                 SUM(self_size).  TP: heap_graph_dominator_tree (stdlib).
  object_tree  : BFS shortest-path tree -> TREE ACCUMULATE UP SUM(self_size).
                 TP: _heap_graph_object_tree_aggregation (stdlib, uses pipe
                 TREE ACCUMULATE in main).  NB TP orders BFS seeds by class
                 name; we tie-break by id, so parents can differ (timing only).
  dfs / bfs / dom_raw : the bare operators (graph_reachable_dfs/bfs,
                 graph_dominator_tree) over the dominator edge set.
"""
import argparse
import csv
import glob
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(ROOT, 'perfetto.duckdb_extension')
DATA = os.environ.get('GRAPH_DATA',
                      os.path.expanduser('~/bt_bench/graphs_data'))
BINS = {
    'main':
        os.path.expanduser('~/bt_bench/tp_arrow/out/rel/trace_processor_shell'),
    'legacy':
        os.path.expanduser('~/dev/perfetto/out/release/trace_processor_shell'),
}

WEAK_KINDS = "('KIND_FINALIZER_REFERENCE', 'KIND_PHANTOM_REFERENCE', 'KIND_WEAK_REFERENCE')"

# TP SQL ---------------------------------------------------------------------
TP_EDGES = """
INCLUDE PERFETTO MODULE android.memory.heap_graph.excluded_refs;
INCLUDE PERFETTO MODULE graphs.search;
INCLUDE PERFETTO MODULE graphs.dominator_tree;
CREATE PERFETTO TABLE _pf_e AS
SELECT ref.owner_id AS src, ref.owned_id AS dst
FROM heap_graph_reference AS ref
JOIN heap_graph_object AS o ON ref.owner_id = o.id
WHERE o.reachable AND ref.id NOT IN _excluded_refs AND ref.owned_id IS NOT NULL;
"""
TP_DFS = """SELECT node_id AS id, parent_node_id AS parent_id FROM graph_reachable_dfs!(
  (SELECT src AS source_node_id, dst AS dest_node_id FROM _pf_e ORDER BY src, dst),
  (SELECT id AS node_id FROM heap_graph_object WHERE root_type IS NOT NULL ORDER BY id DESC))"""
TP_BFS = """SELECT node_id AS id, parent_node_id AS parent_id FROM graph_reachable_bfs!(
  (SELECT src AS source_node_id, dst AS dest_node_id FROM _pf_e ORDER BY src, dst),
  (SELECT id AS node_id FROM heap_graph_object WHERE root_type IS NOT NULL ORDER BY id))"""
TP_DOM_RAW = """SELECT node_id AS id, dominator_node_id AS parent_id FROM graph_dominator_tree!(
  (SELECT src AS source_node_id, dst AS dest_node_id FROM _pf_e
   UNION ALL
   SELECT (SELECT max(id) + 1 FROM heap_graph_object), id
   FROM heap_graph_object WHERE root_type IS NOT NULL),
  (SELECT max(id) + 1 FROM heap_graph_object))"""
TP_DOM_STDLIB = """INCLUDE PERFETTO MODULE android.memory.heap_graph.dominator_tree;
SELECT count(*) AS n, sum(dominated_size_bytes) AS s, sum(depth) AS d
FROM heap_graph_dominator_tree;"""
TP_OBJECT_TREE = """INCLUDE PERFETTO MODULE android.memory.heap_graph.object_tree;
SELECT count(*) AS n, sum(cumulative_size) AS s FROM _heap_graph_object_tree_aggregation;"""

TP_OPS = {
    'dom_retained':
        TP_DOM_STDLIB,
    'object_tree':
        TP_OBJECT_TREE,
    'dfs':
        TP_EDGES + f'SELECT count(*), sum(parent_id) FROM ({TP_DFS});',
    'bfs':
        TP_EDGES + f'SELECT count(*), sum(parent_id) FROM ({TP_BFS});',
    'dom_raw':
        TP_EDGES + f'SELECT count(*), sum(parent_id) FROM ({TP_DOM_RAW});',
    'edges_only':
        TP_EDGES + 'SELECT count(*) FROM _pf_e;',
    'load':
        'SELECT 1;',
}


def traces():
  with open(os.path.join(DATA, 'traces.txt')) as f:
    out = []
    for line in f:
      idx, path = line.split()
      out.append((int(idx), path))
    return out


def sh(cmd, **kw):
  return subprocess.run(cmd, check=True, **kw)


def run_tp(binary, trace, sql, out_csv=None):
  """Runs one TP process; returns (load_s, query_s, wall_s, stdout)."""
  with tempfile.TemporaryDirectory() as td:
    qf = os.path.join(td, 'q.sql')
    pf = os.path.join(td, 'perf.txt')
    with open(qf, 'w') as f:
      f.write(sql)
    t0 = time.time()
    stdout = open(out_csv, 'w') if out_csv else subprocess.PIPE
    p = subprocess.run([BINS[binary], '-q', qf, '--perf-file', pf, trace],
                       stdout=stdout,
                       stderr=subprocess.PIPE,
                       text=True)
    wall = time.time() - t0
    if out_csv:
      stdout.close()
    if p.returncode != 0:
      raise RuntimeError(f'TP failed on {trace}: {p.stderr[-2000:]}')
    with open(pf) as f:
      load_ns, query_ns = [int(x) for x in f.read().strip().split(',')[:2]]
    return load_ns / 1e9, query_ns / 1e9, wall, (
        p.stdout if not out_csv else None)


# scan-select ------------------------------------------------------------------
def cmd_scan_select(a):
  rows = []
  for f in a.scan:
    for line in open(f):
      parts = line.split()
      if len(parts) == 3 and parts[0].isdigit() and int(parts[0]) > 0:
        rows.append((int(parts[0]), int(parts[1]), parts[2]))
  rows = sorted(set(rows))
  n = min(a.n, len(rows))
  # Spread over the size range by rank, always incl. the largest by objects,
  # the largest by references and any --include traces.
  pick = {round(i * (len(rows) - 1) / max(n - 1, 1)) for i in range(n)}
  pick.add(max(range(len(rows)), key=lambda i: rows[i][1]))
  inc = {os.path.realpath(p) for p in (a.include or [])}
  pick |= {i for i, r in enumerate(rows) if os.path.realpath(r[2]) in inc}
  pick = sorted(pick)
  os.makedirs(DATA, exist_ok=True)
  with open(os.path.join(DATA, 'traces.txt'), 'w') as f:
    for k, i in enumerate(pick):
      f.write(f'{k} {rows[i][2]}\n')
      print(k, rows[i][0], rows[i][1], os.path.basename(rows[i][2]))


# extract ----------------------------------------------------------------------
KEEP = {
    '__intrinsic_heap_graph_object': ('object', [
        'id', 'upid', 'graph_sample_ts', 'self_size', 'native_size',
        'reference_set_id', 'reachable', 'type_id', 'root_type'
    ]),
    '__intrinsic_heap_graph_reference':
        ('reference',
         ['id', 'reference_set_id', 'owner_id', 'owned_id', 'field_name']),
    '__intrinsic_heap_graph_class': ('class', ['id', 'name', 'kind']),
}


def extract_one(item):
  import pyarrow as pa
  import pyarrow.ipc as ipc
  import pyarrow.parquet as pq
  idx, path = item
  t0 = time.time()
  with tempfile.TemporaryDirectory(dir=DATA) as td:
    tarp = os.path.join(td, 'x.tar')
    sh([BINS['main'], 'export', '--quiet', 'arrow_tar', '-o', tarp, path],
       stdout=subprocess.DEVNULL,
       stderr=subprocess.DEVNULL)
    t1 = time.time()
    with tarfile.open(tarp) as tf:
      for m in tf:
        base = os.path.basename(m.name)[:-len('.arrow')]
        if base not in KEEP:
          continue
        name, cols = KEEP[base]
        t = ipc.open_file(pa.BufferReader(tf.extractfile(m).read())).read_all()
        t = t.select(cols)
        arrs = []
        for i, fld in enumerate(t.schema):
          c = t.column(i)
          if pa.types.is_dictionary(fld.type):
            c = c.cast(pa.string())
          arrs.append(c)
        t = pa.table(arrs, names=cols)
        t = t.add_column(0, 'trace_idx', pa.array([idx] * t.num_rows,
                                                  pa.int32()))
        d = os.path.join(DATA, 'raw', name)
        os.makedirs(d, exist_ok=True)
        pq.write_table(
            t,
            os.path.join(d, f't{idx:03d}.parquet'),
            compression='zstd',
            row_group_size=1 << 20)
  print(
      f'extract {idx}: export {t1 - t0:.1f}s, total {time.time() - t0:.1f}s',
      flush=True)


def cmd_extract(a):
  with ThreadPoolExecutor(a.par) as ex:
    list(ex.map(extract_one, traces()))


# oracle -----------------------------------------------------------------------
def oracle_one(item):
  import duckdb
  idx, path = item
  out = os.path.join(DATA, 'oracle_tmp')
  os.makedirs(out, exist_ok=True)
  for name, sql in [
      ('dfs', TP_EDGES + TP_DFS + ';'),
      ('bfs', TP_EDGES + TP_BFS + ';'),
      ('dom',
       'INCLUDE PERFETTO MODULE android.memory.heap_graph.dominator_tree;\n'
       'SELECT id, idom_id AS parent_id, depth - 1 AS depth, dominated_obj_count, '
       'dominated_size_bytes, dominated_native_size_bytes FROM heap_graph_dominator_tree;'
      ),
  ]:
    f = os.path.join(out, f'{name}_{idx:03d}.csv')
    run_tp('main', path, sql, out_csv=f)
    con = duckdb.connect()
    con.execute(
        f"""COPY (SELECT {idx}::INT AS trace_idx, * FROM read_csv('{f}', header=true,
                    nullstr='[NULL]', skip=0)) TO '{out}/{name}_{idx:03d}.parquet' (FORMAT parquet)"""
    )
    con.close()
    os.remove(f)
  print(f'oracle {idx} done', flush=True)


def cmd_oracle(a):
  with ThreadPoolExecutor(a.par) as ex:
    list(ex.map(oracle_one, traces()))
  import duckdb
  con = duckdb.connect()
  for name in ['dfs', 'bfs', 'dom']:
    con.execute(f"COPY (SELECT * FROM '{DATA}/oracle_tmp/{name}_*.parquet') TO "
                f"'{DATA}/tp_{name}.parquet' (FORMAT parquet)")


# DuckDB inputs ----------------------------------------------------------------
def duck_views(con, src='parquet'):
  """Defines the heap graph inputs exactly like TP's stdlib does."""
  raw = os.path.join(DATA, 'raw')
  if src == 'parquet':
    for t in ['object', 'reference', 'class']:
      con.execute(
          f"CREATE OR REPLACE VIEW hg_{t} AS SELECT * FROM '{raw}/{t}/*.parquet'"
      )
  con.execute(f"""
    CREATE OR REPLACE VIEW heap_excluded AS
    WITH wc AS (SELECT trace_idx, id AS type_id FROM hg_class WHERE kind IN {WEAK_KINDS}),
    wo AS (SELECT DISTINCT o.trace_idx, o.reference_set_id FROM hg_object o
           JOIN wc USING (trace_idx, type_id) WHERE o.reference_set_id IS NOT NULL),
    rr AS (SELECT trace_idx, id, reference_set_id FROM hg_reference
           WHERE field_name = 'java.lang.ref.Reference.referent')
    SELECT rr.trace_idx, rr.id FROM rr JOIN wo USING (trace_idx, reference_set_id)"""
             )
  con.execute("""
    CREATE OR REPLACE VIEW heap_object AS
    SELECT trace_idx, upid, id, self_size, native_size, 1 AS self_count FROM hg_object"""
             )
  con.execute("""
    CREATE OR REPLACE VIEW heap_edge AS
    SELECT o.trace_idx, o.upid, ref.owner_id AS src, ref.owned_id AS dst
    FROM hg_reference ref
    JOIN hg_object o ON o.trace_idx = ref.trace_idx AND o.id = ref.owner_id
    WHERE o.reachable != 0 AND ref.owned_id IS NOT NULL
      AND NOT EXISTS (SELECT 1 FROM heap_excluded x
                      WHERE x.trace_idx = ref.trace_idx AND x.id = ref.id)""")
  # object_tree's edge set: no reachable-owner filter.
  con.execute("""
    CREATE OR REPLACE VIEW heap_edge_all AS
    SELECT o.trace_idx, o.upid, ref.owner_id AS src, ref.owned_id AS dst
    FROM hg_reference ref
    JOIN hg_object o ON o.trace_idx = ref.trace_idx AND o.id = ref.owner_id
    WHERE ref.owned_id IS NOT NULL
      AND NOT EXISTS (SELECT 1 FROM heap_excluded x
                      WHERE x.trace_idx = ref.trace_idx AND x.id = ref.id)""")
  con.execute("""
    CREATE OR REPLACE VIEW heap_root AS
    SELECT trace_idx, upid, id FROM hg_object WHERE root_type IS NOT NULL""")


def has_macro(con, name):
  return con.execute(
      "SELECT count(*) FROM duckdb_functions() WHERE function_name = ?",
      [name]).fetchone()[0] > 0


DUCK_PER = "per := ['trace_idx', 'upid']"
DUCK_DB = os.path.join(DATA, 'bench.duckdb')
INPUTS = ['heap_object', 'heap_edge', 'heap_edge_all', 'heap_root']


def duck_pipelines(con):
  """name -> list of SQL statements; the last one's result is checksummed."""
  acc = has_macro(con, 'pf_tree_accumulate_up')
  p = {}
  p['edges_only'] = ["SELECT count(*), sum(dst) FROM heap_edge"]
  for op in ['dfs', 'bfs', 'dom_raw']:
    fn = {
        'dfs': 'pf_graph_dfs_tree',
        'bfs': 'pf_graph_bfs_tree',
        'dom_raw': 'pf_graph_dominator_tree'
    }[op]
    p[op] = [
        f"SELECT count(*), sum(parent_id), sum(depth), sum(self_size) FROM "
        f"{fn}('heap_object', 'heap_edge', 'heap_root', {DUCK_PER})"
    ]
  if acc:
    p['dom_retained'] = [
        f"SELECT count(*), sum(total) FROM pf_tree_accumulate_up("
        f"\"pf_graph_dominator_tree('heap_object', 'heap_edge', 'heap_root', {DUCK_PER})\", "
        f"'self_size', {DUCK_PER})"
    ]
    p['object_tree'] = [
        # Objects not in the shortest-path tree are their own roots (TP: LEFT JOIN).
        f"SELECT count(*), sum(total) FROM pf_tree_accumulate_up("
        f"\"(SELECT o.trace_idx, o.upid, o.id, b.parent_id, o.self_size "
        f"FROM heap_object o LEFT JOIN pf_graph_bfs_tree('heap_object', 'heap_edge_all', 'heap_root', {DUCK_PER}) b "
        f"USING (trace_idx, upid, id))\", 'self_size', {DUCK_PER})"
    ]
  return p


def duck_connect(threads):
  import duckdb
  con = duckdb.connect(DUCK_DB, config={'allow_unsigned_extensions': 'true'})
  con.execute(f"LOAD '{EXT}'")
  con.execute(f"SET memory_limit='{os.environ.get('DUCK_MEM', '16GB')}'")
  con.execute(f'SET threads={threads}')
  con.execute(f"SET temp_directory='{DATA}/duck_tmp'")
  con.execute('SET preserve_insertion_order=false')
  return con


def cmd_duck_prep(a):
  """Materializes the graph inputs (objects, filtered edges, roots) into the
  file-backed DuckDB database, like TP after trace load."""
  con = duck_connect(a.threads)
  duck_views(con, src='parquet')
  res = {}
  for v in INPUTS:
    t0 = time.time()
    con.execute(f"CREATE OR REPLACE TABLE m_{v} AS SELECT * FROM {v}")
    res[v] = time.time() - t0
    print(
        v,
        f'{res[v]:.1f}s',
        con.execute(f'SELECT count(*) FROM m_{v}').fetchone()[0],
        flush=True)
  con.execute('CHECKPOINT')
  os.makedirs(os.path.join(DATA, 'results'), exist_ok=True)
  json.dump(
      dict(engine='duckdb_prep', threads=a.threads, seconds=res),
      open(os.path.join(DATA, 'results', 'duckprep.json'), 'w'))


def cmd_duck(a):
  res = []
  for threads in [int(x) for x in a.threads.split(',')]:
    for mode in a.modes.split(','):
      con = duck_connect(threads)
      duck_views(con, src='parquet')
      if mode == 'load_excluded':
        # Inputs from DuckDB native storage (built by duck-prep).
        for v in INPUTS:
          con.execute(f"CREATE OR REPLACE VIEW {v} AS SELECT * FROM m_{v}")
      # else: everything (incl. the excluded-refs edge filter) from raw Parquet.
      pipes = duck_pipelines(con)
      ops = a.ops.split(',') if a.ops else list(pipes)
      for op in ops:
        if op not in pipes:
          print(f'skip {op} (needs pf_tree_accumulate_up)')
          continue
        best = None
        for rep in range(a.reps):
          t0 = time.time()
          for s in pipes[op][:-1]:
            con.execute(s)
          chk = con.execute(pipes[op][-1]).fetchone()
          dt = time.time() - t0
          best = dt if best is None else min(best, dt)
        r = dict(
            engine='duckdb',
            threads=threads,
            mode=mode,
            op=op,
            seconds=best,
            check=[str(x) for x in chk])
        print(json.dumps(r), flush=True)
        res.append(r)
      con.close()
  os.makedirs(os.path.join(DATA, 'results'), exist_ok=True)
  with open(os.path.join(DATA, 'results', f'duck_{a.tag}.json'), 'w') as f:
    json.dump(res, f, indent=1)


def cmd_scale(a):
  """Many independent random graphs at once (DuckDB only; TP's stdlib has no
  partitioned form): G graphs x N nodes, ~3 out-edges/node, 1% seeds."""
  import duckdb
  res = []
  for g, n in [
      (int(x.split('x')[0]), int(x.split('x')[1])) for x in a.shapes.split(',')
  ]:
    con = duckdb.connect(config={'allow_unsigned_extensions': 'true'})
    con.execute(f"LOAD '{EXT}'")
    con.execute(f"SET memory_limit='{os.environ.get('DUCK_MEM', '16GB')}'")
    con.execute(f'SET threads={a.threads}')
    con.execute('SET preserve_insertion_order=false')
    con.execute(
        f"""CREATE TABLE sn AS SELECT (range // {n})::INT AS g, range % {n} AS id,
                    (hash(range) % 1000)::BIGINT AS self_size FROM range({g * n})"""
    )
    # Chain-ish backbone + random forward/backward edges (cycles, diamonds).
    con.execute(f"""CREATE TABLE se AS
        SELECT g, id AS src, (id + 1) % {n} AS dst FROM sn
        UNION ALL SELECT g, id, hash(id, g, 1) % {n} FROM sn
        UNION ALL SELECT g, id, hash(id, g, 2) % {n} FROM sn""")
    con.execute(
        f"CREATE TABLE ss AS SELECT g, id FROM sn WHERE hash(id, g, 3) % 100 = 0 OR id = 0"
    )
    for op in ['dfs', 'bfs', 'dominator']:
      t0 = time.time()
      chk = con.execute(
          f"SELECT count(*), sum(parent_id) FROM pf_graph_{op}_tree('sn', 'se', 'ss', per := ['g'])"
      ).fetchone()
      dt = time.time() - t0
      r = dict(
          engine='duckdb_scale',
          graphs=g,
          nodes_per_graph=n,
          edges=3 * g * n,
          threads=a.threads,
          op=op,
          seconds=dt,
          check=[str(x) for x in chk])
      print(json.dumps(r), flush=True)
      res.append(r)
    con.close()
  os.makedirs(os.path.join(DATA, 'results'), exist_ok=True)
  with open(os.path.join(DATA, 'results', f'scale_t{a.threads}.json'),
            'w') as f:
    json.dump(res, f, indent=1)


def cmd_tp(a):
  items = traces()
  ops = a.ops.split(',')
  res = []
  for op in ops:
    sql = TP_OPS[op]
    t0 = time.time()

    def one(item):
      idx, path = item
      try:
        load, query, wall, out = run_tp(a.bin, path, sql)
        return dict(
            trace_idx=idx,
            load=load,
            query=query,
            wall=wall,
            out=(out or '').strip().splitlines()[-1:])
      except Exception as e:  # noqa
        return dict(trace_idx=idx, error=str(e)[-500:])

    with ThreadPoolExecutor(a.par) as ex:
      per = list(ex.map(one, items))
    batch_wall = time.time() - t0
    ok = [r for r in per if 'error' not in r]
    r = dict(
        engine=f'tp_{a.bin}',
        par=a.par,
        op=op,
        batch_wall=batch_wall,
        sum_load=sum(x['load'] for x in ok),
        sum_query=sum(x['query'] for x in ok),
        sum_wall=sum(x['wall'] for x in ok),
        n_ok=len(ok),
        n=len(per),
        per_trace=per)
    print(
        json.dumps({
            k: v for k, v in r.items() if k != 'per_trace'
        }),
        flush=True)
    for x in per:
      if 'error' in x:
        print('  error', x['trace_idx'], x['error'][-300:])
    res.append(r)
  os.makedirs(os.path.join(DATA, 'results'), exist_ok=True)
  with open(
      os.path.join(DATA, 'results', f'tp_{a.bin}_p{a.par}_{a.tag}.json'),
      'w') as f:
    json.dump(res, f, indent=1)


def cmd_report(a):
  tp, duck = {}, {}
  for f in sorted(glob.glob(os.path.join(DATA, 'results', '*.json'))):
    data = json.load(open(f))
    if not isinstance(data, list):
      continue
    for r in data:
      if r['engine'].startswith('tp_'):
        tp[(r['engine'], r['par'], r['op'])] = r
      elif r['engine'] == 'duckdb':
        duck[(r['threads'], r['mode'], r['op'])] = r['seconds']
  ops = [
      'dom_retained', 'object_tree', 'dfs', 'bfs', 'dom_raw', 'edges_only',
      'load'
  ]
  cols = []
  for eng in ['tp_main', 'tp_legacy']:
    pars = sorted({k[1] for k in tp if k[0] == eng})
    for p in pars:
      if p == 1:
        cols.append((f'{eng} seq query', lambda op, e=eng: tp.get(
            (e, 1, op), {}).get('sum_query')))
        cols.append((f'{eng} seq wall', lambda op, e=eng: tp.get(
            (e, 1, op), {}).get('sum_wall')))
      else:
        cols.append((f'{eng} P={p} wall', lambda op, e=eng, p=p: tp.get(
            (e, p, op), {}).get('batch_wall')))
  for th in sorted({k[0] for k in duck}):
    cols.append((f'duck {th}t (tables)', lambda op, th=th: duck.get(
        (th, 'load_excluded', op))))
    cols.append((f'duck {th}t (parquet)', lambda op, th=th: duck.get(
        (th, 'load_included', op))))
  print('| op | ' + ' | '.join(c[0] for c in cols) + ' |')
  print('|---' * (len(cols) + 1) + '|')
  for op in ops:
    vals = []
    for _, fn in cols:
      v = fn(op)
      vals.append('' if v is None else f'{v:.2f}')
    print(f'| {op} | ' + ' | '.join(vals) + ' |')


def main():
  ap = argparse.ArgumentParser()
  sp = ap.add_subparsers(dest='cmd', required=True)
  s = sp.add_parser('scan-select')
  s.add_argument('--scan', nargs='+', required=True)
  s.add_argument('--n', type=int, default=20)
  s.add_argument('--include', nargs='*')
  s = sp.add_parser('extract')
  s.add_argument('--par', type=int, default=4)
  s = sp.add_parser('oracle')
  s.add_argument('--par', type=int, default=4)
  s = sp.add_parser('tp')
  s.add_argument('--bin', default='main', choices=list(BINS))
  s.add_argument('--par', type=int, default=1)
  s.add_argument(
      '--ops', default='load,dom_retained,object_tree,dfs,bfs,dom_raw')
  s.add_argument('--tag', default='run')
  s = sp.add_parser('duck-prep')
  s.add_argument('--threads', type=int, default=48)
  s = sp.add_parser('duck')
  s.add_argument('--threads', default='1,48')
  s.add_argument('--modes', default='load_excluded,load_included')
  s.add_argument('--ops', default='')
  s.add_argument('--reps', type=int, default=2)
  s.add_argument('--tag', default='run')
  s = sp.add_parser('scale')
  s.add_argument('--threads', type=int, default=48)
  s.add_argument(
      '--shapes',
      default='100000x100,10000x1000,1000x10000,100x100000,10x1000000')
  sp.add_parser('report')
  a = ap.parse_args()
  {
      'scan-select': cmd_scan_select,
      'extract': cmd_extract,
      'oracle': cmd_oracle,
      'tp': cmd_tp,
      'duck': cmd_duck,
      'duck-prep': cmd_duck_prep,
      'scale': cmd_scale,
      'report': cmd_report
  }[a.cmd](
      a)


if __name__ == '__main__':
  main()
