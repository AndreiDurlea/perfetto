# Graph operators (spec §7) in DuckDB

`GRAPH ( DFS | BFS | DOMINATOR ) TREE NODES n EDGES e FROM s` turns a directed
graph into a tree relation (`id`, `parent_id`, `depth` + the node payload) that
composes with the tree operators (§6). Native kernel: `pf_graph_tree_agg` in
[`src/graphs.cc`](../src/graphs.cc); SQL surface in
[`sql/graphs.sql`](../sql/graphs.sql).

```python
import duckdb
con = duckdb.connect(config={'allow_unsigned_extensions': 'true'})
con.execute("LOAD '/usr/local/google/home/zezeozue/bt_bench/perfetto_duckdb/perfetto.duckdb_extension'")
```

## Operator -> DuckDB

| PerfettoSQL Next | DuckDB |
|---|---|
| `GRAPH DFS TREE NODES n EDGES e FROM s` | `SELECT * FROM pf_graph_dfs_tree('n', 'e', 's' [, per := [...]])` |
| `GRAPH BFS TREE NODES n EDGES e FROM s` | `SELECT * FROM pf_graph_bfs_tree('n', 'e', 's' [, per := [...]])` |
| `GRAPH DOMINATOR TREE NODES n EDGES e FROM s` | `SELECT * FROM pf_graph_dominator_tree('n', 'e', 's' [, per := [...]])` |

Arguments are table / view names (DuckDB `query_table`):

* `n` (NODES): column `id` + any payload columns.
* `e` (EDGES): columns `src`, `dst`.
* `s` (FROM): column `id` (seeds / roots).
* `per`: partition columns present in all three relations, e.g.
  `['trace_idx', 'upid']`. Every partition is an independent graph; they are
  solved in parallel (DuckDB's hash aggregate partitions the input, groups are
  solved concurrently). Omit for a single graph.

Output columns: the `per` columns, `id`, `parent_id` (NULL at roots), `depth`
(0 at roots), then the node payload. A payload column called `parent_id` or
`depth` is replaced by the tree's. Row order is unspecified. Columns starting
with `_pf_` are reserved.

## Semantics

Fidelity target: TP main does not implement the `GRAPH ... TREE` pipe
operators yet, so semantics follow the spec and TP's legacy stdlib
(`graph_reachable_dfs`, `graph_reachable_bfs`, `graph_dominator_tree`,
`android.memory.heap_graph.dominator_tree`), and are verified against them on
real heap dumps (see Correctness).

* **Vertex set** = distinct non-NULL `n.id`. Edges with a NULL endpoint or an
  endpoint that is not a node are ignored; seeds that are not nodes are
  ignored. Duplicate edges, duplicate seeds and self loops are harmless. Node
  ids must be unique per partition (they are the payload join key; duplicate
  node rows duplicate output rows, like any join).
* **Only nodes reachable from the seeds are returned**; DFS, BFS and DOMINATOR
  return the same node set.
* **Determinism / ties by node id**: successors are visited in ascending `dst`
  id and seeds in ascending id. This equals TP's `graph_reachable_dfs` /
  `graph_reachable_bfs` when TP is fed edges ordered by `(src, dst)` and starts
  ordered by id (DESC for DFS, which uses a stack; ASC for BFS).
* **DFS**: depth-first forest: for each seed in ascending id order that is not
  yet visited, a (recursive-order) DFS rooted at it. A seed reached from an
  earlier seed is an ordinary node with a parent (as in TP).
* **BFS**: every seed is a root (depth 0); each other node's parent is the
  first node, in BFS order, with an edge to it: a shortest-hop tree, `depth` =
  hop distance from the nearest seed.
* **DOMINATOR**: immediate-dominator tree (Lengauer-Tarjan with path
  compression, O(m log n)). One seed: it is the root. Several seeds: a virtual
  super-root with an edge to every seed is inserted and stripped, so nodes
  immediately dominated by it (the seeds and nodes reachable from several
  seeds) are roots with `parent_id` NULL, exactly like
  `heap_graph_dominator_tree.idom_id`. `depth` = `heap_graph_dominator_tree.depth - 1`.
* Empty inputs, partitions without seeds or nodes produce no rows.

## Example: Java heap retained size (spec §7 example)

```
GRAPH DOMINATOR TREE NODES heap_objects EDGES heap_refs FROM gc_roots
|> TREE ACCUMULATE UP SUM(self_size) AS retained;
```

DuckDB, over heap graphs extracted from many traces at once (raw TP tables
`heap_graph_object`, `heap_graph_reference`, `heap_graph_class` with a
`trace_idx` column; the edge filter is the one used by TP's
`android.memory.heap_graph.raw_dominator_tree`):

```sql
CREATE VIEW heap_objects AS
SELECT trace_idx, upid, id, self_size, native_size FROM heap_graph_object;

CREATE VIEW heap_refs AS
SELECT o.trace_idx, o.upid, r.owner_id AS src, r.owned_id AS dst
FROM heap_graph_reference r
JOIN heap_graph_object o ON o.trace_idx = r.trace_idx AND o.id = r.owner_id
WHERE o.reachable != 0 AND r.owned_id IS NOT NULL
  AND NOT EXISTS (  -- android.memory.heap_graph.excluded_refs
    SELECT 1 FROM heap_graph_object ro
    JOIN heap_graph_class c ON c.trace_idx = ro.trace_idx AND c.id = ro.type_id
    WHERE ro.trace_idx = r.trace_idx AND ro.reference_set_id = r.reference_set_id
      AND c.kind IN ('KIND_FINALIZER_REFERENCE', 'KIND_PHANTOM_REFERENCE', 'KIND_WEAK_REFERENCE')
      AND r.field_name = 'java.lang.ref.Reference.referent');

CREATE VIEW gc_roots AS
SELECT trace_idx, upid, id FROM heap_graph_object WHERE root_type IS NOT NULL;

-- GRAPH DOMINATOR TREE NODES heap_objects EDGES heap_refs FROM gc_roots
CREATE TABLE dom AS
SELECT * FROM pf_graph_dominator_tree('heap_objects', 'heap_refs', 'gc_roots',
                                      per := ['trace_idx', 'upid']);

-- |> TREE ACCUMULATE UP SUM(self_size) AS retained   (trees family)
SELECT * REPLACE (total AS retained)  -- or: total AS retained
FROM pf_tree_accumulate_up('dom', 'self_size', per := ['trace_idx', 'upid']);
```

`retained` equals TP's `heap_graph_dominator_tree.dominated_size_bytes` per
object and `parent_id` equals its `idom_id`.

Similarly, the RFC-0041 §6.5 heap class tree pipeline (`GRAPH BFS TREE |> TREE FLATTEN RECURSION BY class_name |> TREE KEY SIBLINGS BY class_name, heap_type AS (node_id, parent_node_id)`) maps directly to:

```sql
CREATE TABLE bfs_objects AS
SELECT * FROM pf_graph_bfs_tree('heap_objects', 'heap_refs', 'gc_roots',
                                per := ['trace_idx', 'upid']);

CREATE TABLE norec_objects AS
SELECT * FROM pf_tree_flatten_recursion('bfs_objects', ['class_name'],
                                        per := ['trace_idx', 'upid']);

CREATE TABLE heap_object_nodes AS
SELECT * FROM pf_tree_key_siblings('norec_objects', ['class_name', 'heap_type'],
                                   key_col := 'node_id', parent_key_col := 'parent_node_id',
                                   per := ['trace_idx', 'upid']);
```

## Direct kernel use

```sql
pf_graph_tree_agg(kind INTEGER, role INTEGER, a BIGINT, b BIGINT)
  -> STRUCT(id BIGINT, parent_id BIGINT, depth BIGINT)[]
-- kind: 0 DFS, 1 BFS, 2 DOMINATOR
-- role: 0 node (a = id), 1 edge (a = src, b = dst), 2 seed (a = id)
SELECT trace_idx, unnest(pf_graph_tree_agg(2, role, a, b), recursive := true)
FROM (SELECT trace_idx, 0 AS role, id AS a, NULL::BIGINT AS b FROM n
      UNION ALL SELECT trace_idx, 1, src, dst FROM e
      UNION ALL SELECT trace_idx, 2, id, NULL FROM s)
GROUP BY trace_idx;
```

## Implementation notes

* One aggregate state per graph collects node ids, edges and seeds. Finalize
  maps ids to dense ranks (direct table when the id range is <= 4n, else sort
  + open addressing), builds predecessor CSR by bucketing on dst and successor
  CSR by walking dst in ascending order, which yields successor lists sorted by
  dst id with no comparison sort (O(n + m)).
* DFS uses an explicit stack of edge cursors (recursive order, O(depth)
  memory); BFS a flat queue; Lengauer-Tarjan runs in DFS-preorder space with
  iterative path compression (no recursion, safe on million-deep chains).
* Large chunks of groups are solved with up to `PF_GRAPH_THREADS` (default 48)
  threads, since DuckDB finalizes a GROUP BY with few huge groups on one
  thread.
* Memory: the kernel's buffers are not tracked by DuckDB's `memory_limit`;
  budget roughly 40 B per node + 30 B per edge on top of DuckDB.

## Correctness

`python tests/test_graphs.py` (exits non-zero on failure):

* Independent brute-force Python reference (TP-style push-all-children DFS,
  textbook BFS, and dominators by definition: d dom v iff v is unreachable
  without d) on chains, diamonds, cycles, self loops, duplicate edges/seeds,
  unreachable nodes, multiple roots, seeds reaching seeds, the Lengauer-Tarjan
  paper graph, dangling / NULL edges, seeds not in NODES, dense / offset /
  sparse 63-bit / negative ids, 30 partitions at once and 150 random graphs.
* TP oracle on real Java heap dumps (when `bench/bench_graphs.py extract` and
  `oracle` have been run; data dir `GRAPH_ORACLE_DIR`, default
  `~/bt_bench/graphs_data`): exact row-level equality of (id, parent_id) with
  `graph_reachable_dfs` / `graph_reachable_bfs`, of (id, parent_id, depth)
  with `heap_graph_dominator_tree`, and retained size vs
  `dominated_size_bytes` once `pf_tree_accumulate_up` is available.

## Benchmarks (22 Android Java heap dumps: 126.5M objects, 231.7M edges, 117.6M reachable objects)

| Operator | Output Rows | PerfettoSQL / TP (1T) | DuckDB 1T | 1T Speedup (%) | DuckDB 48T | 48T Speedup vs TP 1T (%) | 48T Speedup vs TP 22-proc Wall (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `GRAPH DFS TREE` | 117,637,905 | 137.12 s | **16.80 s** | **8.16× (+716%)** | **3.65 s** | **37.57× (+3,657%)** | **9.24× (+824%)** |
| `GRAPH BFS TREE` | 117,637,905 | 140.09 s | **14.90 s** | **9.40× (+840%)** | **3.32 s** | **42.20× (+4,120%)** | **10.10× (+910%)** |
| `GRAPH DOMINATOR TREE` (Lengauer-Tarjan) | 117,637,905 | 171.78 s | **21.40 s** | **8.03× (+703%)** | **4.41 s** | **38.95× (+3,795%)** | **8.15× (+715%)** |
| `GRAPH DOMINATOR TREE` + `TREE ACCUMULATE UP` (Retained Size) | 117,637,905 | 288.87 s | **27.90 s** | **10.35× (+935%)** | **6.12 s** | **47.20× (+4,620%)** | **8.88× (+788%)** |
| `GRAPH BFS TREE` + `TREE ACCUMULATE UP` | 117,637,905 | 543.69 s | **21.30 s** | **25.53× (+2,453%)** | **5.04 s** | **107.88× (+10,688%)** | **15.86× (+1,486%)** |

