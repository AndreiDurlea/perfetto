# Tree operators (spec §6) in DuckDB

The tree family of the `perfetto` DuckDB extension. It implements every operator in §6 of [`docs/spec.md`](spec.md). Native C++ kernels live in [`src/trees.cc`](../src/trees.cc) and SQL table macros in [`sql/trees.sql`](../sql/trees.sql).

```python
import duckdb
con = duckdb.connect(config={'allow_unsigned_extensions': 'true'})
con.execute("LOAD './perfetto.duckdb_extension'")
```

## Conventions

- Every operator is a table macro taking `tbl` as a table/view name or parenthesized subquery string `'(SELECT ...)'` with columns `id`, `parent_id` (`NULL` = root), and partition columns `per := ['trace_idx']`.
- Each partition in `per` is an independent forest solved in parallel.
- Aggregate specifications are SQL strings `'expr AS name'`, e.g. `['sum(dur) AS total', 'count(*) AS n']`.

## Operators

| PerfettoSQL Next | DuckDB Macro |
|---|---|
| `TREE ACCUMULATE UP SUM(col) AS total` | `pf_tree_accumulate_up(tbl, 'col', per := [...])` |
| `TREE ACCUMULATE DOWN SUM(col) AS total` | `pf_tree_accumulate_down(tbl, 'col', per := [...])` |
| `TREE ACCUMULATE (UP\|DOWN) aggs` | `pf_tree_accumulate(tbl, 'up'\|'down', ['sum(v) AS s', ...], per := [...])` |
| `TREE KEEP IF (ANCESTOR\|DESCENDANT) OF marker [OVER over]` | `pf_tree_keep_if(tbl, 'ancestor'\|'descendant', marker, per := [...], over := NULL)` |
| `TREE DROP IF (ANCESTOR\|DESCENDANT) OF marker [OVER over]` | `pf_tree_drop_if(tbl, 'ancestor'\|'descendant', marker, per := [...], over := NULL)` |
| `TREE PRUNE AT marker [OVER over]` | `pf_tree_prune(tbl, marker, per := [...], over := NULL)` |
| `TREE REPARENT TO NEAREST ANCESTOR IN marker [OVER over]` | `pf_tree_reparent(tbl, marker, per := [...], over := NULL)` |
| `TREE FLATTEN RECURSION BY keys` | `pf_tree_flatten_recursion(tbl, keys, per := [...])` |
| `TREE KEY SIBLINGS BY keys AS (k, pk)` | `pf_tree_key_siblings(tbl, keys, key_col := 'node_id', parent_key_col := 'parent_node_id', per := [...], ordered := false)` |
| `TREE CONTRACT AT marker [OVER over] AGGREGATE aggs` | `pf_tree_contract(tbl, marker, aggs, per := [...], over := NULL, full_row := false)` |
| `TREE ABSORB AT marker [OVER over] AGGREGATE aggs` | `pf_tree_absorb(tbl, marker, aggs, per := [...], over := NULL, full_row := false)` |
| `TREE MERGE SIBLINGS BY keys [ORDERED] AGGREGATE aggs` | `pf_tree_merge_siblings(tbl, keys, aggs, per := [...], ordered := false)` |
| `TREE MERGE INTO PARENT BY keys [INDIRECT] AGGREGATE aggs` | `pf_tree_merge_into_parent(tbl, keys, aggs, per := [...], indirect := false)` |
| `TREE INVERT` | `pf_tree_invert(tbl, per := [...])` |
| `TREE EXPAND (UP\|DOWN) operand BY keys ...` | `pf_tree_expand(tbl, 'up'\|'down', operand, keys, ordered_by := [...], charge := [...], charge_to := NULL, per := [...])` |

## Implementation Notes

- **Zero-hash dense path**: Slice IDs per trace are dense (`0 .. n-1`). `Tree::Build` detects sorted dense IDs in O(n) and resolves parent pointers by direct array indexing without sorting or hash tables, falling back to `std::lower_bound` when IDs are sparse or filtered.
- **Direct C++ folding for `CONTRACT` and `ABSORB`**: When `over IS NULL`, `full_row := false` (default), and `aggs` is `['sum(col) AS ...']` or `['sum(col) AS ...', 'count(*) AS ...']`, `pf_tree_contract_sum_agg` and `pf_tree_absorb_sum_agg` compute the nearest surviving/marker ancestor and fold `SUM` + `COUNT(*)` in a single top-down array pass in C++, avoiding a 20M-row SQL `HASH_JOIN` + `HASH_GROUP_BY`. Pass `full_row := true` to retain non-key payload columns via the generic SQL group-by path.
- **Parallel finalize**: When DuckDB invokes `Finalize` with multiple partitions, `ParallelFor` distributes trees across `PF_TREE_THREADS` (or `PF_THREADS`, default hardware concurrency up to 32) worker threads.

## Benchmarks (100 Android traces, 19,645,364 slices)

| Operator | PerfettoSQL Next / TP (1T) | DuckDB 1T | 1T Speedup (%) | DuckDB 32T | 32T Speedup vs TP 1T (%) |
|---|---:|---:|---:|---:|---:|
| `TREE ACCUMULATE UP SUM(dur)` | 3.230 s | **0.760 s** | **4.25× (+325%)** | **0.396 s** | **8.16× (+716%)** |
| `TREE ACCUMULATE DOWN SUM(dur)` | 3.100 s | **0.606 s** | **5.12× (+412%)** | **0.337 s** | **9.20× (+820%)** |
| `TREE KEEP IF DESCENDANT OF` | 14.379 s | **1.908 s** | **7.54× (+654%)** | **0.832 s** | **17.28× (+1,628%)** |
| `TREE KEEP IF ANCESTOR OF` | 16.430 s | **1.630 s** | **10.08× (+908%)** | **0.812 s** | **20.23× (+1,923%)** |
| `TREE PRUNE AT` | 15.814 s | **1.894 s** | **8.35× (+735%)** | **0.202 s** | **78.29× (+7,729%)** |
| `TREE REPARENT TO NEAREST ANCESTOR` | 26.854 s | **3.688 s** | **7.28× (+628%)** | **0.560 s** | **47.95× (+4,695%)** |
| `TREE ABSORB AT ... AGGREGATE SUM(dur)` | 36.262 s | **2.010 s** | **18.04× (+1,704%)** | **1.451 s** | **24.99× (+2,399%)** |
| `TREE CONTRACT AT ... AGGREGATE SUM(dur)` | 13.562 s | **2.018 s** | **6.72× (+572%)** | **1.402 s** | **9.67× (+867%)** |
| `TREE MERGE SIBLINGS BY name` | 158.081 s | **4.425 s** | **35.72× (+3,472%)** | **1.485 s** | **106.45× (+10,545%)** |
| `TREE MERGE INTO PARENT BY name` | 158.659 s | **7.688 s** | **20.64× (+1,964%)** | **1.220 s** | **130.05× (+12,905%)** |
| `TREE INVERT` | 52.952 s | **3.634 s** | **14.57× (+1,357%)** | **1.075 s** | **49.26× (+4,826%)** |
