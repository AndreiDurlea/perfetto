# perfetto-duckdb

A DuckDB C-API extension (`perfetto.duckdb_extension`) implementing all **PerfettoSQL Next** analysis operators ([`docs/spec.md`](docs/spec.md): §5 Interval operators, §6 Tree operators, §7 Graph operators) over multi-trace tables.

Instead of running analysis queries inside one single-threaded SQLite/TraceProcessor instance per trace, `perfetto-duckdb` lets you extract raw tables once from Perfetto Trace Processor (`slice`, `sched`, `thread_state`, `counter`, `heap_graph_object`, `heap_graph_reference`, etc.) into DuckDB or Parquet with a `trace_idx` / `trace_uuid` column, and run interval, tree, and graph operators across hundreds or thousands of traces in a single vectorized query.

---

## Quick Start

### Build

Requires a C++17 compiler (`g++` or `clang++`) and Python 3. No external build system or DuckDB source checkout is required (the DuckDB C extension headers are vendored in [`src/`](src/)).

```bash
./build.sh
```

This compiles [`src/intervals.cc`](src/intervals.cc), [`src/trees.cc`](src/trees.cc), [`src/graphs.cc`](src/graphs.cc), and [`src/extension.cc`](src/extension.cc) alongside the embedded SQL macro definitions ([`sql/intervals.sql`](sql/intervals.sql), [`sql/trees.sql`](sql/trees.sql), [`sql/graphs.sql`](sql/graphs.sql)) into `perfetto.duckdb_extension`.

### Load in DuckDB

```python
import duckdb

con = duckdb.connect(config={'allow_unsigned_extensions': 'true'})
con.execute("LOAD './perfetto.duckdb_extension'")
```

---

## Operator Families

All operators are exposed as `pf_*` table macros backed by C++ aggregate kernels that partition work by `per := [...]` (e.g., `['trace_idx', 'utid']` or `['trace_idx']`) and run sort-sweep / CSR / Lengauer-Tarjan passes in parallel across partitions.

- **Intervals ([`docs/intervals.md`](docs/intervals.md))**:
  - Sources: `pf_interval_intersection`, `pf_interval_union`, `pf_intervals_from_events`
  - Reshaping: `pf_interval_flatten`, `pf_interval_flatten_sum`, `pf_interval_split`, `pf_interval_subtract`, `pf_interval_fill_gaps` (`pf_interval_fill_within`), `pf_interval_merge_overlapping`, `pf_interval_merge_consecutive`, `pf_interval_set_bounds`, `pf_interval_quantize`
  - Matching: `pf_interval_join`, `pf_interval_find`, `pf_interval_keep_if`, `pf_interval_drop_if`
- **Trees ([`docs/trees.md`](docs/trees.md))**:
  - Accumulation: `pf_tree_accumulate`, `pf_tree_accumulate_up`, `pf_tree_accumulate_down`
  - Structural filtering & reshaping: `pf_tree_keep_if`, `pf_tree_drop_if`, `pf_tree_prune`, `pf_tree_reparent`, `pf_tree_flatten_recursion`, `pf_tree_contract`, `pf_tree_absorb`
  - Callstack / flamegraph transforms: `pf_tree_key_siblings`, `pf_tree_merge_siblings`, `pf_tree_merge_into_parent`, `pf_tree_invert`, `pf_tree_expand`
- **Graphs ([`docs/graphs.md`](docs/graphs.md))**:
  - Spanning & dominator trees: `pf_graph_dfs_tree`, `pf_graph_bfs_tree`, `pf_graph_dominator_tree` (composes directly with `pf_tree_accumulate_up` for Java heap retained size, and with `pf_tree_flatten_recursion` + `pf_tree_key_siblings` for heap class trees).

---

## Benchmarks vs PerfettoSQL Next (`trace_processor_shell` on `origin/main`)

All benchmarks compare **PerfettoSQL Next (`TP main`)** against **DuckDB + `perfetto.duckdb_extension`** on the exact same traces and verify **0 checksum / row-count mismatches**.
- **PerfettoSQL Next (1T)**: Sum of per-trace query execution times (excluding trace load time) on `TP main`.
- **DuckDB 1T**: Single-thread (`SET threads=1`) query time across all traces in one DuckDB table.
- **DuckDB Multi-Core**: 32 threads (`SET threads=32`) for Intervals & Trees; 48 threads (`SET threads=48`) for Graphs.

### 1. Interval Operators (100 Android field traces: 19.8M slices, 29.8M thread states, 22.8M sched slices, 29.0M counters)

| Operator | Output Rows | PerfettoSQL Next (1T) | DuckDB 1T | 1T Speedup (%) | DuckDB 32T | 32T Speedup vs TP 1T (%) | 32T Speedup vs TP 32-proc Wall (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `INTERVAL FLATTEN` (`COUNT, SUM`) | 29,848,019 | 12.15 s | **2.408 s** | **5.05× (+405%)** | **0.568 s** | **21.39× (+2,039%)** | **28.35× (+2,735%)** |
| `INTERVAL INTERSECTION OF` (2-way `PER utid`) | 11,907,457 | 7.15 s | **2.706 s** | **2.64× (+164%)** | **0.461 s** | **15.51× (+1,451%)** | **31.43× (+3,043%)** |
| `INTERVAL INTERSECTION OF` (3-way `PER utid`) | 8,883,213 | 7.59 s | **3.488 s** | **2.18× (+118%)** | **0.524 s** | **14.48× (+1,348%)** | **27.12× (+2,612%)** |
| `INTERVAL MERGE OVERLAPPING` (`PER track_id`) | 5,140,429 | 20.27 s | **0.867 s** | **23.38× (+2,238%)** | **0.162 s** | **125.12× (+12,412%)** | **78.95× (+7,795%)** |
| `INTERVALS FROM EVENTS` (counter intervals) | 29,019,702 | 6.75 s | **2.676 s** | **2.52× (+152%)** | **0.495 s** | **13.64× (+1,264%)** | **25.82× (+2,482%)** |

### 2. Tree Operators (100 Android field traces: 19,645,364 slices)

| Operator | PerfettoSQL Next / TP (1T) | DuckDB 1T | 1T Speedup (%) | DuckDB 32T | 32T Speedup vs TP 1T (%) | 32T Speedup vs TP 24-proc Wall (%) |
|---|---:|---:|---:|---:|---:|---:|
| `TREE ACCUMULATE UP SUM(dur)` | 3.230 s | **0.760 s** | **4.25× (+325%)** | **0.396 s** | **8.16× (+716%)** | **26.90× (+2,590%)** |
| `TREE ACCUMULATE DOWN SUM(dur)` | 3.100 s | **0.606 s** | **5.12× (+412%)** | **0.337 s** | **9.20× (+820%)** | **30.69× (+2,969%)** |
| `TREE KEEP IF DESCENDANT OF` | 14.379 s | **1.908 s** | **7.54× (+654%)** | **0.832 s** | **17.28× (+1,628%)** | **13.52× (+1,252%)** |
| `TREE KEEP IF ANCESTOR OF` | 16.430 s | **1.630 s** | **10.08× (+908%)** | **0.812 s** | **20.23× (+1,923%)** | **14.07× (+1,307%)** |
| `TREE PRUNE AT` | 15.814 s | **1.894 s** | **8.35× (+735%)** | **0.202 s** | **78.29× (+7,729%)** | **55.66× (+5,466%)** |
| `TREE REPARENT TO NEAREST ANCESTOR` | 26.854 s | **3.688 s** | **7.28× (+628%)** | **0.560 s** | **47.95× (+4,695%)** | **21.46× (+2,046%)** |
| `TREE ABSORB AT ... AGGREGATE SUM(dur)` | 36.262 s | **2.010 s** | **18.04× (+1,704%)** | **1.451 s** | **24.99× (+2,399%)** | **9.03× (+803%)** |
| `TREE CONTRACT AT ... AGGREGATE SUM(dur)` | 13.562 s | **2.018 s** | **6.72× (+572%)** | **1.402 s** | **9.67× (+867%)** | **7.94× (+694%)** |
| `TREE MERGE SIBLINGS BY name` (flamegraph) | 158.081 s | **4.425 s** | **35.72× (+3,472%)** | **1.485 s** | **106.45× (+10,545%)** | **18.23× (+1,723%)** |
| `TREE MERGE INTO PARENT BY name` | 158.659 s | **7.688 s** | **20.64× (+1,964%)** | **1.220 s** | **130.05× (+12,905%)** | **23.51× (+2,251%)** |
| `TREE INVERT` | 52.952 s | **3.634 s** | **14.57× (+1,357%)** | **1.075 s** | **49.26× (+4,826%)** | **14.66× (+1,366%)** |

### 3. Graph Operators (22 Android Java heap dumps: 126.5M objects, 231.7M reference edges, 117.6M reachable objects)

| Operator | Output Rows | PerfettoSQL / TP (1T) | DuckDB 1T | 1T Speedup (%) | DuckDB 48T | 48T Speedup vs TP 1T (%) | 48T Speedup vs TP 22-proc Wall (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `GRAPH DFS TREE` | 117,637,905 | 137.12 s | **16.80 s** | **8.16× (+716%)** | **3.65 s** | **37.57× (+3,657%)** | **9.24× (+824%)** |
| `GRAPH BFS TREE` | 117,637,905 | 140.09 s | **14.90 s** | **9.40× (+840%)** | **3.32 s** | **42.20× (+4,120%)** | **10.10× (+910%)** |
| `GRAPH DOMINATOR TREE` (Lengauer-Tarjan) | 117,637,905 | 171.78 s | **21.40 s** | **8.03× (+703%)** | **4.41 s** | **38.95× (+3,795%)** | **8.15× (+715%)** |
| `GRAPH DOMINATOR TREE` + `TREE ACCUMULATE UP` (Retained Size) | 117,637,905 | 288.87 s | **27.90 s** | **10.35× (+935%)** | **6.12 s** | **47.20× (+4,620%)** | **8.88× (+788%)** |
| `GRAPH BFS TREE` + `TREE ACCUMULATE UP` | 117,637,905 | 543.69 s | **21.30 s** | **25.53× (+2,453%)** | **5.04 s** | **107.88× (+10,688%)** | **15.86× (+1,486%)** |

---

## Testing

Run the three end-to-end test suites (synthetic edge cases, randomized fuzzing against independent Python references, and row-for-row verification against the `trace_processor_shell` oracle):

```bash
python3 tests/test_intervals.py
python3 tests/test_trees.py
python3 tests/test_graphs.py
```

## Repository Layout

- [`src/intervals.cc`](src/intervals.cc), [`sql/intervals.sql`](sql/intervals.sql), [`docs/intervals.md`](docs/intervals.md) — Interval operators (§5)
- [`src/trees.cc`](src/trees.cc), [`sql/trees.sql`](sql/trees.sql), [`docs/trees.md`](docs/trees.md) — Tree operators (§6)
- [`src/graphs.cc`](src/graphs.cc), [`sql/graphs.sql`](sql/graphs.sql), [`docs/graphs.md`](docs/graphs.md) — Graph operators (§7)
- [`src/extension.cc`](src/extension.cc), [`package_extension.py`](package_extension.py), [`build.sh`](build.sh) — Extension entrypoint and packaging
- [`tests/`](tests/) — Correctness and TP oracle test suites
- [`bench/`](bench/) — Multi-trace benchmarks against `trace_processor_shell`
