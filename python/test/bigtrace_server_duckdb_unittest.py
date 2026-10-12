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
"""DuckDB mode tests: result tables, typing, cancellation and concurrency."""

import time

from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig
from test.bigtrace_server_fixture import (
    BigtraceServerTestCase,
    COLD_TRACE,
    resolve_trace_processor_bin,
    trace_path,
    wait_for,
)

_COLD_ONLY = [{"field": "file_name", "op": "glob", "value": "*startup_cold*"}]
# Streams for hours unless interrupted.
_SLOW_DUCKDB_SQL = "SELECT sum(i % 7) FROM range(1000000000000) t(i)"


class BigtraceServerDuckdbTest(BigtraceServerTestCase):

  def _traces_query(self, sql, **extra):
    status, data, _ = self._request("POST", "/execute_bigtrace_query", {
        "perfetto_sql": sql,
        "limit": 100000,
        "trace_filters": _COLD_ONLY,
        **extra
    })
    self.assertEqual(status, 200, data)
    return data

  def _duckdb_query(self, sql, limit=1000):
    return self._request("POST", "/execute_bigtrace_query", {
        "engine": "duckdb",
        "perfetto_sql": sql,
        "limit": limit
    })

  def _status(self, uid):
    return self._request("GET", f"/query_executions/{uid}:status")[1]

  def _wait_runs_finished(self, timeout=10.0):
    executor = self.server.executor
    wait_for(lambda: not any(r.is_alive() for r in executor._runs), timeout)

  def test_result_table_is_queryable_as_soon_as_query_succeeds(self):
    status, data, _ = self._request(
        "POST", "/execute_bigtrace_query_async", {
            "perfetto_sql": "SELECT ts, dur FROM slice LIMIT 50",
            "limit": 1000,
            "trace_filters": _COLD_ONLY,
        })
    self.assertEqual(status, 200, data)
    uid, table = data["queryUuid"], data["tableName"]
    wait_for(lambda: self._status(uid)["status"] != "IN_PROGRESS")
    final = self._status(uid)
    self.assertEqual(final["status"], "SUCCESS")
    status, res, _ = self._duckdb_query(f"SELECT count(*) AS n FROM {table}")
    self.assertEqual(status, 200, res)
    self.assertGreater(final["processedRows"], 0)
    self.assertEqual(res["rows"][0]["values"], [str(final["processedRows"])])

  def test_duckdb_query_adds_exactly_one_history_entry(self):
    before = {
        e["queryUuid"]
        for e in self._request("GET", "/query_executions")[1]["queryExecutions"]
    }
    status, res, _ = self._duckdb_query("SELECT 42 AS answer")
    self.assertEqual(status, 200, res)
    after = self._request("GET", "/query_executions")[1]["queryExecutions"]
    added = [e for e in after if e["queryUuid"] not in before]
    self.assertEqual(len(added), 1)
    self.assertEqual(added[0]["queryEngine"], "duckdb")
    self.assertEqual(added[0]["queryUuid"], res["queryUuid"])

  def test_values_are_typed_from_the_result_schema(self):
    status, res, _ = self._duckdb_query("""
        SELECT 1::BIGINT AS i, 2.5::DECIMAL(10, 2) AS d,
               TIMESTAMPTZ '2020-01-01 00:00:00+00' AS tz, [1, 2] AS l,
               true AS b, NULL::INTEGER AS n, 'x' AS s""")
    self.assertEqual(status, 200, res)
    self.assertEqual([(c["name"], c["type"]) for c in res["schema"]],
                     [("i", "INT64"), ("d", "DOUBLE"), ("tz", "STRING"),
                      ("l", "STRING"), ("b", "BOOLEAN"), ("n", "INT64"),
                      ("s", "STRING")])
    values = res["rows"][0]["values"]
    self.assertEqual(values[0], "1")
    self.assertEqual(float(values[1]), 2.5)
    self.assertTrue(values[2].startswith("2020-01-01"))
    self.assertEqual(values[3], "[1, 2]")
    self.assertIsNone(values[5])
    self.assertEqual(values[6], "x")

  def test_duckdb_error_fails_the_query(self):
    status, res, _ = self._duckdb_query("SELECT * FROM no_such_table_xyz")
    self.assertEqual(status, 400)
    self.assertIn("no_such_table_xyz", res["detail"])

  def test_cancel_interrupts_a_running_duckdb_query(self):
    status, data, _ = self._request("POST", "/execute_bigtrace_query_async", {
        "engine": "duckdb",
        "perfetto_sql": _SLOW_DUCKDB_SQL,
        "limit": 10
    })
    self.assertEqual(status, 200, data)
    uid = data["queryUuid"]
    self._request("POST", f"/query_executions/{uid}:cancel", {})
    self._wait_runs_finished()
    self.assertEqual(self._status(uid)["status"], "CANCELLED")

  def test_trace_queries_finish_while_a_duckdb_query_runs(self):
    _, data, _ = self._request("POST", "/execute_bigtrace_query_async", {
        "engine": "duckdb",
        "perfetto_sql": _SLOW_DUCKDB_SQL,
        "limit": 10
    })
    slow_uid = data["queryUuid"]
    try:
      wait_for(lambda: "_duck_cursor" in self.server.executor._executions.get(
          slow_uid, {}))
      start = time.monotonic()
      res = self._traces_query("SELECT 1 AS one")
      self.assertEqual(self._status(res["queryUuid"])["status"], "SUCCESS")
      self.assertEqual(self._request("GET", "/query_executions")[0], 200)
      self.assertLess(time.monotonic() - start, 10)
    finally:
      self._request("POST", f"/query_executions/{slow_uid}:cancel", {})
      self._wait_runs_finished()

  def test_deleting_a_query_drops_its_table(self):
    res = self._traces_query("SELECT 1 AS one", table_name="to_delete")
    tables = lambda: {
        t["name"] for t in self._request("GET", "/duckdb/tables")[1]["tables"]
    }
    self.assertIn("to_delete", tables())
    self._request("DELETE", f"/query_executions/{res['queryUuid']}")
    self.assertNotIn("to_delete", tables())
    status, _, _ = self._duckdb_query("SELECT * FROM to_delete")
    self.assertEqual(status, 400)

  def test_deleting_an_older_run_keeps_a_newer_table_of_the_same_name(self):
    old = self._traces_query("SELECT 1 AS v", table_name="reused")
    self._traces_query("SELECT 2 AS v", table_name="reused")
    self._request("DELETE", f"/query_executions/{old['queryUuid']}")
    status, res, _ = self._duckdb_query("SELECT DISTINCT v FROM reused")
    self.assertEqual(status, 200, res)
    self.assertEqual([r["values"] for r in res["rows"]], [["2"]])

  def test_interval_flatten_over_a_slice_result_table(self):
    if not self.server.executor.duckdb.extension_loaded:
      self.skipTest("perfetto.duckdb_extension not available")
    res = self._traces_query(
        "SELECT id, ts, dur, depth, track_id FROM slice",
        table_name="cold_slices")
    total = res["totalFilteredRows"]
    self.assertGreater(total, 0)
    per = "per := ['_trace_uuid', 'track_id']"
    rel = "'(SELECT * FROM cold_slices WHERE dur >= 0)'"
    status, flat_sum, _ = self._duckdb_query(
        f"SELECT count(*) AS n, max(count) AS max_count "
        f"FROM pf_interval_flatten_sum({rel}, ['depth'], {per})")
    self.assertEqual(status, 200, flat_sum)
    n, max_count = (int(v) for v in flat_sum["rows"][0]["values"])
    self.assertGreater(n, 0)
    self.assertGreaterEqual(max_count, 1)
    status, flat, _ = self._duckdb_query(
        f"SELECT count(*) AS n FROM pf_interval_flatten({rel}, "
        f"'count(*) AS concurrency, max(depth) AS max_depth', {per})")
    self.assertEqual(status, 200, flat)
    self.assertEqual(int(flat["rows"][0]["values"][0]), n)


class BigtraceServerIntervalWorkflowsTest(BigtraceServerTestCase):
  """End-to-end tests for interval workflows (A, B, C), TP comparison, and edge cases."""

  def _duckdb_query(self, sql, limit=1000):
    return self._request("POST", "/execute_bigtrace_query", {
        "engine": "duckdb",
        "perfetto_sql": sql,
        "limit": limit
    })

  def test_workflow_a_slices_sched_intersection_and_aggregate(self):
    if not self.server.executor.duckdb.extension_loaded:
      self.skipTest("perfetto.duckdb_extension not available")
    status, batch_res, _ = self._request(
        "POST", "/execute_bigtrace_query", {
            "queries": [
                {
                    "perfetto_sql": (
                        "SELECT s.id, s.ts, s.dur, s.depth, s.name, s.track_id, "
                        "tt.utid FROM slice s JOIN thread_track tt ON s.track_id = tt.id"
                    ),
                    "table_name": "slices"
                },
                {
                    "perfetto_sql": (
                        "SELECT id, ts, dur, cpu, utid FROM sched WHERE utid != 0"
                    ),
                    "table_name": "sched"
                },
            ],
            "limit": 100000,
            "trace_filters": _COLD_ONLY,
        })
    self.assertEqual(status, 200, batch_res)
    self.assertEqual(len(batch_res["executions"]), 2)
    self.assertEqual([e["tableName"] for e in batch_res["executions"]],
                     ["slices", "sched"])

    duck_sql = """
        SELECT s.name, sum(dur) AS running_dur
        FROM pf_interval_intersection(
          ['(SELECT * FROM slices WHERE dur >= 0)', '(SELECT * FROM sched WHERE dur >= 0)'],
          ['s', 'c'],
          per := ['_trace_uuid', 'utid']
        )
        GROUP BY s.name
        ORDER BY running_dur DESC
        LIMIT 10
    """
    status, res, _ = self._duckdb_query(duck_sql)
    self.assertEqual(status, 200, res)
    self.assertGreater(len(res["rows"]), 0)
    top_name = res["rows"][0]["values"][0]
    top_dur = float(res["rows"][0]["values"][1])
    self.assertTrue(top_name)
    self.assertGreater(top_dur, 0.0)

    # Struct columns projected directly returned to UI as JSON strings
    struct_sql = """
        SELECT ts, dur, _trace_uuid, utid, s, c
        FROM pf_interval_intersection(
          ['(SELECT * FROM slices WHERE dur >= 0)', '(SELECT * FROM sched WHERE dur >= 0)'],
          ['s', 'c'],
          per := ['_trace_uuid', 'utid']
        )
        LIMIT 2
    """
    status, sres, _ = self._duckdb_query(struct_sql)
    self.assertEqual(status, 200, sres)
    self.assertGreater(len(sres["rows"]), 0)
    s_val = sres["rows"][0]["values"][4]
    c_val = sres["rows"][0]["values"][5]
    self.assertTrue(s_val.startswith("{") and "id" in s_val and "name" in s_val)
    self.assertTrue(c_val.startswith("{") and "id" in c_val and "cpu" in c_val)

  def test_workflow_b_self_interval_intersect_slices(self):
    if not self.server.executor.duckdb.extension_loaded:
      self.skipTest("perfetto.duckdb_extension not available")
    status, res, _ = self._request(
        "POST", "/execute_bigtrace_query", {
            "perfetto_sql": (
                "SELECT s.id, s.ts, s.dur, s.depth, s.name, s.track_id, tt.utid "
                "FROM slice s JOIN thread_track tt ON s.track_id = tt.id"),
            "table_name": "slices_b",
            "limit": 100000,
            "trace_filters": _COLD_ONLY,
        })
    self.assertEqual(status, 200, res)

    # Self-intersect per track_id
    status, res_track, _ = self._duckdb_query("""
        SELECT count(*) AS cnt, sum(dur) AS total_dur
        FROM pf_interval_intersection(
          ['(SELECT * FROM slices_b WHERE dur >= 0)', '(SELECT * FROM slices_b WHERE dur >= 0)'],
          ['a', 'b'],
          per := ['_trace_uuid', 'track_id']
        )
    """)
    self.assertEqual(status, 200, res_track)
    cnt_track = int(res_track["rows"][0]["values"][0])
    dur_track = float(res_track["rows"][0]["values"][1])
    self.assertGreater(cnt_track, 0)
    self.assertGreater(dur_track, 0.0)

    # Self-intersect per utid
    status, res_utid, _ = self._duckdb_query("""
        SELECT count(*) AS cnt, sum(dur) AS total_dur
        FROM pf_interval_intersection(
          ['(SELECT * FROM slices_b WHERE dur >= 0)', '(SELECT * FROM slices_b WHERE dur >= 0)'],
          ['a', 'b'],
          per := ['_trace_uuid', 'utid']
        )
    """)
    self.assertEqual(status, 200, res_utid)
    cnt_utid = int(res_utid["rows"][0]["values"][0])
    dur_utid = float(res_utid["rows"][0]["values"][1])
    self.assertGreater(cnt_utid, 0)
    self.assertGreater(dur_utid, 0.0)

  def test_workflow_c_flatten_and_flatten_sum_slices(self):
    if not self.server.executor.duckdb.extension_loaded:
      self.skipTest("perfetto.duckdb_extension not available")
    status, res, _ = self._request(
        "POST", "/execute_bigtrace_query", {
            "perfetto_sql": (
                "SELECT s.id, s.ts, s.dur, s.depth, s.name, s.track_id, tt.utid "
                "FROM slice s JOIN thread_track tt ON s.track_id = tt.id"),
            "table_name": "slices_c",
            "limit": 100000,
            "trace_filters": _COLD_ONLY,
        })
    self.assertEqual(status, 200, res)

    per = "per := ['_trace_uuid', 'track_id']"
    rel = "'(SELECT * FROM slices_c WHERE dur >= 0)'"
    status, flat_sum, _ = self._duckdb_query(
        f"SELECT count(*) AS n, max(count) AS max_count "
        f"FROM pf_interval_flatten_sum({rel}, ['depth'], {per})")
    self.assertEqual(status, 200, flat_sum)
    n, max_count = (int(v) for v in flat_sum["rows"][0]["values"])
    self.assertGreater(n, 0)
    self.assertGreaterEqual(max_count, 1)

    status, flat, _ = self._duckdb_query(
        f"SELECT count(*) AS n FROM pf_interval_flatten({rel}, "
        f"'count(*) AS concurrency, max(depth) AS max_depth', {per})")
    self.assertEqual(status, 200, flat)
    self.assertEqual(int(flat["rows"][0]["values"][0]), n)

  def test_correctness_compare_trace_processor_and_duckdb_intersection(self):
    if not self.server.executor.duckdb.extension_loaded:
      self.skipTest("perfetto.duckdb_extension not available")

    # 1. Compute ground truth directly with TraceProcessor
    tp = TraceProcessor(
        trace=trace_path(COLD_TRACE),
        config=TraceProcessorConfig(bin_path=resolve_trace_processor_bin()))
    try:
      tp.query("INCLUDE PERFETTO MODULE intervals.intersect;")
      tp.query("""
          CREATE PERFETTO TABLE cmp_slices AS
          SELECT s.id, s.ts, s.dur, tt.utid
          FROM slice s JOIN thread_track tt ON s.track_id = tt.id
          WHERE s.dur >= 0;
      """)
      tp.query("""
          CREATE PERFETTO TABLE cmp_sched AS
          SELECT id, ts, dur, utid
          FROM sched
          WHERE utid != 0 AND dur >= 0;
      """)
      tp_res = list(
          tp.query("""
          SELECT count(*) AS cnt, sum(dur) AS total_dur
          FROM _interval_intersect!((cmp_slices, cmp_sched), (utid));
      """))
      expected_cnt = int(tp_res[0].cnt)
      expected_dur = int(tp_res[0].total_dur)
    finally:
      tp.close()

    # 2. Compute via Bigtrace Server + DuckDB pf_interval_intersection
    status, batch_res, _ = self._request(
        "POST", "/execute_bigtrace_query", {
            "queries": [
                {
                    "perfetto_sql": (
                        "SELECT s.id, s.ts, s.dur, s.depth, s.name, s.track_id, "
                        "tt.utid FROM slice s JOIN thread_track tt ON s.track_id = tt.id"
                    ),
                    "table_name": "slices_cmp"
                },
                {
                    "perfetto_sql": (
                        "SELECT id, ts, dur, cpu, utid FROM sched WHERE utid != 0"
                    ),
                    "table_name": "sched_cmp"
                },
            ],
            "limit":
                100000,
            "trace_filters": [{
                "field": "file_name",
                "op": "=",
                "value": COLD_TRACE
            }],
        })
    self.assertEqual(status, 200, batch_res)

    status, duck_res, _ = self._duckdb_query("""
        SELECT count(*) AS cnt, sum(dur) AS total_dur
        FROM pf_interval_intersection(
          ['(SELECT * FROM slices_cmp WHERE dur >= 0)', '(SELECT * FROM sched_cmp WHERE dur >= 0)'],
          ['s', 'c'],
          per := ['_trace_uuid', 'utid']
        );
    """)
    self.assertEqual(status, 200, duck_res)
    actual_cnt = int(duck_res["rows"][0]["values"][0])
    actual_dur = int(float(duck_res["rows"][0]["values"][1]))

    self.assertEqual(actual_cnt, expected_cnt)
    self.assertEqual(actual_dur, expected_dur)

  def test_multi_query_duplicate_table_names_disambiguated(self):
    status, res, _ = self._request(
        "POST", "/execute_bigtrace_query", {
            "queries": [
                {
                    "perfetto_sql": "SELECT 1 AS col1;",
                    "table_name": "dup_tbl"
                },
                {
                    "perfetto_sql": "SELECT 2 AS col2;",
                    "table_name": "dup_tbl"
                },
            ]
        })
    self.assertEqual(status, 200, res)
    names = [e["tableName"] for e in res["executions"]]
    self.assertEqual(names, ["dup_tbl", "dup_tbl_2"])

    tables = {
        t["name"] for t in self._request("GET", "/duckdb/tables")[1]["tables"]
    }
    self.assertIn("dup_tbl", tables)
    self.assertIn("dup_tbl_2", tables)

    s1, r1, _ = self._duckdb_query("SELECT col1 FROM dup_tbl")
    self.assertEqual(s1, 200, r1)
    self.assertEqual(r1["rows"][0]["values"][0], "1")

    s2, r2, _ = self._duckdb_query("SELECT col2 FROM dup_tbl_2")
    self.assertEqual(s2, 200, r2)
    self.assertEqual(r2["rows"][0]["values"][0], "2")

  def test_multi_query_one_fails_while_other_succeeds_sync_and_async(self):
    # Sync: partial success returns 200 with successful query data
    status, res, _ = self._request(
        "POST", "/execute_bigtrace_query", {
            "queries": [
                {
                    "perfetto_sql": "SELECT * FROM nonexistent_table_xyz;",
                    "table_name": "bad_sync"
                },
                {
                    "perfetto_sql": "SELECT 42 AS answer;",
                    "table_name": "good_sync"
                },
            ]
        })
    self.assertEqual(status, 200, res)
    self.assertEqual(len(res["executions"]), 2)
    self.assertEqual(res["rows"][0]["values"][0], "42")
    bad_uid = res["executions"][0]["queryUuid"]
    good_uid = res["executions"][1]["queryUuid"]
    self.assertEqual(
        self._request("GET",
                      f"/query_executions/{bad_uid}:status")[1]["status"],
        "FAILED")
    self.assertEqual(
        self._request("GET",
                      f"/query_executions/{good_uid}:status")[1]["status"],
        "SUCCESS")

    # Async: both tracked, one FAILED and one SUCCESS
    status, ares, _ = self._request(
        "POST", "/execute_bigtrace_query_async", {
            "queries": [
                {
                    "perfetto_sql": "SELECT * FROM nonexistent_table_xyz;",
                    "table_name": "bad_async"
                },
                {
                    "perfetto_sql": "SELECT 100 AS answer;",
                    "table_name": "good_async"
                },
            ]
        })
    self.assertEqual(status, 200, ares)
    bad_async_uid = ares["executions"][0]["queryUuid"]
    good_async_uid = ares["executions"][1]["queryUuid"]
    wait_for(lambda: self._request(
        "GET", f"/query_executions/{good_async_uid}:status")[1]["status"] !=
             "IN_PROGRESS")
    self.assertEqual(
        self._request("GET",
                      f"/query_executions/{bad_async_uid}:status")[1]["status"],
        "FAILED")
    self.assertEqual(
        self._request(
            "GET", f"/query_executions/{good_async_uid}:status")[1]["status"],
        "SUCCESS")

  def test_column_name_collisions_like_id(self):
    status, res, _ = self._request(
        "POST", "/execute_bigtrace_query", {
            "perfetto_sql":
                ("SELECT s.id, tt.id FROM slice s "
                 "JOIN thread_track tt ON s.track_id = tt.id LIMIT 5;"),
            "table_name": "colliding_ids"
        })
    self.assertEqual(status, 200, res)
    row0 = res["rows"][0]["values"]
    self.assertNotEqual(row0[0], row0[1])

    status, dres, _ = self._duckdb_query(
        "SELECT id, id_1 FROM colliding_ids LIMIT 5")
    self.assertEqual(status, 200, dres)
    drow0 = dres["rows"][0]["values"]
    self.assertEqual(drow0[0], row0[0])
    self.assertEqual(drow0[1], row0[1])

  def test_operators_work_on_result_tables_and_inline_sql(self):
    if not self.server.executor.duckdb.extension_loaded:
      self.skipTest("perfetto.duckdb_extension not available")
    status, batch_res, _ = self._request(
        "POST", "/execute_bigtrace_query", {
            "queries": [
                {
                    "perfetto_sql": (
                        "SELECT s.id, s.parent_id, s.ts, s.dur, s.depth, "
                        "s.name, s.track_id, tt.utid "
                        "FROM slice s JOIN thread_track tt ON s.track_id = tt.id "
                        "WHERE s.dur >= 0 LIMIT 200"),
                    "table_name": "slices_clean",
                },
                {
                    "perfetto_sql": ("SELECT id, ts, dur, cpu, utid FROM sched "
                                     "WHERE utid != 0 AND dur >= 0 LIMIT 200"),
                    "table_name": "sched_clean",
                },
            ],
            "limit": 10000,
            "trace_filters": _COLD_ONLY,
        })
    self.assertEqual(status, 200, batch_res)

    # 1. Interval flatten: direct table name vs inline SQL (no parens) vs inline SQL (parens),
    #    and string agg vs list agg.
    s_tbl, r_tbl, _ = self._duckdb_query(
        "SELECT count(*) FROM pf_interval_flatten("
        "'slices_clean', 'count(*) AS n', per := ['_trace_uuid', 'track_id'])")
    s_lst, r_lst, _ = self._duckdb_query(
        "SELECT count(*) FROM pf_interval_flatten("
        "'slices_clean', ['count(*) AS n'], per := ['_trace_uuid', 'track_id'])"
    )
    s_inl, r_inl, _ = self._duckdb_query(
        "SELECT count(*) FROM pf_interval_flatten("
        "'SELECT * FROM slices_clean WHERE dur >= 0;', ['count(*) AS n'], "
        "per := ['_trace_uuid', 'track_id'])")
    self.assertEqual(s_tbl, 200, r_tbl)
    self.assertEqual(s_lst, 200, r_lst)
    self.assertEqual(s_inl, 200, r_inl)
    self.assertGreater(int(r_tbl["rows"][0]["values"][0]), 0)
    self.assertEqual(r_tbl["rows"], r_lst["rows"])
    self.assertEqual(r_tbl["rows"], r_inl["rows"])

    # 2. Interval intersection: direct table names vs inline SQL without parens.
    s_int_tbl, r_int_tbl, _ = self._duckdb_query(
        "SELECT count(*) FROM pf_interval_intersection("
        "['slices_clean', 'sched_clean'], ['s', 'c'], "
        "per := ['_trace_uuid', 'utid'])")
    s_int_inl, r_int_inl, _ = self._duckdb_query(
        "SELECT count(*) FROM pf_interval_intersection("
        "['SELECT * FROM slices_clean', 'SELECT * FROM sched_clean'], "
        "['s', 'c'], per := ['_trace_uuid', 'utid'])")
    self.assertEqual(s_int_tbl, 200, r_int_tbl)
    self.assertEqual(s_int_inl, 200, r_int_inl)
    self.assertEqual(r_int_tbl["rows"], r_int_inl["rows"])

    # 3. Tree operator: direct table name vs inline SQL, list agg vs string agg.
    #    Null out parent_id when the parent row wasn't inside the LIMIT 200 sample.
    tree_sql = (
        "SELECT s._trace_uuid, s.id, "
        "CASE WHEN p.id IS NOT NULL THEN s.parent_id ELSE NULL END AS parent_id, "
        "s.dur FROM slices_clean s "
        "LEFT JOIN slices_clean p ON s._trace_uuid = p._trace_uuid AND s.parent_id = p.id"
    )
    s_tree, r_tree, _ = self._duckdb_query(
        f"SELECT count(*), sum(total) FROM pf_tree_accumulate("
        f"'{tree_sql}', 'up', 'sum(dur) AS total', per := '_trace_uuid')")
    self.assertEqual(s_tree, 200, r_tree)
    self.assertGreater(int(r_tree["rows"][0]["values"][0]), 0)

    # 4. Graph operator: inline SQL on both nodes, edges, and seeds.
    s_graph, r_graph, _ = self._duckdb_query(
        "SELECT count(*) FROM pf_graph_dfs_tree("
        "'SELECT _trace_uuid, id, name FROM slices_clean', "
        "'SELECT _trace_uuid, parent_id AS src, id AS dst FROM slices_clean WHERE parent_id IS NOT NULL', "
        "'SELECT _trace_uuid, id FROM slices_clean WHERE parent_id IS NULL', "
        "per := '_trace_uuid')")
    self.assertEqual(s_graph, 200, r_graph)
    self.assertGreater(int(r_graph["rows"][0]["values"][0]), 0)

  def test_failed_query_does_not_claim_table_and_single_delete_preserves_siblings(
      self):
    # A failed query ("select * sched" syntax error) must not create or claim a table.
    s_fail, r_fail, _ = self._request("POST", "/execute_bigtrace_query", {
        "perfetto_sql": "SELECT * sched",
        "table_name": "sched_never_written",
    })
    self.assertEqual(s_fail, 400, r_fail)

    s_chk, r_chk, _ = self._request("POST", "/check_table_exists", {
        "table_name": "sched_never_written",
    })
    self.assertEqual(s_chk, 200, r_chk)
    self.assertFalse(r_chk["exists"])

    # Create two real queries, delete one, and verify only that one disappears.
    _, q1, _ = self._request("POST", "/execute_bigtrace_query", {
        "perfetto_sql": "SELECT 1 AS a",
        "table_name": "hist_keep",
    })
    _, q2, _ = self._request("POST", "/execute_bigtrace_query", {
        "perfetto_sql": "SELECT 2 AS b",
        "table_name": "hist_drop",
    })
    s_del, _, _ = self._request(
        "DELETE", f"/query_executions/{q2['queryUuid']}?drop_table=true")
    self.assertEqual(s_del, 200)

    _, hist, _ = self._request("GET", "/query_executions")
    uuids = {e["queryUuid"] for e in hist["queryExecutions"]}
    self.assertNotIn(q2["queryUuid"], uuids)
    self.assertIn(q1["queryUuid"], uuids)


if __name__ == "__main__":
  import unittest
  unittest.main()
