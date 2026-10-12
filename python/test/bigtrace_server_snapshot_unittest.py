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
"""Snapshot and session tests for the Bigtrace server."""

import os
import sqlite3
import tempfile
import time
import unittest

from perfetto.bigtrace.server import (
    SnapshotStore,
    TraceProcessorPool,
    resolve_trace_processor_bin,
)
from perfetto.bigtrace.server import query_result
from perfetto.bigtrace.server.cli import parse_args
from perfetto.trace_processor.api import TraceProcessorConfig
from perfetto.trace_processor.platform import PlatformDelegate
from test.bigtrace_server_fixture import (
    COLD_TRACE,
    HOT_TRACE,
    base_url,
    copy_trace,
    request,
    running_server,
)

COUNT_SQL = "SELECT count(*) AS n FROM slice"


def _count(tp) -> int:
  return list(tp.query(COUNT_SQL))[0].n


class SnapshotTest(unittest.TestCase):

  def setUp(self):
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    self.tmp = tmp.name
    self.bin_path = resolve_trace_processor_bin()
    self.tp_config = TraceProcessorConfig(bin_path=self.bin_path)
    self.cold = copy_trace(COLD_TRACE, self.tmp, "cold.perfetto-trace")
    self.hot = copy_trace(HOT_TRACE, self.tmp, "hot.perfetto-trace")
    self.snapshot_dir = os.path.join(self.tmp, "snapshots")

  def _store(self, budget_bytes=2**30) -> SnapshotStore:
    return SnapshotStore(self.snapshot_dir, budget_bytes, self.bin_path)

  def _pool(self, store):
    pool = TraceProcessorPool(self.tp_config, store)
    self.addCleanup(pool.close)
    return pool

  def _query_once(self, pool, trace, sql=COUNT_SQL, query=None):
    """Runs `sql` like the executor does; returns (rows, from_snapshot)."""
    own_query = query is None
    query = query or pool.start_query(sql)
    tp = pool.acquire(trace, query=query)
    try:
      result = query_result.query(tp, sql)
      rows = [
          tuple(r)
          for b in result.batches
          for r in query_result.decode_batch(b, len(result.columns))
      ]
      return rows, pool.loaded_from_snapshot(tp)
    finally:
      pool.release(trace, tp)
      if own_query:
        pool.end_query()

  def _snapshot(self, store, trace):
    store.create(trace, self._pool(None).shell_flags())

  def test_snapshot_created_after_first_query_and_reused(self):
    store = self._store()
    self.addCleanup(store.close)
    pool = self._pool(store)
    first, from_snapshot = self._query_once(pool, self.cold)
    self.assertFalse(from_snapshot)
    store.wait()  # Snapshots are created once no query is running.
    self.assertTrue(os.path.isfile(store.find(self.cold)))
    rows, from_snapshot = self._query_once(pool, self.cold)
    self.assertTrue(from_snapshot)
    self.assertEqual(rows, first)

  def test_no_snapshots_are_created_while_a_query_runs(self):
    store = self._store()
    self.addCleanup(store.close)
    pool = self._pool(store)
    query = pool.start_query(COUNT_SQL)
    self._query_once(pool, self.cold, query=query)
    time.sleep(1)
    self.assertIsNone(store.find(self.cold))
    pool.end_query()
    store.wait()
    self.assertIsNotNone(store.find(self.cold))

  def test_trace_change_or_new_shell_version_misses(self):
    store = self._store()
    self._snapshot(store, self.cold)
    store.version = "Perfetto v0-other"
    self.assertIsNone(store.find(self.cold))
    store.version = self._store().version
    self.assertIsNotNone(store.find(self.cold))
    os.utime(self.cold, ns=(1, 1))
    self.assertIsNone(store.find(self.cold))

  def test_corrupt_snapshot_is_deleted_and_trace_parsed(self):
    store = self._store()
    self._snapshot(store, self.cold)
    snapshot = store.find(self.cold)
    with open(snapshot, "wb") as f:
      f.write(b"not a snapshot" * 100)
    pool = self._pool(store)
    with self.assertLogs("perfetto.bigtrace.server.trace_processor_pool",
                         "WARNING"):
      tp = pool.acquire(self.cold)
    self.assertFalse(pool.loaded_from_snapshot(tp))
    self.assertGreater(_count(tp), 0)
    pool.release(self.cold, tp, discard=True)
    self.assertFalse(os.path.exists(snapshot))

  def test_least_recently_used_snapshots_are_evicted_over_budget(self):
    store = self._store(budget_bytes=1)
    self._snapshot(store, self.cold)
    self._snapshot(store, self.hot)
    self.assertIsNone(store.find(self.cold))
    self.assertEqual(len(os.listdir(self.snapshot_dir)), 1)
    # A new store picks up what is on disk.
    self.assertEqual(self._store().total_bytes(), store.total_bytes())

  def test_metadata_queries_use_the_raw_trace(self):
    store = self._store()
    self._snapshot(store, self.cold)
    pool = self._pool(store)
    _, from_snapshot = self._query_once(pool, self.cold)
    self.assertTrue(from_snapshot)
    sql = "SELECT count(*) AS n FROM stats"
    raw_stats, from_snapshot = self._query_once(pool, self.cold, sql)
    self.assertFalse(from_snapshot)
    self.assertGreater(raw_stats[0][0], 3)


class CatalogAndCliTest(unittest.TestCase):

  def test_sqlite_metadata_db_is_the_manifest(self):
    with tempfile.TemporaryDirectory() as tmp:
      trace = copy_trace(COLD_TRACE, tmp, "cold.perfetto-trace")
      with sqlite3.connect(os.path.join(tmp, "metadata.db")) as conn:
        conn.execute("CREATE TABLE traces (trace_uuid TEXT, "
                     "absolute_local_file_path TEXT, device_name TEXT)")
        conn.execute("INSERT INTO traces VALUES ('u1', ?, 'pixel')", (trace,))
      conn.close()
      with running_server(trace_dir=tmp) as server:
        traces = server.catalog.get_traces()
        self.assertEqual([t.trace_uuid for t in traces], ["u1"])
        self.assertEqual(traces[0].get_value("device_name"), "pixel")

  def test_free_ports_are_not_handed_out_twice(self):
    # Shells bind their port only after parsing; a port handed out twice
    # would connect a client to another trace's shell.
    delegate = PlatformDelegate()
    ports = [delegate.get_bind_addr(0)[1] for _ in range(1000)]
    self.assertEqual(len(set(ports)), len(ports))

  def test_defaults_and_explicit_flags(self):
    args = parse_args([])
    self.assertIsNone(args.snapshot_dir)
    self.assertIsNone(args.parquet_dir)
    self.assertIsNone(args.duckdb_path)
    self.assertEqual(args.snapshot_budget_gb, 50)
    args = parse_args([
        "--snapshot-dir", "/tmp/snaps", "--parquet-dir", "/tmp/pq",
        "--duckdb-path", "/tmp/bt.duckdb", "--snapshot-budget-gb", "10"
    ])
    self.assertEqual(args.snapshot_dir, "/tmp/snaps")
    self.assertEqual(args.parquet_dir, "/tmp/pq")
    self.assertEqual(args.duckdb_path, "/tmp/bt.duckdb")
    self.assertEqual(args.snapshot_budget_gb, 10)


if __name__ == "__main__":
  unittest.main()
