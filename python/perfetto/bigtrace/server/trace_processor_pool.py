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
"""Opens a TraceProcessor per trace, optionally backed by parse-once snapshots."""

from concurrent.futures import ThreadPoolExecutor
import dataclasses as dc
import logging
import re
import threading
from typing import Iterable, List, Optional
import weakref

from perfetto.trace_processor.api import (
    TraceProcessor,
    TraceProcessorConfig,
    TraceProcessorException,
)
from perfetto.bigtrace.server.snapshot import NEEDS_RAW_TRACE, SnapshotStore

logger = logging.getLogger(__name__)

# Large traces can take minutes to parse.
_LOAD_TIMEOUT_SECS = 600


def load_trace(
    trace_path: str,
    tp_config: TraceProcessorConfig,
    cancel_event: Optional[threading.Event] = None,
) -> TraceProcessor:
  """Starts a TraceProcessor with the trace loaded.

  The shell reads the file itself: streaming it through the Python API costs
  Python CPU time per chunk, which serializes parallel loads on the GIL.
  """
  config = dc.replace(
      tp_config,
      cancel_event=cancel_event,
      extra_flags=list(tp_config.extra_flags or []) + [trace_path],
      load_timeout=max(tp_config.load_timeout, _LOAD_TIMEOUT_SECS))
  try:
    return TraceProcessor(config=config)
  except TraceProcessorException as e:
    match = re.search(r"stderr: (.*?)\nIf this is", str(e), re.DOTALL)
    lines = match.group(1).strip().splitlines() if match else []
    raise TraceProcessorException(lines[-1] if lines else str(e)) from None


def close_quietly(tp: TraceProcessor) -> None:
  try:
    tp.close()
  except Exception:  # pylint: disable=broad-except
    logger.debug("Error closing TraceProcessor", exc_info=True)


def _close_all(tps: Iterable[TraceProcessor]) -> None:
  with ThreadPoolExecutor(max_workers=32) as closer:
    closer.map(close_quietly, tps)


@dc.dataclass(frozen=True)
class PoolQuery:
  """A query as the pool sees it; see TraceProcessorPool.start_query."""
  sql: str = ""
  needs_raw: bool = False


class TraceProcessorPool:
  """Opens and closes a TraceProcessor for each trace in a query.

  Snapshots (optional): with a SnapshotStore, a trace loads from its snapshot
  if it has one. Otherwise it is parsed, and release() schedules a snapshot,
  which a separate shell makes once no query is running (see start_query).

  Fidelity: snapshots rewrite metadata, stats and trace_file, so a query using
  them (NEEDS_RAW_TRACE) gets an instance parsed from the raw trace.
  """

  def __init__(self,
               tp_config: Optional[TraceProcessorConfig] = None,
               snapshots: Optional[SnapshotStore] = None):
    self.tp_config = tp_config or TraceProcessorConfig()
    self.snapshots = snapshots
    self._lock = threading.Lock()
    # Active instances mapped to whether they loaded from a snapshot.
    self._instances: "weakref.WeakKeyDictionary[TraceProcessor, bool]" = (
        weakref.WeakKeyDictionary())
    self._closed = False

  def start_query(self, sql: str) -> PoolQuery:
    """Registers a query that is about to run on many traces.

    Holds back snapshots until the matching end_query().
    """
    if self.snapshots:
      self.snapshots.pause()
    return PoolQuery(sql, bool(NEEDS_RAW_TRACE.search(sql)))

  def end_query(self) -> None:
    if self.snapshots:
      self.snapshots.resume()

  def acquire(
      self,
      trace_path: str,
      cancel_event: Optional[threading.Event] = None,
      query: PoolQuery = PoolQuery()
  ) -> TraceProcessor:
    snapshot = None
    if self.snapshots and not query.needs_raw:
      snapshot = self.snapshots.find(trace_path)
    if snapshot:
      try:
        tp = load_trace(snapshot, self.tp_config, cancel_event)
        return self._track(tp, from_snapshot=True)
      except TraceProcessorException as e:
        if self._closed or (cancel_event and cancel_event.is_set()):
          raise
        logger.warning("Deleting unloadable snapshot %s: %s", snapshot, e)
        self.snapshots.discard(snapshot)
    tp = load_trace(trace_path, self.tp_config, cancel_event)
    return self._track(tp, from_snapshot=False)

  def release(self,
              trace_path: str,
              tp: TraceProcessor,
              discard: bool = False) -> None:
    with self._lock:
      from_snapshot = self._instances.pop(tp, True)
    if self.snapshots and not from_snapshot and not discard:
      self.snapshots.schedule(trace_path, self.shell_flags())
    close_quietly(tp)

  def loaded_from_snapshot(self, tp: TraceProcessor) -> bool:
    with self._lock:
      return bool(self._instances.get(tp, False))

  def shell_flags(self) -> List[str]:
    """Flags of the shells this pool starts, which snapshots must match."""
    cfg = self.tp_config
    return ([] if cfg.ingest_ftrace_in_raw else ["--no-ftrace-raw"]) + (
        ["--dev"] if cfg.enable_dev_features else []) + list(cfg.extra_flags or
                                                             [])

  def close(self) -> None:
    """Closes every active instance."""
    with self._lock:
      self._closed = True
      instances = list(self._instances.keys())
      self._instances.clear()
    _close_all(instances)

  def _track(self, tp: TraceProcessor, from_snapshot: bool) -> TraceProcessor:
    with self._lock:
      if not self._closed:
        self._instances[tp] = from_snapshot
        return tp
    close_quietly(tp)
    raise TraceProcessorException("Trace processor pool is closed")
