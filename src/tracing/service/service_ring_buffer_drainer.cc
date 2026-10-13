/*
 * Copyright (C) 2026 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/tracing/service/service_ring_buffer_drainer.h"

#include <utility>

#include "perfetto/base/compiler.h"
#include "perfetto/base/logging.h"
#include "perfetto/base/time.h"
#include "src/tracing/service/trace_buffer_v2.h"

namespace perfetto::tracing_v2 {

namespace {

// Delay retries so writers that repeatedly win chunk state transitions
// cannot keep the service sequence busy.
constexpr uint32_t kDrainRetryDelayMs = 1;

}  // namespace

ServiceRingBufferDrainer::Delegate::~Delegate() = default;

ServiceRingBufferDrainer::ServiceRingBufferDrainer(
    std::shared_ptr<SharedMemory> memory,
    uint32_t chunk_size_bytes,
    ProducerID producer_id,
    ClientIdentity client_identity,
    Delegate* delegate,
    base::TaskRunner* task_runner)
    : producer_id_(producer_id),
      client_identity_(client_identity),
      delegate_(delegate),
      memory_(std::move(memory)),
      ring_buffer_(static_cast<uint8_t*>(memory_->start()),
                   memory_->size(),
                   chunk_size_bytes),
      reader_(&ring_buffer_, this),
      weak_runner_(task_runner) {}

ServiceRingBufferDrainer::~ServiceRingBufferDrainer() {
  PERFETTO_DCHECK_THREAD(thread_checker_);
}

void ServiceRingBufferDrainer::Drain() {
  PERFETTO_DCHECK_THREAD(thread_checker_);

  if (PERFETTO_UNLIKELY(reader_.has_protocol_error()))
    return;

  // Bound each pass to num_chunks() logical positions, including positions
  // reserved during the pass.
  const SharedRingBufferReader::Stats stats_before = reader_.GetStats();
  const base::TimeNanos time_before = base::GetWallTimeNs();
  const auto result = reader_.Drain(ring_buffer_.num_chunks());
  ++drain_passes_;
  drain_time_ns_ +=
      static_cast<uint64_t>((base::GetWallTimeNs() - time_before).count());
  const SharedRingBufferReader::Stats stats_after = reader_.GetStats();

  // The reader discards malformed chunks and chunks in an unknown format. It
  // reports them as loss for the writer. Count them here, once per pass.
  const uint64_t num_rejected =
      (stats_after.malformed_chunks - stats_before.malformed_chunks) +
      (stats_after.unsupported_format_chunks -
       stats_before.unsupported_format_chunks);
  if (PERFETTO_UNLIKELY(num_rejected != 0))
    delegate_->OnRingBufferChunksDiscarded(num_rejected);

  if (PERFETTO_UNLIKELY(reader_.has_protocol_error())) {
    // Record the error in every authorized destination. Later drains return
    // above, so each destination counts this violation only once.
    delegate_->ForEachRingBufferDestination(
        [](TraceBufferV2& buffer) { buffer.RecordAbiViolation(); });
    return;
  }

  if (result.needs_another_drain() && !retry_scheduled_) {
    retry_scheduled_ = true;
    weak_runner_.PostDelayedTask(
        [this] {
          // Clear the flag so Drain() can schedule another retry if needed.
          retry_scheduled_ = false;
          Drain();
        },
        kDrainRetryDelayMs);
  }
}

void ServiceRingBufferDrainer::OnChunkRead(
    const SharedRingBufferReader::ChunkContents& chunk) {
  if (PERFETTO_UNLIKELY(!chunk.writer_id || chunk.writer_id > kMaxWriterID)) {
    ++invalid_writer_chunks_;
    delegate_->OnRingBufferChunksDiscarded(1);
    return;
  }

  TraceBufferV2* buffer =
      delegate_->GetRingBufferDestination(chunk.target_buffer);
  if (PERFETTO_UNLIKELY(!buffer)) {
    ++invalid_destination_chunks_;
    delegate_->OnRingBufferChunksDiscarded(1);
    RecordWriterLoss(chunk.writer_id);
    return;
  }

  // The reader reports a chunk with kFlagDataLoss through OnDataLoss(). Only
  // the continuation flags can reach this point.
  PERFETTO_DCHECK(!(chunk.payload_flags & kFlagDataLoss));
  const TraceBufferV2::PacketSequenceProperties sequence{
      producer_id_, client_identity_, chunk.writer_id};
  // On rejection, the trace buffer counts the chunk in its own statistics and
  // records the loss on the sequence.
  //
  // TODO(sashwinbalaji): the payload is copied twice: from shared memory into
  // the reader's scratch, then into a TBv2 chunk. The reader could pass the
  // decoded sizes and one payload range, so that TBv2 copies once. Measure
  // the drain cost in a profile first.
  const bool stored = buffer->CopyChunkV2Untrusted(
      sequence, chunk.fragments, chunk.num_fragments,
      chunk.payload_flags & kFlagContinuesFromPrevChunk,
      chunk.payload_flags & kFlagContinuesOnNextChunk);
  if (PERFETTO_UNLIKELY(!stored))
    ++trace_buffer_rejected_chunks_;
}

void ServiceRingBufferDrainer::OnDataLoss(WriterID writer_id) {
  // Record the loss in |writer_id|'s sequence so it can be reported through
  // previous_packet_dropped.
  //
  // Drain() counts chunks discarded by the reader in chunks_discarded.
  // Loss reported by the producer does not increase that counter.
  RecordWriterLoss(writer_id);
}

void ServiceRingBufferDrainer::RecordWriterLoss(WriterID writer_id) {
  if (!writer_id || writer_id > kMaxWriterID)
    return;
  // The destination of a lost chunk is unknown. RecordChunkV2DataLoss() changes
  // only a buffer that holds this writer's sequence.
  // - A writer has one destination, so at most one buffer matches.
  // - Before the writer's first stored chunk, no buffer matches.
  delegate_->ForEachRingBufferDestination(
      [this, writer_id](TraceBufferV2& buffer) {
        buffer.RecordChunkV2DataLoss(producer_id_, writer_id);
      });
}

void ServiceRingBufferDrainer::FillStats(
    TraceStats::TracingV2ProducerStats* stats) const {
  PERFETTO_DCHECK_THREAD(thread_checker_);
  using Stats = TraceStats::TracingV2ProducerStats;

  stats->set_chunk_size_bytes(ring_buffer_.chunk_size());
  stats->set_num_chunks(ring_buffer_.num_chunks());
  if (reader_.has_protocol_error())
    stats->set_protocol_error(reader_.protocol_error());

  // A counter that is still zero stays unset, as in the trace buffer stats.
  auto set_counter = [stats](void (Stats::*setter)(uint64_t), uint64_t value) {
    if (value != 0)
      (stats->*setter)(value);
  };

  const SharedRingBufferReader::Stats reader = reader_.GetStats();
  set_counter(&Stats::set_chunks_read, reader.chunks_read);
  set_counter(&Stats::set_payload_bytes, reader.payload_bytes);
  set_counter(&Stats::set_data_loss_chunks, reader.data_loss_chunks);
  set_counter(&Stats::set_data_loss_fragments, reader.data_loss_fragments);
  set_counter(&Stats::set_malformed_chunks, reader.malformed_chunks);
  set_counter(&Stats::set_unsupported_format_chunks,
              reader.unsupported_format_chunks);
  set_counter(&Stats::set_rewrite_requests, reader.rewrite_requests);
  set_counter(&Stats::set_positions_skipped, reader.positions_skipped);

  set_counter(&Stats::set_invalid_writer_chunks, invalid_writer_chunks_);
  set_counter(&Stats::set_invalid_destination_chunks,
              invalid_destination_chunks_);
  set_counter(&Stats::set_trace_buffer_rejected_chunks,
              trace_buffer_rejected_chunks_);
  set_counter(&Stats::set_drain_passes, drain_passes_);
  set_counter(&Stats::set_drain_time_ns, drain_time_ns_);

  // The producer controls these values, so they are reported as they are.
  const SharedRingBuffer::WriterStats writers =
      ring_buffer_.LoadWriterStatsRelaxed();
  set_counter(&Stats::set_writer_dropped_packets, writers.dropped_packets);
  set_counter(&Stats::set_writer_stalls, writers.stalls);
  set_counter(&Stats::set_writer_stall_time_ms, writers.stall_time_ms);
}

}  // namespace perfetto::tracing_v2
