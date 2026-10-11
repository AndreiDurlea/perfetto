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

#ifndef SRC_TRACE_PROCESSOR_RPC_ARROW_QUERY_RESULT_SERIALIZER_H_
#define SRC_TRACE_PROCESSOR_RPC_ARROW_QUERY_RESULT_SERIALIZER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "perfetto/base/status.h"
#include "perfetto/trace_processor/iterator.h"

namespace perfetto::trace_processor {

// Serializes the result of a query as an Arrow IPC stream: a schema message,
// one or more record batches and the end-of-stream marker. Readers such as
// pyarrow (pa.ipc.open_stream) and DuckDB consume it without per-cell work, so
// this is the fast path for pulling large results out of trace processor.
//
// SQLite values are dynamically typed, so the result is buffered in columnar
// form first and each column gets the narrowest Arrow type that holds all of
// its values:
//   only NULLs                 -> null
//   integers                   -> int64
//   integers and/or doubles    -> float64
//   any string                 -> utf8 (numbers are converted to text)
//   any blob                   -> binary
// Buffering needs memory proportional to the result size, which is what lets
// the types be exact; the existing protobuf path streams instead.
class ArrowQueryResultSerializer {
 public:
  using WriteFn = std::function<base::Status(const uint8_t*, size_t)>;

  static constexpr uint32_t kDefaultMaxBatchRows = 128 * 1024;

  explicit ArrowQueryResultSerializer(
      Iterator,
      uint32_t max_batch_rows = kDefaultMaxBatchRows);
  ~ArrowQueryResultSerializer();

  ArrowQueryResultSerializer(const ArrowQueryResultSerializer&) = delete;
  ArrowQueryResultSerializer& operator=(const ArrowQueryResultSerializer&) =
      delete;

  // Runs the query to completion and buffers its result. Returns the query
  // error, if any. Must be called once, before Write().
  base::Status Run();

  // Writes the buffered result as an Arrow IPC stream through |write|.
  base::Status Write(const WriteFn& write);

  uint64_t row_count() const { return rows_; }

 private:
  enum class Kind : uint8_t { kNull, kInt64, kDouble, kString, kBinary };

  struct Column {
    std::string name;
    Kind kind = Kind::kNull;
    uint64_t null_count = 0;
    // One byte per row; packed into an Arrow bitmap when written.
    std::vector<uint8_t> valid;
    std::vector<int64_t> ints;
    std::vector<double> doubles;
    // For kString and kBinary: end offset of each row in |bytes|.
    std::vector<uint64_t> ends;
    std::string bytes;
  };

  void AppendCell(Column*, const SqlValue&);
  void SetKind(Column*, Kind);
  void NumbersToText(Column*);
  void AppendText(Column*, const char*, size_t);

  base::Status WriteSchema(const WriteFn&);
  base::Status WriteBatch(uint64_t begin, uint64_t end, const WriteFn&);

  Iterator it_;
  uint32_t max_batch_rows_;
  uint64_t rows_ = 0;
  bool ran_ = false;
  std::vector<Column> columns_;
  std::vector<uint8_t> scratch_;
};

}  // namespace perfetto::trace_processor

#endif  // SRC_TRACE_PROCESSOR_RPC_ARROW_QUERY_RESULT_SERIALIZER_H_
