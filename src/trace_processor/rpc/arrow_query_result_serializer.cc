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

#include "src/trace_processor/rpc/arrow_query_result_serializer.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "perfetto/base/logging.h"
#include "perfetto/base/status.h"
#include "perfetto/ext/base/status_macros.h"
#include "perfetto/trace_processor/basic_types.h"
#include "perfetto/trace_processor/iterator.h"
#include "src/trace_processor/core/dataframe/arrow_internal.h"
#include "src/trace_processor/util/flatbuffer_writer.h"

namespace perfetto::trace_processor {

namespace {

using W = util::FlatBufferWriter;
using namespace core::dataframe::arrow_internal;

// Arrow Type union members not used by the dataframe serializer.
constexpr uint8_t kTypeNull = 1;
constexpr uint8_t kTypeBinary = 4;

// Offsets of variable-width columns are int32, so a batch must keep each
// column's byte payload below this.
constexpr uint64_t kMaxBatchBytes = 1ull << 30;

constexpr uint8_t kZeros[kArrowAlignment] = {};

// Writes |v| as text the way it is usually shown to users: integers in base
// 10, doubles with enough digits to round-trip.
size_t FormatInt(int64_t v, char* buf, size_t size) {
  return static_cast<size_t>(snprintf(buf, size, "%" PRId64, v));
}

size_t FormatDouble(double v, char* buf, size_t size) {
  return static_cast<size_t>(snprintf(buf, size, "%.17g", v));
}

W::Offset BuildEmptyTable(W& w) {
  w.StartTable();
  return w.EndTable();
}

W::Offset BuildMessage(W& w,
                       uint8_t header_type,
                       W::Offset header,
                       int64_t body_length) {
  w.StartTable();
  w.FieldI16(message_field::kVersion, kMetadataV5);
  w.FieldU8(message_field::kHeaderType, header_type);
  w.FieldOffset(message_field::kHeader, header);
  w.FieldI64(message_field::kBodyLength, body_length);
  return w.EndTable();
}

// Writes one IPC message: prefix, flatbuffer metadata padded so the body
// starts aligned, then nothing else (the caller streams the body).
base::Status WriteMessageHeader(W* w,
                                W::Offset root,
                                const ArrowQueryResultSerializer::WriteFn& out) {
  w->Finish(root);
  uint64_t padded = PaddedMetadataSize(w->size());
  MessagePrefix prefix{kContinuation, static_cast<int32_t>(padded)};
  RETURN_IF_ERROR(out(reinterpret_cast<const uint8_t*>(&prefix),
                      sizeof(prefix)));
  RETURN_IF_ERROR(out(w->data(), w->size()));
  size_t pad = static_cast<size_t>(padded - w->size());
  return pad ? out(kZeros, pad) : base::OkStatus();
}

base::Status WritePadded(const uint8_t* data,
                         uint64_t size,
                         const ArrowQueryResultSerializer::WriteFn& out) {
  if (size) {
    RETURN_IF_ERROR(out(data, static_cast<size_t>(size)));
  }
  size_t pad = static_cast<size_t>(AlignToArrow(size) - size);
  return pad ? out(kZeros, pad) : base::OkStatus();
}

}  // namespace

ArrowQueryResultSerializer::ArrowQueryResultSerializer(Iterator it,
                                                       uint32_t max_batch_rows)
    : it_(std::move(it)), max_batch_rows_(std::max(max_batch_rows, 1u)) {}

ArrowQueryResultSerializer::~ArrowQueryResultSerializer() = default;

base::Status ArrowQueryResultSerializer::Run() {
  PERFETTO_CHECK(!ran_);
  ran_ = true;
  uint32_t cols = it_.ColumnCount();
  columns_.resize(cols);
  for (uint32_t c = 0; c < cols; ++c) {
    columns_[c].name = it_.GetColumnName(c);
  }
  while (it_.Next()) {
    for (uint32_t c = 0; c < cols; ++c) {
      AppendCell(&columns_[c], it_.Get(c));
    }
    ++rows_;
  }
  return it_.Status();
}

void ArrowQueryResultSerializer::SetKind(Column* col, Kind kind) {
  size_t rows = col->valid.size();
  if (col->kind == Kind::kNull) {
    switch (kind) {
      case Kind::kInt64:
        col->ints.assign(rows, 0);
        break;
      case Kind::kDouble:
        col->doubles.assign(rows, 0);
        break;
      case Kind::kString:
      case Kind::kBinary:
        col->ends.assign(rows, 0);
        break;
      case Kind::kNull:
        break;
    }
  } else if (col->kind == Kind::kInt64 && kind == Kind::kDouble) {
    col->doubles.reserve(col->ints.capacity());
    for (int64_t v : col->ints) {
      col->doubles.push_back(static_cast<double>(v));
    }
    col->ints = {};
  } else {
    PERFETTO_DCHECK(kind == col->kind);
  }
  col->kind = kind;
}

void ArrowQueryResultSerializer::NumbersToText(Column* col) {
  PERFETTO_DCHECK(col->kind == Kind::kInt64 || col->kind == Kind::kDouble);
  size_t rows = col->valid.size();
  col->ends.reserve(rows);
  char buf[32];
  for (size_t i = 0; i < rows; ++i) {
    if (col->valid[i]) {
      size_t len = col->kind == Kind::kInt64
                       ? FormatInt(col->ints[i], buf, sizeof(buf))
                       : FormatDouble(col->doubles[i], buf, sizeof(buf));
      col->bytes.append(buf, len);
    }
    col->ends.push_back(col->bytes.size());
  }
  col->ints = {};
  col->doubles = {};
  col->kind = Kind::kString;
}

void ArrowQueryResultSerializer::AppendText(Column* col,
                                            const char* data,
                                            size_t size) {
  col->bytes.append(data, size);
  col->ends.push_back(col->bytes.size());
}

void ArrowQueryResultSerializer::AppendCell(Column* col, const SqlValue& v) {
  char buf[32];
  switch (v.type) {
    case SqlValue::Type::kNull:
      switch (col->kind) {
        case Kind::kInt64:
          col->ints.push_back(0);
          break;
        case Kind::kDouble:
          col->doubles.push_back(0);
          break;
        case Kind::kString:
        case Kind::kBinary:
          col->ends.push_back(col->bytes.size());
          break;
        case Kind::kNull:
          break;
      }
      col->valid.push_back(0);
      ++col->null_count;
      return;
    case SqlValue::Type::kLong:
      if (col->kind == Kind::kNull) {
        SetKind(col, Kind::kInt64);
      }
      if (col->kind == Kind::kInt64) {
        col->ints.push_back(v.long_value);
      } else if (col->kind == Kind::kDouble) {
        col->doubles.push_back(static_cast<double>(v.long_value));
      } else {
        AppendText(col, buf, FormatInt(v.long_value, buf, sizeof(buf)));
      }
      break;
    case SqlValue::Type::kDouble:
      if (col->kind == Kind::kNull || col->kind == Kind::kInt64) {
        SetKind(col, Kind::kDouble);
      }
      if (col->kind == Kind::kDouble) {
        col->doubles.push_back(v.double_value);
      } else {
        AppendText(col, buf, FormatDouble(v.double_value, buf, sizeof(buf)));
      }
      break;
    case SqlValue::Type::kString:
      if (col->kind == Kind::kNull) {
        SetKind(col, Kind::kString);
      } else if (col->kind == Kind::kInt64 || col->kind == Kind::kDouble) {
        NumbersToText(col);
      }
      AppendText(col, v.string_value, strlen(v.string_value));
      break;
    case SqlValue::Type::kBytes:
      if (col->kind == Kind::kNull) {
        SetKind(col, Kind::kBinary);
      } else if (col->kind == Kind::kInt64 || col->kind == Kind::kDouble) {
        NumbersToText(col);
      }
      col->kind = Kind::kBinary;
      AppendText(col, static_cast<const char*>(v.bytes_value), v.bytes_count);
      break;
  }
  col->valid.push_back(1);
}

base::Status ArrowQueryResultSerializer::WriteSchema(const WriteFn& out) {
  W w;
  std::vector<W::Offset> fields;
  fields.reserve(columns_.size());
  for (const Column& col : columns_) {
    uint8_t type_type;
    W::Offset type;
    switch (col.kind) {
      case Kind::kNull:
        type_type = kTypeNull;
        type = BuildEmptyTable(w);
        break;
      case Kind::kInt64:
        type_type = kTypeInt;
        w.StartTable();
        w.FieldI32(int_field::kBitWidth, 64);
        w.FieldBool(int_field::kIsSigned, true);
        type = w.EndTable();
        break;
      case Kind::kDouble:
        type_type = kTypeFloatingPoint;
        w.StartTable();
        w.FieldI16(floating_point_field::kPrecision, kPrecisionDouble);
        type = w.EndTable();
        break;
      case Kind::kString:
        type_type = kTypeUtf8;
        type = BuildEmptyTable(w);
        break;
      case Kind::kBinary:
        type_type = kTypeBinary;
        type = BuildEmptyTable(w);
        break;
    }
    auto name = w.WriteString(col.name);
    w.StartTable();
    w.FieldOffset(field_field::kName, name);
    w.FieldBool(field_field::kNullable, true);
    w.FieldU8(field_field::kTypeType, type_type);
    w.FieldOffset(field_field::kType, type);
    fields.push_back(w.EndTable());
  }
  auto field_vector =
      w.WriteVecOffsets(fields.data(), static_cast<uint32_t>(fields.size()));
  w.StartTable();
  w.FieldI16(schema_field::kEndianness, kEndiannessLittle);
  w.FieldOffset(schema_field::kFields, field_vector);
  auto schema = w.EndTable();
  return WriteMessageHeader(&w, BuildMessage(w, kHeaderSchema, schema, 0), out);
}

base::Status ArrowQueryResultSerializer::WriteBatch(uint64_t begin,
                                                    uint64_t end,
                                                    const WriteFn& out) {
  const uint64_t rows = end - begin;
  std::vector<FieldNode> nodes;
  std::vector<ArrowBuffer> buffers;
  uint64_t cursor = 0;
  auto add_buffer = [&](uint64_t length) {
    buffers.push_back({static_cast<int64_t>(cursor),
                       static_cast<int64_t>(length)});
    cursor += AlignToArrow(length);
  };

  // Plan the body layout: Arrow metadata lists every buffer before the body.
  std::vector<uint64_t> nulls(columns_.size());
  for (size_t c = 0; c < columns_.size(); ++c) {
    const Column& col = columns_[c];
    if (col.kind == Kind::kNull) {
      nodes.push_back({static_cast<int64_t>(rows), static_cast<int64_t>(rows)});
      continue;
    }
    if (col.null_count) {
      nulls[c] = static_cast<uint64_t>(
          std::count(col.valid.begin() + static_cast<ptrdiff_t>(begin),
                     col.valid.begin() + static_cast<ptrdiff_t>(end), 0));
    }
    nodes.push_back(
        {static_cast<int64_t>(rows), static_cast<int64_t>(nulls[c])});
    add_buffer(nulls[c] ? ValidityBufferSize(static_cast<uint32_t>(rows)) : 0);
    if (col.kind == Kind::kInt64 || col.kind == Kind::kDouble) {
      add_buffer(rows * sizeof(int64_t));
    } else {
      uint64_t first = begin ? col.ends[begin - 1] : 0;
      add_buffer((rows + 1) * sizeof(int32_t));
      add_buffer(col.ends[end - 1] - first);
    }
  }

  W w;
  auto buffer_vector =
      w.WriteVecStruct(buffers.data(), sizeof(ArrowBuffer),
                       static_cast<uint32_t>(buffers.size()), alignof(int64_t));
  auto node_vector =
      w.WriteVecStruct(nodes.data(), sizeof(FieldNode),
                       static_cast<uint32_t>(nodes.size()), alignof(int64_t));
  w.StartTable();
  w.FieldI64(record_batch_field::kLength, static_cast<int64_t>(rows));
  w.FieldOffset(record_batch_field::kNodes, node_vector);
  w.FieldOffset(record_batch_field::kBuffers, buffer_vector);
  auto batch = w.EndTable();
  RETURN_IF_ERROR(WriteMessageHeader(
      &w,
      BuildMessage(w, kHeaderRecordBatch, batch, static_cast<int64_t>(cursor)),
      out));

  // Stream the body in the order planned above.
  for (size_t c = 0; c < columns_.size(); ++c) {
    const Column& col = columns_[c];
    if (col.kind == Kind::kNull) {
      continue;
    }
    if (nulls[c]) {
      scratch_.assign(ValidityBufferSize(static_cast<uint32_t>(rows)), 0);
      for (uint64_t i = 0; i < rows; ++i) {
        if (col.valid[begin + i]) {
          scratch_[i / kBitsPerByte] |= BitmapMask(static_cast<uint32_t>(i));
        }
      }
      RETURN_IF_ERROR(WritePadded(scratch_.data(), scratch_.size(), out));
    }
    if (col.kind == Kind::kInt64) {
      RETURN_IF_ERROR(WritePadded(
          reinterpret_cast<const uint8_t*>(col.ints.data() + begin),
          rows * sizeof(int64_t), out));
    } else if (col.kind == Kind::kDouble) {
      RETURN_IF_ERROR(WritePadded(
          reinterpret_cast<const uint8_t*>(col.doubles.data() + begin),
          rows * sizeof(double), out));
    } else {
      uint64_t first = begin ? col.ends[begin - 1] : 0;
      scratch_.resize((rows + 1) * sizeof(int32_t));
      auto* offsets = reinterpret_cast<int32_t*>(scratch_.data());
      offsets[0] = 0;
      for (uint64_t i = 0; i < rows; ++i) {
        offsets[i + 1] = static_cast<int32_t>(col.ends[begin + i] - first);
      }
      RETURN_IF_ERROR(WritePadded(scratch_.data(), scratch_.size(), out));
      RETURN_IF_ERROR(
          WritePadded(reinterpret_cast<const uint8_t*>(col.bytes.data()) + first,
                      col.ends[end - 1] - first, out));
    }
  }
  return base::OkStatus();
}

base::Status ArrowQueryResultSerializer::Write(const WriteFn& out) {
  PERFETTO_CHECK(ran_);
  RETURN_IF_ERROR(WriteSchema(out));
  // Returns the payload size of the largest variable-width column in
  // [begin, end).
  auto max_bytes = [this](uint64_t begin, uint64_t end) {
    uint64_t max = 0;
    for (const Column& col : columns_) {
      if (col.kind == Kind::kString || col.kind == Kind::kBinary) {
        uint64_t first = begin ? col.ends[begin - 1] : 0;
        max = std::max(max, col.ends[end - 1] - first);
      }
    }
    return max;
  };
  for (uint64_t begin = 0; begin < rows_;) {
    uint64_t end = std::min(rows_, begin + max_batch_rows_);
    // Halve the batch until its strings fit in int32 offsets.
    while (max_bytes(begin, end) > kMaxBatchBytes) {
      if (end - begin == 1) {
        return base::ErrStatus("Arrow export: a single cell exceeds %" PRIu64
                               " bytes",
                               kMaxBatchBytes);
      }
      end = begin + (end - begin) / 2;
    }
    RETURN_IF_ERROR(WriteBatch(begin, end, out));
    begin = end;
  }
  const MessagePrefix eos{kContinuation, 0};
  return out(reinterpret_cast<const uint8_t*>(&eos), sizeof(eos));
}

}  // namespace perfetto::trace_processor
