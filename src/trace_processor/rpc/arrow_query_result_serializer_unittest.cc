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

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "perfetto/base/status.h"
#include "perfetto/trace_processor/trace_processor.h"
#include "src/trace_processor/core/dataframe/arrow_internal.h"
#include "src/trace_processor/util/flatbuffer_reader.h"
#include "test/gtest_and_gmock.h"

namespace perfetto::trace_processor {
namespace {

using namespace core::dataframe::arrow_internal;
using R = util::FlatBufferReader;

constexpr uint8_t kNull = 1;
constexpr uint8_t kBinary = 4;

// A tiny Arrow IPC stream reader, just enough to check what the serializer
// wrote: field names and types, and the values of each record batch.
struct Field {
  std::string name;
  uint8_t type;
  int32_t bit_width = 0;
};

struct Batch {
  int64_t rows = 0;
  std::vector<FieldNode> nodes;
  std::vector<ArrowBuffer> buffers;
  const uint8_t* body = nullptr;
};

struct Stream {
  std::vector<Field> fields;
  std::vector<Batch> batches;
  bool saw_eos = false;
};

Stream Parse(const std::vector<uint8_t>& data) {
  Stream s;
  size_t pos = 0;
  while (pos + sizeof(MessagePrefix) <= data.size()) {
    MessagePrefix prefix;
    memcpy(&prefix, data.data() + pos, sizeof(prefix));
    EXPECT_EQ(prefix.continuation, kContinuation);
    pos += sizeof(prefix);
    if (prefix.metadata_size == 0) {
      s.saw_eos = true;
      break;
    }
    EXPECT_EQ((pos + static_cast<size_t>(prefix.metadata_size)) % 8, 0u);
    std::optional<R> msg = R::GetRoot(
        data.data() + pos, static_cast<uint32_t>(prefix.metadata_size));
    EXPECT_TRUE(msg.has_value());
    pos += static_cast<size_t>(prefix.metadata_size);
    auto body_length = msg->Scalar<int64_t>(message_field::kBodyLength);
    uint8_t header_type = msg->Scalar<uint8_t>(message_field::kHeaderType);
    R header = msg->Table(message_field::kHeader);
    if (header_type == kHeaderSchema) {
      auto fields = header.VecTable(schema_field::kFields);
      for (uint32_t i = 0; i < fields.size(); ++i) {
        R f = fields[i];
        Field field{std::string(f.String(field_field::kName)),
                    f.Scalar<uint8_t>(field_field::kTypeType)};
        if (field.type == kTypeInt) {
          field.bit_width =
              f.Table(field_field::kType).Scalar<int32_t>(int_field::kBitWidth);
        }
        s.fields.push_back(field);
      }
    } else {
      EXPECT_EQ(header_type, kHeaderRecordBatch);
      Batch b;
      b.rows = header.Scalar<int64_t>(record_batch_field::kLength);
      auto nodes = header.VecScalar<FieldNode>(record_batch_field::kNodes);
      for (uint32_t i = 0; i < nodes.size(); ++i) {
        b.nodes.push_back(nodes[i]);
      }
      auto buffers = header.VecScalar<ArrowBuffer>(record_batch_field::kBuffers);
      for (uint32_t i = 0; i < buffers.size(); ++i) {
        b.buffers.push_back(buffers[i]);
      }
      b.body = data.data() + pos;
      s.batches.push_back(b);
    }
    pos += static_cast<size_t>(body_length);
  }
  EXPECT_EQ(pos, data.size());
  return s;
}

// Cell accessors over the buffers of |batch|; |buf| is the index of the
// column's first buffer.
bool IsValid(const Batch& b, size_t buf, int64_t row) {
  if (b.buffers[buf].length == 0) {
    return true;
  }
  const uint8_t* bitmap = b.body + b.buffers[buf].offset;
  return bitmap[row / 8] & (1 << (row % 8));
}

template <typename T>
T Value(const Batch& b, size_t buf, int64_t row) {
  T v;
  memcpy(&v, b.body + b.buffers[buf + 1].offset + row * int64_t{sizeof(T)},
         sizeof(T));
  return v;
}

std::string Text(const Batch& b, size_t buf, int64_t row) {
  const auto* offsets =
      reinterpret_cast<const int32_t*>(b.body + b.buffers[buf + 1].offset);
  const auto* chars =
      reinterpret_cast<const char*>(b.body + b.buffers[buf + 2].offset);
  return std::string(chars + offsets[row],
                     static_cast<size_t>(offsets[row + 1] - offsets[row]));
}

class ArrowQueryResultSerializerTest : public ::testing::Test {
 protected:
  ArrowQueryResultSerializerTest()
      : tp_(TraceProcessor::CreateInstance(Config())) {
    tp_->NotifyEndOfFile();
  }

  std::vector<uint8_t> Serialize(const std::string& sql,
                                 uint32_t batch_rows = 1024) {
    ArrowQueryResultSerializer serializer(tp_->ExecuteQuery(sql), batch_rows);
    EXPECT_TRUE(serializer.Run().ok());
    std::vector<uint8_t> out;
    EXPECT_TRUE(serializer
                    .Write([&](const uint8_t* d, size_t n) {
                      out.insert(out.end(), d, d + n);
                      return base::OkStatus();
                    })
                    .ok());
    return out;
  }

  std::unique_ptr<TraceProcessor> tp_;
};

TEST_F(ArrowQueryResultSerializerTest, TypesAndValues) {
  Stream s = Parse(
      Serialize("SELECT 1 AS i, 2.5 AS d, 'abc' AS s, NULL AS n, x'0102' AS b "
                "UNION ALL SELECT NULL, NULL, NULL, NULL, NULL"));
  ASSERT_TRUE(s.saw_eos);
  ASSERT_EQ(s.fields.size(), 5u);
  EXPECT_EQ(s.fields[0].name, "i");
  EXPECT_EQ(s.fields[0].type, kTypeInt);
  EXPECT_EQ(s.fields[0].bit_width, 64);
  EXPECT_EQ(s.fields[1].type, kTypeFloatingPoint);
  EXPECT_EQ(s.fields[2].type, kTypeUtf8);
  EXPECT_EQ(s.fields[3].type, kNull);
  EXPECT_EQ(s.fields[4].type, kBinary);

  ASSERT_EQ(s.batches.size(), 1u);
  const Batch& b = s.batches[0];
  ASSERT_EQ(b.rows, 2);
  // Buffers: i(2) d(2) s(3) n(0) b(3).
  ASSERT_EQ(b.buffers.size(), 10u);
  EXPECT_EQ(Value<int64_t>(b, 0, 0), 1);
  EXPECT_FALSE(IsValid(b, 0, 1));
  EXPECT_EQ(Value<double>(b, 2, 0), 2.5);
  EXPECT_EQ(Text(b, 4, 0), "abc");
  EXPECT_FALSE(IsValid(b, 4, 1));
  EXPECT_EQ(b.nodes[3].null_count, 2);
  EXPECT_EQ(Text(b, 7, 0), "\x01\x02");
}

TEST_F(ArrowQueryResultSerializerTest, MixedTypesAreWidened) {
  Stream s = Parse(Serialize(
      "SELECT 1 AS a, 1 AS b UNION ALL SELECT 2.5, 'x' UNION ALL SELECT 3, 4"));
  ASSERT_EQ(s.fields.size(), 2u);
  EXPECT_EQ(s.fields[0].type, kTypeFloatingPoint);
  EXPECT_EQ(s.fields[1].type, kTypeUtf8);
  const Batch& b = s.batches[0];
  EXPECT_EQ(Value<double>(b, 0, 0), 1.0);
  EXPECT_EQ(Value<double>(b, 0, 1), 2.5);
  EXPECT_EQ(Value<double>(b, 0, 2), 3.0);
  EXPECT_EQ(Text(b, 2, 0), "1");
  EXPECT_EQ(Text(b, 2, 1), "x");
  EXPECT_EQ(Text(b, 2, 2), "4");
}

TEST_F(ArrowQueryResultSerializerTest, SplitsIntoBatches) {
  Stream s = Parse(Serialize(
      "WITH RECURSIVE n(x) AS (SELECT 0 UNION ALL SELECT x + 1 FROM n "
      "WHERE x < 2499) SELECT x, 's' || x AS t FROM n",
      /*batch_rows=*/1000));
  ASSERT_EQ(s.batches.size(), 3u);
  EXPECT_EQ(s.batches[0].rows, 1000);
  EXPECT_EQ(s.batches[2].rows, 500);
  // String offsets restart at zero in every batch.
  EXPECT_EQ(Value<int64_t>(s.batches[1], 0, 0), 1000);
  EXPECT_EQ(Text(s.batches[1], 2, 0), "s1000");
  EXPECT_EQ(Text(s.batches[2], 2, 499), "s2499");
}

TEST_F(ArrowQueryResultSerializerTest, EmptyResultHasSchemaOnly) {
  Stream s = Parse(Serialize("SELECT 1 AS a WHERE 0"));
  ASSERT_EQ(s.fields.size(), 1u);
  EXPECT_EQ(s.fields[0].type, kNull);
  EXPECT_TRUE(s.batches.empty());
  EXPECT_TRUE(s.saw_eos);
}

TEST_F(ArrowQueryResultSerializerTest, StatementWithoutOutput) {
  Stream s = Parse(Serialize("CREATE PERFETTO TABLE t AS SELECT 1 AS x"));
  EXPECT_TRUE(s.fields.empty());
  EXPECT_TRUE(s.batches.empty());
  EXPECT_TRUE(s.saw_eos);
}

TEST_F(ArrowQueryResultSerializerTest, QueryErrorIsReturned) {
  ArrowQueryResultSerializer serializer(
      tp_->ExecuteQuery("SELECT * FROM no_such_table"));
  base::Status status = serializer.Run();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(), testing::HasSubstr("no_such_table"));
}

}  // namespace
}  // namespace perfetto::trace_processor
