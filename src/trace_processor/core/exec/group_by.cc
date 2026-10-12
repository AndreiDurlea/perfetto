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

#include "src/trace_processor/core/exec/group_by.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "perfetto/base/compiler.h"
#include "perfetto/base/status.h"
#include "perfetto/ext/base/utils.h"
#include "src/trace_processor/core/common/storage_types.h"
#include "src/trace_processor/core/exec/column_view.h"
#include "src/trace_processor/core/util/span.h"

namespace perfetto::trace_processor::core::exec {

GroupBy::GroupBy(std::vector<uint32_t> key_columns)
    : key_columns_(std::move(key_columns)) {}
GroupBy::~GroupBy() = default;
GroupBy::State::~State() {
  group_keys.Clear();
}

std::unique_ptr<Breaker::State> GroupBy::CreateState() const {
  return std::make_unique<State>();
}

namespace {

bool IsIntKey(const ColumnView& column) {
  if (column.kind() == ColumnView::Kind::kSequence) {
    return true;
  }
  if (column.kind() != ColumnView::Kind::kFlat) {
    return false;
  }
  StorageType t = column.type();
  return t.Is<Uint32>() || t.Is<Int32>() || t.Is<Int64>();
}

}  // namespace

bool GroupBy::Consume(const RowBatch& in, Breaker::State& state) const {
  auto& s = static_cast<State&>(state);
  uint32_t count = in.size();
  if (count == 0) {
    return true;
  }

  if (key_columns_.size() == 1) {
    const ColumnView& col = in.column(key_columns_[0]);
    if (!s.int_key_mode) {
      s.int_key_mode = IsIntKey(col) ? 1 : -1;
    } else if (*s.int_key_mode == 1 && !IsIntKey(col)) {
      s.status = base::ErrStatus(
          "GROUP BY: key 1 must be a column holding one type");
      return false;
    }
    if (*s.int_key_mode == 1) {
      uint64_t at = s.groups.size();
      s.groups.resize(at + count);
      uint32_t* groups = s.groups.data() + at;
      auto get_int_group = [&](int64_t val) -> uint32_t {
        if (val >= 0 && val < 65536) {
          auto uval = static_cast<uint32_t>(val);
          if (PERFETTO_UNLIKELY(uval >= s.small_int_groups.size())) {
            s.small_int_groups.resize(uval + 1, State::kNoGroup);
          }
          uint32_t g = s.small_int_groups[uval];
          if (PERFETTO_UNLIKELY(g == State::kNoGroup)) {
            g = s.num_groups++;
            s.small_int_groups[uval] = g;
          }
          return g;
        }
        if (const uint32_t* found = s.int_group_of.Find(val)) {
          return *found;
        }
        uint32_t g = s.num_groups++;
        s.int_group_of.Insert(val, g);
        return g;
      };
      auto get_null_group = [&]() -> uint32_t {
        if (PERFETTO_UNLIKELY(s.null_group == State::kNoGroup)) {
          s.null_group = s.num_groups++;
        }
        return s.null_group;
      };
      auto process_typed = [&](auto getter) {
        const auto* validity = col.validity();
        const auto& sel = col.selection();
        if (!validity && sel.is_range()) {
          uint32_t off = sel.offset();
          for (uint32_t r = 0; r < count; ++r) {
            groups[r] = get_int_group(getter(off + r));
          }
        } else {
          for (uint32_t r = 0; r < count; ++r) {
            uint32_t idx = sel.GetIndex(r);
            if (validity && !validity->is_set(idx)) {
              groups[r] = get_null_group();
            } else {
              groups[r] = get_int_group(getter(idx));
            }
          }
        }
      };
      if (col.kind() == ColumnView::Kind::kSequence) {
        process_typed([](uint32_t idx) { return static_cast<int64_t>(idx); });
      } else if (col.type().Is<Uint32>()) {
        const auto* d = static_cast<const uint32_t*>(col.data());
        process_typed([d](uint32_t idx) { return static_cast<int64_t>(d[idx]); });
      } else if (col.type().Is<Int32>()) {
        const auto* d = static_cast<const int32_t*>(col.data());
        process_typed([d](uint32_t idx) { return static_cast<int64_t>(d[idx]); });
      } else {
        const auto* d = static_cast<const int64_t*>(col.data());
        process_typed([d](uint32_t idx) { return d[idx]; });
      }
      s.status = s.rows.Append(in);
      return s.status.ok();
    }
  }

  if (std::optional<uint32_t> bad = s.keys.Encode(in, key_columns_)) {
    s.status = base::ErrStatus(
        "GROUP BY: key %u must be a column holding one type", *bad + 1);
    return false;
  }
  uint64_t at = s.groups.size();
  s.groups.resize(at + count);
  uint32_t* groups = s.groups.data() + at;
  for (uint32_t row = 0; row < count; ++row) {
    std::string_view key = s.keys.Key(row);
    uint32_t group;
    if (const uint32_t* found = s.group_of.Find(key)) {
      group = *found;
    } else {
      group = s.num_groups++;
      void* copy;
      s.group_keys.Alloc(static_cast<uint32_t>(base::AlignUp<8>(key.size())),
                         &copy);
      memcpy(copy, key.data(), key.size());
      s.group_of.Insert(
          std::string_view(static_cast<const char*>(copy), key.size()), group);
    }
    groups[row] = group;
  }
  s.status = s.rows.Append(in);
  return s.status.ok();
}

bool GroupBy::Finalize(Breaker::State& state) const {
  auto& s = static_cast<State&>(state);
  auto groups = s.num_groups;
  auto rows = static_cast<uint32_t>(s.groups.size());
  const uint32_t* group_of_row = s.groups.data();

  // Since group numbers are assigned in order of first appearance (0, 1, 2...),
  // if `group_of_row` is non-decreasing then identical groups are already
  // contiguous in arrival order!
  bool already_grouped = true;
  for (uint32_t r = 1; r < rows; ++r) {
    if (group_of_row[r - 1] > group_of_row[r]) {
      already_grouped = false;
      break;
    }
  }
  if (already_grouped) {
    s.order.clear();
    return true;
  }

  // A counting sort on the group.
  s.next.clear();
  s.next.push_back_multiple(0, groups);
  s.order.resize(rows);
  s.ordered_groups.resize(rows);
  uint32_t* PERFETTO_RESTRICT next = s.next.data();
  uint32_t* PERFETTO_RESTRICT order = s.order.data();
  uint32_t* PERFETTO_RESTRICT ordered_groups = s.ordered_groups.data();
  for (uint32_t row = 0; row < rows; ++row) {
    ++next[group_of_row[row]];
  }
  uint32_t start = 0;
  for (uint32_t group = 0; group < groups; ++group) {
    uint32_t end = start + next[group];
    std::fill(ordered_groups + start, ordered_groups + end, group);
    next[group] = start;
    start = end;
  }
  for (uint32_t row = 0; row < rows; ++row) {
    order[next[group_of_row[row]]++] = row;
  }
  return true;
}

bool GroupBy::Serve(RowBatch& out, Breaker::State& state) const {
  auto& s = static_cast<State&>(state);
  if (s.emitted == s.rows.size()) {
    return false;
  }
  uint32_t count = std::min(kMaxBatchRows, s.rows.size() - s.emitted);
  if (s.order.empty()) {
    count = s.rows.View(&out, s.emitted, count);
    ColumnView group =
        ColumnView::Reference(StorageType{Uint32{}}, s.groups.data());
    group.SetRange(s.emitted);
    out.AddColumn(group);
  } else {
    const uint32_t* begin = s.order.data() + s.emitted;
    count = s.rows.View(&out, Span<const uint32_t>(begin, begin + count));
    ColumnView group =
        ColumnView::Reference(StorageType{Uint32{}}, s.ordered_groups.data());
    group.SetRange(s.emitted);
    out.AddColumn(group);
  }
  s.emitted += count;
  return true;
}

void GroupBy::State::Reset() {
  Breaker::State::Reset();
  State& s = *this;
  s.group_of.Clear();
  s.group_keys.Clear();
  s.int_key_mode = std::nullopt;
  s.num_groups = 0;
  s.null_group = kNoGroup;
  s.small_int_groups.clear();
  s.int_group_of.Clear();
  s.groups.clear();
  s.rows.Clear();
  s.order.clear();
  s.ordered_groups.clear();
  s.emitted = 0;
}

}  // namespace perfetto::trace_processor::core::exec
