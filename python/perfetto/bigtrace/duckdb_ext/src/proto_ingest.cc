// Copyright (C) 2026 The Android Open Source Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common.h"  // no-include-violation-check

namespace {

constexpr uint32_t kResultBatchField = 3;
constexpr uint32_t kBatchCellsField = 1;
constexpr uint32_t kBatchVarintsField = 2;
constexpr uint32_t kBatchFloat64sField = 3;
constexpr uint32_t kBatchBlobsField = 4;
constexpr uint32_t kBatchStringsField = 5;

constexpr uint8_t kCellNull = 1;
constexpr uint8_t kCellVarint = 2;
constexpr uint8_t kCellFloat64 = 3;
constexpr uint8_t kCellString = 4;
constexpr uint8_t kCellBlob = 5;

enum class ColKind : uint8_t { I64, F64, BOOL, STR };

struct StrSlice {
  const char* ptr;
  size_t len;
};

inline bool ReadVarint(const uint8_t*& p, const uint8_t* end, uint64_t& out) {
  if (__builtin_expect(p < end && *p < 0x80, 1)) {
    out = *p++;
    return true;
  }
  if (__builtin_expect(p + 1 < end && p[1] < 0x80, 1)) {
    out = (static_cast<uint64_t>(p[0]) & 0x7F) |
          (static_cast<uint64_t>(p[1]) << 7);
    p += 2;
    return true;
  }
  uint64_t val = 0;
  uint32_t shift = 0;
  while (p < end && shift < 64) {
    uint8_t b = *p++;
    val |= static_cast<uint64_t>(b & 0x7F) << shift;
    if (b < 0x80) {
      out = val;
      return true;
    }
    shift += 7;
  }
  return false;
}

static std::string Base64Encode(const uint8_t* data, size_t len) {
  static constexpr char kTbl[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.resize(((len + 2) / 3) * 4);
  char* dst = out.data();
  size_t i = 0;
  for (; i + 3 <= len; i += 3) {
    uint32_t v = (static_cast<uint32_t>(data[i]) << 16) |
                 (static_cast<uint32_t>(data[i + 1]) << 8) |
                 static_cast<uint32_t>(data[i + 2]);
    *dst++ = kTbl[(v >> 18) & 0x3F];
    *dst++ = kTbl[(v >> 12) & 0x3F];
    *dst++ = kTbl[(v >> 6) & 0x3F];
    *dst++ = kTbl[v & 0x3F];
  }
  if (i < len) {
    uint32_t v = static_cast<uint32_t>(data[i]) << 16;
    if (i + 1 < len)
      v |= static_cast<uint32_t>(data[i + 1]) << 8;
    *dst++ = kTbl[(v >> 18) & 0x3F];
    *dst++ = kTbl[(v >> 12) & 0x3F];
    *dst++ = (i + 1 < len) ? kTbl[(v >> 6) & 0x3F] : '=';
    *dst++ = '=';
  }
  return out;
}

struct BindData {
  std::vector<std::string> paths;
  std::vector<std::string> col_names;
  std::vector<ColKind> col_kinds;
  bool unlink_files = true;
  bool all_i64 = false;
};

struct GlobalState {
  std::atomic<size_t> next_file{0};
};

struct BatchCursor {
  const uint8_t* cells = nullptr;
  size_t num_rows = 0;
  size_t row = 0;
  const uint8_t* var_p = nullptr;
  const uint8_t* var_end = nullptr;
  const uint8_t* f64_p = nullptr;
  const uint8_t* f64_end = nullptr;
  std::vector<StrSlice> strs;
  size_t str_idx = 0;
  std::vector<std::pair<const uint8_t*, size_t>> blobs;
  size_t blob_idx = 0;
};

// Reusable 4 MiB per-thread read buffer: ZERO per-file allocations,
// ZERO std::vector memset(0), ZERO std::fopen global locks, ZERO __madvise!
class LocalState {
 public:
  static constexpr size_t kInitialBufCap = 4ULL << 20;

  LocalState() : buf_cap_(kInitialBufCap), buf_(new uint8_t[kInitialBufCap]) {}

  ~LocalState() { CloseFd(); }

  int32_t cur_file_ord = 0;
  BatchCursor cur_batch;
  bool has_batch = false;

  bool NextBatch(size_t n_cols) {
    has_batch = false;
    while (true) {
      if (!EnsureBytes(20)) {
        if (buf_pos_ >= buf_len_)
          return false;
      }
      const uint8_t* p = buf_.get() + buf_pos_;
      const uint8_t* end = buf_.get() + buf_len_;
      uint64_t tag = 0;
      if (!ReadVarint(p, end, tag)) {
        buf_pos_ = buf_len_;
        return false;
      }
      uint32_t field = static_cast<uint32_t>(tag >> 3);
      uint32_t wt = static_cast<uint32_t>(tag & 7);
      if (wt == 0) {
        uint64_t dummy = 0;
        if (!ReadVarint(p, end, dummy)) {
          buf_pos_ = buf_len_;
          return false;
        }
        buf_pos_ = static_cast<size_t>(p - buf_.get());
      } else if (wt == 2) {
        uint64_t len = 0;
        if (!ReadVarint(p, end, len)) {
          buf_pos_ = buf_len_;
          return false;
        }
        buf_pos_ = static_cast<size_t>(p - buf_.get());
        if (!EnsureBytes(static_cast<size_t>(len))) {
          buf_pos_ = buf_len_;
          return false;
        }
        const uint8_t* b_start = buf_.get() + buf_pos_;
        const uint8_t* b_end = b_start + len;
        buf_pos_ += static_cast<size_t>(len);
        if (field == kResultBatchField) {
          ParseCellsBatch(b_start, b_end, n_cols);
          if (cur_batch.num_rows > 0) {
            has_batch = true;
            return true;
          }
        }
      } else if (wt == 1) {
        size_t hdr = static_cast<size_t>(p - buf_.get());
        buf_pos_ = hdr;
        if (!EnsureBytes(8))
          return false;
        buf_pos_ += 8;
      } else if (wt == 5) {
        size_t hdr = static_cast<size_t>(p - buf_.get());
        buf_pos_ = hdr;
        if (!EnsureBytes(4))
          return false;
        buf_pos_ += 4;
      } else {
        buf_pos_ = buf_len_;
        return false;
      }
    }
  }

  bool LoadNextFile(const BindData& bind, GlobalState& gs) {
    const size_t n_cols = bind.col_names.size();
    while (true) {
      CloseFd();
      size_t idx = gs.next_file.fetch_add(1, std::memory_order_relaxed);
      if (idx >= bind.paths.size())
        return false;
      cur_file_ord = static_cast<int32_t>(idx);
      const std::string& path = bind.paths[idx];
      fd_ = ::open(path.c_str(), O_RDONLY);
      if (bind.unlink_files) {
        ::unlink(path.c_str());
      }
      if (fd_ < 0)
        continue;
#ifdef POSIX_FADV_SEQUENTIAL
      ::posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
      buf_pos_ = 0;
      buf_len_ = 0;
      eof_ = false;
      FillBuffer();
      if (NextBatch(n_cols))
        return true;
    }
  }

 private:
  int fd_ = -1;
  bool eof_ = true;
  size_t buf_cap_ = 0;
  size_t buf_pos_ = 0;
  size_t buf_len_ = 0;
  std::unique_ptr<uint8_t[]> buf_;

  void CloseFd() {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  bool EnsureBytes(size_t need) {
    if (buf_len_ - buf_pos_ >= need)
      return true;
    if (eof_)
      return false;
    size_t rem = buf_len_ - buf_pos_;
    if (need > buf_cap_) {
      size_t new_cap = std::max<size_t>(buf_cap_ * 2, need + (size_t(1) << 20));
      std::unique_ptr<uint8_t[]> nb(new uint8_t[new_cap]);
      if (rem > 0) {
        std::memcpy(nb.get(), buf_.get() + buf_pos_, rem);
      }
      buf_.swap(nb);
      buf_cap_ = new_cap;
    } else if (rem > 0 && buf_pos_ > 0) {
      std::memmove(buf_.get(), buf_.get() + buf_pos_, rem);
    }
    buf_pos_ = 0;
    buf_len_ = rem;
    FillBuffer();
    return (buf_len_ - buf_pos_) >= need;
  }

  void FillBuffer() {
    while (!eof_ && buf_len_ < buf_cap_) {
      ssize_t n = ::read(fd_, buf_.get() + buf_len_, buf_cap_ - buf_len_);
      if (n <= 0) {
        eof_ = true;
        break;
      }
      buf_len_ += static_cast<size_t>(n);
    }
  }

  void ParseCellsBatch(const uint8_t* p, const uint8_t* end, size_t n_cols) {
    cur_batch.cells = nullptr;
    cur_batch.num_rows = 0;
    cur_batch.row = 0;
    cur_batch.var_p = nullptr;
    cur_batch.var_end = nullptr;
    cur_batch.f64_p = nullptr;
    cur_batch.f64_end = nullptr;
    cur_batch.strs.clear();
    cur_batch.str_idx = 0;
    cur_batch.blobs.clear();
    cur_batch.blob_idx = 0;

    const uint8_t* str_p = nullptr;
    const uint8_t* str_end = nullptr;
    size_t cells_len = 0;

    while (p < end) {
      uint64_t tag = 0;
      if (!ReadVarint(p, end, tag))
        break;
      uint32_t field = static_cast<uint32_t>(tag >> 3);
      uint32_t wt = static_cast<uint32_t>(tag & 7);
      if (wt == 0) {
        uint64_t dummy = 0;
        if (!ReadVarint(p, end, dummy))
          break;
      } else if (wt == 2) {
        uint64_t len = 0;
        if (!ReadVarint(p, end, len) || static_cast<size_t>(end - p) < len)
          break;
        const uint8_t* s = p;
        p += len;
        switch (field) {
          case kBatchCellsField:
            cur_batch.cells = s;
            cells_len = len;
            break;
          case kBatchVarintsField:
            cur_batch.var_p = s;
            cur_batch.var_end = s + len;
            break;
          case kBatchFloat64sField:
            cur_batch.f64_p = s;
            cur_batch.f64_end = s + len;
            break;
          case kBatchBlobsField:
            cur_batch.blobs.emplace_back(s, len);
            break;
          case kBatchStringsField:
            str_p = s;
            str_end = s + len;
            break;
          default:
            break;
        }
      } else if (wt == 1) {
        p = (end - p >= 8) ? (p + 8) : end;
      } else if (wt == 5) {
        p = (end - p >= 4) ? (p + 4) : end;
      } else {
        break;
      }
    }

    if (str_p && str_p < str_end) {
      const char* sp = reinterpret_cast<const char*>(str_p);
      const char* se = reinterpret_cast<const char*>(str_end);
      while (sp < se) {
        const char* nul =
            static_cast<const char*>(std::memchr(sp, '\0', se - sp));
        if (!nul) {
          cur_batch.strs.push_back({sp, static_cast<size_t>(se - sp)});
          break;
        }
        cur_batch.strs.push_back({sp, static_cast<size_t>(nul - sp)});
        sp = nul + 1;
      }
    }

    cur_batch.num_rows = n_cols ? (cells_len / n_cols) : 0;
    cur_batch.row = 0;
  }
};

static std::vector<std::string> ReadStringListValue(duckdb_value list_val) {
  std::vector<std::string> out;
  idx_t n = duckdb_get_list_size(list_val);
  out.reserve(n);
  for (idx_t i = 0; i < n; i++) {
    duckdb_value elem = duckdb_get_list_child(list_val, i);
    char* s = duckdb_get_varchar(elem);
    out.emplace_back(s ? s : "");
    if (s)
      duckdb_free(s);
    duckdb_destroy_value(&elem);
  }
  return out;
}

static void DestroyBindData(void* p) {
  delete static_cast<BindData*>(p);
}
static void DestroyGlobalState(void* p) {
  delete static_cast<GlobalState*>(p);
}
static void DestroyLocalState(void* p) {
  delete static_cast<LocalState*>(p);
}

static void ScanProtoBind(duckdb_bind_info info) {
  auto* bind = new BindData();
  duckdb_value v_paths = duckdb_bind_get_parameter(info, 0);
  duckdb_value v_names = duckdb_bind_get_parameter(info, 1);
  duckdb_value v_types = duckdb_bind_get_parameter(info, 2);
  duckdb_value v_unlink = duckdb_bind_get_parameter(info, 3);

  bind->paths = ReadStringListValue(v_paths);
  bind->col_names = ReadStringListValue(v_names);
  std::vector<std::string> raw_types = ReadStringListValue(v_types);
  bind->unlink_files = duckdb_get_bool(v_unlink);

  duckdb_destroy_value(&v_paths);
  duckdb_destroy_value(&v_names);
  duckdb_destroy_value(&v_types);
  duckdb_destroy_value(&v_unlink);

  duckdb_logical_type t_int = duckdb_create_logical_type(DUCKDB_TYPE_INTEGER);
  duckdb_logical_type t_varchar =
      duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
  duckdb_logical_type t_bigint = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
  duckdb_logical_type t_double = duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE);
  duckdb_logical_type t_bool = duckdb_create_logical_type(DUCKDB_TYPE_BOOLEAN);

  duckdb_bind_add_result_column(info, "_trace_ord", t_int);
  bind->all_i64 = !bind->col_names.empty();
  for (size_t i = 0; i < bind->col_names.size(); i++) {
    const std::string& ts = (i < raw_types.size()) ? raw_types[i] : "STRING";
    if (ts == "INT64" || ts == "BIGINT") {
      bind->col_kinds.push_back(ColKind::I64);
      duckdb_bind_add_result_column(info, bind->col_names[i].c_str(), t_bigint);
    } else if (ts == "DOUBLE" || ts == "FLOAT64") {
      bind->all_i64 = false;
      bind->col_kinds.push_back(ColKind::F64);
      duckdb_bind_add_result_column(info, bind->col_names[i].c_str(), t_double);
    } else if (ts == "BOOLEAN") {
      bind->all_i64 = false;
      bind->col_kinds.push_back(ColKind::BOOL);
      duckdb_bind_add_result_column(info, bind->col_names[i].c_str(), t_bool);
    } else {
      bind->all_i64 = false;
      bind->col_kinds.push_back(ColKind::STR);
      duckdb_bind_add_result_column(info, bind->col_names[i].c_str(),
                                    t_varchar);
    }
  }

  duckdb_destroy_logical_type(&t_int);
  duckdb_destroy_logical_type(&t_varchar);
  duckdb_destroy_logical_type(&t_bigint);
  duckdb_destroy_logical_type(&t_double);
  duckdb_destroy_logical_type(&t_bool);

  duckdb_bind_set_cardinality(
      info,
      std::max<idx_t>(1, static_cast<idx_t>(bind->paths.size())) * 1000000ULL,
      false);
  duckdb_bind_set_bind_data(info, bind, DestroyBindData);
}

static void ScanProtoGlobalInit(duckdb_init_info info) {
  auto* bind = static_cast<const BindData*>(duckdb_init_get_bind_data(info));
  auto* gs = new GlobalState();
  idx_t max_threads =
      std::max<idx_t>(1, static_cast<idx_t>(bind->paths.size()));
  duckdb_init_set_max_threads(info, max_threads);
  duckdb_init_set_init_data(info, gs, DestroyGlobalState);
}

static void ScanProtoLocalInit(duckdb_init_info info) {
  duckdb_init_set_init_data(info, new LocalState(), DestroyLocalState);
}

static inline void SetInvalid(duckdb_vector vec,
                              uint64_t*& validity,
                              idx_t row) {
  if (!validity) {
    duckdb_vector_ensure_validity_writable(vec);
    validity = duckdb_vector_get_validity(vec);
  }
  duckdb_validity_set_row_invalid(validity, row);
}

static void ScanProtoFunc(duckdb_function_info info, duckdb_data_chunk output) {
  const auto& bind =
      *static_cast<const BindData*>(duckdb_function_get_bind_data(info));
  auto& gs = *static_cast<GlobalState*>(duckdb_function_get_init_data(info));
  auto& ls =
      *static_cast<LocalState*>(duckdb_function_get_local_init_data(info));

  const size_t n_cols = bind.col_names.size();
  const idx_t cap = duckdb_vector_size();

  int32_t* ord_ptr = static_cast<int32_t*>(
      duckdb_vector_get_data(duckdb_data_chunk_get_vector(output, 0)));
  duckdb_vector vecs[64];
  void* data_ptrs[64];
  uint64_t* validities[64] = {};
  std::vector<duckdb_vector> vecs_heap;
  std::vector<void*> ptrs_heap;
  std::vector<uint64_t*> vals_heap;
  duckdb_vector* vec_p = vecs;
  void** ptr_p = data_ptrs;
  uint64_t** val_p = validities;
  if (n_cols > 64) {
    vecs_heap.resize(n_cols);
    ptrs_heap.resize(n_cols, nullptr);
    vals_heap.resize(n_cols, nullptr);
    vec_p = vecs_heap.data();
    ptr_p = ptrs_heap.data();
    val_p = vals_heap.data();
  }
  for (size_t c = 0; c < n_cols; c++) {
    vec_p[c] = duckdb_data_chunk_get_vector(output, c + 1);
    ptr_p[c] = duckdb_vector_get_data(vec_p[c]);
  }

  const ColKind* kinds = bind.col_kinds.data();
  idx_t out_row = 0;
  char num_buf[64];

  while (out_row < cap) {
    if (!ls.has_batch || ls.cur_batch.row >= ls.cur_batch.num_rows) {
      if (!ls.NextBatch(n_cols) && !ls.LoadNextFile(bind, gs)) {
        break;
      }
    }
    BatchCursor& b = ls.cur_batch;
    idx_t take =
        std::min<idx_t>(cap - out_row, static_cast<idx_t>(b.num_rows - b.row));
    std::fill_n(ord_ptr + out_row, take, ls.cur_file_ord);

    const uint8_t* cell_p = b.cells + b.row * n_cols;
    const uint8_t* var_p = b.var_p;
    const uint8_t* const var_end = b.var_end;

    // Fast path: all columns BIGINT (`ColKind::I64`) — zero type dispatch!
    if (bind.all_i64) {
      int64_t** i64_ptrs = reinterpret_cast<int64_t**>(ptr_p);
      for (idx_t r = 0; r < take; r++) {
        idx_t dst_row = out_row + r;
        for (size_t c = 0; c < n_cols; c++) {
          uint8_t ctype = *cell_p++;
          if (__builtin_expect(ctype == kCellVarint, 1)) {
            uint64_t raw = 0;
            ReadVarint(var_p, var_end, raw);
            i64_ptrs[c][dst_row] = static_cast<int64_t>(raw);
          } else {
            SetInvalid(vec_p[c], val_p[c], dst_row);
          }
        }
      }
      b.var_p = var_p;
      b.row += take;
      out_row += take;
      continue;
    }

    const StrSlice* strs = b.strs.data();
    size_t str_idx = b.str_idx;

    for (idx_t r = 0; r < take; r++) {
      idx_t dst_row = out_row + r;
      for (size_t c = 0; c < n_cols; c++) {
        uint8_t ctype = *cell_p++;
        ColKind kind = kinds[c];
        if (ctype == kCellVarint) {
          uint64_t raw = 0;
          ReadVarint(var_p, var_end, raw);
          int64_t v = static_cast<int64_t>(raw);
          if (__builtin_expect(kind == ColKind::I64, 1)) {
            static_cast<int64_t*>(ptr_p[c])[dst_row] = v;
          } else if (kind == ColKind::F64) {
            static_cast<double*>(ptr_p[c])[dst_row] = static_cast<double>(v);
          } else if (kind == ColKind::BOOL) {
            static_cast<bool*>(ptr_p[c])[dst_row] = (v != 0);
          } else {
            int len = std::snprintf(num_buf, sizeof(num_buf), "%lld",
                                    static_cast<long long>(v));
            duckdb_vector_assign_string_element_len(vec_p[c], dst_row, num_buf,
                                                    len > 0 ? len : 0);
          }
        } else if (ctype == kCellString) {
          const StrSlice& sl = strs[str_idx++];
          if (__builtin_expect(kind == ColKind::STR, 1)) {
            duckdb_vector_assign_string_element_len(vec_p[c], dst_row, sl.ptr,
                                                    sl.len);
          } else {
            SetInvalid(vec_p[c], val_p[c], dst_row);
          }
        } else if (ctype == kCellNull) {
          SetInvalid(vec_p[c], val_p[c], dst_row);
        } else if (ctype == kCellFloat64) {
          double v = 0.0;
          if (b.f64_end - b.f64_p >= 8) {
            std::memcpy(&v, b.f64_p, 8);
            b.f64_p += 8;
          }
          if (kind == ColKind::F64) {
            static_cast<double*>(ptr_p[c])[dst_row] = v;
          } else if (kind == ColKind::I64) {
            static_cast<int64_t*>(ptr_p[c])[dst_row] = static_cast<int64_t>(v);
          } else if (kind == ColKind::BOOL) {
            static_cast<bool*>(ptr_p[c])[dst_row] = (v != 0.0);
          } else {
            int len = std::snprintf(num_buf, sizeof(num_buf), "%.17g", v);
            duckdb_vector_assign_string_element_len(vec_p[c], dst_row, num_buf,
                                                    len > 0 ? len : 0);
          }
        } else if (ctype == kCellBlob) {
          std::string b64;
          if (b.blob_idx < b.blobs.size()) {
            b64 = Base64Encode(b.blobs[b.blob_idx].first,
                               b.blobs[b.blob_idx].second);
            b.blob_idx++;
          }
          if (kind == ColKind::STR) {
            duckdb_vector_assign_string_element_len(vec_p[c], dst_row,
                                                    b64.data(), b64.size());
          } else {
            SetInvalid(vec_p[c], val_p[c], dst_row);
          }
        } else {
          SetInvalid(vec_p[c], val_p[c], dst_row);
        }
      }
    }

    b.var_p = var_p;
    b.str_idx = str_idx;
    b.row += take;
    out_row += take;
  }

  duckdb_data_chunk_set_size(output, out_row);
}

}  // namespace

bool RegisterProtoIngestFunctions(duckdb_connection conn) {
  duckdb_table_function tf = duckdb_create_table_function();
  duckdb_table_function_set_name(tf, "pf_scan_tp_proto");

  duckdb_logical_type t_varchar =
      duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
  duckdb_logical_type t_varchar_list = duckdb_create_list_type(t_varchar);
  duckdb_logical_type t_bool = duckdb_create_logical_type(DUCKDB_TYPE_BOOLEAN);

  duckdb_table_function_add_parameter(tf, t_varchar_list);  // spool_paths
  duckdb_table_function_add_parameter(tf, t_varchar_list);  // col_names
  duckdb_table_function_add_parameter(tf, t_varchar_list);  // col_types
  duckdb_table_function_add_parameter(tf, t_bool);          // unlink_files

  duckdb_destroy_logical_type(&t_varchar);
  duckdb_destroy_logical_type(&t_varchar_list);
  duckdb_destroy_logical_type(&t_bool);

  duckdb_table_function_set_bind(tf, ScanProtoBind);
  duckdb_table_function_set_init(tf, ScanProtoGlobalInit);
  duckdb_table_function_set_local_init(tf, ScanProtoLocalInit);
  duckdb_table_function_set_function(tf, ScanProtoFunc);
  duckdb_table_function_supports_projection_pushdown(tf, false);

  duckdb_state st = duckdb_register_table_function(conn, tf);
  duckdb_destroy_table_function(&tf);
  return st == DuckDBSuccess;
}
