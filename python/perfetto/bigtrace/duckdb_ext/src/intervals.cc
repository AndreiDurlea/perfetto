// Interval operators (spec §5) for the perfetto DuckDB extension.
//
// Every kernel is an aggregate function: the SQL macro layer
// (sql/intervals.sql) numbers each operand's rows, tags them with an operand
// index ("side"), and GROUPs BY the PER columns, so DuckDB's parallel hash
// aggregate does the partitioning and each group is solved by one sort + sweep
// here. Kernels return LIST<STRUCT> of bounds plus operand row numbers; the
// macros unnest that and join the rows back to recover payload columns.
//
// Interval semantics (identical to TP's PerfettoSQL Next): a row with
// dur > 0 covers [ts, ts + dur); dur = 0 is a point at ts which intersects an
// interval covering it (start <= ts < end) or a point at the same ts. Rows
// with NULL ts or dur are skipped; ts < 0 or dur < 0 is an error.

#include "common.h"  // no-include-violation-check

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

constexpr int64_t kNone = -1;

struct Rec {
  int64_t s;     // start
  int64_t e;     // end (s + dur)
  int64_t ep;    // effective end for overlap tests: max(e, s + 1)
  int64_t row;   // operand row number
  int32_t side;  // operand index
};

inline bool ByStart(const Rec& a, const Rec& b) {
  if (a.s != b.s)
    return a.s < b.s;
  if (a.e != b.e)
    return a.e < b.e;
  return a.row < b.row;
}

// Two records overlap under PerfettoSQL point semantics.
inline bool Overlaps(const Rec& a, const Rec& b) {
  return a.s < b.ep && b.s < a.ep;
}

struct Span16 {
  int64_t s;
  int64_t e;
};

struct Rec24 {
  int64_t s;
  int64_t e;
  int64_t row;
};

inline bool ByStart24(const Rec24& a, const Rec24& b) {
  if (a.s != b.s)
    return a.s < b.s;
  if (a.e != b.e)
    return a.e < b.e;
  return a.row < b.row;
}

//===----------------------------------------------------------------------===//
// Thread-local bump arena so 50,000 per-track groups do ZERO malloc/free calls.
//===----------------------------------------------------------------------===//
template <typename T>
struct ArenaVec {
  T* data = nullptr;
  uint32_t size = 0;
  uint32_t cap = 0;

  ArenaVec() = default;
  void clear() {
    std::free(data);
    data = nullptr;
    size = 0;
    cap = 0;
  }
  ArenaVec(const ArenaVec&) = delete;
  ArenaVec& operator=(const ArenaVec&) = delete;
  ArenaVec(ArenaVec&& o) noexcept : data(o.data), size(o.size), cap(o.cap) {
    o.data = nullptr;
    o.size = 0;
    o.cap = 0;
  }
  ArenaVec& operator=(ArenaVec&& o) noexcept {
    if (this != &o) {
      std::free(data);
      data = o.data;
      size = o.size;
      cap = o.cap;
      o.data = nullptr;
      o.size = 0;
      o.cap = 0;
    }
    return *this;
  }
  ~ArenaVec() { std::free(data); }

  void Reserve(uint32_t need) {
    if (need <= cap)
      return;
    uint32_t new_cap = cap ? cap * 2 : 8;
    while (new_cap < need)
      new_cap *= 2;
    data = static_cast<T*>(std::realloc(data, size_t(new_cap) * sizeof(T)));
    cap = new_cap;
  }

  void Push(const T& v) {
    if (size == cap)
      Reserve(size + 1);
    data[size++] = v;
  }

  void push_back(const T& v) { Push(v); }
};

//===----------------------------------------------------------------------===//
// State plumbing: lightweight move-only POD arrays (zero cross-thread
// aliasing).
//===----------------------------------------------------------------------===//
struct State {
  ArenaVec<Span16> spans;  // fast path for FlattenSum0 / MergeCount
  bool spans_sorted = true;
  int64_t last_span_s = std::numeric_limits<int64_t>::min();

  ArenaVec<Rec24> sides[2];  // fast side-separated storage for N=2
  bool side_sorted[2] = {true, true};
  int64_t last_side_s[2] = {std::numeric_limits<int64_t>::min(),
                            std::numeric_limits<int64_t>::min()};
  int64_t min_s[2] = {std::numeric_limits<int64_t>::max(),
                      std::numeric_limits<int64_t>::max()};
  int64_t max_e[2] = {std::numeric_limits<int64_t>::min(),
                      std::numeric_limits<int64_t>::min()};

  std::vector<Rec> recs;
  size_t total_recs = 0;
  int64_t param = 0;        // operand count / relationship code
  int32_t width = 0;        // number of SUM values per row (flatten fast path)
  bool local_rows = false;  // rec.row indexes vals (flatten fast path)
  std::vector<int64_t> vals;     // width values per rec
  std::vector<uint8_t> present;  // width validity flags per rec
  std::vector<State*> parts;

  void MaterializeRecs() {
    if (sides[0].size == 0 && sides[1].size == 0)
      return;
    recs.reserve(recs.size() + sides[0].size + sides[1].size);
    for (int sd = 0; sd < 2; sd++) {
      for (uint32_t i = 0; i < sides[sd].size; i++) {
        const Rec24& x = sides[sd].data[i];
        int64_t ep = x.e > x.s ? x.e : x.s + 1;
        recs.push_back({x.s, x.e, ep, x.row, sd});
      }
      sides[sd].size = 0;
    }
  }

  void Merge(State& o) {
    total_recs += o.total_recs;
    o.total_recs = 0;
    param = std::max(param, o.param);
    width = std::max(width, o.width);
    if (o.spans.size > 0 && recs.empty() && o.recs.empty() && parts.empty() &&
        o.parts.empty()) {
      if (spans.size > 0 && o.spans.data[0].s < last_span_s)
        spans_sorted = false;
      spans_sorted &= o.spans_sorted;
      spans.Reserve(spans.size + o.spans.size);
      std::memcpy(spans.data + spans.size, o.spans.data,
                  size_t(o.spans.size) * sizeof(Span16));
      spans.size += o.spans.size;
      last_span_s = o.last_span_s;
      o.spans.size = 0;
      return;
    }
    if ((o.sides[0].size > 0 || o.sides[1].size > 0) && recs.empty() &&
        o.recs.empty() && parts.empty() && o.parts.empty()) {
      for (int sd = 0; sd < 2; sd++) {
        if (o.sides[sd].size == 0)
          continue;
        if (sides[sd].size > 0 && o.sides[sd].data[0].s < last_side_s[sd])
          side_sorted[sd] = false;
        side_sorted[sd] &= o.side_sorted[sd];
        min_s[sd] = std::min(min_s[sd], o.min_s[sd]);
        max_e[sd] = std::max(max_e[sd], o.max_e[sd]);
        sides[sd].Reserve(sides[sd].size + o.sides[sd].size);
        std::memcpy(sides[sd].data + sides[sd].size, o.sides[sd].data,
                    size_t(o.sides[sd].size) * sizeof(Rec24));
        sides[sd].size += o.sides[sd].size;
        last_side_s[sd] = o.last_side_s[sd];
        o.sides[sd].size = 0;
      }
      return;
    }
    MaterializeRecs();
    o.MaterializeRecs();
    State* moved = new State();
    moved->recs.swap(o.recs);
    moved->vals.swap(o.vals);
    moved->present.swap(o.present);
    moved->parts.swap(o.parts);
    moved->width = o.width;
    moved->local_rows = o.local_rows;
    parts.push_back(moved);
  }

  void CollectParts(std::vector<State*>& all) {
    for (State* p : parts) {
      all.push_back(p);
      p->CollectParts(all);
      p->parts.clear();
    }
    parts.clear();
  }

  void Gather() {
    if (parts.empty())
      return;
    MaterializeRecs();
    std::vector<State*> all;
    CollectParts(all);
    size_t total = recs.size();
    for (State* p : all)
      total += p->recs.size();
    recs.reserve(total);
    for (State* p : all) {
      if (local_rows) {
        int64_t base = static_cast<int64_t>(recs.size());
        for (Rec r : p->recs) {
          r.row += base;
          recs.push_back(r);
        }
        vals.insert(vals.end(), p->vals.begin(), p->vals.end());
        present.insert(present.end(), p->present.begin(), p->present.end());
      } else {
        recs.insert(recs.end(), p->recs.begin(), p->recs.end());
      }
      delete p;
    }
  }
};

inline State* AllocState() {
  return new State();
}

State*& Ptr(duckdb_aggregate_state s) {
  return *reinterpret_cast<State**>(s);
}
idx_t StateSize(duckdb_function_info) {
  return sizeof(State*);
}
void StateInit(duckdb_function_info, duckdb_aggregate_state s) {
  Ptr(s) = nullptr;
}
void StateDestroy(duckdb_aggregate_state* states, idx_t count) {
  for (idx_t i = 0; i < count; i++) {
    if (State* st = Ptr(states[i])) {
      for (State* p : st->parts)
        delete p;
      Ptr(states[i]) = nullptr;
      delete st;
    }
  }
}
void StateCombine(duckdb_function_info,
                  duckdb_aggregate_state* source,
                  duckdb_aggregate_state* target,
                  idx_t count) {
  for (idx_t i = 0; i < count; i++) {
    State*& src = Ptr(source[i]);
    State*& dst = Ptr(target[i]);
    if (!src)
      continue;
    if (!dst) {
      dst = src;
      src = nullptr;
      continue;
    }
    dst->Merge(*src);
  }
}

template <typename T>
struct Col {
  const T* data;
  uint64_t* validity;
  explicit Col(duckdb_vector v)
      : data(static_cast<const T*>(duckdb_vector_get_data(v))),
        validity(duckdb_vector_get_validity(v)) {}
  bool Valid(idx_t r) const {
    return !validity || ((validity[r >> 6] >> (r & 63)) & 1);
  }
};

// Reads (side INT, row BIGINT, ts BIGINT, dur BIGINT [, param INT]).
void UpdateGeneric(duckdb_function_info info,
                   duckdb_data_chunk input,
                   duckdb_aggregate_state* states) {
  idx_t n = duckdb_data_chunk_get_size(input);
  idx_t ncols = duckdb_data_chunk_get_column_count(input);
  Col<int32_t> side(duckdb_data_chunk_get_vector(input, 0));
  Col<int64_t> row(duckdb_data_chunk_get_vector(input, 1));
  Col<int64_t> ts(duckdb_data_chunk_get_vector(input, 2));
  Col<int64_t> dur(duckdb_data_chunk_get_vector(input, 3));
  bool has_param = ncols > 4;
  const int32_t* param =
      has_param ? static_cast<const int32_t*>(duckdb_vector_get_data(
                      duckdb_data_chunk_get_vector(input, 4)))
                : nullptr;

  // Fast run-grouped path when no column has validity bitmasks and side in
  // {0,1}.
  if (!has_param && !ts.validity && !dur.validity && !side.validity &&
      !row.validity) {
    idx_t r = 0;
    while (r < n) {
      duckdb_aggregate_state sp = states[r];
      int32_t sd = side.data[r];
      if ((sd & ~1) != 0) {
        State*& st = Ptr(sp);
        if (!st)
          st = AllocState();
        st->MaterializeRecs();
        int64_t s = ts.data[r], d = dur.data[r];
        if (s < 0 || d < 0) {
          duckdb_aggregate_function_set_error(
              info, "pf_interval: a row's ts or dur is below zero");
          return;
        }
        int64_t end = s + d;
        st->recs.push_back({s, end, d > 0 ? end : s + 1, row.data[r], sd});
        st->total_recs++;
        r++;
        continue;
      }
      idx_t e = r + 1;
      while (e < n && states[e] == sp && side.data[e] == sd)
        e++;
      State*& st = Ptr(sp);
      if (!st)
        st = AllocState();
      if (!st->recs.empty()) {
        for (idx_t k = r; k < e; k++) {
          int64_t s = ts.data[k], d = dur.data[k];
          if (s < 0 || d < 0) {
            duckdb_aggregate_function_set_error(
                info, "pf_interval: a row's ts or dur is below zero");
            return;
          }
          int64_t end = s + d;
          st->recs.push_back({s, end, d > 0 ? end : s + 1, row.data[k], sd});
        }
        st->total_recs += (e - r);
        r = e;
        continue;
      }
      auto& vec = st->sides[sd];
      idx_t len = e - r;
      vec.Reserve(vec.size + static_cast<uint32_t>(len));
      Rec24* dst = vec.data + vec.size;
      int64_t neg = 0;
      int64_t prev_s = st->last_side_s[sd];
      bool sorted = st->side_sorted[sd];
      int64_t min_s = st->min_s[sd];
      int64_t max_e = st->max_e[sd];
      for (idx_t k = 0; k < len; k++) {
        int64_t s = ts.data[r + k];
        int64_t d = dur.data[r + k];
        neg |= s | d;
        int64_t end = s + d;
        int64_t ep = d > 0 ? end : s + 1;
        if (s < prev_s)
          sorted = false;
        prev_s = s;
        if (s < min_s)
          min_s = s;
        if (ep > max_e)
          max_e = ep;
        dst[k] = {s, end, row.data[r + k]};
      }
      if (neg < 0) {
        duckdb_aggregate_function_set_error(
            info, "pf_interval: a row's ts or dur is below zero");
        return;
      }
      vec.size += static_cast<uint32_t>(len);
      st->last_side_s[sd] = prev_s;
      st->side_sorted[sd] = sorted;
      st->min_s[sd] = min_s;
      st->max_e[sd] = max_e;
      st->total_recs += len;
      r = e;
    }
    return;
  }

  for (idx_t r = 0; r < n; r++) {
    State*& st = Ptr(states[r]);
    if (!st)
      st = AllocState();
    st->MaterializeRecs();
    if (has_param)
      st->param = param[r];
    if (!ts.Valid(r) || !dur.Valid(r) || !side.Valid(r) || !row.Valid(r))
      continue;
    int64_t s = ts.data[r], d = dur.data[r];
    if (s < 0 || (d < 0 && !(has_param && param[r] == 1 && side.data[r] == 1 &&
                             d == -1))) {
      duckdb_aggregate_function_set_error(
          info, "pf_interval: a row's ts or dur is below zero");
      return;
    }
    int64_t end = d == -1 ? std::numeric_limits<int64_t>::max() : s + d;
    int64_t ep = d > 0 ? end : (d == -1 ? end : s + 1);
    st->recs.push_back({s, end, ep, row.data[r], side.data[r]});
    st->total_recs++;
  }
}

//===----------------------------------------------------------------------===//
// Output: LIST<STRUCT(f0 BIGINT, ..., f{K-1} BIGINT [, rows BIGINT[]])>.
//===----------------------------------------------------------------------===//
struct OutBuf {
  int k = 0;                    // scalar fields per record (<= 5)
  bool has_list = false;        // trailing BIGINT[] field
  std::vector<int64_t> col[5];  // columnar scalar fields
  std::vector<uint8_t>
      col3_valid;  // optional validity for col[3] (1-sum flatten)
  std::vector<uint32_t> list_off;  // per record offset into list_vals
  std::vector<int64_t> list_vals;
  std::vector<uint8_t> list_valid;  // parallel to list_vals (1 = valid)
  size_t size() const { return k ? col[0].size() : list_off.size(); }
  void Reserve(size_t n) {
    for (int f = 0; f < k; f++)
      col[f].reserve(n);
  }
};

inline void Emit(OutBuf& o, std::initializer_list<int64_t> v) {
  int f = 0;
  for (int64_t x : v)
    o.col[f++].push_back(x);
}
inline void BeginList(OutBuf& o) {
  o.list_off.push_back(static_cast<uint32_t>(o.list_vals.size()));
}
inline void ListPush(OutBuf& o, int64_t v, bool valid = true) {
  o.list_vals.push_back(v);
  o.list_valid.push_back(valid);
}

using Solver = void (*)(State&, OutBuf&);

int PoolThreads() {
  const char* env = getenv("PF_THREADS");
  int n = env ? atoi(env) : 1;
  return std::max(1, std::min(n, 48));
}

// Solves every group of this finalize call, using a reusable thread-local
// output buffer in single-threaded mode (zero per-group heap allocations, one
// bulk memcpy per column) or a worker pool when PF_THREADS > 1.
template <int K, bool LIST, Solver SOLVE>
void FinalizeWith(duckdb_function_info,
                  duckdb_aggregate_state* source,
                  duckdb_vector result,
                  idx_t count,
                  idx_t offset) {
  std::vector<State*> states(count);
  size_t work = 0;
  for (idx_t i = 0; i < count; i++) {
    states[i] = Ptr(source[i]);
    if (states[i])
      work += states[i]->total_recs;
  }
  int threads = PoolThreads();
  if (threads <= 1 || count <= 1 || work < 20000) {
    thread_local OutBuf tls_out;
    thread_local std::vector<idx_t> tls_counts;
    tls_out.k = K;
    tls_out.has_list = LIST;
    for (int f = 0; f < K; f++)
      tls_out.col[f].clear();
    tls_out.col3_valid.clear();
    tls_out.list_vals.clear();
    tls_out.list_valid.clear();
    tls_out.list_off.clear();
    tls_counts.assign(count, 0);

    for (idx_t i = 0; i < count; i++) {
      idx_t before = tls_out.size();
      if (states[i]) {
        states[i]->Gather();
        SOLVE(*states[i], tls_out);
      }
      tls_counts[i] = tls_out.size() - before;
    }

    idx_t base0 = duckdb_list_vector_get_size(result);
    idx_t total_n = tls_out.size();
    idx_t total_l = LIST ? tls_out.list_vals.size() : 0;
    bool any_col3_null = false, any_list_null = false;
    for (uint8_t v : tls_out.col3_valid) {
      if (!v) {
        any_col3_null = true;
        break;
      }
    }
    if (LIST) {
      for (uint8_t v : tls_out.list_valid) {
        if (!v) {
          any_list_null = true;
          break;
        }
      }
    }

    duckdb_list_vector_reserve(result, base0 + total_n);
    auto* entries =
        static_cast<duckdb_list_entry*>(duckdb_vector_get_data(result));
    duckdb_vector child = duckdb_list_vector_get_child(result);
    for (int f = 0; f < K; f++) {
      int64_t* dst = static_cast<int64_t*>(
          duckdb_vector_get_data(duckdb_struct_vector_get_child(child, f)));
      if (total_n)
        std::memcpy(dst + base0, tls_out.col[f].data(),
                    total_n * sizeof(int64_t));
    }
    if (any_col3_null && K > 3) {
      duckdb_vector c3 = duckdb_struct_vector_get_child(child, 3);
      duckdb_vector_ensure_validity_writable(c3);
      uint64_t* col3_val = duckdb_vector_get_validity(c3);
      for (idx_t i = 0; i < total_n; i++)
        if (!tls_out.col3_valid[i])
          duckdb_validity_set_row_invalid(col3_val, base0 + i);
    }
    if (LIST) {
      duckdb_vector lv = duckdb_struct_vector_get_child(child, K);
      idx_t lbase = duckdb_list_vector_get_size(lv);
      duckdb_list_vector_reserve(lv, lbase + total_l);
      auto* lentries =
          static_cast<duckdb_list_entry*>(duckdb_vector_get_data(lv));
      duckdb_vector lchild = duckdb_list_vector_get_child(lv);
      int64_t* ld = static_cast<int64_t*>(duckdb_vector_get_data(lchild));
      if (total_l)
        std::memcpy(ld + lbase, tls_out.list_vals.data(),
                    total_l * sizeof(int64_t));
      if (any_list_null) {
        duckdb_vector_ensure_validity_writable(lchild);
        uint64_t* lval = duckdb_vector_get_validity(lchild);
        for (idx_t i = 0; i < total_l; i++)
          if (!tls_out.list_valid[i])
            duckdb_validity_set_row_invalid(lval, lbase + i);
      }
      for (idx_t i = 0; i < total_n; i++) {
        uint32_t off = tls_out.list_off[i];
        uint32_t end = i + 1 < total_n ? tls_out.list_off[i + 1]
                                       : static_cast<uint32_t>(total_l);
        lentries[base0 + i].offset = lbase + off;
        lentries[base0 + i].length = end - off;
      }
      duckdb_list_vector_set_size(lv, lbase + total_l);
    }
    idx_t base = base0;
    for (idx_t r = 0; r < count; r++) {
      entries[offset + r].offset = base;
      entries[offset + r].length = tls_counts[r];
      base += tls_counts[r];
    }
    duckdb_list_vector_set_size(result, base);
    return;
  }

  std::vector<OutBuf> outs(count);
  for (auto& o : outs) {
    o.k = K;
    o.has_list = LIST;
  }
  std::atomic<size_t> next{0};
  auto worker = [&]() {
    for (size_t i; (i = next.fetch_add(1)) < count;)
      if (states[i]) {
        states[i]->Gather();
        SOLVE(*states[i], outs[i]);
      }
  };
  std::vector<std::thread> pool;
  int n = static_cast<int>(std::min<size_t>(threads, count));
  for (int t = 1; t < n; t++)
    pool.emplace_back(worker);
  worker();
  for (auto& t : pool)
    t.join();

  idx_t base0 = duckdb_list_vector_get_size(result);
  idx_t total_n = 0, total_l = 0;
  bool any_col3_null = false, any_list_null = false;
  for (idx_t i = 0; i < count; i++) {
    total_n += outs[i].size();
    if (!outs[i].col3_valid.empty()) {
      for (uint8_t v : outs[i].col3_valid)
        if (!v) {
          any_col3_null = true;
          break;
        }
    }
    if (LIST) {
      total_l += outs[i].list_vals.size();
      if (!any_list_null) {
        for (uint8_t v : outs[i].list_valid)
          if (!v) {
            any_list_null = true;
            break;
          }
      }
    }
  }
  duckdb_list_vector_reserve(result, base0 + total_n);
  auto* entries =
      static_cast<duckdb_list_entry*>(duckdb_vector_get_data(result));
  duckdb_vector child = duckdb_list_vector_get_child(result);
  int64_t* col_ptrs[5] = {};
  for (int f = 0; f < K; f++) {
    col_ptrs[f] = static_cast<int64_t*>(
        duckdb_vector_get_data(duckdb_struct_vector_get_child(child, f)));
  }
  uint64_t* col3_val = nullptr;
  if (any_col3_null && K > 3) {
    duckdb_vector c3 = duckdb_struct_vector_get_child(child, 3);
    duckdb_vector_ensure_validity_writable(c3);
    col3_val = duckdb_vector_get_validity(c3);
  }
  duckdb_vector lv = nullptr, lchild = nullptr;
  duckdb_list_entry* lentries = nullptr;
  int64_t* ld = nullptr;
  uint64_t* lval = nullptr;
  idx_t lbase = 0;
  if (LIST) {
    lv = duckdb_struct_vector_get_child(child, K);
    lbase = duckdb_list_vector_get_size(lv);
    duckdb_list_vector_reserve(lv, lbase + total_l);
    lentries = static_cast<duckdb_list_entry*>(duckdb_vector_get_data(lv));
    lchild = duckdb_list_vector_get_child(lv);
    ld = static_cast<int64_t*>(duckdb_vector_get_data(lchild));
    if (any_list_null) {
      duckdb_vector_ensure_validity_writable(lchild);
      lval = duckdb_vector_get_validity(lchild);
    }
  }
  idx_t base = base0;
  for (idx_t r = 0; r < count; r++) {
    const OutBuf& out = outs[r];
    idx_t n = out.size();
    for (int f = 0; f < K; f++) {
      if (n)
        std::memcpy(col_ptrs[f] + base, out.col[f].data(), n * sizeof(int64_t));
    }
    if (col3_val && !out.col3_valid.empty()) {
      for (idx_t i = 0; i < n; i++)
        if (!out.col3_valid[i])
          duckdb_validity_set_row_invalid(col3_val, base + i);
    }
    if (LIST) {
      idx_t lt = out.list_vals.size();
      if (lt)
        std::memcpy(ld + lbase, out.list_vals.data(), lt * sizeof(int64_t));
      if (lval) {
        for (idx_t i = 0; i < lt; i++)
          if (!out.list_valid[i])
            duckdb_validity_set_row_invalid(lval, lbase + i);
      }
      for (idx_t i = 0; i < n; i++) {
        uint32_t off = out.list_off[i];
        uint32_t end =
            i + 1 < n ? out.list_off[i + 1] : static_cast<uint32_t>(lt);
        lentries[base + i].offset = lbase + off;
        lentries[base + i].length = end - off;
      }
      lbase += lt;
    }
    entries[offset + r].offset = base;
    entries[offset + r].length = n;
    base += n;
  }
  if (LIST)
    duckdb_list_vector_set_size(lv, lbase);
  duckdb_list_vector_set_size(result, base);
}

//===----------------------------------------------------------------------===//
// 2-way overlap sweep: calls emit(a, b) for every overlapping pair, A and B
// sorted by start. Each pair is found once, when its later-starting member
// arrives (A first on ties). Skips pushing dead intervals onto active lists.
//===----------------------------------------------------------------------===//
template <typename F>
void OverlapSweep(const std::vector<Rec>& A,
                  const std::vector<Rec>& B,
                  F emit) {
  thread_local std::vector<uint32_t> act_a, act_b;
  act_a.clear();
  act_b.clear();
  size_t i = 0, j = 0;
  while (i < A.size() || j < B.size()) {
    if (j >= B.size() || (i < A.size() && A[i].s <= B[j].s)) {
      if (j >= B.size() && act_b.empty())
        break;
      const Rec& x = A[i];
      size_t w = 0;
      for (size_t k = 0; k < act_b.size(); k++) {
        const Rec& y = B[act_b[k]];
        if (y.ep > x.s) {
          act_b[w++] = act_b[k];
          emit(x, y);
        }
      }
      act_b.resize(w);
      if (j < B.size() && x.ep > B[j].s)
        act_a.push_back(static_cast<uint32_t>(i));
      i++;
    } else {
      if (i >= A.size() && act_a.empty())
        break;
      const Rec& x = B[j];
      size_t w = 0;
      for (size_t k = 0; k < act_a.size(); k++) {
        const Rec& y = A[act_a[k]];
        if (y.ep > x.s) {
          act_a[w++] = act_a[k];
          emit(y, x);
        }
      }
      act_a.resize(w);
      if (i < A.size() && x.ep > A[i].s)
        act_b.push_back(static_cast<uint32_t>(j));
      j++;
    }
  }
}

template <typename EmitPair>
void Sweep24(const Rec24* A,
             size_t na,
             const Rec24* B,
             size_t nb,
             EmitPair emit) {
  thread_local std::vector<uint32_t> act_a, act_b;
  act_a.clear();
  act_b.clear();
  size_t i = 0, j = 0;
  while (i < na || j < nb) {
    if (j >= nb || (i < na && A[i].s <= B[j].s)) {
      if (act_b.empty()) {
        if (j >= nb)
          break;
        const int64_t bs = B[j].s;
        while (i < na && A[i].s <= bs) {
          const int64_t ep = A[i].e + (A[i].e == A[i].s);
          if (ep > bs)
            act_a.push_back(static_cast<uint32_t>(i));
          i++;
        }
        continue;
      }
      const Rec24& x = A[i];
      size_t w = 0;
      for (size_t k = 0; k < act_b.size(); k++) {
        const Rec24& y = B[act_b[k]];
        const int64_t yep = y.e + (y.e == y.s);
        if (yep > x.s) {
          act_b[w++] = act_b[k];
          emit(x, y);
        }
      }
      act_b.resize(w);
      const int64_t xep = x.e + (x.e == x.s);
      if (j < nb && xep > B[j].s)
        act_a.push_back(static_cast<uint32_t>(i));
      i++;
    } else {
      if (act_a.empty()) {
        if (i >= na)
          break;
        const int64_t as = A[i].s;
        while (j < nb && B[j].s < as) {
          const int64_t ep = B[j].e + (B[j].e == B[j].s);
          if (ep > as)
            act_b.push_back(static_cast<uint32_t>(j));
          j++;
        }
        continue;
      }
      const Rec24& x = B[j];
      size_t w = 0;
      for (size_t k = 0; k < act_a.size(); k++) {
        const Rec24& y = A[act_a[k]];
        const int64_t yep = y.e + (y.e == y.s);
        if (yep > x.s) {
          act_a[w++] = act_a[k];
          emit(y, x);
        }
      }
      act_a.resize(w);
      const int64_t xep = x.e + (x.e == x.s);
      if (i < na && xep > A[i].s)
        act_b.push_back(static_cast<uint32_t>(j));
      j++;
    }
  }
}

void SplitSides(State& st, int nsides, std::vector<std::vector<Rec>>& out) {
  st.MaterializeRecs();
  out.assign(nsides, {});
  for (const Rec& r : st.recs)
    if (r.side >= 0 && r.side < nsides)
      out[r.side].push_back(r);
  for (auto& v : out)
    if (!std::is_sorted(v.begin(), v.end(), ByStart))
      std::sort(v.begin(), v.end(), ByStart);
}

//===----------------------------------------------------------------------===//
// INTERVAL INTERSECTION OF (N operands): successive narrowing, smallest
// operand first (as TP). Fast direct paths for N=2 and N=3 (no nested list).
//===----------------------------------------------------------------------===//
void SolveIntersect2(State& st, OutBuf& out) {
  if (!st.recs.empty() && st.sides[0].size == 0 && st.sides[1].size == 0) {
    // Fallback if populated via MaterializeRecs.
    std::vector<std::vector<Rec>> ops;
    SplitSides(st, 2, ops);
    if (ops[0].empty() || ops[1].empty())
      return;
    OverlapSweep(ops[0], ops[1], [&](const Rec& a, const Rec& b) {
      int64_t s = std::max(a.s, b.s);
      int64_t e = std::max(s, std::min(a.e, b.e));
      Emit(out, {s, e - s, a.row, b.row});
    });
    return;
  }
  const size_t n0 = st.sides[0].size, n1 = st.sides[1].size;
  if (n0 == 0 || n1 == 0 || st.max_e[0] <= st.min_s[1] ||
      st.max_e[1] <= st.min_s[0]) {
    return;
  }
  auto by_start24 = [](const Rec24& a, const Rec24& b) {
    return a.s != b.s ? a.s < b.s : a.e < b.e;
  };
  if (!st.side_sorted[0]) {
    std::sort(st.sides[0].data, st.sides[0].data + n0, by_start24);
    st.side_sorted[0] = true;
  }
  if (!st.side_sorted[1]) {
    std::sort(st.sides[1].data, st.sides[1].data + n1, by_start24);
    st.side_sorted[1] = true;
  }
  Sweep24(st.sides[0].data, n0, st.sides[1].data, n1,
          [&](const Rec24& a, const Rec24& b) {
            int64_t s = std::max(a.s, b.s);
            int64_t e = std::max(s, std::min(a.e, b.e));
            Emit(out, {s, e - s, a.row, b.row});
          });
}

void SolveIntersect3(State& st, OutBuf& out) {
  std::vector<std::vector<Rec>> ops;
  SplitSides(st, 3, ops);
  if (ops[0].empty() || ops[1].empty() || ops[2].empty())
    return;
  int order[3] = {0, 1, 2};
  std::stable_sort(order, order + 3,
                   [&](int a, int b) { return ops[a].size() < ops[b].size(); });
  int o0 = order[0], o1 = order[1], o2 = order[2];
  std::vector<Rec> mid;
  std::vector<std::pair<int64_t, int64_t>> mid_rows;
  OverlapSweep(ops[o0], ops[o1], [&](const Rec& a, const Rec& b) {
    int64_t s = std::max(a.s, b.s);
    int64_t e = std::max(s, std::min(a.e, b.e));
    mid.push_back(
        {s, e, e > s ? e : s + 1, static_cast<int64_t>(mid.size()), 0});
    mid_rows.push_back({a.row, b.row});
  });
  if (mid.empty())
    return;
  if (!std::is_sorted(mid.begin(), mid.end(), ByStart)) {
    std::sort(mid.begin(), mid.end(), ByStart);
  }
  OverlapSweep(mid, ops[o2], [&](const Rec& a, const Rec& b) {
    int64_t s = std::max(a.s, b.s);
    int64_t e = std::max(s, std::min(a.e, b.e));
    int64_t r[3];
    auto pr = mid_rows[static_cast<size_t>(a.row)];
    r[o0] = pr.first;
    r[o1] = pr.second;
    r[o2] = b.row;
    Emit(out, {s, e - s, r[0], r[1], r[2]});
  });
}

void SolveIntersect(State& st, OutBuf& out) {
  int n = static_cast<int>(st.param);
  if (n < 2)
    return;
  std::vector<std::vector<Rec>> ops;
  SplitSides(st, n, ops);
  std::vector<int> order(n);
  for (int i = 0; i < n; i++) {
    if (ops[i].empty())
      return;
    order[i] = i;
  }
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return ops[a].size() < ops[b].size(); });
  // Running regions: Rec (bounds; row = index into rows table).
  std::vector<Rec> run, next;
  std::vector<int64_t> rows, next_rows;
  for (const Rec& r : ops[order[0]]) {
    Rec x = r;
    x.row = static_cast<int64_t>(run.size());
    run.push_back(x);
    for (int k = 0; k < n; k++)
      rows.push_back(k == order[0] ? r.row : kNone);
  }
  for (int step = 1; step < n && !run.empty(); step++) {
    int which = order[step];
    next.clear();
    next_rows.clear();
    OverlapSweep(run, ops[which], [&](const Rec& a, const Rec& b) {
      int64_t s = std::max(a.s, b.s);
      int64_t e = std::max(s, std::min(a.e, b.e));
      Rec x{s, e, e > s ? e : s + 1, static_cast<int64_t>(next.size()), 0};
      next.push_back(x);
      const int64_t* from = &rows[static_cast<size_t>(a.row) * n];
      next_rows.insert(next_rows.end(), from, from + n);
      next_rows[next_rows.size() - n + which] = b.row;
    });
    // Re-sort regions by start for the next narrowing step, remapping rows.
    std::vector<uint32_t> idx(next.size());
    for (size_t i = 0; i < idx.size(); i++)
      idx[i] = static_cast<uint32_t>(i);
    if (step + 1 < n) {
      std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
        return ByStart(next[a], next[b]);
      });
    }
    run.clear();
    rows.clear();
    for (uint32_t i : idx) {
      Rec x = next[i];
      x.row = static_cast<int64_t>(run.size());
      run.push_back(x);
      rows.insert(rows.end(), next_rows.begin() + static_cast<size_t>(i) * n,
                  next_rows.begin() + static_cast<size_t>(i + 1) * n);
    }
  }
  for (size_t i = 0; i < run.size(); i++) {
    Emit(out, {run[i].s, run[i].e - run[i].s});
    BeginList(out);
    for (int k = 0; k < n; k++)
      ListPush(out, rows[i * n + k]);
  }
}

//===----------------------------------------------------------------------===//
// Elementary-segment sweep shared by FLATTEN and UNION (TP FLATTEN rules):
// positive rows cover [s, e); segments split at every distinct start (of any
// row, points included) and end; rows with dur = 0 at a timestamp produce one
// extra zero-width segment holding those points plus the positive rows live
// at that instant (starting there included, ending there excluded). Gaps emit
// nothing. Calls seg(ts, dur, live_positive, points_or_null).
//===----------------------------------------------------------------------===//
template <typename H>
void HookSweep(std::vector<Rec>& recs, H& h) {
  if (!std::is_sorted(recs.begin(), recs.end(),
                      [](const Rec& a, const Rec& b) { return a.s < b.s; }))
    std::sort(recs.begin(), recs.end(),
              [](const Rec& a, const Rec& b) { return a.s < b.s; });
  using E = std::pair<int64_t, uint32_t>;
  auto cmp = [](const E& a, const E& b) { return a.first > b.first; };
  std::vector<E> heap;
  size_t live = 0, instants = 0;
  int64_t cursor = 0, input_ts = 0;
  auto advance = [&](int64_t time) {
    if (instants) {
      h.Seg(input_ts, 0, true);
      h.ClearInstants();
      instants = 0;
    }
    while (!heap.empty() && heap.front().first <= time) {
      int64_t end = heap.front().first;
      if (live && cursor < end)
        h.Seg(cursor, end - cursor, false);
      cursor = end;
      while (!heap.empty() && heap.front().first == end) {
        h.Remove(heap.front().second);
        live--;
        std::pop_heap(heap.begin(), heap.end(), cmp);
        heap.pop_back();
      }
    }
    if (live && cursor < time)
      h.Seg(cursor, time - cursor, false);
    cursor = time;
  };
  bool first = true;
  for (uint32_t i = 0; i < recs.size(); i++) {
    const Rec& r = recs[i];
    if (first || r.s != input_ts) {
      advance(r.s);
      input_ts = r.s;
      first = false;
    }
    if (r.e > r.s) {
      h.Add(i);
      live++;
      heap.push_back({r.e, i});
      std::push_heap(heap.begin(), heap.end(), cmp);
    } else {
      h.AddInstant(i);
      instants++;
    }
  }
  if (!first)
    advance(std::numeric_limits<int64_t>::max());
}

// Keeps the live set explicitly: seg(ts, dur, live, instants_or_null).
template <typename F>
void SegmentSweep(std::vector<Rec>& recs, F seg) {
  struct Hooks {
    F& seg;
    std::vector<uint32_t> live, pos, inst;
    void Add(uint32_t i) {
      pos[i] = static_cast<uint32_t>(live.size());
      live.push_back(i);
    }
    void Remove(uint32_t i) {
      uint32_t p = pos[i], last = live.back();
      live[p] = last;
      pos[last] = p;
      live.pop_back();
    }
    void AddInstant(uint32_t i) { inst.push_back(i); }
    void ClearInstants() { inst.clear(); }
    void Seg(int64_t ts, int64_t dur, bool with_inst) {
      seg(ts, dur, live, with_inst ? &inst : nullptr);
    }
  } h{seg, {}, std::vector<uint32_t>(recs.size()), {}};
  HookSweep(recs, h);
}

// INTERVAL FLATTEN, generic: (ts, dur, rows[]) per segment.
void SolveFlatten(State& st, OutBuf& out) {
  st.MaterializeRecs();
  auto& recs = st.recs;
  SegmentSweep(recs,
               [&](int64_t ts, int64_t dur, const std::vector<uint32_t>& live,
                   const std::vector<uint32_t>* inst) {
                 Emit(out, {ts, dur});
                 BeginList(out);
                 for (uint32_t i : live)
                   ListPush(out, recs[i].row);
                 if (inst)
                   for (uint32_t i : *inst)
                     ListPush(out, recs[i].row);
               });
}

template <typename EmitSeg>
void SweepFlattenSpans(Span16* recs, size_t n, bool sorted, EmitSeg emit_seg) {
  if (n == 0)
    return;
  if (!sorted) {
    std::sort(recs, recs + n,
              [](const Span16& a, const Span16& b) { return a.s < b.s; });
  }
  int64_t inline_heap[64];
  std::vector<int64_t> overflow_heap;
  int64_t* hptr = inline_heap;
  size_t hsize = 0;
  size_t hcap = 64;
  auto cmp = [](int64_t a, int64_t b) { return a > b; };
  int64_t live_cnt = 0, inst_cnt = 0, cursor = 0, input_ts = 0;
  auto advance = [&](int64_t time) {
    if (inst_cnt) {
      emit_seg(input_ts, 0, live_cnt + inst_cnt);
      inst_cnt = 0;
    }
    while (hsize > 0 && hptr[0] <= time) {
      int64_t end = hptr[0];
      if (live_cnt && cursor < end)
        emit_seg(cursor, end - cursor, live_cnt);
      cursor = end;
      while (hsize > 0 && hptr[0] == end) {
        live_cnt--;
        if (hsize == 1) {
          hsize = 0;
          break;
        }
        std::pop_heap(hptr, hptr + hsize, cmp);
        hsize--;
      }
    }
    if (live_cnt && cursor < time)
      emit_seg(cursor, time - cursor, live_cnt);
    cursor = time;
  };
  bool first = true;
  for (size_t i = 0; i < n; i++) {
    const Span16& r = recs[i];
    if (first || r.s != input_ts) {
      advance(r.s);
      input_ts = r.s;
      first = false;
    }
    if (r.e > r.s) {
      live_cnt++;
      if (__builtin_expect(hsize == hcap, 0)) {
        overflow_heap.reserve(hcap * 2);
        overflow_heap.assign(hptr, hptr + hsize);
        hptr = overflow_heap.data();
        hcap = overflow_heap.capacity();
      }
      if (hptr == inline_heap) {
        hptr[hsize++] = r.e;
      } else {
        overflow_heap.push_back(r.e);
        hptr = overflow_heap.data();
        hsize = overflow_heap.size();
        hcap = overflow_heap.capacity();
      }
      if (hsize > 1)
        std::push_heap(hptr, hptr + hsize, cmp);
    } else {
      inst_cnt++;
    }
  }
  if (!first)
    advance(std::numeric_limits<int64_t>::max());
}

// INTERVAL FLATTEN fast path (0 sums): (ts, dur, cnt) per segment.
void SolveFlattenSum0(State& st, OutBuf& out) {
  out.Reserve(st.spans.size * 2);
  SweepFlattenSpans(
      st.spans.data, st.spans.size, st.spans_sorted,
      [&](int64_t ts, int64_t dur, int64_t cnt) { Emit(out, {ts, dur, cnt}); });
}

// INTERVAL FLATTEN fast path (1 sum): (ts, dur, cnt, s0) per segment, with v0
// stored in Rec::row and v0_valid (0/1) in Rec::side (no nested list!).
void SolveFlattenSum1(State& st, OutBuf& out) {
  auto& recs = st.recs;
  if (!std::is_sorted(recs.begin(), recs.end(),
                      [](const Rec& a, const Rec& b) { return a.s < b.s; }))
    std::sort(recs.begin(), recs.end(),
              [](const Rec& a, const Rec& b) { return a.s < b.s; });
  out.Reserve(recs.size() * 2);
  struct HE {
    int64_t end;
    int64_t val;
    int32_t valid;
  };
  auto cmp = [](const HE& a, const HE& b) { return a.end > b.end; };
  HE inline_heap[64];
  std::vector<HE> overflow_heap;
  HE* hptr = inline_heap;
  size_t hsize = 0, hcap = 64;
  int64_t live_cnt = 0, inst_cnt = 0, live_sum = 0, inst_sum = 0;
  int64_t live_held = 0, inst_held = 0, cursor = 0, input_ts = 0;
  bool any_null_out = !out.col3_valid.empty();
  auto emit_seg = [&](int64_t ts, int64_t dur, int64_t cnt, int64_t sm,
                      int64_t hd) {
    Emit(out, {ts, dur, cnt, sm});
    if (hd == 0 && !any_null_out) {
      any_null_out = true;
      out.col3_valid.assign(out.col[0].size() - 1, 1);
    }
    if (any_null_out)
      out.col3_valid.push_back(hd > 0);
  };
  auto advance = [&](int64_t time) {
    if (inst_cnt) {
      emit_seg(input_ts, 0, live_cnt + inst_cnt, live_sum + inst_sum,
               live_held + inst_held);
      inst_cnt = 0;
      inst_sum = 0;
      inst_held = 0;
    }
    while (hsize > 0 && hptr[0].end <= time) {
      int64_t end = hptr[0].end;
      if (live_cnt && cursor < end)
        emit_seg(cursor, end - cursor, live_cnt, live_sum, live_held);
      cursor = end;
      while (hsize > 0 && hptr[0].end == end) {
        const HE top = hptr[0];
        live_cnt--;
        live_sum -= top.val;
        live_held -= top.valid;
        std::pop_heap(hptr, hptr + hsize, cmp);
        hsize--;
      }
    }
    if (live_cnt && cursor < time)
      emit_seg(cursor, time - cursor, live_cnt, live_sum, live_held);
    cursor = time;
  };
  bool first = true;
  for (const Rec& r : recs) {
    if (first || r.s != input_ts) {
      advance(r.s);
      input_ts = r.s;
      first = false;
    }
    if (r.e > r.s) {
      live_cnt++;
      live_sum += r.row;
      live_held += r.side;
      if (__builtin_expect(hsize == hcap, 0)) {
        overflow_heap.reserve(hcap * 2);
        overflow_heap.assign(hptr, hptr + hsize);
        hptr = overflow_heap.data();
        hcap = overflow_heap.capacity();
      }
      if (hptr == inline_heap) {
        hptr[hsize++] = {r.e, r.row, r.side};
      } else {
        overflow_heap.push_back({r.e, r.row, r.side});
        hptr = overflow_heap.data();
        hsize = overflow_heap.size();
        hcap = overflow_heap.capacity();
      }
      if (hsize > 1)
        std::push_heap(hptr, hptr + hsize, cmp);
    } else {
      inst_cnt++;
      inst_sum += r.row;
      inst_held += r.side;
    }
  }
  if (!first)
    advance(std::numeric_limits<int64_t>::max());
}

void UpdateFlattenSum0(duckdb_function_info info,
                       duckdb_data_chunk input,
                       duckdb_aggregate_state* states) {
  idx_t n = duckdb_data_chunk_get_size(input);
  Col<int64_t> ts(duckdb_data_chunk_get_vector(input, 0));
  Col<int64_t> dur(duckdb_data_chunk_get_vector(input, 1));
  const bool all_valid = !ts.validity && !dur.validity;

  idx_t r = 0;
  while (r < n) {
    State*& st_ref = Ptr(states[r]);
    if (!st_ref)
      st_ref = AllocState();
    State* st = st_ref;
    idx_t run_end = r + 1;
    while (run_end < n && Ptr(states[run_end]) == st)
      run_end++;

    for (idx_t i = r; i < run_end; i++) {
      if (!all_valid && (!ts.Valid(i) || !dur.Valid(i)))
        continue;
      int64_t s = ts.data[i], d = dur.data[i];
      if ((s | d) < 0) {
        duckdb_aggregate_function_set_error(
            info, "pf_interval_flatten: a row's ts or dur is below zero");
        return;
      }
      if (st->spans_sorted) {
        if (s < st->last_span_s)
          st->spans_sorted = false;
        else
          st->last_span_s = s;
      }
      st->spans.push_back({s, s + d});
      st->total_recs++;
    }
    r = run_end;
  }
}

void UpdateFlattenSum1(duckdb_function_info info,
                       duckdb_data_chunk input,
                       duckdb_aggregate_state* states) {
  idx_t n = duckdb_data_chunk_get_size(input);
  Col<int64_t> ts(duckdb_data_chunk_get_vector(input, 0));
  Col<int64_t> dur(duckdb_data_chunk_get_vector(input, 1));
  Col<int64_t> val(duckdb_data_chunk_get_vector(input, 2));
  for (idx_t r = 0; r < n; r++) {
    State*& st = Ptr(states[r]);
    if (!st)
      st = new State();
    if (!ts.Valid(r) || !dur.Valid(r))
      continue;
    int64_t s = ts.data[r], d = dur.data[r];
    if (s < 0 || d < 0) {
      duckdb_aggregate_function_set_error(
          info, "pf_interval_flatten: a row's ts or dur is below zero");
      return;
    }
    bool ok = val.Valid(r);
    st->recs.push_back({s, s + d, 0, ok ? val.data[r] : 0, ok ? 1 : 0});
    st->total_recs++;
  }
}

// INTERVAL FLATTEN fast path (N sums): (ts, dur, count, sums[]) per segment.
void SolveFlattenSum(State& st, OutBuf& out) {
  auto& recs = st.recs;
  const int w = st.width;
  struct Hooks {
    State& st;
    std::vector<Rec>& recs;
    OutBuf& out;
    int w;
    int64_t live_cnt = 0, inst_cnt = 0;
    std::vector<int64_t> live_sum, inst_sum;
    std::vector<int64_t> live_held, inst_held;
    void Acc(uint32_t i,
             std::vector<int64_t>& sum,
             std::vector<int64_t>& held,
             int sign) {
      size_t base = static_cast<size_t>(recs[i].row) * w;
      for (int k = 0; k < w; k++)
        if (st.present[base + k]) {
          sum[k] += sign * st.vals[base + k];
          held[k] += sign;
        }
    }
    void Add(uint32_t i) {
      live_cnt++;
      if (w)
        Acc(i, live_sum, live_held, 1);
    }
    void Remove(uint32_t i) {
      live_cnt--;
      if (w)
        Acc(i, live_sum, live_held, -1);
    }
    void AddInstant(uint32_t i) {
      inst_cnt++;
      if (w)
        Acc(i, inst_sum, inst_held, 1);
    }
    void ClearInstants() {
      inst_cnt = 0;
      std::fill(inst_sum.begin(), inst_sum.end(), 0);
      std::fill(inst_held.begin(), inst_held.end(), 0);
    }
    void Seg(int64_t ts, int64_t dur, bool with_inst) {
      Emit(out, {ts, dur, live_cnt + (with_inst ? inst_cnt : 0)});
      BeginList(out);
      for (int k = 0; k < w; k++) {
        int64_t sm = live_sum[k] + (with_inst ? inst_sum[k] : 0);
        int64_t hd = live_held[k] + (with_inst ? inst_held[k] : 0);
        ListPush(out, sm, hd > 0);
      }
    }
  } h{st,
      recs,
      out,
      w,
      0,
      0,
      std::vector<int64_t>(w),
      std::vector<int64_t>(w),
      std::vector<int64_t>(w),
      std::vector<int64_t>(w)};
  out.Reserve(recs.size() * 2);
  HookSweep(recs, h);
}

// Reads (ts BIGINT, dur BIGINT, vals BIGINT[]) for the N-sum flatten fast path.
void UpdateFlattenSum(duckdb_function_info info,
                      duckdb_data_chunk input,
                      duckdb_aggregate_state* states) {
  idx_t n = duckdb_data_chunk_get_size(input);
  Col<int64_t> ts(duckdb_data_chunk_get_vector(input, 0));
  Col<int64_t> dur(duckdb_data_chunk_get_vector(input, 1));
  duckdb_vector lv = duckdb_data_chunk_get_vector(input, 2);
  auto* lentries = static_cast<duckdb_list_entry*>(duckdb_vector_get_data(lv));
  uint64_t* lvalid = duckdb_vector_get_validity(lv);
  duckdb_vector lchild = duckdb_list_vector_get_child(lv);
  Col<int64_t> vals(lchild);
  for (idx_t r = 0; r < n; r++) {
    State*& st = Ptr(states[r]);
    if (!st)
      st = new State();
    if (!ts.Valid(r) || !dur.Valid(r))
      continue;
    int64_t s = ts.data[r], d = dur.data[r];
    if (s < 0 || d < 0) {
      duckdb_aggregate_function_set_error(
          info, "pf_interval_flatten: a row's ts or dur is below zero");
      return;
    }
    int w = 0;
    idx_t off = 0;
    if (!lvalid || duckdb_validity_row_is_valid(lvalid, r)) {
      w = static_cast<int>(lentries[r].length);
      off = lentries[r].offset;
    }
    st->width = std::max(st->width, w);
    st->local_rows = true;
    int64_t idx = static_cast<int64_t>(st->recs.size());
    st->recs.push_back({s, s + d, d > 0 ? s + d : s + 1, idx, 0});
    st->total_recs++;
    for (int k = 0; k < w; k++) {
      bool ok = vals.Valid(off + k);
      st->vals.push_back(ok ? vals.data[off + k] : 0);
      st->present.push_back(ok);
    }
  }
}

//===----------------------------------------------------------------------===//
// INTERVAL UNION OF (N operands): elementary segments where any operand
// covers, one fragment per combination of each operand's live row (kNone
// where the operand is absent); consecutive segments carrying the same
// combination coalesce. Zero-width point segments follow FLATTEN's rules.
//===----------------------------------------------------------------------===//
void SolveUnion(State& st, OutBuf& out) {
  int n = static_cast<int>(st.param);
  if (n < 1)
    return;
  st.MaterializeRecs();
  auto& recs = st.recs;
  struct VecHash {
    size_t operator()(const std::vector<int64_t>& v) const {
      size_t h = 1469598103934665603ull;
      for (int64_t x : v)
        h = (h ^ static_cast<size_t>(x)) * 1099511628211ull;
      return h;
    }
  };
  std::unordered_map<std::vector<int64_t>, int64_t, VecHash> open, still;
  int64_t open_end = std::numeric_limits<int64_t>::min();
  std::vector<std::vector<int64_t>> per(n);
  auto emit_frag = [&](const std::vector<int64_t>& t, int64_t s, int64_t e) {
    Emit(out, {s, e - s});
    BeginList(out);
    for (int64_t v : t)
      ListPush(out, v);
  };
  auto flush = [&]() {
    for (auto& kv : open)
      emit_frag(kv.first, kv.second, open_end);
    open.clear();
  };
  auto combos = [&](auto fn) {
    std::vector<int64_t> t(n, kNone);
    std::vector<size_t> ix(n, 0);
    while (true) {
      for (int k = 0; k < n; k++)
        t[k] = per[k].empty() ? kNone : per[k][ix[k]];
      fn(t);
      int k = 0;
      for (; k < n; k++) {
        if (per[k].empty())
          continue;
        if (++ix[k] < per[k].size())
          break;
        ix[k] = 0;
      }
      if (k == n)
        break;
    }
  };
  SegmentSweep(
      recs, [&](int64_t ts, int64_t dur, const std::vector<uint32_t>& live,
                const std::vector<uint32_t>* inst) {
        for (auto& v : per)
          v.clear();
        for (uint32_t i : live)
          per[recs[i].side].push_back(recs[i].row);
        if (inst)
          for (uint32_t i : *inst)
            per[recs[i].side].push_back(recs[i].row);
        for (auto& v : per)
          std::sort(v.begin(), v.end());
        if (dur == 0) {
          combos([&](const std::vector<int64_t>& t) { emit_frag(t, ts, ts); });
          return;
        }
        if (ts != open_end)
          flush();
        still.clear();
        combos([&](const std::vector<int64_t>& t) {
          auto it = open.find(t);
          still.emplace(t, it == open.end() ? ts : it->second);
          if (it != open.end())
            open.erase(it);
        });
        // Whatever remains in `open` ended at this boundary.
        for (auto& kv : open)
          emit_frag(kv.first, kv.second, open_end);
        open.swap(still);
        open_end = ts + dur;
      });
  flush();
}

//===----------------------------------------------------------------------===//
// INTERVAL SPLIT: tile each side-0 row at side-1 boundaries; (ts, dur, r0, r1)
// with r1 = kNone where no operand row covers the piece.
//===----------------------------------------------------------------------===//
void SolveSplit(State& st, OutBuf& out) {
  std::vector<std::vector<Rec>> sides;
  SplitSides(st, 2, sides);
  auto& A = sides[0];
  auto& B = sides[1];
  // Overlapping operand rows per A row (by index in A).
  std::vector<std::pair<uint32_t, uint32_t>> pairs;
  {
    std::vector<Rec> Ai = A;
    for (size_t i = 0; i < Ai.size(); i++)
      Ai[i].row = static_cast<int64_t>(i);
    std::vector<Rec> Bi = B;
    for (size_t i = 0; i < Bi.size(); i++)
      Bi[i].row = static_cast<int64_t>(i);
    OverlapSweep(Ai, Bi, [&](const Rec& a, const Rec& b) {
      pairs.push_back(
          {static_cast<uint32_t>(a.row), static_cast<uint32_t>(b.row)});
    });
  }
  std::sort(pairs.begin(), pairs.end());
  std::vector<int64_t> cuts;
  std::vector<const Rec*> ov;
  size_t p = 0;
  for (size_t i = 0; i < A.size(); i++) {
    const Rec& a = A[i];
    ov.clear();
    while (p < pairs.size() && pairs[p].first == i)
      ov.push_back(&B[pairs[p++].second]);
    if (ov.empty()) {
      Emit(out, {a.s, a.e - a.s, a.row, kNone});
      continue;
    }
    if (a.e == a.s) {  // a point: one piece per covering operand row
      for (const Rec* b : ov)
        Emit(out, {a.s, 0, a.row, b->row});
      continue;
    }
    cuts.clear();
    cuts.push_back(a.s);
    cuts.push_back(a.e);
    for (const Rec* b : ov) {
      if (b->s > a.s && b->s < a.e)
        cuts.push_back(b->s);
      if (b->e > a.s && b->e < a.e)
        cuts.push_back(b->e);
    }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    for (size_t c = 0; c + 1 < cuts.size(); c++) {
      int64_t s = cuts[c], e = cuts[c + 1];
      // Zero-width piece for operand points at s (attached with positive
      // operand rows live at s), as FLATTEN does.
      bool has_point = false;
      for (const Rec* b : ov)
        has_point |= (b->e == b->s && b->s == s);
      if (has_point) {
        for (const Rec* b : ov)
          if ((b->e == b->s && b->s == s) ||
              (b->e > b->s && b->s <= s && s < b->e))
            Emit(out, {s, 0, a.row, b->row});
      }
      bool any = false;
      for (const Rec* b : ov)
        if (b->e > b->s && b->s <= s && e <= b->e) {
          Emit(out, {s, e - s, a.row, b->row});
          any = true;
        }
      if (!any)
        Emit(out, {s, e - s, a.row, kNone});
    }
  }
}

//===----------------------------------------------------------------------===//
// Merged coverage of a sorted set: disjoint positive intervals (overlapping
// and abutting rows coalesce) plus isolated points not inside them.
//===----------------------------------------------------------------------===//
struct Coverage {
  std::vector<int64_t> cs, ce;  // positive, disjoint, sorted
  std::vector<int64_t> pts;     // sorted points outside [cs, ce)
  void Build(const std::vector<Rec>& sorted) {
    std::vector<int64_t> raw_pts;
    for (const Rec& r : sorted) {
      if (r.e == r.s) {
        raw_pts.push_back(r.s);
        continue;
      }
      if (!cs.empty() && r.s <= ce.back()) {
        ce.back() = std::max(ce.back(), r.e);
      } else {
        cs.push_back(r.s);
        ce.push_back(r.e);
      }
    }
    for (int64_t p : raw_pts)
      if (!CoversInstantPositive(p) && (pts.empty() || pts.back() != p))
        pts.push_back(p);
  }
  // Index of the first coverage interval with ce > t.
  size_t FirstEndingAfter(int64_t t) const {
    return static_cast<size_t>(std::upper_bound(ce.begin(), ce.end(), t) -
                               ce.begin());
  }
  bool CoversInstantPositive(int64_t t) const {
    size_t k = FirstEndingAfter(t);
    return k < cs.size() && cs[k] <= t;
  }
  bool CoversInstant(int64_t t) const {
    return CoversInstantPositive(t) ||
           std::binary_search(pts.begin(), pts.end(), t);
  }
};

//===----------------------------------------------------------------------===//
// INTERVAL SUBTRACT: side-0 rows minus side-1 coverage; (ts, dur, r0).
//===----------------------------------------------------------------------===//
void SolveSubtract(State& st, OutBuf& out) {
  std::vector<std::vector<Rec>> sides;
  SplitSides(st, 2, sides);
  Coverage cov;
  cov.Build(sides[1]);
  for (const Rec& a : sides[0]) {
    if (a.e == a.s) {
      if (!cov.CoversInstant(a.s))
        Emit(out, {a.s, 0, a.row});
      continue;
    }
    int64_t cur = a.s;
    for (size_t k = cov.FirstEndingAfter(a.s);
         k < cov.cs.size() && cov.cs[k] < a.e; k++) {
      if (cov.cs[k] > cur)
        Emit(out, {cur, cov.cs[k] - cur, a.row});
      cur = std::max(cur, cov.ce[k]);
      if (cur >= a.e)
        break;
    }
    if (cur < a.e)
      Emit(out, {cur, a.e - cur, a.row});
  }
}

//===----------------------------------------------------------------------===//
// Relationships (JOIN against rows, KEEP/DROP against merged coverage).
//===----------------------------------------------------------------------===//
enum Rel : int64_t {
  kOverlapping = 0,
  kCoveringBegin = 1,
  kCoveringEnd = 2,
  kCoveringBounds = 3,
  kWithinBounds = 4,
};

// Does operand row b stand in `rel` to my row a?
inline bool RelHolds(int64_t rel, const Rec& a, const Rec& b) {
  bool a_pt = a.e == a.s, b_pt = b.e == b.s;
  auto covers_instant = [&](int64_t t) {
    return b_pt ? b.s == t : (b.s <= t && t < b.e);
  };
  switch (rel) {
    case kOverlapping:
      return Overlaps(a, b);
    case kCoveringBegin:
      return covers_instant(a.s);
    case kCoveringEnd:
      return covers_instant(a.e);
    case kCoveringBounds:
      if (a_pt)
        return covers_instant(a.s);
      return !b_pt && b.s <= a.s && a.e <= b.e;
    case kWithinBounds:
      if (a_pt)
        return b_pt && b.s == a.s;
      if (b_pt)
        return a.s <= b.s && b.s < a.e;
      return a.s <= b.s && b.e <= a.e;
  }
  return false;
}

// INTERVAL JOIN: (r0, r1) per match; r1 = kNone for unmatched rows when the
// relationship code has bit 8 set (LEFT).
void SolveJoin(State& st, OutBuf& out) {
  int64_t rel = st.param & 0xff;
  bool left = (st.param & 0x100) != 0;
  std::vector<std::vector<Rec>> sides;
  SplitSides(st, 2, sides);
  auto& A = sides[0];
  auto& B = sides[1];
  std::vector<uint8_t> matched(left ? A.size() : 0, 0);
  // Probes: for COVERING BEGIN/END, a's anchor instant as a point; otherwise
  // a itself. Every relationship implies the probe overlaps b.
  std::vector<Rec> probe(A.size());
  for (size_t i = 0; i < A.size(); i++) {
    Rec p = A[i];
    if (rel == kCoveringBegin || rel == kCoveringEnd) {
      int64_t t = rel == kCoveringBegin ? A[i].s : A[i].e;
      p = {t, t, t + 1, 0, 0};
    }
    p.row = static_cast<int64_t>(i);
    probe[i] = p;
  }
  if (rel == kCoveringEnd)
    std::sort(probe.begin(), probe.end(), ByStart);
  OverlapSweep(probe, B, [&](const Rec& p, const Rec& b) {
    const Rec& a = A[static_cast<size_t>(p.row)];
    if (!RelHolds(rel, a, b))
      return;
    Emit(out, {a.row, b.row});
    if (left)
      matched[static_cast<size_t>(p.row)] = 1;
  });
  if (left)
    for (size_t i = 0; i < A.size(); i++)
      if (!matched[i])
        Emit(out, {A[i].row, kNone});
}

// INTERVAL FIND: param encodes:
//   bit 0: edge_end (0 = starting [b.s], 1 = ending [b.e])
//   bit 1: dir_before (0 = after [y >= x], 1 = before [y <= x])
//   bits 2..3: anchor (0 = begin [a.s], 1 = end [a.e], 2 = bounds)
//   bit 8: left_join (emit (a.row, kNone) when unmatched)
void SolveFind(State& st, OutBuf& out) {
  bool edge_end = (st.param & 1) != 0;
  bool dir_before = (st.param & 2) != 0;
  int anchor = (st.param >> 2) & 3;
  bool left = (st.param & 0x100) != 0;

  std::vector<std::vector<Rec>> sides;
  SplitSides(st, 2, sides);
  auto& A = sides[0];
  auto& B = sides[1];
  if (A.empty())
    return;
  out.Reserve(A.size());
  if (B.empty()) {
    if (left) {
      for (const Rec& a : A)
        Emit(out, {a.row, kNone});
    }
    return;
  }

  auto y_of = [&](const Rec& b) { return edge_end ? b.e : b.s; };
  auto cmp_b = [&](const Rec& p, const Rec& q) {
    int64_t yp = y_of(p), yq = y_of(q);
    if (yp != yq)
      return yp < yq;
    if (dir_before) {
      if (p.s != q.s)
        return p.s > q.s;
      return (p.e - p.s) > (q.e - q.s);
    }
    if (p.s != q.s)
      return p.s < q.s;
    return (p.e - p.s) < (q.e - q.s);
  };
  if (!std::is_sorted(B.begin(), B.end(), cmp_b))
    std::sort(B.begin(), B.end(), cmp_b);

  const Rec* b_beg = B.data();
  const Rec* b_end = B.data() + B.size();
  for (const Rec& a : A) {
    int64_t x = (anchor == 0 || (anchor == 2 && dir_before)) ? a.s : a.e;
    const Rec* hit = nullptr;
    if (!dir_before) {
      const Rec* it = std::lower_bound(
          b_beg, b_end, x,
          [&](const Rec& b, int64_t v) { return y_of(b) < v; });
      if (it != b_end)
        hit = it;
    } else {
      const Rec* it = std::upper_bound(
          b_beg, b_end, x,
          [&](int64_t v, const Rec& b) { return v < y_of(b); });
      if (it != b_beg)
        hit = it - 1;
    }
    if (hit)
      Emit(out, {a.row, hit->row});
    else if (left)
      Emit(out, {a.row, kNone});
  }
}

void UpdateQuantize(duckdb_function_info info,
                    duckdb_data_chunk input,
                    duckdb_aggregate_state* states) {
  idx_t n = duckdb_data_chunk_get_size(input);
  Col<int64_t> row(duckdb_data_chunk_get_vector(input, 0));
  Col<int64_t> ts(duckdb_data_chunk_get_vector(input, 1));
  Col<int64_t> dur(duckdb_data_chunk_get_vector(input, 2));
  Col<int64_t> orig(duckdb_data_chunk_get_vector(input, 3));
  Col<int64_t> wid(duckdb_data_chunk_get_vector(input, 4));
  for (idx_t r = 0; r < n; r++) {
    State*& st = Ptr(states[r]);
    if (!st)
      st = AllocState();
    if (!ts.Valid(r) || !dur.Valid(r) || !wid.Valid(r))
      continue;
    int64_t s = ts.data[r], d = dur.data[r], w = wid.data[r];
    if (__builtin_expect(d < 0, 0)) {
      duckdb_aggregate_function_set_error(
          info, "pf_interval_quantize: dur below zero");
      return;
    }
    int64_t o = orig.Valid(r) ? orig.data[r] : 0;
    st->param = w;
    st->recs.push_back({s, d, o, row.data[r], 0});
    st->total_recs++;
  }
}

void SolveQuantize(State& st, OutBuf& out) {
  int64_t w = st.param;
  if (w <= 0 || st.recs.empty())
    return;
  out.Reserve(st.recs.size() * 2);
  auto floor_div = [w](int64_t x) -> int64_t {
    return x >= 0 ? (x / w) : ((x + 1) / w - 1);
  };
  for (const Rec& r : st.recs) {
    int64_t s = r.s, d = r.e, o = r.ep;
    int64_t k0 = floor_div(s - o);
    int64_t k1 = d == 0 ? k0 : floor_div(s + d - 1 - o);
    for (int64_t k = k0; k <= k1; k++) {
      int64_t bucket = o + k * w;
      int64_t out_ts = std::max(s, bucket);
      int64_t out_dur = d == 0 ? 0 : std::min(s + d, bucket + w) - out_ts;
      Emit(out, {r.row, out_ts, out_dur, bucket});
    }
  }
}

// Streaming vectorized scalar kernel for INTERVAL QUANTIZE:
// pf_interval_quantize_segs(ts, dur, origin, width)
//   -> LIST<STRUCT(ts BIGINT, dur BIGINT, bucket BIGINT)>
// Processes each 2048-row vector chunk directly in L1 cache across worker
// threads without allocating State or running SQL scalar arithmetic loops.
void QuantizeSegsScalar(duckdb_function_info info,
                        duckdb_data_chunk input,
                        duckdb_vector output) {
  idx_t n = duckdb_data_chunk_get_size(input);
  Col<int64_t> ts(duckdb_data_chunk_get_vector(input, 0));
  Col<int64_t> dur(duckdb_data_chunk_get_vector(input, 1));
  Col<int64_t> orig(duckdb_data_chunk_get_vector(input, 2));
  Col<int64_t> wid(duckdb_data_chunk_get_vector(input, 3));

  idx_t total = 0;
  for (idx_t r = 0; r < n; r++) {
    if (!ts.Valid(r) || !dur.Valid(r) || !wid.Valid(r))
      continue;
    int64_t s = ts.data[r], d = dur.data[r], w = wid.data[r];
    if (__builtin_expect(d < 0, 0)) {
      duckdb_scalar_function_set_error(info,
                                       "pf_interval_quantize: dur below zero");
      return;
    }
    int64_t o = orig.Valid(r) ? orig.data[r] : 0;
    int64_t d0 = s - o;
    int64_t k0 = d0 >= 0 ? (d0 / w) : ((d0 + 1) / w - 1);
    int64_t k1 = k0;
    if (d > 0) {
      int64_t d1 = s + d - 1 - o;
      k1 = d1 >= 0 ? (d1 / w) : ((d1 + 1) / w - 1);
    }
    total += static_cast<idx_t>(k1 - k0 + 1);
  }

  duckdb_list_vector_reserve(output, total);
  duckdb_list_vector_set_size(output, total);
  auto* entries =
      static_cast<duckdb_list_entry*>(duckdb_vector_get_data(output));
  duckdb_vector elem = duckdb_list_vector_get_child(output);
  auto* out_ts = static_cast<int64_t*>(
      duckdb_vector_get_data(duckdb_struct_vector_get_child(elem, 0)));
  auto* out_dur = static_cast<int64_t*>(
      duckdb_vector_get_data(duckdb_struct_vector_get_child(elem, 1)));
  auto* out_bkt = static_cast<int64_t*>(
      duckdb_vector_get_data(duckdb_struct_vector_get_child(elem, 2)));

  idx_t p = 0;
  for (idx_t r = 0; r < n; r++) {
    if (!ts.Valid(r) || !dur.Valid(r) || !wid.Valid(r)) {
      entries[r] = {p, 0};
      continue;
    }
    int64_t s = ts.data[r], d = dur.data[r], w = wid.data[r];
    int64_t o = orig.Valid(r) ? orig.data[r] : 0;
    int64_t d0 = s - o;
    int64_t k0 = d0 >= 0 ? (d0 / w) : ((d0 + 1) / w - 1);
    if (d == 0) {
      int64_t b = o + k0 * w;
      out_ts[p] = s;
      out_dur[p] = 0;
      out_bkt[p] = b;
      entries[r] = {p, 1};
      p++;
      continue;
    }
    int64_t d1 = s + d - 1 - o;
    int64_t k1 = d1 >= 0 ? (d1 / w) : ((d1 + 1) / w - 1);
    int64_t cnt = k1 - k0 + 1;
    entries[r] = {p, static_cast<idx_t>(cnt)};
    int64_t b = o + k0 * w;
    if (cnt == 1) {
      out_ts[p] = s;
      out_dur[p] = d;
      out_bkt[p] = b;
      p++;
    } else {
      int64_t first_ts = std::max(s, b);
      out_ts[p] = first_ts;
      out_dur[p] = (b + w) - first_ts;
      out_bkt[p] = b;
      b += w;
      p++;
      for (int64_t k = k0 + 1; k < k1; k++) {
        out_ts[p] = b;
        out_dur[p] = w;
        out_bkt[p] = b;
        b += w;
        p++;
      }
      out_ts[p] = b;
      out_dur[p] = (s + d) - b;
      out_bkt[p] = b;
      p++;
    }
  }
}

// Fast single-pass C++ kernel for INTERVAL MERGE CONSECUTIVE:
// pf_interval_merge_consecutive_count_agg(row, ts, dur, khash, 1)
//   -> LIST<STRUCT(ts BIGINT, dur BIGINT, r0 BIGINT, cnt BIGINT)>
void SolveMergeConsecutiveCount(State& st, OutBuf& out) {
  auto& r = st.recs;
  const size_t n = r.size();
  if (n == 0)
    return;
  if (!std::is_sorted(r.begin(), r.end(), ByStart))
    std::sort(r.begin(), r.end(), ByStart);
  out.Reserve(n / 2 + 1);
  size_t i = 0;
  while (i < n) {
    int64_t s = r[i].s, e = r[i].s + r[i].e, k = r[i].ep;
    size_t j = i + 1;
    while (j < n && r[j].s <= (r[j - 1].s + r[j - 1].e) && r[j].ep == k) {
      e = std::max(e, r[j].s + r[j].e);
      j++;
    }
    Emit(out, {s, e - s, r[i].row, static_cast<int64_t>(j - i)});
    i = j;
  }
}

// INTERVAL KEEP/DROP IF: side-0 rows whose relationship to the side-1 merged
// coverage holds; (r0).
void SolveKeep(State& st, OutBuf& out) {
  int64_t rel = st.param;
  std::vector<std::vector<Rec>> sides;
  SplitSides(st, 2, sides);
  Coverage cov;
  cov.Build(sides[1]);
  for (const Rec& a : sides[0]) {
    bool a_pt = a.e == a.s;
    bool ok = false;
    switch (rel) {
      case kOverlapping:
        if (a_pt) {
          ok = cov.CoversInstant(a.s);
        } else {
          size_t k = cov.FirstEndingAfter(a.s);
          ok = (k < cov.cs.size() && cov.cs[k] < a.e);
          if (!ok) {
            auto it = std::lower_bound(cov.pts.begin(), cov.pts.end(), a.s);
            ok = it != cov.pts.end() && *it < a.e;
          }
        }
        break;
      case kCoveringBegin:
        ok = cov.CoversInstant(a.s);
        break;
      case kCoveringEnd:
        ok = cov.CoversInstant(a.e);
        break;
      case kCoveringBounds:
        if (a_pt) {
          ok = cov.CoversInstant(a.s);
        } else {
          size_t k = cov.FirstEndingAfter(a.s);
          ok = k < cov.cs.size() && cov.cs[k] <= a.s && a.e <= cov.ce[k];
        }
        break;
      case kWithinBounds:
        if (a_pt) {
          ok = std::binary_search(cov.pts.begin(), cov.pts.end(), a.s);
        } else {
          auto it = std::lower_bound(cov.cs.begin(), cov.cs.end(), a.s);
          ok = it != cov.cs.end() &&
               cov.ce[static_cast<size_t>(it - cov.cs.begin())] <= a.e;
          if (!ok) {
            auto pt = std::lower_bound(cov.pts.begin(), cov.pts.end(), a.s);
            ok = pt != cov.pts.end() && *pt < a.e;
          }
        }
        break;
    }
    if (ok)
      Emit(out, {a.row});
  }
}

//===----------------------------------------------------------------------===//
// INTERVAL MERGE OVERLAPPING: coalesce overlapping and abutting rows (points
// merge into an interval covering or abutting them); (ts, dur, rows[]).
//===----------------------------------------------------------------------===//
void SolveMerge(State& st, OutBuf& out) {
  st.MaterializeRecs();
  auto& r = st.recs;
  if (!std::is_sorted(r.begin(), r.end(), ByStart))
    std::sort(r.begin(), r.end(), ByStart);
  size_t i = 0;
  while (i < r.size()) {
    int64_t s = r[i].s, e = r[i].e;
    size_t j = i + 1;
    while (j < r.size() && r[j].s <= e) {
      e = std::max(e, r[j].e);
      j++;
    }
    Emit(out, {s, e - s});
    BeginList(out);
    for (size_t k = i; k < j; k++)
      ListPush(out, r[k].row);
    i = j;
  }
}

void SolveMergeCount(State& st, OutBuf& out) {
  const size_t n = st.spans.size;
  if (n == 0)
    return;
  Span16* r = st.spans.data;
  if (!st.spans_sorted) {
    std::sort(r, r + n,
              [](const Span16& a, const Span16& b) { return a.s < b.s; });
    st.spans_sorted = true;
  }
  out.Reserve(n);
  size_t i = 0;
  while (i < n) {
    int64_t s = r[i].s, e = r[i].e;
    size_t j = i + 1;
    while (j < n && r[j].s <= e) {
      e = std::max(e, r[j].e);
      j++;
    }
    Emit(out, {s, e - s, static_cast<int64_t>(j - i)});
    i = j;
  }
}

//===----------------------------------------------------------------------===//
// Keyed per-trace aggregates: group outer query by trace_idx only
// (PERFECT_HASH_GROUP_BY over ~N_traces), while bucketing the inner integer key
// (track_id / utid / cpu) inside L1/L2 cache with zero heap allocations and
// dead-key pruning prior to sweep.
//===----------------------------------------------------------------------===//
struct KeySpan24 {
  uint32_t gid;
  uint32_t pad;
  int64_t s;
  int64_t e;
};

struct KeyRec32 {
  uint32_t gid;
  uint32_t pad;
  int64_t s;
  int64_t e;
  int64_t row;
};

struct RawSpan24 {
  int64_t key;
  int64_t s;
  int64_t e;
};

struct RawRec32 {
  int64_t key;
  int64_t s;
  int64_t e;
  int64_t row;
};

struct SideMeta {
  int64_t last_s = std::numeric_limits<int64_t>::min();
  int64_t min_s = std::numeric_limits<int64_t>::max();
  int64_t max_e = std::numeric_limits<int64_t>::min();
  uint32_t cnt = 0;
  uint8_t sorted = 1;
};

struct KeyState {
  static constexpr int64_t kNullKey = std::numeric_limits<int64_t>::min() + 1;
  static constexpr int64_t kNullVal = std::numeric_limits<int64_t>::min() + 3;
  bool has_null_key = false;
  bool has_null_val = false;
  ArenaVec<RawSpan24> f_raw;
  ArenaVec<RawRec32> ii_raw[2];
};

struct KeyScratch {
  OutBuf out;
  std::vector<uint8_t> col4_valid;
  std::vector<idx_t> counts;
  std::vector<int64_t> keys;
  std::vector<uint32_t> dense;
  std::unordered_map<int64_t, uint32_t> sparse;
  std::vector<uint32_t> f_cnt;
  std::vector<uint8_t> f_sorted;
  std::vector<int64_t> f_last_s;
  std::vector<uint32_t> f_gid;
  std::vector<SideMeta> ii_meta[2];
  std::vector<uint32_t> ii_gid[2];
  std::vector<uint8_t> live;
  std::vector<size_t> off0, cur0, off1, cur1;
  std::vector<Span16> flat_spans;
  std::vector<Rec24> flat0, flat1;

  KeyScratch() { dense.resize(65536, 0); }

  void ResetKeys() {
    for (int64_t k : keys) {
      if (static_cast<uint64_t>(k) < 65536ull)
        dense[static_cast<size_t>(k)] = 0;
    }
    if (!sparse.empty())
      sparse.clear();
    keys.clear();
  }

  uint32_t InternFlatten(int64_t key) {
    if (static_cast<uint64_t>(key) < 65536ull) {
      uint32_t& slot = dense[static_cast<size_t>(key)];
      if (slot)
        return slot - 1;
      uint32_t gid = static_cast<uint32_t>(keys.size());
      keys.push_back(key);
      f_cnt.push_back(0);
      f_sorted.push_back(1);
      f_last_s.push_back(std::numeric_limits<int64_t>::min());
      slot = gid + 1;
      return gid;
    }
    auto [it, inserted] =
        sparse.emplace(key, static_cast<uint32_t>(keys.size()));
    if (inserted) {
      keys.push_back(key);
      f_cnt.push_back(0);
      f_sorted.push_back(1);
      f_last_s.push_back(std::numeric_limits<int64_t>::min());
    }
    return it->second;
  }

  uint32_t LookupII(int64_t key) const {
    if (static_cast<uint64_t>(key) < 65536ull) {
      uint32_t slot = dense[static_cast<size_t>(key)];
      return slot ? (slot - 1) : UINT32_MAX;
    }
    auto it = sparse.find(key);
    return it == sparse.end() ? UINT32_MAX : it->second;
  }

  uint32_t InternII(int64_t key) {
    if (static_cast<uint64_t>(key) < 65536ull) {
      uint32_t& slot = dense[static_cast<size_t>(key)];
      if (slot)
        return slot - 1;
      uint32_t gid = static_cast<uint32_t>(keys.size());
      keys.push_back(key);
      ii_meta[0].push_back({});
      ii_meta[1].push_back({});
      slot = gid + 1;
      return gid;
    }
    auto [it, inserted] =
        sparse.emplace(key, static_cast<uint32_t>(keys.size()));
    if (inserted) {
      keys.push_back(key);
      ii_meta[0].push_back({});
      ii_meta[1].push_back({});
    }
    return it->second;
  }
};

KeyState*& KeyPtr(duckdb_aggregate_state s) {
  return *reinterpret_cast<KeyState**>(s);
}
idx_t KeyStateSize(duckdb_function_info) {
  return sizeof(KeyState*);
}
void KeyStateInit(duckdb_function_info, duckdb_aggregate_state s) {
  KeyPtr(s) = nullptr;
}
void KeyStateDestroy(duckdb_aggregate_state* s, idx_t n) {
  for (idx_t i = 0; i < n; i++) {
    delete KeyPtr(s[i]);
    KeyPtr(s[i]) = nullptr;
  }
}

void KeyStateCombine(duckdb_function_info,
                     duckdb_aggregate_state* source,
                     duckdb_aggregate_state* target,
                     idx_t count) {
  for (idx_t i = 0; i < count; i++) {
    KeyState*& src = KeyPtr(source[i]);
    if (!src)
      continue;
    KeyState*& dst = KeyPtr(target[i]);
    if (!dst) {
      dst = src;
      src = nullptr;
      continue;
    }
    if (src->f_raw.size > 0) {
      size_t n = src->f_raw.size;
      size_t cur = dst->f_raw.size;
      dst->f_raw.Reserve(cur + n);
      std::memcpy(dst->f_raw.data + cur, src->f_raw.data,
                  n * sizeof(RawSpan24));
      dst->f_raw.size = static_cast<uint32_t>(cur + n);
    }
    for (int d = 0; d < 2; d++) {
      if (src->ii_raw[d].size > 0) {
        size_t n = src->ii_raw[d].size;
        size_t cur = dst->ii_raw[d].size;
        dst->ii_raw[d].Reserve(cur + n);
        std::memcpy(dst->ii_raw[d].data + cur, src->ii_raw[d].data,
                    n * sizeof(RawRec32));
        dst->ii_raw[d].size = static_cast<uint32_t>(cur + n);
      }
    }
    dst->has_null_key |= src->has_null_key;
    dst->has_null_val |= src->has_null_val;
    delete src;
    src = nullptr;
  }
}

void UpdateFlattenSum1Key(duckdb_function_info info,
                          duckdb_data_chunk input,
                          duckdb_aggregate_state* states) {
  idx_t n = duckdb_data_chunk_get_size(input);
  Col<int64_t> key_col(duckdb_data_chunk_get_vector(input, 0));
  Col<int64_t> ts(duckdb_data_chunk_get_vector(input, 1));
  Col<int64_t> dur(duckdb_data_chunk_get_vector(input, 2));
  Col<int64_t> val(duckdb_data_chunk_get_vector(input, 3));
  const bool all_valid =
      !key_col.validity && !ts.validity && !dur.validity && !val.validity;

  idx_t r = 0;
  while (r < n) {
    KeyState*& st_ref = KeyPtr(states[r]);
    if (!st_ref)
      st_ref = new KeyState();
    KeyState* st = st_ref;
    idx_t run_end = r + 1;
    while (run_end < n && KeyPtr(states[run_end]) == st)
      run_end++;

    if (all_valid) {
      size_t need = static_cast<size_t>(run_end - r);
      st->ii_raw[0].Reserve(st->ii_raw[0].size + need);
      RawRec32* out = st->ii_raw[0].data + st->ii_raw[0].size;
      for (idx_t i = r; i < run_end; i++) {
        int64_t s = ts.data[i], d = dur.data[i];
        if (__builtin_expect((s | d) < 0, 0)) {
          duckdb_aggregate_function_set_error(
              info, "pf_interval_flatten: a row's ts or dur is below zero");
          return;
        }
        *out++ = {key_col.data[i], s, s + d, val.data[i]};
      }
      st->ii_raw[0].size += static_cast<uint32_t>(need);
    } else {
      for (idx_t i = r; i < run_end; i++) {
        if (!ts.Valid(i) || !dur.Valid(i))
          continue;
        int64_t s = ts.data[i], d = dur.data[i];
        if (__builtin_expect((s | d) < 0, 0)) {
          duckdb_aggregate_function_set_error(
              info, "pf_interval_flatten: a row's ts or dur is below zero");
          return;
        }
        int64_t key = key_col.Valid(i)
                          ? key_col.data[i]
                          : (st->has_null_key = true, KeyState::kNullKey);
        int64_t v = val.Valid(i)
                        ? val.data[i]
                        : (st->has_null_val = true, KeyState::kNullVal);
        st->ii_raw[0].push_back({key, s, s + d, v});
      }
    }
    r = run_end;
  }
}

void SolveFlattenSum1Key(KeyState& st, KeyScratch& sc) {
  const size_t total = st.ii_raw[0].size;
  if (total == 0)
    return;
  sc.ResetKeys();
  sc.f_cnt.clear();
  sc.f_sorted.clear();
  sc.f_last_s.clear();
  sc.f_gid.resize(total);

  const RawRec32* raw = st.ii_raw[0].data;
  for (size_t i = 0; i < total; i++) {
    uint32_t gid = sc.InternFlatten(raw[i].key);
    sc.f_gid[i] = gid;
    if (sc.f_sorted[gid]) {
      if (raw[i].s < sc.f_last_s[gid])
        sc.f_sorted[gid] = 0;
      else
        sc.f_last_s[gid] = raw[i].s;
    }
    sc.f_cnt[gid]++;
  }

  const size_t ng = sc.keys.size();
  sc.off0.resize(ng + 1);
  sc.cur0.resize(ng);
  size_t sum = 0;
  for (size_t g = 0; g < ng; g++) {
    sc.off0[g] = sum;
    sc.cur0[g] = sum;
    sum += sc.f_cnt[g];
  }
  sc.off0[ng] = sum;
  sc.flat0.resize(sum);
  for (size_t i = 0; i < total; i++) {
    uint32_t g = sc.f_gid[i];
    sc.flat0[sc.cur0[g]++] = {raw[i].s, raw[i].e, raw[i].row};
  }
  const bool has_null_val = st.has_null_val;
  st.ii_raw[0].clear();

  struct HE {
    int64_t end;
    int64_t val;
    int32_t valid;
  };
  auto cmp = [](const HE& a, const HE& b) { return a.end > b.end; };
  HE inline_heap[64];
  std::vector<HE> overflow_heap;
  bool any_null_out = !sc.col4_valid.empty();

  auto emit_seg = [&](int64_t key, int64_t ts, int64_t dur, int64_t cnt,
                      int64_t sm, int64_t hd) {
    Emit(sc.out, {key, ts, dur, cnt, sm});
    if (hd == 0 && !any_null_out) {
      any_null_out = true;
      sc.col4_valid.assign(sc.out.col[0].size() - 1, 1);
    }
    if (any_null_out)
      sc.col4_valid.push_back(hd > 0);
  };

  for (size_t g = 0; g < ng; g++) {
    size_t cnt = sc.f_cnt[g];
    if (cnt == 0)
      continue;
    const int64_t key = sc.keys[g];
    Rec24* spans = sc.flat0.data() + sc.off0[g];
    if (!sc.f_sorted[g]) {
      std::sort(spans, spans + cnt,
                [](const Rec24& a, const Rec24& b) { return a.s < b.s; });
    }
    HE* hptr = inline_heap;
    size_t hsize = 0, hcap = 64;
    int64_t live_cnt = 0, inst_cnt = 0, live_sum = 0, inst_sum = 0;
    int64_t live_held = 0, inst_held = 0, cursor = 0, input_ts = 0;

    auto advance = [&](int64_t time) {
      if (inst_cnt) {
        emit_seg(key, input_ts, 0, live_cnt + inst_cnt, live_sum + inst_sum,
                 live_held + inst_held);
        inst_cnt = 0;
        inst_sum = 0;
        inst_held = 0;
      }
      while (hsize > 0 && hptr[0].end <= time) {
        int64_t end = hptr[0].end;
        if (live_cnt && cursor < end)
          emit_seg(key, cursor, end - cursor, live_cnt, live_sum, live_held);
        cursor = end;
        while (hsize > 0 && hptr[0].end == end) {
          const HE top = hptr[0];
          live_cnt--;
          live_sum -= top.val;
          live_held -= top.valid;
          std::pop_heap(hptr, hptr + hsize, cmp);
          hsize--;
        }
      }
      if (live_cnt && cursor < time)
        emit_seg(key, cursor, time - cursor, live_cnt, live_sum, live_held);
      cursor = time;
    };

    bool first = true;
    for (size_t i = 0; i < cnt; i++) {
      const Rec24& r = spans[i];
      if (first || r.s != input_ts) {
        advance(r.s);
        input_ts = r.s;
        first = false;
      }
      bool ok = !has_null_val || r.row != KeyState::kNullVal;
      int64_t rv = ok ? r.row : 0;
      int32_t hd = ok ? 1 : 0;
      if (r.e > r.s) {
        live_cnt++;
        live_sum += rv;
        live_held += hd;
        if (__builtin_expect(hsize == hcap, 0)) {
          overflow_heap.reserve(hcap * 2);
          overflow_heap.assign(hptr, hptr + hsize);
          hptr = overflow_heap.data();
          hcap = overflow_heap.capacity();
        }
        if (hptr == inline_heap) {
          hptr[hsize++] = {r.e, rv, hd};
        } else {
          overflow_heap.push_back({r.e, rv, hd});
          hptr = overflow_heap.data();
          hsize = overflow_heap.size();
          hcap = overflow_heap.capacity();
        }
        if (hsize > 1)
          std::push_heap(hptr, hptr + hsize, cmp);
      } else {
        inst_cnt++;
        inst_sum += rv;
        inst_held += hd;
      }
    }
    if (!first)
      advance(std::numeric_limits<int64_t>::max());
  }
}

void UpdateFlattenSum0Key(duckdb_function_info info,
                          duckdb_data_chunk input,
                          duckdb_aggregate_state* states) {
  idx_t n = duckdb_data_chunk_get_size(input);
  Col<int64_t> key_col(duckdb_data_chunk_get_vector(input, 0));
  Col<int64_t> ts(duckdb_data_chunk_get_vector(input, 1));
  Col<int64_t> dur(duckdb_data_chunk_get_vector(input, 2));
  const bool all_valid = !key_col.validity && !ts.validity && !dur.validity;

  idx_t r = 0;
  while (r < n) {
    KeyState*& st_ref = KeyPtr(states[r]);
    if (!st_ref)
      st_ref = new KeyState();
    KeyState* st = st_ref;
    idx_t run_end = r + 1;
    while (run_end < n && KeyPtr(states[run_end]) == st)
      run_end++;

    if (all_valid) {
      size_t need = static_cast<size_t>(run_end - r);
      st->f_raw.Reserve(st->f_raw.size + need);
      RawSpan24* out = st->f_raw.data + st->f_raw.size;
      for (idx_t i = r; i < run_end; i++) {
        int64_t s = ts.data[i], d = dur.data[i];
        if (__builtin_expect((s | d) < 0, 0)) {
          duckdb_aggregate_function_set_error(
              info, "pf_interval_flatten: a row's ts or dur is below zero");
          return;
        }
        *out++ = {key_col.data[i], s, s + d};
      }
      st->f_raw.size += static_cast<uint32_t>(need);
    } else {
      for (idx_t i = r; i < run_end; i++) {
        if (!ts.Valid(i) || !dur.Valid(i))
          continue;
        int64_t s = ts.data[i], d = dur.data[i];
        if (__builtin_expect((s | d) < 0, 0)) {
          duckdb_aggregate_function_set_error(
              info, "pf_interval_flatten: a row's ts or dur is below zero");
          return;
        }
        int64_t key = key_col.Valid(i)
                          ? key_col.data[i]
                          : (st->has_null_key = true, KeyState::kNullKey);
        st->f_raw.push_back({key, s, s + d});
      }
    }
    r = run_end;
  }
}

void SolveFlattenSum0Key(KeyState& st, KeyScratch& sc) {
  const size_t total = st.f_raw.size;
  if (total == 0)
    return;
  sc.ResetKeys();
  sc.f_cnt.clear();
  sc.f_sorted.clear();
  sc.f_last_s.clear();
  sc.f_gid.resize(total);

  const RawSpan24* raw = st.f_raw.data;
  for (size_t i = 0; i < total; i++) {
    uint32_t gid = sc.InternFlatten(raw[i].key);
    sc.f_gid[i] = gid;
    if (sc.f_sorted[gid]) {
      if (raw[i].s < sc.f_last_s[gid])
        sc.f_sorted[gid] = 0;
      else
        sc.f_last_s[gid] = raw[i].s;
    }
    sc.f_cnt[gid]++;
  }

  const size_t ng = sc.keys.size();
  sc.off0.resize(ng + 1);
  sc.cur0.resize(ng);
  size_t sum = 0;
  for (size_t g = 0; g < ng; g++) {
    sc.off0[g] = sum;
    sc.cur0[g] = sum;
    sum += sc.f_cnt[g];
  }
  sc.off0[ng] = sum;
  sc.flat_spans.resize(sum);
  for (size_t i = 0; i < total; i++) {
    uint32_t g = sc.f_gid[i];
    sc.flat_spans[sc.cur0[g]++] = {raw[i].s, raw[i].e};
  }
  st.f_raw.clear();

  for (size_t g = 0; g < ng; g++) {
    size_t cnt = sc.f_cnt[g];
    if (cnt == 0)
      continue;
    const int64_t key = sc.keys[g];
    SweepFlattenSpans(sc.flat_spans.data() + sc.off0[g], cnt,
                      sc.f_sorted[g] != 0,
                      [&](int64_t ts, int64_t dur, int64_t c) {
                        Emit(sc.out, {key, ts, dur, c});
                      });
  }
}

void UpdateIntersect2Key(duckdb_function_info info,
                         duckdb_data_chunk input,
                         duckdb_aggregate_state* states) {
  idx_t n = duckdb_data_chunk_get_size(input);
  Col<int64_t> key_col(duckdb_data_chunk_get_vector(input, 0));
  Col<int32_t> side(duckdb_data_chunk_get_vector(input, 1));
  Col<int64_t> row(duckdb_data_chunk_get_vector(input, 2));
  Col<int64_t> ts(duckdb_data_chunk_get_vector(input, 3));
  Col<int64_t> dur(duckdb_data_chunk_get_vector(input, 4));
  const bool all_valid = !key_col.validity && !side.validity && !row.validity &&
                         !ts.validity && !dur.validity;

  idx_t r = 0;
  while (r < n) {
    KeyState*& st_ref = KeyPtr(states[r]);
    if (!st_ref)
      st_ref = new KeyState();
    KeyState* st = st_ref;
    idx_t run_end = r + 1;
    while (run_end < n && KeyPtr(states[run_end]) == st)
      run_end++;

    if (all_valid && side.data[r] == side.data[run_end - 1]) {
      int32_t sd0 = side.data[r] & 1;
      bool uniform = true;
      for (idx_t i = r + 1; i + 1 < run_end; i++) {
        if ((side.data[i] & 1) != sd0) {
          uniform = false;
          break;
        }
      }
      if (uniform) {
        size_t need = static_cast<size_t>(run_end - r);
        st->ii_raw[sd0].Reserve(st->ii_raw[sd0].size + need);
        RawRec32* out = st->ii_raw[sd0].data + st->ii_raw[sd0].size;
        for (idx_t i = r; i < run_end; i++) {
          int64_t s = ts.data[i], d = dur.data[i];
          if (__builtin_expect((s | d) < 0, 0)) {
            duckdb_aggregate_function_set_error(
                info, "pf_interval_intersect: a row's ts or dur is below zero");
            return;
          }
          *out++ = {key_col.data[i], s, s + d, row.data[i]};
        }
        st->ii_raw[sd0].size += static_cast<uint32_t>(need);
        r = run_end;
        continue;
      }
    }

    for (idx_t i = r; i < run_end; i++) {
      if (!all_valid) {
        if (!ts.Valid(i) || !dur.Valid(i) || !row.Valid(i) || !side.Valid(i))
          continue;
      }
      int64_t s = ts.data[i], d = dur.data[i];
      if (__builtin_expect((s | d) < 0, 0)) {
        duckdb_aggregate_function_set_error(
            info, "pf_interval_intersect: a row's ts or dur is below zero");
        return;
      }
      int32_t sd = side.data[i] & 1;
      int64_t key = all_valid
                        ? key_col.data[i]
                        : (key_col.Valid(i)
                               ? key_col.data[i]
                               : (st->has_null_key = true, KeyState::kNullKey));
      st->ii_raw[sd].push_back({key, s, s + d, row.data[i]});
    }
    r = run_end;
  }
}

__attribute__((noinline)) void SolveIntersect2Key(KeyState& st,
                                                  KeyScratch& sc) {
  const size_t n0 = st.ii_raw[0].size;
  const size_t n1 = st.ii_raw[1].size;
  if (n0 == 0 || n1 == 0) {
    st.ii_raw[0].clear();
    st.ii_raw[1].clear();
    return;
  }

  sc.ResetKeys();
  sc.ii_meta[0].clear();
  sc.ii_meta[1].clear();

  const int small_sd = (n0 <= n1) ? 0 : 1;
  const int large_sd = 1 - small_sd;
  const size_t n_small = st.ii_raw[small_sd].size;
  const size_t n_large = st.ii_raw[large_sd].size;
  const RawRec32* r_small = st.ii_raw[small_sd].data;
  const RawRec32* r_large = st.ii_raw[large_sd].data;

  sc.ii_gid[small_sd].resize(n_small);
  for (size_t i = 0; i < n_small; i++) {
    uint32_t gid = sc.InternII(r_small[i].key);
    sc.ii_gid[small_sd][i] = gid;
    SideMeta& m = sc.ii_meta[small_sd][gid];
    int64_t s = r_small[i].s, e = r_small[i].e;
    if (m.sorted) {
      if (s < m.last_s)
        m.sorted = 0;
      else
        m.last_s = s;
    }
    if (s < m.min_s)
      m.min_s = s;
    int64_t ep = e + (e == s);
    if (ep > m.max_e)
      m.max_e = ep;
    m.cnt++;
  }

  const size_t ng = sc.keys.size();
  sc.off0.resize(ng + 1);
  sc.cur0.resize(ng);
  size_t s_small = 0;
  for (size_t g = 0; g < ng; g++) {
    sc.off0[g] = s_small;
    sc.cur0[g] = s_small;
    s_small += sc.ii_meta[small_sd][g].cnt;
  }
  sc.off0[ng] = s_small;
  sc.flat0.resize(s_small);
  const uint32_t* gid_small = sc.ii_gid[small_sd].data();
  for (size_t i = 0; i < n_small; i++) {
    uint32_t g = gid_small[i];
    sc.flat0[sc.cur0[g]++] = {r_small[i].s, r_small[i].e, r_small[i].row};
  }
  auto by_start24 = [](const Rec24& a, const Rec24& b) {
    return a.s != b.s ? a.s < b.s : a.e < b.e;
  };
  for (size_t g = 0; g < ng; g++) {
    if (!sc.ii_meta[small_sd][g].sorted) {
      Rec24* a = sc.flat0.data() + sc.off0[g];
      std::sort(a, a + sc.ii_meta[small_sd][g].cnt, by_start24);
    }
    sc.cur0[g] = sc.off0[g];
  }

  // Direct uninitialized capacity write (only grow when full; zero memset):
  const size_t out_before = sc.out.size();
  size_t out_n = out_before;
  size_t cap = sc.out.col[0].size();
  if (out_n + n_large + 64 > cap) {
    cap = std::max<size_t>(cap * 2, out_n + n_large + 64);
    for (int f = 0; f < 5; f++)
      sc.out.col[f].resize(cap);
  }
  int64_t* c0 = sc.out.col[0].data();
  int64_t* c1 = sc.out.col[1].data();
  int64_t* c2 = sc.out.col[2].data();
  int64_t* c3 = sc.out.col[3].data();
  int64_t* c4 = sc.out.col[4].data();

  // Single-pass L1/L2 streaming probe when large_sd is monotonic per key:
  bool large_ordered = true;
  const uint32_t* dense_ptr = sc.dense.data();
  const bool has_sparse = !sc.sparse.empty();
  SideMeta* meta_small = sc.ii_meta[small_sd].data();
  SideMeta* meta_large = sc.ii_meta[large_sd].data();
  size_t* cur0_ptr = sc.cur0.data();
  const size_t* off0_ptr = sc.off0.data();
  const Rec24* flat_a = sc.flat0.data();

  for (size_t i = 0; i < n_large; i++) {
    int64_t key = r_large[i].key;
    uint32_t g;
    if (static_cast<uint64_t>(key) < 65536ull) {
      uint32_t slot = dense_ptr[static_cast<size_t>(key)];
      if (!slot)
        continue;
      g = slot - 1;
    } else if (has_sparse) {
      auto it = sc.sparse.find(key);
      if (it == sc.sparse.end())
        continue;
      g = it->second;
    } else {
      continue;
    }
    int64_t bs = r_large[i].s;
    if (__builtin_expect(bs < meta_large[g].last_s, 0)) {
      large_ordered = false;
      break;
    }
    meta_large[g].last_s = bs;
    const SideMeta& ms = meta_small[g];
    int64_t be = r_large[i].e;
    if (be < ms.min_s || bs > ms.max_e)
      continue;
    size_t pos = cur0_ptr[g];
    const size_t end = off0_ptr[g + 1];
    while (pos < end &&
           flat_a[pos].e + (flat_a[pos].e == flat_a[pos].s) <= bs) {
      pos++;
    }
    cur0_ptr[g] = pos;
    int64_t bp = be + (be == bs);
    int64_t brow = r_large[i].row;
    for (size_t p = pos; p < end; p++) {
      const Rec24& x = flat_a[p];
      if (x.s >= bp)
        break;
      if (x.e + (x.e == x.s) <= bs)
        continue;
      int64_t s = std::max(x.s, bs);
      int64_t e = std::max(s, std::min(x.e, be));
      if (__builtin_expect(out_n == cap, 0)) {
        cap *= 2;
        for (int f = 0; f < 5; f++)
          sc.out.col[f].resize(cap);
        c0 = sc.out.col[0].data();
        c1 = sc.out.col[1].data();
        c2 = sc.out.col[2].data();
        c3 = sc.out.col[3].data();
        c4 = sc.out.col[4].data();
      }
      c0[out_n] = key;
      c1[out_n] = s;
      c2[out_n] = e - s;
      if (small_sd == 0) {
        c3[out_n] = x.row;
        c4[out_n] = brow;
      } else {
        c3[out_n] = brow;
        c4[out_n] = x.row;
      }
      out_n++;
    }
  }
  if (large_ordered) {
    for (int f = 0; f < 5; f++)
      sc.out.col[f].resize(out_n);
    st.ii_raw[0].clear();
    st.ii_raw[1].clear();
    return;
  }

  // Fallback for out-of-order large_sd: roll back partial outputs and sweep.
  for (int f = 0; f < 5; f++)
    sc.out.col[f].resize(out_before);
  for (size_t g = 0; g < ng; g++)
    sc.ii_meta[large_sd][g] = {};

  sc.ii_gid[large_sd].resize(n_large);
  for (size_t i = 0; i < n_large; i++) {
    uint32_t gid = sc.LookupII(r_large[i].key);
    if (gid == UINT32_MAX) {
      sc.ii_gid[large_sd][i] = UINT32_MAX;
      continue;
    }
    const SideMeta& ms = sc.ii_meta[small_sd][gid];
    int64_t s = r_large[i].s, e = r_large[i].e;
    if (e < ms.min_s || s > ms.max_e) {
      sc.ii_gid[large_sd][i] = UINT32_MAX;
      continue;
    }
    sc.ii_gid[large_sd][i] = gid;
    SideMeta& ml = sc.ii_meta[large_sd][gid];
    if (ml.sorted) {
      if (s < ml.last_s)
        ml.sorted = 0;
      else
        ml.last_s = s;
    }
    if (s < ml.min_s)
      ml.min_s = s;
    int64_t ep = e + (e == s);
    if (ep > ml.max_e)
      ml.max_e = ep;
    ml.cnt++;
  }

  sc.live.assign(ng, 0);
  sc.off1.resize(ng + 1);
  sc.cur1.resize(ng);
  size_t s_large = 0;
  for (size_t g = 0; g < ng; g++) {
    const SideMeta& ms = sc.ii_meta[small_sd][g];
    const SideMeta& ml = sc.ii_meta[large_sd][g];
    bool ok =
        ms.cnt > 0 && ml.cnt > 0 && ms.max_e > ml.min_s && ml.max_e > ms.min_s;
    sc.live[g] = ok ? 1 : 0;
    sc.off1[g] = s_large;
    sc.cur1[g] = s_large;
    if (ok)
      s_large += ml.cnt;
  }
  sc.off1[ng] = s_large;
  sc.flat1.resize(s_large);
  const uint32_t* gid_large = sc.ii_gid[large_sd].data();
  for (size_t i = 0; i < n_large; i++) {
    uint32_t g = gid_large[i];
    if (g != UINT32_MAX && sc.live[g])
      sc.flat1[sc.cur1[g]++] = {r_large[i].s, r_large[i].e, r_large[i].row};
  }
  st.ii_raw[0].clear();
  st.ii_raw[1].clear();

  for (size_t g = 0; g < ng; g++) {
    if (!sc.live[g])
      continue;
    Rec24* a = sc.flat0.data() + sc.off0[g];
    Rec24* b = sc.flat1.data() + sc.off1[g];
    size_t na = sc.ii_meta[small_sd][g].cnt;
    size_t nb = sc.cur1[g] - sc.off1[g];
    if (!na || !nb)
      continue;
    if (!sc.ii_meta[large_sd][g].sorted)
      std::sort(b, b + nb, by_start24);
    const int64_t key = sc.keys[g];
    if (small_sd == 0) {
      Sweep24(a, na, b, nb, [&](const Rec24& x, const Rec24& y) {
        int64_t s = std::max(x.s, y.s);
        int64_t e = std::max(s, std::min(x.e, y.e));
        Emit(sc.out, {key, s, e - s, x.row, y.row});
      });
    } else {
      Sweep24(b, nb, a, na, [&](const Rec24& x, const Rec24& y) {
        int64_t s = std::max(x.s, y.s);
        int64_t e = std::max(s, std::min(x.e, y.e));
        Emit(sc.out, {key, s, e - s, x.row, y.row});
      });
    }
  }
}

template <int K, void (*SOLVE)(KeyState&, KeyScratch&)>
void FinalizeKeyWith(duckdb_function_info,
                     duckdb_aggregate_state* source,
                     duckdb_vector result,
                     idx_t count,
                     idx_t offset) {
  static thread_local KeyScratch tls_sc;
  KeyScratch& sc = tls_sc;
  OutBuf& tls_out = sc.out;
  std::vector<idx_t>& tls_counts = sc.counts;
  tls_out.k = K;
  tls_out.has_list = false;
  for (int f = 0; f < K; f++)
    tls_out.col[f].clear();
  sc.col4_valid.clear();
  tls_counts.assign(count, 0);
  bool any_null_key = false;

  for (idx_t i = 0; i < count; i++) {
    idx_t before = tls_out.size();
    KeyState* st = KeyPtr(source[i]);
    if (st) {
      any_null_key |= st->has_null_key;
      SOLVE(*st, sc);
    }
    tls_counts[i] = tls_out.size() - before;
  }

  idx_t base0 = duckdb_list_vector_get_size(result);
  idx_t total_n = tls_out.size();
  duckdb_list_vector_reserve(result, base0 + total_n);
  auto* entries =
      static_cast<duckdb_list_entry*>(duckdb_vector_get_data(result));
  duckdb_vector child = duckdb_list_vector_get_child(result);
  for (int f = 0; f < K; f++) {
    int64_t* dst = static_cast<int64_t*>(
        duckdb_vector_get_data(duckdb_struct_vector_get_child(child, f)));
    if (total_n)
      std::memcpy(dst + base0, tls_out.col[f].data(),
                  total_n * sizeof(int64_t));
  }
  if (any_null_key && total_n) {
    duckdb_vector c0 = duckdb_struct_vector_get_child(child, 0);
    duckdb_vector_ensure_validity_writable(c0);
    uint64_t* val0 = duckdb_vector_get_validity(c0);
    for (idx_t i = 0; i < total_n; i++) {
      if (tls_out.col[0][i] == KeyState::kNullKey)
        duckdb_validity_set_row_invalid(val0, base0 + i);
    }
  }
  if (K == 5 && !sc.col4_valid.empty() && total_n) {
    duckdb_vector c4 = duckdb_struct_vector_get_child(child, 4);
    duckdb_vector_ensure_validity_writable(c4);
    uint64_t* val4 = duckdb_vector_get_validity(c4);
    for (idx_t i = 0; i < total_n; i++) {
      if (!sc.col4_valid[i])
        duckdb_validity_set_row_invalid(val4, base0 + i);
    }
  }
  idx_t base = base0;
  for (idx_t r = 0; r < count; r++) {
    entries[offset + r].offset = base;
    entries[offset + r].length = tls_counts[r];
    base += tls_counts[r];
  }
  duckdb_list_vector_set_size(result, base);
}

//===----------------------------------------------------------------------===//
// Registration.
//===----------------------------------------------------------------------===//
duckdb_logical_type BigInt() {
  return duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
}

duckdb_logical_type OutType(const std::vector<const char*>& scalars,
                            const char* list_name) {
  std::vector<duckdb_logical_type> types;
  std::vector<const char*> names = scalars;
  for (size_t i = 0; i < scalars.size(); i++)
    types.push_back(BigInt());
  if (list_name) {
    duckdb_logical_type b = BigInt();
    types.push_back(duckdb_create_list_type(b));
    duckdb_destroy_logical_type(&b);
    names.push_back(list_name);
  }
  duckdb_logical_type st = duckdb_create_struct_type(
      types.data(), names.data(), static_cast<idx_t>(types.size()));
  for (auto& t : types)
    duckdb_destroy_logical_type(&t);
  duckdb_logical_type list = duckdb_create_list_type(st);
  duckdb_destroy_logical_type(&st);
  return list;
}

bool Register(duckdb_connection conn,
              const char* name,
              const std::vector<duckdb_logical_type>& params,
              duckdb_logical_type ret,
              duckdb_aggregate_update_t update,
              duckdb_aggregate_finalize_t finalize) {
  duckdb_aggregate_function f = duckdb_create_aggregate_function();
  duckdb_aggregate_function_set_name(f, name);
  for (duckdb_logical_type t : params) {
    duckdb_aggregate_function_add_parameter(f, t);
    duckdb_logical_type tmp = t;
    duckdb_destroy_logical_type(&tmp);
  }
  duckdb_aggregate_function_set_return_type(f, ret);
  duckdb_aggregate_function_set_functions(f, StateSize, StateInit, update,
                                          StateCombine, finalize);
  duckdb_aggregate_function_set_destructor(f, StateDestroy);
  duckdb_aggregate_function_set_special_handling(f);
  bool ok = duckdb_register_aggregate_function(conn, f) == DuckDBSuccess;
  duckdb_destroy_aggregate_function(&f);
  duckdb_destroy_logical_type(&ret);
  return ok;
}

duckdb_logical_type T(duckdb_type t) {
  return duckdb_create_logical_type(t);
}

std::vector<duckdb_logical_type> Generic(bool with_param) {
  std::vector<duckdb_logical_type> v = {
      T(DUCKDB_TYPE_INTEGER), T(DUCKDB_TYPE_BIGINT), T(DUCKDB_TYPE_BIGINT),
      T(DUCKDB_TYPE_BIGINT)};
  if (with_param)
    v.push_back(T(DUCKDB_TYPE_INTEGER));
  return v;
}

void UpdateFromEvents(duckdb_function_info,
                      duckdb_data_chunk input,
                      duckdb_aggregate_state* states) {
  idx_t n = duckdb_data_chunk_get_size(input);
  Col<int32_t> side(duckdb_data_chunk_get_vector(input, 0));
  Col<int64_t> row(duckdb_data_chunk_get_vector(input, 1));
  Col<int64_t> ts(duckdb_data_chunk_get_vector(input, 2));
  Col<int64_t> close_at(duckdb_data_chunk_get_vector(input, 3));
  const int32_t* param = static_cast<const int32_t*>(
      duckdb_vector_get_data(duckdb_data_chunk_get_vector(input, 4)));
  for (idx_t r = 0; r < n; r++) {
    State*& st = Ptr(states[r]);
    if (!st)
      st = new State();
    st->param = param[r];
    if (!ts.Valid(r) || !side.Valid(r))
      continue;
    int32_t sd = side.data[r];
    if (sd == 0 && !row.Valid(r))
      continue;
    int64_t cl = close_at.Valid(r) ? close_at.data[r] : 0;
    st->recs.push_back({ts.data[r], cl, 0, sd == 0 ? row.data[r] : 0, sd});
    st->total_recs++;
  }
}

void SolveFromEvents(State& st, OutBuf& out) {
  auto& recs = st.recs;
  if (recs.empty())
    return;
  auto by_ev = [](const Rec& a, const Rec& b) {
    return a.s != b.s ? a.s < b.s : a.row < b.row;
  };
  bool has_stops = false;
  for (const Rec& r : recs) {
    if (r.side != 0) {
      has_stops = true;
      break;
    }
  }
  if (!has_stops) {
    if (!std::is_sorted(recs.begin(), recs.end(), by_ev))
      std::stable_sort(recs.begin(), recs.end(), by_ev);
    size_t n = recs.size();
    out.Reserve(n);
    for (size_t i = 0; i + 1 < n; i++)
      Emit(out, {recs[i].row, recs[i].s, recs[i + 1].s - recs[i].s});
    if (st.param == 1)
      Emit(out, {recs[n - 1].row, recs[n - 1].s, -1});
    else if (st.param == 2)
      Emit(out,
           {recs[n - 1].row, recs[n - 1].s, recs[n - 1].e - recs[n - 1].s});
    return;
  }
  std::vector<Rec> ev;
  std::vector<int64_t> stops;
  ev.reserve(recs.size());
  for (const Rec& r : recs) {
    if (r.side == 0)
      ev.push_back(r);
    else
      stops.push_back(r.s);
  }
  if (ev.empty())
    return;
  if (!std::is_sorted(ev.begin(), ev.end(), by_ev))
    std::stable_sort(ev.begin(), ev.end(), by_ev);
  if (!std::is_sorted(stops.begin(), stops.end()))
    std::sort(stops.begin(), stops.end());
  out.Reserve(ev.size());
  size_t sc = 0, n = ev.size();
  for (size_t i = 0; i < n; i++) {
    while (sc < stops.size() && stops[sc] < ev[i].s)
      sc++;
    bool he = i + 1 < n, hs = sc < stops.size();
    if (he || hs) {
      int64_t end_ts = (he && hs) ? std::min(ev[i + 1].s, stops[sc])
                                  : (he ? ev[i + 1].s : stops[sc]);
      Emit(out, {ev[i].row, ev[i].s, end_ts - ev[i].s});
    } else if (st.param == 1) {
      Emit(out, {ev[i].row, ev[i].s, -1});
    } else if (st.param == 2) {
      Emit(out, {ev[i].row, ev[i].s, ev[i].e - ev[i].s});
    }
  }
}

bool RegisterKey(duckdb_connection conn,
                 const char* name,
                 const std::vector<duckdb_logical_type>& params,
                 duckdb_logical_type ret,
                 duckdb_aggregate_update_t update,
                 duckdb_aggregate_finalize_t finalize) {
  duckdb_aggregate_function f = duckdb_create_aggregate_function();
  duckdb_aggregate_function_set_name(f, name);
  for (duckdb_logical_type t : params) {
    duckdb_aggregate_function_add_parameter(f, t);
    duckdb_logical_type tmp = t;
    duckdb_destroy_logical_type(&tmp);
  }
  duckdb_aggregate_function_set_return_type(f, ret);
  duckdb_aggregate_function_set_functions(f, KeyStateSize, KeyStateInit, update,
                                          KeyStateCombine, finalize);
  duckdb_aggregate_function_set_destructor(f, KeyStateDestroy);
  duckdb_aggregate_function_set_special_handling(f);
  bool ok = duckdb_register_aggregate_function(conn, f) == DuckDBSuccess;
  duckdb_destroy_aggregate_function(&f);
  duckdb_destroy_logical_type(&ret);
  return ok;
}

}  // namespace

bool RegisterIntervalFunctions(duckdb_connection conn) {
  bool ok = true;
  // (side, row, ts, dur, n_operands) -> [(ts, dur, rows[n])]
  ok &= Register(conn, "pf_interval_intersect_agg", Generic(true),
                 OutType({"ts", "dur"}, "rows"), UpdateGeneric,
                 FinalizeWith<2, true, SolveIntersect>);
  ok &= Register(conn, "pf_interval_intersect2_agg", Generic(false),
                 OutType({"ts", "dur", "r0", "r1"}, nullptr), UpdateGeneric,
                 FinalizeWith<4, false, SolveIntersect2>);
  ok &= RegisterKey(
      conn, "pf_interval_intersect2_key_agg",
      {T(DUCKDB_TYPE_BIGINT), T(DUCKDB_TYPE_INTEGER), T(DUCKDB_TYPE_BIGINT),
       T(DUCKDB_TYPE_BIGINT), T(DUCKDB_TYPE_BIGINT)},
      OutType({"key", "ts", "dur", "r0", "r1"}, nullptr), UpdateIntersect2Key,
      FinalizeKeyWith<5, SolveIntersect2Key>);
  ok &= Register(conn, "pf_interval_intersect3_agg", Generic(false),
                 OutType({"ts", "dur", "r0", "r1", "r2"}, nullptr),
                 UpdateGeneric, FinalizeWith<5, false, SolveIntersect3>);
  ok &= Register(conn, "pf_intervals_from_events_agg", Generic(true),
                 OutType({"id", "ts", "dur"}, nullptr), UpdateFromEvents,
                 FinalizeWith<3, false, SolveFromEvents>);
  ok &= Register(conn, "pf_interval_union_agg", Generic(true),
                 OutType({"ts", "dur"}, "rows"), UpdateGeneric,
                 FinalizeWith<2, true, SolveUnion>);
  // (side, row, ts, dur) -> [(ts, dur, rows[])]
  ok &= Register(conn, "pf_interval_flatten_agg", Generic(false),
                 OutType({"ts", "dur"}, "rows"), UpdateGeneric,
                 FinalizeWith<2, true, SolveFlatten>);
  ok &= Register(conn, "pf_interval_merge_agg", Generic(false),
                 OutType({"ts", "dur"}, "rows"), UpdateGeneric,
                 FinalizeWith<2, true, SolveMerge>);
  ok &= Register(conn, "pf_interval_merge_count_agg",
                 {T(DUCKDB_TYPE_BIGINT), T(DUCKDB_TYPE_BIGINT)},
                 OutType({"ts", "dur", "cnt"}, nullptr), UpdateFlattenSum0,
                 FinalizeWith<3, false, SolveMergeCount>);
  // Flatten fast paths: 0-sum, 1-sum, N-sum
  ok &= Register(conn, "pf_interval_flatten_sum0_agg",
                 {T(DUCKDB_TYPE_BIGINT), T(DUCKDB_TYPE_BIGINT)},
                 OutType({"ts", "dur", "cnt"}, nullptr), UpdateFlattenSum0,
                 FinalizeWith<3, false, SolveFlattenSum0>);
  ok &= RegisterKey(
      conn, "pf_interval_flatten_sum0_key_agg",
      {T(DUCKDB_TYPE_BIGINT), T(DUCKDB_TYPE_BIGINT), T(DUCKDB_TYPE_BIGINT)},
      OutType({"key", "ts", "dur", "cnt"}, nullptr), UpdateFlattenSum0Key,
      FinalizeKeyWith<4, SolveFlattenSum0Key>);
  ok &= Register(
      conn, "pf_interval_flatten_sum1_agg",
      {T(DUCKDB_TYPE_BIGINT), T(DUCKDB_TYPE_BIGINT), T(DUCKDB_TYPE_BIGINT)},
      OutType({"ts", "dur", "cnt", "s0"}, nullptr), UpdateFlattenSum1,
      FinalizeWith<4, false, SolveFlattenSum1>);
  ok &= RegisterKey(conn, "pf_interval_flatten_sum1_key_agg",
                    {T(DUCKDB_TYPE_BIGINT), T(DUCKDB_TYPE_BIGINT),
                     T(DUCKDB_TYPE_BIGINT), T(DUCKDB_TYPE_BIGINT)},
                    OutType({"key", "ts", "dur", "cnt", "s0"}, nullptr),
                    UpdateFlattenSum1Key,
                    FinalizeKeyWith<5, SolveFlattenSum1Key>);
  {
    duckdb_logical_type b = BigInt();
    std::vector<duckdb_logical_type> p = {T(DUCKDB_TYPE_BIGINT),
                                          T(DUCKDB_TYPE_BIGINT),
                                          duckdb_create_list_type(b)};
    duckdb_destroy_logical_type(&b);
    ok &= Register(conn, "pf_interval_flatten_sum_agg", p,
                   OutType({"ts", "dur", "cnt"}, "sums"), UpdateFlattenSum,
                   FinalizeWith<3, true, SolveFlattenSum>);
  }
  ok &= Register(conn, "pf_interval_split_agg", Generic(false),
                 OutType({"ts", "dur", "r0", "r1"}, nullptr), UpdateGeneric,
                 FinalizeWith<4, false, SolveSplit>);
  ok &= Register(conn, "pf_interval_subtract_agg", Generic(false),
                 OutType({"ts", "dur", "r0"}, nullptr), UpdateGeneric,
                 FinalizeWith<3, false, SolveSubtract>);
  // (side, row, ts, dur, rel | LEFT<<8) -> [(r0, r1)]
  ok &= Register(conn, "pf_interval_join_agg", Generic(true),
                 OutType({"r0", "r1"}, nullptr), UpdateGeneric,
                 FinalizeWith<2, false, SolveJoin>);
  ok &= Register(conn, "pf_interval_find_agg", Generic(true),
                 OutType({"r0", "r1"}, nullptr), UpdateGeneric,
                 FinalizeWith<2, false, SolveFind>);
  ok &= Register(
      conn, "pf_interval_quantize_agg",
      {T(DUCKDB_TYPE_BIGINT), T(DUCKDB_TYPE_BIGINT), T(DUCKDB_TYPE_BIGINT),
       T(DUCKDB_TYPE_BIGINT), T(DUCKDB_TYPE_BIGINT)},
      OutType({"r0", "ts", "dur", "bucket"}, nullptr), UpdateQuantize,
      FinalizeWith<4, false, SolveQuantize>);
  ok &= Register(conn, "pf_interval_keep_agg", Generic(true),
                 OutType({"r0"}, nullptr), UpdateGeneric,
                 FinalizeWith<1, false, SolveKeep>);
  {
    duckdb_scalar_function sf = duckdb_create_scalar_function();
    duckdb_scalar_function_set_name(sf, "pf_interval_quantize_segs");
    for (int i = 0; i < 4; i++) {
      duckdb_logical_type b = BigInt();
      duckdb_scalar_function_add_parameter(sf, b);
      duckdb_destroy_logical_type(&b);
    }
    duckdb_logical_type ret = OutType({"ts", "dur", "bucket"}, nullptr);
    duckdb_scalar_function_set_return_type(sf, ret);
    duckdb_scalar_function_set_function(sf, QuantizeSegsScalar);
    duckdb_scalar_function_set_special_handling(sf);
    ok &= duckdb_register_scalar_function(conn, sf) == DuckDBSuccess;
    duckdb_destroy_scalar_function(&sf);
    duckdb_destroy_logical_type(&ret);
  }
  ok &= Register(
      conn, "pf_interval_merge_consecutive_count_agg",
      {T(DUCKDB_TYPE_BIGINT), T(DUCKDB_TYPE_BIGINT), T(DUCKDB_TYPE_BIGINT),
       T(DUCKDB_TYPE_BIGINT), T(DUCKDB_TYPE_BIGINT)},
      OutType({"ts", "dur", "r0", "cnt"}, nullptr), UpdateQuantize,
      FinalizeWith<4, false, SolveMergeConsecutiveCount>);
  return ok;
}
