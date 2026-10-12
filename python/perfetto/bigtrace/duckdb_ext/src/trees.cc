// Tree operators (spec §6). Owned by the trees agent.
//
// Native kernels are AGGREGATE functions: DuckDB's parallel hash aggregate
// does the PER partitioning (GROUP BY trace_idx, ...), every group's state
// collects (id, parent_id, value...) columns, and finalize runs one O(n) pass
// over the group's forest and returns LIST<STRUCT(id, ...)> sorted by id
// (so several kernels over the same group zip row by row), consumed with
// `unnest(..., recursive := true)`. See sql/trees.sql for the operator layer.
//
// The forest: NULL parent_id = root; ids are arbitrary BIGINTs. TP slice
// tables (ids 0..n-1, parent before child) take a no-sort, no-lookup fast
// path. Errors follow TP's TREE operators: a row has no id / a node is its
// own parent / more than one row has the same node / a row names a parent
// which is not itself a row in the input (only when strict; otherwise that
// row is a root) / the rows are not a tree (cycle).
//
// Kernels:
//   pf_tree_accumulate_agg(id, parent_id, val T, dir INT, fn INT)
//       -> STRUCT(id, v T)[]                         T = BIGINT | DOUBLE
//       dir 0 = UP (subtree incl. self), 1 = DOWN (root->node path)
//       fn 0 SUM (NULL counts 0; BIGINT overflow is an error), 1 MIN, 2 MAX
//          (NULLs ignored; NULL if none), 3 COUNT(*), 4 COUNT(val)
//   pf_tree_accumulate_id_agg(id, parent_id, val T, dir INT, fn INT)
//       -> STRUCT(id, v BIGINT)[]: id of the chosen node (NULL if none):
//       fn 5 ARG_MIN, 6 ARG_MAX (ties: smallest id), 7 FIRST (DOWN: path
//       root), 8 LAST (DOWN: the node itself)
//   pf_tree_flags_agg(id, parent_id, mark BOOL, mode INT, strict BOOL)
//       -> BIGINT[]: ids for which the flag holds. mode 0 has a marked strict
//       ancestor, 1 has a marked strict descendant, 2 marked or has a marked
//       ancestor, 3 marked or has a marked descendant
//   pf_tree_nearest_agg(id, parent_id, inset BOOL, mode INT, strict BOOL)
//       -> STRUCT(id, v)[]: mode 0 nearest strict ancestor in the set,
//       mode 1 topmost ancestor-or-self in the set (NULL if none)
//   pf_tree_merge_siblings_agg(id, parent_id, key BIGINT, ordered BOOL)
//       -> STRUCT(id, mid, pmid)[]: merged node by root-to-node key path
//       (mids dense per forest); ORDERED unifies only consecutive siblings
//       (id order) with equal keys
//   pf_tree_merge_into_parent_agg(id, parent_id, key BIGINT, indirect BOOL)
//       -> STRUCT(id, owner, parent_id)[]: surviving node each row folds into
//       (itself if it survives) and the new parent of survivors
//   pf_tree_invert_agg(id, parent_id)
//       -> STRUCT(id, parent_id, node_id, leaf_id)[] (new dense ids): one
//       reversed path per leaf
//   pf_tree_ancestors_agg(id, parent_id, strict BOOL)
//       -> STRUCT(id, aid, dist)[]: all (node, strict ancestor, distance)
#include "common.h"  // no-include-violation-check

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int32_t kNone = -1;
constexpr int64_t kNullParent = INT64_MIN;

int NumThreads() {
  const char* e = getenv("PF_TREE_THREADS");
  int v = e ? atoi(e) : 48;
  return v < 1 ? 1 : v;
}

template <typename F>
void ParallelFor(idx_t n, F&& fn) {
  int nt = static_cast<int>(std::min<idx_t>(n, NumThreads()));
  if (nt <= 1) {
    for (idx_t i = 0; i < n; i++)
      fn(i);
    return;
  }
  std::atomic<idx_t> next{0};
  std::vector<std::thread> ts;
  ts.reserve(nt);
  for (int t = 0; t < nt; t++)
    ts.emplace_back([&] {
      for (idx_t i; (i = next.fetch_add(1)) < n;)
        fn(i);
    });
  for (auto& t : ts)
    t.join();
}

// ---------------------------------------------------------------------------
// Fast validity bitmask helpers & thread-local 8K-row chunk pool.
// ---------------------------------------------------------------------------
inline bool BitValid(const uint64_t* v, idx_t r) {
  return !v || ((v[r >> 6] >> (r & 63)) & 1);
}

struct Chunk {
  static constexpr uint32_t kCap = 8192;
  uint32_t len = 0;
  int64_t id[kCap];
  int64_t parent[kCap];
  int64_t v[kCap];
};

struct ChunkPool {
  std::vector<Chunk*> free_list;
  ~ChunkPool() {
    for (Chunk* c : free_list)
      delete c;
  }
};

static thread_local ChunkPool tls_chunk_pool;

inline Chunk* AllocChunk() {
  if (!tls_chunk_pool.free_list.empty()) {
    Chunk* c = tls_chunk_pool.free_list.back();
    tls_chunk_pool.free_list.pop_back();
    c->len = 0;
    return c;
  }
  return new Chunk();
}

inline void FreeChunk(Chunk* c) {
  if (tls_chunk_pool.free_list.size() < 32) {
    c->len = 0;
    tls_chunk_pool.free_list.push_back(c);
  } else {
    delete c;
  }
}

// ---------------------------------------------------------------------------
// Aggregate state: the columns of one group.
// ---------------------------------------------------------------------------
struct State {
  std::vector<Chunk*> chunks;       // unrolled 8K blocks (zero realloc copies)
  idx_t chunk_rows = 0;             // rows stored across `chunks`
  std::vector<int64_t> id, parent;  // parent kNullParent = NULL
  std::vector<int64_t> v;           // value bits (int64 or double), key
  std::vector<int64_t> v2;          // second int64 value (e.g. sum payload)
  std::vector<uint8_t> vnull;       // value is NULL (empty when all valid)
  std::vector<uint8_t> flag;        // boolean input
  int32_t p1 = 0, p2 = 0;           // constant parameters
  bool have_params = false;
  bool id_null = false;
  std::vector<State*> parts;

  ~State() {
    for (Chunk* c : chunks)
      FreeChunk(c);
    for (State* p : parts)
      delete p;
  }

  bool ValNull(idx_t i) const { return !vnull.empty() && vnull[i]; }

  void EnsureVnull() {
    MaterializeChunks();
    if (vnull.size() < v.size())
      vnull.resize(v.size(), 0);
  }

  void MaterializeChunks() {
    if (chunks.empty())
      return;
    if (chunks.size() > 1 &&
        !std::is_sorted(chunks.begin(), chunks.end(),
                        [](const Chunk* a, const Chunk* b) {
                          return a->id[0] < b->id[0];
                        })) {
      std::sort(
          chunks.begin(), chunks.end(),
          [](const Chunk* a, const Chunk* b) { return a->id[0] < b->id[0]; });
    }
    size_t old_n = id.size();
    size_t total = old_n + chunk_rows;
    id.resize(total);
    parent.resize(total);
    v.resize(total);
    size_t pos = old_n;
    for (Chunk* c : chunks) {
      std::memcpy(id.data() + pos, c->id, c->len * 8);
      std::memcpy(parent.data() + pos, c->parent, c->len * 8);
      std::memcpy(v.data() + pos, c->v, c->len * 8);
      pos += c->len;
      FreeChunk(c);
    }
    chunks.clear();
    chunk_rows = 0;
  }

  void Merge(State& o) {
    if (!have_params && o.have_params) {
      p1 = o.p1;
      p2 = o.p2;
      have_params = true;
    }
    id_null |= o.id_null;
    if (!o.chunks.empty() && id.empty() && o.id.empty() && parts.empty() &&
        o.parts.empty() && v2.empty() && o.v2.empty() && vnull.empty() &&
        o.vnull.empty() && flag.empty() && o.flag.empty()) {
      for (Chunk* c : o.chunks)
        chunks.push_back(c);
      chunk_rows += o.chunk_rows;
      o.chunks.clear();
      o.chunk_rows = 0;
      return;
    }
    MaterializeChunks();
    o.MaterializeChunks();
    if (!o.id.empty()) {
      auto* m = new State();
      m->id.swap(o.id);
      m->parent.swap(o.parent);
      m->v.swap(o.v);
      m->v2.swap(o.v2);
      m->vnull.swap(o.vnull);
      m->flag.swap(o.flag);
      parts.push_back(m);
    }
    for (State* p : o.parts)
      parts.push_back(p);
    o.parts.clear();
  }

  void Flatten() {
    MaterializeChunks();
    if (parts.empty())
      return;
    std::vector<State*> all;
    std::vector<State*> st;
    if (!id.empty()) {
      auto* self = new State();
      self->id.swap(id);
      self->parent.swap(parent);
      self->v.swap(v);
      self->v2.swap(v2);
      self->vnull.swap(vnull);
      self->flag.swap(flag);
      all.push_back(self);
    }
    for (State* p : parts)
      st.push_back(p);
    parts.clear();
    while (!st.empty()) {
      State* p = st.back();
      st.pop_back();
      p->MaterializeChunks();
      for (State* q : p->parts)
        st.push_back(q);
      p->parts.clear();
      if (!p->id.empty()) {
        all.push_back(p);
      } else {
        delete p;
      }
    }
    std::sort(all.begin(), all.end(),
              [](State* a, State* b) { return a->id.front() < b->id.front(); });
    size_t n = 0;
    bool any_v = false, any_v2 = false, any_vnull = false, any_flag = false;
    for (State* p : all) {
      n += p->id.size();
      any_v |= !p->v.empty();
      any_v2 |= !p->v2.empty();
      any_vnull |= !p->vnull.empty();
      any_flag |= !p->flag.empty();
    }
    id.reserve(n);
    parent.reserve(n);
    if (any_v)
      v.reserve(n);
    if (any_v2)
      v2.reserve(n);
    if (any_vnull)
      vnull.reserve(n);
    if (any_flag)
      flag.reserve(n);
    for (State* p : all) {
      id.insert(id.end(), p->id.begin(), p->id.end());
      parent.insert(parent.end(), p->parent.begin(), p->parent.end());
      if (any_v)
        v.insert(v.end(), p->v.begin(), p->v.end());
      if (any_v2)
        v2.insert(v2.end(), p->v2.begin(), p->v2.end());
      if (any_vnull) {
        if (p->vnull.empty()) {
          vnull.insert(vnull.end(), p->id.size(), 0);
        } else {
          vnull.insert(vnull.end(), p->vnull.begin(), p->vnull.end());
        }
      }
      if (any_flag)
        flag.insert(flag.end(), p->flag.begin(), p->flag.end());
      delete p;
    }
  }
};

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
    delete Ptr(states[i]);
    Ptr(states[i]) = nullptr;
  }
}
void StateCombine(duckdb_function_info,
                  duckdb_aggregate_state* src,
                  duckdb_aggregate_state* dst,
                  idx_t count) {
  for (idx_t i = 0; i < count; i++) {
    State*& s = Ptr(src[i]);
    State*& d = Ptr(dst[i]);
    if (!s)
      continue;
    if (!d) {
      d = s;
      s = nullptr;
      continue;
    }
    d->Merge(*s);
  }
}

struct Col {
  const void* data;
  uint64_t* validity;
  explicit Col(duckdb_vector v)
      : data(duckdb_vector_get_data(v)),
        validity(duckdb_vector_get_validity(v)) {}
  bool Valid(idx_t r) const { return BitValid(validity, r); }
  template <typename T>
  T At(idx_t r) const {
    return static_cast<const T*>(data)[r];
  }
};

// Sets dst[k - r] = null_val for every row k in [r, e) where validity bit is 0.
inline void PatchNullsI64(int64_t* dst,
                          const uint64_t* val,
                          idx_t r,
                          idx_t e,
                          int64_t null_val) {
  if (!val)
    return;
  idx_t w0 = r >> 6, w1 = (e - 1) >> 6;
  for (idx_t w = w0; w <= w1; w++) {
    uint64_t inv = ~val[w];
    if (!inv)
      continue;
    while (inv) {
      idx_t bit = static_cast<idx_t>(__builtin_ctzll(inv));
      idx_t q = (w << 6) + bit;
      if (q >= r && q < e)
        dst[q - r] = null_val;
      inv &= inv - 1;
    }
  }
}

inline bool RangeAllValid(const uint64_t* val, idx_t r, idx_t e) {
  if (!val || r >= e)
    return true;
  idx_t w0 = r >> 6, w1 = (e - 1) >> 6;
  for (idx_t w = w0; w <= w1; w++) {
    uint64_t inv = ~val[w];
    if (w == w0 && (r & 63))
      inv &= ~0ULL << (r & 63);
    if (w == w1 && (e & 63))
      inv &= (1ULL << (e & 63)) - 1;
    if (inv)
      return false;
  }
  return true;
}

// Input layout of an update: (id, parent_id [, value] [, p1 [, p2]]).
enum class ValKind { kNone, kI64, kF64, kBool, kBoolI64, kI64I64 };

template <ValKind K, int NParams>
void Update(duckdb_function_info,
            duckdb_data_chunk input,
            duckdb_aggregate_state* states) {
  idx_t n = duckdb_data_chunk_get_size(input);
  if (n == 0)
    return;
  Col id(duckdb_data_chunk_get_vector(input, 0));
  Col par(duckdb_data_chunk_get_vector(input, 1));
  constexpr int vcol = K == ValKind::kNone ? -1 : 2;
  constexpr int pcol =
      K == ValKind::kNone
          ? 2
          : ((K == ValKind::kBoolI64 || K == ValKind::kI64I64) ? 4 : 3);
  Col val(duckdb_data_chunk_get_vector(input, vcol < 0 ? 0 : vcol));
  Col val2(duckdb_data_chunk_get_vector(
      input, (K == ValKind::kBoolI64 || K == ValKind::kI64I64) ? 3 : 0));
  // Constant parameters: INTEGER or BOOLEAN, read from the first valid row.
  int32_t pv[2] = {0, 0};
  bool pvalid[2] = {false, false};
  for (int k = 0; k < NParams; k++) {
    duckdb_vector v = duckdb_data_chunk_get_vector(input, pcol + k);
    Col c(v);
    duckdb_logical_type t = duckdb_vector_get_column_type(v);
    bool is_bool = duckdb_get_type_id(t) == DUCKDB_TYPE_BOOLEAN;
    duckdb_destroy_logical_type(&t);
    for (idx_t r = 0; r < n; r++)
      if (c.Valid(r)) {
        pv[k] = is_bool ? c.At<bool>(r) : c.At<int32_t>(r);
        pvalid[k] = true;
        break;
      }
  }
  const int64_t* ids = static_cast<const int64_t*>(id.data);
  const int64_t* ps = static_cast<const int64_t*>(par.data);
  idx_t r = 0;
  while (r < n) {
    // A run of rows sharing one state (inputs are usually grouped).
    duckdb_aggregate_state sp = states[r];
    idx_t e = r + 1;
    while (e < n && states[e] == sp)
      e++;
    State*& st = Ptr(sp);
    if (!st)
      st = new State();
    if (!st->have_params) {
      st->p1 = pv[0];
      st->p2 = pv[1];
      st->have_params = pvalid[0] || NParams == 0;
    }
    if constexpr (K == ValKind::kI64 || K == ValKind::kF64) {
      if (st->id.empty() && st->vnull.empty() &&
          RangeAllValid(id.validity, r, e) &&
          RangeAllValid(val.validity, r, e)) {
        const int64_t* vs = static_cast<const int64_t*>(val.data);
        idx_t cur = r;
        while (cur < e) {
          if (st->chunks.empty() || st->chunks.back()->len == Chunk::kCap)
            st->chunks.push_back(AllocChunk());
          Chunk* c = st->chunks.back();
          idx_t take = std::min<idx_t>(e - cur, Chunk::kCap - c->len);
          std::memcpy(c->id + c->len, ids + cur, take * 8);
          std::memcpy(c->parent + c->len, ps + cur, take * 8);
          PatchNullsI64(c->parent + c->len, par.validity, cur, cur + take,
                        kNullParent);
          std::memcpy(c->v + c->len, vs + cur, take * 8);
          c->len += static_cast<uint32_t>(take);
          st->chunk_rows += take;
          cur += take;
        }
        r = e;
        continue;
      }
    }
    st->MaterializeChunks();
    if (RangeAllValid(id.validity, r, e)) {
      idx_t base = st->id.size();
      idx_t len = e - r;
      st->id.insert(st->id.end(), ids + r, ids + e);
      st->parent.insert(st->parent.end(), ps + r, ps + e);
      PatchNullsI64(st->parent.data() + base, par.validity, r, e, kNullParent);
      if constexpr (K == ValKind::kI64 || K == ValKind::kF64) {
        const int64_t* vs = static_cast<const int64_t*>(val.data);
        st->v.insert(st->v.end(), vs + r, vs + e);
        if (!RangeAllValid(val.validity, r, e)) {
          st->EnsureVnull();
          for (idx_t k = 0; k < len; k++)
            if (!val.Valid(r + k)) {
              st->v[base + k] = 0;
              st->vnull[base + k] = 1;
            }
        } else if (!st->vnull.empty()) {
          st->vnull.insert(st->vnull.end(), len, 0);
        }
      } else if constexpr (K == ValKind::kBool) {
        const uint8_t* bs = static_cast<const uint8_t*>(val.data);
        st->flag.insert(st->flag.end(), bs + r, bs + e);
        if (!RangeAllValid(val.validity, r, e)) {
          for (idx_t k = 0; k < len; k++)
            if (!val.Valid(r + k))
              st->flag[base + k] = 0;
        }
      } else if constexpr (K == ValKind::kBoolI64) {
        const uint8_t* bs = static_cast<const uint8_t*>(val.data);
        st->flag.insert(st->flag.end(), bs + r, bs + e);
        if (!RangeAllValid(val.validity, r, e)) {
          for (idx_t k = 0; k < len; k++)
            if (!val.Valid(r + k))
              st->flag[base + k] = 0;
        }
        const int64_t* vs = static_cast<const int64_t*>(val2.data);
        st->v.insert(st->v.end(), vs + r, vs + e);
        if (!RangeAllValid(val2.validity, r, e)) {
          st->EnsureVnull();
          for (idx_t k = 0; k < len; k++)
            if (!val2.Valid(r + k)) {
              st->v[base + k] = 0;
              st->vnull[base + k] = 1;
            }
        } else if (!st->vnull.empty()) {
          st->vnull.insert(st->vnull.end(), len, 0);
        }
      } else if constexpr (K == ValKind::kI64I64) {
        const int64_t* ks = static_cast<const int64_t*>(val.data);
        st->v.insert(st->v.end(), ks + r, ks + e);
        if (!RangeAllValid(val.validity, r, e)) {
          st->EnsureVnull();
          for (idx_t k = 0; k < len; k++)
            if (!val.Valid(r + k)) {
              st->v[base + k] = 0;
              st->vnull[base + k] = 1;
            }
        } else if (!st->vnull.empty()) {
          st->vnull.insert(st->vnull.end(), len, 0);
        }
        const int64_t* vs = static_cast<const int64_t*>(val2.data);
        st->v2.insert(st->v2.end(), vs + r, vs + e);
        if (!RangeAllValid(val2.validity, r, e)) {
          for (idx_t k = 0; k < len; k++)
            if (!val2.Valid(r + k))
              st->v2[base + k] = kNullParent;
        }
      }
      r = e;
      continue;
    }
    st->EnsureVnull();
    idx_t base = st->id.size();
    idx_t len = e - r;
    st->id.resize(base + len);
    st->parent.resize(base + len);
    if constexpr (K == ValKind::kI64 || K == ValKind::kF64) {
      st->v.resize(base + len);
      st->vnull.resize(base + len);
    } else if constexpr (K == ValKind::kBool) {
      st->flag.resize(base + len);
    } else if constexpr (K == ValKind::kBoolI64) {
      st->flag.resize(base + len);
      st->v.resize(base + len);
      st->vnull.resize(base + len);
    } else if constexpr (K == ValKind::kI64I64) {
      st->v.resize(base + len);
      st->v2.resize(base + len);
      st->vnull.resize(base + len);
    }
    idx_t k = base;
    for (idx_t q = r; q < e; q++) {
      if (id.validity && !id.Valid(q)) {
        st->id_null = true;
        continue;
      }
      st->id[k] = ids[q];
      st->parent[k] = (!par.validity || par.Valid(q)) ? ps[q] : kNullParent;
      if constexpr (K == ValKind::kI64 || K == ValKind::kF64) {
        bool ok = !val.validity || val.Valid(q);
        st->v[k] = ok ? static_cast<const int64_t*>(val.data)[q] : 0;
        st->vnull[k] = !ok;
      } else if constexpr (K == ValKind::kBool) {
        st->flag[k] = (!val.validity || val.Valid(q)) &&
                      static_cast<const bool*>(val.data)[q];
      } else if constexpr (K == ValKind::kBoolI64) {
        st->flag[k] = (!val.validity || val.Valid(q)) &&
                      static_cast<const bool*>(val.data)[q];
        bool ok = !val2.validity || val2.Valid(q);
        st->v[k] = ok ? static_cast<const int64_t*>(val2.data)[q] : 0;
        st->vnull[k] = !ok;
      } else if constexpr (K == ValKind::kI64I64) {
        bool ok = !val.validity || val.Valid(q);
        st->v[k] = ok ? static_cast<const int64_t*>(val.data)[q] : 0;
        st->vnull[k] = !ok;
        bool ok2 = !val2.validity || val2.Valid(q);
        st->v2[k] =
            ok2 ? static_cast<const int64_t*>(val2.data)[q] : kNullParent;
      }
      k++;
    }
    if (k != base + len) {
      st->id.resize(k);
      st->parent.resize(k);
      if (!st->v.empty() || K == ValKind::kI64 || K == ValKind::kF64 ||
          K == ValKind::kBoolI64 || K == ValKind::kI64I64) {
        st->v.resize(k);
        st->vnull.resize(k);
      }
      if (K == ValKind::kI64I64)
        st->v2.resize(k);
      if (K == ValKind::kBool || K == ValKind::kBoolI64)
        st->flag.resize(k);
    }
    r = e;
  }
}

// ---------------------------------------------------------------------------
// The forest of one group, rows sorted by id.
// ---------------------------------------------------------------------------
struct Tree {
  idx_t n = 0;
  std::vector<int32_t> par;     // parent row or kNone
  std::vector<uint32_t> order;  // parent-first rows; empty = 0..n-1 is
  std::string err;

  uint32_t At(idx_t k) const { return order.empty() ? (uint32_t)k : order[k]; }

  bool Fail(const std::string& e) {
    err = "pf_tree: " + e;
    return false;
  }

  // Sorts the state's columns by id, then resolves parents.
  bool Build(State& s, bool strict, bool sort_kids = false) {
    s.Flatten();
    if (s.id_null)
      return Fail("a row has no id");
    n = s.id.size();
    if (n >= (1ull << 31))
      return Fail("too many nodes in one tree");
    const int64_t* id = s.id.data();
    bool sorted = true;
    for (idx_t i = 1; i < n && sorted; i++)
      sorted = id[i - 1] < id[i];
    if (!sorted) {
      std::vector<uint32_t> perm(n);
      std::iota(perm.begin(), perm.end(), 0);
      std::sort(perm.begin(), perm.end(),
                [&](uint32_t a, uint32_t b) { return id[a] < id[b]; });
      auto apply = [&](auto& vec) {
        if (vec.empty())
          return;
        std::remove_reference_t<decltype(vec)> tmp(n);
        for (idx_t i = 0; i < n; i++)
          tmp[i] = vec[perm[i]];
        vec.swap(tmp);
      };
      apply(s.id);
      apply(s.parent);
      apply(s.v);
      apply(s.v2);
      apply(s.vnull);
      apply(s.flag);
      id = s.id.data();
      for (idx_t i = 1; i < n; i++)
        if (id[i] == id[i - 1])
          return Fail("more than one row has the same node");
    }
    par.assign(n, kNone);
    bool dense = n == 0 || (id[0] == 0 && id[n - 1] == (int64_t)n - 1);
    bool parent_first = true;
    for (idx_t i = 0; i < n; i++) {
      int64_t q = s.parent[i];
      if (q == kNullParent)
        continue;
      if (q == id[i])
        return Fail("a node is its own parent");
      int64_t prow;
      if (dense) {
        prow = (q >= 0 && q < (int64_t)n) ? q : -1;
      } else {
        const int64_t* it = std::lower_bound(id, id + n, q);
        prow = (it != id + n && *it == q) ? it - id : -1;
      }
      if (prow < 0) {
        if (strict)
          return Fail(
              "a row names a parent which is not itself a row in the input");
        continue;
      }
      par[i] = (int32_t)prow;
      if (prow > (int64_t)i)
        parent_first = false;
    }
    if (parent_first && !sort_kids)
      return true;
    // BFS from roots (children in id order): parent-first, detects cycles.
    std::vector<uint32_t> start(n + 1, 0), kids(n);
    for (idx_t i = 0; i < n; i++)
      if (par[i] != kNone)
        start[par[i] + 1]++;
    for (idx_t i = 0; i < n; i++)
      start[i + 1] += start[i];
    {
      std::vector<uint32_t> fill(start.begin(), start.end() - 1);
      for (idx_t i = 0; i < n; i++)
        if (par[i] != kNone)
          kids[fill[par[i]]++] = (uint32_t)i;
    }
    order.clear();
    order.reserve(n);
    for (idx_t i = 0; i < n; i++)
      if (par[i] == kNone)
        order.push_back((uint32_t)i);
    for (idx_t k = 0; k < order.size(); k++) {
      uint32_t x = order[k];
      for (uint32_t j = start[x]; j < start[x + 1]; j++)
        order.push_back(kids[j]);
    }
    if (order.size() != n)
      return Fail("the rows are not a tree, because " +
                  std::to_string(n - order.size()) + " of them are in a cycle");
    return true;
  }

  void Children(std::vector<uint32_t>& start,
                std::vector<uint32_t>& kids) const {
    start.assign(n + 1, 0);
    kids.resize(n);
    for (idx_t i = 0; i < n; i++)
      if (par[i] != kNone)
        start[par[i] + 1]++;
    for (idx_t i = 0; i < n; i++)
      start[i + 1] += start[i];
    std::vector<uint32_t> fill(start.begin(), start.end() - 1);
    for (idx_t i = 0; i < n; i++)
      if (par[i] != kNone)
        kids[fill[par[i]]++] = (uint32_t)i;
  }
};

// ---------------------------------------------------------------------------
// Output: per group, `width` 8-byte columns (+ null masks).
// ---------------------------------------------------------------------------
struct Out {
  const int64_t* col0_ref = nullptr;
  std::vector<std::unique_ptr<int64_t[]>> cols;
  std::vector<std::vector<uint8_t>> nulls;
  idx_t len = 0;
  void Init(int width, idx_t n, const int64_t* id_ref = nullptr) {
    col0_ref = id_ref;
    cols.resize(width);
    for (int c = 0; c < width; c++) {
      if (c == 0 && id_ref) {
        cols[c].reset();
      } else {
        cols[c].reset(n ? new int64_t[n] : nullptr);
      }
    }
    nulls.assign(width, {});
    len = n;
  }
  void SetNull(int c, idx_t i) {
    if (nulls[c].empty())
      nulls[c].assign(len, 0);
    nulls[c][i] = 1;
  }
};

// width 0 = LIST<BIGINT> from cols[0], else LIST<STRUCT(width cols)>.
void WriteLists(duckdb_vector result,
                idx_t offset,
                std::vector<Out>& outs,
                int width) {
  idx_t count = outs.size();
  auto* entries =
      static_cast<duckdb_list_entry*>(duckdb_vector_get_data(result));
  idx_t base = duckdb_list_vector_get_size(result);
  idx_t total = 0;
  std::vector<idx_t> offs(count);
  for (idx_t i = 0; i < count; i++) {
    offs[i] = base + total;
    total += outs[i].len;
  }
  duckdb_list_vector_reserve(result, base + total);
  duckdb_vector child = duckdb_list_vector_get_child(result);
  int nc = width == 0 ? 1 : width;
  std::vector<duckdb_vector> vecs(nc);
  std::vector<int64_t*> data(nc);
  for (int c = 0; c < nc; c++) {
    vecs[c] = width == 0 ? child : duckdb_struct_vector_get_child(child, c);
    data[c] = static_cast<int64_t*>(duckdb_vector_get_data(vecs[c]));
  }
  ParallelFor(count, [&](idx_t i) {
    if (!outs[i].len)
      return;
    for (int c = 0; c < nc; c++) {
      const int64_t* src =
          (c == 0 && outs[i].col0_ref)
              ? outs[i].col0_ref
              : ((size_t)c < outs[i].cols.size() ? outs[i].cols[c].get()
                                                 : nullptr);
      if (src)
        memcpy(data[c] + offs[i], src, outs[i].len * 8);
    }
  });
  for (int c = 0; c < nc; c++) {
    bool any = false;
    for (auto& o : outs)
      any |= (size_t)c < o.nulls.size() && !o.nulls[c].empty();
    if (!any)
      continue;
    duckdb_vector_ensure_validity_writable(vecs[c]);
    uint64_t* val = duckdb_vector_get_validity(vecs[c]);
    for (idx_t i = 0; i < count; i++)
      if ((size_t)c < outs[i].nulls.size() && !outs[i].nulls[c].empty())
        for (idx_t k = 0; k < outs[i].len; k++)
          if (outs[i].nulls[c][k])
            duckdb_validity_set_row_invalid(val, offs[i] + k);
  }
  duckdb_list_vector_set_size(result, base + total);
  for (idx_t i = 0; i < count; i++) {
    entries[offset + i].offset = offs[i];
    entries[offset + i].length = outs[i].len;
  }
}

// Shared finalize driver: builds each group's tree (in parallel) and calls
// fn(state, tree, out) -> error string.
template <typename F>
void FinalizeWith(duckdb_function_info info,
                  duckdb_aggregate_state* source,
                  duckdb_vector result,
                  idx_t count,
                  idx_t offset,
                  int width,
                  bool strict_default,
                  int strict_param,
                  bool sort_kids,
                  F&& fn) {
  std::vector<Out> outs(count);
  std::vector<std::string> errs(count);
  ParallelFor(count, [&](idx_t i) {
    State* st = Ptr(source[i]);
    State empty;
    State& s = st ? *st : empty;
    bool strict = strict_default;
    if (strict_param == 1)
      strict = s.p1;
    if (strict_param == 2)
      strict = s.p2;
    Tree t;
    if (!t.Build(s, strict, sort_kids)) {
      errs[i] = t.err;
      return;
    }
    std::string e = fn(s, t, outs[i]);
    if (!e.empty())
      errs[i] = "pf_tree: " + e;
  });
  for (idx_t i = 0; i < count; i++)
    if (!errs[i].empty()) {
      duckdb_aggregate_function_set_error(info, errs[i].c_str());
      return;
    }
  WriteLists(result, offset, outs, width);
}

// ---------------------------------------------------------------------------
// ACCUMULATE
// ---------------------------------------------------------------------------
enum Fn {
  kSum = 0,
  kMin,
  kMax,
  kCount,
  kCountNN,
  kArgMin,
  kArgMax,
  kFirst,
  kLast
};

template <typename T>
T& AsT(int64_t& bits) {
  return *reinterpret_cast<T*>(&bits);
}

inline bool AddChecked(int64_t a, int64_t b, int64_t* o) {
  return !__builtin_add_overflow(a, b, o);
}
inline bool AddChecked(double a, double b, double* o) {
  *o = a + b;
  return true;
}

template <typename T>
std::string AccValues(State& s, const Tree& t, Out& o) {
  int dir = s.p1, fn = s.p2;
  if (fn < kSum || fn > kCountNN)
    return "bad accumulate function";
  idx_t n = t.n;
  o.Init(2, n, s.id.data());
  T* acc = reinterpret_cast<T*>(o.cols[1].get());
  std::vector<uint8_t> has;
  auto valid = [&](idx_t i) {
    if (s.ValNull(i))
      return false;
    if constexpr (std::is_floating_point<T>::value)
      return !std::isnan(AsT<T>(s.v[i]));
    return true;
  };
  if (fn == kMin || fn == kMax)
    has.assign(n, 0);
  if (fn == kSum && s.vnull.empty()) {
    if (n)
      memcpy(acc, s.v.data(), n * 8);
  } else {
    for (idx_t i = 0; i < n; i++) {
      switch (fn) {
        case kSum:
          acc[i] = s.ValNull(i) ? T(0) : AsT<T>(s.v[i]);
          break;
        case kCount:
          acc[i] = 1;
          break;
        case kCountNN:
          acc[i] = s.ValNull(i) ? 0 : 1;
          break;
        default:
          if (valid(i)) {
            acc[i] = AsT<T>(s.v[i]);
            has[i] = 1;
          } else {
            acc[i] = T(0);
          }
      }
    }
  }
  auto combine = [&](idx_t into, idx_t from) -> bool {
    if (fn == kMin || fn == kMax) {
      if (!has[from])
        return true;
      if (!has[into] ||
          (fn == kMin ? acc[from] < acc[into] : acc[from] > acc[into])) {
        acc[into] = acc[from];
        has[into] = 1;
      }
      return true;
    }
    return AddChecked(acc[into], acc[from], &acc[into]);
  };
  if (dir == 0) {
    if (t.order.empty()) {
      for (idx_t i = n; i-- > 0;)
        if (t.par[i] != kNone && !combine(t.par[i], i))
          return "integer overflow";
    } else {
      for (idx_t k = n; k-- > 0;) {
        uint32_t i = t.At(k);
        if (t.par[i] != kNone && !combine(t.par[i], i))
          return "integer overflow";
      }
    }
  } else {
    for (idx_t k = 0; k < n; k++) {
      uint32_t i = t.At(k);
      if (t.par[i] != kNone && !combine(i, t.par[i]))
        return "integer overflow";
    }
  }
  if (fn == kMin || fn == kMax)
    for (idx_t i = 0; i < n; i++)
      if (!has[i])
        o.SetNull(1, i);
  return "";
}

template <typename T>
std::string AccIds(State& s, const Tree& t, Out& o) {
  int dir = s.p1, fn = s.p2;
  if (fn < kArgMin || fn > kLast)
    return "bad accumulate function";
  idx_t n = t.n;
  o.Init(2, n, s.id.data());
  std::vector<int32_t> pick(n, kNone);
  if (fn == kFirst || fn == kLast) {
    if (dir != 1)
      return "FIRST/LAST are only defined for ACCUMULATE DOWN";
    for (idx_t k = 0; k < n; k++) {
      uint32_t i = t.At(k);
      pick[i] =
          (fn == kLast || t.par[i] == kNone) ? (int32_t)i : pick[t.par[i]];
    }
  } else {
    auto valid = [&](idx_t i) {
      if (s.ValNull(i))
        return false;
      if constexpr (std::is_floating_point<T>::value)
        return !std::isnan(AsT<T>(s.v[i]));
      return true;
    };
    for (idx_t i = 0; i < n; i++)
      pick[i] = valid(i) ? (int32_t)i : kNone;
    // Rows are sorted by id, so the smaller row wins ties.
    auto better = [&](int32_t a, int32_t b) {
      if (a == kNone)
        return false;
      if (b == kNone)
        return true;
      T x = AsT<T>(s.v[a]), y = AsT<T>(s.v[b]);
      if (x != y)
        return fn == kArgMin ? x < y : x > y;
      return a < b;
    };
    if (dir == 0) {
      for (idx_t k = n; k-- > 0;) {
        uint32_t i = t.At(k);
        int32_t p = t.par[i];
        if (p != kNone && better(pick[i], pick[p]))
          pick[p] = pick[i];
      }
    } else {
      for (idx_t k = 0; k < n; k++) {
        uint32_t i = t.At(k);
        int32_t p = t.par[i];
        if (p != kNone && better(pick[p], pick[i]))
          pick[i] = pick[p];
      }
    }
  }
  for (idx_t i = 0; i < n; i++) {
    if (pick[i] == kNone) {
      o.cols[1][i] = 0;
      o.SetNull(1, i);
    } else {
      o.cols[1][i] = s.id[pick[i]];
    }
  }
  return "";
}

template <typename T>
void AccFinalize(duckdb_function_info info,
                 duckdb_aggregate_state* src,
                 duckdb_vector result,
                 idx_t count,
                 idx_t offset) {
  // Fast zero-allocation direct-to-vector path for dense parent-first SUM
  // trees.
  bool can_fast = true;
  idx_t total_rows = 0;
  for (idx_t i = 0; i < count; i++) {
    State* st = Ptr(src[i]);
    if (!st)
      continue;
    if (st->id_null || !st->parts.empty() || !st->id.empty() ||
        !st->vnull.empty() || st->p2 != kSum || (st->p1 != 0 && st->p1 != 1)) {
      can_fast = false;
      break;
    }
    if (st->chunk_rows >= (1ull << 31)) {
      can_fast = false;
      break;
    }
    if (st->chunk_rows > 0) {
      if (st->chunks.size() > 1 &&
          !std::is_sorted(st->chunks.begin(), st->chunks.end(),
                          [](const Chunk* a, const Chunk* b) {
                            return a->id[0] < b->id[0];
                          })) {
        std::sort(
            st->chunks.begin(), st->chunks.end(),
            [](const Chunk* a, const Chunk* b) { return a->id[0] < b->id[0]; });
      }
      if (st->chunks.front()->id[0] != 0 ||
          st->chunks.back()->id[st->chunks.back()->len - 1] !=
              static_cast<int64_t>(st->chunk_rows) - 1) {
        can_fast = false;
        break;
      }
    }
    total_rows += st->chunk_rows;
  }

  if (can_fast) {
    idx_t base = duckdb_list_vector_get_size(result);
    std::vector<idx_t> offs(count);
    idx_t cur_off = base;
    for (idx_t i = 0; i < count; i++) {
      offs[i] = cur_off;
      if (State* st = Ptr(src[i]))
        cur_off += st->chunk_rows;
    }
    duckdb_list_vector_reserve(result, base + total_rows);
    auto* entries =
        static_cast<duckdb_list_entry*>(duckdb_vector_get_data(result));
    duckdb_vector child = duckdb_list_vector_get_child(result);
    int64_t* out_id = static_cast<int64_t*>(
        duckdb_vector_get_data(duckdb_struct_vector_get_child(child, 0)));
    T* out_v = reinterpret_cast<T*>(
        duckdb_vector_get_data(duckdb_struct_vector_get_child(child, 1)));

    std::atomic<bool> fast_ok{true};
    ParallelFor(count, [&](idx_t i) {
      State* st = Ptr(src[i]);
      if (!st || st->chunk_rows == 0)
        return;
      if (!fast_ok.load(std::memory_order_relaxed))
        return;
      idx_t off = offs[i];
      int64_t* id_dst = out_id + off;
      T* v_dst = out_v + off;
      idx_t pos = 0;
      for (Chunk* c : st->chunks) {
        uint32_t len = c->len;
        const int64_t* cid = c->id;
        // Verify ids are strictly dense [pos .. pos + len - 1] in registers.
        uint64_t bad = 0;
        for (uint32_t k = 0; k < len; k++)
          bad |= static_cast<uint64_t>(cid[k] ^ static_cast<int64_t>(pos + k));
        if (bad) {
          fast_ok.store(false, std::memory_order_relaxed);
          return;
        }
        std::memcpy(id_dst + pos, cid, len * 8);
        std::memcpy(v_dst + pos, c->v, len * 8);
        pos += len;
      }
      if (st->p1 == 0) {
        // UP: single backward pass across chunks directly in v_dst.
        idx_t c_base = pos;
        for (size_t ci = st->chunks.size(); ci-- > 0;) {
          Chunk* c = st->chunks[ci];
          c_base -= c->len;
          const int64_t* cpar = c->parent;
          for (uint32_t k = c->len; k-- > 0;) {
            int64_t p = cpar[k];
            if (p == kNullParent)
              continue;
            idx_t idx = c_base + k;
            if (static_cast<uint64_t>(p) >= static_cast<uint64_t>(idx) ||
                !AddChecked(v_dst[p], v_dst[idx], &v_dst[p])) {
              fast_ok.store(false, std::memory_order_relaxed);
              return;
            }
          }
        }
      } else {
        // DOWN: single forward pass across chunks directly in v_dst.
        idx_t c_base = 0;
        for (Chunk* c : st->chunks) {
          const int64_t* cpar = c->parent;
          for (uint32_t k = 0; k < c->len; k++) {
            int64_t p = cpar[k];
            if (p == kNullParent)
              continue;
            idx_t idx = c_base + k;
            if (static_cast<uint64_t>(p) >= static_cast<uint64_t>(idx) ||
                !AddChecked(v_dst[idx], v_dst[p], &v_dst[idx])) {
              fast_ok.store(false, std::memory_order_relaxed);
              return;
            }
          }
          c_base += c->len;
        }
      }
    });

    if (fast_ok.load(std::memory_order_relaxed)) {
      duckdb_list_vector_set_size(result, base + total_rows);
      for (idx_t i = 0; i < count; i++) {
        State* st = Ptr(src[i]);
        entries[offset + i].offset = offs[i];
        entries[offset + i].length = st ? st->chunk_rows : 0;
      }
      return;
    }
  }

  FinalizeWith(
      info, src, result, count, offset, 2, true, 0, false,
      [](State& s, const Tree& t, Out& o) { return AccValues<T>(s, t, o); });
}

template <typename T>
void AccIdFinalize(duckdb_function_info info,
                   duckdb_aggregate_state* src,
                   duckdb_vector result,
                   idx_t count,
                   idx_t offset) {
  FinalizeWith(
      info, src, result, count, offset, 2, true, 0, false,
      [](State& s, const Tree& t, Out& o) { return AccIds<T>(s, t, o); });
}

// ---------------------------------------------------------------------------
// Flags / nearest
// ---------------------------------------------------------------------------
void FlagsFinalize(duckdb_function_info info,
                   duckdb_aggregate_state* src,
                   duckdb_vector result,
                   idx_t count,
                   idx_t offset) {
  FinalizeWith(info, src, result, count, offset, 0, false, 2, false,
               [](State& s, const Tree& t, Out& o) {
                 int mode = s.p1;
                 idx_t n = t.n;
                 const uint8_t* m = s.flag.data();
                 std::vector<uint8_t> f(n, 0);  // has marked strict anc / desc
                 if (mode == 0 || mode == 2) {
                   for (idx_t k = 0; k < n; k++) {
                     uint32_t i = t.At(k);
                     int32_t p = t.par[i];
                     f[i] = p != kNone && (m[p] || f[p]);
                   }
                 } else if (mode == 1 || mode == 3) {
                   for (idx_t k = n; k-- > 0;) {
                     uint32_t i = t.At(k);
                     int32_t p = t.par[i];
                     if (p != kNone && (m[i] || f[i]))
                       f[p] = 1;
                   }
                 } else {
                   return std::string("bad flags mode");
                 }
                 bool self = mode >= 2;
                 idx_t cnt = 0;
                 for (idx_t i = 0; i < n; i++)
                   cnt += f[i] || (self && m[i]);
                 o.Init(1, cnt);
                 idx_t k = 0;
                 for (idx_t i = 0; i < n; i++)
                   if (f[i] || (self && m[i]))
                     o.cols[0][k++] = s.id[i];
                 return std::string();
               });
}

void NearestFinalize(duckdb_function_info info,
                     duckdb_aggregate_state* src,
                     duckdb_vector result,
                     idx_t count,
                     idx_t offset) {
  FinalizeWith(info, src, result, count, offset, 2, false, 2, false,
               [](State& s, const Tree& t, Out& o) {
                 int mode = s.p1;
                 idx_t n = t.n;
                 const uint8_t* in = s.flag.data();
                 std::vector<int32_t> res(n, kNone);
                 for (idx_t k = 0; k < n; k++) {
                   uint32_t i = t.At(k);
                   int32_t p = t.par[i];
                   if (mode == 0) {
                     res[i] = p == kNone ? kNone : (in[p] ? p : res[p]);
                   } else if (mode == 1) {
                     res[i] = (p != kNone && res[p] != kNone)
                                  ? res[p]
                                  : (in[i] ? (int32_t)i : kNone);
                   } else {
                     return std::string("bad nearest mode");
                   }
                 }
                 o.Init(2, n, s.id.data());
                 for (idx_t i = 0; i < n; i++) {
                   if (res[i] == kNone) {
                     o.cols[1][i] = 0;
                     o.SetNull(1, i);
                   } else {
                     o.cols[1][i] = s.id[res[i]];
                   }
                 }
                 return std::string();
               });
}

void ContractFinalize(duckdb_function_info info,
                      duckdb_aggregate_state* src,
                      duckdb_vector result,
                      idx_t count,
                      idx_t offset) {
  FinalizeWith(info, src, result, count, offset, 3, false, 1, false,
               [](State& s, const Tree& t, Out& o) {
                 idx_t n = t.n;
                 const uint8_t* in = s.flag.data();
                 std::vector<int32_t> res(n, kNone);
                 for (idx_t k = 0; k < n; k++) {
                   uint32_t i = t.At(k);
                   int32_t p = t.par[i];
                   res[i] = p == kNone ? kNone : (in[p] ? p : res[p]);
                 }
                 idx_t cnt = 0;
                 for (idx_t i = 0; i < n; i++)
                   cnt += (in[i] || res[i] != kNone);
                 o.Init(3, cnt);
                 idx_t w = 0;
                 for (idx_t i = 0; i < n; i++) {
                   int32_t own = in[i] ? (int32_t)i : res[i];
                   if (own == kNone)
                     continue;
                   o.cols[0][w] = s.id[i];
                   o.cols[1][w] = s.id[own];
                   if (!in[i] || res[i] == kNone) {
                     o.cols[2][w] = 0;
                     o.SetNull(2, w);
                   } else {
                     o.cols[2][w] = s.id[res[i]];
                   }
                   w++;
                 }
                 return std::string();
               });
}

void ContractSumFinalize(duckdb_function_info info,
                         duckdb_aggregate_state* src,
                         duckdb_vector result,
                         idx_t count,
                         idx_t offset) {
  FinalizeWith(info, src, result, count, offset, 4, false, 1, false,
               [](State& s, const Tree& t, Out& o) {
                 idx_t n = t.n;
                 const uint8_t* in = s.flag.data();
                 std::vector<int32_t> res(n, kNone);
                 std::vector<int64_t> sum_v(n, 0);
                 std::vector<int64_t> cnt_n(n, 0);
                 std::vector<uint8_t> has_v(n, 0);
                 for (idx_t k = 0; k < n; k++) {
                   uint32_t i = t.At(k);
                   int32_t p = t.par[i];
                   res[i] = p == kNone ? kNone : (in[p] ? p : res[p]);
                   int32_t own = in[i] ? (int32_t)i : res[i];
                   if (own == kNone)
                     continue;
                   cnt_n[own]++;
                   if (!s.ValNull(i)) {
                     sum_v[own] += s.v[i];
                     has_v[own] = 1;
                   }
                 }
                 idx_t cnt = 0;
                 for (idx_t i = 0; i < n; i++)
                   cnt += (in[i] != 0);
                 o.Init(4, cnt);
                 idx_t w = 0;
                 for (idx_t i = 0; i < n; i++) {
                   if (!in[i])
                     continue;
                   o.cols[0][w] = s.id[i];
                   if (res[i] == kNone) {
                     o.cols[1][w] = 0;
                     o.SetNull(1, w);
                   } else {
                     o.cols[1][w] = s.id[res[i]];
                   }
                   if (!has_v[i]) {
                     o.cols[2][w] = 0;
                     o.SetNull(2, w);
                   } else {
                     o.cols[2][w] = sum_v[i];
                   }
                   o.cols[3][w] = cnt_n[i];
                   w++;
                 }
                 return std::string();
               });
}

void AbsorbSumFinalize(duckdb_function_info info,
                       duckdb_aggregate_state* src,
                       duckdb_vector result,
                       idx_t count,
                       idx_t offset) {
  FinalizeWith(info, src, result, count, offset, 4, false, 1, false,
               [](State& s, const Tree& t, Out& o) {
                 idx_t n = t.n;
                 const uint8_t* in = s.flag.data();
                 std::vector<int32_t> res(n, kNone);
                 std::vector<int64_t> sum_v(n, 0);
                 std::vector<int64_t> cnt_n(n, 0);
                 std::vector<uint8_t> has_v(n, 0);
                 for (idx_t k = 0; k < n; k++) {
                   uint32_t i = t.At(k);
                   int32_t p = t.par[i];
                   res[i] = (p != kNone && res[p] != kNone)
                                ? res[p]
                                : (in[i] ? (int32_t)i : kNone);
                   int32_t own = res[i] != kNone ? res[i] : (int32_t)i;
                   cnt_n[own]++;
                   if (!s.ValNull(i)) {
                     sum_v[own] += s.v[i];
                     has_v[own] = 1;
                   }
                 }
                 idx_t cnt = 0;
                 for (idx_t i = 0; i < n; i++)
                   cnt += (cnt_n[i] > 0);
                 o.Init(4, cnt);
                 idx_t w = 0;
                 for (idx_t i = 0; i < n; i++) {
                   if (cnt_n[i] == 0)
                     continue;
                   o.cols[0][w] = s.id[i];
                   int32_t p = t.par[i];
                   if (p == kNone) {
                     o.cols[1][w] = 0;
                     o.SetNull(1, w);
                   } else {
                     o.cols[1][w] = s.id[p];
                   }
                   if (!has_v[i]) {
                     o.cols[2][w] = 0;
                     o.SetNull(2, w);
                   } else {
                     o.cols[2][w] = sum_v[i];
                   }
                   o.cols[3][w] = cnt_n[i];
                   w++;
                 }
                 return std::string();
               });
}

// ---------------------------------------------------------------------------
// MERGE SIBLINGS / MERGE INTO PARENT
// ---------------------------------------------------------------------------
struct PairHash {
  size_t operator()(const std::pair<int64_t, int64_t>& p) const {
    uint64_t h = (uint64_t)p.first * 0x9E3779B97F4A7C15ull ^ (uint64_t)p.second;
    h ^= h >> 31;
    h *= 0xBF58476D1CE4E5B9ull;
    return h ^ (h >> 29);
  }
};

constexpr int64_t kNullKey = 0x7fffffffffffffd1;

struct SibSlot {
  int64_t pm;
  int64_t key;
  int64_t mid;
};

template <typename KeyFn>
int64_t ComputeMergedSiblings(const Tree& t,
                              bool ordered,
                              KeyFn&& key_fn,
                              std::vector<int64_t>& mid) {
  idx_t n = t.n;
  mid.resize(n);
  int64_t next = 0;
  if (!ordered) {
    size_t cap = 16;
    while (cap < n * 2)
      cap <<= 1;
    size_t mask = cap - 1;
    std::unique_ptr<SibSlot[]> ht(new SibSlot[cap]);
    for (size_t s0 = 0; s0 < cap; s0++)
      ht[s0].pm = -2;
    for (idx_t k = 0; k < n; k++) {
      uint32_t i = t.At(k);
      int64_t pm = t.par[i] == kNone ? -1 : mid[t.par[i]];
      int64_t ky = key_fn(i);
      uint64_t h = (static_cast<uint64_t>(pm) * 0x9e3779b97f4a7c15ULL) ^
                   (static_cast<uint64_t>(ky) * 0xbf58476d1ce4e5b9ULL);
      h ^= h >> 30;
      size_t idx = static_cast<size_t>(h) & mask;
      while (true) {
        SibSlot& sl = ht[idx];
        if (sl.pm == -2) {
          sl.pm = pm;
          sl.key = ky;
          sl.mid = next;
          mid[i] = next++;
          break;
        }
        if (sl.pm == pm && sl.key == ky) {
          mid[i] = sl.mid;
          break;
        }
        idx = (idx + 1) & mask;
      }
    }
  } else {
    // Last child (key, mid) per merged parent; slot 0 = roots.
    std::vector<int64_t> last_key(1, 0), last_mid(1, -1);
    for (idx_t k = 0; k < n; k++) {
      uint32_t i = t.At(k);
      size_t slot = t.par[i] == kNone ? 0 : (size_t)mid[t.par[i]] + 1;
      int64_t ky = key_fn(i);
      if (last_mid[slot] >= 0 && last_key[slot] == ky) {
        mid[i] = last_mid[slot];
      } else {
        mid[i] = next++;
        last_key.push_back(0);
        last_mid.push_back(-1);
        last_key[slot] = ky;
        last_mid[slot] = mid[i];
      }
    }
  }
  return next;
}

void MergeSiblingsFinalize(duckdb_function_info info,
                           duckdb_aggregate_state* src,
                           duckdb_vector result,
                           idx_t count,
                           idx_t offset) {
  FinalizeWith(info, src, result, count, offset, 3, true, 0, true,
               [](State& s, const Tree& t, Out& o) {
                 idx_t n = t.n;
                 auto K = [&](idx_t i) {
                   return s.ValNull(i) ? kNullKey : s.v[i];
                 };
                 std::vector<int64_t> mid;
                 ComputeMergedSiblings(t, s.p1 != 0, K, mid);
                 o.Init(3, n, s.id.data());
                 for (idx_t i = 0; i < n; i++) {
                   o.cols[1][i] = mid[i];
                   if (t.par[i] == kNone) {
                     o.cols[2][i] = 0;
                     o.SetNull(2, i);
                   } else {
                     o.cols[2][i] = mid[t.par[i]];
                   }
                 }
                 return std::string();
               });
}

template <typename KeyFn>
void ComputeMergeIntoParentOwners(const Tree& t,
                                  bool indirect,
                                  KeyFn&& key_fn,
                                  std::vector<int32_t>& own) {
  idx_t n = t.n;
  own.resize(n);
  for (idx_t i = 0; i < n; i++)
    own[i] = (int32_t)i;
  if (!indirect) {
    for (idx_t k = 0; k < n; k++) {
      uint32_t i = t.At(k);
      int32_t p = t.par[i];
      if (p != kNone && key_fn(p) == key_fn(i))
        own[i] = own[p];
    }
  } else {
    // DFS keeping the root->node path; a node whose key already occurs on the
    // path folds itself and every node between it and the outermost occurrence
    // into that occurrence.
    std::vector<uint32_t> start, kids;
    t.Children(start, kids);
    std::unordered_map<int64_t, uint32_t> first;  // key -> depth
    first.reserve(64);
    std::vector<uint32_t> path;
    std::vector<uint8_t> inserted(n, 0);
    std::vector<std::pair<uint32_t, uint32_t>> st;
    for (idx_t root = 0; root < n; root++) {
      if (t.par[root] != kNone)
        continue;
      st.push_back({(uint32_t)root, 0});
      while (!st.empty()) {
        uint32_t x = st.back().first;
        if (st.back().second == 0) {
          uint32_t d = path.size();
          path.push_back(x);
          int64_t kx = key_fn(x);
          auto it = first.find(kx);
          if (it == first.end()) {
            first.emplace(kx, d);
            inserted[x] = 1;
          } else {
            int32_t target = own[path[it->second]];
            for (uint32_t j = d; j > it->second; j--) {
              if (own[path[j]] == target)
                break;
              own[path[j]] = target;
            }
          }
        }
        if (st.back().second < start[x + 1] - start[x]) {
          uint32_t c = kids[start[x] + st.back().second++];
          st.push_back({c, 0});
        } else {
          if (inserted[x])
            first.erase(key_fn(x));
          path.pop_back();
          st.pop_back();
        }
      }
    }
    for (idx_t k = 0; k < n; k++) {
      uint32_t i = t.At(k);
      if (own[i] != (int32_t)i)
        own[i] = own[own[i]];
    }
  }
}

void MergeIntoParentFinalize(duckdb_function_info info,
                             duckdb_aggregate_state* src,
                             duckdb_vector result,
                             idx_t count,
                             idx_t offset) {
  FinalizeWith(info, src, result, count, offset, 3, true, 0, false,
               [](State& s, const Tree& t, Out& o) {
                 idx_t n = t.n;
                 auto K = [&](idx_t i) {
                   return s.ValNull(i) ? kNullKey : s.v[i];
                 };
                 std::vector<int32_t> own;
                 ComputeMergeIntoParentOwners(t, s.p1 != 0, K, own);
                 o.Init(3, n, s.id.data());
                 for (idx_t i = 0; i < n; i++) {
                   o.cols[1][i] = s.id[own[i]];
                   int32_t p = t.par[i];
                   if (own[i] != (int32_t)i || p == kNone) {
                     o.cols[2][i] = 0;
                     o.SetNull(2, i);
                   } else {
                     o.cols[2][i] = s.id[own[p]];
                   }
                 }
                 return std::string();
               });
}

void MergeSiblingsSumFinalize(duckdb_function_info info,
                              duckdb_aggregate_state* src,
                              duckdb_vector result,
                              idx_t count,
                              idx_t offset) {
  FinalizeWith(
      info, src, result, count, offset, 5, true, 0, true,
      [](State& s, const Tree& t, Out& o) {
        idx_t n = t.n;
        auto K = [&](idx_t i) { return s.ValNull(i) ? kNullKey : s.v[i]; };
        std::vector<int64_t> mid;
        int64_t mcnt = ComputeMergedSiblings(t, s.p1 != 0, K, mid);
        o.Init(5, static_cast<idx_t>(mcnt), nullptr);
        std::fill_n(o.cols[3].get(), static_cast<size_t>(mcnt), int64_t{0});
        std::fill_n(o.cols[4].get(), static_cast<size_t>(mcnt), int64_t{0});
        std::vector<uint8_t> seen(static_cast<size_t>(mcnt), 0);
        std::vector<int64_t> hd(static_cast<size_t>(mcnt), 0);
        for (idx_t i = 0; i < n; i++) {
          idx_t m = static_cast<idx_t>(mid[i]);
          if (!seen[m]) {
            seen[m] = 1;
            o.cols[0][m] = s.id[i];  // representative row id for key columns
            o.cols[1][m] = mid[i];   // merged node id (0 .. mcnt-1)
            if (t.par[i] == kNone) {
              o.cols[2][m] = 0;
              o.SetNull(2, m);
            } else {
              o.cols[2][m] = mid[t.par[i]];
            }
          }
          if (s.v2[i] != kNullParent) {
            if (__builtin_add_overflow(o.cols[3][m], s.v2[i], &o.cols[3][m]))
              return std::string("pf_tree: integer sum overflow");
            hd[m]++;
          }
          o.cols[4][m]++;
        }
        for (idx_t m = 0; m < static_cast<idx_t>(mcnt); m++) {
          if (hd[m] == 0)
            o.SetNull(3, m);
        }
        return std::string();
      });
}

void MergeIntoParentSumFinalize(duckdb_function_info info,
                                duckdb_aggregate_state* src,
                                duckdb_vector result,
                                idx_t count,
                                idx_t offset) {
  FinalizeWith(
      info, src, result, count, offset, 4, true, 0, false,
      [](State& s, const Tree& t, Out& o) {
        idx_t n = t.n;
        auto K = [&](idx_t i) { return s.ValNull(i) ? kNullKey : s.v[i]; };
        std::vector<int32_t> own;
        ComputeMergeIntoParentOwners(t, s.p1 != 0, K, own);
        std::vector<int64_t> sum(n, 0), cnt(n, 0), hd(n, 0);
        idx_t keep_n = 0;
        for (idx_t i = 0; i < n; i++) {
          int32_t target = own[i];
          if (target == static_cast<int32_t>(i))
            keep_n++;
          if (s.v2[i] != kNullParent) {
            if (__builtin_add_overflow(sum[target], s.v2[i], &sum[target]))
              return std::string("pf_tree: integer sum overflow");
            hd[target]++;
          }
          cnt[target]++;
        }
        o.Init(4, keep_n, nullptr);
        idx_t out_i = 0;
        for (idx_t i = 0; i < n; i++) {
          if (own[i] != static_cast<int32_t>(i))
            continue;
          o.cols[0][out_i] = s.id[i];
          int32_t p = t.par[i];
          if (p == kNone) {
            o.cols[1][out_i] = 0;
            o.SetNull(1, out_i);
          } else {
            o.cols[1][out_i] = s.id[own[p]];
          }
          o.cols[2][out_i] = sum[i];
          if (hd[i] == 0)
            o.SetNull(2, out_i);
          o.cols[3][out_i] = cnt[i];
          out_i++;
        }
        return std::string();
      });
}

// ---------------------------------------------------------------------------
// INVERT / ANCESTORS
// ---------------------------------------------------------------------------
void InvertFinalize(duckdb_function_info info,
                    duckdb_aggregate_state* src,
                    duckdb_vector result,
                    idx_t count,
                    idx_t offset) {
  FinalizeWith(info, src, result, count, offset, 4, true, 0, false,
               [](State& s, const Tree& t, Out& o) {
                 idx_t n = t.n;
                 std::vector<uint8_t> has_kid(n, 0);
                 std::vector<uint32_t> depth(n, 0);
                 for (idx_t k = 0; k < n; k++) {
                   uint32_t i = t.At(k);
                   if (t.par[i] != kNone) {
                     has_kid[t.par[i]] = 1;
                     depth[i] = depth[t.par[i]] + 1;
                   }
                 }
                 idx_t total = 0;
                 for (idx_t i = 0; i < n; i++)
                   if (!has_kid[i])
                     total += depth[i] + 1;
                 o.Init(4, total);
                 idx_t k = 0;
                 for (idx_t l = 0; l < n; l++) {
                   if (has_kid[l])
                     continue;
                   bool first = true;
                   for (int32_t x = (int32_t)l; x != kNone; x = t.par[x]) {
                     o.cols[0][k] = (int64_t)k;
                     if (first) {
                       o.SetNull(1, k);
                       first = false;
                     } else {
                       o.cols[1][k] = (int64_t)k - 1;
                     }
                     o.cols[2][k] = s.id[x];
                     o.cols[3][k] = s.id[l];
                     k++;
                   }
                 }
                 return std::string();
               });
}

void FlattenRecursionFinalize(duckdb_function_info info,
                              duckdb_aggregate_state* src,
                              duckdb_vector result,
                              idx_t count,
                              idx_t offset) {
  FinalizeWith(info, src, result, count, offset, 2, true, 0, false,
               [](State& s, const Tree& t, Out& o) {
                 idx_t n = t.n;
                 auto K = [&](idx_t i) {
                   return s.ValNull(i) ? kNullKey : s.v[i];
                 };
                 std::vector<int32_t> new_par(n, kNone);
                 for (idx_t k = 0; k < n; k++) {
                   uint32_t i = t.At(k);
                   int32_t p = t.par[i];
                   while (p != kNone && K(p) == K(i)) {
                     p = new_par[p];
                   }
                   new_par[i] = p;
                 }
                 o.Init(2, n, s.id.data());
                 for (idx_t i = 0; i < n; i++) {
                   if (new_par[i] == kNone) {
                     o.cols[1][i] = 0;
                     o.SetNull(1, i);
                   } else {
                     o.cols[1][i] = s.id[new_par[i]];
                   }
                 }
                 return std::string();
               });
}

void AncestorsFinalize(duckdb_function_info info,
                       duckdb_aggregate_state* src,
                       duckdb_vector result,
                       idx_t count,
                       idx_t offset) {
  FinalizeWith(info, src, result, count, offset, 3, false, 1, false,
               [](State& s, const Tree& t, Out& o) {
                 idx_t n = t.n;
                 std::vector<uint32_t> depth(n, 0);
                 idx_t total = 0;
                 for (idx_t k = 0; k < n; k++) {
                   uint32_t i = t.At(k);
                   if (t.par[i] != kNone)
                     depth[i] = depth[t.par[i]] + 1;
                   total += depth[i];
                 }
                 o.Init(3, total);
                 idx_t k = 0;
                 for (idx_t i = 0; i < n; i++) {
                   int64_t d = 1;
                   for (int32_t p = t.par[i]; p != kNone; p = t.par[p], d++) {
                     o.cols[0][k] = s.id[i];
                     o.cols[1][k] = s.id[p];
                     o.cols[2][k] = d;
                     k++;
                   }
                 }
                 return std::string();
               });
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------
duckdb_logical_type LT(duckdb_type t) {
  return duckdb_create_logical_type(t);
}

duckdb_logical_type ListOf(duckdb_logical_type c) {
  duckdb_logical_type l = duckdb_create_list_type(c);
  duckdb_destroy_logical_type(&c);
  return l;
}

// LIST<STRUCT(names...)>; all BIGINT unless the field name is "v" and vtype
// is given.
duckdb_logical_type ListOfStruct(const std::vector<const char*>& names,
                                 duckdb_type vtype = DUCKDB_TYPE_BIGINT) {
  std::vector<duckdb_logical_type> ts(names.size());
  for (size_t i = 0; i < names.size(); i++)
    ts[i] = LT(strcmp(names[i], "v") == 0 ? vtype : DUCKDB_TYPE_BIGINT);
  std::vector<const char*> nm(names);
  duckdb_logical_type s =
      duckdb_create_struct_type(ts.data(), nm.data(), ts.size());
  for (auto& t : ts)
    duckdb_destroy_logical_type(&t);
  return ListOf(s);
}

duckdb_aggregate_function MakeAgg(const char* name,
                                  std::vector<duckdb_type> params,
                                  duckdb_logical_type ret,
                                  duckdb_aggregate_update_t update,
                                  duckdb_aggregate_finalize_t finalize) {
  duckdb_aggregate_function f = duckdb_create_aggregate_function();
  duckdb_aggregate_function_set_name(f, name);
  for (duckdb_type p : params) {
    duckdb_logical_type lt = LT(p);
    duckdb_aggregate_function_add_parameter(f, lt);
    duckdb_destroy_logical_type(&lt);
  }
  duckdb_aggregate_function_set_return_type(f, ret);
  duckdb_destroy_logical_type(&ret);
  duckdb_aggregate_function_set_functions(f, StateSize, StateInit, update,
                                          StateCombine, finalize);
  duckdb_aggregate_function_set_destructor(f, StateDestroy);
  duckdb_aggregate_function_set_special_handling(f);
  return f;
}

bool RegisterAggSet(duckdb_connection con,
                    const char* name,
                    std::vector<duckdb_aggregate_function> fs) {
  duckdb_aggregate_function_set set =
      duckdb_create_aggregate_function_set(name);
  bool ok = true;
  for (auto& f : fs) {
    ok &= duckdb_add_aggregate_function_to_set(set, f) == DuckDBSuccess;
    duckdb_destroy_aggregate_function(&f);
  }
  ok &= duckdb_register_aggregate_function_set(con, set) == DuckDBSuccess;
  duckdb_destroy_aggregate_function_set(&set);
  return ok;
}

}  // namespace

bool RegisterTreeFunctions(duckdb_connection con) {
  constexpr auto I64 = DUCKDB_TYPE_BIGINT, F64 = DUCKDB_TYPE_DOUBLE,
                 I32 = DUCKDB_TYPE_INTEGER, BOOL = DUCKDB_TYPE_BOOLEAN;
  bool ok = true;
  const char* acc = "pf_tree_accumulate_agg";
  ok &= RegisterAggSet(
      con, acc,
      {MakeAgg(acc, {I64, I64, I64, I32, I32}, ListOfStruct({"id", "v"}, I64),
               Update<ValKind::kI64, 2>, AccFinalize<int64_t>),
       MakeAgg(acc, {I64, I64, F64, I32, I32}, ListOfStruct({"id", "v"}, F64),
               Update<ValKind::kF64, 2>, AccFinalize<double>)});
  const char* acci = "pf_tree_accumulate_id_agg";
  ok &= RegisterAggSet(
      con, acci,
      {MakeAgg(acci, {I64, I64, I64, I32, I32}, ListOfStruct({"id", "v"}),
               Update<ValKind::kI64, 2>, AccIdFinalize<int64_t>),
       MakeAgg(acci, {I64, I64, F64, I32, I32}, ListOfStruct({"id", "v"}),
               Update<ValKind::kF64, 2>, AccIdFinalize<double>)});
  ok &= RegisterAggSet(
      con, "pf_tree_flags_agg",
      {MakeAgg("pf_tree_flags_agg", {I64, I64, BOOL, I32, BOOL},
               ListOf(LT(I64)), Update<ValKind::kBool, 2>, FlagsFinalize)});
  ok &= RegisterAggSet(
      con, "pf_tree_nearest_agg",
      {MakeAgg("pf_tree_nearest_agg", {I64, I64, BOOL, I32, BOOL},
               ListOfStruct({"id", "v"}), Update<ValKind::kBool, 2>,
               NearestFinalize)});
  ok &= RegisterAggSet(con, "pf_tree_contract_agg",
                       {MakeAgg("pf_tree_contract_agg", {I64, I64, BOOL, BOOL},
                                ListOfStruct({"id", "owner", "parent_id"}),
                                Update<ValKind::kBool, 1>, ContractFinalize)});
  ok &= RegisterAggSet(
      con, "pf_tree_contract_sum_agg",
      {MakeAgg("pf_tree_contract_sum_agg", {I64, I64, BOOL, I64, BOOL},
               ListOfStruct({"id", "parent_id", "v", "n"}),
               Update<ValKind::kBoolI64, 1>, ContractSumFinalize)});
  ok &= RegisterAggSet(
      con, "pf_tree_absorb_sum_agg",
      {MakeAgg("pf_tree_absorb_sum_agg", {I64, I64, BOOL, I64, BOOL},
               ListOfStruct({"id", "parent_id", "v", "n"}),
               Update<ValKind::kBoolI64, 1>, AbsorbSumFinalize)});
  ok &= RegisterAggSet(
      con, "pf_tree_merge_siblings_agg",
      {MakeAgg("pf_tree_merge_siblings_agg", {I64, I64, I64, BOOL},
               ListOfStruct({"id", "mid", "pmid"}), Update<ValKind::kI64, 1>,
               MergeSiblingsFinalize)});
  ok &= RegisterAggSet(
      con, "pf_tree_merge_siblings_sum_agg",
      {MakeAgg("pf_tree_merge_siblings_sum_agg", {I64, I64, I64, I64, BOOL},
               ListOfStruct({"rep_id", "id", "parent_id", "v", "n"}),
               Update<ValKind::kI64I64, 1>, MergeSiblingsSumFinalize)});
  ok &= RegisterAggSet(
      con, "pf_tree_merge_into_parent_agg",
      {MakeAgg("pf_tree_merge_into_parent_agg", {I64, I64, I64, BOOL},
               ListOfStruct({"id", "owner", "parent_id"}),
               Update<ValKind::kI64, 1>, MergeIntoParentFinalize)});
  ok &= RegisterAggSet(
      con, "pf_tree_merge_into_parent_sum_agg",
      {MakeAgg("pf_tree_merge_into_parent_sum_agg", {I64, I64, I64, I64, BOOL},
               ListOfStruct({"id", "parent_id", "v", "n"}),
               Update<ValKind::kI64I64, 1>, MergeIntoParentSumFinalize)});
  ok &= RegisterAggSet(
      con, "pf_tree_flatten_recursion_agg",
      {MakeAgg("pf_tree_flatten_recursion_agg", {I64, I64, I64},
               ListOfStruct({"id", "parent_id"}), Update<ValKind::kI64, 0>,
               FlattenRecursionFinalize)});
  ok &= RegisterAggSet(
      con, "pf_tree_invert_agg",
      {MakeAgg("pf_tree_invert_agg", {I64, I64},
               ListOfStruct({"id", "parent_id", "node_id", "leaf_id"}),
               Update<ValKind::kNone, 0>, InvertFinalize)});
  ok &= RegisterAggSet(con, "pf_tree_ancestors_agg",
                       {MakeAgg("pf_tree_ancestors_agg", {I64, I64, BOOL},
                                ListOfStruct({"id", "aid", "dist"}),
                                Update<ValKind::kNone, 1>, AncestorsFinalize)});
  return ok;
}
