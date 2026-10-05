// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE READS AND WRITES, ON TILES: a tile's load and store (one round trip a tile), every rank's
// tile, a tile whose rows are tensors, and a row's scalar. How a group is loaded (one 16-byte
// global instruction, issued together) is impl's.

#pragma once

#ifndef HIP_COMMS_COMMON_INTERFACE
#error "include common/common.cuh, common's one interface, not its parts"
#endif

#include "tile.cuh"
#include "utils.cuh"

namespace hip_comms {

// PEER MEMORY THROUGH GLOBAL INSTRUCTIONS. A pointer read out of a struct has no address
// space the compiler can prove, so it emits `flat_load`, which checks the aperture and
// waits on both counters; casting to address space 1 gives `global_load_dwordx4` /
// `global_store_dwordx4`.
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(1))) u32x4 global_u32x4;

// ISSUED HERE, NOT WHERE THE COMPILER LIKES: no memory operation moves across this point (the
// empty asm's memory clobber) and no instruction is scheduled across it (sched_barrier), so every
// load above it is in flight before anything below it runs. The scheduler alone sank a reduce's
// peer loads past the adds of the first ones (ISA 2026-10-01T00-39-51Z); with sched_barrier alone,
// a tile's weight loads were sunk to their uses, one register quad and one round trip each (ISA
// 2026-10-03T21-00-27Z).
namespace impl {
template <typename PACK>
DINLINE PACK global_load(const PACK* p) {
  static_assert(sizeof(PACK) == 16, "a pack is 16 bytes");
  const u32x4 raw = *(const global_u32x4*)(p);
  PACK v;
  __builtin_memcpy(&v, &raw, 16);
  return v;
}

template <typename PACK>
DINLINE void global_store(PACK* p, const PACK& v) {
  static_assert(sizeof(PACK) == 16, "a pack is 16 bytes");
  u32x4 raw;
  __builtin_memcpy(&raw, &v, 16);
  *(global_u32x4*)(p) = raw;
}

DINLINE void issued() {
  asm volatile("" ::: "memory");
  __builtin_amdgcn_sched_barrier(0);
}
}  // namespace impl

namespace impl {
// A ROW'S PACK k AS A UNIFORM BASE PLUS AN UNSIGNED 32-BIT BYTE OFFSET: the form global_load and
// global_store take with the base in scalar registers (saddr), so every tensor sharing the
// columns shares one offset register. A signed column added to the pointer was a 64-bit vector
// address a tensor a pack, 28 VGPRs held across AttnRes's loops (liveness, 2026-10-04T15-28-51Z).
// THE ADDRESS IS BUILT AT ITS LOAD: the offset made opaque here, so the 64-bit sum is not hoisted
// and held a tensor a pack (an add a load; AttnRes 90 -> 70 VGPRs, 2026-10-04 compile).
template <typename PACK, typename TILE>
DINLINE PACK* at_col(PACK* row, const TILE& t, int k) {
  using Byte      = std::conditional_t<std::is_const_v<PACK>, const char, char>;
  uint32_t offset = static_cast<uint32_t>(t.col(k)) * uint32_t{sizeof(PACK)};
  asm volatile("" : "+v"(offset));
  return reinterpret_cast<PACK*>(reinterpret_cast<Byte*>(row) + offset);
}
template <typename PACK>
DINLINE PACK pack_load(const PACK* p) {
  if constexpr (sizeof(PACK) == 16)
    return global_load(p);
  else
    return *p;
}
// WHETHER A LOAD SKIPS ROW m: only a tile of several rows a thread or of rows (a reduce-scatter's
// at decode), whose rows past M would re-read the last over the links. A one-row tile's row is
// real wherever a kernel makes one, and a branch around its loads cost registers: the zeros the
// tile starts with live beside the loads (126 -> 178 VGPRs, spilling at 16384: 23-46-07Z).
template <typename TILE>
DINLINE bool skipped(const TILE& t, int m) {
  if constexpr (TILE::kRows == 1 && TILE::kThreadsM == 1) return false;
  return !t.live(m);
}
// A loaded group into the tile's elements, and a held group out as one load or store's worth.
template <typename TILE>
DINLINE void held(TILE& t, int m, int k, const typename TILE::Pack& p) {
  __builtin_memcpy(t.v[m][k], &p, sizeof(p));
}
template <typename TILE>
DINLINE typename TILE::Pack pack(const TILE& t, int m, int k) {
  typename TILE::Pack p;
  __builtin_memcpy(&p, t.v[m][k], sizeof(p));
  return p;
}
}  // namespace impl

// A LOAD NEVER BRANCHES: a thread outside a narrow tile loads its clamped, in-bounds place and
// its values go unused. Under participates() the load was an exec branch, so every tile's old
// registers stayed live across AttnRes's loops (threadIdx < 256 is not folded from the launch
// bounds; ISA 2026-10-04T16-39-15Z).
template <typename TILE>
DINLINE void tile_load(TILE& t, const typename TILE::Acc* data,
                         int64_t row_stride) {
  using P = typename TILE::Pack;
  const P* at = reinterpret_cast<const P*>(data);
#pragma unroll
  for (int m = 0; m < TILE::kRows; ++m) {
    if (impl::skipped(t, m)) continue;
    const P* row = at + int64_t{t.row(m)} * (row_stride / t.kPack);
#pragma unroll
    for (int k = 0; k < t.K; ++k) impl::held(t, m, k, impl::pack_load(impl::at_col(row, t, k)));
  }
  impl::issued();
}

template <typename TILE>
DINLINE void tile_store(typename TILE::Acc* data, int64_t row_stride, const TILE& t) {
  using P = typename TILE::Pack;
  if (!t.participates()) return;
  P* at   = reinterpret_cast<P*>(data);
#pragma unroll
  for (int m = 0; m < TILE::kRows; ++m) {
    if (!t.live(m)) continue;
    P* row = at + int64_t{t.row(m)} * (row_stride / t.kPack);
#pragma unroll
    for (int k = 0; k < t.K; ++k) {
      if (t.mask(k) == 0.0f) continue;
      if constexpr (sizeof(P) == 16)
        impl::global_store(impl::at_col(row, t, k), impl::pack(t, m, k));
      else
        *impl::at_col(row, t, k) = impl::pack(t, m, k);
    }
  }
}

// EVERY PEER'S TILE, all in flight together: `data(r)` is rank r's tensor, `row_stride` its
// elements between rows. A pack position at a time, every rank's for it, so the address is
// computed once a position (the other way round cost 16 scalar instructions at two packs: ISA
// 2026-10-01T00-31-14Z). Nothing waits until a tile is used (peers_reduce), so loads issued here
// can run under other work.
template <typename TILE, int WORLD, typename RANK_DATA>
DINLINE void peers_load(TILE (&t)[WORLD], RANK_DATA data,
                        int64_t row_stride) {
  using P = typename TILE::Pack;
#pragma unroll
  for (int m = 0; m < TILE::kRows; ++m) {
    if (impl::skipped(t[0], m)) continue;
#pragma unroll
    for (int k = 0; k < t[0].K; ++k) {
      const int64_t i = int64_t{t[0].row(m)} * (row_stride / t[0].kPack) + t[0].col(k);
#pragma unroll
      for (int r = 0; r < WORLD; ++r)
        impl::held(t[r], m, k, impl::pack_load(reinterpret_cast<const P*>(data(r)) + i));
    }
  }
  impl::issued();
}

// A TILE WHOSE COLUMNS ARE SPLIT AMONG THE RANKS, `slice` columns each (the last rank's to the
// end): each pack from its owner's tensor, `data(r)`. A SLICE IS WHOLE WAVES, so a wave's packs
// have one owner.
template <int WORLD, typename TILE, typename RANK_DATA>
DINLINE void sliced_load(TILE& t, RANK_DATA data, int64_t row_stride,
                         int slice) {
  using P = typename TILE::Pack;
#pragma unroll
  for (int m = 0; m < TILE::kRows; ++m) {
    if (impl::skipped(t, m)) continue;
#pragma unroll
    for (int k = 0; k < t.K; ++k) {
      const int owner = min(t.col(k) / (slice / t.kPack), WORLD - 1);
      impl::held(t, m, k,
                 impl::pack_load(reinterpret_cast<const P*>(data(owner)) +
                                 int64_t{t.row(m)} * (row_stride / t.kPack) + t.col(k)));
    }
  }
  impl::issued();
}

// A TILE WHOSE ROWS ARE DIFFERENT TENSORS' (a row a peer): row m's at `row_data(m)`, its first
// element, the tile's columns counted from it. tile_gather loads every row together, one round trip;
// tile_scatter stores row m into `row_data(m)` up to its own `row_n(m)` columns (a short last
// slice).
template <typename TILE, typename ROW_DATA>
DINLINE void tile_gather(TILE& t, ROW_DATA row_data) {
  using P = typename TILE::Pack;
  if (!t.participates()) return;
#pragma unroll
  for (int m = 0; m < TILE::kRows; ++m) {
    const P* row = reinterpret_cast<const P*>(row_data(t.tile_row(m)));
#pragma unroll
    for (int k = 0; k < t.K; ++k) impl::held(t, m, k, impl::pack_load(impl::at_col(row, t, k)));
  }
  impl::issued();
}

// ...a row only `row_n(m)` columns long (a short last slice): its columns past that read its last
// group, and an empty row reads nothing.
template <typename TILE, typename ROW_DATA, typename ROW_N>
DINLINE void tile_gather(TILE& t, ROW_DATA row_data, ROW_N row_n) {
  using P = typename TILE::Pack;
  if (!t.participates()) return;
#pragma unroll
  for (int m = 0; m < TILE::kRows; ++m) {
    const int r = t.tile_row(m);
    const int n = row_n(r) / t.kPack;
    if (n == 0) continue;
    const P* row = reinterpret_cast<const P*>(row_data(r));
#pragma unroll
    for (int k = 0; k < t.K; ++k) {
      const int c = t.offs_n / t.kPack + t.lane() + k * t.kThreadsN;
      impl::held(t, m, k, impl::pack_load(row + (c < n ? c : n - 1)));
    }
  }
  impl::issued();
}

template <typename TILE, typename ROW_DATA, typename ROW_N>
DINLINE void tile_scatter(ROW_DATA row_data, ROW_N row_n, const TILE& t) {
  using P = typename TILE::Pack;
  if (!t.participates()) return;
#pragma unroll
  for (int m = 0; m < TILE::kRows; ++m) {
    const int r = t.tile_row(m);
    P* row      = reinterpret_cast<P*>(row_data(r));
    const int n = row_n(r) / t.kPack;
#pragma unroll
    for (int k = 0; k < t.K; ++k) {
      const int c = t.offs_n / t.kPack + t.lane() + k * t.kThreadsN;
      if (c >= n) continue;
      if constexpr (sizeof(P) == 16)
        impl::global_store(row + c, impl::pack(t, m, k));
      else
        row[c] = impl::pack(t, m, k);
    }
  }
}

// A ROW'S SCALAR (a norm's scale), one float a row of `scalars`: thread 0 stores row `row`'s; every
// rank's for row `row`, `data(r)` its scalars, loaded by every thread, all in flight together (a
// wave's lanes read one address: one request a wave), issued after any tile loads before them.
DINLINE void block_store_row_scalar(float* scalars, int row, float v) {
  if (threadIdx.x == 0) *(__attribute__((address_space(1))) float*)(scalars + row) = v;
}
template <int WORLD, typename RANK_DATA>
DINLINE void peers_load_row_scalars(RANK_DATA data, int row, float (&out)[WORLD]) {
#pragma unroll
  for (int r = 0; r < WORLD; ++r)
    out[r] = *(const __attribute__((address_space(1))) float*)(data(r) + row);
  impl::issued();
}

}  // namespace hip_comms
