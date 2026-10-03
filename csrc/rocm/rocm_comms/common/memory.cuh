// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE READS AND WRITES: how a pack is loaded and stored, each pair with who can see a store and
// when (the global pair for this GPU's memory and a peer's read after a sync, the uncached pair for
// what a rank writes into a peer and the peer reads back: no kernel does today, a remote write
// must), and a row's load and store at a thread's columns.

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

template <typename PACK>
DINLINE PACK thread_load(const PACK* p) {
  static_assert(sizeof(PACK) == 16, "a pack is 16 bytes");
  const u32x4 raw = *(const global_u32x4*)(p);
  PACK v;
  __builtin_memcpy(&v, &raw, 16);
  return v;
}

template <typename PACK>
DINLINE void thread_store(PACK* p, const PACK& v) {
  static_assert(sizeof(PACK) == 16, "a pack is 16 bytes");
  u32x4 raw;
  __builtin_memcpy(&raw, &v, 16);
  *(global_u32x4*)(p) = raw;
}

// WHAT A PEER PUSHED, read past every cache (system-scope loads, `sc0 sc1`), as
// QuickReduce reads what it receives: a peer's stores into this GPU's memory do not reach
// this GPU's L2, so a plain load could return a line cached before they landed.
template <typename PACK>
DINLINE PACK thread_load_uncached(const PACK* p) {
  static_assert(sizeof(PACK) == 16, "a pack is 16 bytes");
  const auto* q     = reinterpret_cast<const uint64_t*>(p);
  const uint64_t raw[2] = {
      __scoped_atomic_load_n(q, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM),
      __scoped_atomic_load_n(q + 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM)};
  PACK v;
  __builtin_memcpy(&v, raw, 16);
  return v;
}

// A STORE PAST EVERY CACHE: one 16-byte store at system scope (`sc0 sc1`, written through), the
// twin of `thread_load_uncached`. Unused today: it was the push's store into a peer while scratch
// was allocated cached (a plain store there could be acknowledged before the peer saw it, and the
// slowest rank's writes landed after its barrier flag). Scratch is now allocated uncached and the
// push stores plainly into it, as aiter's does (500dd54535; tests passed and timings were about
// equal, 2026-09-30T23-10-02Z). IN ASM because no builtin spells this store: a
// system-scope atomic store compiled to a compare-and-swap loop and an L2 writeback each.
template <typename PACK>
DINLINE void thread_store_uncached(PACK* p, const PACK& v) {
  static_assert(sizeof(PACK) == 16, "a pack is 16 bytes");
  u32x4 raw;
  __builtin_memcpy(&raw, &v, 16);
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1" ::"v"(p), "v"(raw) : "memory");
}

// ISSUED HERE, NOT WHERE THE COMPILER LIKES: no memory operation moves across this point (the
// empty asm's memory clobber) and no instruction is scheduled across it (sched_barrier), so every
// load above it is in flight before anything below it runs. The scheduler alone sank a reduce's
// peer loads past the adds of the first ones (ISA 2026-10-01T00-39-51Z); with sched_barrier alone,
// a tile's weight loads were sunk to their uses, one register quad and one round trip each (ISA
// 2026-10-03T21-00-27Z).
namespace impl {
DINLINE void issued() {
  asm volatile("" ::: "memory");
  __builtin_amdgcn_sched_barrier(0);
}
}  // namespace impl

// PACK i OF EVERY SOURCE, all in flight together; `read(r, i)` is pack i of source r. Nothing
// waits until a pack is used (peers_reduce), so loads issued here can run under other work.
template <typename DTYPE, int NGPUS, typename READ_PEER>
DINLINE PeerPacks<DTYPE, NGPUS> peers_load(READ_PEER read, int64_t i) {
  PeerPacks<DTYPE, NGPUS> out;
#pragma unroll
  for (int r = 0; r < NGPUS; ++r) out.p[r][0] = read(r, i);
  impl::issued();
  return out;
}

// A TILE'S LOAD AND STORE, `data` the tensor's first element and `row_stride` elements between its
// rows: every load issued together, ONE ROUND TRIP A TILE (a row past M reads row M - 1, a pack
// past N the last one), and a store of only the rows below M and the packs below N (stores do not
// hold up loads, so the guard costs nothing). LOAD A TILE WHOLE BEFORE STORING ANYTHING: a pack
// loaded between stores waits a round trip, since a store may alias it. A 16-byte pack goes through
// the one-pack global instructions above; an fp32 tile's pack is 32 bytes and loads as the compiler
// chooses.
namespace impl {
template <typename PACK>
DINLINE PACK pack_load(const PACK* p) {
  if constexpr (sizeof(PACK) == 16)
    return thread_load(p);
  else
    return *p;
}
}  // namespace impl

template <typename TILE>
DINLINE void thread_load(TILE& t, const typename TILE::Acc* data,
                         int64_t row_stride) {
  using P = typename TILE::Pack;
  const P* at = reinterpret_cast<const P*>(data);
#pragma unroll
  for (int m = 0; m < TILE::kRows; ++m) {
    const P* row = at + int64_t{t.row(m)} * (row_stride / t.kPack);
#pragma unroll
    for (int k = 0; k < t.K; ++k) t.v[m][k] = impl::pack_load(row + t.col(k));
  }
  impl::issued();
}

template <typename TILE>
DINLINE void thread_store(typename TILE::Acc* data, int64_t row_stride, const TILE& t) {
  using P = typename TILE::Pack;
  P* at   = reinterpret_cast<P*>(data);
#pragma unroll
  for (int m = 0; m < TILE::kRows; ++m) {
    if (!t.live(m)) continue;
    P* row = at + int64_t{t.row(m)} * (row_stride / t.kPack);
#pragma unroll
    for (int k = 0; k < t.K; ++k) {
      if (t.mask(k) == 0.0f) continue;
      if constexpr (sizeof(P) == 16)
        thread_store(row + t.col(k), t.v[m][k]);
      else
        row[t.col(k)] = t.v[m][k];
    }
  }
}

// EVERY PEER'S TILE, all in flight together: `data(r)` is rank r's tensor, `row_stride` its
// elements between rows. A pack position at a time, every rank's for it, so the address is
// computed once a position (the other way round cost 16 scalar instructions at two packs: ISA
// 2026-10-01T00-31-14Z). Nothing waits until a tile is used (peers_reduce), so loads issued here
// can run under other work.
template <typename TILE, int NGPUS, typename RANK_DATA>
DINLINE void peers_load(TILE (&t)[NGPUS], RANK_DATA data,
                        int64_t row_stride) {
  using P = typename TILE::Pack;
#pragma unroll
  for (int m = 0; m < TILE::kRows; ++m)
#pragma unroll
    for (int k = 0; k < t[0].K; ++k) {
      const int64_t i = int64_t{t[0].row(m)} * (row_stride / t[0].kPack) + t[0].col(k);
#pragma unroll
      for (int r = 0; r < NGPUS; ++r)
        t[r].v[m][k] = impl::pack_load(reinterpret_cast<const P*>(data(r)) + i);
    }
  impl::issued();
}

// A TILE WHOSE COLUMNS ARE SPLIT AMONG THE RANKS, `slice` columns each (the last rank's to the
// end): each pack from its owner's tensor, `data(r)`. A SLICE IS WHOLE WAVES, so a wave's packs
// have one owner.
template <int NGPUS, typename TILE, typename RANK_DATA>
DINLINE void sliced_load(TILE& t, RANK_DATA data, int64_t row_stride,
                         int slice) {
  using P = typename TILE::Pack;
#pragma unroll
  for (int m = 0; m < TILE::kRows; ++m)
#pragma unroll
    for (int k = 0; k < t.K; ++k) {
      const int owner = min(t.col(k) / (slice / t.kPack), NGPUS - 1);
      t.v[m][k] = impl::pack_load(reinterpret_cast<const P*>(data(owner)) +
                                  int64_t{t.row(m)} * (row_stride / t.kPack) + t.col(k));
    }
  impl::issued();
}

}  // namespace hip_comms
