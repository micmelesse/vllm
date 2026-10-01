// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE READS AND WRITES: how a pack is loaded and stored, each pair with who can see a store and
// when (the global pair for this GPU's memory and a peer's read after a sync, the uncached pair for
// what a rank writes into a peer and the peer reads back: no kernel does today, a remote write
// must), and a Fragment's load and store.

#pragma once

#ifndef HIP_COMMS_COMMON_INTERFACE
#error "include common/common.cuh, common's one interface, not its parts"
#endif

#include "utils.cuh"

namespace hip_comms {

// PEER MEMORY THROUGH GLOBAL INSTRUCTIONS. A pointer read out of a struct has no address
// space the compiler can prove, so it emits `flat_load`, which checks the aperture and
// waits on both counters; casting to address space 1 gives `global_load_dwordx4` /
// `global_store_dwordx4`.
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(1))) u32x4 global_u32x4;

template <typename V>
DINLINE V thread_load(const V* p) {
  static_assert(sizeof(V) == 16, "a pack is 16 bytes");
  const u32x4 raw = *(const global_u32x4*)(p);
  V v;
  __builtin_memcpy(&v, &raw, 16);
  return v;
}

template <typename V>
DINLINE void thread_store(V* p, const V& v) {
  static_assert(sizeof(V) == 16, "a pack is 16 bytes");
  u32x4 raw;
  __builtin_memcpy(&raw, &v, 16);
  *(global_u32x4*)(p) = raw;
}

// WHAT A PEER PUSHED, read past every cache (system-scope loads, `sc0 sc1`), as
// QuickReduce reads what it receives: a peer's stores into this GPU's memory do not reach
// this GPU's L2, so a plain load could return a line cached before they landed.
template <typename V>
DINLINE V thread_load_uncached(const V* p) {
  static_assert(sizeof(V) == 16, "a pack is 16 bytes");
  const auto* q     = reinterpret_cast<const uint64_t*>(p);
  const uint64_t raw[2] = {
      __scoped_atomic_load_n(q, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM),
      __scoped_atomic_load_n(q + 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM)};
  V v;
  __builtin_memcpy(&v, raw, 16);
  return v;
}

// WHAT A RANK WRITES INTO A PEER, stored past every cache: one 16-byte store at system scope
// (`sc0 sc1`, written through to the peer), the twin of `thread_load_uncached`. A plain store to a
// peer's memory can be acknowledged before the peer can see it, so a wave's `s_waitcnt
// vmcnt(0)` before the barrier did not mean the data had landed, and the slowest rank's
// writes arrived after its barrier flag (readers read the previous call's values). With the
// scope bits the same wait covers it. IN ASM because no builtin spells this store: a
// system-scope atomic store compiled to a compare-and-swap loop and an L2 writeback each.
template <typename V>
DINLINE void thread_store_uncached(V* p, const V& v) {
  static_assert(sizeof(V) == 16, "a pack is 16 bytes");
  u32x4 raw;
  __builtin_memcpy(&raw, &v, 16);
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1" ::"v"(p), "v"(raw) : "memory");
}

// A FRAGMENT'S LOAD AND STORE: every load issued (a pack past the row reads the last one), and a
// store only of the packs inside the row (stores do not hold up loads, so the guard costs
// nothing). A 16-byte pack goes through the one-pack global instructions above; an fp32 weight's
// pack is 32 bytes and loads as the compiler chooses.
template <int K, typename V>
DINLINE void thread_load(const V* row, const Fragment<K>& f, V (&out)[K]) {
#pragma unroll
  for (int k = 0; k < K; ++k) {
    if constexpr (sizeof(V) == 16) out[k] = thread_load(row + f.at[k]);
    else out[k] = row[f.at[k]];
  }
}

// ISSUED HERE, NOT WHERE THE COMPILER LIKES: no instruction is scheduled across this point, so
// every load above it is in flight before anything below it runs. Without it the scheduler sank
// some of a reduce's peer loads past the adds of the first ones, so their round trips ran partly
// one after another (0.4 us at 4-16 tokens: ISA 2026-10-01T00-39-51Z).
namespace impl {
DINLINE void issued() { __builtin_amdgcn_sched_barrier(0); }
}  // namespace impl

// PACK i OF EVERY SOURCE, all in flight together; `read(r, i)` is pack i of source r. Nothing
// waits until a pack is used (peers_reduce), so loads issued here can run under other work.
template <typename T, int ngpus, typename Read>
DINLINE PeerPacks<T, ngpus> peers_load(Read read, int64_t i) {
  PeerPacks<T, ngpus> out;
#pragma unroll
  for (int r = 0; r < ngpus; ++r) out.p[r][0] = read(r, i);
  impl::issued();
  return out;
}

// THIS THREAD'S FRAGMENT OF ROW `row` from every source: every pack's loads go out together (a pack
// past the row reads the last one, weighted zero where it is used), where an `if (i < packs)` made
// each pack's loads wait on the one before.
template <typename T, int ngpus, int K, typename Read>
DINLINE PeerPacks<T, ngpus, K> peers_load(Read read, int row, int packs, const Fragment<K>& f) {
  // A pack position at a time, every source's for it: the address is computed once a position (the
  // other way round cost 16 scalar instructions at two packs: ISA 2026-10-01T00-31-14Z).
  PeerPacks<T, ngpus, K> out;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int64_t i = int64_t{row} * packs + f.at[k];
#pragma unroll
    for (int r = 0; r < ngpus; ++r) out.p[r][k] = read(r, i);
  }
  impl::issued();
  return out;
}

template <int K, typename V>
DINLINE void thread_store(V* row, const Fragment<K>& f, const V (&v)[K]) {
#pragma unroll
  for (int k = 0; k < K; ++k) {
    if (f.in[k] == 0.0f) continue;
    if constexpr (sizeof(V) == 16) thread_store(row + f.at[k], v[k]);
    else row[f.at[k]] = v[k];
  }
}

}  // namespace hip_comms
