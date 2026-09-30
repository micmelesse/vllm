// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE READS AND WRITES: how a pack is loaded and stored, each pair with who can see a store and
// when (the global pair for this GPU's memory and a peer's read after a sync, the uncached pair for
// what a rank writes into a peer and the peer reads back: no kernel does today, a remote write
// must), and a Fragment's load and store.

#pragma once

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

// A FRAGMENT'S LOAD AND STORE: every load issued (a pack past the row reads the last one; any
// element type, an fp32 weight's pack is 32 bytes), and a store only of the packs inside the row
// (stores do not hold up loads, so the guard costs nothing).
template <int K, typename V>
DINLINE void thread_load(const V* row, const Fragment<K>& f, V (&out)[K]) {
#pragma unroll
  for (int k = 0; k < K; ++k) out[k] = row[f.at[k]];
}

template <int K, typename V>
DINLINE void thread_store(V* row, const Fragment<K>& f, const V (&v)[K]) {
#pragma unroll
  for (int k = 0; k < K; ++k)
    if (f.in[k] != 0.0f) row[f.at[k]] = v[k];
}

}  // namespace hip_comms
