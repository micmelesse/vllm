// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// p2p's primitives, behind p2p.cuh: `World`, `start`, the barriers, and in `impl` what
// they and the phases are made of. A wait that outlives the timeout
// prints where it was and traps, so a hang is an error. Checked, every index is
// bounds-checked and every wait is skewed by a random per-block delay, so a race shows on
// every run. Indices are in 16-byte packs of T.

#pragma once

#ifndef HIP_COMMS_P2P_INTERFACE
#error "include p2p/p2p.cuh, p2p's one interface, not its parts"
#endif

#include <hip/hip_runtime.h>

#include <cstdint>

#include "../../utils.cuh"
#include "peers.cuh"

namespace hip_comms::p2p {

// A KERNEL'S VIEW OF THE WORLD, returned by `start`: the launch's `Peers` and every rank's
// input, index 0 this rank's. Plain values; every p2p function takes it, which is how
// T and the world size reach them without being spelled at each call. ROTATED by rank,
// so the ranks do not all read rank 0 first; each then sums in a different order, so a
// pulled sum agrees across ranks to one ULP of T, not bitwise.
template <typename T, int ngpus>
struct World {
  using V = typename traits<T>::V;
  Peers peers;
  const V* in[ngpus];
};

namespace impl {

// Rank `peer`'s scratch, the bytes after its signal block. BY SELECT, NOT an index into
// the pointer array: a runtime index into a register array moves it to scratch memory.
template <typename T, int ngpus>
DINLINE typename traits<T>::V* scratch(const World<T, ngpus>& w, int peer) {
  using V = typename traits<T>::V;
  V* at   = reinterpret_cast<V*>(w.peers.signals.s[0] + 1);
#pragma unroll
  for (int i = 1; i < ngpus; ++i)
    if (peer == i) at = reinterpret_cast<V*>(w.peers.signals.s[i] + 1);
  return at;
}

DINLINE void check(const Peers& p, bool ok, const char* what, int peer, int64_t idx,
                   int64_t limit) {
  if (p.checked && (!ok || idx < 0)) {
    printf("rocm_comms: rank %d block %d thread %d: %s(peer %d, idx %lld) outside "
           "[0, %lld)\n",
           p.rank, blockIdx.x, threadIdx.x, what, peer, static_cast<long long>(idx),
           static_cast<long long>(limit));
    __builtin_trap();
  }
}

// Checked only: up to ~32 x 8K cycles, different per rank, block and peer barrier (the
// block's sequence number, which start, peer_barrier and close advance).
DINLINE void skew(const Peers& p) {
  if (!p.checked) return;
  uint32_t h = static_cast<uint32_t>(p.rank) * 73856093u ^ blockIdx.x * 19349663u ^
               p.self->seq[blockIdx.x] * 83492791u;
  h ^= h >> 13;
  h *= 0x5bd1e995u;
  for (uint32_t n = (h ^ (h >> 15)) % 32; n > 0; --n) __builtin_amdgcn_s_sleep(127);
}

// EVERY WAVE'S STORES DONE before a block barrier that one wave then releases:
// `__syncthreads` waits for none (gfx9 emits no vmcnt wait before s_barrier; seen in the
// ISA), so the releasing wave's fence would cover only its own stores, and another
// wave's could still be in flight when the flag lands -- a push kernel's remote stores
// above all.
DINLINE void wait_stores() { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); }

// The builtin takes the ordering as a literal, so each case is spelled out.
template <int kOrder, int kScope>
DINLINE void fence() {
  constexpr bool system = kScope == __MEMORY_SCOPE_SYSTEM;
  if constexpr (kOrder == __ATOMIC_RELEASE) {
    if constexpr (system) __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    else __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  } else {
    if constexpr (system) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    else __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  }
}

// Spins relaxed and acquires once, after: an acquire per poll would invalidate the
// caches on every iteration of every spinning block.
template <bool kAcquire, int kScope>
DINLINE void wait(const Peers& p, const uint32_t* flag, uint32_t want, const char* what,
                  int peer) {
  const uint64_t t0 = wall_clock64();
  uint32_t seen;
  while ((seen = __scoped_atomic_load_n(flag, __ATOMIC_RELAXED, kScope)) < want) {
    if (wall_clock64() - t0 > p.timeout_ticks) {
      printf("rocm_comms: rank %d block %d timed out in %s, peer %d: flag %u, want %u\n",
             p.rank, blockIdx.x, what, peer, seen, want);
      __builtin_trap();
    }
  }
  if constexpr (kAcquire) fence<__ATOMIC_ACQUIRE, kScope>();
}

// Block b waits for block b on every rank, and for no other block: enough at the ends,
// where it says "every peer has launched" or "every peer is done reading me", and not
// enough between phases, which is what `world_barrier` is for. kOrdered: the store
// releases and the wait acquires, so what the block put before is visible to its peers'
// same-numbered block after (a peer_barrier). Unordered, it only says when
// (start: every peer has launched; close: every peer is done reading us).
template <int ngpus, bool kOrdered>
DINLINE void pair_blocks(const Peers& p, bool start) {
  if (!start) {
    if constexpr (kOrdered) wait_stores();
    __syncthreads();
  }
  Signal* self     = p.self;
  const uint32_t f = self->seq[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    uint32_t* theirs = start ? &p.signals.s[threadIdx.x]->start[blockIdx.x][p.rank]
                             : &p.signals.s[threadIdx.x]->end[blockIdx.x][p.rank];
    uint32_t* mine   = start ? &self->start[blockIdx.x][threadIdx.x]
                             : &self->end[blockIdx.x][threadIdx.x];
    __scoped_atomic_store_n(theirs, f, kOrdered ? __ATOMIC_RELEASE : __ATOMIC_RELAXED,
                            __MEMORY_SCOPE_SYSTEM);
    wait<kOrdered, __MEMORY_SCOPE_DEVICE>(p, mine, f, start ? "start" : "peer barrier",
                                          threadIdx.x);
  }
  __syncthreads();
  if (threadIdx.x == 0) self->seq[blockIdx.x] = f;
}

// The grid on this device, then (kPeers) one exchange with the peers by the last block
// to arrive, then the grid released. ONE FENCE PER BLOCK, by thread 0 after the block
// barrier: every wave first waits for its own stores (`wait_stores`), and a release
// writes back the whole L2, so one covers the block where one per thread wrote it back
// 512 times.
template <int ngpus, bool kPeers>
DINLINE void barrier(const Peers& p) {
  constexpr int kScope = kPeers ? __MEMORY_SCOPE_SYSTEM : __MEMORY_SCOPE_DEVICE;
  skew(p);
  wait_stores();
  __syncthreads();
  if (threadIdx.x == 0) {
    fence<__ATOMIC_RELEASE, kScope>();
    Signal* self     = p.self;
    const uint32_t g = __scoped_atomic_load_n(&self->gen, __ATOMIC_ACQUIRE, kScope);
    if (__scoped_atomic_fetch_add(&self->arrive, 1u, __ATOMIC_ACQ_REL,
                                  __MEMORY_SCOPE_DEVICE) == gridDim.x - 1) {
      __scoped_atomic_store_n(&self->arrive, 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      if constexpr (kPeers) {
        const uint32_t e = self->epoch + 1;
        self->epoch      = e;
        fence<__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM>();
#pragma unroll
        for (int i = 0; i < ngpus; ++i)
          __scoped_atomic_store_n(&p.signals.s[i]->peer[p.rank], e, __ATOMIC_RELAXED,
                                  __MEMORY_SCOPE_SYSTEM);
#pragma unroll
        for (int i = 0; i < ngpus; ++i)
          wait<true, __MEMORY_SCOPE_SYSTEM>(p, &self->peer[i], e, "world_barrier: peer", i);
      }
      __scoped_atomic_fetch_add(&self->gen, 1u, __ATOMIC_RELEASE, kScope);
    } else {
      wait<true, kScope>(p, &self->gen, g + 1, kPeers ? "world_barrier" : "grid_barrier",
                         -1);
    }
  }
  __syncthreads();
}

// kBatch input packs at once, idx + u * stride for u < kBatch (those at or past `limit`
// are skipped), summed over ranks in fp32 and rounded once: every load from every peer is
// issued before any is added, so kBatch x ngpus are in flight rather than ngpus.
template <int kBatch, typename T, int ngpus>
DINLINE void sum(const World<T, ngpus>& w, int64_t idx, int64_t stride, int64_t limit,
                 typename traits<T>::V (&out)[kBatch]) {
  using V         = typename traits<T>::V;
  constexpr int N = traits<T>::N;
  V raw[kBatch][ngpus];
#pragma unroll
  for (int u = 0; u < kBatch; ++u) {
    const int64_t at = idx + u * stride;
    if (at < limit) {
      check(w.peers, at < w.peers.input_packs, "sum", -1, at, w.peers.input_packs);
#pragma unroll
      for (int i = 0; i < ngpus; ++i) raw[u][i] = load_global(w.in[i] + at);
    }
  }
#pragma unroll
  for (int u = 0; u < kBatch; ++u) {
    float acc[N];
#pragma unroll
    for (int j = 0; j < N; ++j) acc[j] = static_cast<float>(raw[u][0].d[j]);
#pragma unroll
    for (int i = 1; i < ngpus; ++i)
#pragma unroll
      for (int j = 0; j < N; ++j) acc[j] += static_cast<float>(raw[u][i].d[j]);
#pragma unroll
    for (int j = 0; j < N; ++j) out[u].d[j] = static_cast<T>(acc[j]);
  }
}

// This rank's own input pack.
template <typename T, int ngpus>
DINLINE typename traits<T>::V mine(const World<T, ngpus>& w, int64_t idx) {
  check(w.peers, idx < w.peers.input_packs, "mine", -1, idx, w.peers.input_packs);
  return load_global(w.in[0] + idx);
}

template <typename T, int ngpus>
DINLINE typename traits<T>::V get(const World<T, ngpus>& w, int peer, int64_t idx) {
  check(w.peers, peer >= 0 && peer < ngpus && idx < w.peers.scratch_packs, "get", peer, idx,
        w.peers.scratch_packs);
  return load_global(scratch(w, peer) + idx);
}

// A pack peers pushed into this rank's scratch (see `load_uncached`).
template <typename T, int ngpus>
DINLINE typename traits<T>::V get_pushed(const World<T, ngpus>& w, int64_t idx) {
  check(w.peers, idx < w.peers.scratch_packs, "get_pushed", w.peers.rank, idx,
        w.peers.scratch_packs);
  return load_uncached(scratch(w, w.peers.rank) + idx);
}

// A float of peer's scratch, `idx` in floats: a codec's scales.
template <typename T, int ngpus>
DINLINE void put_float(const World<T, ngpus>& w, int peer, int64_t idx, float v) {
  check(w.peers, peer >= 0 && peer < ngpus && idx < 4 * w.peers.scratch_packs, "put_float",
        peer, idx, 4 * w.peers.scratch_packs);
  __scoped_atomic_store_n(reinterpret_cast<uint32_t*>(scratch(w, peer)) + idx,
                          __builtin_bit_cast(uint32_t, v), __ATOMIC_RELAXED,
                          __MEMORY_SCOPE_SYSTEM);
}

template <typename T, int ngpus>
DINLINE float get_float(const World<T, ngpus>& w, int peer, int64_t idx) {
  check(w.peers, peer >= 0 && peer < ngpus && idx < 4 * w.peers.scratch_packs, "get_float",
        peer, idx, 4 * w.peers.scratch_packs);
  const auto* at = reinterpret_cast<const uint32_t*>(scratch(w, peer)) + idx;
  return __builtin_bit_cast(float, __scoped_atomic_load_n(at, __ATOMIC_RELAXED,
                                                          __MEMORY_SCOPE_SYSTEM));
}

template <typename T, int ngpus>
DINLINE void put(const World<T, ngpus>& w, int peer, int64_t idx,
                 const typename traits<T>::V& v) {
  check(w.peers, peer >= 0 && peer < ngpus && idx < w.peers.scratch_packs, "put",
                peer, idx, w.peers.scratch_packs);
  store_global(scratch(w, peer) + idx, v);
}

// A direct pointer to n packs of peer's scratch, checked once, for a hot loop.
template <typename T, int ngpus>
DINLINE const typename traits<T>::V* ptr(const World<T, ngpus>& w, int peer, int64_t idx,
                                         int64_t n) {
  check(w.peers, peer >= 0 && peer < ngpus && idx + n <= w.peers.scratch_packs,
                "ptr", peer, idx + n, w.peers.scratch_packs);
  return scratch(w, peer) + idx;
}

}  // namespace impl

// =================================================================================
// THE PRIMITIVES (listed in p2p.cuh).
// =================================================================================

template <typename T, int ngpus>
DINLINE World<T, ngpus> start(const Peers& p) {
  impl::skew(p);
  impl::pair_blocks<ngpus, false>(p, true);
  World<T, ngpus> w{p, {}};
#pragma unroll
  for (int i = 0; i < ngpus; ++i)
    w.in[i] = reinterpret_cast<const typename traits<T>::V*>(
        p.inputs->p[(p.rank + i) % ngpus]);
  return w;
}

// This block and the same block on every peer (every rank, block b with block b), and no
// other block: one flag per peer, where `world_barrier` waits for the whole grid. A read
// after it may see ONLY what the same block on that peer wrote, so both phases must give
// each block the same rows (vLLM's custom all-reduce, and the rule its two-stage kernel
// states).
template <typename T, int ngpus>
DINLINE void peer_barrier(const World<T, ngpus>& w) {
  impl::skew(w.peers);
  impl::pair_blocks<ngpus, true>(w.peers, false);
}

template <typename T, int ngpus>
DINLINE void world_barrier(const World<T, ngpus>& w) {
  impl::barrier<ngpus, true>(w.peers);
}

// Every block of THIS rank's kernel: puts to our own scratch before it are visible to our
// own reads after it. Cheaper than `world_barrier`, and wrong for anything a peer put.
template <typename T, int ngpus>
DINLINE void grid_barrier(const World<T, ngpus>& w) {
  impl::barrier<ngpus, false>(w.peers);
}

template <typename T, int ngpus>
DINLINE void close(const World<T, ngpus>& w) {
  impl::skew(w.peers);
  impl::pair_blocks<ngpus, false>(w.peers, false);
}

}  // namespace hip_comms::p2p
