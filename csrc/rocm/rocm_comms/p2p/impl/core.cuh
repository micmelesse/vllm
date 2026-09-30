// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// p2p's primitives, behind p2p.cuh: `Peer`, `Self`, their reads and writes, `barrier`, and in
// `impl` what the barriers are made of. A
// wait that outlives the timeout
// prints where it was and traps, so a hang is an error. Checked, every index is
// bounds-checked and every wait is skewed by a random per-block delay, so a race shows on
// every run. Indices are in 16-byte packs of T.

#pragma once

#ifndef HIP_COMMS_P2P_INTERFACE
#error "include p2p/p2p.cuh, p2p's one interface, not its parts"
#endif

#include <hip/hip_runtime.h>

#include <cstdint>

#include "../../common/memory.cuh"
#include "peers.cuh"

namespace hip_comms::p2p {

namespace impl {


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
// wave's could still be in flight when the flag lands -- a remote store
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

}  // namespace impl

// =================================================================================
// EVERYTHING ONE GPU DOES WITH ANOTHER (listed in p2p.cuh): a rank's buffers are read and
// written only through these, and ordered only by `barrier`.
// =================================================================================

namespace impl {

// Rank r's scratch, the bytes after its signal block, picked from the kernel's arguments: BY
// SELECT, never an index, since a runtime index into a pointer array puts the array in scratch
// memory (seen in the ISA: 152 bytes a lane and a scratch load per read).
template <typename T, int ngpus>
DINLINE typename traits<T>::V* scratch_of(const Peers& p, int r) {
  using V = typename traits<T>::V;
  V* at   = reinterpret_cast<V*>(p.signals.s[0] + 1);
#pragma unroll
  for (int k = 1; k < ngpus; ++k)
    if (r == k) at = reinterpret_cast<V*>(p.signals.s[k] + 1);
  return at;
}

}  // namespace impl

// ONE RANK'S BUFFERS FOR THIS LAUNCH, read-only, held for the kernel and never handed out. Made
// before the loops that read it: one `peer(p, r)` for a rank chosen at run time, or for every rank
//   Peer<T, ngpus> all[ngpus];
//   for (int r = 0; r < ngpus; ++r) all[r] = peer<T, ngpus>(p, r);   // unrolled: r a constant
// so the pointers stay in registers. Nothing takes a runtime index into a set of ranks: that is
// what put them in scratch memory (152 B a lane, +4 us at 1.8 MB).
template <typename T, int ngpus>
class Peer {
  using V = typename traits<T>::V;
  const V* in_      = nullptr;
  const V* scratch_ = nullptr;

 public:
  DINLINE Peer() = default;
  DINLINE Peer(const Peers& p, int r)
      : in_(reinterpret_cast<const V*>(p.inputs->p[r])),
        scratch_(impl::scratch_of<T, ngpus>(p, r)) {}

  template <typename U, int n>
  friend DINLINE typename traits<U>::V read_input(const Peer<U, n>& peer, int64_t i);
  template <typename U, int n>
  friend DINLINE typename traits<U>::V read_scratch(const Peer<U, n>& peer, int64_t i);
};

// THIS RANK'S OWN BUFFERS: its scratch is the only thing a pull kernel writes. Writing into a
// peer's memory is another operation, with its own visibility rule (common/memory.cuh's
// uncached store).
template <typename T, int ngpus>
class Self {
  using V = typename traits<T>::V;
  V* scratch_;

 public:
  explicit DINLINE Self(const Peers& p) : scratch_(impl::scratch_of<T, ngpus>(p, p.rank)) {}

  template <typename U, int n>
  friend DINLINE void write_scratch(const Self<U, n>& self, int64_t i,
                                    const typename traits<U>::V& v);
};

template <typename T, int ngpus>
DINLINE Peer<T, ngpus> peer(const Peers& p, int r) {
  return Peer<T, ngpus>(p, r);
}

template <typename T, int ngpus>
DINLINE Self<T, ngpus> self(const Peers& p) {
  return Self<T, ngpus>(p);
}

// Pack i of that rank's input for this launch: the pull receive.
template <typename T, int ngpus>
DINLINE typename traits<T>::V read_input(const Peer<T, ngpus>& peer, int64_t i) {
  return load_global(peer.in_ + i);
}

// Pack i of what that rank left in its scratch, after a barrier that made it visible.
template <typename T, int ngpus>
DINLINE typename traits<T>::V read_scratch(const Peer<T, ngpus>& peer, int64_t i) {
  return load_global(peer.scratch_ + i);
}

// Pack i of this rank's scratch, for its peers to read after a barrier that makes it visible.
template <typename T, int ngpus>
DINLINE void write_scratch(const Self<T, ngpus>& self, int64_t i,
                           const typename traits<T>::V& v) {
  store_global(self.scratch_ + i, v);
}

// WHO A BARRIER WAITS FOR: this block and the same block on every rank, or every block of this
// rank.
enum class Among { peers, grid };
// WHAT HOLDS ONCE IT IS PASSED: every peer has launched (so its input is ready to read); what this
// side wrote before is visible to the other side after; every peer is done reading this rank (so
// its buffers may be reused). Among the grid, only `visible` means anything.
enum class Ensure { launched, visible, read };

// A read after a peers barrier may see only what the SAME BLOCK on the peer wrote before it, so
// both sides must index the same data by the same block.
template <int ngpus, Among kAmong, Ensure kEnsure>
DINLINE void barrier(const Peers& p) {
  static_assert(kAmong == Among::peers || kEnsure == Ensure::visible,
                "a grid barrier only makes this rank's writes visible to its other blocks");
  if constexpr (kAmong == Among::grid) {
    impl::barrier<ngpus, false>(p);
  } else {
    impl::skew(p);
    impl::pair_blocks<ngpus, kEnsure == Ensure::visible>(p, kEnsure == Ensure::launched);
  }
}

}  // namespace hip_comms::p2p
