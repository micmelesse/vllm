// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// p2p's synchronization, behind p2p.cuh: `barrier`, the flags, and in `impl` what they are made
// of. A rank's buffers and their reads and writes are buffers.cuh's.
//
// HIP_COMMS_DEBUG=1 BUILDS THE TESTS' MACHINERY IN: a wait that outlives the timeout prints where
// it was and traps, so a hang is an error, and every wait is skewed by a random per-block
// delay, so a race shows on every run. Off (the default), none of it is in the kernels: a clock
// read on every poll and a printf path in every kernel are not free.

#pragma once

#ifndef HIP_COMMS_P2P_INTERFACE
#error "include p2p/p2p.cuh, p2p's one interface, not its parts"
#endif

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>

#include "../../common/common.cuh"
#include "peers.cuh"

namespace hip_comms::p2p {

#ifndef HIP_COMMS_DEBUG
#define HIP_COMMS_DEBUG 0
#endif

namespace impl {

// Debug builds only: up to ~32 x 8K cycles, different per rank, block and peer barrier (the
// block's sequence number, which start, peer_barrier and close advance).
DINLINE void skew(const DevComm& p) {
  if (!HIP_COMMS_DEBUG) return;
  uint32_t h = static_cast<uint32_t>(p.rank) * 73856093u ^ blockIdx.x * 19349663u ^
               own_signals(p).seq(blockIdx.x) * 83492791u;
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
DINLINE void wait(const DevComm& p, const Counter& flag, uint32_t want, const char* what,
                  int peer) {
#if HIP_COMMS_DEBUG
  const uint64_t t0 = wall_clock64();
  uint32_t seen;
  while ((seen = flag.load<__ATOMIC_RELAXED, kScope>()) < want) {
    if (wall_clock64() - t0 > p.timeout_ticks) {
      printf("rocm_comms: rank %d block %d timed out in %s, peer %d: flag %u, want %u\n",
             p.rank, blockIdx.x, what, peer, seen, want);
      __builtin_trap();
    }
  }
#else
  (void)p, (void)what, (void)peer;
  while (flag.load<__ATOMIC_RELAXED, kScope>() < want) {
  }
#endif
  if constexpr (kAcquire) fence<__ATOMIC_ACQUIRE, kScope>();
}

// Block b waits for block b on every rank, and for no other block: enough at the ends,
// where it says "every peer has launched" or "every peer is done reading me", and not
// enough between phases, which is what `world_barrier` is for. kOrdered: the store
// releases and the wait acquires, so what the block put before is visible to its peers'
// same-numbered block after (a peer_barrier). Unordered, it only says when
// (start: every peer has launched; close: every peer is done reading us).
template <int ngpus, bool kOrdered>
DINLINE void pair_blocks(const DevComm& p, bool start) {
  if (!start) {
    if constexpr (kOrdered) wait_stores();
    __syncthreads();
  }
  const Signals own = own_signals(p);
  const uint32_t f  = own.seq(blockIdx.x) + 1;
  if (threadIdx.x < ngpus) {
    const Signals peer   = lane_signals(p, threadIdx.x);
    const Counter theirs = start ? peer.start(blockIdx.x, p.rank) : peer.end(blockIdx.x, p.rank);
    const Counter mine   = start ? own.start(blockIdx.x, threadIdx.x)
                                 : own.end(blockIdx.x, threadIdx.x);
    theirs.store<kOrdered ? __ATOMIC_RELEASE : __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM>(f);
    wait<kOrdered, __MEMORY_SCOPE_DEVICE>(p, mine, f, start ? "start" : "peer barrier",
                                          threadIdx.x);
  }
  __syncthreads();
  if (threadIdx.x == 0) own.set_seq(blockIdx.x, f);
}

// The grid on this device, then (kPeers) one exchange with the peers by the last block
// to arrive, then the grid released. ONE FENCE PER BLOCK, by thread 0 after the block
// barrier: every wave first waits for its own stores (`wait_stores`), and a release
// writes back the whole L2, so one covers the block where one per thread wrote it back
// 512 times.
template <int ngpus, bool kPeers>
DINLINE void barrier(const DevComm& p) {
  constexpr int kScope = kPeers ? __MEMORY_SCOPE_SYSTEM : __MEMORY_SCOPE_DEVICE;
  skew(p);
  wait_stores();
  __syncthreads();
  if (threadIdx.x == 0) {
    fence<__ATOMIC_RELEASE, kScope>();
    const Signals own = own_signals(p);
    const uint32_t g  = own.gen().load<__ATOMIC_ACQUIRE, kScope>();
    if (own.arrive().fetch_add<__ATOMIC_ACQ_REL, __MEMORY_SCOPE_DEVICE>(1u) == gridDim.x - 1) {
      own.arrive().store<__ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE>(0u);
      if constexpr (kPeers) {
        const uint32_t e = own.epoch() + 1;
        own.set_epoch(e);
        fence<__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM>();
#pragma unroll
        for (int i = 0; i < ngpus; ++i)
          lane_signals(p, i).peer(p.rank).store<__ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM>(e);
#pragma unroll
        for (int i = 0; i < ngpus; ++i)
          wait<true, __MEMORY_SCOPE_SYSTEM>(p, own.peer(i), e, "world_barrier: peer", i);
      }
      own.gen().fetch_add<__ATOMIC_RELEASE, kScope>(1u);
    } else {
      wait<true, kScope>(p, own.gen(), g + 1, kPeers ? "world_barrier" : "grid_barrier", -1);
    }
  }
  __syncthreads();
}

}  // namespace impl

// =================================================================================
// HOW ONE GPU SYNCHRONIZES WITH ANOTHER (listed in p2p.cuh): the barriers and flags, written
// against a rank's Signals (buffers.cuh); a rank's data buffers are ordered only by `barrier`.
// =================================================================================

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
DINLINE void barrier(const DevComm& p) {
  static_assert(kAmong == Among::peers || kEnsure == Ensure::visible,
                "a grid barrier only makes this rank's writes visible to its other blocks");
  if constexpr (kAmong == Among::grid) {
    impl::barrier<ngpus, false>(p);
  } else {
    impl::skew(p);
    impl::pair_blocks<ngpus, kEnsure == Ensure::visible>(p, kEnsure == Ensure::launched);
  }
}

// A FLAG TO ONE PEER, with the barriers' own store and spin: `v` lands in `peer`'s signal block,
// in its slot for this rank, and `wait_flag` spins until `peer`'s flag here reaches `v`. Flags only
// grow, so a caller counts on from the last value it used.
DINLINE void write_flag(const DevComm& p, int peer, uint32_t v) {
  signals(p, peer).flag(p.rank).store<__ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM>(v);
}

DINLINE void wait_flag(const DevComm& p, int peer, uint32_t v) {
  impl::wait<false, __MEMORY_SCOPE_DEVICE>(p, own_signals(p).flag(peer), v, "flag", peer);
}

}  // namespace hip_comms::p2p
