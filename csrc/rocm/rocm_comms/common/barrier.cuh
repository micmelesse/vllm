// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE RANKS' SYNCHRONIZATION: a rank's Signal block, whose only view (`Signals`) hands out its
// counters, each touched only atomically; `barrier`, the flags, and in `impl` what they are made
// of. A rank's buffers are peers.cuh's.
//
// HIP_COMMS_DEBUG=1 BUILDS THE TESTS' MACHINERY IN: a wait that outlives the timeout prints where
// it was before it traps, and every wait is skewed by a random per-block delay, so a race shows
// on every run. Off (the default), a wait still backs off and traps at the timeout, but reads the
// clock every 256 polls and prints nothing: a clock read a poll and a printf path in every kernel
// are not free.

#pragma once

#ifndef HIP_COMMS_COMMON_INTERFACE
#error "include common/interface.cuh, common's one interface, not its parts"
#endif

#include <hip/hip_runtime.h>
#include <array>
#include <cstdint>

#include "peers.cuh"
#include "utils.cuh"

namespace hip_comms {

// ONE SIGNAL COUNTER: only atomic operations, each with its order and scope spelled at the call. A
// plain load or store of a counter cannot be written.
class Counter {
  uint32_t* at_;

 public:
  explicit DINLINE Counter(uint32_t* at) : at_(at) {}
  template <int MEMORY_ORDER, int MEMORY_SCOPE>
  DINLINE void store(uint32_t v) const {
    __scoped_atomic_store_n(at_, v, MEMORY_ORDER, MEMORY_SCOPE);
  }
  template <int MEMORY_ORDER, int MEMORY_SCOPE>
  DINLINE uint32_t load() const {
    return __scoped_atomic_load_n(at_, MEMORY_ORDER, MEMORY_SCOPE);
  }
  template <int MEMORY_ORDER, int MEMORY_SCOPE>
  DINLINE uint32_t fetch_add(uint32_t v) const {
    return __scoped_atomic_fetch_add(at_, v, MEMORY_ORDER, MEMORY_SCOPE);
  }
};

// A RANK'S SIGNAL BLOCK: its counters, handed out one at a time. `seq` (a block's own, written only
// by that block) and `epoch` (written only by the grid's last block to arrive) are plain: nothing
// else ever touches them.
class Signals {
  Signal* s_;

 public:
  explicit DINLINE Signals(Signal* s) : s_(s) {}
  // Block `block`'s pairing slot for rank `rank`, at a launch's start or at its other barriers.
  // AN INDEX KEEPS ITS CALLER'S TYPE (a block unsigned, as blockIdx.x; a rank as passed): an int
  // where the caller had an unsigned costs a sign extension in every kernel.
  template <typename RANK_TYPE>
  DINLINE Counter start(unsigned block, RANK_TYPE rank) const {
    return Counter(&s_->start[block][rank]);
  }
  template <typename RANK_TYPE>
  DINLINE Counter end(unsigned block, RANK_TYPE rank) const {
    return Counter(&s_->end[block][rank]);
  }
  // The grid barrier's: rank `rank`'s epoch here, the arrivals, the generation.
  DINLINE Counter peer(int rank) const { return Counter(&s_->peer[rank]); }
  DINLINE Counter arrive() const { return Counter(&s_->arrive); }
  DINLINE Counter gen() const { return Counter(&s_->gen); }
  // The last flag rank `rank` wrote here (write_flag).
  DINLINE Counter flag(int rank) const { return Counter(&s_->flag[rank]); }
  DINLINE uint32_t seq(unsigned block) const { return s_->seq[block]; }
  DINLINE void set_seq(unsigned block, uint32_t v) const { s_->seq[block] = v; }
  DINLINE uint32_t epoch() const { return s_->epoch; }
  DINLINE void set_epoch(uint32_t v) const { s_->epoch = v; }
};

namespace impl {

// Rank r's signal block, by select.
DINLINE Signal* signal_of(const PeerSignals& peer_signals, int r) {
  Signal* at = peer_signals.s[0];
#pragma unroll
  for (int k = 1; k < kMaxRanks; ++k)
    if (r == k) at = peer_signals.s[k];
  return at;
}

}  // namespace impl

// This rank's signal block; rank r's, r the same across the wave (by select); and thread i's peer
// i's, each thread its own (a per-lane load, as the pairing barrier issues it).
namespace impl {
DINLINE Signals own_signals(Signal* self_signal) { return Signals(self_signal); }

DINLINE Signals signals(const PeerSignals& peer_signals, int r) {
  return Signals(impl::signal_of(peer_signals, r));
}

template <typename INDEX_TYPE>
DINLINE Signals lane_signals(const PeerSignals& peer_signals, INDEX_TYPE i) {
  return Signals(peer_signals.s[i]);
}
}  // namespace impl



#ifndef HIP_COMMS_DEBUG
#define HIP_COMMS_DEBUG 0
#endif

namespace impl {

// Debug builds only: up to ~32 x 8K cycles, different per rank, block and peer barrier (the
// block's sequence number, which every peers barrier advances).
DINLINE void skew(int rank, uint32_t seq) {
  if (!HIP_COMMS_DEBUG) return;
  uint32_t h = static_cast<uint32_t>(rank) * 73856093u ^ blockIdx.x * 19349663u ^ seq * 83492791u;
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
template <int MEMORY_ORDER, int MEMORY_SCOPE>
DINLINE void fence() {
  constexpr bool system = MEMORY_SCOPE == __MEMORY_SCOPE_SYSTEM;
  if constexpr (MEMORY_ORDER == __ATOMIC_RELEASE) {
    if constexpr (system) __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    else __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  } else {
    if constexpr (system) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    else __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  }
}

// Spins relaxed and acquires once, after: an acquire per poll would invalidate the
// caches on every iteration of every spinning block.
template <bool ACQUIRE, int MEMORY_SCOPE>
DINLINE void wait(uint64_t timeout_ticks, int rank, const Counter& flag, uint32_t want,
                  const char* what, int peer) {
#if HIP_COMMS_DEBUG
  const uint64_t t0 = wall_clock64();
  uint32_t seen;
  while ((seen = flag.load<__ATOMIC_RELAXED, MEMORY_SCOPE>()) < want) {
    if (wall_clock64() - t0 > timeout_ticks) {
      printf("rocm_comms: rank %d block %d timed out in %s, peer %d: flag %u, want %u\n",
             rank, blockIdx.x, what, peer, seen, want);
      __builtin_trap();
    }
  }
#else
  // A RELEASE BUILD BACKS OFF AND TIMES OUT. A sleep a poll (64 cycles) keeps a spinning block
  // from hammering the link with polls; a clock read every 256 polls bounds a hang to the build's
  // timeout, a trap rather than every rank's blocks spinning until the process is killed. THE TRAP
  // SAYS WHY FIRST: which rank, block, barrier and peer, and the peer's count against the one
  // wanted (a peer short by a few is a launch one rank skipped). Printed only on the way to the
  // trap; a bare trap was a silent HSA 0x1016 (the 2026-10-05 e2e test).
  const uint64_t t0 = wall_clock64();
  for (uint32_t n = 1; flag.load<__ATOMIC_RELAXED, MEMORY_SCOPE>() < want; ++n) {
    __builtin_amdgcn_s_sleep(1);
    if ((n & 255u) == 0 && wall_clock64() - t0 > timeout_ticks) {
      printf("rocm_comms: rank %d block %d timed out in %s, peer %d: count %u, want %u\n", rank,
             blockIdx.x, what, peer, flag.load<__ATOMIC_RELAXED, MEMORY_SCOPE>(), want);
      __builtin_trap();
    }
  }
#endif
  if constexpr (ACQUIRE) fence<__ATOMIC_ACQUIRE, MEMORY_SCOPE>();
}

// Block b waits for block b on every rank, and for no other block: enough at the ends,
// where it says "every peer has launched" or "every peer is done reading me", and not
// enough between phases, which is what `world_barrier` is for. kOrdered: the store
// releases and the wait acquires, so what the block put before is visible to its peers'
// same-numbered block after (a peer_barrier). Unordered, it only says when
// (start: every peer has launched; close: every peer is done reading us).
template <int WORLD, bool ORDERED>
DINLINE void pair_blocks(const PeerSignals& peer_signals, Signal* self_signal, int rank,
                         uint64_t timeout_ticks, bool start, uint32_t f) {
  if (!start) {
    if constexpr (ORDERED) wait_stores();
    __syncthreads();
  }
  const Signals own = own_signals(self_signal);
  if (threadIdx.x < WORLD) {
    const Signals peer   = lane_signals(peer_signals, threadIdx.x);
    const Counter theirs = start ? peer.start(blockIdx.x, rank) : peer.end(blockIdx.x, rank);
    const Counter mine   = start ? own.start(blockIdx.x, threadIdx.x)
                                 : own.end(blockIdx.x, threadIdx.x);
    theirs.store<ORDERED ? __ATOMIC_RELEASE : __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM>(f);
    wait<ORDERED, __MEMORY_SCOPE_DEVICE>(timeout_ticks, rank, mine, f,
                                          start ? "start" : "peer barrier", threadIdx.x);
  }
  __syncthreads();
}

// The grid on this device, then (kPeers) one exchange with the peers by the last block
// to arrive, then the grid released. ONE FENCE PER BLOCK, by thread 0 after the block
// barrier: every wave first waits for its own stores (`wait_stores`), and a release
// writes back the whole L2, so one covers the block where one per thread wrote it back
// 512 times.
template <int WORLD, bool PEERS>
DINLINE void grid_barrier(const PeerSignals& peer_signals, Signal* self_signal, int rank,
                     uint64_t timeout_ticks) {
  constexpr int kScope = PEERS ? __MEMORY_SCOPE_SYSTEM : __MEMORY_SCOPE_DEVICE;
  skew(rank, 0);
  wait_stores();
  __syncthreads();
  if (threadIdx.x == 0) {
    fence<__ATOMIC_RELEASE, kScope>();
    const Signals own = own_signals(self_signal);
    const uint32_t g  = own.gen().load<__ATOMIC_ACQUIRE, kScope>();
    if (own.arrive().fetch_add<__ATOMIC_ACQ_REL, __MEMORY_SCOPE_DEVICE>(1u) == gridDim.x - 1) {
      own.arrive().store<__ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE>(0u);
      if constexpr (PEERS) {
        const uint32_t e = own.epoch() + 1;
        own.set_epoch(e);
        fence<__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM>();
#pragma unroll
        for (int i = 0; i < WORLD; ++i)
          lane_signals(peer_signals, i)
              .peer(rank)
              .store<__ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM>(e);
#pragma unroll
        for (int i = 0; i < WORLD; ++i)
          wait<true, __MEMORY_SCOPE_SYSTEM>(timeout_ticks, rank, own.peer(i), e,
                                            "world_barrier: peer", i);
      }
      own.gen().fetch_add<__ATOMIC_RELEASE, kScope>(1u);
    } else {
      wait<true, kScope>(timeout_ticks, rank, own.gen(), g + 1,
                         PEERS ? "world_barrier" : "grid_barrier", -1);
    }
  }
  __syncthreads();
}

}  // namespace impl

// =================================================================================
// HOW ONE GPU SYNCHRONIZES WITH ANOTHER (listed in interface.cuh): the barriers and flags, written
// against a rank's Signals (buffers.cuh); a rank's data buffers are ordered only by `barrier`.
// =================================================================================

// WHO A BARRIER WAITS FOR: this block and the same block on every rank (peers), every block of
// this rank (grid), or every block of every rank (world: the grid, then one exchange among the
// ranks, so after it any block may read what any block on any rank wrote before it).
enum class Group { peers, grid, world };
// WHAT HOLDS ONCE IT IS PASSED: every peer has launched (so its input is ready to read); what this
// side wrote before is visible to the other side after; every peer is done reading this rank (so
// its buffers may be reused). Among the grid, only `visible` means anything.
enum class Until { launched, visible, read };

// A KERNEL'S SYNCHRONIZATION, one per kernel: every rank's signal block, this rank's, its rank and
// how long a wait may last before it traps, and this block's sequence number, read once here and
// carried in a register. Each peers barrier advances it; finish() stores it for the next call, the
// kernel's last statement; barrier<Group, Until>(sync) below. Stored by every barrier, the store sat before the kernel's first loads
// and shared a register with their addresses: the loads waited for it (vmcnt(0), +0.18 us at
// 1 x 7168, 2026-10-04T17-55-54Z).
// A read after a peers barrier may see only what the SAME BLOCK on the peer wrote before it, so
// both sides must index the same data by the same block.
template <int WORLD>
class Sync;
namespace impl {
template <Group GROUP, Until UNTIL, int WORLD>
DINLINE void barrier(Sync<WORLD>& sync);
}  // namespace impl

template <int WORLD>
class Sync {
  const PeerSignals& peer_signals_;
  Signal* self_signal_;
  int rank_;
  uint64_t timeout_ticks_;
  uint32_t seq_;

  template <Group GROUP, Until UNTIL, int W>
  friend DINLINE void impl::barrier(Sync<W>& sync);

 public:
  DINLINE Sync(const PeerSignals& peer_signals, Signal* self_signal, int rank,
               uint64_t timeout_ticks)
      : peer_signals_(peer_signals),
        self_signal_(self_signal),
        rank_(rank),
        timeout_ticks_(timeout_ticks),
        seq_(impl::own_signals(self_signal).seq(blockIdx.x)) {}

  // A FLAG TO ONE PEER, with the barriers' own store and spin: `v` lands in `peer`'s signal block,
  // in its slot for this rank, and wait_flag spins until `peer`'s flag here reaches `v`. Flags
  // only grow, so a caller counts on from the last value it used.
  DINLINE void write_flag(int peer, uint32_t v) const {
    impl::signals(peer_signals_, peer).flag(rank_).template store<__ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM>(v);
  }
  DINLINE void wait_flag(int peer, uint32_t v) const {
    impl::wait<false, __MEMORY_SCOPE_DEVICE>(timeout_ticks_, rank_,
                                             impl::own_signals(self_signal_).flag(peer), v, "flag", peer);
  }

  // THE SEQUENCE FOR THE NEXT CALL, once, after every barrier: a kernel that does not call it
  // pairs its next call's blocks against stale numbers and hangs (lint_kernels.sh checks).
  DINLINE void finish() const {
    if (threadIdx.x == 0) impl::own_signals(self_signal_).set_seq(blockIdx.x, seq_);
  }
};

// THE BARRIER, one for every case: barrier<Group::peers, Until::read>(sync). A namespace function,
// as std::get is, so a kernel templated on WORLD calls it without `sync.template barrier<...>`.
namespace impl {
template <Group GROUP, Until UNTIL, int WORLD>
DINLINE void barrier(Sync<WORLD>& sync) {
  static_assert(GROUP == Group::peers || UNTIL == Until::visible,
                "a grid or world barrier is a visibility barrier");
  if constexpr (GROUP == Group::grid) {
    impl::grid_barrier<WORLD, false>(sync.peer_signals_, sync.self_signal_, sync.rank_,
                                sync.timeout_ticks_);
  } else if constexpr (GROUP == Group::world) {
    impl::grid_barrier<WORLD, true>(sync.peer_signals_, sync.self_signal_, sync.rank_,
                               sync.timeout_ticks_);
  } else {
    impl::skew(sync.rank_, sync.seq_);
    ++sync.seq_;
    impl::pair_blocks<WORLD, UNTIL == Until::visible>(sync.peer_signals_, sync.self_signal_,
                                                      sync.rank_, sync.timeout_ticks_,
                                                      UNTIL == Until::launched, sync.seq_);
  }
}
}  // namespace impl

}  // namespace hip_comms
