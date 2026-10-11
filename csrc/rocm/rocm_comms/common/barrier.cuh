// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE RANKS' SYNCHRONIZATION: a rank's Signal block, whose only view (`Signals`) hands out its
// counters, each touched only atomically; `barrier`, the flags, and in `impl` what they are made
// of. A rank's buffers are peers.cuh's.
//
// A WAIT NEVER HANGS AND NEVER TRAPS: past the timeout it writes the rank's fault record (where,
// which peer, the count against the one wanted) and leaves the kernel, and every wait leaves once
// the record's `abort` is set; the host reads the record (peers.cuh's `Fault`). HIP_COMMS_DEBUG=1
// skews every wait by a random per-block delay, so a race shows on every run.

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

// This rank's signal block; rank r's, from the table of every rank's; and thread i's peer i's,
// each thread its own (a per-lane load, as the pairing barrier issues it).
namespace impl {
DINLINE Signals own_signals(Signal* self_signal_ptr) { return Signals(self_signal_ptr); }

DINLINE Signals signals(Signal* const* signal_ptrs, int r) {
  return Signals(signal_ptrs[r]);
}

template <typename INDEX_TYPE>
DINLINE Signals lane_signals(Signal* const* signal_ptrs, INDEX_TYPE i) {
  return Signals(signal_ptrs[i]);
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

// THE FAULT, written once by the first wait on this rank to give up: `state` claimed 0 -> 1, the
// fields, then 2 released, so the host reads them only whole. Then `abort`, so every other wait on
// this rank leaves at its next check rather than spinning out its own timeout.
DINLINE void record_fault(Fault* f, Where where, int rank, int peer, uint32_t count,
                          uint32_t want, uint64_t elapsed) {
  uint32_t clear = 0;
  if (__scoped_atomic_compare_exchange_n(&f->state, &clear, 1u, false, __ATOMIC_RELAXED,
                                         __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM)) {
    f->where         = static_cast<uint32_t>(where);
    f->rank          = rank;
    f->block         = static_cast<int32_t>(blockIdx.x);
    f->peer          = peer;
    f->count         = count;
    f->want          = want;
    f->elapsed_ticks = elapsed;
    __scoped_atomic_store_n(&f->state, 2u, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
  }
  __scoped_atomic_store_n(&f->abort, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}

// Spins relaxed and acquires once, after: an acquire per poll would invalidate the caches on every
// iteration of every spinning block. A sleep a poll (64 cycles) keeps a spinning block from
// hammering the link. EVERY 256 POLLS, the slow path: the clock, and the fault record's `abort`
// (host-mapped, so only here, never in the fast path). Past the timeout the wait records the fault;
// timed out or aborted, the WAVE ENDS (s_endpgm): an ended wave no longer counts at the block's
// s_barrier, so the block's other waves run on to their own next wait, see `abort`, and end too.
// The kernel's output is then garbage and the communicator broken (the host's watchdog).
template <bool ACQUIRE, int MEMORY_SCOPE>
DINLINE void wait(uint64_t timeout_ticks, Signal* self_signal_ptr, int rank, const Counter& flag,
                  uint32_t want, Where where, int peer) {
  const uint64_t t0 = wall_clock64();
  for (uint32_t n = 1; flag.load<__ATOMIC_RELAXED, MEMORY_SCOPE>() < want; ++n) {
    __builtin_amdgcn_s_sleep(1);
    if ((n & 255u) != 0) continue;
    Fault* f              = self_signal_ptr->fault;
    const uint64_t waited = wall_clock64() - t0;
    const bool aborted =
        __scoped_atomic_load_n(&f->abort, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM) != 0;
    if (!aborted && waited <= timeout_ticks) continue;
    if (!aborted)
      record_fault(f, where, rank, peer, flag.load<__ATOMIC_RELAXED, MEMORY_SCOPE>(), want,
                   waited);
    __builtin_amdgcn_endpgm();
  }
  if constexpr (ACQUIRE) fence<__ATOMIC_ACQUIRE, MEMORY_SCOPE>();
}

// Block b waits for block b on every rank, and for no other block: enough at the ends,
// where it says "every peer has launched" or "every peer is done reading me", and not
// enough between phases, which is what `world_barrier` is for. kOrdered: the store
// releases and the wait acquires, so what the block put before is visible to its peers'
// same-numbered block after (a peer_barrier). Unordered, it only says when
// (start: every peer has launched; close: every peer is done reading us).
template <int WORLD, bool ORDERED>
DINLINE void pair_blocks(Signal* const* signal_ptrs, Signal* self_signal_ptr, int rank,
                         uint64_t timeout_ticks, bool start, uint32_t f) {
  if (!start) {
    if constexpr (ORDERED) wait_stores();
    __syncthreads();
  }
  const Signals own = own_signals(self_signal_ptr);
  if (threadIdx.x < WORLD) {
    const Signals peer   = lane_signals(signal_ptrs, threadIdx.x);
    const Counter theirs = start ? peer.start(blockIdx.x, rank) : peer.end(blockIdx.x, rank);
    const Counter mine   = start ? own.start(blockIdx.x, threadIdx.x)
                                 : own.end(blockIdx.x, threadIdx.x);
    theirs.store<ORDERED ? __ATOMIC_RELEASE : __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM>(f);
    wait<ORDERED, __MEMORY_SCOPE_DEVICE>(timeout_ticks, self_signal_ptr, rank, mine, f,
                                          start ? Where::start : Where::peer_barrier,
                                          static_cast<int>(threadIdx.x));
  }
  __syncthreads();
}

// The grid on this device, then (kPeers) one exchange with the peers by the last block
// to arrive, then the grid released. ONE FENCE PER BLOCK, by thread 0 after the block
// barrier: every wave first waits for its own stores (`wait_stores`), and a release
// writes back the whole L2, so one covers the block where one per thread wrote it back
// 512 times.
template <int WORLD, bool PEERS>
DINLINE void grid_barrier(Signal* const* signal_ptrs, Signal* self_signal_ptr, int rank,
                     uint64_t timeout_ticks) {
  constexpr int kScope = PEERS ? __MEMORY_SCOPE_SYSTEM : __MEMORY_SCOPE_DEVICE;
  skew(rank, 0);
  wait_stores();
  __syncthreads();
  if (threadIdx.x == 0) {
    fence<__ATOMIC_RELEASE, kScope>();
    const Signals own = own_signals(self_signal_ptr);
    const uint32_t g  = own.gen().load<__ATOMIC_ACQUIRE, kScope>();
    if (own.arrive().fetch_add<__ATOMIC_ACQ_REL, __MEMORY_SCOPE_DEVICE>(1u) == gridDim.x - 1) {
      own.arrive().store<__ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE>(0u);
      if constexpr (PEERS) {
        const uint32_t e = own.epoch() + 1;
        own.set_epoch(e);
        fence<__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM>();
#pragma unroll
        for (int i = 0; i < WORLD; ++i)
          lane_signals(signal_ptrs, i)
              .peer(rank)
              .store<__ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM>(e);
#pragma unroll
        for (int i = 0; i < WORLD; ++i)
          wait<true, __MEMORY_SCOPE_SYSTEM>(timeout_ticks, self_signal_ptr, rank, own.peer(i), e,
                                            Where::world_peer, i);
      }
      own.gen().fetch_add<__ATOMIC_RELEASE, kScope>(1u);
    } else {
      wait<true, kScope>(timeout_ticks, self_signal_ptr, rank, own.gen(), g + 1,
                         PEERS ? Where::world : Where::grid, -1);
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
// how long a wait may last before it gives up, and this block's sequence number, read once here and
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
  Signal* const* signal_ptrs_;
  Signal* self_signal_ptr_;
  int rank_;
  uint64_t timeout_ticks_;
  uint32_t seq_;

  template <Group GROUP, Until UNTIL, int W>
  friend DINLINE void impl::barrier(Sync<W>& sync);

 public:
  DINLINE Sync(Signal* const* signal_ptrs, Signal* self_signal_ptr, int rank,
               uint64_t timeout_ticks)
      : signal_ptrs_(signal_ptrs),
        self_signal_ptr_(self_signal_ptr),
        rank_(rank),
        timeout_ticks_(timeout_ticks),
        seq_(impl::own_signals(self_signal_ptr).seq(blockIdx.x)) {}

  // A FLAG TO ONE PEER, with the barriers' own store and spin: `v` lands in `peer`'s signal block,
  // in its slot for this rank, and wait_flag spins until `peer`'s flag here reaches `v`. Flags
  // only grow, so a caller counts on from the last value it used.
  DINLINE void write_flag(int peer, uint32_t v) const {
    impl::signals(signal_ptrs_, peer).flag(rank_).template store<__ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM>(v);
  }
  DINLINE void wait_flag(int peer, uint32_t v) const {
    impl::wait<false, __MEMORY_SCOPE_DEVICE>(timeout_ticks_, self_signal_ptr_, rank_,
                                             impl::own_signals(self_signal_ptr_).flag(peer), v,
                                             Where::flag, peer);
  }

  // THE SEQUENCE FOR THE NEXT CALL, once, after every barrier: a kernel that does not call it
  // pairs its next call's blocks against stale numbers and hangs (lint_kernels.sh checks).
  DINLINE void finish() const {
    if (threadIdx.x == 0) impl::own_signals(self_signal_ptr_).set_seq(blockIdx.x, seq_);
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
    impl::grid_barrier<WORLD, false>(sync.signal_ptrs_, sync.self_signal_ptr_, sync.rank_,
                                sync.timeout_ticks_);
  } else if constexpr (GROUP == Group::world) {
    impl::grid_barrier<WORLD, true>(sync.signal_ptrs_, sync.self_signal_ptr_, sync.rank_,
                               sync.timeout_ticks_);
  } else {
    impl::skew(sync.rank_, sync.seq_);
    ++sync.seq_;
    impl::pair_blocks<WORLD, UNTIL == Until::visible>(sync.signal_ptrs_, sync.self_signal_ptr_,
                                                      sync.rank_, sync.timeout_ticks_,
                                                      UNTIL == Until::launched, sync.seq_);
  }
}
}  // namespace impl

}  // namespace hip_comms
