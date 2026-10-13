// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE RANKS' SYNCHRONIZATION: a rank's Signal block, whose only view (`Signals`) hands out its
// flags, each touched only atomically; `barrier`, and in `impl` what it is made of. A rank's buffers
// are peers.cuh's.
//
// NOTHING SURVIVES A LAUNCH: every flag is 0 between launches. A barrier sets 1 in each peer's slot
// for it, waits for 1 in each of its own, and sets those back to 0. A slot is never set twice before
// it is cleared, because the barriers of a launch alternate between the two flag sets (`parity`):
// a peer sets my parity-p slot again only at its barrier after next, which it passes only once I
// have signalled my next barrier, which I do only after my clear has landed. Every kernel has an
// even number of barriers (dev/lint_kernels.sh), so each launch starts at parity 0 on all-zero
// flags, and a graph replay is the same launch again.
//
// A WAIT NEVER HANGS AND NEVER TRAPS: past the timeout it writes the rank's fault record (where,
// which peer, the count against the one wanted) and leaves the kernel, and every wait leaves once
// the record's `abort` is set; the host reads the record (peers.cuh's `Fault`). HIP_CUSTOM_COMMUNICATOR_DEBUG=1
// skews every wait by a random per-block delay, so a race shows on every run.

#pragma once

#ifndef HIP_CUSTOM_COMMUNICATOR_COMMON_INTERFACE
#error "include common/interface.cuh, common's one interface, not its parts"
#endif

#include <hip/hip_runtime.h>
#include <array>
#include <cstdint>

#include "peers.cuh"
#include "utils.cuh"

namespace hip_custom_communicator {

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

// A RANK'S SIGNAL BLOCK: its flags, handed out one at a time. `flag(parity, block, writer)` is the
// slot that block `block` on rank `writer` sets in this rank's block at a barrier of that parity.
class Signals {
  Signal* s_;

 public:
  explicit DINLINE Signals(Signal* s) : s_(s) {}
  // AN INDEX KEEPS ITS CALLER'S TYPE (a block unsigned, as blockIdx.x; a rank as passed): an int
  // where the caller had an unsigned costs a sign extension in every kernel.
  template <typename RANK_TYPE>
  DINLINE Counter flag(uint32_t parity, unsigned block, RANK_TYPE writer) const {
    return Counter(&s_->flags[parity][block][writer]);
  }
};

// This rank's signal block, and thread i's peer i's, each thread its own (a per-lane load, as the pairing barrier issues it).
namespace impl {
DINLINE Signals own_signals(Signal* self_signal_ptr) { return Signals(self_signal_ptr); }

template <typename INDEX_TYPE>
DINLINE Signals lane_signals(Signal* const* signal_ptrs, INDEX_TYPE i) {
  return Signals(signal_ptrs[i]);
}
}  // namespace impl



#ifndef HIP_CUSTOM_COMMUNICATOR_DEBUG
#define HIP_CUSTOM_COMMUNICATOR_DEBUG 0
#endif

namespace impl {

// Debug builds only: up to ~32 x 8K cycles, different per rank, block and barrier (the launch's
// barriers so far).
DINLINE void skew(int rank, uint32_t barriers) {
  if (!HIP_CUSTOM_COMMUNICATOR_DEBUG) return;
  uint32_t h = static_cast<uint32_t>(rank) * 73856093u ^ blockIdx.x * 19349663u ^ barriers * 83492791u;
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
// enough between phases, where a block reads only what the same-numbered block wrote. kOrdered:
// the store releases and the wait acquires, so what the block put before is visible to its peers'
// same-numbered block after. Unordered, it only says when (first: every peer has launched; read:
// every peer is done reading us).
//
// Lane i sets its slot in peer i's block, waits for peer i's in this rank's, and clears it. The
// clear must land before the lane's NEXT flag goes out, or a peer that sees that flag could set
// the slot again before the clear does and lose it: an ordered barrier's wait_stores covers it; an
// unordered one waits for the lane's own stores just before it signals, after the whole phase
// between, so the clear has long landed.
template <int WORLD, bool ORDERED>
DINLINE void pair_blocks(Signal* const* signal_ptrs, Signal* self_signal_ptr, int rank,
                         uint64_t timeout_ticks, bool first, uint32_t parity) {
  if (!first) {
    if constexpr (ORDERED) wait_stores();
    __syncthreads();
  }
  if (threadIdx.x < WORLD) {
    if constexpr (!ORDERED) wait_stores();
    const Counter theirs = lane_signals(signal_ptrs, threadIdx.x).flag(parity, blockIdx.x, rank);
    const Counter mine   = own_signals(self_signal_ptr).flag(parity, blockIdx.x, threadIdx.x);
    theirs.store<ORDERED ? __ATOMIC_RELEASE : __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM>(1u);
    wait<ORDERED, __MEMORY_SCOPE_DEVICE>(timeout_ticks, self_signal_ptr, rank, mine, 1u,
                                          first ? Where::start : Where::peer_barrier,
                                          static_cast<int>(threadIdx.x));
    mine.store<__ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM>(0u);
  }
  __syncthreads();
}

}  // namespace impl

// =================================================================================
// HOW ONE GPU SYNCHRONIZES WITH ANOTHER (listed in interface.cuh): the barriers and flags, written
// against a rank's Signals (buffers.cuh); a rank's data buffers are ordered only by `barrier`.
// =================================================================================

// WHAT HOLDS ONCE A BARRIER IS PASSED: every peer has launched (so its input is ready to read; a
// launch's first barrier); what this side wrote before is visible to the other side after; every
// peer is done reading this rank (so its buffers may be reused). Every barrier pairs this block with
// the same block on every rank.
enum class Until { launched, visible, read };

// A KERNEL'S SYNCHRONIZATION, one per kernel: every rank's signal block, this rank's, its rank, how
// long a wait may last before it gives up, and the launch's barriers so far, which pick each
// barrier's flag set (its parity). Nothing in it outlives the launch.
// A read after a barrier may see only what the SAME BLOCK on the peer wrote before it, so both
// sides must index the same data by the same block.
template <int WORLD>
class Sync;
namespace impl {
template <Until UNTIL, int WORLD>
DINLINE void barrier(Sync<WORLD>& sync);
}  // namespace impl

template <int WORLD>
class Sync {
  Signal* const* signal_ptrs_;
  Signal* self_signal_ptr_;
  int rank_;
  uint64_t timeout_ticks_;
  uint32_t barriers_ = 0;

  template <Until UNTIL, int W>
  friend DINLINE void impl::barrier(Sync<W>& sync);

 public:
  DINLINE Sync(Signal* const* signal_ptrs, Signal* self_signal_ptr, int rank,
               uint64_t timeout_ticks)
      : signal_ptrs_(signal_ptrs),
        self_signal_ptr_(self_signal_ptr),
        rank_(rank),
        timeout_ticks_(timeout_ticks) {}
};

// THE BARRIER, one for every case: barrier<Until::read>(sync). A namespace function, as std::get
// is, so a kernel templated on WORLD calls it without `sync.template barrier<...>`.
namespace impl {
template <Until UNTIL, int WORLD>
DINLINE void barrier(Sync<WORLD>& sync) {
  impl::skew(sync.rank_, sync.barriers_);
  impl::pair_blocks<WORLD, UNTIL == Until::visible>(sync.signal_ptrs_, sync.self_signal_ptr_,
                                                    sync.rank_, sync.timeout_ticks_,
                                                    UNTIL == Until::launched, sync.barriers_ & 1u);
  ++sync.barriers_;
}
}  // namespace impl

}  // namespace hip_custom_communicator
