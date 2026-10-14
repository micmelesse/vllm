// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE RANKS' SYNCHRONIZATION, over the SIGNALS buffer, [R, 2, B, R]:
// `signals[owner][parity][block] [writer]` is the flag block `block` on rank
// `writer` sets in rank `owner`'s memory, at a barrier of that parity. The
// outer R is a table of every rank's address; the inner dimensions are the
// buffer's strides, passed like any other buffer's. Each flag is touched only
// atomically (`Counter`); `barrier`, and in `impl` what it is made of. A rank's
// data buffers are peers.cuh's.
//
// NOTHING SURVIVES A LAUNCH: every flag is 0 between launches. A barrier sets 1
// in each peer's slot for it, waits for 1 in each of its own, and sets those
// back to 0. A slot is never set twice before it is cleared, because the
// barriers of a launch alternate between the two flag sets (`parity`): a peer
// sets my parity-p slot again only at its barrier after next, which it passes
// only once I have signalled my next barrier, which I do only after my clear
// has landed. Every kernel has an even number of barriers
// (dev/lint_kernels.sh), so each launch starts at parity 0 on all-zero flags,
// and a graph replay is the same launch again.
//
// A WAIT NEVER HANGS AND NEVER TRAPS: past the timeout it writes the rank's
// fault record (where, which peer, the count against the one wanted) and leaves
// the kernel, and every wait leaves once the status's `abort` is set; the host
// reads the status (peers.cuh's `StatusField`). HIP_CUSTOM_COMMUNICATOR_DEBUG=1
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

// ONE FLAG: only atomic operations, each with its order and scope spelled at
// the call. A plain load or store of a flag cannot be written.
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
};

#ifndef HIP_CUSTOM_COMMUNICATOR_DEBUG
  #define HIP_CUSTOM_COMMUNICATOR_DEBUG 0
#endif

namespace impl {

// Debug builds only: up to ~32 x 8K cycles, different per rank, block and
// barrier (the launch's barriers so far).
DINLINE void skew(int rank, uint32_t barriers) {
  if (!HIP_CUSTOM_COMMUNICATOR_DEBUG) return;
  uint32_t h = static_cast<uint32_t>(rank) * 73856093u ^
               blockIdx.x * 19349663u ^ barriers * 83492791u;
  h ^= h >> 13;
  h *= 0x5bd1e995u;
  for (uint32_t n = (h ^ (h >> 15)) % 32; n > 0; --n)
    __builtin_amdgcn_s_sleep(127);
}

// EVERY WAVE'S STORES DONE before a block barrier that one wave then releases:
// `__syncthreads` waits for none (gfx9 emits no vmcnt wait before s_barrier;
// seen in the ISA), so the releasing wave's fence would cover only its own
// stores, and another wave's could still be in flight when the flag lands -- a
// remote store above all.
DINLINE void wait_stores() { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); }

// The builtin takes the ordering as a literal, so each case is spelled out.
template <int MEMORY_ORDER, int MEMORY_SCOPE>
DINLINE void fence() {
  constexpr bool system = MEMORY_SCOPE == __MEMORY_SCOPE_SYSTEM;
  if constexpr (MEMORY_ORDER == __ATOMIC_RELEASE) {
    if constexpr (system)
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    else
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  } else {
    if constexpr (system)
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    else
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  }
}

// One word of this rank's status, by its field, at the buffer's stride.
DINLINE uint32_t* status_word(uint32_t* status_ptr, int64_t status_stride_field,
                              StatusField f) {
  return status_ptr + static_cast<int64_t>(f) * status_stride_field;
}

// THE FAULT, written once by the first wait on this rank to give up: `state`
// claimed 0 -> 1, the fields, then 2 released, so the host reads them only
// whole. Then `abort`, so every other wait on this rank leaves at its next
// check rather than spinning out its own timeout.
DINLINE void record_fault(uint32_t* status_ptr, int64_t status_stride_field,
                          Where where, int rank, int peer, uint32_t count,
                          uint32_t want, uint64_t elapsed) {
  const auto word = [&](StatusField f) {
    return status_word(status_ptr, status_stride_field, f);
  };
  uint32_t none = static_cast<uint32_t>(FaultState::none);
  if (__scoped_atomic_compare_exchange_n(
          word(StatusField::state), &none,
          static_cast<uint32_t>(FaultState::writing), false, __ATOMIC_RELAXED,
          __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM)) {
    *word(StatusField::where) = static_cast<uint32_t>(where);
    *word(StatusField::rank) = static_cast<uint32_t>(rank);
    *word(StatusField::block) = blockIdx.x;
    *word(StatusField::peer) = static_cast<uint32_t>(peer);
    *word(StatusField::count) = count;
    *word(StatusField::want) = want;
    *word(StatusField::elapsed_low) = static_cast<uint32_t>(elapsed);
    *word(StatusField::elapsed_high) = static_cast<uint32_t>(elapsed >> 32);
    __scoped_atomic_store_n(word(StatusField::state),
                            static_cast<uint32_t>(FaultState::written),
                            __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
  }
  __scoped_atomic_store_n(word(StatusField::abort), 1u, __ATOMIC_RELAXED,
                          __MEMORY_SCOPE_SYSTEM);
}

// Spins relaxed and acquires once, after: an acquire per poll would invalidate
// the caches on every iteration of every spinning block. A sleep a poll (64
// cycles) keeps a spinning block from hammering the link. EVERY 256 POLLS, the
// slow path: the clock, and the status's `abort` (host-mapped, so only here,
// never in the fast path). Past the timeout the wait records the fault; timed
// out or aborted, the WAVE ENDS (s_endpgm): an ended wave no longer counts at
// the block's s_barrier, so the block's other waves run on to their own next
// wait, see `abort`, and end too. The kernel's output is then garbage and the
// communicator broken (the host's watchdog).
template <bool ACQUIRE, int MEMORY_SCOPE>
DINLINE void wait(uint64_t timeout_ticks, uint32_t* status_ptr,
                  int64_t status_stride_field, int rank, const Counter& flag,
                  uint32_t want, Where where, int peer) {
  const uint64_t t0 = wall_clock64();
  for (uint32_t n = 1; flag.load<__ATOMIC_RELAXED, MEMORY_SCOPE>() < want;
       ++n) {
    __builtin_amdgcn_s_sleep(1);
    if ((n & 255u) != 0) continue;
    const uint64_t waited = wall_clock64() - t0;
    const bool aborted =
        __scoped_atomic_load_n(
            status_word(status_ptr, status_stride_field, StatusField::abort),
            __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM) != 0;
    if (!aborted && waited <= timeout_ticks) continue;
    if (!aborted)
      record_fault(status_ptr, status_stride_field, where, rank, peer,
                   flag.load<__ATOMIC_RELAXED, MEMORY_SCOPE>(), want, waited);
    __builtin_amdgcn_endpgm();
  }
  if constexpr (ACQUIRE) fence<__ATOMIC_ACQUIRE, MEMORY_SCOPE>();
}

}  // namespace impl

// =================================================================================
// HOW ONE GPU SYNCHRONIZES WITH ANOTHER (listed in interface.cuh): the
// barriers, written against the signals buffer; a rank's data buffers are
// ordered only by `barrier`.
// =================================================================================

// WHAT HOLDS ONCE A BARRIER IS PASSED: every peer has launched (so its input is
// ready to read; a launch's first barrier); what this side wrote before is
// visible to the other side after; every peer is done reading this rank (so its
// buffers may be reused). Every barrier pairs this block with the same block on
// every rank.
enum class Until { launched, visible, read };

// A KERNEL'S SYNCHRONIZATION, one per kernel: the signals buffer (every rank's
// address and the strides), the status buffer, this rank, how long a wait may
// last before it gives up, and the launch's barriers so far, which pick each
// barrier's flag set (its parity). Nothing in it outlives the launch. A read
// after a barrier may see only what the SAME BLOCK on the peer wrote before it,
// so both sides must index the same data by the same block.
template <int WORLD>
class Sync;
namespace impl {
template <Until UNTIL, int WORLD>
DINLINE void barrier(Sync<WORLD>& sync);
}  // namespace impl

template <int WORLD>
class Sync {
  uint32_t* const* signals_ptrs_;
  int64_t signals_stride_parity_;
  int64_t signals_stride_block_;
  int64_t signals_stride_writer_;
  uint32_t* status_ptr_;
  int64_t status_stride_field_;
  int rank_;
  uint64_t timeout_ticks_;
  uint32_t barriers_ = 0;

  // Rank `owner`'s flag for block `block` on rank `writer`, at `parity`. AN
  // INDEX KEEPS ITS CALLER'S TYPE (a block unsigned, as blockIdx.x; a rank as
  // passed): an int where the caller had an unsigned costs a sign extension in
  // every kernel.
  template <typename OWNER_TYPE, typename WRITER_TYPE>
  DINLINE Counter flag(OWNER_TYPE owner, uint32_t parity, unsigned block,
                       WRITER_TYPE writer) const {
    return Counter(signals_ptrs_[owner] + parity * signals_stride_parity_ +
                   block * signals_stride_block_ +
                   writer * signals_stride_writer_);
  }

  template <Until UNTIL, int W>
  friend DINLINE void impl::barrier(Sync<W>& sync);

 public:
  DINLINE Sync(uint32_t* const* signals_ptrs, int64_t signals_stride_parity,
               int64_t signals_stride_block, int64_t signals_stride_writer,
               uint32_t* status_ptr, int64_t status_stride_field, int rank,
               uint64_t timeout_ticks)
      : signals_ptrs_(signals_ptrs),
        signals_stride_parity_(signals_stride_parity),
        signals_stride_block_(signals_stride_block),
        signals_stride_writer_(signals_stride_writer),
        status_ptr_(status_ptr),
        status_stride_field_(status_stride_field),
        rank_(rank),
        timeout_ticks_(timeout_ticks) {}
};

// THE BARRIER, one for every case: barrier<Until::read>(sync). A namespace
// function, as std::get is, so a kernel templated on WORLD calls it without
// `sync.template barrier<...>`.
//
// Block b waits for block b on every rank, and for no other block: enough at
// the ends, where it says "every peer has launched" or "every peer is done
// reading me", and not enough between phases, where a block reads only what the
// same-numbered block wrote. Ordered
// (`visible`): the store releases and the wait acquires, so what the block put
// before is visible to its peers' same-numbered block after. Unordered, it only
// says when (launched: every peer has launched; read: every peer is done
// reading us).
//
// Lane i sets its slot in peer i's signals, waits for peer i's in this rank's,
// and clears it. The clear must land before the lane's NEXT flag goes out, or a
// peer that sees that flag could set the slot again before the clear does and
// lose it: an ordered barrier's wait_stores covers it; an unordered one waits
// for the lane's own stores just before it signals, after the whole phase
// between, so the clear has long landed.
namespace impl {
template <Until UNTIL, int WORLD>
DINLINE void barrier(Sync<WORLD>& sync) {
  constexpr bool kOrdered = UNTIL == Until::visible;
  constexpr bool kFirst = UNTIL == Until::launched;
  const uint32_t parity = sync.barriers_ & 1u;
  impl::skew(sync.rank_, sync.barriers_);
  if constexpr (!kFirst) {
    if constexpr (kOrdered) wait_stores();
    __syncthreads();
  }
  if (threadIdx.x < WORLD) {
    if constexpr (!kOrdered) wait_stores();
    const Counter theirs =
        sync.flag(threadIdx.x, parity, blockIdx.x, sync.rank_);
    const Counter mine = sync.flag(sync.rank_, parity, blockIdx.x, threadIdx.x);
    theirs.template store<kOrdered ? __ATOMIC_RELEASE : __ATOMIC_RELAXED,
                          __MEMORY_SCOPE_SYSTEM>(
        static_cast<uint32_t>(FlagState::arrived));
    wait<kOrdered, __MEMORY_SCOPE_DEVICE>(
        sync.timeout_ticks_, sync.status_ptr_, sync.status_stride_field_,
        sync.rank_, mine, static_cast<uint32_t>(FlagState::arrived),
        kFirst ? Where::start : Where::peer_barrier,
        static_cast<int>(threadIdx.x));
    mine.template store<__ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM>(
        static_cast<uint32_t>(FlagState::clear));
  }
  __syncthreads();
  ++sync.barriers_;
}
}  // namespace impl

}  // namespace hip_custom_communicator
