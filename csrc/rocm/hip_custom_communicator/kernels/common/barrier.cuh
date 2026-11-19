#pragma once

// barrier: every rank's block at the same point, by flags in each rank's
// signals.

#include "types.cuh"

namespace hip_custom_communicator {

enum class Until { launched, visible, read, started };

template <int WORLD>
// A barrier across ranks through flags in every peer's signal buffer.
DINLINE void barrier(uint32_t* const* signals_ptr_array,
                     int signals_ptr_array_size, int64_t signals_stride_parity,
                     int64_t signals_stride_block,
                     int64_t signals_stride_writer, int signals_size_parity,
                     int signals_size_block, int signals_size_writer,
                     uint32_t* status_ptr, int64_t status_stride_field,
                     int status_size_field, int rank, uint64_t timeout_ticks,
                     Until until, uint32_t parity) {
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG && parity > 1) internal::check_failed();
  const bool ordered = until == Until::launched || until == Until::visible;
  const bool first = until == Until::launched || until == Until::started;

  if (HIP_CUSTOM_COMMUNICATOR_DEBUG) {
    uint32_t h = static_cast<uint32_t>(rank) * 73856093u ^
                 blockIdx.x * 19349663u ^ parity * 83492791u;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    for (uint32_t n = (h ^ (h >> 15)) % 32; n > 0; --n)
      __builtin_amdgcn_s_sleep(127);
  }

  const auto flag = [&](uint32_t* owner_signals, auto owner, unsigned block,
                        auto writer) {
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
        (static_cast<int64_t>(owner) < 0 ||
         static_cast<int64_t>(owner) >= WORLD ||
         static_cast<int64_t>(parity) >= signals_size_parity ||
         static_cast<int64_t>(block) >= signals_size_block ||
         static_cast<int64_t>(writer) < 0 ||
         static_cast<int64_t>(writer) >= signals_size_writer))
      internal::check_failed();
    return owner_signals + parity * signals_stride_parity +
           block * signals_stride_block + writer * signals_stride_writer;
  };

  uint32_t* theirs_ptr = nullptr;
  uint32_t* mine_ptr = nullptr;
  if (threadIdx.x < WORLD) {
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG && WORLD > signals_ptr_array_size)
      internal::check_failed();

    const uint64_t entry = ((const __attribute__((address_space(1)))
                             uint64_t*)signals_ptr_array)[threadIdx.x];
    theirs_ptr =
        flag((uint32_t*)(__attribute__((address_space(1))) uint32_t*)entry,
             threadIdx.x, blockIdx.x, rank);
    mine_ptr = flag((uint32_t*)(__attribute__((address_space(1))) uint32_t*)
                        internal::ptr_array_at(signals_ptr_array,
                                               signals_ptr_array_size, rank),
                    rank, blockIdx.x, threadIdx.x);
  }

  if (ordered) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  if (threadIdx.x < WORLD) {
    if (!ordered) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

    __scoped_atomic_store_n(
        theirs_ptr, static_cast<uint32_t>(FlagState::arrived),
        ordered ? __ATOMIC_RELEASE : __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);

    const uint32_t want = static_cast<uint32_t>(FlagState::arrived);

    const auto word = [&](StatusField f) {
      if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
          (static_cast<int64_t>(f) < 0 ||
           static_cast<int64_t>(f) >= status_size_field))
        internal::check_failed();
      return status_ptr + static_cast<int64_t>(f) * status_stride_field;
    };
    const uint64_t t0 = wall_clock64();
    for (uint32_t n = 1; __scoped_atomic_load_n(mine_ptr, __ATOMIC_RELAXED,
                                                __MEMORY_SCOPE_SYSTEM) < want;
         ++n) {
      if ((n & 255u) != 0) continue;
      const uint64_t waited = wall_clock64() - t0;
      const bool aborted =
          __scoped_atomic_load_n(word(StatusField::abort), __ATOMIC_RELAXED,
                                 __MEMORY_SCOPE_SYSTEM) != 0;
      if (!aborted && waited <= timeout_ticks) continue;
      if (!aborted)
        internal::give_up(status_ptr, status_stride_field, status_size_field,
                          first ? Where::start : Where::peer_barrier, rank,
                          threadIdx.x,
                          __scoped_atomic_load_n(mine_ptr, __ATOMIC_RELAXED,
                                                 __MEMORY_SCOPE_SYSTEM),
                          want, waited);
      __builtin_amdgcn_endpgm();
    }

    if (ordered) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");

    __scoped_atomic_store_n(mine_ptr, static_cast<uint32_t>(FlagState::clear),
                            __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
  __syncthreads();
}

}  // namespace hip_custom_communicator
