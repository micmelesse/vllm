// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Moving and synchronising: the shared signal layout, peer pointers, and the barrier.

#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace hip_comms {

constexpr int kMaxRanks  = 8;
constexpr int kMaxBlocks = 36;

// ---------------------------------------------------------------------------------
// Shared layout. One IPC allocation per rank holds the signal block AND the scratch the
// two-stage algorithm needs, so adding that algorithm changes no layout: scratch is
// simply the bytes after the struct.
// ---------------------------------------------------------------------------------

// TWO counter arrays, not one. A peer block can reach the second barrier while this one
// is still at the first, and with a single array the peer would write counter+1 while we
// busy-wait on counter. `seq` is the per-block monotonic sequence number.
struct Signal {
  alignas(128) uint32_t start[kMaxBlocks][kMaxRanks];
  alignas(128) uint32_t end[kMaxBlocks][kMaxRanks];
  alignas(128) uint32_t seq[kMaxBlocks];
};

struct __align__(16) PeerPtrs { void* p[kMaxRanks]; };
struct __align__(16) PeerSignals { Signal* s[kMaxRanks]; };

__device__ __forceinline__ void* scratch_of(Signal* sig) { return sig + 1; }

#define DINLINE __device__ __forceinline__

// ---------------------------------------------------------------------------------
// Barrier. System scope to REACH a peer's memory, device scope to poll our own.
// ---------------------------------------------------------------------------------

template <int ngpus>
DINLINE void barrier_start(PeerSignals sigs, Signal* self, int rank) {
  uint32_t f = self->seq[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    __scoped_atomic_store_n(&sigs.s[threadIdx.x]->start[blockIdx.x][rank], f,
                            __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    while (__scoped_atomic_load_n(&self->start[blockIdx.x][threadIdx.x],
                                  __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) < f);
  }
  __syncthreads();
  if (threadIdx.x == 0) self->seq[blockIdx.x] = f;
}

// `final_sync` drops the release/acquire pair: nothing after the last barrier reads what
// this kernel wrote, so ordering costs without buying anything.
template <int ngpus, bool final_sync>
DINLINE void barrier_end(PeerSignals sigs, Signal* self, int rank) {
  __syncthreads();
  uint32_t f = self->seq[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    __scoped_atomic_store_n(&sigs.s[threadIdx.x]->end[blockIdx.x][rank], f,
                            final_sync ? __ATOMIC_RELAXED : __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
    while (__scoped_atomic_load_n(&self->end[blockIdx.x][threadIdx.x],
                                  final_sync ? __ATOMIC_RELAXED : __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_DEVICE) < f);
  }
  if constexpr (!final_sync) __syncthreads();
  if (threadIdx.x == 0) self->seq[blockIdx.x] = f;
}

}  // namespace hip_comms
