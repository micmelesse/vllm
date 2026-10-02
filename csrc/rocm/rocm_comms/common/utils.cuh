// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE HELPERS every op and kernel uses: the pack (16 bytes, the unit everything loads, sums and
// stores in) and the Fragment (a thread's packs of a row a block owns).

#pragma once

#ifndef HIP_COMMS_COMMON_INTERFACE
#error "include common/common.cuh, common's one interface, not its parts"
#endif

#include <hip/hip_runtime.h>

#include <cstdint>

#include "../build.cuh"

#define DINLINE __device__ __forceinline__

namespace hip_comms {

template <typename T, int N>
struct __align__(sizeof(T) * N) vec {
  T d[N];
};

// A PACK: kBuild.memory.pack_bytes (build.cuh: the widest load, 8 bf16), the unit every
// kernel loads, sums and stores in. `num_packs` counts them.

template <typename T>
struct traits {
  static constexpr int N = kBuild.memory.pack_bytes / sizeof(T);
  using V = vec<T, N>;
};

// WHO HOLDS WHAT: a Fragment (CUTLASS's, CuTe's and rocWMMA's name for a thread's registers of a
// distributed tile) is this thread's packs of a row a block owns. Its indices are clamped into the
// row and a pack past its end is weighted zero (`in`), so nothing reading it branches: a load under
// a runtime `if` cannot be hoisted past the branch, and loads meant to be in flight together then
// wait one at a time.
//
// Packs threadIdx.x + k * blockDim.x, k < K, of a row `len` packs long: `at` clamped into the row,
// `in` 1 for a pack inside it and 0 past its end.
template <int K>
struct Fragment {
  int at[K];
  float in[K];
};

template <int K>
DINLINE Fragment<K> fragment(int len) {
  Fragment<K> f;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    f.at[k]     = i < len ? i : len - 1;
    f.in[k]     = i < len ? 1.0f : 0.0f;
  }
  return f;
}

// EVERY SOURCE'S PACKS IN REGISTERS: a pack from each of `ngpus` sources (the ranks), K of them a
// source for a Fragment of a row. What peers_load issues and peers_reduce consumes, so a kernel
// can start one row's loads and work on another's while they are in flight.
template <typename T, int ngpus, int K = 1>
struct PeerPacks {
  typename traits<T>::V p[ngpus][K];
};

// PER-PHASE TIMESTAMPS, for finding where a kernel's time goes: thread 0 of each block records the
// device clock (100 MHz) at a phase boundary, after the block's threads have all reached it, into a
// device-global [block][phase] table the host reads back (rocm_comms_stamps). Built only with
// HIP_COMMS_STAMPS, since the barrier it adds would perturb a production kernel.
#ifndef HIP_COMMS_STAMPS
#define HIP_COMMS_STAMPS 0
#endif
constexpr int kStampBlocks = kMaxComputeUnits;
constexpr int kStampPhases = 8;
__device__ uint64_t g_stamps[kStampBlocks][kStampPhases];

DINLINE void block_stamp(int phase) {
  if constexpr (HIP_COMMS_STAMPS) {
    // Every outstanding load landed first, so a phase owns its own memory latency.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x < kStampBlocks) g_stamps[blockIdx.x][phase] = wall_clock64();
  }
}

}  // namespace hip_comms
