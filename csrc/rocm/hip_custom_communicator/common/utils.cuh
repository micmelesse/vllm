// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE HELPERS every op and kernel uses: the pack (16 bytes, the unit everything loads, sums and
// stores in), and every source's packs in registers.

#pragma once

#ifndef HIP_CUSTOM_COMMUNICATOR_COMMON_INTERFACE
#error "include common/interface.cuh, common's one interface, not its parts"
#endif

#include <hip/hip_runtime.h>

#include <cstdint>

#include "build.cuh"

#define DINLINE __device__ __forceinline__

// THE TARGETS OUR KERNELS ARE WRITTEN FOR: MI300 (gfx942) and MI355 (gfx950). vLLM's ROCm build
// compiles every file for more (gfx90a and the Radeon gfx11xx/gfx12xx), where our kernels use what
// does not exist (DPP row broadcasts, gfx9's cache bits), so on those every kernel's body is left out
// and traps. Nothing launches one there: `supported` refuses the device first. The host pass, which
// generates no device code, sees the whole body.
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx942__) || defined(__gfx950__)
#define HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED 1
#else
#define HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED 0
#endif

namespace hip_custom_communicator {

template <typename DTYPE, int VEC_SIZE>
struct __align__(sizeof(DTYPE) * VEC_SIZE) vec {
  DTYPE d[VEC_SIZE];
};

// A PACK: kBuild.memory.pack_bytes (common/build.cuh: the widest load, 8 bf16), the unit every
// kernel loads, sums and stores in. `num_packs` counts them.

template <typename DTYPE>
struct traits {
  static constexpr int N = kBuild.memory.pack_bytes / sizeof(DTYPE);
  using V = vec<DTYPE, N>;
};


// PER-PHASE TIMESTAMPS, for finding where a kernel's time goes: thread 0 of each block records the
// device clock (100 MHz) at a phase boundary, after the block's threads have all reached it, into a
// device-global [block][phase] table the host reads back (hip_custom_communicator_stamps). Built only with
// HIP_CUSTOM_COMMUNICATOR_STAMPS, since the barrier it adds would perturb a production kernel.
#ifndef HIP_CUSTOM_COMMUNICATOR_STAMPS
#define HIP_CUSTOM_COMMUNICATOR_STAMPS 0
#endif
constexpr int kStampBlocks = kMaxComputeUnits;
constexpr int kStampPhases = 8;
__device__ uint64_t g_stamps[kStampBlocks][kStampPhases];

namespace impl {
DINLINE void block_stamp(int phase) {
  if constexpr (HIP_CUSTOM_COMMUNICATOR_STAMPS) {
    // Every outstanding load landed first, so a phase owns its own memory latency.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x < kStampBlocks) g_stamps[blockIdx.x][phase] = wall_clock64();
  }
}
}  // namespace impl

}  // namespace hip_custom_communicator
