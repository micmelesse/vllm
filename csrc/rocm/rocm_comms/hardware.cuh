// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE HARDWARE: each target's facts, as the device reports them (`rocminfo`). No decision lives
// here; tune.cuh makes them from these facts and the input. A new target is one more `Hardware`;
// `kTarget` is the one this build is for.

#pragma once

#include <cstdint>

namespace hip_comms {

struct Hardware {
  int wave_size;         // lanes per wave (rocminfo: Wavefront Size)
  int max_workgroup;     // threads per block the device allows (rocminfo: Workgroup Max Size)
  int compute_units;     // rocminfo: Compute Unit
  int max_waves_per_cu;  // rocminfo: Max Waves Per CU
};

// gfx950, AMD Instinct MI355X, as `rocminfo` reports it on n11 (2026-09-29, all 8 GPUs). gfx9 runs
// 64-lane waves only; 32-lane waves are RDNA's.
constexpr Hardware kGfx950 = {
    64,    // wave_size
    1024,  // max_workgroup
    256,   // compute_units
    32,    // max_waves_per_cu
};

// THE TARGET THIS BUILD IS FOR, and the sizes the device code compiles against.
constexpr const Hardware& kTarget = kGfx950;
constexpr int kWaveSize = kTarget.wave_size;

// OUR LIMIT, NOT THE HARDWARE'S: every kernel is built for at most this many threads per block
// (its __launch_bounds__), so a block holds at most kMaxWaves; the device would allow
// kTarget.max_workgroup.
constexpr int kMaxThreads = 512;
constexpr int kMaxWaves   = kMaxThreads / kWaveSize;

// OUR LIMIT TOO: how many packs of one row a thread holds in registers, packs threadIdx.x +
// k * blockDim.x for k < kMaxRowPacks. A row wider than kMaxRowPacks x blockDim is refused by
// the host (`admits`); a push kernel's group is the same kMaxRowPacks packs.
constexpr int kMaxRowPacks = 4;
static_assert(kMaxThreads <= kTarget.max_workgroup && kMaxThreads % kWaveSize == 0,
              "the kernels' block limit must be whole waves the device can launch");

// THE COMPILER'S WAVE SIZE AGREES with the target's, or the in-wave shuffles are wrong.
#if defined(__AMDGCN_WAVEFRONT_SIZE)
static_assert(__AMDGCN_WAVEFRONT_SIZE == kWaveSize,
              "hardware.cuh's wave size is not the compiler's");
#endif

}  // namespace hip_comms
