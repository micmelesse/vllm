// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE HARDWARE, in one place: each target's facts, as the device reports them (`rocminfo`), and
// every launch decision tuned for it. A new target is one more `Hardware`; `kTarget` is the one
// this build is for, and the rest of the code reads it.

#pragma once

#include <climits>
#include <cstdint>

namespace hip_comms {

// EVERY DECISION TUNED TO THE HARDWARE, per op, for gfx950 (8 x MI355X), from the
// microbench sweep of 2026-09-27 at Kimi-K3's shapes:
//   one_shot_max_bytes   one-shot at or under, two-shot above (12.9 vs 27.1 us at
//                        16 x 7168; two-shot wins by 3.67 MB)
//   fused_max_bytes      declined above, so the caller runs the unfused ops (kNever: never
//                        declined; kAlways: always)
//   blocks, threads      decode is flat in both; two-shot gains ~13% from 16 to 36 blocks;
//                        the GEMM tail wants one block per 16-column tile (56 at Kimi-K3)
//   gemm_lanes_per_col   the GEMM tail's lanes per column (a template instantiation)
//   push_one_shot,       each shot's direction: the push kernel over the pull one
//   push_two_shot
//   quant_bits           a push kernel's codec: 16 (T itself), 8 or 4; the lossy two stay
//                        off until the model's accuracy is checked with them
struct OpTuning {
  int64_t one_shot_max_bytes;
  int64_t fused_max_bytes;
  int one_shot_blocks;
  int two_shot_blocks;
  int threads;
  int gemm_lanes_per_col;
  bool push_one_shot;
  bool push_two_shot;
  int quant_bits;
};

// The number of ops the table is indexed by (launch.cuh `Op`, which asserts it).
constexpr int kOps = 5;

// A TARGET: its facts, and every launch decision tuned for it, per op.
struct Hardware {
  int wave_size;         // lanes per wave (rocminfo: Wavefront Size)
  int max_workgroup;     // threads per block the device allows (rocminfo: Workgroup Max Size)
  int compute_units;     // rocminfo: Compute Unit
  int max_waves_per_cu;  // rocminfo: Max Waves Per CU
  OpTuning ops[kOps];
};

constexpr int64_t kNever  = INT64_MAX;
constexpr int64_t kAlways = -1;
constexpr int64_t kKiB    = 1024;

// Indexed by Op. AttnRes past one-shot's range loses to unfused (2143 vs 1093 us at 4096
// rows); the GEMM tail is declined until its rewrite measures faster than unfused (37 us at
// 16 rows).
// gfx950, AMD Instinct MI355X, as `rocminfo` reports it on n11 (2026-09-29, all 8 GPUs). gfx9 runs
// 64-lane waves only; 32-lane waves are RDNA's.
constexpr Hardware kGfx950 = {
    64,    // wave_size
    1024,  // max_workgroup
    256,   // compute_units
    32,    // max_waves_per_cu
    {
    /* all_reduce            */ {512 * kKiB, kNever, 16, 36, 512, 0, false, false, 16},
    /* rms_norm              */ {512 * kKiB, kNever, 16, 36, 512, 0, false, false, 16},
    /* add_rms_norm          */ {512 * kKiB, kNever, 16, 36, 512, 0, false, false, 16},
    /* add_attn_res_rms_norm */ {512 * kKiB, 512 * kKiB, 16, 36, 512, 0, false, false, 16},
    /* rms_norm_gemm_add     */ {512 * kKiB, kAlways, 56, 56, 512, 4, false, false, 16},
    }};

// THE TARGET THIS BUILD IS FOR, and the sizes the device code compiles against.
constexpr const Hardware& kTarget = kGfx950;
constexpr int kWaveSize = kTarget.wave_size;

// OUR LIMIT, NOT THE HARDWARE'S: every kernel is built for at most this many threads per block
// (its __launch_bounds__), so a block holds at most kMaxWaves; the device would allow
// kTarget.max_workgroup.
constexpr int kMaxThreads = 512;
constexpr int kMaxWaves   = kMaxThreads / kWaveSize;
static_assert(kMaxThreads <= kTarget.max_workgroup && kMaxThreads % kWaveSize == 0,
              "the kernels' block limit must be whole waves the device can launch");

// THE COMPILER'S WAVE SIZE AGREES with the target's, or the in-wave shuffles are wrong.
#if defined(__AMDGCN_WAVEFRONT_SIZE)
static_assert(__AMDGCN_WAVEFRONT_SIZE == kWaveSize, "hardware.cuh's wave size is not the compiler's");
#endif

}  // namespace hip_comms
