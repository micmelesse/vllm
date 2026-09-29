// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// HOW EACH CALL RUNS: which kernel, and how wide, decided from the hardware (hardware.cuh) and the
// input. Each kernel on the critical path has its own function stating its rule; the rest run a
// basic config until each is measured. launch.cuh says what kernels exist; this picks one.

#pragma once

#include <algorithm>
#include <climits>
#include <cstdint>

#include "hardware.cuh"
#include "launch.cuh"
#include "p2p/p2p.cuh"

namespace hip_comms {
namespace tune {

// =================================================================================================
// THE CRITICAL PATH: the plain all-reduce's pull kernels, from the launch-config sweep on n11
// (bench, 2026-09-29T19-24-48Z: tokens 1-64 at hidden 3584 bf16, blocks 1-64, threads 64-512).
// =================================================================================================

// ONE WAVE PER BLOCK, AS MANY BLOCKS AS THE SIGNAL SLOTS ALLOW. Waves in one block only add a
// barrier inside it: at 1 token one-shot took 7.8 us at 64 threads, 8.1 at 128, 9.0 at 256 and
// 10.9 at 512. Blocks run apart, and from 16 tokens up more of them help, best at the slot cap.
// This is within 0.1 us of the sweep's fastest config at every size from 1 to 64 tokens.
constexpr Launch pull(Kernel k, const Hardware& hw) {
  return {k, std::min(p2p::kMaxBlocks, hw.compute_units), hw.wave_size, 0, 16};
}

constexpr Launch pull_one_shot(const Hardware& hw) { return pull(Kernel::pull_one_shot, hw); }
constexpr Launch pull_two_shot(const Hardware& hw) { return pull(Kernel::pull_two_shot, hw); }

// ONE-SHOT UP TO HERE, TWO-SHOT PAST IT. One-shot reads every peer's whole buffer ((N-1)P) in one
// round trip; two-shot moves less (2(N-1)/N P) in two. Measured: one-shot wins at 112 KiB (10.12
// vs 10.67 us), two-shot at 224 KiB (10.96 vs 13.13); the lines cross near 135 KiB. To be derived
// from the link's latency and bandwidth once hardware.cuh carries them.
constexpr int64_t kPullOneShotMaxBytes = 128 * 1024;

// =================================================================================================
// THE BASIC CONFIG: everything else (the push kernels and the fused ops), at the values the
// 2026-09-27 microbench chose, until each is swept.
// =================================================================================================

constexpr int kOps = 5;
static_assert(static_cast<int>(Op::rms_norm_gemm_add) + 1 == kOps, "a basic config per Op");

constexpr int64_t kNever  = INT64_MAX;
constexpr int64_t kAlways = -1;
constexpr int64_t kKiB    = 1024;

//   one_shot_max_bytes   one-shot at or under, two-shot above
//   fused_max_bytes      declined above, so the caller runs the unfused ops (kNever: never
//                        declined; kAlways: always)
//   blocks, threads      the grid and block; the GEMM tail wants one block per 16-column tile
//                        (56 at Kimi-K3)
//   gemm_lanes_per_col   the GEMM tail's lanes per column (a template instantiation)
//   push_one_shot,       each shot's direction: the push kernel over the pull one
//   push_two_shot
//   quant_bits           a push kernel's codec: 16 (T itself), 8 or 4; the lossy two stay
//                        off until the model's accuracy is checked with them
struct Basic {
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

// Indexed by Op. AttnRes past one-shot's range loses to unfused (2143 vs 1093 us at 4096 rows);
// the GEMM tail is declined until its rewrite measures faster than unfused (37 us at 16 rows).
constexpr Basic kBasic[kOps] = {
    /* all_reduce            */ {512 * kKiB, kNever, 16, 36, 512, 0, false, false, 16},
    /* rms_norm              */ {512 * kKiB, kNever, 16, 36, 512, 0, false, false, 16},
    /* add_rms_norm          */ {512 * kKiB, kNever, 16, 36, 512, 0, false, false, 16},
    /* add_attn_res_rms_norm */ {512 * kKiB, 512 * kKiB, 16, 36, 512, 0, false, false, 16},
    /* rms_norm_gemm_add     */ {512 * kKiB, kAlways, 56, 56, 512, 4, false, false, 16},
};

constexpr const Basic& basic(Op op) { return kBasic[static_cast<int>(op)]; }

}  // namespace tune

// =================================================================================================
// THE PICK: the plain pull all-reduce by its rules, everything else by its basic config.
// =================================================================================================

inline Launch pick(Op op, int64_t rows, int64_t bytes) {
  const tune::Basic& t = tune::basic(op);
  if (bytes > t.fused_max_bytes) return {Kernel::none, 0, 0, 0, 0};
  if (op == Op::all_reduce && !t.push_one_shot && !t.push_two_shot)
    return bytes <= tune::kPullOneShotMaxBytes ? tune::pull_one_shot(kTarget)
                                               : tune::pull_two_shot(kTarget);
  const bool one_shot = bytes <= t.one_shot_max_bytes &&
                        !(op == Op::rms_norm_gemm_add && rows > kGemmTailOneShotRows);
  const Kernel k = kernel_of(op, one_shot ? t.push_one_shot : t.push_two_shot, !one_shot);
  return {k, grid_of(k, one_shot ? t.one_shot_blocks : t.two_shot_blocks, rows), t.threads,
          t.gemm_lanes_per_col, t.quant_bits};
}

}  // namespace hip_comms
