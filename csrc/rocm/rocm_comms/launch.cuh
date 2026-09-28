// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// WHICH KERNEL RUNS, AND HOW WIDE. The caller names an op; this picks one kernel from the
// flat list and its launch geometry, for gfx950. Nothing above C++ sees an algorithm or a
// geometry.

#pragma once

#include <climits>
#include <cstdint>

#include "utils.cuh"

namespace hip_comms {

// What the caller asked for.
enum class Op : int {
  all_reduce            = 0,
  rms_norm              = 1,
  add_rms_norm          = 2,
  add_attn_res_rms_norm = 3,
  rms_norm_gemm_add     = 4,
};

// Every `__global__` there is, once. `none` is a decline: the caller runs the unfused ops.
enum class Kernel : int {
  none                           = -1,
  one_shot                       = 0,
  two_shot                       = 1,
  one_shot_rms_norm              = 2,
  two_shot_rms_norm              = 3,
  one_shot_add_rms_norm          = 4,
  two_shot_add_rms_norm          = 5,
  one_shot_add_attn_res_rms_norm = 6,
  two_shot_add_attn_res_rms_norm = 7,
  one_shot_rms_norm_gemm_add     = 8,
  two_shot_rms_norm_gemm_add     = 9,
  // all_reduce's quantized two-shot (allreduce_two_shot_quantized.cuh).
  two_shot_int8 = 10,
  two_shot_int4 = 11,
};

// What runs: the kernel, its grid and block, and for the GEMM tail its lanes per column (0
// for every other kernel).
struct Launch {
  Kernel kernel;
  int grid;
  int threads;
  int gemm_lanes_per_col;
};

// Each op has a one-shot and a two-shot kernel, numbered as a pair; the quantized two-shots
// come after, both all_reduce.
constexpr bool is_quantized(Kernel k) {
  return k == Kernel::two_shot_int8 || k == Kernel::two_shot_int4;
}
constexpr Kernel kernel_of(Op op, bool two_shot) {
  return static_cast<Kernel>(2 * static_cast<int>(op) + (two_shot ? 1 : 0));
}
constexpr Op op_of(Kernel k) {
  return is_quantized(k) ? Op::all_reduce : static_cast<Op>(static_cast<int>(k) / 2);
}
constexpr bool is_two_shot(Kernel k) {
  return is_quantized(k) || static_cast<int>(k) % 2 == 1;
}
constexpr Kernel quantized_kernel(int bits) {
  return bits == 8 ? Kernel::two_shot_int8 : Kernel::two_shot_int4;
}

// EVERY DECISION TUNED TO THE HARDWARE, per op, for gfx950 (8 x MI355X), from the
// microbench sweep of 2026-09-27 at Kimi-K3's shapes:
//   one_shot_max_bytes   one-shot at or under, two-shot above (12.9 vs 27.1 us at
//                        16 x 7168; two-shot wins by 3.67 MB)
//   fused_max_bytes      declined above, so the caller runs the unfused ops (kNever: never
//                        declined; kAlways: always)
//   blocks, threads      decode is flat in both; two-shot gains ~13% from 16 to 36 blocks;
//                        the GEMM tail wants one block per 16-column tile (56 at Kimi-K3)
//   gemm_lanes_per_col   the GEMM tail's lanes per column (a template instantiation)
//   quant_bits           the two-shot quantized to 8 or 4 bits (0: not); all_reduce only,
//                        off until the model's accuracy is checked with it
struct OpTuning {
  int64_t one_shot_max_bytes;
  int64_t fused_max_bytes;
  int one_shot_blocks;
  int two_shot_blocks;
  int threads;
  int gemm_lanes_per_col;
  int quant_bits;
};

constexpr int64_t kNever  = INT64_MAX;
constexpr int64_t kAlways = -1;
constexpr int64_t kKiB    = 1024;

// Indexed by Op. AttnRes past one-shot's range loses to unfused (2143 vs 1093 us at 4096
// rows); the GEMM tail is declined until its rewrite measures faster than unfused (37 us at
// 16 rows).
constexpr OpTuning kGfx950[] = {
    /* all_reduce            */ {512 * kKiB, kNever, 16, 36, 512, 0, 0},
    /* rms_norm              */ {512 * kKiB, kNever, 16, 36, 512, 0, 0},
    /* add_rms_norm          */ {512 * kKiB, kNever, 16, 36, 512, 0, 0},
    /* add_attn_res_rms_norm */ {512 * kKiB, 512 * kKiB, 16, 36, 512, 0, 0},
    /* rms_norm_gemm_add     */ {512 * kKiB, kAlways, 56, 56, 512, 4, 0},
};

constexpr const OpTuning& tuning(Op op) { return kGfx950[static_cast<int>(op)]; }

// A row op's one-shot kernel gives each block a row, so it needs no more blocks than rows;
// everything else strides over the whole grid.
constexpr int grid_of(Kernel k, int blocks, int64_t rows) {
  const Op op = op_of(k);
  const bool row_per_block =
      !is_two_shot(k) && op != Op::all_reduce && op != Op::rms_norm_gemm_add;
  return row_per_block && rows < blocks ? static_cast<int>(rows) : blocks;
}

inline Launch pick(Op op, int64_t rows, int64_t bytes) {
  const OpTuning& t = tuning(op);
  if (bytes > t.fused_max_bytes) return {Kernel::none, 0, 0, 0};
  const bool one_shot =
      bytes <= t.one_shot_max_bytes && !(op == Op::rms_norm_gemm_add && rows > kGemmRows);
  const Kernel k = !one_shot && t.quant_bits ? quantized_kernel(t.quant_bits)
                                              : kernel_of(op, !one_shot);
  return {k, grid_of(k, one_shot ? t.one_shot_blocks : t.two_shot_blocks, rows), t.threads,
          t.gemm_lanes_per_col};
}

}  // namespace hip_comms
