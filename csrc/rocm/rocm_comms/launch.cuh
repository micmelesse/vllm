// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// WHICH KERNEL RUNS, AND HOW WIDE. The caller names an op; this picks one kernel from the
// flat list and its launch geometry, for gfx950. Nothing above C++ sees an algorithm or a
// geometry.

#pragma once

#include <climits>
#include <cstdint>

#include "fusions/rms_norm_gemm_add.cuh"
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

// Every `__global__` there is, once, named by its shot, its direction and what it fuses.
// `none` is a decline: the caller runs the unfused ops.
//   PULL: a rank reads its peers' buffers (inputs, or scratch they filled)
//   PUSH: a rank writes into its peers' inboxes and reads only its own memory; the push
//         kernels take a Codec, so they are the quantized ones
// Each op has all four: one- and two-shot, pull and push.
enum class Kernel : int {
  none                                = -1,
  pull_one_shot                       = 0,
  push_one_shot                       = 1,
  pull_two_shot                       = 2,
  push_two_shot                       = 3,
  pull_one_shot_rms_norm              = 4,
  push_one_shot_rms_norm              = 5,
  pull_two_shot_rms_norm              = 6,
  push_two_shot_rms_norm              = 7,
  pull_one_shot_add_rms_norm          = 8,
  push_one_shot_add_rms_norm          = 9,
  pull_two_shot_add_rms_norm          = 10,
  push_two_shot_add_rms_norm          = 11,
  pull_one_shot_add_attn_res_rms_norm = 12,
  push_one_shot_add_attn_res_rms_norm = 13,
  pull_two_shot_add_attn_res_rms_norm = 14,
  push_two_shot_add_attn_res_rms_norm = 15,
  pull_one_shot_rms_norm_gemm_add     = 16,
  push_one_shot_rms_norm_gemm_add     = 17,
  pull_two_shot_rms_norm_gemm_add     = 18,
  push_two_shot_rms_norm_gemm_add     = 19,
};

// The GEMM tail's one-shot kernel takes one GEMM pass: at most this many rows.
constexpr int kGemmTailOneShotRows = fusions::rms_norm_gemm_add::kRows;

// What each kernel is, in Kernel's order.
struct KernelInfo {
  Kernel kernel;
  Op op;
  bool push;
  bool two_shot;
};

constexpr KernelInfo kKernels[] = {
    {Kernel::pull_one_shot, Op::all_reduce, false, false},
    {Kernel::push_one_shot, Op::all_reduce, true, false},
    {Kernel::pull_two_shot, Op::all_reduce, false, true},
    {Kernel::push_two_shot, Op::all_reduce, true, true},
    {Kernel::pull_one_shot_rms_norm, Op::rms_norm, false, false},
    {Kernel::push_one_shot_rms_norm, Op::rms_norm, true, false},
    {Kernel::pull_two_shot_rms_norm, Op::rms_norm, false, true},
    {Kernel::push_two_shot_rms_norm, Op::rms_norm, true, true},
    {Kernel::pull_one_shot_add_rms_norm, Op::add_rms_norm, false, false},
    {Kernel::push_one_shot_add_rms_norm, Op::add_rms_norm, true, false},
    {Kernel::pull_two_shot_add_rms_norm, Op::add_rms_norm, false, true},
    {Kernel::push_two_shot_add_rms_norm, Op::add_rms_norm, true, true},
    {Kernel::pull_one_shot_add_attn_res_rms_norm, Op::add_attn_res_rms_norm, false, false},
    {Kernel::push_one_shot_add_attn_res_rms_norm, Op::add_attn_res_rms_norm, true, false},
    {Kernel::pull_two_shot_add_attn_res_rms_norm, Op::add_attn_res_rms_norm, false, true},
    {Kernel::push_two_shot_add_attn_res_rms_norm, Op::add_attn_res_rms_norm, true, true},
    {Kernel::pull_one_shot_rms_norm_gemm_add, Op::rms_norm_gemm_add, false, false},
    {Kernel::push_one_shot_rms_norm_gemm_add, Op::rms_norm_gemm_add, true, false},
    {Kernel::pull_two_shot_rms_norm_gemm_add, Op::rms_norm_gemm_add, false, true},
    {Kernel::push_two_shot_rms_norm_gemm_add, Op::rms_norm_gemm_add, true, true},
};
constexpr int kNumKernels = sizeof(kKernels) / sizeof(KernelInfo);

constexpr bool kernels_in_order() {
  for (int i = 0; i < kNumKernels; ++i)
    if (static_cast<int>(kKernels[i].kernel) != i) return false;
  return true;
}
static_assert(kernels_in_order(), "kKernels must list every Kernel in its order");

constexpr const KernelInfo& info(Kernel k) { return kKernels[static_cast<int>(k)]; }
constexpr Op op_of(Kernel k) { return info(k).op; }
constexpr bool is_push(Kernel k) { return info(k).push; }
constexpr bool is_two_shot(Kernel k) { return info(k).two_shot; }

// The op's kernel of that direction and shot; none where it has none.
constexpr Kernel kernel_of(Op op, bool push, bool two_shot) {
  for (const KernelInfo& i : kKernels)
    if (i.op == op && i.two_shot == two_shot && i.push == push) return i.kernel;
  return Kernel::none;
}

// What runs: the kernel, its grid and block, for the GEMM tail its lanes per column (0 for
// every other kernel), and for a push kernel its codec's bits (16: T itself).
struct Launch {
  Kernel kernel;
  int grid;
  int threads;
  int gemm_lanes_per_col;
  int quant_bits;
};

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

constexpr int64_t kNever  = INT64_MAX;
constexpr int64_t kAlways = -1;
constexpr int64_t kKiB    = 1024;

// Indexed by Op. AttnRes past one-shot's range loses to unfused (2143 vs 1093 us at 4096
// rows); the GEMM tail is declined until its rewrite measures faster than unfused (37 us at
// 16 rows).
constexpr OpTuning kGfx950[] = {
    /* all_reduce            */ {512 * kKiB, kNever, 16, 36, 512, 0, false, false, 16},
    /* rms_norm              */ {512 * kKiB, kNever, 16, 36, 512, 0, false, false, 16},
    /* add_rms_norm          */ {512 * kKiB, kNever, 16, 36, 512, 0, false, false, 16},
    /* add_attn_res_rms_norm */ {512 * kKiB, 512 * kKiB, 16, 36, 512, 0, false, false, 16},
    /* rms_norm_gemm_add     */ {512 * kKiB, kAlways, 56, 56, 512, 4, false, false, 16},
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
  if (bytes > t.fused_max_bytes) return {Kernel::none, 0, 0, 0, 0};
  const bool one_shot = bytes <= t.one_shot_max_bytes &&
                        !(op == Op::rms_norm_gemm_add && rows > kGemmTailOneShotRows);
  const Kernel k = kernel_of(op, one_shot ? t.push_one_shot : t.push_two_shot, !one_shot);
  return {k, grid_of(k, one_shot ? t.one_shot_blocks : t.two_shot_blocks, rows), t.threads,
          t.gemm_lanes_per_col, t.quant_bits};
}

}  // namespace hip_comms
