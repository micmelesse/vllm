// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// WHICH KERNEL RUNS, AND HOW WIDE. The caller names an op; this picks one kernel from the
// flat list and describes its launch geometry; tune.cuh picks one for a call. Nothing above C++
// sees an algorithm or a geometry.

#pragma once

#include <climits>
#include <cstdint>

#include "fusions/rms_norm_gemm_add.cuh"
#include "utils.cuh"

namespace hip_comms {

// What the caller asked for: an all-reduce, alone or with what it fuses. Named as the Python
// methods are (`comm.all_reduce_rms_norm(...)`).
enum class Op : int {
  all_reduce                       = 0,
  all_reduce_rms_norm              = 1,
  all_reduce_add_rms_norm          = 2,
  all_reduce_add_attn_res_rms_norm = 3,
  all_reduce_rms_norm_gemm_add     = 4,
};

// Every `__global__` there is, once, named by its direction, its shot and what it fuses.
// `none` is a decline: the caller runs the unfused ops.
//   PULL: a rank reads its peers' buffers (inputs, or scratch they filled)
//   PUSH: a rank writes into its peers' inboxes and reads only its own memory; the push
//         kernels take a Codec, so they are the quantized ones
// Each op has all four: one- and two-shot, pull and push.
enum class Kernel : int {
  none                                           = -1,
  all_reduce_pull_one_shot                       = 0,
  all_reduce_push_one_shot                       = 1,
  all_reduce_pull_two_shot                       = 2,
  all_reduce_push_two_shot                       = 3,
  all_reduce_pull_one_shot_rms_norm              = 4,
  all_reduce_push_one_shot_rms_norm              = 5,
  all_reduce_pull_two_shot_rms_norm              = 6,
  all_reduce_push_two_shot_rms_norm              = 7,
  all_reduce_pull_one_shot_add_rms_norm          = 8,
  all_reduce_push_one_shot_add_rms_norm          = 9,
  all_reduce_pull_two_shot_add_rms_norm          = 10,
  all_reduce_push_two_shot_add_rms_norm          = 11,
  all_reduce_pull_one_shot_add_attn_res_rms_norm = 12,
  all_reduce_push_one_shot_add_attn_res_rms_norm = 13,
  all_reduce_pull_two_shot_add_attn_res_rms_norm = 14,
  all_reduce_push_two_shot_add_attn_res_rms_norm = 15,
  all_reduce_pull_one_shot_rms_norm_gemm_add     = 16,
  all_reduce_push_one_shot_rms_norm_gemm_add     = 17,
  all_reduce_pull_two_shot_rms_norm_gemm_add     = 18,
  all_reduce_push_two_shot_rms_norm_gemm_add     = 19,
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
    {Kernel::all_reduce_pull_one_shot, Op::all_reduce, false, false},
    {Kernel::all_reduce_push_one_shot, Op::all_reduce, true, false},
    {Kernel::all_reduce_pull_two_shot, Op::all_reduce, false, true},
    {Kernel::all_reduce_push_two_shot, Op::all_reduce, true, true},
    {Kernel::all_reduce_pull_one_shot_rms_norm, Op::all_reduce_rms_norm, false, false},
    {Kernel::all_reduce_push_one_shot_rms_norm, Op::all_reduce_rms_norm, true, false},
    {Kernel::all_reduce_pull_two_shot_rms_norm, Op::all_reduce_rms_norm, false, true},
    {Kernel::all_reduce_push_two_shot_rms_norm, Op::all_reduce_rms_norm, true, true},
    {Kernel::all_reduce_pull_one_shot_add_rms_norm, Op::all_reduce_add_rms_norm, false, false},
    {Kernel::all_reduce_push_one_shot_add_rms_norm, Op::all_reduce_add_rms_norm, true, false},
    {Kernel::all_reduce_pull_two_shot_add_rms_norm, Op::all_reduce_add_rms_norm, false, true},
    {Kernel::all_reduce_push_two_shot_add_rms_norm, Op::all_reduce_add_rms_norm, true, true},
    {Kernel::all_reduce_pull_one_shot_add_attn_res_rms_norm,
     Op::all_reduce_add_attn_res_rms_norm, false, false},
    {Kernel::all_reduce_push_one_shot_add_attn_res_rms_norm,
     Op::all_reduce_add_attn_res_rms_norm, true, false},
    {Kernel::all_reduce_pull_two_shot_add_attn_res_rms_norm,
     Op::all_reduce_add_attn_res_rms_norm, false, true},
    {Kernel::all_reduce_push_two_shot_add_attn_res_rms_norm,
     Op::all_reduce_add_attn_res_rms_norm, true, true},
    {Kernel::all_reduce_pull_one_shot_rms_norm_gemm_add,
     Op::all_reduce_rms_norm_gemm_add, false, false},
    {Kernel::all_reduce_push_one_shot_rms_norm_gemm_add,
     Op::all_reduce_rms_norm_gemm_add, true, false},
    {Kernel::all_reduce_pull_two_shot_rms_norm_gemm_add,
     Op::all_reduce_rms_norm_gemm_add, false, true},
    {Kernel::all_reduce_push_two_shot_rms_norm_gemm_add,
     Op::all_reduce_rms_norm_gemm_add, true, true},
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

// A row op's one-shot kernel gives each block a row, so it needs no more blocks than rows;
// everything else strides over the whole grid.
constexpr int grid_of(Kernel k, int blocks, int64_t rows) {
  const Op op = op_of(k);
  const bool row_per_block =
      !is_two_shot(k) && op != Op::all_reduce && op != Op::all_reduce_rms_norm_gemm_add;
  return row_per_block && rows < blocks ? static_cast<int>(rows) : blocks;
}

}  // namespace hip_comms
