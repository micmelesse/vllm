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

// Every `__global__` there is, once, named by its shot and what it fuses; all of them pull (a rank
// reads its peers' buffers: their inputs, or scratch they filled). `none` is a decline: the caller
// runs the unfused ops. Each op has a one-shot and a two-shot.
enum class Kernel : int {
  none                                           = -1,
  all_reduce_pull_one_shot                       = 0,
  all_reduce_pull_two_shot                       = 1,
  all_reduce_pull_one_shot_rms_norm              = 2,
  all_reduce_pull_two_shot_rms_norm              = 3,
  all_reduce_pull_one_shot_add_rms_norm          = 4,
  all_reduce_pull_two_shot_add_rms_norm          = 5,
  all_reduce_pull_one_shot_add_attn_res_rms_norm = 6,
  all_reduce_pull_two_shot_add_attn_res_rms_norm = 7,
  all_reduce_pull_one_shot_rms_norm_gemm_add     = 8,
  all_reduce_pull_two_shot_rms_norm_gemm_add     = 9,
};

// The GEMM tail's one-shot kernel takes one GEMM pass: at most this many rows.
constexpr int kGemmTailOneShotRows = fusions::rms_norm_gemm_add::kRows;

// What each kernel is, in Kernel's order.
struct KernelInfo {
  Kernel kernel;
  Op op;
  bool two_shot;
};

constexpr KernelInfo kKernels[] = {
    {Kernel::all_reduce_pull_one_shot, Op::all_reduce, false},
    {Kernel::all_reduce_pull_two_shot, Op::all_reduce, true},
    {Kernel::all_reduce_pull_one_shot_rms_norm, Op::all_reduce_rms_norm, false},
    {Kernel::all_reduce_pull_two_shot_rms_norm, Op::all_reduce_rms_norm, true},
    {Kernel::all_reduce_pull_one_shot_add_rms_norm, Op::all_reduce_add_rms_norm, false},
    {Kernel::all_reduce_pull_two_shot_add_rms_norm, Op::all_reduce_add_rms_norm, true},
    {Kernel::all_reduce_pull_one_shot_add_attn_res_rms_norm, Op::all_reduce_add_attn_res_rms_norm,
     false},
    {Kernel::all_reduce_pull_two_shot_add_attn_res_rms_norm, Op::all_reduce_add_attn_res_rms_norm,
     true},
    {Kernel::all_reduce_pull_one_shot_rms_norm_gemm_add, Op::all_reduce_rms_norm_gemm_add, false},
    {Kernel::all_reduce_pull_two_shot_rms_norm_gemm_add, Op::all_reduce_rms_norm_gemm_add, true},
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
constexpr bool is_two_shot(Kernel k) { return info(k).two_shot; }

// The op's kernel of that direction and shot; none where it has none.
constexpr Kernel kernel_of(Op op, bool two_shot) {
  for (const KernelInfo& i : kKernels)
    if (i.op == op && i.two_shot == two_shot) return i.kernel;
  return Kernel::none;
}

// What runs: the kernel, its grid and block, for the GEMM tail its lanes per column (0 for
// every other kernel), the precision the caller accepts (16: T itself; no kernel quantizes yet),
// and for a row kernel the packs of its row each thread holds (0 for every other kernel).
struct Launch {
  Kernel kernel;
  int grid;
  int threads;
  int gemm_lanes_per_col;
  int quant_bits;
  int row_packs;
};

// A PULL ROW KERNEL'S BUILDS: each thread holds kRowPacks packs of its row in registers, built at
// powers of two up to 16, which holds a 7168-wide bf16 row (896 packs) at one wave per block.
constexpr int kRowPacksBuilt[] = {1, 2, 4, 8, 16};

constexpr bool has_row_packs(Kernel k) { return op_of(k) != Op::all_reduce; }

// The smallest build that holds a row of `packs` over `threads`; 0 when none does.
constexpr int row_packs_for(int64_t packs, int threads) {
  for (const int b : kRowPacksBuilt)
    if (int64_t{b} * threads >= packs) return b;
  return 0;
}

// A row op's one-shot kernel gives each block a row, so it needs no more blocks than rows;
// everything else strides over the whole grid.
constexpr int grid_of(Kernel k, int blocks, int64_t rows) {
  const Op op = op_of(k);
  const bool row_per_block =
      !is_two_shot(k) && op != Op::all_reduce && op != Op::all_reduce_rms_norm_gemm_add;
  return row_per_block && rows < blocks ? static_cast<int>(rows) : blocks;
}

}  // namespace hip_comms
