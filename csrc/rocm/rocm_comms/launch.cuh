// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// WHICH KERNEL RUNS, AND HOW WIDE. The caller names an op; this picks one kernel from the
// flat list and describes its launch geometry; tune.cuh picks one for a call. Nothing above C++
// sees an algorithm or a geometry.

#pragma once

#include <climits>
#include <cstdint>

#include "common/common.cuh"
#include "build.cuh"

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
// reads its peers' buffers: their inputs, or scratch they filled). `none` is no kernel named: a
// launch not forced, which tune.cuh picks. Each op has a one-shot and a two-shot.
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

// A ROW KERNEL'S BUILDS: each thread holds kRowPacks packs of its row in registers, built at powers
// of two up to what the op fits without spilling (build.cuh derives it). A build past that would
// only ever run slower, so it is not built.
constexpr int max_row_packs(Op op) {
  switch (op) {
    case Op::all_reduce: return 0;
    case Op::all_reduce_add_attn_res_rms_norm: return kBuild.attn_res_row_packs;
    default: return kBuild.norm_row_packs;
  }
}

constexpr bool has_row_packs(Kernel k) { return op_of(k) != Op::all_reduce; }

// The smallest build of `op` that holds a row of `packs` over `threads`; 0 when none does.
constexpr int row_packs_for(Op op, int64_t packs, int threads) {
  for (int b = 1; b <= max_row_packs(op); b *= 2)
    if (int64_t{b} * threads >= packs) return b;
  return 0;
}

// A ROW OP GIVES EACH BLOCK WHOLE ROWS, so it needs no more blocks than it has rows: a one-shot all
// of them, a two-shot its rank's slice. An idle block still pays every barrier (each pairs with its
// twin on every peer): the norms' two-shot at 32 tokens ran 36 blocks for 4 rows a rank. The GEMM
// tail's GEMM strides over column tiles, the plain all-reduce over packs, and the norms' two-shot
// gather over the slice's packs behind a world barrier, so they keep theirs.
constexpr int grid_of(Kernel k, int blocks, int64_t rows, int world) {
  const Op op = op_of(k);
  const bool strides = op == Op::all_reduce || op == Op::all_reduce_rms_norm_gemm_add ||
                       k == Kernel::all_reduce_pull_two_shot_rms_norm ||
                       k == Kernel::all_reduce_pull_two_shot_add_rms_norm;
  if (strides) return blocks;
  const int64_t mine = is_two_shot(k) ? (rows + world - 1) / world : rows;
  return mine < blocks ? static_cast<int>(mine) : blocks;
}

}  // namespace hip_comms
