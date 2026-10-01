// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE CATALOG: what each Kernel is (its op, its shot) and the row builds each op has.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <cstdint>

namespace hip_comms {

// What each kernel is, in Kernel's order: its op, its shot, and for a row kernel the most packs of
// its row a thread holds (target/build.cuh derives each; 0 for a kernel without rows). A build past
// that would only ever run slower, so it is not built.
struct KernelInfo {
  Kernel kernel;
  Op op;
  bool two_shot;
  int max_row_packs;
};

constexpr KernelInfo kKernels[] = {
    {Kernel::all_reduce_pull_one_shot, Op::all_reduce, false, 0},
    {Kernel::all_reduce_pull_two_shot, Op::all_reduce, true, 0},
    {Kernel::all_reduce_pull_one_shot_rms_norm, Op::all_reduce_rms_norm, false,
     kBuild.norm_row_packs},
    {Kernel::all_reduce_pull_two_shot_rms_norm, Op::all_reduce_rms_norm, true,
     kBuild.pipelined_row_packs},
    {Kernel::all_reduce_pull_one_shot_add_rms_norm, Op::all_reduce_add_rms_norm, false,
     kBuild.norm_row_packs},
    {Kernel::all_reduce_pull_two_shot_add_rms_norm, Op::all_reduce_add_rms_norm, true,
     kBuild.pipelined_row_packs},
    {Kernel::all_reduce_pull_one_shot_add_attn_res_rms_norm, Op::all_reduce_add_attn_res_rms_norm,
     false, kBuild.attn_res_row_packs},
    {Kernel::all_reduce_pull_two_shot_add_attn_res_rms_norm, Op::all_reduce_add_attn_res_rms_norm,
     true, kBuild.attn_res_row_packs},
    {Kernel::all_reduce_pull_one_shot_rms_norm_gemm_add, Op::all_reduce_rms_norm_gemm_add, false,
     kBuild.norm_row_packs},
    {Kernel::all_reduce_pull_two_shot_rms_norm_gemm_add, Op::all_reduce_rms_norm_gemm_add, true,
     kBuild.norm_row_packs},
    {Kernel::all_reduce_push_two_shot_rms_norm, Op::all_reduce_rms_norm, true,
     kBuild.norm_row_packs},
    {Kernel::all_reduce_push_two_shot_add_rms_norm, Op::all_reduce_add_rms_norm, true,
     kBuild.norm_row_packs},
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

constexpr int max_row_packs(Kernel k) { return info(k).max_row_packs; }
constexpr bool has_row_packs(Kernel k) { return max_row_packs(k) > 0; }

// The smallest build of `k` that holds a row of `packs` over `threads`; 0 when none does.
constexpr int row_packs_for(Kernel k, int64_t packs, int threads) {
  for (int b = 1; b <= max_row_packs(k); b *= 2)
    if (int64_t{b} * threads >= packs) return b;
  return 0;
}

}  // namespace hip_comms
