// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE CATALOG: what each Template is (its op, its shot) and the row builds each op has.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <cstdint>

namespace hip_comms {

// What each template is, in Template's order: its op, its shot, and for a row template the most
// packs of its row a thread holds (machine/build.cuh derives each; 0 for one without rows). A build
// past that would only ever run slower, so it is not built.
struct TemplateInfo {
  Template fn;
  Op op;
  bool two_shot;
  int max_row_packs;
};

constexpr TemplateInfo kTemplates[] = {
    {Template::all_reduce_pull_one_shot, Op::all_reduce, false, 0},
    {Template::all_reduce_pull_two_shot, Op::all_reduce, true, 0},
    {Template::all_reduce_pull_one_shot_rms_norm, Op::all_reduce_rms_norm, false,
     kBuild.norm_row_packs},
    {Template::all_reduce_pull_two_shot_rms_norm, Op::all_reduce_rms_norm, true,
     kBuild.pipelined_row_packs},
    {Template::all_reduce_pull_one_shot_add_rms_norm, Op::all_reduce_add_rms_norm, false,
     kBuild.norm_row_packs},
    {Template::all_reduce_pull_two_shot_add_rms_norm, Op::all_reduce_add_rms_norm, true,
     kBuild.pipelined_row_packs},
    {Template::all_reduce_pull_one_shot_add_attn_res_rms_norm, Op::all_reduce_add_attn_res_rms_norm,
     false, kBuild.attn_res_row_packs},
    {Template::all_reduce_pull_two_shot_add_attn_res_rms_norm, Op::all_reduce_add_attn_res_rms_norm,
     true, kBuild.attn_res_row_packs},
    {Template::all_reduce_pull_one_shot_rms_norm_gemm_add, Op::all_reduce_rms_norm_gemm_add, false,
     kBuild.norm_row_packs},
    {Template::all_reduce_pull_two_shot_rms_norm_gemm_add, Op::all_reduce_rms_norm_gemm_add, true,
     kBuild.norm_row_packs},
    {Template::all_reduce_push_two_shot_rms_norm, Op::all_reduce_rms_norm, true,
     kBuild.norm_row_packs},
    {Template::all_reduce_push_two_shot_add_rms_norm, Op::all_reduce_add_rms_norm, true,
     kBuild.norm_row_packs},
    {Template::all_reduce_push_two_shot_add_attn_res_rms_norm, Op::all_reduce_add_attn_res_rms_norm,
     true, kBuild.attn_res_row_packs},
    {Template::all_reduce_pull_one_shot_rms_norm_gemm, Op::all_reduce_rms_norm_gemm, false,
     kBuild.norm_row_packs},
    {Template::all_reduce_pull_two_shot_rms_norm_gemm, Op::all_reduce_rms_norm_gemm, true,
     kBuild.norm_row_packs},
    {Template::all_reduce_pull_one_shot_rms_scale_add, Op::all_reduce_rms_scale_add, false,
     kBuild.scale_add_row_packs},
    {Template::all_reduce_pull_two_shot_rms_scale_add, Op::all_reduce_rms_scale_add, true,
     kBuild.scale_add_pipelined_row_packs},
};
constexpr int kNumTemplates = sizeof(kTemplates) / sizeof(TemplateInfo);

constexpr bool templates_in_order() {
  for (int i = 0; i < kNumTemplates; ++i)
    if (static_cast<int>(kTemplates[i].fn) != i) return false;
  return true;
}
static_assert(templates_in_order(), "kTemplates must list every Template in its order");

constexpr const TemplateInfo& info(Template k) { return kTemplates[static_cast<int>(k)]; }
constexpr Op op_of(Template k) { return info(k).op; }
constexpr bool is_two_shot(Template k) { return info(k).two_shot; }
// A norm then a GEMM, written or added: their GEMM phase strides over column tiles.
constexpr bool gemms(Op op) {
  return op == Op::all_reduce_rms_norm_gemm || op == Op::all_reduce_rms_norm_gemm_add;
}

constexpr int max_row_packs(Template k) { return info(k).max_row_packs; }

constexpr bool has_row_packs(Template k) { return max_row_packs(k) > 0; }

// The smallest build of `k` that holds a row of `packs` over `threads`; 0 when none does.
constexpr int row_packs_for(Template k, int64_t packs, int threads) {
  for (int b = 1; b <= max_row_packs(k); b *= 2)
    if (int64_t{b} * threads >= packs) return b;
  return 0;
}

}  // namespace hip_comms
