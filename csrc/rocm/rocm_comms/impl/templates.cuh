// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE CATALOG: what each Template is (its op, its shot) and the builds (Instances) each one has.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <cstdint>
#include <optional>
#include <span>
#include <string>

namespace hip_comms {

// A ROW TEMPLATE'S BUILD, compiled in: its tile and its threads a block (composable_kernel's
// instance; Triton's autotune config without the grid, which is the launch's). tile_n counts the
// elements of a 16-bit dtype, the only ones built. What a template lists is exactly what dispatch
// instantiates, what check accepts and what a tuner searches.
struct Instance {
  int tile_m;
  int tile_n;
  int threads_per_block;
};

// EACH FAMILY'S BUILDS. A build that spills shows in its code object (Handle::resources_of) and
// loses on the clock; none is ruled out by policy.
constexpr Instance kNormInstances[] = {{1, 4096, 512}, {1, 8192, 512}, {1, 16384, 512}};
constexpr Instance kPipelinedNormInstances[] = {{1, 4096, 512}, {1, 8192, 512}};
constexpr Instance kAttnResInstances[] = {{1, 4096, 512}, {1, 8192, 512}};
constexpr Instance kAttnResPullInstances[] = {{1, 4096, 512}, {1, 8192, 512}, {2, 4096, 512},
                                              {2, 8192, 512}, {4, 4096, 512}, {4, 8192, 512}};
constexpr Instance kGemmTailInstances[] = {{1, 4096, 512}, {1, 8192, 512}, {1, 16384, 512}};
constexpr Instance kScaleAddInstances[] = {{1, 4096, 512}, {1, 8192, 512}};

// What each template is, in Template's order: its op, its shot, and its builds (none for the plain
// all-reduce, which has no tile).
struct TemplateInfo {
  Template fn;
  const char* name;
  Op op;
  bool two_shot;
  std::span<const Instance> instances;
};

// A TEMPLATE AND ITS NAME FROM ONE TOKEN, so the name cannot differ from the enum's.
#define HIP_COMMS_NAMED(t) Template::t, #t

constexpr TemplateInfo kTemplates[] = {
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot), Op::all_reduce, false, {}},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot), Op::all_reduce, true, {}},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_rms_norm), Op::all_reduce_rms_norm, false,
     kNormInstances},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_rms_norm), Op::all_reduce_rms_norm, true,
     kPipelinedNormInstances},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_add_rms_norm), Op::all_reduce_add_rms_norm, false,
     kNormInstances},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_add_rms_norm), Op::all_reduce_add_rms_norm, true,
     kPipelinedNormInstances},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_add_attn_res_rms_norm),
     Op::all_reduce_add_attn_res_rms_norm, false, kAttnResInstances},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_add_attn_res_rms_norm),
     Op::all_reduce_add_attn_res_rms_norm, true, kAttnResPullInstances},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_rms_norm_gemm_add), Op::all_reduce_rms_norm_gemm_add,
     false, kGemmTailInstances},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_rms_norm_gemm_add), Op::all_reduce_rms_norm_gemm_add,
     true, kGemmTailInstances},
    {HIP_COMMS_NAMED(all_reduce_push_two_shot_rms_norm), Op::all_reduce_rms_norm, true,
     kNormInstances},
    {HIP_COMMS_NAMED(all_reduce_push_two_shot_add_rms_norm), Op::all_reduce_add_rms_norm, true,
     kNormInstances},
    {HIP_COMMS_NAMED(all_reduce_push_two_shot_add_attn_res_rms_norm),
     Op::all_reduce_add_attn_res_rms_norm, true, kAttnResInstances},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_rms_norm_gemm), Op::all_reduce_rms_norm_gemm, false,
     kGemmTailInstances},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_rms_norm_gemm), Op::all_reduce_rms_norm_gemm, true,
     kGemmTailInstances},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_rms_scale_add), Op::all_reduce_rms_scale_add, false,
     kScaleAddInstances},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_rms_scale_add), Op::all_reduce_rms_scale_add, true,
     kScaleAddInstances},
};
#undef HIP_COMMS_NAMED
constexpr int kNumTemplates = sizeof(kTemplates) / sizeof(TemplateInfo);

constexpr bool templates_in_order() {
  for (int i = 0; i < kNumTemplates; ++i)
    if (static_cast<int>(kTemplates[i].fn) != i) return false;
  return true;
}
static_assert(templates_in_order(), "kTemplates must list every Template in its order");

constexpr const TemplateInfo& info(Template k) { return kTemplates[static_cast<int>(k)]; }
constexpr Op op_of(Template k) { return info(k).op; }
constexpr const char* to_string(Template k) { return info(k).name; }
// The template a name names, or none: how a caller forces one (the bench's sweeps, the tests).
inline std::optional<Template> template_named(const std::string& name) {
  for (const TemplateInfo& t : kTemplates)
    if (name == t.name) return t.fn;
  return std::nullopt;
}
constexpr bool is_two_shot(Template k) { return info(k).two_shot; }
// A norm then a GEMM, written or added: their GEMM phase strides over column tiles.
constexpr bool gemms(Op op) {
  return op == Op::all_reduce_rms_norm_gemm || op == Op::all_reduce_rms_norm_gemm_add;
}

constexpr std::span<const Instance> instances_of(Template k) { return info(k).instances; }
constexpr bool has_tiles(Template k) { return !instances_of(k).empty(); }

// Whether template `k` builds `c`'s tile at `c`'s threads.
constexpr bool built(Template k, const KernelConfig& c) {
  for (const Instance& i : instances_of(k))
    if (i.tile_m == c.tile_m && i.tile_n == c.tile_n && i.threads_per_block == c.threads_per_block)
      return true;
  return false;
}
constexpr bool built_at(Template k, int threads_per_block) {
  for (const Instance& i : instances_of(k))
    if (i.threads_per_block == threads_per_block) return true;
  return false;
}

// THE SMALLEST BUILT TILE_N of `k` that covers `cols` at `tile_m` rows and `threads_per_block`;
// 0 when none does.
constexpr int tile_n_for(Template k, int64_t cols, int tile_m, int threads_per_block) {
  int best = 0;
  for (const Instance& i : instances_of(k))
    if (i.tile_m == tile_m && i.threads_per_block == threads_per_block && i.tile_n >= cols &&
        (best == 0 || i.tile_n < best))
      best = i.tile_n;
  return best;
}

}  // namespace hip_comms
