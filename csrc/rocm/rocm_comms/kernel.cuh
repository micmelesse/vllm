// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE KERNEL: what runs. A `Template` (a family of compiled kernels), its arguments from the call
// (`TemplateArgs`), and its config (`KernelConfig`: its family's launch and fields, Triton's
// autotune config). select chooses one (select.cuh); dispatch turns it into the compiled instance.

#pragma once

#include <variant>

#include "build.cuh"

namespace hip_comms {

// Every `__global__` template there is, named by its shot and what it fuses: a family of kernels,
// one per set of template arguments.
enum class Template : int {
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
  all_reduce_push_two_shot_rms_norm              = 10,
  all_reduce_push_two_shot_add_rms_norm          = 11,
  all_reduce_push_two_shot_add_attn_res_rms_norm = 12,
  all_reduce_pull_one_shot_rms_norm_gemm         = 13,
  all_reduce_pull_two_shot_rms_norm_gemm         = 14,
  all_reduce_pull_one_shot_rms_scale_add         = 15,
  all_reduce_pull_two_shot_rms_scale_add         = 16,
  add_attn_res_rms_norm                          = 17,  // experimental: no all-reduce
};

// A TEMPLATE'S ARGUMENTS, one struct per family: only the parameters that family has; the tile and
// the launch are the Kernel's KernelConfig.
// `staged`: the build that copies an eager input into its staging a pass at a time (any size),
// not the one that reads a registered or captured input in place; plan decides it.
struct AllReduceTemplateArgs {
  int world;
  DType dtype;
  bool staged;
};
struct NormTemplateArgs {
  int world;
  DType dtype;
  DType weight;  // dtype, or f32
};
struct AttnResTemplateArgs {
  int world;
  DType dtype;
  bool prefix;
};
struct GemmTemplateArgs {
  int world;
  DType dtype;
};
struct ScaleAddTemplateArgs {
  int world;
  DType dtype;
};
using TemplateArgs = std::variant<AllReduceTemplateArgs, NormTemplateArgs, AttnResTemplateArgs,
                                  GemmTemplateArgs, ScaleAddTemplateArgs>;

// HOW A KERNEL IS LAUNCHED: its block's threads (Triton's num_warps x 64) and its grid. Every
// kernel has one; the grid is the launch's alone, never compiled in.
struct LaunchConfig {
  int threads_per_block;
  int blocks_per_grid;
};

// EACH KERNEL FAMILY'S CONFIG, Triton's autotune config: its launch and its own fields, nothing it
// lacks. The tile is TILE_M rows x TILE_N columns in elements (tile_n 0 when no build holds the
// call's row, which check refuses); TILE_K the reduced dimension a step (the GEMM's K a pass,
// AttnRes's sources a step); SLICE_K lanes splitting one output's K (CUTLASS's sliced-K). Every
// field is compiled in except the launch's grid and reduce_scatter_blocks.
// The plain all-reduce: no tile (it strides over packs).
struct AllReduceConfig {
  LaunchConfig launch;
};
// The norms and the one-all-reduce tail: one row a tile.
struct RowConfig {
  LaunchConfig launch;
  int tile_n;
};
// AttnRes's one-shot and push: one row a tile, TILE_K sources a step.
struct AttnResConfig {
  LaunchConfig launch;
  int tile_n;
  int tile_k;
};
// AttnRes's pull two-shot: TILE_M rows a tile, and its reduce-scatter on the grid's first
// `reduce_scatter_blocks` blocks (its reads queue behind the links past a few dozen; AttnRes after
// it on every block).
struct AttnResPullConfig {
  LaunchConfig launch;
  int tile_m;
  int tile_n;
  int tile_k;
  int reduce_scatter_blocks;
};
// The GEMM tails.
struct GemmConfig {
  LaunchConfig launch;
  int tile_m;
  int tile_n;
  int tile_k;
  int slice_k;
};
using KernelConfig =
    std::variant<AllReduceConfig, RowConfig, AttnResConfig, AttnResPullConfig, GemmConfig>;

// THE FIELDS EVERY FAMILY SHARES, and those it may have: 0 where it has none (the plain
// all-reduce's tile; one row a tile is TILE_M 1).
constexpr const LaunchConfig& launch_of(const KernelConfig& c) {
  return std::visit([](const auto& f) -> const LaunchConfig& { return f.launch; }, c);
}
constexpr LaunchConfig& launch_of(KernelConfig& c) {
  return std::visit([](auto& f) -> LaunchConfig& { return f.launch; }, c);
}
constexpr int tile_m_of(const KernelConfig& c) {
  return std::visit(
      [](const auto& f) {
        if constexpr (requires { f.tile_m; }) return f.tile_m;
        else if constexpr (requires { f.tile_n; }) return 1;
        else return 0;
      },
      c);
}
constexpr int tile_n_of(const KernelConfig& c) {
  return std::visit(
      [](const auto& f) {
        if constexpr (requires { f.tile_n; }) return f.tile_n;
        else return 0;
      },
      c);
}
// A CONFIG'S TILE_N SET, where its family has one.
constexpr void set_tile_n(KernelConfig& c, int n) {
  std::visit(
      [&](auto& f) {
        if constexpr (requires { f.tile_n; }) f.tile_n = n;
      },
      c);
}
// FAMILY `i`'S CONFIG WITH EVERY KNOB 0: a forced template's own, all left to it.
template <size_t I = 0>
constexpr KernelConfig zero_config(size_t family) {
  if constexpr (I + 1 < std::variant_size_v<KernelConfig>)
    if (family != I) return zero_config<I + 1>(family);
  return KernelConfig{std::in_place_index<I>};
}

// WHAT SELECT RETURNS: one kernel, the template with its arguments decided (the compiled
// instruction sequence), and its KernelConfig.
struct Kernel {
  Template fn;
  TemplateArgs args;
  KernelConfig config;
};

}  // namespace hip_comms
