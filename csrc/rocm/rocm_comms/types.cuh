// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE VOCABULARY, every type the library names: what is compiled (`Template`, and each family's
// config, `KernelConfig`), what a caller asks for (`OpType`, and the forcing: `Algorithm`,
// `Direction`), and per op its LAUNCH, the normal form select returns and launch runs: which
// kernel (its algorithm and direction), what is compiled in, its grid, and its arguments, each a
// plain field, 0 where the template has none. `DType` is build.cuh's.

#pragma once

#include <cstdint>
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

// HOW A CALLER FORCES A TEMPLATE: the algorithm (one-shot reads every peer's whole input; two-shot
// reduce-scatters then gathers) and the direction (pull: this rank reads its peers; push: they
// write into it).
enum class Algorithm : int { one_shot = 0, two_shot = 1 };
enum class Direction : int { pull = 0, push = 1 };

// WHICH OP the caller asked for: an all-reduce, alone or with what it fuses, and the experimental
// ops (no all-reduce).
enum class OpType : int {
  all_reduce                       = 0,
  all_reduce_rms_norm              = 1,
  all_reduce_add_rms_norm          = 2,
  all_reduce_add_attn_res_rms_norm = 3,
  all_reduce_rms_norm_gemm_add     = 4,
  all_reduce_rms_norm_gemm         = 5,
  all_reduce_rms_scale_add         = 6,
  add_attn_res_rms_norm            = 7,  // experimental: no all-reduce
};

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
// FAMILY `i`'S CONFIG WITH EVERY FIELD 0: a forced template's own, all left to it.
template <size_t I = 0>
constexpr KernelConfig zero_config(size_t family) {
  if constexpr (I + 1 < std::variant_size_v<KernelConfig>)
    if (family != I) return zero_config<I + 1>(family);
  return KernelConfig{std::in_place_index<I>};
}


// =================================================================================================
// EACH OP'S LAUNCH: the kernel select decided, and every argument it runs with. Which kernel
// (algorithm, direction), what is compiled in (the tile, threads_per_block), the grid
// (blocks_per_grid, and the pull's reduce_scatter_blocks), then the kernel's arguments.
// =================================================================================================

// `staged`: the build that copies an eager input through the staging a pass at a time, not the
// one that reads a registered or captured input in place.
struct AllReduceLaunch {
  Algorithm algorithm;
  Direction direction;
  int threads_per_block;
  int blocks_per_grid;
  bool staged;
  void* out;
  const void* inp;
  int64_t bytes;
  DType dtype;
};

// `weight_dtype`: dtype, or f32.
struct AllReduceRmsNormLaunch {
  Algorithm algorithm;
  Direction direction;
  int tile_n;
  int threads_per_block;
  int blocks_per_grid;
  void* out;
  const void* inp;
  const void* weight;
  DType dtype;
  DType weight_dtype;
  int64_t rows;
  int64_t hidden;
  float eps;
};

struct AllReduceAddRmsNormLaunch {
  Algorithm algorithm;
  Direction direction;
  int tile_n;
  int threads_per_block;
  int blocks_per_grid;
  void* out;
  void* residual_out;
  const void* inp;
  const void* residual;
  const void* weight;
  DType dtype;
  DType weight_dtype;
  int64_t rows;
  int64_t hidden;
  float eps;
};

// tile_m 1 on the one-shot and push (a row a tile); reduce_scatter_blocks the pull two-shot's, 0
// on the others. `blocks` is [rows, sources, hidden] at its strides in elements; `write_idx` < 0
// writes no block; `out_norm_weight` null: no output norm.
struct AllReduceAddAttnResRmsNormLaunch {
  Algorithm algorithm;
  Direction direction;
  int tile_m;
  int tile_n;
  int tile_k;
  int threads_per_block;
  int blocks_per_grid;
  int reduce_scatter_blocks;
  void* prefix;
  void* out;
  const void* inp;
  void* blocks;
  int64_t block_stride_m;
  int64_t block_stride_r;
  const void* norm_weight;
  const void* qk_weight;
  const void* out_norm_weight;
  DType dtype;
  int64_t rows;
  int64_t hidden;
  int num_blocks;
  int write_idx;
  float eps;
  float out_eps;
  bool has_prefix;
};

// out [rows, n_cols] at `out_stride`; gemm_weight [n_cols, hidden]; `workspace` inp's shape.
struct AllReduceRmsNormGemmLaunch {
  Algorithm algorithm;
  Direction direction;
  int tile_m;
  int tile_n;
  int tile_k;
  int slice_k;
  int threads_per_block;
  int blocks_per_grid;
  void* out;
  int64_t out_stride;
  const void* inp;
  const void* norm_weight;
  float eps;
  const void* gemm_weight;
  int64_t n_cols;
  void* workspace;
  DType dtype;
  int64_t rows;
  int64_t hidden;
};

struct AllReduceRmsNormGemmAddLaunch {
  Algorithm algorithm;
  Direction direction;
  int tile_m;
  int tile_n;
  int tile_k;
  int slice_k;
  int threads_per_block;
  int blocks_per_grid;
  void* out;
  int64_t out_stride;
  const void* inp;
  const void* norm_weight;
  float eps;
  const void* gemm_weight;
  int64_t n_cols;
  void* workspace;
  DType dtype;
  int64_t rows;
  int64_t hidden;
};

// inp's row [shared | projected | latent], widths hidden, hidden, latent; out [rows, hidden].
struct AllReduceRmsScaleAddLaunch {
  Algorithm algorithm;
  Direction direction;
  int tile_n;
  int threads_per_block;
  int blocks_per_grid;
  void* out;
  const void* inp;
  DType dtype;
  int64_t rows;
  int64_t hidden;
  int64_t latent;
  float eps;
};

// Experimental, one rank: its one template is the pull one-shot by name only (it reads no peer).
struct AddAttnResRmsNormLaunch {
  Algorithm algorithm;
  Direction direction;
  int tile_n;
  int tile_k;
  int threads_per_block;
  int blocks_per_grid;
  void* prefix;
  void* out;
  const void* delta;
  void* blocks;
  int64_t block_stride_m;
  int64_t block_stride_r;
  const void* norm_weight;
  const void* qk_weight;
  const void* out_norm_weight;
  DType dtype;
  int64_t rows;
  int64_t hidden;
  int num_blocks;
  int write_idx;
  float eps;
  float out_eps;
};

}  // namespace hip_comms
