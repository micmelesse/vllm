// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE VOCABULARY, every type the tables and the handle name: why a call
// cannot run (`Error`), what is tuned and forced (`Template`, and each family's
// build, `KernelConfig`), and what a caller asks for (`OpType`, and the
// forcing: `Algorithm`, `Direction`). `DType` is common.cuh's; what runs
// (`Kernel`, `Config`) is config.cuh's.

#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <optional>
#include <variant>

#include "kernels/common.cuh"
#include "kernels/reduce_scatter.cuh"

namespace hip_custom_communicator {

// WHY A CALL CANNOT RUN, every reason there is. The numbers cross to Python
// (hip_custom_communicator.Error), so a reason is only ever added at the end.
// `disabled` and `no_such_op` are the communicator's own.
enum class Error : int {
  disabled = 0,
  no_such_op = 1,
  not_contiguous = 2,
  not_two_d = 3,
  dtype_not_built = 4,
  world_not_built = 5,
  row_not_packs = 6,
  template_not_this_ops = 7,
  row_too_wide = 8,
  block_not_a_wave_per_peer = 9,
  partials_too_small = 10,
  grid_not_resident = 11,
  inputs_too_small = 12,
  device_not_built = 13,
  device_not_tuned = 14,
  weight_not_built = 15,
  no_such_template = 16,
  no_such_group = 17,
  ranks_disagree = 18,
  groups_disagree = 19,
  threads_not_built = 20,
  tile_not_built = 21,
  launch_out_of_range = 22,
  field_not_positive = 23,
  field_not_this_templates = 24,
  waves_not_built = 25,
  inner_stride_not_one = 26,
  row_stride_not_packs = 27,
  row_narrower_than_world = 28,  // retired: every two-shot takes a row narrower than a
                                // pack a rank (the late ranks' slices empty)
  forced_incomplete = 29,
  no_such_quantization = 30,
  quantization_not_supported = 31,
  row_not_int4_groups = 32,  // retired: every INT4 kernel takes a row of whole packs, its last
                             // group short
  quantized_too_small = 33,  // retired: INT4 is an encoding inside the
                             // regions (inputs_too_small, partials_too_small)
};
constexpr int kNumErrors = 34;

constexpr const char* to_string(Error e) {
  switch (e) {
    case Error::disabled:
      return "disabled: the communicator is disabled";
    case Error::no_such_op:
      return "no_such_op: the backend has no such op";
    case Error::not_contiguous:
      return "not_contiguous: the input is not contiguous";
    case Error::not_two_d:
      return "not_two_d: a fused op takes a 2-D input";
    case Error::dtype_not_built:
      return "dtype_not_built: only float16 and bfloat16 are built";
    case Error::world_not_built:
      return "world_not_built: the world size is not 2, 4 or 8";
    case Error::row_not_packs:
      return "row_not_packs: the row is not whole 16-byte packs";
    case Error::template_not_this_ops:
      return "template_not_this_ops: the forced template is not this op's";
    case Error::row_too_wide:
      return "row_too_wide: the row is wider than the template's widest build "
             "holds";
    case Error::block_not_a_wave_per_peer:
      return "block_not_a_wave_per_peer: a two-shot block must be one wave per "
             "peer";
    case Error::partials_too_small:
      return "partials_too_small: the two-shot's published sums, exact or INT4, "
             "exceed the partials";
    case Error::grid_not_resident:
      return "grid_not_resident: the grid exceeds the blocks the GPU holds "
             "resident";
    case Error::inputs_too_small:
      return "inputs_too_small: the input this kernel publishes whole, exact "
             "or INT4, exceeds the inputs";
    case Error::device_not_built:
      return "device_not_built: this build holds no code for the device";
    case Error::device_not_tuned:
      return "device_not_tuned: the device is not the one select is calibrated "
             "for";
    case Error::weight_not_built:
      return "weight_not_built: a norm's weight is in the call's dtype or fp32";
    case Error::no_such_template:
      return "no_such_template: the op has no template of the forced algorithm "
             "and direction";
    case Error::no_such_group:
      return "no_such_group: no process group is registered by that name";
    case Error::ranks_disagree:
      return "ranks_disagree: the ranks captured different numbers of buffers";
    case Error::groups_disagree:
      return "groups_disagree: the CPU and device groups differ in size or in "
             "this rank";
    case Error::threads_not_built:
      return "threads_not_built: no build of the template runs at that block "
             "size";
    case Error::waves_not_built:
      return "waves_not_built: no build of the template at that block size has "
             "those waves a EU";
    case Error::tile_not_built:
      return "tile_not_built: no build of the template has that TILE_M or "
             "TILE_N";
    case Error::launch_out_of_range:
      return "launch_out_of_range: threads_per_block is whole waves up to the "
             "build's maximum, "
             "blocks_per_grid at least 1 and at most the resident maximum";
    case Error::field_not_positive:
      return "field_not_positive: a forced config field is positive";
    case Error::field_not_this_templates:
      return "field_not_this_templates: the forced template has no such config "
             "field";
    case Error::inner_stride_not_one:
      return "inner_stride_not_one: a buffer's innermost stride is not 1 (the "
             "loads are packs)";
    case Error::row_stride_not_packs:
      return "row_stride_not_packs: a buffer's row stride is not whole 16-byte "
             "packs";
    case Error::row_narrower_than_world:
      return "row_narrower_than_world: a two-shot row has fewer packs than "
             "ranks, so a rank's "
             "slice is empty";
    case Error::forced_incomplete:
      return "forced_incomplete: a forced call gives every field of its config "
             "(the template, "
             "its build and its launch) or none";
    case Error::no_such_quantization:
      return "no_such_quantization: the quantization is none or int4";
    case Error::quantization_not_supported:
      return "quantization_not_supported: INT4 is not supported for the selected "
             "kernel";
    case Error::row_not_int4_groups:
      return "row_not_int4_groups: an INT4 row is whole groups of 32 for every "
             "rank";
    case Error::quantized_too_small:
      return "quantized_too_small: retired";
  }
  return "unknown";
}

// Every `__global__` template there is, named by its shot and what it fuses: a
// family of kernels, one per set of template arguments.
enum class Template : int {
  all_reduce_pull_one_shot = 0,
  all_reduce_pull_two_shot = 1,
  all_reduce_pull_one_shot_rms_norm = 2,
  all_reduce_pull_two_shot_rms_norm = 3,
  all_reduce_pull_one_shot_add_rms_norm = 4,
  all_reduce_pull_two_shot_add_rms_norm = 5,
  all_reduce_pull_one_shot_add_attn_res_rms_norm = 6,
  all_reduce_pull_two_shot_add_attn_res_rms_norm = 7,
  all_reduce_push_two_shot_rms_norm = 8,
  all_reduce_push_two_shot_add_rms_norm = 9,
  all_reduce_push_two_shot_add_attn_res_rms_norm = 10,
  all_reduce_push_one_shot = 11,
  all_reduce_push_two_shot = 12,
  all_reduce_pull_split_add_attn_res_rms_norm = 13,
  all_reduce_pull_split_rms_norm = 14,
  all_reduce_pull_split_add_rms_norm = 15,
};

// HOW A CALLER FORCES A TEMPLATE: the algorithm (one-shot reads every peer's
// whole input; two-shot reduce-scatters then gathers; split launches two
// kernels of ours, our plain all-reduce into this rank's local sums
// (Handle::sums), then our separate kernel of the op's epilogue on them) and
// the direction (common.cuh's `Direction`, which its reduce-scatters take too;
// a split's all-reduce is the plain all-reduce's tuned pick, a pull).
enum class Algorithm : int { one_shot = 0, two_shot = 1, split = 2 };

// WHICH OP the caller asked for: an all-reduce, alone or with what it fuses.
enum class OpType : int {
  all_reduce = 0,
  all_reduce_rms_norm = 1,
  all_reduce_add_rms_norm = 2,
  all_reduce_add_attn_res_rms_norm = 3,
};

// A KERNEL'S CONFIG, in two parts and every field always set: its BUILD, what
// is compiled in (one of the builds its template lists, config.cuh), and its
// LAUNCH, what is chosen at each call. A tuned kernel holds both, a forced call
// gives both, and select hands both to the launch; no field is ever 0 for "left
// to someone else".
//
// EACH OP'S BUILD, Triton's autotune config less the grid: the block's threads
// (Triton's num_warps x 64), the waves a SIMD (an EU) the compiler fits its
// registers to (__launch_bounds__'s second value, Triton's waves_per_eu; 1 asks
// nothing), and the tile, TILE_M rows x TILE_N columns in elements; TILE_K is
// the reduced dimension a step (AttnRes's sources a step); `quantization`, what
// crosses the links (common.cuh's QUANTIZATION): a two-shot's build is listed
// as it is and moving INT4 (select's `and_int4`), forced like any field or
// named by a tuned point. One type an op, so a config
// cannot cross ops, though three hold the same fields. all_reduce: a tile a
// work item, the grid striding over the buffer's.
struct AllReduceBuildConfig {
  int threads_per_block;
  int waves_per_eu;
  int tile_m;
  int tile_n;
  int quantization = Quantization::none;  // a Quantization
  bool operator==(const AllReduceBuildConfig&) const = default;
};
// all_reduce_rms_norm.
struct AllReduceRmsNormBuildConfig {
  int threads_per_block;
  int waves_per_eu;
  int tile_m;
  int tile_n;
  int quantization = Quantization::none;  // a Quantization
  bool operator==(const AllReduceRmsNormBuildConfig&) const = default;
};
// all_reduce_add_rms_norm.
struct AllReduceAddRmsNormBuildConfig {
  int threads_per_block;
  int waves_per_eu;
  int tile_m;
  int tile_n;
  int quantization = Quantization::none;  // a Quantization
  bool operator==(const AllReduceAddRmsNormBuildConfig&) const = default;
};
// all_reduce_add_attn_res_rms_norm: TILE_K sources a step.
struct AllReduceAddAttnResRmsNormBuildConfig {
  int threads_per_block;
  int waves_per_eu;
  int tile_m;
  int tile_n;
  int tile_k;
  int quantization = Quantization::none;  // a Quantization
  bool operator==(const AllReduceAddAttnResRmsNormBuildConfig&) const = default;
};
using BuildConfig =
    std::variant<AllReduceBuildConfig, AllReduceRmsNormBuildConfig,
                 AllReduceAddRmsNormBuildConfig,
                 AllReduceAddAttnResRmsNormBuildConfig>;

// A KERNEL'S LAUNCH: its grid, never compiled in, and the blocks of it the
// fused AttnRes pull two-shot gives its reduce-scatter (its reads queue behind
// the links past a few dozen; AttnRes after it on every block).
// reduce_scatter_blocks 0 is no split, every block both: the only value any
// other kernel takes.
struct LaunchConfig {
  int blocks_per_grid;
  int reduce_scatter_blocks;
};

// THE COMPLETE CONFIG: the build and the launch.
struct KernelConfig {
  BuildConfig build;
  LaunchConfig launch;
};

// THE FIELDS EVERY BUILD HAS.
constexpr int threads_of(const BuildConfig& b) {
  return std::visit([](const auto& f) { return f.threads_per_block; }, b);
}
constexpr int waves_of(const BuildConfig& b) {
  return std::visit([](const auto& f) { return f.waves_per_eu; }, b);
}
constexpr int tile_m_of(const BuildConfig& b) {
  return std::visit([](const auto& f) { return f.tile_m; }, b);
}
constexpr int tile_n_of(const BuildConfig& b) {
  return std::visit([](const auto& f) { return f.tile_n; }, b);
}
constexpr int quantization_of(const BuildConfig& b) {
  return std::visit([](const auto& f) { return f.quantization; }, b);
}
// The same build, moving `q` across the links.
constexpr BuildConfig quantized(BuildConfig b, int q) {
  std::visit([q](auto& f) { f.quantization = q; }, b);
  return b;
}

// THE TUNED POINTS (vllm/.../hip_custom_communicator_configs/<arch>.json, loaded by Python and
// handed to open, checked there, held by the handle): each a measured call (its op, world, width
// and tokens) with the kernel and config that won there. A call runs its nearest point's
// (config.cuh's `lookup`). `mode`: the table the point belongs to, the communicator's exact one or
// its quantized one (every size: exact kernels where quantizing does not pay, INT4 where it does).
// `pick` false: not a choice, a kernel's config for a call that forces it where no point picks it.
enum class Mode : int { exact = 0, quantized = 1 };
struct TunedPoint {
  OpType op;
  Mode mode;
  bool pick;
  int world;
  int64_t hidden;
  int64_t tokens;
  Template fn;
  KernelConfig config;
};
// A point on the wire from Python: its fields in this order, int64 each (`Template` and `OpType`
// by their enums' numbers, `build_info`'s order).
constexpr int kPointFields = 15;

constexpr int elem_bytes(DType d) { return d == DType::f32 ? 4 : 2; }
// A row of `elems` of `dtype` in 16-byte packs.
constexpr int64_t packs_of(int64_t elems, DType dtype) {
  return elems * elem_bytes(dtype) / kBuild.memory.pack_bytes;
}

}  // namespace hip_custom_communicator
