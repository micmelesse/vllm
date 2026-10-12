// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE VOCABULARY, every type the library names: why a call cannot run (`Error`), what is
// compiled (`Template`, and each family's config, `KernelConfig`), what a caller asks for
// (`OpType`, and the forcing: `Algorithm`, `Direction`), each op's kernel signatures, and per op
// its LAUNCH, the normal form select returns and launch runs: which kernel (the compiled instance,
// its algorithm, direction and world), what is compiled in, its grid, its stream and its
// arguments, each a plain field, 0 where the template has none. `DType` is common/build.cuh's.

#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <variant>

#include "common/interface.cuh"

namespace hip_custom_comms {

// WHY A CALL CANNOT RUN, every reason there is. The numbers cross to Python (hip_custom_comms.Error), so
// a reason is only ever added at the end. `disabled` and `no_such_op` are the communicator's own.
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
  scratch_too_small = 10,
  grid_not_resident = 11,
  staging_too_small = 12,
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
  row_narrower_than_world = 28,
  forced_incomplete = 29,
};
constexpr int kNumErrors = 30;

constexpr const char* to_string(Error e) {
  switch (e) {
    case Error::disabled: return "disabled: the communicator is disabled";
    case Error::no_such_op: return "no_such_op: the backend has no such op";
    case Error::not_contiguous: return "not_contiguous: the input is not contiguous";
    case Error::not_two_d: return "not_two_d: a fused op takes a 2-D input";
    case Error::dtype_not_built: return "dtype_not_built: only float16 and bfloat16 are built";
    case Error::world_not_built: return "world_not_built: the world size is not 2, 4 or 8";
    case Error::row_not_packs: return "row_not_packs: the row is not whole 16-byte packs";
    case Error::template_not_this_ops:
      return "template_not_this_ops: the forced template is not this op's";
    case Error::row_too_wide:
      return "row_too_wide: the row is wider than the template's widest build holds";
    case Error::block_not_a_wave_per_peer:
      return "block_not_a_wave_per_peer: a two-shot block must be one wave per peer";
    case Error::scratch_too_small:
      return "scratch_too_small: the two-shot scratch exceeds the scratch";
    case Error::grid_not_resident:
      return "grid_not_resident: the grid exceeds the blocks the GPU holds resident";
    case Error::staging_too_small:
      return "staging_too_small: an eager input this kernel reads in place exceeds the staging";
    case Error::device_not_built:
      return "device_not_built: this build holds no code for the device";
    case Error::device_not_tuned:
      return "device_not_tuned: the device is not the one select is calibrated for";
    case Error::weight_not_built:
      return "weight_not_built: a norm's weight is in the call's dtype or fp32";
    case Error::no_such_template:
      return "no_such_template: the op has no template of the forced algorithm and direction";
    case Error::no_such_group: return "no_such_group: no process group is registered by that name";
    case Error::ranks_disagree:
      return "ranks_disagree: the ranks captured different numbers of buffers";
    case Error::groups_disagree:
      return "groups_disagree: the CPU and device groups differ in size or in this rank";
    case Error::threads_not_built:
      return "threads_not_built: no build of the template runs at that block size";
    case Error::waves_not_built:
      return "waves_not_built: no build of the template at that block size has those waves a EU";
    case Error::tile_not_built:
      return "tile_not_built: no build of the template has that TILE_M or TILE_N";
    case Error::launch_out_of_range:
      return "launch_out_of_range: threads_per_block is whole waves up to the build's maximum, "
             "blocks_per_grid at least 1 and at most the resident maximum";
    case Error::field_not_positive:
      return "field_not_positive: a forced config field is positive";
    case Error::field_not_this_templates:
      return "field_not_this_templates: the forced template has no such config field";
    case Error::inner_stride_not_one:
      return "inner_stride_not_one: a buffer's innermost stride is not 1 (the loads are packs)";
    case Error::row_stride_not_packs:
      return "row_stride_not_packs: a buffer's row stride is not whole 16-byte packs";
    case Error::row_narrower_than_world:
      return "row_narrower_than_world: a two-shot row has fewer packs than ranks, so a rank's "
             "slice is empty";
    case Error::forced_incomplete:
      return "forced_incomplete: a forced call gives every field of its config (the template, "
             "its build and its launch) or none";
  }
  return "unknown";
}

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
  all_reduce_push_two_shot_rms_norm              = 8,
  all_reduce_push_two_shot_add_rms_norm          = 9,
  all_reduce_push_two_shot_add_attn_res_rms_norm = 10,
};

// HOW A CALLER FORCES A TEMPLATE: the algorithm (one-shot reads every peer's whole input; two-shot
// reduce-scatters then gathers) and the direction (pull: this rank reads its peers; push: they
// write into it).
enum class Algorithm : int { one_shot = 0, two_shot = 1 };
enum class Direction : int { pull = 0, push = 1 };

// WHICH OP the caller asked for: an all-reduce, alone or with what it fuses.
enum class OpType : int {
  all_reduce                       = 0,
  all_reduce_rms_norm              = 1,
  all_reduce_add_rms_norm          = 2,
  all_reduce_add_attn_res_rms_norm = 3,
};

// A KERNEL'S CONFIG, in two parts and every field always set: its BUILD, what is compiled in (one
// of the builds its template lists, select.cuh), and its LAUNCH, what is chosen at each call. A
// tuned kernel holds both, a forced call gives both, and select hands both to the launch; no
// field is ever 0 for "left to someone else".
//
// EACH OP'S BUILD, Triton's autotune config less the grid: the block's threads (Triton's num_warps
// x 64), the waves a SIMD (an EU) the compiler fits its registers to (__launch_bounds__'s second
// value, Triton's waves_per_eu; 1 asks nothing), and the tile, TILE_M rows x TILE_N columns in
// elements; TILE_K is the reduced dimension a step (AttnRes's sources a step). One type an op,
// so a config cannot cross ops, though three hold the same fields.
// all_reduce: a tile a work item, the grid striding over the buffer's.
struct AllReduceBuildConfig {
  int threads_per_block;
  int waves_per_eu;
  int tile_m;
  int tile_n;
  bool operator==(const AllReduceBuildConfig&) const = default;
};
// all_reduce_rms_norm.
struct AllReduceRmsNormBuildConfig {
  int threads_per_block;
  int waves_per_eu;
  int tile_m;
  int tile_n;
  bool operator==(const AllReduceRmsNormBuildConfig&) const = default;
};
// all_reduce_add_rms_norm.
struct AllReduceAddRmsNormBuildConfig {
  int threads_per_block;
  int waves_per_eu;
  int tile_m;
  int tile_n;
  bool operator==(const AllReduceAddRmsNormBuildConfig&) const = default;
};
// all_reduce_add_attn_res_rms_norm: TILE_K sources a step.
struct AllReduceAddAttnResRmsNormBuildConfig {
  int threads_per_block;
  int waves_per_eu;
  int tile_m;
  int tile_n;
  int tile_k;
  bool operator==(const AllReduceAddAttnResRmsNormBuildConfig&) const = default;
};
using BuildConfig =
    std::variant<AllReduceBuildConfig, AllReduceRmsNormBuildConfig, AllReduceAddRmsNormBuildConfig,
                 AllReduceAddAttnResRmsNormBuildConfig>;

// A KERNEL'S LAUNCH: its grid, never compiled in, and the blocks of it the fused AttnRes pull
// two-shot gives its reduce-scatter (its reads queue behind the links past a few dozen; AttnRes
// after it on every block). reduce_scatter_blocks 0 is no split, every block both: the only value
// any other kernel takes.
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



constexpr int elem_bytes(DType d) { return d == DType::f32 ? 4 : 2; }
// A row of `elems` of `dtype` in 16-byte packs.
constexpr int64_t packs_of(int64_t elems, DType dtype) {
  return elems * elem_bytes(dtype) / kBuild.memory.pack_bytes;
}

// =================================================================================================
// EACH KERNEL'S SIGNATURE, element pointers as void* (a peer table's as void* const*): every
// compiled instance has one of these
// (select asserts it when it picks the instance), so a launch calls the kernel select chose
// through it, with the kernel's own arguments. A kernel's arguments are its buffers and values,
// in this order: every rank's input (a device table: a captured launch's are filled after the
// capture), scratch or staging, each only where the kernel uses it; the synchronization state
// (every rank's signal block, this rank's, its rank, the wait limit); then its own. EVERY BUFFER
// IS FOLLOWED BY ITS STRIDES, in elements, every dimension's (STANDARDS, *One file is the
// interface*): a [rows, cols] buffer's stride_m and stride_n, a vector's stride_n alone. A peer
// buffer is a device table of every rank's address (vLLM's RankData), its strides every rank's.
// =================================================================================================

// The plain all-reduce: out, inp_size_m, inp_size_n; staged, the staging first (and the two-shot's
// scratch before it), its own input after out, and the rows a band.
using AllReduceOneShotKernel =
    void (*)(const void* const*, int64_t, int64_t, Signal* const*, void*, int, uint64_t, void*,
          int64_t,
             int64_t, int, int);
using AllReduceTwoShotKernel =
    void (*)(const void* const*, int64_t, int64_t, void* const*, int64_t, int64_t, Signal* const*,
          void*, int,
             uint64_t, void*, int64_t, int64_t, int, int);
using AllReduceOneShotStagedKernel =
    void (*)(void* const*, int64_t, int64_t, Signal* const*, void*, int, uint64_t, void*, int64_t,
             int64_t, const void*, int64_t, int64_t, int, int, int);
using AllReduceTwoShotStagedKernel =
    void (*)(void* const*, int64_t, int64_t, void* const*, int64_t, int64_t, Signal* const*, void*,
          int,
             uint64_t, void*, int64_t, int64_t, const void*, int64_t, int64_t, int, int, int);
// out, weight, eps, inp_size_m, inp_size_n; the two-shots (pull and push) with every rank's
// scratch.
using AllReduceRmsNormOneShotKernel =
    void (*)(const void* const*, int64_t, int64_t, Signal* const*, void*, int, uint64_t, void*,
          int64_t,
             int64_t, const void*, int64_t, float, int, int);
using AllReduceRmsNormTwoShotKernel =
    void (*)(const void* const*, int64_t, int64_t, void* const*, int64_t, int64_t, Signal* const*,
          void*, int,
             uint64_t, void*, int64_t, int64_t, const void*, int64_t, float, int, int);
// out, residual_out, residual, weight, eps, inp_size_m, inp_size_n.
using AllReduceAddRmsNormOneShotKernel =
    void (*)(const void* const*, int64_t, int64_t, Signal* const*, void*, int, uint64_t, void*,
          int64_t,
             int64_t, void*, int64_t, int64_t, const void*, int64_t, int64_t, const void*,
             int64_t, float, int, int);
using AllReduceAddRmsNormTwoShotKernel =
    void (*)(const void* const*, int64_t, int64_t, void* const*, int64_t, int64_t, Signal* const*,
          void*, int,
             uint64_t, void*, int64_t, int64_t, void*, int64_t, int64_t, const void*, int64_t,
             int64_t, const void*, int64_t, float, int, int);
// prefix, blocks (m, r, n), norm_w, qk_w, out_norm_w, out, num_blocks, write_idx, eps, out_eps,
// inp_size_m, inp_size_n; the push with every rank's scratch; the pull with it and its
// reduce_scatter_blocks last.
#define HIP_CUSTOM_COMMS_ATTN_RES_ARGS                                                                  \
  void*, int64_t, int64_t, void*, int64_t, int64_t, int64_t, const void*, int64_t, const void*, \
      int64_t, const void*, int64_t, void*, int64_t, int64_t, int, int, float, float, int, int
using AllReduceAddAttnResRmsNormOneShotKernel =
    void (*)(const void* const*, int64_t, int64_t, Signal* const*, void*, int, uint64_t,
             HIP_CUSTOM_COMMS_ATTN_RES_ARGS);
using AllReduceAddAttnResRmsNormPushKernel =
    void (*)(const void* const*, int64_t, int64_t, void* const*, int64_t, int64_t, Signal* const*,
          void*, int,
             uint64_t, HIP_CUSTOM_COMMS_ATTN_RES_ARGS);
using AllReduceAddAttnResRmsNormPullKernel =
    void (*)(const void* const*, int64_t, int64_t, void* const*, int64_t, int64_t, Signal* const*,
          void*, int,
             uint64_t, HIP_CUSTOM_COMMS_ATTN_RES_ARGS, int);
#undef HIP_CUSTOM_COMMS_ATTN_RES_ARGS

// =================================================================================================
// EACH OP'S LAUNCH: the kernel select decided, and every argument it runs with. Which kernel (the
// compiled instance, `kernel`, one of its op's signatures above; and what it is: its algorithm,
// direction and world), what is compiled in (the tile, threads_per_block), the grid
// (blocks_per_grid, and the pull's reduce_scatter_blocks), the stream (select decides on it: a
// stream being captured reads its input in place), then the kernel's arguments.
// =================================================================================================

// `staged`: the build that copies an eager input through the staging a pass at a time, not the
// one that reads a registered or captured input in place.
struct AllReduceLaunch {
  const void* kernel;
  Algorithm algorithm;
  Direction direction;
  int world;
  int tile_m;
  int tile_n;
  int threads_per_block;
  int blocks_per_grid;
  int waves_per_eu;
  bool staged;
  hipStream_t stream;
  void* out;
  int64_t out_stride_m, out_stride_n;
  const void* inp;
  int64_t inp_stride_m, inp_stride_n;
  int64_t scratch_stride_m, scratch_stride_n;  // the two-shot's: a row of a rank's slice a row
  int64_t staging_stride_m, staging_stride_n;  // the staged builds': a band dense
  int band_m;                               // the staged builds': the rows a pass
  DType dtype;
  int64_t inp_size_m;
  int64_t inp_size_n;
};

// `weight_dtype`: dtype, or f32.
struct AllReduceRmsNormLaunch {
  const void* kernel;
  Algorithm algorithm;
  Direction direction;
  int world;
  int tile_m;
  int tile_n;
  int threads_per_block;
  int blocks_per_grid;
  int waves_per_eu;
  hipStream_t stream;
  void* out;
  int64_t out_stride_m, out_stride_n;
  const void* inp;
  int64_t inp_stride_m, inp_stride_n;
  const void* weight;
  int64_t weight_stride_n;
  int64_t scratch_stride_m, scratch_stride_n;  // the two-shots': a row of hidden a row
  DType dtype;
  DType weight_dtype;
  int64_t inp_size_m;
  int64_t inp_size_n;
  float eps;
};

struct AllReduceAddRmsNormLaunch {
  const void* kernel;
  Algorithm algorithm;
  Direction direction;
  int world;
  int tile_m;
  int tile_n;
  int threads_per_block;
  int blocks_per_grid;
  int waves_per_eu;
  hipStream_t stream;
  void* out;
  int64_t out_stride_m, out_stride_n;
  void* residual_out;
  int64_t residual_out_stride_m, residual_out_stride_n;
  const void* inp;
  int64_t inp_stride_m, inp_stride_n;
  const void* residual;
  int64_t residual_stride_m, residual_stride_n;
  const void* weight;
  int64_t weight_stride_n;
  int64_t scratch_stride_m, scratch_stride_n;  // the two-shots': a row of hidden a row
  DType dtype;
  DType weight_dtype;
  int64_t inp_size_m;
  int64_t inp_size_n;
  float eps;
};

// tile_m 1 on the one-shot and push (a row a tile); reduce_scatter_blocks the pull two-shot's, 0
// on the others. `blocks` is [rows, sources, hidden] at its strides in elements; `write_idx` < 0
// writes no block; `out_norm_weight` null: no output norm.
struct AllReduceAddAttnResRmsNormLaunch {
  const void* kernel;
  Algorithm algorithm;
  Direction direction;
  int world;
  int tile_m;
  int tile_n;
  int tile_k;
  int threads_per_block;
  int blocks_per_grid;
  int waves_per_eu;
  int reduce_scatter_blocks;
  hipStream_t stream;
  void* prefix;
  int64_t prefix_stride_m, prefix_stride_n;
  void* out;
  int64_t out_stride_m, out_stride_n;
  const void* inp;
  int64_t inp_stride_m, inp_stride_n;
  void* blocks;
  int64_t blocks_stride_m, blocks_stride_r, blocks_stride_n;
  const void* norm_weight;
  int64_t norm_weight_stride_n;
  const void* qk_weight;
  int64_t qk_weight_stride_n;
  const void* out_norm_weight;
  int64_t out_norm_weight_stride_n;
  int64_t scratch_stride_m, scratch_stride_n;  // the two-shots': a row of hidden a row
  DType dtype;
  int64_t inp_size_m;
  int64_t inp_size_n;
  int num_blocks;
  int write_idx;
  float eps;
  float out_eps;
  bool has_prefix;
};

}  // namespace hip_custom_comms
