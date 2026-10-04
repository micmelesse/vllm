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

#include "common/common.cuh"

namespace hip_comms {

// WHY A CALL CANNOT RUN, every reason there is. The numbers cross to Python (rocm_comms.Error), so
// a reason is only ever added at the end. `disabled` and `no_such_op` are the communicator's own.
enum class Error : int {
  disabled = 0,
  no_such_op = 1,
  not_contiguous = 2,
  not_two_d = 3,
  output_not_two_d = 4,
  dtype_not_built = 5,
  world_not_built = 6,
  row_not_packs = 7,
  widths_not_packs = 8,
  row_not_wider_than_output = 9,
  template_not_this_ops = 10,
  row_too_wide = 11,
  block_not_a_wave_per_peer = 12,
  block_exceeds_lds = 13,
  scratch_too_small = 14,
  grid_not_resident = 15,
  staging_too_small = 16,
  device_not_built = 17,
  device_not_tuned = 18,
  weight_not_built = 19,
  no_such_template = 20,
  no_such_group = 21,
  ranks_disagree = 22,
  groups_disagree = 23,
  threads_not_built = 24,
  tile_not_built = 25,
  direction_without_algorithm = 26,
  config_without_algorithm = 27,
  launch_incomplete = 28,
  launch_out_of_range = 29,
  field_without_launch = 30,
  field_not_positive = 31,
  field_not_this_templates = 32,
  probe_out_of_range = 33,
  waves_not_built = 34,
};
constexpr int kNumErrors = 35;

constexpr const char* to_string(Error e) {
  switch (e) {
    case Error::disabled: return "disabled: the communicator is disabled";
    case Error::no_such_op: return "no_such_op: the backend has no such op";
    case Error::not_contiguous: return "not_contiguous: the input is not contiguous";
    case Error::not_two_d: return "not_two_d: a fused op takes a 2-D input";
    case Error::output_not_two_d: return "output_not_two_d: the output is not 2-D";
    case Error::dtype_not_built: return "dtype_not_built: only float16 and bfloat16 are built";
    case Error::world_not_built: return "world_not_built: the world size is not 2, 4 or 8";
    case Error::row_not_packs: return "row_not_packs: the row is not whole 16-byte packs";
    case Error::widths_not_packs:
      return "widths_not_packs: the output's and the latent's widths are not whole packs";
    case Error::row_not_wider_than_output:
      return "row_not_wider_than_output: the input's row is not wider than twice the output's";
    case Error::template_not_this_ops:
      return "template_not_this_ops: the forced template is not this op's";
    case Error::row_too_wide:
      return "row_too_wide: the row is wider than the template's widest build holds";
    case Error::block_not_a_wave_per_peer:
      return "block_not_a_wave_per_peer: a two-shot block must be one wave per peer";
    case Error::block_exceeds_lds:
      return "block_exceeds_lds: the GEMM tail's block exceeds what its LDS holds";
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
    case Error::direction_without_algorithm:
      return "direction_without_algorithm: a forced direction needs its algorithm";
    case Error::config_without_algorithm:
      return "config_without_algorithm: a forced config needs its algorithm where the op has "
             "more than one template";
    case Error::launch_incomplete:
      return "launch_incomplete: a forced launch gives threads_per_block and blocks_per_grid "
             "together";
    case Error::launch_out_of_range:
      return "launch_out_of_range: threads_per_block is whole waves up to the build's maximum, "
             "blocks_per_grid at least 1 and at most the resident maximum";
    case Error::field_without_launch:
      return "field_without_launch: a forced config field needs its launch";
    case Error::field_not_positive:
      return "field_not_positive: a forced config field is positive";
    case Error::field_not_this_templates:
      return "field_not_this_templates: the forced template has no such config field";
    case Error::probe_out_of_range:
      return "probe_out_of_range: a probe's peer, iterations, bytes or grid is out of range";
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

// WHAT THE PROBE'S LINK TRAFFIC MOVES: pulled from the peers, pushed into them, both at once with
// the blocks split between the two, or both from every block.
enum class Traffic : int { pull = 0, push = 1, split = 2, each = 3 };

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

// HOW A KERNEL IS LAUNCHED: its block's threads (Triton's num_warps x 64), its grid, and the waves
// a SIMD (an EU) the compiler fits its registers to (__launch_bounds__'s second value, Triton's
// waves_per_eu; 1 asks nothing). Every kernel has one; the grid is the launch's alone, never
// compiled in.
struct LaunchConfig {
  int threads_per_block;
  int blocks_per_grid;
  int waves_per_eu;
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
template <size_t FAMILY_INDEX = 0>
constexpr KernelConfig zero_config(size_t family) {
  if constexpr (FAMILY_INDEX + 1 < std::variant_size_v<KernelConfig>)
    if (family != FAMILY_INDEX) return zero_config<FAMILY_INDEX + 1>(family);
  return KernelConfig{std::in_place_index<FAMILY_INDEX>};
}



constexpr int elem_bytes(DType d) { return d == DType::f32 ? 4 : 2; }
// A row of `elems` of `dtype` in 16-byte packs.
constexpr int64_t packs_of(int64_t elems, DType dtype) {
  return elems * elem_bytes(dtype) / kBuild.memory.pack_bytes;
}

// =================================================================================================
// EACH KERNEL'S SIGNATURE, element pointers as void*: every compiled instance has one of these
// (select asserts it when it picks the instance), so a launch calls the kernel select chose
// through it, with the kernel's own arguments. A kernel's arguments are its buffers and values,
// in this order: every rank's input (a device table: a captured launch's are filled after the
// capture), scratch or staging, each only where the kernel uses it; the synchronization state
// (every rank's signal block, this rank's, its rank, the wait limit); then its own.
// =================================================================================================

// The plain all-reduce: out and its packs; staged, its own input and the packs a staging holds,
// the two-shot the packs a scratch holds before them.
using AllReduceOneShotKernel =
    void (*)(const void*, PeerSignals, void*, int, uint64_t, void*, int);
using AllReduceTwoShotKernel =
    void (*)(const void*, PeerPtrs, PeerSignals, void*, int, uint64_t, void*, int);
using AllReduceOneShotStagedKernel =
    void (*)(PeerPtrs, PeerSignals, void*, int, uint64_t, void*, int64_t, const void*,
             int64_t);
using AllReduceTwoShotStagedKernel =
    void (*)(PeerPtrs, PeerPtrs, PeerSignals, void*, int, uint64_t, int64_t, void*,
             int64_t, const void*, int64_t);
// out, weight, eps, rows, packs; the two-shots (pull and push) with every rank's scratch.
using AllReduceRmsNormOneShotKernel =
    void (*)(const void*, PeerSignals, void*, int, uint64_t, void*, const void*, float, int,
             int);
using AllReduceRmsNormTwoShotKernel =
    void (*)(const void*, PeerPtrs, PeerSignals, void*, int, uint64_t, void*, const void*,
             float, int, int);
// out, residual_out, residual, weight, eps, rows, packs.
using AllReduceAddRmsNormOneShotKernel =
    void (*)(const void*, PeerSignals, void*, int, uint64_t, void*, void*, const void*,
             const void*, float, int, int);
using AllReduceAddRmsNormTwoShotKernel =
    void (*)(const void*, PeerPtrs, PeerSignals, void*, int, uint64_t, void*, void*,
             const void*, const void*, float, int, int);
// prefix, blocks, block_stride_m, block_stride_r, norm_w, qk_w, out_norm_w, out, num_blocks,
// write_idx, eps, out_eps, rows, packs; the push with every rank's scratch; the pull with it and
// its reduce_scatter_blocks last.
using AllReduceAddAttnResRmsNormOneShotKernel =
    void (*)(const void*, PeerSignals, void*, int, uint64_t, void*, void*, int64_t, int64_t,
             const void*, const void*, const void*, void*, int, int, float, float, int, int);
using AllReduceAddAttnResRmsNormPushKernel =
    void (*)(const void*, PeerPtrs, PeerSignals, void*, int, uint64_t, void*, void*,
             int64_t, int64_t, const void*, const void*, const void*, void*, int, int, float, float,
             int, int);
using AllReduceAddAttnResRmsNormPullKernel =
    void (*)(const void*, PeerPtrs, PeerSignals, void*, int, uint64_t, void*, void*,
             int64_t, int64_t, const void*, const void*, const void*, void*, int, int, float, float,
             int, int, int);
// norm_w, eps, gemm_w, n_cols, out, out_stride, workspace, rows, packs: written or added
// (all_reduce_rms_norm_gemm and _gemm_add), one shape.
using AllReduceRmsNormGemmOneShotKernel =
    void (*)(const void*, PeerSignals, void*, int, uint64_t, const void*, float, const void*,
             int, void*, int64_t, void*, int, int);
using AllReduceRmsNormGemmTwoShotKernel =
    void (*)(const void*, PeerPtrs, PeerSignals, void*, int, uint64_t, const void*, float,
             const void*, int, void*, int64_t, void*, int, int);
// out, eps, rows, hidden_packs, latent_packs.
using AllReduceRmsScaleAddOneShotKernel =
    void (*)(const void*, PeerSignals, void*, int, uint64_t, void*, float, int, int, int);
using AllReduceRmsScaleAddTwoShotKernel =
    void (*)(const void*, PeerPtrs, PeerSignals, void*, int, uint64_t, void*, float, int,
             int, int);
// Experimental, no peers: prefix, delta, blocks, block_stride_m, block_stride_r, norm_w, qk_w,
// out_norm_w, out, num_blocks, write_idx, eps, out_eps, rows, packs.
using AddAttnResRmsNormKernel =
    void (*)(void*, const void*, void*, int64_t, int64_t, const void*, const void*, const void*,
             void*, int, int, float, float, int, int);
// The probe's: nothing of its own; peer, flag base, iterations, ticks; every rank's input (the
// streamed buffer's) and staging, then mode, peer, pullers, packs, sink.
using ProbeBarrierKernel = void (*)(PeerSignals, void*, int, uint64_t);
using PingPongKernel = void (*)(PeerSignals, void*, int, uint64_t, int, uint32_t, int, void*);
using LinkTrafficKernel =
    void (*)(const void*, PeerPtrs, PeerSignals, void*, int, uint64_t, int, int, int,
             int64_t, void*);


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
  int threads_per_block;
  int blocks_per_grid;
  int waves_per_eu;
  bool staged;
  hipStream_t stream;
  void* out;
  const void* inp;
  int64_t bytes;
  DType dtype;
};

// `weight_dtype`: dtype, or f32.
struct AllReduceRmsNormLaunch {
  const void* kernel;
  Algorithm algorithm;
  Direction direction;
  int world;
  int tile_n;
  int threads_per_block;
  int blocks_per_grid;
  int waves_per_eu;
  hipStream_t stream;
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
  const void* kernel;
  Algorithm algorithm;
  Direction direction;
  int world;
  int tile_n;
  int threads_per_block;
  int blocks_per_grid;
  int waves_per_eu;
  hipStream_t stream;
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
  const void* kernel;
  Algorithm algorithm;
  Direction direction;
  int world;
  int tile_m;
  int tile_n;
  int tile_k;
  int slice_k;
  int threads_per_block;
  int blocks_per_grid;
  int waves_per_eu;
  hipStream_t stream;
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
  const void* kernel;
  Algorithm algorithm;
  Direction direction;
  int world;
  int tile_m;
  int tile_n;
  int tile_k;
  int slice_k;
  int threads_per_block;
  int blocks_per_grid;
  int waves_per_eu;
  hipStream_t stream;
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
  const void* kernel;
  Algorithm algorithm;
  Direction direction;
  int world;
  int tile_n;
  int threads_per_block;
  int blocks_per_grid;
  int waves_per_eu;
  hipStream_t stream;
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
  const void* kernel;
  Algorithm algorithm;
  Direction direction;
  int world;
  int tile_n;
  int tile_k;
  int threads_per_block;
  int blocks_per_grid;
  int waves_per_eu;
  hipStream_t stream;
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

// THE PROBE'S, experimental: one block a rank, so the next measurement starts on every rank
// together; a flag to `peer` and back `iters` times, the device clock ticks written to `ticks`;
// every thread streaming `bytes` of `buffer` (null: the staging) by `mode`, the first `pullers`
// blocks pulling where the mode is split, the pulled packs folded into `sink`.
struct ProbeBarrierLaunch {
  const void* kernel;
  int world;
  int threads_per_block;
  int blocks_per_grid;
  hipStream_t stream;
};

struct PingPongLaunch {
  const void* kernel;
  int world;
  int threads_per_block;
  int blocks_per_grid;
  hipStream_t stream;
  int peer;
  int iters;
  void* ticks;
};

struct LinkTrafficLaunch {
  const void* kernel;
  int world;
  int threads_per_block;
  int blocks_per_grid;
  hipStream_t stream;
  const void* buffer;
  int64_t bytes;
  Traffic mode;
  int peer;
  int pullers;
  void* sink;
};

}  // namespace hip_comms
