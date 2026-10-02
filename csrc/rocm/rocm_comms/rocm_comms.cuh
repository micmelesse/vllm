// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ROCM_COMMS, THE ONE INTERFACE: our collectives, torch-free. An OP is what a caller asks for (an
// API call); a KERNEL is what runs, one compiled instruction sequence (vllm CONTEXT's lingo). Every
// op is op(handle, args, options), two steps:
//   plan(handle, args, options) -> Kernel or Error   the only decision: select's kernel (its
//                                     template, arguments and launch, impl/select.cuh), or the
//                                     first Error it meets here (impl/check.cuh)
//   launch(handle, kernel, args, stream)  runs it; decides nothing (impl/launch.cuh)
// An op returns the kernel it launched, or the Error and launches nothing. check and launch find
// the compiled function a Kernel names the same way (impl/dispatch.cuh).
//
// Handle (handle.cuh)     the runtime state: the peers' memory, mapped once; kBuild's runtime twin
// AllReduceArgs, NormArgs, AttnResArgs, GemmTailArgs, ScaleAddArgs   one op's call
// Options                 how the caller wants it run: precision, a forced template, the stream
// Kernel                  what runs: a template, its arguments, its grid and block
// all_reduce, all_reduce_rms_norm (and _add_), all_reduce_add_attn_res_rms_norm,
// all_reduce_rms_norm_gemm(_add), all_reduce_rms_scale_add      the ops
// Error, to_string(Error)  why a call cannot run here: every reason, one list
// supported(device, world)  Supported or the Error: whether the library runs on a device and world
// kBuild (build.cuh)  the build's facts: what is compiled, the memory, the kernels'
// geometry

#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>

#include "p2p/p2p.cuh"
#include "build.cuh"
#include "handle.cuh"
#include "machine/hardware.cuh"

namespace hip_comms {


template <typename T, size_t N>
constexpr bool built_in(const std::array<T, N>& built, T x) {
  for (const T& b : built)
    if (b == x) return true;
  return false;
}
constexpr bool dtype_built(DType d) { return built_in(kBuild.supports.dtypes, d); }
constexpr bool world_built(int world) { return built_in(kBuild.supports.worlds, world); }

// What the caller asked for: an all-reduce, alone or with what it fuses, as Python names them.
enum class Op : int {
  all_reduce                       = 0,
  all_reduce_rms_norm              = 1,
  all_reduce_add_rms_norm          = 2,
  all_reduce_add_attn_res_rms_norm = 3,
  all_reduce_rms_norm_gemm_add     = 4,
  all_reduce_rms_norm_gemm         = 5,
  all_reduce_rms_scale_add         = 6,
};

// AN OP'S NAME FROM ITS ENUM TOKEN, as Python names it.
#define HIP_COMMS_CASE(o) \
  case Op::o: return #o;
constexpr const char* to_string(Op op) {
  switch (op) {
    HIP_COMMS_CASE(all_reduce)
    HIP_COMMS_CASE(all_reduce_rms_norm)
    HIP_COMMS_CASE(all_reduce_add_rms_norm)
    HIP_COMMS_CASE(all_reduce_add_attn_res_rms_norm)
    HIP_COMMS_CASE(all_reduce_rms_norm_gemm_add)
    HIP_COMMS_CASE(all_reduce_rms_norm_gemm)
    HIP_COMMS_CASE(all_reduce_rms_scale_add)
  }
  return "unknown";
}
#undef HIP_COMMS_CASE
constexpr int kNumOps = static_cast<int>(Op::all_reduce_rms_scale_add) + 1;

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
// `splits`: the slices of a row's hidden, a block each.
struct ScaleAddTemplateArgs {
  int world;
  DType dtype;
  int splits;
};
using TemplateArgs = std::variant<AllReduceTemplateArgs, NormTemplateArgs, AttnResTemplateArgs,
                                  GemmTemplateArgs, ScaleAddTemplateArgs>;

// HOW A KERNEL IS TILED AND LAUNCHED, Triton's autotune config: the tile (TILE_M rows x TILE_N
// columns in elements, tile_n 0 when no build holds the call's row, which check refuses; TILE_K of
// the reduced dimension, the GEMM's K a pass or AttnRes's sources a step; SLICE_K lanes splitting
// one output's K, CUTLASS's sliced-K), the block's threads (Triton's num_warps x 64) and the grid.
// A row kernel compiles all but the grid in; a field its kernel does not have is 0, and the plain
// all-reduce has no tile.
struct KernelConfig {
  int tile_m;
  int tile_n;
  int tile_k;
  int slice_k;
  int threads_per_block;
  int blocks_per_grid;
};

// WHAT SELECT RETURNS: one kernel, the template with its arguments decided (the compiled
// instruction sequence), and its KernelConfig.
struct Kernel {
  Template fn;
  TemplateArgs args;
  KernelConfig config;
};

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
  quantized_not_built = 13,
  block_exceeds_lds = 14,
  scratch_too_small = 15,
  grid_not_resident = 16,
  staging_too_small = 17,
  device_not_built = 18,
  device_not_tuned = 19,
  weight_not_built = 20,
  no_such_template = 21,
  no_such_group = 22,
  ranks_disagree = 23,
  groups_disagree = 24,
  threads_not_built = 25,
  tile_not_built = 26,
};
constexpr int kNumErrors = 27;

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
    case Error::quantized_not_built: return "quantized_not_built: no kernel quantizes yet";
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
      return "no_such_template: the forced template's name names none";
    case Error::no_such_group: return "no_such_group: no process group is registered by that name";
    case Error::ranks_disagree:
      return "ranks_disagree: the ranks captured different numbers of buffers";
    case Error::groups_disagree:
      return "groups_disagree: the CPU and device groups differ in size or in this rank";
    case Error::threads_not_built:
      return "threads_not_built: no build of the template runs at that block size";
    case Error::tile_not_built:
      return "tile_not_built: no build of the template has that TILE_M or TILE_N";
  }
  return "unknown";
}

// A LOSSY PRECISION on the wire, in bits; none is exact.
enum class QuantBits : int { eight = 8, four = 4 };

// HOW THE CALLER WANTS A CALL RUN: a lossy precision, and Kernel's own choices forced (a sweep's, a
// tuner's): its template (at its own default config), and with it its KernelConfig, where a zero
// tile_m or tile_n is the template's own tile for the call. The call's facts (TemplateArgs) are
// always select's, from the call.
struct Options {
  std::optional<QuantBits> quant_bits;  // none: exact
  std::optional<Template> fn;           // none: select's
  std::optional<KernelConfig> kernel_config;  // none: select's; only with fn
  hipStream_t stream;
};

struct AllReduceArgs {
  void* out;
  const void* inp;
  int64_t bytes;
  DType dtype;
};

// add: fused_add_rms_norm, with residual and residual_out (written); otherwise rms_norm, the two
// null.
struct NormArgs {
  bool add;
  void* out;
  const void* inp;
  const void* weight;
  DType dtype;
  DType weight_dtype;  // dtype, or f32
  int64_t rows;
  int64_t hidden;
  float eps;
  void* residual_out;
  const void* residual;
};

struct AttnResArgs {
  void* prefix;
  void* out;
  const void* inp;
  void* blocks;  // [rows, sources, hidden]
  int64_t block_stride_m;
  int64_t block_stride_r;
  const void* norm_weight;
  const void* qk_weight;
  const void* out_norm_weight;  // null: none
  DType dtype;
  int64_t rows;
  int64_t hidden;
  int num_blocks;
  int write_idx;
  float eps;
  float out_eps;
  bool has_prefix;
};

// add: out += the product (rms_norm_gemm_add, Kimi-K3's latent tail); otherwise it is written
// (rms_norm_gemm). out is [rows, n_cols] at out_stride, so a column slice of a wider buffer.
struct GemmTailArgs {
  bool add;
  void* out;
  int64_t out_stride;
  const void* inp;
  const void* norm_weight;
  float eps;
  const void* gemm_weight;  // [n_cols, hidden]
  int64_t n_cols;
  void* workspace;
  DType dtype;
  int64_t rows;
  int64_t hidden;
};

// WHAT THE LIBRARY RUNS ON, once `supported` finds it can: the device's arch, as HIP names it.
struct Supported {
  std::string arch;
};

// inp is [rows, 2 * hidden + latent], [shared | projected | latent]; out [rows, hidden] = shared +
// projected * rsqrt(mean(latent^2) + eps), all three summed over the ranks first.
struct ScaleAddArgs {
  void* out;
  const void* inp;
  DType dtype;
  int64_t rows;
  int64_t hidden;
  int64_t latent;
  float eps;
};

}  // namespace hip_comms

#define HIP_COMMS_INTERFACE
#include "impl/templates.cuh"
#include "impl/select.cuh"
#include "impl/dispatch.cuh"
#include "impl/check.cuh"
#include "impl/supported.cuh"
#include "impl/launch.cuh"
#include "impl/ops.cuh"
#undef HIP_COMMS_INTERFACE
