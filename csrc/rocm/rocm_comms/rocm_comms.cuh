// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ROCM_COMMS, THE ONE INTERFACE: our collectives, torch-free. An OP is what a caller asks for (an
// API call); a KERNEL is what runs, one compiled instruction sequence (vllm CONTEXT's lingo). Every
// op is op(handle, args, options), the same three steps:
//   select(args, world, options) -> Kernel   the only choice: the template, its arguments, its
//                                            launch (impl/select.cuh)
//   validate(handle, kernel, args, options)  the only no: raises with the reason
//   launch(handle, kernel, args, stream)     runs it; decides nothing (impl/launch.cuh)
// Both of the last two find the compiled function a Kernel names the same way (impl/dispatch.cuh).
//
// Handle                  the state across calls: the peers' memory, mapped once (p2p's Group)
// AllReduceArgs, NormArgs, AttnResArgs, GemmTailArgs, ScaleAddArgs   one op's call
// Options                 how the caller wants it run: precision, a forced template, the stream
// Kernel                  what runs: a template, its arguments, its grid and block
// all_reduce, all_reduce_rms_norm (and _add_), all_reduce_add_attn_res_rms_norm,
// all_reduce_rms_norm_gemm(_add), all_reduce_rms_scale_add      the ops
// why_not(handle, args, options)  why a call cannot run, or empty: `admits` for vLLM

#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <optional>
#include <string>
#include <variant>

#include "p2p/p2p.cuh"
#include "machine/build.cuh"
#include "machine/hardware.cuh"

namespace hip_comms {

using Handle = p2p::host::Group;

enum class DType { f16, bf16, f32 };

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
};

// A TEMPLATE'S ARGUMENTS, one struct per family: only the parameters that family has. `row_packs`
// is the packs of a row a thread holds, its row build: none when no build holds the call's row
// (validate refuses it).
struct AllReduceTemplateArgs {
  int world;
  DType dtype;
};
struct NormTemplateArgs {
  int world;
  DType dtype;
  DType weight;  // dtype, or f32
  std::optional<int> row_packs;
};
struct AttnResTemplateArgs {
  int world;
  DType dtype;
  std::optional<int> row_packs;
  bool prefix;
};
struct GemmTemplateArgs {
  int world;
  DType dtype;
  int lanes;  // grid_gemm's lanes a column
  std::optional<int> row_packs;
};
// `splits`: the slices of a row's hidden, a block each.
struct ScaleAddTemplateArgs {
  int world;
  DType dtype;
  std::optional<int> row_packs;
  int splits;
};
using TemplateArgs = std::variant<AllReduceTemplateArgs, NormTemplateArgs, AttnResTemplateArgs,
                                  GemmTemplateArgs, ScaleAddTemplateArgs>;

// WHAT SELECT RETURNS: one kernel, the template with its arguments decided (the compiled
// instruction sequence), and its launch.
struct Kernel {
  Template fn;
  TemplateArgs args;
  int grid;
  int threads;
};

// A template the caller forces, at its grid and block (the bench's sweeps); select decides its
// arguments from the call as for its own choice.
struct Forced {
  Template fn;
  int grid;
  int threads;
};

struct Options {
  int quant_bits;                // the precision accepted on the wire: 16 (exact), 8 or 4
  std::optional<Forced> forced;  // none: select's
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

// add: out[:, col0:col0+N] += the product (rms_norm_gemm_add, Kimi-K3's latent tail); otherwise
// it is written (rms_norm_gemm).
struct GemmTailArgs {
  bool add;
  void* out;
  int64_t out_stride;
  int out_col0;
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
#include "impl/validate.cuh"
#include "impl/launch.cuh"
#include "impl/ops.cuh"
#undef HIP_COMMS_INTERFACE
