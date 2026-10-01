// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ROCM_COMMS, THE ONE INTERFACE: our collectives, torch-free. Every op is
// op(handle, args, options), and every op is the same three steps:
//   select(handle, args, options) -> KernelSpec   the only choice, as data (impl/select.cuh)
//   validate(handle, spec, args, options)         the only no: raises with the reason
//   launch(handle, spec, args, stream)            runs it; decides nothing (impl/launch.cuh)
// Both of the last two find the compiled kernel a spec names the same way (impl/instances.cuh).
//
// Handle                  the state across calls: the peers' memory, mapped once (p2p's Group)
// Input                   the workload a call is: its shape and group (machine/ says the rest)
// AllReduceArgs, NormArgs, AttnResArgs, GemmTailArgs   one op's inputs and outputs
// Options                 how the caller wants it run: precision, a forced kernel, the stream
// KernelSpec              what runs: a kernel, its grid and block, its row build
// all_reduce, all_reduce_rms_norm (and _add_), all_reduce_add_attn_res_rms_norm,
// all_reduce_rms_norm_gemm(_add)      the ops
// why_not(handle, op, input, options)  why a call cannot run, or empty: `admits` for vLLM

#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <optional>
#include <string>

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
};

// Every `__global__` there is, named by its shot and what it fuses.
enum class Kernel : int {
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
};

// SELECT'S RESULT: which compiled kernel runs and how. With the call's own facts (its world,
// dtype and variant) it names exactly one instance.
struct KernelSpec {
  Kernel kernel;
  int grid;
  int threads;
  int row_packs;  // a row kernel's packs of a row a thread holds, its build; 0 for the others
};

// A kernel the caller forces, at its grid and block (the bench's sweeps); select derives the rest.
struct Forced {
  Kernel kernel;
  int grid;
  int threads;
};

struct Options {
  int quant_bits;                // the precision accepted on the wire: 16 (exact), 8 or 4
  std::optional<Forced> forced;  // none: select's
  hipStream_t stream;
};

struct Input {
  int64_t rows;    // tokens
  int64_t hidden;  // a row's length, in elements
  int elem_bytes;
  int64_t cols;    // the GEMM tail's output columns; 0 for every other op
  int world;
};

struct AllReduceArgs {
  void* out;
  const void* inp;
  int64_t bytes;
  DType dtype;
};

// residual and residual_out null: rms_norm; given: fused_add_rms_norm.
struct NormArgs {
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

}  // namespace hip_comms

#define HIP_COMMS_INTERFACE
#include "impl/kernels.cuh"
#include "impl/select.cuh"
#include "impl/instances.cuh"
#include "impl/validate.cuh"
#include "impl/launch.cuh"
#include "impl/ops.cuh"
#undef HIP_COMMS_INTERFACE
