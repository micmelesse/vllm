// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE CALL: what a caller passes an op, torch-free. One Args struct per op family (the torch
// boundary fills it from the tensors: their pointers, dtype and shape), and the Options it runs
// under.

#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <optional>

#include "build.cuh"
#include "kernel.cuh"

namespace hip_comms {

// HOW THE CALLER WANTS A CALL RUN: its stream, and Kernel's own choices forced (a sweep's, a
// tuner's): its template (at its own default config), and with it its KernelConfig (its family's),
// where a zero field is the template's own for the call. The call's facts (TemplateArgs) are
// always select's, from the call.
struct Options {
  std::optional<Template> fn;                 // none: select's
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
