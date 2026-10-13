// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE API: every op a caller can run, and nothing else: each is select_<op> (select.cuh: the op's
// launch, everything decided, or the first Error the call meets) then launch_<op> (launch.cuh,
// which decides nothing), and returns the launch it ran or the Error. Every argument is a
// primitive: pointers, sizes, a DType, and the op's own forcing (an Algorithm and a Direction,
// then its config fields, each none for select's). Python's `should` reads its plain call
// from kOps (build_info).

#pragma once

#ifndef HIP_CUSTOM_COMMUNICATOR_INTERFACE
#error "include hip_custom_communicator.cuh, the one interface, not its parts"
#endif

#include <hip/hip_runtime.h>

#include <cstdint>
#include <optional>
#include <variant>

namespace hip_custom_communicator {

// =================================================================================================
// all_reduce: the sum of every rank's `input` into `output`, each at its own strides and sizes.
// =================================================================================================

inline std::variant<AllReduceLaunch, Error> all_reduce(
    Handle& h, const void* input, int64_t input_stride_m, int64_t input_stride_n, int64_t input_size_m,
    int64_t input_size_n, void* output, int64_t output_stride_m, int64_t output_stride_n,
    int64_t output_size_m, int64_t output_size_n, DType dtype,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_m, std::optional<int> tile_n, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, std::optional<int> waves_per_eu, hipStream_t stream) {
  if (const std::optional<Error> e = layout_refused(
          dtype, {{output_stride_m, output_stride_n}, {input_stride_m, input_stride_n}}))
    return *e;
  std::variant<AllReduceLaunch, Error> l = select_all_reduce(
      h, input, input_stride_m, input_stride_n, input_size_m, input_size_n, output, output_stride_m,
      output_stride_n, output_size_m, output_size_n, dtype,
      algorithm, direction, tile_m, tile_n, threads_per_block, blocks_per_grid, waves_per_eu,
      stream);
  if (const AllReduceLaunch* got = std::get_if<AllReduceLaunch>(&l))
    launch_all_reduce(h, *got);
  return l;
}

// =================================================================================================
// THE NORMS: output = rms_norm(all_reduce(input), weight), or with a residual vLLM's
// fused_add_rms_norm (output and residual_output), its roundings exactly. input [rows, hidden]; the
// weight `weight_dtype`, dtype or f32.
// =================================================================================================

inline std::variant<AllReduceRmsNormLaunch, Error> all_reduce_rms_norm(
    Handle& h, const void* input, int64_t input_stride_m, int64_t input_stride_n, int64_t input_size_m,
    int64_t input_size_n, const void* weight, int64_t weight_stride_n, int64_t weight_size_n,
    void* output, int64_t output_stride_m, int64_t output_stride_n, int64_t output_size_m,
    int64_t output_size_n, DType dtype, DType weight_dtype, float eps,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_m, std::optional<int> tile_n, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, std::optional<int> waves_per_eu, hipStream_t stream) {
  if (const std::optional<Error> e = layout_refused(
          dtype, {{output_stride_m, output_stride_n}, {input_stride_m, input_stride_n}}))
    return *e;
  if (const std::optional<Error> e = layout_refused(weight_dtype, {{0, weight_stride_n}}))
    return *e;
  std::variant<AllReduceRmsNormLaunch, Error> l = select_all_reduce_rms_norm(
      h, input, input_stride_m, input_stride_n, input_size_m, input_size_n, weight, weight_stride_n,
      weight_size_n, output, output_stride_m, output_stride_n, output_size_m, output_size_n, dtype,
      weight_dtype, eps,
      algorithm, direction, tile_m, tile_n,
      threads_per_block, blocks_per_grid, waves_per_eu, stream);
  if (const AllReduceRmsNormLaunch* got = std::get_if<AllReduceRmsNormLaunch>(&l))
    launch_all_reduce_rms_norm(h, *got);
  return l;
}

inline std::variant<AllReduceAddRmsNormLaunch, Error> all_reduce_add_rms_norm(
    Handle& h, const void* input, int64_t input_stride_m, int64_t input_stride_n, int64_t input_size_m,
    int64_t input_size_n, const void* residual, int64_t residual_stride_m,
    int64_t residual_stride_n, int64_t residual_size_m, int64_t residual_size_n,
    const void* weight, int64_t weight_stride_n, int64_t weight_size_n, void* output,
    int64_t output_stride_m, int64_t output_stride_n, int64_t output_size_m, int64_t output_size_n,
    void* residual_output, int64_t residual_output_stride_m, int64_t residual_output_stride_n,
    int64_t residual_output_size_m, int64_t residual_output_size_n, DType dtype,
    DType weight_dtype, float eps,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_m, std::optional<int> tile_n, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, std::optional<int> waves_per_eu, hipStream_t stream) {
  if (const std::optional<Error> e = layout_refused(
          dtype, {{output_stride_m, output_stride_n},
                  {residual_output_stride_m, residual_output_stride_n},
                  {input_stride_m, input_stride_n},
                  {residual_stride_m, residual_stride_n}}))
    return *e;
  if (const std::optional<Error> e = layout_refused(weight_dtype, {{0, weight_stride_n}}))
    return *e;
  std::variant<AllReduceAddRmsNormLaunch, Error> l = select_all_reduce_add_rms_norm(
      h, input, input_stride_m, input_stride_n, input_size_m, input_size_n, residual, residual_stride_m,
      residual_stride_n, residual_size_m, residual_size_n, weight, weight_stride_n, weight_size_n,
      output, output_stride_m, output_stride_n, output_size_m, output_size_n, residual_output,
      residual_output_stride_m, residual_output_stride_n, residual_output_size_m,
      residual_output_size_n, dtype, weight_dtype, eps,
      algorithm, direction, tile_m, tile_n, threads_per_block, blocks_per_grid, waves_per_eu, stream);
  if (const AllReduceAddRmsNormLaunch* got = std::get_if<AllReduceAddRmsNormLaunch>(&l))
    launch_all_reduce_add_rms_norm(h, *got);
  return l;
}

// =================================================================================================
// all_reduce_add_attn_res_rms_norm: Kimi-K3's AttnRes and its RMSNorm on each row of the
// all-reduced sum of `input` (the kernels spell it out). With a `prefix` the sum is added to it,
// into `prefix_output`; without (null), the sum IS the new prefix. `block_output`, when not null,
// gets the new prefix too: the slot of `blocks` the caller appends to.
// =================================================================================================

inline std::variant<AllReduceAddAttnResRmsNormLaunch, Error> all_reduce_add_attn_res_rms_norm(
    Handle& h, const void* input, int64_t input_stride_m, int64_t input_stride_n, int64_t input_size_m,
    int64_t input_size_n, const void* prefix, int64_t prefix_stride_m, int64_t prefix_stride_n,
    int64_t prefix_size_m, int64_t prefix_size_n, const void* blocks, int64_t blocks_stride_m,
    int64_t blocks_stride_r, int64_t blocks_stride_n, int64_t blocks_size_m, int64_t blocks_size_r,
    int64_t blocks_size_n, const void* norm_weight, int64_t norm_weight_stride_n,
    int64_t norm_weight_size_n, const void* qk_weight, int64_t qk_weight_stride_n,
    int64_t qk_weight_size_n, const void* output_norm_weight, int64_t output_norm_weight_stride_n,
    int64_t output_norm_weight_size_n, void* prefix_output, int64_t prefix_output_stride_m,
    int64_t prefix_output_stride_n, int64_t prefix_output_size_m, int64_t prefix_output_size_n,
    void* output, int64_t output_stride_m, int64_t output_stride_n, int64_t output_size_m,
    int64_t output_size_n, void* block_output, int64_t block_output_stride_m,
    int64_t block_output_stride_n, int64_t block_output_size_m, int64_t block_output_size_n,
    DType dtype, int num_blocks, float eps, float output_eps,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_m, std::optional<int> tile_n, std::optional<int> tile_k,
    std::optional<int> reduce_scatter_blocks, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, std::optional<int> waves_per_eu, hipStream_t stream) {
  if (const std::optional<Error> e = layout_refused(
          dtype, {{prefix_stride_m, prefix_stride_n},
                  {output_stride_m, output_stride_n},
                  {input_stride_m, input_stride_n},
                  {blocks_stride_m, blocks_stride_n},
                  {blocks_stride_r, blocks_stride_n},
                  {0, norm_weight_stride_n},
                  {0, qk_weight_stride_n}}))
    return *e;
  if (output_norm_weight != nullptr)
    if (const std::optional<Error> e = layout_refused(dtype, {{0, output_norm_weight_stride_n}}))
      return *e;
  std::variant<AllReduceAddAttnResRmsNormLaunch, Error> l =
      select_all_reduce_add_attn_res_rms_norm(
          h, input, input_stride_m, input_stride_n, input_size_m, input_size_n, prefix, prefix_stride_m,
      prefix_stride_n, prefix_size_m, prefix_size_n, blocks, blocks_stride_m, blocks_stride_r,
      blocks_stride_n, blocks_size_m, blocks_size_r, blocks_size_n, norm_weight,
      norm_weight_stride_n, norm_weight_size_n, qk_weight, qk_weight_stride_n, qk_weight_size_n,
      output_norm_weight, output_norm_weight_stride_n, output_norm_weight_size_n, prefix_output,
      prefix_output_stride_m, prefix_output_stride_n, prefix_output_size_m, prefix_output_size_n,
      output, output_stride_m, output_stride_n, output_size_m, output_size_n, block_output,
      block_output_stride_m, block_output_stride_n, block_output_size_m, block_output_size_n,
      dtype, num_blocks, eps, output_eps,
      algorithm, direction, tile_m, tile_n, tile_k, reduce_scatter_blocks, threads_per_block,
          blocks_per_grid, waves_per_eu, stream);
  if (const auto* got = std::get_if<AllReduceAddAttnResRmsNormLaunch>(&l))
    launch_all_reduce_add_attn_res_rms_norm(h, *got);
  return l;
}

}  // namespace hip_custom_communicator
