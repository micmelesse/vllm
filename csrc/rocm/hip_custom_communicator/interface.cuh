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
// all_reduce: the sum of every rank's `inp` (`bytes` of `dtype`) into `out`.
// =================================================================================================

inline std::variant<AllReduceLaunch, Error> all_reduce(
    Handle& h, void* out, int64_t out_stride_m, int64_t out_stride_n, const void* inp,
    int64_t inp_stride_m, int64_t inp_stride_n, int64_t m, int64_t n, DType dtype,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_m, std::optional<int> tile_n, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, std::optional<int> waves_per_eu, hipStream_t stream) {
  if (const std::optional<Error> e = layout_refused(
          dtype, {{out_stride_m, out_stride_n}, {inp_stride_m, inp_stride_n}}))
    return *e;
  std::variant<AllReduceLaunch, Error> l = select_all_reduce(
      h, out, out_stride_m, out_stride_n, inp, inp_stride_m, inp_stride_n, m, n, dtype,
      algorithm, direction, tile_m, tile_n, threads_per_block, blocks_per_grid, waves_per_eu,
      stream);
  if (const AllReduceLaunch* got = std::get_if<AllReduceLaunch>(&l))
    launch_all_reduce(h, *got);
  return l;
}

// =================================================================================================
// THE NORMS: out = rms_norm(all_reduce(inp), weight), or with a residual vLLM's fused_add_rms_norm
// (out and residual_out), its roundings exactly. inp [rows, hidden]; the weight `weight_dtype`,
// dtype or f32.
// =================================================================================================

inline std::variant<AllReduceRmsNormLaunch, Error> all_reduce_rms_norm(
    Handle& h, void* out, int64_t out_stride_m, int64_t out_stride_n, const void* inp,
    int64_t inp_stride_m, int64_t inp_stride_n, const void* weight, int64_t weight_stride_n,
    DType dtype, DType weight_dtype, int64_t m, int64_t n, float eps,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_m, std::optional<int> tile_n, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, std::optional<int> waves_per_eu, hipStream_t stream) {
  if (const std::optional<Error> e = layout_refused(
          dtype, {{out_stride_m, out_stride_n}, {inp_stride_m, inp_stride_n}}))
    return *e;
  if (const std::optional<Error> e = layout_refused(weight_dtype, {{0, weight_stride_n}}))
    return *e;
  std::variant<AllReduceRmsNormLaunch, Error> l = select_all_reduce_rms_norm(
      h, out, out_stride_m, out_stride_n, inp, inp_stride_m, inp_stride_n, weight,
      weight_stride_n, dtype, weight_dtype, m, n, eps, algorithm, direction, tile_m, tile_n,
      threads_per_block, blocks_per_grid, waves_per_eu, stream);
  if (const AllReduceRmsNormLaunch* got = std::get_if<AllReduceRmsNormLaunch>(&l))
    launch_all_reduce_rms_norm(h, *got);
  return l;
}

inline std::variant<AllReduceAddRmsNormLaunch, Error> all_reduce_add_rms_norm(
    Handle& h, void* out, int64_t out_stride_m, int64_t out_stride_n, void* residual_out,
    int64_t residual_out_stride_m, int64_t residual_out_stride_n, const void* inp,
    int64_t inp_stride_m, int64_t inp_stride_n, const void* residual, int64_t residual_stride_m,
    int64_t residual_stride_n, const void* weight, int64_t weight_stride_n, DType dtype,
    DType weight_dtype, int64_t m, int64_t n, float eps,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_m, std::optional<int> tile_n, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, std::optional<int> waves_per_eu, hipStream_t stream) {
  if (const std::optional<Error> e = layout_refused(
          dtype, {{out_stride_m, out_stride_n},
                  {residual_out_stride_m, residual_out_stride_n},
                  {inp_stride_m, inp_stride_n},
                  {residual_stride_m, residual_stride_n}}))
    return *e;
  if (const std::optional<Error> e = layout_refused(weight_dtype, {{0, weight_stride_n}}))
    return *e;
  std::variant<AllReduceAddRmsNormLaunch, Error> l = select_all_reduce_add_rms_norm(
      h, out, out_stride_m, out_stride_n, residual_out, residual_out_stride_m,
      residual_out_stride_n, inp, inp_stride_m, inp_stride_n, residual, residual_stride_m,
      residual_stride_n, weight, weight_stride_n, dtype, weight_dtype, m, n, eps,
      algorithm, direction, tile_m, tile_n, threads_per_block, blocks_per_grid, waves_per_eu, stream);
  if (const AllReduceAddRmsNormLaunch* got = std::get_if<AllReduceAddRmsNormLaunch>(&l))
    launch_all_reduce_add_rms_norm(h, *got);
  return l;
}

// =================================================================================================
// all_reduce_add_attn_res_rms_norm: Kimi-K3's AttnRes and its RMSNorm on each row of the
// all-reduced sum of `inp` (the kernels spell it out). With `has_prefix` the sum is added to
// `prefix` in place; without, the sum IS the new prefix.
// =================================================================================================

inline std::variant<AllReduceAddAttnResRmsNormLaunch, Error> all_reduce_add_attn_res_rms_norm(
    Handle& h, void* prefix, int64_t prefix_stride_m, int64_t prefix_stride_n, void* out,
    int64_t out_stride_m, int64_t out_stride_n, const void* inp, int64_t inp_stride_m,
    int64_t inp_stride_n, void* blocks, int64_t blocks_stride_m, int64_t blocks_stride_r,
    int64_t blocks_stride_n, const void* norm_weight, int64_t norm_weight_stride_n,
    const void* qk_weight, int64_t qk_weight_stride_n, const void* out_norm_weight,
    int64_t out_norm_weight_stride_n,
    DType dtype, int64_t m, int64_t n, int num_blocks,
    int write_idx, float eps, float out_eps, bool has_prefix,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_m, std::optional<int> tile_n, std::optional<int> tile_k,
    std::optional<int> reduce_scatter_blocks, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, std::optional<int> waves_per_eu, hipStream_t stream) {
  if (const std::optional<Error> e = layout_refused(
          dtype, {{prefix_stride_m, prefix_stride_n},
                  {out_stride_m, out_stride_n},
                  {inp_stride_m, inp_stride_n},
                  {blocks_stride_m, blocks_stride_n},
                  {blocks_stride_r, blocks_stride_n},
                  {0, norm_weight_stride_n},
                  {0, qk_weight_stride_n}}))
    return *e;
  if (out_norm_weight != nullptr)
    if (const std::optional<Error> e = layout_refused(dtype, {{0, out_norm_weight_stride_n}}))
      return *e;
  std::variant<AllReduceAddAttnResRmsNormLaunch, Error> l =
      select_all_reduce_add_attn_res_rms_norm(
          h, prefix, prefix_stride_m, prefix_stride_n, out, out_stride_m, out_stride_n, inp,
          inp_stride_m, inp_stride_n, blocks, blocks_stride_m, blocks_stride_r, blocks_stride_n,
          norm_weight, norm_weight_stride_n, qk_weight, qk_weight_stride_n, out_norm_weight,
          out_norm_weight_stride_n,
          dtype, m, n, num_blocks, write_idx, eps, out_eps, has_prefix,
          algorithm, direction, tile_m, tile_n, tile_k, reduce_scatter_blocks, threads_per_block,
          blocks_per_grid, waves_per_eu, stream);
  if (const auto* got = std::get_if<AllReduceAddAttnResRmsNormLaunch>(&l))
    launch_all_reduce_add_attn_res_rms_norm(h, *got);
  return l;
}

}  // namespace hip_custom_communicator
