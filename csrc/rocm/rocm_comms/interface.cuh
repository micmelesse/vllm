// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE API: every op a caller can run, and nothing else: each is select_<op> (select.cuh: the op's
// launch, everything decided, or the first Error the call meets) then launch_<op> (launch.cuh,
// which decides nothing), and returns the launch it ran or the Error. Every argument is a
// primitive: pointers, sizes, a DType, and the op's own forcing (an Algorithm and a Direction,
// then its config fields, each none for select's). Python's plan_<op> is select_<op>. The
// experimental ops (no stability promise: promoted out or deleted): AttnRes on one rank with no
// Handle, and the probe's, which measure the machine.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <hip/hip_runtime.h>

#include <cstdint>
#include <optional>
#include <variant>

namespace hip_comms {

// =================================================================================================
// all_reduce: the sum of every rank's `inp` (`bytes` of `dtype`) into `out`.
// =================================================================================================

inline std::variant<AllReduceLaunch, Error> all_reduce(
    Handle& h, void* out, const void* inp, int64_t bytes, DType dtype,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> threads_per_block, std::optional<int> blocks_per_grid,
    hipStream_t stream) {
  std::variant<AllReduceLaunch, Error> l = select_all_reduce(
      h, out, inp, bytes, dtype, algorithm, direction, threads_per_block, blocks_per_grid, stream);
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
    Handle& h, void* out, const void* inp, const void* weight, DType dtype, DType weight_dtype,
    int64_t rows, int64_t hidden, float eps, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> tile_n,
    std::optional<int> threads_per_block, std::optional<int> blocks_per_grid,
    hipStream_t stream) {
  std::variant<AllReduceRmsNormLaunch, Error> l = select_all_reduce_rms_norm(
      h, out, inp, weight, dtype, weight_dtype, rows, hidden, eps, algorithm, direction, tile_n,
      threads_per_block, blocks_per_grid, stream);
  if (const AllReduceRmsNormLaunch* got = std::get_if<AllReduceRmsNormLaunch>(&l))
    launch_all_reduce_rms_norm(h, *got);
  return l;
}

inline std::variant<AllReduceAddRmsNormLaunch, Error> all_reduce_add_rms_norm(
    Handle& h, void* out, void* residual_out, const void* inp, const void* residual,
    const void* weight, DType dtype, DType weight_dtype, int64_t rows, int64_t hidden, float eps,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_n, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  std::variant<AllReduceAddRmsNormLaunch, Error> l = select_all_reduce_add_rms_norm(
      h, out, residual_out, inp, residual, weight, dtype, weight_dtype, rows, hidden, eps,
      algorithm, direction, tile_n, threads_per_block, blocks_per_grid, stream);
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
    Handle& h, void* prefix, void* out, const void* inp, void* blocks, int64_t block_stride_m,
    int64_t block_stride_r, const void* norm_weight, const void* qk_weight,
    const void* out_norm_weight, DType dtype, int64_t rows, int64_t hidden, int num_blocks,
    int write_idx, float eps, float out_eps, bool has_prefix,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_m, std::optional<int> tile_n, std::optional<int> tile_k,
    std::optional<int> reduce_scatter_blocks, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  std::variant<AllReduceAddAttnResRmsNormLaunch, Error> l =
      select_all_reduce_add_attn_res_rms_norm(
          h, prefix, out, inp, blocks, block_stride_m, block_stride_r, norm_weight, qk_weight,
          out_norm_weight, dtype, rows, hidden, num_blocks, write_idx, eps, out_eps, has_prefix,
          algorithm, direction, tile_m, tile_n, tile_k, reduce_scatter_blocks, threads_per_block,
          blocks_per_grid, stream);
  if (const auto* got = std::get_if<AllReduceAddAttnResRmsNormLaunch>(&l))
    launch_all_reduce_add_attn_res_rms_norm(h, *got);
  return l;
}

// =================================================================================================
// THE GEMM TAILS: out = rms_norm(all_reduce(inp), norm_weight) @ gemm_weight^T, written
// (all_reduce_rms_norm_gemm) or added (_gemm_add, the latent MoE tail). out [rows, n_cols] at
// `out_stride` (a column slice of a wider buffer); gemm_weight [n_cols, hidden]; `workspace` holds
// the normed rows, inp's shape.
// =================================================================================================

inline std::variant<AllReduceRmsNormGemmLaunch, Error> all_reduce_rms_norm_gemm(
    Handle& h, void* out, int64_t out_stride, const void* inp, const void* norm_weight,
    float eps, const void* gemm_weight, int64_t n_cols, void* workspace, DType dtype,
    int64_t rows, int64_t hidden, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> tile_m, std::optional<int> tile_n,
    std::optional<int> tile_k, std::optional<int> slice_k, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  std::variant<AllReduceRmsNormGemmLaunch, Error> l = select_all_reduce_rms_norm_gemm(
      h, out, out_stride, inp, norm_weight, eps, gemm_weight, n_cols, workspace, dtype, rows,
      hidden, algorithm, direction, tile_m, tile_n, tile_k, slice_k, threads_per_block,
      blocks_per_grid, stream);
  if (const AllReduceRmsNormGemmLaunch* got = std::get_if<AllReduceRmsNormGemmLaunch>(&l))
    launch_all_reduce_rms_norm_gemm(h, *got);
  return l;
}

inline std::variant<AllReduceRmsNormGemmAddLaunch, Error> all_reduce_rms_norm_gemm_add(
    Handle& h, void* out, int64_t out_stride, const void* inp, const void* norm_weight,
    float eps, const void* gemm_weight, int64_t n_cols, void* workspace, DType dtype,
    int64_t rows, int64_t hidden, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> tile_m, std::optional<int> tile_n,
    std::optional<int> tile_k, std::optional<int> slice_k, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  std::variant<AllReduceRmsNormGemmAddLaunch, Error> l = select_all_reduce_rms_norm_gemm_add(
      h, out, out_stride, inp, norm_weight, eps, gemm_weight, n_cols, workspace, dtype, rows,
      hidden, algorithm, direction, tile_m, tile_n, tile_k, slice_k, threads_per_block,
      blocks_per_grid, stream);
  if (const AllReduceRmsNormGemmAddLaunch* got = std::get_if<AllReduceRmsNormGemmAddLaunch>(&l))
    launch_all_reduce_rms_norm_gemm_add(h, *got);
  return l;
}

// =================================================================================================
// all_reduce_rms_scale_add: Kimi-K3's latent MoE tail with one all-reduce. inp's row is [shared |
// projected | latent], widths hidden, hidden and latent, summed over the ranks; out [rows, hidden]
// = shared + projected * rsqrt(mean(latent^2) + eps). Its tile holds the latent whole (each tile
// needs the row's RMS) beside a TILE_N slice of the hidden, so its grid tiles the hidden.
// =================================================================================================

inline std::variant<AllReduceRmsScaleAddLaunch, Error> all_reduce_rms_scale_add(
    Handle& h, void* out, const void* inp, DType dtype, int64_t rows, int64_t hidden,
    int64_t latent, float eps, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> tile_n,
    std::optional<int> threads_per_block, std::optional<int> blocks_per_grid,
    hipStream_t stream) {
  std::variant<AllReduceRmsScaleAddLaunch, Error> l = select_all_reduce_rms_scale_add(
      h, out, inp, dtype, rows, hidden, latent, eps, algorithm, direction, tile_n,
      threads_per_block, blocks_per_grid, stream);
  if (const AllReduceRmsScaleAddLaunch* got = std::get_if<AllReduceRmsScaleAddLaunch>(&l))
    launch_all_reduce_rms_scale_add(h, *got);
  return l;
}

namespace experimental {

// =================================================================================================
// add_attn_res_rms_norm: AttnRes and its RMSNorm on a local `delta`, no all-reduce: prefix +=
// delta (rounded once), then Triton's attn_res over the blocks and the new prefix. Its one
// template needs no algorithm to force a config.
// =================================================================================================

inline std::variant<AddAttnResRmsNormLaunch, Error> add_attn_res_rms_norm(
    void* prefix, void* out, const void* delta, void* blocks, int64_t block_stride_m,
    int64_t block_stride_r, const void* norm_weight, const void* qk_weight,
    const void* out_norm_weight, DType dtype, int64_t rows, int64_t hidden, int num_blocks,
    int write_idx, float eps, float out_eps, std::optional<int> tile_n,
    std::optional<int> tile_k, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  std::variant<AddAttnResRmsNormLaunch, Error> l = select_add_attn_res_rms_norm(
      prefix, out, delta, blocks, block_stride_m, block_stride_r, norm_weight, qk_weight,
      out_norm_weight, dtype, rows, hidden, num_blocks, write_idx, eps, out_eps, tile_n, tile_k,
      threads_per_block, blocks_per_grid, stream);
  if (const AddAttnResRmsNormLaunch* got = std::get_if<AddAttnResRmsNormLaunch>(&l))
    launch_add_attn_res_rms_norm(*got);
  return l;
}


// THE PROBE'S OPS: the machine as the p2p layer sees it, one measurement at a time (calibrate.py
// times them).
inline std::variant<ProbeBarrierLaunch, Error> probe_barrier(Handle& h, hipStream_t stream) {
  std::variant<ProbeBarrierLaunch, Error> l = select_probe_barrier(h, stream);
  if (const ProbeBarrierLaunch* got = std::get_if<ProbeBarrierLaunch>(&l))
    launch_probe_barrier(h, *got);
  return l;
}

inline std::variant<PingPongLaunch, Error> ping_pong(Handle& h, int peer, int iters, void* ticks,
                                                     hipStream_t stream) {
  std::variant<PingPongLaunch, Error> l = select_ping_pong(h, peer, iters, ticks, stream);
  if (const PingPongLaunch* got = std::get_if<PingPongLaunch>(&l)) launch_ping_pong(h, *got);
  return l;
}

inline std::variant<LinkTrafficLaunch, Error> link_traffic(Handle& h, const void* buffer,
                                                           int64_t bytes, Traffic mode, int peer,
                                                           int blocks, int pullers, void* sink,
                                                           hipStream_t stream) {
  std::variant<LinkTrafficLaunch, Error> l =
      select_link_traffic(h, buffer, bytes, mode, peer, blocks, pullers, sink, stream);
  if (const LinkTrafficLaunch* got = std::get_if<LinkTrafficLaunch>(&l))
    launch_link_traffic(h, *got);
  return l;
}

}  // namespace experimental

}  // namespace hip_comms
