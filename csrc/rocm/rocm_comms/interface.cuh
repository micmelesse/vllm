// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE API: every op a caller can run, and nothing else. Per op, `select_<op>` decides everything
// (the kernel, its config, its grid) from the call's primitives and its forcing (an Algorithm and
// a Direction, then its config fields, each none for select's), and returns the op's launch (its
// normal form, types.cuh) or the first Error the call meets; `<op>` is select then launch_<op>
// (launch.cuh), which decides nothing. Python's plan_<op> is select_<op>. The experimental ops
// (no stability promise: promoted out or deleted) run on one rank with no Handle.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <hip/hip_runtime.h>

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>

namespace hip_comms {

using Chosen = std::pair<Template, KernelConfig>;

// =================================================================================================
// all_reduce: the sum of every rank's `inp` (`bytes` of `dtype`) into `out`.
// =================================================================================================

inline std::variant<AllReduceLaunch, Error> select_all_reduce(
    const Handle& h, void* out, const void* inp, int64_t bytes, DType dtype,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> threads_per_block, std::optional<int> blocks_per_grid,
    hipStream_t stream) {
  const std::variant<Chosen, Error> got = chosen_all_reduce(
      h.world_size(), bytes, algorithm, direction, threads_per_block, blocks_per_grid);
  if (const Error* e = std::get_if<Error>(&got)) return *e;
  const auto& [fn, c] = std::get<Chosen>(got);
  // THE BUILD, from what the handle knows: in place when the peers can read the input where it
  // is, otherwise its staged build.
  const bool staged = !h.reads_in_place(inp, stream);
  if (const std::optional<Error> e =
          refused(&h, fn, c, dtype, 1, bytes / elem_bytes(dtype), 0, std::nullopt, inp, staged,
                  stream))
    return *e;
  const LaunchConfig& g = launch_of(c);
  const AllReduceLaunch l{algorithm_of(fn), direction_of(fn), g.threads_per_block,
                          g.blocks_per_grid, staged,      out, inp, bytes, dtype};
  if (!resident(h, l)) return Error::grid_not_resident;
  return l;
}

inline std::variant<AllReduceLaunch, Error> all_reduce(
    Handle& h, void* out, const void* inp, int64_t bytes, DType dtype,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> threads_per_block, std::optional<int> blocks_per_grid,
    hipStream_t stream) {
  std::variant<AllReduceLaunch, Error> l = select_all_reduce(
      h, out, inp, bytes, dtype, algorithm, direction, threads_per_block, blocks_per_grid, stream);
  if (const AllReduceLaunch* got = std::get_if<AllReduceLaunch>(&l))
    launch_all_reduce(h, *got, stream);
  return l;
}

// =================================================================================================
// THE NORMS: out = rms_norm(all_reduce(inp), weight), or with a residual vLLM's fused_add_rms_norm
// (out and residual_out), its roundings exactly. inp [rows, hidden]; the weight `weight_dtype`,
// dtype or f32.
// =================================================================================================

constexpr std::optional<Error> weight_refused(DType dtype, DType weight_dtype) {
  if (weight_dtype != dtype && weight_dtype != DType::f32) return Error::weight_not_built;
  return std::nullopt;
}

inline std::variant<AllReduceRmsNormLaunch, Error> select_all_reduce_rms_norm(
    const Handle& h, void* out, const void* inp, const void* weight, DType dtype,
    DType weight_dtype, int64_t rows, int64_t hidden, float eps,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_n, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  const std::variant<Chosen, Error> got =
      chosen(OpType::all_reduce_rms_norm, h.world_size(), rows, hidden, hidden, hidden, algorithm,
             direction, threads_per_block, blocks_per_grid, {tile_n}, row_config(tile_n));
  if (const Error* e = std::get_if<Error>(&got)) return *e;
  const auto& [fn, c] = std::get<Chosen>(got);
  if (const std::optional<Error> e = refused(&h, fn, c, dtype, rows, hidden, hidden,
                                             weight_refused(dtype, weight_dtype), inp, false,
                                             stream))
    return *e;
  const RowConfig& r = std::get<RowConfig>(c);
  const AllReduceRmsNormLaunch l{algorithm_of(fn),
                                 direction_of(fn),
                                 r.tile_n,
                                 r.launch.threads_per_block,
                                 r.launch.blocks_per_grid,
                                 out,
                                 inp,
                                 weight,
                                 dtype,
                                 weight_dtype,
                                 rows,
                                 hidden,
                                 eps};
  if (!resident(h, l)) return Error::grid_not_resident;
  return l;
}

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
    launch_all_reduce_rms_norm(h, *got, stream);
  return l;
}

inline std::variant<AllReduceAddRmsNormLaunch, Error> select_all_reduce_add_rms_norm(
    const Handle& h, void* out, void* residual_out, const void* inp, const void* residual,
    const void* weight, DType dtype, DType weight_dtype, int64_t rows, int64_t hidden, float eps,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_n, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  const std::variant<Chosen, Error> got =
      chosen(OpType::all_reduce_add_rms_norm, h.world_size(), rows, hidden, hidden, hidden,
             algorithm, direction, threads_per_block, blocks_per_grid, {tile_n},
             row_config(tile_n));
  if (const Error* e = std::get_if<Error>(&got)) return *e;
  const auto& [fn, c] = std::get<Chosen>(got);
  if (const std::optional<Error> e = refused(&h, fn, c, dtype, rows, hidden, hidden,
                                             weight_refused(dtype, weight_dtype), inp, false,
                                             stream))
    return *e;
  const RowConfig& r = std::get<RowConfig>(c);
  const AllReduceAddRmsNormLaunch l{algorithm_of(fn),
                                    direction_of(fn),
                                    r.tile_n,
                                    r.launch.threads_per_block,
                                    r.launch.blocks_per_grid,
                                    out,
                                    residual_out,
                                    inp,
                                    residual,
                                    weight,
                                    dtype,
                                    weight_dtype,
                                    rows,
                                    hidden,
                                    eps};
  if (!resident(h, l)) return Error::grid_not_resident;
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
    launch_all_reduce_add_rms_norm(h, *got, stream);
  return l;
}

// =================================================================================================
// all_reduce_add_attn_res_rms_norm: Kimi-K3's AttnRes and its RMSNorm on each row of the
// all-reduced sum of `inp` (the kernels spell it out). With `has_prefix` the sum is added to
// `prefix` in place; without, the sum IS the new prefix.
// =================================================================================================

inline std::variant<AllReduceAddAttnResRmsNormLaunch, Error>
select_all_reduce_add_attn_res_rms_norm(
    const Handle& h, void* prefix, void* out, const void* inp, void* blocks,
    int64_t block_stride_m, int64_t block_stride_r, const void* norm_weight,
    const void* qk_weight, const void* out_norm_weight, DType dtype, int64_t rows, int64_t hidden,
    int num_blocks, int write_idx, float eps, float out_eps, bool has_prefix,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_m, std::optional<int> tile_n, std::optional<int> tile_k,
    std::optional<int> reduce_scatter_blocks, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  const std::variant<Chosen, Error> got =
      chosen(OpType::all_reduce_add_attn_res_rms_norm, h.world_size(), rows, hidden, hidden,
             hidden, algorithm, direction, threads_per_block, blocks_per_grid,
             {tile_m, tile_n, tile_k, reduce_scatter_blocks},
             attn_res_config(tile_m, tile_n, tile_k, reduce_scatter_blocks));
  if (const Error* e = std::get_if<Error>(&got)) return *e;
  const auto& [fn, c] = std::get<Chosen>(got);
  if (const std::optional<Error> e =
          refused(&h, fn, c, dtype, rows, hidden, hidden, std::nullopt, inp, false, stream))
    return *e;
  // THE PULL'S FAMILY has TILE_M and its reduce-scatter blocks; the others are a row a tile.
  const AttnResPullConfig* pull = std::get_if<AttnResPullConfig>(&c);
  const LaunchConfig& g         = launch_of(c);
  const AllReduceAddAttnResRmsNormLaunch l{algorithm_of(fn),
                                           direction_of(fn),
                                           pull ? pull->tile_m : 1,
                                           tile_n_of(c),
                                           pull ? pull->tile_k : std::get<AttnResConfig>(c).tile_k,
                                           g.threads_per_block,
                                           g.blocks_per_grid,
                                           pull ? pull->reduce_scatter_blocks : 0,
                                           prefix,
                                           out,
                                           inp,
                                           blocks,
                                           block_stride_m,
                                           block_stride_r,
                                           norm_weight,
                                           qk_weight,
                                           out_norm_weight,
                                           dtype,
                                           rows,
                                           hidden,
                                           num_blocks,
                                           write_idx,
                                           eps,
                                           out_eps,
                                           has_prefix};
  if (!resident(h, l)) return Error::grid_not_resident;
  return l;
}

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
    launch_all_reduce_add_attn_res_rms_norm(h, *got, stream);
  return l;
}

// =================================================================================================
// THE GEMM TAILS: out = rms_norm(all_reduce(inp), norm_weight) @ gemm_weight^T, written
// (all_reduce_rms_norm_gemm) or added (_gemm_add, the latent MoE tail). out [rows, n_cols] at
// `out_stride` (a column slice of a wider buffer); gemm_weight [n_cols, hidden]; `workspace` holds
// the normed rows, inp's shape.
// =================================================================================================

inline std::variant<AllReduceRmsNormGemmLaunch, Error> select_all_reduce_rms_norm_gemm(
    const Handle& h, void* out, int64_t out_stride, const void* inp, const void* norm_weight,
    float eps, const void* gemm_weight, int64_t n_cols, void* workspace, DType dtype,
    int64_t rows, int64_t hidden, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> tile_m, std::optional<int> tile_n,
    std::optional<int> tile_k, std::optional<int> slice_k, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  const std::variant<Chosen, Error> got =
      chosen(OpType::all_reduce_rms_norm_gemm, h.world_size(), rows, hidden, hidden, hidden,
             algorithm, direction, threads_per_block, blocks_per_grid,
             {tile_m, tile_n, tile_k, slice_k}, gemm_config(tile_m, tile_n, tile_k, slice_k));
  if (const Error* e = std::get_if<Error>(&got)) return *e;
  const auto& [fn, c] = std::get<Chosen>(got);
  if (const std::optional<Error> e =
          refused(&h, fn, c, dtype, rows, hidden, hidden, std::nullopt, inp, false, stream))
    return *e;
  const GemmConfig& g = std::get<GemmConfig>(c);
  const AllReduceRmsNormGemmLaunch l{algorithm_of(fn),
                                     direction_of(fn),
                                     g.tile_m,
                                     g.tile_n,
                                     g.tile_k,
                                     g.slice_k,
                                     g.launch.threads_per_block,
                                     g.launch.blocks_per_grid,
                                     out,
                                     out_stride,
                                     inp,
                                     norm_weight,
                                     eps,
                                     gemm_weight,
                                     n_cols,
                                     workspace,
                                     dtype,
                                     rows,
                                     hidden};
  if (!resident(h, l)) return Error::grid_not_resident;
  return l;
}

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
    launch_all_reduce_rms_norm_gemm(h, *got, stream);
  return l;
}

inline std::variant<AllReduceRmsNormGemmAddLaunch, Error> select_all_reduce_rms_norm_gemm_add(
    const Handle& h, void* out, int64_t out_stride, const void* inp, const void* norm_weight,
    float eps, const void* gemm_weight, int64_t n_cols, void* workspace, DType dtype,
    int64_t rows, int64_t hidden, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> tile_m, std::optional<int> tile_n,
    std::optional<int> tile_k, std::optional<int> slice_k, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  const std::variant<Chosen, Error> got =
      chosen(OpType::all_reduce_rms_norm_gemm_add, h.world_size(), rows, hidden, hidden, hidden,
             algorithm, direction, threads_per_block, blocks_per_grid,
             {tile_m, tile_n, tile_k, slice_k}, gemm_config(tile_m, tile_n, tile_k, slice_k));
  if (const Error* e = std::get_if<Error>(&got)) return *e;
  const auto& [fn, c] = std::get<Chosen>(got);
  if (const std::optional<Error> e =
          refused(&h, fn, c, dtype, rows, hidden, hidden, std::nullopt, inp, false, stream))
    return *e;
  const GemmConfig& g = std::get<GemmConfig>(c);
  const AllReduceRmsNormGemmAddLaunch l{algorithm_of(fn),
                                        direction_of(fn),
                                        g.tile_m,
                                        g.tile_n,
                                        g.tile_k,
                                        g.slice_k,
                                        g.launch.threads_per_block,
                                        g.launch.blocks_per_grid,
                                        out,
                                        out_stride,
                                        inp,
                                        norm_weight,
                                        eps,
                                        gemm_weight,
                                        n_cols,
                                        workspace,
                                        dtype,
                                        rows,
                                        hidden};
  if (!resident(h, l)) return Error::grid_not_resident;
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
    launch_all_reduce_rms_norm_gemm_add(h, *got, stream);
  return l;
}

// =================================================================================================
// all_reduce_rms_scale_add: Kimi-K3's latent MoE tail with one all-reduce. inp's row is [shared |
// projected | latent], widths hidden, hidden and latent, summed over the ranks; out [rows, hidden]
// = shared + projected * rsqrt(mean(latent^2) + eps). Its tile holds the latent whole (each tile
// needs the row's RMS) beside a TILE_N slice of the hidden, so its grid tiles the hidden.
// =================================================================================================

inline std::variant<AllReduceRmsScaleAddLaunch, Error> select_all_reduce_rms_scale_add(
    const Handle& h, void* out, const void* inp, DType dtype, int64_t rows, int64_t hidden,
    int64_t latent, float eps, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> tile_n,
    std::optional<int> threads_per_block, std::optional<int> blocks_per_grid,
    hipStream_t stream) {
  const int64_t row = 2 * hidden + latent;
  const std::variant<Chosen, Error> got =
      chosen(OpType::all_reduce_rms_scale_add, h.world_size(), rows, row, latent, hidden,
             algorithm, direction, threads_per_block, blocks_per_grid, {tile_n},
             row_config(tile_n));
  if (const Error* e = std::get_if<Error>(&got)) return *e;
  const auto& [fn, c] = std::get<Chosen>(got);
  const int e = elem_bytes(dtype);
  const std::optional<Error> widths =
      hidden * e % kBuild.memory.pack_bytes != 0 || latent * e % kBuild.memory.pack_bytes != 0 ||
              latent < 1
          ? std::optional<Error>{Error::widths_not_packs}
          : std::nullopt;
  if (const std::optional<Error> err =
          refused(&h, fn, c, dtype, rows, row, latent, widths, inp, false, stream))
    return *err;
  const RowConfig& r = std::get<RowConfig>(c);
  const AllReduceRmsScaleAddLaunch l{algorithm_of(fn),
                                     direction_of(fn),
                                     r.tile_n,
                                     r.launch.threads_per_block,
                                     r.launch.blocks_per_grid,
                                     out,
                                     inp,
                                     dtype,
                                     rows,
                                     hidden,
                                     latent,
                                     eps};
  if (!resident(h, l)) return Error::grid_not_resident;
  return l;
}

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
    launch_all_reduce_rms_scale_add(h, *got, stream);
  return l;
}

namespace experimental {

// =================================================================================================
// add_attn_res_rms_norm: AttnRes and its RMSNorm on a local `delta`, no all-reduce: prefix +=
// delta (rounded once), then Triton's attn_res over the blocks and the new prefix. Its one
// template needs no algorithm to force a config.
// =================================================================================================

inline std::variant<AddAttnResRmsNormLaunch, Error> select_add_attn_res_rms_norm(
    void* prefix, void* out, const void* delta, void* blocks, int64_t block_stride_m,
    int64_t block_stride_r, const void* norm_weight, const void* qk_weight,
    const void* out_norm_weight, DType dtype, int64_t rows, int64_t hidden, int num_blocks,
    int write_idx, float eps, float out_eps, std::optional<int> tile_n,
    std::optional<int> tile_k, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid) {
  const std::variant<Chosen, Error> got =
      chosen(OpType::add_attn_res_rms_norm, 1, rows, hidden, hidden, hidden, std::nullopt,
             std::nullopt, threads_per_block, blocks_per_grid, {tile_n, tile_k},
             attn_res_config(std::nullopt, tile_n, tile_k, std::nullopt));
  if (const Error* e = std::get_if<Error>(&got)) return *e;
  const auto& [fn, c] = std::get<Chosen>(got);
  if (const std::optional<Error> e = refused(nullptr, fn, c, dtype, rows, hidden, hidden,
                                             std::nullopt, delta, false, nullptr))
    return *e;
  const AttnResConfig& a = std::get<AttnResConfig>(c);
  return AddAttnResRmsNormLaunch{algorithm_of(fn),
                                 direction_of(fn),
                                 a.tile_n,
                                 a.tile_k,
                                 a.launch.threads_per_block,
                                 a.launch.blocks_per_grid,
                                 prefix,
                                 out,
                                 delta,
                                 blocks,
                                 block_stride_m,
                                 block_stride_r,
                                 norm_weight,
                                 qk_weight,
                                 out_norm_weight,
                                 dtype,
                                 rows,
                                 hidden,
                                 num_blocks,
                                 write_idx,
                                 eps,
                                 out_eps};
}

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
      threads_per_block, blocks_per_grid);
  if (const AddAttnResRmsNormLaunch* got = std::get_if<AddAttnResRmsNormLaunch>(&l))
    launch_add_attn_res_rms_norm(*got, stream);
  return l;
}

}  // namespace experimental

}  // namespace hip_comms
