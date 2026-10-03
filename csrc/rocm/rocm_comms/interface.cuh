// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE API: every op a caller can run, one function each, and nothing else. Every argument is a
// primitive: pointers, sizes, a DType, and the op's own forcing (an Algorithm and a Direction,
// then its config fields, each none for select's). Each op is plan (select's kernel, or the first
// Error check meets), then launch; it returns the kernel it launched, or the Error and launches
// nothing. Each plan_<op> answers what <op> would run, from the call's facts alone. The internal
// bundles (each op's Args, the Options, its KernelConfig) are made here and go no further out.
// The experimental ops (no stability promise: promoted out or deleted) run on one rank with no
// Handle.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <hip/hip_runtime.h>

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <variant>

namespace hip_comms {

// =================================================================================================
// THE FORCING, made into Options: the template the algorithm and direction name (an op with one
// template needs neither to force a config), and its config from the launch and the op's fields
// (`config` builds its family's from them); a mistake in it is its Error.
// =================================================================================================

template <typename Config>
std::variant<Options, Error> forced(OpType type, std::optional<Algorithm> algorithm,
                                    std::optional<Direction> direction,
                                    std::optional<int> threads_per_block,
                                    std::optional<int> blocks_per_grid,
                                    std::initializer_list<std::optional<int>> fields,
                                    Config config, hipStream_t stream) {
  if (direction && !algorithm) return Error::direction_without_algorithm;
  if (threads_per_block.has_value() != blocks_per_grid.has_value())
    return Error::launch_incomplete;
  const bool launched = blocks_per_grid.has_value();
  if (launched && (*blocks_per_grid < 1 || *blocks_per_grid > p2p::kMaxBlocks ||
                   *threads_per_block < kWaveSize ||
                   *threads_per_block > kBuild.kernels.max_threads ||
                   *threads_per_block % kWaveSize != 0))
    return Error::launch_out_of_range;
  for (const std::optional<int>& f : fields) {
    if (f && !launched) return Error::field_without_launch;
    if (f && *f < 1) return Error::field_not_positive;
  }
  std::optional<Template> fn;
  if (algorithm) {
    fn = template_for(type, *algorithm == Algorithm::two_shot, direction == Direction::push);
    if (!fn) return Error::no_such_template;
  }
  if (!launched) return Options{fn, std::nullopt, stream};
  if (!fn && op(type).templates.size() == 1) fn = op(type).templates[0];
  if (!fn) return Error::config_without_algorithm;
  const std::variant<KernelConfig, Error> c =
      config(LaunchConfig{*threads_per_block, *blocks_per_grid}, *fn);
  if (const Error* e = std::get_if<Error>(&c)) return *e;
  return Options{fn, std::get<KernelConfig>(c), stream};
}

// A FORCED FIELD's value, or 0: the template's own.
constexpr int own(std::optional<int> v) { return v.value_or(0); }

// An op with a Handle: its plan under the forcing, then the kernel launched.
template <typename Args>
std::variant<Kernel, Error> run(Handle& h, const Args& a, const std::variant<Options, Error>& o) {
  if (const Error* e = std::get_if<Error>(&o)) return *e;
  std::variant<Kernel, Error> p = plan(h, a, std::get<Options>(o));
  if (const Kernel* k = std::get_if<Kernel>(&p)) launch(h, *k, a, std::get<Options>(o).stream);
  return p;
}
template <typename Args>
std::variant<Kernel, Error> planned(const Handle& h, const Args& a,
                                    const std::variant<Options, Error>& o) {
  if (const Error* e = std::get_if<Error>(&o)) return *e;
  return plan(h, a, std::get<Options>(o));
}

// EACH FAMILY'S FORCED CONFIG from the op's fields.
inline auto all_reduce_config() {
  return [](LaunchConfig l, Template) -> std::variant<KernelConfig, Error> {
    return AllReduceConfig{l};
  };
}
inline auto row_config(std::optional<int> tile_n) {
  return [=](LaunchConfig l, Template) -> std::variant<KernelConfig, Error> {
    return RowConfig{l, own(tile_n)};
  };
}
// AttnRes's: the pull's family (TILE_M and its reduce-scatter blocks) for the pull two-shot, the
// one-shot's and push's (neither of those) otherwise.
inline auto attn_res_config(std::optional<int> tile_m, std::optional<int> tile_n,
                            std::optional<int> tile_k, std::optional<int> reduce_scatter_blocks) {
  return [=](LaunchConfig l, Template t) -> std::variant<KernelConfig, Error> {
    if (t == Template::all_reduce_pull_two_shot_add_attn_res_rms_norm)
      return AttnResPullConfig{l, own(tile_m), own(tile_n), own(tile_k),
                               own(reduce_scatter_blocks)};
    if (tile_m || reduce_scatter_blocks) return Error::field_not_this_templates;
    return AttnResConfig{l, own(tile_n), own(tile_k)};
  };
}
inline auto gemm_config(std::optional<int> tile_m, std::optional<int> tile_n,
                        std::optional<int> tile_k, std::optional<int> slice_k) {
  return [=](LaunchConfig l, Template) -> std::variant<KernelConfig, Error> {
    return GemmConfig{l, own(tile_m), own(tile_n), own(tile_k), own(slice_k)};
  };
}

// =================================================================================================
// THE OPS.
// =================================================================================================

// The sum of every rank's `inp` (`bytes` of `dtype`) into `out`.
inline std::variant<Kernel, Error> all_reduce(
    Handle& h, void* out, const void* inp, int64_t bytes, DType dtype,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> threads_per_block, std::optional<int> blocks_per_grid,
    hipStream_t stream) {
  return run(h, AllReduceArgs{out, inp, bytes, dtype},
             forced(OpType::all_reduce, algorithm, direction, threads_per_block, blocks_per_grid,
                    {}, all_reduce_config(), stream));
}
inline std::variant<Kernel, Error> plan_all_reduce(
    const Handle& h, int64_t bytes, DType dtype, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  return planned(h, AllReduceArgs{nullptr, nullptr, bytes, dtype},
                 forced(OpType::all_reduce, algorithm, direction, threads_per_block,
                        blocks_per_grid, {}, all_reduce_config(), stream));
}

// out = rms_norm(all_reduce(inp), weight): vLLM's roundings exactly. inp and out [rows, hidden];
// the weight `weight_dtype`, dtype or f32.
inline std::variant<Kernel, Error> all_reduce_rms_norm(
    Handle& h, void* out, const void* inp, const void* weight, DType dtype, DType weight_dtype,
    int64_t rows, int64_t hidden, float eps, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> tile_n,
    std::optional<int> threads_per_block, std::optional<int> blocks_per_grid,
    hipStream_t stream) {
  return run(h,
             NormArgs{false, out, inp, weight, dtype, weight_dtype, rows, hidden, eps, nullptr,
                      nullptr},
             forced(OpType::all_reduce_rms_norm, algorithm, direction, threads_per_block,
                    blocks_per_grid, {tile_n}, row_config(tile_n), stream));
}

// out, residual_out = fused_add_rms_norm(all_reduce(inp), residual, weight): vLLM's roundings.
inline std::variant<Kernel, Error> all_reduce_add_rms_norm(
    Handle& h, void* out, void* residual_out, const void* inp, const void* residual,
    const void* weight, DType dtype, DType weight_dtype, int64_t rows, int64_t hidden, float eps,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_n, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  return run(h,
             NormArgs{true, out, inp, weight, dtype, weight_dtype, rows, hidden, eps,
                      residual_out, residual},
             forced(OpType::all_reduce_add_rms_norm, algorithm, direction, threads_per_block,
                    blocks_per_grid, {tile_n}, row_config(tile_n), stream));
}

// What either norm op would run.
inline std::variant<Kernel, Error> plan_all_reduce_rms_norm(
    const Handle& h, DType dtype, DType weight_dtype, int64_t rows, int64_t hidden,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_n, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  return planned(h,
                 NormArgs{false, nullptr, nullptr, nullptr, dtype, weight_dtype, rows, hidden,
                          0.f, nullptr, nullptr},
                 forced(OpType::all_reduce_rms_norm, algorithm, direction, threads_per_block,
                        blocks_per_grid, {tile_n}, row_config(tile_n), stream));
}
inline std::variant<Kernel, Error> plan_all_reduce_add_rms_norm(
    const Handle& h, DType dtype, DType weight_dtype, int64_t rows, int64_t hidden,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_n, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  return planned(h,
                 NormArgs{true, nullptr, nullptr, nullptr, dtype, weight_dtype, rows, hidden,
                          0.f, nullptr, nullptr},
                 forced(OpType::all_reduce_add_rms_norm, algorithm, direction, threads_per_block,
                        blocks_per_grid, {tile_n}, row_config(tile_n), stream));
}

// Kimi-K3's AttnRes and its RMSNorm on each row of the all-reduced sum of `inp` (the kernels spell
// it out). With `has_prefix` the sum is added to `prefix` in place; without, the sum IS the new
// prefix. `blocks` is [rows, sources, hidden] at `block_stride_m` and `block_stride_r` elements;
// `write_idx` < 0 writes no block; `out_norm_weight` null: no output norm.
inline std::variant<Kernel, Error> all_reduce_add_attn_res_rms_norm(
    Handle& h, void* prefix, void* out, const void* inp, void* blocks, int64_t block_stride_m,
    int64_t block_stride_r, const void* norm_weight, const void* qk_weight,
    const void* out_norm_weight, DType dtype, int64_t rows, int64_t hidden, int num_blocks,
    int write_idx, float eps, float out_eps, bool has_prefix,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_m, std::optional<int> tile_n, std::optional<int> tile_k,
    std::optional<int> reduce_scatter_blocks, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  return run(h,
             AttnResArgs{prefix, out, inp, blocks, block_stride_m, block_stride_r, norm_weight,
                         qk_weight, out_norm_weight, dtype, rows, hidden, num_blocks, write_idx,
                         eps, out_eps, has_prefix},
             forced(OpType::all_reduce_add_attn_res_rms_norm, algorithm, direction,
                    threads_per_block, blocks_per_grid,
                    {tile_m, tile_n, tile_k, reduce_scatter_blocks},
                    attn_res_config(tile_m, tile_n, tile_k, reduce_scatter_blocks), stream));
}
inline std::variant<Kernel, Error> plan_all_reduce_add_attn_res_rms_norm(
    const Handle& h, DType dtype, int64_t rows, int64_t hidden,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_m, std::optional<int> tile_n, std::optional<int> tile_k,
    std::optional<int> reduce_scatter_blocks, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  return planned(h,
                 AttnResArgs{nullptr, nullptr, nullptr, nullptr, 0, 0, nullptr, nullptr, nullptr,
                             dtype, rows, hidden, 0, -1, 0.f, 0.f, false},
                 forced(OpType::all_reduce_add_attn_res_rms_norm, algorithm, direction,
                        threads_per_block, blocks_per_grid,
                        {tile_m, tile_n, tile_k, reduce_scatter_blocks},
                        attn_res_config(tile_m, tile_n, tile_k, reduce_scatter_blocks), stream));
}

// out = rms_norm(all_reduce(inp), norm_weight) @ gemm_weight^T, out [rows, n_cols] at
// `out_stride` (a column slice of a wider buffer); gemm_weight [n_cols, hidden]; `workspace` holds
// the normed rows, inp's shape.
inline std::variant<Kernel, Error> all_reduce_rms_norm_gemm(
    Handle& h, void* out, int64_t out_stride, const void* inp, const void* norm_weight,
    float eps, const void* gemm_weight, int64_t n_cols, void* workspace, DType dtype,
    int64_t rows, int64_t hidden, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> tile_m, std::optional<int> tile_n,
    std::optional<int> tile_k, std::optional<int> slice_k, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  return run(h,
             GemmTailArgs{false, out, out_stride, inp, norm_weight, eps, gemm_weight, n_cols,
                          workspace, dtype, rows, hidden},
             forced(OpType::all_reduce_rms_norm_gemm, algorithm, direction, threads_per_block,
                    blocks_per_grid, {tile_m, tile_n, tile_k, slice_k},
                    gemm_config(tile_m, tile_n, tile_k, slice_k), stream));
}

// The latent MoE tail: as all_reduce_rms_norm_gemm, the product added into out.
inline std::variant<Kernel, Error> all_reduce_rms_norm_gemm_add(
    Handle& h, void* out, int64_t out_stride, const void* inp, const void* norm_weight,
    float eps, const void* gemm_weight, int64_t n_cols, void* workspace, DType dtype,
    int64_t rows, int64_t hidden, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> tile_m, std::optional<int> tile_n,
    std::optional<int> tile_k, std::optional<int> slice_k, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  return run(h,
             GemmTailArgs{true, out, out_stride, inp, norm_weight, eps, gemm_weight, n_cols,
                          workspace, dtype, rows, hidden},
             forced(OpType::all_reduce_rms_norm_gemm_add, algorithm, direction,
                    threads_per_block, blocks_per_grid, {tile_m, tile_n, tile_k, slice_k},
                    gemm_config(tile_m, tile_n, tile_k, slice_k), stream));
}

// What either GEMM-tail op would run.
inline std::variant<Kernel, Error> plan_all_reduce_rms_norm_gemm(
    const Handle& h, int64_t n_cols, DType dtype, int64_t rows, int64_t hidden,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_m, std::optional<int> tile_n, std::optional<int> tile_k,
    std::optional<int> slice_k, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  return planned(h,
                 GemmTailArgs{.add = false, .n_cols = n_cols, .dtype = dtype, .rows = rows,
                              .hidden = hidden},
                 forced(OpType::all_reduce_rms_norm_gemm, algorithm, direction,
                        threads_per_block, blocks_per_grid, {tile_m, tile_n, tile_k, slice_k},
                        gemm_config(tile_m, tile_n, tile_k, slice_k), stream));
}
inline std::variant<Kernel, Error> plan_all_reduce_rms_norm_gemm_add(
    const Handle& h, int64_t n_cols, DType dtype, int64_t rows, int64_t hidden,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_m, std::optional<int> tile_n, std::optional<int> tile_k,
    std::optional<int> slice_k, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  return planned(h,
                 GemmTailArgs{.add = true, .n_cols = n_cols, .dtype = dtype, .rows = rows,
                              .hidden = hidden},
                 forced(OpType::all_reduce_rms_norm_gemm_add, algorithm, direction,
                        threads_per_block, blocks_per_grid, {tile_m, tile_n, tile_k, slice_k},
                        gemm_config(tile_m, tile_n, tile_k, slice_k), stream));
}

// Kimi-K3's latent MoE tail with one all-reduce: inp's row [shared | projected | latent], widths
// hidden, hidden and latent, summed over the ranks; out [rows, hidden] = shared + projected *
// rsqrt(mean(latent^2) + eps).
inline std::variant<Kernel, Error> all_reduce_rms_scale_add(
    Handle& h, void* out, const void* inp, DType dtype, int64_t rows, int64_t hidden,
    int64_t latent, float eps, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> tile_n,
    std::optional<int> threads_per_block, std::optional<int> blocks_per_grid,
    hipStream_t stream) {
  return run(h, ScaleAddArgs{out, inp, dtype, rows, hidden, latent, eps},
             forced(OpType::all_reduce_rms_scale_add, algorithm, direction, threads_per_block,
                    blocks_per_grid, {tile_n}, row_config(tile_n), stream));
}
inline std::variant<Kernel, Error> plan_all_reduce_rms_scale_add(
    const Handle& h, DType dtype, int64_t rows, int64_t hidden, int64_t latent,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_n, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  return planned(h, ScaleAddArgs{nullptr, nullptr, dtype, rows, hidden, latent, 0.f},
                 forced(OpType::all_reduce_rms_scale_add, algorithm, direction,
                        threads_per_block, blocks_per_grid, {tile_n}, row_config(tile_n),
                        stream));
}

namespace experimental {

// AttnRes and its RMSNorm on a local `delta`, no all-reduce: prefix += delta (rounded once), then
// Triton's attn_res over the blocks and the new prefix. Its one template needs no algorithm.
inline std::variant<Kernel, Error> add_attn_res_rms_norm(
    void* prefix, void* out, const void* delta, void* blocks, int64_t block_stride_m,
    int64_t block_stride_r, const void* norm_weight, const void* qk_weight,
    const void* out_norm_weight, DType dtype, int64_t rows, int64_t hidden, int num_blocks,
    int write_idx, float eps, float out_eps, std::optional<int> tile_n,
    std::optional<int> tile_k, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  const AddAttnResArgs a{prefix,    out,       delta,      blocks,          block_stride_m,
                         block_stride_r, norm_weight, qk_weight, out_norm_weight, dtype,
                         rows,      hidden,    num_blocks, write_idx,       eps,
                         out_eps};
  const std::variant<Options, Error> o =
      forced(OpType::add_attn_res_rms_norm, std::nullopt, std::nullopt, threads_per_block,
             blocks_per_grid, {tile_n, tile_k},
             attn_res_config(std::nullopt, tile_n, tile_k, std::nullopt), stream);
  if (const Error* e = std::get_if<Error>(&o)) return *e;
  std::variant<Kernel, Error> p = plan(a, std::get<Options>(o));
  if (const Kernel* k = std::get_if<Kernel>(&p)) launch(*k, a, stream);
  return p;
}

}  // namespace experimental

}  // namespace hip_comms
