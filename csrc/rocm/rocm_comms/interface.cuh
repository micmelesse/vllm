// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE API: every op a caller can run, and nothing else: each is select_<op> (select.cuh: the op's
// launch, everything decided, or the first Error the call meets) then launch_<op> (launch.cuh,
// which decides nothing), and returns the launch it ran or the Error. Every argument is a
// primitive: pointers, sizes, a DType, and the op's own forcing (an Algorithm and a Direction,
// then its config fields, each none for select's). Python's plan_<op> is select_<op>. The
// experimental ops (no stability promise: promoted out or deleted): AttnRes on one rank with no
// Handle, and the probe, which measures the machine.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <hip/hip_runtime.h>

#include <c10/util/BFloat16.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "kernels/probe_barrier.cuh"
#include "kernels/probe_link_traffic.cuh"
#include "kernels/probe_ping_pong.cuh"

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


// =================================================================================================
// probe: the machine as the p2p layer sees it, measured, every rank together (`gather` goes round
// them). The round trip to each peer, then GB/s by name, `<buffer>_<mode>`: the buffer the staging
// (uncached) or a registered allocation (cached); the mode pulled from one peer (rank ^ 1) or every
// peer, pushed into every peer, or both at once (each way), the blocks split or each doing both;
// then the grids the links take. Each the median of `trials`, a probe barrier before each; `bytes`
// a peer's share, at most the staging; `traffic_iters` launches a trial, `ping_iters` round trips
// a trial. machine/hardware.cuh records its answers.
// =================================================================================================

inline std::variant<ProbeResult, Error> probe(Handle& h, const Gather& gather, int64_t bytes,
                                              int ping_iters, int traffic_iters, int trials,
                                              hipStream_t stream) {
  if (ping_iters < 1 || traffic_iters < 1 || trials < 1 || bytes < kBuild.memory.pack_bytes)
    return Error::probe_out_of_range;
  bytes           = std::min<int64_t>(bytes, h.staging_bytes()) / 16 * 16;
  const int world = h.world_size(), rank = h.rank();
  int device = 0, khz = 0;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, device));
  void (*barrier)(p2p::DevComm)                                    = nullptr;
  void (*traffic)(p2p::DevComm, int, int, int, int64_t, uint32_t*) = nullptr;
  by_world(world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    barrier          = probe_barrier<NG>;
    traffic          = link_traffic<c10::BFloat16, NG>;
  });
  const auto together = [&]() { barrier<<<dim3(1), dim3(64), 0, stream>>>(h.dev_comm()); };
  const auto median   = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  uint64_t* ticks = nullptr;
  uint32_t* sink  = nullptr;
  HIP_CHECK(hipMalloc(&ticks, sizeof(uint64_t)));
  HIP_CHECK(hipMalloc(&sink, sizeof(uint32_t)));

  // THE ROUND TRIP to each peer: round d pairs this rank with rank ^ d, both calling together.
  ProbeResult got;
  got.ping_ns.assign(world, 0.0);
  for (int d = 1; d < world; ++d) {
    const int peer = rank ^ d;
    std::vector<double> each;
    for (int t = 0; t < trials; ++t) {
      together();
      const uint32_t base = h.take_flags(peer, ping_pong_flags(ping_iters));
      ping_pong<<<dim3(1), dim3(64), 0, stream>>>(h.dev_comm(), peer, base, ping_iters, ticks);
      uint64_t host = 0;
      HIP_CHECK(hipMemcpyAsync(&host, ticks, sizeof(host), hipMemcpyDeviceToHost, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
      each.push_back(static_cast<double>(host) * 1e6 / khz / ping_iters);
    }
    got.ping_ns[peer] = median(each);
  }

  // THE LINKS: GB/s in and out of this rank, the whole device streaming, over the staging (the
  // symmetric memory, uncached) and over an ordinary allocation every rank registers (cached).
  void* cached = nullptr;
  HIP_CHECK(hipMalloc(&cached, static_cast<size_t>(bytes)));
  HIP_CHECK(hipMemset(cached, 0, static_cast<size_t>(bytes)));
  h.register_buffer(cached, gather);
  const dim3 block(kBuild.kernels.max_threads);
  const auto gbytes = [&](const void* over, Traffic mode, int peer,
                          int blocks = kTarget.compute_units, int pullers = -1) {
    const p2p::DevComm p = h.dev_comm(over, bytes, stream);
    if (pullers < 0) pullers = blocks / 2;
    const auto run = [&]() {
      traffic<<<dim3(blocks), block, 0, stream>>>(p, static_cast<int>(mode), peer, pullers,
                                                  bytes / 16, sink);
    };
    std::vector<double> each;
    for (int t = 0; t < trials; ++t) {
      together();
      run();  // untimed: the first touch
      hipEvent_t start, stop;
      HIP_CHECK(hipEventCreate(&start));
      HIP_CHECK(hipEventCreate(&stop));
      HIP_CHECK(hipEventRecord(start, stream));
      for (int i = 0; i < traffic_iters; ++i) run();
      HIP_CHECK(hipEventRecord(stop, stream));
      HIP_CHECK(hipEventSynchronize(stop));
      float ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
      HIP_CHECK(hipEventDestroy(start));
      HIP_CHECK(hipEventDestroy(stop));
      const int peers = peer >= 0 ? 1 : world - 1;
      each.push_back(static_cast<double>(bytes) * peers * traffic_iters / (ms * 1e-3) / 1e9);
    }
    return median(each);
  };
  const auto measured = [&](const std::string& name, double gbps) {
    got.names.push_back(name);
    got.gbytes_per_s.push_back(gbps);
  };
  for (const auto& [over, where] : {std::pair<const void*, const char*>{h.staging(), "staging"},
                                    std::pair<const void*, const char*>{cached, "cached"}})
    for (const auto& [mode, peer, what] :
         {std::tuple<Traffic, int, const char*>{Traffic::pull, rank ^ 1, "pull_one"},
          {Traffic::pull, -1, "pull"},
          {Traffic::push, -1, "push"},
          {Traffic::split, -1, "split"},
          {Traffic::each, -1, "each"}})
      measured(std::string(where) + "_" + what, gbytes(over, mode, peer));
  // THE GRID: how many blocks the links take before reads queue (the kernels' reduce runs on
  // dozens).
  for (const int blocks : {16, 32, 48, 64, 96, 128}) {
    measured("cached_pull_b" + std::to_string(blocks), gbytes(cached, Traffic::pull, -1, blocks));
    measured("cached_push_b" + std::to_string(blocks), gbytes(cached, Traffic::push, -1, blocks));
  }
  // BOTH AT ONCE AT SANE GRIDS: `pullers` blocks pull, the rest push; GB/s each way.
  for (const auto& [pullers, pushers] :
       {std::pair<int, int>{32, 32}, {32, 64}, {32, 128}, {48, 48}, {48, 144}, {64, 64}})
    measured("cached_split_p" + std::to_string(pullers) + "_q" + std::to_string(pushers),
             gbytes(cached, Traffic::split, -1, pullers + pushers, pullers));

  // Every rank done reading every other's before any frees its copy.
  together();
  HIP_CHECK(hipStreamSynchronize(stream));
  (void)gather(std::string(1, '\0'));
  h.forget_buffer(cached);
  HIP_CHECK(hipFree(cached));
  HIP_CHECK(hipFree(ticks));
  HIP_CHECK(hipFree(sink));
  return got;
}

}  // namespace experimental

}  // namespace hip_comms
