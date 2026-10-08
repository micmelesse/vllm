// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// LAUNCH, NO DECISIONS: each op's launch_<op> runs the kernel its launch names (select chose it) on
// the launch's grid, block and stream, with the kernel's arguments from the launch and the peers'
// view of its input. A branch here follows a field of the launch; it chooses nothing.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <hip/hip_runtime.h>

#include <cstdint>

namespace hip_comms {

// A KERNEL CALLED THROUGH ITS OP'S SIGNATURE (types.cuh): each argument is converted to its
// parameter, so a wrong one is a compile error, then launched as `kernel<<<blocks, threads, 0,
// stream>>>(args...)` launches it.
template <typename SIGNATURE>
struct Call;
template <typename... PARAMS>
struct Call<void (*)(PARAMS...)> {
  static void run(const void* kernel, int blocks, int threads, hipStream_t stream, PARAMS... args) {
    void* argv[] = {static_cast<void*>(&args)...};
    HIP_CHECK(hipLaunchKernel(kernel, dim3(blocks), dim3(threads), argv, 0, stream));
  }
};

// EVERY LAUNCH OVER THE PEERS hands its kernel, in this order: every rank's input (a device table:
// a captured launch's are filled after the capture; an eager input's is the staging's, copied in
// on the launch's stream), every rank's scratch or staging where the kernel uses it, then every
// rank's signal block, this rank's, its rank and the wait limit; then the kernel's own arguments.
// A ONE-SHOT reads only the inputs; a TWO-SHOT also every rank's scratch.

// A [rows, cols] BUFFER'S BYTES at its row stride, first element to last: what an eager input's
// copy into the registered buffer moves, so the peers read it at its own layout.
inline int64_t extent_bytes(int64_t rows, int64_t cols, int64_t stride_m, DType dtype) {
  return rows > 0 ? ((rows - 1) * stride_m + cols) * elem_bytes(dtype) : 0;
}

// STAGED: the kernel copies its own input through its staging a pass at a time, so no peer reads it
// where it is; in place, the peers read it through its input table.
inline void launch_all_reduce(Handle& h, const AllReduceLaunch& l) {
  const int rows  = static_cast<int>(l.rows);
  const int packs = static_cast<int>(packs_of(l.cols, l.dtype));
  const bool one_shot = l.algorithm == Algorithm::one_shot;
  if (l.staged && one_shot)
    Call<AllReduceOneShotStagedKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, h.peer_staging(),
        l.staging_stride_m, l.staging_stride_n, h.peer_signals(), h.self_signal(), h.rank(),
        h.timeout_ticks(), l.out, l.out_stride_m, l.out_stride_n, l.inp, l.inp_stride_m,
        l.inp_stride_n, rows, packs, l.band_rows);
  else if (l.staged)
    Call<AllReduceTwoShotStagedKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, h.peer_scratch(),
        l.scratch_stride_m, l.scratch_stride_n, h.peer_staging(), l.staging_stride_m,
        l.staging_stride_n, h.peer_signals(), h.self_signal(), h.rank(), h.timeout_ticks(),
        l.out, l.out_stride_m, l.out_stride_n, l.inp, l.inp_stride_m, l.inp_stride_n, rows,
        packs, l.band_rows);
  else {
    const PeerPtrs* inputs =
        h.peer_inputs(l.inp, extent_bytes(l.rows, l.cols, l.inp_stride_m, l.dtype), l.stream);
    if (one_shot)
      Call<AllReduceOneShotKernel>::run(l.kernel, l.blocks_per_grid, l.threads_per_block,
                                        l.stream, inputs, l.inp_stride_m, l.inp_stride_n,
                                        h.peer_signals(), h.self_signal(), h.rank(),
                                        h.timeout_ticks(), l.out, l.out_stride_m, l.out_stride_n,
                                        rows, packs);
    else
      Call<AllReduceTwoShotKernel>::run(l.kernel, l.blocks_per_grid, l.threads_per_block,
                                        l.stream, inputs, l.inp_stride_m, l.inp_stride_n,
                                        h.peer_scratch(), l.scratch_stride_m, l.scratch_stride_n,
                                        h.peer_signals(), h.self_signal(), h.rank(),
                                        h.timeout_ticks(), l.out, l.out_stride_m, l.out_stride_n,
                                        rows, packs);
  }
}

inline void launch_all_reduce_rms_norm(Handle& h, const AllReduceRmsNormLaunch& l) {
  const int64_t bytes = extent_bytes(l.rows, l.hidden, l.inp_stride_m, l.dtype);
  const int rows = static_cast<int>(l.rows), packs = static_cast<int>(packs_of(l.hidden, l.dtype));
  const PeerPtrs* inputs = h.peer_inputs(l.inp, bytes, l.stream);
  if (l.algorithm == Algorithm::one_shot)
    Call<AllReduceRmsNormOneShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, inputs, l.inp_stride_m,
        l.inp_stride_n, h.peer_signals(), h.self_signal(), h.rank(), h.timeout_ticks(), l.out,
        l.out_stride_m, l.out_stride_n, l.weight, l.weight_stride_n, l.eps, rows, packs);
  else
    Call<AllReduceRmsNormTwoShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, inputs, l.inp_stride_m,
        l.inp_stride_n, h.peer_scratch(), l.scratch_stride_m, l.scratch_stride_n,
        h.peer_signals(), h.self_signal(), h.rank(), h.timeout_ticks(), l.out, l.out_stride_m,
        l.out_stride_n, l.weight, l.weight_stride_n, l.eps, rows, packs);
}

inline void launch_all_reduce_add_rms_norm(Handle& h, const AllReduceAddRmsNormLaunch& l) {
  const int64_t bytes = extent_bytes(l.rows, l.hidden, l.inp_stride_m, l.dtype);
  const int rows = static_cast<int>(l.rows), packs = static_cast<int>(packs_of(l.hidden, l.dtype));
  const PeerPtrs* inputs = h.peer_inputs(l.inp, bytes, l.stream);
  if (l.algorithm == Algorithm::one_shot)
    Call<AllReduceAddRmsNormOneShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, inputs, l.inp_stride_m,
        l.inp_stride_n, h.peer_signals(), h.self_signal(), h.rank(), h.timeout_ticks(), l.out,
        l.out_stride_m, l.out_stride_n, l.residual_out, l.residual_out_stride_m,
        l.residual_out_stride_n, l.residual, l.residual_stride_m, l.residual_stride_n, l.weight,
        l.weight_stride_n, l.eps, rows, packs);
  else
    Call<AllReduceAddRmsNormTwoShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, inputs, l.inp_stride_m,
        l.inp_stride_n, h.peer_scratch(), l.scratch_stride_m, l.scratch_stride_n,
        h.peer_signals(), h.self_signal(), h.rank(), h.timeout_ticks(), l.out, l.out_stride_m,
        l.out_stride_n, l.residual_out, l.residual_out_stride_m, l.residual_out_stride_n,
        l.residual, l.residual_stride_m, l.residual_stride_n, l.weight, l.weight_stride_n, l.eps,
        rows, packs);
}

// THE PULL TWO-SHOT takes its reduce-scatter blocks last.
inline void launch_all_reduce_add_attn_res_rms_norm(Handle& h,
                                                    const AllReduceAddAttnResRmsNormLaunch& l) {
  const int64_t bytes = extent_bytes(l.rows, l.hidden, l.inp_stride_m, l.dtype);
  const int rows = static_cast<int>(l.rows), packs = static_cast<int>(packs_of(l.hidden, l.dtype));
  const PeerPtrs* inputs = h.peer_inputs(l.inp, bytes, l.stream);
  if (l.algorithm == Algorithm::one_shot)
    Call<AllReduceAddAttnResRmsNormOneShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, inputs, l.inp_stride_m,
        l.inp_stride_n, h.peer_signals(), h.self_signal(), h.rank(), h.timeout_ticks(),
        l.prefix, l.prefix_stride_m, l.prefix_stride_n, l.blocks, l.blocks_stride_m,
        l.blocks_stride_r, l.blocks_stride_n, l.norm_weight, l.norm_weight_stride_n, l.qk_weight,
        l.qk_weight_stride_n, l.out_norm_weight, l.out_norm_weight_stride_n, l.out, l.out_stride_m,
        l.out_stride_n, l.num_blocks, l.write_idx, l.eps, l.out_eps, rows, packs);
  else if (l.direction == Direction::push)
    Call<AllReduceAddAttnResRmsNormPushKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, inputs, l.inp_stride_m,
        l.inp_stride_n, h.peer_scratch(), l.scratch_stride_m, l.scratch_stride_n,
        h.peer_signals(), h.self_signal(), h.rank(), h.timeout_ticks(),
        l.prefix, l.prefix_stride_m, l.prefix_stride_n, l.blocks, l.blocks_stride_m,
        l.blocks_stride_r, l.blocks_stride_n, l.norm_weight, l.norm_weight_stride_n, l.qk_weight,
        l.qk_weight_stride_n, l.out_norm_weight, l.out_norm_weight_stride_n, l.out, l.out_stride_m,
        l.out_stride_n, l.num_blocks, l.write_idx, l.eps, l.out_eps, rows, packs);
  else
    Call<AllReduceAddAttnResRmsNormPullKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, inputs, l.inp_stride_m,
        l.inp_stride_n, h.peer_scratch(), l.scratch_stride_m, l.scratch_stride_n,
        h.peer_signals(), h.self_signal(), h.rank(), h.timeout_ticks(),
        l.prefix, l.prefix_stride_m, l.prefix_stride_n, l.blocks, l.blocks_stride_m,
        l.blocks_stride_r, l.blocks_stride_n, l.norm_weight, l.norm_weight_stride_n, l.qk_weight,
        l.qk_weight_stride_n, l.out_norm_weight, l.out_norm_weight_stride_n, l.out, l.out_stride_m,
        l.out_stride_n, l.num_blocks, l.write_idx, l.eps, l.out_eps, rows, packs, l.reduce_scatter_blocks);
}

inline void launch_all_reduce_rms_norm_gemm(Handle& h, const AllReduceRmsNormGemmLaunch& l) {
  const int64_t bytes = extent_bytes(l.rows, l.hidden, l.inp_stride_m, l.dtype);
  const int rows = static_cast<int>(l.rows), packs = static_cast<int>(packs_of(l.hidden, l.dtype));
  const PeerPtrs* inputs = h.peer_inputs(l.inp, bytes, l.stream);
  if (l.algorithm == Algorithm::one_shot)
    Call<AllReduceRmsNormGemmOneShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, inputs, l.inp_stride_m,
        l.inp_stride_n, h.peer_signals(), h.self_signal(), h.rank(), h.timeout_ticks(),
        l.norm_weight, l.norm_weight_stride_n, l.eps, l.gemm_weight, l.gemm_weight_stride_m,
        l.gemm_weight_stride_n, static_cast<int>(l.n_cols), l.out, l.out_stride_m, l.out_stride_n,
        l.workspace, l.workspace_stride_m, l.workspace_stride_n, rows, packs);
  else
    Call<AllReduceRmsNormGemmTwoShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, inputs, l.inp_stride_m,
        l.inp_stride_n, h.peer_scratch(), l.scratch_stride_m, l.scratch_stride_n,
        h.peer_signals(), h.self_signal(), h.rank(), h.timeout_ticks(), l.norm_weight,
        l.norm_weight_stride_n, l.eps, l.gemm_weight, l.gemm_weight_stride_m,
        l.gemm_weight_stride_n, static_cast<int>(l.n_cols), l.out, l.out_stride_m, l.out_stride_n,
        l.workspace, l.workspace_stride_m, l.workspace_stride_n, rows, packs);
}

inline void launch_all_reduce_rms_norm_gemm_add(Handle& h, const AllReduceRmsNormGemmAddLaunch& l) {
  const int64_t bytes = extent_bytes(l.rows, l.hidden, l.inp_stride_m, l.dtype);
  const int rows = static_cast<int>(l.rows), packs = static_cast<int>(packs_of(l.hidden, l.dtype));
  const PeerPtrs* inputs = h.peer_inputs(l.inp, bytes, l.stream);
  if (l.algorithm == Algorithm::one_shot)
    Call<AllReduceRmsNormGemmOneShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, inputs, l.inp_stride_m,
        l.inp_stride_n, h.peer_signals(), h.self_signal(), h.rank(), h.timeout_ticks(),
        l.norm_weight, l.norm_weight_stride_n, l.eps, l.gemm_weight, l.gemm_weight_stride_m,
        l.gemm_weight_stride_n, static_cast<int>(l.n_cols), l.out, l.out_stride_m, l.out_stride_n,
        l.workspace, l.workspace_stride_m, l.workspace_stride_n, rows, packs);
  else
    Call<AllReduceRmsNormGemmTwoShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, inputs, l.inp_stride_m,
        l.inp_stride_n, h.peer_scratch(), l.scratch_stride_m, l.scratch_stride_n,
        h.peer_signals(), h.self_signal(), h.rank(), h.timeout_ticks(), l.norm_weight,
        l.norm_weight_stride_n, l.eps, l.gemm_weight, l.gemm_weight_stride_m,
        l.gemm_weight_stride_n, static_cast<int>(l.n_cols), l.out, l.out_stride_m, l.out_stride_n,
        l.workspace, l.workspace_stride_m, l.workspace_stride_n, rows, packs);
}

// inp's row is [shared | projected | latent].
inline void launch_all_reduce_rms_scale_add(Handle& h, const AllReduceRmsScaleAddLaunch& l) {
  const int64_t bytes = extent_bytes(l.rows, 2 * l.hidden + l.latent, l.inp_stride_m, l.dtype);
  const int rows      = static_cast<int>(l.rows);
  const int hp = static_cast<int>(packs_of(l.hidden, l.dtype));
  const int lp = static_cast<int>(packs_of(l.latent, l.dtype));
  const PeerPtrs* inputs = h.peer_inputs(l.inp, bytes, l.stream);
  if (l.algorithm == Algorithm::one_shot)
    Call<AllReduceRmsScaleAddOneShotKernel>::run(l.kernel, l.blocks_per_grid, l.threads_per_block,
                                        l.stream, inputs, l.inp_stride_m, l.inp_stride_n,
                                        h.peer_signals(), h.self_signal(), h.rank(),
                                        h.timeout_ticks(), l.out, l.out_stride_m, l.out_stride_n,
                                        l.eps, rows, hp, lp);
  else
    Call<AllReduceRmsScaleAddTwoShotKernel>::run(l.kernel, l.blocks_per_grid, l.threads_per_block,
                                        l.stream, inputs, l.inp_stride_m, l.inp_stride_n,
                                        h.peer_scratch(), l.scratch_stride_m, l.scratch_stride_n,
                                        h.peer_signals(), h.self_signal(), h.rank(),
                                        h.timeout_ticks(), l.out, l.out_stride_m, l.out_stride_n,
                                        l.eps, rows, hp, lp);
}

namespace experimental {

// No peers: its own tensors only.
inline void launch_add_attn_res_rms_norm(const AddAttnResRmsNormLaunch& l) {
  const int rows = static_cast<int>(l.rows), packs = static_cast<int>(packs_of(l.hidden, l.dtype));
  Call<AddAttnResRmsNormKernel>::run(
      l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, l.delta, l.delta_stride_m,
      l.delta_stride_n, l.prefix, l.prefix_stride_m, l.prefix_stride_n, l.blocks, l.blocks_stride_m,
      l.blocks_stride_r, l.blocks_stride_n, l.norm_weight, l.norm_weight_stride_n, l.qk_weight,
      l.qk_weight_stride_n, l.out_norm_weight, l.out_norm_weight_stride_n, l.out, l.out_stride_m,
      l.out_stride_n, l.num_blocks, l.write_idx, l.eps, l.out_eps, rows, packs);
}

// The probe's: a barrier over the signals only; the ping-pong's flags from this pair's next ones
// (flags only grow); the link traffic over every rank's copy of its buffer and staging.
inline void launch_probe_barrier(Handle& h, const ProbeBarrierLaunch& l) {
  Call<ProbeBarrierKernel>::run(l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream,
                                h.peer_signals(), h.self_signal(), h.rank(), h.timeout_ticks());
}

inline void launch_ping_pong(Handle& h, const PingPongLaunch& l) {
  const uint32_t base = h.take_flags(l.peer, ping_pong_flags(l.iters));
  Call<PingPongKernel>::run(l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream,
                            h.peer_signals(), h.self_signal(), h.rank(), h.timeout_ticks(),
                            l.peer, base, l.iters, l.ticks);
}

inline void launch_link_traffic(Handle& h, const LinkTrafficLaunch& l) {
  Call<LinkTrafficKernel>::run(l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream,
                               h.peer_inputs(l.buffer, l.bytes, l.stream), h.peer_staging(),
                               h.peer_signals(), h.self_signal(), h.rank(), h.timeout_ticks(),
                               static_cast<int>(l.mode), l.peer, l.pullers,
                               l.bytes / kBuild.memory.pack_bytes, l.sink);
}

}  // namespace experimental

}  // namespace hip_comms
