// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce, then RMSNorm, then a GEMM whose result is written into an output
// (`all_reduce_pull_one_shot_rms_norm_gemm`) or added into it (`..._gemm_add`, the tail of
// Kimi-K3's latent MoE): one body, a kernel per op, so a trace names the op that ran.

#pragma once

#include "../common/interface.cuh"

namespace hip_comms {

// Every rank reduces and norms every row into `workspace` ([m, n] of its own); a
// grid barrier; the GEMM over every row, TILE_M a pass.
// SLICE_K is the GEMM's lanes a column, TILE_K its K staged in LDS a pass.
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N, int TILE_K, int SLICE_K,
          int THREADS_PER_BLOCK, bool ADD_RESIDUAL>
DINLINE void all_reduce_pull_one_shot_rms_norm_gemm_body(
    const PeerPtrs* __restrict__ peer_inputs, int64_t inp_stride_m, int64_t inp_stride_n,
    PeerSignals peer_signals, Signal* self_signal, int rank, uint64_t timeout_ticks,
    Ptr<const DTYPE> norm_w,
    float eps, Ptr<const DTYPE> gemm_w, int n_cols, Ptr<DTYPE> out, Ptr<DTYPE> workspace, int m,
    int n) {
  Sync<WORLD> sync{peer_signals, self_signal, rank, timeout_ticks};
  using Row              = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  using RowF             = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK, float>;
  const float inv_hidden = 1.0f / static_cast<float>(n);

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto inputs = rank_ptrs<const DTYPE, WORLD>(*peer_inputs, inp_stride_m, inp_stride_n);
  block_stamp(0);
  barrier<Group::peers, Until::launched>(sync);
  block_stamp(1);

  // 2. Each of this block's rows: read it from every rank in rank order, sum, norm, into the
  //    workspace.
  for (int row = blockIdx.x; row < m; row += gridDim.x) {
    Row peers[WORLD];
#pragma unroll
    for (int r = 0; r < WORLD; ++r) peers[r] = Row{m, n, row, 0};
    tile_load(peers, inputs);
    const RowF s = peers_reduce(peers).template to<float>();
    // The norm, rounding as vLLM's reference rms_norm does (weight in DTYPE):
    //   out = DTYPE(DTYPE(s * rsqrt(mean(s^2) + eps)) * float(w)), s = float(DTYPE(sum over ranks))
    Row wk{1, n, 0, 0};
    tile_load(wk, norm_w);  // under the reduction
    float ss[1];
    partial_dot(s, s, ss);
    block_reduce<Sum, THREADS_PER_BLOCK>(ss);
    const float scale = rsqrtf(ss[0] * inv_hidden + eps);
    const RowF w = wk.template to<float>();
    // out = T(T(s * scale) * float(w)), as the reference rounds
    Row x = tile_mul(tile_mul(s, scale).template to<DTYPE>().template to<float>(), w)
                .template to<DTYPE>();
    tile_store(x, workspace);
  }

  // 3. The GEMM reads rows other blocks of this rank wrote.
  block_stamp(2);
  barrier<Group::grid, Until::visible>(sync);
  block_stamp(3);

  // 4. The GEMM over every row.
  for (int r0 = 0; r0 < m; r0 += TILE_M)
    grid_gemm<TILE_M, TILE_K, SLICE_K, ADD_RESIDUAL, THREADS_PER_BLOCK>(
        workspace.data + r0 * workspace.stride_m, workspace.stride_m, min(TILE_M, m - r0), n,
        gemm_w.data, gemm_w.stride_m, n_cols, out.data + r0 * out.stride_m, out.stride_m);

  block_stamp(4);
  // 5. No rank may overwrite its input until every peer has read it.
  barrier<Group::peers, Until::read>(sync);
  sync.finish();
}

// THE KERNELS, one per op, both the body above: the GEMM's result written (rms_norm_gemm) or
// added into `out` (rms_norm_gemm_add).
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N, int TILE_K, int SLICE_K,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot_rms_norm_gemm(
    const PeerPtrs* __restrict__ peer_inputs,
    int64_t inp_stride_m, int64_t inp_stride_n, PeerSignals peer_signals, Signal* self_signal,
    int rank, uint64_t timeout_ticks, const DTYPE* __restrict__ norm_w_ptr,
    int64_t norm_w_stride_n, float eps, const DTYPE* __restrict__ gemm_w_ptr,
    int64_t gemm_w_stride_m, int64_t gemm_w_stride_n, int n_cols, DTYPE* __restrict__ out_ptr,
    int64_t out_stride_m, int64_t out_stride_n, DTYPE* __restrict__ workspace_ptr,
    int64_t workspace_stride_m, int64_t workspace_stride_n, int m, int n) {
  if constexpr (gemm_fits(kDevice, TILE_M, TILE_K, SLICE_K, THREADS_PER_BLOCK))
    all_reduce_pull_one_shot_rms_norm_gemm_body<DTYPE, WORLD, TILE_M, TILE_N, TILE_K, SLICE_K,
                                                THREADS_PER_BLOCK, false>(
        peer_inputs, inp_stride_m, inp_stride_n, peer_signals, self_signal, rank, timeout_ticks,
        local_ptr(norm_w_ptr, 0, norm_w_stride_n, rank), eps,
        local_ptr(gemm_w_ptr, gemm_w_stride_m, gemm_w_stride_n, rank), n_cols,
        local_ptr(out_ptr, out_stride_m, out_stride_n, rank),
        local_ptr(workspace_ptr, workspace_stride_m, workspace_stride_n, rank), m, n);
  else
    __builtin_trap();
}

template <typename DTYPE, int WORLD, int TILE_M, int TILE_N, int TILE_K, int SLICE_K,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot_rms_norm_gemm_add(
    const PeerPtrs* __restrict__ peer_inputs,
    int64_t inp_stride_m, int64_t inp_stride_n, PeerSignals peer_signals, Signal* self_signal,
    int rank, uint64_t timeout_ticks, const DTYPE* __restrict__ norm_w_ptr,
    int64_t norm_w_stride_n, float eps, const DTYPE* __restrict__ gemm_w_ptr,
    int64_t gemm_w_stride_m, int64_t gemm_w_stride_n, int n_cols, DTYPE* __restrict__ out_ptr,
    int64_t out_stride_m, int64_t out_stride_n, DTYPE* __restrict__ workspace_ptr,
    int64_t workspace_stride_m, int64_t workspace_stride_n, int m, int n) {
  if constexpr (gemm_fits(kDevice, TILE_M, TILE_K, SLICE_K, THREADS_PER_BLOCK))
    all_reduce_pull_one_shot_rms_norm_gemm_body<DTYPE, WORLD, TILE_M, TILE_N, TILE_K, SLICE_K,
                                                THREADS_PER_BLOCK, true>(
        peer_inputs, inp_stride_m, inp_stride_n, peer_signals, self_signal, rank, timeout_ticks,
        local_ptr(norm_w_ptr, 0, norm_w_stride_n, rank), eps,
        local_ptr(gemm_w_ptr, gemm_w_stride_m, gemm_w_stride_n, rank), n_cols,
        local_ptr(out_ptr, out_stride_m, out_stride_n, rank),
        local_ptr(workspace_ptr, workspace_stride_m, workspace_stride_n, rank), m, n);
  else
    __builtin_trap();
}

}  // namespace hip_comms
