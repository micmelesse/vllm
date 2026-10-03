// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce, then RMSNorm, then a GEMM whose result is written into an output
// (`all_reduce_pull_one_shot_rms_norm_gemm`) or added into it (`..._gemm_add`, the tail of
// Kimi-K3's latent MoE): one body, a kernel per op, so a trace names the op that ran.

#pragma once

#include "../p2p/p2p.cuh"
#include "../common/common.cuh"

namespace hip_comms {

// Every rank reduces and norms every row into `workspace` ([rows, packs] of its own); a
// grid barrier; the GEMM over every row, TILE_M a pass.
// SLICE_K is the GEMM's lanes a column, TILE_K its K staged in LDS a pass.
template <typename DTYPE, int NGPUS, int TILE_M, int TILE_N, int TILE_K, int SLICE_K,
          int THREADS_PER_BLOCK, bool ADD_RESIDUAL>
DINLINE void all_reduce_pull_one_shot_rms_norm_gemm_body(
    const p2p::PeerPtrs* __restrict__ peer_inputs, p2p::PeerSignals peer_signals,
    p2p::Signal* self_signal, int rank, uint64_t timeout_ticks, const DTYPE* __restrict__ norm_w,
    float eps, const DTYPE* __restrict__ gemm_w, int n_cols, DTYPE* __restrict__ out, int64_t out_stride,
    DTYPE* __restrict__ workspace, int rows, int packs) {
  constexpr int NL       = traits<DTYPE>::N;
  using Row              = Tile<DTYPE, 1, TILE_N, THREADS_PER_BLOCK>;
  using RowF             = Tile<DTYPE, 1, TILE_N, THREADS_PER_BLOCK, float>;
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  using V                = typename traits<DTYPE>::V;
  V* normed              = reinterpret_cast<V*>(workspace);
  const int cols = packs * NL;  // the row, in elements

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto inputs = p2p::inputs<DTYPE, NGPUS>(*peer_inputs);
  block_stamp(0);
  p2p::barrier<NGPUS, p2p::Among::peers, p2p::Ensure::launched>(
      peer_signals, self_signal, rank, timeout_ticks);
  block_stamp(1);
  const auto input = [&](int r) { return inputs[r].data(); };

  // 2. Each of this block's rows: read it from every rank in rank order, sum, norm, into the
  //    workspace.
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    Row peers[NGPUS];
#pragma unroll
    for (int r = 0; r < NGPUS; ++r) peers[r] = Row{rows, cols, row, 0};
    peers_load(peers, input, cols);
    const RowF s = peers_reduce(peers).template to<float>();
    // The norm, rounding as vLLM's reference rms_norm does (weight in DTYPE):
    //   out = DTYPE(DTYPE(s * rsqrt(mean(s^2) + eps)) * float(w)), s = float(DTYPE(sum over ranks))
    Row wk{1, cols, 0, 0};
    thread_load(wk, norm_w, 0);  // under the reduction
    float ss[1];
    thread_dot(s, s, ss);
    block_reduce<Sum>(ss);
    const float scale = rsqrtf(ss[0] * inv_hidden + eps);
    const RowF w = wk.template to<float>();
    RowF x{rows, cols, row, 0};
#pragma unroll
    for (int k = 0; k < RowF::K; ++k)
#pragma unroll
      for (int j = 0; j < NL; ++j)
        x.v[0][k].d[j] = static_cast<float>(static_cast<DTYPE>(s.v[0][k].d[j] * scale)) * w.v[0][k].d[j];
    thread_store(workspace, cols, x.template to<DTYPE>());
  }

  // 3. The GEMM reads rows other blocks of this rank wrote.
  block_stamp(2);
  p2p::barrier<NGPUS, p2p::Among::grid, p2p::Ensure::visible>(
      peer_signals, self_signal, rank, timeout_ticks);
  block_stamp(3);

  // 4. The GEMM over every row.
  for (int r0 = 0; r0 < rows; r0 += TILE_M)
    grid_gemm<TILE_M, TILE_K, SLICE_K, ADD_RESIDUAL, DTYPE>([&](int r) { return normed + (r0 + r) * packs; },
                                                min(TILE_M, rows - r0), gemm_w, n_cols, packs,
                                                out + r0 * out_stride, out_stride);

  block_stamp(4);
  // 5. No rank may overwrite its input until every peer has read it.
  p2p::barrier<NGPUS, p2p::Among::peers, p2p::Ensure::read>(peer_signals, self_signal, rank,
                                                            timeout_ticks);
}

// THE KERNELS, one per op, both the body above: the GEMM's result written (rms_norm_gemm) or
// added into `out` (rms_norm_gemm_add).
template <typename DTYPE, int NGPUS, int TILE_M, int TILE_N, int TILE_K, int SLICE_K,
          int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)
    all_reduce_pull_one_shot_rms_norm_gemm(const p2p::PeerPtrs* __restrict__ peer_inputs,
                                           p2p::PeerSignals peer_signals, p2p::Signal* self_signal,
                                           int rank, uint64_t timeout_ticks,
                                           const DTYPE* __restrict__ norm_w, float eps,
                                           const DTYPE* __restrict__ gemm_w, int n_cols,
                                           DTYPE* __restrict__ out, int64_t out_stride,
                                           DTYPE* __restrict__ workspace, int rows, int packs) {
  if constexpr (gemm_fits(kDevice, TILE_M, TILE_K, SLICE_K, THREADS_PER_BLOCK))
    all_reduce_pull_one_shot_rms_norm_gemm_body<DTYPE, NGPUS, TILE_M, TILE_N, TILE_K, SLICE_K,
                                                THREADS_PER_BLOCK, false>(
                                                     peer_inputs, peer_signals, self_signal, rank,
                                                    timeout_ticks, norm_w, eps, gemm_w, n_cols, out,
                                                    out_stride, workspace, rows, packs);
  else
    __builtin_trap();
}

template <typename DTYPE, int NGPUS, int TILE_M, int TILE_N, int TILE_K, int SLICE_K,
          int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)
    all_reduce_pull_one_shot_rms_norm_gemm_add(const p2p::PeerPtrs* __restrict__ peer_inputs,
                                               p2p::PeerSignals peer_signals,
                                               p2p::Signal* self_signal, int rank,
                                               uint64_t timeout_ticks, const DTYPE* __restrict__ norm_w,
                                               float eps, const DTYPE* __restrict__ gemm_w, int n_cols,
                                               DTYPE* __restrict__ out, int64_t out_stride,
                                               DTYPE* __restrict__ workspace, int rows, int packs) {
  if constexpr (gemm_fits(kDevice, TILE_M, TILE_K, SLICE_K, THREADS_PER_BLOCK))
    all_reduce_pull_one_shot_rms_norm_gemm_body<DTYPE, NGPUS, TILE_M, TILE_N, TILE_K, SLICE_K,
                                                THREADS_PER_BLOCK, true>(
                                                     peer_inputs, peer_signals, self_signal, rank,
                                                    timeout_ticks, norm_w, eps, gemm_w, n_cols, out,
                                                    out_stride, workspace, rows, packs);
  else
    __builtin_trap();
}

}  // namespace hip_comms
