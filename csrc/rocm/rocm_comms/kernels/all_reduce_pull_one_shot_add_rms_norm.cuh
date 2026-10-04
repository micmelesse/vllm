// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce then RMSNorm (`all_reduce_pull_one_shot_rms_norm`), and
// all-reduce then add then RMSNorm (`all_reduce_pull_one_shot_add_rms_norm`): one body, a
// kernel per op, so a trace names the op that ran.

#pragma once

#include "../common/common.cuh"

namespace hip_comms {

// Every rank reduces every row and norms it where the sum lands in registers, saving an HBM round
// trip and a launch against an all-reduce then a norm kernel. A block owns a row. `residual` and
// `residual_out` are unused (null) unless kAdd; `weight` keeps its own dtype W (T or fp32), as
// vLLM's reference ops round to the WEIGHT's dtype (`vllm/ir/ops/layernorm.py`).
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, bool ADD_RESIDUAL, int TILE_N, int THREADS_PER_BLOCK>
DINLINE void all_reduce_pull_one_shot_add_rms_norm_body(
    const PeerPtrs* __restrict__ peer_inputs, PeerSignals peer_signals,
    Signal* self_signal, int rank, uint64_t timeout_ticks, DTYPE* __restrict__ out,
    DTYPE* __restrict__ residual_out, const DTYPE* __restrict__ residual, const WEIGHT_DTYPE* __restrict__ weight,
    float eps, int rows, int packs) {
  constexpr int NL       = traits<DTYPE>::N;
  using Row              = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK>;
  using RowF             = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, float>;
  using Weight           = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, WEIGHT_DTYPE>;
  const int cols         = packs * NL;  // the row, in elements
  const float inv_hidden = 1.0f / static_cast<float>(cols);

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto inputs = rank_inputs<DTYPE, WORLD>(*peer_inputs);
  block_stamp(0);
  barrier<WORLD, Among::peers, Ensure::launched>(
      peer_signals, self_signal, rank, timeout_ticks);
  block_stamp(1);
  const auto input = [&](int r) { return inputs[r]; };

  // 2. Each of this block's rows: read it from every rank in rank order and sum, then (ADD_RESIDUAL) add
  //    the residual, then RMSNorm, rounding as the reference does:
  //      s   = float(DTYPE(sum over ranks))             the all-reduce output, as it would land
  //      s  += float(residual); residual_out = DTYPE(s) ADD_RESIDUAL only (fused_add_rms_norm)
  //      out = DTYPE(WEIGHT_DTYPE(WEIGHT_DTYPE(s * rsqrt(mean(s^2) + eps)) * float(w)))
  //    The variance is of `s` before any further rounding, kept in registers between the passes.
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const Row at{rows, cols, row, 0};
    // Every load of the row before any store: the peers' and the residual together, the weight
    // under the reduction.
    Row peers[WORLD];
#pragma unroll
    for (int r = 0; r < WORLD; ++r) peers[r] = at;
    peers_load(peers, input, cols);
    Row res = at;
    if constexpr (ADD_RESIDUAL) tile_load(res, residual, cols);
    RowF s = peers_reduce(peers).template to<float>();
    block_stamp(2);
    if constexpr (ADD_RESIDUAL) {
      s = tile_add(s, res.template to<float>());
      tile_store(residual_out, cols, s.template to<DTYPE>());
    }
    Weight w{1, cols, 0, 0};
    tile_load(w, weight, 0);
    float ss[1];
    partial_dot(s, s, ss);
    block_reduce<Sum, THREADS_PER_BLOCK>(ss);
    block_stamp(3);
    const float scale = rsqrtf(ss[0] * inv_hidden + eps);
    // out = T(W(W(s * scale) * float(w))), as the reference rounds
    Row normed = tile_mul(tile_mul(s, scale).template to<WEIGHT_DTYPE>().template to<float>(),
                            w.template to<float>())
                     .template to<WEIGHT_DTYPE>()
                     .template to<DTYPE>();
    tile_store(out, cols, normed);
  }

  block_stamp(4);
  // 3. No rank may overwrite its input until every peer has read it.
  barrier<WORLD, Among::peers, Ensure::read>(peer_signals, self_signal, rank,
                                                            timeout_ticks);
  block_stamp(5);
}

// THE KERNELS, one per op, both the body above.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot_rms_norm(const PeerPtrs* __restrict__ peer_inputs,
                                      PeerSignals peer_signals, Signal* self_signal,
                                      int rank, uint64_t timeout_ticks, DTYPE* __restrict__ out,
                                      const WEIGHT_DTYPE* __restrict__ weight, float eps, int rows,
                                      int packs) {
  all_reduce_pull_one_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, false, TILE_N, THREADS_PER_BLOCK>(
       peer_inputs, peer_signals, self_signal, rank, timeout_ticks, out, nullptr, nullptr, weight,
      eps, rows, packs);
}

template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot_add_rms_norm(const PeerPtrs* __restrict__ peer_inputs,
                                          PeerSignals peer_signals, Signal* self_signal,
                                          int rank, uint64_t timeout_ticks, DTYPE* __restrict__ out,
                                          DTYPE* __restrict__ residual_out,
                                          const DTYPE* __restrict__ residual,
                                          const WEIGHT_DTYPE* __restrict__ weight, float eps, int rows,
                                          int packs) {
  all_reduce_pull_one_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, true, TILE_N, THREADS_PER_BLOCK>(
       peer_inputs, peer_signals, self_signal, rank, timeout_ticks, out, residual_out, residual,
      weight, eps, rows, packs);
}

}  // namespace hip_comms
