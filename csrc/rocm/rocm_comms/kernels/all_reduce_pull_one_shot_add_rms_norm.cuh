// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce then RMSNorm (`all_reduce_pull_one_shot_rms_norm`), and
// all-reduce then add then RMSNorm (`all_reduce_pull_one_shot_add_rms_norm`): one body, a
// kernel per op, so a trace names the op that ran.

#pragma once

#include "../p2p/p2p.cuh"
#include "../common/common.cuh"

namespace hip_comms {

// Every rank reduces every row and norms it where the sum lands in registers, saving an HBM round
// trip and a launch against an all-reduce then a norm kernel. A block owns a row. `residual` and
// `residual_out` are unused (null) unless kAdd; `weight` keeps its own dtype W (T or fp32), as
// vLLM's reference ops round to the WEIGHT's dtype (`vllm/ir/ops/layernorm.py`).
template <typename T, typename W, int ngpus, bool kAdd, int TILE_N, int THREADS_PER_BLOCK>
DINLINE void all_reduce_pull_one_shot_add_rms_norm_body(
    const p2p::PeerPtrs* __restrict__ peer_inputs, p2p::PeerSignals peer_signals,
    p2p::Signal* self_signal, int rank, uint64_t timeout_ticks, T* __restrict__ out,
    T* __restrict__ residual_out, const T* __restrict__ residual, const W* __restrict__ weight,
    float eps, int rows, int packs) {
  constexpr int NL       = traits<T>::N;
  using Row              = Tile<T, 1, TILE_N, THREADS_PER_BLOCK>;
  using RowF             = Tile<T, 1, TILE_N, THREADS_PER_BLOCK, float>;
  using Weight           = Tile<T, 1, TILE_N, THREADS_PER_BLOCK, W>;
  const int cols         = packs * NL;  // the row, in elements
  const float inv_hidden = 1.0f / static_cast<float>(cols);

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto inputs = p2p::inputs<T, ngpus>(*peer_inputs);
  block_stamp(0);
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(
      peer_signals, self_signal, rank, timeout_ticks);
  block_stamp(1);
  const auto input = [&](int r) { return inputs[r].data(); };

  // 2. Each of this block's rows: read it from every rank in rank order and sum, then (kAdd) add
  //    the residual, then RMSNorm, rounding as the reference does:
  //      s   = float(T(sum over ranks))             the all-reduce output, as it would land
  //      s  += float(residual); residual_out = T(s) kAdd only (fused_add_rms_norm)
  //      out = T(W(W(s * rsqrt(mean(s^2) + eps)) * float(w)))
  //    The variance is of `s` before any further rounding, kept in registers between the passes.
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const Row at{rows, cols, row, 0};
    // Every load of the row before any store: the peers' and the residual together, the weight
    // under the reduction.
    Row peers[ngpus];
#pragma unroll
    for (int r = 0; r < ngpus; ++r) peers[r] = at;
    peers_load(peers, input, cols);
    Row res = at;
    if constexpr (kAdd) thread_load(res, residual, cols);
    RowF s = peers_reduce(peers).template to<float>();
    block_stamp(2);
    if constexpr (kAdd) {
      const RowF r = res.template to<float>();
#pragma unroll
      for (int k = 0; k < RowF::K; ++k)
#pragma unroll
        for (int j = 0; j < NL; ++j) s.v[0][k].d[j] += r.v[0][k].d[j];
      thread_store(residual_out, cols, s.template to<T>());
    }
    Weight w{1, cols, 0, 0};
    thread_load(w, weight, 0);
    float ss[1];
    thread_dot(s, s, ss);
    block_reduce<Sum>(ss);
    block_stamp(3);
    const float scale = rsqrtf(ss[0] * inv_hidden + eps);
    Row normed = at;
#pragma unroll
    for (int k = 0; k < Row::K; ++k)
#pragma unroll
      for (int j = 0; j < NL; ++j) {
        const float x       = static_cast<float>(static_cast<W>(s.v[0][k].d[j] * scale));
        normed.v[0][k].d[j] = static_cast<T>(static_cast<W>(x * static_cast<float>(w.v[0][k].d[j])));
      }
    thread_store(out, cols, normed);
  }

  block_stamp(4);
  // 3. No rank may overwrite its input until every peer has read it.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::read>(peer_signals, self_signal, rank,
                                                            timeout_ticks);
  block_stamp(5);
}

// THE KERNELS, one per op, both the body above.
template <typename T, typename W, int ngpus, int TILE_N, int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)
    all_reduce_pull_one_shot_rms_norm(const p2p::PeerPtrs* __restrict__ peer_inputs,
                                      p2p::PeerSignals peer_signals, p2p::Signal* self_signal,
                                      int rank, uint64_t timeout_ticks, T* __restrict__ out,
                                      const W* __restrict__ weight, float eps, int rows,
                                      int packs) {
  all_reduce_pull_one_shot_add_rms_norm_body<T, W, ngpus, false, TILE_N, THREADS_PER_BLOCK>(
       peer_inputs, peer_signals, self_signal, rank, timeout_ticks, out, nullptr, nullptr, weight,
      eps, rows, packs);
}

template <typename T, typename W, int ngpus, int TILE_N, int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)
    all_reduce_pull_one_shot_add_rms_norm(const p2p::PeerPtrs* __restrict__ peer_inputs,
                                          p2p::PeerSignals peer_signals, p2p::Signal* self_signal,
                                          int rank, uint64_t timeout_ticks, T* __restrict__ out,
                                          T* __restrict__ residual_out,
                                          const T* __restrict__ residual,
                                          const W* __restrict__ weight, float eps, int rows,
                                          int packs) {
  all_reduce_pull_one_shot_add_rms_norm_body<T, W, ngpus, true, TILE_N, THREADS_PER_BLOCK>(
       peer_inputs, peer_signals, self_signal, rank, timeout_ticks, out, residual_out, residual,
      weight, eps, rows, packs);
}

}  // namespace hip_comms
