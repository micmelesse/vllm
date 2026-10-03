// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce fused with Kimi-K3's attention residual (AttnRes) and its
// RMSNorm.

#pragma once

#include "../p2p/p2p.cuh"
#include "../common/common.cuh"
#include "shared/attn_res.cuh"

namespace hip_comms {

// A block owns a row, as the fused norm does: every rank reduces every row, so there is
// nothing to gather. `blocks` is [rows, num_sources, hidden] with row and source strides
// in elements; `write_idx` < 0 writes no block.
template <typename T, int ngpus, bool kPrefix, int TILE_N, int TILE_K, int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)
    all_reduce_pull_one_shot_add_attn_res_rms_norm(const p2p::PeerPtrs* __restrict__ peer_inputs,
                                                   p2p::PeerSignals peer_signals,
                                                   p2p::Signal* self_signal, int rank,
                                                   uint64_t timeout_ticks, T* __restrict__ prefix,
                                                   T* __restrict__ blocks, int64_t block_stride_m,
                                                   int64_t block_stride_r,
                                                   const T* __restrict__ norm_w,
                                                   const T* __restrict__ qk_w,
                                                   const T* __restrict__ out_norm_w,
                                                   T* __restrict__ out, int num_blocks,
                                                   int write_idx, float eps, float out_eps,
                                                   int rows, int packs) {
  using Row              = Tile<T, 1, TILE_N, THREADS_PER_BLOCK>;
  const int cols         = packs * traits<T>::N;  // the row, in elements
  const float inv_hidden = 1.0f / static_cast<float>(cols);

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto inputs = p2p::inputs<T, ngpus>(*peer_inputs);
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(
      peer_signals, self_signal, rank, timeout_ticks);
  const auto input = [&](int r) { return inputs[r].data(); };

  // 2. Each of this block's tiles (one row: TILE_M = 1): read it from every rank in rank order,
  //    sum, AttnRes.
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    Row peers[ngpus];
#pragma unroll
    for (int r = 0; r < ngpus; ++r) peers[r] = Row{rows, cols, row, 0};
    peers_load(peers, input, cols);
    block_attn_res_tile<kPrefix, TILE_K>(peers_reduce(peers), prefix, blocks, block_stride_m,
                                         block_stride_r, write_idx, norm_w, qk_w, out_norm_w, out,
                                         num_blocks, eps, out_eps, inv_hidden);
  }

  // 3. No rank may overwrite its input until every peer has read it.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::read>(peer_signals, self_signal, rank,
                                                            timeout_ticks);
}

}  // namespace hip_comms
