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
template <typename T, int ngpus, bool kPrefix, int BLOCK_N, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS, 1) all_reduce_pull_one_shot_add_attn_res_rms_norm(
    p2p::DevComm p, T* __restrict__ prefix, T* __restrict__ blocks, int64_t block_stride_m,
    int64_t block_stride_r, const T* __restrict__ norm_w, const T* __restrict__ qk_w,
    const T* __restrict__ out_norm_w, T* __restrict__ out, int num_blocks, int write_idx, float eps,
    float out_eps, int rows, int packs) {
  constexpr int kRowPacks = packs_per_thread<T, BLOCK_N, NUM_THREADS>();
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  V* pre                 = reinterpret_cast<V*>(prefix);
  V* o                   = reinterpret_cast<V*>(out);
  const int cols = packs * NL;  // the row, in elements
  const auto thread_cols = thread_offs<T, NUM_THREADS>(Tile<1, BLOCK_N>{rows, cols, 0, 0});
  // The block row `row` writes, or none.
  auto written = [&](int row) -> V* {
    return write_idx < 0 ? nullptr
                         : reinterpret_cast<V*>(blocks + row * block_stride_m +
                                                write_idx * block_stride_r);
  };

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto inputs = p2p::inputs<T, ngpus>(p);
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);
  const auto read = [&](int r, int64_t i) { return p2p::read_input(inputs[r], i); };

  // 2. Each of this block's tiles (one row: BLOCK_M = 1): read it from every rank in rank order,
  //    sum, AttnRes.
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const Tile<1, BLOCK_N> tile{rows, cols, row, 0};
    V sum[1][kRowPacks];
    peers_reduce(peers_load<T, ngpus>(read, row, packs, thread_cols), sum[0]);
    block_attn_res_tile<T, kPrefix>(sum, tile, thread_cols, pre, written, blocks, block_stride_m,
                                    block_stride_r, norm_w, qk_w, out_norm_w, o, num_blocks, eps,
                                    out_eps, inv_hidden);
  }

  // 3. No rank may overwrite its input until every peer has read it.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::read>(p);
}

}  // namespace hip_comms
