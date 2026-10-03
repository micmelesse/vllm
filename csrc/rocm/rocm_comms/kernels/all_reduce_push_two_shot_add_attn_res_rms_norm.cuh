// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot all-reduce (the reduce-scatter pulled, the all-gather pushed) fused with Kimi-K3's
// attention residual (AttnRes) and its RMSNorm: the push norm's two phases, with the one-shot
// AttnRes's row in place of the norm.

#pragma once

#include "../p2p/p2p.cuh"
#include "../common/common.cuh"
#include "shared/attn_res.cuh"

namespace hip_comms {

// THE SLICE IS COLUMNS, as the push norm's: each rank owns packs [rank * S, rank * S + S) of every
// row, sums them over the ranks and pushes the sums into every rank's scratch, so after the sync
// each rank holds the whole sum locally and computes AttnRes on its rows itself: what crosses the
// links is the sum, once, and AttnRes's two outputs (the prefix and out) are never gathered. ROW q
// BELONGS TO BLOCK q % gridDim.x IN BOTH PHASES: after the sync a block may read only what the
// same block on a peer wrote. `blocks` is [rows, num_sources, hidden] with row and source strides
// in elements; `write_idx` < 0 writes no block.
template <typename T, int ngpus, bool kPrefix, int TILE_N, int TILE_K, int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)
    all_reduce_push_two_shot_add_attn_res_rms_norm(const p2p::PeerPtrs* __restrict__ peer_inputs,
                                                   p2p::PeerPtrs peer_scratch,
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
  constexpr int kRowPacks = packs_per_thread<T, TILE_N, THREADS_PER_BLOCK>();
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  V* pre                 = reinterpret_cast<V*>(prefix);
  V* o                   = reinterpret_cast<V*>(out);
  const int slice        = (packs + ngpus - 1) / ngpus;
  const int col0         = rank * slice;
  const int own_packs = max(0, min(slice, packs - col0));  // the last rank's may be short
  const int my_rows      = rows > static_cast<int>(blockIdx.x)
                               ? (rows - blockIdx.x + gridDim.x - 1) / gridDim.x
                               : 0;
  const int cols = packs * NL;  // the row, in elements
  const auto thread_cols = thread_offs<T, THREADS_PER_BLOCK>(Tile<1, TILE_N>{rows, cols, 0, 0});
  // The block row `row` writes, or none.
  auto written = [&](int row) -> V* {
    return write_idx < 0 ? nullptr
                         : reinterpret_cast<V*>(blocks + row * block_stride_m +
                                                write_idx * block_stride_r);
  };

  // 1. Wait until every peer has launched, so its input is ready.
  block_stamp(0);
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(
      peer_signals, self_signal, rank, timeout_ticks);
  block_stamp(1);

  // THE RANKS' POINTERS AFTER THE BARRIER, as in the pull kernels (held across it they spilled).
  const auto inputs = p2p::inputs<T, ngpus>(*peer_inputs);
  const auto scratches = p2p::scratches<T, ngpus>(peer_scratch);
  const auto read  = [&](int r, int64_t i) { return p2p::read_input(inputs[r], i); };

  // 2. This rank's columns of this block's rows, summed over the ranks in rank order and pushed
  //    to every rank (itself too), at their place in the tensor.
  for (int64_t e = threadIdx.x; e < int64_t{my_rows} * own_packs; e += blockDim.x) {
    const int64_t q = e / own_packs;
    const int64_t row = blockIdx.x + q * gridDim.x;
    const int64_t i = row * packs + col0 + (e - q * own_packs);
    const V sum       = peers_reduce(peers_load<T, ngpus>(read, i));
#pragma unroll
    for (int r = 0; r < ngpus; ++r) p2p::write_scratch(scratches[r], i, sum);
  }
  block_stamp(2);

  // 3. Every rank's sums are in this rank's scratch, and every peer has read this rank's input.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::visible>(
      peer_signals, self_signal, rank, timeout_ticks);
  block_stamp(3);

  // 4. This block's tiles (one row): the sum out of this rank's scratch, then AttnRes, as the
  //    one-shot does. The next call's first sync keeps a peer from pushing into this scratch while
  //    it is read (a peer's next kernel starts only once this one has finished).
  const auto own_scratch = p2p::scratch<T, ngpus>(peer_scratch, rank);
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const Tile<1, TILE_N> tile{rows, cols, row, 0};
    const int64_t base = int64_t{row} * packs;
    V sum[1][kRowPacks];
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k)
      sum[0][k] = p2p::read_scratch(own_scratch, base + thread_cols.offs_n[k]);
    block_attn_res_tile<T, kPrefix, TILE_K>(sum, tile, thread_cols, pre, written, blocks,
                                            block_stride_m, block_stride_r, norm_w, qk_w,
                                            out_norm_w, o, num_blocks, eps, out_eps, inv_hidden);
  }
  block_stamp(4);
}

}  // namespace hip_comms
