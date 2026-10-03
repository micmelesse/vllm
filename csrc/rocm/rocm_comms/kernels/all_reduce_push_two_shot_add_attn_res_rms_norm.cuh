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
template <typename DTYPE, int WORLD, bool HAS_PREFIX, int TILE_N, int TILE_K, int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)
    all_reduce_push_two_shot_add_attn_res_rms_norm(const p2p::PeerPtrs* __restrict__ peer_inputs,
                                                   p2p::PeerPtrs peer_scratch,
                                                   p2p::PeerSignals peer_signals,
                                                   p2p::Signal* self_signal, int rank,
                                                   uint64_t timeout_ticks, DTYPE* __restrict__ prefix,
                                                   DTYPE* __restrict__ blocks, int64_t block_stride_m,
                                                   int64_t block_stride_r,
                                                   const DTYPE* __restrict__ norm_w,
                                                   const DTYPE* __restrict__ qk_w,
                                                   const DTYPE* __restrict__ out_norm_w,
                                                   DTYPE* __restrict__ out, int num_blocks,
                                                   int write_idx, float eps, float out_eps,
                                                   int rows, int packs) {
  constexpr int NL       = traits<DTYPE>::N;
  using Row              = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK>;
  const int cols         = packs * traits<DTYPE>::N;  // the row, in elements
  // A REDUCE-SCATTER TILE: THREADS_PER_BLOCK / 64 rows of this rank's columns, a wave a row.
  constexpr int SLICE_N = (TILE_N / WORLD + kWaveSize * NL - 1) / (kWaveSize * NL) * (kWaveSize * NL);
  using Slice = Tile<DTYPE, THREADS_PER_BLOCK / kWaveSize, SLICE_N, THREADS_PER_BLOCK / kWaveSize,
                     kWaveSize>;
  const float inv_hidden = 1.0f / static_cast<float>(cols);
  const int slice        = (packs + WORLD - 1) / WORLD;
  const int col0         = rank * slice;
  const int own_packs = max(0, min(slice, packs - col0));  // the last rank's may be short
  const int my_rows      = rows > static_cast<int>(blockIdx.x)
                               ? (rows - blockIdx.x + gridDim.x - 1) / gridDim.x
                               : 0;

  // 1. Wait until every peer has launched, so its input is ready.
  block_stamp(0);
  p2p::barrier<WORLD, p2p::Among::peers, p2p::Ensure::launched>(
      peer_signals, self_signal, rank, timeout_ticks);
  block_stamp(1);

  // THE RANKS' POINTERS AFTER THE BARRIER, as in the pull kernels (held across it they spilled).
  const auto inputs = p2p::inputs<DTYPE, WORLD>(*peer_inputs);
  const auto scratches = p2p::scratches<DTYPE, WORLD>(peer_scratch);
  const auto input = [&](int r) { return inputs[r].data(); };

  // 2. This rank's columns of this block's rows, summed over the ranks in rank order and pushed
  //    to every rank (itself too), at their place in the tensor: tiles of this block's rows (every
  //    gridDim.x-th), A WAVE A ROW, since a rank's columns are a narrow slice.
  if (own_packs > 0) {
    for (int q = 0; q < my_rows; q += Slice::kThreadsM) {
      const Slice at{rows, (col0 + own_packs) * NL, static_cast<int>(blockIdx.x + q * gridDim.x),
                     col0 * NL, static_cast<int>(gridDim.x)};
      Slice peers[WORLD];
#pragma unroll
      for (int r = 0; r < WORLD; ++r) peers[r] = at;
      peers_load(peers, input, cols);
      const Slice sum = peers_reduce(peers);
#pragma unroll
      for (int r = 0; r < WORLD; ++r) tile_store(scratches[r].data(), cols, sum);
    }
  }
  block_stamp(2);

  // 3. Every rank's sums are in this rank's scratch, and every peer has read this rank's input.
  p2p::barrier<WORLD, p2p::Among::peers, p2p::Ensure::visible>(
      peer_signals, self_signal, rank, timeout_ticks);
  block_stamp(3);

  // 4. This block's tiles (one row): the sum out of this rank's scratch, then AttnRes, as the
  //    one-shot does. The next call's first sync keeps a peer from pushing into this scratch while
  //    it is read (a peer's next kernel starts only once this one has finished).
  const auto own_scratch = p2p::scratch<DTYPE, WORLD>(peer_scratch, rank);
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    Row sum{rows, cols, row, 0};
    tile_load(sum, own_scratch.data(), cols);
    block_attn_res_tile<HAS_PREFIX, TILE_K>(sum, prefix, blocks, block_stride_m, block_stride_r,
                                         write_idx, norm_w, qk_w, out_norm_w, out, num_blocks, eps,
                                         out_eps, inv_hidden);
  }
  block_stamp(4);
}

}  // namespace hip_comms
