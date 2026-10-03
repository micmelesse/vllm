// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce fused with Kimi-K3's attention residual (AttnRes) and its
// RMSNorm.

#pragma once

#include "../p2p/p2p.cuh"
#include "../common/common.cuh"
#include "shared/attn_res.cuh"

namespace hip_comms {

// THE SLICE IS COLUMNS, AND BOTH HALVES PULL, as the plain two-shot all-reduce moves its bytes:
// each rank sums its columns of every row over the ranks into its own scratch; after the sync each
// block reads its rows' columns from their owners and computes AttnRes on them itself. So the links
// carry what an all-reduce's do, AttnRes's two outputs (the prefix and out) are never gathered, and
// every rank does the AttnRes the unfused path would. A SLICE IS WHOLE WAVES (64 packs), so every
// wave's packs have one owner and p2p::scratch's rank is the same across the wave. EACH PHASE AT
// ITS OWN GRID: the reduce-scatter on a few blocks (reads), AttnRes on all of them (compute a
// row), so a world barrier between them. `blocks` is [rows, num_sources, hidden] with row and
// source strides in elements; `write_idx` < 0 writes no block.
template <typename T, int ngpus, bool kPrefix, int TILE_M, int TILE_N, int TILE_K,
          int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)
    all_reduce_pull_two_shot_add_attn_res_rms_norm(const p2p::PeerPtrs* __restrict__ peer_inputs,
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
                                                   int rows, int packs, int reduce_scatter_blocks) {
  using Rows             = Tile<T, TILE_M, TILE_N, THREADS_PER_BLOCK>;
  const int cols         = packs * traits<T>::N;  // the row, in elements
  const float inv_hidden = 1.0f / static_cast<float>(cols);
  const int per_rank     = (packs + ngpus - 1) / ngpus;
  const int slice        = (per_rank + kWaveSize - 1) / kWaveSize * kWaveSize;
  const int col0         = min(rank * slice, packs);
  const int own_packs = max(0, min(slice, packs - col0));  // a late rank's may be short or none

  // 1. Wait until every peer has launched, so its input is ready.
  block_stamp(0);
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(
      peer_signals, self_signal, rank, timeout_ticks);
  block_stamp(1);

  // THE RANKS' POINTERS AFTER THE BARRIER, as in the other two-shots (held across it they spilled).
  const auto inputs = p2p::inputs<T, ngpus>(*peer_inputs);
  const auto read  = [&](int r, int64_t i) { return p2p::read_input(inputs[r], i); };
  const auto own_scratch  = p2p::scratch<T, ngpus>(peer_scratch, rank);

  // 2. This rank's columns of every row, summed over the ranks in rank order, into this rank's
  //    scratch at their place in the tensor, BY THE FIRST reduce_scatter_blocks BLOCKS
  //    only: reads queue behind the links past a few dozen blocks (its config, select.cuh). THE
  //    (ROW, COLUMN) STEPS, NOT DIVIDED: a 64-bit division a pack was a software routine on every
  //    16 bytes.
  const int reducers = min(static_cast<int>(gridDim.x), reduce_scatter_blocks);
  if (own_packs > 0 && static_cast<int>(blockIdx.x) < reducers) {
    const int reduce_rows = (rows - static_cast<int>(blockIdx.x) + reducers - 1) / reducers;
    int q = threadIdx.x / own_packs;  // this thread's row among the block's, and its column
    int c = threadIdx.x - q * own_packs;
    const int dq = blockDim.x / own_packs, dc = blockDim.x - dq * own_packs;
    for (; q < reduce_rows;) {
      const int64_t i = (int64_t{blockIdx.x} + int64_t{q} * reducers) * packs + col0 + c;
      p2p::write_scratch(own_scratch, i, peers_reduce(peers_load<T, ngpus>(read, i)));
      q += dq;
      c += dc;
      if (c >= own_packs) {
        c -= own_packs;
        ++q;
      }
    }
  }
  block_stamp(2);

  // 3. Every rank's columns are in its scratch, and every peer has read this rank's input: A WORLD
  //    BARRIER, since the blocks that wrote a row's columns are not the ones that read them.
  p2p::barrier<ngpus, p2p::Among::world, p2p::Ensure::visible>(
      peer_signals, self_signal, rank, timeout_ticks);
  block_stamp(3);

  // 4. This block's tiles of TILE_M rows: each pack from the rank that owns its columns, then
  //    AttnRes for every row of the tile at once. The next call's first sync keeps a rank from
  //    overwriting its scratch while it is read (a peer's next kernel starts only once this one has
  //    finished).
  for (int offs_m = blockIdx.x * TILE_M; offs_m < rows; offs_m += gridDim.x * TILE_M) {
    Rows sum{rows, cols, offs_m, 0};
    sliced_load<ngpus>(sum, [&](int r) { return p2p::scratch<T, ngpus>(peer_scratch, r).data(); },
                       cols, slice * traits<T>::N);
    block_attn_res_tile<kPrefix, TILE_K>(sum, prefix, blocks, block_stride_m, block_stride_r,
                                         write_idx, norm_w, qk_w, out_norm_w, out, num_blocks, eps,
                                         out_eps, inv_hidden);
  }
  block_stamp(4);
}

}  // namespace hip_comms
