// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce fused with Kimi-K3's attention residual (AttnRes) and its
// RMSNorm.

#pragma once

#include "../common/common.cuh"
#include "shared/attn_res.cuh"

namespace hip_comms {

// THE SLICE IS COLUMNS, AND BOTH HALVES PULL, as the plain two-shot all-reduce moves its bytes:
// each rank sums its columns of every row over the ranks into its own scratch; after the sync each
// block reads its rows' columns from their owners and computes AttnRes on them itself. So the links
// carry what an all-reduce's do, AttnRes's two outputs (the prefix and out) are never gathered, and
// every rank does the AttnRes the unfused path would. A SLICE IS WHOLE WAVES (64 packs), so every
// wave's packs have one owner and scratch's rank is the same across the wave. EACH PHASE AT
// ITS OWN GRID: the reduce-scatter on a few blocks (reads), AttnRes on all of them (compute a
// row), so a world barrier between them. `blocks` is [rows, num_sources, hidden] with row and
// source strides in elements; `write_idx` < 0 writes no block.
template <typename DTYPE, int WORLD, bool HAS_PREFIX, int TILE_M, int TILE_N, int TILE_K,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot_add_attn_res_rms_norm(const PeerPtrs* __restrict__ peer_inputs,
                                                   PeerPtrs peer_scratch,
                                                   PeerSignals peer_signals,
                                                   Signal* self_signal, int rank,
                                                   uint64_t timeout_ticks, DTYPE* __restrict__ prefix,
                                                   DTYPE* __restrict__ blocks, int64_t block_stride_m,
                                                   int64_t block_stride_r,
                                                   const DTYPE* __restrict__ norm_w,
                                                   const DTYPE* __restrict__ qk_w,
                                                   const DTYPE* __restrict__ out_norm_w,
                                                   DTYPE* __restrict__ out, int num_blocks,
                                                   int write_idx, float eps, float out_eps,
                                                   int rows, int packs, int reduce_scatter_blocks) {
  Sync<WORLD> sync{peer_signals, self_signal, rank, timeout_ticks};
  using Rows             = Tile<DTYPE, TILE_M, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  const int cols         = packs * traits<DTYPE>::N;  // the row, in elements
  constexpr int NL = traits<DTYPE>::N;
  // A REDUCE-SCATTER TILE: a row of threads as wide as a rank's slice of a TILE_N row (in whole
  // waves), a group a thread, so a row's slice is one round trip (a wave a row took two at 7168,
  // 0.8 us at 16-32 tokens), and as many rows at once as the block holds. A group a thread, not
  // a slice's worth: every peer's groups of a whole slice were 32 packs at 16384, spilling.
  constexpr int kSliceWaves = (TILE_N / WORLD / NL + kWaveSize - 1) / kWaveSize * kWaveSize;
  constexpr int SLICE_LANES = kSliceWaves < THREADS_PER_BLOCK ? kSliceWaves : THREADS_PER_BLOCK;
  using Slice = Tile<DTYPE, THREADS_PER_BLOCK / SLICE_LANES, SLICE_LANES * NL,
                     THREADS_PER_BLOCK / SLICE_LANES, SLICE_LANES, THREADS_PER_BLOCK>;
  const float inv_hidden = 1.0f / static_cast<float>(cols);
  const int per_rank     = (packs + WORLD - 1) / WORLD;
  const int slice        = (per_rank + kWaveSize - 1) / kWaveSize * kWaveSize;
  const int col0         = min(rank * slice, packs);
  const int own_packs = max(0, min(slice, packs - col0));  // a late rank's may be short or none

  // 1. Wait until every peer has launched, so its input is ready.
  block_stamp(0);
  barrier<Group::peers, Until::launched>(sync);
  block_stamp(1);

  // THE RANKS' POINTERS AFTER THE BARRIER, as in the other two-shots (held across it they spilled).
  const auto inputs = rank_inputs<DTYPE, WORLD>(*peer_inputs);
  const auto input = [&](int r) { return inputs[r]; };
  const auto own_scratch  = rank_scratch<DTYPE, WORLD>(peer_scratch, rank);

  // 2. This rank's columns of every row, summed over the ranks in rank order, into this rank's
  //    scratch at their place in the tensor, BY THE FIRST reduce_scatter_blocks BLOCKS
  //    only: reads queue behind the links past a few dozen blocks (its config, select.cuh). Tiles
  //    of a reducer's rows (every reducers-th), a wave a row.
  const int reducers = min(static_cast<int>(gridDim.x), reduce_scatter_blocks);
  if (own_packs > 0 && static_cast<int>(blockIdx.x) < reducers) {
    const int reduce_rows = (rows - static_cast<int>(blockIdx.x) + reducers - 1) / reducers;
    for (int q = 0; q < reduce_rows; q += Slice::kThreadsM)
    for (int c = col0 * NL; c < (col0 + own_packs) * NL; c += Slice::kTileN) {
      const Slice at{rows, (col0 + own_packs) * NL, static_cast<int>(blockIdx.x) + q * reducers,
                     c, reducers};
      Slice peers[WORLD];
#pragma unroll
      for (int r = 0; r < WORLD; ++r) peers[r] = at;
      peers_load(peers, input, cols);
      tile_store(own_scratch, cols, peers_reduce(peers));
    }
  }
  block_stamp(2);

  // 3. Every rank's columns are in its scratch, and every peer has read this rank's input: A WORLD
  //    BARRIER, since the blocks that wrote a row's columns are not the ones that read them.
  barrier<Group::world, Until::visible>(sync);
  block_stamp(3);

  // 4. This block's tiles of TILE_M rows: each pack from the rank that owns its columns, then
  //    AttnRes for every row of the tile at once. The next call's first sync keeps a rank from
  //    overwriting its scratch while it is read (a peer's next kernel starts only once this one has
  //    finished).
  for (int offs_m = blockIdx.x * TILE_M; offs_m < rows; offs_m += gridDim.x * TILE_M) {
    Rows sum{rows, cols, offs_m, 0};
    sliced_load<WORLD>(sum, [&](int r) { return rank_scratch<DTYPE, WORLD>(peer_scratch, r); },
                       cols, slice * traits<DTYPE>::N);
    block_attn_res_tile<HAS_PREFIX, TILE_K>(sum, prefix, blocks, block_stride_m, block_stride_r,
                                         write_idx, norm_w, qk_w, out_norm_w, out, num_blocks, eps,
                                         out_eps, inv_hidden);
  }
  block_stamp(4);
  sync.finish();
}

}  // namespace hip_comms
