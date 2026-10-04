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
// wave's packs have one owner and scratch's rank is the same across the wave. A BLOCK REDUCES
// THE ROWS IT THEN NORMS, so it waits only for its twin on every rank (a peers barrier, as the
// plain two-shot's) and the grid is AttnRes's to size. The reduce-scatter on 32 blocks behind a
// world barrier left every other block waiting as long as it then computed, on a grid the barrier
// capped at the 192 blocks resident (thread trace 2026-10-04T20-14-36Z). `blocks` is [rows,
// num_sources, hidden] with row and source strides in elements; `write_idx` < 0 writes no block.
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
  static_assert(TILE_M == 1, "a block reduces the rows it norms, one a tile");
  Sync<WORLD> sync{peer_signals, self_signal, rank, timeout_ticks};
  using Rows             = Tile<DTYPE, TILE_M, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  const int cols         = packs * traits<DTYPE>::N;  // the row, in elements
  constexpr int NL = traits<DTYPE>::N;
  // THE REDUCE-SCATTER'S CHUNK, the plain two-shot's: its rows are the ranks, a row of threads a
  // rank (common's reduce_scatter).
  constexpr int LANES = THREADS_PER_BLOCK / WORLD;
  static_assert(LANES * WORLD == THREADS_PER_BLOCK, "a block is a row of threads a rank");
  using Ranks = Tile<DTYPE, WORLD, LANES * NL, WORLD, LANES, THREADS_PER_BLOCK>;
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
  // THE PEERS READ ROTATED, rank + r's r-th, as the plain two-shot reads them: in rank order every
  // GPU read rank 0 first, a link at a time carrying the machine's reads, and the reduce-scatter
  // took 1.6x the plain two-shot's cycles for its bytes (thread traces 2026-10-04T21-03-09Z,
  // 21-26-11Z). Each slice is summed by one rank, so the order differing by rank is harmless.
  const auto input = [&](int r) { return inputs[(rank + r) % WORLD]; };
  const auto own_scratch  = rank_scratch<DTYPE, WORLD>(peer_scratch, rank);

  // 2. Reduce-scatter: this rank's columns of this block's rows (every gridDim.x-th, AttnRes's
  //    below) summed over the ranks into this rank's scratch, at their place in the tensor, a
  //    chunk at a time.
  (void)reduce_scatter_blocks;
  const int end = (col0 + own_packs) * NL;  // this rank's columns end, in elements
  for (int row = blockIdx.x; row < rows; row += gridDim.x)
    for (int c = col0 * NL; c < end; c += Ranks::kTileN) {
      Ranks chunk{WORLD, end, 0, c};
      reduce_scatter(chunk, [&](int w) { return input(w) + int64_t{row} * cols; },
                     own_scratch + int64_t{row} * cols);
    }
  block_stamp(2);

  // 3. This block's rows' columns are in every rank's scratch: its twin on every rank wrote them.
  barrier<Group::peers, Until::visible>(sync);
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
