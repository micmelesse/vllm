// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce fused with Kimi-K3's attention residual (AttnRes) and its
// RMSNorm.

#pragma once

#include "../common/interface.cuh"
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
// capped at the 192 blocks resident (thread trace 2026-10-04T20-14-36Z). `blocks` is [m,
// num_sources, n] with row and source strides in elements; `write_idx` < 0 writes no block.
template <typename DTYPE, int WORLD, bool HAS_PREFIX, int TILE_M, int TILE_N, int TILE_K,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot_add_attn_res_rms_norm(
        const PeerPtrs* __restrict__ peer_inputs, int64_t inp_stride_m, int64_t inp_stride_n,
        PeerPtrs peer_scratch, int64_t scratch_stride_m, int64_t scratch_stride_n,
        PeerSignals peer_signals, Signal* self_signal, int rank, uint64_t timeout_ticks,
        DTYPE* __restrict__ prefix_ptr, int64_t prefix_stride_m, int64_t prefix_stride_n,
        DTYPE* __restrict__ blocks_ptr, int64_t blocks_stride_m, int64_t blocks_stride_r,
        int64_t blocks_stride_n, const DTYPE* __restrict__ norm_w_ptr, int64_t norm_w_stride_n,
        const DTYPE* __restrict__ qk_w_ptr, int64_t qk_w_stride_n,
        const DTYPE* __restrict__ out_norm_w_ptr, int64_t out_norm_w_stride_n,
        DTYPE* __restrict__ out_ptr, int64_t out_stride_m, int64_t out_stride_n, int num_blocks,
        int write_idx, float eps, float out_eps, int m, int n, int reduce_scatter_blocks) {
  Sync<WORLD> sync{peer_signals, self_signal, rank, timeout_ticks};
  using Rows             = Tile<DTYPE, TILE_M, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  constexpr int NL = traits<DTYPE>::N;
  const int packs        = n / NL;  // the row, in packs (n is whole packs)
  // THE REDUCE-SCATTER'S CHUNK, the plain two-shot's: its rows are the ranks, a row of threads a
  // rank (common's reduce_scatter).
  constexpr int LANES = THREADS_PER_BLOCK / WORLD;
  static_assert(LANES * WORLD == THREADS_PER_BLOCK, "a block is a row of threads a rank");
  using Ranks = Tile<DTYPE, WORLD, LANES * NL, WORLD, LANES, THREADS_PER_BLOCK>;
  const float inv_hidden = 1.0f / static_cast<float>(n);
  const int per_rank     = (packs + WORLD - 1) / WORLD;
  // EVERY RANK AN EQUAL SLICE: rounded up to whole waves (so a wave's packs had one owner), rank 7
  // owned nothing at 3584 and 7168 and the others reduce-scattered 8/7 of the row.
  const int slice        = per_rank;
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
  const auto prefix     = local_ptr(prefix_ptr, prefix_stride_m, prefix_stride_n, rank);
  const auto norm_w     = local_ptr(norm_w_ptr, 0, norm_w_stride_n, rank);
  const auto qk_w       = local_ptr(qk_w_ptr, 0, qk_w_stride_n, rank);
  const auto out_norm_w = local_ptr(out_norm_w_ptr, 0, out_norm_w_stride_n, rank);
  const auto out        = local_ptr(out_ptr, out_stride_m, out_stride_n, rank);

  // A WAVE'S PACKS MAY HAVE TWO OWNERS, so the owner's scratch is picked in each lane.
  const auto scratches  = rank_scratches<DTYPE, WORLD>(peer_scratch);
  const auto scratch_of = [&](int r) {
    DTYPE* at = scratches[0];
#pragma unroll
    for (int k = 1; k < WORLD; ++k) at = r == k ? scratches[k] : at;
    return at;
  };
  const int end = (col0 + own_packs) * NL;  // this rank's columns end, in elements
  // One tile's reduce-scatter: this rank's columns of its rows summed over the ranks into this
  // rank's scratch, a chunk at a time.
  const auto reduce_tile = [&](int offs_m) {
    for (int row = offs_m; row < min(offs_m + TILE_M, m); ++row)
      for (int c = col0 * NL; c < end; c += Ranks::kTileN) {
        Ranks chunk{WORLD, end, 0, c};
        reduce_scatter(chunk, [&](int w) { return input(w) + row * inp_stride_m; },
                       own_scratch + row * scratch_stride_m);
      }
  };
  // One tile's packs from the ranks that own them, then AttnRes on it.
  const auto attn_res_tile = [&](int offs_m) {
    Rows sum{m, n, offs_m, 0};
    sliced_load<WORLD>(sum, scratch_of, scratch_stride_m, slice * NL);
    block_attn_res_tile<HAS_PREFIX, TILE_K>(sum, prefix, blocks_ptr, blocks_stride_m, blocks_stride_r,
                                         write_idx, norm_w, qk_w, out_norm_w, out, num_blocks,
                                         eps, out_eps, inv_hidden);
  };
  // Every block its own tiles: reduce-scatter them, meet its twins, AttnRes on them.
  (void)reduce_scatter_blocks;
  const int tiles = (m + TILE_M - 1) / TILE_M;
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) reduce_tile(t * TILE_M);
  block_stamp(2);
  barrier<Group::peers, Until::visible>(sync);
  block_stamp(3);
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) attn_res_tile(t * TILE_M);
  block_stamp(4);
  sync.finish();
}

}  // namespace hip_comms
