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

  const auto scratch_of = [&](int r) { return rank_scratch<DTYPE, WORLD>(peer_scratch, r); };
  const int end = (col0 + own_packs) * NL;  // this rank's columns end, in elements
  (void)reduce_scatter_blocks;
  if constexpr (THREADS_PER_BLOCK < 512) {
    // 2. Reduce-scatter: this rank's columns of this block's tiles' rows (every gridDim.x-th
    //    tile, AttnRes's below) summed over the ranks into this rank's scratch, a chunk at a time.
    for (int offs_m = blockIdx.x * TILE_M; offs_m < rows; offs_m += gridDim.x * TILE_M)
      for (int row = offs_m; row < min(offs_m + TILE_M, rows); ++row)
        for (int c = col0 * NL; c < end; c += Ranks::kTileN) {
          Ranks chunk{WORLD, end, 0, c};
          reduce_scatter(chunk, [&](int w) { return input(w) + int64_t{row} * cols; },
                         own_scratch + int64_t{row} * cols);
        }
    block_stamp(2);
    // 3. This block's tiles' columns are in every rank's scratch: its twin on every rank wrote them.
    barrier<Group::peers, Until::visible>(sync);
    block_stamp(3);
    // 4. Each tile's packs from the ranks that own them, then AttnRes on it.
    for (int offs_m = blockIdx.x * TILE_M; offs_m < rows; offs_m += gridDim.x * TILE_M) {
      Rows sum{rows, cols, offs_m, 0};
      sliced_load<WORLD>(sum, scratch_of, cols, slice * NL);
      block_attn_res_tile<HAS_PREFIX, TILE_K>(sum, prefix, blocks, block_stride_m, block_stride_r,
                                           write_idx, norm_w, qk_w, out_norm_w, out, num_blocks,
                                           eps, out_eps, inv_hidden);
    }
  } else {
    // A PIPELINE OF A BLOCK'S TILES (TILE_M rows each), the links under AttnRes: the first half's
    // waves (consumers) run AttnRes on tile i while the second half's (producers) gather tile i + 1
    // from its owners and reduce-scatter tile i + 2, its peers barrier the producers' own (part_barrier). Run one after
    // the other, the reduce-scatter (link bound) and AttnRes (a row at a time) each took about half
    // the kernel (thread traces 2026-10-04T22-41-39Z, 2026-10-05T00-11-09Z). The block has one
    // hardware barrier, so the producers pass AttnRes's barriers one for one, their waits placed
    // after all but the last.
    constexpr int HALF = THREADS_PER_BLOCK / 2;
    using Half = Tile<DTYPE, TILE_M, TILE_N, 1, HALF, HALF>;
    // A tile's slice, a pack a row a producer thread or more (fewer ranks, wider slices).
    constexpr int OWN_N = TILE_N / WORLD > HALF * NL ? TILE_N / WORLD : HALF * NL;
    using Own = Tile<DTYPE, TILE_M, OWN_N, 1, HALF, HALF>;
    __shared__ LdsTile<Half> staged;
    __shared__ uint32_t met;
    uint32_t gen = 0;
    const bool consumer = __builtin_amdgcn_readfirstlane(threadIdx.x / kWaveSize) < HALF / kWaveSize;
    const int barriers = attn_res_tile_barriers<TILE_K>(num_blocks, out_norm_w != nullptr);
    const int grid = static_cast<int>(gridDim.x);
    const int first = static_cast<int>(blockIdx.x);
    const int tiles = (rows + TILE_M - 1) / TILE_M;
    const int n = first < tiles ? (tiles - 1 - first) / grid + 1 : 0;  // this block's tiles
    const auto row_of = [&](int i) { return (first + i * grid) * TILE_M; };  // tile i's first row
    const auto own_row = [&](int i) { return Own{rows, end, row_of(i), col0 * NL}; };
    // Tile i's slice summed over the ranks into this rank's scratch, then the peers' barrier.
    const auto reduce = [&](int i) {
      Own got[WORLD];
#pragma unroll
      for (int w = 0; w < WORLD; ++w) got[w] = own_row(i);
      peers_load(got, input, cols);
      tile_store(own_scratch, cols, peers_reduce(got));
      part_barrier<HALF, HALF>(sync, &met, gen);
    };
    if (threadIdx.x == HALF) met = 0;
    __syncthreads();
    // The pipeline filled: tiles 0 and 1 reduce-scattered, tile 0 staged.
    if (!consumer) {
      for (int i = 0; i < min(n, 2); ++i) reduce(i);
      if (n > 0) {
        Half g{rows, cols, row_of(0), 0};
        sliced_load<WORLD>(g, scratch_of, cols, slice * NL);
        lds_store(staged, g);
      }
    } else {
      for (int i = 0; i < min(n, 2); ++i) sync.skip();
    }
    __syncthreads();
    for (int i = 0; i < n; ++i) {
      Half sum{rows, cols, row_of(i), 0};
      if (consumer) lds_load(sum, staged);
      __syncthreads();  // the stage is free
      if (consumer) {
        block_attn_res_tile<HAS_PREFIX, TILE_K>(sum, prefix, blocks, block_stride_m,
                                             block_stride_r, write_idx, norm_w, qk_w, out_norm_w,
                                             out, num_blocks, eps, out_eps, inv_hidden);
        if (i + 2 < n) sync.skip();
      } else {
        // Every load issued first, waited for only after all but AttnRes's last barrier.
        Half g{rows, cols, row_of(min(i + 1, n - 1)), 0};
        if (i + 1 < n) sliced_load<WORLD>(g, scratch_of, cols, slice * NL);
        Own got[WORLD];
#pragma unroll
        for (int w = 0; w < WORLD; ++w) got[w] = own_row(min(i + 2, n - 1));
        if (i + 2 < n) peers_load(got, input, cols);
        for (int b = 1; b < barriers; ++b) __syncthreads();
        if (i + 2 < n) {
          tile_store(own_scratch, cols, peers_reduce(got));
          part_barrier<HALF, HALF>(sync, &met, gen);
        }
        if (i + 1 < n) lds_store(staged, g);
        if (barriers > 0) __syncthreads();
      }
      __syncthreads();  // the next row staged
    }
  }
  block_stamp(4);
  sync.finish();
}

}  // namespace hip_comms
