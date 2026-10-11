// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce: every rank sums every rank's input over the whole buffer.
// Modelled on aiter's `cross_device_reduce_1stage`: sync, read every peer, sum, sync. Two kernels:
// in place, every rank reads every peer's registered or captured input where it is; staged, for an
// eager input the peers cannot read.

#pragma once

#include "../common/interface.cuh"

namespace hip_comms {

// THE BUFFER AS [m, n] AT ITS STRIDES, a work item a TILE_M x TILE_N tile (a row of threads
// a tile row), the grid striding over the tiles row-major. Both tile sides are tuned (select.cuh).
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N, int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot(const DTYPE* const* __restrict__ inp_ptrs, int64_t inp_stride_m,
                             int64_t inp_stride_n, Signal* const* __restrict__ signal_ptrs, Signal* self_signal_ptr,
                             int rank, uint64_t timeout_ticks, DTYPE* __restrict__ out_ptr,
                             int64_t out_stride_m, int64_t out_stride_n, int m, int n) {
  Sync<WORLD> sync{signal_ptrs, self_signal_ptr, rank, timeout_ticks};
  using Block       = Tile<DTYPE, TILE_M, TILE_N, TILE_M, THREADS_PER_BLOCK / TILE_M, THREADS_PER_BLOCK>;
  const int tiles_n = (n + TILE_N - 1) / TILE_N;
  const int tiles   = (m + TILE_M - 1) / TILE_M * tiles_n;

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto inp = rank_ptrs<const DTYPE, WORLD>(inp_ptrs, inp_stride_m, inp_stride_n);
  const auto out    = local_ptr(out_ptr, out_stride_m, out_stride_n, rank);
  block_stamp(0);
  barrier<Group::peers, Until::launched>(sync);
  block_stamp(1);

  // 2. Read every rank's input, in rank order, and sum.
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    const Block at{m, n, t / tiles_n * TILE_M, t % tiles_n * TILE_N};
    Block peers[WORLD];
#pragma unroll
    for (int r = 0; r < WORLD; ++r) peers[r] = at;
    tile_load(peers, inp);
    tile_store(peers_reduce(peers), out);
  }
  block_stamp(2);

  // 3. No rank may overwrite its input until every peer has read it.
  barrier<Group::peers, Until::read>(sync);
  block_stamp(5);
  sync.finish();
}

// STAGED: each rank copies its input into its staging `band_m` rows at a time and every rank
// reads every peer's staging, so any size runs in one launch. The staging is ours, a band dense at
// its strides. EACH BLOCK STAGES THE TILES IT READS: what the same block on a peer copied is what a
// peers barrier makes visible.
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N, int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot_staged(DTYPE* const* __restrict__ staging_ptrs, int64_t staging_stride_m,
                                    int64_t staging_stride_n, Signal* const* __restrict__ signal_ptrs,
                                    Signal* self_signal_ptr, int rank, uint64_t timeout_ticks,
                                    DTYPE* __restrict__ out_ptr, int64_t out_stride_m,
                                    int64_t out_stride_n, const DTYPE* __restrict__ inp_ptr,
                                    int64_t inp_stride_m, int64_t inp_stride_n, int m, int n,
                                    int band_m) {
  Sync<WORLD> sync{signal_ptrs, self_signal_ptr, rank, timeout_ticks};
  using Block       = Tile<DTYPE, TILE_M, TILE_N, TILE_M, THREADS_PER_BLOCK / TILE_M, THREADS_PER_BLOCK>;
  const int tiles_n = (n + TILE_N - 1) / TILE_N;
  const auto staging      = rank_ptrs<DTYPE, WORLD>(staging_ptrs, staging_stride_m, staging_stride_n);
  const auto own_staging = rank_ptr<DTYPE, WORLD>(staging_ptrs, rank, staging_stride_m, staging_stride_n);

  for (int r0 = 0; r0 < m; r0 += band_m) {
    const int band  = min(band_m, m - r0);  // this pass's rows
    const int tiles = (band + TILE_M - 1) / TILE_M * tiles_n;
    // This pass's rows of the input and the output, as Ptrs from its first row.
    const auto inp = local_ptr(inp_ptr + r0 * inp_stride_m, inp_stride_m, inp_stride_n, rank);
    const auto out = local_ptr(out_ptr + r0 * out_stride_m, out_stride_m, out_stride_n, rank);
    block_stamp(0);
    // 1. This rank's band into its staging, then visible to the peers (each has staged its own).
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      Block mine{band, n, t / tiles_n * TILE_M, t % tiles_n * TILE_N};
      tile_load(mine, inp);
      tile_store(mine, own_staging);
    }
    barrier<Group::peers, Until::visible>(sync);
    block_stamp(1);
    // 2. Read every rank's staged band, in rank order, and sum.
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      const Block at{band, n, t / tiles_n * TILE_M, t % tiles_n * TILE_N};
      Block peers[WORLD];
#pragma unroll
      for (int r = 0; r < WORLD; ++r) peers[r] = at;
      tile_load(peers, staging);
      tile_store(peers_reduce(peers), out);
    }
    block_stamp(2);
    // 3. No rank may stage its next band until every peer has read this one.
    barrier<Group::peers, Until::read>(sync);
    block_stamp(5);
  }
  sync.finish();
}

}  // namespace hip_comms
