// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce: every rank sums every rank's input over the whole buffer.
// Modelled on aiter's `cross_device_reduce_1stage`: sync, read every peer, sum, sync. Two kernels:
// in place, every rank reads every peer's registered or captured input where it is; staged, for an
// eager input the peers cannot read.

#pragma once

#include "../common/interface.cuh"

namespace hip_custom_communicator {

// THE BUFFER AS [m, n] AT ITS STRIDES, a work item a TILE_M x TILE_N tile (a row of threads
// a tile row), the grid striding over the tiles row-major. Both tile sides are tuned (select.cuh).
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N, int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot(const DTYPE* const* __restrict__ input_ptrs,
                             int64_t input_stride_m,
                             int64_t input_stride_n,
                             int input_size_m,
                             int input_size_n,
                             DTYPE* __restrict__ output_ptr,
                             int64_t output_stride_m,
                             int64_t output_stride_n,
                             int output_size_m,
                             int output_size_n,
                             uint32_t* const* __restrict__ signals_ptrs,
                             int64_t signals_stride_parity,
                             int64_t signals_stride_block,
                             int64_t signals_stride_writer,
                             int signals_size_parity,
                             int signals_size_block,
                             int signals_size_writer,
                             uint32_t* __restrict__ status_ptr,
                             int64_t status_stride_field,
                             int status_size_field,
                             int rank,
                             uint64_t timeout_ticks) {
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  Sync<WORLD> sync{signals_ptrs, signals_stride_parity, signals_stride_block,
                   signals_stride_writer, status_ptr, status_stride_field, rank,
                   timeout_ticks};
  using Block       = Tile<DTYPE, TILE_M, TILE_N, TILE_M, THREADS_PER_BLOCK / TILE_M, THREADS_PER_BLOCK>;
  const int tiles_n = (input_size_n + TILE_N - 1) / TILE_N;
  const int tiles_m = (input_size_m + TILE_M - 1) / TILE_M;

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto input = rank_ptrs<const DTYPE, WORLD>(input_ptrs, input_stride_m, input_stride_n);
  const auto output    = local_ptr(output_ptr, output_stride_m, output_stride_n, rank);
  block_stamp(0);
  barrier<Until::launched>(sync);
  block_stamp(1);

  // 2. Read every rank's input, in rank order, and sum.
  block_for_each_tile(tiles_m, tiles_n, [&](int row, int col) {
    const Block at{input_size_m, input_size_n, row * TILE_M, col * TILE_N};
    Block peers[WORLD];
#pragma unroll
    for (int r = 0; r < WORLD; ++r) peers[r] = at;
    tile_load(peers, input);
    tile_store(peers_reduce(peers), output);
  });
  block_stamp(2);

  // 3. No rank may overwrite its input until every peer has read it.
  barrier<Until::read>(sync);
  block_stamp(5);
#else
  __builtin_trap();
#endif
}

// STAGED: each rank copies its input into its staging `staging_size_m` rows at a time and every rank
// reads every peer's staging, so any size runs in one launch. The staging is ours, a band dense at
// its strides. EACH BLOCK STAGES THE TILES IT READS: what the same block on a peer copied is what a
// peers barrier makes visible.
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N, int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot_staged(const DTYPE* __restrict__ input_ptr,
                                    int64_t input_stride_m,
                                    int64_t input_stride_n,
                                    int input_size_m,
                                    int input_size_n,
                                    DTYPE* __restrict__ output_ptr,
                                    int64_t output_stride_m,
                                    int64_t output_stride_n,
                                    int output_size_m,
                                    int output_size_n,
                                    DTYPE* const* __restrict__ staging_ptrs,
                                    int64_t staging_stride_m,
                                    int64_t staging_stride_n,
                                    int staging_size_m,
                                    int staging_size_n,
                                    uint32_t* const* __restrict__ signals_ptrs,
                                    int64_t signals_stride_parity,
                                    int64_t signals_stride_block,
                                    int64_t signals_stride_writer,
                                    int signals_size_parity,
                                    int signals_size_block,
                                    int signals_size_writer,
                                    uint32_t* __restrict__ status_ptr,
                                    int64_t status_stride_field,
                                    int status_size_field,
                                    int rank,
                                    uint64_t timeout_ticks) {
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  Sync<WORLD> sync{signals_ptrs, signals_stride_parity, signals_stride_block,
                   signals_stride_writer, status_ptr, status_stride_field, rank,
                   timeout_ticks};
  using Block       = Tile<DTYPE, TILE_M, TILE_N, TILE_M, THREADS_PER_BLOCK / TILE_M, THREADS_PER_BLOCK>;
  const int tiles_n = (input_size_n + TILE_N - 1) / TILE_N;
  const auto staging      = rank_ptrs<DTYPE, WORLD>(staging_ptrs, staging_stride_m,
  staging_stride_n);
  const auto own_staging = rank_ptr<DTYPE, WORLD>(staging_ptrs, rank, staging_stride_m,
  staging_stride_n);

  for (int r0 = 0; r0 < input_size_m; r0 += staging_size_m) {
    const int band  = min(staging_size_m, input_size_m - r0);  // this pass's rows
    const int tiles = (band + TILE_M - 1) / TILE_M * tiles_n;
    // This pass's rows of the input and the output, as Ptrs from its first row.
    const auto input = local_ptr(input_ptr + r0 * input_stride_m, input_stride_m, input_stride_n, rank);
    const auto output = local_ptr(output_ptr + r0 * output_stride_m, output_stride_m, output_stride_n, rank);
    block_stamp(0);
    // 1. This rank's band into its staging, then visible to the peers (each has staged its own).
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      Block mine{band, input_size_n, t / tiles_n * TILE_M, t % tiles_n * TILE_N};
      tile_load(mine, input);
      tile_store(mine, own_staging);
    }
    barrier<Until::visible>(sync);
    block_stamp(1);
    // 2. Read every rank's staged band, in rank order, and sum.
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      const Block at{band, input_size_n, t / tiles_n * TILE_M, t % tiles_n * TILE_N};
      Block peers[WORLD];
#pragma unroll
      for (int r = 0; r < WORLD; ++r) peers[r] = at;
      tile_load(peers, staging);
      tile_store(peers_reduce(peers), output);
    }
    block_stamp(2);
    // 3. No rank may stage its next band until every peer has read this one.
    barrier<Until::read>(sync);
    block_stamp(5);
  }
#else
  __builtin_trap();
#endif
}

}  // namespace hip_custom_communicator
