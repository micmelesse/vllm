// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE PLAIN ALL-REDUCE: its kernels, one section each (one-shot and two-shot
// pull, each with its staged build).

#pragma once

#include "common.cuh"

namespace hip_custom_communicator {

// =============================================================================
// all_reduce_pull_one_shot
//
// One-shot pull all-reduce: every rank sums every rank's input over the whole
// buffer. Modelled on aiter's `cross_device_reduce_1stage`: sync, read every
// peer, sum, sync. Two kernels: in place, every rank reads every peer's
// registered or captured input where it is; staged, for an eager input the
// peers cannot read.
// =============================================================================

// THE BUFFER AS [m, n] AT ITS STRIDES, a work item a TILE_M x TILE_N tile (a
// row of threads a tile row), the grid striding over the tiles row-major. Both
// tile sides are tuned (select.cuh).
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot(
        const DTYPE* const* __restrict__ input_ptrs, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        DTYPE* __restrict__ output_ptr, int64_t output_stride_m,
        int64_t output_stride_n, int output_size_m, int output_size_n,
        uint32_t* const* __restrict__ signals_ptrs,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks) {
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  Sync<WORLD> sync{signals_ptrs,
                   signals_stride_parity,
                   signals_stride_block,
                   signals_stride_writer,
                   status_ptr,
                   status_stride_field,
                   rank,
                   timeout_ticks};
  using Block = Tile<DTYPE, TILE_M, TILE_N, TILE_M, THREADS_PER_BLOCK / TILE_M,
                     THREADS_PER_BLOCK>;
  const int tiles_n = (input_size_n + TILE_N - 1) / TILE_N;
  const int tiles_m = (input_size_m + TILE_M - 1) / TILE_M;

  // 1. Every rank's buffers, then wait until every peer has launched, so its
  // input is ready.
  const auto input =
      rank_ptrs<const DTYPE, WORLD>(input_ptrs, input_stride_m, input_stride_n);
  const auto output =
      local_ptr(output_ptr, output_stride_m, output_stride_n, rank);
  barrier<Until::launched>(sync);

  // 2. Read every rank's input, in rank order, and sum.
  block_for_each_tile(tiles_m, tiles_n, [&](int row, int col) {
    const Block at{input_size_m, input_size_n, row * TILE_M, col * TILE_N};
    Block peers[WORLD];
  #pragma unroll
    for (int r = 0; r < WORLD; ++r) peers[r] = at;
    tile_load(peers, input);
    tile_store(peers_reduce(peers), output);
  });

  // 3. No rank may overwrite its input until every peer has read it.
  barrier<Until::read>(sync);
#else
  __builtin_trap();
#endif
}

// STAGED: each rank copies its input into its staging `staging_size_m` rows at
// a time and every rank reads every peer's staging, so any size runs in one
// launch. The staging is ours, a band dense at its strides. EACH BLOCK STAGES
// THE TILES IT READS: what the same block on a peer copied is what a peers
// barrier makes visible.
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot_staged(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        DTYPE* __restrict__ output_ptr, int64_t output_stride_m,
        int64_t output_stride_n, int output_size_m, int output_size_n,
        DTYPE* const* __restrict__ staging_ptrs, int64_t staging_stride_m,
        int64_t staging_stride_n, int staging_size_m, int staging_size_n,
        uint32_t* const* __restrict__ signals_ptrs,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks) {
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  Sync<WORLD> sync{signals_ptrs,
                   signals_stride_parity,
                   signals_stride_block,
                   signals_stride_writer,
                   status_ptr,
                   status_stride_field,
                   rank,
                   timeout_ticks};
  using Block = Tile<DTYPE, TILE_M, TILE_N, TILE_M, THREADS_PER_BLOCK / TILE_M,
                     THREADS_PER_BLOCK>;
  const int tiles_n = (input_size_n + TILE_N - 1) / TILE_N;
  const auto staging =
      rank_ptrs<DTYPE, WORLD>(staging_ptrs, staging_stride_m, staging_stride_n);
  const auto own_staging = rank_ptr<DTYPE, WORLD>(
      staging_ptrs, rank, staging_stride_m, staging_stride_n);

  for (int r0 = 0; r0 < input_size_m; r0 += staging_size_m) {
    const int band =
        min(staging_size_m, input_size_m - r0);  // this pass's rows
    const int tiles = (band + TILE_M - 1) / TILE_M * tiles_n;
    // This pass's rows of the input and the output, as Ptrs from its first row.
    const auto input = local_ptr(input_ptr + r0 * input_stride_m,
                                 input_stride_m, input_stride_n, rank);
    const auto output = local_ptr(output_ptr + r0 * output_stride_m,
                                  output_stride_m, output_stride_n, rank);
    // 1. This rank's band into its staging, then visible to the peers (each has
    // staged its own).
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      Block mine{band, input_size_n, t / tiles_n * TILE_M,
                 t % tiles_n * TILE_N};
      tile_load(mine, input);
      tile_store(mine, own_staging);
    }
    barrier<Until::visible>(sync);
    // 2. Read every rank's staged band, in rank order, and sum.
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      const Block at{band, input_size_n, t / tiles_n * TILE_M,
                     t % tiles_n * TILE_N};
      Block peers[WORLD];
  #pragma unroll
      for (int r = 0; r < WORLD; ++r) peers[r] = at;
      tile_load(peers, staging);
      tile_store(peers_reduce(peers), output);
    }
    // 3. No rank may stage its next band until every peer has read this one.
    barrier<Until::read>(sync);
  }
#else
  __builtin_trap();
#endif
}

// =============================================================================
// all_reduce_pull_two_shot
//
// Two-shot pull all-reduce: reduce-scatter, then all-gather. aiter's
// `cross_device_reduce_2stage` as written: sync, sum this rank's slice from
// every peer into its scratch, sync, copy every rank's summed slice out of its
// scratch. Two kernels: in place, the peers' registered or captured inputs read
// where they are; staged, for an eager input the peers cannot read.
// =============================================================================

// THE PROBLEM, [R, M, N]: R = WORLD ranks' inputs, each an [inp_size_m,
// inp_size_n] buffer at its strides, summed elementwise. Reduce-scatter (each
// rank sums its share over the ranks into its scratch), sync, all-gather (every
// rank's summed share out to the output). The kernel states the problem;
// common.cuh's COLLECTIVES walk it as [R, 1, M*N], each rank's share one
// contiguous range of the elements, for long reads from each peer (its comment
// says why and what it serves badly). A rank's copy of a chunk is a wave (LANES
// threads); the Chunk tile is that [R, 1, TILE_N] chunk. CONTIGUOUS: the
// input's and the output's rows are back to back, the instance the host picks
// from their strides.
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU, bool CONTIGUOUS>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot(
        const DTYPE* const* __restrict__ input_ptrs, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        DTYPE* __restrict__ output_ptr, int64_t output_stride_m,
        int64_t output_stride_n, int output_size_m, int output_size_n,
        DTYPE* const* __restrict__ scratch_ptrs, int64_t scratch_stride_m,
        int64_t scratch_stride_n, int scratch_size_m, int scratch_size_n,
        uint32_t* const* __restrict__ signals_ptrs,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks) {
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  Sync<WORLD> sync{signals_ptrs,
                   signals_stride_parity,
                   signals_stride_block,
                   signals_stride_writer,
                   status_ptr,
                   status_stride_field,
                   rank,
                   timeout_ticks};
  constexpr int LANES = THREADS_PER_BLOCK / WORLD;  // a rank's copy's threads
  static_assert(LANES * WORLD == THREADS_PER_BLOCK,
                "a block is a group of threads a rank");
  // THE FLAT WALK'S CHUNK, one row of the flat range by nature (TILE_M is the
  // build's, 1: select's two_shot_lays_out): R ranks' copies of TILE_N
  // elements.
  using Chunk =
      Tile<DTYPE, 1, TILE_N, 1, LANES, THREADS_PER_BLOCK, DTYPE, WORLD, WORLD>;

  // 1. Every rank's buffers, then wait until every peer has launched, so its
  // input is ready.
  const auto input = [&](int r) {
    return rank_ptr<const DTYPE, WORLD>(input_ptrs, r, input_stride_m,
                                        input_stride_n);
  };
  const auto scratch = [&](int r) {
    return rank_ptr<DTYPE, WORLD>(scratch_ptrs, r, scratch_stride_m,
                                  scratch_stride_n);
  };
  const auto output =
      local_ptr(output_ptr, output_stride_m, output_stride_n, rank);
  barrier<Until::launched>(sync);

  // 2. Reduce-scatter: this rank's share of the inputs, summed over the ranks,
  // into its scratch.
  reduce_scatter<Chunk, CONTIGUOUS>(rank, input, input_size_m, input_size_n,
                                    scratch(rank));

  // 3. Every rank's sums are visible to its peers.
  barrier<Until::visible>(sync);

  // 4. All-gather: every rank's summed share out of its scratch, at its place
  // in the output. The
  //    next call's first sync keeps a rank from overwriting its scratch while
  //    it is read.
  all_gather<Chunk, CONTIGUOUS>(rank, scratch, output, input_size_m,
                                input_size_n);
#else
  __builtin_trap();
#endif
}

// STAGED: each rank copies its input into its staging a band of
// `staging_size_m` rows at a time and each pass reads the peers' staging, so
// any size runs in one launch. A band is the problem's rows [r0, r0 + band) at
// their strides, the staging ([band, n], ours) and the scratch (a share of it)
// at most what each holds (the host's). EACH BLOCK STAGES, OF EVERY RANK'S
// SHARE, THE CHUNKS THAT RANK'S SAME BLOCK READS, so a peers barrier makes them
// visible.
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU, bool CONTIGUOUS>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot_staged(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        DTYPE* __restrict__ output_ptr, int64_t output_stride_m,
        int64_t output_stride_n, int output_size_m, int output_size_n,
        DTYPE* const* __restrict__ scratch_ptrs, int64_t scratch_stride_m,
        int64_t scratch_stride_n, int scratch_size_m, int scratch_size_n,
        DTYPE* const* __restrict__ staging_ptrs, int64_t staging_stride_m,
        int64_t staging_stride_n, int staging_size_m, int staging_size_n,
        uint32_t* const* __restrict__ signals_ptrs,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks) {
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  Sync<WORLD> sync{signals_ptrs,
                   signals_stride_parity,
                   signals_stride_block,
                   signals_stride_writer,
                   status_ptr,
                   status_stride_field,
                   rank,
                   timeout_ticks};
  constexpr int LANES = THREADS_PER_BLOCK / WORLD;
  static_assert(LANES * WORLD == THREADS_PER_BLOCK,
                "a block is a group of threads a rank");
  // THE FLAT WALK'S CHUNK, one row of the flat range by nature (TILE_M is the
  // build's, 1: select's two_shot_lays_out): R ranks' copies of TILE_N
  // elements.
  using Chunk =
      Tile<DTYPE, 1, TILE_N, 1, LANES, THREADS_PER_BLOCK, DTYPE, WORLD, WORLD>;
  const auto staging = [&](int r) {
    return rank_ptr<DTYPE, WORLD>(staging_ptrs, r, staging_stride_m,
                                  staging_stride_n);
  };
  const auto scratch = [&](int r) {
    return rank_ptr<DTYPE, WORLD>(scratch_ptrs, r, scratch_stride_m,
                                  scratch_stride_n);
  };

  for (int r0 = 0; r0 < input_size_m; r0 += staging_size_m) {
    const int band =
        min(staging_size_m, input_size_m - r0);  // this pass's rows
    // This pass's rows of the input and of the output.
    const auto input = local_ptr(input_ptr + r0 * input_stride_m,
                                 input_stride_m, input_stride_n, rank);
    const auto output = local_ptr(output_ptr + r0 * output_stride_m,
                                  output_stride_m, output_stride_n, rank);
    // 1. This band, every rank's share of it, into this rank's staging (the
    // chunks each rank's same
    //    block reads), then visible (and, past the first pass, every peer has
    //    read this rank's scratch).
    stage<Chunk, CONTIGUOUS>(rank, input, staging(rank), band, input_size_n);
    barrier<Until::visible>(sync);

    // 2. Reduce-scatter: this rank's share of the band from every rank's
    // staging, into its scratch.
    reduce_scatter<Chunk, true>(rank, staging, band, input_size_n,
                                scratch(rank));  // ours, rows back to back

    // 3. Every rank's sums are visible to its peers, and every peer has read
    // this rank's staged
    //    band, so the next one may overwrite it.
    barrier<Until::visible>(sync);

    // 4. All-gather: every rank's summed share out of its scratch, at its place
    // in the output.
    all_gather<Chunk, CONTIGUOUS>(rank, scratch, output, band, input_size_n);
  }
#else
  __builtin_trap();
#endif
}

}  // namespace hip_custom_communicator
