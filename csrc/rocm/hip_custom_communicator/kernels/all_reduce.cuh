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
        uint32_t* __restrict__ signals_ptr, int64_t signals_stride_rank,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks) {
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  // A ROW OF THREADS A TILE ROW: THREADS_M x THREADS_N over the tile, K packs a
  // thread.
  constexpr int THREADS_M = TILE_M;
  constexpr int THREADS_N = THREADS_PER_BLOCK / TILE_M;
  constexpr int P = traits<DTYPE>::N;
  constexpr int ROWS = TILE_M / THREADS_M;
  constexpr int K = TILE_N / (P * THREADS_N);
  const int tiles_n = (input_size_n + TILE_N - 1) / TILE_N;
  const int tiles_m = (input_size_m + TILE_M - 1) / TILE_M;

  // 1. Every rank's input pointer, then wait until every peer has launched, so
  // its input is ready.
  const DTYPE* input[WORLD];
  peers_ptrs(input_ptrs, input);
  barrier<Until::launched, 0, WORLD>(
      signals_ptr, signals_stride_rank, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);

  // 2. Read every rank's input, in rank order, and sum.
  block_for_each_tile(tiles_m, tiles_n, [&](int row, int col) {
    const int offs_m = row * TILE_M;
    const int offs_n = col * TILE_N;
    DTYPE peers[WORLD][ROWS][K][P];
    peers_load<THREADS_M, THREADS_N>(input, input_stride_m, input_stride_n,
                                     input_size_m, input_size_n, peers, offs_m,
                                     offs_n);
    float sum[ROWS][K][P];
    peers_sum(peers, sum);
    DTYPE rounded[ROWS][K][P];
    convert(sum, rounded);
    store<THREADS_M, THREADS_N, THREADS_PER_BLOCK>(rounded, output_ptr, output_stride_m,
                                output_stride_n, input_size_m, input_size_n,
                                offs_m, offs_n);
  });

  // 3. No rank may overwrite its input until every peer has read it.
  barrier<Until::read, 1, WORLD>(
      signals_ptr, signals_stride_rank, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);
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
        DTYPE* __restrict__ staging_ptr, int64_t staging_stride_rank, int64_t staging_stride_m,
        int64_t staging_stride_n, int staging_size_m, int staging_size_n,
        uint32_t* __restrict__ signals_ptr, int64_t signals_stride_rank,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks) {
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  constexpr int THREADS_M = TILE_M;
  constexpr int THREADS_N = THREADS_PER_BLOCK / TILE_M;
  constexpr int P = traits<DTYPE>::N;
  constexpr int ROWS = TILE_M / THREADS_M;
  constexpr int K = TILE_N / (P * THREADS_N);
  const int tiles_n = (input_size_n + TILE_N - 1) / TILE_N;
  DTYPE* const own_staging = peer_ptr(staging_ptr, staging_stride_rank, rank);

  for (int r0 = 0; r0 < input_size_m; r0 += staging_size_m) {
    const int band =
        min(staging_size_m, input_size_m - r0);  // this pass's rows
    const int tiles = (band + TILE_M - 1) / TILE_M * tiles_n;
    // This pass's rows of the input and the output, from its first row.
    const DTYPE* const band_input = input_ptr + r0 * input_stride_m;
    DTYPE* const band_output = output_ptr + r0 * output_stride_m;
    // 1. This rank's band into its staging, then visible to the peers (each
    // has staged its own).
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      DTYPE mine[ROWS][K][P];
      load<THREADS_M, THREADS_N>(band_input, input_stride_m,
                                 input_stride_n, band, input_size_n, mine,
                                 t / tiles_n * TILE_M, t % tiles_n * TILE_N);
      store<THREADS_M, THREADS_N, THREADS_PER_BLOCK>(mine, own_staging, staging_stride_m,
                                  staging_stride_n, band, input_size_n,
                                  t / tiles_n * TILE_M, t % tiles_n * TILE_N);
    }
    barrier<Until::visible, 0, WORLD>(
        signals_ptr, signals_stride_rank, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks);
    // 2. Read every rank's staged band, in rank order, and sum.
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      DTYPE peers[WORLD][ROWS][K][P];
      peers_load<THREADS_M, THREADS_N>(staging_ptr, staging_stride_rank,
                                       staging_stride_m, staging_stride_n, band,
                                       input_size_n,
                                       peers, t / tiles_n * TILE_M,
                                       t % tiles_n * TILE_N);
      float sum[ROWS][K][P];
      peers_sum(peers, sum);
      DTYPE rounded[ROWS][K][P];
      convert(sum, rounded);
      store<THREADS_M, THREADS_N, THREADS_PER_BLOCK>(rounded, band_output,
                                  output_stride_m, output_stride_n, band,
                                  input_size_n, t / tiles_n * TILE_M,
                                  t % tiles_n * TILE_N);
    }
    // 3. No rank may stage its next band until every peer has read this one.
    barrier<Until::read, 1, WORLD>(
        signals_ptr, signals_stride_rank, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks);
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

// THE PROBLEM, [R, M, N]: R = WORLD ranks' inputs, each an [input_size_m,
// input_size_n] buffer at its strides, summed elementwise. Reduce-scatter (each
// rank sums its share over the ranks into its scratch), sync, all-gather (every
// rank's summed share out to the output). The kernel states the problem;
// common.cuh's flat walk cuts it as [R, 1, M*N], each rank's share one
// contiguous range of the elements, for long reads from each peer (its comment
// says why and what it serves badly), a TILE_N chunk a step (TILE_M is the
// build's, 1: select's two_shot_lays_out). CONTIGUOUS: the input's and the
// output's rows are back to back, the instance the host picks from their
// strides. QUANTIZATION: what crosses the links (common.cuh's QUANTIZATION);
// with int4 the input's quantized form is staged in this rank's staging values
// and scales, and the sums' are the scratch values and scales, a group a thread
// (the INT4 flat walk's).
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU, bool CONTIGUOUS,
          int QUANTIZATION>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot(
        const DTYPE* const* __restrict__ input_ptrs, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        DTYPE* __restrict__ output_ptr, int64_t output_stride_m,
        int64_t output_stride_n, int output_size_m, int output_size_n,
        DTYPE* __restrict__ scratch_ptr, int64_t scratch_stride_rank, int64_t scratch_stride_m,
        int64_t scratch_stride_n, int scratch_size_m, int scratch_size_n,
        uint8_t* __restrict__ scratch_values_ptr, int64_t scratch_values_stride_rank,
        int64_t scratch_values_stride_m, int64_t scratch_values_stride_n,
        int scratch_values_size_m, int scratch_values_size_n,
        __fp16* __restrict__ scratch_scales_ptr, int64_t scratch_scales_stride_rank,
        int64_t scratch_scales_stride_m, int64_t scratch_scales_stride_n,
        int scratch_scales_size_m, int scratch_scales_size_n,
        uint8_t* __restrict__ staging_values_ptr, int64_t staging_values_stride_rank,
        int64_t staging_values_stride_m, int64_t staging_values_stride_n,
        int staging_values_size_m, int staging_values_size_n,
        __fp16* __restrict__ staging_scales_ptr, int64_t staging_scales_stride_rank,
        int64_t staging_scales_stride_m, int64_t staging_scales_stride_n,
        int staging_scales_size_m, int staging_scales_size_n,
        uint32_t* __restrict__ signals_ptr, int64_t signals_stride_rank,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks) {
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  // 1. This rank's input staged in QUANTIZATION's form (none: nothing, the
  // peers read it in place), then wait until every peer has launched, so its
  // input is ready.
  stage<WORLD, THREADS_PER_BLOCK, TILE_N, CONTIGUOUS, QUANTIZATION>(
      peer_ptr(input_ptrs, rank), input_stride_m, input_stride_n, input_size_m,
      input_size_n, peer_ptr(staging_values_ptr, staging_values_stride_rank, rank), staging_values_stride_m,
      staging_values_stride_n, staging_values_size_m, staging_values_size_n,
      peer_ptr(staging_scales_ptr, staging_scales_stride_rank, rank), staging_scales_stride_m,
      staging_scales_stride_n, staging_scales_size_m, staging_scales_size_n,
      rank);
  barrier<Until::launched, 0, WORLD>(
      signals_ptr, signals_stride_rank, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);

  // 2. Reduce-scatter: this rank's share of the inputs, summed over the ranks,
  // into its scratch.
  reduce_scatter<WORLD, THREADS_PER_BLOCK, TILE_N, CONTIGUOUS, QUANTIZATION>(
      input_ptrs, input_stride_m, input_stride_n, input_size_m, input_size_n,
      staging_values_ptr, staging_values_stride_rank, staging_values_stride_m, staging_values_stride_n,
      staging_values_size_m, staging_values_size_n, staging_scales_ptr, staging_scales_stride_rank,
      staging_scales_stride_m, staging_scales_stride_n, staging_scales_size_m,
      staging_scales_size_n, scratch_ptr, scratch_stride_rank, scratch_stride_m, scratch_stride_n,
      scratch_size_m, scratch_size_n, scratch_values_ptr, scratch_values_stride_rank,
      scratch_values_stride_m, scratch_values_stride_n, scratch_values_size_m,
      scratch_values_size_n, scratch_scales_ptr, scratch_scales_stride_rank, scratch_scales_stride_m,
      scratch_scales_stride_n, scratch_scales_size_m, scratch_scales_size_n,
      rank);

  // 3. Every rank's sums are visible to its peers.
  barrier<Until::visible, 1, WORLD>(
      signals_ptr, signals_stride_rank, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);

  // 4. All-gather: every rank's summed share out of its scratch, at its place
  // in the output. The
  //    next call's first sync keeps a rank from overwriting its scratch while
  //    it is read.
  all_gather<WORLD, THREADS_PER_BLOCK, TILE_N, CONTIGUOUS, QUANTIZATION>(
      scratch_ptr, scratch_stride_rank, scratch_stride_m, scratch_stride_n, scratch_size_m,
      scratch_size_n, scratch_values_ptr, scratch_values_stride_rank, scratch_values_stride_m,
      scratch_values_stride_n, scratch_values_size_m, scratch_values_size_n,
      scratch_scales_ptr, scratch_scales_stride_rank, scratch_scales_stride_m, scratch_scales_stride_n,
      scratch_scales_size_m, scratch_scales_size_n, output_ptr, output_stride_m,
      output_stride_n, input_size_m, input_size_n, rank);
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
        DTYPE* __restrict__ scratch_ptr, int64_t scratch_stride_rank, int64_t scratch_stride_m,
        int64_t scratch_stride_n, int scratch_size_m, int scratch_size_n,
        DTYPE* __restrict__ staging_ptr, int64_t staging_stride_rank, int64_t staging_stride_m,
        int64_t staging_stride_n, int staging_size_m, int staging_size_n,
        uint32_t* __restrict__ signals_ptr, int64_t signals_stride_rank,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks) {
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  for (int r0 = 0; r0 < input_size_m; r0 += staging_size_m) {
    const int band =
        min(staging_size_m, input_size_m - r0);  // this pass's rows
    // This pass's rows of the input and of the output.
    const DTYPE* const band_input = input_ptr + r0 * input_stride_m;
    DTYPE* const band_output = output_ptr + r0 * output_stride_m;
    // 1. This band, every rank's share of it, into this rank's staging (the
    // chunks each rank's same
    //    block reads), then visible (and, past the first pass, every peer has
    //    read this rank's scratch).
    stage<WORLD, THREADS_PER_BLOCK, TILE_N, CONTIGUOUS>(
        band_input, input_stride_m, input_stride_n, band,
        input_size_n, peer_ptr(staging_ptr, staging_stride_rank, rank), staging_stride_m, staging_stride_n,
        band, input_size_n, rank);
    barrier<Until::visible, 0, WORLD>(
        signals_ptr, signals_stride_rank, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks);

    // 2. Reduce-scatter: this rank's share of the band from every rank's
    // staging, into its scratch.
    reduce_scatter<WORLD, THREADS_PER_BLOCK, TILE_N, true>(  // ours, rows back to back
        staging_ptr, staging_stride_rank, staging_stride_m, staging_stride_n, band, input_size_n,
        scratch_ptr, scratch_stride_rank, scratch_stride_m, scratch_stride_n, scratch_size_m,
        scratch_size_n, rank);

    // 3. Every rank's sums are visible to its peers, and every peer has read
    // this rank's staged
    //    band, so the next one may overwrite it.
    barrier<Until::visible, 1, WORLD>(
        signals_ptr, signals_stride_rank, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks);

    // 4. All-gather: every rank's summed share out of its scratch, at its place
    // in the output.
    all_gather<WORLD, THREADS_PER_BLOCK, TILE_N, CONTIGUOUS>(
        scratch_ptr, scratch_stride_rank, scratch_stride_m, scratch_stride_n, scratch_size_m,
        scratch_size_n, band_output, output_stride_m,
        output_stride_n, band, input_size_n, rank);
  }
#else
  __builtin_trap();
#endif
}

}  // namespace hip_custom_communicator
