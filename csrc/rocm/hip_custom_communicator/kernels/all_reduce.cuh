// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE PLAIN ALL-REDUCE: its kernels, one section each (one-shot and two-shot),
// each PUSHED: a rank reads only its own input (the caller's tensor, which no
// peer reads) and stores it, or each rank's share of it, into the peers'
// receive areas (the arena's `inputs` region, the INBOX: [writer slot, ...]),
// a band of rows a pass; after the ordered barrier each rank sums the WORLD
// slots of its inbox LOCALLY. Nothing is copied in first. The int4 two-shot
// pushes its shares quantized.
//
// THE INBOX IS REUSED, launch after launch and pass after pass, with no barrier
// of its own: a rank pushes into peer s's slot only after passing the barrier
// that follows peer s's local sum of it (the one-shot's closing `read`, the
// two-shots' second, `visible`), and a peer arrives there only once its
// block's sum is done (its loads returned: the sums stored after depend on
// them). Across launches every block of the previous launch passed it, so a
// different grid or another kernel before is covered too: every kernel's reads
// of the inputs region come before its last barrier.

#pragma once

#include "common.cuh"

namespace hip_custom_communicator {

// =============================================================================
// all_reduce_pull_one_shot
//
// One-shot all-reduce: every rank sums every rank's input over the whole
// buffer. PUSHED: each rank stores its own band into slot [rank] of every
// rank's inbox (itself too), sync (ordered), each rank sums the WORLD slots of
// its own inbox, sync. One kernel; no peer reads the caller's input.
// =============================================================================

// THE BUFFER AS [m, n] AT ITS STRIDES, a work item a TILE_M x TILE_N tile (a
// row of threads a tile row), the grid striding over the tiles row-major. Both
// tile sides are tuned (select.cuh). THE INBOX, the inputs region as [WORLD *
// band_rows, n]: writer w's band at rows [w * band_rows, (w + 1) * band_rows),
// band_rows = inputs_size_m / WORLD (the host's: what the region holds), so any
// size runs in one launch, a band a pass. EACH BLOCK PUSHES THE TILES ITS TWIN
// ON EVERY RANK SUMS: what the same block on a peer pushed is what a peers
// barrier makes visible.
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot_pushed(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        DTYPE* __restrict__ output_ptr, int64_t output_stride_m,
        int64_t output_stride_n, int output_size_m, int output_size_n,
        PeerBases bases, int64_t inputs_offset, int64_t inputs_stride_m,
        int64_t inputs_stride_n, int inputs_size_m, int inputs_size_n,
        int64_t signals_offset,
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
  const int band_rows = inputs_size_m / WORLD;  // a slot's rows
  const int64_t slot = int64_t{band_rows} * inputs_stride_m;  // a slot's elements
  // This rank's slot in every inbox, and this rank's own inbox.
  const int64_t mine_offset =
      inputs_offset + int64_t{rank} * slot * static_cast<int64_t>(sizeof(DTYPE));
  const DTYPE* const inbox = peer_ptr<DTYPE>(bases, rank, inputs_offset);

  for (int r0 = 0; r0 < input_size_m; r0 += band_rows) {
    const int band =
        min(band_rows, input_size_m - r0);  // this pass's rows
    const int tiles = (band + TILE_M - 1) / TILE_M * tiles_n;
    // This pass's rows of the input and the output, from its first row.
    const DTYPE* const band_input = input_ptr + r0 * input_stride_m;
    DTYPE* const band_output = output_ptr + r0 * output_stride_m;
    // 1. This rank's band, read here, into slot [rank] of every rank's inbox,
    // then visible to the peers (each has pushed its own into ours).
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      DTYPE mine[ROWS][K][P];
      load<THREADS_M, THREADS_N>(band_input, input_stride_m,
                                 input_stride_n, band, input_size_n, mine,
                                 t / tiles_n * TILE_M, t % tiles_n * TILE_N);
      peers_store<WORLD, THREADS_M, THREADS_N, THREADS_PER_BLOCK>(
          mine, bases, mine_offset, inputs_stride_m, inputs_stride_n, band,
          input_size_n, t / tiles_n * TILE_M, t % tiles_n * TILE_N);
    }
    barrier<Until::visible, 0, WORLD>(
        bases, signals_offset, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks);
    // 2. Every writer's slot of this rank's inbox, local, in rank order, summed.
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      DTYPE peers[WORLD][ROWS][K][P];
      slots_load<THREADS_M, THREADS_N>(inbox, slot, inputs_stride_m,
                                       inputs_stride_n, band, input_size_n,
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
    // 3. No rank may push its next band (this launch's or the next's) until
    // every peer has summed this one out of its inbox.
    barrier<Until::read, 1, WORLD>(
        bases, signals_offset, signals_stride_parity, signals_stride_block,
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
// Two-shot all-reduce: reduce-scatter, then all-gather, the sum kept in this
// rank's partials and pulled by the peers (its Direction, pull). THE
// REDUCE-SCATTER IS PUSHED: each rank reads its own input and stores every
// rank's share of it into that rank's inbox at slot [rank], sync (ordered),
// each rank sums the WORLD slots of its share locally into its partials, sync,
// every rank's summed share pulled out of its partials (aiter's
// `cross_device_reduce_2stage` second half). Two kernels: int4, the shares
// quantized by the kernel before they are pushed; exact (none), a band of rows
// a pass. No peer reads the caller's input.
// =============================================================================

// THE PROBLEM, [R, M, N]: R = WORLD ranks' inputs, each an [input_size_m,
// input_size_n] buffer at its strides, summed elementwise. Reduce-scatter (each
// rank sums its share over the ranks into its partials), sync, all-gather (every
// rank's summed share out to the output). The kernel states the problem;
// common.cuh's flat walk cuts it as [R, 1, M*N], each rank's share one
// contiguous range of the elements, for long runs (its comment says why and
// what it serves badly), a TILE_N chunk a step (TILE_M is the build's, 1:
// select's two_shot_lays_out). CONTIGUOUS: the input's and the output's rows
// are back to back, the instance the host picks from their strides.
// QUANTIZATION: what crosses the links (common.cuh's QUANTIZATION); with int4
// each share is quantized by this rank, from its own input, and pushed into the share's rank's inbox INT4 payload at this rank's slot
// (the payload as an [m, n] one, writer w's slot where share w's groups lie),
// and the sums' are the partials' INT4 payload, a group a thread (the INT4 flat
// walk's). INT4 ONLY: none is the exact build below.
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU, bool CONTIGUOUS,
          int QUANTIZATION>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        DTYPE* __restrict__ output_ptr, int64_t output_stride_m,
        int64_t output_stride_n, int output_size_m, int output_size_n,
        PeerBases bases, int64_t partials_offset, int64_t partials_stride_m,
        int64_t partials_stride_n, int partials_size_m, int partials_size_n,
        int64_t inputs_offset, int64_t inputs_stride_m,
        int64_t inputs_stride_n, int inputs_size_m, int inputs_size_n,
        int64_t signals_offset,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks) {
  // THE INT4 PAYLOADS, inside the regions (common.cuh's `int4_payload`):
  // the same offsets on every rank, from the call's sizes.
  const Int4Payload inputs_int4 = int4_payload(inputs_offset, inputs_size_m, inputs_size_n);
  const Int4Payload partials_int4 = int4_payload(partials_offset, partials_size_m, partials_size_n);
  static_assert(QUANTIZATION == Quantization::int4,
                "none is all_reduce_pull_two_shot_pushed");
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  // 1. Every rank's share of this rank's input, quantized here, into that
  // rank's inbox payload at slot [rank]; then wait until every peer has
  // launched and pushed its own into ours (ordered).
  push_shares<WORLD, THREADS_PER_BLOCK, TILE_N, CONTIGUOUS, QUANTIZATION>(
      input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
      bases, inputs_int4.values_offset, inputs_int4.values_stride_m,
      inputs_int4.values_stride_n, inputs_int4.values_size_m, inputs_int4.values_size_n,
      inputs_int4.scales_offset, inputs_int4.scales_stride_m,
      inputs_int4.scales_stride_n, inputs_int4.scales_size_m, inputs_int4.scales_size_n,
      rank);
  barrier<Until::launched, 0, WORLD>(
      bases, signals_offset, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);

  // 2. Reduce-scatter, local: this rank's share, every writer's slot of its
  // inbox payload summed, into its partials.
  reduce_slots<WORLD, THREADS_PER_BLOCK, TILE_N, CONTIGUOUS, QUANTIZATION, DTYPE>(
      bases, inputs_offset, input_stride_m, input_stride_n, input_size_m, input_size_n,
      inputs_int4.values_offset, inputs_int4.values_stride_m, inputs_int4.values_stride_n,
      inputs_int4.values_size_m, inputs_int4.values_size_n, inputs_int4.scales_offset,
      inputs_int4.scales_stride_m, inputs_int4.scales_stride_n, inputs_int4.scales_size_m,
      inputs_int4.scales_size_n, partials_offset, partials_stride_m, partials_stride_n,
      partials_size_m, partials_size_n, partials_int4.values_offset,
      partials_int4.values_stride_m, partials_int4.values_stride_n, partials_int4.values_size_m,
      partials_int4.values_size_n, partials_int4.scales_offset, partials_int4.scales_stride_m,
      partials_int4.scales_stride_n, partials_int4.scales_size_m, partials_int4.scales_size_n,
      rank);

  // 3. Every rank's sums are visible to its peers, and every peer has summed
  // its inbox (the next launch's push may overwrite it).
  barrier<Until::visible, 1, WORLD>(
      bases, signals_offset, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);

  // 4. All-gather: every rank's summed share out of its partials, at its place
  // in the output. The next call's first sync keeps a rank from overwriting
  // its partials while it is read (they are written only after it).
  all_gather<WORLD, THREADS_PER_BLOCK, TILE_N, CONTIGUOUS, QUANTIZATION>(
      bases, partials_offset, partials_stride_m, partials_stride_n, partials_size_m,
      partials_size_n, partials_int4.values_offset, partials_int4.values_stride_m,
      partials_int4.values_stride_n, partials_int4.values_size_m, partials_int4.values_size_n,
      partials_int4.scales_offset, partials_int4.scales_stride_m, partials_int4.scales_stride_n,
      partials_int4.scales_size_m, partials_int4.scales_size_n, output_ptr, output_stride_m,
      output_stride_n, input_size_m, input_size_n, rank);
#else
  __builtin_trap();
#endif
}

// EXACT, IN PASSES: a band of `partials_size_m` rows a pass (the host's: what
// the inbox and the partials hold), so any size runs in one launch. A band is
// the problem's rows [r0, r0 + band) at their strides. THE INBOX, the inputs
// region as [WORLD, slot]: writer w's slot at row w, inputs_stride_m elements
// a slot, holding writer w's copy of this rank's share of the band from the
// slot's first element. EACH BLOCK PUSHES, OF EVERY RANK'S SHARE, THE CHUNKS
// THAT RANK'S SAME BLOCK SUMS, so a peers barrier makes them visible.
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU, bool CONTIGUOUS>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot_pushed(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        DTYPE* __restrict__ output_ptr, int64_t output_stride_m,
        int64_t output_stride_n, int output_size_m, int output_size_n,
        PeerBases bases, int64_t partials_offset, int64_t partials_stride_m,
        int64_t partials_stride_n, int partials_size_m, int partials_size_n,
        int64_t inputs_offset, int64_t inputs_stride_m,
        int64_t inputs_stride_n, int inputs_size_m, int inputs_size_n,
        int64_t signals_offset,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks) {
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  for (int r0 = 0; r0 < input_size_m; r0 += partials_size_m) {
    const int band =
        min(partials_size_m, input_size_m - r0);  // this pass's rows
    // This pass's rows of the input and of the output.
    const DTYPE* const band_input = input_ptr + r0 * input_stride_m;
    DTYPE* const band_output = output_ptr + r0 * output_stride_m;
    // 1. Every rank's share of this band, read here, into that rank's inbox at
    //    slot [rank] (the chunks each rank's same block sums), then visible
    //    (and, past the first pass, every peer has read this rank's partials).
    push_shares<WORLD, THREADS_PER_BLOCK, TILE_N, CONTIGUOUS>(
        band_input, input_stride_m, input_stride_n, band, input_size_n, bases,
        inputs_offset, inputs_stride_m, inputs_stride_n, inputs_size_m,
        inputs_size_n, rank);
    barrier<Until::visible, 0, WORLD>(
        bases, signals_offset, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks);

    // 2. Reduce-scatter, local: this rank's share of the band, every writer's
    //    slot of its inbox summed, into its partials.
    reduce_slots<WORLD, THREADS_PER_BLOCK, TILE_N, DTYPE>(
        bases, inputs_offset, inputs_stride_m, inputs_stride_n, inputs_size_m,
        inputs_size_n, partials_offset, partials_stride_m, partials_stride_n,
        partials_size_m, partials_size_n, band, input_size_n, rank);

    // 3. Every rank's sums are visible to its peers, and every peer has summed
    //    its inbox, so the next band's (or launch's) push may overwrite it.
    barrier<Until::visible, 1, WORLD>(
        bases, signals_offset, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks);

    // 4. All-gather: every rank's summed share out of its partials, at its place
    // in the output.
    all_gather<WORLD, THREADS_PER_BLOCK, TILE_N, CONTIGUOUS>(
        bases, partials_offset, partials_stride_m, partials_stride_n, partials_size_m,
        partials_size_n, band_output, output_stride_m,
        output_stride_n, band, input_size_n, rank);
  }
#else
  __builtin_trap();
#endif
}

}  // namespace hip_custom_communicator
