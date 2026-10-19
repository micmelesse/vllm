// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE PLAIN ALL-REDUCE: its kernels, one section each. THE DEFAULT, PULLED IN
// PLACE (all_reduce_pull_one_shot, all_reduce_pull_two_shot;
// aiter's gfx950 cross_device_reduce_1stage / 2stage): ONE kernel a shape
// reads every rank's input where the INPUT TABLE says (common.cuh's INTERFACE:
// a captured call's registered tensors, or an eager call's copies in every
// rank's inputs region, handle.cuh's INPUT TABLES), no copy inside the kernel.
// Its inputs are never written while it runs: the one-shot ends with a `read`
// barrier and the two-shot's second barrier follows its every read of them, so
// when a rank's launch ends every peer is done reading its input, and the next
// call's host copy (stream-ordered after it) or the caller may overwrite it.
// The INT4 two-shot (all_reduce_pull_two_shot_int4, either two-shot template's INT4 build) encodes its
// input itself: what crosses the links is quantized by its owner, so it cannot
// be read in place; it pushes the quantized shares (below).
//
// THE PUSHED VARIANT (all_reduce_push_one_shot, all_reduce_push_two_shot; a
// caller forces it by its direction, select never picks it), each PUSHED: a rank reads only its own input (the caller's tensor, which no
// peer reads) and stores it, or each rank's share of it, into the peers'
// receive areas (their staging, the INBOX: [writer slot, ...]),
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

#include "all_gather.cuh"
#include "common.cuh"
#include "reduce_scatter.cuh"

namespace hip_custom_communicator {

// =============================================================================
// all_reduce_pull_one_shot
//
// One-shot all-reduce, PULLED IN PLACE: sync (every rank's input ready and
// visible), read every rank's input through the input table, sum, sync (every
// peer done reading this rank's). aiter's `cross_device_reduce_1stage`. No band
// loop: nothing is staged, so any size runs in one pass.
// =============================================================================

// THE BUFFER AS [m, n], every rank's at the table's strides (input_stride_m,
// input_stride_n), a work item a TILE_M x TILE_N tile, the grid striding over
// the tiles row-major.
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot(
        // THE ARGUMENTS WHAT THE KERNEL READS FIRST LEADS, the one exception to the canonical order
        // (STANDARDS 3.2): the pointer tables (their entries are read before the first barrier),
        // this rank and the signals' strides (the barrier's flag addresses) inside the 56 bytes
        // the kernarg preload puts in SGPRs (CMakeLists' --amdgpu-kernarg-preload-count=32), so
        // none of them waits for a kernarg load; everything else in the canonical order.
        const DTYPE* const* __restrict__ input_ptr_array, int input_ptr_array_size,
        uint32_t* const* signals_ptr_array, int signals_ptr_array_size, int rank,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer,
        int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        DTYPE* __restrict__ output_ptr, int64_t output_stride_m,
        int64_t output_stride_n, int output_size_m, int output_size_n,
        int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, uint64_t timeout_ticks) {
  constexpr int kThreadsM = TILE_M;
  constexpr int kThreadsN = THREADS_PER_BLOCK / TILE_M;
  constexpr int kRows = TILE_M / kThreadsM;
  constexpr int kPackElems = traits<DTYPE>::N;
  constexpr int kPacks = TILE_N / (kPackElems * kThreadsN);
  const int tiles_n = (input_size_n + TILE_N - 1) / TILE_N;
  const int tiles = (input_size_m + TILE_M - 1) / TILE_M * tiles_n;
  // 0. Every rank's input entry, read before the barrier (load's entry form: the table is fixed
  // for the launch), so its round trip runs under the barrier's.
  const DTYPE* input_entries[WORLD];
#pragma unroll
  for (int r = 0; r < WORLD; ++r) input_entries[r] = load(input_ptr_array, input_ptr_array_size, r);
  // 1. Every rank's input is ready (written before its launch: the caller's,
  // or the host's copy) and visible (the first barrier is ordered).
  barrier<WORLD>(
      signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks, Until::started, 0);
  // 2. Every rank's input, in rank order, through the table, summed.
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    DTYPE peers[WORLD][kRows][kPacks][kPackElems];
#pragma unroll
    for (int r = 0; r < WORLD; ++r)
      recv<WORLD, TILE_M, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&input_entries[r], 1, 0, 0, 0, input_stride_m, input_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, peers[r], t / tiles_n * TILE_M, t % tiles_n * TILE_N, 1);
    // Summed in rank order in fp32, rounded to DTYPE once: every rank sums the same copies in the
    // same order, so every rank's sums agree bitwise.
    DTYPE rounded[kRows][kPacks][kPackElems];
#pragma unroll
    for (int i = 0; i < kRows; ++i)
#pragma unroll
      for (int k = 0; k < kPacks; ++k)
#pragma unroll
        for (int j = 0; j < kPackElems; ++j) {
          float acc = static_cast<float>(peers[0][i][k][j]);
#pragma unroll
          for (int r = 1; r < WORLD; ++r) acc += static_cast<float>(peers[r][i][k][j]);
          rounded[i][k][j] = static_cast<DTYPE>(acc);
        }
    store<WORLD, TILE_M, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(rounded, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, t / tiles_n * TILE_M, t % tiles_n * TILE_N, 1);
  }
  // 3. Every peer is done reading this rank's input: the launch may end and
  // the input be overwritten.
  barrier<WORLD>(
      signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks, Until::read, 1);
}

// =============================================================================
// all_reduce_pull_two_shot
//
// Two-shot all-reduce, PULLED IN PLACE (exact): a band of `partials_size_m`
// rows a pass (what the partials hold): sync, this rank's share of the band
// read from every rank's input through the table and summed into its
// partials, sync, every rank's summed share pulled out of its partials. aiter's
// `cross_device_reduce_2stage`.
// =============================================================================

// THE PROBLEM as the pushed two-shot's (the flat walk, each rank's share one
// contiguous range of the band's elements), every rank's input at the table's
// strides. CONTIGUOUS: the table's rows and the output's back to back.
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU, bool CONTIGUOUS>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot(
        // THE ARGUMENTS WHAT THE KERNEL READS FIRST LEADS, the one exception to the canonical order
        // (STANDARDS 3.2): the pointer tables (their entries are read before the first barrier),
        // this rank and the signals' strides (the barrier's flag addresses) inside the 56 bytes
        // the kernarg preload puts in SGPRs (CMakeLists' --amdgpu-kernarg-preload-count=32), so
        // none of them waits for a kernarg load; everything else in the canonical order.
        const DTYPE* const* __restrict__ input_ptr_array, int input_ptr_array_size,
        uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size,
        char* const* __restrict__ partials_ptr_array, int partials_ptr_array_size, int rank,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer,
        int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        DTYPE* __restrict__ output_ptr, int64_t output_stride_m,
        int64_t output_stride_n, int output_size_m, int output_size_n,
        int64_t partials_stride_m,
        int64_t partials_stride_n, int partials_size_m, int partials_size_n,
        int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, uint64_t timeout_ticks) {
  for (int r0 = 0; r0 < input_size_m; r0 += partials_size_m) {
    const int band =
        min(partials_size_m, input_size_m - r0);  // this pass's rows
    DTYPE* const band_output = output_ptr + r0 * output_stride_m;
    // 1-3. Every rank's input ready and visible (and, past the first pass, every peer has read
    //    this rank's partials); this rank's share of the band from every rank's input, where the
    //    table says, summed into its partials; every rank's sums visible and every peer done
    //    reading this band of this rank's input.
    reduce_scatter<WORLD, TILE_M, TILE_N, THREADS_PER_BLOCK, Quantization::none, CONTIGUOUS, Walk::flat, Accumulate::lds, Direction::pull, Until::started>(
        static_cast<const DTYPE*>(nullptr), 0, 1, band, input_size_n, input_ptr_array,
        input_ptr_array_size, input_stride_m, input_stride_n, nullptr, 0, 0, 1, 0, 0,
        partials_ptr_array, partials_ptr_array_size, partials_stride_m, partials_stride_n,
        partials_size_m, partials_size_n, signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field, r0, rank, timeout_ticks);

    // 4. All-gather: every rank's summed share out of its partials, at its
    //    place in the output.
    all_gather<DTYPE, WORLD, TILE_N, THREADS_PER_BLOCK, Quantization::none, CONTIGUOUS, Walk::flat>(partials_ptr_array, partials_ptr_array_size, partials_stride_m,
                      partials_stride_n, partials_size_m, partials_size_n, band_output,
                      output_stride_m, output_stride_n, band, input_size_n, 0, 0, 0, rank);
  }
}

// =============================================================================
// all_reduce_push_one_shot
//
// One-shot all-reduce (the pushed variant, forced only): every rank sums every rank's input over the whole
// buffer. PUSHED: each rank stores its own band into slot [rank] of every
// rank's inbox (itself too), sync (ordered), each rank sums the WORLD slots of
// its own inbox, sync. One kernel; no peer reads the caller's input.
// =============================================================================

// THE BUFFER AS [m, n] AT ITS STRIDES, a work item a TILE_M x TILE_N tile (a
// row of threads a tile row), the grid striding over the tiles row-major. Both
// tile sides are tuned (select.cuh). THE INBOX, the staging as [WORLD *
// band_rows, n]: writer w's band at rows [w * band_rows, (w + 1) * band_rows),
// band_rows = staging_size_m / WORLD (the host's: what the region holds), so any
// size runs in one launch, a band a pass. EACH BLOCK PUSHES THE TILES ITS TWIN
// ON EVERY RANK SUMS: what the same block on a peer pushed is what a peers
// barrier makes visible.
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_push_one_shot(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        DTYPE* __restrict__ output_ptr, int64_t output_stride_m,
        int64_t output_stride_n, int output_size_m, int output_size_n,
        char* const* __restrict__ staging_ptr_array, int staging_ptr_array_size, int64_t staging_stride_m,
        int64_t staging_stride_n, int staging_size_m, int staging_size_n,
        uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks) {
  constexpr int kThreadsM = TILE_M;
  constexpr int kThreadsN = THREADS_PER_BLOCK / TILE_M;
  constexpr int kPackElems = traits<DTYPE>::N;
  constexpr int kRows = TILE_M / kThreadsM;
  constexpr int kPacks = TILE_N / (kPackElems * kThreadsN);
  const int tiles_n = (input_size_n + TILE_N - 1) / TILE_N;
  const int band_rows = staging_size_m / WORLD;  // a slot's rows
  const int64_t slot = int64_t{band_rows} * staging_stride_m;  // a slot's elements

  for (int r0 = 0; r0 < input_size_m; r0 += band_rows) {
    const int band =
        min(band_rows, input_size_m - r0);  // this pass's rows
    const int tiles = (band + TILE_M - 1) / TILE_M * tiles_n;
    // 1. This rank's band (the input's rows r0 to r0 + band), read here, into slot [rank] of every rank's inbox,
    // then visible to the peers (each has pushed its own into ours).
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      DTYPE mine[kRows][kPacks][kPackElems];
      load<WORLD, TILE_M, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1, input_size_m, input_size_n, r0 + band, input_size_n, mine, r0 + t / tiles_n * TILE_M, t % tiles_n * TILE_N, 1);
      for (int r = 0; r < WORLD; ++r)
        send<WORLD, TILE_M, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(mine, staging_ptr_array, staging_ptr_array_size, r, rank, slot, staging_stride_m, staging_stride_n, WORLD, band_rows, staging_size_n, band, input_size_n, t / tiles_n * TILE_M, t % tiles_n * TILE_N, 1);
    }
    barrier<WORLD>(
        signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks, Until::visible, 0);
    // 2. Every writer's slot of this rank's inbox, local, in rank order, summed.
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      DTYPE peers[WORLD][kRows][kPacks][kPackElems];
      load<WORLD, TILE_M, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(staging_ptr_array, staging_ptr_array_size, rank, 0, slot, staging_stride_m, staging_stride_n, WORLD, band_rows, staging_size_n, band, input_size_n, peers, t / tiles_n * TILE_M, t % tiles_n * TILE_N, 1);
      DTYPE rounded[kRows][kPacks][kPackElems];  // summed in rank order in fp32, rounded once (as above)
#pragma unroll
      for (int i = 0; i < kRows; ++i)
#pragma unroll
        for (int k = 0; k < kPacks; ++k)
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) {
            float acc = static_cast<float>(peers[0][i][k][j]);
#pragma unroll
            for (int r = 1; r < WORLD; ++r) acc += static_cast<float>(peers[r][i][k][j]);
            rounded[i][k][j] = static_cast<DTYPE>(acc);
          }
      store<WORLD, TILE_M, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(rounded, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, input_size_m, input_size_n, r0 + band, input_size_n, r0 + t / tiles_n * TILE_M, t % tiles_n * TILE_N, 1);
    }
    // 3. No rank may push its next band (this launch's or the next's) until
    // every peer has summed this one out of its inbox.
    barrier<WORLD>(
        signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks, Until::read, 1);
  }
}

// =============================================================================
// all_reduce_pull_two_shot_int4 and all_reduce_push_two_shot (exact, the
// pushed variant, forced only)
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
          int THREADS_PER_BLOCK, int WAVES_PER_EU,
          int QUANTIZATION, bool CONTIGUOUS>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot_int4(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        DTYPE* __restrict__ output_ptr, int64_t output_stride_m,
        int64_t output_stride_n, int output_size_m, int output_size_n,
        char* const* __restrict__ partials_ptr_array, int partials_ptr_array_size, int64_t partials_stride_m,
        int64_t partials_stride_n, int partials_size_m, int partials_size_n,
        char* const* __restrict__ staging_ptr_array, int staging_ptr_array_size, int64_t staging_stride_m,
        int64_t staging_stride_n, int staging_size_m, int staging_size_n,
        uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks) {
  // THE INT4 PAYLOADS are inside the staging and the partials (common.cuh's
  // INT4 payload layout): the collectives find them from each region's sizes.
  static_assert(QUANTIZATION == Quantization::int4,
                "none is all_reduce_pull_two_shot (or all_reduce_push_two_shot)");
  // 1-3. Every rank's share of this rank's input, quantized here, into that rank's inbox payload
  // at slot [rank]; every peer launched and pushed its own into ours (ordered); this rank's share,
  // every writer's slot summed, into its partials; every rank's sums visible and every peer done
  // with its inbox (the next launch's push may overwrite it).
  reduce_scatter<WORLD, TILE_M, TILE_N, THREADS_PER_BLOCK, QUANTIZATION, CONTIGUOUS, Walk::flat, Accumulate::registers, Direction::push, Until::launched>(
      input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
      static_cast<const DTYPE* const*>(nullptr), 0, 0, 1,
      staging_ptr_array, staging_ptr_array_size, staging_stride_m, staging_stride_n,
      staging_size_m, staging_size_n, partials_ptr_array, partials_ptr_array_size,
      partials_stride_m, partials_stride_n, partials_size_m, partials_size_n, signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field, 0, rank,
      timeout_ticks);

  // 4. All-gather: every rank's summed share out of its partials, at its place
  // in the output. The next call's first sync keeps a rank from overwriting
  // its partials while it is read (they are written only after it).
  all_gather<DTYPE, WORLD, TILE_N, THREADS_PER_BLOCK, QUANTIZATION, CONTIGUOUS, Walk::flat>(
      partials_ptr_array, partials_ptr_array_size, partials_stride_m, partials_stride_n,
      partials_size_m, partials_size_n, output_ptr, output_stride_m, output_stride_n,
      input_size_m, input_size_n, 0, 0, 0, rank);
}

// EXACT, IN PASSES: a band of `partials_size_m` rows a pass (the host's: what
// the inbox and the partials hold), so any size runs in one launch. A band is
// the problem's rows [r0, r0 + band) at their strides. THE INBOX, the staging
// region as [WORLD, slot]: writer w's slot at row w, staging_stride_m elements
// a slot, holding writer w's copy of this rank's share of the band from the
// slot's first element. EACH BLOCK PUSHES, OF EVERY RANK'S SHARE, THE CHUNKS
// THAT RANK'S SAME BLOCK SUMS, so a peers barrier makes them visible.
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU, bool CONTIGUOUS>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_push_two_shot(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        DTYPE* __restrict__ output_ptr, int64_t output_stride_m,
        int64_t output_stride_n, int output_size_m, int output_size_n,
        char* const* __restrict__ partials_ptr_array, int partials_ptr_array_size, int64_t partials_stride_m,
        int64_t partials_stride_n, int partials_size_m, int partials_size_n,
        char* const* __restrict__ staging_ptr_array, int staging_ptr_array_size, int64_t staging_stride_m,
        int64_t staging_stride_n, int staging_size_m, int staging_size_n,
        uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks) {
  for (int r0 = 0; r0 < input_size_m; r0 += partials_size_m) {
    const int band =
        min(partials_size_m, input_size_m - r0);  // this pass's rows
    DTYPE* const band_output = output_ptr + r0 * output_stride_m;
    // 1-3. Every rank's share of this band, read here, into that rank's inbox at slot [rank] (the
    //    chunks each rank's same block sums), then visible (and, past the first pass, every peer
    //    has read this rank's partials); this rank's share, every writer's slot of its inbox
    //    summed, into its partials; every rank's sums visible and every peer done with its inbox,
    //    so the next band's (or launch's) push may overwrite it.
    reduce_scatter<WORLD, TILE_M, TILE_N, THREADS_PER_BLOCK, Quantization::none, CONTIGUOUS, Walk::flat, Accumulate::lds, Direction::push, Until::visible>(
        input_ptr, input_stride_m, input_stride_n, band, input_size_n,
        static_cast<const DTYPE* const*>(nullptr), 0, 0, 1,
        staging_ptr_array, staging_ptr_array_size, staging_stride_m, staging_stride_n,
        staging_size_m, staging_size_n, partials_ptr_array, partials_ptr_array_size,
        partials_stride_m, partials_stride_n, partials_size_m, partials_size_n, signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field, r0, rank,
        timeout_ticks);

    // 4. All-gather: every rank's summed share out of its partials, at its place
    // in the output.
    all_gather<DTYPE, WORLD, TILE_N, THREADS_PER_BLOCK, Quantization::none, CONTIGUOUS, Walk::flat>(partials_ptr_array, partials_ptr_array_size, partials_stride_m,
                      partials_stride_n, partials_size_m, partials_size_n, band_output,
                      output_stride_m, output_stride_n, band, input_size_n, 0, 0, 0, rank);
  }
}

}  // namespace hip_custom_communicator
