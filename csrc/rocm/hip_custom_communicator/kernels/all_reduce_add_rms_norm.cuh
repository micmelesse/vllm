// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ALL-REDUCE THEN RMSNORM, AND THEN ADD THEN RMSNORM: their kernels, one
// section each (pull one-shot, pull two-shot, push two-shot), each kernel of
// the RMSNorm op a wrapper over its add-RMSNorm body; then our separate norm
// kernels (rms_norm_rows, add_rms_norm_rows), the splits' second.

#pragma once

#include "common.cuh"
#include "reduce_scatter.cuh"
#include "rms_norm.cuh"

namespace hip_custom_communicator {

// =============================================================================
// all_reduce_pull_one_shot_add_rms_norm
//
// One-shot pull all-reduce then RMSNorm (`all_reduce_pull_one_shot_rms_norm`),
// and all-reduce then add then RMSNorm
// (`all_reduce_pull_one_shot_add_rms_norm`): one body, a kernel per op, so a
// trace names the op that ran.
// =============================================================================

// Every rank reduces every row and norms it where the sum lands in registers,
// saving an HBM round trip and a launch against an all-reduce then a norm
// kernel. A block owns a row, held as `T v[1][K][P]` (a row a block: one row
// of threads, THREADS_PER_BLOCK lanes, K packs a lane). `residual_ptr` and
// `residual_output_ptr` are unused (null) unless ADD_RESIDUAL; the weight keeps
// its own dtype WEIGHT_DTYPE (DTYPE or fp32), as vLLM's reference ops round to
// the WEIGHT's dtype (`vllm/ir/ops/layernorm.py`). THE ROW'S PLACE IS COUNTED
// IN THE INPUT'S SIZES for every buffer, as before the port (the host checks
// they are equal): each buffer's own sizes are a change of their own, compiled
// and measured. The peers' inputs are theirs after the first barrier.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD,
          int TILE_N, int THREADS_PER_BLOCK, bool ADD_RESIDUAL>
DINLINE void all_reduce_pull_one_shot_add_rms_norm_body(
    const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
    int64_t input_stride_n, int input_size_m, int input_size_n,
    const DTYPE* const* __restrict__ input_ptr_array, int input_ptr_array_size,
    int64_t input_ptr_array_stride_m, int64_t input_ptr_array_stride_n,
    const DTYPE* residual_ptr, int64_t residual_stride_m,
    int64_t residual_stride_n, const WEIGHT_DTYPE* weight_ptr,
    int64_t weight_stride_n, DTYPE* output_ptr, int64_t output_stride_m,
    int64_t output_stride_n, DTYPE* residual_output_ptr,
    int64_t residual_output_stride_m, int64_t residual_output_stride_n,
    char* const* staging_ptr_array, int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
    int status_size_field, int rank, uint64_t timeout_ticks, float eps) {
  constexpr int kPackElems = traits<DTYPE>::N;
  constexpr int kPacks = TILE_N / (kPackElems * THREADS_PER_BLOCK);
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);

  // 1. Wait until every peer has launched: its input, the caller's tensor (a
  // captured call's, registered) or the host's copy in its staging (an eager
  // call's), is ready before it. Read where the input table says, in place.
  // UNORDERED (started, as the plain pull one-shot's): nothing is written before
  // it, and every read after it is of what a peer wrote before its launch.
  barrier<WORLD>(
      signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks, Until::started, 0);

  // 2. Each of this block's rows: read it from every rank in rank order and
  // sum, then (ADD_RESIDUAL) add
  //    the residual, then RMSNorm, rounding as the reference does:
  //      s   = float(DTYPE(sum over ranks))             the all-reduce output,
  //      as it would land s  += float(residual); residual_out = DTYPE(s)
  //      ADD_RESIDUAL only (fused_add_rms_norm) out =
  //      DTYPE(WEIGHT_DTYPE(WEIGHT_DTYPE(s * rsqrt(mean(s^2) + eps)) *
  //      float(w)))
  //    The variance is of `s` before any further rounding, kept in registers
  //    between the passes.
  for (int row = blockIdx.x; row < input_size_m; row += gridDim.x) {
    // Every load of the row before any store, in flight together: the peers',
    // the residual's and the weight's. A load issues where it is written (an
    // address is built at its load), and the weight after the peers' sum was a
    // round trip of its own (+0.34 us at 16 x 3584, 2026-10-04T17-25-35Z).
    DTYPE peers[WORLD][1][kPacks][kPackElems];
    for (int r = 0; r < WORLD; ++r)
      recv<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(input_ptr_array, input_ptr_array_size, r, 0, 0, input_ptr_array_stride_m, input_ptr_array_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, peers[r], row, 0, 1);
    DTYPE res[1][kPacks][kPackElems];
    if constexpr (ADD_RESIDUAL)
      load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&residual_ptr, 1, 0, 0, 0, residual_stride_m, residual_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, res, row, 0, 1);
    WEIGHT_DTYPE w[1][kPacks][kPackElems];
    load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&weight_ptr, 1, 0, 0, 0, 0, weight_stride_n, 1, 1, input_size_n, 1, input_size_n, w, 0, 0, 1);
    // The peers' sum in rank order in fp32, so every rank's agrees bitwise, rounded to DTYPE as
    // the reference's all-reduce is; ADD_RESIDUAL adds the residual in fp32, the new residual that
    // sum rounded to DTYPE (the norm reads the fp32 sum).
    float s[1][kPacks][kPackElems];
    [[maybe_unused]] DTYPE added[1][kPacks][kPackElems];
#pragma unroll
    for (int k = 0; k < kPacks; ++k)
#pragma unroll
      for (int j = 0; j < kPackElems; ++j) {
        float acc = static_cast<float>(peers[0][0][k][j]);
#pragma unroll
        for (int r = 1; r < WORLD; ++r) acc += static_cast<float>(peers[r][0][k][j]);
        s[0][k][j] = static_cast<float>(static_cast<DTYPE>(acc));
        if constexpr (ADD_RESIDUAL) {
          s[0][k][j] += static_cast<float>(res[0][k][j]);
          added[0][k][j] = static_cast<DTYPE>(s[0][k][j]);
        }
      }
    if constexpr (ADD_RESIDUAL) {
      store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(added, &residual_output_ptr, 1, 0, 0, 0, residual_output_stride_m, residual_output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, row, 0, 1);
    }
    // out = T(W(W(s * scale) * float(w))), as the reference rounds (rms_norm.cuh)
    float scale[1];
    DTYPE normed[1][kPacks][kPackElems];
    rms_norm<THREADS_PER_BLOCK>(s, scale, w, normed, eps, inv_hidden,
                                                     input_size_n, 0, NormPart::row, true, Layout::tile);
    store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(normed, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, row, 0, 1);
  }

  // 3. No rank may overwrite its input until every peer has read it.
  barrier<WORLD>(
      signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks, Until::read, 1);
}

// THE KERNELS, one per op, both the body above.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot_rms_norm(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const DTYPE* const* __restrict__ input_ptr_array, int input_ptr_array_size,
    int64_t input_ptr_array_stride_m, int64_t input_ptr_array_stride_n,
        const WEIGHT_DTYPE* __restrict__ weight_ptr, int64_t weight_stride_n,
        int weight_size_n, DTYPE* __restrict__ output_ptr,
        int64_t output_stride_m, int64_t output_stride_n, int output_size_m,
        int output_size_n, char* const* __restrict__ staging_ptr_array, int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks, float eps) {
  all_reduce_pull_one_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, TILE_N, THREADS_PER_BLOCK, false>(
      input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
      input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m, input_ptr_array_stride_n,
      nullptr, 0, 0, weight_ptr, weight_stride_n, output_ptr, output_stride_m,
      output_stride_n, nullptr, 0, 0, staging_ptr_array, staging_ptr_array_size, staging_stride_m, staging_stride_n,
      staging_size_m, staging_size_n, signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
      signals_stride_block, signals_stride_writer, signals_size_parity,
      signals_size_block, signals_size_writer, status_ptr, status_stride_field,
      status_size_field, rank, timeout_ticks, eps);
}

template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot_add_rms_norm(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const DTYPE* const* __restrict__ input_ptr_array, int input_ptr_array_size,
    int64_t input_ptr_array_stride_m, int64_t input_ptr_array_stride_n,
        const DTYPE* __restrict__ residual_ptr, int64_t residual_stride_m,
        int64_t residual_stride_n, int residual_size_m, int residual_size_n,
        const WEIGHT_DTYPE* __restrict__ weight_ptr, int64_t weight_stride_n,
        int weight_size_n, DTYPE* __restrict__ output_ptr,
        int64_t output_stride_m, int64_t output_stride_n, int output_size_m,
        int output_size_n, DTYPE* __restrict__ residual_output_ptr,
        int64_t residual_output_stride_m, int64_t residual_output_stride_n,
        int residual_output_size_m, int residual_output_size_n,
        char* const* __restrict__ staging_ptr_array, int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks, float eps) {
  all_reduce_pull_one_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, TILE_N, THREADS_PER_BLOCK, true>(
      input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
      input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m, input_ptr_array_stride_n,
      residual_ptr, residual_stride_m, residual_stride_n, weight_ptr,
      weight_stride_n, output_ptr, output_stride_m, output_stride_n,
      residual_output_ptr, residual_output_stride_m, residual_output_stride_n,
      staging_ptr_array, staging_ptr_array_size, staging_stride_m, staging_stride_n,
      staging_size_m, staging_size_n, signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks, eps);
}

// =============================================================================
// all_reduce_pull_two_shot_add_rms_norm
//
// Two-shot pull all-reduce then RMSNorm (`all_reduce_pull_two_shot_rms_norm`),
// and all-reduce then add then RMSNorm
// (`all_reduce_pull_two_shot_add_rms_norm`): one body, a kernel per op, so a
// trace names the op that ran. THE SLICE IS ROWS, for large inputs: the
// column-slice kernel (all_reduce_push_two_shot_*) wins small ones and loses at
// prefill.
// =============================================================================

// THE SLICE IS ROWS: each rank owns whole rows, so it computes each row's norm
// alone. It reduces its rows and leaves one row each in its partials, row-major:
// the normed row, or (ADD_RESIDUAL) the new residual and the row's RMS scale;
// after the sync every rank copies every owner's rows out, (ADD_RESIDUAL)
// norming them by their scales as it goes. Every rank does the same arithmetic
// on the same bytes, so every rank holds the same result. THE SAME BLOCK AND
// THREAD INDEX A PACK IN BOTH PHASES: after the sync a block may read only what
// the same block on a peer wrote. The buffers' places are counted as the
// one-shot's are (its head says why). QUANTIZATION: what crosses the links
// (common.cuh's QUANTIZATION), one body each: none (THE EXACT BUILD, below) reads every rank's
// input in place; int4 (THE INT4 BUILD, after it) pushes it quantized, a group a thread.
// THE EXACT BUILD: every rank's input read in place where the input table says (a captured
// call's tensor, or an eager call's host copy in its staging), a pack a thread.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD,
          int TILE_N, int THREADS_PER_BLOCK, bool ADD_RESIDUAL>
DINLINE void all_reduce_pull_two_shot_add_rms_norm_exact_body(
    const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
    int64_t input_stride_n, int input_size_m, int input_size_n,
    const DTYPE* const* __restrict__ input_ptr_array, int input_ptr_array_size,
    int64_t input_ptr_array_stride_m, int64_t input_ptr_array_stride_n,
    const DTYPE* residual_ptr, int64_t residual_stride_m,
    int64_t residual_stride_n, const WEIGHT_DTYPE* weight_ptr,
    int64_t weight_stride_n, DTYPE* output_ptr, int64_t output_stride_m,
    int64_t output_stride_n, DTYPE* residual_output_ptr,
    int64_t residual_output_stride_m, int64_t residual_output_stride_n,
    char* const* partials_ptr_array, int partials_ptr_array_size, int64_t partials_stride_m,
    int64_t partials_stride_n, int partials_size_m, int partials_size_n,
    float* const* scales_ptr_array, int scales_ptr_array_size, int64_t scales_stride_m,
    int scales_size_m, char* const* staging_ptr_array, int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
    int status_size_field, int rank, uint64_t timeout_ticks, float eps) {
  constexpr int kPackElems = traits<DTYPE>::N;
  constexpr int kPacks = TILE_N / (kPackElems * THREADS_PER_BLOCK);
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);
  const int slice_m = (input_size_m + WORLD - 1) / WORLD;
  // A PARTIALS ROW (ADD_RESIDUAL) IS ITS SCALE THEN ITS ROW: [one pack: the row's scale, a float,
  // and padding][input_size_n elements], partials_stride_m = input_size_n + kScaleSlot (the host
  // sizes it: select.cuh's partials_need), so the gather's first step reads a row's scale beside
  // its first pack, from the same row of the same buffer (the separate scales region cost the
  // gather a round trip a row before the rows' loads: ATT 2026-10-18T10-35-18Z). The scale's view
  // is the row as floats, its column 0. RMSNorm's rows are the normed rows alone (no slot).
  constexpr int kScaleSlot = ADD_RESIDUAL ? kPackElems : 0;
  [[maybe_unused]] const int64_t scale_stride_m =
      partials_stride_m * static_cast<int64_t>(sizeof(DTYPE)) / static_cast<int64_t>(sizeof(float));
  const int row_n = input_size_n + kScaleSlot;  // a partials row's elements

  // 1. Wait until every peer has launched (its input is ready before it). UNORDERED (started):
  // nothing written before it, every read after it of a peer's input, written before its launch;
  // this rank's partials are written after it, which every peer finished reading in its last
  // launch (ended before this one started).
  barrier<WORLD>(
      signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks, Until::started, 0);


  // 2. This rank's rows: read each from every rank in rank order and sum, then
  // (ADD_RESIDUAL) add the
  //    residual, then RMSNorm, rounding as the reference does (the one-shot
  //    kernel spells it out), and leave the normed rows or (ADD_RESIDUAL) the
  //    residual rows and scales in this rank's partials. PIPELINED: the next
  //    row's loads go out before this row's reduction and norm, so a block's
  //    compute runs under its next round trip instead of between them (a block
  //    had ~14 rows at 4096 tokens, each 1.24 us of reduction and norm on the
  //    critical path: 2026-10-01T00-01-54Z).
  const int first = rank * slice_m;
  const int last = min(first + slice_m, input_size_m);
  // Row `row` from every rank.
  const auto load_peers = [&](int row, DTYPE(&got)[WORLD][1][kPacks][kPackElems]) {
    for (int r = 0; r < WORLD; ++r)
      recv<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(input_ptr_array, input_ptr_array_size, r, 0, 0, input_ptr_array_stride_m, input_ptr_array_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, got[r], row, 0, 1);
  };
  // THE WEIGHT ONCE, AND EVERY OTHER LOAD BEFORE THE NEXT ROW'S: loads complete
  // in issue order, so waiting on one issued after the peers' would wait on the
  // peers' too.
  WEIGHT_DTYPE w[1][kPacks][kPackElems];
  load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&weight_ptr, 1, 0, 0, 0, 0, weight_stride_n, 1, 1, input_size_n, 1, input_size_n, w, 0, 0, 1);
  // ONE ROW: its residual, then the next row's peer loads into `next`, then
  // this row's sum (its wait covers only its own, older, loads), so the next
  // round trip runs under the reduction and norm. Its rows land in this rank's
  // partials at row - first.
  const auto one_row = [&](int row, const DTYPE(&cur)[WORLD][1][kPacks][kPackElems],
                           DTYPE(&next)[WORLD][1][kPacks][kPackElems]) {
    DTYPE res[1][kPacks][kPackElems];
    if constexpr (ADD_RESIDUAL)
      load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&residual_ptr, 1, 0, 0, 0, residual_stride_m, residual_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, res, row, 0, 1);
    // ONLY A ROW THAT EXISTS: issued here, never hoisted, so the block-uniform
    // branch costs nothing, where a clamped unconditional load re-read the last
    // row (a block's whole round trip again; at 256 tokens every block has one
    // row: 2026-10-01T01-07-56Z).
    if (row + static_cast<int>(gridDim.x) < last)
      load_peers(row + gridDim.x, next);
    // The sum and the residual, rounding as the reference does (see the one-shot kernel):
    float s[1][kPacks][kPackElems];
    [[maybe_unused]] DTYPE added[1][kPacks][kPackElems];
#pragma unroll
    for (int k = 0; k < kPacks; ++k)
#pragma unroll
      for (int j = 0; j < kPackElems; ++j) {
        float acc = static_cast<float>(cur[0][0][k][j]);
#pragma unroll
        for (int r = 1; r < WORLD; ++r) acc += static_cast<float>(cur[r][0][k][j]);
        s[0][k][j] = static_cast<float>(static_cast<DTYPE>(acc));
        if constexpr (ADD_RESIDUAL) {
          s[0][k][j] += static_cast<float>(res[0][k][j]);
          added[0][k][j] = static_cast<DTYPE>(s[0][k][j]);
        }
      }
    if constexpr (ADD_RESIDUAL) {
      store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(added, partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n, input_size_m, row_n, row - first, kScaleSlot, 1);
    }
    // ADD_RESIDUAL LEAVES THE NEW RESIDUAL AND ITS SCALE, NOT THE NORMED ROW:
    // every rank norms it while gathering, so the link carries one row a row,
    // as a plain all-reduce does (gathering both the normed row and the
    // residual was twice that: 215.9 against 155.2 us at 4096 tokens,
    // 2026-10-01T02-17-54Z).
    float scale[1];
    DTYPE normed[1][kPacks][kPackElems];
    if constexpr (ADD_RESIDUAL) {
      rms_norm<THREADS_PER_BLOCK>(s, scale, w, normed, eps, inv_hidden,
                                                         input_size_n, 0, NormPart::scale, true, Layout::tile);
      // The row's scale into its slot, the partials row's first float: a column of one element,
      // so its one pack is thread 0's.
      float scale_row[1][1][1] = {{{scale[0]}}};
      store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
          scale_row, partials_ptr_array, partials_ptr_array_size, rank, 0, 0, scale_stride_m, 1, 1,
          partials_size_m, scale_stride_m, partials_size_m, 1, row - first, 0, 1);
      return;
    }
    // out = T(W(W(s * scale) * float(w))), as the reference rounds (rms_norm.cuh)
    rms_norm<THREADS_PER_BLOCK>(s, scale, w, normed, eps, inv_hidden,
                                                     input_size_n, 0, NormPart::row, true, Layout::tile);
    store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(normed, partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n, input_size_m, input_size_n, row - first, 0, 1);
  };
  // PING-PONG: two buffers that trade roles each row, so no row copies its
  // packs into the other (a copy cost 32 moves a row at one pack a thread: ISA
  // 2026-10-01T00-58-37Z).
  DTYPE a[WORLD][1][kPacks][kPackElems], b[WORLD][1][kPacks][kPackElems];
  int row = first + blockIdx.x;
  if (row < last) load_peers(row, a);
  for (; row < last; row += 2 * gridDim.x) {
    one_row(row, a, b);
    if (row + static_cast<int>(gridDim.x) >= last) break;
    one_row(row + gridDim.x, b, a);
  }

  // 3. Every rank's rows are visible to its peers.
  barrier<WORLD>(
      signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks, Until::visible, 1);

  // 4. Every owner's rows out of its partials, at their place in the output. The
  // next call's
  //    first sync keeps a rank from overwriting its partials while it is read.
  //    EVERY OWNER'S PACK LOADED BEFORE ANY IS STORED, and every load
  //    unconditional (each rank's partials holds slice_m rows, so a slot past
  //    the last row is real): a store between two loads, or a load under an
  //    `if`, made the eight owners' round trips run one after another.
  //    ADD_RESIDUAL: the owners' rows are the new residual; each is normed here
  //    by its owner's scale, read from the slot before the row in its first
  //    step (a wave's lanes read one scale address: one request a wave).
  // A PACK A THREAD, stepping across the row (THREADS_PER_BLOCK packs a step,
  // kPacks = 1): every owner's step in flight is WORLD packs a thread, as the
  // pack-at-a-time copy was (a whole row's was 16 at 7168, and 2-3% slower).
  // THE PARTIALS'S POINTERS are computed where they are read, from the
  // partials' array, so nothing held across the reduce (a table read after the
  // first barrier, held, ran the int4 8192-column build out of SGPRs:
  // 2026-10-15T22-28-01Z).
  for (int l = blockIdx.x; l < slice_m; l += gridDim.x) {
    // ONE STEP: every owner's pack at column c, then (ADD_RESIDUAL) each normed by its owner's
    // scale and stored. THE FIRST STEP ALSO READS THE SCALES, each from its row's slot, issued
    // after that step's packs: one wait covers both, and the later steps hold them (loaded before
    // the loop, they were copied into the loop's registers and waited on before the rows' loads
    // issued: two round trips a row, ATT 2026-10-18T10-35-18Z).
    [[maybe_unused]] float sc[WORLD][1][1][1];
    const auto step = [&](int c, bool scales) {
      // THE GATHER: row l of every owner's partials (owner r's is row r * slice_m + l), one load
      // of every rank's; ADD_RESIDUAL norms the gathered rows below, by their owners' scales.
      DTYPE got[WORLD][1][1][kPackElems];
      for (int r = 0; r < WORLD; ++r)
        recv<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
            partials_ptr_array, partials_ptr_array_size, r, 0, 0, partials_stride_m, partials_stride_n, 1,
            partials_size_m, partials_size_n, slice_m, row_n, got[r], l, c + kScaleSlot, 1);
      if constexpr (ADD_RESIDUAL)
        if (scales)
          for (int r = 0; r < WORLD; ++r)
            recv<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
                partials_ptr_array, partials_ptr_array_size, r, 0, 0, scale_stride_m, 1, 1,
                partials_size_m, scale_stride_m, slice_m, 1, sc[r], l, 0, 1);
      WEIGHT_DTYPE w_step[1][1][kPackElems];
      if constexpr (ADD_RESIDUAL)
        load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&weight_ptr, 1, 0, 0, 0, 0, weight_stride_n, 1, 1, input_size_n, 1, input_size_n, w_step, 0, c, 1);
#pragma unroll
      for (int r = 0; r < WORLD; ++r) {
        const int row = r * slice_m + l;
        if (row >= input_size_m) continue;
        if constexpr (ADD_RESIDUAL) {
          store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(got[r], &residual_output_ptr, 1, 0, 0, 0, residual_output_stride_m, residual_output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, row, c, 1);
          float got_f[1][1][kPackElems];
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) got_f[0][0][j] = static_cast<float>(got[r][0][0][j]);
          float scale[1] = {sc[r][0][0][0]};
          DTYPE normed[1][1][kPackElems];
          rms_norm<THREADS_PER_BLOCK>(got_f, scale, w_step, normed, eps,
                                                             inv_hidden, input_size_n, c, NormPart::apply, true, Layout::tile);
          store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(normed, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, row, c, 1);
        } else {
          store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(got[r], &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, row, c, 1);
        }
      }
    };
    constexpr int kStep = THREADS_PER_BLOCK * kPackElems;
    if constexpr (ADD_RESIDUAL) {
      step(0, true);
      for (int c = kStep; c < input_size_n; c += kStep) step(c, false);
    } else {
      for (int c = 0; c < input_size_n; c += kStep) step(c, false);
    }
  }
}

// THE INT4 BUILD, A GROUP A THREAD (TILE_N = kInt4Group * THREADS_PER_BLOCK: a block's threads
// are one row's groups, thread t group t): every access a group's 16-byte payload and its scale,
// where the pack-a-thread INT4 path read a 4-byte word and a 2-byte scale a pack, as many
// requests as the exact build for a quarter of the bytes (INT4 559.6 against exact 595.4 us at
// 8192 x 7168 rms_norm on 36 blocks; 2026-10-18T04-52-25Z). THE SHAPE IS THE PLAIN INT4
// TWO-SHOT'S (reduce_scatter.cuh's flat walk: push, sum local, pull 16-byte groups) cut by rows:
//   1. THE PUSH: every owner's rows of this rank's input, quantized here and sent to that owner's
//      staging inbox at this rank's slot ([WORLD slots of slice_m rows] INT4; the host passes the
//      staging as [WORLD * slice_m, n]).
//   2. THE SUM, LOCAL: this rank's rows, every writer's group from its own inbox, summed from the
//      words (dequantize's slot form), (ADD_RESIDUAL) the residual added, normed (or its scale
//      taken), and the row left INT4 in this rank's partials (the scale a float in its scales).
//   3. THE GATHER: every owner's rows out of its partials, a 16-byte group of each in flight,
//      decoded, (ADD_RESIDUAL) normed by its owner's scale, stored.
// A block's rows are l = blockIdx.x, + gridDim.x, ... of each owner's slice in all three, so a
// block reads after a barrier only what its twins wrote before it.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD,
          int TILE_N, int THREADS_PER_BLOCK, bool ADD_RESIDUAL>
DINLINE void all_reduce_pull_two_shot_add_rms_norm_int4_body(
    const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
    int64_t input_stride_n, int input_size_m, int input_size_n,
    const DTYPE* residual_ptr, int64_t residual_stride_m,
    int64_t residual_stride_n, const WEIGHT_DTYPE* weight_ptr,
    int64_t weight_stride_n, DTYPE* output_ptr, int64_t output_stride_m,
    int64_t output_stride_n, DTYPE* residual_output_ptr,
    int64_t residual_output_stride_m, int64_t residual_output_stride_n,
    char* const* partials_ptr_array, int partials_ptr_array_size, int64_t partials_stride_m,
    int64_t partials_stride_n, int partials_size_m, int partials_size_n,
    float* const* scales_ptr_array, int scales_ptr_array_size, int64_t scales_stride_m,
    int scales_size_m, char* const* staging_ptr_array, int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
    int status_size_field, int rank, uint64_t timeout_ticks, float eps) {
  constexpr int kPackElems = traits<DTYPE>::N;
  static_assert(kPackElems * kInt4GroupPacks == kInt4Group, "a pack is a word of its group");
  static_assert(TILE_N == kInt4Group * THREADS_PER_BLOCK, "a group a thread: the block one row's groups");
  const int n = input_size_n;
  const float inv_hidden = 1.0f / static_cast<float>(n);
  const int slice_m = (input_size_m + WORLD - 1) / WORLD;
  const int slice = slice_m * n;  // a rank's rows, flat: its slot of an inbox
  const auto rows_of = [&](int r) { return max(0, min(slice_m, input_size_m - r * slice_m)); };
  const int grid = static_cast<int>(gridDim.x);

  // 1. THE PUSH, owners rotated (rank + w), a row of each a step, TWO OWNERS' ROWS IN FLIGHT (as
  // the flat walk's push: every owner's loads hoisted together are 8 groups of input a thread).
  for (int l = blockIdx.x; l < slice_m; l += grid) {
#pragma unroll 2
    for (int w = 0; w < WORLD; ++w) {
      const int r = (rank + w) % WORLD;
      if (l >= rows_of(r)) continue;  // a late owner's short slice (every thread alike)
      DTYPE x[1][kInt4GroupPacks][kPackElems];
      load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::group_run>(
          &input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1, input_size_m, n,
          input_size_m, n, x, r * slice_m + l, 0, 1);
      // QUANTIZED ONCE, by this thread alone (its group), then sent as the payload.
      uint32_t words[kInt4GroupPacks];
      float scale;
      quantize(x[0], words, scale);
      uint32_t payload[1][kInt4GroupPacks][2];
#pragma unroll
      for (int k = 0; k < kInt4GroupPacks; ++k) {
        payload[0][k][0] = words[k];
        payload[0][k][1] = __builtin_bit_cast(uint32_t, scale);
      }
      send<WORLD, 1, THREADS_PER_BLOCK, Quantization::int4, Layout::group_run>(
          payload, staging_ptr_array, staging_ptr_array_size, r, rank, slice, staging_stride_m,
          staging_stride_n, WORLD, staging_size_m, staging_size_n, 1, (l + 1) * n, 0, l * n, 1);
    }
  }
  barrier<WORLD>(
      signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks, Until::launched, 0);

  // 2. THIS RANK'S ROWS. The weight once (the same columns, a thread's group, in every row).
  WEIGHT_DTYPE w[1][kInt4GroupPacks][kPackElems];
  load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::group_run>(
      &weight_ptr, 1, 0, 0, 0, 0, weight_stride_n, 1, 1, n, 1, n, w, 0, 0, 1);
  for (int l = blockIdx.x; l < rows_of(rank); l += grid) {
    const int row = rank * slice_m + l;
    // Every writer's group of the row, from this rank's inbox (slots rotated from this rank's),
    // and the residual's, all in flight before the sum.
    uint32_t got[WORLD][1][kInt4GroupPacks][2];
    load<WORLD, 1, THREADS_PER_BLOCK, Quantization::int4, Layout::group_run>(
        staging_ptr_array, staging_ptr_array_size, rank, rank, slice, staging_stride_m,
        staging_stride_n, WORLD, staging_size_m, staging_size_n, 1, (l + 1) * n, got, 0, l * n, 1);
    DTYPE res[1][kInt4GroupPacks][kPackElems];
    if constexpr (ADD_RESIDUAL)
      load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::group_run>(
          &residual_ptr, 1, 0, 0, 0, residual_stride_m, residual_stride_n, 1, input_size_m, n,
          input_size_m, n, res, row, 0, 1);
    float sum[1][kInt4GroupPacks][kPackElems];
    dequantize<DTYPE>(got, sum);
    // The sum rounded to DTYPE, as the reference's all-reduce lands it; ADD_RESIDUAL adds the
    // residual in fp32, the new residual that sum rounded to DTYPE (the norm reads the fp32 sum).
    float s[1][kInt4GroupPacks][kPackElems];
    [[maybe_unused]] DTYPE added[1][kInt4GroupPacks][kPackElems];
#pragma unroll
    for (int k = 0; k < kInt4GroupPacks; ++k)
#pragma unroll
      for (int j = 0; j < kPackElems; ++j) {
        s[0][k][j] = static_cast<float>(static_cast<DTYPE>(sum[0][k][j]));
        if constexpr (ADD_RESIDUAL) {
          s[0][k][j] += static_cast<float>(res[0][k][j]);
          added[0][k][j] = static_cast<DTYPE>(s[0][k][j]);
        }
      }
    float scale[1];
    DTYPE normed[1][kInt4GroupPacks][kPackElems];
    if constexpr (ADD_RESIDUAL) {
      // THE NEW RESIDUAL AND ITS SCALE, NOT THE NORMED ROW (the exact body says why).
      store<WORLD, 1, THREADS_PER_BLOCK, Quantization::int4, Layout::group_run>(
          added, partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m,
          partials_stride_n, 1, partials_size_m, partials_size_n, 1, (l + 1) * n, 0, l * n, 1);
      rms_norm<THREADS_PER_BLOCK>(s, scale, w, normed, eps, inv_hidden, n, 0, NormPart::scale,
                                  true, Layout::group_run);
      float scale_row[1][1][1] = {{{scale[0]}}};
      store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
          scale_row, scales_ptr_array, scales_ptr_array_size, rank, 0, 0, scales_stride_m, 1, 1,
          scales_size_m, 1, scales_size_m, 1, l, 0, 1);
    } else {
      rms_norm<THREADS_PER_BLOCK>(s, scale, w, normed, eps, inv_hidden, n, 0, NormPart::row,
                                  true, Layout::group_run);
      store<WORLD, 1, THREADS_PER_BLOCK, Quantization::int4, Layout::group_run>(
          normed, partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m,
          partials_stride_n, 1, partials_size_m, partials_size_n, 1, (l + 1) * n, 0, l * n, 1);
    }
  }

  // 3. Every rank's rows are visible to its peers.
  barrier<WORLD>(
      signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks, Until::visible, 1);

  // 4. EVERY OWNER'S ROW l, owners rotated (rank + w): its group raw and (ADD_RESIDUAL) its
  // scale, every owner's in flight before the first is decoded, then each stored at its place.
  // The next call's first sync keeps an owner from overwriting its partials while they are read.
  for (int l = blockIdx.x; l < slice_m; l += grid) {
    [[maybe_unused]] float sc[WORLD][1][1][1];
    if constexpr (ADD_RESIDUAL)
      for (int v = 0; v < WORLD; ++v)
        recv<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
            scales_ptr_array, scales_ptr_array_size, (rank + v) % WORLD, 0, 0, scales_stride_m, 1,
            1, scales_size_m, 1, scales_size_m, 1, sc[v], l, 0, 1);
    uint32_t raw[WORLD][1][kInt4GroupPacks][2];
    for (int v = 0; v < WORLD; ++v)
      recv<WORLD, 1, THREADS_PER_BLOCK, Quantization::int4, Layout::group_run>(
          partials_ptr_array, partials_ptr_array_size, (rank + v) % WORLD, 0, 0,
          partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n, 1,
          (l + 1) * n, raw[v], 0, l * n, 1);
#pragma unroll
    for (int v = 0; v < WORLD; ++v) {
      const int row = (rank + v) % WORLD * slice_m + l;
      if (row >= input_size_m) continue;  // a late owner's short slice
      DTYPE row_got[1][kInt4GroupPacks][kPackElems];
#pragma unroll
      for (int k = 0; k < kInt4GroupPacks; ++k)
        dequantize(raw[v][0][k][0], __builtin_bit_cast(float, raw[v][0][k][1]), row_got[0][k]);
      if constexpr (ADD_RESIDUAL) {
        store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::group_run>(
            row_got, &residual_output_ptr, 1, 0, 0, 0, residual_output_stride_m,
            residual_output_stride_n, 1, input_size_m, n, 1, n, row, 0, 1);
        float got_f[1][kInt4GroupPacks][kPackElems];
#pragma unroll
        for (int k = 0; k < kInt4GroupPacks; ++k)
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) got_f[0][k][j] = static_cast<float>(row_got[0][k][j]);
        float scale[1] = {sc[v][0][0][0]};
        DTYPE normed[1][kInt4GroupPacks][kPackElems];
        rms_norm<THREADS_PER_BLOCK>(got_f, scale, w, normed, eps, inv_hidden, n, 0,
                                    NormPart::apply, true, Layout::group_run);
        store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::group_run>(
            normed, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, input_size_m, n,
            1, n, row, 0, 1);
      } else {
        store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::group_run>(
            row_got, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, input_size_m, n,
            1, n, row, 0, 1);
      }
    }
  }
}


// THE BODY THE KERNELS CALL: QUANTIZATION's.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD,
          int TILE_N, int THREADS_PER_BLOCK, int QUANTIZATION, bool ADD_RESIDUAL>
DINLINE void all_reduce_pull_two_shot_add_rms_norm_body(
    const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
    int64_t input_stride_n, int input_size_m, int input_size_n,
    const DTYPE* const* __restrict__ input_ptr_array, int input_ptr_array_size,
    int64_t input_ptr_array_stride_m, int64_t input_ptr_array_stride_n,
    const DTYPE* residual_ptr, int64_t residual_stride_m,
    int64_t residual_stride_n, const WEIGHT_DTYPE* weight_ptr,
    int64_t weight_stride_n, DTYPE* output_ptr, int64_t output_stride_m,
    int64_t output_stride_n, DTYPE* residual_output_ptr,
    int64_t residual_output_stride_m, int64_t residual_output_stride_n,
    char* const* partials_ptr_array, int partials_ptr_array_size, int64_t partials_stride_m,
    int64_t partials_stride_n, int partials_size_m, int partials_size_n,
    float* const* scales_ptr_array, int scales_ptr_array_size, int64_t scales_stride_m,
    int scales_size_m, char* const* staging_ptr_array, int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
    int status_size_field, int rank, uint64_t timeout_ticks, float eps) {
  if constexpr (QUANTIZATION == Quantization::none)
    all_reduce_pull_two_shot_add_rms_norm_exact_body<DTYPE, WEIGHT_DTYPE, WORLD, TILE_N, THREADS_PER_BLOCK, ADD_RESIDUAL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m, input_ptr_array_stride_n,
        residual_ptr, residual_stride_m, residual_stride_n, weight_ptr, weight_stride_n,
        output_ptr, output_stride_m, output_stride_n, residual_output_ptr,
        residual_output_stride_m, residual_output_stride_n, partials_ptr_array,
        partials_ptr_array_size, partials_stride_m, partials_stride_n, partials_size_m,
        partials_size_n, scales_ptr_array, scales_ptr_array_size, scales_stride_m, scales_size_m,
        staging_ptr_array, staging_ptr_array_size, staging_stride_m, staging_stride_n,
        staging_size_m, staging_size_n, signals_ptr_array, signals_ptr_array_size,
        signals_stride_parity, signals_stride_block, signals_stride_writer, signals_size_parity,
        signals_size_block, signals_size_writer, status_ptr, status_stride_field,
        status_size_field, rank, timeout_ticks, eps);
  else
    all_reduce_pull_two_shot_add_rms_norm_int4_body<DTYPE, WEIGHT_DTYPE, WORLD, TILE_N, THREADS_PER_BLOCK, ADD_RESIDUAL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        residual_ptr, residual_stride_m, residual_stride_n, weight_ptr, weight_stride_n,
        output_ptr, output_stride_m, output_stride_n, residual_output_ptr,
        residual_output_stride_m, residual_output_stride_n, partials_ptr_array,
        partials_ptr_array_size, partials_stride_m, partials_stride_n, partials_size_m,
        partials_size_n, scales_ptr_array, scales_ptr_array_size, scales_stride_m, scales_size_m,
        staging_ptr_array, staging_ptr_array_size, staging_stride_m, staging_stride_n,
        staging_size_m, staging_size_n, signals_ptr_array, signals_ptr_array_size,
        signals_stride_parity, signals_stride_block, signals_stride_writer, signals_size_parity,
        signals_size_block, signals_size_writer, status_ptr, status_stride_field,
        status_size_field, rank, timeout_ticks, eps);
}

// THE KERNELS, one per op, both the body above.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU,
          int QUANTIZATION>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot_rms_norm(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const DTYPE* const* __restrict__ input_ptr_array, int input_ptr_array_size,
    int64_t input_ptr_array_stride_m, int64_t input_ptr_array_stride_n,
        const WEIGHT_DTYPE* __restrict__ weight_ptr, int64_t weight_stride_n,
        int weight_size_n, DTYPE* __restrict__ output_ptr,
        int64_t output_stride_m, int64_t output_stride_n, int output_size_m,
        int output_size_n, char* const* __restrict__ partials_ptr_array, int partials_ptr_array_size,
        int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
        int partials_size_n,
        char* const* __restrict__ staging_ptr_array, int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks, float eps) {
  all_reduce_pull_two_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, TILE_N, THREADS_PER_BLOCK, QUANTIZATION, false>(
      input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
      input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m, input_ptr_array_stride_n,
      nullptr, 0, 0, weight_ptr, weight_stride_n, output_ptr, output_stride_m,
      output_stride_n, nullptr, 0, 0, partials_ptr_array, partials_ptr_array_size, partials_stride_m,
      partials_stride_n, partials_size_m, partials_size_n, nullptr, 0, 0, 0, staging_ptr_array,
      staging_ptr_array_size, staging_stride_m, staging_stride_n, staging_size_m,
      staging_size_n, signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
      signals_stride_block, signals_stride_writer, signals_size_parity,
      signals_size_block, signals_size_writer, status_ptr, status_stride_field,
      status_size_field, rank, timeout_ticks, eps);
}

template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU,
          int QUANTIZATION>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot_add_rms_norm(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const DTYPE* const* __restrict__ input_ptr_array, int input_ptr_array_size,
    int64_t input_ptr_array_stride_m, int64_t input_ptr_array_stride_n,
        const DTYPE* __restrict__ residual_ptr, int64_t residual_stride_m,
        int64_t residual_stride_n, int residual_size_m, int residual_size_n,
        const WEIGHT_DTYPE* __restrict__ weight_ptr, int64_t weight_stride_n,
        int weight_size_n, DTYPE* __restrict__ output_ptr,
        int64_t output_stride_m, int64_t output_stride_n, int output_size_m,
        int output_size_n, DTYPE* __restrict__ residual_output_ptr,
        int64_t residual_output_stride_m, int64_t residual_output_stride_n,
        int residual_output_size_m, int residual_output_size_n,
        char* const* __restrict__ partials_ptr_array, int partials_ptr_array_size, int64_t partials_stride_m,
        int64_t partials_stride_n, int partials_size_m, int partials_size_n,
        float* const* __restrict__ scales_ptr_array, int scales_ptr_array_size,
        int64_t scales_stride_m, int scales_size_m,
        char* const* __restrict__ staging_ptr_array, int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks, float eps) {
  all_reduce_pull_two_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, TILE_N, THREADS_PER_BLOCK, QUANTIZATION, true>(
      input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
      input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m, input_ptr_array_stride_n,
      residual_ptr, residual_stride_m, residual_stride_n, weight_ptr,
      weight_stride_n, output_ptr, output_stride_m, output_stride_n,
      residual_output_ptr, residual_output_stride_m, residual_output_stride_n,
      partials_ptr_array, partials_ptr_array_size, partials_stride_m,
      partials_stride_n, partials_size_m, partials_size_n, scales_ptr_array,
      scales_ptr_array_size, scales_stride_m, scales_size_m, staging_ptr_array,
      staging_ptr_array_size, staging_stride_m, staging_stride_n, staging_size_m,
      staging_size_n, signals_ptr_array, signals_ptr_array_size,
      signals_stride_parity, signals_stride_block, signals_stride_writer,
      signals_size_parity, signals_size_block, signals_size_writer, status_ptr,
      status_stride_field, status_size_field, rank, timeout_ticks, eps);
}

// =============================================================================
// all_reduce_push_two_shot_add_rms_norm
//
// Two-shot all-reduce (the reduce-scatter pulled, the all-gather pushed) then
// RMSNorm
// (`all_reduce_push_two_shot_rms_norm`), and all-reduce then add then RMSNorm
// (`all_reduce_push_two_shot_add_rms_norm`): one body, a kernel per op, so a
// trace names the op that ran.
// =============================================================================

// THE SLICE IS COLUMNS, as in aiter's fused two-stage kernel: each rank owns
// columns [rank * S, rank * S + S) of EVERY row (S = n / ngpus, rounded up to
// whole packs). It sums its columns of a row over the ranks and pushes the sums
// into every rank's partials, so after the sync each rank holds the whole
// all-reduced tensor locally and norms its rows itself: no remote gather, and
// the norm repeated on every rank instead of sent. Every rank computes the same
// bytes (the same sums in the same order), so every rank holds the same result.
// ROW q BELONGS TO BLOCK q % gridDim.x IN BOTH PHASES: after the sync a block
// may read only what the same block on a peer wrote. The buffers' places are
// counted as the one-shot's are (its head says why). QUANTIZATION: what
// crosses the links (common.cuh's QUANTIZATION): with int4 the inputs are read
// from every rank's staged values and scales, and the sums pushed to and read
// from the partials' INT4 payload.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD,
          int TILE_N, int THREADS_PER_BLOCK, int QUANTIZATION, bool ADD_RESIDUAL>
DINLINE void all_reduce_push_two_shot_add_rms_norm_body(
    const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
    int64_t input_stride_n, int input_size_m, int input_size_n,
    const DTYPE* const* __restrict__ input_ptr_array, int input_ptr_array_size,
    int64_t input_ptr_array_stride_m, int64_t input_ptr_array_stride_n,
    const DTYPE* residual_ptr, int64_t residual_stride_m,
    int64_t residual_stride_n, const WEIGHT_DTYPE* weight_ptr,
    int64_t weight_stride_n, DTYPE* output_ptr, int64_t output_stride_m,
    int64_t output_stride_n, DTYPE* residual_output_ptr,
    int64_t residual_output_stride_m, int64_t residual_output_stride_n,
    char* const* partials_ptr_array, int partials_ptr_array_size, int64_t partials_stride_m,
    int64_t partials_stride_n, int partials_size_m, int partials_size_n,
    char* const* staging_ptr_array, int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
    int status_size_field, int rank, uint64_t timeout_ticks, float eps) {
  constexpr int kPackElems = traits<DTYPE>::N;
  constexpr int kPacks = TILE_N / (kPackElems * THREADS_PER_BLOCK);
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);
  // 1-3. none: nothing staged, every rank's input read in place where the input table says.
  // int4: this rank's input QUANTIZED into its staging's INT4 payload, the rows this block's twin
  // on every rank reads. Then every peer launched; this rank's columns of this block's rows,
  // summed over the ranks in rank order and pushed to every rank (itself too), at their place in
  // the tensor, A WAVE A ROW (reduce_scatter.cuh says how wide, and why); every rank's sums in
  // this rank's partials.
  // THE FIRST BARRIER: exact, UNORDERED (started, as the plain pulls'): nothing written before it,
  // every input read in place after it (written before the peers' launches), and the pushes into a
  // peer's partials after it land where that peer finished reading in its last launch, ended before
  // this one started. int4 stages this rank's rows before it, so it stays ordered (launched).
  reduce_scatter<WORLD, 1, TILE_N, THREADS_PER_BLOCK, QUANTIZATION, true, Walk::rows, Accumulate::registers, Direction::push, QUANTIZATION == Quantization::none ? Until::started : Until::launched>(
      input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n, input_ptr_array,
      input_ptr_array_size, input_ptr_array_stride_m, input_ptr_array_stride_n, staging_ptr_array,
      staging_ptr_array_size, staging_stride_m, staging_stride_n, staging_size_m,
      staging_size_n, partials_ptr_array, partials_ptr_array_size, partials_stride_m,
      partials_stride_n, partials_size_m, partials_size_n, signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block, signals_size_writer,
      status_ptr, status_stride_field, status_size_field, 0, rank, timeout_ticks);

  // 4. This block's rows out of this rank's partials: (ADD_RESIDUAL) add the
  // residual, then RMSNorm,
  //    rounding as the reference does (the one-shot kernel spells it out). The
  //    next call's first sync keeps a peer from pushing into this partials while
  //    it is read (a peer's next kernel starts only once this one has
  //    finished).
  // THE PARTIALS, this rank's: with int4 their INT4 payload, worked out from their own sizes where
  // load and store read and write it.
  for (int row = blockIdx.x; row < input_size_m; row += gridDim.x) {
    // Every load of the row before any store, in flight together: the
    // partials's, the residual's and the weight's (one-shot's).
    DTYPE own[1][kPacks][kPackElems], res[1][kPacks][kPackElems];
    load<WORLD, 1, THREADS_PER_BLOCK, QUANTIZATION, Layout::tile>(partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n, input_size_m, input_size_n, own, row, 0, 1);
    if constexpr (ADD_RESIDUAL)
      load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&residual_ptr, 1, 0, 0, 0, residual_stride_m, residual_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, res, row, 0, 1);
    WEIGHT_DTYPE w[1][kPacks][kPackElems];
    load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&weight_ptr, 1, 0, 0, 0, 0, weight_stride_n, 1, 1, input_size_n, 1, input_size_n, w, 0, 0, 1);
    // The owner's sum in fp32; ADD_RESIDUAL adds the residual in fp32, the new residual that sum
    // rounded to DTYPE (as the one-shot kernel's).
    float s[1][kPacks][kPackElems];
    [[maybe_unused]] DTYPE added[1][kPacks][kPackElems];
#pragma unroll
    for (int k = 0; k < kPacks; ++k)
#pragma unroll
      for (int j = 0; j < kPackElems; ++j) {
        s[0][k][j] = static_cast<float>(own[0][k][j]);
        if constexpr (ADD_RESIDUAL) {
          s[0][k][j] += static_cast<float>(res[0][k][j]);
          added[0][k][j] = static_cast<DTYPE>(s[0][k][j]);
        }
      }
    if constexpr (ADD_RESIDUAL) {
      store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(added, &residual_output_ptr, 1, 0, 0, 0, residual_output_stride_m, residual_output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, row, 0, 1);
    }
    // out = T(W(W(s * scale) * float(w))), as the reference rounds (rms_norm.cuh)
    float scale[1];
    DTYPE normed[1][kPacks][kPackElems];
    rms_norm<THREADS_PER_BLOCK>(s, scale, w, normed, eps, inv_hidden,
                                                     input_size_n, 0, NormPart::row, true, Layout::tile);
    store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(normed, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, row, 0, 1);
  }
}

// THE KERNELS, one per op, both the body above.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU,
          int QUANTIZATION>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_push_two_shot_rms_norm(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const DTYPE* const* __restrict__ input_ptr_array, int input_ptr_array_size,
    int64_t input_ptr_array_stride_m, int64_t input_ptr_array_stride_n,
        const WEIGHT_DTYPE* __restrict__ weight_ptr, int64_t weight_stride_n,
        int weight_size_n, DTYPE* __restrict__ output_ptr,
        int64_t output_stride_m, int64_t output_stride_n, int output_size_m,
        int output_size_n, char* const* __restrict__ partials_ptr_array, int partials_ptr_array_size,
        int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
        int partials_size_n,
        char* const* __restrict__ staging_ptr_array, int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks, float eps) {
  all_reduce_push_two_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, TILE_N, THREADS_PER_BLOCK, QUANTIZATION, false>(
      input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
      input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m, input_ptr_array_stride_n,
      nullptr, 0, 0, weight_ptr, weight_stride_n, output_ptr, output_stride_m,
      output_stride_n, nullptr, 0, 0, partials_ptr_array, partials_ptr_array_size, partials_stride_m,
      partials_stride_n, partials_size_m, partials_size_n, staging_ptr_array,
      staging_ptr_array_size, staging_stride_m, staging_stride_n, staging_size_m,
      staging_size_n, signals_ptr_array, signals_ptr_array_size,
      signals_stride_parity, signals_stride_block, signals_stride_writer,
      signals_size_parity, signals_size_block, signals_size_writer, status_ptr,
      status_stride_field, status_size_field, rank, timeout_ticks, eps);
}

template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU,
          int QUANTIZATION>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_push_two_shot_add_rms_norm(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const DTYPE* const* __restrict__ input_ptr_array, int input_ptr_array_size,
    int64_t input_ptr_array_stride_m, int64_t input_ptr_array_stride_n,
        const DTYPE* __restrict__ residual_ptr, int64_t residual_stride_m,
        int64_t residual_stride_n, int residual_size_m, int residual_size_n,
        const WEIGHT_DTYPE* __restrict__ weight_ptr, int64_t weight_stride_n,
        int weight_size_n, DTYPE* __restrict__ output_ptr,
        int64_t output_stride_m, int64_t output_stride_n, int output_size_m,
        int output_size_n, DTYPE* __restrict__ residual_output_ptr,
        int64_t residual_output_stride_m, int64_t residual_output_stride_n,
        int residual_output_size_m, int residual_output_size_n,
        char* const* __restrict__ partials_ptr_array, int partials_ptr_array_size, int64_t partials_stride_m,
        int64_t partials_stride_n, int partials_size_m, int partials_size_n,
        // The pull twin's row scales (their kernel type is one): unused here, never touched.
        float* const* __restrict__ scales_ptr_array, int scales_ptr_array_size,
        int64_t scales_stride_m, int scales_size_m,
        char* const* __restrict__ staging_ptr_array, int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks, float eps) {
  all_reduce_push_two_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, TILE_N, THREADS_PER_BLOCK, QUANTIZATION, true>(
      input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
      input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m, input_ptr_array_stride_n,
      residual_ptr, residual_stride_m, residual_stride_n, weight_ptr,
      weight_stride_n, output_ptr, output_stride_m, output_stride_n,
      residual_output_ptr, residual_output_stride_m, residual_output_stride_n,
      partials_ptr_array, partials_ptr_array_size, partials_stride_m,
      partials_stride_n, partials_size_m, partials_size_n, staging_ptr_array,
      staging_ptr_array_size, staging_stride_m, staging_stride_n, staging_size_m,
      staging_size_n, signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks, eps);
}

// =============================================================================
// rms_norm_rows, add_rms_norm_rows
//
// OUR SEPARATE NORM KERNELS, THE SPLITS' SECOND (STANDARDS 3.2, *We win at the
// op level*): RMSNorm (and fused_add_rms_norm) on rows already all-reduced. The
// norm op's split (select.cuh, Template::all_reduce_pull_split_rms_norm and
// _add_rms_norm) runs the plain all-reduce into this rank's local sums
// (Handle::sums), then this kernel on them, on its own grid: nothing over the
// peers. Every row's math and rounding the one-shot's (above): the sum `input`
// is the all-reduce's, rounded to DTYPE as theirs is, s = float(sum)
// (+ float(residual), residual_out = DTYPE(s)), out = DTYPE(W(W(s * scale) *
// float(w))) (rms_norm.cuh).
// =============================================================================

// A TILE OF TILE_M ROWS a work item, the grid striding over them; the weight
// loaded once, every tile's. Each kernel written out whole (STANDARDS 3.2: an op
// file has no shared body).
template <typename DTYPE, typename WEIGHT_DTYPE, int TILE_M, int TILE_N, int THREADS_PER_BLOCK,
          int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    rms_norm_rows(const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
                  int64_t input_stride_n, int input_size_m, int input_size_n,
                  const WEIGHT_DTYPE* __restrict__ weight_ptr, int64_t weight_stride_n,
                  int weight_size_n, DTYPE* __restrict__ output_ptr, int64_t output_stride_m,
                  int64_t output_stride_n, int output_size_m, int output_size_n, float eps) {
  constexpr int kPackElems = traits<DTYPE>::N;
  constexpr int kPacks = TILE_N / (kPackElems * THREADS_PER_BLOCK);
  const int m = input_size_m, n = input_size_n;
  const float inv_hidden = 1.0f / static_cast<float>(n);
  WEIGHT_DTYPE w[1][kPacks][kPackElems];
  load<1, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
      &weight_ptr, 1, 0, 0, 0, 0, weight_stride_n, 1, 1, n, 1, n, w, 0, 0, 1);
  const int tiles = (m + TILE_M - 1) / TILE_M;
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int offs_m = t * TILE_M;
    DTYPE x[TILE_M][kPacks][kPackElems];
    load<1, TILE_M, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
        &input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1, m, n, m, n, x, offs_m, 0, 1);
    // s = float(sum): the sum as the all-reduce landed it, DTYPE.
    float s[TILE_M][kPacks][kPackElems];
#pragma unroll
    for (int i = 0; i < TILE_M; ++i)
#pragma unroll
      for (int k = 0; k < kPacks; ++k)
#pragma unroll
        for (int j = 0; j < kPackElems; ++j) s[i][k][j] = static_cast<float>(x[i][k][j]);
    float scale[TILE_M];
    DTYPE normed[TILE_M][kPacks][kPackElems];
    rms_norm<THREADS_PER_BLOCK>(s, scale, w, normed, eps, inv_hidden, n, 0, NormPart::row, true,
                                Layout::tile);
    store<1, TILE_M, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
        normed, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, m, n, m, n, offs_m,
        0, 1);
  }
}

template <typename DTYPE, typename WEIGHT_DTYPE, int TILE_M, int TILE_N, int THREADS_PER_BLOCK,
          int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    add_rms_norm_rows(const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
                      int64_t input_stride_n, int input_size_m, int input_size_n,
                      const DTYPE* __restrict__ residual_ptr, int64_t residual_stride_m,
                      int64_t residual_stride_n, int residual_size_m, int residual_size_n,
                      const WEIGHT_DTYPE* __restrict__ weight_ptr, int64_t weight_stride_n,
                      int weight_size_n, DTYPE* __restrict__ output_ptr,
                      int64_t output_stride_m, int64_t output_stride_n, int output_size_m,
                      int output_size_n, DTYPE* __restrict__ residual_output_ptr,
                      int64_t residual_output_stride_m, int64_t residual_output_stride_n,
                      int residual_output_size_m, int residual_output_size_n, float eps) {
  constexpr int kPackElems = traits<DTYPE>::N;
  constexpr int kPacks = TILE_N / (kPackElems * THREADS_PER_BLOCK);
  const int m = input_size_m, n = input_size_n;
  const float inv_hidden = 1.0f / static_cast<float>(n);
  WEIGHT_DTYPE w[1][kPacks][kPackElems];
  load<1, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
      &weight_ptr, 1, 0, 0, 0, 0, weight_stride_n, 1, 1, n, 1, n, w, 0, 0, 1);
  const int tiles = (m + TILE_M - 1) / TILE_M;
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int offs_m = t * TILE_M;
    DTYPE x[TILE_M][kPacks][kPackElems], res[TILE_M][kPacks][kPackElems];
    load<1, TILE_M, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
        &input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1, m, n, m, n, x, offs_m, 0, 1);
    load<1, TILE_M, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
        &residual_ptr, 1, 0, 0, 0, residual_stride_m, residual_stride_n, 1, m, n, m, n, res,
        offs_m, 0, 1);
    // s = float(sum) + float(residual) in fp32, the new residual s rounded to DTYPE (the norm
    // reads the fp32 s).
    float s[TILE_M][kPacks][kPackElems];
    DTYPE added[TILE_M][kPacks][kPackElems];
#pragma unroll
    for (int i = 0; i < TILE_M; ++i)
#pragma unroll
      for (int k = 0; k < kPacks; ++k)
#pragma unroll
        for (int j = 0; j < kPackElems; ++j) {
          s[i][k][j] = static_cast<float>(x[i][k][j]) + static_cast<float>(res[i][k][j]);
          added[i][k][j] = static_cast<DTYPE>(s[i][k][j]);
        }
    store<1, TILE_M, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
        added, &residual_output_ptr, 1, 0, 0, 0, residual_output_stride_m,
        residual_output_stride_n, 1, m, n, m, n, offs_m, 0, 1);
    float scale[TILE_M];
    DTYPE normed[TILE_M][kPacks][kPackElems];
    rms_norm<THREADS_PER_BLOCK>(s, scale, w, normed, eps, inv_hidden, n, 0, NormPart::row, true,
                                Layout::tile);
    store<1, TILE_M, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
        normed, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, m, n, m, n, offs_m,
        0, 1);
  }
}

}  // namespace hip_custom_communicator
