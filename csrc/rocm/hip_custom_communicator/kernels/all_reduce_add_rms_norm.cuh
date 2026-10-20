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


// Each op owns its complete kernel body.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU, int QUANTIZATION>
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
  static_assert(QUANTIZATION == Quantization::none, "this kernel supports only exact transfers");
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

  for (int row = blockIdx.x; row < input_size_m; row += gridDim.x) {
    // A ROW WIDER THAN THE TILE (rms_norm.cuh's NormPart::wide): its chunks in two passes, the scale's
    // and the norm's, every chunk's values read again in the second.
    if (input_size_n > TILE_N) {
      const auto values = [&](int c, float (&s)[1][kPacks][kPackElems]) {
        DTYPE peers[WORLD][1][kPacks][kPackElems];
        for (int r = 0; r < WORLD; ++r)
          recv<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(input_ptr_array, input_ptr_array_size, r, 0, 0, input_ptr_array_stride_m, input_ptr_array_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, peers[r], row, c, 1);
#pragma unroll
        for (int k = 0; k < kPacks; ++k)
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) {
            float acc = static_cast<float>(peers[0][0][k][j]);
#pragma unroll
            for (int r = 1; r < WORLD; ++r) acc += static_cast<float>(peers[r][0][k][j]);
            s[0][k][j] = static_cast<float>(static_cast<DTYPE>(acc));
          }
      };
      float scale[1];
      for (int m = 0; m < 1; ++m) scale[m] = 0.0f;
      {
        const auto chunk = values;
#pragma unroll 1
        for (int c = 0; c < input_size_n; c += TILE_N) {
          float s[1][kPacks][kPackElems];
          chunk(c, s);
          WEIGHT_DTYPE wide_w[1][kPacks][kPackElems];
          DTYPE wide_out[1][kPacks][kPackElems];
          rms_norm<THREADS_PER_BLOCK>(s, scale, wide_w, wide_out, eps, inv_hidden, input_size_n, c,
                                      NormPart::wide, true, Layout::tile);
        }
      }
#pragma unroll 1
      for (int c = 0; c < input_size_n; c += TILE_N) {
        float s[1][kPacks][kPackElems];
        values(c, s);
        // THE CHUNK'S VALUES BEFORE ITS WEIGHT: the load hoisted beside them held both.
        asm volatile("" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        WEIGHT_DTYPE wc[1][kPacks][kPackElems];
        load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&weight_ptr, 1, 0, 0, 0, 0, weight_stride_n, 1, 1, input_size_n, 1, input_size_n, wc, 0, c, 1);
        DTYPE normed[1][kPacks][kPackElems];
        rms_norm<THREADS_PER_BLOCK>(s, scale, wc, normed, eps, inv_hidden, input_size_n, c,
                                    NormPart::apply, true, Layout::tile);
        store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(normed, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, row, c, 1);
      }
      continue;
    }
    // Every load of the row before any store, in flight together: the peers',
    // the residual's and the weight's. A load issues where it is written (an
    // address is built at its load), and the weight after the peers' sum was a
    // round trip of its own (+0.34 us at 16 x 3584, 2026-10-04T17-25-35Z).
    DTYPE peers[WORLD][1][kPacks][kPackElems];
    for (int r = 0; r < WORLD; ++r)
      recv<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(input_ptr_array, input_ptr_array_size, r, 0, 0, input_ptr_array_stride_m, input_ptr_array_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, peers[r], row, 0, 1);

    WEIGHT_DTYPE w[1][kPacks][kPackElems];
    load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&weight_ptr, 1, 0, 0, 0, 0, weight_stride_n, 1, 1, input_size_n, 1, input_size_n, w, 0, 0, 1);
    float s[1][kPacks][kPackElems];
#pragma unroll
    for (int k = 0; k < kPacks; ++k)
#pragma unroll
      for (int j = 0; j < kPackElems; ++j) {
        float acc = static_cast<float>(peers[0][0][k][j]);
#pragma unroll
        for (int r = 1; r < WORLD; ++r) acc += static_cast<float>(peers[r][0][k][j]);
        s[0][k][j] = static_cast<float>(static_cast<DTYPE>(acc));

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

template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU, int QUANTIZATION>
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
  static_assert(QUANTIZATION == Quantization::none, "this kernel supports only exact transfers");
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

  for (int row = blockIdx.x; row < input_size_m; row += gridDim.x) {
    // A ROW WIDER THAN THE TILE (rms_norm.cuh's NormPart::wide): its chunks in two passes, the scale's
    // and the norm's, every chunk's values read again in the second.
    if (input_size_n > TILE_N) {
      // The sum and the residual (`added`, the new residual, for the second pass to write).
      const auto values = [&](int c, float (&s)[1][kPacks][kPackElems],
                              DTYPE (&added)[1][kPacks][kPackElems]) {
        DTYPE peers[WORLD][1][kPacks][kPackElems];
        for (int r = 0; r < WORLD; ++r)
          recv<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(input_ptr_array, input_ptr_array_size, r, 0, 0, input_ptr_array_stride_m, input_ptr_array_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, peers[r], row, c, 1);
#pragma unroll
        for (int k = 0; k < kPacks; ++k)
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) {
            float acc = static_cast<float>(peers[0][0][k][j]);
#pragma unroll
            for (int r = 1; r < WORLD; ++r) acc += static_cast<float>(peers[r][0][k][j]);
            s[0][k][j] = static_cast<float>(static_cast<DTYPE>(acc));
          }
        DTYPE res[1][kPacks][kPackElems];
        load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&residual_ptr, 1, 0, 0, 0, residual_stride_m, residual_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, res, row, c, 1);
#pragma unroll
        for (int k = 0; k < kPacks; ++k)
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) {
            s[0][k][j] += static_cast<float>(res[0][k][j]);
            added[0][k][j] = static_cast<DTYPE>(s[0][k][j]);
          }
      };
      const auto scale_values = [&](int c, float (&s)[1][kPacks][kPackElems]) {
        DTYPE added[1][kPacks][kPackElems];
        values(c, s, added);
      };
      float scale[1];
      for (int m = 0; m < 1; ++m) scale[m] = 0.0f;
      {
        const auto chunk = scale_values;
#pragma unroll 1
        for (int c = 0; c < input_size_n; c += TILE_N) {
          float s[1][kPacks][kPackElems];
          chunk(c, s);
          WEIGHT_DTYPE wide_w[1][kPacks][kPackElems];
          DTYPE wide_out[1][kPacks][kPackElems];
          rms_norm<THREADS_PER_BLOCK>(s, scale, wide_w, wide_out, eps, inv_hidden, input_size_n, c,
                                      NormPart::wide, true, Layout::tile);
        }
      }
#pragma unroll 1
      for (int c = 0; c < input_size_n; c += TILE_N) {
        float s[1][kPacks][kPackElems];
        DTYPE added[1][kPacks][kPackElems];
        values(c, s, added);
        // THE CHUNK'S VALUES BEFORE ITS WEIGHT: the load hoisted beside them held both.
        asm volatile("" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(added, &residual_output_ptr, 1, 0, 0, 0, residual_output_stride_m, residual_output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, row, c, 1);
        WEIGHT_DTYPE wc[1][kPacks][kPackElems];
        load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&weight_ptr, 1, 0, 0, 0, 0, weight_stride_n, 1, 1, input_size_n, 1, input_size_n, wc, 0, c, 1);
        DTYPE normed[1][kPacks][kPackElems];
        rms_norm<THREADS_PER_BLOCK>(s, scale, wc, normed, eps, inv_hidden, input_size_n, c,
                                    NormPart::apply, true, Layout::tile);
        store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(normed, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, row, c, 1);
      }
      continue;
    }
    // Every load of the row before any store, in flight together: the peers',
    // the residual's and the weight's. A load issues where it is written (an
    // address is built at its load), and the weight after the peers' sum was a
    // round trip of its own (+0.34 us at 16 x 3584, 2026-10-04T17-25-35Z).
    DTYPE peers[WORLD][1][kPacks][kPackElems];
    for (int r = 0; r < WORLD; ++r)
      recv<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(input_ptr_array, input_ptr_array_size, r, 0, 0, input_ptr_array_stride_m, input_ptr_array_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, peers[r], row, 0, 1);
    DTYPE res[1][kPacks][kPackElems];
    load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&residual_ptr, 1, 0, 0, 0, residual_stride_m, residual_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, res, row, 0, 1);
    WEIGHT_DTYPE w[1][kPacks][kPackElems];
    load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&weight_ptr, 1, 0, 0, 0, 0, weight_stride_n, 1, 1, input_size_n, 1, input_size_n, w, 0, 0, 1);
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
        {
          s[0][k][j] += static_cast<float>(res[0][k][j]);
          added[0][k][j] = static_cast<DTYPE>(s[0][k][j]);
        }
      }
    {
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

// =============================================================================
// all_reduce_pull_two_shot_add_rms_norm
//
// Two-shot pull all-reduce then RMSNorm (`all_reduce_pull_two_shot_rms_norm`),
// and all-reduce then add then RMSNorm
// (`all_reduce_pull_two_shot_add_rms_norm`): a body per kernel, so a
// trace names the op that ran. THE SLICE IS ROWS, for large inputs: the
// column-slice kernel (all_reduce_push_two_shot_*) wins small ones and loses at
// prefill.
// =============================================================================



// THE BODY THE KERNELS CALL: QUANTIZATION's.

// Each op owns its complete kernel body.
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
  static_assert(QUANTIZATION == Quantization::none || QUANTIZATION == Quantization::int4,
                "unsupported quantization");
  if constexpr (QUANTIZATION == Quantization::none) {
    constexpr int kPackElems = traits<DTYPE>::N;
    constexpr int kPacks = TILE_N / (kPackElems * THREADS_PER_BLOCK);
    const float inv_hidden = 1.0f / static_cast<float>(input_size_n);
    const int slice_m = (input_size_m + WORLD - 1) / WORLD;
    constexpr int kScaleSlot = 0;
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

      // ONLY A ROW THAT EXISTS: issued here, never hoisted, so the block-uniform
      // branch costs nothing, where a clamped unconditional load re-read the last
      // row (a block's whole round trip again; at 256 tokens every block has one
      // row: 2026-10-01T01-07-56Z).
      if (row + static_cast<int>(gridDim.x) < last)
        load_peers(row + gridDim.x, next);
      // The sum and the residual, rounding as the reference does (see the one-shot kernel):
      float s[1][kPacks][kPackElems];
#pragma unroll
      for (int k = 0; k < kPacks; ++k)
#pragma unroll
        for (int j = 0; j < kPackElems; ++j) {
          float acc = static_cast<float>(cur[0][0][k][j]);
#pragma unroll
          for (int r = 1; r < WORLD; ++r) acc += static_cast<float>(cur[r][0][k][j]);
          s[0][k][j] = static_cast<float>(static_cast<DTYPE>(acc));

        }

      float scale[1];
      DTYPE normed[1][kPacks][kPackElems];

      // out = T(W(W(s * scale) * float(w))), as the reference rounds (rms_norm.cuh)
      rms_norm<THREADS_PER_BLOCK>(s, scale, w, normed, eps, inv_hidden,
                                                       input_size_n, 0, NormPart::row, true, Layout::tile);
      store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(normed, partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n, input_size_m, input_size_n, row - first, 0, 1);
    };
    // PING-PONG: two buffers that trade roles each row, so no row copies its
    // packs into the other (a copy cost 32 moves a row at one pack a thread: ISA
    // 2026-10-01T00-58-37Z).
    // A ROW WIDER THAN THE TILE (rms_norm.cuh's NormPart::wide): its chunks in two passes, every
    // peer's read again in the second; no ping-pong.
    if (input_size_n > TILE_N) {
      for (int row = first + blockIdx.x; row < last; row += gridDim.x) {
        const auto values = [&](int c, float (&s)[1][kPacks][kPackElems]) {
          DTYPE peers[WORLD][1][kPacks][kPackElems];
          for (int r = 0; r < WORLD; ++r)
            recv<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(input_ptr_array, input_ptr_array_size, r, 0, 0, input_ptr_array_stride_m, input_ptr_array_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, peers[r], row, c, 1);
#pragma unroll
          for (int k = 0; k < kPacks; ++k)
#pragma unroll
            for (int j = 0; j < kPackElems; ++j) {
              float acc = static_cast<float>(peers[0][0][k][j]);
#pragma unroll
              for (int r = 1; r < WORLD; ++r) acc += static_cast<float>(peers[r][0][k][j]);
              s[0][k][j] = static_cast<float>(static_cast<DTYPE>(acc));
            }
        };
        float scale[1];
        for (int m = 0; m < 1; ++m) scale[m] = 0.0f;
        {
          const auto chunk = values;
#pragma unroll 1
          for (int c = 0; c < input_size_n; c += TILE_N) {
            float s[1][kPacks][kPackElems];
            chunk(c, s);
            WEIGHT_DTYPE wide_w[1][kPacks][kPackElems];
            DTYPE wide_out[1][kPacks][kPackElems];
            rms_norm<THREADS_PER_BLOCK>(s, scale, wide_w, wide_out, eps, inv_hidden, input_size_n, c,
                                        NormPart::wide, true, Layout::tile);
          }
        }
#pragma unroll 1
        for (int c = 0; c < input_size_n; c += TILE_N) {
          float s[1][kPacks][kPackElems];
          values(c, s);
          // THE CHUNK'S VALUES BEFORE ITS WEIGHT: the load hoisted beside them held both.
          asm volatile("" ::: "memory");
          __builtin_amdgcn_sched_barrier(0);
          WEIGHT_DTYPE wc[1][kPacks][kPackElems];
          load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&weight_ptr, 1, 0, 0, 0, 0, weight_stride_n, 1, 1, input_size_n, 1, input_size_n, wc, 0, c, 1);
          DTYPE normed[1][kPacks][kPackElems];
          rms_norm<THREADS_PER_BLOCK>(s, scale, wc, normed, eps, inv_hidden, input_size_n, c,
                                      NormPart::apply, true, Layout::tile);
          store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(normed, partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n, input_size_m, input_size_n, row - first, c, 1);
        }
      }
    } else {
      DTYPE a[WORLD][1][kPacks][kPackElems], b[WORLD][1][kPacks][kPackElems];
      int row = first + blockIdx.x;
      if (row < last) load_peers(row, a);
      for (; row < last; row += 2 * gridDim.x) {
        one_row(row, a, b);
        if (row + static_cast<int>(gridDim.x) >= last) break;
        one_row(row + gridDim.x, b, a);
      }
    }

    // 3. Every rank's rows are visible to its peers.
    barrier<WORLD>(
        signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks, Until::visible, 1);

    for (int l = blockIdx.x; l < slice_m; l += gridDim.x) {
      const auto step = [&](int c, bool scales) {
        DTYPE got[WORLD][1][1][kPackElems];
        for (int r = 0; r < WORLD; ++r)
          recv<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
              partials_ptr_array, partials_ptr_array_size, r, 0, 0, partials_stride_m, partials_stride_n, 1,
              partials_size_m, partials_size_n, slice_m, row_n, got[r], l, c + kScaleSlot, 1);

#pragma unroll
        for (int r = 0; r < WORLD; ++r) {
          const int row = r * slice_m + l;
          if (row >= input_size_m) continue;
          {
            store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(got[r], &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, row, c, 1);
          }
        }
      };
      constexpr int kStep = THREADS_PER_BLOCK * kPackElems;
      {
        for (int c = 0; c < input_size_n; c += kStep) step(c, false);
      }
    }
  }
  else {
    constexpr int kPackElems = traits<DTYPE>::N;
    static_assert(kPackElems * kInt4GroupPacks == kInt4Group, "a pack is a word of its group");
    static_assert(TILE_N == kInt4Group * THREADS_PER_BLOCK, "a group a thread: the block one row's groups");
    const int n = input_size_n;
    // A PAYLOAD ROW IS WHOLE GROUPS (its last group's packs past n nobody's), so every row's
    // groups start on a group of its slot.
    const int n_pad = (n + kInt4Group - 1) / kInt4Group * kInt4Group;
    const float inv_hidden = 1.0f / static_cast<float>(n);
    const int slice_m = (input_size_m + WORLD - 1) / WORLD;
    const int slice = slice_m * n_pad;  // a rank's rows, flat: its slot of an inbox
    const auto rows_of = [&](int r) { return max(0, min(slice_m, input_size_m - r * slice_m)); };
    const int grid = static_cast<int>(gridDim.x);

    // 1. THE PUSH, owners rotated (rank + w), a row of each a step, TWO OWNERS' ROWS IN FLIGHT (as
    // the flat walk's push: every owner's loads hoisted together are 8 groups of input a thread).
    // Owner r's row l, its groups from column c (a TILE_N chunk: one, unless the row is wider).
    const auto push = [&](int l, int r, int c) {
      DTYPE x[1][kInt4GroupPacks][kPackElems];
      load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::group_run>(
          &input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1, input_size_m, n,
          input_size_m, n, x, r * slice_m + l, c, 1);
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
          staging_stride_n, WORLD, staging_size_m, staging_size_n, 1, l * n_pad + n, 0,
          l * n_pad + c, 1);
    };
    if (n <= TILE_N) {
      for (int l = blockIdx.x; l < slice_m; l += grid) {
#pragma unroll 2
        for (int w = 0; w < WORLD; ++w) {
          const int r = (rank + w) % WORLD;
          if (l >= rows_of(r)) continue;  // a late owner's short slice (every thread alike)
          push(l, r, 0);
        }
      }
    } else {
      // A ROW WIDER THAN THE TILE: its chunks one after another.
      for (int l = blockIdx.x; l < slice_m; l += grid)
        for (int w = 0; w < WORLD; ++w) {
          const int r = (rank + w) % WORLD;
          if (l >= rows_of(r)) continue;
#pragma unroll 1
          for (int c = 0; c < n; c += TILE_N) push(l, r, c);
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
    // THE ROW'S VALUES FROM COLUMN c: every writer's group, from this rank's inbox (slots rotated
    // from this rank's), summed (a wide row's chunk: rms_norm.cuh's NormPart::wide).
    const auto values = [&](int l, int c, float (&s)[1][kInt4GroupPacks][kPackElems]) {
      uint32_t got[WORLD][1][kInt4GroupPacks][2];
      load<WORLD, 1, THREADS_PER_BLOCK, Quantization::int4, Layout::group_run>(
          staging_ptr_array, staging_ptr_array_size, rank, rank, slice, staging_stride_m,
          staging_stride_n, WORLD, staging_size_m, staging_size_n, 1, l * n_pad + n, got, 0,
          l * n_pad + c, 1);
      float sum[1][kInt4GroupPacks][kPackElems];
      dequantize<DTYPE>(got, sum);
#pragma unroll
      for (int k = 0; k < kInt4GroupPacks; ++k)
#pragma unroll
        for (int j = 0; j < kPackElems; ++j)
          s[0][k][j] = static_cast<float>(static_cast<DTYPE>(sum[0][k][j]));
    };
    if (n > TILE_N)
      for (int l = blockIdx.x; l < rows_of(rank); l += grid) {
        float scale[1];
        for (int m = 0; m < 1; ++m) scale[m] = 0.0f;
        {
          const auto chunk = [&](int c, float (&s)[1][kInt4GroupPacks][kPackElems]) { values(l, c, s); };
#pragma unroll 1
          for (int c = 0; c < n; c += TILE_N) {
            float s[1][kInt4GroupPacks][kPackElems];
            chunk(c, s);
            WEIGHT_DTYPE wide_w[1][kInt4GroupPacks][kPackElems];
            DTYPE wide_out[1][kInt4GroupPacks][kPackElems];
            rms_norm<THREADS_PER_BLOCK>(s, scale, wide_w, wide_out, eps, inv_hidden, n, c,
                                        NormPart::wide, true, Layout::group_run);
          }
        }
#pragma unroll 1
        for (int c = 0; c < n; c += TILE_N) {
          float s[1][kInt4GroupPacks][kPackElems];
          values(l, c, s);
          // THE CHUNK'S VALUES BEFORE ITS WEIGHT: the load hoisted beside them held both.
          asm volatile("" ::: "memory");
          __builtin_amdgcn_sched_barrier(0);
          WEIGHT_DTYPE wc[1][kInt4GroupPacks][kPackElems];
          load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::group_run>(
              &weight_ptr, 1, 0, 0, 0, 0, weight_stride_n, 1, 1, n, 1, n, wc, 0, c, 1);
          DTYPE normed[1][kInt4GroupPacks][kPackElems];
          rms_norm<THREADS_PER_BLOCK>(s, scale, wc, normed, eps, inv_hidden, n, c,
                                      NormPart::apply, true, Layout::group_run);
          store<WORLD, 1, THREADS_PER_BLOCK, Quantization::int4, Layout::group_run>(
              normed, partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m,
              partials_stride_n, 1, partials_size_m, partials_size_n, 1, l * n_pad + n, 0,
              l * n_pad + c, 1);
        }
      }
    else for (int l = blockIdx.x; l < rows_of(rank); l += grid) {
      // Every writer's group of the row, from this rank's inbox (slots rotated from this rank's),
      // and the residual's, all in flight before the sum.
      uint32_t got[WORLD][1][kInt4GroupPacks][2];
      load<WORLD, 1, THREADS_PER_BLOCK, Quantization::int4, Layout::group_run>(
          staging_ptr_array, staging_ptr_array_size, rank, rank, slice, staging_stride_m,
          staging_stride_n, WORLD, staging_size_m, staging_size_n, 1, l * n_pad + n, got, 0, l * n_pad, 1);

      float sum[1][kInt4GroupPacks][kPackElems];
      dequantize<DTYPE>(got, sum);
      float s[1][kInt4GroupPacks][kPackElems];
#pragma unroll
      for (int k = 0; k < kInt4GroupPacks; ++k)
#pragma unroll
        for (int j = 0; j < kPackElems; ++j) {
          s[0][k][j] = static_cast<float>(static_cast<DTYPE>(sum[0][k][j]));

        }
      float scale[1];
      DTYPE normed[1][kInt4GroupPacks][kPackElems];
      {
        rms_norm<THREADS_PER_BLOCK>(s, scale, w, normed, eps, inv_hidden, n, 0, NormPart::row,
                                    true, Layout::group_run);
        store<WORLD, 1, THREADS_PER_BLOCK, Quantization::int4, Layout::group_run>(
            normed, partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m,
            partials_stride_n, 1, partials_size_m, partials_size_n, 1, l * n_pad + n, 0, l * n_pad, 1);
      }
    }

    // 3. Every rank's rows are visible to its peers.
    barrier<WORLD>(
        signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks, Until::visible, 1);

    // Every owner's row l from column c (one chunk, unless the row is wider than the tile).
    const auto gather = [&](int l, int c) {
      uint32_t raw[WORLD][1][kInt4GroupPacks][2];
      for (int v = 0; v < WORLD; ++v)
        recv<WORLD, 1, THREADS_PER_BLOCK, Quantization::int4, Layout::group_run>(
            partials_ptr_array, partials_ptr_array_size, (rank + v) % WORLD, 0, 0,
            partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n, 1,
            l * n_pad + n, raw[v], 0, l * n_pad + c, 1);
#pragma unroll
      for (int v = 0; v < WORLD; ++v) {
        const int row = (rank + v) % WORLD * slice_m + l;
        if (row >= input_size_m) continue;  // a late owner's short slice
        DTYPE row_got[1][kInt4GroupPacks][kPackElems];
#pragma unroll
        for (int k = 0; k < kInt4GroupPacks; ++k)
          dequantize(raw[v][0][k][0], __builtin_bit_cast(float, raw[v][0][k][1]), row_got[0][k]);
        {
          store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::group_run>(
              row_got, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, input_size_m, n,
              1, n, row, c, 1);
        }
      }
    };
    if (n <= TILE_N)
      for (int l = blockIdx.x; l < slice_m; l += grid) gather(l, 0);
    else
      for (int l = blockIdx.x; l < slice_m; l += grid)
#pragma unroll 1
        for (int c = 0; c < n; c += TILE_N) gather(l, c);
  }
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
  static_assert(QUANTIZATION == Quantization::none || QUANTIZATION == Quantization::int4,
                "unsupported quantization");
  if constexpr (QUANTIZATION == Quantization::none) {
    constexpr int kPackElems = traits<DTYPE>::N;
    constexpr int kPacks = TILE_N / (kPackElems * THREADS_PER_BLOCK);
    const float inv_hidden = 1.0f / static_cast<float>(input_size_n);
    const int slice_m = (input_size_m + WORLD - 1) / WORLD;
    constexpr int kScaleSlot = kPackElems;
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
          {
            s[0][k][j] += static_cast<float>(res[0][k][j]);
            added[0][k][j] = static_cast<DTYPE>(s[0][k][j]);
          }
        }
      {
        store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(added, partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n, input_size_m, row_n, row - first, kScaleSlot, 1);
      }
      float scale[1];
      DTYPE normed[1][kPacks][kPackElems];
      {
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
    // A ROW WIDER THAN THE TILE (rms_norm.cuh's NormPart::wide): its scale over its chunks, then the
    // new residual written chunk by chunk, every peer's read again; no ping-pong.
    if (input_size_n > TILE_N) {
      for (int row = first + blockIdx.x; row < last; row += gridDim.x) {
        const auto values = [&](int c, float (&s)[1][kPacks][kPackElems],
                                DTYPE (&added)[1][kPacks][kPackElems]) {
          DTYPE peers[WORLD][1][kPacks][kPackElems];
          for (int r = 0; r < WORLD; ++r)
            recv<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(input_ptr_array, input_ptr_array_size, r, 0, 0, input_ptr_array_stride_m, input_ptr_array_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, peers[r], row, c, 1);
#pragma unroll
          for (int k = 0; k < kPacks; ++k)
#pragma unroll
            for (int j = 0; j < kPackElems; ++j) {
              float acc = static_cast<float>(peers[0][0][k][j]);
#pragma unroll
              for (int r = 1; r < WORLD; ++r) acc += static_cast<float>(peers[r][0][k][j]);
              s[0][k][j] = static_cast<float>(static_cast<DTYPE>(acc));
            }
          DTYPE res[1][kPacks][kPackElems];
          load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&residual_ptr, 1, 0, 0, 0, residual_stride_m, residual_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, res, row, c, 1);
#pragma unroll
          for (int k = 0; k < kPacks; ++k)
#pragma unroll
            for (int j = 0; j < kPackElems; ++j) {
              s[0][k][j] += static_cast<float>(res[0][k][j]);
              added[0][k][j] = static_cast<DTYPE>(s[0][k][j]);
            }
        };
        const auto scale_values = [&](int c, float (&s)[1][kPacks][kPackElems]) {
          DTYPE added[1][kPacks][kPackElems];
          values(c, s, added);
        };
        float scale[1];
        for (int m = 0; m < 1; ++m) scale[m] = 0.0f;
        {
          const auto chunk = scale_values;
#pragma unroll 1
          for (int c = 0; c < input_size_n; c += TILE_N) {
            float s[1][kPacks][kPackElems];
            chunk(c, s);
            WEIGHT_DTYPE wide_w[1][kPacks][kPackElems];
            DTYPE wide_out[1][kPacks][kPackElems];
            rms_norm<THREADS_PER_BLOCK>(s, scale, wide_w, wide_out, eps, inv_hidden, input_size_n, c,
                                        NormPart::wide, true, Layout::tile);
          }
        }
#pragma unroll 1
        for (int c = 0; c < input_size_n; c += TILE_N) {
          float s[1][kPacks][kPackElems];
          DTYPE added[1][kPacks][kPackElems];
          values(c, s, added);
          // THE CHUNK'S VALUES BEFORE ITS WEIGHT: the load hoisted beside them held both.
          asm volatile("" ::: "memory");
          __builtin_amdgcn_sched_barrier(0);
          store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(added, partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n, input_size_m, row_n, row - first, kScaleSlot + c, 1);
        }
        float scale_row[1][1][1] = {{{scale[0]}}};
        store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
            scale_row, partials_ptr_array, partials_ptr_array_size, rank, 0, 0, scale_stride_m, 1, 1,
            partials_size_m, scale_stride_m, partials_size_m, 1, row - first, 0, 1);
      }
    } else {
      DTYPE a[WORLD][1][kPacks][kPackElems], b[WORLD][1][kPacks][kPackElems];
      int row = first + blockIdx.x;
      if (row < last) load_peers(row, a);
      for (; row < last; row += 2 * gridDim.x) {
        one_row(row, a, b);
        if (row + static_cast<int>(gridDim.x) >= last) break;
        one_row(row + gridDim.x, b, a);
      }
    }

    // 3. Every rank's rows are visible to its peers.
    barrier<WORLD>(
        signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks, Until::visible, 1);

    for (int l = blockIdx.x; l < slice_m; l += gridDim.x) {
      [[maybe_unused]] float sc[WORLD][1][1][1];
      const auto step = [&](int c, bool scales) {
        DTYPE got[WORLD][1][1][kPackElems];
        for (int r = 0; r < WORLD; ++r)
          recv<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
              partials_ptr_array, partials_ptr_array_size, r, 0, 0, partials_stride_m, partials_stride_n, 1,
              partials_size_m, partials_size_n, slice_m, row_n, got[r], l, c + kScaleSlot, 1);
        if (scales)
            for (int r = 0; r < WORLD; ++r)
              recv<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
                  partials_ptr_array, partials_ptr_array_size, r, 0, 0, scale_stride_m, 1, 1,
                  partials_size_m, scale_stride_m, slice_m, 1, sc[r], l, 0, 1);
        WEIGHT_DTYPE w_step[1][1][kPackElems];
        load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&weight_ptr, 1, 0, 0, 0, 0, weight_stride_n, 1, 1, input_size_n, 1, input_size_n, w_step, 0, c, 1);
#pragma unroll
        for (int r = 0; r < WORLD; ++r) {
          const int row = r * slice_m + l;
          if (row >= input_size_m) continue;
          {
            store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(got[r], &residual_output_ptr, 1, 0, 0, 0, residual_output_stride_m, residual_output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, row, c, 1);
            float got_f[1][1][kPackElems];
#pragma unroll
            for (int j = 0; j < kPackElems; ++j) got_f[0][0][j] = static_cast<float>(got[r][0][0][j]);
            float scale[1] = {sc[r][0][0][0]};
            DTYPE normed[1][1][kPackElems];
            rms_norm<THREADS_PER_BLOCK>(got_f, scale, w_step, normed, eps,
                                                               inv_hidden, input_size_n, c, NormPart::apply, true, Layout::tile);
            store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(normed, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, row, c, 1);
          }
        }
      };
      constexpr int kStep = THREADS_PER_BLOCK * kPackElems;
      {
        step(0, true);
        for (int c = kStep; c < input_size_n; c += kStep) step(c, false);
      }
    }
  }
  else {
    constexpr int kPackElems = traits<DTYPE>::N;
    static_assert(kPackElems * kInt4GroupPacks == kInt4Group, "a pack is a word of its group");
    static_assert(TILE_N == kInt4Group * THREADS_PER_BLOCK, "a group a thread: the block one row's groups");
    const int n = input_size_n;
    // A PAYLOAD ROW IS WHOLE GROUPS (its last group's packs past n nobody's), so every row's
    // groups start on a group of its slot.
    const int n_pad = (n + kInt4Group - 1) / kInt4Group * kInt4Group;
    const float inv_hidden = 1.0f / static_cast<float>(n);
    const int slice_m = (input_size_m + WORLD - 1) / WORLD;
    const int slice = slice_m * n_pad;  // a rank's rows, flat: its slot of an inbox
    const auto rows_of = [&](int r) { return max(0, min(slice_m, input_size_m - r * slice_m)); };
    const int grid = static_cast<int>(gridDim.x);

    // 1. THE PUSH, owners rotated (rank + w), a row of each a step, TWO OWNERS' ROWS IN FLIGHT (as
    // the flat walk's push: every owner's loads hoisted together are 8 groups of input a thread).
    // Owner r's row l, its groups from column c (a TILE_N chunk: one, unless the row is wider).
    const auto push = [&](int l, int r, int c) {
      DTYPE x[1][kInt4GroupPacks][kPackElems];
      load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::group_run>(
          &input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1, input_size_m, n,
          input_size_m, n, x, r * slice_m + l, c, 1);
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
          staging_stride_n, WORLD, staging_size_m, staging_size_n, 1, l * n_pad + n, 0,
          l * n_pad + c, 1);
    };
    if (n <= TILE_N) {
      for (int l = blockIdx.x; l < slice_m; l += grid) {
#pragma unroll 2
        for (int w = 0; w < WORLD; ++w) {
          const int r = (rank + w) % WORLD;
          if (l >= rows_of(r)) continue;  // a late owner's short slice (every thread alike)
          push(l, r, 0);
        }
      }
    } else {
      // A ROW WIDER THAN THE TILE: its chunks one after another.
      for (int l = blockIdx.x; l < slice_m; l += grid)
        for (int w = 0; w < WORLD; ++w) {
          const int r = (rank + w) % WORLD;
          if (l >= rows_of(r)) continue;
#pragma unroll 1
          for (int c = 0; c < n; c += TILE_N) push(l, r, c);
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
    // THE ROW'S VALUES FROM COLUMN c: every writer's group, from this rank's inbox (slots rotated
    // from this rank's), summed, and the residual added (`added`, the new residual).
    const auto values = [&](int l, int c, float (&s)[1][kInt4GroupPacks][kPackElems],
                            DTYPE (&added)[1][kInt4GroupPacks][kPackElems]) {
      uint32_t got[WORLD][1][kInt4GroupPacks][2];
      load<WORLD, 1, THREADS_PER_BLOCK, Quantization::int4, Layout::group_run>(
          staging_ptr_array, staging_ptr_array_size, rank, rank, slice, staging_stride_m,
          staging_stride_n, WORLD, staging_size_m, staging_size_n, 1, l * n_pad + n, got, 0,
          l * n_pad + c, 1);
      DTYPE res[1][kInt4GroupPacks][kPackElems];
      load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::group_run>(
          &residual_ptr, 1, 0, 0, 0, residual_stride_m, residual_stride_n, 1, input_size_m, n,
          input_size_m, n, res, rank * slice_m + l, c, 1);
      float sum[1][kInt4GroupPacks][kPackElems];
      dequantize<DTYPE>(got, sum);
#pragma unroll
      for (int k = 0; k < kInt4GroupPacks; ++k)
#pragma unroll
        for (int j = 0; j < kPackElems; ++j) {
          s[0][k][j] = static_cast<float>(static_cast<DTYPE>(sum[0][k][j])) +
                       static_cast<float>(res[0][k][j]);
          added[0][k][j] = static_cast<DTYPE>(s[0][k][j]);
        }
    };
    if (n > TILE_N)
      for (int l = blockIdx.x; l < rows_of(rank); l += grid) {
        // A ROW WIDER THAN THE TILE (rms_norm.cuh's NormPart::wide): its scale over its chunks, then
        // the new residual, chunk by chunk.
        float scale[1];
        for (int m = 0; m < 1; ++m) scale[m] = 0.0f;
        {
          const auto chunk = [&](int c, float (&s)[1][kInt4GroupPacks][kPackElems]) {
                      DTYPE added[1][kInt4GroupPacks][kPackElems];
                      values(l, c, s, added);
                    };
#pragma unroll 1
          for (int c = 0; c < n; c += TILE_N) {
            float s[1][kInt4GroupPacks][kPackElems];
            chunk(c, s);
            WEIGHT_DTYPE wide_w[1][kInt4GroupPacks][kPackElems];
            DTYPE wide_out[1][kInt4GroupPacks][kPackElems];
            rms_norm<THREADS_PER_BLOCK>(s, scale, wide_w, wide_out, eps, inv_hidden, n, c,
                                        NormPart::wide, true, Layout::group_run);
          }
        }
#pragma unroll 1
        for (int c = 0; c < n; c += TILE_N) {
          float s[1][kInt4GroupPacks][kPackElems];
          DTYPE added[1][kInt4GroupPacks][kPackElems];
          values(l, c, s, added);
          // THE CHUNK'S VALUES BEFORE ITS WEIGHT: the load hoisted beside them held both.
          asm volatile("" ::: "memory");
          __builtin_amdgcn_sched_barrier(0);
          store<WORLD, 1, THREADS_PER_BLOCK, Quantization::int4, Layout::group_run>(
              added, partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m,
              partials_stride_n, 1, partials_size_m, partials_size_n, 1, l * n_pad + n, 0,
              l * n_pad + c, 1);
        }
        float scale_row[1][1][1] = {{{scale[0]}}};
        store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
            scale_row, scales_ptr_array, scales_ptr_array_size, rank, 0, 0, scales_stride_m, 1, 1,
            scales_size_m, 1, scales_size_m, 1, l, 0, 1);
      }
    else for (int l = blockIdx.x; l < rows_of(rank); l += grid) {
      const int row = rank * slice_m + l;
      // Every writer's group of the row, from this rank's inbox (slots rotated from this rank's),
      // and the residual's, all in flight before the sum.
      uint32_t got[WORLD][1][kInt4GroupPacks][2];
      load<WORLD, 1, THREADS_PER_BLOCK, Quantization::int4, Layout::group_run>(
          staging_ptr_array, staging_ptr_array_size, rank, rank, slice, staging_stride_m,
          staging_stride_n, WORLD, staging_size_m, staging_size_n, 1, l * n_pad + n, got, 0, l * n_pad, 1);
      DTYPE res[1][kInt4GroupPacks][kPackElems];
      load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::group_run>(
            &residual_ptr, 1, 0, 0, 0, residual_stride_m, residual_stride_n, 1, input_size_m, n,
            input_size_m, n, res, row, 0, 1);
      float sum[1][kInt4GroupPacks][kPackElems];
      dequantize<DTYPE>(got, sum);
      float s[1][kInt4GroupPacks][kPackElems];
      [[maybe_unused]] DTYPE added[1][kInt4GroupPacks][kPackElems];
#pragma unroll
      for (int k = 0; k < kInt4GroupPacks; ++k)
#pragma unroll
        for (int j = 0; j < kPackElems; ++j) {
          s[0][k][j] = static_cast<float>(static_cast<DTYPE>(sum[0][k][j]));
          {
            s[0][k][j] += static_cast<float>(res[0][k][j]);
            added[0][k][j] = static_cast<DTYPE>(s[0][k][j]);
          }
        }
      float scale[1];
      DTYPE normed[1][kInt4GroupPacks][kPackElems];
      {
        // THE NEW RESIDUAL AND ITS SCALE, NOT THE NORMED ROW (the exact body says why).
        store<WORLD, 1, THREADS_PER_BLOCK, Quantization::int4, Layout::group_run>(
            added, partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m,
            partials_stride_n, 1, partials_size_m, partials_size_n, 1, l * n_pad + n, 0, l * n_pad, 1);
        rms_norm<THREADS_PER_BLOCK>(s, scale, w, normed, eps, inv_hidden, n, 0, NormPart::scale,
                                    true, Layout::group_run);
        float scale_row[1][1][1] = {{{scale[0]}}};
        store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
            scale_row, scales_ptr_array, scales_ptr_array_size, rank, 0, 0, scales_stride_m, 1, 1,
            scales_size_m, 1, scales_size_m, 1, l, 0, 1);
      }
    }

    // 3. Every rank's rows are visible to its peers.
    barrier<WORLD>(
        signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks, Until::visible, 1);

    // Every owner's row l from column c (one chunk, unless the row is wider than the tile), the
    // weight's chunk with it.
    const auto gather = [&](int l, int c, const WEIGHT_DTYPE (&w)[1][kInt4GroupPacks][kPackElems]) {
      [[maybe_unused]] float sc[WORLD][1][1][1];
      for (int v = 0; v < WORLD; ++v)
          recv<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
              scales_ptr_array, scales_ptr_array_size, (rank + v) % WORLD, 0, 0, scales_stride_m, 1,
              1, scales_size_m, 1, scales_size_m, 1, sc[v], l, 0, 1);
      uint32_t raw[WORLD][1][kInt4GroupPacks][2];
      for (int v = 0; v < WORLD; ++v)
        recv<WORLD, 1, THREADS_PER_BLOCK, Quantization::int4, Layout::group_run>(
            partials_ptr_array, partials_ptr_array_size, (rank + v) % WORLD, 0, 0,
            partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n, 1,
            l * n_pad + n, raw[v], 0, l * n_pad + c, 1);
#pragma unroll
      for (int v = 0; v < WORLD; ++v) {
        const int row = (rank + v) % WORLD * slice_m + l;
        if (row >= input_size_m) continue;  // a late owner's short slice
        DTYPE row_got[1][kInt4GroupPacks][kPackElems];
#pragma unroll
        for (int k = 0; k < kInt4GroupPacks; ++k)
          dequantize(raw[v][0][k][0], __builtin_bit_cast(float, raw[v][0][k][1]), row_got[0][k]);
        {
          store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::group_run>(
              row_got, &residual_output_ptr, 1, 0, 0, 0, residual_output_stride_m,
              residual_output_stride_n, 1, input_size_m, n, 1, n, row, c, 1);
          float got_f[1][kInt4GroupPacks][kPackElems];
#pragma unroll
          for (int k = 0; k < kInt4GroupPacks; ++k)
#pragma unroll
            for (int j = 0; j < kPackElems; ++j) got_f[0][k][j] = static_cast<float>(row_got[0][k][j]);
          float scale[1] = {sc[v][0][0][0]};
          DTYPE normed[1][kInt4GroupPacks][kPackElems];
          rms_norm<THREADS_PER_BLOCK>(got_f, scale, w, normed, eps, inv_hidden, n, c,
                                      NormPart::apply, true, Layout::group_run);
          store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::group_run>(
              normed, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, input_size_m, n,
              1, n, row, c, 1);
        }
      }
    };
    if (n <= TILE_N)
      for (int l = blockIdx.x; l < slice_m; l += grid) gather(l, 0, w);
    else
      for (int l = blockIdx.x; l < slice_m; l += grid)
#pragma unroll 1
        for (int c = 0; c < n; c += TILE_N) {
          WEIGHT_DTYPE wc[1][kInt4GroupPacks][kPackElems];
          load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::group_run>(
              &weight_ptr, 1, 0, 0, 0, 0, weight_stride_n, 1, 1, n, 1, n, wc, 0, c, 1);
          gather(l, c, wc);
        }
  }
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

// Each op owns its complete kernel body.
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
  static_assert(QUANTIZATION == Quantization::none || QUANTIZATION == Quantization::int4,
                "unsupported quantization");
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

  for (int row = blockIdx.x; row < input_size_m; row += gridDim.x) {
    // A ROW WIDER THAN THE TILE (rms_norm.cuh's NormPart::wide): its chunks of the partials (the whole
    // sum, every rank's columns) in two passes.
    if (input_size_n > TILE_N) {
      const auto values = [&](int c, float (&s)[1][kPacks][kPackElems]) {
        DTYPE own[1][kPacks][kPackElems];
        load<WORLD, 1, THREADS_PER_BLOCK, QUANTIZATION, Layout::tile>(partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n, input_size_m, input_size_n, own, row, c, 1);
#pragma unroll
        for (int k = 0; k < kPacks; ++k)
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) s[0][k][j] = static_cast<float>(own[0][k][j]);
      };
      float scale[1];
      for (int m = 0; m < 1; ++m) scale[m] = 0.0f;
      {
        const auto chunk = values;
#pragma unroll 1
        for (int c = 0; c < input_size_n; c += TILE_N) {
          float s[1][kPacks][kPackElems];
          chunk(c, s);
          WEIGHT_DTYPE wide_w[1][kPacks][kPackElems];
          DTYPE wide_out[1][kPacks][kPackElems];
          rms_norm<THREADS_PER_BLOCK>(s, scale, wide_w, wide_out, eps, inv_hidden, input_size_n, c,
                                      NormPart::wide, true, Layout::tile);
        }
      }
#pragma unroll 1
      for (int c = 0; c < input_size_n; c += TILE_N) {
        float s[1][kPacks][kPackElems];
        values(c, s);
        // THE CHUNK'S VALUES BEFORE ITS WEIGHT: the load hoisted beside them held both.
        asm volatile("" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        WEIGHT_DTYPE wc[1][kPacks][kPackElems];
        load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&weight_ptr, 1, 0, 0, 0, 0, weight_stride_n, 1, 1, input_size_n, 1, input_size_n, wc, 0, c, 1);
        DTYPE normed[1][kPacks][kPackElems];
        rms_norm<THREADS_PER_BLOCK>(s, scale, wc, normed, eps, inv_hidden, input_size_n, c,
                                    NormPart::apply, true, Layout::tile);
        store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(normed, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, row, c, 1);
      }
      continue;
    }
    // Every load of the row before any store, in flight together: the
    // partials's, the residual's and the weight's (one-shot's).
    DTYPE own[1][kPacks][kPackElems];
    load<WORLD, 1, THREADS_PER_BLOCK, QUANTIZATION, Layout::tile>(partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n, input_size_m, input_size_n, own, row, 0, 1);

    WEIGHT_DTYPE w[1][kPacks][kPackElems];
    load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&weight_ptr, 1, 0, 0, 0, 0, weight_stride_n, 1, 1, input_size_n, 1, input_size_n, w, 0, 0, 1);
    float s[1][kPacks][kPackElems];
#pragma unroll
    for (int k = 0; k < kPacks; ++k)
#pragma unroll
      for (int j = 0; j < kPackElems; ++j) {
        s[0][k][j] = static_cast<float>(own[0][k][j]);

      }

    // out = T(W(W(s * scale) * float(w))), as the reference rounds (rms_norm.cuh)
    float scale[1];
    DTYPE normed[1][kPacks][kPackElems];
    rms_norm<THREADS_PER_BLOCK>(s, scale, w, normed, eps, inv_hidden,
                                                     input_size_n, 0, NormPart::row, true, Layout::tile);
    store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(normed, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, row, 0, 1);
  }
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
  static_assert(QUANTIZATION == Quantization::none || QUANTIZATION == Quantization::int4,
                "unsupported quantization");
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

  for (int row = blockIdx.x; row < input_size_m; row += gridDim.x) {
    // A ROW WIDER THAN THE TILE (rms_norm.cuh's NormPart::wide): its chunks of the partials and the
    // residual in two passes, the second writing the new residual.
    if (input_size_n > TILE_N) {
      const auto values = [&](int c, float (&s)[1][kPacks][kPackElems],
                              DTYPE (&added)[1][kPacks][kPackElems]) {
        DTYPE own[1][kPacks][kPackElems];
        load<WORLD, 1, THREADS_PER_BLOCK, QUANTIZATION, Layout::tile>(partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n, input_size_m, input_size_n, own, row, c, 1);
        DTYPE res[1][kPacks][kPackElems];
        load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&residual_ptr, 1, 0, 0, 0, residual_stride_m, residual_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, res, row, c, 1);
#pragma unroll
        for (int k = 0; k < kPacks; ++k)
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) {
            s[0][k][j] = static_cast<float>(own[0][k][j]) + static_cast<float>(res[0][k][j]);
            added[0][k][j] = static_cast<DTYPE>(s[0][k][j]);
          }
      };
      float scale[1];
      for (int m = 0; m < 1; ++m) scale[m] = 0.0f;
      {
        const auto chunk = [&](int c, float (&s)[1][kPacks][kPackElems]) {
                  DTYPE added[1][kPacks][kPackElems];
                  values(c, s, added);
                };
#pragma unroll 1
        for (int c = 0; c < input_size_n; c += TILE_N) {
          float s[1][kPacks][kPackElems];
          chunk(c, s);
          WEIGHT_DTYPE wide_w[1][kPacks][kPackElems];
          DTYPE wide_out[1][kPacks][kPackElems];
          rms_norm<THREADS_PER_BLOCK>(s, scale, wide_w, wide_out, eps, inv_hidden, input_size_n, c,
                                      NormPart::wide, true, Layout::tile);
        }
      }
#pragma unroll 1
      for (int c = 0; c < input_size_n; c += TILE_N) {
        float s[1][kPacks][kPackElems];
        DTYPE added[1][kPacks][kPackElems];
        values(c, s, added);
        // THE CHUNK'S VALUES BEFORE ITS WEIGHT: the load hoisted beside them held both.
        asm volatile("" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(added, &residual_output_ptr, 1, 0, 0, 0, residual_output_stride_m, residual_output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, row, c, 1);
        WEIGHT_DTYPE wc[1][kPacks][kPackElems];
        load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&weight_ptr, 1, 0, 0, 0, 0, weight_stride_n, 1, 1, input_size_n, 1, input_size_n, wc, 0, c, 1);
        DTYPE normed[1][kPacks][kPackElems];
        rms_norm<THREADS_PER_BLOCK>(s, scale, wc, normed, eps, inv_hidden, input_size_n, c,
                                    NormPart::apply, true, Layout::tile);
        store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(normed, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, row, c, 1);
      }
      continue;
    }
    // Every load of the row before any store, in flight together: the
    // partials's, the residual's and the weight's (one-shot's).
    DTYPE own[1][kPacks][kPackElems], res[1][kPacks][kPackElems];
    load<WORLD, 1, THREADS_PER_BLOCK, QUANTIZATION, Layout::tile>(partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n, input_size_m, input_size_n, own, row, 0, 1);
    load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&residual_ptr, 1, 0, 0, 0, residual_stride_m, residual_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, res, row, 0, 1);
    WEIGHT_DTYPE w[1][kPacks][kPackElems];
    load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&weight_ptr, 1, 0, 0, 0, 0, weight_stride_n, 1, 1, input_size_n, 1, input_size_n, w, 0, 0, 1);
    float s[1][kPacks][kPackElems];
    [[maybe_unused]] DTYPE added[1][kPacks][kPackElems];
#pragma unroll
    for (int k = 0; k < kPacks; ++k)
#pragma unroll
      for (int j = 0; j < kPackElems; ++j) {
        s[0][k][j] = static_cast<float>(own[0][k][j]);
        {
          s[0][k][j] += static_cast<float>(res[0][k][j]);
          added[0][k][j] = static_cast<DTYPE>(s[0][k][j]);
        }
      }
    {
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
    // A ROW WIDER THAN THE TILE (rms_norm.cuh's NormPart::wide): the tile's rows' chunks in two passes.
    if (n > TILE_N) {
      const auto values = [&](int c, float (&s)[TILE_M][kPacks][kPackElems]) {
        DTYPE x[TILE_M][kPacks][kPackElems];
        load<1, TILE_M, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
            &input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1, m, n, m, n, x, offs_m, c, 1);
#pragma unroll
        for (int i = 0; i < TILE_M; ++i)
#pragma unroll
          for (int k = 0; k < kPacks; ++k)
#pragma unroll
            for (int j = 0; j < kPackElems; ++j) s[i][k][j] = static_cast<float>(x[i][k][j]);
      };
      float scale[TILE_M];
      for (int m = 0; m < TILE_M; ++m) scale[m] = 0.0f;
      {
        const auto chunk = values;
#pragma unroll 1
        for (int c = 0; c < n; c += TILE_N) {
          float s[TILE_M][kPacks][kPackElems];
          chunk(c, s);
          WEIGHT_DTYPE wide_w[1][kPacks][kPackElems];
          DTYPE wide_out[TILE_M][kPacks][kPackElems];
          rms_norm<THREADS_PER_BLOCK>(s, scale, wide_w, wide_out, eps, inv_hidden, n, c,
                                      NormPart::wide, true, Layout::tile);
        }
      }
#pragma unroll 1
      for (int c = 0; c < n; c += TILE_N) {
        float s[TILE_M][kPacks][kPackElems];
        values(c, s);
        // THE CHUNK'S VALUES BEFORE ITS WEIGHT: the load hoisted beside them held both.
        asm volatile("" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        WEIGHT_DTYPE wc[1][kPacks][kPackElems];
        load<1, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
            &weight_ptr, 1, 0, 0, 0, 0, weight_stride_n, 1, 1, n, 1, n, wc, 0, c, 1);
        DTYPE normed[TILE_M][kPacks][kPackElems];
        rms_norm<THREADS_PER_BLOCK>(s, scale, wc, normed, eps, inv_hidden, n, c, NormPart::apply,
                                    true, Layout::tile);
        store<1, TILE_M, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
            normed, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, m, n, m, n, offs_m,
            c, 1);
      }
      continue;
    }
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
    // A ROW WIDER THAN THE TILE (rms_norm.cuh's NormPart::wide): the tile's rows' chunks in two passes,
    // the second writing the new residual.
    if (n > TILE_N) {
      const auto values = [&](int c, float (&s)[TILE_M][kPacks][kPackElems],
                              DTYPE (&added)[TILE_M][kPacks][kPackElems]) {
        DTYPE x[TILE_M][kPacks][kPackElems];
        load<1, TILE_M, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
            &input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1, m, n, m, n, x, offs_m, c, 1);
        DTYPE res[TILE_M][kPacks][kPackElems];
        load<1, TILE_M, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
            &residual_ptr, 1, 0, 0, 0, residual_stride_m, residual_stride_n, 1, m, n, m, n, res,
            offs_m, c, 1);
#pragma unroll
        for (int i = 0; i < TILE_M; ++i)
#pragma unroll
          for (int k = 0; k < kPacks; ++k)
#pragma unroll
            for (int j = 0; j < kPackElems; ++j) {
              s[i][k][j] = static_cast<float>(x[i][k][j]) + static_cast<float>(res[i][k][j]);
              added[i][k][j] = static_cast<DTYPE>(s[i][k][j]);
            }
      };
      float scale[TILE_M];
      for (int m = 0; m < TILE_M; ++m) scale[m] = 0.0f;
      {
        const auto chunk = [&](int c, float (&s)[TILE_M][kPacks][kPackElems]) {
                  DTYPE added[TILE_M][kPacks][kPackElems];
                  values(c, s, added);
                };
#pragma unroll 1
        for (int c = 0; c < n; c += TILE_N) {
          float s[TILE_M][kPacks][kPackElems];
          chunk(c, s);
          WEIGHT_DTYPE wide_w[1][kPacks][kPackElems];
          DTYPE wide_out[TILE_M][kPacks][kPackElems];
          rms_norm<THREADS_PER_BLOCK>(s, scale, wide_w, wide_out, eps, inv_hidden, n, c,
                                      NormPart::wide, true, Layout::tile);
        }
      }
#pragma unroll 1
      for (int c = 0; c < n; c += TILE_N) {
        float s[TILE_M][kPacks][kPackElems];
        DTYPE added[TILE_M][kPacks][kPackElems];
        values(c, s, added);
        // THE CHUNK'S VALUES BEFORE ITS WEIGHT: the load hoisted beside them held both.
        asm volatile("" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        store<1, TILE_M, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
            added, &residual_output_ptr, 1, 0, 0, 0, residual_output_stride_m,
            residual_output_stride_n, 1, m, n, m, n, offs_m, c, 1);
        WEIGHT_DTYPE wc[1][kPacks][kPackElems];
        load<1, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
            &weight_ptr, 1, 0, 0, 0, 0, weight_stride_n, 1, 1, n, 1, n, wc, 0, c, 1);
        DTYPE normed[TILE_M][kPacks][kPackElems];
        rms_norm<THREADS_PER_BLOCK>(s, scale, wc, normed, eps, inv_hidden, n, c, NormPart::apply,
                                    true, Layout::tile);
        store<1, TILE_M, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(
            normed, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, m, n, m, n, offs_m,
            c, 1);
      }
      continue;
    }
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
