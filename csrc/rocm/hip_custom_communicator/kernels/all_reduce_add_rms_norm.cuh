// SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc.
// All rights reserved.
//
// ALL-REDUCE THEN (ADD THEN) RMSNORM: first the norm on rows, add_rms_norm_rows
// (the split's second kernel, add_rms_norm, is a loop over it); then the fused
// kernel, all_reduce_add_rms_norm, an algorithm a function. Each algorithm is a
// reduction built from the all-reduce's primitives (all_gather.cuh,
// reduce_scatter.cuh) that leaves each block's rows' sums in `output`, then the
// norm on rows over them, in place: one foundation, the communication tuned
// once for every op. The residual is the call's (RESIDUAL).

#pragma once

#include "all_gather.cuh"
#include "common.cuh"
#include "reduce_scatter.cuh"
#include "rms_normalize.cuh"

namespace hip_custom_communicator {

// THE ALL-REDUCE-(ADD-)RMSNORM'S ALGORITHMS: its ALGORITHM, exactly the bodies
// it has. The pulled two-shot slices the sum by rows or by columns (which wins
// is the tune's: the plain all-reduce measured rows 2-7% slower, the earlier
// fused kernel rows faster at prefill).
enum class AllReduceAddRmsNormKernelAlgorithm {
  one_shot_pull,
  two_shot_pull_rows,
  two_shot_pull_columns,
  two_shot_push
};

// =============================================================================
// add_rms_norm_rows: THE NORM ON ROWS, TILE_M rows from row offs_m: s =
// float(input) (with RESIDUAL + float(residual), residual_output = DTYPE(s)),
// output = DTYPE(W(W(s * scale) * float(w))) (rms_normalize.cuh). A row wider
// than TILE_N (CHUNKED) in two passes: the scale over every chunk, then each
// chunk normed. IN PLACE: output may be input itself (the fused kernel's sums,
// the split's), every element written by the thread that read it, after it read
// it; so the rows' pointers are not __restrict__.
// =============================================================================
template <bool RESIDUAL, typename DTYPE, typename WEIGHT_DTYPE, int TILE_M,
          int TILE_N, int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES,
          bool CHUNKED>
DINLINE void add_rms_norm_rows(
    int offs_m, const DTYPE* input_ptr, int64_t input_stride_m,
    int64_t input_stride_n, int input_size_m, int input_size_n,
    const DTYPE* residual_ptr, int64_t residual_stride_m,
    int64_t residual_stride_n, const WEIGHT_DTYPE* weight_ptr,
    int64_t weight_stride_n, DTYPE* output_ptr, int64_t output_stride_m,
    int64_t output_stride_n, DTYPE* residual_output_ptr,
    int64_t residual_output_stride_m, int64_t residual_output_stride_n,
    float eps) {
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
  // A PASS COVERS TILE_N ROUNDED UP TO WHOLE PACKS A THREAD (kSpan columns):
  // the row's columns past the pass masked (rms_normalize's mask, the loads'
  // clip), none skipped.
  constexpr int kPacks = (TILE_N + kPackElems * THREADS_PER_BLOCK - 1) /
                         (kPackElems * THREADS_PER_BLOCK);
  constexpr int kSpan = kPacks * kPackElems * THREADS_PER_BLOCK;
  const int m = input_size_m, n = input_size_n;
  const float inv_hidden = 1.0f / static_cast<float>(n);
  if constexpr (CHUNKED) {
    // 1. THE SCALE, over every chunk of the rows.
    float scale[TILE_M];
    for (int i = 0; i < TILE_M; ++i) scale[i] = 0.0f;
    for (int c = 0; c < n; c += kSpan) {
      DTYPE x[TILE_M][kPacks][kPackElems], res[TILE_M][kPacks][kPackElems];
      load<1, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
          &input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1, m, n, m, n,
          x, offs_m, c, 1);
      if constexpr (RESIDUAL)
        load<1, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
            &residual_ptr, 1, 0, 0, 0, residual_stride_m, residual_stride_n, 1,
            m, n, m, n, res, offs_m, c, 1);
      float s[TILE_M][kPacks][kPackElems];
#pragma unroll
      for (int i = 0; i < TILE_M; ++i)
#pragma unroll
        for (int k = 0; k < kPacks; ++k)
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) {
            s[i][k][j] = static_cast<float>(x[i][k][j]);
            if constexpr (RESIDUAL)
              s[i][k][j] += static_cast<float>(res[i][k][j]);
          }
      WEIGHT_DTYPE unused_w[1][kPacks][kPackElems];
      DTYPE unused_out[TILE_M][kPacks][kPackElems];
      rms_normalize<THREADS_PER_BLOCK, WAVE_SIZE>(
          s, scale, unused_w, unused_out, eps, inv_hidden, n, c, NormPart::wide,
          true, Layout::tile);
    }
    // 2. EACH CHUNK NORMED, the new residual written with it.
    for (int c = 0; c < n; c += kSpan) {
      DTYPE x[TILE_M][kPacks][kPackElems], res[TILE_M][kPacks][kPackElems];
      load<1, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
          &input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1, m, n, m, n,
          x, offs_m, c, 1);
      if constexpr (RESIDUAL)
        load<1, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
            &residual_ptr, 1, 0, 0, 0, residual_stride_m, residual_stride_n, 1,
            m, n, m, n, res, offs_m, c, 1);
      float s[TILE_M][kPacks][kPackElems];
      DTYPE added[TILE_M][kPacks][kPackElems];
#pragma unroll
      for (int i = 0; i < TILE_M; ++i)
#pragma unroll
        for (int k = 0; k < kPacks; ++k)
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) {
            s[i][k][j] = static_cast<float>(x[i][k][j]);
            if constexpr (RESIDUAL)
              s[i][k][j] += static_cast<float>(res[i][k][j]);
            added[i][k][j] = static_cast<DTYPE>(s[i][k][j]);
          }
      if constexpr (RESIDUAL)
        store<1, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
            added, &residual_output_ptr, 1, 0, 0, 0, residual_output_stride_m,
            residual_output_stride_n, 1, m, n, m, n, offs_m, c, 1);
      WEIGHT_DTYPE w[1][kPacks][kPackElems];
      load<1, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
          &weight_ptr, 1, 0, 0, 0, 0, weight_stride_n, 1, 1, n, 1, n, w, 0, c,
          1);
      DTYPE normed[TILE_M][kPacks][kPackElems];
      rms_normalize<THREADS_PER_BLOCK, WAVE_SIZE>(
          s, scale, w, normed, eps, inv_hidden, n, c, NormPart::apply, true,
          Layout::tile);
      store<1, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
          normed, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1,
          m, n, m, n, offs_m, c, 1);
    }
  } else {
    // THE ROWS IN ONE PASS.
    DTYPE x[TILE_M][kPacks][kPackElems], res[TILE_M][kPacks][kPackElems];
    load<1, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
        &input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1, m, n, m, n,
        x, offs_m, 0, 1);
    if constexpr (RESIDUAL)
      load<1, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
          &residual_ptr, 1, 0, 0, 0, residual_stride_m, residual_stride_n, 1, m,
          n, m, n, res, offs_m, 0, 1);
    float s[TILE_M][kPacks][kPackElems];
    DTYPE added[TILE_M][kPacks][kPackElems];
#pragma unroll
    for (int i = 0; i < TILE_M; ++i)
#pragma unroll
      for (int k = 0; k < kPacks; ++k)
#pragma unroll
        for (int j = 0; j < kPackElems; ++j) {
          s[i][k][j] = static_cast<float>(x[i][k][j]);
          if constexpr (RESIDUAL)
            s[i][k][j] += static_cast<float>(res[i][k][j]);
          added[i][k][j] = static_cast<DTYPE>(s[i][k][j]);
        }
    if constexpr (RESIDUAL)
      store<1, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
          added, &residual_output_ptr, 1, 0, 0, 0, residual_output_stride_m,
          residual_output_stride_n, 1, m, n, m, n, offs_m, 0, 1);
    WEIGHT_DTYPE w[1][kPacks][kPackElems];
    load<1, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
        &weight_ptr, 1, 0, 0, 0, 0, weight_stride_n, 1, 1, n, 1, n, w, 0, 0, 1);
    float scale[TILE_M];
    DTYPE normed[TILE_M][kPacks][kPackElems];
    rms_normalize<THREADS_PER_BLOCK, WAVE_SIZE>(s, scale, w, normed, eps,
                                                inv_hidden, n, 0, NormPart::row,
                                                true, Layout::tile);
    store<1, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
        normed, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, m,
        n, m, n, offs_m, 0, 1);
  }
}

// =============================================================================
// add_rms_norm: THE NORM ON ROWS AS ITS OWN KERNEL (STANDARDS 3.2, *We win at
// the op level*), the split's second launch: our all-reduce into a tensor of
// this rank's, then this kernel on it, in place, on its own grid: nothing over
// the peers. A tile of TILE_M rows a work item, the grid striding over them.
// =============================================================================
template <bool RESIDUAL, typename DTYPE, typename WEIGHT_DTYPE, int TILE_M,
          int TILE_N, int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES,
          int WAVES_PER_EU, bool CHUNKED>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    add_rms_norm(const DTYPE* input_ptr, int64_t input_stride_m,
                 int64_t input_stride_n, int input_size_m, int input_size_n,
                 const DTYPE* residual_ptr, int64_t residual_stride_m,
                 int64_t residual_stride_n, int residual_size_m,
                 int residual_size_n,
                 const WEIGHT_DTYPE* __restrict__ weight_ptr,
                 int64_t weight_stride_n, int weight_size_n, DTYPE* output_ptr,
                 int64_t output_stride_m, int64_t output_stride_n,
                 int output_size_m, int output_size_n,
                 DTYPE* residual_output_ptr, int64_t residual_output_stride_m,
                 int64_t residual_output_stride_n, int residual_output_size_m,
                 int residual_output_size_n, float eps) {
  const int tiles = (input_size_m + TILE_M - 1) / TILE_M;
  for (int t = blockIdx.x; t < tiles; t += gridDim.x)
    add_rms_norm_rows<RESIDUAL, DTYPE, WEIGHT_DTYPE, TILE_M, TILE_N,
                      THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, CHUNKED>(
        t * TILE_M, input_ptr, input_stride_m, input_stride_n, input_size_m,
        input_size_n, residual_ptr, residual_stride_m, residual_stride_n,
        weight_ptr, weight_stride_n, output_ptr, output_stride_m,
        output_stride_n, residual_output_ptr, residual_output_stride_m,
        residual_output_stride_n, eps);
}

// =============================================================================
// THE FUSED KERNEL'S ALGORITHMS: each a reduction, then add_rms_norm_rows on
// this block's rows. Row q is block q % gridDim.x's in every walk, so the block
// that norms a row is the block that wrote its sum; within the block a thread
// norms the elements it wrote (the same Layout::tile walk).
// =============================================================================

// THE PULLED ONE-SHOT: every rank's chunk of the row through the input table
// (all_gather's tile_pull, the all-reduce's pulled one-shot), summed in rank
// order in fp32, rounded once, into `output`; then the row normed. Exact alone:
// an INT4 build of it compiles this exact body, which never runs (the host
// refuses an INT4 call of it by name, quantization_not_supported).
template <bool RESIDUAL, typename DTYPE, typename WEIGHT_DTYPE, int WORLD,
          int TILE_N, int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES,
          bool CHUNKED>
DINLINE void all_reduce_add_rms_norm_one_shot_pull(
    int input_size_m, int input_size_n, const DTYPE* const* input_ptr_array,
    int input_ptr_array_size, int64_t input_ptr_array_stride_m,
    int64_t input_ptr_array_stride_n, const DTYPE* residual_ptr,
    int64_t residual_stride_m, int64_t residual_stride_n,
    const WEIGHT_DTYPE* weight_ptr, int64_t weight_stride_n, DTYPE* output_ptr,
    int64_t output_stride_m, int64_t output_stride_n,
    DTYPE* residual_output_ptr, int64_t residual_output_stride_m,
    int64_t residual_output_stride_n, uint32_t* const* signals_ptr_array,
    int signals_ptr_array_size, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    uint32_t* status_ptr, int64_t status_stride_field, int status_size_field,
    int rank, uint64_t timeout_ticks, float eps) {
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
  constexpr int kPacks = (TILE_N + kPackElems * THREADS_PER_BLOCK - 1) /
                         (kPackElems * THREADS_PER_BLOCK);
  constexpr int kSpan = kPacks * kPackElems * THREADS_PER_BLOCK;
  const int m = input_size_m, n = input_size_n;
  // 1. Every rank's input, through the input table, read before the barrier so
  //    its round trip overlaps it; ready before the peer's launch.
  const DTYPE* entries[WORLD];
#pragma unroll
  for (int r = 0; r < WORLD; ++r)
    entries[r] = load(input_ptr_array, input_ptr_array_size, r);
  barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                 signals_stride_parity, signals_stride_block,
                 signals_stride_writer, signals_size_parity, signals_size_block,
                 signals_size_writer, status_ptr, status_stride_field,
                 status_size_field, rank, timeout_ticks, Until::started, 0);
  // 2. This block's rows: summed into `output`, chunk by chunk, then normed.
  for (int row = blockIdx.x; row < m; row += gridDim.x) {
    for (int c = 0; c < n; c += kSpan) {
      DTYPE copies[WORLD][1][kPacks][kPackElems];
      all_gather<AllGatherAlgorithm::tile_pull, false, DTYPE, WORLD, 1,
                 THREADS_PER_BLOCK, WAVE_SIZE, false>(
          entries, input_ptr_array_stride_m, input_ptr_array_stride_n, m, n, n,
          row, c, copies);
      DTYPE sum[1][kPacks][kPackElems];
#pragma unroll
      for (int k = 0; k < kPacks; ++k)
#pragma unroll
        for (int j = 0; j < kPackElems; ++j) {
          float acc = static_cast<float>(copies[0][0][k][j]);
#pragma unroll
          for (int r = 1; r < WORLD; ++r)
            acc += static_cast<float>(copies[r][0][k][j]);
          sum[0][k][j] = static_cast<DTYPE>(acc);
        }
      store<1, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
          sum, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, m,
          n, m, n, row, c, 1);
    }
    add_rms_norm_rows<RESIDUAL, DTYPE, WEIGHT_DTYPE, 1, TILE_N,
                      THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, CHUNKED>(
        row, output_ptr, output_stride_m, output_stride_n, m, n, residual_ptr,
        residual_stride_m, residual_stride_n, weight_ptr, weight_stride_n,
        output_ptr, output_stride_m, output_stride_n, residual_output_ptr,
        residual_output_stride_m, residual_output_stride_n, eps);
  }
  // 3. Every peer done reading this rank's input: the caller may write it.
  barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                 signals_stride_parity, signals_stride_block,
                 signals_stride_writer, signals_size_parity, signals_size_block,
                 signals_size_writer, status_ptr, status_stride_field,
                 status_size_field, rank, timeout_ticks, Until::read, 1);
}

// THE PULLED TWO-SHOT, SLICED BY ROWS: each rank sums its rows (ceil(m /
// WORLD), every column) into its partials (reduce_scatter's rows_pull); each
// row then pulled from its owner into `output` (all_gather's rows), and normed.
// INT4: the inputs quantized into the staging, the sums into the partials, as
// the primitives move them.
template <bool RESIDUAL, bool QUANTIZE_INT4, typename DTYPE,
          typename WEIGHT_DTYPE, int WORLD, int TILE_N, int THREADS_PER_BLOCK,
          int WAVE_SIZE, int PACK_BYTES, bool CHUNKED, bool TAIL>
DINLINE void all_reduce_add_rms_norm_two_shot_pull_rows(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* const* input_ptr_array,
    int input_ptr_array_size, int64_t input_ptr_array_stride_m,
    int64_t input_ptr_array_stride_n, const DTYPE* residual_ptr,
    int64_t residual_stride_m, int64_t residual_stride_n,
    const WEIGHT_DTYPE* weight_ptr, int64_t weight_stride_n, DTYPE* output_ptr,
    int64_t output_stride_m, int64_t output_stride_n,
    DTYPE* residual_output_ptr, int64_t residual_output_stride_m,
    int64_t residual_output_stride_n, char* const* partials_ptr_array,
    int partials_ptr_array_size, int64_t partials_stride_m,
    int64_t partials_stride_n, int partials_size_m, int partials_size_n,
    char* const* staging_ptr_array, int staging_ptr_array_size,
    int64_t staging_stride_m, int64_t staging_stride_n, int staging_size_m,
    int staging_size_n, uint32_t* const* signals_ptr_array,
    int signals_ptr_array_size, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    uint32_t* status_ptr, int64_t status_stride_field, int status_size_field,
    int rank, uint64_t timeout_ticks, float eps) {
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
  constexpr int kPacks = (TILE_N + kPackElems * THREADS_PER_BLOCK - 1) /
                         (kPackElems * THREADS_PER_BLOCK);
  constexpr int kSpan = kPacks * kPackElems * THREADS_PER_BLOCK;
  const int m = input_size_m, n = input_size_n;
  // 1. This rank's rows, summed over every rank into its partials, visible.
  reduce_scatter<ReduceScatterAlgorithm::rows_pull, QUANTIZE_INT4, WORLD, 1,
                 TILE_N, THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, true, TAIL>(
      input_ptr, input_stride_m, input_stride_n, m, n, input_ptr_array,
      input_ptr_array_size, input_ptr_array_stride_m, input_ptr_array_stride_n,
      staging_ptr_array, staging_ptr_array_size, staging_stride_m,
      staging_stride_n, staging_size_m, staging_size_n, partials_ptr_array,
      partials_ptr_array_size, partials_stride_m, partials_stride_n,
      partials_size_m, partials_size_n, signals_ptr_array,
      signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);
  // 2. This block's rows: each from its owner into `output`, then normed.
  const int slice = (m + WORLD - 1) / WORLD;
  for (int row = blockIdx.x; row < m; row += gridDim.x) {
    for (int c = 0; c < n; c += kSpan) {
      DTYPE sum[1][kPacks][kPackElems];
      all_gather<AllGatherAlgorithm::rows, QUANTIZE_INT4, DTYPE, WORLD, TILE_N,
                 THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, true, TAIL>(
          partials_ptr_array, partials_ptr_array_size, partials_stride_m,
          partials_stride_n, partials_size_m, partials_size_n, sum, 0, 0, m, n,
          row, c, slice, rank);
      store<1, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
          sum, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, m,
          n, m, n, row, c, 1);
    }
    add_rms_norm_rows<RESIDUAL, DTYPE, WEIGHT_DTYPE, 1, TILE_N,
                      THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, CHUNKED>(
        row, output_ptr, output_stride_m, output_stride_n, m, n, residual_ptr,
        residual_stride_m, residual_stride_n, weight_ptr, weight_stride_n,
        output_ptr, output_stride_m, output_stride_n, residual_output_ptr,
        residual_output_stride_m, residual_output_stride_n, eps);
  }
}

// THE PULLED TWO-SHOT, SLICED BY COLUMNS: each rank sums its columns (whole
// packs, ceil(packs / WORLD) a rank) of every row into its partials
// (reduce_scatter's columns_pull); each row then pulled pack by pack from the
// owners of its columns into `output` (all_gather's columns), and normed.
template <bool RESIDUAL, bool QUANTIZE_INT4, typename DTYPE,
          typename WEIGHT_DTYPE, int WORLD, int TILE_N, int THREADS_PER_BLOCK,
          int WAVE_SIZE, int PACK_BYTES, bool CHUNKED, bool TAIL>
DINLINE void all_reduce_add_rms_norm_two_shot_pull_columns(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* const* input_ptr_array,
    int input_ptr_array_size, int64_t input_ptr_array_stride_m,
    int64_t input_ptr_array_stride_n, const DTYPE* residual_ptr,
    int64_t residual_stride_m, int64_t residual_stride_n,
    const WEIGHT_DTYPE* weight_ptr, int64_t weight_stride_n, DTYPE* output_ptr,
    int64_t output_stride_m, int64_t output_stride_n,
    DTYPE* residual_output_ptr, int64_t residual_output_stride_m,
    int64_t residual_output_stride_n, char* const* partials_ptr_array,
    int partials_ptr_array_size, int64_t partials_stride_m,
    int64_t partials_stride_n, int partials_size_m, int partials_size_n,
    char* const* staging_ptr_array, int staging_ptr_array_size,
    int64_t staging_stride_m, int64_t staging_stride_n, int staging_size_m,
    int staging_size_n, uint32_t* const* signals_ptr_array,
    int signals_ptr_array_size, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    uint32_t* status_ptr, int64_t status_stride_field, int status_size_field,
    int rank, uint64_t timeout_ticks, float eps) {
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
  constexpr int kPacks = (TILE_N + kPackElems * THREADS_PER_BLOCK - 1) /
                         (kPackElems * THREADS_PER_BLOCK);
  constexpr int kSpan = kPacks * kPackElems * THREADS_PER_BLOCK;
  const int m = input_size_m, n = input_size_n;
  // 1. This rank's columns of every row, summed over every rank into its
  //    partials, visible.
  reduce_scatter<ReduceScatterAlgorithm::columns_pull, QUANTIZE_INT4, WORLD, 1,
                 TILE_N, THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, true, TAIL>(
      input_ptr, input_stride_m, input_stride_n, m, n, input_ptr_array,
      input_ptr_array_size, input_ptr_array_stride_m, input_ptr_array_stride_n,
      staging_ptr_array, staging_ptr_array_size, staging_stride_m,
      staging_stride_n, staging_size_m, staging_size_n, partials_ptr_array,
      partials_ptr_array_size, partials_stride_m, partials_stride_n,
      partials_size_m, partials_size_n, signals_ptr_array,
      signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);
  // 2. This block's rows: each pack from its columns' owner into `output`, then
  //    normed. A rank's slice, in elements, as reduce_scatter cut it.
  const int slice = (n / kPackElems + WORLD - 1) / WORLD * kPackElems;
  for (int row = blockIdx.x; row < m; row += gridDim.x) {
    for (int c = 0; c < n; c += kSpan) {
      DTYPE sum[1][kPacks][kPackElems];
      all_gather<AllGatherAlgorithm::columns, QUANTIZE_INT4, DTYPE, WORLD,
                 TILE_N, THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, true, TAIL>(
          partials_ptr_array, partials_ptr_array_size, partials_stride_m,
          partials_stride_n, partials_size_m, partials_size_n, sum, 0, 0, m, n,
          row, c, slice, rank);
      store<1, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
          sum, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, m,
          n, m, n, row, c, 1);
    }
    add_rms_norm_rows<RESIDUAL, DTYPE, WEIGHT_DTYPE, 1, TILE_N,
                      THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, CHUNKED>(
        row, output_ptr, output_stride_m, output_stride_n, m, n, residual_ptr,
        residual_stride_m, residual_stride_n, weight_ptr, weight_stride_n,
        output_ptr, output_stride_m, output_stride_n, residual_output_ptr,
        residual_output_stride_m, residual_output_stride_n, eps);
  }
}

// THE PUSHED TWO-SHOT (the reduce-scatter pulled, the all-gather pushed), as
// aiter's fused two-stage kernel: each rank sums its columns of every row and
// pushes the sums into every rank's partials (reduce_scatter's columns_push),
// so each rank holds the whole sum locally; each row then read from this rank's
// partials into `output` (INT4: decoded), and normed.
template <bool RESIDUAL, bool QUANTIZE_INT4, typename DTYPE,
          typename WEIGHT_DTYPE, int WORLD, int TILE_N, int THREADS_PER_BLOCK,
          int WAVE_SIZE, int PACK_BYTES, bool CHUNKED, bool TAIL>
DINLINE void all_reduce_add_rms_norm_two_shot_push(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* const* input_ptr_array,
    int input_ptr_array_size, int64_t input_ptr_array_stride_m,
    int64_t input_ptr_array_stride_n, const DTYPE* residual_ptr,
    int64_t residual_stride_m, int64_t residual_stride_n,
    const WEIGHT_DTYPE* weight_ptr, int64_t weight_stride_n, DTYPE* output_ptr,
    int64_t output_stride_m, int64_t output_stride_n,
    DTYPE* residual_output_ptr, int64_t residual_output_stride_m,
    int64_t residual_output_stride_n, char* const* partials_ptr_array,
    int partials_ptr_array_size, int64_t partials_stride_m,
    int64_t partials_stride_n, int partials_size_m, int partials_size_n,
    char* const* staging_ptr_array, int staging_ptr_array_size,
    int64_t staging_stride_m, int64_t staging_stride_n, int staging_size_m,
    int staging_size_n, uint32_t* const* signals_ptr_array,
    int signals_ptr_array_size, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    uint32_t* status_ptr, int64_t status_stride_field, int status_size_field,
    int rank, uint64_t timeout_ticks, float eps) {
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
  constexpr int kPacks = (TILE_N + kPackElems * THREADS_PER_BLOCK - 1) /
                         (kPackElems * THREADS_PER_BLOCK);
  constexpr int kSpan = kPacks * kPackElems * THREADS_PER_BLOCK;
  const int m = input_size_m, n = input_size_n;
  // 1. Every rank's columns of every row, summed by their owners and pushed
  //    into this rank's partials, visible.
  reduce_scatter<ReduceScatterAlgorithm::columns_push, QUANTIZE_INT4, WORLD, 1,
                 TILE_N, THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, true, TAIL>(
      input_ptr, input_stride_m, input_stride_n, m, n, input_ptr_array,
      input_ptr_array_size, input_ptr_array_stride_m, input_ptr_array_stride_n,
      staging_ptr_array, staging_ptr_array_size, staging_stride_m,
      staging_stride_n, staging_size_m, staging_size_n, partials_ptr_array,
      partials_ptr_array_size, partials_stride_m, partials_stride_n,
      partials_size_m, partials_size_n, signals_ptr_array,
      signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);
  // 2. This block's rows: each from this rank's partials into `output`, then
  //    normed.
  for (int row = blockIdx.x; row < m; row += gridDim.x) {
    for (int c = 0; c < n; c += kSpan) {
      DTYPE sum[1][kPacks][kPackElems];
      load<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4, Layout::tile,
           TAIL>(partials_ptr_array, partials_ptr_array_size, rank, 0, 0,
                 partials_stride_m, partials_stride_n, 1, partials_size_m,
                 partials_size_n, m, n, sum, row, c, 1);
      store<1, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
          sum, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, m,
          n, m, n, row, c, 1);
    }
    add_rms_norm_rows<RESIDUAL, DTYPE, WEIGHT_DTYPE, 1, TILE_N,
                      THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, CHUNKED>(
        row, output_ptr, output_stride_m, output_stride_n, m, n, residual_ptr,
        residual_stride_m, residual_stride_n, weight_ptr, weight_stride_n,
        output_ptr, output_stride_m, output_stride_n, residual_output_ptr,
        residual_output_stride_m, residual_output_stride_n, eps);
  }
}

// =============================================================================
// all_reduce_add_rms_norm: THE ALL-REDUCE THEN (THE ADD AND) RMSNORM, ONE
// KERNEL: its ALGORITHM (AllReduceAddRmsNormKernelAlgorithm) a template
// parameter, each algorithm its own function (above); RESIDUAL, the call's:
// with it the residual is added to the sum and the sum written out (vLLM's
// fused_add_rms_norm), without it the sum's own norm (its rms_norm), the
// residual's arguments unread. Every two-shot's partials are the [m, n] sum.
// =============================================================================
template <AllReduceAddRmsNormKernelAlgorithm ALGORITHM, bool RESIDUAL,
          bool QUANTIZE_INT4, typename DTYPE, typename WEIGHT_DTYPE, int WORLD,
          int TILE_N, int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES,
          int WAVES_PER_EU, bool CHUNKED, bool TAIL>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_add_rms_norm(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const DTYPE* const* __restrict__ input_ptr_array,
        int input_ptr_array_size, int64_t input_ptr_array_stride_m,
        int64_t input_ptr_array_stride_n, const DTYPE* residual_ptr,
        int64_t residual_stride_m, int64_t residual_stride_n,
        int residual_size_m, int residual_size_n,
        const WEIGHT_DTYPE* __restrict__ weight_ptr, int64_t weight_stride_n,
        int weight_size_n, DTYPE* output_ptr, int64_t output_stride_m,
        int64_t output_stride_n, int output_size_m, int output_size_n,
        DTYPE* residual_output_ptr, int64_t residual_output_stride_m,
        int64_t residual_output_stride_n, int residual_output_size_m,
        int residual_output_size_n,
        char* const* __restrict__ partials_ptr_array,
        int partials_ptr_array_size, int64_t partials_stride_m,
        int64_t partials_stride_n, int partials_size_m, int partials_size_n,
        char* const* __restrict__ staging_ptr_array, int staging_ptr_array_size,
        int64_t staging_stride_m, int64_t staging_stride_n, int staging_size_m,
        int staging_size_n, uint32_t* const* __restrict__ signals_ptr_array,
        int signals_ptr_array_size, int64_t signals_stride_parity,
        int64_t signals_stride_block, int64_t signals_stride_writer,
        int signals_size_parity, int signals_size_block,
        int signals_size_writer, uint32_t* __restrict__ status_ptr,
        int64_t status_stride_field, int status_size_field, int rank,
        uint64_t timeout_ticks, float eps) {
  // THE DISPATCH, an algorithm a function; RESIDUAL (the call's) whether it
  // adds a residual, QUANTIZE_INT4 (the host's) whether it moves INT4.
  if constexpr (ALGORITHM == AllReduceAddRmsNormKernelAlgorithm::one_shot_pull)
    all_reduce_add_rms_norm_one_shot_pull<RESIDUAL, DTYPE, WEIGHT_DTYPE, WORLD,
                                          TILE_N, THREADS_PER_BLOCK, WAVE_SIZE,
                                          PACK_BYTES, CHUNKED>(
        input_size_m, input_size_n, input_ptr_array, input_ptr_array_size,
        input_ptr_array_stride_m, input_ptr_array_stride_n, residual_ptr,
        residual_stride_m, residual_stride_n, weight_ptr, weight_stride_n,
        output_ptr, output_stride_m, output_stride_n, residual_output_ptr,
        residual_output_stride_m, residual_output_stride_n, signals_ptr_array,
        signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks, eps);
  else if constexpr (ALGORITHM ==
                     AllReduceAddRmsNormKernelAlgorithm::two_shot_pull_rows)
    all_reduce_add_rms_norm_two_shot_pull_rows<
        RESIDUAL, QUANTIZE_INT4, DTYPE, WEIGHT_DTYPE, WORLD, TILE_N,
        THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, CHUNKED, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m,
        input_ptr_array_stride_n, residual_ptr, residual_stride_m,
        residual_stride_n, weight_ptr, weight_stride_n, output_ptr,
        output_stride_m, output_stride_n, residual_output_ptr,
        residual_output_stride_m, residual_output_stride_n, partials_ptr_array,
        partials_ptr_array_size, partials_stride_m, partials_stride_n,
        partials_size_m, partials_size_n, staging_ptr_array,
        staging_ptr_array_size, staging_stride_m, staging_stride_n,
        staging_size_m, staging_size_n, signals_ptr_array,
        signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks, eps);
  else if constexpr (ALGORITHM ==
                     AllReduceAddRmsNormKernelAlgorithm::two_shot_pull_columns)
    all_reduce_add_rms_norm_two_shot_pull_columns<
        RESIDUAL, QUANTIZE_INT4, DTYPE, WEIGHT_DTYPE, WORLD, TILE_N,
        THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, CHUNKED, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m,
        input_ptr_array_stride_n, residual_ptr, residual_stride_m,
        residual_stride_n, weight_ptr, weight_stride_n, output_ptr,
        output_stride_m, output_stride_n, residual_output_ptr,
        residual_output_stride_m, residual_output_stride_n, partials_ptr_array,
        partials_ptr_array_size, partials_stride_m, partials_stride_n,
        partials_size_m, partials_size_n, staging_ptr_array,
        staging_ptr_array_size, staging_stride_m, staging_stride_n,
        staging_size_m, staging_size_n, signals_ptr_array,
        signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks, eps);
  else if constexpr (ALGORITHM ==
                     AllReduceAddRmsNormKernelAlgorithm::two_shot_push)
    all_reduce_add_rms_norm_two_shot_push<
        RESIDUAL, QUANTIZE_INT4, DTYPE, WEIGHT_DTYPE, WORLD, TILE_N,
        THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, CHUNKED, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m,
        input_ptr_array_stride_n, residual_ptr, residual_stride_m,
        residual_stride_n, weight_ptr, weight_stride_n, output_ptr,
        output_stride_m, output_stride_n, residual_output_ptr,
        residual_output_stride_m, residual_output_stride_n, partials_ptr_array,
        partials_ptr_array_size, partials_stride_m, partials_stride_n,
        partials_size_m, partials_size_n, staging_ptr_array,
        staging_ptr_array_size, staging_stride_m, staging_stride_n,
        staging_size_m, staging_size_n, signals_ptr_array,
        signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks, eps);
}

}  // namespace hip_custom_communicator
