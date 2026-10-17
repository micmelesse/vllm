// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ALL-REDUCE THEN ADD, ATTNRES AND RMSNORM: the AttnRes tile every kernel
// shares, then the kernels, one section each (pull one-shot, pull two-shot,
// push two-shot).

#pragma once

#include <type_traits>
#include "all_gather.cuh"
#include "common.cuh"
#include "reduce_scatter.cuh"
#include "rms_norm.cuh"

namespace hip_custom_communicator {

// ATTNRES'S OWN MATH (moved from common.cuh, I3: only AttnRes calls it): the online softmax, a
// tile's registers zeroed, and the weighted sum's fused multiply-add.
// -------------------------------------------------------------------------------------------------
// SOFTMAX
//
// THE ONLINE SOFTMAX, at thread scope: a softmax over logits that arrive a tile
// at a time, folded into a running weighted sum without a second pass (flash
// attention's).
// -------------------------------------------------------------------------------------------------

// THE SOFTMAX SO FAR, two registers: the largest logit seen (`softmax_max`,
// from -INFINITY) and the sum of exp(logit - max) (`softmax_denominator`, from
// 0). Folds N logits into them (a -INFINITY logit is no source). Returns the
// scale the weighted sum so far takes, and in `scale` each logit's weight; the
// sum is final once divided by softmax_denominator.
template <int NUM_LOGITS>
DINLINE float thread_softmax_fold(float& softmax_max, float& softmax_denominator,
                                  const float (&logit)[NUM_LOGITS],
                                  float (&scale)[NUM_LOGITS]) {
  float new_max = softmax_max;
#pragma unroll
  for (int t = 0; t < NUM_LOGITS; ++t) new_max = fmaxf(new_max, logit[t]);
  const float old_scale = __expf(softmax_max - new_max);
  softmax_denominator *= old_scale;
#pragma unroll
  for (int t = 0; t < NUM_LOGITS; ++t) {
    scale[t] = __expf(logit[t] - new_max);
    softmax_denominator += scale[t];
  }
  softmax_max = new_max;
  return old_scale;
}

template <typename T, int ROWS, int K, int P>
DINLINE void zeros(T (&output)[ROWS][K][P]) {
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < P; ++j) output[i][k][j] = T(0);
}

// output = a * a_row_scale + b * b_row_scale, a row's scales each.
template <typename B, int ROWS, int K, int P>
DINLINE void fma(const float (&a)[ROWS][K][P], const float (&a_row_scale)[ROWS],
                 const B (&b)[ROWS][K][P], const float (&b_row_scale)[ROWS],
                 float (&output)[ROWS][K][P]) {
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < P; ++j)
        output[i][k][j] = fmaf(static_cast<float>(b[i][k][j]), b_row_scale[i],
                               a[i][k][j] * a_row_scale[i]);
}


// =============================================================================
// attn_res
//
// A tile of Kimi-K3's attention residual (AttnRes) and its RMSNorm, every
// AttnRes kernel's: part of an op, composed of common's primitives, so it is
// neither a kernel nor a primitive.
// =============================================================================

// A TILE OF ATTNRES, every AttnRes kernel's: the tile's ROWS rows at once,
// TILE_K sources a step (the reduced dimension, Triton's BLOCK_L), so each step
// pays ONE block reduction for every row and source of the tile. Rounds as
// `vllm/models/kimi_k3/amd/ops/attn_res.py` does:
//   d = float(T(sum over ranks)); u = kPrefix ? float(T(float(prefix) + d)) : d
//   (the prefix) logit(src) = dot(src, norm_w * qk_w) * rsqrt(mean(src^2) +
//   eps), src the blocks, then u m = softmax(logits) . sources, online, a tile
//   of sources at a time; out = T(m), or T(m * rsqrt(mean(m^2) + out_eps) *
//   out_w)
// `sum` is this thread's registers of the tile's rows summed over the ranks, a
// row the block's (THREADS threads along it). THE TILE'S PLACE IS THE SUM'S:
// rows offs_m.. of an input_size_m x input_size_n tensor, every buffer read and
// written at that place (rows at its own row stride: the prefix, its output,
// the output, `blocks` [m, sources, n] at its row and source strides in
// elements, its columns contiguous, the host checks), so every tensor sharing
// the columns shares one clamped column (one offset register, at_col's).
// `block_output_ptr` is the one block written, null when none is. An input is
// only read and an output only written (STANDARDS): the old prefix is read from
// `prefix_ptr`, the new one written to `prefix_output_ptr`, never the same
// memory. A row past input_size_m (the last tile's) reads row input_size_m - 1
// and writes nothing, so every thread still reaches every reduction.
template <bool HAS_PREFIX, int TILE_K, int THREADS, typename DTYPE, int ROWS,
          int K, int P>
DINLINE void block_attn_res_tile(
    const DTYPE (&sum)[ROWS][K][P], const DTYPE* prefix_ptr,
    int64_t prefix_stride_m, int64_t prefix_stride_n, const DTYPE* blocks_ptr,
    int64_t blocks_stride_m, int64_t blocks_stride_r, int64_t blocks_stride_n,
    const DTYPE* norm_weight_ptr, int64_t norm_weight_stride_n,
    const DTYPE* qk_weight_ptr, int64_t qk_weight_stride_n,
    const DTYPE* output_norm_weight_ptr, int64_t output_norm_weight_stride_n,
    DTYPE* prefix_output_ptr, int64_t prefix_output_stride_m,
    int64_t prefix_output_stride_n, DTYPE* output_ptr, int64_t output_stride_m,
    int64_t output_stride_n, DTYPE* block_output_ptr,
    int64_t block_output_stride_m, int64_t block_output_stride_n,
    int input_size_m, int input_size_n, int offs_m, int offs_n, int num_blocks,
    float eps, float output_eps, float inv_hidden) {
  // The new prefix: the sum over the ranks added to the old one, rounded once
  // to DTYPE; the old prefix loaded whole before anything is stored.
  DTYPE new_prefix[ROWS][K][P];
  if constexpr (HAS_PREFIX) {
    DTYPE old_prefix[ROWS][K][P];
    load<1, THREADS>(&prefix_ptr, 1, 0, 0, 0, prefix_stride_m, prefix_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, old_prefix, offs_m, offs_n);
    float old_f[ROWS][K][P], sum_f[ROWS][K][P], new_f[ROWS][K][P];
    convert(old_prefix, old_f);
    convert(sum, sum_f);
    add(old_f, sum_f, new_f);
    convert(new_f, new_prefix);
  } else {
    convert(sum, new_prefix);
  }
  store<1, THREADS, THREADS>(new_prefix, &prefix_output_ptr, 1, 0, 0, 0, prefix_output_stride_m, prefix_output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, offs_m, offs_n);
  if (block_output_ptr != nullptr)
    store<1, THREADS, THREADS>(new_prefix, &block_output_ptr, 1, 0, 0, 0, block_output_stride_m, block_output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, offs_m, offs_n);
  // THE PREFIX IS NOT HELD ACROSS THE LOOP: the last source, it is read back
  // from `prefix_output_ptr`, where this launch just stored it (Triton's kernel
  // does the same: held in fp32 it was 32 VGPRs at 7168, 256 threads).
  float acc[ROWS][K][P];
  convert(new_prefix, acc);
  if (num_blocks != 0) {
    // A WEIGHT IS ONE ROW, every row's: row 0 of a 1 x input_size_n tensor.
    DTYPE norm_weight[1][K][P], qk_weight[1][K][P];
    load<1, THREADS>(&norm_weight_ptr, 1, 0, 0, 0, 0, norm_weight_stride_n, 1, 1, input_size_n, 1, input_size_n, norm_weight, 0, offs_n);
    load<1, THREADS>(&qk_weight_ptr, 1, 0, 0, 0, 0, qk_weight_stride_n, 1, 1, input_size_n, 1, input_size_n, qk_weight, 0, offs_n);
    float norm_weight_f[1][K][P], qk_weight_f[1][K][P], w[1][K][P];
    convert(norm_weight, norm_weight_f);
    convert(qk_weight, qk_weight_f);
    mul(norm_weight_f, qk_weight_f, w);
    // THE SUM STARTS AT ZERO, here: registers start unset (a zeroed default
    // was 32 moves a source in the loop, ISA 2026-10-04T02-22-41Z).
    zeros(acc);
    // THE SOFTMAX SO FAR, a row's: its largest logit and the sum of exp(logit - max).
    float softmax_max[ROWS], softmax_denominator[ROWS];
#pragma unroll
    for (int m = 0; m < ROWS; ++m) {
      softmax_max[m] = -INFINITY;
      softmax_denominator[m] = 0.0f;
    }
    // FULL STEPS OF SOURCES, THEN ONE LAST STEP with what remains, the prefix
    // and any padding: in the loop every source is a block's, so it has no
    // branch and no zeroed registers a source (16 moves a step at 3584, ISA
    // 2026-10-04T03-31-28Z).
    const int full = num_blocks / TILE_K * TILE_K;
    const auto step = [&](int src0, auto last) {
      DTYPE v[TILE_K][ROWS][K][P];
      float sums[1][ROWS * 2 * TILE_K];
#pragma unroll
      for (int s = 0; s < TILE_K; ++s) {
        const int src = src0 + s;
        // THE LAST STEP'S SOURCE IS A BLOCK OR THE PREFIX, picked by its
        // pointer and loaded once: zeroed registers and a load under a branch
        // held 16 VGPRs at the kernel's peak (ISA 2026-10-04T16-26-29Z). A
        // padding source re-reads the prefix; its logit is -inf.
        const bool block = !decltype(last)::value || src < num_blocks;
        const DTYPE* const source_ptr = block ? blocks_ptr : prefix_output_ptr;
        load<1, THREADS>(&source_ptr, 1, 0, block ? src : 0, block ? blocks_stride_r : 0, block ? blocks_stride_m : prefix_output_stride_m, block ? blocks_stride_n : prefix_output_stride_n, block ? num_blocks : 1, input_size_m, input_size_n, input_size_m, input_size_n, v[s], offs_m, offs_n);
        float ss[ROWS], dw[ROWS];
        partial_dot<THREADS>(v[s], v[s], ss, input_size_n, offs_n);
        partial_dot<THREADS>(v[s], w, dw, input_size_n, offs_n);
#pragma unroll
        for (int m = 0; m < ROWS; ++m) {
          sums[0][(m * TILE_K + s) * 2] = ss[m];
          sums[0][(m * TILE_K + s) * 2 + 1] = dw[m];
        }
      }
      block_reduce<Sum, ReduceAxis::all, 1, THREADS>(sums, sums);
      float old_scale[ROWS], scale[TILE_K][ROWS];
#pragma unroll
      for (int m = 0; m < ROWS; ++m) {
        float logit[TILE_K];
#pragma unroll
        for (int s = 0; s < TILE_K; ++s)
          logit[s] =
              !decltype(last)::value || src0 + s <= num_blocks
                  ? sums[0][(m * TILE_K + s) * 2 + 1] *
                        rsqrtf(sums[0][(m * TILE_K + s) * 2] * inv_hidden + eps)
                  : -INFINITY;
        float row_scale[TILE_K];
        old_scale[m] = thread_softmax_fold(softmax_max[m], softmax_denominator[m], logit, row_scale);
#pragma unroll
        for (int s = 0; s < TILE_K; ++s) scale[s][m] = row_scale[s];
      }
      fma(acc, old_scale, v[0], scale[0], acc);
      if constexpr (TILE_K > 1) {
        float ones[ROWS];
#pragma unroll
        for (int m = 0; m < ROWS; ++m) ones[m] = 1.0f;
#pragma unroll
        for (int s = 1; s < TILE_K; ++s) fma(acc, ones, v[s], scale[s], acc);
      }
    };
    for (int src0 = 0; src0 < full; src0 += TILE_K)
      step(src0, std::false_type{});
    step(full, std::true_type{});
    float inv_den[ROWS];
#pragma unroll
    for (int m = 0; m < ROWS; ++m) inv_den[m] = 1.0f / softmax_denominator[m];
    mul(acc, inv_den, acc);
  }

  // The output, normed when out_norm_w is given: its weight in flight under the
  // one reduction for every row's sum of squares.
  DTYPE result[ROWS][K][P];
  if (output_norm_weight_ptr != nullptr) {
    DTYPE output_norm_weight[1][K][P];
    load<1, THREADS>(&output_norm_weight_ptr, 1, 0, 0, 0, 0, output_norm_weight_stride_n, 1, 1, input_size_n, 1, input_size_n, output_norm_weight, 0, offs_n);
    float scale[ROWS];
    rms_norm<THREADS, NormPart::row, false>(acc, scale, output_norm_weight, result, output_eps,
                                            inv_hidden, input_size_n, offs_n);
  } else {
    convert(acc, result);
  }
  store<1, THREADS, THREADS>(result, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, offs_m, offs_n);
}

// =============================================================================
// all_reduce_pull_one_shot_add_attn_res_rms_norm
//
// One-shot pull all-reduce fused with Kimi-K3's attention residual (AttnRes)
// and its RMSNorm.
// =============================================================================

// A block owns a row, as the fused norm does: every rank reduces every row, so
// there is nothing to gather. `blocks` is [m, num_sources, n] at its row,
// source and column strides in elements; `write_idx` < 0 writes no block.
template <typename DTYPE, int WORLD, bool HAS_PREFIX, int TILE_N, int TILE_K,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot_add_attn_res_rms_norm(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const DTYPE* const* __restrict__ input_ptr_array, int input_ptr_array_size,
        int64_t input_ptr_array_stride_m, int64_t input_ptr_array_stride_n,
        const DTYPE* __restrict__ prefix_ptr, int64_t prefix_stride_m,
        int64_t prefix_stride_n, int prefix_size_m, int prefix_size_n,
        const DTYPE* __restrict__ blocks_ptr, int64_t blocks_stride_m,
        int64_t blocks_stride_r, int64_t blocks_stride_n, int blocks_size_m,
        int blocks_size_r, int blocks_size_n,
        const DTYPE* __restrict__ norm_weight_ptr, int64_t norm_weight_stride_n,
        int norm_weight_size_n, const DTYPE* __restrict__ qk_weight_ptr,
        int64_t qk_weight_stride_n, int qk_weight_size_n,
        const DTYPE* __restrict__ output_norm_weight_ptr,
        int64_t output_norm_weight_stride_n, int output_norm_weight_size_n,
        DTYPE* __restrict__ prefix_output_ptr, int64_t prefix_output_stride_m,
        int64_t prefix_output_stride_n, int prefix_output_size_m,
        int prefix_output_size_n, DTYPE* __restrict__ output_ptr,
        int64_t output_stride_m, int64_t output_stride_n, int output_size_m,
        int output_size_n, DTYPE* __restrict__ block_output_ptr,
        int64_t block_output_stride_m, int64_t block_output_stride_n,
        int block_output_size_m, int block_output_size_n,
        uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks, int num_blocks,
        float eps, float output_eps) {
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  constexpr int P = traits<DTYPE>::N;
  constexpr int K = TILE_N / (P * THREADS_PER_BLOCK);
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);

  // 1. Wait until every peer has launched: its input, the caller's tensor (a
  // captured call's, registered) or the host's copy in its staging (an eager
  // call's), is ready before it, and the first barrier is ordered. Read where
  // the input table says, in place.
  barrier<Until::launched, 0, WORLD>(
      signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);

  // 2. Each of this block's rows: read it from every rank in rank order, sum,
  //    AttnRes.
  for (int row = blockIdx.x; row < input_size_m; row += gridDim.x) {
    DTYPE peers[WORLD][1][K][P];
    load<1, THREADS_PER_BLOCK, Ranks::every>(input_ptr_array, input_ptr_array_size, 0, 0, 0, input_ptr_array_stride_m, input_ptr_array_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, peers, row, 0);
    float sum_f[1][K][P];
    peers_sum(peers, sum_f);
    DTYPE sum[1][K][P];
    convert(sum_f, sum);
    block_attn_res_tile<HAS_PREFIX, TILE_K, THREADS_PER_BLOCK>(
        sum, prefix_ptr, prefix_stride_m, prefix_stride_n, blocks_ptr,
        blocks_stride_m, blocks_stride_r, blocks_stride_n, norm_weight_ptr,
        norm_weight_stride_n, qk_weight_ptr, qk_weight_stride_n,
        output_norm_weight_ptr, output_norm_weight_stride_n, prefix_output_ptr,
        prefix_output_stride_m, prefix_output_stride_n, output_ptr,
        output_stride_m, output_stride_n, block_output_ptr,
        block_output_stride_m, block_output_stride_n, input_size_m,
        input_size_n, row, 0, num_blocks, eps, output_eps, inv_hidden);
  }

  // 3. No rank may overwrite its input until every peer has read it.
  barrier<Until::read, 1, WORLD>(
      signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);
#else
  __builtin_trap();
#endif
}

// =============================================================================
// all_reduce_pull_two_shot_add_attn_res_rms_norm
//
// Two-shot pull all-reduce fused with Kimi-K3's attention residual (AttnRes)
// and its RMSNorm.
// =============================================================================

// THE SLICE IS COLUMNS, AND BOTH HALVES PULL, as the plain two-shot all-reduce
// moves its bytes: each rank sums its columns of every row over the ranks into
// its own partials; after the sync each block reads its rows' columns from their
// owners and computes AttnRes on them itself. So the links carry what an
// all-reduce's do, AttnRes's two outputs (the prefix and out) are never
// gathered, and every rank does the AttnRes the unfused path would. A SLICE IS
// WHOLE WAVES (64 packs), so every wave's packs have one owner and partials's
// rank is the same across the wave. A BLOCK REDUCES THE ROWS IT THEN NORMS, so
// it waits only for its twin on every rank (a peers barrier, as the plain
// two-shot's) and the grid is AttnRes's to size. The reduce-scatter on 32
// blocks behind a world barrier left every other block waiting as long as it
// then computed, on a grid the barrier capped at the 192 blocks resident
// (thread trace 2026-10-04T20-14-36Z). `blocks` is [m, num_sources, n] with row
// and source strides in elements; `write_idx` < 0 writes no block.
// QUANTIZATION: what crosses the links (common.cuh's QUANTIZATION): with int4
// the inputs are read from every rank's staged values and scales, and the sums
// stored in and gathered from the partials' INT4 payload.
template <typename DTYPE, int WORLD, bool HAS_PREFIX, int TILE_M, int TILE_N,
          int TILE_K, int THREADS_PER_BLOCK, int WAVES_PER_EU,
          int QUANTIZATION>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot_add_attn_res_rms_norm(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const DTYPE* const* __restrict__ input_ptr_array, int input_ptr_array_size,
        int64_t input_ptr_array_stride_m, int64_t input_ptr_array_stride_n,
        const DTYPE* __restrict__ prefix_ptr, int64_t prefix_stride_m,
        int64_t prefix_stride_n, int prefix_size_m, int prefix_size_n,
        const DTYPE* __restrict__ blocks_ptr, int64_t blocks_stride_m,
        int64_t blocks_stride_r, int64_t blocks_stride_n, int blocks_size_m,
        int blocks_size_r, int blocks_size_n,
        const DTYPE* __restrict__ norm_weight_ptr, int64_t norm_weight_stride_n,
        int norm_weight_size_n, const DTYPE* __restrict__ qk_weight_ptr,
        int64_t qk_weight_stride_n, int qk_weight_size_n,
        const DTYPE* __restrict__ output_norm_weight_ptr,
        int64_t output_norm_weight_stride_n, int output_norm_weight_size_n,
        DTYPE* __restrict__ prefix_output_ptr, int64_t prefix_output_stride_m,
        int64_t prefix_output_stride_n, int prefix_output_size_m,
        int prefix_output_size_n, DTYPE* __restrict__ output_ptr,
        int64_t output_stride_m, int64_t output_stride_n, int output_size_m,
        int output_size_n, DTYPE* __restrict__ block_output_ptr,
        int64_t block_output_stride_m, int64_t block_output_stride_n,
        int block_output_size_m, int block_output_size_n,
        char* const* __restrict__ partials_ptr_array, int partials_ptr_array_size,
        int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
        int partials_size_n, char* const* __restrict__ staging_ptr_array,
        int staging_ptr_array_size, int64_t staging_stride_m, int64_t staging_stride_n,
        int staging_size_m, int staging_size_n,
        uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks, int num_blocks,
        float eps, float output_eps, int reduce_scatter_blocks) {
  // THE INT4 PAYLOADS are inside the staging and the partials (common.cuh's
  // INT4 payload layout): found from each region's own sizes where read or
  // written.
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  constexpr int P = traits<DTYPE>::N;
  constexpr int K = TILE_N / (P * THREADS_PER_BLOCK);
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);
  // EVERY RANK AN EQUAL SLICE OF THE ROW, in elements: rounded up to whole
  // waves (so a wave's packs had one owner), rank 7 owned nothing at 3584 and
  // 7168 and the others reduce-scattered 8/7 of the row.
  const int slice_n = (input_size_n / P + WORLD - 1) / WORLD * P;

  // 1. none: nothing staged, every rank's input read in place where the
  // input table says (a captured call's tensor, or an eager call's host copy
  // in its staging). int4: this rank's input QUANTIZED into its staging's INT4
  // payload (an encoding: what crosses the links is quantized by its owner),
  // the rows this block's twin on every rank reads. Then wait until every
  // peer has launched (and staged its own; the first barrier is ordered).
  // 2. This rank's columns of this block's rows (tiles of TILE_M, every gridDim.x-th), summed
  //    over the ranks into this rank's partials: THE PEERS READ ROTATED, rank + r's r-th, as the
  //    plain two-shot reads them (in rank order every GPU read rank 0 first, a link at a time
  //    carrying the machine's reads, and the reduce-scatter took 1.6x the plain two-shot's cycles
  //    for its bytes: thread traces 2026-10-04T21-03-09Z, 21-26-11Z), summed through LDS.
  // 3. Every rank's sums are in its partials: this block's twin on every rank has reduced the
  //    same rows.
  (void)reduce_scatter_blocks;
  reduce_scatter<WORLD, THREADS_PER_BLOCK, TILE_M, TILE_N, Walk::rows, Accumulate::lds,
                 Direction::pull, QUANTIZATION, true, Until::launched>(
      input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n, input_ptr_array,
      input_ptr_array_size, input_ptr_array_stride_m, input_ptr_array_stride_n,
      staging_ptr_array, staging_ptr_array_size, staging_stride_m, staging_stride_n,
      staging_size_m, staging_size_n, partials_ptr_array, partials_ptr_array_size,
      partials_stride_m, partials_stride_n, partials_size_m, partials_size_n, signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block, signals_size_writer,
      status_ptr, status_stride_field, status_size_field, 0, rank,
      timeout_ticks);

  // 4. Each of this block's tiles: its packs from the ranks that own them (A
  //    WAVE'S PACKS MAY HAVE TWO OWNERS, so each lane reads its owner's partials
  //    from its base, picked per lane), at the input's place in the partials, then AttnRes. A
  //    TILE'S REGISTERS ARE A FUNCTION'S, the lambda's: declared in the loop's
  //    body, a row past input_size_m (never loaded) kept the last tile's
  //    values, so they were held across the whole tile, 16 VGPRs at TILE_M 2
  //    (150 -> 166; ISA harness 2026-10-15).
  const auto attn_res_tile = [&](int offs_m) {
    DTYPE sum[TILE_M][K][P];
    all_gather<WORLD, THREADS_PER_BLOCK, TILE_N, Walk::rows, QUANTIZATION, true, DTYPE>(
        partials_ptr_array, partials_ptr_array_size, partials_stride_m, partials_stride_n,
        partials_size_m, partials_size_n, sum, 0, 0, input_size_m, input_size_n, offs_m, 0,
        slice_n, rank);
    block_attn_res_tile<HAS_PREFIX, TILE_K, THREADS_PER_BLOCK>(
        sum, prefix_ptr, prefix_stride_m, prefix_stride_n, blocks_ptr,
        blocks_stride_m, blocks_stride_r, blocks_stride_n, norm_weight_ptr,
        norm_weight_stride_n, qk_weight_ptr, qk_weight_stride_n,
        output_norm_weight_ptr, output_norm_weight_stride_n, prefix_output_ptr,
        prefix_output_stride_m, prefix_output_stride_n, output_ptr,
        output_stride_m, output_stride_n, block_output_ptr,
        block_output_stride_m, block_output_stride_n, input_size_m,
        input_size_n, offs_m, 0, num_blocks, eps, output_eps, inv_hidden);
  };
  const int tiles = (input_size_m + TILE_M - 1) / TILE_M;
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) attn_res_tile(t * TILE_M);
#else
  __builtin_trap();
#endif
}

// =============================================================================
// all_reduce_push_two_shot_add_attn_res_rms_norm
//
// Two-shot all-reduce (the reduce-scatter pulled, the all-gather pushed) fused
// with Kimi-K3's attention residual (AttnRes) and its RMSNorm: the push norm's
// two phases, with the one-shot AttnRes's row in place of the norm.
// =============================================================================

// THE SLICE IS COLUMNS, as the push norm's: each rank owns packs [rank * S,
// rank * S + S) of every row, sums them over the ranks and pushes the sums into
// every rank's partials, so after the sync each rank holds the whole sum locally
// and computes AttnRes on its rows itself: what crosses the links is the sum,
// once, and AttnRes's two outputs (the prefix and out) are never gathered. ROW
// q BELONGS TO BLOCK q % gridDim.x IN BOTH PHASES: after the sync a block may
// read only what the same block on a peer wrote. `blocks` is [m, num_sources,
// n] with row and source strides in elements; `write_idx` < 0 writes no block.
// QUANTIZATION as the pull's: with int4 the sums pushed to and read from the
// partials' INT4 payload.
template <typename DTYPE, int WORLD, bool HAS_PREFIX, int TILE_N, int TILE_K,
          int THREADS_PER_BLOCK, int WAVES_PER_EU, int QUANTIZATION>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_push_two_shot_add_attn_res_rms_norm(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const DTYPE* const* __restrict__ input_ptr_array, int input_ptr_array_size,
        int64_t input_ptr_array_stride_m, int64_t input_ptr_array_stride_n,
        const DTYPE* __restrict__ prefix_ptr, int64_t prefix_stride_m,
        int64_t prefix_stride_n, int prefix_size_m, int prefix_size_n,
        const DTYPE* __restrict__ blocks_ptr, int64_t blocks_stride_m,
        int64_t blocks_stride_r, int64_t blocks_stride_n, int blocks_size_m,
        int blocks_size_r, int blocks_size_n,
        const DTYPE* __restrict__ norm_weight_ptr, int64_t norm_weight_stride_n,
        int norm_weight_size_n, const DTYPE* __restrict__ qk_weight_ptr,
        int64_t qk_weight_stride_n, int qk_weight_size_n,
        const DTYPE* __restrict__ output_norm_weight_ptr,
        int64_t output_norm_weight_stride_n, int output_norm_weight_size_n,
        DTYPE* __restrict__ prefix_output_ptr, int64_t prefix_output_stride_m,
        int64_t prefix_output_stride_n, int prefix_output_size_m,
        int prefix_output_size_n, DTYPE* __restrict__ output_ptr,
        int64_t output_stride_m, int64_t output_stride_n, int output_size_m,
        int output_size_n, DTYPE* __restrict__ block_output_ptr,
        int64_t block_output_stride_m, int64_t block_output_stride_n,
        int block_output_size_m, int block_output_size_n,
        char* const* __restrict__ partials_ptr_array, int partials_ptr_array_size,
        int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
        int partials_size_n, char* const* __restrict__ staging_ptr_array,
        int staging_ptr_array_size, int64_t staging_stride_m, int64_t staging_stride_n,
        int staging_size_m, int staging_size_n,
        uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks, int num_blocks,
        float eps, float output_eps) {
  // THE INT4 PAYLOADS are inside the staging and the partials (common.cuh's
  // INT4 payload layout): found from each region's own sizes where read or
  // written.
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  constexpr int P = traits<DTYPE>::N;
  constexpr int K = TILE_N / (P * THREADS_PER_BLOCK);
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);
  // 2. This rank's columns of this block's rows, summed over the ranks in rank order and pushed
  //    to every rank (itself too), at their place in the tensor: A SLICE OF LANES A ROW, as wide
  //    as a rank's slice of a TILE_N row in whole waves, so a row's slice is one round trip (a
  //    wave a row took two at 7168, 0.8 us at 16-32 tokens), and as many rows at once as the block
  //    holds; a pack a thread, not a slice's worth (every peer's packs of a whole slice were 32
  //    packs at 16384, spilling). The ranks' pointers are read after the barrier (held across it
  //    they spilled).
  // 3. Every rank's sums are in this rank's partials, and every peer has read this rank's input.
  reduce_scatter<WORLD, THREADS_PER_BLOCK, 1, TILE_N, Walk::rows, Accumulate::registers,
                 Direction::push, QUANTIZATION, true, Until::launched>(
      input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n, input_ptr_array,
      input_ptr_array_size, input_ptr_array_stride_m, input_ptr_array_stride_n,
      staging_ptr_array, staging_ptr_array_size, staging_stride_m, staging_stride_n,
      staging_size_m, staging_size_n, partials_ptr_array, partials_ptr_array_size,
      partials_stride_m, partials_stride_n, partials_size_m, partials_size_n, signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block, signals_size_writer,
      status_ptr, status_stride_field, status_size_field, 0, rank,
      timeout_ticks);

  // 4. This block's rows: the sum out of this rank's partials (at the input's
  //    place in it), then AttnRes, as the one-shot does. The next call's first
  //    sync keeps a peer from pushing into this partials while it is read (a
  //    peer's next kernel starts only once this one has finished).
  //    int4: its INT4 payload, worked out from the partials' own sizes by load.
  for (int row = blockIdx.x; row < input_size_m; row += gridDim.x) {
    DTYPE sum[1][K][P];
    load<1, THREADS_PER_BLOCK, Ranks::own, QUANTIZATION>(partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n, input_size_m, input_size_n, sum, row, 0);
    block_attn_res_tile<HAS_PREFIX, TILE_K, THREADS_PER_BLOCK>(
        sum, prefix_ptr, prefix_stride_m, prefix_stride_n, blocks_ptr,
        blocks_stride_m, blocks_stride_r, blocks_stride_n, norm_weight_ptr,
        norm_weight_stride_n, qk_weight_ptr, qk_weight_stride_n,
        output_norm_weight_ptr, output_norm_weight_stride_n, prefix_output_ptr,
        prefix_output_stride_m, prefix_output_stride_n, output_ptr,
        output_stride_m, output_stride_n, block_output_ptr,
        block_output_stride_m, block_output_stride_n, input_size_m,
        input_size_n, row, 0, num_blocks, eps, output_eps, inv_hidden);
  }
#else
  __builtin_trap();
#endif
}

}  // namespace hip_custom_communicator
