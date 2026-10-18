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
// row the block's (THREADS_PER_BLOCK threads along it). THE TILE'S PLACE IS THE SUM'S:
// rows offs_m.. of an input_size_m x input_size_n tensor, every buffer read and
// written at that place (rows at its own row stride: the prefix, its output,
// the output, `blocks` [m, sources, n] at its row and source strides in
// elements, its columns contiguous, the host checks), so every tensor sharing
// the columns shares one clamped column (one offset register, load's).
// `block_output_ptr` is the one block written, null when none is. An input is
// only read and an output only written (STANDARDS): the old prefix is read from
// `prefix_ptr`, the new one written to `prefix_output_ptr`, never the same
// memory. A row past input_size_m (the last tile's) reads row input_size_m - 1
// and writes nothing, so every thread still reaches every reduction.
// LAYOUT: how the thread's packs lie in the row, every access's (common.cuh's Layout): tile (pack
// k of thread t at t + k * THREADS_PER_BLOCK, the exact kernels'), or group_run (a group a thread,
// its K packs consecutive at t * K + k, the INT4 pull's: its sum lands a group a thread).
template <int TILE_K, int THREADS_PER_BLOCK, Layout LAYOUT, typename DTYPE, int ROWS,
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
  // Pack k's column in packs, LAYOUT's: the mask of the packs past input_size_n.
  [[maybe_unused]] const auto pack_at = [&](int k) {
    const int t = static_cast<int>(threadIdx.x) % THREADS_PER_BLOCK;
    return LAYOUT == Layout::tile ? offs_n / P + t + k * THREADS_PER_BLOCK : offs_n / P + t * K + k;
  };
  // The new prefix: the sum over the ranks added to the old one, rounded once
  // to DTYPE; the old prefix loaded whole before anything is stored.
  DTYPE new_prefix[ROWS][K][P];
  if (prefix_ptr != nullptr) {  // a scalar branch: the pointer is the launch's
    DTYPE old_prefix[ROWS][K][P];
    load<1, ROWS, THREADS_PER_BLOCK, Quantization::none, LAYOUT>(&prefix_ptr, 1, 0, 0, 0, prefix_stride_m, prefix_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, old_prefix, offs_m, offs_n, 1);
#pragma unroll
    for (int i = 0; i < ROWS; ++i)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < P; ++j)
          new_prefix[i][k][j] = static_cast<DTYPE>(static_cast<float>(old_prefix[i][k][j]) +
                                                   static_cast<float>(sum[i][k][j]));
  } else {
#pragma unroll
    for (int i = 0; i < ROWS; ++i)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < P; ++j) new_prefix[i][k][j] = sum[i][k][j];
  }
  store<1, ROWS, THREADS_PER_BLOCK, Quantization::none, LAYOUT>(new_prefix, &prefix_output_ptr, 1, 0, 0, 0, prefix_output_stride_m, prefix_output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, offs_m, offs_n, 1);
  if (block_output_ptr != nullptr)
    store<1, ROWS, THREADS_PER_BLOCK, Quantization::none, LAYOUT>(new_prefix, &block_output_ptr, 1, 0, 0, 0, block_output_stride_m, block_output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, offs_m, offs_n, 1);
  // THE PREFIX IS NOT HELD ACROSS THE LOOP: the last source, it is read back
  // from `prefix_output_ptr`, where this launch just stored it (Triton's kernel
  // does the same: held in fp32 it was 32 VGPRs at 7168, 256 threads).
  float acc[ROWS][K][P];
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < P; ++j) acc[i][k][j] = static_cast<float>(new_prefix[i][k][j]);
  if (num_blocks != 0) {
    // A WEIGHT IS ONE ROW, every row's: row 0 of a 1 x input_size_n tensor.
    DTYPE norm_weight[1][K][P], qk_weight[1][K][P];
    load<1, 1, THREADS_PER_BLOCK, Quantization::none, LAYOUT>(&norm_weight_ptr, 1, 0, 0, 0, 0, norm_weight_stride_n, 1, 1, input_size_n, 1, input_size_n, norm_weight, 0, offs_n, 1);
    load<1, 1, THREADS_PER_BLOCK, Quantization::none, LAYOUT>(&qk_weight_ptr, 1, 0, 0, 0, 0, qk_weight_stride_n, 1, 1, input_size_n, 1, input_size_n, qk_weight, 0, offs_n, 1);
    float w[1][K][P];
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < P; ++j)
        w[0][k][j] = static_cast<float>(norm_weight[0][k][j]) * static_cast<float>(qk_weight[0][k][j]);
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
        load<1, ROWS, THREADS_PER_BLOCK, Quantization::none, LAYOUT>(&source_ptr, 1, 0, block ? src : 0, block ? blocks_stride_r : 0, block ? blocks_stride_m : prefix_output_stride_m, block ? blocks_stride_n : prefix_output_stride_n, block ? num_blocks : 1, input_size_m, input_size_n, input_size_m, input_size_n, v[s], offs_m, offs_n, 1);
        // This thread's share of each row's sum of squares, then of its dot with w (rms_norm.cuh's
        // mask: packs past input_size_n count zero, multiplied in so nothing branches).
        float ss[ROWS], dw[ROWS];
#pragma unroll
        for (int m = 0; m < ROWS; ++m) {
          ss[m] = 0.0f;
#pragma unroll
          for (int k = 0; k < K; ++k) {
            float ss_k = 0.0f;
#pragma unroll
            for (int j = 0; j < P; ++j)
              ss_k = fmaf(static_cast<float>(v[s][m][k][j]), static_cast<float>(v[s][m][k][j]), ss_k);
            const int n = pack_at(k);
            ss[m] += (n < input_size_n / P ? 1.0f : 0.0f) * ss_k;
          }
        }
#pragma unroll
        for (int m = 0; m < ROWS; ++m) {
          dw[m] = 0.0f;
#pragma unroll
          for (int k = 0; k < K; ++k) {
            float dw_k = 0.0f;
#pragma unroll
            for (int j = 0; j < P; ++j)
              dw_k = fmaf(static_cast<float>(v[s][m][k][j]), w[0][k][j], dw_k);
            const int n = pack_at(k);
            dw[m] += (n < input_size_n / P ? 1.0f : 0.0f) * dw_k;
          }
        }
#pragma unroll
        for (int m = 0; m < ROWS; ++m) {
          sums[0][(m * TILE_K + s) * 2] = ss[m];
          sums[0][(m * TILE_K + s) * 2 + 1] = dw[m];
        }
      }
      reduce<1, THREADS_PER_BLOCK, ReduceAxis::all>(sums, sums);
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
#pragma unroll
    for (int i = 0; i < ROWS; ++i)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < P; ++j) acc[i][k][j] *= inv_den[i];
  }

  // The output, normed when out_norm_w is given: its weight in flight under the
  // one reduction for every row's sum of squares.
  DTYPE result[ROWS][K][P];
  if (output_norm_weight_ptr != nullptr) {
    DTYPE output_norm_weight[1][K][P];
    load<1, 1, THREADS_PER_BLOCK, Quantization::none, LAYOUT>(&output_norm_weight_ptr, 1, 0, 0, 0, 0, output_norm_weight_stride_n, 1, 1, input_size_n, 1, input_size_n, output_norm_weight, 0, offs_n, 1);
    float scale[ROWS];
    rms_norm<THREADS_PER_BLOCK>(acc, scale, output_norm_weight, result, output_eps,
                                            inv_hidden, input_size_n, offs_n, NormPart::row, false, LAYOUT);
  } else {
#pragma unroll
    for (int i = 0; i < ROWS; ++i)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < P; ++j) result[i][k][j] = static_cast<DTYPE>(acc[i][k][j]);
  }
  store<1, ROWS, THREADS_PER_BLOCK, Quantization::none, LAYOUT>(result, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, offs_m, offs_n, 1);
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
template <typename DTYPE, int WORLD, int TILE_N, int TILE_K,
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

  // 2. Each of this block's rows: read it from every rank in rank order, sum,
  //    AttnRes.
  for (int row = blockIdx.x; row < input_size_m; row += gridDim.x) {
    DTYPE peers[WORLD][1][kPacks][kPackElems];
    for (int r = 0; r < WORLD; ++r)
      recv<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(input_ptr_array, input_ptr_array_size, r, 0, 0, input_ptr_array_stride_m, input_ptr_array_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, peers[r], row, 0, 1);
    // Summed in rank order in fp32, so every rank's sum agrees bitwise, rounded to DTYPE once.
    DTYPE sum[1][kPacks][kPackElems];
#pragma unroll
    for (int k = 0; k < kPacks; ++k)
#pragma unroll
      for (int j = 0; j < kPackElems; ++j) {
        float acc = static_cast<float>(peers[0][0][k][j]);
#pragma unroll
        for (int r = 1; r < WORLD; ++r) acc += static_cast<float>(peers[r][0][k][j]);
        sum[0][k][j] = static_cast<DTYPE>(acc);
      }
    block_attn_res_tile<TILE_K, THREADS_PER_BLOCK, Layout::tile>(
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
  barrier<WORLD>(
      signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks, Until::read, 1);
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
// QUANTIZATION: what crosses the links (common.cuh's QUANTIZATION), one body
// each: none (THE EXACT BUILD, in the kernel) reads every rank's input in
// place, a column slice a rank; int4 (THE INT4 BUILD, below) pushes it
// quantized, a group a thread, a row slice a rank.

// THE INT4 BUILD, A GROUP A THREAD (TILE_N = kInt4Group * THREADS_PER_BLOCK: a block's threads are
// one row's groups, thread t group t), the pulled norms' INT4 shape (all_reduce_add_rms_norm.cuh)
// with AttnRes in place of the norm: the column split's INT4 read a 4-byte word and a 2-byte scale
// a pack (as many requests as the exact build for a quarter of the bytes) and summed through LDS,
// 762.5 against exact 643.1 us at 8192 x 7168 (2026-10-16T03-50-18Z).
//   1. THE PUSH: every owner's rows of this rank's input, quantized here and sent to that owner's
//      staging inbox at this rank's slot ([WORLD slots of slice_m rows] INT4; the host passes the
//      staging as [WORLD * slice_m, n]).
//   2. THE SUM, LOCAL: this rank's rows, every writer's group from its own inbox, summed from the
//      words (dequantize's slot form), left INT4 in this rank's partials.
//   3. THE GATHER AND ATTNRES: every owner's rows out of its partials, decoded into DTYPE (the sum
//      every rank's AttnRes starts from, as the column split's gather lands it), then AttnRes on
//      each, a group a thread, by every rank (its two outputs are never gathered).
// THE WORK IS UNITS, a row l of the owners of one class (owner % classes), the class's unit
// u = class * slice_m + l, block u % gridDim.x's in all three phases, so a block reads after a
// barrier only what its twins wrote before it. classes: the most of 1, 2, 4 .. WORLD with
// classes * slice_m <= gridDim.x: with
// one class a block took every owner's row l, so no more than slice_m blocks worked and each
// AttnRes'd WORLD rows in turn (1024 x 3584: 84 us on 128 blocks or more, QuickReduce INT4 +
// attn_res 58; 2026-10-18T07-09-39Z).
template <typename DTYPE, int WORLD, int TILE_N, int TILE_K, int THREADS_PER_BLOCK>
DINLINE void all_reduce_pull_two_shot_add_attn_res_rms_norm_int4_body(
    const DTYPE* __restrict__ input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* prefix_ptr, int64_t prefix_stride_m,
    int64_t prefix_stride_n, const DTYPE* blocks_ptr, int64_t blocks_stride_m,
    int64_t blocks_stride_r, int64_t blocks_stride_n, const DTYPE* norm_weight_ptr,
    int64_t norm_weight_stride_n, const DTYPE* qk_weight_ptr, int64_t qk_weight_stride_n,
    const DTYPE* output_norm_weight_ptr, int64_t output_norm_weight_stride_n,
    DTYPE* prefix_output_ptr, int64_t prefix_output_stride_m, int64_t prefix_output_stride_n,
    DTYPE* output_ptr, int64_t output_stride_m, int64_t output_stride_n, DTYPE* block_output_ptr,
    int64_t block_output_stride_m, int64_t block_output_stride_n,
    char* const* partials_ptr_array, int partials_ptr_array_size, int64_t partials_stride_m,
    int64_t partials_stride_n, int partials_size_m, int partials_size_n,
    char* const* staging_ptr_array, int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size,
    int64_t signals_stride_parity, int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    uint32_t* __restrict__ status_ptr, int64_t status_stride_field, int status_size_field,
    int rank, uint64_t timeout_ticks, int num_blocks, float eps, float output_eps) {
  constexpr int kPackElems = traits<DTYPE>::N;
  static_assert(kPackElems * kInt4GroupPacks == kInt4Group, "a pack is a word of its group");
  static_assert(TILE_N == kInt4Group * THREADS_PER_BLOCK, "a group a thread: the block one row's groups");
  const int n = input_size_n;
  const float inv_hidden = 1.0f / static_cast<float>(n);
  const int slice_m = (input_size_m + WORLD - 1) / WORLD;
  const int slice = slice_m * n;  // a rank's rows, flat: its slot of an inbox
  const auto rows_of = [&](int r) { return max(0, min(slice_m, input_size_m - r * slice_m)); };
  const int grid = static_cast<int>(gridDim.x);
  // THE CLASSES, a power of two (an owner's class its low bits), and this block's: with one, its
  // rows l = blockIdx.x, + gridDim.x, ...; with more, every block one unit at most (units <=
  // gridDim.x), its row and class divided out once, here.
  int classes = 1;
  while (classes * 2 <= WORLD && classes * 2 * slice_m <= grid) classes *= 2;
  const int mask = classes - 1;
  const int c = classes == 1 ? 0 : static_cast<int>(blockIdx.x) / slice_m;
  const int first = classes == 1 || static_cast<int>(blockIdx.x) < classes * slice_m
                        ? static_cast<int>(blockIdx.x) - c * slice_m
                        : slice_m;
  const int step = classes == 1 ? grid : slice_m;

  // 1. THE PUSH, owners rotated (rank + w), a row of each a step, two owners' rows in flight.
  for (int l = first; l < slice_m; l += step) {
#pragma unroll 2
    for (int w = 0; w < WORLD; ++w) {
      const int r = (rank + w) % WORLD;
      // Another class's owner, or a late owner's short slice (every thread alike).
      if ((r & mask) != c || l >= rows_of(r)) continue;
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

  // 2. THIS RANK'S ROWS: every writer's group of the row from this rank's inbox (slots rotated
  // from this rank's), summed in fp32 and left INT4 in this rank's partials.
  for (int l = first; l < slice_m; l += step) {
    if ((rank & mask) != c || l >= rows_of(rank)) continue;
    uint32_t got[WORLD][1][kInt4GroupPacks][2];
    load<WORLD, 1, THREADS_PER_BLOCK, Quantization::int4, Layout::group_run>(
        staging_ptr_array, staging_ptr_array_size, rank, rank, slice, staging_stride_m,
        staging_stride_n, WORLD, staging_size_m, staging_size_n, 1, (l + 1) * n, got, 0, l * n, 1);
    float sum[1][kInt4GroupPacks][kPackElems];
    dequantize<DTYPE>(got, sum);
    store<WORLD, 1, THREADS_PER_BLOCK, Quantization::int4, Layout::group_run>(
        sum, partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m,
        partials_stride_n, 1, partials_size_m, partials_size_n, 1, (l + 1) * n, 0, l * n, 1);
  }

  // 3. Every rank's rows are visible to its peers.
  barrier<WORLD>(
      signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks, Until::visible, 1);

  // 4. EVERY OWNER'S ROW l, owners rotated (rank + v): the row's group from its owner, decoded,
  // then AttnRes on it. The next call's first sync keeps an owner from overwriting its partials
  // while they are read.
  for (int l = first; l < slice_m; l += step) {
    for (int v = 0; v < WORLD; ++v) {
      const int owner = (rank + v) % WORLD;
      if ((owner & mask) != c || l >= rows_of(owner)) continue;
      uint32_t raw[1][kInt4GroupPacks][2];
      recv<WORLD, 1, THREADS_PER_BLOCK, Quantization::int4, Layout::group_run>(
          partials_ptr_array, partials_ptr_array_size, owner, 0, 0, partials_stride_m,
          partials_stride_n, 1, partials_size_m, partials_size_n, 1, (l + 1) * n, raw, 0, l * n, 1);
      DTYPE sum[1][kInt4GroupPacks][kPackElems];
#pragma unroll
      for (int k = 0; k < kInt4GroupPacks; ++k)
        dequantize(raw[0][k][0], __builtin_bit_cast(float, raw[0][k][1]), sum[0][k]);
      block_attn_res_tile<TILE_K, THREADS_PER_BLOCK, Layout::group_run>(
          sum, prefix_ptr, prefix_stride_m, prefix_stride_n, blocks_ptr, blocks_stride_m,
          blocks_stride_r, blocks_stride_n, norm_weight_ptr, norm_weight_stride_n, qk_weight_ptr,
          qk_weight_stride_n, output_norm_weight_ptr, output_norm_weight_stride_n,
          prefix_output_ptr, prefix_output_stride_m, prefix_output_stride_n, output_ptr,
          output_stride_m, output_stride_n, block_output_ptr, block_output_stride_m,
          block_output_stride_n, input_size_m, n, owner * slice_m + l, 0, num_blocks, eps,
          output_eps, inv_hidden);
    }
  }
}

template <typename DTYPE, int WORLD, int TILE_M, int TILE_N,
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
  if constexpr (QUANTIZATION == Quantization::int4) {
    static_assert(TILE_M == 1, "the INT4 build is a row a block");
    (void)reduce_scatter_blocks;
    all_reduce_pull_two_shot_add_attn_res_rms_norm_int4_body<DTYPE, WORLD, TILE_N, TILE_K,
                                                             THREADS_PER_BLOCK>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n, prefix_ptr,
        prefix_stride_m, prefix_stride_n, blocks_ptr, blocks_stride_m, blocks_stride_r,
        blocks_stride_n, norm_weight_ptr, norm_weight_stride_n, qk_weight_ptr,
        qk_weight_stride_n, output_norm_weight_ptr, output_norm_weight_stride_n,
        prefix_output_ptr, prefix_output_stride_m, prefix_output_stride_n, output_ptr,
        output_stride_m, output_stride_n, block_output_ptr, block_output_stride_m,
        block_output_stride_n, partials_ptr_array, partials_ptr_array_size, partials_stride_m,
        partials_stride_n, partials_size_m, partials_size_n, staging_ptr_array,
        staging_ptr_array_size, staging_stride_m, staging_stride_n, staging_size_m,
        staging_size_n, signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
        signals_stride_block, signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field, rank,
        timeout_ticks, num_blocks, eps, output_eps);
  } else {
    // THE EXACT BUILD.
    constexpr int kPackElems = traits<DTYPE>::N;
    constexpr int kPacks = TILE_N / (kPackElems * THREADS_PER_BLOCK);
    const float inv_hidden = 1.0f / static_cast<float>(input_size_n);
    // EVERY RANK AN EQUAL SLICE OF THE ROW, in elements: rounded up to whole
    // waves (so a wave's packs had one owner), rank 7 owned nothing at 3584 and
    // 7168 and the others reduce-scattered 8/7 of the row.
    const int slice_n = (input_size_n / kPackElems + WORLD - 1) / WORLD * kPackElems;

    // 1. Nothing staged, every rank's input read in place where the input table
    // says (a captured call's tensor, or an eager call's host copy in its
    // staging). Wait until every peer has launched, UNORDERED (started, as the plain pull
    // two-shot's): nothing written before it, every read after it of what a peer wrote before
    // its launch.
    // 2. This rank's columns of this block's rows (tiles of TILE_M, every gridDim.x-th), summed
    //    over the ranks into this rank's partials: THE PEERS READ ROTATED, rank + r's r-th, as the
    //    plain two-shot reads them (in rank order every GPU read rank 0 first, a link at a time
    //    carrying the machine's reads, and the reduce-scatter took 1.6x the plain two-shot's cycles
    //    for its bytes: thread traces 2026-10-04T21-03-09Z, 21-26-11Z), summed through LDS.
    // 3. Every rank's sums are in its partials: this block's twin on every rank has reduced the
    //    same rows.
    (void)reduce_scatter_blocks;
    reduce_scatter<WORLD, TILE_M, TILE_N, THREADS_PER_BLOCK, Quantization::none, true, Walk::rows, Accumulate::lds, Direction::pull, Until::started>(
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
      DTYPE sum[TILE_M][kPacks][kPackElems];
      all_gather<DTYPE, WORLD, TILE_N, THREADS_PER_BLOCK, Quantization::none, true, Walk::rows>(
          partials_ptr_array, partials_ptr_array_size, partials_stride_m, partials_stride_n,
          partials_size_m, partials_size_n, sum, 0, 0, input_size_m, input_size_n, offs_m, 0,
          slice_n, rank);
      block_attn_res_tile<TILE_K, THREADS_PER_BLOCK, Layout::tile>(
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
  }
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
template <typename DTYPE, int WORLD, int TILE_N, int TILE_K,
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
  constexpr int kPackElems = traits<DTYPE>::N;
  constexpr int kPacks = TILE_N / (kPackElems * THREADS_PER_BLOCK);
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);
  // 2. This rank's columns of this block's rows, summed over the ranks in rank order and pushed
  //    to every rank (itself too), at their place in the tensor: A SLICE OF LANES A ROW, as wide
  //    as a rank's slice of a TILE_N row in whole waves, so a row's slice is one round trip (a
  //    wave a row took two at 7168, 0.8 us at 16-32 tokens), and as many rows at once as the block
  //    holds; a pack a thread, not a slice's worth (every peer's packs of a whole slice were 32
  //    packs at 16384, spilling). The ranks' pointers are read after the barrier (held across it
  //    they spilled).
  // 3. Every rank's sums are in this rank's partials, and every peer has read this rank's input.
  // THE FIRST BARRIER: exact, UNORDERED (started, as the plain pulls'): nothing written before it,
  // every input read in place after it (written before the peers' launches), and the pushes into a
  // peer's partials after it land where that peer finished reading in its last launch, ended before
  // this one started. int4 stages this rank's rows before it, so it stays ordered (launched).
  reduce_scatter<WORLD, 1, TILE_N, THREADS_PER_BLOCK, QUANTIZATION, true, Walk::rows, Accumulate::registers, Direction::push, QUANTIZATION == Quantization::none ? Until::started : Until::launched>(
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
    DTYPE sum[1][kPacks][kPackElems];
    load<WORLD, 1, THREADS_PER_BLOCK, QUANTIZATION, Layout::tile>(partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n, input_size_m, input_size_n, sum, row, 0, 1);
    block_attn_res_tile<TILE_K, THREADS_PER_BLOCK, Layout::tile>(
        sum, prefix_ptr, prefix_stride_m, prefix_stride_n, blocks_ptr,
        blocks_stride_m, blocks_stride_r, blocks_stride_n, norm_weight_ptr,
        norm_weight_stride_n, qk_weight_ptr, qk_weight_stride_n,
        output_norm_weight_ptr, output_norm_weight_stride_n, prefix_output_ptr,
        prefix_output_stride_m, prefix_output_stride_n, output_ptr,
        output_stride_m, output_stride_n, block_output_ptr,
        block_output_stride_m, block_output_stride_n, input_size_m,
        input_size_n, row, 0, num_blocks, eps, output_eps, inv_hidden);
  }
}

}  // namespace hip_custom_communicator
