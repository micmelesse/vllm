// SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc.
// All rights reserved.
//
// ATTNRES, written once: Kimi-K3's attention residual and its RMSNorm on a tile
// of summed rows, for every AttnRes kernel (the fused all-reduce kernels and
// our separate one, all_reduce_add_attn_res_rms_norm.cuh), with the math only
// it uses (the online softmax, a tile zeroed, the weighted sum's multiply-add).

#pragma once

#include <type_traits>
#include "common.cuh"
#include "rms_normalize.cuh"

namespace hip_custom_communicator {

// ATTNRES, THE ONE FUNCTION every AttnRes kernel calls (the fused ones and our
// separate one alike: the single place its math lives): the tile's ROWS rows at
// once, TILE_K sources a step (the reduced dimension, Triton's BLOCK_L), so
// each step pays ONE block reduction for every row and source of the tile.
// Rounds as `vllm/models/kimi_k3/amd/ops/attn_res.py` does: d = float(T(sum
// over ranks)); u = kPrefix ? float(T(float(prefix) + d)) : d (the prefix)
// logit(src) = dot(src, norm_w * qk_w) * rsqrt(mean(src^2) + eps), src the
// blocks, then u m = softmax(logits) . sources, online, a tile of sources at a
// time; out = T(m), or T(m * rsqrt(mean(m^2) + out_eps) * out_w) `sum_ptr` is
// the tile's rows summed over the ranks, in memory at strides (sum_stride_m,
// sum_stride_n): the kernel's own (read, never written, here), loaded a row the
// block's (THREADS_PER_BLOCK threads along it, K packs of P a thread). THE
// TILE'S PLACE IS THE SUM'S: rows offs_m.. of an input_size_m x input_size_n
// tensor, every buffer read and written at that place (rows at its own row
// stride: the prefix, its output, the output, `blocks` [m, sources, n] at its
// row and source strides in elements, its columns contiguous, the host checks),
// so every tensor sharing the columns shares one clamped column (one offset
// register, load's). `block_output_ptr` is the one block written, null when
// none is. An input is only read and an output only written (STANDARDS): the
// old prefix is read from `prefix_ptr`, the new one written to
// `prefix_output_ptr`, never the same memory. A row past input_size_m (the last
// tile's) reads row input_size_m - 1 and writes nothing, so every thread still
// reaches every reduction.
// LAYOUT: how the thread's packs lie in the row, every access's (common.cuh's
// Layout): tile (pack k of thread t at t + k * THREADS_PER_BLOCK, the exact
// kernels'), or group_run (whole groups a thread, its group j the row's t + j *
// THREADS_PER_BLOCK, the INT4 pull's: common.cuh's Layout).
// A ROW WIDER THAN THE TILE (CHUNKED): attn_res's math over the row's chunks of
// `tile_n` columns (the tile's, K * P * THREADS_PER_BLOCK, from column 0), the
// block's ROWS rows at offs_m (offs_n unused), in passes, each over every
// chunk:
//   A. the new prefix (and the one block written), from the sum's chunk at
//   column c;
//   B. every source's logit, TILE_K sources a step, its sums of squares and
//   dots with norm_w * qk_w over the chunks, one block reduction a step; the
//   logits kept in the launch's dynamic LDS, [ROWS][num_blocks + 1] floats (the
//   host sizes it, attn_res_wide_lds_bytes), their softmax's largest and
//   denominator kept as they come (online);
//   C. with out_norm_w, the mix's sum of squares (mix = the softmax's weighted
//   sum of the sources);
//   D. the mix again, normed or not, stored.
// Every pass reads what the one before stored at the same thread's places (the
// layout is the tile's in every pass), so no pass waits on another's threads;
// the LDS is written by thread 0 between two block barriers. Rounds as
// attn_res: d as summed, the prefix T(prefix + d), out T(m * scale * w) or
// T(m). The softmax's sum is taken directly (each source weighted by exp(logit
// - max) / denominator), not online as attn_res's: the same value, its fp32
//   rounding apart.
// THE ROWS IN ONE PASS, `sum` the thread's registers of them (the reduction's,
// or the pointer form's load): attn_res's math on a tile that fits a pass.
template <int TILE_K, int THREADS_PER_BLOCK, int WAVE_SIZE, Layout LAYOUT,
          bool TAIL, typename DTYPE, int ROWS, int K, int P>
DINLINE void attn_res(
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
  // Pack k's column in packs, LAYOUT's: the mask of the packs past
  // input_size_n.
  [[maybe_unused]] const auto pack_at = [&](int k) {
    const int t = static_cast<int>(threadIdx.x) % THREADS_PER_BLOCK;
    return LAYOUT == Layout::tile
               ? offs_n / P + t + k * THREADS_PER_BLOCK
               : offs_n / P +
                     ((k / kInt4GroupPacks) * THREADS_PER_BLOCK + t) *
                         kInt4GroupPacks +
                     k % kInt4GroupPacks;
  };
  // The new prefix: the sum over the ranks added to the old one, rounded once
  // to DTYPE; the old prefix loaded whole before anything is stored.
  DTYPE new_prefix[ROWS][K][P];
  if (prefix_ptr != nullptr) {  // a scalar branch: the pointer is the launch's
    DTYPE old_prefix[ROWS][K][P];
    load<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
        &prefix_ptr, 1, 0, 0, 0, prefix_stride_m, prefix_stride_n, 1,
        input_size_m, input_size_n, input_size_m, input_size_n, old_prefix,
        offs_m, offs_n, 1);
#pragma unroll
    for (int i = 0; i < ROWS; ++i)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < P; ++j)
          new_prefix[i][k][j] =
              static_cast<DTYPE>(static_cast<float>(old_prefix[i][k][j]) +
                                 static_cast<float>(sum[i][k][j]));
  } else {
#pragma unroll
    for (int i = 0; i < ROWS; ++i)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < P; ++j) new_prefix[i][k][j] = sum[i][k][j];
  }
  store<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
      new_prefix, &prefix_output_ptr, 1, 0, 0, 0, prefix_output_stride_m,
      prefix_output_stride_n, 1, input_size_m, input_size_n, input_size_m,
      input_size_n, offs_m, offs_n, 1);
  if (block_output_ptr != nullptr)
    store<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
        new_prefix, &block_output_ptr, 1, 0, 0, 0, block_output_stride_m,
        block_output_stride_n, 1, input_size_m, input_size_n, input_size_m,
        input_size_n, offs_m, offs_n, 1);
  // THE PREFIX IS NOT HELD ACROSS THE LOOP: the last source, it is read back
  // from `prefix_output_ptr`, where this launch just stored it (Triton's kernel
  // does the same: held in fp32 it was 32 VGPRs at 7168, 256 threads).
  float acc[ROWS][K][P];
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < P; ++j)
        acc[i][k][j] = static_cast<float>(new_prefix[i][k][j]);
  if (num_blocks != 0) {
    // A WEIGHT IS ONE ROW, every row's: row 0 of a 1 x input_size_n tensor.
    DTYPE norm_weight[1][K][P], qk_weight[1][K][P];
    load<1, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
        &norm_weight_ptr, 1, 0, 0, 0, 0, norm_weight_stride_n, 1, 1,
        input_size_n, 1, input_size_n, norm_weight, 0, offs_n, 1);
    load<1, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
        &qk_weight_ptr, 1, 0, 0, 0, 0, qk_weight_stride_n, 1, 1, input_size_n,
        1, input_size_n, qk_weight, 0, offs_n, 1);
    float w[1][K][P];
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < P; ++j)
        w[0][k][j] = static_cast<float>(norm_weight[0][k][j]) *
                     static_cast<float>(qk_weight[0][k][j]);
    // THE SUM STARTS AT ZERO, here: registers start unset (a zeroed default was
    // 32 moves a source in the loop, ISA 2026-10-04T02-22-41Z).
#pragma unroll
    for (int i = 0; i < ROWS; ++i)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < P; ++j) acc[i][k][j] = float(0);
    // THE SOFTMAX SO FAR, a row's: its largest logit and the sum of exp(logit
    // - max).
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
        load<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
            &source_ptr, 1, 0, block ? src : 0, block ? blocks_stride_r : 0,
            block ? blocks_stride_m : prefix_output_stride_m,
            block ? blocks_stride_n : prefix_output_stride_n,
            block ? num_blocks : 1, input_size_m, input_size_n, input_size_m,
            input_size_n, v[s], offs_m, offs_n, 1);
        // This thread's share of each row's sum of squares, then of its dot
        // with w (rms_normalize.cuh's mask: packs past input_size_n count zero,
        // multiplied in so nothing branches).
        float ss[ROWS], dw[ROWS];
#pragma unroll
        for (int m = 0; m < ROWS; ++m) {
          ss[m] = 0.0f;
#pragma unroll
          for (int k = 0; k < K; ++k) {
            float ss_k = 0.0f;
#pragma unroll
            for (int j = 0; j < P; ++j)
              ss_k = fmaf(static_cast<float>(v[s][m][k][j]),
                          static_cast<float>(v[s][m][k][j]), ss_k);
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
      reduce<1, THREADS_PER_BLOCK, WAVE_SIZE, ReduceAxis::all>(sums, sums);
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
        float new_max = softmax_max[m];
#pragma unroll
        for (int t = 0; t < TILE_K; ++t) new_max = fmaxf(new_max, logit[t]);
        const float previous_scale = __expf(softmax_max[m] - new_max);
        softmax_denominator[m] *= previous_scale;
#pragma unroll
        for (int t = 0; t < TILE_K; ++t) {
          row_scale[t] = __expf(logit[t] - new_max);
          softmax_denominator[m] += row_scale[t];
        }
        softmax_max[m] = new_max;
        old_scale[m] = previous_scale;
#pragma unroll
        for (int s = 0; s < TILE_K; ++s) scale[s][m] = row_scale[s];
      }
#pragma unroll
      for (int i = 0; i < ROWS; ++i)
#pragma unroll
        for (int k = 0; k < K; ++k)
#pragma unroll
          for (int j = 0; j < P; ++j)
            acc[i][k][j] = fmaf(static_cast<float>(v[0][i][k][j]), scale[0][i],
                                acc[i][k][j] * old_scale[i]);
      if constexpr (TILE_K > 1) {
        float ones[ROWS];
#pragma unroll
        for (int m = 0; m < ROWS; ++m) ones[m] = 1.0f;
#pragma unroll
        for (int s = 1; s < TILE_K; ++s)
#pragma unroll
          for (int i = 0; i < ROWS; ++i)
#pragma unroll
            for (int k = 0; k < K; ++k)
#pragma unroll
              for (int j = 0; j < P; ++j)
                acc[i][k][j] = fmaf(static_cast<float>(v[s][i][k][j]),
                                    scale[s][i], acc[i][k][j] * ones[i]);
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
    load<1, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
        &output_norm_weight_ptr, 1, 0, 0, 0, 0, output_norm_weight_stride_n, 1,
        1, input_size_n, 1, input_size_n, output_norm_weight, 0, offs_n, 1);
    float scale[ROWS];
    rms_normalize<THREADS_PER_BLOCK, WAVE_SIZE>(
        acc, scale, output_norm_weight, result, output_eps, inv_hidden,
        input_size_n, offs_n, NormPart::row, false, LAYOUT);
  } else {
#pragma unroll
    for (int i = 0; i < ROWS; ++i)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < P; ++j)
          result[i][k][j] = static_cast<DTYPE>(acc[i][k][j]);
  }
  store<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
      result, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1,
      input_size_m, input_size_n, input_size_m, input_size_n, offs_m, offs_n,
      1);
}

template <int TILE_K, int THREADS_PER_BLOCK, int WAVE_SIZE, Layout LAYOUT,
          bool TAIL, int ROWS, int K, int P, bool CHUNKED, typename DTYPE>
DINLINE void attn_res(
    const DTYPE* sum_ptr, int64_t sum_stride_m, int64_t sum_stride_n,
    const DTYPE* prefix_ptr, int64_t prefix_stride_m, int64_t prefix_stride_n,
    const DTYPE* blocks_ptr, int64_t blocks_stride_m, int64_t blocks_stride_r,
    int64_t blocks_stride_n, const DTYPE* norm_weight_ptr,
    int64_t norm_weight_stride_n, const DTYPE* qk_weight_ptr,
    int64_t qk_weight_stride_n, const DTYPE* output_norm_weight_ptr,
    int64_t output_norm_weight_stride_n, DTYPE* prefix_output_ptr,
    int64_t prefix_output_stride_m, int64_t prefix_output_stride_n,
    DTYPE* output_ptr, int64_t output_stride_m, int64_t output_stride_n,
    DTYPE* block_output_ptr, int64_t block_output_stride_m,
    int64_t block_output_stride_n, int input_size_m, int input_size_n,
    int offs_m, int offs_n, int num_blocks, float eps, float output_eps,
    float inv_hidden) {
  // A PASS'S COLUMNS: K packs of P a thread, the block's threads along the row.
  constexpr int tile_n = K * P * THREADS_PER_BLOCK;
  if constexpr (CHUNKED) {
    const int n_all = input_size_n;
    [[maybe_unused]] const auto pack_at = [&](int k, int c) {
      const int t = static_cast<int>(threadIdx.x) % THREADS_PER_BLOCK;
      return LAYOUT == Layout::tile
                 ? c / P + t + k * THREADS_PER_BLOCK
                 : c / P +
                       ((k / kInt4GroupPacks) * THREADS_PER_BLOCK + t) *
                           kInt4GroupPacks +
                       k % kInt4GroupPacks;
    };
    // A. THE NEW PREFIX, chunk by chunk, stored (and the block written) before
    // any source is read.
    for (int c = 0; c < n_all; c += tile_n) {
      DTYPE d[ROWS][K][P];
      load<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
          &sum_ptr, 1, 0, 0, 0, sum_stride_m, sum_stride_n, 1, input_size_m,
          n_all, input_size_m, n_all, d, offs_m, c, 1);
      DTYPE new_prefix[ROWS][K][P];
      if (prefix_ptr != nullptr) {
        DTYPE old_prefix[ROWS][K][P];
        load<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
            &prefix_ptr, 1, 0, 0, 0, prefix_stride_m, prefix_stride_n, 1,
            input_size_m, n_all, input_size_m, n_all, old_prefix, offs_m, c, 1);
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
#pragma unroll
          for (int k = 0; k < K; ++k)
#pragma unroll
            for (int j = 0; j < P; ++j)
              new_prefix[i][k][j] =
                  static_cast<DTYPE>(static_cast<float>(old_prefix[i][k][j]) +
                                     static_cast<float>(d[i][k][j]));
      } else {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
#pragma unroll
          for (int k = 0; k < K; ++k)
#pragma unroll
            for (int j = 0; j < P; ++j) new_prefix[i][k][j] = d[i][k][j];
      }
      store<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
          new_prefix, &prefix_output_ptr, 1, 0, 0, 0, prefix_output_stride_m,
          prefix_output_stride_n, 1, input_size_m, n_all, input_size_m, n_all,
          offs_m, c, 1);
      if (block_output_ptr != nullptr)
        store<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
            new_prefix, &block_output_ptr, 1, 0, 0, 0, block_output_stride_m,
            block_output_stride_n, 1, input_size_m, n_all, input_size_m, n_all,
            offs_m, c, 1);
    }
    // Source src's chunk at column c: a block, or the new prefix (the last
    // source).
    const auto source = [&](int src, int c, DTYPE(&v)[ROWS][K][P]) {
      const bool block = src < num_blocks;
      const DTYPE* const source_ptr = block ? blocks_ptr : prefix_output_ptr;
      load<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
          &source_ptr, 1, 0, block ? src : 0, block ? blocks_stride_r : 0,
          block ? blocks_stride_m : prefix_output_stride_m,
          block ? blocks_stride_n : prefix_output_stride_n,
          block ? num_blocks : 1, input_size_m, n_all, input_size_m, n_all, v,
          offs_m, c, 1);
    };
    extern __shared__ float wide_logits[];  // [ROWS][num_blocks + 1]
    const int sources = num_blocks + 1;
    float softmax_max[ROWS], softmax_denominator[ROWS];
#pragma unroll
    for (int m = 0; m < ROWS; ++m) {
      softmax_max[m] = -INFINITY;
      softmax_denominator[m] = 0.0f;
    }
    if (num_blocks != 0) {
      // B. EVERY SOURCE'S LOGIT, TILE_K a step.
      for (int src0 = 0; src0 < sources; src0 += TILE_K) {
        float sums[1][ROWS * 2 * TILE_K];
#pragma unroll
        for (int q = 0; q < ROWS * 2 * TILE_K; ++q) sums[0][q] = 0.0f;
        for (int c = 0; c < n_all; c += tile_n) {
          DTYPE norm_weight[1][K][P], qk_weight[1][K][P];
          load<1, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
              &norm_weight_ptr, 1, 0, 0, 0, 0, norm_weight_stride_n, 1, 1,
              n_all, 1, n_all, norm_weight, 0, c, 1);
          load<1, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
              &qk_weight_ptr, 1, 0, 0, 0, 0, qk_weight_stride_n, 1, 1, n_all, 1,
              n_all, qk_weight, 0, c, 1);
#pragma unroll
          for (int s = 0; s < TILE_K; ++s) {
            const int src = src0 + s < sources
                                ? src0 + s
                                : num_blocks;  // padding: the prefix again
            DTYPE v[ROWS][K][P];
            source(src, c, v);
#pragma unroll
            for (int m = 0; m < ROWS; ++m)
#pragma unroll
              for (int k = 0; k < K; ++k) {
                float ss_k = 0.0f, dw_k = 0.0f;
#pragma unroll
                for (int j = 0; j < P; ++j) {
                  const float x = static_cast<float>(v[m][k][j]);
                  ss_k = fmaf(x, x, ss_k);
                  dw_k = fmaf(x,
                              static_cast<float>(norm_weight[0][k][j]) *
                                  static_cast<float>(qk_weight[0][k][j]),
                              dw_k);
                }
                const float in = pack_at(k, c) < n_all / P ? 1.0f : 0.0f;
                sums[0][(m * TILE_K + s) * 2] += in * ss_k;
                sums[0][(m * TILE_K + s) * 2 + 1] += in * dw_k;
              }
          }
        }
        reduce<1, THREADS_PER_BLOCK, WAVE_SIZE, ReduceAxis::all>(sums, sums);
#pragma unroll
        for (int m = 0; m < ROWS; ++m)
#pragma unroll
          for (int s = 0; s < TILE_K; ++s) {
            if (src0 + s >= sources) continue;
            const float logit =
                sums[0][(m * TILE_K + s) * 2 + 1] *
                rsqrtf(sums[0][(m * TILE_K + s) * 2] * inv_hidden + eps);
            if (threadIdx.x == 0) wide_logits[m * sources + src0 + s] = logit;
            const float new_max = fmaxf(softmax_max[m], logit);
            softmax_denominator[m] =
                softmax_denominator[m] * __expf(softmax_max[m] - new_max) +
                __expf(logit - new_max);
            softmax_max[m] = new_max;
          }
      }
      __syncthreads();  // every logit in the LDS before any thread reads one
    }
    // THE MIX'S CHUNK at column c: the new prefix alone with no blocks, else
    // every source weighted.
    const auto mix = [&](int c, float (&acc)[ROWS][K][P]) {
      if (num_blocks == 0) {
        DTYPE v[ROWS][K][P];
        source(num_blocks, c, v);
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
#pragma unroll
          for (int k = 0; k < K; ++k)
#pragma unroll
            for (int j = 0; j < P; ++j)
              acc[i][k][j] = static_cast<float>(v[i][k][j]);
        return;
      }
#pragma unroll
      for (int i = 0; i < ROWS; ++i)
#pragma unroll
        for (int k = 0; k < K; ++k)
#pragma unroll
          for (int j = 0; j < P; ++j) acc[i][k][j] = 0.0f;
      for (int src = 0; src < sources; ++src) {
        DTYPE v[ROWS][K][P];
        source(src, c, v);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
          const float p =
              __expf(wide_logits[i * sources + src] - softmax_max[i]) /
              softmax_denominator[i];
#pragma unroll
          for (int k = 0; k < K; ++k)
#pragma unroll
            for (int j = 0; j < P; ++j)
              acc[i][k][j] =
                  fmaf(static_cast<float>(v[i][k][j]), p, acc[i][k][j]);
        }
      }
    };
    // C. WITH out_norm_w, THE MIX'S SCALE over its chunks; D. the mix again,
    // stored.
    float scale[ROWS];
    if (output_norm_weight_ptr != nullptr) {
#pragma unroll
      for (int m = 0; m < ROWS; ++m) scale[m] = 0.0f;
      for (int c = 0; c < n_all; c += tile_n) {
        float acc[ROWS][K][P];
        mix(c, acc);
        DTYPE wide_w[1][K][P], wide_out[ROWS][K][P];
        rms_normalize<THREADS_PER_BLOCK, WAVE_SIZE>(
            acc, scale, wide_w, wide_out, output_eps, inv_hidden, n_all, c,
            NormPart::wide, false, LAYOUT);
      }
    }
    for (int c = 0; c < n_all; c += tile_n) {
      float acc[ROWS][K][P];
      mix(c, acc);
      DTYPE result[ROWS][K][P];
      if (output_norm_weight_ptr != nullptr) {
        DTYPE output_norm_weight[1][K][P];
        load<1, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
            &output_norm_weight_ptr, 1, 0, 0, 0, 0, output_norm_weight_stride_n,
            1, 1, n_all, 1, n_all, output_norm_weight, 0, c, 1);
        rms_normalize<THREADS_PER_BLOCK, WAVE_SIZE>(
            acc, scale, output_norm_weight, result, output_eps, inv_hidden,
            n_all, c, NormPart::apply, false, LAYOUT);
      } else {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
#pragma unroll
          for (int k = 0; k < K; ++k)
#pragma unroll
            for (int j = 0; j < P; ++j)
              result[i][k][j] = static_cast<DTYPE>(acc[i][k][j]);
      }
      store<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
          result, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1,
          input_size_m, n_all, input_size_m, n_all, offs_m, c, 1);
    }
    __syncthreads();  // the LDS's logits read by every thread before the next
                      // rows' are written
  } else {
    DTYPE sum[ROWS][K][P];
    load<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
        &sum_ptr, 1, 0, 0, 0, sum_stride_m, sum_stride_n, 1, input_size_m,
        input_size_n, input_size_m, input_size_n, sum, offs_m, offs_n, 1);
    attn_res<TILE_K, THREADS_PER_BLOCK, WAVE_SIZE, LAYOUT, TAIL>(
        sum, prefix_ptr, prefix_stride_m, prefix_stride_n, blocks_ptr,
        blocks_stride_m, blocks_stride_r, blocks_stride_n, norm_weight_ptr,
        norm_weight_stride_n, qk_weight_ptr, qk_weight_stride_n,
        output_norm_weight_ptr, output_norm_weight_stride_n, prefix_output_ptr,
        prefix_output_stride_m, prefix_output_stride_n, output_ptr,
        output_stride_m, output_stride_n, block_output_ptr,
        block_output_stride_m, block_output_stride_n, input_size_m,
        input_size_n, offs_m, offs_n, num_blocks, eps, output_eps, inv_hidden);
  }
}

}  // namespace hip_custom_communicator
