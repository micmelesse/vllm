// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// A tile of Kimi-K3's attention residual (AttnRes) and its RMSNorm, every AttnRes kernel's: part of
// an op, composed of common's primitives, so it is neither a kernel nor a primitive.

#pragma once

#include <type_traits>

#include "../../common/interface.cuh"

namespace hip_comms {

// A TILE OF ATTNRES, every AttnRes kernel's: the tile's TILE_M rows at once, TILE_K sources a step
// (the reduced dimension, Triton's BLOCK_L), so each step pays ONE block reduction for every row
// and source of the tile. Rounds as `vllm/models/kimi_k3/amd/ops/attn_res.py` does:
//   d = float(T(sum over ranks)); u = kPrefix ? float(T(float(prefix) + d)) : d (the prefix)
//   logit(src) = dot(src, norm_w * qk_w) * rsqrt(mean(src^2) + eps), src the blocks, then u
//   m = softmax(logits) . sources, online, a tile of sources at a time; out = T(m), or
//   T(m * rsqrt(mean(m^2) + out_eps) * out_w)
// `sum` is the tile's rows summed over the ranks; the prefix and out are rows of the tile's width
// at their Ptrs' strides, `blocks` [rows, sources, hidden] at its row and source strides in
// elements (its columns contiguous, the host checks), and `write_idx` < 0
// writes no block. A row past M (the last tile's) reads row M - 1 and writes nothing, so every
// thread still reaches every reduction.
template <bool HAS_PREFIX, int TILE_K, typename DTYPE, int TILE_M, int TILE_N, int THREADS_PER_BLOCK>
DINLINE void block_attn_res_tile(const Tile<DTYPE, TILE_M, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>& sum, Ptr<DTYPE> prefix,
                                 DTYPE* blocks, int64_t block_stride_m, int64_t block_stride_r,
                                 int write_idx, Ptr<const DTYPE> norm_w, Ptr<const DTYPE> qk_w,
                                 Ptr<const DTYPE> out_norm_w, Ptr<DTYPE> out, int num_blocks,
                                 float eps, float out_eps, float inv_hidden) {
  using Rows    = Tile<DTYPE, TILE_M, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  using RowsF   = Tile<DTYPE, TILE_M, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK, float>;
  using Weight  = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  using WeightF = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK, float>;
  const Weight at_cols{1, sum.N, 0, sum.offs_n};

  // The new prefix: the sum over the ranks added to the old one, rounded once to DTYPE; the old
  // prefix loaded whole before anything is stored.
  Rows np = sum;
  if constexpr (HAS_PREFIX) {
    Rows old = sum.template like<DTYPE>();
    tile_load(old, prefix);
    np = tile_add(old.template to<float>(), sum.template to<float>()).template to<DTYPE>();
  }
  tile_store(np, prefix);
  if (write_idx >= 0) tile_store(blocks + write_idx * block_stride_r, block_stride_m, np);
  // THE PREFIX IS NOT HELD ACROSS THE LOOP: the last source, it is read back where it was just
  // stored (Triton's kernel does the same: held in fp32 it was 32 VGPRs at 7168, 256 threads).
  RowsF acc = np.template to<float>();
  if (num_blocks != 0) {
    Weight nw = at_cols, qk = at_cols;
    tile_load(nw, norm_w);
    tile_load(qk, qk_w);
    const WeightF w = tile_mul(nw.template to<float>(), qk.template to<float>());
    acc = sum.template zeros<float>();
    OnlineSoftmax softmax[TILE_M];
    // FULL STEPS OF SOURCES, THEN ONE LAST STEP with what remains, the prefix and any padding: in
    // the loop every source is a block's, so it has no branch and no zeroed tile a source (16
    // moves a step at 3584, ISA 2026-10-04T03-31-28Z).
    const int full = num_blocks / TILE_K * TILE_K;
    const auto step = [&](int src0, auto last) {
      Rows v[TILE_K];
      float sums[TILE_M * 2 * TILE_K];
#pragma unroll
      for (int s = 0; s < TILE_K; ++s) {
        const int src = src0 + s;
        // THE LAST STEP'S SOURCE IS A BLOCK OR THE PREFIX, picked by its pointer and loaded once: a
        // zeroed tile and a load under a branch held 16 VGPRs at the kernel's peak (ISA
        // 2026-10-04T16-26-29Z). A padding source re-reads the prefix; its logit is -inf.
        const bool block = !decltype(last)::value || src < num_blocks;
        v[s]             = sum.template like<DTYPE>();
        tile_load(v[s], block ? blocks + src * block_stride_r : prefix.data,
                  block ? block_stride_m : prefix.stride_m);
        float ss[TILE_M], dw[TILE_M];
        partial_dot(v[s], v[s], ss);
        partial_dot(v[s], w, dw);
#pragma unroll
        for (int m = 0; m < TILE_M; ++m) {
          sums[(m * TILE_K + s) * 2]     = ss[m];
          sums[(m * TILE_K + s) * 2 + 1] = dw[m];
        }
      }
      block_reduce<Sum, THREADS_PER_BLOCK>(sums);
      float old_scale[TILE_M], scale[TILE_K][TILE_M];
#pragma unroll
      for (int m = 0; m < TILE_M; ++m) {
        float logit[TILE_K];
#pragma unroll
        for (int s = 0; s < TILE_K; ++s)
          logit[s] = !decltype(last)::value || src0 + s <= num_blocks
                         ? sums[(m * TILE_K + s) * 2 + 1] *
                               rsqrtf(sums[(m * TILE_K + s) * 2] * inv_hidden + eps)
                         : -INFINITY;
        float row_scale[TILE_K];
        old_scale[m] = thread_softmax_fold(softmax[m], logit, row_scale);
#pragma unroll
        for (int s = 0; s < TILE_K; ++s) scale[s][m] = row_scale[s];
      }
      acc = tile_fma(acc, old_scale, v[0], scale[0]);
      if constexpr (TILE_K > 1) {
        float ones[TILE_M];
#pragma unroll
        for (int m = 0; m < TILE_M; ++m) ones[m] = 1.0f;
#pragma unroll
        for (int s = 1; s < TILE_K; ++s) acc = tile_fma(acc, ones, v[s], scale[s]);
      }
    };
    for (int src0 = 0; src0 < full; src0 += TILE_K) step(src0, std::false_type{});
    step(full, std::true_type{});
    float inv_den[TILE_M];
#pragma unroll
    for (int m = 0; m < TILE_M; ++m) inv_den[m] = 1.0f / softmax[m].denominator;
    acc = tile_mul(acc, inv_den);
  }

  // The output, normed when out_norm_w is given: its weight in flight under the one reduction for
  // every row's sum of squares.
  Rows result = sum.template like<DTYPE>();
  if (out_norm_w.data != nullptr) {
    Weight g_in = at_cols;
    tile_load(g_in, out_norm_w);
    float ss[TILE_M];
    partial_dot(acc, acc, ss);
    block_reduce<Sum, THREADS_PER_BLOCK>(ss);
    float scale[TILE_M];
#pragma unroll
    for (int m = 0; m < TILE_M; ++m) scale[m] = rsqrtf(ss[m] * inv_hidden + out_eps);
    result = tile_mul(tile_mul(acc, scale), g_in.template to<float>()).template to<DTYPE>();
  } else {
    result = acc.template to<DTYPE>();
  }
  tile_store(result, out);
}

}  // namespace hip_comms
