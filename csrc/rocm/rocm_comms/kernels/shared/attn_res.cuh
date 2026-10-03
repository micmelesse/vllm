// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// A tile of Kimi-K3's attention residual (AttnRes) and its RMSNorm, every AttnRes kernel's: part of
// an op, composed of common's primitives, so it is neither a kernel nor a primitive.

#pragma once

#include "../../common/common.cuh"
#include "../../machine/build.cuh"

namespace hip_comms {

// A TILE OF ATTNRES, every AttnRes kernel's: the tile's TILE_M rows at once, TILE_K sources a step
// (the reduced dimension, Triton's BLOCK_L), so each step pays ONE block reduction for every row
// and source of the tile. Rounds as `vllm/models/kimi_k3/amd/ops/attn_res.py` does:
//   d = float(T(sum over ranks)); u = kPrefix ? float(T(float(prefix) + d)) : d (the prefix)
//   logit(src) = dot(src, norm_w * qk_w) * rsqrt(mean(src^2) + eps), src the blocks, then u
//   m = softmax(logits) . sources, online, a tile of sources at a time; out = T(m), or
//   T(m * rsqrt(mean(m^2) + out_eps) * out_w)
// `sum` is the tile's rows summed over the ranks; the prefix and out are rows of the tile's width,
// `blocks` [rows, sources, hidden] at row and source strides in elements, and `write_idx` < 0
// writes no block. A row past M (the last tile's) reads row M - 1 and writes nothing, so every
// thread still reaches every reduction.
template <bool kPrefix, int TILE_K, typename T, int TILE_M, int TILE_N, int THREADS_PER_BLOCK>
DINLINE void block_attn_res_tile(const Tile<T, TILE_M, TILE_N, THREADS_PER_BLOCK>& sum, T* prefix,
                                 T* blocks, int64_t block_stride_m, int64_t block_stride_r,
                                 int write_idx, const T* __restrict__ norm_w,
                                 const T* __restrict__ qk_w, const T* __restrict__ out_norm_w,
                                 T* out, int num_blocks, float eps, float out_eps,
                                 float inv_hidden) {
  using Rows    = Tile<T, TILE_M, TILE_N, THREADS_PER_BLOCK>;
  using RowsF   = Tile<T, TILE_M, TILE_N, THREADS_PER_BLOCK, float>;
  using Weight  = Tile<T, 1, TILE_N, THREADS_PER_BLOCK>;
  using WeightF = Tile<T, 1, TILE_N, THREADS_PER_BLOCK, float>;
  constexpr int NL = Rows::kPack;
  const int64_t stride = sum.N;
  const Weight at_cols{1, sum.N, 0, sum.offs_n};

  // The new prefix: the sum over the ranks added to the old one, rounded once to T; the old
  // prefix loaded whole before anything is stored.
  Rows np = sum;
  if constexpr (kPrefix) {
    Rows old = sum.template like<T>();
    thread_load(old, prefix, stride);
#pragma unroll
    for (int m = 0; m < TILE_M; ++m)
#pragma unroll
      for (int k = 0; k < Rows::K; ++k)
#pragma unroll
        for (int j = 0; j < NL; ++j)
          np.v[m][k].d[j] = static_cast<T>(static_cast<float>(old.v[m][k].d[j]) +
                                           static_cast<float>(sum.v[m][k].d[j]));
  }
  thread_store(prefix, stride, np);
  if (write_idx >= 0) thread_store(blocks + write_idx * block_stride_r, block_stride_m, np);
  const RowsF u = np.template to<float>();

  RowsF acc = u;
  if (num_blocks != 0) {
    Weight nw = at_cols, qk = at_cols;
    thread_load(nw, norm_w, 0);
    thread_load(qk, qk_w, 0);
    WeightF w = nw.template to<float>();
    const WeightF q = qk.template to<float>();
#pragma unroll
    for (int k = 0; k < WeightF::K; ++k)
#pragma unroll
      for (int j = 0; j < NL; ++j) w.v[0][k].d[j] *= q.v[0][k].d[j];
    acc = u.template like<float>();
    OnlineSoftmax softmax[TILE_M];
    for (int src0 = 0; src0 <= num_blocks; src0 += TILE_K) {
      RowsF v[TILE_K];
      float sums[TILE_M * 2 * TILE_K];
#pragma unroll
      for (int s = 0; s < TILE_K; ++s) {
        const int src = src0 + s;
        if (src < num_blocks) {
          Rows raw = sum.template like<T>();
          thread_load(raw, blocks + src * block_stride_r, block_stride_m);
          v[s] = raw.template to<float>();
        } else {
          v[s] = src == num_blocks ? u : u.template like<float>();
        }
        float ss[TILE_M], dw[TILE_M];
        thread_dot(v[s], v[s], ss);
        thread_dot(v[s], w, dw);
#pragma unroll
        for (int m = 0; m < TILE_M; ++m) {
          sums[(m * TILE_K + s) * 2]     = ss[m];
          sums[(m * TILE_K + s) * 2 + 1] = dw[m];
        }
      }
      block_reduce<Sum>(sums);
#pragma unroll
      for (int m = 0; m < TILE_M; ++m) {
        float logit[TILE_K];
#pragma unroll
        for (int s = 0; s < TILE_K; ++s)
          logit[s] = src0 + s <= num_blocks
                         ? sums[(m * TILE_K + s) * 2 + 1] *
                               rsqrtf(sums[(m * TILE_K + s) * 2] * inv_hidden + eps)
                         : -INFINITY;
        float scale[TILE_K];
        const float old_scale = thread_softmax_fold(softmax[m], logit, scale);
#pragma unroll
        for (int k = 0; k < RowsF::K; ++k)
#pragma unroll
          for (int j = 0; j < NL; ++j) {
            float a = acc.v[m][k].d[j] * old_scale;
#pragma unroll
            for (int s = 0; s < TILE_K; ++s) a += scale[s] * v[s].v[m][k].d[j];
            acc.v[m][k].d[j] = a;
          }
      }
    }
#pragma unroll
    for (int m = 0; m < TILE_M; ++m) {
      const float inv_den = 1.0f / softmax[m].denominator;
#pragma unroll
      for (int k = 0; k < RowsF::K; ++k)
#pragma unroll
        for (int j = 0; j < NL; ++j) acc.v[m][k].d[j] *= inv_den;
    }
  }

  // The output, normed when out_norm_w is given: its weight in flight under the one reduction for
  // every row's sum of squares.
  Rows result = sum.template like<T>();
  if (out_norm_w != nullptr) {
    Weight g_in = at_cols;
    thread_load(g_in, out_norm_w, 0);
    float ss[TILE_M];
    thread_dot(acc, acc, ss);
    block_reduce<Sum>(ss);
    const WeightF g = g_in.template to<float>();
#pragma unroll
    for (int m = 0; m < TILE_M; ++m) {
      const float scale = rsqrtf(ss[m] * inv_hidden + out_eps);
#pragma unroll
      for (int k = 0; k < Rows::K; ++k)
#pragma unroll
        for (int j = 0; j < NL; ++j)
          result.v[m][k].d[j] = static_cast<T>(acc.v[m][k].d[j] * scale * g.v[0][k].d[j]);
    }
  } else {
    result = acc.template to<T>();
  }
  thread_store(out, stride, result);
}

}  // namespace hip_comms
