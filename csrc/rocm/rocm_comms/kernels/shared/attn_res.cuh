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
// `sum[m]` is row offs_m + m's sum over the ranks; a row past M (the last tile's) reads row M - 1
// and writes nothing, so every thread still reaches every reduction. `written(row)` is the block
// row a row writes, or null.
template <typename T, bool kPrefix, int TILE_K, int TILE_M, int TILE_N, int kRowPacks,
          typename Written>
DINLINE void block_attn_res_tile(const typename traits<T>::V (&sum)[TILE_M][kRowPacks],
                                 const Tile<TILE_M, TILE_N>& tile,
                                 const ThreadOffs<kRowPacks>& thread_cols,
                                 typename traits<T>::V* pre, Written written, const T* blocks,
                                 int64_t block_stride_m, int64_t block_stride_r,
                                 const T* __restrict__ norm_w, const T* __restrict__ qk_w,
                                 const T* __restrict__ out_norm_w, typename traits<T>::V* o,
                                 int num_blocks, float eps, float out_eps, float inv_hidden) {
  using V = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  int64_t base[TILE_M];
  const T* row_blocks[TILE_M];
  bool live[TILE_M];
#pragma unroll
  for (int m = 0; m < TILE_M; ++m) {
    const int row = tile.offs_m + m;
    live[m] = row < tile.M;
    const int at = live[m] ? row : tile.M - 1;
    base[m] = int64_t{at} * (tile.N / NL);  // the row, in packs
    row_blocks[m] = blocks + int64_t{at} * block_stride_m;
  }
  // The new prefix: the sum over the ranks added to the old one, rounded once to T.
  float u[TILE_M][kRowPacks][NL];
#pragma unroll
  for (int m = 0; m < TILE_M; ++m) {
    V* w_row = live[m] ? written(tile.offs_m + m) : nullptr;
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) {
      V np;
      if constexpr (kPrefix) {
        const V old = pre[base[m] + thread_cols.offs_n[k]];
#pragma unroll
        for (int j = 0; j < NL; ++j)
          np.d[j] =
              static_cast<T>(static_cast<float>(old.d[j]) + static_cast<float>(sum[m][k].d[j]));
      } else {
        np = sum[m][k];
      }
      thread_unpack<T>(np, u[m][k]);
      if (live[m] && thread_cols.mask_n[k] != 0.0f) {
        pre[base[m] + thread_cols.offs_n[k]] = np;
        if (w_row) w_row[thread_cols.offs_n[k]] = np;
      }
    }
  }
  float acc[TILE_M][kRowPacks][NL];
  if (num_blocks == 0) {
#pragma unroll
    for (int m = 0; m < TILE_M; ++m)
#pragma unroll
      for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
        for (int j = 0; j < NL; ++j) acc[m][k][j] = u[m][k][j];
  } else {
    float w[kRowPacks][NL];
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) {
      float a[NL], b[NL];
      thread_unpack<T>(reinterpret_cast<const V*>(norm_w)[thread_cols.offs_n[k]], a);
      thread_unpack<T>(reinterpret_cast<const V*>(qk_w)[thread_cols.offs_n[k]], b);
#pragma unroll
      for (int j = 0; j < NL; ++j) w[k][j] = a[j] * b[j];
    }
#pragma unroll
    for (int m = 0; m < TILE_M; ++m)
#pragma unroll
      for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
        for (int j = 0; j < NL; ++j) acc[m][k][j] = 0.0f;
    OnlineSoftmax softmax[TILE_M];
    for (int src0 = 0; src0 <= num_blocks; src0 += TILE_K) {
      float v[TILE_M][TILE_K][kRowPacks][NL];
      float sums[TILE_M * 2 * TILE_K];
#pragma unroll
      for (int m = 0; m < TILE_M; ++m)
#pragma unroll
        for (int s = 0; s < TILE_K; ++s) {
          const int src = src0 + s;
          if (src < num_blocks) {
            const V* at_src = reinterpret_cast<const V*>(row_blocks[m] + src * block_stride_r);
#pragma unroll
            for (int k = 0; k < kRowPacks; ++k)
              thread_unpack<T>(at_src[thread_cols.offs_n[k]], v[m][s][k]);
          } else {
#pragma unroll
            for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
              for (int j = 0; j < NL; ++j) v[m][s][k][j] = src == num_blocks ? u[m][k][j] : 0.0f;
          }
          sums[(m * TILE_K + s) * 2] = thread_dot(v[m][s], v[m][s], thread_cols);
          sums[(m * TILE_K + s) * 2 + 1] = thread_dot(v[m][s], w, thread_cols);
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
        for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
          for (int j = 0; j < NL; ++j) {
            float a = acc[m][k][j] * old_scale;
#pragma unroll
            for (int s = 0; s < TILE_K; ++s) a += scale[s] * v[m][s][k][j];
            acc[m][k][j] = a;
          }
      }
    }
#pragma unroll
    for (int m = 0; m < TILE_M; ++m) {
      const float inv_den = 1.0f / softmax[m].denominator;
#pragma unroll
      for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
        for (int j = 0; j < NL; ++j) acc[m][k][j] *= inv_den;
    }
  }
  // The output, normed when out_norm_w is given: one reduction for every row's sum of squares.
  float scale[TILE_M];
#pragma unroll
  for (int m = 0; m < TILE_M; ++m) scale[m] = 1.0f;
  if (out_norm_w != nullptr) {
    float ss[TILE_M];
#pragma unroll
    for (int m = 0; m < TILE_M; ++m) ss[m] = thread_dot(acc[m], acc[m], thread_cols);
    block_reduce<Sum>(ss);
#pragma unroll
    for (int m = 0; m < TILE_M; ++m) scale[m] = rsqrtf(ss[m] * inv_hidden + out_eps);
  }
#pragma unroll
  for (int m = 0; m < TILE_M; ++m) {
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) {
      V result;
      if (out_norm_w != nullptr) {
        float g[NL];
        thread_unpack<T>(reinterpret_cast<const V*>(out_norm_w)[thread_cols.offs_n[k]], g);
#pragma unroll
        for (int j = 0; j < NL; ++j) result.d[j] = static_cast<T>(acc[m][k][j] * scale[m] * g[j]);
      } else {
        result = thread_pack<T>(acc[m][k]);
      }
      if (live[m] && thread_cols.mask_n[k] != 0.0f) o[base[m] + thread_cols.offs_n[k]] = result;
    }
  }
}

}  // namespace hip_comms
