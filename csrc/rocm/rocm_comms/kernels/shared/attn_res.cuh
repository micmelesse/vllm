// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One row of Kimi-K3's attention residual (AttnRes) and its RMSNorm, every AttnRes kernel's: an
// op's row, composed of common's primitives, so it is neither a kernel nor a primitive.

#pragma once

#include "../../common/common.cuh"
#include "../../build.cuh"

namespace hip_comms {

// ONE ROW OF ATTNRES, once the row's sum over the ranks is in `sum` (packs of this thread's
// Tile): the new prefix (written back, and to the block row when `written` is not null), the
// softmax over the stored blocks and the prefix, and the output, normed when `out_norm_w` is given.
// A block owns the row; every AttnRes kernel computes its rows with it, so the rounding is one.
template <typename T, bool kPrefix, int kRowPacks>
DINLINE void block_attn_res_row(const typename traits<T>::V (&sum)[kRowPacks], int64_t base,
                                const ThreadOffs<kRowPacks>& thread_cols,
                                typename traits<T>::V* pre, typename traits<T>::V* written,
                                const T* row_blocks, int64_t block_stride_r,
                                const T* __restrict__ norm_w, const T* __restrict__ qk_w,
                                const T* __restrict__ out_norm_w, typename traits<T>::V* o,
                                int num_blocks, float eps, float out_eps, float inv_hidden) {
  using V             = typename traits<T>::V;
  constexpr int NL    = traits<T>::N;
  constexpr int kTile = kBuild.kernels.attn_res_sources;
  // The AttnRes, rounding as `vllm/models/kimi_k3/amd/ops/attn_res.py` does:
  //   d = float(T(sum over ranks)); u = kPrefix ? float(T(float(prefix) + d)) : d (the prefix)
  //   logit(src) = dot(src, norm_w * qk_w) * rsqrt(mean(src^2) + eps), src the blocks, then u
  //   m = softmax(logits) . sources, online, a tile of sources at a time; out = T(m), or
  //   T(m * rsqrt(mean(m^2) + out_eps) * out_w)
  float u[kRowPacks][NL];
  V new_prefix[kRowPacks];
#pragma unroll
  for (int k = 0; k < kRowPacks; ++k) {
    if constexpr (kPrefix) {
      const V old = pre[base + thread_cols.offs_n[k]];
#pragma unroll
      for (int j = 0; j < NL; ++j)
        new_prefix[k].d[j] =
            static_cast<T>(static_cast<float>(old.d[j]) + static_cast<float>(sum[k].d[j]));
    } else {
      new_prefix[k] = sum[k];
    }
    thread_unpack<T>(new_prefix[k], u[k]);
    if (thread_cols.mask_n[k] != 0.0f) {
      pre[base + thread_cols.offs_n[k]] = new_prefix[k];
      if (written) written[thread_cols.offs_n[k]] = new_prefix[k];
    }
  }
  float m[kRowPacks][NL];
  if (num_blocks == 0) {
    // With only the prefix source, the softmax is exactly one.
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
      for (int j = 0; j < NL; ++j) m[k][j] = u[k][j];
  } else {
    float w[kRowPacks][NL];
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) {
      float a[NL], b[NL];
      thread_unpack<T>(reinterpret_cast<const V*>(norm_w)[thread_cols.offs_n[k]], a);
      thread_unpack<T>(reinterpret_cast<const V*>(qk_w)[thread_cols.offs_n[k]], b);
#pragma unroll
      for (int j = 0; j < NL; ++j) {
        w[k][j] = a[j] * b[j];
        m[k][j] = 0.0f;
      }
    }
    // kTile SOURCES A REDUCTION, as Triton's kernel takes them: a reduction is a block
    // sync, so a row pays one per tile, not one per source. The blocks first, the prefix last.
    OnlineSoftmax softmax;
    for (int src0 = 0; src0 <= num_blocks; src0 += kTile) {
      float v[kTile][kRowPacks][NL];
      float sums[2 * kTile];
#pragma unroll
      for (int t = 0; t < kTile; ++t) {
        const int src = src0 + t;
        if (src < num_blocks) {
          const V* at_src = reinterpret_cast<const V*>(row_blocks + src * block_stride_r);
#pragma unroll
          for (int k = 0; k < kRowPacks; ++k)
            thread_unpack<T>(at_src[thread_cols.offs_n[k]], v[t][k]);
        } else {
#pragma unroll
          for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
            for (int j = 0; j < NL; ++j) v[t][k][j] = src == num_blocks ? u[k][j] : 0.0f;
        }
        sums[2 * t] = thread_dot(v[t], v[t], thread_cols);
        sums[2 * t + 1] = thread_dot(v[t], w, thread_cols);
      }
      block_reduce<Sum>(sums);
      float logit[kTile];
#pragma unroll
      for (int t = 0; t < kTile; ++t)
        logit[t] = src0 + t <= num_blocks
                       ? sums[2 * t + 1] * rsqrtf(sums[2 * t] * inv_hidden + eps)
                       : -INFINITY;
      float scale[kTile];
      const float old_scale = thread_softmax_fold(softmax, logit, scale);
#pragma unroll
      for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
        for (int j = 0; j < NL; ++j) {
          float acc = m[k][j] * old_scale;
#pragma unroll
          for (int t = 0; t < kTile; ++t) acc += scale[t] * v[t][k][j];
          m[k][j] = acc;
        }
    }
    const float inv_den = 1.0f / softmax.denominator;
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
      for (int j = 0; j < NL; ++j) m[k][j] *= inv_den;
  }
  // The output, normed when out_norm_w is given.
  float scale = 1.0f;
  if (out_norm_w != nullptr) {
    float ss[1] = {thread_dot(m, m, thread_cols)};
    block_reduce<Sum>(ss);
    scale = rsqrtf(ss[0] * inv_hidden + out_eps);
  }
#pragma unroll
  for (int k = 0; k < kRowPacks; ++k) {
    V result;
    if (out_norm_w != nullptr) {
      float g[NL];
      thread_unpack<T>(reinterpret_cast<const V*>(out_norm_w)[thread_cols.offs_n[k]], g);
#pragma unroll
      for (int j = 0; j < NL; ++j) result.d[j] = static_cast<T>(m[k][j] * scale * g[j]);
    } else {
      result = thread_pack<T>(m[k]);
    }
    if (thread_cols.mask_n[k] != 0.0f) o[base + thread_cols.offs_n[k]] = result;
  }
}

// A TILE OF ATTNRES: block_attn_res_row's work for the tile's BLOCK_M rows at once, so each source
// pays ONE block reduction for every row of the tile, not one a row. `sum[m]` is row offs_m + m's
// sum over the ranks; a row past M (the last tile's) reads row M - 1 and writes nothing, so every
// thread still reaches every reduction. `written(row)` is the block row a row writes, or null.
template <typename T, bool kPrefix, int BLOCK_M, int kRowPacks, typename Written>
DINLINE void block_attn_res_tile(const typename traits<T>::V (&sum)[BLOCK_M][kRowPacks],
                                 const Tile<BLOCK_M, kRowPacks>& tile,
                                 const ThreadOffs<kRowPacks>& thread_cols,
                                 typename traits<T>::V* pre, Written written, const T* blocks,
                                 int64_t block_stride_m, int64_t block_stride_r,
                                 const T* __restrict__ norm_w, const T* __restrict__ qk_w,
                                 const T* __restrict__ out_norm_w, typename traits<T>::V* o,
                                 int num_blocks, float eps, float out_eps, float inv_hidden) {
  using V = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  constexpr int kTile = kBuild.kernels.attn_res_sources;
  int64_t base[BLOCK_M];
  const T* row_blocks[BLOCK_M];
  bool live[BLOCK_M];
#pragma unroll
  for (int m = 0; m < BLOCK_M; ++m) {
    const int row = tile.offs_m + m;
    live[m] = row < tile.M;
    const int at = live[m] ? row : tile.M - 1;
    base[m] = int64_t{at} * tile.N;
    row_blocks[m] = blocks + int64_t{at} * block_stride_m;
  }
  // The new prefix, as block_attn_res_row rounds it.
  float u[BLOCK_M][kRowPacks][NL];
#pragma unroll
  for (int m = 0; m < BLOCK_M; ++m) {
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
  float acc[BLOCK_M][kRowPacks][NL];
  if (num_blocks == 0) {
#pragma unroll
    for (int m = 0; m < BLOCK_M; ++m)
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
    for (int m = 0; m < BLOCK_M; ++m)
#pragma unroll
      for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
        for (int j = 0; j < NL; ++j) acc[m][k][j] = 0.0f;
    OnlineSoftmax softmax[BLOCK_M];
    for (int src0 = 0; src0 <= num_blocks; src0 += kTile) {
      float v[BLOCK_M][kTile][kRowPacks][NL];
      float sums[BLOCK_M * 2 * kTile];
#pragma unroll
      for (int m = 0; m < BLOCK_M; ++m)
#pragma unroll
        for (int s = 0; s < kTile; ++s) {
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
          sums[(m * kTile + s) * 2] = thread_dot(v[m][s], v[m][s], thread_cols);
          sums[(m * kTile + s) * 2 + 1] = thread_dot(v[m][s], w, thread_cols);
        }
      block_reduce<Sum>(sums);
#pragma unroll
      for (int m = 0; m < BLOCK_M; ++m) {
        float logit[kTile];
#pragma unroll
        for (int s = 0; s < kTile; ++s)
          logit[s] = src0 + s <= num_blocks
                         ? sums[(m * kTile + s) * 2 + 1] *
                               rsqrtf(sums[(m * kTile + s) * 2] * inv_hidden + eps)
                         : -INFINITY;
        float scale[kTile];
        const float old_scale = thread_softmax_fold(softmax[m], logit, scale);
#pragma unroll
        for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
          for (int j = 0; j < NL; ++j) {
            float a = acc[m][k][j] * old_scale;
#pragma unroll
            for (int s = 0; s < kTile; ++s) a += scale[s] * v[m][s][k][j];
            acc[m][k][j] = a;
          }
      }
    }
#pragma unroll
    for (int m = 0; m < BLOCK_M; ++m) {
      const float inv_den = 1.0f / softmax[m].denominator;
#pragma unroll
      for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
        for (int j = 0; j < NL; ++j) acc[m][k][j] *= inv_den;
    }
  }
  // The output, normed when out_norm_w is given: one reduction for every row's sum of squares.
  float scale[BLOCK_M];
#pragma unroll
  for (int m = 0; m < BLOCK_M; ++m) scale[m] = 1.0f;
  if (out_norm_w != nullptr) {
    float ss[BLOCK_M];
#pragma unroll
    for (int m = 0; m < BLOCK_M; ++m) ss[m] = thread_dot(acc[m], acc[m], thread_cols);
    block_reduce<Sum>(ss);
#pragma unroll
    for (int m = 0; m < BLOCK_M; ++m) scale[m] = rsqrtf(ss[m] * inv_hidden + out_eps);
  }
#pragma unroll
  for (int m = 0; m < BLOCK_M; ++m) {
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
