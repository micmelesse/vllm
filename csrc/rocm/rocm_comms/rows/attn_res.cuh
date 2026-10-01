// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One row of Kimi-K3's attention residual (AttnRes) and its RMSNorm, every AttnRes kernel's: an
// op's row, composed of common's primitives, so it is neither a kernel nor a primitive.

#pragma once

#include "../common/common.cuh"
#include "../machine/build.cuh"

namespace hip_comms {

// ONE ROW OF ATTNRES, once the row's sum over the ranks is in `sum` (packs of this thread's
// Fragment): the new prefix (written back, and to the block row when `written` is not null), the
// softmax over the stored blocks and the prefix, and the output, normed when `out_norm_w` is given.
// A block owns the row; every AttnRes kernel computes its rows with it, so the rounding is one.
template <typename T, bool kPrefix, int kRowPacks>
DINLINE void block_attn_res_row(const typename traits<T>::V (&sum)[kRowPacks], int64_t base,
                          const Fragment<kRowPacks>& f, typename traits<T>::V* pre,
                          typename traits<T>::V* written, const T* row_blocks,
                          int64_t block_stride_r, const T* __restrict__ norm_w,
                          const T* __restrict__ qk_w, const T* __restrict__ out_norm_w,
                          typename traits<T>::V* o, int num_blocks, float eps, float out_eps,
                          float inv_hidden) {
  using V             = typename traits<T>::V;
  constexpr int NL    = traits<T>::N;
  constexpr int kTile = kBuild.attn_res_sources;
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
      const V old = pre[base + f.at[k]];
#pragma unroll
      for (int j = 0; j < NL; ++j)
        new_prefix[k].d[j] =
            static_cast<T>(static_cast<float>(old.d[j]) + static_cast<float>(sum[k].d[j]));
    } else {
      new_prefix[k] = sum[k];
    }
    thread_unpack<T>(new_prefix[k], u[k]);
    if (f.in[k] != 0.0f) {
      pre[base + f.at[k]] = new_prefix[k];
      if (written) written[f.at[k]] = new_prefix[k];
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
      thread_unpack<T>(reinterpret_cast<const V*>(norm_w)[f.at[k]], a);
      thread_unpack<T>(reinterpret_cast<const V*>(qk_w)[f.at[k]], b);
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
          for (int k = 0; k < kRowPacks; ++k) thread_unpack<T>(at_src[f.at[k]], v[t][k]);
        } else {
#pragma unroll
          for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
            for (int j = 0; j < NL; ++j) v[t][k][j] = src == num_blocks ? u[k][j] : 0.0f;
        }
        sums[2 * t]     = thread_dot(v[t], v[t], f);
        sums[2 * t + 1] = thread_dot(v[t], w, f);
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
    float ss[1] = {thread_dot(m, m, f)};
    block_reduce<Sum>(ss);
    scale = rsqrtf(ss[0] * inv_hidden + out_eps);
  }
#pragma unroll
  for (int k = 0; k < kRowPacks; ++k) {
    V result;
    if (out_norm_w != nullptr) {
      float g[NL];
      thread_unpack<T>(reinterpret_cast<const V*>(out_norm_w)[f.at[k]], g);
#pragma unroll
      for (int j = 0; j < NL; ++j) result.d[j] = static_cast<T>(m[k][j] * scale * g[j]);
    } else {
      result = thread_pack<T>(m[k]);
    }
    if (f.in[k] != 0.0f) o[base + f.at[k]] = result;
  }
}

}  // namespace hip_comms
