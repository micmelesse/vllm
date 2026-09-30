// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE COMPUTATION of all-reduce + Kimi-K3's attention residual (AttnRes) + its RMSNorm,
// on a row already reduced over ranks: every AttnRes kernel.

#pragma once

#include "../common/memory.cuh"
#include "../common/reduce.cuh"

namespace hip_comms::fusions::add_attn_res_rms_norm {

// THE MOST SOURCES A ROW MIXES: the stored blocks and the prefix (Kimi-K3: up to 9 + 1).
constexpr int kMaxSources = 10;

// ONE ROW of all-reduce + AttnRes by the whole block, matching
// `vllm/models/kimi_k3/amd/ops/attn_res.py` rounding for rounding:
//
//   d   = float(T(sum over ranks))                   the all-reduce output, as it lands
//   u   = kPrefix ? float(T(float(prefix) + d)) : d  the running prefix, updated or started
//   prefix_out = T(u); blocks[write] = T(u)          the new prefix, and the block written
//   per source s (the stored blocks, then u):
//       logit_s = dot(s, norm_w * qk_w) * rsqrt(mean(s^2) + eps)
//   m   = softmax(logit) . sources                     every source loaded and reduced at once
//   out = T(m), or T(m * rsqrt(mean(m^2) + out_eps) * out_norm_w)
//
// The prefix and the mix stay in registers across the sources, so only the stored blocks
// are read back. The sums run in a different order than Triton's, so a result agrees to
// the rounding of the last few bits, not bitwise. Direction-free as `add_rms_norm::row`:
// `sum` is this thread's share of the reduced row, the new prefix leaves through
// `store_prefix(k, i, v)` and the output through `store_out(k, i, v)`.
template <typename T, bool kPrefix, int K, typename StorePrefix, typename StoreOut>
DINLINE void row(const typename traits<T>::V (&sum)[K],
                 const typename traits<T>::V* prefix, const T* blocks,
                 int64_t block_stride_r, const typename traits<T>::V* norm_w,
                 const typename traits<T>::V* qk_w, const typename traits<T>::V* out_norm_w,
                 int num_blocks, int row, int packs, float inv_hidden, float eps,
                 float out_eps, StorePrefix store_prefix, StoreOut store_out) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  const int base   = row * packs;
  float u[K][NL];
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    if (i >= packs) break;
    V rounded;
    if constexpr (kPrefix) {
      const V p = prefix[base + i];
#pragma unroll
      for (int j = 0; j < NL; ++j) {
        rounded.d[j] = static_cast<T>(static_cast<float>(p.d[j]) +
                                      static_cast<float>(sum[k].d[j]));
        u[k][j]      = static_cast<float>(rounded.d[j]);
      }
    } else {
      rounded = sum[k];
#pragma unroll
      for (int j = 0; j < NL; ++j) u[k][j] = static_cast<float>(sum[k].d[j]);
    }
    store_prefix(k, i, rounded);
  }

  float m[K][NL];
  if (num_blocks == 0) {
    // With only the prefix source, the softmax is exactly one.
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < NL; ++j) m[k][j] = u[k][j];
  } else {
    float w[K][NL];
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const int i = threadIdx.x + k * blockDim.x;
#pragma unroll
      for (int j = 0; j < NL; ++j) w[k][j] = 0.0f;
      if (i >= packs) continue;
      const V a = norm_w[i], b = qk_w[i];
#pragma unroll
      for (int j = 0; j < NL; ++j)
        w[k][j] = static_cast<float>(a.d[j]) * static_cast<float>(b.d[j]);
    }
    // 1. EVERY SOURCE'S SUM OF SQUARES AND WEIGHTED DOT, the stored blocks first and the prefix
    //    last (source num_blocks), as the reference orders them. THE LOADS ARE BRANCH-FREE: a
    //    load under `if (s < num_blocks)` cannot be hoisted past the branch, so each source waited
    //    on its own round trip. Every address is clamped to a real slot and pack instead, all of
    //    them load together, and a source past num_blocks (or a pack past the row) counts zero.
    //    Each is folded into its sums as it lands and not kept (holding nine spilled).
    float sums[2 * kMaxSources];
    const int last = num_blocks - 1;  // num_blocks > 0 here
#pragma unroll
    for (int s = 0; s < kMaxSources - 1; ++s) {
      const V* src = reinterpret_cast<const V*>(blocks + (s < last ? s : last) * block_stride_r);
      const float on = s < num_blocks ? 1.0f : 0.0f;
      float ss = 0.0f, dot = 0.0f;
#pragma unroll
      for (int k = 0; k < K; ++k) {
        const int i     = threadIdx.x + k * blockDim.x;
        const float in  = i < packs ? on : 0.0f;
        const V x       = src[i < packs ? i : packs - 1];
#pragma unroll
        for (int j = 0; j < NL; ++j) {
          const float v = in * static_cast<float>(x.d[j]);
          ss += v * v;
          dot += v * w[k][j];
        }
      }
      sums[2 * s]     = ss;
      sums[2 * s + 1] = dot;
    }
    {
      // The prefix, source num_blocks, from registers.
      float ss = 0.0f, dot = 0.0f;
#pragma unroll
      for (int k = 0; k < K; ++k) {
        const float in = threadIdx.x + k * blockDim.x < packs ? 1.0f : 0.0f;
#pragma unroll
        for (int j = 0; j < NL; ++j) {
          const float v = in * u[k][j];
          ss += v * v;
          dot += v * w[k][j];
        }
      }
      sums[2 * (kMaxSources - 1)]     = ss;
      sums[2 * (kMaxSources - 1) + 1] = dot;
    }
    // 2. ONE block reduction for every source's sums, where each source took its own.
    block_sum_n(sums);
    // 3. THE SOFTMAX over the sources' logits: the stored blocks in slots 0..num_blocks-1, the
    //    prefix in the last slot; the slots between count zero.
    float logit[kMaxSources];
    float max_logit = -INFINITY;
#pragma unroll
    for (int s = 0; s < kMaxSources; ++s) {
      const bool live = s < num_blocks || s == kMaxSources - 1;
      logit[s]        = sums[2 * s + 1] * rsqrtf(sums[2 * s] * inv_hidden + eps);
      if (live) max_logit = fmaxf(max_logit, logit[s]);
    }
    float denominator = 0.0f;
#pragma unroll
    for (int s = 0; s < kMaxSources; ++s) {
      const bool live = s < num_blocks || s == kMaxSources - 1;
      logit[s]        = live ? __expf(logit[s] - max_logit) : 0.0f;
      denominator += logit[s];
    }
    const float prefix_weight = logit[kMaxSources - 1];
    const float inv_den = 1.0f / denominator;
    // 4. THE MIX, the stored sources read again (just read, so from cache), branch-free as in 1:
    //    a source past num_blocks has weight zero.
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const int i  = threadIdx.x + k * blockDim.x;
      const int at = i < packs ? i : packs - 1;
      float mix[NL];
#pragma unroll
      for (int j = 0; j < NL; ++j) mix[j] = prefix_weight * u[k][j];
#pragma unroll
      for (int s = 0; s < kMaxSources - 1; ++s) {
        const V x = reinterpret_cast<const V*>(blocks + (s < last ? s : last) * block_stride_r)[at];
#pragma unroll
        for (int j = 0; j < NL; ++j) mix[j] += logit[s] * static_cast<float>(x.d[j]);
      }
#pragma unroll
      for (int j = 0; j < NL; ++j) m[k][j] = mix[j] * inv_den;
    }
  }

  float scale = 1.0f;
  if (out_norm_w != nullptr) {
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const int i = threadIdx.x + k * blockDim.x;
      if (i >= packs) break;
#pragma unroll
      for (int j = 0; j < NL; ++j) ss += m[k][j] * m[k][j];
    }
    scale = rsqrtf(block_sum2(ss, 0.0f).x * inv_hidden + out_eps);
  }
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    if (i >= packs) break;
    V o;
    if (out_norm_w != nullptr) {
      const V g = out_norm_w[i];
#pragma unroll
      for (int j = 0; j < NL; ++j)
        o.d[j] = static_cast<T>(m[k][j] * scale * static_cast<float>(g.d[j]));
    } else {
#pragma unroll
      for (int j = 0; j < NL; ++j) o.d[j] = static_cast<T>(m[k][j]);
    }
    store_out(k, i, o);
  }
  // Before the next row reuses the reductions' shared slots.
  __syncthreads();
}

}  // namespace hip_comms::fusions::add_attn_res_rms_norm
