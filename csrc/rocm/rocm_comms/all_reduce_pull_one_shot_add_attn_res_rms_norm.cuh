// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce fused with Kimi-K3's attention residual (AttnRes) and its
// RMSNorm.

#pragma once

#include "p2p/p2p.cuh"
#include "common/dot.cuh"
#include "common/elementwise.cuh"
#include "common/memory.cuh"
#include "common/reduce.cuh"
#include "common/utils.cuh"
#include "launch.cuh"

namespace hip_comms {

// A block owns a row, as the fused norm does: every rank reduces every row, so there is
// nothing to gather. `blocks` is [rows, num_sources, hidden] with row and source strides
// in elements; `write_idx` < 0 writes no block.
template <typename T, int ngpus, bool kPrefix, int kRowPacks>
__global__ void __launch_bounds__(kMaxThreads, 1)
    all_reduce_pull_one_shot_add_attn_res_rms_norm(
        p2p::DevComm p, T* __restrict__ prefix, T* __restrict__ blocks,
        int64_t block_stride_m, int64_t block_stride_r, const T* __restrict__ norm_w,
        const T* __restrict__ qk_w, const T* __restrict__ out_norm_w, T* __restrict__ out,
        int num_blocks, int write_idx, float eps, float out_eps, int rows, int packs) {
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  V* pre                 = reinterpret_cast<V*>(prefix);
  V* o                   = reinterpret_cast<V*>(out);
  const auto f           = fragment<kRowPacks>(packs);
  // The block row `row` writes, or none.
  auto written = [&](int row) -> V* {
    return write_idx < 0 ? nullptr
                         : reinterpret_cast<V*>(blocks + row * block_stride_m +
                                                write_idx * block_stride_r);
  };

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto peers = p2p::peers<T, ngpus>(p);
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);
  const auto read = [&](int r, int64_t i) { return p2p::read_input(peers[r], i); };

  // 2. Each of this block's rows: read it from every rank in rank order, sum, AttnRes.
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t base = int64_t{row} * packs;
    V sum[kRowPacks];
    peers_reduce<T, ngpus>(read, row, packs, f, sum);
    // The AttnRes, rounding as `vllm/models/kimi_k3/amd/ops/attn_res.py` does:
    //   d = float(T(sum over ranks)); u = kPrefix ? float(T(float(prefix) + d)) : d (the prefix)
    //   logit(src) = dot(src, norm_w * qk_w) * rsqrt(mean(src^2) + eps), src the blocks, then u
    //   m = softmax(logits) . sources, online, one source at a time; out = T(m), or
    //   T(m * rsqrt(mean(m^2) + out_eps) * out_w)
    const T* row_blocks = blocks + int64_t{row} * block_stride_m;
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
      unpack<T>(new_prefix[k], u[k]);
      if (f.in[k] != 0.0f) {
        pre[base + f.at[k]] = new_prefix[k];
        if (V* dst = written(row)) dst[f.at[k]] = new_prefix[k];
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
        unpack<T>(reinterpret_cast<const V*>(norm_w)[f.at[k]], a);
        unpack<T>(reinterpret_cast<const V*>(qk_w)[f.at[k]], b);
#pragma unroll
        for (int j = 0; j < NL; ++j) {
          w[k][j] = a[j] * b[j];
          m[k][j] = 0.0f;
        }
      }
      float max_logit = -INFINITY, denominator = 0.0f;
      for (int src = 0; src <= num_blocks; ++src) {
        // The stored blocks first, the prefix last, as the reference orders its sources.
        float v[kRowPacks][NL];
        if (src < num_blocks) {
          const V* at_src = reinterpret_cast<const V*>(row_blocks + src * block_stride_r);
#pragma unroll
          for (int k = 0; k < kRowPacks; ++k) unpack<T>(at_src[f.at[k]], v[k]);
        } else {
#pragma unroll
          for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
            for (int j = 0; j < NL; ++j) v[k][j] = u[k][j];
        }
        float sums[2] = {thread_dot(v, v, f), thread_dot(v, w, f)};
        block_reduce<Sum>(sums);
        const float logit      = sums[1] * rsqrtf(sums[0] * inv_hidden + eps);
        const float new_max    = fmaxf(max_logit, logit);
        const float old_scale  = __expf(max_logit - new_max);
        const float this_scale = __expf(logit - new_max);
        denominator            = denominator * old_scale + this_scale;
        max_logit              = new_max;
#pragma unroll
        for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
          for (int j = 0; j < NL; ++j) m[k][j] = m[k][j] * old_scale + this_scale * v[k][j];
      }
      const float inv_den = 1.0f / denominator;
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
        unpack<T>(reinterpret_cast<const V*>(out_norm_w)[f.at[k]], g);
#pragma unroll
        for (int j = 0; j < NL; ++j) result.d[j] = static_cast<T>(m[k][j] * scale * g[j]);
      } else {
        result = round_pack<T>(m[k]);
      }
      if (f.in[k] != 0.0f) o[base + f.at[k]] = result;
    }
  }

  // 3. No rank may overwrite its input until every peer has read it.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::read>(p);
}

}  // namespace hip_comms
