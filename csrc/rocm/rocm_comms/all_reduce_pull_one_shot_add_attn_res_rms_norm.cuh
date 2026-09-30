// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce fused with Kimi-K3's attention residual (AttnRes) and its
// RMSNorm.

#pragma once

#include "p2p/p2p.cuh"
#include "common/memory.cuh"
#include "common/reduce.cuh"
#include "common/row.cuh"
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
  const auto sh          = share<kRowPacks>(packs);
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
    sum_row<T, ngpus>(read, row, packs, sh, sum);
    // The AttnRes, rounding as `vllm/models/kimi_k3/amd/ops/attn_res.py` does:
    //   d = float(T(sum over ranks)); u = kPrefix ? float(T(float(prefix) + d)) : d (the prefix)
    //   logit(src) = dot(src, norm_w * qk_w) * rsqrt(mean(src^2) + eps), src the blocks, then u
    //   m = softmax(logits) . sources; out = T(m), or T(m * rsqrt(mean(m^2) + out_eps) * out_w)
    // Every source's loads are branch-free (row.cuh), so they are in flight together; each source
    // is folded into its sums as it lands and not kept (holding nine spilled); ONE block reduction
    // takes every source's sums; the mix reads the sources again (just read, so from cache).
    const T* row_blocks = blocks + int64_t{row} * block_stride_m;
    float u[kRowPacks][NL];
    V new_prefix[kRowPacks];
    if constexpr (kPrefix) {
      V old_prefix[kRowPacks];
      load(pre + base, sh, old_prefix);
#pragma unroll
      for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
        for (int j = 0; j < NL; ++j) {
          new_prefix[k].d[j] = static_cast<T>(static_cast<float>(old_prefix[k].d[j]) +
                                              static_cast<float>(sum[k].d[j]));
          u[k][j]            = static_cast<float>(new_prefix[k].d[j]);
        }
    } else {
#pragma unroll
      for (int k = 0; k < kRowPacks; ++k) {
        new_prefix[k] = sum[k];
        unpack<T>(sum[k], u[k]);
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
      {
        V a[kRowPacks], b[kRowPacks];
        load(reinterpret_cast<const V*>(norm_w), sh, a);
        load(reinterpret_cast<const V*>(qk_w), sh, b);
#pragma unroll
        for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
          for (int j = 0; j < NL; ++j)
            w[k][j] = sh.in[k] * static_cast<float>(a[k].d[j]) * static_cast<float>(b[k].d[j]);
      }
      // Every source's sum of squares and weighted dot: the stored blocks in slots
      // 0..num_blocks-1 (a slot past them reads the last one again, weighted zero), the prefix in
      // the last slot.
      const int last = num_blocks - 1;
      float sums[2 * kAttnResMaxSources];
#pragma unroll
      for (int src = 0; src < kAttnResMaxSources - 1; ++src) {
        const int slot  = src < last ? src : last;
        const V* at_src = reinterpret_cast<const V*>(row_blocks + slot * block_stride_r);
        const float on  = src < num_blocks ? 1.0f : 0.0f;
        V x[kRowPacks];
        load(at_src, sh, x);
        float ss = 0.0f, dot = 0.0f;
#pragma unroll
        for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
          for (int j = 0; j < NL; ++j) {
            const float v = on * sh.in[k] * static_cast<float>(x[k].d[j]);
            ss += v * v;
            dot += v * w[k][j];
          }
        sums[2 * src]     = ss;
        sums[2 * src + 1] = dot;
      }
      {
        float ss = 0.0f, dot = 0.0f;
#pragma unroll
        for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
          for (int j = 0; j < NL; ++j) {
            const float v = sh.in[k] * u[k][j];
            ss += v * v;
            dot += v * w[k][j];
          }
        sums[2 * (kAttnResMaxSources - 1)]     = ss;
        sums[2 * (kAttnResMaxSources - 1) + 1] = dot;
      }
      block_sum(sums);
      // The softmax over the live slots (the stored blocks and the prefix).
      float logit[kAttnResMaxSources];
      float max_logit = -INFINITY;
#pragma unroll
      for (int src = 0; src < kAttnResMaxSources; ++src) {
        logit[src] = sums[2 * src + 1] * rsqrtf(sums[2 * src] * inv_hidden + eps);
        const bool live = src < num_blocks || src == kAttnResMaxSources - 1;
        if (live) max_logit = fmaxf(max_logit, logit[src]);
      }
      float denominator = 0.0f;
#pragma unroll
      for (int src = 0; src < kAttnResMaxSources; ++src) {
        const bool live = src < num_blocks || src == kAttnResMaxSources - 1;
        logit[src]      = live ? __expf(logit[src] - max_logit) : 0.0f;
        denominator += logit[src];
      }
      const float inv_den = 1.0f / denominator;
      // The mix, the stored sources read again.
#pragma unroll
      for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
        for (int j = 0; j < NL; ++j) m[k][j] = logit[kAttnResMaxSources - 1] * u[k][j];
#pragma unroll
      for (int src = 0; src < kAttnResMaxSources - 1; ++src) {
        const int slot  = src < last ? src : last;
        const V* at_src = reinterpret_cast<const V*>(row_blocks + slot * block_stride_r);
        V x[kRowPacks];
        load(at_src, sh, x);
#pragma unroll
        for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
          for (int j = 0; j < NL; ++j) m[k][j] += logit[src] * static_cast<float>(x[k].d[j]);
      }
#pragma unroll
      for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
        for (int j = 0; j < NL; ++j) m[k][j] *= inv_den;
    }
    // The output, normed when out_norm_w is given.
    V result[kRowPacks];
    if (out_norm_w != nullptr) {
      float ss[1] = {0.0f};
#pragma unroll
      for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
        for (int j = 0; j < NL; ++j) ss[0] += sh.in[k] * m[k][j] * m[k][j];
      block_sum(ss);
      const float scale = rsqrtf(ss[0] * inv_hidden + out_eps);
      V g[kRowPacks];
      load(reinterpret_cast<const V*>(out_norm_w), sh, g);
#pragma unroll
      for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
        for (int j = 0; j < NL; ++j)
          result[k].d[j] = static_cast<T>(m[k][j] * scale * static_cast<float>(g[k].d[j]));
    } else {
#pragma unroll
      for (int k = 0; k < kRowPacks; ++k) result[k] = round_pack<T>(m[k]);
    }
    store(pre + base, sh, new_prefix);
    if (V* dst = written(row)) store(dst, sh, new_prefix);
    store(o + base, sh, result);
  }

  // 3. No rank may overwrite its input until every peer has read it.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::read>(p);
}

}  // namespace hip_comms
