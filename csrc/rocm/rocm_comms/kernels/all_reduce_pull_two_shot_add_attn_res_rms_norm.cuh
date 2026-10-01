// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce fused with Kimi-K3's attention residual (AttnRes) and its
// RMSNorm.

#pragma once

#include "../p2p/p2p.cuh"
#include "../common/common.cuh"

namespace hip_comms {

// Each rank owns whole rows: it reduces them, updates the prefix, runs AttnRes and leaves the
// out and prefix rows in its scratch, row-major; after the sync every rank copies every owner's
// rows into `out`, `prefix` and the written block. A row's replicated `prefix` and `blocks` are
// read and then overwritten by the one block that takes it, so the order is the block's own. THE
// SAME BLOCK AND THREAD INDEX A PACK IN BOTH PHASES: after the sync a block may read only what
// the same block on a peer wrote.
template <typename T, int ngpus, bool kPrefix, int kRowPacks>
__global__ void __launch_bounds__(kMaxThreads, 1)
    all_reduce_pull_two_shot_add_attn_res_rms_norm(
        p2p::DevComm p, T* __restrict__ prefix, T* __restrict__ blocks,
        int64_t block_stride_m, int64_t block_stride_r, const T* __restrict__ norm_w,
        const T* __restrict__ qk_w, const T* __restrict__ out_norm_w, T* __restrict__ out,
        int num_blocks, int write_idx, float eps, float out_eps, int rows, int packs) {
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  V* pre                 = reinterpret_cast<V*>(prefix);
  V* o                   = reinterpret_cast<V*>(out);
  const int slice_rows   = (rows + ngpus - 1) / ngpus;
  const int64_t pre_at   = int64_t{slice_rows} * packs;  // the prefix rows, after the out rows
  const auto f           = fragment<kRowPacks>(packs);
  // The block row `row` writes, or none.
  auto written = [&](int row) -> V* {
    return write_idx < 0 ? nullptr
                         : reinterpret_cast<V*>(blocks + row * block_stride_m +
                                                write_idx * block_stride_r);
  };

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto self = p2p::self<T, ngpus>(p);
  const auto peers = p2p::peers<T, ngpus>(p);
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);
  const auto read = [&](int r, int64_t i) { return p2p::read_input(peers[r], i); };

  // 2. This rank's rows: read each from every rank in rank order, sum, AttnRes, and leave the
  //    out and prefix rows in this rank's scratch.
  const int first = p.rank * slice_rows;
  const int last  = min(first + slice_rows, rows);
  for (int row = first + blockIdx.x; row < last; row += gridDim.x) {
    const int64_t base = int64_t{row} * packs;
    const int64_t at   = int64_t{row - first} * packs;
    V sum[kRowPacks];
    peers_reduce(peers_load<T, ngpus>(read, row, packs, f), sum);
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
      thread_unpack<T>(new_prefix[k], u[k]);
      if (f.in[k] != 0.0f) p2p::write_scratch(self, pre_at + at + f.at[k], new_prefix[k]);
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
      float max_logit = -INFINITY, denominator = 0.0f;
      for (int src = 0; src <= num_blocks; ++src) {
        // The stored blocks first, the prefix last, as the reference orders its sources.
        float v[kRowPacks][NL];
        if (src < num_blocks) {
          const V* at_src = reinterpret_cast<const V*>(row_blocks + src * block_stride_r);
#pragma unroll
          for (int k = 0; k < kRowPacks; ++k) thread_unpack<T>(at_src[f.at[k]], v[k]);
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
        thread_unpack<T>(reinterpret_cast<const V*>(out_norm_w)[f.at[k]], g);
#pragma unroll
        for (int j = 0; j < NL; ++j) result.d[j] = static_cast<T>(m[k][j] * scale * g[j]);
      } else {
        result = thread_pack<T>(m[k]);
      }
      if (f.in[k] != 0.0f) p2p::write_scratch(self, at + f.at[k], result);
    }
  }

  // 3. Every rank's rows are visible to its peers.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::visible>(p);

  // 4. Every owner's rows out of its scratch, into `out`, `prefix` and the written block. The
  //    next call's first sync keeps a rank from overwriting its scratch while it is read.
  for (int l = blockIdx.x; l < slice_rows; l += gridDim.x) {
    for (int i = threadIdx.x; i < packs; i += blockDim.x) {
      // EVERY OWNER'S PACK LOADED BEFORE ANY IS STORED, and every load unconditional (each rank's
      // scratch holds slice_rows rows, so a slot past the last row is real): a store between two
      // loads, or a load under an `if`, made the owners' round trips run one after another.
      const int64_t at = int64_t{l} * packs + i;
      V got[ngpus], got_pre[ngpus];
#pragma unroll
      for (int r = 0; r < ngpus; ++r) {
        got[r]     = p2p::read_scratch(peers[r], at);
        got_pre[r] = p2p::read_scratch(peers[r], pre_at + at);
      }
#pragma unroll
      for (int r = 0; r < ngpus; ++r) {
        const int row = r * slice_rows + l;
        if (row >= rows) continue;
        thread_store(o + int64_t{row} * packs + i, got[r]);
        thread_store(pre + int64_t{row} * packs + i, got_pre[r]);
        if (V* dst = written(row)) thread_store(dst + i, got_pre[r]);
      }
    }
  }
}

}  // namespace hip_comms
