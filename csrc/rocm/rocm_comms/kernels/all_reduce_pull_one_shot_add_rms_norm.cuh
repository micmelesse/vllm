// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce then RMSNorm (`all_reduce_pull_one_shot_rms_norm`), and
// all-reduce then add then RMSNorm (`all_reduce_pull_one_shot_add_rms_norm`): one body, a
// kernel per op, so a trace names the op that ran.

#pragma once

#include "../p2p/p2p.cuh"
#include "../common/common.cuh"

namespace hip_comms {

// Every rank reduces every row and norms it where the sum lands in registers, saving an HBM round
// trip and a launch against an all-reduce then a norm kernel. A block owns a row. `residual` and
// `residual_out` are unused (null) unless kAdd; `weight` keeps its own dtype W (T or fp32), as
// vLLM's reference ops round to the WEIGHT's dtype (`vllm/ir/ops/layernorm.py`).
template <typename T, typename W, int ngpus, bool kAdd, int kRowPacks>
DINLINE void all_reduce_pull_one_shot_add_rms_norm_body(p2p::DevComm p, T* __restrict__ out,
                                                        T* __restrict__ residual_out,
                                                        const T* __restrict__ residual,
                                                        const W* __restrict__ weight, float eps,
                                                        int rows, int packs) {
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  const V* res_in        = reinterpret_cast<const V*>(residual);
  const auto* wv         = reinterpret_cast<const vec<W, NL>*>(weight);
  V* res_out             = reinterpret_cast<V*>(residual_out);
  V* o                   = reinterpret_cast<V*>(out);
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  const int cols = packs * NL;  // the row, in elements
  const auto thread_cols = thread_offs<T>(Tile<1, kRowPacks>{rows, cols, 0, 0});

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto inputs = p2p::inputs<T, ngpus>(p);
  block_stamp(0);
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);
  block_stamp(1);
  const auto read = [&](int r, int64_t i) { return p2p::read_input(inputs[r], i); };

  // 2. Each of this block's rows: read it from every rank in rank order and sum, then (kAdd) add
  //    the residual, then RMSNorm, rounding as the reference does:
  //      s   = float(T(sum over ranks))             the all-reduce output, as it would land
  //      s  += float(residual); residual_out = T(s) kAdd only (fused_add_rms_norm)
  //      out = T(W(W(s * rsqrt(mean(s^2) + eps)) * float(w)))
  //    The variance is of `s` before any further rounding, kept in registers between the passes.
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t base = int64_t{row} * packs;
    V sum[kRowPacks];
    peers_reduce(peers_load<T, ngpus>(read, row, packs, thread_cols), sum);
    block_stamp(2);
    float s[kRowPacks][NL];
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) {
      thread_unpack<T>(sum[k], s[k]);
      if constexpr (kAdd) {
        float r[NL];
        thread_unpack<T>(res_in[base + thread_cols.offs_n[k]], r);
#pragma unroll
        for (int j = 0; j < NL; ++j) s[k][j] += r[j];
        if (thread_cols.mask_n[k] != 0.0f)
          res_out[base + thread_cols.offs_n[k]] = thread_pack<T>(s[k]);
      }
    }
    float ss[1] = {thread_dot(s, s, thread_cols)};
    block_reduce<Sum>(ss);
    block_stamp(3);
    const float scale = rsqrtf(ss[0] * inv_hidden + eps);
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) {
      const vec<W, NL> w = wv[thread_cols.offs_n[k]];
      V normed;
#pragma unroll
      for (int j = 0; j < NL; ++j) {
        const float x = static_cast<float>(static_cast<W>(s[k][j] * scale));
        normed.d[j]   = static_cast<T>(static_cast<W>(x * static_cast<float>(w.d[j])));
      }
      if (thread_cols.mask_n[k] != 0.0f) o[base + thread_cols.offs_n[k]] = normed;
    }
  }

  block_stamp(4);
  // 3. No rank may overwrite its input until every peer has read it.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::read>(p);
  block_stamp(5);
}

// THE KERNELS, one per op, both the body above.
template <typename T, typename W, int ngpus, int kRowPacks>
__global__ void __launch_bounds__(kBuild.kernels.max_threads, 1)
    all_reduce_pull_one_shot_rms_norm(p2p::DevComm p, T* __restrict__ out,
                                      const W* __restrict__ weight, float eps, int rows,
                                      int packs) {
  all_reduce_pull_one_shot_add_rms_norm_body<T, W, ngpus, false, kRowPacks>(
      p, out, nullptr, nullptr, weight, eps, rows, packs);
}

template <typename T, typename W, int ngpus, int kRowPacks>
__global__ void __launch_bounds__(kBuild.kernels.max_threads, 1)
    all_reduce_pull_one_shot_add_rms_norm(p2p::DevComm p, T* __restrict__ out,
                                          T* __restrict__ residual_out,
                                          const T* __restrict__ residual,
                                          const W* __restrict__ weight, float eps, int rows,
                                          int packs) {
  all_reduce_pull_one_shot_add_rms_norm_body<T, W, ngpus, true, kRowPacks>(
      p, out, residual_out, residual, weight, eps, rows, packs);
}

}  // namespace hip_comms
