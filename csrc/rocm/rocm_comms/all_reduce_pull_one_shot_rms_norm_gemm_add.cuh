// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce, then RMSNorm, then a GEMM whose result is added into an
// output: the tail of Kimi-K3's latent MoE (`fused_all_reduce.latent_tail`).

#pragma once

#include "p2p/p2p.cuh"
#include "common/dot.cuh"
#include "common/elementwise.cuh"
#include "common/memory.cuh"
#include "common/reduce.cuh"
#include "common/utils.cuh"

namespace hip_comms {

// Every rank reduces and norms every row into `workspace` ([rows, packs] of its own); a
// grid barrier; the GEMM over every row. At most kGemmRows rows: one GEMM pass.
// kLanesPerCol is the GEMM's lanes per column (tune.cuh).
template <typename T, int ngpus, int kLanesPerCol, int kRowPacks>
__global__ void __launch_bounds__(gemm_max_threads(kLanesPerCol), 1)
    all_reduce_pull_one_shot_rms_norm_gemm_add(
    p2p::DevComm p, const T* __restrict__ norm_w, float eps, const T* __restrict__ gemm_w,
    int n_cols, T* __restrict__ out, int64_t out_stride, int out_col0,
    T* __restrict__ workspace, int rows, int packs) {
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  const V* weight        = reinterpret_cast<const V*>(norm_w);
  V* normed              = reinterpret_cast<V*>(workspace);
  const auto f           = fragment<kRowPacks>(packs);

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto peers = p2p::peers<T, ngpus>(p);
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);
  const auto read = [&](int r, int64_t i) { return p2p::read_input(peers[r], i); };

  // 2. Each of this block's rows: read it from every rank in rank order, sum, norm, into the
  //    workspace.
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t base = int64_t{row} * packs;
    V sum[kRowPacks];
    peers_reduce<T, ngpus>(read, row, packs, f, sum);
    // The norm, rounding as vLLM's reference rms_norm does (weight in T):
    //   out = T(T(s * rsqrt(mean(s^2) + eps)) * float(w)), s = float(T(sum over ranks))
    float s[kRowPacks][NL];
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) thread_unpack<T>(sum[k], s[k]);
    float ss[1] = {thread_dot(s, s, f)};
    block_reduce<Sum>(ss);
    const float scale = rsqrtf(ss[0] * inv_hidden + eps);
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) {
      float w[NL], x[NL];
      thread_unpack<T>(weight[f.at[k]], w);
#pragma unroll
      for (int j = 0; j < NL; ++j)
        x[j] = static_cast<float>(static_cast<T>(s[k][j] * scale)) * w[j];
      if (f.in[k] != 0.0f) thread_store(normed + base + f.at[k], thread_pack<T>(x));
    }
  }

  // 3. The GEMM reads rows other blocks of this rank wrote.
  p2p::barrier<ngpus, p2p::Among::grid, p2p::Ensure::visible>(p);

  // 4. The GEMM over every row.
  grid_gemm<kLanesPerCol, T>([&](int r) { return normed + r * packs; }, rows, gemm_w, n_cols,
                             packs, out, out_stride, out_col0);

  // 5. No rank may overwrite its input until every peer has read it.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::read>(p);
}

}  // namespace hip_comms
