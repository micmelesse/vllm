// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce, then RMSNorm, then a GEMM whose result is added into an
// output: the tail of Kimi-K3's latent MoE (`fused_all_reduce.latent_tail`).

#pragma once

#include "fusions/rms_norm_gemm_add.cuh"
#include "p2p/p2p.cuh"
#include "common/memory.cuh"
#include "common/reduce.cuh"

namespace hip_comms {

// Every rank reduces and norms every row into `workspace` ([rows, packs] of its own); a
// grid barrier; the GEMM over every row. At most fusion::kRows rows: one GEMM pass.
// kLanesPerCol is the GEMM's lanes per column (tune.cuh).
template <typename T, int ngpus, int kLanesPerCol, int kRowPacks>
__global__ void __launch_bounds__(fusions::rms_norm_gemm_add::max_threads(kLanesPerCol), 1)
    all_reduce_pull_one_shot_rms_norm_gemm_add(
    p2p::Peers p, const T* __restrict__ norm_w, float eps, const T* __restrict__ gemm_w,
    int n_cols, T* __restrict__ out, int64_t out_stride, int out_col0,
    T* __restrict__ workspace, int rows, int packs) {
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  namespace fusion       = fusions::rms_norm_gemm_add;
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  const V* weight        = reinterpret_cast<const V*>(norm_w);
  V* normed              = reinterpret_cast<V*>(workspace);

  // 1. Wait until every peer has launched, so its input is ready.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);

  // 2. Each of this block's rows: read it from every rank in rank order, sum, norm, into the
  //    workspace.
  const auto ranks = p2p::ranks<T, ngpus>(p);
  const auto read  = [&](int r, int64_t i) { return p2p::read_input(ranks, r, i); };
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    V sum[kRowPacks];
    sum_row<T, ngpus>(read, row, packs, sum);
    fusion::norm_row<T>(sum, weight, packs, inv_hidden, eps, [&](int, int i, const V& v) {
      store_global(normed + row * packs + i, v);
    });
  }

  // 3. The GEMM reads rows other blocks of this rank wrote.
  p2p::barrier<ngpus, p2p::Among::grid, p2p::Ensure::visible>(p);

  // 4. The GEMM over every row.

  fusion::gemm<kLanesPerCol, T>([&](int r) { return normed + r * packs; }, rows, gemm_w,
                                n_cols, packs, out, out_stride, out_col0);

  // 5. No rank may overwrite its input until every peer has read it.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::read>(p);
}

}  // namespace hip_comms
