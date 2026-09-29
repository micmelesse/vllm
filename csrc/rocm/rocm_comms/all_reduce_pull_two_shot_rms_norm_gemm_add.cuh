// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce, then RMSNorm, then a GEMM whose result is added into an
// output: the tail of Kimi-K3's latent MoE (`fused_all_reduce.latent_tail`).

#pragma once

#include "fusions/rms_norm_gemm_add.cuh"
#include "p2p/p2p.cuh"
#include "utils.cuh"

namespace hip_comms {

// Each rank reduces and norms the rows it owns and shares them; after the barrier every
// rank gathers every normed row into `workspace` ([rows, packs] of its own); a grid
// barrier; the GEMM over every row, fusion::kRows per pass. The input is read only
// before the barrier, so no close.
template <typename T, int ngpus, int kLanesPerCol>
__global__ void __launch_bounds__(kMaxThreads, 1) all_reduce_pull_two_shot_rms_norm_gemm_add(
    p2p::Peers p, const T* __restrict__ norm_w, float eps, const T* __restrict__ gemm_w,
    int n_cols, T* __restrict__ out, int64_t out_stride, int out_col0,
    T* __restrict__ workspace, int rows, int packs) {
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  namespace fusion       = fusions::rms_norm_gemm_add;
  const auto w           = p2p::start<T, ngpus>(p);
  const auto tiling      = tiles::rows(rows, packs, ngpus);
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  const V* weight        = reinterpret_cast<const V*>(norm_w);
  V* normed              = reinterpret_cast<V*>(workspace);
  const auto slot        = p2p::pull::slot(w, tiling);

  for (int row = tiling.first(p.rank); row < tiling.end(p.rank); row = tiling.next(row)) {
    V sum[kMaxRowPacks];
    p2p::pull::reduce(w, tiling, row, sum);
    V n[kMaxRowPacks] = {};
    fusion::norm_row<T>(sum, weight, packs, inv_hidden, eps,
                        [&](int k, int, const V& v) { n[k] = v; });
    p2p::pull::share(w, slot, tiling, row, n);
  }

  p2p::peer_barrier(w);

  p2p::pull::gather(w, slot, tiling, [&](int row, int k, const V& v) {
    store_global(normed + tiling.pos(row, k), v);
  });

  // The GEMM reads rows other blocks of this rank gathered.
  p2p::grid_barrier(w);

  for (int r0 = 0; r0 < rows; r0 += fusion::kRows) {
    fusion::gemm<kLanesPerCol, T>([&](int r) { return normed + (r0 + r) * packs; },
                                  min(fusion::kRows, rows - r0), gemm_w, n_cols, packs,
                                  out + r0 * out_stride, out_stride, out_col0);
  }
}

}  // namespace hip_comms
