// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot push all-reduce, then RMSNorm, then a GEMM whose result is added into an
// output: the tail of Kimi-K3's latent MoE (`fused_all_reduce.latent_tail`).

#pragma once

#include "fusions/rms_norm_gemm_add.cuh"
#include "p2p/p2p.cuh"
#include "common/memory.cuh"
#include "hardware.cuh"

namespace hip_comms {

// Every rank's rows, encoded by kBits' codec, into every rank's slot; one barrier; each
// block reduces and norms its rows into `workspace`; a grid barrier; the GEMM over every
// row. At most fusion::kRows rows: one GEMM pass.
template <typename T, int ngpus, int kBits, int kLanesPerCol>
__global__ void __launch_bounds__(kMaxThreads, 1) all_reduce_push_one_shot_rms_norm_gemm_add(
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
  const auto slot        = p2p::push::slot<kBits>(w, tiling, p2p::To::all);

  p2p::push::scatter(w, slot, tiling);

  p2p::peer_barrier(w);

  for (int row = tiling.first(); row < tiling.end(); row = tiling.next(row)) {
    V sum[p2p::kPushGroupPacks];
    p2p::push::reduce(w, slot, tiling, row, sum);
    fusion::norm_row<T>(sum, weight, packs, inv_hidden, eps, [&](int, int i, const V& v) {
      store_global(normed + row * packs + i, v);
    });
  }

  // The GEMM reads rows other blocks of this rank wrote.
  p2p::grid_barrier(w);

  fusion::gemm<kLanesPerCol, T>([&](int r) { return normed + r * packs; }, rows, gemm_w,
                                n_cols, packs, out, out_stride, out_col0);
}

}  // namespace hip_comms
