// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot all-reduce, then RMSNorm, then a GEMM whose result is added into an output: the
// tail of Kimi-K3's latent MoE (`fused_all_reduce.latent_tail`).

#pragma once

#include "ipc.cuh"
#include "utils.cuh"

namespace hip_comms {

// Matching the model's ops rounding for rounding where they round:
//
//   n   = rms_norm(T(sum over ranks), norm_w, eps)   `add_rms_norm_row`, landing as T
//   out[:, col0:col0+N] = T(float(out) + n @ W^T)   `gemm_add_rows`
//
// At most kGemmRows rows: one GEMM pass.
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) allreduce_one_shot_rms_norm_gemm_add(
    ipc::Peers p, const T* __restrict__ norm_w, float eps, const T* __restrict__ gemm_w,
    int n_cols, T* __restrict__ out, int64_t out_stride, int out_col0, int rows,
    int packs) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  ipc::Comm<T, ngpus> c(p);
  const int rank = c.rank();

  // PHASE 1 -- this block's rows, reduced and normed, into our own scratch.
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  for (int row = blockIdx.x; row < rows; row += gridDim.x)
    add_rms_norm_row<T, T, false>(
        c, nullptr, reinterpret_cast<const V*>(norm_w), row, packs, inv_hidden, eps,
        [](int, const V&) {}, [&](int i, const V& v) { c.put(rank, row * packs + i, v); });

  // Phase 2 reads rows other blocks wrote.
  c.sync();

  gemm_add_rows<T>([&](int r, int k) { return c.get(rank, r * packs + k); }, rows, gemm_w,
                   n_cols, packs, out, out_stride, out_col0);
  c.close();
}

}  // namespace hip_comms
