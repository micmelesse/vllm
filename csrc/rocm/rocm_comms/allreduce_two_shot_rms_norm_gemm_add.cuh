// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot all-reduce, then RMSNorm, then a GEMM whose result is added into an output: the
// tail of Kimi-K3's latent MoE (`fused_all_reduce.latent_tail`).

#pragma once

#include "ipc.cuh"
#include "utils.cuh"

namespace hip_comms {

// Each rank owns ceil(rows/ngpus) whole rows: it reduces and norms them into its own
// scratch, and after the world barrier every rank reads every normed row from its owner's
// scratch for the GEMM, kGemmRows rows per pass. The same roundings as the one-shot
// kernel.
// kLanesPerCol is the GEMM's lanes per column, tuned in launch.cuh.
template <typename T, int ngpus, int kLanesPerCol>
__global__ void __launch_bounds__(kMaxThreads, 1) allreduce_two_shot_rms_norm_gemm_add(
    ipc::Peers p, const T* __restrict__ norm_w, float eps, const T* __restrict__ gemm_w,
    int n_cols, T* __restrict__ out, int64_t out_stride, int out_col0, int rows,
    int packs) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  ipc::Comm<T, ngpus> c(p);
  const int rank  = c.rank();
  const int chunk = (rows + ngpus - 1) / ngpus;

  {
    const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
    const int begin        = rank * chunk;
    const int end          = min(begin + chunk, rows);
    for (int row = begin + blockIdx.x; row < end; row += gridDim.x) {
      const int local = (row - begin) * packs;
      add_rms_norm_row<T, T, false>(
          c, nullptr, reinterpret_cast<const V*>(norm_w), row, packs, inv_hidden, eps,
          [](int, const V&) {}, [&](int i, const V& v) { c.put(rank, local + i, v); });
    }
  }

  c.world_barrier();

  for (int r0 = 0; r0 < rows; r0 += kGemmRows) {
    gemm_add_rows<kLanesPerCol, T>(
        [&](int r) {
          const int row   = r0 + r;
          const int owner = row / chunk;
          return c.ptr(owner, (row - owner * chunk) * packs, packs);
        },
        min(kGemmRows, rows - r0), gemm_w, n_cols, packs, out + r0 * out_stride,
        out_stride, out_col0);
  }
  c.close();
}

}  // namespace hip_comms
