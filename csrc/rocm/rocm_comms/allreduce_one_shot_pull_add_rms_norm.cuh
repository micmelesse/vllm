// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce then RMSNorm (`allreduce_one_shot_pull_rms_norm`), and
// all-reduce then add then RMSNorm (`allreduce_one_shot_pull_add_rms_norm`): one body, a
// kernel per op, so a trace names the op that ran.

#pragma once

#include "fusions/add_rms_norm.cuh"
#include "ipc.cuh"
#include "utils.cuh"

namespace hip_comms {

// The sum is already in registers when one-shot is about to store it, so normalising there
// saves an HBM round trip and a launch against an all-reduce followed by a norm kernel.
//
// A BLOCK OWNS A ROW, because the variance needs the whole row: one block per row,
// striding over rows, where plain one-shot is grid-stride over the flat buffer.
// `residual` and `residual_out` are unused (null) unless kAdd.
// `weight` is in its own dtype W: T, or fp32 (see `add_rms_norm_row`).
template <typename T, typename W, int ngpus, bool kAdd>
DINLINE void one_shot_pull_add_rms_norm_body(ipc::Peers p, T* __restrict__ out,
                                             T* __restrict__ residual_out,
                                             const T* __restrict__ residual,
                                             const W* __restrict__ weight, float eps,
                                             int rows, int packs) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  ipc::Comm<T, ngpus> c(p);
  const V* res_in        = reinterpret_cast<const V*>(residual);
  const auto* w          = reinterpret_cast<const vec<W, NL>*>(weight);
  V* res_out             = reinterpret_cast<V*>(residual_out);
  V* o                   = reinterpret_cast<V*>(out);
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  // Uniform across the block, so every `__syncthreads` inside is reached by every thread.
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    V sum[kMaxRowPacks];
    c.sum_row(row * packs, packs, sum);
    add_rms_norm_row<T, W, kAdd>(
        sum, res_in, w, row, packs, inv_hidden, eps,
        [&](int, int i, const V& v) { res_out[row * packs + i] = v; },
        [&](int, int i, const V& v) { o[row * packs + i] = v; });
  }
  c.close();
}

// THE KERNELS, one per op, both the body above.
template <typename T, typename W, int ngpus>
__global__ void __launch_bounds__(kMaxThreads, 1) allreduce_one_shot_pull_rms_norm(
    ipc::Peers p, T* __restrict__ out, const W* __restrict__ weight, float eps, int rows,
    int packs) {
  one_shot_pull_add_rms_norm_body<T, W, ngpus, false>(p, out, nullptr, nullptr, weight, eps,
                                                      rows, packs);
}

template <typename T, typename W, int ngpus>
__global__ void __launch_bounds__(kMaxThreads, 1) allreduce_one_shot_pull_add_rms_norm(
    ipc::Peers p, T* __restrict__ out, T* __restrict__ residual_out,
    const T* __restrict__ residual, const W* __restrict__ weight, float eps, int rows,
    int packs) {
  one_shot_pull_add_rms_norm_body<T, W, ngpus, true>(p, out, residual_out, residual, weight,
                                                     eps, rows, packs);
}

}  // namespace hip_comms
