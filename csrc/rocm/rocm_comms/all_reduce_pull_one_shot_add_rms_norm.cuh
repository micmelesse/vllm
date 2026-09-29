// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce then RMSNorm (`all_reduce_pull_one_shot_rms_norm`), and
// all-reduce then add then RMSNorm (`all_reduce_pull_one_shot_add_rms_norm`): one body, a
// kernel per op, so a trace names the op that ran.

#pragma once

#include "fusions/add_rms_norm.cuh"
#include "p2p/p2p.cuh"
#include "utils.cuh"

namespace hip_comms {

// Every rank reduces every row and norms it where the sum lands in registers, saving an
// HBM round trip and a launch against an all-reduce then a norm kernel. A block owns a
// row. `residual` and `residual_out` are unused (null) unless kAdd; `weight` is in its
// own dtype W, T or fp32 (see `fusion::row`). Peers read this rank's input to the end:
// close.
template <typename T, typename W, int ngpus, bool kAdd>
DINLINE void all_reduce_pull_one_shot_add_rms_norm_body(p2p::Peers p, T* __restrict__ out,
                                             T* __restrict__ residual_out,
                                             const T* __restrict__ residual,
                                             const W* __restrict__ weight, float eps,
                                             int rows, int packs) {
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  namespace fusion       = fusions::add_rms_norm;
  const auto w           = p2p::start<T, ngpus>(p);
  const auto tiling      = tiles::rows(rows, packs, ngpus);
  const V* res_in        = reinterpret_cast<const V*>(residual);
  const auto* wv         = reinterpret_cast<const vec<W, NL>*>(weight);
  V* res_out             = reinterpret_cast<V*>(residual_out);
  V* o                   = reinterpret_cast<V*>(out);
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  for (int row = tiling.first(); row < tiling.end(); row = tiling.next(row)) {
    V sum[kMaxRowPacks];
    p2p::pull::reduce(w, tiling, row, sum);
    fusion::row<T, W, kAdd>(
        sum, res_in, wv, row, packs, inv_hidden, eps,
        [&](int, int i, const V& v) { res_out[row * packs + i] = v; },
        [&](int, int i, const V& v) { o[row * packs + i] = v; });
  }
  p2p::close(w);
}

// THE KERNELS, one per op, both the body above.
template <typename T, typename W, int ngpus>
__global__ void __launch_bounds__(kMaxThreads, 1) all_reduce_pull_one_shot_rms_norm(
    p2p::Peers p, T* __restrict__ out, const W* __restrict__ weight, float eps, int rows,
    int packs) {
  all_reduce_pull_one_shot_add_rms_norm_body<T, W, ngpus, false>(p, out, nullptr, nullptr,
                                                                 weight, eps, rows, packs);
}

template <typename T, typename W, int ngpus>
__global__ void __launch_bounds__(kMaxThreads, 1) all_reduce_pull_one_shot_add_rms_norm(
    p2p::Peers p, T* __restrict__ out, T* __restrict__ residual_out,
    const T* __restrict__ residual, const W* __restrict__ weight, float eps, int rows,
    int packs) {
  all_reduce_pull_one_shot_add_rms_norm_body<T, W, ngpus, true>(p, out, residual_out, residual,
                                                                weight, eps, rows, packs);
}

}  // namespace hip_comms
