// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce then RMSNorm (`all_reduce_pull_one_shot_rms_norm`), and
// all-reduce then add then RMSNorm (`all_reduce_pull_one_shot_add_rms_norm`): one body, a
// kernel per op, so a trace names the op that ran.

#pragma once

#include "fusions/add_rms_norm.cuh"
#include "p2p/p2p.cuh"
#include "common/memory.cuh"
#include "common/reduce.cuh"

namespace hip_comms {

// Every rank reduces every row and norms it where the sum lands in registers, saving an
// HBM round trip and a launch against an all-reduce then a norm kernel. A block owns a
// row. `residual` and `residual_out` are unused (null) unless kAdd; `weight` is in its
// own dtype W, T or fp32 (see `fusion::row`).
template <typename T, typename W, int ngpus, bool kAdd, int kRowPacks>
DINLINE void all_reduce_pull_one_shot_add_rms_norm_body(p2p::DevComm p, T* __restrict__ out,
                                                        T* __restrict__ residual_out,
                                                        const T* __restrict__ residual,
                                                        const W* __restrict__ weight, float eps,
                                                        int rows, int packs) {
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  namespace fusion       = fusions::add_rms_norm;
  const V* res_in        = reinterpret_cast<const V*>(residual);
  const auto* wv         = reinterpret_cast<const vec<W, NL>*>(weight);
  V* res_out             = reinterpret_cast<V*>(residual_out);
  V* o                   = reinterpret_cast<V*>(out);
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto peers = p2p::peers<T, ngpus>(p);
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);
  const auto read = [&](int r, int64_t i) { return p2p::read_input(peers[r], i); };

  // 2. Each of this block's rows: read it from every rank in rank order, sum, norm.
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    V sum[kRowPacks];
    sum_row<T, ngpus>(read, row, packs, sum);
    fusion::row<T, W, kAdd>(
        sum, res_in, wv, row, packs, inv_hidden, eps,
        [&](int, int i, const V& v) { res_out[row * packs + i] = v; },
        [&](int, int i, const V& v) { o[row * packs + i] = v; });
  }

  // 3. No rank may overwrite its input until every peer has read it.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::read>(p);
}

// THE KERNELS, one per op, both the body above.
template <typename T, typename W, int ngpus, int kRowPacks>
__global__ void __launch_bounds__(kMaxThreads, 1) all_reduce_pull_one_shot_rms_norm(
    p2p::DevComm p, T* __restrict__ out, const W* __restrict__ weight, float eps, int rows,
    int packs) {
  all_reduce_pull_one_shot_add_rms_norm_body<T, W, ngpus, false, kRowPacks>(
      p, out, nullptr, nullptr, weight, eps, rows, packs);
}

template <typename T, typename W, int ngpus, int kRowPacks>
__global__ void __launch_bounds__(kMaxThreads, 1) all_reduce_pull_one_shot_add_rms_norm(
    p2p::DevComm p, T* __restrict__ out, T* __restrict__ residual_out,
    const T* __restrict__ residual, const W* __restrict__ weight, float eps, int rows,
    int packs) {
  all_reduce_pull_one_shot_add_rms_norm_body<T, W, ngpus, true, kRowPacks>(
      p, out, residual_out, residual, weight, eps, rows, packs);
}

}  // namespace hip_comms
