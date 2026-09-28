// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot push all-reduce then RMSNorm (`allreduce_one_shot_push_rms_norm`), and
// all-reduce then add then RMSNorm (`allreduce_one_shot_push_add_rms_norm`): one body, a
// kernel per op, so a trace names the op that ran.

#pragma once

#include "ipc.cuh"
#include "utils.cuh"

namespace hip_comms {

// PUSH (see allreduce_one_shot_push.cuh): every rank's rows, encoded by kBits' Codec, into
// every rank's inbox; one barrier; each block reduces its rows out of its own inbox and
// norms them as the pull kernel does. A block owns a row in both phases.
// `residual` and `residual_out` are unused (null) unless kAdd.
// `weight` is in its own dtype W: T, or fp32 (see `add_rms_norm_row`).
template <typename T, typename W, int ngpus, int kBits, bool kAdd>
DINLINE void one_shot_push_add_rms_norm_body(ipc::Peers p, T* __restrict__ out,
                                             T* __restrict__ residual_out,
                                             const T* __restrict__ residual,
                                             const W* __restrict__ weight, float eps,
                                             int rows, int packs) {
  using V          = typename traits<T>::V;
  using C          = Codec<T, kBits>;
  constexpr int NL = traits<T>::N;
  ipc::Comm<T, ngpus> c(p);
  const ipc::Inbox<C, ngpus> box(rows * blockDim.x);
  c.template broadcast_rows<C>(box, rows, packs);

  c.peer_block_barrier();

  const V* res_in        = reinterpret_cast<const V*>(residual);
  const auto* w          = reinterpret_cast<const vec<W, NL>*>(weight);
  V* res_out             = reinterpret_cast<V*>(residual_out);
  V* o                   = reinterpret_cast<V*>(out);
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    V sum[kMaxRowPacks];
    c.template reduce_row<C>(box, row, packs, sum);
    add_rms_norm_row<T, W, kAdd>(
        sum, res_in, w, row, packs, inv_hidden, eps,
        [&](int, int i, const V& v) { res_out[row * packs + i] = v; },
        [&](int, int i, const V& v) { o[row * packs + i] = v; });
  }
}

// THE KERNELS, one per op, both the body above.
template <typename T, typename W, int ngpus, int kBits>
__global__ void __launch_bounds__(kMaxThreads, 1) allreduce_one_shot_push_rms_norm(
    ipc::Peers p, T* __restrict__ out, const W* __restrict__ weight, float eps, int rows,
    int packs) {
  one_shot_push_add_rms_norm_body<T, W, ngpus, kBits, false>(p, out, nullptr, nullptr,
                                                             weight, eps, rows, packs);
}

template <typename T, typename W, int ngpus, int kBits>
__global__ void __launch_bounds__(kMaxThreads, 1) allreduce_one_shot_push_add_rms_norm(
    ipc::Peers p, T* __restrict__ out, T* __restrict__ residual_out,
    const T* __restrict__ residual, const W* __restrict__ weight, float eps, int rows,
    int packs) {
  one_shot_push_add_rms_norm_body<T, W, ngpus, kBits, true>(p, out, residual_out, residual,
                                                            weight, eps, rows, packs);
}

}  // namespace hip_comms
