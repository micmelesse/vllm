// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot all-reduce then RMSNorm (`allreduce_two_shot_rms_norm`), and all-reduce then
// add then RMSNorm (`allreduce_two_shot_add_rms_norm`): one body, a kernel per op, so a
// trace names the op that ran.

#pragma once

#include "ipc.cuh"
#include "utils.cuh"

namespace hip_comms {

// THE SLICE IS ROWS, NOT ELEMENTS. Plain two-shot slices the flat buffer, which would leave
// a rank holding part of a row and need a second exchange of partial sums of squares. Here
// each rank owns ceil(rows/ngpus) WHOLE rows, so it can finish the norm alone:
//
//   phase 1  reduce our rows across ranks, (kAdd: add the residual, replicated on every
//            rank,) normalise, and put them in our scratch: [out rows | residual rows]
//   sync
//   phase 2  every rank gathers every rank's rows into its own outputs
//
// Every output element is computed by exactly one rank, so all ranks hold identical bytes.
// Fewer rows than ranks is correct and unbalanced: the ranks past the end own nothing and
// still reach every sync.
// `weight` is in its own dtype W: T, or fp32 (see `add_rms_norm_row`).
template <typename T, typename W, int ngpus, bool kAdd>
DINLINE void two_shot_add_rms_norm(ipc::Peers p, T* __restrict__ out,
                                     T* __restrict__ residual_out,
                                     const T* __restrict__ residual,
                                     const W* __restrict__ weight, float eps, int rows,
                                     int packs) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  ipc::Comm<T, ngpus> c(p);
  const int rank  = c.rank();
  const int chunk = (rows + ngpus - 1) / ngpus;
  // Where the residual half starts in a rank's scratch, in packs.
  const int half = chunk * packs;

  {
    const V* res_in        = reinterpret_cast<const V*>(residual);
    const auto* w          = reinterpret_cast<const vec<W, NL>*>(weight);
    const int begin        = rank * chunk;
    const int end          = min(begin + chunk, rows);
    const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
    for (int row = begin + blockIdx.x; row < end; row += gridDim.x) {
      const int local = (row - begin) * packs;
      add_rms_norm_row<T, W, kAdd>(
          c, res_in, w, row, packs, inv_hidden, eps,
          [&](int i, const V& v) { c.put(rank, half + local + i, v); },
          [&](int i, const V& v) { c.put(rank, local + i, v); });
    }
  }

  c.sync();

  V* o       = reinterpret_cast<V*>(out);
  V* res_out = reinterpret_cast<V*>(residual_out);
  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int i = 0; i < ngpus; ++i) {
    const int begin = i * chunk;
    const int n     = (min(begin + chunk, rows) - begin) * packs;
    for (int k = tid; k < n; k += stride) {
      o[begin * packs + k] = c.get(i, k);
      if constexpr (kAdd) res_out[begin * packs + k] = c.get(i, half + k);
    }
  }
  c.close();
}

// THE KERNELS, one per op, both the body above.
template <typename T, typename W, int ngpus>
__global__ void __launch_bounds__(512, 1) allreduce_two_shot_rms_norm(
    ipc::Peers p, T* __restrict__ out, const W* __restrict__ weight, float eps, int rows,
    int packs) {
  two_shot_add_rms_norm<T, W, ngpus, false>(p, out, nullptr, nullptr, weight, eps, rows,
                                              packs);
}

template <typename T, typename W, int ngpus>
__global__ void __launch_bounds__(512, 1) allreduce_two_shot_add_rms_norm(
    ipc::Peers p, T* __restrict__ out, T* __restrict__ residual_out,
    const T* __restrict__ residual, const W* __restrict__ weight, float eps, int rows,
    int packs) {
  two_shot_add_rms_norm<T, W, ngpus, true>(p, out, residual_out, residual, weight, eps,
                                             rows, packs);
}

}  // namespace hip_comms
