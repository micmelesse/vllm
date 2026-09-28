// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce then RMSNorm (`allreduce_two_shot_pull_rms_norm`), and
// all-reduce then add then RMSNorm (`allreduce_two_shot_pull_add_rms_norm`): one body, a
// kernel per op, so a trace names the op that ran.

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
//   world_barrier
//   phase 2  every rank gathers every rank's rows into its own outputs
//
// Every output element is computed by exactly one rank, so all ranks hold identical bytes.
// Fewer rows than ranks is correct and unbalanced: the ranks past the end own nothing and
// still reach every barrier.
// `weight` is in its own dtype W: T, or fp32 (see `add_rms_norm_row`).
template <typename T, typename W, int ngpus, bool kAdd>
DINLINE void two_shot_pull_add_rms_norm_body(ipc::Peers p, T* __restrict__ out,
                                             T* __restrict__ residual_out,
                                             const T* __restrict__ residual,
                                             const W* __restrict__ weight, float eps,
                                             int rows, int packs) {
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
      V sum[kMaxRowPacks];
      c.sum_row(row * packs, packs, sum);
      add_rms_norm_row<T, W, kAdd>(
          sum, res_in, w, row, packs, inv_hidden, eps,
          [&](int, int i, const V& v) { c.put(rank, half + local + i, v); },
          [&](int, int i, const V& v) { c.put(rank, local + i, v); });
    }
  }

  // The gather below gives each block the local rows it wrote above, so the same-numbered
  // blocks are all it must wait for; the input is read only above, so no close.
  c.peer_block_barrier();

  V* o       = reinterpret_cast<V*>(out);
  V* res_out = reinterpret_cast<V*>(residual_out);
  c.template gather_rows<kAdd ? 2 : 1>(
      chunk, rows, packs, half, [&](int region, int row, int k, const V& v) {
        store_global((region == 0 ? o : res_out) + row * packs + k, v);
      });
}

// THE KERNELS, one per op, both the body above.
template <typename T, typename W, int ngpus>
__global__ void __launch_bounds__(kMaxThreads, 1) allreduce_two_shot_pull_rms_norm(
    ipc::Peers p, T* __restrict__ out, const W* __restrict__ weight, float eps, int rows,
    int packs) {
  two_shot_pull_add_rms_norm_body<T, W, ngpus, false>(p, out, nullptr, nullptr, weight, eps,
                                                      rows, packs);
}

template <typename T, typename W, int ngpus>
__global__ void __launch_bounds__(kMaxThreads, 1) allreduce_two_shot_pull_add_rms_norm(
    ipc::Peers p, T* __restrict__ out, T* __restrict__ residual_out,
    const T* __restrict__ residual, const W* __restrict__ weight, float eps, int rows,
    int packs) {
  two_shot_pull_add_rms_norm_body<T, W, ngpus, true>(p, out, residual_out, residual, weight,
                                                     eps, rows, packs);
}

}  // namespace hip_comms
