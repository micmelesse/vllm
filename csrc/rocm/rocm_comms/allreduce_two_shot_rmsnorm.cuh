// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot all-reduce fused with residual add + RMSNorm.

#pragma once

#include "ipc.cuh"
#include "utils.cuh"

namespace hip_comms {

// THE SLICE IS ROWS, NOT ELEMENTS. Plain two-shot slices the flat buffer, which would leave
// a rank holding part of a row and need a second exchange of partial sums of squares. Here
// each rank owns ceil(rows/ngpus) WHOLE rows, so it can finish the norm alone:
//
//   phase 1  reduce our rows across ranks, add the residual (replicated on every rank),
//            normalise, and publish both outputs in our scratch: [out rows | residual rows]
//   phase 2  every rank copies every rank's two halves into its own outputs
//
// (ngpus-1)/ngpus x N in, 2 x (ngpus-1)/ngpus x N back: 2.6N against one-shot's 7N at 8.
// Every output element is computed by exactly one rank, so all ranks hold identical bytes.
// Fewer rows than ranks is correct and unbalanced: the ranks past the end own nothing and
// still run both barriers.
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) allreduce_two_shot_rmsnorm(
    ipc::Peers p, T* __restrict__ out, T* __restrict__ residual_out,
    const T* __restrict__ residual, const T* __restrict__ weight, float eps, int rows,
    int packs) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  const int rank   = p.rank();
  // ROTATED by rank, as every variant is, so the ranks do not all read rank 0 first.
  const V* ptrs[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; ++i)
    ptrs[i] = p.input<V>((rank + i) % ngpus);

  const int chunk = (rows + ngpus - 1) / ngpus;
  // Where the residual half starts in a rank's scratch, in vectors.
  const int half = chunk * packs;

  p.barrier_start<ngpus>();

  // PHASE 1 -- our rows, finished, into our own scratch.
  {
    const V* res_in        = reinterpret_cast<const V*>(residual);
    const V* w             = reinterpret_cast<const V*>(weight);
    V* mine                = p.scratch<V>(rank);
    const int begin        = rank * chunk;
    const int end          = begin + chunk < rows ? begin + chunk : rows;
    const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
    for (int row = begin + blockIdx.x; row < end; row += gridDim.x) {
      const int local = (row - begin) * packs;
      add_rmsnorm_row<T, ngpus>(ptrs, res_in, w, row, packs, inv_hidden, eps,
                                mine + half + local, mine + local);
    }
  }

  // NOT `final_sync`: the peers are about to READ what we just wrote.
  p.barrier_end<ngpus, false>();

  // PHASE 2 -- gather. Rank i's rows sit in rank i's scratch.
  {
    V* o             = reinterpret_cast<V*>(out);
    V* res_out       = reinterpret_cast<V*>(residual_out);
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
#pragma unroll
    for (int i = 0; i < ngpus; ++i) {
      const int begin = i * chunk;
      const int end   = begin + chunk < rows ? begin + chunk : rows;
      const int n     = (end - begin) * packs;
      const V* src    = p.scratch<V>(i);
      for (int idx = tid; idx < n; idx += stride) {
        o[begin * packs + idx]       = src[idx];
        res_out[begin * packs + idx] = src[half + idx];
      }
    }
  }

  // A rank that returns lets its INPUT be reused while a peer is still reading it.
  p.barrier_end<ngpus, true>();
}

}  // namespace hip_comms
