// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot all-reduce fused with RMSNorm: `rms_norm`, or `fused_add_rms_norm` when kAdd.

#pragma once

#include "ipc.cuh"
#include "utils.cuh"

namespace hip_comms {

// THE SLICE IS ROWS, NOT ELEMENTS. Plain two-shot slices the flat buffer, which would leave
// a rank holding part of a row and need a second exchange of partial sums of squares. Here
// each rank owns ceil(rows/ngpus) WHOLE rows, so it can finish the norm alone:
//
//   phase 1  reduce our rows across ranks, (kAdd: add the residual, replicated on every
//            rank,) normalise, and publish in our scratch: [out rows | residual rows]
//   phase 2  every rank copies every rank's rows into its own outputs
//
// Every output element is computed by exactly one rank, so all ranks hold identical bytes.
// Fewer rows than ranks is correct and unbalanced: the ranks past the end own nothing and
// still run both barriers.
// `weight` is in its own dtype W: T, or fp32 (see `rms_norm_row`).
template <typename T, typename W, int ngpus, bool kAdd>
__global__ void __launch_bounds__(512, 1) allreduce_two_shot_rms_norm(
    ipc::Peers p, T* __restrict__ out, T* __restrict__ residual_out,
    const T* __restrict__ residual, const W* __restrict__ weight, float eps, int rows,
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
    const auto* w          = reinterpret_cast<const vec<W, NL>*>(weight);
    V* mine                = p.scratch<V>(rank);
    const int begin        = rank * chunk;
    const int end          = begin + chunk < rows ? begin + chunk : rows;
    const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
    for (int row = begin + blockIdx.x; row < end; row += gridDim.x) {
      const int local = (row - begin) * packs;
      rms_norm_row<T, W, ngpus, kAdd>(ptrs, res_in, w, row, packs, inv_hidden, eps,
                                      kAdd ? mine + half + local : nullptr, mine + local);
    }
  }

  // NOT `final_sync`: the peers are about to READ what we just wrote.
  p.barrier_end<ngpus, false>();

  // PHASE 2 -- gather. Rank i's rows sit in rank i's scratch.
  //
  // BY THE ROWS THIS BLOCK'S PEERS WROTE, not grid-stride over the flat buffer. The
  // barrier above is PER BLOCK: block b here has waited for block b on every rank and for
  // no other block. Phase 1 gave block b the rows begin + b, begin + b + gridDim.x, ...,
  // so those rows, and only those, are known finished in every peer's scratch. A flat
  // gather read rows other blocks were still writing: wrong output on two-shot at shapes
  // and timings where the blocks drifted apart (first run in `test`, 2026-09-26).
  {
    V* o       = reinterpret_cast<V*>(out);
    V* res_out = reinterpret_cast<V*>(residual_out);
#pragma unroll
    for (int i = 0; i < ngpus; ++i) {
      const int begin = i * chunk;
      const int end   = begin + chunk < rows ? begin + chunk : rows;
      const V* src    = p.scratch<V>(i);
      for (int row = begin + blockIdx.x; row < end; row += gridDim.x) {
        const int local = (row - begin) * packs;
        for (int k = threadIdx.x; k < packs; k += blockDim.x) {
          o[row * packs + k] = src[local + k];
          if constexpr (kAdd) res_out[row * packs + k] = src[half + local + k];
        }
      }
    }
  }

  // A rank that returns lets its INPUT be reused while a peer is still reading it.
  p.barrier_end<ngpus, true>();
}

}  // namespace hip_comms
