// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot all-reduce fused with residual add + RMSNorm.

#pragma once

#include "ipc.cuh"
#include "utils.cuh"

namespace hip_comms {

// The sum is already in registers when one-shot is about to store it, so normalising there
// saves an HBM round trip and a launch against an all-reduce followed by a norm kernel.
//
// A BLOCK OWNS A ROW, because the variance needs the whole row: one block per row,
// striding over rows, where plain one-shot is grid-stride over the flat buffer. (aiter's
// 1-stage fused gate, `hidden_dim / pack_size <= 1024`, is the same constraint.)
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) allreduce_one_shot_rmsnorm(
    ipc::Peers p, T* __restrict__ out, T* __restrict__ residual_out,
    const T* __restrict__ residual, const T* __restrict__ weight, float eps, int rows,
    int packs) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  const int rank   = p.rank();
  const V* ptrs[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; ++i)
    ptrs[i] = p.input<V>((rank + i) % ngpus);

  p.barrier_start<ngpus>();

  const V* res_in        = reinterpret_cast<const V*>(residual);
  const V* w             = reinterpret_cast<const V*>(weight);
  V* res_out             = reinterpret_cast<V*>(residual_out);
  V* o                   = reinterpret_cast<V*>(out);
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  // Uniform across the block, so every `__syncthreads` inside is reached by every thread.
  for (int row = blockIdx.x; row < rows; row += gridDim.x)
    add_rmsnorm_row<T, ngpus>(ptrs, res_in, w, row, packs, inv_hidden, eps,
                              res_out + row * packs, o + row * packs);

  // A rank that returns lets its INPUT be reused while a peer is still reading it.
  p.barrier_end<ngpus, true>();
}

}  // namespace hip_comms
