// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot all-reduce.

#pragma once

#include "ipc.cuh"
#include "utils.cuh"

namespace hip_comms {

// ONE-SHOT: every rank sums every peer's input over the whole buffer into its own output.
// Moves ngpus x the bytes of two-shot and needs no barrier between phases, so it wins while
// the barrier, not the bytes, dominates.
template <typename T, int ngpus>
__global__ void __launch_bounds__(kMaxThreads, 1)
    allreduce_one_shot(ipc::Peers p, T* __restrict__ out, int size) {
  using V = typename traits<T>::V;
  ipc::Comm<T, ngpus> c(p);
  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  V* dst           = reinterpret_cast<V*>(out);
  // kSumBatch packs a thread, all loaded before any is stored.
  for (int idx = tid; idx < size; idx += stride * kSumBatch) {
    V v[kSumBatch];
    c.template sum<kSumBatch>(idx, stride, size, v);
#pragma unroll
    for (int u = 0; u < kSumBatch; ++u)
      if (idx + u * stride < size) store_global(dst + idx + u * stride, v[u]);
  }
  c.close();
}

}  // namespace hip_comms
