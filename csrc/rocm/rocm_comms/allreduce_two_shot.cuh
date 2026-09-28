// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot all-reduce: reduce-scatter, then all-gather.

#pragma once

#include "ipc.cuh"
#include "utils.cuh"

namespace hip_comms {

// TWO-SHOT: each rank sums one slice into its own scratch, then every rank gathers every
// slice. (ngpus-1)/ngpus x N moved twice against one-shot's (ngpus-1) x N, for one more
// world barrier: the right algorithm once the bytes dominate.
//
// THE SLICE IS ceil(size/ngpus) AND THE LAST RANK TAKES WHAT IS LEFT, so a buffer that
// does not divide by ngpus is still reduced exactly once, and a rank with an empty slice
// still reaches every barrier.
template <typename T, int ngpus>
__global__ void __launch_bounds__(kMaxThreads, 1)
    allreduce_two_shot(ipc::Peers p, T* __restrict__ out, int size) {
  using V = typename traits<T>::V;
  ipc::Comm<T, ngpus> c(p);
  const int chunk  = (size + ngpus - 1) / ngpus;
  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const int rank   = c.rank();

  const int mine_begin = rank * chunk;
  const int mine_end   = min(mine_begin + chunk, size);
  // kSumBatch packs a thread in both phases, all loaded before any is stored.
  for (int idx = mine_begin + tid; idx < mine_end; idx += stride * kSumBatch) {
    V v[kSumBatch];
    c.template sum<kSumBatch>(idx, stride, mine_end, v);
#pragma unroll
    for (int u = 0; u < kSumBatch; ++u)
      if (idx + u * stride < mine_end) c.put(rank, idx + u * stride - mine_begin, v[u]);
  }

  c.world_barrier();

  V* dst = reinterpret_cast<V*>(out);
  for (int i = 0; i < ngpus; ++i) {
    const int begin = i * chunk;
    const int end   = min(begin + chunk, size);
    for (int idx = begin + tid; idx < end; idx += stride * kSumBatch) {
      V g[kSumBatch];
#pragma unroll
      for (int u = 0; u < kSumBatch; ++u)
        if (idx + u * stride < end) g[u] = c.get(i, idx + u * stride - begin);
#pragma unroll
      for (int u = 0; u < kSumBatch; ++u)
        if (idx + u * stride < end) store_global(dst + idx + u * stride, g[u]);
    }
  }
  c.close();
}

}  // namespace hip_comms
