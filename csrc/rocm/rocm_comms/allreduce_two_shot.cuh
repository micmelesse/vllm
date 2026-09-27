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
// sync: the right algorithm once the bytes dominate.
//
// THE SLICE IS ceil(size/ngpus) AND THE LAST RANK TAKES WHAT IS LEFT, so a buffer that
// does not divide by ngpus is still reduced exactly once, and a rank with an empty slice
// still reaches every sync.
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    allreduce_two_shot(ipc::Peers p, T* __restrict__ out, int size) {
  using V = typename traits<T>::V;
  ipc::Comm<T, ngpus> c(p);
  const int chunk  = (size + ngpus - 1) / ngpus;
  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const int rank   = c.rank();

  const int mine_begin = rank * chunk;
  const int mine_end   = min(mine_begin + chunk, size);
  for (int idx = mine_begin + tid; idx < mine_end; idx += stride)
    c.put(rank, idx - mine_begin, c.sum(idx));

  c.sync();

  V* dst = reinterpret_cast<V*>(out);
  for (int i = 0; i < ngpus; ++i) {
    const int begin = i * chunk;
    const int end   = min(begin + chunk, size);
    for (int idx = begin + tid; idx < end; idx += stride) dst[idx] = c.get(i, idx - begin);
  }
  c.close();
}

}  // namespace hip_comms
