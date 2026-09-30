// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce: every rank sums every rank's input over the whole buffer.
// Modelled on aiter's `cross_device_reduce_1stage`: sync, read every peer, sum, sync.

#pragma once

#include "p2p/p2p.cuh"
#include "common/memory.cuh"
#include "common/reduce.cuh"

namespace hip_comms {

// `num_packs` packs, a thread a pack at a time over the whole grid.
template <typename T, int ngpus>
__global__ void __launch_bounds__(kMaxThreads, 1)
    all_reduce_pull_one_shot(p2p::DevComm p, T* __restrict__ out, int num_packs) {
  using V = typename traits<T>::V;

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto peers = p2p::peers<T, ngpus>(p);
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);
  const auto read = [&](int r, int64_t i) { return p2p::read_input(peers[r], i); };

  // 2. Read every rank's input, in rank order, and sum.
  V* dst = reinterpret_cast<V*>(out);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_packs; i += gridDim.x * blockDim.x)
    store_global(dst + i, sum_packs<T, ngpus>(read, i));

  // 3. No rank may overwrite its input until every peer has read it.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::read>(p);
}

}  // namespace hip_comms
