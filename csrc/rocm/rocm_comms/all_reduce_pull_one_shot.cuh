// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce: every rank sums every rank's input over the whole buffer.
// Modelled on aiter's `cross_device_reduce_1stage`: sync, read every peer, sum, sync.

#pragma once

#include "p2p/p2p.cuh"
#include "utils.cuh"

namespace hip_comms {

// `packs` 16-byte packs, a thread a pack at a time over the whole grid.
template <typename T, int ngpus>
__global__ void __launch_bounds__(kMaxThreads, 1)
    all_reduce_pull_one_shot(p2p::Peers p, T* __restrict__ out, int packs) {
  using V = typename traits<T>::V;

  // 1. Every rank's input is in memory its peers can read (registered, or staged); wait
  //    until every peer has launched, so its input is ready.
  p2p::simple::start_sync<ngpus>(p);

  // 2. Read every rank's input, in rank order, and sum.
  const V* in[ngpus];
#pragma unroll
  for (int r = 0; r < ngpus; ++r) in[r] = p2p::simple::input<T>(p, r);
  V* dst = reinterpret_cast<V*>(out);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < packs; i += gridDim.x * blockDim.x)
    store_global(dst + i, sum_packs<T, ngpus>(in, i));

  // 3. No rank may overwrite its input until every peer has read it.
  p2p::simple::end_sync<ngpus, true>(p);
}

}  // namespace hip_comms
