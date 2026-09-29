// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce: every rank sums every rank's input over the whole buffer.

#pragma once

#include "p2p/p2p.cuh"
#include "utils.cuh"

namespace hip_comms {

// Moves ngpus x the bytes of two-shot and has no barrier between phases, so it wins while
// the barrier, not the bytes, dominates. Peers read this rank's input to the end: close.
template <typename T, int ngpus>
__global__ void __launch_bounds__(kMaxThreads, 1)
    all_reduce_pull_one_shot(p2p::Peers p, T* __restrict__ out, int size) {
  using V           = typename traits<T>::V;
  const auto w      = p2p::start<T, ngpus>(p);
  const auto tiling = tiles::buffer(size, 1);
  V* dst            = reinterpret_cast<V*>(out);
  for (int u = tiling.first(); u < tiling.end(); u = tiling.next(u)) {
    V v[kMaxRowPacks];
    p2p::pull::reduce(w, tiling, u, v);
    tiles::store(dst, tiling, u, v);
  }
  p2p::close(w);
}

}  // namespace hip_comms
