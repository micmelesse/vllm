// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce: reduce-scatter, then all-gather.

#pragma once

#include "p2p/p2p.cuh"
#include "utils.cuh"

namespace hip_comms {

// Each rank sums the rows it owns and shares them in its scratch; after the barrier every
// rank gathers every owner's rows. (ngpus-1)/ngpus x N moved twice against one-shot's
// (ngpus-1) x N, for one barrier: the right algorithm once the bytes dominate. The input
// is read only before the barrier, so no close.
template <typename T, int ngpus>
__global__ void __launch_bounds__(kMaxThreads, 1)
    allreduce_two_shot_pull(p2p::Peers p, T* __restrict__ out, int size) {
  using V           = typename traits<T>::V;
  const auto w      = p2p::start<T, ngpus>(p);
  const auto tiling = tiles::buffer_rows(size);
  const auto mine   = tiles::owned(tiling, p.rank, ngpus);
  const auto slot   = p2p::pull::slot(w, tiling);
  V* dst            = reinterpret_cast<V*>(out);

  for (int row = mine.begin + blockIdx.x; row < mine.end; row += gridDim.x) {
    V v[kMaxRowPacks];
    p2p::pull::reduce(w, tiling, row, v);
    p2p::pull::share(w, slot, tiling, row, v);
  }

  p2p::peer_barrier(w);

  p2p::pull::gather(w, slot, tiling, [&](int row, int i, const V& v) {
    store_global(dst + row * tiling.packs + i, v);
  });
}

}  // namespace hip_comms
