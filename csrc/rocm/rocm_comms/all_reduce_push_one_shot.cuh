// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot push all-reduce: every rank's whole input into every rank's slot, encoded by
// kBits' codec (16: T itself; 8, 4: QuickReduce's scheme), then each sums locally.

#pragma once

#include "p2p/p2p.cuh"
#include "common/memory.cuh"
#include "hardware.cuh"

namespace hip_comms {

// A rank reads only its own input and its own slot: one barrier and no close. Every rank
// sums the same bytes in the same order, so every rank holds the same output.
template <typename T, int ngpus, int kBits>
__global__ void __launch_bounds__(kMaxThreads, 1)
    all_reduce_push_one_shot(p2p::Peers p, T* __restrict__ out, int size) {
  using V           = typename traits<T>::V;
  const auto w      = p2p::start<T, ngpus>(p);
  const auto tiling = tiles::buffer(size, 1);
  const auto slot   = p2p::push::slot<kBits>(w, tiling, p2p::To::all);
  V* dst            = reinterpret_cast<V*>(out);

  p2p::push::scatter(w, slot, tiling);

  p2p::peer_barrier(w);

  for (int u = tiling.first(); u < tiling.end(); u = tiling.next(u)) {
    V v[kMaxRowPacks];
    p2p::push::reduce(w, slot, tiling, u, v);
    tiles::store(dst, tiling, u, v);
  }
}

}  // namespace hip_comms
