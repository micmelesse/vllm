// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot push all-reduce: reduce-scatter, then all-gather, every transfer a store into
// a peer's slot, encoded by kBits' codec (16: T itself; 8, 4: QuickReduce's scheme).

#pragma once

#include "p2p/p2p.cuh"
#include "common/memory.cuh"
#include "hardware.cuh"

namespace hip_comms {

// A rank reads only its own input and its own slots, so a sender can encode what it
// sends and the input is never read remotely (no close). Two barriers against the pull
// kernel's one; the same bytes on the links at kBits 16. Every rank decodes the same
// bytes, so every rank holds the same output.
template <typename T, int ngpus, int kBits>
__global__ void __launch_bounds__(kMaxThreads, 1)
    all_reduce_push_two_shot(p2p::Peers p, T* __restrict__ out, int size) {
  using V           = typename traits<T>::V;
  const auto w      = p2p::start<T, ngpus>(p);
  const auto tiling = tiles::buffer(size, ngpus);
  const auto in     = p2p::push::slot<kBits>(w, tiling, p2p::To::owners);
  const auto sum    = p2p::push::slot<kBits>(w, tiling, p2p::To::owners, in);
  V* dst            = reinterpret_cast<V*>(out);

  p2p::push::scatter(w, in, tiling);

  p2p::peer_barrier(w);

  for (int u = tiling.first(p.rank); u < tiling.end(p.rank); u = tiling.next(u)) {
    V v[kMaxRowPacks];
    p2p::push::reduce(w, in, tiling, u, v);
    p2p::push::share(w, sum, tiling, u, v);
  }

  p2p::peer_barrier(w);

  p2p::push::gather(w, sum, tiling, [&](int u, int k, const V& v) {
    store_global(dst + tiling.pos(u, k), v);
  });
}

}  // namespace hip_comms
