// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot all-reduce: reduce-scatter, then all-gather.

#pragma once

#include "p2p/device.cuh"
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
    allreduce_two_shot_pull(p2p::Peers p, T* __restrict__ out, int size) {
  using V    = typename traits<T>::V;
  using core = p2p::Core<T, ngpus>;
  using pull = p2p::Pull<T, ngpus>;
  core::start(p);
  const auto in   = core::inputs(p);
  const int chunk = (size + ngpus - 1) / ngpus;
  const int rank  = p.rank;

  const int mine_begin = rank * chunk;
  const int mine_end   = min(mine_begin + chunk, size);
  pull::reduce_flat(p, in, mine_begin, mine_end,
                    [&](int at, const V& v) { core::put(p, rank, at - mine_begin, v); });

  // The gather below gives each thread the positions it wrote above, so the same-numbered
  // blocks are all it must wait for; the input is read only above, so no close.
  core::peer_block_barrier(p);

  V* dst = reinterpret_cast<V*>(out);
  pull::gather_flat(p, chunk, size, [&](int at, const V& v) { store_global(dst + at, v); });
}

}  // namespace hip_comms
