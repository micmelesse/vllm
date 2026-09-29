// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce: reduce-scatter, then all-gather. Modelled on aiter's
// `cross_device_reduce_2stage`: sync, sum this rank's slice from every peer into its scratch,
// sync, copy every rank's summed slice out of its scratch.

#pragma once

#include "p2p/p2p.cuh"
#include "utils.cuh"

namespace hip_comms {

// `packs` 16-byte packs cut into one slice per rank (the last one short), a thread a pack at a
// time over the whole grid. THE SAME THREAD INDEXES A PACK IN BOTH PHASES: after the sync a
// block may read only what the same block on a peer wrote.
template <typename T, int ngpus>
__global__ void __launch_bounds__(kMaxThreads, 1)
    all_reduce_pull_two_shot(p2p::Peers p, T* __restrict__ out, int packs) {
  using V          = typename traits<T>::V;
  const int slice  = (packs + ngpus - 1) / ngpus;
  const int first  = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  // 1. Wait until every peer has launched, so its input is ready.
  p2p::simple::start_sync<ngpus>(p);

  // 2. Reduce-scatter: this rank's slice, read from every rank in rank order and summed,
  //    into this rank's scratch.
  const V* in[ngpus];
#pragma unroll
  for (int r = 0; r < ngpus; ++r) in[r] = p2p::simple::input<T>(p, r);
  const int base = p.rank * slice;
  const int mine = min(slice, packs - base);
  V* sums        = p2p::simple::scratch<T, ngpus>(p, p.rank);
  for (int i = first; i < mine; i += stride)
    store_global(sums + i, sum_packs<T, ngpus>(in, base + i));

  // 3. Every rank's sums are visible to its peers.
  p2p::simple::end_sync<ngpus, false>(p);

  // 4. All-gather: every rank's slice out of its scratch, at its place in the output. The
  //    next call's first sync keeps a rank from overwriting its scratch while it is read.
  V* dst = reinterpret_cast<V*>(out);
  for (int i = first; i < slice; i += stride) {
#pragma unroll
    for (int r = 0; r < ngpus; ++r)
      if (r * slice + i < packs)
        store_global(dst + r * slice + i, load_global(p2p::simple::scratch<T, ngpus>(p, r) + i));
  }
}

}  // namespace hip_comms
