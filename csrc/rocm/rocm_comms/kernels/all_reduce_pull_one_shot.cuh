// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce: every rank sums every rank's input over the whole buffer.
// Modelled on aiter's `cross_device_reduce_1stage`: sync, read every peer, sum, sync. In two
// builds: in place, every rank reads every peer's registered or captured input where it is;
// staged (`kStaged`), for an eager input the peers cannot read, each rank copies its input into
// its staging a pass at a time and every rank reads every peer's staging, so any size runs in one
// launch.

#pragma once

#include "../p2p/p2p.cuh"
#include "../common/common.cuh"
#include "shared/staged.cuh"

namespace hip_comms {

// `num_packs` packs, a thread a pack at a time over the whole grid. STAGED, in passes of what a
// staging holds, EACH THREAD STAGING THE PACKS IT READS: what the same block on a peer copied is
// what a peers barrier makes visible.
template <typename T, int ngpus, bool kStaged>
__global__ void __launch_bounds__(kBuild.kernels.max_threads, 1)
    all_reduce_pull_one_shot(p2p::DevComm p, T* __restrict__ out, PackCount<kStaged> num_packs,
                             Staged<T, kStaged> staged) {
  if constexpr (kStaged) {
    using V             = typename traits<T>::V;
    const V* own        = reinterpret_cast<const V*>(staged.own_input);
    const int64_t first = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    const int64_t step  = int64_t{gridDim.x} * blockDim.x;
    const auto stagings = p2p::stagings<T, ngpus>(p);
    const auto own_staging = p2p::staging<T, ngpus>(p, p.rank);
    const auto read     = [&](int r, int64_t i) { return p2p::read_staging(stagings[r], i); };
    V* dst              = reinterpret_cast<V*>(out);

    for (int64_t c0 = 0; c0 < num_packs; c0 += staged.stage_packs) {
      const int64_t n = min(staged.stage_packs, num_packs - c0);
      block_stamp(0);
      // 1. This rank's pass into its staging, then visible to the peers (each has staged its own).
      for (int64_t i = first; i < n; i += step) p2p::write_staging(own_staging, i, own[c0 + i]);
      p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::visible>(p);
      block_stamp(1);
      // 2. Read every rank's staged pass, in rank order, and sum.
      for (int64_t i = first; i < n; i += step)
        thread_store(dst + c0 + i, peers_reduce(peers_load<T, ngpus>(read, i)));
      block_stamp(2);
      // 3. No rank may stage its next pass until every peer has read this one.
      p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::read>(p);
      block_stamp(5);
    }
  } else {
    using V = typename traits<T>::V;

    // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
    const auto inputs = p2p::inputs<T, ngpus>(p);
    block_stamp(0);
    p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);
    block_stamp(1);
    const auto read = [&](int r, int64_t i) { return p2p::read_input(inputs[r], i); };

    // 2. Read every rank's input, in rank order, and sum.
    V* dst = reinterpret_cast<V*>(out);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_packs; i += gridDim.x * blockDim.x)
      thread_store(dst + i, peers_reduce(peers_load<T, ngpus>(read, i)));
    block_stamp(2);

    // 3. No rank may overwrite its input until every peer has read it.
    p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::read>(p);
    block_stamp(5);
  }
}

}  // namespace hip_comms
