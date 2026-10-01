// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce of an EAGER input (`all_reduce_pull_one_shot_staged`): the peers cannot
// read it where it is, so each rank copies it into its staging a pass at a time and every rank
// reads every peer's staging. Any size runs in one launch. A registered or captured input reads in
// place instead (all_reduce_pull_one_shot).

#pragma once

#include "../p2p/p2p.cuh"
#include "../common/common.cuh"

namespace hip_comms {

// `num_packs` packs in passes of what a staging holds, a thread a pack at a time over the grid.
// EACH THREAD STAGES THE PACKS IT READS: what the same block on a peer copied is what a peers
// barrier makes visible.
template <typename T, int ngpus>
__global__ void __launch_bounds__(kMaxThreads, 1)
    all_reduce_pull_one_shot_staged(p2p::DevComm p, T* __restrict__ out,
                                    const T* __restrict__ own_input, int64_t num_packs,
                                    int64_t stage_packs) {
  using V             = typename traits<T>::V;
  const V* own        = reinterpret_cast<const V*>(own_input);
  const int64_t first = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  const int64_t step  = int64_t{gridDim.x} * blockDim.x;
  const auto stagings = p2p::stagings<T, ngpus>(p);
  const auto own_staging = p2p::staging<T, ngpus>(p, p.rank);
  const auto read     = [&](int r, int64_t i) { return p2p::read_staging(stagings[r], i); };
  V* dst              = reinterpret_cast<V*>(out);

  for (int64_t c0 = 0; c0 < num_packs; c0 += stage_packs) {
    const int64_t n = min(stage_packs, num_packs - c0);
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
}

}  // namespace hip_comms
