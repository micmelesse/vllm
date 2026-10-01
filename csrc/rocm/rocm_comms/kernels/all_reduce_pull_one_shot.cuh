// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce: every rank sums every rank's input over the whole buffer.
// Modelled on aiter's `cross_device_reduce_1stage`: sync, read every peer, sum, sync.

#pragma once

#include "../p2p/p2p.cuh"
#include "../common/common.cuh"

namespace hip_comms {

// `num_packs` packs, a thread a pack at a time over the whole grid. AN EAGER INPUT IS STAGED HERE,
// a pass at a time through the staging (DevComm::local), so any size runs in this one launch; a
// captured input is read in place, in one pass. EACH THREAD STAGES THE PACKS IT READS: what the
// same block on a peer copied is what a peers barrier makes visible.
template <typename T, int ngpus>
__global__ void __launch_bounds__(kMaxThreads, 1)
    all_reduce_pull_one_shot(p2p::DevComm p, T* __restrict__ out, int64_t num_packs) {
  using V = typename traits<T>::V;
  const V* local      = reinterpret_cast<const V*>(p.local);
  const int64_t pass  = local ? p.stage_packs : num_packs;
  const int64_t first = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  const int64_t step  = int64_t{gridDim.x} * blockDim.x;

  // Every rank's buffers.
  const auto peers = p2p::peers<T, ngpus>(p);
  const auto self  = p2p::self<T, ngpus>(p);
  V* dst           = reinterpret_cast<V*>(out);
  for (int64_t c0 = 0; c0 < num_packs; c0 += pass) {
    const int64_t n = min(pass, num_packs - c0);
    block_stamp(0);
    // 1. A staged pass: this rank's packs into its staging, then visible to the peers. In place:
    //    wait until every peer has launched, so its input is ready.
    if (local) {
      for (int64_t i = first; i < n; i += step) p2p::write_input(self, i, local[c0 + i]);
      p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::visible>(p);
    } else {
      p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);
    }
    block_stamp(1);
    // 2. Read every rank's pass, in rank order, and sum.
    const int64_t at = local ? 0 : c0;
    const auto read  = [&](int r, int64_t i) { return p2p::read_input(peers[r], at + i); };
    for (int64_t i = first; i < n; i += step)
      thread_store(dst + c0 + i, peers_reduce(peers_load<T, ngpus>(read, i)));
    block_stamp(2);
    // 3. No rank may overwrite its input (or the next pass its staging) until every peer has read
    //    it.
    p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::read>(p);
    block_stamp(5);
  }
}

}  // namespace hip_comms
