// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE PROBE'S BARRIER: every rank has reached it, so the next measurement starts together
// (experimental::probe).

#pragma once

#include "../machine/build.cuh"
#include "../p2p/p2p.cuh"

namespace hip_comms {

// One block, its twin on every rank: so the measurement after it starts on every rank together.
template <int NGPUS>
__global__ void probe_barrier(p2p::PeerSignals peer_signals, p2p::Signal* self_signal, int rank,
                              uint64_t timeout_ticks) {
  p2p::barrier<NGPUS, p2p::Among::peers, p2p::Ensure::launched>(
      peer_signals, self_signal, rank, timeout_ticks);
}

}  // namespace hip_comms
