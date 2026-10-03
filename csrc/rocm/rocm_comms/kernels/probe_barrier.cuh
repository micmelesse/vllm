// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE PROBE'S BARRIER: every rank has reached it, so the next measurement starts together
// (experimental::probe).

#pragma once

#include "../common/common.cuh"


namespace hip_comms {

// One block, its twin on every rank: so the measurement after it starts on every rank together.
template <int WORLD>
__global__ void probe_barrier(PeerSignals peer_signals, Signal* self_signal, int rank,
                              uint64_t timeout_ticks) {
  barrier<WORLD, Among::peers, Ensure::launched>(
      peer_signals, self_signal, rank, timeout_ticks);
}

}  // namespace hip_comms
