// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE PROBE'S BARRIER: every rank has reached it, so the next measurement starts together
// (experimental::probe).

#pragma once

#include "../build.cuh"
#include "../p2p/p2p.cuh"

namespace hip_comms {

// One block, its twin on every rank: so the measurement after it starts on every rank together.
template <int ngpus>
__global__ void probe_barrier(p2p::DevComm p) {
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);
}

}  // namespace hip_comms
