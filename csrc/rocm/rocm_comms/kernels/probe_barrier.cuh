// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE PROBE'S BARRIER: every rank has reached it, so the next measurement starts together
// (experimental::probe).

#pragma once

#include "../common/interface.cuh"


namespace hip_comms {

// One block, its twin on every rank: so the measurement after it starts on every rank together.
template <int WORLD>
__global__ void probe_barrier(Signal* const* __restrict__ signal_ptrs, Signal* self_signal_ptr,
                              int rank,
                              uint64_t timeout_ticks) {
  Sync<WORLD> sync{signal_ptrs, self_signal_ptr, rank, timeout_ticks};
  barrier<Group::peers, Until::launched>(sync);
  sync.finish();
}

}  // namespace hip_comms
