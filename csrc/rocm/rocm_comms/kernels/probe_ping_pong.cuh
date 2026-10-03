// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE PROBE'S ROUND TRIP: a flag to a peer and back, one thread a rank (experimental::probe).

#pragma once

#include "../machine/build.cuh"

namespace hip_comms {

// One thread on each rank of the pair: the lower rank writes and waits, the higher waits and writes
// back, `kWarm` untimed then `iters` timed, from flag value `base` on. Device wall-clock ticks.
__global__ void ping_pong(PeerSignals peer_signals, Signal* self_signal, int rank,
                          uint64_t timeout_ticks, int peer, uint32_t base, int iters,
                          uint64_t* ticks) {
  constexpr int kWarm = 16;
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const bool first = rank < peer;
  uint64_t t0      = 0;
  for (int i = 1; i <= kWarm + iters; ++i) {
    if (i == kWarm + 1) t0 = wall_clock64();
    const uint32_t v = base + static_cast<uint32_t>(i);
    if (!first) wait_flag(self_signal, rank, timeout_ticks, peer, v);
    write_flag(peer_signals, rank, peer, v);
    if (first) wait_flag(self_signal, rank, timeout_ticks, peer, v);
  }
  *ticks = wall_clock64() - t0;
}

// The flag values a launch with `iters` uses past its warm-up.
constexpr uint32_t ping_pong_flags(int iters) { return 16 + static_cast<uint32_t>(iters); }

}  // namespace hip_comms
