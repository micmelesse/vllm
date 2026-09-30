// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE LINK'S ROUND TRIP, as the p2p layer pays it: one thread on each rank of a pair, flags back
// and forth over p2p flags. A measurement, not a collective; target/hardware.cuh records its
// answer.

#pragma once

#include "../p2p/p2p.cuh"

namespace hip_comms {

// One thread on each rank of the pair: the lower rank writes and waits, the higher waits and writes
// back, `kWarm` untimed then `iters` timed, from flag value `base` on. Device wall-clock ticks.
__global__ void ping_pong(p2p::DevComm p, int peer, uint32_t base, int iters, uint64_t* ticks) {
  constexpr int kWarm = 16;
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const bool first = p.rank < peer;
  uint64_t t0      = 0;
  for (int i = 1; i <= kWarm + iters; ++i) {
    if (i == kWarm + 1) t0 = wall_clock64();
    const uint32_t v = base + static_cast<uint32_t>(i);
    if (!first) p2p::wait_flag(p, peer, v);
    p2p::write_flag(p, peer, v);
    if (first) p2p::wait_flag(p, peer, v);
  }
  *ticks = wall_clock64() - t0;
}

// The flag values a launch with `iters` uses past its warm-up.
constexpr uint32_t ping_pong_flags(int iters) { return 16 + static_cast<uint32_t>(iters); }

}  // namespace hip_comms
