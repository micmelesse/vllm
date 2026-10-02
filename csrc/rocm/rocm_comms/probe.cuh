// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE PROBE: the machine as the p2p layer sees it, measured, not a collective. machine/hardware.cuh
// records its answers. rocm_comms_probe runs them all, every rank together:
//   ping_pong       a flag to a peer and back, one thread a rank: the link's round trip
//   link_traffic    every thread streaming packs: pulled from one peer or every peer, pushed into
//                   every peer, or both at once (half the blocks each)
//   probe_barrier   one block: every rank has reached it, so the next measurement starts together

#pragma once

#include "machine/build.cuh"
#include "p2p/p2p.cuh"

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

enum class Traffic : int { pull = 0, push = 1, both = 2 };

// `peer` one rank (a pull only) or -1 for every other rank. `both`: the even blocks pull, the
// odd push. The pulled packs are folded into `sink` only if they equal an impossible value,
// which keeps the loads without a store per load.
template <typename T, int ngpus>
__global__ void __launch_bounds__(kMaxThreads, 1)
    link_traffic(p2p::DevComm p, int mode, int peer, int64_t packs, uint32_t* sink) {
  using V              = typename traits<T>::V;
  const auto traffic   = static_cast<Traffic>(mode);
  const int parity     = static_cast<int>(blockIdx.x % 2);
  const bool pulls     = traffic == Traffic::pull || (traffic == Traffic::both && parity == 0);
  // This block's place among the blocks in its role, and how many there are.
  const int index      = traffic == Traffic::both ? blockIdx.x / 2 : blockIdx.x;
  const int blocks     = traffic == Traffic::both ? (gridDim.x + 1 - parity) / 2 : gridDim.x;
  const int64_t first  = int64_t{index} * blockDim.x + threadIdx.x;
  const int64_t stride = int64_t{blocks} * blockDim.x;
  if (pulls) {
    const auto inputs = p2p::inputs<T, ngpus>(p);
    uint32_t acc      = 0;
    for (int64_t i = first; i < packs; i += stride) {
#pragma unroll
      for (int r = 0; r < ngpus; ++r)
        if (r != p.rank && (peer < 0 || r == peer)) {
          const V v         = p2p::read_input(inputs[r], i);
          const uint32_t* w = reinterpret_cast<const uint32_t*>(&v);
#pragma unroll
          for (int j = 0; j < 4; ++j) acc ^= w[j];
        }
    }
    if (acc == 0x9e3779b9u) *sink = acc;
  } else {
    const auto stagings = p2p::stagings<T, ngpus>(p);
    V v;
    uint32_t* w = reinterpret_cast<uint32_t*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) w[j] = static_cast<uint32_t>(first) ^ j;
    for (int64_t i = first; i < packs; i += stride) {
#pragma unroll
      for (int r = 0; r < ngpus; ++r)
        if (r != p.rank) p2p::write_staging(stagings[r], i, v);
    }
  }
}

// One block, its twin on every rank: so the measurement after it starts on every rank together.
template <int ngpus>
__global__ void probe_barrier(p2p::DevComm p) {
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);
}

}  // namespace hip_comms
