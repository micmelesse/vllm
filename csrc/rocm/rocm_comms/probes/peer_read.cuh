// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE LINKS' ACHIEVABLE BANDWIDTH, as the p2p layer reads: every thread streams its share of a
// peer's buffer (or every peer's at once) over p2p::read_input. A measurement, not a collective;
// machine/hardware.cuh's Calibration records its answer.

#pragma once

#include "../machine/build.cuh"
#include "../p2p/p2p.cuh"

namespace hip_comms {

// `peer` one rank, or -1 for every other rank at once. The loads are folded into `sink` only if
// they equal an impossible value, which keeps them without a store per load.
template <typename T, int ngpus>
__global__ void __launch_bounds__(kMaxThreads, 1)
    peer_read(p2p::DevComm p, int peer, int64_t packs, uint32_t* sink) {
  using V              = typename traits<T>::V;
  const int64_t first  = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  const int64_t stride = int64_t{gridDim.x} * blockDim.x;
  uint32_t acc         = 0;
  auto fold            = [&](const V& v) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) acc ^= w[j];
  };
  if (peer >= 0) {
    const auto their_input = p2p::input<T, ngpus>(p, peer);
    for (int64_t i = first; i < packs; i += stride) fold(p2p::read_input(their_input, i));
  } else {
    const auto inputs = p2p::inputs<T, ngpus>(p);
    for (int64_t i = first; i < packs; i += stride) {
#pragma unroll
      for (int r = 0; r < ngpus; ++r)
        if (r != p.rank) fold(p2p::read_input(inputs[r], i));
    }
  }
  if (acc == 0x9e3779b9u) *sink = acc;
}

}  // namespace hip_comms
