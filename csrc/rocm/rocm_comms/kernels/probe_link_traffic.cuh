// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE PROBE'S LINK TRAFFIC: every thread streaming packs over the staging or a registered (cached)
// buffer, pulled from one peer or every peer, pushed into every peer, or both at once
// (experimental::probe).

#pragma once

#include "../machine/build.cuh"
#include "../p2p/p2p.cuh"
#include "../types.cuh"

namespace hip_comms {

// EVERY THREAD STREAMING 16-byte packs: pulled from the buffer `peer_inputs` names on `peer` (a
// pull only) or every other rank (the staging, or a registered buffer), pushed into every other
// rank's staging, both at once with the blocks `split` (the first `pullers` pull, the rest push) or with
// `each` block doing both, pack by pack. The pulled packs are folded into `sink` only if they
// equal an impossible value, which keeps the loads without a store per load.
template <typename T, int ngpus>
__global__ void __launch_bounds__(kBuild.kernels.max_threads, 1)
    link_traffic(const p2p::PeerPtrs* __restrict__ peer_inputs, p2p::PeerPtrs peer_staging,
                 p2p::PeerSignals peer_signals, p2p::Signal* self_signal, int rank,
                 uint64_t timeout_ticks, int mode, int peer, int pullers, int64_t packs,
                 uint32_t* sink) {
  using V              = typename traits<T>::V;
  const auto traffic   = static_cast<Traffic>(mode);
  const bool split     = traffic == Traffic::split;
  const bool puller    = static_cast<int>(blockIdx.x) < pullers;
  const bool pulls     = traffic == Traffic::pull || traffic == Traffic::each || (split && puller);
  const bool pushes    = traffic == Traffic::push || traffic == Traffic::each || (split && !puller);
  // This block's place among the blocks in its role, and how many there are.
  const int grid       = static_cast<int>(gridDim.x);
  const int index      = split && !puller ? blockIdx.x - pullers : blockIdx.x;
  const int blocks     = !split ? grid : puller ? pullers : grid - pullers;
  const int64_t first  = int64_t{index} * blockDim.x + threadIdx.x;
  const int64_t stride = int64_t{blocks} * blockDim.x;
  const auto buffers   = p2p::inputs<T, ngpus>(*peer_inputs);
  // A push's target: never the pulled buffer.
  const auto stagings  = p2p::stagings<T, ngpus>(peer_staging);
  V v;
  uint32_t* w = reinterpret_cast<uint32_t*>(&v);
#pragma unroll
  for (int j = 0; j < 4; ++j) w[j] = static_cast<uint32_t>(first) ^ j;
  uint32_t acc = 0;
  for (int64_t i = first; i < packs; i += stride) {
    if (pulls) {
#pragma unroll
      for (int r = 0; r < ngpus; ++r)
        if (r != rank && (peer < 0 || r == peer)) {
          const V got       = p2p::read_input(buffers[r], i);
          const uint32_t* g = reinterpret_cast<const uint32_t*>(&got);
#pragma unroll
          for (int j = 0; j < 4; ++j) acc ^= g[j];
        }
    }
    if (pushes) {
#pragma unroll
      for (int r = 0; r < ngpus; ++r)
        if (r != rank) p2p::write_staging(stagings[r], i, v);
    }
  }
  if (acc == 0x9e3779b9u) *sink = acc;
}

}  // namespace hip_comms
