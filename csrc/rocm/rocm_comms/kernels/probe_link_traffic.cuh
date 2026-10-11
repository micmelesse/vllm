// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE PROBE'S LINK TRAFFIC: every thread streaming packs over the staging or a registered (cached)
// buffer, pulled from one peer or every peer, pushed into every peer, or both at once
// (experimental::probe).

#pragma once

#include "../common/interface.cuh"

#include "../types.cuh"

namespace hip_comms {

// EVERY BLOCK STREAMING CHUNKS: pulled from the buffer `inp_ptrs` names on `peer` (a pull
// only) or every other rank (the staging, or a registered buffer), all of them in flight together,
// pushed into every other rank's staging, both at once with the blocks `split` (the first
// `pullers` pull, the rest push) or with `each` block doing both, chunk by chunk. The pulled chunks
// are summed and the sum folded into `sink` only if it is impossible (a negative sum of squares),
// which keeps the loads without a store per load.
template <typename DTYPE, int WORLD>
__global__ void __launch_bounds__(kBuild.kernels.max_threads, 1)
    link_traffic(const DTYPE* const* __restrict__ inp_ptrs, DTYPE* const* __restrict__ staging_ptrs,
                 Signal* const* __restrict__ signal_ptrs, Signal* self_signal_ptr, int rank,
                 uint64_t timeout_ticks, int mode, int peer, int pullers, int64_t packs,
                 uint32_t* sink_ptr) {
  constexpr int THREADS = kBuild.kernels.max_threads;
  using Chunk           = Tile<DTYPE, 1, THREADS * traits<DTYPE>::N, 1, THREADS, THREADS>;
  const auto traffic    = static_cast<Traffic>(mode);
  const bool split      = traffic == Traffic::split;
  const bool puller     = static_cast<int>(blockIdx.x) < pullers;
  const bool pulls      = traffic == Traffic::pull || traffic == Traffic::each || (split && puller);
  const bool pushes     = traffic == Traffic::push || traffic == Traffic::each || (split && !puller);
  // This block's place among the blocks in its role, and how many there are.
  const int grid        = static_cast<int>(gridDim.x);
  const int index       = split && !puller ? blockIdx.x - pullers : blockIdx.x;
  const int blocks      = !split ? grid : puller ? pullers : grid - pullers;
  const int len         = static_cast<int>(packs) * traits<DTYPE>::N;  // in elements
  // Every other rank, from the next one; a push's target is never the pulled buffer.
  std::array<Ptr<const DTYPE>, WORLD - 1> others;
#pragma unroll
  for (int i = 0; i < WORLD - 1; ++i)
    others[i] = rank_ptr<const DTYPE, WORLD>(inp_ptrs, (rank + 1 + i) % WORLD, len, 1);
  const Ptr<const DTYPE> one = peer < 0 ? Ptr<const DTYPE>{nullptr, len, 1, peer}
                                        : rank_ptr<const DTYPE, WORLD>(inp_ptrs, peer, len, 1);
  const auto staging   = rank_ptrs<DTYPE, WORLD>(staging_ptrs, len, 1);
  auto sum              = Chunk{1, len, 0, 0}.template zeros<float>();
  for (int offs_n = index * Chunk::kTileN; offs_n < len; offs_n += blocks * Chunk::kTileN) {
    const Chunk at{1, len, 0, offs_n};
    if (pulls && peer < 0) {
      Chunk got[WORLD - 1];
#pragma unroll
      for (int i = 0; i < WORLD - 1; ++i) got[i] = at;
      tile_load(got, others);
#pragma unroll
      for (int i = 0; i < WORLD - 1; ++i) sum = tile_add(sum, got[i].template to<float>());
    } else if (pulls) {
      Chunk got = at;
      tile_load(got, one);
      sum = tile_add(sum, got.template to<float>());
    }
    if (pushes) {
#pragma unroll
      for (int r = 0; r < WORLD; ++r)
        if (r != rank) tile_store(at, staging[r]);
    }
  }
  float d[1];
  partial_dot(sum, sum, d);
  if (d[0] < 0.0f) *sink_ptr = 1u;
}

}  // namespace hip_comms
