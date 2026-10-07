// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce: every rank sums every rank's input over the whole buffer.
// Modelled on aiter's `cross_device_reduce_1stage`: sync, read every peer, sum, sync. Two kernels:
// in place, every rank reads every peer's registered or captured input where it is; staged, for an
// eager input the peers cannot read.

#pragma once

#include "../common/interface.cuh"

namespace hip_comms {

// THE BUFFER AS ONE ROW, a block's chunk THREADS_PER_BLOCK groups of it, the grid striding over
// chunks: thread t of block b holds group b x THREADS_PER_BLOCK + t, as a pack a thread did.
template <typename DTYPE, int WORLD, int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot(const PeerPtrs* __restrict__ peer_inputs,
                             PeerSignals peer_signals, Signal* self_signal, int rank,
                             uint64_t timeout_ticks, DTYPE* __restrict__ out, int num_packs) {
  Sync<WORLD> sync{peer_signals, self_signal, rank, timeout_ticks};
  using Chunk   = Tile<DTYPE, 1, THREADS_PER_BLOCK * traits<DTYPE>::N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  const int len = num_packs * traits<DTYPE>::N;  // the buffer, in elements

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto inputs = rank_ptrs<const DTYPE, WORLD>(*peer_inputs, len);
  block_stamp(0);
  barrier<Group::peers, Until::launched>(sync);
  block_stamp(1);

  // 2. Read every rank's input, in rank order, and sum.
  for (int offs_n = blockIdx.x * Chunk::kTileN; offs_n < len; offs_n += gridDim.x * Chunk::kTileN) {
    Chunk peers[WORLD];
#pragma unroll
    for (int r = 0; r < WORLD; ++r) peers[r] = Chunk{1, len, 0, offs_n};
    tile_load(peers, inputs);
    tile_store(peers_reduce(peers), local_ptr(out, len, rank));
  }
  block_stamp(2);

  // 3. No rank may overwrite its input until every peer has read it.
  barrier<Group::peers, Until::read>(sync);
  block_stamp(5);
  sync.finish();
}

// STAGED: each rank copies `own_input` into its staging `stage_packs` at a time and every rank
// reads every peer's staging, so any size runs in one launch; `num_packs` in 64 bits. EACH BLOCK
// STAGES THE CHUNKS IT READS: what the same block on a peer copied is what a peers barrier makes
// visible.
template <typename DTYPE, int WORLD, int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot_staged(PeerPtrs peer_staging, PeerSignals peer_signals,
                                    Signal* self_signal, int rank, uint64_t timeout_ticks,
                                    DTYPE* __restrict__ out, int64_t num_packs,
                                    const DTYPE* __restrict__ own_input, int64_t stage_packs) {
  Sync<WORLD> sync{peer_signals, self_signal, rank, timeout_ticks};
  using Chunk            = Tile<DTYPE, 1, THREADS_PER_BLOCK * traits<DTYPE>::N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  constexpr int NL       = traits<DTYPE>::N;

  for (int64_t c0 = 0; c0 < num_packs; c0 += stage_packs) {
    const int len = static_cast<int>(min(stage_packs, num_packs - c0)) * NL;  // this pass
    const int64_t at = c0 * NL;
    const auto staged      = rank_ptrs<DTYPE, WORLD>(peer_staging, len);
    const auto own_staging = rank_ptr<DTYPE, WORLD>(peer_staging, rank, len);
    block_stamp(0);
    // 1. This rank's pass into its staging, then visible to the peers (each has staged its own).
    for (int offs_n = blockIdx.x * Chunk::kTileN; offs_n < len;
         offs_n += gridDim.x * Chunk::kTileN) {
      Chunk mine{1, len, 0, offs_n};
      tile_load(mine, local_ptr(own_input + at, len, rank));
      tile_store(mine, own_staging);
    }
    barrier<Group::peers, Until::visible>(sync);
    block_stamp(1);
    // 2. Read every rank's staged pass, in rank order, and sum.
    for (int offs_n = blockIdx.x * Chunk::kTileN; offs_n < len;
         offs_n += gridDim.x * Chunk::kTileN) {
      Chunk peers[WORLD];
#pragma unroll
      for (int r = 0; r < WORLD; ++r) peers[r] = Chunk{1, len, 0, offs_n};
      tile_load(peers, staged);
      tile_store(peers_reduce(peers), local_ptr(out + at, len, rank));
    }
    block_stamp(2);
    // 3. No rank may stage its next pass until every peer has read this one.
    barrier<Group::peers, Until::read>(sync);
    block_stamp(5);
  }
  sync.finish();
}

}  // namespace hip_comms
