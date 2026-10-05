// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce: reduce-scatter, then all-gather. aiter's `cross_device_reduce_2stage`
// as written: sync, sum this rank's slice from every peer into its scratch, sync, copy every
// rank's summed slice out of its scratch. Two kernels: in place, the peers' registered or captured
// inputs read where they are; staged, for an eager input the peers cannot read.

#pragma once

#include "../common/interface.cuh"

namespace hip_comms {

// THE BUFFER AS ONE ROW CUT INTO ONE SLICE A RANK (the last one short), moved a chunk at a time
// as a tile whose ROWS ARE THE RANKS: row w is rank + w's (ROTATED, SO THE RANKS SPREAD OVER THE
// LINKS: at any moment the eight GPUs read eight different peers, where in rank order every GPU
// reads rank 0 first), a row's threads a wave (THREADS_PER_BLOCK / WORLD lanes) each loading one
// group. A rank sums its slice's chunk over the rows (block_reduce over M, in row order, the rows
// meeting in LDS) into its scratch, then copies every rank's summed chunk out. The sum's order
// differs by rank, which is harmless: each slice is summed by one rank, so every rank copies the
// same bytes. THE SAME BLOCK READS IN THE SECOND PHASE WHAT THE SAME BLOCK ON EACH RANK WROTE IN
// THE FIRST: both stride over chunks alike.
template <typename DTYPE, int WORLD, int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot(const PeerPtrs* __restrict__ peer_inputs,
                             PeerPtrs peer_scratch, PeerSignals peer_signals,
                             Signal* self_signal, int rank, uint64_t timeout_ticks,
                             DTYPE* __restrict__ out, int num_packs) {
  Sync<WORLD> sync{peer_signals, self_signal, rank, timeout_ticks};
  constexpr int NL    = traits<DTYPE>::N;
  constexpr int LANES = THREADS_PER_BLOCK / WORLD;
  static_assert(LANES * WORLD == THREADS_PER_BLOCK, "a block is a row of threads a rank");
  using Ranks     = Tile<DTYPE, WORLD, LANES * NL, WORLD, LANES, THREADS_PER_BLOCK>;
  const int len   = num_packs * NL;                              // the buffer, in elements
  const int slice = (num_packs + WORLD - 1) / WORLD * NL;        // a rank's, in elements
  const int first = rank * slice;
  const int mine  = max(0, min(slice, len - first));             // a late rank's may be short
  const auto rotated = [&](int w) { return (rank + w) % WORLD; };

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  // A ROW'S RANK WHEN IT IS READ: a row is a wave's, so its pointer is one scalar select (every
  // rank's, rotated, up front was ~20 scalar loads before the first barrier).
  const auto own_scratch = rank_scratch<DTYPE, WORLD>(peer_scratch, rank);
  const auto input       = [&](int w) {
    return rank_input<DTYPE, WORLD>(*peer_inputs, rotated(w)) + first;
  };
  const auto scratch = [&](int w) { return rank_scratch<DTYPE, WORLD>(peer_scratch, rotated(w)); };
  block_stamp(0);
  barrier<Group::peers, Until::launched>(sync);
  block_stamp(1);

  // 2. Reduce-scatter: this rank's slice, summed over every rank, into its scratch.
  for (int offs_n = blockIdx.x * Ranks::kTileN; offs_n < mine;
       offs_n += gridDim.x * Ranks::kTileN) {
    Ranks got{WORLD, mine, 0, offs_n};
    reduce_scatter(got, input, own_scratch);
  }

  block_stamp(2);
  // 3. Every rank's sums are visible to its peers.
  barrier<Group::peers, Until::visible>(sync);
  block_stamp(3);

  // 4. All-gather: every rank's summed chunk out of its scratch, at its place in the output. The
  //    next call's first sync keeps a rank from overwriting its scratch while it is read.
  for (int offs_n = blockIdx.x * Ranks::kTileN; offs_n < slice;
       offs_n += gridDim.x * Ranks::kTileN) {
    Ranks got{WORLD, slice, 0, offs_n};
    all_gather(got, scratch, [&](int w) { return out + rotated(w) * slice; },
               [&](int w) { return max(0, min(slice, len - rotated(w) * slice)); });
  }
  block_stamp(5);
  sync.finish();
}

// STAGED: each rank copies `own_input` into its staging a pass at a time and each pass reads the
// peers' staging, so any size runs in one launch; `num_packs` in 64 bits. A pass is at most what a
// staging holds (`stage_packs`) and what the scratch holds (a slice a rank). EACH BLOCK STAGES, OF
// EVERY RANK'S SLICE, THE CHUNKS THAT RANK'S SAME BLOCK READS, so a peers barrier makes them
// visible.
template <typename DTYPE, int WORLD, int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot_staged(PeerPtrs peer_scratch, PeerPtrs peer_staging,
                                    PeerSignals peer_signals, Signal* self_signal,
                                    int rank, uint64_t timeout_ticks, int64_t scratch_packs,
                                    DTYPE* __restrict__ out, int64_t num_packs,
                                    const DTYPE* __restrict__ own_input, int64_t stage_packs) {
  Sync<WORLD> sync{peer_signals, self_signal, rank, timeout_ticks};
  constexpr int NL    = traits<DTYPE>::N;
  constexpr int LANES = THREADS_PER_BLOCK / WORLD;
  static_assert(LANES * WORLD == THREADS_PER_BLOCK, "a block is a row of threads a rank");
  using Ranks        = Tile<DTYPE, WORLD, LANES * NL, WORLD, LANES, THREADS_PER_BLOCK>;
  const int64_t pass = min(stage_packs, scratch_packs * WORLD);
  const auto rotated = [&](int w) { return (rank + w) % WORLD; };
  const auto own_scratch = rank_scratch<DTYPE, WORLD>(peer_scratch, rank);
  const auto own_staging = rank_staging<DTYPE, WORLD>(peer_staging, rank);
  const auto staging     = [&](int w) { return rank_staging<DTYPE, WORLD>(peer_staging, rotated(w)); };
  const auto scratch     = [&](int w) { return rank_scratch<DTYPE, WORLD>(peer_scratch, rotated(w)); };

  for (int64_t c0 = 0; c0 < num_packs; c0 += pass) {
    const int len    = static_cast<int>(min(pass, num_packs - c0)) * NL;  // this pass, elements
    const int slice  = (len / NL + WORLD - 1) / WORLD * NL;
    const int first  = rank * slice;
    const int mine   = max(0, min(slice, len - first));
    const int64_t at = c0 * NL;
    const auto slice_n = [&](int w) { return max(0, min(slice, len - rotated(w) * slice)); };
    block_stamp(0);
    // 1. Every rank's slice of this pass into this rank's staging, the chunks each rank's same
    //    block reads, then visible (and, past the first pass, every peer has read this rank's
    //    scratch).
    for (int offs_n = blockIdx.x * Ranks::kTileN; offs_n < slice;
         offs_n += gridDim.x * Ranks::kTileN) {
      Ranks part{WORLD, slice, 0, offs_n};
      tile_gather(part, [&](int w) { return own_input + at + rotated(w) * slice; }, slice_n);
      tile_scatter([&](int w) { return own_staging + rotated(w) * slice; }, slice_n, part);
    }
    barrier<Group::peers, Until::visible>(sync);
    block_stamp(1);

    // 2. Reduce-scatter: this rank's slice from every rank's staging, into its scratch.
    for (int offs_n = blockIdx.x * Ranks::kTileN; offs_n < mine;
         offs_n += gridDim.x * Ranks::kTileN) {
      Ranks got{WORLD, mine, 0, offs_n};
      reduce_scatter(got, [&](int w) { return staging(w) + first; }, own_scratch);
    }

    block_stamp(2);
    // 3. Every rank's sums are visible to its peers, and every peer has read this rank's staged
    //    pass, so the next one may overwrite it.
    barrier<Group::peers, Until::visible>(sync);
    block_stamp(3);

    // 4. All-gather: every rank's summed chunk out of its scratch, at its place in the output.
    for (int offs_n = blockIdx.x * Ranks::kTileN; offs_n < slice;
         offs_n += gridDim.x * Ranks::kTileN) {
      Ranks got{WORLD, slice, 0, offs_n};
      all_gather(got, scratch, [&](int w) { return out + at + rotated(w) * slice; }, slice_n);
    }
    block_stamp(5);
  }
  sync.finish();
}

}  // namespace hip_comms
