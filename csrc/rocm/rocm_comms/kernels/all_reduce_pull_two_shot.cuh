// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce: reduce-scatter, then all-gather. aiter's `cross_device_reduce_2stage`
// as written: sync, sum this rank's slice from every peer into its scratch, sync, copy every
// rank's summed slice out of its scratch. Two kernels: in place, the peers' registered or captured
// inputs read where they are; staged, for an eager input the peers cannot read.

#pragma once

#include "../p2p/p2p.cuh"
#include "../common/common.cuh"

namespace hip_comms {

// THE BUFFER AS ONE ROW CUT INTO ONE SLICE A RANK (the last one short), each in chunks of
// THREADS_PER_BLOCK groups, the grid striding over them. A rank sums its slice over every rank into
// its scratch, every thread its own columns from all of them, aiter's two-stage; then copies every
// rank's summed slice out. ROTATED, SO THE RANKS SPREAD OVER THE LINKS: a rank reads rank + w for
// w = 0.. in order, where in rank order every GPU reads rank 0 first; the sum's order differs by
// rank, which is harmless, since each slice is summed by one rank. THE SAME BLOCK READS IN THE
// SECOND PHASE WHAT THE SAME BLOCK ON EACH RANK WROTE IN THE FIRST: both stride over chunks alike.
template <typename DTYPE, int WORLD, int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)
    all_reduce_pull_two_shot(const p2p::PeerPtrs* __restrict__ peer_inputs,
                             p2p::PeerPtrs peer_scratch, p2p::PeerSignals peer_signals,
                             p2p::Signal* self_signal, int rank, uint64_t timeout_ticks,
                             DTYPE* __restrict__ out, int num_packs) {
  using Chunk     = Tile<DTYPE, 1, THREADS_PER_BLOCK * traits<DTYPE>::N, 1, THREADS_PER_BLOCK>;
  constexpr int NL = traits<DTYPE>::N;
  const int len   = num_packs * NL;                              // the buffer, in elements
  const int slice = (num_packs + WORLD - 1) / WORLD * NL;        // a rank's, in elements
  const int first = rank * slice;
  const int mine  = max(0, min(slice, len - first));             // a late rank's may be short

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto own_scratch = p2p::scratch<DTYPE, WORLD>(peer_scratch, rank);
  const auto scratches   = p2p::scratches<DTYPE, WORLD>(peer_scratch);
  const DTYPE* rotated[WORLD];
#pragma unroll
  for (int w = 0; w < WORLD; ++w)
    rotated[w] = p2p::input<DTYPE, WORLD>(*peer_inputs, (rank + w) % WORLD).data() + first;
  block_stamp(0);
  p2p::barrier<WORLD, p2p::Among::peers, p2p::Ensure::launched>(
      peer_signals, self_signal, rank, timeout_ticks);
  block_stamp(1);

  // 2. Reduce-scatter: this rank's slice, summed over every rank, into its scratch.
  for (int offs_n = blockIdx.x * Chunk::kTileN; offs_n < mine;
       offs_n += gridDim.x * Chunk::kTileN) {
    Chunk peers[WORLD];
#pragma unroll
    for (int w = 0; w < WORLD; ++w) peers[w] = Chunk{1, mine, 0, offs_n};
    peers_load(peers, [&](int w) { return rotated[w]; }, mine);
    tile_store(own_scratch.data(), mine, peers_reduce(peers));
  }

  block_stamp(2);
  // 3. Every rank's sums are visible to its peers.
  p2p::barrier<WORLD, p2p::Among::peers, p2p::Ensure::visible>(
      peer_signals, self_signal, rank, timeout_ticks);
  block_stamp(3);

  // 4. All-gather: every rank's slice out of its scratch, at its place in the output, every
  //    rank's chunk loaded before any is stored. The next call's first sync keeps a rank from
  //    overwriting its scratch while it is read.
  for (int offs_n = blockIdx.x * Chunk::kTileN; offs_n < slice;
       offs_n += gridDim.x * Chunk::kTileN) {
    Chunk got[WORLD];
#pragma unroll
    for (int r = 0; r < WORLD; ++r) got[r] = Chunk{1, slice, 0, offs_n};
    peers_load(got, [&](int r) { return scratches[r].data(); }, slice);
#pragma unroll
    for (int r = 0; r < WORLD; ++r) {
      got[r].N = max(0, min(slice, len - r * slice));  // the rank's slice, a late one short
      if (got[r].N > 0) tile_store(out + r * slice, slice, got[r]);
    }
  }
  block_stamp(5);
}

// STAGED: each rank copies `own_input` into its staging a pass at a time and each pass reads the
// peers' staging, so any size runs in one launch; `num_packs` in 64 bits. A pass is at most what a
// staging holds (`stage_packs`) and what the scratch holds (a slice a rank). EACH BLOCK STAGES, OF
// EVERY RANK'S SLICE, THE CHUNKS THAT RANK'S SAME BLOCK READS, so a peers barrier makes them
// visible.
template <typename DTYPE, int WORLD, int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)
    all_reduce_pull_two_shot_staged(p2p::PeerPtrs peer_scratch, p2p::PeerPtrs peer_staging,
                                    p2p::PeerSignals peer_signals, p2p::Signal* self_signal,
                                    int rank, uint64_t timeout_ticks, int64_t scratch_packs,
                                    DTYPE* __restrict__ out, int64_t num_packs,
                                    const DTYPE* __restrict__ own_input, int64_t stage_packs) {
  using Chunk        = Tile<DTYPE, 1, THREADS_PER_BLOCK * traits<DTYPE>::N, 1, THREADS_PER_BLOCK>;
  constexpr int NL   = traits<DTYPE>::N;
  const int64_t pass = min(stage_packs, scratch_packs * WORLD);
  const auto own_scratch = p2p::scratch<DTYPE, WORLD>(peer_scratch, rank);
  const auto scratches   = p2p::scratches<DTYPE, WORLD>(peer_scratch);
  const auto own_staging = p2p::staging<DTYPE, WORLD>(peer_staging, rank);

  for (int64_t c0 = 0; c0 < num_packs; c0 += pass) {
    const int len    = static_cast<int>(min(pass, num_packs - c0)) * NL;  // this pass, elements
    const int slice  = (len / NL + WORLD - 1) / WORLD * NL;
    const int first  = rank * slice;
    const int mine   = max(0, min(slice, len - first));
    const int64_t at = c0 * NL;
    const DTYPE* rotated[WORLD];
#pragma unroll
    for (int w = 0; w < WORLD; ++w)
      rotated[w] = p2p::staging<DTYPE, WORLD>(peer_staging, (rank + w) % WORLD).data() + first;
    block_stamp(0);
    // 1. Every rank's slice of this pass into this rank's staging, the chunks each rank's same
    //    block reads, then visible (and, past the first pass, every peer has read this rank's
    //    scratch).
    for (int offs_n = blockIdx.x * Chunk::kTileN; offs_n < slice;
         offs_n += gridDim.x * Chunk::kTileN) {
#pragma unroll
      for (int r = 0; r < WORLD; ++r) {
        const int n = max(0, min(slice, len - r * slice));
        if (offs_n >= n) continue;
        Chunk part{1, n, 0, offs_n};
        tile_load(part, own_input + at + r * slice, n);
        tile_store(own_staging.data() + r * slice, n, part);
      }
    }
    p2p::barrier<WORLD, p2p::Among::peers, p2p::Ensure::visible>(
        peer_signals, self_signal, rank, timeout_ticks);
    block_stamp(1);

    // 2. Reduce-scatter: this rank's slice from every rank's staging, into its scratch.
    for (int offs_n = blockIdx.x * Chunk::kTileN; offs_n < mine;
         offs_n += gridDim.x * Chunk::kTileN) {
      Chunk peers[WORLD];
#pragma unroll
      for (int w = 0; w < WORLD; ++w) peers[w] = Chunk{1, mine, 0, offs_n};
      peers_load(peers, [&](int w) { return rotated[w]; }, mine);
      tile_store(own_scratch.data(), mine, peers_reduce(peers));
    }

    block_stamp(2);
    // 3. Every rank's sums are visible to its peers, and every peer has read this rank's staged
    //    pass, so the next one may overwrite it.
    p2p::barrier<WORLD, p2p::Among::peers, p2p::Ensure::visible>(
        peer_signals, self_signal, rank, timeout_ticks);
    block_stamp(3);

    // 4. All-gather: every rank's slice out of its scratch, at its place in the output.
    for (int offs_n = blockIdx.x * Chunk::kTileN; offs_n < slice;
         offs_n += gridDim.x * Chunk::kTileN) {
      Chunk got[WORLD];
#pragma unroll
      for (int r = 0; r < WORLD; ++r) got[r] = Chunk{1, slice, 0, offs_n};
      peers_load(got, [&](int r) { return scratches[r].data(); }, slice);
#pragma unroll
      for (int r = 0; r < WORLD; ++r) {
        got[r].N = max(0, min(slice, len - r * slice));
        if (got[r].N > 0) tile_store(out + at + r * slice, slice, got[r]);
      }
    }
    block_stamp(5);
  }
}

}  // namespace hip_comms
