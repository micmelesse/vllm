// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE COLLECTIVES' STEPS, built from common's tiles: one tested and tuned way to do each, which the
// kernels call rather than write again (the fused AttnRes pull's own reduce-scatter took 1.6x the
// plain two-shot's cycles for its bytes, thread traces 2026-10-04T21-03-09Z and 21-26-11Z).

#pragma once

#ifndef HIP_COMMS_COMMON_INTERFACE
#error "include common/common.cuh, common's one interface, not its parts"
#endif

namespace hip_comms {

// THE LOADS ARE THE COMMUNICATION: `rank_chunk(w)` points into rank + w's memory, mapped into this
// GPU's (peers.cuh), so loading the chunk reads it over the links; the sum and the store are local.
// The kernel syncs around these calls (Sync): before, every peer's data is ready; between the
// reduce-scatter and the all-gather, every rank's sums are visible.
//
// THE COLLECTIVES WORK A CHUNK AT A TIME, as a tile whose ROWS ARE THE RANKS: row w is rank + w's
// copy of the chunk (ROTATED, so at any moment the GPUs read different peers), a row of threads a
// rank (TILE::kThreadsM == WORLD). `chunk` is the place: its N the slice's end, its offs_n the
// chunk's first column. A rank's pointer is picked a wave at a time (`rank_chunk(w)`, a scalar
// select): every rank's held up front was ~20 scalar loads before the first barrier.

// REDUCE-SCATTER: this rank's slice summed over every rank, a chunk a call.
//   rank_chunk(w)  where rank + w's copy of the chunk starts
//   sum_out        where this rank's summed chunk goes (its scratch)
// The sum's order differs by rank, which is harmless: each chunk is summed by one rank.
template <typename TILE, typename RANK_CHUNK>
DINLINE void reduce_scatter(TILE& chunk, RANK_CHUNK rank_chunk, typename TILE::Dtype* sum_out) {
  // 1. Every rank's copy of the chunk, a row each, in one round trip.
  tile_gather(chunk, rank_chunk);
  // 2. Summed over the rows (the ranks), in row order, through LDS.
  const auto sum = block_reduce<Sum, Axis::m>(chunk);
  // 3. Stored as DTYPE in this rank's slice.
  tile_store(sum_out, chunk.N, sum.template to<typename TILE::Dtype>());
}

// ALL-GATHER: every rank's summed slice into the output, a chunk a call.
//   rank_chunk(w)  where rank + w's summed copy of the chunk starts (its scratch)
//   out_chunk(w)   where that chunk goes in the output
//   out_len(w)     how many of its columns are real there (a late rank's slice is short)
template <typename TILE, typename RANK_CHUNK, typename OUT_CHUNK, typename OUT_LEN>
DINLINE void all_gather(TILE& chunk, RANK_CHUNK rank_chunk, OUT_CHUNK out_chunk, OUT_LEN out_len) {
  // 1. Every rank's summed copy of the chunk, a row each, in one round trip.
  tile_gather(chunk, rank_chunk);
  // 2. Each row stored where its rank's slice goes.
  tile_scatter(out_chunk, out_len, chunk);
}

}  // namespace hip_comms
