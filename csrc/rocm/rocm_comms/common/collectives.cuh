// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE COLLECTIVES' STEPS, built from common's tiles: one tested and tuned way to do each, which the
// kernels call rather than write again (the fused AttnRes pull's own reduce-scatter took 1.6x the
// plain two-shot's cycles for its bytes, thread traces 2026-10-04T21-03-09Z and 21-26-11Z).

#pragma once

#ifndef HIP_COMMS_COMMON_INTERFACE
#error "include common/interface.cuh, common's one interface, not its parts"
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
namespace impl {
template <typename TILE, typename RANK_CHUNK>
DINLINE void reduce_scatter(TILE& chunk, RANK_CHUNK rank_chunk, typename TILE::Dtype* sum_out) {
  // 1. Every rank's copy of the chunk, a row each, in one round trip.
  tile_gather(chunk, rank_chunk);
  // 2. Summed over the rows (the ranks), in row order, through LDS.
  const auto sum = block_reduce<Sum, Axis::m>(chunk);
  // 3. Stored as DTYPE in this rank's slice.
  tile_store(sum_out, chunk.N, sum.template to<typename TILE::Dtype>());
}
}  // namespace impl


// ON A 3-D TILE, R x M x N, THE RANKS ITS R AXIS: rank r's copy a group of threads (whole waves),
// read at that rank's strides through `rank_ptr(r)` (a Ptr where the chunk's column 0 is, ROTATED
// by the caller so the GPUs read different peers at once). A chunk is TILE_M rows of TILE_N
// columns of a slice, so a rank's slice of several rows fills the tile where one row of it would
// leave the last chunk part empty.

// REDUCE-SCATTER: this rank's chunk summed over every rank's copy, into `sum_out` (its scratch, a
// Ptr at the chunk's column 0).
namespace impl {
template <typename TILE, typename RANK_PTR>
DINLINE void reduce_scatter(TILE& chunk, RANK_PTR rank_ptr, const Ptr<typename TILE::Dtype>& sum_out) {
  // 1. Every rank's copy of the chunk, in one round trip.
  ranks_load(chunk, rank_ptr);
  // 2. Summed over the ranks, in rank order, through LDS.
  const auto sum = block_reduce<Sum, Axis::r>(chunk);
  // 3. Stored as DTYPE in this rank's slice.
  tile_store(sum_out.data, sum_out.stride_m, sum.template to<typename TILE::Dtype>());
}
}  // namespace impl

// ALL-GATHER: every rank's summed chunk into the output, a chunk a call.
//   rank_ptr(r)  rank r's summed chunk (its scratch)
//   out_ptr(r)   where that chunk goes in the output
//   out_n(r)     how many of its columns are real there (a late rank's slice is short)
namespace impl {
template <typename TILE, typename RANK_PTR, typename OUT_PTR, typename OUT_N>
DINLINE void all_gather(TILE& chunk, RANK_PTR rank_ptr, OUT_PTR out_ptr, OUT_N out_n) {
  // 1. Every rank's summed copy of the chunk, in one round trip.
  ranks_load(chunk, rank_ptr);
  // 2. Each rank's copy stored where its slice goes.
  ranks_store(out_ptr, out_n, chunk);
}
}  // namespace impl

}  // namespace hip_comms
