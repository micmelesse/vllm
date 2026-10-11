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

// THE DECOMPOSITION, 3-D: RANKS x ROWS x COLUMNS. The buffer is [m, n] at its strides, every row cut
// into one column slice a rank (the last one short), so any number of rows splits over every rank.
// A work item is a 3-D tile, R x TILE_M x TILE_N: TILE_M rows of a TILE_N-wide chunk of a slice,
// and R the ranks, copy r rank + r's (ROTATED, SO THE RANKS SPREAD OVER THE LINKS: at any moment
// the eight GPUs read eight different peers, where in rank order every GPU reads rank 0 first),
// a copy a wave (THREADS_PER_BLOCK / WORLD threads, laid TILE_M x LANES / TILE_M over its rows and
// columns). A rank sums its slice's chunk over the ranks (block_reduce over R, in rank order,
// meeting in LDS) into its scratch ([m, slice], ours, at its strides), then copies every rank's
// summed chunk out. The sum's order differs by rank, which is harmless: each slice is summed by
// one rank, so every rank copies the same bytes. THE SAME BLOCK READS IN THE SECOND PHASE WHAT THE
// SAME BLOCK ON EACH RANK WROTE IN THE FIRST: both walk the same tiles of the full slice alike.
// WHY 3-D: the ranks are the axis the sum runs over, so they are the tile's own (R), not folded into
// its rows; and TILE_M rows let a slice narrower than the threads fill them. A tile of one row
// holds TILE_N columns, so a slice that is not whole tiles leaves its last chunk part empty (7168
// columns: a rank's 896 as a 512 and a 384-column chunk, the threads 87.5% busy; 2026-10-11); a
// tile of TILE_M rows of the whole slice (4 x 896 at 7168, 8 x 448 at 3584) is 7 groups a thread,
// every thread busy, and each peer read in runs of TILE_M rows. IT SERVES BADLY a call of fewer
// rows than TILE_M (decode's 1-8 tokens): the tile's other rows are empty, so the tuned list gives
// those TILE_M 1 or the one-shot.
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N, int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot(const DTYPE* const* __restrict__ inp_ptrs, int64_t inp_stride_m,
                             int64_t inp_stride_n, DTYPE* const* __restrict__ scratch_ptrs,
                             int64_t scratch_stride_m,
                             int64_t scratch_stride_n, Signal* const* __restrict__ signal_ptrs,
                             Signal* self_signal_ptr, int rank, uint64_t timeout_ticks,
                             DTYPE* __restrict__ out_ptr, int64_t out_stride_m,
                             int64_t out_stride_n, int inp_size_m, int inp_size_n) {
  Sync<WORLD> sync{signal_ptrs, self_signal_ptr, rank, timeout_ticks};
  constexpr int NL    = traits<DTYPE>::N;
  constexpr int LANES = THREADS_PER_BLOCK / WORLD;  // a rank's copy's threads
  static_assert(LANES * WORLD == THREADS_PER_BLOCK, "a block is a group of threads a rank");
  static_assert(LANES % TILE_M == 0, "a rank's threads lay a row of threads a tile row");
  using Chunk = Tile<DTYPE, TILE_M, TILE_N, TILE_M, LANES / TILE_M, THREADS_PER_BLOCK, DTYPE, WORLD,
                     WORLD>;
  const int slice   = (inp_size_n / NL + WORLD - 1) / WORLD * NL;  // a rank's columns, in elements
  const int first   = rank * slice;
  const int mine    = max(0, min(slice, inp_size_n - first));      // a late rank's may be short
  const int chunks  = (slice + TILE_N - 1) / TILE_N;      // a row's, by the full slice
  const int tiles_m = (inp_size_m + TILE_M - 1) / TILE_M;
  const auto rotated = [&](int w) { return (rank + w) % WORLD; };
  const auto slice_n = [&](int w) { return max(0, min(slice, inp_size_n - rotated(w) * slice)); };

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  // A COPY'S RANK WHEN IT IS READ: a copy is a wave, so its pointer is one scalar select (every
  // rank's, rotated, up front was ~20 scalar loads before the first barrier).
  const auto own_scratch =
      rank_ptr<DTYPE, WORLD>(scratch_ptrs, rank, scratch_stride_m, scratch_stride_n);
  const auto input = [&](int w) {  // rank + w's copy of this rank's slice
    auto p = rank_ptr<const DTYPE, WORLD>(inp_ptrs, rotated(w), inp_stride_m, inp_stride_n);
    p.data += first * p.stride_n;
    return p;
  };
  const auto scratch = [&](int w) {  // rank + w's summed slice
    return rank_ptr<DTYPE, WORLD>(scratch_ptrs, rotated(w), scratch_stride_m, scratch_stride_n);
  };
  const auto out = [&](int w) {  // where rank + w's slice goes
    return local_ptr(out_ptr + rotated(w) * slice * out_stride_n, out_stride_m, out_stride_n, rank);
  };
  block_stamp(0);
  barrier<Group::peers, Until::launched>(sync);
  block_stamp(1);

  // 2. Reduce-scatter: this rank's slice of each tile's rows, summed over every rank, into its
  //    scratch.
  block_for_each_tile(tiles_m, chunks, [&](int tile_row, int chunk) {
    Chunk got{inp_size_m, mine, tile_row * TILE_M, chunk * TILE_N};
    reduce_scatter(got, input, own_scratch);
  });

  block_stamp(2);
  // 3. Every rank's sums are visible to its peers.
  barrier<Group::peers, Until::visible>(sync);
  block_stamp(3);

  // 4. All-gather: every rank's summed chunk of each tile's rows out of its scratch, at its place in
  //    the output. The next call's first sync keeps a rank from overwriting its scratch while it is
  //    read.
  block_for_each_tile(tiles_m, chunks, [&](int tile_row, int chunk) {
    Chunk got{inp_size_m, slice, tile_row * TILE_M, chunk * TILE_N};
    all_gather(got, scratch, out, slice_n);
  });
  block_stamp(5);
  sync.finish();
}

// STAGED: each rank copies its input into its staging a band of `band_m` rows at a time and each
// pass reads the peers' staging, so any size runs in one launch. The staging ([band, n]) and
// the scratch ([band, slice]) are ours, at their strides; a band is at most what each holds (the
// host's). EACH BLOCK STAGES, OF EVERY RANK'S SLICE, THE CHUNKS THAT RANK'S SAME BLOCK READS, so a
// peers barrier makes them visible. The tile as the in-place kernel's.
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N, int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot_staged(DTYPE* const* __restrict__ scratch_ptrs,
                                    int64_t scratch_stride_m,
                                    int64_t scratch_stride_n,
                                    DTYPE* const* __restrict__ staging_ptrs,
                                    int64_t staging_stride_m, int64_t staging_stride_n,
                                    Signal* const* __restrict__ signal_ptrs,
                                    Signal* self_signal_ptr, int rank,
                                    uint64_t timeout_ticks, DTYPE* __restrict__ out_ptr,
                                    int64_t out_stride_m, int64_t out_stride_n,
                                    const DTYPE* __restrict__ inp_ptr, int64_t inp_stride_m,
                                    int64_t inp_stride_n, int inp_size_m, int inp_size_n,
                                    int band_m) {
  Sync<WORLD> sync{signal_ptrs, self_signal_ptr, rank, timeout_ticks};
  constexpr int NL    = traits<DTYPE>::N;
  constexpr int LANES = THREADS_PER_BLOCK / WORLD;
  static_assert(LANES * WORLD == THREADS_PER_BLOCK, "a block is a group of threads a rank");
  static_assert(LANES % TILE_M == 0, "a rank's threads lay a row of threads a tile row");
  using Chunk = Tile<DTYPE, TILE_M, TILE_N, TILE_M, LANES / TILE_M, THREADS_PER_BLOCK, DTYPE, WORLD,
                     WORLD>;
  const int slice   = (inp_size_n / NL + WORLD - 1) / WORLD * NL;
  const int first   = rank * slice;
  const int mine    = max(0, min(slice, inp_size_n - first));
  const int chunks  = (slice + TILE_N - 1) / TILE_N;
  const auto rotated = [&](int w) { return (rank + w) % WORLD; };
  const auto slice_n = [&](int w) { return max(0, min(slice, inp_size_n - rotated(w) * slice)); };
  const auto own_scratch =
      rank_ptr<DTYPE, WORLD>(scratch_ptrs, rank, scratch_stride_m, scratch_stride_n);
  const auto staging = [&](int w) {  // rank + w's staged copy of this rank's slice
    auto p = rank_ptr<DTYPE, WORLD>(staging_ptrs, rotated(w), staging_stride_m, staging_stride_n);
    p.data += first * p.stride_n;
    return p;
  };
  const auto scratch = [&](int w) {
    return rank_ptr<DTYPE, WORLD>(scratch_ptrs, rotated(w), scratch_stride_m, scratch_stride_n);
  };

  for (int r0 = 0; r0 < inp_size_m; r0 += band_m) {
    const int band    = min(band_m, inp_size_m - r0);  // this pass's rows
    const int tiles_m = (band + TILE_M - 1) / TILE_M;
    // This pass's rows, each rank + w's slice of them: of the input, of this rank's staging, and
    // of the output.
    const auto inp_slice = [&](int w) {
      return local_ptr(inp_ptr + r0 * inp_stride_m + rotated(w) * slice * inp_stride_n,
                       inp_stride_m, inp_stride_n, rank);
    };
    const auto own_staging = [&](int w) {
      auto p = rank_ptr<DTYPE, WORLD>(staging_ptrs, rank, staging_stride_m, staging_stride_n);
      p.data += rotated(w) * slice * p.stride_n;
      return p;
    };
    const auto out = [&](int w) {
      return local_ptr(out_ptr + r0 * out_stride_m + rotated(w) * slice * out_stride_n,
                       out_stride_m, out_stride_n, rank);
    };
    block_stamp(0);
    // 1. Every rank's slice of this band's rows into this rank's staging, the chunks each rank's
    //    same block reads, then visible (and, past the first pass, every peer has read this rank's
    //    scratch).
    block_for_each_tile(tiles_m, chunks, [&](int tile_row, int chunk) {
      Chunk part{band, slice, tile_row * TILE_M, chunk * TILE_N};
      ranks_load(part, inp_slice, slice_n);
      ranks_store(own_staging, slice_n, part);
    });
    barrier<Group::peers, Until::visible>(sync);
    block_stamp(1);

    // 2. Reduce-scatter: this rank's slice of the band from every rank's staging, into its scratch.
    block_for_each_tile(tiles_m, chunks, [&](int tile_row, int chunk) {
      Chunk got{band, mine, tile_row * TILE_M, chunk * TILE_N};
      reduce_scatter(got, staging, own_scratch);
    });

    block_stamp(2);
    // 3. Every rank's sums are visible to its peers, and every peer has read this rank's staged
    //    band, so the next one may overwrite it.
    barrier<Group::peers, Until::visible>(sync);
    block_stamp(3);

    // 4. All-gather: every rank's summed chunk out of its scratch, at its place in the output.
    block_for_each_tile(tiles_m, chunks, [&](int tile_row, int chunk) {
      Chunk got{band, slice, tile_row * TILE_M, chunk * TILE_N};
      all_gather(got, scratch, out, slice_n);
    });
    block_stamp(5);
  }
  sync.finish();
}

}  // namespace hip_comms
