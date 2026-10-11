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

// THE BUFFER AS [m, n] AT ITS STRIDES, EVERY ROW CUT INTO ONE COLUMN SLICE A RANK (the last
// one short, as the push two-shot norms cut theirs), so any number of rows splits over every rank.
// A work item is TILE_M rows of a TILE_N-wide chunk of a slice, moved as a tile whose ROWS ARE THE
// RANKS: row w is rank + w's (ROTATED, SO THE RANKS SPREAD OVER THE LINKS: at any moment the eight
// GPUs read eight different peers, where in rank order every GPU reads rank 0 first), a row's
// threads a wave (THREADS_PER_BLOCK / WORLD lanes) each loading one group. A rank sums its slice's
// chunk over the rows (block_reduce over M, in row order, the rows meeting in LDS) into its scratch
// ([m, slice], ours, at its strides), then copies every rank's summed chunk out. The sum's order
// differs by rank, which is harmless: each slice is summed by one rank, so every rank copies the
// same bytes. THE SAME BLOCK READS IN THE SECOND PHASE WHAT THE SAME BLOCK ON EACH RANK WROTE IN
// THE FIRST: both count a row's chunks by the full slice and stride over the work items alike.
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
  constexpr int LANES = THREADS_PER_BLOCK / WORLD;
  static_assert(LANES * WORLD == THREADS_PER_BLOCK, "a block is a row of threads a rank");
  // A ROW IS WHOLE WAVES, so its rank is the same across each wave and rank_ptr's scalar select
  // is right; a wave holding several ranks' rows would read the first lane's rank in every lane.
  static_assert(LANES % kWaveSize == 0, "a rank's row of threads is whole waves");
  using Ranks       = Tile<DTYPE, WORLD, TILE_N, WORLD, LANES, THREADS_PER_BLOCK>;
  const int slice   = (inp_size_n / NL + WORLD - 1) / WORLD * NL;  // a rank's columns, in elements
  const int first   = rank * slice;
  const int mine    = max(0, min(slice, inp_size_n - first));      // a late rank's may be short
  const int chunks  = (slice + TILE_N - 1) / TILE_N;      // a row's, by the full slice
  const int items   = (inp_size_m + TILE_M - 1) / TILE_M * chunks;
  const auto rotated = [&](int w) { return (rank + w) % WORLD; };
  const auto slice_n = [&](int w) { return max(0, min(slice, inp_size_n - rotated(w) * slice)); };

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  // A ROW'S RANK WHEN IT IS READ: a row is whole waves, so its pointer is one scalar select (every
  // rank's, rotated, up front was ~20 scalar loads before the first barrier).
  const auto own_scratch =
      rank_ptr<DTYPE, WORLD>(scratch_ptrs, rank, scratch_stride_m, scratch_stride_n);
  const auto input = [&](int w) {
    return rank_ptr<const DTYPE, WORLD>(inp_ptrs, rotated(w), inp_stride_m, inp_stride_n);
  };
  const auto scratch = [&](int w) {
    return rank_ptr<DTYPE, WORLD>(scratch_ptrs, rotated(w), scratch_stride_m, scratch_stride_n);
  };
  block_stamp(0);
  barrier<Group::peers, Until::launched>(sync);
  block_stamp(1);

  // 2. Reduce-scatter: this rank's slice of each of the item's rows, summed over every rank, into
  //    its scratch.
  for (int t = blockIdx.x; t < items; t += gridDim.x) {
    const int row0 = t / chunks * TILE_M, offs_n = t % chunks * TILE_N;
#pragma unroll
    for (int i = 0; i < TILE_M; ++i) {
      const int row = row0 + i;
      if (row >= inp_size_m) break;
      Ranks got{WORLD, mine, 0, offs_n};
      reduce_scatter(
          got,
          [&](int w) {
            const auto p = input(w);
            return p.data + row * p.stride_m + first * p.stride_n;
          },
          own_scratch.data + row * own_scratch.stride_m);
    }
  }

  block_stamp(2);
  // 3. Every rank's sums are visible to its peers.
  barrier<Group::peers, Until::visible>(sync);
  block_stamp(3);

  // 4. All-gather: every rank's summed chunk of each of the item's rows out of its scratch, at its
  //    place in the output. The next call's first sync keeps a rank from overwriting its scratch
  //    while it is read.
  for (int t = blockIdx.x; t < items; t += gridDim.x) {
    const int row0 = t / chunks * TILE_M, offs_n = t % chunks * TILE_N;
#pragma unroll
    for (int i = 0; i < TILE_M; ++i) {
      const int row = row0 + i;
      if (row >= inp_size_m) break;
      Ranks got{WORLD, slice, 0, offs_n};
      all_gather(
          got,
          [&](int w) {
            const auto p = scratch(w);
            return p.data + row * p.stride_m;
          },
          [&](int w) { return out_ptr + row * out_stride_m + rotated(w) * slice * out_stride_n; },
          slice_n);
    }
  }
  block_stamp(5);
  sync.finish();
}

// STAGED: each rank copies its input into its staging a band of `band_m` rows at a time and each
// pass reads the peers' staging, so any size runs in one launch. The staging ([band, n]) and
// the scratch ([band, slice]) are ours, at their strides; a band is at most what each holds (the
// host's). EACH BLOCK STAGES, OF EVERY RANK'S SLICE, THE CHUNKS THAT RANK'S SAME BLOCK READS, so a
// peers barrier makes them visible.
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
  static_assert(LANES * WORLD == THREADS_PER_BLOCK, "a block is a row of threads a rank");
  // A ROW IS WHOLE WAVES, so its rank is the same across each wave and rank_ptr's scalar select
  // is right; a wave holding several ranks' rows would read the first lane's rank in every lane.
  static_assert(LANES % kWaveSize == 0, "a rank's row of threads is whole waves");
  using Ranks       = Tile<DTYPE, WORLD, TILE_N, WORLD, LANES, THREADS_PER_BLOCK>;
  const int slice   = (inp_size_n / NL + WORLD - 1) / WORLD * NL;
  const int first   = rank * slice;
  const int mine    = max(0, min(slice, inp_size_n - first));
  const int chunks  = (slice + TILE_N - 1) / TILE_N;
  const auto rotated = [&](int w) { return (rank + w) % WORLD; };
  const auto slice_n = [&](int w) { return max(0, min(slice, inp_size_n - rotated(w) * slice)); };
  const auto own_scratch =
      rank_ptr<DTYPE, WORLD>(scratch_ptrs, rank, scratch_stride_m, scratch_stride_n);
  const auto own_staging =
      rank_ptr<DTYPE, WORLD>(staging_ptrs, rank, staging_stride_m, staging_stride_n);
  const auto staging = [&](int w) {
    return rank_ptr<DTYPE, WORLD>(staging_ptrs, rotated(w), staging_stride_m, staging_stride_n);
  };
  const auto scratch = [&](int w) {
    return rank_ptr<DTYPE, WORLD>(scratch_ptrs, rotated(w), scratch_stride_m, scratch_stride_n);
  };

  for (int r0 = 0; r0 < inp_size_m; r0 += band_m) {
    const int band  = min(band_m, inp_size_m - r0);  // this pass's rows
    const int items = (band + TILE_M - 1) / TILE_M * chunks;
    block_stamp(0);
    // 1. Every rank's slice of this band's rows into this rank's staging, the chunks each rank's
    //    same block reads, then visible (and, past the first pass, every peer has read this rank's
    //    scratch).
    for (int t = blockIdx.x; t < items; t += gridDim.x) {
      const int b0 = t / chunks * TILE_M, offs_n = t % chunks * TILE_N;
#pragma unroll
      for (int i = 0; i < TILE_M; ++i) {
        const int b = b0 + i;  // the band's row
        if (b >= band) break;
        Ranks part{WORLD, slice, 0, offs_n};
        tile_gather(
            part,
            [&](int w) {
              return inp_ptr + (r0 + b) * inp_stride_m + rotated(w) * slice * inp_stride_n;
            },
            slice_n);
        tile_scatter(
            [&](int w) {
              return own_staging.data + b * own_staging.stride_m +
                     rotated(w) * slice * own_staging.stride_n;
            },
            slice_n, part);
      }
    }
    barrier<Group::peers, Until::visible>(sync);
    block_stamp(1);

    // 2. Reduce-scatter: this rank's slice of the band from every rank's staging, into its scratch.
    for (int t = blockIdx.x; t < items; t += gridDim.x) {
      const int b0 = t / chunks * TILE_M, offs_n = t % chunks * TILE_N;
#pragma unroll
      for (int i = 0; i < TILE_M; ++i) {
        const int b = b0 + i;
        if (b >= band) break;
        Ranks got{WORLD, mine, 0, offs_n};
        reduce_scatter(
            got,
            [&](int w) {
              const auto p = staging(w);
              return p.data + b * p.stride_m + first * p.stride_n;
            },
            own_scratch.data + b * own_scratch.stride_m);
      }
    }

    block_stamp(2);
    // 3. Every rank's sums are visible to its peers, and every peer has read this rank's staged
    //    band, so the next one may overwrite it.
    barrier<Group::peers, Until::visible>(sync);
    block_stamp(3);

    // 4. All-gather: every rank's summed chunk out of its scratch, at its place in the output.
    for (int t = blockIdx.x; t < items; t += gridDim.x) {
      const int b0 = t / chunks * TILE_M, offs_n = t % chunks * TILE_N;
#pragma unroll
      for (int i = 0; i < TILE_M; ++i) {
        const int b = b0 + i;
        if (b >= band) break;
        Ranks got{WORLD, slice, 0, offs_n};
        all_gather(
            got,
            [&](int w) {
              const auto p = scratch(w);
              return p.data + b * p.stride_m;
            },
            [&](int w) {
              return out_ptr + (r0 + b) * out_stride_m + rotated(w) * slice * out_stride_n;
            },
            slice_n);
      }
    }
    block_stamp(5);
  }
  sync.finish();
}

}  // namespace hip_comms
