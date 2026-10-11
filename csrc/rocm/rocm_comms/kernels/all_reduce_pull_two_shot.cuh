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

// THE PROBLEM, [R, M, N]: R = WORLD ranks' inputs, each an [inp_size_m, inp_size_n] buffer at its
// strides, summed elementwise. Reduce-scatter (each rank sums its share over the ranks into its
// scratch), sync, all-gather (every rank's summed share out to the output). The kernel states the
// problem; collectives.cuh walks it as [R, 1, M*N], each rank's share one contiguous range of the
// elements, for long reads from each peer (its comment says why and what it serves badly). A rank's
// copy of a chunk is a wave (LANES threads); the Ranks tile is that [R, 1, TILE_N] chunk.
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N, int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot(const DTYPE* const* __restrict__ inp_ptrs, int64_t inp_stride_m,
                             int64_t inp_stride_n, DTYPE* const* __restrict__ scratch_ptrs,
                             int64_t scratch_stride_m, int64_t scratch_stride_n,
                             Signal* const* __restrict__ signal_ptrs, Signal* self_signal_ptr,
                             int rank, uint64_t timeout_ticks, DTYPE* __restrict__ out_ptr,
                             int64_t out_stride_m, int64_t out_stride_n, int inp_size_m,
                             int inp_size_n) {
  Sync<WORLD> sync{signal_ptrs, self_signal_ptr, rank, timeout_ticks};
  constexpr int LANES = THREADS_PER_BLOCK / WORLD;  // a rank's copy's threads
  static_assert(LANES * WORLD == THREADS_PER_BLOCK, "a block is a group of threads a rank");
  static_assert(TILE_M == 1, "the walk's chunk is one row of the flat range");
  using Ranks = Tile<DTYPE, 1, TILE_N, 1, LANES, THREADS_PER_BLOCK, DTYPE, WORLD, WORLD>;

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto inp = [&](int r) {
    return rank_ptr<const DTYPE, WORLD>(inp_ptrs, r, inp_stride_m, inp_stride_n);
  };
  const auto scratch = [&](int r) {
    return rank_ptr<DTYPE, WORLD>(scratch_ptrs, r, scratch_stride_m, scratch_stride_n);
  };
  const auto out = local_ptr(out_ptr, out_stride_m, out_stride_n, rank);
  block_stamp(0);
  barrier<Group::peers, Until::launched>(sync);
  block_stamp(1);

  // 2. Reduce-scatter: this rank's share of the inputs, summed over the ranks, into its scratch.
  reduce_scatter<Ranks>(rank, inp, inp_size_m, inp_size_n, scratch(rank));

  block_stamp(2);
  // 3. Every rank's sums are visible to its peers.
  barrier<Group::peers, Until::visible>(sync);
  block_stamp(3);

  // 4. All-gather: every rank's summed share out of its scratch, at its place in the output. The
  //    next call's first sync keeps a rank from overwriting its scratch while it is read.
  all_gather<Ranks>(rank, scratch, out, inp_size_m, inp_size_n);
  block_stamp(5);
  sync.finish();
}

// STAGED: each rank copies its input into its staging a band of `band_m` rows at a time and each
// pass reads the peers' staging, so any size runs in one launch. A band is the problem's rows
// [r0, r0 + band) at their strides, the staging ([band, n], ours) and the scratch (a share of it)
// at most what each holds (the host's). EACH BLOCK STAGES, OF EVERY RANK'S SHARE, THE CHUNKS THAT
// RANK'S SAME BLOCK READS, so a peers barrier makes them visible.
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N, int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot_staged(DTYPE* const* __restrict__ scratch_ptrs,
                                    int64_t scratch_stride_m, int64_t scratch_stride_n,
                                    DTYPE* const* __restrict__ staging_ptrs,
                                    int64_t staging_stride_m, int64_t staging_stride_n,
                                    Signal* const* __restrict__ signal_ptrs,
                                    Signal* self_signal_ptr, int rank, uint64_t timeout_ticks,
                                    DTYPE* __restrict__ out_ptr, int64_t out_stride_m,
                                    int64_t out_stride_n, const DTYPE* __restrict__ inp_ptr,
                                    int64_t inp_stride_m, int64_t inp_stride_n, int inp_size_m,
                                    int inp_size_n, int band_m) {
  Sync<WORLD> sync{signal_ptrs, self_signal_ptr, rank, timeout_ticks};
  constexpr int LANES = THREADS_PER_BLOCK / WORLD;
  static_assert(LANES * WORLD == THREADS_PER_BLOCK, "a block is a group of threads a rank");
  static_assert(TILE_M == 1, "the walk's chunk is one row of the flat range");
  using Ranks = Tile<DTYPE, 1, TILE_N, 1, LANES, THREADS_PER_BLOCK, DTYPE, WORLD, WORLD>;
  const auto staging = [&](int r) {
    return rank_ptr<DTYPE, WORLD>(staging_ptrs, r, staging_stride_m, staging_stride_n);
  };
  const auto scratch = [&](int r) {
    return rank_ptr<DTYPE, WORLD>(scratch_ptrs, r, scratch_stride_m, scratch_stride_n);
  };

  for (int r0 = 0; r0 < inp_size_m; r0 += band_m) {
    const int band = min(band_m, inp_size_m - r0);  // this pass's rows
    // This pass's rows of the input and of the output.
    const auto inp = local_ptr(inp_ptr + r0 * inp_stride_m, inp_stride_m, inp_stride_n, rank);
    const auto out = local_ptr(out_ptr + r0 * out_stride_m, out_stride_m, out_stride_n, rank);
    block_stamp(0);
    // 1. This band, every rank's share of it, into this rank's staging (the chunks each rank's same
    //    block reads), then visible (and, past the first pass, every peer has read this rank's
    //    scratch).
    stage<Ranks>(rank, inp, staging(rank), band, inp_size_n);
    barrier<Group::peers, Until::visible>(sync);
    block_stamp(1);

    // 2. Reduce-scatter: this rank's share of the band from every rank's staging, into its scratch.
    reduce_scatter<Ranks>(rank, staging, band, inp_size_n, scratch(rank));

    block_stamp(2);
    // 3. Every rank's sums are visible to its peers, and every peer has read this rank's staged
    //    band, so the next one may overwrite it.
    barrier<Group::peers, Until::visible>(sync);
    block_stamp(3);

    // 4. All-gather: every rank's summed share out of its scratch, at its place in the output.
    all_gather<Ranks>(rank, scratch, out, band, inp_size_n);
    block_stamp(5);
  }
  sync.finish();
}

}  // namespace hip_comms
