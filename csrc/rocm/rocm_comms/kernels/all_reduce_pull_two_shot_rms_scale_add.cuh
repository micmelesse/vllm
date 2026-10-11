// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce of a row [shared | projected | latent], then out = shared + projected *
// rsqrt(mean(latent^2) + eps) (`all_reduce_pull_two_shot_rms_scale_add`). THE SLICE IS ROWS: each
// rank reduces its rows and leaves them FINISHED (hidden wide) in its scratch, and every rank
// gathers every owner's. The gather moves output rows, not input rows (7168 against 17920 wide in
// Kimi-K3), so it moves less than a two-shot all-reduce of the input. The one-shot is
// all_reduce_pull_one_shot_rms_scale_add, for fewer rows than ranks.

#pragma once

#include "../common/interface.cuh"

namespace hip_comms {

// A BLOCK OWNS ONE (ROW, SLICE) OF THIS RANK'S ROWS, `splits` slices a row, as in the one-shot.
// THE SAME BLOCK AND THREAD INDEX A PACK IN BOTH PHASES: a block gathers exactly the (row, slice)s
// the same block on each owner finished, which is what a peers barrier makes visible.
// The input is [m, 2 x n + latent_size_n] (shared and projected n wide, the latent
// latent_size_n), the output [m, n].
template <typename DTYPE, int WORLD, int TILE_N, int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot_rms_scale_add(
        const DTYPE* const* __restrict__ inp_ptrs, int64_t inp_stride_m, int64_t inp_stride_n,
        DTYPE* const* __restrict__ scratch_ptrs, int64_t scratch_stride_m, int64_t scratch_stride_n,
        Signal* const* __restrict__ signal_ptrs, Signal* self_signal_ptr, int rank,
        uint64_t timeout_ticks,
        DTYPE* __restrict__ out_ptr, int64_t out_stride_m, int64_t out_stride_n, float eps,
        int inp_size_m, int inp_size_n, int latent_size_n) {
  Sync<WORLD> sync{signal_ptrs, self_signal_ptr, rank, timeout_ticks};
  constexpr int NL       = traits<DTYPE>::N;
  using Row              = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  using RowF             = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK, float>;
  const int hidden_packs = inp_size_n / NL;  // the hidden in packs, as its slices are cut
  // A ROW'S COLUMN TILES: its hidden in TILE_N slices, spread evenly over as many.
  const int splits       = (hidden_packs + TILE_N / NL - 1) / (TILE_N / NL);
  const int slice        = (hidden_packs + splits - 1) / splits;
  const float inv_latent = 1.0f / static_cast<float>(latent_size_n);
  const auto out        = local_ptr(out_ptr, out_stride_m, out_stride_n, rank);
  const int slice_rows   = (inp_size_m + WORLD - 1) / WORLD;
  // Rank r's rows: [r x slice_rows, its last), the last rank's fewer (or none).
  const auto rows_of     = [&](int r) { return max(0, min(slice_rows,
      inp_size_m - r * slice_rows)); };
  // (Row, slice) w's tile of the hidden at row `row`: one row cut to the slice's columns.
  const auto slice_of    = [&](int w, int row) {
    const int first = (w % splits) * slice;
    return Row{inp_size_m, min(first + slice, hidden_packs) * NL, row, first * NL};
  };

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto inp      = rank_ptrs<const DTYPE, WORLD>(inp_ptrs, inp_stride_m, inp_stride_n);
  const auto own_scratch = rank_ptr<DTYPE, WORLD>(scratch_ptrs, rank, scratch_stride_m,
  scratch_stride_n);
  const auto scratch   = rank_ptrs<DTYPE, WORLD>(scratch_ptrs, scratch_stride_m, scratch_stride_n);
  block_stamp(0);
  barrier<Group::peers, Until::launched>(sync);
  block_stamp(1);
  // EACH RANK'S THREE SPANS, the shared, the projected and the latent, as Ptrs into its input.
  const auto& shared = inp;
  std::array<Ptr<const DTYPE>, WORLD> proj, latent;
#pragma unroll
  for (int r = 0; r < WORLD; ++r) {
    proj[r]   = Ptr<const DTYPE>{inp[r].data + inp_size_n * inp_stride_n, inp_stride_m,
        inp_stride_n, r};
    latent[r] = Ptr<const DTYPE>{inp[r].data + 2 * inp_size_n * inp_stride_n, inp_stride_m,
        inp_stride_n, r};
  }

  // 2. This rank's rows, finished: each (row, slice) from every rank, the latent's sum of squares
  //    over the block, then the slice into this rank's scratch, at the row's place among its own.
  const int first_row = rank * slice_rows;
  for (int w = blockIdx.x; w < rows_of(rank) * splits; w += gridDim.x) {
    const int row                = first_row + w / splits;
    const Row hid = slice_of(w, row);
    const Row lat{inp_size_m, latent_size_n, row, 0};
    Row sh[WORLD], pj[WORLD], lt[WORLD];
#pragma unroll
    for (int r = 0; r < WORLD; ++r) {
      sh[r] = hid;
      pj[r] = hid;
      lt[r] = lat;
    }
    tile_load(sh, shared);
    tile_load(pj, proj);
    tile_load(lt, latent);
    const RowF l = peers_reduce(lt).template to<float>();
    float ss[1];
    partial_dot(l, l, ss);
    block_stamp(2);
    block_reduce<Sum, THREADS_PER_BLOCK>(ss);
    const float scale = rsqrtf(ss[0] * inv_latent + eps);
    const RowF s = peers_reduce(sh).template to<float>();
    const RowF q = peers_reduce(pj).template to<float>();
    Row r = tile_add(s, tile_mul(q, scale)).template to<DTYPE>();
    r.offs_m = w / splits;  // at the row's place among this rank's
    tile_store(r, own_scratch);
  }

  block_stamp(3);
  // 3. Every rank's finished rows are visible to its peers, and every peer has read this rank's
  //    input.
  barrier<Group::peers, Until::visible>(sync);
  block_stamp(4);

  // 4. Every owner's finished rows out of its scratch, at their place in the output: this block's
  //    (row, slice)s of each. EVERY OWNER'S PACKS LOADED BEFORE ANY IS STORED. The next call's
  //    first sync keeps a rank from overwriting its scratch while it is read.
  for (int w = blockIdx.x; w < slice_rows * splits; w += gridDim.x) {
    const int l = w / splits;
    Row got[WORLD];
#pragma unroll
    for (int r = 0; r < WORLD; ++r) got[r] = slice_of(w, l);
    tile_load(got, scratch);
#pragma unroll
    for (int r = 0; r < WORLD; ++r) {
      if (l >= rows_of(r)) continue;
      got[r].offs_m = r * slice_rows + l;
      tile_store(got[r], out);
    }
  }
  block_stamp(5);
  sync.finish();
}

}  // namespace hip_comms
