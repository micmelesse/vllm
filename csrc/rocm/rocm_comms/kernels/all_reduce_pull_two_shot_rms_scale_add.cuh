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
template <typename DTYPE, int WORLD, int TILE_N, int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot_rms_scale_add(const PeerPtrs* __restrict__ peer_inputs,
                                           PeerPtrs peer_scratch,
                                           PeerSignals peer_signals, Signal* self_signal,
                                           int rank, uint64_t timeout_ticks, DTYPE* __restrict__ out,
                                           float eps, int rows,
                                           int hidden_packs, int latent_packs) {
  Sync<WORLD> sync{peer_signals, self_signal, rank, timeout_ticks};
  constexpr int NL       = traits<DTYPE>::N;
  using Row              = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  using RowF             = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK, float>;
  const int packs        = 2 * hidden_packs + latent_packs;
  const int hidden       = hidden_packs * NL;  // in elements
  const int64_t stride   = int64_t{packs} * NL;
  // A ROW'S COLUMN TILES: its hidden in TILE_N slices, spread evenly over as many.
  const int splits       = (hidden_packs + TILE_N / NL - 1) / (TILE_N / NL);
  const int slice        = (hidden_packs + splits - 1) / splits;
  const float inv_latent = 1.0f / static_cast<float>(latent_packs * NL);
  const int slice_rows   = (rows + WORLD - 1) / WORLD;
  // Rank r's rows: [r x slice_rows, its last), the last rank's fewer (or none).
  const auto rows_of     = [&](int r) { return max(0, min(slice_rows, rows - r * slice_rows)); };
  // (Row, slice) w's tile of the hidden at row `row`: one row cut to the slice's columns.
  const auto slice_of    = [&](int w, int row) {
    const int first = (w % splits) * slice;
    return Row{rows, min(first + slice, hidden_packs) * NL, row, first * NL};
  };

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto inputs      = rank_ptrs<const DTYPE, WORLD>(*peer_inputs, stride);
  const auto own_scratch = rank_ptr<DTYPE, WORLD>(peer_scratch, rank, hidden);
  const auto scratches   = rank_ptrs<DTYPE, WORLD>(peer_scratch, hidden);
  block_stamp(0);
  barrier<Group::peers, Until::launched>(sync);
  block_stamp(1);
  // EACH RANK'S THREE SPANS, the shared, the projected and the latent, as Ptrs into its input.
  const auto& shared = inputs;
  std::array<Ptr<const DTYPE>, WORLD> proj, latent;
#pragma unroll
  for (int r = 0; r < WORLD; ++r) {
    proj[r]   = Ptr<const DTYPE>{inputs[r].data + hidden, stride, r};
    latent[r] = Ptr<const DTYPE>{inputs[r].data + 2 * hidden, stride, r};
  }

  // 2. This rank's rows, finished: each (row, slice) from every rank, the latent's sum of squares
  //    over the block, then the slice into this rank's scratch, at the row's place among its own.
  const int first_row = rank * slice_rows;
  for (int w = blockIdx.x; w < rows_of(rank) * splits; w += gridDim.x) {
    const int row                = first_row + w / splits;
    const Row hid = slice_of(w, row);
    const Row lat{rows, latent_packs * NL, row, 0};
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
    tile_load(got, scratches);
#pragma unroll
    for (int r = 0; r < WORLD; ++r) {
      if (l >= rows_of(r)) continue;
      got[r].offs_m = r * slice_rows + l;
      tile_store(got[r], local_ptr(out, hidden, rank));
    }
  }
  block_stamp(5);
  sync.finish();
}

}  // namespace hip_comms
