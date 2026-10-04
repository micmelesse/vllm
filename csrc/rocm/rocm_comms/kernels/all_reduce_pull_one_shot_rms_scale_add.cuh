// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce of a row [shared | projected | latent], then out = shared + projected *
// rsqrt(mean(latent^2) + eps): Kimi-K3's latent MoE tail with one all-reduce, its up-projection
// run on each rank's partial latent before it (`all_reduce_pull_one_shot_rms_scale_add`).

#pragma once

#include "../common/common.cuh"

namespace hip_comms {

// A BLOCK OWNS ONE ROW'S SLICE OF THE HIDDEN, `splits` slices a row (its TILE_N column tiles), so
// a row wider than a block takes several blocks rather than more registers; each reads the whole
// latent for the row's RMS (its 1/rms is the same in every slice). Rounds as the reference does:
// each span's sum lands as T, the all-reduce output, then out = T(float(shared) +
// float(projected) * scale).
template <typename DTYPE, int WORLD, int TILE_N, int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot_rms_scale_add(const PeerPtrs* __restrict__ peer_inputs,
                                           PeerSignals peer_signals, Signal* self_signal,
                                           int rank, uint64_t timeout_ticks, DTYPE* __restrict__ out,
                                           float eps, int rows,
                                           int hidden_packs, int latent_packs) {
  constexpr int NL       = traits<DTYPE>::N;
  using Row              = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK>;
  using RowF             = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, float>;
  const int packs        = 2 * hidden_packs + latent_packs;
  const int hidden       = hidden_packs * NL;  // in elements
  const int64_t stride   = int64_t{packs} * NL;
  // A ROW'S COLUMN TILES: its hidden in TILE_N slices, spread evenly over as many.
  const int splits       = (hidden_packs + TILE_N / NL - 1) / (TILE_N / NL);
  const int slice        = (hidden_packs + splits - 1) / splits;
  const float inv_latent = 1.0f / static_cast<float>(latent_packs * NL);

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto inputs = rank_inputs<DTYPE, WORLD>(*peer_inputs);
  block_stamp(0);
  barrier<WORLD, Among::peers, Ensure::launched>(
      peer_signals, self_signal, rank, timeout_ticks);
  block_stamp(1);
  const auto shared = [&](int r) { return inputs[r]; };
  const auto proj   = [&](int r) { return inputs[r] + hidden; };
  const auto latent = [&](int r) { return inputs[r] + 2 * hidden; };

  // 2. Each of this block's (row, slice): the slice's shared and projected packs and the row's
  //    latent from every rank, all in flight together; the latent's sum of squares over the block,
  //    then the slice out.
  for (int w = blockIdx.x; w < rows * splits; w += gridDim.x) {
    const int row   = w / splits;
    const int first = (w % splits) * slice;
    const int len   = min(slice, hidden_packs - first);
    const Row hid{rows, (first + len) * NL, row, first * NL};
    const Row lat{rows, latent_packs * NL, row, 0};
    Row sh[WORLD], pj[WORLD], lt[WORLD];
#pragma unroll
    for (int r = 0; r < WORLD; ++r) {
      sh[r] = hid;
      pj[r] = hid;
      lt[r] = lat;
    }
    peers_load(sh, shared, stride);
    peers_load(pj, proj, stride);
    peers_load(lt, latent, stride);
    const RowF l = peers_reduce(lt).template to<float>();
    float ss[1];
    partial_dot(l, l, ss);
    block_stamp(2);
    block_reduce<Sum, THREADS_PER_BLOCK>(ss);
    block_stamp(3);
    const float scale = rsqrtf(ss[0] * inv_latent + eps);
    const RowF s = peers_reduce(sh).template to<float>();
    const RowF q = peers_reduce(pj).template to<float>();
    Row r = tile_add(s, tile_mul(q, scale)).template to<DTYPE>();
    tile_store(out, hidden, r);
  }

  block_stamp(4);
  // 3. No rank may overwrite its input until every peer has read it.
  barrier<WORLD, Among::peers, Ensure::read>(peer_signals, self_signal, rank,
                                                            timeout_ticks);
  block_stamp(5);
}

}  // namespace hip_comms
