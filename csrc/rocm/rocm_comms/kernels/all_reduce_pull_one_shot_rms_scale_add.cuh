// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce of a row [shared | projected | latent], then out = shared + projected *
// rsqrt(mean(latent^2) + eps): Kimi-K3's latent MoE tail with one all-reduce, its up-projection
// run on each rank's partial latent before it (`all_reduce_pull_one_shot_rms_scale_add`).

#pragma once

#include "../common/interface.cuh"

namespace hip_comms {

// A BLOCK OWNS ONE ROW'S SLICE OF THE HIDDEN, `splits` slices a row (its TILE_N column tiles), so
// a row wider than a block takes several blocks rather than more registers; each reads the whole
// latent for the row's RMS (its 1/rms is the same in every slice). Rounds as the reference does:
// each span's sum lands as T, the all-reduce output, then out = T(float(shared) +
// float(projected) * scale).
// The input is [m, 2 x n + latent_size_n] (shared and projected n wide, the latent
// latent_size_n), the output [m, n].
template <typename DTYPE, int WORLD, int TILE_N, int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot_rms_scale_add(
        const DTYPE* const* __restrict__ inp_ptrs, int64_t inp_stride_m, int64_t inp_stride_n,
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

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto inp = rank_ptrs<const DTYPE, WORLD>(inp_ptrs, inp_stride_m, inp_stride_n);
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

  // 2. Each of this block's (row, slice): the slice's shared and projected packs and the row's
  //    latent from every rank, all in flight together; the latent's sum of squares over the block,
  //    then the slice out.
  for (int w = blockIdx.x; w < inp_size_m * splits; w += gridDim.x) {
    const int row   = w / splits;
    const int first = (w % splits) * slice;
    const int len   = min(slice, hidden_packs - first);
    const Row hid{inp_size_m, (first + len) * NL, row, first * NL};
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
    block_stamp(3);
    const float scale = rsqrtf(ss[0] * inv_latent + eps);
    const RowF s = peers_reduce(sh).template to<float>();
    const RowF q = peers_reduce(pj).template to<float>();
    Row r = tile_add(s, tile_mul(q, scale)).template to<DTYPE>();
    tile_store(r, out);
  }

  block_stamp(4);
  // 3. No rank may overwrite its input until every peer has read it.
  barrier<Group::peers, Until::read>(sync);
  block_stamp(5);
  sync.finish();
}

}  // namespace hip_comms
