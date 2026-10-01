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

#include "../p2p/p2p.cuh"
#include "../common/common.cuh"

namespace hip_comms {

// A BLOCK OWNS ONE (ROW, SLICE) OF THIS RANK'S ROWS, `splits` slices a row, as in the one-shot.
// THE SAME BLOCK AND THREAD INDEX A PACK IN BOTH PHASES: a block gathers exactly the (row, slice)s
// the same block on each owner finished, which is what a peers barrier makes visible.
template <typename T, int ngpus, int kRowPacks>
__global__ void __launch_bounds__(kMaxThreads, 1) all_reduce_pull_two_shot_rms_scale_add(
    p2p::DevComm p, T* __restrict__ out, float eps, int rows, int hidden_packs, int latent_packs,
    int splits) {
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  V* o                   = reinterpret_cast<V*>(out);
  const int packs        = 2 * hidden_packs + latent_packs;
  const int slice        = (hidden_packs + splits - 1) / splits;
  const float inv_latent = 1.0f / static_cast<float>(latent_packs * NL);
  const auto fl          = fragment<kRowPacks>(latent_packs);
  const int slice_rows   = (rows + ngpus - 1) / ngpus;
  // Rank r's rows: [r x slice_rows, its last), the last rank's fewer (or none).
  const auto rows_of     = [&](int r) { return max(0, min(slice_rows, rows - r * slice_rows)); };
  // This (row, slice)'s packs of the hidden, in this thread's fragment.
  const auto slice_of    = [&](int w) {
    const int first        = (w % splits) * slice;
    Fragment<kRowPacks> fh = fragment<kRowPacks>(min(slice, hidden_packs - first));
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) fh.at[k] += first;
    return fh;
  };

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto inputs      = p2p::inputs<T, ngpus>(p);
  const auto own_scratch = p2p::scratch<T, ngpus>(p, p.rank);
  const auto scratches   = p2p::scratches<T, ngpus>(p);
  block_stamp(0);
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);
  block_stamp(1);
  const auto shared = [&](int r, int64_t i) { return p2p::read_input(inputs[r], i); };
  const auto proj   = [&](int r, int64_t i) {
    return p2p::read_input(inputs[r], i + hidden_packs);
  };
  const auto latent = [&](int r, int64_t i) {
    return p2p::read_input(inputs[r], i + 2 * hidden_packs);
  };

  // 2. This rank's rows, finished: each (row, slice) from every rank, the latent's sum of squares
  //    over the block, then the slice into this rank's scratch, at the row's place among its own.
  const int first_row = p.rank * slice_rows;
  for (int w = blockIdx.x; w < rows_of(p.rank) * splits; w += gridDim.x) {
    const int row                = first_row + w / splits;
    const Fragment<kRowPacks> fh = slice_of(w);
    const auto sh                = peers_load<T, ngpus>(shared, row, packs, fh);
    const auto pj                = peers_load<T, ngpus>(proj, row, packs, fh);
    const auto lt                = peers_load<T, ngpus>(latent, row, packs, fl);
    V l_sum[kRowPacks];
    peers_reduce(lt, l_sum);
    float l[kRowPacks][NL];
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) thread_unpack<T>(l_sum[k], l[k]);
    float ss[1] = {thread_dot(l, l, fl)};
    block_stamp(2);
    block_reduce<Sum>(ss);
    const float scale = rsqrtf(ss[0] * inv_latent + eps);
    V s_sum[kRowPacks], p_sum[kRowPacks];
    peers_reduce(sh, s_sum);
    peers_reduce(pj, p_sum);
    const int64_t at = int64_t{w / splits} * hidden_packs;
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) {
      float s[NL], q[NL];
      thread_unpack<T>(s_sum[k], s);
      thread_unpack<T>(p_sum[k], q);
      V r;
#pragma unroll
      for (int j = 0; j < NL; ++j) r.d[j] = static_cast<T>(s[j] + q[j] * scale);
      if (fh.in[k] != 0.0f) p2p::write_scratch(own_scratch, at + fh.at[k], r);
    }
  }

  block_stamp(3);
  // 3. Every rank's finished rows are visible to its peers, and every peer has read this rank's
  //    input.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::visible>(p);
  block_stamp(4);

  // 4. Every owner's finished rows out of its scratch, at their place in the output: this block's
  //    (row, slice)s of each. EVERY OWNER'S PACKS LOADED BEFORE ANY IS STORED. The next call's
  //    first sync keeps a rank from overwriting its scratch while it is read.
  for (int w = blockIdx.x; w < slice_rows * splits; w += gridDim.x) {
    const Fragment<kRowPacks> fh = slice_of(w);
    const int l                  = w / splits;
    V got[ngpus][kRowPacks];
#pragma unroll
    for (int r = 0; r < ngpus; ++r)
#pragma unroll
      for (int k = 0; k < kRowPacks; ++k)
        got[r][k] = p2p::read_scratch(scratches[r], int64_t{l} * hidden_packs + fh.at[k]);
#pragma unroll
    for (int r = 0; r < ngpus; ++r) {
      if (l >= rows_of(r)) continue;
      const int64_t base = (int64_t{r} * slice_rows + l) * hidden_packs;
#pragma unroll
      for (int k = 0; k < kRowPacks; ++k)
        if (fh.in[k] != 0.0f) thread_store(o + base + fh.at[k], got[r][k]);
    }
  }
  block_stamp(5);
}

}  // namespace hip_comms
