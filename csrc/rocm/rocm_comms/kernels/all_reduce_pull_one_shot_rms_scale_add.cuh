// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce of a row [shared | projected | latent], then out = shared + projected *
// rsqrt(mean(latent^2) + eps): Kimi-K3's latent MoE tail with one all-reduce, its up-projection
// run on each rank's partial latent before it (`all_reduce_pull_one_shot_rms_scale_add`).

#pragma once

#include "../p2p/p2p.cuh"
#include "../common/common.cuh"

namespace hip_comms {

// A BLOCK OWNS ONE ROW'S SLICE OF THE HIDDEN, `splits` slices a row, so a row wider than a block
// takes several blocks rather than more registers; each reads the whole latent for the row's RMS
// (its 1/rms is the same in every slice). Rounds as the reference does: each span's sum lands as
// T, the all-reduce output, then out = T(float(shared) + float(projected) * scale).
template <typename T, int ngpus, int kRowPacks>
__global__ void __launch_bounds__(kMaxThreads, 1) all_reduce_pull_one_shot_rms_scale_add(
    p2p::DevComm p, T* __restrict__ out, float eps, int rows, int hidden_packs, int latent_packs,
    int splits) {
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  V* o                   = reinterpret_cast<V*>(out);
  const int packs        = 2 * hidden_packs + latent_packs;
  const int slice        = (hidden_packs + splits - 1) / splits;
  const float inv_latent = 1.0f / static_cast<float>(latent_packs * NL);
  const auto fl          = fragment<kRowPacks>(latent_packs);

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto inputs = p2p::inputs<T, ngpus>(p);
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

  // 2. Each of this block's (row, slice): the slice's shared and projected packs and the row's
  //    latent from every rank, all in flight together; the latent's sum of squares over the block,
  //    then the slice out.
  for (int w = blockIdx.x; w < rows * splits; w += gridDim.x) {
    const int row   = w / splits;
    const int first = (w % splits) * slice;
    const int len   = min(slice, hidden_packs - first);
    Fragment<kRowPacks> fh = fragment<kRowPacks>(len);
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) fh.at[k] += first;
    const auto sh = peers_load<T, ngpus>(shared, row, packs, fh);
    const auto pj = peers_load<T, ngpus>(proj, row, packs, fh);
    const auto lt = peers_load<T, ngpus>(latent, row, packs, fl);
    V l_sum[kRowPacks];
    peers_reduce(lt, l_sum);
    float l[kRowPacks][NL];
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) thread_unpack<T>(l_sum[k], l[k]);
    float ss[1] = {thread_dot(l, l, fl)};
    block_stamp(2);
    block_reduce<Sum>(ss);
    block_stamp(3);
    const float scale = rsqrtf(ss[0] * inv_latent + eps);
    V s_sum[kRowPacks], p_sum[kRowPacks];
    peers_reduce(sh, s_sum);
    peers_reduce(pj, p_sum);
    const int64_t base = int64_t{row} * hidden_packs;
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) {
      float s[NL], q[NL];
      thread_unpack<T>(s_sum[k], s);
      thread_unpack<T>(p_sum[k], q);
      V r;
#pragma unroll
      for (int j = 0; j < NL; ++j) r.d[j] = static_cast<T>(s[j] + q[j] * scale);
      if (fh.in[k] != 0.0f) o[base + fh.at[k]] = r;
    }
  }

  block_stamp(4);
  // 3. No rank may overwrite its input until every peer has read it.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::read>(p);
  block_stamp(5);
}

}  // namespace hip_comms
