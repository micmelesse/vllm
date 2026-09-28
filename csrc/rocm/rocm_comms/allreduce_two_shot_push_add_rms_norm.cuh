// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot push all-reduce then RMSNorm (`allreduce_two_shot_push_rms_norm`), and
// all-reduce then add then RMSNorm (`allreduce_two_shot_push_add_rms_norm`): one body, a
// kernel per op, so a trace names the op that ran.

#pragma once

#include "fusions/add_rms_norm.cuh"
#include "p2p/p2p.cuh"
#include "utils.cuh"

namespace hip_comms {

// Each rank owns whole rows, as the pull kernel does, and every transfer is a store into
// a peer's slot: each input row, encoded, to its owner; barrier; the owner reduces, norms
// and shares the out row (encoded) and the residual row with every rank; barrier; every
// rank gathers. THE RESIDUAL IS NEVER QUANTIZED (16 bits whatever kBits): it is
// rewritten every layer, so a codec's error would accumulate over depth.
template <typename T, typename W, int ngpus, int kBits, bool kAdd>
DINLINE void two_shot_push_add_rms_norm_body(p2p::Peers p, T* __restrict__ out,
                                             T* __restrict__ residual_out,
                                             const T* __restrict__ residual,
                                             const W* __restrict__ weight, float eps,
                                             int rows, int packs) {
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  namespace fusion       = fusions::add_rms_norm;
  const auto w           = p2p::start<T, ngpus>(p);
  const auto tiling      = tiles::rows(rows, packs, ngpus);
  const V* res_in        = reinterpret_cast<const V*>(residual);
  const auto* wv         = reinterpret_cast<const vec<W, NL>*>(weight);
  V* res_out             = reinterpret_cast<V*>(residual_out);
  V* o                   = reinterpret_cast<V*>(out);
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  const auto in          = p2p::push::slot<kBits>(w, tiling, p2p::To::owners);
  const auto out_slot    = p2p::push::slot<kBits>(w, tiling, p2p::To::owners, in);
  const auto res_slot    = p2p::push::slot<16>(w, tiling, p2p::To::owners, out_slot);

  p2p::push::scatter(w, in, tiling);

  p2p::peer_barrier(w);

  for (int row = tiling.first(p.rank); row < tiling.end(p.rank); row = tiling.next(row)) {
    V sum[kMaxRowPacks];
    p2p::push::reduce(w, in, tiling, row, sum);
    V normed[kMaxRowPacks] = {}, res[kMaxRowPacks] = {};
    fusion::row<T, W, kAdd>(
        sum, res_in, wv, row, packs, inv_hidden, eps,
        [&](int k, int, const V& v) { res[k] = v; },
        [&](int k, int, const V& v) { normed[k] = v; });
    p2p::push::share(w, out_slot, tiling, row, normed);
    if constexpr (kAdd) p2p::push::share(w, res_slot, tiling, row, res);
  }

  p2p::peer_barrier(w);

  p2p::push::gather(w, out_slot, tiling, [&](int row, int k, const V& v) {
    store_global(o + tiling.pos(row, k), v);
  });
  if constexpr (kAdd)
    p2p::push::gather(w, res_slot, tiling, [&](int row, int k, const V& v) {
      store_global(res_out + tiling.pos(row, k), v);
    });
}

// THE KERNELS, one per op, both the body above.
template <typename T, typename W, int ngpus, int kBits>
__global__ void __launch_bounds__(kMaxThreads, 1) allreduce_two_shot_push_rms_norm(
    p2p::Peers p, T* __restrict__ out, const W* __restrict__ weight, float eps, int rows,
    int packs) {
  two_shot_push_add_rms_norm_body<T, W, ngpus, kBits, false>(p, out, nullptr, nullptr,
                                                             weight, eps, rows, packs);
}

template <typename T, typename W, int ngpus, int kBits>
__global__ void __launch_bounds__(kMaxThreads, 1) allreduce_two_shot_push_add_rms_norm(
    p2p::Peers p, T* __restrict__ out, T* __restrict__ residual_out,
    const T* __restrict__ residual, const W* __restrict__ weight, float eps, int rows,
    int packs) {
  two_shot_push_add_rms_norm_body<T, W, ngpus, kBits, true>(p, out, residual_out, residual,
                                                            weight, eps, rows, packs);
}

}  // namespace hip_comms
