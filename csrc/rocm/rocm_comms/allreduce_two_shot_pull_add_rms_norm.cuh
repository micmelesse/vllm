// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce then RMSNorm (`allreduce_two_shot_pull_rms_norm`), and
// all-reduce then add then RMSNorm (`allreduce_two_shot_pull_add_rms_norm`): one body, a
// kernel per op, so a trace names the op that ran.

#pragma once

#include "fusions/add_rms_norm.cuh"
#include "p2p/p2p.cuh"
#include "utils.cuh"

namespace hip_comms {

// THE SLICE IS ROWS: each rank owns whole rows, so it finishes the norm alone. It
// reduces its rows, (kAdd: adds the replicated residual,) norms them and shares the out
// row and (kAdd) the residual row in its scratch; after the barrier every rank gathers
// every owner's rows. Every output element is computed by one rank, so every rank holds
// the same bytes. The input is read only before the barrier, so no close.
template <typename T, typename W, int ngpus, bool kAdd>
DINLINE void two_shot_pull_add_rms_norm_body(p2p::Peers p, T* __restrict__ out,
                                             T* __restrict__ residual_out,
                                             const T* __restrict__ residual,
                                             const W* __restrict__ weight, float eps,
                                             int rows, int packs) {
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  namespace fusion       = fusions::add_rms_norm;
  const auto w           = p2p::start<T, ngpus>(p);
  const auto tiling      = tiles::rows_of(rows, packs);
  const V* res_in        = reinterpret_cast<const V*>(residual);
  const auto* wv         = reinterpret_cast<const vec<W, NL>*>(weight);
  V* res_out             = reinterpret_cast<V*>(residual_out);
  V* o                   = reinterpret_cast<V*>(out);
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  const auto mine        = tiles::owned(tiling, p.rank, ngpus);
  const auto out_slot    = p2p::pull::slot(w, tiling);
  const auto res_slot    = p2p::pull::slot(w, tiling, out_slot);

  for (int row = mine.begin + blockIdx.x; row < mine.end; row += gridDim.x) {
    V sum[kMaxRowPacks];
    p2p::pull::reduce(w, tiling, row, sum);
    V normed[kMaxRowPacks] = {}, res[kMaxRowPacks] = {};
    fusion::row<T, W, kAdd>(
        sum, res_in, wv, row, packs, inv_hidden, eps,
        [&](int k, int, const V& v) { res[k] = v; },
        [&](int k, int, const V& v) { normed[k] = v; });
    p2p::pull::share(w, out_slot, tiling, row, normed);
    if constexpr (kAdd) p2p::pull::share(w, res_slot, tiling, row, res);
  }

  p2p::peer_barrier(w);

  p2p::pull::gather(w, out_slot, tiling, [&](int row, int i, const V& v) {
    store_global(o + row * packs + i, v);
  });
  if constexpr (kAdd)
    p2p::pull::gather(w, res_slot, tiling, [&](int row, int i, const V& v) {
      store_global(res_out + row * packs + i, v);
    });
}

// THE KERNELS, one per op, both the body above.
template <typename T, typename W, int ngpus>
__global__ void __launch_bounds__(kMaxThreads, 1) allreduce_two_shot_pull_rms_norm(
    p2p::Peers p, T* __restrict__ out, const W* __restrict__ weight, float eps, int rows,
    int packs) {
  two_shot_pull_add_rms_norm_body<T, W, ngpus, false>(p, out, nullptr, nullptr,
                                                      weight, eps, rows, packs);
}

template <typename T, typename W, int ngpus>
__global__ void __launch_bounds__(kMaxThreads, 1) allreduce_two_shot_pull_add_rms_norm(
    p2p::Peers p, T* __restrict__ out, T* __restrict__ residual_out,
    const T* __restrict__ residual, const W* __restrict__ weight, float eps, int rows,
    int packs) {
  two_shot_pull_add_rms_norm_body<T, W, ngpus, true>(p, out, residual_out, residual,
                                                     weight, eps, rows, packs);
}

}  // namespace hip_comms
