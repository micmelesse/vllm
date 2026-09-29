// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce fused with Kimi-K3's attention residual (AttnRes) and its
// RMSNorm.

#pragma once

#include "fusions/add_attn_res_rms_norm.cuh"
#include "p2p/p2p.cuh"
#include "utils.cuh"

namespace hip_comms {

// Each rank owns whole rows: it reduces them, updates the prefix, runs AttnRes and shares
// the out and prefix rows in its scratch; after the barrier every rank gathers every
// owner's rows into `out`, `prefix` and the written block. A row's replicated `prefix`
// and `blocks` are read and then overwritten by the one block that takes it, so the order
// is the block's own. The input is read only before the barrier, so no close.
template <typename T, int ngpus, bool kPrefix>
__global__ void __launch_bounds__(kMaxThreads, 1)
    all_reduce_pull_two_shot_add_attn_res_rms_norm(
        p2p::Peers p, T* __restrict__ prefix, T* __restrict__ blocks,
        int64_t block_stride_m, int64_t block_stride_r, const T* __restrict__ norm_w,
        const T* __restrict__ qk_w, const T* __restrict__ out_norm_w, T* __restrict__ out,
        int num_blocks, int write_idx, float eps, float out_eps, int rows, int packs) {
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  namespace fusion       = fusions::add_attn_res_rms_norm;
  const auto w           = p2p::start<T, ngpus>(p);
  const auto tiling      = tiles::rows(rows, packs, ngpus);
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  V* pre                 = reinterpret_cast<V*>(prefix);
  V* o                   = reinterpret_cast<V*>(out);
  // The block row `row` writes, or none.
  auto written = [&](int row) -> V* {
    return write_idx < 0 ? nullptr
                         : reinterpret_cast<V*>(blocks + row * block_stride_m +
                                                write_idx * block_stride_r);
  };
  const auto out_slot    = p2p::pull::slot(w, tiling);
  const auto pre_slot    = p2p::pull::slot(w, tiling, out_slot);

  for (int row = tiling.first(p.rank); row < tiling.end(p.rank); row = tiling.next(row)) {
    V sum[kMaxRowPacks];
    p2p::pull::reduce(w, tiling, row, sum);
    V mixed[kMaxRowPacks] = {}, shared_pre[kMaxRowPacks] = {};
    fusion::row<T, kPrefix>(
        sum, pre, blocks + row * block_stride_m, block_stride_r,
        reinterpret_cast<const V*>(norm_w), reinterpret_cast<const V*>(qk_w),
        reinterpret_cast<const V*>(out_norm_w), num_blocks, row, packs, inv_hidden, eps,
        out_eps, [&](int k, int, const V& v) { shared_pre[k] = v; },
        [&](int k, int, const V& v) { mixed[k] = v; });
    p2p::pull::share(w, out_slot, tiling, row, mixed);
    p2p::pull::share(w, pre_slot, tiling, row, shared_pre);
  }

  p2p::peer_barrier(w);

  p2p::pull::gather(w, out_slot, tiling, [&](int row, int k, const V& v) {
    store_global(o + tiling.pos(row, k), v);
  });
  p2p::pull::gather(w, pre_slot, tiling, [&](int row, int k, const V& v) {
    store_global(pre + tiling.pos(row, k), v);
    if (V* dst = written(row)) store_global(dst + tiling.pack(k), v);
  });
}

}  // namespace hip_comms
