// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot push all-reduce fused with Kimi-K3's attention residual (AttnRes) and its
// RMSNorm.

#pragma once

#include "fusions/add_attn_res_rms_norm.cuh"
#include "p2p/p2p.cuh"
#include "common/memory.cuh"
#include "hardware.cuh"

namespace hip_comms {

// Each rank owns whole rows, as the pull kernel does, and every transfer is a store into
// a peer's slot: each input row, encoded, to its owner; barrier; the owner reduces, runs
// AttnRes and shares the out row (encoded) and the prefix row with every rank; barrier;
// every rank gathers. THE PREFIX IS NEVER QUANTIZED (16 bits whatever kBits): it is the
// running residual.
template <typename T, int ngpus, int kBits, bool kPrefix>
__global__ void __launch_bounds__(kMaxThreads, 1)
    all_reduce_push_two_shot_add_attn_res_rms_norm(
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
  const auto in          = p2p::push::slot<kBits>(w, tiling, p2p::To::owners);
  const auto out_slot    = p2p::push::slot<kBits>(w, tiling, p2p::To::owners, in);
  const auto pre_slot    = p2p::push::slot<16>(w, tiling, p2p::To::owners, out_slot);

  p2p::push::scatter(w, in, tiling);

  p2p::peer_barrier(w);

  for (int row = tiling.first(p.rank); row < tiling.end(p.rank); row = tiling.next(row)) {
    V sum[p2p::kPushGroupPacks];
    p2p::push::reduce(w, in, tiling, row, sum);
    V mixed[p2p::kPushGroupPacks] = {}, shared_pre[p2p::kPushGroupPacks] = {};
    fusion::row<T, kPrefix>(
        sum, pre, blocks + row * block_stride_m, block_stride_r,
        reinterpret_cast<const V*>(norm_w), reinterpret_cast<const V*>(qk_w),
        reinterpret_cast<const V*>(out_norm_w), num_blocks, row, packs, inv_hidden, eps,
        out_eps, [&](int k, int, const V& v) { shared_pre[k] = v; },
        [&](int k, int, const V& v) { mixed[k] = v; });
    p2p::push::share(w, out_slot, tiling, row, mixed);
    p2p::push::share(w, pre_slot, tiling, row, shared_pre);
  }

  p2p::peer_barrier(w);

  p2p::push::gather(w, out_slot, tiling, [&](int row, int k, const V& v) {
    store_global(o + tiling.pos(row, k), v);
  });
  p2p::push::gather(w, pre_slot, tiling, [&](int row, int k, const V& v) {
    store_global(pre + tiling.pos(row, k), v);
    if (V* dst = written(row)) store_global(dst + tiling.pack(k), v);
  });
}

}  // namespace hip_comms
