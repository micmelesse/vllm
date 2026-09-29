// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot push all-reduce fused with Kimi-K3's attention residual (AttnRes) and its
// RMSNorm.

#pragma once

#include "fusions/add_attn_res_rms_norm.cuh"
#include "p2p/p2p.cuh"
#include "utils.cuh"

namespace hip_comms {

// Every rank's rows, encoded by kBits' codec, into every rank's slot; one barrier; each
// block reduces its rows out of its own slot and runs AttnRes on them as the pull kernel
// does.
template <typename T, int ngpus, int kBits, bool kPrefix>
__global__ void __launch_bounds__(kMaxThreads, 1)
    all_reduce_push_one_shot_add_attn_res_rms_norm(
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
  const auto slot        = p2p::push::slot<kBits>(w, tiling, p2p::To::all);

  p2p::push::scatter(w, slot, tiling);

  p2p::peer_barrier(w);

  for (int row = tiling.first(); row < tiling.end(); row = tiling.next(row)) {
    V sum[kMaxRowPacks];
    p2p::push::reduce(w, slot, tiling, row, sum);
    fusion::row<T, kPrefix>(
        sum, pre, blocks + row * block_stride_m, block_stride_r,
        reinterpret_cast<const V*>(norm_w), reinterpret_cast<const V*>(qk_w),
        reinterpret_cast<const V*>(out_norm_w), num_blocks, row, packs, inv_hidden, eps,
        out_eps, [&](int, int i, const V& v) {
          pre[row * packs + i] = v;
          if (V* dst = written(row)) dst[i] = v;
        },
        [&](int, int i, const V& v) { o[row * packs + i] = v; });
  }
}

}  // namespace hip_comms
