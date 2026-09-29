// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce fused with Kimi-K3's attention residual (AttnRes) and its
// RMSNorm.

#pragma once

#include "fusions/add_attn_res_rms_norm.cuh"
#include "p2p/p2p.cuh"
#include "common/memory.cuh"
#include "common/reduce.cuh"

namespace hip_comms {

// A block owns a row, as the fused norm does: every rank reduces every row, so there is
// nothing to gather. `blocks` is [rows, num_sources, hidden] with row and source strides
// in elements; `write_idx` < 0 writes no block.
template <typename T, int ngpus, bool kPrefix>
__global__ void __launch_bounds__(kMaxThreads, 1)
    all_reduce_pull_one_shot_add_attn_res_rms_norm(
        p2p::Peers p, T* __restrict__ prefix, T* __restrict__ blocks,
        int64_t block_stride_m, int64_t block_stride_r, const T* __restrict__ norm_w,
        const T* __restrict__ qk_w, const T* __restrict__ out_norm_w, T* __restrict__ out,
        int num_blocks, int write_idx, float eps, float out_eps, int rows, int packs) {
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  namespace fusion       = fusions::add_attn_res_rms_norm;
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  V* pre                 = reinterpret_cast<V*>(prefix);
  V* o                   = reinterpret_cast<V*>(out);
  // The block row `row` writes, or none.
  auto written = [&](int row) -> V* {
    return write_idx < 0 ? nullptr
                         : reinterpret_cast<V*>(blocks + row * block_stride_m +
                                                write_idx * block_stride_r);
  };

  // 1. Wait until every peer has launched, so its input is ready.
  p2p::simple::start_sync<ngpus>(p);

  // 2. Each of this block's rows: read it from every rank in rank order, sum, AttnRes.
  const V* in[ngpus];
#pragma unroll
  for (int r = 0; r < ngpus; ++r) in[r] = p2p::simple::input<T>(p, r);
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    V sum[kMaxRowPacks];
    sum_row<T, ngpus>(in, row, packs, sum);
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

  // 3. No rank may overwrite its input until every peer has read it.
  p2p::simple::end_sync<ngpus, true>(p);
}

}  // namespace hip_comms
