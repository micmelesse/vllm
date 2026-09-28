// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot all-reduce fused with Kimi-K3's attention residual (AttnRes) and its RMSNorm.

#pragma once

#include "fusions/add_attn_res_rms_norm.cuh"
#include "p2p/pull.cuh"
#include "utils.cuh"

namespace hip_comms {

// A BLOCK OWNS A ROW, striding over rows, as the fused norm does: every rank reduces every
// row, so there is nothing to gather. `blocks` is [rows, num_sources, hidden] with row and
// source strides in elements; `write_idx` < 0 writes no block.
template <typename T, int ngpus, bool kPrefix>
__global__ void __launch_bounds__(kMaxThreads, 1)
    allreduce_one_shot_pull_add_attn_res_rms_norm(
    p2p::Peers p, T* __restrict__ prefix, T* __restrict__ blocks, int64_t block_stride_m,
    int64_t block_stride_r, const T* __restrict__ norm_w, const T* __restrict__ qk_w,
    const T* __restrict__ out_norm_w, T* __restrict__ out, int num_blocks, int write_idx,
    float eps, float out_eps, int rows, int packs) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  using core       = p2p::Core<T, ngpus>;
  using pull       = p2p::Pull<T, ngpus>;
  namespace fusion = fusions::add_attn_res_rms_norm;
  core::start(p);
  const auto in          = core::inputs(p);
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  V* pre                 = reinterpret_cast<V*>(prefix);
  V* o                   = reinterpret_cast<V*>(out);
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const T* row_blocks = blocks + row * block_stride_m;
    V* dst              = write_idx >= 0 ? reinterpret_cast<V*>(const_cast<T*>(row_blocks) +
                                                   write_idx * block_stride_r)
                            : nullptr;
    V sum[kMaxRowPacks];
    pull::sum_row(p, in, row * packs, packs, sum);
    fusion::row<T, kPrefix>(
        sum, pre, row_blocks, block_stride_r, reinterpret_cast<const V*>(norm_w),
        reinterpret_cast<const V*>(qk_w), reinterpret_cast<const V*>(out_norm_w),
        num_blocks, row, packs, inv_hidden, eps, out_eps,
        [&](int, int i, const V& v) {
          pre[row * packs + i] = v;
          if (dst != nullptr) dst[i] = v;
        },
        [&](int, int i, const V& v) { o[row * packs + i] = v; });
  }
  core::close(p);
}

}  // namespace hip_comms
