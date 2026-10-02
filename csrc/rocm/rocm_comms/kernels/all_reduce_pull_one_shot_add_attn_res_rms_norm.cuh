// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce fused with Kimi-K3's attention residual (AttnRes) and its
// RMSNorm.

#pragma once

#include "../p2p/p2p.cuh"
#include "../common/common.cuh"
#include "shared/attn_res.cuh"

namespace hip_comms {

// A block owns a row, as the fused norm does: every rank reduces every row, so there is
// nothing to gather. `blocks` is [rows, num_sources, hidden] with row and source strides
// in elements; `write_idx` < 0 writes no block.
template <typename T, int ngpus, bool kPrefix, int kRowPacks>
__global__ void __launch_bounds__(kBuild.kernels.max_threads, 1)
    all_reduce_pull_one_shot_add_attn_res_rms_norm(
        p2p::DevComm p, T* __restrict__ prefix, T* __restrict__ blocks, int64_t block_stride_m,
        int64_t block_stride_r, const T* __restrict__ norm_w, const T* __restrict__ qk_w,
        const T* __restrict__ out_norm_w, T* __restrict__ out, int num_blocks, int write_idx,
        float eps, float out_eps, int rows, int packs) {
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  V* pre                 = reinterpret_cast<V*>(prefix);
  V* o                   = reinterpret_cast<V*>(out);
  const auto f           = fragment<kRowPacks>(packs);
  // The block row `row` writes, or none.
  auto written = [&](int row) -> V* {
    return write_idx < 0 ? nullptr
                         : reinterpret_cast<V*>(blocks + row * block_stride_m +
                                                write_idx * block_stride_r);
  };

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto inputs = p2p::inputs<T, ngpus>(p);
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);
  const auto read = [&](int r, int64_t i) { return p2p::read_input(inputs[r], i); };

  // 2. Each of this block's rows: read it from every rank in rank order, sum, AttnRes.
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t base = int64_t{row} * packs;
    V sum[kRowPacks];
    peers_reduce(peers_load<T, ngpus>(read, row, packs, f), sum);
    block_attn_res_row<T, kPrefix, kRowPacks>(
        sum, base, f, pre, written(row), blocks + int64_t{row} * block_stride_m, block_stride_r,
        norm_w, qk_w, out_norm_w, o, num_blocks, eps, out_eps, inv_hidden);
  }

  // 3. No rank may overwrite its input until every peer has read it.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::read>(p);
}

}  // namespace hip_comms
