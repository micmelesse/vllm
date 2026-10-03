// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Kimi-K3's attention residual (AttnRes) and its RMSNorm on a local delta, no all-reduce:
// Triton's `_attn_res_kernel` with HAS_DELTA, and the AttnRes half of the fused kernels alone.

#pragma once

#include "../common/common.cuh"
#include "shared/attn_res.cuh"

namespace hip_comms {

// A ROW A TILE, the grid striding over rows: each row's delta read from local memory, then the
// tile every AttnRes kernel computes (shared/attn_res.cuh), so its instructions are the fused
// kernels' AttnRes. `blocks` is [rows, num_sources, hidden] with row and source strides in
// elements; `write_idx` < 0 writes no block.
template <typename T, int TILE_N, int TILE_K, int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)
    add_attn_res_rms_norm(T* __restrict__ prefix, const T* __restrict__ delta,
                          T* __restrict__ blocks, int64_t block_stride_m, int64_t block_stride_r,
                          const T* __restrict__ norm_w, const T* __restrict__ qk_w,
                          const T* __restrict__ out_norm_w, T* __restrict__ out, int num_blocks,
                          int write_idx, float eps, float out_eps, int rows, int packs) {
  constexpr int kRowPacks = packs_per_thread<T, TILE_N, THREADS_PER_BLOCK>();
  using V                 = typename traits<T>::V;
  constexpr int NL        = traits<T>::N;
  const float inv_hidden  = 1.0f / static_cast<float>(packs * NL);
  const int cols          = packs * NL;
  const auto thread_cols  = thread_offs<T, THREADS_PER_BLOCK>(Tile<1, TILE_N>{rows, cols, 0, 0});
  const V* d              = reinterpret_cast<const V*>(delta);
  auto written            = [&](int row) -> V* {
    return write_idx < 0 ? nullptr
                                    : reinterpret_cast<V*>(blocks + row * block_stride_m +
                                                           write_idx * block_stride_r);
  };
  for (int offs_m = blockIdx.x; offs_m < rows; offs_m += gridDim.x) {
    const Tile<1, TILE_N> tile{rows, cols, offs_m, 0};
    const int64_t base = int64_t{offs_m} * packs;
    V sum[1][kRowPacks];
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) sum[0][k] = d[base + thread_cols.offs_n[k]];
    block_attn_res_tile<T, true, TILE_K>(sum, tile, thread_cols, reinterpret_cast<V*>(prefix),
                                         written, blocks, block_stride_m, block_stride_r, norm_w,
                                         qk_w, out_norm_w, reinterpret_cast<V*>(out), num_blocks,
                                         eps, out_eps, inv_hidden);
  }
}

}  // namespace hip_comms
