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
// kernels' AttnRes.
template <typename DTYPE, int TILE_N, int TILE_K, int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)
    add_attn_res_rms_norm(DTYPE* __restrict__ prefix, const DTYPE* __restrict__ delta,
                          DTYPE* __restrict__ blocks, int64_t block_stride_m, int64_t block_stride_r,
                          const DTYPE* __restrict__ norm_w, const DTYPE* __restrict__ qk_w,
                          const DTYPE* __restrict__ out_norm_w, DTYPE* __restrict__ out, int num_blocks,
                          int write_idx, float eps, float out_eps, int rows, int packs) {
  using Row              = Tile<DTYPE, 1, TILE_N, THREADS_PER_BLOCK>;
  const int cols         = packs * traits<DTYPE>::N;  // the row, in elements
  const float inv_hidden = 1.0f / static_cast<float>(cols);
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    Row sum{rows, cols, row, 0};
    thread_load(sum, delta, cols);
    block_attn_res_tile<true, TILE_K>(sum, prefix, blocks, block_stride_m, block_stride_r,
                                      write_idx, norm_w, qk_w, out_norm_w, out, num_blocks, eps,
                                      out_eps, inv_hidden);
  }
}

}  // namespace hip_comms
