// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Kimi-K3's attention residual (AttnRes) and its RMSNorm on a local delta, no all-reduce:
// Triton's `_attn_res_kernel` with HAS_DELTA, and the AttnRes half of the fused kernels alone.

#pragma once

#include "../common/interface.cuh"
#include "shared/attn_res.cuh"

namespace hip_comms {

// A ROW A TILE, the grid striding over rows: each row's delta read from local memory, then the
// tile every AttnRes kernel computes (shared/attn_res.cuh), so its instructions are the fused
// kernels' AttnRes.
template <typename DTYPE, int TILE_N, int TILE_K, int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    add_attn_res_rms_norm(
        const DTYPE* __restrict__ delta_ptr, int64_t delta_stride_m, int64_t delta_stride_n,
        DTYPE* __restrict__ prefix_ptr, int64_t prefix_stride_m, int64_t prefix_stride_n,
        DTYPE* __restrict__ blocks_ptr, int64_t blocks_stride_m, int64_t blocks_stride_r,
        int64_t blocks_stride_n, const DTYPE* __restrict__ norm_w_ptr, int64_t norm_w_stride_n,
        const DTYPE* __restrict__ qk_w_ptr, int64_t qk_w_stride_n,
        const DTYPE* __restrict__ out_norm_w_ptr, int64_t out_norm_w_stride_n,
        DTYPE* __restrict__ out_ptr, int64_t out_stride_m, int64_t out_stride_n, int num_blocks,
        int write_idx, float eps, float out_eps, int m, int n) {
  const int rank = 0;  // no peers: the one rank
  const auto prefix     = local_ptr(prefix_ptr, prefix_stride_m, prefix_stride_n, rank);
  const auto norm_w     = local_ptr(norm_w_ptr, 0, norm_w_stride_n, rank);
  const auto qk_w       = local_ptr(qk_w_ptr, 0, qk_w_stride_n, rank);
  const auto out_norm_w = local_ptr(out_norm_w_ptr, 0, out_norm_w_stride_n, rank);
  const auto out        = local_ptr(out_ptr, out_stride_m, out_stride_n, rank);
  const auto delta      = local_ptr(delta_ptr, delta_stride_m, delta_stride_n, rank);
  using Row              = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  const float inv_hidden = 1.0f / static_cast<float>(n);
  for (int row = blockIdx.x; row < m; row += gridDim.x) {
    Row sum{m, n, row, 0};
    tile_load(sum, delta);
    block_attn_res_tile<true, TILE_K>(sum, prefix, blocks_ptr, blocks_stride_m, blocks_stride_r,
                                      write_idx, norm_w, qk_w, out_norm_w, out, num_blocks, eps,
                                      out_eps, inv_hidden);
  }
}

}  // namespace hip_comms
