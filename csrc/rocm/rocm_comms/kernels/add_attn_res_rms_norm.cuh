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
// ITS OCCUPANCY STATED, NOT LEFT TO THE REGISTER ALLOCATOR: waves a SIMD for each build (HIP's
// launch bounds' second argument is amdgpu_waves_per_eu; a separate attribute lost to its 1), Triton's
// at 256 threads (3 at 7168, 6 at 3584), where they already sit at 512. Left to it, 7168 x 256
// swung 166 -> 221 VGPRs with changes unrelated to it, 3 waves to 2 (2026-10-04T03-19-22Z).
template <int TILE_N, int THREADS_PER_BLOCK>
constexpr int attn_res_waves() {
  return THREADS_PER_BLOCK == 256 ? (TILE_N > 4096 ? 3 : 6) : (TILE_N > 4096 ? 5 : 8);
}

template <typename DTYPE, int TILE_N, int TILE_K, int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, (attn_res_waves<TILE_N, THREADS_PER_BLOCK>()))
    add_attn_res_rms_norm(DTYPE* __restrict__ prefix, const DTYPE* __restrict__ delta,
                          DTYPE* __restrict__ blocks, int64_t block_stride_m, int64_t block_stride_r,
                          const DTYPE* __restrict__ norm_w, const DTYPE* __restrict__ qk_w,
                          const DTYPE* __restrict__ out_norm_w, DTYPE* __restrict__ out, int num_blocks,
                          int write_idx, float eps, float out_eps, int rows, int packs) {
  using Row              = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK>;
  const int cols         = packs * traits<DTYPE>::N;  // the row, in elements
  const float inv_hidden = 1.0f / static_cast<float>(cols);
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    Row sum{rows, cols, row, 0};
    tile_load(sum, delta, cols);
    block_attn_res_tile<true, TILE_K>(sum, prefix, blocks, block_stride_m, block_stride_r,
                                      write_idx, norm_w, qk_w, out_norm_w, out, num_blocks, eps,
                                      out_eps, inv_hidden);
  }
}

}  // namespace hip_comms
