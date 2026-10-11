// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce fused with Kimi-K3's attention residual (AttnRes) and its
// RMSNorm.

#pragma once

#include "../common/interface.cuh"
#include "shared/attn_res.cuh"

namespace hip_comms {

// A block owns a row, as the fused norm does: every rank reduces every row, so there is
// nothing to gather. `blocks` is [m, num_sources, n] at its row, source and column
// strides in elements; `write_idx` < 0 writes no block.
template <typename DTYPE, int WORLD, bool HAS_PREFIX, int TILE_N, int TILE_K,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot_add_attn_res_rms_norm(
        const DTYPE* const* __restrict__ inp_ptrs, int64_t inp_stride_m, int64_t inp_stride_n,
        Signal* const* __restrict__ signal_ptrs, Signal* self_signal_ptr, int rank, uint64_t timeout_ticks,
        DTYPE* __restrict__ prefix_ptr, int64_t prefix_stride_m, int64_t prefix_stride_n,
        DTYPE* __restrict__ blocks_ptr, int64_t blocks_stride_m, int64_t blocks_stride_r,
        int64_t blocks_stride_n, const DTYPE* __restrict__ norm_w_ptr, int64_t norm_w_stride_n,
        const DTYPE* __restrict__ qk_w_ptr, int64_t qk_w_stride_n,
        const DTYPE* __restrict__ out_norm_w_ptr, int64_t out_norm_w_stride_n,
        DTYPE* __restrict__ out_ptr, int64_t out_stride_m, int64_t out_stride_n, int num_blocks,
        int write_idx, float eps, float out_eps, int m, int n) {
  Sync<WORLD> sync{signal_ptrs, self_signal_ptr, rank, timeout_ticks};
  using Row              = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  const float inv_hidden = 1.0f / static_cast<float>(n);

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto inp = rank_ptrs<const DTYPE, WORLD>(inp_ptrs, inp_stride_m, inp_stride_n);
  const auto prefix     = local_ptr(prefix_ptr, prefix_stride_m, prefix_stride_n, rank);
  const auto norm_w     = local_ptr(norm_w_ptr, 0, norm_w_stride_n, rank);
  const auto qk_w       = local_ptr(qk_w_ptr, 0, qk_w_stride_n, rank);
  const auto out_norm_w = local_ptr(out_norm_w_ptr, 0, out_norm_w_stride_n, rank);
  const auto out        = local_ptr(out_ptr, out_stride_m, out_stride_n, rank);
  barrier<Group::peers, Until::launched>(sync);

  // 2. Each of this block's tiles (one row: TILE_M = 1): read it from every rank in rank order,
  //    sum, AttnRes.
  for (int row = blockIdx.x; row < m; row += gridDim.x) {
    Row peers[WORLD];
#pragma unroll
    for (int r = 0; r < WORLD; ++r) peers[r] = Row{m, n, row, 0};
    tile_load(peers, inp);
    block_attn_res_tile<HAS_PREFIX, TILE_K>(peers_reduce(peers), prefix, blocks_ptr,
                                         blocks_stride_m, blocks_stride_r, write_idx, norm_w, qk_w, out_norm_w, out,
                                         num_blocks, eps, out_eps, inv_hidden);
  }

  // 3. No rank may overwrite its input until every peer has read it.
  barrier<Group::peers, Until::read>(sync);
  sync.finish();
}

}  // namespace hip_comms
