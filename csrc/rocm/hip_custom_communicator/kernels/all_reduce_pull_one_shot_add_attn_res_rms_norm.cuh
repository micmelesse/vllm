// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce fused with Kimi-K3's attention residual (AttnRes) and its
// RMSNorm.

#pragma once

#include "../common/interface.cuh"
#include "shared/attn_res.cuh"

namespace hip_custom_communicator {

// A block owns a row, as the fused norm does: every rank reduces every row, so there is
// nothing to gather. `blocks` is [m, num_sources, n] at its row, source and column
// strides in elements; `write_idx` < 0 writes no block.
template <typename DTYPE, int WORLD, bool HAS_PREFIX, int TILE_N, int TILE_K,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot_add_attn_res_rms_norm(const DTYPE* const* __restrict__ input_ptrs,
                                                   int64_t input_stride_m,
                                                   int64_t input_stride_n,
                                                   int input_size_m,
                                                   int input_size_n,
                                                   const DTYPE* __restrict__ prefix_ptr,
                                                   int64_t prefix_stride_m,
                                                   int64_t prefix_stride_n,
                                                   int prefix_size_m,
                                                   int prefix_size_n,
                                                   const DTYPE* __restrict__ blocks_ptr,
                                                   int64_t blocks_stride_m,
                                                   int64_t blocks_stride_r,
                                                   int64_t blocks_stride_n,
                                                   int blocks_size_m,
                                                   int blocks_size_r,
                                                   int blocks_size_n,
                                                   const DTYPE* __restrict__ norm_weight_ptr,
                                                   int64_t norm_weight_stride_n,
                                                   int norm_weight_size_n,
                                                   const DTYPE* __restrict__ qk_weight_ptr,
                                                   int64_t qk_weight_stride_n,
                                                   int qk_weight_size_n,
                                                   const DTYPE* __restrict__ output_norm_weight_ptr,
                                                   int64_t output_norm_weight_stride_n,
                                                   int output_norm_weight_size_n,
                                                   DTYPE* __restrict__ prefix_output_ptr,
                                                   int64_t prefix_output_stride_m,
                                                   int64_t prefix_output_stride_n,
                                                   int prefix_output_size_m,
                                                   int prefix_output_size_n,
                                                   DTYPE* __restrict__ output_ptr,
                                                   int64_t output_stride_m,
                                                   int64_t output_stride_n,
                                                   int output_size_m,
                                                   int output_size_n,
                                                   DTYPE* __restrict__ block_output_ptr,
                                                   int64_t block_output_stride_m,
                                                   int64_t block_output_stride_n,
                                                   int block_output_size_m,
                                                   int block_output_size_n,
                                                   uint32_t* const* __restrict__ signals_ptrs,
                                                   int64_t signals_stride_parity,
                                                   int64_t signals_stride_block,
                                                   int64_t signals_stride_writer,
                                                   int signals_size_parity,
                                                   int signals_size_block,
                                                   int signals_size_writer,
                                                   Fault* fault_ptr,
                                                   int rank,
                                                   uint64_t timeout_ticks,
                                                   int num_blocks,
                                                   float eps,
                                                   float output_eps) {
  Sync<WORLD> sync{signals_ptrs, signals_stride_parity, signals_stride_block,
                   signals_stride_writer, fault_ptr, rank, timeout_ticks};
  using Row              = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto input = rank_ptrs<const DTYPE, WORLD>(input_ptrs, input_stride_m, input_stride_n);
  const auto prefix     = local_ptr(prefix_ptr, prefix_stride_m, prefix_stride_n, rank);
  const auto prefix_output = local_ptr(prefix_output_ptr, prefix_output_stride_m,
                                         prefix_output_stride_n, rank);
  const auto block_output  = local_ptr(block_output_ptr, block_output_stride_m,
                                         block_output_stride_n, rank);
  const auto norm_weight     = local_ptr(norm_weight_ptr, 0, norm_weight_stride_n, rank);
  const auto qk_weight       = local_ptr(qk_weight_ptr, 0, qk_weight_stride_n, rank);
  const auto output_norm_weight = local_ptr(output_norm_weight_ptr, 0, output_norm_weight_stride_n, rank);
  const auto output        = local_ptr(output_ptr, output_stride_m, output_stride_n, rank);
  barrier<Until::launched>(sync);

  // 2. Each of this block's tiles (one row: TILE_M = 1): read it from every rank in rank order,
  //    sum, AttnRes.
  for (int row = blockIdx.x; row < input_size_m; row += gridDim.x) {
    Row peers[WORLD];
#pragma unroll
    for (int r = 0; r < WORLD; ++r) peers[r] = Row{input_size_m, input_size_n, row, 0};
    tile_load(peers, input);
    block_attn_res_tile<HAS_PREFIX, TILE_K>(
        peers_reduce(peers), prefix, blocks_ptr, blocks_stride_m, blocks_stride_r, norm_weight, qk_weight,
        output_norm_weight, prefix_output, output, block_output, num_blocks, eps, output_eps,
        inv_hidden);
  }

  // 3. No rank may overwrite its input until every peer has read it.
  barrier<Until::read>(sync);
}

}  // namespace hip_custom_communicator
