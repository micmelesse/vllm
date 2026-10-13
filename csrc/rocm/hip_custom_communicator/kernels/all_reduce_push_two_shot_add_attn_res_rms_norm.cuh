// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot all-reduce (the reduce-scatter pulled, the all-gather pushed) fused with Kimi-K3's
// attention residual (AttnRes) and its RMSNorm: the push norm's two phases, with the one-shot
// AttnRes's row in place of the norm.

#pragma once

#include "../common/interface.cuh"
#include "shared/attn_res.cuh"

namespace hip_custom_communicator {

// THE SLICE IS COLUMNS, as the push norm's: each rank owns packs [rank * S, rank * S + S) of every
// row, sums them over the ranks and pushes the sums into every rank's scratch, so after the sync
// each rank holds the whole sum locally and computes AttnRes on its rows itself: what crosses the
// links is the sum, once, and AttnRes's two outputs (the prefix and out) are never gathered. ROW q
// BELONGS TO BLOCK q % gridDim.x IN BOTH PHASES: after the sync a block may read only what the
// same block on a peer wrote. `blocks` is [m, num_sources, n] with row and source strides
// in elements; `write_idx` < 0 writes no block.
template <typename DTYPE, int WORLD, bool HAS_PREFIX, int TILE_N, int TILE_K,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_push_two_shot_add_attn_res_rms_norm(const DTYPE* const* __restrict__ input_ptrs,
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
                                                   DTYPE* const* __restrict__ scratch_ptrs,
                                                   int64_t scratch_stride_m,
                                                   int64_t scratch_stride_n,
                                                   int scratch_size_m,
                                                   int scratch_size_n,
                                                   uint32_t* const* __restrict__ signals_ptrs,
                                                   int64_t signals_stride_parity,
                                                   int64_t signals_stride_block,
                                                   int64_t signals_stride_writer,
                                                   int signals_size_parity,
                                                   int signals_size_block,
                                                   int signals_size_writer,
                                                   uint64_t* __restrict__ status_ptr,
                                                   int64_t status_stride_field,
                                                   int status_size_field,
                                                   int rank,
                                                   uint64_t timeout_ticks,
                                                   int num_blocks,
                                                   float eps,
                                                   float output_eps) {
  Sync<WORLD> sync{signals_ptrs, signals_stride_parity, signals_stride_block,
                   signals_stride_writer, status_ptr, status_stride_field, rank,
                   timeout_ticks};
  constexpr int NL       = traits<DTYPE>::N;
  using Row              = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  // A REDUCE-SCATTER TILE: a row of threads as wide as a rank's slice of a TILE_N row (in whole
  // waves), a group a thread, so a row's slice is one round trip (a wave a row took two at 7168,
  // 0.8 us at 16-32 tokens), and as many rows at once as the block holds. A group a thread, not
  // a slice's worth: every peer's groups of a whole slice were 32 packs at 16384, spilling.
  constexpr int kSliceWaves = (TILE_N / WORLD / NL + kWaveSize - 1) / kWaveSize * kWaveSize;
  constexpr int SLICE_LANES = kSliceWaves < THREADS_PER_BLOCK ? kSliceWaves : THREADS_PER_BLOCK;
  using Slice = Tile<DTYPE, THREADS_PER_BLOCK / SLICE_LANES, SLICE_LANES * NL,
                     THREADS_PER_BLOCK / SLICE_LANES, SLICE_LANES, THREADS_PER_BLOCK>;
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);
  const int packs        = input_size_n / NL;  // the row, in packs (n is whole packs)
  const int slice        = (packs + WORLD - 1) / WORLD;
  const int col0         = rank * slice;
  const int own_packs = max(0, min(slice, packs - col0));  // the last rank's may be short
  const int my_rows      = input_size_m > static_cast<int>(blockIdx.x)
                               ? (input_size_m - blockIdx.x + gridDim.x - 1) / gridDim.x
                               : 0;

  // 1. Wait until every peer has launched, so its input is ready.
  block_stamp(0);
  barrier<Until::launched>(sync);
  block_stamp(1);

  // THE RANKS' POINTERS AFTER THE BARRIER, as in the pull kernels (held across it they spilled).
  const auto input = rank_ptrs<const DTYPE, WORLD>(input_ptrs, input_stride_m, input_stride_n);
  const auto scratch = rank_ptrs<DTYPE, WORLD>(scratch_ptrs, scratch_stride_m, scratch_stride_n);

  // 2. This rank's columns of this block's rows, summed over the ranks in rank order and pushed
  //    to every rank (itself too), at their place in the tensor: tiles of this block's rows (every
  //    gridDim.x-th), A WAVE A ROW, since a rank's columns are a narrow slice.
  if (own_packs > 0) {
    for (int q = 0; q < my_rows; q += Slice::kThreadsM)
    for (int c = col0 * NL; c < (col0 + own_packs) * NL; c += Slice::kTileN) {
      const Slice at{input_size_m, (col0 + own_packs) * NL,
      static_cast<int>(blockIdx.x + q * gridDim.x),
                     c, static_cast<int>(gridDim.x)};
      Slice peers[WORLD];
#pragma unroll
      for (int r = 0; r < WORLD; ++r) peers[r] = at;
      tile_load(peers, input);
      const Slice sum = peers_reduce(peers);
#pragma unroll
      for (int r = 0; r < WORLD; ++r) tile_store(sum, scratch[r]);
    }
  }
  block_stamp(2);

  // 3. Every rank's sums are in this rank's scratch, and every peer has read this rank's input.
  barrier<Until::visible>(sync);
  block_stamp(3);

  // 4. This block's tiles (one row): the sum out of this rank's scratch, then AttnRes, as the
  //    one-shot does. The next call's first sync keeps a peer from pushing into this scratch while
  //    it is read (a peer's next kernel starts only once this one has finished).
  const auto own_scratch = rank_ptr<DTYPE, WORLD>(scratch_ptrs, rank, scratch_stride_m,
  scratch_stride_n);
  const auto prefix     = local_ptr(prefix_ptr, prefix_stride_m, prefix_stride_n, rank);
  const auto prefix_output = local_ptr(prefix_output_ptr, prefix_output_stride_m,
                                         prefix_output_stride_n, rank);
  const auto block_output  = local_ptr(block_output_ptr, block_output_stride_m,
                                         block_output_stride_n, rank);
  const auto norm_weight     = local_ptr(norm_weight_ptr, 0, norm_weight_stride_n, rank);
  const auto qk_weight       = local_ptr(qk_weight_ptr, 0, qk_weight_stride_n, rank);
  const auto output_norm_weight = local_ptr(output_norm_weight_ptr, 0, output_norm_weight_stride_n, rank);
  const auto output        = local_ptr(output_ptr, output_stride_m, output_stride_n, rank);
  for (int row = blockIdx.x; row < input_size_m; row += gridDim.x) {
    Row sum{input_size_m, input_size_n, row, 0};
    tile_load(sum, own_scratch);
    block_attn_res_tile<HAS_PREFIX, TILE_K>(
        sum, prefix, blocks_ptr, blocks_stride_m, blocks_stride_r, norm_weight, qk_weight,
        output_norm_weight, prefix_output, output, block_output, num_blocks, eps, output_eps,
        inv_hidden);
  }
  block_stamp(4);
}

}  // namespace hip_custom_communicator
