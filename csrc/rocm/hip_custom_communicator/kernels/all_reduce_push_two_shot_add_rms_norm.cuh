// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot all-reduce (the reduce-scatter pulled, the all-gather pushed) then RMSNorm
// (`all_reduce_push_two_shot_rms_norm`), and all-reduce then add then RMSNorm
// (`all_reduce_push_two_shot_add_rms_norm`): one body, a kernel per op, so a trace names the op
// that ran.

#pragma once

#include "../common/interface.cuh"

namespace hip_custom_communicator {

// THE SLICE IS COLUMNS, as in aiter's fused two-stage kernel: each rank owns columns
// [rank * S, rank * S + S) of EVERY row (S = n / ngpus, rounded up to whole packs). It sums its
// columns of a row over the ranks and pushes the sums into every rank's scratch, so after the sync
// each rank holds the whole all-reduced tensor locally and norms its rows itself: no remote
// gather, and the norm repeated on every rank instead of sent. Every rank computes the same bytes
// (the same sums in the same order), so every rank holds the same result. ROW q BELONGS TO BLOCK
// q % gridDim.x IN BOTH PHASES: after the sync a block may read only what the same block on a peer
// wrote.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, bool ADD_RESIDUAL, int TILE_N, int THREADS_PER_BLOCK>
DINLINE void all_reduce_push_two_shot_add_rms_norm_body(
    const DTYPE* const* __restrict__ input_ptrs, int64_t input_stride_m, int64_t input_stride_n,
    DTYPE* const* __restrict__ scratch_ptrs, int64_t scratch_stride_m, int64_t scratch_stride_n,
    uint32_t* const* __restrict__ signals_ptrs, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer, uint32_t* __restrict__ status_ptr,
    int64_t status_stride_field, int rank,
    uint64_t timeout_ticks,
    Ptr<DTYPE> output, Ptr<DTYPE> residual_output, Ptr<const DTYPE> residual,
    Ptr<const WEIGHT_DTYPE> weight, float eps, int input_size_m, int input_size_n) {
  Sync<WORLD> sync{signals_ptrs, signals_stride_parity, signals_stride_block,
                   signals_stride_writer, status_ptr, status_stride_field, rank,
                   timeout_ticks};
  constexpr int NL       = traits<DTYPE>::N;
  using Row              = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  using RowF             = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK, float>;
  using Weight           = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK, WEIGHT_DTYPE>;
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);
  const int slice        = (input_size_n / NL + WORLD - 1) / WORLD * NL;  // a rank's, in elements
  const int col0         = rank * slice;
  const int own_n        = max(0, min(slice, input_size_n - col0));  // the last rank's may be short
  const int my_m         = input_size_m > static_cast<int>(blockIdx.x)
                               ? (input_size_m - blockIdx.x + gridDim.x - 1) / gridDim.x
                               : 0;
  // A REDUCE-SCATTER TILE: a row of threads as wide as a rank's slice of a TILE_N row (in whole
  // waves), a group a thread, so a row's slice is one round trip (a wave a row took two at 7168,
  // 0.8 us at 16-32 tokens), and as many rows at once as the block holds. A group a thread, not
  // a slice's worth: every peer's groups of a whole slice were 32 packs at 16384, spilling.
  constexpr int kSliceWaves = (TILE_N / WORLD / NL + kWaveSize - 1) / kWaveSize * kWaveSize;
  constexpr int SLICE_LANES = kSliceWaves < THREADS_PER_BLOCK ? kSliceWaves : THREADS_PER_BLOCK;
  using Slice = Tile<DTYPE, THREADS_PER_BLOCK / SLICE_LANES, SLICE_LANES * NL,
                     THREADS_PER_BLOCK / SLICE_LANES, SLICE_LANES, THREADS_PER_BLOCK>;

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
  if (own_n > 0) {
    for (int q = 0; q < my_m; q += Slice::kThreadsM)
    for (int c = col0; c < col0 + own_n; c += Slice::kTileN) {
      const Slice at{input_size_m, col0 + own_n, static_cast<int>(blockIdx.x + q * gridDim.x),
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

  // 3. Every rank's sums are in this rank's scratch.
  barrier<Until::visible>(sync);
  block_stamp(3);

  // 4. This block's rows out of this rank's scratch: (ADD_RESIDUAL) add the residual, then RMSNorm,
  //    rounding as the reference does (the one-shot kernel spells it out). The next call's first
  //    sync keeps a peer from pushing into this scratch while it is read (a peer's next kernel
  //    starts only once this one has finished).
  const auto own_scratch = rank_ptr<DTYPE, WORLD>(scratch_ptrs, rank, scratch_stride_m,
  scratch_stride_n);
  for (int row = blockIdx.x; row < input_size_m; row += gridDim.x) {
    const Row at{input_size_m, input_size_n, row, 0};
    // Every load of the row before any store, in flight together: the scratch's, the residual's
    // and the weight's (one-shot's).
    Row own = at, res = at;
    tile_load(own, own_scratch);
    if constexpr (ADD_RESIDUAL) tile_load(res, residual);
    Weight w{1, input_size_n, 0, 0};
    tile_load(w, weight);
    RowF s = own.template to<float>();
    if constexpr (ADD_RESIDUAL) {
      s = tile_add(s, res.template to<float>());
      tile_store(s.template to<DTYPE>(), residual_output);
    }
    float ss[1];
    partial_dot(s, s, ss);
    block_reduce<Sum, THREADS_PER_BLOCK>(ss);
    const float scale = rsqrtf(ss[0] * inv_hidden + eps);
    // out = T(W(W(s * scale) * float(w))), as the reference rounds
    Row normed = tile_mul(tile_mul(s, scale).template to<WEIGHT_DTYPE>().template to<float>(),
                            w.template to<float>())
                     .template to<WEIGHT_DTYPE>()
                     .template to<DTYPE>();
    tile_store(normed, output);
  }
  block_stamp(4);
}

// THE KERNELS, one per op, both the body above.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_push_two_shot_rms_norm(const DTYPE* const* __restrict__ input_ptrs,
                                      int64_t input_stride_m,
                                      int64_t input_stride_n,
                                      int input_size_m,
                                      int input_size_n,
                                      const WEIGHT_DTYPE* __restrict__ weight_ptr,
                                      int64_t weight_stride_n,
                                      int weight_size_n,
                                      DTYPE* __restrict__ output_ptr,
                                      int64_t output_stride_m,
                                      int64_t output_stride_n,
                                      int output_size_m,
                                      int output_size_n,
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
                                      uint32_t* __restrict__ status_ptr,
                                      int64_t status_stride_field,
                                      int status_size_field,
                                      int rank,
                                      uint64_t timeout_ticks,
                                      float eps) {
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  all_reduce_push_two_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, false, TILE_N, THREADS_PER_BLOCK>(
      input_ptrs, input_stride_m, input_stride_n, scratch_ptrs, scratch_stride_m, scratch_stride_n,
      signals_ptrs, signals_stride_parity, signals_stride_block, signals_stride_writer, status_ptr, status_stride_field,
      rank, timeout_ticks,
      local_ptr(output_ptr, output_stride_m, output_stride_n, rank), Ptr<DTYPE>{nullptr, 0, 0, rank},
      Ptr<const DTYPE>{nullptr, 0, 0, rank}, local_ptr(weight_ptr, 0, weight_stride_n, rank),
      eps, input_size_m, input_size_n);
#else
  __builtin_trap();
#endif
}

template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_push_two_shot_add_rms_norm(const DTYPE* const* __restrict__ input_ptrs,
                                          int64_t input_stride_m,
                                          int64_t input_stride_n,
                                          int input_size_m,
                                          int input_size_n,
                                          const DTYPE* __restrict__ residual_ptr,
                                          int64_t residual_stride_m,
                                          int64_t residual_stride_n,
                                          int residual_size_m,
                                          int residual_size_n,
                                          const WEIGHT_DTYPE* __restrict__ weight_ptr,
                                          int64_t weight_stride_n,
                                          int weight_size_n,
                                          DTYPE* __restrict__ output_ptr,
                                          int64_t output_stride_m,
                                          int64_t output_stride_n,
                                          int output_size_m,
                                          int output_size_n,
                                          DTYPE* __restrict__ residual_output_ptr,
                                          int64_t residual_output_stride_m,
                                          int64_t residual_output_stride_n,
                                          int residual_output_size_m,
                                          int residual_output_size_n,
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
                                          uint32_t* __restrict__ status_ptr,
                                          int64_t status_stride_field,
                                          int status_size_field,
                                          int rank,
                                          uint64_t timeout_ticks,
                                          float eps) {
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  all_reduce_push_two_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, true, TILE_N, THREADS_PER_BLOCK>(
      input_ptrs, input_stride_m, input_stride_n, scratch_ptrs, scratch_stride_m, scratch_stride_n,
      signals_ptrs, signals_stride_parity, signals_stride_block, signals_stride_writer, status_ptr, status_stride_field,
      rank, timeout_ticks,
      local_ptr(output_ptr, output_stride_m, output_stride_n, rank),
      local_ptr(residual_output_ptr, residual_output_stride_m, residual_output_stride_n, rank),
      local_ptr(residual_ptr, residual_stride_m, residual_stride_n, rank),
      local_ptr(weight_ptr, 0, weight_stride_n, rank), eps, input_size_m, input_size_n);
#else
  __builtin_trap();
#endif
}

}  // namespace hip_custom_communicator
