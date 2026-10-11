// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce then RMSNorm (`all_reduce_pull_one_shot_rms_norm`), and
// all-reduce then add then RMSNorm (`all_reduce_pull_one_shot_add_rms_norm`): one body, a
// kernel per op, so a trace names the op that ran.

#pragma once

#include "../common/interface.cuh"

namespace hip_comms {

// Every rank reduces every row and norms it where the sum lands in registers, saving an HBM round
// trip and a launch against an all-reduce then a norm kernel. A block owns a row. `residual` and
// `residual_out` are unused (null) unless kAdd; `weight` keeps its own dtype W (T or fp32), as
// vLLM's reference ops round to the WEIGHT's dtype (`vllm/ir/ops/layernorm.py`). The kernels turn
// their arguments into Ptrs; the peers' inputs become theirs after the first barrier.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, bool ADD_RESIDUAL, int TILE_N, int THREADS_PER_BLOCK>
DINLINE void all_reduce_pull_one_shot_add_rms_norm_body(
    const DTYPE* const* __restrict__ inp_ptrs, int64_t inp_stride_m, int64_t inp_stride_n,
    Signal* const* __restrict__ signal_ptrs, Signal* self_signal_ptr, int rank,
    uint64_t timeout_ticks,
    Ptr<DTYPE> out, Ptr<DTYPE> residual_out, Ptr<const DTYPE> residual, Ptr<const WEIGHT_DTYPE> weight,
    float eps, int inp_size_m, int inp_size_n) {
  Sync<WORLD> sync{signal_ptrs, self_signal_ptr, rank, timeout_ticks};
  using Row              = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  using RowF             = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK, float>;
  using Weight           = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK, WEIGHT_DTYPE>;
  const float inv_hidden = 1.0f / static_cast<float>(inp_size_n);

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto inp = rank_ptrs<const DTYPE, WORLD>(inp_ptrs, inp_stride_m, inp_stride_n);
  block_stamp(0);
  barrier<Group::peers, Until::launched>(sync);
  block_stamp(1);

  // 2. Each of this block's rows: read it from every rank in rank order and sum, then (ADD_RESIDUAL) add
  //    the residual, then RMSNorm, rounding as the reference does:
  //      s   = float(DTYPE(sum over ranks))             the all-reduce output, as it would land
  //      s  += float(residual); residual_out = DTYPE(s) ADD_RESIDUAL only (fused_add_rms_norm)
  //      out = DTYPE(WEIGHT_DTYPE(WEIGHT_DTYPE(s * rsqrt(mean(s^2) + eps)) * float(w)))
  //    The variance is of `s` before any further rounding, kept in registers between the passes.
  for (int row = blockIdx.x; row < inp_size_m; row += gridDim.x) {
    const Row at{inp_size_m, inp_size_n, row, 0};
    // Every load of the row before any store, in flight together: the peers', the residual's and
    // the weight's. A load issues where it is written (an address is built at its load), and the
    // weight after the peers' sum was a round trip of its own (+0.34 us at 16 x 3584,
    // 2026-10-04T17-25-35Z).
    Row peers[WORLD];
#pragma unroll
    for (int r = 0; r < WORLD; ++r) peers[r] = at;
    tile_load(peers, inp);
    Row res = at;
    if constexpr (ADD_RESIDUAL) tile_load(res, residual);
    Weight w{1, inp_size_n, 0, 0};
    tile_load(w, weight);
    RowF s = peers_reduce(peers).template to<float>();
    block_stamp(2);
    if constexpr (ADD_RESIDUAL) {
      s = tile_add(s, res.template to<float>());
      tile_store(s.template to<DTYPE>(), residual_out);
    }
    float ss[1];
    partial_dot(s, s, ss);
    block_reduce<Sum, THREADS_PER_BLOCK>(ss);
    block_stamp(3);
    const float scale = rsqrtf(ss[0] * inv_hidden + eps);
    // out = T(W(W(s * scale) * float(w))), as the reference rounds
    Row normed = tile_mul(tile_mul(s, scale).template to<WEIGHT_DTYPE>().template to<float>(),
                            w.template to<float>())
                     .template to<WEIGHT_DTYPE>()
                     .template to<DTYPE>();
    tile_store(normed, out);
  }

  block_stamp(4);
  // 3. No rank may overwrite its input until every peer has read it.
  barrier<Group::peers, Until::read>(sync);
  block_stamp(5);
  sync.finish();
}

// THE KERNELS, one per op, both the body above.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot_rms_norm(const DTYPE* const* __restrict__ inp_ptrs,
                                      int64_t inp_stride_m, int64_t inp_stride_n,
                                      Signal* const* __restrict__ signal_ptrs,
                                      Signal* self_signal_ptr,
                                      int rank, uint64_t timeout_ticks,
                                      DTYPE* __restrict__ out_ptr, int64_t out_stride_m,
                                      int64_t out_stride_n,
                                      const WEIGHT_DTYPE* __restrict__ weight_ptr,
                                      int64_t weight_stride_n, float eps, int inp_size_m,
                                      int inp_size_n) {
  all_reduce_pull_one_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, false, TILE_N, THREADS_PER_BLOCK>(
      inp_ptrs, inp_stride_m, inp_stride_n, signal_ptrs, self_signal_ptr, rank, timeout_ticks,
      local_ptr(out_ptr, out_stride_m, out_stride_n, rank), Ptr<DTYPE>{nullptr, 0, 0, rank},
      Ptr<const DTYPE>{nullptr, 0, 0, rank}, local_ptr(weight_ptr, 0, weight_stride_n, rank),
      eps, inp_size_m, inp_size_n);
}

template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot_add_rms_norm(const DTYPE* const* __restrict__ inp_ptrs,
                                          int64_t inp_stride_m, int64_t inp_stride_n,
                                          Signal* const* __restrict__ signal_ptrs,
                                          Signal* self_signal_ptr,
                                          int rank, uint64_t timeout_ticks,
                                          DTYPE* __restrict__ out_ptr, int64_t out_stride_m,
                                          int64_t out_stride_n,
                                          DTYPE* __restrict__ residual_out_ptr,
                                          int64_t residual_out_stride_m,
                                          int64_t residual_out_stride_n,
                                          const DTYPE* __restrict__ residual_ptr,
                                          int64_t residual_stride_m, int64_t residual_stride_n,
                                          const WEIGHT_DTYPE* __restrict__ weight_ptr,
                                          int64_t weight_stride_n, float eps, int inp_size_m,
                                          int inp_size_n) {
  all_reduce_pull_one_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, true, TILE_N, THREADS_PER_BLOCK>(
      inp_ptrs, inp_stride_m, inp_stride_n, signal_ptrs, self_signal_ptr, rank, timeout_ticks,
      local_ptr(out_ptr, out_stride_m, out_stride_n, rank),
      local_ptr(residual_out_ptr, residual_out_stride_m, residual_out_stride_n, rank),
      local_ptr(residual_ptr, residual_stride_m, residual_stride_n, rank),
      local_ptr(weight_ptr, 0, weight_stride_n, rank), eps, inp_size_m, inp_size_n);
}

}  // namespace hip_comms
