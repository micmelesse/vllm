// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot all-reduce (the reduce-scatter pulled, the all-gather pushed) then RMSNorm
// (`all_reduce_push_two_shot_rms_norm`), and all-reduce then add then RMSNorm
// (`all_reduce_push_two_shot_add_rms_norm`): one body, a kernel per op, so a trace names the op
// that ran.

#pragma once

#include "../p2p/p2p.cuh"
#include "../common/common.cuh"

namespace hip_comms {

// THE SLICE IS COLUMNS, as in aiter's fused two-stage kernel: each rank owns packs
// [rank * S, rank * S + S) of EVERY row (S = packs / ngpus, rounded up). It sums its columns of a
// row over the ranks and pushes the sums into every rank's scratch, so after the sync each rank
// holds the whole all-reduced tensor locally and norms its rows itself: no remote gather, and the
// norm repeated on every rank instead of sent. Every rank computes the same bytes (the same sums
// in the same order), so every rank holds the same result. ROW q BELONGS TO BLOCK q % gridDim.x
// IN BOTH PHASES: after the sync a block may read only what the same block on a peer wrote.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, bool ADD_RESIDUAL, int TILE_N, int THREADS_PER_BLOCK>
DINLINE void all_reduce_push_two_shot_add_rms_norm_body(
    const p2p::PeerPtrs* __restrict__ peer_inputs, p2p::PeerPtrs peer_scratch,
    p2p::PeerSignals peer_signals, p2p::Signal* self_signal, int rank, uint64_t timeout_ticks,
    DTYPE* __restrict__ out, DTYPE* __restrict__ residual_out, const DTYPE* __restrict__ residual,
    const WEIGHT_DTYPE* __restrict__ weight, float eps, int rows, int packs) {
  constexpr int NL       = traits<DTYPE>::N;
  using Row              = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK>;
  using RowF             = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, float>;
  using Weight           = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, WEIGHT_DTYPE>;
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  const int slice        = (packs + WORLD - 1) / WORLD;
  const int col0         = rank * slice;
  const int own_packs = max(0, min(slice, packs - col0));  // the last rank's may be short
  const int my_rows      = rows > static_cast<int>(blockIdx.x)
                               ? (rows - blockIdx.x + gridDim.x - 1) / gridDim.x
                               : 0;
  const int cols = packs * NL;  // the row, in elements
  // A REDUCE-SCATTER TILE: THREADS_PER_BLOCK / 64 rows of this rank's columns, a wave a row.
  constexpr int SLICE_N = (TILE_N / WORLD + kWaveSize * NL - 1) / (kWaveSize * NL) * (kWaveSize * NL);
  using Slice = Tile<DTYPE, THREADS_PER_BLOCK / kWaveSize, SLICE_N, THREADS_PER_BLOCK / kWaveSize,
                     kWaveSize>;

  // 1. Wait until every peer has launched, so its input is ready.
  block_stamp(0);
  p2p::barrier<WORLD, p2p::Among::peers, p2p::Ensure::launched>(
      peer_signals, self_signal, rank, timeout_ticks);
  block_stamp(1);

  // THE RANKS' POINTERS AFTER THE BARRIER, as in the pull kernels (held across it they spilled).
  const auto inputs = p2p::inputs<DTYPE, WORLD>(*peer_inputs);
  const auto scratches = p2p::scratches<DTYPE, WORLD>(peer_scratch);
  const auto input = [&](int r) { return inputs[r].data(); };

  // 2. This rank's columns of this block's rows, summed over the ranks in rank order and pushed
  //    to every rank (itself too), at their place in the tensor: tiles of this block's rows (every
  //    gridDim.x-th), A WAVE A ROW, since a rank's columns are a narrow slice.
  if (own_packs > 0) {
    for (int q = 0; q < my_rows; q += Slice::kThreadsM) {
      const Slice at{rows, (col0 + own_packs) * NL, static_cast<int>(blockIdx.x + q * gridDim.x),
                     col0 * NL, static_cast<int>(gridDim.x)};
      Slice peers[WORLD];
#pragma unroll
      for (int r = 0; r < WORLD; ++r) peers[r] = at;
      peers_load(peers, input, cols);
      const Slice sum = peers_reduce(peers);
#pragma unroll
      for (int r = 0; r < WORLD; ++r) thread_store(scratches[r].data(), cols, sum);
    }
  }
  block_stamp(2);

  // 3. Every rank's sums are in this rank's scratch.
  p2p::barrier<WORLD, p2p::Among::peers, p2p::Ensure::visible>(
      peer_signals, self_signal, rank, timeout_ticks);
  block_stamp(3);

  // 4. This block's rows out of this rank's scratch: (ADD_RESIDUAL) add the residual, then RMSNorm,
  //    rounding as the reference does (the one-shot kernel spells it out). The next call's first
  //    sync keeps a peer from pushing into this scratch while it is read (a peer's next kernel
  //    starts only once this one has finished).
  const auto own_scratch = p2p::scratch<DTYPE, WORLD>(peer_scratch, rank);
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const Row at{rows, cols, row, 0};
    // Every load of the row before any store: the scratch's and the residual together, the weight
    // under the reduction.
    Row own = at, res = at;
    thread_load(own, own_scratch.data(), cols);
    if constexpr (ADD_RESIDUAL) thread_load(res, residual, cols);
    RowF s = own.template to<float>();
    if constexpr (ADD_RESIDUAL) {
      s = thread_add(s, res.template to<float>());
      thread_store(residual_out, cols, s.template to<DTYPE>());
    }
    Weight w{1, cols, 0, 0};
    thread_load(w, weight, 0);
    float ss[1];
    thread_dot(s, s, ss);
    block_reduce<Sum>(ss);
    const float scale = rsqrtf(ss[0] * inv_hidden + eps);
    // out = T(W(W(s * scale) * float(w))), as the reference rounds
    Row normed = thread_mul(thread_mul(s, scale).template to<WEIGHT_DTYPE>().template to<float>(),
                            w.template to<float>())
                     .template to<WEIGHT_DTYPE>()
                     .template to<DTYPE>();
    thread_store(out, cols, normed);
  }
  block_stamp(4);
}

// THE KERNELS, one per op, both the body above.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N, int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)
    all_reduce_push_two_shot_rms_norm(const p2p::PeerPtrs* __restrict__ peer_inputs,
                                      p2p::PeerPtrs peer_scratch, p2p::PeerSignals peer_signals,
                                      p2p::Signal* self_signal, int rank, uint64_t timeout_ticks,
                                      DTYPE* __restrict__ out, const WEIGHT_DTYPE* __restrict__ weight, float eps,
                                      int rows, int packs) {
  all_reduce_push_two_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, false, TILE_N, THREADS_PER_BLOCK>(
       peer_inputs, peer_scratch, peer_signals, self_signal, rank, timeout_ticks, out, nullptr,
      nullptr, weight, eps, rows, packs);
}

template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N, int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)
    all_reduce_push_two_shot_add_rms_norm(const p2p::PeerPtrs* __restrict__ peer_inputs,
                                          p2p::PeerPtrs peer_scratch, p2p::PeerSignals peer_signals,
                                          p2p::Signal* self_signal, int rank,
                                          uint64_t timeout_ticks, DTYPE* __restrict__ out,
                                          DTYPE* __restrict__ residual_out,
                                          const DTYPE* __restrict__ residual,
                                          const WEIGHT_DTYPE* __restrict__ weight, float eps, int rows,
                                          int packs) {
  all_reduce_push_two_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, true, TILE_N, THREADS_PER_BLOCK>(
       peer_inputs, peer_scratch, peer_signals, self_signal, rank, timeout_ticks, out, residual_out,
      residual, weight, eps, rows, packs);
}

}  // namespace hip_comms
