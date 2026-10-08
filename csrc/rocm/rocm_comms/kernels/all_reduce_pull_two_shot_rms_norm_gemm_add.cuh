// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce, then RMSNorm, then a GEMM whose result is written into an output
// (`all_reduce_pull_two_shot_rms_norm_gemm`) or added into it (`..._gemm_add`, the tail of
// Kimi-K3's latent MoE): one body, a kernel per op, so a trace names the op that ran.

#pragma once

#include "../common/interface.cuh"

namespace hip_comms {

// Each rank reduces and norms the rows it owns into its scratch, row-major; after the sync
// every rank copies every normed row into `workspace` ([rows, packs] of its own); a grid sync;
// the GEMM over every row, TILE_M per pass. THE SAME BLOCK AND THREAD INDEX A
// PACK IN BOTH PHASES: after the sync a block may read only what the same block on a peer wrote.
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N, int TILE_K, int SLICE_K,
          int THREADS_PER_BLOCK, bool ADD_RESIDUAL>
DINLINE void all_reduce_pull_two_shot_rms_norm_gemm_body(
    const PeerPtrs* __restrict__ peer_inputs, PeerPtrs peer_scratch,
    PeerSignals peer_signals, Signal* self_signal, int rank, uint64_t timeout_ticks,
    const DTYPE* __restrict__ norm_w, float eps, const DTYPE* __restrict__ gemm_w, int n_cols,
    DTYPE* __restrict__ out, int64_t out_stride, DTYPE* __restrict__ workspace, int rows, int packs) {
  Sync<WORLD> sync{peer_signals, self_signal, rank, timeout_ticks};
  constexpr int NL       = traits<DTYPE>::N;
  using Row              = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  using RowF             = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK, float>;
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  const int cols = packs * NL;  // the row, in elements
  const int slice_rows   = (rows + WORLD - 1) / WORLD;

  // 1. Wait until every peer has launched, so its input is ready.
  block_stamp(0);
  barrier<Group::peers, Until::launched>(sync);
  block_stamp(1);

  // THE RANKS' POINTERS AFTER THE BARRIER here: held across it, the 8-pack build keeps 68 B of
  // scratch (the ISA gate, 2026-09-30).
  const auto inputs = rank_ptrs<const DTYPE, WORLD>(*peer_inputs, cols);
  const auto scratches = rank_ptrs<DTYPE, WORLD>(peer_scratch, cols);
  const auto own_scratch = rank_ptr<DTYPE, WORLD>(peer_scratch, rank, cols);

  // 2. This rank's rows: read each from every rank in rank order, sum, norm, into this rank's
  //    scratch.
  const int first = rank * slice_rows;
  const int last  = min(first + slice_rows, rows);
  for (int row = first + blockIdx.x; row < last; row += gridDim.x) {
    Row peers[WORLD];
#pragma unroll
    for (int r = 0; r < WORLD; ++r) peers[r] = Row{rows, cols, row, 0};
    tile_load(peers, inputs);
    const RowF s = peers_reduce(peers).template to<float>();
    // The norm, rounding as vLLM's reference rms_norm does (weight in DTYPE):
    //   out = DTYPE(DTYPE(s * rsqrt(mean(s^2) + eps)) * float(w)), s = float(DTYPE(sum over ranks))
    Row wk{1, cols, 0, 0};
    tile_load(wk, local_ptr(norm_w, 0, rank));  // under the reduction
    float ss[1];
    partial_dot(s, s, ss);
    block_reduce<Sum, THREADS_PER_BLOCK>(ss);
    const float scale = rsqrtf(ss[0] * inv_hidden + eps);
    const RowF w = wk.template to<float>();
    // out = T(T(s * scale) * float(w)), as the reference rounds
    Row x = tile_mul(tile_mul(s, scale).template to<DTYPE>().template to<float>(), w)
                .template to<DTYPE>();
    x.offs_m = row - first;
    tile_store(x, own_scratch);
  }

  // 3. Every rank's normed rows are visible to its peers.
  block_stamp(2);
  barrier<Group::peers, Until::visible>(sync);
  block_stamp(3);

  // 4. Every owner's normed rows out of its scratch, into the workspace: EVERY OWNER'S TILE LOADED
  //    BEFORE ANY IS STORED (the compiler cannot prove the output and the peers' scratch apart, so
  //    a store between two loads held the next load back, and the eight owners' round trips ran one
  //    after another).
  using Chunk = Tile<DTYPE, 1, THREADS_PER_BLOCK * NL, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;  // WORLD packs a thread
  for (int l = blockIdx.x; l < slice_rows; l += gridDim.x)
    for (int c = 0; c < cols; c += Chunk::kTileN) {
      Chunk got[WORLD];
#pragma unroll
      for (int r = 0; r < WORLD; ++r) got[r] = Chunk{slice_rows, cols, l, c};
      tile_load(got, scratches);
#pragma unroll
      for (int r = 0; r < WORLD; ++r) {
        got[r].M      = rows;
        got[r].offs_m = r * slice_rows + l;
        tile_store(got[r], local_ptr(workspace, cols, rank));
      }
    }

  // 5. The GEMM reads rows other blocks of this rank copied.
  block_stamp(4);
  barrier<Group::grid, Until::visible>(sync);
  block_stamp(5);

  // 6. The GEMM over every row, TILE_M per pass.
  for (int r0 = 0; r0 < rows; r0 += TILE_M) {
    grid_gemm<TILE_M, TILE_K, SLICE_K, ADD_RESIDUAL, THREADS_PER_BLOCK>(workspace + int64_t{r0} * cols, cols,
                                                     min(TILE_M, rows - r0), cols, gemm_w, n_cols,
                                                     out + r0 * out_stride, out_stride);
  }
  block_stamp(6);
  sync.finish();
}

// THE KERNELS, one per op, both the body above: the GEMM's result written (rms_norm_gemm) or
// added into `out` (rms_norm_gemm_add).
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N, int TILE_K, int SLICE_K,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot_rms_norm_gemm(const PeerPtrs* __restrict__ peer_inputs,
                                           PeerPtrs peer_scratch,
                                           PeerSignals peer_signals, Signal* self_signal,
                                           int rank, uint64_t timeout_ticks,
                                           const DTYPE* __restrict__ norm_w, float eps,
                                           const DTYPE* __restrict__ gemm_w, int n_cols,
                                           DTYPE* __restrict__ out, int64_t out_stride,
                                           DTYPE* __restrict__ workspace, int rows, int packs) {
  if constexpr (gemm_fits(kDevice, TILE_M, TILE_K, SLICE_K, THREADS_PER_BLOCK))
    all_reduce_pull_two_shot_rms_norm_gemm_body<DTYPE, WORLD, TILE_M, TILE_N, TILE_K, SLICE_K,
                                                THREADS_PER_BLOCK, false>(
                                                     peer_inputs, peer_scratch, peer_signals,
                                                    self_signal, rank, timeout_ticks, norm_w, eps,
                                                    gemm_w, n_cols, out, out_stride, workspace,
                                                    rows, packs);
  else
    __builtin_trap();
}

template <typename DTYPE, int WORLD, int TILE_M, int TILE_N, int TILE_K, int SLICE_K,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot_rms_norm_gemm_add(const PeerPtrs* __restrict__ peer_inputs,
                                               PeerPtrs peer_scratch,
                                               PeerSignals peer_signals,
                                               Signal* self_signal, int rank,
                                               uint64_t timeout_ticks, const DTYPE* __restrict__ norm_w,
                                               float eps, const DTYPE* __restrict__ gemm_w, int n_cols,
                                               DTYPE* __restrict__ out, int64_t out_stride,
                                               DTYPE* __restrict__ workspace, int rows, int packs) {
  if constexpr (gemm_fits(kDevice, TILE_M, TILE_K, SLICE_K, THREADS_PER_BLOCK))
    all_reduce_pull_two_shot_rms_norm_gemm_body<DTYPE, WORLD, TILE_M, TILE_N, TILE_K, SLICE_K,
                                                THREADS_PER_BLOCK, true>(
                                                     peer_inputs, peer_scratch, peer_signals,
                                                    self_signal, rank, timeout_ticks, norm_w, eps,
                                                    gemm_w, n_cols, out, out_stride, workspace,
                                                    rows, packs);
  else
    __builtin_trap();
}

}  // namespace hip_comms
