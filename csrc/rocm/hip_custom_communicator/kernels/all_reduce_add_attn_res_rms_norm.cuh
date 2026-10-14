// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ALL-REDUCE THEN ADD, ATTNRES AND RMSNORM: the AttnRes tile every kernel
// shares, then the kernels, one section each (pull one-shot, pull two-shot,
// push two-shot).

#pragma once

#include <type_traits>
#include "common.cuh"

namespace hip_custom_communicator {

// =============================================================================
// attn_res
//
// A tile of Kimi-K3's attention residual (AttnRes) and its RMSNorm, every
// AttnRes kernel's: part of an op, composed of common's primitives, so it is
// neither a kernel nor a primitive.
// =============================================================================

// A TILE OF ATTNRES, every AttnRes kernel's: the tile's TILE_M rows at once,
// TILE_K sources a step (the reduced dimension, Triton's BLOCK_L), so each step
// pays ONE block reduction for every row and source of the tile. Rounds as
// `vllm/models/kimi_k3/amd/ops/attn_res.py` does:
//   d = float(T(sum over ranks)); u = kPrefix ? float(T(float(prefix) + d)) : d
//   (the prefix) logit(src) = dot(src, norm_w * qk_w) * rsqrt(mean(src^2) +
//   eps), src the blocks, then u m = softmax(logits) . sources, online, a tile
//   of sources at a time; out = T(m), or T(m * rsqrt(mean(m^2) + out_eps) *
//   out_w)
// `sum` is the tile's rows summed over the ranks; the prefix, its output and
// the output are rows of the tile's width at their Ptrs' strides, `blocks` [m,
// sources, n] at its row and source strides in elements (its columns
// contiguous, the host checks), and `block_output` the one block written, a
// null Ptr when none is. An input is only read and an output only written
// (STANDARDS): the old prefix is read from `prefix`, the new one written to
// `prefix_output`, never the same memory. A row past M (the last tile's) reads
// row M - 1 and writes nothing, so every thread still reaches every reduction.
template <bool HAS_PREFIX, int TILE_K, typename DTYPE, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK>
DINLINE void block_attn_res_tile(
    const Tile<DTYPE, TILE_M, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>&
        sum,
    Ptr<const DTYPE> prefix, const DTYPE* blocks, int64_t blocks_stride_m,
    int64_t blocks_stride_r, Ptr<const DTYPE> norm_weight,
    Ptr<const DTYPE> qk_weight, Ptr<const DTYPE> output_norm_weight,
    Ptr<DTYPE> prefix_output, Ptr<DTYPE> output, Ptr<DTYPE> block_output,
    int num_blocks, float eps, float output_eps, float inv_hidden) {
  using Rows =
      Tile<DTYPE, TILE_M, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  using RowsF = Tile<DTYPE, TILE_M, TILE_N, 1, THREADS_PER_BLOCK,
                     THREADS_PER_BLOCK, float>;
  using Weight =
      Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  using WeightF =
      Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK, float>;
  const Weight at_cols{1, sum.N, 0, sum.offs_n};

  // The new prefix: the sum over the ranks added to the old one, rounded once
  // to DTYPE; the old prefix loaded whole before anything is stored.
  Rows np = sum;
  if constexpr (HAS_PREFIX) {
    Rows old = sum.template like<DTYPE>();
    tile_load(old, prefix);
    np = tile_add(old.template to<float>(), sum.template to<float>())
             .template to<DTYPE>();
  }
  tile_store(np, prefix_output);
  if (block_output.data != nullptr) tile_store(np, block_output);
  // THE PREFIX IS NOT HELD ACROSS THE LOOP: the last source, it is read back
  // from `prefix_output`, where this launch just stored it (Triton's kernel
  // does the same: held in fp32 it was 32 VGPRs at 7168, 256 threads).
  RowsF acc = np.template to<float>();
  if (num_blocks != 0) {
    Weight nw = at_cols, qk = at_cols;
    tile_load(nw, norm_weight);
    tile_load(qk, qk_weight);
    const WeightF w =
        tile_mul(nw.template to<float>(), qk.template to<float>());
    acc = sum.template zeros<float>();
    OnlineSoftmax softmax[TILE_M];
    // FULL STEPS OF SOURCES, THEN ONE LAST STEP with what remains, the prefix
    // and any padding: in the loop every source is a block's, so it has no
    // branch and no zeroed tile a source (16 moves a step at 3584, ISA
    // 2026-10-04T03-31-28Z).
    const int full = num_blocks / TILE_K * TILE_K;
    const auto step = [&](int src0, auto last) {
      Rows v[TILE_K];
      float sums[TILE_M * 2 * TILE_K];
#pragma unroll
      for (int s = 0; s < TILE_K; ++s) {
        const int src = src0 + s;
        // THE LAST STEP'S SOURCE IS A BLOCK OR THE PREFIX, picked by its
        // pointer and loaded once: a zeroed tile and a load under a branch held
        // 16 VGPRs at the kernel's peak (ISA 2026-10-04T16-26-29Z). A padding
        // source re-reads the prefix; its logit is -inf.
        const bool block = !decltype(last)::value || src < num_blocks;
        v[s] = sum.template like<DTYPE>();
        tile_load(v[s],
                  block ? blocks + src * blocks_stride_r : prefix_output.data,
                  block ? blocks_stride_m : prefix_output.stride_m);
        float ss[TILE_M], dw[TILE_M];
        partial_dot(v[s], v[s], ss);
        partial_dot(v[s], w, dw);
#pragma unroll
        for (int m = 0; m < TILE_M; ++m) {
          sums[(m * TILE_K + s) * 2] = ss[m];
          sums[(m * TILE_K + s) * 2 + 1] = dw[m];
        }
      }
      block_reduce<Sum, THREADS_PER_BLOCK>(sums);
      float old_scale[TILE_M], scale[TILE_K][TILE_M];
#pragma unroll
      for (int m = 0; m < TILE_M; ++m) {
        float logit[TILE_K];
#pragma unroll
        for (int s = 0; s < TILE_K; ++s)
          logit[s] =
              !decltype(last)::value || src0 + s <= num_blocks
                  ? sums[(m * TILE_K + s) * 2 + 1] *
                        rsqrtf(sums[(m * TILE_K + s) * 2] * inv_hidden + eps)
                  : -INFINITY;
        float row_scale[TILE_K];
        old_scale[m] = thread_softmax_fold(softmax[m], logit, row_scale);
#pragma unroll
        for (int s = 0; s < TILE_K; ++s) scale[s][m] = row_scale[s];
      }
      acc = tile_fma(acc, old_scale, v[0], scale[0]);
      if constexpr (TILE_K > 1) {
        float ones[TILE_M];
#pragma unroll
        for (int m = 0; m < TILE_M; ++m) ones[m] = 1.0f;
#pragma unroll
        for (int s = 1; s < TILE_K; ++s)
          acc = tile_fma(acc, ones, v[s], scale[s]);
      }
    };
    for (int src0 = 0; src0 < full; src0 += TILE_K)
      step(src0, std::false_type{});
    step(full, std::true_type{});
    float inv_den[TILE_M];
#pragma unroll
    for (int m = 0; m < TILE_M; ++m) inv_den[m] = 1.0f / softmax[m].denominator;
    acc = tile_mul(acc, inv_den);
  }

  // The output, normed when out_norm_w is given: its weight in flight under the
  // one reduction for every row's sum of squares.
  Rows result = sum.template like<DTYPE>();
  if (output_norm_weight.data != nullptr) {
    Weight g_in = at_cols;
    tile_load(g_in, output_norm_weight);
    float ss[TILE_M];
    partial_dot(acc, acc, ss);
    block_reduce<Sum, THREADS_PER_BLOCK>(ss);
    float scale[TILE_M];
#pragma unroll
    for (int m = 0; m < TILE_M; ++m)
      scale[m] = rsqrtf(ss[m] * inv_hidden + output_eps);
    result = tile_mul(tile_mul(acc, scale), g_in.template to<float>())
                 .template to<DTYPE>();
  } else {
    result = acc.template to<DTYPE>();
  }
  tile_store(result, output);
}

// =============================================================================
// all_reduce_pull_one_shot_add_attn_res_rms_norm
//
// One-shot pull all-reduce fused with Kimi-K3's attention residual (AttnRes)
// and its RMSNorm.
// =============================================================================

// A block owns a row, as the fused norm does: every rank reduces every row, so
// there is nothing to gather. `blocks` is [m, num_sources, n] at its row,
// source and column strides in elements; `write_idx` < 0 writes no block.
template <typename DTYPE, int WORLD, bool HAS_PREFIX, int TILE_N, int TILE_K,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot_add_attn_res_rms_norm(
        const DTYPE* const* __restrict__ input_ptrs, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const DTYPE* __restrict__ prefix_ptr, int64_t prefix_stride_m,
        int64_t prefix_stride_n, int prefix_size_m, int prefix_size_n,
        const DTYPE* __restrict__ blocks_ptr, int64_t blocks_stride_m,
        int64_t blocks_stride_r, int64_t blocks_stride_n, int blocks_size_m,
        int blocks_size_r, int blocks_size_n,
        const DTYPE* __restrict__ norm_weight_ptr, int64_t norm_weight_stride_n,
        int norm_weight_size_n, const DTYPE* __restrict__ qk_weight_ptr,
        int64_t qk_weight_stride_n, int qk_weight_size_n,
        const DTYPE* __restrict__ output_norm_weight_ptr,
        int64_t output_norm_weight_stride_n, int output_norm_weight_size_n,
        DTYPE* __restrict__ prefix_output_ptr, int64_t prefix_output_stride_m,
        int64_t prefix_output_stride_n, int prefix_output_size_m,
        int prefix_output_size_n, DTYPE* __restrict__ output_ptr,
        int64_t output_stride_m, int64_t output_stride_n, int output_size_m,
        int output_size_n, DTYPE* __restrict__ block_output_ptr,
        int64_t block_output_stride_m, int64_t block_output_stride_n,
        int block_output_size_m, int block_output_size_n,
        uint32_t* const* __restrict__ signals_ptrs,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks, int num_blocks,
        float eps, float output_eps) {
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  Sync<WORLD> sync{signals_ptrs,
                   signals_stride_parity,
                   signals_stride_block,
                   signals_stride_writer,
                   status_ptr,
                   status_stride_field,
                   rank,
                   timeout_ticks};
  using Row = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);

  // 1. Every rank's buffers, then wait until every peer has launched, so its
  // input is ready.
  const auto input =
      rank_ptrs<const DTYPE, WORLD>(input_ptrs, input_stride_m, input_stride_n);
  const auto prefix =
      local_ptr(prefix_ptr, prefix_stride_m, prefix_stride_n, rank);
  const auto prefix_output = local_ptr(
      prefix_output_ptr, prefix_output_stride_m, prefix_output_stride_n, rank);
  const auto block_output = local_ptr(block_output_ptr, block_output_stride_m,
                                      block_output_stride_n, rank);
  const auto norm_weight =
      local_ptr(norm_weight_ptr, 0, norm_weight_stride_n, rank);
  const auto qk_weight = local_ptr(qk_weight_ptr, 0, qk_weight_stride_n, rank);
  const auto output_norm_weight =
      local_ptr(output_norm_weight_ptr, 0, output_norm_weight_stride_n, rank);
  const auto output =
      local_ptr(output_ptr, output_stride_m, output_stride_n, rank);
  barrier<Until::launched>(sync);

  // 2. Each of this block's tiles (one row: TILE_M = 1): read it from every
  // rank in rank order,
  //    sum, AttnRes.
  for (int row = blockIdx.x; row < input_size_m; row += gridDim.x) {
    Row peers[WORLD];
  #pragma unroll
    for (int r = 0; r < WORLD; ++r)
      peers[r] = Row{input_size_m, input_size_n, row, 0};
    tile_load(peers, input);
    block_attn_res_tile<HAS_PREFIX, TILE_K>(
        peers_reduce(peers), prefix, blocks_ptr, blocks_stride_m,
        blocks_stride_r, norm_weight, qk_weight, output_norm_weight,
        prefix_output, output, block_output, num_blocks, eps, output_eps,
        inv_hidden);
  }

  // 3. No rank may overwrite its input until every peer has read it.
  barrier<Until::read>(sync);
#else
  __builtin_trap();
#endif
}

// =============================================================================
// all_reduce_pull_two_shot_add_attn_res_rms_norm
//
// Two-shot pull all-reduce fused with Kimi-K3's attention residual (AttnRes)
// and its RMSNorm.
// =============================================================================

// THE SLICE IS COLUMNS, AND BOTH HALVES PULL, as the plain two-shot all-reduce
// moves its bytes: each rank sums its columns of every row over the ranks into
// its own scratch; after the sync each block reads its rows' columns from their
// owners and computes AttnRes on them itself. So the links carry what an
// all-reduce's do, AttnRes's two outputs (the prefix and out) are never
// gathered, and every rank does the AttnRes the unfused path would. A SLICE IS
// WHOLE WAVES (64 packs), so every wave's packs have one owner and scratch's
// rank is the same across the wave. A BLOCK REDUCES THE ROWS IT THEN NORMS, so
// it waits only for its twin on every rank (a peers barrier, as the plain
// two-shot's) and the grid is AttnRes's to size. The reduce-scatter on 32
// blocks behind a world barrier left every other block waiting as long as it
// then computed, on a grid the barrier capped at the 192 blocks resident
// (thread trace 2026-10-04T20-14-36Z). `blocks` is [m, num_sources, n] with row
// and source strides in elements; `write_idx` < 0 writes no block.
template <typename DTYPE, int WORLD, bool HAS_PREFIX, int TILE_M, int TILE_N,
          int TILE_K, int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot_add_attn_res_rms_norm(
        const DTYPE* const* __restrict__ input_ptrs, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const DTYPE* __restrict__ prefix_ptr, int64_t prefix_stride_m,
        int64_t prefix_stride_n, int prefix_size_m, int prefix_size_n,
        const DTYPE* __restrict__ blocks_ptr, int64_t blocks_stride_m,
        int64_t blocks_stride_r, int64_t blocks_stride_n, int blocks_size_m,
        int blocks_size_r, int blocks_size_n,
        const DTYPE* __restrict__ norm_weight_ptr, int64_t norm_weight_stride_n,
        int norm_weight_size_n, const DTYPE* __restrict__ qk_weight_ptr,
        int64_t qk_weight_stride_n, int qk_weight_size_n,
        const DTYPE* __restrict__ output_norm_weight_ptr,
        int64_t output_norm_weight_stride_n, int output_norm_weight_size_n,
        DTYPE* __restrict__ prefix_output_ptr, int64_t prefix_output_stride_m,
        int64_t prefix_output_stride_n, int prefix_output_size_m,
        int prefix_output_size_n, DTYPE* __restrict__ output_ptr,
        int64_t output_stride_m, int64_t output_stride_n, int output_size_m,
        int output_size_n, DTYPE* __restrict__ block_output_ptr,
        int64_t block_output_stride_m, int64_t block_output_stride_n,
        int block_output_size_m, int block_output_size_n,
        DTYPE* const* __restrict__ scratch_ptrs, int64_t scratch_stride_m,
        int64_t scratch_stride_n, int scratch_size_m, int scratch_size_n,
        uint32_t* const* __restrict__ signals_ptrs,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks, int num_blocks,
        float eps, float output_eps, int reduce_scatter_blocks) {
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  Sync<WORLD> sync{signals_ptrs,
                   signals_stride_parity,
                   signals_stride_block,
                   signals_stride_writer,
                   status_ptr,
                   status_stride_field,
                   rank,
                   timeout_ticks};
  using Rows =
      Tile<DTYPE, TILE_M, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  constexpr int NL = traits<DTYPE>::N;
  const int packs = input_size_n / NL;  // the row, in packs (n is whole packs)
  // THE REDUCE-SCATTER'S CHUNK, the plain two-shot's: its rows are the ranks, a
  // row of threads a rank (common's reduce_scatter).
  constexpr int LANES = THREADS_PER_BLOCK / WORLD;
  static_assert(LANES * WORLD == THREADS_PER_BLOCK,
                "a block is a row of threads a rank");
  using Ranks = Tile<DTYPE, WORLD, LANES * NL, WORLD, LANES, THREADS_PER_BLOCK>;
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);
  const int per_rank = (packs + WORLD - 1) / WORLD;
  // EVERY RANK AN EQUAL SLICE: rounded up to whole waves (so a wave's packs had
  // one owner), rank 7 owned nothing at 3584 and 7168 and the others
  // reduce-scattered 8/7 of the row.
  const int slice = per_rank;
  const int col0 = min(rank * slice, packs);
  const int own_packs =
      max(0, min(slice, packs - col0));  // a late rank's may be short or none

  // 1. Wait until every peer has launched, so its input is ready.
  block_stamp(0);
  barrier<Until::launched>(sync);
  block_stamp(1);

  // THE PEERS READ ROTATED, rank + r's r-th, as the plain two-shot reads them:
  // in rank order every GPU read rank 0 first, a link at a time carrying the
  // machine's reads, and the reduce-scatter took 1.6x the plain two-shot's
  // cycles for its bytes (thread traces 2026-10-04T21-03-09Z, 21-26-11Z). Each
  // slice is summed by one rank, so the order differing by rank is harmless. A
  // TILE ROW'S RANK DIFFERS ACROSS THE WAVE, so each lane reads its own from
  // the table (an array of every rank's, indexed at run time, was scratch: 136
  // B, 520 B as Ptrs).
  const auto input = [&](int r) { return input_ptrs[(rank + r) % WORLD]; };
  const auto own_scratch =
      rank_ptr<DTYPE, WORLD>(scratch_ptrs, rank, scratch_stride_m,
                             scratch_stride_n)
          .data;
  const auto prefix =
      local_ptr(prefix_ptr, prefix_stride_m, prefix_stride_n, rank);
  const auto prefix_output = local_ptr(
      prefix_output_ptr, prefix_output_stride_m, prefix_output_stride_n, rank);
  const auto block_output = local_ptr(block_output_ptr, block_output_stride_m,
                                      block_output_stride_n, rank);
  const auto norm_weight =
      local_ptr(norm_weight_ptr, 0, norm_weight_stride_n, rank);
  const auto qk_weight = local_ptr(qk_weight_ptr, 0, qk_weight_stride_n, rank);
  const auto output_norm_weight =
      local_ptr(output_norm_weight_ptr, 0, output_norm_weight_stride_n, rank);
  const auto output =
      local_ptr(output_ptr, output_stride_m, output_stride_n, rank);

  // A WAVE'S PACKS MAY HAVE TWO OWNERS, so each lane reads its owner's scratch
  // from the table.
  const auto scratch_of = [&](int r) { return scratch_ptrs[r]; };
  const int end =
      (col0 + own_packs) * NL;  // this rank's columns end, in elements
  // One tile's reduce-scatter: this rank's columns of its rows summed over the
  // ranks into this rank's scratch, a chunk at a time.
  const auto reduce_tile = [&](int offs_m) {
    for (int row = offs_m; row < min(offs_m + TILE_M, input_size_m); ++row)
      for (int c = col0 * NL; c < end; c += Ranks::kTileN) {
        Ranks chunk{WORLD, end, 0, c};
        reduce_scatter(
            chunk, [&](int w) { return input(w) + row * input_stride_m; },
            own_scratch + row * scratch_stride_m);
      }
  };
  // One tile's packs from the ranks that own them, then AttnRes on it.
  const auto attn_res_tile = [&](int offs_m) {
    Rows sum{input_size_m, input_size_n, offs_m, 0};
    sliced_load<WORLD>(sum, scratch_of, scratch_stride_m, slice * NL);
    block_attn_res_tile<HAS_PREFIX, TILE_K>(
        sum, prefix, blocks_ptr, blocks_stride_m, blocks_stride_r, norm_weight,
        qk_weight, output_norm_weight, prefix_output, output, block_output,
        num_blocks, eps, output_eps, inv_hidden);
  };
  // Every block its own tiles: reduce-scatter them, meet its twins, AttnRes on
  // them.
  (void)reduce_scatter_blocks;
  const int tiles = (input_size_m + TILE_M - 1) / TILE_M;
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) reduce_tile(t * TILE_M);
  block_stamp(2);
  barrier<Until::visible>(sync);
  block_stamp(3);
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) attn_res_tile(t * TILE_M);
  block_stamp(4);
#else
  __builtin_trap();
#endif
}

// =============================================================================
// all_reduce_push_two_shot_add_attn_res_rms_norm
//
// Two-shot all-reduce (the reduce-scatter pulled, the all-gather pushed) fused
// with Kimi-K3's attention residual (AttnRes) and its RMSNorm: the push norm's
// two phases, with the one-shot AttnRes's row in place of the norm.
// =============================================================================

// THE SLICE IS COLUMNS, as the push norm's: each rank owns packs [rank * S,
// rank * S + S) of every row, sums them over the ranks and pushes the sums into
// every rank's scratch, so after the sync each rank holds the whole sum locally
// and computes AttnRes on its rows itself: what crosses the links is the sum,
// once, and AttnRes's two outputs (the prefix and out) are never gathered. ROW
// q BELONGS TO BLOCK q % gridDim.x IN BOTH PHASES: after the sync a block may
// read only what the same block on a peer wrote. `blocks` is [m, num_sources,
// n] with row and source strides in elements; `write_idx` < 0 writes no block.
template <typename DTYPE, int WORLD, bool HAS_PREFIX, int TILE_N, int TILE_K,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_push_two_shot_add_attn_res_rms_norm(
        const DTYPE* const* __restrict__ input_ptrs, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const DTYPE* __restrict__ prefix_ptr, int64_t prefix_stride_m,
        int64_t prefix_stride_n, int prefix_size_m, int prefix_size_n,
        const DTYPE* __restrict__ blocks_ptr, int64_t blocks_stride_m,
        int64_t blocks_stride_r, int64_t blocks_stride_n, int blocks_size_m,
        int blocks_size_r, int blocks_size_n,
        const DTYPE* __restrict__ norm_weight_ptr, int64_t norm_weight_stride_n,
        int norm_weight_size_n, const DTYPE* __restrict__ qk_weight_ptr,
        int64_t qk_weight_stride_n, int qk_weight_size_n,
        const DTYPE* __restrict__ output_norm_weight_ptr,
        int64_t output_norm_weight_stride_n, int output_norm_weight_size_n,
        DTYPE* __restrict__ prefix_output_ptr, int64_t prefix_output_stride_m,
        int64_t prefix_output_stride_n, int prefix_output_size_m,
        int prefix_output_size_n, DTYPE* __restrict__ output_ptr,
        int64_t output_stride_m, int64_t output_stride_n, int output_size_m,
        int output_size_n, DTYPE* __restrict__ block_output_ptr,
        int64_t block_output_stride_m, int64_t block_output_stride_n,
        int block_output_size_m, int block_output_size_n,
        DTYPE* const* __restrict__ scratch_ptrs, int64_t scratch_stride_m,
        int64_t scratch_stride_n, int scratch_size_m, int scratch_size_n,
        uint32_t* const* __restrict__ signals_ptrs,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks, int num_blocks,
        float eps, float output_eps) {
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  Sync<WORLD> sync{signals_ptrs,
                   signals_stride_parity,
                   signals_stride_block,
                   signals_stride_writer,
                   status_ptr,
                   status_stride_field,
                   rank,
                   timeout_ticks};
  constexpr int NL = traits<DTYPE>::N;
  using Row = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  // A REDUCE-SCATTER TILE: a row of threads as wide as a rank's slice of a
  // TILE_N row (in whole waves), a group a thread, so a row's slice is one
  // round trip (a wave a row took two at 7168, 0.8 us at 16-32 tokens), and as
  // many rows at once as the block holds. A group a thread, not a slice's
  // worth: every peer's groups of a whole slice were 32 packs at 16384,
  // spilling.
  constexpr int kSliceWaves =
      (TILE_N / WORLD / NL + kWaveSize - 1) / kWaveSize * kWaveSize;
  constexpr int SLICE_LANES =
      kSliceWaves < THREADS_PER_BLOCK ? kSliceWaves : THREADS_PER_BLOCK;
  using Slice =
      Tile<DTYPE, THREADS_PER_BLOCK / SLICE_LANES, SLICE_LANES * NL,
           THREADS_PER_BLOCK / SLICE_LANES, SLICE_LANES, THREADS_PER_BLOCK>;
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);
  const int packs = input_size_n / NL;  // the row, in packs (n is whole packs)
  const int slice = (packs + WORLD - 1) / WORLD;
  const int col0 = rank * slice;
  const int own_packs =
      max(0, min(slice, packs - col0));  // the last rank's may be short
  const int my_rows =
      input_size_m > static_cast<int>(blockIdx.x)
          ? (input_size_m - blockIdx.x + gridDim.x - 1) / gridDim.x
          : 0;

  // 1. Wait until every peer has launched, so its input is ready.
  block_stamp(0);
  barrier<Until::launched>(sync);
  block_stamp(1);

  // THE RANKS' POINTERS AFTER THE BARRIER, as in the pull kernels (held across
  // it they spilled).
  const auto input =
      rank_ptrs<const DTYPE, WORLD>(input_ptrs, input_stride_m, input_stride_n);
  const auto scratch =
      rank_ptrs<DTYPE, WORLD>(scratch_ptrs, scratch_stride_m, scratch_stride_n);

  // 2. This rank's columns of this block's rows, summed over the ranks in rank
  // order and pushed
  //    to every rank (itself too), at their place in the tensor: tiles of this
  //    block's rows (every gridDim.x-th), A WAVE A ROW, since a rank's columns
  //    are a narrow slice.
  if (own_packs > 0) {
    for (int q = 0; q < my_rows; q += Slice::kThreadsM)
      for (int c = col0 * NL; c < (col0 + own_packs) * NL; c += Slice::kTileN) {
        const Slice at{input_size_m, (col0 + own_packs) * NL,
                       static_cast<int>(blockIdx.x + q * gridDim.x), c,
                       static_cast<int>(gridDim.x)};
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

  // 3. Every rank's sums are in this rank's scratch, and every peer has read
  // this rank's input.
  barrier<Until::visible>(sync);
  block_stamp(3);

  // 4. This block's tiles (one row): the sum out of this rank's scratch, then
  // AttnRes, as the
  //    one-shot does. The next call's first sync keeps a peer from pushing into
  //    this scratch while it is read (a peer's next kernel starts only once
  //    this one has finished).
  const auto own_scratch = rank_ptr<DTYPE, WORLD>(
      scratch_ptrs, rank, scratch_stride_m, scratch_stride_n);
  const auto prefix =
      local_ptr(prefix_ptr, prefix_stride_m, prefix_stride_n, rank);
  const auto prefix_output = local_ptr(
      prefix_output_ptr, prefix_output_stride_m, prefix_output_stride_n, rank);
  const auto block_output = local_ptr(block_output_ptr, block_output_stride_m,
                                      block_output_stride_n, rank);
  const auto norm_weight =
      local_ptr(norm_weight_ptr, 0, norm_weight_stride_n, rank);
  const auto qk_weight = local_ptr(qk_weight_ptr, 0, qk_weight_stride_n, rank);
  const auto output_norm_weight =
      local_ptr(output_norm_weight_ptr, 0, output_norm_weight_stride_n, rank);
  const auto output =
      local_ptr(output_ptr, output_stride_m, output_stride_n, rank);
  for (int row = blockIdx.x; row < input_size_m; row += gridDim.x) {
    Row sum{input_size_m, input_size_n, row, 0};
    tile_load(sum, own_scratch);
    block_attn_res_tile<HAS_PREFIX, TILE_K>(
        sum, prefix, blocks_ptr, blocks_stride_m, blocks_stride_r, norm_weight,
        qk_weight, output_norm_weight, prefix_output, output, block_output,
        num_blocks, eps, output_eps, inv_hidden);
  }
  block_stamp(4);
#else
  __builtin_trap();
#endif
}

}  // namespace hip_custom_communicator
