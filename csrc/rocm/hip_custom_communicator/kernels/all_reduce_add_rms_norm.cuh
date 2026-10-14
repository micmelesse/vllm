// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ALL-REDUCE THEN RMSNORM, AND THEN ADD THEN RMSNORM: their kernels, one section each (pull one-shot, pull two-shot,
// push two-shot), each kernel of the RMSNorm op a wrapper over its add-RMSNorm body.

#pragma once

#include "../common/interface.cuh"

namespace hip_custom_communicator {

// =============================================================================
// all_reduce_pull_one_shot_add_rms_norm
//
// One-shot pull all-reduce then RMSNorm (`all_reduce_pull_one_shot_rms_norm`),
// and all-reduce then add then RMSNorm
// (`all_reduce_pull_one_shot_add_rms_norm`): one body, a kernel per op, so a
// trace names the op that ran.
// =============================================================================

// Every rank reduces every row and norms it where the sum lands in registers,
// saving an HBM round trip and a launch against an all-reduce then a norm
// kernel. A block owns a row. `residual` and `residual_out` are unused (null)
// unless kAdd; `weight` keeps its own dtype W (T or fp32), as vLLM's reference
// ops round to the WEIGHT's dtype (`vllm/ir/ops/layernorm.py`). The kernels
// turn their arguments into Ptrs; the peers' inputs become theirs after the
// first barrier.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, bool ADD_RESIDUAL,
          int TILE_N, int THREADS_PER_BLOCK>
DINLINE void all_reduce_pull_one_shot_add_rms_norm_body(
    const DTYPE* const* __restrict__ input_ptrs, int64_t input_stride_m,
    int64_t input_stride_n, uint32_t* const* __restrict__ signals_ptrs,
    int64_t signals_stride_parity, int64_t signals_stride_block,
    int64_t signals_stride_writer, uint32_t* __restrict__ status_ptr,
    int64_t status_stride_field, int rank, uint64_t timeout_ticks,
    Ptr<DTYPE> output, Ptr<DTYPE> residual_output, Ptr<const DTYPE> residual,
    Ptr<const WEIGHT_DTYPE> weight, float eps, int input_size_m,
    int input_size_n) {
  Sync<WORLD> sync{signals_ptrs,
                   signals_stride_parity,
                   signals_stride_block,
                   signals_stride_writer,
                   status_ptr,
                   status_stride_field,
                   rank,
                   timeout_ticks};
  using Row = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  using RowF =
      Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK, float>;
  using Weight = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK,
                      WEIGHT_DTYPE>;
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);

  // 1. Every rank's buffers, then wait until every peer has launched, so its
  // input is ready.
  const auto input =
      rank_ptrs<const DTYPE, WORLD>(input_ptrs, input_stride_m, input_stride_n);
  block_stamp(0);
  barrier<Until::launched>(sync);
  block_stamp(1);

  // 2. Each of this block's rows: read it from every rank in rank order and
  // sum, then (ADD_RESIDUAL) add
  //    the residual, then RMSNorm, rounding as the reference does:
  //      s   = float(DTYPE(sum over ranks))             the all-reduce output,
  //      as it would land s  += float(residual); residual_out = DTYPE(s)
  //      ADD_RESIDUAL only (fused_add_rms_norm) out =
  //      DTYPE(WEIGHT_DTYPE(WEIGHT_DTYPE(s * rsqrt(mean(s^2) + eps)) *
  //      float(w)))
  //    The variance is of `s` before any further rounding, kept in registers
  //    between the passes.
  for (int row = blockIdx.x; row < input_size_m; row += gridDim.x) {
    const Row at{input_size_m, input_size_n, row, 0};
    // Every load of the row before any store, in flight together: the peers',
    // the residual's and the weight's. A load issues where it is written (an
    // address is built at its load), and the weight after the peers' sum was a
    // round trip of its own (+0.34 us at 16 x 3584, 2026-10-04T17-25-35Z).
    Row peers[WORLD];
#pragma unroll
    for (int r = 0; r < WORLD; ++r) peers[r] = at;
    tile_load(peers, input);
    Row res = at;
    if constexpr (ADD_RESIDUAL) tile_load(res, residual);
    Weight w{1, input_size_n, 0, 0};
    tile_load(w, weight);
    RowF s = peers_reduce(peers).template to<float>();
    block_stamp(2);
    if constexpr (ADD_RESIDUAL) {
      s = tile_add(s, res.template to<float>());
      tile_store(s.template to<DTYPE>(), residual_output);
    }
    float ss[1];
    partial_dot(s, s, ss);
    block_reduce<Sum, THREADS_PER_BLOCK>(ss);
    block_stamp(3);
    const float scale = rsqrtf(ss[0] * inv_hidden + eps);
    // out = T(W(W(s * scale) * float(w))), as the reference rounds
    Row normed =
        tile_mul(
            tile_mul(s, scale).template to<WEIGHT_DTYPE>().template to<float>(),
            w.template to<float>())
            .template to<WEIGHT_DTYPE>()
            .template to<DTYPE>();
    tile_store(normed, output);
  }

  block_stamp(4);
  // 3. No rank may overwrite its input until every peer has read it.
  barrier<Until::read>(sync);
  block_stamp(5);
}

// THE KERNELS, one per op, both the body above.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot_rms_norm(
        const DTYPE* const* __restrict__ input_ptrs, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const WEIGHT_DTYPE* __restrict__ weight_ptr, int64_t weight_stride_n,
        int weight_size_n, DTYPE* __restrict__ output_ptr,
        int64_t output_stride_m, int64_t output_stride_n, int output_size_m,
        int output_size_n, uint32_t* const* __restrict__ signals_ptrs,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks, float eps) {
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  all_reduce_pull_one_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, false,
                                             TILE_N, THREADS_PER_BLOCK>(
      input_ptrs, input_stride_m, input_stride_n, signals_ptrs,
      signals_stride_parity, signals_stride_block, signals_stride_writer,
      status_ptr, status_stride_field, rank, timeout_ticks,
      local_ptr(output_ptr, output_stride_m, output_stride_n, rank),
      Ptr<DTYPE>{nullptr, 0, 0, rank}, Ptr<const DTYPE>{nullptr, 0, 0, rank},
      local_ptr(weight_ptr, 0, weight_stride_n, rank), eps, input_size_m,
      input_size_n);
#else
  __builtin_trap();
#endif
}

template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_one_shot_add_rms_norm(
        const DTYPE* const* __restrict__ input_ptrs, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const DTYPE* __restrict__ residual_ptr, int64_t residual_stride_m,
        int64_t residual_stride_n, int residual_size_m, int residual_size_n,
        const WEIGHT_DTYPE* __restrict__ weight_ptr, int64_t weight_stride_n,
        int weight_size_n, DTYPE* __restrict__ output_ptr,
        int64_t output_stride_m, int64_t output_stride_n, int output_size_m,
        int output_size_n, DTYPE* __restrict__ residual_output_ptr,
        int64_t residual_output_stride_m, int64_t residual_output_stride_n,
        int residual_output_size_m, int residual_output_size_n,
        uint32_t* const* __restrict__ signals_ptrs,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks, float eps) {
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  all_reduce_pull_one_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, true,
                                             TILE_N, THREADS_PER_BLOCK>(
      input_ptrs, input_stride_m, input_stride_n, signals_ptrs,
      signals_stride_parity, signals_stride_block, signals_stride_writer,
      status_ptr, status_stride_field, rank, timeout_ticks,
      local_ptr(output_ptr, output_stride_m, output_stride_n, rank),
      local_ptr(residual_output_ptr, residual_output_stride_m,
                residual_output_stride_n, rank),
      local_ptr(residual_ptr, residual_stride_m, residual_stride_n, rank),
      local_ptr(weight_ptr, 0, weight_stride_n, rank), eps, input_size_m,
      input_size_n);
#else
  __builtin_trap();
#endif
}

// =============================================================================
// all_reduce_pull_two_shot_add_rms_norm
//
// Two-shot pull all-reduce then RMSNorm (`all_reduce_pull_two_shot_rms_norm`),
// and all-reduce then add then RMSNorm
// (`all_reduce_pull_two_shot_add_rms_norm`): one body, a kernel per op, so a
// trace names the op that ran. THE SLICE IS ROWS, for large inputs: the
// column-slice kernel (all_reduce_push_two_shot_*) wins small ones and loses at
// prefill.
// =============================================================================

// THE SLICE IS ROWS: each rank owns whole rows, so it computes each row's norm
// alone. It reduces its rows and leaves one row each in its scratch, row-major:
// the normed row, or (kAdd) the new residual and the row's RMS scale; after the
// sync every rank copies every owner's rows out, (kAdd) norming them by their
// scales as it goes. Every rank does the same arithmetic on the same bytes, so
// every rank holds the same result. THE SAME BLOCK AND THREAD INDEX A PACK IN
// BOTH PHASES: after the sync a block may read only what the same block on a
// peer wrote.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, bool ADD_RESIDUAL,
          int TILE_N, int THREADS_PER_BLOCK>
DINLINE void all_reduce_pull_two_shot_add_rms_norm_body(
    const DTYPE* const* __restrict__ input_ptrs, int64_t input_stride_m,
    int64_t input_stride_n, DTYPE* const* __restrict__ scratch_ptrs,
    int64_t scratch_stride_m, int64_t scratch_stride_n,
    uint32_t* const* __restrict__ signals_ptrs, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    uint32_t* __restrict__ status_ptr, int64_t status_stride_field, int rank,
    uint64_t timeout_ticks, Ptr<DTYPE> output, Ptr<DTYPE> residual_output,
    Ptr<const DTYPE> residual, Ptr<const WEIGHT_DTYPE> weight, float eps,
    int input_size_m, int input_size_n) {
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
  using RowF =
      Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK, float>;
  using Weight = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK,
                      WEIGHT_DTYPE>;
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);
  const int slice_m = (input_size_m + WORLD - 1) / WORLD;
  // ADD_RESIDUAL: each owned row's RMS scale, a float a row, after the rows in
  // scratch.
  const int64_t scale_at = int64_t{slice_m} * scratch_stride_m;  // in elements
  const auto scales = [&](DTYPE* at) {
    return reinterpret_cast<float*>(at + scale_at);
  };

  // 1. Wait until every peer has launched, so its input is ready.
  block_stamp(0);
  barrier<Until::launched>(sync);
  block_stamp(1);

  // THE RANKS' POINTERS AFTER THE BARRIER here: held across it, the 8-pack
  // build keeps 68 B of scratch (the ISA gate, 2026-09-30).
  const auto input =
      rank_ptrs<const DTYPE, WORLD>(input_ptrs, input_stride_m, input_stride_n);
  const auto scratch =
      rank_ptrs<DTYPE, WORLD>(scratch_ptrs, scratch_stride_m, scratch_stride_n);
  const auto own_scratch = rank_ptr<DTYPE, WORLD>(
      scratch_ptrs, rank, scratch_stride_m, scratch_stride_n);

  // 2. This rank's rows: read each from every rank in rank order and sum, then
  // (ADD_RESIDUAL) add the
  //    residual, then RMSNorm, rounding as the reference does (the one-shot
  //    kernel spells it out), and leave the normed rows or (ADD_RESIDUAL) the
  //    residual rows and scales in this rank's scratch. PIPELINED: the next
  //    row's loads go out before this row's reduction and norm, so a block's
  //    compute runs under its next round trip instead of between them (a block
  //    had ~14 rows at 4096 tokens, each 1.24 us of reduction and norm on the
  //    critical path: 2026-10-01T00-01-54Z).
  const int first = rank * slice_m;
  const int last = min(first + slice_m, input_size_m);
  // Row `row` from every rank.
  using Peers = Row[WORLD];
  const auto load = [&](int row, Peers& got) {
#pragma unroll
    for (int r = 0; r < WORLD; ++r)
      got[r] = Row{input_size_m, input_size_n, row, 0};
    tile_load(got, input);
  };
  // THE WEIGHT ONCE, AND EVERY OTHER LOAD BEFORE THE NEXT ROW'S: loads complete
  // in issue order, so waiting on one issued after the peers' would wait on the
  // peers' too.
  Weight w{1, input_size_n, 0, 0};
  tile_load(w, weight);
  // ONE ROW: its residual, then the next row's peer loads into `next`, then
  // this row's sum (its wait covers only its own, older, loads), so the next
  // round trip runs under the reduction and norm. Its rows land in this rank's
  // scratch at row - first.
  const auto one_row = [&](int row, const Peers& cur, Peers& next) {
    Row res{input_size_m, input_size_n, row, 0};
    if constexpr (ADD_RESIDUAL) tile_load(res, residual);
    // ONLY A ROW THAT EXISTS: issued here, never hoisted, so the block-uniform
    // branch costs nothing, where a clamped unconditional load re-read the last
    // row (a block's whole round trip again; at 256 tokens every block has one
    // row: 2026-10-01T01-07-56Z).
    if (row + static_cast<int>(gridDim.x) < last) load(row + gridDim.x, next);
    RowF s = peers_reduce(cur).template to<float>();
    block_stamp(2);
    // The norm, rounding as the reference does (see the one-shot kernel):
    if constexpr (ADD_RESIDUAL) {
      s = tile_add(s, res.template to<float>());
      Row added = s.template to<DTYPE>();
      added.offs_m = row - first;
      tile_store(added, own_scratch);
    }
    float ss[1];
    partial_dot(s, s, ss);
    block_reduce<Sum, THREADS_PER_BLOCK>(ss);
    block_stamp(3);
    const float scale = rsqrtf(ss[0] * inv_hidden + eps);
    // ADD_RESIDUAL LEAVES THE NEW RESIDUAL AND ITS SCALE, NOT THE NORMED ROW:
    // every rank norms it while gathering, so the link carries one row a row,
    // as a plain all-reduce does (gathering both the normed row and the
    // residual was twice that: 215.9 against 155.2 us at 4096 tokens,
    // 2026-10-01T02-17-54Z).
    if constexpr (ADD_RESIDUAL) {
      block_store_row_scalar(scales(own_scratch.data), row - first, scale);
      return;
    }
    // out = T(W(W(s * scale) * float(w))), as the reference rounds
    Row normed =
        tile_mul(
            tile_mul(s, scale).template to<WEIGHT_DTYPE>().template to<float>(),
            w.template to<float>())
            .template to<WEIGHT_DTYPE>()
            .template to<DTYPE>();
    normed.offs_m = row - first;
    tile_store(normed, own_scratch);
  };
  // PING-PONG: two buffers that trade roles each row, so no row copies its
  // packs into the other (a copy cost 32 moves a row at one pack a thread: ISA
  // 2026-10-01T00-58-37Z).
  Peers a, b;
  int row = first + blockIdx.x;
  if (row < last) load(row, a);
  for (; row < last; row += 2 * gridDim.x) {
    one_row(row, a, b);
    if (row + static_cast<int>(gridDim.x) >= last) break;
    one_row(row + gridDim.x, b, a);
  }

  block_stamp(4);
  // 3. Every rank's rows are visible to its peers.
  barrier<Until::visible>(sync);
  block_stamp(5);

  // 4. Every owner's rows out of its scratch, at their place in the output. The
  // next call's
  //    first sync keeps a rank from overwriting its scratch while it is read.
  //    EVERY OWNER'S PACK LOADED BEFORE ANY IS STORED, and every load
  //    unconditional (each rank's scratch holds slice_m rows, so a slot past
  //    the last row is real): a store between two loads, or a load under an
  //    `if`, made the eight owners' round trips run one after another.
  //    ADD_RESIDUAL: the owners' rows are the new residual; each is normed here
  //    by its owner's scale (the weight loaded once, before the loop). Every
  //    thread loads the 8 scales after the 8 tiles, all in flight before any
  //    wait (a wave's lanes read one scale address: one request a wave). Left
  //    to the compiler, the scales were loaded and waited on before the packs
  //    were issued, two round trips a row (ISA 2026-10-01T02-33-11Z).
  // A CHUNK A GROUP A THREAD, stepping across the row: every owner's chunk in
  // flight is WORLD packs a thread, as the pack-at-a-time copy was (a whole
  // row's was 16 at 7168, and 2-3% slower).
  using Chunk = Tile<DTYPE, 1, THREADS_PER_BLOCK * NL, 1, THREADS_PER_BLOCK,
                     THREADS_PER_BLOCK>;
  using ChunkW = TileAs<Chunk, WEIGHT_DTYPE>;
  for (int l = blockIdx.x; l < slice_m; l += gridDim.x) {
    float sc[WORLD];
    if constexpr (ADD_RESIDUAL)
      peers_load_row_scalars<WORLD>(
          [&](int r) { return scales(scratch[r].data); }, l, sc);
    for (int c = 0; c < input_size_n; c += Chunk::kTileN) {
      Chunk got[WORLD];
#pragma unroll
      for (int r = 0; r < WORLD; ++r)
        got[r] = Chunk{slice_m, input_size_n, l, c};
      tile_load(got, scratch);
      ChunkW wc{1, input_size_n, 0, c};
      if constexpr (ADD_RESIDUAL) tile_load(wc, weight);
#pragma unroll
      for (int r = 0; r < WORLD; ++r) {
        const int row = r * slice_m + l;
        if (row >= input_size_m) continue;
        Chunk at = got[r];
        at.M = input_size_m;
        at.offs_m = row;
        if constexpr (ADD_RESIDUAL) {
          tile_store(at, residual_output);
          const Chunk normed = tile_mul(tile_mul(at.template to<float>(), sc[r])
                                            .template to<WEIGHT_DTYPE>()
                                            .template to<float>(),
                                        wc.template to<float>())
                                   .template to<WEIGHT_DTYPE>()
                                   .template to<DTYPE>();
          tile_store(normed, output);
        } else {
          tile_store(at, output);
        }
      }
    }
  }
  block_stamp(6);
}

// THE KERNELS, one per op, both the body above.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot_rms_norm(
        const DTYPE* const* __restrict__ input_ptrs, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const WEIGHT_DTYPE* __restrict__ weight_ptr, int64_t weight_stride_n,
        int weight_size_n, DTYPE* __restrict__ output_ptr,
        int64_t output_stride_m, int64_t output_stride_n, int output_size_m,
        int output_size_n, DTYPE* const* __restrict__ scratch_ptrs,
        int64_t scratch_stride_m, int64_t scratch_stride_n, int scratch_size_m,
        int scratch_size_n, uint32_t* const* __restrict__ signals_ptrs,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks, float eps) {
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  all_reduce_pull_two_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, false,
                                             TILE_N, THREADS_PER_BLOCK>(
      input_ptrs, input_stride_m, input_stride_n, scratch_ptrs,
      scratch_stride_m, scratch_stride_n, signals_ptrs, signals_stride_parity,
      signals_stride_block, signals_stride_writer, status_ptr,
      status_stride_field, rank, timeout_ticks,
      local_ptr(output_ptr, output_stride_m, output_stride_n, rank),
      Ptr<DTYPE>{nullptr, 0, 0, rank}, Ptr<const DTYPE>{nullptr, 0, 0, rank},
      local_ptr(weight_ptr, 0, weight_stride_n, rank), eps, input_size_m,
      input_size_n);
#else
  __builtin_trap();
#endif
}

template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot_add_rms_norm(
        const DTYPE* const* __restrict__ input_ptrs, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const DTYPE* __restrict__ residual_ptr, int64_t residual_stride_m,
        int64_t residual_stride_n, int residual_size_m, int residual_size_n,
        const WEIGHT_DTYPE* __restrict__ weight_ptr, int64_t weight_stride_n,
        int weight_size_n, DTYPE* __restrict__ output_ptr,
        int64_t output_stride_m, int64_t output_stride_n, int output_size_m,
        int output_size_n, DTYPE* __restrict__ residual_output_ptr,
        int64_t residual_output_stride_m, int64_t residual_output_stride_n,
        int residual_output_size_m, int residual_output_size_n,
        DTYPE* const* __restrict__ scratch_ptrs, int64_t scratch_stride_m,
        int64_t scratch_stride_n, int scratch_size_m, int scratch_size_n,
        uint32_t* const* __restrict__ signals_ptrs,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks, float eps) {
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  all_reduce_pull_two_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, true,
                                             TILE_N, THREADS_PER_BLOCK>(
      input_ptrs, input_stride_m, input_stride_n, scratch_ptrs,
      scratch_stride_m, scratch_stride_n, signals_ptrs, signals_stride_parity,
      signals_stride_block, signals_stride_writer, status_ptr,
      status_stride_field, rank, timeout_ticks,
      local_ptr(output_ptr, output_stride_m, output_stride_n, rank),
      local_ptr(residual_output_ptr, residual_output_stride_m,
                residual_output_stride_n, rank),
      local_ptr(residual_ptr, residual_stride_m, residual_stride_n, rank),
      local_ptr(weight_ptr, 0, weight_stride_n, rank), eps, input_size_m,
      input_size_n);
#else
  __builtin_trap();
#endif
}

// =============================================================================
// all_reduce_push_two_shot_add_rms_norm
//
// Two-shot all-reduce (the reduce-scatter pulled, the all-gather pushed) then
// RMSNorm
// (`all_reduce_push_two_shot_rms_norm`), and all-reduce then add then RMSNorm
// (`all_reduce_push_two_shot_add_rms_norm`): one body, a kernel per op, so a
// trace names the op that ran.
// =============================================================================

// THE SLICE IS COLUMNS, as in aiter's fused two-stage kernel: each rank owns
// columns [rank * S, rank * S + S) of EVERY row (S = n / ngpus, rounded up to
// whole packs). It sums its columns of a row over the ranks and pushes the sums
// into every rank's scratch, so after the sync each rank holds the whole
// all-reduced tensor locally and norms its rows itself: no remote gather, and
// the norm repeated on every rank instead of sent. Every rank computes the same
// bytes (the same sums in the same order), so every rank holds the same result.
// ROW q BELONGS TO BLOCK q % gridDim.x IN BOTH PHASES: after the sync a block
// may read only what the same block on a peer wrote.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, bool ADD_RESIDUAL,
          int TILE_N, int THREADS_PER_BLOCK>
DINLINE void all_reduce_push_two_shot_add_rms_norm_body(
    const DTYPE* const* __restrict__ input_ptrs, int64_t input_stride_m,
    int64_t input_stride_n, DTYPE* const* __restrict__ scratch_ptrs,
    int64_t scratch_stride_m, int64_t scratch_stride_n,
    uint32_t* const* __restrict__ signals_ptrs, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    uint32_t* __restrict__ status_ptr, int64_t status_stride_field, int rank,
    uint64_t timeout_ticks, Ptr<DTYPE> output, Ptr<DTYPE> residual_output,
    Ptr<const DTYPE> residual, Ptr<const WEIGHT_DTYPE> weight, float eps,
    int input_size_m, int input_size_n) {
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
  using RowF =
      Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK, float>;
  using Weight = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK,
                      WEIGHT_DTYPE>;
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);
  const int slice =
      (input_size_n / NL + WORLD - 1) / WORLD * NL;  // a rank's, in elements
  const int col0 = rank * slice;
  const int own_n =
      max(0, min(slice, input_size_n - col0));  // the last rank's may be short
  const int my_m = input_size_m > static_cast<int>(blockIdx.x)
                       ? (input_size_m - blockIdx.x + gridDim.x - 1) / gridDim.x
                       : 0;
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
  if (own_n > 0) {
    for (int q = 0; q < my_m; q += Slice::kThreadsM)
      for (int c = col0; c < col0 + own_n; c += Slice::kTileN) {
        const Slice at{input_size_m, col0 + own_n,
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

  // 3. Every rank's sums are in this rank's scratch.
  barrier<Until::visible>(sync);
  block_stamp(3);

  // 4. This block's rows out of this rank's scratch: (ADD_RESIDUAL) add the
  // residual, then RMSNorm,
  //    rounding as the reference does (the one-shot kernel spells it out). The
  //    next call's first sync keeps a peer from pushing into this scratch while
  //    it is read (a peer's next kernel starts only once this one has
  //    finished).
  const auto own_scratch = rank_ptr<DTYPE, WORLD>(
      scratch_ptrs, rank, scratch_stride_m, scratch_stride_n);
  for (int row = blockIdx.x; row < input_size_m; row += gridDim.x) {
    const Row at{input_size_m, input_size_n, row, 0};
    // Every load of the row before any store, in flight together: the
    // scratch's, the residual's and the weight's (one-shot's).
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
    Row normed =
        tile_mul(
            tile_mul(s, scale).template to<WEIGHT_DTYPE>().template to<float>(),
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
    all_reduce_push_two_shot_rms_norm(
        const DTYPE* const* __restrict__ input_ptrs, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const WEIGHT_DTYPE* __restrict__ weight_ptr, int64_t weight_stride_n,
        int weight_size_n, DTYPE* __restrict__ output_ptr,
        int64_t output_stride_m, int64_t output_stride_n, int output_size_m,
        int output_size_n, DTYPE* const* __restrict__ scratch_ptrs,
        int64_t scratch_stride_m, int64_t scratch_stride_n, int scratch_size_m,
        int scratch_size_n, uint32_t* const* __restrict__ signals_ptrs,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks, float eps) {
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  all_reduce_push_two_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, false,
                                             TILE_N, THREADS_PER_BLOCK>(
      input_ptrs, input_stride_m, input_stride_n, scratch_ptrs,
      scratch_stride_m, scratch_stride_n, signals_ptrs, signals_stride_parity,
      signals_stride_block, signals_stride_writer, status_ptr,
      status_stride_field, rank, timeout_ticks,
      local_ptr(output_ptr, output_stride_m, output_stride_n, rank),
      Ptr<DTYPE>{nullptr, 0, 0, rank}, Ptr<const DTYPE>{nullptr, 0, 0, rank},
      local_ptr(weight_ptr, 0, weight_stride_n, rank), eps, input_size_m,
      input_size_n);
#else
  __builtin_trap();
#endif
}

template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_push_two_shot_add_rms_norm(
        const DTYPE* const* __restrict__ input_ptrs, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const DTYPE* __restrict__ residual_ptr, int64_t residual_stride_m,
        int64_t residual_stride_n, int residual_size_m, int residual_size_n,
        const WEIGHT_DTYPE* __restrict__ weight_ptr, int64_t weight_stride_n,
        int weight_size_n, DTYPE* __restrict__ output_ptr,
        int64_t output_stride_m, int64_t output_stride_n, int output_size_m,
        int output_size_n, DTYPE* __restrict__ residual_output_ptr,
        int64_t residual_output_stride_m, int64_t residual_output_stride_n,
        int residual_output_size_m, int residual_output_size_n,
        DTYPE* const* __restrict__ scratch_ptrs, int64_t scratch_stride_m,
        int64_t scratch_stride_n, int scratch_size_m, int scratch_size_n,
        uint32_t* const* __restrict__ signals_ptrs,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks, float eps) {
#if HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED
  all_reduce_push_two_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, true,
                                             TILE_N, THREADS_PER_BLOCK>(
      input_ptrs, input_stride_m, input_stride_n, scratch_ptrs,
      scratch_stride_m, scratch_stride_n, signals_ptrs, signals_stride_parity,
      signals_stride_block, signals_stride_writer, status_ptr,
      status_stride_field, rank, timeout_ticks,
      local_ptr(output_ptr, output_stride_m, output_stride_n, rank),
      local_ptr(residual_output_ptr, residual_output_stride_m,
                residual_output_stride_n, rank),
      local_ptr(residual_ptr, residual_stride_m, residual_stride_n, rank),
      local_ptr(weight_ptr, 0, weight_stride_n, rank), eps, input_size_m,
      input_size_n);
#else
  __builtin_trap();
#endif
}

}  // namespace hip_custom_communicator
