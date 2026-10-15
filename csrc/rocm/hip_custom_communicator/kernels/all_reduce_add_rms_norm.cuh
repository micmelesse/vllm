// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ALL-REDUCE THEN RMSNORM, AND THEN ADD THEN RMSNORM: their kernels, one
// section each (pull one-shot, pull two-shot, push two-shot), each kernel of
// the RMSNorm op a wrapper over its add-RMSNorm body.

#pragma once

#include "common.cuh"

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
// kernel. A block owns a row, held as `T v[1][K][P]` (a row a block: one row
// of threads, THREADS_PER_BLOCK lanes, K packs a lane). `residual_ptr` and
// `residual_output_ptr` are unused (null) unless ADD_RESIDUAL; the weight keeps
// its own dtype WEIGHT_DTYPE (DTYPE or fp32), as vLLM's reference ops round to
// the WEIGHT's dtype (`vllm/ir/ops/layernorm.py`). THE ROW'S PLACE IS COUNTED
// IN THE INPUT'S SIZES for every buffer, as before the port (the host checks
// they are equal): each buffer's own sizes are a change of their own, compiled
// and measured. The peers' inputs are theirs after the first barrier.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, bool ADD_RESIDUAL,
          int TILE_N, int THREADS_PER_BLOCK>
DINLINE void all_reduce_pull_one_shot_add_rms_norm_body(
    const DTYPE* const* __restrict__ input_ptrs, int64_t input_stride_m,
    int64_t input_stride_n, int input_size_m, int input_size_n,
    const DTYPE* residual_ptr, int64_t residual_stride_m,
    int64_t residual_stride_n, const WEIGHT_DTYPE* weight_ptr,
    int64_t weight_stride_n, DTYPE* output_ptr, int64_t output_stride_m,
    int64_t output_stride_n, DTYPE* residual_output_ptr,
    int64_t residual_output_stride_m, int64_t residual_output_stride_n,
    uint32_t* const* __restrict__ signals_ptrs, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
    int status_size_field, int rank, uint64_t timeout_ticks, float eps) {
  constexpr int P = traits<DTYPE>::N;
  constexpr int K = TILE_N / (P * THREADS_PER_BLOCK);
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);

  // 1. Every rank's input pointer, then wait until every peer has launched, so
  // its input is ready. THE TABLE READ ONCE, BEFORE THE BARRIER: read by
  // peers_load in the loop, after the barrier's atomics, the compiler could
  // not prove it unchanged and loaded it again every row, 4 vector loads a row
  // where it was one scalar load a kernel (ISA 2026-10-15T18-32-04Z).
  const DTYPE* input[WORLD];
  peers_ptrs<WORLD>(input_ptrs, input);
  barrier<Until::launched, 0, WORLD>(
      signals_ptrs, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);

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
    // Every load of the row before any store, in flight together: the peers',
    // the residual's and the weight's. A load issues where it is written (an
    // address is built at its load), and the weight after the peers' sum was a
    // round trip of its own (+0.34 us at 16 x 3584, 2026-10-04T17-25-35Z).
    DTYPE peers[WORLD][1][K][P];
    peers_load<1, THREADS_PER_BLOCK>(input, input_stride_m, input_stride_n,
                                     input_size_m, input_size_n, peers, row,
                                     0);
    DTYPE res[1][K][P];
    if constexpr (ADD_RESIDUAL)
      load<1, THREADS_PER_BLOCK>(residual_ptr, residual_stride_m,
                                 residual_stride_n, input_size_m, input_size_n,
                                 res, row, 0);
    WEIGHT_DTYPE w[1][K][P];
    load<1, THREADS_PER_BLOCK>(weight_ptr, 0, weight_stride_n, 1, input_size_n,
                               w, 0, 0);
    float s[1][K][P];
    peers_sum(peers, s);
    round<DTYPE>(s);
    if constexpr (ADD_RESIDUAL) {
      float res_f[1][K][P];
      convert(res, res_f);
      add(s, res_f, s);
      DTYPE added[1][K][P];
      convert(s, added);
      store<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>(added, residual_output_ptr,
                                  residual_output_stride_m,
                                  residual_output_stride_n, input_size_m,
                                  input_size_n, row, 0);
    }
    float ss[1];
    partial_dot<THREADS_PER_BLOCK>(s, s, ss, input_size_n, 0);
    block_reduce<Sum, THREADS_PER_BLOCK>(ss);
    const float scale = rsqrtf(ss[0] * inv_hidden + eps);
    // out = T(W(W(s * scale) * float(w))), as the reference rounds
    float normed_f[1][K][P];
    mul(s, scale, normed_f);
    round<WEIGHT_DTYPE>(normed_f);
    float w_f[1][K][P];
    convert(w, w_f);
    mul(normed_f, w_f, normed_f);
    WEIGHT_DTYPE normed_w[1][K][P];
    convert(normed_f, normed_w);
    DTYPE normed[1][K][P];
    convert(normed_w, normed);
    store<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>(normed, output_ptr, output_stride_m,
                                output_stride_n, input_size_m, input_size_n,
                                row, 0);
  }

  // 3. No rank may overwrite its input until every peer has read it.
  barrier<Until::read, 1, WORLD>(
      signals_ptrs, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);
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
      input_ptrs, input_stride_m, input_stride_n, input_size_m, input_size_n,
      nullptr, 0, 0, weight_ptr, weight_stride_n, output_ptr, output_stride_m,
      output_stride_n, nullptr, 0, 0, signals_ptrs, signals_stride_parity,
      signals_stride_block, signals_stride_writer, signals_size_parity,
      signals_size_block, signals_size_writer, status_ptr, status_stride_field,
      status_size_field, rank, timeout_ticks, eps);
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
      input_ptrs, input_stride_m, input_stride_n, input_size_m, input_size_n,
      residual_ptr, residual_stride_m, residual_stride_n, weight_ptr,
      weight_stride_n, output_ptr, output_stride_m, output_stride_n,
      residual_output_ptr, residual_output_stride_m, residual_output_stride_n,
      signals_ptrs, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks, eps);
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
// the normed row, or (ADD_RESIDUAL) the new residual and the row's RMS scale;
// after the sync every rank copies every owner's rows out, (ADD_RESIDUAL)
// norming them by their scales as it goes. Every rank does the same arithmetic
// on the same bytes, so every rank holds the same result. THE SAME BLOCK AND
// THREAD INDEX A PACK IN BOTH PHASES: after the sync a block may read only what
// the same block on a peer wrote. The buffers' places are counted as the
// one-shot's are (its head says why).
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, bool ADD_RESIDUAL,
          int TILE_N, int THREADS_PER_BLOCK>
DINLINE void all_reduce_pull_two_shot_add_rms_norm_body(
    const DTYPE* const* __restrict__ input_ptrs, int64_t input_stride_m,
    int64_t input_stride_n, int input_size_m, int input_size_n,
    const DTYPE* residual_ptr, int64_t residual_stride_m,
    int64_t residual_stride_n, const WEIGHT_DTYPE* weight_ptr,
    int64_t weight_stride_n, DTYPE* output_ptr, int64_t output_stride_m,
    int64_t output_stride_n, DTYPE* residual_output_ptr,
    int64_t residual_output_stride_m, int64_t residual_output_stride_n,
    DTYPE* const* __restrict__ scratch_ptrs, int64_t scratch_stride_m,
    int64_t scratch_stride_n, uint32_t* const* __restrict__ signals_ptrs,
    int64_t signals_stride_parity, int64_t signals_stride_block,
    int64_t signals_stride_writer, int signals_size_parity,
    int signals_size_block, int signals_size_writer,
    uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
    int status_size_field, int rank, uint64_t timeout_ticks, float eps) {
  constexpr int P = traits<DTYPE>::N;
  constexpr int K = TILE_N / (P * THREADS_PER_BLOCK);
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);
  const int slice_m = (input_size_m + WORLD - 1) / WORLD;
  // ADD_RESIDUAL: each owned row's RMS scale, a float a row, after the rows in
  // scratch.
  const int64_t scale_at = int64_t{slice_m} * scratch_stride_m;  // in elements

  // 1. Wait until every peer has launched, so its input is ready.
  barrier<Until::launched, 0, WORLD>(
      signals_ptrs, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);

  // THE RANKS' POINTERS AFTER THE BARRIER here: held across it, the 8-pack
  // build keeps 68 B of scratch (the ISA gate, 2026-09-30). Read once, out of
  // the loops: peers_load reading its table every row loaded it again each
  // time (the one-shot's head).
  const DTYPE* input[WORLD];
  peers_ptrs<WORLD>(input_ptrs, input);
  DTYPE* scratch[WORLD];
  peers_ptrs<WORLD>(scratch_ptrs, scratch);
  DTYPE* const own_scratch = peer_ptr(scratch_ptrs, rank);

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
  const auto load_peers = [&](int row, DTYPE(&got)[WORLD][1][K][P]) {
    peers_load<1, THREADS_PER_BLOCK>(input, input_stride_m, input_stride_n,
                                     input_size_m, input_size_n, got, row, 0);
  };
  // THE WEIGHT ONCE, AND EVERY OTHER LOAD BEFORE THE NEXT ROW'S: loads complete
  // in issue order, so waiting on one issued after the peers' would wait on the
  // peers' too.
  WEIGHT_DTYPE w[1][K][P];
  load<1, THREADS_PER_BLOCK>(weight_ptr, 0, weight_stride_n, 1, input_size_n,
                             w, 0, 0);
  // ONE ROW: its residual, then the next row's peer loads into `next`, then
  // this row's sum (its wait covers only its own, older, loads), so the next
  // round trip runs under the reduction and norm. Its rows land in this rank's
  // scratch at row - first.
  const auto one_row = [&](int row, const DTYPE(&cur)[WORLD][1][K][P],
                           DTYPE(&next)[WORLD][1][K][P]) {
    DTYPE res[1][K][P];
    if constexpr (ADD_RESIDUAL)
      load<1, THREADS_PER_BLOCK>(residual_ptr, residual_stride_m,
                                 residual_stride_n, input_size_m, input_size_n,
                                 res, row, 0);
    // ONLY A ROW THAT EXISTS: issued here, never hoisted, so the block-uniform
    // branch costs nothing, where a clamped unconditional load re-read the last
    // row (a block's whole round trip again; at 256 tokens every block has one
    // row: 2026-10-01T01-07-56Z).
    if (row + static_cast<int>(gridDim.x) < last)
      load_peers(row + gridDim.x, next);
    float s[1][K][P];
    peers_sum(cur, s);
    round<DTYPE>(s);
    // The norm, rounding as the reference does (see the one-shot kernel):
    if constexpr (ADD_RESIDUAL) {
      float res_f[1][K][P];
      convert(res, res_f);
      add(s, res_f, s);
      DTYPE added[1][K][P];
      convert(s, added);
      store<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>(added, own_scratch, scratch_stride_m,
                                  scratch_stride_n, input_size_m, input_size_n,
                                  row - first, 0);
    }
    float ss[1];
    partial_dot<THREADS_PER_BLOCK>(s, s, ss, input_size_n, 0);
    block_reduce<Sum, THREADS_PER_BLOCK>(ss);
    const float scale = rsqrtf(ss[0] * inv_hidden + eps);
    // ADD_RESIDUAL LEAVES THE NEW RESIDUAL AND ITS SCALE, NOT THE NORMED ROW:
    // every rank norms it while gathering, so the link carries one row a row,
    // as a plain all-reduce does (gathering both the normed row and the
    // residual was twice that: 215.9 against 155.2 us at 4096 tokens,
    // 2026-10-01T02-17-54Z).
    if constexpr (ADD_RESIDUAL) {
      store_row_scalar(scale, reinterpret_cast<float*>(own_scratch + scale_at),
                       row - first);
      return;
    }
    // out = T(W(W(s * scale) * float(w))), as the reference rounds
    float normed_f[1][K][P];
    mul(s, scale, normed_f);
    round<WEIGHT_DTYPE>(normed_f);
    float w_f[1][K][P];
    convert(w, w_f);
    mul(normed_f, w_f, normed_f);
    WEIGHT_DTYPE normed_w[1][K][P];
    convert(normed_f, normed_w);
    DTYPE normed[1][K][P];
    convert(normed_w, normed);
    store<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>(normed, own_scratch, scratch_stride_m,
                                scratch_stride_n, input_size_m, input_size_n,
                                row - first, 0);
  };
  // PING-PONG: two buffers that trade roles each row, so no row copies its
  // packs into the other (a copy cost 32 moves a row at one pack a thread: ISA
  // 2026-10-01T00-58-37Z).
  DTYPE a[WORLD][1][K][P], b[WORLD][1][K][P];
  int row = first + blockIdx.x;
  if (row < last) load_peers(row, a);
  for (; row < last; row += 2 * gridDim.x) {
    one_row(row, a, b);
    if (row + static_cast<int>(gridDim.x) >= last) break;
    one_row(row + gridDim.x, b, a);
  }

  // 3. Every rank's rows are visible to its peers.
  barrier<Until::visible, 1, WORLD>(
      signals_ptrs, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);

  // 4. Every owner's rows out of its scratch, at their place in the output. The
  // next call's
  //    first sync keeps a rank from overwriting its scratch while it is read.
  //    EVERY OWNER'S PACK LOADED BEFORE ANY IS STORED, and every load
  //    unconditional (each rank's scratch holds slice_m rows, so a slot past
  //    the last row is real): a store between two loads, or a load under an
  //    `if`, made the eight owners' round trips run one after another.
  //    ADD_RESIDUAL: the owners' rows are the new residual; each is normed here
  //    by its owner's scale (the weight loaded once, before the loop). Every
  //    thread loads the 8 scales after the 8 rows, all in flight before any
  //    wait (a wave's lanes read one scale address: one request a wave). Left
  //    to the compiler, the scales were loaded and waited on before the packs
  //    were issued, two round trips a row (ISA 2026-10-01T02-33-11Z).
  // A PACK A THREAD, stepping across the row (THREADS_PER_BLOCK packs a step,
  // K = 1): every owner's step in flight is WORLD packs a thread, as the
  // pack-at-a-time copy was (a whole row's was 16 at 7168, and 2-3% slower).
  for (int l = blockIdx.x; l < slice_m; l += gridDim.x) {
    float sc[WORLD];
    if constexpr (ADD_RESIDUAL)
      peers_load_row_scalars<WORLD>(scratch, sc,
                                    scale_at * int64_t{sizeof(DTYPE)}, l);
    for (int c = 0; c < input_size_n; c += THREADS_PER_BLOCK * P) {
      DTYPE got[WORLD][1][1][P];
      peers_load<1, THREADS_PER_BLOCK>(scratch, scratch_stride_m,
                                       scratch_stride_n, slice_m, input_size_n,
                                       got, l, c);
      WEIGHT_DTYPE w_step[1][1][P];
      if constexpr (ADD_RESIDUAL)
        load<1, THREADS_PER_BLOCK>(weight_ptr, 0, weight_stride_n, 1,
                                   input_size_n, w_step, 0, c);
#pragma unroll
      for (int r = 0; r < WORLD; ++r) {
        const int row = r * slice_m + l;
        if (row >= input_size_m) continue;
        if constexpr (ADD_RESIDUAL) {
          store<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>(got[r], residual_output_ptr,
                                      residual_output_stride_m,
                                      residual_output_stride_n, input_size_m,
                                      input_size_n, row, c);
          float normed_f[1][1][P];
          convert(got[r], normed_f);
          mul(normed_f, sc[r], normed_f);
          round<WEIGHT_DTYPE>(normed_f);
          float w_f[1][1][P];
          convert(w_step, w_f);
          mul(normed_f, w_f, normed_f);
          WEIGHT_DTYPE normed_w[1][1][P];
          convert(normed_f, normed_w);
          DTYPE normed[1][1][P];
          convert(normed_w, normed);
          store<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>(normed, output_ptr, output_stride_m,
                                      output_stride_n, input_size_m,
                                      input_size_n, row, c);
        } else {
          store<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>(got[r], output_ptr, output_stride_m,
                                      output_stride_n, input_size_m,
                                      input_size_n, row, c);
        }
      }
    }
  }
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
      input_ptrs, input_stride_m, input_stride_n, input_size_m, input_size_n,
      nullptr, 0, 0, weight_ptr, weight_stride_n, output_ptr, output_stride_m,
      output_stride_n, nullptr, 0, 0, scratch_ptrs, scratch_stride_m,
      scratch_stride_n, signals_ptrs, signals_stride_parity,
      signals_stride_block, signals_stride_writer, signals_size_parity,
      signals_size_block, signals_size_writer, status_ptr, status_stride_field,
      status_size_field, rank, timeout_ticks, eps);
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
      input_ptrs, input_stride_m, input_stride_n, input_size_m, input_size_n,
      residual_ptr, residual_stride_m, residual_stride_n, weight_ptr,
      weight_stride_n, output_ptr, output_stride_m, output_stride_n,
      residual_output_ptr, residual_output_stride_m, residual_output_stride_n,
      scratch_ptrs, scratch_stride_m, scratch_stride_n, signals_ptrs,
      signals_stride_parity, signals_stride_block, signals_stride_writer,
      signals_size_parity, signals_size_block, signals_size_writer, status_ptr,
      status_stride_field, status_size_field, rank, timeout_ticks, eps);
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
// may read only what the same block on a peer wrote. The buffers' places are
// counted as the one-shot's are (its head says why).
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, bool ADD_RESIDUAL,
          int TILE_N, int THREADS_PER_BLOCK>
DINLINE void all_reduce_push_two_shot_add_rms_norm_body(
    const DTYPE* const* __restrict__ input_ptrs, int64_t input_stride_m,
    int64_t input_stride_n, int input_size_m, int input_size_n,
    const DTYPE* residual_ptr, int64_t residual_stride_m,
    int64_t residual_stride_n, const WEIGHT_DTYPE* weight_ptr,
    int64_t weight_stride_n, DTYPE* output_ptr, int64_t output_stride_m,
    int64_t output_stride_n, DTYPE* residual_output_ptr,
    int64_t residual_output_stride_m, int64_t residual_output_stride_n,
    DTYPE* const* __restrict__ scratch_ptrs, int64_t scratch_stride_m,
    int64_t scratch_stride_n, int scratch_size_m, int scratch_size_n,
    uint32_t* const* __restrict__ signals_ptrs, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
    int status_size_field, int rank, uint64_t timeout_ticks, float eps) {
  constexpr int P = traits<DTYPE>::N;
  constexpr int K = TILE_N / (P * THREADS_PER_BLOCK);
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);
  // THIS RANK'S COLUMNS AND THIS BLOCK'S ROWS, AT KERNEL ENTRY: worked out after
  // the barrier, the grid division ran in VALU and the reduce-scatter's loops
  // were exec-masked (ISA 2026-10-15T18-36-19Z).
  const int slice =
      (input_size_n / P + WORLD - 1) / WORLD * P;  // a rank's, in elements
  const int col0 = rank * slice;
  const int own_n =
      max(0, min(slice, input_size_n - col0));  // the last rank's may be short
  const int my_m = input_size_m > static_cast<int>(blockIdx.x)
                       ? (input_size_m - blockIdx.x + gridDim.x - 1) / gridDim.x
                       : 0;

  // 1. Wait until every peer has launched, so its input is ready.
  barrier<Until::launched, 0, WORLD>(
      signals_ptrs, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);

  // 2. This rank's columns of this block's rows, summed over the ranks in rank
  // order and pushed
  //    to every rank (itself too), at their place in the tensor: this block's
  //    rows (every gridDim.x-th), A WAVE A ROW, since a rank's columns are a
  //    narrow slice (reduce_scatter_registers says how wide, and why). The
  //    ranks' pointers are read after the barrier, as in the pull kernels (held
  //    across it they spilled).
  reduce_scatter_registers<WORLD, THREADS_PER_BLOCK, TILE_N, Direction::push>(
      input_ptrs, input_stride_m, input_stride_n, input_size_m, input_size_n,
      scratch_ptrs, scratch_stride_m, scratch_stride_n, scratch_size_m,
      scratch_size_n, col0, own_n, my_m, rank);

  // 3. Every rank's sums are in this rank's scratch.
  barrier<Until::visible, 1, WORLD>(
      signals_ptrs, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);

  // 4. This block's rows out of this rank's scratch: (ADD_RESIDUAL) add the
  // residual, then RMSNorm,
  //    rounding as the reference does (the one-shot kernel spells it out). The
  //    next call's first sync keeps a peer from pushing into this scratch while
  //    it is read (a peer's next kernel starts only once this one has
  //    finished).
  const DTYPE* const own_scratch = peer_ptr(scratch_ptrs, rank);
  for (int row = blockIdx.x; row < input_size_m; row += gridDim.x) {
    // Every load of the row before any store, in flight together: the
    // scratch's, the residual's and the weight's (one-shot's).
    DTYPE own[1][K][P], res[1][K][P];
    load<1, THREADS_PER_BLOCK>(own_scratch, scratch_stride_m, scratch_stride_n,
                               input_size_m, input_size_n, own, row, 0);
    if constexpr (ADD_RESIDUAL)
      load<1, THREADS_PER_BLOCK>(residual_ptr, residual_stride_m,
                                 residual_stride_n, input_size_m, input_size_n,
                                 res, row, 0);
    WEIGHT_DTYPE w[1][K][P];
    load<1, THREADS_PER_BLOCK>(weight_ptr, 0, weight_stride_n, 1, input_size_n,
                               w, 0, 0);
    float s[1][K][P];
    convert(own, s);
    if constexpr (ADD_RESIDUAL) {
      float res_f[1][K][P];
      convert(res, res_f);
      add(s, res_f, s);
      DTYPE added[1][K][P];
      convert(s, added);
      store<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>(added, residual_output_ptr,
                                  residual_output_stride_m,
                                  residual_output_stride_n, input_size_m,
                                  input_size_n, row, 0);
    }
    float ss[1];
    partial_dot<THREADS_PER_BLOCK>(s, s, ss, input_size_n, 0);
    block_reduce<Sum, THREADS_PER_BLOCK>(ss);
    const float scale = rsqrtf(ss[0] * inv_hidden + eps);
    // out = T(W(W(s * scale) * float(w))), as the reference rounds
    float normed_f[1][K][P];
    mul(s, scale, normed_f);
    round<WEIGHT_DTYPE>(normed_f);
    float w_f[1][K][P];
    convert(w, w_f);
    mul(normed_f, w_f, normed_f);
    WEIGHT_DTYPE normed_w[1][K][P];
    convert(normed_f, normed_w);
    DTYPE normed[1][K][P];
    convert(normed_w, normed);
    store<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>(normed, output_ptr, output_stride_m,
                                output_stride_n, input_size_m, input_size_n,
                                row, 0);
  }
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
      input_ptrs, input_stride_m, input_stride_n, input_size_m, input_size_n,
      nullptr, 0, 0, weight_ptr, weight_stride_n, output_ptr, output_stride_m,
      output_stride_n, nullptr, 0, 0, scratch_ptrs, scratch_stride_m,
      scratch_stride_n, scratch_size_m, scratch_size_n, signals_ptrs,
      signals_stride_parity, signals_stride_block, signals_stride_writer,
      signals_size_parity, signals_size_block, signals_size_writer, status_ptr,
      status_stride_field, status_size_field, rank, timeout_ticks, eps);
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
      input_ptrs, input_stride_m, input_stride_n, input_size_m, input_size_n,
      residual_ptr, residual_stride_m, residual_stride_n, weight_ptr,
      weight_stride_n, output_ptr, output_stride_m, output_stride_n,
      residual_output_ptr, residual_output_stride_m, residual_output_stride_n,
      scratch_ptrs, scratch_stride_m, scratch_stride_n, scratch_size_m,
      scratch_size_n, signals_ptrs, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks, eps);
#else
  __builtin_trap();
#endif
}

}  // namespace hip_custom_communicator
