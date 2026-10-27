// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ALL-REDUCE THEN ADD, ATTNRES AND RMSNORM: the kernels, one section each (pull
// one-shot, pull two-shot, push two-shot), then our separate AttnRes kernel
// (add_attn_res_rms_norm), on rows. Every one computes AttnRes by
// the one function, attn_res (attn_res.cuh).

#pragma once

#include <type_traits>
#include "all_gather.cuh"
#include "common.cuh"
#include "attn_res.cuh"
#include "reduce_scatter.cuh"

namespace hip_custom_communicator {

// THE ALL-REDUCE-ADD-ATTNRES-RMSNORM'S ALGORITHMS: its ALGORITHM, exactly the bodies it has.
enum class AllReduceAddAttnResRmsNormAlgorithm { one_shot_pull, two_shot_pull, two_shot_push };

// =============================================================================
// all_reduce_pull_one_shot_add_attn_res_rms_norm
//
// One-shot pull all-reduce fused with Kimi-K3's attention residual (AttnRes)
// and its RMSNorm.
// =============================================================================

// A block owns a row, as the fused norm does: every rank reduces every row, so
// there is nothing to gather. `blocks` is [m, num_sources, n] at its row,
// source and column strides in elements; `write_idx` < 0 writes no block.


// =============================================================================
// all_reduce_pull_two_shot_add_attn_res_rms_norm
//
// Two-shot pull all-reduce fused with Kimi-K3's attention residual (AttnRes)
// and its RMSNorm.
// =============================================================================

// THE SLICE IS COLUMNS, AND BOTH HALVES PULL, as the plain two-shot all-reduce
// moves its bytes: each rank sums its columns of every row over the ranks into
// its own partials; after the sync each block reads its rows' columns from their
// owners and computes AttnRes on them itself. So the links carry what an
// all-reduce's do, AttnRes's two outputs (the prefix and out) are never
// gathered, and every rank does the AttnRes the unfused path would. A SLICE IS
// WHOLE WAVES (64 packs), so every wave's packs have one owner and partials's
// rank is the same across the wave. A BLOCK REDUCES THE ROWS IT THEN NORMS, so
// it waits only for its twin on every rank (a peers barrier, as the plain
// two-shot's) and the grid is AttnRes's to size. The reduce-scatter on 32
// blocks behind a world barrier left every other block waiting as long as it
// then computed, on a grid the barrier capped at the 192 blocks resident
// (thread trace 2026-10-04T20-14-36Z). `blocks` is [m, num_sources, n] with row
// and source strides in elements; `write_idx` < 0 writes no block.
// QUANTIZE_INT4: what crosses the links (common.cuh's QUANTIZE_INT4), one body
// each: none (THE EXACT BUILD, in the kernel) reads every rank's input in
// place, a column slice a rank; int4 (THE INT4 BUILD, below) pushes it
// quantized, whole groups a thread, a row slice a rank.

// THE INT4 BUILD, WHOLE GROUPS A THREAD (TILE_N = kGroups * kInt4Group * THREADS_PER_BLOCK: a
// block's threads are a chunk's groups, thread t groups t, t + THREADS_PER_BLOCK, ..., each its
// own), the pulled norms' INT4 shape (all_reduce_add_rms_norm.cuh)
// with AttnRes in place of the norm: the column split's INT4 read a 4-byte word and a 2-byte scale
// a pack (as many requests as the exact build for a quarter of the bytes) and summed through LDS,
// 762.5 against exact 643.1 us at 8192 x 7168 (2026-10-16T03-50-18Z).
//   1. THE PUSH: every owner's rows of this rank's input, quantized here and sent to that owner's
//      staging inbox at this rank's slot ([WORLD slots of slice_m rows] INT4; the host passes the
//      staging as [WORLD * slice_m, n]).
//   2. THE SUM, LOCAL: this rank's rows, every writer's group from its own inbox, summed from the
//      words (dequantize's slot form), left INT4 in this rank's partials.
//   3. THE GATHER AND ATTNRES: every owner's rows out of its partials, decoded into DTYPE (the sum
//      every rank's AttnRes starts from, as the column split's gather lands it), then AttnRes on
//      each, its groups a thread, by every rank (its two outputs are never gathered).
// THE WORK IS UNITS, a row l of the owners of one class (owner % classes), the class's unit
// u = class * slice_m + l, block u % gridDim.x's in all three phases, so a block reads after a
// barrier only what its twins wrote before it. classes: the most of 1, 2, 4 .. WORLD with
// classes * slice_m <= gridDim.x: with
// one class a block took every owner's row l, so no more than slice_m blocks worked and each
// AttnRes'd WORLD rows in turn (1024 x 3584: 84 us on 128 blocks or more, QuickReduce INT4 +
// attn_res 58; 2026-10-18T07-09-39Z).



// =============================================================================
// all_reduce_push_two_shot_add_attn_res_rms_norm
//
// Two-shot all-reduce (the reduce-scatter pulled, the all-gather pushed) fused
// with Kimi-K3's attention residual (AttnRes) and its RMSNorm: the push norm's
// two phases, with the one-shot AttnRes's row in place of the norm.
// =============================================================================

// THE SLICE IS COLUMNS, as the push norm's: each rank owns packs [rank * S,
// rank * S + S) of every row, sums them over the ranks and pushes the sums into
// every rank's partials, so after the sync each rank holds the whole sum locally
// and computes AttnRes on its rows itself: what crosses the links is the sum,
// once, and AttnRes's two outputs (the prefix and out) are never gathered. ROW
// q BELONGS TO BLOCK q % gridDim.x IN BOTH PHASES: after the sync a block may
// read only what the same block on a peer wrote. `blocks` is [m, num_sources,
// n] with row and source strides in elements; `write_idx` < 0 writes no block.
// QUANTIZE_INT4 as the pull's: with int4 the sums pushed to and read from the
// partials' INT4 payload.

// =============================================================================
// all_reduce_add_attn_res_rms_norm: THE ALL-REDUCE THEN ATTNRES AND ITS RMSNORM, ONE KERNEL: its ALGORITHM (AllReduceAddAttnResRmsNormAlgorithm) is a
// template parameter, each algorithm its own body under `if constexpr`.
// =============================================================================
template <AllReduceAddAttnResRmsNormAlgorithm ALGORITHM, bool QUANTIZE_INT4, typename DTYPE, int WORLD,
          int TILE_M, int TILE_N, int TILE_K, int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES,
          int WAVES_PER_EU, bool CHUNKED, bool TAIL>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_add_attn_res_rms_norm(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const DTYPE* const* __restrict__ input_ptr_array, int input_ptr_array_size,
        int64_t input_ptr_array_stride_m, int64_t input_ptr_array_stride_n,
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
        char* const* __restrict__ partials_ptr_array, int partials_ptr_array_size,
        int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
        int partials_size_n,
        char* const* __restrict__ staging_ptr_array,
        int staging_ptr_array_size, int64_t staging_stride_m, int64_t staging_stride_n,
        int staging_size_m, int staging_size_n,
        uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks, int num_blocks,
        float eps, float output_eps) {
  if constexpr (ALGORITHM == AllReduceAddAttnResRmsNormAlgorithm::one_shot_pull) {
    // PULL ONE-SHOT (all_reduce_pull_one_shot_add_attn_res_rms_norm, above).
    // EXACT ALONE: an INT4 build of it compiles this exact body, which never runs (the host refuses
    // an INT4 call of it by name, quantization_not_supported).
    constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
    // A PASS COVERS TILE_N ROUNDED UP TO WHOLE PACKS A THREAD (kSpan columns): any tile at any block,
    // the row's columns past the pass masked (rms_normalize's mask, the loads' clip), none skipped.
    constexpr int kPacks =
        (TILE_N + kPackElems * THREADS_PER_BLOCK - 1) / (kPackElems * THREADS_PER_BLOCK);
    constexpr int kSpan = kPacks * kPackElems * THREADS_PER_BLOCK;
    const float inv_hidden = 1.0f / static_cast<float>(input_size_n);

    // 1. Wait until every peer has launched: its input, the caller's tensor (a
    // captured call's, registered) or the host's copy in its staging (an eager
    // call's), is ready before it. Read where the input table says, in place.
    // UNORDERED (started, as the plain pull one-shot's): nothing is written before
    // it, and every read after it is of what a peer wrote before its launch.
    barrier<WORLD>(
        signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks, Until::started, 0);

    // 2. Each of this block's rows: read it from every rank in rank order, sum,
    //    AttnRes.
    for (int row = blockIdx.x; row < input_size_m; row += gridDim.x) {
      // A ROW WIDER THAN THE TILE (attn_res.cuh's sum_chunk): its chunks in passes, the sum's
      // chunk read once, in the first.
      if constexpr (CHUNKED) {
        const auto sum_chunk = [&](int col, DTYPE (&sum)[1][kPacks][kPackElems]) {
          DTYPE peers[WORLD][1][kPacks][kPackElems];
          for (int r = 0; r < WORLD; ++r)
            recv<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(input_ptr_array, input_ptr_array_size, r, 0, 0, input_ptr_array_stride_m, input_ptr_array_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, peers[r], row, col, 1);
  #pragma unroll
          for (int k = 0; k < kPacks; ++k)
  #pragma unroll
            for (int j = 0; j < kPackElems; ++j) {
              float acc = static_cast<float>(peers[0][0][k][j]);
  #pragma unroll
              for (int r = 1; r < WORLD; ++r) acc += static_cast<float>(peers[r][0][k][j]);
              sum[0][k][j] = static_cast<DTYPE>(acc);
            }
        };
        DTYPE unread[1][kPacks][kPackElems];  // a wide row's sum is sum_chunk's, by chunk
        attn_res<TILE_K, THREADS_PER_BLOCK, WAVE_SIZE, Layout::tile, false>(
            unread, prefix_ptr, prefix_stride_m, prefix_stride_n, blocks_ptr, blocks_stride_m,
            blocks_stride_r, blocks_stride_n, norm_weight_ptr, norm_weight_stride_n,
            qk_weight_ptr, qk_weight_stride_n, output_norm_weight_ptr,
            output_norm_weight_stride_n, prefix_output_ptr, prefix_output_stride_m,
            prefix_output_stride_n, output_ptr, output_stride_m, output_stride_n,
            block_output_ptr, block_output_stride_m, block_output_stride_n, input_size_m,
            input_size_n, row, 0, num_blocks, eps, output_eps, inv_hidden, sum_chunk,
            kSpan);
        continue;
      }
      DTYPE peers[WORLD][1][kPacks][kPackElems];
      for (int r = 0; r < WORLD; ++r)
        recv<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(input_ptr_array, input_ptr_array_size, r, 0, 0, input_ptr_array_stride_m, input_ptr_array_stride_n, 1, input_size_m, input_size_n, input_size_m, input_size_n, peers[r], row, 0, 1);
      // Summed in rank order in fp32, so every rank's sum agrees bitwise, rounded to DTYPE once.
      DTYPE sum[1][kPacks][kPackElems];
  #pragma unroll
      for (int k = 0; k < kPacks; ++k)
  #pragma unroll
        for (int j = 0; j < kPackElems; ++j) {
          float acc = static_cast<float>(peers[0][0][k][j]);
  #pragma unroll
          for (int r = 1; r < WORLD; ++r) acc += static_cast<float>(peers[r][0][k][j]);
          sum[0][k][j] = static_cast<DTYPE>(acc);
        }
      attn_res<TILE_K, THREADS_PER_BLOCK, WAVE_SIZE, Layout::tile, false>(
          sum, prefix_ptr, prefix_stride_m, prefix_stride_n, blocks_ptr,
          blocks_stride_m, blocks_stride_r, blocks_stride_n, norm_weight_ptr,
          norm_weight_stride_n, qk_weight_ptr, qk_weight_stride_n,
          output_norm_weight_ptr, output_norm_weight_stride_n, prefix_output_ptr,
          prefix_output_stride_m, prefix_output_stride_n, output_ptr,
          output_stride_m, output_stride_n, block_output_ptr,
          block_output_stride_m, block_output_stride_n, input_size_m,
          input_size_n, row, 0, num_blocks, eps, output_eps, inv_hidden);
    }

    // 3. No rank may overwrite its input until every peer has read it.
    barrier<WORLD>(
        signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks, Until::read, 1);
  } else if constexpr (ALGORITHM == AllReduceAddAttnResRmsNormAlgorithm::two_shot_pull) {
    // PULL TWO-SHOT (all_reduce_pull_two_shot_add_attn_res_rms_norm, above).
    if constexpr (QUANTIZE_INT4) {
      static_assert(TILE_M == 1, "the INT4 build is a row a block");
      constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
      static_assert(kPackElems * kInt4GroupPacks == kInt4Group, "a pack is a word of its group");
      // A THREAD LOOPS OVER ITS GROUPS (common.cuh's group_run: group j of thread t is the row's
      // t + j * THREADS_PER_BLOCK), kGroups of them, so the tile and the block are two fields: one
      // group a thread is the loop of one, the code it was.
      constexpr int kGroups =
          (TILE_N + kInt4Group * THREADS_PER_BLOCK - 1) / (kInt4Group * THREADS_PER_BLOCK);
      // A PASS COVERS TILE_N ROUNDED UP TO WHOLE GROUPS A THREAD (kGroupSpan columns): any tile at
      // any block, the row's groups past it masked (the loads' clip, the TAIL instance's last group).
      constexpr int kGroupSpan = kGroups * kInt4Group * THREADS_PER_BLOCK;
      constexpr int kThreadPacks = kGroups * kInt4GroupPacks;
      // REAL: a thread quantizes its groups alone (a group's scale is its own 32 values' largest),
      // so it holds whole groups, the same number in every thread (its registers are compiled); a
      // tile narrower than a group a thread would leave threads none (a group split over lanes is
      // another layout, not this body's).
      const int n = input_size_n;
      // A PAYLOAD ROW IS WHOLE GROUPS (its last group's packs past n nobody's), so every row's
      // groups start on a group of its slot.
      const int n_pad = TAIL ? (n + kInt4Group - 1) / kInt4Group * kInt4Group : n;
      const float inv_hidden = 1.0f / static_cast<float>(n);
      const int slice_m = (input_size_m + WORLD - 1) / WORLD;
      const int slice = slice_m * n_pad;  // a rank's rows, flat: its slot of an inbox
      const auto rows_of = [&](int r) { return max(0, min(slice_m, input_size_m - r * slice_m)); };
      const int grid = static_cast<int>(gridDim.x);
      // THE CLASSES, a power of two (an owner's class its low bits), and this block's: with one, its
      // rows l = blockIdx.x, + gridDim.x, ...; with more, every block one unit at most (units <=
      // gridDim.x), its row and class divided out once, here.
      int classes = 1;
      while (classes * 2 <= WORLD && classes * 2 * slice_m <= grid) classes *= 2;
      const int mask = classes - 1;
      const int c = classes == 1 ? 0 : static_cast<int>(blockIdx.x) / slice_m;
      const int first = classes == 1 || static_cast<int>(blockIdx.x) < classes * slice_m
                            ? static_cast<int>(blockIdx.x) - c * slice_m
                            : slice_m;
      const int step = classes == 1 ? grid : slice_m;

      // 1. THE PUSH, owners rotated (rank + w), a row of each a step, two owners' rows in flight.
      // Owner r's row l, its groups from column col (a kGroupSpan chunk: one, unless the row is wider).
      const auto push = [&](int l, int r, int col) {
        DTYPE x[1][kThreadPacks][kPackElems];
        load<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::group_run, TAIL>(
            &input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1, input_size_m, n,
            input_size_m, n, x, r * slice_m + l, col, 1);
        // QUANTIZED ONCE, by this thread alone (each of its groups), then sent as the payload. One
        // group as it always was (through quantize's group form the TAIL builds scheduled
        // differently: ISA 2026-10-22).
        uint32_t payload[1][kThreadPacks][2];
        if constexpr (kGroups == 1) {
          uint32_t words[kInt4GroupPacks];
          float scale;
          quantize(x[0], words, scale);
  #pragma unroll
          for (int k = 0; k < kInt4GroupPacks; ++k) {
            payload[0][k][0] = words[k];
            payload[0][k][1] = __builtin_bit_cast(uint32_t, scale);
          }
        } else {
          quantize(x, payload);
        }
        send<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, true, Layout::group_run, TAIL>(
            payload, staging_ptr_array, staging_ptr_array_size, r, rank, slice, staging_stride_m,
            staging_stride_n, WORLD, staging_size_m, staging_size_n, 1, TAIL ? l * n_pad + n : (l + 1) * n, 0,
            l * n_pad + col, 1);
      };
      if constexpr (!CHUNKED) {
        for (int l = first; l < slice_m; l += step) {
  #pragma unroll 2
          for (int w = 0; w < WORLD; ++w) {
            const int r = (rank + w) % WORLD;
            // Another class's owner, or a late owner's short slice (every thread alike).
            if ((r & mask) != c || l >= rows_of(r)) continue;
            push(l, r, 0);
          }
        }
      } else {
        // A ROW WIDER THAN THE TILE: its chunks one after another.
        for (int l = first; l < slice_m; l += step)
          for (int w = 0; w < WORLD; ++w) {
            const int r = (rank + w) % WORLD;
            if ((r & mask) != c || l >= rows_of(r)) continue;
            for (int col = 0; col < n; col += kGroupSpan) push(l, r, col);
          }
      }
      barrier<WORLD>(
          signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
          signals_stride_writer, signals_size_parity, signals_size_block,
          signals_size_writer, status_ptr, status_stride_field, status_size_field,
          rank, timeout_ticks, Until::launched, 0);

      // 2. THIS RANK'S ROWS: every writer's group of the row from this rank's inbox (slots rotated
      // from this rank's), summed in fp32 and left INT4 in this rank's partials.
      for (int l = first; l < slice_m; l += step) {
        if ((rank & mask) != c || l >= rows_of(rank)) continue;
        // The row's groups from column col: one chunk, or a CHUNKED row's every chunk.
        const auto sum = [&](int col) {
          uint32_t got[WORLD][1][kThreadPacks][2];
          load<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, true, Layout::group_run, TAIL>(
              staging_ptr_array, staging_ptr_array_size, rank, rank, slice, staging_stride_m,
              staging_stride_n, WORLD, staging_size_m, staging_size_n, 1, TAIL ? l * n_pad + n : (l + 1) * n, got, 0,
              l * n_pad + col, 1);
          float summed[1][kThreadPacks][kPackElems];
          dequantize<DTYPE>(got, summed);
          store<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, true, Layout::group_run, TAIL>(
              summed, partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m,
              partials_stride_n, 1, partials_size_m, partials_size_n, 1, TAIL ? l * n_pad + n : (l + 1) * n, 0,
              l * n_pad + col, 1);
        };
        if constexpr (CHUNKED)
          for (int col = 0; col < n; col += kGroupSpan) sum(col);
        else
          sum(0);
      }

      // 3. Every rank's rows are visible to its peers.
      barrier<WORLD>(
          signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
          signals_stride_writer, signals_size_parity, signals_size_block,
          signals_size_writer, status_ptr, status_stride_field, status_size_field,
          rank, timeout_ticks, Until::visible, 1);

      // 4. EVERY OWNER'S ROW l, owners rotated (rank + v): the row's group from its owner, decoded,
      // then AttnRes on it. The next call's first sync keeps an owner from overwriting its partials
      // while they are read.
      for (int l = first; l < slice_m; l += step) {
        for (int v = 0; v < WORLD; ++v) {
          const int owner = (rank + v) % WORLD;
          if ((owner & mask) != c || l >= rows_of(owner)) continue;
          // Owner's row l from column col, decoded.
          const auto sum_chunk = [&](int col, DTYPE (&sum)[1][kThreadPacks][kPackElems]) {
            uint32_t raw[1][kThreadPacks][2];
            recv<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, true, Layout::group_run, TAIL>(
                partials_ptr_array, partials_ptr_array_size, owner, 0, 0, partials_stride_m,
                partials_stride_n, 1, partials_size_m, partials_size_n, 1, TAIL ? l * n_pad + n : (l + 1) * n, raw, 0,
                l * n_pad + col, 1);
  #pragma unroll
            for (int k = 0; k < kThreadPacks; ++k)
              dequantize(raw[0][k][0], __builtin_bit_cast(float, raw[0][k][1]), sum[0][k]);
          };
          if constexpr (CHUNKED) {  // a row wider than the tile (attn_res.cuh's sum_chunk)
            DTYPE unread[1][kThreadPacks][kPackElems];  // a wide row's sum is sum_chunk's, by chunk
            attn_res<TILE_K, THREADS_PER_BLOCK, WAVE_SIZE, Layout::group_run, TAIL>(
                unread, prefix_ptr, prefix_stride_m, prefix_stride_n, blocks_ptr, blocks_stride_m,
                blocks_stride_r, blocks_stride_n, norm_weight_ptr, norm_weight_stride_n,
                qk_weight_ptr, qk_weight_stride_n, output_norm_weight_ptr,
                output_norm_weight_stride_n, prefix_output_ptr, prefix_output_stride_m,
                prefix_output_stride_n, output_ptr, output_stride_m, output_stride_n,
                block_output_ptr, block_output_stride_m, block_output_stride_n, input_size_m, n,
                owner * slice_m + l, 0, num_blocks, eps, output_eps, inv_hidden, sum_chunk,
                kGroupSpan);
            continue;
          }
          uint32_t raw[1][kThreadPacks][2];
          recv<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, true, Layout::group_run, TAIL>(
              partials_ptr_array, partials_ptr_array_size, owner, 0, 0, partials_stride_m,
              partials_stride_n, 1, partials_size_m, partials_size_n, 1, TAIL ? l * n_pad + n : (l + 1) * n, raw, 0, l * n_pad, 1);
          DTYPE sum[1][kThreadPacks][kPackElems];
  #pragma unroll
          for (int k = 0; k < kThreadPacks; ++k)
            dequantize(raw[0][k][0], __builtin_bit_cast(float, raw[0][k][1]), sum[0][k]);
          attn_res<TILE_K, THREADS_PER_BLOCK, WAVE_SIZE, Layout::group_run, TAIL>(
              sum, prefix_ptr, prefix_stride_m, prefix_stride_n, blocks_ptr, blocks_stride_m,
              blocks_stride_r, blocks_stride_n, norm_weight_ptr, norm_weight_stride_n, qk_weight_ptr,
              qk_weight_stride_n, output_norm_weight_ptr, output_norm_weight_stride_n,
              prefix_output_ptr, prefix_output_stride_m, prefix_output_stride_n, output_ptr,
              output_stride_m, output_stride_n, block_output_ptr, block_output_stride_m,
              block_output_stride_n, input_size_m, n, owner * slice_m + l, 0, num_blocks, eps,
              output_eps, inv_hidden);
        }
      }
    } else {
      // THE EXACT BUILD.
      constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
      // A PASS COVERS TILE_N ROUNDED UP TO WHOLE PACKS A THREAD (kSpan columns): any tile at any block,
      // the row's columns past the pass masked (rms_normalize's mask, the loads' clip), none skipped.
      constexpr int kPacks =
          (TILE_N + kPackElems * THREADS_PER_BLOCK - 1) / (kPackElems * THREADS_PER_BLOCK);
      constexpr int kSpan = kPacks * kPackElems * THREADS_PER_BLOCK;
      const float inv_hidden = 1.0f / static_cast<float>(input_size_n);
      // EVERY RANK AN EQUAL SLICE OF THE ROW, in elements: rounded up to whole
      // waves (so a wave's packs had one owner), rank 7 owned nothing at 3584 and
      // 7168 and the others reduce-scattered 8/7 of the row.
      const int slice_n = (input_size_n / kPackElems + WORLD - 1) / WORLD * kPackElems;

      // 1. Nothing staged, every rank's input read in place where the input table
      // says (a captured call's tensor, or an eager call's host copy in its
      // staging). Wait until every peer has launched, UNORDERED (started, as the plain pull
      // two-shot's): nothing written before it, every read after it of what a peer wrote before
      // its launch.
      // 2. This rank's columns of this block's rows (tiles of TILE_M, every gridDim.x-th), summed
      //    over the ranks into this rank's partials: THE PEERS READ ROTATED, rank + r's r-th, as the
      //    plain two-shot reads them (in rank order every GPU read rank 0 first, a link at a time
      //    carrying the machine's reads, and the reduce-scatter took 1.6x the plain two-shot's cycles
      //    for its bytes: thread traces 2026-10-04T21-03-09Z, 21-26-11Z), summed through LDS.
      // 3. Every rank's sums are in its partials: this block's twin on every rank has reduced the
      //    same rows.
      reduce_scatter<ReduceScatterAlgorithm::columns_pull, false, WORLD, TILE_M, TILE_N, THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, true, false>(
          input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n, input_ptr_array,
          input_ptr_array_size, input_ptr_array_stride_m, input_ptr_array_stride_n,
          staging_ptr_array, staging_ptr_array_size, staging_stride_m, staging_stride_n,
          staging_size_m, staging_size_n, partials_ptr_array, partials_ptr_array_size,
          partials_stride_m, partials_stride_n, partials_size_m, partials_size_n, signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
          signals_stride_writer, signals_size_parity, signals_size_block, signals_size_writer,
          status_ptr, status_stride_field, status_size_field, rank,
          timeout_ticks);

      // 4. Each of this block's tiles: its packs from the ranks that own them (A
      //    WAVE'S PACKS MAY HAVE TWO OWNERS, so each lane reads its owner's partials
      //    from its base, picked per lane), at the input's place in the partials, then AttnRes. A
      //    TILE'S REGISTERS ARE THE LOOP BODY'S SCOPE: a row past input_size_m (never loaded) must
      //    not keep the last tile's values (held across the whole tile they were 16 VGPRs at
      //    TILE_M 2, 150 -> 166; ISA harness 2026-10-15).
      const int tiles = (input_size_m + TILE_M - 1) / TILE_M;
      for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
        const int offs_m = t * TILE_M;
          // A ROW WIDER THAN THE TILE (attn_res.cuh's sum_chunk): its chunks in passes, the sum's
        // chunk read once, in the first.
        if constexpr (CHUNKED) {
          const auto sum_chunk = [&](int col, DTYPE (&sum)[TILE_M][kPacks][kPackElems]) {
            all_gather<AllGatherAlgorithm::columns, false, DTYPE, WORLD, TILE_N, THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, true, false>(
                partials_ptr_array, partials_ptr_array_size, partials_stride_m, partials_stride_n,
                partials_size_m, partials_size_n, sum, 0, 0, input_size_m, input_size_n, offs_m,
                col, slice_n, rank);
          };
          DTYPE unread[TILE_M][kPacks][kPackElems];  // a wide row's sum is sum_chunk's, by chunk
          attn_res<TILE_K, THREADS_PER_BLOCK, WAVE_SIZE, Layout::tile, TAIL>(
              unread, prefix_ptr, prefix_stride_m, prefix_stride_n, blocks_ptr, blocks_stride_m,
              blocks_stride_r, blocks_stride_n, norm_weight_ptr, norm_weight_stride_n,
              qk_weight_ptr, qk_weight_stride_n, output_norm_weight_ptr,
              output_norm_weight_stride_n, prefix_output_ptr, prefix_output_stride_m,
              prefix_output_stride_n, output_ptr, output_stride_m, output_stride_n,
              block_output_ptr, block_output_stride_m, block_output_stride_n, input_size_m,
              input_size_n, offs_m, 0, num_blocks, eps, output_eps, inv_hidden, sum_chunk,
              kSpan);
          continue;
        }
        DTYPE sum[TILE_M][kPacks][kPackElems];
        all_gather<AllGatherAlgorithm::columns, false, DTYPE, WORLD, TILE_N, THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, true, false>(
            partials_ptr_array, partials_ptr_array_size, partials_stride_m, partials_stride_n,
            partials_size_m, partials_size_n, sum, 0, 0, input_size_m, input_size_n, offs_m, 0,
            slice_n, rank);
        attn_res<TILE_K, THREADS_PER_BLOCK, WAVE_SIZE, Layout::tile, TAIL>(
            sum, prefix_ptr, prefix_stride_m, prefix_stride_n, blocks_ptr,
            blocks_stride_m, blocks_stride_r, blocks_stride_n, norm_weight_ptr,
            norm_weight_stride_n, qk_weight_ptr, qk_weight_stride_n,
            output_norm_weight_ptr, output_norm_weight_stride_n, prefix_output_ptr,
            prefix_output_stride_m, prefix_output_stride_n, output_ptr,
            output_stride_m, output_stride_n, block_output_ptr,
            block_output_stride_m, block_output_stride_n, input_size_m,
            input_size_n, offs_m, 0, num_blocks, eps, output_eps, inv_hidden);
      }
    }
  } else {
    // PUSH TWO-SHOT (all_reduce_push_two_shot_add_attn_res_rms_norm, above).
    static_assert(ALGORITHM == AllReduceAddAttnResRmsNormAlgorithm::two_shot_push,
                  "all_reduce_add_attn_res_rms_norm: no body for this algorithm");
    // THE INT4 PAYLOADS are inside the staging and the partials (common.cuh's
    // INT4 payload layout): found from each region's own sizes where read or
    // written.
    constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
    // A PASS COVERS TILE_N ROUNDED UP TO WHOLE PACKS A THREAD (kSpan columns): any tile at any block,
    // the row's columns past the pass masked (rms_normalize's mask, the loads' clip), none skipped.
    constexpr int kPacks =
        (TILE_N + kPackElems * THREADS_PER_BLOCK - 1) / (kPackElems * THREADS_PER_BLOCK);
    constexpr int kSpan = kPacks * kPackElems * THREADS_PER_BLOCK;
    const float inv_hidden = 1.0f / static_cast<float>(input_size_n);
    // 2. This rank's columns of this block's rows, summed over the ranks in rank order and pushed
    //    to every rank (itself too), at their place in the tensor: A SLICE OF LANES A ROW, as wide
    //    as a rank's slice of a TILE_N row in whole waves, so a row's slice is one round trip (a
    //    wave a row took two at 7168, 0.8 us at 16-32 tokens), and as many rows at once as the block
    //    holds; a pack a thread, not a slice's worth (every peer's packs of a whole slice were 32
    //    packs at 16384, spilling). The ranks' pointers are read after the barrier (held across it
    //    they spilled).
    // 3. Every rank's sums are in this rank's partials, and every peer has read this rank's input.
    // THE FIRST BARRIER: exact, UNORDERED (started, as the plain pulls'): nothing written before it,
    // every input read in place after it (written before the peers' launches), and the pushes into a
    // peer's partials after it land where that peer finished reading in its last launch, ended before
    // this one started. int4 stages this rank's rows before it, so it stays ordered (launched).
    reduce_scatter<ReduceScatterAlgorithm::columns_push, QUANTIZE_INT4, WORLD, 1, TILE_N, THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, true, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n, input_ptr_array,
        input_ptr_array_size, input_ptr_array_stride_m, input_ptr_array_stride_n,
        staging_ptr_array, staging_ptr_array_size, staging_stride_m, staging_stride_n,
        staging_size_m, staging_size_n, partials_ptr_array, partials_ptr_array_size,
        partials_stride_m, partials_stride_n, partials_size_m, partials_size_n, signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block, signals_size_writer,
        status_ptr, status_stride_field, status_size_field, rank,
        timeout_ticks);

    // 4. This block's rows: the sum out of this rank's partials (at the input's
    //    place in it), then AttnRes, as the one-shot does. The next call's first
    //    sync keeps a peer from pushing into this partials while it is read (a
    //    peer's next kernel starts only once this one has finished).
    //    int4: its INT4 payload, worked out from the partials' own sizes by load.
    for (int row = blockIdx.x; row < input_size_m; row += gridDim.x) {
      // A ROW WIDER THAN THE TILE (attn_res.cuh's sum_chunk): its chunks in passes, the sum's
      // chunk read once, in the first.
      if constexpr (CHUNKED) {
        const auto sum_chunk = [&](int col, DTYPE (&sum)[1][kPacks][kPackElems]) {
          load<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4, Layout::tile, TAIL>(partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n, input_size_m, input_size_n, sum, row, col, 1);
        };
        DTYPE unread[1][kPacks][kPackElems];  // a wide row's sum is sum_chunk's, by chunk
        attn_res<TILE_K, THREADS_PER_BLOCK, WAVE_SIZE, Layout::tile, TAIL>(
            unread, prefix_ptr, prefix_stride_m, prefix_stride_n, blocks_ptr, blocks_stride_m,
            blocks_stride_r, blocks_stride_n, norm_weight_ptr, norm_weight_stride_n,
            qk_weight_ptr, qk_weight_stride_n, output_norm_weight_ptr,
            output_norm_weight_stride_n, prefix_output_ptr, prefix_output_stride_m,
            prefix_output_stride_n, output_ptr, output_stride_m, output_stride_n,
            block_output_ptr, block_output_stride_m, block_output_stride_n, input_size_m,
            input_size_n, row, 0, num_blocks, eps, output_eps, inv_hidden, sum_chunk,
            kSpan);
        continue;
      }
      DTYPE sum[1][kPacks][kPackElems];
      load<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4, Layout::tile, TAIL>(partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n, input_size_m, input_size_n, sum, row, 0, 1);
      attn_res<TILE_K, THREADS_PER_BLOCK, WAVE_SIZE, Layout::tile, TAIL>(
          sum, prefix_ptr, prefix_stride_m, prefix_stride_n, blocks_ptr,
          blocks_stride_m, blocks_stride_r, blocks_stride_n, norm_weight_ptr,
          norm_weight_stride_n, qk_weight_ptr, qk_weight_stride_n,
          output_norm_weight_ptr, output_norm_weight_stride_n, prefix_output_ptr,
          prefix_output_stride_m, prefix_output_stride_n, output_ptr,
          output_stride_m, output_stride_n, block_output_ptr,
          block_output_stride_m, block_output_stride_n, input_size_m,
          input_size_n, row, 0, num_blocks, eps, output_eps, inv_hidden);
    }
  }
}



// =============================================================================
// add_attn_res_rms_norm
//
// OUR SEPARATE ATTNRES KERNEL, ON ROWS: Kimi-K3's AttnRes and its
// RMSNorm on rows already all-reduced (STANDARDS 3.2, *We win at the op level*:
// where fusing stops paying, the op runs our all-reduce, then this). Python
// composes the op as two kernels (Template::add_attn_res_rms_norm): our
// all-reduce into a tensor of this rank's, then
// this kernel on it: AttnRes on its own
// grid at its own occupancy, not on the communication kernel's (the fused INT4
// pull ran it at 250 VGPRs, 2 waves a SIMD, and more blocks did not help: 1024
// slower than 512 at 8192 x 3584; 2026-10-18T13-15-40Z). Every row's math and
// rounding the fused kernels' (attn_res): the sum `input` is the
// all-reduce's, rounded to DTYPE as theirs is. Nothing over the peers.
// =============================================================================

// A TILE OF TILE_M ROWS a work item, the grid striding over them; `blocks` is
// [m, num_sources, n] at its row, source and column strides in elements;
// `block_output_ptr` null writes no block.
template <typename DTYPE, int TILE_M, int TILE_N, int TILE_K, int THREADS_PER_BLOCK, int WAVE_SIZE,
          int PACK_BYTES, int WAVES_PER_EU, bool CHUNKED>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    add_attn_res_rms_norm(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
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
        int block_output_size_m, int block_output_size_n, int num_blocks,
        float eps, float output_eps) {
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
  // A PASS COVERS TILE_N ROUNDED UP TO WHOLE PACKS A THREAD (kSpan columns): any tile at any block,
  // the row's columns past the pass masked (rms_normalize's mask, the loads' clip), none skipped.
  constexpr int kPacks =
      (TILE_N + kPackElems * THREADS_PER_BLOCK - 1) / (kPackElems * THREADS_PER_BLOCK);
  constexpr int kSpan = kPacks * kPackElems * THREADS_PER_BLOCK;
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);
  // A TILE'S REGISTERS ARE THE LOOP BODY'S SCOPE (the fused pull's reason).
  const int tiles = (input_size_m + TILE_M - 1) / TILE_M;
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int offs_m = t * TILE_M;
    // A ROW WIDER THAN THE TILE (attn_res.cuh's sum_chunk): its chunks in passes, the sum's
    // chunk read once, in the first.
    if constexpr (CHUNKED) {
      const auto sum_chunk = [&](int col, DTYPE (&sum)[TILE_M][kPacks][kPackElems]) {
        load<1, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
            &input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1, input_size_m,
            input_size_n, input_size_m, input_size_n, sum, offs_m, col, 1);
      };
      DTYPE unread[TILE_M][kPacks][kPackElems];  // a wide row's sum is sum_chunk's, by chunk
      attn_res<TILE_K, THREADS_PER_BLOCK, WAVE_SIZE, Layout::tile, false>(
          unread, prefix_ptr, prefix_stride_m, prefix_stride_n, blocks_ptr, blocks_stride_m,
          blocks_stride_r, blocks_stride_n, norm_weight_ptr, norm_weight_stride_n,
          qk_weight_ptr, qk_weight_stride_n, output_norm_weight_ptr,
          output_norm_weight_stride_n, prefix_output_ptr, prefix_output_stride_m,
          prefix_output_stride_n, output_ptr, output_stride_m, output_stride_n,
          block_output_ptr, block_output_stride_m, block_output_stride_n, input_size_m,
          input_size_n, offs_m, 0, num_blocks, eps, output_eps, inv_hidden, sum_chunk,
          kSpan);
      continue;
    }
    DTYPE sum[TILE_M][kPacks][kPackElems];
    load<1, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
        &input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1, input_size_m,
        input_size_n, input_size_m, input_size_n, sum, offs_m, 0, 1);
    attn_res<TILE_K, THREADS_PER_BLOCK, WAVE_SIZE, Layout::tile, false>(
        sum, prefix_ptr, prefix_stride_m, prefix_stride_n, blocks_ptr,
        blocks_stride_m, blocks_stride_r, blocks_stride_n, norm_weight_ptr,
        norm_weight_stride_n, qk_weight_ptr, qk_weight_stride_n,
        output_norm_weight_ptr, output_norm_weight_stride_n, prefix_output_ptr,
        prefix_output_stride_m, prefix_output_stride_n, output_ptr,
        output_stride_m, output_stride_n, block_output_ptr,
        block_output_stride_m, block_output_stride_n, input_size_m,
        input_size_n, offs_m, 0, num_blocks, eps, output_eps, inv_hidden);
  }
}

}  // namespace hip_custom_communicator
