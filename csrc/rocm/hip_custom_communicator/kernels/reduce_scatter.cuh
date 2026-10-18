// SPDX-License-Identifier: MIT
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE REDUCE-SCATTER, written once with the common ops (load, store, barrier, reduce, the
// codec): from every rank's input to every rank's sums in the partials, visible to its peers. Its
// variations are template parameters, its walks written inline (the head of this file says why
// each walk is the one it is).

#pragma once

#include "common.cuh"

namespace hip_custom_communicator {

// THE COLLECTIVES (reduce_scatter here, all_gather in all_gather.cuh, which walks the same
// walk): one tested and tuned way to do each, which the kernels call rather than write again
// (the fused AttnRes pull's own reduce-scatter took 1.6x the plain two-shot's cycles for its
// bytes, thread traces 2026-10-04T21-03-09Z and 21-26-11Z). Their vocabulary (Direction, Walk,
// Accumulate) is theirs, here; types.cuh's forcing names Direction too.

// THE LOADS ARE THE COMMUNICATION: a rank's pointer points into its memory,
// opened in this process over IPC (handle.cuh), so loading reads it over the links; the sum
// and the store are local. The kernel syncs around these calls (the barriers):
// before, every peer's data is ready; between the reduce-scatter and the
// all-gather, every rank's sums are visible. RANKS ARE READ ROTATED, rank + w's
// w-th, so at any moment the GPUs read different peers; each range or slice is
// summed by one rank, so the order differing by rank is harmless.
//
// THE FLAT WALK (the plain two-shot): an elementwise [R, M, N] problem -- R
// ranks' copies of an [M, N] buffer, each at its own strides -- WALKED AS
// [R, 1, M*N]: the rows are only the buffer's layout, so the elements are
// numbered row-major (e = row * N + col, common.cuh's element_at, Layout flat) and each rank's SHARE is one
// contiguous range of them, a TILE_N chunk a step. WHY: the links and HBM
// reward long reads from each peer. Cut a row at a time, a rank's share was its
// slice of every row -- 1.8 KB runs at N = 7168, the plain all-reduce +2%
// against the flat walk; 128-column tiles, 256 B runs, +3-7% (2026-10-11) --
// while a contiguous buffer's range is one run of the whole share. IT SERVES
// BADLY a strided buffer: its runs are its rows wherever it has them, and each
// pack's row and column are divided out. The walk is the same for every step
// and every rank, so a block reads in the all-gather what the same block on
// each rank wrote in the reduce-scatter. A RANK'S COPY OF A CHUNK IS kLanes =
// THREADS_PER_BLOCK / WORLD threads, whole waves (its rank's pointer one scalar select;
// every rank's held up front was ~20 scalar loads before the first barrier).
//
// CONTIGUOUS, whether an [*, n] buffer's rows are back to back so its range is
// one run, is a template parameter of the kernel, its instance chosen on the
// host from the strides as given. A compile-time choice, not a branch in the
// kernel: both paths in one kernel cost ~0.1 us a call at 128-512 tokens
// against the contiguous path alone (2026-10-12T01-17-18Z), and read from the
// strides in the kernel (I8) the contiguous instance grew 856 -> 1338
// instructions (pull), 926 -> 1394 (push), 3269 -> 5439 (int4) (ISA
// 2026-10-18T00-20-45Z).

// WHICH WAY A REDUCE-SCATTER'S SUM GOES, and a caller's forcing (types.cuh): pull, this rank
// reads its peers and keeps the sum in its partials; push, the sum is stored in every rank's
// (itself too), so after the sync each rank holds the whole sum locally.
enum class Direction : int { pull = 0, push = 1 };
// HOW reduce_scatter AND all_gather WALK THE DATA (reduce_scatter.cuh, all_gather.cuh): flat, the
// elements numbered row-major and a rank's share a contiguous range (above); rows, a rank's columns
// of every row. And where reduce_scatter sums a rank's copies: through LDS or in registers.
enum class Walk : int { flat = 0, rows = 1 };
enum class Accumulate : int { lds = 0, registers = 1 };

// [PUSH OR STAGE] BARRIER(UNTIL, 0) SUM BARRIER(visible, 1): before the first barrier, a pushed
// walk stores every rank's share of this rank's input in that rank's staging at this rank's slot,
// and an int4 rows walk stages this rank's rows quantized in its own staging (the rows this
// block's twin on every rank reads, tiles of TILE_M rows blockIdx.x, blockIdx.x + gridDim.x, ...);
// after the second, every rank's sums are in the partials and every peer is done reading.
//
// WALK::flat (the plain two-shot): rows offs_m.. offs_m + input_size_m of the [*, input_size_n]
// problem numbered row-major, each rank's SHARE one contiguous range of them, a TILE_N chunk a step,
// a rank's copy of a chunk kLanes = THREADS_PER_BLOCK / WORLD threads (whole waves, its rank rank + w,
// rotated). PULL (ACCUMULATE lds, none): every rank's input read in place through the INPUT TABLE
// (input_ptr_array at its strides). PUSH (none, lds): this rank's input (input_ptr) pushed into
// every rank's staging inbox, [WORLD, slot] (writer w's slot at row w, staging_stride_m elements a
// slot), then the slots summed locally. PUSH (int4, registers): A GROUP A THREAD, each block a
// contiguous range of every share's groups (ceil(groups / grid), THREADS_PER_BLOCK a step), quantized here
// into every rank's staging INT4 payload at this rank's place (writer w's at share w's), the
// writers' groups summed from the words (no pack of every writer decoded) and quantized into this rank's partials.
// The plain all-reduce at 72 blocks: 71.8 / 109.0 / 178.2 / 286.6 us at 1024-8192 tokens of
// 7168, against the exact 75.5 / 142.0 / 273.3 / 535.4 (2026-10-16T03-14-06Z).
//
// WALK::rows (the fused ops' two-shots): this rank's columns of every row, equal slices of whole
// packs a rank (a late rank's short or empty). ACCUMULATE lds (pull): a chunk of a row at a time,
// WORLD rows of kLanes threads, row w rank rank + w's copy recv'd per lane (a row of threads' rank), summed
// over the rows through LDS into this rank's partials. ACCUMULATE registers: a row of threads as
// wide as a rank's slice of a TILE_N row (whole waves), a pack a thread, as many rows at once as the
// block holds, every rank's pack summed in registers in rank order; push stores the sum in every
// rank's partials, pull in this rank's. none reads every rank's input in place; int4 every rank's
// staged INT4 payload, an INT4 payload's layout from its own buffer's sizes.
template <int WORLD, int TILE_M, int TILE_N, int THREADS_PER_BLOCK, int QUANTIZATION, bool CONTIGUOUS,
          Walk WALK,
          Accumulate ACCUMULATE, Direction DIRECTION, Until UNTIL, typename DTYPE>
DINLINE void reduce_scatter(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n, int input_size_m,
    int input_size_n, const DTYPE* const* input_ptr_array, int input_ptr_array_size,
    int64_t input_ptr_array_stride_m, int64_t input_ptr_array_stride_n,
    char* const* staging_ptr_array, int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    char* const* partials_ptr_array, int partials_ptr_array_size, int64_t partials_stride_m,
    int64_t partials_stride_n, int partials_size_m, int partials_size_n,
    uint32_t* const* signals_ptr_array, int signals_ptr_array_size, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer, int signals_size_parity,
    int signals_size_block, int signals_size_writer, uint32_t* status_ptr,
    int64_t status_stride_field, int status_size_field, int offs_m, int rank,
    uint64_t timeout_ticks) {
  constexpr int kPackElems = traits<DTYPE>::N;
  constexpr bool kInt4 = QUANTIZATION == Quantization::int4;
  if constexpr (WALK == Walk::flat) {
    // THE SHARES of the band's elements, in whole packs (int4: whole groups), the last rank's short.
    // A rank's share is slot r of a [*, input_size_n] buffer viewed flat (stride `slice` elements:
    // common.cuh's Layout run / flat).
    constexpr int kUnit = kInt4 ? kInt4Group : kPackElems;
    const int64_t total = int64_t{input_size_m} * input_size_n;
    const int slice = static_cast<int>((total / kUnit + WORLD - 1) / WORLD * kUnit);
    const auto share_len = [&](int r) {
      return static_cast<int>(max(int64_t{0}, min(int64_t{slice}, total - int64_t{r} * slice)));
    };
    if constexpr (kInt4) {
      static_assert(DIRECTION == Direction::push && ACCUMULATE == Accumulate::registers,
                    "the int4 flat walk pushes and sums in registers");
      constexpr Layout kInput = CONTIGUOUS ? Layout::group_run : Layout::group_flat;
      const int groups_all = slice / kInt4Group;
      const int grid = static_cast<int>(gridDim.x);
      const int per_block = (groups_all + grid - 1) / grid;
      const int g_first = static_cast<int>(blockIdx.x) * per_block;
      const int g_end = min(g_first + per_block, groups_all);
      // 1. THE PUSH: every rank's share of this rank's input, a group a thread, quantized here and
      // sent to that rank's staging payload at this rank's slot. TWO RANKS' GROUPS IN FLIGHT:
      // unrolled whole, every rank's 4 loads were hoisted together, 128 VGPRs of input, spilling.
      for (int g0 = g_first; g0 < g_end; g0 += THREADS_PER_BLOCK) {
#pragma unroll 2
        for (int w = 0; w < WORLD; ++w) {
          const int r = (rank + w) % WORLD;
          const int len = share_len(r);
          if (len / kInt4Group == 0) continue;  // a late rank's empty share (every block alike)
          DTYPE x[1][kInt4GroupPacks][kPackElems];
          load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, kInput>(
              &input_ptr, 1, 0, r, slice, input_stride_m, input_stride_n, WORLD, input_size_m,
              input_size_n, input_size_m, len, x, offs_m, g0 * kInt4Group, 1);
          // QUANTIZED ONCE, by this thread alone (its group), then sent as the payload.
          uint32_t words[kInt4GroupPacks];
          float scale;
          quantize(x[0], words, scale);
          uint32_t payload[1][kInt4GroupPacks][2];
#pragma unroll
          for (int k = 0; k < kInt4GroupPacks; ++k) {
            payload[0][k][0] = words[k];
            payload[0][k][1] = __builtin_bit_cast(uint32_t, scale);
          }
          send<WORLD, 1, THREADS_PER_BLOCK, QUANTIZATION, Layout::group_run>(
              payload, staging_ptr_array, staging_ptr_array_size, r, rank, slice,
              staging_stride_m, staging_stride_n, WORLD, staging_size_m, staging_size_n, 1,
              min(g_end, len / kInt4Group) * kInt4Group, 0, g0 * kInt4Group, 1);
        }
      }
      barrier<WORLD>(signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
                              signals_stride_block, signals_stride_writer, signals_size_parity,
                              signals_size_block, signals_size_writer, status_ptr,
                              status_stride_field, status_size_field, rank, timeout_ticks, UNTIL, 0);
      // 2. THE SUM, LOCAL: this rank's share, every writer's pushed group read from this rank's own
      // inbox (its slots, rotated from this rank's), all in flight, summed in fp32 from the words,
      // quantized and stored in this rank's partials.
      const int len = share_len(rank);
      if (len / kInt4Group != 0) {  // a late rank's share is empty at few groups (every block alike)
        for (int g0 = g_first; g0 < g_end; g0 += THREADS_PER_BLOCK) {
          uint32_t words[WORLD][1][kInt4GroupPacks][2];
          load<WORLD, 1, THREADS_PER_BLOCK, QUANTIZATION, Layout::group_run>(
              staging_ptr_array, staging_ptr_array_size, rank, rank, slice, staging_stride_m,
              staging_stride_n, WORLD, staging_size_m, staging_size_n, 1, len, words, 0,
              g0 * kInt4Group, 1);
          // EVERY WRITER'S GROUP SUMMED FROM THE WORDS, in fp32 in writer order: each element (q - 8) *
          // scale landed in DTYPE, as a quantized load lands it, then added. No pack of every writer decoded
          // first: decoded whole, the push kernels held 172 VGPRs (102 unquantized; ISA
          // 2026-10-15T22-47-01Z). WITH BF16, TWO WRITERS' TERMS A STEP: landed together (one
          // v_cvt_pk_bf16_f32) and added by one v_dot2c_f32_bf16 against (1, 1), where each was a
          // conversion, a shift and an add. v_dot2_f32_bf16 IS GFX950'S (dot12-insts), the
          // one arch built (CMakeLists.txt).
          const auto scale_of = [&](int w) { return __builtin_bit_cast(float, words[w][0][0][1]); };
          float sum[1][kInt4GroupPacks][kPackElems];
#pragma unroll
          for (int k = 0; k < kInt4GroupPacks; ++k) {
#pragma unroll
            for (int j = 0; j < kPackElems; ++j) sum[0][k][j] = 0.0f;
            int w = 0;
            if constexpr (kBf16<DTYPE>) {
              const bf16x2 ones = __builtin_bit_cast(bf16x2, 0x3f803f80u);
#pragma unroll
              for (; w + 1 < WORLD; w += 2) {
                float a[kPackElems], b[kPackElems];
                dequantize(words[w][0][k][0], scale_of(w), a);
                dequantize(words[w + 1][0][k][0], scale_of(w + 1), b);
#pragma unroll
                for (int j = 0; j < kPackElems; ++j)
                  sum[0][k][j] = __builtin_amdgcn_fdot2_f32_bf16(
                      __builtin_convertvector(f32x2{a[j], b[j]}, bf16x2), ones, sum[0][k][j], false);
                // THE SUMS DONE BEFORE THE NEXT DEQUANTIZE'S VOLATILE ASM: the scheduler, free to use 256
                // VGPRs at 512 threads, decoded every writer's words of a group ahead of the first sum and
                // spilled (the plain kernel 256 VGPRs, 264 B of partials: ISA 2026-10-16T02-22-15Z).
#pragma unroll
                for (int j = 0; j < kPackElems; ++j) asm volatile("" ::"v"(sum[0][k][j]));
              }
            }
#pragma unroll
            for (; w < WORLD; ++w) {
              float a[kPackElems];
              dequantize(words[w][0][k][0], scale_of(w), a);
#pragma unroll
              for (int j = 0; j < kPackElems; ++j) sum[0][k][j] += static_cast<float>(static_cast<DTYPE>(a[j]));
#pragma unroll
              for (int j = 0; j < kPackElems; ++j) asm volatile("" ::"v"(sum[0][k][j]));  // as above
            }
          }
          store<WORLD, 1, THREADS_PER_BLOCK, QUANTIZATION, Layout::group_run>(
              sum, partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m,
              partials_stride_n, 1, partials_size_m, partials_size_n, 1,
              min(g_end, len / kInt4Group) * kInt4Group, 0, g0 * kInt4Group, 1);
        }
      }
      barrier<WORLD>(signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
                              signals_stride_block, signals_stride_writer, signals_size_parity,
                              signals_size_block, signals_size_writer, status_ptr,
                              status_stride_field, status_size_field, rank, timeout_ticks, Until::visible, 1);
    } else {
      static_assert(ACCUMULATE == Accumulate::lds, "the exact flat walk sums through LDS");
      constexpr Layout kInput = CONTIGUOUS ? Layout::run : Layout::flat;
      constexpr int kLanes = THREADS_PER_BLOCK / WORLD;
      constexpr int kPacks = TILE_N / (kPackElems * kLanes);
      static_assert(kLanes * WORLD == THREADS_PER_BLOCK, "a block is a group of threads a rank");
      static_assert(kLanes % kWaveSize == 0, "a rank's copy is whole waves");
      static_assert(kPacks * kPackElems * kLanes == TILE_N, "a chunk is whole packs a thread");
      const int chunks = (slice + TILE_N - 1) / TILE_N;
      if constexpr (DIRECTION == Direction::push) {
        // 1. THE PUSH: this rank's input, every rank's share of it, read here (no peer reads the
        // caller's input) and sent to THAT rank's inbox at this rank's slot, each block the
        // chunks that rank's same block sums (the same walk). A row of threads a rank.
        for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
          const int r = (rank + static_cast<int>(threadIdx.x) / kLanes) % WORLD;
          const int len = share_len(r);
          if (len == 0) continue;
          DTYPE part[1][kPacks][kPackElems];
          load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, kInput>(
              &input_ptr, 1, 0, r, slice, input_stride_m, input_stride_n, WORLD, input_size_m,
              input_size_n, input_size_m, len, part, offs_m, chunk * TILE_N, 1);
          send<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::run>(
              part, staging_ptr_array, staging_ptr_array_size, r, rank, staging_stride_m,
              staging_stride_m, staging_stride_n, WORLD, staging_size_m, staging_size_n, 1, len, 0,
              chunk * TILE_N, 1);
        }
      }
      barrier<WORLD>(signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
                              signals_stride_block, signals_stride_writer, signals_size_parity,
                              signals_size_block, signals_size_writer, status_ptr,
                              status_stride_field, status_size_field, rank, timeout_ticks, UNTIL, 0);
      // 2. THE SUM: this rank's share, a TILE_N chunk a step, every rank's copy a row of threads
      // (pull: rank w's input, received, at this rank's share; push: writer w's slot of this
      // rank's inbox, loaded) in one round trip, summed over the ranks in rotated order through
      // LDS, rounded once to DTYPE, stored in this rank's partials.
      const int len = share_len(rank);
      if (len != 0) {  // a late rank's share is empty at few packs (every block alike)
        for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
          const int w = (rank + static_cast<int>(threadIdx.x) / kLanes) % WORLD;
          DTYPE got[1][kPacks][kPackElems];
          if constexpr (DIRECTION == Direction::push)
            load<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, Layout::run>(
                staging_ptr_array, staging_ptr_array_size, rank, w, staging_stride_m,
                staging_stride_m, staging_stride_n, WORLD, staging_size_m, staging_size_n, 1, len,
                got, 0, chunk * TILE_N, 1);
          else
            recv<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, kInput>(
                input_ptr_array, input_ptr_array_size, w, rank, slice, input_ptr_array_stride_m,
                input_ptr_array_stride_n, WORLD, input_size_m, input_size_n, input_size_m, len, got,
                offs_m, chunk * TILE_N, 1);
          DTYPE rounded[1][kPacks][kPackElems];
          reduce<WORLD, THREADS_PER_BLOCK, ReduceAxis::m>(got[0], rounded[0]);
          store<WORLD, WORLD, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(rounded, partials_ptr_array, partials_ptr_array_size, rank, 0, 0,
                                   0, 1, 1, 1, len, 1, len, 0, chunk * TILE_N, 1);
        }
      }
      barrier<WORLD>(signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
                              signals_stride_block, signals_stride_writer, signals_size_parity,
                              signals_size_block, signals_size_writer, status_ptr,
                              status_stride_field, status_size_field, rank, timeout_ticks, Until::visible, 1);
    }
  } else {
    // THIS RANK'S COLUMNS, in whole packs (a late rank's short or empty), and this block's rows,
    // worked out before the first barrier, in scalar registers: after it they were vector
    // arithmetic (a division by the grid a lane; ISA 2026-10-15T18-38-16Z).
    const int packs = input_size_n / kPackElems;  // the row, in packs (n is whole packs)
    const int slice = (packs + WORLD - 1) / WORLD;
    const int col0 = min(rank * slice, packs);
    const int own_packs = max(0, min(slice, packs - col0));
    const int end = (col0 + own_packs) * kPackElems;  // this rank's columns end, in elements
    [[maybe_unused]] const int block_size_m =
        input_size_m > static_cast<int>(blockIdx.x)
            ? (input_size_m - blockIdx.x + gridDim.x - 1) / gridDim.x
            : 0;
    // 1. int4: this block's rows of this rank's input, whole, quantized into its staging's INT4
    // payload at the same place, kTiles tiles a step, every load in flight before the first store.
    if constexpr (kInt4) {
      constexpr int kTiles = TILE_M < 4 ? 4 / TILE_M : 1;
      const int tiles = (input_size_m + TILE_M - 1) / TILE_M;
      for (int t = blockIdx.x; t < tiles; t += kTiles * gridDim.x)
        for (int c = 0; c < input_size_n; c += THREADS_PER_BLOCK * kPackElems) {
          DTYPE x[kTiles][TILE_M][1][kPackElems];
#pragma unroll
          for (int s = 0; s < kTiles; ++s)
            load<WORLD, TILE_M, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(&input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1,
                             input_size_m, input_size_n, input_size_m, input_size_n, x[s],
                             (t + s * static_cast<int>(gridDim.x)) * TILE_M, c, 1);
#pragma unroll
          for (int s = 0; s < kTiles; ++s)
            store<WORLD, TILE_M, THREADS_PER_BLOCK, QUANTIZATION, Layout::tile>(
                x[s], staging_ptr_array, staging_ptr_array_size, rank, 0, 0, staging_stride_m,
                staging_stride_n, 1, staging_size_m, staging_size_n, input_size_m, input_size_n,
                (t + s * static_cast<int>(gridDim.x)) * TILE_M, c, 1);
        }
    }
    barrier<WORLD>(signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
                              signals_stride_block, signals_stride_writer, signals_size_parity,
                              signals_size_block, signals_size_writer, status_ptr,
                              status_stride_field, status_size_field, rank, timeout_ticks, UNTIL, 0);
    if constexpr (ACCUMULATE == Accumulate::lds) {
      static_assert(DIRECTION == Direction::pull, "the rows walk through LDS pulls");
      constexpr int kLanes = THREADS_PER_BLOCK / WORLD;
      static_assert(kLanes * WORLD == THREADS_PER_BLOCK, "a block is a row of threads a rank");
      using Summed = std::conditional_t<kInt4, float, DTYPE>;
      const int tiles = (input_size_m + TILE_M - 1) / TILE_M;
      for (int t = blockIdx.x; t < tiles; t += gridDim.x)
        for (int row = t * TILE_M; row < min(t * TILE_M + TILE_M, input_size_m); ++row)
          for (int c = col0 * kPackElems; c < end; c += kLanes * kPackElems) {
            // 2. Every rank's copy of the chunk, a row of threads each, in one round trip.
            const int r = (rank + static_cast<int>(threadIdx.x) / kLanes) % WORLD;
            DTYPE got[1][1][kPackElems];
            if constexpr (!kInt4)
              recv<WORLD, WORLD, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(input_ptr_array, input_ptr_array_size, r, 0, 0, input_ptr_array_stride_m, input_ptr_array_stride_n, 1, input_size_m, input_size_n, input_size_m, end, got, row, c, 0);
            else
              recv<WORLD, WORLD, THREADS_PER_BLOCK, QUANTIZATION, Layout::tile>(staging_ptr_array, staging_ptr_array_size, r, 0, 0, staging_stride_m, staging_stride_n, 1, staging_size_m, staging_size_n, input_size_m, end, got, row, c, 0);
            // 3. Summed over the rows (the ranks), in row order, through LDS, rounded once.
            Summed rounded[1][1][kPackElems];
            reduce<WORLD, THREADS_PER_BLOCK, ReduceAxis::m>(got[0], rounded[0]);
            // 4. Stored in this rank's slice.
            store<WORLD, WORLD, THREADS_PER_BLOCK, QUANTIZATION, Layout::tile>(
                rounded, partials_ptr_array, partials_ptr_array_size, rank, 0, 0,
                partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n,
                row + 1, end, row, c, 1);
          }
    } else {
      constexpr int kSliceWaves = (TILE_N / WORLD / kPackElems + kWaveSize - 1) / kWaveSize * kWaveSize;
      constexpr int kThreadsN = kSliceWaves < THREADS_PER_BLOCK ? kSliceWaves : THREADS_PER_BLOCK;
      constexpr int kThreadsM = THREADS_PER_BLOCK / kThreadsN;  // rows of threads at once
      using Summed = std::conditional_t<kInt4, float, DTYPE>;
      const int offs_n = col0 * kPackElems;
      const int size_n = own_packs * kPackElems;
      if (size_n > 0)  // a late rank owns no columns
        for (int q = 0; q < block_size_m; q += kThreadsM)
          for (int c = offs_n; c < offs_n + size_n; c += kThreadsN * kPackElems) {
            const int row = static_cast<int>(blockIdx.x + q * gridDim.x);
            Summed rounded[1][1][kPackElems];
            if constexpr (!kInt4) {
              DTYPE peers[WORLD][1][1][kPackElems];
              for (int r = 0; r < WORLD; ++r)
                recv<WORLD, kThreadsM, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(input_ptr_array, input_ptr_array_size, r, 0, 0, input_ptr_array_stride_m, input_ptr_array_stride_n, 1, input_size_m, input_size_n, input_size_m, offs_n + size_n, peers[r], row, c, static_cast<int>(gridDim.x));
              // Summed in rank order in fp32, so every rank's sum agrees bitwise, rounded once.
#pragma unroll
              for (int j = 0; j < kPackElems; ++j) {
                float acc = static_cast<float>(peers[0][0][0][j]);
#pragma unroll
                for (int r = 1; r < WORLD; ++r) acc += static_cast<float>(peers[r][0][0][j]);
                rounded[0][0][j] = static_cast<Summed>(acc);
              }
            } else {
              // INT4'S SUM IS FROM THE WORDS (the flat walk's: a decoded tile of every peer held 172
              // VGPRs): every rank's word and scale of this thread's pack received raw, all in flight.
              uint32_t raw[WORLD][1][1][2];
              for (int r = 0; r < WORLD; ++r)
                recv<WORLD, kThreadsM, THREADS_PER_BLOCK, QUANTIZATION, Layout::tile>(
                    staging_ptr_array, staging_ptr_array_size, r, 0, 0, staging_stride_m,
                    staging_stride_n, 1, staging_size_m, staging_size_n, input_size_m,
                    offs_n + size_n, raw[r], row, c, static_cast<int>(gridDim.x));
              const auto peer_words = [&](int r) { return raw[r][0][0][0]; };
              const auto peer_scales = [&](int r) { return __builtin_bit_cast(float, raw[r][0][0][1]); };
              // Every rank's pack summed from the words, in fp32 in rank order (the flat walk's sum above).
              float sum[kPackElems];
#pragma unroll
              for (int j = 0; j < kPackElems; ++j) sum[j] = 0.0f;
              int w = 0;
              if constexpr (kBf16<DTYPE>) {
                const bf16x2 ones = __builtin_bit_cast(bf16x2, 0x3f803f80u);
#pragma unroll
                for (; w + 1 < WORLD; w += 2) {
                  float a[kPackElems], b[kPackElems];
                  dequantize(peer_words(w), peer_scales(w), a);
                  dequantize(peer_words(w + 1), peer_scales(w + 1), b);
#pragma unroll
                  for (int j = 0; j < kPackElems; ++j)
                    sum[j] = __builtin_amdgcn_fdot2_f32_bf16(
                        __builtin_convertvector(f32x2{a[j], b[j]}, bf16x2), ones, sum[j], false);
#pragma unroll
                  for (int j = 0; j < kPackElems; ++j) asm volatile("" ::"v"(sum[j]));
                }
              }
#pragma unroll
              for (; w < WORLD; ++w) {
                float a[kPackElems];
                dequantize(peer_words(w), peer_scales(w), a);
#pragma unroll
                for (int j = 0; j < kPackElems; ++j) sum[j] += static_cast<float>(static_cast<DTYPE>(a[j]));
#pragma unroll
                for (int j = 0; j < kPackElems; ++j) asm volatile("" ::"v"(sum[j]));
              }
#pragma unroll
              for (int j = 0; j < kPackElems; ++j) rounded[0][0][j] = sum[j];

            }
            // Push: sent to every rank's partials, the stores issued together (int4: quantized
            // once); pull: stored in this rank's.
            if constexpr (DIRECTION == Direction::push) {
                // INT4: quantized ONCE, then the same payload sent to every rank.
                [[maybe_unused]] uint32_t payload[1][1][2];
                if constexpr (kInt4) {
                  uint32_t word[1];
                  float scale;
                  quantize(rounded[0], word, scale);
                  payload[0][0][0] = word[0];
                  payload[0][0][1] = __builtin_bit_cast(uint32_t, scale);
                }
                for (int r = 0; r < WORLD; ++r) {
                  if constexpr (kInt4)
                    send<WORLD, kThreadsM, THREADS_PER_BLOCK, QUANTIZATION, Layout::tile>(
                        payload, partials_ptr_array, partials_ptr_array_size, r, 0, 0,
                        partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n,
                        input_size_m, offs_n + size_n, row, c, static_cast<int>(gridDim.x));
                  else
                    send<WORLD, kThreadsM, THREADS_PER_BLOCK, QUANTIZATION, Layout::tile>(
                        rounded, partials_ptr_array, partials_ptr_array_size, r, 0, 0,
                        partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n,
                        input_size_m, offs_n + size_n, row, c, static_cast<int>(gridDim.x));
                }
            } else
              store<WORLD, kThreadsM, THREADS_PER_BLOCK, QUANTIZATION, Layout::tile>(
                  rounded, partials_ptr_array, partials_ptr_array_size, rank, 0, 0,
                  partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n,
                  input_size_m, offs_n + size_n, row, c, static_cast<int>(gridDim.x));
          }
    }
    barrier<WORLD>(signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
                              signals_stride_block, signals_stride_writer, signals_size_parity,
                              signals_size_block, signals_size_writer, status_ptr,
                              status_stride_field, status_size_field, rank, timeout_ticks, Until::visible, 1);
  }
}

}  // namespace hip_custom_communicator
