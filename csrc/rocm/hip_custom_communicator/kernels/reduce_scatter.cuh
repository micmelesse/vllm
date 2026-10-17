// SPDX-License-Identifier: MIT
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE REDUCE-SCATTER, written once with the common ops (load, store, barrier, block_reduce, the
// codec): from every rank's input to every rank's sums in the partials, visible to its peers. Its
// variations are template parameters, its walks written inline (common.cuh's COLLECTIVES says why
// each walk is the one it is).

#pragma once

#include "common.cuh"

namespace hip_custom_communicator {

// [PUSH OR STAGE] BARRIER(UNTIL, 0) SUM BARRIER(visible, 1): before the first barrier, a pushed
// walk stores every rank's share of this rank's input in that rank's staging at this rank's slot,
// and an int4 rows walk stages this rank's rows quantized in its own staging (the rows this
// block's twin on every rank reads, tiles of TILE_M rows blockIdx.x, blockIdx.x + gridDim.x, ...);
// after the second, every rank's sums are in the partials and every peer is done reading.
//
// WALK::flat (the plain two-shot): rows offs_m.. offs_m + input_size_m of the [*, input_size_n]
// problem numbered row-major, each rank's SHARE one contiguous range of them, a TILE_N chunk a step,
// a rank's copy of a chunk LANES = THREADS / WORLD threads (whole waves, its rank rank + w,
// rotated). PULL (ACCUMULATE lds, none): every rank's input read in place through the INPUT TABLE
// (input_ptr_array at its strides). PUSH (none, lds): this rank's input (input_ptr) pushed into
// every rank's staging inbox, [WORLD, slot] (writer w's slot at row w, staging_stride_m elements a
// slot), then the slots summed locally. PUSH (int4, registers): A GROUP A THREAD, each block a
// contiguous range of every share's groups (ceil(groups / grid), THREADS a step), quantized here
// into every rank's staging INT4 payload at this rank's place (writer w's at share w's), the
// writers' groups summed from the words (int4_peers_sum) and quantized into this rank's partials.
// The plain all-reduce at 72 blocks: 71.8 / 109.0 / 178.2 / 286.6 us at 1024-8192 tokens of
// 7168, against the exact 75.5 / 142.0 / 273.3 / 535.4 (2026-10-16T03-14-06Z).
//
// WALK::rows (the fused ops' two-shots): this rank's columns of every row, equal slices of whole
// packs a rank (a late rank's short or empty). ACCUMULATE lds (pull): a chunk of a row at a time,
// WORLD rows of LANES threads, row w rank rank + w's copy picked per lane (Ranks::lane), summed
// over the rows through LDS into this rank's partials. ACCUMULATE registers: a row of threads as
// wide as a rank's slice of a TILE_N row (whole waves), a pack a thread, as many rows at once as the
// block holds, every rank's pack summed in registers in rank order; push stores the sum in every
// rank's partials, pull in this rank's. none reads every rank's input in place; int4 every rank's
// staged INT4 payload, an INT4 payload's layout from its own buffer's sizes.
template <int WORLD, int THREADS, int TILE_M, int TILE_N, Walk WALK, Accumulate ACCUMULATE,
          Direction DIRECTION, int QUANTIZATION, bool CONTIGUOUS, Until UNTIL, typename T>
DINLINE void reduce_scatter(
    const T* input_ptr, int64_t input_stride_m, int64_t input_stride_n, int input_size_m,
    int input_size_n, const T* const* input_ptr_array, int input_ptr_array_size,
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
  constexpr int P = traits<T>::N;
  constexpr bool kInt4 = QUANTIZATION == Quantization::int4;
  using V = vec<T, P>;
  // Element e of an [*, n] buffer numbered row-major (the flat walk's address).
  const auto element_at = [](auto* ptr, int64_t stride_m, int64_t stride_n, int64_t e, int n) {
    if (stride_m == int64_t{n} * stride_n) return ptr + e * stride_n;
    const int64_t row = e < (int64_t{1} << 31)
                            ? int64_t{static_cast<uint32_t>(e) / static_cast<uint32_t>(n)}
                            : e / n;
    return ptr + row * stride_m + (e - row * n) * stride_n;
  };

  if constexpr (WALK == Walk::flat) {
    // THE SHARES of the band's elements, in whole packs (int4: whole groups), the last rank's short.
    constexpr int kUnit = kInt4 ? kInt4Group : P;
    const int64_t total = int64_t{input_size_m} * input_size_n;
    const int slice = static_cast<int>((total / kUnit + WORLD - 1) / WORLD * kUnit);
    const auto share_len = [&](int r) {
      return static_cast<int>(max(int64_t{0}, min(int64_t{slice}, total - int64_t{r} * slice)));
    };
    if constexpr (kInt4) {
      static_assert(DIRECTION == Direction::push && ACCUMULATE == Accumulate::registers,
                    "the int4 flat walk pushes and sums in registers");
      const int groups_all = slice / kInt4Group;
      const int grid = static_cast<int>(gridDim.x);
      const int per_block = (groups_all + grid - 1) / grid;
      const int g_first = static_cast<int>(blockIdx.x) * per_block;
      const int g_end = min(g_first + per_block, groups_all);
      // 1. THE PUSH: every rank's share of this rank's input, a group a thread, quantized here and
      // stored in that rank's staging payload at this rank's slot. TWO RANKS' GROUPS IN FLIGHT:
      // unrolled whole, every rank's 4 loads were hoisted together, 128 VGPRs of input, spilling.
      const T* const band_input = input_ptr + offs_m * input_stride_m;
      const int64_t mine = int64_t{rank} * slice;  // this rank's slot in every inbox
      for (int g0 = g_first; g0 < g_end; g0 += THREADS) {
        const int g = g0 + static_cast<int>(threadIdx.x);
#pragma unroll 2
        for (int w = 0; w < WORLD; ++w) {
          const int r = (rank + w) % WORLD;
          const int groups = share_len(r) / kInt4Group;
          if (groups == 0) continue;  // a late rank's empty share (every block alike)
          const int64_t first = int64_t{r} * slice;
          const int at = min(g, groups - 1);
          const V* group;
          if constexpr (CONTIGUOUS)
            group = reinterpret_cast<const V*>(
                at_offset(band_input + first * input_stride_n,
                          static_cast<uint32_t>(at) * (kInt4Group * sizeof(T))));
          else
            group = reinterpret_cast<const V*>(element_at(band_input, input_stride_m,
                                                          input_stride_n,
                                                          first + int64_t{at} * kInt4Group,
                                                          input_size_n));
          T x[kInt4GroupPacks][P];
#pragma unroll
          for (int k = 0; k < kInt4GroupPacks; ++k) held(x[k], pack_load(group + k));
          uint32_t words[kInt4GroupPacks];
          float scale;
          encode<kInt4GroupPacks>(x, words, scale);
          if (g < g_end && g < groups) {
            uint8_t* const values = ptr_array_at<uint8_t>(staging_ptr_array, staging_ptr_array_size, r) + mine / 2;
            __fp16* const scales =
                reinterpret_cast<__fp16*>(ptr_array_at(staging_ptr_array, staging_ptr_array_size, r) +
                                          int4_scales_offset(staging_size_m, staging_size_n)) +
                mine / kInt4Group;
            *(global_u32x4*)at_offset(values, static_cast<uint32_t>(g) * 16) =
                u32x4{words[0], words[1], words[2], words[3]};
            *(global_f16*)at_offset(scales, static_cast<uint32_t>(g) * kInt4ScaleBytes) =
                static_cast<__fp16>(scale);
          }
        }
      }
      barrier<UNTIL, 0, WORLD>(signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
                              signals_stride_block, signals_stride_writer, signals_size_parity,
                              signals_size_block, signals_size_writer, status_ptr,
                              status_stride_field, status_size_field, rank, timeout_ticks);
      // 2. THE SUM, LOCAL: this rank's share, every writer's pushed group read from this rank's own
      // inbox, all in flight (rotated writer order), summed in fp32 from the words, quantized and
      // stored in this rank's partials.
      const int groups = share_len(rank) / kInt4Group;
      if (groups != 0) {  // a late rank's share is empty at few groups (every block alike)
        const uint8_t* values[WORLD];
        const __fp16* scales[WORLD];
#pragma unroll
        for (int w = 0; w < WORLD; ++w) {
          const int64_t slot = int64_t{(rank + w) % WORLD} * slice;
          values[w] = ptr_array_at<uint8_t>(staging_ptr_array, staging_ptr_array_size, rank) + slot / 2;
          scales[w] = reinterpret_cast<__fp16*>(ptr_array_at(staging_ptr_array, staging_ptr_array_size, rank) +
                                                int4_scales_offset(staging_size_m, staging_size_n)) +
                      slot / kInt4Group;
        }
        uint8_t* const own_values = ptr_array_at<uint8_t>(partials_ptr_array, partials_ptr_array_size, rank);
        __fp16* const own_scales = reinterpret_cast<__fp16*>(
            ptr_array_at(partials_ptr_array, partials_ptr_array_size, rank) +
            int4_scales_offset(partials_size_m, partials_size_n));
        for (int g0 = g_first; g0 < g_end; g0 += THREADS) {
          const int g = g0 + static_cast<int>(threadIdx.x);
          const int at = min(g, groups - 1);
          uint32_t words[WORLD][1][kInt4GroupPacks];
          float group_scales[WORLD][1][kInt4GroupPacks];
#pragma unroll
          for (int w = 0; w < WORLD; ++w) {
            const u32x4 v = *(const global_u32x4*)at_offset(values[w], static_cast<uint32_t>(at) * 16);
            const float scale = *(const global_f16*)at_offset(
                scales[w], static_cast<uint32_t>(at) * kInt4ScaleBytes);
#pragma unroll
            for (int k = 0; k < kInt4GroupPacks; ++k) {
              words[w][0][k] = v[k];
              group_scales[w][0][k] = scale;
            }
          }
          issued();
          float sum[1][kInt4GroupPacks][P];
          int4_peers_sum<T>(words, group_scales, sum);
          uint32_t out[kInt4GroupPacks];
          float scale;
          encode<kInt4GroupPacks>(sum[0], out, scale);
          if (g < g_end && g < groups) {
            *(global_u32x4*)at_offset(own_values, static_cast<uint32_t>(g) * 16) =
                u32x4{out[0], out[1], out[2], out[3]};
            *(global_f16*)at_offset(own_scales, static_cast<uint32_t>(g) * kInt4ScaleBytes) =
                static_cast<__fp16>(scale);
          }
        }
      }
      barrier<Until::visible, 1, WORLD>(signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
                              signals_stride_block, signals_stride_writer, signals_size_parity,
                              signals_size_block, signals_size_writer, status_ptr,
                              status_stride_field, status_size_field, rank, timeout_ticks);
    } else {
      static_assert(ACCUMULATE == Accumulate::lds, "the exact flat walk sums through LDS");
      constexpr int LANES = THREADS / WORLD;
      constexpr int K = TILE_N / (P * LANES);
      static_assert(LANES * WORLD == THREADS, "a block is a group of threads a rank");
      static_assert(LANES % kWaveSize == 0, "a rank's copy is whole waves");
      static_assert(K * P * LANES == TILE_N, "a chunk is whole packs a thread");
      const int chunks = (slice + TILE_N - 1) / TILE_N;
      if constexpr (DIRECTION == Direction::push) {
        // 1. THE PUSH: this rank's input, every rank's share of it, read here (no peer reads the
        // caller's input) and stored in THAT rank's inbox at this rank's slot, each block the
        // chunks that rank's same block sums (the same walk).
        const T* const band_input = input_ptr + offs_m * input_stride_m;
        block_for_each_tile(1, chunks, [&](int, int chunk) {
          const int r = (rank + static_cast<int>(threadIdx.x) / LANES) % WORLD;
          const int len = share_len(r);
          if (len == 0) return;
          const int64_t first = int64_t{r} * slice;
          T part[K][P];
          if constexpr (CONTIGUOUS) {
            // The last pack, the run's start, then the column: the order the two-shot's
            // instructions came out in (ISA 2026-10-15).
#pragma unroll
            for (int k = 0; k < K; ++k) {
              const int last = len / P - 1;
              const V* run = reinterpret_cast<const V*>(band_input + first * input_stride_n);
              const int n = chunk * TILE_N / P + static_cast<int>(threadIdx.x) % LANES + k * LANES;
              const int col = n < len / P ? n : last;
              uint32_t offset = static_cast<uint32_t>(col) * uint32_t{sizeof(V)};
              asm volatile("" : "+v"(offset));
              held(part[k], pack_load(reinterpret_cast<const V*>(
                                reinterpret_cast<const char*>(run) + offset)));
            }
          } else {
#pragma unroll
            for (int k = 0; k < K; ++k)
              held(part[k], pack_load(reinterpret_cast<const V*>(element_at(
                                band_input, input_stride_m, input_stride_n,
                                first + int64_t{pack_of<P, LANES>(len, chunk * TILE_N, k)} * P,
                                input_size_n))));
          }
          issued();
          V* const inbox = reinterpret_cast<V*>(
              ptr_array_at<T>(staging_ptr_array, staging_ptr_array_size, r) + rank * staging_stride_m);
#pragma unroll
          for (int k = 0; k < K; ++k) {
            const int c = chunk * TILE_N / P + static_cast<int>(threadIdx.x) % LANES + k * LANES;
            if (c >= len / P) continue;
            pack_store(at_col<P, LANES>(inbox, len, chunk * TILE_N, k), pack(part[k]));
          }
        });
      }
      barrier<UNTIL, 0, WORLD>(signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
                              signals_stride_block, signals_stride_writer, signals_size_parity,
                              signals_size_block, signals_size_writer, status_ptr,
                              status_stride_field, status_size_field, rank, timeout_ticks);
      // 2. THE SUM: this rank's share, a TILE_N chunk a step, every rank's copy (pull: its input
      // through the table, from this rank's first element; push: writer w's slot of this rank's
      // inbox) in one round trip, summed over the ranks in rotated order through LDS, rounded once
      // to T, stored in this rank's partials.
      const int len = share_len(rank);
      if (len != 0) {  // a late rank's share is empty at few packs (every block alike)
        [[maybe_unused]] const T* const inbox =
            ptr_array_at<T>(staging_ptr_array, staging_ptr_array_size, rank);
        constexpr bool kRun = DIRECTION == Direction::push || CONTIGUOUS;
        const int64_t stride_m = DIRECTION == Direction::push ? staging_stride_m : input_ptr_array_stride_m;
        const int64_t stride_n = DIRECTION == Direction::push ? staging_stride_n : input_ptr_array_stride_n;
        const int64_t first = DIRECTION == Direction::push ? 0 : int64_t{rank} * slice;
        block_for_each_tile(1, chunks, [&](int, int chunk) {
          const int w = (rank + static_cast<int>(threadIdx.x) / LANES) % WORLD;
          const T* from;
          if constexpr (DIRECTION == Direction::push)
            from = inbox + w * staging_stride_m;
          else
            from = ptr_array_at(input_ptr_array, input_ptr_array_size, w) +
                   offs_m * input_ptr_array_stride_m;
          T got[1][K][P];
          if constexpr (kRun) {
#pragma unroll
            for (int k = 0; k < K; ++k) {
              const int last = len / P - 1;
              const V* run = reinterpret_cast<const V*>(from + first * stride_n);
              const int n = chunk * TILE_N / P + static_cast<int>(threadIdx.x) % LANES + k * LANES;
              const int col = n < len / P ? n : last;
              uint32_t offset = static_cast<uint32_t>(col) * uint32_t{sizeof(V)};
              asm volatile("" : "+v"(offset));
              held(got[0][k], pack_load(reinterpret_cast<const V*>(
                                  reinterpret_cast<const char*>(run) + offset)));
            }
          } else {
#pragma unroll
            for (int k = 0; k < K; ++k)
              held(got[0][k], pack_load(reinterpret_cast<const V*>(element_at(
                                  from, stride_m, stride_n,
                                  first + int64_t{pack_of<P, LANES>(len, chunk * TILE_N, k)} * P,
                                  input_size_n))));
          }
          issued();
          T rounded[1][K][P];
          block_reduce<Sum, ReduceAxis::m, WORLD, LANES>(got[0], rounded[0]);
          store<1, LANES, THREADS>(rounded, partials_ptr_array, partials_ptr_array_size, rank, 0, 0,
                                   0, 1, 1, 1, len, 1, len, 0, chunk * TILE_N, 1);
        });
      }
      barrier<Until::visible, 1, WORLD>(signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
                              signals_stride_block, signals_stride_writer, signals_size_parity,
                              signals_size_block, signals_size_writer, status_ptr,
                              status_stride_field, status_size_field, rank, timeout_ticks);
    }
  } else {
    // THIS RANK'S COLUMNS, in whole packs (a late rank's short or empty), and this block's rows,
    // worked out before the first barrier, in scalar registers: after it they were vector
    // arithmetic (a division by the grid a lane; ISA 2026-10-15T18-38-16Z).
    const int packs = input_size_n / P;  // the row, in packs (n is whole packs)
    const int slice = (packs + WORLD - 1) / WORLD;
    const int col0 = min(rank * slice, packs);
    const int own_packs = max(0, min(slice, packs - col0));
    const int end = (col0 + own_packs) * P;  // this rank's columns end, in elements
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
        for (int c = 0; c < input_size_n; c += THREADS * P) {
          T x[kTiles][TILE_M][1][P];
#pragma unroll
          for (int s = 0; s < kTiles; ++s)
            load<1, THREADS>(&input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1,
                             input_size_m, input_size_n, input_size_m, input_size_n, x[s],
                             (t + s * static_cast<int>(gridDim.x)) * TILE_M, c, 1);
#pragma unroll
          for (int s = 0; s < kTiles; ++s)
            store<1, THREADS, THREADS, Ranks::own, QUANTIZATION>(
                x[s], staging_ptr_array, staging_ptr_array_size, rank, 0, 0, staging_stride_m,
                staging_stride_n, 1, staging_size_m, staging_size_n, input_size_m, input_size_n,
                (t + s * static_cast<int>(gridDim.x)) * TILE_M, c, 1);
        }
    }
    barrier<UNTIL, 0, WORLD>(signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
                              signals_stride_block, signals_stride_writer, signals_size_parity,
                              signals_size_block, signals_size_writer, status_ptr,
                              status_stride_field, status_size_field, rank, timeout_ticks);
    if constexpr (ACCUMULATE == Accumulate::lds) {
      static_assert(DIRECTION == Direction::pull, "the rows walk through LDS pulls");
      constexpr int LANES = THREADS / WORLD;
      static_assert(LANES * WORLD == THREADS, "a block is a row of threads a rank");
      using Summed = std::conditional_t<kInt4, float, T>;
      const int tiles = (input_size_m + TILE_M - 1) / TILE_M;
      for (int t = blockIdx.x; t < tiles; t += gridDim.x)
        for (int row = t * TILE_M; row < min(t * TILE_M + TILE_M, input_size_m); ++row)
          for (int c = col0 * P; c < end; c += LANES * P) {
            // 2. Every rank's copy of the chunk, a row of threads each, in one round trip.
            const int r = (rank + static_cast<int>(threadIdx.x) / LANES) % WORLD;
            T got[1][1][P];
            if constexpr (!kInt4)
              load<1, LANES, Ranks::lane, Quantization::none, WORLD>(
                  input_ptr_array, input_ptr_array_size, r, 0, 0, input_ptr_array_stride_m,
                  input_ptr_array_stride_n, 1, input_size_m, input_size_n, input_size_m, end,
                  got, row, c, 1);
            else
              load<1, LANES, Ranks::lane, QUANTIZATION, WORLD>(
                  staging_ptr_array, staging_ptr_array_size, r, 0, 0, staging_stride_m,
                  staging_stride_n, 1, staging_size_m, staging_size_n, input_size_m, end, got,
                  row, c, 1);
            // 3. Summed over the rows (the ranks), in row order, through LDS, rounded once.
            Summed rounded[1][1][P];
            block_reduce<Sum, ReduceAxis::m, WORLD, LANES>(got[0], rounded[0]);
            // 4. Stored in this rank's slice.
            store<1, LANES, THREADS, Ranks::own, QUANTIZATION>(
                rounded, partials_ptr_array, partials_ptr_array_size, rank, 0, 0,
                partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n,
                row + 1, end, row, c, 1);
          }
    } else {
      constexpr int kSliceWaves = (TILE_N / WORLD / P + kWaveSize - 1) / kWaveSize * kWaveSize;
      constexpr int LANES = kSliceWaves < THREADS ? kSliceWaves : THREADS;
      constexpr int ROWS = THREADS / LANES;  // rows of threads at once
      constexpr Ranks kStoreRanks = DIRECTION == Direction::push ? Ranks::every : Ranks::own;
      using Summed = std::conditional_t<kInt4, float, T>;
      const int offs_n = col0 * P;
      const int size_n = own_packs * P;
      // INT4'S SUM IS FROM THE WORDS (int4_peers_sum: a decoded tile of every peer held 172
      // VGPRs), so its loads are the words and scales, every rank's at its staging's place.
      const uint8_t* input_values[WORLD];
      const __fp16* input_scales[WORLD];
      if constexpr (kInt4) {
#pragma unroll
        for (int r = 0; r < WORLD; ++r)
          input_values[r] = ptr_array_at<const uint8_t>(staging_ptr_array, staging_ptr_array_size, r);
#pragma unroll
        for (int r = 0; r < WORLD; ++r)
          input_scales[r] = reinterpret_cast<const __fp16*>(
              input_values[r] + int4_scales_offset(staging_size_m, staging_size_n));
      }
      if (size_n > 0)  // a late rank owns no columns
        for (int q = 0; q < block_size_m; q += ROWS)
          for (int c = offs_n; c < offs_n + size_n; c += LANES * P) {
            const int row = static_cast<int>(blockIdx.x + q * gridDim.x);
            float sum[1][1][P];
            if constexpr (!kInt4) {
              T peers[WORLD][1][1][P];
              load<ROWS, LANES, Ranks::every>(input_ptr_array, input_ptr_array_size, 0, 0, 0,
                                              input_ptr_array_stride_m, input_ptr_array_stride_n,
                                              1, input_size_m, input_size_n, input_size_m,
                                              offs_n + size_n, peers, row, c,
                                              static_cast<int>(gridDim.x));
              peers_sum(peers, sum);
            } else {
              uint32_t peer_words[WORLD][1][1];
              float peer_scales[WORLD][1][1];
              int4_peers_issue<ROWS, LANES, P>(input_values, int4_values_size_n(staging_size_n),
                                               input_scales, int4_scales_size_n(staging_size_n),
                                               input_size_m, offs_n + size_n, peer_words,
                                               peer_scales, row, c, static_cast<int>(gridDim.x));
              int4_peers_sum<T>(peer_words, peer_scales, sum);
            }
            Summed rounded[1][1][P];
            convert(sum, rounded);
            // Push: every rank's partials, the stores issued together (int4: quantized once);
            // pull: this rank's.
            store<ROWS, LANES, THREADS, kStoreRanks, QUANTIZATION, WORLD>(
                rounded, partials_ptr_array, partials_ptr_array_size, rank, 0, 0,
                partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n,
                input_size_m, offs_n + size_n, row, c, static_cast<int>(gridDim.x));
          }
    }
    barrier<Until::visible, 1, WORLD>(signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
                              signals_stride_block, signals_stride_writer, signals_size_parity,
                              signals_size_block, signals_size_writer, status_ptr,
                              status_stride_field, status_size_field, rank, timeout_ticks);
  }
}

}  // namespace hip_custom_communicator
