// SPDX-License-Identifier: MIT
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE REDUCE-SCATTER: every rank's input summed into its share, in its partials, visible to its
// peers. Written once and called by every two-shot (all_gather.cuh walks the same walk back).

#pragma once

#include "common.cuh"

namespace hip_custom_communicator {

// THE LOADS ARE THE COMMUNICATION: a peer's pointer is its memory opened over IPC, so a load
// reads over the links; the sum and the store are local. The peers are read rotated (rank + w the
// w-th), so at any moment the GPUs read different peers.
//
// THE FLAT WALK (the plain two-shots): the [m, n] elements numbered row-major, each rank's share
// one contiguous range of them, walked in TILE_N chunks, the grid striding over the chunks. Long
// runs from each peer are what the links reward (a row-sliced share was 2-7% slower). A rank's
// copy of a chunk is kLanes = THREADS_PER_BLOCK / WORLD threads; the block is WORLD rows of
// threads. A block reads in the all-gather what its twin on each rank wrote here.
//
// CONTIGUOUS (the input's rows back to back, so a share is one run) is a template parameter
// chosen on the host: a strided buffer divides each pack's row and column out (element_at).

// WHICH WAY THE SUM GOES: pull, this rank reads its peers and keeps its share's sum; push, each
// rank's share is stored into that rank's inbox, which it then sums locally.
enum class Direction : int { pull = 0, push = 1 };
// HOW THE COLLECTIVES WALK: flat (above), or rows, a rank's columns of every row (the fused ops).
// And where the copies are summed: through LDS or in registers.
enum class Walk : int { flat = 0, rows = 1 };
enum class Accumulate : int { lds = 0, registers = 1 };

// [PUSH OR STAGE] BARRIER(UNTIL, 0) SUM BARRIER(visible, 1). After the second barrier every
// rank's sums are in the partials and every peer is done reading.
//   flat, pull:           every rank's input read in place through the input table.
//   flat, push, exact:    this rank's input pushed into every rank's staging inbox
//                         ([WORLD, slot], writer w at row w), then the slots summed locally.
//   flat, push, int4:     each block an equal range of every share's groups (ceil(groups / grid)),
//                         quantized, pushed, summed from the words, quantized into the partials.
//   rows, lds (pull):     a chunk of a row at a time, rank w's copy a row of threads, summed
//                         through LDS into this rank's partials.
//   rows, registers:      a row of threads as wide as a rank's slice of a row, every rank's pack
//                         summed in registers; push stores the sum in every rank's partials.
template <int WORLD, int TILE_M, int TILE_N, int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES,
          bool QUANTIZE_INT4, bool CONTIGUOUS, Walk WALK, Accumulate ACCUMULATE, Direction DIRECTION,
          Until UNTIL, bool TAIL, typename DTYPE>
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
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
  constexpr bool kInt4 = QUANTIZE_INT4;
  if constexpr (WALK == Walk::flat) {
    // THE SHARES of the band's elements: rank r's is slot r of the band viewed flat.
    constexpr int kUnit = kInt4 ? kInt4Group : kPackElems;
    const int64_t total = int64_t{input_size_m} * input_size_n;
    // A share is whole packs (exact) or whole groups (int4), the last rank's to the band's end;
    // with TAIL its last group is short and masked.
    const int64_t units = TAIL ? (total + kUnit - 1) / kUnit : total / kUnit;
    const int slice = static_cast<int>((units + WORLD - 1) / WORLD * kUnit);
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
      // 1. THE PUSH: every rank's share of this rank's input, a group a thread, quantized and
      //    sent to that rank's inbox at slot [rank]. A row of threads a rank, so every rank's
      //    group is in flight at once (narrower than a wave: a lane's own rank).
      constexpr int kRankLanes = THREADS_PER_BLOCK / WORLD;
      static_assert(kRankLanes * WORLD == THREADS_PER_BLOCK, "a block is a row of threads a rank");
      const int r = (rank + static_cast<int>(threadIdx.x) / kRankLanes) % WORLD;
      const int len = share_len(r);
      if (TAIL ? len != 0 : len / kInt4Group != 0)  // a late rank's empty share (its row's)
        for (int g0 = g_first; g0 < g_end; g0 += kRankLanes) {
          DTYPE x[1][kInt4GroupPacks][kPackElems];
          load<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, false, kInput, TAIL>(
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
          send<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4, Layout::group_run, TAIL>(
              payload, staging_ptr_array, staging_ptr_array_size, r, rank, slice,
              staging_stride_m, staging_stride_n, WORLD, staging_size_m, staging_size_n, 1,
              TAIL ? min(g_end * kInt4Group, len) : min(g_end, len / kInt4Group) * kInt4Group, 0,
              g0 * kInt4Group, 1);
        }
      barrier<WORLD>(signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
                              signals_stride_block, signals_stride_writer, signals_size_parity,
                              signals_size_block, signals_size_writer, status_ptr,
                              status_stride_field, status_size_field, rank, timeout_ticks, UNTIL, 0);
      // 2. THE SUM, LOCAL: every writer's group of this share from this rank's inbox, summed in
      //    fp32 from the words, quantized into this rank's partials.
      const int own = share_len(rank);
      // A late rank's share is empty at few groups (every block alike).
      if (TAIL ? own != 0 : own / kInt4Group != 0) {
        for (int g0 = g_first; g0 < g_end; g0 += THREADS_PER_BLOCK) {
          uint32_t words[WORLD][1][kInt4GroupPacks][2];
          load<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4, Layout::group_run, TAIL>(
              staging_ptr_array, staging_ptr_array_size, rank, rank, slice, staging_stride_m,
              staging_stride_n, WORLD, staging_size_m, staging_size_n, 1, own, words, 0,
              g0 * kInt4Group, 1);
          // EVERY WRITER'S GROUP SUMMED FROM THE WORDS (dequantize's slot form, common.cuh).
          float sum[1][kInt4GroupPacks][kPackElems];
          dequantize<DTYPE>(words, sum);
          store<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4, Layout::group_run, TAIL>(
              sum, partials_ptr_array, partials_ptr_array_size, rank, 0, 0, partials_stride_m,
              partials_stride_n, 1, partials_size_m, partials_size_n, 1,
              TAIL ? min(g_end * kInt4Group, own) : min(g_end, own / kInt4Group) * kInt4Group, 0,
              g0 * kInt4Group, 1);
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
      // The chunk in whole packs (TILE_N rounded up); the share's packs past it are masked.
      constexpr int kChunkN = (TILE_N + kPackElems - 1) / kPackElems * kPackElems;
      constexpr int kPacks = (kChunkN + kPackElems * kLanes - 1) / (kPackElems * kLanes);
      static_assert(kLanes * WORLD == THREADS_PER_BLOCK, "a block is a group of threads a rank");
      // Whole waves a rank: its rank and entry are scalar; else each lane's own.
      constexpr bool kWaves = kLanes % WAVE_SIZE == 0;
      // A RANK'S LANES WALK THE CHUNK IN WHOLE STEPS, else the last step is masked at the chunk.
      constexpr bool kWhole = kPacks * kPackElems * kLanes == kChunkN;
      const int chunks = (slice + kChunkN - 1) / kChunkN;
      if constexpr (DIRECTION == Direction::push) {
        // 1. THE PUSH: every rank's share of this rank's input into that rank's inbox at slot
        //    [rank], each block the chunks its twin block on that rank sums.
        for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
          const int r = (rank + static_cast<int>(threadIdx.x) / kLanes) % WORLD;
          const int len = share_len(r);
          if (len == 0) continue;
          // The share's end, or the chunk's where its last step is masked.
          const int end = kWhole ? len : min(len, chunk * kChunkN + kChunkN);
          DTYPE part[1][kPacks][kPackElems];
          load<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, kInput, TAIL>(
              &input_ptr, 1, 0, r, slice, input_stride_m, input_stride_n, WORLD, input_size_m,
              input_size_n, input_size_m, end, part, offs_m, chunk * kChunkN, 1);
          send<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::run>(
              part, staging_ptr_array, staging_ptr_array_size, r, rank, staging_stride_m,
              staging_stride_m, staging_stride_n, WORLD, staging_size_m, staging_size_n, 1, end, 0,
              chunk * kChunkN, 1);
        }
      }
      // This row of threads' rank w: (pull) its input entry and this rank's partials entry, read
      // before the barrier.
      const int w = (rank + static_cast<int>(threadIdx.x) / kLanes) % WORLD;
      [[maybe_unused]] const DTYPE* input_entry = nullptr;
      // (Part of a wave a rank, a lane's own: recv reads each lane's entry from the table itself.)
      if constexpr (DIRECTION == Direction::pull && kWaves)
        input_entry = load(input_ptr_array, input_ptr_array_size, w);
      char* const partials_entry = load(partials_ptr_array, partials_ptr_array_size, rank);
      barrier<WORLD>(signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
                              signals_stride_block, signals_stride_writer, signals_size_parity,
                              signals_size_block, signals_size_writer, status_ptr,
                              status_stride_field, status_size_field, rank, timeout_ticks, UNTIL, 0);
      // 2. THE SUM: this rank's share a chunk a step, every rank's copy a row of threads (pull:
      //    rank w's input; push: writer w's slot of this inbox), summed in rotated order through
      //    LDS, rounded once, stored in this rank's partials.
      const int len = share_len(rank);
      if (len != 0) {  // a late rank's share is empty at few packs (every block alike)
        for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
          // The share's end, or the chunk's where its last step is masked.
          const int end = kWhole ? len : min(len, chunk * kChunkN + kChunkN);
          DTYPE got[1][kPacks][kPackElems];
          if constexpr (DIRECTION == Direction::push)
            load<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::run>(
                staging_ptr_array, staging_ptr_array_size, rank, w, staging_stride_m,
                staging_stride_m, staging_stride_n, WORLD, staging_size_m, staging_size_n, 1, end,
                got, 0, chunk * kChunkN, 1);
          else if constexpr (kWaves)
            recv<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, kInput, TAIL>(
                &input_entry, 1, 0, rank, slice, input_ptr_array_stride_m,
                input_ptr_array_stride_n, WORLD, input_size_m, input_size_n, input_size_m, end, got,
                offs_m, chunk * kChunkN, 1);
          else
            recv<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, kInput, TAIL>(
                input_ptr_array, input_ptr_array_size, w, rank, slice, input_ptr_array_stride_m,
                input_ptr_array_stride_n, WORLD, input_size_m, input_size_n, input_size_m, end, got,
                offs_m, chunk * kChunkN, 1);
          DTYPE rounded[1][kPacks][kPackElems];
          reduce<WORLD, THREADS_PER_BLOCK, WAVE_SIZE, ReduceAxis::m>(got[0], rounded[0]);
          store<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(rounded, &partials_entry, 1, 0, 0, 0,
                                   0, 1, 1, 1, len, 1, end, 0, chunk * kChunkN, 1);
        }
      }
      barrier<WORLD>(signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
                              signals_stride_block, signals_stride_writer, signals_size_parity,
                              signals_size_block, signals_size_writer, status_ptr,
                              status_stride_field, status_size_field, rank, timeout_ticks, Until::visible, 1);
    }
  } else {
    // This rank's columns (whole packs) and this block's rows, computed before the first
    // barrier so they stay in scalar registers.
    const int packs = input_size_n / kPackElems;  // the row, in packs (n is whole packs)
    // int4: whole groups a rank (a group is 4 lanes' packs), the last rank's last group short
    // where the row is not whole groups (TAIL).
    constexpr int kSliceUnit = TAIL ? kInt4GroupPacks : 1;
    const int slice = ((packs + kSliceUnit - 1) / kSliceUnit + WORLD - 1) / WORLD * kSliceUnit;
    const int col0 = min(rank * slice, packs);
    const int own_packs = max(0, min(slice, packs - col0));
    const int end = (col0 + own_packs) * kPackElems;  // this rank's columns end, in elements
    [[maybe_unused]] const int block_size_m =
        input_size_m > static_cast<int>(blockIdx.x)
            ? (input_size_m - blockIdx.x + gridDim.x - 1) / gridDim.x
            : 0;
    // 1. int4: this block's rows of this rank's input, quantized into its staging, every load
    //    in flight before the first store.
    if constexpr (kInt4) {
      constexpr int kTiles = TILE_M < 4 ? 4 / TILE_M : 1;
      const int tiles = (input_size_m + TILE_M - 1) / TILE_M;
      for (int t = blockIdx.x; t < tiles; t += kTiles * gridDim.x)
        for (int c = 0; c < input_size_n; c += THREADS_PER_BLOCK * kPackElems) {
          DTYPE x[kTiles][TILE_M][1][kPackElems];
#pragma unroll
          for (int s = 0; s < kTiles; ++s)
            load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(&input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1,
                             input_size_m, input_size_n, input_size_m, input_size_n, x[s],
                             (t + s * static_cast<int>(gridDim.x)) * TILE_M, c, 1);
#pragma unroll
          for (int s = 0; s < kTiles; ++s)
            store<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4, Layout::tile, TAIL>(
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
              recv<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(input_ptr_array, input_ptr_array_size, r, 0, 0, input_ptr_array_stride_m, input_ptr_array_stride_n, 1, input_size_m, input_size_n, input_size_m, end, got, row, c, 0);
            else
              recv<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4, Layout::tile, TAIL>(staging_ptr_array, staging_ptr_array_size, r, 0, 0, staging_stride_m, staging_stride_n, 1, staging_size_m, staging_size_n, input_size_m, end, got, row, c, 0);
            // 3. Summed over the rows (the ranks), in row order, through LDS, rounded once.
            Summed rounded[1][1][kPackElems];
            reduce<WORLD, THREADS_PER_BLOCK, WAVE_SIZE, ReduceAxis::m>(got[0], rounded[0]);
            // 4. Stored in this rank's slice.
            store<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4, Layout::tile, TAIL>(
                rounded, partials_ptr_array, partials_ptr_array_size, rank, 0, 0,
                partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n,
                row + 1, end, row, c, 1);
          }
    } else {
      constexpr int kSliceWaves = (TILE_N / WORLD / kPackElems + WAVE_SIZE - 1) / WAVE_SIZE * WAVE_SIZE;
      // A row of threads: the narrowest whole waves at least the slice's that divide the block.
      constexpr int kThreadsN = [] {
        for (int t = kSliceWaves < THREADS_PER_BLOCK ? kSliceWaves : THREADS_PER_BLOCK;
             t < THREADS_PER_BLOCK; t += WAVE_SIZE)
          if (THREADS_PER_BLOCK % t == 0) return t;
        return THREADS_PER_BLOCK;
      }();
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
                recv<WORLD, kThreadsM, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(input_ptr_array, input_ptr_array_size, r, 0, 0, input_ptr_array_stride_m, input_ptr_array_stride_n, 1, input_size_m, input_size_n, input_size_m, offs_n + size_n, peers[r], row, c, static_cast<int>(gridDim.x));
              // Summed in rank order in fp32, so every rank's sum agrees bitwise, rounded once.
#pragma unroll
              for (int j = 0; j < kPackElems; ++j) {
                float acc = static_cast<float>(peers[0][0][0][j]);
#pragma unroll
                for (int r = 1; r < WORLD; ++r) acc += static_cast<float>(peers[r][0][0][j]);
                rounded[0][0][j] = static_cast<Summed>(acc);
              }
            } else {
              // INT4 sums from the words: every rank's word and scale of this thread's pack, raw.
              uint32_t raw[WORLD][1][1][2];
              for (int r = 0; r < WORLD; ++r)
                recv<WORLD, kThreadsM, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4, Layout::tile, TAIL>(
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
            // Push: sent to every rank's partials (int4: quantized once); pull: stored in this rank's.
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
                    send<WORLD, kThreadsM, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4, Layout::tile, TAIL>(
                        payload, partials_ptr_array, partials_ptr_array_size, r, 0, 0,
                        partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n,
                        input_size_m, offs_n + size_n, row, c, static_cast<int>(gridDim.x));
                  else
                    send<WORLD, kThreadsM, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4, Layout::tile, TAIL>(
                        rounded, partials_ptr_array, partials_ptr_array_size, r, 0, 0,
                        partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n,
                        input_size_m, offs_n + size_n, row, c, static_cast<int>(gridDim.x));
                }
            } else
              store<WORLD, kThreadsM, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4, Layout::tile, TAIL>(
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
