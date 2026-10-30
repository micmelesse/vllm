// SPDX-License-Identifier: MIT Copyright (c) 2026, Advanced Micro Devices, Inc.
// All rights reserved.
//
// THE ALL-GATHER, written once with the common ops (load, store, the codec):
// every rank's piece to every rank. The two-shots' (flat, columns): every
// rank's summed share out of its partials, after reduce_scatter's second
// barrier, into the output. The one-shots' (tile_pull, tile_push): every rank's
// copy of one tile of the input into this thread's registers, which the
// one-shot then sums. Its variations are template parameters, its walks written
// inline.

#pragma once

#include <type_traits>

#include "common.cuh"
#include "reduce_scatter.cuh"  // the walk all_gather walks (a block reads what its twin wrote)

namespace hip_custom_communicator {

// WHICH ALL-GATHER: the reduce-scatter's cut walked back (flat, each rank's
// contiguous range of the flattened tensor; columns, each pack from the rank
// owning its columns), or a tile of every rank's input (tile_pull, read where
// each rank published it; tile_push, sent as lines into every rank's inbox and
// polled out of this one's; tile_staged, out of this rank's inbox in the
// staging, which every rank filled before a barrier).
enum class AllGatherAlgorithm {
  flat,
  rows,
  columns,
  tile_pull,
  tile_push,
  tile_staged
};

// flat (the plain two-shot): `output` a buffer, DTYPE*, of output_size_m x
// output_size_n at its strides; every rank's share (reduce_scatter's flat walk:
// rows numbered row-major, a rank's share one contiguous range) stored at its
// place, a TILE_N chunk a step, a rank's copy kLanes = THREADS_PER_BLOCK /
// WORLD threads (rotated, rank + w). int4: a group a thread, each block the
// contiguous range of groups reduce_scatter's same block summed, a rank a row
// of threads, decoded and stored.
//
// rows (the fused ops' pull): `output` a thread's tile, ROWS x K packs from row
// offs_m and column offs_n of the output_size_m x output_size_n tensor
// (output's strides unused): each row from the rank owning it, `slice` rows a
// rank (the last rank's to the end), its partials at the row's place
// (reduce_scatter's rows walk). int4: the owner's words and scales, decoded.
//
// columns (the fused ops' pull): `output` a thread's tile, ROWS x K packs from
// row offs_m and column offs_n of the output_size_m x output_size_n tensor
// (output's strides unused): each pack from the rank owning its columns,
// `slice` columns a rank (the last rank's to the end), the owner's partials
// picked in each lane (a wave's packs may have two owners). int4: the owner's
// words and scales, decoded.
template <AllGatherAlgorithm ALGORITHM, bool QUANTIZE_INT4, typename DTYPE,
          int WORLD, int TILE_N, int THREADS_PER_BLOCK, int WAVE_SIZE,
          int PACK_BYTES, bool CONTIGUOUS, bool TAIL, typename OUTPUT>
DINLINE void all_gather(char* const* partials_ptr_array,
                        int partials_ptr_array_size, int64_t partials_stride_m,
                        int64_t partials_stride_n, int partials_size_m,
                        int partials_size_n, OUTPUT&& output,
                        int64_t output_stride_m, int64_t output_stride_n,
                        int output_size_m, int output_size_n, int offs_m,
                        int offs_n, int slice, int rank) {
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
  constexpr bool kInt4 = QUANTIZE_INT4;
  if constexpr (ALGORITHM == AllGatherAlgorithm::flat) {
    // A rank's share is slot r of the output viewed flat (stride `slice`
    // elements: common.cuh's Layout run / flat), as reduce_scatter's.
    DTYPE* const output_ptr = output;
    constexpr int kUnit = kInt4 ? kInt4Group : kPackElems;
    const int64_t total = int64_t{output_size_m} * output_size_n;
    // A SHARE IS WHOLE UNITS, the last rank's to the band's end (int4: its last
    // group short where the band is not whole groups; the clips below are in
    // elements, so a short group is masked).
    const int64_t units =
        TAIL ? (total + kUnit - 1) / kUnit : total / kUnit;  // reduce_scatter's
    const int slice = static_cast<int>((units + WORLD - 1) / WORLD * kUnit);
    const auto share_len = [&](int r) {
      return static_cast<int>(
          max(int64_t{0}, min(int64_t{slice}, total - int64_t{r} * slice)));
    };
    if constexpr (kInt4) {
      constexpr Layout kOutput =
          CONTIGUOUS ? Layout::group_run : Layout::group_flat;
      const int groups_all = slice / kInt4Group;
      const int grid = static_cast<int>(gridDim.x);
      const int per_block = (groups_all + grid - 1) / grid;
      const int g_first = static_cast<int>(blockIdx.x) * per_block;
      const int g_end = min(g_first + per_block, groups_all);
      // A RANK A ROW OF THREADS (kRankLanes, rotated rank + w; part of a wave,
      // its rank a lane's own: recv picks each lane's), as reduce_scatter's
      // push: every rank's group in flight at once and each thread decoding
      // one.
      constexpr int kRankLanes = THREADS_PER_BLOCK / WORLD;
      static_assert(kRankLanes * WORLD == THREADS_PER_BLOCK,
                    "a block is a row of threads a rank");
      const int r = (rank + static_cast<int>(threadIdx.x) / kRankLanes) % WORLD;
      const int end = TAIL ? min(g_end * kInt4Group, share_len(r))
                           : min(g_end, share_len(r) / kInt4Group) * kInt4Group;
      for (int g0 = g_first; g0 < g_end; g0 += kRankLanes) {
        // 1. Rank r's group of its summed share, raw.
        uint32_t words[1][kInt4GroupPacks][2];
        recv<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
             Layout::group_run, TAIL>(
            partials_ptr_array, partials_ptr_array_size, r, 0, 0,
            partials_stride_m, partials_stride_n, 1, partials_size_m,
            partials_size_n, 1, slice, words, 0, g0 * kInt4Group, 1);
        // 2. Decoded and stored where rank r's share goes.
        DTYPE x[1][kInt4GroupPacks][kPackElems];
#pragma unroll
        for (int k = 0; k < kInt4GroupPacks; ++k)
          dequantize(words[0][k][0], __builtin_bit_cast(float, words[0][k][1]),
                     x[0][k]);
        store<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, false, kOutput, TAIL>(
            x, &output_ptr, 1, 0, r, slice, output_stride_m, output_stride_n,
            WORLD, output_size_m, output_size_n, 1, end, 0, g0 * kInt4Group, 1);
      }
    } else {
      constexpr Layout kOutput = CONTIGUOUS ? Layout::run : Layout::flat;
      constexpr int kLanes = THREADS_PER_BLOCK / WORLD;
      // THE CHUNK IN WHOLE PACKS (TILE_N rounded up): any TILE_N walks, the
      // slice's packs past it masked at its end.
      constexpr int kChunkN =
          (TILE_N + kPackElems - 1) / kPackElems * kPackElems;
      constexpr int kPacks =
          (kChunkN + kPackElems * kLanes - 1) / (kPackElems * kLanes);
      static_assert(kLanes * WORLD == THREADS_PER_BLOCK,
                    "a block is a group of threads a rank");
      // reduce_scatter's walk: whole waves a rank, else a lane's own rank;
      // whole steps a chunk, else the last masked at the chunk's end.
      constexpr bool kWaves = kLanes % WAVE_SIZE == 0;
      constexpr bool kWhole = kPacks * kPackElems * kLanes == kChunkN;
      const int chunks = (slice + kChunkN - 1) / kChunkN;
      // A ROW OF THREADS' RANK, whole waves, so read from the first lane: recv
      // then takes its scalar path alone (from a lane's value it kept the
      // per-lane select too, WORLD - 1 rank compares whose masks spilled to a
      // VGPR in the plain two-shot: ISA 2026-10-19T03-16-52Z). Part of a wave,
      // a lane's own (recv's per-lane select).
      const int r =
          kWaves ? __builtin_amdgcn_readfirstlane(
                       (rank + static_cast<int>(threadIdx.x) / kLanes) % WORLD)
                 : (rank + static_cast<int>(threadIdx.x) / kLanes) % WORLD;
      for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
        DTYPE got[1][kPacks][kPackElems];
        // 1. Rank r's summed chunk, a row of threads a rank, in one round trip.
        recv<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
            partials_ptr_array, partials_ptr_array_size, r, 0, 0, 0, 1, 1, 1,
            slice, 1, kWhole ? slice : min(slice, chunk * kChunkN + kChunkN),
            got, 0, chunk * kChunkN, 0);
        // 2. Stored where rank r's share goes, its first `len` elements (the
        //    chunk's, masked).
        store<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, kOutput, TAIL>(
            got, &output_ptr, 1, 0, r, slice, output_stride_m, output_stride_n,
            WORLD, output_size_m, output_size_n, 1,
            kWhole ? share_len(r)
                   : min(share_len(r), chunk * kChunkN + kChunkN),
            0, chunk * kChunkN, 1);
      }
    }
  } else if constexpr (ALGORITHM == AllGatherAlgorithm::rows) {
    using Tile = std::remove_reference_t<OUTPUT>;
    constexpr int kRows = std::extent_v<Tile, 0>;
    constexpr int kPacks = std::extent_v<Tile, 1>;
    static_assert(std::extent_v<Tile, 2> == kPackElems,
                  "a tile's pack is the dtype's");
    (void)output_stride_m;
    (void)output_stride_n;
    // EACH ROW FROM THE RANK OWNING IT, a recv a pack (a row's packs have one
    // owner). int4: every pack's raw words and scales first, all in flight,
    // then decoded.
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      const int owner = min((offs_m + i) / slice, WORLD - 1);
      if constexpr (!kInt4) {
#pragma unroll
        for (int k = 0; k < kPacks; ++k) {
          DTYPE pack[1][1][kPackElems];
          recv<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
               Layout::tile, TAIL>(
              partials_ptr_array, partials_ptr_array_size, owner, 0, 0,
              partials_stride_m, partials_stride_n, 1, partials_size_m,
              partials_size_n, output_size_m, output_size_n, pack, offs_m + i,
              offs_n + k * THREADS_PER_BLOCK * kPackElems, 1);
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) output[i][k][j] = pack[0][0][j];
        }
      } else {
        uint32_t raw[kPacks][1][1][2];
#pragma unroll
        for (int k = 0; k < kPacks; ++k)
          recv<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
               Layout::tile, TAIL>(
              partials_ptr_array, partials_ptr_array_size, owner, 0, 0,
              partials_stride_m, partials_stride_n, 1, partials_size_m,
              partials_size_n, output_size_m, output_size_n, raw[k], offs_m + i,
              offs_n + k * THREADS_PER_BLOCK * kPackElems, 1);
#pragma unroll
        for (int k = 0; k < kPacks; ++k)
          dequantize(raw[k][0][0][0],
                     __builtin_bit_cast(float, raw[k][0][0][1]), output[i][k]);
      }
    }
  } else {
    using Tile = std::remove_reference_t<OUTPUT>;
    constexpr int kRows = std::extent_v<Tile, 0>;
    constexpr int kPacks = std::extent_v<Tile, 1>;
    static_assert(std::extent_v<Tile, 2> == kPackElems,
                  "a tile's pack is the dtype's");
    (void)output_stride_m;
    (void)output_stride_n;
    // EACH PACK FROM THE RANK OWNING ITS COLUMNS, `slice` columns a rank (the
    // last rank's to the end): a per-lane rank (a wave's packs may have two
    // owners), so a recv a pack, the pack's column as recv clamps it (past
    // output_size_n the last below it). int4: every pack's raw words and scales
    // first, all in flight, then decoded.
    const auto owner_of = [&](int k) {
      const int n = offs_n / kPackElems +
                    static_cast<int>(threadIdx.x) % THREADS_PER_BLOCK +
                    k * THREADS_PER_BLOCK;
      const int col =
          n < output_size_n / kPackElems ? n : output_size_n / kPackElems - 1;
      return min(col / (slice / kPackElems), WORLD - 1);
    };
    if constexpr (!kInt4) {
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        DTYPE pack[kRows][1][kPackElems];
        recv<WORLD, kRows, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
             Layout::tile, TAIL>(
            partials_ptr_array, partials_ptr_array_size, owner_of(k), 0, 0,
            partials_stride_m, partials_stride_n, 1, partials_size_m,
            partials_size_n, output_size_m, output_size_n, pack, offs_m,
            offs_n + k * THREADS_PER_BLOCK * kPackElems, 1);
#pragma unroll
        for (int i = 0; i < kRows; ++i)
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) output[i][k][j] = pack[i][0][j];
      }
    } else {
      uint32_t raw[kPacks][kRows][1][2];
#pragma unroll
      for (int k = 0; k < kPacks; ++k)
        recv<WORLD, kRows, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
             Layout::tile, TAIL>(
            partials_ptr_array, partials_ptr_array_size, owner_of(k), 0, 0,
            partials_stride_m, partials_stride_n, 1, partials_size_m,
            partials_size_n, output_size_m, output_size_n, raw[k], offs_m,
            offs_n + k * THREADS_PER_BLOCK * kPackElems, 1);
#pragma unroll
      for (int i = 0; i < kRows; ++i)
#pragma unroll
        for (int k = 0; k < kPacks; ++k)
          dequantize(raw[k][i][0][0],
                     __builtin_bit_cast(float, raw[k][i][0][1]), output[i][k]);
    }
  }
}

// tile_pull (the pulled one-shot): every rank's copy of the tile at row offs_m
// and column offs_n of the size_m x size_n input into `copies`, a TILE_M x
// kPacks tile a rank, each read from `entries[r]`, where rank r published it,
// at strides (stride_m, stride_n): its input through the input table, or its
// INT4 payload, decoded.
template <AllGatherAlgorithm ALGORITHM, bool QUANTIZE_INT4, typename DTYPE,
          int WORLD, int TILE_M, int THREADS_PER_BLOCK, int WAVE_SIZE,
          bool TAIL, typename ENTRY, int ROWS, int PACKS, int ELEMS>
DINLINE void all_gather(ENTRY (&entries)[WORLD], int64_t stride_m,
                        int64_t stride_n, int size_m, int size_n, int clip_n,
                        int offs_m, int offs_n,
                        DTYPE (&copies)[WORLD][ROWS][PACKS][ELEMS]) {
  static_assert(ALGORITHM == AllGatherAlgorithm::tile_pull,
                "a tile from where each rank published it");
#pragma unroll
  for (int r = 0; r < WORLD; ++r)
    recv<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
         Layout::tile, TAIL>(&entries[r], 1, 0, 0, 0, stride_m, stride_n, 1,
                             size_m, size_n, size_m, clip_n, copies[r], offs_m,
                             offs_n, 1);
}

// tile_push (the pushed one-shot): this rank's tile, `mine`, sent into slot
// [rank] of every rank's inbox in the unfenced region as lines carrying each
// row's epoch (`epochs`), then every rank's copy polled out of this rank's
// inbox until every line carries it: no barrier. INT4: a line a pack, its word
// and its group's scale, decoded on arrival.
template <AllGatherAlgorithm ALGORITHM, bool QUANTIZE_INT4, typename DTYPE,
          int WORLD, int TILE_M, int THREADS_PER_BLOCK, int WAVE_SIZE, int ROWS,
          int PACKS, int ELEMS>
DINLINE void all_gather(const DTYPE (&mine)[ROWS][PACKS][ELEMS],
                        const uint32_t (&epochs)[ROWS],
                        char* const* inbox_ptr_array, int inbox_ptr_array_size,
                        int64_t inbox_stride_slot, int64_t inbox_stride_m,
                        int64_t inbox_stride_n, int inbox_size_slot,
                        int inbox_size_m, int inbox_size_n, int clip_m,
                        int clip_n, int offs_m, int rank, uint32_t* status_ptr,
                        int64_t status_stride_field, int status_size_field,
                        uint64_t timeout_ticks,
                        DTYPE (&copies)[WORLD][ROWS][PACKS][ELEMS]) {
  static_assert(ALGORITHM == AllGatherAlgorithm::tile_push,
                "a tile sent and polled as lines");
  // INT4: the tile quantized once, every pack's word and its group's scale,
  // then sent raw to every rank. No branch on a runtime rank between the sends
  // (one serialized the remote stores).
  if constexpr (QUANTIZE_INT4) {
    uint32_t payload[ROWS][PACKS][2];
#pragma unroll
    for (int i = 0; i < ROWS; ++i)
#pragma unroll
      for (int k = 0; k < PACKS; ++k) {
        float scale;
        quantize(reinterpret_cast<const DTYPE(&)[1][ELEMS]>(mine[i][k]),
                 reinterpret_cast<uint32_t (&)[1]>(payload[i][k][0]), scale);
        payload[i][k][1] = __builtin_bit_cast(uint32_t, scale);
      }
#pragma unroll
    for (int r = 0; r < WORLD; ++r)
      send<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, true, Layout::line>(
          payload, inbox_ptr_array, inbox_ptr_array_size, r, rank,
          inbox_stride_slot, inbox_stride_m, inbox_stride_n, inbox_size_slot,
          inbox_size_m, inbox_size_n, clip_m, clip_n, offs_m, 0, 1, epochs);
  } else {
#pragma unroll
    for (int r = 0; r < WORLD; ++r)
      send<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::line>(
          mine, inbox_ptr_array, inbox_ptr_array_size, r, rank,
          inbox_stride_slot, inbox_stride_m, inbox_stride_n, inbox_size_slot,
          inbox_size_m, inbox_size_n, clip_m, clip_n, offs_m, 0, 1, epochs);
  }
  load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
       Layout::line>(inbox_ptr_array, inbox_ptr_array_size, rank, 0,
                     inbox_stride_slot, inbox_stride_m, inbox_stride_n,
                     inbox_size_slot, inbox_size_m, inbox_size_n, clip_m,
                     clip_n, copies, offs_m, 0, 1, epochs, status_ptr,
                     status_stride_field, status_size_field, timeout_ticks);
}

// tile_staged (the staged one-shot): every rank's copy of the tile at row
// offs_m and column offs_n out of this rank's inbox in the staging, slot r rank
// r's, each an [m, n] buffer at the slot's stride (INT4: its payload, decoded),
// which every rank filled before the launch's barrier.
template <AllGatherAlgorithm ALGORITHM, bool QUANTIZE_INT4, typename DTYPE,
          int WORLD, int TILE_M, int THREADS_PER_BLOCK, int WAVE_SIZE,
          bool TAIL, int ROWS, int PACKS, int ELEMS>
DINLINE void all_gather(char* const* inbox_ptr_array, int inbox_ptr_array_size,
                        int64_t inbox_stride_slot, int size_m, int size_n,
                        int clip_n, int offs_m, int offs_n, int rank,
                        DTYPE (&copies)[WORLD][ROWS][PACKS][ELEMS]) {
  static_assert(ALGORITHM == AllGatherAlgorithm::tile_staged,
                "a tile out of the staged inbox");
  load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4, Layout::tile,
       TAIL>(inbox_ptr_array, inbox_ptr_array_size, rank, 0, inbox_stride_slot,
             size_n, 1, WORLD, size_m, size_n, size_m, clip_n, copies, offs_m,
             offs_n, 1);
}

}  // namespace hip_custom_communicator
