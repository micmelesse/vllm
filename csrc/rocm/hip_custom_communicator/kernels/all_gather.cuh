// SPDX-License-Identifier: MIT
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE ALL-GATHER, written once with the common ops (load, store, the codec): every rank's summed
// share out of its partials, after reduce_scatter's second barrier. Its variations are template
// parameters, its walks written inline.

#pragma once

#include <type_traits>

#include "common.cuh"
#include "reduce_scatter.cuh"  // the walk all_gather walks (a block reads what its twin wrote)

namespace hip_custom_communicator {

// WALK::flat (the plain two-shot): `output` a buffer, DTYPE*, of output_size_m x output_size_n at its
// strides; every rank's share (reduce_scatter's flat walk: rows numbered row-major, a rank's share
// one contiguous range) stored at its place, a TILE_N chunk a step, a rank's copy kLanes = THREADS_PER_BLOCK /
// WORLD threads (rotated, rank + w). int4: a group a thread, each block the contiguous range of
// groups reduce_scatter's same block summed, every rank's group in flight, decoded and stored.
//
// WALK::rows (the fused ops' pull): `output` a thread's tile, ROWS x K packs from row offs_m and
// column offs_n of the output_size_m x output_size_n tensor (output's strides unused): each pack
// from the rank owning its columns, slice_n columns a rank (the last rank's to the end), the
// owner's partials picked in each lane (a wave's packs may have two owners). int4: the owner's
// words and scales, decoded.
template <typename DTYPE, int WORLD, int TILE_N, int THREADS_PER_BLOCK, int QUANTIZATION, bool CONTIGUOUS,
          Walk WALK, typename OUTPUT>
DINLINE void all_gather(char* const* partials_ptr_array, int partials_ptr_array_size,
                        int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
                        int partials_size_n, OUTPUT&& output, int64_t output_stride_m,
                        int64_t output_stride_n, int output_size_m, int output_size_n, int offs_m,
                        int offs_n, int slice_n, int rank) {
  constexpr int kPackElems = traits<DTYPE>::N;
  constexpr bool kInt4 = QUANTIZATION == Quantization::int4;
  if constexpr (WALK == Walk::flat) {
    // A rank's share is slot r of the output viewed flat (stride `slice` elements: common.cuh's
    // Layout run / flat), as reduce_scatter's.
    DTYPE* const output_ptr = output;
    constexpr int kUnit = kInt4 ? kInt4Group : kPackElems;
    const int64_t total = int64_t{output_size_m} * output_size_n;
    const int slice = static_cast<int>((total / kUnit + WORLD - 1) / WORLD * kUnit);
    const auto share_len = [&](int r) {
      return static_cast<int>(max(int64_t{0}, min(int64_t{slice}, total - int64_t{r} * slice)));
    };
    if constexpr (kInt4) {
      constexpr Layout kOutput = CONTIGUOUS ? Layout::group_run : Layout::group_flat;
      const int groups_all = slice / kInt4Group;
      const int grid = static_cast<int>(gridDim.x);
      const int per_block = (groups_all + grid - 1) / grid;
      const int g_first = static_cast<int>(blockIdx.x) * per_block;
      const int g_end = min(g_first + per_block, groups_all);
      for (int g0 = g_first; g0 < g_end; g0 += THREADS_PER_BLOCK) {
        // 1. Every rank's group of its summed share, raw, rotated (rank + w), all in flight before
        // the first is decoded.
        uint32_t words[WORLD][1][kInt4GroupPacks][2];
        for (int w = 0; w < WORLD; ++w)
          recv<WORLD, 1, THREADS_PER_BLOCK, QUANTIZATION, Layout::group_run>(
              partials_ptr_array, partials_ptr_array_size, (rank + w) % WORLD, 0, 0,
              partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n, 1, slice,
              words[w], 0, g0 * kInt4Group, 1);
        // 2. Each decoded and stored where its rank's share goes.
#pragma unroll
        for (int w = 0; w < WORLD; ++w) {
          const int r = (rank + w) % WORLD;
          DTYPE x[1][kInt4GroupPacks][kPackElems];
#pragma unroll
          for (int k = 0; k < kInt4GroupPacks; ++k)
            dequantize(words[w][0][k][0], __builtin_bit_cast(float, words[w][0][k][1]), x[0][k]);
          store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, kOutput>(
              x, &output_ptr, 1, 0, r, slice, output_stride_m, output_stride_n, WORLD,
              output_size_m, output_size_n, 1, min(g_end, share_len(r) / kInt4Group) * kInt4Group,
              0, g0 * kInt4Group, 1);
        }
      }
    } else {
      constexpr Layout kOutput = CONTIGUOUS ? Layout::run : Layout::flat;
      constexpr int kLanes = THREADS_PER_BLOCK / WORLD;
      constexpr int kPacks = TILE_N / (kPackElems * kLanes);
      static_assert(kLanes * WORLD == THREADS_PER_BLOCK, "a block is a group of threads a rank");
      static_assert(kLanes % kWaveSize == 0, "a rank's copy is whole waves");
      const int chunks = (slice + TILE_N - 1) / TILE_N;
      for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
        const int r = (rank + static_cast<int>(threadIdx.x) / kLanes) % WORLD;
        DTYPE got[1][kPacks][kPackElems];
        // 1. Rank r's summed chunk, a row of threads a rank, in one round trip.
        recv<WORLD, WORLD, THREADS_PER_BLOCK, Quantization::none, Layout::tile>(partials_ptr_array, partials_ptr_array_size, r, 0, 0, 0, 1, 1, 1, slice, 1,
                       slice, got, 0, chunk * TILE_N, 0);
        // 2. Stored where rank r's share goes, its first `len` elements.
        store<WORLD, 1, THREADS_PER_BLOCK, Quantization::none, kOutput>(
            got, &output_ptr, 1, 0, r, slice, output_stride_m, output_stride_n, WORLD,
            output_size_m, output_size_n, 1, share_len(r), 0, chunk * TILE_N, 1);
      }
    }
  } else {
    using Tile = std::remove_reference_t<OUTPUT>;
    constexpr int kRows = std::extent_v<Tile, 0>;
    constexpr int kPacks = std::extent_v<Tile, 1>;
    static_assert(std::extent_v<Tile, 2> == kPackElems, "a tile's pack is the dtype's");
    (void)output_stride_m;
    (void)output_stride_n;
    // EACH PACK FROM THE RANK OWNING ITS COLUMNS, slice_n columns a rank (the last rank's to the
    // end): a per-lane rank (a wave's packs may have two owners), so a recv a pack, the pack's
    // column as recv clamps it (past output_size_n the last below it). int4: every pack's raw
    // words and scales first, all in flight, then decoded.
    const auto owner_of = [&](int k) {
      const int n = offs_n / kPackElems + static_cast<int>(threadIdx.x) % THREADS_PER_BLOCK + k * THREADS_PER_BLOCK;
      const int col = n < output_size_n / kPackElems ? n : output_size_n / kPackElems - 1;
      return min(col / (slice_n / kPackElems), WORLD - 1);
    };
    if constexpr (!kInt4) {
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        DTYPE pack[kRows][1][kPackElems];
        recv<WORLD, kRows, THREADS_PER_BLOCK, QUANTIZATION, Layout::tile>(
            partials_ptr_array, partials_ptr_array_size, owner_of(k), 0, 0, partials_stride_m,
            partials_stride_n, 1, partials_size_m, partials_size_n, output_size_m, output_size_n,
            pack, offs_m, offs_n + k * THREADS_PER_BLOCK * kPackElems, 1);
#pragma unroll
        for (int i = 0; i < kRows; ++i)
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) output[i][k][j] = pack[i][0][j];
      }
    } else {
      uint32_t raw[kPacks][kRows][1][2];
#pragma unroll
      for (int k = 0; k < kPacks; ++k)
        recv<WORLD, kRows, THREADS_PER_BLOCK, QUANTIZATION, Layout::tile>(
            partials_ptr_array, partials_ptr_array_size, owner_of(k), 0, 0, partials_stride_m,
            partials_stride_n, 1, partials_size_m, partials_size_n, output_size_m, output_size_n,
            raw[k], offs_m, offs_n + k * THREADS_PER_BLOCK * kPackElems, 1);
#pragma unroll
      for (int i = 0; i < kRows; ++i)
#pragma unroll
        for (int k = 0; k < kPacks; ++k)
          dequantize(raw[k][i][0][0], __builtin_bit_cast(float, raw[k][i][0][1]), output[i][k]);
    }
  }
}

}  // namespace hip_custom_communicator
