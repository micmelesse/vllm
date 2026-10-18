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
template <int WORLD, int THREADS_PER_BLOCK, int TILE_N, Walk WALK, int QUANTIZATION,
          typename DTYPE, typename OUTPUT>
DINLINE void all_gather(char* const* partials_ptr_array, int partials_ptr_array_size,
                        int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
                        int partials_size_n, OUTPUT&& output, int64_t output_stride_m,
                        int64_t output_stride_n, int output_size_m, int output_size_n, int offs_m,
                        int offs_n, int slice_n, int rank) {
  constexpr int kPackElems = traits<DTYPE>::N;
  // THE OUTPUT'S ROWS BACK TO BACK (a share stored as one run), from its strides: uniform across
  // the launch, a scalar branch.
  [[maybe_unused]] const bool contiguous = output_stride_m == int64_t{output_size_n} * output_stride_n;
  constexpr bool kInt4 = QUANTIZATION == Quantization::int4;
  using V = vec<DTYPE, kPackElems>;
  // A PACK IS ONE 16-BYTE GLOBAL INSTRUCTION (reduce_scatter.cuh's).
  static_assert(sizeof(V) == sizeof(u32x4), "a pack is one 16-byte global access");
  if constexpr (WALK == Walk::flat) {
    // Element e of an [*, n] buffer numbered row-major (the flat walk's address).
    const auto element_at = [](DTYPE* ptr, int64_t stride_m, int64_t stride_n, int64_t e, int n) {
      if (stride_m == int64_t{n} * stride_n) return ptr + e * stride_n;
      const int64_t row = e < (int64_t{1} << 31)
                              ? int64_t{static_cast<uint32_t>(e) / static_cast<uint32_t>(n)}
                              : e / n;
      return ptr + row * stride_m + (e - row * n) * stride_n;
    };
    DTYPE* const output_ptr = output;
    constexpr int kUnit = kInt4 ? kInt4Group : kPackElems;
    const int64_t total = int64_t{output_size_m} * output_size_n;
    const int slice = static_cast<int>((total / kUnit + WORLD - 1) / WORLD * kUnit);
    const auto share_len = [&](int r) {
      return static_cast<int>(max(int64_t{0}, min(int64_t{slice}, total - int64_t{r} * slice)));
    };
    if constexpr (kInt4) {
      const uint8_t* values[WORLD];
      const __fp16* scales[WORLD];
#pragma unroll
      for (int w = 0; w < WORLD; ++w) {
        values[w] = ptr_array_at<uint8_t>(partials_ptr_array, partials_ptr_array_size, (rank + w) % WORLD);
        scales[w] = reinterpret_cast<__fp16*>(ptr_array_at(partials_ptr_array, partials_ptr_array_size, (rank + w) % WORLD) +
                                              int4_scales_offset(partials_size_m, partials_size_n));
      }
      const int groups_all = slice / kInt4Group;
      const int grid = static_cast<int>(gridDim.x);
      const int per_block = (groups_all + grid - 1) / grid;
      const int g_first = static_cast<int>(blockIdx.x) * per_block;
      const int g_end = min(g_first + per_block, groups_all);
      for (int g0 = g_first; g0 < g_end; g0 += THREADS_PER_BLOCK) {
        const int g = g0 + static_cast<int>(threadIdx.x);
        u32x4 words[WORLD];
        float group_scales[WORLD];
#pragma unroll
        for (int w = 0; w < WORLD; ++w) {
          const int groups = share_len((rank + w) % WORLD) / kInt4Group;
          const int at = max(min(g, groups - 1), 0);
          // A uniform base plus a 32-bit byte offset, opaque so the address is built at its load.
          uint32_t value_at = static_cast<uint32_t>(at) * 16;
          uint32_t scale_at = static_cast<uint32_t>(at) * kInt4ScaleBytes;
          asm volatile("" : "+v"(value_at));
          asm volatile("" : "+v"(scale_at));
          words[w] = *(const global_u32x4*)(values[w] + value_at);
          group_scales[w] =
              *(const global_f16*)(reinterpret_cast<const char*>(scales[w]) + scale_at);
        }
        // ISSUED HERE: no memory operation nor instruction crosses this point, so every rank's
        // group is in flight before the first is decoded.
        asm volatile("" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int w = 0; w < WORLD; ++w) {
          const int r = (rank + w) % WORLD;
          const int groups = share_len(r) / kInt4Group;
          if (!(g < g_end && g < groups)) continue;
          const int64_t first = int64_t{r} * slice;
          V* group;
          if (contiguous) {
            uint32_t offset = static_cast<uint32_t>(g) * (kInt4Group * sizeof(DTYPE));
            asm volatile("" : "+v"(offset));
            group = reinterpret_cast<V*>(
                reinterpret_cast<char*>(output_ptr + first * output_stride_n) + offset);
          } else
            group = reinterpret_cast<V*>(element_at(output_ptr, output_stride_m, output_stride_n,
                                                    first + int64_t{g} * kInt4Group, output_size_n));
#pragma unroll
          for (int k = 0; k < kInt4GroupPacks; ++k) {
            DTYPE x[kPackElems];
            dequantize(words[w][k], group_scales[w], x);
            u32x4 raw;
            __builtin_memcpy(&raw, x, sizeof(raw));
            *(global_u32x4*)(group + k) = raw;
          }
        }
      }
    } else {
      constexpr int kLanes = THREADS_PER_BLOCK / WORLD;
      constexpr int kPacks = TILE_N / (kPackElems * kLanes);
      static_assert(kLanes * WORLD == THREADS_PER_BLOCK, "a block is a group of threads a rank");
      static_assert(kLanes % kWaveSize == 0, "a rank's copy is whole waves");
      const int chunks = (slice + TILE_N - 1) / TILE_N;
      for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
        const int r = (rank + static_cast<int>(threadIdx.x) / kLanes) % WORLD;
        DTYPE got[1][kPacks][kPackElems];
        // 1. Rank r's summed chunk, in one round trip.
        load<1, THREADS_PER_BLOCK, Ranks::lane, Quantization::none, WORLD>(partials_ptr_array, partials_ptr_array_size, r, 0, 0, 0, 1, 1, 1, slice, 1,
                       slice, got, 0, chunk * TILE_N, 1);
        // 2. Stored where rank r's share goes, its first `len` elements.
        const int64_t first = int64_t{r} * slice;
        const int len = share_len(r);
        V* const run = reinterpret_cast<V*>(output_ptr + first * output_stride_n);
#pragma unroll
        for (int k = 0; k < kPacks; ++k) {
          const int c = chunk * TILE_N / kPackElems + static_cast<int>(threadIdx.x) % kLanes + k * kLanes;
          if (c >= len / kPackElems) continue;
          V* at;
          if (contiguous) {
            // A uniform base plus a 32-bit byte offset, opaque so the address is built at its
            // store (c is below the share, so no clamp).
            uint32_t offset = static_cast<uint32_t>(c) * uint32_t{sizeof(V)};
            asm volatile("" : "+v"(offset));
            at = reinterpret_cast<V*>(reinterpret_cast<char*>(run) + offset);
          } else {
            at = reinterpret_cast<V*>(element_at(output_ptr, output_stride_m, output_stride_n,
                                                 first + int64_t{c} * kPackElems, output_size_n));
          }
          u32x4 raw;
          __builtin_memcpy(&raw, got[0][k], sizeof(raw));
          *(global_u32x4*)(at) = raw;
        }
      }
    }
  } else {
    using Tile = std::remove_reference_t<OUTPUT>;
    constexpr int kRows = std::extent_v<Tile, 0>;
    constexpr int kPacks = std::extent_v<Tile, 1>;
    static_assert(std::extent_v<Tile, 2> == kPackElems, "a tile's pack is the dtype's");
    (void)output_stride_m;
    (void)output_stride_n;
    // Row i of the tile: past output_size_m the last row (one row of threads, so the row is the
    // block's and its address scalar); a tile of several rows skips a row past it instead of
    // re-reading the last over the links (one row's is real wherever a kernel makes one, and a
    // branch around its loads cost registers: 126 -> 178 VGPRs, 23-46-07Z). Pack k's column, past
    // output_size_n the last below it, so every load is in bounds and none is under a branch.
    const auto row_of = [&](int i) {
      return offs_m + i < output_size_m ? offs_m + i : output_size_m - 1;
    };
    const auto col_of = [&](int k) {
      const int n = offs_n / kPackElems + static_cast<int>(threadIdx.x) % THREADS_PER_BLOCK + k * THREADS_PER_BLOCK;
      return n < output_size_n / kPackElems ? n : output_size_n / kPackElems - 1;
    };
    if constexpr (!kInt4) {
#pragma unroll
      for (int i = 0; i < kRows; ++i) {
        if (kRows > 1 && offs_m + i >= output_size_m) continue;
#pragma unroll
        for (int k = 0; k < kPacks; ++k) {
          const int col = col_of(k);
          const int owner = min(col / (slice_n / kPackElems), WORLD - 1);
          const u32x4 raw = *(const global_u32x4*)(
              reinterpret_cast<const V*>(lane_ptr_array_at<WORLD, const DTYPE>(partials_ptr_array, partials_ptr_array_size, owner)) +
              int64_t{row_of(i)} * (partials_stride_m / kPackElems) + col);
          __builtin_memcpy(output[i][k], &raw, sizeof(raw));
        }
      }
      // ISSUED HERE: every owner's pack in flight before anything below runs.
      asm volatile("" ::: "memory");
      __builtin_amdgcn_sched_barrier(0);
    } else {
      static_assert(THREADS_PER_BLOCK % kInt4GroupPacks == 0, "a group is 4 lanes' packs");
      // The partials' INT4 payload, from their own sizes (the INT4 payload layout).
      const int64_t values_stride_m = partials_size_n / 2;
      const int64_t scales_stride_m = partials_size_n / kInt4Group;
      const int64_t scales_offset = int4_scales_offset(partials_size_m, partials_size_n);
      uint32_t words[kRows][kPacks];
      float scales[kRows][kPacks];
#pragma unroll
      for (int i = 0; i < kRows; ++i) {
        if (kRows > 1 && offs_m + i >= output_size_m) continue;
#pragma unroll
        for (int k = 0; k < kPacks; ++k) {
          const int col = col_of(k);
          const int owner = min(col / (slice_n / kPackElems), WORLD - 1);
          const int64_t row = row_of(i);
          // The pack's word of values and its group's scale, each a row's base plus an opaque
          // 32-bit byte offset (built at its load).
          uint32_t value_at = static_cast<uint32_t>(col) * kInt4WordBytes;
          uint32_t scale_at = static_cast<uint32_t>(col / kInt4GroupPacks) * kInt4ScaleBytes;
          asm volatile("" : "+v"(value_at));
          asm volatile("" : "+v"(scale_at));
          words[i][k] = *(const global_u32*)(lane_ptr_array_at<WORLD, const uint8_t>(partials_ptr_array, partials_ptr_array_size, owner) + row * values_stride_m +
                                             value_at);
          scales[i][k] = *(const global_f16*)(
              reinterpret_cast<const char*>(
                  reinterpret_cast<const __fp16*>(lane_ptr_array_at<WORLD, const char>(partials_ptr_array, partials_ptr_array_size, owner) + scales_offset) +
                  row * scales_stride_m) +
              scale_at);
        }
      }
      asm volatile("" ::: "memory");  // issued (above)
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < kRows; ++i)
#pragma unroll
        for (int k = 0; k < kPacks; ++k) dequantize(words[i][k], scales[i][k], output[i][k]);
    }
  }
}

}  // namespace hip_custom_communicator
