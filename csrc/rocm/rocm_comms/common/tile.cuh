// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE TILE: how a kernel cuts its work, in Triton's terms (TILE_M, offs_m, M, a mask for the last
// tile), and what a thread holds of it.

#pragma once

#ifndef HIP_COMMS_COMMON_INTERFACE
#error "include common/common.cuh, common's one interface, not its parts"
#endif

#include "utils.cuh"

namespace hip_comms {

// A TILE, in Triton's terms, and THE ONE TYPE A KERNEL WORKS IN: TILE_M rows of an M x N tensor from
// row offs_m and TILE_N of its columns from offs_n (a block's work at once), and the elements of it
// this thread holds. Rows and columns count elements, as the tensor does. Row m is real when
// offs_m + m < M and a column when it is below N (the last tile may be short; a scale-add slice is
// a tile whose N is the slice's end).
//   DTYPE        the tensor's dtype in memory, which sets how its columns spread over the threads:
//                a thread holds kPack-element groups (one 16-byte load of DTYPE), every
//                THREADS_PER_BLOCK-th, so a quantized DTYPE holds wider groups and nothing else changes
//   THREADS_PER_BLOCK  the threads sharing the tile; compile-time, since it sizes the registers
//   ACC_DTYPE    what the elements are held as: DTYPE as loaded, float for the math (`to<float>()`
//                keeps every element where it is), or a weight's own dtype loaded in DTYPE's layout
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_PER_BLOCK, typename ACC_DTYPE = DTYPE>
struct Tile {
  static constexpr int kPack = traits<DTYPE>::N;  // elements in a group
  static_assert(TILE_N % (kPack * THREADS_PER_BLOCK) == 0,
                "a tile's columns are whole groups for every thread");
  static constexpr int K = TILE_N / (kPack * THREADS_PER_BLOCK);  // groups of a row this thread holds
  static constexpr int kRows = TILE_M;
  using Acc  = ACC_DTYPE;
  using Pack = vec<ACC_DTYPE, kPack>;
  int M, N;
  int offs_m, offs_n;
  Pack v[TILE_M][K];

  // THIS THREAD'S GROUP k OF A ROW, in groups from the row's start: past N clamped to the last, so
  // every load stays in bounds; its mask is 1 below N and 0 past it (a float, multiplied in, so
  // nothing reading it branches: a load under a runtime `if` cannot be hoisted past the branch, and
  // loads meant to be in flight together then wait one at a time).
  DINLINE int col(int k) const {
    const int n = offs_n / kPack + static_cast<int>(threadIdx.x) + k * THREADS_PER_BLOCK;
    return n < N / kPack ? n : N / kPack - 1;
  }
  DINLINE float mask(int k) const {
    return offs_n / kPack + static_cast<int>(threadIdx.x) + k * THREADS_PER_BLOCK < N / kPack ? 1.0f
                                                                                       : 0.0f;
  }
  // Row m, a row past M reading row M - 1 (and storing nothing).
  DINLINE bool live(int m) const { return offs_m + m < M; }
  DINLINE int row(int m) const { return live(m) ? offs_m + m : M - 1; }

  // The same place held as U, its elements converted (rounded once) or empty.
  template <typename AS_DTYPE>
  DINLINE Tile<DTYPE, TILE_M, TILE_N, THREADS_PER_BLOCK, AS_DTYPE> to() const {
    Tile<DTYPE, TILE_M, TILE_N, THREADS_PER_BLOCK, AS_DTYPE> t{M, N, offs_m, offs_n};
#pragma unroll
    for (int m = 0; m < TILE_M; ++m)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < kPack; ++j) t.v[m][k].d[j] = static_cast<AS_DTYPE>(v[m][k].d[j]);
    return t;
  }
  template <typename AS_DTYPE>
  DINLINE Tile<DTYPE, TILE_M, TILE_N, THREADS_PER_BLOCK, AS_DTYPE> like() const {
    return {M, N, offs_m, offs_n};
  }
};

}  // namespace hip_comms
