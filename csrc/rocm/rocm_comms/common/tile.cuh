// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE TILE: how a kernel cuts its work, in Triton's terms (TILE_M, offs_m, M, a mask for the last
// tile), and what a thread holds of it.

#pragma once

#ifndef HIP_COMMS_COMMON_INTERFACE
#error "include common/common.cuh, common's one interface, not its parts"
#endif

#include <type_traits>

#include "utils.cuh"

namespace hip_comms {

// A TILE, in Triton's terms, and THE ONE TYPE A KERNEL WORKS IN: a chunk of an M x N tensor, the
// block's work at once (TILE_M rows from offs_m, every row_step-th, and TILE_N columns from offs_n),
// and the elements of it this thread holds. Rows and columns count elements, as the tensor does. A
// row is real when it is below M and a column when it is below N (the last tile may be short; a
// slice is a tile whose N is the slice's end).
//   DTYPE              the tensor's dtype in memory: a thread holds kPack-element groups, one
//                      16-byte load of it, so a quantized DTYPE holds wider groups and nothing else
//                      changes
//   THREADS_M x THREADS_N  THE LAYOUT, the block's threads over the tile as the tile is TILE_M x
//                      TILE_N elements: THREADS_N threads share a row's columns, every THREADS_N-th
//                      group each, and THREADS_M rows go at once. 1 x the block for a norm, reducing
//                      along the row; 8 x 64 (a wave a row) for a narrow slice of many rows.
//                      Compile-time, since it sizes the registers; THREADS_M x THREADS_N is the block
//   ACC_DTYPE          what the elements are held as: DTYPE as loaded, float for the math
//                      (`to<float>()` keeps every element where it is), or a weight's own dtype
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N,
          typename ACC_DTYPE = DTYPE>
struct Tile {
  static constexpr int kPack     = traits<DTYPE>::N;                  // elements in a group
  static constexpr int kTileM   = TILE_M;  // the chunk, rows x columns in elements
  static constexpr int kTileN   = TILE_N;
  static constexpr int kThreads = THREADS_M * THREADS_N;  // the block
  static constexpr int kThreadsM = THREADS_M;
  static constexpr int kThreadsN = THREADS_N;
  static_assert(TILE_M % THREADS_M == 0, "a tile's rows are whole for every row of threads");
  static_assert(TILE_N % (kPack * THREADS_N) == 0,
                "a tile's columns are whole groups for every thread of a row");
  static constexpr int kRows = TILE_M / THREADS_M;          // rows this thread holds
  static constexpr int K     = TILE_N / (kPack * THREADS_N);  // groups of a row it holds
  using Dtype = DTYPE;
  using Acc  = ACC_DTYPE;
  using Pack = vec<ACC_DTYPE, kPack>;  // how a group is loaded and stored, not how it is held
  int M, N;
  int offs_m, offs_n;
  int row_step = 1;
  ACC_DTYPE v[kRows][K][kPack];

  // THIS THREAD'S GROUP k OF A ROW, in groups from the row's start: past N clamped to the last, so
  // every load stays in bounds; its mask is 1 below N and 0 past it (a float, multiplied in, so
  // nothing reading it branches: a load under a runtime `if` cannot be hoisted past the branch, and
  // loads meant to be in flight together then wait one at a time).
  // WHETHER THIS THREAD HOLDS ANY OF THE TILE: a layout may cover fewer threads than the block (a
  // reduction's one row, the first THREADS_N's); the others load and store nothing. Free when the
  // layout is the block, which the launch bounds tell the compiler.
  DINLINE bool participates() const { return static_cast<int>(threadIdx.x) < kThreads; }
  DINLINE int lane() const { return static_cast<int>(threadIdx.x) % THREADS_N; }
  DINLINE int col(int k) const {
    const int n = offs_n / kPack + lane() + k * THREADS_N;
    return n < N / kPack ? n : N / kPack - 1;
  }
  DINLINE float mask(int k) const {
    return offs_n / kPack + lane() + k * THREADS_N < N / kPack ? 1.0f : 0.0f;
  }
  // This thread's row m, a row past M reading row M - 1 (and storing nothing).
  DINLINE int tile_row(int m) const {
    // One row of threads: the row is the block's, so its address stays scalar.
    if constexpr (THREADS_M == 1) return m;
    return static_cast<int>(threadIdx.x) / THREADS_N + m * THREADS_M;
  }
  DINLINE bool live(int m) const { return offs_m + tile_row(m) * row_step < M; }
  DINLINE int row(int m) const { return live(m) ? offs_m + tile_row(m) * row_step : M - 1; }

  // The same place held as AS_DTYPE, its elements converted (rounded once) or empty.
  template <typename AS_DTYPE>
  using As = Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, AS_DTYPE>;
  template <typename AS_DTYPE>
  DINLINE As<AS_DTYPE> like() const {
    return {M, N, offs_m, offs_n, row_step};
  }
  template <typename AS_DTYPE>
  DINLINE As<AS_DTYPE> to() const {
    As<AS_DTYPE> t = like<AS_DTYPE>();
#pragma unroll
    for (int m = 0; m < kRows; ++m)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < kPack; ++j) t.v[m][k][j] = static_cast<AS_DTYPE>(v[m][k][j]);
    return t;
  }
};

// The tile TILE held as AS_DTYPE (a float tile for the math, a weight's own dtype), and whether a
// type is a tile.
template <typename TILE, typename AS_DTYPE>
using TileAs = typename TILE::template As<AS_DTYPE>;
template <typename X>
struct is_tile : std::false_type {};
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N, typename ACC_DTYPE>
struct is_tile<Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, ACC_DTYPE>>
    : std::true_type {};

}  // namespace hip_comms
