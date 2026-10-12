// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE TILE: how a kernel cuts its work, in Triton's terms (TILE_M, offs_m, M, a mask for the last
// tile), and what a thread holds of it.

#pragma once

#ifndef HIP_CUSTOM_COMMS_COMMON_INTERFACE
#error "include common/interface.cuh, common's one interface, not its parts"
#endif

#include <type_traits>

#include "utils.cuh"

namespace hip_custom_comms {

// A TILE, in Triton's terms, and THE ONE TYPE A KERNEL WORKS IN: a chunk of an M x N tensor, the
// block's work at once (TILE_M rows from offs_m, every row_step-th, and TILE_N columns from offs_n),
// and the elements of it this thread holds. A 3-D TILE IS R x M x N: TILE_R copies of the chunk,
// one a rank (R the ranks, the axis a collective reduces over; K stays the GEMM's), each read from
// its own rank's memory at that rank's strides. Rows and columns count elements, as the tensor does. A
// row is real when it is below M and a column when it is below N (the last tile may be short; a
// slice is a tile whose N is the slice's end).
//   DTYPE              the tensor's dtype in memory: a thread holds kPack-element groups, one
//                      16-byte load of it, so a quantized DTYPE holds wider groups and nothing else
//                      changes
//   THREADS_M x THREADS_N  THE LAYOUT, the block's threads over the tile as the tile is TILE_M x
//                      TILE_N elements: THREADS_N threads share a row's columns, every THREADS_N-th
//                      group each, and THREADS_M rows go at once. 1 x the block for a norm, reducing
//                      along the row; 8 x 64 (a wave a row) for a narrow slice of many rows.
//                      Compile-time, since it sizes the registers
//   THREADS_PER_BLOCK  the kernel's block: a layout of all of it holds a place in every thread, so
//                      nothing compares threadIdx (the launch bounds do not tell the compiler)
//   TILE_R, THREADS_R  the rank axis and its layout: THREADS_R groups of THREADS_M x THREADS_N
//                      threads, group r holding rank r's copy (THREADS_R == TILE_R: a thread holds
//                      one rank's elements, the collectives meeting in LDS). 1 x 1 for a 2-D tile
//   ACC_DTYPE          what the elements are held as: DTYPE as loaded, float for the math
//                      (`to<float>()` keeps every element where it is), or a weight's own dtype
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N,
          int THREADS_PER_BLOCK, typename ACC_DTYPE = DTYPE, int TILE_R = 1, int THREADS_R = 1>
struct Tile {
  static constexpr int kPack     = traits<DTYPE>::N;                  // elements in a group
  static constexpr int kTileR   = TILE_R;  // the chunk, ranks x rows x columns in elements
  static constexpr int kTileM   = TILE_M;
  static constexpr int kTileN   = TILE_N;
  static constexpr int kThreadsR = THREADS_R;
  static constexpr int kSliceThreads = THREADS_M * THREADS_N;  // a rank's copy's
  static constexpr int kThreads = THREADS_R * kSliceThreads;   // the layout's
  static_assert(THREADS_R == TILE_R || TILE_R == 1,
                "a thread holds one rank's copy (THREADS_R == TILE_R) or the tile is 2-D");
  static constexpr int kThreadsPerBlock = THREADS_PER_BLOCK;
  static_assert(kThreads <= THREADS_PER_BLOCK, "a layout is the block's threads or fewer");
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

  // A TILE IS ITS PLACE; ITS ELEMENTS START UNSET, as registers do: zeroed by default they were 32
  // moves a source in AttnRes's loop, every tile then loaded over them (ISA 2026-10-04T02-22-41Z).
  // zeros<U>() where a sum starts.
  Tile() = default;
  DINLINE Tile(int m, int n, int offs_m_, int offs_n_, int row_step_ = 1)
      : M(m), N(n), offs_m(offs_m_), offs_n(offs_n_), row_step(row_step_) {}

  // THIS THREAD'S GROUP k OF A ROW, in groups from the row's start: past N clamped to the last, so
  // every load stays in bounds; its mask is 1 below N and 0 past it (a float, multiplied in, so
  // nothing reading it branches: a load under a runtime `if` cannot be hoisted past the branch, and
  // loads meant to be in flight together then wait one at a time).
  // WHETHER THIS THREAD HOLDS ANY OF THE TILE: a layout may cover fewer threads than the block (a
  // reduction's one row, the first THREADS_N's); the others store nothing. A compare only for a
  // layout narrower than the block, and loads do not branch on it either way.
  DINLINE bool participates() const {
    if constexpr (kThreads == THREADS_PER_BLOCK) return true;
    else return static_cast<int>(threadIdx.x) < kThreads;
  }
  DINLINE int lane() const { return static_cast<int>(threadIdx.x) % THREADS_N; }
  // THIS THREAD'S RANK COPY, r of R: its group of THREADS_M x THREADS_N threads.
  DINLINE int slice() const {
    if constexpr (THREADS_R == 1) return 0;
    else return static_cast<int>(threadIdx.x) / kSliceThreads;
  }
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
    else if constexpr (THREADS_R == 1)
      return static_cast<int>(threadIdx.x) / THREADS_N + m * THREADS_M;
    else return static_cast<int>(threadIdx.x) % kSliceThreads / THREADS_N + m * THREADS_M;
  }
  DINLINE bool live(int m) const { return offs_m + tile_row(m) * row_step < M; }
  DINLINE int row(int m) const { return live(m) ? offs_m + tile_row(m) * row_step : M - 1; }

  // The same place held as AS_DTYPE, its elements converted (rounded once), unset, or zero.
  template <typename AS_DTYPE>
  using As = Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK, AS_DTYPE, TILE_R,
                  THREADS_R>;
  template <typename AS_DTYPE>
  DINLINE As<AS_DTYPE> like() const {
    return As<AS_DTYPE>(M, N, offs_m, offs_n, row_step);
  }
  template <typename AS_DTYPE>
  DINLINE As<AS_DTYPE> zeros() const {
    As<AS_DTYPE> t = like<AS_DTYPE>();
#pragma unroll
    for (int m = 0; m < kRows; ++m)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < kPack; ++j) t.v[m][k][j] = AS_DTYPE(0);
    return t;
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
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N,
          int THREADS_PER_BLOCK, typename ACC_DTYPE, int TILE_R, int THREADS_R>
struct is_tile<Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK, ACC_DTYPE,
                    TILE_R, THREADS_R>> : std::true_type {};

// THIS BLOCK'S TILES, grid-strided in row-major order over `tiles_m` x `tiles_n` tiles: `body(row,
// col)` for each, in tiles. The row and column are kept as the walk advances, each step an add, a
// compare and a carry, where dividing them out of the work item's index cost a 32-bit division a
// tile (ISA 2026-10-11, the plain all-reduce against the flat walk); the division is once, here.
namespace impl {
template <typename BODY>
DINLINE void block_for_each_tile(int tiles_m, int tiles_n, BODY body) {
  const int step_rows = static_cast<int>(gridDim.x) / tiles_n;
  const int step_cols = static_cast<int>(gridDim.x) % tiles_n;
  int row             = static_cast<int>(blockIdx.x) / tiles_n;
  int col             = static_cast<int>(blockIdx.x) % tiles_n;
  while (row < tiles_m) {
    body(row, col);
    row += step_rows;
    col += step_cols;
    if (col >= tiles_n) {
      col -= tiles_n;
      ++row;
    }
  }
}
}  // namespace impl

}  // namespace hip_custom_comms
