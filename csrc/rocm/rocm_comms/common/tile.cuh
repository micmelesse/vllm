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
// this thread holds. Rows and columns count elements, as the tensor does; how a thread holds its
// columns (packs of NL elements, its own every THREADS-th) is the tile's business, not the
// kernel's. Row m is real when offs_m + m < M and a column when it is below N (the last tile may be
// short; a scale-add slice is a tile whose N is the slice's end). E is the element type: a tile of
// bf16 as it is loaded, `to<float>()` for the math, the same layout whatever E is.
template <typename E, int TILE_M, int TILE_N, int THREADS, int NL = traits<E>::N>
struct Tile {
  static_assert(TILE_N % (NL * THREADS) == 0, "a tile's columns are whole packs for every thread");
  static constexpr int K = TILE_N / (NL * THREADS);  // packs of a row this thread holds
  using Pack = vec<E, NL>;
  int M, N;
  int offs_m, offs_n;
  Pack v[TILE_M][K];

  // THIS THREAD'S PACK k OF A ROW, in packs from the row's start: past N clamped to the last pack,
  // so every load stays in bounds; its mask is 1 below N and 0 past it (a float, multiplied in, so
  // nothing reading it branches: a load under a runtime `if` cannot be hoisted past the branch, and
  // loads meant to be in flight together then wait one at a time).
  DINLINE int col(int k) const {
    const int n = offs_n / NL + static_cast<int>(threadIdx.x) + k * THREADS;
    return n < N / NL ? n : N / NL - 1;
  }
  DINLINE float mask(int k) const {
    return offs_n / NL + static_cast<int>(threadIdx.x) + k * THREADS < N / NL ? 1.0f : 0.0f;
  }
  // Row m, a row past M reading row M - 1 (and storing nothing).
  DINLINE bool live(int m) const { return offs_m + m < M; }
  DINLINE int row(int m) const { return live(m) ? offs_m + m : M - 1; }

  // The same place in another element type, its elements converted (rounded once) or empty.
  template <typename U>
  DINLINE Tile<U, TILE_M, TILE_N, THREADS, NL> to() const {
    Tile<U, TILE_M, TILE_N, THREADS, NL> t{M, N, offs_m, offs_n};
#pragma unroll
    for (int m = 0; m < TILE_M; ++m)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < NL; ++j) t.v[m][k].d[j] = static_cast<U>(v[m][k].d[j]);
    return t;
  }
  template <typename U>
  DINLINE Tile<U, TILE_M, TILE_N, THREADS, NL> like() const {
    return {M, N, offs_m, offs_n};
  }
};

}  // namespace hip_comms
