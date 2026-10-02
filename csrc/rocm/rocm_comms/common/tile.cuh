// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE TILE: how a kernel cuts its work, in Triton's terms (TILE_M, offs_m, M, a mask for the last
// tile), and a thread's columns of it.

#pragma once

#ifndef HIP_COMMS_COMMON_INTERFACE
#error "include common/common.cuh, common's one interface, not its parts"
#endif

#include "utils.cuh"

namespace hip_comms {

// A TILE, in Triton's terms: TILE_M rows of an M x N tensor from row offs_m, and TILE_N of its
// columns from offs_n, a block's work at once. Rows and columns count elements, as the tensor does;
// a pack (16 bytes, traits<T>::N elements) is only how a thread reads its columns. Row m of the
// tile is real when offs_m + m < M and a column when it is below N (the last tile may be short); a
// scale-add slice is a tile whose N is the slice's end. The type is what sizes registers; the
// values say where the tile is.
template <int TILE_M, int TILE_N>
struct Tile {
  int M, N;
  int offs_m, offs_n;
};

// THIS THREAD'S COLUMNS OF A TILE, in packs, the same in every row: packs offs_n / pack + thread +
// k * THREADS_PER_BLOCK, k < K, and their mask, 1 below N and 0 past it (a float, multiplied in, so
// nothing reading it branches: a load under a runtime `if` cannot be hoisted past the branch, and
// loads meant to be in flight together then wait one at a time). Past N an offset is clamped to the
// last pack, so every load stays in bounds. N and offs_n are whole packs.
template <int K>
struct ThreadOffs {
  int offs_n[K];
  float mask_n[K];
};

// A thread's packs of a row of a TILE_N tile over THREADS_PER_BLOCK threads.
template <typename T, int TILE_N, int THREADS_PER_BLOCK>
constexpr int packs_per_thread() {
  static_assert(TILE_N % (traits<T>::N * THREADS_PER_BLOCK) == 0,
                "a tile's columns are whole packs for every thread");
  return TILE_N / (traits<T>::N * THREADS_PER_BLOCK);
}

template <typename T, int THREADS_PER_BLOCK, int TILE_M, int TILE_N>
DINLINE ThreadOffs<packs_per_thread<T, TILE_N, THREADS_PER_BLOCK>()> thread_offs(
    const Tile<TILE_M, TILE_N>& tile) {
  constexpr int K = packs_per_thread<T, TILE_N, THREADS_PER_BLOCK>();
  constexpr int pack = traits<T>::N;
  const int first = tile.offs_n / pack, end = tile.N / pack;
  ThreadOffs<K> cols;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int n = first + threadIdx.x + k * THREADS_PER_BLOCK;
    cols.offs_n[k] = n < end ? n : end - 1;
    cols.mask_n[k] = n < end ? 1.0f : 0.0f;
  }
  return cols;
}

}  // namespace hip_comms
