// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE TILE: how a kernel cuts its work, in Triton's terms (BLOCK_M, offs_m, M, a mask for the last
// tile), and a thread's columns of it.

#pragma once

#ifndef HIP_COMMS_COMMON_INTERFACE
#error "include common/common.cuh, common's one interface, not its parts"
#endif

#include "utils.cuh"

namespace hip_comms {

// A TILE, in Triton's terms: BLOCK_M rows of an M x N tensor from row offs_m, and its columns from
// offs_n to N, a block's work at once. Rows and columns count elements, as the tensor does; a pack
// (16 bytes, traits<T>::N elements) is only how a thread reads its columns, so thread_offs derives
// it from the dtype. Row m of the tile is real when offs_m + m < M (the last tile may be short); a
// scale-add slice is a tile whose N is the slice's end. The type carries what sizes registers
// (BLOCK_M rows, kPacks packs a thread of each); the values say where the tile is.
template <int BLOCK_M, int kPacks>
struct Tile {
  int M, N;
  int offs_m, offs_n;
};

// THIS THREAD'S COLUMNS OF A TILE, in packs, the same in every row: packs offs_n / pack +
// threadIdx.x + k * blockDim.x, k < K, and their mask, 1 inside the tile and 0 past N (a float,
// multiplied in, so nothing reading it branches: a load under a runtime `if` cannot be hoisted past
// the branch, and loads meant to be in flight together then wait one at a time). Past N an offset
// is clamped to the last pack, so every load stays in bounds. N and offs_n are whole packs.
template <int K>
struct ThreadOffs {
  int offs_n[K];
  float mask_n[K];
};

template <typename T, int BLOCK_M, int K>
DINLINE ThreadOffs<K> thread_offs(const Tile<BLOCK_M, K>& tile) {
  constexpr int pack = traits<T>::N;
  const int first = tile.offs_n / pack, end = tile.N / pack;
  ThreadOffs<K> cols;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int n = first + threadIdx.x + k * blockDim.x;
    cols.offs_n[k] = n < end ? n : end - 1;
    cols.mask_n[k] = n < end ? 1.0f : 0.0f;
  }
  return cols;
}

}  // namespace hip_comms
