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
// offs_n to N, a block's work at once; kPacks is the packs a thread holds of each row. Columns count
// packs (16 bytes), the unit every pointer here indexes. Row m of the tile is real when
// offs_m + m < M (the last tile may be short); a scale-add slice is a tile whose N is the slice's
// end. The type carries what sizes registers (BLOCK_M rows, kPacks packs a thread of each); the
// values say where the tile is.
template <int BLOCK_M, int kPacks>
struct Tile {
  int M, N;
  int offs_m, offs_n;
};

// THIS THREAD'S COLUMNS OF A TILE, the same in every row: packs offs_n + threadIdx.x + k *
// blockDim.x, k < K, and their mask, 1 inside the tile and 0 past N (a float, multiplied in, so
// nothing reading it branches: a load under a runtime `if` cannot be hoisted past the branch, and
// loads meant to be in flight together then wait one at a time). Past N an offset is clamped to the
// last column, so every load stays in bounds.
template <int K>
struct ThreadOffs {
  int offs_n[K];
  float mask_n[K];
};

template <int BLOCK_M, int K>
DINLINE ThreadOffs<K> thread_offs(const Tile<BLOCK_M, K>& t) {
  ThreadOffs<K> cols;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int n  = t.offs_n + threadIdx.x + k * blockDim.x;
    cols.offs_n[k] = n < t.N ? n : t.N - 1;
    cols.mask_n[k] = n < t.N ? 1.0f : 0.0f;
  }
  return cols;
}

}  // namespace hip_comms
