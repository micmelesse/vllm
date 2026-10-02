// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE TILE: how a kernel cuts its work. The rows a block works on at once, cut to a slice of
// columns, and this thread's packs of each.

#pragma once

#ifndef HIP_COMMS_COMMON_INTERFACE
#error "include common/common.cuh, common's one interface, not its parts"
#endif

#include "utils.cuh"

namespace hip_comms {

// WHO HOLDS WHAT: a Tile is the kRows rows a block works on at once, cut to `len` columns from
// `first`, and this thread's kPacks packs of each, at the same columns in every row (CuTe's and
// Triton's tile; the thread's share is CUTLASS's fragment). How a kernel cuts its work, by rows and
// by columns, is its tile's shape. Its indices are clamped into the slice and a pack past its end is
// weighted zero (`in`), so nothing reading it branches: a load under a runtime `if` cannot be
// hoisted past the branch, and loads meant to be in flight together then wait one at a time.
//
// Packs first + threadIdx.x + k * blockDim.x, k < kPacks: `at` clamped into the slice, `in` 1 for a
// pack inside it and 0 past its end.
template <int kRows, int kPacks>
struct Tile {
  static constexpr int rows  = kRows;
  static constexpr int packs = kPacks;
  int at[kPacks];
  float in[kPacks];
};

template <int kRows, int kPacks>
DINLINE Tile<kRows, kPacks> tile(int len, int first = 0) {
  Tile<kRows, kPacks> t;
#pragma unroll
  for (int k = 0; k < kPacks; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    t.at[k]     = first + (i < len ? i : len - 1);
    t.in[k]     = i < len ? 1.0f : 0.0f;
  }
  return t;
}

}  // namespace hip_comms
