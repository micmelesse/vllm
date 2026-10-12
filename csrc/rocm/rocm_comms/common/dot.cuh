// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE DOT: a thread's share of a row's dot (a block_reduce finishes it).

#pragma once

#ifndef HIP_COMMS_COMMON_INTERFACE
#error "include common/interface.cuh, common's one interface, not its parts"
#endif

#include "build.cuh"
#include "reduce.cuh"
#include "utils.cuh"

namespace hip_comms {

// This thread's share of dot(a, b) over each row of the tile, columns past N counting zero: the
// partials a block_reduce turns into the rows' dots (a sum of squares is partial_dot(x, x)). A
// one-row b (a weight) is every row's.
namespace impl {
template <typename A, typename B>
DINLINE void partial_dot(const A& a, const B& b, float (&d)[A::kRows]) {
  static_assert(A::kThreadsM == 1, "a dot's partials reduce over the block, so a row is the block's");
  static_assert(B::kRows == A::kRows || B::kRows == 1, "b is a's shape or one row");
#pragma unroll
  for (int m = 0; m < A::kRows; ++m) {
    const int mb = B::kRows == 1 ? 0 : m;
    d[m] = 0.0f;
#pragma unroll
    for (int k = 0; k < A::K; ++k) {
      float dk = 0.0f;
#pragma unroll
      for (int j = 0; j < A::kPack; ++j) dk = fmaf(static_cast<float>(a.v[m][k][j]), static_cast<float>(b.v[mb][k][j]), dk);
      d[m] += a.mask(k) * dk;
    }
  }
}
}  // namespace impl

}  // namespace hip_comms
