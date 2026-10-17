// SPDX-License-Identifier: MIT
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE RMSNORM ROW, written once, for the norm kernels and AttnRes's output norm.

#pragma once

#include "common.cuh"

namespace hip_custom_communicator {

// WHAT A CALL DOES OF THE ROW: its scale only (the pulled two-shot's ADD_RESIDUAL leaves the
// residual and its scale for the gather to norm), the norm at a given scale (that gather), or both.
enum class NormPart : int { scale = 0, apply = 1, row = 2 };

// THE NORM OF A THREAD'S ROWS, `input` (the rows' fp32 values, columns offs_n of size_n), with the
// weight's tile (every row's, loaded by the caller so it is in flight under the reduction):
//   scale[m] = rsqrt(mean(input[m]^2) + eps), one block reduction for every row
//   output   = T(W(W(input * scale) * float(weight)))      vLLM's rounding (ROUND_SCALED)
//   output   = T(input * scale * float(weight))            AttnRes's output norm (its reference)
// The variance is of `input` before any further rounding. A block's threads call it together
// (scale's block_reduce).
template <int THREADS_PER_BLOCK, NormPart PART, bool ROUND_SCALED, typename W, typename T, int ROWS, int K,
          int P>
DINLINE void rms_norm(const float (&input)[ROWS][K][P], float (&scale)[ROWS],
                      const W (&weight)[1][K][P], T (&output)[ROWS][K][P], float eps,
                      float inv_hidden, int size_n, int offs_n) {
  if constexpr (PART != NormPart::apply) {
    // This thread's share of each row's sum of squares, packs past size_n counting zero: its mask,
    // 1 below size_n and 0 past it, is multiplied in so nothing branches (load clamps a pack past
    // the clip to the last in bounds, and a load under a runtime `if` cannot be hoisted past it).
    float ss[1][ROWS];
#pragma unroll
    for (int m = 0; m < ROWS; ++m) {
      ss[0][m] = 0.0f;
#pragma unroll
      for (int k = 0; k < K; ++k) {
        float ss_k = 0.0f;
#pragma unroll
        for (int j = 0; j < P; ++j) ss_k = fmaf(input[m][k][j], input[m][k][j], ss_k);
        const int n = offs_n / P + static_cast<int>(threadIdx.x) % THREADS_PER_BLOCK + k * THREADS_PER_BLOCK;
        ss[0][m] += (n < size_n / P ? 1.0f : 0.0f) * ss_k;
      }
    }
    block_reduce<Sum, ReduceAxis::all, 1, THREADS_PER_BLOCK>(ss, ss);
#pragma unroll
    for (int m = 0; m < ROWS; ++m) scale[m] = rsqrtf(ss[0][m] * inv_hidden + eps);
  }
  if constexpr (PART != NormPart::scale) {
    // Each rounding the reference does a pass of its own over the tile (one fused pass's
    // conversions were not paired: +100 instructions in AttnRes, ISA I4).
    float normed[ROWS][K][P];
#pragma unroll
    for (int m = 0; m < ROWS; ++m)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < P; ++j) normed[m][k][j] = input[m][k][j] * scale[m];
    if constexpr (ROUND_SCALED) {
#pragma unroll
      for (int m = 0; m < ROWS; ++m)
#pragma unroll
        for (int k = 0; k < K; ++k)
#pragma unroll
          for (int j = 0; j < P; ++j)
            normed[m][k][j] = static_cast<float>(static_cast<W>(normed[m][k][j]));
    }
#pragma unroll
    for (int m = 0; m < ROWS; ++m)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < P; ++j) normed[m][k][j] *= static_cast<float>(weight[0][k][j]);
#pragma unroll
    for (int m = 0; m < ROWS; ++m)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < P; ++j)
          output[m][k][j] = ROUND_SCALED ? static_cast<T>(static_cast<W>(normed[m][k][j]))
                                         : static_cast<T>(normed[m][k][j]);
  }
}

}  // namespace hip_custom_communicator
