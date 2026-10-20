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
// (scale's reduce). `layout` is how the thread's packs lie in the row, for the mask of the packs
// past size_n: Layout::tile (pack k of thread t at t + k * THREADS_PER_BLOCK) or Layout::group_run
// (a group a thread: its K packs consecutive, at t * K + k); a literal at every call.
// A THREAD'S SHARE OF EACH ROW'S SUM OF SQUARES, added to `ss`: packs past size_n count zero, their
// mask multiplied in so nothing branches (load clamps a pack past the clip to the last in bounds,
// and a load under a runtime `if` cannot be hoisted past it). `layout` as rms_norm's.
template <int THREADS_PER_BLOCK, int ROWS, int K, int P>
DINLINE void add_squares(const float (&input)[ROWS][K][P], float (&ss)[1][ROWS], int size_n,
                         int offs_n, Layout layout) {
#pragma unroll
  for (int m = 0; m < ROWS; ++m) {
#pragma unroll
    for (int k = 0; k < K; ++k) {
      float ss_k = 0.0f;
#pragma unroll
      for (int j = 0; j < P; ++j) ss_k = fmaf(input[m][k][j], input[m][k][j], ss_k);
      const int t = static_cast<int>(threadIdx.x) % THREADS_PER_BLOCK;
      const int n = layout == Layout::tile ? offs_n / P + t + k * THREADS_PER_BLOCK
                                           : offs_n / P + t * K + k;
      ss[0][m] += (n < size_n / P ? 1.0f : 0.0f) * ss_k;
    }
  }
}

// EACH ROW'S SCALE from the threads' shares of its sum of squares, one block reduction for every
// row: rsqrt(mean(input^2) + eps).
template <int THREADS_PER_BLOCK, int ROWS>
DINLINE void scales_of(float (&ss)[1][ROWS], float (&scale)[ROWS], float inv_hidden, float eps) {
  reduce<1, THREADS_PER_BLOCK, ReduceAxis::all>(ss, ss);
#pragma unroll
  for (int m = 0; m < ROWS; ++m) scale[m] = rsqrtf(ss[0][m] * inv_hidden + eps);
}

template <int THREADS_PER_BLOCK, typename W, typename T, int ROWS, int K,
          int P>
DINLINE void rms_norm(const float (&input)[ROWS][K][P], float (&scale)[ROWS],
                      const W (&weight)[1][K][P], T (&output)[ROWS][K][P], float eps,
                      float inv_hidden, int size_n, int offs_n, NormPart part,
                      bool round_scaled, Layout layout) {
  if (part != NormPart::apply) {
    float ss[1][ROWS];
#pragma unroll
    for (int m = 0; m < ROWS; ++m) ss[0][m] = 0.0f;
    add_squares<THREADS_PER_BLOCK>(input, ss, size_n, offs_n, layout);
    scales_of<THREADS_PER_BLOCK>(ss, scale, inv_hidden, eps);
  }
  if (part != NormPart::scale) {
    // Each rounding the reference does a pass of its own over the tile (one fused pass's
    // conversions were not paired: +100 instructions in AttnRes, ISA I4).
    float normed[ROWS][K][P];
#pragma unroll
    for (int m = 0; m < ROWS; ++m)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < P; ++j) normed[m][k][j] = input[m][k][j] * scale[m];
    if (round_scaled) {
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
          output[m][k][j] = round_scaled ? static_cast<T>(static_cast<W>(normed[m][k][j]))
                                         : static_cast<T>(normed[m][k][j]);
  }
}

// A ROW WIDER THAN THE TILE: its scale over its chunks of `tile_n` columns (a tile's, from column
// 0), each chunk's fp32 values computed by `values(c, s)` (the chunk at column c of the block's
// ROWS rows, in LAYOUT) and their squares summed, one block reduction at the end. A WIDE ROW IS
// NORMED IN TWO PASSES over its chunks, this one and the caller's, which computes each chunk's
// values afresh and norms them at this scale (NormPart::apply): `values` reads only what the
// kernel does not write before its second pass, and writes nothing (the second pass writes).
template <int THREADS_PER_BLOCK, Layout LAYOUT, int ROWS, int K, int P, typename VALUES>
DINLINE void wide_scale(int size_n, int tile_n, float eps, float inv_hidden, VALUES&& values,
                        float (&scale)[ROWS]) {
  float ss[1][ROWS];
#pragma unroll
  for (int m = 0; m < ROWS; ++m) ss[0][m] = 0.0f;
  for (int c = 0; c < size_n; c += tile_n) {
    float s[ROWS][K][P];
    values(c, s);
    add_squares<THREADS_PER_BLOCK>(s, ss, size_n, c, LAYOUT);
  }
  scales_of<THREADS_PER_BLOCK>(ss, scale, inv_hidden, eps);
}

}  // namespace hip_custom_communicator
