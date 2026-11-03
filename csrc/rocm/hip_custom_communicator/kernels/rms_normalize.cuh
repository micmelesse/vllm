// SPDX-License-Identifier: MIT Copyright (c) 2026, Advanced Micro Devices, Inc.
// All rights reserved.
//
// THE RMSNORM ROW, written once, for the norm kernels and AttnRes's output
// norm.

#pragma once

#include "common.cuh"

namespace hip_custom_communicator {

// WHAT A CALL DOES OF THE ROW: its scale only (the pulled two-shot's
// ADD_RESIDUAL leaves the residual and its scale for the gather to norm), the
// norm at a given scale (that gather), or both.
// WIDE: a chunk of a row wider than the tile (the chunk at offs_n, the tile's K
// * P * THREADS_PER_BLOCK columns), its squares added to `scale` (zeros before
// the row's first chunk); at the row's last chunk the sums become the scales. A
// WIDE ROW IS NORMED IN TWO PASSES over its chunks: this one, then each chunk's
// values computed afresh and normed at the scale (apply).
enum class NormPart : int { scale = 0, apply = 1, row = 2, wide = 3 };

// THE NORM OF A THREAD'S ROWS, `input` (the rows' fp32 values, columns offs_n
// of size_n), with the weight's tile (every row's, loaded by the caller so it
// is in flight under the reduction): scale[m] = rsqrt(mean(input[m]^2) + eps),
// one block reduction for every row output   = T(W(W(input * scale) *
// float(weight)))      vLLM's rounding (ROUND_SCALED) output   = T(input *
// scale * float(weight))            AttnRes's output norm (its reference) The
// variance is of `input` before any further rounding. A block's threads call it
// together (scale's reduce). `layout` is how the thread's packs lie in the row,
// for the mask of the packs past size_n: Layout::tile (pack k of thread t at t
// + k * THREADS_PER_BLOCK) or Layout::group_run (whole groups a thread, group j
// of thread t the row's t + j * THREADS_PER_BLOCK, its kInt4GroupPacks packs
// consecutive: common.cuh's Layout); a literal at every call.
template <int THREADS_PER_BLOCK, int WAVE_SIZE, typename W, typename T,
          int ROWS, int K, int P>
DINLINE void rms_normalize(const float (&input)[ROWS][K][P],
                           float (&scale)[ROWS], const W (&weight)[1][K][P],
                           T (&output)[ROWS][K][P], float eps, float inv_hidden,
                           int size_n, int offs_n, NormPart part,
                           bool round_scaled, Layout layout) {
  if (part != NormPart::apply) {
    // This thread's share of each row's sum of squares, packs past size_n
    // counting zero, by a select (nothing branches: load clamps a pack past the
    // clip to the last in bounds, and a load under a runtime `if` cannot be
    // hoisted past it). A select, not a multiply by 0: a lane past size_n may
    // hold anything (its clamped pack another thread's, not yet written), and 0
    // * NaN is NaN, the row's whole scale with it.
    float ss[1][ROWS];
#pragma unroll
    for (int m = 0; m < ROWS; ++m) {
      ss[0][m] = part == NormPart::wide ? scale[m] : 0.0f;
#pragma unroll
      for (int k = 0; k < K; ++k) {
        float ss_k = 0.0f;
#pragma unroll
        for (int j = 0; j < P; ++j)
          ss_k = fmaf(input[m][k][j], input[m][k][j], ss_k);
        const int t = static_cast<int>(threadIdx.x) % THREADS_PER_BLOCK;
        const int n =
            layout == Layout::tile
                ? offs_n / P + t + k * THREADS_PER_BLOCK
                : offs_n / P +
                      ((k / kInt4GroupPacks) * THREADS_PER_BLOCK + t) *
                          kInt4GroupPacks +
                      k % kInt4GroupPacks;
        ss[0][m] += n < size_n / P ? ss_k : 0.0f;
      }
    }
    if (part == NormPart::wide && offs_n + K * P * THREADS_PER_BLOCK < size_n) {
#pragma unroll
      for (int m = 0; m < ROWS; ++m)
        scale[m] = ss[0][m];  // the row's next chunk adds to it
      return;
    }
    reduce<1, THREADS_PER_BLOCK, WAVE_SIZE, ReduceAxis::all>(ss, ss);
#pragma unroll
    for (int m = 0; m < ROWS; ++m)
      scale[m] = rsqrtf(ss[0][m] * inv_hidden + eps);
  }
  if (part != NormPart::scale && part != NormPart::wide) {
    // Each rounding the reference does a pass of its own over the tile (one
    // fused pass's conversions were not paired: +100 instructions in AttnRes,
    // ISA I4).
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
            normed[m][k][j] =
                static_cast<float>(static_cast<W>(normed[m][k][j]));
    }
#pragma unroll
    for (int m = 0; m < ROWS; ++m)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < P; ++j)
          normed[m][k][j] *= static_cast<float>(weight[0][k][j]);
#pragma unroll
    for (int m = 0; m < ROWS; ++m)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < P; ++j)
          output[m][k][j] =
              round_scaled ? static_cast<T>(static_cast<W>(normed[m][k][j]))
                           : static_cast<T>(normed[m][k][j]);
  }
}

}  // namespace hip_custom_communicator
