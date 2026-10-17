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
template <int THREADS, NormPart PART, bool ROUND_SCALED, typename W, typename T, int ROWS, int K,
          int P>
DINLINE void rms_norm(const float (&input)[ROWS][K][P], float (&scale)[ROWS],
                      const W (&weight)[1][K][P], T (&output)[ROWS][K][P], float eps,
                      float inv_hidden, int size_n, int offs_n) {
  if constexpr (PART != NormPart::apply) {
    float ss[1][ROWS];
    partial_dot<THREADS>(input, input, ss[0], size_n, offs_n);
    block_reduce<Sum, ReduceAxis::all, 1, THREADS>(ss, ss);
#pragma unroll
    for (int m = 0; m < ROWS; ++m) scale[m] = rsqrtf(ss[0][m] * inv_hidden + eps);
  }
  if constexpr (PART != NormPart::scale) {
    float normed_f[ROWS][K][P];
    mul(input, scale, normed_f);
    if constexpr (ROUND_SCALED) round<W>(normed_f);
    float weight_f[1][K][P];
    convert(weight, weight_f);
    mul(normed_f, weight_f, normed_f);
    if constexpr (ROUND_SCALED) {
      W normed_w[ROWS][K][P];
      convert(normed_f, normed_w);
      convert(normed_w, output);
    } else {
      convert(normed_f, output);
    }
  }
}

}  // namespace hip_custom_communicator
