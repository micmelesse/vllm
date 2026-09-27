// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot all-reduce, then RMSNorm, then a GEMM whose result is added into an output: the
// tail of Kimi-K3's latent MoE (`latent_moe_runner._shard_up_proj_tail`).

#pragma once

#include "ipc.cuh"
#include "utils.cuh"

namespace hip_comms {

// The most rows the GEMM phase takes: each wave holds kGemmRows x kGemmCols accumulators.
// A decode step is 16 tokens; a larger batch goes to the unfused path.
constexpr int kGemmRows = 16;
constexpr int kGemmCols = 4;

// Matching the model's ops rounding for rounding where they round:
//
//   n   = rms_norm(T(sum over ranks), norm_w, eps)   `add_rms_norm_row`, landing as T
//   out[:, col0:col0+N] = T(float(out) + n @ W^T)   the GEMM in fp32, rounded once, added
//
// The GEMM's sum runs in a different order than hipBLASLt's, so a result agrees to the
// rounding of the last bits, not bitwise.
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) allreduce_one_shot_rms_norm_gemm_add(
    ipc::Peers p, const T* __restrict__ norm_w, float eps, const T* __restrict__ gemm_w,
    int n_cols, T* __restrict__ out, int64_t out_stride, int out_col0, int rows,
    int packs) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  ipc::Comm<T, ngpus> c(p);
  const int rank = c.rank();

  // PHASE 1 -- this block's rows, reduced and normed, into our own scratch.
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  for (int row = blockIdx.x; row < rows; row += gridDim.x)
    add_rms_norm_row<T, T, false>(
        c, nullptr, reinterpret_cast<const V*>(norm_w), row, packs, inv_hidden, eps,
        [](int, const V&) {}, [&](int i, const V& v) { c.put(rank, row * packs + i, v); });

  // Phase 2 reads rows other blocks wrote.
  c.sync();

  // PHASE 2 -- each wave, kGemmCols output columns for every row: the latent row is loaded
  // once per pack and used for all of them, the weight rows once each.
  const int lane  = threadIdx.x % warpSize;
  const int waves = gridDim.x * (blockDim.x / warpSize);
  const int wave  = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  const V* wv     = reinterpret_cast<const V*>(gemm_w);
  for (int c0 = wave * kGemmCols; c0 < n_cols; c0 += waves * kGemmCols) {
    float acc[kGemmRows][kGemmCols];
#pragma unroll
    for (int r = 0; r < kGemmRows; ++r)
#pragma unroll
      for (int c = 0; c < kGemmCols; ++c) acc[r][c] = 0.0f;
    for (int k = lane; k < packs; k += warpSize) {
      float w[kGemmCols][NL];
#pragma unroll
      for (int c = 0; c < kGemmCols; ++c) {
        if (c0 + c >= n_cols) break;
        const V x = wv[(c0 + c) * packs + k];
#pragma unroll
        for (int j = 0; j < NL; ++j) w[c][j] = static_cast<float>(x.d[j]);
      }
#pragma unroll
      for (int r = 0; r < kGemmRows; ++r) {
        if (r >= rows) break;
        const V x = c.get(rank, r * packs + k);
        float a[NL];
#pragma unroll
        for (int j = 0; j < NL; ++j) a[j] = static_cast<float>(x.d[j]);
#pragma unroll
        for (int c = 0; c < kGemmCols; ++c) {
          if (c0 + c >= n_cols) break;
#pragma unroll
          for (int j = 0; j < NL; ++j) acc[r][c] += a[j] * w[c][j];
        }
      }
    }
    // The wave's partial sums, reduced so every lane holds every total; lane (r, c) writes.
#pragma unroll
    for (int r = 0; r < kGemmRows; ++r) {
      if (r >= rows) break;
#pragma unroll
      for (int c = 0; c < kGemmCols; ++c) {
        float v = acc[r][c];
        for (int off = warpSize / 2; off > 0; off >>= 1) v += __shfl_xor(v, off, warpSize);
        if (lane == r * kGemmCols + c && c0 + c < n_cols) {
          T* at = out + r * out_stride + out_col0 + c0 + c;
          *at   = static_cast<T>(static_cast<float>(*at) + v);
        }
      }
    }
  }
  c.close();
}

}  // namespace hip_comms
