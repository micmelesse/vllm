// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE GEMM PHASE of the latent MoE tail, `out[:, col0:col0+N] += x @ W^T` over normed
// rows wherever they live: every rms_norm_gemm_add kernel, pull or push.

#pragma once

#include "../utils.cuh"

namespace hip_comms {

// The most rows one pass takes: a lane holds one output column's sums for each of them.
constexpr int kGemmRows = 16;
// The K-chunk of x staged in LDS at a time, in packs. gfx950 has 160 KB of LDS, so all of
// Kimi-K3's latent K (448 packs, 112 KB) goes in at once: one staging pass and one barrier
// pair per tile. gfx942 has 64 KB: 96 packs (24 KB) beside the widest reduce tile (32 KB).
#if defined(__gfx950__)
constexpr int kGemmChunk = 448;
#else
constexpr int kGemmChunk = 96;
#endif

// out[r, col0 + n] = T(float(out[r, col0 + n]) + sum_k x[r][k] * w[n][k]) for r < rows,
// rows <= kGemmRows, the sum in fp32 and rounded once. `row(r)` points at row r of x,
// wherever it lives.
//
// x is staged in LDS a K-chunk at a time (coalesced, once per block per chunk), so the hot
// loop's row reads are LDS reads, not a global round trip per K-step.
//
// A SKINNY GEMM: a lane keeps one column's row sums in registers; K is split over the
// kLanesPerCol lanes of a column (tuned in launch.cuh) and over the waves of the
// block; shuffles and an LDS pass add the splits; blocks stride over tiles of
// kWaveSize / kLanesPerCol columns. A column's lanes read adjacent packs of its weight row.
// The order of the sum differs from hipBLASLt's, so a result agrees to the rounding of
// the last bits, not bitwise.
template <int kLanesPerCol, typename T, typename Row>
DINLINE void gemm_add_rows(Row row, int rows, const T* __restrict__ gemm_w, int n_cols,
                           int packs, T* __restrict__ out, int64_t out_stride,
                           int out_col0) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  constexpr int kTile = kWaveSize / kLanesPerCol;
  static_assert(kTile * kLanesPerCol == kWaveSize, "a column's lanes must divide a wave");
  __shared__ float partial[kMaxWaves][kGemmRows][kTile];
  __shared__ V xs[kGemmRows][kGemmChunk];
  const int lane   = threadIdx.x % kWaveSize;
  const int wave   = threadIdx.x / kWaveSize;
  const int waves  = blockDim.x / kWaveSize;
  const int column = lane % kTile;
  const int splits = waves * kLanesPerCol;
  const int split  = wave * kLanesPerCol + lane / kTile;
  const V* wv      = reinterpret_cast<const V*>(gemm_w);
  const int tiles  = (n_cols + kTile - 1) / kTile;
  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const int n   = tile * kTile + column;
    const V* wrow = wv + static_cast<int64_t>(n < n_cols ? n : 0) * packs;
    float acc[kGemmRows];
#pragma unroll
    for (int r = 0; r < kGemmRows; ++r) acc[r] = 0.0f;
    for (int k0 = 0; k0 < packs; k0 += kGemmChunk) {
      const int chunk = min(kGemmChunk, packs - k0);
      // Rows past `rows` are never staged; their sums read stale LDS and are never stored.
      for (int i = threadIdx.x; i < rows * chunk; i += blockDim.x)
        xs[i / chunk][i % chunk] = row(i / chunk)[k0 + i % chunk];
      __syncthreads();
      for (int k = split; k < chunk; k += splits) {
        const V wx = wrow[k0 + k];
        float w[NL];
#pragma unroll
        for (int j = 0; j < NL; ++j) w[j] = static_cast<float>(wx.d[j]);
#pragma unroll
        for (int r = 0; r < kGemmRows; ++r) {
          const V xr = xs[r][k];
#pragma unroll
          for (int j = 0; j < NL; ++j) acc[r] += static_cast<float>(xr.d[j]) * w[j];
        }
      }
      // Before the next chunk overwrites `xs`.
      __syncthreads();
    }
    // A column's lanes are kTile apart in the wave.
#pragma unroll
    for (int r = 0; r < kGemmRows; ++r)
#pragma unroll
      for (int s = kTile; s < kWaveSize; s <<= 1)
        acc[r] += __shfl_xor(acc[r], s, kWaveSize);
    if (lane < kTile) {
#pragma unroll
      for (int r = 0; r < kGemmRows; ++r) partial[wave][r][column] = acc[r];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kGemmRows * kTile; i += blockDim.x) {
      const int r   = i / kTile;
      const int col = tile * kTile + i % kTile;
      if (r < rows && col < n_cols) {
        float v = 0.0f;
        for (int q = 0; q < waves; ++q) v += partial[q][r][i % kTile];
        T* at = out + r * out_stride + out_col0 + col;
        *at   = static_cast<T>(static_cast<float>(*at) + v);
      }
    }
    // Before the next tile overwrites `partial`.
    __syncthreads();
  }
}

}  // namespace hip_comms
