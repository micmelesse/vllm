// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE DOTS, by scope: a thread's share of a row's dot (a block_reduce finishes it), and the grid's
// skinny GEMM, every block striding over tiles of output columns, with the LDS contract it keeps
// (which a kernel's launch bounds and admit check read).

#pragma once

#ifndef HIP_COMMS_COMMON_INTERFACE
#error "include common/common.cuh, common's one interface, not its parts"
#endif

#include "../build.cuh"
#include "reduce.cuh"
#include "utils.cuh"

namespace hip_comms {

// This thread's share of dot(a, b) over the row, packs past its end counting zero: the partial a
// block_reduce turns into the row's dot (a sum of squares is thread_dot(x, x)).
template <int K, int N>
DINLINE float thread_dot(const float (&a)[K][N], const float (&b)[K][N],
                         const ThreadOffs<K>& thread_cols) {
  float d = 0.0f;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    float dk = 0.0f;
#pragma unroll
    for (int j = 0; j < N; ++j) dk += a[k][j] * b[k][j];
    d += thread_cols.mask_n[k] * dk;
  }
  return d;
}

// kBuild.kernels.gemm_rows and kBuild.kernels.gemm_chunk, the rows a pass and the K-chunk staged in
// LDS, are build.cuh's. Its LDS: the staged chunk and a norm's block_reduce, plus one
// [kBuild.kernels.gemm_rows][tile] float partial per wave, tile = kWaveSize / lanes columns. The
// device decides how many waves that allows.
constexpr int64_t kGemmLdsFixed =
    int64_t{kBuild.kernels.gemm_rows} * kBuild.kernels.gemm_chunk * kBuild.memory.pack_bytes +
    block_reduce_lds_bytes(1);
constexpr int64_t gemm_lds_per_wave(int lanes_per_col) {
  return int64_t{kBuild.kernels.gemm_rows} * (kWaveSize / lanes_per_col) * sizeof(float);
}
constexpr int gemm_max_waves(int lanes_per_col) {
  const int fit = lds_max_waves(kDevice, kGemmLdsFixed, gemm_lds_per_wave(lanes_per_col));
  return fit < kBuild.kernels.max_waves ? fit : kBuild.kernels.max_waves;
}
constexpr int gemm_max_threads(int lanes_per_col) {
  return gemm_max_waves(lanes_per_col) * kWaveSize;
}
static_assert(gemm_max_waves(1) >= kBuild.kernels.max_waves,
              "the GEMM tail holds the widest block at every lane split");

// out[r, n] = T(sum_k x[r][k] * w[n][k]), or with kAccumulate
// T(float(out[r, n]) + sum_k x[r][k] * w[n][k]), for r < rows, rows <= kBuild.kernels.gemm_rows,
// the sum in fp32 and rounded once. `row(r)` points at row r of x, wherever it lives.
//
// x is staged in LDS a K-chunk at a time (coalesced, once per block per chunk), so the hot
// loop's row reads are LDS reads, not a global round trip per K-step.
//
// A SKINNY GEMM: a lane keeps one column's row sums in registers; K is split over the
// kLanesPerCol lanes of a column (the build's gemm_lanes) and over the waves of the
// block; shuffles and an LDS pass add the splits; blocks stride over tiles of
// kWaveSize / kLanesPerCol columns. A column's lanes read adjacent packs of its weight row.
// The order of the sum differs from hipBLASLt's, so a result agrees to the rounding of
// the last bits, not bitwise.
template <int kLanesPerCol, bool kAccumulate, typename T, typename Row>
DINLINE void grid_gemm(Row row, int rows, const T* __restrict__ gemm_w, int n_cols, int packs,
                  T* __restrict__ out, int64_t out_stride) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  constexpr int kTile = kWaveSize / kLanesPerCol;
  static_assert(kTile * kLanesPerCol == kWaveSize, "a column's lanes must divide a wave");
  __shared__ float partial[gemm_max_waves(kLanesPerCol)][kBuild.kernels.gemm_rows][kTile];
  __shared__ V xs[kBuild.kernels.gemm_rows][kBuild.kernels.gemm_chunk];
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
    float acc[kBuild.kernels.gemm_rows];
#pragma unroll
    for (int r = 0; r < kBuild.kernels.gemm_rows; ++r) acc[r] = 0.0f;
    for (int k0 = 0; k0 < packs; k0 += kBuild.kernels.gemm_chunk) {
      const int chunk = min(kBuild.kernels.gemm_chunk, packs - k0);
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
        for (int r = 0; r < kBuild.kernels.gemm_rows; ++r) {
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
    for (int r = 0; r < kBuild.kernels.gemm_rows; ++r)
#pragma unroll
      for (int s = kTile; s < kWaveSize; s <<= 1)
        acc[r] += __shfl_xor(acc[r], s, kWaveSize);
    if (lane < kTile) {
#pragma unroll
      for (int r = 0; r < kBuild.kernels.gemm_rows; ++r) partial[wave][r][column] = acc[r];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kBuild.kernels.gemm_rows * kTile; i += blockDim.x) {
      const int r   = i / kTile;
      const int col = tile * kTile + i % kTile;
      if (r < rows && col < n_cols) {
        float v = 0.0f;
        for (int q = 0; q < waves; ++q) v += partial[q][r][i % kTile];
        T* at = out + r * out_stride + col;
        if constexpr (kAccumulate) *at = static_cast<T>(static_cast<float>(*at) + v);
        else *at = static_cast<T>(v);
      }
    }
    // Before the next tile overwrites `partial`.
    __syncthreads();
  }
}

}  // namespace hip_comms
