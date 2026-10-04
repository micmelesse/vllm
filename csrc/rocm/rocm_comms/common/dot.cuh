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

#include "build.cuh"
#include "reduce.cuh"
#include "utils.cuh"

namespace hip_comms {

// This thread's share of dot(a, b) over each row of the tile, columns past N counting zero: the
// partials a block_reduce turns into the rows' dots (a sum of squares is partial_dot(x, x)). A
// one-row b (a weight) is every row's.
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
      for (int j = 0; j < A::kPack; ++j) dk += a.v[m][k][j] * b.v[mb][k][j];
      d[m] += a.mask(k) * dk;
    }
  }
}

// THE GEMM TAIL'S LDS, for its tile: a TILE_M x TILE_K chunk of x staged, a norm's block_reduce,
// and a [TILE_M][wave / SLICE_K] fp32 partial per wave. The device decides how many waves fit.
constexpr int64_t gemm_lds_fixed(int tile_m, int tile_k) {
  return int64_t{tile_m} * tile_k * kBuild.memory.pack_bytes + block_reduce_lds_bytes(1);
}
constexpr int64_t gemm_lds_per_wave(const Hardware& hw, int tile_m, int slice_k) {
  return int64_t{tile_m} * (hw.wave_size / slice_k) * sizeof(float);
}
constexpr int gemm_max_waves(const Hardware& hw, int tile_m, int tile_k, int slice_k) {
  const int fit =
      lds_max_waves(hw, gemm_lds_fixed(tile_m, tile_k), gemm_lds_per_wave(hw, tile_m, slice_k));
  return fit < kBuild.kernels.max_waves ? fit : kBuild.kernels.max_waves;
}
constexpr int gemm_max_threads(const Hardware& hw, int tile_m, int tile_k, int slice_k) {
  return gemm_max_waves(hw, tile_m, tile_k, slice_k) * hw.wave_size;
}
// Whether a GEMM-tail config fits `hw`: its block's waves and their partials in LDS beside the
// staged chunk. A config built for the tuning target that `hw` cannot hold compiles to a trap there
// (check refuses it first: a device that is not the target is never tuned, `supported`).
constexpr bool gemm_fits(const Hardware& hw, int tile_m, int tile_k, int slice_k, int threads) {
  return gemm_lds_fixed(tile_m, tile_k) < hw.lds_bytes &&
         threads <= gemm_max_threads(hw, tile_m, tile_k, slice_k);
}
// THE LARGEST TILE_K THE LDS STAGES at `tile_m` rows: what `hw` holds beside the widest block's
// partials at one lane a column (the most a split gives) and a block_reduce's.
constexpr int gemm_tile_k_fit(const Hardware& hw, int tile_m) {
  const int64_t partials = int64_t{kBuild.kernels.max_waves} * tile_m * hw.wave_size * 4;
  const int64_t reduce = (int64_t{kBuild.kernels.max_waves} + 1) * 4;
  return static_cast<int>((hw.lds_bytes - partials - reduce) /
                          (int64_t{tile_m} * kBuild.memory.pack_bytes));
}

// out[r, n] = T(sum_k x[r][k] * w[n][k]), or with kAccumulate
// T(float(out[r, n]) + sum_k x[r][k] * w[n][k]), for r < rows, rows <= TILE_M,
// the sum in fp32 and rounded once. x is `rows` rows of `cols` elements at `x_stride` elements
// apart; how it is read (a pack a lane, staged in LDS) is this function's.
//
// x is staged in LDS TILE_K packs at a time (coalesced, once per block per chunk), so the hot
// loop's row reads are LDS reads, not a global round trip per K-step.
//
// A SKINNY GEMM: a lane keeps one column's row sums in registers; K is split over the
// SLICE_K lanes of a column (CUTLASS's sliced-K) and over the waves of the
// block; shuffles and an LDS pass add the splits; blocks stride over tiles of
// kWaveSize / SLICE_K columns. A column's lanes read adjacent packs of its weight row.
// The order of the sum differs from hipBLASLt's, so a result agrees to the rounding of
// the last bits, not bitwise.
template <int TILE_M, int TILE_K, int SLICE_K, bool ACCUMULATE, typename DTYPE>
DINLINE void grid_gemm(const DTYPE* x, int64_t x_stride, int rows, int cols,
                       const DTYPE* __restrict__ gemm_w, int n_cols, DTYPE* __restrict__ out,
                       int64_t out_stride) {
  using V          = typename traits<DTYPE>::V;
  constexpr int NL = traits<DTYPE>::N;
  const int packs  = cols / NL;
  const auto row   = [&](int r) { return reinterpret_cast<const V*>(x + r * x_stride); };
  constexpr int kTile = kWaveSize / SLICE_K;
  static_assert(kTile * SLICE_K == kWaveSize, "a column's lanes must divide a wave");
  __shared__ float partial[gemm_max_waves(kDevice, TILE_M, TILE_K, SLICE_K)][TILE_M][kTile];
  __shared__ V xs[TILE_M][TILE_K];
  const int lane   = threadIdx.x % kWaveSize;
  const int wave   = threadIdx.x / kWaveSize;
  const int waves  = blockDim.x / kWaveSize;
  const int column = lane % kTile;
  const int splits = waves * SLICE_K;
  const int split = wave * SLICE_K + lane / kTile;
  const V* wv      = reinterpret_cast<const V*>(gemm_w);
  const int tiles  = (n_cols + kTile - 1) / kTile;
  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const int n   = tile * kTile + column;
    const V* wrow = wv + static_cast<int64_t>(n < n_cols ? n : 0) * packs;
    float acc[TILE_M];
#pragma unroll
    for (int r = 0; r < TILE_M; ++r) acc[r] = 0.0f;
    for (int k0 = 0; k0 < packs; k0 += TILE_K) {
      const int chunk = min(TILE_K, packs - k0);
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
        for (int r = 0; r < TILE_M; ++r) {
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
    for (int r = 0; r < TILE_M; ++r)
#pragma unroll
      for (int s = kTile; s < kWaveSize; s <<= 1)
        acc[r] += __shfl_xor(acc[r], s, kWaveSize);
    if (lane < kTile) {
#pragma unroll
      for (int r = 0; r < TILE_M; ++r) partial[wave][r][column] = acc[r];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < TILE_M * kTile; i += blockDim.x) {
      const int r   = i / kTile;
      const int col = tile * kTile + i % kTile;
      if (r < rows && col < n_cols) {
        float v = 0.0f;
        for (int q = 0; q < waves; ++q) v += partial[q][r][i % kTile];
        DTYPE* at = out + r * out_stride + col;
        if constexpr (ACCUMULATE) *at = static_cast<DTYPE>(static_cast<float>(*at) + v);
        else *at = static_cast<DTYPE>(v);
      }
    }
    // Before the next tile overwrites `partial`.
    __syncthreads();
  }
}

}  // namespace hip_comms
