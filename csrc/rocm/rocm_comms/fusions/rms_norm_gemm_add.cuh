// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE COMPUTATION of all-reduce + RMSNorm + GEMM-add, the tail of Kimi-K3's latent MoE
// (`fused_all_reduce.latent_tail`), in its two stages: `norm_row` on a row already reduced
// over ranks, then (after the kernel's barrier: the GEMM reads every normed row) `gemm`.
// Every rms_norm_gemm_add kernel, pull or push.

#pragma once

#include "../common/memory.cuh"
#include "../common/reduce.cuh"

namespace hip_comms::fusions::rms_norm_gemm_add {

// STAGE 1: one row RMSNormed by the whole block, matching vLLM's `rms_norm`
// (`vllm/ir/ops/layernorm.py`) rounding for rounding, the weight in T:
//
//   s   = float(T(sum over ranks))        the all-reduce output, as it would land
//   out = T(T(s * rsqrt(mean(s^2) + eps)) * float(w))
//
// `sum` is this thread's share of the row, sum[k] the pack threadIdx.x + k * blockDim.x;
// the normed packs leave through `store(k, i, v)`, i the pack within the row.
template <typename T, int K, typename Store>
DINLINE void norm_row(const typename traits<T>::V (&sum)[K],
                      const typename traits<T>::V* weight, int packs, float inv_hidden,
                      float eps, Store store) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  float s[K][NL];
  float acc = 0.0f;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    if (i >= packs) break;
#pragma unroll
    for (int j = 0; j < NL; ++j) {
      s[k][j] = static_cast<float>(sum[k].d[j]);
      acc += s[k][j] * s[k][j];
    }
  }
  const float scale = rsqrtf(block_sum(acc) * inv_hidden + eps);
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    if (i >= packs) break;
    const V w = weight[i];
    V o;
#pragma unroll
    for (int j = 0; j < NL; ++j) {
      const float x = static_cast<float>(static_cast<T>(s[k][j] * scale));
      o.d[j]        = static_cast<T>(x * static_cast<float>(w.d[j]));
    }
    store(k, i, o);
  }
  // Before the next row reuses `block_sum`'s shared slots.
  __syncthreads();
}

// STAGE 2, THE GEMM.
// The most rows one pass takes: a lane holds one output column's sums for each of them.
constexpr int kRows = 16;
// The K-chunk of x staged in LDS at a time, in packs. With 160 KiB of LDS all of Kimi-K3's latent
// K (448 packs, 112 KiB) goes in at once: one staging pass and one barrier pair per tile. With
// 64 KiB, 96 packs (24 KiB).
constexpr int kChunk = kDevice.lds_bytes >= 160 * kKiB ? 448 : 96;

// ITS LDS: the staged chunk and the block sums, plus one [kRows][tile] float partial per wave,
// tile = kWaveSize / lanes columns. The device decides how many waves that allows.
constexpr int64_t kLdsFixed = int64_t{kRows} * kChunk * kPackBytes + kBlockSumLdsBytes;
constexpr int64_t lds_per_wave(int lanes_per_col) {
  return int64_t{kRows} * (kWaveSize / lanes_per_col) * sizeof(float);
}
constexpr int max_waves(int lanes_per_col) {
  return lds_max_waves(kDevice, kLdsFixed, lds_per_wave(lanes_per_col));
}
constexpr int max_threads(int lanes_per_col) { return max_waves(lanes_per_col) * kWaveSize; }
static_assert(max_waves(1) >= 8, "the GEMM tail holds 512 threads at every lane split");

// out[r, col0 + n] = T(float(out[r, col0 + n]) + sum_k x[r][k] * w[n][k]) for r < rows,
// rows <= kRows, the sum in fp32 and rounded once. `row(r)` points at row r of x,
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
DINLINE void gemm(Row row, int rows, const T* __restrict__ gemm_w, int n_cols, int packs,
                  T* __restrict__ out, int64_t out_stride, int out_col0) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  constexpr int kTile = kWaveSize / kLanesPerCol;
  static_assert(kTile * kLanesPerCol == kWaveSize, "a column's lanes must divide a wave");
  __shared__ float partial[max_waves(kLanesPerCol)][kRows][kTile];
  __shared__ V xs[kRows][kChunk];
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
    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.0f;
    for (int k0 = 0; k0 < packs; k0 += kChunk) {
      const int chunk = min(kChunk, packs - k0);
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
        for (int r = 0; r < kRows; ++r) {
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
    for (int r = 0; r < kRows; ++r)
#pragma unroll
      for (int s = kTile; s < kWaveSize; s <<= 1)
        acc[r] += __shfl_xor(acc[r], s, kWaveSize);
    if (lane < kTile) {
#pragma unroll
      for (int r = 0; r < kRows; ++r) partial[wave][r][column] = acc[r];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kRows * kTile; i += blockDim.x) {
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

}  // namespace hip_comms::fusions::rms_norm_gemm_add
