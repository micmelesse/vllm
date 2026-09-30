// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE SUMS: a pack or a row over every rank (fp32, rounded once, in the order given), and a
// value over the threads of a block.

#pragma once

#include <cmath>

#include "../hardware.cuh"
#include "memory.cuh"
#include "pack.cuh"

namespace hip_comms {

// ONE PACK SUMMED OVER `ngpus` SOURCES, in fp32 and rounded once to T; `read(r, i)` is pack i
// of source r. Every read before any add, so the ngpus loads are in flight together; the
// sources in order, so callers whose sources are the ranks agree bitwise.
template <typename T, int ngpus, typename Read>
DINLINE typename traits<T>::V sum_packs(Read read, int64_t i) {
  using V         = typename traits<T>::V;
  constexpr int N = traits<T>::N;
  V raw[ngpus];
#pragma unroll
  for (int r = 0; r < ngpus; ++r) raw[r] = read(r, i);
  float acc[N];
#pragma unroll
  for (int j = 0; j < N; ++j) acc[j] = static_cast<float>(raw[0].d[j]);
#pragma unroll
  for (int r = 1; r < ngpus; ++r)
#pragma unroll
    for (int j = 0; j < N; ++j) acc[j] += static_cast<float>(raw[r].d[j]);
  V out;
#pragma unroll
  for (int j = 0; j < N; ++j) out.d[j] = static_cast<T>(acc[j]);
  return out;
}

// THIS THREAD'S PACKS OF ROW `row` (pack threadIdx.x + k * blockDim.x, k < K) summed over the
// `ngpus` sources, into sum[k]. A block owns the row; its norm needs every pack.
template <typename T, int ngpus, int K, typename Read>
DINLINE void sum_row(Read read, int row, int packs, typename traits<T>::V (&sum)[K]) {
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    if (i < packs) sum[k] = sum_packs<T, ngpus>(read, int64_t{row} * packs + i);
  }
}

// The LDS `block_sum` and `block_sum2` take between them, for a kernel budgeting the rest.
constexpr int64_t kBlockSumLdsBytes = (kMaxWaves + 1) * (sizeof(float) + sizeof(float2));

// Sum of `v` over the block. A block wider than kMaxThreads cannot be launched, so
// `partial` cannot be overrun.
DINLINE float block_sum(float v) {
  __shared__ float partial[kMaxWaves];
  __shared__ float total;
  const int lane = threadIdx.x % kWaveSize;
  const int warp = threadIdx.x / kWaveSize;
  for (int off = kWaveSize / 2; off > 0; off >>= 1) v += __shfl_down(v, off, kWaveSize);
  if (lane == 0) partial[warp] = v;
  __syncthreads();
  const int warps = (blockDim.x + kWaveSize - 1) / kWaveSize;
  if (warp == 0) {
    v = (lane < warps) ? partial[lane] : 0.0f;
    for (int off = kWaveSize / 2; off > 0; off >>= 1) v += __shfl_down(v, off, kWaveSize);
    if (lane == 0) total = v;
  }
  __syncthreads();
  return total;
}

// Two sums over the block in one pass: AttnRes needs a source's sum of squares and its
// weighted dot together, and one pass is half the barriers of two `block_sum`s.
DINLINE float2 block_sum2(float a, float b) {
  __shared__ float2 partial[kMaxWaves];
  __shared__ float2 total;
  const int lane = threadIdx.x % kWaveSize;
  const int warp = threadIdx.x / kWaveSize;
  for (int off = kWaveSize / 2; off > 0; off >>= 1) {
    a += __shfl_down(a, off, kWaveSize);
    b += __shfl_down(b, off, kWaveSize);
  }
  if (lane == 0) partial[warp] = make_float2(a, b);
  __syncthreads();
  const int warps = (blockDim.x + kWaveSize - 1) / kWaveSize;
  if (warp == 0) {
    float2 v = (lane < warps) ? partial[lane] : make_float2(0.0f, 0.0f);
    for (int off = kWaveSize / 2; off > 0; off >>= 1) {
      v.x += __shfl_down(v.x, off, kWaveSize);
      v.y += __shfl_down(v.y, off, kWaveSize);
    }
    if (lane == 0) total = v;
  }
  __syncthreads();
  return total;
}

// N sums over the block in one pass, `v` in place: every wave reduces its N, then every thread adds
// the waves' partials (at most kMaxWaves) itself. AttnRes reduces all its sources' sums at once
// with it, one pass where a block_sum2 per source was one round each.
template <int N>
DINLINE void block_sum_n(float (&v)[N]) {
  __shared__ float partial[kMaxWaves][N];
  const int lane = threadIdx.x % kWaveSize;
  const int warp = threadIdx.x / kWaveSize;
#pragma unroll
  for (int n = 0; n < N; ++n)
    for (int off = kWaveSize / 2; off > 0; off >>= 1) v[n] += __shfl_down(v[n], off, kWaveSize);
  if (lane == 0) {
#pragma unroll
    for (int n = 0; n < N; ++n) partial[warp][n] = v[n];
  }
  __syncthreads();
  const int warps = (blockDim.x + kWaveSize - 1) / kWaveSize;
#pragma unroll
  for (int n = 0; n < N; ++n) {
    float t = 0.0f;
    for (int w = 0; w < warps; ++w) t += partial[w][n];
    v[n] = t;
  }
  // Before a later call reuses `partial`.
  __syncthreads();
}

}  // namespace hip_comms
