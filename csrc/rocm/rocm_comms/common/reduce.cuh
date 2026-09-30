// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE REDUCTIONS, by scope, each built on the one below: over the PEERS (a pack or a fragment
// summed over every rank, fp32, rounded once, in rank order), over a WAVE (shuffles), over a BLOCK
// (a wave's result, then one LDS pass). N values at once at every scope: each block pass is two
// barriers and an LDS round trip, so a kernel with several reductions takes them together.

#pragma once

#include <cmath>

#include "../hardware.cuh"
#include "memory.cuh"
#include "utils.cuh"

namespace hip_comms {

// ONE PACK SUMMED OVER `ngpus` SOURCES, in fp32 and rounded once to T; `read(r, i)` is pack i
// of source r. Every read before any add, so the ngpus loads are in flight together; the
// sources in order, so callers whose sources are the ranks agree bitwise.
template <typename T, int ngpus, typename Read>
DINLINE typename traits<T>::V peers_reduce(Read read, int64_t i) {
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

// THIS THREAD'S FRAGMENT OF ROW `row` summed over the `ngpus` sources, into sum[k]. A block owns
// the row; its norm needs every pack. Every pack's loads go out together: a pack past the row sums
// the last one again (weighted zero where it is used), where an `if (i < packs)` made each pack's
// loads wait on the one before.
template <typename T, int ngpus, int K, typename Read>
DINLINE void peers_reduce(Read read, int row, int packs, const Fragment<K>& f,
                          typename traits<T>::V (&sum)[K]) {
#pragma unroll
  for (int k = 0; k < K; ++k) sum[k] = peers_reduce<T, ngpus>(read, int64_t{row} * packs + f.at[k]);
}

// THE OPERATIONS a wave or block reduction combines with.
struct Sum {
  static DINLINE float apply(float a, float b) { return a + b; }
  static constexpr float kIdentity = 0.0f;
};
struct Max {
  static DINLINE float apply(float a, float b) { return fmaxf(a, b); }
  static constexpr float kIdentity = -INFINITY;
};

// N VALUES OVER THE WAVE, in place, every lane left holding the results (butterfly shuffles).
template <typename Op, int N>
DINLINE void wave_reduce(float (&v)[N]) {
#pragma unroll
  for (int n = 0; n < N; ++n)
#pragma unroll
    for (int off = kWaveSize / 2; off > 0; off >>= 1)
      v[n] = Op::apply(v[n], __shfl_xor(v[n], off, kWaveSize));
}

// N VALUES OVER THE BLOCK, in place: each wave reduces its N, one wave combines the waves' partials
// with shuffles (not every thread reading every partial: that was N x waves LDS reads a thread),
// and the N totals are broadcast through LDS once.
template <typename Op, int N>
DINLINE void block_reduce(float (&v)[N]) {
  __shared__ float partial[kMaxWaves][N];
  __shared__ float total[N];
  const int lane  = threadIdx.x % kWaveSize;
  const int wave  = threadIdx.x / kWaveSize;
  const int waves = (blockDim.x + kWaveSize - 1) / kWaveSize;
  wave_reduce<Op>(v);
  if (lane == 0) {
#pragma unroll
    for (int n = 0; n < N; ++n) partial[wave][n] = v[n];
  }
  __syncthreads();
  if (wave == 0) {
#pragma unroll
    for (int n = 0; n < N; ++n) {
      float x = lane < waves ? partial[lane][n] : Op::kIdentity;
#pragma unroll
      for (int off = kMaxWaves / 2; off > 0; off >>= 1)
        x = Op::apply(x, __shfl_xor(x, off, kWaveSize));
      if (lane == 0) total[n] = x;
    }
  }
  __syncthreads();
#pragma unroll
  for (int n = 0; n < N; ++n) v[n] = total[n];
}

// The LDS a block_reduce<Op, N> takes, for a kernel budgeting the rest.
constexpr int64_t block_reduce_lds_bytes(int n) {
  return (int64_t{kMaxWaves} + 1) * n * sizeof(float);
}

}  // namespace hip_comms
