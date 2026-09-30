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
#include "row.cuh"

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

// THIS THREAD'S PACKS OF ROW `row` (row.cuh's Share) summed over the `ngpus` sources, into sum[k]. A
// block owns the row; its norm needs every pack. Every pack's loads go out together: a pack past
// the row sums the last one again (weighted zero where it is used), where an `if (i < packs)` made
// each pack's loads wait on the one before.
template <typename T, int ngpus, int K, typename Read>
DINLINE void sum_row(Read read, int row, int packs, const Share<K>& sh,
                     typename traits<T>::V (&sum)[K]) {
#pragma unroll
  for (int k = 0; k < K; ++k) sum[k] = sum_packs<T, ngpus>(read, int64_t{row} * packs + sh.at[k]);
}

// N SUMS OVER THE BLOCK IN ONE PASS, `v` in place: every wave reduces its N, then every thread adds
// the waves' partials (at most kMaxWaves) itself. The one block reduction: a kernel with several
// sums to take takes them together, since each pass is two barriers and an LDS round trip.
template <int N>
DINLINE void block_sum(float (&v)[N]) {
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

// The LDS a block_sum<N> takes, for a kernel budgeting the rest.
constexpr int64_t block_sum_lds_bytes(int n) { return int64_t{kMaxWaves} * n * sizeof(float); }

}  // namespace hip_comms
