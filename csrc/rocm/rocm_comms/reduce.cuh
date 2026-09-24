// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Summing one 16-byte packet across ranks.

#pragma once

#include "peer.cuh"

namespace hip_comms {

// ---------------------------------------------------------------------------------
// Vectorized reduce. 16 bytes per thread, accumulated in fp32 so a bf16 sum of 8 values
// rounds once at the end rather than eight times along the way.
// ---------------------------------------------------------------------------------

template <typename T, int N>
struct __align__(sizeof(T) * N) vec {
  T d[N];
};

template <typename T>
struct traits {
  static constexpr int N = 16 / sizeof(T);
  using V = vec<T, N>;
};

template <typename T, int ngpus>
DINLINE typename traits<T>::V reduce_at(const typename traits<T>::V* const ptrs[],
                                        int idx) {
  constexpr int N = traits<T>::N;
  float acc[N];
  auto v0 = ptrs[0][idx];
#pragma unroll
  for (int j = 0; j < N; ++j) acc[j] = static_cast<float>(v0.d[j]);
#pragma unroll
  for (int i = 1; i < ngpus; ++i) {
    auto v = ptrs[i][idx];
#pragma unroll
    for (int j = 0; j < N; ++j) acc[j] += static_cast<float>(v.d[j]);
  }
  typename traits<T>::V out;
#pragma unroll
  for (int j = 0; j < N; ++j) out.d[j] = static_cast<T>(acc[j]);
  return out;
}

}  // namespace hip_comms
