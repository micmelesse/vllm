// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE HELPERS every op and kernel uses: the pack (16 bytes, the unit everything loads, sums and
// stores in) and the Fragment (a thread's packs of a row a block owns).

#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#define DINLINE __device__ __forceinline__

namespace hip_comms {

template <typename T, int N>
struct __align__(sizeof(T) * N) vec {
  T d[N];
};

// A PACK: 16 bytes, one vector load (8 bf16), the unit every kernel loads, sums and stores in,
// as in vLLM's and aiter's custom all-reduce and NCCL. `num_packs` counts them.
constexpr int kPackBytes = 16;

template <typename T>
struct traits {
  static constexpr int N = kPackBytes / sizeof(T);
  using V = vec<T, N>;
};

// WHO HOLDS WHAT: a Fragment (CUTLASS's, CuTe's and rocWMMA's name for a thread's registers of a
// distributed tile) is this thread's packs of a row a block owns. Its indices are clamped into the
// row and a pack past its end is weighted zero (`in`), so nothing reading it branches: a load under
// a runtime `if` cannot be hoisted past the branch, and loads meant to be in flight together then
// wait one at a time.
//
// Packs threadIdx.x + k * blockDim.x, k < K, of a row `len` packs long: `at` clamped into the row,
// `in` 1 for a pack inside it and 0 past its end.
template <int K>
struct Fragment {
  int at[K];
  float in[K];
};

template <int K>
DINLINE Fragment<K> fragment(int len) {
  Fragment<K> f;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    f.at[k]     = i < len ? i : len - 1;
    f.in[k]     = i < len ? 1.0f : 0.0f;
  }
  return f;
}

}  // namespace hip_comms
