// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE PACK: what every kernel, fusion and p2p phase loads, sums and stores.

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
template <typename T>
struct traits {
  static constexpr int N = 16 / sizeof(T);
  using V = vec<T, N>;
};

}  // namespace hip_comms
