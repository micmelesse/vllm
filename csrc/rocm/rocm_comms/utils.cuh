// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Shared by the collectives: the 16-byte vector, its sum across ranks, and one row
// of all-reduce + RMSNorm.

#pragma once

#include <hip/hip_runtime.h>

#define DINLINE __device__ __forceinline__

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

// Sum of `v` over the block. 8 = 512 / 64, the launch bound over the wavefront: a block
// wider than the bound cannot be launched, so `partial` cannot be overrun.
DINLINE float block_sum(float v) {
  __shared__ float partial[8];
  __shared__ float total;
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  for (int off = warpSize / 2; off > 0; off >>= 1) v += __shfl_down(v, off, warpSize);
  if (lane == 0) partial[warp] = v;
  __syncthreads();
  const int warps = (blockDim.x + warpSize - 1) / warpSize;
  if (warp == 0) {
    v = (lane < warps) ? partial[lane] : 0.0f;
    for (int off = warpSize / 2; off > 0; off >>= 1) v += __shfl_down(v, off, warpSize);
    if (lane == 0) total = v;
  }
  __syncthreads();
  return total;
}

// How many 16-byte packs of one row a thread holds in registers. A row wider than
// kMaxRowPacks x blockDim is refused by the host.
constexpr int kMaxRowPacks = 4;

// ONE ROW of all-reduce + RMSNorm by the whole block, matching vLLM's reference ops
// (`vllm/ir/ops/layernorm.py`) rounding for rounding:
//
//   s   = float(T(sum over ranks))                  the all-reduce output, as it would land
//   s  += float(residual); res_dst = T(s)           kAdd only: fused_add_rms_norm
//   out = T(float(T(s * rsqrt(mean(s^2) + eps))) * float(w))
//
// The variance is taken from `s` before any further rounding, and `s` stays in registers
// between the two passes, so nothing is read back.
template <typename T, int ngpus, bool kAdd>
DINLINE void rms_norm_row(const typename traits<T>::V* const ptrs[],
                          const typename traits<T>::V* residual,
                          const typename traits<T>::V* weight, int row, int packs,
                          float inv_hidden, float eps, typename traits<T>::V* res_dst,
                          typename traits<T>::V* out_dst) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  const int base   = row * packs;
  float s[kMaxRowPacks][NL];
  float acc = 0.0f;
#pragma unroll
  for (int k = 0; k < kMaxRowPacks; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    if (i >= packs) break;
    const V sum = reduce_at<T, ngpus>(ptrs, base + i);
#pragma unroll
    for (int j = 0; j < NL; ++j) s[k][j] = static_cast<float>(sum.d[j]);
    if constexpr (kAdd) {
      const V r = residual[base + i];
      V rounded;
#pragma unroll
      for (int j = 0; j < NL; ++j) {
        s[k][j] += static_cast<float>(r.d[j]);
        rounded.d[j] = static_cast<T>(s[k][j]);
      }
      res_dst[i] = rounded;
    }
#pragma unroll
    for (int j = 0; j < NL; ++j) acc += s[k][j] * s[k][j];
  }
  const float scale = rsqrtf(block_sum(acc) * inv_hidden + eps);
#pragma unroll
  for (int k = 0; k < kMaxRowPacks; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    if (i >= packs) break;
    const V w = weight[i];
    V o;
#pragma unroll
    for (int j = 0; j < NL; ++j)
      o.d[j] = static_cast<T>(static_cast<float>(static_cast<T>(s[k][j] * scale)) *
                              static_cast<float>(w.d[j]));
    out_dst[i] = o;
  }
  // Before the next row reuses `block_sum`'s shared slots.
  __syncthreads();
}

}  // namespace hip_comms
