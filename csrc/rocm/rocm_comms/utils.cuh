// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Shared by the collectives: the 16-byte vector, its sum across ranks, and one row
// of the fused residual add + RMSNorm.

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

// ONE ROW of all-reduce + residual add + RMSNorm, by the whole block. `row` indexes the
// inputs and `residual`; the two outputs are written at `res_dst` and `out_dst`, which
// is where the variants differ (the output tensors, or scratch).
//
// TWO PASSES OVER THE ROW: the second reads back the residual the first wrote -- a row this
// block wrote microseconds ago, so L2 rather than HBM -- instead of holding the row in
// registers, which would bound the hidden size silently.
//
// THE VARIANCE IS AN fp32 SUM OF fp32 SQUARES, matching `vllm.ir.ops.fused_add_rms_norm`.
// What differs: the second pass reads the ROUNDED residual where the reference scales the
// unrounded fp32 value. One rounding, inside the norm's own tolerance.
template <typename T, int ngpus>
DINLINE void add_rmsnorm_row(const typename traits<T>::V* const ptrs[],
                             const typename traits<T>::V* residual,
                             const typename traits<T>::V* weight, int row, int packs,
                             float inv_hidden, float eps,
                             typename traits<T>::V* res_dst,
                             typename traits<T>::V* out_dst) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  const int base   = row * packs;
  float acc        = 0.0f;
  for (int i = threadIdx.x; i < packs; i += blockDim.x) {
    V sum = reduce_at<T, ngpus>(ptrs, base + i);
    V r   = residual[base + i];
#pragma unroll
    for (int j = 0; j < NL; ++j) {
      const float s = static_cast<float>(sum.d[j]) + static_cast<float>(r.d[j]);
      sum.d[j]      = static_cast<T>(s);
      acc += s * s;
    }
    res_dst[i] = sum;
  }
  const float scale = rsqrtf(block_sum(acc) * inv_hidden + eps);
  for (int i = threadIdx.x; i < packs; i += blockDim.x) {
    V r  = res_dst[i];
    V wv = weight[i];
#pragma unroll
    for (int j = 0; j < NL; ++j)
      r.d[j] = static_cast<T>(static_cast<float>(r.d[j]) * scale *
                              static_cast<float>(wv.d[j]));
    out_dst[i] = r;
  }
  // Before the next row reuses `block_sum`'s shared slots.
  __syncthreads();
}

}  // namespace hip_comms
