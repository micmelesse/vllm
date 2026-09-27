// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot all-reduce fused with Kimi-K3's attention residual (AttnRes).

#pragma once

#include "ipc.cuh"
#include "utils.cuh"

namespace hip_comms {

// Two sums over the block in one pass: AttnRes needs a source's sum of squares and its
// weighted dot together, and one pass is half the barriers of two `block_sum`s.
DINLINE float2 block_sum2(float a, float b) {
  __shared__ float2 partial[8];
  __shared__ float2 total;
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    a += __shfl_down(a, off, warpSize);
    b += __shfl_down(b, off, warpSize);
  }
  if (lane == 0) partial[warp] = make_float2(a, b);
  __syncthreads();
  const int warps = (blockDim.x + warpSize - 1) / warpSize;
  if (warp == 0) {
    float2 v = (lane < warps) ? partial[lane] : make_float2(0.0f, 0.0f);
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      v.x += __shfl_down(v.x, off, warpSize);
      v.y += __shfl_down(v.y, off, warpSize);
    }
    if (lane == 0) total = v;
  }
  __syncthreads();
  return total;
}

// ONE ROW of all-reduce + AttnRes by the whole block, matching
// `vllm/models/kimi_k3/amd/ops/attn_res.py` rounding for rounding:
//
//   d   = float(T(sum over ranks))                   the all-reduce output, as it lands
//   u   = kPrefix ? float(T(float(prefix) + d)) : d  the running prefix, updated or started
//   prefix_out = T(u); blocks[write] = T(u)          the new prefix, and the block written
//   per source s (the stored blocks, then u):
//       logit_s = dot(s, norm_w * qk_w) * rsqrt(mean(s^2) + eps)
//   m   = softmax(logit) . sources                     online, one source at a time
//   out = T(m), or T(m * rsqrt(mean(m^2) + out_eps) * out_norm_w)
//
// The prefix and the mix stay in registers across the sources, so only the stored blocks
// are read back. The sums run in a different order than Triton's, so a result agrees to
// the rounding of the last few bits, not bitwise.
template <typename T, int ngpus, bool kPrefix>
DINLINE void add_attn_res_rms_norm_row(const typename traits<T>::V* const ptrs[],
                          typename traits<T>::V* prefix, const T* blocks,
                          int64_t block_stride_r, T* block_dst,
                          const typename traits<T>::V* norm_w,
                          const typename traits<T>::V* qk_w,
                          const typename traits<T>::V* out_norm_w, int num_blocks, int row,
                          int packs, float inv_hidden, float eps, float out_eps,
                          typename traits<T>::V* out) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  const int base   = row * packs;
  float u[kMaxRowPacks][NL];
#pragma unroll
  for (int k = 0; k < kMaxRowPacks; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    if (i >= packs) break;
    const V sum = reduce_at<T, ngpus>(ptrs, base + i);
    V rounded;
    if constexpr (kPrefix) {
      const V p = prefix[base + i];
#pragma unroll
      for (int j = 0; j < NL; ++j) {
        rounded.d[j] = static_cast<T>(static_cast<float>(p.d[j]) +
                                      static_cast<float>(sum.d[j]));
        u[k][j]      = static_cast<float>(rounded.d[j]);
      }
    } else {
      rounded = sum;
#pragma unroll
      for (int j = 0; j < NL; ++j) u[k][j] = static_cast<float>(sum.d[j]);
    }
    prefix[base + i] = rounded;
    if (block_dst != nullptr) reinterpret_cast<V*>(block_dst)[i] = rounded;
  }

  float m[kMaxRowPacks][NL];
  if (num_blocks == 0) {
    // With only the prefix source, the softmax is exactly one.
#pragma unroll
    for (int k = 0; k < kMaxRowPacks; ++k)
#pragma unroll
      for (int j = 0; j < NL; ++j) m[k][j] = u[k][j];
  } else {
    float w[kMaxRowPacks][NL];
#pragma unroll
    for (int k = 0; k < kMaxRowPacks; ++k) {
      const int i = threadIdx.x + k * blockDim.x;
#pragma unroll
      for (int j = 0; j < NL; ++j) m[k][j] = 0.0f;
      if (i >= packs) continue;
      const V a = norm_w[i], b = qk_w[i];
#pragma unroll
      for (int j = 0; j < NL; ++j)
        w[k][j] = static_cast<float>(a.d[j]) * static_cast<float>(b.d[j]);
    }
    float max_logit = -INFINITY, denominator = 0.0f;
    for (int s = 0; s <= num_blocks; ++s) {
      // The stored blocks first, the prefix last, as the reference orders its sources.
      const V* src = reinterpret_cast<const V*>(blocks + s * block_stride_r);
      float v[kMaxRowPacks][NL];
      float ss = 0.0f, dot = 0.0f;
#pragma unroll
      for (int k = 0; k < kMaxRowPacks; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i >= packs) break;
        if (s < num_blocks) {
          const V x = src[i];
#pragma unroll
          for (int j = 0; j < NL; ++j) v[k][j] = static_cast<float>(x.d[j]);
        } else {
#pragma unroll
          for (int j = 0; j < NL; ++j) v[k][j] = u[k][j];
        }
#pragma unroll
        for (int j = 0; j < NL; ++j) {
          ss += v[k][j] * v[k][j];
          dot += v[k][j] * w[k][j];
        }
      }
      const float2 sums       = block_sum2(ss, dot);
      const float logit       = sums.y * rsqrtf(sums.x * inv_hidden + eps);
      const float new_max     = fmaxf(max_logit, logit);
      const float old_scale   = __expf(max_logit - new_max);
      const float this_scale  = __expf(logit - new_max);
      denominator             = denominator * old_scale + this_scale;
      max_logit               = new_max;
#pragma unroll
      for (int k = 0; k < kMaxRowPacks; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i >= packs) break;
#pragma unroll
        for (int j = 0; j < NL; ++j) m[k][j] = m[k][j] * old_scale + this_scale * v[k][j];
      }
    }
    const float inv_den = 1.0f / denominator;
#pragma unroll
    for (int k = 0; k < kMaxRowPacks; ++k)
#pragma unroll
      for (int j = 0; j < NL; ++j) m[k][j] *= inv_den;
  }

  float scale = 1.0f;
  if (out_norm_w != nullptr) {
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < kMaxRowPacks; ++k) {
      const int i = threadIdx.x + k * blockDim.x;
      if (i >= packs) break;
#pragma unroll
      for (int j = 0; j < NL; ++j) ss += m[k][j] * m[k][j];
    }
    scale = rsqrtf(block_sum2(ss, 0.0f).x * inv_hidden + out_eps);
  }
#pragma unroll
  for (int k = 0; k < kMaxRowPacks; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    if (i >= packs) break;
    V o;
    if (out_norm_w != nullptr) {
      const V g = out_norm_w[i];
#pragma unroll
      for (int j = 0; j < NL; ++j)
        o.d[j] = static_cast<T>(m[k][j] * scale * static_cast<float>(g.d[j]));
    } else {
#pragma unroll
      for (int j = 0; j < NL; ++j) o.d[j] = static_cast<T>(m[k][j]);
    }
    out[base + i] = o;
  }
  // Before the next row reuses the reductions' shared slots.
  __syncthreads();
}

// A BLOCK OWNS A ROW, striding over rows, as the fused norm does: every rank reduces every
// row, so there is nothing to gather. `blocks` is [rows, num_sources, hidden] with row and
// source strides in elements; `write_idx` < 0 writes no block.
template <typename T, int ngpus, bool kPrefix>
__global__ void __launch_bounds__(512, 1) allreduce_one_shot_add_attn_res_rms_norm(
    ipc::Peers p, T* __restrict__ prefix, T* __restrict__ blocks, int64_t block_stride_m,
    int64_t block_stride_r, const T* __restrict__ norm_w, const T* __restrict__ qk_w,
    const T* __restrict__ out_norm_w, T* __restrict__ out, int num_blocks, int write_idx,
    float eps, float out_eps, int rows, int packs) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  const int rank   = p.rank();
  const V* ptrs[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; ++i)
    ptrs[i] = p.input<V>((rank + i) % ngpus);

  p.barrier_start<ngpus>();

  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const T* row_blocks = blocks + row * block_stride_m;
    T* dst = write_idx >= 0 ? blocks + row * block_stride_m + write_idx * block_stride_r
                            : nullptr;
    add_attn_res_rms_norm_row<T, ngpus, kPrefix>(
        ptrs, reinterpret_cast<V*>(prefix), row_blocks, block_stride_r, dst,
        reinterpret_cast<const V*>(norm_w), reinterpret_cast<const V*>(qk_w),
        reinterpret_cast<const V*>(out_norm_w), num_blocks, row, packs, inv_hidden, eps,
        out_eps, reinterpret_cast<V*>(out));
  }

  // A rank that returns lets its INPUT be reused while a peer is still reading it.
  p.barrier_end<ngpus, true>();
}

}  // namespace hip_comms
