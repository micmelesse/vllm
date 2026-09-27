// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Shared by the collectives: the 16-byte vector, block sums, the rows of all-reduce +
// RMSNorm and + AttnRes, and the latent MoE tail's GEMM phase.

#pragma once

#include <hip/hip_runtime.h>

#include <cmath>

#define DINLINE __device__ __forceinline__

namespace hip_comms {

// ---------------------------------------------------------------------------------
// The 16-byte vector every kernel moves.
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

// ONE ROW of all-reduce, then (kAdd) add, then RMSNorm by the whole block, matching vLLM's
// reference ops (`vllm/ir/ops/layernorm.py`) rounding for rounding:
//
//   s   = float(T(sum over ranks))                  the all-reduce output, as it would land
//   s  += float(residual); res_dst = T(s)           kAdd only: fused_add_rms_norm
//   x   = W(s * rsqrt(mean(s^2) + eps))             `x.to(weight.dtype)`
//   out = T(W(x * float(w)))                        the product in W, then to the input's T
//
// THE WEIGHT KEEPS ITS OWN DTYPE W, as the reference does: it rounds to the WEIGHT's dtype,
// so an fp32 weight multiplies an unrounded x, and casting it to T first would be a
// different op. With W == T this is the one rounding it always was.
//
// The variance is taken from `s` before any further rounding, and `s` stays in registers
// between the two passes, so nothing is read back.
//
// `c` is the kernel's `ipc::Comm` (the sum comes from its `sum`); the results leave
// through `store_res(i, v)` and `store_out(i, v)`, i the pack within the row, so a kernel
// can land them in its output or in scratch through `put`.
template <typename T, typename W, bool kAdd, typename C, typename StoreRes,
          typename StoreOut>
DINLINE void add_rms_norm_row(const C& c, const typename traits<T>::V* residual,
                              const vec<W, traits<T>::N>* weight, int row, int packs,
                              float inv_hidden, float eps, StoreRes store_res,
                              StoreOut store_out) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  const int base   = row * packs;
  float s[kMaxRowPacks][NL];
  float acc = 0.0f;
#pragma unroll
  for (int k = 0; k < kMaxRowPacks; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    if (i >= packs) break;
    const V sum = c.sum(base + i);
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
      store_res(i, rounded);
    }
#pragma unroll
    for (int j = 0; j < NL; ++j) acc += s[k][j] * s[k][j];
  }
  const float scale = rsqrtf(block_sum(acc) * inv_hidden + eps);
#pragma unroll
  for (int k = 0; k < kMaxRowPacks; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    if (i >= packs) break;
    const vec<W, NL> w = weight[i];
    V o;
#pragma unroll
    for (int j = 0; j < NL; ++j) {
      const float x  = static_cast<float>(static_cast<W>(s[k][j] * scale));
      const float xw = static_cast<float>(static_cast<W>(x * static_cast<float>(w.d[j])));
      o.d[j]         = static_cast<T>(xw);
    }
    store_out(i, o);
  }
  // Before the next row reuses `block_sum`'s shared slots.
  __syncthreads();
}

// ---------------------------------------------------------------------------------
// One row of all-reduce + Kimi-K3's AttnRes + its RMSNorm (the one-shot and two-shot
// AttnRes kernels).
// ---------------------------------------------------------------------------------

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
// the rounding of the last few bits, not bitwise. `c` is the kernel's `ipc::Comm`; the
// new prefix leaves through `store_prefix(i, v)` and the output through `store_out(i,
// v)`, i the pack within the row, so a kernel lands them in place or in scratch.
template <typename T, bool kPrefix, typename C, typename StorePrefix, typename StoreOut>
DINLINE void add_attn_res_rms_norm_row(const C& c,
                          const typename traits<T>::V* prefix, const T* blocks,
                          int64_t block_stride_r,
                          const typename traits<T>::V* norm_w,
                          const typename traits<T>::V* qk_w,
                          const typename traits<T>::V* out_norm_w, int num_blocks, int row,
                          int packs, float inv_hidden, float eps, float out_eps,
                          StorePrefix store_prefix, StoreOut store_out) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  const int base   = row * packs;
  float u[kMaxRowPacks][NL];
#pragma unroll
  for (int k = 0; k < kMaxRowPacks; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    if (i >= packs) break;
    const V sum = c.sum(base + i);
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
    store_prefix(i, rounded);
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
    store_out(i, o);
  }
  // Before the next row reuses the reductions' shared slots.
  __syncthreads();
}

// ---------------------------------------------------------------------------------
// The GEMM phase of the latent MoE tail (the one-shot and two-shot GEMM-tail kernels).
// ---------------------------------------------------------------------------------

// The most rows one pass takes: each wave holds kGemmRows x kGemmCols accumulators.
constexpr int kGemmRows = 16;
constexpr int kGemmCols = 4;

// out[r, col0 + n] = T(float(out[r, col0 + n]) + sum_k x[r][k] * w[n][k]) for r < rows,
// rows <= kGemmRows, the sum in fp32 and rounded once. `load(r, k)` returns pack k of row
// r of x, wherever it lives. Each wave takes kGemmCols columns for every row: x is loaded
// once per pack and used for all of them, the weight rows once each. The sum runs in a
// different order than hipBLASLt's, so a result agrees to the rounding of the last bits,
// not bitwise.
template <typename T, typename Load>
DINLINE void gemm_add_rows(Load load, int rows, const T* __restrict__ gemm_w, int n_cols,
                           int packs, T* __restrict__ out, int64_t out_stride,
                           int out_col0) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  const int lane   = threadIdx.x % warpSize;
  const int waves  = gridDim.x * (blockDim.x / warpSize);
  const int wave   = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  const V* wv      = reinterpret_cast<const V*>(gemm_w);
  for (int c0 = wave * kGemmCols; c0 < n_cols; c0 += waves * kGemmCols) {
    float acc[kGemmRows][kGemmCols];
#pragma unroll
    for (int r = 0; r < kGemmRows; ++r)
#pragma unroll
      for (int q = 0; q < kGemmCols; ++q) acc[r][q] = 0.0f;
    // EVERY LOOP OVER r AND q HAS A CONSTANT TRIP COUNT, predicated on rows and n_cols
    // rather than broken out of: a runtime bound stops the unroll, `acc[r]` becomes a
    // runtime index, and the accumulators move to scratch memory (536 bytes/lane, 10x
    // slower, 2026-09-27).
    for (int k = lane; k < packs; k += warpSize) {
      float w[kGemmCols][NL];
#pragma unroll
      for (int q = 0; q < kGemmCols; ++q) {
        V x{};
        if (c0 + q < n_cols) x = wv[(c0 + q) * packs + k];
#pragma unroll
        for (int j = 0; j < NL; ++j) w[q][j] = static_cast<float>(x.d[j]);
      }
#pragma unroll
      for (int r = 0; r < kGemmRows; ++r) {
        V x{};
        if (r < rows) x = load(r, k);
        float a[NL];
#pragma unroll
        for (int j = 0; j < NL; ++j) a[j] = static_cast<float>(x.d[j]);
#pragma unroll
        for (int q = 0; q < kGemmCols; ++q)
#pragma unroll
          for (int j = 0; j < NL; ++j) acc[r][q] += a[j] * w[q][j];
      }
    }
    // The wave's partial sums, reduced so every lane holds every total; lane (r, q) writes.
#pragma unroll
    for (int r = 0; r < kGemmRows; ++r) {
#pragma unroll
      for (int q = 0; q < kGemmCols; ++q) {
        float v = acc[r][q];
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, 64);
        if (r < rows && lane == r * kGemmCols + q && c0 + q < n_cols) {
          T* at = out + r * out_stride + out_col0 + c0 + q;
          *at   = static_cast<T>(static_cast<float>(*at) + v);
        }
      }
    }
  }
}

}  // namespace hip_comms
