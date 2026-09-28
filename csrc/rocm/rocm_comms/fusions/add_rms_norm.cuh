// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE COMPUTATION of all-reduce + RMSNorm and all-reduce + add + RMSNorm, on a row already
// reduced over ranks: every rms_norm and add_rms_norm kernel, and the GEMM tail's norm,
// pull or push.

#pragma once

#include "../utils.cuh"

namespace hip_comms::fusions::add_rms_norm {

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
// DIRECTION-FREE: `sum` is this thread's share of the row already reduced over ranks and
// rounded to T (by `p2p::pull::reduce` or `p2p::push::reduce`), sum[k] the
// pack threadIdx.x + k * blockDim.x. The results leave through `store_res(k, i, v)` and
// `store_out(k, i, v)`, i the pack within the row, so a kernel can land them in its
// output, in scratch, or in registers to push.
template <typename T, typename W, bool kAdd, typename StoreRes, typename StoreOut>
DINLINE void row(const typename traits<T>::V (&sum)[kMaxRowPacks],
                 const typename traits<T>::V* residual, const vec<W, traits<T>::N>* weight,
                 int row, int packs, float inv_hidden, float eps, StoreRes store_res,
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
#pragma unroll
    for (int j = 0; j < NL; ++j) s[k][j] = static_cast<float>(sum[k].d[j]);
    if constexpr (kAdd) {
      const V r = residual[base + i];
      V rounded;
#pragma unroll
      for (int j = 0; j < NL; ++j) {
        s[k][j] += static_cast<float>(r.d[j]);
        rounded.d[j] = static_cast<T>(s[k][j]);
      }
      store_res(k, i, rounded);
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
    store_out(k, i, o);
  }
  // Before the next row reuses `block_sum`'s shared slots.
  __syncthreads();
}

}  // namespace hip_comms::fusions::add_rms_norm
