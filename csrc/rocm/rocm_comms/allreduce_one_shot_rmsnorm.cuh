// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot all-reduce fused with RMSNorm.

#pragma once

#include "ipc.cuh"
#include "utils.cuh"

namespace hip_comms {

// ONE-SHOT, FUSED WITH RMSNorm. The sum is already in registers when one-shot is about to
// store it, so normalising there costs one HBM round trip less than an all-reduce kernel
// followed by a norm kernel, and one launch less.
//
// THE DECOMPOSITION IS THE WHOLE COST, and it is why this is a separate kernel rather than a
// flag on the one above. RMSNorm needs the sum of squares across a WHOLE ROW, so a block has
// to own a row; plain one-shot is grid-stride over the flat buffer and a block owns whatever
// it lands on. So this is one block per row, striding over rows, and the reduction inside a
// block is what the norm needs. (aiter reaches the same constraint from the other side and
// spells it `hidden_dim / pack_size <= 1024` -- their gate for the 1-stage fused path.)
//
// TWO PASSES OVER THE ROW, NOT ONE, and the second reads what the first wrote. Holding the
// row in registers would avoid it, but only up to a hidden size the register file allows,
// and the bound would then be a silent wrong answer rather than a refusal. `residual_out` has
// to be written anyway, so the re-read is of a 16 KB row this block wrote microseconds ago --
// L2, not HBM. If profiling says otherwise, a register-resident variant is the next step.
//
// THE VARIANCE IS AN fp32 SUM OF fp32 SQUARES, matching `vllm.ir.ops.fused_add_rms_norm`.
// What differs from it: the second pass reads back the ROUNDED residual, where the reference
// scales the unrounded fp32 value. One rounding, inside the norm's own tolerance, and the
// equivalence test is what says so rather than this comment.
DINLINE float block_sum(float v) {
  // 8 = 512 / 64, the launch bound over the wavefront. A block wider than the bound cannot
  // be launched, so this cannot be overrun.
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

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) allreduce_one_shot_rmsnorm(
    ipc::Peers p, T* __restrict__ out, T* __restrict__ residual_out,
    const T* __restrict__ residual, const T* __restrict__ weight, float eps, int rows,
    int packs) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  const int rank   = p.rank();
  const V* ptrs[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; ++i)
    ptrs[i] = p.input<V>((rank + i) % ngpus);

  p.barrier_start<ngpus>();

  const V* res_in = reinterpret_cast<const V*>(residual);
  V* res_out      = reinterpret_cast<V*>(residual_out);
  const V* w      = reinterpret_cast<const V*>(weight);
  V* o            = reinterpret_cast<V*>(out);
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);

  // A BLOCK OWNS A ROW AT A TIME. Uniform across the block, so every `__syncthreads`
  // below is reached by every thread in it.
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int base = row * packs;
    float acc      = 0.0f;
    for (int i = threadIdx.x; i < packs; i += blockDim.x) {
      V sum = reduce_at<T, ngpus>(ptrs, base + i);
      V r   = res_in[base + i];
#pragma unroll
      for (int j = 0; j < NL; ++j) {
        const float s = static_cast<float>(sum.d[j]) + static_cast<float>(r.d[j]);
        sum.d[j]      = static_cast<T>(s);
        acc += s * s;
      }
      res_out[base + i] = sum;
    }
    const float scale = rsqrtf(block_sum(acc) * inv_hidden + eps);
    for (int i = threadIdx.x; i < packs; i += blockDim.x) {
      V r  = res_out[base + i];
      V wv = w[i];
#pragma unroll
      for (int j = 0; j < NL; ++j)
        r.d[j] = static_cast<T>(static_cast<float>(r.d[j]) * scale *
                                static_cast<float>(wv.d[j]));
      o[base + i] = r;
    }
    // Before the next row reuses `block_sum`'s shared slots.
    __syncthreads();
  }
  // Same reason as one-shot's: a rank that returns lets its INPUT be reused while a peer
  // is still reading it.
  p.barrier_end<ngpus, true>();
}

}  // namespace hip_comms
