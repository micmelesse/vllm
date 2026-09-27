// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot all-reduce fused with Kimi-K3's attention residual (AttnRes) and its RMSNorm.

#pragma once

#include "ipc.cuh"
#include "utils.cuh"

namespace hip_comms {

// Each rank owns ceil(rows/ngpus) WHOLE rows, as the two-shot norm does:
//
//   phase 1  our rows: reduce, update the prefix, AttnRes; put [out rows | prefix rows] in
//            our scratch
//   world_barrier
//   phase 2  every rank gathers every rank's rows into `out`, `prefix` and the written
//            block
//
// Phase 1 reads the replicated `prefix` and `blocks`; phase 2 overwrites them, which the
// world_barrier between makes safe. Every output element is computed by exactly one rank.
template <typename T, int ngpus, bool kPrefix>
__global__ void __launch_bounds__(512, 1) allreduce_two_shot_add_attn_res_rms_norm(
    ipc::Peers p, T* __restrict__ prefix, T* __restrict__ blocks, int64_t block_stride_m,
    int64_t block_stride_r, const T* __restrict__ norm_w, const T* __restrict__ qk_w,
    const T* __restrict__ out_norm_w, T* __restrict__ out, int num_blocks, int write_idx,
    float eps, float out_eps, int rows, int packs) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  ipc::Comm<T, ngpus> c(p);
  const int rank  = c.rank();
  const int chunk = (rows + ngpus - 1) / ngpus;
  // Where the prefix half starts in a rank's scratch, in packs.
  const int half = chunk * packs;

  {
    const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
    const int begin        = rank * chunk;
    const int end          = min(begin + chunk, rows);
    for (int row = begin + blockIdx.x; row < end; row += gridDim.x) {
      const int local = (row - begin) * packs;
      add_attn_res_rms_norm_row<T, kPrefix>(
          c, reinterpret_cast<const V*>(prefix), blocks + row * block_stride_m,
          block_stride_r, reinterpret_cast<const V*>(norm_w),
          reinterpret_cast<const V*>(qk_w), reinterpret_cast<const V*>(out_norm_w),
          num_blocks, row, packs, inv_hidden, eps, out_eps,
          [&](int i, const V& v) { c.put(rank, half + local + i, v); },
          [&](int i, const V& v) { c.put(rank, local + i, v); });
    }
  }

  c.world_barrier();

  V* o             = reinterpret_cast<V*>(out);
  V* pre           = reinterpret_cast<V*>(prefix);
  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int i = 0; i < ngpus; ++i) {
    const int begin = i * chunk;
    const int n     = (min(begin + chunk, rows) - begin) * packs;
    for (int k = tid; k < n; k += stride) {
      const int row = begin + k / packs;
      const V u     = c.get(i, half + k);
      o[begin * packs + k]   = c.get(i, k);
      pre[begin * packs + k] = u;
      if (write_idx >= 0)
        reinterpret_cast<V*>(blocks + row * block_stride_m +
                             write_idx * block_stride_r)[k % packs] = u;
    }
  }
  c.close();
}

}  // namespace hip_comms
