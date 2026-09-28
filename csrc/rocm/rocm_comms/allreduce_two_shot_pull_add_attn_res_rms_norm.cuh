// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot all-reduce fused with Kimi-K3's attention residual (AttnRes) and its RMSNorm.

#pragma once

#include "fusions/add_attn_res_rms_norm.cuh"
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
__global__ void __launch_bounds__(kMaxThreads, 1)
    allreduce_two_shot_pull_add_attn_res_rms_norm(
    ipc::Peers p, T* __restrict__ prefix, T* __restrict__ blocks, int64_t block_stride_m,
    int64_t block_stride_r, const T* __restrict__ norm_w, const T* __restrict__ qk_w,
    const T* __restrict__ out_norm_w, T* __restrict__ out, int num_blocks, int write_idx,
    float eps, float out_eps, int rows, int packs) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  namespace fusion = fusions::add_attn_res_rms_norm;
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
      V sum[kMaxRowPacks];
      c.sum_row(row * packs, packs, sum);
      fusion::row<T, kPrefix>(
          sum, reinterpret_cast<const V*>(prefix), blocks + row * block_stride_m,
          block_stride_r, reinterpret_cast<const V*>(norm_w),
          reinterpret_cast<const V*>(qk_w), reinterpret_cast<const V*>(out_norm_w),
          num_blocks, row, packs, inv_hidden, eps, out_eps,
          [&](int, int i, const V& v) { c.put(rank, half + local + i, v); },
          [&](int, int i, const V& v) { c.put(rank, local + i, v); });
    }
  }

  // The gather below gives each block the local rows it wrote above, so the same-numbered
  // blocks are all it must wait for; the input is read only above, so no close.
  c.peer_block_barrier();

  V* o             = reinterpret_cast<V*>(out);
  V* pre           = reinterpret_cast<V*>(prefix);
  c.template gather_rows<2>(
      chunk, rows, packs, half, [&](int region, int row, int k, const V& v) {
        if (region == 0) {
          store_global(o + row * packs + k, v);
          return;
        }
        store_global(pre + row * packs + k, v);
        if (write_idx >= 0)
          store_global(reinterpret_cast<V*>(blocks + row * block_stride_m +
                                            write_idx * block_stride_r) + k, v);
      });
}

}  // namespace hip_comms
