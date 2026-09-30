// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce fused with Kimi-K3's attention residual (AttnRes) and its
// RMSNorm.

#pragma once

#include "fusions/add_attn_res_rms_norm.cuh"
#include "p2p/p2p.cuh"
#include "common/memory.cuh"
#include "common/reduce.cuh"

namespace hip_comms {

// Each rank owns whole rows: it reduces them, updates the prefix, runs AttnRes and leaves the
// out and prefix rows in its scratch, row-major; after the sync every rank copies every owner's
// rows into `out`, `prefix` and the written block. A row's replicated `prefix` and `blocks` are
// read and then overwritten by the one block that takes it, so the order is the block's own. THE
// SAME BLOCK AND THREAD INDEX A PACK IN BOTH PHASES: after the sync a block may read only what
// the same block on a peer wrote.
template <typename T, int ngpus, bool kPrefix, int kRowPacks>
__global__ void __launch_bounds__(kMaxThreads, 1)
    all_reduce_pull_two_shot_add_attn_res_rms_norm(
        p2p::Peers p, T* __restrict__ prefix, T* __restrict__ blocks,
        int64_t block_stride_m, int64_t block_stride_r, const T* __restrict__ norm_w,
        const T* __restrict__ qk_w, const T* __restrict__ out_norm_w, T* __restrict__ out,
        int num_blocks, int write_idx, float eps, float out_eps, int rows, int packs) {
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  namespace fusion       = fusions::add_attn_res_rms_norm;
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  V* pre                 = reinterpret_cast<V*>(prefix);
  V* o                   = reinterpret_cast<V*>(out);
  const int slice_rows   = (rows + ngpus - 1) / ngpus;
  const int64_t pre_at   = int64_t{slice_rows} * packs;  // the prefix rows, after the out rows
  // The block row `row` writes, or none.
  auto written = [&](int row) -> V* {
    return write_idx < 0 ? nullptr
                         : reinterpret_cast<V*>(blocks + row * block_stride_m +
                                                write_idx * block_stride_r);
  };

  // 1. Wait until every peer has launched, so its input is ready.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);

  // 2. This rank's rows: read each from every rank in rank order, sum, AttnRes, and leave the
  //    out and prefix rows in this rank's scratch.
  p2p::Peer<T, ngpus> all[ngpus];
#pragma unroll
  for (int r = 0; r < ngpus; ++r) all[r] = p2p::peer<T, ngpus>(p, r);
  const auto read = [&](int r, int64_t i) { return p2p::read_input(all[r], i); };
  const auto self = p2p::self<T, ngpus>(p);
  const int first = p.rank * slice_rows;
  const int last  = min(first + slice_rows, rows);
  for (int row = first + blockIdx.x; row < last; row += gridDim.x) {
    V sum[kRowPacks];
    sum_row<T, ngpus>(read, row, packs, sum);
    const int64_t at = int64_t{row - first} * packs;
    fusion::row<T, kPrefix>(
        sum, pre, blocks + row * block_stride_m, block_stride_r,
        reinterpret_cast<const V*>(norm_w), reinterpret_cast<const V*>(qk_w),
        reinterpret_cast<const V*>(out_norm_w), num_blocks, row, packs, inv_hidden, eps,
        out_eps, [&](int, int i, const V& v) { p2p::write_scratch(self, pre_at + at + i, v); },
        [&](int, int i, const V& v) { p2p::write_scratch(self, at + i, v); });
  }

  // 3. Every rank's rows are visible to its peers.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::visible>(p);

  // 4. Every owner's rows out of its scratch, into `out`, `prefix` and the written block. The
  //    next call's first sync keeps a rank from overwriting its scratch while it is read.
  for (int l = blockIdx.x; l < slice_rows; l += gridDim.x) {
    for (int i = threadIdx.x; i < packs; i += blockDim.x) {
    // EVERY OWNER'S PACK LOADED BEFORE ANY IS STORED: the compiler cannot prove the output
    // and the peers' scratch apart, so a store between two loads holds the next load back
    // until the store is done, and the eight owners' round trips run one after another.
      const int64_t at = int64_t{l} * packs + i;
      V got[ngpus] = {}, got_pre[ngpus] = {};
#pragma unroll
      for (int r = 0; r < ngpus; ++r) {
        if (r * slice_rows + l >= rows) continue;
        got[r]     = p2p::read_scratch(all[r], at);
        got_pre[r] = p2p::read_scratch(all[r], pre_at + at);
      }
#pragma unroll
      for (int r = 0; r < ngpus; ++r) {
        const int row = r * slice_rows + l;
        if (row >= rows) continue;
        store_global(o + int64_t{row} * packs + i, got[r]);
        store_global(pre + int64_t{row} * packs + i, got_pre[r]);
        if (V* dst = written(row)) store_global(dst + i, got_pre[r]);
      }
    }
  }
}

}  // namespace hip_comms
