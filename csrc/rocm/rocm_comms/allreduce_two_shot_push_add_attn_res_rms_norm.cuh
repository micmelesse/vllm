// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot push all-reduce fused with Kimi-K3's attention residual (AttnRes) and its
// RMSNorm.

#pragma once

#include "fusions/add_attn_res_rms_norm.cuh"
#include "p2p/device.cuh"
#include "utils.cuh"

namespace hip_comms {

// Each rank owns ceil(rows/ngpus) WHOLE rows, as the pull kernel does, and every transfer
// is a store into a peer's inbox:
//
//   phase 1  each of our input rows, encoded, into its owner's inbox
//   peer_block_barrier
//   phase 2  our rows: reduced out of the inbox, prefix updated, AttnRes; the output row,
//            encoded, and the prefix row, unquantized, into every rank's inboxes
//   peer_block_barrier
//   phase 3  every owner's rows out of our inboxes into `out`, `prefix` and the written
//            block
//
// THE PREFIX IS NEVER QUANTIZED (Codec 16 whatever kBits): it is the running residual.
// Phase 2 reads the replicated `prefix` and `blocks` of a row and phase 3 overwrites them,
// both in the one block that takes that local row, so the order is the block's own.
template <typename T, int ngpus, int kBits, bool kPrefix>
__global__ void __launch_bounds__(kMaxThreads, 1)
    allreduce_two_shot_push_add_attn_res_rms_norm(
        p2p::Peers p, T* __restrict__ prefix, T* __restrict__ blocks,
        int64_t block_stride_m, int64_t block_stride_r, const T* __restrict__ norm_w,
        const T* __restrict__ qk_w, const T* __restrict__ out_norm_w, T* __restrict__ out,
        int num_blocks, int write_idx, float eps, float out_eps, int rows, int packs) {
  using V          = typename traits<T>::V;
  using C          = p2p::Codec<T, kBits>;
  using R          = p2p::Codec<T, 16>;
  constexpr int NL = traits<T>::N;
  using core       = p2p::Core<T, ngpus>;
  using push       = p2p::Push<T, ngpus, C>;
  using push_res   = p2p::Push<T, ngpus, R>;
  namespace fusion = fusions::add_attn_res_rms_norm;
  core::start(p);
  const auto in    = core::inputs(p);
  const int rank   = p.rank;
  const int chunk  = (rows + ngpus - 1) / ngpus;
  const int groups = chunk * blockDim.x;
  const p2p::Inbox<C, ngpus> box_in(groups);
  const p2p::Inbox<C, ngpus> box_out(groups, 1, box_in.end());
  const p2p::Inbox<R, ngpus> box_pre(groups, 1, box_out.end());
  push::scatter_rows(p, in, box_in, chunk, rows, packs);

  core::peer_block_barrier(p);

  {
    const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
    const int begin        = rank * chunk;
    const int end          = min(begin + chunk, rows);
    for (int row = begin + blockIdx.x; row < end; row += gridDim.x) {
      V sum[kMaxRowPacks];
      push::reduce_row(p, box_in, row - begin, packs, sum);
      V mixed[kMaxRowPacks] = {}, pre[kMaxRowPacks] = {};
      fusion::row<T, kPrefix>(
          sum, reinterpret_cast<const V*>(prefix), blocks + row * block_stride_m,
          block_stride_r, reinterpret_cast<const V*>(norm_w),
          reinterpret_cast<const V*>(qk_w), reinterpret_cast<const V*>(out_norm_w),
          num_blocks, row, packs, inv_hidden, eps, out_eps,
          [&](int k, int, const V& v) { pre[k] = v; },
          [&](int k, int, const V& v) { mixed[k] = v; });
      push::broadcast_row(p, box_out, row - begin, packs, mixed);
      push_res::broadcast_row(p, box_pre, row - begin, packs, pre);
    }
  }

  core::peer_block_barrier(p);

  V* o   = reinterpret_cast<V*>(out);
  V* pre = reinterpret_cast<V*>(prefix);
  push::gather_inbox_rows(p, box_out, chunk, rows, packs,
      [&](int row, int i, const V& v) { store_global(o + row * packs + i, v); });
  push_res::gather_inbox_rows(
      p, box_pre, chunk, rows, packs, [&](int row, int i, const V& v) {
        store_global(pre + row * packs + i, v);
        if (write_idx >= 0)
          store_global(reinterpret_cast<V*>(blocks + row * block_stride_m +
                                            write_idx * block_stride_r) + i, v);
      });
}

}  // namespace hip_comms
