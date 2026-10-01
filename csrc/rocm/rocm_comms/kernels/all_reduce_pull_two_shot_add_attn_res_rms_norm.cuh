// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce fused with Kimi-K3's attention residual (AttnRes) and its
// RMSNorm.

#pragma once

#include "../p2p/p2p.cuh"
#include "../common/common.cuh"
#include "../fusions/attn_res.cuh"

namespace hip_comms {

// THE SLICE IS COLUMNS, AND BOTH HALVES PULL, as the plain two-shot all-reduce moves its bytes:
// each rank sums its columns of every row over the ranks into its own scratch; after the sync each
// block reads its rows' columns from their owners and computes AttnRes on them itself. So the links
// carry what an all-reduce's do, AttnRes's two outputs (the prefix and out) are never gathered, and
// every rank does the AttnRes the unfused path would. A SLICE IS WHOLE WAVES (64 packs), so every
// wave's packs have one owner and p2p::peer's rank is the same across the wave. ROW q BELONGS TO
// BLOCK q % gridDim.x IN BOTH PHASES: after the sync a block may read only what the same block on
// a peer wrote. `blocks` is [rows, num_sources, hidden] with row and source strides in elements;
// `write_idx` < 0 writes no block.
template <typename T, int ngpus, bool kPrefix, int kRowPacks>
__global__ void __launch_bounds__(kMaxThreads, 1)
    all_reduce_pull_two_shot_add_attn_res_rms_norm(
        p2p::DevComm p, T* __restrict__ prefix, T* __restrict__ blocks,
        int64_t block_stride_m, int64_t block_stride_r, const T* __restrict__ norm_w,
        const T* __restrict__ qk_w, const T* __restrict__ out_norm_w, T* __restrict__ out,
        int num_blocks, int write_idx, float eps, float out_eps, int rows, int packs) {
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  V* pre                 = reinterpret_cast<V*>(prefix);
  V* o                   = reinterpret_cast<V*>(out);
  const int per_rank     = (packs + ngpus - 1) / ngpus;
  const int slice        = (per_rank + kWaveSize - 1) / kWaveSize * kWaveSize;
  const int col0         = min(p.rank * slice, packs);
  const int cols         = max(0, min(slice, packs - col0));  // a late rank's may be short or none
  const int my_rows      = rows > static_cast<int>(blockIdx.x)
                               ? (rows - blockIdx.x + gridDim.x - 1) / gridDim.x
                               : 0;
  const auto f           = fragment<kRowPacks>(packs);
  // The block row `row` writes, or none.
  auto written = [&](int row) -> V* {
    return write_idx < 0 ? nullptr
                         : reinterpret_cast<V*>(blocks + row * block_stride_m +
                                                write_idx * block_stride_r);
  };

  // 1. Wait until every peer has launched, so its input is ready.
  block_stamp(0);
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);
  block_stamp(1);

  // THE RANKS' POINTERS AFTER THE BARRIER, as in the other two-shots (held across it they spilled).
  const auto peers = p2p::peers<T, ngpus>(p);
  const auto read  = [&](int r, int64_t i) { return p2p::read_input(peers[r], i); };
  const auto self  = p2p::self<T, ngpus>(p);

  // 2. This rank's columns of this block's rows, summed over the ranks in rank order, into this
  //    rank's scratch at their place in the tensor.
  for (int64_t e = threadIdx.x; e < int64_t{my_rows} * cols; e += blockDim.x) {
    const int64_t q   = e / cols;
    const int64_t row = blockIdx.x + q * gridDim.x;
    const int64_t i   = row * packs + col0 + (e - q * cols);
    p2p::write_scratch(self, i, peers_reduce(peers_load<T, ngpus>(read, i)));
  }
  block_stamp(2);

  // 3. Every rank's columns are in its scratch, and every peer has read this rank's input.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::visible>(p);
  block_stamp(3);

  // 4. This block's rows: each pack from the rank that owns its columns, then AttnRes, as the
  //    one-shot does. The next call's first sync keeps a rank from overwriting its scratch while
  //    it is read (a peer's next kernel starts only once this one has finished).
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t base = int64_t{row} * packs;
    V sum[kRowPacks];
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) {
      const int owner = min(f.at[k] / slice, ngpus - 1);
      sum[k]          = p2p::read_scratch(p2p::peer<T, ngpus>(p, owner), base + f.at[k]);
    }
    block_attn_res_row<T, kPrefix, kRowPacks>(
        sum, base, f, pre, written(row), blocks + int64_t{row} * block_stride_m, block_stride_r,
        norm_w, qk_w, out_norm_w, o, num_blocks, eps, out_eps, inv_hidden);
  }
  block_stamp(4);
}

}  // namespace hip_comms
