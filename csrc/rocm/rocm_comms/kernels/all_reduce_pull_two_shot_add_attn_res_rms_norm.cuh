// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce fused with Kimi-K3's attention residual (AttnRes) and its
// RMSNorm.

#pragma once

#include "../p2p/p2p.cuh"
#include "../common/common.cuh"
#include "shared/attn_res.cuh"

namespace hip_comms {

// THE SLICE IS COLUMNS, AND BOTH HALVES PULL, as the plain two-shot all-reduce moves its bytes:
// each rank sums its columns of every row over the ranks into its own scratch; after the sync each
// block reads its rows' columns from their owners and computes AttnRes on them itself. So the links
// carry what an all-reduce's do, AttnRes's two outputs (the prefix and out) are never gathered, and
// every rank does the AttnRes the unfused path would. A SLICE IS WHOLE WAVES (64 packs), so every
// wave's packs have one owner and p2p::scratch's rank is the same across the wave. EACH PHASE AT
// ITS OWN GRID: the reduce-scatter and the gather on a few blocks (reads queue behind the links past
// a few dozen), AttnRes on all of them from local memory (compute a row), so a world barrier and a
// grid barrier between them. `blocks` is [rows, num_sources, hidden] with row and
// source strides in elements; `write_idx` < 0 writes no block.
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
  const auto inputs = p2p::inputs<T, ngpus>(p);
  const auto read  = [&](int r, int64_t i) { return p2p::read_input(inputs[r], i); };
  const auto own_scratch  = p2p::scratch<T, ngpus>(p, p.rank);

  // 2. This rank's columns of every row, summed over the ranks in rank order, into this rank's
  //    scratch at their place in the tensor, BY THE FIRST kBuild.attn_res_reduce_blocks BLOCKS
  //    only: reads queue behind the links past a few dozen blocks (machine/hardware.cuh). THE (ROW,
  //    COLUMN) STEPS, NOT DIVIDED: a 64-bit division a pack was a software routine on every 16
  //    bytes.
  const int reducers = min(static_cast<int>(gridDim.x), kBuild.attn_res_reduce_blocks);
  if (cols > 0 && static_cast<int>(blockIdx.x) < reducers) {
    const int reduce_rows = (rows - static_cast<int>(blockIdx.x) + reducers - 1) / reducers;
    int q = threadIdx.x / cols;  // this thread's row among the block's, and its column
    int c = threadIdx.x - q * cols;
    const int dq = blockDim.x / cols, dc = blockDim.x - dq * cols;
    for (; q < reduce_rows;) {
      const int64_t i = (int64_t{blockIdx.x} + int64_t{q} * reducers) * packs + col0 + c;
      p2p::write_scratch(own_scratch, i, peers_reduce(peers_load<T, ngpus>(read, i)));
      q += dq;
      c += dc;
      if (c >= cols) {
        c -= cols;
        ++q;
      }
    }
  }
  block_stamp(2);

  // 3. Every rank's columns are in its scratch, and every peer has read this rank's input: A WORLD
  //    BARRIER, since the blocks that wrote a row's columns are not the ones that read them.
  p2p::barrier<ngpus, p2p::Among::world, p2p::Ensure::visible>(p);
  block_stamp(3);

  // 4. Every row's reduced columns pulled from their owners into `out`, BY THE FIRST
  //    kBuild.attn_res_gather_blocks BLOCKS only, a wave a (row, 64-pack chunk) so its owner is the
  //    same across the wave, kInFlight chunks a wave at once. `out` is this rank's own, read back by
  //    other blocks after the grid barrier and overwritten row by row with AttnRes's output.
  constexpr int kInFlight = 8;
  const int gatherers     = min(static_cast<int>(gridDim.x), kBuild.attn_res_gather_blocks);
  if (static_cast<int>(blockIdx.x) < gatherers) {
    const int waves  = blockDim.x / kWaveSize;
    const int lane   = threadIdx.x % kWaveSize;
    const int chunks = (packs + kWaveSize - 1) / kWaveSize;
    const int units  = rows * chunks;
    const int step   = gatherers * waves;
    for (int u0 = blockIdx.x * waves + threadIdx.x / kWaveSize; u0 < units; u0 += step * kInFlight) {
      V got[kInFlight];
      int64_t at[kInFlight];
#pragma unroll
      for (int j = 0; j < kInFlight; ++j) {
        const int u     = u0 + j * step;
        const int row   = u / chunks;
        const int chunk = u - row * chunks;
        const int col   = chunk * kWaveSize + lane;
        at[j]           = u < units && col < packs ? int64_t{row} * packs + col : -1;
        const int owner = min(chunk * kWaveSize / slice, ngpus - 1);
        if (at[j] >= 0) got[j] = p2p::read_scratch(p2p::scratch<T, ngpus>(p, owner), at[j]);
      }
#pragma unroll
      for (int j = 0; j < kInFlight; ++j)
        if (at[j] >= 0) thread_store(o + at[j], got[j]);
    }
  }
  block_stamp(4);
  p2p::barrier<ngpus, p2p::Among::grid, p2p::Ensure::visible>(p);
  block_stamp(5);

  // 5. This block's rows from `out`, then AttnRes, as the one-shot does. The next call's first sync
  //    keeps a rank from overwriting its scratch while it is read (a peer's next kernel starts only
  //    once this one has finished).
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t base = int64_t{row} * packs;
    V sum[kRowPacks];
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) sum[k] = thread_load(o + base + f.at[k]);
    block_attn_res_row<T, kPrefix, kRowPacks>(
        sum, base, f, pre, written(row), blocks + int64_t{row} * block_stride_m, block_stride_r,
        norm_w, qk_w, out_norm_w, o, num_blocks, eps, out_eps, inv_hidden);
  }
  block_stamp(6);
}

}  // namespace hip_comms
