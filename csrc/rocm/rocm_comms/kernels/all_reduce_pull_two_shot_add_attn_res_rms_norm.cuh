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
// ITS OWN GRID: the reduce-scatter on a few blocks (reads), AttnRes on all of them (compute a
// row), so a world barrier between them. `blocks` is [rows, num_sources, hidden] with row and
// source strides in elements; `write_idx` < 0 writes no block. FOLDING EARLY (a `workspace`, else
// null): the blocks that do not reduce fold each row's stored sources into it before the first
// barrier, while the others reduce-scatter and the late ranks arrive, and after the world barrier a
// row folds only its own sum: fp32 [rows][hidden], then each row's running max and denominator.
template <typename T, int ngpus, bool kPrefix, int kRowPacks>
__global__ void __launch_bounds__(kMaxThreads, 1)
    all_reduce_pull_two_shot_add_attn_res_rms_norm(
        p2p::DevComm p, T* __restrict__ prefix, T* __restrict__ blocks,
        int64_t block_stride_m, int64_t block_stride_r, const T* __restrict__ norm_w,
        const T* __restrict__ qk_w, const T* __restrict__ out_norm_w, T* __restrict__ out,
        int num_blocks, int write_idx, float eps, float out_eps, int rows, int packs,
        float* __restrict__ workspace) {
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

  const int reducers = min(static_cast<int>(gridDim.x), kBuild.attn_res_reduce_blocks);
  // Each row's partial: its weighted sum, a pack's 8 floats at the pack's place, then its stats.
  static_assert(NL == 8, "the partial is stored and read as two float4 a pack");
  const int64_t row_floats = int64_t{packs} * NL;
  float* const stats       = workspace + int64_t{rows} * row_floats;
  float w[kRowPacks][NL];
  if (workspace != nullptr) thread_attn_res_weights<T, kRowPacks>(norm_w, qk_w, f, w);

  // 0. FOLDING EARLY, by the blocks that do not reduce: needs nothing of the peers, so before the
  //    first barrier.
  if (workspace != nullptr && static_cast<int>(blockIdx.x) >= reducers) {
    for (int row = blockIdx.x - reducers; row < rows; row += gridDim.x - reducers) {
      float m[kRowPacks][NL];
      OnlineSoftmax softmax;
      block_attn_res_fold<T, kRowPacks>(blocks + int64_t{row} * block_stride_m, block_stride_r, f,
                                        w, num_blocks, eps, inv_hidden, m, softmax);
      float* const at = workspace + int64_t{row} * row_floats;
#pragma unroll
      for (int k = 0; k < kRowPacks; ++k)
        if (f.in[k] != 0.0f) {
          float4* const to = reinterpret_cast<float4*>(at + int64_t{f.at[k]} * NL);
          to[0] = make_float4(m[k][0], m[k][1], m[k][2], m[k][3]);
          to[1] = make_float4(m[k][4], m[k][5], m[k][6], m[k][7]);
        }
      if (threadIdx.x == 0) {
        stats[2 * int64_t{row}]     = softmax.max;
        stats[2 * int64_t{row} + 1] = softmax.denominator;
      }
    }
  }

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

  // 4. This block's rows: each pack from the rank that owns its columns, then AttnRes, as the
  //    one-shot does. The next call's first sync keeps a rank from overwriting its scratch while
  //    it is read (a peer's next kernel starts only once this one has finished).
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t base = int64_t{row} * packs;
    V sum[kRowPacks];
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) {
      const int owner = min(f.at[k] / slice, ngpus - 1);
      sum[k]          = p2p::read_scratch(p2p::scratch<T, ngpus>(p, owner), base + f.at[k]);
    }
    if (workspace != nullptr) {
      // The row's stored sources, folded early, then its own sum.
      const float* const at = workspace + int64_t{row} * row_floats;
      float m[kRowPacks][NL];
#pragma unroll
      for (int k = 0; k < kRowPacks; ++k) {
        float4 lo = make_float4(0.f, 0.f, 0.f, 0.f), hi = lo;
        if (f.in[k] != 0.0f) {
          const float4* const from = reinterpret_cast<const float4*>(at + int64_t{f.at[k]} * NL);
          lo = from[0];
          hi = from[1];
        }
        m[k][0] = lo.x, m[k][1] = lo.y, m[k][2] = lo.z, m[k][3] = lo.w;
        m[k][4] = hi.x, m[k][5] = hi.y, m[k][6] = hi.z, m[k][7] = hi.w;
      }
      const OnlineSoftmax softmax{stats[2 * int64_t{row}], stats[2 * int64_t{row} + 1]};
      block_attn_res_finish<T, kPrefix, kRowPacks>(sum, base, f, pre, written(row), w, m, softmax,
                                                   out_norm_w, o, eps, out_eps, inv_hidden);
    } else {
      block_attn_res_row<T, kPrefix, kRowPacks>(
          sum, base, f, pre, written(row), blocks + int64_t{row} * block_stride_m, block_stride_r,
          norm_w, qk_w, out_norm_w, o, num_blocks, eps, out_eps, inv_hidden);
    }
  }
  block_stamp(4);
}

}  // namespace hip_comms
