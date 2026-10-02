// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce, then RMSNorm, then a GEMM whose result is written into an output
// (`all_reduce_pull_two_shot_rms_norm_gemm`) or added into it (`..._gemm_add`, the tail of
// Kimi-K3's latent MoE): one body, a kernel per op, so a trace names the op that ran.

#pragma once

#include "../p2p/p2p.cuh"
#include "../common/common.cuh"

namespace hip_comms {

// Each rank reduces and norms the rows it owns into its scratch, row-major; after the sync
// every rank copies every normed row into `workspace` ([rows, packs] of its own); a grid sync;
// the GEMM over every row, kBuild.kernels.gemm_rows per pass. THE SAME BLOCK AND THREAD INDEX A
// PACK IN BOTH PHASES: after the sync a block may read only what the same block on a peer wrote.
template <typename T, int ngpus, int kLanesPerCol, int kRowPacks, bool kAdd>
DINLINE void all_reduce_pull_two_shot_rms_norm_gemm_body(
    p2p::DevComm p, const T* __restrict__ norm_w, float eps, const T* __restrict__ gemm_w,
    int n_cols, T* __restrict__ out, int64_t out_stride,
    T* __restrict__ workspace, int rows, int packs) {
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  const V* weight        = reinterpret_cast<const V*>(norm_w);
  V* normed              = reinterpret_cast<V*>(workspace);
  const int cols = packs * NL;  // the row, in elements
  const auto thread_cols = thread_offs<T>(Tile<1, kRowPacks>{rows, cols, 0, 0});
  const int slice_rows   = (rows + ngpus - 1) / ngpus;

  // 1. Wait until every peer has launched, so its input is ready.
  block_stamp(0);
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);
  block_stamp(1);

  // THE RANKS' POINTERS AFTER THE BARRIER here: held across it, the 8-pack build keeps 68 B of
  // scratch (the ISA gate, 2026-09-30).
  const auto inputs = p2p::inputs<T, ngpus>(p);
  const auto scratches = p2p::scratches<T, ngpus>(p);
  const auto read = [&](int r, int64_t i) { return p2p::read_input(inputs[r], i); };
  const auto own_scratch = p2p::scratch<T, ngpus>(p, p.rank);

  // 2. This rank's rows: read each from every rank in rank order, sum, norm, into this rank's
  //    scratch.
  const int first = p.rank * slice_rows;
  const int last  = min(first + slice_rows, rows);
  for (int row = first + blockIdx.x; row < last; row += gridDim.x) {
    const int64_t at = int64_t{row - first} * packs;
    V sum[kRowPacks];
    peers_reduce(peers_load<T, ngpus>(read, row, packs, thread_cols), sum);
    // The norm, rounding as vLLM's reference rms_norm does (weight in T):
    //   out = T(T(s * rsqrt(mean(s^2) + eps)) * float(w)), s = float(T(sum over ranks))
    float s[kRowPacks][NL];
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) thread_unpack<T>(sum[k], s[k]);
    float ss[1] = {thread_dot(s, s, thread_cols)};
    block_reduce<Sum>(ss);
    const float scale = rsqrtf(ss[0] * inv_hidden + eps);
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) {
      float w[NL], x[NL];
      thread_unpack<T>(weight[thread_cols.offs_n[k]], w);
#pragma unroll
      for (int j = 0; j < NL; ++j)
        x[j] = static_cast<float>(static_cast<T>(s[k][j] * scale)) * w[j];
      if (thread_cols.mask_n[k] != 0.0f)
        p2p::write_scratch(own_scratch, at + thread_cols.offs_n[k], thread_pack<T>(x));
    }
  }

  // 3. Every rank's normed rows are visible to its peers.
  block_stamp(2);
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::visible>(p);
  block_stamp(3);

  // 4. Every owner's normed rows out of its scratch, into the workspace.
  for (int l = blockIdx.x; l < slice_rows; l += gridDim.x) {
    for (int i = threadIdx.x; i < packs; i += blockDim.x) {
    // EVERY OWNER'S PACK LOADED BEFORE ANY IS STORED: the compiler cannot prove the output
    // and the peers' scratch apart, so a store between two loads holds the next load back
    // until the store is done, and the eight owners' round trips run one after another.
      // Every load unconditional: each rank's scratch holds slice_rows rows, so a slot past the
      // last row is real, and a load under an `if` waits on the one before.
      V got[ngpus];
#pragma unroll
      for (int r = 0; r < ngpus; ++r)
        got[r] = p2p::read_scratch(scratches[r], int64_t{l} * packs + i);
#pragma unroll
      for (int r = 0; r < ngpus; ++r) {
        const int row = r * slice_rows + l;
        if (row < rows) thread_store(normed + int64_t{row} * packs + i, got[r]);
      }
    }
  }

  // 5. The GEMM reads rows other blocks of this rank copied.
  block_stamp(4);
  p2p::barrier<ngpus, p2p::Among::grid, p2p::Ensure::visible>(p);
  block_stamp(5);

  // 6. The GEMM over every row, kBuild.kernels.gemm_rows per pass.
  for (int r0 = 0; r0 < rows; r0 += kBuild.kernels.gemm_rows) {
    grid_gemm<kLanesPerCol, kAdd, T>([&](int r) { return normed + (r0 + r) * packs; },
                                     min(kBuild.kernels.gemm_rows, rows - r0), gemm_w, n_cols,
                                     packs, out + r0 * out_stride, out_stride);
  }
  block_stamp(6);
}

// THE KERNELS, one per op, both the body above: the GEMM's result written (rms_norm_gemm) or
// added into `out` (rms_norm_gemm_add).
template <typename T, int ngpus, int kLanesPerCol, int kRowPacks>
__global__ void __launch_bounds__(gemm_max_threads(kLanesPerCol), 1)
    all_reduce_pull_two_shot_rms_norm_gemm(
    p2p::DevComm p, const T* __restrict__ norm_w, float eps, const T* __restrict__ gemm_w,
    int n_cols, T* __restrict__ out, int64_t out_stride,
    T* __restrict__ workspace, int rows, int packs) {
  all_reduce_pull_two_shot_rms_norm_gemm_body<T, ngpus, kLanesPerCol, kRowPacks, false>(
      p, norm_w, eps, gemm_w, n_cols, out, out_stride, workspace, rows, packs);
}

template <typename T, int ngpus, int kLanesPerCol, int kRowPacks>
__global__ void __launch_bounds__(gemm_max_threads(kLanesPerCol), 1)
    all_reduce_pull_two_shot_rms_norm_gemm_add(
    p2p::DevComm p, const T* __restrict__ norm_w, float eps, const T* __restrict__ gemm_w,
    int n_cols, T* __restrict__ out, int64_t out_stride,
    T* __restrict__ workspace, int rows, int packs) {
  all_reduce_pull_two_shot_rms_norm_gemm_body<T, ngpus, kLanesPerCol, kRowPacks, true>(
      p, norm_w, eps, gemm_w, n_cols, out, out_stride, workspace, rows, packs);
}


}  // namespace hip_comms
