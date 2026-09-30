// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce, then RMSNorm, then a GEMM whose result is added into an
// output: the tail of Kimi-K3's latent MoE (`fused_all_reduce.latent_tail`).

#pragma once

#include "fusions/rms_norm_gemm_add.cuh"
#include "p2p/p2p.cuh"
#include "common/memory.cuh"
#include "common/reduce.cuh"

namespace hip_comms {

// Each rank reduces and norms the rows it owns into its scratch, row-major; after the sync
// every rank copies every normed row into `workspace` ([rows, packs] of its own); a grid sync;
// the GEMM over every row, fusion::kRows per pass. THE SAME BLOCK AND THREAD INDEX A PACK IN
// BOTH PHASES: after the sync a block may read only what the same block on a peer wrote.
template <typename T, int ngpus, int kLanesPerCol, int kRowPacks>
__global__ void __launch_bounds__(fusions::rms_norm_gemm_add::max_threads(kLanesPerCol), 1)
    all_reduce_pull_two_shot_rms_norm_gemm_add(
    p2p::Peers p, const T* __restrict__ norm_w, float eps, const T* __restrict__ gemm_w,
    int n_cols, T* __restrict__ out, int64_t out_stride, int out_col0,
    T* __restrict__ workspace, int rows, int packs) {
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  namespace fusion       = fusions::rms_norm_gemm_add;
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  const V* weight        = reinterpret_cast<const V*>(norm_w);
  V* normed              = reinterpret_cast<V*>(workspace);
  const int slice_rows   = (rows + ngpus - 1) / ngpus;

  // THE RANKS' POINTERS BEFORE THE BARRIER: their loads hide under its wait.
  p2p::Peer<T, ngpus> all[ngpus];
#pragma unroll
  for (int r = 0; r < ngpus; ++r) all[r] = p2p::peer<T, ngpus>(p, r);
  const auto read = [&](int r, int64_t i) { return p2p::read_input(all[r], i); };
  const auto self = p2p::self<T, ngpus>(p);

  // 1. Wait until every peer has launched, so its input is ready.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);

  // 2. This rank's rows: read each from every rank in rank order, sum, norm, into this rank's
  //    scratch.
  const int first = p.rank * slice_rows;
  const int last  = min(first + slice_rows, rows);
  for (int row = first + blockIdx.x; row < last; row += gridDim.x) {
    V sum[kRowPacks];
    sum_row<T, ngpus>(read, row, packs, sum);
    const int64_t at = int64_t{row - first} * packs;
    fusion::norm_row<T>(sum, weight, packs, inv_hidden, eps,
                        [&](int, int i, const V& v) { p2p::write_scratch(self, at + i, v); });
  }

  // 3. Every rank's normed rows are visible to its peers.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::visible>(p);

  // 4. Every owner's normed rows out of its scratch, into the workspace.
  for (int l = blockIdx.x; l < slice_rows; l += gridDim.x) {
    for (int i = threadIdx.x; i < packs; i += blockDim.x) {
    // EVERY OWNER'S PACK LOADED BEFORE ANY IS STORED: the compiler cannot prove the output
    // and the peers' scratch apart, so a store between two loads holds the next load back
    // until the store is done, and the eight owners' round trips run one after another.
      V got[ngpus] = {};
#pragma unroll
      for (int r = 0; r < ngpus; ++r)
        if (r * slice_rows + l < rows)
          got[r] = p2p::read_scratch(all[r], int64_t{l} * packs + i);
#pragma unroll
      for (int r = 0; r < ngpus; ++r) {
        const int row = r * slice_rows + l;
        if (row < rows) store_global(normed + int64_t{row} * packs + i, got[r]);
      }
    }
  }

  // 5. The GEMM reads rows other blocks of this rank copied.
  p2p::barrier<ngpus, p2p::Among::grid, p2p::Ensure::visible>(p);

  // 6. The GEMM over every row, fusion::kRows per pass.
  for (int r0 = 0; r0 < rows; r0 += fusion::kRows) {
    fusion::gemm<kLanesPerCol, T>([&](int r) { return normed + (r0 + r) * packs; },
                                  min(fusion::kRows, rows - r0), gemm_w, n_cols, packs,
                                  out + r0 * out_stride, out_stride, out_col0);
  }
}

}  // namespace hip_comms
