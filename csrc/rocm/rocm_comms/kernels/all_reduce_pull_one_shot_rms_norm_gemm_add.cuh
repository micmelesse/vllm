// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce, then RMSNorm, then a GEMM whose result is written into an output
// (`all_reduce_pull_one_shot_rms_norm_gemm`) or added into it (`..._gemm_add`, the tail of
// Kimi-K3's latent MoE): one body, a kernel per op, so a trace names the op that ran.

#pragma once

#include "../p2p/p2p.cuh"
#include "../common/common.cuh"

namespace hip_comms {

// Every rank reduces and norms every row into `workspace` ([rows, packs] of its own); a
// grid barrier; the GEMM over every row, kBuild.kernels.gemm_rows a pass.
// kLanesPerCol is the GEMM's lanes per column (the build's gemm_lanes).
template <typename T, int ngpus, int kLanesPerCol, int kRowPacks, bool kAdd>
DINLINE void all_reduce_pull_one_shot_rms_norm_gemm_body(
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

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto inputs = p2p::inputs<T, ngpus>(p);
  block_stamp(0);
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);
  block_stamp(1);
  const auto read = [&](int r, int64_t i) { return p2p::read_input(inputs[r], i); };

  // 2. Each of this block's rows: read it from every rank in rank order, sum, norm, into the
  //    workspace.
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t base = int64_t{row} * packs;
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
        thread_store(normed + base + thread_cols.offs_n[k], thread_pack<T>(x));
    }
  }

  // 3. The GEMM reads rows other blocks of this rank wrote.
  block_stamp(2);
  p2p::barrier<ngpus, p2p::Among::grid, p2p::Ensure::visible>(p);
  block_stamp(3);

  // 4. The GEMM over every row.
  for (int r0 = 0; r0 < rows; r0 += kBuild.kernels.gemm_rows)
    grid_gemm<kLanesPerCol, kAdd, T>([&](int r) { return normed + (r0 + r) * packs; },
                                     min(kBuild.kernels.gemm_rows, rows - r0), gemm_w, n_cols,
                                     packs, out + r0 * out_stride, out_stride);

  block_stamp(4);
  // 5. No rank may overwrite its input until every peer has read it.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::read>(p);
}

// THE KERNELS, one per op, both the body above: the GEMM's result written (rms_norm_gemm) or
// added into `out` (rms_norm_gemm_add).
template <typename T, int ngpus, int kLanesPerCol, int kRowPacks>
__global__ void __launch_bounds__(gemm_max_threads(kLanesPerCol), 1)
    all_reduce_pull_one_shot_rms_norm_gemm(
    p2p::DevComm p, const T* __restrict__ norm_w, float eps, const T* __restrict__ gemm_w,
    int n_cols, T* __restrict__ out, int64_t out_stride,
    T* __restrict__ workspace, int rows, int packs) {
  all_reduce_pull_one_shot_rms_norm_gemm_body<T, ngpus, kLanesPerCol, kRowPacks, false>(
      p, norm_w, eps, gemm_w, n_cols, out, out_stride, workspace, rows, packs);
}

template <typename T, int ngpus, int kLanesPerCol, int kRowPacks>
__global__ void __launch_bounds__(gemm_max_threads(kLanesPerCol), 1)
    all_reduce_pull_one_shot_rms_norm_gemm_add(
    p2p::DevComm p, const T* __restrict__ norm_w, float eps, const T* __restrict__ gemm_w,
    int n_cols, T* __restrict__ out, int64_t out_stride,
    T* __restrict__ workspace, int rows, int packs) {
  all_reduce_pull_one_shot_rms_norm_gemm_body<T, ngpus, kLanesPerCol, kRowPacks, true>(
      p, norm_w, eps, gemm_w, n_cols, out, out_stride, workspace, rows, packs);
}


}  // namespace hip_comms
