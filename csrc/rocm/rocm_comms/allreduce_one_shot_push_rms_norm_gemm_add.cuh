// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot push all-reduce, then RMSNorm, then a GEMM whose result is added into an
// output: the tail of Kimi-K3's latent MoE (`fused_all_reduce.latent_tail`).

#pragma once

#include "fusions/rms_norm_gemm_add.cuh"
#include "p2p/push.cuh"
#include "utils.cuh"

namespace hip_comms {

// PUSH (see allreduce_one_shot_push.cuh): every rank's rows, encoded by kBits' Codec, into
// every rank's inbox; one barrier; each block reduces and norms its rows out of its own
// inbox into our scratch after the inbox; a grid barrier; the GEMM over every row, as the
// pull kernel does. At most kRows rows: one GEMM pass.
// kLanesPerCol is the GEMM's lanes per column, tuned in launch.cuh.
template <typename T, int ngpus, int kBits, int kLanesPerCol>
__global__ void __launch_bounds__(kMaxThreads, 1) allreduce_one_shot_push_rms_norm_gemm_add(
    p2p::Peers p, const T* __restrict__ norm_w, float eps, const T* __restrict__ gemm_w,
    int n_cols, T* __restrict__ out, int64_t out_stride, int out_col0, int rows,
    int packs) {
  using V          = typename traits<T>::V;
  using C          = p2p::Codec<T, kBits>;
  constexpr int NL = traits<T>::N;
  using core       = p2p::Core<T, ngpus>;
  using push       = p2p::Push<T, ngpus, C>;
  namespace fusion = fusions::rms_norm_gemm_add;
  core::start(p);
  const auto in  = core::inputs(p);
  const int rank = p.rank;
  const p2p::Inbox<C, ngpus> box(rows * blockDim.x);
  // The normed rows, plain, after the inbox.
  const int normed = box.end();
  push::broadcast_rows(p, in, box, rows, packs);

  core::peer_block_barrier(p);

  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    V sum[kMaxRowPacks];
    push::reduce_row(p, box, row, packs, sum);
    fusion::norm_row<T>(
        sum, reinterpret_cast<const V*>(norm_w), packs, inv_hidden, eps,
        [&](int, int i, const V& v) { core::put(p, rank, normed + row * packs + i, v); });
  }

  // The GEMM reads rows other blocks of this rank wrote, in our own scratch.
  core::grid_barrier(p);

  fusion::gemm<kLanesPerCol, T>(
      [&](int r) { return core::ptr(p, rank, normed + r * packs, packs); }, rows, gemm_w,
      n_cols, packs, out, out_stride, out_col0);
}

}  // namespace hip_comms
