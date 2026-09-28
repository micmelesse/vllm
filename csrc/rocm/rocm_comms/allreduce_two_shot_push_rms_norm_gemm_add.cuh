// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot push all-reduce, then RMSNorm, then a GEMM whose result is added into an
// output: the tail of Kimi-K3's latent MoE (`fused_all_reduce.latent_tail`).

#pragma once

#include "fusions/rms_norm_gemm_add.cuh"
#include "ipc.cuh"
#include "utils.cuh"

namespace hip_comms {

// Each rank owns ceil(rows/ngpus) WHOLE rows, as the pull kernel does, and every transfer
// is a store into a peer's inbox:
//
//   phase 1  each of our input rows, encoded, into its owner's inbox
//   peer_block_barrier
//   phase 2  our rows: reduced out of the inbox and normed; the normed row, encoded, into
//            every rank's inbox
//   peer_block_barrier
//   phase 3  every owner's normed rows out of our inbox, plain, into our scratch after
//            the inboxes
//   grid_barrier
//   phase 4  the GEMM over every row, kRows rows per pass, from our own scratch
//
// The same roundings as the pull kernel (the normed rows land as T; a codec below 16
// bits rounds them further). kLanesPerCol is the GEMM's lanes per column, tuned in
// launch.cuh.
template <typename T, int ngpus, int kBits, int kLanesPerCol>
__global__ void __launch_bounds__(kMaxThreads, 1) allreduce_two_shot_push_rms_norm_gemm_add(
    ipc::Peers p, const T* __restrict__ norm_w, float eps, const T* __restrict__ gemm_w,
    int n_cols, T* __restrict__ out, int64_t out_stride, int out_col0, int rows,
    int packs) {
  using V          = typename traits<T>::V;
  using C          = Codec<T, kBits>;
  constexpr int NL = traits<T>::N;
  namespace fusion = fusions::rms_norm_gemm_add;
  ipc::Comm<T, ngpus> c(p);
  const int rank   = c.rank();
  const int chunk  = (rows + ngpus - 1) / ngpus;
  const int groups = chunk * blockDim.x;
  const ipc::Inbox<C, ngpus> box_in(groups);
  const ipc::Inbox<C, ngpus> box_out(groups, 1, box_in.end());
  // Every normed row, plain, after the inboxes.
  const int normed = box_out.end();
  c.template scatter_rows<C>(box_in, chunk, rows, packs);

  c.peer_block_barrier();

  {
    const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
    const int begin        = rank * chunk;
    const int end          = min(begin + chunk, rows);
    for (int row = begin + blockIdx.x; row < end; row += gridDim.x) {
      V sum[kMaxRowPacks];
      c.template reduce_row<C>(box_in, row - begin, packs, sum);
      V n[kMaxRowPacks] = {};
      fusion::norm_row<T>(sum, reinterpret_cast<const V*>(norm_w), packs, inv_hidden, eps,
                          [&](int k, int, const V& v) { n[k] = v; });
      c.template broadcast_row<C>(box_out, row - begin, packs, n);
    }
  }

  c.peer_block_barrier();

  c.template gather_inbox_rows<C>(box_out, chunk, rows, packs,
                                  [&](int row, int i, const V& v) {
                                    c.put(rank, normed + row * packs + i, v);
                                  });

  // The GEMM reads rows other blocks of this rank wrote, in our own scratch.
  c.grid_barrier();

  for (int r0 = 0; r0 < rows; r0 += fusion::kRows) {
    fusion::gemm<kLanesPerCol, T>(
        [&](int r) { return c.ptr(rank, normed + (r0 + r) * packs, packs); },
        min(fusion::kRows, rows - r0), gemm_w, n_cols, packs, out + r0 * out_stride,
        out_stride, out_col0);
  }
}

}  // namespace hip_comms
