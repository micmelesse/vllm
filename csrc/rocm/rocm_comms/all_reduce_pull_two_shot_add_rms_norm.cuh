// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot all-reduce (the reduce-scatter pulled, the all-gather pushed) then RMSNorm
// (`all_reduce_pull_two_shot_rms_norm`), and all-reduce then add then RMSNorm (`all_reduce_pull_two_shot_add_rms_norm`): one body, a
// kernel per op, so a trace names the op that ran.

#pragma once

#include "p2p/p2p.cuh"
#include "common/common.cuh"

namespace hip_comms {

// THE SLICE IS COLUMNS, as in aiter's fused two-stage kernel: each rank owns packs
// [rank * S, rank * S + S) of EVERY row (S = packs / ngpus, rounded up). It sums its columns of a
// row over the ranks and pushes the sums into every rank's scratch, so after the sync each rank
// holds the whole all-reduced tensor locally and norms its rows itself: no remote gather, and the
// norm repeated on every rank instead of sent. Every rank computes the same bytes (the same sums
// in the same order), so every rank holds the same result. ROW q BELONGS TO BLOCK q % gridDim.x
// IN BOTH PHASES: after the sync a block may read only what the same block on a peer wrote.
template <typename T, typename W, int ngpus, bool kAdd, int kRowPacks>
DINLINE void all_reduce_pull_two_shot_add_rms_norm_body(p2p::DevComm p, T* __restrict__ out,
                                                        T* __restrict__ residual_out,
                                                        const T* __restrict__ residual,
                                                        const W* __restrict__ weight, float eps,
                                                        int rows, int packs) {
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  const V* res_in        = reinterpret_cast<const V*>(residual);
  const auto* wv         = reinterpret_cast<const vec<W, NL>*>(weight);
  V* res_out             = reinterpret_cast<V*>(residual_out);
  V* o                   = reinterpret_cast<V*>(out);
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  const int slice        = (packs + ngpus - 1) / ngpus;
  const int col0         = p.rank * slice;
  const int cols         = max(0, min(slice, packs - col0));  // the last rank's may be short
  const int my_rows      = rows > static_cast<int>(blockIdx.x)
                               ? (rows - blockIdx.x + gridDim.x - 1) / gridDim.x
                               : 0;
  const auto f           = fragment<kRowPacks>(packs);

  // 1. Wait until every peer has launched, so its input is ready.
  block_stamp(0);
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);
  block_stamp(1);

  // THE RANKS' POINTERS AFTER THE BARRIER, as in the pull kernels (held across it they spilled).
  const auto peers = p2p::peers<T, ngpus>(p);
  const auto read  = [&](int r, int64_t i) { return p2p::read_input(peers[r], i); };

  // 2. This rank's columns of this block's rows, summed over the ranks in rank order and pushed
  //    to every rank (itself too), at their place in the tensor.
  for (int64_t e = threadIdx.x; e < int64_t{my_rows} * cols; e += blockDim.x) {
    const int64_t q   = e / cols;
    const int64_t row = blockIdx.x + q * gridDim.x;
    const int64_t i   = row * packs + col0 + (e - q * cols);
    const V sum       = peers_reduce<T, ngpus>(read, i);
#pragma unroll
    for (int r = 0; r < ngpus; ++r) p2p::write_scratch(peers[r], i, sum);
  }
  block_stamp(2);

  // 3. Every rank's sums are in this rank's scratch.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::visible>(p);
  block_stamp(3);

  // 4. This block's rows out of this rank's scratch: (kAdd) add the residual, then RMSNorm,
  //    rounding as the reference does (the one-shot kernel spells it out). The next call's first
  //    sync keeps a peer from pushing into this scratch while it is read (a peer's next kernel
  //    starts only once this one has finished).
  const auto self = p2p::self<T, ngpus>(p);
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t base = int64_t{row} * packs;
    float s[kRowPacks][NL];
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k)
      thread_unpack<T>(p2p::read_scratch(self, base + f.at[k]), s[k]);
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) {
      if constexpr (kAdd) {
        float r[NL];
        thread_unpack<T>(res_in[base + f.at[k]], r);
#pragma unroll
        for (int j = 0; j < NL; ++j) s[k][j] += r[j];
        if (f.in[k] != 0.0f) thread_store(res_out + base + f.at[k], thread_pack<T>(s[k]));
      }
    }
    float ss[1] = {thread_dot(s, s, f)};
    block_reduce<Sum>(ss);
    const float scale = rsqrtf(ss[0] * inv_hidden + eps);
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) {
      const vec<W, NL> w = wv[f.at[k]];
      V normed;
#pragma unroll
      for (int j = 0; j < NL; ++j) {
        const float x = static_cast<float>(static_cast<W>(s[k][j] * scale));
        normed.d[j]   = static_cast<T>(static_cast<W>(x * static_cast<float>(w.d[j])));
      }
      if (f.in[k] != 0.0f) thread_store(o + base + f.at[k], normed);
    }
  }
  block_stamp(4);
}

// THE KERNELS, one per op, both the body above.
template <typename T, typename W, int ngpus, int kRowPacks>
__global__ void __launch_bounds__(kMaxThreads, 1) all_reduce_pull_two_shot_rms_norm(
    p2p::DevComm p, T* __restrict__ out, const W* __restrict__ weight, float eps, int rows,
    int packs) {
  all_reduce_pull_two_shot_add_rms_norm_body<T, W, ngpus, false, kRowPacks>(
      p, out, nullptr, nullptr, weight, eps, rows, packs);
}

template <typename T, typename W, int ngpus, int kRowPacks>
__global__ void __launch_bounds__(kMaxThreads, 1) all_reduce_pull_two_shot_add_rms_norm(
    p2p::DevComm p, T* __restrict__ out, T* __restrict__ residual_out,
    const T* __restrict__ residual, const W* __restrict__ weight, float eps, int rows,
    int packs) {
  all_reduce_pull_two_shot_add_rms_norm_body<T, W, ngpus, true, kRowPacks>(
      p, out, residual_out, residual, weight, eps, rows, packs);
}

}  // namespace hip_comms
