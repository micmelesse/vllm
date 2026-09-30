// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce then RMSNorm (`all_reduce_pull_two_shot_rms_norm`), and
// all-reduce then add then RMSNorm (`all_reduce_pull_two_shot_add_rms_norm`): one body, a
// kernel per op, so a trace names the op that ran.

#pragma once

#include "p2p/p2p.cuh"
#include "common/common.cuh"

namespace hip_comms {

// THE SLICE IS ROWS: each rank owns whole rows, so it finishes the norm alone. It reduces
// its rows, (kAdd: adds the replicated residual,) norms them and leaves the out rows and
// (kAdd) the residual rows in its scratch, row-major; after the sync every rank copies every
// owner's rows out. Every output element is computed by one rank, so every rank holds the same
// bytes. THE SAME BLOCK AND THREAD INDEX A PACK IN BOTH PHASES: after the sync a block may read
// only what the same block on a peer wrote.
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
  const int slice_rows   = (rows + ngpus - 1) / ngpus;
  const int64_t res_at   = int64_t{slice_rows} * packs;  // the residual rows, after the out rows
  const auto f           = fragment<kRowPacks>(packs);

  // 1. Wait until every peer has launched, so its input is ready.
  block_stamp(0);
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);
  block_stamp(1);

  // THE RANKS' POINTERS AFTER THE BARRIER here: held across it, the 8-pack build keeps 68 B of
  // scratch (the ISA gate, 2026-09-30).
  const auto peers = p2p::peers<T, ngpus>(p);
  const auto read  = [&](int r, int64_t i) { return p2p::read_input(peers[r], i); };
  const auto self  = p2p::self<T, ngpus>(p);

  // 2. This rank's rows: read each from every rank in rank order and sum, then (kAdd) add the
  //    residual, then RMSNorm, rounding as the reference does (the one-shot kernel spells it out),
  //    and leave the out rows and (kAdd) the residual rows in this rank's scratch.
  const int first = p.rank * slice_rows;
  const int last  = min(first + slice_rows, rows);
  for (int row = first + blockIdx.x; row < last; row += gridDim.x) {
    const int64_t base = int64_t{row} * packs;
    const int64_t at   = int64_t{row - first} * packs;
    V sum[kRowPacks];
    peers_reduce<T, ngpus>(read, row, packs, f, sum);
    block_stamp(2);
    // The norm, rounding as the reference does (see the one-shot kernel):
    float s[kRowPacks][NL];
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) {
      thread_unpack<T>(sum[k], s[k]);
      if constexpr (kAdd) {
        float r[NL];
        thread_unpack<T>(res_in[base + f.at[k]], r);
#pragma unroll
        for (int j = 0; j < NL; ++j) s[k][j] += r[j];
        if (f.in[k] != 0.0f) p2p::write_scratch(self, res_at + at + f.at[k], thread_pack<T>(s[k]));
      }
    }
    float ss[1] = {thread_dot(s, s, f)};
    block_reduce<Sum>(ss);
    block_stamp(3);
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
      if (f.in[k] != 0.0f) p2p::write_scratch(self, at + f.at[k], normed);
    }
  }

  block_stamp(4);
  // 3. Every rank's rows are visible to its peers.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::visible>(p);
  block_stamp(5);

  // 4. Every owner's rows out of its scratch, at their place in the output. The next call's
  //    first sync keeps a rank from overwriting its scratch while it is read. EVERY OWNER'S PACK
  //    LOADED BEFORE ANY IS STORED, and every load unconditional (each rank's scratch holds
  //    slice_rows rows, so a slot past the last row is real): a store between two loads, or a
  //    load under an `if`, made the eight owners' round trips run one after another.
  for (int l = blockIdx.x; l < slice_rows; l += gridDim.x) {
    for (int i = threadIdx.x; i < packs; i += blockDim.x) {
      const int64_t at = int64_t{l} * packs + i;
      V got[ngpus], got_res[ngpus];
#pragma unroll
      for (int r = 0; r < ngpus; ++r) {
        got[r] = p2p::read_scratch(peers[r], at);
        if constexpr (kAdd) got_res[r] = p2p::read_scratch(peers[r], res_at + at);
      }
#pragma unroll
      for (int r = 0; r < ngpus; ++r) {
        const int row = r * slice_rows + l;
        if (row >= rows) continue;
        thread_store(o + int64_t{row} * packs + i, got[r]);
        if constexpr (kAdd) thread_store(res_out + int64_t{row} * packs + i, got_res[r]);
      }
    }
  }
  block_stamp(6);
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
