// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce then RMSNorm (`all_reduce_pull_two_shot_rms_norm`), and
// all-reduce then add then RMSNorm (`all_reduce_pull_two_shot_add_rms_norm`): one body, a
// kernel per op, so a trace names the op that ran.

#pragma once

#include "fusions/add_rms_norm.cuh"
#include "p2p/p2p.cuh"
#include "common/memory.cuh"
#include "common/reduce.cuh"

namespace hip_comms {

// THE SLICE IS ROWS: each rank owns whole rows, so it finishes the norm alone. It reduces
// its rows, (kAdd: adds the replicated residual,) norms them and leaves the out rows and
// (kAdd) the residual rows in its scratch, row-major; after the sync every rank copies every
// owner's rows out. Every output element is computed by one rank, so every rank holds the same
// bytes. THE SAME BLOCK AND THREAD INDEX A PACK IN BOTH PHASES: after the sync a block may read
// only what the same block on a peer wrote.
template <typename T, typename W, int ngpus, bool kAdd, int kRowPacks>
DINLINE void all_reduce_pull_two_shot_add_rms_norm_body(p2p::Peers p, T* __restrict__ out,
                                                        T* __restrict__ residual_out,
                                                        const T* __restrict__ residual,
                                                        const W* __restrict__ weight, float eps,
                                                        int rows, int packs) {
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  namespace fusion       = fusions::add_rms_norm;
  const V* res_in        = reinterpret_cast<const V*>(residual);
  const auto* wv         = reinterpret_cast<const vec<W, NL>*>(weight);
  V* res_out             = reinterpret_cast<V*>(residual_out);
  V* o                   = reinterpret_cast<V*>(out);
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  const int slice_rows   = (rows + ngpus - 1) / ngpus;
  const int64_t res_at   = int64_t{slice_rows} * packs;  // the residual rows, after the out rows

  // 1. Wait until every peer has launched, so its input is ready.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);

  // 2. This rank's rows: read each from every rank in rank order, sum, norm, and leave the
  //    result in this rank's scratch.
  p2p::Peer<T, ngpus> all[ngpus];
#pragma unroll
  for (int r = 0; r < ngpus; ++r) all[r] = p2p::peer<T, ngpus>(p, r);
  const auto read = [&](int r, int64_t i) { return p2p::read_input(all[r], i); };
  const auto self = p2p::self<T, ngpus>(p);
  const int first  = p.rank * slice_rows;
  const int last   = min(first + slice_rows, rows);
  for (int row = first + blockIdx.x; row < last; row += gridDim.x) {
    V sum[kRowPacks];
    sum_row<T, ngpus>(read, row, packs, sum);
    const int64_t at = int64_t{row - first} * packs;
    fusion::row<T, W, kAdd>(
        sum, res_in, wv, row, packs, inv_hidden, eps,
        [&](int, int i, const V& v) { p2p::write_scratch(self, res_at + at + i, v); },
        [&](int, int i, const V& v) { p2p::write_scratch(self, at + i, v); });
  }

  // 3. Every rank's rows are visible to its peers.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::visible>(p);

  // 4. Every owner's rows out of its scratch, at their place in the output. The next call's
  //    first sync keeps a rank from overwriting its scratch while it is read.
  for (int l = blockIdx.x; l < slice_rows; l += gridDim.x) {
    for (int i = threadIdx.x; i < packs; i += blockDim.x) {
    // EVERY OWNER'S PACK LOADED BEFORE ANY IS STORED: the compiler cannot prove the output
    // and the peers' scratch apart, so a store between two loads holds the next load back
    // until the store is done, and the eight owners' round trips run one after another.
      const int64_t at = int64_t{l} * packs + i;
      V got[ngpus] = {}, got_res[ngpus] = {};
#pragma unroll
      for (int r = 0; r < ngpus; ++r) {
        if (r * slice_rows + l >= rows) continue;
        got[r] = p2p::read_scratch(all[r], at);
        if constexpr (kAdd) got_res[r] = p2p::read_scratch(all[r], res_at + at);
      }
#pragma unroll
      for (int r = 0; r < ngpus; ++r) {
        const int row = r * slice_rows + l;
        if (row >= rows) continue;
        store_global(o + int64_t{row} * packs + i, got[r]);
        if constexpr (kAdd) store_global(res_out + int64_t{row} * packs + i, got_res[r]);
      }
    }
  }
}

// THE KERNELS, one per op, both the body above.
template <typename T, typename W, int ngpus, int kRowPacks>
__global__ void __launch_bounds__(kMaxThreads, 1) all_reduce_pull_two_shot_rms_norm(
    p2p::Peers p, T* __restrict__ out, const W* __restrict__ weight, float eps, int rows,
    int packs) {
  all_reduce_pull_two_shot_add_rms_norm_body<T, W, ngpus, false, kRowPacks>(
      p, out, nullptr, nullptr, weight, eps, rows, packs);
}

template <typename T, typename W, int ngpus, int kRowPacks>
__global__ void __launch_bounds__(kMaxThreads, 1) all_reduce_pull_two_shot_add_rms_norm(
    p2p::Peers p, T* __restrict__ out, T* __restrict__ residual_out,
    const T* __restrict__ residual, const W* __restrict__ weight, float eps, int rows,
    int packs) {
  all_reduce_pull_two_shot_add_rms_norm_body<T, W, ngpus, true, kRowPacks>(
      p, out, residual_out, residual, weight, eps, rows, packs);
}

}  // namespace hip_comms
