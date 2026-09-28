// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot push all-reduce then RMSNorm (`allreduce_two_shot_push_rms_norm`), and
// all-reduce then add then RMSNorm (`allreduce_two_shot_push_add_rms_norm`): one body, a
// kernel per op, so a trace names the op that ran.

#pragma once

#include "fusions/add_rms_norm.cuh"
#include "p2p/device.cuh"
#include "utils.cuh"

namespace hip_comms {

// Each rank owns ceil(rows/ngpus) WHOLE rows, as the pull kernel does, and every transfer
// is a store into a peer's inbox:
//
//   phase 1  each of our input rows, encoded, into its owner's inbox
//   peer_block_barrier
//   phase 2  our rows: reduced out of the inbox, (kAdd: add the residual,) normalised; the
//            normed row, encoded, and (kAdd) the residual row, unquantized, into every
//            rank's inboxes
//   peer_block_barrier
//   phase 3  every owner's rows out of our inboxes into our outputs
//
// THE RESIDUAL IS NEVER QUANTIZED (Codec 16 whatever kBits): it is rewritten every layer,
// so a codec's error would accumulate over depth.
// `weight` is in its own dtype W: T, or fp32 (see `fusion::row`).
template <typename T, typename W, int ngpus, int kBits, bool kAdd>
DINLINE void two_shot_push_add_rms_norm_body(p2p::Peers p, T* __restrict__ out,
                                             T* __restrict__ residual_out,
                                             const T* __restrict__ residual,
                                             const W* __restrict__ weight, float eps,
                                             int rows, int packs) {
  using V          = typename traits<T>::V;
  using C          = p2p::Codec<T, kBits>;
  using R          = p2p::Codec<T, 16>;
  constexpr int NL = traits<T>::N;
  using core       = p2p::Core<T, ngpus>;
  using push       = p2p::Push<T, ngpus, C>;
  using push_res   = p2p::Push<T, ngpus, R>;
  namespace fusion = fusions::add_rms_norm;
  core::start(p);
  const auto in    = core::inputs(p);
  const int rank   = p.rank;
  const int chunk  = (rows + ngpus - 1) / ngpus;
  const int groups = chunk * blockDim.x;
  const p2p::Inbox<C, ngpus> box_in(groups);
  const p2p::Inbox<C, ngpus> box_out(groups, 1, box_in.end());
  const p2p::Inbox<R, ngpus> box_res(groups, 1, box_out.end());
  push::scatter_rows(p, in, box_in, chunk, rows, packs);

  core::peer_block_barrier(p);

  {
    const V* res_in        = reinterpret_cast<const V*>(residual);
    const auto* w          = reinterpret_cast<const vec<W, NL>*>(weight);
    const int begin        = rank * chunk;
    const int end          = min(begin + chunk, rows);
    const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
    for (int row = begin + blockIdx.x; row < end; row += gridDim.x) {
      V sum[kMaxRowPacks];
      push::reduce_row(p, box_in, row - begin, packs, sum);
      V normed[kMaxRowPacks] = {}, res[kMaxRowPacks] = {};
      fusion::row<T, W, kAdd>(
          sum, res_in, w, row, packs, inv_hidden, eps,
          [&](int k, int, const V& v) { res[k] = v; },
          [&](int k, int, const V& v) { normed[k] = v; });
      push::broadcast_row(p, box_out, row - begin, packs, normed);
      if constexpr (kAdd) push_res::broadcast_row(p, box_res, row - begin, packs, res);
    }
  }

  core::peer_block_barrier(p);

  V* o = reinterpret_cast<V*>(out);
  push::gather_inbox_rows(p, box_out, chunk, rows, packs,
      [&](int row, int i, const V& v) { store_global(o + row * packs + i, v); });
  if constexpr (kAdd) {
    V* res_out = reinterpret_cast<V*>(residual_out);
    push_res::gather_inbox_rows(p, box_res, chunk, rows, packs,
        [&](int row, int i, const V& v) { store_global(res_out + row * packs + i, v); });
  }
}

// THE KERNELS, one per op, both the body above.
template <typename T, typename W, int ngpus, int kBits>
__global__ void __launch_bounds__(kMaxThreads, 1) allreduce_two_shot_push_rms_norm(
    p2p::Peers p, T* __restrict__ out, const W* __restrict__ weight, float eps, int rows,
    int packs) {
  two_shot_push_add_rms_norm_body<T, W, ngpus, kBits, false>(p, out, nullptr, nullptr,
                                                             weight, eps, rows, packs);
}

template <typename T, typename W, int ngpus, int kBits>
__global__ void __launch_bounds__(kMaxThreads, 1) allreduce_two_shot_push_add_rms_norm(
    p2p::Peers p, T* __restrict__ out, T* __restrict__ residual_out,
    const T* __restrict__ residual, const W* __restrict__ weight, float eps, int rows,
    int packs) {
  two_shot_push_add_rms_norm_body<T, W, ngpus, kBits, true>(p, out, residual_out, residual,
                                                            weight, eps, rows, packs);
}

}  // namespace hip_comms
