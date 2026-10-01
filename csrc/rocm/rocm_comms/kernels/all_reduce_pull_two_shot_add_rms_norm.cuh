// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce then RMSNorm (`all_reduce_pull_two_shot_rms_norm`), and
// all-reduce then add then RMSNorm (`all_reduce_pull_two_shot_add_rms_norm`): one body, a
// kernel per op, so a trace names the op that ran. THE SLICE IS ROWS, for large inputs: the
// column-slice kernel (all_reduce_push_two_shot_*) wins small ones and loses at prefill.

#pragma once

#include "../p2p/p2p.cuh"
#include "../common/common.cuh"

namespace hip_comms {

// THE SLICE IS ROWS: each rank owns whole rows, so it computes each row's norm alone. It reduces
// its rows and leaves one row each in its scratch, row-major: the normed row, or (kAdd) the new
// residual and the row's RMS scale; after the sync every rank copies every owner's rows out,
// (kAdd) norming them by their scales as it goes. Every rank does the same arithmetic on the same
// bytes, so every rank holds the same result. THE SAME BLOCK AND THREAD INDEX A PACK IN BOTH
// PHASES: after the sync a block may read only what the same block on a peer wrote.
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
  // kAdd: each owned row's RMS scale, a pack a row (the float in its first lane), after the rows.
  const int64_t scale_at = int64_t{slice_rows} * packs;
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
  //    and leave the normed rows or (kAdd) the residual rows and scales in this rank's scratch.
  //    PIPELINED: the next row's loads go out before this row's reduction and norm, so a block's
  //    compute runs under its next round trip instead of between them (a block had ~14 rows at
  //    4096 tokens, each 1.24 us of reduction and norm on the critical path: 2026-10-01T00-01-54Z).
  const int first = p.rank * slice_rows;
  const int last  = min(first + slice_rows, rows);
  // Row `row`'s packs from every rank.
  const auto load = [&](int row) { return peers_load<T, ngpus>(read, row, packs, f); };
  // THE WEIGHT ONCE, AND EVERY OTHER LOAD BEFORE THE NEXT ROW'S: loads complete in issue order, so
  // waiting on one issued after the peers' would wait on the peers' too.
  vec<W, NL> w[kRowPacks];
#pragma unroll
  for (int k = 0; k < kRowPacks; ++k) w[k] = wv[f.at[k]];
  // ONE ROW: its residual, then the next row's peer loads into `next`, then this row's sum (its
  // wait covers only its own, older, loads), so the next round trip runs under the reduction and
  // norm.
  using Packs = PeerPacks<T, ngpus, kRowPacks>;
  const auto one_row = [&](int row, const Packs& cur, Packs& next) {
    const int64_t base = int64_t{row} * packs;
    const int64_t at   = int64_t{row - first} * packs;
    V res[kRowPacks];
    if constexpr (kAdd) {
#pragma unroll
      for (int k = 0; k < kRowPacks; ++k) res[k] = res_in[base + f.at[k]];
    }
    // ONLY A ROW THAT EXISTS: issued here, never hoisted, so the block-uniform branch costs
    // nothing, where a clamped unconditional load re-read the last row (a block's whole round trip
    // again; at 256 tokens every block has one row: 2026-10-01T01-07-56Z).
    if (row + static_cast<int>(gridDim.x) < last) next = load(row + gridDim.x);
    V sum[kRowPacks];
    peers_reduce(cur, sum);
    float s[kRowPacks][NL];
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) thread_unpack<T>(sum[k], s[k]);
    block_stamp(2);
    // The norm, rounding as the reference does (see the one-shot kernel):
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) {
      if constexpr (kAdd) {
        float r[NL];
        thread_unpack<T>(res[k], r);
#pragma unroll
        for (int j = 0; j < NL; ++j) s[k][j] += r[j];
        if (f.in[k] != 0.0f) p2p::write_scratch(self, at + f.at[k], thread_pack<T>(s[k]));
      }
    }
    float ss[1] = {thread_dot(s, s, f)};
    block_reduce<Sum>(ss);
    block_stamp(3);
    const float scale = rsqrtf(ss[0] * inv_hidden + eps);
    // kAdd LEAVES THE NEW RESIDUAL AND ITS SCALE, NOT THE NORMED ROW: every rank norms it while
    // gathering, so the link carries one row a row, as a plain all-reduce does (gathering both the
    // normed row and the residual was twice that: 215.9 against 155.2 us at 4096 tokens,
    // 2026-10-01T02-17-54Z).
    if constexpr (kAdd) {
      if (threadIdx.x == 0) {
        const vec<float, 4> sc = {{scale, 0.0f, 0.0f, 0.0f}};
        p2p::write_scratch(self, scale_at + (row - first), __builtin_bit_cast(V, sc));
      }
      return;
    }
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) {
      V normed;
#pragma unroll
      for (int j = 0; j < NL; ++j) {
        const float x = static_cast<float>(static_cast<W>(s[k][j] * scale));
        normed.d[j]   = static_cast<T>(static_cast<W>(x * static_cast<float>(w[k].d[j])));
      }
      if (f.in[k] != 0.0f) p2p::write_scratch(self, at + f.at[k], normed);
    }
  };
  // PING-PONG: two buffers that trade roles each row, so no row copies its packs into the other
  // (a copy cost 32 moves a row at one pack a thread: ISA 2026-10-01T00-58-37Z).
  Packs a, b;
  int row = first + blockIdx.x;
  if (row < last) a = load(row);
  for (; row < last; row += 2 * gridDim.x) {
    one_row(row, a, b);
    if (row + static_cast<int>(gridDim.x) >= last) break;
    one_row(row + gridDim.x, b, a);
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
  //    kAdd: the owners' rows are the new residual; each is normed here by its owner's scale,
  //    which a row's 8 scales bring through LDS once (uncached scratch, read per pack, would
  //    double the bytes again).
  __shared__ float scales[ngpus];
  for (int l = blockIdx.x; l < slice_rows; l += gridDim.x) {
    if constexpr (kAdd) {
      if (threadIdx.x == 0) {
        V sc[ngpus];
#pragma unroll
        for (int r = 0; r < ngpus; ++r) sc[r] = p2p::read_scratch(peers[r], scale_at + l);
#pragma unroll
        for (int r = 0; r < ngpus; ++r) scales[r] = __builtin_bit_cast(vec<float, 4>, sc[r]).d[0];
      }
      __syncthreads();
    }
    for (int i = threadIdx.x; i < packs; i += blockDim.x) {
      const int64_t at = int64_t{l} * packs + i;
      V got[ngpus];
#pragma unroll
      for (int r = 0; r < ngpus; ++r) got[r] = p2p::read_scratch(peers[r], at);
      vec<W, NL> w;
      if constexpr (kAdd) w = wv[i];
#pragma unroll
      for (int r = 0; r < ngpus; ++r) {
        const int row = r * slice_rows + l;
        if (row >= rows) continue;
        if constexpr (kAdd) {
          thread_store(res_out + int64_t{row} * packs + i, got[r]);
          float x[NL];
          thread_unpack<T>(got[r], x);
          V normed;
#pragma unroll
          for (int j = 0; j < NL; ++j) {
            const float y = static_cast<float>(static_cast<W>(x[j] * scales[r]));
            normed.d[j]   = static_cast<T>(static_cast<W>(y * static_cast<float>(w.d[j])));
          }
          thread_store(o + int64_t{row} * packs + i, normed);
        } else {
          thread_store(o + int64_t{row} * packs + i, got[r]);
        }
      }
    }
    if constexpr (kAdd) __syncthreads();
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
