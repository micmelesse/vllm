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
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, bool ADD_RESIDUAL, int TILE_N, int THREADS_PER_BLOCK>
DINLINE void all_reduce_pull_two_shot_add_rms_norm_body(
    const p2p::PeerPtrs* __restrict__ peer_inputs, p2p::PeerPtrs peer_scratch,
    p2p::PeerSignals peer_signals, p2p::Signal* self_signal, int rank, uint64_t timeout_ticks,
    DTYPE* __restrict__ out, DTYPE* __restrict__ residual_out, const DTYPE* __restrict__ residual,
    const WEIGHT_DTYPE* __restrict__ weight, float eps, int rows, int packs) {
  using V                = typename traits<DTYPE>::V;
  constexpr int NL       = traits<DTYPE>::N;
  using Row              = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK>;
  using RowF             = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, float>;
  using Weight           = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, WEIGHT_DTYPE>;
  const auto* wv         = reinterpret_cast<const vec<WEIGHT_DTYPE, NL>*>(weight);
  V* res_out             = reinterpret_cast<V*>(residual_out);
  V* o                   = reinterpret_cast<V*>(out);
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  const int slice_rows   = (rows + WORLD - 1) / WORLD;
  // ADD_RESIDUAL: each owned row's RMS scale, a pack a row (the float in its first lane), after the rows.
  const int64_t scale_at = int64_t{slice_rows} * packs;
  const int cols = packs * NL;  // the row, in elements

  // 1. Wait until every peer has launched, so its input is ready.
  block_stamp(0);
  p2p::barrier<WORLD, p2p::Among::peers, p2p::Ensure::launched>(
      peer_signals, self_signal, rank, timeout_ticks);
  block_stamp(1);

  // THE RANKS' POINTERS AFTER THE BARRIER here: held across it, the 8-pack build keeps 68 B of
  // scratch (the ISA gate, 2026-09-30).
  const auto inputs = p2p::inputs<DTYPE, WORLD>(*peer_inputs);
  const auto scratches = p2p::scratches<DTYPE, WORLD>(peer_scratch);
  const auto input = [&](int r) { return inputs[r].data(); };
  const auto own_scratch  = p2p::scratch<DTYPE, WORLD>(peer_scratch, rank);

  // 2. This rank's rows: read each from every rank in rank order and sum, then (ADD_RESIDUAL) add the
  //    residual, then RMSNorm, rounding as the reference does (the one-shot kernel spells it out),
  //    and leave the normed rows or (ADD_RESIDUAL) the residual rows and scales in this rank's scratch.
  //    PIPELINED: the next row's loads go out before this row's reduction and norm, so a block's
  //    compute runs under its next round trip instead of between them (a block had ~14 rows at
  //    4096 tokens, each 1.24 us of reduction and norm on the critical path: 2026-10-01T00-01-54Z).
  const int first = rank * slice_rows;
  const int last  = min(first + slice_rows, rows);
  // Row `row` from every rank.
  using Peers = Row[WORLD];
  const auto load = [&](int row, Peers& got) {
#pragma unroll
    for (int r = 0; r < WORLD; ++r) got[r] = Row{rows, cols, row, 0};
    peers_load(got, input, cols);
  };
  // THE WEIGHT ONCE, AND EVERY OTHER LOAD BEFORE THE NEXT ROW'S: loads complete in issue order, so
  // waiting on one issued after the peers' would wait on the peers' too.
  Weight w{1, cols, 0, 0};
  thread_load(w, weight, 0);
  // ONE ROW: its residual, then the next row's peer loads into `next`, then this row's sum (its
  // wait covers only its own, older, loads), so the next round trip runs under the reduction and
  // norm. Its rows land in this rank's scratch at row - first.
  const auto one_row = [&](int row, const Peers& cur, Peers& next) {
    Row res{rows, cols, row, 0};
    if constexpr (ADD_RESIDUAL) thread_load(res, residual, cols);
    // ONLY A ROW THAT EXISTS: issued here, never hoisted, so the block-uniform branch costs
    // nothing, where a clamped unconditional load re-read the last row (a block's whole round trip
    // again; at 256 tokens every block has one row: 2026-10-01T01-07-56Z).
    if (row + static_cast<int>(gridDim.x) < last) load(row + gridDim.x, next);
    RowF s = peers_reduce(cur).template to<float>();
    block_stamp(2);
    // The norm, rounding as the reference does (see the one-shot kernel):
    if constexpr (ADD_RESIDUAL) {
      s = thread_add(s, res.template to<float>());
      Row added = s.template to<DTYPE>();
      added.offs_m = row - first;
      thread_store(own_scratch.data(), cols, added);
    }
    float ss[1];
    thread_dot(s, s, ss);
    block_reduce<Sum>(ss);
    block_stamp(3);
    const float scale = rsqrtf(ss[0] * inv_hidden + eps);
    // ADD_RESIDUAL LEAVES THE NEW RESIDUAL AND ITS SCALE, NOT THE NORMED ROW: every rank norms it while
    // gathering, so the link carries one row a row, as a plain all-reduce does (gathering both the
    // normed row and the residual was twice that: 215.9 against 155.2 us at 4096 tokens,
    // 2026-10-01T02-17-54Z).
    if constexpr (ADD_RESIDUAL) {
      if (threadIdx.x == 0) {
        const vec<float, 4> sc = {{scale, 0.0f, 0.0f, 0.0f}};
        p2p::write_scratch(own_scratch, scale_at + (row - first), __builtin_bit_cast(V, sc));
      }
      return;
    }
    // out = T(W(W(s * scale) * float(w))), as the reference rounds
    Row normed = thread_mul(thread_mul(s, scale).template to<WEIGHT_DTYPE>().template to<float>(),
                            w.template to<float>())
                     .template to<WEIGHT_DTYPE>()
                     .template to<DTYPE>();
    normed.offs_m = row - first;
    thread_store(own_scratch.data(), cols, normed);
  };
  // PING-PONG: two buffers that trade roles each row, so no row copies its packs into the other
  // (a copy cost 32 moves a row at one pack a thread: ISA 2026-10-01T00-58-37Z).
  Peers a, b;
  int row = first + blockIdx.x;
  if (row < last) load(row, a);
  for (; row < last; row += 2 * gridDim.x) {
    one_row(row, a, b);
    if (row + static_cast<int>(gridDim.x) >= last) break;
    one_row(row + gridDim.x, b, a);
  }

  block_stamp(4);
  // 3. Every rank's rows are visible to its peers.
  p2p::barrier<WORLD, p2p::Among::peers, p2p::Ensure::visible>(
      peer_signals, self_signal, rank, timeout_ticks);
  block_stamp(5);

  // 4. Every owner's rows out of its scratch, at their place in the output. The next call's
  //    first sync keeps a rank from overwriting its scratch while it is read. EVERY OWNER'S PACK
  //    LOADED BEFORE ANY IS STORED, and every load unconditional (each rank's scratch holds
  //    slice_rows rows, so a slot past the last row is real): a store between two loads, or a
  //    load under an `if`, made the eight owners' round trips run one after another.
  //    ADD_RESIDUAL: the owners' rows are the new residual; each is normed here by its owner's scale. Every
  //    thread loads the 8 scales after its 8 packs, all in flight before any wait (a wave's lanes
  //    read one scale address: one request a wave). Left to the compiler, the scales were loaded
  //    and waited on before the packs were issued, two round trips a row (ISA
  //    2026-10-01T02-33-11Z).
  const auto gathered = [&](int r, int64_t i) { return p2p::read_scratch(scratches[r], i); };
  for (int l = blockIdx.x; l < slice_rows; l += gridDim.x) {
    for (int i = threadIdx.x; i < packs; i += blockDim.x) {
      const int64_t at = int64_t{l} * packs + i;
      const PeerPacks<DTYPE, WORLD> rows_of = peers_load<DTYPE, WORLD>(gathered, at);
      V sc[WORLD];
      if constexpr (ADD_RESIDUAL) {
#pragma unroll
        for (int r = 0; r < WORLD; ++r) sc[r] = p2p::read_scratch(scratches[r], scale_at + l);
      }
      V got[WORLD];
#pragma unroll
      for (int r = 0; r < WORLD; ++r) got[r] = rows_of.p[r][0];
      vec<WEIGHT_DTYPE, NL> w;
      if constexpr (ADD_RESIDUAL) w = wv[i];
#pragma unroll
      for (int r = 0; r < WORLD; ++r) {
        const int row = r * slice_rows + l;
        if (row >= rows) continue;
        if constexpr (ADD_RESIDUAL) {
          thread_store(res_out + int64_t{row} * packs + i, got[r]);
          const float scale = __builtin_bit_cast(vec<float, 4>, sc[r]).d[0];
          float x[NL];
          thread_unpack<DTYPE>(got[r], x);
          V normed;
#pragma unroll
          for (int j = 0; j < NL; ++j) {
            const float y = static_cast<float>(static_cast<WEIGHT_DTYPE>(x[j] * scale));
            normed.d[j]   = static_cast<DTYPE>(static_cast<WEIGHT_DTYPE>(y * static_cast<float>(w.d[j])));
          }
          thread_store(o + int64_t{row} * packs + i, normed);
        } else {
          thread_store(o + int64_t{row} * packs + i, got[r]);
        }
      }
    }
  }
  block_stamp(6);
}

// THE KERNELS, one per op, both the body above.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N, int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)
    all_reduce_pull_two_shot_rms_norm(const p2p::PeerPtrs* __restrict__ peer_inputs,
                                      p2p::PeerPtrs peer_scratch, p2p::PeerSignals peer_signals,
                                      p2p::Signal* self_signal, int rank, uint64_t timeout_ticks,
                                      DTYPE* __restrict__ out, const WEIGHT_DTYPE* __restrict__ weight, float eps,
                                      int rows, int packs) {
  all_reduce_pull_two_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, false, TILE_N, THREADS_PER_BLOCK>(
       peer_inputs, peer_scratch, peer_signals, self_signal, rank, timeout_ticks, out, nullptr,
      nullptr, weight, eps, rows, packs);
}

template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N, int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)
    all_reduce_pull_two_shot_add_rms_norm(const p2p::PeerPtrs* __restrict__ peer_inputs,
                                          p2p::PeerPtrs peer_scratch, p2p::PeerSignals peer_signals,
                                          p2p::Signal* self_signal, int rank,
                                          uint64_t timeout_ticks, DTYPE* __restrict__ out,
                                          DTYPE* __restrict__ residual_out,
                                          const DTYPE* __restrict__ residual,
                                          const WEIGHT_DTYPE* __restrict__ weight, float eps, int rows,
                                          int packs) {
  all_reduce_pull_two_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, true, TILE_N, THREADS_PER_BLOCK>(
       peer_inputs, peer_scratch, peer_signals, self_signal, rank, timeout_ticks, out, residual_out,
      residual, weight, eps, rows, packs);
}

}  // namespace hip_comms
