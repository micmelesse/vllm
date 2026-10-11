// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce then RMSNorm (`all_reduce_pull_two_shot_rms_norm`), and
// all-reduce then add then RMSNorm (`all_reduce_pull_two_shot_add_rms_norm`): one body, a
// kernel per op, so a trace names the op that ran. THE SLICE IS ROWS, for large inputs: the
// column-slice kernel (all_reduce_push_two_shot_*) wins small ones and loses at prefill.

#pragma once

#include "../common/interface.cuh"

namespace hip_comms {

// THE SLICE IS ROWS: each rank owns whole rows, so it computes each row's norm alone. It reduces
// its rows and leaves one row each in its scratch, row-major: the normed row, or (kAdd) the new
// residual and the row's RMS scale; after the sync every rank copies every owner's rows out,
// (kAdd) norming them by their scales as it goes. Every rank does the same arithmetic on the same
// bytes, so every rank holds the same result. THE SAME BLOCK AND THREAD INDEX A PACK IN BOTH
// PHASES: after the sync a block may read only what the same block on a peer wrote.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, bool ADD_RESIDUAL, int TILE_N, int THREADS_PER_BLOCK>
DINLINE void all_reduce_pull_two_shot_add_rms_norm_body(
    const DTYPE* const* __restrict__ inp_ptrs, int64_t inp_stride_m, int64_t inp_stride_n,
    DTYPE* const* __restrict__ scratch_ptrs, int64_t scratch_stride_m, int64_t scratch_stride_n,
    Signal* const* __restrict__ signal_ptrs, Signal* self_signal_ptr, int rank,
    uint64_t timeout_ticks,
    Ptr<DTYPE> out, Ptr<DTYPE> residual_out, Ptr<const DTYPE> residual,
    Ptr<const WEIGHT_DTYPE> weight, float eps, int inp_size_m, int inp_size_n) {
  Sync<WORLD> sync{signal_ptrs, self_signal_ptr, rank, timeout_ticks};
  constexpr int NL       = traits<DTYPE>::N;
  using Row              = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  using RowF             = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK, float>;
  using Weight           = Tile<DTYPE, 1, TILE_N, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK, WEIGHT_DTYPE>;
  const float inv_hidden = 1.0f / static_cast<float>(inp_size_n);
  const int slice_m      = (inp_size_m + WORLD - 1) / WORLD;
  // ADD_RESIDUAL: each owned row's RMS scale, a float a row, after the rows in scratch.
  const int64_t scale_at = int64_t{slice_m} * scratch_stride_m;  // in elements
  const auto scales = [&](DTYPE* at) { return reinterpret_cast<float*>(at + scale_at); };

  // 1. Wait until every peer has launched, so its input is ready.
  block_stamp(0);
  barrier<Group::peers, Until::launched>(sync);
  block_stamp(1);

  // THE RANKS' POINTERS AFTER THE BARRIER here: held across it, the 8-pack build keeps 68 B of
  // scratch (the ISA gate, 2026-09-30).
  const auto inp = rank_ptrs<const DTYPE, WORLD>(inp_ptrs, inp_stride_m, inp_stride_n);
  const auto scratch = rank_ptrs<DTYPE, WORLD>(scratch_ptrs, scratch_stride_m, scratch_stride_n);
  const auto own_scratch  = rank_ptr<DTYPE, WORLD>(scratch_ptrs, rank, scratch_stride_m,
  scratch_stride_n);

  // 2. This rank's rows: read each from every rank in rank order and sum, then (ADD_RESIDUAL) add the
  //    residual, then RMSNorm, rounding as the reference does (the one-shot kernel spells it out),
  //    and leave the normed rows or (ADD_RESIDUAL) the residual rows and scales in this rank's scratch.
  //    PIPELINED: the next row's loads go out before this row's reduction and norm, so a block's
  //    compute runs under its next round trip instead of between them (a block had ~14 rows at
  //    4096 tokens, each 1.24 us of reduction and norm on the critical path: 2026-10-01T00-01-54Z).
  const int first = rank * slice_m;
  const int last  = min(first + slice_m, inp_size_m);
  // Row `row` from every rank.
  using Peers = Row[WORLD];
  const auto load = [&](int row, Peers& got) {
#pragma unroll
    for (int r = 0; r < WORLD; ++r) got[r] = Row{inp_size_m, inp_size_n, row, 0};
    tile_load(got, inp);
  };
  // THE WEIGHT ONCE, AND EVERY OTHER LOAD BEFORE THE NEXT ROW'S: loads complete in issue order, so
  // waiting on one issued after the peers' would wait on the peers' too.
  Weight w{1, inp_size_n, 0, 0};
  tile_load(w, weight);
  // ONE ROW: its residual, then the next row's peer loads into `next`, then this row's sum (its
  // wait covers only its own, older, loads), so the next round trip runs under the reduction and
  // norm. Its rows land in this rank's scratch at row - first.
  const auto one_row = [&](int row, const Peers& cur, Peers& next) {
    Row res{inp_size_m, inp_size_n, row, 0};
    if constexpr (ADD_RESIDUAL) tile_load(res, residual);
    // ONLY A ROW THAT EXISTS: issued here, never hoisted, so the block-uniform branch costs
    // nothing, where a clamped unconditional load re-read the last row (a block's whole round trip
    // again; at 256 tokens every block has one row: 2026-10-01T01-07-56Z).
    if (row + static_cast<int>(gridDim.x) < last) load(row + gridDim.x, next);
    RowF s = peers_reduce(cur).template to<float>();
    block_stamp(2);
    // The norm, rounding as the reference does (see the one-shot kernel):
    if constexpr (ADD_RESIDUAL) {
      s = tile_add(s, res.template to<float>());
      Row added = s.template to<DTYPE>();
      added.offs_m = row - first;
      tile_store(added, own_scratch);
    }
    float ss[1];
    partial_dot(s, s, ss);
    block_reduce<Sum, THREADS_PER_BLOCK>(ss);
    block_stamp(3);
    const float scale = rsqrtf(ss[0] * inv_hidden + eps);
    // ADD_RESIDUAL LEAVES THE NEW RESIDUAL AND ITS SCALE, NOT THE NORMED ROW: every rank norms it while
    // gathering, so the link carries one row a row, as a plain all-reduce does (gathering both the
    // normed row and the residual was twice that: 215.9 against 155.2 us at 4096 tokens,
    // 2026-10-01T02-17-54Z).
    if constexpr (ADD_RESIDUAL) {
      block_store_row_scalar(scales(own_scratch.data), row - first, scale);
      return;
    }
    // out = T(W(W(s * scale) * float(w))), as the reference rounds
    Row normed = tile_mul(tile_mul(s, scale).template to<WEIGHT_DTYPE>().template to<float>(),
                            w.template to<float>())
                     .template to<WEIGHT_DTYPE>()
                     .template to<DTYPE>();
    normed.offs_m = row - first;
    tile_store(normed, own_scratch);
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
  barrier<Group::peers, Until::visible>(sync);
  block_stamp(5);

  // 4. Every owner's rows out of its scratch, at their place in the output. The next call's
  //    first sync keeps a rank from overwriting its scratch while it is read. EVERY OWNER'S PACK
  //    LOADED BEFORE ANY IS STORED, and every load unconditional (each rank's scratch holds
  //    slice_m rows, so a slot past the last row is real): a store between two loads, or a
  //    load under an `if`, made the eight owners' round trips run one after another.
  //    ADD_RESIDUAL: the owners' rows are the new residual; each is normed here by its owner's
  //    scale (the weight loaded once, before the loop). Every thread loads the 8 scales after the 8
  //    tiles, all in flight before any wait (a wave's lanes read one scale address: one request a
  //    wave). Left to the compiler, the scales were loaded and waited on before the packs were
  //    issued, two round trips a row (ISA 2026-10-01T02-33-11Z).
  // A CHUNK A GROUP A THREAD, stepping across the row: every owner's chunk in flight is WORLD packs
  // a thread, as the pack-at-a-time copy was (a whole row's was 16 at 7168, and 2-3% slower).
  using Chunk  = Tile<DTYPE, 1, THREADS_PER_BLOCK * NL, 1, THREADS_PER_BLOCK, THREADS_PER_BLOCK>;
  using ChunkW = TileAs<Chunk, WEIGHT_DTYPE>;
  for (int l = blockIdx.x; l < slice_m; l += gridDim.x) {
    float sc[WORLD];
    if constexpr (ADD_RESIDUAL)
      peers_load_row_scalars<WORLD>([&](int r) { return scales(scratch[r].data); }, l, sc);
    for (int c = 0; c < inp_size_n; c += Chunk::kTileN) {
      Chunk got[WORLD];
#pragma unroll
      for (int r = 0; r < WORLD; ++r) got[r] = Chunk{slice_m, inp_size_n, l, c};
      tile_load(got, scratch);
      ChunkW wc{1, inp_size_n, 0, c};
      if constexpr (ADD_RESIDUAL) tile_load(wc, weight);
#pragma unroll
      for (int r = 0; r < WORLD; ++r) {
        const int row = r * slice_m + l;
        if (row >= inp_size_m) continue;
        Chunk at = got[r];
        at.M      = inp_size_m;
        at.offs_m = row;
        if constexpr (ADD_RESIDUAL) {
          tile_store(at, residual_out);
          const Chunk normed =
              tile_mul(tile_mul(at.template to<float>(), sc[r]).template to<WEIGHT_DTYPE>().template to<float>(),
                       wc.template to<float>())
                  .template to<WEIGHT_DTYPE>()
                  .template to<DTYPE>();
          tile_store(normed, out);
        } else {
          tile_store(at, out);
        }
      }
    }
  }
  block_stamp(6);
  sync.finish();
}

// THE KERNELS, one per op, both the body above.
template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot_rms_norm(const DTYPE* const* __restrict__ inp_ptrs,
                                      int64_t inp_stride_m, int64_t inp_stride_n,
                                      DTYPE* const* __restrict__ scratch_ptrs,
                                      int64_t scratch_stride_m,
                                      int64_t scratch_stride_n,
                                      Signal* const* __restrict__ signal_ptrs,
                                      Signal* self_signal_ptr, int rank, uint64_t timeout_ticks,
                                      DTYPE* __restrict__ out_ptr, int64_t out_stride_m,
                                      int64_t out_stride_n,
                                      const WEIGHT_DTYPE* __restrict__ weight_ptr,
                                      int64_t weight_stride_n, float eps, int inp_size_m,
                                      int inp_size_n) {
  all_reduce_pull_two_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, false, TILE_N, THREADS_PER_BLOCK>(
      inp_ptrs, inp_stride_m, inp_stride_n, scratch_ptrs, scratch_stride_m, scratch_stride_n,
      signal_ptrs, self_signal_ptr, rank, timeout_ticks,
      local_ptr(out_ptr, out_stride_m, out_stride_n, rank), Ptr<DTYPE>{nullptr, 0, 0, rank},
      Ptr<const DTYPE>{nullptr, 0, 0, rank}, local_ptr(weight_ptr, 0, weight_stride_n, rank),
      eps, inp_size_m, inp_size_n);
}

template <typename DTYPE, typename WEIGHT_DTYPE, int WORLD, int TILE_N,
          int THREADS_PER_BLOCK, int WAVES_PER_EU>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_pull_two_shot_add_rms_norm(const DTYPE* const* __restrict__ inp_ptrs,
                                          int64_t inp_stride_m, int64_t inp_stride_n,
                                          DTYPE* const* __restrict__ scratch_ptrs,
                                          int64_t scratch_stride_m,
                                          int64_t scratch_stride_n,
                                          Signal* const* __restrict__ signal_ptrs,
                                          Signal* self_signal_ptr, int rank, uint64_t timeout_ticks,
                                          DTYPE* __restrict__ out_ptr, int64_t out_stride_m,
                                          int64_t out_stride_n,
                                          DTYPE* __restrict__ residual_out_ptr,
                                          int64_t residual_out_stride_m,
                                          int64_t residual_out_stride_n,
                                          const DTYPE* __restrict__ residual_ptr,
                                          int64_t residual_stride_m, int64_t residual_stride_n,
                                          const WEIGHT_DTYPE* __restrict__ weight_ptr,
                                          int64_t weight_stride_n, float eps, int inp_size_m,
                                          int inp_size_n) {
  all_reduce_pull_two_shot_add_rms_norm_body<DTYPE, WEIGHT_DTYPE, WORLD, true, TILE_N, THREADS_PER_BLOCK>(
      inp_ptrs, inp_stride_m, inp_stride_n, scratch_ptrs, scratch_stride_m, scratch_stride_n,
      signal_ptrs, self_signal_ptr, rank, timeout_ticks,
      local_ptr(out_ptr, out_stride_m, out_stride_n, rank),
      local_ptr(residual_out_ptr, residual_out_stride_m, residual_out_stride_n, rank),
      local_ptr(residual_ptr, residual_stride_m, residual_stride_n, rank),
      local_ptr(weight_ptr, 0, weight_stride_n, rank), eps, inp_size_m, inp_size_n);
}

}  // namespace hip_comms
