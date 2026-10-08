// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// COMMON'S INTERFACE: everything a kernel may call, each a small function that passes its arguments
// on to its implementation (impl::<name> in the file its section names), every template parameter
// by name, so what reaches each primitive is visible here. The only common/ header anything
// includes; its parts refuse to be included any other way, and a kernel never names impl::
// (dev/lint_kernels.sh).
//
// CONCURRENCY IS PART OF EACH PRIMITIVE'S CONTRACT, so a kernel keeps it by using them:
//   1. Global memory only through the tile loads and stores: a kernel never indexes a row.
//   2. A tile is loaded whole (tile_load, peers_load: one round trip) before anything is stored;
//      a pack loaded between stores waits a round trip, since a store may alias it.
//   3. A load that does not depend on a reduction is issued before it (weights, the next row or
//      source), so its round trip runs under the reduction.
//
// THE TYPES a kernel names, defined in their files (a type cannot be passed on):
//   hardware.cuh   Hardware, kTarget, kWaveSize: the device's documented facts
//   build.cuh      kBuild: what is compiled for the device
//   tile.cuh       Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK, ACC_DTYPE>
//                  {M, N, offs_m, offs_n, row_step}: a chunk of an M x N tensor, the block's
//                  threads over it THREADS_M x THREADS_N, this thread's elements held as ACC_DTYPE;
//                  to<U>(), like<U>(), zeros<U>()
//   reduce.cuh     Sum, Max (a reduction's operation), Axis::m | n
//   softmax.cuh    OnlineSoftmax
//   peers.cuh      PeerPtrs, PeerSignals, Signal, kMaxRanks, kMaxBlocks
//   barrier.cuh    Sync<WORLD>{peer_signals, self_signal, rank, timeout}: write_flag(peer, v),
//                  wait_flag(peer, v), finish() (the kernel's last statement); Group, Until

#pragma once

#define HIP_COMMS_COMMON_INTERFACE
#include "utils.cuh"
#include "tile.cuh"
#include "memory.cuh"
#include "elementwise.cuh"
#include "reduce.cuh"
#include "dot.cuh"
#include "softmax.cuh"
#include "peers.cuh"
#include "ptr.cuh"
#include "barrier.cuh"
#include "collectives.cuh"
#undef HIP_COMMS_COMMON_INTERFACE

namespace hip_comms {

// ===============================================================================================
// MEMORY: GLOBAL MEMORY ONLY THROUGH TILES (memory.cuh)
// ===============================================================================================

// One round trip a tile; a row past M reads the last row.
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N, int THREADS_PER_BLOCK, typename ACC_DTYPE>
DINLINE void tile_load(Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK, ACC_DTYPE>& tile, const ACC_DTYPE* data, int64_t row_stride) {
  impl::tile_load(tile, data, row_stride);
}

// Stores only the rows below M and the columns below N.
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N, int THREADS_PER_BLOCK, typename ACC_DTYPE>
DINLINE void tile_store(ACC_DTYPE* data, int64_t row_stride, const Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK, ACC_DTYPE>& tile) {
  impl::tile_store(data, row_stride, tile);
}

// Every rank's tile in flight together: rank r's tensor at rank_data(r).
// TILE FIRST, POINTER SECOND: a tile from a Ptr, a tile to a Ptr; a group of tiles from a group of
// Ptrs (every load issued together, each pack's address computed once; the group shares a stride).
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N, int THREADS_PER_BLOCK, typename ACC_DTYPE, typename T>
DINLINE void tile_load(Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK, ACC_DTYPE>& tile, const Ptr<T>& ptr) {
  impl::tile_load(tile, ptr.data, ptr.stride_m);
}
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N, int THREADS_PER_BLOCK, typename ACC_DTYPE>
DINLINE void tile_store(const Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK, ACC_DTYPE>& tile, const Ptr<ACC_DTYPE>& ptr) {
  impl::tile_store(ptr.data, ptr.stride_m, tile);
}
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N, int THREADS_PER_BLOCK, typename ACC_DTYPE, std::size_t N, typename T>
DINLINE void tile_load(Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK, ACC_DTYPE> (&tiles)[N], const std::array<Ptr<T>, N>& ptrs) {
  impl::peers_load(tiles, [&](int r) { return ptrs[r].data; }, ptrs[0].stride_m);
}
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N, int THREADS_PER_BLOCK, typename ACC_DTYPE, int WORLD, typename RANK_DATA>
DINLINE void peers_load(Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK, ACC_DTYPE> (&tiles)[WORLD], RANK_DATA rank_data, int64_t row_stride) {
  impl::peers_load(tiles, rank_data, row_stride);
}

// Each column from the rank owning it, `slice` columns a rank (the owner picked in each lane).
template <int WORLD, typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N, int THREADS_PER_BLOCK, typename ACC_DTYPE, typename RANK_DATA>
DINLINE void sliced_load(Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK, ACC_DTYPE>& tile, RANK_DATA rank_data, int64_t row_stride, int slice) {
  impl::sliced_load<WORLD>(tile, rank_data, row_stride, slice);
}

// A tile whose rows are different tensors (a row a rank): row m at row_data(m).
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N, int THREADS_PER_BLOCK, typename ACC_DTYPE, typename ROW_DATA>
DINLINE void tile_gather(Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK, ACC_DTYPE>& tile, ROW_DATA row_data) {
  impl::tile_gather(tile, row_data);
}

// As above, row m only to its own row_n(m) columns (a short last slice).
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N, int THREADS_PER_BLOCK, typename ACC_DTYPE, typename ROW_DATA, typename ROW_N>
DINLINE void tile_gather(Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK, ACC_DTYPE>& tile, ROW_DATA row_data, ROW_N row_n) {
  impl::tile_gather(tile, row_data, row_n);
}

// Row m of the tile into row_data(m), up to its row_n(m) columns.
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N, int THREADS_PER_BLOCK, typename ACC_DTYPE, typename ROW_DATA, typename ROW_N>
DINLINE void tile_scatter(ROW_DATA row_data, ROW_N row_n, const Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK, ACC_DTYPE>& tile) {
  impl::tile_scatter(row_data, row_n, tile);
}

// A row's scalar (a norm's scale), stored once a block.
DINLINE void block_store_row_scalar(float* scalars, int row, float value) {
  impl::block_store_row_scalar(scalars, row, value);
}

// Every rank's scalar for `row`.
template <int WORLD, typename RANK_DATA>
DINLINE void peers_load_row_scalars(RANK_DATA rank_data, int row, float (&out)[WORLD]) {
  impl::peers_load_row_scalars<WORLD>(rank_data, row, out);
}

// ===============================================================================================
// ELEMENTWISE: FLOAT TILE MATH (elementwise.cuh)
// ===============================================================================================

// a + b, b of a's shape or one row (a weight).
template <typename A_DTYPE, int A_TILE_M, int A_TILE_N, int A_THREADS_M, int A_THREADS_N, int A_THREADS_PER_BLOCK, typename A_ACC_DTYPE, typename B_DTYPE, int B_TILE_M, int B_TILE_N, int B_THREADS_M, int B_THREADS_N, int B_THREADS_PER_BLOCK, typename B_ACC_DTYPE>
DINLINE Tile<A_DTYPE, A_TILE_M, A_TILE_N, A_THREADS_M, A_THREADS_N, A_THREADS_PER_BLOCK, A_ACC_DTYPE> tile_add(const Tile<A_DTYPE, A_TILE_M, A_TILE_N, A_THREADS_M, A_THREADS_N, A_THREADS_PER_BLOCK, A_ACC_DTYPE>& a, const Tile<B_DTYPE, B_TILE_M, B_TILE_N, B_THREADS_M, B_THREADS_N, B_THREADS_PER_BLOCK, B_ACC_DTYPE>& b) {
  return impl::tile_add(a, b);
}

// a * b, b of a's shape or one row (a weight).
template <typename A_DTYPE, int A_TILE_M, int A_TILE_N, int A_THREADS_M, int A_THREADS_N, int A_THREADS_PER_BLOCK, typename A_ACC_DTYPE, typename B_DTYPE, int B_TILE_M, int B_TILE_N, int B_THREADS_M, int B_THREADS_N, int B_THREADS_PER_BLOCK, typename B_ACC_DTYPE>
DINLINE Tile<A_DTYPE, A_TILE_M, A_TILE_N, A_THREADS_M, A_THREADS_N, A_THREADS_PER_BLOCK, A_ACC_DTYPE> tile_mul(const Tile<A_DTYPE, A_TILE_M, A_TILE_N, A_THREADS_M, A_THREADS_N, A_THREADS_PER_BLOCK, A_ACC_DTYPE>& a, const Tile<B_DTYPE, B_TILE_M, B_TILE_N, B_THREADS_M, B_THREADS_N, B_THREADS_PER_BLOCK, B_ACC_DTYPE>& b) {
  return impl::tile_mul(a, b);
}

// Each row of a times its scale.
template <typename A_DTYPE, int A_TILE_M, int A_TILE_N, int A_THREADS_M, int A_THREADS_N, int A_THREADS_PER_BLOCK, typename A_ACC_DTYPE>
DINLINE Tile<A_DTYPE, A_TILE_M, A_TILE_N, A_THREADS_M, A_THREADS_N, A_THREADS_PER_BLOCK, A_ACC_DTYPE> tile_mul(const Tile<A_DTYPE, A_TILE_M, A_TILE_N, A_THREADS_M, A_THREADS_N, A_THREADS_PER_BLOCK, A_ACC_DTYPE>& a, const float (&row_scale)[A_TILE_M / A_THREADS_M]) {
  return impl::tile_mul(a, row_scale);
}

// Every element of a times one scale.
template <typename A_DTYPE, int A_TILE_M, int A_TILE_N, int A_THREADS_M, int A_THREADS_N, int A_THREADS_PER_BLOCK, typename A_ACC_DTYPE>
DINLINE Tile<A_DTYPE, A_TILE_M, A_TILE_N, A_THREADS_M, A_THREADS_N, A_THREADS_PER_BLOCK, A_ACC_DTYPE> tile_mul(const Tile<A_DTYPE, A_TILE_M, A_TILE_N, A_THREADS_M, A_THREADS_N, A_THREADS_PER_BLOCK, A_ACC_DTYPE>& a, float scale) {
  return impl::tile_mul(a, scale);
}

// a * a_row_scale + b * b_row_scale, a row's scales each.
template <typename A_DTYPE, int A_TILE_M, int A_TILE_N, int A_THREADS_M, int A_THREADS_N, int A_THREADS_PER_BLOCK, typename A_ACC_DTYPE, typename B_DTYPE, int B_TILE_M, int B_TILE_N, int B_THREADS_M, int B_THREADS_N, int B_THREADS_PER_BLOCK, typename B_ACC_DTYPE>
DINLINE Tile<A_DTYPE, A_TILE_M, A_TILE_N, A_THREADS_M, A_THREADS_N, A_THREADS_PER_BLOCK, A_ACC_DTYPE> tile_fma(const Tile<A_DTYPE, A_TILE_M, A_TILE_N, A_THREADS_M, A_THREADS_N, A_THREADS_PER_BLOCK, A_ACC_DTYPE>& a, const float (&a_row_scale)[A_TILE_M / A_THREADS_M],
                        const Tile<B_DTYPE, B_TILE_M, B_TILE_N, B_THREADS_M, B_THREADS_N, B_THREADS_PER_BLOCK, B_ACC_DTYPE>& b, const float (&b_row_scale)[A_TILE_M / A_THREADS_M]) {
  return impl::tile_fma(a, a_row_scale, b, b_row_scale);
}

// ===============================================================================================
// REDUCE: OVER THE RANKS, A WAVE OR THE BLOCK (reduce.cuh)
// ===============================================================================================

// Every rank's tile summed, in rank order in fp32, rounded once.
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N, int THREADS_PER_BLOCK, typename ACC_DTYPE, int WORLD>
DINLINE Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK, ACC_DTYPE> peers_reduce(const Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK, ACC_DTYPE> (&tiles)[WORLD]) {
  return impl::peers_reduce(tiles);
}

// NUM_VALUES values over the wave, every lane left holding the results.
template <typename REDUCE_OP, int NUM_VALUES>
DINLINE void wave_reduce(float (&values)[NUM_VALUES]) {
  impl::wave_reduce<REDUCE_OP, NUM_VALUES>(values);
}

// NUM_VALUES values over the block, every thread left holding the results.
template <typename REDUCE_OP, int THREADS_PER_BLOCK, int NUM_VALUES>
DINLINE void block_reduce(float (&values)[NUM_VALUES]) {
  impl::block_reduce<REDUCE_OP, THREADS_PER_BLOCK, NUM_VALUES>(values);
}

// A tile reduced over its rows (Axis::m): one fp32 row, the first THREADS_N threads'.
template <typename REDUCE_OP, Axis AXIS, typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N, int THREADS_PER_BLOCK, typename ACC_DTYPE>
DINLINE Tile<DTYPE, 1, TILE_N, 1, THREADS_N, THREADS_PER_BLOCK, float> block_reduce(
    const Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK, ACC_DTYPE>& tile) {
  return impl::block_reduce<REDUCE_OP, AXIS>(tile);
}

// Each row of a tile reduced over its columns (Axis::n), into out[row].
template <typename REDUCE_OP, Axis AXIS, typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N, int THREADS_PER_BLOCK, typename ACC_DTYPE>
DINLINE void block_reduce(const Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK, ACC_DTYPE>& tile, float (&out)[TILE_M / THREADS_M]) {
  impl::block_reduce<REDUCE_OP, AXIS>(tile, out);
}

// ===============================================================================================
// DOT (dot.cuh)
// ===============================================================================================

// This thread's share of each row's dot of a and b (b of a's shape or one row, a weight).
template <typename A_DTYPE, int A_TILE_M, int A_TILE_N, int A_THREADS_M, int A_THREADS_N, int A_THREADS_PER_BLOCK, typename A_ACC_DTYPE, typename B_DTYPE, int B_TILE_M, int B_TILE_N, int B_THREADS_M, int B_THREADS_N, int B_THREADS_PER_BLOCK, typename B_ACC_DTYPE>
DINLINE void partial_dot(const Tile<A_DTYPE, A_TILE_M, A_TILE_N, A_THREADS_M, A_THREADS_N, A_THREADS_PER_BLOCK, A_ACC_DTYPE>& a, const Tile<B_DTYPE, B_TILE_M, B_TILE_N, B_THREADS_M, B_THREADS_N, B_THREADS_PER_BLOCK, B_ACC_DTYPE>& b, float (&d)[A_TILE_M / A_THREADS_M]) {
  impl::partial_dot(a, b, d);
}

// The skinny GEMM, out (or out +=, ACCUMULATE) = x[rows, cols] . w[n_cols, cols]^T, the grid
// striding over tiles of columns.
template <int TILE_M, int TILE_K, int SLICE_K, bool ACCUMULATE, int THREADS_PER_BLOCK,
          typename DTYPE>
DINLINE void grid_gemm(const DTYPE* x, int64_t x_stride, int rows, int cols,
                       const DTYPE* __restrict__ gemm_w, int64_t gemm_w_stride, int n_cols,
                       DTYPE* __restrict__ out, int64_t out_stride) {
  impl::grid_gemm<TILE_M, TILE_K, SLICE_K, ACCUMULATE, THREADS_PER_BLOCK, DTYPE>(
      x, x_stride, rows, cols, gemm_w, gemm_w_stride, n_cols, out, out_stride);
}

// ===============================================================================================
// SOFTMAX (softmax.cuh)
// ===============================================================================================

// A tile of logits folded into a running softmax: each logit's scale, and the old sum's.
template <int NUM_LOGITS>
DINLINE float thread_softmax_fold(OnlineSoftmax& softmax, const float (&logit)[NUM_LOGITS],
                                  float (&scale)[NUM_LOGITS]) {
  return impl::thread_softmax_fold<NUM_LOGITS>(softmax, logit, scale);
}

// ===============================================================================================
// PEERS: THE RANKS' BUFFERS, RANK R THE SAME ACROSS THE WAVE (peers.cuh)
// ===============================================================================================

template <typename DTYPE, int WORLD>
DINLINE const DTYPE* rank_input(const PeerPtrs& peer_ptrs, int rank) {
  return impl::rank_input<DTYPE, WORLD>(peer_ptrs, rank);
}
template <typename DTYPE, int WORLD>
DINLINE DTYPE* rank_staging(const PeerPtrs& peer_ptrs, int rank) {
  return impl::rank_staging<DTYPE, WORLD>(peer_ptrs, rank);
}
template <typename DTYPE, int WORLD>
DINLINE DTYPE* rank_scratch(const PeerPtrs& peer_ptrs, int rank) {
  return impl::rank_scratch<DTYPE, WORLD>(peer_ptrs, rank);
}
template <typename DTYPE, int WORLD>
DINLINE std::array<const DTYPE*, WORLD> rank_inputs(const PeerPtrs& peer_ptrs) {
  return impl::rank_inputs<DTYPE, WORLD>(peer_ptrs);
}
template <typename DTYPE, int WORLD>
DINLINE std::array<DTYPE*, WORLD> rank_stagings(const PeerPtrs& peer_ptrs) {
  return impl::rank_stagings<DTYPE, WORLD>(peer_ptrs);
}
template <typename DTYPE, int WORLD>
DINLINE std::array<DTYPE*, WORLD> rank_scratches(const PeerPtrs& peer_ptrs) {
  return impl::rank_scratches<DTYPE, WORLD>(peer_ptrs);
}
// THE BOUNDARY, a kernel's first lines (ptr.cuh): every rank's buffer as a Ptr (rank r's at r),
// and a local tensor as one.
template <typename T, int WORLD>
DINLINE std::array<Ptr<T>, WORLD> rank_ptrs(const PeerPtrs& peer_ptrs, int64_t stride_m, int64_t stride_n) {
  return impl::rank_ptrs<T, WORLD>(peer_ptrs, stride_m, stride_n);
}
template <typename T, int WORLD>
DINLINE Ptr<T> rank_ptr(const PeerPtrs& peer_ptrs, int rank, int64_t stride_m, int64_t stride_n) {
  return impl::rank_ptr<T, WORLD>(peer_ptrs, rank, stride_m, stride_n);
}
template <typename T>
DINLINE Ptr<T> local_ptr(T* data, int64_t stride_m, int64_t stride_n, int rank) {
  return impl::local_ptr<T>(data, stride_m, stride_n, rank);
}

// ===============================================================================================
// BARRIER: THE RANKS' SYNCHRONIZATION (barrier.cuh)
// ===============================================================================================

// GROUP peers (this block and the same block on every rank), grid (every block of this rank) or
// world; UNTIL launched (every peer's input is ready), visible (what was written before is seen
// after) or read (every peer is done reading this rank).
template <Group GROUP, Until UNTIL, int WORLD>
DINLINE void barrier(Sync<WORLD>& sync) {
  impl::barrier<GROUP, UNTIL, WORLD>(sync);
}

// ===============================================================================================
// COLLECTIVES: THE TWO-SHOT'S STEPS, ONE TESTED WAY EACH (collectives.cuh)
// ===============================================================================================

// A chunk of this rank's slice summed over the ranks into sum_out: rank w's copy at rank_chunk(w).
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N, int THREADS_PER_BLOCK, typename ACC_DTYPE, typename RANK_CHUNK>
DINLINE void reduce_scatter(Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK, ACC_DTYPE>& chunk, RANK_CHUNK rank_chunk, DTYPE* sum_out) {
  impl::reduce_scatter(chunk, rank_chunk, sum_out);
}

// Every rank's summed chunk into the output: rank w's at rank_chunk(w), its rows to out_chunk(m)
// up to out_len(m) columns.
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N, int THREADS_PER_BLOCK, typename ACC_DTYPE, typename RANK_CHUNK, typename OUT_CHUNK, typename OUT_LEN>
DINLINE void all_gather(Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK, ACC_DTYPE>& chunk, RANK_CHUNK rank_chunk, OUT_CHUNK out_chunk,
                        OUT_LEN out_len) {
  impl::all_gather(chunk, rank_chunk, out_chunk, out_len);
}

// ===============================================================================================
// UTILS (utils.cuh)
// ===============================================================================================

// A timestamp for phase `phase` of this block, in a stamps build only.
DINLINE void block_stamp(int phase) { impl::block_stamp(phase); }

}  // namespace hip_comms
