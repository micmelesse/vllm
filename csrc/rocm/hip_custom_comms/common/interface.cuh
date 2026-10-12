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
//   tile.cuh       Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK, ACC_DTYPE,
//                  TILE_R, THREADS_R> {M, N, offs_m, offs_n, row_step}: a chunk of an M x N
//                  tensor, the block's threads over it THREADS_M x THREADS_N, this thread's elements
//                  held as ACC_DTYPE; R x M x N with a copy a rank when TILE_R > 1 (a group of
//                  threads a rank, slice() this thread's); to<U>(), like<U>(), zeros<U>()
//   reduce.cuh     Sum, Max (a reduction's operation), Axis::m | n
//   softmax.cuh    OnlineSoftmax
//   peers.cuh      Signal, kMaxRanks, kMaxBlocks
//   barrier.cuh    Sync<WORLD>{signal_ptrs, self_signal_ptr, rank, timeout}: write_flag(peer, v),
//                  wait_flag(peer, v), finish() (the kernel's last statement); Group, Until

#pragma once

#define HIP_CUSTOM_COMMS_COMMON_INTERFACE
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
#undef HIP_CUSTOM_COMMS_COMMON_INTERFACE

namespace hip_custom_comms {

// ===============================================================================================
// TILES: WHICH TILES A BLOCK WORKS ON (tile.cuh)
// ===============================================================================================

// This block's tiles, grid-strided in row-major order over tiles_m x tiles_n tiles: body(row, col)
// for each, in tiles, without a division a tile.
template <typename BODY>
DINLINE void block_for_each_tile(int tiles_m, int tiles_n, BODY body) {
  impl::block_for_each_tile(tiles_m, tiles_n, body);
}

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

// A 3-D tile reduced over its ranks (Axis::r): the TILE_M x TILE_N fp32 tile, the first rank's
// threads'.
template <typename REDUCE_OP, Axis AXIS, typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N, int THREADS_PER_BLOCK, typename ACC_DTYPE, int TILE_R, int THREADS_R>
DINLINE Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK, float> block_reduce(
    const Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK, ACC_DTYPE, TILE_R, THREADS_R>& tile)
  requires(AXIS == Axis::r)
{
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
// POINTERS: A KERNEL'S ARGUMENTS AS Ptrs (ptr.cuh)
// ===============================================================================================

// ONE FORMAT (STANDARDS 3.2): a kernel's arguments are raw pointers, strides, sizes and scalars;
// a peer buffer is a device table of every rank's address (`x_ptrs`, as vLLM's RankData). Its
// first lines make them Ptrs, named for the buffer (`inp`, `scratch`, `out`), and its body
// touches memory only through them.
// THE BOUNDARY (ptr.cuh): every rank's buffer as a Ptr (rank r's at r), and a local one as one.
template <typename T, int WORLD>
DINLINE std::array<Ptr<T>, WORLD> rank_ptrs(T* const* ptrs, int64_t stride_m, int64_t stride_n) {
  return impl::rank_ptrs<T, WORLD>(ptrs, stride_m, stride_n);
}
template <typename T, int WORLD>
DINLINE Ptr<T> rank_ptr(T* const* ptrs, int rank, int64_t stride_m, int64_t stride_n) {
  return impl::rank_ptr<T, WORLD>(ptrs, rank, stride_m, stride_n);
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

// AN ELEMENTWISE [R, M, N] PROBLEM: R ranks' copies of an [m, n] buffer, rank r's at inp(r) (a Ptr
// at its strides). TILE is the [R, 1, TILE_N] chunk the walk moves (collectives.cuh: walked as
// [R, 1, m*n], each rank's share one contiguous range of the elements, for long runs).
// Reduce-scatter: this rank's share, summed over the ranks, into its scratch.
template <typename TILE, bool CONTIGUOUS, typename RANK_PTR, typename T>
DINLINE void reduce_scatter(int rank, RANK_PTR inp, int m, int n, const Ptr<T>& scratch) {
  impl::reduce_scatter<TILE, CONTIGUOUS>(rank, inp, m, n, scratch);
}
// All-gather: every rank's summed share, rank r's at scratch(r), at its place in `out`.
template <typename TILE, bool CONTIGUOUS, typename RANK_PTR, typename T>
DINLINE void all_gather(int rank, RANK_PTR scratch, const Ptr<T>& out, int m, int n) {
  impl::all_gather<TILE, CONTIGUOUS>(rank, scratch, out, m, n);
}
// Staged, first: this rank's input, every rank's share of it, into its staging.
template <typename TILE, bool CONTIGUOUS, typename T>
DINLINE void stage(int rank, const Ptr<const T>& inp, const Ptr<T>& staging, int m, int n) {
  impl::stage<TILE, CONTIGUOUS>(rank, inp, staging, m, n);
}

// ===============================================================================================
// UTILS (utils.cuh)
// ===============================================================================================

// A timestamp for phase `phase` of this block, in a stamps build only.
DINLINE void block_stamp(int phase) { impl::block_stamp(phase); }

}  // namespace hip_custom_comms
