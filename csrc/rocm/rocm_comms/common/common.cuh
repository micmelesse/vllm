// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// COMMON, THE OPS' ONE INTERFACE: the only common/ header anything includes, and below, the whole
// of what a kernel may use. An op on a tile is named for it (tile_load, tile_add), a partial
// result for being one (partial_dot), and an op that synchronizes or reaches other ranks for its
// scope (block_reduce, peers_load); each prevents a footgun its comment names. Its parts refuse to
// be included any other way.
//
// CONCURRENCY IS PART OF EACH OP'S CONTRACT, so a kernel keeps it by using them:
//   1. Global memory only through memory.cuh's tile loads and stores: a kernel never indexes a row.
//   2. A tile is loaded whole (tile_load, peers_load: one round trip) before anything is stored;
//      a pack loaded between stores waits a round trip, since a store may alias it.
//   3. A load that does not depend on a reduction is issued before it (weights, the next row or
//      source), so its round trip runs under the reduction.
//
// THE TILE IS THE INTERFACE: a kernel names tiles and what a thread holds of one; packs, rows and
// pack offsets are common's, and no kernel reaches them (dev/lint_kernels.sh).
//
// hardware.cuh      the device: Hardware (documented facts), the machine model over it (residency,
//                   registers, LDS), kDevice / kTarget, kWaveSize
// build.cuh         the build: kBuild, what is compiled for the device, fixed at compile time
// tile.cuh
//   Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK, ACC_DTYPE = DTYPE>{M, N, offs_m, offs_n,
//                         row_step = 1}   a chunk of an M x N tensor, the block's: TILE_M rows from
//                         offs_m (every row_step-th) x TILE_N columns from offs_n, the block's
//                         threads over it THREADS_M x THREADS_N (the layout), and this thread's
//                         elements of it held as ACC_DTYPE; to<U>() holds them as U, like<U>() is
//                         the same place empty, TileAs<TILE, U> its type
// memory.cuh
//   tile_load(tile, data, row_stride), tile_store(data, row_stride, tile)   one round trip a
//                         tile; a row past M reads the last, stores only rows below M
//   peers_load(tiles[ngpus], data(r), row_stride)   every rank's tile in flight together
//   sliced_load<ngpus>(tile, data(r), row_stride, slice)   each column from the rank owning it
//   tile_gather(tile, row_data(m) [, row_n(m)]), tile_scatter(row_data, row_n, tile)   a tile
//                         whose rows are tensors (a row a rank)
//   block_store_row_scalar, peers_load_row_scalars   a row's scalar (a norm's scale)
// elementwise.cuh
//   tile_add(a, b), tile_mul(a, b), tile_mul(a, scale or row_scale)   float tile math, b
//                         a's shape or one row
// reduce.cuh
//   peers_reduce(tiles[WORLD]) -> tile           summed in rank order in fp32, rounded once
//   block_reduce<Op, Axis::m>(tile) -> one row, block_reduce<Op, Axis::n>(tile, out[rows])   a
//                         tile reduced over its rows or each row over its columns
//   wave_reduce<Op, N>(v), block_reduce<Op, THREADS_PER_BLOCK>(v)   N values at once; Op is Sum or Max
// dot.cuh
//   partial_dot(a, b, d[TILE_M])                  this thread's share of each row's dot (b may be
//                                                one row, a weight)
//   grid_gemm<TILE_M, TILE_K, SLICE_K, ACCUMULATE, THREADS_PER_BLOCK>(x, x_stride, rows, cols, w, n_cols, out, stride)  the
//                                               skinny GEMM, written or accumulated,
//                                               with its tile (TILE_M rows, TILE_K, SLICE_K;
//                                               gemm_max_threads)
// softmax.cuh
//   OnlineSoftmax, thread_softmax_fold(s, logit, scale)   a softmax a tile of logits at a time,
//                                               folded into a running weighted sum
// peers.cuh         the ranks' memory, every rank's buffer as a tensor (r the same across the wave)
//   PeerPtrs, PeerSignals, Signal, kMaxRanks, kMaxBlocks   what a kernel takes them in
//   rank_input<DTYPE, WORLD>(p, r), rank_inputs<DTYPE, WORLD>(p)   a rank's input, read in place
//   rank_staging(p, r), rank_stagings(p)   a rank's staging: a staged kernel copies its input there
//   rank_scratch(p, r), rank_scratches(p)  a rank's scratch: a two-shot's partial sums
// barrier.cuh       the ranks' synchronization
//   Sync<WORLD> sync{p, self, rank, timeout}   a kernel's synchronization, one per kernel
//   barrier<Group, Until>(sync)   Group::peers (this block and the same block on every rank),
//                         grid (every block of this rank) or world; Until::launched (every peer's
//                         input is ready), visible (what was written before is seen after), read
//                         (every peer is done reading this rank)
//   sync.write_flag(peer, v), sync.wait_flag(peer, v)   v into `peer`'s slot for this rank; until
//                         it reaches v
//   sync.finish()         the block's sequence stored for the next call: the kernel's last statement
//   own_signals(p), signals(p, r), lane_signals(p, i)   a rank's Signal block as `Signals`, its
//                         counters (`Counter`) touched only atomically

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
#include "barrier.cuh"
#undef HIP_COMMS_COMMON_INTERFACE
