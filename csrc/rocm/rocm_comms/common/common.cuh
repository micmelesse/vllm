// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// COMMON, THE OPS' ONE INTERFACE: the only common/ header anything includes, and below, the whole
// of what a kernel may use. Every op is named for its scope (thread, wave, block, grid, peers), as
// hipCUB names its own; each prevents a footgun its comment names. Its parts refuse to be included
// any other way.
//
// utils.cuh         the helpers
//   traits<T>::V                 a pack: 16 bytes, the unit everything loads, sums and stores in
//   PeerPacks<T, ngpus, K>       every source's packs, in registers (peers_load's, peers_reduce's)
// tile.cuh          how a kernel cuts its work
//   Tile<R, K>, tile<R, K>(len, first)   a block's R rows, cut to len columns from first, and
//                                this thread's K packs of each, clamped into the slice
// memory.cuh        reads and writes
//   thread_load(p), thread_store(p, v)          one pack, global instructions
//   thread_load(row, f, out), thread_store(row, f, v)   a Tile row: every load issued, stores
//                                               only inside the row
//   thread_load_uncached(p), thread_store_uncached(p, v)   one pack past every cache (system
//                                               scope); no kernel uses them now
//   peers_load<T, ngpus>(read, i), peers_load<T, ngpus>(read, row, packs, f)   every source's
//                                               pack (or Tile row) in flight; waits at its use
// elementwise.cuh
//   thread_unpack(v, x), thread_pack(x)         a pack to fp32 and back, rounding once
// reduce.cuh
//   peers_reduce(packs) -> V, peers_reduce(packs, sum)   each pack summed over its sources, fp32,
//                                               source order, rounded once
//   wave_reduce<Op, N>(v), block_reduce<Op, N>(v)   N values at once; Op is Sum or Max
// dot.cuh
//   thread_dot(a, b, f)                         this thread's share of a row's dot
//   grid_gemm<kLanesPerCol, kAccumulate, T>(row, rows, w, n_cols, packs, out, stride)  the
//                                               skinny GEMM, written or accumulated,
//                                               with its geometry (kBuild.kernels.gemm_rows,
//                                               gemm_max_threads)
// softmax.cuh
//   OnlineSoftmax, thread_softmax_fold(s, logit, scale)   a softmax a tile of logits at a time,
//                                               folded into a running weighted sum

#pragma once

#define HIP_COMMS_COMMON_INTERFACE
#include "utils.cuh"
#include "tile.cuh"
#include "memory.cuh"
#include "elementwise.cuh"
#include "reduce.cuh"
#include "dot.cuh"
#include "softmax.cuh"
#undef HIP_COMMS_COMMON_INTERFACE
