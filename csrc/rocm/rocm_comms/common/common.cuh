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
//   Fragment<K>, fragment<K>(len)   this thread's K packs of a row a block owns, clamped into it
// memory.cuh        reads and writes
//   thread_load(p), thread_store(p, v)          one pack, global instructions
//   thread_load(row, f, out), thread_store(row, f, v)   a Fragment: every load issued, stores
//                                               only inside the row
//   thread_load_uncached(p), thread_store_uncached(p, v)   one pack past every cache (system
//                                               scope); for the first kernel that writes into a
//                                               peer, none yet
// elementwise.cuh
//   thread_unpack(v, x), thread_pack(x)         a pack to fp32 and back, rounding once
// reduce.cuh
//   peers_reduce<T, ngpus>(read, i)             one pack summed over the ranks, fp32, rank order
//   peers_reduce<T, ngpus>(read, row, packs, f, sum)   a Fragment of a row, the same
//   wave_reduce<Op, N>(v), block_reduce<Op, N>(v)   N values at once; Op is Sum or Max
// dot.cuh
//   thread_dot(a, b, f)                         this thread's share of a row's dot
//   grid_gemm<kLanesPerCol, T>(row, rows, w, n_cols, packs, out, stride, col0)   the skinny GEMM,
//                                               with its geometry (kGemmRows, gemm_max_threads)

#pragma once

#define HIP_COMMS_COMMON_INTERFACE
#include "utils.cuh"
#include "memory.cuh"
#include "elementwise.cuh"
#include "reduce.cuh"
#include "dot.cuh"
#undef HIP_COMMS_COMMON_INTERFACE
