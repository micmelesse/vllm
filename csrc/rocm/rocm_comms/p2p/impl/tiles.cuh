// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// The push kernels' tilings, behind p2p.cuh.

#pragma once

#ifndef HIP_COMMS_P2P_INTERFACE
#error "include p2p/p2p.cuh, p2p's one interface, not its parts"
#endif

#include "../../common/memory.cuh"
#include "../../hardware.cuh"

namespace hip_comms {

// =================================================================================
// TILINGS, FOR THE PUSH KERNELS ONLY (and the host's sizing of their scratch): who
// handles which packs. They go when push moves to p2p::simple. Two kinds, one interface.
// A thread's share of a UNIT is packs k < kPushGroupPacks (one batched load per peer, one push
// group). A two-shot gives every unit an OWNER rank; a rank's units
// are its LOCAL units 0, 1, ... . A kernel walks its units with
//   for (int u = t.first(rank); u < t.end(rank); u = t.next(u))   (a two-shot's own)
//   for (int u = t.first(); u < t.end(); u = t.next(u))           (every unit)
// and its locals likewise (first_local, locals, next_local). A LANE is a thread's place
// among the threads that share a unit: a slot holds, per local unit, kPushGroupPacks x lanes.
//
// Rows (the fused ops): a unit is a token's row of `packs` packs; block b takes rows b,
// b + grid, ...; thread t holds packs t + k x blockDim.
// Buffer (the plain all-reduce): `slices` slices (1: one-shot; the world: two-shot), a
// unit is one grid-wide pass over a slice: grid thread g holds packs g + (i x
// kPushGroupPacks + k) x (grid x threads) of it, i the pass. Every thread does every pass,
// so every thread moves the same bytes whatever the size.
// =================================================================================
namespace tiles {

struct Rows {
  int rows, packs, size, chunk, threads;

  __host__ __device__ int units() const { return rows; }
  __host__ __device__ int locals() const { return chunk; }
  __host__ __device__ int lanes() const { return threads; }
  DINLINE int first() const { return blockIdx.x; }
  DINLINE int end() const { return rows; }
  DINLINE int first(int owner) const { return owner * chunk + blockIdx.x; }
  DINLINE int end(int owner) const {
    return owner * chunk + chunk < rows ? owner * chunk + chunk : rows;
  }
  DINLINE int next(int u) const { return u + gridDim.x; }
  DINLINE int first_local() const { return blockIdx.x; }
  DINLINE int next_local(int l) const { return l + gridDim.x; }
  DINLINE int owner(int u) const { return u / chunk; }
  DINLINE int local(int u) const { return u - owner(u) * chunk; }
  DINLINE int unit(int owner, int l) const { return owner * chunk + l; }
  DINLINE int lane() const { return threadIdx.x; }
  // Pack k of this thread's share, within the row, and in the whole [rows, packs].
  DINLINE int pack(int k) const { return threadIdx.x + k * blockDim.x; }
  DINLINE int pos(int u, int k) const { return u * packs + pack(k); }
  DINLINE bool has(int u, int k) const { return u < rows && pack(k) < packs; }
};

struct Buffer {
  int size, slices, chunk, stride, iters;

  __host__ __device__ int units() const { return slices * iters; }
  __host__ __device__ int locals() const { return iters; }
  __host__ __device__ int lanes() const { return stride; }
  DINLINE int first() const { return 0; }
  DINLINE int end() const { return slices * iters; }
  DINLINE int first(int owner) const { return owner * iters; }
  DINLINE int end(int owner) const { return owner * iters + iters; }
  DINLINE int next(int u) const { return u + 1; }
  DINLINE int first_local() const { return 0; }
  DINLINE int next_local(int l) const { return l + 1; }
  DINLINE int owner(int u) const { return u / iters; }
  DINLINE int local(int u) const { return u - owner(u) * iters; }
  DINLINE int unit(int owner, int l) const { return owner * iters + l; }
  DINLINE int lane() const { return blockIdx.x * blockDim.x + threadIdx.x; }
  DINLINE int in_slice(int u, int k) const {
    return lane() + (local(u) * p2p::kPushGroupPacks + k) * stride;
  }
  DINLINE int pos(int u, int k) const { return owner(u) * chunk + in_slice(u, k); }
  DINLINE bool has(int u, int k) const {
    return in_slice(u, k) < chunk && pos(u, k) < size;
  }
};

__host__ __device__ inline Rows rows(int rows, int packs, int world, int threads) {
  return {rows, packs, rows * packs, (rows + world - 1) / world, threads};
}

__host__ __device__ inline Buffer buffer(int64_t size, int slices, int grid, int threads) {
  const int chunk  = static_cast<int>((size + slices - 1) / slices);
  const int stride = grid * threads;
  const int pass   = stride * p2p::kPushGroupPacks;
  return {static_cast<int>(size), slices, chunk, stride, (chunk + pass - 1) / pass};
}

// The same, at this launch's grid and block.
DINLINE Rows rows(int rows_, int packs, int world) {
  return rows(rows_, packs, world, blockDim.x);
}
DINLINE Buffer buffer(int size, int slices) {
  return buffer(size, slices, gridDim.x, blockDim.x);
}

// How many of this thread's packs of unit u exist (a prefix of k).
template <typename Tiling>
DINLINE int members(const Tiling& t, int u) {
  int n = 0;
#pragma unroll
  for (int k = 0; k < p2p::kPushGroupPacks; ++k) n += t.has(u, k) ? 1 : 0;
  return n;
}

// This thread's share of unit u into `dst`, a buffer laid out as the tiling's positions.
template <typename Tiling, typename V>
DINLINE void store(V* dst, const Tiling& t, int u, const V (&v)[p2p::kPushGroupPacks]) {
#pragma unroll
  for (int k = 0; k < p2p::kPushGroupPacks; ++k)
    if (t.has(u, k)) store_global(dst + t.pos(u, k), v[k]);
}

}  // namespace tiles

}  // namespace hip_comms
