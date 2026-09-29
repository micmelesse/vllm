// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Shared by the collectives and the fusions: the 16-byte vector, global and uncached
// loads, block sums, the hardware's sizes. The peer layer is p2p/, what a fused op
// computes is fusions/.

#pragma once

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#define DINLINE __device__ __forceinline__

namespace hip_comms {

// ---------------------------------------------------------------------------------
// The 16-byte vector every kernel moves.
// ---------------------------------------------------------------------------------

template <typename T, int N>
struct __align__(sizeof(T) * N) vec {
  T d[N];
};

template <typename T>
struct traits {
  static constexpr int N = 16 / sizeof(T);
  using V = vec<T, N>;
};

// PEER MEMORY THROUGH GLOBAL INSTRUCTIONS. A pointer read out of a struct has no address
// space the compiler can prove, so it emits `flat_load`, which checks the aperture and
// waits on both counters; casting to address space 1 gives `global_load_dwordx4` /
// `global_store_dwordx4`.
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(1))) u32x4 global_u32x4;

template <typename V>
DINLINE V load_global(const V* p) {
  static_assert(sizeof(V) == 16, "a pack is 16 bytes");
  const u32x4 raw = *(const global_u32x4*)(p);
  V v;
  __builtin_memcpy(&v, &raw, 16);
  return v;
}

template <typename V>
DINLINE void store_global(V* p, const V& v) {
  static_assert(sizeof(V) == 16, "a pack is 16 bytes");
  u32x4 raw;
  __builtin_memcpy(&raw, &v, 16);
  *(global_u32x4*)(p) = raw;
}

// WHAT A PEER PUSHED, read past every cache (system-scope loads, `sc0 sc1`), as
// QuickReduce reads what it receives: a peer's stores into this GPU's memory do not reach
// this GPU's L2, so a plain load could return a line cached before they landed.
template <typename V>
DINLINE V load_uncached(const V* p) {
  static_assert(sizeof(V) == 16, "a pack is 16 bytes");
  const auto* q     = reinterpret_cast<const uint64_t*>(p);
  const uint64_t raw[2] = {
      __scoped_atomic_load_n(q, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM),
      __scoped_atomic_load_n(q + 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM)};
  V v;
  __builtin_memcpy(&v, raw, 16);
  return v;
}

// WHAT A RANK PUSHES into a peer, stored past every cache (system-scope stores): the twin of
// `load_uncached`. A plain store to a peer's memory can be acknowledged before the peer can
// see it, so a wave's `s_waitcnt vmcnt(0)` before the barrier did not mean the data had
// landed, and the slowest rank's pushes arrived after its barrier flag (readers then read the
// previous call's values). A system-scope store completes only once it is visible there.
template <typename V>
DINLINE void store_uncached(V* p, const V& v) {
  static_assert(sizeof(V) == 16, "a pack is 16 bytes");
  uint64_t raw[2];
  __builtin_memcpy(raw, &v, 16);
  auto* q = reinterpret_cast<uint64_t*>(p);
  __scoped_atomic_store_n(q, raw[0], __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  __scoped_atomic_store_n(q + 1, raw[1], __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}

// PACKS PER PEER A THREAD HAS IN FLIGHT in a batched `sum`: bandwidth is bytes in flight
// over latency, and one pack per peer per wait left the one-shot at a quarter of aiter's.
constexpr int kSumBatch = 4;

// THE HARDWARE, named once: gfx9 runs 64-lane waves, and every kernel here is built for at
// most kMaxThreads per block (its __launch_bounds__), so a block holds at most kMaxWaves.
constexpr int kWaveSize   = 64;
constexpr int kMaxThreads = 512;
constexpr int kMaxWaves   = kMaxThreads / kWaveSize;

// Sum of `v` over the block. A block wider than kMaxThreads cannot be launched, so
// `partial` cannot be overrun.
DINLINE float block_sum(float v) {
  __shared__ float partial[kMaxWaves];
  __shared__ float total;
  const int lane = threadIdx.x % kWaveSize;
  const int warp = threadIdx.x / kWaveSize;
  for (int off = kWaveSize / 2; off > 0; off >>= 1) v += __shfl_down(v, off, kWaveSize);
  if (lane == 0) partial[warp] = v;
  __syncthreads();
  const int warps = (blockDim.x + kWaveSize - 1) / kWaveSize;
  if (warp == 0) {
    v = (lane < warps) ? partial[lane] : 0.0f;
    for (int off = kWaveSize / 2; off > 0; off >>= 1) v += __shfl_down(v, off, kWaveSize);
    if (lane == 0) total = v;
  }
  __syncthreads();
  return total;
}

// Two sums over the block in one pass: AttnRes needs a source's sum of squares and its
// weighted dot together, and one pass is half the barriers of two `block_sum`s.
DINLINE float2 block_sum2(float a, float b) {
  __shared__ float2 partial[kMaxWaves];
  __shared__ float2 total;
  const int lane = threadIdx.x % kWaveSize;
  const int warp = threadIdx.x / kWaveSize;
  for (int off = kWaveSize / 2; off > 0; off >>= 1) {
    a += __shfl_down(a, off, kWaveSize);
    b += __shfl_down(b, off, kWaveSize);
  }
  if (lane == 0) partial[warp] = make_float2(a, b);
  __syncthreads();
  const int warps = (blockDim.x + kWaveSize - 1) / kWaveSize;
  if (warp == 0) {
    float2 v = (lane < warps) ? partial[lane] : make_float2(0.0f, 0.0f);
    for (int off = kWaveSize / 2; off > 0; off >>= 1) {
      v.x += __shfl_down(v.x, off, kWaveSize);
      v.y += __shfl_down(v.y, off, kWaveSize);
    }
    if (lane == 0) total = v;
  }
  __syncthreads();
  return total;
}

// How many 16-byte packs of one row a thread holds in registers: packs threadIdx.x +
// k * blockDim.x, k < kMaxRowPacks. A row wider than kMaxRowPacks x blockDim is refused by
// the host. The same as kSumBatch, so a thread's share of a row is one batched `sum` and
// one push group.
constexpr int kMaxRowPacks = kSumBatch;

// =================================================================================
// TILINGS: who handles which packs. Two kinds, one interface, so every p2p phase is
// written once. A thread's share of a UNIT is packs k < kMaxRowPacks (one batched load
// per peer, one push group). A two-shot gives every unit an OWNER rank; a rank's units
// are its LOCAL units 0, 1, ... . A kernel walks its units with
//   for (int u = t.first(rank); u < t.end(rank); u = t.next(u))   (a two-shot's own)
//   for (int u = t.first(); u < t.end(); u = t.next(u))           (every unit)
// and its locals likewise (first_local, locals, next_local). A LANE is a thread's place
// among the threads that share a unit: a slot holds, per local unit, kMaxRowPacks x lanes.
//
// Rows (the fused ops): a unit is a token's row of `packs` packs; block b takes rows b,
// b + grid, ...; thread t holds packs t + k x blockDim.
// Buffer (the plain all-reduce): `slices` slices (1: one-shot; the world: two-shot), a
// unit is one grid-wide pass over a slice: grid thread g holds packs g + (i x
// kMaxRowPacks + k) x (grid x threads) of it, i the pass. Every thread does every pass,
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
    return lane() + (local(u) * kMaxRowPacks + k) * stride;
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
  const int pass   = stride * kMaxRowPacks;
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
  for (int k = 0; k < kMaxRowPacks; ++k) n += t.has(u, k) ? 1 : 0;
  return n;
}

// This thread's share of unit u into `dst`, a buffer laid out as the tiling's positions.
template <typename Tiling, typename V>
DINLINE void store(V* dst, const Tiling& t, int u, const V (&v)[kMaxRowPacks]) {
#pragma unroll
  for (int k = 0; k < kMaxRowPacks; ++k)
    if (t.has(u, k)) store_global(dst + t.pos(u, k), v[k]);
}

}  // namespace tiles

}  // namespace hip_comms
