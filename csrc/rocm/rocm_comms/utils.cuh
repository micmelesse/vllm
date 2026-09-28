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
// ROWS, THE ONE WORK DISTRIBUTION. Block b takes rows b, b + grid, ...; thread t of a
// block holds packs t + k x blockDim of a row, k < kMaxRowPacks. A fused op's rows are
// its tokens; a plain buffer is cut into rows sized to the launch (`buffer_rows`), its
// last row short. `size` is the packs that exist. A two-shot splits the rows into ranks:
// rank r owns rows [r x chunk, r x chunk + chunk).
// =================================================================================
namespace tiles {

struct Rows {
  int rows;
  int packs;
  int size;
};

__host__ __device__ inline Rows rows_of(int rows, int packs) {
  return {rows, packs, rows * packs};
}

// A buffer of `size` packs at `grid` x `threads`: rows `threads` x u packs wide, u the
// fewest packs a thread needs (at most kMaxRowPacks) for the grid to cover the buffer in
// one pass, so a small buffer still spreads over every block.
__host__ __device__ inline Rows buffer_rows(int64_t size, int grid, int threads) {
  const int64_t pass = int64_t{grid} * threads;
  int64_t u          = (size + pass - 1) / pass;
  u                  = u < 1 ? 1 : (u > kMaxRowPacks ? kMaxRowPacks : u);
  const int packs    = static_cast<int>(u) * threads;
  return {static_cast<int>((size + packs - 1) / packs), packs, static_cast<int>(size)};
}

DINLINE Rows buffer_rows(int size) { return buffer_rows(size, gridDim.x, blockDim.x); }

// The rows a rank owns in a two-shot.
__host__ __device__ inline int chunk_of(const Rows& r, int world) {
  return (r.rows + world - 1) / world;
}

struct Owned {
  int begin;
  int end;
};

__host__ __device__ inline Owned owned(const Rows& r, int rank, int world) {
  const int chunk = chunk_of(r, world);
  const int begin = rank * chunk;
  return {begin, begin + chunk < r.rows ? begin + chunk : r.rows};
}

// This thread's pack k of a row, within the row; whether it exists; how many do (the
// ones that exist are a prefix of k).
DINLINE int pack(int k) { return threadIdx.x + k * blockDim.x; }

DINLINE bool has(const Rows& r, int row, int k) {
  return pack(k) < r.packs && row * r.packs + pack(k) < r.size;
}

DINLINE int members(const Rows& r, int row) {
  int n = 0;
#pragma unroll
  for (int k = 0; k < kMaxRowPacks; ++k) n += has(r, row, k) ? 1 : 0;
  return n;
}

// This thread's share of `row` into `dst`, a [size]-pack buffer.
template <typename V>
DINLINE void store_row(V* dst, const Rows& r, int row, const V (&v)[kMaxRowPacks]) {
#pragma unroll
  for (int k = 0; k < kMaxRowPacks; ++k)
    if (has(r, row, k)) store_global(dst + row * r.packs + pack(k), v[k]);
}

}  // namespace tiles

}  // namespace hip_comms
