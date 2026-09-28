// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Shared by the collectives: the 16-byte vector, global and uncached loads, block sums,
// and the push kernels' codec. What a fused op computes on the reduced rows is in
// fusions/.

#pragma once

#include <hip/hip_runtime.h>

#include <cmath>

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
// one push GROUP (ipc::Groups::row).
constexpr int kMaxRowPacks = kSumBatch;

// A group's kSumBatch packs as floats and back, rounding to T: what a push kernel reduces
// and encodes in, against the packs a row helper takes.
template <typename T>
DINLINE void floats_of(const typename traits<T>::V (&v)[kSumBatch],
                       float (&x)[kSumBatch * traits<T>::N]) {
  constexpr int N = traits<T>::N;
#pragma unroll
  for (int u = 0; u < kSumBatch; ++u)
#pragma unroll
    for (int j = 0; j < N; ++j) x[u * N + j] = static_cast<float>(v[u].d[j]);
}

template <typename T>
DINLINE void packs_of(const float (&x)[kSumBatch * traits<T>::N],
                      typename traits<T>::V (&v)[kSumBatch]) {
  constexpr int N = traits<T>::N;
#pragma unroll
  for (int u = 0; u < kSumBatch; ++u)
#pragma unroll
    for (int j = 0; j < N; ++j) v[u].d[j] = static_cast<T>(x[u * N + j]);
}

// ---------------------------------------------------------------------------------
// THE PUSH KERNELS' CODEC: what a group of kSumBatch packs (one thread's batch, 32 values
// of a 2-byte T) looks like on the wire. kBits 16 is T itself (no scale); 8 and 4 are
// QuickReduce's symmetric integers with one fp32 scale per group.
// ---------------------------------------------------------------------------------

template <typename T, int kBits>
struct Codec {
  using V                    = typename traits<T>::V;
  static constexpr int N     = traits<T>::N;
  static constexpr int kVals = kSumBatch * N;
  static_assert(sizeof(T) == 2, "the codec is built for 2-byte T");
  static_assert(kBits == 16 || kBits == 8 || kBits == 4, "16 (T), INT8 and INT4 are built");
  static constexpr bool kScaled = kBits < 16;
  // The payload of one group, in 16-byte packs: 32 values x kBits.
  static constexpr int kPayloadPacks = kVals * kBits / 8 / 16;
  static constexpr int kMax          = kScaled ? (1 << (kBits - 1)) - 1 : 0;

  // x -> payload; returns the scale, absmax / kMax (0 for an all-zero group; unused at 16).
  static DINLINE float encode(const float (&x)[kVals], V (&payload)[kPayloadPacks]) {
    if constexpr (!kScaled) {
#pragma unroll
      for (int i = 0; i < kVals; ++i) payload[i / N].d[i % N] = static_cast<T>(x[i]);
      return 1.0f;
    } else {
      float amax = 0.0f;
#pragma unroll
      for (int i = 0; i < kVals; ++i) amax = fmaxf(amax, fabsf(x[i]));
      const float scale = amax / kMax;
      const float inv   = amax > 0.0f ? kMax / amax : 0.0f;
      unsigned char bytes[kPayloadPacks * 16];
#pragma unroll
      for (int i = 0; i < kVals; ++i) {
        const int q =
            static_cast<int>(fminf(fmaxf(rintf(x[i] * inv), -kMax - 1.0f), kMax));
        if constexpr (kBits == 8) {
          bytes[i] = static_cast<unsigned char>(q & 0xFF);
        } else if (i % 2 == 0) {
          bytes[i / 2] = static_cast<unsigned char>(q & 0xF);
        } else {
          bytes[i / 2] |= static_cast<unsigned char>((q & 0xF) << 4);
        }
      }
      __builtin_memcpy(payload, bytes, sizeof(bytes));
      return scale;
    }
  }

  // payload, scale -> x.
  static DINLINE void decode(const V (&payload)[kPayloadPacks], float scale,
                             float (&x)[kVals]) {
    if constexpr (!kScaled) {
#pragma unroll
      for (int i = 0; i < kVals; ++i) x[i] = static_cast<float>(payload[i / N].d[i % N]);
    } else {
      unsigned char bytes[kPayloadPacks * 16];
      __builtin_memcpy(bytes, payload, sizeof(bytes));
#pragma unroll
      for (int i = 0; i < kVals; ++i) {
        int q;
        if constexpr (kBits == 8) {
          q = static_cast<signed char>(bytes[i]);
        } else {
          const int nib = (bytes[i / 2] >> (4 * (i % 2))) & 0xF;
          q             = nib >= 8 ? nib - 16 : nib;
        }
        x[i] = static_cast<float>(q) * scale;
      }
    }
  }
};

}  // namespace hip_comms
