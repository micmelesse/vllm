// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Shared by the collectives: the 16-byte vector, block sums, the rows of all-reduce +
// RMSNorm and + AttnRes, and the latent MoE tail's GEMM phase.

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

// How many 16-byte packs of one row a thread holds in registers: packs threadIdx.x +
// k * blockDim.x, k < kMaxRowPacks. A row wider than kMaxRowPacks x blockDim is refused by
// the host. The same as kSumBatch, so a thread's share of a row is one batched `sum` and
// one push GROUP (ipc::Groups::row).
constexpr int kMaxRowPacks = kSumBatch;

// ONE ROW of all-reduce, then (kAdd) add, then RMSNorm by the whole block, matching vLLM's
// reference ops (`vllm/ir/ops/layernorm.py`) rounding for rounding:
//
//   s   = float(T(sum over ranks))                  the all-reduce output, as it would land
//   s  += float(residual); res_dst = T(s)           kAdd only: fused_add_rms_norm
//   x   = W(s * rsqrt(mean(s^2) + eps))             `x.to(weight.dtype)`
//   out = T(W(x * float(w)))                        the product in W, then to the input's T
//
// THE WEIGHT KEEPS ITS OWN DTYPE W, as the reference does: it rounds to the WEIGHT's dtype,
// so an fp32 weight multiplies an unrounded x, and casting it to T first would be a
// different op. With W == T this is the one rounding it always was.
//
// The variance is taken from `s` before any further rounding, and `s` stays in registers
// between the two passes, so nothing is read back.
//
// DIRECTION-FREE: `sum` is this thread's share of the row already reduced over ranks and
// rounded to T (pulled by `Comm::sum_row`, or out of a push kernel's inbox), sum[k] the
// pack threadIdx.x + k * blockDim.x. The results leave through `store_res(k, i, v)` and
// `store_out(k, i, v)`, i the pack within the row, so a kernel can land them in its
// output, in scratch, or in registers to push.
template <typename T, typename W, bool kAdd, typename StoreRes, typename StoreOut>
DINLINE void add_rms_norm_row(const typename traits<T>::V (&sum)[kMaxRowPacks],
                              const typename traits<T>::V* residual,
                              const vec<W, traits<T>::N>* weight, int row, int packs,
                              float inv_hidden, float eps, StoreRes store_res,
                              StoreOut store_out) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  const int base   = row * packs;
  float s[kMaxRowPacks][NL];
  float acc = 0.0f;
#pragma unroll
  for (int k = 0; k < kMaxRowPacks; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    if (i >= packs) break;
#pragma unroll
    for (int j = 0; j < NL; ++j) s[k][j] = static_cast<float>(sum[k].d[j]);
    if constexpr (kAdd) {
      const V r = residual[base + i];
      V rounded;
#pragma unroll
      for (int j = 0; j < NL; ++j) {
        s[k][j] += static_cast<float>(r.d[j]);
        rounded.d[j] = static_cast<T>(s[k][j]);
      }
      store_res(k, i, rounded);
    }
#pragma unroll
    for (int j = 0; j < NL; ++j) acc += s[k][j] * s[k][j];
  }
  const float scale = rsqrtf(block_sum(acc) * inv_hidden + eps);
#pragma unroll
  for (int k = 0; k < kMaxRowPacks; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    if (i >= packs) break;
    const vec<W, NL> w = weight[i];
    V o;
#pragma unroll
    for (int j = 0; j < NL; ++j) {
      const float x  = static_cast<float>(static_cast<W>(s[k][j] * scale));
      const float xw = static_cast<float>(static_cast<W>(x * static_cast<float>(w.d[j])));
      o.d[j]         = static_cast<T>(xw);
    }
    store_out(k, i, o);
  }
  // Before the next row reuses `block_sum`'s shared slots.
  __syncthreads();
}

// ---------------------------------------------------------------------------------
// One row of all-reduce + Kimi-K3's AttnRes + its RMSNorm (the one-shot and two-shot
// AttnRes kernels).
// ---------------------------------------------------------------------------------

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

// ONE ROW of all-reduce + AttnRes by the whole block, matching
// `vllm/models/kimi_k3/amd/ops/attn_res.py` rounding for rounding:
//
//   d   = float(T(sum over ranks))                   the all-reduce output, as it lands
//   u   = kPrefix ? float(T(float(prefix) + d)) : d  the running prefix, updated or started
//   prefix_out = T(u); blocks[write] = T(u)          the new prefix, and the block written
//   per source s (the stored blocks, then u):
//       logit_s = dot(s, norm_w * qk_w) * rsqrt(mean(s^2) + eps)
//   m   = softmax(logit) . sources                     online, one source at a time
//   out = T(m), or T(m * rsqrt(mean(m^2) + out_eps) * out_norm_w)
//
// The prefix and the mix stay in registers across the sources, so only the stored blocks
// are read back. The sums run in a different order than Triton's, so a result agrees to
// the rounding of the last few bits, not bitwise. Direction-free as `add_rms_norm_row`:
// `sum` is this thread's share of the reduced row, the new prefix leaves through
// `store_prefix(k, i, v)` and the output through `store_out(k, i, v)`.
template <typename T, bool kPrefix, typename StorePrefix, typename StoreOut>
DINLINE void add_attn_res_rms_norm_row(const typename traits<T>::V (&sum)[kMaxRowPacks],
                                       const typename traits<T>::V* prefix,
                                       const T* blocks, int64_t block_stride_r,
                                       const typename traits<T>::V* norm_w,
                                       const typename traits<T>::V* qk_w,
                                       const typename traits<T>::V* out_norm_w,
                                       int num_blocks, int row, int packs,
                                       float inv_hidden, float eps, float out_eps,
                                       StorePrefix store_prefix, StoreOut store_out) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  const int base   = row * packs;
  float u[kMaxRowPacks][NL];
#pragma unroll
  for (int k = 0; k < kMaxRowPacks; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    if (i >= packs) break;
    V rounded;
    if constexpr (kPrefix) {
      const V p = prefix[base + i];
#pragma unroll
      for (int j = 0; j < NL; ++j) {
        rounded.d[j] = static_cast<T>(static_cast<float>(p.d[j]) +
                                      static_cast<float>(sum[k].d[j]));
        u[k][j]      = static_cast<float>(rounded.d[j]);
      }
    } else {
      rounded = sum[k];
#pragma unroll
      for (int j = 0; j < NL; ++j) u[k][j] = static_cast<float>(sum[k].d[j]);
    }
    store_prefix(k, i, rounded);
  }

  float m[kMaxRowPacks][NL];
  if (num_blocks == 0) {
    // With only the prefix source, the softmax is exactly one.
#pragma unroll
    for (int k = 0; k < kMaxRowPacks; ++k)
#pragma unroll
      for (int j = 0; j < NL; ++j) m[k][j] = u[k][j];
  } else {
    float w[kMaxRowPacks][NL];
#pragma unroll
    for (int k = 0; k < kMaxRowPacks; ++k) {
      const int i = threadIdx.x + k * blockDim.x;
#pragma unroll
      for (int j = 0; j < NL; ++j) m[k][j] = 0.0f;
      if (i >= packs) continue;
      const V a = norm_w[i], b = qk_w[i];
#pragma unroll
      for (int j = 0; j < NL; ++j)
        w[k][j] = static_cast<float>(a.d[j]) * static_cast<float>(b.d[j]);
    }
    float max_logit = -INFINITY, denominator = 0.0f;
    for (int s = 0; s <= num_blocks; ++s) {
      // The stored blocks first, the prefix last, as the reference orders its sources.
      const V* src = reinterpret_cast<const V*>(blocks + s * block_stride_r);
      float v[kMaxRowPacks][NL];
      float ss = 0.0f, dot = 0.0f;
#pragma unroll
      for (int k = 0; k < kMaxRowPacks; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i >= packs) break;
        if (s < num_blocks) {
          const V x = src[i];
#pragma unroll
          for (int j = 0; j < NL; ++j) v[k][j] = static_cast<float>(x.d[j]);
        } else {
#pragma unroll
          for (int j = 0; j < NL; ++j) v[k][j] = u[k][j];
        }
#pragma unroll
        for (int j = 0; j < NL; ++j) {
          ss += v[k][j] * v[k][j];
          dot += v[k][j] * w[k][j];
        }
      }
      const float2 sums       = block_sum2(ss, dot);
      const float logit       = sums.y * rsqrtf(sums.x * inv_hidden + eps);
      const float new_max     = fmaxf(max_logit, logit);
      const float old_scale   = __expf(max_logit - new_max);
      const float this_scale  = __expf(logit - new_max);
      denominator             = denominator * old_scale + this_scale;
      max_logit               = new_max;
#pragma unroll
      for (int k = 0; k < kMaxRowPacks; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i >= packs) break;
#pragma unroll
        for (int j = 0; j < NL; ++j) m[k][j] = m[k][j] * old_scale + this_scale * v[k][j];
      }
    }
    const float inv_den = 1.0f / denominator;
#pragma unroll
    for (int k = 0; k < kMaxRowPacks; ++k)
#pragma unroll
      for (int j = 0; j < NL; ++j) m[k][j] *= inv_den;
  }

  float scale = 1.0f;
  if (out_norm_w != nullptr) {
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < kMaxRowPacks; ++k) {
      const int i = threadIdx.x + k * blockDim.x;
      if (i >= packs) break;
#pragma unroll
      for (int j = 0; j < NL; ++j) ss += m[k][j] * m[k][j];
    }
    scale = rsqrtf(block_sum2(ss, 0.0f).x * inv_hidden + out_eps);
  }
#pragma unroll
  for (int k = 0; k < kMaxRowPacks; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    if (i >= packs) break;
    V o;
    if (out_norm_w != nullptr) {
      const V g = out_norm_w[i];
#pragma unroll
      for (int j = 0; j < NL; ++j)
        o.d[j] = static_cast<T>(m[k][j] * scale * static_cast<float>(g.d[j]));
    } else {
#pragma unroll
      for (int j = 0; j < NL; ++j) o.d[j] = static_cast<T>(m[k][j]);
    }
    store_out(k, i, o);
  }
  // Before the next row reuses the reductions' shared slots.
  __syncthreads();
}

// ---------------------------------------------------------------------------------
// The GEMM phase of the latent MoE tail (the one-shot and two-shot GEMM-tail kernels).
// ---------------------------------------------------------------------------------

// The most rows one pass takes: a lane holds one output column's sums for each of them.
constexpr int kGemmRows = 16;
// The K-chunk of x staged in LDS at a time, in packs. gfx950 has 160 KB of LDS, so all of
// Kimi-K3's latent K (448 packs, 112 KB) goes in at once: one staging pass and one barrier
// pair per tile. gfx942 has 64 KB: 96 packs (24 KB) beside the widest reduce tile (32 KB).
#if defined(__gfx950__)
constexpr int kGemmChunk = 448;
#else
constexpr int kGemmChunk = 96;
#endif

// out[r, col0 + n] = T(float(out[r, col0 + n]) + sum_k x[r][k] * w[n][k]) for r < rows,
// rows <= kGemmRows, the sum in fp32 and rounded once. `row(r)` points at row r of x,
// wherever it lives.
//
// x is staged in LDS a K-chunk at a time (coalesced, once per block per chunk), so the hot
// loop's row reads are LDS reads, not a global round trip per K-step.
//
// A SKINNY GEMM: a lane keeps one column's row sums in registers; K is split over the
// kLanesPerCol lanes of a column (tuned in launch.cuh) and over the waves of the
// block; shuffles and an LDS pass add the splits; blocks stride over tiles of
// kWaveSize / kLanesPerCol columns. A column's lanes read adjacent packs of its weight row.
// The order of the sum differs from hipBLASLt's, so a result agrees to the rounding of
// the last bits, not bitwise.
template <int kLanesPerCol, typename T, typename Row>
DINLINE void gemm_add_rows(Row row, int rows, const T* __restrict__ gemm_w, int n_cols,
                           int packs, T* __restrict__ out, int64_t out_stride,
                           int out_col0) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  constexpr int kTile = kWaveSize / kLanesPerCol;
  static_assert(kTile * kLanesPerCol == kWaveSize, "a column's lanes must divide a wave");
  __shared__ float partial[kMaxWaves][kGemmRows][kTile];
  __shared__ V xs[kGemmRows][kGemmChunk];
  const int lane   = threadIdx.x % kWaveSize;
  const int wave   = threadIdx.x / kWaveSize;
  const int waves  = blockDim.x / kWaveSize;
  const int column = lane % kTile;
  const int splits = waves * kLanesPerCol;
  const int split  = wave * kLanesPerCol + lane / kTile;
  const V* wv      = reinterpret_cast<const V*>(gemm_w);
  const int tiles  = (n_cols + kTile - 1) / kTile;
  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const int n   = tile * kTile + column;
    const V* wrow = wv + static_cast<int64_t>(n < n_cols ? n : 0) * packs;
    float acc[kGemmRows];
#pragma unroll
    for (int r = 0; r < kGemmRows; ++r) acc[r] = 0.0f;
    for (int k0 = 0; k0 < packs; k0 += kGemmChunk) {
      const int chunk = min(kGemmChunk, packs - k0);
      // Rows past `rows` are never staged; their sums read stale LDS and are never stored.
      for (int i = threadIdx.x; i < rows * chunk; i += blockDim.x)
        xs[i / chunk][i % chunk] = row(i / chunk)[k0 + i % chunk];
      __syncthreads();
      for (int k = split; k < chunk; k += splits) {
        const V wx = wrow[k0 + k];
        float w[NL];
#pragma unroll
        for (int j = 0; j < NL; ++j) w[j] = static_cast<float>(wx.d[j]);
#pragma unroll
        for (int r = 0; r < kGemmRows; ++r) {
          const V xr = xs[r][k];
#pragma unroll
          for (int j = 0; j < NL; ++j) acc[r] += static_cast<float>(xr.d[j]) * w[j];
        }
      }
      // Before the next chunk overwrites `xs`.
      __syncthreads();
    }
    // A column's lanes are kTile apart in the wave.
#pragma unroll
    for (int r = 0; r < kGemmRows; ++r)
#pragma unroll
      for (int s = kTile; s < kWaveSize; s <<= 1)
        acc[r] += __shfl_xor(acc[r], s, kWaveSize);
    if (lane < kTile) {
#pragma unroll
      for (int r = 0; r < kGemmRows; ++r) partial[wave][r][column] = acc[r];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kGemmRows * kTile; i += blockDim.x) {
      const int r   = i / kTile;
      const int col = tile * kTile + i % kTile;
      if (r < rows && col < n_cols) {
        float v = 0.0f;
        for (int q = 0; q < waves; ++q) v += partial[q][r][i % kTile];
        T* at = out + r * out_stride + out_col0 + col;
        *at   = static_cast<T>(static_cast<float>(*at) + v);
      }
    }
    // Before the next tile overwrites `partial`.
    __syncthreads();
  }
}

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
