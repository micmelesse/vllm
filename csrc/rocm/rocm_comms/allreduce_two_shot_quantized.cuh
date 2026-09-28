// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot all-reduce with both transfers quantized to kBits (QuickReduce's scheme).

#pragma once

#include "ipc.cuh"
#include "utils.cuh"

namespace hip_comms {

// PUSH, NOT PULL: the sender encodes, so a rank writes its compressed input into each
// owner's inbox rather than letting peers read its raw input.
//
//   phase 1  my input slice for every owner d, encoded, into d's inbox for me
//   peer_block_barrier
//   phase 2  my slice from every inbox, decoded and summed in fp32, rounded to T, encoded
//            once and pushed into every rank's gather inbox for me (my own too, so
//            every rank decodes the same bytes and all hold identical output)
//   peer_block_barrier
//   phase 3  every rank's slice from the gather inboxes, decoded, into `out`
//
// A GROUP is kSumBatch packs of one thread, positions tid + (j * kSumBatch + u) * stride,
// with one scale. Every phase gives a thread the same groups, so the same-numbered blocks
// are all a phase waits for. The input is read only by its own rank, so no close.
//
// Scratch, per region (inbox, gather) and source rank, a slot: the payloads of every group,
// then their scales.
template <typename T, int ngpus, int kBits>
__global__ void __launch_bounds__(kMaxThreads, 1)
    allreduce_two_shot_quantized(ipc::Peers p, T* __restrict__ out, int size) {
  using V          = typename traits<T>::V;
  using Codec      = QuantCodec<T, kBits>;
  constexpr int NL = traits<T>::N;
  constexpr int P  = Codec::kPayloadPacks;
  ipc::Comm<T, ngpus> c(p);
  const int rank     = c.rank();
  const int chunk    = (size + ngpus - 1) / ngpus;
  const int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride   = gridDim.x * blockDim.x;
  const int iters    = (chunk + stride * kSumBatch - 1) / (stride * kSumBatch);
  const int n_groups = iters * stride;
  const int slot     = n_groups * P + (n_groups + 3) / 4;
  auto payload_at    = [&](int region, int src, int g) {
    return (region * ngpus + src) * slot + g * P;
  };
  auto scale_at = [&](int region, int src, int g) {
    return 4 * ((region * ngpus + src) * slot + n_groups * P) + g;
  };
  // Whether member u of group j exists in owner `d`'s slice.
  auto member = [&](int d, int j, int u) {
    const int k = tid + (j * kSumBatch + u) * stride;
    return k < chunk && d * chunk + k < size;
  };
  auto pos = [&](int d, int j, int u) {
    return d * chunk + tid + (j * kSumBatch + u) * stride;
  };

  for (int j = 0; j < iters; ++j) {
    const int g = j * stride + tid;
    for (int d = 0; d < ngpus; ++d) {
      float x[Codec::kVals];
#pragma unroll
      for (int u = 0; u < kSumBatch; ++u) {
        V v{};
        if (member(d, j, u)) v = c.mine(pos(d, j, u));
#pragma unroll
        for (int k = 0; k < NL; ++k) x[u * NL + k] = static_cast<float>(v.d[k]);
      }
      V q[P];
      const float s = Codec::encode(x, q);
#pragma unroll
      for (int w = 0; w < P; ++w) c.put(d, payload_at(0, rank, g) + w, q[w]);
      c.put(d, scale_at(0, rank, g), s);
    }
  }

  c.peer_block_barrier();

  for (int j = 0; j < iters; ++j) {
    const int g = j * stride + tid;
    float acc[Codec::kVals] = {};
    for (int src = 0; src < ngpus; ++src) {
      V q[P];
#pragma unroll
      for (int w = 0; w < P; ++w) q[w] = c.get(rank, payload_at(0, src, g) + w);
      float x[Codec::kVals];
      Codec::decode(q, c.get_float(rank, scale_at(0, src, g)), x);
#pragma unroll
      for (int i = 0; i < Codec::kVals; ++i) acc[i] += x[i];
    }
    // Rounded to T first, as the unquantized sum would land.
#pragma unroll
    for (int i = 0; i < Codec::kVals; ++i)
      acc[i] = static_cast<float>(static_cast<T>(acc[i]));
    V q[P];
    const float s = Codec::encode(acc, q);
    for (int d = 0; d < ngpus; ++d) {
#pragma unroll
      for (int w = 0; w < P; ++w) c.put(d, payload_at(1, rank, g) + w, q[w]);
      c.put(d, scale_at(1, rank, g), s);
    }
  }

  c.peer_block_barrier();

  V* dst = reinterpret_cast<V*>(out);
  for (int j = 0; j < iters; ++j) {
    const int g = j * stride + tid;
    for (int src = 0; src < ngpus; ++src) {
      V q[P];
#pragma unroll
      for (int w = 0; w < P; ++w) q[w] = c.get(rank, payload_at(1, src, g) + w);
      float x[Codec::kVals];
      Codec::decode(q, c.get_float(rank, scale_at(1, src, g)), x);
#pragma unroll
      for (int u = 0; u < kSumBatch; ++u) {
        if (!member(src, j, u)) continue;
        V v;
#pragma unroll
        for (int k = 0; k < NL; ++k) v.d[k] = static_cast<T>(x[u * NL + k]);
        store_global(dst + pos(src, j, u), v);
      }
    }
  }
}

// The scratch the kernel above needs on each rank, in bytes, at a grid of `stride` threads.
inline int64_t quantized_scratch_bytes(int kbits, int64_t flat_packs, int world,
                                       int64_t stride) {
  const int64_t chunk    = (flat_packs + world - 1) / world;
  const int64_t iters    = (chunk + stride * kSumBatch - 1) / (stride * kSumBatch);
  const int64_t n_groups = iters * stride;
  const int64_t payload  = kSumBatch * 8 * kbits / 8 / 16;
  return 2 * world * (n_groups * payload + (n_groups + 3) / 4) * 16;
}

}  // namespace hip_comms
