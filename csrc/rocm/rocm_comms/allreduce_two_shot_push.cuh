// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot all-reduce, push: reduce-scatter, then all-gather, every transfer a store into
// a peer's inbox, encoded by kBits' Codec (16: T itself; 8, 4: QuickReduce's scheme).

#pragma once

#include "ipc.cuh"
#include "utils.cuh"

namespace hip_comms {

// PUSH, NOT PULL: a rank reads only its own input and its own inbox, so its input is never
// read remotely (no close), and a sender can encode what it sends.
//
//   phase 1  my input's slice for every owner d, encoded, into d's inbox (region 0)
//   peer_block_barrier
//   phase 2  my slice from every source, decoded and summed in fp32 in rank order, rounded
//            to T, encoded once and pushed into every rank's region 1 (mine too, so every
//            rank decodes the same bytes and all hold identical output)
//   peer_block_barrier
//   phase 3  every owner's slice decoded into `out`
//
// Two barriers against the pull kernel's one, and the same bytes on the links at kBits 16.
template <typename T, int ngpus, int kBits>
__global__ void __launch_bounds__(kMaxThreads, 1)
    allreduce_two_shot_push(ipc::Peers p, T* __restrict__ out, int size) {
  using V          = typename traits<T>::V;
  using C          = Codec<T, kBits>;
  constexpr int NL = traits<T>::N;
  constexpr int P  = C::kPayloadPacks;
  ipc::Comm<T, ngpus> c(p);
  const int chunk = (size + ngpus - 1) / ngpus;
  const ipc::Groups grp(chunk);
  const ipc::Inbox<C, ngpus> box(grp.count());

  for (int j = 0; j < grp.iters; ++j) {
    for (int d = 0; d < ngpus; ++d) {
      float x[C::kVals];
      c.template mine_group<C>(grp, j, d * chunk, size, x);
      V q[P];
      const float s = C::encode(x, q);
      c.template push<C>(d, box, 0, grp.id(j), q, s);
    }
  }

  c.peer_block_barrier();

  for (int j = 0; j < grp.iters; ++j) {
    float acc[C::kVals] = {};
    for (int src = 0; src < ngpus; ++src) {
      float x[C::kVals];
      c.template read_inbox<C>(box, 0, src, grp.id(j), x);
#pragma unroll
      for (int i = 0; i < C::kVals; ++i) acc[i] += x[i];
    }
    // Rounded to T first, as the unquantized sum lands.
#pragma unroll
    for (int i = 0; i < C::kVals; ++i) acc[i] = static_cast<float>(static_cast<T>(acc[i]));
    V q[P];
    const float s = C::encode(acc, q);
    for (int d = 0; d < ngpus; ++d) c.template push<C>(d, box, 1, grp.id(j), q, s);
  }

  c.peer_block_barrier();

  V* dst = reinterpret_cast<V*>(out);
  for (int j = 0; j < grp.iters; ++j) {
    for (int src = 0; src < ngpus; ++src) {
      float x[C::kVals];
      c.template read_inbox<C>(box, 1, src, grp.id(j), x);
#pragma unroll
      for (int u = 0; u < kSumBatch; ++u) {
        if (!grp.has(j, u, src * chunk, size)) continue;
        V v;
#pragma unroll
        for (int k = 0; k < NL; ++k) v.d[k] = static_cast<T>(x[u * NL + k]);
        store_global(dst + src * chunk + grp.at(j, u), v);
      }
    }
  }
}

}  // namespace hip_comms
