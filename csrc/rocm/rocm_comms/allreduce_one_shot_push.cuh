// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot all-reduce, push: every rank's whole input into every rank's inbox, encoded by
// kBits' Codec (16: T itself; 8, 4: QuickReduce's scheme), then each sums locally.

#pragma once

#include "ipc.cuh"
#include "utils.cuh"

namespace hip_comms {

// PUSH, NOT PULL: a rank reads only its own input and its own inbox.
//
//   phase 1  my whole input, encoded once, into every rank's inbox (mine too)
//   peer_block_barrier
//   phase 2  every source decoded from my inbox and summed in fp32 in rank order, into
//            `out` (every rank the same bytes in the same order: identical output)
//
// One barrier and no close (the input is never read remotely), against the pull kernel's
// start handshake plus close; the same ngpus x the bytes of two-shot.
template <typename T, int ngpus, int kBits>
__global__ void __launch_bounds__(kMaxThreads, 1)
    allreduce_one_shot_push(ipc::Peers p, T* __restrict__ out, int size) {
  using V          = typename traits<T>::V;
  using C          = Codec<T, kBits>;
  constexpr int NL = traits<T>::N;
  constexpr int P  = C::kPayloadPacks;
  ipc::Comm<T, ngpus> c(p);
  const ipc::Groups grp(size);
  const ipc::Inbox<C, ngpus> box(grp.count());

  for (int j = 0; j < grp.iters; ++j) {
    float x[C::kVals];
    c.template mine_group<C>(grp, j, 0, size, x);
    V q[P];
    const float s = C::encode(x, q);
    for (int d = 0; d < ngpus; ++d) c.template push<C>(d, box, 0, grp.id(j), q, s);
  }

  c.peer_block_barrier();

  V* dst = reinterpret_cast<V*>(out);
  for (int j = 0; j < grp.iters; ++j) {
    float acc[C::kVals] = {};
    for (int src = 0; src < ngpus; ++src) {
      float x[C::kVals];
      c.template read_inbox<C>(box, 0, src, grp.id(j), x);
#pragma unroll
      for (int i = 0; i < C::kVals; ++i) acc[i] += x[i];
    }
#pragma unroll
    for (int u = 0; u < kSumBatch; ++u) {
      if (!grp.has(j, u, 0, size)) continue;
      V v;
#pragma unroll
      for (int k = 0; k < NL; ++k) v.d[k] = static_cast<T>(acc[u * NL + k]);
      store_global(dst + grp.at(j, u), v);
    }
  }
}

}  // namespace hip_comms
