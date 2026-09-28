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
  using V = typename traits<T>::V;
  using C = Codec<T, kBits>;
  ipc::Comm<T, ngpus> c(p);
  const auto grp = ipc::Groups::flat(size);
  const ipc::Inbox<C, ngpus> box(grp.count());

  for (int j = 0; j < grp.iters; ++j) {
    float x[C::kVals];
    c.template mine_group<C>(grp, j, 0, size, x);
    c.template broadcast<C>(box, 0, grp.id(j), grp.members(j, 0, size), x);
  }

  c.peer_block_barrier();

  V* dst = reinterpret_cast<V*>(out);
  for (int j = 0; j < grp.iters; ++j) {
    const int n = grp.members(j, 0, size);
    float acc[C::kVals];
    c.template reduce_inbox<C>(box, 0, grp.id(j), n, acc);
    V v[kSumBatch];
    packs_of<T>(acc, v);
#pragma unroll
    for (int u = 0; u < kSumBatch; ++u)
      if (u < n) store_global(dst + grp.at(j, u), v[u]);
  }
}

}  // namespace hip_comms
