// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot all-reduce.

#pragma once

#include "ipc.cuh"
#include "utils.cuh"

namespace hip_comms {

// ONE-SHOT: every rank sums every peer's input over the whole buffer into its own output.
// Moves ngpus x the bytes of two-shot and needs no barrier between phases, so it wins while
// the barrier, not the bytes, dominates.
template <typename T, int ngpus>
__global__ void __launch_bounds__(kMaxThreads, 1)
    allreduce_one_shot_pull(ipc::Peers p, T* __restrict__ out, int size) {
  using V = typename traits<T>::V;
  ipc::Comm<T, ngpus> c(p);
  V* dst = reinterpret_cast<V*>(out);
  c.reduce_flat(0, size, [&](int at, const V& v) { store_global(dst + at, v); });
  c.close();
}

}  // namespace hip_comms
