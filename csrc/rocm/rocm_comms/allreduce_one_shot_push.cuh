// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot all-reduce, push: every rank's whole input into every rank's inbox, encoded by
// kBits' Codec (16: T itself; 8, 4: QuickReduce's scheme), then each sums locally.

#pragma once

#include "p2p/p2p.cuh"
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
    allreduce_one_shot_push(p2p::Peers p, T* __restrict__ out, int size) {
  using V        = typename traits<T>::V;
  using C        = p2p::Codec<T, kBits>;
  const auto w   = p2p::start<T, ngpus>(p);
  const auto box = p2p::push::buffer_inbox<C>(w, size);
  V* dst         = reinterpret_cast<V*>(out);

  p2p::push::broadcast_buffer(w, box, size);

  p2p::peer_block_barrier(w);

  p2p::push::reduce_buffer(w, box, size,
                           [&](int at, const V& v) { store_global(dst + at, v); });
}

}  // namespace hip_comms
