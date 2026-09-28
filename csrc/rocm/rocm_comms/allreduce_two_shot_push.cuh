// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot all-reduce, push: reduce-scatter, then all-gather, every transfer a store into
// a peer's inbox, encoded by kBits' Codec (16: T itself; 8, 4: QuickReduce's scheme).

#pragma once

#include "p2p/p2p.cuh"
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
    allreduce_two_shot_push(p2p::Peers p, T* __restrict__ out, int size) {
  using V            = typename traits<T>::V;
  using C            = p2p::Codec<T, kBits>;
  const auto w       = p2p::start<T, ngpus>(p);
  const int chunk    = (size + ngpus - 1) / ngpus;
  const auto box_in  = p2p::push::buffer_inbox<C>(w, chunk);
  const auto box_out = p2p::push::buffer_inbox<C>(w, chunk, box_in.end());
  V* dst             = reinterpret_cast<V*>(out);

  p2p::push::scatter_buffer(w, box_in, chunk, size);

  p2p::peer_block_barrier(w);

  p2p::push::reduce_broadcast_slice(w, box_in, box_out, chunk, size);

  p2p::peer_block_barrier(w);

  p2p::push::gather_buffer(w, box_out, chunk, size,
                           [&](int at, const V& v) { store_global(dst + at, v); });
}

}  // namespace hip_comms
