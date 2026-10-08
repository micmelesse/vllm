// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE POINTER, beside the tile (tile.cuh) the other primitive every kernel is written in: what a
// kernel touches, local or another rank's, as a Ptr -- its data, the stride of its rows in
// elements, and whose memory it is -- and the boundary that makes them, a kernel's first lines.

#pragma once

#ifndef HIP_COMMS_COMMON_INTERFACE
#error "include common/interface.cuh, common's one interface, not its parts"
#endif

#include <array>
#include <cstdint>
#include <type_traits>

#include "peers.cuh"

namespace hip_comms {

// A POINTER A KERNEL TOUCHES, local or another rank's: its data, the stride of its rows in
// elements, and whose memory it is. A kernel turns its arguments into these in its first lines
// (`rank_ptrs`, `local_ptr`) and every load and store after takes one.
template <typename T, typename S = int64_t>
struct Ptr {
  T* data;
  S stride;
  int rank;
};

namespace impl {

// EVERY RANK'S BUFFER AS A Ptr, rank r's data at r (rank r the same across the wave, as rank_of
// makes it), each with the buffer's row stride.
template <typename T, int WORLD, typename S>
DINLINE std::array<Ptr<T, S>, WORLD> rank_ptrs(const PeerPtrs& p, S stride) {
  std::array<Ptr<T, S>, WORLD> all;
#pragma unroll
  for (int r = 0; r < WORLD; ++r)
    all[r] = Ptr<T, S>{impl::rank_of<std::remove_const_t<T>, WORLD>(p, r), stride, r};
  return all;
}
// ONE RANK'S BUFFER AS A Ptr (this rank's own, most often), by the same rank_of.
template <typename T, int WORLD, typename S>
DINLINE Ptr<T, S> rank_ptr(const PeerPtrs& p, int rank, S stride) {
  return Ptr<T, S>{impl::rank_of<std::remove_const_t<T>, WORLD>(p, rank), stride, rank};
}
template <typename T, typename S>
DINLINE Ptr<T, S> local_ptr(T* data, S stride, int rank) { return Ptr<T, S>{data, stride, rank}; }

}  // namespace impl
}  // namespace hip_comms
