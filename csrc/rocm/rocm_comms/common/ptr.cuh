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
template <typename T>
struct Ptr {
  T* data;
  int64_t stride;
  int rank;
};

namespace impl {

// EVERY RANK'S BUFFER AS A Ptr, rank r's data at r (rank r the same across the wave, as rank_of
// makes it), each with the buffer's row stride.
template <typename T, int WORLD>
DINLINE std::array<Ptr<T>, WORLD> rank_ptrs(const PeerPtrs& p, int64_t stride) {
  std::array<Ptr<T>, WORLD> all;
#pragma unroll
  for (int r = 0; r < WORLD; ++r)
    all[r] = Ptr<T>{impl::rank_of<std::remove_const_t<T>, WORLD>(p, r), stride, r};
  return all;
}
// ONE RANK'S BUFFER AS A Ptr (this rank's own, most often), by the same rank_of.
template <typename T, int WORLD>
DINLINE Ptr<T> rank_ptr(const PeerPtrs& p, int rank, int64_t stride) {
  return Ptr<T>{impl::rank_of<std::remove_const_t<T>, WORLD>(p, rank), stride, rank};
}
template <typename T>
DINLINE Ptr<T> local_ptr(T* data, int64_t stride, int rank) { return Ptr<T>{data, stride, rank}; }

}  // namespace impl
}  // namespace hip_comms
