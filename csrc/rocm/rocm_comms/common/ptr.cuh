// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE POINTER, beside the tile (tile.cuh) the other primitive every kernel is written in: what a
// kernel touches, local or another rank's, as a Ptr -- its data, its layout (both strides, in
// elements), and whose memory it is -- and the boundary that makes them, a kernel's first lines.

#pragma once

#ifndef HIP_COMMS_COMMON_INTERFACE
#error "include common/interface.cuh, common's one interface, not its parts"
#endif

#include <array>
#include <cstdint>
#include <type_traits>

#include "peers.cuh"

namespace hip_comms {

// A POINTER A KERNEL TOUCHES, local or another rank's: its data, its layout (the strides of its
// rows and columns, in elements, as the kernel's arguments gave them) and whose memory it is. A
// kernel turns its arguments into these in its first lines (`rank_ptrs`, `rank_ptr`, `local_ptr`)
// and every load and store after takes one. The packed loads need stride_n 1; the host checks it.
template <typename T>
struct Ptr {
  T* data;
  int64_t stride_m;
  int64_t stride_n;
  int rank;
};

namespace impl {

// EVERY RANK'S BUFFER AS A Ptr, rank r's data at r (rank r the same across the wave, as rank_of
// makes it), each with the buffer's row stride.
template <typename T, int WORLD>
DINLINE std::array<Ptr<T>, WORLD> rank_ptrs(T* const* p, int64_t stride_m, int64_t stride_n) {
  std::array<Ptr<T>, WORLD> all;
#pragma unroll
  for (int r = 0; r < WORLD; ++r)
    all[r] = Ptr<T>{impl::rank_of(p, r), stride_m, stride_n, r};
  return all;
}
// ONE RANK'S BUFFER AS A Ptr (this rank's own, most often), by the same rank_of.
template <typename T, int WORLD>
DINLINE Ptr<T> rank_ptr(T* const* p, int rank, int64_t stride_m, int64_t stride_n) {
  return Ptr<T>{impl::rank_of(p, rank), stride_m, stride_n, rank};
}
template <typename T>
DINLINE Ptr<T> local_ptr(T* data, int64_t stride_m, int64_t stride_n, int rank) {
  return Ptr<T>{data, stride_m, stride_n, rank};
}

}  // namespace impl
}  // namespace hip_comms
