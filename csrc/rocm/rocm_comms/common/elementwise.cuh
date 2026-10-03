// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE ELEMENTWISE OPS, at thread scope: TILE MATH, on float tiles (a fused op spells each rounding
// its reference does as a to<>() between them), and the one-pack forms the flat loops still use.

#pragma once

#ifndef HIP_COMMS_COMMON_INTERFACE
#error "include common/common.cuh, common's one interface, not its parts"
#endif

#include "tile.cuh"
#include "utils.cuh"

namespace hip_comms {

// A FLOAT TILE'S MATH, element by element: thread_add and thread_mul with a tile of its shape or a
// one-row tile (a weight, every row's), and thread_mul by one scalar or a scalar a row (a row's norm
// scale).
namespace impl {
template <typename DTYPE, int TILE_M, int B_TILE_M, int TILE_N, int THREADS_PER_BLOCK, typename OP>
DINLINE Tile<DTYPE, TILE_M, TILE_N, THREADS_PER_BLOCK, float> zip(
    const Tile<DTYPE, TILE_M, TILE_N, THREADS_PER_BLOCK, float>& a,
    const Tile<DTYPE, B_TILE_M, TILE_N, THREADS_PER_BLOCK, float>& b, OP op) {
  static_assert(B_TILE_M == TILE_M || B_TILE_M == 1, "b is a's shape or one row");
  auto out = a;
#pragma unroll
  for (int m = 0; m < TILE_M; ++m)
#pragma unroll
    for (int k = 0; k < a.K; ++k)
#pragma unroll
      for (int j = 0; j < a.kPack; ++j)
        out.v[m][k][j] = op(a.v[m][k][j], b.v[B_TILE_M == 1 ? 0 : m][k][j]);
  return out;
}
}  // namespace impl

template <typename DTYPE, int TILE_M, int B_TILE_M, int TILE_N, int THREADS_PER_BLOCK>
DINLINE Tile<DTYPE, TILE_M, TILE_N, THREADS_PER_BLOCK, float> thread_add(
    const Tile<DTYPE, TILE_M, TILE_N, THREADS_PER_BLOCK, float>& a,
    const Tile<DTYPE, B_TILE_M, TILE_N, THREADS_PER_BLOCK, float>& b) {
  return impl::zip(a, b, [](float x, float y) { return x + y; });
}

template <typename DTYPE, int TILE_M, int B_TILE_M, int TILE_N, int THREADS_PER_BLOCK>
DINLINE Tile<DTYPE, TILE_M, TILE_N, THREADS_PER_BLOCK, float> thread_mul(
    const Tile<DTYPE, TILE_M, TILE_N, THREADS_PER_BLOCK, float>& a,
    const Tile<DTYPE, B_TILE_M, TILE_N, THREADS_PER_BLOCK, float>& b) {
  return impl::zip(a, b, [](float x, float y) { return x * y; });
}

template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_PER_BLOCK>
DINLINE Tile<DTYPE, TILE_M, TILE_N, THREADS_PER_BLOCK, float> thread_mul(
    const Tile<DTYPE, TILE_M, TILE_N, THREADS_PER_BLOCK, float>& a, const float (&row_scale)[TILE_M]) {
  auto out = a;
#pragma unroll
  for (int m = 0; m < TILE_M; ++m)
#pragma unroll
    for (int k = 0; k < a.K; ++k)
#pragma unroll
      for (int j = 0; j < a.kPack; ++j) out.v[m][k][j] = a.v[m][k][j] * row_scale[m];
  return out;
}

template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_PER_BLOCK>
DINLINE Tile<DTYPE, TILE_M, TILE_N, THREADS_PER_BLOCK, float> thread_mul(
    const Tile<DTYPE, TILE_M, TILE_N, THREADS_PER_BLOCK, float>& a, float scale) {
  float row_scale[TILE_M];
#pragma unroll
  for (int m = 0; m < TILE_M; ++m) row_scale[m] = scale;
  return thread_mul(a, row_scale);
}


// A pack as fp32, and fp32 rounded once back to a pack of T.
template <typename DTYPE>
DINLINE void thread_unpack(const typename traits<DTYPE>::V& v, float (&x)[traits<DTYPE>::N]) {
#pragma unroll
  for (int j = 0; j < traits<DTYPE>::N; ++j) x[j] = static_cast<float>(v.d[j]);
}

template <typename DTYPE>
DINLINE typename traits<DTYPE>::V thread_pack(const float (&x)[traits<DTYPE>::N]) {
  typename traits<DTYPE>::V v;
#pragma unroll
  for (int j = 0; j < traits<DTYPE>::N; ++j) v.d[j] = static_cast<DTYPE>(x[j]);
  return v;
}

}  // namespace hip_comms
