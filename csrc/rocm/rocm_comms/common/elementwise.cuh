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
template <typename A, typename B, typename OP>
DINLINE A zip(const A& a, const B& b, OP op) {
  static_assert(std::is_same_v<typename A::Acc, float> && std::is_same_v<typename B::Acc, float>,
                "tile math is on float tiles; to<float>() first");
  static_assert(B::kRows == A::kRows || B::kRows == 1, "b is a's shape or one row");
  static_assert(B::K == A::K && B::kPack == A::kPack, "b holds a's columns");
  A out = a;
#pragma unroll
  for (int m = 0; m < A::kRows; ++m)
#pragma unroll
    for (int k = 0; k < A::K; ++k)
#pragma unroll
      for (int j = 0; j < A::kPack; ++j)
        out.v[m][k][j] = op(a.v[m][k][j], b.v[B::kRows == 1 ? 0 : m][k][j]);
  return out;
}
}  // namespace impl

template <typename A, typename B, std::enable_if_t<is_tile<B>::value, int> = 0>
DINLINE A thread_add(const A& a, const B& b) {
  return impl::zip(a, b, [](float x, float y) { return x + y; });
}

template <typename A, typename B, std::enable_if_t<is_tile<B>::value, int> = 0>
DINLINE A thread_mul(const A& a, const B& b) {
  return impl::zip(a, b, [](float x, float y) { return x * y; });
}

template <typename A>
DINLINE A thread_mul(const A& a, const float (&row_scale)[A::kRows]) {
  static_assert(std::is_same_v<typename A::Acc, float>, "tile math is on float tiles");
  A out = a;
#pragma unroll
  for (int m = 0; m < A::kRows; ++m)
#pragma unroll
    for (int k = 0; k < A::K; ++k)
#pragma unroll
      for (int j = 0; j < A::kPack; ++j) out.v[m][k][j] = a.v[m][k][j] * row_scale[m];
  return out;
}

template <typename A>
DINLINE A thread_mul(const A& a, float scale) {
  float row_scale[A::kRows];
#pragma unroll
  for (int m = 0; m < A::kRows; ++m) row_scale[m] = scale;
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
