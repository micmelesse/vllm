// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE ELEMENTWISE OPS, at thread scope: TILE MATH, on float tiles (a fused op spells each rounding
// its reference does as a to<>() between them).

#pragma once

#ifndef HIP_COMMS_COMMON_INTERFACE
#error "include common/common.cuh, common's one interface, not its parts"
#endif

#include "tile.cuh"
#include "utils.cuh"

namespace hip_comms {

// A FLOAT TILE'S MATH, element by element: tile_add and tile_mul with a tile of its shape or a
// one-row tile (a weight, every row's), and tile_mul by one scalar or a scalar a row (a row's norm
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
DINLINE A tile_add(const A& a, const B& b) {
  return impl::zip(a, b, [](float x, float y) { return x + y; });
}

template <typename A, typename B, std::enable_if_t<is_tile<B>::value, int> = 0>
DINLINE A tile_mul(const A& a, const B& b) {
  return impl::zip(a, b, [](float x, float y) { return x * y; });
}

template <typename A>
DINLINE A tile_mul(const A& a, const float (&row_scale)[A::kRows]) {
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
DINLINE A tile_mul(const A& a, float scale) {
  float row_scale[A::kRows];
#pragma unroll
  for (int m = 0; m < A::kRows; ++m) row_scale[m] = scale;
  return tile_mul(a, row_scale);
}

// a x a_row_scale + b x b_row_scale, element by element, as fused multiply-adds (a softmax's fold:
// the old sum rescaled plus a new source weighted). Two multiplies and an add were three VALU an
// element and, vectorized, packed multiplies with a chain of adds (ISA 2026-10-04T02-22-41Z).
template <typename A, typename B>
DINLINE A tile_fma(const A& a, const float (&a_row_scale)[A::kRows], const B& b,
                   const float (&b_row_scale)[A::kRows]) {
  static_assert(std::is_same_v<typename A::Acc, float> && std::is_same_v<typename B::Acc, float>,
                "tile math is on float tiles");
  A out = a;
#pragma unroll
  for (int m = 0; m < A::kRows; ++m)
#pragma unroll
    for (int k = 0; k < A::K; ++k)
#pragma unroll
      for (int j = 0; j < A::kPack; ++j)
        out.v[m][k][j] = fmaf(b.v[m][k][j], b_row_scale[m], a.v[m][k][j] * a_row_scale[m]);
  return out;
}

}  // namespace hip_comms
