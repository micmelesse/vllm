// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE ELEMENTWISE OPS, at thread scope: a pack to fp32 and back, the rounding points a fused op
// must place where its reference rounds.

#pragma once

#include "utils.cuh"

namespace hip_comms {

// A pack as fp32, and fp32 rounded once back to a pack of T.
template <typename T>
DINLINE void unpack(const typename traits<T>::V& v, float (&x)[traits<T>::N]) {
#pragma unroll
  for (int j = 0; j < traits<T>::N; ++j) x[j] = static_cast<float>(v.d[j]);
}

template <typename T>
DINLINE typename traits<T>::V round_pack(const float (&x)[traits<T>::N]) {
  typename traits<T>::V v;
#pragma unroll
  for (int j = 0; j < traits<T>::N; ++j) v.d[j] = static_cast<T>(x[j]);
  return v;
}

}  // namespace hip_comms
