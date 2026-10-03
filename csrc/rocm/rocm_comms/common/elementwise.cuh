// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE ELEMENTWISE OPS, at thread scope: a pack to fp32 and back, the rounding points a fused op
// must place where its reference rounds.

#pragma once

#ifndef HIP_COMMS_COMMON_INTERFACE
#error "include common/common.cuh, common's one interface, not its parts"
#endif

#include "utils.cuh"

namespace hip_comms {

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
