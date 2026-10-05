// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE ONLINE SOFTMAX, at thread scope: a softmax over logits that arrive a tile at a time, folded
// into a running weighted sum without a second pass (flash attention's).

#pragma once

#ifndef HIP_COMMS_COMMON_INTERFACE
#error "include common/interface.cuh, common's one interface, not its parts"
#endif

#include <cmath>

#include "utils.cuh"

namespace hip_comms {

// The softmax so far: the largest logit seen and the sum of exp(logit - max).
struct OnlineSoftmax {
  float max         = -INFINITY;
  float denominator = 0.0f;
};

// Folds N logits into `s` (a -INFINITY logit is no source). Returns the scale the weighted sum so
// far takes, and in `scale` each logit's weight; the sum is final once divided by s.denominator.
namespace impl {
template <int NUM_LOGITS>
DINLINE float thread_softmax_fold(OnlineSoftmax& s, const float (&logit)[NUM_LOGITS], float (&scale)[NUM_LOGITS]) {
  float new_max = s.max;
#pragma unroll
  for (int t = 0; t < NUM_LOGITS; ++t) new_max = fmaxf(new_max, logit[t]);
  const float old_scale = __expf(s.max - new_max);
  s.denominator *= old_scale;
#pragma unroll
  for (int t = 0; t < NUM_LOGITS; ++t) {
    scale[t] = __expf(logit[t] - new_max);
    s.denominator += scale[t];
  }
  s.max = new_max;
  return old_scale;
}
}  // namespace impl

}  // namespace hip_comms
