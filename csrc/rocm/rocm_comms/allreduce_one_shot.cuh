// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot all-reduce.

#pragma once

#include "ipc.cuh"
#include "utils.cuh"

namespace hip_comms {

// ONE-SHOT: every rank reads every peer's input over the whole buffer and writes its own
// output. Moves ngpus x the bytes of a two-stage, so it is the wrong algorithm at 1 MiB
// and the right one to land first: it exercises the peer handshake end to end and is
// correct, which makes two-stage a pure performance change against a working baseline.
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    allreduce_one_shot(ipc::Peers p, T* __restrict__ out, int size) {
  using V        = typename traits<T>::V;
  const int rank = p.rank();
  // ROTATED by rank, so the ranks do not all hammer rank 0's buffer first. The consequence is
  // worth knowing: every rank sums the same ngpus values in a DIFFERENT order, so the outputs
  // agree only to within one ULP of T rather than bitwise. `reduce_at` accumulates in float and
  // rounds once, which bounds it there. Rank 0's order is 0..ngpus-1, so rank 0 alone matches a
  // sequential reference exactly -- a test that reads only rank 0 will call this bit-exact.
  const V* ptrs[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; ++i)
    ptrs[i] = p.input<V>((rank + i) % ngpus);

  p.barrier_start<ngpus>();
  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int idx = tid; idx < size; idx += stride)
    reinterpret_cast<V*>(out)[idx] = reduce_at<T, ngpus>(ptrs, idx);
  // Required, not defensive: without it a rank can return and let its INPUT be reused
  // while a peer is still reading that input.
  p.barrier_end<ngpus, true>();
}

}  // namespace hip_comms
