// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot all-reduce: reduce-scatter, then all-gather.

#pragma once

#include "peer.cuh"
#include "reduce.cuh"

namespace hip_comms {

// TWO-SHOT: reduce-scatter, then all-gather. Every rank owns one slice of the buffer,
// reduces ONLY that slice by reading every peer's copy of it, publishes the result in its
// own scratch, and then every rank copies all ngpus slices back out.
//
// WHY IT EXISTS: bytes. One-shot moves (ngpus-1) x N per rank in one pass; this moves
// (ngpus-1)/ngpus x N twice, so 1.75N against 7N at ngpus=8 -- a 4x reduction, which is
// exactly the ratio measured between vLLM's two-stage and our one-shot on a real capture.
// It costs one more barrier, so it is the WRONG algorithm below the crossover where that
// barrier dominates and the right one above it. Neither is universally better and the
// caller picks: `algo` is the caller's decision, as it is for blocks and threads.
//
// THE SLICE IS ceil(size/ngpus) AND THE LAST RANK TAKES WHAT IS LEFT, so a buffer that
// does not divide by ngpus is still reduced exactly once everywhere -- no padding, no
// element summed twice, and a rank whose slice is empty still runs both barriers.
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    two_shot_all_reduce(const PeerPtrs* peers, PeerSignals sigs, Signal* self,
                        T* __restrict__ out, int rank, int size) {
  using V = typename traits<T>::V;

  // ROTATED by rank, for the reason one-shot rotates: the ranks do not all read rank 0
  // first. The same consequence follows -- each rank sums in a different order, so the
  // slices agree to within one ULP rather than bitwise. Unlike one-shot, EVERY element of
  // the output here was summed by exactly one rank, so all ranks see identical bytes;
  // what differs is only which order that one rank used.
  const V* ptrs[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; ++i)
    ptrs[i] = reinterpret_cast<const V*>(peers->p[(rank + i) % ngpus]);

  const int chunk = (size + ngpus - 1) / ngpus;
  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  barrier_start<ngpus>(sigs, self, rank);

  // PHASE 1 -- reduce-scatter. Our slice, summed across every rank, into our own scratch.
  {
    const int begin = rank * chunk;
    const int end   = begin + chunk < size ? begin + chunk : size;
    V* mine = reinterpret_cast<V*>(scratch_of(self));
    for (int idx = begin + tid; idx < end; idx += stride)
      mine[idx - begin] = reduce_at<T, ngpus>(ptrs, idx);
  }

  // NOT `final_sync`: the peers are about to READ what we just wrote, so this barrier has
  // to carry the release/acquire pair that the last one is allowed to drop.
  barrier_end<ngpus, false>(sigs, self, rank);

  // PHASE 2 -- all-gather. Slice i is finished and sitting in rank i's scratch; every rank
  // copies all ngpus of them into its own output.
  {
    V* dst = reinterpret_cast<V*>(out);
#pragma unroll
    for (int i = 0; i < ngpus; ++i) {
      const int begin = i * chunk;
      const int end   = begin + chunk < size ? begin + chunk : size;
      const V* src = reinterpret_cast<const V*>(scratch_of(sigs.s[i]));
      for (int idx = begin + tid; idx < end; idx += stride)
        dst[idx] = src[idx - begin];
    }
  }

  // Required for the same reason one-shot's is: without it a rank can return and let its
  // INPUT be reused while a peer is still reading that input.
  barrier_end<ngpus, true>(sigs, self, rank);
}

}  // namespace hip_comms
