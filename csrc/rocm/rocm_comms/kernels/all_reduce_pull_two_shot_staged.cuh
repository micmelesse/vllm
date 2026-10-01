// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce of an EAGER input (`all_reduce_pull_two_shot_staged`): each rank copies
// its input into its staging a pass at a time, then the passes run as all_reduce_pull_two_shot
// does (reduce-scatter into scratch, all-gather out of it), reading the peers' staging. Any size
// runs in one launch. A registered or captured input reads in place (all_reduce_pull_two_shot).

#pragma once

#include "../p2p/p2p.cuh"
#include "../common/common.cuh"

namespace hip_comms {

// As all_reduce_pull_two_shot, a pass at a time: a pass is at most what a staging holds and what
// the scratch holds (a slice a rank). WAVE W STAGES THE SLICE ITS PEER READS, at the packs the
// peer's same block reads, so a peers barrier makes it visible.
template <typename T, int ngpus>
__global__ void __launch_bounds__(kMaxThreads, 1)
    all_reduce_pull_two_shot_staged(p2p::DevComm p, T* __restrict__ out,
                                    const T* __restrict__ own_input, int64_t num_packs,
                                    int64_t stage_packs) {
  using V            = typename traits<T>::V;
  constexpr int N    = traits<T>::N;
  const int lanes    = blockDim.x / ngpus;  // host: blockDim is ngpus whole waves
  const int wave     = threadIdx.x / lanes;
  const int lane     = threadIdx.x % lanes;
  const int peer     = (p.rank + wave) % ngpus;
  const int first    = blockIdx.x * lanes + lane;
  const int stride   = gridDim.x * lanes;
  const V* own       = reinterpret_cast<const V*>(own_input);
  const int64_t pass = min(stage_packs, p.scratch_packs * ngpus);
  __shared__ V got[kMaxThreads];

  const auto self    = p2p::self<T, ngpus>(p);
  const auto them    = p2p::peer<T, ngpus>(p, peer);
  const auto mine    = p2p::staging<T, ngpus>(p, p.rank);
  const auto theirs  = p2p::staging<T, ngpus>(p, peer);
  V* dst             = reinterpret_cast<V*>(out);

  for (int64_t c0 = 0; c0 < num_packs; c0 += pass) {
    const int64_t n       = min(pass, num_packs - c0);
    const int slice_packs = static_cast<int>((n + ngpus - 1) / ngpus);
    block_stamp(0);
    // 1. Wave w stages the slice its peer reads, then it is visible (and, past the first pass,
    //    every peer has read this rank's scratch).
    for (int i = first; i < slice_packs; i += stride) {
      const int64_t at = int64_t{peer} * slice_packs + i;
      if (at < n) p2p::write_staging(mine, at, own[c0 + at]);
    }
    p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::visible>(p);
    block_stamp(1);

    // 2. Reduce-scatter: each wave loads this rank's slice from its peer's staging into LDS, and
    //    wave 0 sums the ngpus loads into this rank's scratch. EVERY WAVE RUNS EVERY PASS: the
    //    waves share a lane's packs, so they leave the loop together and the barriers match.
    const int64_t base = int64_t{p.rank} * slice_packs;
    const int count    = static_cast<int>(min(int64_t{slice_packs}, n - base));
    for (int i = first; i < count; i += stride) {
      got[threadIdx.x] = p2p::read_staging(theirs, base + i);
      __syncthreads();
      if (wave == 0) {
        float acc[N];
#pragma unroll
        for (int j = 0; j < N; ++j) acc[j] = static_cast<float>(got[lane].d[j]);
#pragma unroll
        for (int w = 1; w < ngpus; ++w)
#pragma unroll
          for (int j = 0; j < N; ++j) acc[j] += static_cast<float>(got[w * lanes + lane].d[j]);
        V s;
#pragma unroll
        for (int j = 0; j < N; ++j) s.d[j] = static_cast<T>(acc[j]);
        p2p::write_scratch(self, i, s);
      }
      __syncthreads();
    }

    block_stamp(2);
    // 3. Every rank's sums are visible to its peers, and every peer has read this rank's staged
    //    pass, so the next one may overwrite it.
    p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::visible>(p);
    block_stamp(3);

    // 4. All-gather: wave w copies its peer's slice out of that peer's scratch, at its place in
    //    the output. The next pass's (or call's) first sync keeps a rank from overwriting its
    //    scratch while it is read.
    for (int i = first; i < slice_packs; i += stride)
      if (int64_t{peer} * slice_packs + i < n)
        thread_store(dst + c0 + int64_t{peer} * slice_packs + i, p2p::read_scratch(them, i));
    block_stamp(5);
  }
}

}  // namespace hip_comms
