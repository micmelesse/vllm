// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce: reduce-scatter, then all-gather. aiter's `cross_device_reduce_2stage`
// as written: sync, sum this rank's slice from every peer into its scratch, sync, copy every
// rank's summed slice out of its scratch. In two builds: in place, the peers' registered or
// captured inputs read where they are; staged (`kStaged`), for an eager input, each rank copies
// its input into its staging a pass at a time and each pass reads the peers' staging, so any size
// runs in one launch.

#pragma once

#include "../p2p/p2p.cuh"
#include "../common/common.cuh"
#include "shared/staged.cuh"

namespace hip_comms {

// A BLOCK IS ONE WAVE PER PEER. `num_packs` packs cut into one slice of `slice_packs` per rank
// (the last one short); a block's wave w works with peer (rank + w) % ngpus, and its lane l with
// pack blockIdx.x x lanes + l of the slice (then every grid's worth after it). ROTATED, SO THE
// RANKS SPREAD OVER THE LINKS: at any moment the eight GPUs read eight different peers, where in
// rank order every GPU reads rank 0 first. The sum's order differs by rank, which is harmless:
// each slice is summed by one rank, so every rank copies the same bytes.
//
// THE SAME BLOCK AND LANE INDEX A PACK IN BOTH PHASES: wave 0's lane l writes it, the peers' wave
// w lane l of the same block read it, after that block's sync.
//
// STAGED, a pass at a time: a pass is at most what a staging holds and what the scratch holds (a
// slice a rank). WAVE W STAGES THE SLICE ITS PEER READS, at the packs the peer's same block reads,
// so a peers barrier makes it visible.
template <typename T, int ngpus, bool kStaged>
__global__ void __launch_bounds__(kMaxThreads, 1)
    all_reduce_pull_two_shot(p2p::DevComm p, T* __restrict__ out, PackCount<kStaged> num_packs,
                             Staged<T, kStaged> staged) {
  if constexpr (kStaged) {
    using V            = typename traits<T>::V;
    constexpr int N    = traits<T>::N;
    const int lanes    = blockDim.x / ngpus;  // host: blockDim is ngpus whole waves
    const int wave     = threadIdx.x / lanes;
    const int lane     = threadIdx.x % lanes;
    const int peer     = (p.rank + wave) % ngpus;
    const int first    = blockIdx.x * lanes + lane;
    const int stride   = gridDim.x * lanes;
    const V* own       = reinterpret_cast<const V*>(staged.own_input);
    const int64_t pass = min(staged.stage_packs, p.scratch_packs * ngpus);
    __shared__ V got[kMaxThreads];

    const auto own_scratch    = p2p::scratch<T, ngpus>(p, p.rank);
    const auto their_scratch    = p2p::scratch<T, ngpus>(p, peer);
    const auto own_staging = p2p::staging<T, ngpus>(p, p.rank);
    const auto their_staging = p2p::staging<T, ngpus>(p, peer);
    V* dst             = reinterpret_cast<V*>(out);

    for (int64_t c0 = 0; c0 < num_packs; c0 += pass) {
      const int64_t n       = min(pass, num_packs - c0);
      const int slice_packs = static_cast<int>((n + ngpus - 1) / ngpus);
      block_stamp(0);
      // 1. Wave w stages the slice its peer reads, then it is visible (and, past the first pass,
      //    every peer has read this rank's scratch).
      for (int i = first; i < slice_packs; i += stride) {
        const int64_t at = int64_t{peer} * slice_packs + i;
        if (at < n) p2p::write_staging(own_staging, at, own[c0 + at]);
      }
      p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::visible>(p);
      block_stamp(1);

      // 2. Reduce-scatter: each wave loads this rank's slice from its peer's staging into LDS, and
      //    wave 0 sums the ngpus loads into this rank's scratch. EVERY WAVE RUNS EVERY PASS: the
      //    waves share a lane's packs, so they leave the loop together and the barriers match.
      const int64_t base = int64_t{p.rank} * slice_packs;
      const int count    = static_cast<int>(min(int64_t{slice_packs}, n - base));
      for (int i = first; i < count; i += stride) {
        got[threadIdx.x] = p2p::read_staging(their_staging, base + i);
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
          p2p::write_scratch(own_scratch, i, s);
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
          thread_store(dst + c0 + int64_t{peer} * slice_packs + i,
                       p2p::read_scratch(their_scratch, i));
      block_stamp(5);
    }
  } else {
    using V               = typename traits<T>::V;
    constexpr int N       = traits<T>::N;
    const int lanes       = blockDim.x / ngpus;  // host: blockDim is ngpus whole waves
    const int wave        = threadIdx.x / lanes;
    const int lane        = threadIdx.x % lanes;
    const int peer        = (p.rank + wave) % ngpus;
    const int slice_packs = (num_packs + ngpus - 1) / ngpus;
    const int first       = blockIdx.x * lanes + lane;
    const int stride      = gridDim.x * lanes;
    __shared__ V got[kMaxThreads];

    // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
    const auto own_scratch = p2p::scratch<T, ngpus>(p, p.rank);
    const auto their_scratch = p2p::scratch<T, ngpus>(p, peer);
    const auto their_input = p2p::input<T, ngpus>(p, peer);
    block_stamp(0);
    p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);
    block_stamp(1);

    // 2. Reduce-scatter: each wave loads this rank's slice from its peer into LDS, and wave 0 sums
    //    the ngpus loads into this rank's scratch. EVERY WAVE RUNS EVERY PASS: the waves share a
    //    lane's packs, so they leave the loop together and the barriers inside it match.
    const int base    = p.rank * slice_packs;
    const int mine    = min(slice_packs, num_packs - base);
    for (int i = first; i < mine; i += stride) {
      got[threadIdx.x] = p2p::read_input(their_input, base + i);
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
        p2p::write_scratch(own_scratch, i, s);
      }
      __syncthreads();
    }

    block_stamp(2);
    // 3. Every rank's sums are visible to its peers.
    p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::visible>(p);
    block_stamp(3);

    // 4. All-gather: wave w copies its peer's slice out of that peer's scratch, at its place in the
    //    output. The next call's first sync keeps a rank from overwriting its scratch while it is
    //    read.
    V* dst = reinterpret_cast<V*>(out);
    for (int i = first; i < slice_packs; i += stride)
      if (peer * slice_packs + i < num_packs)
        thread_store(dst + peer * slice_packs + i, p2p::read_scratch(their_scratch, i));
    block_stamp(5);
  }
}

}  // namespace hip_comms
