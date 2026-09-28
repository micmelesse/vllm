// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE PULL PATTERNS: a rank reads its peers' buffers (their inputs, or scratch they
// filled) and writes only its own. Free functions over `Peers` and the kernel's `Inputs`,
// built on core.cuh; `Pull<T, ngpus>` holds nothing.

#pragma once

#include "core.cuh"

namespace hip_comms::p2p {

//   reduce_flat(p, in, begin, end, store)   a flat range summed over ranks, batched
//   sum_row(p, in, base, packs, v)          this thread's share of a row, summed, batched
//   gather_flat(p, chunk, size, store)      after a peer_block_barrier: every rank's slice
//   gather_rows<k>(p, chunk, rows, ...)     the same, whole rows, k regions
template <typename T, int ngpus>
struct Pull {
  using core   = Core<T, ngpus>;
  using V      = typename core::V;
  using Inputs = typename core::Inputs;

  // THE FLAT REDUCE: positions [begin, end) summed over ranks, grid-strided, kSumBatch
  // packs a thread all loaded before any is stored. store(position, v). A gather_flat after
  // a peer_block_barrier reads back exactly these positions, thread for thread.
  template <typename Store>
  static DINLINE void reduce_flat(const Peers& p, const Inputs& in, int begin, int end,
                                  Store store) {
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int idx = begin + tid; idx < end; idx += stride * kSumBatch) {
      V v[kSumBatch];
      core::template sum<kSumBatch>(p, in, idx, stride, end, v);
#pragma unroll
      for (int u = 0; u < kSumBatch; ++u)
        if (idx + u * stride < end) store(idx + u * stride, v[u]);
    }
  }

  // This thread's share of row pack range [base, base + packs), summed over ranks: v[k]
  // is pack threadIdx.x + k * blockDim.x, every peer's load issued before any is added.
  static DINLINE void sum_row(const Peers& p, const Inputs& in, int base, int packs,
                              V (&v)[kMaxRowPacks]) {
    core::template sum<kMaxRowPacks>(p, in, base + threadIdx.x, blockDim.x, base + packs,
                                     v);
  }

  // THE TWO-SHOT GATHERS, after a peer_block_barrier: each reads back exactly what the
  // same-numbered block on every peer put, every peer at once (index outer, peer inner,
  // so a load is in flight on every link), and hands each pack to `store`.
  //
  // gather_flat: a flat buffer sliced `chunk` packs per rank; each thread takes the
  // positions tid, tid + grid, ... it reduced. store(position in the whole buffer, v).
  template <typename Store>
  static DINLINE void gather_flat(const Peers& p, int chunk, int size, Store store) {
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int k = tid; k < chunk; k += stride) {
      V g[ngpus];
#pragma unroll
      for (int i = 0; i < ngpus; ++i)
        if (i * chunk + k < size) g[i] = core::get(p, i, k);
#pragma unroll
      for (int i = 0; i < ngpus; ++i)
        if (i * chunk + k < size) store(i * chunk + k, g[i]);
    }
  }

  // gather_rows: `chunk` whole rows per rank; block b takes local rows b, b + grid, ...
  // as it reduced them. kRegions regions, region_packs apart in the scratch.
  // store(region, row, pack within the row, v).
  template <int kRegions, typename Store>
  static DINLINE void gather_rows(const Peers& p, int chunk, int rows, int packs,
                                  int region_packs, Store store) {
    for (int lr = blockIdx.x; lr < chunk; lr += gridDim.x) {
      for (int k = threadIdx.x; k < packs; k += blockDim.x) {
        const int at = lr * packs + k;
        V g[kRegions][ngpus];
#pragma unroll
        for (int i = 0; i < ngpus; ++i)
          if (i * chunk + lr < rows)
#pragma unroll
            for (int r = 0; r < kRegions; ++r)
              g[r][i] = core::get(p, i, r * region_packs + at);
#pragma unroll
        for (int i = 0; i < ngpus; ++i)
          if (i * chunk + lr < rows)
#pragma unroll
            for (int r = 0; r < kRegions; ++r)
              store(r, i * chunk + lr, k, g[r][i]);
      }
    }
  }
};

}  // namespace hip_comms::p2p
