// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// p2p::pull, behind p2p.cuh: a rank reads its peers' buffers (their inputs, or scratch
// they filled) and writes only its own.

#pragma once

#ifndef HIP_COMMS_P2P_INTERFACE
#error "include p2p/p2p.cuh, p2p's one interface, not its parts"
#endif

#include "core.cuh"

namespace hip_comms::p2p::pull {

// THE BUFFER REDUCE: pack positions [begin, end) of the whole buffer summed over ranks,
// grid-strided, kSumBatch packs a thread all loaded before any is stored. store(pos, v).
// A gather_buffer after a peer_block_barrier reads back exactly these positions, thread
// for thread.
template <typename T, int ngpus, typename Store>
DINLINE void reduce_buffer(const World<T, ngpus>& w, int begin, int end, Store store) {
  using V          = typename traits<T>::V;
  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int idx = begin + tid; idx < end; idx += stride * kSumBatch) {
    V v[kSumBatch];
    impl::sum<kSumBatch>(w, idx, stride, end, v);
#pragma unroll
    for (int u = 0; u < kSumBatch; ++u)
      if (idx + u * stride < end) store(idx + u * stride, v[u]);
  }
}

// This thread's share of row pack range [base, base + packs), summed over ranks: v[k] is
// pack threadIdx.x + k * blockDim.x, every peer's load issued before any is added.
template <typename T, int ngpus>
DINLINE void sum_row(const World<T, ngpus>& w, int base, int packs,
                     typename traits<T>::V (&v)[kMaxRowPacks]) {
  impl::sum<kMaxRowPacks>(w, base + threadIdx.x, blockDim.x, base + packs, v);
}

// THE TWO-SHOT GATHERS, after a peer_block_barrier: each reads back exactly what the
// same-numbered block on every peer put, every peer at once (index outer, peer inner,
// so a load is in flight on every link), and hands each pack to `store`.
//
// gather_buffer: the buffer sliced `chunk` packs per rank; each thread takes the
// positions tid, tid + grid, ... it reduced. store(position in the whole buffer, v).
template <typename T, int ngpus, typename Store>
DINLINE void gather_buffer(const World<T, ngpus>& w, int chunk, int size, Store store) {
  using V          = typename traits<T>::V;
  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int k = tid; k < chunk; k += stride) {
    V g[ngpus];
#pragma unroll
    for (int i = 0; i < ngpus; ++i)
      if (i * chunk + k < size) g[i] = impl::get(w, i, k);
#pragma unroll
    for (int i = 0; i < ngpus; ++i)
      if (i * chunk + k < size) store(i * chunk + k, g[i]);
  }
}

// gather_rows: `chunk` whole rows per rank; block b takes local rows b, b + grid, ... as
// it reduced them. kRegions regions, region_packs apart in the scratch.
// store(region, row, pack within the row, v).
template <int kRegions, typename T, int ngpus, typename Store>
DINLINE void gather_rows(const World<T, ngpus>& w, int chunk, int rows, int packs,
                         int region_packs, Store store) {
  using V = typename traits<T>::V;
  for (int lr = blockIdx.x; lr < chunk; lr += gridDim.x) {
    for (int k = threadIdx.x; k < packs; k += blockDim.x) {
      const int at = lr * packs + k;
      V g[kRegions][ngpus];
#pragma unroll
      for (int i = 0; i < ngpus; ++i)
        if (i * chunk + lr < rows)
#pragma unroll
          for (int r = 0; r < kRegions; ++r)
            g[r][i] = impl::get(w, i, r * region_packs + at);
#pragma unroll
      for (int i = 0; i < ngpus; ++i)
        if (i * chunk + lr < rows)
#pragma unroll
          for (int r = 0; r < kRegions; ++r) store(r, i * chunk + lr, k, g[r][i]);
    }
  }
}

}  // namespace hip_comms::p2p::pull
