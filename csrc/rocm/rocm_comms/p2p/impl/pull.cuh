// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// p2p::pull, behind p2p.cuh: a rank reads its peers' buffers (their inputs, or what they
// shared into their own scratch) and writes only its own. Every phase is over
// tiles::Rows, a thread's share of a row at a time.

#pragma once

#ifndef HIP_COMMS_P2P_INTERFACE
#error "include p2p/p2p.cuh, p2p's one interface, not its parts"
#endif

#include "core.cuh"

namespace hip_comms::p2p {

// A PULL SLOT: this rank's rows of a two-shot, plain T, in its own scratch, where
// `pull::share` leaves them for `pull::gather`. Made by `pull::slot`, passed back.
struct PullSlot {
  int base;
  int chunk;
  int packs;
  DINLINE int end() const { return base + chunk * packs; }
};

namespace pull {

template <typename T, int ngpus>
DINLINE PullSlot slot(const World<T, ngpus>&, const tiles::Rows& rows, int base = 0) {
  return {base, tiles::chunk_of(rows, ngpus), rows.packs};
}

// The next slot, after `prev` (any slot) in the scratch.
template <typename T, int ngpus, typename Prev>
DINLINE PullSlot slot(const World<T, ngpus>& w, const tiles::Rows& rows, const Prev& prev) {
  return slot(w, rows, prev.end());
}

// This thread's share of `row` summed over every rank's input, rounded once to T: v[k] is
// pack k (tiles::pack), every peer's load issued before any is added.
template <typename T, int ngpus>
DINLINE void reduce(const World<T, ngpus>& w, const tiles::Rows& rows, int row,
                    typename traits<T>::V (&v)[kMaxRowPacks]) {
  const int base  = row * rows.packs;
  const int limit = base + rows.packs < rows.size ? base + rows.packs : rows.size;
  impl::sum<kMaxRowPacks>(w, base + threadIdx.x, blockDim.x, limit, v);
}

// This thread's share of owned `row`'s result into this rank's `slot`, for every rank to
// gather after a peer_barrier.
template <typename T, int ngpus>
DINLINE void share(const World<T, ngpus>& w, const PullSlot& slot, const tiles::Rows& rows,
                   int row, const typename traits<T>::V (&v)[kMaxRowPacks]) {
  const int at = slot.base + (row - w.peers.rank * slot.chunk) * slot.packs;
#pragma unroll
  for (int k = 0; k < kMaxRowPacks; ++k)
    if (tiles::has(rows, row, k)) impl::put(w, w.peers.rank, at + tiles::pack(k), v[k]);
}

// After a peer_barrier: every owner's shared rows out of its `slot`, every owner at once
// (a load in flight on every link). A block takes the local rows it shared.
// store(row, pack within the row, v).
template <typename T, int ngpus, typename Store>
DINLINE void gather(const World<T, ngpus>& w, const PullSlot& slot, const tiles::Rows& rows,
                    Store store) {
  using V = typename traits<T>::V;
  for (int lr = blockIdx.x; lr < slot.chunk; lr += gridDim.x) {
#pragma unroll
    for (int k = 0; k < kMaxRowPacks; ++k) {
      const int at = slot.base + lr * slot.packs + tiles::pack(k);
      V g[ngpus];
#pragma unroll
      for (int i = 0; i < ngpus; ++i) {
        const int row = i * slot.chunk + lr;
        if (row < rows.rows && tiles::has(rows, row, k)) g[i] = impl::get(w, i, at);
      }
#pragma unroll
      for (int i = 0; i < ngpus; ++i) {
        const int row = i * slot.chunk + lr;
        if (row < rows.rows && tiles::has(rows, row, k)) store(row, tiles::pack(k), g[i]);
      }
    }
  }
}

}  // namespace pull

}  // namespace hip_comms::p2p
