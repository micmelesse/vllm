// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// p2p::pull, behind p2p.cuh: a rank reads its peers' buffers (their inputs, or what they
// shared into their own scratch) and writes only its own. Every phase takes a tiling
// (tiles::Rows or tiles::Buffer) and a unit of it, a thread's share at a time.

#pragma once

#ifndef HIP_COMMS_P2P_INTERFACE
#error "include p2p/p2p.cuh, p2p's one interface, not its parts"
#endif

#include "core.cuh"

namespace hip_comms::p2p {

// A PULL SLOT: this rank's units of a two-shot, plain T, in its own scratch, where
// `pull::share` leaves them for `pull::gather`: per local unit, pack k of every lane at
// (l x kMaxRowPacks + k) x lanes + lane, so a unit's packs are read coalesced. Made by
// `pull::slot`, passed back.
struct PullSlot {
  int base;
  int locals;
  int lanes;
  DINLINE int at(int l, int k, int lane) const {
    return base + (l * kMaxRowPacks + k) * lanes + lane;
  }
  DINLINE int end() const { return base + locals * kMaxRowPacks * lanes; }
};

namespace pull {

template <typename T, int ngpus, typename Tiling>
DINLINE PullSlot slot(const World<T, ngpus>&, const Tiling& t, int base = 0) {
  return {base, t.locals(), t.lanes()};
}

// The next slot, after `prev` (any slot) in the scratch.
template <typename T, int ngpus, typename Tiling, typename Prev>
DINLINE PullSlot slot(const World<T, ngpus>& w, const Tiling& t, const Prev& prev) {
  return slot(w, t, prev.end());
}

// This thread's share of unit u summed over every rank's input, rounded once to T.
template <typename T, int ngpus, typename Tiling>
DINLINE void reduce(const World<T, ngpus>& w, const Tiling& t, int u,
                    typename traits<T>::V (&v)[kMaxRowPacks]) {
  impl::sum(w, t, u, v);
}

// This thread's share of owned unit u's result into this rank's `slot`, for every rank to
// gather after a peer_barrier.
template <typename T, int ngpus, typename Tiling>
DINLINE void share(const World<T, ngpus>& w, const PullSlot& s, const Tiling& t, int u,
                   const typename traits<T>::V (&v)[kMaxRowPacks]) {
#pragma unroll
  for (int k = 0; k < kMaxRowPacks; ++k)
    if (t.has(u, k)) impl::put(w, w.peers.rank, s.at(t.local(u), k, t.lane()), v[k]);
}

// After a peer_barrier: every owner's shared units out of its `slot`, every owner at once
// (a load in flight on every link); a thread takes the local units it shared.
// store(unit, k, v). STRAIGHT-LINE, as `impl::sum`: a local unit's packs from every owner
// are loaded with no branch between them (a missing one reads the slot's first pack and
// is not stored), so they are in flight together.
template <typename T, int ngpus, typename Tiling, typename Store>
DINLINE void gather(const World<T, ngpus>& w, const PullSlot& s, const Tiling& t,
                    Store store) {
  using V = typename traits<T>::V;
  for (int l = t.first_local(); l < t.locals(); l = t.next_local(l)) {
    V g[kMaxRowPacks][ngpus];
#pragma unroll
    for (int i = 0; i < ngpus; ++i) {
      const V* from = impl::ptr(w, i, s.at(l, 0, 0), s.end() - s.at(l, 0, 0));
#pragma unroll
      for (int k = 0; k < kMaxRowPacks; ++k)
        g[k][i] = load_global(from + (s.at(l, k, t.lane()) - s.at(l, 0, 0)));
    }
#pragma unroll
    for (int k = 0; k < kMaxRowPacks; ++k)
#pragma unroll
      for (int i = 0; i < ngpus; ++i)
        if (t.has(t.unit(i, l), k)) store(t.unit(i, l), k, g[k][i]);
  }
}

}  // namespace pull

}  // namespace hip_comms::p2p
