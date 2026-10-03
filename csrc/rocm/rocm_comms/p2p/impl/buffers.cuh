// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// A RANK'S MEMORY, behind p2p.cuh: its data buffers (input, staging, scratch), one view per kind
// with their reads and writes, ordered only by core.cuh's `barrier`; and its Signal block, whose
// only view hands out its counters, each touched only atomically. Data indices are in 16-byte packs
// of T.
//
// A RANK'S MEMORY, two places:
//   ours, one allocation (Handle's symmetric memory), mapped by every peer at startup:
//     [ Signal | scratch | staging ]
//   each handed to a kernel as its own argument: every rank's signal block, scratch and
//   staging (PeerPtrs by value);
//   the caller's, one tensor a call:
//     [ input ]   read in place only when registered or captured, its peers' addresses a device
//                 table (`const PeerPtrs*`: a captured launch's are filled after the capture);
//                 otherwise a staged kernel copies it into the staging
// The Signal block is its synchronization state (`Signals`): never read as data.

#pragma once

#ifndef HIP_COMMS_P2P_INTERFACE
#error "include p2p/p2p.cuh, p2p's one interface, not its parts"
#endif

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>

#include "../../common/common.cuh"
#include "peers.cuh"

namespace hip_comms::p2p {

namespace impl {

// RANK r'S BUFFER from the kernel's pointer array for its kind (every rank's input, scratch or
// staging), BY SELECT, never an index: a runtime index into a pointer array puts the array in
// scratch memory (seen in the ISA: 152 bytes a lane and a scratch load per read), where a constant
// index keeps the loads scalar and issued together.
template <typename T, int ngpus>
DINLINE typename traits<T>::V* rank_of(const PeerPtrs& ptrs, int r) {
  using V = typename traits<T>::V;
  V* at   = reinterpret_cast<V*>(ptrs.p[0]);
#pragma unroll
  for (int k = 1; k < ngpus; ++k)
    if (r == k) at = reinterpret_cast<V*>(ptrs.p[k]);
  return at;
}

}  // namespace impl

// A RANK'S BUFFER, one view per kind, every kind the same shape: a rank's input (what the in-place
// kernels read where it is), its staging (where a staged kernel copies its rank's input) and its
// scratch (a two-shot's partial sums). A rank is only its index. Made before the loops that use it,
// by `<kind>s(p)` for every rank or `<kind>(p, r)` for one chosen at run time (built from the
// kernel's arguments, never picked out of an array: a select over a local array compiles to an
// index and puts it in scratch, 144 B a lane). Nothing takes a runtime index into a set of ranks.
enum class Kind { input, staging, scratch };

template <typename T, int ngpus, Kind kKind>
class Buffer {
  using V = typename traits<T>::V;
  V* at_  = nullptr;

 public:
  DINLINE Buffer() = default;
  // `r` IS THE SAME ACROSS THE WAVE (a constant, or a per-wave rank): read from the first lane,
  // the compiler knows it, and the pointer loads are scalar rather than one per lane.
  DINLINE Buffer(const PeerPtrs& ptrs, int r) {
    at_ = impl::rank_of<T, ngpus>(ptrs, __builtin_amdgcn_readfirstlane(r));
  }
  DINLINE V* at() const { return at_; }
  // The rank's tensor, for a tile's load or store (common/memory.cuh).
  DINLINE T* data() const { return reinterpret_cast<T*>(at_); }
};

template <typename T, int ngpus>
using Input = Buffer<T, ngpus, Kind::input>;
template <typename T, int ngpus>
using Staging = Buffer<T, ngpus, Kind::staging>;
template <typename T, int ngpus>
using Scratch = Buffer<T, ngpus, Kind::scratch>;

// ONE RANK'S BUFFER OF A KIND, and EVERY RANK'S: how a kernel begins, before its start barrier so
// the pointer loads hide under the wait.
template <typename T, int ngpus, Kind kKind>
DINLINE std::array<Buffer<T, ngpus, kKind>, ngpus> every(const PeerPtrs& p) {
  std::array<Buffer<T, ngpus, kKind>, ngpus> all;
#pragma unroll
  for (int r = 0; r < ngpus; ++r) all[r] = Buffer<T, ngpus, kKind>(p, r);
  return all;
}

template <typename T, int ngpus>
DINLINE Input<T, ngpus> input(const PeerPtrs& p, int r) { return Input<T, ngpus>(p, r); }
template <typename T, int ngpus>
DINLINE Staging<T, ngpus> staging(const PeerPtrs& p, int r) { return Staging<T, ngpus>(p, r); }
template <typename T, int ngpus>
DINLINE Scratch<T, ngpus> scratch(const PeerPtrs& p, int r) { return Scratch<T, ngpus>(p, r); }
template <typename T, int ngpus>
DINLINE std::array<Input<T, ngpus>, ngpus> inputs(const PeerPtrs& p) {
  return every<T, ngpus, Kind::input>(p);
}
template <typename T, int ngpus>
DINLINE std::array<Staging<T, ngpus>, ngpus> stagings(const PeerPtrs& p) {
  return every<T, ngpus, Kind::staging>(p);
}
template <typename T, int ngpus>
DINLINE std::array<Scratch<T, ngpus>, ngpus> scratches(const PeerPtrs& p) {
  return every<T, ngpus, Kind::scratch>(p);
}

// PACK i OF A RANK'S BUFFER. Read after a barrier that made what its writer wrote visible: an
// input read in place (the pull receive), a peer's staged input, a peer's partial sums. Written for
// the peers to read after such a barrier: a rank's own staging and scratch, or (the push send, a
// plain store, since scratch is allocated uncached as aiter pushes into its own) a peer's scratch.
// What one block writes, the same block on the other side sees.
template <typename T, int ngpus>
DINLINE typename traits<T>::V read_input(const Input<T, ngpus>& b, int64_t i) {
  return thread_load(b.at() + i);
}
template <typename T, int ngpus>
DINLINE typename traits<T>::V read_staging(const Staging<T, ngpus>& b, int64_t i) {
  return thread_load(b.at() + i);
}
template <typename T, int ngpus>
DINLINE void write_staging(const Staging<T, ngpus>& b, int64_t i, const typename traits<T>::V& v) {
  thread_store(b.at() + i, v);
}
template <typename T, int ngpus>
DINLINE typename traits<T>::V read_scratch(const Scratch<T, ngpus>& b, int64_t i) {
  return thread_load(b.at() + i);
}
template <typename T, int ngpus>
DINLINE void write_scratch(const Scratch<T, ngpus>& b, int64_t i, const typename traits<T>::V& v) {
  thread_store(b.at() + i, v);
}

// ONE SIGNAL COUNTER: only atomic operations, each with its order and scope spelled at the call. A
// plain load or store of a counter cannot be written.
class Counter {
  uint32_t* at_;

 public:
  explicit DINLINE Counter(uint32_t* at) : at_(at) {}
  template <int kOrder, int kScope>
  DINLINE void store(uint32_t v) const {
    __scoped_atomic_store_n(at_, v, kOrder, kScope);
  }
  template <int kOrder, int kScope>
  DINLINE uint32_t load() const {
    return __scoped_atomic_load_n(at_, kOrder, kScope);
  }
  template <int kOrder, int kScope>
  DINLINE uint32_t fetch_add(uint32_t v) const {
    return __scoped_atomic_fetch_add(at_, v, kOrder, kScope);
  }
};

// A RANK'S SIGNAL BLOCK: its counters, handed out one at a time. `seq` (a block's own, written only
// by that block) and `epoch` (written only by the grid's last block to arrive) are plain: nothing
// else ever touches them.
class Signals {
  Signal* s_;

 public:
  explicit DINLINE Signals(Signal* s) : s_(s) {}
  // Block `block`'s pairing slot for rank `rank`, at a launch's start or at its other barriers.
  // AN INDEX KEEPS ITS CALLER'S TYPE (a block unsigned, as blockIdx.x; a rank as passed): an int
  // where the caller had an unsigned costs a sign extension in every kernel.
  template <typename R>
  DINLINE Counter start(unsigned block, R rank) const {
    return Counter(&s_->start[block][rank]);
  }
  template <typename R>
  DINLINE Counter end(unsigned block, R rank) const {
    return Counter(&s_->end[block][rank]);
  }
  // The grid barrier's: rank `rank`'s epoch here, the arrivals, the generation.
  DINLINE Counter peer(int rank) const { return Counter(&s_->peer[rank]); }
  DINLINE Counter arrive() const { return Counter(&s_->arrive); }
  DINLINE Counter gen() const { return Counter(&s_->gen); }
  // The last flag rank `rank` wrote here (write_flag).
  DINLINE Counter flag(int rank) const { return Counter(&s_->flag[rank]); }
  DINLINE uint32_t seq(unsigned block) const { return s_->seq[block]; }
  DINLINE void set_seq(unsigned block, uint32_t v) const { s_->seq[block] = v; }
  DINLINE uint32_t epoch() const { return s_->epoch; }
  DINLINE void set_epoch(uint32_t v) const { s_->epoch = v; }
};

namespace impl {

// Rank r's signal block, by select.
DINLINE Signal* signal_of(const PeerSignals& peer_signals, int r) {
  Signal* at = peer_signals.s[0];
#pragma unroll
  for (int k = 1; k < kMaxRanks; ++k)
    if (r == k) at = peer_signals.s[k];
  return at;
}

}  // namespace impl

// This rank's signal block; rank r's, r the same across the wave (by select); and thread i's peer
// i's, each thread its own (a per-lane load, as the pairing barrier issues it).
DINLINE Signals own_signals(Signal* self_signal) { return Signals(self_signal); }
DINLINE Signals signals(const PeerSignals& peer_signals, int r) {
  return Signals(impl::signal_of(peer_signals, r));
}
template <typename I>
DINLINE Signals lane_signals(const PeerSignals& peer_signals, I i) {
  return Signals(peer_signals.s[i]);
}

}  // namespace hip_comms::p2p
