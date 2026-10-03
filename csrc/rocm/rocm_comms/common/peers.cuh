// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE RANKS' MEMORY: every rank's buffers as plain tensors a tile loads and stores, and the types
// a kernel takes them in. A RANK'S MEMORY, two places:
//   ours, one allocation (Handle's symmetric memory), mapped by every peer at startup:
//     [ Signal | scratch | staging ]
//   each handed to a kernel as its own argument: every rank's signal block, scratch and
//   staging (PeerPtrs by value);
//   the caller's, one tensor a call:
//     [ input ]   read in place only when registered or captured, its peers' addresses a device
//                 table (`const PeerPtrs*`: a captured launch's are filled after the capture);
//                 otherwise a staged kernel copies it into the staging
// A buffer is read or written by tile_load / tile_store / peers_load, ordered only by
// barrier.cuh's `barrier`. The Signal block is barrier.cuh's, never read as data.

#pragma once

#ifndef HIP_COMMS_COMMON_INTERFACE
#error "include common/common.cuh, common's one interface, not its parts"
#endif

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>

#include "../machine/hardware.cuh"
#include "utils.cuh"

namespace hip_comms {

constexpr int kMaxRanks  = kMaxPeers;
// ONE SIGNAL SLOT PER BLOCK THE HARDWARE CAN HOLD RESIDENT, on any target built: the signal block
// never rules out a grid. Which grid is fast is select's (select.cuh).
constexpr int kMaxBlocks = kMaxResidentBlocks;

// One IPC allocation per rank holds the signal block AND the scratch: scratch is simply
// the bytes after the struct.
//
// TWO counter arrays, not one. A peer block can reach the second barrier while this one
// is still at the first, and with a single array the peer would write counter+1 while we
// busy-wait on counter. `seq` is the per-block monotonic sequence number.
//
// The barriers use the rest: `peer[r]` is the last world barrier rank r posted here,
// `arrive` and `gen` the grid barrier on this device, `epoch` the syncs this rank has
// completed.
struct Signal {
  alignas(128) uint32_t start[kMaxBlocks][kMaxRanks];
  alignas(128) uint32_t end[kMaxBlocks][kMaxRanks];
  alignas(128) uint32_t seq[kMaxBlocks];
  alignas(128) uint32_t peer[kMaxRanks];
  // `flag[r]`: the last flag rank r wrote here (write_flag).
  alignas(128) uint32_t flag[kMaxRanks];
  alignas(128) uint32_t arrive;
  alignas(128) uint32_t gen;
  alignas(128) uint32_t epoch;
};

struct __align__(16) PeerPtrs { void* p[kMaxRanks]; };
struct __align__(16) PeerSignals { Signal* s[kMaxRanks]; };


// RANK r'S BUFFER, its first element. `r` IS THE SAME ACROSS THE WAVE (a constant, or a per-wave
// rank): read from the first lane, the compiler knows it, and the pointer loads are scalar rather
// than one per lane. `rank_<kind>s` every rank's, how a kernel begins (before its start barrier);
// an input is read only.
namespace impl {
template <typename DTYPE, int WORLD>
DINLINE DTYPE* rank_of(const PeerPtrs& ptrs, int r) {
  r = __builtin_amdgcn_readfirstlane(r);
  void* at = ptrs.p[0];
#pragma unroll
  for (int k = 1; k < WORLD; ++k)
    if (r == k) at = ptrs.p[k];
  return static_cast<DTYPE*>(at);
}
template <typename DTYPE, int WORLD>
DINLINE std::array<DTYPE*, WORLD> every(const PeerPtrs& p) {
  std::array<DTYPE*, WORLD> all;
#pragma unroll
  for (int r = 0; r < WORLD; ++r) all[r] = rank_of<DTYPE, WORLD>(p, r);
  return all;
}
}  // namespace impl

template <typename DTYPE, int WORLD>
DINLINE const DTYPE* rank_input(const PeerPtrs& p, int r) { return impl::rank_of<DTYPE, WORLD>(p, r); }
template <typename DTYPE, int WORLD>
DINLINE DTYPE* rank_staging(const PeerPtrs& p, int r) { return impl::rank_of<DTYPE, WORLD>(p, r); }
template <typename DTYPE, int WORLD>
DINLINE DTYPE* rank_scratch(const PeerPtrs& p, int r) { return impl::rank_of<DTYPE, WORLD>(p, r); }
template <typename DTYPE, int WORLD>
DINLINE std::array<const DTYPE*, WORLD> rank_inputs(const PeerPtrs& p) {
  std::array<const DTYPE*, WORLD> all;
#pragma unroll
  for (int r = 0; r < WORLD; ++r) all[r] = impl::rank_of<DTYPE, WORLD>(p, r);
  return all;
}
template <typename DTYPE, int WORLD>
DINLINE std::array<DTYPE*, WORLD> rank_stagings(const PeerPtrs& p) { return impl::every<DTYPE, WORLD>(p); }
template <typename DTYPE, int WORLD>
DINLINE std::array<DTYPE*, WORLD> rank_scratches(const PeerPtrs& p) { return impl::every<DTYPE, WORLD>(p); }

}  // namespace hip_comms
