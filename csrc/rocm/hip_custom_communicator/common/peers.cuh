// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE RANKS' MEMORY: every rank's buffers as plain tensors a tile loads and stores, and the types
// a kernel takes them in. A RANK'S MEMORY, two places:
//   ours, two allocations (Handle's symmetric memory), mapped by every peer at startup:
//     [ signals ]  and  [ scratch | staging ]
//   each handed to a kernel as its own argument, a device table of every rank's address (its
//   signals, scratch and staging) and the buffer's strides, written once when the handle opens;
//   the caller's, one tensor a call:
//     [ input ]   read in place only when registered or captured, its peers' addresses a device
//                 table (a captured launch's are filled after the capture);
//                 otherwise a staged kernel copies it into the staging
// A buffer is read or written by tile_load / tile_store / peers_load, ordered only by
// barrier.cuh's `barrier`. The signals are barrier.cuh's, never read as data.

#pragma once

#ifndef HIP_CUSTOM_COMMUNICATOR_COMMON_INTERFACE
#error "include common/interface.cuh, common's one interface, not its parts"
#endif

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>

#include "hardware.cuh"
#include "utils.cuh"

namespace hip_custom_communicator {

constexpr int kMaxRanks  = kMaxPeers;
// ONE SIGNAL SLOT PER BLOCK THE HARDWARE CAN HOLD RESIDENT, on any target built: the signal block
// never rules out a grid. Which grid is fast is select's (select.cuh).
constexpr int kMaxBlocks = kMaxResidentBlocks;

// WHERE A WAIT WAS when it gave up, the fault record's `where`; Python's `Where`, number for
// number. 0 is no fault.
enum class Where : uint32_t {
  start = 1,     // a launch's first barrier: the peer had not launched
  peer_barrier,  // a later barrier
};
constexpr int kNumWheres = 2;  // numbered from 1
constexpr const char* to_string(Where w) {
  switch (w) {
    case Where::start: return "start";
    case Where::peer_barrier: return "peer_barrier";
  }
  return "?";
}

// A RANK'S STATUS, in HOST-MAPPED memory (the handle's), so the host reads it and sets `abort` while
// a kernel still runs, and a graph replay, which runs none of our host code, still reports: a
// buffer of 32-bit words, one a field, `status[field]` at its stride (`status_stride_field`), the
// field a StatusField. 32-BIT, as the record's fields always were: 64-bit words cost the fault path
// a register pair a field and an occupancy step in 157 of 440 kernels (2026-10-13T16-59-48Z). A wait that outlives the timeout writes the fault once (the first block to
// claim `state` wins) and sets `abort`; every wait checks `abort` with its clock and leaves, so one
// fault ends the rank's kernel instead of each block spinning out its own timeout. The host sets
// `abort` to end a kernel it has given up on.
enum class StatusField : int {
  abort,          // nonzero: every wait on this rank leaves
  state,          // 0 clear, 1 being written, 2 written (read the rest only at 2)
  where,          // Where
  rank,           // this rank
  block,          // the block that gave up
  peer,           // the rank it waited for
  count,          // the flag it read last
  want,           // the value it waited for
  elapsed_low,    // how long it waited, in the device wall clock's ticks: the low word
  elapsed_high,   // and the high
};
constexpr int kStatusFields = 10;




// RANK r'S BUFFER, its first element, which rank_ptr and rank_ptrs (ptr.cuh) make a Ptr of: a
// kernel's arguments carry every rank's address of a buffer (a device table, `ptrs`) and this
// reads one. `r` IS THE SAME ACROSS THE WAVE (a constant, or a per-wave rank): read from the first
// lane, the compiler knows it, and the load is scalar rather than one per lane.
namespace impl {
template <typename T>
DINLINE T* rank_of(T* const* ptrs, int r) {
  return ptrs[__builtin_amdgcn_readfirstlane(r)];
}
}  // namespace impl

}  // namespace hip_custom_communicator
