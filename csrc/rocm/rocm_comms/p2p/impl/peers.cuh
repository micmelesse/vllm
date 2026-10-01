// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE CONTRACT between p2p's two sides, behind p2p.cuh: what the host (host.cuh) maps and
// fills, and what a kernel (core.cuh) reads. Plain data only.

#pragma once

#ifndef HIP_COMMS_P2P_INTERFACE
#error "include p2p/p2p.cuh, p2p's one interface, not its parts"
#endif

#include <hip/hip_runtime.h>

#include <cstdint>

#include "../../common/common.cuh"
#include "../../machine/hardware.cuh"

namespace hip_comms::p2p {

constexpr int kMaxRanks  = kMaxPeers;
// ONE SIGNAL SLOT PER BLOCK THE HARDWARE CAN HOLD RESIDENT, on any target built: the signal block
// never rules out a grid. Which grid is fast is select's (impl/select.cuh).
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
  // `flag[r]`: the last flag rank r wrote here (p2p::write_flag).
  alignas(128) uint32_t flag[kMaxRanks];
  alignas(128) uint32_t arrive;
  alignas(128) uint32_t gen;
  alignas(128) uint32_t epoch;
};

struct __align__(16) PeerPtrs { void* p[kMaxRanks]; };
struct __align__(16) PeerSignals { Signal* s[kMaxRanks]; };

// THE DEVICE COMMUNICATOR, what a launch passes by value (NCCL's ncclDevComm): every rank's input
// (through a slot of the peer-pointer slab), every rank's signal block and scratch, and this
// launch's limits. Plain fields; `host::Group::dev_comm` fills one per launch.
struct DevComm {
  int rank;
  const PeerPtrs* inputs;      // device memory: every rank's input for this launch
  PeerSignals signals;         // every rank's signal block; its scratch follows it
  Signal* self;                // this rank's
  int64_t input_packs;         // 16-byte packs of the input
  int64_t scratch_packs;       // of each rank's scratch
  uint64_t timeout_ticks;      // a wait longer than this traps
};

}  // namespace hip_comms::p2p
