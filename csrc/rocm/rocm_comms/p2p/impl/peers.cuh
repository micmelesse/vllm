// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE CONTRACT between p2p's two sides, behind p2p.cuh: what the host (host.cuh) maps and
// fills, and what a kernel (core.cuh, push.cuh) reads. Plain data only.

#pragma once

#ifndef HIP_COMMS_P2P_INTERFACE
#error "include p2p/p2p.cuh, p2p's one interface, not its parts"
#endif

#include <hip/hip_runtime.h>

#include <cstdint>

#include "../../utils.cuh"

namespace hip_comms::p2p {

constexpr int kMaxRanks  = 8;
constexpr int kMaxBlocks = 64;

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
  alignas(128) uint32_t arrive;
  alignas(128) uint32_t gen;
  alignas(128) uint32_t epoch;
};

struct __align__(16) PeerPtrs { void* p[kMaxRanks]; };
struct __align__(16) PeerSignals { Signal* s[kMaxRanks]; };

// WHAT A LAUNCH PASSES, by value: every rank's input (through a slot of the peer-pointer
// slab), every rank's signal block and scratch, and this launch's limits. Plain fields;
// `host::Group::peers` fills one per launch.
struct Peers {
  int rank;
  bool checked;                // bounds checks and random skew: the tests' mode
  const PeerPtrs* inputs;      // device memory: every rank's input for this launch
  PeerSignals signals;         // every rank's signal block; its scratch follows it
  Signal* self;                // this rank's
  int64_t input_packs;         // 16-byte packs of the input
  int64_t scratch_packs;       // of each rank's scratch
  uint64_t timeout_ticks;      // a wait longer than this traps
};

// THE SLOT SIZE, in packs, which the host needs to size scratch and push.cuh lays out on
// the device (they must agree). A push slot holds, per source rank, a group
// per lane per unit it holds (`held`: every unit for To::all, the local units for
// To::owners) at kbits (16: T itself), then the scales of a scaled codec.

inline int64_t push_slot_packs(int kbits, int64_t held, int lanes, int world) {
  const int64_t groups  = held * lanes;
  const int64_t payload = kSumBatch * 8 * kbits / 8 / 16;
  const int64_t scales  = kbits < 16 ? (groups + 3) / 4 : 0;
  return int64_t{world} * (groups * payload + scales);
}

}  // namespace hip_comms::p2p
