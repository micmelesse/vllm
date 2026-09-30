// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// P2P, THE PEER LAYER'S ONE INTERFACE: the only p2p header anything includes, and below,
// the whole of what a caller may use. Its parts (impl/) refuse to be included any other
// way, and what they keep in `p2p::impl` is theirs.
//
// p2p::simple::              everything one GPU does with another
//   input<T>(p, rank)              that rank's input for this launch
//   scratch<T, ngpus>(p, rank)     that rank's scratch
//   start_sync<ngpus>(p)           first: every peer has launched, its input ready to read
//   end_sync<ngpus, kFinal>(p)     later: every peer has reached here; unless kFinal, what
//                                  this block wrote before is visible to the same block on
//                                  every peer after
//   grid_sync<ngpus>(p)            every block of this rank: what any wrote before is
//                                  visible to all after
//
// p2p::impl::Codec<T, kBits>   a group of kCodecGroupPacks packs on the wire: T itself (16) or
//                              QuickReduce's integers (8, 4) under one fp32 scale; for the
//                              quantized kernel that will use it
//
// p2p::host::                the host code (rocm_comms.cu)
//   Group                          the one lifetime object: maps the peers' memory,
//                                  registers buffers, `peers(input)` per launch
//   Handle, handle_and_offset(ptr) a tensor's IPC handle
// and in p2p::, for sizing: Signal, PeerPtrs, kMaxBlocks, kMaxRanks.

#pragma once

#define HIP_COMMS_P2P_INTERFACE
#include "impl/peers.cuh"
#include "impl/core.cuh"
#include "impl/codec.cuh"
#include "impl/host.cuh"
#undef HIP_COMMS_P2P_INTERFACE
