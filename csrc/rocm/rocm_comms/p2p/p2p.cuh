// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// P2P, THE PEER LAYER'S ONE INTERFACE: the only p2p header anything includes, and below,
// the whole of what a caller may use. Its parts (impl/) refuse to be included any other
// way, and what they keep in `p2p::impl` is theirs.
//
// p2p::                     all one GPU does with another; a peer's address never leaves it
//   ranks<T, ngpus>(p)             every rank's buffers for this launch, held for the kernel
//   read_input(ranks, r, i)        pack i of rank r's input
//   read_scratch(ranks, r, i)      pack i of what rank r left in its scratch
//   write_scratch(ranks, i, v)     pack i of this rank's scratch, for its peers to read
//   barrier<ngpus, Among, Ensure>(p)   Among::peers (this block and the same block on every
//                                  rank) or Among::grid (every block of this rank); Ensure::
//                                  launched (every peer's input is ready), visible (what was
//                                  written before is seen after), read (every peer is done
//                                  reading this rank)
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
