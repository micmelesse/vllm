// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// P2P, THE PEER LAYER'S ONE INTERFACE: the only p2p header anything includes, and below,
// the whole of what a caller may use. Its parts (impl/) refuse to be included any other
// way, and what they keep in `p2p::impl` is theirs.
//
// p2p::                     all one GPU does with another; a peer's address never leaves it
//   self<T, ngpus>(p)              this rank's own buffers
//   peers<T, ngpus>(p)             every rank's: with self, how a kernel begins, before its start
//                                  barrier
//   peer<T, ngpus>(p, r)           rank r's alone, r chosen at run time and the same across the wave
//   read_input(peer, i)            pack i of that rank's input
//   read_scratch(peer|self, i)     pack i of that rank's (or this rank's) scratch
//   write_scratch(peer|self, i, v) the same, written: through a peer, the push
//   barrier<ngpus, Among, Ensure>(p)   Among::peers (this block and the same block on every
//                                  rank) or Among::grid (every block of this rank); Ensure::
//                                  launched (every peer's input is ready), visible (what was
//                                  written before is seen after), read (every peer is done
//                                  reading this rank)
//   write_flag(p, peer, v)         v into `peer`'s signal slot for this rank
//   wait_flag(p, peer, v)          until `peer`'s flag here reaches v
//
// p2p::impl::Codec<T, kBits>   a group of kCodecGroupPacks packs on the wire: T itself (16) or
//                              QuickReduce's integers (8, 4) under one fp32 scale; for the
//                              quantized kernel that will use it
//
// p2p::host::                the host code (the ops, rocm_comms.cuh)
//   Group                          the one lifetime object: maps the peers' memory,
//                                  registers buffers, `dev_comm(input, bytes, stream)` per launch
//   IpcHandle, handle_and_offset(ptr)  a buffer's IPC handle
// and in p2p::, for sizing: Signal, PeerPtrs, kMaxBlocks, kMaxRanks.

#pragma once

#define HIP_COMMS_P2P_INTERFACE
#include "impl/peers.cuh"
#include "impl/core.cuh"
#include "impl/codec.cuh"
#include "impl/host.cuh"
#undef HIP_COMMS_P2P_INTERFACE
