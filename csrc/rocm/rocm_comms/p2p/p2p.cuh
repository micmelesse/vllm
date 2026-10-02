// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// P2P, THE PEER LAYER'S ONE INTERFACE: the only p2p header anything includes, and below,
// the whole of what a caller may use. Its parts (impl/) refuse to be included any other
// way, and what they keep in `p2p::impl` is theirs.
//
// p2p::                     all one GPU does with another; a peer's address never leaves it
//   A RANK'S BUFFERS, one view per kind, a rank only its index (this rank's own: r = p.rank):
//   input<T, ngpus>(p, r), inputs<T, ngpus>(p)          its input, read where it is (in place)
//   staging<T, ngpus>(p, r), stagings<T, ngpus>(p)      its staging: a staged kernel copies its
//                                                       rank's input there
//   scratch<T, ngpus>(p, r), scratches<T, ngpus>(p)     its scratch: a two-shot's partial sums
//                                  `<kind>s` every rank's, how a kernel begins (before its start
//                                  barrier); `<kind>(p, r)` one, r the same across the wave
//   read_input(b, i)               pack i of an input
//   read_staging(b, i), write_staging(b, i, v)   pack i of a staging: a peer's read, this rank's
//                                  own written
//   read_scratch(b, i), write_scratch(b, i, v)   pack i of a scratch, either rank's: written to
//                                  a peer's, the push
//   barrier<ngpus, Among, Ensure>(p)   Among::peers (this block and the same block on every
//                                  rank) or Among::grid (every block of this rank); Ensure::
//                                  launched (every peer's input is ready), visible (what was
//                                  written before is seen after), read (every peer is done
//                                  reading this rank)
//   write_flag(p, peer, v)         v into `peer`'s signal slot for this rank
//   wait_flag(p, peer, v)          until `peer`'s flag here reaches v
//   own_signals(p), signals(p, r)  a rank's Signal block, as `Signals`: its counters (`Counter`),
//                                  each touched only atomically; the barriers' and flags' only
//                                  way in
//
// p2p::impl::Codec<T, kBits>   a group of kCodecGroupPacks packs on the wire: T itself (16) or
//                              QuickReduce's integers (8, 4) under one fp32 scale; for the
//                              quantized kernel that will use it
//
// p2p::host::                the host code (the ops, rocm_comms.cuh)
//   Handle                         the one lifetime object: maps the peers' memory,
//                                  registers buffers, `dev_comm(input, bytes, stream)` per launch
//                                  (`dev_comm_staged(bytes)` for a staged kernel)
//   IpcHandle, handle_and_offset(ptr)  a buffer's IPC handle
// and in p2p::, for sizing: Signal, PeerPtrs, kMaxBlocks, kMaxRanks.

#pragma once

#define HIP_COMMS_P2P_INTERFACE
#include "impl/peers.cuh"
#include "impl/buffers.cuh"
#include "impl/core.cuh"
#include "impl/codec.cuh"
#include "impl/host.cuh"
#undef HIP_COMMS_P2P_INTERFACE
