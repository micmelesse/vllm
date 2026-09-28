// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// P2P, THE PEER LAYER'S ONE INTERFACE: the only p2p header anything includes, and below,
// the whole of what a caller may use. Its parts (impl/) refuse to be included any other
// way, and what they keep in `p2p::impl` is theirs.
//
// Every phase takes a TILING `t` (utils.cuh: tiles::Rows for a fused op's rows,
// tiles::Buffer for a plain buffer) and a UNIT `u` of it, a thread's share v[k] at a time;
// a kernel walks its units with t.first / t.end / t.next (a two-shot, its own: rank
// argument). `w` is the World `start` returns; T and the world size ride on it. A SLOT is
// where a phase leaves data for a later one: made by `slot`, passed back, never looked
// into; slots are laid end to end (`slot(w, t, ..., previous_slot)`).
//
// p2p::                      every kernel
//   start<T, ngpus>(p)             first; returns w once every peer has launched
//   peer_barrier(w)                this block and the same block on every peer: what
//                                  each wrote before is visible to the others after
//   grid_barrier(w)                every block of this rank
//   world_barrier(w)               every block of every rank
//   close(w)                       last, in a kernel whose peers read its input late
//
// p2p::pull::                a rank reads its peers
//   reduce(w, t, u, v)                 this thread's share of a unit, summed over ranks
//   slot(w, t[, prev])                 this rank's units of a two-shot, in its scratch
//   share(w, slot, t, u, v)            an owned unit's result into the slot
//   gather(w, slot, t, store)          after a peer_barrier, every owner's shared units
//
// p2p::push::                a rank writes into its peers' slots
//   slot<kBits>(w, t, To, [prev])      To::owners (two-shot) or To::all (one-shot);
//                                      kBits 16 (T itself), 8 or 4
//   scatter(w, slot, t)                this rank's input units into the slot
//   reduce(w, slot, t, u, v)           after a peer_barrier, a unit summed out of the slot
//   share(w, slot, t, u, v)            an owned unit's result into every rank's slot
//   gather(w, slot, t, store)          after a peer_barrier, every owner's shared units
//
//   store(u, k, v): pack k of this thread's share of unit u, at t.pos(u, k)
//
// p2p::host::                the host code (rocm_comms.cu)
//   Group                          the one lifetime object: maps the peers' memory,
//                                  registers buffers, `peers(input)` per launch
//   Handle, handle_and_offset(ptr) a tensor's IPC handle
// and in p2p::, for sizing: Signal, PeerPtrs, kMaxBlocks, kMaxRanks, pull_slot_packs,
// push_slot_packs.

#pragma once

#define HIP_COMMS_P2P_INTERFACE
#include "impl/peers.cuh"
#include "impl/core.cuh"
#include "impl/pull.cuh"
#include "impl/push.cuh"
#include "impl/host.cuh"
#undef HIP_COMMS_P2P_INTERFACE
