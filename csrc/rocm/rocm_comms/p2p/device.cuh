// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE DEVICE SIDE'S ONE INTERFACE: the only p2p header a kernel includes, and below, the
// whole of what a kernel may use. Its parts (core.cuh, pull.cuh, push.cuh) refuse to be
// included any other way, and what they keep private stays theirs.
//
// TYPES
//   Peers                      what a launch passes, by value (peers.cuh)
//   Core<T, ngpus>::Inputs     every rank's input pointer, a value the kernel holds
//   Codec<T, kBits>            a push group on the wire: 16 (T itself), 8, 4 bits
//   Groups, Inbox<C, ngpus>    a push kernel's groups and its inbox layout
//   floats_of / packs_of       a group's packs as floats and back
//
// Core<T, ngpus>               every kernel
//   start(p)                   first; returns once every peer has launched this kernel
//   inputs(p)                  every rank's input, rotated by rank
//   put(p, peer, idx, v)       a pack into peer's scratch (a pull kernel's own)
//   ptr(p, peer, idx, n)       n packs of peer's scratch, checked once, for hot loops
//   peer_block_barrier(p)      this block and its same-numbered peer blocks: what each
//                              put before is visible to the others after
//   world_barrier(p)           the same for every block of every rank
//   grid_barrier(p)            every block of this rank, its own scratch only
//   close(p)                   last; after it this rank's input may be reused
//
// Pull<T, ngpus>               a rank reads its peers
//   reduce_flat(p, in, begin, end, store)       a flat range summed over ranks
//   sum_row(p, in, base, packs, v)              this thread's share of a row, summed
//   gather_flat(p, chunk, size, store)          every rank's slice, after the barrier
//   gather_rows<k>(p, chunk, rows, packs, ...)  every rank's rows, k regions
//
// Push<T, ngpus, Codec>        a rank writes into its peers' inboxes
//   mine_group / send / broadcast / read_inbox / reduce_inbox   one group
//   broadcast_rows / scatter_rows       phase 1 (one-shot / two-shot)
//   reduce_row                          a row's share out of the inbox
//   broadcast_row / gather_inbox_rows   two-shot phases 2 and 3

#pragma once

#define HIP_COMMS_P2P_DEVICE
#include "core.cuh"
#include "pull.cuh"
#include "push.cuh"
#undef HIP_COMMS_P2P_DEVICE
