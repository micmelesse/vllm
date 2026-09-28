// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// P2P, THE PEER LAYER'S ONE INTERFACE: the only p2p header anything includes, and below,
// the whole of what a caller may use. Its parts (impl/) refuse to be included any other
// way, and what they keep in `p2p::impl` is theirs. `w` is the `World` `start` returns;
// T and the world size ride on it, so no call spells them. Indices are in 16-byte packs
// of T; a store callback is `store(position, v)` unless noted.
//
// TYPES
//   Peers                  what a launch passes its kernel, by value
//   World<T, ngpus>        a kernel's view: the Peers and every rank's input
//   Codec<T, kBits>        a push group on the wire: 16 (T itself), 8, 4 bits
//   Inbox<C, ngpus>        a push kernel's inbox; made by push::*_inbox, `.end()` where the
//                          next may start
//
// p2p::                    every kernel
//   start<T, ngpus>(p)             first; returns w once every peer has launched
//   put(w, peer, idx, v)           a pack into peer's scratch (a pull kernel's own)
//   ptr(w, peer, idx, n)           n packs of peer's scratch, checked once, for hot loops
//   peer_block_barrier(w)          this block and its same-numbered peer blocks: what
//                                  each wrote before is visible to the others after
//   world_barrier(w)               the same for every block of every rank
//   grid_barrier(w)                every block of this rank, its own scratch only
//   close(w)                       last, where peers read this rank's input late
//
// p2p::pull::              a rank reads its peers
//   reduce_buffer(w, begin, end, store)            a range of the buffer summed over ranks
//   gather_buffer(w, chunk, size, store)           every rank's slice, after the barrier
//   sum_row(w, base, packs, v)                     this thread's share of a row, summed
//   gather_rows<k>(w, chunk, rows, packs, region_packs, store(region, row, i, v))
//
// p2p::push::              a rank writes into its peers' inboxes
//   buffer_inbox<C>(w, span[, base]) / row_inbox<C>(w, rows[, base])
//   broadcast_buffer(w, box, size)                 one-shot phase 1
//   reduce_buffer(w, box, size, store)             one-shot phase 2
//   scatter_buffer(w, box, chunk, size)            two-shot phase 1
//   reduce_broadcast_slice(w, in, out, chunk, size)  two-shot phase 2
//   gather_buffer(w, box, chunk, size, store)      two-shot phase 3
//   broadcast_rows(w, box, rows, packs)            one-shot phase 1 of a row op
//   scatter_rows(w, box, chunk, rows, packs)       two-shot phase 1 of a row op
//   reduce_row(w, box, row, packs, sum)            a row's share out of the inbox
//   broadcast_row(w, box, row, packs, v)           two-shot phase 2 of a row op
//   gather_rows(w, box, chunk, rows, packs, store(row, i, v))  two-shot phase 3
//
// p2p::host::              the host code (rocm_comms.cu)
//   Group                          the one lifetime object: maps the peers' memory,
//                                  registers buffers, `peers(input)` per launch
//   Handle, handle_and_offset(ptr) a tensor's IPC handle
// and in p2p::, for sizing: Signal, PeerPtrs, kMaxBlocks, kMaxRanks, buffer_groups,
// inbox_packs.

#pragma once

#define HIP_COMMS_P2P_INTERFACE
#include "impl/peers.cuh"
#include "impl/core.cuh"
#include "impl/pull.cuh"
#include "impl/push.cuh"
#include "impl/host.cuh"
#undef HIP_COMMS_P2P_INTERFACE
