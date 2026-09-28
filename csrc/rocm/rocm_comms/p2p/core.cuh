// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE DEVICE LAYER'S PRIMITIVES: peer memory, the rank's own input, and the barriers that
// order them. Free functions over the `Peers` a launch passes; `Core<T, ngpus>` only names
// the element type and the world size once, and holds nothing. The pull and push patterns
// (pull.cuh, push.cuh) are built on these.

#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "../utils.cuh"

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
// `ipc::Group::peers` fills one per launch.
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

// =================================================================================
// THE PRIMITIVES. Indices are in 16-byte packs of T.
//
//   start(p)                   first; returns once every peer has launched this kernel
//   inputs(p)                  every rank's input pointer, rotated by rank: a value the
//                              kernel holds and passes to what reads inputs
//   sum<k>(p, in, idx, ...)    input packs summed over ranks, fp32, rounded once, batched
//   mine(p, in, idx)           this rank's own input pack
//   put / get (p, peer, idx)   a pack into / out of peer's scratch
//   ptr(p, peer, idx, n)       a direct pointer to n packs of it, checked once
//   get_pushed(p, idx)         a pack peers pushed here, read past the caches
//   put_float / get_float      a float of peer's scratch (a codec's scales)
//   peer_block_barrier(p)      this block and its same-numbered peer blocks: what each put
//                              before is visible to the others after
//   world_barrier(p)           the same for every block of every rank
//   grid_barrier(p)            every block of this rank, its own scratch only
//   close(p)                   last; after it this rank's input may be reused (unneeded
//                              when the input is last read before a peer_block_barrier)
//
// A wait that outlives the timeout prints where it was and traps, so a hang is an error.
// Checked, every index is bounds-checked and every wait is skewed by a random per-block
// delay, so a race shows on every run.
// =================================================================================

template <typename T, int ngpus>
struct Core {
  using V = typename traits<T>::V;

  // Every rank's input, index 0 this rank's. ROTATED by rank, so the ranks do not all read
  // rank 0 first; each rank then sums in a different order, so a pulled sum agrees across
  // ranks to one ULP of T, not bitwise.
  struct Inputs {
    const V* p[ngpus];
  };

  static DINLINE void start(const Peers& p) {
    skew(p);
    pair_blocks<false>(p, true);
  }

  static DINLINE Inputs inputs(const Peers& p) {
    Inputs in;
#pragma unroll
    for (int i = 0; i < ngpus; ++i)
      in.p[i] = reinterpret_cast<const V*>(p.inputs->p[(p.rank + i) % ngpus]);
    return in;
  }

  // kBatch packs at once, idx + u * stride for u < kBatch (those at or past `limit` are
  // skipped): every load from every peer is issued before any is added, so kBatch x ngpus
  // are in flight rather than ngpus.
  template <int kBatch>
  static DINLINE void sum(const Peers& p, const Inputs& in, int64_t idx, int64_t stride,
                          int64_t limit, V (&out)[kBatch]) {
    constexpr int N = traits<T>::N;
    V raw[kBatch][ngpus];
#pragma unroll
    for (int u = 0; u < kBatch; ++u) {
      const int64_t at = idx + u * stride;
      if (at < limit) {
        check(p, at < p.input_packs, "sum", -1, at, p.input_packs);
#pragma unroll
        for (int i = 0; i < ngpus; ++i) raw[u][i] = load_global(in.p[i] + at);
      }
    }
#pragma unroll
    for (int u = 0; u < kBatch; ++u) {
      float acc[N];
#pragma unroll
      for (int j = 0; j < N; ++j) acc[j] = static_cast<float>(raw[u][0].d[j]);
#pragma unroll
      for (int i = 1; i < ngpus; ++i)
#pragma unroll
        for (int j = 0; j < N; ++j) acc[j] += static_cast<float>(raw[u][i].d[j]);
#pragma unroll
      for (int j = 0; j < N; ++j) out[u].d[j] = static_cast<T>(acc[j]);
    }
  }

  static DINLINE V mine(const Peers& p, const Inputs& in, int64_t idx) {
    check(p, idx < p.input_packs, "mine", -1, idx, p.input_packs);
    return load_global(in.p[0] + idx);
  }

  static DINLINE void put(const Peers& p, int peer, int64_t idx, const V& v) {
    check(p, peer >= 0 && peer < ngpus && idx < p.scratch_packs, "put", peer, idx,
          p.scratch_packs);
    store_global(scratch(p, peer) + idx, v);
  }

  static DINLINE V get(const Peers& p, int peer, int64_t idx) {
    check(p, peer >= 0 && peer < ngpus && idx < p.scratch_packs, "get", peer, idx,
          p.scratch_packs);
    return load_global(scratch(p, peer) + idx);
  }

  // SHMEM's `shmem_ptr`: a hot loop reads through this rather than paying `get`'s check on
  // every load, which keeps the loads free to issue back to back.
  static DINLINE const V* ptr(const Peers& p, int peer, int64_t idx, int64_t n) {
    check(p, peer >= 0 && peer < ngpus && idx + n <= p.scratch_packs, "ptr", peer, idx + n,
          p.scratch_packs);
    return scratch(p, peer) + idx;
  }

  static DINLINE V get_pushed(const Peers& p, int64_t idx) {
    check(p, idx < p.scratch_packs, "get_pushed", p.rank, idx, p.scratch_packs);
    return load_uncached(scratch(p, p.rank) + idx);
  }

  // `idx` in floats.
  static DINLINE void put_float(const Peers& p, int peer, int64_t idx, float v) {
    check(p, peer >= 0 && peer < ngpus && idx < 4 * p.scratch_packs, "put_float", peer,
          idx, 4 * p.scratch_packs);
    __scoped_atomic_store_n(reinterpret_cast<uint32_t*>(scratch(p, peer)) + idx,
                            __builtin_bit_cast(uint32_t, v), __ATOMIC_RELAXED,
                            __MEMORY_SCOPE_SYSTEM);
  }

  static DINLINE float get_float(const Peers& p, int peer, int64_t idx) {
    check(p, peer >= 0 && peer < ngpus && idx < 4 * p.scratch_packs, "get_float", peer,
          idx, 4 * p.scratch_packs);
    const auto* at = reinterpret_cast<const uint32_t*>(scratch(p, peer)) + idx;
    return __builtin_bit_cast(float, __scoped_atomic_load_n(at, __ATOMIC_RELAXED,
                                                            __MEMORY_SCOPE_SYSTEM));
  }

  // This block and the same-numbered block on every peer, and no other block: one peer
  // write each, where `world_barrier` waits for the whole grid. A get after it may read
  // ONLY what the same-numbered block on that peer put, so both phases must give each
  // block the same indices (vLLM's custom all-reduce, and the rule its two-stage kernel
  // states).
  static DINLINE void peer_block_barrier(const Peers& p) {
    skew(p);
    pair_blocks<true>(p, false);
  }

  static DINLINE void world_barrier(const Peers& p) { barrier<true>(p); }

  // Every block of THIS rank's kernel: puts to our own scratch before it are visible to our
  // own gets after it. Cheaper than `world_barrier`, and wrong for anything a peer put.
  static DINLINE void grid_barrier(const Peers& p) { barrier<false>(p); }

  static DINLINE void close(const Peers& p) {
    skew(p);
    pair_blocks<false>(p, false);
  }

  // ---------------------------------------------------------------------------------
  // The layer's own: what the primitives above are made of.
  // ---------------------------------------------------------------------------------

  // Rank `peer`'s scratch, the bytes after its signal block. BY SELECT, NOT an index into
  // the pointer array: a runtime index into a register array moves it to scratch memory.
  static DINLINE V* scratch(const Peers& p, int peer) {
    V* at = reinterpret_cast<V*>(p.signals.s[0] + 1);
#pragma unroll
    for (int i = 1; i < ngpus; ++i)
      if (peer == i) at = reinterpret_cast<V*>(p.signals.s[i] + 1);
    return at;
  }

  // The grid on this device, then (kPeers) one exchange with the peers by the last block
  // to arrive, then the grid released. ONE FENCE PER BLOCK, by thread 0 after the block
  // barrier: every wave first waits for its own stores (`wait_stores`), and a release
  // writes back the whole L2, so one covers the block where one per thread wrote it back
  // 512 times.
  template <bool kPeers>
  static DINLINE void barrier(const Peers& p) {
    constexpr int kScope = kPeers ? __MEMORY_SCOPE_SYSTEM : __MEMORY_SCOPE_DEVICE;
    skew(p);
    wait_stores();
    __syncthreads();
    if (threadIdx.x == 0) {
      fence<__ATOMIC_RELEASE, kScope>();
      Signal* self     = p.self;
      const uint32_t g = __scoped_atomic_load_n(&self->gen, __ATOMIC_ACQUIRE, kScope);
      if (__scoped_atomic_fetch_add(&self->arrive, 1u, __ATOMIC_ACQ_REL,
                                    __MEMORY_SCOPE_DEVICE) == gridDim.x - 1) {
        __scoped_atomic_store_n(&self->arrive, 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
        if constexpr (kPeers) {
          const uint32_t e = self->epoch + 1;
          self->epoch      = e;
          fence<__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM>();
#pragma unroll
          for (int i = 0; i < ngpus; ++i)
            __scoped_atomic_store_n(&p.signals.s[i]->peer[p.rank], e, __ATOMIC_RELAXED,
                                    __MEMORY_SCOPE_SYSTEM);
#pragma unroll
          for (int i = 0; i < ngpus; ++i)
            wait<true, __MEMORY_SCOPE_SYSTEM>(p, &self->peer[i], e, "world_barrier: peer",
                                              i);
        }
        __scoped_atomic_fetch_add(&self->gen, 1u, __ATOMIC_RELEASE, kScope);
      } else {
        wait<true, kScope>(p, &self->gen, g + 1,
                           kPeers ? "world_barrier" : "grid_barrier", -1);
      }
    }
    __syncthreads();
  }

  // Block b waits for block b on every rank, and for no other block: enough at the ends,
  // where it says "every peer has launched" or "every peer is done reading me", and not
  // enough between phases, which is what `world_barrier` is for. kOrdered: the store
  // releases and the wait acquires, so what the block put before is visible to its peers'
  // same-numbered block after (a peer_block_barrier). Unordered, it only says when
  // (start: every peer has launched; close: every peer is done reading us).
  template <bool kOrdered>
  static DINLINE void pair_blocks(const Peers& p, bool start) {
    if (!start) {
      if constexpr (kOrdered) wait_stores();
      __syncthreads();
    }
    Signal* self     = p.self;
    const uint32_t f = self->seq[blockIdx.x] + 1;
    if (threadIdx.x < ngpus) {
      uint32_t* theirs = start ? &p.signals.s[threadIdx.x]->start[blockIdx.x][p.rank]
                               : &p.signals.s[threadIdx.x]->end[blockIdx.x][p.rank];
      uint32_t* mine   = start ? &self->start[blockIdx.x][threadIdx.x]
                               : &self->end[blockIdx.x][threadIdx.x];
      __scoped_atomic_store_n(theirs, f, kOrdered ? __ATOMIC_RELEASE : __ATOMIC_RELAXED,
                              __MEMORY_SCOPE_SYSTEM);
      wait<kOrdered, __MEMORY_SCOPE_DEVICE>(p, mine, f, start ? "start" : "peer barrier",
                                            threadIdx.x);
    }
    __syncthreads();
    if (threadIdx.x == 0) self->seq[blockIdx.x] = f;
  }

  // EVERY WAVE'S STORES DONE before a block barrier that one wave then releases:
  // `__syncthreads` waits for none (gfx9 emits no vmcnt wait before s_barrier; seen in the
  // ISA), so the releasing wave's fence would cover only its own stores, and another
  // wave's could still be in flight when the flag lands -- a push kernel's remote stores
  // above all.
  static DINLINE void wait_stores() { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); }

  // The builtin takes the ordering as a literal, so each case is spelled out.
  template <int kOrder, int kScope>
  static DINLINE void fence() {
    constexpr bool system = kScope == __MEMORY_SCOPE_SYSTEM;
    if constexpr (kOrder == __ATOMIC_RELEASE) {
      if constexpr (system) __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
      else __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    } else {
      if constexpr (system) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
      else __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
  }

  // Spins relaxed and acquires once, after: an acquire per poll would invalidate the
  // caches on every iteration of every spinning block.
  template <bool kAcquire, int kScope>
  static DINLINE void wait(const Peers& p, const uint32_t* flag, uint32_t want,
                           const char* what, int peer) {
    const uint64_t t0 = wall_clock64();
    uint32_t seen;
    while ((seen = __scoped_atomic_load_n(flag, __ATOMIC_RELAXED, kScope)) < want) {
      if (wall_clock64() - t0 > p.timeout_ticks) {
        printf("rocm_comms: rank %d block %d timed out in %s, peer %d: flag %u, want %u\n",
               p.rank, blockIdx.x, what, peer, seen, want);
        __builtin_trap();
      }
    }
    if constexpr (kAcquire) fence<__ATOMIC_ACQUIRE, kScope>();
  }

  static DINLINE void check(const Peers& p, bool ok, const char* what, int peer,
                            int64_t idx, int64_t limit) {
    if (p.checked && (!ok || idx < 0)) {
      printf("rocm_comms: rank %d block %d thread %d: %s(peer %d, idx %lld) outside "
             "[0, %lld)\n",
             p.rank, blockIdx.x, threadIdx.x, what, peer, static_cast<long long>(idx),
             static_cast<long long>(limit));
      __builtin_trap();
    }
  }

  // Checked only: up to ~32 x 8K cycles, different per rank, block and peer barrier (the
  // block's sequence number, which start, peer_block_barrier and close advance).
  static DINLINE void skew(const Peers& p) {
    if (!p.checked) return;
    uint32_t h = static_cast<uint32_t>(p.rank) * 73856093u ^ blockIdx.x * 19349663u ^
                 p.self->seq[blockIdx.x] * 83492791u;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    for (uint32_t n = (h ^ (h >> 15)) % 32; n > 0; --n) __builtin_amdgcn_s_sleep(127);
  }
};

}  // namespace hip_comms::p2p
