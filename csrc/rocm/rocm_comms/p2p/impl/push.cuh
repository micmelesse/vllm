// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// p2p::push, behind p2p.cuh: a rank reads only its own input and its own scratch, and
// what crosses a link is a store into a peer's SLOT, encoded by a codec (16 bits: T
// itself; 8, 4: QuickReduce's integers). Every phase takes a tiling (tiles::Rows or
// tiles::Buffer) and a unit of it.
//
// A thread's share of a unit is one GROUP, and crosses a link as one codec payload (and
// one scale); group idx x lanes + lane of a slot, idx the unit's index there (the unit
// itself when every rank holds every unit, its local index when only its owner does).
// Every phase gives a thread the same groups, so the same block on every peer is all a
// phase waits for. Only the payload covering a group's existing packs is sent.

#pragma once

#ifndef HIP_COMMS_P2P_INTERFACE
#error "include p2p/p2p.cuh, p2p's one interface, not its parts"
#endif

#include "core.cuh"

namespace hip_comms::p2p {

namespace impl {

// A group's kSumBatch packs as floats and back, rounding to T: what a push kernel reduces
// and encodes in, against the packs a row helper takes.
template <typename T>
DINLINE void floats_of(const typename traits<T>::V (&v)[kSumBatch],
                       float (&x)[kSumBatch * traits<T>::N]) {
  constexpr int N = traits<T>::N;
#pragma unroll
  for (int u = 0; u < kSumBatch; ++u)
#pragma unroll
    for (int j = 0; j < N; ++j) x[u * N + j] = static_cast<float>(v[u].d[j]);
}

template <typename T>
DINLINE void packs_of(const float (&x)[kSumBatch * traits<T>::N],
                      typename traits<T>::V (&v)[kSumBatch]) {
  constexpr int N = traits<T>::N;
#pragma unroll
  for (int u = 0; u < kSumBatch; ++u)
#pragma unroll
    for (int j = 0; j < N; ++j) v[u].d[j] = static_cast<T>(x[u * N + j]);
}

// ---------------------------------------------------------------------------------
// THE PUSH KERNELS' CODEC: what a group of kSumBatch packs (one thread's batch, 32 values
// of a 2-byte T) looks like on the wire. kBits 16 is T itself (no scale); 8 and 4 are
// QuickReduce's symmetric integers with one fp32 scale per group.
// ---------------------------------------------------------------------------------

template <typename T, int kBits>
struct Codec {
  using V                    = typename traits<T>::V;
  static constexpr int N     = traits<T>::N;
  static constexpr int kVals = kSumBatch * N;
  static_assert(sizeof(T) == 2, "the codec is built for 2-byte T");
  static_assert(kBits == 16 || kBits == 8 || kBits == 4, "16 (T), INT8 and INT4 are built");
  static constexpr bool kScaled = kBits < 16;
  // The payload of one group, in 16-byte packs: 32 values x kBits.
  static constexpr int kPayloadPacks = kVals * kBits / 8 / 16;
  static constexpr int kMax          = kScaled ? (1 << (kBits - 1)) - 1 : 0;

  // x -> payload; returns the scale, absmax / kMax (0 for an all-zero group; unused at 16).
  static DINLINE float encode(const float (&x)[kVals], V (&payload)[kPayloadPacks]) {
    if constexpr (!kScaled) {
#pragma unroll
      for (int i = 0; i < kVals; ++i) payload[i / N].d[i % N] = static_cast<T>(x[i]);
      return 1.0f;
    } else {
      float amax = 0.0f;
#pragma unroll
      for (int i = 0; i < kVals; ++i) amax = fmaxf(amax, fabsf(x[i]));
      const float scale = amax / kMax;
      const float inv   = amax > 0.0f ? kMax / amax : 0.0f;
      unsigned char bytes[kPayloadPacks * 16];
#pragma unroll
      for (int i = 0; i < kVals; ++i) {
        const int q =
            static_cast<int>(fminf(fmaxf(rintf(x[i] * inv), -kMax - 1.0f), kMax));
        if constexpr (kBits == 8) {
          bytes[i] = static_cast<unsigned char>(q & 0xFF);
        } else if (i % 2 == 0) {
          bytes[i / 2] = static_cast<unsigned char>(q & 0xF);
        } else {
          bytes[i / 2] |= static_cast<unsigned char>((q & 0xF) << 4);
        }
      }
      __builtin_memcpy(payload, bytes, sizeof(bytes));
      return scale;
    }
  }

  // payload, scale -> x.
  static DINLINE void decode(const V (&payload)[kPayloadPacks], float scale,
                             float (&x)[kVals]) {
    if constexpr (!kScaled) {
#pragma unroll
      for (int i = 0; i < kVals; ++i) x[i] = static_cast<float>(payload[i / N].d[i % N]);
    } else {
      unsigned char bytes[kPayloadPacks * 16];
      __builtin_memcpy(bytes, payload, sizeof(bytes));
#pragma unroll
      for (int i = 0; i < kVals; ++i) {
        int q;
        if constexpr (kBits == 8) {
          q = static_cast<signed char>(bytes[i]);
        } else {
          const int nib = (bytes[i / 2] >> (4 * (i % 2))) & 0xF;
          q             = nib >= 8 ? nib - 16 : nib;
        }
        x[i] = static_cast<float>(q) * scale;
      }
    }
  }
};

}  // namespace impl

// Where `push::scatter` sends a row: to its owner (a two-shot) or to every rank (a
// one-shot).
enum class To { owners, all };

// A PUSH SLOT: a region of every rank's scratch that each rank pushes into (a slot per
// source rank inside it), at one codec. Made by `push::slot`, passed back.
template <class C, int ngpus>
struct PushSlot {
  To to;
  int groups;   // per source: the units it holds x lanes
  int base;     // in packs of the scratch
  int per_src;  // packs per source: every group's payload, then (scaled) their scales
  DINLINE int payload(int src, int g) const {
    return base + src * per_src + g * C::kPayloadPacks;
  }
  DINLINE int scale(int src, int g) const {
    return 4 * (base + src * per_src + groups * C::kPayloadPacks) + g;
  }
  DINLINE int end() const { return base + ngpus * per_src; }
};

namespace impl {

template <class C>
DINLINE int payload_packs(int n) {
  return (n * C::kPayloadPacks + kSumBatch - 1) / kSumBatch;
}

// This thread's share of unit u of this rank's input, as floats (missing packs zero).
template <class C, typename T, int ngpus, typename Tiling>
DINLINE void mine_unit(const World<T, ngpus>& w, const Tiling& t, int u,
                       float (&x)[C::kVals]) {
  typename traits<T>::V v[kSumBatch];
#pragma unroll
  for (int k = 0; k < kSumBatch; ++k)
    v[k] = t.has(u, k) ? mine(w, t.pos(u, k)) : typename traits<T>::V{};
  floats_of<T>(v, x);
}

template <class C, typename T, int ngpus>
DINLINE void push_payload(const World<T, ngpus>& w, int peer, const PushSlot<C, ngpus>& s,
                          int g, int n, const typename traits<T>::V (&q)[C::kPayloadPacks],
                          float scale) {
  if (n == 0) return;
  const int at   = s.payload(w.peers.rank, g);
  const int sent = payload_packs<C>(n);
#pragma unroll
  for (int k = 0; k < C::kPayloadPacks; ++k)
    if (k < sent) put(w, peer, at + k, q[k]);
  if constexpr (C::kScaled) put_float(w, peer, s.scale(w.peers.rank, g), scale);
}

// Group g encoded into `peer`'s slot for this rank, its first n packs.
template <class C, typename T, int ngpus>
DINLINE void send(const World<T, ngpus>& w, int peer, const PushSlot<C, ngpus>& s, int g,
                  int n, const float (&x)[C::kVals]) {
  typename traits<T>::V q[C::kPayloadPacks];
  const float scale = C::encode(x, q);
  push_payload(w, peer, s, g, n, q, scale);
}

// The same into every rank's slot, encoded once.
template <class C, typename T, int ngpus>
DINLINE void broadcast(const World<T, ngpus>& w, const PushSlot<C, ngpus>& s, int g, int n,
                       const float (&x)[C::kVals]) {
  typename traits<T>::V q[C::kPayloadPacks];
  const float scale = C::encode(x, q);
#pragma unroll
  for (int d = 0; d < ngpus; ++d) push_payload(w, d, s, g, n, q, scale);
}

// Group g of `src`'s part of this rank's slot, its first n packs, decoded (the rest are
// garbage and never used).
template <class C, typename T, int ngpus>
DINLINE void read(const World<T, ngpus>& w, const PushSlot<C, ngpus>& s, int src, int g,
                  int n, float (&x)[C::kVals]) {
  typename traits<T>::V q[C::kPayloadPacks];
  const int at   = s.payload(src, g);
  const int sent = payload_packs<C>(n);
#pragma unroll
  for (int k = 0; k < C::kPayloadPacks; ++k)
    q[k] = k < sent ? get_pushed(w, at + k) : typename traits<T>::V{};
  float scale = 1.0f;
  if constexpr (C::kScaled)
    if (n > 0) scale = get_float(w, w.peers.rank, s.scale(src, g));
  C::decode(q, scale, x);
}

// This thread's group for a unit's index `idx` in a slot.
template <typename Tiling>
DINLINE int group(const Tiling& t, int idx) {
  return idx * t.lanes() + t.lane();
}

}  // namespace impl

namespace push {

// A slot for the tiling's units at kBits, from pack `base` of the scratch.
template <int kBits, typename T, int ngpus, typename Tiling>
DINLINE PushSlot<impl::Codec<T, kBits>, ngpus> slot(const World<T, ngpus>&, const Tiling& t,
                                                    To to, int base = 0) {
  using C          = impl::Codec<T, kBits>;
  const int groups = (to == To::all ? t.units() : t.locals()) * t.lanes();
  const int per    = groups * C::kPayloadPacks + (C::kScaled ? (groups + 3) / 4 : 0);
  return {to, groups, base, per};
}

// The next slot, after `prev` (any slot) in the scratch.
template <int kBits, typename T, int ngpus, typename Tiling, typename Prev>
DINLINE PushSlot<impl::Codec<T, kBits>, ngpus> slot(const World<T, ngpus>& w,
                                                    const Tiling& t, To to,
                                                    const Prev& prev) {
  return slot<kBits>(w, t, to, prev.end());
}

// PHASE 1: this rank's input units into the slot: each to its owner (To::owners), or every
// unit to every rank (To::all). A thread takes the units (local units, for owners) it
// will reduce.
template <class C, typename T, int ngpus, typename Tiling>
DINLINE void scatter(const World<T, ngpus>& w, const PushSlot<C, ngpus>& s,
                     const Tiling& t) {
  if (s.to == To::all) {
    for (int u = t.first(); u < t.end(); u = t.next(u)) {
      float x[C::kVals];
      impl::mine_unit<C>(w, t, u, x);
      impl::broadcast(w, s, impl::group(t, u), tiles::members(t, u), x);
    }
    return;
  }
  for (int l = t.first_local(); l < t.locals(); l = t.next_local(l)) {
    for (int d = 0; d < ngpus; ++d) {
      const int u = t.unit(d, l);
      float x[C::kVals];
      impl::mine_unit<C>(w, t, u, x);
      impl::send(w, d, s, impl::group(t, l), tiles::members(t, u), x);
    }
  }
}

// This thread's share of unit u summed over every source in the slot, in rank order
// (every rank the same bits), rounded to T: what `pull::reduce` gives a pull kernel.
template <class C, typename T, int ngpus, typename Tiling>
DINLINE void reduce(const World<T, ngpus>& w, const PushSlot<C, ngpus>& s, const Tiling& t,
                    int u, typename traits<T>::V (&v)[kMaxRowPacks]) {
  const int idx = s.to == To::all ? u : t.local(u);
  const int n   = tiles::members(t, u);
  float acc[C::kVals] = {};
  for (int src = 0; src < ngpus; ++src) {
    float x[C::kVals];
    impl::read(w, s, src, impl::group(t, idx), n, x);
#pragma unroll
    for (int i = 0; i < C::kVals; ++i) acc[i] += x[i];
  }
  impl::packs_of<T>(acc, v);
}

// This thread's share of owned unit u's result, encoded once, into every rank's slot, for
// every rank to gather after a peer_barrier.
template <class C, typename T, int ngpus, typename Tiling>
DINLINE void share(const World<T, ngpus>& w, const PushSlot<C, ngpus>& s, const Tiling& t,
                   int u, const typename traits<T>::V (&v)[kMaxRowPacks]) {
  float x[C::kVals];
  impl::floats_of<T>(v, x);
  impl::broadcast(w, s, impl::group(t, t.local(u)), tiles::members(t, u), x);
}

// After a peer_barrier: every owner's shared units out of this rank's slot; a thread
// takes the local units it shared. store(unit, k, v).
template <class C, typename T, int ngpus, typename Tiling, typename Store>
DINLINE void gather(const World<T, ngpus>& w, const PushSlot<C, ngpus>& s, const Tiling& t,
                    Store store) {
  for (int l = t.first_local(); l < t.locals(); l = t.next_local(l)) {
    for (int src = 0; src < ngpus; ++src) {
      const int u = t.unit(src, l);
      const int n = tiles::members(t, u);
      if (n == 0) continue;
      float x[C::kVals];
      impl::read(w, s, src, impl::group(t, l), n, x);
      typename traits<T>::V v[kSumBatch];
      impl::packs_of<T>(x, v);
#pragma unroll
      for (int k = 0; k < kSumBatch; ++k)
        if (k < n) store(u, k, v[k]);
    }
  }
}

}  // namespace push

}  // namespace hip_comms::p2p
