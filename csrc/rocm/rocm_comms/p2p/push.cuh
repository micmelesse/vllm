// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE PUSH PATTERNS: a rank reads only its own input and its own scratch, and what
// crosses a link is a store into a peer's INBOX, encoded by a Codec (16 bits: T itself;
// 8, 4: QuickReduce's integers). Built on core.cuh; `Push<T, ngpus, C>` holds nothing.

#pragma once

#include "core.cuh"

namespace hip_comms::p2p {

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

// =================================================================================
// THE UNITS. A GROUP is one thread's kSumBatch packs of a span, positions
// lane + (j * kSumBatch + u) * stride, and it crosses a link as one Codec payload (and
// one scale). Every phase of a push kernel gives a thread the same groups, so the
// same-numbered peer blocks are all a phase waits for. Members that exist are a prefix
// (u < members), and only the payload covering them is sent.
// =================================================================================

struct Groups {
  int span, lane, stride, first, iters;
  DINLINE Groups(int span, int lane, int stride, int first)
      : span(span),
        lane(lane),
        stride(stride),
        first(first),
        iters((span + stride * kSumBatch - 1) / (stride * kSumBatch)) {}
  // Grid-strided over a flat span: the flat kernels.
  static DINLINE Groups flat(int span) {
    return Groups(span, blockIdx.x * blockDim.x + threadIdx.x, gridDim.x * blockDim.x, 0);
  }
  // This thread's share of one row, the packs a row helper holds (one group, j = 0), as
  // group local_row x blockDim + threadIdx of an inbox.
  static DINLINE Groups row(int packs, int local_row) {
    return Groups(packs, threadIdx.x, blockDim.x, local_row * blockDim.x);
  }
  // A flat span's groups, for its inbox.
  DINLINE int count() const { return iters * stride; }
  DINLINE int id(int j) const { return first + j * stride + lane; }
  // Member u of group j, as a position within the span.
  DINLINE int at(int j, int u) const { return lane + (j * kSumBatch + u) * stride; }
  // Whether it exists: inside the span, and `base + at` inside the buffer's `end`.
  DINLINE bool has(int j, int u, int base, int end) const {
    const int k = at(j, u);
    return k < span && base + k < end;
  }
  DINLINE int members(int j, int base, int end) const {
    int n = 0;
#pragma unroll
    for (int u = 0; u < kSumBatch; ++u) n += has(j, u, base, end) ? 1 : 0;
    return n;
  }
};

// INBOXES of one codec: `regions` regions, each a slot per source rank; a slot holds
// every group's payload, then (a scaled codec) every group's scale. They start at pack
// `base` of the scratch, so a kernel lays several end to end (`end`), each with its own
// codec. Indices in packs (payload) and floats (scale).
template <class C, int ngpus>
struct Inbox {
  int groups, regions, base, slot;
  DINLINE Inbox(int groups, int regions = 1, int base = 0)
      : groups(groups),
        regions(regions),
        base(base),
        slot(groups * C::kPayloadPacks + (C::kScaled ? (groups + 3) / 4 : 0)) {}
  DINLINE int payload(int region, int src, int g) const {
    return base + (region * ngpus + src) * slot + g * C::kPayloadPacks;
  }
  DINLINE int scale(int region, int src, int g) const {
    return 4 * (base + (region * ngpus + src) * slot + groups * C::kPayloadPacks) + g;
  }
  DINLINE int end() const { return base + regions * ngpus * slot; }
};

// The host's side of the same layout, in packs: a flat span's groups at a grid of
// `grid_threads`, and the packs of an Inbox of `groups` groups at kbits (16: T itself).
inline int64_t flat_groups(int64_t span, int64_t grid_threads) {
  return (span + grid_threads * kSumBatch - 1) / (grid_threads * kSumBatch) * grid_threads;
}

inline int64_t inbox_packs(int kbits, int regions, int world, int64_t groups) {
  const int64_t payload = kSumBatch * 8 * kbits / 8 / 16;
  const int64_t scales  = kbits < 16 ? (groups + 3) / 4 : 0;
  return int64_t{regions} * world * (groups * payload + scales);
}

// =================================================================================
//   mine_group(p, in, grp, j, base, end, x)  this rank's input over one group, as floats
//   send(p, peer, box, region, g, n, x)      a group encoded into peer's inbox, in this
//                                            rank's slot
//   broadcast(p, box, region, g, n, x)       the same, encoded once, into every rank's
//   read_inbox(p, box, region, src, g, n, x) a group of one source from this rank's inbox
//   reduce_inbox(p, box, region, g, n, acc)  a group summed over every source, rank order
// and their row shapes, the fused push kernels' phases:
//   broadcast_rows    one-shot phase 1: every input row into every rank's inbox
//   scatter_rows      two-shot phase 1: each input row into its owner's inbox
//   reduce_row        a row's share reduced out of the inbox (push's `Pull::sum_row`)
//   broadcast_row     two-shot phase 2: an owned row's result into every inbox
//   gather_inbox_rows two-shot phase 3: every owner's rows out of this inbox
// =================================================================================

template <typename T, int ngpus, class C>
struct Push {
  using core   = Core<T, ngpus>;
  using V      = typename core::V;
  using Inputs = typename core::Inputs;
  using Box    = Inbox<C, ngpus>;
  static constexpr int kVals = C::kVals;

  // This rank's input over group j of `grp`, the span starting at pack `base`; members
  // past `end` read as zero.
  static DINLINE void mine_group(const Peers& p, const Inputs& in, const Groups& grp,
                                 int j, int base, int end, float (&x)[kVals]) {
    V v[kSumBatch];
#pragma unroll
    for (int u = 0; u < kSumBatch; ++u)
      v[u] = grp.has(j, u, base, end) ? core::mine(p, in, base + grp.at(j, u)) : V{};
    floats_of<T>(v, x);
  }

  // Group g encoded into `peer`'s inbox (region, this rank's slot), its first n members.
  static DINLINE void send(const Peers& p, int peer, const Box& box, int region, int g,
                           int n, const float (&x)[kVals]) {
    V q[C::kPayloadPacks];
    const float s = C::encode(x, q);
    push(p, peer, box, region, g, n, q, s);
  }

  // The same into every rank's inbox, encoded once.
  static DINLINE void broadcast(const Peers& p, const Box& box, int region, int g, int n,
                                const float (&x)[kVals]) {
    V q[C::kPayloadPacks];
    const float s = C::encode(x, q);
#pragma unroll
    for (int d = 0; d < ngpus; ++d) push(p, d, box, region, g, n, q, s);
  }

  // Group g of `src`'s slot in this rank's own inbox, its first n members, decoded (the
  // rest are garbage and never used).
  static DINLINE void read_inbox(const Peers& p, const Box& box, int region, int src,
                                 int g, int n, float (&x)[kVals]) {
    V q[C::kPayloadPacks];
    const int at   = box.payload(region, src, g);
    const int sent = payload_packs(n);
#pragma unroll
    for (int w = 0; w < C::kPayloadPacks; ++w)
      q[w] = w < sent ? core::get_pushed(p, at + w) : V{};
    float scale = 1.0f;
    if constexpr (C::kScaled)
      if (n > 0) scale = core::get_float(p, p.rank, box.scale(region, src, g));
    C::decode(q, scale, x);
  }

  // Group g summed over every source's slot, in rank order: every rank that reduces the
  // same group gets the same float bits.
  static DINLINE void reduce_inbox(const Peers& p, const Box& box, int region, int g,
                                   int n, float (&acc)[kVals]) {
#pragma unroll
    for (int i = 0; i < kVals; ++i) acc[i] = 0.0f;
    for (int src = 0; src < ngpus; ++src) {
      float x[kVals];
      read_inbox(p, box, region, src, g, n, x);
#pragma unroll
      for (int i = 0; i < kVals; ++i) acc[i] += x[i];
    }
  }

  // THE ROW SHAPES. A row is one group per thread (Groups::row), group
  // row x blockDim + threadIdx of an inbox of rows x blockDim groups; a block takes rows
  // blockIdx.x, + gridDim.x, ... in every phase, so a peer_block_barrier covers them. The
  // two-shot rows are local: rank r owns rows [r x chunk, r x chunk + chunk).

  // One-shot phase 1: every row of this rank's input into every rank's inbox.
  static DINLINE void broadcast_rows(const Peers& p, const Inputs& in, const Box& box,
                                     int rows, int packs) {
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
      const auto grp = Groups::row(packs, row);
      float x[kVals];
      mine_group(p, in, grp, 0, row * packs, (row + 1) * packs, x);
      broadcast(p, box, 0, grp.id(0), grp.members(0, 0, packs), x);
    }
  }

  // Two-shot phase 1: each row of this rank's input into its owner's inbox, at its local
  // row.
  static DINLINE void scatter_rows(const Peers& p, const Inputs& in, const Box& box,
                                   int chunk, int rows, int packs) {
    for (int lr = blockIdx.x; lr < chunk; lr += gridDim.x) {
      const auto grp = Groups::row(packs, lr);
      for (int d = 0; d < ngpus; ++d) {
        const int row = d * chunk + lr;
        if (row >= rows) break;
        float x[kVals];
        mine_group(p, in, grp, 0, row * packs, (row + 1) * packs, x);
        send(p, d, box, 0, grp.id(0), grp.members(0, 0, packs), x);
      }
    }
  }

  // Row `row` of the inbox (local, in a two-shot) summed over every source and rounded
  // to T: what `Pull::sum_row` gives a pull kernel.
  static DINLINE void reduce_row(const Peers& p, const Box& box, int row, int packs,
                                 V (&sum)[kMaxRowPacks]) {
    const auto grp = Groups::row(packs, row);
    float acc[kVals];
    reduce_inbox(p, box, 0, grp.id(0), grp.members(0, 0, packs), acc);
    packs_of<T>(acc, sum);
  }

  // Two-shot phase 2: this thread's share of owned local row `row`'s result, v[k] the
  // pack threadIdx.x + k * blockDim.x, into every rank's inbox.
  static DINLINE void broadcast_row(const Peers& p, const Box& box, int row, int packs,
                                    const V (&v)[kMaxRowPacks]) {
    const auto grp = Groups::row(packs, row);
    float x[kVals];
    floats_of<T>(v, x);
    broadcast(p, box, 0, grp.id(0), grp.members(0, 0, packs), x);
  }

  // Two-shot phase 3: every owner's rows out of this rank's inbox. store(row, i, v), i the
  // pack within the row.
  template <typename Store>
  static DINLINE void gather_inbox_rows(const Peers& p, const Box& box, int chunk,
                                        int rows, int packs, Store store) {
    for (int lr = blockIdx.x; lr < chunk; lr += gridDim.x) {
      const auto grp = Groups::row(packs, lr);
      const int n    = grp.members(0, 0, packs);
      for (int src = 0; src < ngpus; ++src) {
        const int row = src * chunk + lr;
        if (row >= rows) break;
        float x[kVals];
        read_inbox(p, box, 0, src, grp.id(0), n, x);
        V v[kSumBatch];
        packs_of<T>(x, v);
#pragma unroll
        for (int u = 0; u < kSumBatch; ++u)
          if (u < n) store(row, grp.at(0, u), v[u]);
      }
    }
  }

  // ---------------------------------------------------------------------------------
  // The pattern's own.
  // ---------------------------------------------------------------------------------

  // The payload packs that cover a group's first n members.
  static DINLINE int payload_packs(int n) {
    return (n * C::kPayloadPacks + kSumBatch - 1) / kSumBatch;
  }

  static DINLINE void push(const Peers& p, int peer, const Box& box, int region, int g,
                           int n, const V (&q)[C::kPayloadPacks], float scale) {
    if (n == 0) return;
    const int at   = box.payload(region, p.rank, g);
    const int sent = payload_packs(n);
#pragma unroll
    for (int w = 0; w < C::kPayloadPacks; ++w)
      if (w < sent) core::put(p, peer, at + w, q[w]);
    if constexpr (C::kScaled) core::put_float(p, peer, box.scale(region, p.rank, g), scale);
  }
};

}  // namespace hip_comms::p2p
