// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// p2p::push, behind p2p.cuh: a rank reads only its own input and its own scratch, and
// what crosses a link is a store into a peer's INBOX, encoded by a Codec (16 bits: T
// itself; 8, 4: QuickReduce's integers).
//
// A GROUP is one thread's kSumBatch packs of a span, positions
// lane + (j * kSumBatch + u) * stride, and crosses a link as one Codec payload (and one
// scale). Every phase gives a thread the same groups, so the same-numbered peer blocks
// are all a phase waits for. Members that exist are a prefix (u < members), and only
// the payload covering them is sent. A BUFFER is the whole input grid-strided; a ROW is
// one group per thread (group row x blockDim + threadIdx), a block taking rows
// blockIdx.x, + gridDim.x, ... in every phase. Two-shot slices and rows are local: rank
// r owns [r x chunk, r x chunk + chunk).

#pragma once

#ifndef HIP_COMMS_P2P_INTERFACE
#error "include p2p/p2p.cuh, p2p's one interface, not its parts"
#endif

#include "core.cuh"

namespace hip_comms::p2p {

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

struct Groups {
  int span, lane, stride, first, iters;
  DINLINE Groups(int span, int lane, int stride, int first)
      : span(span),
        lane(lane),
        stride(stride),
        first(first),
        iters((span + stride * kSumBatch - 1) / (stride * kSumBatch)) {}
  // Grid-strided over the whole buffer (or a slice of it): the buffer phases.
  static DINLINE Groups buffer(int span) {
    return Groups(span, blockIdx.x * blockDim.x + threadIdx.x, gridDim.x * blockDim.x, 0);
  }
  // This thread's share of one row, the packs a row helper holds (one group, j = 0), as
  // group local_row x blockDim + threadIdx of an inbox.
  static DINLINE Groups row(int packs, int local_row) {
    return Groups(packs, threadIdx.x, blockDim.x, local_row * blockDim.x);
  }
  // A buffer span's groups, for its inbox.
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

// The payload packs that cover a group's first n members.
template <class C>
DINLINE int payload_packs(int n) {
  return (n * C::kPayloadPacks + kSumBatch - 1) / kSumBatch;
}

// This rank's input over group j of `grp`, the span starting at pack `base`; members past
// `end` read as zero.
template <class C, typename T, int ngpus>
DINLINE void mine_group(const World<T, ngpus>& w, const Groups& grp, int j, int base,
                        int end, float (&x)[C::kVals]) {
  typename traits<T>::V v[kSumBatch];
#pragma unroll
  for (int u = 0; u < kSumBatch; ++u)
    v[u] = grp.has(j, u, base, end) ? mine(w, base + grp.at(j, u))
                                    : typename traits<T>::V{};
  floats_of<T>(v, x);
}

template <class C, typename T, int ngpus>
DINLINE void push_payload(const World<T, ngpus>& w, int peer, const Inbox<C, ngpus>& box,
                          int region, int g, int n,
                          const typename traits<T>::V (&q)[C::kPayloadPacks], float scale) {
  if (n == 0) return;
  const int at   = box.payload(region, w.peers.rank, g);
  const int sent = payload_packs<C>(n);
#pragma unroll
  for (int k = 0; k < C::kPayloadPacks; ++k)
    if (k < sent) put(w, peer, at + k, q[k]);
  if constexpr (C::kScaled) put_float(w, peer, box.scale(region, w.peers.rank, g), scale);
}

// Group g encoded into `peer`'s inbox (region, this rank's slot), its first n members.
template <class C, typename T, int ngpus>
DINLINE void send(const World<T, ngpus>& w, int peer, const Inbox<C, ngpus>& box,
                  int region, int g, int n, const float (&x)[C::kVals]) {
  typename traits<T>::V q[C::kPayloadPacks];
  const float s = C::encode(x, q);
  push_payload(w, peer, box, region, g, n, q, s);
}

// The same into every rank's inbox, encoded once.
template <class C, typename T, int ngpus>
DINLINE void broadcast(const World<T, ngpus>& w, const Inbox<C, ngpus>& box, int region,
                       int g, int n, const float (&x)[C::kVals]) {
  typename traits<T>::V q[C::kPayloadPacks];
  const float s = C::encode(x, q);
#pragma unroll
  for (int d = 0; d < ngpus; ++d) push_payload(w, d, box, region, g, n, q, s);
}

// Group g of `src`'s slot in this rank's own inbox, its first n members, decoded (the
// rest are garbage and never used).
template <class C, typename T, int ngpus>
DINLINE void read_inbox(const World<T, ngpus>& w, const Inbox<C, ngpus>& box, int region,
                        int src, int g, int n, float (&x)[C::kVals]) {
  typename traits<T>::V q[C::kPayloadPacks];
  const int at   = box.payload(region, src, g);
  const int sent = payload_packs<C>(n);
#pragma unroll
  for (int k = 0; k < C::kPayloadPacks; ++k)
    q[k] = k < sent ? get_pushed(w, at + k) : typename traits<T>::V{};
  float scale = 1.0f;
  if constexpr (C::kScaled)
    if (n > 0) scale = get_float(w, w.peers.rank, box.scale(region, src, g));
  C::decode(q, scale, x);
}

// Group g summed over every source's slot, in rank order: every rank that reduces the
// same group gets the same float bits.
template <class C, typename T, int ngpus>
DINLINE void reduce_inbox(const World<T, ngpus>& w, const Inbox<C, ngpus>& box, int region,
                          int g, int n, float (&acc)[C::kVals]) {
#pragma unroll
  for (int i = 0; i < C::kVals; ++i) acc[i] = 0.0f;
  for (int src = 0; src < ngpus; ++src) {
    float x[C::kVals];
    read_inbox(w, box, region, src, g, n, x);
#pragma unroll
    for (int i = 0; i < C::kVals; ++i) acc[i] += x[i];
  }
}

// A group's decoded members stored at `base` + their positions; store(pos, v).
template <class C, typename T, typename Store>
DINLINE void store_group(const Groups& grp, int j, int base, int n,
                         const float (&x)[C::kVals], Store store) {
  typename traits<T>::V v[kSumBatch];
  packs_of<T>(x, v);
#pragma unroll
  for (int u = 0; u < kSumBatch; ++u)
    if (u < n) store(base + grp.at(j, u), v[u]);
}

}  // namespace impl

namespace push {

// =================================================================================
// THE INBOXES a kernel lays out, end to end from pack `base` of the scratch (`end()`
// is where the next may start): one for a buffer span, one for rows.
// =================================================================================

template <class C, typename T, int ngpus>
DINLINE Inbox<C, ngpus> buffer_inbox(const World<T, ngpus>&, int span, int base = 0) {
  return Inbox<C, ngpus>(impl::Groups::buffer(span).count(), 1, base);
}

template <class C, typename T, int ngpus>
DINLINE Inbox<C, ngpus> row_inbox(const World<T, ngpus>&, int rows, int base = 0) {
  return Inbox<C, ngpus>(rows * static_cast<int>(blockDim.x), 1, base);
}

// =================================================================================
// THE BUFFER PHASES (the plain all-reduce).
// =================================================================================

// One-shot phase 1: this rank's whole input into every rank's inbox (this one's too).
template <class C, typename T, int ngpus>
DINLINE void broadcast_buffer(const World<T, ngpus>& w, const Inbox<C, ngpus>& box,
                              int size) {
  const auto grp = impl::Groups::buffer(size);
  for (int j = 0; j < grp.iters; ++j) {
    float x[C::kVals];
    impl::mine_group<C>(w, grp, j, 0, size, x);
    impl::broadcast(w, box, 0, grp.id(j), grp.members(j, 0, size), x);
  }
}

// One-shot phase 2: every source summed out of this rank's inbox, in rank order (every
// rank the same bits). store(position, v).
template <class C, typename T, int ngpus, typename Store>
DINLINE void reduce_buffer(const World<T, ngpus>& w, const Inbox<C, ngpus>& box, int size,
                           Store store) {
  const auto grp = impl::Groups::buffer(size);
  for (int j = 0; j < grp.iters; ++j) {
    const int n = grp.members(j, 0, size);
    float acc[C::kVals];
    impl::reduce_inbox(w, box, 0, grp.id(j), n, acc);
    impl::store_group<C, T>(grp, j, 0, n, acc, store);
  }
}

// Two-shot phase 1: this rank's input, each owner's slice into that owner's inbox.
template <class C, typename T, int ngpus>
DINLINE void scatter_buffer(const World<T, ngpus>& w, const Inbox<C, ngpus>& box,
                            int chunk, int size) {
  const auto grp = impl::Groups::buffer(chunk);
  for (int j = 0; j < grp.iters; ++j) {
    for (int d = 0; d < ngpus; ++d) {
      float x[C::kVals];
      impl::mine_group<C>(w, grp, j, d * chunk, size, x);
      impl::send(w, d, box, 0, grp.id(j), grp.members(j, d * chunk, size), x);
    }
  }
}

// Two-shot phase 2: this rank's slice summed out of `in`, rounded to T (as the
// unquantized sum lands), encoded once and pushed into every rank's `out`.
template <class C, typename T, int ngpus>
DINLINE void reduce_broadcast_slice(const World<T, ngpus>& w, const Inbox<C, ngpus>& in,
                                    const Inbox<C, ngpus>& out, int chunk, int size) {
  const auto grp = impl::Groups::buffer(chunk);
  for (int j = 0; j < grp.iters; ++j) {
    const int n = grp.members(j, w.peers.rank * chunk, size);
    float acc[C::kVals];
    impl::reduce_inbox(w, in, 0, grp.id(j), n, acc);
#pragma unroll
    for (int i = 0; i < C::kVals; ++i) acc[i] = static_cast<float>(static_cast<T>(acc[i]));
    impl::broadcast(w, out, 0, grp.id(j), n, acc);
  }
}

// Two-shot phase 3: every owner's slice out of this rank's inbox. store(position, v).
template <class C, typename T, int ngpus, typename Store>
DINLINE void gather_buffer(const World<T, ngpus>& w, const Inbox<C, ngpus>& box, int chunk,
                           int size, Store store) {
  const auto grp = impl::Groups::buffer(chunk);
  for (int j = 0; j < grp.iters; ++j) {
    for (int src = 0; src < ngpus; ++src) {
      const int n = grp.members(j, src * chunk, size);
      float x[C::kVals];
      impl::read_inbox(w, box, 0, src, grp.id(j), n, x);
      impl::store_group<C, T>(grp, j, src * chunk, n, x, store);
    }
  }
}

// =================================================================================
// THE ROW PHASES (the fused ops).
// =================================================================================

// One-shot phase 1: every row of this rank's input into every rank's inbox.
template <class C, typename T, int ngpus>
DINLINE void broadcast_rows(const World<T, ngpus>& w, const Inbox<C, ngpus>& box, int rows,
                            int packs) {
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const auto grp = impl::Groups::row(packs, row);
    float x[C::kVals];
    impl::mine_group<C>(w, grp, 0, row * packs, (row + 1) * packs, x);
    impl::broadcast(w, box, 0, grp.id(0), grp.members(0, 0, packs), x);
  }
}

// Two-shot phase 1: each row of this rank's input into its owner's inbox, at its local
// row.
template <class C, typename T, int ngpus>
DINLINE void scatter_rows(const World<T, ngpus>& w, const Inbox<C, ngpus>& box, int chunk,
                          int rows, int packs) {
  for (int lr = blockIdx.x; lr < chunk; lr += gridDim.x) {
    const auto grp = impl::Groups::row(packs, lr);
    for (int d = 0; d < ngpus; ++d) {
      const int row = d * chunk + lr;
      if (row >= rows) break;
      float x[C::kVals];
      impl::mine_group<C>(w, grp, 0, row * packs, (row + 1) * packs, x);
      impl::send(w, d, box, 0, grp.id(0), grp.members(0, 0, packs), x);
    }
  }
}

// Row `row` of the inbox (local, in a two-shot) summed over every source and rounded to
// T: what `pull::sum_row` gives a pull kernel. sum[k] is pack threadIdx.x + k x blockDim.
template <class C, typename T, int ngpus>
DINLINE void reduce_row(const World<T, ngpus>& w, const Inbox<C, ngpus>& box, int row,
                        int packs, typename traits<T>::V (&sum)[kMaxRowPacks]) {
  const auto grp = impl::Groups::row(packs, row);
  float acc[C::kVals];
  impl::reduce_inbox(w, box, 0, grp.id(0), grp.members(0, 0, packs), acc);
  impl::packs_of<T>(acc, sum);
}

// Two-shot phase 2: this thread's share of owned local row `row`'s result, v[k] the pack
// threadIdx.x + k x blockDim, into every rank's inbox.
template <class C, typename T, int ngpus>
DINLINE void broadcast_row(const World<T, ngpus>& w, const Inbox<C, ngpus>& box, int row,
                           int packs, const typename traits<T>::V (&v)[kMaxRowPacks]) {
  const auto grp = impl::Groups::row(packs, row);
  float x[C::kVals];
  impl::floats_of<T>(v, x);
  impl::broadcast(w, box, 0, grp.id(0), grp.members(0, 0, packs), x);
}

// Two-shot phase 3: every owner's rows out of this rank's inbox. store(row, i, v), i the
// pack within the row.
template <class C, typename T, int ngpus, typename Store>
DINLINE void gather_rows(const World<T, ngpus>& w, const Inbox<C, ngpus>& box, int chunk,
                         int rows, int packs, Store store) {
  for (int lr = blockIdx.x; lr < chunk; lr += gridDim.x) {
    const auto grp = impl::Groups::row(packs, lr);
    const int n    = grp.members(0, 0, packs);
    for (int src = 0; src < ngpus; ++src) {
      const int row = src * chunk + lr;
      if (row >= rows) break;
      float x[C::kVals];
      impl::read_inbox(w, box, 0, src, grp.id(0), n, x);
      impl::store_group<C, T>(grp, 0, 0, n, x,
                                [&](int i, const auto& v) { store(row, i, v); });
    }
  }
}

}  // namespace push

}  // namespace hip_comms::p2p
