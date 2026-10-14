// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE COLLECTIVES' STEPS, built from common's tiles: one tested and tuned way
// to do each, which the kernels call rather than write again (the fused AttnRes
// pull's own reduce-scatter took 1.6x the plain two-shot's cycles for its
// bytes, thread traces 2026-10-04T21-03-09Z and 21-26-11Z).

#pragma once

#ifndef HIP_CUSTOM_COMMUNICATOR_COMMON_INTERFACE
  #error "include common/interface.cuh, common's one interface, not its parts"
#endif

namespace hip_custom_communicator {

// THE LOADS ARE THE COMMUNICATION: `rank_chunk(w)` points into rank + w's
// memory, mapped into this GPU's (peers.cuh), so loading the chunk reads it
// over the links; the sum and the store are local. The kernel syncs around
// these calls (Sync): before, every peer's data is ready; between the
// reduce-scatter and the all-gather, every rank's sums are visible.
//
// THE COLLECTIVES WORK A CHUNK AT A TIME, as a tile whose ROWS ARE THE RANKS:
// row w is rank + w's copy of the chunk (ROTATED, so at any moment the GPUs
// read different peers), a row of threads a rank (TILE::kThreadsM == WORLD).
// `chunk` is the place: its N the slice's end, its offs_n the chunk's first
// column. A rank's pointer is picked a wave at a time (`rank_chunk(w)`, a
// scalar select): every rank's held up front was ~20 scalar loads before the
// first barrier.

// REDUCE-SCATTER: this rank's slice summed over every rank, a chunk a call.
//   rank_chunk(w)  where rank + w's copy of the chunk starts
//   sum_out        where this rank's summed chunk goes (its scratch)
// The sum's order differs by rank, which is harmless: each chunk is summed by
// one rank.
namespace impl {
template <typename TILE, typename RANK_CHUNK>
DINLINE void reduce_scatter(TILE& chunk, RANK_CHUNK rank_chunk,
                            typename TILE::Dtype* sum_out) {
  // 1. Every rank's copy of the chunk, a row each, in one round trip.
  tile_gather(chunk, rank_chunk);
  // 2. Summed over the rows (the ranks), in row order, through LDS.
  const auto sum = block_reduce<Sum, Axis::m>(chunk);
  // 3. Stored as DTYPE in this rank's slice.
  tile_store(sum_out, chunk.N, sum.template to<typename TILE::Dtype>());
}
}  // namespace impl

// AN ELEMENTWISE [R, M, N] PROBLEM -- R ranks' copies of an [M, N] buffer, each
// at its own strides -- WALKED AS [R, 1, M*N]: the rows are only the buffer's
// layout, so the elements are numbered row-major (e = row * N + col, flat_at)
// and each rank's SHARE is one contiguous range of them, a rank's tile copy
// TILE_N of it. WHY: the links and HBM reward long reads from each peer. Cut a
// row at a time, a rank's share was its slice of every row -- 1.8 KB runs at N
// = 7168, the plain all-reduce +2% against the flat walk; 128-column tiles, 256
// B runs, +3-7% (2026-10-11) -- while a contiguous buffer's range is one run of
// the whole share. IT SERVES BADLY a strided buffer: its runs are its rows
// wherever it has them, and each pack's row and column are divided out. The
// walk is the same for every step and every rank, so a block reads in the
// all-gather what the same block on each rank wrote in the reduce-scatter. A
// copy is whole waves (its rank's pointer one scalar select); copy w is rank +
// w's, ROTATED, so at any moment the GPUs read different peers.
namespace impl {
// THE SHARES of m x n elements over WORLD ranks, in whole packs (the last
// rank's short), and the chunks of TILE_N a share takes.
struct Shares {
  int64_t total;
  int slice;
  int chunks;
  DINLINE int64_t first(int r) const { return int64_t{r} * slice; }
  DINLINE int len(int r) const {
    return static_cast<int>(
        max(int64_t{0}, min(int64_t{slice}, total - first(r))));
  }
};
template <typename TILE>
DINLINE Shares shares_of(int m, int n) {
  constexpr int WORLD = TILE::kTileR;
  const int64_t total = int64_t{m} * n;
  const int slice =
      static_cast<int>((total / TILE::kPack + WORLD - 1) / WORLD * TILE::kPack);
  return {total, slice, (slice + TILE::kTileN - 1) / TILE::kTileN};
}
// WHETHER AN [*, n] BUFFER'S ROWS ARE BACK TO BACK, so its range is one run:
// CONTIGUOUS, a template parameter of the kernel, its instance chosen on the
// host from the strides as given. A compile-time choice, not a branch in the
// kernel: both paths in one kernel cost ~0.1 us a call at 128-512 tokens
// against the contiguous path alone (the strided code and its registers,
// 2026-10-12T01-17-18Z). THIS THREAD'S COPY OF THE CHUNK, from the range
// starting at element `first` of an [*, n] buffer: one run when its rows are
// back to back (the address a uniform base and a lane's offset, as tile_load
// builds it), else each pack's own row and column.
template <bool CONTIGUOUS, typename TILE, typename T>
DINLINE void range_load(TILE& t, const Ptr<T>& p, int n, int64_t first) {
  using P = typename TILE::Pack;
  if constexpr (CONTIGUOUS) {
    tile_load(t, p.data + first * p.stride_n, 0);
  } else {
#pragma unroll
    for (int k = 0; k < t.K; ++k)
      held(t, 0, k,
           pack_load(reinterpret_cast<const P*>(
               flat_at(p, first + int64_t{t.col(k)} * t.kPack, n))));
    issued();
  }
}
// ...and stored, its first `len` elements.
template <bool CONTIGUOUS, typename TILE, typename T>
DINLINE void range_store(const TILE& t, const Ptr<T>& p, int n, int64_t first,
                         int len) {
  using P = typename TILE::Pack;
  P* run = reinterpret_cast<P*>(p.data + first * p.stride_n);
#pragma unroll
  for (int k = 0; k < t.K; ++k) {
    const int c = t.offs_n / t.kPack + t.lane() + k * t.kThreadsN;
    if (c >= len / t.kPack) continue;
    P* at;
    if constexpr (CONTIGUOUS)
      at = at_col(run, t, k);
    else
      at = reinterpret_cast<P*>(flat_at(p, first + int64_t{c} * t.kPack, n));
    if constexpr (sizeof(P) == 16)
      global_store(at, pack(t, 0, k));
    else
      *at = pack(t, 0, k);
  }
}

// REDUCE-SCATTER: this rank's share of the inputs, summed over the ranks, into
// its scratch (one contiguous run from scratch.data). inp(r) is rank r's input
// at its strides, every rank's alike.
template <typename TILE, bool CONTIGUOUS, typename RANK_PTR, typename T>
DINLINE void reduce_scatter(int rank, RANK_PTR input, int m, int n,
                            const Ptr<T>& scratch) {
  static_assert(TILE::kTileM == 1 && TILE::kThreadsR == TILE::kTileR,
                "an [R, 1, TILE_N] chunk");
  static_assert(TILE::kSliceThreads % kWaveSize == 0,
                "a rank's copy is whole waves");
  constexpr int WORLD = TILE::kTileR;
  const Shares s = shares_of<TILE>(m, n);
  if (s.len(rank) == 0)
    return;  // a late rank's share is empty at few packs (every block alike)
  block_for_each_tile(1, s.chunks, [&](int, int chunk) {
    TILE got{1, s.len(rank), 0, chunk * TILE::kTileN};
    // 1. Every rank's copy of this chunk of this rank's share, in one round
    // trip.
    range_load<CONTIGUOUS>(got, input((rank + got.slice()) % WORLD), n,
                           s.first(rank));
    // 2. Summed over the ranks, in rank order, through LDS.
    const auto sum = block_reduce<Sum, Axis::r>(got);
    // 3. Stored as T in this rank's scratch.
    tile_store(scratch.data, 0, sum.template to<T>());
  });
}

// ALL-GATHER: every rank's summed share out of its scratch, at its place in the
// output.
template <typename TILE, bool CONTIGUOUS, typename RANK_PTR, typename T>
DINLINE void all_gather(int rank, RANK_PTR scratch, const Ptr<T>& output, int m,
                        int n) {
  static_assert(TILE::kTileM == 1 && TILE::kThreadsR == TILE::kTileR,
                "an [R, 1, TILE_N] chunk");
  static_assert(TILE::kSliceThreads % kWaveSize == 0,
                "a rank's copy is whole waves");
  constexpr int WORLD = TILE::kTileR;
  const Shares s = shares_of<TILE>(m, n);
  block_for_each_tile(1, s.chunks, [&](int, int chunk) {
    TILE got{1, s.slice, 0, chunk * TILE::kTileN};
    const int r = (rank + got.slice()) % WORLD;
    // 1. Rank r's summed chunk, in one round trip.
    tile_load(got, scratch(r).data, 0);
    // 2. Stored where rank r's share goes.
    range_store<CONTIGUOUS>(got, output, n, s.first(r), s.len(r));
  });
}

// STAGED, BEFORE THE REDUCE-SCATTER: this rank's input, every rank's share of
// it, into its staging (an [m, n] buffer, ours, its rows back to back: the
// host's staging_stride_m = n), each block the chunks that rank's same block
// reads.
template <typename TILE, bool CONTIGUOUS, typename T>
DINLINE void stage(int rank, const Ptr<const T>& input, const Ptr<T>& staging,
                   int m, int n) {
  static_assert(TILE::kTileM == 1 && TILE::kThreadsR == TILE::kTileR,
                "an [R, 1, TILE_N] chunk");
  constexpr int WORLD = TILE::kTileR;
  const Shares s = shares_of<TILE>(m, n);
  block_for_each_tile(1, s.chunks, [&](int, int chunk) {
    const int r = (rank + TILE{}.slice()) % WORLD;
    const int len = s.len(r);
    if (len == 0) return;
    TILE part{1, len, 0, chunk * TILE::kTileN};
    range_load<CONTIGUOUS>(part, input, n, s.first(r));
    range_store<true>(part, staging, n, s.first(r), len);
  });
}
}  // namespace impl

}  // namespace hip_custom_communicator
