// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE REDUCTIONS, by scope, each built on the one below: over the PEERS (a pack
// or a tile row summed over every rank, fp32, rounded once, in rank order),
// over a WAVE (shuffles), over a BLOCK (a wave's result, then one LDS pass). N
// values at once at every scope: each block pass is two barriers and an LDS
// round trip, so a kernel with several reductions takes them together.

#pragma once

#ifndef HIP_CUSTOM_COMMUNICATOR_COMMON_INTERFACE
  #error "include common/interface.cuh, common's one interface, not its parts"
#endif

#include <cmath>
#include <type_traits>

#include "build.cuh"
#include "memory.cuh"
#include "utils.cuh"

namespace hip_custom_communicator {

// EACH PACK SUMMED OVER ITS `ngpus` SOURCES, in fp32 in source order and
// rounded once to T, into sum[k]: callers whose sources are the ranks agree
// bitwise. Waits on the loads only here. EVERY PEER'S TILE SUMMED, in rank
// order in fp32 and rounded once, as a pack's is.
namespace impl {
template <typename TILE, int WORLD>
DINLINE TILE peers_reduce(const TILE (&t)[WORLD]) {
  constexpr int NL = TILE::kPack;
  TILE sum = t[0].template like<typename TILE::Acc>();
#pragma unroll
  for (int m = 0; m < TILE::kRows; ++m)
#pragma unroll
    for (int k = 0; k < sum.K; ++k) {
      float acc[NL];
#pragma unroll
      for (int j = 0; j < NL; ++j) acc[j] = static_cast<float>(t[0].v[m][k][j]);
#pragma unroll
      for (int r = 1; r < WORLD; ++r)
#pragma unroll
        for (int j = 0; j < NL; ++j)
          acc[j] += static_cast<float>(t[r].v[m][k][j]);
#pragma unroll
      for (int j = 0; j < NL; ++j)
        sum.v[m][k][j] = static_cast<typename TILE::Acc>(acc[j]);
    }
  return sum;
}
}  // namespace impl

// THE OPERATIONS a wave or block reduction combines with.
struct Sum {
  static DINLINE float apply(float a, float b) { return a + b; }
  static constexpr float kIdentity = 0.0f;
};
struct Max {
  static DINLINE float apply(float a, float b) { return fmaxf(a, b); }
  static constexpr float kIdentity = -INFINITY;
};

// ONE DPP STEP: `v` from the lane `ctrl` names, as a VALU operand, no LDS.
// `row_mask` picks which 16-lane rows take it; the others read `identity`,
// which leaves them as they were. A FULL ROW MASK NEEDS NO `old`: left
// undefined (mov_dpp), the compiler fuses the move into the add that uses it
// (v_add_f32_dpp); an explicit identity kept a v_mov_b32_dpp and an add a step
// (ISA 2026-10-04T03-07-41Z).
namespace impl {
template <int DPP_CTRL, int DPP_ROW_MASK, typename REDUCE_OP>
DINLINE float dpp(float v) {
  int moved;
  if constexpr (DPP_ROW_MASK == 0xf)
    moved = __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), DPP_CTRL, 0xf,
                                     0xf, false);
  else
    moved = __builtin_amdgcn_update_dpp(
        __builtin_bit_cast(int, REDUCE_OP::kIdentity),
        __builtin_bit_cast(int, v), DPP_CTRL, DPP_ROW_MASK, 0xf, false);
  return __builtin_bit_cast(float, moved);
}
}  // namespace impl

// N VALUES OVER THE WAVE, in place, every lane left holding the results. The
// moved value first: only a VOP2's src0 takes DPP, so (dpp(x), x) fuses each
// step into one v_add_f32_dpp where (x, dpp(x)) was a v_mov_b32_dpp and an add
// (ISA 2026-10-04T02-22-41Z). DPP, as rocPRIM's and Composable Kernel's wave
// reductions do: each step is a VALU operand from another lane. The butterfly
// __shfl_xor it replaced compiled to ds_bpermute, an LDS round trip a step, six
// of them waiting on each other (ISA 2026-09-30T20-43-18Z; stamps: 0.92 us for
// one block_reduce).
//   swap neighbours, swap pairs (quad_perm [1,0,3,2], [2,3,0,1]): each quad
//   holds its sum mirror in 8 lanes, then in 16 (row_half_mirror, row_mirror):
//   each row holds its sum lane 15 into rows 1 and 3, lane 31 into rows 2 and 3
//   (row_bcast15, row_bcast31): lane 63 holds the wave's, and readlane hands it
//   to every lane.
namespace impl {
template <typename REDUCE_OP, int NUM_VALUES>
DINLINE void wave_reduce(float (&v)[NUM_VALUES]) {
  static_assert(kWaveSize == 64, "the DPP sequence is for 64-lane waves");
#pragma unroll
  for (int n = 0; n < NUM_VALUES; ++n) {
    float x = v[n];
    x = REDUCE_OP::apply(dpp<0xb1, 0xf, REDUCE_OP>(x),
                         x);  // quad_perm [1,0,3,2]
    x = REDUCE_OP::apply(dpp<0x4e, 0xf, REDUCE_OP>(x),
                         x);  // quad_perm [2,3,0,1]
    x = REDUCE_OP::apply(dpp<0x141, 0xf, REDUCE_OP>(x), x);  // row_half_mirror
    x = REDUCE_OP::apply(dpp<0x140, 0xf, REDUCE_OP>(x), x);  // row_mirror
    x = REDUCE_OP::apply(dpp<0x142, 0xa, REDUCE_OP>(x),
                         x);  // row_bcast15 into rows 1, 3
    x = REDUCE_OP::apply(dpp<0x143, 0xc, REDUCE_OP>(x),
                         x);  // row_bcast31 into rows 2, 3
    v[n] = __builtin_bit_cast(
        float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, x), 63));
  }
}
}  // namespace impl

// N VALUES OVER THE BLOCK, in place: each wave reduces its N, then every thread
// sums the waves' partials in wave order (the same order in every thread, so
// every thread holds the same totals). One wave reducing the partials over all
// 64 lanes, then broadcasting them, was a second DPP pass a value, a second
// barrier and an LDS round trip to add 4 numbers at 256 threads: half of
// AttnRes's loop (ISA 2026-10-04T03-31-28Z). The trailing barrier keeps the
// next call from overwriting partials still being read. THE BLOCK'S WAVES ARE
// COMPILED IN: read from blockDim, every thread loaded the widest block's
// partials and kept 12 VGPRs at AttnRes's peak (ISA 2026-10-04T16-39-15Z).
namespace impl {
template <typename REDUCE_OP, int THREADS_PER_BLOCK, int NUM_VALUES>
DINLINE void block_reduce(float (&v)[NUM_VALUES]) {
  constexpr int kWaves = (THREADS_PER_BLOCK + kWaveSize - 1) / kWaveSize;
  __shared__ float partial[NUM_VALUES][kWaves];
  const int lane = threadIdx.x % kWaveSize;
  const int wave = threadIdx.x / kWaveSize;
  wave_reduce<REDUCE_OP>(v);
  if (lane == 0) {
#pragma unroll
    for (int n = 0; n < NUM_VALUES; ++n) partial[n][wave] = v[n];
  }
  __syncthreads();
#pragma unroll
  for (int n = 0; n < NUM_VALUES; ++n) {
    float x = partial[n][0];
#pragma unroll
    for (int w = 1; w < kWaves; ++w) x = REDUCE_OP::apply(x, partial[n][w]);
    v[n] = x;
  }
  __syncthreads();
}
}  // namespace impl

// A TILE REDUCED OVER AN AXIS, Triton's tl.sum(x, axis) for a float tile:
//   Axis::m  over the rows, every column's: the rows meet in LDS as they are
//   held (a bf16 tile's
//            16 bytes a group, as the old two-shot's), reduced in fp32 in row
//            order, and the one-row fp32 result is the first THREADS_N threads'
//            (a Tile of one row of threads)
//   Axis::n  over the columns, every row's, into `out` (a row's value in every
//   thread of the row):
//            each thread's groups, then the row's threads (a wave's or the
//            block's)
//   Axis::r  over a 3-D tile's ranks, every element's: each rank's copy meets
//   in LDS as it is held,
//            reduced in fp32 in rank order, and the TILE_M x TILE_N fp32 result
//            is the first rank's threads' (a 2-D tile of THREADS_M x THREADS_N
//            threads)
enum class Axis { m, n, r };

namespace impl {
// OVER THE RANKS: every thread's packs into LDS at its rank's place, a pack a
// thread side by side (consecutive threads' 16 bytes adjacent), then the first
// rank's threads sum each pack over the ranks. LDS: TILE_R x TILE_M x TILE_N
// elements as held.
template <typename REDUCE_OP, typename TILE>
DINLINE Tile<typename TILE::Dtype, TILE::kTileM, TILE::kTileN, TILE::kThreadsM,
             TILE::kThreadsN, TILE::kThreadsPerBlock, float>
block_reduce_ranks(const TILE& t) {
  static_assert(TILE::kThreadsR == TILE::kTileR,
                "a thread holds one rank's copy");
  using Acc = typename TILE::Acc;
  using P = typename TILE::Pack;
  constexpr int kPacks = TILE::kRows * TILE::K;  // a thread's
  __shared__ P part[TILE::kTileR][kPacks][TILE::kSliceThreads];
  Tile<typename TILE::Dtype, TILE::kTileM, TILE::kTileN, TILE::kThreadsM,
       TILE::kThreadsN, TILE::kThreadsPerBlock, float>
      output{t.M, t.N, t.offs_m, t.offs_n, t.row_step};
  const int mine = static_cast<int>(threadIdx.x) % TILE::kSliceThreads;
#pragma unroll
  for (int m = 0; m < TILE::kRows; ++m)
#pragma unroll
    for (int k = 0; k < TILE::K; ++k)
      part[t.slice()][m * TILE::K + k][mine] = pack(t, m, k);
  __syncthreads();
  if (output.participates()) {
#pragma unroll
    for (int m = 0; m < TILE::kRows; ++m)
#pragma unroll
      for (int k = 0; k < TILE::K; ++k) {
        float x[TILE::kPack];
        P p = part[0][m * TILE::K + k][mine];
#pragma unroll
        for (int j = 0; j < TILE::kPack; ++j)
          x[j] = static_cast<float>(reinterpret_cast<const Acc*>(&p)[j]);
#pragma unroll
        for (int r = 1; r < TILE::kTileR; ++r) {
          p = part[r][m * TILE::K + k][mine];
#pragma unroll
          for (int j = 0; j < TILE::kPack; ++j)
            x[j] = REDUCE_OP::apply(
                x[j], static_cast<float>(reinterpret_cast<const Acc*>(&p)[j]));
        }
#pragma unroll
        for (int j = 0; j < TILE::kPack; ++j) output.v[m][k][j] = x[j];
      }
  }
  __syncthreads();  // before the next reduction writes `part`
  return output;
}

template <typename REDUCE_OP, Axis AXIS, typename TILE>
DINLINE Tile<typename TILE::Dtype, 1, TILE::kTileN, 1, TILE::kThreadsN,
             TILE::kThreadsPerBlock, float>
block_reduce_rows(const TILE& t) {
  static_assert(AXIS == Axis::m, "a row's reduction (Axis::n) writes `out`");
  static_assert(TILE::kRows == 1, "a thread holds one row of the rows reduced");
  constexpr int kRowLanes = TILE::kThreadsM;
  using Acc = typename TILE::Acc;
  __shared__ Acc part[kRowLanes][TILE::kTileN];
  Tile<typename TILE::Dtype, 1, TILE::kTileN, 1, TILE::kThreadsN,
       TILE::kThreadsPerBlock, float>
      output{1, t.N, 0, t.offs_n};
  const int lane_row = static_cast<int>(threadIdx.x) / TILE::kThreadsN;
#pragma unroll
  for (int k = 0; k < TILE::K; ++k)
#pragma unroll
    for (int j = 0; j < TILE::kPack; ++j) {
      part[lane_row][(t.lane() + k * TILE::kThreadsN) * TILE::kPack + j] =
          t.v[0][k][j];
    }
  __syncthreads();
  if (output.participates()) {
#pragma unroll
    for (int k = 0; k < TILE::K; ++k)
#pragma unroll
      for (int j = 0; j < TILE::kPack; ++j) {
        const int c = (output.lane() + k * TILE::kThreadsN) * TILE::kPack + j;
        float x = static_cast<float>(part[0][c]);
#pragma unroll
        for (int r = 1; r < kRowLanes; ++r)
          x = REDUCE_OP::apply(x, static_cast<float>(part[r][c]));
        output.v[0][k][j] = x;
      }
  }
  __syncthreads();  // before the next reduction writes `part`
  return output;
}

// A TILE OVER AN AXIS: the ranks' (a 3-D tile's) or the rows' result tile.
template <typename REDUCE_OP, Axis AXIS, typename TILE>
DINLINE auto block_reduce(const TILE& t) {
  if constexpr (AXIS == Axis::r)
    return block_reduce_ranks<REDUCE_OP>(t);
  else
    return block_reduce_rows<REDUCE_OP, AXIS>(t);
}

template <typename REDUCE_OP, Axis AXIS, typename TILE>
DINLINE void block_reduce(const TILE& t, float (&output)[TILE::kRows]) {
  static_assert(AXIS == Axis::n,
                "a column's reduction (Axis::m) returns a tile");
  static_assert(std::is_same_v<typename TILE::Acc, float>,
                "a reduction is of a float tile");
#pragma unroll
  for (int m = 0; m < TILE::kRows; ++m) {
    float x = REDUCE_OP::kIdentity;
#pragma unroll
    for (int k = 0; k < TILE::K; ++k)
#pragma unroll
      for (int j = 0; j < TILE::kPack; ++j)
        x = t.mask(k) != 0.0f ? REDUCE_OP::apply(x, t.v[m][k][j]) : x;
    output[m] = x;
  }
  if constexpr (TILE::kThreadsM == 1)
    block_reduce<REDUCE_OP, TILE::kThreads>(output);
  else if constexpr (TILE::kThreadsN == kWaveSize)
    wave_reduce<REDUCE_OP>(output);
  else
    static_assert(TILE::kThreadsN == kWaveSize,
                  "a row is the block's or a wave's");
}
}  // namespace impl

// The LDS a block_reduce<Op, N> takes, for a kernel budgeting the rest.
constexpr int64_t block_reduce_lds_bytes(int n) {
  return int64_t{kBuild.kernels.max_waves} * n * sizeof(float);
}

}  // namespace hip_custom_communicator
