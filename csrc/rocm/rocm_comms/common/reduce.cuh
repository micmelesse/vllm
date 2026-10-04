// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE REDUCTIONS, by scope, each built on the one below: over the PEERS (a pack or a tile row
// summed over every rank, fp32, rounded once, in rank order), over a WAVE (shuffles), over a BLOCK
// (a wave's result, then one LDS pass). N values at once at every scope: each block pass is two
// barriers and an LDS round trip, so a kernel with several reductions takes them together.

#pragma once

#ifndef HIP_COMMS_COMMON_INTERFACE
#error "include common/common.cuh, common's one interface, not its parts"
#endif

#include <cmath>
#include <type_traits>

#include "build.cuh"
#include "memory.cuh"
#include "utils.cuh"

namespace hip_comms {

// EACH PACK SUMMED OVER ITS `ngpus` SOURCES, in fp32 in source order and rounded once to T, into
// sum[k]: callers whose sources are the ranks agree bitwise. Waits on the loads only here.
// EVERY PEER'S TILE SUMMED, in rank order in fp32 and rounded once, as a pack's is.
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
        for (int j = 0; j < NL; ++j) acc[j] += static_cast<float>(t[r].v[m][k][j]);
#pragma unroll
      for (int j = 0; j < NL; ++j) sum.v[m][k][j] = static_cast<typename TILE::Acc>(acc[j]);
    }
  return sum;
}


// THE OPERATIONS a wave or block reduction combines with.
struct Sum {
  static DINLINE float apply(float a, float b) { return a + b; }
  static constexpr float kIdentity = 0.0f;
};
struct Max {
  static DINLINE float apply(float a, float b) { return fmaxf(a, b); }
  static constexpr float kIdentity = -INFINITY;
};

// ONE DPP STEP: `v` from the lane `ctrl` names, as a VALU operand, no LDS. `row_mask` picks which
// 16-lane rows take it; the others read `identity`, which leaves them as they were.
// A FULL ROW MASK NEEDS NO `old`: left undefined (mov_dpp), the compiler fuses the move into the
// add that uses it (v_add_f32_dpp); an explicit identity kept a v_mov_b32_dpp and an add a step
// (ISA 2026-10-04T03-07-41Z).
template <int DPP_CTRL, int DPP_ROW_MASK, typename REDUCE_OP>
DINLINE float dpp(float v) {
  int moved;
  if constexpr (DPP_ROW_MASK == 0xf)
    moved = __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), DPP_CTRL, 0xf, 0xf, false);
  else
    moved = __builtin_amdgcn_update_dpp(__builtin_bit_cast(int, REDUCE_OP::kIdentity),
                                        __builtin_bit_cast(int, v), DPP_CTRL, DPP_ROW_MASK, 0xf,
                                        false);
  return __builtin_bit_cast(float, moved);
}

// N VALUES OVER THE WAVE, in place, every lane left holding the results. The moved value first:
// only a VOP2's src0 takes DPP, so (dpp(x), x) fuses each step into one v_add_f32_dpp where
// (x, dpp(x)) was a v_mov_b32_dpp and an add (ISA 2026-10-04T02-22-41Z). DPP, as rocPRIM's and
// Composable Kernel's wave reductions do: each step is a VALU operand from another lane. The
// butterfly __shfl_xor it replaced compiled to ds_bpermute, an LDS round trip a step, six of them
// waiting on each other (ISA 2026-09-30T20-43-18Z; stamps: 0.92 us for one block_reduce).
//   swap neighbours, swap pairs (quad_perm [1,0,3,2], [2,3,0,1]): each quad holds its sum
//   mirror in 8 lanes, then in 16 (row_half_mirror, row_mirror): each row holds its sum
//   lane 15 into rows 1 and 3, lane 31 into rows 2 and 3 (row_bcast15, row_bcast31): lane 63
//   holds the wave's, and readlane hands it to every lane.
template <typename REDUCE_OP, int NUM_VALUES>
DINLINE void wave_reduce(float (&v)[NUM_VALUES]) {
  static_assert(kWaveSize == 64, "the DPP sequence is for 64-lane waves");
#pragma unroll
  for (int n = 0; n < NUM_VALUES; ++n) {
    float x = v[n];
    x = REDUCE_OP::apply(dpp<0xb1, 0xf, REDUCE_OP>(x), x);   // quad_perm [1,0,3,2]
    x = REDUCE_OP::apply(dpp<0x4e, 0xf, REDUCE_OP>(x), x);   // quad_perm [2,3,0,1]
    x = REDUCE_OP::apply(dpp<0x141, 0xf, REDUCE_OP>(x), x);  // row_half_mirror
    x = REDUCE_OP::apply(dpp<0x140, 0xf, REDUCE_OP>(x), x);  // row_mirror
    x = REDUCE_OP::apply(dpp<0x142, 0xa, REDUCE_OP>(x), x);  // row_bcast15 into rows 1, 3
    x = REDUCE_OP::apply(dpp<0x143, 0xc, REDUCE_OP>(x), x);  // row_bcast31 into rows 2, 3
    v[n] = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, x), 63));
  }
}

// N VALUES OVER THE BLOCK, in place: each wave reduces its N, one wave combines the waves' partials
// the same way (not every thread reading every partial: that was N x waves LDS reads a thread),
// and the N totals are broadcast through LDS once.
template <typename REDUCE_OP, int NUM_VALUES>
DINLINE void block_reduce(float (&v)[NUM_VALUES]) {
  __shared__ float partial[kBuild.kernels.max_waves][NUM_VALUES];
  __shared__ float total[NUM_VALUES];
  const int lane  = threadIdx.x % kWaveSize;
  const int wave  = threadIdx.x / kWaveSize;
  const int waves = (blockDim.x + kWaveSize - 1) / kWaveSize;
  wave_reduce<REDUCE_OP>(v);
  if (lane == 0) {
#pragma unroll
    for (int n = 0; n < NUM_VALUES; ++n) partial[wave][n] = v[n];
  }
  __syncthreads();
  if (wave == 0) {
#pragma unroll
    for (int n = 0; n < NUM_VALUES; ++n) {
      float x[1] = {lane < waves ? partial[lane][n] : REDUCE_OP::kIdentity};
      wave_reduce<REDUCE_OP>(x);
      if (lane == 0) total[n] = x[0];
    }
  }
  __syncthreads();
#pragma unroll
  for (int n = 0; n < NUM_VALUES; ++n) v[n] = total[n];
}

// A TILE REDUCED OVER AN AXIS, Triton's tl.sum(x, axis) for a float tile:
//   Axis::m  over the rows, every column's: the rows meet in LDS as they are held (a bf16 tile's
//            16 bytes a group, as the old two-shot's), reduced in fp32 in row order, and the
//            one-row fp32 result is the first THREADS_N threads' (a Tile of one row of threads)
//   Axis::n  over the columns, every row's, into `out` (a row's value in every thread of the row):
//            each thread's groups, then the row's threads (a wave's or the block's)
enum class Axis { m, n };

template <typename REDUCE_OP, Axis AXIS, typename TILE>
DINLINE Tile<typename TILE::Dtype, 1, TILE::kTileN, 1, TILE::kThreadsN, float> block_reduce(
    const TILE& t) {
  static_assert(AXIS == Axis::m, "a row's reduction (Axis::n) writes `out`");
  static_assert(TILE::kRows == 1, "a thread holds one row of the rows reduced");
  constexpr int kRowLanes = TILE::kThreadsM;
  using Acc = typename TILE::Acc;
  __shared__ Acc part[kRowLanes][TILE::kTileN];
  Tile<typename TILE::Dtype, 1, TILE::kTileN, 1, TILE::kThreadsN, float> out{1, t.N, 0, t.offs_n};
  const int lane_row = static_cast<int>(threadIdx.x) / TILE::kThreadsN;
#pragma unroll
  for (int k = 0; k < TILE::K; ++k)
#pragma unroll
    for (int j = 0; j < TILE::kPack; ++j) {
      part[lane_row][(t.lane() + k * TILE::kThreadsN) * TILE::kPack + j] = t.v[0][k][j];
    }
  __syncthreads();
  if (out.participates()) {
#pragma unroll
    for (int k = 0; k < TILE::K; ++k)
#pragma unroll
      for (int j = 0; j < TILE::kPack; ++j) {
        const int c = (out.lane() + k * TILE::kThreadsN) * TILE::kPack + j;
        float x = static_cast<float>(part[0][c]);
#pragma unroll
        for (int r = 1; r < kRowLanes; ++r) x = REDUCE_OP::apply(x, static_cast<float>(part[r][c]));
        out.v[0][k][j] = x;
      }
  }
  __syncthreads();  // before the next reduction writes `part`
  return out;
}

template <typename REDUCE_OP, Axis AXIS, typename TILE>
DINLINE void block_reduce(const TILE& t, float (&out)[TILE::kRows]) {
  static_assert(AXIS == Axis::n, "a column's reduction (Axis::m) returns a tile");
  static_assert(std::is_same_v<typename TILE::Acc, float>, "a reduction is of a float tile");
#pragma unroll
  for (int m = 0; m < TILE::kRows; ++m) {
    float x = REDUCE_OP::kIdentity;
#pragma unroll
    for (int k = 0; k < TILE::K; ++k)
#pragma unroll
      for (int j = 0; j < TILE::kPack; ++j)
        x = t.mask(k) != 0.0f ? REDUCE_OP::apply(x, t.v[m][k][j]) : x;
    out[m] = x;
  }
  if constexpr (TILE::kThreadsM == 1)
    block_reduce<REDUCE_OP>(out);
  else if constexpr (TILE::kThreadsN == kWaveSize)
    wave_reduce<REDUCE_OP>(out);
  else
    static_assert(TILE::kThreadsN == kWaveSize, "a row is the block's or a wave's");
}

// The LDS a block_reduce<Op, N> takes, for a kernel budgeting the rest.
constexpr int64_t block_reduce_lds_bytes(int n) {
  return (int64_t{kBuild.kernels.max_waves} + 1) * n * sizeof(float);
}

}  // namespace hip_comms
