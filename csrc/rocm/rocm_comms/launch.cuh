// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// WHICH KERNEL RUNS, AND HOW WIDE. The caller names an op; this picks one kernel from the
// flat list and its launch geometry, for gfx950. Nothing above C++ sees an algorithm or a
// geometry.

#pragma once

#include <cstdint>

#include "utils.cuh"

namespace hip_comms {

// What the caller asked for.
enum class Op : int {
  all_reduce            = 0,
  rms_norm              = 1,
  add_rms_norm          = 2,
  add_attn_res_rms_norm = 3,
  rms_norm_gemm_add     = 4,
};

// Every `__global__` there is, once. `none` is a decline: the caller runs the unfused ops.
enum class Kernel : int {
  none                           = -1,
  one_shot                       = 0,
  two_shot                       = 1,
  one_shot_rms_norm              = 2,
  two_shot_rms_norm              = 3,
  one_shot_add_rms_norm          = 4,
  two_shot_add_rms_norm          = 5,
  one_shot_add_attn_res_rms_norm = 6,
  two_shot_add_attn_res_rms_norm = 7,
  one_shot_rms_norm_gemm_add     = 8,
  two_shot_rms_norm_gemm_add     = 9,
};

struct Launch {
  Kernel kernel;
  int blocks;
  int threads;
};

// Each op has a one-shot and a two-shot kernel, numbered as a pair.
constexpr Kernel kernel_of(Op op, bool two_shot) {
  return static_cast<Kernel>(2 * static_cast<int>(op) + (two_shot ? 1 : 0));
}
constexpr Op op_of(Kernel k) { return static_cast<Op>(static_cast<int>(k) / 2); }
constexpr bool is_two_shot(Kernel k) { return static_cast<int>(k) % 2 == 1; }

// THE gfx950 TABLE, until the microbench sweep replaces it. One-shot moves ngpus x the
// bytes and pays one barrier; two-shot moves about 2x and pays a sync, so one-shot wins
// while the buffer is small. 16 blocks of 512 threads is vLLM's setting, not yet measured
// here.
constexpr int64_t kOneShotMaxBytes = int64_t{512} << 10;
constexpr int kBlocks              = 16;
constexpr int kThreads             = 512;
// THE GEMM TAIL IS DECLINED at every size: its GEMM runs on the collective's 16 blocks,
// one row at a time, and loses to all-reduce + norm + hipBLASLt even at one row (50 vs 37
// us; 272 vs 37 at 16 rows, microbench 2026-09-27). Forcing it through the override still
// runs it.
inline Launch pick(Op op, int64_t rows, int64_t bytes) {
  if (op == Op::rms_norm_gemm_add) return {Kernel::none, 0, 0};
  const bool one_shot =
      bytes <= kOneShotMaxBytes && !(op == Op::rms_norm_gemm_add && rows > kGemmRows);
  return {kernel_of(op, !one_shot), kBlocks, kThreads};
}

}  // namespace hip_comms
