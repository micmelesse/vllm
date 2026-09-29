// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// WHAT RUNS FOR A CALL: `tune(op, input, hw)` splits on the op and calls that op's own
// `tune_<op>`, which picks the kernel AND its launch config from the hardware's facts
// (hardware.cuh) and the input, one rule in one place, with the sweep it came from written beside
// it. An op not yet swept says so and states what it runs. launch.cuh says what kernels exist.

#pragma once

#include <algorithm>
#include <cstdint>

#include "hardware.cuh"
#include "launch.cuh"
#include "p2p/p2p.cuh"

namespace hip_comms {

// WHAT A CALL IS: its shape, and the precision it accepts. Tuning reads it and never changes the
// answer.
struct Input {
  int64_t rows;    // tokens
  int64_t hidden;  // a row's length, in elements
  int elem_bytes;  // 2 for bf16 and fp16
  int64_t cols;    // the GEMM tail's output columns (its weight is [cols, hidden]); 0 without one
  int quant_bits;  // the precision the caller accepts on the wire: 16 (exact), 8 or 4. A push
                   // kernel's codec, passed through untouched; the pull kernels move T itself.
};

// WHAT THE HARDWARE MOVES: to it, a plain all-reduce is only its byte count.
constexpr int64_t bytes(Input in) { return in.rows * in.hidden * in.elem_bytes; }

constexpr int64_t kKiB = 1024;

// THE GEMM TAIL'S LANES PER COLUMN: a column's lanes split its reduction over the hidden size, so a
// wave covers wave / lanes columns. 4 was picked at Kimi-K3's shape and is not yet swept; it is a
// template instantiation (1, 2, 4 or 8). Every GEMM-tail launch takes it, forced or not; no other
// kernel has lanes.
constexpr int kGemmLanesPerCol = 4;
static_assert(kGemmLanesPerCol == 1 || kGemmLanesPerCol == 2 || kGemmLanesPerCol == 4 ||
                  kGemmLanesPerCol == 8,
              "the GEMM tail is built for 1, 2, 4 or 8 lanes per column");

// `k` at `blocks` x `threads`: its grid cut to the rows where it gives each block a row, its lanes
// if it is the GEMM tail, the call's precision passed through.
constexpr Launch at(Kernel k, Input in, int blocks, int threads) {
  const int lanes = op_of(k) == Op::all_reduce_rms_norm_gemm_add ? kGemmLanesPerCol : 0;
  return {k, grid_of(k, blocks, in.rows), threads, lanes, in.quant_bits};
}

// No kernel: the caller runs the unfused ops.
constexpr Launch declined() { return {Kernel::none, 0, 0, 0, 0}; }

// =================================================================================================
// ALL-REDUCE: the critical path, from the launch-config sweep on n11 (bench, 2026-09-29T19-24-48Z:
// tokens 1-64 at hidden 3584 bf16, blocks 1-64, threads 64-512).
//
// ONE WAVE PER BLOCK, AS MANY BLOCKS AS THE SIGNAL SLOTS ALLOW. Waves in one block only add a
// barrier inside it: at 1 token one-shot took 7.8 us at 64 threads, 8.1 at 128, 9.0 at 256 and
// 10.9 at 512. Blocks run apart, and from 16 tokens up more of them help, best at the slot cap.
// Within 0.1 us of the sweep's fastest config at every size from 1 to 64 tokens.
//
// PULL ONE-SHOT UP TO 128 KiB, PULL TWO-SHOT PAST IT, at that width. One-shot reads every peer's
// whole buffer ((N-1)P) in one round trip; two-shot moves less (2(N-1)/N P) in two. One-shot won at
// 112 KiB (10.12 vs 10.67 us), two-shot at 224 KiB (10.96 vs 13.13); the lines cross near 135 KiB.
// To be derived from the link's latency and bandwidth once hardware.cuh carries them. Push is
// parked (2x pull at 1 token, BACKLOG).
// =================================================================================================

constexpr int64_t kPullOneShotMaxBytes = 128 * kKiB;

constexpr Launch tune_all_reduce(Input in, const Hardware& hw) {
  const Kernel k = bytes(in) <= kPullOneShotMaxBytes ? Kernel::all_reduce_pull_one_shot
                                                      : Kernel::all_reduce_pull_two_shot;
  return at(k, in, std::min(p2p::kMaxBlocks, hw.compute_units), hw.wave_size);
}

// =================================================================================================
// THE FUSED OPS, NOT YET SWEPT: pull one-shot up to 512 KiB at 16 blocks, pull two-shot past it at
// 36, 512 threads each, as the 2026-09-27 microbench chose (one-shot 12.9 vs two-shot 27.1 us at
// 16 x 7168), until each is measured the way the all-reduce was.
// =================================================================================================

constexpr int64_t kFusedOneShotMaxBytes = 512 * kKiB;

constexpr Launch fused_untuned(Kernel one_shot, Kernel two_shot, Input in) {
  return bytes(in) <= kFusedOneShotMaxBytes ? at(one_shot, in, 16, 512) : at(two_shot, in, 36, 512);
}

constexpr Launch tune_all_reduce_rms_norm(Input in, const Hardware&) {
  return fused_untuned(Kernel::all_reduce_pull_one_shot_rms_norm,
                       Kernel::all_reduce_pull_two_shot_rms_norm, in);
}

constexpr Launch tune_all_reduce_add_rms_norm(Input in, const Hardware&) {
  return fused_untuned(Kernel::all_reduce_pull_one_shot_add_rms_norm,
                       Kernel::all_reduce_pull_two_shot_add_rms_norm, in);
}

// PAST ONE-SHOT'S RANGE IT DECLINES: two-shot lost to unfused there (2143 vs 1093 us at 4096 rows).
constexpr Launch tune_all_reduce_add_attn_res_rms_norm(Input in, const Hardware&) {
  return bytes(in) <= kFusedOneShotMaxBytes
             ? at(Kernel::all_reduce_pull_one_shot_add_attn_res_rms_norm, in, 16, 512)
             : declined();
}

// DECLINED until its rewrite measures faster than unfused (37 us at 16 rows); it runs only forced,
// by the tests and the bench.
constexpr Launch tune_all_reduce_rms_norm_gemm_add(Input, const Hardware&) { return declined(); }

// =================================================================================================
// THE ONE ENTRY: the op's own function.
// =================================================================================================

constexpr Launch tune(Op op, Input in, const Hardware& hw) {
  switch (op) {
    case Op::all_reduce: return tune_all_reduce(in, hw);
    case Op::all_reduce_rms_norm: return tune_all_reduce_rms_norm(in, hw);
    case Op::all_reduce_add_rms_norm: return tune_all_reduce_add_rms_norm(in, hw);
    case Op::all_reduce_add_attn_res_rms_norm: return tune_all_reduce_add_attn_res_rms_norm(in, hw);
    case Op::all_reduce_rms_norm_gemm_add: return tune_all_reduce_rms_norm_gemm_add(in, hw);
  }
  return declined();
}

}  // namespace hip_comms
