// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE OPS: each op Python calls, its name, and the kernels it picks among on the target: for each
// world and row width, the template and KernelConfig that won from a number of rows up. The lists
// are data, written by the tuner (the bench in tune mode) from a sweep of every template's configs
// (impl/templates.cuh); select reads them (impl/select.cuh). An entry says what ran fastest, not
// why: the why is the sweep's figure, cited beside it.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <cstdint>
#include <span>

namespace hip_comms {

// From `rows` rows up (to the next entry's), at `world` ranks and rows `hidden` elements wide:
// template `fn` at `config`.
struct TunedKernel {
  int world;
  int64_t rows;
  int64_t hidden;
  Template fn;
  KernelConfig config;
};

// gfx950 on n11, bf16, 8 ranks. Transcribed from the size thresholds the sweeps set (each
// crossover cites its run) at the widths we run, until the tuner writes them.
// rms_norm: one-shot through 128 KiB (at 64 KiB it lost at 16 tokens, 11.43 against 10.56 us;
// 2026-09-30T21-06-57Z); the push through 1.75 MiB (256 tokens of 3584: 18.04 against the
// pull's 18.41), the pull from 2.6 MiB (2026-10-01T03-26-44Z).
constexpr TunedKernel kRmsNormKernels[] = {
    {8, 1, 3584, Template::all_reduce_pull_one_shot_rms_norm, {1, 4096, 0, 0, 512, 16}},
    {8, 19, 3584, Template::all_reduce_push_two_shot_rms_norm, {1, 4096, 0, 0, 512, 256}},
    {8, 257, 3584, Template::all_reduce_pull_two_shot_rms_norm, {1, 4096, 0, 0, 512, 48}},
    {8, 1, 7168, Template::all_reduce_pull_one_shot_rms_norm, {1, 8192, 0, 0, 512, 16}},
    {8, 10, 7168, Template::all_reduce_push_two_shot_rms_norm, {1, 8192, 0, 0, 512, 256}},
    {8, 129, 7168, Template::all_reduce_pull_two_shot_rms_norm, {1, 8192, 0, 0, 512, 48}},
};

// add_rms_norm: the one-shot not swept (rms_norm's); the push through 1.31 MiB (192 tokens of
// 3584: 15.48 against the pull's 16.28), the pull at 1.75 MiB (18.36 against the push's 18.46;
// 2026-10-01T03-26-44Z).
constexpr TunedKernel kAddRmsNormKernels[] = {
    {8, 1, 3584, Template::all_reduce_pull_one_shot_add_rms_norm, {1, 4096, 0, 0, 512, 16}},
    {8, 19, 3584, Template::all_reduce_push_two_shot_add_rms_norm, {1, 4096, 0, 0, 512, 256}},
    {8, 193, 3584, Template::all_reduce_pull_two_shot_add_rms_norm, {1, 4096, 0, 0, 512, 48}},
    {8, 1, 7168, Template::all_reduce_pull_one_shot_add_rms_norm, {1, 8192, 0, 0, 512, 16}},
    {8, 10, 7168, Template::all_reduce_push_two_shot_add_rms_norm, {1, 8192, 0, 0, 512, 256}},
    {8, 97, 7168, Template::all_reduce_pull_two_shot_add_rms_norm, {1, 8192, 0, 0, 512, 48}},
};

// AttnRes: the push from 1 token (it beat the one-shot, 12.89 against 13.68 us; 14.03 against
// 15.18 at 8; 2026-10-01T03-57-23Z), through 3.5 MiB against the pull at its grid (256 tokens
// of 7168: 34.8 against 35.8-38.7 us; 2026-10-01T22-56-58Z, 2026-10-01T23-00-47Z).
constexpr TunedKernel kAttnResKernels[] = {
    {8, 1, 3584, Template::all_reduce_push_two_shot_add_attn_res_rms_norm,
     {1, 4096, 1, 0, 512, 256}},
    {8, 513, 3584, Template::all_reduce_pull_two_shot_add_attn_res_rms_norm,
     {1, 4096, 1, 0, 512, 192}},
    {8, 1, 7168, Template::all_reduce_push_two_shot_add_attn_res_rms_norm,
     {1, 8192, 1, 0, 512, 256}},
    {8, 257, 7168, Template::all_reduce_pull_two_shot_add_attn_res_rms_norm,
     {1, 8192, 1, 0, 512, 192}},
};

// The GEMM tails: the one-shot through one GEMM pass of rows (16), where the one-shot kernel
// once had to stop; not swept.
constexpr TunedKernel kRmsNormGemmKernels[] = {
    {8, 1, 3584, Template::all_reduce_pull_one_shot_rms_norm_gemm,
     {16, 4096, kGemmTileK, 4, 512, 56}},
    {8, 17, 3584, Template::all_reduce_pull_two_shot_rms_norm_gemm,
     {16, 4096, kGemmTileK, 4, 512, 56}},
    {8, 1, 7168, Template::all_reduce_pull_one_shot_rms_norm_gemm,
     {16, 8192, kGemmTileK, 4, 512, 56}},
    {8, 17, 7168, Template::all_reduce_pull_two_shot_rms_norm_gemm,
     {16, 8192, kGemmTileK, 4, 512, 56}},
};

// As rms_norm_gemm's.
constexpr TunedKernel kRmsNormGemmAddKernels[] = {
    {8, 1, 3584, Template::all_reduce_pull_one_shot_rms_norm_gemm_add,
     {16, 4096, kGemmTileK, 4, 512, 56}},
    {8, 17, 3584, Template::all_reduce_pull_two_shot_rms_norm_gemm_add,
     {16, 4096, kGemmTileK, 4, 512, 56}},
    {8, 1, 7168, Template::all_reduce_pull_one_shot_rms_norm_gemm_add,
     {16, 8192, kGemmTileK, 4, 512, 56}},
    {8, 17, 7168, Template::all_reduce_pull_two_shot_rms_norm_gemm_add,
     {16, 8192, kGemmTileK, 4, 512, 56}},
};

// The one-all-reduce tail, [T, 17920] (a latent of 3584): the one-shot while there are fewer
// rows than ranks, the row two-shot from a row a rank (10.8 against 12.3 us at 1 token, 13.2
// against 14.3 at 8 the other way; 2026-10-01T21-20-57Z).
constexpr TunedKernel kRmsScaleAddKernels[] = {
    {8, 1, 17920, Template::all_reduce_pull_one_shot_rms_scale_add, {1, 4096, 0, 0, 512, 256}},
    {8, 8, 17920, Template::all_reduce_pull_two_shot_rms_scale_add, {1, 4096, 0, 0, 512, 32}},
};

// AN OP: what Python calls, by the name it calls it, and its tuned kernels (none for the plain
// all-reduce, whose grid is still derived; see select).
struct Op {
  OpType type;
  const char* name;
  std::span<const TunedKernel> kernels;
};

// AN OP AND ITS NAME FROM ONE TOKEN, so the name cannot differ from the enum's.
#define HIP_COMMS_NAMED(o) OpType::o, #o

constexpr Op kOps[] = {
    {HIP_COMMS_NAMED(all_reduce), {}},
    {HIP_COMMS_NAMED(all_reduce_rms_norm), kRmsNormKernels},
    {HIP_COMMS_NAMED(all_reduce_add_rms_norm), kAddRmsNormKernels},
    {HIP_COMMS_NAMED(all_reduce_add_attn_res_rms_norm), kAttnResKernels},
    {HIP_COMMS_NAMED(all_reduce_rms_norm_gemm_add), kRmsNormGemmAddKernels},
    {HIP_COMMS_NAMED(all_reduce_rms_norm_gemm), kRmsNormGemmKernels},
    {HIP_COMMS_NAMED(all_reduce_rms_scale_add), kRmsScaleAddKernels},
};
#undef HIP_COMMS_NAMED
constexpr int kNumOps = sizeof(kOps) / sizeof(Op);

constexpr bool ops_in_order() {
  for (int i = 0; i < kNumOps; ++i)
    if (static_cast<int>(kOps[i].type) != i) return false;
  return true;
}
static_assert(ops_in_order(), "kOps must list every OpType in its order");

constexpr const Op& op(OpType t) { return kOps[static_cast<int>(t)]; }
constexpr const char* to_string(OpType t) { return op(t).name; }

// EVERY OP BUT THE PLAIN ALL-REDUCE HAS KERNELS, each one its own and built.
constexpr bool ops_have_kernels() {
  for (const Op& o : kOps) {
    if (o.type == OpType::all_reduce) continue;
    if (o.kernels.empty()) return false;
    for (const TunedKernel& k : o.kernels)
      if (op_of(k.fn) != o.type || !built(k.fn, k.config)) return false;
  }
  return true;
}
static_assert(ops_have_kernels(), "an op has no kernels, or one is another op's or not built");

}  // namespace hip_comms
