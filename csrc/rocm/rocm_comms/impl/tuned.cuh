// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE TUNED TABLE: for each op, at a world and a row width, the template and KernelConfig that won
// from a number of tokens up. Data only, written by the tuner (the bench in tune mode) from a
// sweep of every template's configs (impl/templates.cuh) on the target; select reads it
// (impl/select.cuh). A row says what ran fastest, not why: the why is the sweep's figure, cited in
// the row's comment.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <cstdint>
#include <span>

namespace hip_comms {

// From `tokens` rows up (to the next row's), at `world` ranks and rows `hidden` elements wide, op
// `op` runs `fn` at `config`.
struct Tuned {
  Op op;
  int world;
  int64_t hidden;
  int64_t tokens;
  Template fn;
  KernelConfig config;
};

// gfx950 on n11, bf16, 8 ranks. Transcribed from the size thresholds the sweeps set (each row's
// crossover cites its run) at the widths we run, until the tuner writes it.
constexpr Tuned kGfx950Tuned[] = {
    // rms_norm: one-shot through 128 KiB (at 64 KiB it lost at 16 tokens, 11.43 against 10.56 us;
    // 2026-09-30T21-06-57Z); the push through 1.75 MiB (256 tokens of 3584: 18.04 against the
    // pull's 18.41), the pull from 2.6 MiB (2026-10-01T03-26-44Z).
    {Op::all_reduce_rms_norm, 8, 3584, 1, Template::all_reduce_pull_one_shot_rms_norm,
     {1, 4096, 0, 0, 512, 16}},
    {Op::all_reduce_rms_norm, 8, 3584, 19, Template::all_reduce_push_two_shot_rms_norm,
     {1, 4096, 0, 0, 512, 256}},
    {Op::all_reduce_rms_norm, 8, 3584, 257, Template::all_reduce_pull_two_shot_rms_norm,
     {1, 4096, 0, 0, 512, 48}},
    {Op::all_reduce_rms_norm, 8, 7168, 1, Template::all_reduce_pull_one_shot_rms_norm,
     {1, 8192, 0, 0, 512, 16}},
    {Op::all_reduce_rms_norm, 8, 7168, 10, Template::all_reduce_push_two_shot_rms_norm,
     {1, 8192, 0, 0, 512, 256}},
    {Op::all_reduce_rms_norm, 8, 7168, 129, Template::all_reduce_pull_two_shot_rms_norm,
     {1, 8192, 0, 0, 512, 48}},
    // add_rms_norm: the one-shot not swept (rms_norm's); the push through 1.31 MiB (192 tokens of
    // 3584: 15.48 against the pull's 16.28), the pull at 1.75 MiB (18.36 against the push's 18.46;
    // 2026-10-01T03-26-44Z).
    {Op::all_reduce_add_rms_norm, 8, 3584, 1, Template::all_reduce_pull_one_shot_add_rms_norm,
     {1, 4096, 0, 0, 512, 16}},
    {Op::all_reduce_add_rms_norm, 8, 3584, 19, Template::all_reduce_push_two_shot_add_rms_norm,
     {1, 4096, 0, 0, 512, 256}},
    {Op::all_reduce_add_rms_norm, 8, 3584, 193, Template::all_reduce_pull_two_shot_add_rms_norm,
     {1, 4096, 0, 0, 512, 48}},
    {Op::all_reduce_add_rms_norm, 8, 7168, 1, Template::all_reduce_pull_one_shot_add_rms_norm,
     {1, 8192, 0, 0, 512, 16}},
    {Op::all_reduce_add_rms_norm, 8, 7168, 10, Template::all_reduce_push_two_shot_add_rms_norm,
     {1, 8192, 0, 0, 512, 256}},
    {Op::all_reduce_add_rms_norm, 8, 7168, 97, Template::all_reduce_pull_two_shot_add_rms_norm,
     {1, 8192, 0, 0, 512, 48}},
    // AttnRes: the push from 1 token (it beat the one-shot, 12.89 against 13.68 us; 14.03 against
    // 15.18 at 8; 2026-10-01T03-57-23Z), through 3.5 MiB against the pull at its grid (256 tokens
    // of 7168: 34.8 against 35.8-38.7 us; 2026-10-01T22-56-58Z, 2026-10-01T23-00-47Z).
    {Op::all_reduce_add_attn_res_rms_norm, 8, 3584, 1,
     Template::all_reduce_push_two_shot_add_attn_res_rms_norm, {1, 4096, 1, 0, 512, 256}},
    {Op::all_reduce_add_attn_res_rms_norm, 8, 3584, 513,
     Template::all_reduce_pull_two_shot_add_attn_res_rms_norm, {1, 4096, 1, 0, 512, 192}},
    {Op::all_reduce_add_attn_res_rms_norm, 8, 7168, 1,
     Template::all_reduce_push_two_shot_add_attn_res_rms_norm, {1, 8192, 1, 0, 512, 256}},
    {Op::all_reduce_add_attn_res_rms_norm, 8, 7168, 257,
     Template::all_reduce_pull_two_shot_add_attn_res_rms_norm, {1, 8192, 1, 0, 512, 192}},
    // The GEMM tails: the one-shot through one GEMM pass of rows (16), where the one-shot kernel
    // once had to stop; not swept.
    {Op::all_reduce_rms_norm_gemm, 8, 3584, 1, Template::all_reduce_pull_one_shot_rms_norm_gemm,
     {16, 4096, kGemmTileK, 4, 512, 56}},
    {Op::all_reduce_rms_norm_gemm, 8, 3584, 17, Template::all_reduce_pull_two_shot_rms_norm_gemm,
     {16, 4096, kGemmTileK, 4, 512, 56}},
    {Op::all_reduce_rms_norm_gemm, 8, 7168, 1, Template::all_reduce_pull_one_shot_rms_norm_gemm,
     {16, 8192, kGemmTileK, 4, 512, 56}},
    {Op::all_reduce_rms_norm_gemm, 8, 7168, 17, Template::all_reduce_pull_two_shot_rms_norm_gemm,
     {16, 8192, kGemmTileK, 4, 512, 56}},
    {Op::all_reduce_rms_norm_gemm_add, 8, 3584, 1,
     Template::all_reduce_pull_one_shot_rms_norm_gemm_add, {16, 4096, kGemmTileK, 4, 512, 56}},
    {Op::all_reduce_rms_norm_gemm_add, 8, 3584, 17,
     Template::all_reduce_pull_two_shot_rms_norm_gemm_add, {16, 4096, kGemmTileK, 4, 512, 56}},
    {Op::all_reduce_rms_norm_gemm_add, 8, 7168, 1,
     Template::all_reduce_pull_one_shot_rms_norm_gemm_add, {16, 8192, kGemmTileK, 4, 512, 56}},
    {Op::all_reduce_rms_norm_gemm_add, 8, 7168, 17,
     Template::all_reduce_pull_two_shot_rms_norm_gemm_add, {16, 8192, kGemmTileK, 4, 512, 56}},
    // The one-all-reduce tail, [T, 17920] (a latent of 3584): the one-shot while there are fewer
    // rows than ranks, the row two-shot from a row a rank (10.8 against 12.3 us at 1 token, 13.2
    // against 14.3 at 8 the other way; 2026-10-01T21-20-57Z).
    {Op::all_reduce_rms_scale_add, 8, 17920, 1, Template::all_reduce_pull_one_shot_rms_scale_add,
     {1, 4096, 0, 0, 512, 256}},
    {Op::all_reduce_rms_scale_add, 8, 17920, 8, Template::all_reduce_pull_two_shot_rms_scale_add,
     {1, 4096, 0, 0, 512, 32}},
};

// THE TARGET'S TABLE.
constexpr std::span<const Tuned> kTargetTuned = kGfx950Tuned;

}  // namespace hip_comms
