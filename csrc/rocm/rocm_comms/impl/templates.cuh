// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE CATALOG: what each Template is (its op, its shot) and the KernelConfigs it is built for.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <cstdint>
#include <optional>
#include <span>
#include <string>

namespace hip_comms {

// A ROW TEMPLATE'S CONFIGS: the KernelConfigs it is built for, in the order select prefers them
// (its default for a call is the first whose tile covers the call's row). The compiled part
// (tile_m, tile_n, threads_per_block) is what dispatch instantiates, what check accepts and what a
// tuner searches; blocks_per_grid is each config's launch (cut to the rows where a block takes a
// row), and a caller may force another. tile_n counts the elements of a 16-bit dtype, the only ones
// built. A config that spills shows in its code object (Handle::resources_of) and loses on the
// clock; none is ruled out by policy. Each grid cites the sweep that set it, at 512 threads (256
// was worse for the GEMM tail, 2026-09-28; not swept for the others, which copy it).

// The norms: one-shot, the push two-shot (a column split), the pull two-shot (a row split).
// Not swept: grid_of cuts it to the rows, so it matters only past 16 rows.
constexpr KernelConfig kRmsNormOneShotConfigs[] = {
    {1, 4096, 0, 0, 512, 16}, {1, 8192, 0, 0, 512, 16}, {1, 16384, 0, 0, 512, 16}};
// 256 the best of 48-256 at 192-256 tokens (15.17 and 18.04 against 15.55 and 18.11 at 128;
// 2026-10-01T03-26-44Z); a row a block below that.
constexpr KernelConfig kRmsNormPushConfigs[] = {
    {1, 4096, 0, 0, 512, 256}, {1, 8192, 0, 0, 512, 256}, {1, 16384, 0, 0, 512, 256}};
// Pipelined, 48 the best of 36-96 at 2048-4096 tokens (79.3 and 145.4 against 81.0 and 147.6 at
// 36), within 0.8 of 36 below (2026-10-01T02-59-52Z).
constexpr KernelConfig kRmsNormPullConfigs[] = {{1, 4096, 0, 0, 512, 48}, {1, 8192, 0, 0, 512, 48}};
// Not swept: rms_norm's.
constexpr KernelConfig kAddRmsNormOneShotConfigs[] = {
    {1, 4096, 0, 0, 512, 16}, {1, 8192, 0, 0, 512, 16}, {1, 16384, 0, 0, 512, 16}};
// 256 the best of 48-256 at 192 tokens (15.48 against 16.75 at 128; 2026-10-01T03-26-44Z).
constexpr KernelConfig kAddRmsNormPushConfigs[] = {
    {1, 4096, 0, 0, 512, 256}, {1, 8192, 0, 0, 512, 256}, {1, 16384, 0, 0, 512, 256}};
// 48 the best of 36-96 at every size from 512 to 4096 tokens (80.6 and 149.7 us at 2048 and 4096
// against 84.3 and 154.7 at 36; 2026-10-01T02-59-52Z).
constexpr KernelConfig kAddRmsNormPullConfigs[] = {{1, 4096, 0, 0, 512, 48},
                                                   {1, 8192, 0, 0, 512, 48}};

// AttnRes, one source a step (TILE_K): at 4 (Triton's tile) the row got slower, 6.48 -> 8.16 us a
// row and 147.5 -> 171.0 at 4096 tokens, with 100 -> 166 VGPRs (stamps 2026-10-01T06-26-55Z
// against 04-15-48Z). The one-shot's grid not swept: the norms'.
constexpr KernelConfig kAttnResOneShotConfigs[] = {{1, 4096, 1, 0, 512, 16},
                                                   {1, 8192, 1, 0, 512, 16}};
// 256 the best of 32-256 at 256-1024 tokens (34.58, 70.78, 143.98 against 39.93, 83.15, 159.33 at
// 128), a row a block below that (2026-10-01T03-57-23Z).
constexpr KernelConfig kAttnResPushConfigs[] = {{1, 4096, 1, 0, 512, 256},
                                                {1, 8192, 1, 0, 512, 256}};
// The column split wants a wide grid (AttnRes is compute a row): at 7168, 192 is within about 5%
// of the best of 16-256 from 512 to 4096 tokens; 4096 at 447.2 us against 1160.7 at the 36 it had
// (2026-10-01T22-56-58Z, 2026-10-01T23-00-47Z). TILE_M 2 and 4 lost at every grid, best 480.0 us
// at 128 blocks against 412.8 at 1 on 192 (4096 x 7168, 2026-10-02T21-19-22Z), so they come last.
constexpr KernelConfig kAttnResPullConfigs[] = {
    {1, 4096, 1, 0, 512, 192}, {1, 8192, 1, 0, 512, 192}, {2, 4096, 1, 0, 512, 192},
    {2, 8192, 1, 0, 512, 192}, {4, 4096, 1, 0, 512, 192}, {4, 8192, 1, 0, 512, 192}};

// The GEMM tails, both ops and both shots: 16 rows a GEMM pass (one fp32 accumulator a row in each
// lane); TILE_K what gfx950's LDS stages at 16 rows beside the widest block's partials; SLICE_K 4,
// picked at Kimi-K3's shape, 1, 2 and 8 worse at 1 row (2026-09-28); 56 blocks the best at 1 row
// (2026-09-28, log). Not swept for the rest, which copy it.
constexpr int kGemmTileK = gemm_tile_k_fit(kTarget, 16);
constexpr KernelConfig kGemmTailConfigs[] = {{16, 4096, kGemmTileK, 4, 512, 56},
                                             {16, 8192, kGemmTileK, 4, 512, 56},
                                             {16, 16384, kGemmTileK, 4, 512, 56}};
constexpr bool gemm_tail_configs_fit() {
  for (const KernelConfig& c : kGemmTailConfigs)
    if (!gemm_fits(kTarget, c.tile_m, c.tile_k, c.slice_k, c.threads_per_block)) return false;
  return true;
}
static_assert(gemm_tail_configs_fit(),
              "every GEMM-tail config holds its block in the target's LDS");

// The one-all-reduce tail. The one-shot: a block a (row, slice) up to one a compute unit.
constexpr KernelConfig kRmsScaleAddOneShotConfigs[] = {{1, 4096, 0, 0, 512, 256},
                                                       {1, 8192, 0, 0, 512, 256}};
// About 32 blocks keeps the links fed; more queue behind them: at [T, 17920] bf16, within 1% of the
// best of 8-48 blocks from 8 to 4096 tokens, and 1024 tokens 146.9 us at 32 against 258.4 at 256
// (2026-10-01T22-07-13Z).
constexpr KernelConfig kRmsScaleAddTwoShotConfigs[] = {{1, 4096, 0, 0, 512, 32},
                                                       {1, 8192, 0, 0, 512, 32}};

// What each template is, in Template's order: its op, its shot, and its builds (none for the plain
// all-reduce, which has no tile).
struct TemplateInfo {
  Template fn;
  const char* name;
  Op op;
  bool two_shot;
  std::span<const KernelConfig> configs;
};

// A TEMPLATE AND ITS NAME FROM ONE TOKEN, so the name cannot differ from the enum's.
#define HIP_COMMS_NAMED(t) Template::t, #t

constexpr TemplateInfo kTemplates[] = {
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot), Op::all_reduce, false, {}},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot), Op::all_reduce, true, {}},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_rms_norm), Op::all_reduce_rms_norm, false,
     kRmsNormOneShotConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_rms_norm), Op::all_reduce_rms_norm, true,
     kRmsNormPullConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_add_rms_norm), Op::all_reduce_add_rms_norm, false,
     kAddRmsNormOneShotConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_add_rms_norm), Op::all_reduce_add_rms_norm, true,
     kAddRmsNormPullConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_add_attn_res_rms_norm),
     Op::all_reduce_add_attn_res_rms_norm, false, kAttnResOneShotConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_add_attn_res_rms_norm),
     Op::all_reduce_add_attn_res_rms_norm, true, kAttnResPullConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_rms_norm_gemm_add), Op::all_reduce_rms_norm_gemm_add,
     false, kGemmTailConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_rms_norm_gemm_add), Op::all_reduce_rms_norm_gemm_add,
     true, kGemmTailConfigs},
    {HIP_COMMS_NAMED(all_reduce_push_two_shot_rms_norm), Op::all_reduce_rms_norm, true,
     kRmsNormPushConfigs},
    {HIP_COMMS_NAMED(all_reduce_push_two_shot_add_rms_norm), Op::all_reduce_add_rms_norm, true,
     kAddRmsNormPushConfigs},
    {HIP_COMMS_NAMED(all_reduce_push_two_shot_add_attn_res_rms_norm),
     Op::all_reduce_add_attn_res_rms_norm, true, kAttnResPushConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_rms_norm_gemm), Op::all_reduce_rms_norm_gemm, false,
     kGemmTailConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_rms_norm_gemm), Op::all_reduce_rms_norm_gemm, true,
     kGemmTailConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_rms_scale_add), Op::all_reduce_rms_scale_add, false,
     kRmsScaleAddOneShotConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_rms_scale_add), Op::all_reduce_rms_scale_add, true,
     kRmsScaleAddTwoShotConfigs},
};
#undef HIP_COMMS_NAMED
constexpr int kNumTemplates = sizeof(kTemplates) / sizeof(TemplateInfo);

constexpr bool templates_in_order() {
  for (int i = 0; i < kNumTemplates; ++i)
    if (static_cast<int>(kTemplates[i].fn) != i) return false;
  return true;
}
static_assert(templates_in_order(), "kTemplates must list every Template in its order");

constexpr const TemplateInfo& info(Template k) { return kTemplates[static_cast<int>(k)]; }
constexpr Op op_of(Template k) { return info(k).op; }
constexpr const char* to_string(Template k) { return info(k).name; }
// The template a name names, or none: how a caller forces one (the bench's sweeps, the tests).
inline std::optional<Template> template_named(const std::string& name) {
  for (const TemplateInfo& t : kTemplates)
    if (name == t.name) return t.fn;
  return std::nullopt;
}
constexpr bool is_two_shot(Template k) { return info(k).two_shot; }
// A norm then a GEMM, written or added: their GEMM phase strides over column tiles.
constexpr bool gemms(Op op) {
  return op == Op::all_reduce_rms_norm_gemm || op == Op::all_reduce_rms_norm_gemm_add;
}

constexpr std::span<const KernelConfig> configs_of(Template k) { return info(k).configs; }
constexpr bool has_tiles(Template k) { return !configs_of(k).empty(); }

// Whether two configs are one build: the same tile at the same threads (the grid is a launch's).
constexpr bool same_build(const KernelConfig& a, const KernelConfig& b) {
  return a.tile_m == b.tile_m && a.tile_n == b.tile_n && a.tile_k == b.tile_k &&
         a.slice_k == b.slice_k && a.threads_per_block == b.threads_per_block;
}
// Whether template `k` builds `c`'s tile at `c`'s threads.
constexpr bool built(Template k, const KernelConfig& c) {
  for (const KernelConfig& b : configs_of(k))
    if (same_build(b, c)) return true;
  return false;
}
constexpr bool built_at(Template k, int threads_per_block) {
  for (const KernelConfig& b : configs_of(k))
    if (b.threads_per_block == threads_per_block) return true;
  return false;
}
// Whether two templates are built for the same tiles and threads (one dispatch serves both).
constexpr bool same_builds(Template a, Template b) {
  if (configs_of(a).size() != configs_of(b).size()) return false;
  for (size_t i = 0; i < configs_of(a).size(); ++i)
    if (!same_build(configs_of(a)[i], configs_of(b)[i])) return false;
  return true;
}

// SELECT'S DEFAULT for `k` on a row of `cols`: its first config whose tile covers it, or none.
constexpr std::optional<KernelConfig> config_for(Template k, int64_t cols) {
  for (const KernelConfig& c : configs_of(k))
    if (c.tile_n >= cols) return c;
  return std::nullopt;
}

// A FORCED LAUNCH'S TILE: `k`'s default tile (its first config's tile_m, tile_k, slice_k) at
// `threads_per_block`, with the smallest built TILE_N covering `cols`; tile_n 0 when none does.
constexpr KernelConfig tile_for(Template k, int64_t cols, int threads_per_block) {
  KernelConfig t = configs_of(k)[0];
  t.threads_per_block = threads_per_block;
  t.tile_n = 0;
  for (const KernelConfig& c : configs_of(k))
    if (c.tile_m == t.tile_m && c.tile_k == t.tile_k && c.slice_k == t.slice_k &&
        c.threads_per_block == threads_per_block && c.tile_n >= cols &&
        (t.tile_n == 0 || c.tile_n < t.tile_n))
      t.tile_n = c.tile_n;
  return t;
}

}  // namespace hip_comms
