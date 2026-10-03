// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE OPS, one file owning the idea: an OP is what a caller asks for (an API call, as Python names
// it). Each owns its name, the templates that can run it (the catalog below: each template and the
// KernelConfigs it is built for), and the kernels it
// picks among on the target: for each world and row width, the template and KernelConfig that won
// from a number of rows up, data the tuner writes (the bench in tune mode, from a sweep of every
// template's configs); select reads them (select.cuh). An entry says what ran fastest, not
// why: the why is the sweep's figure, cited beside it. Then each op's entry point: plan, then
// launch.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <hip/hip_runtime.h>

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <variant>

namespace hip_comms {

// =================================================================================================
// THE CATALOG: each template, and the KernelConfigs it is built for.
// =================================================================================================

// A TEMPLATE'S CONFIGS, each of its family (kernel.cuh), in the order select prefers them (its
// default for a call is the first whose tile covers the call's row). The fields compiled in (the
// tile, the threads) are what dispatch instantiates, what check accepts and what a tuner
// searches; the grid (cut to the tiles there are) and reduce_scatter_blocks are each config's
// launch, and a caller may force others. tile_n counts the elements of a 16-bit dtype, the only
// ones built. A config that spills shows in its code object (Handle::resources_of) and loses on
// the clock; none is ruled out by policy. Each grid cites the sweep that set it, at 512 threads
// (256 was worse for the GEMM tail, 2026-09-28; not swept for the others, which copy it).

// The norms: one-shot, the push two-shot (a column split), the pull two-shot (a row split).
// Not swept: select cuts it to the rows, so it matters only past 16 rows.
constexpr KernelConfig kRmsNormOneShotConfigs[] = {
    RowConfig{{512, 16}, 4096},
    RowConfig{{512, 16}, 8192},
    RowConfig{{512, 16}, 16384},
};
// 256 the best of 48-256 at 192-256 tokens (15.17 and 18.04 against 15.55 and 18.11 at 128;
// 2026-10-01T03-26-44Z); a row a block below that.
constexpr KernelConfig kRmsNormPushConfigs[] = {
    RowConfig{{512, 256}, 4096},
    RowConfig{{512, 256}, 8192},
    RowConfig{{512, 256}, 16384},
};
// Pipelined, 48 the best of 36-96 at 2048-4096 tokens (79.3 and 145.4 against 81.0 and 147.6 at
// 36), within 0.8 of 36 below (2026-10-01T02-59-52Z).
constexpr KernelConfig kRmsNormPullConfigs[] = {
    RowConfig{{512, 48}, 4096},
    RowConfig{{512, 48}, 8192},
};
// Not swept: rms_norm's.
constexpr KernelConfig kAddRmsNormOneShotConfigs[] = {
    RowConfig{{512, 16}, 4096},
    RowConfig{{512, 16}, 8192},
    RowConfig{{512, 16}, 16384},
};
// 256 the best of 48-256 at 192 tokens (15.48 against 16.75 at 128; 2026-10-01T03-26-44Z).
constexpr KernelConfig kAddRmsNormPushConfigs[] = {
    RowConfig{{512, 256}, 4096},
    RowConfig{{512, 256}, 8192},
    RowConfig{{512, 256}, 16384},
};
// 48 the best of 36-96 at every size from 512 to 4096 tokens (80.6 and 149.7 us at 2048 and 4096
// against 84.3 and 154.7 at 36; 2026-10-01T02-59-52Z).
constexpr KernelConfig kAddRmsNormPullConfigs[] = {
    RowConfig{{512, 48}, 4096},
    RowConfig{{512, 48}, 8192},
};

// AttnRes, one source a step (TILE_K): at 4 (Triton's tile) the row got slower, 6.48 -> 8.16 us a
// row and 147.5 -> 171.0 at 4096 tokens, with 100 -> 166 VGPRs (stamps 2026-10-01T06-26-55Z
// against 04-15-48Z). The one-shot's grid not swept: the norms'.
constexpr KernelConfig kAttnResOneShotConfigs[] = {
    AttnResConfig{{512, 16}, 4096, 1},
    AttnResConfig{{512, 16}, 8192, 1},
};
// 256 the best of 32-256 at 256-1024 tokens (34.58, 70.78, 143.98 against 39.93, 83.15, 159.33 at
// 128), a row a block below that (2026-10-01T03-57-23Z).
constexpr KernelConfig kAttnResPushConfigs[] = {
    AttnResConfig{{512, 256}, 4096, 1},
    AttnResConfig{{512, 256}, 8192, 1},
};
// The column split wants a wide grid (AttnRes is compute a row): at 7168, 192 is within about 5% of
// the best of 16-256 from 512 to 4096 tokens; 4096 at 447.2 us against 1160.7 at the 36 it had
// (2026-10-01T22-56-58Z, 2026-10-01T23-00-47Z). TILE_M 2 and 4 lost at every grid, best 480.0 us at
// 128 blocks against 412.8 at 1 on 192 (4096 x 7168, 2026-10-02T21-19-22Z), so they come last. Its
// reduce-scatter on the grid's first 32 blocks (reduce_scatter_blocks): its reads queue behind the
// links past a few dozen blocks while AttnRes is compute a row and wants every block. At 4096
// tokens it took 146.6 us on 32 blocks against 218.9 on 192, AttnRes 1110.5 against 199.4 (stamps,
// 2026-10-01T23-45-31Z and 2026-10-01T23-50-54Z).
constexpr KernelConfig kAttnResPullConfigs[] = {
    AttnResPullConfig{{512, 192}, 1, 4096, 1, 32},
    AttnResPullConfig{{512, 192}, 1, 8192, 1, 32},
    AttnResPullConfig{{512, 192}, 2, 4096, 1, 32},
    AttnResPullConfig{{512, 192}, 2, 8192, 1, 32},
    AttnResPullConfig{{512, 192}, 4, 4096, 1, 32},
    AttnResPullConfig{{512, 192}, 4, 8192, 1, 32},
};

// The GEMM tails, both ops and both shots: 16 rows a GEMM pass (one fp32 accumulator a row in each
// lane); TILE_K what gfx950's LDS stages at 16 rows beside the widest block's partials; SLICE_K 4,
// picked at Kimi-K3's shape, 1, 2 and 8 worse at 1 row (2026-09-28); 56 blocks the best at 1 row
// (2026-09-28, log). Not swept for the rest, which copy it.
constexpr int kGemmTileK = gemm_tile_k_fit(kTarget, 16);
constexpr KernelConfig kGemmTailConfigs[] = {
    GemmConfig{{512, 56}, 16, 4096, kGemmTileK, 4},
    GemmConfig{{512, 56}, 16, 8192, kGemmTileK, 4},
    GemmConfig{{512, 56}, 16, 16384, kGemmTileK, 4},
};
constexpr bool gemm_tail_configs_fit() {
  for (const KernelConfig& k : kGemmTailConfigs) {
    const GemmConfig& c = std::get<GemmConfig>(k);
    if (!gemm_fits(kTarget, c.tile_m, c.tile_k, c.slice_k, c.launch.threads_per_block))
      return false;
  }
  return true;
}
static_assert(gemm_tail_configs_fit(),
              "every GEMM-tail config holds its block in the target's LDS");

// The one-all-reduce tail. The one-shot: a block a (row, slice) up to one a compute unit.
constexpr KernelConfig kRmsScaleAddOneShotConfigs[] = {
    RowConfig{{512, 256}, 4096},
    RowConfig{{512, 256}, 8192},
};
// About 32 blocks keeps the links fed; more queue behind them: at [T, 17920] bf16, within 1% of the
// best of 8-48 blocks from 8 to 4096 tokens, and 1024 tokens 146.9 us at 32 against 258.4 at 256
// (2026-10-01T22-07-13Z).
constexpr KernelConfig kRmsScaleAddTwoShotConfigs[] = {
    RowConfig{{512, 32}, 4096},
    RowConfig{{512, 32}, 8192},
};

// What each template is, in Template's order: its op, its shot, and its builds (none for the plain
// all-reduce, which has no tile).
struct TemplateInfo {
  Template fn;
  const char* name;
  bool two_shot;
  bool push;  // peers push into this rank (else it pulls from them)
  std::span<const KernelConfig> configs;
};

// A TEMPLATE AND ITS NAME FROM ONE TOKEN, so the name cannot differ from the enum's.
#define HIP_COMMS_NAMED(t) Template::t, #t

constexpr TemplateInfo kTemplates[] = {
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot), false, false, {}},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot), true, false, {}},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_rms_norm), false, false,
     kRmsNormOneShotConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_rms_norm), true, false,
     kRmsNormPullConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_add_rms_norm), false, false,
     kAddRmsNormOneShotConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_add_rms_norm), true, false,
     kAddRmsNormPullConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_add_attn_res_rms_norm), false, false,
     kAttnResOneShotConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_add_attn_res_rms_norm), true, false,
     kAttnResPullConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_rms_norm_gemm_add), false, false, kGemmTailConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_rms_norm_gemm_add), true, false, kGemmTailConfigs},
    {HIP_COMMS_NAMED(all_reduce_push_two_shot_rms_norm), true, true,
     kRmsNormPushConfigs},
    {HIP_COMMS_NAMED(all_reduce_push_two_shot_add_rms_norm), true, true,
     kAddRmsNormPushConfigs},
    {HIP_COMMS_NAMED(all_reduce_push_two_shot_add_attn_res_rms_norm), true, true,
     kAttnResPushConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_rms_norm_gemm), false, false,
     kGemmTailConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_rms_norm_gemm), true, false,
     kGemmTailConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_rms_scale_add), false, false,
     kRmsScaleAddOneShotConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_rms_scale_add), true, false,
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
constexpr const char* to_string(Template k) { return info(k).name; }
constexpr bool is_two_shot(Template k) { return info(k).two_shot; }

constexpr std::span<const KernelConfig> configs_of(Template k) { return info(k).configs; }
constexpr bool has_tiles(Template k) { return !configs_of(k).empty(); }

// Whether two configs are one build: the same family, the same fields compiled in, at the same
// threads (the grid and reduce_scatter_blocks are a launch's).
constexpr bool same_build(const KernelConfig& a, const KernelConfig& b) {
  if (a.index() != b.index()) return false;
  return std::visit(
      [&](const auto& x) {
        const auto& y = std::get<std::decay_t<decltype(x)>>(b);
        bool same     = x.launch.threads_per_block == y.launch.threads_per_block;
        if constexpr (requires { x.tile_m; }) same = same && x.tile_m == y.tile_m;
        if constexpr (requires { x.tile_n; }) same = same && x.tile_n == y.tile_n;
        if constexpr (requires { x.tile_k; }) same = same && x.tile_k == y.tile_k;
        if constexpr (requires { x.slice_k; }) same = same && x.slice_k == y.slice_k;
        return same;
      },
      a);
}
// Whether template `k` builds `c`.
constexpr bool built(Template k, const KernelConfig& c) {
  for (const KernelConfig& b : configs_of(k))
    if (same_build(b, c)) return true;
  return false;
}
constexpr bool built_at(Template k, int threads_per_block) {
  for (const KernelConfig& b : configs_of(k))
    if (launch_of(b).threads_per_block == threads_per_block) return true;
  return false;
}
// THE FAMILY A TEMPLATE'S CONFIGS ARE (KernelConfig's index): its list's, and every entry the same
// (configs_one_family); the plain all-reduce's, which has no list, AllReduceConfig.
constexpr size_t family_of(Template k) {
  return configs_of(k).empty() ? 0 : configs_of(k)[0].index();
}
constexpr bool configs_one_family() {
  for (const TemplateInfo& t : kTemplates)
    for (const KernelConfig& c : t.configs)
      if (c.index() != family_of(t.fn)) return false;
  return true;
}
static_assert(configs_one_family(), "a template's configs are not all of one family");
// Whether two templates are built for the same tiles and threads (one dispatch serves both).
constexpr bool same_builds(Template a, Template b) {
  if (configs_of(a).size() != configs_of(b).size()) return false;
  for (size_t i = 0; i < configs_of(a).size(); ++i)
    if (!same_build(configs_of(a)[i], configs_of(b)[i])) return false;
  return true;
}

// WHICH OP the caller asked for: an all-reduce, alone or with what it fuses (its name and kernels
// are its `Op`, below).
enum class OpType : int {
  all_reduce                       = 0,
  all_reduce_rms_norm              = 1,
  all_reduce_add_rms_norm          = 2,
  all_reduce_add_attn_res_rms_norm = 3,
  all_reduce_rms_norm_gemm_add     = 4,
  all_reduce_rms_norm_gemm         = 5,
  all_reduce_rms_scale_add         = 6,
};

// From `rows` rows up (to the next entry's), at `world` ranks and rows `hidden` elements wide:
// template `fn` at `config`.
struct TunedKernel {
  int world;
  int64_t rows;
  int64_t hidden;
  Template fn;
  KernelConfig config;
};

// AN OP: what Python calls, by the name it calls it; the templates that can run it; and its tuned
// kernels (none for the plain all-reduce, whose grid is still derived; see select).
struct Op {
  OpType type;
  const char* name;
  std::span<const Template> templates;
  std::span<const TunedKernel> kernels;
};

// =================================================================================================
// EACH OP'S TEMPLATES.
// =================================================================================================

constexpr Template kAllReduceTemplates[] = {
    Template::all_reduce_pull_one_shot,
    Template::all_reduce_pull_two_shot,
};
constexpr Template kRmsNormTemplates[] = {
    Template::all_reduce_pull_one_shot_rms_norm,
    Template::all_reduce_pull_two_shot_rms_norm,
    Template::all_reduce_push_two_shot_rms_norm,
};
constexpr Template kAddRmsNormTemplates[] = {
    Template::all_reduce_pull_one_shot_add_rms_norm,
    Template::all_reduce_pull_two_shot_add_rms_norm,
    Template::all_reduce_push_two_shot_add_rms_norm,
};
constexpr Template kAttnResTemplates[] = {
    Template::all_reduce_pull_one_shot_add_attn_res_rms_norm,
    Template::all_reduce_pull_two_shot_add_attn_res_rms_norm,
    Template::all_reduce_push_two_shot_add_attn_res_rms_norm,
};
constexpr Template kRmsNormGemmAddTemplates[] = {
    Template::all_reduce_pull_one_shot_rms_norm_gemm_add,
    Template::all_reduce_pull_two_shot_rms_norm_gemm_add,
};
constexpr Template kRmsNormGemmTemplates[] = {
    Template::all_reduce_pull_one_shot_rms_norm_gemm,
    Template::all_reduce_pull_two_shot_rms_norm_gemm,
};
constexpr Template kRmsScaleAddTemplates[] = {
    Template::all_reduce_pull_one_shot_rms_scale_add,
    Template::all_reduce_pull_two_shot_rms_scale_add,
};

// =================================================================================================
// EACH OP'S KERNELS.
// =================================================================================================

// gfx950 on n11, bf16, 8 ranks. Transcribed from the size thresholds the sweeps set (each
// crossover cites its run) at the widths we run, until the tuner writes them.
// rms_norm: one-shot through 128 KiB (at 64 KiB it lost at 16 tokens, 11.43 against 10.56 us;
// 2026-09-30T21-06-57Z); the push through 1.75 MiB (256 tokens of 3584: 18.04 against the
// pull's 18.41), the pull from 2.6 MiB (2026-10-01T03-26-44Z).
constexpr TunedKernel kRmsNormKernels[] = {
    {8, 1, 3584, Template::all_reduce_pull_one_shot_rms_norm, RowConfig{{512, 16}, 4096}},
    {8, 19, 3584, Template::all_reduce_push_two_shot_rms_norm, RowConfig{{512, 256}, 4096}},
    {8, 257, 3584, Template::all_reduce_pull_two_shot_rms_norm, RowConfig{{512, 48}, 4096}},
    {8, 1, 7168, Template::all_reduce_pull_one_shot_rms_norm, RowConfig{{512, 16}, 8192}},
    {8, 10, 7168, Template::all_reduce_push_two_shot_rms_norm, RowConfig{{512, 256}, 8192}},
    {8, 129, 7168, Template::all_reduce_pull_two_shot_rms_norm, RowConfig{{512, 48}, 8192}},
};

// add_rms_norm: the one-shot not swept (rms_norm's); the push through 1.31 MiB (192 tokens of
// 3584: 15.48 against the pull's 16.28), the pull at 1.75 MiB (18.36 against the push's 18.46;
// 2026-10-01T03-26-44Z).
constexpr TunedKernel kAddRmsNormKernels[] = {
    {8, 1, 3584, Template::all_reduce_pull_one_shot_add_rms_norm, RowConfig{{512, 16}, 4096}},
    {8, 19, 3584, Template::all_reduce_push_two_shot_add_rms_norm, RowConfig{{512, 256}, 4096}},
    {8, 193, 3584, Template::all_reduce_pull_two_shot_add_rms_norm, RowConfig{{512, 48}, 4096}},
    {8, 1, 7168, Template::all_reduce_pull_one_shot_add_rms_norm, RowConfig{{512, 16}, 8192}},
    {8, 10, 7168, Template::all_reduce_push_two_shot_add_rms_norm, RowConfig{{512, 256}, 8192}},
    {8, 97, 7168, Template::all_reduce_pull_two_shot_add_rms_norm, RowConfig{{512, 48}, 8192}},
};

// AttnRes: the push from 1 token (it beat the one-shot, 12.89 against 13.68 us; 14.03 against
// 15.18 at 8; 2026-10-01T03-57-23Z), through 3.5 MiB against the pull at its grid (256 tokens
// of 7168: 34.8 against 35.8-38.7 us; 2026-10-01T22-56-58Z, 2026-10-01T23-00-47Z).
constexpr TunedKernel kAttnResKernels[] = {
    {8, 1, 3584, Template::all_reduce_push_two_shot_add_attn_res_rms_norm,
     AttnResConfig{{512, 256}, 4096, 1}},
    {8, 513, 3584, Template::all_reduce_pull_two_shot_add_attn_res_rms_norm,
     AttnResPullConfig{{512, 192}, 1, 4096, 1, 32}},
    {8, 1, 7168, Template::all_reduce_push_two_shot_add_attn_res_rms_norm,
     AttnResConfig{{512, 256}, 8192, 1}},
    {8, 257, 7168, Template::all_reduce_pull_two_shot_add_attn_res_rms_norm,
     AttnResPullConfig{{512, 192}, 1, 8192, 1, 32}},
};

// The GEMM tails: the one-shot through one GEMM pass of rows (16), where the one-shot kernel
// once had to stop; not swept.
constexpr TunedKernel kRmsNormGemmKernels[] = {
    {8, 1, 3584, Template::all_reduce_pull_one_shot_rms_norm_gemm,
     GemmConfig{{512, 56}, 16, 4096, kGemmTileK, 4}},
    {8, 17, 3584, Template::all_reduce_pull_two_shot_rms_norm_gemm,
     GemmConfig{{512, 56}, 16, 4096, kGemmTileK, 4}},
    {8, 1, 7168, Template::all_reduce_pull_one_shot_rms_norm_gemm,
     GemmConfig{{512, 56}, 16, 8192, kGemmTileK, 4}},
    {8, 17, 7168, Template::all_reduce_pull_two_shot_rms_norm_gemm,
     GemmConfig{{512, 56}, 16, 8192, kGemmTileK, 4}},
};

// As rms_norm_gemm's.
constexpr TunedKernel kRmsNormGemmAddKernels[] = {
    {8, 1, 3584, Template::all_reduce_pull_one_shot_rms_norm_gemm_add,
     GemmConfig{{512, 56}, 16, 4096, kGemmTileK, 4}},
    {8, 17, 3584, Template::all_reduce_pull_two_shot_rms_norm_gemm_add,
     GemmConfig{{512, 56}, 16, 4096, kGemmTileK, 4}},
    {8, 1, 7168, Template::all_reduce_pull_one_shot_rms_norm_gemm_add,
     GemmConfig{{512, 56}, 16, 8192, kGemmTileK, 4}},
    {8, 17, 7168, Template::all_reduce_pull_two_shot_rms_norm_gemm_add,
     GemmConfig{{512, 56}, 16, 8192, kGemmTileK, 4}},
};

// The one-all-reduce tail, [T, 17920] (a latent of 3584): the one-shot while there are fewer
// rows than ranks, the row two-shot from a row a rank (10.8 against 12.3 us at 1 token, 13.2
// against 14.3 at 8 the other way; 2026-10-01T21-20-57Z).
constexpr TunedKernel kRmsScaleAddKernels[] = {
    {8, 1, 17920, Template::all_reduce_pull_one_shot_rms_scale_add, RowConfig{{512, 256}, 4096}},
    {8, 8, 17920, Template::all_reduce_pull_two_shot_rms_scale_add, RowConfig{{512, 32}, 4096}},
};

// =================================================================================================
// THE OPS, in OpType's order.
// =================================================================================================

// AN OP AND ITS NAME FROM ONE TOKEN, so the name cannot differ from the enum's.
#define HIP_COMMS_NAMED(o) OpType::o, #o

constexpr Op kOps[] = {
    {HIP_COMMS_NAMED(all_reduce), kAllReduceTemplates, {}},
    {HIP_COMMS_NAMED(all_reduce_rms_norm), kRmsNormTemplates, kRmsNormKernels},
    {HIP_COMMS_NAMED(all_reduce_add_rms_norm), kAddRmsNormTemplates, kAddRmsNormKernels},
    {HIP_COMMS_NAMED(all_reduce_add_attn_res_rms_norm), kAttnResTemplates, kAttnResKernels},
    {HIP_COMMS_NAMED(all_reduce_rms_norm_gemm_add), kRmsNormGemmAddTemplates,
     kRmsNormGemmAddKernels},
    {HIP_COMMS_NAMED(all_reduce_rms_norm_gemm), kRmsNormGemmTemplates, kRmsNormGemmKernels},
    {HIP_COMMS_NAMED(all_reduce_rms_scale_add), kRmsScaleAddTemplates, kRmsScaleAddKernels},
};
#undef HIP_COMMS_NAMED
constexpr int kNumOps = sizeof(kOps) / sizeof(Op);

constexpr const Op& op(OpType t) { return kOps[static_cast<int>(t)]; }
constexpr const char* to_string(OpType t) { return op(t).name; }

// THE OP'S TEMPLATE at a shot and direction, how a caller forces one; none where the op has no
// such template.
constexpr std::optional<Template> template_for(OpType o, bool two_shot, bool push) {
  for (const Template t : op(o).templates)
    if (info(t).two_shot == two_shot && info(t).push == push) return t;
  return std::nullopt;
}

// THE OP A TEMPLATE RUNS: the one whose templates name it.
constexpr OpType op_of(Template k) {
  for (const Op& o : kOps)
    for (const Template t : o.templates)
      if (t == k) return o.type;
  return OpType::all_reduce;  // never: ops_own_templates holds every template to one op
}

// EACH CALL'S OP.
constexpr OpType op_of(const AllReduceArgs&) { return OpType::all_reduce; }
constexpr OpType op_of(const NormArgs& a) {
  return a.add ? OpType::all_reduce_add_rms_norm : OpType::all_reduce_rms_norm;
}
constexpr OpType op_of(const AttnResArgs&) { return OpType::all_reduce_add_attn_res_rms_norm; }
constexpr OpType op_of(const GemmTailArgs& a) {
  return a.add ? OpType::all_reduce_rms_norm_gemm_add : OpType::all_reduce_rms_norm_gemm;
}
constexpr OpType op_of(const ScaleAddArgs&) { return OpType::all_reduce_rms_scale_add; }

// A norm then a GEMM, written or added: their GEMM phase strides over column tiles.
constexpr bool gemms(OpType op) {
  return op == OpType::all_reduce_rms_norm_gemm || op == OpType::all_reduce_rms_norm_gemm_add;
}

constexpr bool ops_in_order() {
  for (int i = 0; i < kNumOps; ++i)
    if (static_cast<int>(kOps[i].type) != i) return false;
  return true;
}
static_assert(ops_in_order(), "kOps must list every OpType in its order");

// EVERY TEMPLATE RUNS EXACTLY ONE OP.
constexpr bool ops_own_templates() {
  for (const TemplateInfo& t : kTemplates) {
    int owners = 0;
    for (const Op& o : kOps)
      for (const Template u : o.templates) owners += u == t.fn;
    if (owners != 1) return false;
  }
  return true;
}
static_assert(ops_own_templates(), "a template runs no op, or more than one");

// EVERY OP BUT THE PLAIN ALL-REDUCE HAS KERNELS, each one of its own templates and built.
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

// =================================================================================================
// EACH OP'S ENTRY POINT: the plan's kernel launched, or its Error and nothing launched. plan
// (check.cuh) and launch (launch.cuh) come later in the interface.
// =================================================================================================

template <typename Args>
std::variant<Kernel, Error> plan(const Handle& h, const Args& a, const Options& o);
inline void launch(Handle& h, const Kernel& k, const AllReduceArgs& a, hipStream_t s);
inline void launch(Handle& h, const Kernel& k, const NormArgs& a, hipStream_t s);
inline void launch(Handle& h, const Kernel& k, const AttnResArgs& a, hipStream_t s);
inline void launch(Handle& h, const Kernel& k, const GemmTailArgs& a, hipStream_t s);
inline void launch(Handle& h, const Kernel& k, const ScaleAddArgs& a, hipStream_t s);

namespace impl {

template <typename Args>
std::variant<Kernel, Error> run(Handle& h, const Args& a, const Options& o) {
  std::variant<Kernel, Error> p = plan(h, a, o);
  if (const Kernel* k = std::get_if<Kernel>(&p)) launch(h, *k, a, o.stream);
  return p;
}

}  // namespace impl

inline std::variant<Kernel, Error> all_reduce(Handle& h, const AllReduceArgs& a,
                                                 const Options& o) {
  return impl::run(h, a, o);
}

// rms_norm(all_reduce(inp)), or with a residual fused_add_rms_norm: vLLM's roundings exactly.
inline std::variant<Kernel, Error> all_reduce_rms_norm(Handle& h, const NormArgs& a,
                                                 const Options& o) {
  return impl::run(h, a, o);
}

// Kimi-K3's AttnRes and its RMSNorm on each row of the all-reduced sum (the kernels spell it out).
inline std::variant<Kernel, Error> all_reduce_add_attn_res_rms_norm(Handle& h, const AttnResArgs& a,
                                                 const Options& o) {
  return impl::run(h, a, o);
}

// RMSNorm of the sum, then out = normed @ gemm_weight^T.
inline std::variant<Kernel, Error> all_reduce_rms_norm_gemm(Handle& h, const GemmTailArgs& a,
                                                 const Options& o) {
  return impl::run(h, a, o);
}

// The latent MoE tail: RMSNorm of the sum, then out += normed @ gemm_weight^T.
inline std::variant<Kernel, Error> all_reduce_rms_norm_gemm_add(Handle& h, const GemmTailArgs& a,
                                                 const Options& o) {
  return impl::run(h, a, o);
}

// Kimi-K3's latent MoE tail with one all-reduce: out = shared + projected * 1/rms(latent), all
// three summed over the ranks.
inline std::variant<Kernel, Error> all_reduce_rms_scale_add(Handle& h, const ScaleAddArgs& a,
                                                 const Options& o) {
  return impl::run(h, a, o);
}

}  // namespace hip_comms
