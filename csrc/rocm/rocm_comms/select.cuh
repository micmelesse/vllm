// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SELECT, THE ONLY CHOICE, read by each op's select_<op> (interface.cuh): what was tuned (each
// op's kernels: for each world and row width, the template and KernelConfig that won from a
// number of rows up, written by the tuner, tune.py, from a sweep; an entry says what ran fastest,
// the why is the sweep's figure, cited beside it), then the steps every select takes, each on
// primitives: the forcing made a template and config, or the tuned kernel picked; the config
// fitted to the call (a zero field the template's own, the tile widened to cover the row, the
// grid cut to the tiles there are); and the first Error the kernel meets here.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "kernels/add_attn_res_rms_norm.cuh"
#include "kernels/all_reduce_pull_one_shot.cuh"
#include "kernels/all_reduce_pull_one_shot_add_attn_res_rms_norm.cuh"
#include "kernels/all_reduce_pull_one_shot_add_rms_norm.cuh"
#include "kernels/all_reduce_pull_one_shot_rms_norm_gemm_add.cuh"
#include "kernels/all_reduce_pull_one_shot_rms_scale_add.cuh"
#include "kernels/all_reduce_pull_two_shot.cuh"
#include "kernels/all_reduce_pull_two_shot_add_attn_res_rms_norm.cuh"
#include "kernels/all_reduce_pull_two_shot_add_rms_norm.cuh"
#include "kernels/all_reduce_pull_two_shot_rms_norm_gemm_add.cuh"
#include "kernels/all_reduce_pull_two_shot_rms_scale_add.cuh"
#include "kernels/all_reduce_push_two_shot_add_attn_res_rms_norm.cuh"
#include "kernels/all_reduce_push_two_shot_add_rms_norm.cuh"
#include "kernels/probe_barrier.cuh"
#include "kernels/probe_link_traffic.cuh"
#include "kernels/probe_ping_pong.cuh"

namespace hip_comms {

// =================================================================================================
// THE CATALOG: each template, and the KernelConfigs it is built for.
// =================================================================================================

// A TEMPLATE'S CONFIGS, each of its family (types.cuh), in the order select prefers them (its
// default for a call is the first whose tile covers the call's row). The fields compiled in (the
// tile, the threads) are what dispatch instantiates, what check accepts and what a tuner
// searches; the grid (cut to the tiles there are) and reduce_scatter_blocks are each config's
// launch, and a caller may force others. tile_n counts the elements of a 16-bit dtype, the only
// ones built. A config that spills shows in its code object (Handle::resources_of) and loses on
// the clock; none is ruled out by policy. Each grid cites the sweep that set it, at 512 threads
// (256 was worse for the GEMM tail, 2026-09-28; not swept for the others, which copy it).

// The norms: one-shot, the push two-shot (a column split), the pull two-shot (a row split).
// Not swept: select cuts it to the rows, so it matters only past 16 rows.
// The plain one-shot: a chunk THREADS_PER_BLOCK groups wide, at each block size a launch may take
// (64, a wave, its own: all_reduce_config).
constexpr KernelConfig kAllReduceOneShotConfigs[] = {
    AllReduceConfig{{64, 0, 1}},
    AllReduceConfig{{128, 0, 1}},
    AllReduceConfig{{256, 0, 1}},
    AllReduceConfig{{512, 0, 1}},
};
// The plain two-shot: the same chunks (512, a wave per peer at eight ranks, its own).
constexpr KernelConfig kAllReduceTwoShotConfigs[] = {
    AllReduceConfig{{64, 0, 1}},
    AllReduceConfig{{128, 0, 1}},
    AllReduceConfig{{256, 0, 1}},
    AllReduceConfig{{512, 0, 1}},
};
constexpr KernelConfig kRmsNormOneShotConfigs[] = {
    RowConfig{{512, 16, 1}, 4096},
    RowConfig{{512, 16, 1}, 8192},
    RowConfig{{512, 16, 1}, 16384},
};
// 256 the best of 48-256 at 192-256 tokens (15.17 and 18.04 against 15.55 and 18.11 at 128;
// 2026-10-01T03-26-44Z); a row a block below that.
constexpr KernelConfig kRmsNormPushConfigs[] = {
    RowConfig{{512, 256, 1}, 4096},
    RowConfig{{512, 256, 1}, 8192},
    RowConfig{{512, 256, 1}, 16384},
};
// Pipelined, 48 the best of 36-96 at 2048-4096 tokens (79.3 and 145.4 against 81.0 and 147.6 at
// 36), within 0.8 of 36 below (2026-10-01T02-59-52Z).
constexpr KernelConfig kRmsNormPullConfigs[] = {
    RowConfig{{512, 48, 1}, 4096},
    RowConfig{{512, 48, 1}, 8192},
};
// Not swept: rms_norm's.
constexpr KernelConfig kAddRmsNormOneShotConfigs[] = {
    RowConfig{{512, 16, 1}, 4096},
    RowConfig{{512, 16, 1}, 8192},
    RowConfig{{512, 16, 1}, 16384},
};
// 256 the best of 48-256 at 192 tokens (15.48 against 16.75 at 128; 2026-10-01T03-26-44Z).
constexpr KernelConfig kAddRmsNormPushConfigs[] = {
    RowConfig{{512, 256, 1}, 4096},
    RowConfig{{512, 256, 1}, 8192},
    RowConfig{{512, 256, 1}, 16384},
};
// 48 the best of 36-96 at every size from 512 to 4096 tokens (80.6 and 149.7 us at 2048 and 4096
// against 84.3 and 154.7 at 36; 2026-10-01T02-59-52Z).
constexpr KernelConfig kAddRmsNormPullConfigs[] = {
    RowConfig{{512, 48, 1}, 4096},
    RowConfig{{512, 48, 1}, 8192},
};

// AttnRes, one source a step (TILE_K): at 4 (Triton's tile) the row got slower, 6.48 -> 8.16 us a
// row and 147.5 -> 171.0 at 4096 tokens, with 100 -> 166 VGPRs (stamps 2026-10-01T06-26-55Z
// against 04-15-48Z). The one-shot's grid not swept: the norms'.
constexpr KernelConfig kAttnResOneShotConfigs[] = {
    AttnResConfig{{512, 16, 1}, 4096, 1},
    AttnResConfig{{512, 16, 1}, 8192, 1},
};
// 256 the best of 32-256 at 256-1024 tokens (34.58, 70.78, 143.98 against 39.93, 83.15, 159.33 at
// 128), a row a block below that (2026-10-01T03-57-23Z).
// 256 threads a block: Triton's program for AttnRes, and more blocks resident at once (AttnRes
// from local memory took 192.8 / 109.6 / 86.0 us on 192 / 384 / 512 blocks, stamps
// 2026-10-02T20-36-56Z); its grid not swept yet.
constexpr KernelConfig kAttnResPushConfigs[] = {
    AttnResConfig{{512, 256, 1}, 4096, 1},
    AttnResConfig{{512, 256, 1}, 8192, 1},
    AttnResConfig{{256, 512, 1}, 4096, 1},
    AttnResConfig{{256, 512, 1}, 8192, 1},
};
// The column split wants a wide grid (AttnRes is compute a row): at 7168, 192 is within about 5% of
// the best of 16-256 from 512 to 4096 tokens; 4096 at 447.2 us against 1160.7 at the 36 it had
// (2026-10-01T22-56-58Z, 2026-10-01T23-00-47Z). TILE_M 2 and 4 lost at every grid, best 480.0 us at
// 128 blocks against 412.8 at 1 on 192 (4096 x 7168, 2026-10-02T21-19-22Z). Its
// reduce-scatter on the grid's first 32 blocks (reduce_scatter_blocks): its reads queue behind the
// links past a few dozen blocks while AttnRes is compute a row and wants every block. At 4096
// tokens it took 146.6 us on 32 blocks against 218.9 on 192, AttnRes 1110.5 against 199.4 (stamps,
// 2026-10-01T23-45-31Z and 2026-10-01T23-50-54Z).
// 256 threads as the push's, its grid doubled (not swept yet). TILE_M 2 and 4 are no longer
// built: they lost at every grid, and 4 spilled 25-30 VGPRs.
constexpr KernelConfig kAttnResPullConfigs[] = {
    AttnResPullConfig{{512, 192, 1}, 1, 4096, 1, 32},
    AttnResPullConfig{{512, 192, 1}, 1, 8192, 1, 32},
    AttnResPullConfig{{256, 384, 1}, 1, 4096, 1, 32},
    AttnResPullConfig{{256, 384, 1}, 1, 8192, 1, 32},
};

// The GEMM tails, both ops and both shots: 16 rows a GEMM pass (one fp32 accumulator a row in each
// lane); TILE_K what gfx950's LDS stages at 16 rows beside the widest block's partials; SLICE_K 4,
// picked at Kimi-K3's shape, 1, 2 and 8 worse at 1 row (2026-09-28); 56 blocks the best at 1 row
// (2026-09-28, log). Not swept for the rest, which copy it.
constexpr int kGemmTileK = gemm_tile_k_fit(kTarget, 16);
constexpr KernelConfig kGemmTailConfigs[] = {
    GemmConfig{{512, 56, 1}, 16, 4096, kGemmTileK, 4},
    GemmConfig{{512, 56, 1}, 16, 8192, kGemmTileK, 4},
    GemmConfig{{512, 56, 1}, 16, 16384, kGemmTileK, 4},
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
    RowConfig{{512, 256, 1}, 4096},
    RowConfig{{512, 256, 1}, 8192},
};
// About 32 blocks keeps the links fed; more queue behind them: at [T, 17920] bf16, within 1% of the
// best of 8-48 blocks from 8 to 4096 tokens, and 1024 tokens 146.9 us at 32 against 258.4 at 256
// (2026-10-01T22-07-13Z).
constexpr KernelConfig kRmsScaleAddTwoShotConfigs[] = {
    RowConfig{{512, 32, 1}, 4096},
    RowConfig{{512, 32, 1}, 8192},
};

// EXPERIMENTAL, AttnRes on a local delta: a row a block, the grid striding over rows. 512 blocks
// took the AttnRes phase from local scratch to 86.0 us at 4096 x 7168 (192: 192.8; stamps
// 2026-10-02T20-36-56Z); not swept.
constexpr KernelConfig kAddAttnResConfigs[] = {
    AttnResConfig{{512, 512, 1}, 4096, 1},
    AttnResConfig{{512, 512, 1}, 8192, 1},
    AttnResConfig{{256, 512, 1}, 4096, 1},
    AttnResConfig{{256, 512, 1}, 8192, 1},
};

// What each template is, in Template's order: its op, its shot, and its builds (none for the plain
// all-reduce, which has no tile).
struct TemplateInfo {
  Template fn;
  const char* name;
  OpType op;
  bool two_shot;
  bool push;  // peers push into this rank (else it pulls from them)
  std::span<const KernelConfig> configs;
};

// A TEMPLATE AND ITS NAME FROM ONE TOKEN, so the name cannot differ from the enum's.
#define HIP_COMMS_NAMED(t) Template::t, #t

constexpr TemplateInfo kTemplates[] = {
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot), OpType::all_reduce,
     false, false, kAllReduceOneShotConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot), OpType::all_reduce,
     true, false, kAllReduceTwoShotConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_rms_norm), OpType::all_reduce_rms_norm,
     false, false, kRmsNormOneShotConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_rms_norm), OpType::all_reduce_rms_norm,
     true, false, kRmsNormPullConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_add_rms_norm), OpType::all_reduce_add_rms_norm,
     false, false, kAddRmsNormOneShotConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_add_rms_norm), OpType::all_reduce_add_rms_norm,
     true, false, kAddRmsNormPullConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_add_attn_res_rms_norm),
     OpType::all_reduce_add_attn_res_rms_norm, false, false, kAttnResOneShotConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_add_attn_res_rms_norm),
     OpType::all_reduce_add_attn_res_rms_norm, true, false, kAttnResPullConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_rms_norm_gemm_add),
     OpType::all_reduce_rms_norm_gemm_add, false, false, kGemmTailConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_rms_norm_gemm_add),
     OpType::all_reduce_rms_norm_gemm_add, true, false, kGemmTailConfigs},
    {HIP_COMMS_NAMED(all_reduce_push_two_shot_rms_norm), OpType::all_reduce_rms_norm,
     true, true, kRmsNormPushConfigs},
    {HIP_COMMS_NAMED(all_reduce_push_two_shot_add_rms_norm), OpType::all_reduce_add_rms_norm,
     true, true, kAddRmsNormPushConfigs},
    {HIP_COMMS_NAMED(all_reduce_push_two_shot_add_attn_res_rms_norm),
     OpType::all_reduce_add_attn_res_rms_norm, true, true, kAttnResPushConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_rms_norm_gemm), OpType::all_reduce_rms_norm_gemm,
     false, false, kGemmTailConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_rms_norm_gemm), OpType::all_reduce_rms_norm_gemm,
     true, false, kGemmTailConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_one_shot_rms_scale_add), OpType::all_reduce_rms_scale_add,
     false, false, kRmsScaleAddOneShotConfigs},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot_rms_scale_add), OpType::all_reduce_rms_scale_add,
     true, false, kRmsScaleAddTwoShotConfigs},
    {HIP_COMMS_NAMED(add_attn_res_rms_norm), OpType::add_attn_res_rms_norm,
     false, false, kAddAttnResConfigs},
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
// Whether a template is built at listed configs (its threads compiled in), and whether those have a
// tile's fields too (the plain all-reduce's are threads alone).
constexpr bool has_builds(Template k) { return !configs_of(k).empty(); }
constexpr bool has_tiles(Template k) {
  return has_builds(k) && !std::holds_alternative<AllReduceConfig>(configs_of(k)[0]);
}

// Whether two configs are one build: the same family, the same fields compiled in, at the same
// threads (the grid and reduce_scatter_blocks are a launch's).
constexpr bool same_build(const KernelConfig& a, const KernelConfig& b) {
  if (a.index() != b.index()) return false;
  return std::visit(
      [&](const auto& x) {
        const auto& y = std::get<std::decay_t<decltype(x)>>(b);
        bool same     = x.launch.threads_per_block == y.launch.threads_per_block &&
                    x.launch.waves_per_eu == y.launch.waves_per_eu;
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
constexpr bool built_at(Template k, int threads_per_block, int waves_per_eu) {
  for (const KernelConfig& b : configs_of(k))
    if (launch_of(b).threads_per_block == threads_per_block &&
        launch_of(b).waves_per_eu == waves_per_eu)
      return true;
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


constexpr OpType op_of(Template k) { return info(k).op; }
constexpr Algorithm algorithm_of(Template k) {
  return info(k).two_shot ? Algorithm::two_shot : Algorithm::one_shot;
}
constexpr Direction direction_of(Template k) {
  return info(k).push ? Direction::push : Direction::pull;
}
// OP `o`'S TEMPLATE at an algorithm and direction, or none: an op has at most one of each.
constexpr std::optional<Template> template_of(OpType o, Algorithm a, Direction d) {
  for (const TemplateInfo& t : kTemplates)
    if (t.op == o && algorithm_of(t.fn) == a && direction_of(t.fn) == d) return t.fn;
  return std::nullopt;
}
constexpr bool one_template_a_shot() {
  for (const TemplateInfo& t : kTemplates)
    if (template_of(t.op, algorithm_of(t.fn), direction_of(t.fn)) != t.fn) return false;
  return true;
}
static_assert(one_template_a_shot(), "an op has two templates at one algorithm and direction");

// =================================================================================================
// OVER WHAT IS COMPILED, at compile time: the template arguments are constexpr, so each value (the
// world, a dtype, a built config) is turned into its template parameter here; one that was not
// built raises.
// =================================================================================================

template <typename WRAPPED>
struct type {
  using t = WRAPPED;
};

template <int INT_VALUE>
using constant = std::integral_constant<int, INT_VALUE>;

[[noreturn]] inline void not_built(const std::string& what) {
  throw std::runtime_error("hip_comms: " + what + " not built");
}

// THE C++ TYPE OF A BUILT DTYPE.
template <DType DTYPE_ID>
struct of_dtype;
template <>
struct of_dtype<DType::f16> {
  using t = c10::Half;
};
template <>
struct of_dtype<DType::bf16> {
  using t = c10::BFloat16;
};

// OVER THE BUILT LISTS (kBuild.supports), so what is compiled is what they say.
template <typename VISITOR, size_t... INDICES>
void by_world_in(int world, VISITOR& f, std::index_sequence<INDICES...>) {
  constexpr auto& built = kBuild.supports.worlds;
  if (!((world == built[INDICES] && (f(constant<built[INDICES]>{}), true)) || ...))
    not_built("world size " + std::to_string(world));
}

template <typename VISITOR>
void by_world(int world, VISITOR&& f) {
  by_world_in(world, f, std::make_index_sequence<kBuild.supports.worlds.size()>{});
}

template <typename VISITOR, size_t... INDICES>
void by_dtype_in(DType d, VISITOR& f, std::index_sequence<INDICES...>) {
  constexpr auto& built = kBuild.supports.dtypes;
  if (!((d == built[INDICES] && (f(type<typename of_dtype<built[INDICES]>::t>{}), true)) || ...))
    not_built("dtype");
}

template <typename VISITOR>
void by_dtype(DType d, VISITOR&& f) {
  by_dtype_in(d, f, std::make_index_sequence<kBuild.supports.dtypes.size()>{});
}

// A norm's weight: T itself, or fp32.
template <typename DTYPE, typename VISITOR>
void by_weight(DType weight, DType dtype, VISITOR&& f) {
  if (weight == dtype) return f(type<DTYPE>{});
  if (weight == DType::f32) return f(type<float>{});
  not_built("weight dtype");
}

// A TEMPLATE'S BUILD, compiled in: the one of template K's configs (the catalog above) `c` names
// (all but its launch's grid and reduce_scatter_blocks, which are run time), and only those. f is
// handed it as config_constant<C>, C its family's config, every field a constant expression.
template <Template TEMPLATE>
using family_t = std::variant_alternative_t<family_of(TEMPLATE), KernelConfig>;
template <Template TEMPLATE, size_t CONFIG_INDEX>
constexpr family_t<TEMPLATE> built_config = std::get<family_t<TEMPLATE>>(configs_of(TEMPLATE)[CONFIG_INDEX]);
template <auto BUILT_CONFIG>
struct config_constant {
  static constexpr auto value = BUILT_CONFIG;
};

template <Template TEMPLATE, typename VISITOR, size_t... INDICES>
void by_config_in(const KernelConfig& c, VISITOR& f, std::index_sequence<INDICES...>) {
  if (!((same_build(c, configs_of(TEMPLATE)[INDICES]) && (f(config_constant<built_config<TEMPLATE, INDICES>>{}), true)) ||
        ...))
    not_built(std::string("template ") + to_string(TEMPLATE) + " at that tile and threads");
}

template <Template TEMPLATE, typename VISITOR>
void by_config(const KernelConfig& c, VISITOR&& f) {
  by_config_in<TEMPLATE>(c, f, std::make_index_sequence<configs_of(TEMPLATE).size()>{});
}

[[noreturn]] inline void not_this_ops(Template k) {
  throw std::runtime_error("hip_comms: template " + std::to_string(static_cast<int>(k)) +
                           " is not this op's");
}



// =================================================================================================
// THE COMPILED INSTANCE select picks: from the template and config it chose, the call's dtypes and
// the world, one compiled kernel, checked here to have its op's signature (types.cuh) with its
// element pointers erased, which is what a launch calls it through.
// =================================================================================================

template <typename PARAM>
struct erase {
  using t = PARAM;
};
template <typename PARAM>
struct erase<PARAM*> {
  using t = void*;
};
template <typename PARAM>
struct erase<const PARAM*> {
  using t = const void*;
};
template <typename KERNEL_PTR>
struct erased;
template <typename... PARAMS>
struct erased<void (*)(PARAMS...)> {
  using t = void (*)(typename erase<PARAMS>::t...);
};

// `kernel`, as the launch holds it, once it is checked to be callable as `Signature`.
template <typename SIGNATURE, typename... PARAMS>
const void* instance(void (*kernel)(PARAMS...)) {
  static_assert(std::is_same_v<typename erased<void (*)(PARAMS...)>::t, SIGNATURE>,
                "a compiled kernel does not have its op's signature");
  return reinterpret_cast<const void*>(kernel);
}

// WHETHER `kernel` HOLDS A GRID OF `blocks` BLOCKS OF `threads` RESIDENT: only the compiled kernel
// knows what it uses.
inline bool resident(const Handle& h, const void* kernel, int blocks, int threads) {
  return blocks <= resident_blocks(kTarget, h.resources_of(kernel), threads);
}

// From `rows` rows up (to the next entry's), at `world` ranks and rows `hidden` elements wide:
// template `fn` at `config`.
struct TunedKernel {
  int world;
  int64_t rows;
  int64_t hidden;
  Template fn;
  KernelConfig config;
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
    {8, 1, 3584, Template::all_reduce_pull_one_shot_rms_norm, RowConfig{{512, 16, 1}, 4096}},
    {8, 19, 3584, Template::all_reduce_push_two_shot_rms_norm, RowConfig{{512, 256, 1}, 4096}},
    {8, 257, 3584, Template::all_reduce_pull_two_shot_rms_norm, RowConfig{{512, 48, 1}, 4096}},
    {8, 1, 7168, Template::all_reduce_pull_one_shot_rms_norm, RowConfig{{512, 16, 1}, 8192}},
    {8, 10, 7168, Template::all_reduce_push_two_shot_rms_norm, RowConfig{{512, 256, 1}, 8192}},
    {8, 129, 7168, Template::all_reduce_pull_two_shot_rms_norm, RowConfig{{512, 48, 1}, 8192}},
};

// add_rms_norm: the one-shot not swept (rms_norm's); the push through 1.31 MiB (192 tokens of
// 3584: 15.48 against the pull's 16.28), the pull at 1.75 MiB (18.36 against the push's 18.46;
// 2026-10-01T03-26-44Z).
constexpr TunedKernel kAddRmsNormKernels[] = {
    {8, 1, 3584, Template::all_reduce_pull_one_shot_add_rms_norm, RowConfig{{512, 16, 1}, 4096}},
    {8, 19, 3584, Template::all_reduce_push_two_shot_add_rms_norm, RowConfig{{512, 256, 1}, 4096}},
    {8, 193, 3584, Template::all_reduce_pull_two_shot_add_rms_norm, RowConfig{{512, 48, 1}, 4096}},
    {8, 1, 7168, Template::all_reduce_pull_one_shot_add_rms_norm, RowConfig{{512, 16, 1}, 8192}},
    {8, 10, 7168, Template::all_reduce_push_two_shot_add_rms_norm, RowConfig{{512, 256, 1}, 8192}},
    {8, 97, 7168, Template::all_reduce_pull_two_shot_add_rms_norm, RowConfig{{512, 48, 1}, 8192}},
};

// AttnRes: the push from 1 token (it beat the one-shot, 12.89 against 13.68 us; 14.03 against
// 15.18 at 8; 2026-10-01T03-57-23Z), through 3.5 MiB against the pull at its grid (256 tokens
// of 7168: 34.8 against 35.8-38.7 us; 2026-10-01T22-56-58Z, 2026-10-01T23-00-47Z).
constexpr TunedKernel kAttnResKernels[] = {
    {8, 1, 3584, Template::all_reduce_push_two_shot_add_attn_res_rms_norm,
     AttnResConfig{{512, 256, 1}, 4096, 1}},
    {8, 513, 3584, Template::all_reduce_pull_two_shot_add_attn_res_rms_norm,
     AttnResPullConfig{{512, 192, 1}, 1, 4096, 1, 32}},
    {8, 1, 7168, Template::all_reduce_push_two_shot_add_attn_res_rms_norm,
     AttnResConfig{{512, 256, 1}, 8192, 1}},
    {8, 257, 7168, Template::all_reduce_pull_two_shot_add_attn_res_rms_norm,
     AttnResPullConfig{{512, 192, 1}, 1, 8192, 1, 32}},
};

// The GEMM tails: the one-shot through one GEMM pass of rows (16), where the one-shot kernel
// once had to stop; not swept.
constexpr TunedKernel kRmsNormGemmKernels[] = {
    {8, 1, 3584, Template::all_reduce_pull_one_shot_rms_norm_gemm,
     GemmConfig{{512, 56, 1}, 16, 4096, kGemmTileK, 4}},
    {8, 17, 3584, Template::all_reduce_pull_two_shot_rms_norm_gemm,
     GemmConfig{{512, 56, 1}, 16, 4096, kGemmTileK, 4}},
    {8, 1, 7168, Template::all_reduce_pull_one_shot_rms_norm_gemm,
     GemmConfig{{512, 56, 1}, 16, 8192, kGemmTileK, 4}},
    {8, 17, 7168, Template::all_reduce_pull_two_shot_rms_norm_gemm,
     GemmConfig{{512, 56, 1}, 16, 8192, kGemmTileK, 4}},
};

// As rms_norm_gemm's.
constexpr TunedKernel kRmsNormGemmAddKernels[] = {
    {8, 1, 3584, Template::all_reduce_pull_one_shot_rms_norm_gemm_add,
     GemmConfig{{512, 56, 1}, 16, 4096, kGemmTileK, 4}},
    {8, 17, 3584, Template::all_reduce_pull_two_shot_rms_norm_gemm_add,
     GemmConfig{{512, 56, 1}, 16, 4096, kGemmTileK, 4}},
    {8, 1, 7168, Template::all_reduce_pull_one_shot_rms_norm_gemm_add,
     GemmConfig{{512, 56, 1}, 16, 8192, kGemmTileK, 4}},
    {8, 17, 7168, Template::all_reduce_pull_two_shot_rms_norm_gemm_add,
     GemmConfig{{512, 56, 1}, 16, 8192, kGemmTileK, 4}},
};

// The one-all-reduce tail, [T, 17920] (a latent of 3584): the one-shot while there are fewer
// rows than ranks, the row two-shot from a row a rank (10.8 against 12.3 us at 1 token, 13.2
// against 14.3 at 8 the other way; 2026-10-01T21-20-57Z).
constexpr TunedKernel kRmsScaleAddKernels[] = {
    {8, 1, 17920, Template::all_reduce_pull_one_shot_rms_scale_add, RowConfig{{512, 256, 1}, 4096}},
    {8, 8, 17920, Template::all_reduce_pull_two_shot_rms_scale_add, RowConfig{{512, 32, 1}, 4096}},
};

// Experimental, one rank (world 1: there are no peers): not swept.
constexpr TunedKernel kAddAttnResKernels[] = {
    {1, 1, 3584, Template::add_attn_res_rms_norm, AttnResConfig{{512, 512, 1}, 4096, 1}},
    {1, 1, 7168, Template::add_attn_res_rms_norm, AttnResConfig{{512, 512, 1}, 8192, 1}},
};

// =================================================================================================
// THE OPS, in OpType's order: each one's name (as Python calls it) and its tuned kernels (none for
// the plain all-reduce, whose grid is still derived).
// =================================================================================================

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
    {HIP_COMMS_NAMED(add_attn_res_rms_norm), kAddAttnResKernels},
};
#undef HIP_COMMS_NAMED
constexpr int kNumOps = sizeof(kOps) / sizeof(Op);

constexpr const Op& op(OpType t) { return kOps[static_cast<int>(t)]; }
constexpr const char* to_string(OpType t) { return op(t).name; }

constexpr bool ops_in_order() {
  for (int i = 0; i < kNumOps; ++i)
    if (static_cast<int>(kOps[i].type) != i) return false;
  return true;
}
static_assert(ops_in_order(), "kOps must list every OpType in its order");

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

// THE OP'S TEMPLATES, in the catalog's order, and how many.
constexpr int templates_of(OpType o) {
  int n = 0;
  for (const TemplateInfo& t : kTemplates) n += t.op == o;
  return n;
}
constexpr Template only_template(OpType o) {
  for (const TemplateInfo& t : kTemplates)
    if (t.op == o) return t.fn;
  return Template::all_reduce_pull_one_shot;  // never: every op has a template
}

// A norm then a GEMM, written or added: their GEMM phase strides over column tiles.
constexpr bool gemms(OpType op) {
  return op == OpType::all_reduce_rms_norm_gemm || op == OpType::all_reduce_rms_norm_gemm_add;
}

// A COLUMN TWO-SHOT reduce-scatters by columns, so every block keeps every row; a row two-shot
// gives each rank its rows.
constexpr bool slices_columns(Template k) {
  return k == Template::all_reduce_pull_two_shot_add_attn_res_rms_norm ||
         k == Template::all_reduce_push_two_shot_rms_norm ||
         k == Template::all_reduce_push_two_shot_add_rms_norm ||
         k == Template::all_reduce_push_two_shot_add_attn_res_rms_norm;
}

// =================================================================================================
// 1. THE FORCING: the template the algorithm and direction name (an op with one template needs
// neither to force a config), and its config from the launch and the op's fields (`config` builds
// its family's from them, a zero field the template's own); a mistake in it is its Error. None:
// select picks.
// =================================================================================================

using Forced = std::optional<std::pair<Template, std::optional<KernelConfig>>>;

template <typename CONFIG_TYPE>
constexpr std::variant<Forced, Error> forced(OpType o, std::optional<Algorithm> algorithm,
                                             std::optional<Direction> direction,
                                             std::optional<int> threads_per_block,
                                             std::optional<int> blocks_per_grid,
                                             std::initializer_list<std::optional<int>> fields,
                                             CONFIG_TYPE config) {
  if (direction && !algorithm) return Error::direction_without_algorithm;
  if (threads_per_block.has_value() != blocks_per_grid.has_value())
    return Error::launch_incomplete;
  const bool launched = blocks_per_grid.has_value();
  if (launched && (*blocks_per_grid < 1 || *blocks_per_grid > kMaxBlocks ||
                   *threads_per_block < kWaveSize ||
                   *threads_per_block > kBuild.kernels.max_threads ||
                   *threads_per_block % kWaveSize != 0))
    return Error::launch_out_of_range;
  for (const std::optional<int>& f : fields) {
    if (f && !launched) return Error::field_without_launch;
    if (f && *f < 1) return Error::field_not_positive;
  }
  std::optional<Template> fn;
  if (algorithm) {
    fn = template_of(o, *algorithm, direction.value_or(Direction::pull));
    if (!fn) return Error::no_such_template;
  }
  if (!launched) return fn ? Forced{{*fn, std::nullopt}} : Forced{};
  if (!fn && templates_of(o) == 1) fn = only_template(o);
  if (!fn) return Error::config_without_algorithm;
  const std::variant<KernelConfig, Error> c =
      config(LaunchConfig{*threads_per_block, *blocks_per_grid, 0}, *fn);
  if (const Error* e = std::get_if<Error>(&c)) return *e;
  return Forced{{*fn, std::get<KernelConfig>(c)}};
}

// A FORCED FIELD's value, or 0: the template's own.
constexpr int own(std::optional<int> v) { return v.value_or(0); }

// =================================================================================================
// 2. THE PICK.
// =================================================================================================

// A PLAIN ALL-REDUCE: pull one-shot up to 64 KiB, pull two-shot past it, the grid the size of the
// work up to the links' bandwidth-delay product. The one op not yet tuned into a kernel list:
// it has no tile, and its grid follows the bytes (the tuner will replace it; PLAN 5.3.12.4.5.2).
//
// ONE WAVE PER BLOCK. Waves in one block only add a barrier inside it: at 1 token one-shot took
// 7.8 us at 64 threads, 8.1 at 128, 9.0 at 256 and 10.9 at 512 (bench, 2026-09-29T19-24-48Z).
// One-shot reads every peer's whole buffer ((N-1)P) in one round trip; two-shot moves less
// (2(N-1)/N P) in two: one-shot won at 56 KiB (7.12 vs 7.83 us), two-shot at 112 KiB (7.87 vs
// 8.19), uncached scratch (2026-09-30T18-00-30Z). TWO-SHOT'S BLOCK IS A ROW OF THREADS A RANK
// (aiter's two-stage, a wave a peer). A GRID THE SIZE OF THE WORK, as aiter sizes its own: every
// block pays for every sync
// (at 16 tokens two-shot ran 11.00 us on 16 blocks, 12.31 on 64), capped at what keeps the links
// busy: their bandwidth-delay product, 7 x 76.8 GB/s x 1334 ns = 717 KB, 88 blocks of 512 threads
// (the sweep: flat from 80 to 128). Every pass full: 3.7 MB at 88 blocks took 5.09 passes, 23.78
// us, against 90 blocks in 5 full ones.
// `loads`: the packs a block has in flight a pass.
constexpr int link_filling_blocks(const Hardware& hw, const Calibration& cal, int loads) {
  const double in_flight = hw.xgmi_links * hw.xgmi_gbytes_per_s_a_way * cal.ping_pong_ns;
  const double per_pass  = static_cast<double>(loads) * kBuild.memory.pack_bytes;
  const int blocks       = static_cast<int>(in_flight / per_pass + 0.999);
  return blocks < hw.compute_units ? blocks : hw.compute_units;
}
constexpr KernelConfig all_reduce_config(Template t, int64_t bytes, int world) {
  const Hardware& hw    = kTarget;
  const bool one_shot   = t == Template::all_reduce_pull_one_shot;
  const int64_t packs   = (bytes + kBuild.memory.pack_bytes - 1) / kBuild.memory.pack_bytes;
  const int64_t work    = one_shot ? packs : (packs + world - 1) / world;
  const int64_t need    = (work + hw.wave_size - 1) / hw.wave_size;
  // THE TWO-SHOT'S BLOCK IS A ROW OF THREADS A RANK (a wave each at eight), a thread one load: a
  // tiled two-shot whose thread read every rank put 8x the links' bytes in flight on this grid and
  // lost 0.8 us at 16-64 tokens however launched (2026-10-04T01-07-11Z).
  const int threads     = one_shot ? hw.wave_size : hw.wave_size * world;
  const int fill        = link_filling_blocks(hw, kTargetCalibration, threads);
  const int64_t passes  = need > fill ? need / fill : 1;
  const int64_t even    = (need + passes - 1) / passes;
  // NOT std::min: hipify turns it into HIP's device `min`, which is not constexpr.
  const int blocks = static_cast<int>(even < hw.compute_units ? even : hw.compute_units);
  return AllReduceConfig{{threads, blocks, 1}};
}
constexpr int64_t distance(int64_t a, int64_t b) { return a < b ? b - a : a - b; }

// A TILED OP'S TUNED KERNEL for `rows` rows `hidden` wide at `world` ranks. Among the op's kernels,
// those at the call's world, else at the nearest tuned world; among those, the call's width, else
// the nearest tuned width, reading the call's rows as that width's rows of the same bytes (a
// crossover is about bytes); then the last entry whose rows the call reaches, the first when it
// reaches none. At a width the entry's tile does not cover (`tile_cols`), its tile_n is left to
// fitting (the smallest built that covers).
constexpr TunedKernel pick(OpType o, int world, int64_t rows, int64_t hidden, int64_t tile_cols) {
  const std::span<const TunedKernel> kernels = op(o).kernels;
  int w = 0;
  for (const TunedKernel& k : kernels)
    if (w == 0 || distance(k.world, world) < distance(w, world)) w = k.world;
  int64_t h = 0;
  for (const TunedKernel& k : kernels)
    if (k.world == w && (h == 0 || distance(k.hidden, hidden) < distance(h, hidden)))
      h = k.hidden;
  const int64_t as_rows    = rows * hidden / h;
  const TunedKernel* first = nullptr;
  const TunedKernel* found = nullptr;
  for (const TunedKernel& k : kernels) {
    if (k.world != w || k.hidden != h) continue;
    if (!first || k.rows < first->rows) first = &k;
    if (k.rows <= as_rows && (!found || k.rows > found->rows)) found = &k;
  }
  TunedKernel got = found ? *found : *first;
  if (tile_n_of(got.config) < tile_cols) set_tile_n(got.config, 0);
  return got;
}

// =================================================================================================
// 3. THE CONFIG FITTED TO THE CALL.
// =================================================================================================

// THE TEMPLATE'S OWN CONFIG for a tile of `tile_cols` columns: its first listed config whose tile
// covers them (its first when none does).
constexpr KernelConfig default_config(Template t, int64_t tile_cols) {
  for (const KernelConfig& c : configs_of(t))
    if (tile_n_of(c) >= tile_cols) return c;
  return configs_of(t)[0];
}

// THE SMALLEST BUILT TILE_N covering `cols` at `c`'s other fields, or 0 when none does (the check
// refuses it).
constexpr int smallest_tile_n(Template t, const KernelConfig& c, int64_t cols) {
  int n = 0;
  for (const KernelConfig& b : configs_of(t)) {
    KernelConfig same = c;
    set_tile_n(same, tile_n_of(b));
    if (same_build(same, b) && tile_n_of(b) >= cols && (n == 0 || tile_n_of(b) < n))
      n = tile_n_of(b);
  }
  return n;
}

// THE TILES THERE ARE, one block's work each: this rank's rows (a row two-shot's share, every row
// otherwise) in TILE_M, by the grid's `grid_cols` columns in TILE_N. A grid wider is idle blocks,
// each still paying every barrier (the pull norm at 32 tokens ran 36 blocks for 4 rows a rank).
// The GEMM tail's GEMM strides over column tiles, so it is not cut.
constexpr int64_t tiles_of(Template t, const KernelConfig& c, int64_t rows, int64_t grid_cols,
                           int world) {
  if (gemms(op_of(t))) return launch_of(c).blocks_per_grid;
  const bool row_split = is_two_shot(t) && !slices_columns(t);
  const int64_t mine   = row_split ? (rows + world - 1) / world : rows;
  const int64_t tm = tile_m_of(c), tn = tile_n_of(c);
  return (mine + tm - 1) / tm * ((grid_cols + tn - 1) / tn);
}

// A TILED KERNEL'S CONFIG FITTED TO THE CALL: a zero field the template's own (another family's
// config is left for the check to refuse), tile_n the smallest built covering `tile_cols` where
// none is given, and the grid cut to the tiles there are.
constexpr KernelConfig fitted(Template t, KernelConfig c, int64_t rows, int64_t tile_cols,
                              int64_t grid_cols, int world) {
  const KernelConfig own_config = default_config(t, tile_cols);
  if (c.index() != own_config.index()) return c;
  std::visit(
      [&](auto& f) {
        const auto& o     = std::get<std::decay_t<decltype(f)>>(own_config);
        const auto filled = [](int& field, int its) {
          if (field == 0) field = its;
        };
        filled(f.launch.threads_per_block, o.launch.threads_per_block);
        filled(f.launch.blocks_per_grid, o.launch.blocks_per_grid);
        filled(f.launch.waves_per_eu, o.launch.waves_per_eu);
        if constexpr (requires { f.tile_m; }) filled(f.tile_m, o.tile_m);
        if constexpr (requires { f.tile_k; }) filled(f.tile_k, o.tile_k);
        if constexpr (requires { f.slice_k; }) filled(f.slice_k, o.slice_k);
        if constexpr (requires { f.reduce_scatter_blocks; })
          filled(f.reduce_scatter_blocks, o.reduce_scatter_blocks);
      },
      c);
  if (tile_n_of(c) == 0) set_tile_n(c, smallest_tile_n(t, c, tile_cols));
  if (tile_n_of(c) == 0) return c;
  const int64_t tiles = tiles_of(t, c, rows, grid_cols, world);
  int& blocks         = launch_of(c).blocks_per_grid;
  if (tiles < blocks) blocks = static_cast<int>(tiles > 0 ? tiles : 1);
  return c;
}

// =================================================================================================
// 4. THE CHECK: the first Error kernel `fn` at `c` meets on the call here, in this order, or none.
// Every Error is a capability (a kernel that cannot take the input), never "the unfused ops would
// be faster". Residency, which needs the compiled kernel, is each select's last check.
// =================================================================================================

// A KERNEL'S SCRATCH, row-major (a staged build needs none: it runs in passes of what the
// scratch holds): the plain two-shot's slice of packs; a column two-shot the whole reduced tensor
// (each rank's columns at their place); a row two-shot its rank's rows, twice where it leaves two
// results (out and the residual). A one-shot reads the inputs and keeps nothing.
constexpr int64_t scratch_need(Template t, int64_t rows, int64_t packs, int world) {
  if (!is_two_shot(t)) return 0;
  const OpType o = op_of(t);
  if (o == OpType::all_reduce)
    return (rows * packs + world - 1) / world * kBuild.memory.pack_bytes;
  if (slices_columns(t)) return rows * packs * kBuild.memory.pack_bytes;
  const bool two = o == OpType::all_reduce_add_rms_norm;
  return (rows + world - 1) / world * packs * (two ? 2 : 1) * kBuild.memory.pack_bytes;
}

// `row_elems`: the row the kernel reduces, in elements; `own`: the op's own Error on its own
// arguments, checked after the dtype. With no Handle (an experimental op, one rank) the checks
// that need peers are left out.
inline std::optional<Error> refused(const Handle* h, Template fn, const KernelConfig& c,
                                    DType dtype, int64_t rows, int64_t row_elems,
                                    int64_t tile_cols, std::optional<Error> own_error,
                                    const void* inp, bool staged, hipStream_t stream) {
  const int world = h ? h->world_size() : 1;
  if (h && !world_built(world)) return Error::world_not_built;
  if (!dtype_built(dtype)) return Error::dtype_not_built;
  if (own_error) return own_error;
  if (row_elems * elem_bytes(dtype) % kBuild.memory.pack_bytes != 0) return Error::row_not_packs;
  const int threads = launch_of(c).threads_per_block;
  if (c.index() != family_of(fn)) return Error::tile_not_built;
  if (has_builds(fn) && !built_at(fn, threads)) return Error::threads_not_built;
  if (has_builds(fn) && !built_at(fn, threads, launch_of(c).waves_per_eu))
    return Error::waves_not_built;
  if (has_tiles(fn) && tile_n_of(c) == 0) return Error::row_too_wide;
  if (has_tiles(fn) && !built(fn, c)) return Error::tile_not_built;
  if (has_tiles(fn) && tile_n_of(c) < tile_cols) return Error::row_too_wide;
  if (const GemmConfig* g = std::get_if<GemmConfig>(&c);
      g && !gemm_fits(kTarget, g->tile_m, g->tile_k, g->slice_k, threads))
    return Error::block_exceeds_lds;
  if (!h) return std::nullopt;
  const int64_t packs = packs_of(row_elems, dtype);
  if (!staged && scratch_need(fn, rows, packs, world) > h->scratch_bytes())
    return Error::scratch_too_small;
  // AN IN-PLACE BUILD ON AN EAGER INPUT reads it through the staging, copied in whole first.
  const int64_t bytes = rows * row_elems * elem_bytes(dtype);
  if (!staged && !h->reads_in_place(inp, stream) && bytes > h->staging_bytes())
    return Error::staging_too_small;
  return std::nullopt;
}


// THE WEIGHT a norm takes: its input's dtype, or fp32.
constexpr std::optional<Error> weight_refused(DType dtype, DType weight_dtype) {
  if (weight_dtype != dtype && weight_dtype != DType::f32) return Error::weight_not_built;
  return std::nullopt;
}

// =================================================================================================
// EACH OP'S SELECT: the op's launch, everything decided, from the call's primitives and its
// forcing, or the first Error the call meets. Each in four steps: 1. the kernel (the forced
// template and config, a zero field or a config not forced the template's own; else the op's
// tuned kernel; fitted to the call), 2. the checks, 3. its compiled instance, 4. the launch.
// =================================================================================================

// all_reduce: the sum of every rank's `inp` (`bytes` of `dtype`) into `out`. Not tiled: its
// launch, where not forced, is derived from the bytes. In place when the peers can read the input
// where it is (registered, or captured on this stream), otherwise its staged kernel.
inline std::variant<AllReduceLaunch, Error> select_all_reduce(
    const Handle& h, void* out, const void* inp, int64_t bytes, DType dtype,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> threads_per_block, std::optional<int> blocks_per_grid,
    hipStream_t stream) {
  const int world = h.world_size();
  // 1.
  const std::variant<Forced, Error> forcing =
      forced(OpType::all_reduce, algorithm, direction, threads_per_block, blocks_per_grid, {},
             [](LaunchConfig l, Template) -> std::variant<KernelConfig, Error> {
               return AllReduceConfig{l};
             });
  if (const Error* e = std::get_if<Error>(&forcing)) return *e;
  const Forced& f   = std::get<Forced>(forcing);
  const Template fn = f ? f->first
                        : bytes <= kTargetCalibration.all_reduce_one_shot_max_bytes
                              ? Template::all_reduce_pull_one_shot
                              : Template::all_reduce_pull_two_shot;
  KernelConfig c = f && f->second ? *f->second : AllReduceConfig{{0, 0, 1}};
  const LaunchConfig derived = launch_of(all_reduce_config(fn, bytes, world));
  LaunchConfig& g            = launch_of(c);
  if (g.threads_per_block == 0) g.threads_per_block = derived.threads_per_block;
  if (g.blocks_per_grid == 0) g.blocks_per_grid = derived.blocks_per_grid;
  if (g.waves_per_eu == 0) g.waves_per_eu = derived.waves_per_eu;
  const bool staged = !h.reads_in_place(inp, stream);
  // 2.
  if (const std::optional<Error> e = refused(&h, fn, c, dtype, 1, bytes / elem_bytes(dtype), 0,
                                             std::nullopt, inp, staged, stream))
    return *e;
  // 3.
  const void* kernel = nullptr;
  by_world(world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    by_dtype(dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      switch (fn) {
        case Template::all_reduce_pull_one_shot:
          by_config<Template::all_reduce_pull_one_shot>(c, [&](auto built) {
            constexpr int THREADS = decltype(built)::value.launch.threads_per_block;
            constexpr int WAVES   = decltype(built)::value.launch.waves_per_eu;
            kernel = staged ? instance<AllReduceOneShotStagedKernel>(
                                  all_reduce_pull_one_shot_staged<T, NG, THREADS, WAVES>)
                            : instance<AllReduceOneShotKernel>(
                                  all_reduce_pull_one_shot<T, NG, THREADS, WAVES>);
          });
          return;
        case Template::all_reduce_pull_two_shot:
          by_config<Template::all_reduce_pull_two_shot>(c, [&](auto built) {
            constexpr int THREADS = decltype(built)::value.launch.threads_per_block;
            constexpr int WAVES   = decltype(built)::value.launch.waves_per_eu;
            kernel = staged ? instance<AllReduceTwoShotStagedKernel>(
                                  all_reduce_pull_two_shot_staged<T, NG, THREADS, WAVES>)
                            : instance<AllReduceTwoShotKernel>(
                                  all_reduce_pull_two_shot<T, NG, THREADS, WAVES>);
          });
          return;
        default: not_this_ops(fn);
      }
    });
  });
  // 4.
  const AllReduceLaunch l{.kernel            = kernel,
                          .algorithm         = algorithm_of(fn),
                          .direction         = direction_of(fn),
                          .world             = world,
                          .threads_per_block = g.threads_per_block,
                          .blocks_per_grid   = g.blocks_per_grid,
                          .staged            = staged,
                          .stream            = stream,
                          .out               = out,
                          .inp               = inp,
                          .bytes             = bytes,
                          .dtype             = dtype};
  if (!resident(h, kernel, l.blocks_per_grid, l.threads_per_block))
    return Error::grid_not_resident;
  return l;
}

// all_reduce_rms_norm: out = rms_norm(all_reduce(inp), weight), vLLM's roundings exactly. inp
// [rows, hidden]; the weight `weight_dtype`, dtype or f32.
inline std::variant<AllReduceRmsNormLaunch, Error> select_all_reduce_rms_norm(
    const Handle& h, void* out, const void* inp, const void* weight, DType dtype,
    DType weight_dtype, int64_t rows, int64_t hidden, float eps,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_n, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  using K         = Template;
  const OpType op = OpType::all_reduce_rms_norm;
  const int world = h.world_size();
  // 1.
  const std::variant<Forced, Error> forcing =
      forced(op, algorithm, direction, threads_per_block, blocks_per_grid, {tile_n},
             [&](LaunchConfig l, Template) -> std::variant<KernelConfig, Error> {
               return RowConfig{l, own(tile_n)};
             });
  if (const Error* e = std::get_if<Error>(&forcing)) return *e;
  const Forced& f         = std::get<Forced>(forcing);
  const TunedKernel tuned = f ? TunedKernel{} : pick(op, world, rows, hidden, hidden);
  const Template fn       = f ? f->first : tuned.fn;
  const KernelConfig c =
      fitted(fn, f ? f->second.value_or(zero_config(family_of(fn))) : tuned.config, rows, hidden,
             hidden, world);
  // 2.
  if (const std::optional<Error> e = refused(&h, fn, c, dtype, rows, hidden, hidden,
                                             weight_refused(dtype, weight_dtype), inp, false,
                                             stream))
    return *e;
  // 3.
  const void* kernel = nullptr;
  by_world(world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    by_dtype(dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      by_weight<T>(weight_dtype, dtype, [&](auto w) {
        using W = typename decltype(w)::t;
        switch (fn) {
          case K::all_reduce_pull_one_shot_rms_norm:
            return by_config<K::all_reduce_pull_one_shot_rms_norm>(c, [&](auto cc) {
              constexpr RowConfig C = decltype(cc)::value;
              constexpr int TN = C.tile_n, TPB = C.launch.threads_per_block,
                WPE = C.launch.waves_per_eu;
              kernel = instance<AllReduceRmsNormOneShotKernel>(
                  all_reduce_pull_one_shot_rms_norm<T, W, NG, TN, TPB, WPE>);
            });
          case K::all_reduce_pull_two_shot_rms_norm:
            return by_config<K::all_reduce_pull_two_shot_rms_norm>(c, [&](auto cc) {
              constexpr RowConfig C = decltype(cc)::value;
              constexpr int TN = C.tile_n, TPB = C.launch.threads_per_block,
                WPE = C.launch.waves_per_eu;
              kernel = instance<AllReduceRmsNormTwoShotKernel>(
                  all_reduce_pull_two_shot_rms_norm<T, W, NG, TN, TPB, WPE>);
            });
          case K::all_reduce_push_two_shot_rms_norm:
            return by_config<K::all_reduce_push_two_shot_rms_norm>(c, [&](auto cc) {
              constexpr RowConfig C = decltype(cc)::value;
              constexpr int TN = C.tile_n, TPB = C.launch.threads_per_block,
                WPE = C.launch.waves_per_eu;
              kernel = instance<AllReduceRmsNormTwoShotKernel>(
                  all_reduce_push_two_shot_rms_norm<T, W, NG, TN, TPB, WPE>);
            });
          default: not_this_ops(fn);
        }
      });
    });
  });
  // 4.
  const RowConfig& r = std::get<RowConfig>(c);
  const AllReduceRmsNormLaunch l{.kernel            = kernel,
                                 .algorithm         = algorithm_of(fn),
                                 .direction         = direction_of(fn),
                                 .world             = world,
                                 .tile_n            = r.tile_n,
                                 .threads_per_block = r.launch.threads_per_block,
                                 .blocks_per_grid   = r.launch.blocks_per_grid,
                                 .stream            = stream,
                                 .out               = out,
                                 .inp               = inp,
                                 .weight            = weight,
                                 .dtype             = dtype,
                                 .weight_dtype      = weight_dtype,
                                 .rows              = rows,
                                 .hidden            = hidden,
                                 .eps               = eps};
  if (!resident(h, kernel, l.blocks_per_grid, l.threads_per_block))
    return Error::grid_not_resident;
  return l;
}

// all_reduce_add_rms_norm: out, residual_out = fused_add_rms_norm(all_reduce(inp), residual,
// weight), vLLM's roundings exactly.
inline std::variant<AllReduceAddRmsNormLaunch, Error> select_all_reduce_add_rms_norm(
    const Handle& h, void* out, void* residual_out, const void* inp, const void* residual,
    const void* weight, DType dtype, DType weight_dtype, int64_t rows, int64_t hidden, float eps,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_n, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  using K         = Template;
  const OpType op = OpType::all_reduce_add_rms_norm;
  const int world = h.world_size();
  // 1.
  const std::variant<Forced, Error> forcing =
      forced(op, algorithm, direction, threads_per_block, blocks_per_grid, {tile_n},
             [&](LaunchConfig l, Template) -> std::variant<KernelConfig, Error> {
               return RowConfig{l, own(tile_n)};
             });
  if (const Error* e = std::get_if<Error>(&forcing)) return *e;
  const Forced& f         = std::get<Forced>(forcing);
  const TunedKernel tuned = f ? TunedKernel{} : pick(op, world, rows, hidden, hidden);
  const Template fn       = f ? f->first : tuned.fn;
  const KernelConfig c =
      fitted(fn, f ? f->second.value_or(zero_config(family_of(fn))) : tuned.config, rows, hidden,
             hidden, world);
  // 2.
  if (const std::optional<Error> e = refused(&h, fn, c, dtype, rows, hidden, hidden,
                                             weight_refused(dtype, weight_dtype), inp, false,
                                             stream))
    return *e;
  // 3.
  const void* kernel = nullptr;
  by_world(world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    by_dtype(dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      by_weight<T>(weight_dtype, dtype, [&](auto w) {
        using W = typename decltype(w)::t;
        switch (fn) {
          case K::all_reduce_pull_one_shot_add_rms_norm:
            return by_config<K::all_reduce_pull_one_shot_add_rms_norm>(c, [&](auto cc) {
              constexpr RowConfig C = decltype(cc)::value;
              constexpr int TN = C.tile_n, TPB = C.launch.threads_per_block,
                WPE = C.launch.waves_per_eu;
              kernel = instance<AllReduceAddRmsNormOneShotKernel>(
                  all_reduce_pull_one_shot_add_rms_norm<T, W, NG, TN, TPB, WPE>);
            });
          case K::all_reduce_pull_two_shot_add_rms_norm:
            return by_config<K::all_reduce_pull_two_shot_add_rms_norm>(c, [&](auto cc) {
              constexpr RowConfig C = decltype(cc)::value;
              constexpr int TN = C.tile_n, TPB = C.launch.threads_per_block,
                WPE = C.launch.waves_per_eu;
              kernel = instance<AllReduceAddRmsNormTwoShotKernel>(
                  all_reduce_pull_two_shot_add_rms_norm<T, W, NG, TN, TPB, WPE>);
            });
          case K::all_reduce_push_two_shot_add_rms_norm:
            return by_config<K::all_reduce_push_two_shot_add_rms_norm>(c, [&](auto cc) {
              constexpr RowConfig C = decltype(cc)::value;
              constexpr int TN = C.tile_n, TPB = C.launch.threads_per_block,
                WPE = C.launch.waves_per_eu;
              kernel = instance<AllReduceAddRmsNormTwoShotKernel>(
                  all_reduce_push_two_shot_add_rms_norm<T, W, NG, TN, TPB, WPE>);
            });
          default: not_this_ops(fn);
        }
      });
    });
  });
  // 4.
  const RowConfig& r = std::get<RowConfig>(c);
  const AllReduceAddRmsNormLaunch l{.kernel            = kernel,
                                    .algorithm         = algorithm_of(fn),
                                    .direction         = direction_of(fn),
                                    .world             = world,
                                    .tile_n            = r.tile_n,
                                    .threads_per_block = r.launch.threads_per_block,
                                    .blocks_per_grid   = r.launch.blocks_per_grid,
                                    .stream            = stream,
                                    .out               = out,
                                    .residual_out      = residual_out,
                                    .inp               = inp,
                                    .residual          = residual,
                                    .weight            = weight,
                                    .dtype             = dtype,
                                    .weight_dtype      = weight_dtype,
                                    .rows              = rows,
                                    .hidden            = hidden,
                                    .eps               = eps};
  if (!resident(h, kernel, l.blocks_per_grid, l.threads_per_block))
    return Error::grid_not_resident;
  return l;
}

// all_reduce_add_attn_res_rms_norm: Kimi-K3's AttnRes and its RMSNorm on each row of the
// all-reduced sum of `inp`. With `has_prefix` the sum is added to `prefix` in place; without, the
// sum IS the new prefix. The pull two-shot's config family has TILE_M and its reduce-scatter
// blocks; the one-shot's and push's (a row a tile) have neither.
inline std::variant<AllReduceAddAttnResRmsNormLaunch, Error>
select_all_reduce_add_attn_res_rms_norm(
    const Handle& h, void* prefix, void* out, const void* inp, void* blocks,
    int64_t block_stride_m, int64_t block_stride_r, const void* norm_weight,
    const void* qk_weight, const void* out_norm_weight, DType dtype, int64_t rows, int64_t hidden,
    int num_blocks, int write_idx, float eps, float out_eps, bool has_prefix,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_m, std::optional<int> tile_n, std::optional<int> tile_k,
    std::optional<int> reduce_scatter_blocks, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  using K         = Template;
  const OpType op = OpType::all_reduce_add_attn_res_rms_norm;
  const int world = h.world_size();
  // 1.
  const std::variant<Forced, Error> forcing =
      forced(op, algorithm, direction, threads_per_block, blocks_per_grid,
             {tile_m, tile_n, tile_k, reduce_scatter_blocks},
             [&](LaunchConfig l, Template t) -> std::variant<KernelConfig, Error> {
               if (t == K::all_reduce_pull_two_shot_add_attn_res_rms_norm)
                 return AttnResPullConfig{l, own(tile_m), own(tile_n), own(tile_k),
                                          own(reduce_scatter_blocks)};
               if (tile_m || reduce_scatter_blocks) return Error::field_not_this_templates;
               return AttnResConfig{l, own(tile_n), own(tile_k)};
             });
  if (const Error* e = std::get_if<Error>(&forcing)) return *e;
  const Forced& f         = std::get<Forced>(forcing);
  const TunedKernel tuned = f ? TunedKernel{} : pick(op, world, rows, hidden, hidden);
  const Template fn       = f ? f->first : tuned.fn;
  const KernelConfig c =
      fitted(fn, f ? f->second.value_or(zero_config(family_of(fn))) : tuned.config, rows, hidden,
             hidden, world);
  // 2.
  if (const std::optional<Error> e =
          refused(&h, fn, c, dtype, rows, hidden, hidden, std::nullopt, inp, false, stream))
    return *e;
  // 3.
  const void* kernel = nullptr;
  by_world(world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    by_dtype(dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      switch (fn) {
        case K::all_reduce_pull_one_shot_add_attn_res_rms_norm:
          return by_config<K::all_reduce_pull_one_shot_add_attn_res_rms_norm>(c, [&](auto cc) {
            constexpr AttnResConfig C = decltype(cc)::value;
            constexpr int TN = C.tile_n, TK = C.tile_k, TPB = C.launch.threads_per_block,
                WPE = C.launch.waves_per_eu;
            kernel = has_prefix
                         ? instance<AllReduceAddAttnResRmsNormOneShotKernel>(
                               all_reduce_pull_one_shot_add_attn_res_rms_norm<T, NG, true, TN,
                                                                              TK, TPB, WPE>)
                         : instance<AllReduceAddAttnResRmsNormOneShotKernel>(
                               all_reduce_pull_one_shot_add_attn_res_rms_norm<T, NG, false, TN,
                                                                              TK, TPB, WPE>);
          });
        case K::all_reduce_pull_two_shot_add_attn_res_rms_norm:
          return by_config<K::all_reduce_pull_two_shot_add_attn_res_rms_norm>(c, [&](auto cc) {
            constexpr AttnResPullConfig C = decltype(cc)::value;
            constexpr int TM = C.tile_m, TN = C.tile_n, TK = C.tile_k,
                          TPB = C.launch.threads_per_block,
                WPE = C.launch.waves_per_eu;
            kernel = has_prefix
                         ? instance<AllReduceAddAttnResRmsNormPullKernel>(
                               all_reduce_pull_two_shot_add_attn_res_rms_norm<T, NG, true, TM, TN,
                                                                              TK, TPB, WPE>)
                         : instance<AllReduceAddAttnResRmsNormPullKernel>(
                               all_reduce_pull_two_shot_add_attn_res_rms_norm<T, NG, false, TM,
                                                                              TN, TK, TPB, WPE>);
          });
        case K::all_reduce_push_two_shot_add_attn_res_rms_norm:
          return by_config<K::all_reduce_push_two_shot_add_attn_res_rms_norm>(c, [&](auto cc) {
            constexpr AttnResConfig C = decltype(cc)::value;
            constexpr int TN = C.tile_n, TK = C.tile_k, TPB = C.launch.threads_per_block,
                WPE = C.launch.waves_per_eu;
            kernel = has_prefix
                         ? instance<AllReduceAddAttnResRmsNormPushKernel>(
                               all_reduce_push_two_shot_add_attn_res_rms_norm<T, NG, true, TN,
                                                                              TK, TPB, WPE>)
                         : instance<AllReduceAddAttnResRmsNormPushKernel>(
                               all_reduce_push_two_shot_add_attn_res_rms_norm<T, NG, false, TN,
                                                                              TK, TPB, WPE>);
          });
        default: not_this_ops(fn);
      }
    });
  });
  // 4.
  const AttnResPullConfig* pull = std::get_if<AttnResPullConfig>(&c);
  const LaunchConfig& g         = launch_of(c);
  const int tm = pull ? pull->tile_m : 1;
  const int tk = pull ? pull->tile_k : std::get<AttnResConfig>(c).tile_k;
  const int rs = pull ? pull->reduce_scatter_blocks : 0;
  const AllReduceAddAttnResRmsNormLaunch l{.kernel                = kernel,
                                           .algorithm             = algorithm_of(fn),
                                           .direction             = direction_of(fn),
                                           .world                 = world,
                                           .tile_m                = tm,
                                           .tile_n                = tile_n_of(c),
                                           .tile_k                = tk,
                                           .threads_per_block     = g.threads_per_block,
                                           .blocks_per_grid       = g.blocks_per_grid,
                                           .reduce_scatter_blocks = rs,
                                           .stream                = stream,
                                           .prefix                = prefix,
                                           .out                   = out,
                                           .inp                   = inp,
                                           .blocks                = blocks,
                                           .block_stride_m        = block_stride_m,
                                           .block_stride_r        = block_stride_r,
                                           .norm_weight           = norm_weight,
                                           .qk_weight             = qk_weight,
                                           .out_norm_weight       = out_norm_weight,
                                           .dtype                 = dtype,
                                           .rows                  = rows,
                                           .hidden                = hidden,
                                           .num_blocks            = num_blocks,
                                           .write_idx             = write_idx,
                                           .eps                   = eps,
                                           .out_eps               = out_eps,
                                           .has_prefix            = has_prefix};
  if (!resident(h, kernel, l.blocks_per_grid, l.threads_per_block))
    return Error::grid_not_resident;
  return l;
}

// all_reduce_rms_norm_gemm: out = rms_norm(all_reduce(inp), norm_weight) @ gemm_weight^T, out
// [rows, n_cols] at `out_stride` (a column slice of a wider buffer); gemm_weight [n_cols, hidden];
// `workspace` holds the normed rows, inp's shape.
inline std::variant<AllReduceRmsNormGemmLaunch, Error> select_all_reduce_rms_norm_gemm(
    const Handle& h, void* out, int64_t out_stride, const void* inp, const void* norm_weight,
    float eps, const void* gemm_weight, int64_t n_cols, void* workspace, DType dtype,
    int64_t rows, int64_t hidden, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> tile_m, std::optional<int> tile_n,
    std::optional<int> tile_k, std::optional<int> slice_k, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  using K         = Template;
  const OpType op = OpType::all_reduce_rms_norm_gemm;
  const int world = h.world_size();
  // 1.
  const std::variant<Forced, Error> forcing =
      forced(op, algorithm, direction, threads_per_block, blocks_per_grid,
             {tile_m, tile_n, tile_k, slice_k},
             [&](LaunchConfig l, Template) -> std::variant<KernelConfig, Error> {
               return GemmConfig{l, own(tile_m), own(tile_n), own(tile_k), own(slice_k)};
             });
  if (const Error* e = std::get_if<Error>(&forcing)) return *e;
  const Forced& f         = std::get<Forced>(forcing);
  const TunedKernel tuned = f ? TunedKernel{} : pick(op, world, rows, hidden, hidden);
  const Template fn       = f ? f->first : tuned.fn;
  const KernelConfig c =
      fitted(fn, f ? f->second.value_or(zero_config(family_of(fn))) : tuned.config, rows, hidden,
             hidden, world);
  // 2.
  if (const std::optional<Error> e =
          refused(&h, fn, c, dtype, rows, hidden, hidden, std::nullopt, inp, false, stream))
    return *e;
  // 3. The four GEMM-tail templates share one config list, so one lookup serves each.
  const void* kernel = nullptr;
  by_world(world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    by_dtype(dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      by_config<K::all_reduce_pull_one_shot_rms_norm_gemm_add>(c, [&](auto cc) {
        constexpr GemmConfig C = decltype(cc)::value;
        constexpr int TM = C.tile_m, TN = C.tile_n, TK = C.tile_k, SK = C.slice_k,
                      TPB = C.launch.threads_per_block,
                WPE = C.launch.waves_per_eu;
        switch (fn) {
          case K::all_reduce_pull_one_shot_rms_norm_gemm:
            kernel = instance<AllReduceRmsNormGemmOneShotKernel>(
                all_reduce_pull_one_shot_rms_norm_gemm<T, NG, TM, TN, TK, SK, TPB, WPE>);
            return;
          case K::all_reduce_pull_two_shot_rms_norm_gemm:
            kernel = instance<AllReduceRmsNormGemmTwoShotKernel>(
                all_reduce_pull_two_shot_rms_norm_gemm<T, NG, TM, TN, TK, SK, TPB, WPE>);
            return;
          default: not_this_ops(fn);
        }
      });
    });
  });
  // 4.
  const GemmConfig& g = std::get<GemmConfig>(c);
  const AllReduceRmsNormGemmLaunch l{.kernel            = kernel,
                                     .algorithm         = algorithm_of(fn),
                                     .direction         = direction_of(fn),
                                     .world             = world,
                                     .tile_m            = g.tile_m,
                                     .tile_n            = g.tile_n,
                                     .tile_k            = g.tile_k,
                                     .slice_k           = g.slice_k,
                                     .threads_per_block = g.launch.threads_per_block,
                                     .blocks_per_grid   = g.launch.blocks_per_grid,
                                     .stream            = stream,
                                     .out               = out,
                                     .out_stride        = out_stride,
                                     .inp               = inp,
                                     .norm_weight       = norm_weight,
                                     .eps               = eps,
                                     .gemm_weight       = gemm_weight,
                                     .n_cols            = n_cols,
                                     .workspace         = workspace,
                                     .dtype             = dtype,
                                     .rows              = rows,
                                     .hidden            = hidden};
  if (!resident(h, kernel, l.blocks_per_grid, l.threads_per_block))
    return Error::grid_not_resident;
  return l;
}

// all_reduce_rms_norm_gemm_add: the latent MoE tail: as all_reduce_rms_norm_gemm, the
// product added into out.
inline std::variant<AllReduceRmsNormGemmAddLaunch, Error> select_all_reduce_rms_norm_gemm_add(
    const Handle& h, void* out, int64_t out_stride, const void* inp, const void* norm_weight,
    float eps, const void* gemm_weight, int64_t n_cols, void* workspace, DType dtype,
    int64_t rows, int64_t hidden, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> tile_m, std::optional<int> tile_n,
    std::optional<int> tile_k, std::optional<int> slice_k, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  using K         = Template;
  const OpType op = OpType::all_reduce_rms_norm_gemm_add;
  const int world = h.world_size();
  // 1.
  const std::variant<Forced, Error> forcing =
      forced(op, algorithm, direction, threads_per_block, blocks_per_grid,
             {tile_m, tile_n, tile_k, slice_k},
             [&](LaunchConfig l, Template) -> std::variant<KernelConfig, Error> {
               return GemmConfig{l, own(tile_m), own(tile_n), own(tile_k), own(slice_k)};
             });
  if (const Error* e = std::get_if<Error>(&forcing)) return *e;
  const Forced& f         = std::get<Forced>(forcing);
  const TunedKernel tuned = f ? TunedKernel{} : pick(op, world, rows, hidden, hidden);
  const Template fn       = f ? f->first : tuned.fn;
  const KernelConfig c =
      fitted(fn, f ? f->second.value_or(zero_config(family_of(fn))) : tuned.config, rows, hidden,
             hidden, world);
  // 2.
  if (const std::optional<Error> e =
          refused(&h, fn, c, dtype, rows, hidden, hidden, std::nullopt, inp, false, stream))
    return *e;
  // 3. The four GEMM-tail templates share one config list, so one lookup serves each.
  const void* kernel = nullptr;
  by_world(world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    by_dtype(dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      by_config<K::all_reduce_pull_one_shot_rms_norm_gemm_add>(c, [&](auto cc) {
        constexpr GemmConfig C = decltype(cc)::value;
        constexpr int TM = C.tile_m, TN = C.tile_n, TK = C.tile_k, SK = C.slice_k,
                      TPB = C.launch.threads_per_block,
                WPE = C.launch.waves_per_eu;
        switch (fn) {
          case K::all_reduce_pull_one_shot_rms_norm_gemm_add:
            kernel = instance<AllReduceRmsNormGemmOneShotKernel>(
                all_reduce_pull_one_shot_rms_norm_gemm_add<T, NG, TM, TN, TK, SK, TPB, WPE>);
            return;
          case K::all_reduce_pull_two_shot_rms_norm_gemm_add:
            kernel = instance<AllReduceRmsNormGemmTwoShotKernel>(
                all_reduce_pull_two_shot_rms_norm_gemm_add<T, NG, TM, TN, TK, SK, TPB, WPE>);
            return;
          default: not_this_ops(fn);
        }
      });
    });
  });
  // 4.
  const GemmConfig& g = std::get<GemmConfig>(c);
  const AllReduceRmsNormGemmAddLaunch l{.kernel            = kernel,
                                        .algorithm         = algorithm_of(fn),
                                        .direction         = direction_of(fn),
                                        .world             = world,
                                        .tile_m            = g.tile_m,
                                        .tile_n            = g.tile_n,
                                        .tile_k            = g.tile_k,
                                        .slice_k           = g.slice_k,
                                        .threads_per_block = g.launch.threads_per_block,
                                        .blocks_per_grid   = g.launch.blocks_per_grid,
                                        .stream            = stream,
                                        .out               = out,
                                        .out_stride        = out_stride,
                                        .inp               = inp,
                                        .norm_weight       = norm_weight,
                                        .eps               = eps,
                                        .gemm_weight       = gemm_weight,
                                        .n_cols            = n_cols,
                                        .workspace         = workspace,
                                        .dtype             = dtype,
                                        .rows              = rows,
                                        .hidden            = hidden};
  if (!resident(h, kernel, l.blocks_per_grid, l.threads_per_block))
    return Error::grid_not_resident;
  return l;
}

// all_reduce_rms_scale_add: Kimi-K3's latent MoE tail with one all-reduce. inp's row is [shared |
// projected | latent], widths hidden, hidden and latent, summed over the ranks; out [rows, hidden]
// = shared + projected * rsqrt(mean(latent^2) + eps). Its tile holds the latent whole (each tile
// needs the row's RMS) beside a TILE_N slice of the hidden, so its grid tiles the hidden.
inline std::variant<AllReduceRmsScaleAddLaunch, Error> select_all_reduce_rms_scale_add(
    const Handle& h, void* out, const void* inp, DType dtype, int64_t rows, int64_t hidden,
    int64_t latent, float eps, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> tile_n,
    std::optional<int> threads_per_block, std::optional<int> blocks_per_grid,
    hipStream_t stream) {
  using K           = Template;
  const OpType op   = OpType::all_reduce_rms_scale_add;
  const int world   = h.world_size();
  const int64_t row = 2 * hidden + latent;
  // 1.
  const std::variant<Forced, Error> forcing =
      forced(op, algorithm, direction, threads_per_block, blocks_per_grid, {tile_n},
             [&](LaunchConfig l, Template) -> std::variant<KernelConfig, Error> {
               return RowConfig{l, own(tile_n)};
             });
  if (const Error* e = std::get_if<Error>(&forcing)) return *e;
  const Forced& f         = std::get<Forced>(forcing);
  const TunedKernel tuned = f ? TunedKernel{} : pick(op, world, rows, row, latent);
  const Template fn       = f ? f->first : tuned.fn;
  const KernelConfig c =
      fitted(fn, f ? f->second.value_or(zero_config(family_of(fn))) : tuned.config, rows, latent,
             hidden, world);
  // 2.
  const int e = elem_bytes(dtype);
  const std::optional<Error> widths =
      hidden * e % kBuild.memory.pack_bytes != 0 || latent * e % kBuild.memory.pack_bytes != 0 ||
              latent < 1
          ? std::optional<Error>{Error::widths_not_packs}
          : std::nullopt;
  if (const std::optional<Error> err =
          refused(&h, fn, c, dtype, rows, row, latent, widths, inp, false, stream))
    return *err;
  // 3. The one-shot's and the two-shot's builds are one list.
  static_assert(same_builds(K::all_reduce_pull_one_shot_rms_scale_add,
                            K::all_reduce_pull_two_shot_rms_scale_add));
  const void* kernel = nullptr;
  by_world(world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    by_dtype(dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      by_config<K::all_reduce_pull_one_shot_rms_scale_add>(c, [&](auto cc) {
        constexpr RowConfig C = decltype(cc)::value;
        constexpr int TN = C.tile_n, TPB = C.launch.threads_per_block,
                WPE = C.launch.waves_per_eu;
        switch (fn) {
          case K::all_reduce_pull_one_shot_rms_scale_add:
            kernel = instance<AllReduceRmsScaleAddOneShotKernel>(
                all_reduce_pull_one_shot_rms_scale_add<T, NG, TN, TPB, WPE>);
            return;
          case K::all_reduce_pull_two_shot_rms_scale_add:
            kernel = instance<AllReduceRmsScaleAddTwoShotKernel>(
                all_reduce_pull_two_shot_rms_scale_add<T, NG, TN, TPB, WPE>);
            return;
          default: not_this_ops(fn);
        }
      });
    });
  });
  // 4.
  const RowConfig& r = std::get<RowConfig>(c);
  const AllReduceRmsScaleAddLaunch l{.kernel            = kernel,
                                     .algorithm         = algorithm_of(fn),
                                     .direction         = direction_of(fn),
                                     .world             = world,
                                     .tile_n            = r.tile_n,
                                     .threads_per_block = r.launch.threads_per_block,
                                     .blocks_per_grid   = r.launch.blocks_per_grid,
                                     .stream            = stream,
                                     .out               = out,
                                     .inp               = inp,
                                     .dtype             = dtype,
                                     .rows              = rows,
                                     .hidden            = hidden,
                                     .latent            = latent,
                                     .eps               = eps};
  if (!resident(h, kernel, l.blocks_per_grid, l.threads_per_block))
    return Error::grid_not_resident;
  return l;
}

namespace experimental {

// add_attn_res_rms_norm: AttnRes and its RMSNorm on a local `delta`, no all-reduce: prefix +=
// delta (rounded once), then Triton's attn_res over the blocks and the new prefix. One rank, no
// Handle: its one template needs no algorithm to force a config, and the checks that need peers
// are left out.
inline std::variant<AddAttnResRmsNormLaunch, Error> select_add_attn_res_rms_norm(
    void* prefix, void* out, const void* delta, void* blocks, int64_t block_stride_m,
    int64_t block_stride_r, const void* norm_weight, const void* qk_weight,
    const void* out_norm_weight, DType dtype, int64_t rows, int64_t hidden, int num_blocks,
    int write_idx, float eps, float out_eps, std::optional<int> tile_n,
    std::optional<int> tile_k, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  const OpType op = OpType::add_attn_res_rms_norm;
  const int world = 1;
  // 1.
  const std::variant<Forced, Error> forcing =
      forced(op, std::nullopt, std::nullopt, threads_per_block, blocks_per_grid, {tile_n, tile_k},
             [&](LaunchConfig l, Template) -> std::variant<KernelConfig, Error> {
               return AttnResConfig{l, own(tile_n), own(tile_k)};
             });
  if (const Error* e = std::get_if<Error>(&forcing)) return *e;
  const Forced& f         = std::get<Forced>(forcing);
  const TunedKernel tuned = f ? TunedKernel{} : pick(op, world, rows, hidden, hidden);
  const Template fn       = f ? f->first : tuned.fn;
  const KernelConfig c =
      fitted(fn, f ? f->second.value_or(zero_config(family_of(fn))) : tuned.config, rows, hidden,
             hidden, world);
  // 2.
  if (const std::optional<Error> e = refused(nullptr, fn, c, dtype, rows, hidden, hidden,
                                             std::nullopt, delta, false, stream))
    return *e;
  // 3.
  const void* kernel = nullptr;
  by_dtype(dtype, [&](auto t) {
    using T = typename decltype(t)::t;
    by_config<Template::add_attn_res_rms_norm>(c, [&](auto cc) {
      constexpr AttnResConfig C = decltype(cc)::value;
      constexpr int TN = C.tile_n, TK = C.tile_k, TPB = C.launch.threads_per_block,
                WPE = C.launch.waves_per_eu;
      kernel = instance<AddAttnResRmsNormKernel>(add_attn_res_rms_norm<T, TN, TK, TPB, WPE>);
    });
  });
  // 4.
  const AttnResConfig& a = std::get<AttnResConfig>(c);
  return AddAttnResRmsNormLaunch{.kernel            = kernel,
                                 .algorithm         = algorithm_of(fn),
                                 .direction         = direction_of(fn),
                                 .world             = world,
                                 .tile_n            = a.tile_n,
                                 .tile_k            = a.tile_k,
                                 .threads_per_block = a.launch.threads_per_block,
                                 .blocks_per_grid   = a.launch.blocks_per_grid,
                                 .stream            = stream,
                                 .prefix            = prefix,
                                 .out               = out,
                                 .delta             = delta,
                                 .blocks            = blocks,
                                 .block_stride_m    = block_stride_m,
                                 .block_stride_r    = block_stride_r,
                                 .norm_weight       = norm_weight,
                                 .qk_weight         = qk_weight,
                                 .out_norm_weight   = out_norm_weight,
                                 .dtype             = dtype,
                                 .rows              = rows,
                                 .hidden            = hidden,
                                 .num_blocks        = num_blocks,
                                 .write_idx         = write_idx,
                                 .eps               = eps,
                                 .out_eps           = out_eps};
}

// THE PROBE'S: one block of 64 threads, its twin on every rank.
inline std::variant<ProbeBarrierLaunch, Error> select_probe_barrier(const Handle& h,
                                                                    hipStream_t stream) {
  const int world = h.world_size();
  if (!world_built(world)) return Error::world_not_built;
  const void* kernel = nullptr;
  by_world(world, [&](auto ng) {
    kernel = instance<ProbeBarrierKernel>(probe_barrier<decltype(ng)::value>);
  });
  return ProbeBarrierLaunch{.kernel            = kernel,
                            .world             = world,
                            .threads_per_block = kWaveSize,
                            .blocks_per_grid   = 1,
                            .stream            = stream};
}

// One thread on each rank of the pair (a wave launched).
inline std::variant<PingPongLaunch, Error> select_ping_pong(const Handle& h, int peer, int iters,
                                                            void* ticks, hipStream_t stream) {
  const int world = h.world_size();
  if (!world_built(world)) return Error::world_not_built;
  if (peer < 0 || peer >= world || peer == h.rank() || iters < 1)
    return Error::probe_out_of_range;
  return PingPongLaunch{.kernel            = instance<PingPongKernel>(ping_pong),
                        .world             = world,
                        .threads_per_block = kWaveSize,
                        .blocks_per_grid   = 1,
                        .stream            = stream,
                        .peer              = peer,
                        .iters             = iters,
                        .ticks             = ticks};
}

// Every thread of `blocks` full blocks streaming; `peer` < 0 every peer. A buffer the peers read
// (the staging, or one registered) whole packs long, within the staging's size.
inline std::variant<LinkTrafficLaunch, Error> select_link_traffic(
    const Handle& h, const void* buffer, int64_t bytes, Traffic mode, int peer, int blocks,
    int pullers, void* sink, hipStream_t stream) {
  const int world = h.world_size();
  if (!world_built(world)) return Error::world_not_built;
  if (bytes < kBuild.memory.pack_bytes || bytes % kBuild.memory.pack_bytes != 0 ||
      bytes > h.staging_bytes() || peer >= world || peer == h.rank() || blocks < 1 ||
      blocks > kMaxBlocks || pullers < 0 || pullers > blocks)
    return Error::probe_out_of_range;
  const void* kernel = nullptr;
  by_world(world, [&](auto ng) {
    kernel = instance<LinkTrafficKernel>(link_traffic<c10::BFloat16, decltype(ng)::value>);
  });
  const LinkTrafficLaunch l{.kernel            = kernel,
                            .world             = world,
                            .threads_per_block = kBuild.kernels.max_threads,
                            .blocks_per_grid   = blocks,
                            .stream            = stream,
                            .buffer            = buffer ? buffer : h.staging(),
                            .bytes             = bytes,
                            .mode              = mode,
                            .peer              = peer,
                            .pullers           = pullers,
                            .sink              = sink};
  if (!resident(h, kernel, l.blocks_per_grid, l.threads_per_block))
    return Error::grid_not_resident;
  return l;
}

}  // namespace experimental

// =================================================================================================
// EVERY TUNED SELECTION FITS, for every tiled op at the smallest call and a large one: an op never
// declines (a fusion that is on runs its fused op), and a kernel past a capability is a compile
// error, not one that overruns its signal slots or register arrays.
// =================================================================================================

constexpr bool fits(Template fn, const KernelConfig& c) {
  const LaunchConfig& l = launch_of(c);
  if (l.blocks_per_grid < 1 || l.blocks_per_grid > kMaxBlocks) return false;
  if (has_tiles(fn) && tile_n_of(c) == 0) return false;
  const int t = l.threads_per_block;
  return t >= kWaveSize && t <= kBuild.kernels.max_threads && t % kWaveSize == 0;
}
// The tuned kernel for `rows` rows of `row` elements, fitted.
constexpr bool tuned_fits(OpType o, int world, int64_t rows, int64_t row, int64_t tile_cols,
                          int64_t grid_cols) {
  const TunedKernel t = pick(o, world, rows, row, tile_cols);
  return fits(t.fn, fitted(t.fn, t.config, rows, tile_cols, grid_cols, world));
}
constexpr bool selections_fit() {
  for (const std::array<int64_t, 3> call : {std::array<int64_t, 3>{2, 1, 8},
                                            std::array<int64_t, 3>{kMaxRanks, 4096, 7168}}) {
    const int w = static_cast<int>(call[0]);
    const int64_t rows = call[1], hidden = call[2], latent = hidden / 2;
    for (const OpType o :
         {OpType::all_reduce_rms_norm, OpType::all_reduce_add_rms_norm,
          OpType::all_reduce_add_attn_res_rms_norm, OpType::all_reduce_rms_norm_gemm,
          OpType::all_reduce_rms_norm_gemm_add})
      if (!tuned_fits(o, w, rows, hidden, hidden, hidden)) return false;
    // The one-all-reduce tail: [shared | projected | latent].
    if (!tuned_fits(OpType::all_reduce_rms_scale_add, w, rows, 2 * hidden + latent, latent,
                    hidden))
      return false;
    if (!tuned_fits(OpType::add_attn_res_rms_norm, 1, rows, hidden, hidden, hidden)) return false;
    // The plain all-reduce's derived launch.
    for (const Template t :
         {Template::all_reduce_pull_one_shot, Template::all_reduce_pull_two_shot})
      if (!fits(t, all_reduce_config(t, rows * hidden * 2, w))) return false;
  }
  return true;
}
static_assert(selections_fit(), "a selection declines, or exceeds a kernel capability");

}  // namespace hip_comms
