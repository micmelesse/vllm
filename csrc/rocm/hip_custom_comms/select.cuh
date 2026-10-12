// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SELECT, THE ONLY CHOICE, read by each op's select_<op> (interface.cuh): what was tuned (each
// op's kernels: for each world and row width, the template and complete KernelConfig, its build
// and its launch, that won from a number of rows up, written by the tuner, tune.py, from a sweep;
// an entry says what ran fastest, the why is the sweep's figure, cited beside it), then the steps
// every select takes, each on primitives: the forced kernel, given whole, or the tuned kernel
// picked as listed; the grid cut to the tiles there are; and the first Error the kernel meets
// here. Nothing is filled in: a config is complete wherever it is written.

#pragma once

#ifndef HIP_CUSTOM_COMMS_INTERFACE
#error "include hip_custom_comms.cuh, the one interface, not its parts"
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
#include <utility>
#include <type_traits>
#include <utility>
#include <variant>

#include "kernels/all_reduce_pull_one_shot.cuh"
#include "kernels/all_reduce_pull_one_shot_add_attn_res_rms_norm.cuh"
#include "kernels/all_reduce_pull_one_shot_add_rms_norm.cuh"
#include "kernels/all_reduce_pull_two_shot.cuh"
#include "kernels/all_reduce_pull_two_shot_add_attn_res_rms_norm.cuh"
#include "kernels/all_reduce_pull_two_shot_add_rms_norm.cuh"
#include "kernels/all_reduce_push_two_shot_add_attn_res_rms_norm.cuh"
#include "kernels/all_reduce_push_two_shot_add_rms_norm.cuh"

namespace hip_custom_comms {

// =================================================================================================
// THE CATALOG: each template, and the BuildConfigs it is built for.
// =================================================================================================

// A TEMPLATE'S BUILDS, each of its op's BuildConfig (types.cuh): what dispatch instantiates, what
// the check accepts and what a tuner searches. The launch (the grid, reduce_scatter_blocks) is not
// a build's: each tuned kernel (below) gives its own, and a forced call gives its own. tile_n
// counts the elements of a 16-bit dtype, the only ones built. A build that spills shows in its code
// object (Handle::resources_of) and loses on the clock; none is ruled out by policy. The sweeps
// cited beside a list set its builds and the grids its tuned kernels carry.

// The norms: one-shot, the push two-shot (a column split), the pull two-shot (a row split).
// Not swept: select cuts it to the rows, so it matters only past 16 rows.
// The plain one-shot: a tile a row high, a group a thread wide, at each block size a launch may
// take. The grid is the size's (kAllReduceKernels, which picks among these).
constexpr BuildConfig kAllReduceOneShotConfigs[] = {
    AllReduceBuildConfig{64, 1, 1, 512},
    AllReduceBuildConfig{128, 1, 1, 1024},
    AllReduceBuildConfig{256, 1, 1, 2048},
    AllReduceBuildConfig{512, 1, 1, 4096},
};
// The plain two-shot: a group a thread of a row of threads a rank, at eight ranks (512, a wave per
// peer). The grid is the size's (kAllReduceKernels).
// 512 threads, 64 a rank's row: a row is whole waves (the kernel's static_assert).
constexpr BuildConfig kAllReduceTwoShotConfigs[] = {
    AllReduceBuildConfig{512, 1, 1, 512},
};
constexpr BuildConfig kRmsNormOneShotConfigs[] = {
    AllReduceRmsNormBuildConfig{512, 1, 1, 4096},
    AllReduceRmsNormBuildConfig{512, 1, 1, 8192},
    AllReduceRmsNormBuildConfig{512, 1, 1, 16384},
};
// 256 the best of 48-256 at 192-256 tokens (15.17 and 18.04 against 15.55 and 18.11 at 128;
// 2026-10-01T03-26-44Z); a row a block below that.
constexpr BuildConfig kRmsNormPushConfigs[] = {
    AllReduceRmsNormBuildConfig{512, 1, 1, 4096},
    AllReduceRmsNormBuildConfig{512, 1, 1, 8192},
    AllReduceRmsNormBuildConfig{512, 1, 1, 16384},
};
// Pipelined, 48 the best of 36-96 at 2048-4096 tokens (79.3 and 145.4 against 81.0 and 147.6 at
// 36), within 0.8 of 36 below (2026-10-01T02-59-52Z).
constexpr BuildConfig kRmsNormPullConfigs[] = {
    AllReduceRmsNormBuildConfig{512, 1, 1, 4096},
    AllReduceRmsNormBuildConfig{512, 1, 1, 8192},
};
// Not swept: rms_norm's.
constexpr BuildConfig kAddRmsNormOneShotConfigs[] = {
    AllReduceAddRmsNormBuildConfig{512, 1, 1, 4096},
    AllReduceAddRmsNormBuildConfig{512, 1, 1, 8192},
    AllReduceAddRmsNormBuildConfig{512, 1, 1, 16384},
};
// 256 the best of 48-256 at 192 tokens (15.48 against 16.75 at 128; 2026-10-01T03-26-44Z).
constexpr BuildConfig kAddRmsNormPushConfigs[] = {
    AllReduceAddRmsNormBuildConfig{512, 1, 1, 4096},
    AllReduceAddRmsNormBuildConfig{512, 1, 1, 8192},
    AllReduceAddRmsNormBuildConfig{512, 1, 1, 16384},
};
// 48 the best of 36-96 at every size from 512 to 4096 tokens (80.6 and 149.7 us at 2048 and 4096
// against 84.3 and 154.7 at 36; 2026-10-01T02-59-52Z).
constexpr BuildConfig kAddRmsNormPullConfigs[] = {
    AllReduceAddRmsNormBuildConfig{512, 1, 1, 4096},
    AllReduceAddRmsNormBuildConfig{512, 1, 1, 8192},
};

// AttnRes, one source a step (TILE_K): at 4 (Triton's tile) the row got slower, 6.48 -> 8.16 us a
// row and 147.5 -> 171.0 at 4096 tokens, with 100 -> 166 VGPRs (stamps 2026-10-01T06-26-55Z
// against 04-15-48Z). The one-shot's grid not swept: the norms'.
constexpr BuildConfig kAttnResOneShotConfigs[] = {
    AllReduceAddAttnResRmsNormBuildConfig{512, 1, 1, 4096, 1},
    AllReduceAddAttnResRmsNormBuildConfig{512, 1, 1, 8192, 1},
};
// 256 the best of 32-256 at 256-1024 tokens (34.58, 70.78, 143.98 against 39.93, 83.15, 159.33 at
// 128), a row a block below that (2026-10-01T03-57-23Z).
// 256 threads a block: Triton's program for AttnRes, and more blocks resident at once (AttnRes
// from local memory took 192.8 / 109.6 / 86.0 us on 192 / 384 / 512 blocks, stamps
// 2026-10-02T20-36-56Z); its grid not swept yet.
constexpr BuildConfig kAttnResPushConfigs[] = {
    AllReduceAddAttnResRmsNormBuildConfig{512, 1, 1, 4096, 1},
    AllReduceAddAttnResRmsNormBuildConfig{512, 1, 1, 8192, 1},
    AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 4096, 1},
    AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 8192, 1},
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
constexpr BuildConfig kAttnResPullConfigs[] = {
    AllReduceAddAttnResRmsNormBuildConfig{512, 1, 1, 4096, 1},
    AllReduceAddAttnResRmsNormBuildConfig{512, 1, 1, 8192, 1},
    AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 4096, 1},
    AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 8192, 1},
    AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 4096, 2},
    AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 8192, 2},
    AllReduceAddAttnResRmsNormBuildConfig{256, 1, 2, 4096, 1},
    AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 8192, 3},
    AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 4096, 3},
    AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 4096, 4},
};

// What each template is, in Template's order: its op, its shot, and its builds.
struct TemplateInfo {
  Template fn;
  const char* name;
  OpType op;
  bool two_shot;
  bool push;  // peers push into this rank (else it pulls from them)
  std::span<const BuildConfig> builds;
};

// A TEMPLATE AND ITS NAME FROM ONE TOKEN, so the name cannot differ from the enum's.
#define HIP_CUSTOM_COMMS_NAMED(t) Template::t, #t

constexpr TemplateInfo kTemplates[] = {
    {HIP_CUSTOM_COMMS_NAMED(all_reduce_pull_one_shot), OpType::all_reduce,
     false, false, kAllReduceOneShotConfigs},
    {HIP_CUSTOM_COMMS_NAMED(all_reduce_pull_two_shot), OpType::all_reduce,
     true, false, kAllReduceTwoShotConfigs},
    {HIP_CUSTOM_COMMS_NAMED(all_reduce_pull_one_shot_rms_norm), OpType::all_reduce_rms_norm,
     false, false, kRmsNormOneShotConfigs},
    {HIP_CUSTOM_COMMS_NAMED(all_reduce_pull_two_shot_rms_norm), OpType::all_reduce_rms_norm,
     true, false, kRmsNormPullConfigs},
    {HIP_CUSTOM_COMMS_NAMED(all_reduce_pull_one_shot_add_rms_norm), OpType::all_reduce_add_rms_norm,
     false, false, kAddRmsNormOneShotConfigs},
    {HIP_CUSTOM_COMMS_NAMED(all_reduce_pull_two_shot_add_rms_norm), OpType::all_reduce_add_rms_norm,
     true, false, kAddRmsNormPullConfigs},
    {HIP_CUSTOM_COMMS_NAMED(all_reduce_pull_one_shot_add_attn_res_rms_norm),
     OpType::all_reduce_add_attn_res_rms_norm, false, false, kAttnResOneShotConfigs},
    {HIP_CUSTOM_COMMS_NAMED(all_reduce_pull_two_shot_add_attn_res_rms_norm),
     OpType::all_reduce_add_attn_res_rms_norm, true, false, kAttnResPullConfigs},
    {HIP_CUSTOM_COMMS_NAMED(all_reduce_push_two_shot_rms_norm), OpType::all_reduce_rms_norm,
     true, true, kRmsNormPushConfigs},
    {HIP_CUSTOM_COMMS_NAMED(all_reduce_push_two_shot_add_rms_norm), OpType::all_reduce_add_rms_norm,
     true, true, kAddRmsNormPushConfigs},
    {HIP_CUSTOM_COMMS_NAMED(all_reduce_push_two_shot_add_attn_res_rms_norm),
     OpType::all_reduce_add_attn_res_rms_norm, true, true, kAttnResPushConfigs},
};
#undef HIP_CUSTOM_COMMS_NAMED
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

constexpr std::span<const BuildConfig> builds_of(Template k) { return info(k).builds; }
constexpr bool every_template_built() {
  for (const TemplateInfo& t : kTemplates)
    if (t.builds.empty()) return false;
  return true;
}
static_assert(every_template_built(), "a template lists no builds");

// Whether template `k` builds `b`.
constexpr bool built(Template k, const BuildConfig& b) {
  for (const BuildConfig& listed : builds_of(k))
    if (listed == b) return true;
  return false;
}
// Whether any build of `k` runs at `threads_per_block` (and `waves_per_eu`): the check's names for
// a build that is not listed.
constexpr bool built_at(Template k, int threads_per_block) {
  for (const BuildConfig& listed : builds_of(k))
    if (threads_of(listed) == threads_per_block) return true;
  return false;
}
constexpr bool built_at(Template k, int threads_per_block, int waves_per_eu) {
  for (const BuildConfig& listed : builds_of(k))
    if (threads_of(listed) == threads_per_block && waves_of(listed) == waves_per_eu) return true;
  return false;
}
// THE BUILD TYPE A TEMPLATE'S BUILDS ARE (BuildConfig's index): its op's, every entry the same.
constexpr size_t family_of(Template k) { return builds_of(k)[0].index(); }
constexpr bool builds_one_family() {
  for (const TemplateInfo& t : kTemplates)
    for (const BuildConfig& b : t.builds)
      if (b.index() != family_of(t.fn)) return false;
  return true;
}
static_assert(builds_one_family(), "a template's builds are not all of one family");
// Whether two templates list the same builds (one dispatch serves both).
constexpr bool same_builds(Template a, Template b) {
  if (builds_of(a).size() != builds_of(b).size()) return false;
  for (size_t i = 0; i < builds_of(a).size(); ++i)
    if (builds_of(a)[i] != builds_of(b)[i]) return false;
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
  throw std::runtime_error("hip_custom_comms: " + what + " not built");
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

// A TEMPLATE'S BUILD, compiled in: the one of template K's builds (the catalog above) `b` is, and
// only those. f is handed it as build_constant<B>, B its op's BuildConfig, every field a constant
// expression.
template <Template TEMPLATE>
using family_t = std::variant_alternative_t<family_of(TEMPLATE), BuildConfig>;
template <Template TEMPLATE, size_t BUILD_INDEX>
constexpr family_t<TEMPLATE> listed_build =
    std::get<family_t<TEMPLATE>>(builds_of(TEMPLATE)[BUILD_INDEX]);
template <auto LISTED_BUILD>
struct build_constant {
  static constexpr auto value = LISTED_BUILD;
};

template <Template TEMPLATE, typename VISITOR, size_t... INDICES>
void by_build_in(const BuildConfig& b, VISITOR& f, std::index_sequence<INDICES...>) {
  if (!((b == builds_of(TEMPLATE)[INDICES] &&
         (f(build_constant<listed_build<TEMPLATE, INDICES>>{}), true)) ||
        ...))
    not_built(std::string("template ") + to_string(TEMPLATE) + " at that build");
}

template <Template TEMPLATE, typename VISITOR>
void by_build(const BuildConfig& b, VISITOR&& f) {
  by_build_in<TEMPLATE>(b, f, std::make_index_sequence<builds_of(TEMPLATE).size()>{});
}

[[noreturn]] inline void not_this_ops(Template k) {
  throw std::runtime_error("hip_custom_comms: template " + std::to_string(static_cast<int>(k)) +
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
// A peer table, every rank's address of a buffer: its elements erased, the table kept. The signal
// table is not an element pointer and keeps its type.
template <typename PARAM>
struct erase<PARAM* const*> {
  using t = void* const*;
};
template <typename PARAM>
struct erase<const PARAM* const*> {
  using t = const void* const*;
};
template <>
struct erase<Signal* const*> {
  using t = Signal* const*;
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

// From `m` rows up (to the next entry's), at `world` ranks and rows `n` elements wide: template
// `fn` at `config`, its build and its launch, as measured.
struct TunedKernel {
  int world;
  int64_t m;
  int64_t n;
  Template fn;
  KernelConfig config;
};

// =================================================================================================
// EACH OP'S KERNELS.
// =================================================================================================

// all_reduce, gfx950 on n11, bf16, 8 ranks: the one-shot while a row's tiles are few (through 8
// tokens of 3584, 4 of 7168; swept 2026-10-10T23-22-13Z), then the two-shot. From 64 tokens of 7168
// (128 of 3584) the two-shot's grid is the flat walk's measured best, swept uniformly at 40-256
// blocks (2026-10-12T01-27-40Z, 01-45-53Z; one run a point): 80-120 blocks at mid sizes, 72 from
// 1 MB-ish shares up, where the flat walk beats be0e1f4e9c (7168 x 4096: 262.4 against 263.9 us).
// Equal bytes measure alike (the walk is over the flat range), so the two widths share grids.
constexpr TunedKernel kAllReduceKernels[] = {
    {8, 1, 3584, Template::all_reduce_pull_one_shot,
     {AllReduceBuildConfig{128, 1, 1, 1024}, {32, 0}}},
    {8, 16, 3584, Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {16, 0}}},
    {8, 32, 3584, Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {32, 0}}},
    {8, 64, 3584, Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {64, 0}}},
    {8, 256, 3584, Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {80, 0}}},
    {8, 512, 3584, Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {120, 0}}},
    {8, 1024, 3584, Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {104, 0}}},
    {8, 2048, 3584, Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {72, 0}}},
    {8, 1, 7168, Template::all_reduce_pull_one_shot,
     {AllReduceBuildConfig{128, 1, 1, 1024}, {32, 0}}},
    {8, 8, 7168, Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {16, 0}}},
    {8, 16, 7168, Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {32, 0}}},
    {8, 32, 7168, Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {64, 0}}},
    {8, 64, 7168, Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {64, 0}}},
    {8, 128, 7168, Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {80, 0}}},
    {8, 256, 7168, Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {120, 0}}},
    {8, 512, 7168, Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {104, 0}}},
    {8, 1024, 7168, Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {72, 0}}},
};

// THE PLAIN ALL-REDUCE'S BUILDS THAT LAY OUT WHOLE at `world` ranks and `nl` elements a pack, which
// its dispatch instantiates and nothing else: the one-shot's TILE_M rows a THREADS / TILE_M-thread
// row each across TILE_N; the two-shot's [R, 1, TILE_N] chunk, a rank's THREADS / world threads a
// group each across it (collectives.cuh's flat walk). Every tuned entry must lay out, or select
// would refuse it at run time (tile_not_built): checked here, at compile time.
constexpr bool one_shot_lays_out(const AllReduceBuildConfig& c, int nl) {
  return c.threads_per_block % c.tile_m == 0 &&
         c.tile_n % (nl * (c.threads_per_block / c.tile_m)) == 0;
}
constexpr bool two_shot_lays_out(const AllReduceBuildConfig& c, int world, int nl) {
  return c.tile_m == 1 && c.threads_per_block % world == 0 &&
         c.tile_n % (nl * (c.threads_per_block / world)) == 0;
}
constexpr bool all_reduce_kernels_lay_out() {
  const int nl = kBuild.memory.pack_bytes / 2;  // the tuned dtypes', bf16 and fp16
  for (const TunedKernel& k : kAllReduceKernels) {
    const auto& c = std::get<AllReduceBuildConfig>(k.config.build);
    if (k.fn == Template::all_reduce_pull_one_shot && !one_shot_lays_out(c, nl)) return false;
    if (k.fn == Template::all_reduce_pull_two_shot && !two_shot_lays_out(c, k.world, nl))
      return false;
  }
  return true;
}
static_assert(all_reduce_kernels_lay_out(), "a tuned all-reduce build its dispatch never instantiates");

// gfx950 on n11, bf16, 8 ranks. Transcribed from the size thresholds the sweeps set (each
// crossover cites its run) at the widths we run, until the tuner writes them.
// rms_norm: one-shot through 128 KiB (at 64 KiB it lost at 16 tokens, 11.43 against 10.56 us;
// 2026-09-30T21-06-57Z); the push through 1.75 MiB (256 tokens of 3584: 18.04 against the
// pull's 18.41), the pull from 2.6 MiB (2026-10-01T03-26-44Z).
constexpr TunedKernel kRmsNormKernels[] = {
    {8, 1, 3584, Template::all_reduce_pull_one_shot_rms_norm,
     {AllReduceRmsNormBuildConfig{512, 1, 1, 4096}, {16, 0}}},
    {8, 19, 3584, Template::all_reduce_push_two_shot_rms_norm,
     {AllReduceRmsNormBuildConfig{512, 1, 1, 4096}, {256, 0}}},
    {8, 257, 3584, Template::all_reduce_pull_two_shot_rms_norm,
     {AllReduceRmsNormBuildConfig{512, 1, 1, 4096}, {48, 0}}},
    {8, 1, 7168, Template::all_reduce_pull_one_shot_rms_norm,
     {AllReduceRmsNormBuildConfig{512, 1, 1, 8192}, {16, 0}}},
    {8, 10, 7168, Template::all_reduce_push_two_shot_rms_norm,
     {AllReduceRmsNormBuildConfig{512, 1, 1, 8192}, {256, 0}}},
    {8, 129, 7168, Template::all_reduce_pull_two_shot_rms_norm,
     {AllReduceRmsNormBuildConfig{512, 1, 1, 8192}, {48, 0}}},
    // PAST THE MODEL'S WIDTHS, to the widest build: the one-shot, its grid the 8192's; not swept.
    {8, 1, 16384, Template::all_reduce_pull_one_shot_rms_norm,
     {AllReduceRmsNormBuildConfig{512, 1, 1, 16384}, {16, 0}}},
};

// add_rms_norm: the one-shot not swept (rms_norm's); the push through 1.31 MiB (192 tokens of
// 3584: 15.48 against the pull's 16.28), the pull at 1.75 MiB (18.36 against the push's 18.46;
// 2026-10-01T03-26-44Z).
constexpr TunedKernel kAddRmsNormKernels[] = {
    {8, 1, 3584, Template::all_reduce_pull_one_shot_add_rms_norm,
     {AllReduceAddRmsNormBuildConfig{512, 1, 1, 4096}, {16, 0}}},
    {8, 19, 3584, Template::all_reduce_push_two_shot_add_rms_norm,
     {AllReduceAddRmsNormBuildConfig{512, 1, 1, 4096}, {256, 0}}},
    {8, 193, 3584, Template::all_reduce_pull_two_shot_add_rms_norm,
     {AllReduceAddRmsNormBuildConfig{512, 1, 1, 4096}, {48, 0}}},
    {8, 1, 7168, Template::all_reduce_pull_one_shot_add_rms_norm,
     {AllReduceAddRmsNormBuildConfig{512, 1, 1, 8192}, {16, 0}}},
    {8, 10, 7168, Template::all_reduce_push_two_shot_add_rms_norm,
     {AllReduceAddRmsNormBuildConfig{512, 1, 1, 8192}, {256, 0}}},
    {8, 97, 7168, Template::all_reduce_pull_two_shot_add_rms_norm,
     {AllReduceAddRmsNormBuildConfig{512, 1, 1, 8192}, {48, 0}}},
    // Past the model's widths, as rms_norm's.
    {8, 1, 16384, Template::all_reduce_pull_one_shot_add_rms_norm,
     {AllReduceAddRmsNormBuildConfig{512, 1, 1, 16384}, {16, 0}}},
};

// AttnRes: the push from 1 token (it beat the one-shot, 12.89 against 13.68 us; 14.03 against
// 15.18 at 8; 2026-10-01T03-57-23Z); the pull past 256 tokens at 3584 and 192 at 7168, with
// balanced slices (2026-10-05T03-40-33Z: 3584 x 384 27.7 against the push's 30.4; 7168 x 192
// 27.3 against 27.7).
// THE ONE-SHOT BESIDE THE PUSH from 1 token: listed after it, so it runs only where the push does
// not (a row narrower than a pack a rank); its grid the norms' one-shot's, not swept.
constexpr TunedKernel kAttnResKernels[] = {
    {8, 1, 3584, Template::all_reduce_push_two_shot_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{512, 1, 1, 4096, 1}, {256, 0}}},
    {8, 1, 3584, Template::all_reduce_pull_one_shot_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{512, 1, 1, 4096, 1}, {16, 0}}},
    {8, 257, 3584, Template::all_reduce_pull_two_shot_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 4096, 3}, {256, 0}}},
    {8, 1, 7168, Template::all_reduce_push_two_shot_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{512, 1, 1, 8192, 1}, {256, 0}}},
    {8, 1, 7168, Template::all_reduce_pull_one_shot_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{512, 1, 1, 8192, 1}, {16, 0}}},
    {8, 193, 7168, Template::all_reduce_pull_two_shot_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 8192, 3}, {256, 0}}},
    {8, 512, 7168, Template::all_reduce_pull_two_shot_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 8192, 1}, {256, 0}}},
    {8, 1024, 7168, Template::all_reduce_pull_two_shot_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 8192, 1}, {192, 0}}},
};

// =================================================================================================
// THE OPS, in OpType's order: each one's name (as Python calls it) and its tuned kernels.
// =================================================================================================

struct Op {
  OpType type;
  const char* name;
  std::span<const TunedKernel> kernels;
};

// AN OP AND ITS NAME FROM ONE TOKEN, so the name cannot differ from the enum's.
#define HIP_CUSTOM_COMMS_NAMED(o) OpType::o, #o

constexpr Op kOps[] = {
    {HIP_CUSTOM_COMMS_NAMED(all_reduce), kAllReduceKernels},
    {HIP_CUSTOM_COMMS_NAMED(all_reduce_rms_norm), kRmsNormKernels},
    {HIP_CUSTOM_COMMS_NAMED(all_reduce_add_rms_norm), kAddRmsNormKernels},
    {HIP_CUSTOM_COMMS_NAMED(all_reduce_add_attn_res_rms_norm), kAttnResKernels},
};
#undef HIP_CUSTOM_COMMS_NAMED
constexpr int kNumOps = sizeof(kOps) / sizeof(Op);

constexpr const Op& op(OpType t) { return kOps[static_cast<int>(t)]; }
constexpr const char* to_string(OpType t) { return op(t).name; }

constexpr bool ops_in_order() {
  for (int i = 0; i < kNumOps; ++i)
    if (static_cast<int>(kOps[i].type) != i) return false;
  return true;
}
static_assert(ops_in_order(), "kOps must list every OpType in its order");

// EVERY OP HAS KERNELS, each one of its own templates, its build listed and its launch one the
// forcing would accept (a grid of 1 to kMaxBlocks blocks, a split only on the AttnRes pull).
constexpr bool launch_fits(Template t, const LaunchConfig& l) {
  if (l.blocks_per_grid < 1 || l.blocks_per_grid > kMaxBlocks) return false;
  if (l.reduce_scatter_blocks < 0) return false;
  return l.reduce_scatter_blocks == 0 ||
         t == Template::all_reduce_pull_two_shot_add_attn_res_rms_norm;
}
constexpr bool ops_have_kernels() {
  for (const Op& o : kOps) {
    if (o.kernels.empty()) return false;
    for (const TunedKernel& k : o.kernels)
      if (op_of(k.fn) != o.type || !built(k.fn, k.config.build) ||
          !launch_fits(k.fn, k.config.launch))
        return false;
  }
  return true;
}
static_assert(ops_have_kernels(),
              "an op has no kernels, or one is another op's, not built or not launchable");

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

// A COLUMN TWO-SHOT reduce-scatters by columns, so every block keeps every row; a row two-shot
// gives each rank its rows.
constexpr bool slices_columns(Template k) {
  return k == Template::all_reduce_pull_two_shot_add_attn_res_rms_norm ||
         k == Template::all_reduce_push_two_shot_rms_norm ||
         k == Template::all_reduce_push_two_shot_add_rms_norm ||
         k == Template::all_reduce_push_two_shot_add_attn_res_rms_norm;
}

// A TEMPLATE THAT CUTS EVERY ROW INTO A COLUMN SLICE A RANK: the plain two-shot and the column
// splits. It needs a 16-byte pack for every rank; with fewer, a rank's slice is empty.
constexpr bool slices_a_row(Template k) {
  return k == Template::all_reduce_pull_two_shot || slices_columns(k);
}

// WHETHER EVERY RANK GETS COLUMNS: a template that slices a row needs a load's worth (16 bytes) of
// the row's `n` for each of the `world` ranks; any other template takes any row.
constexpr bool every_rank_gets_columns(Template k, int64_t n, DType dtype, int world) {
  return !slices_a_row(k) || n * elem_bytes(dtype) >= int64_t{world} * kBuild.memory.pack_bytes;
}

// =================================================================================================
// 1. THE FORCING, ALL OR NOTHING: the template (its algorithm and direction, where the op has more
// than one), every field of its build, and its launch, or none of them. Some but not all is
// forced_incomplete; a value out of range is its Error; a build the template does not list is the
// check's to name. None: select picks.
// =================================================================================================

// The kernel a call runs: its template and its complete config.
struct Chosen {
  Template fn;
  KernelConfig config;
};

// `build_fields` are the op's BuildConfig's fields as the call forced them, in its order;
// `reduce_scatter_blocks` the AttnRes split, which only that op takes (another passes none);
// `build_of` makes the op's BuildConfig from the fields, called only when every one is given.
template <typename BUILD_OF>
constexpr std::variant<std::optional<Chosen>, Error> forced(
    OpType o, std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::initializer_list<std::optional<int>> build_fields, std::optional<int> blocks_per_grid,
    std::optional<int> reduce_scatter_blocks, BUILD_OF build_of) {
  const bool names_template = templates_of(o) > 1;
  const bool splits         = o == OpType::all_reduce_add_attn_res_rms_norm;
  int given = 0, needed = 0;
  const auto count = [&](bool is_given) {
    ++needed;
    given += is_given;
  };
  if (names_template) {
    count(algorithm.has_value());
    count(direction.has_value());
  }
  for (const std::optional<int>& f : build_fields) count(f.has_value());
  count(blocks_per_grid.has_value());
  if (splits) count(reduce_scatter_blocks.has_value());
  if (given == 0 && !algorithm && !direction) return std::optional<Chosen>{};
  if (given != needed) return Error::forced_incomplete;
  // THE TEMPLATE: the one an op with one template has, else the algorithm and direction's.
  const std::optional<Template> fn =
      names_template ? template_of(o, *algorithm, *direction) : std::optional{only_template(o)};
  if (!fn || (algorithm && algorithm_of(*fn) != *algorithm) ||
      (direction && direction_of(*fn) != *direction))
    return Error::no_such_template;
  // THE RANGES: whole waves up to the build's maximum threads, a grid of 1 to kMaxBlocks blocks,
  // waves a EU the CU holds, positive tile fields, a split only on the AttnRes pull.
  const std::span<const std::optional<int>> f(build_fields.begin(), build_fields.size());
  const int threads = *f[0], waves = *f[1];
  if (threads < kWaveSize || threads > kBuild.kernels.max_threads || threads % kWaveSize != 0 ||
      *blocks_per_grid < 1 || *blocks_per_grid > kMaxBlocks || waves < 1 ||
      waves > kTarget.max_waves_per_cu / kTarget.simds_per_cu)
    return Error::launch_out_of_range;
  for (size_t i = 2; i < f.size(); ++i)
    if (*f[i] < 1) return Error::field_not_positive;
  const int split = splits ? *reduce_scatter_blocks : 0;
  if (split < 0) return Error::field_not_positive;
  if (split > 0 && *fn != Template::all_reduce_pull_two_shot_add_attn_res_rms_norm)
    return Error::field_not_this_templates;
  return std::optional<Chosen>{Chosen{*fn, {build_of(), {*blocks_per_grid, split}}}};
}

// =================================================================================================
// 2. THE PICK: an op's tuned kernel for the call, exactly as listed, or the Error that no listed
// kernel runs it.
// =================================================================================================

constexpr int64_t distance(int64_t a, int64_t b) { return a < b ? b - a : a - b; }

// Among the op's kernels at the call's world (else the nearest tuned world), THOSE THAT RUN THE
// CALL: a kernel that slices a row needs a pack for every rank (every_rank_gets_columns), and a
// tile that must hold the row (`tile_cols`, the op's) holds it. Among those, the nearest tuned
// width, reading the call's rows as that width's rows of the same bytes (a crossover is about
// bytes); then the entry with the most rows the call reaches, the first listed where two tie, the
// fewest rows where it reaches none. None runs: a row too wide for every tuned tile, or too narrow
// for every slicing kernel.
constexpr std::variant<TunedKernel, Error> pick(OpType o, int world, int64_t m, int64_t n,
                                                int64_t tile_cols, DType dtype) {
  const std::span<const TunedKernel> kernels = op(o).kernels;
  int w = 0;
  for (const TunedKernel& k : kernels)
    if (w == 0 || distance(k.world, world) < distance(w, world)) w = k.world;
  const auto columns = [&](const TunedKernel& k) {
    return every_rank_gets_columns(k.fn, n, dtype, world);
  };
  const auto runs = [&](const TunedKernel& k) {
    return k.world == w && columns(k) && tile_n_of(k.config.build) >= tile_cols;
  };
  int64_t h = 0;
  for (const TunedKernel& k : kernels)
    if (runs(k) && (h == 0 || distance(k.n, n) < distance(h, n))) h = k.n;
  if (h == 0) {
    for (const TunedKernel& k : kernels)
      if (k.world == w && columns(k)) return Error::row_too_wide;
    return Error::row_narrower_than_world;
  }
  const int64_t as_rows   = m * n / h;
  const TunedKernel* most  = nullptr;
  const TunedKernel* least = nullptr;
  for (const TunedKernel& k : kernels) {
    if (!runs(k) || k.n != h) continue;
    if (!least || k.m < least->m) least = &k;
    if (k.m <= as_rows && (!most || k.m > most->m)) most = &k;
  }
  return most ? *most : *least;
}

// THE KERNEL A CALL RUNS: the forced one, else the tuned one.
inline std::variant<Chosen, Error> chosen(const std::variant<std::optional<Chosen>, Error>& forcing,
                                          OpType o, int world, int64_t m, int64_t n,
                                          int64_t tile_cols, DType dtype) {
  if (const Error* e = std::get_if<Error>(&forcing)) return *e;
  if (const std::optional<Chosen>& f = std::get<std::optional<Chosen>>(forcing)) return *f;
  const std::variant<TunedKernel, Error> tuned = pick(o, world, m, n, tile_cols, dtype);
  if (const Error* e = std::get_if<Error>(&tuned)) return *e;
  const TunedKernel& k = std::get<TunedKernel>(tuned);
  return Chosen{k.fn, k.config};
}

// =================================================================================================
// 3. THE GRID CUT TO THE WORK: a rule on the launch, never a field filled in.
// =================================================================================================

// THE TILES THERE ARE, one block's work each: this rank's rows (a row two-shot's share, every row
// otherwise) in TILE_M, by the grid's `grid_cols` columns in TILE_N. A grid wider is idle blocks,
// each still paying every barrier (the pull norm at 32 tokens ran 36 blocks for 4 rows a rank).
constexpr int64_t tiles_of(Template t, const KernelConfig& c, int64_t m, int64_t grid_cols,
                           int world) {
  const bool row_split = is_two_shot(t) && !slices_columns(t);
  const int64_t mine   = row_split ? (m + world - 1) / world : m;
  const int64_t tm = tile_m_of(c.build), tn = tile_n_of(c.build);
  return (mine + tm - 1) / tm * ((grid_cols + tn - 1) / tn);
}

// A TILED OP'S GRID, at most the tiles there are (at least one block). THE FUSED ATTNRES PULL'S
// GRID ALSO DIVIDES ITS TILES: each block's most tiles kept, the blocks cut until every block but
// the last has that many (384 rows on 256 blocks left half the blocks one row: 32.2 against 27.7
// us on 192, 2026-10-05T03-45-51Z). The other templates' not yet measured. The plain all-reduce
// runs its grid as given: its tuned grids were measured as they run.
constexpr Chosen cut_to_the_work(Chosen k, int64_t m, int64_t grid_cols, int world) {
  const int64_t tiles = tiles_of(k.fn, k.config, m, grid_cols, world);
  int& blocks         = k.config.launch.blocks_per_grid;
  if (tiles < blocks) blocks = static_cast<int>(tiles > 0 ? tiles : 1);
  if (k.fn == Template::all_reduce_pull_two_shot_add_attn_res_rms_norm && tiles > 0) {
    const int64_t most = (tiles + blocks - 1) / blocks;
    blocks = static_cast<int>((tiles + most - 1) / most);
  }
  return k;
}

// =================================================================================================
// 4. THE CHECK: the first Error kernel `fn` at `c` meets on the call here, in this order, or none.
// Every Error is a capability (a kernel that cannot take the input), never "the unfused ops would
// be faster". Residency, which needs the compiled kernel, is each select's last check.
// =================================================================================================

// A KERNEL'S SCRATCH, row-major (a staged build needs none: it runs in passes of what the
// scratch holds): the plain two-shot its slice of every row; a column two-shot the whole reduced tensor
// (each rank's columns at their place); a row two-shot its rank's rows, twice where it leaves two
// results (out and the residual). A one-shot reads the inputs and keeps nothing.
constexpr int64_t scratch_need(Template t, int64_t m, int64_t packs, int world) {
  if (!is_two_shot(t)) return 0;
  const OpType o = op_of(t);
  if (o == OpType::all_reduce) return m * ((packs + world - 1) / world) * kBuild.memory.pack_bytes;
  if (slices_columns(t)) return m * packs * kBuild.memory.pack_bytes;
  const bool two = o == OpType::all_reduce_add_rms_norm;
  return (m + world - 1) / world * packs * (two ? 2 : 1) * kBuild.memory.pack_bytes;
}

// A LAYOUT THE PACKED LOADS CAN READ: every buffer's innermost stride 1 and its rows starting on
// whole packs ({stride_m, stride_n} a buffer, in elements of `dtype`; a vector's stride_m 0).
inline std::optional<Error> layout_refused(
    DType dtype, std::initializer_list<std::pair<int64_t, int64_t>> strides) {
  for (const auto& [m, n] : strides) {
    if (n != 1) return Error::inner_stride_not_one;
    if (m * elem_bytes(dtype) % kBuild.memory.pack_bytes != 0) return Error::row_stride_not_packs;
  }
  return std::nullopt;
}

// `row_elems`: the row the kernel reduces, in elements; `own`: the op's own Error on its own
// arguments, checked after the dtype. With no Handle the checks that need
// peers are left out.
inline std::optional<Error> refused(const Handle* h, Template fn, const KernelConfig& c,
                                    DType dtype, int64_t m, int64_t n,
                                    int64_t tile_cols, std::optional<Error> own_error,
                                    const void* inp, bool staged, hipStream_t stream) {
  const int world = h ? h->world_size() : 1;
  if (h && !world_built(world)) return Error::world_not_built;
  if (!dtype_built(dtype)) return Error::dtype_not_built;
  if (own_error) return own_error;
  if (n * elem_bytes(dtype) % kBuild.memory.pack_bytes != 0) return Error::row_not_packs;
  // A BUILD THE TEMPLATE DOES NOT LIST, named by what differs: its threads, its waves, its tile.
  const int threads = threads_of(c.build);
  if (c.build.index() != family_of(fn)) return Error::tile_not_built;
  if (!built_at(fn, threads)) return Error::threads_not_built;
  if (!built_at(fn, threads, waves_of(c.build))) return Error::waves_not_built;
  if (!built(fn, c.build)) return Error::tile_not_built;
  if (tile_n_of(c.build) < tile_cols) return Error::row_too_wide;
  if (!h) return std::nullopt;
  const int64_t packs = packs_of(n, dtype);
  // A TWO-SHOT ROW CUT INTO A SLICE A RANK needs a pack for every rank: with fewer, the ranks past
  // the row own empty slices, which the kernels do not handle (a GPU memory fault, 2026-10-10, at
  // 8 bf16 columns on 8 ranks). Refused, so the caller takes the one-shot or RCCL.
  if (!every_rank_gets_columns(fn, n, dtype, world)) return Error::row_narrower_than_world;
  if (!staged && scratch_need(fn, m, packs, world) > h->scratch_bytes())
    return Error::scratch_too_small;
  // AN IN-PLACE BUILD ON AN EAGER INPUT reads it through the staging, copied in whole first.
  const int64_t bytes = m * n * elem_bytes(dtype);
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
// template and config, else the op's tuned kernel, its grid cut to the work), 2. the checks, 3. its
// compiled instance, 4. the launch.
// =================================================================================================

// all_reduce: the sum of every rank's `inp` ([rows, cols] of `dtype` at its strides) into `out`.
// In place when the peers can read the input where it is (registered, or captured on this stream),
// otherwise its staged kernel, a band of rows a pass.
inline std::variant<AllReduceLaunch, Error> select_all_reduce(
    const Handle& h, void* out, int64_t out_stride_m, int64_t out_stride_n, const void* inp,
    int64_t inp_stride_m, int64_t inp_stride_n, int64_t m, int64_t n, DType dtype,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_m, std::optional<int> tile_n, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, std::optional<int> waves_per_eu, hipStream_t stream) {
  const int world = h.world_size();
  // 1. The tile does not hold the row: the all-reduce tiles across it.
  const std::variant<Chosen, Error> picked = chosen(
      forced(OpType::all_reduce, algorithm, direction,
             {threads_per_block, waves_per_eu, tile_m, tile_n}, blocks_per_grid, std::nullopt,
             [&] {
               return AllReduceBuildConfig{*threads_per_block, *waves_per_eu, *tile_m, *tile_n};
             }),
      OpType::all_reduce, world, m, n, 0, dtype);
  if (const Error* e = std::get_if<Error>(&picked)) return *e;
  const Template fn        = std::get<Chosen>(picked).fn;
  const KernelConfig c     = std::get<Chosen>(picked).config;
  const AllReduceBuildConfig& a = std::get<AllReduceBuildConfig>(c.build);
  const bool staged = !h.reads_in_place(inp, stream);
  // THE STAGED BAND: the rows a pass, what the staging holds (and the two-shot's scratch, a row of
  // its slice a row).
  const int64_t row_bytes   = n * elem_bytes(dtype);
  const int64_t packs       = n * elem_bytes(dtype) / kBuild.memory.pack_bytes;
  const int64_t slice_bytes = (packs + world - 1) / world * kBuild.memory.pack_bytes;
  int64_t band              = row_bytes > 0 ? h.staging_bytes() / row_bytes : 0;
  if (is_two_shot(fn) && slice_bytes > 0 && h.scratch_bytes() / slice_bytes < band)
    band = h.scratch_bytes() / slice_bytes;
  // 2.
  if (const std::optional<Error> e = refused(&h, fn, c, dtype, m, n, 0, std::nullopt, inp,
                                             staged, stream))
    return *e;
  if (staged && band < 1) return Error::staging_too_small;
  // 3. Only a tile its threads lay out whole at this world: TILE_N whole groups for every thread of
  //    a tile row (the two-shot's rows are the ranks).
  const void* kernel = nullptr;
  by_world(world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    by_dtype(dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      constexpr int NL = traits<T>::N;
      switch (fn) {
        case Template::all_reduce_pull_one_shot:
          by_build<Template::all_reduce_pull_one_shot>(c.build, [&](auto built) {
            constexpr AllReduceBuildConfig C = decltype(built)::value;
            constexpr int THREADS = C.threads_per_block, WAVES = C.waves_per_eu;
            constexpr int TM = C.tile_m, TN = C.tile_n;
            if constexpr (one_shot_lays_out(C, NL))
              kernel = staged ? instance<AllReduceOneShotStagedKernel>(
                                    all_reduce_pull_one_shot_staged<T, NG, TM, TN, THREADS, WAVES>)
                              : instance<AllReduceOneShotKernel>(
                                    all_reduce_pull_one_shot<T, NG, TM, TN, THREADS, WAVES>);
          });
          return;
        case Template::all_reduce_pull_two_shot:
          by_build<Template::all_reduce_pull_two_shot>(c.build, [&](auto built) {
            constexpr AllReduceBuildConfig C = decltype(built)::value;
            constexpr int THREADS = C.threads_per_block, WAVES = C.waves_per_eu;
            constexpr int TM = C.tile_m, TN = C.tile_n;
            // THE INSTANCE FOR THE LAYOUT AS GIVEN: rows back to back read each peer's share as one
            // run, other strides the strided walk (collectives.cuh); the input is not reshaped.
            const bool contiguous = inp_stride_m == n * inp_stride_n && out_stride_m == n * out_stride_n;
            if constexpr (two_shot_lays_out(C, NG, NL)) {
              const auto at = [&](auto layout) {
                constexpr bool CONTIGUOUS = decltype(layout)::value;
                return staged ? instance<AllReduceTwoShotStagedKernel>(
                                    all_reduce_pull_two_shot_staged<T, NG, TM, TN, THREADS, WAVES, CONTIGUOUS>)
                              : instance<AllReduceTwoShotKernel>(
                                    all_reduce_pull_two_shot<T, NG, TM, TN, THREADS, WAVES, CONTIGUOUS>);
              };
              kernel = contiguous ? at(std::true_type{}) : at(std::false_type{});
            }
          });
          return;
        default: not_this_ops(fn);
      }
    });
  });
  if (kernel == nullptr) return Error::tile_not_built;
  // 4.
  const AllReduceLaunch l{.kernel            = kernel,
                          .algorithm         = algorithm_of(fn),
                          .direction         = direction_of(fn),
                          .world             = world,
                          .tile_m            = a.tile_m,
                          .tile_n            = a.tile_n,
                          .threads_per_block = a.threads_per_block,
                          .blocks_per_grid   = c.launch.blocks_per_grid,
                          .waves_per_eu      = a.waves_per_eu,
                          .staged            = staged,
                          .stream            = stream,
                          .out               = out,
                          .out_stride_m      = out_stride_m,
                          .out_stride_n      = out_stride_n,
                          .inp               = inp,
                          .inp_stride_m      = inp_stride_m,
                          .inp_stride_n      = inp_stride_n,
                          .scratch_stride_m  = slice_bytes / elem_bytes(dtype),
                          .scratch_stride_n  = 1,
                          .staging_stride_m  = n,
                          .staging_stride_n  = 1,
                          .band_m         = static_cast<int>(band < m ? band : m),
                          .dtype             = dtype,
                          .inp_size_m              = m,
                          .inp_size_n              = n};
  if (!resident(h, kernel, l.blocks_per_grid, l.threads_per_block))
    return Error::grid_not_resident;
  return l;
}

// all_reduce_rms_norm: out = rms_norm(all_reduce(inp), weight), vLLM's roundings exactly. inp
// [rows, hidden]; the weight `weight_dtype`, dtype or f32.
inline std::variant<AllReduceRmsNormLaunch, Error> select_all_reduce_rms_norm(
    const Handle& h, void* out, int64_t out_stride_m, int64_t out_stride_n, const void* inp,
    int64_t inp_stride_m, int64_t inp_stride_n, const void* weight, int64_t weight_stride_n,
    DType dtype, DType weight_dtype, int64_t m, int64_t n, float eps,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_m, std::optional<int> tile_n, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, std::optional<int> waves_per_eu, hipStream_t stream) {
  using K         = Template;
  const OpType op = OpType::all_reduce_rms_norm;
  const int world = h.world_size();
  // 1.
  const std::variant<Chosen, Error> picked = chosen(
      forced(op, algorithm, direction, {threads_per_block, waves_per_eu, tile_m, tile_n},
             blocks_per_grid, std::nullopt,
             [&] {
               return AllReduceRmsNormBuildConfig{*threads_per_block, *waves_per_eu, *tile_m,
                                                  *tile_n};
             }),
      op, world, m, n, n, dtype);
  if (const Error* e = std::get_if<Error>(&picked)) return *e;
  const Chosen k        = cut_to_the_work(std::get<Chosen>(picked), m, n, world);
  const Template fn     = k.fn;
  const KernelConfig& c = k.config;
  // 2.
  if (const std::optional<Error> e = refused(&h, fn, c, dtype, m, n, n,
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
            return by_build<K::all_reduce_pull_one_shot_rms_norm>(c.build, [&](auto cc) {
              constexpr AllReduceRmsNormBuildConfig C = decltype(cc)::value;
              constexpr int TN = C.tile_n, TPB = C.threads_per_block,
                WPE = C.waves_per_eu;
              kernel = instance<AllReduceRmsNormOneShotKernel>(
                  all_reduce_pull_one_shot_rms_norm<T, W, NG, TN, TPB, WPE>);
            });
          case K::all_reduce_pull_two_shot_rms_norm:
            return by_build<K::all_reduce_pull_two_shot_rms_norm>(c.build, [&](auto cc) {
              constexpr AllReduceRmsNormBuildConfig C = decltype(cc)::value;
              constexpr int TN = C.tile_n, TPB = C.threads_per_block,
                WPE = C.waves_per_eu;
              kernel = instance<AllReduceRmsNormTwoShotKernel>(
                  all_reduce_pull_two_shot_rms_norm<T, W, NG, TN, TPB, WPE>);
            });
          case K::all_reduce_push_two_shot_rms_norm:
            return by_build<K::all_reduce_push_two_shot_rms_norm>(c.build, [&](auto cc) {
              constexpr AllReduceRmsNormBuildConfig C = decltype(cc)::value;
              constexpr int TN = C.tile_n, TPB = C.threads_per_block,
                WPE = C.waves_per_eu;
              kernel = instance<AllReduceRmsNormTwoShotKernel>(
                  all_reduce_push_two_shot_rms_norm<T, W, NG, TN, TPB, WPE>);
            });
          default: not_this_ops(fn);
        }
      });
    });
  });
  // 4.
  const AllReduceRmsNormBuildConfig& r = std::get<AllReduceRmsNormBuildConfig>(c.build);
  const AllReduceRmsNormLaunch l{.kernel            = kernel,
                                 .algorithm         = algorithm_of(fn),
                                 .direction         = direction_of(fn),
                                 .world             = world,
                                 .tile_m            = r.tile_m,
                                 .tile_n            = r.tile_n,
                                 .threads_per_block = r.threads_per_block,
                                 .blocks_per_grid   = c.launch.blocks_per_grid,
                                 .waves_per_eu      = r.waves_per_eu,
                                 .stream            = stream,
                                 .out               = out,
                                 .out_stride_m      = out_stride_m,
                                 .out_stride_n      = out_stride_n,
                                 .inp               = inp,
                                 .inp_stride_m      = inp_stride_m,
                                 .inp_stride_n      = inp_stride_n,
                                 .weight            = weight,
                                 .weight_stride_n   = weight_stride_n,
                                 .scratch_stride_m  = n,
                                 .scratch_stride_n  = 1,
                                 .dtype             = dtype,
                                 .weight_dtype      = weight_dtype,
                                 .inp_size_m              = m,
                                 .inp_size_n            = n,
                                 .eps               = eps};
  if (!resident(h, kernel, l.blocks_per_grid, l.threads_per_block))
    return Error::grid_not_resident;
  return l;
}

// all_reduce_add_rms_norm: out, residual_out = fused_add_rms_norm(all_reduce(inp), residual,
// weight), vLLM's roundings exactly.
inline std::variant<AllReduceAddRmsNormLaunch, Error> select_all_reduce_add_rms_norm(
    const Handle& h, void* out, int64_t out_stride_m, int64_t out_stride_n, void* residual_out,
    int64_t residual_out_stride_m, int64_t residual_out_stride_n, const void* inp,
    int64_t inp_stride_m, int64_t inp_stride_n, const void* residual, int64_t residual_stride_m,
    int64_t residual_stride_n, const void* weight, int64_t weight_stride_n, DType dtype,
    DType weight_dtype, int64_t m, int64_t n, float eps,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_m, std::optional<int> tile_n, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, std::optional<int> waves_per_eu, hipStream_t stream) {
  using K         = Template;
  const OpType op = OpType::all_reduce_add_rms_norm;
  const int world = h.world_size();
  // 1.
  const std::variant<Chosen, Error> picked = chosen(
      forced(op, algorithm, direction, {threads_per_block, waves_per_eu, tile_m, tile_n},
             blocks_per_grid, std::nullopt,
             [&] {
               return AllReduceAddRmsNormBuildConfig{*threads_per_block, *waves_per_eu, *tile_m,
                                                     *tile_n};
             }),
      op, world, m, n, n, dtype);
  if (const Error* e = std::get_if<Error>(&picked)) return *e;
  const Chosen k        = cut_to_the_work(std::get<Chosen>(picked), m, n, world);
  const Template fn     = k.fn;
  const KernelConfig& c = k.config;
  // 2.
  if (const std::optional<Error> e = refused(&h, fn, c, dtype, m, n, n,
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
            return by_build<K::all_reduce_pull_one_shot_add_rms_norm>(c.build, [&](auto cc) {
              constexpr AllReduceAddRmsNormBuildConfig C = decltype(cc)::value;
              constexpr int TN = C.tile_n, TPB = C.threads_per_block,
                WPE = C.waves_per_eu;
              kernel = instance<AllReduceAddRmsNormOneShotKernel>(
                  all_reduce_pull_one_shot_add_rms_norm<T, W, NG, TN, TPB, WPE>);
            });
          case K::all_reduce_pull_two_shot_add_rms_norm:
            return by_build<K::all_reduce_pull_two_shot_add_rms_norm>(c.build, [&](auto cc) {
              constexpr AllReduceAddRmsNormBuildConfig C = decltype(cc)::value;
              constexpr int TN = C.tile_n, TPB = C.threads_per_block,
                WPE = C.waves_per_eu;
              kernel = instance<AllReduceAddRmsNormTwoShotKernel>(
                  all_reduce_pull_two_shot_add_rms_norm<T, W, NG, TN, TPB, WPE>);
            });
          case K::all_reduce_push_two_shot_add_rms_norm:
            return by_build<K::all_reduce_push_two_shot_add_rms_norm>(c.build, [&](auto cc) {
              constexpr AllReduceAddRmsNormBuildConfig C = decltype(cc)::value;
              constexpr int TN = C.tile_n, TPB = C.threads_per_block,
                WPE = C.waves_per_eu;
              kernel = instance<AllReduceAddRmsNormTwoShotKernel>(
                  all_reduce_push_two_shot_add_rms_norm<T, W, NG, TN, TPB, WPE>);
            });
          default: not_this_ops(fn);
        }
      });
    });
  });
  // 4.
  const AllReduceAddRmsNormBuildConfig& r = std::get<AllReduceAddRmsNormBuildConfig>(c.build);
  const AllReduceAddRmsNormLaunch l{.kernel            = kernel,
                                    .algorithm         = algorithm_of(fn),
                                    .direction         = direction_of(fn),
                                    .world             = world,
                                    .tile_m            = r.tile_m,
                                    .tile_n            = r.tile_n,
                                    .threads_per_block = r.threads_per_block,
                                    .blocks_per_grid   = c.launch.blocks_per_grid,
                                    .waves_per_eu      = r.waves_per_eu,
                                    .stream            = stream,
                                    .out               = out,
                                    .out_stride_m      = out_stride_m,
                                    .out_stride_n      = out_stride_n,
                                    .residual_out      = residual_out,
                                    .residual_out_stride_m = residual_out_stride_m,
                                    .residual_out_stride_n = residual_out_stride_n,
                                    .inp               = inp,
                                    .inp_stride_m      = inp_stride_m,
                                    .inp_stride_n      = inp_stride_n,
                                    .residual          = residual,
                                    .residual_stride_m = residual_stride_m,
                                    .residual_stride_n = residual_stride_n,
                                    .weight            = weight,
                                    .weight_stride_n   = weight_stride_n,
                                    .scratch_stride_m  = n,
                                    .scratch_stride_n  = 1,
                                    .dtype             = dtype,
                                    .weight_dtype      = weight_dtype,
                                    .inp_size_m              = m,
                                    .inp_size_n            = n,
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
    const Handle& h, void* prefix, int64_t prefix_stride_m, int64_t prefix_stride_n, void* out,
    int64_t out_stride_m, int64_t out_stride_n, const void* inp, int64_t inp_stride_m,
    int64_t inp_stride_n, void* blocks, int64_t blocks_stride_m, int64_t blocks_stride_r,
    int64_t blocks_stride_n, const void* norm_weight, int64_t norm_weight_stride_n,
    const void* qk_weight, int64_t qk_weight_stride_n, const void* out_norm_weight,
    int64_t out_norm_weight_stride_n,
    DType dtype, int64_t m, int64_t n,
    int num_blocks, int write_idx, float eps, float out_eps, bool has_prefix,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_m, std::optional<int> tile_n, std::optional<int> tile_k,
    std::optional<int> reduce_scatter_blocks, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, std::optional<int> waves_per_eu, hipStream_t stream) {
  using K         = Template;
  const OpType op = OpType::all_reduce_add_attn_res_rms_norm;
  const int world = h.world_size();
  // 1.
  const std::variant<Chosen, Error> picked = chosen(
      forced(op, algorithm, direction, {threads_per_block, waves_per_eu, tile_m, tile_n, tile_k},
             blocks_per_grid, reduce_scatter_blocks,
             [&] {
               return AllReduceAddAttnResRmsNormBuildConfig{*threads_per_block, *waves_per_eu,
                                                            *tile_m, *tile_n, *tile_k};
             }),
      op, world, m, n, n, dtype);
  if (const Error* e = std::get_if<Error>(&picked)) return *e;
  const Chosen k        = cut_to_the_work(std::get<Chosen>(picked), m, n, world);
  const Template fn     = k.fn;
  const KernelConfig& c = k.config;
  // 2.
  if (const std::optional<Error> e =
          refused(&h, fn, c, dtype, m, n, n, std::nullopt, inp, false, stream))
    return *e;
  // 3.
  const void* kernel = nullptr;
  by_world(world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    by_dtype(dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      switch (fn) {
        case K::all_reduce_pull_one_shot_add_attn_res_rms_norm:
          return by_build<K::all_reduce_pull_one_shot_add_attn_res_rms_norm>(c.build, [&](auto cc) {
            constexpr AllReduceAddAttnResRmsNormBuildConfig C = decltype(cc)::value;
            constexpr int TN = C.tile_n, TK = C.tile_k, TPB = C.threads_per_block,
                WPE = C.waves_per_eu;
            kernel = has_prefix
                         ? instance<AllReduceAddAttnResRmsNormOneShotKernel>(
                               all_reduce_pull_one_shot_add_attn_res_rms_norm<T, NG, true, TN,
                                                                              TK, TPB, WPE>)
                         : instance<AllReduceAddAttnResRmsNormOneShotKernel>(
                               all_reduce_pull_one_shot_add_attn_res_rms_norm<T, NG, false, TN,
                                                                              TK, TPB, WPE>);
          });
        case K::all_reduce_pull_two_shot_add_attn_res_rms_norm:
          return by_build<K::all_reduce_pull_two_shot_add_attn_res_rms_norm>(c.build, [&](auto cc) {
            constexpr AllReduceAddAttnResRmsNormBuildConfig C = decltype(cc)::value;
            constexpr int TM = C.tile_m, TN = C.tile_n, TK = C.tile_k,
                          TPB = C.threads_per_block,
                WPE = C.waves_per_eu;
            kernel = has_prefix
                         ? instance<AllReduceAddAttnResRmsNormPullKernel>(
                               all_reduce_pull_two_shot_add_attn_res_rms_norm<T, NG, true, TM, TN,
                                                                              TK, TPB, WPE>)
                         : instance<AllReduceAddAttnResRmsNormPullKernel>(
                               all_reduce_pull_two_shot_add_attn_res_rms_norm<T, NG, false, TM,
                                                                              TN, TK, TPB, WPE>);
          });
        case K::all_reduce_push_two_shot_add_attn_res_rms_norm:
          return by_build<K::all_reduce_push_two_shot_add_attn_res_rms_norm>(c.build, [&](auto cc) {
            constexpr AllReduceAddAttnResRmsNormBuildConfig C = decltype(cc)::value;
            constexpr int TN = C.tile_n, TK = C.tile_k, TPB = C.threads_per_block,
                WPE = C.waves_per_eu;
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
  const AllReduceAddAttnResRmsNormBuildConfig& a =
      std::get<AllReduceAddAttnResRmsNormBuildConfig>(c.build);
  const int tm = a.tile_m, tk = a.tile_k, rs = c.launch.reduce_scatter_blocks;
  const AllReduceAddAttnResRmsNormLaunch l{.kernel                = kernel,
                                           .algorithm             = algorithm_of(fn),
                                           .direction             = direction_of(fn),
                                           .world                 = world,
                                           .tile_m                = tm,
                                           .tile_n                = a.tile_n,
                                           .tile_k                = tk,
                                           .threads_per_block     = a.threads_per_block,
                                           .blocks_per_grid       = c.launch.blocks_per_grid,
                                           .waves_per_eu          = a.waves_per_eu,
                                           .reduce_scatter_blocks = rs,
                                           .stream                = stream,
                                           .prefix                = prefix,
                                           .prefix_stride_m       = prefix_stride_m,
                                           .prefix_stride_n       = prefix_stride_n,
                                           .out                   = out,
                                           .out_stride_m          = out_stride_m,
                                           .out_stride_n          = out_stride_n,
                                           .inp                   = inp,
                                           .inp_stride_m          = inp_stride_m,
                                           .inp_stride_n          = inp_stride_n,
                                           .blocks                = blocks,
                                           .blocks_stride_m       = blocks_stride_m,
                                           .blocks_stride_r       = blocks_stride_r,
                                           .blocks_stride_n       = blocks_stride_n,
                                           .norm_weight           = norm_weight,
                                           .norm_weight_stride_n  = norm_weight_stride_n,
                                           .qk_weight             = qk_weight,
                                           .qk_weight_stride_n    = qk_weight_stride_n,
                                           .out_norm_weight       = out_norm_weight,
                                           .out_norm_weight_stride_n = out_norm_weight_stride_n,
                                           .scratch_stride_m      = n,
                                           .scratch_stride_n      = 1,
                                           .dtype                 = dtype,
                                           .inp_size_m                  = m,
                                           .inp_size_n                = n,
                                           .num_blocks            = num_blocks,
                                           .write_idx             = write_idx,
                                           .eps                   = eps,
                                           .out_eps               = out_eps,
                                           .has_prefix            = has_prefix};
  if (!resident(h, kernel, l.blocks_per_grid, l.threads_per_block))
    return Error::grid_not_resident;
  return l;
}

// =================================================================================================
// EVERY TUNED SELECTION FITS, for every op at the smallest call (8 columns on 2 ranks: narrower
// than any slicing kernel takes) and a large one: an op never declines (a fusion that is on runs
// its fused op), and a kernel past a capability is a compile error, not one that overruns its
// signal slots.
// =================================================================================================

// The tuned kernel for `m` rows of `row` elements, cut to the work: picked, and launchable.
constexpr bool tuned_fits(OpType o, int world, int64_t m, int64_t row, int64_t tile_cols,
                          int64_t grid_cols) {
  const std::variant<TunedKernel, Error> t = pick(o, world, m, row, tile_cols, DType::bf16);
  if (std::holds_alternative<Error>(t)) return false;
  const TunedKernel& k = std::get<TunedKernel>(t);
  const Chosen cut     = cut_to_the_work(Chosen{k.fn, k.config}, m, grid_cols, world);
  return launch_fits(cut.fn, cut.config.launch);
}
constexpr bool selections_fit() {
  for (const std::array<int64_t, 3> call : {std::array<int64_t, 3>{2, 1, 8},
                                            std::array<int64_t, 3>{kMaxRanks, 4096, 7168}}) {
    const int w = static_cast<int>(call[0]);
    const int64_t m = call[1], n = call[2];
    for (const OpType o :
         {OpType::all_reduce_rms_norm, OpType::all_reduce_add_rms_norm,
          OpType::all_reduce_add_attn_res_rms_norm})
      if (!tuned_fits(o, w, m, n, n, n)) return false;
    // The plain all-reduce's tuned kernel, as measured (its grid not cut).
    if (std::holds_alternative<Error>(pick(OpType::all_reduce, w, m, n, 0, DType::bf16)))
      return false;
  }
  return true;
}
static_assert(selections_fit(), "a selection declines, or exceeds a kernel capability");

}  // namespace hip_custom_comms
