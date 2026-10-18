// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SELECT, THE ONLY CHOICE, read by each op's select_<op> (the API,
// hip_custom_communicator.cuh): what was tuned (each op's kernels: for each
// world and row width, the template and complete KernelConfig, its build and
// its launch, that won from a number of rows up, written by the tuner, tune.py,
// from a sweep; an entry says what ran fastest, the why is the sweep's figure,
// cited beside it), then the steps every select takes, each on primitives: the
// forced kernel, given whole, or the tuned kernel picked as listed; the grid
// cut to the tiles there are; and the first Error the kernel meets here.
// Nothing is filled in: a config is complete wherever it is written.

#pragma once

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
#include <vector>

#include "kernels/all_reduce.cuh"
#include "kernels/all_reduce_add_attn_res_rms_norm.cuh"
#include "kernels/all_reduce_add_rms_norm.cuh"
#include "handle.cuh"
#include "kernels/common.cuh"
#include "types.cuh"

namespace hip_custom_communicator {

// WHAT A BUFFER MUST HOLD FOR AN [m, n] INT4 PAYLOAD (kernels/common.cuh's layout): the values,
// then the scales at int4_scales_offset.
constexpr int64_t int4_payload_bytes(int64_t m, int64_t n) {
  return int4_scales_offset(m, n) + m * (n / kInt4Group) * kInt4ScaleBytes;
}

// =================================================================================================
// THE CATALOG: each template, and the BuildConfigs it is built for.
// =================================================================================================

// A TEMPLATE'S BUILDS, each of its op's BuildConfig (types.cuh): what dispatch
// instantiates, what the check accepts and what a tuner searches. The launch
// (the grid, reduce_scatter_blocks) is not a build's: each tuned kernel (below)
// gives its own, and a forced call gives its own. tile_n counts the elements of
// a 16-bit dtype, the only ones built. A build that spills shows in its code
// object (gp report isa) and loses on the clock; none is ruled out by
// policy. The sweeps cited beside a list set its builds and the grids its tuned
// kernels carry.

// A TWO-SHOT TEMPLATE'S BUILDS: each listed build as it is, then the same
// moving INT4 (common.cuh's QUANTIZATION), forced as any build is or taken by
// select for a call the handle quantizes (`quantizes`). A one-shot
// moves its input as it is.
template <size_t N>
constexpr std::array<BuildConfig, 2 * N> and_int4(
    const BuildConfig (&exact)[N]) {
  std::array<BuildConfig, 2 * N> all{};
  for (size_t i = 0; i < N; ++i) {
    all[i] = exact[i];
    all[N + i] = quantized(exact[i], Quantization::int4);
  }
  return all;
}

// The norms: one-shot, the push two-shot (a column split), the pull two-shot (a
// row split). Not swept: select cuts it to the rows, so it matters only past 16
// rows. The plain one-shot: a tile a row high, a group a thread wide, at each
// block size a launch may take. The grid is the size's (kAllReduceKernels,
// which picks among these).
constexpr BuildConfig kAllReduceOneShotConfigs[] = {
    AllReduceBuildConfig{64, 1, 1, 512},
    AllReduceBuildConfig{128, 1, 1, 1024},
    AllReduceBuildConfig{256, 1, 1, 2048},
    AllReduceBuildConfig{512, 1, 1, 4096},
};
// The plain two-shot: a group a thread of a row of threads a rank, at eight
// ranks (512, a wave per peer). The grid is the size's (kAllReduceKernels). 512
// threads, 64 a rank's row: a row is whole waves (the kernel's static_assert).
constexpr BuildConfig kAllReduceTwoShotExactConfigs[] = {
    AllReduceBuildConfig{512, 1, 1, 512},
};
constexpr auto kAllReduceTwoShotConfigs = and_int4(kAllReduceTwoShotExactConfigs);
constexpr BuildConfig kRmsNormOneShotConfigs[] = {
    AllReduceRmsNormBuildConfig{512, 1, 1, 4096},
    AllReduceRmsNormBuildConfig{512, 1, 1, 8192},
    AllReduceRmsNormBuildConfig{512, 1, 1, 16384},
};
// 256 the best of 48-256 at 192-256 tokens (15.17 and 18.04 against 15.55
// and 18.11 at 128; 2026-10-01T03-26-44Z); a row a block below that.
constexpr BuildConfig kRmsNormPushExactConfigs[] = {
    AllReduceRmsNormBuildConfig{512, 1, 1, 4096},
    AllReduceRmsNormBuildConfig{512, 1, 1, 8192},
    AllReduceRmsNormBuildConfig{512, 1, 1, 16384},
};
constexpr auto kRmsNormPushConfigs = and_int4(kRmsNormPushExactConfigs);
// Pipelined, 48 the best of 36-96 at 2048-4096 tokens (79.3 and 145.4
// against 81.0 and 147.6 at 36), within 0.8 of 36 below (2026-10-01T02-59-52Z).
// INT4 A GROUP A THREAD (the kernel's INT4 body): a block one row's groups, TILE_N / 32 threads
// (kRmsNormInt4Kernels).
constexpr BuildConfig kRmsNormPullConfigs[] = {
    AllReduceRmsNormBuildConfig{512, 1, 1, 4096},
    AllReduceRmsNormBuildConfig{512, 1, 1, 8192},
    AllReduceRmsNormBuildConfig{128, 1, 1, 4096, Quantization::int4},
    AllReduceRmsNormBuildConfig{256, 1, 1, 8192, Quantization::int4},
};
// Not swept: rms_norm's.
constexpr BuildConfig kAddRmsNormOneShotConfigs[] = {
    AllReduceAddRmsNormBuildConfig{512, 1, 1, 4096},
    AllReduceAddRmsNormBuildConfig{512, 1, 1, 8192},
    AllReduceAddRmsNormBuildConfig{512, 1, 1, 16384},
};
// 256 the best of 48-256 at 192 tokens (15.48 against 16.75 at 128;
// 2026-10-01T03-26-44Z).
constexpr BuildConfig kAddRmsNormPushExactConfigs[] = {
    AllReduceAddRmsNormBuildConfig{512, 1, 1, 4096},
    AllReduceAddRmsNormBuildConfig{512, 1, 1, 8192},
    AllReduceAddRmsNormBuildConfig{512, 1, 1, 16384},
};
constexpr auto kAddRmsNormPushConfigs = and_int4(kAddRmsNormPushExactConfigs);
// 48 the best of 36-96 at every size from 512 to 4096 tokens (80.6 and 149.7 us
// at 2048 and 4096 against 84.3 and 154.7 at 36; 2026-10-01T02-59-52Z).
// INT4 a group a thread, as rms_norm's.
constexpr BuildConfig kAddRmsNormPullConfigs[] = {
    AllReduceAddRmsNormBuildConfig{512, 1, 1, 4096},
    AllReduceAddRmsNormBuildConfig{512, 1, 1, 8192},
    AllReduceAddRmsNormBuildConfig{128, 1, 1, 4096, Quantization::int4},
    AllReduceAddRmsNormBuildConfig{256, 1, 1, 8192, Quantization::int4},
};

// AttnRes, one source a step (TILE_K): at 4 (Triton's tile) the row got
// slower, 6.48 -> 8.16 us a row and 147.5 -> 171.0 at 4096 tokens, with 100 ->
// 166 VGPRs (stamps 2026-10-01T06-26-55Z against 04-15-48Z). The one-shot's
// grid not swept: the norms'.
constexpr BuildConfig kAttnResOneShotConfigs[] = {
    AllReduceAddAttnResRmsNormBuildConfig{512, 1, 1, 4096, 1},
    AllReduceAddAttnResRmsNormBuildConfig{512, 1, 1, 8192, 1},
};
// 256 the best of 32-256 at 256-1024 tokens (34.58, 70.78, 143.98
// against 39.93, 83.15, 159.33 at 128), a row a block below that
// (2026-10-01T03-57-23Z). 256 threads a block: Triton's program for AttnRes,
// and more blocks resident at once (AttnRes from local memory took 192.8 /
// 109.6 / 86.0 us on 192 / 384 / 512 blocks, stamps 2026-10-02T20-36-56Z); its
// grid not swept yet.
constexpr BuildConfig kAttnResPushExactConfigs[] = {
    AllReduceAddAttnResRmsNormBuildConfig{512, 1, 1, 4096, 1},
    AllReduceAddAttnResRmsNormBuildConfig{512, 1, 1, 8192, 1},
    AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 4096, 1},
    AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 8192, 1},
};
constexpr auto kAttnResPushConfigs = and_int4(kAttnResPushExactConfigs);
// The column split wants a wide grid (AttnRes is compute a row): at 7168, 192
// is within about 5% of the best of 16-256 from 512 to 4096 tokens; 4096 at
// 447.2 us against 1160.7 at the 36 it had (2026-10-01T22-56-58Z,
// 2026-10-01T23-00-47Z). TILE_M 2 and 4 lost at every grid, best 480.0 us at
// 128 blocks against 412.8 at 1 on 192 (4096 x 7168, 2026-10-02T21-19-22Z). Its
// reduce-scatter on the grid's first 32 blocks (reduce_scatter_blocks): its
// reads queue behind the links past a few dozen blocks while AttnRes is compute
// a row and wants every block. At 4096 tokens it took 146.6 us on 32 blocks
// against 218.9 on 192, AttnRes 1110.5 against 199.4 (stamps,
// 2026-10-01T23-45-31Z and 2026-10-01T23-50-54Z).
// 256 threads as the push's, its grid doubled (not swept yet). TILE_M 2 and 4
// are no longer built: they lost at every grid, and 4 spilled 25-30 VGPRs.
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
    // INT4 A GROUP A THREAD (the kernel's INT4 body): a block one row's groups, TILE_N / 32
    // threads, a row a tile (kAttnResInt4Kernels).
    AllReduceAddAttnResRmsNormBuildConfig{128, 1, 1, 4096, 1, Quantization::int4},
    AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 8192, 1, Quantization::int4},
    AllReduceAddAttnResRmsNormBuildConfig{128, 1, 1, 4096, 3, Quantization::int4},
    AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 8192, 3, Quantization::int4},
};

// OUR SEPARATE NORM KERNELS (the norm splits' second, rms_norm_rows and add_rms_norm_rows): a
// row a tile, Layout::tile, a block's threads along the row; each as it is and with its
// all-reduce moving INT4.
constexpr BuildConfig kRmsNormSplitExactConfigs[] = {
    AllReduceRmsNormBuildConfig{256, 1, 1, 4096},
    AllReduceRmsNormBuildConfig{512, 1, 1, 4096},
    AllReduceRmsNormBuildConfig{256, 1, 1, 8192},
    AllReduceRmsNormBuildConfig{512, 1, 1, 8192},
};
constexpr auto kRmsNormSplitConfigs = and_int4(kRmsNormSplitExactConfigs);
constexpr BuildConfig kAddRmsNormSplitExactConfigs[] = {
    AllReduceAddRmsNormBuildConfig{256, 1, 1, 4096},
    AllReduceAddRmsNormBuildConfig{512, 1, 1, 4096},
    AllReduceAddRmsNormBuildConfig{256, 1, 1, 8192},
    AllReduceAddRmsNormBuildConfig{512, 1, 1, 8192},
};
constexpr auto kAddRmsNormSplitConfigs = and_int4(kAddRmsNormSplitExactConfigs);
// OUR SEPARATE ATTNRES KERNEL (the split's second, add_attn_res_rms_norm_rows): a row a tile, the
// fused pull's exact tile (TILE_N a row's packs over the threads, Layout::tile), at Triton's
// AttnRes program (256 threads, a source a step: the reference's prefill config) and the
// neighbours a sweep picks among; each as it is and with its all-reduce moving INT4.
constexpr BuildConfig kAttnResSplitExactConfigs[] = {
    AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 4096, 1},
    AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 4096, 2},
    AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 4096, 3},
    AllReduceAddAttnResRmsNormBuildConfig{512, 1, 1, 4096, 1},
    AllReduceAddAttnResRmsNormBuildConfig{128, 1, 1, 4096, 1},
    AllReduceAddAttnResRmsNormBuildConfig{256, 1, 2, 4096, 1},
    AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 8192, 1},
    AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 8192, 2},
    AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 8192, 3},
    AllReduceAddAttnResRmsNormBuildConfig{512, 1, 1, 8192, 1},
};
constexpr auto kAttnResSplitConfigs = and_int4(kAttnResSplitExactConfigs);

// What each template is, in Template's order: its op, its shot, and its builds.
struct TemplateInfo {
  Template fn;
  const char* name;
  OpType op;
  Algorithm algorithm;
  bool push;  // peers push into this rank (else it pulls from them)
  std::span<const BuildConfig> builds;
};

// A TEMPLATE AND ITS NAME FROM ONE TOKEN, so the name cannot differ from the
// enum's.
#define HIP_CUSTOM_COMMUNICATOR_NAMED(t) Template::t, #t

constexpr TemplateInfo kTemplates[] = {
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_pull_one_shot),
     OpType::all_reduce, Algorithm::one_shot, false, kAllReduceOneShotConfigs},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_pull_two_shot),
     OpType::all_reduce, Algorithm::two_shot, false, kAllReduceTwoShotConfigs},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_pull_one_shot_rms_norm),
     OpType::all_reduce_rms_norm, Algorithm::one_shot, false, kRmsNormOneShotConfigs},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_pull_two_shot_rms_norm),
     OpType::all_reduce_rms_norm, Algorithm::two_shot, false, kRmsNormPullConfigs},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_pull_one_shot_add_rms_norm),
     OpType::all_reduce_add_rms_norm, Algorithm::one_shot, false, kAddRmsNormOneShotConfigs},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_pull_two_shot_add_rms_norm),
     OpType::all_reduce_add_rms_norm, Algorithm::two_shot, false, kAddRmsNormPullConfigs},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(
         all_reduce_pull_one_shot_add_attn_res_rms_norm),
     OpType::all_reduce_add_attn_res_rms_norm, Algorithm::one_shot, false,
     kAttnResOneShotConfigs},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(
         all_reduce_pull_two_shot_add_attn_res_rms_norm),
     OpType::all_reduce_add_attn_res_rms_norm, Algorithm::two_shot, false,
     kAttnResPullConfigs},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_push_two_shot_rms_norm),
     OpType::all_reduce_rms_norm, Algorithm::two_shot, true, kRmsNormPushConfigs},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_push_two_shot_add_rms_norm),
     OpType::all_reduce_add_rms_norm, Algorithm::two_shot, true, kAddRmsNormPushConfigs},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(
         all_reduce_push_two_shot_add_attn_res_rms_norm),
     OpType::all_reduce_add_attn_res_rms_norm, Algorithm::two_shot, true, kAttnResPushConfigs},
    // THE PLAIN ALL-REDUCE PUSHED (kernels/all_reduce.cuh): a variant a caller
    // forces by its direction; no tuned entry names it (the in-place pull is
    // faster: 262.5 against 333 us at 4096 x 7168 bf16, n11, 2026-10-17). The
    // pull's builds. Their INT4 builds run the one INT4 kernel either
    // template's does (all_reduce_pull_two_shot_int4: the shares quantized by
    // their owner and pushed).
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_push_one_shot),
     OpType::all_reduce, Algorithm::one_shot, true, kAllReduceOneShotConfigs},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_push_two_shot),
     OpType::all_reduce, Algorithm::two_shot, true, kAllReduceTwoShotConfigs},
    // THE OP AS TWO KERNELS OF OURS (Algorithm::split; STANDARDS 3.2, *We win at the op level*):
    // the plain all-reduce, its tuned pick at the build's quantization, into this rank's local
    // sums (Handle::sums), then our separate AttnRes kernel on them (add_attn_res_rms_norm_rows), on
    // its own grid at its own occupancy. Its builds are that kernel's.
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_pull_split_add_attn_res_rms_norm),
     OpType::all_reduce_add_attn_res_rms_norm, Algorithm::split, false, kAttnResSplitConfigs},
    // THE NORMS AS TWO KERNELS OF OURS: the plain all-reduce, then our separate norm kernel.
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_pull_split_rms_norm),
     OpType::all_reduce_rms_norm, Algorithm::split, false, kRmsNormSplitConfigs},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_pull_split_add_rms_norm),
     OpType::all_reduce_add_rms_norm, Algorithm::split, false, kAddRmsNormSplitConfigs},
};
#undef HIP_CUSTOM_COMMUNICATOR_NAMED
constexpr int kNumTemplates = sizeof(kTemplates) / sizeof(TemplateInfo);

constexpr bool templates_in_order() {
  for (int i = 0; i < kNumTemplates; ++i)
    if (static_cast<int>(kTemplates[i].fn) != i) return false;
  return true;
}
static_assert(templates_in_order(),
              "kTemplates must list every Template in its order");

constexpr const TemplateInfo& info(Template k) {
  return kTemplates[static_cast<int>(k)];
}
constexpr const char* to_string(Template k) { return info(k).name; }
constexpr bool is_two_shot(Template k) { return info(k).algorithm == Algorithm::two_shot; }
// A SPLIT: two kernels of ours, the plain all-reduce then the op's own (Algorithm::split).
constexpr bool is_split(Template k) { return info(k).algorithm == Algorithm::split; }

constexpr std::span<const BuildConfig> builds_of(Template k) {
  return info(k).builds;
}
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
// Whether any build of `k` runs at `threads_per_block` (and `waves_per_eu`):
// the check's names for a build that is not listed.
constexpr bool built_at(Template k, int threads_per_block) {
  for (const BuildConfig& listed : builds_of(k))
    if (threads_of(listed) == threads_per_block) return true;
  return false;
}
constexpr bool built_at(Template k, int threads_per_block, int waves_per_eu) {
  for (const BuildConfig& listed : builds_of(k))
    if (threads_of(listed) == threads_per_block &&
        waves_of(listed) == waves_per_eu)
      return true;
  return false;
}
// THE BUILD TYPE A TEMPLATE'S BUILDS ARE (BuildConfig's index): its op's, every
// entry the same.
constexpr size_t family_of(Template k) { return builds_of(k)[0].index(); }
constexpr bool builds_one_family() {
  for (const TemplateInfo& t : kTemplates)
    for (const BuildConfig& b : t.builds)
      if (b.index() != family_of(t.fn)) return false;
  return true;
}
static_assert(builds_one_family(),
              "a template's builds are not all of one family");
// Whether two templates list the same builds (one dispatch serves both).
constexpr bool same_builds(Template a, Template b) {
  if (builds_of(a).size() != builds_of(b).size()) return false;
  for (size_t i = 0; i < builds_of(a).size(); ++i)
    if (builds_of(a)[i] != builds_of(b)[i]) return false;
  return true;
}

constexpr OpType op_of(Template k) { return info(k).op; }
constexpr Algorithm algorithm_of(Template k) { return info(k).algorithm; }
constexpr Direction direction_of(Template k) {
  return info(k).push ? Direction::push : Direction::pull;
}
// OP `o`'S TEMPLATE at an algorithm and direction, or none: an op has at most
// one of each.
constexpr std::optional<Template> template_of(OpType o, Algorithm a,
                                              Direction d) {
  for (const TemplateInfo& t : kTemplates)
    if (t.op == o && algorithm_of(t.fn) == a && direction_of(t.fn) == d)
      return t.fn;
  return std::nullopt;
}
constexpr bool one_template_a_shot() {
  for (const TemplateInfo& t : kTemplates)
    if (template_of(t.op, algorithm_of(t.fn), direction_of(t.fn)) != t.fn)
      return false;
  return true;
}
static_assert(one_template_a_shot(),
              "an op has two templates at one algorithm and direction");

// =================================================================================================
// OVER WHAT IS COMPILED, at compile time: the template arguments are constexpr,
// so each value (the world, a dtype, a built config) is turned into its
// template parameter here; one that was not built raises.
// =================================================================================================

template <typename WRAPPED>
struct type {
  using t = WRAPPED;
};

template <int INT_VALUE>
using constant = std::integral_constant<int, INT_VALUE>;

[[noreturn]] inline void not_built(const std::string& what) {
  throw std::runtime_error("hip_custom_communicator: " + what + " not built");
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
  if (!((world == built[INDICES] && (f(constant<built[INDICES]>{}), true)) ||
        ...))
    not_built("world size " + std::to_string(world));
}

template <typename VISITOR>
void by_world(int world, VISITOR&& f) {
  by_world_in(world, f,
              std::make_index_sequence<kBuild.supports.worlds.size()>{});
}

template <typename VISITOR, size_t... INDICES>
void by_dtype_in(DType d, VISITOR& f, std::index_sequence<INDICES...>) {
  constexpr auto& built = kBuild.supports.dtypes;
  if (!((d == built[INDICES] &&
         (f(type<typename of_dtype<built[INDICES]>::t>{}), true)) ||
        ...))
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

// A TEMPLATE'S BUILD, compiled in: the one of template K's builds (the catalog
// above) `b` is, and only those. f is handed it as build_constant<B>, B its
// op's BuildConfig, every field a constant expression.
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
void by_build_in(const BuildConfig& b, VISITOR& f,
                 std::index_sequence<INDICES...>) {
  if (!((b == builds_of(TEMPLATE)[INDICES] &&
         (f(build_constant<listed_build<TEMPLATE, INDICES>>{}), true)) ||
        ...))
    not_built(std::string("template ") + to_string(TEMPLATE) +
              " at that build");
}

template <Template TEMPLATE, typename VISITOR>
void by_build(const BuildConfig& b, VISITOR&& f) {
  by_build_in<TEMPLATE>(b, f,
                        std::make_index_sequence<builds_of(TEMPLATE).size()>{});
}

[[noreturn]] inline void not_this_ops(Template k) {
  throw std::runtime_error("hip_custom_communicator: template " +
                           std::to_string(static_cast<int>(k)) +
                           " is not this op's");
}

// =================================================================================================
// THE COMPILED INSTANCE select picks: from the template and config it chose,
// the call's dtypes and the world, one compiled kernel, checked here to have
// its op's signature (types.cuh) with its element pointers erased, which is
// what a launch calls it through.
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

// `kernel`, as the launch holds it, once it is checked to be callable as
// `Signature`.
template <typename SIGNATURE, typename... PARAMS>
const void* instance(void (*kernel)(PARAMS...)) {
  static_assert(
      std::is_same_v<typename erased<void (*)(PARAMS...)>::t, SIGNATURE>,
      "a compiled kernel does not have its op's signature");
  return reinterpret_cast<const void*>(kernel);
}

// WHETHER `kernel` HOLDS A GRID OF `blocks` BLOCKS OF `threads` RESIDENT: only
// the compiled kernel knows what it uses.
inline bool resident(const Handle& h, const void* kernel, int blocks,
                     int threads) {
  return blocks <= h.resident_blocks(kernel, threads);
}

// From `m` rows up (to the next entry's), at `world` ranks and rows `n`
// elements wide: template `fn` at `config`, its build and its launch, as
// measured.
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

// all_reduce, gfx950 on n11, bf16, 8 ranks: the one-shot while a row's tiles
// are few (through 8 tokens of 3584, 4 of 7168; swept 2026-10-10T23-22-13Z),
// then the two-shot. From 64 tokens of 7168 (128 of 3584) the two-shot's grid
// is the flat walk's measured best, swept uniformly at 40-256 blocks
// (2026-10-12T01-27-40Z, 01-45-53Z; one run a point): 80-120 blocks at mid
// sizes, 72 from 1 MB-ish shares up, where the flat walk beats be0e1f4e9c (7168
// x 4096: 262.4 against 263.9 us). Equal bytes measure alike (the walk is over
// the flat range), so the two widths share grids. At 48 tokens a chunk a block
// (48 and 84 blocks: 48x3584 10.2 -> 9.0 us, 48x7168 11.2 -> 10.4;
// 2026-10-18T08-56-55Z); 8 tokens of 7168 the one-shot on 56 blocks (8.5 ->
// 8.1, 09-02-30Z; the same bytes at 16 of 3584 lost, 8.4 -> 9.0).
constexpr TunedKernel kAllReduceKernels[] = {
    {8,
     1,
     3584,
     Template::all_reduce_pull_one_shot,
     {AllReduceBuildConfig{128, 1, 1, 1024}, {32, 0}}},
    {8,
     16,
     3584,
     Template::all_reduce_pull_one_shot,
     {AllReduceBuildConfig{128, 1, 1, 1024}, {64, 0}}},
    {8,
     32,
     3584,
     Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {32, 0}}},
    {8,
     48,
     3584,
     Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {48, 0}}},
    {8,
     64,
     3584,
     Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {64, 0}}},
    {8,
     256,
     3584,
     Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {80, 0}}},
    {8,
     512,
     3584,
     Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {120, 0}}},
    {8,
     1024,
     3584,
     Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {104, 0}}},
    {8,
     2048,
     3584,
     Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {72, 0}}},
    {8,
     1,
     7168,
     Template::all_reduce_pull_one_shot,
     {AllReduceBuildConfig{128, 1, 1, 1024}, {32, 0}}},
    {8,
     8,
     7168,
     Template::all_reduce_pull_one_shot,
     {AllReduceBuildConfig{128, 1, 1, 1024}, {56, 0}}},
    {8,
     16,
     7168,
     Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {32, 0}}},
    {8,
     32,
     7168,
     Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {64, 0}}},
    {8,
     48,
     7168,
     Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {84, 0}}},
    {8,
     64,
     7168,
     Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {64, 0}}},
    {8,
     128,
     7168,
     Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {80, 0}}},
    {8,
     256,
     7168,
     Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {120, 0}}},
    {8,
     512,
     7168,
     Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {104, 0}}},
    {8,
     1024,
     7168,
     Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512}, {72, 0}}},
};

// THE PLAIN ALL-REDUCE'S BUILDS THAT LAY OUT WHOLE at `world` ranks and `nl`
// elements a pack, which its dispatch instantiates and nothing else: the
// one-shot's TILE_M rows a THREADS / TILE_M-thread row each across TILE_N; the
// two-shot's [R, 1, TILE_N] chunk, a rank's THREADS / world threads a group
// each across it (common.cuh's COLLECTIVES: the flat walk). Every tuned entry
// must lay out, or select would refuse it at run time (tile_not_built): checked
// here, at compile time.
constexpr bool one_shot_lays_out(const AllReduceBuildConfig& c, int nl) {
  return c.threads_per_block % c.tile_m == 0 &&
         c.tile_n % (nl * (c.threads_per_block / c.tile_m)) == 0;
}
constexpr bool two_shot_lays_out(const AllReduceBuildConfig& c, int world,
                                 int nl) {
  return c.tile_m == 1 && c.threads_per_block % world == 0 &&
         c.tile_n % (nl * (c.threads_per_block / world)) == 0;
}
constexpr bool all_reduce_kernels_lay_out() {
  const int nl =
      kBuild.memory.pack_bytes / 2;  // the tuned dtypes', bf16 and fp16
  for (const TunedKernel& k : kAllReduceKernels) {
    const auto& c = std::get<AllReduceBuildConfig>(k.config.build);
    if (k.fn == Template::all_reduce_pull_one_shot && !one_shot_lays_out(c, nl))
      return false;
    if (k.fn == Template::all_reduce_pull_two_shot &&
        !two_shot_lays_out(c, k.world, nl))
      return false;
  }
  return true;
}
static_assert(all_reduce_kernels_lay_out(),
              "a tuned all-reduce build its dispatch never instantiates");

// gfx950 on n11, bf16, 8 ranks. Transcribed from the size thresholds the sweeps
// set (each crossover cites its run) at the widths we run, until the tuner
// writes them. rms_norm: one-shot through 128 KiB (at 64 KiB it lost at 16
// tokens, 11.43 against 10.56 us; 2026-09-30T21-06-57Z); the push through 1.75
// MiB (256 tokens of 3584: 18.04 against the pull's 18.41), the pull from 2.6
// MiB (2026-10-01T03-26-44Z). At 7168 the push through 256 tokens since its first
// barrier is unordered (31.1 against the pull's 32.6 us; 2026-10-18T10-14-39Z).
constexpr TunedKernel kRmsNormKernels[] = {
    {8,
     1,
     3584,
     Template::all_reduce_pull_one_shot_rms_norm,
     {AllReduceRmsNormBuildConfig{512, 1, 1, 4096}, {16, 0}}},
    {8,
     19,
     3584,
     Template::all_reduce_push_two_shot_rms_norm,
     {AllReduceRmsNormBuildConfig{512, 1, 1, 4096}, {256, 0}}},
    {8,
     257,
     3584,
     Template::all_reduce_pull_two_shot_rms_norm,
     {AllReduceRmsNormBuildConfig{512, 1, 1, 4096}, {48, 0}}},
    {8,
     1,
     7168,
     Template::all_reduce_pull_one_shot_rms_norm,
     {AllReduceRmsNormBuildConfig{512, 1, 1, 8192}, {16, 0}}},
    {8,
     10,
     7168,
     Template::all_reduce_push_two_shot_rms_norm,
     {AllReduceRmsNormBuildConfig{512, 1, 1, 8192}, {256, 0}}},
    // 32 blocks through 2048 tokens (512 / 1024 / 2048: 52.4 / 88.9 / 160.6 against 54.9 / 91.0
    // / 165.5 us on 48; more blocks lose, 61.4 / 137.5 at 128: 2026-10-18T10-26-41Z, 10-14-39Z),
    // 48 from 4096 (not swept since).
    {8,
     257,
     7168,
     Template::all_reduce_pull_two_shot_rms_norm,
     {AllReduceRmsNormBuildConfig{512, 1, 1, 8192}, {32, 0}}},
    {8,
     4096,
     7168,
     Template::all_reduce_pull_two_shot_rms_norm,
     {AllReduceRmsNormBuildConfig{512, 1, 1, 8192}, {48, 0}}},
    // PAST THE MODEL'S WIDTHS, to the widest build: the one-shot, its grid the
    // 8192's; not swept.
    {8,
     1,
     16384,
     Template::all_reduce_pull_one_shot_rms_norm,
     {AllReduceRmsNormBuildConfig{512, 1, 1, 16384}, {16, 0}}},
    {8,
     4096,
     3584,
     Template::all_reduce_pull_split_rms_norm,
     {AllReduceRmsNormBuildConfig{256, 1, 1, 4096}, {8192, 0}}},
    {8,
     4096,
     7168,
     Template::all_reduce_pull_split_rms_norm,
     {AllReduceRmsNormBuildConfig{256, 1, 1, 8192}, {8192, 0}}},
};

// add_rms_norm: the one-shot not swept (rms_norm's); the push through 1.31 MiB
// (192 tokens of 3584: 15.48 against the pull's 16.28), the pull at 1.75 MiB
// (18.36 against the push's 18.46; 2026-10-01T03-26-44Z). Since the push's first
// barrier is unordered, the push through 512 tokens at 3584 (256 / 512: 18.9 / 32.6
// against the pull's 22.5 / 35.3 us) and 256 at 7168 (128 / 256: 18.4 / 32.0
// against 24.3 / 36.2; 2026-10-18T10-14-39Z).
constexpr TunedKernel kAddRmsNormKernels[] = {
    {8,
     1,
     3584,
     Template::all_reduce_pull_one_shot_add_rms_norm,
     {AllReduceAddRmsNormBuildConfig{512, 1, 1, 4096}, {16, 0}}},
    {8,
     19,
     3584,
     Template::all_reduce_push_two_shot_add_rms_norm,
     {AllReduceAddRmsNormBuildConfig{512, 1, 1, 4096}, {256, 0}}},
    {8,
     513,
     3584,
     Template::all_reduce_pull_two_shot_add_rms_norm,
     {AllReduceAddRmsNormBuildConfig{512, 1, 1, 4096}, {48, 0}}},
    {8,
     1,
     7168,
     Template::all_reduce_pull_one_shot_add_rms_norm,
     {AllReduceAddRmsNormBuildConfig{512, 1, 1, 8192}, {16, 0}}},
    {8,
     10,
     7168,
     Template::all_reduce_push_two_shot_add_rms_norm,
     {AllReduceAddRmsNormBuildConfig{512, 1, 1, 8192}, {256, 0}}},
    // 40 blocks through 2048 tokens (512 / 1024 / 2048: 58.3 / 99.2 / 173.5 against 60.1 / 102.8
    // / 175.8 us on 48; 2026-10-18T10-26-41Z), 48 from 4096 (not swept since).
    {8,
     257,
     7168,
     Template::all_reduce_pull_two_shot_add_rms_norm,
     {AllReduceAddRmsNormBuildConfig{512, 1, 1, 8192}, {40, 0}}},
    {8,
     4096,
     7168,
     Template::all_reduce_pull_two_shot_add_rms_norm,
     {AllReduceAddRmsNormBuildConfig{512, 1, 1, 8192}, {48, 0}}},
    // Past the model's widths, as rms_norm's.
    {8,
     1,
     16384,
     Template::all_reduce_pull_one_shot_add_rms_norm,
     {AllReduceAddRmsNormBuildConfig{512, 1, 1, 16384}, {16, 0}}},
    {8,
     4096,
     3584,
     Template::all_reduce_pull_split_add_rms_norm,
     {AllReduceAddRmsNormBuildConfig{256, 1, 1, 4096}, {8192, 0}}},
    {8,
     4096,
     7168,
     Template::all_reduce_pull_split_add_rms_norm,
     {AllReduceAddRmsNormBuildConfig{256, 1, 1, 8192}, {8192, 0}}},
};

// AttnRes: the push from 1 token (it beat the one-shot, 12.89 against 13.68
// us; 14.03 against 15.18 at 8; 2026-10-01T03-57-23Z); the pull past 256 tokens
// at 3584 and 192 at 7168, with balanced slices (2026-10-05T03-40-33Z: 3584 x
// 384 27.7 against the push's 30.4; 7168 x 192 27.3 against 27.7). THE ONE-SHOT
// BESIDE THE PUSH from 1 token: listed after it, so it runs only where the push
// does not (a row narrower than a pack a rank); its grid the norms' one-shot's,
// not swept. THE PULL AT 512 THREADS, TILE_K 1, from 128 tokens: it beat the push
// (unordered first barrier) at 128 / 256 tokens, 15.4 / 21.5 against 16.0 / 22.0 us
// at 3584, 21.8 / 34.1 against 22.3 / 36.9 at 7168, and the 256-thread TILE_K 3 pull
// at 256 x 7168 (36.3) (2026-10-18T10-14-39Z, 10-08-46Z).
constexpr TunedKernel kAttnResKernels[] = {
    {8,
     1,
     3584,
     Template::all_reduce_push_two_shot_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{512, 1, 1, 4096, 1}, {256, 0}}},
    {8,
     1,
     3584,
     Template::all_reduce_pull_one_shot_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{512, 1, 1, 4096, 1}, {16, 0}}},
    {8,
     128,
     3584,
     Template::all_reduce_pull_two_shot_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{512, 1, 1, 4096, 1}, {256, 0}}},
    {8,
     257,
     3584,
     Template::all_reduce_pull_two_shot_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 4096, 3}, {256, 0}}},
    {8,
     1,
     7168,
     Template::all_reduce_push_two_shot_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{512, 1, 1, 8192, 1}, {256, 0}}},
    {8,
     1,
     7168,
     Template::all_reduce_pull_one_shot_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{512, 1, 1, 8192, 1}, {16, 0}}},
    {8,
     128,
     7168,
     Template::all_reduce_pull_two_shot_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{512, 1, 1, 8192, 1}, {256, 0}}},
    {8,
     512,
     7168,
     Template::all_reduce_pull_two_shot_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 8192, 1}, {256, 0}}},
    {8,
     1024,
     7168,
     Template::all_reduce_pull_two_shot_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 8192, 1}, {192, 0}}},
    {8,
     4096,
     3584,
     Template::all_reduce_pull_split_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 4096, 1}, {8192, 0}}},
    {8,
     4096,
     7168,
     Template::all_reduce_pull_split_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 8192, 1}, {8192, 0}}},
};

// INT4'S CROSSOVER, an op's: the least bytes a two-shot call moves INT4 at when
// the handle's quantization is int4, the exact build below it. Measured at each
// op's exact grids, INT4 on against off (tuned picks, 7168 bf16, 8 ranks,
// 2026-10-16T03-50-18Z against 04-01-55Z), us at 512 / 1024 / 8192 tokens:
//   plain      41.7 / 75.4 / 535.1 exact, 88.2 / 72.4 / 284.4 INT4: 1024 tokens up
//   rms_norm   42.8 / 76.8 / 547.8 exact, 48.8 / 62.0 / 362.2 INT4: 1024 up
//   add_rms    51.2 / 92.4 / 566.5 exact, 58.1 / 84.2 / 471.4 INT4: 1024 up
//   AttnRes    57.7 / 92.9 / 643.1 exact, 51.3 / 105.5 / 762.5 INT4: NEVER for now
//     (its reduce-scatter is still the LDS one, a word a thread, one wave summing;
//     the plain path's register sum, a group a thread, is what made INT4 win there)
// So 14 MiB (1024 tokens of 7168) for the three, and AttnRes never; the norms'
// and AttnRes since 7 MiB, their INT4 a group a thread, and the plain one since
// 3.5 MiB (below).
struct Int4Crossover {
  OpType op;
  int64_t min_bytes;
};
constexpr Int4Crossover kInt4Crossovers[] = {
    // The plain all-reduce from 3.5 MiB since its push and gather are a rank a row of threads
    // (forced INT4 at 512 x 3584 and 256 x 7168: 19.1 us against the exact pick's 23.7 / 23.8;
    // at 1.75 MiB a tie, 256 x 3584 15.3 against 16.1, 128 x 7168 15.3 against ~15.4;
    // 2026-10-18T12-14-52Z, 11-55-21Z).
    {OpType::all_reduce, 3584 * 1024},
    // The norms from 7 MiB since their INT4 is a group a thread (forced INT4
    // against the exact pick at 1024 x 3584 and 512 x 7168: rms 39.0 / 40.4
    // against 44.8 / 54.5 us, add 44.0 / 45.9 against 63.0 / 59.3; at 3.5 MiB a
    // tie, 30.3 / 31.3 against 28.5 / 32.6; 2026-10-18T05-49-21Z).
    {OpType::all_reduce_rms_norm, 7 * 1024 * 1024},
    {OpType::all_reduce_add_rms_norm, 7 * 1024 * 1024},
    // AttnRes from 7 MiB since its INT4 is a group a thread (1024 x 3584: 47.4 us against the
    // exact pick's 54.8; 2026-10-18T07-50-18Z, 04-52-25Z).
    {OpType::all_reduce_add_attn_res_rms_norm, 7 * 1024 * 1024},
};
// EACH OP'S INT4 KERNELS, its own list, consulted only from the op's crossover up
// (kInt4Crossovers), where the exact lists pick two-shots (a one-shot moves its input as
// it is, so it has no INT4 entry). They began as the exact lists' two-shots moving INT4
// at the same grids (the plain all-reduce at 72 blocks: 71.8 / 109.0 / 178.2 / 286.6 us
// at 1024-8192 tokens of 7168 against the exact 75.5 / 142.0 / 273.3 / 535.4;
// 2026-10-16T03-14-06Z); each op's INT4 body and grid are now its own (below). THE
// PULLED NORMS' INT4 IS A GROUP A THREAD (their INT4 body, all_reduce_add_rms_norm.cuh):
// TILE_N / 32 threads, a block one row's groups.
constexpr bool int4_group_a_thread(Template k) {
  return k == Template::all_reduce_pull_two_shot_rms_norm ||
         k == Template::all_reduce_pull_two_shot_add_rms_norm ||
         k == Template::all_reduce_pull_two_shot_add_attn_res_rms_norm;
}
// THE STAGING'S ROWS AS A KERNEL VIEWS IT: the input's m, or a group-a-thread
// INT4 norm's inbox, WORLD slots of an owner's ceil(m / WORLD) rows.
constexpr int64_t staged_rows(Template k, int quantization, int64_t m, int world) {
  return int4_group_a_thread(k) && quantization == Quantization::int4
             ? world * ((m + world - 1) / world)
             : m;
}
// THE PLAIN ALL-REDUCE'S INT4 KERNELS, its own (no longer the exact list's grids) since its
// push and gather are a rank a row of threads (reduce_scatter.cuh). Grids from its forced sweep
// (16-256 blocks, 128-8192 tokens; 2026-10-18T12-05-38Z, 12-14-52Z): 56 blocks through 3.5 MiB
// (512 x 3584: 19.1 us, 64 blocks 19.4, 128 20.2), 128 from 7 MiB (8192 x 7168: 183.0 against
// 212.7 on 96 and 229.2 on 64; 1024 x 3584 26.3 against the best, 80, 26.0). The exact
// list's grids (72-120) had left it 1.10-1.19x QuickReduce INT4's.
constexpr TunedKernel kAllReduceInt4Kernels[] = {
    {8,
     1,
     3584,
     Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512, Quantization::int4}, {56, 0}}},
    {8,
     1024,
     3584,
     Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512, Quantization::int4}, {128, 0}}},
    {8,
     1,
     7168,
     Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512, Quantization::int4}, {56, 0}}},
    {8,
     512,
     7168,
     Template::all_reduce_pull_two_shot,
     {AllReduceBuildConfig{512, 1, 1, 512, Quantization::int4}, {128, 0}}},
};
// THE PULLED NORMS' INT4 KERNELS, their own: from the crossover
// up select picks the pull, a group a thread, at the grid its sweep set
// (forced, 64-384 blocks, 1024-8192 tokens; 2026-10-18T05-42-13Z): 256 blocks
// of 128 threads at 3584, 128 of 256 at 7168, each within 10% of the best grid
// at every size (rms 8192 x 7168 228.9 us, QuickReduce INT4 + rms_norm 412.3;
// add 266.9 against 448.6). The grid the exact kernels run, 48, left INT4 at
// 0.98-1.9x QuickReduce's (2026-10-18T05-23-31Z): a block a row, a group a
// thread, so the grid is the rows in flight.
constexpr TunedKernel kRmsNormInt4Kernels[] = {
    {8,
     1,
     3584,
     Template::all_reduce_pull_two_shot_rms_norm,
     {AllReduceRmsNormBuildConfig{128, 1, 1, 4096, Quantization::int4}, {256, 0}}},
    {8,
     1,
     7168,
     Template::all_reduce_pull_two_shot_rms_norm,
     {AllReduceRmsNormBuildConfig{256, 1, 1, 8192, Quantization::int4}, {128, 0}}},
    {8,
     4096,
     3584,
     Template::all_reduce_pull_split_rms_norm,
     {AllReduceRmsNormBuildConfig{256, 1, 1, 4096, Quantization::int4}, {8192, 0}}},
    {8,
     4096,
     7168,
     Template::all_reduce_pull_split_rms_norm,
     {AllReduceRmsNormBuildConfig{256, 1, 1, 8192, Quantization::int4}, {8192, 0}}},
};
constexpr TunedKernel kAddRmsNormInt4Kernels[] = {
    {8,
     1,
     3584,
     Template::all_reduce_pull_two_shot_add_rms_norm,
     {AllReduceAddRmsNormBuildConfig{128, 1, 1, 4096, Quantization::int4}, {256, 0}}},
    {8,
     1,
     7168,
     Template::all_reduce_pull_two_shot_add_rms_norm,
     {AllReduceAddRmsNormBuildConfig{256, 1, 1, 8192, Quantization::int4}, {128, 0}}},
    {8,
     4096,
     3584,
     Template::all_reduce_pull_split_add_rms_norm,
     {AllReduceAddRmsNormBuildConfig{256, 1, 1, 4096, Quantization::int4}, {8192, 0}}},
    {8,
     4096,
     7168,
     Template::all_reduce_pull_split_add_rms_norm,
     {AllReduceAddRmsNormBuildConfig{256, 1, 1, 8192, Quantization::int4}, {8192, 0}}},
};
// AttnRes's INT4 kernels, its own: the pull, a group a thread, a row split in
// units of a row and a class of owners (its INT4 body), TILE_K 3 (the best at every prefill size,
// 1-15% under 1). Grids from the sweep (forced, 256-512 blocks, 1024-8192 tokens;
// 2026-10-18T07-50-18Z): 512 where the units are few (an owner's slice <= 256 rows: two classes
// or more) and at 8192 x 3584, 256 from 4096 x 7168 (309 against 339 us at 512). 384 lost
// everywhere (one class, uneven). At 3584 the 2-wave blocks hold 4 a CU (resident_blocks), so 1024
// at 2048-4096 tokens, a wave a SIMD more under AttnRes (2048: 82.5 against 95.1 us at 512; 4096:
// 161.7 against 171.8 at 256; 8192 341.7 against 302.6 at 512; 2026-10-18T13-15-40Z).
constexpr TunedKernel kAttnResInt4Kernels[] = {
    {8,
     1,
     3584,
     Template::all_reduce_pull_two_shot_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{128, 1, 1, 4096, 3, Quantization::int4}, {512, 0}}},
    {8,
     2048,
     3584,
     Template::all_reduce_pull_two_shot_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{128, 1, 1, 4096, 3, Quantization::int4}, {1024, 0}}},
    {8,
     8192,
     3584,
     Template::all_reduce_pull_two_shot_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{128, 1, 1, 4096, 3, Quantization::int4}, {512, 0}}},
    {8,
     4096,
     3584,
     Template::all_reduce_pull_split_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 4096, 1, Quantization::int4}, {8192, 0}}},
    {8,
     1,
     7168,
     Template::all_reduce_pull_two_shot_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 8192, 3, Quantization::int4}, {512, 0}}},
    {8,
     4096,
     7168,
     Template::all_reduce_pull_two_shot_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 8192, 3, Quantization::int4}, {256, 0}}},
    {8,
     4096,
     7168,
     Template::all_reduce_pull_split_add_attn_res_rms_norm,
     {AllReduceAddAttnResRmsNormBuildConfig{256, 1, 1, 8192, 1, Quantization::int4}, {8192, 0}}},
};

constexpr int64_t int4_min_bytes(OpType o) {
  for (const Int4Crossover& c : kInt4Crossovers)
    if (c.op == o) return c.min_bytes;
  return -1;  // never: every op has one (ops_have_crossovers)
}

// =================================================================================================
// THE OPS, in OpType's order: each one's name (as Python calls it), its tuned
// kernels, and what a plain call of it (nothing forced) must be to run, which
// `build_info` hands Python's `should`.
// =================================================================================================

// A NORM'S WEIGHT, besides its input's dtype: fp32 (by_weight compiles exactly
// these).
constexpr DType kNormWeights[] = {DType::f32};

struct Op {
  OpType type;
  const char* name;
  std::span<const TunedKernel> kernels;
  // Its INT4 kernels: what a call the handle quantizes runs (kInt4Crossovers).
  std::span<const TunedKernel> int4_kernels;
  // The dtypes its input may have.
  std::span<const DType> dtypes;
  // The dtypes its weight may have besides its input's; none for an op with no
  // weight.
  std::span<const DType> weights;
  // Whether a tile holds the whole row (its tile_n at least the row: a norm
  // reads the whole row); otherwise the tiles cross it.
  bool tile_holds_row;
  // Whether its input is [rows, cols] as given (2-D), not any shape read as
  // rows of its last dimension.
  bool two_d;
};

// AN OP AND ITS NAME FROM ONE TOKEN, so the name cannot differ from the enum's.
#define HIP_CUSTOM_COMMUNICATOR_NAMED(o) OpType::o, #o

constexpr Op kOps[] = {
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce),
     kAllReduceKernels, kAllReduceInt4Kernels,
     kBuild.supports.dtypes,
     {},
     false,
     false},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_rms_norm), kRmsNormKernels, kRmsNormInt4Kernels,
     kBuild.supports.dtypes, kNormWeights, true, true},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_add_rms_norm), kAddRmsNormKernels, kAddRmsNormInt4Kernels,
     kBuild.supports.dtypes, kNormWeights, true, true},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_add_attn_res_rms_norm),
     kAttnResKernels, kAttnResInt4Kernels,
     kBuild.supports.dtypes,
     {},
     true,
     true},
};
#undef HIP_CUSTOM_COMMUNICATOR_NAMED
constexpr int kNumOps = sizeof(kOps) / sizeof(Op);

constexpr const Op& op(OpType t) { return kOps[static_cast<int>(t)]; }
constexpr const char* to_string(OpType t) { return op(t).name; }

constexpr bool ops_in_order() {
  for (int i = 0; i < kNumOps; ++i)
    if (static_cast<int>(kOps[i].type) != i) return false;
  return true;
}
static_assert(ops_in_order(), "kOps must list every OpType in its order");

constexpr bool ops_have_crossovers() {
  if (sizeof(kInt4Crossovers) / sizeof(Int4Crossover) != kNumOps) return false;
  for (const Op& o : kOps)
    if (int4_min_bytes(o.type) < 0) return false;
  return true;
}
static_assert(ops_have_crossovers(),
              "kInt4Crossovers must give every op its crossover once");

// EVERY OP HAS KERNELS, each one of its own templates, its build listed and its
// launch one the forcing would accept (a grid of 1 to kMaxBlocks blocks, a
// split only on the AttnRes pull).
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
// EVERY OP'S INT4 KERNELS ARE ITS OWN TWO-SHOTS, BUILT MOVING INT4 and launchable.
constexpr bool ops_have_int4_kernels() {
  for (const Op& o : kOps) {
    if (o.int4_kernels.empty()) return false;
    for (const TunedKernel& k : o.int4_kernels)
      if (op_of(k.fn) != o.type || algorithm_of(k.fn) == Algorithm::one_shot ||
          quantization_of(k.config.build) != Quantization::int4 ||
          !built(k.fn, k.config.build) || !launch_fits(k.fn, k.config.launch))
        return false;
  }
  return true;
}
static_assert(ops_have_int4_kernels(),
              "an op has no INT4 kernels, or one is not its own INT4 two-shot");
static_assert(ops_have_kernels(),
              "an op has no kernels, or one is another op's, not built or not "
              "launchable");

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

// A COLUMN TWO-SHOT reduce-scatters by columns, so every block keeps every row;
// a row two-shot gives each rank its rows.
constexpr bool slices_columns(Template k) {
  return k == Template::all_reduce_pull_two_shot_add_attn_res_rms_norm ||
         k == Template::all_reduce_push_two_shot_rms_norm ||
         k == Template::all_reduce_push_two_shot_add_rms_norm ||
         k == Template::all_reduce_push_two_shot_add_attn_res_rms_norm;
}

// A TEMPLATE THAT CUTS EVERY ROW INTO A COLUMN SLICE A RANK: the plain two-shot
// and the column splits. It needs a 16-byte pack for every rank; with fewer, a
// rank's slice is empty.
constexpr bool slices_a_row(Template k) {
  return k == Template::all_reduce_pull_two_shot ||
         k == Template::all_reduce_push_two_shot || slices_columns(k);
}

// WHETHER EVERY RANK GETS COLUMNS: a template that slices a row needs a load's
// worth (16 bytes) of the row's `n` for each of the `world` ranks; any other
// template takes any row.
constexpr bool every_rank_gets_columns(Template k, int64_t n, DType dtype,
                                       int world) {
  return !slices_a_row(k) ||
         n * elem_bytes(dtype) >= int64_t{world} * kBuild.memory.pack_bytes;
}

// =================================================================================================
// 1. THE FORCING, ALL OR NOTHING: the template (its algorithm and direction,
// where the op has more than one), every field of its build, and its launch, or
// none of them. Some but not all is forced_incomplete; a value out of range is
// its Error; a build the template does not list is the check's to name. None:
// select picks. The build's `quantization` alone may be left out of a forcing,
// which is then none (the build as listed), never the handle's: a forced call
// runs what it names; given without the rest, it is forced_incomplete.
// =================================================================================================

// The kernel a call runs: its template and its complete config.
struct Chosen {
  Template fn;
  KernelConfig config;
};

// `build_fields` are the op's BuildConfig's fields as the call forced them, in
// its order, but its quantization, which is `quantization`;
// `reduce_scatter_blocks` the AttnRes split, which only that op takes (another
// passes none); `build_of` makes the op's BuildConfig from the fields, called
// only when every one is given.
template <typename BUILD_OF>
constexpr std::variant<std::optional<Chosen>, Error> forced(
    OpType o, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction,
    std::initializer_list<std::optional<int>> build_fields,
    std::optional<int> quantization, std::optional<int> blocks_per_grid,
    std::optional<int> reduce_scatter_blocks, BUILD_OF build_of) {
  const bool names_template = templates_of(o) > 1;
  const bool splits = o == OpType::all_reduce_add_attn_res_rms_norm;
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
  if (given == 0 && !algorithm && !direction && !quantization)
    return std::optional<Chosen>{};
  if (given != needed) return Error::forced_incomplete;
  const int q = quantization.value_or(Quantization::none);
  if (q != Quantization::none && q != Quantization::int4)
    return Error::no_such_quantization;
  // THE TEMPLATE: the one an op with one template has, else the algorithm and
  // direction's.
  const std::optional<Template> fn =
      names_template ? template_of(o, *algorithm, *direction)
                     : std::optional{only_template(o)};
  if (!fn || (algorithm && algorithm_of(*fn) != *algorithm) ||
      (direction && direction_of(*fn) != *direction))
    return Error::no_such_template;
  // THE RANGES: whole waves up to the build's maximum threads, a grid of 1 to
  // kMaxBlocks blocks, waves a EU the CU holds, positive tile fields, a split
  // only on the AttnRes pull.
  const std::span<const std::optional<int>> f(build_fields.begin(),
                                              build_fields.size());
  const int threads = *f[0], waves = *f[1];
  if (threads < kWaveSize || threads > kBuild.kernels.max_threads ||
      threads % kWaveSize != 0 || *blocks_per_grid < 1 ||
      *blocks_per_grid > kMaxBlocks || waves < 1 ||
      waves > kTarget.max_waves_per_cu / kTarget.simds_per_cu)
    return Error::launch_out_of_range;
  for (size_t i = 2; i < f.size(); ++i)
    if (*f[i] < 1) return Error::field_not_positive;
  const int split = splits ? *reduce_scatter_blocks : 0;
  if (split < 0) return Error::field_not_positive;
  if (split > 0 &&
      *fn != Template::all_reduce_pull_two_shot_add_attn_res_rms_norm)
    return Error::field_not_this_templates;
  return std::optional<Chosen>{
      Chosen{*fn, {quantized(build_of(), q), {*blocks_per_grid, split}}}};
}

// =================================================================================================
// 2. THE PICK: an op's tuned kernel for the call, exactly as listed, or the
// Error that no listed kernel runs it.
// =================================================================================================

constexpr int64_t distance(int64_t a, int64_t b) {
  return a < b ? b - a : a - b;
}

// Among the op's kernels at the call's world (else the nearest tuned world),
// THOSE THAT RUN THE CALL: a kernel that slices a row needs a pack for every
// rank (every_rank_gets_columns), and a tile that must hold the row
// (`tile_cols`, the op's) holds it. Among those, the nearest tuned width,
// reading the call's rows as that width's rows of the same bytes (a crossover
// is about bytes); then the entry with the most rows the call reaches, the
// first listed where two tie, the fewest rows where it reaches none. None runs:
// a row too wide for every tuned tile, or too narrow for every slicing kernel.
// THE WORLD AN OP'S KERNELS ARE PICKED AT for a call at `world`: its own, else
// the nearest tuned.
constexpr int tuned_world(OpType o, int world) {
  int w = 0;
  for (const TunedKernel& k : op(o).kernels)
    if (w == 0 || distance(k.world, world) < distance(w, world)) w = k.world;
  return w;
}

constexpr std::variant<TunedKernel, Error> pick(
    OpType o, int world, int64_t m, int64_t n, int64_t tile_cols, DType dtype,
    std::span<const TunedKernel> kernels) {
  const int w = tuned_world(o, world);
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
  const int64_t as_rows = m * n / h;
  const TunedKernel* most = nullptr;
  const TunedKernel* least = nullptr;
  for (const TunedKernel& k : kernels) {
    if (!runs(k) || k.n != h) continue;
    if (!least || k.m < least->m) least = &k;
    if (k.m <= as_rows && (!most || k.m > most->m)) most = &k;
  }
  return most ? *most : *least;
}
// The op's exact kernels.
constexpr std::variant<TunedKernel, Error> pick(OpType o, int world, int64_t m,
                                                int64_t n, int64_t tile_cols,
                                                DType dtype) {
  return pick(o, world, m, n, tile_cols, dtype, op(o).kernels);
}

// =================================================================================================
// THE QUANTIZATION: which calls move INT4 (common.cuh's QUANTIZATION), by the
// handle's quantization and the op's crossover (kInt4Crossovers), and the
// INT4 payload's room in the regions.
// =================================================================================================

// WHETHER AN INT4 TWO-SHOT TAKES A ROW OF `n` ELEMENTS at `world` ranks: whole
// groups, and whole groups for every rank, so every rank's slice (a column
// split's columns, the flat walk's share) starts on a group and a group is 4
// lanes' packs in every layout. 3584 and 7168 at 8 ranks: 14 and 28 groups a
// rank.
constexpr bool int4_takes_row(int64_t n, int world) {
  return n % kInt4Group == 0 && (n / kInt4Group) % world == 0;
}

// THE TUNED KERNEL MOVES INT4 when the handle's quantization is int4, the call
// is its op's crossover or more, its kernel a two-shot, its dtype 16-bit (a
// pack is a word of its group), its row `int4_takes_row` and the quantized
// regions hold it; then its build moves INT4, its template, tile and grid as
// tuned. Else, and for every one-shot, its build as it is. A forced call is
// never changed here: its quantization is its own.
inline bool quantizes(const Handle& h, OpType o, int64_t m, int64_t n,
                      DType dtype) {
  return h.quantization() == Quantization::int4 && elem_bytes(dtype) == 2 &&
         m * n * elem_bytes(dtype) >= int4_min_bytes(o) &&
         int4_takes_row(n, h.world_size()) &&
         int4_payload_bytes(m, n) <= h.staging_bytes() &&
         int4_payload_bytes(1, m * n) <= h.partials_bytes();
}

// THE KERNEL A CALL RUNS: the forced one as forced, else the tuned one at the
// handle's quantization.
inline std::variant<Chosen, Error> chosen(
    const Handle& h, const std::variant<std::optional<Chosen>, Error>& forcing,
    OpType o, int64_t m, int64_t n, int64_t tile_cols, DType dtype) {
  if (const Error* e = std::get_if<Error>(&forcing)) return *e;
  if (const std::optional<Chosen>& f = std::get<std::optional<Chosen>>(forcing))
    return *f;
  const std::variant<TunedKernel, Error> tuned =
      pick(o, h.world_size(), m, n, tile_cols, dtype,
           quantizes(h, o, m, n, dtype) ? op(o).int4_kernels : op(o).kernels);
  if (const Error* e = std::get_if<Error>(&tuned)) return *e;
  const TunedKernel& k = std::get<TunedKernel>(tuned);
  return Chosen{k.fn, k.config};
}

// =================================================================================================
// 3. THE GRID CUT TO THE WORK: a rule on the launch, never a field filled in.
// =================================================================================================

// THE TILES THERE ARE, one block's work each: this rank's rows (a row
// two-shot's share, every row otherwise) in TILE_M, by the grid's `grid_cols`
// columns in TILE_N. A grid wider is idle blocks, each still paying every
// barrier (the pull norm at 32 tokens ran 36 blocks for 4 rows a rank).
constexpr int64_t tiles_of(Template t, const KernelConfig& c, int64_t m,
                           int64_t grid_cols, int world) {
  const bool row_split = is_two_shot(t) && !slices_columns(t);
  const int64_t mine = row_split ? (m + world - 1) / world : m;
  const int64_t tm = tile_m_of(c.build), tn = tile_n_of(c.build);
  return (mine + tm - 1) / tm * ((grid_cols + tn - 1) / tn);
}

// A TILED OP'S GRID, at most the tiles there are (at least one block). THE
// FUSED ATTNRES PULL'S GRID ALSO DIVIDES ITS TILES: each block's most tiles
// kept, the blocks cut until every block but the last has that many (384 rows
// on 256 blocks left half the blocks one row: 32.2 against 27.7 us on 192,
// 2026-10-05T03-45-51Z). The other templates' not yet measured. The plain
// all-reduce runs its grid as given: its tuned grids were measured as they run.
constexpr Chosen cut_to_the_work(Chosen k, int64_t m, int64_t grid_cols,
                                 int world) {
  const int64_t tiles = tiles_of(k.fn, k.config, m, grid_cols, world);
  int& blocks = k.config.launch.blocks_per_grid;
  if (tiles < blocks) blocks = static_cast<int>(tiles > 0 ? tiles : 1);
  if ((k.fn == Template::all_reduce_pull_two_shot_add_attn_res_rms_norm || is_split(k.fn)) &&
      tiles > 0) {
    const int64_t most = (tiles + blocks - 1) / blocks;
    blocks = static_cast<int>((tiles + most - 1) / most);
  }
  return k;
}

// =================================================================================================
// 4. THE CHECK: the first Error kernel `fn` at `c` meets on the call here, in
// this order, or none. Every Error is a capability (a kernel that cannot take
// the input), never "the unfused ops would be faster". Residency, which needs
// the compiled kernel, is each select's last check.
// =================================================================================================

// A KERNEL'S PARTIALS, row-major (a build in passes needs none: it runs in passes
// of what the partials holds): the plain two-shot its slice of every row; a
// column two-shot the whole reduced tensor (each rank's columns at their
// place); a row two-shot its rank's rows, twice where it leaves two results
// (out and the residual). A one-shot reads the inputs and keeps nothing. THE
// ADD-RMSNORM ROW TWO-SHOT's rows each carry their scale in a pack before the
// row (kernels/all_reduce_add_rms_norm.cuh, its exact body: a row with its
// scale, one buffer), a pack a row more.
constexpr int64_t kScaleSlotPacks = 1;
constexpr int64_t partials_need(Template t, int64_t m, int64_t packs,
                               int world) {
  if (!is_two_shot(t)) return 0;
  const OpType o = op_of(t);
  if (o == OpType::all_reduce)
    return m * ((packs + world - 1) / world) * kBuild.memory.pack_bytes;
  if (slices_columns(t)) return m * packs * kBuild.memory.pack_bytes;
  const int64_t row_packs =
      t == Template::all_reduce_pull_two_shot_add_rms_norm ? packs + kScaleSlotPacks : packs;
  return (m + world - 1) / world * row_packs * kBuild.memory.pack_bytes;
}

// A LAYOUT THE PACKED LOADS CAN READ: every buffer's innermost stride 1 and its
// rows starting on whole packs ({stride_m, stride_n} a buffer, in elements of
// `dtype`; a vector's stride_m 0).
inline std::optional<Error> layout_refused(
    DType dtype, std::initializer_list<std::pair<int64_t, int64_t>> strides) {
  for (const auto& [m, n] : strides) {
    if (n != 1) return Error::inner_stride_not_one;
    if (m * elem_bytes(dtype) % kBuild.memory.pack_bytes != 0)
      return Error::row_stride_not_packs;
  }
  return std::nullopt;
}

// `row_elems`: the row the kernel reduces, in elements; `own`: the op's own
// Error on its own arguments, checked after the dtype. With no Handle the
// checks that need peers are left out.
inline std::optional<Error> refused(const Handle* h, Template fn,
                                    const KernelConfig& c, DType dtype,
                                    int64_t m, int64_t n, int64_t tile_cols,
                                    std::optional<Error> own_error,
                                    bool in_passes) {
  const int world = h ? h->world_size() : 1;
  if (h && !world_built(world)) return Error::world_not_built;
  if (!dtype_built(dtype)) return Error::dtype_not_built;
  if (own_error) return own_error;
  if (n * elem_bytes(dtype) % kBuild.memory.pack_bytes != 0)
    return Error::row_not_packs;
  // INT4 ON A ONE-SHOT, which reads its peers' inputs as they are: named before
  // the build lookup, which lists no such build.
  const bool int4 = quantization_of(c.build) == Quantization::int4;
  if (int4 && algorithm_of(fn) == Algorithm::one_shot)
    return Error::quantization_not_two_shot;
  // A BUILD THE TEMPLATE DOES NOT LIST, named by what differs: its threads, its
  // waves, its tile.
  const int threads = threads_of(c.build);
  if (c.build.index() != family_of(fn)) return Error::tile_not_built;
  if (!built_at(fn, threads)) return Error::threads_not_built;
  if (!built_at(fn, threads, waves_of(c.build))) return Error::waves_not_built;
  if (!built(fn, c.build)) return Error::tile_not_built;
  if (tile_n_of(c.build) < tile_cols) return Error::row_too_wide;
  if (!h) return std::nullopt;
  const int64_t packs = packs_of(n, dtype);
  // A TWO-SHOT ROW CUT INTO A SLICE A RANK needs a pack for every rank: with
  // fewer, the ranks past the row own empty slices, which the kernels do not
  // handle (a GPU memory fault, 2026-10-10, at 8 bf16 columns on 8 ranks).
  // Refused, so the caller takes the one-shot or RCCL.
  if (!every_rank_gets_columns(fn, n, dtype, world))
    return Error::row_narrower_than_world;
  // AN INT4 ROW is whole groups for every rank (quantizes's rule, which a
  // forced call meets here).
  if (int4 && !int4_takes_row(n, world)) return Error::row_not_int4_groups;
  // An INT4 payload lies in the region that holds it exact (common.cuh's
  // INT4 payload layout): the input's in the inputs, the sums' in the partials, at
  // most one row of the partials's elements (its every view's bound).
  if (int4 && int4_payload_bytes(staged_rows(fn, Quantization::int4, m, world), n) >
                  h->staging_bytes())
    return Error::inputs_too_small;
  const int64_t partials = partials_need(fn, m, packs, world);
  if (int4 && int4_payload_bytes(1, partials / elem_bytes(dtype)) >
                  h->partials_bytes())
    return Error::partials_too_small;
  if (!in_passes && partials > h->partials_bytes())
    return Error::partials_too_small;
  // A BUILD THAT DOES NOT RUN IN PASSES has its input in the inputs whole:
  // an eager exact call's host copy (handle.cuh's INPUT TABLES; a captured
  // one's is read where it is, refused here alike, so the eager and captured
  // calls of one shape agree), or an INT4 call's quantized payload (the INT4
  // all-reduce: pushed).
  const int64_t bytes = m * n * elem_bytes(dtype);
  if (!in_passes && bytes > h->staging_bytes())
    return Error::inputs_too_small;
  return std::nullopt;
}

// THE WEIGHT a norm takes: its input's dtype, or one of its op's `weights`.
constexpr std::optional<Error> weight_refused(OpType o, DType dtype,
                                              DType weight_dtype) {
  if (weight_dtype == dtype) return std::nullopt;
  for (const DType w : op(o).weights)
    if (weight_dtype == w) return std::nullopt;
  return Error::weight_not_built;
}

// THE COLUMNS AN OP'S TILE MUST HOLD on a row of `n`: the row where its tile
// holds it, else none.
constexpr int64_t tile_cols(OpType o, int64_t n) {
  return op(o).tile_holds_row ? n : 0;
}

// THE TUNED KERNEL AS A QUERY (`build_info().tuned`): what select launches for
// `m` rows of `n` at `world` ranks in `dtype`, picked as `chosen` picks (`pick`)
// from the op's list of `quantization` (its INT4 list for int4 at any bytes; a
// plain call reads that list only from its crossover up, `quantizes`), among the
// entries of template `fn` where one is named (a forced launch: select's pick at
// that template), its grid cut to the work as the op's select cuts it (every op
// but the plain all-reduce, which runs its grid as given). None where that list
// has no entry of `fn`. A function of its arguments alone: no handle, no device.
inline std::variant<std::optional<Chosen>, Error> tuned(
    OpType o, int world, int64_t m, int64_t n, DType dtype, int quantization,
    std::optional<Template> fn) {
  const std::span<const TunedKernel> listed =
      quantization == Quantization::int4 ? op(o).int4_kernels : op(o).kernels;
  std::vector<TunedKernel> of_fn;
  for (const TunedKernel& k : listed)
    if (!fn || k.fn == *fn) of_fn.push_back(k);
  if (of_fn.empty()) return std::optional<Chosen>{};
  const std::variant<TunedKernel, Error> picked =
      pick(o, world, m, n, tile_cols(o, n), dtype, of_fn);
  if (const Error* e = std::get_if<Error>(&picked)) return *e;
  const TunedKernel& k = std::get<TunedKernel>(picked);
  const Chosen c{k.fn, k.config};
  return std::optional<Chosen>{
      o == OpType::all_reduce ? c : cut_to_the_work(c, m, n, world)};
}

// =================================================================================================
// EACH OP'S SELECT: the op's launch, everything decided, from the call's
// primitives and its forcing, or the first Error the call meets. Each in four
// steps: 1. the kernel (the forced template and config, else the op's tuned
// kernel, its grid cut to the work), 2. the checks, 3. its compiled
// instance, 4. the launch.
// =================================================================================================

// all_reduce: the sum of every rank's `inp` ([rows, cols] of `dtype` at its
// strides) into `out`. PULLED IN PLACE (the default, exact): the kernel reads
// every rank's input through the input table (handle.cuh's INPUT TABLES), a
// captured call's tensors where they are, an eager call's host copies in the
// inputs region, which must hold it whole; the two-shot a band of rows a pass,
// what its partials hold. INT4 (the pull two-shot's INT4 build): the shares
// quantized and pushed whole. PUSHED (forced): the input pushed into the peers'
// inboxes, a band a pass.
inline std::variant<AllReduceLaunch, Error> select_all_reduce(
    const Handle& h, const void* input, int64_t input_stride_m,
    int64_t input_stride_n, int64_t input_size_m, int64_t input_size_n,
    void* output, int64_t output_stride_m, int64_t output_stride_n,
    int64_t output_size_m, int64_t output_size_n, DType dtype,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_m, std::optional<int> tile_n,
    std::optional<int> threads_per_block, std::optional<int> blocks_per_grid,
    std::optional<int> waves_per_eu, std::optional<int> quantization,
    hipStream_t stream) {
  const int64_t m = input_size_m,
                n = input_size_n;  // the problem's shape: the input's
  const int world = h.world_size();
  // 1. The tile does not hold the row: the all-reduce tiles across it.
  const std::variant<Chosen, Error> picked = chosen(
      h,
      forced(OpType::all_reduce, algorithm, direction,
             {threads_per_block, waves_per_eu, tile_m, tile_n}, quantization,
             blocks_per_grid, std::nullopt,
             [&] {
               return AllReduceBuildConfig{*threads_per_block, *waves_per_eu,
                                           *tile_m, *tile_n};
             }),
      OpType::all_reduce, m, n, tile_cols(OpType::all_reduce, n), dtype);
  if (const Error* e = std::get_if<Error>(&picked)) return *e;
  const Chosen k = std::get<Chosen>(picked);
  const Template fn = k.fn;
  const KernelConfig c = k.config;
  const AllReduceBuildConfig& a = std::get<AllReduceBuildConfig>(c.build);
  // IN PLACE: the exact pull, every rank's input read where its table says.
  const bool in_place = direction_of(fn) == Direction::pull &&
                        a.quantization == Quantization::none;
  // IN PASSES (none): a band of rows a pass (the in-place one-shot's one band,
  // the whole input). INT4 pushes its input's shares quantized, whole,
  // instead.
  const bool in_passes = a.quantization == Quantization::none;
  // THE BAND: the rows a pass, what the inbox (the inputs region) holds in
  // WORLD slots. The one-shot's slot is the band's rows, [WORLD * band, n];
  // the two-shot's is one rank's share of the band (the flat walk's, whole
  // packs, rounded up), [WORLD, slot], whose band the partials also hold (a
  // row of its slice a row). Rounding up costs a two-shot at most one pack a
  // slot, so the band one row less than the inputs' rows always fits.
  const int64_t row_bytes = n * elem_bytes(dtype);
  const int64_t pack = kBuild.memory.pack_bytes;
  const int64_t packs = n * elem_bytes(dtype) / pack;
  const int64_t slice_bytes = (packs + world - 1) / world * pack;
  const auto slot_packs = [&](int64_t rows) {
    return (rows * packs + world - 1) / world;
  };
  int64_t band = 0;
  if (in_place && !is_two_shot(fn)) {
    band = m;  // nothing staged: one pass
  } else if (in_place) {
    // THE IN-PLACE TWO-SHOT'S BAND: what the partials hold, a row of its
    // slice a row; the input is read where it is.
    band = slice_bytes > 0 ? h.partials_bytes() / slice_bytes : 0;
  } else if (row_bytes > 0 && !is_two_shot(fn)) {
    band = h.staging_bytes() / (world * row_bytes);
  } else if (row_bytes > 0) {
    band = h.staging_bytes() / row_bytes;
    if (band > 0 && world * slot_packs(band) * pack > h.staging_bytes()) --band;
    if (slice_bytes > 0 && h.partials_bytes() / slice_bytes < band)
      band = h.partials_bytes() / slice_bytes;
  }
  const int64_t rows = band < m ? band : m;  // a pass's rows
  // 2.
  if (const std::optional<Error> e =
          refused(&h, fn, c, dtype, m, n, tile_cols(OpType::all_reduce, n),
                  std::nullopt, in_passes))
    return *e;
  if (in_place && is_two_shot(fn) && band < 1)
    return Error::partials_too_small;
  if (in_passes && band < 1) return Error::inputs_too_small;
  // AN EAGER IN-PLACE CALL'S INPUT IS COPIED WHOLE into the inputs region
  // (its table, handle.cuh's INPUT TABLES); a captured one is read where it
  // is, registered at capture exit.
  if (in_place) {
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    HIP_CHECK(hipStreamIsCapturing(stream, &capture));
    if (capture != hipStreamCaptureStatusActive &&
        m * n * elem_bytes(dtype) > h.staging_bytes())
      return Error::inputs_too_small;
  }
  // 3. Only a tile its threads lay out whole at this world: TILE_N whole groups
  // for every thread of
  //    a tile row (the two-shot's rows are the ranks).
  const void* kernel = nullptr;
  by_world(world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    by_dtype(dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      constexpr int NL = traits<T>::N;
      switch (fn) {
        case Template::all_reduce_pull_one_shot:
          by_build<Template::all_reduce_pull_one_shot>(
              c.build, [&](auto built) {
                constexpr AllReduceBuildConfig C = decltype(built)::value;
                constexpr int THREADS = C.threads_per_block,
                              WAVES = C.waves_per_eu;
                constexpr int TM = C.tile_m, TN = C.tile_n;
                if constexpr (one_shot_lays_out(C, NL))
                  kernel = instance<AllReduceOneShotInPlaceKernel>(
                      all_reduce_pull_one_shot<T, NG, TM, TN, THREADS,
                                                        WAVES>);
              });
          return;
        case Template::all_reduce_push_one_shot:
          by_build<Template::all_reduce_push_one_shot>(
              c.build, [&](auto built) {
                constexpr AllReduceBuildConfig C = decltype(built)::value;
                constexpr int THREADS = C.threads_per_block,
                              WAVES = C.waves_per_eu;
                constexpr int TM = C.tile_m, TN = C.tile_n;
                if constexpr (one_shot_lays_out(C, NL))
                  kernel = instance<AllReduceOneShotPushedKernel>(
                      all_reduce_push_one_shot<T, NG, TM, TN, THREADS, WAVES>);
              });
          return;
        case Template::all_reduce_push_two_shot:
          by_build<Template::all_reduce_push_two_shot>(c.build, [&](auto
                                                                        built) {
            constexpr AllReduceBuildConfig C = decltype(built)::value;
            constexpr int THREADS = C.threads_per_block, WAVES = C.waves_per_eu;
            constexpr int TM = C.tile_m, TN = C.tile_n;
            const bool contiguous = input_stride_m == n * input_stride_n &&
                                    output_stride_m == n * output_stride_n;
            if constexpr (two_shot_lays_out(C, NG, NL)) {
              const auto at = [&](auto layout) {
                constexpr bool CONTIGUOUS = decltype(layout)::value;
                if constexpr (C.quantization == Quantization::none)
                  return instance<AllReduceTwoShotPushedKernel>(
                      all_reduce_push_two_shot<T, NG, TM, TN, THREADS, WAVES,
                                               CONTIGUOUS>);
                else
                  return instance<AllReduceTwoShotKernel>(
                      all_reduce_pull_two_shot_int4<T, NG, TM, TN, THREADS,
                                                    WAVES, C.quantization,
                                                    CONTIGUOUS>);
              };
              kernel =
                  contiguous ? at(std::true_type{}) : at(std::false_type{});
            }
          });
          return;
        case Template::all_reduce_pull_two_shot:
          by_build<Template::all_reduce_pull_two_shot>(c.build, [&](auto
                                                                        built) {
            constexpr AllReduceBuildConfig C = decltype(built)::value;
            constexpr int THREADS = C.threads_per_block, WAVES = C.waves_per_eu;
            constexpr int TM = C.tile_m, TN = C.tile_n;
            // THE INSTANCE FOR THE LAYOUT AS GIVEN: rows back to back read each
            // peer's share as one run, other strides the strided walk
            // (reduce_scatter.cuh's CONTIGUOUS); the input is not reshaped. In
            // place, every rank's input is read at the table's strides: a
            // captured call's are the caller's (every rank's alike), an eager
            // call's its copy's, dense rows, so the input's rows back to back
            // answer for both.
            const bool contiguous = input_stride_m == n * input_stride_n &&
                                    output_stride_m == n * output_stride_n;
            if constexpr (two_shot_lays_out(C, NG, NL)) {
              const auto at = [&](auto layout) {
                constexpr bool CONTIGUOUS = decltype(layout)::value;
                if constexpr (C.quantization == Quantization::none)
                  return instance<AllReduceTwoShotInPlaceKernel>(
                      all_reduce_pull_two_shot<T, NG, TM, TN, THREADS, WAVES,
                                               CONTIGUOUS>);
                else
                  return instance<AllReduceTwoShotKernel>(
                      all_reduce_pull_two_shot_int4<T, NG, TM, TN, THREADS,
                                                    WAVES, C.quantization,
                                                    CONTIGUOUS>);
              };
              kernel =
                  contiguous ? at(std::true_type{}) : at(std::false_type{});
            }
          });
          return;
        default:
          not_this_ops(fn);
      }
    });
  });
  if (kernel == nullptr) return Error::tile_not_built;
  // 4.
  const AllReduceLaunch l{
      .kernel = kernel,
      .algorithm = algorithm_of(fn),
      .direction = direction_of(fn),
      .world = world,
      .tile_m = a.tile_m,
      .tile_n = a.tile_n,
      .threads_per_block = a.threads_per_block,
      .blocks_per_grid = c.launch.blocks_per_grid,
      .waves_per_eu = a.waves_per_eu,
      .quantization = a.quantization,
      .stream = stream,
      .input = input,
      .input_stride_m = input_stride_m,
      .input_stride_n = input_stride_n,
      .input_size_m = m,
      .input_size_n = n,
      .output = output,
      .output_stride_m = output_stride_m,
      .output_stride_n = output_stride_n,
      .output_size_m = output_size_m,
      .output_size_n = output_size_n,
      .partials_stride_m = slice_bytes / elem_bytes(dtype),
      .partials_stride_n = 1,
      .partials_size_m = in_passes ? rows : m,
      .partials_size_n = slice_bytes / elem_bytes(dtype),
      // THE INBOX: the one-shot's [WORLD * rows, n], the exact two-shot's
      // [WORLD, slot] (a slot one rank's share of a band), the INT4 two-shot's
      // payload found from the problem's [m, n] (its slots where the shares lie).
      // IN PLACE: none (the table's strides are the launch's, handle.cuh's
      // InputTable).
      .staging_stride_m = in_place                         ? 0
                         : !in_passes || !is_two_shot(fn) ? n
                             : slot_packs(rows) * pack / elem_bytes(dtype),
      .staging_stride_n = in_place ? 0 : 1,
      .staging_size_m = in_place     ? 0
                       : !in_passes ? m
                       : is_two_shot(fn) ? world
                                         : world * rows,
      .staging_size_n = in_place                         ? 0
                       : !in_passes || !is_two_shot(fn) ? n
                           : slot_packs(rows) * pack / elem_bytes(dtype),
      .dtype = dtype};
  if (!resident(h, kernel, l.blocks_per_grid, l.threads_per_block))
    return Error::grid_not_resident;
  return l;
}

// A SPLIT'S ALL-REDUCE (Algorithm::split): the plain all-reduce's tuned kernel at the split
// build's `quantization` (its INT4 list or its exact one), forced whole, into this rank's local
// sums (Handle::sums), dense rows; its own select checks it (its band, its regions, its
// residency). The caller's own check holds the sums to the inputs region's size.
inline std::variant<AllReduceLaunch, Error> split_all_reduce(
    const Handle& h, const void* input, int64_t input_stride_m, int64_t input_stride_n,
    int64_t m, int64_t n, DType dtype, int quantization, hipStream_t stream) {
  const Op& plain = op(OpType::all_reduce);
  const std::variant<TunedKernel, Error> t =
      pick(OpType::all_reduce, h.world_size(), m, n, tile_cols(OpType::all_reduce, n), dtype,
           quantization == Quantization::int4 ? plain.int4_kernels : plain.kernels);
  if (const Error* e = std::get_if<Error>(&t)) return *e;
  const TunedKernel& r = std::get<TunedKernel>(t);
  const AllReduceBuildConfig& b = std::get<AllReduceBuildConfig>(r.config.build);
  return select_all_reduce(h, input, input_stride_m, input_stride_n, m, n, h.sums(), n, 1, m, n,
                           dtype, algorithm_of(r.fn), direction_of(r.fn), b.tile_m, b.tile_n,
                           b.threads_per_block, r.config.launch.blocks_per_grid, b.waves_per_eu,
                           b.quantization, stream);
}

// all_reduce_rms_norm: out = rms_norm(all_reduce(inp), weight), vLLM's
// roundings exactly. inp [rows, hidden]; the weight `weight_dtype`, dtype or
// f32.
inline std::variant<AllReduceRmsNormLaunch, Error> select_all_reduce_rms_norm(
    const Handle& h, const void* input, int64_t input_stride_m,
    int64_t input_stride_n, int64_t input_size_m, int64_t input_size_n,
    const void* weight, int64_t weight_stride_n, int64_t weight_size_n,
    void* output, int64_t output_stride_m, int64_t output_stride_n,
    int64_t output_size_m, int64_t output_size_n, DType dtype,
    DType weight_dtype, float eps, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> tile_m,
    std::optional<int> tile_n, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, std::optional<int> waves_per_eu,
    std::optional<int> quantization, hipStream_t stream) {
  const int64_t m = input_size_m,
                n = input_size_n;  // the problem's shape: the input's
  using K = Template;
  const OpType op = OpType::all_reduce_rms_norm;
  const int world = h.world_size();
  // 1.
  const std::variant<Chosen, Error> picked =
      chosen(h,
             forced(op, algorithm, direction,
                    {threads_per_block, waves_per_eu, tile_m, tile_n},
                    quantization, blocks_per_grid, std::nullopt,
                    [&] {
                      return AllReduceRmsNormBuildConfig{
                          *threads_per_block, *waves_per_eu, *tile_m, *tile_n};
                    }),
             op, m, n, tile_cols(op, n), dtype);
  if (const Error* e = std::get_if<Error>(&picked)) return *e;
  const Chosen k = cut_to_the_work(std::get<Chosen>(picked), m, n, world);
  const Template fn = k.fn;
  const KernelConfig& c = k.config;
  // 2.
  if (const std::optional<Error> e = refused(
          &h, fn, c, dtype, m, n, tile_cols(op, n),
          weight_refused(op, dtype, weight_dtype), false))
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
            return by_build<K::all_reduce_pull_one_shot_rms_norm>(
                c.build, [&](auto cc) {
                  constexpr AllReduceRmsNormBuildConfig C = decltype(cc)::value;
                  constexpr int TN = C.tile_n, TPB = C.threads_per_block,
                                WPE = C.waves_per_eu;
                  kernel = instance<AllReduceRmsNormOneShotKernel>(
                      all_reduce_pull_one_shot_rms_norm<T, W, NG, TN, TPB,
                                                        WPE>);
                });
          case K::all_reduce_pull_two_shot_rms_norm:
            return by_build<K::all_reduce_pull_two_shot_rms_norm>(
                c.build, [&](auto cc) {
                  constexpr AllReduceRmsNormBuildConfig C = decltype(cc)::value;
                  constexpr int TN = C.tile_n, TPB = C.threads_per_block,
                                WPE = C.waves_per_eu;
                  kernel = instance<AllReduceRmsNormTwoShotKernel>(
                      all_reduce_pull_two_shot_rms_norm<T, W, NG, TN, TPB,
                                                        WPE, C.quantization>);
                });
          case K::all_reduce_push_two_shot_rms_norm:
            return by_build<K::all_reduce_push_two_shot_rms_norm>(
                c.build, [&](auto cc) {
                  constexpr AllReduceRmsNormBuildConfig C = decltype(cc)::value;
                  constexpr int TN = C.tile_n, TPB = C.threads_per_block,
                                WPE = C.waves_per_eu;
                  kernel = instance<AllReduceRmsNormTwoShotKernel>(
                      all_reduce_push_two_shot_rms_norm<T, W, NG, TN, TPB,
                                                        WPE, C.quantization>);
                });
          case K::all_reduce_pull_split_rms_norm:
            return by_build<K::all_reduce_pull_split_rms_norm>(
                c.build, [&](auto cc) {
                  constexpr AllReduceRmsNormBuildConfig C = decltype(cc)::value;
                  constexpr int TM = C.tile_m, TN = C.tile_n, TPB = C.threads_per_block,
                                WPE = C.waves_per_eu;
                  kernel = instance<RmsNormRowsKernel>(rms_norm_rows<T, W, TM, TN, TPB, WPE>);
                });
          default:
            not_this_ops(fn);
        }
      });
    });
  });
  const AllReduceRmsNormBuildConfig& r =
      std::get<AllReduceRmsNormBuildConfig>(c.build);
  // THE SPLIT'S ALL-REDUCE (split_all_reduce).
  std::optional<AllReduceLaunch> all_reduce;
  if (is_split(fn)) {
    const std::variant<AllReduceLaunch, Error> got = split_all_reduce(
        h, input, input_stride_m, input_stride_n, m, n, dtype, r.quantization, stream);
    if (const Error* e = std::get_if<Error>(&got)) return *e;
    all_reduce = std::get<AllReduceLaunch>(got);
  }
  // 4.
  // THE PARTIALS'S ROWS, what partials_need sized it for (a one-shot keeps none).
  const int64_t partials_rows =
      n > 0 ? partials_need(fn, m,
                           n * elem_bytes(dtype) / kBuild.memory.pack_bytes,
                           world) /
                  (n * elem_bytes(dtype))
            : 0;
  const AllReduceRmsNormLaunch l{.kernel = kernel,
                                 .algorithm = algorithm_of(fn),
                                 .direction = direction_of(fn),
                                 .world = world,
                                 .tile_m = r.tile_m,
                                 .tile_n = r.tile_n,
                                 .threads_per_block = r.threads_per_block,
                                 .blocks_per_grid = c.launch.blocks_per_grid,
                                 .waves_per_eu = r.waves_per_eu,
                                 .quantization = r.quantization,
                                 .stream = stream,
                                 .input = input,
                                 .input_stride_m = input_stride_m,
                                 .input_stride_n = input_stride_n,
                                 .input_size_m = m,
                                 .input_size_n = n,
                                 .weight = weight,
                                 .weight_stride_n = weight_stride_n,
                                 .weight_size_n = weight_size_n,
                                 .output = output,
                                 .output_stride_m = output_stride_m,
                                 .output_stride_n = output_stride_n,
                                 .output_size_m = output_size_m,
                                 .output_size_n = output_size_n,
                                 .partials_stride_m = n,
                                 .partials_stride_n = 1,
                                 .partials_size_m = partials_rows,
                                 .partials_size_n = n,
                               .staging_stride_m = n,
                               .staging_stride_n = 1,
                               .staging_size_m = staged_rows(fn, r.quantization, m, world),
                               .staging_size_n = n,
                                 .dtype = dtype,
                                 .weight_dtype = weight_dtype,
                                 .eps = eps,
                                 .all_reduce = all_reduce};
  // A SPLIT'S NORM KERNEL waits on no peer, so any grid runs: residency is its all-reduce's.
  if (!is_split(fn) && !resident(h, kernel, l.blocks_per_grid, l.threads_per_block))
    return Error::grid_not_resident;
  return l;
}

// all_reduce_add_rms_norm: out, residual_out =
// fused_add_rms_norm(all_reduce(inp), residual, weight), vLLM's roundings
// exactly.
inline std::variant<AllReduceAddRmsNormLaunch, Error>
select_all_reduce_add_rms_norm(
    const Handle& h, const void* input, int64_t input_stride_m,
    int64_t input_stride_n, int64_t input_size_m, int64_t input_size_n,
    const void* residual, int64_t residual_stride_m, int64_t residual_stride_n,
    int64_t residual_size_m, int64_t residual_size_n, const void* weight,
    int64_t weight_stride_n, int64_t weight_size_n, void* output,
    int64_t output_stride_m, int64_t output_stride_n, int64_t output_size_m,
    int64_t output_size_n, void* residual_output,
    int64_t residual_output_stride_m, int64_t residual_output_stride_n,
    int64_t residual_output_size_m, int64_t residual_output_size_n, DType dtype,
    DType weight_dtype, float eps, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> tile_m,
    std::optional<int> tile_n, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, std::optional<int> waves_per_eu,
    std::optional<int> quantization, hipStream_t stream) {
  const int64_t m = input_size_m,
                n = input_size_n;  // the problem's shape: the input's
  using K = Template;
  const OpType op = OpType::all_reduce_add_rms_norm;
  const int world = h.world_size();
  // 1.
  const std::variant<Chosen, Error> picked =
      chosen(h,
             forced(op, algorithm, direction,
                    {threads_per_block, waves_per_eu, tile_m, tile_n},
                    quantization, blocks_per_grid, std::nullopt,
                    [&] {
                      return AllReduceAddRmsNormBuildConfig{
                          *threads_per_block, *waves_per_eu, *tile_m, *tile_n};
                    }),
             op, m, n, tile_cols(op, n), dtype);
  if (const Error* e = std::get_if<Error>(&picked)) return *e;
  const Chosen k = cut_to_the_work(std::get<Chosen>(picked), m, n, world);
  const Template fn = k.fn;
  const KernelConfig& c = k.config;
  // 2.
  if (const std::optional<Error> e = refused(
          &h, fn, c, dtype, m, n, tile_cols(op, n),
          weight_refused(op, dtype, weight_dtype), false))
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
            return by_build<K::all_reduce_pull_one_shot_add_rms_norm>(
                c.build, [&](auto cc) {
                  constexpr AllReduceAddRmsNormBuildConfig C =
                      decltype(cc)::value;
                  constexpr int TN = C.tile_n, TPB = C.threads_per_block,
                                WPE = C.waves_per_eu;
                  kernel = instance<AllReduceAddRmsNormOneShotKernel>(
                      all_reduce_pull_one_shot_add_rms_norm<T, W, NG, TN, TPB,
                                                            WPE>);
                });
          case K::all_reduce_pull_two_shot_add_rms_norm:
            return by_build<K::all_reduce_pull_two_shot_add_rms_norm>(
                c.build, [&](auto cc) {
                  constexpr AllReduceAddRmsNormBuildConfig C =
                      decltype(cc)::value;
                  constexpr int TN = C.tile_n, TPB = C.threads_per_block,
                                WPE = C.waves_per_eu;
                  kernel = instance<AllReduceAddRmsNormTwoShotKernel>(
                      all_reduce_pull_two_shot_add_rms_norm<T, W, NG, TN, TPB,
                                                            WPE, C.quantization>);
                });
          case K::all_reduce_push_two_shot_add_rms_norm:
            return by_build<K::all_reduce_push_two_shot_add_rms_norm>(
                c.build, [&](auto cc) {
                  constexpr AllReduceAddRmsNormBuildConfig C =
                      decltype(cc)::value;
                  constexpr int TN = C.tile_n, TPB = C.threads_per_block,
                                WPE = C.waves_per_eu;
                  kernel = instance<AllReduceAddRmsNormTwoShotKernel>(
                      all_reduce_push_two_shot_add_rms_norm<T, W, NG, TN, TPB,
                                                            WPE, C.quantization>);
                });
          case K::all_reduce_pull_split_add_rms_norm:
            return by_build<K::all_reduce_pull_split_add_rms_norm>(
                c.build, [&](auto cc) {
                  constexpr AllReduceAddRmsNormBuildConfig C = decltype(cc)::value;
                  constexpr int TM = C.tile_m, TN = C.tile_n, TPB = C.threads_per_block,
                                WPE = C.waves_per_eu;
                  kernel =
                      instance<AddRmsNormRowsKernel>(add_rms_norm_rows<T, W, TM, TN, TPB, WPE>);
                });
          default:
            not_this_ops(fn);
        }
      });
    });
  });
  const AllReduceAddRmsNormBuildConfig& r =
      std::get<AllReduceAddRmsNormBuildConfig>(c.build);
  // THE SPLIT'S ALL-REDUCE (split_all_reduce).
  std::optional<AllReduceLaunch> all_reduce;
  if (is_split(fn)) {
    const std::variant<AllReduceLaunch, Error> got = split_all_reduce(
        h, input, input_stride_m, input_stride_n, m, n, dtype, r.quantization, stream);
    if (const Error* e = std::get_if<Error>(&got)) return *e;
    all_reduce = std::get<AllReduceLaunch>(got);
  }
  // 4.
  // THE PARTIALS'S ROWS, what partials_need sized it for (a one-shot keeps none),
  // at its row stride: the exact pull two-shot's row is its scale's pack then the
  // row (partials_need); the INT4 build's rows are the row alone, its scales in
  // the scales region.
  const bool scale_slot = fn == Template::all_reduce_pull_two_shot_add_rms_norm &&
                          r.quantization == Quantization::none;
  const int64_t partials_stride_m =
      n + (scale_slot ? kScaleSlotPacks * kBuild.memory.pack_bytes / elem_bytes(dtype) : 0);
  const int64_t partials_rows =
      n > 0 ? partials_need(fn, m,
                           n * elem_bytes(dtype) / kBuild.memory.pack_bytes,
                           world) /
                  (partials_stride_m * elem_bytes(dtype))
            : 0;
  const AllReduceAddRmsNormLaunch l{
      .kernel = kernel,
      .algorithm = algorithm_of(fn),
      .direction = direction_of(fn),
      .world = world,
      .tile_m = r.tile_m,
      .tile_n = r.tile_n,
      .threads_per_block = r.threads_per_block,
      .blocks_per_grid = c.launch.blocks_per_grid,
      .waves_per_eu = r.waves_per_eu,
      .quantization = r.quantization,
      .stream = stream,
      .input = input,
      .input_stride_m = input_stride_m,
      .input_stride_n = input_stride_n,
      .input_size_m = m,
      .input_size_n = n,
      .residual = residual,
      .residual_stride_m = residual_stride_m,
      .residual_stride_n = residual_stride_n,
      .residual_size_m = residual_size_m,
      .residual_size_n = residual_size_n,
      .weight = weight,
      .weight_stride_n = weight_stride_n,
      .weight_size_n = weight_size_n,
      .output = output,
      .output_stride_m = output_stride_m,
      .output_stride_n = output_stride_n,
      .output_size_m = output_size_m,
      .output_size_n = output_size_n,
      .residual_output = residual_output,
      .residual_output_stride_m = residual_output_stride_m,
      .residual_output_stride_n = residual_output_stride_n,
      .residual_output_size_m = residual_output_size_m,
      .residual_output_size_n = residual_output_size_n,
      .partials_stride_m = partials_stride_m,
      .partials_stride_n = 1,
      .partials_size_m = partials_rows,
      .partials_size_n = partials_stride_m,
      .scales_stride_m = 1,
      .scales_size_m = fn == Template::all_reduce_pull_two_shot_add_rms_norm && !scale_slot
                           ? (m + world - 1) / world
                           : 0,
      .staging_stride_m = n,
      .staging_stride_n = 1,
      .staging_size_m = staged_rows(fn, r.quantization, m, world),
      .staging_size_n = n,
      .dtype = dtype,
      .weight_dtype = weight_dtype,
      .eps = eps,
      .all_reduce = all_reduce};
  if (l.scales_size_m > h.scales_size_m()) return Error::partials_too_small;
  // A SPLIT'S NORM KERNEL waits on no peer, so any grid runs: residency is its all-reduce's.
  if (!is_split(fn) && !resident(h, kernel, l.blocks_per_grid, l.threads_per_block))
    return Error::grid_not_resident;
  return l;
}

// all_reduce_add_attn_res_rms_norm: Kimi-K3's AttnRes and its RMSNorm on each
// row of the all-reduced sum of `input`. With a `prefix` the sum is added to
// it, into `prefix_output`; without (null), the sum IS the new prefix.
// `block_output`, when not null, gets the new prefix too. The pull two-shot's
// config family has TILE_M and its reduce-scatter blocks; the one-shot's and
// push's (a row a tile) have neither.
inline std::variant<AllReduceAddAttnResRmsNormLaunch, Error>
select_all_reduce_add_attn_res_rms_norm(
    const Handle& h, const void* input, int64_t input_stride_m,
    int64_t input_stride_n, int64_t input_size_m, int64_t input_size_n,
    const void* prefix, int64_t prefix_stride_m, int64_t prefix_stride_n,
    int64_t prefix_size_m, int64_t prefix_size_n, const void* blocks,
    int64_t blocks_stride_m, int64_t blocks_stride_r, int64_t blocks_stride_n,
    int64_t blocks_size_m, int64_t blocks_size_r, int64_t blocks_size_n,
    const void* norm_weight, int64_t norm_weight_stride_n,
    int64_t norm_weight_size_n, const void* qk_weight,
    int64_t qk_weight_stride_n, int64_t qk_weight_size_n,
    const void* output_norm_weight, int64_t output_norm_weight_stride_n,
    int64_t output_norm_weight_size_n, void* prefix_output,
    int64_t prefix_output_stride_m, int64_t prefix_output_stride_n,
    int64_t prefix_output_size_m, int64_t prefix_output_size_n, void* output,
    int64_t output_stride_m, int64_t output_stride_n, int64_t output_size_m,
    int64_t output_size_n, void* block_output, int64_t block_output_stride_m,
    int64_t block_output_stride_n, int64_t block_output_size_m,
    int64_t block_output_size_n, DType dtype, int num_blocks, float eps,
    float output_eps, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> tile_m,
    std::optional<int> tile_n, std::optional<int> tile_k,
    std::optional<int> reduce_scatter_blocks,
    std::optional<int> threads_per_block, std::optional<int> blocks_per_grid,
    std::optional<int> waves_per_eu, std::optional<int> quantization,
    hipStream_t stream) {
  const int64_t m = input_size_m,
                n = input_size_n;  // the problem's shape: the input's
  using K = Template;
  const OpType op = OpType::all_reduce_add_attn_res_rms_norm;
  const int world = h.world_size();
  // 1.
  const std::variant<Chosen, Error> picked =
      chosen(h,
             forced(op, algorithm, direction,
                    {threads_per_block, waves_per_eu, tile_m, tile_n, tile_k},
                    quantization, blocks_per_grid, reduce_scatter_blocks,
                    [&] {
                      return AllReduceAddAttnResRmsNormBuildConfig{
                          *threads_per_block, *waves_per_eu, *tile_m, *tile_n,
                          *tile_k};
                    }),
             op, m, n, tile_cols(op, n), dtype);
  if (const Error* e = std::get_if<Error>(&picked)) return *e;
  const Chosen k = cut_to_the_work(std::get<Chosen>(picked), m, n, world);
  const Template fn = k.fn;
  const KernelConfig& c = k.config;
  // 2.
  if (const std::optional<Error> e =
          refused(&h, fn, c, dtype, m, n, tile_cols(op, n), std::nullopt,
                  false))
    return *e;
  // 3.
  const void* kernel = nullptr;
  by_world(world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    by_dtype(dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      switch (fn) {
        case K::all_reduce_pull_one_shot_add_attn_res_rms_norm:
          return by_build<K::all_reduce_pull_one_shot_add_attn_res_rms_norm>(
              c.build, [&](auto cc) {
                constexpr AllReduceAddAttnResRmsNormBuildConfig C =
                    decltype(cc)::value;
                constexpr int TN = C.tile_n, TK = C.tile_k,
                              TPB = C.threads_per_block, WPE = C.waves_per_eu;
                kernel =
                    instance<AllReduceAddAttnResRmsNormOneShotKernel>(
                              all_reduce_pull_one_shot_add_attn_res_rms_norm<
                                  T, NG, TN, TK, TPB, WPE>);
              });
        case K::all_reduce_pull_two_shot_add_attn_res_rms_norm:
          return by_build<K::all_reduce_pull_two_shot_add_attn_res_rms_norm>(
              c.build, [&](auto cc) {
                constexpr AllReduceAddAttnResRmsNormBuildConfig C =
                    decltype(cc)::value;
                constexpr int TM = C.tile_m, TN = C.tile_n, TK = C.tile_k,
                              TPB = C.threads_per_block, WPE = C.waves_per_eu;
                kernel =
                    instance<AllReduceAddAttnResRmsNormPullKernel>(
                              all_reduce_pull_two_shot_add_attn_res_rms_norm<
                                  T, NG, TM, TN, TK, TPB, WPE,
                                  C.quantization>);
              });
        case K::all_reduce_push_two_shot_add_attn_res_rms_norm:
          return by_build<K::all_reduce_push_two_shot_add_attn_res_rms_norm>(
              c.build, [&](auto cc) {
                constexpr AllReduceAddAttnResRmsNormBuildConfig C =
                    decltype(cc)::value;
                constexpr int TN = C.tile_n, TK = C.tile_k,
                              TPB = C.threads_per_block, WPE = C.waves_per_eu;
                kernel =
                    instance<AllReduceAddAttnResRmsNormPushKernel>(
                              all_reduce_push_two_shot_add_attn_res_rms_norm<
                                  T, NG, TN, TK, TPB, WPE,
                                  C.quantization>);
              });
        case K::all_reduce_pull_split_add_attn_res_rms_norm:
          // OUR SEPARATE ATTNRES KERNEL: no world, no quantization (its all-reduce's).
          return by_build<K::all_reduce_pull_split_add_attn_res_rms_norm>(
              c.build, [&](auto cc) {
                constexpr AllReduceAddAttnResRmsNormBuildConfig C =
                    decltype(cc)::value;
                constexpr int TM = C.tile_m, TN = C.tile_n, TK = C.tile_k,
                              TPB = C.threads_per_block, WPE = C.waves_per_eu;
                kernel = instance<AddAttnResRmsNormKernel>(
                    add_attn_res_rms_norm_rows<T, TM, TN, TK, TPB, WPE>);
              });
        default:
          not_this_ops(fn);
      }
    });
  });
  const AllReduceAddAttnResRmsNormBuildConfig& a =
      std::get<AllReduceAddAttnResRmsNormBuildConfig>(c.build);
  // THE SPLIT'S ALL-REDUCE (split_all_reduce).
  std::optional<AllReduceLaunch> all_reduce;
  if (is_split(fn)) {
    const std::variant<AllReduceLaunch, Error> got = split_all_reduce(
        h, input, input_stride_m, input_stride_n, m, n, dtype, a.quantization, stream);
    if (const Error* e = std::get_if<Error>(&got)) return *e;
    all_reduce = std::get<AllReduceLaunch>(got);
  }
  // 4.
  const int tm = a.tile_m, tk = a.tile_k, rs = c.launch.reduce_scatter_blocks;
  // THE PARTIALS'S ROWS, what partials_need sized it for (a one-shot keeps none).
  const int64_t partials_rows =
      n > 0 ? partials_need(fn, m,
                           n * elem_bytes(dtype) / kBuild.memory.pack_bytes,
                           world) /
                  (n * elem_bytes(dtype))
            : 0;
  const AllReduceAddAttnResRmsNormLaunch l{
      .kernel = kernel,
      .algorithm = algorithm_of(fn),
      .direction = direction_of(fn),
      .world = world,
      .tile_m = tm,
      .tile_n = a.tile_n,
      .tile_k = tk,
      .threads_per_block = a.threads_per_block,
      .blocks_per_grid = c.launch.blocks_per_grid,
      .waves_per_eu = a.waves_per_eu,
      .quantization = a.quantization,
      .reduce_scatter_blocks = rs,
      .stream = stream,
      .input = input,
      .input_stride_m = input_stride_m,
      .input_stride_n = input_stride_n,
      .input_size_m = m,
      .input_size_n = n,
      .prefix = prefix,
      .prefix_stride_m = prefix_stride_m,
      .prefix_stride_n = prefix_stride_n,
      .prefix_size_m = prefix_size_m,
      .prefix_size_n = prefix_size_n,
      .blocks = blocks,
      .blocks_stride_m = blocks_stride_m,
      .blocks_stride_r = blocks_stride_r,
      .blocks_stride_n = blocks_stride_n,
      .blocks_size_m = blocks_size_m,
      .blocks_size_r = blocks_size_r,
      .blocks_size_n = blocks_size_n,
      .norm_weight = norm_weight,
      .norm_weight_stride_n = norm_weight_stride_n,
      .norm_weight_size_n = norm_weight_size_n,
      .qk_weight = qk_weight,
      .qk_weight_stride_n = qk_weight_stride_n,
      .qk_weight_size_n = qk_weight_size_n,
      .output_norm_weight = output_norm_weight,
      .output_norm_weight_stride_n = output_norm_weight_stride_n,
      .output_norm_weight_size_n = output_norm_weight_size_n,
      .prefix_output = prefix_output,
      .prefix_output_stride_m = prefix_output_stride_m,
      .prefix_output_stride_n = prefix_output_stride_n,
      .prefix_output_size_m = prefix_output_size_m,
      .prefix_output_size_n = prefix_output_size_n,
      .output = output,
      .output_stride_m = output_stride_m,
      .output_stride_n = output_stride_n,
      .output_size_m = output_size_m,
      .output_size_n = output_size_n,
      .block_output = block_output,
      .block_output_stride_m = block_output_stride_m,
      .block_output_stride_n = block_output_stride_n,
      .block_output_size_m = block_output_size_m,
      .block_output_size_n = block_output_size_n,
      .partials_stride_m = n,
      .partials_stride_n = 1,
      .partials_size_m = partials_rows,
      .partials_size_n = n,
      .staging_stride_m = n,
      .staging_stride_n = 1,
      .staging_size_m = staged_rows(fn, a.quantization, m, world),
      .staging_size_n = n,
      .dtype = dtype,
      .num_blocks = num_blocks,
      .eps = eps,
      .output_eps = output_eps,
      .all_reduce = all_reduce};
  // A SPLIT'S ATTNRES KERNEL waits on no peer, so any grid runs: residency is its all-reduce's.
  if (!is_split(fn) && !resident(h, kernel, l.blocks_per_grid, l.threads_per_block))
    return Error::grid_not_resident;
  return l;
}

// =================================================================================================
// EVERY TUNED SELECTION FITS, for every op at the smallest call (8 columns on 2
// ranks: narrower than any slicing kernel takes) and a large one: an op never
// declines (a fusion that is on runs its fused op), and a kernel past a
// capability is a compile error, not one that overruns its signal slots.
// =================================================================================================

// The tuned kernel for `m` rows of `row` elements, cut to the work: picked, and
// launchable.
constexpr bool tuned_fits(OpType o, int world, int64_t m, int64_t row,
                          int64_t tile_cols, int64_t grid_cols) {
  const std::variant<TunedKernel, Error> t =
      pick(o, world, m, row, tile_cols, DType::bf16);
  if (std::holds_alternative<Error>(t)) return false;
  const TunedKernel& k = std::get<TunedKernel>(t);
  const Chosen cut =
      cut_to_the_work(Chosen{k.fn, k.config}, m, grid_cols, world);
  return launch_fits(cut.fn, cut.config.launch);
}
constexpr bool selections_fit() {
  for (const std::array<int64_t, 3> call :
       {std::array<int64_t, 3>{2, 1, 8},
        std::array<int64_t, 3>{kMaxRanks, 4096, 7168}}) {
    const int w = static_cast<int>(call[0]);
    const int64_t m = call[1], n = call[2];
    for (const OpType o :
         {OpType::all_reduce_rms_norm, OpType::all_reduce_add_rms_norm,
          OpType::all_reduce_add_attn_res_rms_norm})
      if (!tuned_fits(o, w, m, n, n, n)) return false;
    // The plain all-reduce's tuned kernel, as measured (its grid not cut).
    if (std::holds_alternative<Error>(
            pick(OpType::all_reduce, w, m, n, 0, DType::bf16)))
      return false;
  }
  return true;
}
static_assert(selections_fit(),
              "a selection declines, or exceeds a kernel capability");

// =================================================================================================
// A PLAIN CALL'S FACTS, an op's (nothing forced, contiguous, whole packs): its
// row and its bytes up to these always run. `build_info` hands them to Python's
// `should`, which admits nothing past them.
// =================================================================================================

// THE WIDEST ROW A PLAIN CALL RUNS, in elements: at every built world, the
// widest tile of its tuned kernels there; none (0) for an op whose tiles cross
// the row.
constexpr int64_t widest_row(OpType o) {
  if (!op(o).tile_holds_row) return 0;
  int64_t widest = 0;
  for (const int world : kBuild.supports.worlds) {
    const int w = tuned_world(o, world);
    int64_t at_world = 0;
    for (const TunedKernel& k : op(o).kernels)
      if (k.world == w && tile_n_of(k.config.build) > at_world)
        at_world = tile_n_of(k.config.build);
    if (widest == 0 || at_world < widest) widest = at_world;
  }
  return widest;
}

// THE EXCHANGE, SIZED FROM THE PROBLEM (STANDARDS, *A kernel's own buffers are
// sized from the problem it serves*): the largest call a communicator serves is
// `max_rows` rows of at most `max_cols` columns (for a model, the server's
// tokens a batch by its hidden size), and each region holds its largest
// payload at that call, in every built dtype and in either encoding (exact or
// INT4: common.cuh's INT4 payload layout): the inputs the whole input (an eager
// exact call's host copy, read through the eager input table; the pushed
// all-reduce's inbox of WORLD slots, a band of rows a pass, so at least a row
// a slot), or its INT4 payload; the partials the most any two-shot keeps
// (`partials_need`), or its INT4 payload (bounded as one row of its elements).
// Each need grows with the rows and the row's width, so a call inside the box
// fits by construction. Each rounded to a whole pack, and at least one.
struct ExchangeSizes {
  int64_t partials_bytes;
  int64_t staging_bytes;
};
constexpr ExchangeSizes exchange_sizes(int64_t max_rows, int64_t max_cols,
                                       int world) {
  const int64_t pack = kBuild.memory.pack_bytes;
  const auto whole = [pack](int64_t bytes) {
    return bytes < pack ? pack : (bytes + pack - 1) / pack * pack;
  };
  int64_t partials = 0, staging = int4_payload_bytes(max_rows, max_cols);
  for (const DType d : kBuild.supports.dtypes) {
    const int64_t packs = packs_of(max_cols, d);
    for (const TemplateInfo& t : kTemplates) {
      const int64_t need = partials_need(t.fn, max_rows, packs, world);
      if (need > partials) partials = need;
      const int64_t int4 = int4_payload_bytes(1, need / elem_bytes(d));
      if (need > 0 && int4 > partials) partials = int4;
    }
    const int64_t input = max_rows * max_cols * elem_bytes(d);
    if (input > staging) staging = input;
    // THE PUSHED ALL-REDUCE'S INBOX (forced) takes a pass of at least one
    // row: the one-shot's WORLD slots of a row each, which covers the
    // two-shot's WORLD shares of one row rounded up to whole packs. The
    // in-place pull's eager copy is the input whole, above.
    const int64_t one_row_each = world * max_cols * elem_bytes(d);
    if (one_row_each > staging) staging = one_row_each;
  }
  return {whole(partials), whole(staging)};
}

// THE MOST BYTES A PLAIN CALL RUNS ON THESE BUFFERS, whatever its shape: what
// the inputs holds (an eager input is copied in whole: by a per-row kernel, or
// by the host for the in-place all-reduce), and what the partials
// holds at the most partials a byte of its input needs among its tuned kernels:
// at one row of one pack, where rounding up is all of it (a row two-shot
// leaving two results, two). Python admits a call by it or by the box the
// exchange was sized for (`exchange_sizes`), whichever is larger.
constexpr int64_t max_bytes(OpType o, int64_t partials_bytes,
                            int64_t staging_bytes) {
  int64_t per_byte = 0;
  for (const TunedKernel& k : op(o).kernels) {
    const int64_t need =
        partials_need(k.fn, 1, 1, kMaxRanks) / kBuild.memory.pack_bytes;
    if (need > per_byte) per_byte = need;
  }
  // THE PLAIN ALL-REDUCE as the fused ops: its tuned kernels (the in-place
  // pull) copy an eager input into the inputs whole. (The pushed variant,
  // forced, runs any call in the box: its inbox holds a row a rank,
  // `exchange_sizes`.)
  if (per_byte == 0) return staging_bytes;
  const int64_t partials = partials_bytes / per_byte;
  return partials < staging_bytes ? partials : staging_bytes;
}

// EVERY WHOLE-PACK ROW UP TO THE WIDEST IS PICKED, at every built world and
// dtype. Past kMaxRanks packs every kernel takes the row's columns, so its
// widest tile at the world holds any row up to widest_row; the narrower rows,
// which a slicing kernel cannot take, are checked one by one. One op a check,
// each its own constant evaluation.
constexpr bool plain_rows_pick(OpType o) {
  const int64_t widest = widest_row(o);
  for (const int world : kBuild.supports.worlds)
    for (const DType d : op(o).dtypes)
      for (int64_t p = 1; p <= kMaxRanks; ++p) {
        const int64_t n = p * kBuild.memory.pack_bytes / elem_bytes(d);
        if (widest != 0 && n > widest) break;
        if (std::holds_alternative<Error>(
                pick(o, world, 1, n, tile_cols(o, n), d)))
          return false;
      }
  return true;
}
static_assert(plain_rows_pick(OpType::all_reduce),
              "a plain call's row under the widest is picked by no kernel");
static_assert(plain_rows_pick(OpType::all_reduce_rms_norm),
              "a plain call's row under the widest is picked by no kernel");
static_assert(plain_rows_pick(OpType::all_reduce_add_rms_norm),
              "a plain call's row under the widest is picked by no kernel");
static_assert(plain_rows_pick(OpType::all_reduce_add_attn_res_rms_norm),
              "a plain call's row under the widest is picked by no kernel");

}  // namespace hip_custom_communicator
