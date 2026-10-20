// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Built kernels, tuned tables, and the complete configuration of each call.

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

// WHAT A BUFFER MUST HOLD FOR AN [m, n] INT4 PAYLOAD, sized by the host from its own facts: a
// 4-bit value an element (m * n / 2 bytes), rounded up to the scales' 256-byte alignment, then an
// fp16 scale for each group of 32 elements of a row. The kernels place the payload themselves; a
// CPU test holds this to the end of their layout's last scale.
constexpr int64_t kInt4PayloadGroup = 32;
constexpr int64_t kInt4PayloadScalesAlign = 256;
constexpr int64_t int4_payload_bytes(int64_t m, int64_t n) {
  const int64_t values = (m * n / 2 + kInt4PayloadScalesAlign - 1) / kInt4PayloadScalesAlign *
                         kInt4PayloadScalesAlign;
  return values + m * ((n + kInt4PayloadGroup - 1) / kInt4PayloadGroup) *
                      static_cast<int64_t>(sizeof(c10::Half));
}

// THE INT4 FLAT WALK'S SHARE (reduce_scatter.cuh, the plain push two-shot's INT4): an [m, n]
// input's elements numbered row-major, a rank's share whole groups, the last rank's to the end.
// Its payloads lie on views of their own, [1, world * share] in the inbox and [1, share] in the
// partials: on the input's [m, n] the rounded-up shares would run past its values.
constexpr int64_t int4_flat_share(int64_t m, int64_t n, int world) {
  return ((m * n + kInt4PayloadGroup - 1) / kInt4PayloadGroup + world - 1) / world *
         kInt4PayloadGroup;
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
// a tuned point (the quantized table's INT4 points). A one-shot
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
// THE PUSHED ONE-SHOT (the flag in the line): a tile a row high, a pack a
// thread; aiter's ar_ll_gfx9 runs 256 threads of one 8-byte packet each, 1024
// elements a block (7 blocks at 7168), which {128, 1, 1, 1024} matches block for
// block at a 16-byte pack a thread. Every build's tile fits a lines slot
// (handle.cuh's kLineSlotBytes, checked below). Not swept yet.
constexpr BuildConfig kAllReducePushOneShotConfigs[] = {
    AllReduceBuildConfig{64, 1, 1, 512},
    AllReduceBuildConfig{128, 1, 1, 1024},
    AllReduceBuildConfig{256, 1, 1, 1024},
};
// THE PUSHED ONE-SHOT'S PACK: a line's payload (kernels/all_reduce.cuh), so its
// builds lay out at that many elements a thread.
constexpr int push_one_shot_pack(int elem_bytes) { return kLinePayloadBytes / elem_bytes; }
// The plain two-shot: 512 threads divided into whole waves per peer. The
// eight-rank tile is 512 elements; four and two ranks need 1024 and 2048
// elements respectively so every lane owns a complete pack. Keep the tuned
// eight-rank build first; fallback picks use the smallest valid tile below.
constexpr BuildConfig kAllReduceTwoShotExactConfigs[] = {
    AllReduceBuildConfig{512, 1, 1, 512},
    AllReduceBuildConfig{512, 1, 1, 1024},
    AllReduceBuildConfig{512, 1, 1, 2048},
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
     OpType::all_reduce, Algorithm::two_shot, false, kAllReduceTwoShotExactConfigs},
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
    // Plain push supports exact and INT4 (the one-shot exact only: the flag in the line, its own
    // builds); plain pull supports exact only.
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_push_one_shot),
     OpType::all_reduce, Algorithm::one_shot, true, kAllReducePushOneShotConfigs},
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

inline bool resident(const Handle& h, const void* kernel, int blocks,
                     int threads) {
  return blocks <= h.resident_blocks(kernel, threads);
}

// =================================================================================================
// EACH OP'S KERNELS.
// =================================================================================================

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
  return world > 0 && c.tile_m == 1 && c.threads_per_block % world == 0 &&
         (c.threads_per_block / world) % kWaveSize == 0 &&
         c.tile_n % (nl * (c.threads_per_block / world)) == 0;
}
// EVERY PUSHED ONE-SHOT BUILD'S TILE FITS A LINES SLOT (handle.cuh's LINES) at
// every built dtype.
constexpr bool push_one_shot_tiles_fit() {
  for (const BuildConfig& b : kAllReducePushOneShotConfigs)
    for (const DType d : kBuild.supports.dtypes)
      if (tile_m_of(b) != 1 || tile_n_of(b) * elem_bytes(d) > kLineSlotBytes) return false;
  return true;
}
static_assert(push_one_shot_tiles_fit(),
              "a pushed one-shot tile wider than its lines slot");

// THE PULLED NORMS' INT4 IS A GROUP A THREAD (their INT4 body, all_reduce_add_rms_norm.cuh):
// TILE_N / 32 threads, a block one row's groups.
constexpr bool int4_group_a_thread(Template k) {
  return k == Template::all_reduce_pull_two_shot_rms_norm ||
         k == Template::all_reduce_pull_two_shot_add_rms_norm ||
         k == Template::all_reduce_pull_two_shot_add_attn_res_rms_norm;
}
// A ROW OF THE STAGING AND PARTIALS AS A KERNEL VIEWS IT: n, or a group-a-thread INT4 norm's
// payload row, whole groups (its kernels' n_pad: every row's groups start on a group).
constexpr int64_t payload_row(Template k, int quantization, int64_t n) {
  return int4_group_a_thread(k) && quantization == Quantization::int4
             ? (n + kInt4PayloadGroup - 1) / kInt4PayloadGroup * kInt4PayloadGroup
             : n;
}
// THE STAGING'S ROWS AS A KERNEL VIEWS IT: the input's m, or a group-a-thread
// INT4 norm's inbox, WORLD slots of an owner's ceil(m / WORLD) rows.
constexpr int64_t staged_rows(Template k, int quantization, int64_t m, int world) {
  return int4_group_a_thread(k) && quantization == Quantization::int4
             ? world * ((m + world - 1) / world)
             : m;
}

// =================================================================================================
// THE OPS, in OpType's order: each one's name (as Python calls it) and what a plain call of it
// (nothing forced) must be to run, which `build_info` hands Python's `should`. What each runs at
// a size is the tuned points' (types.cuh's TunedPoint), the handle's.
// =================================================================================================

// A NORM'S WEIGHT, besides its input's dtype: fp32 (by_weight compiles exactly
// these).
constexpr DType kNormWeights[] = {DType::f32};

struct Op {
  OpType type;
  const char* name;
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
     kBuild.supports.dtypes,
     {},
     false,
     false},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_rms_norm),
     kBuild.supports.dtypes, kNormWeights, true, true},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_add_rms_norm),
     kBuild.supports.dtypes, kNormWeights, true, true},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_add_attn_res_rms_norm),
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

// EVERY OP HAS KERNELS, each one of its own templates, its build listed and its
// launch one the forcing would accept (a grid of 1 to kMaxBlocks blocks, a
// split only on the AttnRes pull).
constexpr bool launch_fits(Template t, const LaunchConfig& l) {
  if (l.blocks_per_grid < 1 || l.blocks_per_grid > kMaxBlocks) return false;
  if (l.reduce_scatter_blocks < 0) return false;
  return l.reduce_scatter_blocks == 0 ||
         t == Template::all_reduce_pull_two_shot_add_attn_res_rms_norm;
}

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
// 2. THE LOOKUP: a call runs its nearest tuned point's kernel and config (types.cuh's
// TunedPoint: the handle's, from the arch's config file, `check`ed at open). Nothing here is a
// tuned number: the points are.
// =================================================================================================

constexpr int64_t distance(int64_t a, int64_t b) {
  return a < b ? b - a : a - b;
}


// THE NEAREST POINT `take` admits: at the handle's world, the nearest width, then the nearest
// size (rows x width, so a width no point lists reads the nearest width's points at the same
// bytes), the smaller on a tie. Whether its kernel takes the call is the kernel's to say: a
// kernel that cannot is refused by its checks, never passed over for another. None where `take`
// admits no point.
template <typename TAKE>
inline const TunedPoint* nearest(const Handle& h, TAKE take, int64_t m, int64_t n) {
  const auto admitted = [&](const TunedPoint& p) { return p.world == h.world_size() && take(p); };
  int64_t width = 0;
  for (const TunedPoint& p : h.points())
    if (admitted(p) && (width == 0 || distance(p.hidden, n) < distance(width, n)))
      width = p.hidden;
  const TunedPoint* best = nullptr;
  for (const TunedPoint& p : h.points()) {
    if (p.hidden != width || !admitted(p)) continue;
    const int64_t d = distance(p.tokens * p.hidden, m * n);
    const int64_t held = best ? distance(best->tokens * best->hidden, m * n) : 0;
    if (!best || d < held || (d == held && p.tokens < best->tokens)) best = &p;
  }
  return best;
}

// THE KERNEL A PLAIN CALL RUNS: the nearest point its communicator's table picks (`check` holds
// that every op has picks at every world and mode it builds).
inline Chosen lookup(const Handle& h, OpType o, int64_t m, int64_t n) {
  const Mode mode = h.mode();
  const TunedPoint* p = nearest(h, [&](const TunedPoint& p) {
    return p.op == o && p.mode == mode && p.pick;
  }, m, n);
  return Chosen{p->fn, p->config};
}

// A KERNEL'S CONFIG FOR A CALL THAT NAMES IT (a forced launch at the tuned config: gp's kernel
// lines, the tests): the nearest point of kernel `fn` moving `quantization`, a pick or a config,
// in either table; moving INT4 where it has no INT4 point, its nearest exact point moved INT4 (a
// build it lists). None where it has neither.
inline std::optional<Chosen> kernel_config(const Handle& h, Template fn, int quantization,
                                           int64_t m, int64_t n) {
  const auto of = [&](int q) {
    return nearest(h, [&](const TunedPoint& p) {
      return p.fn == fn && quantization_of(p.config.build) == q &&
             built(fn, quantized(p.config.build, quantization));
    }, m, n);
  };
  const TunedPoint* p = of(quantization);
  if (!p && quantization == Quantization::int4) p = of(Quantization::none);
  if (!p) return std::nullopt;
  return Chosen{p->fn, {quantized(p->config.build, quantization), p->config.launch}};
}

// THE KERNEL A CALL RUNS: the forced one as forced, else its nearest point.
inline std::variant<Chosen, Error> chosen(
    const Handle& h, const std::variant<std::optional<Chosen>, Error>& forcing,
    OpType o, int64_t m, int64_t n) {
  if (const Error* e = std::get_if<Error>(&forcing)) return *e;
  if (const std::optional<Chosen>& f = std::get<std::optional<Chosen>>(forcing))
    return *f;
  return lookup(h, o, m, n);
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
  if (int4 && (algorithm_of(fn) == Algorithm::one_shot ||
               fn == Template::all_reduce_pull_two_shot))
    return Error::quantization_not_supported;
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
  // An INT4 payload lies in the region that holds it exact (common.cuh's
  // INT4 payload layout): the input's in the inputs, the sums' in the partials, at
  // most one row of the partials's elements (its every view's bound); the flat walk's
  // on its shares' own views (int4_flat_share).
  const bool flat = fn == Template::all_reduce_push_two_shot;
  const int64_t share = int4_flat_share(m, n, world);
  const int64_t row = payload_row(fn, Quantization::int4, n);
  if (int4 && (flat ? int4_payload_bytes(1, world * share)
                    : int4_payload_bytes(staged_rows(fn, Quantization::int4, m, world), row)) >
                  h->staging_bytes())
    return Error::inputs_too_small;
  const int64_t partials = partials_need(fn, m, packs, world);
  if (int4 && (flat                       ? int4_payload_bytes(1, share)
               : int4_group_a_thread(fn) ? int4_payload_bytes((m + world - 1) / world, row)
                                         : int4_payload_bytes(1, partials / elem_bytes(dtype))) >
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

// THE TUNED CONFIG AS A QUERY (`HipCustomCommunicator.tuned`): with no kernel named, what a plain
// call of `m` rows of `n` in `dtype` runs on this communicator (`lookup`); with kernel `fn` named,
// its config moving `quantization` (`kernel_config`): a forced launch at the tuned config. None
// where `fn` has no point.
inline std::variant<std::optional<Chosen>, Error> tuned(const Handle& h, OpType o, int64_t m,
                                                        int64_t n, DType dtype, int quantization,
                                                        std::optional<Template> fn) {
  if (fn) return kernel_config(h, *fn, quantization, m, n);
  return std::optional<Chosen>{lookup(h, o, m, n)};
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
// what its partials hold. INT4 (the push two-shot's INT4 build): the shares
// quantized and pushed whole. PUSHED (forced): the input pushed into the peers'
// inboxes, a band a pass.
enum class Kernel {
  all_reduce_pull_one_shot,
  all_reduce_push_one_shot,
  all_reduce_pull_two_shot,
  all_reduce_push_two_shot,
  all_reduce_pull_one_shot_rms_norm,
  all_reduce_pull_two_shot_rms_norm,
  all_reduce_push_two_shot_rms_norm,
  rms_norm_rows,
  all_reduce_pull_one_shot_add_rms_norm,
  all_reduce_pull_two_shot_add_rms_norm,
  all_reduce_push_two_shot_add_rms_norm,
  add_rms_norm_rows,
  all_reduce_pull_one_shot_add_attn_res_rms_norm,
  all_reduce_pull_two_shot_add_attn_res_rms_norm,
  all_reduce_push_two_shot_add_attn_res_rms_norm,
  add_attn_res_rms_norm_rows,
};

struct ConfigFields {
  Kernel kernel{};
  Algorithm algorithm{};
  Direction direction{};
  int world{};
  int tile_m{};
  int tile_n{};
  int tile_k{};
  int threads{};
  int blocks{};
  int waves_per_eu{};
  int quantization{};
  int reduce_scatter_blocks{};
  int shared_memory_bytes{};
  DType dtype{};
  DType weight_dtype{};
  bool contiguous{};
  bool in_place{};
  int64_t partials_stride_m{};
  int64_t partials_stride_n{};
  int64_t partials_size_m{};
  int64_t partials_size_n{};
  int64_t staging_stride_m{};
  int64_t staging_stride_n{};
  int64_t staging_size_m{};
  int64_t staging_size_n{};
  // THE PUSHED ONE-SHOT'S LINES INBOX, [slot, m, n] in payload elements (kernels/all_reduce.cuh;
  // handle.cuh's LINES): zeros for every other kernel.
  int64_t lines_stride_slot{};
  int64_t lines_stride_m{};
  int64_t lines_stride_n{};
  int64_t lines_size_slot{};
  int64_t lines_size_m{};
  int64_t lines_size_n{};
  int64_t scales_stride_m{};
  int64_t scales_size_m{};
  int64_t sums_stride_n{};
};

// A split runs the all-reduce config first, then this config on local sums.
struct Config : ConfigFields {
  ConfigFields all_reduce{};
  bool split = false;
};

constexpr Kernel kernel_of(Template fn) {
  switch (fn) {
    case Template::all_reduce_pull_one_shot:
      return Kernel::all_reduce_pull_one_shot;
    case Template::all_reduce_pull_two_shot:
      return Kernel::all_reduce_pull_two_shot;
    case Template::all_reduce_pull_one_shot_rms_norm:
      return Kernel::all_reduce_pull_one_shot_rms_norm;
    case Template::all_reduce_pull_two_shot_rms_norm:
      return Kernel::all_reduce_pull_two_shot_rms_norm;
    case Template::all_reduce_pull_one_shot_add_rms_norm:
      return Kernel::all_reduce_pull_one_shot_add_rms_norm;
    case Template::all_reduce_pull_two_shot_add_rms_norm:
      return Kernel::all_reduce_pull_two_shot_add_rms_norm;
    case Template::all_reduce_pull_one_shot_add_attn_res_rms_norm:
      return Kernel::all_reduce_pull_one_shot_add_attn_res_rms_norm;
    case Template::all_reduce_pull_two_shot_add_attn_res_rms_norm:
      return Kernel::all_reduce_pull_two_shot_add_attn_res_rms_norm;
    case Template::all_reduce_push_two_shot_rms_norm:
      return Kernel::all_reduce_push_two_shot_rms_norm;
    case Template::all_reduce_push_two_shot_add_rms_norm:
      return Kernel::all_reduce_push_two_shot_add_rms_norm;
    case Template::all_reduce_push_two_shot_add_attn_res_rms_norm:
      return Kernel::all_reduce_push_two_shot_add_attn_res_rms_norm;
    case Template::all_reduce_push_one_shot:
      return Kernel::all_reduce_push_one_shot;
    case Template::all_reduce_push_two_shot:
      return Kernel::all_reduce_push_two_shot;
    case Template::all_reduce_pull_split_add_attn_res_rms_norm:
      return Kernel::add_attn_res_rms_norm_rows;
    case Template::all_reduce_pull_split_rms_norm:
      return Kernel::rms_norm_rows;
    case Template::all_reduce_pull_split_add_rms_norm:
      return Kernel::add_rms_norm_rows;
  }
  throw std::runtime_error("unknown kernel template");
}

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

template <>
struct of_dtype<DType::f32> { using t = float; };

template <typename T>
constexpr DType dtype_of = std::is_same_v<T, c10::Half> ? DType::f16 : std::is_same_v<T,
    c10::BFloat16> ? DType::bf16 : DType::f32;

template <Template FN, typename F>
void with_config_build(const Config& config, F& f) {
  by_world(config.world, [&](auto world) {
    by_dtype(config.dtype, [&](auto dtype) {
      using T = typename decltype(dtype)::t;
      // Only the norm families instantiate a separate weight dtype.
      const auto weight = [&](auto w) {
        using W = typename decltype(w)::t;
        by_build<FN>([&]() -> BuildConfig {
          using B = family_t<FN>;
          B b{};
          b.threads_per_block = config.threads;
          b.waves_per_eu = config.waves_per_eu;
          b.tile_m = config.tile_m;
          b.tile_n = config.tile_n;
          b.quantization = config.quantization;
          if constexpr (requires { b.tile_k; }) b.tile_k = config.tile_k;
          return b;
        }(), [&](auto b) {
          constexpr auto B = decltype(b)::value;
          constexpr int WORLD = decltype(world)::value;
          constexpr bool valid = [=] {
            if constexpr (op_of(FN) != OpType::all_reduce) return true;
            else if constexpr (FN == Template::all_reduce_push_one_shot)
              return one_shot_lays_out(B, push_one_shot_pack(sizeof(T)));
            else if constexpr (algorithm_of(FN) == Algorithm::one_shot)
              return one_shot_lays_out(B, traits<T>::N);
            else return two_shot_lays_out(B, WORLD, traits<T>::N);
          }();
          if constexpr (valid) {
            const auto layout = [&]<bool CONTIGUOUS>() {
              constexpr Config C = [=] {
                Config c{};
                c.kernel = kernel_of(FN);
                c.algorithm = algorithm_of(FN);
                c.direction = direction_of(FN);
                c.world = WORLD;
                c.dtype = dtype_of<T>;
                c.weight_dtype = dtype_of<W>;
                c.tile_m = B.tile_m;
                c.tile_n = B.tile_n;
                if constexpr (requires { B.tile_k; }) c.tile_k = B.tile_k;
                c.threads = B.threads_per_block;
                c.waves_per_eu = B.waves_per_eu;
                c.quantization = B.quantization;
                c.contiguous = CONTIGUOUS;
                return c;
              }();
              f.template operator()<C>();
            };
            if constexpr (op_of(FN) == OpType::all_reduce && is_two_shot(FN)) {
              if (config.contiguous) layout.template operator()<true>();
              else layout.template operator()<false>();
            } else layout.template operator()<false>();
          }
        });
      };
      if constexpr (op_of(FN) == OpType::all_reduce_rms_norm ||
                    op_of(FN) == OpType::all_reduce_add_rms_norm)
        by_weight<T>(config.weight_dtype, config.dtype, weight);
      else weight(type<T>{});
    });
  });
}

template <OpType OP, typename F>
void with_config(const Config& config, F&& f) {
  if constexpr (op_of(Template::all_reduce_pull_one_shot) == OP)
    if (config.kernel ==
        kernel_of(Template::all_reduce_pull_one_shot))
      return with_config_build<Template::all_reduce_pull_one_shot>(config, f);
  if constexpr (op_of(Template::all_reduce_pull_two_shot) == OP)
    if (config.kernel ==
        kernel_of(Template::all_reduce_pull_two_shot))
      return with_config_build<Template::all_reduce_pull_two_shot>(config, f);
  if constexpr (op_of(Template::all_reduce_pull_one_shot_rms_norm) == OP)
    if (config.kernel ==
        kernel_of(Template::all_reduce_pull_one_shot_rms_norm))
      return with_config_build<Template::all_reduce_pull_one_shot_rms_norm>(config, f);
  if constexpr (op_of(Template::all_reduce_pull_two_shot_rms_norm) == OP)
    if (config.kernel ==
        kernel_of(Template::all_reduce_pull_two_shot_rms_norm))
      return with_config_build<Template::all_reduce_pull_two_shot_rms_norm>(config, f);
  if constexpr (op_of(Template::all_reduce_pull_one_shot_add_rms_norm) == OP)
    if (config.kernel ==
        kernel_of(Template::all_reduce_pull_one_shot_add_rms_norm))
      return with_config_build<Template::all_reduce_pull_one_shot_add_rms_norm>(config, f);
  if constexpr (op_of(Template::all_reduce_pull_two_shot_add_rms_norm) == OP)
    if (config.kernel ==
        kernel_of(Template::all_reduce_pull_two_shot_add_rms_norm))
      return with_config_build<Template::all_reduce_pull_two_shot_add_rms_norm>(config, f);
  if constexpr (op_of(Template::all_reduce_pull_one_shot_add_attn_res_rms_norm) == OP)
    if (config.kernel ==
        kernel_of(Template::all_reduce_pull_one_shot_add_attn_res_rms_norm))
      return with_config_build<Template::all_reduce_pull_one_shot_add_attn_res_rms_norm>(config, f);
  if constexpr (op_of(Template::all_reduce_pull_two_shot_add_attn_res_rms_norm) == OP)
    if (config.kernel ==
        kernel_of(Template::all_reduce_pull_two_shot_add_attn_res_rms_norm))
      return with_config_build<Template::all_reduce_pull_two_shot_add_attn_res_rms_norm>(config, f);
  if constexpr (op_of(Template::all_reduce_push_two_shot_rms_norm) == OP)
    if (config.kernel ==
        kernel_of(Template::all_reduce_push_two_shot_rms_norm))
      return with_config_build<Template::all_reduce_push_two_shot_rms_norm>(config, f);
  if constexpr (op_of(Template::all_reduce_push_two_shot_add_rms_norm) == OP)
    if (config.kernel ==
        kernel_of(Template::all_reduce_push_two_shot_add_rms_norm))
      return with_config_build<Template::all_reduce_push_two_shot_add_rms_norm>(config, f);
  if constexpr (op_of(Template::all_reduce_push_two_shot_add_attn_res_rms_norm) == OP)
    if (config.kernel ==
        kernel_of(Template::all_reduce_push_two_shot_add_attn_res_rms_norm))
      return with_config_build<Template::all_reduce_push_two_shot_add_attn_res_rms_norm>(config, f);
  if constexpr (op_of(Template::all_reduce_push_one_shot) == OP)
    if (config.kernel ==
        kernel_of(Template::all_reduce_push_one_shot))
      return with_config_build<Template::all_reduce_push_one_shot>(config, f);
  if constexpr (op_of(Template::all_reduce_push_two_shot) == OP)
    if (config.kernel ==
        kernel_of(Template::all_reduce_push_two_shot))
      return with_config_build<Template::all_reduce_push_two_shot>(config, f);
  if constexpr (op_of(Template::all_reduce_pull_split_add_attn_res_rms_norm) == OP)
    if (config.kernel ==
        kernel_of(Template::all_reduce_pull_split_add_attn_res_rms_norm))
      return with_config_build<Template::all_reduce_pull_split_add_attn_res_rms_norm>(config, f);
  if constexpr (op_of(Template::all_reduce_pull_split_rms_norm) == OP)
    if (config.kernel ==
        kernel_of(Template::all_reduce_pull_split_rms_norm))
      return with_config_build<Template::all_reduce_pull_split_rms_norm>(config, f);
  if constexpr (op_of(Template::all_reduce_pull_split_add_rms_norm) == OP)
    if (config.kernel ==
        kernel_of(Template::all_reduce_pull_split_add_rms_norm))
      return with_config_build<Template::all_reduce_pull_split_add_rms_norm>(config, f);
  not_built("kernel config");
}

template <Config C>
const void* kernel_pointer() {
  using T = typename of_dtype<C.dtype>::t;
  using W = typename of_dtype<C.weight_dtype>::t;
  if constexpr (C.kernel == Kernel::all_reduce_pull_one_shot)
    return reinterpret_cast<const void*>(
        all_reduce_pull_one_shot<T, C.world, C.tile_m, C.tile_n, C.threads, C.waves_per_eu,
            C.quantization>);
  if constexpr (C.kernel == Kernel::all_reduce_push_one_shot)
    return reinterpret_cast<const void*>(
        all_reduce_push_one_shot<T, C.world, C.tile_m, C.tile_n, C.threads, C.waves_per_eu,
            C.quantization>);
  if constexpr (C.kernel == Kernel::all_reduce_pull_two_shot)
    return reinterpret_cast<const void*>(
        all_reduce_pull_two_shot<T, C.world, C.tile_m, C.tile_n, C.threads, C.waves_per_eu,
            C.contiguous, C.quantization>);
  if constexpr (C.kernel == Kernel::all_reduce_push_two_shot)
    return reinterpret_cast<const void*>(
        all_reduce_push_two_shot<T, C.world, C.tile_m, C.tile_n, C.threads, C.waves_per_eu,
            C.contiguous, C.quantization>);
  if constexpr (C.kernel == Kernel::all_reduce_pull_one_shot_rms_norm)
    return reinterpret_cast<const void*>(
        all_reduce_pull_one_shot_rms_norm<T, W, C.world, C.tile_n, C.threads, C.waves_per_eu,
            C.quantization>);
  if constexpr (C.kernel == Kernel::all_reduce_pull_two_shot_rms_norm)
    return reinterpret_cast<const void*>(
        all_reduce_pull_two_shot_rms_norm<T, W, C.world, C.tile_n, C.threads, C.waves_per_eu,
            C.quantization>);
  if constexpr (C.kernel == Kernel::all_reduce_push_two_shot_rms_norm)
    return reinterpret_cast<const void*>(
        all_reduce_push_two_shot_rms_norm<T, W, C.world, C.tile_n, C.threads, C.waves_per_eu,
            C.quantization>);
  if constexpr (C.kernel == Kernel::rms_norm_rows)
    return reinterpret_cast<const void*>(
        rms_norm_rows<T, W, C.tile_m, C.tile_n, C.threads, C.waves_per_eu>);
  if constexpr (C.kernel == Kernel::all_reduce_pull_one_shot_add_rms_norm)
    return reinterpret_cast<const void*>(
        all_reduce_pull_one_shot_add_rms_norm<T, W, C.world, C.tile_n, C.threads, C.waves_per_eu,
            C.quantization>);
  if constexpr (C.kernel == Kernel::all_reduce_pull_two_shot_add_rms_norm)
    return reinterpret_cast<const void*>(
        all_reduce_pull_two_shot_add_rms_norm<T, W, C.world, C.tile_n, C.threads, C.waves_per_eu,
            C.quantization>);
  if constexpr (C.kernel == Kernel::all_reduce_push_two_shot_add_rms_norm)
    return reinterpret_cast<const void*>(
        all_reduce_push_two_shot_add_rms_norm<T, W, C.world, C.tile_n, C.threads, C.waves_per_eu,
            C.quantization>);
  if constexpr (C.kernel == Kernel::add_rms_norm_rows)
    return reinterpret_cast<const void*>(
        add_rms_norm_rows<T, W, C.tile_m, C.tile_n, C.threads, C.waves_per_eu>);
  if constexpr (C.kernel == Kernel::all_reduce_pull_one_shot_add_attn_res_rms_norm)
    return reinterpret_cast<const void*>(
        all_reduce_pull_one_shot_add_attn_res_rms_norm<T, C.world, C.tile_n, C.tile_k, C.threads,
            C.waves_per_eu, C.quantization>);
  if constexpr (C.kernel == Kernel::all_reduce_pull_two_shot_add_attn_res_rms_norm)
    return reinterpret_cast<const void*>(
        all_reduce_pull_two_shot_add_attn_res_rms_norm<T, C.world, C.tile_m, C.tile_n, C.tile_k,
            C.threads, C.waves_per_eu, C.quantization>);
  if constexpr (C.kernel == Kernel::all_reduce_push_two_shot_add_attn_res_rms_norm)
    return reinterpret_cast<const void*>(
        all_reduce_push_two_shot_add_attn_res_rms_norm<T, C.world, C.tile_n, C.tile_k, C.threads,
            C.waves_per_eu, C.quantization>);
  if constexpr (C.kernel == Kernel::add_attn_res_rms_norm_rows)
    return reinterpret_cast<const void*>(
        add_attn_res_rms_norm_rows<T, C.tile_m, C.tile_n, C.tile_k, C.threads, C.waves_per_eu>);
}

inline std::variant<Config, Error> all_reduce_config(
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
      OpType::all_reduce, m, n);
  if (const Error* e = std::get_if<Error>(&picked)) return *e;
  const Chosen k = std::get<Chosen>(picked);
  const Template fn = k.fn;
  KernelConfig c = k.config;
  const AllReduceBuildConfig& a = std::get<AllReduceBuildConfig>(c.build);
  // IN PLACE: the exact pull, every rank's input read where its table says.
  const bool in_place = direction_of(fn) == Direction::pull &&
                        a.quantization == Quantization::none;
  // THE LINES: the pushed one-shot pushes into its own inbox (handle.cuh's LINES), a tile a
  // block a pass, nothing staged. ITS GRID, a rule on the launch as the other ops' cut: at most
  // its tiles (a wider grid is idle blocks) and the lines' blocks (each block its own slots).
  const bool lines = fn == Template::all_reduce_push_one_shot;
  if (lines) {
    const int64_t tiles = (m + a.tile_m - 1) / a.tile_m * ((n + a.tile_n - 1) / a.tile_n);
    int& blocks = c.launch.blocks_per_grid;
    if (tiles < blocks) blocks = static_cast<int>(tiles > 0 ? tiles : 1);
    if (blocks > kLineBlocks) blocks = kLineBlocks;
  }
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
  if ((in_place && !is_two_shot(fn)) || lines) {
    band = m;  // nothing staged: one pass (the lines' passes are the kernel's)
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
  Config config{};
  config.shared_memory_bytes = 0;
  config.weight_dtype = dtype;
  config.kernel = kernel_of(fn);
  config.algorithm = algorithm_of(fn);
  config.direction = direction_of(fn);
  config.world = world;
  config.tile_m = a.tile_m;
  config.tile_n = a.tile_n;
  config.threads = a.threads_per_block;
  config.blocks = c.launch.blocks_per_grid;
  config.waves_per_eu = a.waves_per_eu;
  config.quantization = a.quantization;
  // THE INT4 FLAT WALK'S PAYLOADS on views of their own (int4_flat_share).
  const bool int4_flat = !in_passes && is_two_shot(fn);
  const int64_t share = int4_flat_share(m, n, world);
  config.partials_stride_m = int4_flat ? share : slice_bytes / elem_bytes(dtype);
  config.partials_stride_n = 1;
  config.partials_size_m = int4_flat ? 1 : in_passes ? rows : m;
  config.partials_size_n = int4_flat ? share : slice_bytes / elem_bytes(dtype);
  config.staging_stride_m = in_place || lines                ? 0
                         : int4_flat                       ? world * share
                         : !in_passes || !is_two_shot(fn) ? n
                             : slot_packs(rows) * pack / elem_bytes(dtype);
  config.staging_stride_n = in_place || lines ? 0 : 1;
  config.staging_size_m = in_place || lines ? 0
                       : int4_flat ? 1
                       : !in_passes ? m
                       : is_two_shot(fn) ? world
                                         : world * rows;
  config.staging_size_n = in_place || lines                ? 0
                       : int4_flat                       ? world * share
                       : !in_passes || !is_two_shot(fn) ? n
                           : slot_packs(rows) * pack / elem_bytes(dtype);
  // THE LINES: [writer, row, n] payload elements, 2 * kLineBlocks rows (bank, block) of
  // kLineSlotBytes of payload each whatever the build (so a block's row is the same bytes at
  // every build), a writer's rows contiguous.
  config.lines_stride_slot = lines ? 2 * kLineBlocks * (kLineSlotBytes / elem_bytes(dtype)) : 0;
  config.lines_stride_m = lines ? kLineSlotBytes / elem_bytes(dtype) : 0;
  config.lines_stride_n = lines ? 1 : 0;
  config.lines_size_slot = lines ? world : 0;
  config.lines_size_m = lines ? 2 * kLineBlocks : 0;
  config.lines_size_n = lines ? kLineSlotBytes / elem_bytes(dtype) : 0;
  config.dtype = dtype;
  config.contiguous = input_stride_m == n * input_stride_n &&
      output_stride_m == n * output_stride_n;
  config.in_place = config.direction == Direction::pull &&
      config.quantization == Quantization::none;
  const void* kernel = nullptr;
  with_config<OpType::all_reduce>(config, [&]<Config C>() { kernel = kernel_pointer<C>(); });
  if (!kernel) return Error::tile_not_built;
  if (!resident(h, kernel, config.blocks, config.threads))
    return Error::grid_not_resident;
  return config;
}

// A SPLIT'S ALL-REDUCE (Algorithm::split): the plain all-reduce's nearest pick moving the split
// build's `quantization` (from the quantized table moving INT4, the exact one exact), forced
// whole, into this rank's local
// sums (Handle::sums), dense rows; its own select checks it (its band, its regions, its
// residency). The caller's own check holds the sums to the inputs region's size.
inline std::variant<Config, Error> split_all_reduce_config(
    const Handle& h, const void* input, int64_t input_stride_m, int64_t input_stride_n,
    int64_t m, int64_t n, DType dtype, int quantization, hipStream_t stream) {
  const Mode mode = quantization == Quantization::int4 ? Mode::quantized : Mode::exact;
  const TunedPoint* r = nearest(h, [&](const TunedPoint& p) {
    return p.op == OpType::all_reduce && p.mode == mode && p.pick &&
           quantization_of(p.config.build) == quantization;
  }, m, n);
  if (!r) return Error::quantization_not_supported;
  const AllReduceBuildConfig& b = std::get<AllReduceBuildConfig>(r->config.build);
  return all_reduce_config(h, input, input_stride_m, input_stride_n, m, n, h.sums(), n, 1, m, n,
                           dtype, algorithm_of(r->fn), direction_of(r->fn), b.tile_m, b.tile_n,
                           b.threads_per_block, r->config.launch.blocks_per_grid, b.waves_per_eu,
                           b.quantization, stream);
}

// all_reduce_rms_norm: out = rms_norm(all_reduce(inp), weight), vLLM's
// roundings exactly. inp [rows, hidden]; the weight `weight_dtype`, dtype or
// f32.
inline std::variant<Config, Error> all_reduce_rms_norm_config(
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
  const std::variant<Chosen, Error> picked =
      chosen(h,
             forced(op, algorithm, direction,
                    {threads_per_block, waves_per_eu, tile_m, tile_n},
                    quantization, blocks_per_grid, std::nullopt,
                    [&] {
                      return AllReduceRmsNormBuildConfig{
                          *threads_per_block, *waves_per_eu, *tile_m, *tile_n};
                    }),
             op, m, n);
  if (const Error* e = std::get_if<Error>(&picked)) return *e;
  const Chosen k = std::get<Chosen>(picked);
  const Template fn = k.fn;
  const KernelConfig& c = k.config;
  if (const std::optional<Error> e = refused(
          &h, fn, c, dtype, m, n, tile_cols(op, n),
          weight_refused(op, dtype, weight_dtype), false))
    return *e;
  const AllReduceRmsNormBuildConfig& r =
      std::get<AllReduceRmsNormBuildConfig>(c.build);
  // THE SPLIT'S ALL-REDUCE (split_all_reduce).
  std::optional<Config> all_reduce;
  if (is_split(fn)) {
    const std::variant<Config, Error> got = split_all_reduce_config(
        h, input, input_stride_m, input_stride_n, m, n, dtype, r.quantization, stream);
    if (const Error* e = std::get_if<Error>(&got)) return *e;
    all_reduce = std::get<Config>(got);
  }
  // THE PARTIALS'S ROWS, what partials_need sized it for (a one-shot keeps none).
  const int64_t partials_rows =
      n > 0 ? partials_need(fn, m,
                           n * elem_bytes(dtype) / kBuild.memory.pack_bytes,
                           world) /
                  (n * elem_bytes(dtype))
            : 0;
  Config config{};
  config.shared_memory_bytes = 0;
  config.weight_dtype = dtype;
  config.kernel = kernel_of(fn);
  config.algorithm = algorithm_of(fn);
  config.direction = direction_of(fn);
  config.world = world;
  config.tile_m = r.tile_m;
  config.tile_n = r.tile_n;
  config.threads = r.threads_per_block;
  config.blocks = c.launch.blocks_per_grid;
  config.waves_per_eu = r.waves_per_eu;
  config.quantization = r.quantization;
  const int64_t row_n = payload_row(fn, r.quantization, n);
  config.partials_stride_m = row_n;
  config.partials_stride_n = 1;
  config.partials_size_m = partials_rows;
  config.partials_size_n = row_n;
  config.staging_stride_m = row_n;
  config.staging_stride_n = 1;
  config.staging_size_m = staged_rows(fn, r.quantization, m, world);
  config.staging_size_n = row_n;
  config.dtype = dtype;
  config.weight_dtype = weight_dtype;
  config.split = all_reduce.has_value();
  if (all_reduce) {
    config.all_reduce = *all_reduce;
    config.sums_stride_n = 1;
  }
  config.contiguous = input_stride_m == n * input_stride_n &&
      output_stride_m == n * output_stride_n;
  config.in_place = config.direction == Direction::pull &&
      config.quantization == Quantization::none;
  const void* kernel = nullptr;
  with_config<op>(config, [&]<Config C>() { kernel = kernel_pointer<C>(); });
  if (!kernel) return Error::tile_not_built;
  // A SPLIT'S NORM KERNEL waits on no peer, so any grid runs: residency is its all-reduce's.
  if (!is_split(fn) && !resident(h, kernel, config.blocks, config.threads))
    return Error::grid_not_resident;
  return config;
}

// all_reduce_add_rms_norm: out, residual_out =
// fused_add_rms_norm(all_reduce(inp), residual, weight), vLLM's roundings
// exactly.
inline std::variant<Config, Error>
all_reduce_add_rms_norm_config(
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
  const std::variant<Chosen, Error> picked =
      chosen(h,
             forced(op, algorithm, direction,
                    {threads_per_block, waves_per_eu, tile_m, tile_n},
                    quantization, blocks_per_grid, std::nullopt,
                    [&] {
                      return AllReduceAddRmsNormBuildConfig{
                          *threads_per_block, *waves_per_eu, *tile_m, *tile_n};
                    }),
             op, m, n);
  if (const Error* e = std::get_if<Error>(&picked)) return *e;
  const Chosen k = std::get<Chosen>(picked);
  const Template fn = k.fn;
  const KernelConfig& c = k.config;
  if (const std::optional<Error> e = refused(
          &h, fn, c, dtype, m, n, tile_cols(op, n),
          weight_refused(op, dtype, weight_dtype), false))
    return *e;
  const AllReduceAddRmsNormBuildConfig& r =
      std::get<AllReduceAddRmsNormBuildConfig>(c.build);
  // THE SPLIT'S ALL-REDUCE (split_all_reduce).
  std::optional<Config> all_reduce;
  if (is_split(fn)) {
    const std::variant<Config, Error> got = split_all_reduce_config(
        h, input, input_stride_m, input_stride_n, m, n, dtype, r.quantization, stream);
    if (const Error* e = std::get_if<Error>(&got)) return *e;
    all_reduce = std::get<Config>(got);
  }
  // THE PARTIALS'S ROWS, what partials_need sized it for (a one-shot keeps none),
  // at its row stride: the exact pull two-shot's row is its scale's pack then the
  // row (partials_need); the INT4 build's rows are the row alone, its scales in
  // the scales region.
  const bool scale_slot = fn == Template::all_reduce_pull_two_shot_add_rms_norm &&
                          r.quantization == Quantization::none;
  const int64_t slot = scale_slot ? kScaleSlotPacks * kBuild.memory.pack_bytes / elem_bytes(dtype) : 0;
  const int64_t partials_stride_m = payload_row(fn, r.quantization, n) + slot;
  const int64_t partials_rows =
      n > 0 ? partials_need(fn, m,
                           n * elem_bytes(dtype) / kBuild.memory.pack_bytes,
                           world) /
                  ((n + slot) * elem_bytes(dtype))
            : 0;
  Config config{};
  config.shared_memory_bytes = 0;
  config.weight_dtype = dtype;
  config.kernel = kernel_of(fn);
  config.algorithm = algorithm_of(fn);
  config.direction = direction_of(fn);
  config.world = world;
  config.tile_m = r.tile_m;
  config.tile_n = r.tile_n;
  config.threads = r.threads_per_block;
  config.blocks = c.launch.blocks_per_grid;
  config.waves_per_eu = r.waves_per_eu;
  config.quantization = r.quantization;
  config.partials_stride_m = partials_stride_m;
  config.partials_stride_n = 1;
  config.partials_size_m = partials_rows;
  config.partials_size_n = partials_stride_m;
  config.scales_stride_m = 1;
  config.scales_size_m = fn == Template::all_reduce_pull_two_shot_add_rms_norm && !scale_slot
                           ? (m + world - 1) / world
                           : 0;
  config.staging_stride_m = payload_row(fn, r.quantization, n);
  config.staging_stride_n = 1;
  config.staging_size_m = staged_rows(fn, r.quantization, m, world);
  config.staging_size_n = payload_row(fn, r.quantization, n);
  config.dtype = dtype;
  config.weight_dtype = weight_dtype;
  config.split = all_reduce.has_value();
  if (all_reduce) {
    config.all_reduce = *all_reduce;
    config.sums_stride_n = 1;
  }
  config.contiguous = input_stride_m == n * input_stride_n &&
      output_stride_m == n * output_stride_n;
  config.in_place = config.direction == Direction::pull &&
      config.quantization == Quantization::none;
  const void* kernel = nullptr;
  with_config<op>(config, [&]<Config C>() { kernel = kernel_pointer<C>(); });
  if (!kernel) return Error::tile_not_built;
  if (config.scales_size_m > h.scales_size_m()) return Error::partials_too_small;
  // A SPLIT'S NORM KERNEL waits on no peer, so any grid runs: residency is its all-reduce's.
  if (!is_split(fn) && !resident(h, kernel, config.blocks, config.threads))
    return Error::grid_not_resident;
  return config;
}

// all_reduce_add_attn_res_rms_norm: Kimi-K3's AttnRes and its RMSNorm on each
// row of the all-reduced sum of `input`. With a `prefix` the sum is added to
// it, into `prefix_output`; without (null), the sum IS the new prefix.
// `block_output`, when not null, gets the new prefix too. The pull two-shot's
// config family has TILE_M and its reduce-scatter blocks; the one-shot's and
// push's (a row a tile) have neither.
inline std::variant<Config, Error>
all_reduce_add_attn_res_rms_norm_config(
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
             op, m, n);
  if (const Error* e = std::get_if<Error>(&picked)) return *e;
  const Chosen k = std::get<Chosen>(picked);
  const Template fn = k.fn;
  const KernelConfig& c = k.config;
  if (const std::optional<Error> e =
          refused(&h, fn, c, dtype, m, n, tile_cols(op, n), std::nullopt,
                  false))
    return *e;
  const AllReduceAddAttnResRmsNormBuildConfig& a =
      std::get<AllReduceAddAttnResRmsNormBuildConfig>(c.build);
  // THE SPLIT'S ALL-REDUCE (split_all_reduce).
  std::optional<Config> all_reduce;
  if (is_split(fn)) {
    const std::variant<Config, Error> got = split_all_reduce_config(
        h, input, input_stride_m, input_stride_n, m, n, dtype, a.quantization, stream);
    if (const Error* e = std::get_if<Error>(&got)) return *e;
    all_reduce = std::get<Config>(got);
  }
  const int tm = a.tile_m, tk = a.tile_k, rs = c.launch.reduce_scatter_blocks;
  // THE PARTIALS'S ROWS, what partials_need sized it for (a one-shot keeps none).
  const int64_t partials_rows =
      n > 0 ? partials_need(fn, m,
                           n * elem_bytes(dtype) / kBuild.memory.pack_bytes,
                           world) /
                  (n * elem_bytes(dtype))
            : 0;
  Config config{};
  config.shared_memory_bytes = 0;
  config.weight_dtype = dtype;
  config.kernel = kernel_of(fn);
  config.algorithm = algorithm_of(fn);
  config.direction = direction_of(fn);
  config.world = world;
  config.tile_m = tm;
  config.tile_n = a.tile_n;
  config.tile_k = tk;
  config.threads = a.threads_per_block;
  config.blocks = c.launch.blocks_per_grid;
  config.waves_per_eu = a.waves_per_eu;
  config.quantization = a.quantization;
  config.reduce_scatter_blocks = rs;
  const int64_t row_n = payload_row(fn, a.quantization, n);
  config.partials_stride_m = row_n;
  config.partials_stride_n = 1;
  config.partials_size_m = partials_rows;
  config.partials_size_n = row_n;
  config.staging_stride_m = row_n;
  config.staging_stride_n = 1;
  config.staging_size_m = staged_rows(fn, a.quantization, m, world);
  config.staging_size_n = row_n;
  config.dtype = dtype;
  config.split = all_reduce.has_value();
  if (all_reduce) {
    config.all_reduce = *all_reduce;
    config.sums_stride_n = 1;
  }
  config.contiguous = input_stride_m == n * input_stride_n &&
      output_stride_m == n * output_stride_n;
  config.in_place = config.direction == Direction::pull &&
      config.quantization == Quantization::none;
  const void* kernel = nullptr;
  with_config<op>(config, [&]<Config C>() { kernel = kernel_pointer<C>(); });
  if (!kernel) return Error::tile_not_built;
  // A SPLIT'S ATTNRES KERNEL waits on no peer, so any grid runs: residency is its all-reduce's.
  if (!is_split(fn) && !resident(h, kernel, config.blocks, config.threads))
    return Error::grid_not_resident;
  return config;
}

// =================================================================================================
// EVERY TUNED SELECTION FITS, for every op at the smallest call (8 columns on 2
// ranks: narrower than any slicing kernel takes) and a large one: an op never
// declines (a fusion that is on runs its fused op), and a kernel past a
// capability is a compile error, not one that overruns its signal slots.
// =================================================================================================

// =================================================================================================
// A PLAIN CALL'S FACTS, an op's (nothing forced, contiguous, whole packs): its
// row and its bytes up to these always run. `build_info` hands them to Python's
// `should`, which admits nothing past them.
// =================================================================================================

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
    // THE PUSHED TWO-SHOT'S INBOX (forced) takes a pass of at least one row:
    // WORLD slots of a row each covers its WORLD shares of one row rounded up
    // to whole packs. The in-place pull's eager copy is the input whole,
    // above. (The pushed one-shot's inbox is its own region, handle.cuh's
    // LINES.)
    const int64_t one_row_each = world * max_cols * elem_bytes(d);
    if (one_row_each > staging) staging = one_row_each;
  }
  return {whole(partials), whole(staging)};
}

// =================================================================================================
// THE POINTS' FACTS AND CHECK: what Python's `should` admits by (from the exact table's picks, a
// plain call's), and the check open makes of a config file before anything is made (what the
// compile-time checks held of the tables before the tables became a file).
// =================================================================================================

// THE MOST BYTES A CALL OF AN OP RUNS ON THESE BUFFERS, whatever its shape: what the inputs holds
// (an eager input is copied in whole), and what the partials holds at the most partials a byte of
// its input needs among the op's templates (any may be a point's): at one row of one pack, where
// rounding up is all of it. Python admits a call by it or by the box the exchange was sized for,
// whichever is larger.
constexpr int64_t max_bytes(OpType o, int64_t partials_bytes, int64_t staging_bytes) {
  int64_t per_byte = 0;
  for (const TemplateInfo& t : kTemplates)
    if (t.op == o) {
      const int64_t need = partials_need(t.fn, 1, 1, kMaxRanks) / kBuild.memory.pack_bytes;
      if (need > per_byte) per_byte = need;
    }
  if (per_byte == 0) return staging_bytes;
  const int64_t partials = partials_bytes / per_byte;
  return partials < staging_bytes ? partials : staging_bytes;
}

// THE CHECK OPEN MAKES OF THE POINTS: the first point a call could not run, named, or a table or
// a kernel left without a point; none if the file is sound. Every point a built kernel of its own
// op at a built world, its build one its kernel lists, its grid in range (the LL push's within its
// lines' blocks), an exact table moving nothing quantized; every op a pick in each table at every
// built world; every kernel a config at every built world for each encoding it builds (an INT4
// build's may be an exact point's, moved INT4, where that build is listed).
inline std::optional<std::string> check(const std::vector<TunedPoint>& points) {
  const auto mode_name = [](Mode m) { return m == Mode::exact ? "exact" : "quantized"; };
  for (size_t i = 0; i < points.size(); ++i) {
    const TunedPoint& p = points[i];
    const auto named = [&](const char* why) {
      return "point " + std::to_string(i) + " (" + op(p.op).name + ", " + mode_name(p.mode) +
             ", world " + std::to_string(p.world) + ", " + std::to_string(p.hidden) + " wide, " +
             std::to_string(p.tokens) + " tokens, " + info(p.fn).name + "): " + why;
    };
    if (op_of(p.fn) != p.op) return named("the kernel is another op's");
    if (!world_built(p.world)) return named("its world is not built");
    if (p.hidden < 1 || p.tokens < 1) return named("no width or no tokens");
    if (p.config.build.index() != family_of(p.fn) || !built(p.fn, p.config.build))
      return named("its build is not one its kernel lists");
    if (!launch_fits(p.fn, p.config.launch)) return named("its grid is out of range");
    if (p.fn == Template::all_reduce_push_one_shot &&
        p.config.launch.blocks_per_grid > kLineBlocks)
      return named("the LL push's grid is past its lines' blocks");
    if (p.mode == Mode::exact && quantization_of(p.config.build) != Quantization::none)
      return named("a point of the exact table moves INT4");
  }
  for (const Op& o : kOps)
    for (const int world : kBuild.supports.worlds)
      for (const Mode mode : {Mode::exact, Mode::quantized}) {
        bool any = false;
        for (const TunedPoint& p : points)
          any |= p.op == o.type && p.world == world && p.mode == mode && p.pick;
        if (!any)
          return std::string(o.name) + ": no point picks at world " + std::to_string(world) +
                 " in the " + mode_name(mode) + " table";
      }
  for (const TemplateInfo& t : kTemplates)
    for (const int world : kBuild.supports.worlds)
      for (const int q : {Quantization::none, Quantization::int4}) {
        bool builds_it = false;
        for (const BuildConfig& b : t.builds) builds_it |= quantization_of(b) == q;
        bool has = false;
        for (const TunedPoint& p : points)
          has |= p.fn == t.fn && p.world == world &&
                 (quantization_of(p.config.build) == q ||
                  (q == Quantization::int4 && built(t.fn, quantized(p.config.build, q))));
        if (builds_it && !has)
          return std::string(t.name) + ": no config at world " + std::to_string(world) +
                 (q == Quantization::int4 ? " moving INT4" : " exact");
      }
  return std::nullopt;
}

}  // namespace hip_custom_communicator
