// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// LAUNCH, NO DECISIONS: what is built (the catalog: each template, its op, its shot, and the
// KernelConfigs it is compiled for, what a tuner searches), a launch's compiled instance (its
// template from its op, algorithm and direction; its build from its fields; every combination
// here is compiled, and one that was not raises), and each op's launch_<op>: the peers' view of
// the input, then that instance at its grid and block. Before select.cuh: select reads the
// catalog, and its residency check the instance.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
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

namespace hip_comms {

constexpr int elem_bytes(DType d) { return d == DType::f32 ? 4 : 2; }
// A row of `elems` of `dtype` in 16-byte packs.
constexpr int64_t packs_of(int64_t elems, DType dtype) {
  return elems * elem_bytes(dtype) / kBuild.memory.pack_bytes;
}

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
// 256 threads a block: Triton's program for AttnRes, and more blocks resident at once (AttnRes
// from local memory took 192.8 / 109.6 / 86.0 us on 192 / 384 / 512 blocks, stamps
// 2026-10-02T20-36-56Z); its grid not swept yet.
constexpr KernelConfig kAttnResPushConfigs[] = {
    AttnResConfig{{512, 256}, 4096, 1},
    AttnResConfig{{512, 256}, 8192, 1},
    AttnResConfig{{256, 512}, 4096, 1},
    AttnResConfig{{256, 512}, 8192, 1},
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
    AttnResPullConfig{{512, 192}, 1, 4096, 1, 32},
    AttnResPullConfig{{512, 192}, 1, 8192, 1, 32},
    AttnResPullConfig{{256, 384}, 1, 4096, 1, 32},
    AttnResPullConfig{{256, 384}, 1, 8192, 1, 32},
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

// EXPERIMENTAL, AttnRes on a local delta: a row a block, the grid striding over rows. 512 blocks
// took the AttnRes phase from local scratch to 86.0 us at 4096 x 7168 (192: 192.8; stamps
// 2026-10-02T20-36-56Z); not swept.
constexpr KernelConfig kAddAttnResConfigs[] = {
    AttnResConfig{{512, 512}, 4096, 1},
    AttnResConfig{{512, 512}, 8192, 1},
    AttnResConfig{{256, 512}, 4096, 1},
    AttnResConfig{{256, 512}, 8192, 1},
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
     false, false, {}},
    {HIP_COMMS_NAMED(all_reduce_pull_two_shot), OpType::all_reduce,
     true, false, {}},
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
// A LAUNCH'S COMPILED INSTANCE. The template arguments are constexpr, so each is a template
// parameter, resolved from the launch alone: dispatch(launch, world, f) hands f the compiled
// function and `bind`, which makes its arguments from the launch and the peers' view.
// =================================================================================================

template <typename T>
struct type {
  using t = T;
};

template <int N>
using constant = std::integral_constant<int, N>;

[[noreturn]] inline void not_built(const std::string& what) {
  throw std::runtime_error("hip_comms: " + what + " not built");
}

// THE C++ TYPE OF A BUILT DTYPE.
template <DType D>
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
template <typename F, size_t... I>
void by_world_in(int world, F& f, std::index_sequence<I...>) {
  constexpr auto& built = kBuild.supports.worlds;
  if (!((world == built[I] && (f(constant<built[I]>{}), true)) || ...))
    not_built("world size " + std::to_string(world));
}

template <typename F>
void by_world(int world, F&& f) {
  by_world_in(world, f, std::make_index_sequence<kBuild.supports.worlds.size()>{});
}

template <typename F, size_t... I>
void by_dtype_in(DType d, F& f, std::index_sequence<I...>) {
  constexpr auto& built = kBuild.supports.dtypes;
  if (!((d == built[I] && (f(type<typename of_dtype<built[I]>::t>{}), true)) || ...))
    not_built("dtype");
}

template <typename F>
void by_dtype(DType d, F&& f) {
  by_dtype_in(d, f, std::make_index_sequence<kBuild.supports.dtypes.size()>{});
}

// A norm's weight: T itself, or fp32.
template <typename T, typename F>
void by_weight(DType weight, DType dtype, F&& f) {
  if (weight == dtype) return f(type<T>{});
  if (weight == DType::f32) return f(type<float>{});
  not_built("weight dtype");
}

// A TEMPLATE'S BUILD, compiled in: the one of template K's configs (the catalog above) `c` names (all but
// its launch's grid and reduce_scatter_blocks, which are run time), and only those. f is handed
// it as config_constant<C>, C its family's config, every field a constant expression.
template <Template K>
using family_t = std::variant_alternative_t<family_of(K), KernelConfig>;
template <Template K, size_t I>
constexpr family_t<K> built_config = std::get<family_t<K>>(configs_of(K)[I]);
template <auto C>
struct config_constant {
  static constexpr auto value = C;
};

template <Template K, typename F, size_t... I>
void by_config_in(const KernelConfig& c, F& f, std::index_sequence<I...>) {
  if (!((same_build(c, configs_of(K)[I]) && (f(config_constant<built_config<K, I>>{}), true)) ||
        ...))
    not_built(std::string("template ") + to_string(K) + " at that tile and threads");
}

template <Template K, typename F>
void by_config(const KernelConfig& c, F&& f) {
  by_config_in<K>(c, f, std::make_index_sequence<configs_of(K).size()>{});
}

[[noreturn]] inline void not_this_ops(Template k) {
  throw std::runtime_error("hip_comms: template " + std::to_string(static_cast<int>(k)) +
                           " is not this op's");
}


// THE TEMPLATE A LAUNCH NAMES: its op's at its algorithm and direction; none is a launch select
// did not make, raised.
inline Template launched(OpType o, Algorithm a, Direction d) {
  if (const std::optional<Template> t = template_of(o, a, d)) return *t;
  throw std::runtime_error(std::string("hip_comms: ") + "no template of that algorithm and "
                           "direction for this op");
}

template <typename F>
void dispatch(const AllReduceLaunch& l, int world, F&& f) {
  const Template fn = launched(OpType::all_reduce, l.algorithm, l.direction);
  const int n       = static_cast<int>(l.bytes / kBuild.memory.pack_bytes);
  by_world(world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    by_dtype(l.dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      // IN PLACE: the input's packs, its int. STAGED: in 64 bits, with its own input and the packs
      // a staging holds.
      const auto in_place = [&](const p2p::DevComm& p) {
        return std::make_tuple(p, static_cast<T*>(l.out), n, Staged<T, false>{});
      };
      const auto staged = [&](const p2p::DevComm& p) {
        return std::make_tuple(
            p, static_cast<T*>(l.out), int64_t{l.bytes / kBuild.memory.pack_bytes},
            Staged<T, true>{static_cast<const T*>(l.inp),
                            kBuild.memory.staging_bytes / kBuild.memory.pack_bytes});
      };
      switch (fn) {
        case Template::all_reduce_pull_one_shot:
          return l.staged ? f(all_reduce_pull_one_shot<T, NG, true>, staged)
                          : f(all_reduce_pull_one_shot<T, NG, false>, in_place);
        case Template::all_reduce_pull_two_shot:
          return l.staged ? f(all_reduce_pull_two_shot<T, NG, true>, staged)
                          : f(all_reduce_pull_two_shot<T, NG, false>, in_place);
        default: not_this_ops(fn);
      }
    });
  });
}

template <typename F>
void dispatch(const AllReduceRmsNormLaunch& l, int world, F&& f) {
  using K              = Template;
  const Template fn    = launched(OpType::all_reduce_rms_norm, l.algorithm, l.direction);
  const KernelConfig c = RowConfig{{l.threads_per_block, l.blocks_per_grid}, l.tile_n};
  const int rows = static_cast<int>(l.rows), packs = static_cast<int>(packs_of(l.hidden, l.dtype));
  by_world(world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    by_dtype(l.dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      by_weight<T>(l.weight_dtype, l.dtype, [&](auto w) {
        using W         = typename decltype(w)::t;
        const auto bind = [&](const p2p::DevComm& p) {
          return std::make_tuple(p, static_cast<T*>(l.out), static_cast<const W*>(l.weight),
                                 l.eps, rows, packs);
        };
        switch (fn) {
          case K::all_reduce_pull_one_shot_rms_norm:
            return by_config<K::all_reduce_pull_one_shot_rms_norm>(c, [&](auto cc) {
              constexpr RowConfig C = decltype(cc)::value;
              f(all_reduce_pull_one_shot_rms_norm<T, W, NG, C.tile_n, C.launch.threads_per_block>,
                bind);
            });
          case K::all_reduce_pull_two_shot_rms_norm:
            return by_config<K::all_reduce_pull_two_shot_rms_norm>(c, [&](auto cc) {
              constexpr RowConfig C = decltype(cc)::value;
              f(all_reduce_pull_two_shot_rms_norm<T, W, NG, C.tile_n, C.launch.threads_per_block>,
                bind);
            });
          case K::all_reduce_push_two_shot_rms_norm:
            return by_config<K::all_reduce_push_two_shot_rms_norm>(c, [&](auto cc) {
              constexpr RowConfig C = decltype(cc)::value;
              f(all_reduce_push_two_shot_rms_norm<T, W, NG, C.tile_n, C.launch.threads_per_block>,
                bind);
            });
          default: not_this_ops(fn);
        }
      });
    });
  });
}

template <typename F>
void dispatch(const AllReduceAddRmsNormLaunch& l, int world, F&& f) {
  using K              = Template;
  const Template fn    = launched(OpType::all_reduce_add_rms_norm, l.algorithm, l.direction);
  const KernelConfig c = RowConfig{{l.threads_per_block, l.blocks_per_grid}, l.tile_n};
  const int rows = static_cast<int>(l.rows), packs = static_cast<int>(packs_of(l.hidden, l.dtype));
  by_world(world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    by_dtype(l.dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      by_weight<T>(l.weight_dtype, l.dtype, [&](auto w) {
        using W         = typename decltype(w)::t;
        const auto bind = [&](const p2p::DevComm& p) {
          return std::make_tuple(p, static_cast<T*>(l.out), static_cast<T*>(l.residual_out),
                                 static_cast<const T*>(l.residual),
                                 static_cast<const W*>(l.weight), l.eps, rows, packs);
        };
        switch (fn) {
          case K::all_reduce_pull_one_shot_add_rms_norm:
            return by_config<K::all_reduce_pull_one_shot_add_rms_norm>(c, [&](auto cc) {
              constexpr RowConfig C = decltype(cc)::value;
              f(all_reduce_pull_one_shot_add_rms_norm<T, W, NG, C.tile_n,
                                                      C.launch.threads_per_block>,
                bind);
            });
          case K::all_reduce_pull_two_shot_add_rms_norm:
            return by_config<K::all_reduce_pull_two_shot_add_rms_norm>(c, [&](auto cc) {
              constexpr RowConfig C = decltype(cc)::value;
              f(all_reduce_pull_two_shot_add_rms_norm<T, W, NG, C.tile_n,
                                                      C.launch.threads_per_block>,
                bind);
            });
          case K::all_reduce_push_two_shot_add_rms_norm:
            return by_config<K::all_reduce_push_two_shot_add_rms_norm>(c, [&](auto cc) {
              constexpr RowConfig C = decltype(cc)::value;
              f(all_reduce_push_two_shot_add_rms_norm<T, W, NG, C.tile_n,
                                                      C.launch.threads_per_block>,
                bind);
            });
          default: not_this_ops(fn);
        }
      });
    });
  });
}

template <typename F>
void dispatch(const AllReduceAddAttnResRmsNormLaunch& l, int world, F&& f) {
  using K = Template;
  const Template fn =
      launched(OpType::all_reduce_add_attn_res_rms_norm, l.algorithm, l.direction);
  const LaunchConfig launch{l.threads_per_block, l.blocks_per_grid};
  const KernelConfig c =
      fn == K::all_reduce_pull_two_shot_add_attn_res_rms_norm
          ? KernelConfig{AttnResPullConfig{launch, l.tile_m, l.tile_n, l.tile_k,
                                           l.reduce_scatter_blocks}}
          : KernelConfig{AttnResConfig{launch, l.tile_n, l.tile_k}};
  const int rows = static_cast<int>(l.rows), packs = static_cast<int>(packs_of(l.hidden, l.dtype));
  by_world(world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    by_dtype(l.dtype, [&](auto t) {
      using T         = typename decltype(t)::t;
      const auto bind = [&](const p2p::DevComm& p) {
        return std::make_tuple(
            p, static_cast<T*>(l.prefix), static_cast<T*>(l.blocks), l.block_stride_m,
            l.block_stride_r, static_cast<const T*>(l.norm_weight),
            static_cast<const T*>(l.qk_weight), static_cast<const T*>(l.out_norm_weight),
            static_cast<T*>(l.out), l.num_blocks, l.write_idx, l.eps, l.out_eps, rows, packs);
      };
      // The pull's reduce-scatter blocks, a launch's choice: run time, after the rest.
      const auto bind_pull = [&](const p2p::DevComm& p) {
        return std::tuple_cat(bind(p), std::make_tuple(l.reduce_scatter_blocks));
      };
      const bool prefix = l.has_prefix;
      switch (fn) {
        case K::all_reduce_pull_one_shot_add_attn_res_rms_norm:
          return by_config<K::all_reduce_pull_one_shot_add_attn_res_rms_norm>(c, [&](auto cc) {
            constexpr AttnResConfig C = decltype(cc)::value;
            constexpr int TN = C.tile_n, TK = C.tile_k, TPB = C.launch.threads_per_block;
            prefix
                ? f(all_reduce_pull_one_shot_add_attn_res_rms_norm<T, NG, true, TN, TK, TPB>,
                    bind)
                : f(all_reduce_pull_one_shot_add_attn_res_rms_norm<T, NG, false, TN, TK, TPB>,
                    bind);
          });
        case K::all_reduce_pull_two_shot_add_attn_res_rms_norm:
          return by_config<K::all_reduce_pull_two_shot_add_attn_res_rms_norm>(c, [&](auto cc) {
            constexpr AttnResPullConfig C = decltype(cc)::value;
            constexpr int TM = C.tile_m, TN = C.tile_n, TK = C.tile_k,
                          TPB = C.launch.threads_per_block;
            prefix ? f(all_reduce_pull_two_shot_add_attn_res_rms_norm<T, NG, true, TM, TN, TK,
                                                                       TPB>,
                       bind_pull)
                   : f(all_reduce_pull_two_shot_add_attn_res_rms_norm<T, NG, false, TM, TN, TK,
                                                                       TPB>,
                       bind_pull);
          });
        case K::all_reduce_push_two_shot_add_attn_res_rms_norm:
          return by_config<K::all_reduce_push_two_shot_add_attn_res_rms_norm>(c, [&](auto cc) {
            constexpr AttnResConfig C = decltype(cc)::value;
            constexpr int TN = C.tile_n, TK = C.tile_k, TPB = C.launch.threads_per_block;
            prefix
                ? f(all_reduce_push_two_shot_add_attn_res_rms_norm<T, NG, true, TN, TK, TPB>,
                    bind)
                : f(all_reduce_push_two_shot_add_attn_res_rms_norm<T, NG, false, TN, TK, TPB>,
                    bind);
          });
        default: not_this_ops(fn);
      }
    });
  });
}

// THE GEMM TAILS, written or added: one config list for all four templates, so one lookup serves
// each.
template <typename L, typename F>
void dispatch_gemm_tail(const L& l, OpType op, int world, F&& f) {
  const Template fn    = launched(op, l.algorithm, l.direction);
  const KernelConfig c = GemmConfig{{l.threads_per_block, l.blocks_per_grid}, l.tile_m, l.tile_n,
                                    l.tile_k, l.slice_k};
  const int rows = static_cast<int>(l.rows), packs = static_cast<int>(packs_of(l.hidden, l.dtype));
  by_world(world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    by_dtype(l.dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      static_assert(configs_of(Template::all_reduce_pull_one_shot_rms_norm_gemm_add).data() ==
                        configs_of(Template::all_reduce_pull_two_shot_rms_norm_gemm_add).data() &&
                    configs_of(Template::all_reduce_pull_one_shot_rms_norm_gemm_add).data() ==
                        configs_of(Template::all_reduce_pull_one_shot_rms_norm_gemm).data() &&
                    configs_of(Template::all_reduce_pull_one_shot_rms_norm_gemm_add).data() ==
                        configs_of(Template::all_reduce_pull_two_shot_rms_norm_gemm).data());
      by_config<Template::all_reduce_pull_one_shot_rms_norm_gemm_add>(c, [&](auto cc) {
        constexpr GemmConfig C = decltype(cc)::value;
        constexpr int TM = C.tile_m, TN = C.tile_n, TK = C.tile_k, SK = C.slice_k,
                      TPB = C.launch.threads_per_block;
        const auto bind = [&](const p2p::DevComm& p) {
          return std::make_tuple(p, static_cast<const T*>(l.norm_weight), l.eps,
                                 static_cast<const T*>(l.gemm_weight), static_cast<int>(l.n_cols),
                                 static_cast<T*>(l.out), l.out_stride,
                                 static_cast<T*>(l.workspace), rows, packs);
        };
        switch (fn) {
          case Template::all_reduce_pull_one_shot_rms_norm_gemm_add:
            return f(all_reduce_pull_one_shot_rms_norm_gemm_add<T, NG, TM, TN, TK, SK, TPB>, bind);
          case Template::all_reduce_pull_two_shot_rms_norm_gemm_add:
            return f(all_reduce_pull_two_shot_rms_norm_gemm_add<T, NG, TM, TN, TK, SK, TPB>, bind);
          case Template::all_reduce_pull_one_shot_rms_norm_gemm:
            return f(all_reduce_pull_one_shot_rms_norm_gemm<T, NG, TM, TN, TK, SK, TPB>, bind);
          case Template::all_reduce_pull_two_shot_rms_norm_gemm:
            return f(all_reduce_pull_two_shot_rms_norm_gemm<T, NG, TM, TN, TK, SK, TPB>, bind);
          default: not_this_ops(fn);
        }
      });
    });
  });
}
template <typename F>
void dispatch(const AllReduceRmsNormGemmLaunch& l, int world, F&& f) {
  dispatch_gemm_tail(l, OpType::all_reduce_rms_norm_gemm, world, f);
}
template <typename F>
void dispatch(const AllReduceRmsNormGemmAddLaunch& l, int world, F&& f) {
  dispatch_gemm_tail(l, OpType::all_reduce_rms_norm_gemm_add, world, f);
}

template <typename F>
void dispatch(const AllReduceRmsScaleAddLaunch& l, int world, F&& f) {
  const Template fn    = launched(OpType::all_reduce_rms_scale_add, l.algorithm, l.direction);
  const KernelConfig c = RowConfig{{l.threads_per_block, l.blocks_per_grid}, l.tile_n};
  const int rows       = static_cast<int>(l.rows);
  const int hp         = static_cast<int>(packs_of(l.hidden, l.dtype));
  const int lp         = static_cast<int>(packs_of(l.latent, l.dtype));
  by_world(world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    by_dtype(l.dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      // The one-shot's and the two-shot's builds are one list.
      constexpr Template K = Template::all_reduce_pull_one_shot_rms_scale_add;
      static_assert(same_builds(K, Template::all_reduce_pull_two_shot_rms_scale_add));
      by_config<K>(c, [&](auto cc) {
        constexpr RowConfig C = decltype(cc)::value;
        constexpr int BN = C.tile_n, NT = C.launch.threads_per_block;
        const auto bind  = [&](const p2p::DevComm& p) {
          return std::make_tuple(p, static_cast<T*>(l.out), l.eps, rows, hp, lp);
        };
        switch (fn) {
          case K: return f(all_reduce_pull_one_shot_rms_scale_add<T, NG, BN, NT>, bind);
          case Template::all_reduce_pull_two_shot_rms_scale_add:
            return f(all_reduce_pull_two_shot_rms_scale_add<T, NG, BN, NT>, bind);
          default: not_this_ops(fn);
        }
      });
    });
  });
}

// EXPERIMENTAL, no peers: `bind` takes nothing.
template <typename F>
void dispatch(const AddAttnResRmsNormLaunch& l, F&& f) {
  const Template fn    = launched(OpType::add_attn_res_rms_norm, l.algorithm, l.direction);
  const KernelConfig c = AttnResConfig{{l.threads_per_block, l.blocks_per_grid}, l.tile_n,
                                       l.tile_k};
  const int rows = static_cast<int>(l.rows), packs = static_cast<int>(packs_of(l.hidden, l.dtype));
  by_dtype(l.dtype, [&](auto t) {
    using T         = typename decltype(t)::t;
    const auto bind = [&]() {
      return std::make_tuple(
          static_cast<T*>(l.prefix), static_cast<const T*>(l.delta), static_cast<T*>(l.blocks),
          l.block_stride_m, l.block_stride_r, static_cast<const T*>(l.norm_weight),
          static_cast<const T*>(l.qk_weight), static_cast<const T*>(l.out_norm_weight),
          static_cast<T*>(l.out), l.num_blocks, l.write_idx, l.eps, l.out_eps, rows, packs);
    };
    if (fn != Template::add_attn_res_rms_norm) not_this_ops(fn);
    by_config<Template::add_attn_res_rms_norm>(c, [&](auto cc) {
      constexpr AttnResConfig C = decltype(cc)::value;
      f(add_attn_res_rms_norm<T, C.tile_n, C.tile_k, C.launch.threads_per_block>, bind);
    });
  });
}

// =================================================================================================
// THE LAUNCHES.
// =================================================================================================

// The kernel at its grid and block, on the stream. hipify reads `<<<...>>>` as text, so it is
// spelled out.
template <typename... P, typename... A>
void start(void (*kernel)(P...), int blocks_per_grid, int threads_per_block, hipStream_t stream,
           A&&... args) {
  kernel<<<dim3(blocks_per_grid), dim3(threads_per_block), 0, stream>>>(std::forward<A>(args)...);
}

// Launch `l` over the peers' view `p`.
template <typename L>
void launch_on(const L& l, int world, const p2p::DevComm& p, hipStream_t s) {
  dispatch(l, world, [&](auto kernel, const auto& bind) {
    std::apply(
        [&](auto&&... xs) { start(kernel, l.blocks_per_grid, l.threads_per_block, s, xs...); },
        bind(p));
  });
}

// WHETHER A LAUNCH HOLDS ITS WHOLE GRID RESIDENT: only the compiled kernel knows what it uses.
template <typename L>
bool resident(const Handle& h, const L& l) {
  bool fits = true;
  dispatch(l, h.world_size(), [&](auto kernel, const auto&) {
    const Resources used = h.resources_of(reinterpret_cast<const void*>(kernel));
    fits = l.blocks_per_grid <= resident_blocks(kTarget, used, l.threads_per_block);
  });
  return fits;
}

// A STAGED BUILD copies the input in itself, so no peer reads it where it is.
inline void launch_all_reduce(Handle& h, const AllReduceLaunch& l, hipStream_t s) {
  launch_on(l, h.world_size(),
            l.staged ? h.dev_comm_staged(l.bytes) : h.dev_comm(l.inp, l.bytes, s), s);
}
inline void launch_all_reduce_rms_norm(Handle& h, const AllReduceRmsNormLaunch& l,
                                       hipStream_t s) {
  launch_on(l, h.world_size(), h.dev_comm(l.inp, l.rows * l.hidden * elem_bytes(l.dtype), s), s);
}
inline void launch_all_reduce_add_rms_norm(Handle& h, const AllReduceAddRmsNormLaunch& l,
                                           hipStream_t s) {
  launch_on(l, h.world_size(), h.dev_comm(l.inp, l.rows * l.hidden * elem_bytes(l.dtype), s), s);
}
inline void launch_all_reduce_add_attn_res_rms_norm(Handle& h,
                                                    const AllReduceAddAttnResRmsNormLaunch& l,
                                                    hipStream_t s) {
  launch_on(l, h.world_size(), h.dev_comm(l.inp, l.rows * l.hidden * elem_bytes(l.dtype), s), s);
}
inline void launch_all_reduce_rms_norm_gemm(Handle& h, const AllReduceRmsNormGemmLaunch& l,
                                            hipStream_t s) {
  launch_on(l, h.world_size(), h.dev_comm(l.inp, l.rows * l.hidden * elem_bytes(l.dtype), s), s);
}
inline void launch_all_reduce_rms_norm_gemm_add(Handle& h, const AllReduceRmsNormGemmAddLaunch& l,
                                                hipStream_t s) {
  launch_on(l, h.world_size(), h.dev_comm(l.inp, l.rows * l.hidden * elem_bytes(l.dtype), s), s);
}
inline void launch_all_reduce_rms_scale_add(Handle& h, const AllReduceRmsScaleAddLaunch& l,
                                            hipStream_t s) {
  const int64_t bytes = l.rows * (2 * l.hidden + l.latent) * elem_bytes(l.dtype);
  launch_on(l, h.world_size(), h.dev_comm(l.inp, bytes, s), s);
}

namespace experimental {

inline void launch_add_attn_res_rms_norm(const AddAttnResRmsNormLaunch& l, hipStream_t s) {
  dispatch(l, [&](auto kernel, const auto& bind) {
    std::apply(
        [&](auto&&... xs) { start(kernel, l.blocks_per_grid, l.threads_per_block, s, xs...); },
        bind());
  });
}

}  // namespace experimental

}  // namespace hip_comms
