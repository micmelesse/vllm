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

#include <array>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>

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

// Experimental, one rank (world 1: there are no peers): not swept.
constexpr TunedKernel kAddAttnResKernels[] = {
    {1, 1, 3584, Template::add_attn_res_rms_norm, AttnResConfig{{512, 512}, 4096, 1}},
    {1, 1, 7168, Template::add_attn_res_rms_norm, AttnResConfig{{512, 512}, 8192, 1}},
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

template <typename Config>
constexpr std::variant<Forced, Error> forced(OpType o, std::optional<Algorithm> algorithm,
                                             std::optional<Direction> direction,
                                             std::optional<int> threads_per_block,
                                             std::optional<int> blocks_per_grid,
                                             std::initializer_list<std::optional<int>> fields,
                                             Config config) {
  if (direction && !algorithm) return Error::direction_without_algorithm;
  if (threads_per_block.has_value() != blocks_per_grid.has_value())
    return Error::launch_incomplete;
  const bool launched = blocks_per_grid.has_value();
  if (launched && (*blocks_per_grid < 1 || *blocks_per_grid > p2p::kMaxBlocks ||
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
      config(LaunchConfig{*threads_per_block, *blocks_per_grid}, *fn);
  if (const Error* e = std::get_if<Error>(&c)) return *e;
  return Forced{{*fn, std::get<KernelConfig>(c)}};
}

// A FORCED FIELD's value, or 0: the template's own.
constexpr int own(std::optional<int> v) { return v.value_or(0); }

// EACH FAMILY'S FORCED CONFIG from the op's fields.
constexpr auto all_reduce_config() {
  return [](LaunchConfig l, Template) -> std::variant<KernelConfig, Error> {
    return AllReduceConfig{l};
  };
}
constexpr auto row_config(std::optional<int> tile_n) {
  return [=](LaunchConfig l, Template) -> std::variant<KernelConfig, Error> {
    return RowConfig{l, own(tile_n)};
  };
}
// AttnRes's: the pull's family (TILE_M and its reduce-scatter blocks) for the pull two-shot, the
// one-shot's and push's (neither of those) otherwise.
constexpr auto attn_res_config(std::optional<int> tile_m, std::optional<int> tile_n,
                               std::optional<int> tile_k,
                               std::optional<int> reduce_scatter_blocks) {
  return [=](LaunchConfig l, Template t) -> std::variant<KernelConfig, Error> {
    if (t == Template::all_reduce_pull_two_shot_add_attn_res_rms_norm)
      return AttnResPullConfig{l, own(tile_m), own(tile_n), own(tile_k),
                               own(reduce_scatter_blocks)};
    if (tile_m || reduce_scatter_blocks) return Error::field_not_this_templates;
    return AttnResConfig{l, own(tile_n), own(tile_k)};
  };
}
constexpr auto gemm_config(std::optional<int> tile_m, std::optional<int> tile_n,
                           std::optional<int> tile_k, std::optional<int> slice_k) {
  return [=](LaunchConfig l, Template) -> std::variant<KernelConfig, Error> {
    return GemmConfig{l, own(tile_m), own(tile_n), own(tile_k), own(slice_k)};
  };
}

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
// 8.19), uncached scratch (2026-09-30T18-00-30Z). TWO-SHOT'S BLOCK IS ONE WAVE PER PEER (aiter's
// two-stage). A GRID THE SIZE OF THE WORK, as aiter sizes its own: every block pays for every sync
// (at 16 tokens two-shot ran 11.00 us on 16 blocks, 12.31 on 64), capped at what keeps the links
// busy: their bandwidth-delay product, 7 x 76.8 GB/s x 1334 ns = 717 KB, 88 blocks of 512 threads
// (the sweep: flat from 80 to 128). Every pass full: 3.7 MB at 88 blocks took 5.09 passes, 23.78
// us, against 90 blocks in 5 full ones.
constexpr int link_filling_blocks(const Hardware& hw, const Calibration& cal, int threads) {
  const double in_flight = hw.xgmi_links * hw.xgmi_gbytes_per_s_a_way * cal.ping_pong_ns;
  const double per_pass  = static_cast<double>(threads) * kBuild.memory.pack_bytes;
  const int blocks       = static_cast<int>(in_flight / per_pass + 0.999);
  return blocks < hw.compute_units ? blocks : hw.compute_units;
}
constexpr KernelConfig all_reduce_config(Template t, int64_t bytes, int world) {
  const Hardware& hw    = kTarget;
  const bool one_shot   = t == Template::all_reduce_pull_one_shot;
  const int64_t packs   = (bytes + kBuild.memory.pack_bytes - 1) / kBuild.memory.pack_bytes;
  const int64_t work    = one_shot ? packs : (packs + world - 1) / world;
  const int64_t need    = (work + hw.wave_size - 1) / hw.wave_size;
  const int threads     = one_shot ? hw.wave_size : hw.wave_size * world;
  const int fill        = link_filling_blocks(hw, kTargetCalibration, threads);
  const int64_t passes  = need > fill ? need / fill : 1;
  const int64_t even    = (need + passes - 1) / passes;
  // NOT std::min: hipify turns it into HIP's device `min`, which is not constexpr.
  const int blocks = static_cast<int>(even < hw.compute_units ? even : hw.compute_units);
  return AllReduceConfig{{threads, blocks}};
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

// A TILED OP'S KERNEL for the call: the forced template and config (a zero field, or a config not
// forced, the template's own), else the op's tuned kernel; fitted to the call either way. `rows`
// rows of `hidden` elements; its tile holds `tile_cols` of them and its grid tiles `grid_cols`.
template <typename Config>
constexpr std::variant<std::pair<Template, KernelConfig>, Error> chosen(
    OpType o, int world, int64_t rows, int64_t hidden, int64_t tile_cols, int64_t grid_cols,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> threads_per_block, std::optional<int> blocks_per_grid,
    std::initializer_list<std::optional<int>> fields, Config config) {
  const std::variant<Forced, Error> f =
      forced(o, algorithm, direction, threads_per_block, blocks_per_grid, fields, config);
  if (const Error* e = std::get_if<Error>(&f)) return *e;
  const Forced& got = std::get<Forced>(f);
  Template fn;
  KernelConfig c;
  if (got) {
    fn = got->first;
    c  = got->second.value_or(zero_config(family_of(fn)));
  } else {
    const TunedKernel p = pick(o, world, rows, hidden, tile_cols);
    fn                  = p.fn;
    c                   = p.config;
  }
  return std::pair{fn, fitted(fn, c, rows, tile_cols, grid_cols, world)};
}

// THE PLAIN ALL-REDUCE'S KERNEL for `bytes`: forced, else pull one-shot up to the calibrated size
// and two-shot past it; its launch, where not forced, derived from the bytes.
constexpr std::variant<std::pair<Template, KernelConfig>, Error> chosen_all_reduce(
    int world, int64_t bytes, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid) {
  const std::variant<Forced, Error> f = forced(OpType::all_reduce, algorithm, direction,
                                               threads_per_block, blocks_per_grid, {},
                                               all_reduce_config());
  if (const Error* e = std::get_if<Error>(&f)) return *e;
  const Forced& got = std::get<Forced>(f);
  const Template fn = got ? got->first
                          : bytes <= kTargetCalibration.all_reduce_one_shot_max_bytes
                                ? Template::all_reduce_pull_one_shot
                                : Template::all_reduce_pull_two_shot;
  KernelConfig c = got && got->second ? *got->second : AllReduceConfig{{0, 0}};
  const KernelConfig derived = all_reduce_config(fn, bytes, world);
  LaunchConfig& l            = launch_of(c);
  if (l.threads_per_block == 0) l.threads_per_block = launch_of(derived).threads_per_block;
  if (l.blocks_per_grid == 0) l.blocks_per_grid = launch_of(derived).blocks_per_grid;
  return std::pair{fn, c};
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
  if (has_tiles(fn) && !built_at(fn, threads)) return Error::threads_not_built;
  if (has_tiles(fn) && tile_n_of(c) == 0) return Error::row_too_wide;
  if (has_tiles(fn) && !built(fn, c)) return Error::tile_not_built;
  if (has_tiles(fn) && tile_n_of(c) < tile_cols) return Error::row_too_wide;
  // TWO-SHOT'S BLOCK IS ONE WAVE PER PEER, so anything else would leave a peer unread.
  if (fn == Template::all_reduce_pull_two_shot && threads % (world * kWaveSize) != 0)
    return Error::block_not_a_wave_per_peer;
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

// =================================================================================================
// EVERY SELECTION FITS, for every op at the smallest call and a large one: an op never declines (a
// fusion that is on runs its fused op), and a kernel past a capability is a compile error, not one
// that overruns its signal slots or register arrays.
// =================================================================================================

constexpr bool fits(const std::variant<std::pair<Template, KernelConfig>, Error>& got) {
  const auto* k = std::get_if<std::pair<Template, KernelConfig>>(&got);
  if (!k) return false;
  const LaunchConfig& l = launch_of(k->second);
  if (l.blocks_per_grid < 1 || l.blocks_per_grid > p2p::kMaxBlocks) return false;
  if (has_tiles(k->first) && tile_n_of(k->second) == 0) return false;
  const int t = l.threads_per_block;
  return t >= kWaveSize && t <= kBuild.kernels.max_threads && t % kWaveSize == 0;
}
constexpr bool selections_fit() {
  constexpr std::nullopt_t none = std::nullopt;
  for (const std::array<int64_t, 3> call : {std::array<int64_t, 3>{2, 1, 8},
                                            std::array<int64_t, 3>{p2p::kMaxRanks, 4096, 7168}}) {
    const int w = static_cast<int>(call[0]);
    const int64_t rows = call[1], hidden = call[2];
    if (!fits(chosen_all_reduce(w, rows * hidden * 2, none, none, none, none))) return false;
    for (const OpType o : {OpType::all_reduce_rms_norm, OpType::all_reduce_add_rms_norm})
      if (!fits(chosen(o, w, rows, hidden, hidden, hidden, none, none, none, none, {},
                       row_config(none))))
        return false;
    // The one-all-reduce tail: [shared | projected | latent], the latent half the hidden.
    const int64_t latent = hidden / 2;
    if (!fits(chosen(OpType::all_reduce_rms_scale_add, w, rows, 2 * hidden + latent, latent,
                     hidden, none, none, none, none, {}, row_config(none))))
      return false;
    for (const OpType o : {OpType::all_reduce_rms_norm_gemm, OpType::all_reduce_rms_norm_gemm_add})
      if (!fits(chosen(o, w, rows, hidden, hidden, hidden, none, none, none, none, {},
                       gemm_config(none, none, none, none))))
        return false;
    for (const OpType o : {OpType::all_reduce_add_attn_res_rms_norm,
                           OpType::add_attn_res_rms_norm})
      if (!fits(chosen(o, o == OpType::add_attn_res_rms_norm ? 1 : w, rows, hidden, hidden,
                       hidden, none, none, none, none, {},
                       attn_res_config(none, none, none, none))))
        return false;
  }
  return true;
}
static_assert(selections_fit(), "a selection declines, or exceeds a kernel capability");

// =================================================================================================
// EACH OP'S SELECT: the op's launch, everything decided (the kernel, its config, its grid), from
// the call's primitives and its forcing, or the first Error the call meets.
// =================================================================================================

using Chosen = std::pair<Template, KernelConfig>;

constexpr std::optional<Error> weight_refused(DType dtype, DType weight_dtype) {
  if (weight_dtype != dtype && weight_dtype != DType::f32) return Error::weight_not_built;
  return std::nullopt;
}

inline std::variant<AllReduceLaunch, Error> select_all_reduce(
    const Handle& h, void* out, const void* inp, int64_t bytes, DType dtype,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> threads_per_block, std::optional<int> blocks_per_grid,
    hipStream_t stream) {
  const std::variant<Chosen, Error> got = chosen_all_reduce(
      h.world_size(), bytes, algorithm, direction, threads_per_block, blocks_per_grid);
  if (const Error* e = std::get_if<Error>(&got)) return *e;
  const auto& [fn, c] = std::get<Chosen>(got);
  // THE BUILD, from what the handle knows: in place when the peers can read the input where it
  // is, otherwise its staged build.
  const bool staged = !h.reads_in_place(inp, stream);
  if (const std::optional<Error> e =
          refused(&h, fn, c, dtype, 1, bytes / elem_bytes(dtype), 0, std::nullopt, inp, staged,
                  stream))
    return *e;
  const LaunchConfig& g = launch_of(c);
  const AllReduceLaunch l{.algorithm = algorithm_of(fn),
                          .direction = direction_of(fn),
                          .threads_per_block = g.threads_per_block,
                          .blocks_per_grid = g.blocks_per_grid,
                          .staged = staged,
                          .stream = stream,
                          .out = out,
                          .inp = inp,
                          .bytes = bytes,
                          .dtype = dtype};
  if (!resident(h, l)) return Error::grid_not_resident;
  return l;
}

inline std::variant<AllReduceRmsNormLaunch, Error> select_all_reduce_rms_norm(
    const Handle& h, void* out, const void* inp, const void* weight, DType dtype,
    DType weight_dtype, int64_t rows, int64_t hidden, float eps,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_n, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  const std::variant<Chosen, Error> got =
      chosen(OpType::all_reduce_rms_norm, h.world_size(), rows, hidden, hidden, hidden, algorithm,
             direction, threads_per_block, blocks_per_grid, {tile_n}, row_config(tile_n));
  if (const Error* e = std::get_if<Error>(&got)) return *e;
  const auto& [fn, c] = std::get<Chosen>(got);
  if (const std::optional<Error> e = refused(&h, fn, c, dtype, rows, hidden, hidden,
                                             weight_refused(dtype, weight_dtype), inp, false,
                                             stream))
    return *e;
  const RowConfig& r = std::get<RowConfig>(c);
  const AllReduceRmsNormLaunch l{.algorithm = algorithm_of(fn),
                                 .direction = direction_of(fn),
                                 .tile_n = r.tile_n,
                                 .threads_per_block = r.launch.threads_per_block,
                                 .blocks_per_grid = r.launch.blocks_per_grid,
                                 .stream = stream,
                                 .out = out,
                                 .inp = inp,
                                 .weight = weight,
                                 .dtype = dtype,
                                 .weight_dtype = weight_dtype,
                                 .rows = rows,
                                 .hidden = hidden,
                                 .eps = eps};
  if (!resident(h, l)) return Error::grid_not_resident;
  return l;
}

inline std::variant<AllReduceAddRmsNormLaunch, Error> select_all_reduce_add_rms_norm(
    const Handle& h, void* out, void* residual_out, const void* inp, const void* residual,
    const void* weight, DType dtype, DType weight_dtype, int64_t rows, int64_t hidden, float eps,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_n, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  const std::variant<Chosen, Error> got =
      chosen(OpType::all_reduce_add_rms_norm, h.world_size(), rows, hidden, hidden, hidden,
             algorithm, direction, threads_per_block, blocks_per_grid, {tile_n},
             row_config(tile_n));
  if (const Error* e = std::get_if<Error>(&got)) return *e;
  const auto& [fn, c] = std::get<Chosen>(got);
  if (const std::optional<Error> e = refused(&h, fn, c, dtype, rows, hidden, hidden,
                                             weight_refused(dtype, weight_dtype), inp, false,
                                             stream))
    return *e;
  const RowConfig& r = std::get<RowConfig>(c);
  const AllReduceAddRmsNormLaunch l{.algorithm = algorithm_of(fn),
                                    .direction = direction_of(fn),
                                    .tile_n = r.tile_n,
                                    .threads_per_block = r.launch.threads_per_block,
                                    .blocks_per_grid = r.launch.blocks_per_grid,
                                    .stream = stream,
                                    .out = out,
                                    .residual_out = residual_out,
                                    .inp = inp,
                                    .residual = residual,
                                    .weight = weight,
                                    .dtype = dtype,
                                    .weight_dtype = weight_dtype,
                                    .rows = rows,
                                    .hidden = hidden,
                                    .eps = eps};
  if (!resident(h, l)) return Error::grid_not_resident;
  return l;
}

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
  const std::variant<Chosen, Error> got =
      chosen(OpType::all_reduce_add_attn_res_rms_norm, h.world_size(), rows, hidden, hidden,
             hidden, algorithm, direction, threads_per_block, blocks_per_grid,
             {tile_m, tile_n, tile_k, reduce_scatter_blocks},
             attn_res_config(tile_m, tile_n, tile_k, reduce_scatter_blocks));
  if (const Error* e = std::get_if<Error>(&got)) return *e;
  const auto& [fn, c] = std::get<Chosen>(got);
  if (const std::optional<Error> e =
          refused(&h, fn, c, dtype, rows, hidden, hidden, std::nullopt, inp, false, stream))
    return *e;
  // THE PULL'S FAMILY has TILE_M and its reduce-scatter blocks; the others are a row a tile.
  const AttnResPullConfig* pull = std::get_if<AttnResPullConfig>(&c);
  const LaunchConfig& g         = launch_of(c);
  const int tm = pull ? pull->tile_m : 1;
  const int tk = pull ? pull->tile_k : std::get<AttnResConfig>(c).tile_k;
  const int rs = pull ? pull->reduce_scatter_blocks : 0;
  const AllReduceAddAttnResRmsNormLaunch l{.algorithm = algorithm_of(fn),
                                           .direction = direction_of(fn),
                                           .tile_m = tm,
                                           .tile_n = tile_n_of(c),
                                           .tile_k = tk,
                                           .threads_per_block = g.threads_per_block,
                                           .blocks_per_grid = g.blocks_per_grid,
                                           .reduce_scatter_blocks = rs,
                                           .stream = stream,
                                           .prefix = prefix,
                                           .out = out,
                                           .inp = inp,
                                           .blocks = blocks,
                                           .block_stride_m = block_stride_m,
                                           .block_stride_r = block_stride_r,
                                           .norm_weight = norm_weight,
                                           .qk_weight = qk_weight,
                                           .out_norm_weight = out_norm_weight,
                                           .dtype = dtype,
                                           .rows = rows,
                                           .hidden = hidden,
                                           .num_blocks = num_blocks,
                                           .write_idx = write_idx,
                                           .eps = eps,
                                           .out_eps = out_eps,
                                           .has_prefix = has_prefix};
  if (!resident(h, l)) return Error::grid_not_resident;
  return l;
}

inline std::variant<AllReduceRmsNormGemmLaunch, Error> select_all_reduce_rms_norm_gemm(
    const Handle& h, void* out, int64_t out_stride, const void* inp, const void* norm_weight,
    float eps, const void* gemm_weight, int64_t n_cols, void* workspace, DType dtype,
    int64_t rows, int64_t hidden, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> tile_m, std::optional<int> tile_n,
    std::optional<int> tile_k, std::optional<int> slice_k, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  const std::variant<Chosen, Error> got =
      chosen(OpType::all_reduce_rms_norm_gemm, h.world_size(), rows, hidden, hidden, hidden,
             algorithm, direction, threads_per_block, blocks_per_grid,
             {tile_m, tile_n, tile_k, slice_k}, gemm_config(tile_m, tile_n, tile_k, slice_k));
  if (const Error* e = std::get_if<Error>(&got)) return *e;
  const auto& [fn, c] = std::get<Chosen>(got);
  if (const std::optional<Error> e =
          refused(&h, fn, c, dtype, rows, hidden, hidden, std::nullopt, inp, false, stream))
    return *e;
  const GemmConfig& g = std::get<GemmConfig>(c);
  const AllReduceRmsNormGemmLaunch l{.algorithm = algorithm_of(fn),
                                     .direction = direction_of(fn),
                                     .tile_m = g.tile_m,
                                     .tile_n = g.tile_n,
                                     .tile_k = g.tile_k,
                                     .slice_k = g.slice_k,
                                     .threads_per_block = g.launch.threads_per_block,
                                     .blocks_per_grid = g.launch.blocks_per_grid,
                                     .stream = stream,
                                     .out = out,
                                     .out_stride = out_stride,
                                     .inp = inp,
                                     .norm_weight = norm_weight,
                                     .eps = eps,
                                     .gemm_weight = gemm_weight,
                                     .n_cols = n_cols,
                                     .workspace = workspace,
                                     .dtype = dtype,
                                     .rows = rows,
                                     .hidden = hidden};
  if (!resident(h, l)) return Error::grid_not_resident;
  return l;
}

inline std::variant<AllReduceRmsNormGemmAddLaunch, Error> select_all_reduce_rms_norm_gemm_add(
    const Handle& h, void* out, int64_t out_stride, const void* inp, const void* norm_weight,
    float eps, const void* gemm_weight, int64_t n_cols, void* workspace, DType dtype,
    int64_t rows, int64_t hidden, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> tile_m, std::optional<int> tile_n,
    std::optional<int> tile_k, std::optional<int> slice_k, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  const std::variant<Chosen, Error> got =
      chosen(OpType::all_reduce_rms_norm_gemm_add, h.world_size(), rows, hidden, hidden, hidden,
             algorithm, direction, threads_per_block, blocks_per_grid,
             {tile_m, tile_n, tile_k, slice_k}, gemm_config(tile_m, tile_n, tile_k, slice_k));
  if (const Error* e = std::get_if<Error>(&got)) return *e;
  const auto& [fn, c] = std::get<Chosen>(got);
  if (const std::optional<Error> e =
          refused(&h, fn, c, dtype, rows, hidden, hidden, std::nullopt, inp, false, stream))
    return *e;
  const GemmConfig& g = std::get<GemmConfig>(c);
  const AllReduceRmsNormGemmAddLaunch l{.algorithm = algorithm_of(fn),
                                        .direction = direction_of(fn),
                                        .tile_m = g.tile_m,
                                        .tile_n = g.tile_n,
                                        .tile_k = g.tile_k,
                                        .slice_k = g.slice_k,
                                        .threads_per_block = g.launch.threads_per_block,
                                        .blocks_per_grid = g.launch.blocks_per_grid,
                                        .stream = stream,
                                        .out = out,
                                        .out_stride = out_stride,
                                        .inp = inp,
                                        .norm_weight = norm_weight,
                                        .eps = eps,
                                        .gemm_weight = gemm_weight,
                                        .n_cols = n_cols,
                                        .workspace = workspace,
                                        .dtype = dtype,
                                        .rows = rows,
                                        .hidden = hidden};
  if (!resident(h, l)) return Error::grid_not_resident;
  return l;
}

inline std::variant<AllReduceRmsScaleAddLaunch, Error> select_all_reduce_rms_scale_add(
    const Handle& h, void* out, const void* inp, DType dtype, int64_t rows, int64_t hidden,
    int64_t latent, float eps, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> tile_n,
    std::optional<int> threads_per_block, std::optional<int> blocks_per_grid,
    hipStream_t stream) {
  const int64_t row = 2 * hidden + latent;
  const std::variant<Chosen, Error> got =
      chosen(OpType::all_reduce_rms_scale_add, h.world_size(), rows, row, latent, hidden,
             algorithm, direction, threads_per_block, blocks_per_grid, {tile_n},
             row_config(tile_n));
  if (const Error* e = std::get_if<Error>(&got)) return *e;
  const auto& [fn, c] = std::get<Chosen>(got);
  const int e = elem_bytes(dtype);
  const std::optional<Error> widths =
      hidden * e % kBuild.memory.pack_bytes != 0 || latent * e % kBuild.memory.pack_bytes != 0 ||
              latent < 1
          ? std::optional<Error>{Error::widths_not_packs}
          : std::nullopt;
  if (const std::optional<Error> err =
          refused(&h, fn, c, dtype, rows, row, latent, widths, inp, false, stream))
    return *err;
  const RowConfig& r = std::get<RowConfig>(c);
  const AllReduceRmsScaleAddLaunch l{.algorithm = algorithm_of(fn),
                                     .direction = direction_of(fn),
                                     .tile_n = r.tile_n,
                                     .threads_per_block = r.launch.threads_per_block,
                                     .blocks_per_grid = r.launch.blocks_per_grid,
                                     .stream = stream,
                                     .out = out,
                                     .inp = inp,
                                     .dtype = dtype,
                                     .rows = rows,
                                     .hidden = hidden,
                                     .latent = latent,
                                     .eps = eps};
  if (!resident(h, l)) return Error::grid_not_resident;
  return l;
}

namespace experimental {

inline std::variant<AddAttnResRmsNormLaunch, Error> select_add_attn_res_rms_norm(
    void* prefix, void* out, const void* delta, void* blocks, int64_t block_stride_m,
    int64_t block_stride_r, const void* norm_weight, const void* qk_weight,
    const void* out_norm_weight, DType dtype, int64_t rows, int64_t hidden, int num_blocks,
    int write_idx, float eps, float out_eps, std::optional<int> tile_n,
    std::optional<int> tile_k, std::optional<int> threads_per_block,
    std::optional<int> blocks_per_grid, hipStream_t stream) {
  const std::variant<Chosen, Error> got =
      chosen(OpType::add_attn_res_rms_norm, 1, rows, hidden, hidden, hidden, std::nullopt,
             std::nullopt, threads_per_block, blocks_per_grid, {tile_n, tile_k},
             attn_res_config(std::nullopt, tile_n, tile_k, std::nullopt));
  if (const Error* e = std::get_if<Error>(&got)) return *e;
  const auto& [fn, c] = std::get<Chosen>(got);
  if (const std::optional<Error> e = refused(nullptr, fn, c, dtype, rows, hidden, hidden,
                                             std::nullopt, delta, false, stream))
    return *e;
  const AttnResConfig& a = std::get<AttnResConfig>(c);
  return AddAttnResRmsNormLaunch{.algorithm = algorithm_of(fn),
                                 .direction = direction_of(fn),
                                 .tile_n = a.tile_n,
                                 .tile_k = a.tile_k,
                                 .threads_per_block = a.launch.threads_per_block,
                                 .blocks_per_grid = a.launch.blocks_per_grid,
                                 .stream = stream,
                                 .prefix = prefix,
                                 .out = out,
                                 .delta = delta,
                                 .blocks = blocks,
                                 .block_stride_m = block_stride_m,
                                 .block_stride_r = block_stride_r,
                                 .norm_weight = norm_weight,
                                 .qk_weight = qk_weight,
                                 .out_norm_weight = out_norm_weight,
                                 .dtype = dtype,
                                 .rows = rows,
                                 .hidden = hidden,
                                 .num_blocks = num_blocks,
                                 .write_idx = write_idx,
                                 .eps = eps,
                                 .out_eps = out_eps};
}

}  // namespace experimental

}  // namespace hip_comms
