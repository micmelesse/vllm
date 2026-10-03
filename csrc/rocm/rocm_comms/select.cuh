// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SELECT, THE ONLY CHOICE: `select(args, world, options)` is the kernel a call runs, in three
// steps: the pick (the op's tuned kernel for the call, op.cuh, or the caller's
// forced template and config), the template's arguments (read off the call), and the pick's
// config fitted to the call (a forced zero field the template's own, the tile widened to cover the
// row, the grid cut to the tiles there are).

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>
#include <variant>

namespace hip_comms {

constexpr int elem_bytes(DType d) { return d == DType::f32 ? 4 : 2; }

// =================================================================================================
// A CALL, as select reads it (its op is op.cuh's op_of): rows (a plain all-reduce is one row), a
// row's elements, the bytes the hardware moves; and the columns its tile must hold and its grid
// tiles.
// =================================================================================================


constexpr int64_t rows_of(const AllReduceArgs&) { return 1; }
constexpr int64_t rows_of(const NormArgs& a) { return a.rows; }
constexpr int64_t rows_of(const AttnResArgs& a) { return a.rows; }
constexpr int64_t rows_of(const GemmTailArgs& a) { return a.rows; }
constexpr int64_t rows_of(const ScaleAddArgs& a) { return a.rows; }
constexpr int64_t rows_of(const AddAttnResArgs& a) { return a.rows; }
constexpr int64_t hidden_of(const AllReduceArgs& a) { return a.bytes / elem_bytes(a.dtype); }
constexpr int64_t hidden_of(const NormArgs& a) { return a.hidden; }
constexpr int64_t hidden_of(const AttnResArgs& a) { return a.hidden; }
constexpr int64_t hidden_of(const GemmTailArgs& a) { return a.hidden; }
constexpr int64_t hidden_of(const AddAttnResArgs& a) { return a.hidden; }
// The row reduced, [shared | projected | latent].
constexpr int64_t hidden_of(const ScaleAddArgs& a) { return 2 * a.hidden + a.latent; }
template <typename Args>
constexpr int64_t bytes_of(const Args& a) {
  return rows_of(a) * hidden_of(a) * elem_bytes(a.dtype);
}
template <typename Args>
constexpr int64_t packs_of(const Args& a) {
  return hidden_of(a) * elem_bytes(a.dtype) / kBuild.memory.pack_bytes;
}

// THE COLUMNS ONE TILE HOLDS, and THE COLUMNS THE GRID TILES. A row op's tile holds its whole row,
// one tile a row. The one-all-reduce tail's holds the latent whole (each tile needs the row's RMS)
// beside a TILE_N slice of the hidden, so its grid tiles the hidden.
template <typename Args>
constexpr int64_t tile_cols(const Args& a) {
  return hidden_of(a);
}
constexpr int64_t tile_cols(const ScaleAddArgs& a) { return a.latent; }
template <typename Args>
constexpr int64_t grid_cols(const Args& a) {
  return hidden_of(a);
}
constexpr int64_t grid_cols(const ScaleAddArgs& a) { return a.hidden; }

// A COLUMN TWO-SHOT reduce-scatters by columns, so every block keeps every row; a row two-shot
// gives each rank its rows.
constexpr bool slices_columns(Template k) {
  return k == Template::all_reduce_pull_two_shot_add_attn_res_rms_norm ||
         k == Template::all_reduce_push_two_shot_rms_norm ||
         k == Template::all_reduce_push_two_shot_add_rms_norm ||
         k == Template::all_reduce_push_two_shot_add_attn_res_rms_norm;
}

// =================================================================================================
// 2. THE TEMPLATE'S ARGUMENTS: what the call fixes in the template, read off it. The plain
// all-reduce's staged build is plan's (it depends on the handle).
// =================================================================================================

constexpr TemplateArgs template_args(const AllReduceArgs& a, int world) {
  return AllReduceTemplateArgs{world, a.dtype, false};
}
constexpr TemplateArgs template_args(const NormArgs& a, int world) {
  return NormTemplateArgs{world, a.dtype, a.weight_dtype};
}
constexpr TemplateArgs template_args(const AttnResArgs& a, int world) {
  return AttnResTemplateArgs{world, a.dtype, a.has_prefix};
}
constexpr TemplateArgs template_args(const GemmTailArgs& a, int world) {
  return GemmTemplateArgs{world, a.dtype};
}
constexpr TemplateArgs template_args(const ScaleAddArgs& a, int world) {
  return ScaleAddTemplateArgs{world, a.dtype};
}
constexpr TemplateArgs template_args(const AddAttnResArgs& a, int world) {
  return AttnResTemplateArgs{world, a.dtype, true};
}

// =================================================================================================
// 1. THE PICK.
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
constexpr KernelConfig all_reduce_config(Template t, const AllReduceArgs& a, int world) {
  const Hardware& hw    = kTarget;
  const bool one_shot   = t == Template::all_reduce_pull_one_shot;
  const int64_t packs   = (bytes_of(a) + kBuild.memory.pack_bytes - 1) / kBuild.memory.pack_bytes;
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
constexpr TunedKernel pick(const AllReduceArgs& a, int world) {
  const Template t = bytes_of(a) <= kTargetCalibration.all_reduce_one_shot_max_bytes
                         ? Template::all_reduce_pull_one_shot
                         : Template::all_reduce_pull_two_shot;
  return {world, rows_of(a), hidden_of(a), t, all_reduce_config(t, a, world)};
}

constexpr int64_t distance(int64_t a, int64_t b) { return a < b ? b - a : a - b; }

// EVERY OTHER OP: its tuned kernel for the call. Among the op's kernels, those at the call's world,
// else at the nearest tuned world; among those, the call's width, else the nearest tuned width,
// reading the call's rows as that width's rows of the same bytes (a crossover is about bytes);
// then the last entry whose rows the call reaches, the first when it reaches none. At a width the
// entry's tile does not cover, its tile_n is left to fitting (the smallest built that covers).
template <typename Args>
constexpr TunedKernel pick(const Args& a, int world) {
  const std::span<const TunedKernel> kernels = op(op_of(a)).kernels;
  int w = 0;
  for (const TunedKernel& k : kernels)
    if (w == 0 || distance(k.world, world) < distance(w, world)) w = k.world;
  int64_t h = 0;
  for (const TunedKernel& k : kernels)
    if (k.world == w && (h == 0 || distance(k.hidden, hidden_of(a)) < distance(h, hidden_of(a))))
      h = k.hidden;
  const int64_t rows       = rows_of(a) * hidden_of(a) / h;
  const TunedKernel* first = nullptr;
  const TunedKernel* found = nullptr;
  for (const TunedKernel& k : kernels) {
    if (k.world != w || k.hidden != h) continue;
    if (!first || k.rows < first->rows) first = &k;
    if (k.rows <= rows && (!found || k.rows > found->rows)) found = &k;
  }
  TunedKernel got = found ? *found : *first;
  if (tile_n_of(got.config) < tile_cols(a)) set_tile_n(got.config, 0);
  return got;
}

// =================================================================================================
// 3. THE PICK'S CONFIG FITTED TO THE CALL.
// =================================================================================================

// THE TEMPLATE'S OWN CONFIG for the call: its first listed config whose tile covers the call's
// columns (its first when none does), or the plain all-reduce's derived one.
template <typename Args>
constexpr KernelConfig default_config(Template t, const Args& a, int world) {
  if constexpr (std::is_same_v<Args, AllReduceArgs>) {
    return all_reduce_config(t, a, world);
  } else {
    for (const KernelConfig& c : configs_of(t))
      if (tile_n_of(c) >= tile_cols(a)) return c;
    return configs_of(t)[0];
  }
}

// THE SMALLEST BUILT TILE_N covering `cols` at `c`'s other fields, or 0 when none does (check
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
// otherwise) in TILE_M, by the grid's columns in TILE_N. A grid wider is idle blocks, each still
// paying every barrier (the pull norm at 32 tokens ran 36 blocks for 4 rows a rank). The plain
// all-reduce strides over packs and the GEMM tail's GEMM over column tiles, so neither is cut.
template <typename Args>
constexpr int64_t tiles_of(Template t, const KernelConfig& c, const Args& a, int world) {
  const OpType op = op_of(t);
  if (op == OpType::all_reduce || gemms(op)) return launch_of(c).blocks_per_grid;
  const bool row_split = is_two_shot(t) && !slices_columns(t);
  const int64_t rows   = row_split ? (rows_of(a) + world - 1) / world : rows_of(a);
  const int64_t cols   = grid_cols(a);
  const int64_t tm = tile_m_of(c), tn = tile_n_of(c);
  return (rows + tm - 1) / tm * ((cols + tn - 1) / tn);
}

// THE PICK'S CONFIG FITTED TO THE CALL: a zero field the template's own (another family's config
// is left for check to refuse), tile_n the smallest built covering the row where none is given,
// and the grid cut to the tiles there are.
template <typename Args>
constexpr KernelConfig fitted(Template t, KernelConfig c, const Args& a, int world) {
  const KernelConfig own = default_config(t, a, world);
  if (c.index() != own.index()) return c;
  std::visit(
      [&](auto& f) {
        const auto& o     = std::get<std::decay_t<decltype(f)>>(own);
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
  if (!has_tiles(t)) return c;
  if (tile_n_of(c) == 0) set_tile_n(c, smallest_tile_n(t, c, tile_cols(a)));
  if (tile_n_of(c) == 0) return c;
  const int64_t tiles = tiles_of(t, c, a, world);
  int& blocks         = launch_of(c).blocks_per_grid;
  if (tiles < blocks) blocks = static_cast<int>(tiles > 0 ? tiles : 1);
  return c;
}

// =================================================================================================
// THE ONE ENTRY.
// =================================================================================================

template <typename Args>
constexpr Kernel select(const Args& a, int world, const Options& o) {
  const TunedKernel p = o.fn ? TunedKernel{world, rows_of(a), hidden_of(a), *o.fn,
                                           o.kernel_config.value_or(zero_config(family_of(*o.fn)))}
                             : pick(a, world);
  return Kernel{p.fn, template_args(a, world), fitted(p.fn, p.config, a, world)};
}

// WHETHER A KERNEL IS ITS STAGED BUILD: the plain all-reduce's, when plan found an eager input.
constexpr bool is_staged(const Kernel& k) {
  const auto* a = std::get_if<AllReduceTemplateArgs>(&k.args);
  return a && a->staged;
}

// EVERY SELECTED KERNEL FITS, for every op at the smallest call and a large one: an op never
// declines (a fusion that is on runs its fused op), and a kernel past a capability is a compile
// error, not one that overruns its signal slots or register arrays.
constexpr bool fits(const Kernel& k) {
  const LaunchConfig& l = launch_of(k.config);
  if (l.blocks_per_grid < 1 || l.blocks_per_grid > p2p::kMaxBlocks) return false;
  if (has_tiles(k.fn) && tile_n_of(k.config) == 0) return false;
  const int t = l.threads_per_block;
  return t >= kWaveSize && t <= kBuild.kernels.max_threads && t % kWaveSize == 0;
}
constexpr bool selections_fit() {
  const Options o{std::nullopt, std::nullopt, nullptr};
  constexpr DType bf = DType::bf16;
  for (const std::array<int64_t, 3> call : {std::array<int64_t, 3>{2, 1, 8},
                                            std::array<int64_t, 3>{p2p::kMaxRanks, 4096, 7168}}) {
    const int w = static_cast<int>(call[0]);
    const int64_t rows = call[1], hidden = call[2];
    if (!fits(select(AllReduceArgs{nullptr, nullptr, rows * hidden * 2, bf}, w, o))) return false;
    for (const bool add : {false, true}) {
      if (!fits(select(NormArgs{add, nullptr, nullptr, nullptr, bf, bf, rows, hidden, 0.f,
                                nullptr, nullptr}, w, o)))
        return false;
      if (!fits(select(GemmTailArgs{.add = add, .dtype = bf, .rows = rows,
                                    .hidden = hidden}, w, o)))
        return false;
    }
    if (!fits(select(AttnResArgs{nullptr, nullptr, nullptr, nullptr, 0, 0, nullptr, nullptr,
                                 nullptr, bf, rows, hidden, 0, -1, 0.f, 0.f, true}, w, o)))
      return false;
    if (!fits(select(ScaleAddArgs{nullptr, nullptr, bf, rows, hidden, hidden / 2, 0.f}, w, o)))
      return false;
    if (!fits(select(AddAttnResArgs{.dtype = bf, .rows = rows, .hidden = hidden}, 1, o)))
      return false;
  }
  return true;
}
static_assert(selections_fit(), "a selection declines, or exceeds a kernel capability");

}  // namespace hip_comms
