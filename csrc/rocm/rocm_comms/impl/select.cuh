// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SELECT, THE ONLY CHOICE: `select(args, world, options)` is the kernel a call runs: the template,
// its arguments, its grid and block. Each op has its own rule (tune_<op>), which picks the template
// and the launch from the call, the hardware's documented facts and what was measured on it
// (machine/hardware.cuh), one rule in one place with the sweep it came from beside it; a forced
// template takes the rule's place. `kernel_for` then decides the template's arguments from the call.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <array>
#include <cstdint>
#include <optional>
#include <variant>

namespace hip_comms {

constexpr int elem_bytes(DType d) { return d == DType::f32 ? 4 : 2; }

// A CALL'S SHAPE, as the rules read it: rows (a plain all-reduce is one row), a row's elements, and
// the bytes the hardware moves.
constexpr int64_t rows_of(const AllReduceArgs&) { return 1; }
constexpr int64_t rows_of(const NormArgs& a) { return a.rows; }
constexpr int64_t rows_of(const AttnResArgs& a) { return a.rows; }
constexpr int64_t rows_of(const GemmTailArgs& a) { return a.rows; }
constexpr int64_t rows_of(const ScaleAddArgs& a) { return a.rows; }
constexpr int64_t hidden_of(const AllReduceArgs& a) { return a.bytes / elem_bytes(a.dtype); }
constexpr int64_t hidden_of(const NormArgs& a) { return a.hidden; }
constexpr int64_t hidden_of(const AttnResArgs& a) { return a.hidden; }
constexpr int64_t hidden_of(const GemmTailArgs& a) { return a.hidden; }
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

// A ROW OP GIVES EACH BLOCK WHOLE ROWS, so it needs no more blocks than it has rows: a one-shot
// and a column two-shot (the pushes, and AttnRes's pull) all of them, a row two-shot its rank's. An idle
// block still pays every barrier (each pairs with its twin on every peer): the pull norm at 32
// tokens ran 36 blocks for 4 rows a rank. The GEMM tail's GEMM strides over column tiles, and the
// plain all-reduce over packs, so both keep theirs.
constexpr bool slices_columns(Template k) {
  return k == Template::all_reduce_pull_two_shot_add_attn_res_rms_norm ||
         k == Template::all_reduce_push_two_shot_rms_norm ||
         k == Template::all_reduce_push_two_shot_add_rms_norm ||
         k == Template::all_reduce_push_two_shot_add_attn_res_rms_norm;
}
constexpr int grid_of(Template k, int blocks, int64_t rows, int world) {
  const Op op = op_of(k);
  if (op == Op::all_reduce || gemms(op)) return blocks;
  const bool row_slice = is_two_shot(k) && !slices_columns(k);
  const int64_t mine   = row_slice ? (rows + world - 1) / world : rows;
  return mine < blocks ? static_cast<int>(mine) : blocks;
}

// THE COLUMNS A ROW TEMPLATE'S TILE COVERS: the row, or the one-all-reduce tail's latent (its
// hidden is cut into slices of the tile's width).
template <typename Args>
constexpr int64_t tile_cols(const Args& a) {
  return hidden_of(a);
}
constexpr int64_t tile_cols(const ScaleAddArgs& a) { return a.latent; }
// A FORCED LAUNCH'S TILE: one row, the smallest built TILE_N covering the call's columns at
// `threads`, or 0 when none does (check refuses it).
template <typename Args>
constexpr int tile_n_of(Template t, const Args& a, int threads) {
  return tile_n_for(t, tile_cols(a), 1, threads);
}
// THE KERNEL: template `t` at `blocks` x `threads` with its arguments and tile from the call, its
// grid cut to the rows where it gives each block a row.
constexpr Kernel kernel_for(Template t, int blocks, int threads, const AllReduceArgs& a,
                            int world) {
  return {t, AllReduceTemplateArgs{world, a.dtype, false},
          KernelConfig{0, 0, threads, grid_of(t, blocks, rows_of(a), world)}};
}
constexpr Kernel kernel_for(Template t, int blocks, int threads, const NormArgs& a, int world) {
  return {
      t, NormTemplateArgs{world, a.dtype, a.weight_dtype},
      KernelConfig{1, tile_n_of(t, a, threads), threads, grid_of(t, blocks, rows_of(a), world)}};
}
constexpr Kernel kernel_for(Template t, int blocks, int threads, const AttnResArgs& a,
                            int world) {
  return {
      t, AttnResTemplateArgs{world, a.dtype, a.has_prefix},
      KernelConfig{1, tile_n_of(t, a, threads), threads, grid_of(t, blocks, rows_of(a), world)}};
}
constexpr Kernel kernel_for(Template t, int blocks, int threads, const GemmTailArgs& a,
                            int world) {
  const int lanes = kBuild.kernels.gemm_lanes;
  return {
      t, GemmTemplateArgs{world, a.dtype, lanes},
      KernelConfig{1, tile_n_of(t, a, threads), threads, grid_of(t, blocks, rows_of(a), world)}};
}
// THE ONE-ALL-REDUCE TAIL: a block holds the whole latent (its build) and a slice of the hidden
// as wide, so a row takes `splits` blocks.
constexpr Kernel kernel_for(Template t, int blocks, int threads, const ScaleAddArgs& a,
                            int world) {
  const int e = elem_bytes(a.dtype);
  const int tile_n = tile_n_of(t, a, threads);
  const int64_t span = tile_n > 0 ? tile_n * e / kBuild.memory.pack_bytes : threads;
  const int64_t hp = a.hidden * e / kBuild.memory.pack_bytes;
  const int splits = static_cast<int>((hp + span - 1) / span);
  // A block a (row, slice): every row's in the one-shot, this rank's in the two-shot.
  const int64_t mine = is_two_shot(t) ? (a.rows + world - 1) / world : a.rows;
  const int64_t work = mine * splits;
  return {t, ScaleAddTemplateArgs{world, a.dtype, splits},
          KernelConfig{1, tile_n, threads,
                       static_cast<int>(work < blocks ? (work > 0 ? work : 1) : blocks)}};
}

// THE KERNEL AT ITS TEMPLATE'S DEFAULT: the first of its configs whose tile covers the call, at
// that config's threads and grid (cut to the rows); a tile_n of 0 when none covers it (check
// refuses it).
template <typename Args>
constexpr Kernel kernel_for(Template t, const Args& a, int world) {
  const std::optional<KernelConfig> c = config_for(t, tile_cols(a));
  const KernelConfig d = c ? *c : configs_of(t)[0];
  Kernel k = kernel_for(t, d.blocks_per_grid, d.threads_per_block, a, world);
  k.config.tile_m = d.tile_m;
  k.config.tile_n = c ? d.tile_n : 0;
  return k;
}

// =================================================================================================
// ALL-REDUCE: the critical path, from the launch-config sweep on n11 (bench, 2026-09-29T19-24-48Z:
// tokens 1-64 at hidden 3584 bf16, blocks 1-64, threads 64-512).
//
// ONE WAVE PER BLOCK. Waves in one block only add a barrier inside it: at 1 token one-shot took
// 7.8 us at 64 threads, 8.1 at 128, 9.0 at 256 and 10.9 at 512.
//
// PULL ONE-SHOT UP TO 64 KiB, PULL TWO-SHOT PAST IT, at that width. One-shot reads every peer's
// whole buffer ((N-1)P) in one round trip; two-shot moves less (2(N-1)/N P) in two. With the
// scratch uncached, one-shot won at 56 KiB (7.12 vs 7.83 us), two-shot at 112 KiB (7.87 vs 8.19).
// Calibration's one_shot_max_bytes: the alpha-beta model's floors cross near 200 KB, but our
// one-shot costs more a byte than the model says, so the measured crossover stands.
// =================================================================================================

// A GRID THE SIZE OF THE WORK, as aiter sizes its own: every block pays for every sync, so a block
// with no pack to move is pure cost (at 16 tokens two-shot ran 11.00 us on 16 blocks, 12.31 on 64).
// A wave's lanes take one pack each in a pass, up to the signal slots and the compute units.
// TWO-SHOT'S BLOCK IS ONE WAVE PER PEER (aiter's two-stage), so it is the wave times the world.
// THE GRID CAP: enough blocks to keep the links busy, and no more, since every block syncs with
// its partner on every peer. The links' bandwidth-delay product is what must be in flight (the
// bandwidth documented, the round trip measured), and a block moves `threads` packs a pass:
// 7 x 76.8 GB/s x 1334 ns = 717 KB, 88 blocks of 512 threads. The sweep agrees (flat from 80 to
// 128; 1.8 MB: 15.42 us at 64, 14.72 at 80, 16.70 at 256).
constexpr int link_filling_blocks(const Hardware& hw, const Calibration& cal, int threads) {
  const double in_flight = hw.xgmi_links * hw.xgmi_gbytes_per_s_a_way * cal.ping_pong_ns;
  const double per_pass = static_cast<double>(threads) * kBuild.memory.pack_bytes;
  const int blocks       = static_cast<int>(in_flight / per_pass + 0.999);
  return blocks < hw.compute_units ? blocks : hw.compute_units;
}

constexpr Kernel tune_all_reduce(const AllReduceArgs& a, int world, const Hardware& hw,
                                 const Calibration& cal) {
  const bool one_shot = bytes_of(a) <= cal.all_reduce_one_shot_max_bytes;
  const Template k = one_shot ? Template::all_reduce_pull_one_shot : Template::all_reduce_pull_two_shot;
  const int64_t packs = (bytes_of(a) + kBuild.memory.pack_bytes - 1) / kBuild.memory.pack_bytes;
  const int64_t work  = one_shot ? packs : (packs + world - 1) / world;
  const int64_t need  = (work + hw.wave_size - 1) / hw.wave_size;
  const int threads = one_shot ? hw.wave_size : hw.wave_size * world;
  // AT LEAST THE LINK-FILLING GRID A PASS, AND EVERY PASS FULL: as many passes as keep each one
  // filling the links, the work spread evenly over them. A cap alone left a near-empty last pass,
  // a whole round trip for a sliver (3.7 MB at 88 blocks: 5.09 passes, 23.78 us, against 90 blocks
  // in 5 full ones; the sweep's 80 was 22.99).
  const int fill       = link_filling_blocks(hw, cal, threads);
  const int64_t passes = need > fill ? need / fill : 1;
  const int64_t even   = (need + passes - 1) / passes;
  // NOT std::min: hipify turns it into HIP's device `min`, which is not constexpr.
  const int blocks = static_cast<int>(even < hw.compute_units ? even : hw.compute_units);
  return kernel_for(k, blocks, threads, a, world);
}

// =================================================================================================
// THE FUSED OPS: one-shot up to Calibration's fused_one_shot_max_bytes, two-shot past it, at its
// fused grids and block.
// =================================================================================================

// THE NORMS ALWAYS RUN FUSED: a fusion flag means the fused op runs, and tuning picks among fused
// kernels, never unfused. The one-shot up to fused_one_shot_max_bytes (7.46 against 10.00 us at 1
// token, 9.95 against 10.34 at 16, 2026-09-30T21-30-15Z); the push two-shot (aiter's column split)
// up to the op's push_max_bytes, where it wins; the pull two-shot (rows) past it, at prefill.
constexpr Kernel fused_norm(Template one_shot, Template push, Template pull, const NormArgs& a,
                            int world, const NormCalibration& c) {
  if (bytes_of(a) <= c.one_shot_max_bytes) return kernel_for(one_shot, a, world);
  if (bytes_of(a) <= c.push_max_bytes) return kernel_for(push, a, world);
  return kernel_for(pull, a, world);
}

constexpr Kernel tune_all_reduce_rms_norm(const NormArgs& a, int world, const Hardware&,
                                          const Calibration& cal) {
  return fused_norm(Template::all_reduce_pull_one_shot_rms_norm,
                    Template::all_reduce_push_two_shot_rms_norm,
                    Template::all_reduce_pull_two_shot_rms_norm, a, world, cal.rms_norm);
}

constexpr Kernel tune_all_reduce_add_rms_norm(const NormArgs& a, int world, const Hardware&,
                                              const Calibration& cal) {
  return fused_norm(Template::all_reduce_pull_one_shot_add_rms_norm,
                    Template::all_reduce_push_two_shot_add_rms_norm,
                    Template::all_reduce_pull_two_shot_add_rms_norm, a, world, cal.add_rms_norm);
}

// AttnRes as the norms: a block a row, never more blocks than rows (grid_of).
constexpr Kernel tune_all_reduce_add_attn_res_rms_norm(const AttnResArgs& a, int world,
                                                       const Hardware&, const Calibration& cal) {
  const AttnResCalibration& c = cal.attn_res;
  if (bytes_of(a) <= c.one_shot_max_bytes)
    return kernel_for(Template::all_reduce_pull_one_shot_add_attn_res_rms_norm, a, world);
  if (bytes_of(a) <= c.push_max_bytes)
    return kernel_for(Template::all_reduce_push_two_shot_add_attn_res_rms_norm, a, world);
  return kernel_for(Template::all_reduce_pull_two_shot_add_attn_res_rms_norm, a, world);
}

// ALWAYS FUSED, as every op: one-shot up to one GEMM pass of rows, two-shot past it, 56 blocks of
// 512 threads (the GEMM strides over column tiles). It is slower than the unfused ops (about 68
// against 20 us at 1 token, 2026-09-30T20-23-38Z): a loss to fix, shown as one.
constexpr Kernel gemm(Template one_shot, Template two_shot, const GemmTailArgs& a, int world,
                      const GemmCalibration& c) {
  return rows_of(a) <= c.one_shot_max_rows ? kernel_for(one_shot, a, world)
                                           : kernel_for(two_shot, a, world);
}

constexpr Kernel tune_all_reduce_rms_norm_gemm(const GemmTailArgs& a, int world, const Hardware&,
                                               const Calibration& cal) {
  return gemm(Template::all_reduce_pull_one_shot_rms_norm_gemm,
              Template::all_reduce_pull_two_shot_rms_norm_gemm, a, world, cal.rms_norm_gemm);
}

constexpr Kernel tune_all_reduce_rms_norm_gemm_add(const GemmTailArgs& a, int world,
                                                   const Hardware&, const Calibration& cal) {
  return gemm(Template::all_reduce_pull_one_shot_rms_norm_gemm_add,
              Template::all_reduce_pull_two_shot_rms_norm_gemm_add, a, world,
              cal.rms_norm_gemm_add);
}

// THE ONE-ALL-REDUCE TAIL: the one-shot while there are fewer rows than ranks (the two-shot would
// leave ranks idle), the row two-shot from a row a rank. Measured at [T, 17920] bf16 x8: one-shot
// 10.8 against two-shot 12.3 us at 1 token, two-shot 13.2 against 14.3 at 8 (2026-10-01T21-20-57Z).
// The one-shot: the one-shot norm's block, a block a (row, slice) up to one a compute unit. The
// two-shot: its calibrated grid (machine/hardware.cuh), each block looping over its (row, slice)s.
constexpr Kernel tune_all_reduce_rms_scale_add(const ScaleAddArgs& a, int world, const Hardware& hw,
                                               const Calibration& cal) {
  if (a.rows < world) return kernel_for(Template::all_reduce_pull_one_shot_rms_scale_add, a, world);
  return kernel_for(Template::all_reduce_pull_two_shot_rms_scale_add, a, world);
}

// =================================================================================================
// THE ONE ENTRY: the op's own rule, or the caller's forced template, then its arguments.
// =================================================================================================

constexpr Kernel rule(const AllReduceArgs& a, int world, const Hardware& hw,
                      const Calibration& cal) {
  return tune_all_reduce(a, world, hw, cal);
}
constexpr Kernel rule(const NormArgs& a, int world, const Hardware& hw, const Calibration& cal) {
  return a.add ? tune_all_reduce_add_rms_norm(a, world, hw, cal)
               : tune_all_reduce_rms_norm(a, world, hw, cal);
}
constexpr Kernel rule(const AttnResArgs& a, int world, const Hardware& hw,
                      const Calibration& cal) {
  return tune_all_reduce_add_attn_res_rms_norm(a, world, hw, cal);
}
constexpr Kernel rule(const GemmTailArgs& a, int world, const Hardware& hw,
                      const Calibration& cal) {
  return a.add ? tune_all_reduce_rms_norm_gemm_add(a, world, hw, cal)
               : tune_all_reduce_rms_norm_gemm(a, world, hw, cal);
}

constexpr Kernel rule(const ScaleAddArgs& a, int world, const Hardware& hw,
                      const Calibration& cal) {
  return tune_all_reduce_rms_scale_add(a, world, hw, cal);
}

template <typename Args>
constexpr Kernel select(const Args& a, int world, const Options& o) {
  Kernel k = rule(a, world, kTarget, kTargetCalibration);
  if (!o.fn) return k;
  // A FORCED KERNEL: the template at its own default config (a row template's list; the plain
  // all-reduce, which has none, at the rule's launch), or at the forced KernelConfig, whose zero
  // tile fields take the template's own tile for the call.
  if (!o.kernel_config)
    return has_tiles(*o.fn)
               ? kernel_for(*o.fn, a, world)
               : kernel_for(*o.fn, k.config.blocks_per_grid, k.config.threads_per_block, a, world);
  const KernelConfig& launch = *o.kernel_config;
  k = kernel_for(*o.fn, launch.blocks_per_grid, launch.threads_per_block, a, world);
  if (o.kernel_config && o.kernel_config->tile_m > 0) k.config.tile_m = o.kernel_config->tile_m;
  if (o.kernel_config && o.kernel_config->tile_n > 0) k.config.tile_n = o.kernel_config->tile_n;
  return k;
}

// WHETHER A KERNEL IS ITS STAGED BUILD: the plain all-reduce's, when plan found an eager input.
constexpr bool is_staged(const Kernel& k) {
  const auto* a = std::get_if<AllReduceTemplateArgs>(&k.args);
  return a && a->staged;
}

// EVERY SELECTED KERNEL FITS, for every op at the smallest call and a large one: a rule never
// declines (a fusion that is on runs its fused op), and a kernel past a capability is a compile
// error, not one that overruns its signal slots or register arrays.
constexpr bool fits(const Kernel& k) {
  if (k.config.blocks_per_grid < 1 || k.config.blocks_per_grid > p2p::kMaxBlocks) return false;
  if (has_tiles(k.fn) && k.config.tile_n == 0) return false;
  const int t = k.config.threads_per_block;
  return t >= kWaveSize && t <= kBuild.kernels.max_threads && t % kWaveSize == 0;
}
constexpr bool selections_fit() {
  const Options o{std::nullopt, std::nullopt, std::nullopt, nullptr};
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
  }
  return true;
}
static_assert(selections_fit(), "a rule declines, or exceeds a kernel capability");

// EACH OP'S CALL AS ITS OP.
constexpr Op op_of(const AllReduceArgs&) { return Op::all_reduce; }
constexpr Op op_of(const NormArgs& a) {
  return a.add ? Op::all_reduce_add_rms_norm : Op::all_reduce_rms_norm;
}
constexpr Op op_of(const AttnResArgs&) { return Op::all_reduce_add_attn_res_rms_norm; }
constexpr Op op_of(const ScaleAddArgs&) { return Op::all_reduce_rms_scale_add; }
constexpr Op op_of(const GemmTailArgs& a) {
  return a.add ? Op::all_reduce_rms_norm_gemm_add : Op::all_reduce_rms_norm_gemm;
}

}  // namespace hip_comms
