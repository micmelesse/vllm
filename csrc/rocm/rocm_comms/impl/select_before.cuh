// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// TEMPORARY: select as it was before it read each op's kernels (fork 16be2d0cb1, verified on n11:
// the size thresholds' choices exactly), every name `before_`, so a static_assert in select.cuh
// holds the new one to it. Deleted with the thresholds once it builds.

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

constexpr OpType before_op_of(Template k) { return op_of(k); }


// From `tokens` rows up (to the next row's), at `world` ranks and rows `hidden` elements wide, op
// `op` runs `fn` at `config`.
struct before_Tuned {
  OpType op;
  int world;
  int64_t hidden;
  int64_t tokens;
  Template fn;
  KernelConfig config;
};

// gfx950 on n11, bf16, 8 ranks. Transcribed from the size thresholds the sweeps set (each row's
// crossover cites its run) at the widths we run, until the tuner writes it.
constexpr before_Tuned before_kGfx950Tuned[] = {
    // rms_norm: one-shot through 128 KiB (at 64 KiB it lost at 16 tokens, 11.43 against 10.56 us;
    // 2026-09-30T21-06-57Z); the push through 1.75 MiB (256 tokens of 3584: 18.04 against the
    // pull's 18.41), the pull from 2.6 MiB (2026-10-01T03-26-44Z).
    {OpType::all_reduce_rms_norm, 8, 3584, 1, Template::all_reduce_pull_one_shot_rms_norm,
     {1, 4096, 0, 0, 512, 16}},
    {OpType::all_reduce_rms_norm, 8, 3584, 19, Template::all_reduce_push_two_shot_rms_norm,
     {1, 4096, 0, 0, 512, 256}},
    {OpType::all_reduce_rms_norm, 8, 3584, 257, Template::all_reduce_pull_two_shot_rms_norm,
     {1, 4096, 0, 0, 512, 48}},
    {OpType::all_reduce_rms_norm, 8, 7168, 1, Template::all_reduce_pull_one_shot_rms_norm,
     {1, 8192, 0, 0, 512, 16}},
    {OpType::all_reduce_rms_norm, 8, 7168, 10, Template::all_reduce_push_two_shot_rms_norm,
     {1, 8192, 0, 0, 512, 256}},
    {OpType::all_reduce_rms_norm, 8, 7168, 129, Template::all_reduce_pull_two_shot_rms_norm,
     {1, 8192, 0, 0, 512, 48}},
    // add_rms_norm: the one-shot not swept (rms_norm's); the push through 1.31 MiB (192 tokens of
    // 3584: 15.48 against the pull's 16.28), the pull at 1.75 MiB (18.36 against the push's 18.46;
    // 2026-10-01T03-26-44Z).
    {OpType::all_reduce_add_rms_norm, 8, 3584, 1, Template::all_reduce_pull_one_shot_add_rms_norm,
     {1, 4096, 0, 0, 512, 16}},
    {OpType::all_reduce_add_rms_norm, 8, 3584, 19, Template::all_reduce_push_two_shot_add_rms_norm,
     {1, 4096, 0, 0, 512, 256}},
    {OpType::all_reduce_add_rms_norm, 8, 3584, 193, Template::all_reduce_pull_two_shot_add_rms_norm,
     {1, 4096, 0, 0, 512, 48}},
    {OpType::all_reduce_add_rms_norm, 8, 7168, 1, Template::all_reduce_pull_one_shot_add_rms_norm,
     {1, 8192, 0, 0, 512, 16}},
    {OpType::all_reduce_add_rms_norm, 8, 7168, 10, Template::all_reduce_push_two_shot_add_rms_norm,
     {1, 8192, 0, 0, 512, 256}},
    {OpType::all_reduce_add_rms_norm, 8, 7168, 97, Template::all_reduce_pull_two_shot_add_rms_norm,
     {1, 8192, 0, 0, 512, 48}},
    // AttnRes: the push from 1 token (it beat the one-shot, 12.89 against 13.68 us; 14.03 against
    // 15.18 at 8; 2026-10-01T03-57-23Z), through 3.5 MiB against the pull at its grid (256 tokens
    // of 7168: 34.8 against 35.8-38.7 us; 2026-10-01T22-56-58Z, 2026-10-01T23-00-47Z).
    {OpType::all_reduce_add_attn_res_rms_norm, 8, 3584, 1,
     Template::all_reduce_push_two_shot_add_attn_res_rms_norm, {1, 4096, 1, 0, 512, 256}},
    {OpType::all_reduce_add_attn_res_rms_norm, 8, 3584, 513,
     Template::all_reduce_pull_two_shot_add_attn_res_rms_norm, {1, 4096, 1, 0, 512, 192}},
    {OpType::all_reduce_add_attn_res_rms_norm, 8, 7168, 1,
     Template::all_reduce_push_two_shot_add_attn_res_rms_norm, {1, 8192, 1, 0, 512, 256}},
    {OpType::all_reduce_add_attn_res_rms_norm, 8, 7168, 257,
     Template::all_reduce_pull_two_shot_add_attn_res_rms_norm, {1, 8192, 1, 0, 512, 192}},
    // The GEMM tails: the one-shot through one GEMM pass of rows (16), where the one-shot kernel
    // once had to stop; not swept.
    {OpType::all_reduce_rms_norm_gemm, 8, 3584, 1, Template::all_reduce_pull_one_shot_rms_norm_gemm,
     {16, 4096, kGemmTileK, 4, 512, 56}},
    {OpType::all_reduce_rms_norm_gemm, 8, 3584, 17, Template::all_reduce_pull_two_shot_rms_norm_gemm,
     {16, 4096, kGemmTileK, 4, 512, 56}},
    {OpType::all_reduce_rms_norm_gemm, 8, 7168, 1, Template::all_reduce_pull_one_shot_rms_norm_gemm,
     {16, 8192, kGemmTileK, 4, 512, 56}},
    {OpType::all_reduce_rms_norm_gemm, 8, 7168, 17, Template::all_reduce_pull_two_shot_rms_norm_gemm,
     {16, 8192, kGemmTileK, 4, 512, 56}},
    {OpType::all_reduce_rms_norm_gemm_add, 8, 3584, 1,
     Template::all_reduce_pull_one_shot_rms_norm_gemm_add, {16, 4096, kGemmTileK, 4, 512, 56}},
    {OpType::all_reduce_rms_norm_gemm_add, 8, 3584, 17,
     Template::all_reduce_pull_two_shot_rms_norm_gemm_add, {16, 4096, kGemmTileK, 4, 512, 56}},
    {OpType::all_reduce_rms_norm_gemm_add, 8, 7168, 1,
     Template::all_reduce_pull_one_shot_rms_norm_gemm_add, {16, 8192, kGemmTileK, 4, 512, 56}},
    {OpType::all_reduce_rms_norm_gemm_add, 8, 7168, 17,
     Template::all_reduce_pull_two_shot_rms_norm_gemm_add, {16, 8192, kGemmTileK, 4, 512, 56}},
    // The one-all-reduce tail, [T, 17920] (a latent of 3584): the one-shot while there are fewer
    // rows than ranks, the row two-shot from a row a rank (10.8 against 12.3 us at 1 token, 13.2
    // against 14.3 at 8 the other way; 2026-10-01T21-20-57Z).
    {OpType::all_reduce_rms_scale_add, 8, 17920, 1, Template::all_reduce_pull_one_shot_rms_scale_add,
     {1, 4096, 0, 0, 512, 256}},
    {OpType::all_reduce_rms_scale_add, 8, 17920, 8, Template::all_reduce_pull_two_shot_rms_scale_add,
     {1, 4096, 0, 0, 512, 32}},
};

// THE TARGET'S TABLE.
constexpr std::span<const before_Tuned> before_kTargetTuned = before_kGfx950Tuned;


// SELECT'S DEFAULT for `k` on a row of `cols`: its first config whose tile covers it, or none.
constexpr std::optional<KernelConfig> before_config_for(Template k, int64_t cols) {
  for (const KernelConfig& c : configs_of(k))
    if (c.tile_n >= cols) return c;
  return std::nullopt;
}

// A FORCED LAUNCH'S TILE: `k`'s default tile (its first config's tile_m, tile_k, slice_k) at
// `threads_per_block`, with the smallest built TILE_N covering `cols`; tile_n 0 when none does.
constexpr KernelConfig before_tile_for(Template k, int64_t cols, int threads_per_block) {
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




constexpr int before_elem_bytes(DType d) { return d == DType::f32 ? 4 : 2; }

// A CALL'S SHAPE, as the rules read it: rows (a plain all-reduce is one row), a row's elements, and
// the bytes the hardware moves.
constexpr int64_t before_rows_of(const AllReduceArgs&) { return 1; }
constexpr int64_t before_rows_of(const NormArgs& a) { return a.rows; }
constexpr int64_t before_rows_of(const AttnResArgs& a) { return a.rows; }
constexpr int64_t before_rows_of(const GemmTailArgs& a) { return a.rows; }
constexpr int64_t before_rows_of(const ScaleAddArgs& a) { return a.rows; }
constexpr int64_t before_hidden_of(const AllReduceArgs& a) { return a.bytes / before_elem_bytes(a.dtype); }
constexpr int64_t before_hidden_of(const NormArgs& a) { return a.hidden; }
constexpr int64_t before_hidden_of(const AttnResArgs& a) { return a.hidden; }
constexpr int64_t before_hidden_of(const GemmTailArgs& a) { return a.hidden; }
// The row reduced, [shared | projected | latent].
constexpr int64_t before_hidden_of(const ScaleAddArgs& a) { return 2 * a.hidden + a.latent; }
// EACH OP'S CALL AS ITS OP.
constexpr OpType before_op_of(const AllReduceArgs&) { return OpType::all_reduce; }
constexpr OpType before_op_of(const NormArgs& a) {
  return a.add ? OpType::all_reduce_add_rms_norm : OpType::all_reduce_rms_norm;
}
constexpr OpType before_op_of(const AttnResArgs&) { return OpType::all_reduce_add_attn_res_rms_norm; }
constexpr OpType before_op_of(const ScaleAddArgs&) { return OpType::all_reduce_rms_scale_add; }
constexpr OpType before_op_of(const GemmTailArgs& a) {
  return a.add ? OpType::all_reduce_rms_norm_gemm_add : OpType::all_reduce_rms_norm_gemm;
}

template <typename Args>
constexpr int64_t before_bytes_of(const Args& a) {
  return before_rows_of(a) * before_hidden_of(a) * before_elem_bytes(a.dtype);
}
template <typename Args>
constexpr int64_t before_packs_of(const Args& a) {
  return before_hidden_of(a) * before_elem_bytes(a.dtype) / kBuild.memory.pack_bytes;
}

// A ROW OP GIVES EACH BLOCK WHOLE ROWS, so it needs no more blocks than it has rows: a one-shot
// and a column two-shot (the pushes, and AttnRes's pull) all of them, a row two-shot its rank's. An idle
// block still pays every barrier (each pairs with its twin on every peer): the pull norm at 32
// tokens ran 36 blocks for 4 rows a rank. The GEMM tail's GEMM strides over column tiles, and the
// plain all-reduce over packs, so both keep theirs.
constexpr bool before_slices_columns(Template k) {
  return k == Template::all_reduce_pull_two_shot_add_attn_res_rms_norm ||
         k == Template::all_reduce_push_two_shot_rms_norm ||
         k == Template::all_reduce_push_two_shot_add_rms_norm ||
         k == Template::all_reduce_push_two_shot_add_attn_res_rms_norm;
}
constexpr int before_grid_of(Template k, int blocks, int64_t rows, int world) {
  const OpType op = before_op_of(k);
  if (op == OpType::all_reduce || gemms(op)) return blocks;
  const bool row_slice = is_two_shot(k) && !before_slices_columns(k);
  const int64_t mine   = row_slice ? (rows + world - 1) / world : rows;
  return mine < blocks ? static_cast<int>(mine) : blocks;
}

// THE COLUMNS A ROW TEMPLATE'S TILE COVERS: the row, or the one-all-reduce tail's latent (its
// hidden is cut into slices of the tile's width).
template <typename Args>
constexpr int64_t before_tile_cols(const Args& a) {
  return before_hidden_of(a);
}
constexpr int64_t before_tile_cols(const ScaleAddArgs& a) { return a.latent; }
// A LAUNCH AT `threads`: the template's default tile (impl/templates.cuh's before_tile_for) covering the
// call's columns, at `blocks` cut to the rows where it gives each block a row.
template <typename Args>
constexpr KernelConfig before_config_at(Template t, const Args& a, int blocks, int threads, int world) {
  KernelConfig c = before_tile_for(t, before_tile_cols(a), threads);
  c.blocks_per_grid = before_grid_of(t, blocks, before_rows_of(a), world);
  return c;
}
// THE KERNEL: template `t` at `blocks` x `threads` with its arguments and tile from the call.
constexpr Kernel before_kernel_for(Template t, int blocks, int threads, const AllReduceArgs& a,
                            int world) {
  return {t, AllReduceTemplateArgs{world, a.dtype, false},
          KernelConfig{0, 0, 0, 0, threads, before_grid_of(t, blocks, before_rows_of(a), world)}};
}
constexpr Kernel before_kernel_for(Template t, int blocks, int threads, const NormArgs& a, int world) {
  return {t, NormTemplateArgs{world, a.dtype, a.weight_dtype},
          before_config_at(t, a, blocks, threads, world)};
}
constexpr Kernel before_kernel_for(Template t, int blocks, int threads, const AttnResArgs& a,
                            int world) {
  return {t, AttnResTemplateArgs{world, a.dtype, a.has_prefix},
          before_config_at(t, a, blocks, threads, world)};
}
constexpr Kernel before_kernel_for(Template t, int blocks, int threads, const GemmTailArgs& a,
                            int world) {
  return {t, GemmTemplateArgs{world, a.dtype}, before_config_at(t, a, blocks, threads, world)};
}
// THE ONE-ALL-REDUCE TAIL: a block holds the whole latent (its build) and a slice of the hidden
// as wide, so a row takes `splits` blocks.
constexpr Kernel before_kernel_for(Template t, int blocks, int threads, const ScaleAddArgs& a,
                            int world) {
  const int e = before_elem_bytes(a.dtype);
  KernelConfig c = before_tile_for(t, before_tile_cols(a), threads);
  const int64_t span = c.tile_n > 0 ? c.tile_n * e / kBuild.memory.pack_bytes : threads;
  const int64_t hp = a.hidden * e / kBuild.memory.pack_bytes;
  const int splits = static_cast<int>((hp + span - 1) / span);
  // A block a (row, slice): every row's in the one-shot, this rank's in the two-shot.
  const int64_t mine = is_two_shot(t) ? (a.rows + world - 1) / world : a.rows;
  const int64_t work = mine * splits;
  c.blocks_per_grid = static_cast<int>(work < blocks ? (work > 0 ? work : 1) : blocks);
  return {t, ScaleAddTemplateArgs{world, a.dtype}, c};
}

// THE KERNEL AT ITS TEMPLATE'S DEFAULT: the first of its configs whose tile covers the call, at
// that config's threads and grid (cut to the rows); a tile_n of 0 when none covers it (check
// refuses it).
template <typename Args>
constexpr Kernel before_kernel_for(Template t, const Args& a, int world) {
  const std::optional<KernelConfig> c = before_config_for(t, before_tile_cols(a));
  const KernelConfig d = c ? *c : configs_of(t)[0];
  Kernel k = before_kernel_for(t, d.blocks_per_grid, d.threads_per_block, a, world);
  k.config.tile_m = d.tile_m;
  k.config.tile_n = c ? d.tile_n : 0;
  k.config.tile_k = d.tile_k;
  k.config.slice_k = d.slice_k;
  return k;
}

// THE KERNEL OF TEMPLATE `t` AT `cfg` on call `a`: `cfg`'s threads and grid (cut to the rows where a
// block takes a row), and its tile fields where set; a zero is the template's own for the call.
template <typename Args>
constexpr Kernel before_kernel_at(Template t, const KernelConfig& cfg, const Args& a, int world) {
  Kernel k = before_kernel_for(t, cfg.blocks_per_grid, cfg.threads_per_block, a, world);
  if (cfg.tile_m > 0) k.config.tile_m = cfg.tile_m;
  if (cfg.tile_n > 0) k.config.tile_n = cfg.tile_n;
  if (cfg.tile_k > 0) k.config.tile_k = cfg.tile_k;
  if (cfg.slice_k > 0) k.config.slice_k = cfg.slice_k;
  return k;
}

// =================================================================================================
// THE ROW OPS: the before_tuned table's row for the call (impl/before_tuned.cuh).
// =================================================================================================

constexpr int64_t before_distance(int64_t a, int64_t b) { return a < b ? b - a : a - b; }

// THE TUNED ROW FOR CALL `a`: among its op's rows, those at the call's world, else at the nearest
// before_tuned world; among those, the call's width, else the nearest before_tuned width, reading the call's rows
// as that width's rows of the same bytes (a crossover is about bytes); then the last row whose
// tokens the call reaches, the first when it reaches none. None when the op has no rows.
template <typename Args>
constexpr const before_Tuned* before_tuned_for(const Args& a, int world, std::span<const before_Tuned> table) {
  const OpType op = before_op_of(a);
  int w = 0;
  for (const before_Tuned& r : table)
    if (r.op == op && (w == 0 || before_distance(r.world, world) < before_distance(w, world))) w = r.world;
  int64_t h = 0;
  for (const before_Tuned& r : table)
    if (r.op == op && r.world == w &&
        (h == 0 || before_distance(r.hidden, before_hidden_of(a)) < before_distance(h, before_hidden_of(a))))
      h = r.hidden;
  if (h == 0) return nullptr;
  const int64_t rows = before_rows_of(a) * before_hidden_of(a) / h;
  const before_Tuned* first = nullptr;
  const before_Tuned* reached = nullptr;
  for (const before_Tuned& r : table) {
    if (r.op != op || r.world != w || r.hidden != h) continue;
    if (!first || r.tokens < first->tokens) first = &r;
    if (r.tokens <= rows && (!reached || r.tokens > reached->tokens)) reached = &r;
  }
  return reached ? reached : first;
}

// THE ROW OP'S KERNEL: its before_tuned row's template at the row's config; at an untuned width whose row
// the config's tile does not cover, the smallest built tile that does.
template <typename Args>
constexpr Kernel before_tuned(const Args& a, int world) {
  const before_Tuned* r = before_tuned_for(a, world, before_kTargetTuned);
  KernelConfig c = r->config;
  if (c.tile_n < before_tile_cols(a)) c.tile_n = 0;
  return before_kernel_at(r->fn, c, a, world);
}

// EVERY ROW OP HAS ROWS, and every row's config is one its template builds.
constexpr bool before_tuned_covers_ops() {
  for (const OpType op : {OpType::all_reduce_rms_norm, OpType::all_reduce_add_rms_norm,
                      OpType::all_reduce_add_attn_res_rms_norm, OpType::all_reduce_rms_norm_gemm,
                      OpType::all_reduce_rms_norm_gemm_add, OpType::all_reduce_rms_scale_add}) {
    bool has = false;
    for (const before_Tuned& r : before_kTargetTuned) has = has || r.op == op;
    if (!has) return false;
  }
  for (const before_Tuned& r : before_kTargetTuned)
    if (before_op_of(r.fn) != r.op || !built(r.fn, r.config)) return false;
  return true;
}
static_assert(before_tuned_covers_ops(), "a row op has no before_tuned rows, or a row's config is not built");

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
constexpr int before_link_filling_blocks(const Hardware& hw, const Calibration& cal, int threads) {
  const double in_flight = hw.xgmi_links * hw.xgmi_gbytes_per_s_a_way * cal.ping_pong_ns;
  const double per_pass = static_cast<double>(threads) * kBuild.memory.pack_bytes;
  const int blocks       = static_cast<int>(in_flight / per_pass + 0.999);
  return blocks < hw.compute_units ? blocks : hw.compute_units;
}

constexpr Kernel before_tune_all_reduce(const AllReduceArgs& a, int world, const Hardware& hw,
                                 const Calibration& cal) {
  const bool one_shot = before_bytes_of(a) <= cal.all_reduce_one_shot_max_bytes;
  const Template k = one_shot ? Template::all_reduce_pull_one_shot : Template::all_reduce_pull_two_shot;
  const int64_t packs = (before_bytes_of(a) + kBuild.memory.pack_bytes - 1) / kBuild.memory.pack_bytes;
  const int64_t work  = one_shot ? packs : (packs + world - 1) / world;
  const int64_t need  = (work + hw.wave_size - 1) / hw.wave_size;
  const int threads = one_shot ? hw.wave_size : hw.wave_size * world;
  // AT LEAST THE LINK-FILLING GRID A PASS, AND EVERY PASS FULL: as many passes as keep each one
  // filling the links, the work spread evenly over them. A cap alone left a near-empty last pass,
  // a whole round trip for a sliver (3.7 MB at 88 blocks: 5.09 passes, 23.78 us, against 90 blocks
  // in 5 full ones; the sweep's 80 was 22.99).
  const int fill       = before_link_filling_blocks(hw, cal, threads);
  const int64_t passes = need > fill ? need / fill : 1;
  const int64_t even   = (need + passes - 1) / passes;
  // NOT std::min: hipify turns it into HIP's device `min`, which is not constexpr.
  const int blocks = static_cast<int>(even < hw.compute_units ? even : hw.compute_units);
  return before_kernel_for(k, blocks, threads, a, world);
}

// =================================================================================================
// THE FUSED OPS: one-shot up to Calibration's fused_one_shot_max_bytes, two-shot past it, at its
// fused grids and block.
// =================================================================================================

// THE NORMS ALWAYS RUN FUSED: a fusion flag means the fused op runs, and tuning picks among fused
// kernels, never unfused. The one-shot up to fused_one_shot_max_bytes (7.46 against 10.00 us at 1
// token, 9.95 against 10.34 at 16, 2026-09-30T21-30-15Z); the push two-shot (aiter's column split)
// up to the op's push_max_bytes, where it wins; the pull two-shot (rows) past it, at prefill.
constexpr Kernel before_fused_norm(Template one_shot, Template push, Template pull, const NormArgs& a,
                            int world, const NormCalibration& c) {
  if (before_bytes_of(a) <= c.one_shot_max_bytes) return before_kernel_for(one_shot, a, world);
  if (before_bytes_of(a) <= c.push_max_bytes) return before_kernel_for(push, a, world);
  return before_kernel_for(pull, a, world);
}

constexpr Kernel before_tune_all_reduce_rms_norm(const NormArgs& a, int world, const Hardware&,
                                          const Calibration& cal) {
  return before_fused_norm(Template::all_reduce_pull_one_shot_rms_norm,
                    Template::all_reduce_push_two_shot_rms_norm,
                    Template::all_reduce_pull_two_shot_rms_norm, a, world, cal.rms_norm);
}

constexpr Kernel before_tune_all_reduce_add_rms_norm(const NormArgs& a, int world, const Hardware&,
                                              const Calibration& cal) {
  return before_fused_norm(Template::all_reduce_pull_one_shot_add_rms_norm,
                    Template::all_reduce_push_two_shot_add_rms_norm,
                    Template::all_reduce_pull_two_shot_add_rms_norm, a, world, cal.add_rms_norm);
}

// AttnRes as the norms: a block a row, never more blocks than rows (before_grid_of).
constexpr Kernel before_tune_all_reduce_add_attn_res_rms_norm(const AttnResArgs& a, int world,
                                                       const Hardware&, const Calibration& cal) {
  const AttnResCalibration& c = cal.attn_res;
  if (before_bytes_of(a) <= c.one_shot_max_bytes)
    return before_kernel_for(Template::all_reduce_pull_one_shot_add_attn_res_rms_norm, a, world);
  if (before_bytes_of(a) <= c.push_max_bytes)
    return before_kernel_for(Template::all_reduce_push_two_shot_add_attn_res_rms_norm, a, world);
  return before_kernel_for(Template::all_reduce_pull_two_shot_add_attn_res_rms_norm, a, world);
}

// ALWAYS FUSED, as every op: one-shot up to one GEMM pass of rows, two-shot past it, 56 blocks of
// 512 threads (the GEMM strides over column tiles). It is slower than the unfused ops (about 68
// against 20 us at 1 token, 2026-09-30T20-23-38Z): a loss to fix, shown as one.
constexpr Kernel before_gemm(Template one_shot, Template two_shot, const GemmTailArgs& a, int world,
                      const GemmCalibration& c) {
  return before_rows_of(a) <= c.one_shot_max_rows ? before_kernel_for(one_shot, a, world)
                                           : before_kernel_for(two_shot, a, world);
}

constexpr Kernel before_tune_all_reduce_rms_norm_gemm(const GemmTailArgs& a, int world, const Hardware&,
                                               const Calibration& cal) {
  return before_gemm(Template::all_reduce_pull_one_shot_rms_norm_gemm,
              Template::all_reduce_pull_two_shot_rms_norm_gemm, a, world, cal.rms_norm_gemm);
}

constexpr Kernel before_tune_all_reduce_rms_norm_gemm_add(const GemmTailArgs& a, int world,
                                                   const Hardware&, const Calibration& cal) {
  return before_gemm(Template::all_reduce_pull_one_shot_rms_norm_gemm_add,
              Template::all_reduce_pull_two_shot_rms_norm_gemm_add, a, world,
              cal.rms_norm_gemm_add);
}

// THE ONE-ALL-REDUCE TAIL: the one-shot while there are fewer rows than ranks (the two-shot would
// leave ranks idle), the row two-shot from a row a rank. Measured at [T, 17920] bf16 x8: one-shot
// 10.8 against two-shot 12.3 us at 1 token, two-shot 13.2 against 14.3 at 8 (2026-10-01T21-20-57Z).
// The one-shot: the one-shot norm's block, a block a (row, slice) up to one a compute unit. The
// two-shot: its calibrated grid (machine/hardware.cuh), each block looping over its (row, slice)s.
constexpr Kernel before_tune_all_reduce_rms_scale_add(const ScaleAddArgs& a, int world, const Hardware& hw,
                                               const Calibration& cal) {
  if (a.rows < world) return before_kernel_for(Template::all_reduce_pull_one_shot_rms_scale_add, a, world);
  return before_kernel_for(Template::all_reduce_pull_two_shot_rms_scale_add, a, world);
}

// =================================================================================================
// THE ONE ENTRY: the op's own before_rule, or the caller's forced template, then its arguments.
// =================================================================================================

constexpr Kernel before_rule(const AllReduceArgs& a, int world, const Hardware& hw,
                      const Calibration& cal) {
  return before_tune_all_reduce(a, world, hw, cal);
}
template <typename Args>
constexpr Kernel before_rule(const Args& a, int world, const Hardware&, const Calibration&) {
  return before_tuned(a, world);
}

// TRANSCRIBED, NOT RETUNED: the table gives each row op the kernel the size thresholds gave, at
// every before_tuned width, on both sides of every crossover.
constexpr Kernel before_legacy_rule(const NormArgs& a, int world, const Hardware& hw,
                             const Calibration& cal) {
  return a.add ? before_tune_all_reduce_add_rms_norm(a, world, hw, cal)
               : before_tune_all_reduce_rms_norm(a, world, hw, cal);
}
constexpr Kernel before_legacy_rule(const AttnResArgs& a, int world, const Hardware& hw,
                             const Calibration& cal) {
  return before_tune_all_reduce_add_attn_res_rms_norm(a, world, hw, cal);
}
constexpr Kernel before_legacy_rule(const GemmTailArgs& a, int world, const Hardware& hw,
                             const Calibration& cal) {
  return a.add ? before_tune_all_reduce_rms_norm_gemm_add(a, world, hw, cal)
               : before_tune_all_reduce_rms_norm_gemm(a, world, hw, cal);
}
constexpr Kernel before_legacy_rule(const ScaleAddArgs& a, int world, const Hardware& hw,
                             const Calibration& cal) {
  return before_tune_all_reduce_rms_scale_add(a, world, hw, cal);
}
constexpr bool before_same_kernel(const Kernel& a, const Kernel& b) {
  return a.fn == b.fn && same_build(a.config, b.config) &&
         a.config.blocks_per_grid == b.config.blocks_per_grid;
}
template <typename Args>
constexpr bool before_transcribed(const Args& a) {
  return before_same_kernel(before_tuned(a, 8), before_legacy_rule(a, 8, kTarget, kTargetCalibration));
}
constexpr bool before_tuned_is_transcribed() {
  constexpr DType bf = DType::bf16;
  for (const int64_t t : {1, 2, 7, 8, 9, 10, 16, 17, 18, 19, 96, 97, 128, 129, 192, 193, 256,
                          257, 512, 513, 4096, 8192}) {
    for (const int64_t h : {3584, 7168}) {
      for (const bool add : {false, true}) {
        if (!before_transcribed(NormArgs{add, nullptr, nullptr, nullptr, bf, bf, t, h, 0.f, nullptr,
                                  nullptr}))
          return false;
        if (!before_transcribed(GemmTailArgs{.add = add, .dtype = bf, .rows = t, .hidden = h}))
          return false;
      }
      if (!before_transcribed(AttnResArgs{nullptr, nullptr, nullptr, nullptr, 0, 0, nullptr, nullptr,
                                   nullptr, bf, t, h, 0, -1, 0.f, 0.f, true}))
        return false;
    }
    if (!before_transcribed(ScaleAddArgs{nullptr, nullptr, bf, t, 7168, 3584, 0.f})) return false;
  }
  return true;
}
static_assert(before_tuned_is_transcribed(), "the before_tuned table is not the thresholds it transcribes");

template <typename Args>
constexpr Kernel before_select(const Args& a, int world, const Options& o) {
  Kernel k = before_rule(a, world, kTarget, kTargetCalibration);
  if (!o.fn) return k;
  // A FORCED KERNEL: the template at its own default config (a row template's list; the plain
  // all-reduce, which has none, at the before_rule's launch), or at the forced KernelConfig, whose zero
  // tile fields take the template's own tile for the call.
  if (!o.kernel_config)
    return has_tiles(*o.fn)
               ? before_kernel_for(*o.fn, a, world)
               : before_kernel_for(*o.fn, k.config.blocks_per_grid, k.config.threads_per_block, a, world);
  return before_kernel_at(*o.fn, *o.kernel_config, a, world);
}


// THE NEW SELECT IS THE OLD ONE: the same template and config for every op, tuned (at the tuned
// widths and two untuned ones, on both sides of every crossover) and forced (each of the op's
// templates at its own config, and at 512 x 36 with its own tile).
constexpr bool same_choice(const Kernel& a, const Kernel& b) {
  return a.fn == b.fn && a.config.tile_m == b.config.tile_m && a.config.tile_n == b.config.tile_n &&
         a.config.tile_k == b.config.tile_k && a.config.slice_k == b.config.slice_k &&
         a.config.threads_per_block == b.config.threads_per_block &&
         a.config.blocks_per_grid == b.config.blocks_per_grid;
}
template <typename Args>
constexpr bool chooses_as_before(const Args& a, int world) {
  const Options none{std::nullopt, std::nullopt, std::nullopt, nullptr};
  if (!same_choice(select(a, world, none), before_select(a, world, none))) return false;
  if constexpr (!std::is_same_v<Args, AllReduceArgs>) {
    for (const TemplateInfo& t : kTemplates) {
      if (t.op != op_of(a)) continue;
      const Options own{std::nullopt, t.fn, std::nullopt, nullptr};
      const Options at{std::nullopt, t.fn, KernelConfig{0, 0, 0, 0, 512, 36}, nullptr};
      if (!same_choice(select(a, world, own), before_select(a, world, own))) return false;
      if (!same_choice(select(a, world, at), before_select(a, world, at))) return false;
    }
  }
  return true;
}
// One op family at one width a static_assert, each within the compiler's step budget.
enum class Checked { all_reduce, norm, add_norm, gemm, gemm_add, attn_res, scale_add };
constexpr bool select_is_as_before(Checked op, int64_t h) {
  constexpr DType bf = DType::bf16;
  for (const int64_t t : {1, 2, 7, 8, 9, 10, 16, 17, 19, 96, 97, 129, 193, 256, 257, 512, 513,
                          4096}) {
    bool same = true;
    switch (op) {
      case Checked::all_reduce:
        same = chooses_as_before(AllReduceArgs{nullptr, nullptr, t * h * 2, bf}, 8);
        break;
      case Checked::norm:
      case Checked::add_norm:
        same = chooses_as_before(NormArgs{op == Checked::add_norm, nullptr, nullptr, nullptr, bf,
                                          bf, t, h, 0.f, nullptr, nullptr}, 8);
        break;
      case Checked::gemm:
      case Checked::gemm_add:
        same = chooses_as_before(
            GemmTailArgs{.add = op == Checked::gemm_add, .dtype = bf, .rows = t, .hidden = h}, 8);
        break;
      case Checked::attn_res:
        same = chooses_as_before(AttnResArgs{nullptr, nullptr, nullptr, nullptr, 0, 0, nullptr,
                                             nullptr, nullptr, bf, t, h, 0, -1, 0.f, 0.f, true},
                                 8);
        break;
      case Checked::scale_add:
        same = chooses_as_before(ScaleAddArgs{nullptr, nullptr, bf, t, h, h / 2, 0.f}, 8);
        break;
    }
    if (!same) return false;
  }
  return true;
}
#define HIP_COMMS_AS_BEFORE(op)                                                       \
  static_assert(select_is_as_before(Checked::op, 2048), "select differs: " #op " 2048"); \
  static_assert(select_is_as_before(Checked::op, 3584), "select differs: " #op " 3584"); \
  static_assert(select_is_as_before(Checked::op, 7168), "select differs: " #op " 7168"); \
  static_assert(select_is_as_before(Checked::op, 8192), "select differs: " #op " 8192");
HIP_COMMS_AS_BEFORE(all_reduce)
HIP_COMMS_AS_BEFORE(norm)
HIP_COMMS_AS_BEFORE(add_norm)
HIP_COMMS_AS_BEFORE(gemm)
HIP_COMMS_AS_BEFORE(gemm_add)
HIP_COMMS_AS_BEFORE(attn_res)
HIP_COMMS_AS_BEFORE(scale_add)
#undef HIP_COMMS_AS_BEFORE

}  // namespace hip_comms
