// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SELECT, THE ONLY CHOICE: `tune(op, input, hw, cal)` splits on the op and calls its own
// `tune_<op>`, which picks the kernel AND its grid and block from the input, the hardware's
// documented facts and what was measured on it (machine/hardware.cuh), one rule in one place, with
// the sweep it came from written beside it. `select` is tune, or the caller's forced spec.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <cstdint>

namespace hip_comms {

// WHAT THE HARDWARE MOVES: to it, a plain all-reduce is only its byte count.
constexpr int64_t bytes(Input in) { return in.rows * in.hidden * in.elem_bytes; }

// A ROW OP GIVES EACH BLOCK WHOLE ROWS, so it needs no more blocks than it has rows: a one-shot
// and the push two-shot (it slices columns) all of them, a pull two-shot its rank's slice. An idle
// block still pays every barrier (each pairs with its twin on every peer): the pull norm at 32
// tokens ran 36 blocks for 4 rows a rank. The GEMM tail's GEMM strides over column tiles, and the
// plain all-reduce over packs, so both keep theirs.
constexpr bool pushes(Kernel k) {
  return k == Kernel::all_reduce_push_two_shot_rms_norm ||
         k == Kernel::all_reduce_push_two_shot_add_rms_norm;
}
constexpr int grid_of(Kernel k, int blocks, int64_t rows, int world) {
  const Op op = op_of(k);
  if (op == Op::all_reduce || op == Op::all_reduce_rms_norm_gemm_add) return blocks;
  const bool row_slice = is_two_shot(k) && !pushes(k);
  const int64_t mine   = row_slice ? (rows + world - 1) / world : rows;
  return mine < blocks ? static_cast<int>(mine) : blocks;
}

// `k` at `blocks` x `threads`, its grid cut to the rows where it gives each block a row.
constexpr KernelSpec spec_of(Kernel k, Input in, int blocks, int threads) {
  return {k, grid_of(k, blocks, in.rows, in.world), threads};
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
  const double per_pass  = static_cast<double>(threads) * kPackBytes;
  const int blocks       = static_cast<int>(in_flight / per_pass + 0.999);
  return blocks < hw.compute_units ? blocks : hw.compute_units;
}

constexpr KernelSpec tune_all_reduce(Input in, const Hardware& hw, const Calibration& cal) {
  const bool one_shot = bytes(in) <= cal.one_shot_max_bytes;
  const Kernel k = one_shot ? Kernel::all_reduce_pull_one_shot : Kernel::all_reduce_pull_two_shot;
  const int64_t packs = (bytes(in) + kPackBytes - 1) / kPackBytes;
  const int64_t work  = one_shot ? packs : (packs + in.world - 1) / in.world;
  const int64_t need  = (work + hw.wave_size - 1) / hw.wave_size;
  const int threads = one_shot ? hw.wave_size : hw.wave_size * in.world;
  // AT LEAST THE LINK-FILLING GRID A PASS, AND EVERY PASS FULL: as many passes as keep each one
  // filling the links, the work spread evenly over them. A cap alone left a near-empty last pass,
  // a whole round trip for a sliver (3.7 MB at 88 blocks: 5.09 passes, 23.78 us, against 90 blocks
  // in 5 full ones; the sweep's 80 was 22.99).
  const int fill       = link_filling_blocks(hw, cal, threads);
  const int64_t passes = need > fill ? need / fill : 1;
  const int64_t even   = (need + passes - 1) / passes;
  // NOT std::min: hipify turns it into HIP's device `min`, which is not constexpr.
  const int blocks = static_cast<int>(even < hw.compute_units ? even : hw.compute_units);
  return spec_of(k, in, blocks, threads);
}

// =================================================================================================
// THE FUSED OPS: one-shot up to Calibration's fused_one_shot_max_bytes, two-shot past it, at its
// fused grids and block.
// =================================================================================================

// THE NORMS ALWAYS RUN FUSED: a fusion flag means the fused op runs, and tuning picks among fused
// kernels, never unfused. The one-shot up to fused_one_shot_max_bytes (7.46 against 10.00 us at 1
// token, 9.95 against 10.34 at 16, 2026-09-30T21-30-15Z); the push two-shot (aiter's column split)
// up to norm_push_max_bytes, where it wins; the pull two-shot (rows) past it, at prefill.
constexpr KernelSpec fused_norm(Kernel one_shot, Kernel push, Kernel pull, Input in,
                                const Calibration& cal) {
  if (bytes(in) <= cal.fused_one_shot_max_bytes)
    return spec_of(one_shot, in, cal.fused_one_shot_blocks, cal.fused_threads);
  if (bytes(in) <= cal.norm_push_max_bytes)
    return spec_of(push, in, cal.norm_push_blocks, cal.fused_threads);
  return spec_of(pull, in, cal.norm_pull_blocks, cal.fused_threads);
}

constexpr KernelSpec tune_all_reduce_rms_norm(Input in, const Hardware&, const Calibration& cal) {
  return fused_norm(Kernel::all_reduce_pull_one_shot_rms_norm,
                    Kernel::all_reduce_push_two_shot_rms_norm,
                    Kernel::all_reduce_pull_two_shot_rms_norm, in, cal);
}

constexpr KernelSpec tune_all_reduce_add_rms_norm(Input in, const Hardware&,
                                                  const Calibration& cal) {
  return fused_norm(Kernel::all_reduce_pull_one_shot_add_rms_norm,
                    Kernel::all_reduce_push_two_shot_add_rms_norm,
                    Kernel::all_reduce_pull_two_shot_add_rms_norm, in, cal);
}

// AttnRes as the norms: a block a row, never more blocks than rows (grid_of).
constexpr KernelSpec tune_all_reduce_add_attn_res_rms_norm(Input in, const Hardware&,
                                                        const Calibration& cal) {
  return bytes(in) <= cal.fused_one_shot_max_bytes
             ? spec_of(Kernel::all_reduce_pull_one_shot_add_attn_res_rms_norm, in,
                       cal.fused_one_shot_blocks, cal.fused_threads)
             : spec_of(Kernel::all_reduce_pull_two_shot_add_attn_res_rms_norm, in,
                       cal.attn_res_two_shot_blocks, cal.fused_threads);
}

// ALWAYS FUSED, as every op: one-shot up to one GEMM pass of rows, two-shot past it, 56 blocks of
// 512 threads (the GEMM strides over column tiles). It is slower than the unfused ops (about 68
// against 20 us at 1 token, 2026-09-30T20-23-38Z): a loss to fix, shown as one.
constexpr KernelSpec tune_all_reduce_rms_norm_gemm_add(Input in, const Hardware&,
                                                   const Calibration& cal) {
  const Kernel k = in.rows <= cal.gemm_one_shot_max_rows
                       ? Kernel::all_reduce_pull_one_shot_rms_norm_gemm_add
                       : Kernel::all_reduce_pull_two_shot_rms_norm_gemm_add;
  return spec_of(k, in, cal.gemm_tail_blocks, cal.fused_threads);
}

// =================================================================================================
// THE ONE ENTRY: the op's own function.
// =================================================================================================

constexpr KernelSpec tune(Op op, Input in, const Hardware& hw, const Calibration& cal) {
  switch (op) {
    case Op::all_reduce: return tune_all_reduce(in, hw, cal);
    case Op::all_reduce_rms_norm: return tune_all_reduce_rms_norm(in, hw, cal);
    case Op::all_reduce_add_rms_norm: return tune_all_reduce_add_rms_norm(in, hw, cal);
    case Op::all_reduce_add_attn_res_rms_norm:
      return tune_all_reduce_add_attn_res_rms_norm(in, hw, cal);
    case Op::all_reduce_rms_norm_gemm_add: return tune_all_reduce_rms_norm_gemm_add(in, hw, cal);
  }
  __builtin_unreachable();  // every Op is a case above
}

// EVERY TUNED SPEC IS A KERNEL THAT FITS, for every op at the smallest input and a large one: a
// tune_<op> never declines (a fusion that is on runs its fused op), and a spec past a capability is
// a compile error, not a kernel that overruns its signal slots or register arrays.
constexpr bool fits(const KernelSpec& k, Input in) {
  if (k.kernel == Kernel::none) return false;
  if (k.grid < 1 || k.grid > p2p::kMaxBlocks) return false;
  if (has_row_packs(k.kernel) &&
      row_packs_for(k.kernel, in.hidden * in.elem_bytes / kPackBytes, k.threads) == 0)
    return false;
  return k.threads >= kWaveSize && k.threads <= kMaxThreads && k.threads % kWaveSize == 0;
}
constexpr bool tuned_specs_fit() {
  for (int op = 0; op <= static_cast<int>(Op::all_reduce_rms_norm_gemm_add); ++op)
    for (const Input in : {Input{1, 8, 2, 0, 2}, Input{4096, 7168, 2, 0, p2p::kMaxRanks}})
      if (!fits(tune(static_cast<Op>(op), in, kTarget, kTargetCalibration), in)) return false;
  return true;
}
static_assert(tuned_specs_fit(), "a tune_<op> declines, or exceeds a kernel capability");

constexpr int elem_bytes(DType d) { return d == DType::f32 ? 4 : 2; }

// EACH OP'S ARGS AS THE OP AND THE INPUT that select and validate read.
constexpr Op op_of(const AllReduceArgs&) { return Op::all_reduce; }
constexpr Op op_of(const NormArgs& a) {
  return a.residual ? Op::all_reduce_add_rms_norm : Op::all_reduce_rms_norm;
}
constexpr Op op_of(const AttnResArgs&) { return Op::all_reduce_add_attn_res_rms_norm; }
constexpr Op op_of(const GemmTailArgs&) { return Op::all_reduce_rms_norm_gemm_add; }

inline Input input_of(const Handle& h, const AllReduceArgs& a) {
  const int e = elem_bytes(a.dtype);
  return {1, a.bytes / e, e, 0, h.world_size()};
}
inline Input input_of(const Handle& h, const NormArgs& a) {
  return {a.rows, a.hidden, elem_bytes(a.dtype), 0, h.world_size()};
}
inline Input input_of(const Handle& h, const AttnResArgs& a) {
  return {a.rows, a.hidden, elem_bytes(a.dtype), 0, h.world_size()};
}
inline Input input_of(const Handle& h, const GemmTailArgs& a) {
  return {a.rows, a.hidden, elem_bytes(a.dtype), a.n_cols, h.world_size()};
}

// What runs: tune's spec, or the forced kernel at the forced grid and block.
inline KernelSpec select(Op op, Input in, const Options& o) {
  const KernelSpec& f = o.forced;
  if (f.kernel == Kernel::none) return tune(op, in, kTarget, kTargetCalibration);
  return spec_of(f.kernel, in, f.grid, f.threads);
}

template <typename Args>
KernelSpec select(const Handle& h, const Args& a, const Options& o) {
  return select(op_of(a), input_of(h, a), o);
}

}  // namespace hip_comms
