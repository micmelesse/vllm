// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE BUILD: every number fixed at compile time, derived from the device's `Hardware` and its
// `Calibration` (hardware.cuh) and nothing else; what depends on the input is select's
// (impl/select.cuh), at run time. A number neither gives is a policy: one named line in `derive`.

#pragma once

#include <cstdint>

#include "hardware.cuh"

namespace hip_comms {

// Vector registers a thread may use when a block of `threads` must fit on one CU (a kernel's
// __launch_bounds__(threads, 1)): its SIMD's file shared by the waves the block puts there, and
// at most the architectural and accumulation registers together.
constexpr int vgprs_per_thread(const Hardware& hw, int threads) {
  const int waves         = (threads + hw.wave_size - 1) / hw.wave_size;
  const int waves_on_simd = (waves + hw.simds_per_cu - 1) / hw.simds_per_cu;
  const int64_t per_lane  = hw.vgpr_file_bytes / hw.simds_per_cu / hw.wave_size / 4;
  const int64_t v         = per_lane / waves_on_simd;
  const int64_t cap       = hw.arch_vgprs + hw.acc_vgprs;
  return static_cast<int>(v < cap ? v : cap);
}

constexpr int floor_pow2(int x) {
  int p = 1;
  while (p * 2 <= x) p *= 2;
  return p;
}

struct Build {
  int pack_bytes;           // a pack: the unit every kernel loads, sums and stores in
  int max_threads;          // the widest block, and every kernel's __launch_bounds__
  int norm_row_packs;       // a norm's (and the GEMM tail's) row packs a thread, at most
  int attn_res_row_packs;   // AttnRes's, at most
  int pipelined_row_packs;  // a pipelined row kernel's (two rows' loads in flight), at most
  int gemm_rows;            // the GEMM tail's rows a pass
  int gemm_chunk;           // the GEMM tail's K-chunk staged in LDS, in packs
  int gemm_lanes;           // the GEMM tail's lanes a column, the one build of it
  int attn_res_sources;     // AttnRes's sources a block_reduce, the one build of it
};

constexpr Build derive(const Hardware& hw, const Calibration& cal) {
  Build b{};
  // A PACK IS THE WIDEST LOAD, as in vLLM's and aiter's custom all-reduce and NCCL.
  b.pack_bytes = hw.max_load_bytes;

  // THE WIDEST BLOCK WHOSE THREADS KEEP EVERY ARCHITECTURAL REGISTER. A bound is a register trade:
  // at 1024 threads a gfx950 thread gets 128 and the row kernels spill (AttnRes at 2 packs, the
  // GEMM tail at any; ISA 2026-09-29T23-48-40Z).
  b.max_threads = hw.wave_size;
  for (int t = hw.wave_size; t <= hw.max_workgroup; t += hw.wave_size)
    if (vgprs_per_thread(hw, t) >= hw.arch_vgprs) b.max_threads = t;

  // A ROW KERNEL'S PACKS A THREAD: each pack keeps a load from every peer in flight at once
  // (peers_reduce). AttnRes then holds fp32 copies of it (the prefix, the weights, the output, and
  // one a source of a reduction's), after the loads are summed, so the larger phase counts. Policy:
  // a row's registers take at most half of the thread's, the rest its addresses, reductions and
  // the norm. At all of them the norms spilled (8 packs of 8 peers is
  // 256 registers: ISA 2026-09-30T20-23-38Z), and AttnRes at 4.
  const int pack_vgprs  = b.pack_bytes / 4;
  const int row_budget  = hw.arch_vgprs / 2;
  const int in_flight   = (hw.xgmi_links + 1) * pack_vgprs;
  // A pack of the narrowest T built (bf16) as fp32, for each copy.
  const int attn_state  = (3 + cal.attn_res_sources_per_reduce) * (b.pack_bytes / 2);
  b.norm_row_packs      = floor_pow2(row_budget / in_flight);
  b.attn_res_row_packs  = floor_pow2(row_budget / (in_flight > attn_state ? in_flight : attn_state));
  // A PIPELINED ROW KERNEL holds the next row's loads beside this row's: twice the in-flight
  // registers (the pull norm two-shot spilled at 4 packs: 2026-10-01T00-06-30Z).
  b.pipelined_row_packs = floor_pow2(row_budget / (2 * in_flight));

  // Policy: THE GEMM TAIL SUMS 16 ROWS A PASS, one fp32 accumulator a row in each lane; more rows
  // loop over passes.
  b.gemm_rows = 16;

  // THE GEMM TAIL'S K-CHUNK IS WHAT LDS HOLDS beside the rest of the block's: the widest block's
  // [gemm_rows][wave_size] fp32 partials (one lane a column, the most partials a split gives) and a
  // block_reduce's (one value per wave, plus the total).
  const int waves           = b.max_threads / hw.wave_size;
  const int64_t partials    = int64_t{waves} * b.gemm_rows * hw.wave_size * 4;
  const int64_t reduce      = (int64_t{waves} + 1) * 4;
  b.gemm_chunk =
      static_cast<int>((hw.lds_bytes - partials - reduce) / (int64_t{b.gemm_rows} * b.pack_bytes));

  // THE GEMM TAIL'S LANES A COLUMN, as measured: a template parameter, so one build, not four.
  b.gemm_lanes = cal.gemm_lanes_per_col;
  b.attn_res_sources = cal.attn_res_sources_per_reduce;
  return b;
}

// THIS COMPILE PASS'S BUILD: the device code for its own target, the host for the tuning target.
constexpr Build kBuild = derive(kDevice, kTargetCalibration);

constexpr int kPackBytes = kBuild.pack_bytes;
constexpr int kMaxThreads = kBuild.max_threads;
constexpr int kMaxWaves = kMaxThreads / kWaveSize;
constexpr int kGemmRows = kBuild.gemm_rows;
constexpr int kGemmChunk = kBuild.gemm_chunk;

static_assert(kMaxThreads <= kDevice.max_workgroup && kMaxThreads % kWaveSize == 0,
              "the block limit must be whole waves the device can launch");
static_assert(kBuild.gemm_lanes == 1 || kBuild.gemm_lanes == 2 || kBuild.gemm_lanes == 4 ||
                  kBuild.gemm_lanes == 8,
              "the GEMM tail splits a wave's lanes over its columns: 1, 2, 4 or 8");
static_assert(kBuild.norm_row_packs <= 8 && kBuild.attn_res_row_packs <= 8 &&
                  kBuild.pipelined_row_packs <= 8,
              "impl/launch.cuh builds up to 8 packs a thread");

// Waves a block may have when its LDS is `fixed` bytes plus `per_wave` for each wave: what the
// device's LDS holds, and no more than the block limit.
constexpr int lds_max_waves(const Hardware& hw, int64_t fixed, int64_t per_wave) {
  const int64_t fit = (hw.lds_bytes - fixed) / per_wave;
  const int64_t cap = hw.max_workgroup / hw.wave_size;
  return static_cast<int>(fit < cap ? fit : cap);
}

}  // namespace hip_comms
