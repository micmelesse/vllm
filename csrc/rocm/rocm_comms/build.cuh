// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE BUILD: every number fixed at compile time, derived from the device's `Hardware` and its
// `Calibration` (hardware.cuh) and nothing else; what depends on the input is select's
// (impl/select.cuh), at run time. A number neither gives is a policy: one named line in `derive`.

#pragma once

#include <array>
#include <cstdint>

#include "machine/hardware.cuh"

namespace hip_comms {

// A TENSOR'S ELEMENT TYPE, as the build names it. f32 is a norm weight's, not a call's.
enum class DType { f16, bf16, f32 };

constexpr const char* to_string(DType d) {
  switch (d) {
    case DType::f16:
      return "f16";
    case DType::bf16:
      return "bf16";
    case DType::f32:
      return "f32";
  }
  return "unknown";
}

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

// WHAT ONE COMPILED KERNEL USES, as the code object records it: a thread's vector registers and
// a block's LDS. Only the compiler knows them, so they are read at run time (Handle::resources_of).
struct Resources {
  int vgprs;
  int64_t lds_bytes;
};

// THE MOST BLOCKS OF A KERNEL RESIDENT AT ONCE ON THE DEVICE: on each CU, as many as its wave
// slots, its SIMDs' register files and its LDS hold. A peer barrier spins until the same block on
// every peer arrives, so a grid past it can hang.
constexpr int resident_blocks(const Hardware& hw, Resources r, int threads) {
  const int waves         = (threads + hw.wave_size - 1) / hw.wave_size;
  const int waves_on_simd = (waves + hw.simds_per_cu - 1) / hw.simds_per_cu;
  const int64_t per_lane  = hw.vgpr_file_bytes / hw.simds_per_cu / hw.wave_size / 4;
  const int used          = r.vgprs < 1 ? 1 : r.vgprs;
  const int vgprs         = (used + hw.vgpr_granule - 1) / hw.vgpr_granule * hw.vgpr_granule;
  int64_t per_cu = hw.max_waves_per_cu / waves;
  const int64_t by_registers = per_lane / vgprs / waves_on_simd;
  if (by_registers < per_cu) per_cu = by_registers;
  if (r.lds_bytes > 0 && hw.lds_bytes / r.lds_bytes < per_cu) per_cu = hw.lds_bytes / r.lds_bytes;
  return static_cast<int>(per_cu * hw.compute_units);
}

// EVERYTHING THE BUILD FIXES, in one value: what is compiled, the memory and its unit, and the
// kernels' launch geometry. Python reads a projection of it (`build_info`).
struct BuildInfo {
  struct Supports {
    std::array<DType, 2> dtypes;  // a call's; dispatch instantiates exactly these
    std::array<int, 3> worlds;
  };
  struct Memory {
    int pack_bytes;          // a pack: the unit every kernel loads, sums and stores in
    int64_t staging_bytes;   // where an eager input is copied for its peers
    int64_t scratch_bytes;   // a two-shot's partial sums, per rank
    int64_t peer_ptr_slots;  // one per registered or captured buffer: its peers' addresses
  };
  struct Kernels {
    int max_threads;              // the widest block, and every kernel's __launch_bounds__
    int max_waves;                // its waves
    int gemm_rows;                // the GEMM tail's rows a pass
    int gemm_chunk;               // the GEMM tail's K-chunk staged in LDS, in packs
    int gemm_lanes;               // the GEMM tail's lanes a column, the one build of it
    int attn_res_sources;         // AttnRes's sources a block_reduce, the one build of it
    int attn_res_reduce_blocks;   // its pull two-shot's blocks that run the reduce-scatter
    double sync_timeout_seconds;  // how long a kernel waits on a peer before it traps
  };
  Supports supports;
  Memory memory;
  Kernels kernels;
};

constexpr BuildInfo derive(const Hardware& hw, const Calibration& cal) {
  BuildInfo info{};
  // WHAT IS COMPILED, one list each: dispatch instantiates exactly these, check refuses the rest.
  info.supports = {{DType::f16, DType::bf16}, {2, 4, 8}};

  BuildInfo::Memory& m = info.memory;
  // A PACK IS THE WIDEST LOAD, as in vLLM's and aiter's custom all-reduce and NCCL.
  m.pack_bytes = hw.max_load_bytes;
  // THE STAGING an eager input is copied into for its peers: policy. A staged build takes any
  // size through it a pass at a time; an in-place build on an eager input (the fused ops) needs it
  // whole, up to Kimi-K3's widest row (the one-all-reduce tail's 4096 x 17920 bf16, 147 MB).
  m.staging_bytes = 256 * kMiB;
  // THE TWO-SHOT SCRATCH, per rank, after the signal block: policy. It holds one rank's slice, so
  // it caps a two-shot buffer at scratch_bytes x ngpus (1 GiB at 8 ranks); a quantized two-shot
  // holds every rank's slice at half width, padded to whole grid strides (Kimi-K3's 4096 x 7168
  // bf16 prefill needs 74 MB at 36 blocks). check refuses a call past it (scratch_too_small).
  m.scratch_bytes = 128 * kMiB;
  // THE PEER-POINTER SLOTS, one per captured launch (capture sizes x collectives a forward): the
  // size vLLM gives the same array, 8 MB of slots. Registration past it fails at capture.
  m.peer_ptr_slots = 131072;

  BuildInfo::Kernels& b = info.kernels;

  // THE WIDEST BLOCK WHOSE THREADS KEEP EVERY ARCHITECTURAL REGISTER. A bound is a register trade:
  // at 1024 threads a gfx950 thread gets 128 and the row kernels spill (AttnRes at 2 packs, the
  // GEMM tail at any; ISA 2026-09-29T23-48-40Z).
  b.max_threads = hw.wave_size;
  for (int t = hw.wave_size; t <= hw.max_workgroup; t += hw.wave_size)
    if (vgprs_per_thread(hw, t) >= hw.arch_vgprs) b.max_threads = t;
  b.max_waves = b.max_threads / hw.wave_size;

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
      static_cast<int>((hw.lds_bytes - partials - reduce) / (int64_t{b.gemm_rows} * m.pack_bytes));

  // THE GEMM TAIL'S LANES A COLUMN, as measured: a template parameter, so one build, not four.
  b.gemm_lanes = cal.gemm_lanes_per_col;
  b.attn_res_sources = cal.attn_res_sources_per_reduce;
  b.attn_res_reduce_blocks = cal.attn_res.pull_reduce_blocks;
  // HOW LONG A KERNEL WAITS ON A PEER before it prints where it was and traps.
  b.sync_timeout_seconds = 10.0;
  return info;
}

// THIS COMPILE PASS'S BUILD: the device code for its own target, the host for the tuning target.
constexpr BuildInfo kBuild = derive(kDevice, kTargetCalibration);

static_assert(kBuild.kernels.max_threads <= kDevice.max_workgroup &&
                  kBuild.kernels.max_threads % kWaveSize == 0,
              "the block limit must be whole waves the device can launch");
static_assert(kBuild.kernels.gemm_lanes == 1 || kBuild.kernels.gemm_lanes == 2 ||
                  kBuild.kernels.gemm_lanes == 4 || kBuild.kernels.gemm_lanes == 8,
              "the GEMM tail splits a wave's lanes over its columns: 1, 2, 4 or 8");
// Waves a block may have when its LDS is `fixed` bytes plus `per_wave` for each wave: what the
// device's LDS holds, and no more than the block limit.
constexpr int lds_max_waves(const Hardware& hw, int64_t fixed, int64_t per_wave) {
  const int64_t fit = (hw.lds_bytes - fixed) / per_wave;
  const int64_t cap = hw.max_workgroup / hw.wave_size;
  return static_cast<int>(fit < cap ? fit : cap);
}

}  // namespace hip_comms
