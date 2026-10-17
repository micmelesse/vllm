// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE MACHINE AND THE BUILD, host facts (comms.md's DECIDED, H8): the target's documented
// facts (HARDWARE), what the build compiles for it (BUILD), and the limits the host sizes the
// peers' buffers by (kMaxRanks, kMaxBlocks). Device code reads two of them through
// kernels/common.cuh: kWaveSize (block_reduce's DPP sequence) and the pack's bytes (traits).

#pragma once

#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace hip_custom_communicator {

// =================================================================================================
// HARDWARE
//
// THE HARDWARE: each target's facts, as the device and AMD's docs report them.
// No decision lives here: BUILD derives what a build is from them,
// select.cuh a launch from them, the input and each op's tuned kernels
// (measured, each citing its sweep). A new target is one more `Hardware`;
// `kTarget` is the one the host tunes for.
// =================================================================================================

constexpr int64_t kKiB = 1024;
constexpr int64_t kMiB = 1024 * kKiB;
constexpr int64_t kGiB = 1024 * kMiB;

struct Hardware {
  // The compute units.
  int wave_size;         // lanes per wave
  int max_workgroup;     // threads per block the device allows
  int compute_units;     // across the device
  int max_waves_per_cu;  // resident at once
  int xcds;              // compute dies; each has its own L2
  int simds_per_cu;      // a wave runs on one SIMD
  // Per CU.
  int64_t vgpr_file_bytes;  // vector registers, split over the SIMDs (4-byte
                            // registers, per lane)
  int64_t sgpr_file_bytes;  // scalar registers
  int64_t lds_bytes;        // shared memory
  int64_t l1_bytes;         // vector L1
  // The memory system.
  int cache_line_bytes;
  int64_t l2_bytes_per_xcd;  // not shared across XCDs: a release to peers
                             // writes it back
  int64_t infinity_cache_bytes;
  int64_t hbm_bytes;
  double hbm_gbytes_per_s;  // peak
  // The fabric to the peers.
  int xgmi_links;                  // one to each peer in an 8-GPU node
  double xgmi_gbytes_per_s_a_way;  // peak, one link, one direction
  // The instruction set.
  int max_load_bytes;  // the widest vector load or store a lane issues
  int arch_vgprs;      // vector registers an instruction names (v0 up)
  int acc_vgprs;       // accumulation registers beside them (a0 up), where the
                       // compiler spills first
  int vgpr_granule;  // a wave's vector registers are allocated in multiples of
                     // it
};

// gfx950, AMD Instinct MI355X (MI350X is the same). Sources: `rocminfo` on n11
// (2026-09-29, all 8 GPUs) for the compute units; ROCm's GPU hardware
// specifications table (gpu-arch-specs) for the register files, caches and LDS;
// ROCm's MI350 architecture page (gpu-arch/mi350) for the XCDs, HBM and fabric.
// A link is 16 lanes at 38.4 Gbps: 76.8 GB/s a way, 7 of them "over 1 TB/s".
constexpr Hardware kGfx950 = {
    64,             // wave_size (gfx9 runs 64-lane waves only)
    1024,           // max_workgroup
    256,            // compute_units (32 active per XCD)
    32,             // max_waves_per_cu
    8,              // xcds
    4,              // simds_per_cu
    512 * kKiB,     // vgpr_file_bytes
    12800,          // sgpr_file_bytes (12.5 KiB)
    160 * kKiB,     // lds_bytes
    32 * kKiB,      // l1_bytes
    128,            // cache_line_bytes
    4 * kMiB,       // l2_bytes_per_xcd
    256 * kMiB,     // infinity_cache_bytes
    288 * kGiB,     // hbm_bytes (8 stacks of 36)
    8000.0,         // hbm_gbytes_per_s
    7,              // xgmi_links
    16 * 38.4 / 8,  // xgmi_gbytes_per_s_a_way
    16,             // max_load_bytes (global_load_dwordx4; CDNA4 ISA guide)
    256,            // arch_vgprs
    256,            // acc_vgprs
    8,              // vgpr_granule (LLVM AMDGPUUsage, gfx90a and later, wave64)
};
static_assert(kGfx950.compute_units % kGfx950.xcds == 0,
              "every XCD has the same CUs");

// THE TARGET: the one arch the build compiles device code for (gfx950,
// CMakeLists.txt) and the host tunes for; `supported` refuses every other.
constexpr const Hardware& kTarget = kGfx950;
constexpr const char* kTargetArch = "gfx950";

constexpr int kWaveSize = kTarget.wave_size;

// THE MOST COMPUTE UNITS ON THE TARGET: for a layout host and device share
// (the peers' signal block).
constexpr int kMaxComputeUnits = kTarget.compute_units;

// THE MOST BLOCKS RESIDENT AT ONCE ON THE TARGET: every CU full of
// one-wave blocks, so the peers' signal block never rules out a grid. Whether
// one kernel's grid is resident is its occupancy, which only the compiled
// kernel knows (resident_blocks below, checked by select).
constexpr int resident_waves(const Hardware& hw) {
  return hw.compute_units * hw.max_waves_per_cu;
}
constexpr int kMaxResidentBlocks = resident_waves(kTarget);

// THE MOST GPUS THAT CAN READ EACH OTHER'S MEMORY DIRECTLY: a link to every
// peer (the full xGMI mesh of one node), so a peer group is at most the links
// plus one. Beyond it there is no load path, only a network collective.
constexpr int kMaxPeers = kTarget.xgmi_links + 1;

// THE MACHINE MODEL: what the device holds, from its `Hardware`.

// Vector registers a thread may use when a block of `threads` must fit on one
// CU (a kernel's
// __launch_bounds__(threads, 1)): its SIMD's file shared by the waves the block
// puts there, and at most the architectural and accumulation registers
// together.
constexpr int vgprs_per_thread(const Hardware& hw, int threads) {
  const int waves = (threads + hw.wave_size - 1) / hw.wave_size;
  const int waves_on_simd = (waves + hw.simds_per_cu - 1) / hw.simds_per_cu;
  const int64_t per_lane =
      hw.vgpr_file_bytes / hw.simds_per_cu / hw.wave_size / 4;
  const int64_t v = per_lane / waves_on_simd;
  const int64_t cap = hw.arch_vgprs + hw.acc_vgprs;
  return static_cast<int>(v < cap ? v : cap);
}

// WHAT ONE COMPILED KERNEL USES, as the code object records it: a thread's
// vector registers and a block's LDS. Only the compiler knows them, so they are
// read at run time (Handle::resources_of).
struct Resources {
  int vgprs;
  int64_t lds_bytes;
};

// THE MOST BLOCKS OF A KERNEL RESIDENT AT ONCE ON THE DEVICE: on each CU, as
// many as its wave slots, its SIMDs' register files and its LDS hold. A peer
// barrier spins until the same block on every peer arrives, so a grid past it
// can hang.
constexpr int resident_blocks(const Hardware& hw, Resources r, int threads) {
  const int waves = (threads + hw.wave_size - 1) / hw.wave_size;
  const int waves_on_simd = (waves + hw.simds_per_cu - 1) / hw.simds_per_cu;
  const int64_t per_lane =
      hw.vgpr_file_bytes / hw.simds_per_cu / hw.wave_size / 4;
  const int used = r.vgprs < 1 ? 1 : r.vgprs;
  const int vgprs =
      (used + hw.vgpr_granule - 1) / hw.vgpr_granule * hw.vgpr_granule;
  int64_t per_cu = hw.max_waves_per_cu / waves;
  const int64_t by_registers = per_lane / vgprs / waves_on_simd;
  if (by_registers < per_cu) per_cu = by_registers;
  if (r.lds_bytes > 0 && hw.lds_bytes / r.lds_bytes < per_cu)
    per_cu = hw.lds_bytes / r.lds_bytes;
  return static_cast<int>(per_cu * hw.compute_units);
}

// THE COMPILER'S WAVE SIZE AGREES with the target's, or the in-wave shuffles
// are wrong.
#if defined(__AMDGCN_WAVEFRONT_SIZE)
static_assert(__AMDGCN_WAVEFRONT_SIZE == kWaveSize,
              "HARDWARE's wave size is not the compiler's");
#endif

// =================================================================================================
// BUILD
//
// THE BUILD: every number fixed at compile time, derived from the device's
// `Hardware` (HARDWARE) and nothing else; what depends on the input is
// select's (select.cuh), at run time. A number neither gives is a policy: one
// named line in `derive`.
// =================================================================================================

// A TENSOR'S ELEMENT TYPE, as the build names it. f32 is a norm weight's, not a
// call's.
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

// EVERYTHING THE BUILD FIXES, in one value: what is compiled, the memory and
// its unit, and the kernels' launch geometry. Python reads a projection of it
// (`build_info`).
struct BuildInfo {
  struct Supports {
    std::array<DType, 2>
        dtypes;  // a call's; dispatch instantiates exactly these
    std::array<int, 3> worlds;
  };
  struct Memory {
    int pack_bytes;  // a pack: the unit every kernel loads, sums and stores in
  };
  struct Kernels {
    int max_threads;  // the widest block, and every kernel's __launch_bounds__
    int max_waves;    // its waves
  };
  Supports supports;
  Memory memory;
  Kernels kernels;
};

constexpr BuildInfo derive(const Hardware& hw) {
  BuildInfo info{};
  // WHAT IS COMPILED, one list each: dispatch instantiates exactly these, check
  // refuses the rest.
  info.supports = {{DType::f16, DType::bf16}, {2, 4, 8}};

  BuildInfo::Memory& m = info.memory;
  // A PACK IS THE WIDEST LOAD, as in vLLM's and aiter's custom all-reduce and
  // NCCL.
  m.pack_bytes = hw.max_load_bytes;
  // THE EXCHANGE (partials and inputs) IS NOT HERE: it is sized at open from
  // the largest call the communicator serves (select.cuh's `exchange_sizes`),
  // never a constant.

  BuildInfo::Kernels& b = info.kernels;

  // THE WIDEST BLOCK WHOSE THREADS KEEP EVERY ARCHITECTURAL REGISTER. A bound
  // is a register trade: at 1024 threads a gfx950 thread gets 128 and the row
  // kernels spill (AttnRes at 2 packs; ISA 2026-09-29T23-48-40Z).
  b.max_threads = hw.wave_size;
  for (int t = hw.wave_size; t <= hw.max_workgroup; t += hw.wave_size)
    if (vgprs_per_thread(hw, t) >= hw.arch_vgprs) b.max_threads = t;
  b.max_waves = b.max_threads / hw.wave_size;

  return info;
}

// THIS COMPILE PASS'S BUILD: the device code for its own target, the host for
// the tuning target.
constexpr BuildInfo kBuild = derive(kTarget);

// WHETHER THE BUILD HOLDS a dtype or a world: dispatch instantiates exactly
// these.
template <typename ELEMENT, size_t NUM_BUILT>
constexpr bool built_in(const std::array<ELEMENT, NUM_BUILT>& built,
                        ELEMENT x) {
  for (const ELEMENT& b : built)
    if (b == x) return true;
  return false;
}
constexpr bool dtype_built(DType d) {
  return built_in(kBuild.supports.dtypes, d);
}
constexpr bool world_built(int world) {
  return built_in(kBuild.supports.worlds, world);
}

static_assert(kBuild.kernels.max_threads <= kTarget.max_workgroup &&
                  kBuild.kernels.max_threads % kWaveSize == 0,
              "the block limit must be whole waves the device can launch");


// THE PEERS' LIMITS, for the layouts host and device share (the signal block, handle.cuh): at
// most kMaxPeers ranks; ONE SIGNAL SLOT PER BLOCK THE HARDWARE CAN HOLD RESIDENT, so the signal
// block never rules out a grid. Which grid is fast is select's (select.cuh).
constexpr int kMaxRanks = kMaxPeers;
constexpr int kMaxBlocks = kMaxResidentBlocks;

}  // namespace hip_custom_communicator
