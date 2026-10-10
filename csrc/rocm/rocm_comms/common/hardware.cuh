// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE HARDWARE: each target's facts, as the device and AMD's docs report them, and what was
// measured on it (`Calibration`). No decision lives here: build.cuh derives what a build is from
// them, select.cuh a launch from them and the input. A new target is one more `Hardware`;
// `kTarget` is the one the host tunes for.

#pragma once

#include <cstdint>

namespace hip_comms {

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
  int64_t vgpr_file_bytes;  // vector registers, split over the SIMDs (4-byte registers, per lane)
  int64_t sgpr_file_bytes;  // scalar registers
  int64_t lds_bytes;        // shared memory
  int64_t l1_bytes;         // vector L1
  // The memory system.
  int cache_line_bytes;
  int64_t l2_bytes_per_xcd;  // not shared across XCDs: a release to peers writes it back
  int64_t infinity_cache_bytes;
  int64_t hbm_bytes;
  double hbm_gbytes_per_s;  // peak
  // The fabric to the peers.
  int xgmi_links;                  // one to each peer in an 8-GPU node
  double xgmi_gbytes_per_s_a_way;  // peak, one link, one direction
  // The instruction set.
  int max_load_bytes;  // the widest vector load or store a lane issues
  int arch_vgprs;      // vector registers an instruction names (v0 up)
  int acc_vgprs;       // accumulation registers beside them (a0 up), where the compiler spills first
  int vgpr_granule;    // a wave's vector registers are allocated in multiples of it
};

// gfx950, AMD Instinct MI355X (MI350X is the same). Sources: `rocminfo` on n11 (2026-09-29, all 8
// GPUs) for the compute units; ROCm's GPU hardware specifications table (gpu-arch-specs) for the
// register files, caches and LDS; ROCm's MI350 architecture page (gpu-arch/mi350) for the XCDs,
// HBM and fabric. A link is 16 lanes at 38.4 Gbps: 76.8 GB/s a way, 7 of them "over 1 TB/s".
constexpr Hardware kGfx950 = {
    64,                  // wave_size (gfx9 runs 64-lane waves only)
    1024,                // max_workgroup
    256,                 // compute_units (32 active per XCD)
    32,                  // max_waves_per_cu
    8,                   // xcds
    4,                   // simds_per_cu
    512 * kKiB,          // vgpr_file_bytes
    12800,               // sgpr_file_bytes (12.5 KiB)
    160 * kKiB,          // lds_bytes
    32 * kKiB,           // l1_bytes
    128,                 // cache_line_bytes
    4 * kMiB,            // l2_bytes_per_xcd
    256 * kMiB,          // infinity_cache_bytes
    288 * kGiB,          // hbm_bytes (8 stacks of 36)
    8000.0,              // hbm_gbytes_per_s
    7,                   // xgmi_links
    16 * 38.4 / 8,       // xgmi_gbytes_per_s_a_way
    16,                  // max_load_bytes (global_load_dwordx4; CDNA4 ISA guide)
    256,                 // arch_vgprs
    256,                 // acc_vgprs
    8,                   // vgpr_granule (LLVM AMDGPUUsage, gfx90a and later, wave64)
};
static_assert(kGfx950.compute_units % kGfx950.xcds == 0, "every XCD has the same CUs");

// gfx942, AMD Instinct MI300X (MI325X is the same but for 256 GiB). Sources: ROCm's GPU hardware
// specifications table for the compute units, register files, caches and LDS; ROCm's MI300 page
// for the XCDs and HBM; AMD's MI300X data sheet for the fabric (7 links, 896 GB/s in all).
constexpr Hardware kGfx942 = {
    64,                  // wave_size
    1024,                // max_workgroup
    304,                 // compute_units (38 active per XCD)
    32,                  // max_waves_per_cu
    8,                   // xcds
    4,                   // simds_per_cu
    512 * kKiB,          // vgpr_file_bytes
    12800,               // sgpr_file_bytes (12.5 KiB)
    64 * kKiB,           // lds_bytes
    32 * kKiB,           // l1_bytes
    128,                 // cache_line_bytes
    4 * kMiB,            // l2_bytes_per_xcd
    256 * kMiB,          // infinity_cache_bytes
    192 * kGiB,          // hbm_bytes
    5300.0,              // hbm_gbytes_per_s
    7,                   // xgmi_links
    64.0,                // xgmi_gbytes_per_s_a_way
    16,                  // max_load_bytes (global_load_dwordx4; CDNA3 ISA guide)
    256,                 // arch_vgprs
    256,                 // acc_vgprs
    8,                   // vgpr_granule (LLVM AMDGPUUsage, gfx90a and later, wave64)
};
static_assert(kGfx942.compute_units % kGfx942.xcds == 0, "every XCD has the same CUs");

// MEASURED ON THE MACHINE, where `Hardware` is documented: by our probes (calibrate.py) and by our
// sweeps (the bench's forced launch configs), so it goes stale when the driver, firmware or our own
// kernels change. Each value cites the run that measured it; one not swept says so and whose it
// copies. A kernel's launch is not here: it is its template's configs (launch.cuh), and every
// op's choice, the plain all-reduce's included, is its tuned kernels (select.cuh).
struct Calibration {
  double ping_pong_ns;                    // a flag to a peer and back, median of every pair
};

// gfx950 on n11. MI300X has none yet.
constexpr Calibration kGfx950Calibration = {
    // calibrate.py, dev run 2026-09-30T19-02-08Z: 28 pairs 1274-1383 ns; a repeat
    // (2026-09-30T19-13-36Z) gave 1282, so about 5% run to run.
    .ping_pong_ns = 1334.0,
};

// THE TARGET THE HOST TUNES FOR, and what was measured on it.
constexpr const Hardware& kTarget               = kGfx950;
constexpr const Calibration& kTargetCalibration = kGfx950Calibration;
constexpr const char* kTargetArch               = "gfx950";

// THE DEVICE THIS COMPILE PASS IS FOR: a build compiles the device code once per offload arch
// (gfx942 and gfx950), each against its own facts; the host pass sees the tuning target.
#if defined(__gfx942__)
constexpr const Hardware& kDevice = kGfx942;
#else
constexpr const Hardware& kDevice = kTarget;
#endif
constexpr int kWaveSize = kDevice.wave_size;

// THE MOST COMPUTE UNITS ON ANY TARGET BUILT: for a layout host and device share (the peers' signal
// block), which one device pass's kDevice cannot size.
constexpr int kMaxComputeUnits =
    kGfx950.compute_units > kGfx942.compute_units ? kGfx950.compute_units : kGfx942.compute_units;

// THE MOST BLOCKS RESIDENT AT ONCE ON ANY TARGET BUILT: every CU full of one-wave blocks, so the peers'
// signal block never rules out a grid. Whether one kernel's grid is resident is its occupancy,
// which only the compiled kernel knows (resident_blocks below, checked by select).
constexpr int resident_waves(const Hardware& hw) { return hw.compute_units * hw.max_waves_per_cu; }
constexpr int kMaxResidentBlocks = resident_waves(kGfx950) > resident_waves(kGfx942)
                                       ? resident_waves(kGfx950)
                                       : resident_waves(kGfx942);

// THE MOST GPUS THAT CAN READ EACH OTHER'S MEMORY DIRECTLY: a link to every peer (the full xGMI
// mesh of one node), so a peer group is at most the links plus one. Beyond it there is no load
// path, only a network collective.
constexpr int kMaxPeers = (kGfx950.xgmi_links > kGfx942.xgmi_links ? kGfx950.xgmi_links
                                                                    : kGfx942.xgmi_links) + 1;

// THE MACHINE MODEL: what the device holds, from its `Hardware`.

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

// Waves a block may have when its LDS is `fixed` bytes plus `per_wave` for each wave: what the
// device's LDS holds, and no more than the block limit.
constexpr int lds_max_waves(const Hardware& hw, int64_t fixed, int64_t per_wave) {
  const int64_t fit = (hw.lds_bytes - fixed) / per_wave;
  const int64_t cap = hw.max_workgroup / hw.wave_size;
  return static_cast<int>(fit < cap ? fit : cap);
}

// THE COMPILER'S WAVE SIZE AGREES with the target's, or the in-wave shuffles are wrong.
#if defined(__AMDGCN_WAVEFRONT_SIZE)
static_assert(__AMDGCN_WAVEFRONT_SIZE == kWaveSize,
              "hardware.cuh's wave size is not the compiler's");
#endif

}  // namespace hip_comms
