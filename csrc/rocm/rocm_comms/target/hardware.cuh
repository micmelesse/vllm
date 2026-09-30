// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE HARDWARE: each target's facts, as the device and AMD's docs report them, and what was
// measured on it (`Calibration`). No decision lives here: build.cuh derives what a build is from
// them, tune.cuh a launch from them and the input. A new target is one more `Hardware`; `kTarget`
// is the one the host tunes for.

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
};
static_assert(kGfx942.compute_units % kGfx942.xcds == 0, "every XCD has the same CUs");

// MEASURED ON THE MACHINE, where `Hardware` is documented: by our probes (calibrate.py) and by our
// sweeps (the bench's forced launch configs), so it goes stale when the driver, firmware or our own
// kernels change. Each field is named for what it holds and cites the run that measured it; a tune
// reads only its input, `Hardware` and `Calibration`, so every number a launch depends on is here.
struct Calibration {
  double ping_pong_ns;               // a p2p flag to a peer and back, median of every pair
  int64_t one_shot_max_bytes;        // the all-reduce's one-shot/two-shot crossover
  int64_t fused_one_shot_max_bytes;  // the fused ops' one-shot/two-shot crossover
  int fused_one_shot_blocks;         // a fused one-shot's grid, at most (one row a block)
  int norm_two_shot_blocks;          // the norms' fused two-shot grid, at most
  int attn_res_two_shot_blocks;      // AttnRes's fused two-shot grid, at most
  int fused_threads;                 // a fused kernel's block
  int gemm_tail_blocks;              // the GEMM tail's grid
  int gemm_lanes_per_col;            // the GEMM tail's lanes a column (a build: 1, 2, 4 or 8)
  int64_t gemm_one_shot_max_rows;    // the GEMM tail's one-shot/two-shot crossover
};

// gfx950 on n11. MI300X has none yet.
constexpr Calibration kGfx950Calibration = {
    // calibrate.py, dev run 2026-09-30T19-02-08Z: 28 pairs 1274-1383 ns; a repeat
    // (2026-09-30T19-13-36Z) gave 1282, so about 5% run to run.
    1334.0,
    // One-shot won at 56 KiB (7.12 against 7.83 us), two-shot at 112 KiB (7.87 against 8.19),
    // uncached scratch (2026-09-30T18-00-30Z).
    64 * kKiB,
    // The norms: moved to 64 KiB they lost at 16 tokens, 11.43 against 10.56 us
    // (2026-09-30T21-06-57Z).
    128 * kKiB,
    // Not swept: grid_of cuts it to the rows, so it matters only past 16 rows.
    16,
    // The column-slice kernel: best at 32-128 tokens, 9.9 / 10.7 / 13.3 us against 36 blocks'
    // 11.4 at 64 and 15.8 at 128 (2026-09-30T23-10-02Z); 72 was best at 1024 and 4096, all losing.
    128,
    // The row-slice kernel: 88 (the link-filling grid) lost at prefill, 169.3 against 154.5 us at
    // 4096 tokens (2026-09-30T21-30-15Z).
    36,
    // 256 was worse for the GEMM tail (2026-09-28); the norms and AttnRes not swept.
    512,
    // The GEMM tail's best at 1 row, 4 lanes a column (2026-09-28, log).
    56,
    // Picked at Kimi-K3's shape; 1, 2 and 8 were worse at 1 row (2026-09-28).
    4,
    // Not swept: one GEMM pass, where the one-shot kernel once had to stop.
    16,
};

// THE TARGET THE HOST TUNES FOR, and what was measured on it.
constexpr const Hardware& kTarget               = kGfx950;
constexpr const Calibration& kTargetCalibration = kGfx950Calibration;

// THE DEVICE THIS COMPILE PASS IS FOR: a build compiles the device code once per offload arch
// (gfx942 and gfx950), each against its own facts; the host pass sees the tuning target.
#if defined(__gfx942__)
constexpr const Hardware& kDevice = kGfx942;
#else
constexpr const Hardware& kDevice = kTarget;
#endif
constexpr int kWaveSize = kDevice.wave_size;

// THE MOST COMPUTE UNITS ON ANY TARGET BUILT: for a layout host and device share (p2p's signal
// block), which one device pass's kDevice cannot size.
constexpr int kMaxComputeUnits =
    kGfx950.compute_units > kGfx942.compute_units ? kGfx950.compute_units : kGfx942.compute_units;

// THE MOST GPUS THAT CAN READ EACH OTHER'S MEMORY DIRECTLY: a link to every peer (the full xGMI
// mesh of one node), so a peer group is at most the links plus one. Beyond it there is no load
// path, only a network collective.
constexpr int kMaxPeers = (kGfx950.xgmi_links > kGfx942.xgmi_links ? kGfx950.xgmi_links
                                                                    : kGfx942.xgmi_links) + 1;

// THE COMPILER'S WAVE SIZE AGREES with the target's, or the in-wave shuffles are wrong.
#if defined(__AMDGCN_WAVEFRONT_SIZE)
static_assert(__AMDGCN_WAVEFRONT_SIZE == kWaveSize,
              "hardware.cuh's wave size is not the compiler's");
#endif

}  // namespace hip_comms
