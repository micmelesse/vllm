// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE HARDWARE: each target's facts, as the device and AMD's docs report them, and what was
// measured on it (`Calibration`). No decision lives here: build.cuh derives what a build is from
// them, impl/select.cuh a launch from them and the input. A new target is one more `Hardware`;
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
// kernels change. EACH OP HAS ITS OWN, AND EACH KERNEL ITS OWN LAUNCH: a value measured on one
// kernel is never another's by sharing a field; one not swept says so and whose it copies. Every
// value cites the run that measured it; a tune reads only its input, `Hardware` and `Calibration`.
struct Launch {
  int blocks;   // the grid, at most (a row kernel's is cut to its rows)
  int threads;  // the block
};

// The norms (rms_norm, add_rms_norm): one-shot, then the push two-shot (a column split), then the
// pull two-shot (a row split), at two crossovers.
struct NormCalibration {
  int64_t one_shot_max_bytes;
  int64_t push_max_bytes;
  Launch one_shot;
  Launch push;
  Launch pull;
};

// AttnRes: one-shot, then the push two-shot, then the pull two-shot (both split columns).
struct AttnResCalibration {
  int64_t one_shot_max_bytes;
  int64_t push_max_bytes;
  Launch one_shot;
  Launch push;
  Launch pull;
  int pull_reduce_blocks;  // of the pull's grid, the blocks that run its reduce-scatter
  int pull_tile_m;         // its AttnRes tile: TILE_M rows a block at once
};

// A norm then a GEMM (rms_norm_gemm, and with the add rms_norm_gemm_add).
struct GemmCalibration {
  int64_t one_shot_max_rows;  // one GEMM pass of rows
  Launch one_shot;
  Launch two_shot;
};

struct Calibration {
  double ping_pong_ns;                    // a p2p flag to a peer and back, median of every pair
  int64_t all_reduce_one_shot_max_bytes;  // the plain all-reduce's (its grid is derived)
  int gemm_lanes_per_col;  // grid_gemm's lanes a column, a build both GEMM ops' kernels share
  int attn_res_sources_per_reduce;  // AttnRes's sources a block_reduce, a build its kernels share
  NormCalibration rms_norm;
  NormCalibration add_rms_norm;
  AttnResCalibration attn_res;
  GemmCalibration rms_norm_gemm;
  GemmCalibration rms_norm_gemm_add;
  Launch rms_scale_add_two_shot;  // the one-all-reduce tail's row two-shot
};

// gfx950 on n11. MI300X has none yet. The fused kernels' 512-thread block: 256 was worse for the
// GEMM tail (2026-09-28); not swept for the others, which copy it.
constexpr Calibration kGfx950Calibration = {
    // calibrate.py, dev run 2026-09-30T19-02-08Z: 28 pairs 1274-1383 ns; a repeat
    // (2026-09-30T19-13-36Z) gave 1282, so about 5% run to run.
    .ping_pong_ns = 1334.0,
    // One-shot won at 56 KiB (7.12 against 7.83 us), two-shot at 112 KiB (7.87 against 8.19),
    // uncached scratch (2026-09-30T18-00-30Z).
    .all_reduce_one_shot_max_bytes = 64 * kKiB,
    // Picked at Kimi-K3's shape on the GEMM tail; 1, 2 and 8 were worse at 1 row (2026-09-28).
    .gemm_lanes_per_col = 4,
    // 1: at 4 (Triton's tile) the row got slower, 6.48 -> 8.16 us a row and 147.5 -> 171.0 at 4096
    // tokens, with 100 -> 166 VGPRs (stamps 2026-10-01T06-26-55Z against 04-15-48Z).
    .attn_res_sources_per_reduce = 1,
    .rms_norm =
        {
            // Moved to 64 KiB it lost at 16 tokens, 11.43 against 10.56 us (2026-09-30T21-06-57Z).
            .one_shot_max_bytes = 128 * kKiB,
            // Push won through 1.75 MiB (256 tokens of 3584 bf16: 18.04 against pull's 18.41 and
            // unfused 18.37), pull from 2.6 MiB (2026-10-01T03-26-44Z).
            .push_max_bytes = 1792 * kKiB,
            // Not swept: grid_of cuts it to the rows, so it matters only past 16 rows.
            .one_shot = {16, 512},
            // 256 the best of 48-256 at 192-256 tokens (15.17 and 18.04 against 15.55 and 18.11 at
            // 128; 2026-10-01T03-26-44Z); a row a block below that.
            .push = {256, 512},
            // Pipelined, 48 the best of 36-96 at 2048-4096 tokens (79.3 and 145.4 against 81.0 and
            // 147.6 at 36), within 0.8 of 36 below (2026-10-01T02-59-52Z).
            .pull = {48, 512},
        },
    .add_rms_norm =
        {
            // Not swept: rms_norm's.
            .one_shot_max_bytes = 128 * kKiB,
            // Push won through 1.31 MiB (192 tokens: 15.48 against pull's 16.28), pull at 1.75 MiB
            // (18.36 against push's 18.46; 2026-10-01T03-26-44Z).
            .push_max_bytes = 1344 * kKiB,
            // Not swept: rms_norm's.
            .one_shot = {16, 512},
            // 256 the best of 48-256 at 192 tokens (15.48 against 16.75 at 128;
            // 2026-10-01T03-26-44Z).
            .push = {256, 512},
            // 48 the best of 36-96 at every size from 512 to 4096 tokens (80.6 and 149.7 us at
            // 2048 and 4096 against 84.3 and 154.7 at 36; 2026-10-01T02-59-52Z).
            .pull = {48, 512},
        },
    .attn_res =
        {
            // None: the push two-shot beat the one-shot from 1 token (12.89 against 13.68 us; 14.03
            // against 15.18 at 8; 2026-10-01T03-57-23Z), so AttnRes starts at the push.
            .one_shot_max_bytes = 0,
            // Push won through 1.75 MiB (128 tokens of 7168 bf16: 22.82 against unfused 23.22) and
            // through 3.5 MiB against the pull at its grid (256 tokens: 34.8 against 35.8-38.7 us,
            // 2026-10-01T22-56-58Z, 2026-10-01T23-00-47Z).
            .push_max_bytes = 3584 * kKiB,
            // Not swept: the norms'.
            .one_shot = {16, 512},
            // 256 the best of 32-256 at 256-1024 tokens (34.58, 70.78, 143.98 against 39.93, 83.15,
            // 159.33 at 128), a row a block below that (2026-10-01T03-57-23Z).
            .push = {256, 512},
            // The column split wants a wide grid (AttnRes is compute a row): at 7168, 192 is within
            // about 5% of the best of 16-256 from 512 to 4096 tokens; 4096 at 447.2 us against
            // 1160.7
            // at the 36 it had, the row-split norm's (2026-10-01T22-56-58Z, 2026-10-01T23-00-47Z).
            .pull = {192, 512},
            // ITS REDUCE-SCATTER ON FEWER: reads queue behind the links past a few dozen blocks,
            // while
            // AttnRes is compute a row and wants the whole grid. At 4096 tokens the reduce-scatter
            // took 146.6 us on 32 blocks against 218.9 on 192, AttnRes 1110.5 against 199.4
            // (stamps, 2026-10-01T23-45-31Z and 2026-10-01T23-50-54Z).
            .pull_reduce_blocks = 32,
            // 1: TILE_M = 2 lost at every grid, best 480.0 us at 128 blocks against 412.8 at 1 on
            // 192
            // (4096 x 7168, 2026-10-02T21-19-22Z).
            .pull_tile_m = 1,
        },
    .rms_norm_gemm =
        {
            // Not swept: rms_norm_gemm_add's.
            .one_shot_max_rows = 16,
            // Not swept: rms_norm_gemm_add's.
            .one_shot = {56, 512},
            // Not swept: rms_norm_gemm_add's.
            .two_shot = {56, 512},
        },
    .rms_norm_gemm_add =
        {
            // Not swept: one GEMM pass, where the one-shot kernel once had to stop.
            .one_shot_max_rows = 16,
            // The best at 1 row, 4 lanes a column (2026-09-28, log).
            .one_shot = {56, 512},
            // Not swept: the one-shot's.
            .two_shot = {56, 512},
        },
    // About 32 blocks keeps the links fed; more queue behind them: at [T, 17920] bf16, within 1% of
    // the best of 8-48 blocks from 8 to 4096 tokens, and 1024 tokens 146.9 us at 32 against 258.4
    // at 256 (2026-10-01T22-07-13Z).
    .rms_scale_add_two_shot = {32, 512},
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

// THE MOST COMPUTE UNITS ON ANY TARGET BUILT: for a layout host and device share (p2p's signal
// block), which one device pass's kDevice cannot size.
constexpr int kMaxComputeUnits =
    kGfx950.compute_units > kGfx942.compute_units ? kGfx950.compute_units : kGfx942.compute_units;

// THE MOST BLOCKS RESIDENT AT ONCE ON ANY TARGET BUILT: every CU full of one-wave blocks, so p2p's
// signal block never rules out a grid. Whether one kernel's grid is resident is its occupancy,
// which only the compiled kernel knows (build.cuh's resident_blocks, checked by validate).
constexpr int resident_waves(const Hardware& hw) { return hw.compute_units * hw.max_waves_per_cu; }
constexpr int kMaxResidentBlocks = resident_waves(kGfx950) > resident_waves(kGfx942)
                                       ? resident_waves(kGfx950)
                                       : resident_waves(kGfx942);

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
