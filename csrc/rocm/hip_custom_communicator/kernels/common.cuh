// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// COMMON: everything a kernel may call and the machine under it, in sections,
// each built on those above it: HARDWARE, BUILD, UTILS, TILE, MEMORY,
// ELEMENTWISE, REDUCE, DOT, SOFTMAX, PEERS, PTR, BARRIER, COLLECTIVES, then
// THE INTERFACE, last because it passes to them all. Each primitive is
// implemented in its section's namespace impl, and the interface gives each one
// a small function that passes its arguments on, every template parameter by
// name, so what reaches each primitive is visible there. A kernel calls the
// interface and never names impl:: (dev/lint_kernels.sh).
//
// CONCURRENCY IS PART OF EACH PRIMITIVE'S CONTRACT, so a kernel keeps it by
// using them:
//   1. Global memory only through the tile loads and stores: a kernel never
//      indexes a row.
//   2. A tile is loaded whole (tile_load, peers_load: one round trip) before
//      anything is stored; a pack loaded between stores waits a round trip,
//      since a store may alias it.
//   3. A load that does not depend on a reduction is issued before it
//      (weights, the next row or source), so its round trip runs under the
//      reduction.
//
// THE TYPES a kernel names, by the section that defines them:
//   HARDWARE   Hardware, kTarget, kWaveSize: the device's documented facts
//   BUILD      kBuild: what is compiled for the device
//   TILE       Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N,
//              THREADS_PER_BLOCK, ACC_DTYPE, TILE_R, THREADS_R>
//              {M, N, offs_m, offs_n, row_step}: a chunk of an M x N tensor,
//              the block's threads over it THREADS_M x THREADS_N, this
//              thread's elements held as ACC_DTYPE; R x M x N with a copy a
//              rank when TILE_R > 1 (a group of threads a rank, slice() this
//              thread's); to<U>(), like<U>(), zeros<U>()
//   REDUCE     Sum, Max (a reduction's operation), Axis::m | n
//   SOFTMAX    OnlineSoftmax
//   PEERS      StatusField, kStatusFields, kMaxRanks, kMaxBlocks
//   BARRIER    Until: what a barrier waits for (launched, visible, read)

#pragma once

#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <type_traits>

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

// gfx942, AMD Instinct MI300X (MI325X is the same but for 256 GiB). Sources:
// ROCm's GPU hardware specifications table for the compute units, register
// files, caches and LDS; ROCm's MI300 page for the XCDs and HBM; AMD's MI300X
// data sheet for the fabric (7 links, 896 GB/s in all).
constexpr Hardware kGfx942 = {
    64,          // wave_size
    1024,        // max_workgroup
    304,         // compute_units (38 active per XCD)
    32,          // max_waves_per_cu
    8,           // xcds
    4,           // simds_per_cu
    512 * kKiB,  // vgpr_file_bytes
    12800,       // sgpr_file_bytes (12.5 KiB)
    64 * kKiB,   // lds_bytes
    32 * kKiB,   // l1_bytes
    128,         // cache_line_bytes
    4 * kMiB,    // l2_bytes_per_xcd
    256 * kMiB,  // infinity_cache_bytes
    192 * kGiB,  // hbm_bytes
    5300.0,      // hbm_gbytes_per_s
    7,           // xgmi_links
    64.0,        // xgmi_gbytes_per_s_a_way
    16,          // max_load_bytes (global_load_dwordx4; CDNA3 ISA guide)
    256,         // arch_vgprs
    256,         // acc_vgprs
    8,           // vgpr_granule (LLVM AMDGPUUsage, gfx90a and later, wave64)
};
static_assert(kGfx942.compute_units % kGfx942.xcds == 0,
              "every XCD has the same CUs");

// THE TARGET THE HOST TUNES FOR.
constexpr const Hardware& kTarget = kGfx950;
constexpr const char* kTargetArch = "gfx950";

// THE DEVICE THIS COMPILE PASS IS FOR: a build compiles the device code once
// per offload arch (gfx942 and gfx950), each against its own facts; the host
// pass sees the tuning target.
#if defined(__gfx942__)
constexpr const Hardware& kDevice = kGfx942;
#else
constexpr const Hardware& kDevice = kTarget;
#endif
constexpr int kWaveSize = kDevice.wave_size;

// THE MOST COMPUTE UNITS ON ANY TARGET BUILT: for a layout host and device
// share (the peers' signal block), which one device pass's kDevice cannot size.
constexpr int kMaxComputeUnits = kGfx950.compute_units > kGfx942.compute_units
                                     ? kGfx950.compute_units
                                     : kGfx942.compute_units;

// THE MOST BLOCKS RESIDENT AT ONCE ON ANY TARGET BUILT: every CU full of
// one-wave blocks, so the peers' signal block never rules out a grid. Whether
// one kernel's grid is resident is its occupancy, which only the compiled
// kernel knows (resident_blocks below, checked by select).
constexpr int resident_waves(const Hardware& hw) {
  return hw.compute_units * hw.max_waves_per_cu;
}
constexpr int kMaxResidentBlocks = resident_waves(kGfx950) >
                                           resident_waves(kGfx942)
                                       ? resident_waves(kGfx950)
                                       : resident_waves(kGfx942);

// THE MOST GPUS THAT CAN READ EACH OTHER'S MEMORY DIRECTLY: a link to every
// peer (the full xGMI mesh of one node), so a peer group is at most the links
// plus one. Beyond it there is no load path, only a network collective.
constexpr int kMaxPeers =
    (kGfx950.xgmi_links > kGfx942.xgmi_links ? kGfx950.xgmi_links
                                             : kGfx942.xgmi_links) +
    1;

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
    int64_t staging_bytes;   // where an eager input is copied for its peers
    int64_t scratch_bytes;   // a two-shot's partial sums, per rank
    int64_t peer_ptr_slots;  // one per registered or captured buffer: its
                             // peers' addresses
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
  // THE STAGING an eager input is copied into for its peers: policy. A staged
  // build takes any size through it a pass at a time; an in-place build on an
  // eager input (the fused ops) needs it whole, up to Kimi-K3's widest row (its
  // 4096 x 7168 bf16 prefill, 59 MB).
  m.staging_bytes = 256 * kMiB;
  // THE TWO-SHOT SCRATCH, per rank, after the signal block: policy. It holds
  // one rank's slice, so it caps a two-shot buffer at scratch_bytes x ngpus (1
  // GiB at 8 ranks); a quantized two-shot holds every rank's slice at half
  // width, padded to whole grid strides (Kimi-K3's 4096 x 7168 bf16 prefill
  // needs 74 MB at 36 blocks). check refuses a call past it
  // (scratch_too_small).
  m.scratch_bytes = 128 * kMiB;
  // THE PEER-POINTER SLOTS, one per captured launch (capture sizes x
  // collectives a forward): the size vLLM gives the same array, 8 MB of slots.
  // Registration past it fails at capture.
  m.peer_ptr_slots = 131072;

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
constexpr BuildInfo kBuild = derive(kDevice);

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

static_assert(kBuild.kernels.max_threads <= kDevice.max_workgroup &&
                  kBuild.kernels.max_threads % kWaveSize == 0,
              "the block limit must be whole waves the device can launch");

// =================================================================================================
// UTILS
//
// THE HELPERS every op and kernel uses: the pack (16 bytes, the unit everything
// loads, sums and stores in), and every source's packs in registers.
// =================================================================================================

#define DINLINE __device__ __forceinline__

// THE TARGETS OUR KERNELS ARE WRITTEN FOR: MI300 (gfx942) and MI355 (gfx950).
// vLLM's ROCm build compiles every file for more (gfx90a and the Radeon
// gfx11xx/gfx12xx), where our kernels use what does not exist (DPP row
// broadcasts, gfx9's cache bits), so on those every kernel's body is left out
// and traps. Nothing launches one there: `supported` refuses the device first.
// The host pass, which generates no device code, sees the whole body.
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx942__) || \
    defined(__gfx950__)
  #define HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED 1
#else
  #define HIP_CUSTOM_COMMUNICATOR_ARCH_SUPPORTED 0
#endif

template <typename DTYPE, int VEC_SIZE>
struct __align__(sizeof(DTYPE) * VEC_SIZE) vec {
  DTYPE d[VEC_SIZE];
};

// A PACK: kBuild.memory.pack_bytes (BUILD: the widest load, 8 bf16),
// the unit every kernel loads, sums and stores in. `num_packs` counts them.

template <typename DTYPE>
struct traits {
  static constexpr int N = kBuild.memory.pack_bytes / sizeof(DTYPE);
  using V = vec<DTYPE, N>;
};

// =================================================================================================
// TILE
//
// THE TILE: how a kernel cuts its work, in Triton's terms (TILE_M, offs_m, M, a
// mask for the last tile), and what a thread holds of it.
// =================================================================================================

// A TILE, in Triton's terms, and THE ONE TYPE A KERNEL WORKS IN: a chunk of an
// M x N tensor, the block's work at once (TILE_M rows from offs_m, every
// row_step-th, and TILE_N columns from offs_n), and the elements of it this
// thread holds. A 3-D TILE IS R x M x N: TILE_R copies of the chunk, one a rank
// (R the ranks, the axis a collective reduces over; K stays the GEMM's), each
// read from its own rank's memory at that rank's strides. Rows and columns
// count elements, as the tensor does. A row is real when it is below M and a
// column when it is below N (the last tile may be short; a slice is a tile
// whose N is the slice's end).
//   DTYPE              the tensor's dtype in memory: a thread holds
//   kPack-element groups, one
//                      16-byte load of it, so a quantized DTYPE holds wider
//                      groups and nothing else changes
//   THREADS_M x THREADS_N  THE LAYOUT, the block's threads over the tile as the
//   tile is TILE_M x
//                      TILE_N elements: THREADS_N threads share a row's
//                      columns, every THREADS_N-th group each, and THREADS_M
//                      rows go at once. 1 x the block for a norm, reducing
//                      along the row; 8 x 64 (a wave a row) for a narrow slice
//                      of many rows. Compile-time, since it sizes the registers
//   THREADS_PER_BLOCK  the kernel's block: a layout of all of it holds a place
//   in every thread, so
//                      nothing compares threadIdx (the launch bounds do not
//                      tell the compiler)
//   TILE_R, THREADS_R  the rank axis and its layout: THREADS_R groups of
//   THREADS_M x THREADS_N
//                      threads, group r holding rank r's copy (THREADS_R ==
//                      TILE_R: a thread holds one rank's elements, the
//                      collectives meeting in LDS). 1 x 1 for a 2-D tile
//   ACC_DTYPE          what the elements are held as: DTYPE as loaded, float
//   for the math
//                      (`to<float>()` keeps every element where it is), or a
//                      weight's own dtype
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N,
          int THREADS_PER_BLOCK, typename ACC_DTYPE = DTYPE, int TILE_R = 1,
          int THREADS_R = 1>
struct Tile {
  static constexpr int kPack = traits<DTYPE>::N;  // elements in a group
  static constexpr int kTileR =
      TILE_R;  // the chunk, ranks x rows x columns in elements
  static constexpr int kTileM = TILE_M;
  static constexpr int kTileN = TILE_N;
  static constexpr int kThreadsR = THREADS_R;
  static constexpr int kSliceThreads =
      THREADS_M * THREADS_N;                                  // a rank's copy's
  static constexpr int kThreads = THREADS_R * kSliceThreads;  // the layout's
  static_assert(THREADS_R == TILE_R || TILE_R == 1,
                "a thread holds one rank's copy (THREADS_R == TILE_R) or the "
                "tile is 2-D");
  static constexpr int kThreadsPerBlock = THREADS_PER_BLOCK;
  static_assert(kThreads <= THREADS_PER_BLOCK,
                "a layout is the block's threads or fewer");
  static constexpr int kThreadsM = THREADS_M;
  static constexpr int kThreadsN = THREADS_N;
  static_assert(TILE_M % THREADS_M == 0,
                "a tile's rows are whole for every row of threads");
  static_assert(TILE_N % (kPack * THREADS_N) == 0,
                "a tile's columns are whole groups for every thread of a row");
  static constexpr int kRows = TILE_M / THREADS_M;  // rows this thread holds
  static constexpr int K =
      TILE_N / (kPack * THREADS_N);  // groups of a row it holds
  using Dtype = DTYPE;
  using Acc = ACC_DTYPE;
  using Pack = vec<ACC_DTYPE, kPack>;  // how a group is loaded and stored, not
                                       // how it is held
  int M, N;
  int offs_m, offs_n;
  int row_step = 1;
  ACC_DTYPE v[kRows][K][kPack];

  // A TILE IS ITS PLACE; ITS ELEMENTS START UNSET, as registers do: zeroed by
  // default they were 32 moves a source in AttnRes's loop, every tile then
  // loaded over them (ISA 2026-10-04T02-22-41Z). zeros<U>() where a sum starts.
  Tile() = default;
  DINLINE Tile(int m, int n, int offs_m_, int offs_n_, int row_step_ = 1)
      : M(m), N(n), offs_m(offs_m_), offs_n(offs_n_), row_step(row_step_) {}

  // THIS THREAD'S GROUP k OF A ROW, in groups from the row's start: past N
  // clamped to the last, so every load stays in bounds; its mask is 1 below N
  // and 0 past it (a float, multiplied in, so nothing reading it branches: a
  // load under a runtime `if` cannot be hoisted past the branch, and loads
  // meant to be in flight together then wait one at a time). WHETHER THIS
  // THREAD HOLDS ANY OF THE TILE: a layout may cover fewer threads than the
  // block (a reduction's one row, the first THREADS_N's); the others store
  // nothing. A compare only for a layout narrower than the block, and loads do
  // not branch on it either way.
  DINLINE bool participates() const {
    if constexpr (kThreads == THREADS_PER_BLOCK)
      return true;
    else
      return static_cast<int>(threadIdx.x) < kThreads;
  }
  DINLINE int lane() const { return static_cast<int>(threadIdx.x) % THREADS_N; }
  // THIS THREAD'S RANK COPY, r of R: its group of THREADS_M x THREADS_N
  // threads.
  DINLINE int slice() const {
    if constexpr (THREADS_R == 1)
      return 0;
    else
      return static_cast<int>(threadIdx.x) / kSliceThreads;
  }
  DINLINE int col(int k) const {
    const int n = offs_n / kPack + lane() + k * THREADS_N;
    return n < N / kPack ? n : N / kPack - 1;
  }
  DINLINE float mask(int k) const {
    return offs_n / kPack + lane() + k * THREADS_N < N / kPack ? 1.0f : 0.0f;
  }
  // This thread's row m, a row past M reading row M - 1 (and storing nothing).
  DINLINE int tile_row(int m) const {
    // One row of threads: the row is the block's, so its address stays scalar.
    if constexpr (THREADS_M == 1)
      return m;
    else if constexpr (THREADS_R == 1)
      return static_cast<int>(threadIdx.x) / THREADS_N + m * THREADS_M;
    else
      return static_cast<int>(threadIdx.x) % kSliceThreads / THREADS_N +
             m * THREADS_M;
  }
  DINLINE bool live(int m) const { return offs_m + tile_row(m) * row_step < M; }
  DINLINE int row(int m) const {
    return live(m) ? offs_m + tile_row(m) * row_step : M - 1;
  }

  // The same place held as AS_DTYPE, its elements converted (rounded once),
  // unset, or zero.
  template <typename AS_DTYPE>
  using As = Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N,
                  THREADS_PER_BLOCK, AS_DTYPE, TILE_R, THREADS_R>;
  template <typename AS_DTYPE>
  DINLINE As<AS_DTYPE> like() const {
    return As<AS_DTYPE>(M, N, offs_m, offs_n, row_step);
  }
  template <typename AS_DTYPE>
  DINLINE As<AS_DTYPE> zeros() const {
    As<AS_DTYPE> t = like<AS_DTYPE>();
#pragma unroll
    for (int m = 0; m < kRows; ++m)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < kPack; ++j) t.v[m][k][j] = AS_DTYPE(0);
    return t;
  }
  template <typename AS_DTYPE>
  DINLINE As<AS_DTYPE> to() const {
    As<AS_DTYPE> t = like<AS_DTYPE>();
#pragma unroll
    for (int m = 0; m < kRows; ++m)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < kPack; ++j)
          t.v[m][k][j] = static_cast<AS_DTYPE>(v[m][k][j]);
    return t;
  }
};

// The tile TILE held as AS_DTYPE (a float tile for the math, a weight's own
// dtype), and whether a type is a tile.
template <typename TILE, typename AS_DTYPE>
using TileAs = typename TILE::template As<AS_DTYPE>;
template <typename X>
struct is_tile : std::false_type {};
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N,
          int THREADS_PER_BLOCK, typename ACC_DTYPE, int TILE_R, int THREADS_R>
struct is_tile<Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N,
                    THREADS_PER_BLOCK, ACC_DTYPE, TILE_R, THREADS_R>>
    : std::true_type {};

// THIS BLOCK'S TILES, grid-strided in row-major order over `tiles_m` x
// `tiles_n` tiles: `body(row, col)` for each, in tiles. The row and column are
// kept as the walk advances, each step an add, a compare and a carry, where
// dividing them out of the work item's index cost a 32-bit division a tile (ISA
// 2026-10-11, the plain all-reduce against the flat walk); the division is
// once, here.
namespace impl {
template <typename BODY>
DINLINE void block_for_each_tile(int tiles_m, int tiles_n, BODY body) {
  const int step_rows = static_cast<int>(gridDim.x) / tiles_n;
  const int step_cols = static_cast<int>(gridDim.x) % tiles_n;
  int row = static_cast<int>(blockIdx.x) / tiles_n;
  int col = static_cast<int>(blockIdx.x) % tiles_n;
  while (row < tiles_m) {
    body(row, col);
    row += step_rows;
    col += step_cols;
    if (col >= tiles_n) {
      col -= tiles_n;
      ++row;
    }
  }
}
}  // namespace impl

// =================================================================================================
// MEMORY
//
// THE READS AND WRITES, ON TILES: a tile's load and store (one round trip a
// tile), every rank's tile, a tile whose rows are tensors, and a row's scalar.
// How a group is loaded (one 16-byte global instruction, issued together) is
// impl's.
// =================================================================================================

// PEER MEMORY THROUGH GLOBAL INSTRUCTIONS. A pointer read out of a struct has
// no address space the compiler can prove, so it emits `flat_load`, which
// checks the aperture and waits on both counters; casting to address space 1
// gives `global_load_dwordx4` / `global_store_dwordx4`.
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(1))) u32x4 global_u32x4;

// ISSUED HERE, NOT WHERE THE COMPILER LIKES: no memory operation moves across
// this point (the empty asm's memory clobber) and no instruction is scheduled
// across it (sched_barrier), so every load above it is in flight before
// anything below it runs. The scheduler alone sank a reduce's peer loads past
// the adds of the first ones (ISA 2026-10-01T00-39-51Z); with sched_barrier
// alone, a tile's weight loads were sunk to their uses, one register quad and
// one round trip each (ISA 2026-10-03T21-00-27Z).
namespace impl {
template <typename PACK>
DINLINE PACK global_load(const PACK* p) {
  static_assert(sizeof(PACK) == 16, "a pack is 16 bytes");
  const u32x4 raw = *(const global_u32x4*)(p);
  PACK v;
  __builtin_memcpy(&v, &raw, 16);
  return v;
}

template <typename PACK>
DINLINE void global_store(PACK* p, const PACK& v) {
  static_assert(sizeof(PACK) == 16, "a pack is 16 bytes");
  u32x4 raw;
  __builtin_memcpy(&raw, &v, 16);
  *(global_u32x4*)(p) = raw;
}

DINLINE void issued() {
  asm volatile("" ::: "memory");
  __builtin_amdgcn_sched_barrier(0);
}

// A ROW'S PACK k AS A UNIFORM BASE PLUS AN UNSIGNED 32-BIT BYTE OFFSET: the
// form global_load and global_store take with the base in scalar registers
// (saddr), so every tensor sharing the columns shares one offset register. A
// signed column added to the pointer was a 64-bit vector address a tensor a
// pack, 28 VGPRs held across AttnRes's loops (liveness, 2026-10-04T15-28-51Z).
// THE ADDRESS IS BUILT AT ITS LOAD: the offset made opaque here, so the 64-bit
// sum is not hoisted and held a tensor a pack (an add a load; AttnRes 90 -> 70
// VGPRs, 2026-10-04 compile).
template <typename PACK, typename TILE>
DINLINE PACK* at_col(PACK* row, const TILE& t, int k) {
  using Byte = std::conditional_t<std::is_const_v<PACK>, const char, char>;
  uint32_t offset = static_cast<uint32_t>(t.col(k)) * uint32_t{sizeof(PACK)};
  asm volatile("" : "+v"(offset));
  return reinterpret_cast<PACK*>(reinterpret_cast<Byte*>(row) + offset);
}
template <typename PACK>
DINLINE PACK pack_load(const PACK* p) {
  if constexpr (sizeof(PACK) == 16)
    return global_load(p);
  else
    return *p;
}
// WHETHER A LOAD SKIPS ROW m: only a tile of several rows a thread or of rows
// (a reduce-scatter's at decode), whose rows past M would re-read the last over
// the links. A one-row tile's row is real wherever a kernel makes one, and a
// branch around its loads cost registers: the zeros the tile starts with live
// beside the loads (126 -> 178 VGPRs, spilling at 16384: 23-46-07Z).
template <typename TILE>
DINLINE bool skipped(const TILE& t, int m) {
  if constexpr (TILE::kRows == 1 && TILE::kThreadsM == 1) return false;
  return !t.live(m);
}
// A loaded group into the tile's elements, and a held group out as one load or
// store's worth.
template <typename TILE>
DINLINE void held(TILE& t, int m, int k, const typename TILE::Pack& p) {
  __builtin_memcpy(t.v[m][k], &p, sizeof(p));
}
template <typename TILE>
DINLINE typename TILE::Pack pack(const TILE& t, int m, int k) {
  typename TILE::Pack p;
  __builtin_memcpy(&p, t.v[m][k], sizeof(p));
  return p;
}
}  // namespace impl

// A LOAD NEVER BRANCHES: a thread outside a narrow tile loads its clamped,
// in-bounds place and its values go unused. Under participates() the load was
// an exec branch, so every tile's old registers stayed live across AttnRes's
// loops (threadIdx < 256 is not folded from the launch bounds; ISA
// 2026-10-04T16-39-15Z).
namespace impl {
template <typename TILE>
DINLINE void tile_load(TILE& t, const typename TILE::Acc* data,
                       int64_t row_stride) {
  using P = typename TILE::Pack;
  const P* at = reinterpret_cast<const P*>(data);
#pragma unroll
  for (int m = 0; m < TILE::kRows; ++m) {
    if (impl::skipped(t, m)) continue;
    const P* row = at + int64_t{t.row(m)} * (row_stride / t.kPack);
#pragma unroll
    for (int k = 0; k < t.K; ++k)
      impl::held(t, m, k, impl::pack_load(impl::at_col(row, t, k)));
  }
  impl::issued();
}

template <typename TILE>
DINLINE void tile_store(typename TILE::Acc* data, int64_t row_stride,
                        const TILE& t) {
  using P = typename TILE::Pack;
  if (!t.participates()) return;
  P* at = reinterpret_cast<P*>(data);
#pragma unroll
  for (int m = 0; m < TILE::kRows; ++m) {
    if (!t.live(m)) continue;
    P* row = at + int64_t{t.row(m)} * (row_stride / t.kPack);
#pragma unroll
    for (int k = 0; k < t.K; ++k) {
      if (t.mask(k) == 0.0f) continue;
      if constexpr (sizeof(P) == 16)
        impl::global_store(impl::at_col(row, t, k), impl::pack(t, m, k));
      else
        *impl::at_col(row, t, k) = impl::pack(t, m, k);
    }
  }
}
}  // namespace impl

// EVERY PEER'S TILE, all in flight together: `data(r)` is rank r's tensor,
// `row_stride` its elements between rows. A pack position at a time, every
// rank's for it, so the address is computed once a position (the other way
// round cost 16 scalar instructions at two packs: ISA 2026-10-01T00-31-14Z).
// Nothing waits until a tile is used (peers_reduce), so loads issued here can
// run under other work.
namespace impl {
template <typename TILE, int WORLD, typename RANK_DATA>
DINLINE void peers_load(TILE (&t)[WORLD], RANK_DATA data, int64_t row_stride) {
  using P = typename TILE::Pack;
#pragma unroll
  for (int m = 0; m < TILE::kRows; ++m) {
    if (impl::skipped(t[0], m)) continue;
#pragma unroll
    for (int k = 0; k < t[0].K; ++k) {
      // THE POSITION ONCE, IN A VECTOR REGISTER: left to the compiler, the
      // row's uniform part was added to every rank's base in scalar registers,
      // an add and a carry a rank (16 instructions a tile at eight ranks, the
      // plain all-reduce against the flat walk: ISA 2026-10-11); held in a
      // VGPR, each rank's address is one shift-and-add of it.
      int64_t i =
          int64_t{t[0].row(m)} * (row_stride / t[0].kPack) + t[0].col(k);
      asm volatile("" : "+v"(i));
#pragma unroll
      for (int r = 0; r < WORLD; ++r)
        impl::held(t[r], m, k,
                   impl::pack_load(reinterpret_cast<const P*>(data(r)) + i));
    }
  }
  impl::issued();
}
}  // namespace impl

// A TILE WHOSE COLUMNS ARE SPLIT AMONG THE RANKS, `slice` columns each (the
// last rank's to the end): each pack from its owner's tensor, `data(r)`, `r`
// the owner in each lane (a wave's packs may have two owners).
namespace impl {
template <int WORLD, typename TILE, typename RANK_DATA>
DINLINE void sliced_load(TILE& t, RANK_DATA data, int64_t row_stride,
                         int slice) {
  using P = typename TILE::Pack;
#pragma unroll
  for (int m = 0; m < TILE::kRows; ++m) {
    if (impl::skipped(t, m)) continue;
#pragma unroll
    for (int k = 0; k < t.K; ++k) {
      const int owner = min(t.col(k) / (slice / t.kPack), WORLD - 1);
      impl::held(t, m, k,
                 impl::pack_load(reinterpret_cast<const P*>(data(owner)) +
                                 int64_t{t.row(m)} * (row_stride / t.kPack) +
                                 t.col(k)));
    }
  }
  impl::issued();
}
}  // namespace impl

// A TILE WHOSE ROWS ARE DIFFERENT TENSORS' (a row a peer): row m's at
// `row_data(m)`, its first element, the tile's columns counted from it.
// tile_gather loads every row together, one round trip (the fused AttnRes
// pull's reduce-scatter).
namespace impl {
template <typename TILE, typename ROW_DATA>
DINLINE void tile_gather(TILE& t, ROW_DATA row_data) {
  using P = typename TILE::Pack;
  if (!t.participates()) return;
#pragma unroll
  for (int m = 0; m < TILE::kRows; ++m) {
    const P* row = reinterpret_cast<const P*>(row_data(t.tile_row(m)));
#pragma unroll
    for (int k = 0; k < t.K; ++k)
      impl::held(t, m, k, impl::pack_load(impl::at_col(row, t, k)));
  }
  impl::issued();
}
}  // namespace impl

// A ROW'S SCALAR (a norm's scale), one float a row of `scalars`: thread 0
// stores row `row`'s; every rank's for row `row`, `data(r)` its scalars, loaded
// by every thread, all in flight together (a wave's lanes read one address: one
// request a wave), issued after any tile loads before them.
namespace impl {
DINLINE void block_store_row_scalar(float* scalars, int row, float v) {
  if (threadIdx.x == 0)
    *(__attribute__((address_space(1))) float*)(scalars + row) = v;
}

template <int WORLD, typename RANK_DATA>
DINLINE void peers_load_row_scalars(RANK_DATA data, int row,
                                    float (&output)[WORLD]) {
#pragma unroll
  for (int r = 0; r < WORLD; ++r)
    output[r] =
        *(const __attribute__((address_space(1))) float*)(data(r) + row);
  impl::issued();
}
}  // namespace impl

// =================================================================================================
// ELEMENTWISE
//
// THE ELEMENTWISE OPS, at thread scope: TILE MATH, on float tiles (a fused op
// spells each rounding its reference does as a to<>() between them).
// =================================================================================================

// A FLOAT TILE'S MATH, element by element: tile_add and tile_mul with a tile of
// its shape or a one-row tile (a weight, every row's), and tile_mul by one
// scalar or a scalar a row (a row's norm scale).
namespace impl {
template <typename A, typename B, typename OP>
DINLINE A zip(const A& a, const B& b, OP op) {
  static_assert(std::is_same_v<typename A::Acc, float> &&
                    std::is_same_v<typename B::Acc, float>,
                "tile math is on float tiles; to<float>() first");
  static_assert(B::kRows == A::kRows || B::kRows == 1,
                "b is a's shape or one row");
  static_assert(B::K == A::K && B::kPack == A::kPack, "b holds a's columns");
  A output = a;
#pragma unroll
  for (int m = 0; m < A::kRows; ++m)
#pragma unroll
    for (int k = 0; k < A::K; ++k)
#pragma unroll
      for (int j = 0; j < A::kPack; ++j)
        output.v[m][k][j] = op(a.v[m][k][j], b.v[B::kRows == 1 ? 0 : m][k][j]);
  return output;
}

template <typename A, typename B, std::enable_if_t<is_tile<B>::value, int> = 0>
DINLINE A tile_add(const A& a, const B& b) {
  return impl::zip(a, b, [](float x, float y) { return x + y; });
}

template <typename A, typename B, std::enable_if_t<is_tile<B>::value, int> = 0>
DINLINE A tile_mul(const A& a, const B& b) {
  return impl::zip(a, b, [](float x, float y) { return x * y; });
}

template <typename A>
DINLINE A tile_mul(const A& a, const float (&row_scale)[A::kRows]) {
  static_assert(std::is_same_v<typename A::Acc, float>,
                "tile math is on float tiles");
  A output = a;
#pragma unroll
  for (int m = 0; m < A::kRows; ++m)
#pragma unroll
    for (int k = 0; k < A::K; ++k)
#pragma unroll
      for (int j = 0; j < A::kPack; ++j)
        output.v[m][k][j] = a.v[m][k][j] * row_scale[m];
  return output;
}

template <typename A>
DINLINE A tile_mul(const A& a, float scale) {
  float row_scale[A::kRows];
#pragma unroll
  for (int m = 0; m < A::kRows; ++m) row_scale[m] = scale;
  return tile_mul(a, row_scale);
}
}  // namespace impl

// a x a_row_scale + b x b_row_scale, element by element, as fused multiply-adds
// (a softmax's fold: the old sum rescaled plus a new source weighted). b may be
// held in its stored dtype, converted here. Two multiplies and an add were
// three VALU an element and, vectorized, packed multiplies with a chain of adds
// (ISA 2026-10-04T02-22-41Z).
namespace impl {
template <typename A, typename B>
DINLINE A tile_fma(const A& a, const float (&a_row_scale)[A::kRows], const B& b,
                   const float (&b_row_scale)[A::kRows]) {
  static_assert(std::is_same_v<typename A::Acc, float>,
                "the sum is a float tile");
  A output = a;
#pragma unroll
  for (int m = 0; m < A::kRows; ++m)
#pragma unroll
    for (int k = 0; k < A::K; ++k)
#pragma unroll
      for (int j = 0; j < A::kPack; ++j)
        output.v[m][k][j] = fmaf(static_cast<float>(b.v[m][k][j]),
                                 b_row_scale[m], a.v[m][k][j] * a_row_scale[m]);
  return output;
}
}  // namespace impl

// =================================================================================================
// REDUCE
//
// THE REDUCTIONS, by scope, each built on the one below: over the PEERS (a pack
// or a tile row summed over every rank, fp32, rounded once, in rank order),
// over a WAVE (shuffles), over a BLOCK (a wave's result, then one LDS pass). N
// values at once at every scope: each block pass is two barriers and an LDS
// round trip, so a kernel with several reductions takes them together.
// =================================================================================================

// EACH PACK SUMMED OVER ITS `ngpus` SOURCES, in fp32 in source order and
// rounded once to T, into sum[k]: callers whose sources are the ranks agree
// bitwise. Waits on the loads only here. EVERY PEER'S TILE SUMMED, in rank
// order in fp32 and rounded once, as a pack's is.
namespace impl {
template <typename TILE, int WORLD>
DINLINE TILE peers_reduce(const TILE (&t)[WORLD]) {
  constexpr int NL = TILE::kPack;
  TILE sum = t[0].template like<typename TILE::Acc>();
#pragma unroll
  for (int m = 0; m < TILE::kRows; ++m)
#pragma unroll
    for (int k = 0; k < sum.K; ++k) {
      float acc[NL];
#pragma unroll
      for (int j = 0; j < NL; ++j) acc[j] = static_cast<float>(t[0].v[m][k][j]);
#pragma unroll
      for (int r = 1; r < WORLD; ++r)
#pragma unroll
        for (int j = 0; j < NL; ++j)
          acc[j] += static_cast<float>(t[r].v[m][k][j]);
#pragma unroll
      for (int j = 0; j < NL; ++j)
        sum.v[m][k][j] = static_cast<typename TILE::Acc>(acc[j]);
    }
  return sum;
}
}  // namespace impl

// THE OPERATIONS a wave or block reduction combines with.
struct Sum {
  static DINLINE float apply(float a, float b) { return a + b; }
  static constexpr float kIdentity = 0.0f;
};
struct Max {
  static DINLINE float apply(float a, float b) { return fmaxf(a, b); }
  static constexpr float kIdentity = -INFINITY;
};

// ONE DPP STEP: `v` from the lane `ctrl` names, as a VALU operand, no LDS.
// `row_mask` picks which 16-lane rows take it; the others read `identity`,
// which leaves them as they were. A FULL ROW MASK NEEDS NO `old`: left
// undefined (mov_dpp), the compiler fuses the move into the add that uses it
// (v_add_f32_dpp); an explicit identity kept a v_mov_b32_dpp and an add a step
// (ISA 2026-10-04T03-07-41Z).
namespace impl {
template <int DPP_CTRL, int DPP_ROW_MASK, typename REDUCE_OP>
DINLINE float dpp(float v) {
  int moved;
  if constexpr (DPP_ROW_MASK == 0xf)
    moved = __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), DPP_CTRL, 0xf,
                                     0xf, false);
  else
    moved = __builtin_amdgcn_update_dpp(
        __builtin_bit_cast(int, REDUCE_OP::kIdentity),
        __builtin_bit_cast(int, v), DPP_CTRL, DPP_ROW_MASK, 0xf, false);
  return __builtin_bit_cast(float, moved);
}
}  // namespace impl

// N VALUES OVER THE WAVE, in place, every lane left holding the results. The
// moved value first: only a VOP2's src0 takes DPP, so (dpp(x), x) fuses each
// step into one v_add_f32_dpp where (x, dpp(x)) was a v_mov_b32_dpp and an add
// (ISA 2026-10-04T02-22-41Z). DPP, as rocPRIM's and Composable Kernel's wave
// reductions do: each step is a VALU operand from another lane. The butterfly
// __shfl_xor it replaced compiled to ds_bpermute, an LDS round trip a step, six
// of them waiting on each other (ISA 2026-09-30T20-43-18Z; stamps: 0.92 us for
// one block_reduce).
//   swap neighbours, swap pairs (quad_perm [1,0,3,2], [2,3,0,1]): each quad
//   holds its sum mirror in 8 lanes, then in 16 (row_half_mirror, row_mirror):
//   each row holds its sum lane 15 into rows 1 and 3, lane 31 into rows 2 and 3
//   (row_bcast15, row_bcast31): lane 63 holds the wave's, and readlane hands it
//   to every lane.
namespace impl {
template <typename REDUCE_OP, int NUM_VALUES>
DINLINE void wave_reduce(float (&v)[NUM_VALUES]) {
  static_assert(kWaveSize == 64, "the DPP sequence is for 64-lane waves");
#pragma unroll
  for (int n = 0; n < NUM_VALUES; ++n) {
    float x = v[n];
    x = REDUCE_OP::apply(dpp<0xb1, 0xf, REDUCE_OP>(x),
                         x);  // quad_perm [1,0,3,2]
    x = REDUCE_OP::apply(dpp<0x4e, 0xf, REDUCE_OP>(x),
                         x);  // quad_perm [2,3,0,1]
    x = REDUCE_OP::apply(dpp<0x141, 0xf, REDUCE_OP>(x), x);  // row_half_mirror
    x = REDUCE_OP::apply(dpp<0x140, 0xf, REDUCE_OP>(x), x);  // row_mirror
    x = REDUCE_OP::apply(dpp<0x142, 0xa, REDUCE_OP>(x),
                         x);  // row_bcast15 into rows 1, 3
    x = REDUCE_OP::apply(dpp<0x143, 0xc, REDUCE_OP>(x),
                         x);  // row_bcast31 into rows 2, 3
    v[n] = __builtin_bit_cast(
        float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, x), 63));
  }
}
}  // namespace impl

// N VALUES OVER THE BLOCK, in place: each wave reduces its N, then every thread
// sums the waves' partials in wave order (the same order in every thread, so
// every thread holds the same totals). One wave reducing the partials over all
// 64 lanes, then broadcasting them, was a second DPP pass a value, a second
// barrier and an LDS round trip to add 4 numbers at 256 threads: half of
// AttnRes's loop (ISA 2026-10-04T03-31-28Z). The trailing barrier keeps the
// next call from overwriting partials still being read. THE BLOCK'S WAVES ARE
// COMPILED IN: read from blockDim, every thread loaded the widest block's
// partials and kept 12 VGPRs at AttnRes's peak (ISA 2026-10-04T16-39-15Z).
namespace impl {
template <typename REDUCE_OP, int THREADS_PER_BLOCK, int NUM_VALUES>
DINLINE void block_reduce(float (&v)[NUM_VALUES]) {
  constexpr int kWaves = (THREADS_PER_BLOCK + kWaveSize - 1) / kWaveSize;
  __shared__ float partial[NUM_VALUES][kWaves];
  const int lane = threadIdx.x % kWaveSize;
  const int wave = threadIdx.x / kWaveSize;
  wave_reduce<REDUCE_OP>(v);
  if (lane == 0) {
#pragma unroll
    for (int n = 0; n < NUM_VALUES; ++n) partial[n][wave] = v[n];
  }
  __syncthreads();
#pragma unroll
  for (int n = 0; n < NUM_VALUES; ++n) {
    float x = partial[n][0];
#pragma unroll
    for (int w = 1; w < kWaves; ++w) x = REDUCE_OP::apply(x, partial[n][w]);
    v[n] = x;
  }
  __syncthreads();
}
}  // namespace impl

// A TILE REDUCED OVER AN AXIS, Triton's tl.sum(x, axis) for a float tile:
//   Axis::m  over the rows, every column's: the rows meet in LDS as they are
//   held (a bf16 tile's
//            16 bytes a group, as the old two-shot's), reduced in fp32 in row
//            order, and the one-row fp32 result is the first THREADS_N threads'
//            (a Tile of one row of threads)
//   Axis::n  over the columns, every row's, into `out` (a row's value in every
//   thread of the row):
//            each thread's groups, then the row's threads (a wave's or the
//            block's)
//   Axis::r  over a 3-D tile's ranks, every element's: each rank's copy meets
//   in LDS as it is held,
//            reduced in fp32 in rank order, and the TILE_M x TILE_N fp32 result
//            is the first rank's threads' (a 2-D tile of THREADS_M x THREADS_N
//            threads)
enum class Axis { m, n, r };

namespace impl {
// OVER THE RANKS: every thread's packs into LDS at its rank's place, a pack a
// thread side by side (consecutive threads' 16 bytes adjacent), then the first
// rank's threads sum each pack over the ranks. LDS: TILE_R x TILE_M x TILE_N
// elements as held.
template <typename REDUCE_OP, typename TILE>
DINLINE Tile<typename TILE::Dtype, TILE::kTileM, TILE::kTileN, TILE::kThreadsM,
             TILE::kThreadsN, TILE::kThreadsPerBlock, float>
block_reduce_ranks(const TILE& t) {
  static_assert(TILE::kThreadsR == TILE::kTileR,
                "a thread holds one rank's copy");
  using Acc = typename TILE::Acc;
  using P = typename TILE::Pack;
  constexpr int kPacks = TILE::kRows * TILE::K;  // a thread's
  __shared__ P part[TILE::kTileR][kPacks][TILE::kSliceThreads];
  Tile<typename TILE::Dtype, TILE::kTileM, TILE::kTileN, TILE::kThreadsM,
       TILE::kThreadsN, TILE::kThreadsPerBlock, float>
      output{t.M, t.N, t.offs_m, t.offs_n, t.row_step};
  const int mine = static_cast<int>(threadIdx.x) % TILE::kSliceThreads;
#pragma unroll
  for (int m = 0; m < TILE::kRows; ++m)
#pragma unroll
    for (int k = 0; k < TILE::K; ++k)
      part[t.slice()][m * TILE::K + k][mine] = pack(t, m, k);
  __syncthreads();
  if (output.participates()) {
#pragma unroll
    for (int m = 0; m < TILE::kRows; ++m)
#pragma unroll
      for (int k = 0; k < TILE::K; ++k) {
        float x[TILE::kPack];
        P p = part[0][m * TILE::K + k][mine];
#pragma unroll
        for (int j = 0; j < TILE::kPack; ++j)
          x[j] = static_cast<float>(reinterpret_cast<const Acc*>(&p)[j]);
#pragma unroll
        for (int r = 1; r < TILE::kTileR; ++r) {
          p = part[r][m * TILE::K + k][mine];
#pragma unroll
          for (int j = 0; j < TILE::kPack; ++j)
            x[j] = REDUCE_OP::apply(
                x[j], static_cast<float>(reinterpret_cast<const Acc*>(&p)[j]));
        }
#pragma unroll
        for (int j = 0; j < TILE::kPack; ++j) output.v[m][k][j] = x[j];
      }
  }
  __syncthreads();  // before the next reduction writes `part`
  return output;
}

template <typename REDUCE_OP, Axis AXIS, typename TILE>
DINLINE Tile<typename TILE::Dtype, 1, TILE::kTileN, 1, TILE::kThreadsN,
             TILE::kThreadsPerBlock, float>
block_reduce_rows(const TILE& t) {
  static_assert(AXIS == Axis::m, "a row's reduction (Axis::n) writes `out`");
  static_assert(TILE::kRows == 1, "a thread holds one row of the rows reduced");
  constexpr int kRowLanes = TILE::kThreadsM;
  using Acc = typename TILE::Acc;
  __shared__ Acc part[kRowLanes][TILE::kTileN];
  Tile<typename TILE::Dtype, 1, TILE::kTileN, 1, TILE::kThreadsN,
       TILE::kThreadsPerBlock, float>
      output{1, t.N, 0, t.offs_n};
  const int lane_row = static_cast<int>(threadIdx.x) / TILE::kThreadsN;
#pragma unroll
  for (int k = 0; k < TILE::K; ++k)
#pragma unroll
    for (int j = 0; j < TILE::kPack; ++j) {
      part[lane_row][(t.lane() + k * TILE::kThreadsN) * TILE::kPack + j] =
          t.v[0][k][j];
    }
  __syncthreads();
  if (output.participates()) {
#pragma unroll
    for (int k = 0; k < TILE::K; ++k)
#pragma unroll
      for (int j = 0; j < TILE::kPack; ++j) {
        const int c = (output.lane() + k * TILE::kThreadsN) * TILE::kPack + j;
        float x = static_cast<float>(part[0][c]);
#pragma unroll
        for (int r = 1; r < kRowLanes; ++r)
          x = REDUCE_OP::apply(x, static_cast<float>(part[r][c]));
        output.v[0][k][j] = x;
      }
  }
  __syncthreads();  // before the next reduction writes `part`
  return output;
}

// A TILE OVER AN AXIS: the ranks' (a 3-D tile's) or the rows' result tile.
template <typename REDUCE_OP, Axis AXIS, typename TILE>
DINLINE auto block_reduce(const TILE& t) {
  if constexpr (AXIS == Axis::r)
    return block_reduce_ranks<REDUCE_OP>(t);
  else
    return block_reduce_rows<REDUCE_OP, AXIS>(t);
}

template <typename REDUCE_OP, Axis AXIS, typename TILE>
DINLINE void block_reduce(const TILE& t, float (&output)[TILE::kRows]) {
  static_assert(AXIS == Axis::n,
                "a column's reduction (Axis::m) returns a tile");
  static_assert(std::is_same_v<typename TILE::Acc, float>,
                "a reduction is of a float tile");
#pragma unroll
  for (int m = 0; m < TILE::kRows; ++m) {
    float x = REDUCE_OP::kIdentity;
#pragma unroll
    for (int k = 0; k < TILE::K; ++k)
#pragma unroll
      for (int j = 0; j < TILE::kPack; ++j)
        x = t.mask(k) != 0.0f ? REDUCE_OP::apply(x, t.v[m][k][j]) : x;
    output[m] = x;
  }
  if constexpr (TILE::kThreadsM == 1)
    block_reduce<REDUCE_OP, TILE::kThreads>(output);
  else if constexpr (TILE::kThreadsN == kWaveSize)
    wave_reduce<REDUCE_OP>(output);
  else
    static_assert(TILE::kThreadsN == kWaveSize,
                  "a row is the block's or a wave's");
}
}  // namespace impl

// The LDS a block_reduce<Op, N> takes, for a kernel budgeting the rest.
constexpr int64_t block_reduce_lds_bytes(int n) {
  return int64_t{kBuild.kernels.max_waves} * n * sizeof(float);
}

// =================================================================================================
// DOT
//
// THE DOT: a thread's share of a row's dot (a block_reduce finishes it).
// =================================================================================================

// This thread's share of dot(a, b) over each row of the tile, columns past N
// counting zero: the partials a block_reduce turns into the rows' dots (a sum
// of squares is partial_dot(x, x)). A one-row b (a weight) is every row's.
namespace impl {
template <typename A, typename B>
DINLINE void partial_dot(const A& a, const B& b, float (&d)[A::kRows]) {
  static_assert(
      A::kThreadsM == 1,
      "a dot's partials reduce over the block, so a row is the block's");
  static_assert(B::kRows == A::kRows || B::kRows == 1,
                "b is a's shape or one row");
#pragma unroll
  for (int m = 0; m < A::kRows; ++m) {
    const int mb = B::kRows == 1 ? 0 : m;
    d[m] = 0.0f;
#pragma unroll
    for (int k = 0; k < A::K; ++k) {
      float dk = 0.0f;
#pragma unroll
      for (int j = 0; j < A::kPack; ++j)
        dk = fmaf(static_cast<float>(a.v[m][k][j]),
                  static_cast<float>(b.v[mb][k][j]), dk);
      d[m] += a.mask(k) * dk;
    }
  }
}
}  // namespace impl

// =================================================================================================
// SOFTMAX
//
// THE ONLINE SOFTMAX, at thread scope: a softmax over logits that arrive a tile
// at a time, folded into a running weighted sum without a second pass (flash
// attention's).
// =================================================================================================

// The softmax so far: the largest logit seen and the sum of exp(logit - max).
struct OnlineSoftmax {
  float max = -INFINITY;
  float denominator = 0.0f;
};

// Folds N logits into `s` (a -INFINITY logit is no source). Returns the scale
// the weighted sum so far takes, and in `scale` each logit's weight; the sum is
// final once divided by s.denominator.
namespace impl {
template <int NUM_LOGITS>
DINLINE float thread_softmax_fold(OnlineSoftmax& s,
                                  const float (&logit)[NUM_LOGITS],
                                  float (&scale)[NUM_LOGITS]) {
  float new_max = s.max;
#pragma unroll
  for (int t = 0; t < NUM_LOGITS; ++t) new_max = fmaxf(new_max, logit[t]);
  const float old_scale = __expf(s.max - new_max);
  s.denominator *= old_scale;
#pragma unroll
  for (int t = 0; t < NUM_LOGITS; ++t) {
    scale[t] = __expf(logit[t] - new_max);
    s.denominator += scale[t];
  }
  s.max = new_max;
  return old_scale;
}
}  // namespace impl

// =================================================================================================
// PEERS
//
// THE RANKS' MEMORY: every rank's buffers as plain tensors a tile loads and
// stores, and the types a kernel takes them in. A RANK'S MEMORY, two places:
//   ours, two allocations (Handle's symmetric memory), mapped by every peer at
//   startup:
//     [ signals ]  and  [ scratch | staging ]
//   each handed to a kernel as its own argument, a device table of every rank's
//   address (its signals, scratch and staging) and the buffer's strides,
//   written once when the handle opens; the caller's, one tensor a call:
//     [ input ]   read in place only when registered or captured, its peers'
//     addresses a device
//                 table (a captured launch's are filled after the capture);
//                 otherwise a staged kernel copies it into the staging
// A buffer is read or written by tile_load / tile_store / peers_load, ordered
// only by BARRIER's `barrier`. The signals are BARRIER's, never read as
// data.
// =================================================================================================

constexpr int kMaxRanks = kMaxPeers;
// ONE SIGNAL SLOT PER BLOCK THE HARDWARE CAN HOLD RESIDENT, on any target
// built: the signal block never rules out a grid. Which grid is fast is
// select's (select.cuh).
constexpr int kMaxBlocks = kMaxResidentBlocks;

// WHERE A WAIT WAS when it gave up, the fault record's `where`; Python's
// `Where`, number for number. 0 is no fault.
enum class Where : uint32_t {
  start = 1,     // a launch's first barrier: the peer had not launched
  peer_barrier,  // a later barrier
};
constexpr int kWhereCount = 2;  // numbered from 1
constexpr const char* to_string(Where w) {
  switch (w) {
    case Where::start:
      return "start";
    case Where::peer_barrier:
      return "peer_barrier";
  }
  return "?";
}

// A RANK'S STATUS, in HOST-MAPPED memory (the handle's), so the host reads it
// and sets `abort` while a kernel still runs, and a graph replay, which runs
// none of our host code, still reports: a buffer of 32-bit words, one a field,
// `status[field]` at its stride (`status_stride_field`), the field a
// StatusField. 32-BIT, as the record's fields always were: 64-bit words cost
// the fault path a register pair a field and an occupancy step in 157 of 440
// kernels (2026-10-13T16-59-48Z). A wait that outlives the timeout writes the
// fault once (the first block to claim `state` wins) and sets `abort`; every
// wait checks `abort` with its clock and leaves, so one fault ends the rank's
// kernel instead of each block spinning out its own timeout. The host sets
// `abort` to end a kernel it has given up on.
// A FLAG'S STATES, the only values a barrier writes: `arrived` into a peer's
// slot for it, `clear` into its own once it has seen it. Between launches every
// flag is `clear`; any other value is not a state the protocol has, so the
// memory is corrupt.
enum class FlagState : uint32_t {
  clear =
      0,  // the writer has not reached this barrier, or the owner has seen it
  arrived =
      1,  // the writer reached this barrier; the owner has not yet seen it
};

// THE FAULT'S STATES, the status's `state` word: claimed by the first wait to
// give up, then written whole; the host reads the fault's fields only once
// `written`.
enum class FaultState : uint32_t {
  none = 0,
  writing = 1,
  written = 2,
};

enum class StatusField : int {
  abort,        // 1: every wait on this rank leaves; 0: they wait
  state,        // a FaultState
  where,        // Where
  rank,         // this rank
  block,        // the block that gave up
  peer,         // the rank it waited for
  count,        // the flag it read last
  want,         // the value it waited for
  elapsed_low,  // how long it waited, in the device wall clock's ticks: the low
                // word
  elapsed_high,  // and the high
};
constexpr int kStatusFields = 10;

// RANK r'S BUFFER, its first element, which rank_ptr and rank_ptrs
// make a Ptr of: a kernel's arguments carry every rank's address of a buffer (a
// device table, `ptrs`) and this reads one. `r` IS THE SAME ACROSS THE WAVE (a
// constant, or a per-wave rank): read from the first lane, the compiler knows
// it, and the load is scalar rather than one per lane.
namespace impl {
template <typename T>
DINLINE T* rank_of(T* const* ptrs, int r) {
  return ptrs[__builtin_amdgcn_readfirstlane(r)];
}
}  // namespace impl

// =================================================================================================
// PTR
//
// THE POINTER, beside the tile (TILE) the other primitive every kernel is
// written in: what a kernel touches, local or another rank's, as a Ptr -- its
// data, its layout (both strides, in elements), and whose memory it is -- and
// the boundary that makes them, a kernel's first lines.
// =================================================================================================

// A POINTER A KERNEL TOUCHES, local or another rank's: its data, its layout
// (the strides of its rows and columns, in elements, as the kernel's arguments
// gave them) and whose memory it is. A kernel turns its arguments into these in
// its first lines (`rank_ptrs`, `rank_ptr`, `local_ptr`) and every load and
// store after takes one. The packed loads need stride_n 1; the host checks it.
template <typename T>
struct Ptr {
  T* data;
  int64_t stride_m;
  int64_t stride_n;
  int rank;
};

namespace impl {

// EVERY RANK'S BUFFER AS A Ptr, rank r's data at r (rank r the same across the
// wave, as rank_of makes it), each with the buffer's row stride.
template <typename T, int WORLD>
DINLINE std::array<Ptr<T>, WORLD> rank_ptrs(T* const* p, int64_t stride_m,
                                            int64_t stride_n) {
  std::array<Ptr<T>, WORLD> all;
#pragma unroll
  for (int r = 0; r < WORLD; ++r)
    all[r] = Ptr<T>{impl::rank_of(p, r), stride_m, stride_n, r};
  return all;
}
// ONE RANK'S BUFFER AS A Ptr (this rank's own, most often), by the same
// rank_of.
template <typename T, int WORLD>
DINLINE Ptr<T> rank_ptr(T* const* p, int rank, int64_t stride_m,
                        int64_t stride_n) {
  return Ptr<T>{impl::rank_of(p, rank), stride_m, stride_n, rank};
}
// ELEMENT e OF A BUFFER OF `n`-ELEMENT ROWS, COUNTED ROW-MAJOR (e = row * n +
// col), at the Ptr's strides: one multiply-add when the rows are back to back
// (stride_m == n * stride_n, a contiguous tensor), else e's row and column
// divided out. `e` a pack's first element and `n` whole packs (the host's
// row_not_packs), so a pack never spans two rows. The division in 32 bits below
// 2^31 elements (a few instructions, where a 64-bit one is a hundred).
template <typename T>
DINLINE T* flat_at(const Ptr<T>& p, int64_t e, int n) {
  if (p.stride_m == int64_t{n} * p.stride_n) return p.data + e * p.stride_n;
  const int64_t row =
      e < (int64_t{1} << 31)
          ? int64_t{static_cast<uint32_t>(e) / static_cast<uint32_t>(n)}
          : e / n;
  return p.data + row * p.stride_m + (e - row * n) * p.stride_n;
}
template <typename T>
DINLINE Ptr<T> local_ptr(T* data, int64_t stride_m, int64_t stride_n,
                         int rank) {
  return Ptr<T>{data, stride_m, stride_n, rank};
}

}  // namespace impl

// =================================================================================================
// BARRIER
//
// THE RANKS' SYNCHRONIZATION, over the SIGNALS buffer, [R, 2, B, R]:
// `signals[owner][parity][block] [writer]` is the flag block `block` on rank
// `writer` sets in rank `owner`'s memory, at a barrier of that parity. The
// outer R is a table of every rank's address; the inner dimensions are the
// buffer's strides, passed like any other buffer's. Each flag is touched only
// atomically (`Counter`); `barrier`, and in `impl` what it is made of. A rank's
// data buffers are PEERS's.
//
// NOTHING SURVIVES A LAUNCH: every flag is 0 between launches. A barrier sets 1
// in each peer's slot for it, waits for 1 in each of its own, and sets those
// back to 0. A slot is never set twice before it is cleared, because the
// barriers of a launch alternate between the two flag sets (`parity`): a peer
// sets my parity-p slot again only at its barrier after next, which it passes
// only once I have signalled my next barrier, which I do only after my clear
// has landed. Every kernel has an even number of barriers
// (dev/lint_kernels.sh), so each launch starts at parity 0 on all-zero flags,
// and a graph replay is the same launch again.
//
// A WAIT NEVER HANGS AND NEVER TRAPS: past the timeout it writes the rank's
// fault record (where, which peer, the count against the one wanted) and leaves
// the kernel, and every wait leaves once the status's `abort` is set; the host
// reads the status (PEERS's `StatusField`). HIP_CUSTOM_COMMUNICATOR_DEBUG=1
// skews every wait by a random per-block delay, so a race shows on every run.
// =================================================================================================

// ONE FLAG: only atomic operations, each with its order and scope spelled at
// the call. A plain load or store of a flag cannot be written.
class Counter {
  uint32_t* at_;

 public:
  explicit DINLINE Counter(uint32_t* at) : at_(at) {}
  template <int MEMORY_ORDER, int MEMORY_SCOPE>
  DINLINE void store(uint32_t v) const {
    __scoped_atomic_store_n(at_, v, MEMORY_ORDER, MEMORY_SCOPE);
  }
  template <int MEMORY_ORDER, int MEMORY_SCOPE>
  DINLINE uint32_t load() const {
    return __scoped_atomic_load_n(at_, MEMORY_ORDER, MEMORY_SCOPE);
  }
};

#ifndef HIP_CUSTOM_COMMUNICATOR_DEBUG
  #define HIP_CUSTOM_COMMUNICATOR_DEBUG 0
#endif

namespace impl {

// Debug builds only: up to ~32 x 8K cycles, different per rank, block and
// barrier parity.
DINLINE void skew(int rank, uint32_t parity) {
  if (!HIP_CUSTOM_COMMUNICATOR_DEBUG) return;
  uint32_t h = static_cast<uint32_t>(rank) * 73856093u ^
               blockIdx.x * 19349663u ^ parity * 83492791u;
  h ^= h >> 13;
  h *= 0x5bd1e995u;
  for (uint32_t n = (h ^ (h >> 15)) % 32; n > 0; --n)
    __builtin_amdgcn_s_sleep(127);
}

// EVERY WAVE'S STORES DONE before a block barrier that one wave then releases:
// `__syncthreads` waits for none (gfx9 emits no vmcnt wait before s_barrier;
// seen in the ISA), so the releasing wave's fence would cover only its own
// stores, and another wave's could still be in flight when the flag lands -- a
// remote store above all.
DINLINE void wait_stores() { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); }

// The builtin takes the ordering as a literal, so each case is spelled out.
template <int MEMORY_ORDER, int MEMORY_SCOPE>
DINLINE void fence() {
  constexpr bool system = MEMORY_SCOPE == __MEMORY_SCOPE_SYSTEM;
  if constexpr (MEMORY_ORDER == __ATOMIC_RELEASE) {
    if constexpr (system)
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    else
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  } else {
    if constexpr (system)
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    else
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  }
}

// AN INDEX INSIDE ITS BUFFER'S SIZE, checked in debug builds only (a release
// build passes the sizes and reads none): one outside is a programming error,
// not a peer's fault, so it traps.
DINLINE void debug_in_bounds(int64_t index, int64_t size) {
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG && (index < 0 || index >= size))
    __builtin_trap();
}

// One word of this rank's status, by its field, at the buffer's stride.
DINLINE uint32_t* status_word(uint32_t* status_ptr, int64_t status_stride_field,
                              int status_size_field, StatusField f) {
  debug_in_bounds(static_cast<int64_t>(f), status_size_field);
  return status_ptr + static_cast<int64_t>(f) * status_stride_field;
}

// THE FAULT, written once by the first wait on this rank to give up: `state`
// claimed 0 -> 1, the fields, then 2 released, so the host reads them only
// whole. Then `abort`, so every other wait on this rank leaves at its next
// check rather than spinning out its own timeout.
DINLINE void record_fault(uint32_t* status_ptr, int64_t status_stride_field,
                          int status_size_field, Where where, int rank,
                          int peer, uint32_t count, uint32_t want,
                          uint64_t elapsed) {
  const auto word = [&](StatusField f) {
    return status_word(status_ptr, status_stride_field, status_size_field, f);
  };
  uint32_t none = static_cast<uint32_t>(FaultState::none);
  if (__scoped_atomic_compare_exchange_n(
          word(StatusField::state), &none,
          static_cast<uint32_t>(FaultState::writing), false, __ATOMIC_RELAXED,
          __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM)) {
    *word(StatusField::where) = static_cast<uint32_t>(where);
    *word(StatusField::rank) = static_cast<uint32_t>(rank);
    *word(StatusField::block) = blockIdx.x;
    *word(StatusField::peer) = static_cast<uint32_t>(peer);
    *word(StatusField::count) = count;
    *word(StatusField::want) = want;
    *word(StatusField::elapsed_low) = static_cast<uint32_t>(elapsed);
    *word(StatusField::elapsed_high) = static_cast<uint32_t>(elapsed >> 32);
    __scoped_atomic_store_n(word(StatusField::state),
                            static_cast<uint32_t>(FaultState::written),
                            __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
  }
  __scoped_atomic_store_n(word(StatusField::abort), 1u, __ATOMIC_RELAXED,
                          __MEMORY_SCOPE_SYSTEM);
}

// Spins relaxed and acquires once, after: an acquire per poll would invalidate
// the caches on every iteration of every spinning block. A sleep a poll (64
// cycles) keeps a spinning block from hammering the link. EVERY 256 POLLS, the
// slow path: the clock, and the status's `abort` (host-mapped, so only here,
// never in the fast path). Past the timeout the wait records the fault; timed
// out or aborted, the WAVE ENDS (s_endpgm): an ended wave no longer counts at
// the block's s_barrier, so the block's other waves run on to their own next
// wait, see `abort`, and end too. The kernel's output is then garbage and the
// communicator broken (the host's watchdog).
template <bool ACQUIRE, int MEMORY_SCOPE>
DINLINE void wait(uint64_t timeout_ticks, uint32_t* status_ptr,
                  int64_t status_stride_field, int status_size_field, int rank,
                  const Counter& flag, uint32_t want, Where where, int peer) {
  const uint64_t t0 = wall_clock64();
  for (uint32_t n = 1; flag.load<__ATOMIC_RELAXED, MEMORY_SCOPE>() < want;
       ++n) {
    __builtin_amdgcn_s_sleep(1);
    if ((n & 255u) != 0) continue;
    const uint64_t waited = wall_clock64() - t0;
    const bool aborted = __scoped_atomic_load_n(
                             status_word(status_ptr, status_stride_field,
                                         status_size_field, StatusField::abort),
                             __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM) != 0;
    if (!aborted && waited <= timeout_ticks) continue;
    if (!aborted)
      record_fault(status_ptr, status_stride_field, status_size_field, where,
                   rank, peer, flag.load<__ATOMIC_RELAXED, MEMORY_SCOPE>(),
                   want, waited);
    __builtin_amdgcn_endpgm();
  }
  if constexpr (ACQUIRE) fence<__ATOMIC_ACQUIRE, MEMORY_SCOPE>();
}

}  // namespace impl

// =================================================================================
// HOW ONE GPU SYNCHRONIZES WITH ANOTHER (listed in the INTERFACE): the
// barriers, written against the signals buffer; a rank's data buffers are
// ordered only by `barrier`.
// =================================================================================

// WHAT HOLDS ONCE A BARRIER IS PASSED: every peer has launched (so its input is
// ready to read; a launch's first barrier); what this side wrote before is
// visible to the other side after; every peer is done reading this rank (so its
// buffers may be reused). Every barrier pairs this block with the same block on
// every rank.
enum class Until { launched, visible, read };

// A FLAG: rank `owner`'s for block `block` on rank `writer`, at `parity`,
// from the signals buffer [R, 2, B, R] (every rank's address, its strides and
// its sizes; debug builds check each index against its size). AN INDEX KEEPS
// ITS CALLER'S TYPE (a block unsigned, as blockIdx.x; a rank as passed): an int
// where the caller had an unsigned costs a sign extension in every kernel.
namespace impl {
template <int WORLD, typename OWNER_TYPE, typename WRITER_TYPE>
DINLINE Counter signal_flag(
    uint32_t* const* signals_ptrs, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    OWNER_TYPE owner, uint32_t parity, unsigned block, WRITER_TYPE writer) {
  debug_in_bounds(static_cast<int64_t>(owner), WORLD);
  debug_in_bounds(parity, signals_size_parity);
  debug_in_bounds(block, signals_size_block);
  debug_in_bounds(static_cast<int64_t>(writer), signals_size_writer);
  return Counter(signals_ptrs[owner] + parity * signals_stride_parity +
                 block * signals_stride_block + writer * signals_stride_writer);
}
}  // namespace impl

// THE BARRIER, one for every case: barrier<Until::read, 1, WORLD>(the signals
// buffer, the status buffer, rank, timeout_ticks), each buffer its pointer,
// strides and sizes, as the kernel took them. Nothing outlives the call. The
// PARITY is the kernel's: its k-th barrier, in source order, takes k % 2, so a
// launch starts on parity 0 and, its barriers even in number, ends on 1
// (dev/lint_kernels.sh checks both). A read after a barrier may see only what
// the SAME BLOCK on the peer wrote before it, so both sides must index the same
// data by the same block.
//
// Block b waits for block b on every rank, and for no other block: enough at
// the ends, where it says "every peer has launched" or "every peer is done
// reading me", and not enough between phases, where a block reads only what the
// same-numbered block wrote. Ordered
// (`visible`): the store releases and the wait acquires, so what the block put
// before is visible to its peers' same-numbered block after. Unordered, it only
// says when (launched: every peer has launched; read: every peer is done
// reading us).
//
// Lane i sets its slot in peer i's signals, waits for peer i's in this rank's,
// and clears it. The clear must land before the lane's NEXT flag goes out, or a
// peer that sees that flag could set the slot again before the clear does and
// lose it: an ordered barrier's wait_stores covers it; an unordered one waits
// for the lane's own stores just before it signals, after the whole phase
// between, so the clear has long landed.
namespace impl {
template <Until UNTIL, int PARITY, int WORLD>
DINLINE void barrier(uint32_t* const* signals_ptrs,
                     int64_t signals_stride_parity,
                     int64_t signals_stride_block,
                     int64_t signals_stride_writer, int signals_size_parity,
                     int signals_size_block, int signals_size_writer,
                     uint32_t* status_ptr, int64_t status_stride_field,
                     int status_size_field, int rank, uint64_t timeout_ticks) {
  static_assert(PARITY == 0 || PARITY == 1, "a barrier's parity is 0 or 1");
  constexpr bool kOrdered = UNTIL == Until::visible;
  constexpr bool kFirst = UNTIL == Until::launched;
  constexpr uint32_t parity = PARITY;
  impl::skew(rank, parity);
  if constexpr (!kFirst) {
    if constexpr (kOrdered) wait_stores();
    __syncthreads();
  }
  if (threadIdx.x < WORLD) {
    if constexpr (!kOrdered) wait_stores();
    const auto flag = [&](auto owner, unsigned block, auto writer) {
      return signal_flag<WORLD>(
          signals_ptrs, signals_stride_parity, signals_stride_block,
          signals_stride_writer, signals_size_parity, signals_size_block,
          signals_size_writer, owner, parity, block, writer);
    };
    const Counter theirs = flag(threadIdx.x, blockIdx.x, rank);
    const Counter mine = flag(rank, blockIdx.x, threadIdx.x);
    theirs.template store<kOrdered ? __ATOMIC_RELEASE : __ATOMIC_RELAXED,
                          __MEMORY_SCOPE_SYSTEM>(
        static_cast<uint32_t>(FlagState::arrived));
    wait<kOrdered, __MEMORY_SCOPE_DEVICE>(
        timeout_ticks, status_ptr, status_stride_field, status_size_field, rank,
        mine, static_cast<uint32_t>(FlagState::arrived),
        kFirst ? Where::start : Where::peer_barrier,
        static_cast<int>(threadIdx.x));
    mine.template store<__ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM>(
        static_cast<uint32_t>(FlagState::clear));
  }
  __syncthreads();
}
}  // namespace impl

// =================================================================================================
// COLLECTIVES
//
// THE COLLECTIVES' STEPS, built from common's tiles: one tested and tuned way
// to do each, which the kernels call rather than write again (the fused AttnRes
// pull's own reduce-scatter took 1.6x the plain two-shot's cycles for its
// bytes, thread traces 2026-10-04T21-03-09Z and 21-26-11Z).
// =================================================================================================

// THE LOADS ARE THE COMMUNICATION: `rank_chunk(w)` points into rank + w's
// memory, mapped into this GPU's (PEERS), so loading the chunk reads it
// over the links; the sum and the store are local. The kernel syncs around
// these calls (the barriers): before, every peer's data is ready; between the
// reduce-scatter and the all-gather, every rank's sums are visible.
//
// THE COLLECTIVES WORK A CHUNK AT A TIME, as a tile whose ROWS ARE THE RANKS:
// row w is rank + w's copy of the chunk (ROTATED, so at any moment the GPUs
// read different peers), a row of threads a rank (TILE::kThreadsM == WORLD).
// `chunk` is the place: its N the slice's end, its offs_n the chunk's first
// column. A rank's pointer is picked a wave at a time (`rank_chunk(w)`, a
// scalar select): every rank's held up front was ~20 scalar loads before the
// first barrier.

// REDUCE-SCATTER: this rank's slice summed over every rank, a chunk a call.
//   rank_chunk(w)  where rank + w's copy of the chunk starts
//   sum_out        where this rank's summed chunk goes (its scratch)
// The sum's order differs by rank, which is harmless: each chunk is summed by
// one rank.
namespace impl {
template <typename TILE, typename RANK_CHUNK>
DINLINE void reduce_scatter(TILE& chunk, RANK_CHUNK rank_chunk,
                            typename TILE::Dtype* sum_out) {
  // 1. Every rank's copy of the chunk, a row each, in one round trip.
  tile_gather(chunk, rank_chunk);
  // 2. Summed over the rows (the ranks), in row order, through LDS.
  const auto sum = block_reduce<Sum, Axis::m>(chunk);
  // 3. Stored as DTYPE in this rank's slice.
  tile_store(sum_out, chunk.N, sum.template to<typename TILE::Dtype>());
}
}  // namespace impl

// AN ELEMENTWISE [R, M, N] PROBLEM -- R ranks' copies of an [M, N] buffer, each
// at its own strides -- WALKED AS [R, 1, M*N]: the rows are only the buffer's
// layout, so the elements are numbered row-major (e = row * N + col, flat_at)
// and each rank's SHARE is one contiguous range of them, a rank's tile copy
// TILE_N of it. WHY: the links and HBM reward long reads from each peer. Cut a
// row at a time, a rank's share was its slice of every row -- 1.8 KB runs at N
// = 7168, the plain all-reduce +2% against the flat walk; 128-column tiles, 256
// B runs, +3-7% (2026-10-11) -- while a contiguous buffer's range is one run of
// the whole share. IT SERVES BADLY a strided buffer: its runs are its rows
// wherever it has them, and each pack's row and column are divided out. The
// walk is the same for every step and every rank, so a block reads in the
// all-gather what the same block on each rank wrote in the reduce-scatter. A
// copy is whole waves (its rank's pointer one scalar select); copy w is rank +
// w's, ROTATED, so at any moment the GPUs read different peers.
namespace impl {
// THE SHARES of m x n elements over WORLD ranks, in whole packs (the last
// rank's short), and the chunks of TILE_N a share takes.
struct Shares {
  int64_t total;
  int slice;
  int chunks;
  DINLINE int64_t first(int r) const { return int64_t{r} * slice; }
  DINLINE int len(int r) const {
    return static_cast<int>(
        max(int64_t{0}, min(int64_t{slice}, total - first(r))));
  }
};
template <typename TILE>
DINLINE Shares shares_of(int m, int n) {
  constexpr int WORLD = TILE::kTileR;
  const int64_t total = int64_t{m} * n;
  const int slice =
      static_cast<int>((total / TILE::kPack + WORLD - 1) / WORLD * TILE::kPack);
  return {total, slice, (slice + TILE::kTileN - 1) / TILE::kTileN};
}
// WHETHER AN [*, n] BUFFER'S ROWS ARE BACK TO BACK, so its range is one run:
// CONTIGUOUS, a template parameter of the kernel, its instance chosen on the
// host from the strides as given. A compile-time choice, not a branch in the
// kernel: both paths in one kernel cost ~0.1 us a call at 128-512 tokens
// against the contiguous path alone (the strided code and its registers,
// 2026-10-12T01-17-18Z). THIS THREAD'S COPY OF THE CHUNK, from the range
// starting at element `first` of an [*, n] buffer: one run when its rows are
// back to back (the address a uniform base and a lane's offset, as tile_load
// builds it), else each pack's own row and column.
template <bool CONTIGUOUS, typename TILE, typename T>
DINLINE void range_load(TILE& t, const Ptr<T>& p, int n, int64_t first) {
  using P = typename TILE::Pack;
  if constexpr (CONTIGUOUS) {
    tile_load(t, p.data + first * p.stride_n, 0);
  } else {
#pragma unroll
    for (int k = 0; k < t.K; ++k)
      held(t, 0, k,
           pack_load(reinterpret_cast<const P*>(
               flat_at(p, first + int64_t{t.col(k)} * t.kPack, n))));
    issued();
  }
}
// ...and stored, its first `len` elements.
template <bool CONTIGUOUS, typename TILE, typename T>
DINLINE void range_store(const TILE& t, const Ptr<T>& p, int n, int64_t first,
                         int len) {
  using P = typename TILE::Pack;
  P* run = reinterpret_cast<P*>(p.data + first * p.stride_n);
#pragma unroll
  for (int k = 0; k < t.K; ++k) {
    const int c = t.offs_n / t.kPack + t.lane() + k * t.kThreadsN;
    if (c >= len / t.kPack) continue;
    P* at;
    if constexpr (CONTIGUOUS)
      at = at_col(run, t, k);
    else
      at = reinterpret_cast<P*>(flat_at(p, first + int64_t{c} * t.kPack, n));
    if constexpr (sizeof(P) == 16)
      global_store(at, pack(t, 0, k));
    else
      *at = pack(t, 0, k);
  }
}

// REDUCE-SCATTER: this rank's share of the inputs, summed over the ranks, into
// its scratch (one contiguous run from scratch.data). inp(r) is rank r's input
// at its strides, every rank's alike.
template <typename TILE, bool CONTIGUOUS, typename RANK_PTR, typename T>
DINLINE void reduce_scatter(int rank, RANK_PTR input, int m, int n,
                            const Ptr<T>& scratch) {
  static_assert(TILE::kTileM == 1 && TILE::kThreadsR == TILE::kTileR,
                "an [R, 1, TILE_N] chunk");
  static_assert(TILE::kSliceThreads % kWaveSize == 0,
                "a rank's copy is whole waves");
  constexpr int WORLD = TILE::kTileR;
  const Shares s = shares_of<TILE>(m, n);
  if (s.len(rank) == 0)
    return;  // a late rank's share is empty at few packs (every block alike)
  block_for_each_tile(1, s.chunks, [&](int, int chunk) {
    TILE got{1, s.len(rank), 0, chunk * TILE::kTileN};
    // 1. Every rank's copy of this chunk of this rank's share, in one round
    // trip.
    range_load<CONTIGUOUS>(got, input((rank + got.slice()) % WORLD), n,
                           s.first(rank));
    // 2. Summed over the ranks, in rank order, through LDS.
    const auto sum = block_reduce<Sum, Axis::r>(got);
    // 3. Stored as T in this rank's scratch.
    tile_store(scratch.data, 0, sum.template to<T>());
  });
}

// ALL-GATHER: every rank's summed share out of its scratch, at its place in the
// output.
template <typename TILE, bool CONTIGUOUS, typename RANK_PTR, typename T>
DINLINE void all_gather(int rank, RANK_PTR scratch, const Ptr<T>& output, int m,
                        int n) {
  static_assert(TILE::kTileM == 1 && TILE::kThreadsR == TILE::kTileR,
                "an [R, 1, TILE_N] chunk");
  static_assert(TILE::kSliceThreads % kWaveSize == 0,
                "a rank's copy is whole waves");
  constexpr int WORLD = TILE::kTileR;
  const Shares s = shares_of<TILE>(m, n);
  block_for_each_tile(1, s.chunks, [&](int, int chunk) {
    TILE got{1, s.slice, 0, chunk * TILE::kTileN};
    const int r = (rank + got.slice()) % WORLD;
    // 1. Rank r's summed chunk, in one round trip.
    tile_load(got, scratch(r).data, 0);
    // 2. Stored where rank r's share goes.
    range_store<CONTIGUOUS>(got, output, n, s.first(r), s.len(r));
  });
}

// STAGED, BEFORE THE REDUCE-SCATTER: this rank's input, every rank's share of
// it, into its staging (an [m, n] buffer, ours, its rows back to back: the
// host's staging_stride_m = n), each block the chunks that rank's same block
// reads.
template <typename TILE, bool CONTIGUOUS, typename T>
DINLINE void stage(int rank, const Ptr<const T>& input, const Ptr<T>& staging,
                   int m, int n) {
  static_assert(TILE::kTileM == 1 && TILE::kThreadsR == TILE::kTileR,
                "an [R, 1, TILE_N] chunk");
  constexpr int WORLD = TILE::kTileR;
  const Shares s = shares_of<TILE>(m, n);
  block_for_each_tile(1, s.chunks, [&](int, int chunk) {
    const int r = (rank + TILE{}.slice()) % WORLD;
    const int len = s.len(r);
    if (len == 0) return;
    TILE part{1, len, 0, chunk * TILE::kTileN};
    range_load<CONTIGUOUS>(part, input, n, s.first(r));
    range_store<true>(part, staging, n, s.first(r), len);
  });
}
}  // namespace impl

// =================================================================================================
// INTERFACE
//
// THE INTERFACE: each primitive a kernel may call (the overview is this file's
// head), passing its arguments to its section's impl.
// =================================================================================================

// ===============================================================================================
// TILES: WHICH TILES A BLOCK WORKS ON
// ===============================================================================================

// This block's tiles, grid-strided in row-major order over tiles_m x tiles_n
// tiles: body(row, col) for each, in tiles, without a division a tile.
template <typename BODY>
DINLINE void block_for_each_tile(int tiles_m, int tiles_n, BODY body) {
  impl::block_for_each_tile(tiles_m, tiles_n, body);
}

// ===============================================================================================
// MEMORY: GLOBAL MEMORY ONLY THROUGH TILES
// ===============================================================================================

// One round trip a tile; a row past M reads the last row.
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N,
          int THREADS_PER_BLOCK, typename ACC_DTYPE>
DINLINE void tile_load(Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N,
                            THREADS_PER_BLOCK, ACC_DTYPE>& tile,
                       const ACC_DTYPE* data, int64_t row_stride) {
  impl::tile_load(tile, data, row_stride);
}

// Stores only the rows below M and the columns below N.
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N,
          int THREADS_PER_BLOCK, typename ACC_DTYPE>
DINLINE void tile_store(ACC_DTYPE* data, int64_t row_stride,
                        const Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N,
                                   THREADS_PER_BLOCK, ACC_DTYPE>& tile) {
  impl::tile_store(data, row_stride, tile);
}

// Every rank's tile in flight together: rank r's tensor at rank_data(r).
// TILE FIRST, POINTER SECOND: a tile from a Ptr, a tile to a Ptr; a group of
// tiles from a group of Ptrs (every load issued together, each pack's address
// computed once; the group shares a stride).
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N,
          int THREADS_PER_BLOCK, typename ACC_DTYPE, typename T>
DINLINE void tile_load(Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N,
                            THREADS_PER_BLOCK, ACC_DTYPE>& tile,
                       const Ptr<T>& ptr) {
  impl::tile_load(tile, ptr.data, ptr.stride_m);
}
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N,
          int THREADS_PER_BLOCK, typename ACC_DTYPE>
DINLINE void tile_store(const Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N,
                                   THREADS_PER_BLOCK, ACC_DTYPE>& tile,
                        const Ptr<ACC_DTYPE>& ptr) {
  impl::tile_store(ptr.data, ptr.stride_m, tile);
}
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N,
          int THREADS_PER_BLOCK, typename ACC_DTYPE, std::size_t N, typename T>
DINLINE void tile_load(Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N,
                            THREADS_PER_BLOCK, ACC_DTYPE> (&tiles)[N],
                       const std::array<Ptr<T>, N>& ptrs) {
  impl::peers_load(
      tiles, [&](int r) { return ptrs[r].data; }, ptrs[0].stride_m);
}
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N,
          int THREADS_PER_BLOCK, typename ACC_DTYPE, int WORLD,
          typename RANK_DATA>
DINLINE void peers_load(Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N,
                             THREADS_PER_BLOCK, ACC_DTYPE> (&tiles)[WORLD],
                        RANK_DATA rank_data, int64_t row_stride) {
  impl::peers_load(tiles, rank_data, row_stride);
}

// Each column from the rank owning it, `slice` columns a rank (the owner picked
// in each lane).
template <int WORLD, typename DTYPE, int TILE_M, int TILE_N, int THREADS_M,
          int THREADS_N, int THREADS_PER_BLOCK, typename ACC_DTYPE,
          typename RANK_DATA>
DINLINE void sliced_load(Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N,
                              THREADS_PER_BLOCK, ACC_DTYPE>& tile,
                         RANK_DATA rank_data, int64_t row_stride, int slice) {
  impl::sliced_load<WORLD>(tile, rank_data, row_stride, slice);
}

// A tile whose rows are different tensors (a row a rank): row m at row_data(m).
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N,
          int THREADS_PER_BLOCK, typename ACC_DTYPE, typename ROW_DATA>
DINLINE void tile_gather(Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N,
                              THREADS_PER_BLOCK, ACC_DTYPE>& tile,
                         ROW_DATA row_data) {
  impl::tile_gather(tile, row_data);
}

// A row's scalar (a norm's scale), stored once a block.
DINLINE void block_store_row_scalar(float* scalars, int row, float value) {
  impl::block_store_row_scalar(scalars, row, value);
}

// Every rank's scalar for `row`.
template <int WORLD, typename RANK_DATA>
DINLINE void peers_load_row_scalars(RANK_DATA rank_data, int row,
                                    float (&output)[WORLD]) {
  impl::peers_load_row_scalars<WORLD>(rank_data, row, output);
}

// ===============================================================================================
// ELEMENTWISE: FLOAT TILE MATH
// ===============================================================================================

// a + b, b of a's shape or one row (a weight).
template <typename A_DTYPE, int A_TILE_M, int A_TILE_N, int A_THREADS_M,
          int A_THREADS_N, int A_THREADS_PER_BLOCK, typename A_ACC_DTYPE,
          typename B_DTYPE, int B_TILE_M, int B_TILE_N, int B_THREADS_M,
          int B_THREADS_N, int B_THREADS_PER_BLOCK, typename B_ACC_DTYPE>
DINLINE Tile<A_DTYPE, A_TILE_M, A_TILE_N, A_THREADS_M, A_THREADS_N,
             A_THREADS_PER_BLOCK, A_ACC_DTYPE>
tile_add(const Tile<A_DTYPE, A_TILE_M, A_TILE_N, A_THREADS_M, A_THREADS_N,
                    A_THREADS_PER_BLOCK, A_ACC_DTYPE>& a,
         const Tile<B_DTYPE, B_TILE_M, B_TILE_N, B_THREADS_M, B_THREADS_N,
                    B_THREADS_PER_BLOCK, B_ACC_DTYPE>& b) {
  return impl::tile_add(a, b);
}

// a * b, b of a's shape or one row (a weight).
template <typename A_DTYPE, int A_TILE_M, int A_TILE_N, int A_THREADS_M,
          int A_THREADS_N, int A_THREADS_PER_BLOCK, typename A_ACC_DTYPE,
          typename B_DTYPE, int B_TILE_M, int B_TILE_N, int B_THREADS_M,
          int B_THREADS_N, int B_THREADS_PER_BLOCK, typename B_ACC_DTYPE>
DINLINE Tile<A_DTYPE, A_TILE_M, A_TILE_N, A_THREADS_M, A_THREADS_N,
             A_THREADS_PER_BLOCK, A_ACC_DTYPE>
tile_mul(const Tile<A_DTYPE, A_TILE_M, A_TILE_N, A_THREADS_M, A_THREADS_N,
                    A_THREADS_PER_BLOCK, A_ACC_DTYPE>& a,
         const Tile<B_DTYPE, B_TILE_M, B_TILE_N, B_THREADS_M, B_THREADS_N,
                    B_THREADS_PER_BLOCK, B_ACC_DTYPE>& b) {
  return impl::tile_mul(a, b);
}

// Each row of a times its scale.
template <typename A_DTYPE, int A_TILE_M, int A_TILE_N, int A_THREADS_M,
          int A_THREADS_N, int A_THREADS_PER_BLOCK, typename A_ACC_DTYPE>
DINLINE Tile<A_DTYPE, A_TILE_M, A_TILE_N, A_THREADS_M, A_THREADS_N,
             A_THREADS_PER_BLOCK, A_ACC_DTYPE>
tile_mul(const Tile<A_DTYPE, A_TILE_M, A_TILE_N, A_THREADS_M, A_THREADS_N,
                    A_THREADS_PER_BLOCK, A_ACC_DTYPE>& a,
         const float (&row_scale)[A_TILE_M / A_THREADS_M]) {
  return impl::tile_mul(a, row_scale);
}

// Every element of a times one scale.
template <typename A_DTYPE, int A_TILE_M, int A_TILE_N, int A_THREADS_M,
          int A_THREADS_N, int A_THREADS_PER_BLOCK, typename A_ACC_DTYPE>
DINLINE Tile<A_DTYPE, A_TILE_M, A_TILE_N, A_THREADS_M, A_THREADS_N,
             A_THREADS_PER_BLOCK, A_ACC_DTYPE>
tile_mul(const Tile<A_DTYPE, A_TILE_M, A_TILE_N, A_THREADS_M, A_THREADS_N,
                    A_THREADS_PER_BLOCK, A_ACC_DTYPE>& a,
         float scale) {
  return impl::tile_mul(a, scale);
}

// a * a_row_scale + b * b_row_scale, a row's scales each.
template <typename A_DTYPE, int A_TILE_M, int A_TILE_N, int A_THREADS_M,
          int A_THREADS_N, int A_THREADS_PER_BLOCK, typename A_ACC_DTYPE,
          typename B_DTYPE, int B_TILE_M, int B_TILE_N, int B_THREADS_M,
          int B_THREADS_N, int B_THREADS_PER_BLOCK, typename B_ACC_DTYPE>
DINLINE Tile<A_DTYPE, A_TILE_M, A_TILE_N, A_THREADS_M, A_THREADS_N,
             A_THREADS_PER_BLOCK, A_ACC_DTYPE>
tile_fma(const Tile<A_DTYPE, A_TILE_M, A_TILE_N, A_THREADS_M, A_THREADS_N,
                    A_THREADS_PER_BLOCK, A_ACC_DTYPE>& a,
         const float (&a_row_scale)[A_TILE_M / A_THREADS_M],
         const Tile<B_DTYPE, B_TILE_M, B_TILE_N, B_THREADS_M, B_THREADS_N,
                    B_THREADS_PER_BLOCK, B_ACC_DTYPE>& b,
         const float (&b_row_scale)[A_TILE_M / A_THREADS_M]) {
  return impl::tile_fma(a, a_row_scale, b, b_row_scale);
}

// ===============================================================================================
// REDUCE: OVER THE RANKS, A WAVE OR THE BLOCK
// ===============================================================================================

// Every rank's tile summed, in rank order in fp32, rounded once.
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N,
          int THREADS_PER_BLOCK, typename ACC_DTYPE, int WORLD>
DINLINE Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK,
             ACC_DTYPE>
peers_reduce(const Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N,
                        THREADS_PER_BLOCK, ACC_DTYPE> (&tiles)[WORLD]) {
  return impl::peers_reduce(tiles);
}

// NUM_VALUES values over the wave, every lane left holding the results.
template <typename REDUCE_OP, int NUM_VALUES>
DINLINE void wave_reduce(float (&values)[NUM_VALUES]) {
  impl::wave_reduce<REDUCE_OP, NUM_VALUES>(values);
}

// NUM_VALUES values over the block, every thread left holding the results.
template <typename REDUCE_OP, int THREADS_PER_BLOCK, int NUM_VALUES>
DINLINE void block_reduce(float (&values)[NUM_VALUES]) {
  impl::block_reduce<REDUCE_OP, THREADS_PER_BLOCK, NUM_VALUES>(values);
}

// A tile reduced over its rows (Axis::m): one fp32 row, the first THREADS_N
// threads'.
template <typename REDUCE_OP, Axis AXIS, typename DTYPE, int TILE_M, int TILE_N,
          int THREADS_M, int THREADS_N, int THREADS_PER_BLOCK,
          typename ACC_DTYPE>
DINLINE Tile<DTYPE, 1, TILE_N, 1, THREADS_N, THREADS_PER_BLOCK, float>
block_reduce(const Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N,
                        THREADS_PER_BLOCK, ACC_DTYPE>& tile) {
  return impl::block_reduce<REDUCE_OP, AXIS>(tile);
}

// A 3-D tile reduced over its ranks (Axis::r): the TILE_M x TILE_N fp32 tile,
// the first rank's threads'.
template <typename REDUCE_OP, Axis AXIS, typename DTYPE, int TILE_M, int TILE_N,
          int THREADS_M, int THREADS_N, int THREADS_PER_BLOCK,
          typename ACC_DTYPE, int TILE_R, int THREADS_R>
DINLINE Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK,
             float>
block_reduce(const Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N,
                        THREADS_PER_BLOCK, ACC_DTYPE, TILE_R, THREADS_R>& tile)
  requires(AXIS == Axis::r)
{
  return impl::block_reduce<REDUCE_OP, AXIS>(tile);
}

// Each row of a tile reduced over its columns (Axis::n), into out[row].
template <typename REDUCE_OP, Axis AXIS, typename DTYPE, int TILE_M, int TILE_N,
          int THREADS_M, int THREADS_N, int THREADS_PER_BLOCK,
          typename ACC_DTYPE>
DINLINE void block_reduce(
    const Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N, THREADS_PER_BLOCK,
               ACC_DTYPE>& tile,
    float (&output)[TILE_M / THREADS_M]) {
  impl::block_reduce<REDUCE_OP, AXIS>(tile, output);
}

// ===============================================================================================
// DOT
// ===============================================================================================

// This thread's share of each row's dot of a and b (b of a's shape or one row,
// a weight).
template <typename A_DTYPE, int A_TILE_M, int A_TILE_N, int A_THREADS_M,
          int A_THREADS_N, int A_THREADS_PER_BLOCK, typename A_ACC_DTYPE,
          typename B_DTYPE, int B_TILE_M, int B_TILE_N, int B_THREADS_M,
          int B_THREADS_N, int B_THREADS_PER_BLOCK, typename B_ACC_DTYPE>
DINLINE void partial_dot(
    const Tile<A_DTYPE, A_TILE_M, A_TILE_N, A_THREADS_M, A_THREADS_N,
               A_THREADS_PER_BLOCK, A_ACC_DTYPE>& a,
    const Tile<B_DTYPE, B_TILE_M, B_TILE_N, B_THREADS_M, B_THREADS_N,
               B_THREADS_PER_BLOCK, B_ACC_DTYPE>& b,
    float (&d)[A_TILE_M / A_THREADS_M]) {
  impl::partial_dot(a, b, d);
}

// ===============================================================================================
// SOFTMAX
// ===============================================================================================

// A tile of logits folded into a running softmax: each logit's scale, and the
// old sum's.
template <int NUM_LOGITS>
DINLINE float thread_softmax_fold(OnlineSoftmax& softmax,
                                  const float (&logit)[NUM_LOGITS],
                                  float (&scale)[NUM_LOGITS]) {
  return impl::thread_softmax_fold<NUM_LOGITS>(softmax, logit, scale);
}

// ===============================================================================================
// POINTERS: A KERNEL'S ARGUMENTS AS Ptrs
// ===============================================================================================

// ONE FORMAT (STANDARDS 3.2): a kernel's arguments are raw pointers, strides,
// sizes and scalars; a peer buffer is a device table of every rank's address
// (`x_ptrs`, as vLLM's RankData). Its first lines make them Ptrs, named for the
// buffer (`inp`, `scratch`, `out`), and its body touches memory only through
// them. THE BOUNDARY (PTR): every rank's buffer as a Ptr (rank r's at r),
// and a local one as one.
template <typename T, int WORLD>
DINLINE std::array<Ptr<T>, WORLD> rank_ptrs(T* const* ptrs, int64_t stride_m,
                                            int64_t stride_n) {
  return impl::rank_ptrs<T, WORLD>(ptrs, stride_m, stride_n);
}
template <typename T, int WORLD>
DINLINE Ptr<T> rank_ptr(T* const* ptrs, int rank, int64_t stride_m,
                        int64_t stride_n) {
  return impl::rank_ptr<T, WORLD>(ptrs, rank, stride_m, stride_n);
}
template <typename T>
DINLINE Ptr<T> local_ptr(T* data, int64_t stride_m, int64_t stride_n,
                         int rank) {
  return impl::local_ptr<T>(data, stride_m, stride_n, rank);
}

// ===============================================================================================
// BARRIER: THE RANKS' SYNCHRONIZATION
// ===============================================================================================

// This block and the same block on every rank, UNTIL launched (every peer's
// input is ready; a launch's first barrier), visible (what was written before
// is seen after) or read (every peer is done reading this rank). A kernel's
// barriers are even in number, its k-th taking PARITY k % 2, so its flags end
// where they began. WORLD is the kernel's, passed on.
template <Until UNTIL, int PARITY, int WORLD>
DINLINE void barrier(uint32_t* const* signals_ptrs,
                     int64_t signals_stride_parity,
                     int64_t signals_stride_block,
                     int64_t signals_stride_writer, int signals_size_parity,
                     int signals_size_block, int signals_size_writer,
                     uint32_t* status_ptr, int64_t status_stride_field,
                     int status_size_field, int rank, uint64_t timeout_ticks) {
  impl::barrier<UNTIL, PARITY, WORLD>(
      signals_ptrs, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);
}

// ===============================================================================================
// COLLECTIVES: THE TWO-SHOT'S STEPS, ONE TESTED WAY EACH
// ===============================================================================================

// A chunk of this rank's slice summed over the ranks into sum_out: rank w's
// copy at rank_chunk(w).
template <typename DTYPE, int TILE_M, int TILE_N, int THREADS_M, int THREADS_N,
          int THREADS_PER_BLOCK, typename ACC_DTYPE, typename RANK_CHUNK>
DINLINE void reduce_scatter(Tile<DTYPE, TILE_M, TILE_N, THREADS_M, THREADS_N,
                                 THREADS_PER_BLOCK, ACC_DTYPE>& chunk,
                            RANK_CHUNK rank_chunk, DTYPE* sum_out) {
  impl::reduce_scatter(chunk, rank_chunk, sum_out);
}

// AN ELEMENTWISE [R, M, N] PROBLEM: R ranks' copies of an [m, n] buffer, rank
// r's at inp(r) (a Ptr at its strides). TILE is the [R, 1, TILE_N] chunk the
// walk moves (COLLECTIVES: walked as [R, 1, m*n], each rank's share one
// contiguous range of the elements, for long runs). Reduce-scatter: this rank's
// share, summed over the ranks, into its scratch.
template <typename TILE, bool CONTIGUOUS, typename RANK_PTR, typename T>
DINLINE void reduce_scatter(int rank, RANK_PTR input, int m, int n,
                            const Ptr<T>& scratch) {
  impl::reduce_scatter<TILE, CONTIGUOUS>(rank, input, m, n, scratch);
}
// All-gather: every rank's summed share, rank r's at scratch(r), at its place
// in `out`.
template <typename TILE, bool CONTIGUOUS, typename RANK_PTR, typename T>
DINLINE void all_gather(int rank, RANK_PTR scratch, const Ptr<T>& output, int m,
                        int n) {
  impl::all_gather<TILE, CONTIGUOUS>(rank, scratch, output, m, n);
}
// Staged, first: this rank's input, every rank's share of it, into its staging.
template <typename TILE, bool CONTIGUOUS, typename T>
DINLINE void stage(int rank, const Ptr<const T>& input, const Ptr<T>& staging,
                   int m, int n) {
  impl::stage<TILE, CONTIGUOUS>(rank, input, staging, m, n);
}

// ===============================================================================================
// UTILS
// ===============================================================================================

}  // namespace hip_custom_communicator
