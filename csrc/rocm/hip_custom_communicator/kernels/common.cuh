// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// COMMON: everything a kernel may call and the machine under it, in sections,
// each built on those above it: HARDWARE, BUILD, UTILS, TILES, MEMORY,
// ELEMENTWISE, REDUCE, DOT, SOFTMAX, PEERS, BARRIER, QUANTIZATION, COLLECTIVES,
// then THE INTERFACE, last because it passes to them all. Each primitive is implemented
// in its section's namespace impl as plain loops on ints and arrays, and the
// interface gives each one a small function that passes its arguments on,
// every template parameter by name, so what reaches each primitive is visible
// there. A kernel calls the interface and never names impl::
// (dev/lint_kernels.sh).
//
// THE ARGUMENTS ARE A KERNEL'S, in its one order: the inputs, then the outputs,
// then the scalars (offs_m, offs_n, row_step, rank, ...). A buffer is a pointer
// (a device table of every rank's for a peer buffer, `x_ptrs`), its strides,
// its sizes, in elements; stride_n is 1 (the host checks it; the packed loads
// need it). A thread's registers are a plain array the caller declares,
// `T v[ROWS][K][P]`, an input or an output like a buffer: P a pack's elements
// (16 bytes of the kernel's dtype; an fp32 array holds the same P), thread t
// lane t % THREADS_N of thread-row t / THREADS_N, holding rows offs_m +
// (t / THREADS_N + i * THREADS_M) * row_step and packs offs_n / P +
// t % THREADS_N + k * THREADS_N. Every rank's is `T v[WORLD][ROWS][K][P]`. The
// shape is template ints only (WORLD, THREADS, THREADS_M, THREADS_N, TILE_N,
// ...), THREADS_M * THREADS_N <= THREADS the block's; array extents are
// deduced. A row past size_m loads the last row and a pack past size_n the last
// pack; stores and column reductions skip them.
//
// CONCURRENCY IS PART OF EACH PRIMITIVE'S CONTRACT, so a kernel keeps it by
// using them:
//   1. Global memory only through the loads and stores here: a kernel never
//      indexes a row.
//   2. A thread's registers are loaded whole (load, peers_load: one round
//      trip) before anything is stored; a pack loaded between stores waits a
//      round trip, since a store may alias it.
//   3. A load that does not depend on a reduction is issued before it
//      (weights, the next row or source), so its round trip runs under the
//      reduction.
//
// THE TYPES a kernel names, by the section that defines them:
//   HARDWARE    Hardware, kTarget, kWaveSize: the device's documented facts
//   BUILD       kBuild: what is compiled for the device
//   UTILS       traits<DTYPE>::N: a pack's elements, the P of an array
//   REDUCE      Sum, Max (a reduction's operation)
//   SOFTMAX     OnlineSoftmax
//   PEERS       StatusField, kStatusFields, kMaxRanks, kMaxBlocks
//   BARRIER     Until: what a barrier waits for (launched, visible, read)
//   QUANTIZATION Quantization: what crosses the links (none, int4), kInt4Group
//   COLLECTIVES Direction: where a reduce-scatter's sum goes (pull, push)

#pragma once

#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
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
  // THE EXCHANGE (scratch and staging) IS NOT HERE: it is sized at open from
  // the largest call the communicator serves (select.cuh's `exchange_sizes`),
  // never a constant.
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

// A PACK: kBuild.memory.pack_bytes (BUILD: the widest load, 8 bf16), the unit
// every kernel loads, sums and stores in; traits<DTYPE>::N its elements, the P
// of a thread's registers.
template <typename DTYPE>
struct traits {
  static constexpr int N = kBuild.memory.pack_bytes / sizeof(DTYPE);
};

// =================================================================================================
// TILES
//
// WHICH TILES A BLOCK WORKS ON: a kernel cuts its work in Triton's terms (TILE_M x TILE_N tiles
// of an M x N tensor, the last one masked), and each block walks its share of them.
// =================================================================================================

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
// THE READS AND WRITES, a thread's registers from and to buffers: a load and a store (one round
// trip), every rank's at once, the all-gather into registers, a row's scalar. How a pack is
// loaded (one 16-byte global instruction, issued together) is impl's.
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

template <typename PACK>
DINLINE PACK pack_load(const PACK* p) {
  if constexpr (sizeof(PACK) == 16)
    return global_load(p);
  else
    return *p;
}
}  // namespace impl

// A THREAD'S REGISTERS AND THEIR PLACE: `T v[ROWS][K][P]`, P a pack's elements; the place in a
// buffer as ints (its sizes, offs_m, offs_n, row_step) and the layout as template ints
// THREADS_M x THREADS_N. Thread t holds rows offs_m + (t / THREADS_N + i * THREADS_M) *
// row_step and packs offs_n / P + t % THREADS_N + k * THREADS_N.
namespace impl {
// Thread-row i's row of the block's rows, a row past size_m reading row size_m - 1.
template <int THREADS_M, int THREADS_N>
DINLINE int thread_row(int i) {
  // One row of threads: the row is the block's, so its address stays scalar.
  if constexpr (THREADS_M == 1)
    return i;
  else
    return static_cast<int>(threadIdx.x) / THREADS_N + i * THREADS_M;
}
template <int THREADS_M, int THREADS_N>
DINLINE bool row_live(int size_m, int offs_m, int row_step, int i) {
  return offs_m + thread_row<THREADS_M, THREADS_N>(i) * row_step < size_m;
}
template <int THREADS_M, int THREADS_N>
DINLINE int row_of(int size_m, int offs_m, int row_step, int i) {
  return row_live<THREADS_M, THREADS_N>(size_m, offs_m, row_step, i)
             ? offs_m + thread_row<THREADS_M, THREADS_N>(i) * row_step
             : size_m - 1;
}
// This thread's pack k of a row, in packs from the row's start, past size_n clamped to the last,
// so every load stays in bounds; its mask 1 below size_n and 0 past it, a float multiplied in so
// nothing reading it branches (a load under a runtime `if` cannot be hoisted past the branch,
// and loads meant to be in flight together then wait one at a time).
template <int P, int THREADS_N>
DINLINE int pack_of(int size_n, int offs_n, int k) {
  const int n =
      offs_n / P + static_cast<int>(threadIdx.x) % THREADS_N + k * THREADS_N;
  return n < size_n / P ? n : size_n / P - 1;
}
template <int P, int THREADS_N>
DINLINE float pack_mask(int size_n, int offs_n, int k) {
  return offs_n / P + static_cast<int>(threadIdx.x) % THREADS_N +
                     k * THREADS_N <
                 size_n / P
             ? 1.0f
             : 0.0f;
}
// WHETHER THIS THREAD HOLDS ANY OF THE LAYOUT: a layout narrower than the block of THREADS
// leaves the rest out. A compare only for a narrower layout: threadIdx.x < THREADS is not
// folded from the launch bounds (a compare and a branch around every store, ISA 2026-10-15).
template <int THREADS_M, int THREADS_N, int THREADS>
DINLINE bool holds() {
  static_assert(THREADS_M * THREADS_N <= THREADS, "a layout is the block or fewer");
  if constexpr (THREADS_M * THREADS_N == THREADS)
    return true;
  else
    return static_cast<int>(threadIdx.x) < THREADS_M * THREADS_N;
}
// WHETHER A LOAD SKIPS ROW i: only several rows a thread or rows of threads, whose rows past
// size_m would re-read the last over the links. One row's is real wherever a kernel makes one,
// and a branch around its loads cost registers (126 -> 178 VGPRs, spilling at 16384: 23-46-07Z).
template <int ROWS, int THREADS_M, int THREADS_N>
DINLINE bool skips(int size_m, int offs_m, int row_step, int i) {
  if constexpr (ROWS == 1 && THREADS_M == 1) return false;
  return !row_live<THREADS_M, THREADS_N>(size_m, offs_m, row_step, i);
}
// A ROW'S PACK k AS A UNIFORM BASE PLUS AN UNSIGNED 32-BIT BYTE OFFSET: the form global_load
// and global_store take with the base in scalar registers (saddr), so every tensor sharing the
// columns shares one offset register. A signed column added to the pointer was a 64-bit vector
// address a tensor a pack, 28 VGPRs held across AttnRes's loops (liveness, 2026-10-04T15-28-51Z).
// THE ADDRESS IS BUILT AT ITS LOAD: the offset made opaque here, so the 64-bit sum is not
// hoisted and held a tensor a pack (an add a load; AttnRes 90 -> 70 VGPRs, 2026-10-04 compile).
template <int P, int THREADS_N, typename PACK>
DINLINE PACK* at_col(PACK* row, int size_n, int offs_n, int k) {
  using Byte = std::conditional_t<std::is_const_v<PACK>, const char, char>;
  uint32_t offset = static_cast<uint32_t>(pack_of<P, THREADS_N>(size_n, offs_n, k)) *
                    uint32_t{sizeof(PACK)};
  asm volatile("" : "+v"(offset));
  return reinterpret_cast<PACK*>(reinterpret_cast<Byte*>(row) + offset);
}
template <typename T, int P>
DINLINE void held(T (&v)[P], const vec<T, P>& p) {
  __builtin_memcpy(v, &p, sizeof(p));
}
template <typename T, int P>
DINLINE vec<T, P> pack(const T (&v)[P]) {
  vec<T, P> p;
  __builtin_memcpy(&p, v, sizeof(p));
  return p;
}
template <typename PACK>
DINLINE void pack_store(PACK* p, const PACK& v) {
  if constexpr (sizeof(PACK) == 16)
    global_store(p, v);
  else
    *p = v;
}

// A LOAD NEVER BRANCHES: a thread outside a narrow layout loads its clamped, in-bounds place and
// its values go unused. Under a thread compare the load was an exec branch, so every array's
// old registers stayed live across AttnRes's loops (ISA 2026-10-04T16-39-15Z).
template <int THREADS_M, int THREADS_N, typename T, int ROWS, int K, int P>
DINLINE void load(const T* input_ptr, int64_t input_stride_m, int input_size_m,
                  int input_size_n, T (&output)[ROWS][K][P], int offs_m,
                  int offs_n, int row_step) {
  using V = vec<T, P>;
  const V* at = reinterpret_cast<const V*>(input_ptr);
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    if (skips<ROWS, THREADS_M, THREADS_N>(input_size_m, offs_m, row_step, i))
      continue;
    const V* row =
        at + int64_t{row_of<THREADS_M, THREADS_N>(input_size_m, offs_m,
                                                   row_step, i)} *
                 (input_stride_m / P);
#pragma unroll
    for (int k = 0; k < K; ++k)
      held(output[i][k],
           pack_load(at_col<P, THREADS_N>(row, input_size_n, offs_n, k)));
  }
  issued();
}

template <int THREADS_M, int THREADS_N, int THREADS, typename T, int ROWS,
          int K, int P>
DINLINE void store(const T (&input)[ROWS][K][P], T* output_ptr,
                   int64_t output_stride_m, int output_size_m,
                   int output_size_n, int offs_m, int offs_n, int row_step) {
  using V = vec<T, P>;
  if (!holds<THREADS_M, THREADS_N, THREADS>()) return;
  V* at = reinterpret_cast<V*>(output_ptr);
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    if (!row_live<THREADS_M, THREADS_N>(output_size_m, offs_m, row_step, i))
      continue;
    V* row = at + int64_t{row_of<THREADS_M, THREADS_N>(output_size_m, offs_m,
                                                        row_step, i)} *
                      (output_stride_m / P);
#pragma unroll
    for (int k = 0; k < K; ++k) {
      if (pack_mask<P, THREADS_N>(output_size_n, offs_n, k) == 0.0f) continue;
      if constexpr (sizeof(V) == 16)
        global_store(at_col<P, THREADS_N>(row, output_size_n, offs_n, k),
                     pack(input[i][k]));
      else
        *at_col<P, THREADS_N>(row, output_size_n, offs_n, k) = pack(input[i][k]);
    }
  }
}

// EVERY PEER'S, all in flight together, rank r's buffer at input[r] (read from its table once). A
// pack position at a time, every rank's for it, so the address is computed once a position (the
// other way round cost 16 scalar instructions at two packs: ISA 2026-10-01T00-31-14Z). Nothing
// waits until the registers are used, so loads issued here run under other work.
template <int THREADS_M, int THREADS_N, typename T, int WORLD, int ROWS, int K,
          int P>
DINLINE void peers_load(const T* const (&input)[WORLD], int64_t input_stride_m,
                        int input_size_m, int input_size_n,
                        T (&output)[WORLD][ROWS][K][P], int offs_m, int offs_n,
                        int row_step) {
  using V = vec<T, P>;
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    if (skips<ROWS, THREADS_M, THREADS_N>(input_size_m, offs_m, row_step, i))
      continue;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      // THE POSITION ONCE, IN A VECTOR REGISTER: left to the compiler, the
      // row's uniform part was added to every rank's base in scalar registers,
      // an add and a carry a rank (16 instructions a tile at eight ranks, the
      // plain all-reduce against the flat walk: ISA 2026-10-11); held in a
      // VGPR, each rank's address is one shift-and-add of it.
      const int64_t stride = input_stride_m / P;
      int64_t at = int64_t{row_of<THREADS_M, THREADS_N>(input_size_m, offs_m,
                                                         row_step, i)} *
                       stride +
                   pack_of<P, THREADS_N>(input_size_n, offs_n, k);
      asm volatile("" : "+v"(at));
#pragma unroll
      for (int r = 0; r < WORLD; ++r)
        held(output[r][i][k],
             pack_load(reinterpret_cast<const V*>(input[r]) + at));
    }
  }
  issued();
}

// THE ALL-GATHER INTO REGISTERS: each pack from the rank owning its columns, slice_n a rank (the
// last rank's to the end), its pointer read in each lane (a wave's packs may have two owners).
template <int WORLD, int THREADS_M, int THREADS_N, typename T, int ROWS, int K,
          int P>
DINLINE void all_gather(const T* const* scratch_ptrs, int64_t scratch_stride_m,
                        int scratch_size_m, int scratch_size_n,
                        T (&output)[ROWS][K][P], int offs_m, int offs_n,
                        int slice_n) {
  using V = vec<T, P>;
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    if (skips<ROWS, THREADS_M, THREADS_N>(scratch_size_m, offs_m, 1, i))
      continue;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const int col = pack_of<P, THREADS_N>(scratch_size_n, offs_n, k);
      const int owner = min(col / (slice_n / P), WORLD - 1);
      held(output[i][k],
           pack_load(reinterpret_cast<const V*>(scratch_ptrs[owner]) +
                     int64_t{row_of<THREADS_M, THREADS_N>(scratch_size_m,
                                                          offs_m, 1, i)} *
                         (scratch_stride_m / P) +
                     col));
    }
  }
  issued();
}

// A ROW'S SCALAR (a norm's scale), one float a row: thread 0 stores it.
DINLINE void store_row_scalar(float value, float* output_ptr, int row) {
  if (threadIdx.x == 0)
    *(__attribute__((address_space(1))) float*)(output_ptr + row) = value;
}

}  // namespace impl

// =================================================================================================
// ELEMENTWISE
//
// THE ELEMENTWISE OPS, at thread scope, on float registers (a fused op spells each rounding its
// reference does as a round<U> or convert between them).
// =================================================================================================

// ELEMENT BY ELEMENT: b of a's rows or one row (a weight, every row's); a scale a row or one.
// fma is a x a_row_scale + b x b_row_scale as fused multiply-adds (a softmax's fold), b in any
// dtype, converted here: two multiplies and an add were three VALU an element and, vectorized,
// packed multiplies with a chain of adds (ISA 2026-10-04T02-22-41Z).
namespace impl {
template <typename T, typename U, int ROWS, int K, int P>
DINLINE void convert(const T (&input)[ROWS][K][P], U (&output)[ROWS][K][P]) {
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < P; ++j) output[i][k][j] = static_cast<U>(input[i][k][j]);
}

template <typename T, int ROWS, int K, int P>
DINLINE void zeros(T (&output)[ROWS][K][P]) {
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < P; ++j) output[i][k][j] = T(0);
}

template <typename U, int ROWS, int K, int P>
DINLINE void round(float (&values)[ROWS][K][P]) {
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < P; ++j)
        values[i][k][j] = static_cast<float>(static_cast<U>(values[i][k][j]));
}

template <int ROWS, int ROWS_B, int K, int P, typename OP>
DINLINE void zip(const float (&a)[ROWS][K][P], const float (&b)[ROWS_B][K][P],
                 float (&output)[ROWS][K][P], OP op) {
  static_assert(ROWS_B == ROWS || ROWS_B == 1, "b is a's shape or one row");
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < P; ++j)
        output[i][k][j] = op(a[i][k][j], b[ROWS_B == 1 ? 0 : i][k][j]);
}

template <int ROWS, int ROWS_B, int K, int P>
DINLINE void add(const float (&a)[ROWS][K][P], const float (&b)[ROWS_B][K][P],
                 float (&output)[ROWS][K][P]) {
  zip(a, b, output, [](float x, float y) { return x + y; });
}

template <int ROWS, int ROWS_B, int K, int P>
DINLINE void mul(const float (&a)[ROWS][K][P], const float (&b)[ROWS_B][K][P],
                 float (&output)[ROWS][K][P]) {
  zip(a, b, output, [](float x, float y) { return x * y; });
}

template <int ROWS, int K, int P>
DINLINE void mul(const float (&a)[ROWS][K][P], const float (&row_scale)[ROWS],
                 float (&output)[ROWS][K][P]) {
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < P; ++j) output[i][k][j] = a[i][k][j] * row_scale[i];
}

template <int ROWS, int K, int P>
DINLINE void mul(const float (&a)[ROWS][K][P], float scale,
                 float (&output)[ROWS][K][P]) {
  float row_scale[ROWS];
#pragma unroll
  for (int i = 0; i < ROWS; ++i) row_scale[i] = scale;
  mul(a, row_scale, output);
}

template <typename B, int ROWS, int K, int P>
DINLINE void fma(const float (&a)[ROWS][K][P], const float (&a_row_scale)[ROWS],
                 const B (&b)[ROWS][K][P], const float (&b_row_scale)[ROWS],
                 float (&output)[ROWS][K][P]) {
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < P; ++j)
        output[i][k][j] = fmaf(static_cast<float>(b[i][k][j]), b_row_scale[i],
                               a[i][k][j] * a_row_scale[i]);
}
}  // namespace impl

// =================================================================================================
// REDUCE
//
// THE REDUCTIONS, by scope, each built on the one below: over the PEERS (every
// rank's registers summed elementwise in fp32, in rank order),
// over a WAVE (shuffles), over a BLOCK (a wave's result, then one LDS pass). N
// values at once at every scope: each block pass is two barriers and an LDS
// round trip, so a kernel with several reductions takes them together.
// =================================================================================================

// EVERY RANK'S REGISTERS SUMMED, in rank order in fp32, left in fp32 (a convert rounds them,
// once): callers whose sources are the ranks agree bitwise. Waits on the loads only here.
namespace impl {
template <typename T, int WORLD, int ROWS, int K, int P>
DINLINE void peers_sum(const T (&input)[WORLD][ROWS][K][P],
                       float (&output)[ROWS][K][P]) {
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int k = 0; k < K; ++k) {
      float acc[P];
#pragma unroll
      for (int j = 0; j < P; ++j) acc[j] = static_cast<float>(input[0][i][k][j]);
#pragma unroll
      for (int r = 1; r < WORLD; ++r)
#pragma unroll
        for (int j = 0; j < P; ++j) acc[j] += static_cast<float>(input[r][i][k][j]);
#pragma unroll
      for (int j = 0; j < P; ++j) output[i][k][j] = acc[j];
    }
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

// =================================================================================================
// DOT
//
// THE DOT: a thread's share of a row's dot (a block_reduce finishes it).
// =================================================================================================

// This thread's share of dot(a, b) over each row, packs past size_n counting zero: the
// partials a block_reduce turns into the rows' dots (a sum of squares is partial_dot(x, x)). A
// one-row b (a weight) is every row's; a and b in any dtype, converted. A row of threads is
// THREADS_N wide (the block's, or a wave's).
namespace impl {
template <int THREADS_N, typename A, typename B, int ROWS, int ROWS_B, int K,
          int P>
DINLINE void partial_dot(const A (&a)[ROWS][K][P], const B (&b)[ROWS_B][K][P],
                         float (&output)[ROWS], int size_n, int offs_n) {
  static_assert(ROWS_B == ROWS || ROWS_B == 1, "b is a's shape or one row");
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    const int ib = ROWS_B == 1 ? 0 : i;
    output[i] = 0.0f;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      float dk = 0.0f;
#pragma unroll
      for (int j = 0; j < P; ++j)
        dk = fmaf(static_cast<float>(a[i][k][j]), static_cast<float>(b[ib][k][j]),
                  dk);
      output[i] += pack_mask<P, THREADS_N>(size_n, offs_n, k) * dk;
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
// THE RANKS' MEMORY: every rank's buffers as plain tensors the reads and writes
// take, and the types a kernel takes them in. A RANK'S MEMORY, two places:
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
// A buffer is read or written by load / store / peers_load, ordered
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

// RANK r'S BUFFER, its first element: a kernel's arguments carry every rank's
// address of a buffer (a device table, `ptrs`) and this reads one. `r` IS THE SAME ACROSS THE WAVE (a
// constant, or a per-wave rank): read from the first lane, the compiler knows
// it, and the load is scalar rather than one per lane.
namespace impl {
template <typename T>
DINLINE T* rank_of(T* const* ptrs, int r) {
  return ptrs[__builtin_amdgcn_readfirstlane(r)];
}

// Every rank's buffer from its table, rank r's at r (rank_of: one scalar load each).
template <int WORLD, typename T>
DINLINE void peers_of(T* const* ptrs, T* (&output)[WORLD]) {
#pragma unroll
  for (int r = 0; r < WORLD; ++r) output[r] = rank_of(ptrs, r);
}

// Every rank's row scalar, `offset_bytes` into rank r's buffer, after any loads before it.
template <int WORLD, typename T>
DINLINE void peers_load_row_scalars(const T* const* input_ptrs,
                                    float (&output)[WORLD],
                                    int64_t offset_bytes, int row) {
#pragma unroll
  for (int r = 0; r < WORLD; ++r)
    output[r] = *(const __attribute__((address_space(1))) float*)(
        reinterpret_cast<const float*>(
            reinterpret_cast<const char*>(rank_of(input_ptrs, r)) +
            offset_bytes) +
        row);
  issued();
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
// QUANTIZATION
//
// WHAT CROSSES THE LINKS in a two-shot, its QUANTIZATION: each buffer of the exchange in its own
// dtype (none), or INT4, quick reduce's format (csrc/quickreduce): a GROUP is 32 consecutive
// elements of a row, held as 4-bit values biased by 8 (a word 8 elements, its elements 2i and
// 2i + 1 in bits 4i and 16 + 4i, so one shift and mask puts both in a 16-bit pair, quick reduce's
// layout) and one fp16 scale, the group's largest magnitude / 7, ROUNDED TO FP16 BEFORE THE VALUES
// ARE, so each is rounded against the scale stored; an element is (q - 8) * scale. A quantized buffer is two
// buffers, its values ([m, n / 2] bytes) and its scales ([m, n / 32] fp16), each a pointer, its
// strides and its sizes. Every function that moves the exchange takes a QUANTIZATION and both forms
// of each buffer it moves (MEMORY's load, store, peers_load and all_gather; the COLLECTIVES), and
// with none is the function it was, the quantized form's arguments unread. A LOAD DEQUANTIZES INTO
// ITS REGISTERS' DTYPE, each element rounded once, as quick reduce lands its values in its dtype; a
// sum is fp32 and quantized from fp32.
// A THREAD'S PACK IS ONE WORD OF ITS GROUP (8 elements of a 16-bit dtype), so a group is the packs
// of 4 consecutive lanes: a layout's THREADS_N and offs_n are whole groups, and its scale is the
// lanes' largest magnitude, two DPP steps with every lane of the quad active. One thread holding a
// whole group of every peer was 46-62 registers and spilled (the int4-wip prototype).
// =================================================================================================

// A TEMPLATE INT, not an enum: a kernel's demangled name then reads `0` or `1` where an
// enum's reads `(hip_custom_communicator::Quantization)1`, which the ISA tools cut at the
// parenthesis, so a kernel's two builds were one file.
namespace Quantization {
constexpr int none = 0;
constexpr int int4 = 1;
}  // namespace Quantization

constexpr int kInt4Group = 32;           // elements a group
constexpr int kInt4GroupPacks = 4;       // a pack is a word of its group's values
constexpr int kInt4WordBytes = 4;        // a pack's values
constexpr int kInt4ScaleBytes = 2;       // a group's scale

namespace impl {
typedef __attribute__((address_space(1))) uint32_t global_u32;
typedef __attribute__((address_space(1))) __fp16 global_f16;

// `offset` bytes past `base`, the offset opaque so the address is built at its access (at_col's).
template <typename E>
DINLINE E* at_offset(E* base, uint32_t offset) {
  using Byte = std::conditional_t<std::is_const_v<E>, const char, char>;
  asm volatile("" : "+v"(offset));
  return reinterpret_cast<E*>(reinterpret_cast<Byte*>(base) + offset);
}

// PACK `pack` OF A QUANTIZED ROW: its word of values, its group's scale.
DINLINE uint32_t int4_word_load(const uint8_t* row, int pack) {
  return *(const global_u32*)at_offset(row, static_cast<uint32_t>(pack) * kInt4WordBytes);
}
DINLINE float int4_scale_load(const __fp16* row, int pack) {
  return *(const global_f16*)at_offset(
      row, static_cast<uint32_t>(pack / kInt4GroupPacks) * kInt4ScaleBytes);
}
DINLINE void int4_word_store(uint8_t* row, int pack, uint32_t word) {
  *(global_u32*)at_offset(row, static_cast<uint32_t>(pack) * kInt4WordBytes) = word;
}
// Once a group, by its first pack's lane.
DINLINE void int4_scale_store(__fp16* row, int pack, float scale) {
  if (pack % kInt4GroupPacks == 0)
    *(global_f16*)at_offset(
        row, static_cast<uint32_t>(pack / kInt4GroupPacks) * kInt4ScaleBytes) =
        static_cast<__fp16>(scale);
}

// A WORD'S 8 ELEMENTS AT THEIR GROUP'S SCALE, (q - 8) * scale, EXACT in fp32, by quick reduce's
// 0x6400 magic: a mask puts elements 2i and 2i + 1 in an fp16 pair, each 1024 + q (i = 0, 2) or
// 1024 + 16q (i = 1, 3: their nibbles masked in place, no shift), OR'd with 0x6400; then one
// v_fma_mix_f32 a value takes a half at scale (or scale / 16) plus bias (-1032 * scale, or -72 *
// scale): a 22-bit product, then an exact difference. 13 instructions a word, in one asm block:
// left to the compiler, each half was converted and fed to packed fmas (1.5 a value). VOLATILE,
// so the blocks keep their order (int4_used_by pins what consumes them between).
DINLINE void int4_terms(uint32_t word, float scale, float (&term)[8]) {
  const float scale16 = scale * 0.0625f;
  const float bias = scale * -1032.0f;
  const float bias16 = scale * -72.0f;
  uint32_t p0, p1, p2, p3;
  asm volatile("v_and_or_b32 %[p0], %[w], %[m0], %[magic]\n\t"
      "v_and_or_b32 %[p1], %[w], %[m1], %[magic]\n\t"
      "v_lshrrev_b32 %[p2], 8, %[w]\n\t"
      "v_and_or_b32 %[p3], %[p2], %[m1], %[magic]\n\t"
      "v_and_or_b32 %[p2], %[p2], %[m0], %[magic]\n\t"
      "v_fma_mix_f32 %[t0], %[p0], %[s], %[b] op_sel_hi:[1,0,0]\n\t"
      "v_fma_mix_f32 %[t1], %[p0], %[s], %[b] op_sel:[1,0,0] op_sel_hi:[1,0,0]\n\t"
      "v_fma_mix_f32 %[t2], %[p1], %[s16], %[b16] op_sel_hi:[1,0,0]\n\t"
      "v_fma_mix_f32 %[t3], %[p1], %[s16], %[b16] op_sel:[1,0,0] op_sel_hi:[1,0,0]\n\t"
      "v_fma_mix_f32 %[t4], %[p2], %[s], %[b] op_sel_hi:[1,0,0]\n\t"
      "v_fma_mix_f32 %[t5], %[p2], %[s], %[b] op_sel:[1,0,0] op_sel_hi:[1,0,0]\n\t"
      "v_fma_mix_f32 %[t6], %[p3], %[s16], %[b16] op_sel_hi:[1,0,0]\n\t"
      "v_fma_mix_f32 %[t7], %[p3], %[s16], %[b16] op_sel:[1,0,0] op_sel_hi:[1,0,0]"
      : [p0] "=&v"(p0), [p1] "=&v"(p1), [p2] "=&v"(p2), [p3] "=&v"(p3), [t0] "=&v"(term[0]),
        [t1] "=&v"(term[1]), [t2] "=&v"(term[2]), [t3] "=&v"(term[3]), [t4] "=&v"(term[4]),
        [t5] "=&v"(term[5]), [t6] "=&v"(term[6]), [t7] "=&v"(term[7])
      : [w] "v"(word), [m0] "s"(0x000F000Fu), [m1] "s"(0x00F000F0u), [magic] "v"(0x64006400u),
        [s] "v"(scale), [b] "v"(bias), [s16] "v"(scale16), [b16] "v"(bias16));
}

typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
// A 16-BIT DTYPE WITH BF16'S 8 MANTISSA DIGITS (c10::BFloat16's numeric_limits).
template <typename U>
constexpr bool kBf16 = sizeof(U) == 2 && std::numeric_limits<U>::digits == 8;

// THE VALUES DONE BEFORE THE NEXT VOLATILE ASM (int4_terms): the scheduler, free to use 256
// VGPRs at 512 threads, decoded every peer's words of a group ahead of the first sum and spilled.
template <int N>
DINLINE void int4_used_by(const float (&values)[N]) {
#pragma unroll
  for (int j = 0; j < N; ++j) asm volatile("" ::"v"(values[j]));
}

// TWO VALUES LANDED IN U, a 32-bit word: with bf16 one v_cvt_pk_bf16_f32, where each element's own
// conversion was one with a zero half, then a shift and an or (12 instructions a pack, not 4).
template <typename U>
DINLINE uint32_t int4_landed(float a, float b) {
  if constexpr (kBf16<U>) {
    return __builtin_bit_cast(uint32_t, __builtin_convertvector(f32x2{a, b}, bf16x2));
  } else {
    return static_cast<uint32_t>(__builtin_bit_cast(uint16_t, static_cast<U>(a))) |
           static_cast<uint32_t>(__builtin_bit_cast(uint16_t, static_cast<U>(b))) << 16;
  }
}

// A PACK'S WORD AT ITS GROUP'S SCALE, each element rounded once to U, a 16-bit dtype. MADE AS A
// LOADED PACK IS, four 32-bit words copied in whole: element by element, each 16-bit value held a
// register of its own and the pull norms' two rows of every peer's packs spilled 188 VGPRs (256
// VGPRs, 592 B of scratch; 2026-10-15T22-21-26Z).
template <typename U, int P>
DINLINE void int4_decode(uint32_t word, float scale, U (&output)[P]) {
  static_assert(P * 4 == 32 && sizeof(U) == 2, "a pack is one word of 8 16-bit values");
  float term[P];
  int4_terms(word, scale, term);
  u32x4 packed;
#pragma unroll
  for (int w = 0; w < P / 2; ++w) packed[w] = int4_landed<U>(term[2 * w], term[2 * w + 1]);
  __builtin_memcpy(output, &packed, sizeof(packed));
}

// A GROUP'S SCALE from its largest magnitude: / 7, to fp16 (past its largest, clamped to it);
// and the inverse its values are rounded at (a group of zeros: 0, so all 8s at scale 0).
DINLINE float int4_scale(float amax) {
  return static_cast<float>(static_cast<__fp16>(fminf(amax * (1.0f / 7.0f), 65504.0f)));
}
DINLINE float int4_inverse(float scale) {
  return scale > 0.0f ? __builtin_amdgcn_rcpf(scale) : 0.0f;
}

// A PACK'S WORD: each value rounded to the nearest of -8..7 times the scale stored, BY THE MAGIC
// NUMBER: fma(x, inv, 1.5 * 2^23 + 8) rounds x * inv to an integer (the sum's ulp is 1) and leaves
// q + 8 in the low 4 bits, the rest of its low byte 0; clamped by med3 (two elements a packed fma).
// Then packed by bytes: element 2i + 2's value shifted 4 and OR'd onto 2i's makes a byte of the
// word, and two v_perm_b32 gather the even elements' bytes and the odd ones' (7 instructions a word,
// where a shift and an OR an element was 16).
template <int P>
DINLINE uint32_t int4_word(const float (&x)[P], float inv) {
  static_assert(P * 4 == 32, "a pack is one word of 8 values");
  constexpr float kLowest = 12582912.0f;  // 1.5 * 2^23, q + 8 = 0
  uint32_t q[P];
#pragma unroll
  for (int p = 0; p < P / 2; ++p) {
    const f32x2 y = __builtin_elementwise_fma(f32x2{x[2 * p], x[2 * p + 1]}, f32x2{inv, inv},
                                              f32x2{kLowest + 8.0f, kLowest + 8.0f});
#pragma unroll
    for (int e = 0; e < 2; ++e)
      q[2 * p + e] = __builtin_bit_cast(
          uint32_t, __builtin_amdgcn_fmed3f(y[e], kLowest, kLowest + 15.0f));
  }
  // byte b of the word: elements 4 * (b % 2) + (b / 2) and that + 2 (QUANTIZATION's layout)
  const auto nibbles = [](uint32_t high, uint32_t low) {
    uint32_t byte;  // one v_lshl_or_b32 (the compiler split it into a shift and an or)
    asm("v_lshl_or_b32 %0, %1, 4, %2" : "=v"(byte) : "v"(high), "v"(low));
    return byte;
  };
  const uint32_t even_low = nibbles(q[2], q[0]), even_high = nibbles(q[6], q[4]);
  const uint32_t odd_low = nibbles(q[3], q[1]), odd_high = nibbles(q[7], q[5]);
  return __builtin_amdgcn_perm(__builtin_amdgcn_perm(odd_high, odd_low, 0x0c0c0400u),
                               __builtin_amdgcn_perm(even_high, even_low, 0x0c0c0400u),
                               0x05040100u);
}

// A PACK TO ITS WORD AND ITS GROUP'S SCALE: the group's largest magnitude over its 4 lanes
// (quad_perm [1,0,3,2], then [2,3,0,1], as unsigned bits: a float max canonicalized each step).
template <typename T, int P>
DINLINE void int4_encode(const T (&input)[P], uint32_t& word, float& scale) {
  float x[P];
  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < P; ++j) {
    x[j] = static_cast<float>(input[j]);
    amax = fmaxf(amax, fabsf(x[j]));
  }
  uint32_t bits = __builtin_bit_cast(uint32_t, amax);
  bits = max(bits, static_cast<uint32_t>(__builtin_amdgcn_mov_dpp(
                       static_cast<int>(bits), 0xb1, 0xf, 0xf, false)));
  bits = max(bits, static_cast<uint32_t>(__builtin_amdgcn_mov_dpp(
                       static_cast<int>(bits), 0x4e, 0xf, 0xf, false)));
  scale = int4_scale(__builtin_bit_cast(float, bits));
  word = int4_word(x, int4_inverse(scale));
}

// A WHOLE GROUP IN ONE THREAD (the flat walk's): its 4 packs to their words and its scale, the
// largest magnitude the thread's own.
template <typename T, int P>
DINLINE void int4_encode_group(const T (&input)[kInt4GroupPacks][P], u32x4& words,
                               float& scale) {
  float x[kInt4GroupPacks][P];
  float amax = 0.0f;
#pragma unroll
  for (int k = 0; k < kInt4GroupPacks; ++k)
#pragma unroll
    for (int j = 0; j < P; ++j) {
      x[k][j] = static_cast<float>(input[k][j]);
      amax = fmaxf(amax, fabsf(x[k][j]));
    }
  scale = int4_scale(amax);
  const float inv = int4_inverse(scale);
#pragma unroll
  for (int k = 0; k < kInt4GroupPacks; ++k) words[k] = int4_word(x[k], inv);
}

// Every pack of a thread's registers, encoded or decoded.
template <typename T, int ROWS, int K, int P>
DINLINE void int4_encode(const T (&input)[ROWS][K][P], uint32_t (&words)[ROWS][K],
                         float (&scales)[ROWS][K]) {
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int k = 0; k < K; ++k) int4_encode(input[i][k], words[i][k], scales[i][k]);
}
template <typename U, int ROWS, int K, int P>
DINLINE void int4_decode(const uint32_t (&words)[ROWS][K], const float (&scales)[ROWS][K],
                         U (&output)[ROWS][K][P]) {
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int k = 0; k < K; ++k) int4_decode(words[i][k], scales[i][k], output[i][k]);
}

// load, store, peers_load and all_gather (MEMORY's) on a quantized buffer: the same places, a word
// and a scale a pack, the sizes the tensor's in elements. Every word and scale in flight before the
// first is decoded.
template <int THREADS_M, int THREADS_N, typename U, int ROWS, int K, int P>
DINLINE void int4_load(const uint8_t* values_ptr, int64_t values_stride_m,
                       const __fp16* scales_ptr, int64_t scales_stride_m, int size_m,
                       int size_n, U (&output)[ROWS][K][P], int offs_m, int offs_n,
                       int row_step) {
  static_assert(THREADS_N % kInt4GroupPacks == 0, "a group is 4 lanes' packs");
  uint32_t words[ROWS][K];
  float scales[ROWS][K];
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    if (skips<ROWS, THREADS_M, THREADS_N>(size_m, offs_m, row_step, i)) continue;
    const int64_t row = row_of<THREADS_M, THREADS_N>(size_m, offs_m, row_step, i);
    const uint8_t* values_row = values_ptr + row * values_stride_m;
    const __fp16* scales_row = scales_ptr + row * scales_stride_m;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const int pack = pack_of<P, THREADS_N>(size_n, offs_n, k);
      words[i][k] = int4_word_load(values_row, pack);
      scales[i][k] = int4_scale_load(scales_row, pack);
    }
  }
  issued();
  int4_decode(words, scales, output);
}

template <int THREADS_M, int THREADS_N, int THREADS, int ROWS, int K>
DINLINE void int4_store(const uint32_t (&words)[ROWS][K], const float (&scales)[ROWS][K],
                        uint8_t* values_ptr, int64_t values_stride_m, __fp16* scales_ptr,
                        int64_t scales_stride_m, int size_m, int size_n, int offs_m,
                        int offs_n, int row_step) {
  constexpr int P = kInt4Group / kInt4GroupPacks;
  static_assert(THREADS_N % kInt4GroupPacks == 0, "a group is 4 lanes' packs");
  if (!holds<THREADS_M, THREADS_N, THREADS>()) return;
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    if (!row_live<THREADS_M, THREADS_N>(size_m, offs_m, row_step, i)) continue;
    const int64_t row = row_of<THREADS_M, THREADS_N>(size_m, offs_m, row_step, i);
    uint8_t* values_row = values_ptr + row * values_stride_m;
    __fp16* scales_row = scales_ptr + row * scales_stride_m;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      if (pack_mask<P, THREADS_N>(size_n, offs_n, k) == 0.0f) continue;
      const int pack = pack_of<P, THREADS_N>(size_n, offs_n, k);
      int4_word_store(values_row, pack, words[i][k]);
      int4_scale_store(scales_row, pack, scales[i][k]);
    }
  }
}

// Every peer's words and scales, the position once a pack in a vector register (peers_load's),
// each rank's word and scale an add of it; all in flight, none waited on.
template <int THREADS_M, int THREADS_N, int P, int WORLD, int ROWS, int K>
DINLINE void int4_peers_issue(const uint8_t* const (&values)[WORLD], int64_t values_stride_m,
                              const __fp16* const (&scales)[WORLD], int64_t scales_stride_m,
                              int size_m, int size_n, uint32_t (&words)[WORLD][ROWS][K],
                              float (&group_scales)[WORLD][ROWS][K], int offs_m, int offs_n,
                              int row_step) {
  static_assert(THREADS_N % kInt4GroupPacks == 0, "a group is 4 lanes' packs");
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    if (skips<ROWS, THREADS_M, THREADS_N>(size_m, offs_m, row_step, i)) continue;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const int64_t row = row_of<THREADS_M, THREADS_N>(size_m, offs_m, row_step, i);
      const int pack = pack_of<P, THREADS_N>(size_n, offs_n, k);
      // 32-BIT BYTE OFFSETS: a quantized buffer is ours, at most the box's (select's
      // exchange_sizes), so each rank's address is its base plus one offset register.
      uint32_t value_at = static_cast<uint32_t>(row * values_stride_m) +
                          static_cast<uint32_t>(pack) * kInt4WordBytes;
      uint32_t scale_at = static_cast<uint32_t>(
          (row * scales_stride_m + pack / kInt4GroupPacks) * kInt4ScaleBytes);
      asm volatile("" : "+v"(value_at));
      asm volatile("" : "+v"(scale_at));
#pragma unroll
      for (int r = 0; r < WORLD; ++r) {
        words[r][i][k] = *(const global_u32*)(values[r] + value_at);
        group_scales[r][i][k] = *(const global_f16*)(
            reinterpret_cast<const char*>(scales[r]) + scale_at);
      }
    }
  }
  issued();
}
template <int THREADS_M, int THREADS_N, typename U, int WORLD, int ROWS, int K, int P>
DINLINE void int4_peers_load(const uint8_t* const (&values)[WORLD], int64_t values_stride_m,
                             const __fp16* const (&scales)[WORLD], int64_t scales_stride_m,
                             int size_m, int size_n, U (&output)[WORLD][ROWS][K][P],
                             int offs_m, int offs_n, int row_step) {
  uint32_t words[WORLD][ROWS][K];
  float group_scales[WORLD][ROWS][K];
  int4_peers_issue<THREADS_M, THREADS_N, P>(values, values_stride_m, scales, scales_stride_m,
                                            size_m, size_n, words, group_scales, offs_m,
                                            offs_n, row_step);
#pragma unroll
  for (int r = 0; r < WORLD; ++r) int4_decode(words[r], group_scales[r], output[r]);
}

// EVERY PEER'S QUANTIZED PACKS SUMMED, peers_sum's: each element (q - 8) * scale landed in T, as a
// quantized load lands it, then added in fp32 in rank order. From the words, no pack of every
// peer made first: decoded whole, the push kernels held 172 VGPRs (102 unquantized; ISA
// 2026-10-15T22-47-01Z). WITH BF16, TWO RANKS' TERMS A STEP: landed together (one
// v_cvt_pk_bf16_f32) and added by one v_dot2c_f32_bf16 against (1, 1), where each was a
// conversion, a shift and an add.
template <typename T, int WORLD, int ROWS, int K, int P>
DINLINE void int4_peers_sum(const uint32_t (&words)[WORLD][ROWS][K],
                            const float (&group_scales)[WORLD][ROWS][K],
                            float (&output)[ROWS][K][P]) {
  const bf16x2 ones = __builtin_bit_cast(bf16x2, 0x3f803f80u);
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int k = 0; k < K; ++k) {
      float acc[P];
#pragma unroll
      for (int j = 0; j < P; ++j) acc[j] = 0.0f;
      int r = 0;
      if constexpr (kBf16<T>) {
#pragma unroll
        for (; r + 1 < WORLD; r += 2) {
          float a[P], b[P];
          int4_terms(words[r][i][k], group_scales[r][i][k], a);
          int4_terms(words[r + 1][i][k], group_scales[r + 1][i][k], b);
#pragma unroll
          for (int j = 0; j < P; ++j)
            acc[j] = __builtin_amdgcn_fdot2_f32_bf16(
                __builtin_convertvector(f32x2{a[j], b[j]}, bf16x2), ones, acc[j], false);
          int4_used_by(acc);
        }
      }
#pragma unroll
      for (; r < WORLD; ++r) {
        float a[P];
        int4_terms(words[r][i][k], group_scales[r][i][k], a);
#pragma unroll
        for (int j = 0; j < P; ++j) acc[j] += static_cast<float>(static_cast<T>(a[j]));
        int4_used_by(acc);
      }
#pragma unroll
      for (int j = 0; j < P; ++j) output[i][k][j] = acc[j];
    }
}

// Each pack from the rank owning its columns, slice_n a rank (all_gather's).
template <int WORLD, int THREADS_M, int THREADS_N, typename U, int ROWS, int K, int P>
DINLINE void int4_all_gather(const uint8_t* const* values_ptrs, int64_t values_stride_m,
                             const __fp16* const* scales_ptrs, int64_t scales_stride_m,
                             int size_m, int size_n, U (&output)[ROWS][K][P], int offs_m,
                             int offs_n, int slice_n) {
  static_assert(THREADS_N % kInt4GroupPacks == 0, "a group is 4 lanes' packs");
  uint32_t words[ROWS][K];
  float scales[ROWS][K];
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    if (skips<ROWS, THREADS_M, THREADS_N>(size_m, offs_m, 1, i)) continue;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const int col = pack_of<P, THREADS_N>(size_n, offs_n, k);
      const int owner = min(col / (slice_n / P), WORLD - 1);
      const int64_t row = row_of<THREADS_M, THREADS_N>(size_m, offs_m, 1, i);
      words[i][k] = int4_word_load(values_ptrs[owner] + row * values_stride_m, col);
      scales[i][k] = int4_scale_load(scales_ptrs[owner] + row * scales_stride_m, col);
    }
  }
  issued();
  int4_decode(words, scales, output);
}

// THE STAGED INPUT PUBLISHED, a stage's last step: every wave's stores done, the block met, then
// released, so the kernel's first barrier (Until::launched, which orders nothing) carries them to
// its peers. Their reads after it need no acquire: the exchange is uncached, a launch reads a staged
// line once, after the barrier's __syncthreads (no load hoisted above it), so nothing cached can
// be stale. An ordered first barrier for int4 alone is not a barrier the lint reads
// (barrier<Until::X, PARITY, WORLD>), and for every build would cost the none builds an s_barrier
// and two fences a launch.
DINLINE void publish_staged() {
  wait_stores();
  __syncthreads();
  fence<__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM>();
}

// RANK r'S BUFFER OF A QUANTIZED FORM (rank_of), or null with none, its table unread.
template <int QUANTIZATION, typename E>
DINLINE E* quantized_rank_of(E* const* ptrs, int r) {
  if constexpr (QUANTIZATION == Quantization::none)
    return nullptr;
  else
    return rank_of(ptrs, r);
}

// THE EXCHANGE'S MEMORY IN QUANTIZATION'S FORM, what the COLLECTIVES and the INTERFACE call: with
// none MEMORY's own function on the plain buffer (the quantized form's arguments unread), with int4
// the one above on its values and scales.
template <int THREADS_M, int THREADS_N, int QUANTIZATION, typename T, typename U,
          int ROWS, int K, int P>
DINLINE void exchanged_load(const T* input_ptr, int64_t input_stride_m,
                            const uint8_t* values_ptr, int64_t values_stride_m,
                            const __fp16* scales_ptr, int64_t scales_stride_m, int size_m,
                            int size_n, U (&output)[ROWS][K][P], int offs_m, int offs_n,
                            int row_step) {
  if constexpr (QUANTIZATION == Quantization::none)
    load<THREADS_M, THREADS_N>(input_ptr, input_stride_m, size_m, size_n, output, offs_m,
                               offs_n, row_step);
  else
    int4_load<THREADS_M, THREADS_N>(values_ptr, values_stride_m, scales_ptr,
                                    scales_stride_m, size_m, size_n, output, offs_m,
                                    offs_n, row_step);
}

template <int THREADS_M, int THREADS_N, int THREADS, int QUANTIZATION, typename T,
          typename U, int ROWS, int K, int P>
DINLINE void exchanged_store(const U (&input)[ROWS][K][P], T* output_ptr,
                             int64_t output_stride_m, uint8_t* values_ptr,
                             int64_t values_stride_m, __fp16* scales_ptr,
                             int64_t scales_stride_m, int size_m, int size_n, int offs_m,
                             int offs_n, int row_step) {
  if constexpr (QUANTIZATION == Quantization::none) {
    store<THREADS_M, THREADS_N, THREADS>(input, output_ptr, output_stride_m, size_m, size_n,
                                         offs_m, offs_n, row_step);
  } else {
    uint32_t words[ROWS][K];
    float scales[ROWS][K];
    int4_encode(input, words, scales);
    int4_store<THREADS_M, THREADS_N, THREADS>(words, scales, values_ptr, values_stride_m,
                                              scales_ptr, scales_stride_m, size_m, size_n,
                                              offs_m, offs_n, row_step);
  }
}

template <int THREADS_M, int THREADS_N, int QUANTIZATION, typename T, typename U,
          int WORLD, int ROWS, int K, int P>
DINLINE void exchanged_peers_load(const T* const (&input)[WORLD], int64_t input_stride_m,
                                  const uint8_t* const (&values)[WORLD],
                                  int64_t values_stride_m,
                                  const __fp16* const (&scales)[WORLD],
                                  int64_t scales_stride_m, int size_m, int size_n,
                                  U (&output)[WORLD][ROWS][K][P], int offs_m, int offs_n,
                                  int row_step) {
  if constexpr (QUANTIZATION == Quantization::none)
    peers_load<THREADS_M, THREADS_N>(input, input_stride_m, size_m, size_n, output, offs_m,
                                     offs_n, row_step);
  else
    int4_peers_load<THREADS_M, THREADS_N>(values, values_stride_m, scales, scales_stride_m,
                                          size_m, size_n, output, offs_m, offs_n,
                                          row_step);
}

template <int WORLD, int THREADS_M, int THREADS_N, int QUANTIZATION, typename T,
          typename U, int ROWS, int K, int P>
DINLINE void exchanged_all_gather(const T* const* scratch_ptrs, int64_t scratch_stride_m,
                                  const uint8_t* const* values_ptrs, int64_t values_stride_m,
                                  const __fp16* const* scales_ptrs,
                                  int64_t scales_stride_m, int size_m, int size_n,
                                  U (&output)[ROWS][K][P], int offs_m, int offs_n,
                                  int slice_n) {
  if constexpr (QUANTIZATION == Quantization::none)
    all_gather<WORLD, THREADS_M, THREADS_N>(scratch_ptrs, scratch_stride_m, size_m, size_n,
                                            output, offs_m, offs_n, slice_n);
  else
    int4_all_gather<WORLD, THREADS_M, THREADS_N>(values_ptrs, values_stride_m, scales_ptrs,
                                                 scales_stride_m, size_m, size_n, output,
                                                 offs_m, offs_n, slice_n);
}
}  // namespace impl

// =================================================================================================
// COLLECTIVES
//
// THE COLLECTIVES' STEPS, buffers to buffers: one tested and tuned way to do
// each, which the kernels call rather than write again (the fused AttnRes
// pull's own reduce-scatter took 1.6x the plain two-shot's cycles for its
// bytes, thread traces 2026-10-04T21-03-09Z and 21-26-11Z).
// =================================================================================================

// THE LOADS ARE THE COMMUNICATION: a rank's pointer points into its memory,
// mapped into this GPU's (PEERS), so loading reads it over the links; the sum
// and the store are local. The kernel syncs around these calls (the barriers):
// before, every peer's data is ready; between the reduce-scatter and the
// all-gather, every rank's sums are visible. RANKS ARE READ ROTATED, rank + w's
// w-th, so at any moment the GPUs read different peers; each range or slice is
// summed by one rank, so the order differing by rank is harmless.
//
// THE FLAT WALK (the plain two-shot): an elementwise [R, M, N] problem -- R
// ranks' copies of an [M, N] buffer, each at its own strides -- WALKED AS
// [R, 1, M*N]: the rows are only the buffer's layout, so the elements are
// numbered row-major (e = row * N + col, flat_at) and each rank's SHARE is one
// contiguous range of them, a TILE_N chunk a step. WHY: the links and HBM
// reward long reads from each peer. Cut a row at a time, a rank's share was its
// slice of every row -- 1.8 KB runs at N = 7168, the plain all-reduce +2%
// against the flat walk; 128-column tiles, 256 B runs, +3-7% (2026-10-11) --
// while a contiguous buffer's range is one run of the whole share. IT SERVES
// BADLY a strided buffer: its runs are its rows wherever it has them, and each
// pack's row and column are divided out. The walk is the same for every step
// and every rank, so a block reads in the all-gather what the same block on
// each rank wrote in the reduce-scatter. A RANK'S COPY OF A CHUNK IS LANES =
// THREADS / WORLD threads, whole waves (its rank's pointer one scalar select;
// every rank's held up front was ~20 scalar loads before the first barrier).
//
// CONTIGUOUS, whether an [*, n] buffer's rows are back to back so its range is
// one run, is a template parameter of the kernel, its instance chosen on the
// host from the strides as given. A compile-time choice, not a branch in the
// kernel: both paths in one kernel cost ~0.1 us a call at 128-512 tokens
// against the contiguous path alone (the strided code and its registers,
// 2026-10-12T01-17-18Z).

// WHICH WAY A REDUCE-SCATTER'S SUM GOES, and a caller's forcing (types.cuh): pull, this rank
// reads its peers and keeps the sum in its scratch; push, the sum is stored in every rank's
// (itself too), so after the sync each rank holds the whole sum locally.
enum class Direction : int { pull = 0, push = 1 };

namespace impl {
// THE SHARES of total elements over WORLD ranks, in whole packs (the last rank's short): a rank's
// slice, its first element and its length.
template <int WORLD, int P>
DINLINE int share_slice(int64_t total) {
  return static_cast<int>((total / P + WORLD - 1) / WORLD * P);
}
DINLINE int64_t share_first(int slice, int r) { return int64_t{r} * slice; }
DINLINE int share_len(int64_t total, int slice, int r) {
  return static_cast<int>(
      max(int64_t{0}, min(int64_t{slice}, total - share_first(slice, r))));
}

// flat_at on a pointer and its strides.
template <typename T>
DINLINE T* flat_at(T* ptr, int64_t stride_m, int64_t stride_n, int64_t e,
                   int n) {
  if (stride_m == int64_t{n} * stride_n) return ptr + e * stride_n;
  const int64_t row =
      e < (int64_t{1} << 31)
          ? int64_t{static_cast<uint32_t>(e) / static_cast<uint32_t>(n)}
          : e / n;
  return ptr + row * stride_m + (e - row * n) * stride_n;
}

// THIS THREAD'S COPY OF A CHUNK of the range from element `first` of an [*, input_size_n]
// buffer, the chunk's place size_n and offs_n: one run when its rows are back to back (the
// address a uniform base and a lane's offset, as load builds it), else each pack's own row and
// column; and stored, its first `len` elements.
template <bool CONTIGUOUS, int THREADS_N, typename T, int K, int P>
DINLINE void range_load(const T* input_ptr, int64_t input_stride_m,
                        int64_t input_stride_n, int input_size_n,
                        T (&output)[1][K][P], int64_t first, int size_n,
                        int offs_n) {
  using V = vec<T, P>;
  if constexpr (CONTIGUOUS) {
    // The last pack, the run's start, then the column: the order the two-shot's
    // instructions came out in before this interface (ISA 2026-10-15).
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const int last = size_n / P - 1;
      const V* run = reinterpret_cast<const V*>(input_ptr + first * input_stride_n);
      const int n = offs_n / P + static_cast<int>(threadIdx.x) % THREADS_N +
                    k * THREADS_N;
      const int col = n < size_n / P ? n : last;
      uint32_t offset = static_cast<uint32_t>(col) * uint32_t{sizeof(V)};
      asm volatile("" : "+v"(offset));
      held(output[0][k],
           pack_load(reinterpret_cast<const V*>(
               reinterpret_cast<const char*>(run) + offset)));
    }
    issued();
  } else {
#pragma unroll
    for (int k = 0; k < K; ++k)
      held(output[0][k],
           pack_load(reinterpret_cast<const V*>(flat_at(
               input_ptr, input_stride_m, input_stride_n,
               first + int64_t{pack_of<P, THREADS_N>(size_n, offs_n, k)} * P,
               input_size_n))));
    issued();
  }
}
template <bool CONTIGUOUS, int THREADS_N, typename T, int K, int P>
DINLINE void range_store(const T (&input)[1][K][P], T* output_ptr,
                         int64_t output_stride_m, int64_t output_stride_n,
                         int output_size_n, int64_t first, int size_n,
                         int offs_n, int len) {
  using V = vec<T, P>;
  V* run = reinterpret_cast<V*>(output_ptr + first * output_stride_n);
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int c =
        offs_n / P + static_cast<int>(threadIdx.x) % THREADS_N + k * THREADS_N;
    if (c >= len / P) continue;
    V* at;
    if constexpr (CONTIGUOUS)
      at = at_col<P, THREADS_N>(run, size_n, offs_n, k);
    else
      at = reinterpret_cast<V*>(flat_at(output_ptr, output_stride_m,
                                        output_stride_n, first + int64_t{c} * P,
                                        output_size_n));
    pack_store(at, pack(input[0][k]));
  }
}

// OVER THE RANKS, block_reduce_ranks on arrays: each LANES-thread copy's packs into LDS at its
// rank's place, consecutive threads' packs adjacent, then the first copy's threads sum each pack
// over the ranks in copy order, in fp32, rounded once to U there (rounded after the branch, the
// sums were phis of fp32 vectors and the LDS reads vectorized differently: ISA 2026-10-15).
template <int WORLD, int LANES, typename T, typename U, int K, int P>
DINLINE void block_sum_ranks(const T (&input)[1][K][P], U (&output)[1][K][P]) {
  using V = vec<T, P>;
  __shared__ V part[WORLD][K][LANES];
  const int mine = static_cast<int>(threadIdx.x) % LANES;
#pragma unroll
  for (int k = 0; k < K; ++k)
    part[static_cast<int>(threadIdx.x) / LANES][k][mine] = pack(input[0][k]);
  __syncthreads();
  if (holds<1, LANES, WORLD * LANES>()) {
#pragma unroll
    for (int k = 0; k < K; ++k) {
      float x[P];
      V p = part[0][k][mine];
#pragma unroll
      for (int j = 0; j < P; ++j)
        x[j] = static_cast<float>(reinterpret_cast<const T*>(&p)[j]);
#pragma unroll
      for (int r = 1; r < WORLD; ++r) {
        p = part[r][k][mine];
#pragma unroll
        for (int j = 0; j < P; ++j)
          x[j] = Sum::apply(x[j],
                            static_cast<float>(reinterpret_cast<const T*>(&p)[j]));
      }
#pragma unroll
      for (int j = 0; j < P; ++j) output[0][k][j] = static_cast<U>(x[j]);
    }
  }
  __syncthreads();  // before the next reduction writes `part`
}

// OVER THE ROWS OF THREADS, block_reduce_rows on arrays: THREADS_M rows of THREADS_N threads,
// one row each, meet in LDS as held; the first row's threads sum each column in row order, in
// fp32, rounded once to U there (as block_sum_ranks).
template <int THREADS_M, int THREADS_N, typename T, typename U, int K, int P>
DINLINE void block_sum_rows(const T (&input)[1][K][P], U (&output)[1][K][P]) {
  __shared__ T part[THREADS_M][THREADS_N * K * P];
  const int lane = static_cast<int>(threadIdx.x) % THREADS_N;
  const int lane_row = static_cast<int>(threadIdx.x) / THREADS_N;
#pragma unroll
  for (int k = 0; k < K; ++k)
#pragma unroll
    for (int j = 0; j < P; ++j)
      part[lane_row][(lane + k * THREADS_N) * P + j] = input[0][k][j];
  __syncthreads();
  if (holds<1, THREADS_N, THREADS_M * THREADS_N>()) {
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < P; ++j) {
        const int c = (lane + k * THREADS_N) * P + j;
        float x = static_cast<float>(part[0][c]);
#pragma unroll
        for (int r = 1; r < THREADS_M; ++r)
          x = Sum::apply(x, static_cast<float>(part[r][c]));
        output[0][k][j] = static_cast<U>(x);
      }
  }
  __syncthreads();  // before the next reduction writes `part`
}

// THE INT4 FLAT WALK: the shares in whole groups, A GROUP A THREAD, so its largest magnitude is
// the thread's own (no DPP), its values one 16-byte load and its scale one (a word a thread loaded
// each scale 4 times and held one 4-byte load in flight a rank). EACH BLOCK A CONTIGUOUS RANGE OF
// EVERY SHARE'S GROUPS, ceil(groups / grid) of them, THREADS a step: the same range on every rank
// in the stage, the reduce-scatter and the all-gather (what the per-block barrier needs), and every
// block busy at any size. `body(g, end)`: group g of a share, live below end.
template <int THREADS, typename BODY>
DINLINE void int4_for_each_group(int slice, BODY body) {
  const int groups = slice / kInt4Group;
  const int grid = static_cast<int>(gridDim.x);
  const int per_block = (groups + grid - 1) / grid;
  const int first = static_cast<int>(blockIdx.x) * per_block;
  const int end = min(first + per_block, groups);
  for (int g = first; g < end; g += THREADS) body(g + static_cast<int>(threadIdx.x), end);
}

// Group g of a quantized run (its values and scales from the run's first group), and stored.
DINLINE u32x4 int4_group_words_load(const uint8_t* values_run, int g) {
  return *(const global_u32x4*)at_offset(values_run, static_cast<uint32_t>(g) * 16);
}
DINLINE float int4_group_scale_load(const __fp16* scales_run, int g) {
  return *(const global_f16*)at_offset(scales_run, static_cast<uint32_t>(g) * kInt4ScaleBytes);
}
DINLINE void int4_group_store(const u32x4& words, float scale, uint8_t* values_run,
                              __fp16* scales_run, int g) {
  *(global_u32x4*)at_offset(values_run, static_cast<uint32_t>(g) * 16) = words;
  *(global_f16*)at_offset(scales_run, static_cast<uint32_t>(g) * kInt4ScaleBytes) =
      static_cast<__fp16>(scale);
}

// Group g's first element in an [*, size_n] buffer, from element `first` (a whole group, so the
// group is in one row): one run when its rows are back to back (range_load's).
template <bool CONTIGUOUS, typename T>
DINLINE T* int4_group_at(T* ptr, int64_t stride_m, int64_t stride_n, int size_n, int64_t first,
                         int g) {
  if constexpr (CONTIGUOUS)
    return at_offset(ptr + first * stride_n,
                     static_cast<uint32_t>(g) * (kInt4Group * sizeof(T)));
  else
    return flat_at(ptr, stride_m, stride_n, first + int64_t{g} * kInt4Group, size_n);
}

// THE INT4 STAGE: every rank's share of this rank's input, a group a thread, quantized into its
// staged values and scales (ours, rows back to back), then published.
template <int WORLD, int THREADS, bool CONTIGUOUS, typename T>
DINLINE void int4_flat_stage(const T* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
                             int input_size_m, int input_size_n, uint8_t* staging_values_ptr,
                             __fp16* staging_scales_ptr) {
  constexpr int P = traits<T>::N;
  using V = vec<T, P>;
  const int64_t total = int64_t{input_size_m} * input_size_n;
  const int slice = share_slice<WORLD, kInt4Group>(total);
  int4_for_each_group<THREADS>(slice, [&](int g, int end) {
    // TWO RANKS' GROUPS IN FLIGHT: unrolled whole, every rank's 4 loads were hoisted together,
    // 128 VGPRs of input, 256 in all, spilling.
#pragma unroll 2
    for (int r = 0; r < WORLD; ++r) {
      const int groups = share_len(total, slice, r) / kInt4Group;
      if (groups == 0) continue;  // a late rank's empty share (every block alike)
      const int64_t first = share_first(slice, r);
      const V* group = reinterpret_cast<const V*>(int4_group_at<CONTIGUOUS>(
          input_ptr, input_stride_m, input_stride_n, input_size_n, first, min(g, groups - 1)));
      T x[kInt4GroupPacks][P];
#pragma unroll
      for (int k = 0; k < kInt4GroupPacks; ++k) held(x[k], pack_load(group + k));
      u32x4 words;
      float scale;
      int4_encode_group(x, words, scale);
      if (g < end && g < groups)
        int4_group_store(words, scale, staging_values_ptr + first / 2,
                         staging_scales_ptr + first / kInt4Group, g);
    }
  });
  publish_staged();
}

// THE INT4 REDUCE-SCATTER: this rank's share, a group a thread, every rank's staged group in
// flight at once (rotated rank order), summed in fp32 in registers (int4_peers_sum: each term
// landed in T), quantized and stored in this rank's scratch (the share's own run). Through LDS, a
// rank's copy a wave, one wave in 8 summed and quantized while the others waited at the barrier.
template <int WORLD, int THREADS, typename T>
DINLINE void int4_flat_reduce_scatter(const uint8_t* const* input_values_ptrs,
                                      const __fp16* const* input_scales_ptrs,
                                      uint8_t* const* scratch_values_ptrs,
                                      __fp16* const* scratch_scales_ptrs, int64_t total,
                                      int rank) {
  constexpr int P = traits<T>::N;
  const int slice = share_slice<WORLD, kInt4Group>(total);
  const int groups = share_len(total, slice, rank) / kInt4Group;
  if (groups == 0) return;  // a late rank's share is empty at few groups (every block alike)
  const int64_t first = share_first(slice, rank);
  const uint8_t* values[WORLD];
  const __fp16* scales[WORLD];
#pragma unroll
  for (int w = 0; w < WORLD; ++w) {
    values[w] = rank_of(input_values_ptrs, (rank + w) % WORLD) + first / 2;
    scales[w] = rank_of(input_scales_ptrs, (rank + w) % WORLD) + first / kInt4Group;
  }
  uint8_t* const own_values = rank_of(scratch_values_ptrs, rank);
  __fp16* const own_scales = rank_of(scratch_scales_ptrs, rank);
  int4_for_each_group<THREADS>(slice, [&](int g, int end) {
    const int at = min(g, groups - 1);
    uint32_t words[WORLD][1][kInt4GroupPacks];
    float group_scales[WORLD][1][kInt4GroupPacks];
#pragma unroll
    for (int w = 0; w < WORLD; ++w) {
      const u32x4 v = int4_group_words_load(values[w], at);
      const float scale = int4_group_scale_load(scales[w], at);
#pragma unroll
      for (int k = 0; k < kInt4GroupPacks; ++k) {
        words[w][0][k] = v[k];
        group_scales[w][0][k] = scale;
      }
    }
    issued();
    float sum[1][kInt4GroupPacks][P];
    int4_peers_sum<T>(words, group_scales, sum);
    u32x4 out;
    float scale;
    int4_encode_group(sum[0], out, scale);
    if (g < end && g < groups) int4_group_store(out, scale, own_values, own_scales, g);
  });
}

// THE INT4 ALL-GATHER: every rank's group g of its summed share, all in flight, dequantized to T
// and stored at its place in the output.
template <int WORLD, int THREADS, bool CONTIGUOUS, typename T>
DINLINE void int4_flat_all_gather(uint8_t* const* scratch_values_ptrs,
                                  __fp16* const* scratch_scales_ptrs, T* output_ptr,
                                  int64_t output_stride_m, int64_t output_stride_n,
                                  int output_size_m, int output_size_n, int rank) {
  constexpr int P = traits<T>::N;
  using V = vec<T, P>;
  const int64_t total = int64_t{output_size_m} * output_size_n;
  const int slice = share_slice<WORLD, kInt4Group>(total);
  const uint8_t* values[WORLD];
  const __fp16* scales[WORLD];
#pragma unroll
  for (int w = 0; w < WORLD; ++w) {
    values[w] = rank_of(scratch_values_ptrs, (rank + w) % WORLD);
    scales[w] = rank_of(scratch_scales_ptrs, (rank + w) % WORLD);
  }
  int4_for_each_group<THREADS>(slice, [&](int g, int end) {
    u32x4 words[WORLD];
    float group_scales[WORLD];
#pragma unroll
    for (int w = 0; w < WORLD; ++w) {
      const int groups = share_len(total, slice, (rank + w) % WORLD) / kInt4Group;
      const int at = max(min(g, groups - 1), 0);
      words[w] = int4_group_words_load(values[w], at);
      group_scales[w] = int4_group_scale_load(scales[w], at);
    }
    issued();
#pragma unroll
    for (int w = 0; w < WORLD; ++w) {
      const int r = (rank + w) % WORLD;
      const int groups = share_len(total, slice, r) / kInt4Group;
      if (!(g < end && g < groups)) continue;
      V* group = reinterpret_cast<V*>(int4_group_at<CONTIGUOUS>(
          output_ptr, output_stride_m, output_stride_n, output_size_n, share_first(slice, r),
          g));
#pragma unroll
      for (int k = 0; k < kInt4GroupPacks; ++k) {
        T x[P];
        int4_decode(words[w][k], group_scales[w], x);
        pack_store(group + k, pack(x));
      }
    }
  });
}

// THE FLAT WALK'S REDUCE-SCATTER: this rank's share, a TILE_N chunk a step, every rank's copy in
// one round trip, summed over the ranks through LDS, rounded once to T, stored in this rank's
// scratch; with int4 the int4 walk's above.
template <int WORLD, int THREADS, int TILE_N, bool CONTIGUOUS, int QUANTIZATION,
          typename T>
DINLINE void reduce_scatter(const T* const* input_ptrs, int64_t input_stride_m,
                            int64_t input_stride_n, int input_size_m,
                            int input_size_n, const uint8_t* const* input_values_ptrs,
                            const __fp16* const* input_scales_ptrs,
                            T* const* scratch_ptrs, uint8_t* const* scratch_values_ptrs,
                            __fp16* const* scratch_scales_ptrs, int rank) {
  if constexpr (QUANTIZATION == Quantization::int4) {
    int4_flat_reduce_scatter<WORLD, THREADS, T>(input_values_ptrs, input_scales_ptrs,
                                                scratch_values_ptrs, scratch_scales_ptrs,
                                                int64_t{input_size_m} * input_size_n, rank);
  } else {
    constexpr int LANES = THREADS / WORLD;
    constexpr int P = traits<T>::N;
    constexpr int K = TILE_N / (P * LANES);
    static_assert(LANES * WORLD == THREADS, "a block is a group of threads a rank");
    static_assert(LANES % kWaveSize == 0, "a rank's copy is whole waves");
    static_assert(K * P * LANES == TILE_N, "a chunk is whole packs a thread");
    T* const scratch = rank_of(scratch_ptrs, rank);
    const int64_t total = int64_t{input_size_m} * input_size_n;
    const int slice = share_slice<WORLD, P>(total);
    const int chunks = (slice + TILE_N - 1) / TILE_N;
    if (share_len(total, slice, rank) == 0)
      return;  // a late rank's share is empty at few packs (every block alike)
    block_for_each_tile(1, chunks, [&](int, int chunk) {
      const int len = share_len(total, slice, rank);
      const int r = (rank + static_cast<int>(threadIdx.x) / LANES) % WORLD;
      T got[1][K][P];
      // 1. Every rank's copy of this chunk of this rank's share, in one round trip.
      range_load<CONTIGUOUS, LANES>(rank_of(input_ptrs, r), input_stride_m, input_stride_n,
                                    input_size_n, got, share_first(slice, rank), len,
                                    chunk * TILE_N);
      // 2. Summed over the ranks, in rotated rank order, through LDS, rounded once.
      T rounded[1][K][P];
      block_sum_ranks<WORLD, LANES>(got, rounded);
      // 3. Stored in this rank's scratch.
      store<1, LANES, THREADS>(rounded, scratch, 0, 1, len, 0, chunk * TILE_N, 1);
    });
  }
}

// THE FLAT WALK'S ALL-GATHER: every rank's summed share out of its scratch, at its place in the
// output, a TILE_N chunk a step; with int4 the int4 walk's above.
template <int WORLD, int THREADS, int TILE_N, bool CONTIGUOUS, int QUANTIZATION,
          typename T>
DINLINE void all_gather(T* const* scratch_ptrs, uint8_t* const* scratch_values_ptrs,
                        __fp16* const* scratch_scales_ptrs, T* output_ptr,
                        int64_t output_stride_m, int64_t output_stride_n,
                        int output_size_m, int output_size_n, int rank) {
  if constexpr (QUANTIZATION == Quantization::int4) {
    int4_flat_all_gather<WORLD, THREADS, CONTIGUOUS>(scratch_values_ptrs, scratch_scales_ptrs,
                                                     output_ptr, output_stride_m,
                                                     output_stride_n, output_size_m,
                                                     output_size_n, rank);
  } else {
    constexpr int LANES = THREADS / WORLD;
    constexpr int P = traits<T>::N;
    constexpr int K = TILE_N / (P * LANES);
    static_assert(LANES * WORLD == THREADS, "a block is a group of threads a rank");
    static_assert(LANES % kWaveSize == 0, "a rank's copy is whole waves");
    const int64_t total = int64_t{output_size_m} * output_size_n;
    const int slice = share_slice<WORLD, P>(total);
    block_for_each_tile(1, (slice + TILE_N - 1) / TILE_N, [&](int, int chunk) {
      const int r = (rank + static_cast<int>(threadIdx.x) / LANES) % WORLD;
      T got[1][K][P];
      // 1. Rank r's summed chunk, in one round trip.
      load<1, LANES>(rank_of(scratch_ptrs, r), 0, 1, slice, got, 0, chunk * TILE_N, 1);
      // 2. Stored where rank r's share goes.
      range_store<CONTIGUOUS, LANES>(got, output_ptr, output_stride_m, output_stride_n,
                                     output_size_n, share_first(slice, r), slice,
                                     chunk * TILE_N, share_len(total, slice, r));
    });
  }
}

// STAGED, BEFORE THE REDUCE-SCATTER: this rank's input, every rank's share of it, into its
// staging (ours, rows back to back), each block the chunks that rank's same block reads.
template <int WORLD, int THREADS, int TILE_N, bool CONTIGUOUS, typename T>
DINLINE void stage(const T* input_ptr, int64_t input_stride_m,
                   int64_t input_stride_n, int input_size_m, int input_size_n,
                   T* staging_ptr, int64_t staging_stride_m,
                   int64_t staging_stride_n, int staging_size_m,
                   int staging_size_n, int rank) {
  constexpr int LANES = THREADS / WORLD;
  constexpr int P = traits<T>::N;
  constexpr int K = TILE_N / (P * LANES);
  static_assert(LANES * WORLD == THREADS, "a block is a group of threads a rank");
  const int64_t total = int64_t{input_size_m} * input_size_n;
  const int slice = share_slice<WORLD, P>(total);
  block_for_each_tile(1, (slice + TILE_N - 1) / TILE_N, [&](int, int chunk) {
    const int r = (rank + static_cast<int>(threadIdx.x) / LANES) % WORLD;
    const int len = share_len(total, slice, r);
    if (len == 0) return;
    T part[1][K][P];
    range_load<CONTIGUOUS, LANES>(input_ptr, input_stride_m, input_stride_n,
                                  input_size_n, part, share_first(slice, r), len,
                                  chunk * TILE_N);
    range_store<true, LANES>(part, staging_ptr, staging_stride_m,
                             staging_stride_n, staging_size_n,
                             share_first(slice, r), len, chunk * TILE_N, len);
  });
}

// THE IN-PLACE TWO-SHOT'S STAGE, in QUANTIZATION's form: with none nothing (the peers read the
// input where it is); with int4 the int4 walk's.
template <int WORLD, int THREADS, int TILE_N, bool CONTIGUOUS, int QUANTIZATION,
          typename T>
DINLINE void stage(const T* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
                   int input_size_m, int input_size_n, uint8_t* staging_values_ptr,
                   __fp16* staging_scales_ptr, int rank) {
  if constexpr (QUANTIZATION == Quantization::int4)
    int4_flat_stage<WORLD, THREADS, CONTIGUOUS>(input_ptr, input_stride_m, input_stride_n,
                                                input_size_m, input_size_n,
                                                staging_values_ptr, staging_scales_ptr);
}

// THE PER-ROW TWO-SHOTS' STAGE, in QUANTIZATION's form: with none nothing; with int4 this block's
// rows of the input, whole, quantized into this rank's staged values and scales at the same place,
// then published. THE ROWS ARE THE ONES THIS BLOCK'S TWIN ON EVERY RANK READS: in each band of
// band_size_m rows (a row split's slice of rows; else all of them), tiles of TILE_M rows blockIdx.x,
// blockIdx.x + gridDim.x, ... (the reduce-scatters' walk). kTiles tiles a step, every load in
// flight before the first is quantized.
template <int THREADS, int TILE_M, int QUANTIZATION, typename T>
DINLINE void stage_rows(const T* input_ptr, int64_t input_stride_m, int input_size_m,
                        int input_size_n, uint8_t* values_ptr, int64_t values_stride_m,
                        __fp16* scales_ptr, int64_t scales_stride_m, int band_size_m) {
  if constexpr (QUANTIZATION == Quantization::int4) {
    constexpr int P = traits<T>::N;
    constexpr int K = 1;  // a pack a row a thread, rows in flight instead
    constexpr int kTiles = TILE_M < 4 ? 4 / TILE_M : 1;
    for (int band = 0; band < input_size_m; band += band_size_m) {
      const int rows = min(band_size_m, input_size_m - band);
      const int tiles = (rows + TILE_M - 1) / TILE_M;
      const T* band_input = input_ptr + int64_t{band} * input_stride_m;
      uint8_t* band_values = values_ptr + int64_t{band} * values_stride_m;
      __fp16* band_scales = scales_ptr + int64_t{band} * scales_stride_m;
      for (int t = blockIdx.x; t < tiles; t += kTiles * gridDim.x)
        for (int c = 0; c < input_size_n; c += THREADS * P * K) {
          T x[kTiles][TILE_M][K][P];
#pragma unroll
          for (int s = 0; s < kTiles; ++s)
            load<1, THREADS>(band_input, input_stride_m, rows, input_size_n, x[s],
                             (t + s * static_cast<int>(gridDim.x)) * TILE_M, c, 1);
#pragma unroll
          for (int s = 0; s < kTiles; ++s) {
            uint32_t words[TILE_M][K];
            float scales[TILE_M][K];
            int4_encode(x[s], words, scales);
            int4_store<1, THREADS, THREADS>(
                words, scales, band_values, values_stride_m, band_scales, scales_stride_m,
                rows, input_size_n, (t + s * static_cast<int>(gridDim.x)) * TILE_M, c, 1);
          }
        }
    }
    publish_staged();
  }
}

// THE PER-ROW REDUCE-SCATTER THROUGH LDS (the pull AttnRes's): this rank's columns of each row,
// a column range a rank (equal slices of whole packs, a late rank's short or empty), block b
// taking tiles of TILE_M rows b, b + grid, ...; a chunk of a row at a time, WORLD rows of
// LANES threads, row w rank rank + w's copy (ROTATED: in rank order every GPU read rank 0 first,
// and the reduce-scatter took 1.6x the plain two-shot's cycles for its bytes, thread traces
// 2026-10-04T21-03-09Z, 21-26-11Z). A ROW OF THREADS' RANK DIFFERS ACROSS THE WAVE (LANES may be
// half one), so each lane reads its own from the table (an array of every rank's, indexed at run
// time, was scratch: 136 B). Summed over the rows through LDS into this rank's scratch.
template <int WORLD, int THREADS, int TILE_M, int QUANTIZATION, typename T>
DINLINE void reduce_scatter_lds(const T* const* input_ptrs,
                                int64_t input_stride_m, int input_size_m, int input_size_n,
                                const uint8_t* const* input_values_ptrs,
                                int64_t input_values_stride_m,
                                const __fp16* const* input_scales_ptrs,
                                int64_t input_scales_stride_m, T* const* scratch_ptrs,
                                int64_t scratch_stride_m, uint8_t* const* scratch_values_ptrs,
                                int64_t scratch_values_stride_m,
                                __fp16* const* scratch_scales_ptrs,
                                int64_t scratch_scales_stride_m, int rank) {
  constexpr int P = traits<T>::N;
  constexpr int LANES = THREADS / WORLD;
  static_assert(LANES * WORLD == THREADS, "a block is a row of threads a rank");
  using V = vec<T, P>;
  using Summed = std::conditional_t<QUANTIZATION == Quantization::none, T, float>;
  const int packs = input_size_n / P;  // the row, in packs (n is whole packs)
  const int slice = (packs + WORLD - 1) / WORLD;
  const int col0 = min(rank * slice, packs);
  const int own_packs = max(0, min(slice, packs - col0));
  T* const own_scratch = rank_of(scratch_ptrs, rank);
  uint8_t* const own_values = quantized_rank_of<QUANTIZATION>(scratch_values_ptrs, rank);
  __fp16* const own_scales = quantized_rank_of<QUANTIZATION>(scratch_scales_ptrs, rank);
  const int end = (col0 + own_packs) * P;  // this rank's columns end, in elements
  const int tiles = (input_size_m + TILE_M - 1) / TILE_M;
  for (int t = blockIdx.x; t < tiles; t += gridDim.x)
    for (int row = t * TILE_M; row < min(t * TILE_M + TILE_M, input_size_m);
         ++row)
      for (int c = col0 * P; c < end; c += LANES * P) {
        // 1. Every rank's copy of the chunk, a row of threads each, in one round trip.
        const int r = (rank + static_cast<int>(threadIdx.x) / LANES) % WORLD;
        T got[1][1][P];
        if constexpr (QUANTIZATION == Quantization::none) {
          const V* at = reinterpret_cast<const V*>(input_ptrs[r] + row * input_stride_m);
          held(got[0][0], pack_load(at_col<P, LANES>(at, end, c, 0)));
          issued();
        } else {
          int4_load<1, LANES>(input_values_ptrs[r], input_values_stride_m,
                              input_scales_ptrs[r], input_scales_stride_m, input_size_m,
                              end, got, row, c, 1);
        }
        // 2. Summed over the rows (the ranks), in row order, through LDS, rounded once.
        Summed rounded[1][1][P];
        block_sum_rows<WORLD, LANES>(got, rounded);
        // 3. Stored in this rank's slice.
        constexpr bool kInt4 = QUANTIZATION == Quantization::int4;
        exchanged_store<1, LANES, THREADS, QUANTIZATION>(
            rounded, own_scratch + row * scratch_stride_m, end,
            kInt4 ? own_values + row * scratch_values_stride_m : nullptr, 0,
            kInt4 ? own_scales + row * scratch_scales_stride_m : nullptr, 0, 1, end, 0, c,
            1);
      }
}

// THE PER-ROW REDUCE-SCATTER IN REGISTERS (the push kernels'): this rank's size_n columns
// from offs_n of block b's block_size_m rows b, b + grid, ... (the caller's, computed at kernel
// entry: after the barrier the grid division went to the VALU and the loops were exec-masked). A row of threads as wide as a rank's slice of a TILE_N row (in whole waves), a pack a
// thread, so a row's slice is one round trip (a wave a row took two at 7168, 0.8 us at 16-32
// tokens), and as many rows at once as the block holds; a pack a thread, not a slice's worth:
// every peer's packs of a whole slice were 32 at 16384, spilling. Each thread sums every
// rank's pack in registers in rank order, no LDS; push stores the sum into every rank's
// scratch, pull into this rank's.
template <int WORLD, int THREADS, int TILE_N, Direction DIRECTION, int QUANTIZATION,
          typename T>
DINLINE void reduce_scatter_registers(
    const T* const* input_ptrs, int64_t input_stride_m, int input_size_m,
    const uint8_t* const* input_values_ptrs, int64_t input_values_stride_m,
    const __fp16* const* input_scales_ptrs, int64_t input_scales_stride_m,
    T* const* scratch_ptrs, int64_t scratch_stride_m, uint8_t* const* scratch_values_ptrs,
    int64_t scratch_values_stride_m, __fp16* const* scratch_scales_ptrs,
    int64_t scratch_scales_stride_m, int offs_n, int size_n, int block_size_m, int rank) {
  constexpr int P = traits<T>::N;
  constexpr int kSliceWaves =
      (TILE_N / WORLD / P + kWaveSize - 1) / kWaveSize * kWaveSize;
  constexpr int LANES = kSliceWaves < THREADS ? kSliceWaves : THREADS;
  constexpr int ROWS = THREADS / LANES;  // rows of threads at once
  constexpr bool kInt4 = QUANTIZATION == Quantization::int4;
  using Summed = std::conditional_t<kInt4, float, T>;
  const T* input[WORLD];
  peers_of(input_ptrs, input);
  const uint8_t* input_values[WORLD];
  const __fp16* input_scales[WORLD];
  if constexpr (kInt4) {
    peers_of(input_values_ptrs, input_values);
    peers_of(input_scales_ptrs, input_scales);
  }
  T* scratch[WORLD];
  uint8_t* scratch_values[WORLD];
  __fp16* scratch_scales[WORLD];
  if constexpr (DIRECTION == Direction::push) {
    peers_of(scratch_ptrs, scratch);
    if constexpr (kInt4) {
      peers_of(scratch_values_ptrs, scratch_values);
      peers_of(scratch_scales_ptrs, scratch_scales);
    }
  }
  if (size_n <= 0) return;  // a late rank owns no columns
  for (int q = 0; q < block_size_m; q += ROWS)
    for (int c = offs_n; c < offs_n + size_n; c += LANES * P) {
      const int offs_m = static_cast<int>(blockIdx.x + q * gridDim.x);
      float sum[1][1][P];
      if constexpr (!kInt4) {
        T peers[WORLD][1][1][P];
        peers_load<ROWS, LANES>(input, input_stride_m, input_size_m,
                                offs_n + size_n, peers, offs_m, c,
                                static_cast<int>(gridDim.x));
        peers_sum(peers, sum);
      } else {
        uint32_t peer_words[WORLD][1][1];
        float peer_scales[WORLD][1][1];
        int4_peers_issue<ROWS, LANES, P>(input_values, input_values_stride_m, input_scales,
                                         input_scales_stride_m, input_size_m,
                                         offs_n + size_n, peer_words, peer_scales, offs_m,
                                         c, static_cast<int>(gridDim.x));
        int4_peers_sum<T>(peer_words, peer_scales, sum);
      }
      Summed rounded[1][1][P];
      convert(sum, rounded);
      if constexpr (!kInt4) {
        if constexpr (DIRECTION == Direction::push) {
#pragma unroll
          for (int r = 0; r < WORLD; ++r)
            store<ROWS, LANES, THREADS>(rounded, scratch[r], scratch_stride_m,
                                        input_size_m, offs_n + size_n, offs_m, c,
                                        static_cast<int>(gridDim.x));
        } else {
          store<ROWS, LANES, THREADS>(rounded, rank_of(scratch_ptrs, rank),
                                      scratch_stride_m, input_size_m,
                                      offs_n + size_n, offs_m, c,
                                      static_cast<int>(gridDim.x));
        }
      } else {
        // INT4 QUANTIZES THE SUM ONCE, whichever ranks it is stored in.
        uint32_t words[1][1];
        float scales[1][1];
        int4_encode(rounded, words, scales);
        const auto store_in = [&](uint8_t* values, __fp16* group_scales) {
          int4_store<ROWS, LANES, THREADS>(words, scales, values, scratch_values_stride_m,
                                           group_scales, scratch_scales_stride_m,
                                           input_size_m, offs_n + size_n, offs_m, c,
                                           static_cast<int>(gridDim.x));
        };
        if constexpr (DIRECTION == Direction::push) {
#pragma unroll
          for (int r = 0; r < WORLD; ++r) store_in(scratch_values[r], scratch_scales[r]);
        } else {
          store_in(rank_of(scratch_values_ptrs, rank), rank_of(scratch_scales_ptrs, rank));
        }
      }
    }
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
// MEMORY: REGISTERS FROM AND TO BUFFERS
// ===============================================================================================

// A THREAD'S REGISTERS `T v[ROWS][K][P]` (P a pack, 16 bytes of the kernel's
// dtype), the buffer as pointer, strides and sizes, the place as ints. Thread t holds rows
// offs_m + (t / THREADS_N + i * THREADS_M) * row_step and packs offs_n / P + t % THREADS_N +
// k * THREADS_N; stride_n must be 1. One round trip; a row past size_m reads the last row.
// THE SIZES ARE WHERE ROWS AND PACKS CLAMP AND MASK, the tensor's: a buffer holding part of
// a tensor at the tensor's places (the scratch, a band of staging) passes the tensor's (or the
// band's) sizes, not its own allocation's.
template <int THREADS_M, int THREADS_N, typename T, int ROWS, int K, int P>
DINLINE void load(const T* input_ptr, int64_t input_stride_m,
                  int64_t input_stride_n, int input_size_m, int input_size_n,
                  T (&output)[ROWS][K][P], int offs_m, int offs_n,
                  int row_step = 1) {
  impl::load<THREADS_M, THREADS_N>(input_ptr, input_stride_m, input_size_m,
                                   input_size_n, output, offs_m, offs_n,
                                   row_step);
}

// Stores only the rows below output_size_m and the packs below output_size_n, from the threads
// of the layout (THREADS the block's, so a layout of all of it compares nothing).
template <int THREADS_M, int THREADS_N, int THREADS, typename T, int ROWS,
          int K, int P>
DINLINE void store(const T (&input)[ROWS][K][P], T* output_ptr,
                   int64_t output_stride_m, int64_t output_stride_n,
                   int output_size_m, int output_size_n, int offs_m, int offs_n,
                   int row_step = 1) {
  impl::store<THREADS_M, THREADS_N, THREADS>(input, output_ptr, output_stride_m,
                                    output_size_m, output_size_n, offs_m,
                                    offs_n, row_step);
}

// EVERY RANK'S POINTER of a buffer, read from its table where the kernel wants them: before the
// first barrier they are held in scalar registers and every tile's loads issue at once (the
// one-shot); held across a barrier the norm kernels spilled (the 8-pack build, 68 B: ISA
// 2026-09-30), so there they are read after it. peers_load takes the array as a table.
template <int WORLD, typename T>
DINLINE void peers_ptrs(T* const* ptrs, T* (&output)[WORLD]) {
  impl::peers_of(ptrs, output);
}

// ONE RANK'S POINTER of a buffer (this rank's own, most often), the same way: a scalar load.
template <typename T>
DINLINE T* peer_ptr(T* const* ptrs, int rank) {
  return impl::rank_of(ptrs, rank);
}

// Every rank's, in rank order, all in flight together: rank r's at input_ptrs[r] (the table,
// or the array peers_ptrs read).
template <int THREADS_M, int THREADS_N, typename T, int WORLD, int ROWS, int K,
          int P>
DINLINE void peers_load(const T* const* input_ptrs, int64_t input_stride_m,
                        int64_t input_stride_n, int input_size_m,
                        int input_size_n, T (&output)[WORLD][ROWS][K][P],
                        int offs_m, int offs_n, int row_step = 1) {
  const T* input[WORLD];
  impl::peers_of(input_ptrs, input);
  impl::peers_load<THREADS_M, THREADS_N>(input, input_stride_m, input_size_m,
                                         input_size_n, output, offs_m, offs_n,
                                         row_step);
}

// THE ALL-GATHER INTO REGISTERS: each pack from the rank owning its columns, slice_n columns a
// rank (the last rank's to the end), the owner picked in each lane. scratch_size_m and
// scratch_size_n clamp as load's do: the tensor's rows and columns, not the scratch's own.
template <int WORLD, int THREADS_M, int THREADS_N, typename T, int ROWS, int K,
          int P>
DINLINE void all_gather(const T* const* scratch_ptrs, int64_t scratch_stride_m,
                        int64_t scratch_stride_n, int scratch_size_m,
                        int scratch_size_n, T (&output)[ROWS][K][P], int offs_m,
                        int offs_n, int slice_n) {
  impl::all_gather<WORLD, THREADS_M, THREADS_N>(
      scratch_ptrs, scratch_stride_m, scratch_size_m, scratch_size_n, output,
      offs_m, offs_n, slice_n);
}

// A BUFFER OF THE EXCHANGE, in QUANTIZATION's form (the QUANTIZATION section): each function
// above with the buffer's quantized form after it, its values and its scales. With none it is that
// function on the plain buffer, the quantized form unread; with int4 the values and scales at the
// same places, a load dequantizing into its registers' dtype and a store quantizing its registers.
// The sizes that clamp and mask are the plain buffer's (the tensor's, in elements).
template <int THREADS_M, int THREADS_N, int QUANTIZATION, typename T, typename U,
          int ROWS, int K, int P>
DINLINE void load(const T* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
                  int input_size_m, int input_size_n, const uint8_t* input_values_ptr,
                  int64_t input_values_stride_m, int64_t input_values_stride_n,
                  int input_values_size_m, int input_values_size_n,
                  const __fp16* input_scales_ptr, int64_t input_scales_stride_m,
                  int64_t input_scales_stride_n, int input_scales_size_m,
                  int input_scales_size_n, U (&output)[ROWS][K][P], int offs_m, int offs_n,
                  int row_step = 1) {
  impl::exchanged_load<THREADS_M, THREADS_N, QUANTIZATION>(
      input_ptr, input_stride_m, input_values_ptr, input_values_stride_m, input_scales_ptr,
      input_scales_stride_m, input_size_m, input_size_n, output, offs_m, offs_n, row_step);
}

template <int THREADS_M, int THREADS_N, int THREADS, int QUANTIZATION, typename T,
          typename U, int ROWS, int K, int P>
DINLINE void store(const U (&input)[ROWS][K][P], T* output_ptr, int64_t output_stride_m,
                   int64_t output_stride_n, int output_size_m, int output_size_n,
                   uint8_t* output_values_ptr, int64_t output_values_stride_m,
                   int64_t output_values_stride_n, int output_values_size_m,
                   int output_values_size_n, __fp16* output_scales_ptr,
                   int64_t output_scales_stride_m, int64_t output_scales_stride_n,
                   int output_scales_size_m, int output_scales_size_n, int offs_m,
                   int offs_n, int row_step = 1) {
  impl::exchanged_store<THREADS_M, THREADS_N, THREADS, QUANTIZATION>(
      input, output_ptr, output_stride_m, output_values_ptr, output_values_stride_m,
      output_scales_ptr, output_scales_stride_m, output_size_m, output_size_n, offs_m,
      offs_n, row_step);
}

template <int THREADS_M, int THREADS_N, int QUANTIZATION, typename T, typename U,
          int WORLD, int ROWS, int K, int P>
DINLINE void peers_load(const T* const* input_ptrs, int64_t input_stride_m,
                        int64_t input_stride_n, int input_size_m, int input_size_n,
                        const uint8_t* const* input_values_ptrs,
                        int64_t input_values_stride_m, int64_t input_values_stride_n,
                        int input_values_size_m, int input_values_size_n,
                        const __fp16* const* input_scales_ptrs,
                        int64_t input_scales_stride_m, int64_t input_scales_stride_n,
                        int input_scales_size_m, int input_scales_size_n,
                        U (&output)[WORLD][ROWS][K][P], int offs_m, int offs_n,
                        int row_step = 1) {
  const T* input[WORLD];
  const uint8_t* values[WORLD];
  const __fp16* scales[WORLD];
  if constexpr (QUANTIZATION == Quantization::none) {
    impl::peers_of(input_ptrs, input);
  } else {
    impl::peers_of(input_values_ptrs, values);
    impl::peers_of(input_scales_ptrs, scales);
  }
  impl::exchanged_peers_load<THREADS_M, THREADS_N, QUANTIZATION>(
      input, input_stride_m, values, input_values_stride_m, scales, input_scales_stride_m,
      input_size_m, input_size_n, output, offs_m, offs_n, row_step);
}

template <int WORLD, int THREADS_M, int THREADS_N, int QUANTIZATION, typename T,
          typename U, int ROWS, int K, int P>
DINLINE void all_gather(const T* const* scratch_ptrs, int64_t scratch_stride_m,
                        int64_t scratch_stride_n, int scratch_size_m, int scratch_size_n,
                        const uint8_t* const* scratch_values_ptrs,
                        int64_t scratch_values_stride_m, int64_t scratch_values_stride_n,
                        int scratch_values_size_m, int scratch_values_size_n,
                        const __fp16* const* scratch_scales_ptrs,
                        int64_t scratch_scales_stride_m, int64_t scratch_scales_stride_n,
                        int scratch_scales_size_m, int scratch_scales_size_n,
                        U (&output)[ROWS][K][P], int offs_m, int offs_n, int slice_n) {
  impl::exchanged_all_gather<WORLD, THREADS_M, THREADS_N, QUANTIZATION>(
      scratch_ptrs, scratch_stride_m, scratch_values_ptrs, scratch_values_stride_m,
      scratch_scales_ptrs, scratch_scales_stride_m, scratch_size_m, scratch_size_n, output,
      offs_m, offs_n, slice_n);
}

// A row's scalar, stored once a block (thread 0's).
DINLINE void store_row_scalar(float value, float* output_ptr, int row) {
  impl::store_row_scalar(value, output_ptr, row);
}

// Every rank's scalar for `row`, `offset_bytes` into each rank's buffer.
template <int WORLD, typename T>
DINLINE void peers_load_row_scalars(const T* const* input_ptrs,
                                    float (&output)[WORLD],
                                    int64_t offset_bytes, int row) {
  impl::peers_load_row_scalars<WORLD>(input_ptrs, output, offset_bytes, row);
}

// ===============================================================================================
// ELEMENTWISE: FLOAT REGISTER MATH
// ===============================================================================================

// Element by element: b of a's rows or one row (a weight).
// Each element converted, rounded once.
template <typename T, typename U, int ROWS, int K, int P>
DINLINE void convert(const T (&input)[ROWS][K][P], U (&output)[ROWS][K][P]) {
  impl::convert(input, output);
}
// Every element zero, where a sum starts (registers start unset).
template <typename T, int ROWS, int K, int P>
DINLINE void zeros(T (&output)[ROWS][K][P]) {
  impl::zeros(output);
}
// In place, rounded to U and back: a reference's rounding between two steps.
template <typename U, int ROWS, int K, int P>
DINLINE void round(float (&values)[ROWS][K][P]) {
  impl::round<U>(values);
}
template <int ROWS, int ROWS_B, int K, int P>
DINLINE void add(const float (&a)[ROWS][K][P], const float (&b)[ROWS_B][K][P],
                 float (&output)[ROWS][K][P]) {
  impl::add(a, b, output);
}
template <int ROWS, int ROWS_B, int K, int P>
DINLINE void mul(const float (&a)[ROWS][K][P], const float (&b)[ROWS_B][K][P],
                 float (&output)[ROWS][K][P]) {
  impl::mul(a, b, output);
}
// Each row of a times its scale.
template <int ROWS, int K, int P>
DINLINE void mul(const float (&a)[ROWS][K][P], const float (&row_scale)[ROWS],
                 float (&output)[ROWS][K][P]) {
  impl::mul(a, row_scale, output);
}
template <int ROWS, int K, int P>
DINLINE void mul(const float (&a)[ROWS][K][P], float scale,
                 float (&output)[ROWS][K][P]) {
  impl::mul(a, scale, output);
}
// a * a_row_scale + b * b_row_scale as fused multiply-adds, b in any dtype.
template <typename B, int ROWS, int K, int P>
DINLINE void fma(const float (&a)[ROWS][K][P], const float (&a_row_scale)[ROWS],
                 const B (&b)[ROWS][K][P], const float (&b_row_scale)[ROWS],
                 float (&output)[ROWS][K][P]) {
  impl::fma(a, a_row_scale, b, b_row_scale, output);
}

// ===============================================================================================
// REDUCE: OVER THE RANKS, A WAVE OR THE BLOCK
// ===============================================================================================

// Every rank's registers summed, in rank order in fp32, left in fp32.
template <typename T, int WORLD, int ROWS, int K, int P>
DINLINE void peers_sum(const T (&input)[WORLD][ROWS][K][P],
                       float (&output)[ROWS][K][P]) {
  impl::peers_sum(input, output);
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

// ===============================================================================================
// DOT
// ===============================================================================================

// This thread's share of each row's dot of a and b (b of a's rows or one row, a weight), a
// row of threads THREADS_N wide, packs past size_n counting zero.
template <int THREADS_N, typename A, typename B, int ROWS, int ROWS_B, int K,
          int P>
DINLINE void partial_dot(const A (&a)[ROWS][K][P], const B (&b)[ROWS_B][K][P],
                         float (&output)[ROWS], int size_n, int offs_n) {
  impl::partial_dot<THREADS_N>(a, b, output, size_n, offs_n);
}

// ===============================================================================================
// SOFTMAX
// ===============================================================================================

// Logits folded into a running softmax: each logit's scale, and the old sum's.
template <int NUM_LOGITS>
DINLINE float thread_softmax_fold(OnlineSoftmax& softmax,
                                  const float (&logit)[NUM_LOGITS],
                                  float (&scale)[NUM_LOGITS]) {
  return impl::thread_softmax_fold<NUM_LOGITS>(softmax, logit, scale);
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

// THE FLAT WALK (the plain two-shot): the [m, n] tensor's elements numbered row-major and cut
// into WORLD ranges, a TILE_N chunk a step (COLLECTIVES says why). Reduce-scatter: this rank's
// range of every rank's input, summed in fp32, into its scratch. CONTIGUOUS: the input's rows
// back to back, the instance the host picks from its strides. QUANTIZATION: each buffer of the
// exchange followed by its quantized form (int4: the input's staged values and scales, `stage`'s;
// the sums' values and scales), read in place of it; the overload without them moves the buffers
// as they are (the staged build's).
template <int WORLD, int THREADS, int TILE_N, bool CONTIGUOUS, int QUANTIZATION,
          typename T>
DINLINE void reduce_scatter(
    const T* const* input_ptrs, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const uint8_t* const* input_values_ptrs,
    int64_t input_values_stride_m, int64_t input_values_stride_n, int input_values_size_m,
    int input_values_size_n, const __fp16* const* input_scales_ptrs,
    int64_t input_scales_stride_m, int64_t input_scales_stride_n, int input_scales_size_m,
    int input_scales_size_n, T* const* scratch_ptrs, int64_t scratch_stride_m,
    int64_t scratch_stride_n, int scratch_size_m, int scratch_size_n,
    uint8_t* const* scratch_values_ptrs, int64_t scratch_values_stride_m,
    int64_t scratch_values_stride_n, int scratch_values_size_m, int scratch_values_size_n,
    __fp16* const* scratch_scales_ptrs, int64_t scratch_scales_stride_m,
    int64_t scratch_scales_stride_n, int scratch_scales_size_m, int scratch_scales_size_n,
    int rank) {
  impl::reduce_scatter<WORLD, THREADS, TILE_N, CONTIGUOUS, QUANTIZATION>(
      input_ptrs, input_stride_m, input_stride_n, input_size_m, input_size_n,
      input_values_ptrs, input_scales_ptrs, scratch_ptrs, scratch_values_ptrs,
      scratch_scales_ptrs, rank);
}
template <int WORLD, int THREADS, int TILE_N, bool CONTIGUOUS, typename T>
DINLINE void reduce_scatter(const T* const* input_ptrs, int64_t input_stride_m,
                            int64_t input_stride_n, int input_size_m,
                            int input_size_n, T* const* scratch_ptrs,
                            int64_t scratch_stride_m, int64_t scratch_stride_n,
                            int scratch_size_m, int scratch_size_n, int rank) {
  impl::reduce_scatter<WORLD, THREADS, TILE_N, CONTIGUOUS, Quantization::none>(
      input_ptrs, input_stride_m, input_stride_n, input_size_m, input_size_n,
      static_cast<const uint8_t* const*>(nullptr), static_cast<const __fp16* const*>(nullptr),
      scratch_ptrs, static_cast<uint8_t* const*>(nullptr),
      static_cast<__fp16* const*>(nullptr), rank);
}
// All-gather: every rank's summed range, from its scratch (int4: its values, dequantized), at its
// place in the output.
template <int WORLD, int THREADS, int TILE_N, bool CONTIGUOUS, int QUANTIZATION,
          typename T>
DINLINE void all_gather(
    T* const* scratch_ptrs, int64_t scratch_stride_m, int64_t scratch_stride_n,
    int scratch_size_m, int scratch_size_n, uint8_t* const* scratch_values_ptrs,
    int64_t scratch_values_stride_m, int64_t scratch_values_stride_n,
    int scratch_values_size_m, int scratch_values_size_n,
    __fp16* const* scratch_scales_ptrs, int64_t scratch_scales_stride_m,
    int64_t scratch_scales_stride_n, int scratch_scales_size_m, int scratch_scales_size_n,
    T* output_ptr, int64_t output_stride_m, int64_t output_stride_n, int output_size_m,
    int output_size_n, int rank) {
  impl::all_gather<WORLD, THREADS, TILE_N, CONTIGUOUS, QUANTIZATION>(
      scratch_ptrs, scratch_values_ptrs, scratch_scales_ptrs, output_ptr, output_stride_m,
      output_stride_n, output_size_m, output_size_n, rank);
}
template <int WORLD, int THREADS, int TILE_N, bool CONTIGUOUS, typename T>
DINLINE void all_gather(T* const* scratch_ptrs, int64_t scratch_stride_m,
                        int64_t scratch_stride_n, int scratch_size_m,
                        int scratch_size_n, T* output_ptr,
                        int64_t output_stride_m, int64_t output_stride_n,
                        int output_size_m, int output_size_n, int rank) {
  impl::all_gather<WORLD, THREADS, TILE_N, CONTIGUOUS, Quantization::none>(
      scratch_ptrs, static_cast<uint8_t* const*>(nullptr),
      static_cast<__fp16* const*>(nullptr), output_ptr, output_stride_m, output_stride_n,
      output_size_m, output_size_n, rank);
}
// Staged, first: this rank's input, every rank's range of it, into its staging.
template <int WORLD, int THREADS, int TILE_N, bool CONTIGUOUS, typename T>
DINLINE void stage(const T* input_ptr, int64_t input_stride_m,
                   int64_t input_stride_n, int input_size_m, int input_size_n,
                   T* staging_ptr, int64_t staging_stride_m,
                   int64_t staging_stride_n, int staging_size_m,
                   int staging_size_n, int rank) {
  impl::stage<WORLD, THREADS, TILE_N, CONTIGUOUS>(
      input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
      staging_ptr, staging_stride_m, staging_stride_n, staging_size_m,
      staging_size_n, rank);
}
// THE IN-PLACE TWO-SHOT'S STAGE, before its first barrier: with none nothing (its peers read the
// input where it is); with int4 every rank's range of it quantized into this rank's staged values
// and scales (the input's quantized form the reduce-scatter reads), published to the peers.
template <int WORLD, int THREADS, int TILE_N, bool CONTIGUOUS, int QUANTIZATION,
          typename T>
DINLINE void stage(const T* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
                   int input_size_m, int input_size_n, uint8_t* staging_values_ptr,
                   int64_t staging_values_stride_m, int64_t staging_values_stride_n,
                   int staging_values_size_m, int staging_values_size_n,
                   __fp16* staging_scales_ptr, int64_t staging_scales_stride_m,
                   int64_t staging_scales_stride_n, int staging_scales_size_m,
                   int staging_scales_size_n, int rank) {
  impl::stage<WORLD, THREADS, TILE_N, CONTIGUOUS, QUANTIZATION>(
      input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
      staging_values_ptr, staging_scales_ptr, rank);
}

// THE PER-ROW TWO-SHOTS' STAGE, before the first barrier: with none nothing; with int4 this
// block's rows of the input, whole, quantized into this rank's staged values and scales at their
// place, published to the peers. Its rows are the ones this block's twin on every rank reads: in
// each band of band_size_m rows (a row split's slice of rows; else input_size_m), tiles of TILE_M
// rows blockIdx.x, blockIdx.x + gridDim.x, ...
template <int THREADS, int TILE_M, int QUANTIZATION, typename T>
DINLINE void stage_rows(const T* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
                        int input_size_m, int input_size_n, uint8_t* staging_values_ptr,
                        int64_t staging_values_stride_m, int64_t staging_values_stride_n,
                        int staging_values_size_m, int staging_values_size_n,
                        __fp16* staging_scales_ptr, int64_t staging_scales_stride_m,
                        int64_t staging_scales_stride_n, int staging_scales_size_m,
                        int staging_scales_size_n, int band_size_m) {
  impl::stage_rows<THREADS, TILE_M, QUANTIZATION>(
      input_ptr, input_stride_m, input_size_m, input_size_n, staging_values_ptr,
      staging_values_stride_m, staging_scales_ptr, staging_scales_stride_m, band_size_m);
}

// THE PER-ROW REDUCE-SCATTERS: each row's columns cut into WORLD ranges, this rank's summed over
// the ranks into scratch at the row's place. Two implementations, kept apart by name. _lds: WORLD
// rows of threads a chunk, ranks rotated, summed through LDS (block b's tiles of TILE_M rows).
// QUANTIZATION as the flat walk's: int4 reads every rank's staged values (stage_rows') and stores
// the sums' values and scales.
template <int WORLD, int THREADS, int TILE_M, int QUANTIZATION, typename T>
DINLINE void reduce_scatter_lds(
    const T* const* input_ptrs, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const uint8_t* const* input_values_ptrs,
    int64_t input_values_stride_m, int64_t input_values_stride_n, int input_values_size_m,
    int input_values_size_n, const __fp16* const* input_scales_ptrs,
    int64_t input_scales_stride_m, int64_t input_scales_stride_n, int input_scales_size_m,
    int input_scales_size_n, T* const* scratch_ptrs, int64_t scratch_stride_m,
    int64_t scratch_stride_n, int scratch_size_m, int scratch_size_n,
    uint8_t* const* scratch_values_ptrs, int64_t scratch_values_stride_m,
    int64_t scratch_values_stride_n, int scratch_values_size_m, int scratch_values_size_n,
    __fp16* const* scratch_scales_ptrs, int64_t scratch_scales_stride_m,
    int64_t scratch_scales_stride_n, int scratch_scales_size_m, int scratch_scales_size_n,
    int rank) {
  impl::reduce_scatter_lds<WORLD, THREADS, TILE_M, QUANTIZATION>(
      input_ptrs, input_stride_m, input_size_m, input_size_n, input_values_ptrs,
      input_values_stride_m, input_scales_ptrs, input_scales_stride_m, scratch_ptrs,
      scratch_stride_m, scratch_values_ptrs, scratch_values_stride_m, scratch_scales_ptrs,
      scratch_scales_stride_m, rank);
}
// _registers: a slice of lanes a row, every rank's pack in registers, summed there; push stores
// the sum in every rank's scratch (int4: quantized once, stored in every rank's values and
// scales). This rank's size_n columns from offs_n (in elements) of block b's block_size_m rows b,
// b + grid, ..., computed by the kernel at its entry.
template <int WORLD, int THREADS, int TILE_N, Direction DIRECTION, int QUANTIZATION,
          typename T>
DINLINE void reduce_scatter_registers(
    const T* const* input_ptrs, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const uint8_t* const* input_values_ptrs,
    int64_t input_values_stride_m, int64_t input_values_stride_n, int input_values_size_m,
    int input_values_size_n, const __fp16* const* input_scales_ptrs,
    int64_t input_scales_stride_m, int64_t input_scales_stride_n, int input_scales_size_m,
    int input_scales_size_n, T* const* scratch_ptrs, int64_t scratch_stride_m,
    int64_t scratch_stride_n, int scratch_size_m, int scratch_size_n,
    uint8_t* const* scratch_values_ptrs, int64_t scratch_values_stride_m,
    int64_t scratch_values_stride_n, int scratch_values_size_m, int scratch_values_size_n,
    __fp16* const* scratch_scales_ptrs, int64_t scratch_scales_stride_m,
    int64_t scratch_scales_stride_n, int scratch_scales_size_m, int scratch_scales_size_n,
    int offs_n, int size_n, int block_size_m, int rank) {
  impl::reduce_scatter_registers<WORLD, THREADS, TILE_N, DIRECTION, QUANTIZATION>(
      input_ptrs, input_stride_m, input_size_m, input_values_ptrs, input_values_stride_m,
      input_scales_ptrs, input_scales_stride_m, scratch_ptrs, scratch_stride_m,
      scratch_values_ptrs, scratch_values_stride_m, scratch_scales_ptrs,
      scratch_scales_stride_m, offs_n, size_n, block_size_m, rank);
}

// ===============================================================================================
// UTILS
// ===============================================================================================

}  // namespace hip_custom_communicator
