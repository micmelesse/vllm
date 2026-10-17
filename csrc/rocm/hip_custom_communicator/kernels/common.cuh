// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// COMMON: everything a kernel may call and the machine under it, in sections,
// each built on those above it: HARDWARE, BUILD, UTILS, TILES, MEMORY,
// ELEMENTWISE, REDUCE, DOT, PEERS, BARRIER, QUANTIZATION, COLLECTIVES,
// then THE INTERFACE, last because it passes to them all. Every function is at file scope
// (no helper namespace): the four ops (load, store, barrier, block_reduce), the INT4 codec
// (encode, decode), the two address functions (ptr_array_at, lane_ptr_array_at) and, until
// the kernel refactor's I1 and I3 remove them, the transitional helpers lint_kernels.sh lists.
//
// THE ARGUMENTS ARE A KERNEL'S, in its one order: the inputs, then the outputs,
// then the scalars (offs_m, offs_n, row_step, rank, ...). A buffer is a pointer,
// its strides, its sizes, in elements; stride_n is 1 (the host checks it; the
// packed loads need it). OUR PEER BUFFERS (the staging, the partials, the
// signals) are each a buffer of its own, at whatever address a peer's opening
// gave, no rank stride: a kernel takes each as a DEVICE ARRAY OF POINTERS of
// every rank's address of it with its size, `char* const* x_ptr_array, int
// x_ptr_array_size` (the world; written once at open by the Handle), and its
// strides and sizes; rank r's is `ptr_array_at(x_ptr_array, x_ptr_array_size, r)`
// (PEERS), one scalar load from the array. How they are allocated is the
// host's business (STANDARDS 3.2). EVERY ARRAY IS DATA WITH ITS SIZE (STANDARDS
// 3.2): a device array of pointers `x_ptr_array` (`T* const*`) with
// `x_ptr_array_size`, never a struct around it, and no function takes one
// without the other. THE CALLER'S INPUT, exact, is read by every rank IN
// PLACE through the INPUT TABLE (below); INT4, a kernel first quantizes it into this rank's
// inputs' INT4 values (`stage_rows`, an encoding) and its peers read them there; the plain
// all-reduce's pushed variant (forced) PUSHES it, each rank storing its input (or each rank's
// share of it) into the peers' inputs, each rank's INBOX, at its own slot (`store`'s every rank,
// `push_shares`), and sums its own inbox locally (`load`'s every slot, `reduce_slots`). Either way the
// first barrier is ordered. THE INPUT TABLE: `const T* const* inputs_ptr_array` with
// `inputs_ptr_array_size` (the world), a device array of every rank's input address (aiter's
// RankData*), rank r's input at inputs_ptr_array[r] at the inputs' strides. The kernel reads every rank's input where the table says, in place, and never
// knows whether it points at callers' tensors (a captured call's, registered at capture exit) or
// at every rank's inputs region (an eager call's, the host's copy): handle.cuh's INPUT TABLES.
// A thread's registers are a plain array the caller declares,
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
//   2. A thread's registers are loaded whole (load, every rank's or own: one round
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
//   PEERS       StatusField, kStatusFields, kMaxRanks, kMaxBlocks (no type
//               for the peers' memory: device arrays of pointers, each with
//               its size)
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

// =================================================================================================
// MEMORY
//
// THE READS AND WRITES, a thread's registers from and to buffers: a load and a store (one round
// trip), every rank's at once, the all-gather into registers, a row's scalar. How a pack is
// loaded (one 16-byte global instruction, issued together) is MEMORY's own (pack_load, pack_store).
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
DINLINE void issued() {
  asm volatile("" ::: "memory");
  __builtin_amdgcn_sched_barrier(0);
}

template <typename PACK>
DINLINE PACK pack_load(const PACK* p) {
  if constexpr (sizeof(PACK) == 16) {
    const u32x4 raw = *(const global_u32x4*)(p);
    PACK v;
    __builtin_memcpy(&v, &raw, 16);
    return v;
  } else {
    return *p;
  }
}

// A THREAD'S REGISTERS AND THEIR PLACE: `T v[ROWS][K][P]`, P a pack's elements; the place in a
// buffer as ints (its sizes, offs_m, offs_n, row_step) and the layout as template ints
// THREADS_M x THREADS_N. Thread t holds rows offs_m + (t / THREADS_N + i * THREADS_M) *
// row_step and packs offs_n / P + t % THREADS_N + k * THREADS_N.
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
// so every load stays in bounds (partial_dot masks past size_n with a float multiplied in, so
// nothing reading it branches (a load under a runtime `if` cannot be hoisted past the branch,
// and loads meant to be in flight together then wait one at a time).
template <int P, int THREADS_N>
DINLINE int pack_of(int size_n, int offs_n, int k) {
  const int n =
      offs_n / P + static_cast<int>(threadIdx.x) % THREADS_N + k * THREADS_N;
  return n < size_n / P ? n : size_n / P - 1;
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
  if constexpr (sizeof(PACK) == 16) {
    u32x4 raw;
    __builtin_memcpy(&raw, &v, 16);
    *(global_u32x4*)(p) = raw;
  } else {
    *p = v;
  }
}

// A ROW'S SCALAR (a norm's scale), one float a row: thread 0 stores it.
DINLINE void store_row_scalar(float value, float* output_ptr, int row) {
  if (threadIdx.x == 0)
    *(__attribute__((address_space(1))) float*)(output_ptr + row) = value;
}


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
template <typename T, typename U, int ROWS, int K, int P>
DINLINE void convert(const T (&input)[ROWS][K][P], U (&output)[ROWS][K][P]) {
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < P; ++j) output[i][k][j] = static_cast<U>(input[i][k][j]);
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

// THE OPERATIONS a wave or block reduction combines with.
struct Sum {
  static DINLINE float apply(float a, float b) { return a + b; }
  static constexpr float kIdentity = 0.0f;
};
struct Max {
  static DINLINE float apply(float a, float b) { return fmaxf(a, b); }
  static constexpr float kIdentity = -INFINITY;
};

// THE AXIS a block_reduce reduces over (template ints, as Quantization): `all`, the whole block;
// `m`, the THREADS_M rows of THREADS_N threads, elementwise a lane.
namespace ReduceAxis {
constexpr int all = 0;
constexpr int m = 1;
}  // namespace ReduceAxis

// THE BLOCK REDUCTION, one for every case: block_reduce<OP, AXIS, THREADS_M, THREADS_N>(input,
// output), OP Sum or Max, every value of [K][P] reduced on its own, in fp32. Input and output may
// be the same array (in place).
//
// AXIS all: over the whole block (THREADS_M * THREADS_N threads, THREADS_M 1), every thread left
// holding the results. Each wave reduces its values by DPP, then every thread combines the waves'
// partials in wave order (the same order in every thread, so every thread holds the same totals).
// One wave reducing the partials over all 64 lanes, then broadcasting them, was a second DPP pass a
// value, a second barrier and an LDS round trip to add 4 numbers at 256 threads: half of AttnRes's
// loop (ISA 2026-10-04T03-31-28Z). The trailing barrier keeps the next call from overwriting
// partials still being read. THE BLOCK'S WAVES ARE COMPILED IN: read from blockDim, every thread
// loaded the widest block's partials and kept 12 VGPRs at AttnRes's peak (ISA 2026-10-04T16-39-15Z).
//
// AXIS m: over the THREADS_M rows of THREADS_N threads (a rank's copy a row, in the collectives),
// each value of a lane with the same lane's of every row: each row's packs into LDS at its row's
// place, consecutive threads' packs adjacent, then the first row's threads combine each pack over
// the rows in row order, rounded once to U there; the other rows' output is unwritten (rounded
// after the branch, the sums were phis of fp32 vectors and the LDS reads vectorized differently:
// ISA 2026-10-15).
template <typename OP, int AXIS, int THREADS_M, int THREADS_N, typename T, typename U, int K,
          int P>
DINLINE void block_reduce(const T (&input)[K][P], U (&output)[K][P]) {
  static_assert(AXIS == ReduceAxis::all || AXIS == ReduceAxis::m, "AXIS is all or m");
  if constexpr (AXIS == ReduceAxis::all) {
    static_assert(THREADS_M == 1, "AXIS all is one row of the block's threads");
    constexpr int kValues = K * P;
    constexpr int kWaves = (THREADS_N + kWaveSize - 1) / kWaveSize;
    __shared__ float partial[kValues][kWaves];
    const int lane = threadIdx.x % kWaveSize;
    const int wave = threadIdx.x / kWaveSize;
    float v[kValues];
#pragma unroll
    for (int n = 0; n < kValues; ++n) v[n] = static_cast<float>(input[n / P][n % P]);
    // 1. OVER THE WAVE, every lane left holding the results, by DPP, as rocPRIM's and Composable
    // Kernel's wave reductions do: each step is a VALU operand from another lane. The butterfly
    // __shfl_xor it replaced compiled to ds_bpermute, an LDS round trip a step, six of them
    // waiting on each other (ISA 2026-09-30T20-43-18Z; stamps: 0.92 us for one block_reduce).
    //   swap neighbours, swap pairs (quad_perm [1,0,3,2], [2,3,0,1]): each quad holds its sum
    //   mirror in 8 lanes, then in 16 (row_half_mirror, row_mirror): each row holds its sum
    //   lane 15 into rows 1 and 3, lane 31 into rows 2 and 3 (row_bcast15, row_bcast31): lane 63
    //   holds the wave's, and readlane hands it to every lane.
    // A DPP STEP, `x` from the lane the control names: `row_mask` picks which 16-lane rows take it;
    // the others read OP's identity, which leaves them as they were. A FULL ROW MASK NEEDS NO
    // `old`: left undefined (mov_dpp), the compiler fuses the move into the add that uses it
    // (v_add_f32_dpp); an explicit identity kept a v_mov_b32_dpp and an add a step (ISA
    // 2026-10-04T03-07-41Z). The moved value first: only a VOP2's src0 takes DPP, so (dpp(x), x)
    // fuses each step into one v_add_f32_dpp where (x, dpp(x)) was a v_mov_b32_dpp and an add
    // (ISA 2026-10-04T02-22-41Z).
    static_assert(kWaveSize == 64, "the DPP sequence is for 64-lane waves");
    const auto bits = [](float x) { return __builtin_bit_cast(int, x); };
    const auto real = [](int x) { return __builtin_bit_cast(float, x); };
    constexpr int kIdentity = __builtin_bit_cast(int, OP::kIdentity);
#pragma unroll
    for (int n = 0; n < kValues; ++n) {
      float x = v[n];
      // quad_perm [1,0,3,2], quad_perm [2,3,0,1], row_half_mirror, row_mirror: every row.
      x = OP::apply(real(__builtin_amdgcn_mov_dpp(bits(x), 0xb1, 0xf, 0xf, false)), x);
      x = OP::apply(real(__builtin_amdgcn_mov_dpp(bits(x), 0x4e, 0xf, 0xf, false)), x);
      x = OP::apply(real(__builtin_amdgcn_mov_dpp(bits(x), 0x141, 0xf, 0xf, false)), x);
      x = OP::apply(real(__builtin_amdgcn_mov_dpp(bits(x), 0x140, 0xf, 0xf, false)), x);
      // row_bcast15 into rows 1, 3; row_bcast31 into rows 2, 3.
      x = OP::apply(real(__builtin_amdgcn_update_dpp(kIdentity, bits(x), 0x142, 0xa, 0xf, false)), x);
      x = OP::apply(real(__builtin_amdgcn_update_dpp(kIdentity, bits(x), 0x143, 0xc, 0xf, false)), x);
      v[n] = real(__builtin_amdgcn_readlane(bits(x), 63));
    }
    // 2. OVER THE WAVES, through LDS, in wave order.
    if (lane == 0) {
#pragma unroll
      for (int n = 0; n < kValues; ++n) partial[n][wave] = v[n];
    }
    __syncthreads();
#pragma unroll
    for (int n = 0; n < kValues; ++n) {
      float x = partial[n][0];
#pragma unroll
      for (int w = 1; w < kWaves; ++w) x = OP::apply(x, partial[n][w]);
      output[n / P][n % P] = static_cast<U>(x);
    }
    __syncthreads();
  } else {
    using V = vec<T, P>;
    __shared__ V part[THREADS_M][K][THREADS_N];
    const int mine = static_cast<int>(threadIdx.x) % THREADS_N;
#pragma unroll
    for (int k = 0; k < K; ++k)
      part[static_cast<int>(threadIdx.x) / THREADS_N][k][mine] = pack(input[k]);
    __syncthreads();
    // THE FIRST ROW OF THREADS HOLDS THE SUM: a compare only when there are several (threadIdx.x <
    // THREADS_N is not folded from the launch bounds: a compare and a branch around every store,
    // ISA 2026-10-15).
    if (THREADS_M == 1 || static_cast<int>(threadIdx.x) < THREADS_N) {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        float x[P];
        V p = part[0][k][mine];
#pragma unroll
        for (int j = 0; j < P; ++j) x[j] = static_cast<float>(reinterpret_cast<const T*>(&p)[j]);
#pragma unroll
        for (int r = 1; r < THREADS_M; ++r) {
          p = part[r][k][mine];
#pragma unroll
          for (int j = 0; j < P; ++j)
            x[j] = OP::apply(x[j], static_cast<float>(reinterpret_cast<const T*>(&p)[j]));
        }
#pragma unroll
        for (int j = 0; j < P; ++j) output[k][j] = static_cast<U>(x[j]);
      }
    }
    __syncthreads();  // before the next reduction writes `part`
  }
}

// =================================================================================================
// DOT
//
// THE DOT: a thread's share of a row's dot (a block_reduce finishes it).
// =================================================================================================

// This thread's share of dot(a, b) over each row, packs past size_n counting zero: the
// partials a block_reduce turns into the rows' dots (a sum of squares is partial_dot(x, x)). A
// one-row b (a weight) is every row's; a and b in any dtype, converted. A row of threads is
// THREADS_N wide (the block's, or a wave's).
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
      // Its mask 1 below size_n and 0 past it, multiplied in so nothing branches (pack_of).
      const int n = offs_n / P + static_cast<int>(threadIdx.x) % THREADS_N + k * THREADS_N;
      output[i] += (n < size_n / P ? 1.0f : 0.0f) * dk;
    }
  }
}

// =================================================================================================
// PEERS
//
// THE RANKS' MEMORY: every rank's buffers as plain tensors the reads and writes
// take, and the types a kernel takes them in. A RANK'S MEMORY, two places:
//   ours, ONE allocation a rank, its ARENA (handle.cuh), exchanged over legacy
//   HIP IPC at open, so every peer's is opened in this process at an address
//   of the runtime's choosing:
//     [ signals | inputs | partials ]  at the same offsets on every rank
//       inputs: a fused kernel's copied-in input, which its peers read; the
//         plain all-reduce's INBOX, [writer slot, ...], which its peers push
//         into and it sums locally
//       partials: a two-shot's sums, which its peers read (or push into)
//   each handed to a kernel as its own device array of every rank's address
//   of it (`x_ptr_array`, `x_ptr_array_size`) with its strides and its sizes
//   (rank r's is `ptr_array_at(x_ptr_array, x_ptr_array_size, r)`); the
//   caller's, one tensor a call:
//     [ input ]   never read by a peer: a fused kernel copies it into this
//                 rank's inputs (or, int4, quantized into the inputs' INT4
//                 values), the plain all-reduce pushes it into the peers'
//                 inboxes, before the first barrier, which is ordered
// A buffer is read or written by load / store, ordered
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

#ifndef HIP_CUSTOM_COMMUNICATOR_DEBUG
  #define HIP_CUSTOM_COMMUNICATOR_DEBUG 0
#endif
// AN INDEX INSIDE ITS BUFFER'S SIZE, checked in debug builds only (a release
// build passes the sizes and reads none): one outside is a programming error,
// not a peer's fault, so it traps.
DINLINE void debug_in_bounds(int64_t index, int64_t size) {
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG && (index < 0 || index >= size))
    __builtin_trap();
}

// ENTRY r OF A DEVICE ARRAY OF POINTERS (`x_ptr_array`, `x_ptr_array_size` entries: a region's
// addresses, a launch's inputs), `r` THE SAME ACROSS THE WAVE (a constant, or a per-wave rank: read
// from the first lane, the compiler knows it). The array is written by the host before the launch
// and never during one, so it is read through the CONSTANT address space: a scalar load (s_load),
// invariant, so neither a barrier's fences nor a store keeps the compiler from reusing or
// rematerializing it, as it did the bases when they were by value in the kernel arguments (a plain
// global load of it after a fence is a vector load, re-issued).
// As a U* when U is given (a region's bytes as its elements). Survives: the one way a uniform rank
// reaches a buffer's address (the scalar load).
template <typename U = void, typename T>
DINLINE std::conditional_t<std::is_void_v<U>, T, U>* ptr_array_at(T* const* x_ptr_array,
                                                                  int x_ptr_array_size, int r) {
  r = __builtin_amdgcn_readfirstlane(r);
  debug_in_bounds(r, x_ptr_array_size);
  typedef T* Entry;
  return reinterpret_cast<std::conditional_t<std::is_void_v<U>, T, U>*>(
      ((const __attribute__((address_space(4))) Entry*)x_ptr_array)[r]);
}

// ENTRY r OF A DEVICE ARRAY OF POINTERS FOR AN r THAT DIFFERS ACROSS THE WAVE (a lane's owner: the
// barrier's peer, an all-gather pack's owner, a row of threads narrower than a wave), as an
// address: every one of the WORLD entries read (scalar) and the lane's picked by a select, never
// an index by the lane (a per-lane load). Each entry passes through readfirstlane first: a select
// of two loads is otherwise folded into a load from a selected address, the per-lane load (ISA
// 2026-10-16: the barrier's peer base was a global_load_dwordx2 off the kernarg pointer).
// Returned as a GLOBAL pointer (U* when U is given), so its accesses stay global_* (from an
// integer they were flat_*). Survives: the one way a per-lane rank reaches a buffer's address.
template <int WORLD, typename U = void, typename T>
DINLINE std::conditional_t<std::is_void_v<U>, T, U>* lane_ptr_array_at(T* const* x_ptr_array,
                                                                       int x_ptr_array_size, int r) {
  static_assert(WORLD >= 1, "a world has a rank");
  debug_in_bounds(WORLD - 1, x_ptr_array_size);
  const auto scalar = [](T* p) {
    const auto u = reinterpret_cast<uint64_t>(p);
    const uint32_t lo = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(u));
    const uint32_t hi = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(u >> 32));
    return (uint64_t{hi} << 32) | lo;
  };
  uint64_t at = scalar(ptr_array_at(x_ptr_array, x_ptr_array_size, 0));
#pragma unroll
  for (int k = 1; k < WORLD; ++k) {
    const uint64_t p = scalar(ptr_array_at(x_ptr_array, x_ptr_array_size, k));
    at = r == k ? p : at;
  }
  auto* const global = reinterpret_cast<__attribute__((address_space(1))) char*>(at);
  return reinterpret_cast<std::conditional_t<std::is_void_v<U>, T, U>*>((char*)global);
}

// =================================================================================================
// BARRIER
//
// THE RANKS' SYNCHRONIZATION, over the SIGNALS buffer, [R, 2, B, R]:
// `signals[owner][parity][block] [writer]` is the flag block `block` on rank
// `writer` sets in rank `owner`'s memory, at a barrier of that parity. Every
// dimension is a stride but the outer R's: owner r's signals are
// signals_ptr_array's entry r (PEERS' `lane_ptr_array_at`).
//
// THE MEMORY ORDERING: the textbook one for cacheable memory shared across
// GPUs, nothing that depends on the signals bypassing a cache, so it is correct
// for whichever way the signals are backed (handle.cuh's kArenaMallocFlags), cached
// in the owner's L2 (MTYPE RW) and, through a peer's IPC mapping, in the peer's
// L2 (MTYPE NC), or uncached. The rules are LLVM's AMDGPU memory model (AMDGPUUsage, "Memory
// Model GFX942" and its code-sequence table; gfx950 takes the same scope bits):
// keeping L2s of different agents coherent takes `buffer_wbl2 sc0 sc1` on the
// writer and `buffer_inv sc0 sc1` on the reader, which is what a SYSTEM-scope
// release and acquire compile to. Torch's ROCm symmetric-memory barrier does
// the same (system-scope release/acquire on the flag, then plain loads of the
// peers' data).
//   SIGNAL: the flag store is a SYSTEM-scope atomic (store sc0 sc1), release
//   when ordered: buffer_wbl2 sc0 sc1 + s_waitcnt first, which writes back this
//   L2's dirty lines -- this rank's own writes the peers will read (its inputs'
//   copy-in, its partials) and its pushes into peers' memory held as NC lines --
//   and completes every store this wave issued, before the flag can be seen.
//   An ordered barrier first waits for every wave's stores (s_waitcnt vmcnt(0)) and
//   meets the block, so the one releasing lane covers the block's stores.
//   WAIT: relaxed SYSTEM-scope loads of this rank's own flag (load sc0 sc1),
//   then one SYSTEM-scope acquire once the flag is seen: s_waitcnt +
//   buffer_inv sc0 sc1, which invalidates this CU's L1 and this L2's non-local
//   (peer) lines. The peers' data read after it are plain loads: a peer's line
//   is fetched again from the peer, which wrote it back before its flag; a
//   local line a peer stored into was invalidated by that store's probe (the
//   PTE C-bit, same section).
//   CLEAR: a relaxed system-scope store into this rank's own flag, ordered
//   before this lane's next flag store by that barrier's own release or
//   wait for its stores (below).
// The caller's input is never read by a peer: its owner copies it into its
// inputs (ours), or pushes it into the peers' inputs (the plain all-reduce's
// inboxes: remote stores, which the release's wait and write-back complete),
// before the launch's first barrier, which is ordered (release before the
// flag, acquire after it), so the reads of the inputs after it see them.
// The tests that would catch a stale line: every op's output against
// torch.distributed's across back-to-back calls on the same buffers, and the
// signals' return to clear after every launch. Each flag is touched only
// atomically (the `__scoped_atomic` builtins), all of it in the one function `barrier`. A rank's
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

// =================================================================================
// HOW ONE GPU SYNCHRONIZES WITH ANOTHER (listed in the INTERFACE): the
// barriers, written against the signals buffer; a rank's data buffers are
// ordered only by `barrier`.
// =================================================================================

// WHAT HOLDS ONCE A BARRIER IS PASSED: every peer has launched and what it
// wrote before (its staged or pushed input) is visible to this side (a launch's first
// barrier, ordered; it differs from `visible` only in naming its fault
// `start`); what this side wrote before is
// visible to the other side after; every peer is done reading this rank (so its
// buffers may be reused). Every barrier pairs this block with the same block on
// every rank.
enum class Until { launched, visible, read };

// THE BARRIER, one for every case, and the whole protocol in one place:
// barrier<Until::read, 1, WORLD>(the signals buffer, the status buffer, rank, timeout_ticks),
// each buffer its pointer, strides and sizes, as the kernel took them. Nothing outlives the
// call. The PARITY is the kernel's: its k-th barrier, in source order, takes k % 2, so a launch
// starts on parity 0 and, its barriers even in number, ends on 1 (dev/lint_kernels.sh checks
// both). WORLD is the kernel's, passed on. A read after a barrier may see only what the SAME
// BLOCK on the peer wrote before it, so both sides must index the same data by the same block.
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
// lose it: an ordered barrier's wait for every wave's stores covers it; an
// unordered one waits for the lane's own stores just before it signals, after
// the whole phase between, so the clear has long landed.
template <Until UNTIL, int PARITY, int WORLD>
DINLINE void barrier(uint32_t* const* signals_ptr_array, int signals_ptr_array_size,
                     int64_t signals_stride_parity,
                     int64_t signals_stride_block,
                     int64_t signals_stride_writer, int signals_size_parity,
                     int signals_size_block, int signals_size_writer,
                     uint32_t* status_ptr, int64_t status_stride_field,
                     int status_size_field, int rank, uint64_t timeout_ticks) {
  static_assert(PARITY == 0 || PARITY == 1, "a barrier's parity is 0 or 1");
  constexpr bool kOrdered = UNTIL != Until::read;
  constexpr bool kFirst = UNTIL == Until::launched;
  constexpr uint32_t parity = PARITY;
  // 1. SKEW, debug builds only: up to ~32 x 8K cycles, different per rank, block and parity.
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG) {
    uint32_t h = static_cast<uint32_t>(rank) * 73856093u ^
                 blockIdx.x * 19349663u ^ parity * 83492791u;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    for (uint32_t n = (h ^ (h >> 15)) % 32; n > 0; --n)
      __builtin_amdgcn_s_sleep(127);
  }
  // 2. EVERY WAVE'S STORES DONE before a block barrier that one wave then releases:
  // `__syncthreads` waits for none (gfx9 emits no vmcnt wait before s_barrier; seen in the
  // ISA), so the releasing wave's fence would cover only its own stores, and another wave's
  // could still be in flight when the flag lands -- a remote store above all.
  if constexpr (kOrdered) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  if (threadIdx.x < WORLD) {
    if constexpr (!kOrdered) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    // 3. THE TWO FLAGS, from the signals buffer [R, 2, B, R]: rank `owner`'s for block `block`
    // on rank `writer`, at `parity` (every rank's address, its strides and its sizes; debug
    // builds check each index against its size). AN INDEX KEEPS ITS CALLER'S TYPE (a block
    // unsigned, as blockIdx.x; a rank as passed): an int where the caller had an unsigned costs
    // a sign extension in every kernel. Theirs: this lane's peer's, written by this rank; mine:
    // this rank's, written by this lane's peer.
    const auto flag = [&](auto owner, unsigned block, auto writer) {
      debug_in_bounds(static_cast<int64_t>(owner), WORLD);
      debug_in_bounds(parity, signals_size_parity);
      debug_in_bounds(block, signals_size_block);
      debug_in_bounds(static_cast<int64_t>(writer), signals_size_writer);
      return lane_ptr_array_at<WORLD>(signals_ptr_array, signals_ptr_array_size,
                                            static_cast<int>(owner)) +
             parity * signals_stride_parity + block * signals_stride_block +
             writer * signals_stride_writer;
    };
    uint32_t* const theirs_ptr = flag(threadIdx.x, blockIdx.x, rank);
    uint32_t* const mine_ptr = flag(rank, blockIdx.x, threadIdx.x);
    // 4. SIGNAL: a SYSTEM-scope atomic store, release when ordered. A flag is touched only by
    // the __scoped_atomic builtins, each with its order and scope spelled at the call.
    __scoped_atomic_store_n(theirs_ptr, static_cast<uint32_t>(FlagState::arrived),
                            kOrdered ? __ATOMIC_RELEASE : __ATOMIC_RELAXED,
                            __MEMORY_SCOPE_SYSTEM);
    // 5. WAIT, AT SYSTEM SCOPE, for the flag a peer's system-scope release stored. Spins
    // relaxed and acquires once, after: an acquire per poll would invalidate the caches on
    // every iteration of every spinning block. A sleep a poll (64 cycles) keeps a spinning
    // block from hammering the link. EVERY 256 POLLS, the slow path: the clock, and the
    // status's `abort` (host-mapped, so only here, never in the fast path). Past the timeout
    // the wait records the fault; timed out or aborted, the WAVE ENDS (s_endpgm): an ended wave
    // no longer counts at the block's s_barrier, so the block's other waves run on to their own
    // next wait, see `abort`, and end too. The kernel's output is then garbage and the
    // communicator broken (the host's watchdog).
    const uint32_t want = static_cast<uint32_t>(FlagState::arrived);
    // One word of this rank's status, by its field, at the buffer's stride.
    const auto word = [&](StatusField f) {
      debug_in_bounds(static_cast<int64_t>(f), status_size_field);
      return status_ptr + static_cast<int64_t>(f) * status_stride_field;
    };
    const uint64_t t0 = wall_clock64();
    for (uint32_t n = 1;
         __scoped_atomic_load_n(mine_ptr, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM) < want;
         ++n) {
      __builtin_amdgcn_s_sleep(1);
      if ((n & 255u) != 0) continue;
      const uint64_t waited = wall_clock64() - t0;
      const bool aborted = __scoped_atomic_load_n(word(StatusField::abort), __ATOMIC_RELAXED,
                                                  __MEMORY_SCOPE_SYSTEM) != 0;
      if (!aborted && waited <= timeout_ticks) continue;
      if (!aborted) {
        // THE FAULT, written once by the first wait on this rank to give up: `state` claimed
        // 0 -> 1, the fields, then 2 released, so the host reads them only whole. Then
        // `abort`, so every other wait on this rank leaves at its next check rather than
        // spinning out its own timeout.
        const uint32_t count =
            __scoped_atomic_load_n(mine_ptr, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
        uint32_t none = static_cast<uint32_t>(FaultState::none);
        if (__scoped_atomic_compare_exchange_n(
                word(StatusField::state), &none,
                static_cast<uint32_t>(FaultState::writing), false, __ATOMIC_RELAXED,
                __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM)) {
          *word(StatusField::where) =
              static_cast<uint32_t>(kFirst ? Where::start : Where::peer_barrier);
          *word(StatusField::rank) = static_cast<uint32_t>(rank);
          *word(StatusField::block) = blockIdx.x;
          *word(StatusField::peer) = threadIdx.x;
          *word(StatusField::count) = count;
          *word(StatusField::want) = want;
          *word(StatusField::elapsed_low) = static_cast<uint32_t>(waited);
          *word(StatusField::elapsed_high) = static_cast<uint32_t>(waited >> 32);
          __scoped_atomic_store_n(word(StatusField::state),
                                  static_cast<uint32_t>(FaultState::written),
                                  __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
        }
        __scoped_atomic_store_n(word(StatusField::abort), 1u, __ATOMIC_RELAXED,
                                __MEMORY_SCOPE_SYSTEM);
      }
      __builtin_amdgcn_endpgm();
    }
    // 6. ACQUIRE, ordered only: one SYSTEM-scope acquire once the flag is seen.
    if constexpr (kOrdered) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    // 7. CLEAR: a relaxed system-scope store into this rank's own flag.
    __scoped_atomic_store_n(mine_ptr, static_cast<uint32_t>(FlagState::clear),
                            __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
  __syncthreads();
}

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
// of each buffer it moves (the COLLECTIVES, all_gather); load and store
// take QUANTIZATION and work out the INT4 payload's layout from the region's own sizes. A LOAD DEQUANTIZES INTO
// ITS REGISTERS' DTYPE, each element rounded once, as quick reduce lands its values in its dtype; a
// sum is fp32 and quantized from fp32.
// A THREAD'S PACK IS ONE WORD OF ITS GROUP (8 elements of a 16-bit dtype), so a group is the packs
// of 4 consecutive lanes: a layout's THREADS_N and offs_n are whole groups, and its scale is the
// lanes' largest magnitude, two DPP steps with every lane of the quad active. One thread holding a
// whole group of every peer was 46-62 registers and spilled (the int4-wip prototype); the flat
// walk's (COLLECTIVES) holds one, a group a thread: the plain kernel 135 VGPRs, none spilled (ISA
// 2026-10-16T03-10-32Z).
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

// INT4 IS AN ENCODING OF A PAYLOAD, NOT A REGION (handle.cuh's region layout): an [m, n] payload
// published INT4 lies inside the region that holds it exact (inputs, partials), its values from
// the region's start ([m, n / 2] bytes, rows back to back), its scales at kInt4ScalesAlign past
// them ([m, n / 32] fp16). The offset is the call's m and n alone, so every rank computes the
// same one; `int4_payload_bytes` is what the region must hold (select.cuh sizes it).
constexpr int64_t kInt4ScalesAlign = 256;
__host__ __device__ constexpr int64_t int4_values_bytes(int64_t m, int64_t n) {
  return m * n / 2;
}
__host__ __device__ constexpr int64_t int4_scales_bytes(int64_t m, int64_t n) {
  return m * (n / kInt4Group) * kInt4ScaleBytes;
}
__host__ __device__ constexpr int64_t int4_scales_offset(int64_t m, int64_t n) {
  return (int4_values_bytes(m, n) + kInt4ScalesAlign - 1) / kInt4ScalesAlign * kInt4ScalesAlign;
}
__host__ __device__ constexpr int64_t int4_payload_bytes(int64_t m, int64_t n) {
  return int4_scales_offset(m, n) + int4_scales_bytes(m, n);
}

// AN [m, n] INT4 PAYLOAD IN A REGION: its values and its scales, each a buffer of its own (inside
// the region, from its own address, PEERS' `ptr_array_at`, its strides and its sizes, in its own elements:
// bytes, fp16), both dense rows. From the region's byte offset and the call's [m, n]: the values at
// the region's offset, [m, int4_values_size_n(n)]; the scales at the region's offset plus
// int4_scales_offset(m, n), [m, int4_scales_size_n(n)]; each row's stride its size_n, the column's
// 1. A kernel computes them where it starts.
__host__ __device__ constexpr int int4_values_size_n(int n) { return n / 2; }
__host__ __device__ constexpr int int4_scales_size_n(int n) { return n / kInt4Group; }

typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
// A 16-BIT DTYPE WITH BF16'S 8 MANTISSA DIGITS (c10::BFloat16's numeric_limits).
template <typename U>
constexpr bool kBf16 = sizeof(U) == 2 && std::numeric_limits<U>::digits == 8;

// DECODE: a word's 8 values at their group's scale. Its own function because load lands them in
// U and the INT4 sum adds them in fp32 from the words (int4_peers_sum, never a decoded tile).
// U float: the exact terms, (q - 8) * scale. U a 16-bit dtype: each rounded once to U, made as a
// loaded pack is, four 32-bit words copied in whole: element by element, each 16-bit value held a
// register of its own and the pull norms' two rows of every peer's packs spilled 188 VGPRs (256
// VGPRs, 592 B of partials; 2026-10-15T22-21-26Z).
// THE TERMS, EXACT in fp32, by quick reduce's 0x6400 magic: a mask puts elements 2i and 2i + 1 in
// an fp16 pair, each 1024 + q (i = 0, 2) or 1024 + 16q (i = 1, 3: their nibbles masked in place,
// no shift), OR'd with 0x6400; then one v_fma_mix_f32 a value takes a half at scale (or scale /
// 16) plus bias (-1032 * scale, or -72 * scale): a 22-bit product, then an exact difference. 13
// instructions a word, in one asm block: left to the compiler, each half was converted and fed to
// packed fmas (1.5 a value, a gfx950 probe compile, 2026-10-16). VOLATILE, so the blocks keep their
// order (int4_peers_sum pins what consumes them between).
// LANDED IN U two values a 32-bit word: with bf16 one v_cvt_pk_bf16_f32, where each element's own
// conversion was one with a zero half, then a shift and an or (12 instructions a pack, not 4; ISA
// 2026-10-16T02-01-12Z).
template <typename U, int P>
DINLINE void decode(uint32_t word, float scale, U (&output)[P]) {
  static_assert(P == 8, "a word is 8 values");
  static_assert(std::is_same_v<U, float> || sizeof(U) == 2, "fp32 terms or a 16-bit dtype");
  const float scale16 = scale * 0.0625f;
  const float bias = scale * -1032.0f;
  const float bias16 = scale * -72.0f;
  float term[P];
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
  if constexpr (std::is_same_v<U, float>) {
#pragma unroll
    for (int j = 0; j < P; ++j) output[j] = term[j];
  } else {
    u32x4 packed;
#pragma unroll
    for (int w = 0; w < P / 2; ++w) {
      const float a = term[2 * w], b = term[2 * w + 1];
      if constexpr (kBf16<U>)
        packed[w] = __builtin_bit_cast(uint32_t, __builtin_convertvector(f32x2{a, b}, bf16x2));
      else
        packed[w] = static_cast<uint32_t>(__builtin_bit_cast(uint16_t, static_cast<U>(a))) |
                    static_cast<uint32_t>(__builtin_bit_cast(uint16_t, static_cast<U>(b))) << 16;
    }
    __builtin_memcpy(output, &packed, sizeof(packed));
  }
}

// ENCODE: a group's packs to their words and the group's scale. Its own function because store
// (a pack a thread, THREAD_PACKS 1) and the INT4 flat walk (a group a thread, THREAD_PACKS
// kInt4GroupPacks) share it; the thread map is a parameter, not implied by the encoding (R7).
// THE GROUP'S LARGEST MAGNITUDE: a pack a thread, over its 4 lanes (quad_perm [1,0,3,2], then
// [2,3,0,1], as unsigned bits: a float max canonicalized each step); a group a thread, the
// thread's own. THE SCALE from it: / 7, to fp16 (past its largest, clamped to it). THE INVERSE the
// values are rounded at, CORRECTLY ROUNDED as the format's 1 / scale (v_rcp_f32's ulp flipped 0.6%
// of a two-shot's elements a step against it: torch emulation of the 2048 x 7168 test,
// 2026-10-16); a group of zeros: 0, so all 8s at scale 0.
// A PACK'S WORD: each value rounded to the nearest of -8..7 times the scale stored, BY THE MAGIC
// NUMBER: x * inv, rounded to fp32 as the format's product is, plus 1.5 * 2^23 + 8 rounds it to an
// integer (the sum's ulp is 1) and leaves q + 8 in the low 4 bits, the rest of its low byte 0;
// clamped by med3 (two elements a packed mul and add). NOT AN FMA: the exact product rounded once
// flipped 0.8% of the elements (the emulation's double rounding, 2026-10-16). Then packed by
// bytes: element 2i + 2's value shifted 4 and OR'd onto 2i's makes a byte of the word, and two
// v_perm_b32 gather the even elements' bytes and the odd ones' (7 instructions a word, where a
// shift and an OR an element was 16).
template <int THREAD_PACKS, typename T, int P>
DINLINE void encode(const T (&input)[THREAD_PACKS][P], uint32_t (&words)[THREAD_PACKS],
                    float& scale) {
  static_assert(P * 4 == 32, "a pack is one word of 8 values");
  static_assert(THREAD_PACKS == 1 || THREAD_PACKS == kInt4GroupPacks,
                "a pack a thread (its quad's group) or a group a thread");
  float x[THREAD_PACKS][P];
  float amax = 0.0f;
#pragma unroll
  for (int k = 0; k < THREAD_PACKS; ++k)
#pragma unroll
    for (int j = 0; j < P; ++j) {
      x[k][j] = static_cast<float>(input[k][j]);
      amax = fmaxf(amax, fabsf(x[k][j]));
    }
  if constexpr (THREAD_PACKS == 1) {
    uint32_t bits = __builtin_bit_cast(uint32_t, amax);
    bits = max(bits, static_cast<uint32_t>(__builtin_amdgcn_mov_dpp(
                         static_cast<int>(bits), 0xb1, 0xf, 0xf, false)));
    bits = max(bits, static_cast<uint32_t>(__builtin_amdgcn_mov_dpp(
                         static_cast<int>(bits), 0x4e, 0xf, 0xf, false)));
    amax = __builtin_bit_cast(float, bits);
  }
  scale = static_cast<float>(static_cast<__fp16>(fminf(amax * (1.0f / 7.0f), 65504.0f)));
  const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
  // A PACK'S WORD from its 8 values at the inverse.
  const auto word = [](const float (&v)[P], float inverse) {
    constexpr float kLowest = 12582912.0f;  // 1.5 * 2^23, q + 8 = 0
    uint32_t q[P];
#pragma unroll
    for (int p = 0; p < P / 2; ++p) {
#pragma clang fp contract(off)
      const f32x2 y = f32x2{v[2 * p], v[2 * p + 1]} * f32x2{inverse, inverse} +
                      f32x2{kLowest + 8.0f, kLowest + 8.0f};
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
  };
#pragma unroll
  for (int k = 0; k < THREAD_PACKS; ++k) words[k] = word(x[k], inv);
}
typedef __attribute__((address_space(1))) uint32_t global_u32;
typedef __attribute__((address_space(1))) __fp16 global_f16;

// `offset` bytes past `base`, the offset opaque so the address is built at its access (at_col's).
template <typename E>
DINLINE E* at_offset(E* base, uint32_t offset) {
  using Byte = std::conditional_t<std::is_const_v<E>, const char, char>;
  asm volatile("" : "+v"(offset));
  return reinterpret_cast<E*>(reinterpret_cast<Byte*>(base) + offset);
}


template <typename U, int ROWS, int K, int P>
DINLINE void int4_decode(const uint32_t (&words)[ROWS][K], const float (&scales)[ROWS][K],
                         U (&output)[ROWS][K][P]) {
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int k = 0; k < K; ++k) decode(words[i][k], scales[i][k], output[i][k]);
}


// Every peer's words and scales (kept with int4_peers_sum: the push kernels' int4 sum from the
// words, where a decoded tile of every peer held 172 VGPRs; inlined in reduce_scatter it was 2% more
// instructions, ISA I3), the position once a pack in a vector register (load's every rank),
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
  [[maybe_unused]] const bf16x2 ones = __builtin_bit_cast(bf16x2, 0x3f803f80u);
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int k = 0; k < K; ++k) {
      float acc[P];
#pragma unroll
      for (int j = 0; j < P; ++j) acc[j] = 0.0f;
      int r = 0;
      // v_dot2_f32_bf16 IS GFX950'S (dot12-insts): gfx942 sums every rank by the loop
      // below. The image builds both targets, the dev loop one, so this broke only the
      // image (2026-10-16, e2e 04-18-33Z's test step).
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx950__)
      if constexpr (kBf16<T>) {
#pragma unroll
        for (; r + 1 < WORLD; r += 2) {
          float a[P], b[P];
          decode(words[r][i][k], group_scales[r][i][k], a);
          decode(words[r + 1][i][k], group_scales[r + 1][i][k], b);
#pragma unroll
          for (int j = 0; j < P; ++j)
            acc[j] = __builtin_amdgcn_fdot2_f32_bf16(
                __builtin_convertvector(f32x2{a[j], b[j]}, bf16x2), ones, acc[j], false);
          // THE SUMS DONE BEFORE THE NEXT DECODE'S VOLATILE ASM: the scheduler, free to use 256
          // VGPRs at 512 threads, decoded every peer's words of a group ahead of the first sum
          // and spilled (the plain kernel 256 VGPRs, 264 B of partials: ISA 2026-10-16T02-22-15Z).
#pragma unroll
          for (int j = 0; j < P; ++j) asm volatile("" ::"v"(acc[j]));
        }
      }
#endif
#pragma unroll
      for (; r < WORLD; ++r) {
        float a[P];
        decode(words[r][i][k], group_scales[r][i][k], a);
#pragma unroll
        for (int j = 0; j < P; ++j) acc[j] += static_cast<float>(static_cast<T>(a[j]));
#pragma unroll
        for (int j = 0; j < P; ++j) asm volatile("" ::"v"(acc[j]));  // as above
      }
#pragma unroll
      for (int j = 0; j < P; ++j) output[i][k][j] = acc[j];
    }
}

// =================================================================================================
// LOAD AND STORE: A THREAD'S REGISTERS FROM AND TO A BUFFER, the one load and the one store
//
// A THREAD'S REGISTERS `U v[ROWS][K][P]` (P a pack, 16 bytes of the kernel's dtype), the place as
// ints: thread t holds rows offs_m + (t / THREADS_N + i * THREADS_M) * row_step and packs
// offs_n / P + t % THREADS_N + k * THREADS_N; stride_n must be 1. One round trip; a row past
// clip_m reads the last row below it, a store skips it.
// THE BUFFER (`input` / `output`), ALWAYS ONE FORM: its pointer array (`x_ptr_array`,
// `x_ptr_array_size` entries: rank r's copy at entry r; a buffer only this rank has is a one-entry
// array, `&x_ptr, 1`, entry 0), the entry `x_rank` and the slot `x_slot`, then its strides and its
// sizes over its three dimensions: slot (`x_stride_slot`, `x_size_slot`: a pushed collective's
// inbox is [writer, m, n], one slot a writer; any other buffer has one slot), m and n. Its sizes
// are what it holds (its INT4 payload's layout comes from them).
// THE CLIP (clip_m, clip_n) is where this call stops, named apart from the buffer's sizes: a buffer
// holding part of a tensor at the tensor's places (the partials, a band of inputs) clips and masks
// at the tensor's (or the band's) sizes, not its own.
// THE VARIATIONS, template parameters:
//   RANKS, which copies: own, entry x_rank's slot x_slot (registers U[ROWS][K][P]); every, every
//     rank's entry at slot x_slot (registers U[WORLD][ROWS][K][P], rank order, all in flight
//     together; a store's same registers to every rank's copy, issued together, posted: the next
//     ordered barrier's release completes and publishes them, BARRIER); every_slot, every slot of
//     entry x_rank (a load's registers U[WORLD][ROWS][K][P], writer order: the inbox's WORLD
//     writers); lane, entry x_rank where x_rank differs across the wave (each lane's rank, the
//     select over the WORLD entries, lane_ptr_array_at).
//   QUANTIZATION: none, the buffer holds the registers' dtype; int4, it holds their INT4 payload
//     (QUANTIZATION), its values from the slot's start and its scales at the offset its own sizes
//     give (int4_scales_offset), a load dequantizing into its registers' dtype, a store quantizing
//     its registers once whichever ranks it stores them in.
// ENTRY r, A SCALAR LOAD OF THE ARRAY (ptr_array_at): the host writes a buffer's array before the
// launch. A ONE-ENTRY ARRAY (its size the literal 1) may be a pointer's own address, `&x_ptr`, so
// it is read as memory, which folds to the pointer (the constant address space's load of a
// pointer's stack copy is not one).
// =================================================================================================

enum class Ranks { own, every, every_slot, lane };

// A LOAD NEVER BRANCHES: a thread outside a narrow layout loads its clamped, in-bounds place and
// its values go unused. Under a thread compare the load was an exec branch, so every array's
// old registers stayed live across AttnRes's loops (ISA 2026-10-04T16-39-15Z).
// EVERY RANK'S: a pack position at a time, every rank's for it, so the address is computed once a
// position (the other way round cost 16 scalar instructions at two packs: ISA
// 2026-10-01T00-31-14Z). Nothing waits until the registers are used, so loads issued here run
// under other work.
template <int THREADS_M, int THREADS_N, Ranks RANKS = Ranks::own,
          int QUANTIZATION = Quantization::none, int WORLD = 1, typename PTR, typename OUTPUT>
DINLINE void load(PTR const* input_ptr_array, int input_ptr_array_size, int input_rank,
                  int input_slot, int64_t input_stride_slot, int64_t input_stride_m,
                  int64_t input_stride_n, int input_size_slot, int input_size_m, int input_size_n,
                  int clip_m, int clip_n, OUTPUT& output, int offs_m, int offs_n,
                  int row_step = 1) {
  constexpr bool kEvery = RANKS == Ranks::every || RANKS == Ranks::every_slot;
  constexpr int COPIES = kEvery ? std::extent_v<OUTPUT, 0> : 1;
  constexpr int ROWS = std::extent_v<OUTPUT, kEvery ? 1 : 0>;
  constexpr int K = std::extent_v<OUTPUT, kEvery ? 2 : 1>;
  constexpr int P = std::extent_v<OUTPUT, kEvery ? 3 : 2>;
  using U = std::remove_all_extents_t<OUTPUT>;
  using V = vec<U, P>;
  // Copy c of the buffer (entry and slot), as its elements (none) or its bytes (int4).
  using E = std::conditional_t<QUANTIZATION == Quantization::none, const U, const uint8_t>;
  const auto base = [&](int c) -> E* {
    const int entry = RANKS == Ranks::every ? c : input_rank;
    const int slot = RANKS == Ranks::every_slot ? c : input_slot;
    debug_in_bounds(slot, input_size_slot);
    // The entries as plain pointers (a kernel's own pointer is __restrict__).
    const auto* const entries = (decltype(+input_ptr_array[0]) const*)input_ptr_array;
    E* at;
    if constexpr (RANKS == Ranks::lane)
      at = lane_ptr_array_at<WORLD, E>(entries, input_ptr_array_size, entry);
    else if (__builtin_constant_p(input_ptr_array_size) && input_ptr_array_size == 1)
      at = reinterpret_cast<E*>(entries[0]);
    else
      at = ptr_array_at<E>(entries, input_ptr_array_size, entry);
    return at + slot * input_stride_slot;
  };
  const int tid = static_cast<int>(threadIdx.x);
  // Thread-row i's row (one row of threads: the block's, so its address stays scalar), a row past
  // clip_m reading the last below it; pack k's column, past clip_n the last below it.
  const auto row_at = [&](int i) {
    return offs_m + (THREADS_M == 1 ? i : tid / THREADS_N + i * THREADS_M) * row_step;
  };
  const auto row_of = [&](int i) { return row_at(i) < clip_m ? row_at(i) : clip_m - 1; };
  // A LOAD SKIPS ROW i only at several rows a thread or rows of threads, whose rows past clip_m
  // would re-read the last over the links. One row's is real wherever a kernel makes one, and a
  // branch around its loads cost registers (126 -> 178 VGPRs, spilling at 16384: 23-46-07Z).
  const auto skips = [&](int i) { return !(ROWS == 1 && THREADS_M == 1) && row_at(i) >= clip_m; };
  const auto pack_of = [&](int k) {
    const int n = offs_n / P + tid % THREADS_N + k * THREADS_N;
    return n < clip_n / P ? n : clip_n / P - 1;
  };
  if constexpr (QUANTIZATION == Quantization::none && !kEvery) {
    const V* at = reinterpret_cast<const V*>(base(0));
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
      if (skips(i)) continue;
      const V* row = at + int64_t{row_of(i)} * (input_stride_m / P);
#pragma unroll
      for (int k = 0; k < K; ++k) {
        // A ROW'S PACK k AS A UNIFORM BASE PLUS AN UNSIGNED 32-BIT BYTE OFFSET: the form
        // global_load takes with the base in scalar registers (saddr), so every tensor sharing the
        // columns shares one offset register. A signed column added to the pointer was a 64-bit
        // vector address a tensor a pack, 28 VGPRs held across AttnRes's loops (liveness,
        // 2026-10-04T15-28-51Z). THE ADDRESS IS BUILT AT ITS LOAD: the offset made opaque here, so
        // the 64-bit sum is not hoisted and held a tensor a pack (AttnRes 90 -> 70 VGPRs).
        uint32_t offset = static_cast<uint32_t>(pack_of(k)) * uint32_t{sizeof(V)};
        asm volatile("" : "+v"(offset));
        const V* p = reinterpret_cast<const V*>(reinterpret_cast<const char*>(row) + offset);
        V v;
        if constexpr (sizeof(V) == 16) {
          const u32x4 raw = *(const global_u32x4*)(p);
          __builtin_memcpy(&v, &raw, 16);
        } else {
          v = *p;
        }
        __builtin_memcpy(output[i][k], &v, sizeof(v));
      }
    }
    // ISSUED HERE, NOT WHERE THE COMPILER LIKES: no memory operation moves across this point (the
    // empty asm's memory clobber) and no instruction is scheduled across it (sched_barrier), so
    // every load above it is in flight before anything below it runs (ISA 2026-10-01T00-39-51Z,
    // 2026-10-03T21-00-27Z).
    asm volatile("" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
  } else if constexpr (QUANTIZATION == Quantization::none) {
    const U* at_copy[COPIES];
#pragma unroll
    for (int c = 0; c < COPIES; ++c) at_copy[c] = base(c);
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
      if (skips(i)) continue;
#pragma unroll
      for (int k = 0; k < K; ++k) {
        // THE POSITION ONCE, IN A VECTOR REGISTER: left to the compiler, the row's uniform part
        // was added to every rank's base in scalar registers, an add and a carry a rank (16
        // instructions a tile at eight ranks, the plain all-reduce against the flat walk: ISA
        // 2026-10-11); held in a VGPR, each rank's address is one shift-and-add of it.
        int64_t at = int64_t{row_of(i)} * (input_stride_m / P) + pack_of(k);
        asm volatile("" : "+v"(at));
#pragma unroll
        for (int c = 0; c < COPIES; ++c) {
          const V* p = reinterpret_cast<const V*>(at_copy[c]) + at;
          V v;
          if constexpr (sizeof(V) == 16) {
            const u32x4 raw = *(const global_u32x4*)(p);
            __builtin_memcpy(&v, &raw, 16);
          } else {
            v = *p;
          }
          __builtin_memcpy(output[c][i][k], &v, sizeof(v));
        }
      }
    }
    asm volatile("" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
  } else {
    static_assert(THREADS_N % kInt4GroupPacks == 0, "a group is 4 lanes' packs");
    // The buffer's INT4 payload, from its own sizes (the INT4 payload layout).
    const int64_t values_stride_m = int4_values_size_n(input_size_n);
    const int64_t scales_stride_m = int4_scales_size_n(input_size_n);
    const int64_t scales_at = int4_scales_offset(input_size_m, input_size_n);
    // Every copy's words and scales, the position once a pack in a vector register (32-BIT BYTE
    // OFFSETS: a quantized buffer is ours, at most the box's, select's exchange_sizes), each
    // copy's word and scale an add of it; all in flight before the first is decoded.
    const uint8_t* values[COPIES];
    const char* scales[COPIES];
#pragma unroll
    for (int c = 0; c < COPIES; ++c) values[c] = base(c);
#pragma unroll
    for (int c = 0; c < COPIES; ++c) scales[c] = reinterpret_cast<const char*>(values[c] + scales_at);
    uint32_t words[COPIES][ROWS][K];
    float group_scales[COPIES][ROWS][K];
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
      if (skips(i)) continue;
#pragma unroll
      for (int k = 0; k < K; ++k) {
        const int64_t row = row_of(i);
        const int pack = pack_of(k);
        uint32_t value_at = static_cast<uint32_t>(row * values_stride_m) +
                            static_cast<uint32_t>(pack) * kInt4WordBytes;
        uint32_t scale_at = static_cast<uint32_t>(
            (row * scales_stride_m + pack / kInt4GroupPacks) * kInt4ScaleBytes);
        asm volatile("" : "+v"(value_at));
        asm volatile("" : "+v"(scale_at));
#pragma unroll
        for (int c = 0; c < COPIES; ++c) {
          words[c][i][k] = *(const global_u32*)(values[c] + value_at);
          group_scales[c][i][k] = *(const global_f16*)(scales[c] + scale_at);
        }
      }
    }
    asm volatile("" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (kEvery) {
#pragma unroll
      for (int c = 0; c < COPIES; ++c) int4_decode(words[c], group_scales[c], output[c]);
    } else {
      int4_decode(words[0], group_scales[0], output);
    }
  }
}

// Stores only the rows below clip_m and the packs below clip_n, from the threads of the layout
// (THREADS the block's, so a layout of all of it compares nothing: threadIdx.x < THREADS is not
// folded from the launch bounds, a compare and a branch around every store, ISA 2026-10-15).
// EVERY RANK'S (WORLD of them): a pack position at a time, every rank's store for it issued together
// (the position once, as load's); int4 quantizes once.
template <int THREADS_M, int THREADS_N, int THREADS, Ranks RANKS = Ranks::own,
          int QUANTIZATION = Quantization::none, int WORLD = 1, typename INPUT, typename PTR>
DINLINE void store(const INPUT& input, PTR const* output_ptr_array, int output_ptr_array_size,
                   int output_rank, int output_slot, int64_t output_stride_slot,
                   int64_t output_stride_m, int64_t output_stride_n, int output_size_slot,
                   int output_size_m, int output_size_n, int clip_m, int clip_n, int offs_m,
                   int offs_n, int row_step = 1) {
  static_assert(RANKS == Ranks::own || RANKS == Ranks::every, "a store is to one copy or every rank's");
  static_assert(THREADS_M * THREADS_N <= THREADS, "a layout is the block or fewer");
  constexpr int COPIES = RANKS == Ranks::every ? WORLD : 1;
  constexpr int ROWS = std::extent_v<INPUT, 0>;
  constexpr int K = std::extent_v<INPUT, 1>;
  constexpr int P = std::extent_v<INPUT, 2>;
  using U = std::remove_all_extents_t<INPUT>;
  using V = vec<U, P>;
  using E = std::conditional_t<QUANTIZATION == Quantization::none, U, uint8_t>;
  const auto base = [&](int c) -> E* {
    const int entry = RANKS == Ranks::every ? c : output_rank;
    debug_in_bounds(output_slot, output_size_slot);
    const auto* const entries = (decltype(+output_ptr_array[0]) const*)output_ptr_array;
    E* at;
    if (__builtin_constant_p(output_ptr_array_size) && output_ptr_array_size == 1)
      at = reinterpret_cast<E*>(entries[0]);
    else
      at = ptr_array_at<E>(entries, output_ptr_array_size, entry);
    return at + output_slot * output_stride_slot;
  };
  const int tid = static_cast<int>(threadIdx.x);
  if constexpr (THREADS_M * THREADS_N < THREADS)
    if (tid >= THREADS_M * THREADS_N) return;
  const auto row_at = [&](int i) {
    return offs_m + (THREADS_M == 1 ? i : tid / THREADS_N + i * THREADS_M) * row_step;
  };
  const auto col_at = [&](int k) { return offs_n / P + tid % THREADS_N + k * THREADS_N; };
  E* at_copy[COPIES];
#pragma unroll
  for (int c = 0; c < COPIES; ++c) at_copy[c] = base(c);
  if constexpr (QUANTIZATION == Quantization::none) {
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
      if (row_at(i) >= clip_m) continue;
#pragma unroll
      for (int k = 0; k < K; ++k) {
        if (col_at(k) >= clip_n / P) continue;
        V v;
        __builtin_memcpy(&v, input[i][k], sizeof(v));
        // The position once (load's): a uniform base plus a 32-bit byte offset for one copy, a
        // vector-register pack index shared by every copy's address for every rank's.
        int64_t at = int64_t{row_at(i)} * (output_stride_m / P) + col_at(k);
        if constexpr (COPIES == 1) {
          uint32_t offset = static_cast<uint32_t>(col_at(k)) * uint32_t{sizeof(V)};
          asm volatile("" : "+v"(offset));
          V* row = reinterpret_cast<V*>(at_copy[0]) + int64_t{row_at(i)} * (output_stride_m / P);
          V* p = reinterpret_cast<V*>(reinterpret_cast<char*>(row) + offset);
          if constexpr (sizeof(V) == 16) {
            u32x4 raw;
            __builtin_memcpy(&raw, &v, 16);
            *(global_u32x4*)(p) = raw;
          } else {
            *p = v;
          }
        } else {
          asm volatile("" : "+v"(at));
#pragma unroll
          for (int c = 0; c < COPIES; ++c) {
            V* p = reinterpret_cast<V*>(at_copy[c]) + at;
            if constexpr (sizeof(V) == 16) {
              u32x4 raw;
              __builtin_memcpy(&raw, &v, 16);
              *(global_u32x4*)(p) = raw;
            } else {
              *p = v;
            }
          }
        }
      }
    }
  } else {
    // INT4 QUANTIZES ONCE, whichever ranks it is stored in.
    static_assert(THREADS_N % kInt4GroupPacks == 0, "a group is 4 lanes' packs");
    constexpr int QP = kInt4Group / kInt4GroupPacks;
    uint32_t words[ROWS][K];
    float scales[ROWS][K];
#pragma unroll
    for (int i = 0; i < ROWS; ++i)
#pragma unroll
      for (int k = 0; k < K; ++k)
        encode<1>(reinterpret_cast<const U(&)[1][P]>(input[i][k]),
                  reinterpret_cast<uint32_t(&)[1]>(words[i][k]), scales[i][k]);
    const int64_t values_stride_m = int4_values_size_n(output_size_n);
    const int64_t scales_stride_m = int4_scales_size_n(output_size_n);
    const int64_t scales_at = int4_scales_offset(output_size_m, output_size_n);
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
      if (row_at(i) >= clip_m) continue;
      const int64_t row = row_at(i);
#pragma unroll
      for (int k = 0; k < K; ++k) {
        const int pack = offs_n / QP + tid % THREADS_N + k * THREADS_N;
        if (pack >= clip_n / QP) continue;
        uint32_t value_at = static_cast<uint32_t>(row * values_stride_m) +
                            static_cast<uint32_t>(pack) * kInt4WordBytes;
        uint32_t scale_at = static_cast<uint32_t>(
            (row * scales_stride_m + pack / kInt4GroupPacks) * kInt4ScaleBytes);
        asm volatile("" : "+v"(value_at));
        asm volatile("" : "+v"(scale_at));
#pragma unroll
        for (int c = 0; c < COPIES; ++c) {
          *(global_u32*)(at_copy[c] + value_at) = words[i][k];
          if (pack % kInt4GroupPacks == 0)
            *(global_f16*)(at_copy[c] + scales_at + scale_at) = static_cast<__fp16>(scales[i][k]);
        }
      }
    }
  }
}

// =================================================================================================
// COLLECTIVES
//
// THE COLLECTIVES' STEPS, buffers to buffers: one tested and tuned way to do
// each, which the kernels call rather than write again (the fused AttnRes
// pull's own reduce-scatter took 1.6x the plain two-shot's cycles for its
// bytes, thread traces 2026-10-04T21-03-09Z and 21-26-11Z).
// =================================================================================================

// THE LOADS ARE THE COMMUNICATION: a rank's pointer points into its memory,
// opened in this process over IPC (PEERS), so loading reads it over the links; the sum
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
// reads its peers and keeps the sum in its partials; push, the sum is stored in every rank's
// (itself too), so after the sync each rank holds the whole sum locally.
enum class Direction : int { pull = 0, push = 1 };
// HOW reduce_scatter AND all_gather WALK THE DATA (reduce_scatter.cuh, all_gather.cuh): flat, the
// elements numbered row-major and a rank's share a contiguous range (above); rows, a rank's columns
// of every row. And where reduce_scatter sums a rank's copies: through LDS or in registers.
enum class Walk : int { flat = 0, rows = 1 };
enum class Accumulate : int { lds = 0, registers = 1 };
// =================================================================================================
// INTERFACE
//
// THE INTERFACE: the primitives a kernel may call that take its buffers as pointer arrays, strides
// and sizes (the overview is this file's head), passing them to their section's `_body` (a
// transitional name: I3 splits the file). load and store, the register math, the tiles, barrier,
// block_reduce and the codec are defined at file scope in their sections above.
// =================================================================================================

// ===============================================================================================
// MEMORY: REGISTERS FROM AND TO BUFFERS
// ===============================================================================================

// Every rank's scalar for `row`, from each rank's region (`input_ptr_array`, one of ours: the
// partials), the scalars `offset_bytes` into it (where the region's own layout puts them, worked
// out from its sizes by the caller).
template <int WORLD>
DINLINE void peers_load_row_scalars(char* const* input_ptr_array, int input_ptr_array_size,
                                    float (&output)[WORLD], int64_t offset_bytes, int row) {
#pragma unroll
  for (int r = 0; r < WORLD; ++r)
    output[r] = *(const __attribute__((address_space(1))) float*)(
        reinterpret_cast<const float*>(
            ptr_array_at<const char>(input_ptr_array, input_ptr_array_size, r) + offset_bytes) +
        row);
  issued();
}

// ===============================================================================================
// UTILS
// ===============================================================================================

}  // namespace hip_custom_communicator
