// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// COMMON: THE API, and only it: the six ops (load and store, this rank's own buffers; send and
// recv, the peers', the fabric; barrier; reduce) and the INT4 codec (quantize, dequantize), each
// at file scope (lint_kernels.sh checks it), with the vocabulary their signatures and bodies need,
// each item next to the function that needs it, and the memory ops' INTERNALS: the two bodies
// (read_buffer for load and recv, write_buffer for store and send) and the address functions
// ptr_array_at and lane_ptr_array_at, how the four and barrier read a buffer's device array of
// pointers (scalar for a rank uniform across the wave, a select over the WORLD entries for one per
// lane), called by nothing else (Michael, I7, option B: "if we can subsume the work in load and
// store we should; they are access"; their comments hold the ISA evidence). A kernel writes its register math and its walks inline, at its use. The machine and
// the build are the host's (../hardware.cuh); device code reads two facts of them here, kWaveSize
// and the pack's bytes.
//
// THE ARGUMENTS ARE A KERNEL'S, in its one order: the inputs, then the outputs, then the
// scalars. A buffer is its pointer array with its size (a buffer only this rank has is a
// one-entry array, `&x_ptr, 1`), its strides and its sizes, in elements; stride_n is 1 (the host
// checks it; the packed loads need it). How a buffer is allocated is the host's (STANDARDS 3.2).
// A thread's registers are a plain array the caller declares, `T v[ROWS][K][P]`, an input or an
// output like a buffer: P a pack's elements (traits<T>::N); a call names the tile's rows TILE_M
// and THREADS_PER_BLOCK, and the thread layout is derived from them (load and store, below);
// every rank's is `T v[WORLD][ROWS][K][P]`.
// The extents are deduced from the array: the array's shape is what the caller drills.
//
// CONCURRENCY IS PART OF EACH OP'S CONTRACT, so a kernel keeps it by using them:
//   1. Global memory only through the loads and stores here: a kernel never indexes a row.
//   2. A thread's registers are loaded whole (load, every rank's or own: one round trip) before
//      anything is stored; a pack loaded between stores waits a round trip, since a store may
//      alias it.
//   3. A load that does not depend on a reduction is issued before it (weights, the next row or
//      source), so its round trip runs under the reduction.
//
// THE VOCABULARY, by the op that needs it:
//   every op     DINLINE; vec<T, P> (a pack in a register); traits<T>::N (a pack's elements, P)
//   load, store, send, recv  u32x4, global_u32x4, global_u32, global_f16 (the global-instruction
//                forms); HIP_CUSTOM_COMMUNICATOR_DEBUG (the index checks)
//   reduce       ReduceAxis (AXIS); kWaveSize
//   internals    read_buffer, write_buffer (the bodies); ptr_array_at, lane_ptr_array_at (the
//                four and barrier only)
//   barrier      Until; the status buffer's StatusField, kStatusFields, Where, FlagState,
//                FaultState (its protocol, which the host reads)
//   the codec    Quantization; kInt4Group, kInt4GroupPacks, kInt4WordBytes, kInt4ScaleBytes;
//                int4_scales_offset (where an encoded buffer keeps its scales, from its own
//                sizes: the one address function); f32x2, bf16x2, kBf16

#pragma once

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "../hardware.cuh"

namespace hip_custom_communicator {

#define DINLINE __device__ __forceinline__

// A PACK IN A REGISTER, the unit load and store move and every op sums in.
template <typename DTYPE, int VEC_SIZE>
struct __align__(sizeof(DTYPE) * VEC_SIZE) vec {
  DTYPE d[VEC_SIZE];
};

// A PACK: kBuild.memory.pack_bytes (hardware.cuh's BUILD: the widest load, 8 bf16);
// traits<DTYPE>::N its elements, the P of a thread's registers.
template <typename DTYPE>
struct traits {
  static constexpr int N = kBuild.memory.pack_bytes / sizeof(DTYPE);
};

// PEER MEMORY THROUGH GLOBAL INSTRUCTIONS (load, store, and the collectives' inline walks). A
// pointer read out of an array has no address space the compiler can prove, so it emits
// `flat_load`, which checks the aperture and waits on both counters; casting to address space 1
// gives `global_load_dwordx4` / `global_store_dwordx4`.
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(1))) u32x4 global_u32x4;

// =================================================================================================
// REDUCE
//
// THE BLOCK REDUCTION, over a WAVE (shuffles), then over the BLOCK (a wave's result, then one
// LDS pass); the peers' sum (every rank's registers elementwise in fp32, in rank order) is
// written inline by each kernel that sums them. N
// values at once at every scope: each block pass is two barriers and an LDS
// round trip, so a kernel with several reductions takes them together.
// =================================================================================================

// THE AXIS a reduce reduces over (template ints, as Quantization): `all`, the whole block;
// `m`, the TILE_M rows of THREADS_PER_BLOCK / TILE_M threads, elementwise a lane.
namespace ReduceAxis {
constexpr int all = 0;
constexpr int m = 1;
}  // namespace ReduceAxis

// THE BLOCK REDUCTION, one for every case: reduce<TILE_M, THREADS_PER_BLOCK, AXIS>(input,
// output), a sum (I9: Max had no caller, so there is no operation parameter), every value of
// [K][P] summed on its own, in fp32. Input and output may be the same array (in place).
//
// AXIS all: over the whole block (THREADS_PER_BLOCK threads, TILE_M 1), every thread left
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
template <int TILE_M, int THREADS_PER_BLOCK, int AXIS, typename T, typename U, int K, int P>
DINLINE void reduce(const T (&input)[K][P], U (&output)[K][P]) {
  static_assert(AXIS == ReduceAxis::all || AXIS == ReduceAxis::m, "AXIS is all or m");
  // The block as TILE_M rows of kThreadsN threads.
  constexpr int kThreadsM = TILE_M;
  constexpr int kThreadsN = THREADS_PER_BLOCK / TILE_M;
  static_assert(kThreadsM * kThreadsN == THREADS_PER_BLOCK, "the block is TILE_M whole rows of threads");
  if constexpr (AXIS == ReduceAxis::all) {
    static_assert(TILE_M == 1, "AXIS all is one row of the block's threads");
    constexpr int kValues = K * P;
    constexpr int kWaves = (kThreadsN + kWaveSize - 1) / kWaveSize;
    __shared__ float partial[kValues][kWaves];
    const int lane = threadIdx.x % kWaveSize;
    const int wave = threadIdx.x / kWaveSize;
    float v[kValues];
#pragma unroll
    for (int n = 0; n < kValues; ++n) v[n] = static_cast<float>(input[n / P][n % P]);
    // 1. OVER THE WAVE, every lane left holding the results, by DPP, as rocPRIM's and Composable
    // Kernel's wave reductions do: each step is a VALU operand from another lane. The butterfly
    // __shfl_xor it replaced compiled to ds_bpermute, an LDS round trip a step, six of them
    // waiting on each other (ISA 2026-09-30T20-43-18Z; stamps: 0.92 us for one reduce).
    //   swap neighbours, swap pairs (quad_perm [1,0,3,2], [2,3,0,1]): each quad holds its sum
    //   mirror in 8 lanes, then in 16 (row_half_mirror, row_mirror): each row holds its sum
    //   lane 15 into rows 1 and 3, lane 31 into rows 2 and 3 (row_bcast15, row_bcast31): lane 63
    //   holds the wave's, and readlane hands it to every lane.
    // A DPP STEP, `x` from the lane the control names: `row_mask` picks which 16-lane rows take it;
    // the others read the sum's identity, 0, which leaves them as they were. A FULL ROW MASK NEEDS NO
    // `old`: left undefined (mov_dpp), the compiler fuses the move into the add that uses it
    // (v_add_f32_dpp); an explicit identity kept a v_mov_b32_dpp and an add a step (ISA
    // 2026-10-04T03-07-41Z). The moved value first: only a VOP2's src0 takes DPP, so (dpp(x), x)
    // fuses each step into one v_add_f32_dpp where (x, dpp(x)) was a v_mov_b32_dpp and an add
    // (ISA 2026-10-04T02-22-41Z).
    static_assert(kWaveSize == 64, "the DPP sequence is for 64-lane waves");
    const auto bits = [](float x) { return __builtin_bit_cast(int, x); };
    const auto real = [](int x) { return __builtin_bit_cast(float, x); };
    constexpr int kIdentity = 0;  // 0.0f's bits, the sum's identity
#pragma unroll
    for (int n = 0; n < kValues; ++n) {
      float x = v[n];
      // quad_perm [1,0,3,2], quad_perm [2,3,0,1], row_half_mirror, row_mirror: every row.
      x = real(__builtin_amdgcn_mov_dpp(bits(x), 0xb1, 0xf, 0xf, false)) + x;
      x = real(__builtin_amdgcn_mov_dpp(bits(x), 0x4e, 0xf, 0xf, false)) + x;
      x = real(__builtin_amdgcn_mov_dpp(bits(x), 0x141, 0xf, 0xf, false)) + x;
      x = real(__builtin_amdgcn_mov_dpp(bits(x), 0x140, 0xf, 0xf, false)) + x;
      // row_bcast15 into rows 1, 3; row_bcast31 into rows 2, 3.
      x = real(__builtin_amdgcn_update_dpp(kIdentity, bits(x), 0x142, 0xa, 0xf, false)) + x;
      x = real(__builtin_amdgcn_update_dpp(kIdentity, bits(x), 0x143, 0xc, 0xf, false)) + x;
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
      for (int w = 1; w < kWaves; ++w) x = x + partial[n][w];
      output[n / P][n % P] = static_cast<U>(x);
    }
    __syncthreads();
  } else {
    using V = vec<T, P>;
    __shared__ V part[kThreadsM][K][kThreadsN];
    const int mine = static_cast<int>(threadIdx.x) % kThreadsN;
#pragma unroll
    for (int k = 0; k < K; ++k)
      __builtin_memcpy(&part[static_cast<int>(threadIdx.x) / kThreadsN][k][mine], input[k], sizeof(V));
    __syncthreads();
    // THE FIRST ROW OF THREADS HOLDS THE SUM: a compare only when there are several (threadIdx.x <
    // kThreadsN is not folded from the launch bounds: a compare and a branch around every store,
    // ISA 2026-10-15).
    if (kThreadsM == 1 || static_cast<int>(threadIdx.x) < kThreadsN) {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        float x[P];
        V p = part[0][k][mine];
#pragma unroll
        for (int j = 0; j < P; ++j) x[j] = static_cast<float>(reinterpret_cast<const T*>(&p)[j]);
#pragma unroll
        for (int r = 1; r < kThreadsM; ++r) {
          p = part[r][k][mine];
#pragma unroll
          for (int j = 0; j < P; ++j)
            x[j] = x[j] + static_cast<float>(reinterpret_cast<const T*>(&p)[j]);
        }
#pragma unroll
        for (int j = 0; j < P; ++j) output[k][j] = static_cast<U>(x[j]);
      }
    }
    __syncthreads();  // before the next reduction writes `part`
  }
}

// AN INDEX INSIDE ITS BUFFER'S SIZE is checked in debug builds only (`if
// (HIP_CUSTOM_COMMUNICATOR_DEBUG && (index < 0 || index >= size)) __builtin_trap();` at each
// index, compiled out of a release build, which passes the sizes and reads none): one outside is a
// programming error, not a peer's fault, so it traps. load, store and barrier check their indices.
#ifndef HIP_CUSTOM_COMMUNICATOR_DEBUG
  #define HIP_CUSTOM_COMMUNICATOR_DEBUG 0
#endif

// ENTRY r OF A DEVICE ARRAY OF POINTERS (`x_ptr_array`, `x_ptr_array_size` entries: a region's
// addresses, a launch's inputs), `r` THE SAME ACROSS THE WAVE (a constant, or a per-wave rank: read
// from the first lane, the compiler knows it). The array is written by the host before the launch
// and never during one, so it is read through the CONSTANT address space: a scalar load (s_load),
// invariant, so neither a barrier's fences nor a store keeps the compiler from reusing or
// rematerializing it, as it did the bases when they were by value in the kernel arguments (a plain
// global load of it after a fence is a vector load, re-issued).
// As a U* when U is given (a region's bytes as its elements). Survives: the one way a uniform rank
// reaches a buffer's address (the scalar load).
// EVIDENCE (I7, 2026-10-17): inlined as `x_ptr_array[r]` in I6 (6460a42b72), 17 of 100 gfx950
// kernels regressed against 6da19121d1, exact default-path ones among them, every one with more
// vmem, fewer SGPRs and more VGPRs (the arrays vector-loaded a lane): pull one-shot add_rms_norm
// VGPR 58 -> 72, occupancy 8 -> 7, vmem 50 -> 57; pull two-shot add_rms_norm none 99 -> 140,
// occupancy 4 -> 3 (int4 112 -> 193, 4 -> 2); push two-shot rms_norm none 54 -> 70, 8 -> 7;
// AttnRes one-shot 68 -> 84, 7 -> 5 (gp report isa diff --summary). So they are API.
template <typename U = void, typename T>
DINLINE std::conditional_t<std::is_void_v<U>, T, U>* ptr_array_at(T* const* x_ptr_array,
                                                                  int x_ptr_array_size, int r) {
  r = __builtin_amdgcn_readfirstlane(r);
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG && (r < 0 || r >= x_ptr_array_size)) __builtin_trap();
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
// EVIDENCE: ptr_array_at's, above.
template <typename U, int WORLD, typename T>
DINLINE std::conditional_t<std::is_void_v<U>, T, U>* lane_ptr_array_at(T* const* x_ptr_array,
                                                                       int x_ptr_array_size, int r) {
  static_assert(WORLD >= 1, "a world has a rank");
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG && (WORLD - 1 < 0 || WORLD - 1 >= x_ptr_array_size)) __builtin_trap();
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
// THE STATUS: barrier's fault protocol, which the host reads (handle.cuh, log.cuh)
// =================================================================================================

// WHERE A WAIT WAS when it gave up, the fault record's `where`; Python's
// `Where`, number for number. 0 is no fault.
enum class Where : uint32_t {
  start = 1,     // a launch's first barrier: the peer had not launched
  peer_barrier,  // a later barrier
};

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

// =================================================================================================
// BARRIER
//
// THE RANKS' SYNCHRONIZATION, over the SIGNALS buffer, [R, 2, B, R]:
// `signals[owner][parity][block] [writer]` is the flag block `block` on rank
// `writer` sets in rank `owner`'s memory, at a barrier of that parity. Every
// dimension is a stride but the outer R's: owner r's signals are
// signals_ptr_array's entry r (lane_ptr_array_at).
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
// flag, acquire after it), so the reads of the inputs after it see them;
// the pulls' first barrier (`started`) is not: what they read was written
// before the launches, which the launch boundaries order (Until, below).
// The tests that would catch a stale line: every op's output against
// torch.distributed's across back-to-back calls on the same buffers, and the
// signals' return to clear after every launch. Each flag is touched only
// atomically (the `__scoped_atomic` builtins), all of it in the one function `barrier`. A rank's
// data buffers are the kernels' (load, store).
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
// reads the status (THE STATUS's `StatusField`). HIP_CUSTOM_COMMUNICATOR_DEBUG=1
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
// STARTED (the pulls' first barrier: nothing written in the launch before it, every read after it
// of what a peer wrote BEFORE its launch) is unordered, as aiter's `start_sync` (relaxed store,
// relaxed polls, no fence): the launch boundaries order it. The writer's kernel end releases at
// agent scope at least, `buffer_wbl2 sc1`, which LLVM's GFX942 model says is what makes local
// writes coherent for another agent's CUs; this launch's start acquires, and nothing in it read a
// peer's input before the barrier, so no stale line of one can be held. Its release and acquire
// cost 0.3-0.5 us each a call at decode (2026-10-18T08-48-34Z, 08-52-54Z against 08-44-19Z).
enum class Until { launched, visible, read, started };

// THE BARRIER, one for every case, and the whole protocol in one place:
// barrier<WORLD>(the signals buffer, the status buffer, rank, timeout_ticks, Until::read, 1),
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
template <int WORLD>
DINLINE void barrier(uint32_t* const* signals_ptr_array, int signals_ptr_array_size,
                     int64_t signals_stride_parity,
                     int64_t signals_stride_block,
                     int64_t signals_stride_writer, int signals_size_parity,
                     int signals_size_block, int signals_size_writer,
                     uint32_t* status_ptr, int64_t status_stride_field,
                     int status_size_field, int rank, uint64_t timeout_ticks, Until until,
                     uint32_t parity) {
  // UNTIL AND PARITY ARE PLAIN ARGUMENTS: barrier is force-inlined and every call passes literals,
  // so they fold as template parameters would (I8, ISA summary).
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG && parity > 1) __builtin_trap();
  const bool ordered = until == Until::launched || until == Until::visible;
  const bool first = until == Until::launched || until == Until::started;
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
  if (ordered) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  if (threadIdx.x < WORLD) {
    if (!ordered) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    // 3. THE TWO FLAGS, from the signals buffer [R, 2, B, R]: rank `owner`'s for block `block`
    // on rank `writer`, at `parity` (every rank's address, its strides and its sizes; debug
    // builds check each index against its size). AN INDEX KEEPS ITS CALLER'S TYPE (a block
    // unsigned, as blockIdx.x; a rank as passed): an int where the caller had an unsigned costs
    // a sign extension in every kernel. Theirs: this lane's peer's, written by this rank; mine:
    // this rank's, written by this lane's peer.
    const auto flag = [&](auto owner, unsigned block, auto writer) {
      if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
          (static_cast<int64_t>(owner) < 0 || static_cast<int64_t>(owner) >= WORLD ||
           static_cast<int64_t>(parity) >= signals_size_parity ||
           static_cast<int64_t>(block) >= signals_size_block ||
           static_cast<int64_t>(writer) < 0 ||
           static_cast<int64_t>(writer) >= signals_size_writer))
        __builtin_trap();
      return lane_ptr_array_at<uint32_t, WORLD>(signals_ptr_array, signals_ptr_array_size,
                                      static_cast<int>(owner)) +
             parity * signals_stride_parity + block * signals_stride_block +
             writer * signals_stride_writer;
    };
    uint32_t* const theirs_ptr = flag(threadIdx.x, blockIdx.x, rank);
    uint32_t* const mine_ptr = flag(rank, blockIdx.x, threadIdx.x);
    // 4. SIGNAL: a SYSTEM-scope atomic store, release when ordered. A flag is touched only by
    // the __scoped_atomic builtins, each with its order and scope spelled at the call.
    __scoped_atomic_store_n(theirs_ptr, static_cast<uint32_t>(FlagState::arrived),
                            ordered ? __ATOMIC_RELEASE : __ATOMIC_RELAXED,
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
      if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
          (static_cast<int64_t>(f) < 0 || static_cast<int64_t>(f) >= status_size_field))
        __builtin_trap();
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
              static_cast<uint32_t>(first ? Where::start : Where::peer_barrier);
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
    if (ordered) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
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
// of each buffer it moves (reduce_scatter, all_gather); load and store
// take QUANTIZATION and work out the INT4 payload's layout from the region's own sizes. A LOAD DEQUANTIZES INTO
// ITS REGISTERS' DTYPE, each element rounded once, as quick reduce lands its values in its dtype; a
// sum is fp32 and quantized from fp32.
// A THREAD'S PACK IS ONE WORD OF ITS GROUP (8 elements of a 16-bit dtype), so a group is the packs
// of 4 consecutive lanes: a layout's THREADS_N and offs_n are whole groups, and its scale is the
// lanes' largest magnitude, two DPP steps with every lane of the quad active. One thread holding a
// whole group of every peer was 46-62 registers and spilled (the int4-wip prototype); the flat
// walk's (reduce_scatter.cuh) holds one, a group a thread: the plain kernel 135 VGPRs, none spilled (ISA
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

// AN [m, n] BUFFER ENCODED INT4 (the INT4 payload) lies inside the buffer that holds it exact
// (staging, partials): its values from the buffer's start, [m, n / 2] bytes, rows back to back (a
// row's stride n / 2); its scales at int4_scales_offset(m, n) bytes past it, [m, n / kInt4Group]
// fp16 (a row's stride n / kInt4Group). The offset is the buffer's own m and n alone (STANDARDS
// 3.2: where an encoded buffer keeps its scales is worked out from its own sizes), so every rank
// computes the same one. What a buffer must hold for it is the host's (select.cuh's
// int4_payload_bytes).
constexpr int64_t kInt4ScalesAlign = 256;
__host__ __device__ constexpr int64_t int4_scales_offset(int64_t m, int64_t n) {
  return (m * n / 2 + kInt4ScalesAlign - 1) / kInt4ScalesAlign * kInt4ScalesAlign;
}

typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
// A 16-BIT DTYPE WITH BF16'S 8 MANTISSA DIGITS (c10::BFloat16's numeric_limits).
template <typename U>
constexpr bool kBf16 = sizeof(U) == 2 && std::numeric_limits<U>::digits == 8;

// DEQUANTIZE: a word's 8 values at their group's scale. Its own function because load lands them in
// U and the INT4 sum adds them in fp32 from the words (reduce_scatter's, never a decoded tile).
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
// order (reduce_scatter's int4 sum pins what consumes them between).
// LANDED IN U two values a 32-bit word: with bf16 one v_cvt_pk_bf16_f32, where each element's own
// conversion was one with a zero half, then a shift and an or (12 instructions a pack, not 4; ISA
// 2026-10-16T02-01-12Z).
template <typename U, int P>
DINLINE void dequantize(uint32_t word, float scale, U (&output)[P]) {
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

// DEQUANTIZE EVERY SLOT'S GROUP INTO ITS SUM: raw INT4 registers of every slot (a pushed inbox's
// writers, or every rank's: load's or recv's uint32_t form, [0] a word and [1] its group's scale),
// summed in fp32 in slot order, each element (q - 8) * scale landed in DTYPE first, as a quantized
// load lands it. The reduce-scatter's INT4 sum (reduce_scatter.cuh) and the pulled norms' row sum
// (all_reduce_add_rms_norm.cuh) are this one function. No pack of every slot decoded first:
// decoded whole, the push kernels held 172 VGPRs (102 unquantized; ISA 2026-10-15T22-47-01Z).
// WITH BF16, TWO SLOTS' TERMS A STEP: landed together (one v_cvt_pk_bf16_f32) and added by one
// v_dot2c_f32_bf16 against (1, 1), where each was a conversion, a shift and an add.
// v_dot2_f32_bf16 IS GFX950'S (dot12-insts), the one arch built (CMakeLists.txt).
template <typename DTYPE, int SLOTS, int K, int P>
DINLINE void dequantize(const uint32_t (&words)[SLOTS][1][K][2], float (&sum)[1][K][P]) {
  const auto scale_of = [&](int w) { return __builtin_bit_cast(float, words[w][0][0][1]); };
#pragma unroll
  for (int k = 0; k < K; ++k) {
#pragma unroll
    for (int j = 0; j < P; ++j) sum[0][k][j] = 0.0f;
    int w = 0;
    if constexpr (kBf16<DTYPE>) {
      const bf16x2 ones = __builtin_bit_cast(bf16x2, 0x3f803f80u);
#pragma unroll
      for (; w + 1 < SLOTS; w += 2) {
        float a[P], b[P];
        dequantize(words[w][0][k][0], scale_of(w), a);
        dequantize(words[w + 1][0][k][0], scale_of(w + 1), b);
#pragma unroll
        for (int j = 0; j < P; ++j)
          sum[0][k][j] = __builtin_amdgcn_fdot2_f32_bf16(
              __builtin_convertvector(f32x2{a[j], b[j]}, bf16x2), ones, sum[0][k][j], false);
        // THE SUMS DONE BEFORE THE NEXT DEQUANTIZE'S VOLATILE ASM: the scheduler, free to use 256
        // VGPRs at 512 threads, decoded every writer's words of a group ahead of the first sum and
        // spilled (the plain kernel 256 VGPRs, 264 B of partials: ISA 2026-10-16T02-22-15Z).
#pragma unroll
        for (int j = 0; j < P; ++j) asm volatile("" ::"v"(sum[0][k][j]));
      }
    }
#pragma unroll
    for (; w < SLOTS; ++w) {
      float a[P];
      dequantize(words[w][0][k][0], scale_of(w), a);
#pragma unroll
      for (int j = 0; j < P; ++j) sum[0][k][j] += static_cast<float>(static_cast<DTYPE>(a[j]));
#pragma unroll
      for (int j = 0; j < P; ++j) asm volatile("" ::"v"(sum[0][k][j]));  // as above
    }
  }
}

// QUANTIZE: a group's packs to their words and the group's scale. Its own function because store
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
DINLINE void quantize(const T (&input)[THREAD_PACKS][P], uint32_t (&words)[THREAD_PACKS],
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

// =================================================================================================
// LOAD, STORE, SEND AND RECV: A THREAD'S REGISTERS FROM AND TO A BUFFER. load and store touch
// this rank's own buffers only; send writes into peers' buffers and recv reads from them (SEND AND
// RECV ARE THE FABRIC, Michael, October 17: a reader sees where data crosses the links, each paired
// with a barrier). The four are one read and one write (read_buffer, write_buffer, below), so a
// peer access compiles as a local one does; only which copies differs.
//
// A THREAD'S REGISTERS `U v[kRows][kPacks][kPackElems]` (a pack, kPackElems elements: 16 bytes of
// the kernel's dtype), the layout from the tile's rows TILE_M and the block's THREADS_PER_BLOCK:
// kThreadsM = TILE_M / kRows rows of threads, kThreadsN = THREADS_PER_BLOCK / kThreadsM threads a
// row; thread t holds rows offs_m + (t / kThreadsN + i * kThreadsM) * row_step and packs
// offs_n / kPackElems + t % kThreadsN + k * kThreadsN; stride_n must be 1. One round trip; a row past
// clip_m reads the last row below it, a write skips it (so a store of TILE_M = WORLD rows of threads
// clipped at its first row is the first row's alone: what a reduce over the rows, AXIS m, leaves).
// THE BUFFER (`input` / `output`), ALWAYS ONE FORM: its pointer array (`x_ptr_array`,
// `x_ptr_array_size` entries: rank r's copy at entry r; a buffer only this rank has is a one-entry
// array, `&x_ptr, 1`, entry 0), the entry `x_rank` and the slot `x_slot`, then its strides and its
// sizes over its three dimensions: slot (`x_stride_slot`, `x_size_slot`: a pushed collective's
// inbox is [writer, m, n], one slot a writer; any other buffer has one slot), m and n. Its sizes
// are what it holds (its INT4 payload's layout comes from them).
// THE CLIP (clip_m, clip_n) is where this call stops, named apart from the buffer's sizes: a buffer
// holding part of a tensor at the tensor's places (the partials, a band of inputs) clips and masks
// at the tensor's (or the band's) sizes, not its own.
// WHICH COPIES: load and store this rank's entry; recv and send any rank's, a lane's own where
// it differs (THE FOUR, below); every rank's is the caller's loop over ranks (Michael, October 17:
// "a loop of recv and send is more readable"). A load whose registers have a slot dimension reads
// every slot of the entry (a pushed inbox's writers).
//   QUANTIZATION: none, the buffer holds the registers' dtype; int4, it holds their INT4 payload
//     (QUANTIZATION), its values from the slot's start and its scales at the offset its own sizes
//     give (int4_scales_offset), a read dequantizing into its registers' dtype, a write quantizing
//     its registers once whichever ranks it writes them to.
// ENTRY r, A SCALAR LOAD OF THE ARRAY (ptr_array_at; per lane: lane_ptr_array_at): the host writes a
// buffer's array before the launch. A ONE-ENTRY ARRAY (its size the literal 1) may be a pointer's
// own address, `&x_ptr`, so it is read as memory, which folds to the pointer (the constant address
// space's load of a pointer's stack copy is not one).
// THE TWO BODIES (read_buffer, write_buffer) are the memory ops' internals with the address
// functions: the one walk each, over an entry already found, which load and recv (store and send)
// call. Their own reason: two copies of each walk would drift.
// =================================================================================================

// HOW A CALL'S THREADS COVER THE BUFFER, the four's LAYOUT (a template parameter: each value is
// different code):
//   tile: the block over TILE_M rows of packs (LOAD, STORE, SEND AND RECV, above).
//   run, flat: THE FLAT WALK (reduce_scatter.cuh): the buffer's rows from offs_m numbered row-major,
//     slot s's elements from element s * x_stride_slot (a rank's share, or a pushed inbox's
//     writer), the block WORLD rows of kLanes = THREADS_PER_BLOCK / WORLD threads (whole waves, so a
//     row of threads' rank and slot stay wave-uniform), thread t's pack k element (offs_n /
//     kPackElems + t % kLanes + k * kLanes) * kPackElems of its slot; a read past clip_n reads the
//     last below it, a write skips it. run: the buffer contiguous, a slot one run (a uniform base
//     and a 32-bit byte offset a pack); flat: strided, a pack's row and column divided out
//     (element_at). Registers U[1][kPacks][kPackElems].
//   group_run, group_flat: THE INT4 FLAT WALK: the block one row, A GROUP A THREAD (thread t's group
//     offs_n / kInt4Group + t, its kInt4GroupPacks packs consecutive, so a thread quantizes alone),
//     slots and run / flat as above. An INT4 payload is dense (group_run): an element's word at
//     element / 2 bytes, its group's scale at element / kInt4Group from the scales' start. Registers
//     U[1][kInt4GroupPacks][kPackElems] (raw: U[1][kInt4GroupPacks][2]; with a slot dimension,
//     every slot's).
enum class Layout { tile, run, flat, group_run, group_flat };

// A LOAD NEVER BRANCHES: a thread outside a narrow layout loads its clamped, in-bounds place and
// its values go unused. Under a thread compare the load was an exec branch, so every array's
// old registers stayed live across AttnRes's loops (ISA 2026-10-04T16-39-15Z).
// THE READ BODY (load's and recv's): copy `input` (a resolved entry, its slots at their stride)
// into the registers. Registers with a slot dimension, U[S][kRows][kPacks][kPackElems], read slots
// input_slot, input_slot + 1, ... (mod WORLD) of it (a pushed inbox's writers): a pack position at a
// time, every slot's for it, so the address is computed once a position (the other way round cost
// 16 scalar instructions at two packs: ISA 2026-10-01T00-31-14Z). Nothing waits until the
// registers are used, so loads issued here run under other work, a loop's calls included.
// RAW INT4: QUANTIZATION int4 into uint32_t registers U[...][kPacks][2] keeps the payload
// undecoded, [0] the pack's word and [1] its group's scale as fp32 bits: a caller summing every
// rank's words (v_dot2, never a decoded tile of every rank: 172 VGPRs, reduce_scatter.cuh) or
// loading every rank's before decoding any.
template <int WORLD, int TILE_M, int THREADS_PER_BLOCK, int QUANTIZATION, Layout LAYOUT,
          typename OUTPUT>
DINLINE void read_buffer(const void* input, int input_slot, int64_t input_stride_slot,
                         int64_t input_stride_m, int64_t input_stride_n, int input_size_slot,
                         int input_size_m, int input_size_n, int clip_m, int clip_n,
                         OUTPUT& output, int offs_m, int offs_n, int row_step) {
  constexpr bool kSlots = std::rank_v<OUTPUT> == 4;
  constexpr int kCopies = kSlots ? std::extent_v<OUTPUT, 0> : 1;
  constexpr int kRows = std::extent_v<OUTPUT, kSlots ? 1 : 0>;
  constexpr int kPacks = std::extent_v<OUTPUT, kSlots ? 2 : 1>;
  constexpr int kPackElems = std::extent_v<OUTPUT, kSlots ? 3 : 2>;
  using U = std::remove_all_extents_t<OUTPUT>;
  constexpr bool kRaw = std::is_same_v<U, uint32_t>;
  static_assert(!kRaw || (QUANTIZATION == Quantization::int4 && kPackElems == 2),
                "uint32_t registers are an INT4 payload's words and scales");
  // THE LAYOUT, from the tile: kThreadsM rows of threads (TILE_M rows, kRows a thread) of kThreadsN
  // threads, the block's.
  constexpr int kThreadsM = TILE_M / kRows;
  constexpr int kThreadsN = THREADS_PER_BLOCK / kThreadsM;
  static_assert(kThreadsM * kRows == TILE_M, "a tile is whole rows a thread");
  static_assert(kThreadsM * kThreadsN == THREADS_PER_BLOCK, "the layout is the block");
  using V = vec<U, kPackElems>;
  // Slot c of the buffer, as its elements (none) or its bytes (int4).
  using E = std::conditional_t<QUANTIZATION == Quantization::none, const U, const uint8_t>;
  const auto base = [&](int c) -> E* {
    const int slot = kSlots ? (input_slot + c) % WORLD : input_slot;
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG && (slot < 0 || slot >= input_size_slot)) __builtin_trap();
    return reinterpret_cast<E*>(input) + slot * input_stride_slot;
  };
  const int tid = static_cast<int>(threadIdx.x);
  // Thread-row i's row (one row of threads: the block's, so its address stays scalar), a row past
  // clip_m reading the last below it; pack k's column, past clip_n the last below it.
  const auto row_at = [&](int i) {
    return offs_m + (kThreadsM == 1 ? i : tid / kThreadsN + i * kThreadsM) * row_step;
  };
  const auto row_of = [&](int i) { return row_at(i) < clip_m ? row_at(i) : clip_m - 1; };
  // A LOAD SKIPS ROW i only at several rows a thread or rows of threads, whose rows past clip_m
  // would re-read the last over the links. One row's is real wherever a kernel makes one, and a
  // branch around its loads cost registers (126 -> 178 VGPRs, spilling at 16384: 23-46-07Z).
  const auto skips = [&](int i) { return !(kRows == 1 && kThreadsM == 1) && row_at(i) >= clip_m; };
  // A pack's elements in the buffer: the registers' (raw INT4 registers hold a word and a scale
  // for a pack of kInt4Group / kInt4GroupPacks elements).
  constexpr int kElems = kRaw ? kInt4Group / kInt4GroupPacks : kPackElems;
  const auto pack_of = [&](int k) {
    const int n = offs_n / kElems + tid % kThreadsN + k * kThreadsN;
    return n < clip_n / kElems ? n : clip_n / kElems - 1;
  };
  // Element e of an [*, n] buffer numbered row-major (the flat walk's address): a pack's row and
  // column divided out, unless the rows are back to back.
  [[maybe_unused]] const auto element_at = [](auto* ptr, int64_t stride_m, int64_t stride_n,
                                              int64_t e, int n) {
    if (stride_m == int64_t{n} * stride_n) return ptr + e * stride_n;
    const int64_t row = e < (int64_t{1} << 31)
                            ? int64_t{static_cast<uint32_t>(e) / static_cast<uint32_t>(n)}
                            : e / n;
    return ptr + row * stride_m + (e - row * n) * stride_n;
  };
  constexpr bool kGroup = LAYOUT == Layout::group_run || LAYOUT == Layout::group_flat;
  constexpr bool kRun = LAYOUT == Layout::run || LAYOUT == Layout::group_run;
  if constexpr (LAYOUT != Layout::tile && !kGroup) {
    // THE FLAT WALK, a pack strided by kLanes, one copy (its rank and slot the row of threads').
    static_assert(QUANTIZATION == Quantization::none && !kSlots && kRows == 1,
                  "the flat walk reads one exact copy a row of threads");
    constexpr int kLanes = THREADS_PER_BLOCK / WORLD;
    static_assert(kLanes * WORLD == THREADS_PER_BLOCK, "a block is a group of threads a rank");
    static_assert(kLanes % kWaveSize == 0, "a rank's copy is whole waves");
    static_assert(sizeof(V) == sizeof(u32x4), "a pack is one 16-byte global access");
    const U* const from = reinterpret_cast<const U*>(input) + int64_t{offs_m} * input_stride_m;
    const int64_t first = int64_t{input_slot} * input_stride_slot;
    if constexpr (kRun) {
      // The last pack, the run's start, then the column: the order the two-shot's instructions
      // came out in (ISA 2026-10-15).
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        const int last = clip_n / kPackElems - 1;
        const V* run = reinterpret_cast<const V*>(from + first * input_stride_n);
        const int n = offs_n / kPackElems + tid % kLanes + k * kLanes;
        const int col = n < clip_n / kPackElems ? n : last;
        uint32_t offset = static_cast<uint32_t>(col) * uint32_t{sizeof(V)};
        asm volatile("" : "+v"(offset));
        const u32x4 raw = *(const global_u32x4*)(reinterpret_cast<const char*>(run) + offset);
        __builtin_memcpy(output[0][k], &raw, sizeof(raw));
      }
    } else {
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        // The pack's column, past the clip the last in it (every load in bounds, none under a
        // branch).
        const int n = offs_n / kPackElems + tid % kLanes + k * kLanes;
        const int col = n < clip_n / kPackElems ? n : clip_n / kPackElems - 1;
        const u32x4 raw = *(const global_u32x4*)(element_at(
            from, input_stride_m, input_stride_n, first + int64_t{col} * kPackElems, input_size_n));
        __builtin_memcpy(output[0][k], &raw, sizeof(raw));
      }
    }
    // ISSUED HERE: no memory operation nor instruction crosses this point, so every load above is
    // in flight before anything below runs (the scheduler alone sank peer loads past the first
    // adds: ISA 2026-10-01T00-39-51Z).
    asm volatile("" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
  } else if constexpr (kGroup) {
    static_assert(kRows == 1 && kPacks == kInt4GroupPacks, "a group a thread");
    // A UNIFORM BASE PLUS A 32-BIT BYTE OFFSET, the offset opaque so the address is built at its
    // load (saddr), not hoisted and held as a 64-bit vector address.
    const int g = offs_n / kInt4Group + tid;
    const int at = min(g, clip_n / kInt4Group - 1);
    if constexpr (QUANTIZATION == Quantization::none) {
      // ONE COPY, exact: its group's packs. Its scheduling is the caller's (the int4 push keeps two
      // ranks' groups in flight: reduce_scatter.cuh).
      static_assert(!kSlots, "an exact group is one copy");
      const U* const from = reinterpret_cast<const U*>(input) + int64_t{offs_m} * input_stride_m;
      const int64_t first = int64_t{input_slot} * input_stride_slot;
      const V* group;
      if constexpr (kRun) {
        uint32_t offset = static_cast<uint32_t>(at) * (kInt4Group * sizeof(U));
        asm volatile("" : "+v"(offset));
        group = reinterpret_cast<const V*>(
            reinterpret_cast<const char*>(from + first * input_stride_n) + offset);
      } else
        group = reinterpret_cast<const V*>(element_at(from, input_stride_m, input_stride_n,
                                                      first + int64_t{at} * kInt4Group,
                                                      input_size_n));
      // A PACK OF A 16-BIT DTYPE one 16-byte load; of fp32 (a norm's weight, a pack of the row's
      // dtype wide) its 16-byte halves, as the tile's wider pack is read.
      static_assert(sizeof(V) % sizeof(u32x4) == 0, "a pack is whole 16-byte loads");
#pragma unroll
      for (int k = 0; k < kInt4GroupPacks; ++k) {
        if constexpr (sizeof(V) == sizeof(u32x4)) {
          const u32x4 raw = *(const global_u32x4*)(group + k);
          __builtin_memcpy(output[0][k], &raw, sizeof(raw));
        } else {
#pragma unroll
          for (int h = 0; h < static_cast<int>(sizeof(V) / sizeof(u32x4)); ++h) {
            const u32x4 raw = *(const global_u32x4*)(reinterpret_cast<const char*>(group + k) +
                                                     h * sizeof(u32x4));
            __builtin_memcpy(reinterpret_cast<char*>(output[0][k]) + h * sizeof(u32x4), &raw,
                             sizeof(raw));
          }
        }
      }
    } else {
      // EVERY SLOT'S PAYLOAD, raw: its group's four words (one 16-byte load) and its scale, the
      // position once in a vector register, all in flight before the first is summed.
      static_assert(kRaw && kRun, "an INT4 group is read raw from a dense payload");
      const int64_t scales_offset = int4_scales_offset(input_size_m, input_size_n);
      const uint8_t* values[kCopies];
      const __fp16* scales[kCopies];
#pragma unroll
      for (int c = 0; c < kCopies; ++c) {
        const int64_t slot =
            int64_t{kSlots ? (input_slot + c) % WORLD : input_slot} * input_stride_slot;
        values[c] = reinterpret_cast<const uint8_t*>(input) + slot / 2;
        scales[c] = reinterpret_cast<const __fp16*>(reinterpret_cast<const uint8_t*>(input) +
                                                    scales_offset) +
                    slot / kInt4Group;
      }
      uint32_t value_at = static_cast<uint32_t>(at) * 16;
      uint32_t scale_at = static_cast<uint32_t>(at) * kInt4ScaleBytes;
      asm volatile("" : "+v"(value_at));
      asm volatile("" : "+v"(scale_at));
#pragma unroll
      for (int c = 0; c < kCopies; ++c) {
        const u32x4 words = *(const global_u32x4*)(values[c] + value_at);
        const uint32_t scale = __builtin_bit_cast(
            uint32_t, static_cast<float>(
                          *(const global_f16*)(reinterpret_cast<const char*>(scales[c]) + scale_at)));
        auto& out = [&]() -> U(&)[kPacks][kPackElems] {
          if constexpr (kSlots) return output[c][0]; else return output[0];
        }();
#pragma unroll
        for (int k = 0; k < kInt4GroupPacks; ++k) {
          out[k][0] = words[k];
          out[k][1] = scale;
        }
      }
      // ISSUED HERE: every slot's group in flight before the first is used.
      asm volatile("" ::: "memory");
      __builtin_amdgcn_sched_barrier(0);
    }
  } else if constexpr (QUANTIZATION == Quantization::none && !kSlots) {
    const V* at = reinterpret_cast<const V*>(base(0));
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      // A ROW OF THE BLOCK'S (one row of threads) is uniform: its 64-bit start stays scalar. A ROW
      // A LANE'S (rows of threads) goes in the 32-bit offset below, beside the column, so the
      // address is the uniform base plus the offset (saddr) and nothing is built in the registers
      // the load writes. As a 64-bit vector address (and then as the opaque offset's copy) it was
      // built a call in those registers, after the call before had issued; a load still pending
      // there among stores is a vmcnt(0) (a wait for every load and store), so each peer's load
      // waited for the one before it: push two-shot rms_norm 20 -> 29 vmcnt(0), I10 against I8.
      // So the lane's offsets are made opaque BEFORE the skip, by a non-volatile asm: identical in
      // every call, one register the calls share (CSE), no copy a call. A row past 4 GiB of its
      // buffer traps under debug.
      constexpr bool kLaneRow = !(kRows == 1 && kThreadsM == 1);
      uint32_t lane_offset[kPacks];
      if constexpr (kLaneRow) {
        if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
            int64_t{row_of(i)} * input_stride_m * int64_t{sizeof(U)} +
                    input_size_n * int64_t{sizeof(U)} >
                (int64_t{1} << 32))
          __builtin_trap();
        const uint32_t row_bytes = static_cast<uint32_t>(row_of(i)) *
                                   static_cast<uint32_t>(input_stride_m * int64_t{sizeof(U)});
#pragma unroll
        for (int k = 0; k < kPacks; ++k) {
          lane_offset[k] = row_bytes + static_cast<uint32_t>(pack_of(k)) * uint32_t{sizeof(V)};
          asm("" : "+v"(lane_offset[k]));
        }
      }
      if (skips(i)) continue;
      const V* row = kLaneRow ? at : at + int64_t{row_of(i)} * (input_stride_m / kPackElems);
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        // A ROW'S PACK k AS A UNIFORM BASE PLUS AN UNSIGNED 32-BIT BYTE OFFSET: the form
        // global_load takes with the base in scalar registers (saddr), so every tensor sharing the
        // columns shares one offset register. A signed column added to the pointer was a 64-bit
        // vector address a tensor a pack, 28 VGPRs held across AttnRes's loops (liveness,
        // 2026-10-04T15-28-51Z). THE ADDRESS IS BUILT AT ITS LOAD: the offset made opaque here, so
        // the 64-bit sum is not hoisted and held a tensor a pack (AttnRes 90 -> 70 VGPRs).
        uint32_t offset = 0;
        if constexpr (kLaneRow)
          offset = lane_offset[k];
        else {
          offset = static_cast<uint32_t>(pack_of(k)) * uint32_t{sizeof(V)};
          asm volatile("" : "+v"(offset));
        }
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
    // 2026-10-03T21-00-27Z). It waits for nothing: a loop of calls has every call's loads out.
    asm volatile("" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
  } else if constexpr (QUANTIZATION == Quantization::none) {
    const U* at_copy[kCopies];
#pragma unroll
    for (int c = 0; c < kCopies; ++c) at_copy[c] = base(c);
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      if (skips(i)) continue;
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        // THE POSITION ONCE, IN A VECTOR REGISTER: left to the compiler, the row's uniform part
        // was added to every slot's base in scalar registers, an add and a carry a slot (16
        // instructions a tile at eight ranks: ISA 2026-10-11); held in a VGPR, each slot's address
        // is one shift-and-add of it.
        int64_t at = int64_t{row_of(i)} * (input_stride_m / kPackElems) + pack_of(k);
        asm volatile("" : "+v"(at));
#pragma unroll
        for (int c = 0; c < kCopies; ++c) {
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
    static_assert(kThreadsN % kInt4GroupPacks == 0, "a group is 4 lanes' packs");
    // The buffer's INT4 payload, from its own sizes (the INT4 payload layout).
    const int64_t values_stride_m = input_size_n / 2;
    const int64_t scales_stride_m = input_size_n / kInt4Group;
    const int64_t scales_at = int4_scales_offset(input_size_m, input_size_n);
    // Every slot's words and scales, the position once a pack in a vector register (32-BIT BYTE
    // OFFSETS: a quantized buffer is ours, at most the box's, select's exchange_sizes), each
    // slot's word and scale an add of it; all in flight before the first is decoded.
    const uint8_t* values[kCopies];
    const char* scales[kCopies];
#pragma unroll
    for (int c = 0; c < kCopies; ++c) values[c] = base(c);
#pragma unroll
    for (int c = 0; c < kCopies; ++c) scales[c] = reinterpret_cast<const char*>(values[c] + scales_at);
    uint32_t words[kCopies][kRows][kPacks];
    float group_scales[kCopies][kRows][kPacks];
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      if (skips(i)) continue;
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        const int64_t row = row_of(i);
        const int pack = pack_of(k);
        uint32_t value_at = static_cast<uint32_t>(row * values_stride_m) +
                            static_cast<uint32_t>(pack) * kInt4WordBytes;
        uint32_t scale_at = static_cast<uint32_t>(
            (row * scales_stride_m + pack / kInt4GroupPacks) * kInt4ScaleBytes);
        asm volatile("" : "+v"(value_at));
        asm volatile("" : "+v"(scale_at));
#pragma unroll
        for (int c = 0; c < kCopies; ++c) {
          words[c][i][k] = *(const global_u32*)(values[c] + value_at);
          group_scales[c][i][k] = *(const global_f16*)(scales[c] + scale_at);
        }
      }
    }
    asm volatile("" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int c = 0; c < kCopies; ++c)
#pragma unroll
      for (int i = 0; i < kRows; ++i)
#pragma unroll
        for (int k = 0; k < kPacks; ++k) {
          auto& out = [&]() -> U(&)[kPackElems] {
            if constexpr (kSlots) return output[c][i][k]; else return output[i][k];
          }();
          if constexpr (kRaw) {
            out[0] = words[c][i][k];
            out[1] = __builtin_bit_cast(uint32_t, group_scales[c][i][k]);
          } else {
            dequantize(words[c][i][k], group_scales[c][i][k], out);
          }
        }
  }
}

// THE WRITE BODY (store's and send's): the registers into `output` (a resolved entry), the rows
// below clip_m and the packs below clip_n, from the threads of the layout (THREADS_PER_BLOCK the
// block's, so a layout of all of it compares nothing: threadIdx.x < THREADS_PER_BLOCK is not folded
// from the launch bounds, a compare and a branch around every store, ISA 2026-10-15). int4:
// registers of a dtype are quantized here; uint32_t registers U[kRows][kPacks][2] are the payload
// already quantized ([0] the word, [1] the group's scale as fp32 bits), so a caller sending to
// every rank quantizes once, before its loop.
template <int WORLD, int TILE_M, int THREADS_PER_BLOCK, int QUANTIZATION, Layout LAYOUT,
          typename INPUT>
DINLINE void write_buffer(const INPUT& input, void* output, int output_slot,
                          int64_t output_stride_slot, int64_t output_stride_m,
                          int64_t output_stride_n, int output_size_slot, int output_size_m,
                          int output_size_n, int clip_m, int clip_n, int offs_m, int offs_n,
                          int row_step) {
  constexpr int kRows = std::extent_v<INPUT, 0>;
  constexpr int kPacks = std::extent_v<INPUT, 1>;
  constexpr int kPackElems = std::extent_v<INPUT, 2>;
  using U = std::remove_all_extents_t<INPUT>;
  constexpr bool kRaw = std::is_same_v<U, uint32_t>;
  static_assert(!kRaw || (QUANTIZATION == Quantization::int4 && kPackElems == 2),
                "uint32_t registers are an INT4 payload's words and scales");
  // THE LAYOUT, from the tile: kThreadsM rows of threads (TILE_M rows, kRows a thread) of kThreadsN
  // threads, the block's.
  constexpr int kThreadsM = TILE_M / kRows;
  constexpr int kThreadsN = THREADS_PER_BLOCK / kThreadsM;
  static_assert(kThreadsM * kRows == TILE_M, "a tile is whole rows a thread");
  static_assert(kThreadsM * kThreadsN == THREADS_PER_BLOCK, "the layout is the block");
  using V = vec<U, kPackElems>;
  using E = std::conditional_t<QUANTIZATION == Quantization::none, U, uint8_t>;
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG && (output_slot < 0 || output_slot >= output_size_slot)) __builtin_trap();
  [[maybe_unused]] E* const at_copy = reinterpret_cast<E*>(output) + output_slot * output_stride_slot;
  const int tid = static_cast<int>(threadIdx.x);
  const auto row_at = [&](int i) {
    return offs_m + (kThreadsM == 1 ? i : tid / kThreadsN + i * kThreadsM) * row_step;
  };
  const auto col_at = [&](int k) { return offs_n / kPackElems + tid % kThreadsN + k * kThreadsN; };
  // Element e of an [*, n] buffer numbered row-major (the flat walk's address): a pack's row and
  // column divided out, unless the rows are back to back.
  [[maybe_unused]] const auto element_at = [](auto* ptr, int64_t stride_m, int64_t stride_n,
                                              int64_t e, int n) {
    if (stride_m == int64_t{n} * stride_n) return ptr + e * stride_n;
    const int64_t row = e < (int64_t{1} << 31)
                            ? int64_t{static_cast<uint32_t>(e) / static_cast<uint32_t>(n)}
                            : e / n;
    return ptr + row * stride_m + (e - row * n) * stride_n;
  };
  constexpr bool kGroup = LAYOUT == Layout::group_run || LAYOUT == Layout::group_flat;
  constexpr bool kRun = LAYOUT == Layout::run || LAYOUT == Layout::group_run;
  if constexpr (LAYOUT != Layout::tile && !kGroup) {
    // THE FLAT WALK, a pack strided by kLanes, one copy (its rank and slot the row of threads').
    static_assert(QUANTIZATION == Quantization::none && kRows == 1,
                  "the flat walk writes one exact copy a row of threads");
    constexpr int kLanes = THREADS_PER_BLOCK / WORLD;
    static_assert(kLanes * WORLD == THREADS_PER_BLOCK, "a block is a group of threads a rank");
    static_assert(kLanes % kWaveSize == 0, "a rank's copy is whole waves");
    static_assert(sizeof(V) == sizeof(u32x4), "a pack is one 16-byte global access");
    U* const to = reinterpret_cast<U*>(output) + int64_t{offs_m} * output_stride_m;
    const int64_t first = int64_t{output_slot} * output_stride_slot;
    U* const run = to + first * output_stride_n;
#pragma unroll
    for (int k = 0; k < kPacks; ++k) {
      const int c = offs_n / kPackElems + tid % kLanes + k * kLanes;
      if (c >= clip_n / kPackElems) continue;
      V* p;
      if constexpr (kRun) {
        // A uniform base plus a 32-bit byte offset, opaque so the address is built at its store
        // (c is below the clip, so no clamp).
        uint32_t offset = static_cast<uint32_t>(c) * uint32_t{sizeof(V)};
        asm volatile("" : "+v"(offset));
        p = reinterpret_cast<V*>(reinterpret_cast<char*>(run) + offset);
      } else {
        p = reinterpret_cast<V*>(element_at(to, output_stride_m, output_stride_n,
                                            first + int64_t{c} * kPackElems, output_size_n));
      }
      u32x4 raw;
      __builtin_memcpy(&raw, input[0][k], sizeof(raw));
      *(global_u32x4*)(p) = raw;
    }
  } else if constexpr (kGroup) {
    static_assert(kRows == 1 && kPacks == kInt4GroupPacks, "a group a thread");
    const int g = offs_n / kInt4Group + tid;
    if constexpr (QUANTIZATION == Quantization::none) {
      // EXACT: the group's packs at their place.
      if (g >= clip_n / kInt4Group) return;
      U* const to = reinterpret_cast<U*>(output) + int64_t{offs_m} * output_stride_m;
      const int64_t first = int64_t{output_slot} * output_stride_slot;
      V* group;
      if constexpr (kRun) {
        uint32_t offset = static_cast<uint32_t>(g) * (kInt4Group * sizeof(U));
        asm volatile("" : "+v"(offset));
        group = reinterpret_cast<V*>(reinterpret_cast<char*>(to + first * output_stride_n) + offset);
      } else
        group = reinterpret_cast<V*>(element_at(to, output_stride_m, output_stride_n,
                                                first + int64_t{g} * kInt4Group, output_size_n));
#pragma unroll
      for (int k = 0; k < kInt4GroupPacks; ++k) {
        u32x4 raw;
        __builtin_memcpy(&raw, input[0][k], sizeof(raw));
        *(global_u32x4*)(group + k) = raw;
      }
    } else {
      // INT4: the group's four words one 16-byte store and its scale, into the dense payload at
      // the slot's place; quantized here by its thread alone, or already (raw registers).
      static_assert(kRun, "an INT4 payload is dense");
      uint32_t words[kInt4GroupPacks];
      float scale;
      if constexpr (kRaw) {
#pragma unroll
        for (int k = 0; k < kInt4GroupPacks; ++k) words[k] = input[0][k][0];
        scale = __builtin_bit_cast(float, input[0][0][1]);
      } else {
        quantize(input[0], words, scale);
      }
      if (g < clip_n / kInt4Group) {
        const int64_t slot = int64_t{output_slot} * output_stride_slot;
        uint8_t* const values = reinterpret_cast<uint8_t*>(output) + slot / 2;
        __fp16* const scales =
            reinterpret_cast<__fp16*>(reinterpret_cast<uint8_t*>(output) +
                                      int4_scales_offset(output_size_m, output_size_n)) +
            slot / kInt4Group;
        uint32_t value_at = static_cast<uint32_t>(g) * 16;
        uint32_t scale_at = static_cast<uint32_t>(g) * kInt4ScaleBytes;
        asm volatile("" : "+v"(value_at));
        asm volatile("" : "+v"(scale_at));
        *(global_u32x4*)(values + value_at) = u32x4{words[0], words[1], words[2], words[3]};
        *(global_f16*)(reinterpret_cast<char*>(scales) + scale_at) = static_cast<__fp16>(scale);
      }
    }
  } else if constexpr (QUANTIZATION == Quantization::none) {
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      if (row_at(i) >= clip_m) continue;
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        if (col_at(k) >= clip_n / kPackElems) continue;
        V v;
        __builtin_memcpy(&v, input[i][k], sizeof(v));
        // A uniform base plus a 32-bit byte offset (load's).
        uint32_t offset = static_cast<uint32_t>(col_at(k)) * uint32_t{sizeof(V)};
        asm volatile("" : "+v"(offset));
        V* row = reinterpret_cast<V*>(at_copy) + int64_t{row_at(i)} * (output_stride_m / kPackElems);
        V* p = reinterpret_cast<V*>(reinterpret_cast<char*>(row) + offset);
        if constexpr (sizeof(V) == 16) {
          u32x4 raw;
          __builtin_memcpy(&raw, &v, 16);
          *(global_u32x4*)(p) = raw;
        } else {
          *p = v;
        }
      }
    }
  } else {
    static_assert(kThreadsN % kInt4GroupPacks == 0, "a group is 4 lanes' packs");
    constexpr int kInt4PackElems = kInt4Group / kInt4GroupPacks;
    uint32_t words[kRows][kPacks];
    float scales[kRows][kPacks];
#pragma unroll
    for (int i = 0; i < kRows; ++i)
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        if constexpr (kRaw) {
          words[i][k] = input[i][k][0];
          scales[i][k] = __builtin_bit_cast(float, input[i][k][1]);
        } else {
          quantize(reinterpret_cast<const U(&)[1][kPackElems]>(input[i][k]),
                   reinterpret_cast<uint32_t(&)[1]>(words[i][k]), scales[i][k]);
        }
      }
    const int64_t values_stride_m = output_size_n / 2;
    const int64_t scales_stride_m = output_size_n / kInt4Group;
    const int64_t scales_at = int4_scales_offset(output_size_m, output_size_n);
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      if (row_at(i) >= clip_m) continue;
      const int64_t row = row_at(i);
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        const int pack = offs_n / kInt4PackElems + tid % kThreadsN + k * kThreadsN;
        if (pack >= clip_n / kInt4PackElems) continue;
        uint32_t value_at = static_cast<uint32_t>(row * values_stride_m) +
                            static_cast<uint32_t>(pack) * kInt4WordBytes;
        uint32_t scale_at = static_cast<uint32_t>(
            (row * scales_stride_m + pack / kInt4GroupPacks) * kInt4ScaleBytes);
        asm volatile("" : "+v"(value_at));
        asm volatile("" : "+v"(scale_at));
        *(global_u32*)(at_copy + value_at) = words[i][k];
        if (pack % kInt4GroupPacks == 0)
          *(global_f16*)(at_copy + scales_at + scale_at) = static_cast<__fp16>(scales[i][k]);
      }
    }
  }
}

// THE FOUR, one form: the buffer's pointer array (`x_ptr_array`, `x_ptr_array_size` entries; a
// buffer only this rank has is `&x_ptr, 1`), the entry `x_rank`, then the slot, strides, sizes,
// clip, registers and place. load and store take THIS RANK'S entry (a scalar read of the array,
// ptr_array_at; a one-entry array folds to the pointer). recv and send take any rank's, and the
// rank may differ by lane (a row of threads' rank, each pack's owner): every rank's is the caller's
// loop, `for (int r = 0; r < WORLD; ++r) recv(..., r, ..., got[r], ...)`, every call's loads out
// before the first is used (none of them waits).
// WHICH PATH A PEER'S ENTRY TAKES, recv's and send's own decision, by the wave's ranks: all lanes
// alike (the ballot of rank != the first lane's is empty), a scalar read of the entry
// (ptr_array_at) and the uniform-base form of every address; else every entry read and the lane's
// selected (lane_ptr_array_at). A rank the compiler knows (a loop's r, a literal) folds the test
// away; a row of threads' rank of whole waves takes the scalar path at run time.
// EVIDENCE for the scalar path where it can: all_gather's whole-wave rows through the select were
// +86-143 VALU a plain two-shot (I8 ISA 2026-10-18T00-13-04Z).

// LOAD: this rank's buffer (entry input_rank) into a thread's registers.
template <int WORLD, int TILE_M, int THREADS_PER_BLOCK, int QUANTIZATION, Layout LAYOUT,
          typename PTR, typename OUTPUT>
DINLINE void load(PTR const* input_ptr_array, int input_ptr_array_size, int input_rank,
                  int input_slot, int64_t input_stride_slot, int64_t input_stride_m,
                  int64_t input_stride_n, int input_size_slot, int input_size_m, int input_size_n,
                  int clip_m, int clip_n, OUTPUT& output, int offs_m, int offs_n, int row_step) {
  // The entries as plain pointers (a kernel's own pointer is __restrict__).
  const auto* const entries = (decltype(+input_ptr_array[0]) const*)input_ptr_array;
  const void* const input = __builtin_constant_p(input_ptr_array_size) && input_ptr_array_size == 1
                                ? entries[0]
                                : ptr_array_at(entries, input_ptr_array_size, input_rank);
  read_buffer<WORLD, TILE_M, THREADS_PER_BLOCK, QUANTIZATION, LAYOUT>(
      input, input_slot, input_stride_slot, input_stride_m, input_stride_n, input_size_slot,
      input_size_m, input_size_n, clip_m, clip_n, output, offs_m, offs_n, row_step);
}

// RECV: rank input_rank's buffer (a peer's, over the fabric; the rank may differ by lane) into a
// thread's registers.
template <int WORLD, int TILE_M, int THREADS_PER_BLOCK, int QUANTIZATION, Layout LAYOUT,
          typename PTR, typename OUTPUT>
DINLINE void recv(PTR const* input_ptr_array, int input_ptr_array_size, int input_rank,
                  int input_slot, int64_t input_stride_slot, int64_t input_stride_m,
                  int64_t input_stride_n, int input_size_slot, int input_size_m, int input_size_n,
                  int clip_m, int clip_n, OUTPUT& output, int offs_m, int offs_n, int row_step) {
  const auto* const entries = (decltype(+input_ptr_array[0]) const*)input_ptr_array;
  const int first = __builtin_amdgcn_readfirstlane(input_rank);
  if (__builtin_amdgcn_ballot_w64(input_rank != first) == 0)
    read_buffer<WORLD, TILE_M, THREADS_PER_BLOCK, QUANTIZATION, LAYOUT>(
        ptr_array_at(entries, input_ptr_array_size, first), input_slot, input_stride_slot,
        input_stride_m, input_stride_n, input_size_slot, input_size_m, input_size_n, clip_m,
        clip_n, output, offs_m, offs_n, row_step);
  else
    read_buffer<WORLD, TILE_M, THREADS_PER_BLOCK, QUANTIZATION, LAYOUT>(
        lane_ptr_array_at<void, WORLD>(entries, input_ptr_array_size, input_rank), input_slot,
        input_stride_slot, input_stride_m, input_stride_n, input_size_slot, input_size_m,
        input_size_n, clip_m, clip_n, output, offs_m, offs_n, row_step);
}

// STORE: a thread's registers into this rank's buffer (entry output_rank).
template <int WORLD, int TILE_M, int THREADS_PER_BLOCK, int QUANTIZATION, Layout LAYOUT,
          typename INPUT, typename PTR>
DINLINE void store(const INPUT& input, PTR const* output_ptr_array, int output_ptr_array_size,
                   int output_rank, int output_slot, int64_t output_stride_slot,
                   int64_t output_stride_m, int64_t output_stride_n, int output_size_slot,
                   int output_size_m, int output_size_n, int clip_m, int clip_n, int offs_m,
                   int offs_n, int row_step) {
  const auto* const entries = (decltype(+output_ptr_array[0]) const*)output_ptr_array;
  void* const output = __builtin_constant_p(output_ptr_array_size) && output_ptr_array_size == 1
                           ? entries[0]
                           : ptr_array_at(entries, output_ptr_array_size, output_rank);
  write_buffer<WORLD, TILE_M, THREADS_PER_BLOCK, QUANTIZATION, LAYOUT>(
      input, output, output_slot, output_stride_slot, output_stride_m, output_stride_n,
      output_size_slot, output_size_m, output_size_n, clip_m, clip_n, offs_m, offs_n, row_step);
}

// SEND: a thread's registers into rank output_rank's buffer (a peer's, over the fabric; the rank
// may differ by lane). Posted: the next ordered barrier's release completes and publishes it
// (BARRIER).
template <int WORLD, int TILE_M, int THREADS_PER_BLOCK, int QUANTIZATION, Layout LAYOUT,
          typename INPUT, typename PTR>
DINLINE void send(const INPUT& input, PTR const* output_ptr_array, int output_ptr_array_size,
                  int output_rank, int output_slot, int64_t output_stride_slot,
                  int64_t output_stride_m, int64_t output_stride_n, int output_size_slot,
                  int output_size_m, int output_size_n, int clip_m, int clip_n, int offs_m,
                  int offs_n, int row_step) {
  const auto* const entries = (decltype(+output_ptr_array[0]) const*)output_ptr_array;
  const int first = __builtin_amdgcn_readfirstlane(output_rank);
  if (__builtin_amdgcn_ballot_w64(output_rank != first) == 0)
    write_buffer<WORLD, TILE_M, THREADS_PER_BLOCK, QUANTIZATION, LAYOUT>(
        input, ptr_array_at(entries, output_ptr_array_size, first), output_slot,
        output_stride_slot, output_stride_m, output_stride_n, output_size_slot, output_size_m,
        output_size_n, clip_m, clip_n, offs_m, offs_n, row_step);
  else
    write_buffer<WORLD, TILE_M, THREADS_PER_BLOCK, QUANTIZATION, LAYOUT>(
        input, lane_ptr_array_at<void, WORLD>(entries, output_ptr_array_size, output_rank),
        output_slot, output_stride_slot, output_stride_m, output_stride_n, output_size_slot,
        output_size_m, output_size_n, clip_m, clip_n, offs_m, offs_n, row_step);
}

}  // namespace hip_custom_communicator
