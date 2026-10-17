// SPDX-License-Identifier: MIT
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE REDUCE-SCATTER, written once with the common ops (load, store, barrier, block_reduce, the
// codec): from every rank's input to every rank's sums in the partials, visible to its peers. Its
// variations are template parameters, its walks written inline (the head of this file says why
// each walk is the one it is).

#pragma once

#include "common.cuh"

namespace hip_custom_communicator {

// THE COLLECTIVES (reduce_scatter here, all_gather in all_gather.cuh, which walks the same
// walk): one tested and tuned way to do each, which the kernels call rather than write again
// (the fused AttnRes pull's own reduce-scatter took 1.6x the plain two-shot's cycles for its
// bytes, thread traces 2026-10-04T21-03-09Z and 21-26-11Z). Their vocabulary (Direction, Walk,
// Accumulate) is theirs, here; types.cuh's forcing names Direction too.

// THE LOADS ARE THE COMMUNICATION: a rank's pointer points into its memory,
// opened in this process over IPC (handle.cuh), so loading reads it over the links; the sum
// and the store are local. The kernel syncs around these calls (the barriers):
// before, every peer's data is ready; between the reduce-scatter and the
// all-gather, every rank's sums are visible. RANKS ARE READ ROTATED, rank + w's
// w-th, so at any moment the GPUs read different peers; each range or slice is
// summed by one rank, so the order differing by rank is harmless.
//
// THE FLAT WALK (the plain two-shot): an elementwise [R, M, N] problem -- R
// ranks' copies of an [M, N] buffer, each at its own strides -- WALKED AS
// [R, 1, M*N]: the rows are only the buffer's layout, so the elements are
// numbered row-major (e = row * N + col, reduce_scatter's element_at) and each rank's SHARE is one
// contiguous range of them, a TILE_N chunk a step. WHY: the links and HBM
// reward long reads from each peer. Cut a row at a time, a rank's share was its
// slice of every row -- 1.8 KB runs at N = 7168, the plain all-reduce +2%
// against the flat walk; 128-column tiles, 256 B runs, +3-7% (2026-10-11) --
// while a contiguous buffer's range is one run of the whole share. IT SERVES
// BADLY a strided buffer: its runs are its rows wherever it has them, and each
// pack's row and column are divided out. The walk is the same for every step
// and every rank, so a block reads in the all-gather what the same block on
// each rank wrote in the reduce-scatter. A RANK'S COPY OF A CHUNK IS LANES =
// THREADS_PER_BLOCK / WORLD threads, whole waves (its rank's pointer one scalar select;
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

// [PUSH OR STAGE] BARRIER(UNTIL, 0) SUM BARRIER(visible, 1): before the first barrier, a pushed
// walk stores every rank's share of this rank's input in that rank's staging at this rank's slot,
// and an int4 rows walk stages this rank's rows quantized in its own staging (the rows this
// block's twin on every rank reads, tiles of TILE_M rows blockIdx.x, blockIdx.x + gridDim.x, ...);
// after the second, every rank's sums are in the partials and every peer is done reading.
//
// WALK::flat (the plain two-shot): rows offs_m.. offs_m + input_size_m of the [*, input_size_n]
// problem numbered row-major, each rank's SHARE one contiguous range of them, a TILE_N chunk a step,
// a rank's copy of a chunk LANES = THREADS_PER_BLOCK / WORLD threads (whole waves, its rank rank + w,
// rotated). PULL (ACCUMULATE lds, none): every rank's input read in place through the INPUT TABLE
// (input_ptr_array at its strides). PUSH (none, lds): this rank's input (input_ptr) pushed into
// every rank's staging inbox, [WORLD, slot] (writer w's slot at row w, staging_stride_m elements a
// slot), then the slots summed locally. PUSH (int4, registers): A GROUP A THREAD, each block a
// contiguous range of every share's groups (ceil(groups / grid), THREADS_PER_BLOCK a step), quantized here
// into every rank's staging INT4 payload at this rank's place (writer w's at share w's), the
// writers' groups summed from the words (no pack of every writer decoded) and quantized into this rank's partials.
// The plain all-reduce at 72 blocks: 71.8 / 109.0 / 178.2 / 286.6 us at 1024-8192 tokens of
// 7168, against the exact 75.5 / 142.0 / 273.3 / 535.4 (2026-10-16T03-14-06Z).
//
// WALK::rows (the fused ops' two-shots): this rank's columns of every row, equal slices of whole
// packs a rank (a late rank's short or empty). ACCUMULATE lds (pull): a chunk of a row at a time,
// WORLD rows of LANES threads, row w rank rank + w's copy picked per lane (Ranks::lane), summed
// over the rows through LDS into this rank's partials. ACCUMULATE registers: a row of threads as
// wide as a rank's slice of a TILE_N row (whole waves), a pack a thread, as many rows at once as the
// block holds, every rank's pack summed in registers in rank order; push stores the sum in every
// rank's partials, pull in this rank's. none reads every rank's input in place; int4 every rank's
// staged INT4 payload, an INT4 payload's layout from its own buffer's sizes.
template <int WORLD, int THREADS_PER_BLOCK, int TILE_M, int TILE_N, Walk WALK, Accumulate ACCUMULATE,
          Direction DIRECTION, int QUANTIZATION, bool CONTIGUOUS, Until UNTIL, typename T>
DINLINE void reduce_scatter(
    const T* input_ptr, int64_t input_stride_m, int64_t input_stride_n, int input_size_m,
    int input_size_n, const T* const* input_ptr_array, int input_ptr_array_size,
    int64_t input_ptr_array_stride_m, int64_t input_ptr_array_stride_n,
    char* const* staging_ptr_array, int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    char* const* partials_ptr_array, int partials_ptr_array_size, int64_t partials_stride_m,
    int64_t partials_stride_n, int partials_size_m, int partials_size_n,
    uint32_t* const* signals_ptr_array, int signals_ptr_array_size, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer, int signals_size_parity,
    int signals_size_block, int signals_size_writer, uint32_t* status_ptr,
    int64_t status_stride_field, int status_size_field, int offs_m, int rank,
    uint64_t timeout_ticks) {
  constexpr int P = traits<T>::N;
  constexpr bool kInt4 = QUANTIZATION == Quantization::int4;
  using V = vec<T, P>;
  // A PACK IS ONE 16-BYTE GLOBAL INSTRUCTION: loaded and stored as a global u32x4 (global_load_dwordx4
  // / global_store_dwordx4), where a pointer read out of an array is otherwise flat_*.
  static_assert(sizeof(V) == sizeof(u32x4), "a pack is one 16-byte global access");
  // Element e of an [*, n] buffer numbered row-major (the flat walk's address).
  const auto element_at = [](auto* ptr, int64_t stride_m, int64_t stride_n, int64_t e, int n) {
    if (stride_m == int64_t{n} * stride_n) return ptr + e * stride_n;
    const int64_t row = e < (int64_t{1} << 31)
                            ? int64_t{static_cast<uint32_t>(e) / static_cast<uint32_t>(n)}
                            : e / n;
    return ptr + row * stride_m + (e - row * n) * stride_n;
  };

  if constexpr (WALK == Walk::flat) {
    // THE SHARES of the band's elements, in whole packs (int4: whole groups), the last rank's short.
    constexpr int kUnit = kInt4 ? kInt4Group : P;
    const int64_t total = int64_t{input_size_m} * input_size_n;
    const int slice = static_cast<int>((total / kUnit + WORLD - 1) / WORLD * kUnit);
    const auto share_len = [&](int r) {
      return static_cast<int>(max(int64_t{0}, min(int64_t{slice}, total - int64_t{r} * slice)));
    };
    if constexpr (kInt4) {
      static_assert(DIRECTION == Direction::push && ACCUMULATE == Accumulate::registers,
                    "the int4 flat walk pushes and sums in registers");
      const int groups_all = slice / kInt4Group;
      const int grid = static_cast<int>(gridDim.x);
      const int per_block = (groups_all + grid - 1) / grid;
      const int g_first = static_cast<int>(blockIdx.x) * per_block;
      const int g_end = min(g_first + per_block, groups_all);
      // 1. THE PUSH: every rank's share of this rank's input, a group a thread, quantized here and
      // stored in that rank's staging payload at this rank's slot. TWO RANKS' GROUPS IN FLIGHT:
      // unrolled whole, every rank's 4 loads were hoisted together, 128 VGPRs of input, spilling.
      const T* const band_input = input_ptr + offs_m * input_stride_m;
      const int64_t mine = int64_t{rank} * slice;  // this rank's slot in every inbox
      for (int g0 = g_first; g0 < g_end; g0 += THREADS_PER_BLOCK) {
        const int g = g0 + static_cast<int>(threadIdx.x);
#pragma unroll 2
        for (int w = 0; w < WORLD; ++w) {
          const int r = (rank + w) % WORLD;
          const int groups = share_len(r) / kInt4Group;
          if (groups == 0) continue;  // a late rank's empty share (every block alike)
          const int64_t first = int64_t{r} * slice;
          const int at = min(g, groups - 1);
          const V* group;
          if constexpr (CONTIGUOUS) {
            // A UNIFORM BASE PLUS A 32-BIT BYTE OFFSET, the offset opaque so the address is built
            // at its load (saddr), not hoisted and held as a 64-bit vector address.
            uint32_t offset = static_cast<uint32_t>(at) * (kInt4Group * sizeof(T));
            asm volatile("" : "+v"(offset));
            group = reinterpret_cast<const V*>(
                reinterpret_cast<const char*>(band_input + first * input_stride_n) + offset);
          } else
            group = reinterpret_cast<const V*>(element_at(band_input, input_stride_m,
                                                          input_stride_n,
                                                          first + int64_t{at} * kInt4Group,
                                                          input_size_n));
          T x[kInt4GroupPacks][P];
#pragma unroll
          for (int k = 0; k < kInt4GroupPacks; ++k) {
            const u32x4 raw = *(const global_u32x4*)(group + k);
            __builtin_memcpy(x[k], &raw, sizeof(raw));
          }
          uint32_t words[kInt4GroupPacks];
          float scale;
          quantize(x, words, scale);
          if (g < g_end && g < groups) {
            uint8_t* const values = ptr_array_at<uint8_t>(staging_ptr_array, staging_ptr_array_size, r) + mine / 2;
            __fp16* const scales =
                reinterpret_cast<__fp16*>(ptr_array_at(staging_ptr_array, staging_ptr_array_size, r) +
                                          int4_scales_offset(staging_size_m, staging_size_n)) +
                mine / kInt4Group;
            uint32_t value_at = static_cast<uint32_t>(g) * 16;
            uint32_t scale_at = static_cast<uint32_t>(g) * kInt4ScaleBytes;
            asm volatile("" : "+v"(value_at));
            asm volatile("" : "+v"(scale_at));
            *(global_u32x4*)(values + value_at) = u32x4{words[0], words[1], words[2], words[3]};
            *(global_f16*)(reinterpret_cast<char*>(scales) + scale_at) = static_cast<__fp16>(scale);
          }
        }
      }
      barrier<UNTIL, 0, WORLD>(signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
                              signals_stride_block, signals_stride_writer, signals_size_parity,
                              signals_size_block, signals_size_writer, status_ptr,
                              status_stride_field, status_size_field, rank, timeout_ticks);
      // 2. THE SUM, LOCAL: this rank's share, every writer's pushed group read from this rank's own
      // inbox, all in flight (rotated writer order), summed in fp32 from the words, quantized and
      // stored in this rank's partials.
      const int groups = share_len(rank) / kInt4Group;
      if (groups != 0) {  // a late rank's share is empty at few groups (every block alike)
        const uint8_t* values[WORLD];
        const __fp16* scales[WORLD];
#pragma unroll
        for (int w = 0; w < WORLD; ++w) {
          const int64_t slot = int64_t{(rank + w) % WORLD} * slice;
          values[w] = ptr_array_at<uint8_t>(staging_ptr_array, staging_ptr_array_size, rank) + slot / 2;
          scales[w] = reinterpret_cast<__fp16*>(ptr_array_at(staging_ptr_array, staging_ptr_array_size, rank) +
                                                int4_scales_offset(staging_size_m, staging_size_n)) +
                      slot / kInt4Group;
        }
        uint8_t* const own_values = ptr_array_at<uint8_t>(partials_ptr_array, partials_ptr_array_size, rank);
        __fp16* const own_scales = reinterpret_cast<__fp16*>(
            ptr_array_at(partials_ptr_array, partials_ptr_array_size, rank) +
            int4_scales_offset(partials_size_m, partials_size_n));
        for (int g0 = g_first; g0 < g_end; g0 += THREADS_PER_BLOCK) {
          const int g = g0 + static_cast<int>(threadIdx.x);
          const int at = min(g, groups - 1);
          uint32_t value_at = static_cast<uint32_t>(at) * 16;
          uint32_t scale_at = static_cast<uint32_t>(at) * kInt4ScaleBytes;
          asm volatile("" : "+v"(value_at));
          asm volatile("" : "+v"(scale_at));
          u32x4 words[WORLD];
          float group_scales[WORLD];
#pragma unroll
          for (int w = 0; w < WORLD; ++w) {
            words[w] = *(const global_u32x4*)(values[w] + value_at);
            group_scales[w] = *(const global_f16*)(reinterpret_cast<const char*>(scales[w]) + scale_at);
          }
          // ISSUED HERE: no memory operation (the memory clobber) nor instruction (sched_barrier)
          // crosses this point, so every writer's loads are in flight before the first is summed.
          asm volatile("" ::: "memory");
          __builtin_amdgcn_sched_barrier(0);
          // EVERY WRITER'S GROUP SUMMED FROM THE WORDS, in fp32 in writer order: each element (q - 8) *
          // scale landed in T, as a quantized load lands it, then added. No pack of every writer decoded
          // first: decoded whole, the push kernels held 172 VGPRs (102 unquantized; ISA
          // 2026-10-15T22-47-01Z). WITH BF16, TWO WRITERS' TERMS A STEP: landed together (one
          // v_cvt_pk_bf16_f32) and added by one v_dot2c_f32_bf16 against (1, 1), where each was a
          // conversion, a shift and an add. v_dot2_f32_bf16 IS GFX950'S (dot12-insts), the
          // one arch built (CMakeLists.txt).
          float sum[kInt4GroupPacks][P];
#pragma unroll
          for (int k = 0; k < kInt4GroupPacks; ++k) {
#pragma unroll
            for (int j = 0; j < P; ++j) sum[k][j] = 0.0f;
            int w = 0;
            if constexpr (kBf16<T>) {
              const bf16x2 ones = __builtin_bit_cast(bf16x2, 0x3f803f80u);
#pragma unroll
              for (; w + 1 < WORLD; w += 2) {
                float a[P], b[P];
                dequantize(words[w][k], group_scales[w], a);
                dequantize(words[w + 1][k], group_scales[w + 1], b);
#pragma unroll
                for (int j = 0; j < P; ++j)
                  sum[k][j] = __builtin_amdgcn_fdot2_f32_bf16(
                      __builtin_convertvector(f32x2{a[j], b[j]}, bf16x2), ones, sum[k][j], false);
                // THE SUMS DONE BEFORE THE NEXT DEQUANTIZE'S VOLATILE ASM: the scheduler, free to use 256
                // VGPRs at 512 threads, decoded every writer's words of a group ahead of the first sum and
                // spilled (the plain kernel 256 VGPRs, 264 B of partials: ISA 2026-10-16T02-22-15Z).
#pragma unroll
                for (int j = 0; j < P; ++j) asm volatile("" ::"v"(sum[k][j]));
              }
            }
#pragma unroll
            for (; w < WORLD; ++w) {
              float a[P];
              dequantize(words[w][k], group_scales[w], a);
#pragma unroll
              for (int j = 0; j < P; ++j) sum[k][j] += static_cast<float>(static_cast<T>(a[j]));
#pragma unroll
              for (int j = 0; j < P; ++j) asm volatile("" ::"v"(sum[k][j]));  // as above
            }
          }
          uint32_t out[kInt4GroupPacks];
          float scale;
          quantize(sum, out, scale);
          if (g < g_end && g < groups) {
            uint32_t out_value_at = static_cast<uint32_t>(g) * 16;
            uint32_t out_scale_at = static_cast<uint32_t>(g) * kInt4ScaleBytes;
            asm volatile("" : "+v"(out_value_at));
            asm volatile("" : "+v"(out_scale_at));
            *(global_u32x4*)(own_values + out_value_at) = u32x4{out[0], out[1], out[2], out[3]};
            *(global_f16*)(reinterpret_cast<char*>(own_scales) + out_scale_at) =
                static_cast<__fp16>(scale);
          }
        }
      }
      barrier<Until::visible, 1, WORLD>(signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
                              signals_stride_block, signals_stride_writer, signals_size_parity,
                              signals_size_block, signals_size_writer, status_ptr,
                              status_stride_field, status_size_field, rank, timeout_ticks);
    } else {
      static_assert(ACCUMULATE == Accumulate::lds, "the exact flat walk sums through LDS");
      constexpr int LANES = THREADS_PER_BLOCK / WORLD;
      constexpr int K = TILE_N / (P * LANES);
      static_assert(LANES * WORLD == THREADS_PER_BLOCK, "a block is a group of threads a rank");
      static_assert(LANES % kWaveSize == 0, "a rank's copy is whole waves");
      static_assert(K * P * LANES == TILE_N, "a chunk is whole packs a thread");
      const int chunks = (slice + TILE_N - 1) / TILE_N;
      if constexpr (DIRECTION == Direction::push) {
        // 1. THE PUSH: this rank's input, every rank's share of it, read here (no peer reads the
        // caller's input) and stored in THAT rank's inbox at this rank's slot, each block the
        // chunks that rank's same block sums (the same walk).
        const T* const band_input = input_ptr + offs_m * input_stride_m;
        for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
          const int r = (rank + static_cast<int>(threadIdx.x) / LANES) % WORLD;
          const int len = share_len(r);
          if (len == 0) continue;
          const int64_t first = int64_t{r} * slice;
          T part[K][P];
          if constexpr (CONTIGUOUS) {
            // The last pack, the run's start, then the column: the order the two-shot's
            // instructions came out in (ISA 2026-10-15).
#pragma unroll
            for (int k = 0; k < K; ++k) {
              const int last = len / P - 1;
              const V* run = reinterpret_cast<const V*>(band_input + first * input_stride_n);
              const int n = chunk * TILE_N / P + static_cast<int>(threadIdx.x) % LANES + k * LANES;
              const int col = n < len / P ? n : last;
              uint32_t offset = static_cast<uint32_t>(col) * uint32_t{sizeof(V)};
              asm volatile("" : "+v"(offset));
              const u32x4 raw =
                  *(const global_u32x4*)(reinterpret_cast<const char*>(run) + offset);
              __builtin_memcpy(part[k], &raw, sizeof(raw));
            }
          } else {
#pragma unroll
            for (int k = 0; k < K; ++k) {
              // The pack's column, past the share the last in it (every load in bounds, none
              // under a branch).
              const int n = chunk * TILE_N / P + static_cast<int>(threadIdx.x) % LANES + k * LANES;
              const int col = n < len / P ? n : len / P - 1;
              const u32x4 raw = *(const global_u32x4*)(element_at(
                  band_input, input_stride_m, input_stride_n, first + int64_t{col} * P,
                  input_size_n));
              __builtin_memcpy(part[k], &raw, sizeof(raw));
            }
          }
          // ISSUED HERE: no memory operation (the memory clobber) nor instruction (sched_barrier)
          // crosses this point, so every load above is in flight before anything below runs (the
          // scheduler alone sank peer loads past the first adds: ISA 2026-10-01T00-39-51Z).
          asm volatile("" ::: "memory");
          __builtin_amdgcn_sched_barrier(0);
          V* const inbox = reinterpret_cast<V*>(ptr_array_at<T>(staging_ptr_array, staging_ptr_array_size, r) +
                                                rank * staging_stride_m);
#pragma unroll
          for (int k = 0; k < K; ++k) {
            const int c = chunk * TILE_N / P + static_cast<int>(threadIdx.x) % LANES + k * LANES;
            if (c >= len / P) continue;
            uint32_t offset = static_cast<uint32_t>(c) * uint32_t{sizeof(V)};
            asm volatile("" : "+v"(offset));
            u32x4 raw;
            __builtin_memcpy(&raw, part[k], sizeof(raw));
            *(global_u32x4*)(reinterpret_cast<char*>(inbox) + offset) = raw;
          }
        }
      }
      barrier<UNTIL, 0, WORLD>(signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
                              signals_stride_block, signals_stride_writer, signals_size_parity,
                              signals_size_block, signals_size_writer, status_ptr,
                              status_stride_field, status_size_field, rank, timeout_ticks);
      // 2. THE SUM: this rank's share, a TILE_N chunk a step, every rank's copy (pull: its input
      // through the table, from this rank's first element; push: writer w's slot of this rank's
      // inbox) in one round trip, summed over the ranks in rotated order through LDS, rounded once
      // to T, stored in this rank's partials.
      const int len = share_len(rank);
      if (len != 0) {  // a late rank's share is empty at few packs (every block alike)
        [[maybe_unused]] const T* const inbox = ptr_array_at<T>(staging_ptr_array, staging_ptr_array_size, rank);
        constexpr bool kRun = DIRECTION == Direction::push || CONTIGUOUS;
        const int64_t stride_m = DIRECTION == Direction::push ? staging_stride_m : input_ptr_array_stride_m;
        const int64_t stride_n = DIRECTION == Direction::push ? staging_stride_n : input_ptr_array_stride_n;
        const int64_t first = DIRECTION == Direction::push ? 0 : int64_t{rank} * slice;
        for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
          const int w = (rank + static_cast<int>(threadIdx.x) / LANES) % WORLD;
          const T* from;
          if constexpr (DIRECTION == Direction::push)
            from = inbox + w * staging_stride_m;
          else {
            from = ptr_array_at(input_ptr_array, input_ptr_array_size, w) + offs_m * input_ptr_array_stride_m;
          }
          T got[1][K][P];
          if constexpr (kRun) {
#pragma unroll
            for (int k = 0; k < K; ++k) {
              const int last = len / P - 1;
              const V* run = reinterpret_cast<const V*>(from + first * stride_n);
              const int n = chunk * TILE_N / P + static_cast<int>(threadIdx.x) % LANES + k * LANES;
              const int col = n < len / P ? n : last;
              uint32_t offset = static_cast<uint32_t>(col) * uint32_t{sizeof(V)};
              asm volatile("" : "+v"(offset));
              const u32x4 raw =
                  *(const global_u32x4*)(reinterpret_cast<const char*>(run) + offset);
              __builtin_memcpy(got[0][k], &raw, sizeof(raw));
            }
          } else {
#pragma unroll
            for (int k = 0; k < K; ++k) {
              const int n = chunk * TILE_N / P + static_cast<int>(threadIdx.x) % LANES + k * LANES;
              const int col = n < len / P ? n : len / P - 1;
              const u32x4 raw = *(const global_u32x4*)(element_at(
                  from, stride_m, stride_n, first + int64_t{col} * P, input_size_n));
              __builtin_memcpy(got[0][k], &raw, sizeof(raw));
            }
          }
          asm volatile("" ::: "memory");  // issued (above)
          __builtin_amdgcn_sched_barrier(0);
          T rounded[1][K][P];
          block_reduce<Sum, ReduceAxis::m, WORLD, LANES>(got[0], rounded[0]);
          store<1, LANES, THREADS_PER_BLOCK>(rounded, partials_ptr_array, partials_ptr_array_size, rank, 0, 0,
                                   0, 1, 1, 1, len, 1, len, 0, chunk * TILE_N, 1);
        }
      }
      barrier<Until::visible, 1, WORLD>(signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
                              signals_stride_block, signals_stride_writer, signals_size_parity,
                              signals_size_block, signals_size_writer, status_ptr,
                              status_stride_field, status_size_field, rank, timeout_ticks);
    }
  } else {
    // THIS RANK'S COLUMNS, in whole packs (a late rank's short or empty), and this block's rows,
    // worked out before the first barrier, in scalar registers: after it they were vector
    // arithmetic (a division by the grid a lane; ISA 2026-10-15T18-38-16Z).
    const int packs = input_size_n / P;  // the row, in packs (n is whole packs)
    const int slice = (packs + WORLD - 1) / WORLD;
    const int col0 = min(rank * slice, packs);
    const int own_packs = max(0, min(slice, packs - col0));
    const int end = (col0 + own_packs) * P;  // this rank's columns end, in elements
    [[maybe_unused]] const int block_size_m =
        input_size_m > static_cast<int>(blockIdx.x)
            ? (input_size_m - blockIdx.x + gridDim.x - 1) / gridDim.x
            : 0;
    // 1. int4: this block's rows of this rank's input, whole, quantized into its staging's INT4
    // payload at the same place, kTiles tiles a step, every load in flight before the first store.
    if constexpr (kInt4) {
      constexpr int kTiles = TILE_M < 4 ? 4 / TILE_M : 1;
      const int tiles = (input_size_m + TILE_M - 1) / TILE_M;
      for (int t = blockIdx.x; t < tiles; t += kTiles * gridDim.x)
        for (int c = 0; c < input_size_n; c += THREADS_PER_BLOCK * P) {
          T x[kTiles][TILE_M][1][P];
#pragma unroll
          for (int s = 0; s < kTiles; ++s)
            load<1, THREADS_PER_BLOCK>(&input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1,
                             input_size_m, input_size_n, input_size_m, input_size_n, x[s],
                             (t + s * static_cast<int>(gridDim.x)) * TILE_M, c, 1);
#pragma unroll
          for (int s = 0; s < kTiles; ++s)
            store<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK, Ranks::own, QUANTIZATION>(
                x[s], staging_ptr_array, staging_ptr_array_size, rank, 0, 0, staging_stride_m,
                staging_stride_n, 1, staging_size_m, staging_size_n, input_size_m, input_size_n,
                (t + s * static_cast<int>(gridDim.x)) * TILE_M, c, 1);
        }
    }
    barrier<UNTIL, 0, WORLD>(signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
                              signals_stride_block, signals_stride_writer, signals_size_parity,
                              signals_size_block, signals_size_writer, status_ptr,
                              status_stride_field, status_size_field, rank, timeout_ticks);
    if constexpr (ACCUMULATE == Accumulate::lds) {
      static_assert(DIRECTION == Direction::pull, "the rows walk through LDS pulls");
      constexpr int LANES = THREADS_PER_BLOCK / WORLD;
      static_assert(LANES * WORLD == THREADS_PER_BLOCK, "a block is a row of threads a rank");
      using Summed = std::conditional_t<kInt4, float, T>;
      const int tiles = (input_size_m + TILE_M - 1) / TILE_M;
      for (int t = blockIdx.x; t < tiles; t += gridDim.x)
        for (int row = t * TILE_M; row < min(t * TILE_M + TILE_M, input_size_m); ++row)
          for (int c = col0 * P; c < end; c += LANES * P) {
            // 2. Every rank's copy of the chunk, a row of threads each, in one round trip.
            const int r = (rank + static_cast<int>(threadIdx.x) / LANES) % WORLD;
            T got[1][1][P];
            if constexpr (!kInt4)
              load<1, LANES, Ranks::lane, Quantization::none, WORLD>(
                  input_ptr_array, input_ptr_array_size, r, 0, 0, input_ptr_array_stride_m,
                  input_ptr_array_stride_n, 1, input_size_m, input_size_n, input_size_m, end,
                  got, row, c, 1);
            else
              load<1, LANES, Ranks::lane, QUANTIZATION, WORLD>(
                  staging_ptr_array, staging_ptr_array_size, r, 0, 0, staging_stride_m,
                  staging_stride_n, 1, staging_size_m, staging_size_n, input_size_m, end, got,
                  row, c, 1);
            // 3. Summed over the rows (the ranks), in row order, through LDS, rounded once.
            Summed rounded[1][1][P];
            block_reduce<Sum, ReduceAxis::m, WORLD, LANES>(got[0], rounded[0]);
            // 4. Stored in this rank's slice.
            store<1, LANES, THREADS_PER_BLOCK, Ranks::own, QUANTIZATION>(
                rounded, partials_ptr_array, partials_ptr_array_size, rank, 0, 0,
                partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n,
                row + 1, end, row, c, 1);
          }
    } else {
      constexpr int kSliceWaves = (TILE_N / WORLD / P + kWaveSize - 1) / kWaveSize * kWaveSize;
      constexpr int LANES = kSliceWaves < THREADS_PER_BLOCK ? kSliceWaves : THREADS_PER_BLOCK;
      constexpr int ROWS = THREADS_PER_BLOCK / LANES;  // rows of threads at once
      constexpr Ranks kStoreRanks = DIRECTION == Direction::push ? Ranks::every : Ranks::own;
      using Summed = std::conditional_t<kInt4, float, T>;
      const int offs_n = col0 * P;
      const int size_n = own_packs * P;
      // INT4'S SUM IS FROM THE WORDS (the flat walk's: a decoded tile of every peer held 172
      // VGPRs), so its loads are the words and scales, every rank's at its staging's place.
      const uint8_t* input_values[WORLD];
      const __fp16* input_scales[WORLD];
      [[maybe_unused]] const int64_t values_stride_m = staging_size_n / 2;
      [[maybe_unused]] const int64_t scales_stride_m = staging_size_n / kInt4Group;
      if constexpr (kInt4) {
#pragma unroll
        for (int r = 0; r < WORLD; ++r) {
          input_values[r] = ptr_array_at<const uint8_t>(staging_ptr_array, staging_ptr_array_size, r);
        }
#pragma unroll
        for (int r = 0; r < WORLD; ++r)
          input_scales[r] = reinterpret_cast<const __fp16*>(
              input_values[r] + int4_scales_offset(staging_size_m, staging_size_n));
      }
      if (size_n > 0)  // a late rank owns no columns
        for (int q = 0; q < block_size_m; q += ROWS)
          for (int c = offs_n; c < offs_n + size_n; c += LANES * P) {
            const int row = static_cast<int>(blockIdx.x + q * gridDim.x);
            Summed rounded[1][1][P];
            if constexpr (!kInt4) {
              T peers[WORLD][1][1][P];
              load<ROWS, LANES, Ranks::every>(input_ptr_array, input_ptr_array_size, 0, 0, 0,
                                              input_ptr_array_stride_m, input_ptr_array_stride_n,
                                              1, input_size_m, input_size_n, input_size_m,
                                              offs_n + size_n, peers, row, c,
                                              static_cast<int>(gridDim.x));
              // Summed in rank order in fp32, so every rank's sum agrees bitwise, rounded once.
#pragma unroll
              for (int j = 0; j < P; ++j) {
                float acc = static_cast<float>(peers[0][0][0][j]);
#pragma unroll
                for (int r = 1; r < WORLD; ++r) acc += static_cast<float>(peers[r][0][0][j]);
                rounded[0][0][j] = static_cast<Summed>(acc);
              }
            } else {
              // Every rank's word and scale of this thread's pack, the position once in a vector register
              // (32-BIT BYTE OFFSETS: a quantized buffer is ours, at most the box's, select's exchange_sizes),
              // each rank's an add of it, all in flight. Only rows of threads skip a row past input_size_m
              // (load's rule: one row's is real wherever the walk makes one).
              static_assert(LANES % kInt4GroupPacks == 0, "a group is 4 lanes' packs");
              uint32_t peer_words[WORLD];
              float peer_scales[WORLD];
              const int thread_row = ROWS == 1 ? 0 : static_cast<int>(threadIdx.x) / LANES;
              if (!(ROWS > 1 && row + thread_row * static_cast<int>(gridDim.x) >= input_size_m)) {
                const int64_t at_m = row + thread_row * static_cast<int>(gridDim.x) < input_size_m
                                         ? row + thread_row * static_cast<int>(gridDim.x)
                                         : input_size_m - 1;
                const int n = c / P + static_cast<int>(threadIdx.x) % LANES;
                const int pack = n < (offs_n + size_n) / P ? n : (offs_n + size_n) / P - 1;
                uint32_t value_at = static_cast<uint32_t>(at_m * values_stride_m) +
                                    static_cast<uint32_t>(pack) * kInt4WordBytes;
                uint32_t scale_at = static_cast<uint32_t>(
                    (at_m * scales_stride_m + pack / kInt4GroupPacks) * kInt4ScaleBytes);
                asm volatile("" : "+v"(value_at));
                asm volatile("" : "+v"(scale_at));
#pragma unroll
                for (int r = 0; r < WORLD; ++r) {
                  peer_words[r] = *(const global_u32*)(input_values[r] + value_at);
                  peer_scales[r] = *(const global_f16*)(
                      reinterpret_cast<const char*>(input_scales[r]) + scale_at);
                }
              }
              asm volatile("" ::: "memory");  // issued: every rank's in flight before the first is summed
              __builtin_amdgcn_sched_barrier(0);
              // Every rank's pack summed from the words, in fp32 in rank order (the flat walk's sum above).
              float sum[P];
#pragma unroll
              for (int j = 0; j < P; ++j) sum[j] = 0.0f;
              int w = 0;
              if constexpr (kBf16<T>) {
                const bf16x2 ones = __builtin_bit_cast(bf16x2, 0x3f803f80u);
#pragma unroll
                for (; w + 1 < WORLD; w += 2) {
                  float a[P], b[P];
                  dequantize(peer_words[w], peer_scales[w], a);
                  dequantize(peer_words[w + 1], peer_scales[w + 1], b);
#pragma unroll
                  for (int j = 0; j < P; ++j)
                    sum[j] = __builtin_amdgcn_fdot2_f32_bf16(
                        __builtin_convertvector(f32x2{a[j], b[j]}, bf16x2), ones, sum[j], false);
#pragma unroll
                  for (int j = 0; j < P; ++j) asm volatile("" ::"v"(sum[j]));
                }
              }
#pragma unroll
              for (; w < WORLD; ++w) {
                float a[P];
                dequantize(peer_words[w], peer_scales[w], a);
#pragma unroll
                for (int j = 0; j < P; ++j) sum[j] += static_cast<float>(static_cast<T>(a[j]));
#pragma unroll
                for (int j = 0; j < P; ++j) asm volatile("" ::"v"(sum[j]));
              }
#pragma unroll
              for (int j = 0; j < P; ++j) rounded[0][0][j] = sum[j];

            }
            // Push: every rank's partials, the stores issued together (int4: quantized once);
            // pull: this rank's.
            store<ROWS, LANES, THREADS_PER_BLOCK, kStoreRanks, QUANTIZATION, WORLD>(
                rounded, partials_ptr_array, partials_ptr_array_size, rank, 0, 0,
                partials_stride_m, partials_stride_n, 1, partials_size_m, partials_size_n,
                input_size_m, offs_n + size_n, row, c, static_cast<int>(gridDim.x));
          }
    }
    barrier<Until::visible, 1, WORLD>(signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
                              signals_stride_block, signals_stride_writer, signals_size_parity,
                              signals_size_block, signals_size_writer, status_ptr,
                              status_stride_field, status_size_field, rank, timeout_ticks);
  }
}

}  // namespace hip_custom_communicator
