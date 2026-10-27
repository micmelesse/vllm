// SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc.
// All rights reserved.
//
// THE PLAIN ALL-REDUCE: one kernel, `all_reduce`, its ALGORITHM a template
// parameter. Each algorithm is its own function, given only the arguments it
// uses; the kernel dispatches. pull one-shot  every rank reads every rank's
// input through the input table and sums. (aiter's cross_device_reduce_1stage)
// pull two-shot  each rank sums its share of every rank's input into its
// partials; then every rank reads every rank's partials. (aiter's
// cross_device_reduce_2stage) push one-shot  each rank writes its input into
// every peer's inbox as lines that carry this call's epoch; each rank waits for
// the epoch in its own inbox and sums. No barrier. (NCCL's LL, aiter's
// ar_ll_gfx9) staged push    each rank stores its input into every rank's inbox
// in the staging; a one-shot       barrier; each rank sums its inbox; a
// barrier. push two-shot  each rank writes every rank's share into that rank's
// inbox; each rank sums its inbox into its partials; then every rank reads
// every partials.
// A QUANTIZE_INT4 instance moves INT4: each rank encodes its own input, so
// nothing is read in place. The exact instance of an INT4 build is the exact
// body.
//
// THE WALKS. The one-shots walk [m, n] in TILE_M x TILE_N tiles, the grid
// striding over them. The two-shots walk each rank's share as one flat range of
// the m * n elements, in TILE_N chunks (TILE_M unused). A tile or chunk is
// whole 16-byte packs; steps past its edge are masked.
//
// THE INPUTS ARE NOT WRITTEN WHILE A KERNEL RUNS: its last barrier follows
// every read of them, so the next call's copy, stream-ordered after it, may
// overwrite them.
//
// THE STAGING INBOX IS REUSED WITHOUT A BARRIER OF ITS OWN: a rank writes into
// peer s's slot only after the barrier that follows peer s's sum of that slot.
//
// THE PUSHED ONE-SHOT'S EPOCH: a counter per row on the device, +1 each call
// that writes the row (never 0), so a graph replay counts on with no host code
// and the inbox is never cleared. A row's bank is its epoch & 1. Two banks need
// no closing barrier: a rank writes epoch e into a bank only after it saw the
// peer's e - 1, which the peer sends only after it finished with e - 2, the
// bank's last use.

#pragma once

#include "all_gather.cuh"
#include "common.cuh"
#include "reduce_scatter.cuh"

namespace hip_custom_communicator {

// THE ALL-REDUCE'S ALGORITHMS: its ALGORITHM, exactly the bodies it has.
enum class AllReduceAlgorithm {
  one_shot_pull,
  one_shot_push,
  two_shot_pull,
  two_shot_push,
  one_shot_push_staged,
};

// THE PULLED ONE-SHOT: every rank reads every rank's input through the input
// table and sums (aiter's cross_device_reduce_1stage).
template <bool QUANTIZE_INT4, typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES, bool CONTIGUOUS,
          bool TAIL>
DINLINE void all_reduce_one_shot_pull(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* const* input_ptr_array,
    int input_ptr_array_size, int64_t input_ptr_array_stride_m,
    int64_t input_ptr_array_stride_n, DTYPE* output_ptr,
    int64_t output_stride_m, int64_t output_stride_n,
    char* const* staging_ptr_array, int staging_ptr_array_size,
    uint32_t* const* signals_ptr_array, int signals_ptr_array_size,
    int64_t signals_stride_parity, int64_t signals_stride_block,
    int64_t signals_stride_writer, int signals_size_parity,
    int signals_size_block, int signals_size_writer, uint32_t* status_ptr,
    int64_t status_stride_field, int status_size_field, int rank,
    uint64_t timeout_ticks) {
  constexpr int kThreadsM = THREADS_PER_BLOCK % TILE_M == 0 ? TILE_M : 1;
  constexpr int kThreadsN = THREADS_PER_BLOCK / kThreadsM;
  constexpr int kRows = TILE_M / kThreadsM;
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
  // The tile in whole packs, INT4 in whole groups (a group is 4 lanes' packs,
  // so its lanes stay together); columns past the input are masked.
  constexpr int kUnit = QUANTIZE_INT4 ? kInt4Group : kPackElems;
  constexpr int kTileN = (TILE_N + kUnit - 1) / kUnit * kUnit;
  constexpr int kPacks =
      (kTileN + kPackElems * kThreadsN - 1) / (kPackElems * kThreadsN);
  constexpr bool kWhole = kPacks * kPackElems * kThreadsN == kTileN;
  const int tiles_n = (input_size_n + kTileN - 1) / kTileN;
  const int tiles = (input_size_m + TILE_M - 1) / TILE_M * tiles_n;
  // 1. INT4: this rank's input, quantized into its staging. A block owns the
  //    same tiles in both passes, so the barrier makes them visible to its twin
  //    block on each peer.
  if constexpr (QUANTIZE_INT4) {
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      const int offs_m = t / tiles_n * TILE_M, offs_n = t % tiles_n * kTileN;
      const int clip_n =
          kWhole ? input_size_n : min(input_size_n, offs_n + kTileN);
      DTYPE mine[kRows][kPacks][kPackElems];
      load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
          &input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1,
          input_size_m, input_size_n, input_size_m, clip_n, mine, offs_m,
          offs_n, 1);
      store<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, true, Layout::tile,
            TAIL>(mine, staging_ptr_array, staging_ptr_array_size, rank, 0, 0,
                  input_size_n, 1, 1, input_size_m, input_size_n, input_size_m,
                  clip_n, offs_m, offs_n, 1);
    }
  }
  // Every rank's entry, read before the barrier so its round trip overlaps it:
  // its staging's payload (INT4), or its input through the input table.
  std::conditional_t<QUANTIZE_INT4, char*, const DTYPE*> entries[WORLD];
#pragma unroll
  for (int r = 0; r < WORLD; ++r) {
    if constexpr (QUANTIZE_INT4)
      entries[r] = load(staging_ptr_array, staging_ptr_array_size, r);
    else
      entries[r] = load(input_ptr_array, input_ptr_array_size, r);
  }
  // 2. Every rank's data ready and visible: INT4's written in this launch
  //    (ordered), the exact inputs before it (the launch boundaries order
  //    them).
  constexpr Until kFirst = QUANTIZE_INT4 ? Until::launched : Until::started;
  barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                 signals_stride_parity, signals_stride_block,
                 signals_stride_writer, signals_size_parity, signals_size_block,
                 signals_size_writer, status_ptr, status_stride_field,
                 status_size_field, rank, timeout_ticks, kFirst, 0);
  // 3. Every rank's tile (INT4: decoded; its payload a row of input_size_n),
  //    summed in rank order in fp32, rounded once: every rank's sums agree
  //    bitwise.
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int offs_m = t / tiles_n * TILE_M, offs_n = t % tiles_n * kTileN;
    const int clip_n =
        kWhole ? input_size_n : min(input_size_n, offs_n + kTileN);
    DTYPE copies[WORLD][kRows][kPacks][kPackElems];
    all_gather<AllGatherAlgorithm::tile_pull, QUANTIZE_INT4, DTYPE, WORLD,
               TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, TAIL>(
        entries,
        QUANTIZE_INT4 ? int64_t{input_size_n} : input_ptr_array_stride_m,
        QUANTIZE_INT4 ? int64_t{1} : input_ptr_array_stride_n, input_size_m,
        input_size_n, clip_n, offs_m, offs_n, copies);
    DTYPE rounded[kRows][kPacks][kPackElems];
#pragma unroll
    for (int i = 0; i < kRows; ++i)
#pragma unroll
      for (int k = 0; k < kPacks; ++k)
#pragma unroll
        for (int j = 0; j < kPackElems; ++j) {
          float acc = static_cast<float>(copies[0][i][k][j]);
#pragma unroll
          for (int r = 1; r < WORLD; ++r)
            acc += static_cast<float>(copies[r][i][k][j]);
          rounded[i][k][j] = static_cast<DTYPE>(acc);
        }
    store<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
        rounded, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1,
        input_size_m, input_size_n, input_size_m, clip_n, offs_m, offs_n, 1);
  }
  // 4. Every peer done reading this rank's data: the next launch may overwrite
  //    it.
  barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                 signals_stride_parity, signals_stride_block,
                 signals_stride_writer, signals_size_parity, signals_size_block,
                 signals_size_writer, status_ptr, status_stride_field,
                 status_size_field, rank, timeout_ticks, Until::read, 1);
}

// THE PULLED TWO-SHOT: each rank sums its share of every rank's input into its
// partials; then every rank reads every rank's partials (aiter's
// cross_device_reduce_2stage).
template <bool QUANTIZE_INT4, typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES, bool CONTIGUOUS,
          bool TAIL>
DINLINE void all_reduce_two_shot_pull(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* const* input_ptr_array,
    int input_ptr_array_size, int64_t input_ptr_array_stride_m,
    int64_t input_ptr_array_stride_n, DTYPE* output_ptr,
    int64_t output_stride_m, int64_t output_stride_n,
    char* const* partials_ptr_array, int partials_ptr_array_size,
    int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
    int partials_size_n, char* const* staging_ptr_array,
    int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    uint32_t* const* signals_ptr_array, int signals_ptr_array_size,
    int64_t signals_stride_parity, int64_t signals_stride_block,
    int64_t signals_stride_writer, int signals_size_parity,
    int signals_size_block, int signals_size_writer, uint32_t* status_ptr,
    int64_t status_stride_field, int status_size_field, int rank,
    uint64_t timeout_ticks) {
  // PULL TWO-SHOT: reduce-scatter (barrier; this rank's share of every rank's
  // input, exact read in place through the input table, INT4 quantized by its
  // owner into its staging, summed into this rank's partials; barrier), then
  // all-gather.
  reduce_scatter<ReduceScatterAlgorithm::flat_pull, QUANTIZE_INT4, WORLD,
                 TILE_M, TILE_N, THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES,
                 CONTIGUOUS, TAIL>(
      input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
      input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m,
      input_ptr_array_stride_n, staging_ptr_array, staging_ptr_array_size,
      staging_stride_m, staging_stride_n, staging_size_m, staging_size_n,
      partials_ptr_array, partials_ptr_array_size, partials_stride_m,
      partials_stride_n, partials_size_m, partials_size_n, signals_ptr_array,
      signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);
  all_gather<AllGatherAlgorithm::flat, QUANTIZE_INT4, DTYPE, WORLD, TILE_N,
             THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, CONTIGUOUS, TAIL>(
      partials_ptr_array, partials_ptr_array_size, partials_stride_m,
      partials_stride_n, partials_size_m, partials_size_n, output_ptr,
      output_stride_m, output_stride_n, input_size_m, input_size_n, 0, 0, 0,
      rank);
}

// THE PUSHED ONE-SHOT: each rank sends its input into every rank's inbox as
// lines carrying this call's epoch; each rank waits for the epoch in its own
// inbox and sums. No barrier (NCCL's LL, aiter's ar_ll_gfx9). INT4: a line a
// pack, its word and its group's scale.
template <bool QUANTIZE_INT4, typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES, bool CONTIGUOUS,
          bool TAIL>
DINLINE void all_reduce_one_shot_push(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, DTYPE* output_ptr,
    int64_t output_stride_m, int64_t output_stride_n,
    char* const* unfenced_ptr_array, int unfenced_ptr_array_size,
    int64_t unfenced_stride_slot, int64_t unfenced_stride_m,
    int64_t unfenced_stride_n, int unfenced_size_slot, int unfenced_size_m,
    int unfenced_size_n, int32_t* unfenced_epochs_ptr,
    int64_t unfenced_epochs_stride_block, int64_t unfenced_epochs_stride_n,
    int unfenced_epochs_size_block, int unfenced_epochs_size_n,
    uint32_t* status_ptr, int64_t status_stride_field, int status_size_field,
    int rank, uint64_t timeout_ticks) {
  constexpr int kThreadsM = THREADS_PER_BLOCK % TILE_M == 0 ? TILE_M : 1;
  constexpr int kThreadsN = THREADS_PER_BLOCK / kThreadsM;
  constexpr int kRows = TILE_M / kThreadsM;
  // A THREAD'S PACK, a line's: exact, its 8-byte payload (aiter's packet), or a
  // narrower build's pack alone in its line; INT4, a word of its group (8
  // values), its scale beside it.
  constexpr int kPackElems =
      QUANTIZE_INT4
          ? traits<DTYPE, PACK_BYTES>::N
          : (PACK_BYTES < kLinePayloadBytes ? PACK_BYTES : kLinePayloadBytes) /
                static_cast<int>(sizeof(DTYPE));
  // The tile in whole packs, INT4 in whole groups (a group is 4 lanes' packs,
  // so its lanes stay together); columns past the input are masked.
  constexpr int kUnit = QUANTIZE_INT4 ? kInt4Group : kPackElems;
  constexpr int kTileN = (TILE_N + kUnit - 1) / kUnit * kUnit;
  constexpr int kPacks =
      (kTileN + kPackElems * kThreadsN - 1) / (kPackElems * kThreadsN);
  constexpr bool kWhole = kPacks * kPackElems * kThreadsN == kTileN;
  const int tiles_n = (input_size_n + kTileN - 1) / kTileN;
  const int tiles = (input_size_m + TILE_M - 1) / TILE_M * tiles_n;
  // The block's TILE_M rows of a bank, from blockIdx.x * TILE_M.
  const int first_row = static_cast<int>(blockIdx.x) * TILE_M;
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
      static_cast<int>(gridDim.x) * TILE_M > unfenced_size_m / 2)
    __builtin_trap();
  // 0. This thread's rows' last epochs, through a plain pointer: they are this
  //    rank's alone.
  int32_t last[kRows][1][kLineBytes / static_cast<int>(sizeof(
                                          int32_t))];  // a 16-byte line a row
  load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
      &unfenced_epochs_ptr, 1, 0, 0, 0, unfenced_epochs_stride_block,
      unfenced_epochs_stride_n, 1, unfenced_epochs_size_block,
      unfenced_epochs_size_n, unfenced_epochs_size_block,
      unfenced_epochs_size_n, last, first_row, 0, 1);
  uint32_t epochs[kRows] = {};
  bool ran = false;
  // 1. Every rank's tile through the lines, summed in rank order in fp32 (this
  //    rank's own copy too), rounded once: every rank's sums agree bitwise.
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int offs_m = t / tiles_n * TILE_M, offs_n = t % tiles_n * kTileN;
    // The tile inside the input: the lines past it are nobody's.
    const int clip_m = min(TILE_M, input_size_m - offs_m);
    const int clip_n =
        kWhole ? input_size_n : min(input_size_n, offs_n + kTileN);
    DTYPE mine[kRows][kPacks][kPackElems];
    load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
        &input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1, input_size_m,
        input_size_n, input_size_m, clip_n, mine, offs_m, offs_n, 1);
    // This call's epoch for each row (last + 1, never 0; read after the first
    // tile's load is issued, so the round trips overlap); its lines go in bank
    // epoch & 1.
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      const uint32_t before = ran ? epochs[i]
                              : kThreadsM == 1
                                  ? __builtin_amdgcn_readfirstlane(
                                        static_cast<uint32_t>(last[i][0][0]))
                                  : static_cast<uint32_t>(last[i][0][0]);
      const bool written =
          (kThreadsM == 1 ? i
                          : static_cast<int>(threadIdx.x) / kThreadsN +
                                i * kThreadsM) < clip_m;
      epochs[i] = !written ? before : before + 1u == 0u ? 2u : before + 1u;
    }
    ran = true;
    DTYPE copies[WORLD][kRows][kPacks][kPackElems];
    all_gather<AllGatherAlgorithm::tile_push, QUANTIZE_INT4, DTYPE, WORLD,
               TILE_M, THREADS_PER_BLOCK, WAVE_SIZE>(
        mine, epochs, unfenced_ptr_array, unfenced_ptr_array_size,
        unfenced_stride_slot, unfenced_stride_m, unfenced_stride_n,
        unfenced_size_slot, unfenced_size_m, unfenced_size_n,
        first_row + clip_m, min(kTileN, input_size_n - offs_n), first_row, rank,
        status_ptr, status_stride_field, status_size_field, timeout_ticks,
        copies);
    DTYPE rounded[kRows][kPacks][kPackElems];
#pragma unroll
    for (int i = 0; i < kRows; ++i)
#pragma unroll
      for (int k = 0; k < kPacks; ++k)
#pragma unroll
        for (int j = 0; j < kPackElems; ++j) {
          float acc = static_cast<float>(copies[0][i][k][j]);
#pragma unroll
          for (int r = 1; r < WORLD; ++r)
            acc += static_cast<float>(copies[r][i][k][j]);
          rounded[i][k][j] = static_cast<DTYPE>(acc);
        }
    store<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
        rounded, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1,
        input_size_m, input_size_n, input_size_m, clip_n, offs_m, offs_n, 1);
    // The block is done with this bank before the next pass sends into it.
    __syncthreads();
  }
  // 2. Each row's epoch stored for the next launch (its row of threads' first
  //    lane).
  int32_t next[kRows][1][kLineBytes / static_cast<int>(sizeof(int32_t))] = {};
#pragma unroll
  for (int i = 0; i < kRows; ++i)
    next[i][0][0] = ran ? static_cast<int32_t>(epochs[i]) : last[i][0][0];
  store<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
      next, &unfenced_epochs_ptr, 1, 0, 0, 0, unfenced_epochs_stride_block,
      unfenced_epochs_stride_n, 1, unfenced_epochs_size_block,
      unfenced_epochs_size_n, unfenced_epochs_size_block,
      unfenced_epochs_size_n, first_row, 0, 1);
}

// THE STAGED PUSHED ONE-SHOT: each rank stores its input into slot [rank] of
// every rank's inbox in the staging (INT4: encoded); a barrier; each rank sums
// its inbox; a barrier (the inboxes free for the next launch). Plain stores
// carry no epoch, so the barriers say what has landed.
template <bool QUANTIZE_INT4, typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES, bool CONTIGUOUS,
          bool TAIL>
DINLINE void all_reduce_one_shot_push_staged(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, DTYPE* output_ptr,
    int64_t output_stride_m, int64_t output_stride_n,
    char* const* staging_ptr_array, int staging_ptr_array_size,
    uint32_t* const* signals_ptr_array, int signals_ptr_array_size,
    int64_t signals_stride_parity, int64_t signals_stride_block,
    int64_t signals_stride_writer, int signals_size_parity,
    int signals_size_block, int signals_size_writer, uint32_t* status_ptr,
    int64_t status_stride_field, int status_size_field, int rank,
    uint64_t timeout_ticks) {
  constexpr int kThreadsM = THREADS_PER_BLOCK % TILE_M == 0 ? TILE_M : 1;
  constexpr int kThreadsN = THREADS_PER_BLOCK / kThreadsM;
  constexpr int kRows = TILE_M / kThreadsM;
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
  // The tile in whole packs, INT4 in whole groups (a group is 4 lanes' packs,
  // so its lanes stay together); columns past the input are masked.
  constexpr int kUnit = QUANTIZE_INT4 ? kInt4Group : kPackElems;
  constexpr int kTileN = (TILE_N + kUnit - 1) / kUnit * kUnit;
  constexpr int kPacks =
      (kTileN + kPackElems * kThreadsN - 1) / (kPackElems * kThreadsN);
  constexpr bool kWhole = kPacks * kPackElems * kThreadsN == kTileN;
  const int tiles_n = (input_size_n + kTileN - 1) / kTileN;
  const int tiles = (input_size_m + TILE_M - 1) / TILE_M * tiles_n;
  // AN INBOX SLOT: rank r's [m, n] copy, in elements; INT4, its payload in
  // bytes, whole kInt4ScalesAlign (the payload's own alignment).
  const int64_t slot = QUANTIZE_INT4
                           ? (int4_payload_bytes(input_size_m, input_size_n) +
                              kInt4ScalesAlign - 1) /
                                 kInt4ScalesAlign * kInt4ScalesAlign
                           : int64_t{input_size_m} * input_size_n;
  // 1. This rank's tile into slot [rank] of every rank's inbox (this one's
  //    too).
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int offs_m = t / tiles_n * TILE_M, offs_n = t % tiles_n * kTileN;
    const int clip_n =
        kWhole ? input_size_n : min(input_size_n, offs_n + kTileN);
    DTYPE mine[kRows][kPacks][kPackElems];
    load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
        &input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1, input_size_m,
        input_size_n, input_size_m, clip_n, mine, offs_m, offs_n, 1);
#pragma unroll
    for (int r = 0; r < WORLD; ++r)
      send<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
           Layout::tile, TAIL>(mine, staging_ptr_array, staging_ptr_array_size,
                               r, rank, slot, input_size_n, 1, WORLD,
                               input_size_m, input_size_n, input_size_m, clip_n,
                               offs_m, offs_n, 1);
  }
  // 2. Every rank's copy in this rank's inbox (ordered).
  barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                 signals_stride_parity, signals_stride_block,
                 signals_stride_writer, signals_size_parity, signals_size_block,
                 signals_size_writer, status_ptr, status_stride_field,
                 status_size_field, rank, timeout_ticks, Until::launched, 0);
  // 3. Every rank's tile out of the inbox, summed in rank order in fp32,
  //    rounded once: every rank's sums agree bitwise.
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int offs_m = t / tiles_n * TILE_M, offs_n = t % tiles_n * kTileN;
    const int clip_n =
        kWhole ? input_size_n : min(input_size_n, offs_n + kTileN);
    DTYPE copies[WORLD][kRows][kPacks][kPackElems];
    all_gather<AllGatherAlgorithm::tile_staged, QUANTIZE_INT4, DTYPE, WORLD,
               TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, TAIL>(
        staging_ptr_array, staging_ptr_array_size, slot, input_size_m,
        input_size_n, clip_n, offs_m, offs_n, rank, copies);
    DTYPE rounded[kRows][kPacks][kPackElems];
#pragma unroll
    for (int i = 0; i < kRows; ++i)
#pragma unroll
      for (int k = 0; k < kPacks; ++k)
#pragma unroll
        for (int j = 0; j < kPackElems; ++j) {
          float acc = static_cast<float>(copies[0][i][k][j]);
#pragma unroll
          for (int r = 1; r < WORLD; ++r)
            acc += static_cast<float>(copies[r][i][k][j]);
          rounded[i][k][j] = static_cast<DTYPE>(acc);
        }
    store<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
        rounded, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1,
        input_size_m, input_size_n, input_size_m, clip_n, offs_m, offs_n, 1);
  }
  // 4. Every rank done with its inbox: the next launch may store into it.
  barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                 signals_stride_parity, signals_stride_block,
                 signals_stride_writer, signals_size_parity, signals_size_block,
                 signals_size_writer, status_ptr, status_stride_field,
                 status_size_field, rank, timeout_ticks, Until::read, 1);
}

// THE PUSHED TWO-SHOT: each rank writes every rank's share into that rank's
// inbox; each rank sums its inbox into its partials; then every rank reads
// every rank's partials.
template <bool QUANTIZE_INT4, typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES, bool CONTIGUOUS,
          bool TAIL>
DINLINE void all_reduce_two_shot_push(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, DTYPE* output_ptr,
    int64_t output_stride_m, int64_t output_stride_n,
    char* const* partials_ptr_array, int partials_ptr_array_size,
    int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
    int partials_size_n, char* const* staging_ptr_array,
    int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    uint32_t* const* signals_ptr_array, int signals_ptr_array_size,
    int64_t signals_stride_parity, int64_t signals_stride_block,
    int64_t signals_stride_writer, int signals_size_parity,
    int signals_size_block, int signals_size_writer, uint32_t* status_ptr,
    int64_t status_stride_field, int status_size_field, int rank,
    uint64_t timeout_ticks) {
  // PUSH TWO-SHOT: reduce-scatter (each rank's share pushed into that rank's
  // inbox at slot [rank], barrier, the inbox summed into this rank's partials,
  // barrier), then all-gather. The next call's first barrier keeps a rank from
  // overwriting its partials while a peer still reads them.
  reduce_scatter<ReduceScatterAlgorithm::flat_push, QUANTIZE_INT4, WORLD,
                 TILE_M, TILE_N, THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES,
                 CONTIGUOUS, TAIL>(
      input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
      static_cast<const DTYPE* const*>(nullptr), 0, 0, 1, staging_ptr_array,
      staging_ptr_array_size, staging_stride_m, staging_stride_n,
      staging_size_m, staging_size_n, partials_ptr_array,
      partials_ptr_array_size, partials_stride_m, partials_stride_n,
      partials_size_m, partials_size_n, signals_ptr_array,
      signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);
  all_gather<AllGatherAlgorithm::flat, QUANTIZE_INT4, DTYPE, WORLD, TILE_N,
             THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, CONTIGUOUS, TAIL>(
      partials_ptr_array, partials_ptr_array_size, partials_stride_m,
      partials_stride_n, partials_size_m, partials_size_n, output_ptr,
      output_stride_m, output_stride_n, input_size_m, input_size_n, 0, 0, 0,
      rank);
}

template <AllReduceAlgorithm ALGORITHM, bool QUANTIZE_INT4, typename DTYPE,
          int WORLD, int TILE_M, int TILE_N, int THREADS_PER_BLOCK,
          int WAVE_SIZE, int PACK_BYTES, int WAVES_PER_EU, bool CONTIGUOUS,
          bool TAIL>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU) all_reduce(
    const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
    int64_t input_stride_n, int input_size_m, int input_size_n,
    const DTYPE* const* __restrict__ input_ptr_array, int input_ptr_array_size,
    int64_t input_ptr_array_stride_m, int64_t input_ptr_array_stride_n,
    DTYPE* __restrict__ output_ptr, int64_t output_stride_m,
    int64_t output_stride_n, int output_size_m, int output_size_n,
    char* const* __restrict__ partials_ptr_array, int partials_ptr_array_size,
    int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
    int partials_size_n, char* const* __restrict__ staging_ptr_array,
    int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    char* const* __restrict__ unfenced_ptr_array, int unfenced_ptr_array_size,
    int64_t unfenced_stride_slot, int64_t unfenced_stride_m,
    int64_t unfenced_stride_n, int unfenced_size_slot, int unfenced_size_m,
    int unfenced_size_n, int32_t* __restrict__ unfenced_epochs_ptr,
    int64_t unfenced_epochs_stride_block, int64_t unfenced_epochs_stride_n,
    int unfenced_epochs_size_block, int unfenced_epochs_size_n,
    uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size,
    int64_t signals_stride_parity, int64_t signals_stride_block,
    int64_t signals_stride_writer, int signals_size_parity,
    int signals_size_block, int signals_size_writer,
    uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
    int status_size_field, int rank, uint64_t timeout_ticks) {
  // THE DISPATCH, an algorithm a function; QUANTIZE_INT4 (the host's, from the
  // config's quantization and quantize_from_bytes) whether this launch moves
  // INT4.
  if constexpr (ALGORITHM == AllReduceAlgorithm::one_shot_pull)
    all_reduce_one_shot_pull<QUANTIZE_INT4, DTYPE, WORLD, TILE_M, TILE_N,
                             THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES,
                             CONTIGUOUS, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m,
        input_ptr_array_stride_n, output_ptr, output_stride_m, output_stride_n,
        staging_ptr_array, staging_ptr_array_size, signals_ptr_array,
        signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks);
  else if constexpr (ALGORITHM == AllReduceAlgorithm::two_shot_pull)
    all_reduce_two_shot_pull<QUANTIZE_INT4, DTYPE, WORLD, TILE_M, TILE_N,
                             THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES,
                             CONTIGUOUS, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m,
        input_ptr_array_stride_n, output_ptr, output_stride_m, output_stride_n,
        partials_ptr_array, partials_ptr_array_size, partials_stride_m,
        partials_stride_n, partials_size_m, partials_size_n, staging_ptr_array,
        staging_ptr_array_size, staging_stride_m, staging_stride_n,
        staging_size_m, staging_size_n, signals_ptr_array,
        signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks);
  else if constexpr (ALGORITHM == AllReduceAlgorithm::one_shot_push)
    all_reduce_one_shot_push<QUANTIZE_INT4, DTYPE, WORLD, TILE_M, TILE_N,
                             THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES,
                             CONTIGUOUS, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        output_ptr, output_stride_m, output_stride_n, unfenced_ptr_array,
        unfenced_ptr_array_size, unfenced_stride_slot, unfenced_stride_m,
        unfenced_stride_n, unfenced_size_slot, unfenced_size_m, unfenced_size_n,
        unfenced_epochs_ptr, unfenced_epochs_stride_block,
        unfenced_epochs_stride_n, unfenced_epochs_size_block,
        unfenced_epochs_size_n, status_ptr, status_stride_field,
        status_size_field, rank, timeout_ticks);
  else if constexpr (ALGORITHM == AllReduceAlgorithm::one_shot_push_staged)
    all_reduce_one_shot_push_staged<QUANTIZE_INT4, DTYPE, WORLD, TILE_M, TILE_N,
                                    THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES,
                                    CONTIGUOUS, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        output_ptr, output_stride_m, output_stride_n, staging_ptr_array,
        staging_ptr_array_size, signals_ptr_array, signals_ptr_array_size,
        signals_stride_parity, signals_stride_block, signals_stride_writer,
        signals_size_parity, signals_size_block, signals_size_writer,
        status_ptr, status_stride_field, status_size_field, rank,
        timeout_ticks);
  else if constexpr (ALGORITHM == AllReduceAlgorithm::two_shot_push)
    all_reduce_two_shot_push<QUANTIZE_INT4, DTYPE, WORLD, TILE_M, TILE_N,
                             THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES,
                             CONTIGUOUS, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        output_ptr, output_stride_m, output_stride_n, partials_ptr_array,
        partials_ptr_array_size, partials_stride_m, partials_stride_n,
        partials_size_m, partials_size_n, staging_ptr_array,
        staging_ptr_array_size, staging_stride_m, staging_stride_n,
        staging_size_m, staging_size_n, signals_ptr_array,
        signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks);
}

}  // namespace hip_custom_communicator
