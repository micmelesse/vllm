// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE PLAIN ALL-REDUCE: its kernels, one section each. THE DEFAULT, PULLED IN
// PLACE (all_reduce_pull_one_shot, all_reduce_pull_two_shot;
// aiter's gfx950 cross_device_reduce_1stage / 2stage): ONE kernel a shape
// reads every rank's input where the INPUT TABLE says (common.cuh's INTERFACE:
// a captured call's registered tensors, or an eager call's copies in every
// rank's inputs region, handle.cuh's INPUT TABLES), no copy inside the kernel.
// Its inputs are never written while it runs: the one-shot ends with a `read`
// barrier and the two-shot's second barrier follows its every read of them, so
// when a rank's launch ends every peer is done reading its input, and the next
// call's host copy (stream-ordered after it) or the caller may overwrite it.
// The INT4 two-shot (all_reduce_push_two_shot<..., int4>) encodes its
// input itself: what crosses the links is quantized by its owner, so it cannot
// be read in place; it pushes the quantized shares (below).
//
// THE PUSHED VARIANTS, each PUSHED: a rank reads only its own input (the
// caller's tensor, which no peer reads) and stores it, or each rank's share of
// it, into the peers' receive areas (the INBOX: [writer slot, ...]); each rank
// sums the WORLD slots of its inbox LOCALLY. Nothing is copied in first.
// all_reduce_push_one_shot (decode; forced until a tuned entry names it)
// pushes LINES, the flag in the data, into its own inbox in the unfenced
// region (the host's) and takes no barrier (its section says why the inbox is safe to
// reuse). all_reduce_push_two_shot (forced only) pushes into the staging, a
// band of rows a pass, and sums after an ordered barrier; the int4 two-shot
// pushes its shares quantized. THE STAGING INBOX IS REUSED, launch after launch
// and pass after pass, with no barrier of its own: a rank pushes into peer s's
// slot only after passing the barrier that follows peer s's local sum of it
// (the two-shots' second, `visible`), and a peer arrives there only once its
// block's sum is done (its loads returned: the sums stored after depend on
// them). Across launches every block of the previous launch passed it, so a
// different grid or another kernel before is covered too: every kernel's reads
// of the inputs region come before its last barrier.

#pragma once

#include "all_gather.cuh"
#include "common.cuh"
#include "reduce_scatter.cuh"

namespace hip_custom_communicator {

// =============================================================================
// all_reduce_pull_one_shot
//
// One-shot all-reduce, PULLED IN PLACE: sync (every rank's input ready and
// visible), read every rank's input through the input table, sum, sync (every
// peer done reading this rank's). aiter's `cross_device_reduce_1stage`. No band
// loop: nothing is staged, so any size runs in one pass.
// =============================================================================

// THE BUFFER AS [m, n], every rank's at the table's strides (input_ptr_array_stride_m,
// input_ptr_array_stride_n), a work item a TILE_M x TILE_N tile, the grid striding over
// the tiles row-major. ANY BUILD WALKS ITS TILE (the dense map, STANDARDS 3.2):
// TILE_M rows of threads a row each where they divide the block, else one row of
// threads looping over the rows; a row of threads' packs in kPacks steps, the
// last step's packs past the tile MASKED where the row of threads is not a
// divisor of the tile (a thread past it loads the tile's last pack and stores
// nothing: the clips are the tile's). The one rule left: the tile is whole packs
// (a pack is one aligned 16-byte access; a tile edge inside one would split it
// between two blocks).


// =============================================================================
// all_reduce_pull_two_shot
//
// Two-shot all-reduce, PULLED IN PLACE (exact): a band of `partials_size_m`
// rows a pass (what the partials hold): sync, this rank's share of the band
// read from every rank's input through the table and summed into its
// partials, sync, every rank's summed share pulled out of its partials. aiter's
// `cross_device_reduce_2stage`.
// =============================================================================

// THE PROBLEM as the pushed two-shot's (the flat walk, each rank's share one
// contiguous range of the band's elements), every rank's input at the table's
// strides. CONTIGUOUS: the table's rows and the output's back to back.


// =============================================================================
// all_reduce_push_one_shot
//
// One-shot all-reduce, PUSHED WITH THE FLAG IN THE LINE (NCCL's LL; aiter's
// `ar_ll_gfx9`): each rank stores its own tile into slot [rank] of every peer's
// inbox as LINES (common.cuh's Layout::line: 8 bytes of payload and this
// call's epoch in each 16-byte line, one store), then polls its OWN inbox
// until every peer's lines carry the epoch, and sums. ONE FABRIC TRIP AND NO
// BARRIER: a line that carries the epoch is its data. The input is read only by
// its owner.
// =============================================================================

// THE BUFFER AS [m, n] AT ITS STRIDES, a work item a TILE_M x TILE_N tile, the
// grid striding over the tiles row-major: A TILE A BLOCK A PASS. THE INBOX
// (`lines`), [writer, row, n] in payload elements, two banks of unfenced_size_m / 2
// rows (the host's unfenced region), a block's TILE_M rows from blockIdx.x * TILE_M
// of a bank (the grid's rows at most a bank's, select's), the row's pitch fixed
// whatever the build; ONE WRITER'S ROWS ARE CONTIGUOUS (aiter's dense packet
// index): as [bank, block, writer] slots, a writer's lines a block apart sat 32 KiB
// apart.
// THE EPOCH (`epochs`, [row, n], a row's word at column 0), PER ROW, ON THE
// DEVICE: read at the start, +1 a pass that writes the row (never 0: the inbox
// starts zeroed; past 0xffffffff it is 2, keeping the banks alternating), written
// back at the end, so a CUDA-graph replay, which runs none of our host code,
// counts on and the inbox is never cleared. A ROW'S, NOT A BLOCK'S: a block's rows
// are its TILE_M's, so at another TILE_M a row is another block's, and a block's
// count was not the row's (at TILE_M 2 block b writes rows 2b and 2b + 1; back at
// TILE_M 1, block 2b + 1, behind block b, counted to the epoch b's stale line
// carried and took it for data). A row's every write counts on its own epoch,
// whichever block writes it. Every rank runs the same calls at the same builds and
// grids, so row q's epoch is the same on every rank. (STANDARDS 3.2 says nothing
// persists between launches; this counter does, by Michael's design, October 18:
// the flag in the line needs a per-call value no host writes.) A ROW'S BANK =
// its epoch & 1 (common.cuh's line layout).
// WHY TWO BANKS NEED NO CLOSING BARRIER (aiter's argument, per row): a rank
// writes row q's epoch e into a peer's bank (e & 1) only after it saw that
// peer's row q lines of e - 1, which the peer sends only once it is done with
// e - 2, the bank's last use: in the same launch, its row's one block passed
// the __syncthreads after its pass e - 2; in an earlier launch, that launch ended
// before the one sending e - 1 began (stream order). A rank starts call e + 2
// only after every peer started e + 1, which they do only after finishing e.
// The argument is a row's, not a thread's: a lane past the clip's columns writes
// its clamped place too (the same bytes as its owner; common.cuh's line write), a
// row past the clip is neither written nor counted, and the __syncthreads closing
// each pass covers every lane.


// =============================================================================
// all_reduce_push_two_shot
//
// Two-shot all-reduce: reduce-scatter, then all-gather, the sum kept in this
// rank's partials and pulled by the peers. THE
// REDUCE-SCATTER IS PUSHED: each rank reads its own input and stores every
// rank's share of it into that rank's inbox at slot [rank], sync (ordered),
// each rank sums the WORLD slots of its share locally into its partials, sync,
// every rank's summed share pulled out of its partials (aiter's
// `cross_device_reduce_2stage` second half). INT4: the shares
// quantized by the kernel before they are pushed; exact (none), a band of rows
// a pass. No peer reads the caller's input.
// =============================================================================

// THE PROBLEM, [R, M, N]: R = WORLD ranks' inputs, each an [input_size_m,
// input_size_n] buffer at its strides, summed elementwise. Reduce-scatter (each
// rank sums its share over the ranks into its partials), sync, all-gather (every
// rank's summed share out to the output). The kernel states the problem;
// common.cuh's flat walk cuts it as [R, 1, M*N], each rank's share one
// contiguous range of the elements, for long runs (its comment says why and
// what it serves badly), a TILE_N chunk a step (the flat walk has no rows: TILE_M is
// the one-shots'; any block and any tile walk it, rounded up to whole packs, the
// collectives' flat walk says how). CONTIGUOUS: the input's and the output's rows
// are back to back, the instance the host picks from their strides.
// QUANTIZATION: what crosses the links (common.cuh's QUANTIZATION); with int4
// each share is quantized by this rank, from its own input, and pushed into the share's rank's inbox INT4 payload at this rank's slot
// (the payload as an [m, n] one, writer w's slot where share w's groups lie),
// and the sums' are the partials' INT4 payload, a group a thread (the INT4 flat
// walk's). NONE: the same shares transferred exactly, a band of rows a pass.

// EXACT, IN PASSES: a band of `partials_size_m` rows a pass (the host's: what
// the inbox and the partials hold), so any size runs in one launch. A band is
// the problem's rows [r0, r0 + band) at their strides. THE INBOX, the staging
// region as [WORLD, slot]: writer w's slot at row w, staging_stride_m elements
// a slot, holding writer w's copy of this rank's share of the band from the
// slot's first element. EACH BLOCK PUSHES, OF EVERY RANK'S SHARE, THE CHUNKS
// THAT RANK'S SAME BLOCK SUMS, so a peers barrier makes them visible.

// =============================================================================
// all_reduce: THE PLAIN ALL-REDUCE, ONE KERNEL: its ALGORITHM (common.cuh's Algorithm) is a
// template parameter, and each algorithm it is built for (pull one-shot, pull two-shot, push one-shot, push two-shot) is its own
// body under `if constexpr`, as its section above says. An algorithm it has no body for does not
// compile.
// =============================================================================
template <typename DTYPE, int WORLD, int TILE_M, int TILE_N, int THREADS_PER_BLOCK, int WAVE_SIZE,
          int PACK_BYTES, int WAVES_PER_EU, int QUANTIZATION, int ALGORITHM,
          bool CONTIGUOUS, bool TAIL>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const DTYPE* const* __restrict__ input_ptr_array, int input_ptr_array_size,
        int64_t input_ptr_array_stride_m, int64_t input_ptr_array_stride_n,
        DTYPE* __restrict__ output_ptr, int64_t output_stride_m,
        int64_t output_stride_n, int output_size_m, int output_size_n,
        char* const* __restrict__ partials_ptr_array, int partials_ptr_array_size, int64_t partials_stride_m,
        int64_t partials_stride_n, int partials_size_m, int partials_size_n,
        char* const* __restrict__ staging_ptr_array, int staging_ptr_array_size, int64_t staging_stride_m,
        int64_t staging_stride_n, int staging_size_m, int staging_size_n,
        char* const* __restrict__ unfenced_ptr_array, int unfenced_ptr_array_size,
        int64_t unfenced_stride_slot, int64_t unfenced_stride_m, int64_t unfenced_stride_n,
        int unfenced_size_slot, int unfenced_size_m, int unfenced_size_n,
        int32_t* __restrict__ unfenced_epochs_ptr, int64_t unfenced_epochs_stride_block, int64_t unfenced_epochs_stride_n, int unfenced_epochs_size_block,
        int unfenced_epochs_size_n,
        uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size,
        int64_t signals_stride_parity, int64_t signals_stride_block,
        int64_t signals_stride_writer, int signals_size_parity,
        int signals_size_block, int signals_size_writer,
        uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
        int status_size_field, int rank, uint64_t timeout_ticks) {
  if constexpr (ALGORITHM == Algorithm::one_shot_pull) {
    // PULL ONE-SHOT (all_reduce_pull_one_shot, above).
    // EXACT ALONE: an INT4 build of it compiles this exact body, which never runs (the host refuses
    // an INT4 call of it by name, quantization_not_supported).
    constexpr int kThreadsM = THREADS_PER_BLOCK % TILE_M == 0 ? TILE_M : 1;
    constexpr int kThreadsN = THREADS_PER_BLOCK / kThreadsM;
    constexpr int kRows = TILE_M / kThreadsM;
    constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
    // THE TILE IN WHOLE PACKS (TILE_N rounded up): a tile edge inside a pack would split one aligned
    // access between two blocks; any TILE_N walks, the row's columns past the input masked.
    constexpr int kTileN = (TILE_N + kPackElems - 1) / kPackElems * kPackElems;
    constexpr int kPacks = (kTileN + kPackElems * kThreadsN - 1) / (kPackElems * kThreadsN);
    // A ROW OF THREADS AS WIDE AS THE TILE IN WHOLE STEPS, else the last step masked at the tile.
    constexpr bool kWhole = kPacks * kPackElems * kThreadsN == kTileN;
    const int tiles_n = (input_size_n + kTileN - 1) / kTileN;
    const int tiles = (input_size_m + TILE_M - 1) / TILE_M * tiles_n;
    // 0. Every rank's input entry, read before the barrier (load's entry form: the table is fixed
    // for the launch), so its round trip runs under the barrier's.
    const DTYPE* input_entries[WORLD];
  #pragma unroll
    for (int r = 0; r < WORLD; ++r) input_entries[r] = load(input_ptr_array, input_ptr_array_size, r);
    // 1. Every rank's input is ready (written before its launch: the caller's,
    // or the host's copy) and visible (the first barrier is ordered).
    barrier<WORLD>(
        signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks, Until::started, 0);
    // 2. Every rank's input, in rank order, through the table, summed.
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      // The columns this tile clips at: the input's, or the tile's end where a step is masked.
      const int clip_n = kWhole ? input_size_n : min(input_size_n, t % tiles_n * kTileN + kTileN);
      DTYPE peers[WORLD][kRows][kPacks][kPackElems];
  #pragma unroll
      for (int r = 0; r < WORLD; ++r)
        recv<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, Quantization::exact, Layout::tile>(&input_entries[r], 1, 0, 0, 0, input_ptr_array_stride_m, input_ptr_array_stride_n, 1, input_size_m, input_size_n, input_size_m, clip_n, peers[r], t / tiles_n * TILE_M, t % tiles_n * kTileN, 1);
      // Summed in rank order in fp32, rounded to DTYPE once: every rank sums the same copies in the
      // same order, so every rank's sums agree bitwise.
      DTYPE rounded[kRows][kPacks][kPackElems];
  #pragma unroll
      for (int i = 0; i < kRows; ++i)
  #pragma unroll
        for (int k = 0; k < kPacks; ++k)
  #pragma unroll
          for (int j = 0; j < kPackElems; ++j) {
            float acc = static_cast<float>(peers[0][i][k][j]);
  #pragma unroll
            for (int r = 1; r < WORLD; ++r) acc += static_cast<float>(peers[r][i][k][j]);
            rounded[i][k][j] = static_cast<DTYPE>(acc);
          }
      store<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, Quantization::exact, Layout::tile>(rounded, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, input_size_m, input_size_n, input_size_m, clip_n, t / tiles_n * TILE_M, t % tiles_n * kTileN, 1);
    }
    // 3. Every peer is done reading this rank's input: the launch may end and
    // the input be overwritten.
    barrier<WORLD>(
        signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks, Until::read, 1);
  } else if constexpr (ALGORITHM == Algorithm::two_shot_pull) {
    // PULL TWO-SHOT (all_reduce_pull_two_shot, above).
    // EXACT ALONE: an INT4 build of it compiles this exact body, which never runs (the host refuses
    // an INT4 call of it by name, quantization_not_supported).
    for (int r0 = 0; r0 < input_size_m; r0 += partials_size_m) {
      const int band =
          min(partials_size_m, input_size_m - r0);  // this pass's rows
      DTYPE* const band_output = output_ptr + r0 * output_stride_m;
      // 1-3. Every rank's input ready and visible (and, past the first pass, every peer has read
      //    this rank's partials); this rank's share of the band from every rank's input, where the
      //    table says, summed into its partials; every rank's sums visible and every peer done
      //    reading this band of this rank's input.
      reduce_scatter<WORLD, TILE_M, TILE_N, THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, Quantization::exact, CONTIGUOUS, Walk::flat, Accumulate::lds, Direction::pull, Until::started, false>(
          static_cast<const DTYPE*>(nullptr), 0, 1, band, input_size_n, input_ptr_array,
          input_ptr_array_size, input_ptr_array_stride_m, input_ptr_array_stride_n, nullptr, 0, 0, 1, 0, 0,
          partials_ptr_array, partials_ptr_array_size, partials_stride_m, partials_stride_n,
          partials_size_m, partials_size_n, signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
          signals_stride_writer, signals_size_parity, signals_size_block,
          signals_size_writer, status_ptr, status_stride_field, status_size_field, r0, rank, timeout_ticks);

      // 4. All-gather: every rank's summed share out of its partials, at its
      //    place in the output.
      all_gather<DTYPE, WORLD, TILE_N, THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, Quantization::exact, CONTIGUOUS, Walk::flat, false>(partials_ptr_array, partials_ptr_array_size, partials_stride_m,
                        partials_stride_n, partials_size_m, partials_size_n, band_output,
                        output_stride_m, output_stride_n, band, input_size_n, 0, 0, 0, rank);
    }
  } else if constexpr (ALGORITHM == Algorithm::one_shot_push) {
    // PUSH ONE-SHOT (all_reduce_push_one_shot, above).
    // EXACT ALONE: an INT4 build of it compiles this exact body, which never runs (the host refuses
    // an INT4 call of it by name, quantization_not_supported).
    // THE TILE'S WALK, the pulled one-shot's (its section): rows of threads where they divide the
    // block, the last step's packs past the tile masked (a lane past it loads, and so sends, the
    // tile's last pack: the same bytes its owner sends; its sum is stored nowhere).
    constexpr int kThreadsM = THREADS_PER_BLOCK % TILE_M == 0 ? TILE_M : 1;
    constexpr int kThreadsN = THREADS_PER_BLOCK / kThreadsM;
    // A THREAD'S PACK IS ONE LINE'S PAYLOAD, 8 bytes (aiter's packet), or the build's narrower pack
    // alone in its line (PACK_BYTES 2 or 4: a row not whole 8 bytes): a block's stores spread over
    // twice the waves of a 16-byte pack, half the stores a wave (16 stores a wave at 128 threads
    // were ~9k cycles of issue stalls, ATT 2026-10-19T04-50-53Z).
    constexpr int kPackElems =
        (PACK_BYTES < kLinePayloadBytes ? PACK_BYTES : kLinePayloadBytes) / static_cast<int>(sizeof(DTYPE));
    constexpr int kRows = TILE_M / kThreadsM;
    // THE TILE IN WHOLE PACKS (TILE_N rounded up): a tile edge inside a pack would split one aligned
    // access between two blocks; any TILE_N walks, the row's columns past the input masked.
    constexpr int kTileN = (TILE_N + kPackElems - 1) / kPackElems * kPackElems;
    constexpr int kPacks = (kTileN + kPackElems * kThreadsN - 1) / (kPackElems * kThreadsN);
    constexpr bool kWhole = kPacks * kPackElems * kThreadsN == kTileN;
    const int tiles_n = (input_size_n + kTileN - 1) / kTileN;
    const int tiles = (input_size_m + TILE_M - 1) / TILE_M * tiles_n;
    // THE BLOCK'S ROWS OF A BANK: TILE_M from blockIdx.x * TILE_M (the inbox's two banks of
    // unfenced_size_m / 2 rows each; the grid's TILE_M rows at most that, select's).
    const int first_row = static_cast<int>(blockIdx.x) * TILE_M;
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG && static_cast<int>(gridDim.x) * TILE_M > unfenced_size_m / 2)
      __builtin_trap();
    // 0. THIS THREAD'S ROWS' LAST EPOCHS (a 16-byte line a row, the row's every thread's). The
    //    epochs are this rank's alone, so a pointer of their own (`&unfenced_epochs_ptr, 1`): through
    //    a device array each launch began with a cold scalar load before the epoch's (ATT
    //    2026-10-19T04-50-53Z: ~5k cycles of lgkmcnt waits before the first send).
    int32_t last[kRows][1][kLineBytes / static_cast<int>(sizeof(int32_t))];  // a 16-byte line a row, any pack
    load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, Quantization::exact, Layout::tile>(
        &unfenced_epochs_ptr, 1, 0, 0, 0, unfenced_epochs_stride_block, unfenced_epochs_stride_n,
        1, unfenced_epochs_size_block, unfenced_epochs_size_n, unfenced_epochs_size_block, unfenced_epochs_size_n, last, first_row, 0,
        1);
    // The epochs are read (a wait for that load) only once the first tile's load is issued, so the
    // two round trips overlap (ATT 2026-10-19T05-22-25Z: ~1.2k cycles a wave waited for the epoch
    // alone). One row of threads' rows are the block's: read from the first lane, scalar.
    uint32_t epochs[kRows] = {};
    bool ran = false;
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      const int offs_m = t / tiles_n * TILE_M, offs_n = t % tiles_n * kTileN;
      // The tile's rows and columns inside the input: the lines past them are nobody's.
      const int clip_m = min(TILE_M, input_size_m - offs_m);
      const int clip_n = min(kTileN, input_size_n - offs_n);
      // 1. This rank's tile, read here, into slot [rank] of every peer's inbox, its lines carrying
      //    the epoch. EVERY RANK'S, THIS ONE'S TOO (its own slot, read back as a peer's in 2), A RANK THE COMPILER
      //    KNOWS: no branch between the sends, whose joins waited for every store before
      //    (s_waitcnt vmcnt(0)), so the remote stores ran one after another (ATT
      //    2026-10-19T04-09-17Z: 40k of 60k cycles a wave; 2.3-2.9x aiter at 1-8 tokens). The
      //    seven peers as (rank + i) % WORLD read from the first lane still compiled both of send's
      //    paths and a scalar load of each peer's entry before its store: 863 -> 1535 instructions,
      //    the stores 57 apart (ISA, d7da3439da).
      // The input's columns this tile clips at: the input's, or the tile's end where a step is
      // masked (so a lane past the tile holds its owner's pack).
      const int input_clip_n = kWhole ? input_size_n : offs_n + clip_n;
      DTYPE mine[kRows][kPacks][kPackElems];
      load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, Quantization::exact, Layout::tile>(&input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1, input_size_m, input_size_n, input_size_m, input_clip_n, mine, offs_m, offs_n, 1);
      // This pass's epoch a row (its last + 1, never 0), its lines in its bank (epoch & 1). A row
      // past the tile's is not written, so not counted.
  #pragma unroll
      for (int i = 0; i < kRows; ++i) {
        const uint32_t before =
            ran ? epochs[i]
                : kThreadsM == 1 ? __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(last[i][0][0]))
                                 : static_cast<uint32_t>(last[i][0][0]);
        const bool written = (kThreadsM == 1 ? i : static_cast<int>(threadIdx.x) / kThreadsN + i * kThreadsM) < clip_m;
        epochs[i] = !written ? before : before + 1u == 0u ? 2u : before + 1u;
      }
      ran = true;
  #pragma unroll
      for (int r = 0; r < WORLD; ++r)
        send<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, Quantization::exact, Layout::line>(mine, unfenced_ptr_array, unfenced_ptr_array_size, r, rank, unfenced_stride_slot, unfenced_stride_m, unfenced_stride_n, unfenced_size_slot, unfenced_size_m, unfenced_size_n, first_row + clip_m, clip_n, first_row, 0, 1, epochs);
      // 2. Every peer's tile out of this rank's inbox, polled until its lines carry the epoch.
      DTYPE peers[WORLD][kRows][kPacks][kPackElems];
      load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, Quantization::exact, Layout::line>(unfenced_ptr_array, unfenced_ptr_array_size, rank, 0, unfenced_stride_slot, unfenced_stride_m, unfenced_stride_n, unfenced_size_slot, unfenced_size_m, unfenced_size_n, first_row + clip_m, clip_n, peers, first_row, 0, 1, epochs, status_ptr, status_stride_field, status_size_field, timeout_ticks);
      // 3. Summed in rank order in fp32, every rank's from the inbox (this rank's own line too, the
      //    same bits as its input: no select on a runtime rank), rounded to DTYPE once: every rank
      //    sums the same copies in the same order, so every rank's sums agree bitwise.
      DTYPE rounded[kRows][kPacks][kPackElems];
  #pragma unroll
      for (int i = 0; i < kRows; ++i)
  #pragma unroll
        for (int k = 0; k < kPacks; ++k)
  #pragma unroll
          for (int j = 0; j < kPackElems; ++j) {
            float acc = static_cast<float>(peers[0][i][k][j]);
  #pragma unroll
            for (int r = 1; r < WORLD; ++r) acc += static_cast<float>(peers[r][i][k][j]);
            rounded[i][k][j] = static_cast<DTYPE>(acc);
          }
      store<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, Quantization::exact, Layout::tile>(rounded, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1, input_size_m, input_size_n, input_size_m, input_clip_n, offs_m, offs_n, 1);
      // 4. The block is done with this pass's bank before any of its threads sends the next pass
      //    (the bank argument above), and, the last pass, every wave has read the epoch (step 0).
      __syncthreads();
    }
    // 5. Each row's epoch, for its next write in the next launch: its row of threads' first lane's
    //    store (the store's lanes past the line's one pack store nothing).
    int32_t next[kRows][1][kLineBytes / static_cast<int>(sizeof(int32_t))] = {};
  #pragma unroll
    for (int i = 0; i < kRows; ++i)
      next[i][0][0] = ran ? static_cast<int32_t>(epochs[i]) : last[i][0][0];
    store<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, Quantization::exact, Layout::tile>(
        next, &unfenced_epochs_ptr, 1, 0, 0, 0, unfenced_epochs_stride_block,
        unfenced_epochs_stride_n, 1, unfenced_epochs_size_block, unfenced_epochs_size_n, unfenced_epochs_size_block, unfenced_epochs_size_n,
        first_row, 0, 1);
  } else {
    // PUSH TWO-SHOT (all_reduce_push_two_shot, above).
    static_assert(ALGORITHM == Algorithm::two_shot_push,
                  "all_reduce: no body for this algorithm");
    static_assert(QUANTIZATION == Quantization::exact || QUANTIZATION == Quantization::int4,
                  "unsupported quantization");
    if constexpr (QUANTIZATION == Quantization::int4) {
      // THE INT4 PAYLOADS are inside the staging and the partials (common.cuh's
      // INT4 payload layout): the collectives find them from each region's sizes.
      // 1-3. Every rank's share of this rank's input, quantized here, into that rank's inbox payload
      // at slot [rank]; every peer launched and pushed its own into ours (ordered); this rank's share,
      // every writer's slot summed, into its partials; every rank's sums visible and every peer done
      // with its inbox (the next launch's push may overwrite it).
      reduce_scatter<WORLD, TILE_M, TILE_N, THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, QUANTIZATION, CONTIGUOUS, Walk::flat, Accumulate::registers, Direction::push, Until::launched, TAIL>(
          input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
          static_cast<const DTYPE* const*>(nullptr), 0, 0, 1,
          staging_ptr_array, staging_ptr_array_size, staging_stride_m, staging_stride_n,
          staging_size_m, staging_size_n, partials_ptr_array, partials_ptr_array_size,
          partials_stride_m, partials_stride_n, partials_size_m, partials_size_n, signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
            signals_stride_writer, signals_size_parity, signals_size_block,
            signals_size_writer, status_ptr, status_stride_field, status_size_field, 0, rank,
          timeout_ticks);

      // 4. All-gather: every rank's summed share out of its partials, at its place
      // in the output. The next call's first sync keeps a rank from overwriting
      // its partials while it is read (they are written only after it).
      all_gather<DTYPE, WORLD, TILE_N, THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, QUANTIZATION, CONTIGUOUS, Walk::flat, TAIL>(
          partials_ptr_array, partials_ptr_array_size, partials_stride_m, partials_stride_n,
          partials_size_m, partials_size_n, output_ptr, output_stride_m, output_stride_n,
          input_size_m, input_size_n, 0, 0, 0, rank);
    } else {
      for (int r0 = 0; r0 < input_size_m; r0 += partials_size_m) {
        const int band =
            min(partials_size_m, input_size_m - r0);  // this pass's rows
        DTYPE* const band_output = output_ptr + r0 * output_stride_m;
        // 1-3. Every rank's share of this band, read here, into that rank's inbox at slot [rank] (the
        //    chunks each rank's same block sums), then visible (and, past the first pass, every peer
        //    has read this rank's partials); this rank's share, every writer's slot of its inbox
        //    summed, into its partials; every rank's sums visible and every peer done with its inbox,
        //    so the next band's (or launch's) push may overwrite it.
        reduce_scatter<WORLD, TILE_M, TILE_N, THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, Quantization::exact, CONTIGUOUS, Walk::flat, Accumulate::lds, Direction::push, Until::visible, false>(
            input_ptr, input_stride_m, input_stride_n, band, input_size_n,
            static_cast<const DTYPE* const*>(nullptr), 0, 0, 1,
            staging_ptr_array, staging_ptr_array_size, staging_stride_m, staging_stride_n,
            staging_size_m, staging_size_n, partials_ptr_array, partials_ptr_array_size,
            partials_stride_m, partials_stride_n, partials_size_m, partials_size_n, signals_ptr_array, signals_ptr_array_size, signals_stride_parity, signals_stride_block,
            signals_stride_writer, signals_size_parity, signals_size_block,
            signals_size_writer, status_ptr, status_stride_field, status_size_field, r0, rank,
            timeout_ticks);

        // 4. All-gather: every rank's summed share out of its partials, at its place
        // in the output.
        all_gather<DTYPE, WORLD, TILE_N, THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, Quantization::exact, CONTIGUOUS, Walk::flat, false>(partials_ptr_array, partials_ptr_array_size, partials_stride_m,
                          partials_stride_n, partials_size_m, partials_size_n, band_output,
                          output_stride_m, output_stride_n, band, input_size_n, 0, 0, 0, rank);
      }
    }
  }
}

}  // namespace hip_custom_communicator
