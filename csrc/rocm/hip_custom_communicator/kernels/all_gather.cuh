#pragma once

// All-gather stages: each rank reads every rank's reduced slice into its
// output.

#include <type_traits>

#include "common.cuh"
#include "reduce_scatter.cuh"

namespace hip_custom_communicator {

enum class AllGatherAlgorithm { flat, rows, columns, tile_pull, tile_push };

template <AllGatherAlgorithm ALGORITHM, Protocol PROTOCOL, bool QUANTIZE_INT4,
          typename DTYPE, int WORLD, int TILE_N, int THREADS_PER_BLOCK,
          int WAVE_SIZE, int PACK_BYTES, bool CONTIGUOUS, bool TAIL,
          typename OUTPUT>
DINLINE void all_gather(char* const* partials_ptr_array,
                        int partials_ptr_array_size, int64_t partials_stride_m,
                        int64_t partials_stride_n, int partials_size_m,
                        int partials_size_n, OUTPUT&& output,
                        int64_t output_stride_m, int64_t output_stride_n,
                        int output_size_m, int output_size_n, int offs_m,
                        int offs_n, int slice, int rank) {
  static_assert(PROTOCOL == Protocol::simple, "this all-gather runs simple");
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
  constexpr bool kInt4 = QUANTIZE_INT4;
  if constexpr (ALGORITHM == AllGatherAlgorithm::flat) {
    DTYPE* const output_ptr = output;
    constexpr int kUnit = kInt4 ? kInt4Group : kPackElems;
    const int64_t total = int64_t{output_size_m} * output_size_n;

    const int64_t units = TAIL ? (total + kUnit - 1) / kUnit : total / kUnit;
    const int slice = static_cast<int>((units + WORLD - 1) / WORLD * kUnit);
    const auto share_len = [&](int r) {
      return static_cast<int>(
          max(int64_t{0}, min(int64_t{slice}, total - int64_t{r} * slice)));
    };
    if constexpr (kInt4) {
      constexpr Layout kOutput =
          CONTIGUOUS ? Layout::group_run : Layout::group_flat;
      const int groups_all = slice / kInt4Group;
      const int grid = static_cast<int>(gridDim.x);
      const int per_block = (groups_all + grid - 1) / grid;
      const int g_first = static_cast<int>(blockIdx.x) * per_block;
      const int g_end = min(g_first + per_block, groups_all);

      constexpr int kRankLanes = THREADS_PER_BLOCK / WORLD;
      static_assert(kRankLanes * WORLD == THREADS_PER_BLOCK,
                    "a block is a row of threads a rank");
      const int r = (rank + static_cast<int>(threadIdx.x) / kRankLanes) % WORLD;
      const int end = TAIL ? min(g_end * kInt4Group, share_len(r))
                           : min(g_end, share_len(r) / kInt4Group) * kInt4Group;
      for (int g0 = g_first; g0 < g_end; g0 += kRankLanes) {
        uint32_t words[1][kInt4GroupPacks][2];
        recv<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
             Protocol::simple, Layout::group_run, TAIL>(
            partials_ptr_array, partials_ptr_array_size, r, 0, 0,
            partials_stride_m, partials_stride_n, 1, partials_size_m,
            partials_size_n, 1, slice, words, 0, g0 * kInt4Group, 1);

        DTYPE x[1][kInt4GroupPacks][kPackElems];
#pragma unroll
        for (int k = 0; k < kInt4GroupPacks; ++k)
          dequantize(words[0][k], x[0][k]);
        store<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, false,
              Protocol::simple, kOutput, TAIL>(
            x, &output_ptr, 1, 0, r, slice, output_stride_m, output_stride_n,
            WORLD, output_size_m, output_size_n, 1, end, 0, g0 * kInt4Group, 1);
      }
    } else {
      constexpr Layout kOutput = CONTIGUOUS ? Layout::run : Layout::flat;
      constexpr int kLanes = THREADS_PER_BLOCK / WORLD;

      constexpr int kChunkN =
          (TILE_N + kPackElems - 1) / kPackElems * kPackElems;
      constexpr int kPacks =
          (kChunkN + kPackElems * kLanes - 1) / (kPackElems * kLanes);
      static_assert(kLanes * WORLD == THREADS_PER_BLOCK,
                    "a block is a group of threads a rank");

      constexpr bool kWaves = kLanes % WAVE_SIZE == 0;
      constexpr bool kWhole = kPacks * kPackElems * kLanes == kChunkN;
      const int chunks = (slice + kChunkN - 1) / kChunkN;

      const int r =
          kWaves ? __builtin_amdgcn_readfirstlane(
                       (rank + static_cast<int>(threadIdx.x) / kLanes) % WORLD)
                 : (rank + static_cast<int>(threadIdx.x) / kLanes) % WORLD;
      for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
        DTYPE got[1][kPacks][kPackElems];

        recv<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, false,
             Protocol::simple, Layout::tile>(
            partials_ptr_array, partials_ptr_array_size, r, 0, 0, 0, 1, 1, 1,
            slice, 1, kWhole ? slice : min(slice, chunk * kChunkN + kChunkN),
            got, 0, chunk * kChunkN, 0);

        store<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, Protocol::simple,
              kOutput, TAIL>(
            got, &output_ptr, 1, 0, r, slice, output_stride_m, output_stride_n,
            WORLD, output_size_m, output_size_n, 1,
            kWhole ? share_len(r)
                   : min(share_len(r), chunk * kChunkN + kChunkN),
            0, chunk * kChunkN, 1);
      }
    }
  } else if constexpr (ALGORITHM == AllGatherAlgorithm::rows) {
    using Tile = std::remove_reference_t<OUTPUT>;
    constexpr int kRows = std::extent_v<Tile, 0>;
    constexpr int kPacks = std::extent_v<Tile, 1>;
    static_assert(std::extent_v<Tile, 2> == kPackElems,
                  "a tile's pack is the dtype's");
    (void)output_stride_m;
    (void)output_stride_n;

#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      const int owner = min((offs_m + i) / slice, WORLD - 1);
      if constexpr (!kInt4) {
#pragma unroll
        for (int k = 0; k < kPacks; ++k) {
          DTYPE pack[1][1][kPackElems];
          recv<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
               Protocol::simple, Layout::tile, TAIL>(
              partials_ptr_array, partials_ptr_array_size, owner, 0, 0,
              partials_stride_m, partials_stride_n, 1, partials_size_m,
              partials_size_n, output_size_m, output_size_n, pack, offs_m + i,
              offs_n + k * THREADS_PER_BLOCK * kPackElems, 1);
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) output[i][k][j] = pack[0][0][j];
        }
      } else {
        uint32_t raw[kPacks][1][1][2];
#pragma unroll
        for (int k = 0; k < kPacks; ++k)
          recv<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
               Protocol::simple, Layout::tile, TAIL>(
              partials_ptr_array, partials_ptr_array_size, owner, 0, 0,
              partials_stride_m, partials_stride_n, 1, partials_size_m,
              partials_size_n, output_size_m, output_size_n, raw[k], offs_m + i,
              offs_n + k * THREADS_PER_BLOCK * kPackElems, 1);
#pragma unroll
        for (int k = 0; k < kPacks; ++k) dequantize(raw[k][0][0], output[i][k]);
      }
    }
  } else {
    using Tile = std::remove_reference_t<OUTPUT>;
    constexpr int kRows = std::extent_v<Tile, 0>;
    constexpr int kPacks = std::extent_v<Tile, 1>;
    static_assert(std::extent_v<Tile, 2> == kPackElems,
                  "a tile's pack is the dtype's");
    (void)output_stride_m;
    (void)output_stride_n;

    const auto owner_of = [&](int k) {
      const int n = offs_n / kPackElems +
                    static_cast<int>(threadIdx.x) % THREADS_PER_BLOCK +
                    k * THREADS_PER_BLOCK;
      const int col =
          n < output_size_n / kPackElems ? n : output_size_n / kPackElems - 1;
      return min(col / (slice / kPackElems), WORLD - 1);
    };
    if constexpr (!kInt4) {
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        DTYPE pack[kRows][1][kPackElems];
        recv<WORLD, kRows, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
             Protocol::simple, Layout::tile, TAIL>(
            partials_ptr_array, partials_ptr_array_size, owner_of(k), 0, 0,
            partials_stride_m, partials_stride_n, 1, partials_size_m,
            partials_size_n, output_size_m, output_size_n, pack, offs_m,
            offs_n + k * THREADS_PER_BLOCK * kPackElems, 1);
#pragma unroll
        for (int i = 0; i < kRows; ++i)
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) output[i][k][j] = pack[i][0][j];
      }
    } else {
      uint32_t raw[kPacks][kRows][1][2];
#pragma unroll
      for (int k = 0; k < kPacks; ++k)
        recv<WORLD, kRows, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
             Protocol::simple, Layout::tile, TAIL>(
            partials_ptr_array, partials_ptr_array_size, owner_of(k), 0, 0,
            partials_stride_m, partials_stride_n, 1, partials_size_m,
            partials_size_n, output_size_m, output_size_n, raw[k], offs_m,
            offs_n + k * THREADS_PER_BLOCK * kPackElems, 1);
#pragma unroll
      for (int i = 0; i < kRows; ++i)
#pragma unroll
        for (int k = 0; k < kPacks; ++k) dequantize(raw[k][i][0], output[i][k]);
    }
  }
}

template <AllGatherAlgorithm ALGORITHM, Protocol PROTOCOL, bool QUANTIZE_INT4,
          typename DTYPE, int WORLD, int TILE_M, int THREADS_PER_BLOCK,
          int WAVE_SIZE, bool TAIL, typename ENTRY, int ROWS, int PACKS,
          int ELEMS>
DINLINE void all_gather(ENTRY (&entries)[WORLD], int64_t stride_m,
                        int64_t stride_n, int size_m, int size_n, int clip_n,
                        int offs_m, int offs_n,
                        DTYPE (&copies)[WORLD][ROWS][PACKS][ELEMS]) {
  static_assert(PROTOCOL == Protocol::simple, "this all-gather runs simple");
  static_assert(ALGORITHM == AllGatherAlgorithm::tile_pull,
                "a tile from where each rank published it");
#pragma unroll
  for (int r = 0; r < WORLD; ++r)
    recv<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
         Protocol::simple, Layout::tile, TAIL>(
        &entries[r], 1, 0, 0, 0, stride_m, stride_n, 1, size_m, size_n, size_m,
        clip_n, copies[r], offs_m, offs_n, 1);
}

// The elements a thread packs per access in a tile push. An LL line is half
// data, so exact LL packs no wider than a line's payload.
template <Protocol PROTOCOL, bool QUANTIZE_INT4, typename DTYPE, int PACK_BYTES>
constexpr int kTilePushPackElems =
    QUANTIZE_INT4 || PROTOCOL == Protocol::simple
        ? traits<DTYPE, PACK_BYTES>::N
        : (PACK_BYTES < kLinePayloadBytes ? PACK_BYTES : kLinePayloadBytes) /
              static_cast<int>(sizeof(DTYPE));

// One tile, pushed into every rank's inbox, then every rank's copy of it read
// out of this rank's. The protocol is how the inbox is written and how a
// reader knows a copy arrived.
//   simple: plain stores into every rank's staging buffer, which holds every
//     rank's whole input; a barrier says every copy is in, a second that every
//     rank has read it.
//   ll: lines of half data, half the row's epoch, into an inbox row for each
//     row the block owns; a reader polls each line until it carries the
//     epoch, so the data is its own signal. Each row's epoch is kept in this
//     rank's epochs buffer from tile to tile and launch to launch, and the
//     next tile reuses the inbox rows.
template <AllGatherAlgorithm ALGORITHM, Protocol PROTOCOL, bool QUANTIZE_INT4,
          typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVE_SIZE, bool TAIL, int ROWS, int PACKS,
          int ELEMS>
DINLINE void all_gather(
    const DTYPE (&mine)[ROWS][PACKS][ELEMS], char* const* staging_ptr_array,
    int staging_ptr_array_size, char* const* unfenced_ptr_array,
    int unfenced_ptr_array_size, int64_t unfenced_stride_slot,
    int64_t unfenced_stride_m, int64_t unfenced_stride_n,
    int unfenced_size_slot, int unfenced_size_m, int unfenced_size_n,
    int32_t* unfenced_epochs_ptr, int64_t unfenced_epochs_stride_block,
    int64_t unfenced_epochs_stride_n, int unfenced_epochs_size_block,
    int unfenced_epochs_size_n, uint32_t* const* signals_ptr_array,
    int signals_ptr_array_size, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    uint32_t* status_ptr, int64_t status_stride_field, int status_size_field,
    int rank, uint64_t timeout_ticks, int input_size_m, int input_size_n,
    int offs_m, int offs_n, int clip_m, int clip_n,
    DTYPE (&copies)[WORLD][ROWS][PACKS][ELEMS]) {
  static_assert(ALGORITHM == AllGatherAlgorithm::tile_push,
                "a tile pushed into every rank's inbox");
  if constexpr (PROTOCOL == Protocol::simple) {
    const int64_t slot = QUANTIZE_INT4
                             ? (int4_payload_bytes(input_size_m, input_size_n) +
                                kInt4ScalesAlign - 1) /
                                   kInt4ScalesAlign * kInt4ScalesAlign
                             : int64_t{input_size_m} * input_size_n;
#pragma unroll
    for (int r = 0; r < WORLD; ++r)
      send<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
           Protocol::simple, Layout::tile, TAIL>(
          mine, staging_ptr_array, staging_ptr_array_size, r, rank, slot,
          input_size_n, 1, WORLD, input_size_m, input_size_n, input_size_m,
          clip_n, offs_m, offs_n, 1);
    barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                   signals_stride_parity, signals_stride_block,
                   signals_stride_writer, signals_size_parity,
                   signals_size_block, signals_size_writer, status_ptr,
                   status_stride_field, status_size_field, rank, timeout_ticks,
                   Until::launched, 0);
    load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
         Protocol::simple, Layout::tile, TAIL>(
        staging_ptr_array, staging_ptr_array_size, rank, 0, slot, input_size_n,
        1, WORLD, input_size_m, input_size_n, input_size_m, clip_n, copies,
        offs_m, offs_n, 1);
    barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                   signals_stride_parity, signals_stride_block,
                   signals_stride_writer, signals_size_parity,
                   signals_size_block, signals_size_writer, status_ptr,
                   status_stride_field, status_size_field, rank, timeout_ticks,
                   Until::read, 1);
  } else if constexpr (PROTOCOL == Protocol::ll) {
    constexpr int kThreadsM = THREADS_PER_BLOCK % TILE_M == 0 ? TILE_M : 1;
    constexpr int kThreadsN = THREADS_PER_BLOCK / kThreadsM;
    const int first_row = static_cast<int>(blockIdx.x) * TILE_M;
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
        static_cast<int>(gridDim.x) * TILE_M > unfenced_size_m / 2)
      __builtin_trap();
    int32_t last[ROWS][1][kLineBytes / static_cast<int>(sizeof(int32_t))];
    load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Protocol::simple,
         Layout::tile>(&unfenced_epochs_ptr, 1, 0, 0, 0,
                       unfenced_epochs_stride_block, unfenced_epochs_stride_n,
                       1, unfenced_epochs_size_block, unfenced_epochs_size_n,
                       unfenced_epochs_size_block, unfenced_epochs_size_n, last,
                       first_row, 0, 1);
    uint32_t epochs[ROWS];
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
      const uint32_t before = kThreadsM == 1
                                  ? __builtin_amdgcn_readfirstlane(
                                        static_cast<uint32_t>(last[i][0][0]))
                                  : static_cast<uint32_t>(last[i][0][0]);
      const bool written =
          (kThreadsM == 1 ? i
                          : static_cast<int>(threadIdx.x) / kThreadsN +
                                i * kThreadsM) < clip_m;
      epochs[i] = !written ? before : before + 1u == 0u ? 2u : before + 1u;
    }
    const int lines_m = first_row + clip_m;
    const int lines_n = min(TILE_N, input_size_n - offs_n);
    // Quantized once, then sent to every rank.
    if constexpr (QUANTIZE_INT4) {
      uint32_t payload[ROWS][PACKS][2];
#pragma unroll
      for (int i = 0; i < ROWS; ++i)
#pragma unroll
        for (int k = 0; k < PACKS; ++k)
          quantize(reinterpret_cast<const DTYPE(&)[1][ELEMS]>(mine[i][k]),
                   reinterpret_cast<uint32_t (&)[1][2]>(payload[i][k]));
#pragma unroll
      for (int r = 0; r < WORLD; ++r)
        send<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, true, Protocol::ll,
             Layout::tile>(payload, unfenced_ptr_array, unfenced_ptr_array_size,
                           r, rank, unfenced_stride_slot, unfenced_stride_m,
                           unfenced_stride_n, unfenced_size_slot,
                           unfenced_size_m, unfenced_size_n, lines_m, lines_n,
                           first_row, 0, 1, epochs);
    } else {
#pragma unroll
      for (int r = 0; r < WORLD; ++r)
        send<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Protocol::ll,
             Layout::tile>(mine, unfenced_ptr_array, unfenced_ptr_array_size, r,
                           rank, unfenced_stride_slot, unfenced_stride_m,
                           unfenced_stride_n, unfenced_size_slot,
                           unfenced_size_m, unfenced_size_n, lines_m, lines_n,
                           first_row, 0, 1, epochs);
    }
    load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
         Protocol::ll, Layout::tile>(
        unfenced_ptr_array, unfenced_ptr_array_size, rank, 0,
        unfenced_stride_slot, unfenced_stride_m, unfenced_stride_n,
        unfenced_size_slot, unfenced_size_m, unfenced_size_n, lines_m, lines_n,
        copies, first_row, 0, 1, epochs, status_ptr, status_stride_field,
        status_size_field, timeout_ticks);
    int32_t next[ROWS][1][kLineBytes / static_cast<int>(sizeof(int32_t))] = {};
#pragma unroll
    for (int i = 0; i < ROWS; ++i)
      next[i][0][0] = static_cast<int32_t>(epochs[i]);
    store<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Protocol::simple,
          Layout::tile>(next, &unfenced_epochs_ptr, 1, 0, 0, 0,
                        unfenced_epochs_stride_block, unfenced_epochs_stride_n,
                        1, unfenced_epochs_size_block, unfenced_epochs_size_n,
                        unfenced_epochs_size_block, unfenced_epochs_size_n,
                        first_row, 0, 1);
    __syncthreads();
  }
}

}  // namespace hip_custom_communicator
