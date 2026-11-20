#pragma once

// All-reduce kernels, one function per algorithm.

#include "all_gather.cuh"
#include "common/address.cuh"
#include "common/barrier.cuh"
#include "common/load.cuh"
#include "common/recv.cuh"
#include "common/reduce.cuh"
#include "common/send.cuh"
#include "common/store.cuh"
#include "common/types.cuh"
#include "reduce_scatter.cuh"

namespace hip_custom_communicator {

enum class AllReduceAlgorithm {
  one_shot_pull,
  one_shot_push,
  two_shot_pull,
  two_shot_push,
};

template <Protocol PROTOCOL, bool QUANTIZE_INT4, typename DTYPE, int WORLD,
          int TILE_M, int TILE_N, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE,
          int PACK_BYTES, bool CONTIGUOUS, bool TAIL>
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
  constexpr int kThreadsM = BLOCK_X % TILE_M == 0 ? TILE_M : 1;
  constexpr int kThreadsN = BLOCK_X / kThreadsM;
  constexpr int kRows = TILE_M / kThreadsM;
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;

  constexpr int kUnit = QUANTIZE_INT4 ? kInt4Group : kPackElems;
  constexpr int kTileN = (TILE_N + kUnit - 1) / kUnit * kUnit;
  constexpr int kPacks =
      (kTileN + kPackElems * kThreadsN - 1) / (kPackElems * kThreadsN);
  constexpr bool kWhole = kPacks * kPackElems * kThreadsN == kTileN;
  const int tiles_n = (input_size_n + kTileN - 1) / kTileN;
  const int tiles = (input_size_m + TILE_M - 1) / TILE_M * tiles_n;

  if constexpr (QUANTIZE_INT4) {
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      const int offs_m = t / tiles_n * TILE_M, offs_n = t % tiles_n * kTileN;
      const int clip_n =
          kWhole ? input_size_n : min(input_size_n, offs_n + kTileN);
      DTYPE mine[kRows][kPacks][kPackElems];
      load<WORLD, TILE_M, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
          input_ptr, 0, 0, input_stride_m, input_stride_n, 1, input_size_m,
          input_size_n, input_size_m, clip_n, mine, offs_m, offs_n, 1);
      send<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE, true, PROTOCOL,
           Layout::tile, TAIL>(mine, staging_ptr_array, staging_ptr_array_size,
                               rank, 1, 0, 0, input_size_n, 1, 1, input_size_m,
                               input_size_n, input_size_m, clip_n, offs_m,
                               offs_n, 1);
    }
  }

  std::conditional_t<QUANTIZE_INT4, char*, const DTYPE*> entries[WORLD];
#pragma unroll
  for (int r = 0; r < WORLD; ++r) {
    if constexpr (QUANTIZE_INT4)
      entries[r] = address(staging_ptr_array, staging_ptr_array_size, r);
    else
      entries[r] = address(input_ptr_array, input_ptr_array_size, r);
  }

  constexpr Until kFirst = QUANTIZE_INT4 ? Until::launched : Until::started;
  barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                 signals_stride_parity, signals_stride_block,
                 signals_stride_writer, signals_size_parity, signals_size_block,
                 signals_size_writer, status_ptr, status_stride_field,
                 status_size_field, rank, timeout_ticks, kFirst, 0);

  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int offs_m = t / tiles_n * TILE_M, offs_n = t % tiles_n * kTileN;
    const int clip_n =
        kWhole ? input_size_n : min(input_size_n, offs_n + kTileN);
    DTYPE copies[WORLD][kRows][kPacks][kPackElems];
    all_gather<AllGatherAlgorithm::tile_pull, PROTOCOL, QUANTIZE_INT4, DTYPE,
               WORLD, TILE_M, TILE_N, BLOCK_X, BLOCK_Y, WAVE_SIZE, PACK_BYTES,
               false, TAIL>(
        entries, 0,
        QUANTIZE_INT4 ? int64_t{input_size_n} : input_ptr_array_stride_m,
        QUANTIZE_INT4 ? int64_t{1} : input_ptr_array_stride_n, input_size_m,
        input_size_n, copies, 0, 0, 0, 0, offs_m, offs_n, 0, clip_n, 0, nullptr,
        0, nullptr, 0, 0, 0, 0, 0, 0, 0, nullptr, nullptr, 0, 0, 0, 0, 0, 0,
        nullptr, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 0, rank, 0);
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
    store<WORLD, TILE_M, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
        rounded, output_ptr, 0, 0, output_stride_m, output_stride_n, 1,
        input_size_m, input_size_n, input_size_m, clip_n, offs_m, offs_n, 1);
  }

  barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                 signals_stride_parity, signals_stride_block,
                 signals_stride_writer, signals_size_parity, signals_size_block,
                 signals_size_writer, status_ptr, status_stride_field,
                 status_size_field, rank, timeout_ticks, Until::read, 1);
}

template <Protocol PROTOCOL, bool QUANTIZE_INT4, typename DTYPE, int WORLD,
          int TILE_M, int TILE_N, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE,
          int PACK_BYTES, bool CONTIGUOUS, bool TAIL>
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
  reduce_scatter<ReduceScatterAlgorithm::flat_pull, PROTOCOL, QUANTIZE_INT4,
                 WORLD, TILE_M, TILE_N, BLOCK_X, BLOCK_Y, WAVE_SIZE, PACK_BYTES,
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
  all_gather<AllGatherAlgorithm::flat_pull, PROTOCOL, QUANTIZE_INT4, DTYPE,
             WORLD, TILE_M, TILE_N, BLOCK_X, BLOCK_Y, WAVE_SIZE, PACK_BYTES,
             CONTIGUOUS, TAIL>(
      partials_ptr_array, partials_ptr_array_size, partials_stride_m,
      partials_stride_n, partials_size_m, partials_size_n, output_ptr,
      output_stride_m, output_stride_n, input_size_m, input_size_n, 0, 0, 0, 0,
      0, nullptr, 0, nullptr, 0, 0, 0, 0, 0, 0, 0, nullptr, nullptr, 0, 0, 0, 0,
      0, 0, nullptr, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 0, rank, 0);
}

template <Protocol PROTOCOL, bool QUANTIZE_INT4, typename DTYPE, int WORLD,
          int TILE_M, int TILE_N, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE,
          int PACK_BYTES, bool CONTIGUOUS, bool TAIL>
DINLINE void all_reduce_one_shot_push(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, DTYPE* output_ptr,
    int64_t output_stride_m, int64_t output_stride_n,
    char* const* staging_ptr_array, int staging_ptr_array_size,
    char* const* unfenced_ptr_array, int unfenced_ptr_array_size,
    int64_t unfenced_stride_slot, int64_t unfenced_stride_m,
    int64_t unfenced_stride_n, int unfenced_size_slot, int unfenced_size_m,
    int unfenced_size_n, int32_t* connection_ptr,
    int64_t connection_stride_block, int64_t connection_stride_word,
    int connection_size_block, int connection_size_word,
    uint32_t* const* counters_ptr_array, int counters_ptr_array_size,
    int64_t counters_stride_kind, int64_t counters_stride_block,
    int64_t counters_stride_peer, int counters_size_block,
    int counters_size_peer, uint32_t* const* signals_ptr_array,
    int signals_ptr_array_size, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    uint32_t* status_ptr, int64_t status_stride_field, int status_size_field,
    int rank, uint64_t timeout_ticks) {
  constexpr int kThreadsM = BLOCK_X % TILE_M == 0 ? TILE_M : 1;
  constexpr int kThreadsN = BLOCK_X / kThreadsM;
  constexpr int kRows = TILE_M / kThreadsM;

  constexpr int kPackElems =
      kTilePushPackElems<PROTOCOL, QUANTIZE_INT4, DTYPE, PACK_BYTES>;

  constexpr int kUnit = QUANTIZE_INT4 ? kInt4Group : kPackElems;
  constexpr int kTileN = (TILE_N + kUnit - 1) / kUnit * kUnit;
  constexpr int kPacks =
      (kTileN + kPackElems * kThreadsN - 1) / (kPackElems * kThreadsN);
  constexpr bool kWhole = kPacks * kPackElems * kThreadsN == kTileN;
  const int tiles_n = (input_size_n + kTileN - 1) / kTileN;
  const int tiles = (input_size_m + TILE_M - 1) / TILE_M * tiles_n;

  // The block's connection, a line of its own: loaded before the first tile,
  // stored after the last (CONNECTION, common/types.cuh).
  Connection connection;
  load<WORLD, 1, BLOCK_X, WAVE_SIZE, false, Layout::block>(
      connection_ptr, 0, 0, connection_stride_block, connection_stride_word, 1,
      connection_size_block, connection_size_word, connection_size_block,
      connection_size_word, connection, static_cast<int>(blockIdx.x), 0, 1);
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int offs_m = t / tiles_n * TILE_M, offs_n = t % tiles_n * kTileN;
    const int clip_m = min(TILE_M, input_size_m - offs_m);
    const int clip_n =
        kWhole ? input_size_n : min(input_size_n, offs_n + kTileN);

    DTYPE copies[WORLD][kRows][kPacks][kPackElems];
    all_gather<AllGatherAlgorithm::tile_push, PROTOCOL, QUANTIZE_INT4, DTYPE,
               WORLD, TILE_M, kTileN, BLOCK_X, BLOCK_Y, WAVE_SIZE, PACK_BYTES,
               CONTIGUOUS, TAIL>(
        input_ptr, 0, input_stride_m, input_stride_n, input_size_m,
        input_size_n, copies, 0, 0, 0, 0, offs_m, offs_n, clip_m, clip_n, 0,
        staging_ptr_array, staging_ptr_array_size, unfenced_ptr_array,
        unfenced_ptr_array_size, unfenced_stride_slot, unfenced_stride_m,
        unfenced_stride_n, unfenced_size_slot, unfenced_size_m, unfenced_size_n,
        &connection, counters_ptr_array, counters_ptr_array_size,
        counters_stride_kind, counters_stride_block, counters_stride_peer,
        counters_size_block, counters_size_peer, signals_ptr_array,
        signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks);

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
    store<WORLD, TILE_M, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
        rounded, output_ptr, 0, 0, output_stride_m, output_stride_n, 1,
        input_size_m, input_size_n, input_size_m, clip_n, offs_m, offs_n, 1);
  }
  store<WORLD, 1, BLOCK_X, WAVE_SIZE, false, Layout::block>(
      connection, connection_ptr, 0, 0, connection_stride_block,
      connection_stride_word, 1, connection_size_block, connection_size_word,
      connection_size_block, connection_size_word, static_cast<int>(blockIdx.x),
      0, 1);
}

template <Protocol PROTOCOL, bool QUANTIZE_INT4, typename DTYPE, int WORLD,
          int TILE_M, int TILE_N, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE,
          int PACK_BYTES, bool CONTIGUOUS, bool TAIL>
DINLINE void all_reduce_two_shot_push(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, DTYPE* output_ptr,
    int64_t output_stride_m, int64_t output_stride_n,
    char* const* partials_ptr_array, int partials_ptr_array_size,
    int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
    int partials_size_n, char* const* staging_ptr_array,
    int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    char* const* unfenced_ptr_array, int unfenced_ptr_array_size,
    int64_t unfenced_stride_slot, int64_t unfenced_stride_m,
    int64_t unfenced_stride_n, int unfenced_size_slot, int unfenced_size_m,
    int unfenced_size_n, int32_t* connection_ptr,
    int64_t connection_stride_block, int64_t connection_stride_word,
    int connection_size_block, int connection_size_word,
    uint32_t* const* counters_ptr_array, int counters_ptr_array_size,
    int64_t counters_stride_kind, int64_t counters_stride_block,
    int64_t counters_stride_peer, int counters_size_block,
    int counters_size_peer, uint32_t* const* signals_ptr_array,
    int signals_ptr_array_size, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    uint32_t* status_ptr, int64_t status_stride_field, int status_size_field,
    int rank, uint64_t timeout_ticks) {
  if constexpr (PROTOCOL == Protocol::simple_write ||
                PROTOCOL == Protocol::simple_read) {
    // A chunk is two rounds: the reduce-scatter (slice w's chunk to rank w,
    // this rank's chunk from every rank, summed) and the all-gather (the sum
    // to every rank, slice w's chunk from rank w). Each lane group talks to
    // one peer, w; the inbox row is the block's.
    static_assert(!QUANTIZE_INT4, "the Simple protocols' two-shot is exact");
    constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
    const int64_t total = int64_t{input_size_m} * input_size_n;
    const int64_t units =
        TAIL ? (total + kPackElems - 1) / kPackElems : total / kPackElems;
    const int slice =
        static_cast<int>((units + WORLD - 1) / WORLD * kPackElems);
    constexpr Layout kTensor = CONTIGUOUS ? Layout::run : Layout::flat;
    constexpr int kLanes = BLOCK_X / WORLD;
    static_assert(kLanes * WORLD == BLOCK_X,
                  "a block is a group of threads a rank");
    constexpr int kChunkN = (TILE_N + kPackElems - 1) / kPackElems * kPackElems;
    constexpr int kPacks =
        (kChunkN + kPackElems * kLanes - 1) / (kPackElems * kLanes);
    const int chunks = (slice + kChunkN - 1) / kChunkN;
    const int w = (rank + static_cast<int>(threadIdx.x) / kLanes) % WORLD;
    const int row = static_cast<int>(blockIdx.x);

    // The block's connection, a line of its own: loaded before the first
    // chunk, stored after the last (CONNECTION, common/types.cuh).
    Connection connection;
    load<WORLD, 1, BLOCK_X, WAVE_SIZE, false, Layout::block>(
        connection_ptr, 0, 0, connection_stride_block, connection_stride_word,
        1, connection_size_block, connection_size_word, connection_size_block,
        connection_size_word, connection, row, 0, 1);
    for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
      const int lo = chunk * kChunkN;
      // A rank's share of the slices: slice long, the last one cut at the end.
      const auto share = [&](int r) {
        return static_cast<int>(
            max(int64_t{0}, min(int64_t{slice}, total - int64_t{r} * slice)));
      };
      const int end_w = min(share(w), lo + kChunkN);
      const int end_me = min(share(rank), lo + kChunkN);
      // A recv reads at least a pack, past the end when the share is empty
      // here: those values go nowhere.
      const int read_w = max(end_w - lo, kPackElems);
      const int read_me = max(end_me - lo, kPackElems);

      DTYPE part[1][kPacks][kPackElems] = {};
      if (end_w > lo)
        load<WORLD, 1, BLOCK_X, WAVE_SIZE, false, kTensor, TAIL>(
            input_ptr, w, slice, input_stride_m, input_stride_n, WORLD,
            input_size_m, input_size_n, input_size_m, end_w, part, 0, lo, 1);
      send<WORLD, 1, BLOCK_X, BLOCK_Y, WAVE_SIZE, false, PROTOCOL, Layout::run>(
          part, unfenced_ptr_array, unfenced_ptr_array_size, w, 1, rank,
          unfenced_stride_slot, unfenced_stride_m, unfenced_stride_n,
          unfenced_size_slot, unfenced_size_m, unfenced_size_n, row + 1,
          max(end_w - lo, 0), row, 0, 1, &connection, counters_ptr_array,
          counters_ptr_array_size, counters_stride_kind, counters_stride_block,
          counters_stride_peer, counters_size_block, counters_size_peer,
          status_ptr, status_stride_field, status_size_field, timeout_ticks);
      DTYPE got[1][kPacks][kPackElems];
      recv<WORLD, 1, BLOCK_X, BLOCK_Y, WAVE_SIZE, false, PROTOCOL, Layout::run>(
          unfenced_ptr_array, unfenced_ptr_array_size, rank, w,
          unfenced_stride_slot, unfenced_stride_m, unfenced_stride_n,
          unfenced_size_slot, unfenced_size_m, unfenced_size_n, row + 1,
          read_me, got, row, 0, 1, &connection, counters_ptr_array,
          counters_ptr_array_size, counters_stride_kind, counters_stride_block,
          counters_stride_peer, counters_size_block, counters_size_peer,
          status_ptr, status_stride_field, status_size_field, timeout_ticks);
      DTYPE sum[1][kPacks][kPackElems];
      reduce<WORLD, BLOCK_X, WAVE_SIZE, ReduceAxis::m_all>(got[0], sum[0]);

      send<WORLD, 1, BLOCK_X, BLOCK_Y, WAVE_SIZE, false, PROTOCOL, Layout::run>(
          sum, unfenced_ptr_array, unfenced_ptr_array_size, w, 1, rank,
          unfenced_stride_slot, unfenced_stride_m, unfenced_stride_n,
          unfenced_size_slot, unfenced_size_m, unfenced_size_n, row + 1,
          max(end_me - lo, 0), row, 0, 1, &connection, counters_ptr_array,
          counters_ptr_array_size, counters_stride_kind, counters_stride_block,
          counters_stride_peer, counters_size_block, counters_size_peer,
          status_ptr, status_stride_field, status_size_field, timeout_ticks);
      DTYPE out[1][kPacks][kPackElems];
      recv<WORLD, 1, BLOCK_X, BLOCK_Y, WAVE_SIZE, false, PROTOCOL, Layout::run>(
          unfenced_ptr_array, unfenced_ptr_array_size, rank, w,
          unfenced_stride_slot, unfenced_stride_m, unfenced_stride_n,
          unfenced_size_slot, unfenced_size_m, unfenced_size_n, row + 1, read_w,
          out, row, 0, 1, &connection, counters_ptr_array,
          counters_ptr_array_size, counters_stride_kind, counters_stride_block,
          counters_stride_peer, counters_size_block, counters_size_peer,
          status_ptr, status_stride_field, status_size_field, timeout_ticks);
      store<WORLD, 1, BLOCK_X, WAVE_SIZE, false, kTensor, TAIL>(
          out, output_ptr, w, slice, output_stride_m, output_stride_n, WORLD,
          input_size_m, input_size_n, 1, end_w, 0, lo, 1);
    }
    store<WORLD, 1, BLOCK_X, WAVE_SIZE, false, Layout::block>(
        connection, connection_ptr, 0, 0, connection_stride_block,
        connection_stride_word, 1, connection_size_block, connection_size_word,
        connection_size_block, connection_size_word, row, 0, 1);
  } else {
    reduce_scatter<ReduceScatterAlgorithm::flat_push, PROTOCOL, QUANTIZE_INT4,
                   WORLD, TILE_M, TILE_N, BLOCK_X, BLOCK_Y, WAVE_SIZE,
                   PACK_BYTES, CONTIGUOUS, TAIL>(
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
    all_gather<AllGatherAlgorithm::flat_pull, PROTOCOL, QUANTIZE_INT4, DTYPE,
               WORLD, TILE_M, TILE_N, BLOCK_X, BLOCK_Y, WAVE_SIZE, PACK_BYTES,
               CONTIGUOUS, TAIL>(
        partials_ptr_array, partials_ptr_array_size, partials_stride_m,
        partials_stride_n, partials_size_m, partials_size_n, output_ptr,
        output_stride_m, output_stride_n, input_size_m, input_size_n, 0, 0, 0,
        0, 0, nullptr, 0, nullptr, 0, 0, 0, 0, 0, 0, 0, nullptr, nullptr, 0, 0,
        0, 0, 0, 0, nullptr, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 0, rank, 0);
  }
}

template <AllReduceAlgorithm ALGORITHM, Protocol PROTOCOL, bool QUANTIZE_INT4,
          typename DTYPE, int WORLD, int TILE_M, int TILE_N, int BLOCK_X,
          int BLOCK_Y, int WAVE_SIZE, int PACK_BYTES, int WAVES_PER_EU,
          bool CONTIGUOUS, bool TAIL>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y, WAVES_PER_EU) all_reduce(
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
    int unfenced_size_n, int32_t* __restrict__ connection_ptr,
    int64_t connection_stride_block, int64_t connection_stride_word,
    int connection_size_block, int connection_size_word,
    uint32_t* const* __restrict__ counters_ptr_array,
    int counters_ptr_array_size, int64_t counters_stride_kind,
    int64_t counters_stride_block, int64_t counters_stride_peer,
    int counters_size_block, int counters_size_peer,
    uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size,
    int64_t signals_stride_parity, int64_t signals_stride_block,
    int64_t signals_stride_writer, int signals_size_parity,
    int signals_size_block, int signals_size_writer,
    uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
    int status_size_field, int rank, uint64_t timeout_ticks) {
  if constexpr (ALGORITHM == AllReduceAlgorithm::one_shot_pull)
    all_reduce_one_shot_pull<PROTOCOL, QUANTIZE_INT4, DTYPE, WORLD, TILE_M,
                             TILE_N, BLOCK_X, BLOCK_Y, WAVE_SIZE, PACK_BYTES,
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
    all_reduce_two_shot_pull<PROTOCOL, QUANTIZE_INT4, DTYPE, WORLD, TILE_M,
                             TILE_N, BLOCK_X, BLOCK_Y, WAVE_SIZE, PACK_BYTES,
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
    all_reduce_one_shot_push<PROTOCOL, QUANTIZE_INT4, DTYPE, WORLD, TILE_M,
                             TILE_N, BLOCK_X, BLOCK_Y, WAVE_SIZE, PACK_BYTES,
                             CONTIGUOUS, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        output_ptr, output_stride_m, output_stride_n, staging_ptr_array,
        staging_ptr_array_size, unfenced_ptr_array, unfenced_ptr_array_size,
        unfenced_stride_slot, unfenced_stride_m, unfenced_stride_n,
        unfenced_size_slot, unfenced_size_m, unfenced_size_n, connection_ptr,
        connection_stride_block, connection_stride_word, connection_size_block,
        connection_size_word, counters_ptr_array, counters_ptr_array_size,
        counters_stride_kind, counters_stride_block, counters_stride_peer,
        counters_size_block, counters_size_peer, signals_ptr_array,
        signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks);
  else if constexpr (ALGORITHM == AllReduceAlgorithm::two_shot_push)
    all_reduce_two_shot_push<PROTOCOL, QUANTIZE_INT4, DTYPE, WORLD, TILE_M,
                             TILE_N, BLOCK_X, BLOCK_Y, WAVE_SIZE, PACK_BYTES,
                             CONTIGUOUS, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        output_ptr, output_stride_m, output_stride_n, partials_ptr_array,
        partials_ptr_array_size, partials_stride_m, partials_stride_n,
        partials_size_m, partials_size_n, staging_ptr_array,
        staging_ptr_array_size, staging_stride_m, staging_stride_n,
        staging_size_m, staging_size_n, unfenced_ptr_array,
        unfenced_ptr_array_size, unfenced_stride_slot, unfenced_stride_m,
        unfenced_stride_n, unfenced_size_slot, unfenced_size_m, unfenced_size_n,
        connection_ptr, connection_stride_block, connection_stride_word,
        connection_size_block, connection_size_word, counters_ptr_array,
        counters_ptr_array_size, counters_stride_kind, counters_stride_block,
        counters_stride_peer, counters_size_block, counters_size_peer,
        signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
        signals_stride_block, signals_stride_writer, signals_size_parity,
        signals_size_block, signals_size_writer, status_ptr,
        status_stride_field, status_size_field, rank, timeout_ticks);
}

}  // namespace hip_custom_communicator
