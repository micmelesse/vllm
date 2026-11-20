#pragma once

// All-reduce kernels, one function per algorithm.

#include "all_gather.cuh"
#include "common/load.cuh"
#include "common/recv.cuh"
#include "common/reduce.cuh"
#include "common/send.cuh"
#include "common/store.cuh"
#include "common/types.cuh"

namespace hip_custom_communicator {

enum class AllReduceAlgorithm {
  one_shot,
  two_shot,
};

template <Protocol PROTOCOL, bool QUANTIZE_INT4, typename DTYPE, int WORLD,
          int TILE_M, int TILE_N, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE,
          int PACK_BYTES, bool CONTIGUOUS, bool TAIL>
DINLINE void all_reduce_one_shot(const DTYPE* input_ptr, int64_t input_stride_m,
                                 int64_t input_stride_n, int input_size_m,
                                 int input_size_n, DTYPE* output_ptr,
                                 int64_t output_stride_m,
                                 int64_t output_stride_n,
                                 const ProtocolArgs& protocol_args, int rank) {
  constexpr int kThreadsM = BLOCK_X % TILE_M == 0 ? TILE_M : 1;
  constexpr int kThreadsN = BLOCK_X / kThreadsM;
  constexpr int kRows = TILE_M / kThreadsM;

  constexpr int kPackElems =
      kTilePackElems<PROTOCOL, QUANTIZE_INT4, DTYPE, PACK_BYTES>;

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
      protocol_args.connection_ptr, 0, 0, protocol_args.connection_stride_block,
      protocol_args.connection_stride_word, 1,
      protocol_args.connection_size_block, protocol_args.connection_size_word,
      protocol_args.connection_size_block, protocol_args.connection_size_word,
      connection, static_cast<int>(blockIdx.x), 0, 1);
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int offs_m = t / tiles_n * TILE_M, offs_n = t % tiles_n * kTileN;
    const int clip_m = min(TILE_M, input_size_m - offs_m);
    const int clip_n =
        kWhole ? input_size_n : min(input_size_n, offs_n + kTileN);

    DTYPE copies[WORLD][kRows][kPacks][kPackElems];
    all_gather<AllGatherAlgorithm::tile, PROTOCOL, QUANTIZE_INT4, DTYPE, WORLD,
               TILE_M, kTileN, BLOCK_X, BLOCK_Y, WAVE_SIZE, PACK_BYTES,
               CONTIGUOUS, TAIL>(input_ptr, input_stride_m, input_stride_n,
                                 input_size_m, input_size_n, copies, 0, 0, 0, 0,
                                 offs_m, offs_n, clip_m, clip_n, 0,
                                 protocol_args, &connection, rank);

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
      connection, protocol_args.connection_ptr, 0, 0,
      protocol_args.connection_stride_block,
      protocol_args.connection_stride_word, 1,
      protocol_args.connection_size_block, protocol_args.connection_size_word,
      protocol_args.connection_size_block, protocol_args.connection_size_word,
      static_cast<int>(blockIdx.x), 0, 1);
}

template <Protocol PROTOCOL, bool QUANTIZE_INT4, typename DTYPE, int WORLD,
          int TILE_M, int TILE_N, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE,
          int PACK_BYTES, bool CONTIGUOUS, bool TAIL>
DINLINE void all_reduce_two_shot(const DTYPE* input_ptr, int64_t input_stride_m,
                                 int64_t input_stride_n, int input_size_m,
                                 int input_size_n, DTYPE* output_ptr,
                                 int64_t output_stride_m,
                                 int64_t output_stride_n,
                                 const ProtocolArgs& protocol_args, int rank) {
  static_assert(
      PROTOCOL == Protocol::simple_write || PROTOCOL == Protocol::simple_read,
      "the two-shot is simple_write or simple_read");
  // A chunk is two rounds: the reduce-scatter (slice w's chunk to rank w,
  // this rank's chunk from every rank, summed) and the all-gather (the sum
  // to every rank, slice w's chunk from rank w). Each lane group talks to
  // one peer, w; the inbox row is the block's.
  if constexpr (QUANTIZE_INT4) {
    // INT4: a lane holds a group a round (the sum's shared memory holds no
    // more at a 512-thread block); the shares are whole groups.
    // The parts are quantized to send, the sum is taken in float from the
    // dequantized parts, and the sum is quantized again for the all-gather.
    constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
    static_assert(kPackElems * kInt4GroupPacks == kInt4Group,
                  "a group is kInt4GroupPacks packs");
    constexpr int kLanes = BLOCK_X / WORLD;
    static_assert(kLanes * WORLD == BLOCK_X,
                  "a block is a group of threads a rank");
    constexpr int kGroups = 1;
    constexpr int kChunkN = kGroups * kLanes * kInt4Group;
    constexpr int kK = kGroups * kInt4GroupPacks;
    constexpr Layout kTensor =
        CONTIGUOUS ? Layout::group_run : Layout::group_flat;
    const int64_t total = int64_t{input_size_m} * input_size_n;
    const int64_t units =
        TAIL ? (total + kInt4Group - 1) / kInt4Group : total / kInt4Group;
    const int slice =
        static_cast<int>((units + WORLD - 1) / WORLD * kInt4Group);
    const int chunks = (slice + kChunkN - 1) / kChunkN;
    const int w = (rank + static_cast<int>(threadIdx.x) / kLanes) % WORLD;
    const int row = static_cast<int>(blockIdx.x);
    const int share_w = static_cast<int>(
        max(int64_t{0}, min(int64_t{slice}, total - int64_t{w} * slice)));
    const int share_me = static_cast<int>(
        max(int64_t{0}, min(int64_t{slice}, total - int64_t{rank} * slice)));
    // The payload's scales follow every slot's and row's values.

    Connection connection;
    load<WORLD, 1, BLOCK_X, WAVE_SIZE, false, Layout::block>(
        protocol_args.connection_ptr, 0, 0,
        protocol_args.connection_stride_block,
        protocol_args.connection_stride_word, 1,
        protocol_args.connection_size_block, protocol_args.connection_size_word,
        protocol_args.connection_size_block, protocol_args.connection_size_word,
        connection, row, 0, 1);
    for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
      const int lo = chunk * kChunkN;
      const int end_w = min(share_w, lo + kChunkN);
      const int end_me = min(share_me, lo + kChunkN);
      // A recv reads at least a group, past the end when the share is
      // empty here: those values go nowhere.
      const int read_w = max(end_w - lo, kInt4Group);
      const int read_me = max(end_me - lo, kInt4Group);

      DTYPE part[1][kK][kPackElems] = {};
      if (end_w > lo)
        load<WORLD, WORLD, BLOCK_X, WAVE_SIZE, false, kTensor, TAIL>(
            input_ptr, w, slice, input_stride_m, input_stride_n, WORLD,
            input_size_m, input_size_n, input_size_m, end_w, part, 0, lo, 1);
      send<WORLD, WORLD, BLOCK_X, BLOCK_Y, WAVE_SIZE, true, PROTOCOL,
           Layout::group_run, TAIL>(part, protocol_args, w, 1, rank, row + 1,
                                    max(end_w - lo, 0), row, 0, 1, &connection);
      uint32_t got[1][1][kK][2];
      recv<WORLD, WORLD, BLOCK_X, BLOCK_Y, WAVE_SIZE, true, PROTOCOL,
           Layout::group_run, TAIL>(protocol_args, rank, w, row + 1, read_me,
                                    got, row, 0, 1, &connection);
      DTYPE parts[kK][kPackElems];
#pragma unroll
      for (int k = 0; k < kK; ++k) {
        const uint32_t pack[2] = {
            got[0][0][k][0],
            got[0][0][k / kInt4GroupPacks * kInt4GroupPacks][1]};
        dequantize(pack, parts[k]);
      }
      float sum[1][kK][kPackElems];
      reduce<WORLD, BLOCK_X, WAVE_SIZE, ReduceAxis::m_all>(parts, sum[0]);

      send<WORLD, WORLD, BLOCK_X, BLOCK_Y, WAVE_SIZE, true, PROTOCOL,
           Layout::group_run, TAIL>(sum, protocol_args, w, 1, rank, row + 1,
                                    max(end_me - lo, 0), row, 0, 1,
                                    &connection);
      uint32_t words[1][1][kK][2];
      recv<WORLD, WORLD, BLOCK_X, BLOCK_Y, WAVE_SIZE, true, PROTOCOL,
           Layout::group_run, TAIL>(protocol_args, rank, w, row + 1, read_w,
                                    words, row, 0, 1, &connection);
      DTYPE out[1][kK][kPackElems];
#pragma unroll
      for (int k = 0; k < kK; ++k) {
        const uint32_t pack[2] = {
            words[0][0][k][0],
            words[0][0][k / kInt4GroupPacks * kInt4GroupPacks][1]};
        dequantize(pack, out[0][k]);
      }
      store<WORLD, WORLD, BLOCK_X, WAVE_SIZE, false, kTensor, TAIL>(
          out, output_ptr, w, slice, output_stride_m, output_stride_n, WORLD,
          input_size_m, input_size_n, 1, end_w, 0, lo, 1);
    }
    store<WORLD, 1, BLOCK_X, WAVE_SIZE, false, Layout::block>(
        connection, protocol_args.connection_ptr, 0, 0,
        protocol_args.connection_stride_block,
        protocol_args.connection_stride_word, 1,
        protocol_args.connection_size_block, protocol_args.connection_size_word,
        protocol_args.connection_size_block, protocol_args.connection_size_word,
        row, 0, 1);
  } else {
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
        protocol_args.connection_ptr, 0, 0,
        protocol_args.connection_stride_block,
        protocol_args.connection_stride_word, 1,
        protocol_args.connection_size_block, protocol_args.connection_size_word,
        protocol_args.connection_size_block, protocol_args.connection_size_word,
        connection, row, 0, 1);
    for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
      const int lo = chunk * kChunkN;
      // A rank's share of the slices is slice long, the last one cut at the
      // end; a chunk ends at the share's end.
      const int end_w =
          min(static_cast<int>(max(
                  int64_t{0}, min(int64_t{slice}, total - int64_t{w} * slice))),
              lo + kChunkN);
      const int end_me = min(
          static_cast<int>(max(
              int64_t{0}, min(int64_t{slice}, total - int64_t{rank} * slice))),
          lo + kChunkN);
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
          part, protocol_args, w, 1, rank, row + 1, max(end_w - lo, 0), row, 0,
          1, &connection);
      DTYPE got[1][kPacks][kPackElems];
      recv<WORLD, 1, BLOCK_X, BLOCK_Y, WAVE_SIZE, false, PROTOCOL, Layout::run>(
          protocol_args, rank, w, row + 1, read_me, got, row, 0, 1,
          &connection);
      DTYPE sum[1][kPacks][kPackElems];
      reduce<WORLD, BLOCK_X, WAVE_SIZE, ReduceAxis::m_all>(got[0], sum[0]);

      send<WORLD, 1, BLOCK_X, BLOCK_Y, WAVE_SIZE, false, PROTOCOL, Layout::run>(
          sum, protocol_args, w, 1, rank, row + 1, max(end_me - lo, 0), row, 0,
          1, &connection);
      DTYPE out[1][kPacks][kPackElems];
      recv<WORLD, 1, BLOCK_X, BLOCK_Y, WAVE_SIZE, false, PROTOCOL, Layout::run>(
          protocol_args, rank, w, row + 1, read_w, out, row, 0, 1, &connection);
      store<WORLD, 1, BLOCK_X, WAVE_SIZE, false, kTensor, TAIL>(
          out, output_ptr, w, slice, output_stride_m, output_stride_n, WORLD,
          input_size_m, input_size_n, 1, end_w, 0, lo, 1);
    }
    store<WORLD, 1, BLOCK_X, WAVE_SIZE, false, Layout::block>(
        connection, protocol_args.connection_ptr, 0, 0,
        protocol_args.connection_stride_block,
        protocol_args.connection_stride_word, 1,
        protocol_args.connection_size_block, protocol_args.connection_size_word,
        protocol_args.connection_size_block, protocol_args.connection_size_word,
        row, 0, 1);
  }
}

template <AllReduceAlgorithm ALGORITHM, Protocol PROTOCOL, bool QUANTIZE_INT4,
          typename DTYPE, int WORLD, int TILE_M, int TILE_N, int BLOCK_X,
          int BLOCK_Y, int WAVE_SIZE, int PACK_BYTES, int WAVES_PER_EU,
          bool CONTIGUOUS, bool TAIL>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y, WAVES_PER_EU)
    all_reduce(const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
               int64_t input_stride_n, int input_size_m, int input_size_n,
               DTYPE* __restrict__ output_ptr, int64_t output_stride_m,
               int64_t output_stride_n, int output_size_m, int output_size_n,
               const ProtocolArgs protocol_args, int rank) {
  if constexpr (ALGORITHM == AllReduceAlgorithm::one_shot)
    all_reduce_one_shot<PROTOCOL, QUANTIZE_INT4, DTYPE, WORLD, TILE_M, TILE_N,
                        BLOCK_X, BLOCK_Y, WAVE_SIZE, PACK_BYTES, CONTIGUOUS,
                        TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        output_ptr, output_stride_m, output_stride_n, protocol_args, rank);
  else if constexpr (ALGORITHM == AllReduceAlgorithm::two_shot)
    all_reduce_two_shot<PROTOCOL, QUANTIZE_INT4, DTYPE, WORLD, TILE_M, TILE_N,
                        BLOCK_X, BLOCK_Y, WAVE_SIZE, PACK_BYTES, CONTIGUOUS,
                        TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        output_ptr, output_stride_m, output_stride_n, protocol_args, rank);
}

}  // namespace hip_custom_communicator
