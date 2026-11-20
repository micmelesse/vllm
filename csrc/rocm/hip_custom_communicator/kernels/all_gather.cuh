#pragma once

// All-gather stages: every rank contributes a chunk, and every rank ends with
// every rank's chunk.

#include <type_traits>
#include <utility>

#include "common/dequantize.cuh"
#include "common/load.cuh"
#include "common/quantize.cuh"
#include "common/recv.cuh"
#include "common/send.cuh"
#include "common/store.cuh"
#include "common/types.cuh"

namespace hip_custom_communicator {

enum class AllGatherAlgorithm { tile };

// The elements a thread packs per access in a tile gather. An LL line is half
// data, so exact LL packs no wider than a line's payload; Lamport's is all
// data.
template <Protocol PROTOCOL, bool QUANTIZE_INT4, typename DTYPE, int PACK_BYTES>
constexpr int kTilePackElems =
    QUANTIZE_INT4 || PROTOCOL != Protocol::ll
        ? traits<DTYPE, PACK_BYTES>::N
        : (PACK_BYTES < kLinePayloadBytes ? PACK_BYTES : kLinePayloadBytes) /
              static_cast<int>(sizeof(DTYPE));

namespace internal {

// This rank's tile of the input, pushed into every rank's inbox, then every
// rank's copy of it read out of this rank's. The protocol is how the inbox is
// written and how a reader knows a copy arrived.
//   simple_write, simple_read: the Simple protocols' head and tail a block and
//     peer (COUNTERS, common/types.cuh).
//   ll, lamport: the data is its own signal, sent into the inbox rows the
//     block owns; send and recv keep the step in the block's connection
//     (CONNECTION, common/types.cuh) and pick the bank from it. LL's lines are
//     half data, half the step, and a reader polls each until it carries the
//     step. Lamport's are all data, and a reader polls each until no element is
//     the sentinel, then writes the sentinel back.
template <Protocol PROTOCOL, bool QUANTIZE_INT4, typename DTYPE, int WORLD,
          int TILE_M, int TILE_N, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE,
          bool TAIL, int ROWS, int PACKS, int ELEMS>
DINLINE void all_gather_tile(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    char* const* unfenced_ptr_array, int unfenced_ptr_array_size,
    int64_t unfenced_stride_slot, int64_t unfenced_stride_m,
    int64_t unfenced_stride_n, int unfenced_size_slot, int unfenced_size_m,
    int unfenced_size_n, Connection* connection,
    uint32_t* const* counters_ptr_array, int counters_ptr_array_size,
    int64_t counters_stride_kind, int64_t counters_stride_block,
    int64_t counters_stride_peer, int counters_size_block,
    int counters_size_peer, uint32_t* status_ptr, int64_t status_stride_field,
    int status_size_field, int rank, uint64_t timeout_ticks, int input_size_m,
    int input_size_n, int offs_m, int offs_n, int clip_m, int clip_n,
    DTYPE (&copies)[WORLD][ROWS][PACKS][ELEMS]) {
  static_assert(PROTOCOL == Protocol::ll || PROTOCOL == Protocol::lamport ||
                    PROTOCOL == Protocol::simple_write ||
                    PROTOCOL == Protocol::simple_read,
                "a tile gather is LL, Lamport, simple_write or simple_read");
  // The block's row of the inbox (CONNECTION).
  const int first_row = static_cast<int>(blockIdx.x) * TILE_M;
  const int lines_m = first_row + clip_m;
  const int lines_n = min(TILE_N, input_size_n - offs_n);
  DTYPE mine[ROWS][PACKS][ELEMS];
  load<WORLD, TILE_M, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
      input_ptr, 0, 0, input_stride_m, input_stride_n, 1, input_size_m,
      input_size_n, input_size_m, clip_n, mine, offs_m, offs_n, 1);
  // Quantized once, then sent to every rank.
  if constexpr (QUANTIZE_INT4) {
    uint32_t payload[ROWS][PACKS][2];
#pragma unroll
    for (int i = 0; i < ROWS; ++i)
#pragma unroll
      for (int k = 0; k < PACKS; ++k)
        quantize(reinterpret_cast<const DTYPE(&)[1][ELEMS]>(mine[i][k]),
                 reinterpret_cast<uint32_t (&)[1][2]>(payload[i][k]));
    send<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE, true, PROTOCOL,
         Layout::tile>(
        payload, unfenced_ptr_array, unfenced_ptr_array_size, 0, WORLD, rank,
        unfenced_stride_slot, unfenced_stride_m, unfenced_stride_n,
        unfenced_size_slot, unfenced_size_m, unfenced_size_n, lines_m, lines_n,
        first_row, 0, 1, connection, counters_ptr_array,
        counters_ptr_array_size, counters_stride_kind, counters_stride_block,
        counters_stride_peer, counters_size_block, counters_size_peer,
        status_ptr, status_stride_field, status_size_field, timeout_ticks);
  } else {
    send<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE, false, PROTOCOL,
         Layout::tile>(
        mine, unfenced_ptr_array, unfenced_ptr_array_size, 0, WORLD, rank,
        unfenced_stride_slot, unfenced_stride_m, unfenced_stride_n,
        unfenced_size_slot, unfenced_size_m, unfenced_size_n, lines_m, lines_n,
        first_row, 0, 1, connection, counters_ptr_array,
        counters_ptr_array_size, counters_stride_kind, counters_stride_block,
        counters_stride_peer, counters_size_block, counters_size_peer,
        status_ptr, status_stride_field, status_size_field, timeout_ticks);
  }
  recv<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE, QUANTIZE_INT4, PROTOCOL,
       Layout::tile>(
      unfenced_ptr_array, unfenced_ptr_array_size, rank, 0,
      unfenced_stride_slot, unfenced_stride_m, unfenced_stride_n,
      unfenced_size_slot, unfenced_size_m, unfenced_size_n, lines_m, lines_n,
      copies, first_row, 0, 1, connection, counters_ptr_array,
      counters_ptr_array_size, counters_stride_kind, counters_stride_block,
      counters_stride_peer, counters_size_block, counters_size_peer, status_ptr,
      status_stride_field, status_size_field, timeout_ticks);
}

}  // namespace internal

// Every rank contributes a chunk; every rank ends with every rank's chunk.
// tile gathers each rank's copy of one tile into output[WORLD], for the
// caller to sum, from this rank's input.
template <AllGatherAlgorithm ALGORITHM, Protocol PROTOCOL, bool QUANTIZE_INT4,
          typename DTYPE, int WORLD, int TILE_M, int TILE_N, int BLOCK_X,
          int BLOCK_Y, int WAVE_SIZE, int PACK_BYTES, bool CONTIGUOUS,
          bool TAIL, typename INPUT, typename OUTPUT>
DINLINE void all_gather(
    INPUT&& input, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, OUTPUT&& output,
    int64_t output_stride_m, int64_t output_stride_n, int output_size_m,
    int output_size_n, int offs_m, int offs_n, int clip_m, int clip_n,
    int slice, char* const* unfenced_ptr_array, int unfenced_ptr_array_size,
    int64_t unfenced_stride_slot, int64_t unfenced_stride_m,
    int64_t unfenced_stride_n, int unfenced_size_slot, int unfenced_size_m,
    int unfenced_size_n, Connection* connection,
    uint32_t* const* counters_ptr_array, int counters_ptr_array_size,
    int64_t counters_stride_kind, int64_t counters_stride_block,
    int64_t counters_stride_peer, int counters_size_block,
    int counters_size_peer, uint32_t* status_ptr, int64_t status_stride_field,
    int status_size_field, int rank, uint64_t timeout_ticks) {
  if constexpr (ALGORITHM == AllGatherAlgorithm::tile)
    internal::all_gather_tile<PROTOCOL, QUANTIZE_INT4, DTYPE, WORLD, TILE_M,
                              TILE_N, BLOCK_X, BLOCK_Y, WAVE_SIZE, TAIL>(
        input, input_stride_m, input_stride_n, unfenced_ptr_array,
        unfenced_ptr_array_size, unfenced_stride_slot, unfenced_stride_m,
        unfenced_stride_n, unfenced_size_slot, unfenced_size_m, unfenced_size_n,
        connection, counters_ptr_array, counters_ptr_array_size,
        counters_stride_kind, counters_stride_block, counters_stride_peer,
        counters_size_block, counters_size_peer, status_ptr,
        status_stride_field, status_size_field, rank, timeout_ticks,
        input_size_m, input_size_n, offs_m, offs_n, clip_m, clip_n, output);
}

}  // namespace hip_custom_communicator
