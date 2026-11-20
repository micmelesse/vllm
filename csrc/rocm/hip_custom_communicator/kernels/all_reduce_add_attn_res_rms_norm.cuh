#pragma once

// All-reduce fused with AttnRes and its RMSNorm.

#include "all_gather.cuh"
#include "attn_res.cuh"
#include "common/address.cuh"
#include "common/load.cuh"
#include "common/recv.cuh"
#include "common/send.cuh"
#include "common/store.cuh"
#include "common/types.cuh"

namespace hip_custom_communicator {

enum class AllReduceAddAttnResRmsNormAlgorithm {
  one_shot_pull,
  two_shot_pull_columns
};

template <int TILE_K, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE, typename DTYPE,
          int ROWS, int K, int P>
DINLINE void add_attn_res_rms_norm_rows(
    int offs_m, const DTYPE (&sum)[ROWS][K][P], int input_size_m,
    int input_size_n, const DTYPE* prefix_ptr, int64_t prefix_stride_m,
    int64_t prefix_stride_n, const DTYPE* blocks_ptr, int64_t blocks_stride_m,
    int64_t blocks_stride_r, int64_t blocks_stride_n,
    const DTYPE* norm_weight_ptr, int64_t norm_weight_stride_n,
    const DTYPE* qk_weight_ptr, int64_t qk_weight_stride_n,
    const DTYPE* output_norm_weight_ptr, int64_t output_norm_weight_stride_n,
    DTYPE* prefix_output_ptr, int64_t prefix_output_stride_m,
    int64_t prefix_output_stride_n, DTYPE* output_ptr, int64_t output_stride_m,
    int64_t output_stride_n, DTYPE* block_output_ptr,
    int64_t block_output_stride_m, int64_t block_output_stride_n,
    int num_blocks, float eps, float output_eps) {
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);
  attn_res<TILE_K, BLOCK_X, BLOCK_Y, WAVE_SIZE, Layout::tile, false>(
      sum, prefix_ptr, prefix_stride_m, prefix_stride_n, blocks_ptr,
      blocks_stride_m, blocks_stride_r, blocks_stride_n, norm_weight_ptr,
      norm_weight_stride_n, qk_weight_ptr, qk_weight_stride_n,
      output_norm_weight_ptr, output_norm_weight_stride_n, prefix_output_ptr,
      prefix_output_stride_m, prefix_output_stride_n, output_ptr,
      output_stride_m, output_stride_n, block_output_ptr, block_output_stride_m,
      block_output_stride_n, input_size_m, input_size_n, offs_m, 0, num_blocks,
      eps, output_eps, inv_hidden);
}

template <typename DTYPE, int TILE_M, int TILE_N, int TILE_K, int BLOCK_X,
          int BLOCK_Y, int WAVE_SIZE, int PACK_BYTES, bool CHUNKED>
DINLINE void add_attn_res_rms_norm_rows(
    int offs_m, const DTYPE* input_ptr, int64_t input_stride_m,
    int64_t input_stride_n, int input_size_m, int input_size_n,
    const DTYPE* prefix_ptr, int64_t prefix_stride_m, int64_t prefix_stride_n,
    const DTYPE* blocks_ptr, int64_t blocks_stride_m, int64_t blocks_stride_r,
    int64_t blocks_stride_n, const DTYPE* norm_weight_ptr,
    int64_t norm_weight_stride_n, const DTYPE* qk_weight_ptr,
    int64_t qk_weight_stride_n, const DTYPE* output_norm_weight_ptr,
    int64_t output_norm_weight_stride_n, DTYPE* prefix_output_ptr,
    int64_t prefix_output_stride_m, int64_t prefix_output_stride_n,
    DTYPE* output_ptr, int64_t output_stride_m, int64_t output_stride_n,
    DTYPE* block_output_ptr, int64_t block_output_stride_m,
    int64_t block_output_stride_n, int num_blocks, float eps,
    float output_eps) {
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
  constexpr int kPacks =
      (TILE_N + kPackElems * BLOCK_X - 1) / (kPackElems * BLOCK_X);
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);
  attn_res<TILE_K, BLOCK_X, BLOCK_Y, WAVE_SIZE, Layout::tile, false, TILE_M,
           kPacks, kPackElems, CHUNKED>(
      input_ptr, input_stride_m, input_stride_n, prefix_ptr, prefix_stride_m,
      prefix_stride_n, blocks_ptr, blocks_stride_m, blocks_stride_r,
      blocks_stride_n, norm_weight_ptr, norm_weight_stride_n, qk_weight_ptr,
      qk_weight_stride_n, output_norm_weight_ptr, output_norm_weight_stride_n,
      prefix_output_ptr, prefix_output_stride_m, prefix_output_stride_n,
      output_ptr, output_stride_m, output_stride_n, block_output_ptr,
      block_output_stride_m, block_output_stride_n, input_size_m, input_size_n,
      offs_m, 0, num_blocks, eps, output_eps, inv_hidden);
}

template <typename DTYPE, int TILE_M, int TILE_N, int TILE_K, int BLOCK_X,
          int BLOCK_Y, int WAVE_SIZE, int PACK_BYTES, int WAVES_PER_EU,
          bool CHUNKED>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y, WAVES_PER_EU)
    add_attn_res_rms_norm(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const DTYPE* __restrict__ prefix_ptr, int64_t prefix_stride_m,
        int64_t prefix_stride_n, int prefix_size_m, int prefix_size_n,
        const DTYPE* __restrict__ blocks_ptr, int64_t blocks_stride_m,
        int64_t blocks_stride_r, int64_t blocks_stride_n, int blocks_size_m,
        int blocks_size_r, int blocks_size_n,
        const DTYPE* __restrict__ norm_weight_ptr, int64_t norm_weight_stride_n,
        int norm_weight_size_n, const DTYPE* __restrict__ qk_weight_ptr,
        int64_t qk_weight_stride_n, int qk_weight_size_n,
        const DTYPE* __restrict__ output_norm_weight_ptr,
        int64_t output_norm_weight_stride_n, int output_norm_weight_size_n,
        DTYPE* __restrict__ prefix_output_ptr, int64_t prefix_output_stride_m,
        int64_t prefix_output_stride_n, int prefix_output_size_m,
        int prefix_output_size_n, DTYPE* __restrict__ output_ptr,
        int64_t output_stride_m, int64_t output_stride_n, int output_size_m,
        int output_size_n, DTYPE* __restrict__ block_output_ptr,
        int64_t block_output_stride_m, int64_t block_output_stride_n,
        int block_output_size_m, int block_output_size_n, int num_blocks,
        float eps, float output_eps) {
  const int tiles = (input_size_m + TILE_M - 1) / TILE_M;
  for (int t = blockIdx.x; t < tiles; t += gridDim.x)
    add_attn_res_rms_norm_rows<DTYPE, TILE_M, TILE_N, TILE_K, BLOCK_X, BLOCK_Y,
                               WAVE_SIZE, PACK_BYTES, CHUNKED>(
        t * TILE_M, input_ptr, input_stride_m, input_stride_n, input_size_m,
        input_size_n, prefix_ptr, prefix_stride_m, prefix_stride_n, blocks_ptr,
        blocks_stride_m, blocks_stride_r, blocks_stride_n, norm_weight_ptr,
        norm_weight_stride_n, qk_weight_ptr, qk_weight_stride_n,
        output_norm_weight_ptr, output_norm_weight_stride_n, prefix_output_ptr,
        prefix_output_stride_m, prefix_output_stride_n, output_ptr,
        output_stride_m, output_stride_n, block_output_ptr,
        block_output_stride_m, block_output_stride_n, num_blocks, eps,
        output_eps);
}

template <Protocol PROTOCOL, bool QUANTIZE_INT4, typename DTYPE, int WORLD,
          int TILE_M, int TILE_N, int TILE_K, int BLOCK_X, int BLOCK_Y,
          int WAVE_SIZE, int PACK_BYTES, bool CHUNKED, bool TAIL>
DINLINE void all_reduce_add_attn_res_rms_norm_one_shot_pull(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* const* input_ptr_array,
    int input_ptr_array_size, int64_t input_ptr_array_stride_m,
    int64_t input_ptr_array_stride_n, const DTYPE* prefix_ptr,
    int64_t prefix_stride_m, int64_t prefix_stride_n, const DTYPE* blocks_ptr,
    int64_t blocks_stride_m, int64_t blocks_stride_r, int64_t blocks_stride_n,
    const DTYPE* norm_weight_ptr, int64_t norm_weight_stride_n,
    const DTYPE* qk_weight_ptr, int64_t qk_weight_stride_n,
    const DTYPE* output_norm_weight_ptr, int64_t output_norm_weight_stride_n,
    DTYPE* prefix_output_ptr, int64_t prefix_output_stride_m,
    int64_t prefix_output_stride_n, DTYPE* output_ptr, int64_t output_stride_m,
    int64_t output_stride_n, DTYPE* block_output_ptr,
    int64_t block_output_stride_m, int64_t block_output_stride_n,
    char* const* partials_ptr_array, int partials_ptr_array_size,
    int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
    int partials_size_n, char* const* staging_ptr_array,
    int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    char* const* inbox_ptr_array, int inbox_ptr_array_size,
    int64_t inbox_stride_slot, int64_t inbox_stride_m, int64_t inbox_stride_n,
    int inbox_size_slot, int inbox_size_m, int inbox_size_n,
    int32_t* connection_ptr, int64_t connection_stride_block,
    int64_t connection_stride_word, int connection_size_block,
    int connection_size_word, uint32_t* const* counters_ptr_array,
    int counters_ptr_array_size, int64_t counters_stride_kind,
    int64_t counters_stride_block, int64_t counters_stride_peer,
    int counters_size_block, int counters_size_peer,
    uint32_t* const* signals_ptr_array, int signals_ptr_array_size,
    int64_t signals_stride_parity, int64_t signals_stride_block,
    int64_t signals_stride_writer, int signals_size_parity,
    int signals_size_block, int signals_size_writer, uint32_t* status_ptr,
    int64_t status_stride_field, int status_size_field, int rank,
    uint64_t timeout_ticks, int num_blocks, float eps, float output_eps) {
  // A tile is a round: this rank's tile to every rank, every rank's from
  // each, through the protocol's inbox (the block's row); the sum and the
  // norm as the pull's.
  static_assert(!QUANTIZE_INT4 && TILE_M == 1,
                "the Simple protocols' fused one-shot is exact, a row a tile");
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
  constexpr int kPacks =
      (TILE_N + kPackElems * BLOCK_X - 1) / (kPackElems * BLOCK_X);
  constexpr int kSpan = kPacks * kPackElems * BLOCK_X;
  const int m = input_size_m, n = input_size_n;
  const int inbox_row = static_cast<int>(blockIdx.x);
  // The block's connection, a line of its own (CONNECTION).
  Connection connection;
  load<WORLD, 1, BLOCK_X, WAVE_SIZE, false, Layout::block>(
      connection_ptr, 0, 0, connection_stride_block, connection_stride_word, 1,
      connection_size_block, connection_size_word, connection_size_block,
      connection_size_word, connection, inbox_row, 0, 1);
  for (int row = blockIdx.x * TILE_M; row < m; row += gridDim.x * TILE_M) {
    for (int c = 0; c < n; c += kSpan) {
      DTYPE mine[TILE_M][kPacks][kPackElems];
      load<WORLD, TILE_M, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
          input_ptr, 0, 0, input_stride_m, input_stride_n, 1, m, n, m, n, mine,
          row, c, 1);
      send<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE, false, PROTOCOL,
           Layout::tile>(
          mine, inbox_ptr_array, inbox_ptr_array_size, 0, WORLD, rank,
          inbox_stride_slot, inbox_stride_m, inbox_stride_n, inbox_size_slot,
          inbox_size_m, inbox_size_n, inbox_row + 1, min(kSpan, n - c),
          inbox_row, 0, 1, &connection, counters_ptr_array,
          counters_ptr_array_size, counters_stride_kind, counters_stride_block,
          counters_stride_peer, counters_size_block, counters_size_peer,
          status_ptr, status_stride_field, status_size_field, timeout_ticks);
      DTYPE copies[WORLD][TILE_M][kPacks][kPackElems];
      recv<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE, false, PROTOCOL,
           Layout::tile>(
          inbox_ptr_array, inbox_ptr_array_size, rank, 0, inbox_stride_slot,
          inbox_stride_m, inbox_stride_n, inbox_size_slot, inbox_size_m,
          inbox_size_n, inbox_row + 1, min(kSpan, n - c), copies, inbox_row, 0,
          1, &connection, counters_ptr_array, counters_ptr_array_size,
          counters_stride_kind, counters_stride_block, counters_stride_peer,
          counters_size_block, counters_size_peer, status_ptr,
          status_stride_field, status_size_field, timeout_ticks);
      DTYPE sum[TILE_M][kPacks][kPackElems];
#pragma unroll
      for (int i = 0; i < TILE_M; ++i)
#pragma unroll
        for (int k = 0; k < kPacks; ++k)
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) {
            float acc = static_cast<float>(copies[0][i][k][j]);
#pragma unroll
            for (int r = 1; r < WORLD; ++r)
              acc += static_cast<float>(copies[r][i][k][j]);
            sum[i][k][j] = static_cast<DTYPE>(acc);
          }

      if constexpr (CHUNKED)
        store<1, TILE_M, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
            sum, output_ptr, 0, 0, output_stride_m, output_stride_n, 1, m, n, m,
            n, row, c, 1);
      else
        add_attn_res_rms_norm_rows<TILE_K, BLOCK_X, BLOCK_Y, WAVE_SIZE>(
            row, sum, m, n, prefix_ptr, prefix_stride_m, prefix_stride_n,
            blocks_ptr, blocks_stride_m, blocks_stride_r, blocks_stride_n,
            norm_weight_ptr, norm_weight_stride_n, qk_weight_ptr,
            qk_weight_stride_n, output_norm_weight_ptr,
            output_norm_weight_stride_n, prefix_output_ptr,
            prefix_output_stride_m, prefix_output_stride_n, output_ptr,
            output_stride_m, output_stride_n, block_output_ptr,
            block_output_stride_m, block_output_stride_n, num_blocks, eps,
            output_eps);
    }
    if constexpr (CHUNKED)
      add_attn_res_rms_norm_rows<DTYPE, TILE_M, TILE_N, TILE_K, BLOCK_X,
                                 BLOCK_Y, WAVE_SIZE, PACK_BYTES, CHUNKED>(
          row, output_ptr, output_stride_m, output_stride_n, m, n, prefix_ptr,
          prefix_stride_m, prefix_stride_n, blocks_ptr, blocks_stride_m,
          blocks_stride_r, blocks_stride_n, norm_weight_ptr,
          norm_weight_stride_n, qk_weight_ptr, qk_weight_stride_n,
          output_norm_weight_ptr, output_norm_weight_stride_n,
          prefix_output_ptr, prefix_output_stride_m, prefix_output_stride_n,
          output_ptr, output_stride_m, output_stride_n, block_output_ptr,
          block_output_stride_m, block_output_stride_n, num_blocks, eps,
          output_eps);
  }

  store<WORLD, 1, BLOCK_X, WAVE_SIZE, false, Layout::block>(
      connection, connection_ptr, 0, 0, connection_stride_block,
      connection_stride_word, 1, connection_size_block, connection_size_word,
      connection_size_block, connection_size_word, inbox_row, 0, 1);
}

template <Protocol PROTOCOL, bool QUANTIZE_INT4, typename DTYPE, int WORLD,
          int TILE_M, int TILE_N, int TILE_K, int BLOCK_X, int BLOCK_Y,
          int WAVE_SIZE, int PACK_BYTES, bool CHUNKED, bool TAIL>
DINLINE void all_reduce_add_attn_res_rms_norm_two_shot_pull_columns(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* const* input_ptr_array,
    int input_ptr_array_size, int64_t input_ptr_array_stride_m,
    int64_t input_ptr_array_stride_n, const DTYPE* prefix_ptr,
    int64_t prefix_stride_m, int64_t prefix_stride_n, const DTYPE* blocks_ptr,
    int64_t blocks_stride_m, int64_t blocks_stride_r, int64_t blocks_stride_n,
    const DTYPE* norm_weight_ptr, int64_t norm_weight_stride_n,
    const DTYPE* qk_weight_ptr, int64_t qk_weight_stride_n,
    const DTYPE* output_norm_weight_ptr, int64_t output_norm_weight_stride_n,
    DTYPE* prefix_output_ptr, int64_t prefix_output_stride_m,
    int64_t prefix_output_stride_n, DTYPE* output_ptr, int64_t output_stride_m,
    int64_t output_stride_n, DTYPE* block_output_ptr,
    int64_t block_output_stride_m, int64_t block_output_stride_n,
    char* const* partials_ptr_array, int partials_ptr_array_size,
    int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
    int partials_size_n, char* const* staging_ptr_array,
    int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    char* const* inbox_ptr_array, int inbox_ptr_array_size,
    int64_t inbox_stride_slot, int64_t inbox_stride_m, int64_t inbox_stride_n,
    int inbox_size_slot, int inbox_size_m, int inbox_size_n,
    int32_t* connection_ptr, int64_t connection_stride_block,
    int64_t connection_stride_word, int connection_size_block,
    int connection_size_word, uint32_t* const* counters_ptr_array,
    int counters_ptr_array_size, int64_t counters_stride_kind,
    int64_t counters_stride_block, int64_t counters_stride_peer,
    int counters_size_block, int counters_size_peer,
    uint32_t* const* signals_ptr_array, int signals_ptr_array_size,
    int64_t signals_stride_parity, int64_t signals_stride_block,
    int64_t signals_stride_writer, int signals_size_parity,
    int signals_size_block, int signals_size_writer, uint32_t* status_ptr,
    int64_t status_stride_field, int status_size_field, int rank,
    uint64_t timeout_ticks, int num_blocks, float eps, float output_eps) {
  // A row's columns are WORLD shares, share w rank w's. A chunk of the
  // shares is two rounds: the reduce-scatter (share w's chunk to rank w,
  // this rank's from every rank, summed) and the all-gather (the sum to
  // every rank, share w's chunk from rank w). Each lane group talks to one
  // peer, w; the inbox row is the block's. The gathered row is stored and
  // normalized there, as the chunked pull's.
  // INT4: a share is whole groups, as Simple's, and a lane holds a group a
  // round; the parts go quantized, are summed in float from their
  // dequantized values, and the sum goes quantized again.
  static_assert(TILE_M == 1,
                "the Simple protocols' fused two-shot is a row a tile");
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
  static_assert(!QUANTIZE_INT4 || kPackElems * kInt4GroupPacks == kInt4Group,
                "a group is kInt4GroupPacks packs");
  constexpr int kLanes = BLOCK_X / WORLD;
  static_assert(kLanes * WORLD == BLOCK_X,
                "a block is a group of threads a rank");
  constexpr int kUnit = QUANTIZE_INT4 ? kInt4Group : kPackElems;
  constexpr int kChunkN =
      QUANTIZE_INT4 ? kLanes * kInt4Group
                    : (TILE_N + kPackElems - 1) / kPackElems * kPackElems;
  constexpr int kPacks =
      (kChunkN + kPackElems * kLanes - 1) / (kPackElems * kLanes);
  const int m = input_size_m, n = input_size_n;
  const int units =
      QUANTIZE_INT4 ? static_cast<int>(int4_groups(n)) : n / kPackElems;
  const int slice = (units + WORLD - 1) / WORLD * kUnit;
  // The INT4 payload's scales follow every slot's and row's values.
  [[maybe_unused]] const int payload_m = inbox_size_slot * inbox_size_m;
  const int w = (rank + static_cast<int>(threadIdx.x) / kLanes) % WORLD;
  const int inbox_row = static_cast<int>(blockIdx.x);
  // A share is slice long, the last one cut at the row's end.
  const int share_w = max(0, min(slice, n - w * slice));
  const int share_me = max(0, min(slice, n - rank * slice));

  Connection connection;
  load<WORLD, 1, BLOCK_X, WAVE_SIZE, false, Layout::block>(
      connection_ptr, 0, 0, connection_stride_block, connection_stride_word, 1,
      connection_size_block, connection_size_word, connection_size_block,
      connection_size_word, connection, inbox_row, 0, 1);
  for (int row = blockIdx.x; row < m; row += gridDim.x) {
    for (int lo = 0; lo < slice; lo += kChunkN) {
      if constexpr (QUANTIZE_INT4) {
        const int end_w = min(share_w, lo + kChunkN);
        const int end_me = min(share_me, lo + kChunkN);
        // A recv reads at least a group, past the end when the share is
        // empty here: those values go nowhere.
        const int read_w = max(end_w - lo, kInt4Group);
        const int read_me = max(end_me - lo, kInt4Group);

        DTYPE part[1][kInt4GroupPacks][kPackElems] = {};
        if (end_w > lo)
          load<WORLD, WORLD, BLOCK_X, WAVE_SIZE, false, Layout::group_flat,
               TAIL>(input_ptr, w, slice, input_stride_m, input_stride_n, WORLD,
                     m, n, m, end_w, part, row, lo, 1);
        send<WORLD, WORLD, BLOCK_X, BLOCK_Y, WAVE_SIZE, true, PROTOCOL,
             Layout::group_run, TAIL>(
            part, inbox_ptr_array, inbox_ptr_array_size, w, 1, rank,
            inbox_stride_slot, inbox_stride_m, inbox_stride_n, inbox_size_slot,
            payload_m, inbox_size_n, inbox_row + 1, max(end_w - lo, 0),
            inbox_row, 0, 1, &connection, counters_ptr_array,
            counters_ptr_array_size, counters_stride_kind,
            counters_stride_block, counters_stride_peer, counters_size_block,
            counters_size_peer, status_ptr, status_stride_field,
            status_size_field, timeout_ticks);
        uint32_t got[1][1][kInt4GroupPacks][2];
        recv<WORLD, WORLD, BLOCK_X, BLOCK_Y, WAVE_SIZE, true, PROTOCOL,
             Layout::group_run, TAIL>(
            inbox_ptr_array, inbox_ptr_array_size, rank, w, inbox_stride_slot,
            inbox_stride_m, inbox_stride_n, inbox_size_slot, payload_m,
            inbox_size_n, inbox_row + 1, read_me, got, inbox_row, 0, 1,
            &connection, counters_ptr_array, counters_ptr_array_size,
            counters_stride_kind, counters_stride_block, counters_stride_peer,
            counters_size_block, counters_size_peer, status_ptr,
            status_stride_field, status_size_field, timeout_ticks);
        DTYPE parts[kInt4GroupPacks][kPackElems];
#pragma unroll
        for (int k = 0; k < kInt4GroupPacks; ++k)
          dequantize(got[0][0][k], parts[k]);
        float sum[1][kInt4GroupPacks][kPackElems];
        reduce<WORLD, BLOCK_X, WAVE_SIZE, ReduceAxis::m_all>(parts, sum[0]);

        send<WORLD, WORLD, BLOCK_X, BLOCK_Y, WAVE_SIZE, true, PROTOCOL,
             Layout::group_run, TAIL>(
            sum, inbox_ptr_array, inbox_ptr_array_size, w, 1, rank,
            inbox_stride_slot, inbox_stride_m, inbox_stride_n, inbox_size_slot,
            payload_m, inbox_size_n, inbox_row + 1, max(end_me - lo, 0),
            inbox_row, 0, 1, &connection, counters_ptr_array,
            counters_ptr_array_size, counters_stride_kind,
            counters_stride_block, counters_stride_peer, counters_size_block,
            counters_size_peer, status_ptr, status_stride_field,
            status_size_field, timeout_ticks);
        uint32_t words[1][1][kInt4GroupPacks][2];
        recv<WORLD, WORLD, BLOCK_X, BLOCK_Y, WAVE_SIZE, true, PROTOCOL,
             Layout::group_run, TAIL>(
            inbox_ptr_array, inbox_ptr_array_size, rank, w, inbox_stride_slot,
            inbox_stride_m, inbox_stride_n, inbox_size_slot, payload_m,
            inbox_size_n, inbox_row + 1, read_w, words, inbox_row, 0, 1,
            &connection, counters_ptr_array, counters_ptr_array_size,
            counters_stride_kind, counters_stride_block, counters_stride_peer,
            counters_size_block, counters_size_peer, status_ptr,
            status_stride_field, status_size_field, timeout_ticks);
        DTYPE out[1][kInt4GroupPacks][kPackElems];
#pragma unroll
        for (int k = 0; k < kInt4GroupPacks; ++k)
          dequantize(words[0][0][k], out[0][k]);
        store<WORLD, WORLD, BLOCK_X, WAVE_SIZE, false, Layout::group_flat,
              TAIL>(out, output_ptr, w, slice, output_stride_m, output_stride_n,
                    WORLD, m, n, m, end_w, row, lo, 1);
      } else {
        const int end_w = min(share_w, lo + kChunkN);
        const int end_me = min(share_me, lo + kChunkN);
        // A recv reads at least a pack, past the end when the share is empty
        // here: those values go nowhere.
        const int read_w = max(end_w - lo, kPackElems);
        const int read_me = max(end_me - lo, kPackElems);

        DTYPE part[1][kPacks][kPackElems] = {};
        if (end_w > lo)
          load<WORLD, 1, BLOCK_X, WAVE_SIZE, false, Layout::flat>(
              input_ptr, w, slice, input_stride_m, input_stride_n, WORLD, m, n,
              m, end_w, part, row, lo, 1);
        send<WORLD, 1, BLOCK_X, BLOCK_Y, WAVE_SIZE, false, PROTOCOL,
             Layout::run>(
            part, inbox_ptr_array, inbox_ptr_array_size, w, 1, rank,
            inbox_stride_slot, inbox_stride_m, inbox_stride_n, inbox_size_slot,
            inbox_size_m, inbox_size_n, inbox_row + 1, max(end_w - lo, 0),
            inbox_row, 0, 1, &connection, counters_ptr_array,
            counters_ptr_array_size, counters_stride_kind,
            counters_stride_block, counters_stride_peer, counters_size_block,
            counters_size_peer, status_ptr, status_stride_field,
            status_size_field, timeout_ticks);
        DTYPE got[1][kPacks][kPackElems];
        recv<WORLD, 1, BLOCK_X, BLOCK_Y, WAVE_SIZE, false, PROTOCOL,
             Layout::run>(
            inbox_ptr_array, inbox_ptr_array_size, rank, w, inbox_stride_slot,
            inbox_stride_m, inbox_stride_n, inbox_size_slot, inbox_size_m,
            inbox_size_n, inbox_row + 1, read_me, got, inbox_row, 0, 1,
            &connection, counters_ptr_array, counters_ptr_array_size,
            counters_stride_kind, counters_stride_block, counters_stride_peer,
            counters_size_block, counters_size_peer, status_ptr,
            status_stride_field, status_size_field, timeout_ticks);
        DTYPE sum[1][kPacks][kPackElems];
        reduce<WORLD, BLOCK_X, WAVE_SIZE, ReduceAxis::m_all>(got[0], sum[0]);

        send<WORLD, 1, BLOCK_X, BLOCK_Y, WAVE_SIZE, false, PROTOCOL,
             Layout::run>(
            sum, inbox_ptr_array, inbox_ptr_array_size, w, 1, rank,
            inbox_stride_slot, inbox_stride_m, inbox_stride_n, inbox_size_slot,
            inbox_size_m, inbox_size_n, inbox_row + 1, max(end_me - lo, 0),
            inbox_row, 0, 1, &connection, counters_ptr_array,
            counters_ptr_array_size, counters_stride_kind,
            counters_stride_block, counters_stride_peer, counters_size_block,
            counters_size_peer, status_ptr, status_stride_field,
            status_size_field, timeout_ticks);
        DTYPE out[1][kPacks][kPackElems];
        recv<WORLD, 1, BLOCK_X, BLOCK_Y, WAVE_SIZE, false, PROTOCOL,
             Layout::run>(
            inbox_ptr_array, inbox_ptr_array_size, rank, w, inbox_stride_slot,
            inbox_stride_m, inbox_stride_n, inbox_size_slot, inbox_size_m,
            inbox_size_n, inbox_row + 1, read_w, out, inbox_row, 0, 1,
            &connection, counters_ptr_array, counters_ptr_array_size,
            counters_stride_kind, counters_stride_block, counters_stride_peer,
            counters_size_block, counters_size_peer, status_ptr,
            status_stride_field, status_size_field, timeout_ticks);
        store<WORLD, 1, BLOCK_X, WAVE_SIZE, false, Layout::flat>(
            out, output_ptr, w, slice, output_stride_m, output_stride_n, WORLD,
            m, n, m, end_w, row, lo, 1);
      }
    }
    // The row is in output, each share stored by its lane group; the norm
    // reads it a tile at a time.
    __syncthreads();
    add_attn_res_rms_norm_rows<DTYPE, 1, TILE_N, TILE_K, BLOCK_X, BLOCK_Y,
                               WAVE_SIZE, PACK_BYTES, true>(
        row, output_ptr, output_stride_m, output_stride_n, m, n, prefix_ptr,
        prefix_stride_m, prefix_stride_n, blocks_ptr, blocks_stride_m,
        blocks_stride_r, blocks_stride_n, norm_weight_ptr, norm_weight_stride_n,
        qk_weight_ptr, qk_weight_stride_n, output_norm_weight_ptr,
        output_norm_weight_stride_n, prefix_output_ptr, prefix_output_stride_m,
        prefix_output_stride_n, output_ptr, output_stride_m, output_stride_n,
        block_output_ptr, block_output_stride_m, block_output_stride_n,
        num_blocks, eps, output_eps);
  }
  store<WORLD, 1, BLOCK_X, WAVE_SIZE, false, Layout::block>(
      connection, connection_ptr, 0, 0, connection_stride_block,
      connection_stride_word, 1, connection_size_block, connection_size_word,
      connection_size_block, connection_size_word, inbox_row, 0, 1);
}

template <AllReduceAddAttnResRmsNormAlgorithm ALGORITHM, Protocol PROTOCOL,
          bool QUANTIZE_INT4, typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int TILE_K, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE, int PACK_BYTES,
          int WAVES_PER_EU, bool CHUNKED, bool TAIL>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y, WAVES_PER_EU)
    all_reduce_add_attn_res_rms_norm(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const DTYPE* const* __restrict__ input_ptr_array,
        int input_ptr_array_size, int64_t input_ptr_array_stride_m,
        int64_t input_ptr_array_stride_n, const DTYPE* __restrict__ prefix_ptr,
        int64_t prefix_stride_m, int64_t prefix_stride_n, int prefix_size_m,
        int prefix_size_n, const DTYPE* __restrict__ blocks_ptr,
        int64_t blocks_stride_m, int64_t blocks_stride_r,
        int64_t blocks_stride_n, int blocks_size_m, int blocks_size_r,
        int blocks_size_n, const DTYPE* __restrict__ norm_weight_ptr,
        int64_t norm_weight_stride_n, int norm_weight_size_n,
        const DTYPE* __restrict__ qk_weight_ptr, int64_t qk_weight_stride_n,
        int qk_weight_size_n, const DTYPE* __restrict__ output_norm_weight_ptr,
        int64_t output_norm_weight_stride_n, int output_norm_weight_size_n,
        DTYPE* __restrict__ prefix_output_ptr, int64_t prefix_output_stride_m,
        int64_t prefix_output_stride_n, int prefix_output_size_m,
        int prefix_output_size_n, DTYPE* output_ptr, int64_t output_stride_m,
        int64_t output_stride_n, int output_size_m, int output_size_n,
        DTYPE* __restrict__ block_output_ptr, int64_t block_output_stride_m,
        int64_t block_output_stride_n, int block_output_size_m,
        int block_output_size_n, char* const* __restrict__ partials_ptr_array,
        int partials_ptr_array_size, int64_t partials_stride_m,
        int64_t partials_stride_n, int partials_size_m, int partials_size_n,
        char* const* __restrict__ staging_ptr_array, int staging_ptr_array_size,
        int64_t staging_stride_m, int64_t staging_stride_n, int staging_size_m,
        int staging_size_n, char* const* inbox_ptr_array,
        int inbox_ptr_array_size, int64_t inbox_stride_slot,
        int64_t inbox_stride_m, int64_t inbox_stride_n, int inbox_size_slot,
        int inbox_size_m, int inbox_size_n, int32_t* connection_ptr,
        int64_t connection_stride_block, int64_t connection_stride_word,
        int connection_size_block, int connection_size_word,
        uint32_t* const* counters_ptr_array, int counters_ptr_array_size,
        int64_t counters_stride_kind, int64_t counters_stride_block,
        int64_t counters_stride_peer, int counters_size_block,
        int counters_size_peer, uint32_t* const* __restrict__ signals_ptr_array,
        int signals_ptr_array_size, int64_t signals_stride_parity,
        int64_t signals_stride_block, int64_t signals_stride_writer,
        int signals_size_parity, int signals_size_block,
        int signals_size_writer, uint32_t* __restrict__ status_ptr,
        int64_t status_stride_field, int status_size_field, int rank,
        uint64_t timeout_ticks, int num_blocks, float eps, float output_eps) {
  // Every stage synchronizes in send and recv, on simple_write or
  // simple_read.
  static_assert(
      PROTOCOL == Protocol::simple_write || PROTOCOL == Protocol::simple_read,
      "the fused add_attn_res_rms_norm is simple_write or simple_read");
  if constexpr (ALGORITHM == AllReduceAddAttnResRmsNormAlgorithm::one_shot_pull)
    all_reduce_add_attn_res_rms_norm_one_shot_pull<
        PROTOCOL, QUANTIZE_INT4, DTYPE, WORLD, TILE_M, TILE_N, TILE_K, BLOCK_X,
        BLOCK_Y, WAVE_SIZE, PACK_BYTES, CHUNKED, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m,
        input_ptr_array_stride_n, prefix_ptr, prefix_stride_m, prefix_stride_n,
        blocks_ptr, blocks_stride_m, blocks_stride_r, blocks_stride_n,
        norm_weight_ptr, norm_weight_stride_n, qk_weight_ptr,
        qk_weight_stride_n, output_norm_weight_ptr, output_norm_weight_stride_n,
        prefix_output_ptr, prefix_output_stride_m, prefix_output_stride_n,
        output_ptr, output_stride_m, output_stride_n, block_output_ptr,
        block_output_stride_m, block_output_stride_n, partials_ptr_array,
        partials_ptr_array_size, partials_stride_m, partials_stride_n,
        partials_size_m, partials_size_n, staging_ptr_array,
        staging_ptr_array_size, staging_stride_m, staging_stride_n,
        staging_size_m, staging_size_n, inbox_ptr_array, inbox_ptr_array_size,
        inbox_stride_slot, inbox_stride_m, inbox_stride_n, inbox_size_slot,
        inbox_size_m, inbox_size_n, connection_ptr, connection_stride_block,
        connection_stride_word, connection_size_block, connection_size_word,
        counters_ptr_array, counters_ptr_array_size, counters_stride_kind,
        counters_stride_block, counters_stride_peer, counters_size_block,
        counters_size_peer, signals_ptr_array, signals_ptr_array_size,
        signals_stride_parity, signals_stride_block, signals_stride_writer,
        signals_size_parity, signals_size_block, signals_size_writer,
        status_ptr, status_stride_field, status_size_field, rank, timeout_ticks,
        num_blocks, eps, output_eps);
  else if constexpr (ALGORITHM ==
                     AllReduceAddAttnResRmsNormAlgorithm::two_shot_pull_columns)
    all_reduce_add_attn_res_rms_norm_two_shot_pull_columns<
        PROTOCOL, QUANTIZE_INT4, DTYPE, WORLD, TILE_M, TILE_N, TILE_K, BLOCK_X,
        BLOCK_Y, WAVE_SIZE, PACK_BYTES, CHUNKED, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m,
        input_ptr_array_stride_n, prefix_ptr, prefix_stride_m, prefix_stride_n,
        blocks_ptr, blocks_stride_m, blocks_stride_r, blocks_stride_n,
        norm_weight_ptr, norm_weight_stride_n, qk_weight_ptr,
        qk_weight_stride_n, output_norm_weight_ptr, output_norm_weight_stride_n,
        prefix_output_ptr, prefix_output_stride_m, prefix_output_stride_n,
        output_ptr, output_stride_m, output_stride_n, block_output_ptr,
        block_output_stride_m, block_output_stride_n, partials_ptr_array,
        partials_ptr_array_size, partials_stride_m, partials_stride_n,
        partials_size_m, partials_size_n, staging_ptr_array,
        staging_ptr_array_size, staging_stride_m, staging_stride_n,
        staging_size_m, staging_size_n, inbox_ptr_array, inbox_ptr_array_size,
        inbox_stride_slot, inbox_stride_m, inbox_stride_n, inbox_size_slot,
        inbox_size_m, inbox_size_n, connection_ptr, connection_stride_block,
        connection_stride_word, connection_size_block, connection_size_word,
        counters_ptr_array, counters_ptr_array_size, counters_stride_kind,
        counters_stride_block, counters_stride_peer, counters_size_block,
        counters_size_peer, signals_ptr_array, signals_ptr_array_size,
        signals_stride_parity, signals_stride_block, signals_stride_writer,
        signals_size_parity, signals_size_block, signals_size_writer,
        status_ptr, status_stride_field, status_size_field, rank, timeout_ticks,
        num_blocks, eps, output_eps);
}

}  // namespace hip_custom_communicator
