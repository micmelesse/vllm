#pragma once

// All-reduce fused with RMSNorm, with or without a residual add.

#include "all_gather.cuh"
#include "common/load.cuh"
#include "common/recv.cuh"
#include "common/send.cuh"
#include "common/store.cuh"
#include "common/types.cuh"
#include "rms_normalize.cuh"

namespace hip_custom_communicator {

enum class AllReduceAddRmsNormKernelAlgorithm { one_shot, two_shot };

template <bool RESIDUAL, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE,
          typename DTYPE, typename WEIGHT_DTYPE, int ROWS, int K, int P>
DINLINE void add_rms_norm_rows(int offs_m, const DTYPE (&sum)[ROWS][K][P],
                               const WEIGHT_DTYPE (&w)[1][K][P],
                               int input_size_m, int input_size_n,
                               const DTYPE* residual_ptr,
                               int64_t residual_stride_m,
                               int64_t residual_stride_n, DTYPE* output_ptr,
                               int64_t output_stride_m, int64_t output_stride_n,
                               DTYPE* residual_output_ptr,
                               int64_t residual_output_stride_m,
                               int64_t residual_output_stride_n, float eps) {
  const int m = input_size_m, n = input_size_n;
  const float inv_hidden = 1.0f / static_cast<float>(n);
  DTYPE res[ROWS][K][P];
  if constexpr (RESIDUAL)
    load<1, ROWS, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
        residual_ptr, 0, 0, residual_stride_m, residual_stride_n, 1, m, n, m, n,
        res, offs_m, 0, 1);
  float s[ROWS][K][P];
  DTYPE added[ROWS][K][P];
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < P; ++j) {
        s[i][k][j] = static_cast<float>(sum[i][k][j]);
        if constexpr (RESIDUAL) s[i][k][j] += static_cast<float>(res[i][k][j]);
        added[i][k][j] = static_cast<DTYPE>(s[i][k][j]);
      }
  if constexpr (RESIDUAL)
    store<1, ROWS, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
        added, residual_output_ptr, 0, 0, residual_output_stride_m,
        residual_output_stride_n, 1, m, n, m, n, offs_m, 0, 1);
  float scale[ROWS];
  DTYPE normed[ROWS][K][P];
  rms_normalize<BLOCK_X, BLOCK_Y, WAVE_SIZE>(s, scale, w, normed, eps,
                                             inv_hidden, n, 0, NormPart::row,
                                             true, Layout::tile);
  store<1, ROWS, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
      normed, output_ptr, 0, 0, output_stride_m, output_stride_n, 1, m, n, m, n,
      offs_m, 0, 1);
}

template <bool RESIDUAL, typename DTYPE, typename WEIGHT_DTYPE, int TILE_M,
          int TILE_N, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE, int PACK_BYTES,
          bool CHUNKED>
DINLINE void add_rms_norm_rows(
    int offs_m, const DTYPE* input_ptr, int64_t input_stride_m,
    int64_t input_stride_n, int input_size_m, int input_size_n,
    const DTYPE* residual_ptr, int64_t residual_stride_m,
    int64_t residual_stride_n, const WEIGHT_DTYPE* weight_ptr,
    int64_t weight_stride_n, DTYPE* output_ptr, int64_t output_stride_m,
    int64_t output_stride_n, DTYPE* residual_output_ptr,
    int64_t residual_output_stride_m, int64_t residual_output_stride_n,
    float eps) {
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;

  constexpr int kPacks =
      (TILE_N + kPackElems * BLOCK_X - 1) / (kPackElems * BLOCK_X);
  constexpr int kSpan = kPacks * kPackElems * BLOCK_X;
  const int m = input_size_m, n = input_size_n;
  const float inv_hidden = 1.0f / static_cast<float>(n);
  if constexpr (CHUNKED) {
    float scale[TILE_M];
    for (int i = 0; i < TILE_M; ++i) scale[i] = 0.0f;
    for (int c = 0; c < n; c += kSpan) {
      DTYPE x[TILE_M][kPacks][kPackElems], res[TILE_M][kPacks][kPackElems];
      load<1, TILE_M, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
          input_ptr, 0, 0, input_stride_m, input_stride_n, 1, m, n, m, n, x,
          offs_m, c, 1);
      if constexpr (RESIDUAL)
        load<1, TILE_M, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
            residual_ptr, 0, 0, residual_stride_m, residual_stride_n, 1, m, n,
            m, n, res, offs_m, c, 1);
      float s[TILE_M][kPacks][kPackElems];
#pragma unroll
      for (int i = 0; i < TILE_M; ++i)
#pragma unroll
        for (int k = 0; k < kPacks; ++k)
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) {
            s[i][k][j] = static_cast<float>(x[i][k][j]);
            if constexpr (RESIDUAL)
              s[i][k][j] += static_cast<float>(res[i][k][j]);
          }
      WEIGHT_DTYPE unused_w[1][kPacks][kPackElems];
      DTYPE unused_out[TILE_M][kPacks][kPackElems];
      rms_normalize<BLOCK_X, BLOCK_Y, WAVE_SIZE>(
          s, scale, unused_w, unused_out, eps, inv_hidden, n, c, NormPart::wide,
          true, Layout::tile);
    }

    for (int c = 0; c < n; c += kSpan) {
      DTYPE x[TILE_M][kPacks][kPackElems], res[TILE_M][kPacks][kPackElems];
      load<1, TILE_M, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
          input_ptr, 0, 0, input_stride_m, input_stride_n, 1, m, n, m, n, x,
          offs_m, c, 1);
      if constexpr (RESIDUAL)
        load<1, TILE_M, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
            residual_ptr, 0, 0, residual_stride_m, residual_stride_n, 1, m, n,
            m, n, res, offs_m, c, 1);
      float s[TILE_M][kPacks][kPackElems];
      DTYPE added[TILE_M][kPacks][kPackElems];
#pragma unroll
      for (int i = 0; i < TILE_M; ++i)
#pragma unroll
        for (int k = 0; k < kPacks; ++k)
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) {
            s[i][k][j] = static_cast<float>(x[i][k][j]);
            if constexpr (RESIDUAL)
              s[i][k][j] += static_cast<float>(res[i][k][j]);
            added[i][k][j] = static_cast<DTYPE>(s[i][k][j]);
          }
      if constexpr (RESIDUAL)
        store<1, TILE_M, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
            added, residual_output_ptr, 0, 0, residual_output_stride_m,
            residual_output_stride_n, 1, m, n, m, n, offs_m, c, 1);
      WEIGHT_DTYPE w[1][kPacks][kPackElems];
      load<1, 1, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
          weight_ptr, 0, 0, 0, weight_stride_n, 1, 1, n, 1, n, w, 0, c, 1);
      DTYPE normed[TILE_M][kPacks][kPackElems];
      rms_normalize<BLOCK_X, BLOCK_Y, WAVE_SIZE>(
          s, scale, w, normed, eps, inv_hidden, n, c, NormPart::apply, true,
          Layout::tile);
      store<1, TILE_M, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
          normed, output_ptr, 0, 0, output_stride_m, output_stride_n, 1, m, n,
          m, n, offs_m, c, 1);
    }
  } else {
    DTYPE x[TILE_M][kPacks][kPackElems];
    load<1, TILE_M, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
        input_ptr, 0, 0, input_stride_m, input_stride_n, 1, m, n, m, n, x,
        offs_m, 0, 1);
    WEIGHT_DTYPE w[1][kPacks][kPackElems];
    load<1, 1, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
        weight_ptr, 0, 0, 0, weight_stride_n, 1, 1, n, 1, n, w, 0, 0, 1);
    add_rms_norm_rows<RESIDUAL, BLOCK_X, BLOCK_Y, WAVE_SIZE>(
        offs_m, x, w, m, n, residual_ptr, residual_stride_m, residual_stride_n,
        output_ptr, output_stride_m, output_stride_n, residual_output_ptr,
        residual_output_stride_m, residual_output_stride_n, eps);
  }
}

template <bool RESIDUAL, typename DTYPE, typename WEIGHT_DTYPE, int TILE_M,
          int TILE_N, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE, int PACK_BYTES,
          int WAVES_PER_EU, bool CHUNKED>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y, WAVES_PER_EU)
    add_rms_norm(const DTYPE* input_ptr, int64_t input_stride_m,
                 int64_t input_stride_n, int input_size_m, int input_size_n,
                 const DTYPE* residual_ptr, int64_t residual_stride_m,
                 int64_t residual_stride_n, int residual_size_m,
                 int residual_size_n,
                 const WEIGHT_DTYPE* __restrict__ weight_ptr,
                 int64_t weight_stride_n, int weight_size_n, DTYPE* output_ptr,
                 int64_t output_stride_m, int64_t output_stride_n,
                 int output_size_m, int output_size_n,
                 DTYPE* residual_output_ptr, int64_t residual_output_stride_m,
                 int64_t residual_output_stride_n, int residual_output_size_m,
                 int residual_output_size_n, float eps) {
  const int tiles = (input_size_m + TILE_M - 1) / TILE_M;
  for (int t = blockIdx.x; t < tiles; t += gridDim.x)
    add_rms_norm_rows<RESIDUAL, DTYPE, WEIGHT_DTYPE, TILE_M, TILE_N, BLOCK_X,
                      BLOCK_Y, WAVE_SIZE, PACK_BYTES, CHUNKED>(
        t * TILE_M, input_ptr, input_stride_m, input_stride_n, input_size_m,
        input_size_n, residual_ptr, residual_stride_m, residual_stride_n,
        weight_ptr, weight_stride_n, output_ptr, output_stride_m,
        output_stride_n, residual_output_ptr, residual_output_stride_m,
        residual_output_stride_n, eps);
}

template <Protocol PROTOCOL, bool RESIDUAL, bool QUANTIZE_INT4, typename DTYPE,
          typename WEIGHT_DTYPE, int WORLD, int TILE_M, int TILE_N, int BLOCK_X,
          int BLOCK_Y, int WAVE_SIZE, int PACK_BYTES, bool CHUNKED, bool TAIL>
DINLINE void all_reduce_add_rms_norm_one_shot(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* residual_ptr,
    int64_t residual_stride_m, int64_t residual_stride_n,
    const WEIGHT_DTYPE* weight_ptr, int64_t weight_stride_n, DTYPE* output_ptr,
    int64_t output_stride_m, int64_t output_stride_n,
    DTYPE* residual_output_ptr, int64_t residual_output_stride_m,
    int64_t residual_output_stride_n, char* const* inbox_ptr_array,
    int inbox_ptr_array_size, int64_t inbox_stride_slot, int64_t inbox_stride_m,
    int64_t inbox_stride_n, int inbox_size_slot, int inbox_size_m,
    int inbox_size_n, int32_t* connection_ptr, int64_t connection_stride_block,
    int64_t connection_stride_word, int connection_size_block,
    int connection_size_word, uint32_t* const* counters_ptr_array,
    int counters_ptr_array_size, int64_t counters_stride_kind,
    int64_t counters_stride_block, int64_t counters_stride_peer,
    int counters_size_block, int counters_size_peer, uint32_t* status_ptr,
    int64_t status_stride_field, int status_size_field, int rank,
    uint64_t timeout_ticks, float eps) {
  // A tile is a round: this rank's tile to every rank, every rank's from
  // each, through the protocol's inbox (the block's row), summed and
  // normalized.
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
  WEIGHT_DTYPE w[1][kPacks][kPackElems];
  if constexpr (!CHUNKED)
    load<1, 1, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
        weight_ptr, 0, 0, 0, weight_stride_n, 1, 1, n, 1, n, w, 0, 0, 1);
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
        add_rms_norm_rows<RESIDUAL, BLOCK_X, BLOCK_Y, WAVE_SIZE>(
            row, sum, w, m, n, residual_ptr, residual_stride_m,
            residual_stride_n, output_ptr, output_stride_m, output_stride_n,
            residual_output_ptr, residual_output_stride_m,
            residual_output_stride_n, eps);
    }
    if constexpr (CHUNKED)
      add_rms_norm_rows<RESIDUAL, DTYPE, WEIGHT_DTYPE, TILE_M, TILE_N, BLOCK_X,
                        BLOCK_Y, WAVE_SIZE, PACK_BYTES, CHUNKED>(
          row, output_ptr, output_stride_m, output_stride_n, m, n, residual_ptr,
          residual_stride_m, residual_stride_n, weight_ptr, weight_stride_n,
          output_ptr, output_stride_m, output_stride_n, residual_output_ptr,
          residual_output_stride_m, residual_output_stride_n, eps);
  }

  store<WORLD, 1, BLOCK_X, WAVE_SIZE, false, Layout::block>(
      connection, connection_ptr, 0, 0, connection_stride_block,
      connection_stride_word, 1, connection_size_block, connection_size_word,
      connection_size_block, connection_size_word, inbox_row, 0, 1);
}

template <Protocol PROTOCOL, bool RESIDUAL, bool QUANTIZE_INT4, typename DTYPE,
          typename WEIGHT_DTYPE, int WORLD, int TILE_M, int TILE_N, int BLOCK_X,
          int BLOCK_Y, int WAVE_SIZE, int PACK_BYTES, bool CHUNKED, bool TAIL>
DINLINE void all_reduce_add_rms_norm_two_shot(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* residual_ptr,
    int64_t residual_stride_m, int64_t residual_stride_n,
    const WEIGHT_DTYPE* weight_ptr, int64_t weight_stride_n, DTYPE* output_ptr,
    int64_t output_stride_m, int64_t output_stride_n,
    DTYPE* residual_output_ptr, int64_t residual_output_stride_m,
    int64_t residual_output_stride_n, char* const* inbox_ptr_array,
    int inbox_ptr_array_size, int64_t inbox_stride_slot, int64_t inbox_stride_m,
    int64_t inbox_stride_n, int inbox_size_slot, int inbox_size_m,
    int inbox_size_n, int32_t* connection_ptr, int64_t connection_stride_block,
    int64_t connection_stride_word, int connection_size_block,
    int connection_size_word, uint32_t* const* counters_ptr_array,
    int counters_ptr_array_size, int64_t counters_stride_kind,
    int64_t counters_stride_block, int64_t counters_stride_peer,
    int counters_size_block, int counters_size_peer, uint32_t* status_ptr,
    int64_t status_stride_field, int status_size_field, int rank,
    uint64_t timeout_ticks, float eps) {
  // A row's columns are WORLD shares, share w rank w's. A chunk of the
  // shares is two rounds: the reduce-scatter (share w's chunk to rank w,
  // this rank's from every rank, summed) and the all-gather (the sum to
  // every rank, share w's chunk from rank w). Each lane group talks to one
  // peer, w; the inbox row is the block's. The gathered row is stored and
  // normalized there, a chunk at a time.
  // INT4: a share is whole groups, and a lane holds a group a
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
    add_rms_norm_rows<RESIDUAL, DTYPE, WEIGHT_DTYPE, 1, TILE_N, BLOCK_X,
                      BLOCK_Y, WAVE_SIZE, PACK_BYTES, true>(
        row, output_ptr, output_stride_m, output_stride_n, m, n, residual_ptr,
        residual_stride_m, residual_stride_n, weight_ptr, weight_stride_n,
        output_ptr, output_stride_m, output_stride_n, residual_output_ptr,
        residual_output_stride_m, residual_output_stride_n, eps);
  }
  store<WORLD, 1, BLOCK_X, WAVE_SIZE, false, Layout::block>(
      connection, connection_ptr, 0, 0, connection_stride_block,
      connection_stride_word, 1, connection_size_block, connection_size_word,
      connection_size_block, connection_size_word, inbox_row, 0, 1);
}

template <AllReduceAddRmsNormKernelAlgorithm ALGORITHM, Protocol PROTOCOL,
          bool RESIDUAL, bool QUANTIZE_INT4, typename DTYPE,
          typename WEIGHT_DTYPE, int WORLD, int TILE_M, int TILE_N, int BLOCK_X,
          int BLOCK_Y, int WAVE_SIZE, int PACK_BYTES, int WAVES_PER_EU,
          bool CHUNKED, bool TAIL>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y, WAVES_PER_EU)
    all_reduce_add_rms_norm(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const DTYPE* residual_ptr, int64_t residual_stride_m,
        int64_t residual_stride_n, int residual_size_m, int residual_size_n,
        const WEIGHT_DTYPE* __restrict__ weight_ptr, int64_t weight_stride_n,
        int weight_size_n, DTYPE* output_ptr, int64_t output_stride_m,
        int64_t output_stride_n, int output_size_m, int output_size_n,
        DTYPE* residual_output_ptr, int64_t residual_output_stride_m,
        int64_t residual_output_stride_n, int residual_output_size_m,
        int residual_output_size_n, char* const* inbox_ptr_array,
        int inbox_ptr_array_size, int64_t inbox_stride_slot,
        int64_t inbox_stride_m, int64_t inbox_stride_n, int inbox_size_slot,
        int inbox_size_m, int inbox_size_n, int32_t* connection_ptr,
        int64_t connection_stride_block, int64_t connection_stride_word,
        int connection_size_block, int connection_size_word,
        uint32_t* const* counters_ptr_array, int counters_ptr_array_size,
        int64_t counters_stride_kind, int64_t counters_stride_block,
        int64_t counters_stride_peer, int counters_size_block,
        int counters_size_peer, uint32_t* __restrict__ status_ptr,
        int64_t status_stride_field, int status_size_field, int rank,
        uint64_t timeout_ticks, float eps) {
  // Every stage synchronizes in send and recv, on simple_write or
  // simple_read.
  static_assert(
      PROTOCOL == Protocol::simple_write || PROTOCOL == Protocol::simple_read,
      "the fused add_rms_norm is simple_write or simple_read");
  if constexpr (ALGORITHM == AllReduceAddRmsNormKernelAlgorithm::one_shot)
    all_reduce_add_rms_norm_one_shot<
        PROTOCOL, RESIDUAL, QUANTIZE_INT4, DTYPE, WEIGHT_DTYPE, WORLD, TILE_M,
        TILE_N, BLOCK_X, BLOCK_Y, WAVE_SIZE, PACK_BYTES, CHUNKED, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        residual_ptr, residual_stride_m, residual_stride_n, weight_ptr,
        weight_stride_n, output_ptr, output_stride_m, output_stride_n,
        residual_output_ptr, residual_output_stride_m, residual_output_stride_n,
        inbox_ptr_array, inbox_ptr_array_size, inbox_stride_slot,
        inbox_stride_m, inbox_stride_n, inbox_size_slot, inbox_size_m,
        inbox_size_n, connection_ptr, connection_stride_block,
        connection_stride_word, connection_size_block, connection_size_word,
        counters_ptr_array, counters_ptr_array_size, counters_stride_kind,
        counters_stride_block, counters_stride_peer, counters_size_block,
        counters_size_peer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks, eps);
  else if constexpr (ALGORITHM == AllReduceAddRmsNormKernelAlgorithm::two_shot)
    all_reduce_add_rms_norm_two_shot<
        PROTOCOL, RESIDUAL, QUANTIZE_INT4, DTYPE, WEIGHT_DTYPE, WORLD, TILE_M,
        TILE_N, BLOCK_X, BLOCK_Y, WAVE_SIZE, PACK_BYTES, CHUNKED, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        residual_ptr, residual_stride_m, residual_stride_n, weight_ptr,
        weight_stride_n, output_ptr, output_stride_m, output_stride_n,
        residual_output_ptr, residual_output_stride_m, residual_output_stride_n,
        inbox_ptr_array, inbox_ptr_array_size, inbox_stride_slot,
        inbox_stride_m, inbox_stride_n, inbox_size_slot, inbox_size_m,
        inbox_size_n, connection_ptr, connection_stride_block,
        connection_stride_word, connection_size_block, connection_size_word,
        counters_ptr_array, counters_ptr_array_size, counters_stride_kind,
        counters_stride_block, counters_stride_peer, counters_size_block,
        counters_size_peer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks, eps);
}

}  // namespace hip_custom_communicator
