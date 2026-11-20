#pragma once

// All-reduce fused with RMSNorm, with or without a residual add.

#include "all_gather.cuh"
#include "common/address.cuh"
#include "common/barrier.cuh"
#include "common/load.cuh"
#include "common/recv.cuh"
#include "common/send.cuh"
#include "common/store.cuh"
#include "common/types.cuh"
#include "reduce_scatter.cuh"
#include "rms_normalize.cuh"

namespace hip_custom_communicator {

enum class AllReduceAddRmsNormKernelAlgorithm {
  one_shot_pull,
  two_shot_pull_rows,
  two_shot_pull_columns,
  two_shot_push
};

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
DINLINE void all_reduce_add_rms_norm_one_shot_pull(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* const* input_ptr_array,
    int input_ptr_array_size, int64_t input_ptr_array_stride_m,
    int64_t input_ptr_array_stride_n, const DTYPE* residual_ptr,
    int64_t residual_stride_m, int64_t residual_stride_n,
    const WEIGHT_DTYPE* weight_ptr, int64_t weight_stride_n, DTYPE* output_ptr,
    int64_t output_stride_m, int64_t output_stride_n,
    DTYPE* residual_output_ptr, int64_t residual_output_stride_m,
    int64_t residual_output_stride_n, char* const* staging_ptr_array,
    int staging_ptr_array_size, char* const* inbox_ptr_array,
    int inbox_ptr_array_size, int64_t inbox_stride_slot, int64_t inbox_stride_m,
    int64_t inbox_stride_n, int inbox_size_slot, int inbox_size_m,
    int inbox_size_n, int32_t* connection_ptr, int64_t connection_stride_block,
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
    uint64_t timeout_ticks, float eps) {
  if constexpr (PROTOCOL == Protocol::simple_write ||
                PROTOCOL == Protocol::simple_read) {
    // A tile is a round: this rank's tile to every rank, every rank's from
    // each, through the protocol's inbox (the block's row); the sum and the
    // norm as the pull's.
    static_assert(
        !QUANTIZE_INT4 && TILE_M == 1,
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
        connection_ptr, 0, 0, connection_stride_block, connection_stride_word,
        1, connection_size_block, connection_size_word, connection_size_block,
        connection_size_word, connection, inbox_row, 0, 1);
    WEIGHT_DTYPE w[1][kPacks][kPackElems];
    if constexpr (!CHUNKED)
      load<1, 1, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
          weight_ptr, 0, 0, 0, weight_stride_n, 1, 1, n, 1, n, w, 0, 0, 1);
    for (int row = blockIdx.x * TILE_M; row < m; row += gridDim.x * TILE_M) {
      for (int c = 0; c < n; c += kSpan) {
        DTYPE mine[TILE_M][kPacks][kPackElems];
        load<WORLD, TILE_M, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
            input_ptr, 0, 0, input_stride_m, input_stride_n, 1, m, n, m, n,
            mine, row, c, 1);
        send<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE, false, PROTOCOL,
             Layout::tile>(
            mine, inbox_ptr_array, inbox_ptr_array_size, 0, WORLD, rank,
            inbox_stride_slot, inbox_stride_m, inbox_stride_n, inbox_size_slot,
            inbox_size_m, inbox_size_n, inbox_row + 1, min(kSpan, n - c),
            inbox_row, 0, 1, &connection, counters_ptr_array,
            counters_ptr_array_size, counters_stride_kind,
            counters_stride_block, counters_stride_peer, counters_size_block,
            counters_size_peer, status_ptr, status_stride_field,
            status_size_field, timeout_ticks);
        DTYPE copies[WORLD][TILE_M][kPacks][kPackElems];
        recv<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE, false, PROTOCOL,
             Layout::tile>(
            inbox_ptr_array, inbox_ptr_array_size, rank, 0, inbox_stride_slot,
            inbox_stride_m, inbox_stride_n, inbox_size_slot, inbox_size_m,
            inbox_size_n, inbox_row + 1, min(kSpan, n - c), copies, inbox_row,
            0, 1, &connection, counters_ptr_array, counters_ptr_array_size,
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
              sum, output_ptr, 0, 0, output_stride_m, output_stride_n, 1, m, n,
              m, n, row, c, 1);
        else
          add_rms_norm_rows<RESIDUAL, BLOCK_X, BLOCK_Y, WAVE_SIZE>(
              row, sum, w, m, n, residual_ptr, residual_stride_m,
              residual_stride_n, output_ptr, output_stride_m, output_stride_n,
              residual_output_ptr, residual_output_stride_m,
              residual_output_stride_n, eps);
      }
      if constexpr (CHUNKED)
        add_rms_norm_rows<RESIDUAL, DTYPE, WEIGHT_DTYPE, TILE_M, TILE_N,
                          BLOCK_X, BLOCK_Y, WAVE_SIZE, PACK_BYTES, CHUNKED>(
            row, output_ptr, output_stride_m, output_stride_n, m, n,
            residual_ptr, residual_stride_m, residual_stride_n, weight_ptr,
            weight_stride_n, output_ptr, output_stride_m, output_stride_n,
            residual_output_ptr, residual_output_stride_m,
            residual_output_stride_n, eps);
    }

    store<WORLD, 1, BLOCK_X, WAVE_SIZE, false, Layout::block>(
        connection, connection_ptr, 0, 0, connection_stride_block,
        connection_stride_word, 1, connection_size_block, connection_size_word,
        connection_size_block, connection_size_word, inbox_row, 0, 1);
  } else {
    static_assert(PROTOCOL == Protocol::simple,
                  "the fused one-shot is Simple, simple_write or simple_read");
    constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
    constexpr int kPacks =
        (TILE_N + kPackElems * BLOCK_X - 1) / (kPackElems * BLOCK_X);
    constexpr int kSpan = kPacks * kPackElems * BLOCK_X;
    const int m = input_size_m, n = input_size_n;

    if constexpr (QUANTIZE_INT4)
      for (int t = blockIdx.x; t * TILE_M < m; t += gridDim.x)
        for (int c = 0; c < n; c += kSpan) {
          DTYPE mine[TILE_M][kPacks][kPackElems];
          load<WORLD, TILE_M, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
              input_ptr, 0, 0, input_stride_m, input_stride_n, 1, m, n, m, n,
              mine, t * TILE_M, c, 1);
          send<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE, true,
               Protocol::simple, Layout::tile, TAIL>(
              mine, staging_ptr_array, staging_ptr_array_size, rank, 1, 0, 0, n,
              1, 1, m, n, m, n, t * TILE_M, c, 1);
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
    barrier<WORLD>(
        signals_ptr_array, signals_ptr_array_size, signals_stride_parity,
        signals_stride_block, signals_stride_writer, signals_size_parity,
        signals_size_block, signals_size_writer, status_ptr,
        status_stride_field, status_size_field, rank, timeout_ticks, kFirst, 0);

    WEIGHT_DTYPE w[1][kPacks][kPackElems];
    if constexpr (!CHUNKED)
      load<1, 1, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
          weight_ptr, 0, 0, 0, weight_stride_n, 1, 1, n, 1, n, w, 0, 0, 1);

    for (int row = blockIdx.x * TILE_M; row < m; row += gridDim.x * TILE_M) {
      for (int c = 0; c < n; c += kSpan) {
        DTYPE copies[WORLD][TILE_M][kPacks][kPackElems];
        all_gather<AllGatherAlgorithm::tile_pull, Protocol::simple,
                   QUANTIZE_INT4, DTYPE, WORLD, TILE_M, TILE_N, BLOCK_X,
                   BLOCK_Y, WAVE_SIZE, PACK_BYTES, false, TAIL>(
            entries, 0, QUANTIZE_INT4 ? int64_t{n} : input_ptr_array_stride_m,
            QUANTIZE_INT4 ? int64_t{1} : input_ptr_array_stride_n, m, n, copies,
            0, 0, 0, 0, row, c, 0, n, 0, nullptr, 0, nullptr, 0, 0, 0, 0, 0, 0,
            0, nullptr, nullptr, 0, 0, 0, 0, 0, 0, nullptr, 0, 0, 0, 0, 0, 0, 0,
            nullptr, 0, 0, rank, 0);
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
              sum, output_ptr, 0, 0, output_stride_m, output_stride_n, 1, m, n,
              m, n, row, c, 1);
        else
          add_rms_norm_rows<RESIDUAL, BLOCK_X, BLOCK_Y, WAVE_SIZE>(
              row, sum, w, m, n, residual_ptr, residual_stride_m,
              residual_stride_n, output_ptr, output_stride_m, output_stride_n,
              residual_output_ptr, residual_output_stride_m,
              residual_output_stride_n, eps);
      }
      if constexpr (CHUNKED)
        add_rms_norm_rows<RESIDUAL, DTYPE, WEIGHT_DTYPE, TILE_M, TILE_N,
                          BLOCK_X, BLOCK_Y, WAVE_SIZE, PACK_BYTES, CHUNKED>(
            row, output_ptr, output_stride_m, output_stride_n, m, n,
            residual_ptr, residual_stride_m, residual_stride_n, weight_ptr,
            weight_stride_n, output_ptr, output_stride_m, output_stride_n,
            residual_output_ptr, residual_output_stride_m,
            residual_output_stride_n, eps);
    }

    barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                   signals_stride_parity, signals_stride_block,
                   signals_stride_writer, signals_size_parity,
                   signals_size_block, signals_size_writer, status_ptr,
                   status_stride_field, status_size_field, rank, timeout_ticks,
                   Until::read, 1);
  }
}

template <bool RESIDUAL, bool QUANTIZE_INT4, typename DTYPE,
          typename WEIGHT_DTYPE, int WORLD, int TILE_M, int TILE_N, int BLOCK_X,
          int BLOCK_Y, int WAVE_SIZE, int PACK_BYTES, bool CHUNKED, bool TAIL>
DINLINE void all_reduce_add_rms_norm_two_shot_pull_rows(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* const* input_ptr_array,
    int input_ptr_array_size, int64_t input_ptr_array_stride_m,
    int64_t input_ptr_array_stride_n, const DTYPE* residual_ptr,
    int64_t residual_stride_m, int64_t residual_stride_n,
    const WEIGHT_DTYPE* weight_ptr, int64_t weight_stride_n, DTYPE* output_ptr,
    int64_t output_stride_m, int64_t output_stride_n,
    DTYPE* residual_output_ptr, int64_t residual_output_stride_m,
    int64_t residual_output_stride_n, char* const* partials_ptr_array,
    int partials_ptr_array_size, int64_t partials_stride_m,
    int64_t partials_stride_n, int partials_size_m, int partials_size_n,
    char* const* staging_ptr_array, int staging_ptr_array_size,
    int64_t staging_stride_m, int64_t staging_stride_n, int staging_size_m,
    int staging_size_n, uint32_t* const* signals_ptr_array,
    int signals_ptr_array_size, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    uint32_t* status_ptr, int64_t status_stride_field, int status_size_field,
    int rank, uint64_t timeout_ticks, float eps) {
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
  constexpr int kPacks =
      (TILE_N + kPackElems * BLOCK_X - 1) / (kPackElems * BLOCK_X);
  constexpr int kSpan = kPacks * kPackElems * BLOCK_X;
  const int m = input_size_m, n = input_size_n;

  reduce_scatter<ReduceScatterAlgorithm::rows_pull, Protocol::simple,
                 QUANTIZE_INT4, WORLD, TILE_M, TILE_N, BLOCK_X, BLOCK_Y,
                 WAVE_SIZE, PACK_BYTES, true, TAIL>(
      input_ptr, input_stride_m, input_stride_n, m, n, input_ptr_array,
      input_ptr_array_size, input_ptr_array_stride_m, input_ptr_array_stride_n,
      staging_ptr_array, staging_ptr_array_size, staging_stride_m,
      staging_stride_n, staging_size_m, staging_size_n, partials_ptr_array,
      partials_ptr_array_size, partials_stride_m, partials_stride_n,
      partials_size_m, partials_size_n, signals_ptr_array,
      signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);

  const int slice = (m + WORLD - 1) / WORLD;

  WEIGHT_DTYPE w[1][kPacks][kPackElems];
  if constexpr (!CHUNKED)
    load<1, 1, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
        weight_ptr, 0, 0, 0, weight_stride_n, 1, 1, n, 1, n, w, 0, 0, 1);

  for (int row = blockIdx.x * TILE_M; row < m; row += gridDim.x * TILE_M) {
    for (int c = 0; c < n; c += kSpan) {
      DTYPE sum[TILE_M][kPacks][kPackElems];
      all_gather<AllGatherAlgorithm::rows_pull, Protocol::simple, QUANTIZE_INT4,
                 DTYPE, WORLD, TILE_M, TILE_N, BLOCK_X, BLOCK_Y, WAVE_SIZE,
                 PACK_BYTES, true, TAIL>(
          partials_ptr_array, partials_ptr_array_size, partials_stride_m,
          partials_stride_n, partials_size_m, partials_size_n, sum, 0, 0, m, n,
          row, c, 0, 0, slice, nullptr, 0, nullptr, 0, 0, 0, 0, 0, 0, 0,
          nullptr, nullptr, 0, 0, 0, 0, 0, 0, nullptr, 0, 0, 0, 0, 0, 0, 0,
          nullptr, 0, 0, rank, 0);

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
}

template <Protocol PROTOCOL, bool RESIDUAL, bool QUANTIZE_INT4, typename DTYPE,
          typename WEIGHT_DTYPE, int WORLD, int TILE_M, int TILE_N, int BLOCK_X,
          int BLOCK_Y, int WAVE_SIZE, int PACK_BYTES, bool CHUNKED, bool TAIL>
DINLINE void all_reduce_add_rms_norm_two_shot_pull_columns(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* const* input_ptr_array,
    int input_ptr_array_size, int64_t input_ptr_array_stride_m,
    int64_t input_ptr_array_stride_n, const DTYPE* residual_ptr,
    int64_t residual_stride_m, int64_t residual_stride_n,
    const WEIGHT_DTYPE* weight_ptr, int64_t weight_stride_n, DTYPE* output_ptr,
    int64_t output_stride_m, int64_t output_stride_n,
    DTYPE* residual_output_ptr, int64_t residual_output_stride_m,
    int64_t residual_output_stride_n, char* const* partials_ptr_array,
    int partials_ptr_array_size, int64_t partials_stride_m,
    int64_t partials_stride_n, int partials_size_m, int partials_size_n,
    char* const* staging_ptr_array, int staging_ptr_array_size,
    int64_t staging_stride_m, int64_t staging_stride_n, int staging_size_m,
    int staging_size_n, char* const* inbox_ptr_array, int inbox_ptr_array_size,
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
    uint64_t timeout_ticks, float eps) {
  if constexpr (PROTOCOL == Protocol::simple_write ||
                PROTOCOL == Protocol::simple_read) {
    // A row's columns are WORLD shares, share w rank w's. A chunk of the
    // shares is two rounds: the reduce-scatter (share w's chunk to rank w,
    // this rank's from every rank, summed) and the all-gather (the sum to
    // every rank, share w's chunk from rank w). Each lane group talks to one
    // peer, w; the inbox row is the block's. The gathered row is stored and
    // normalized there, as the chunked pull's.
    static_assert(
        !QUANTIZE_INT4 && TILE_M == 1,
        "the Simple protocols' fused two-shot is exact, a row a tile");
    constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
    constexpr int kLanes = BLOCK_X / WORLD;
    static_assert(kLanes * WORLD == BLOCK_X,
                  "a block is a group of threads a rank");
    constexpr int kChunkN = (TILE_N + kPackElems - 1) / kPackElems * kPackElems;
    constexpr int kPacks =
        (kChunkN + kPackElems * kLanes - 1) / (kPackElems * kLanes);
    const int m = input_size_m, n = input_size_n;
    const int slice = (n / kPackElems + WORLD - 1) / WORLD * kPackElems;
    const int w = (rank + static_cast<int>(threadIdx.x) / kLanes) % WORLD;
    const int inbox_row = static_cast<int>(blockIdx.x);
    // A share is slice long, the last one cut at the row's end.
    const int share_w = max(0, min(slice, n - w * slice));
    const int share_me = max(0, min(slice, n - rank * slice));

    Connection connection;
    load<WORLD, 1, BLOCK_X, WAVE_SIZE, false, Layout::block>(
        connection_ptr, 0, 0, connection_stride_block, connection_stride_word,
        1, connection_size_block, connection_size_word, connection_size_block,
        connection_size_word, connection, inbox_row, 0, 1);
    for (int row = blockIdx.x; row < m; row += gridDim.x) {
      for (int lo = 0; lo < slice; lo += kChunkN) {
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
  } else {
    static_assert(PROTOCOL == Protocol::simple,
                  "the fused two-shot is Simple, simple_write or simple_read");
    constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
    constexpr int kPacks =
        (TILE_N + kPackElems * BLOCK_X - 1) / (kPackElems * BLOCK_X);
    constexpr int kSpan = kPacks * kPackElems * BLOCK_X;
    const int m = input_size_m, n = input_size_n;

    reduce_scatter<ReduceScatterAlgorithm::columns_pull, Protocol::simple,
                   QUANTIZE_INT4, WORLD, TILE_M, TILE_N, BLOCK_X, BLOCK_Y,
                   WAVE_SIZE, PACK_BYTES, true, TAIL>(
        input_ptr, input_stride_m, input_stride_n, m, n, input_ptr_array,
        input_ptr_array_size, input_ptr_array_stride_m,
        input_ptr_array_stride_n, staging_ptr_array, staging_ptr_array_size,
        staging_stride_m, staging_stride_n, staging_size_m, staging_size_n,
        partials_ptr_array, partials_ptr_array_size, partials_stride_m,
        partials_stride_n, partials_size_m, partials_size_n, signals_ptr_array,
        signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks);

    const int slice = (n / kPackElems + WORLD - 1) / WORLD * kPackElems;

    WEIGHT_DTYPE w[1][kPacks][kPackElems];
    if constexpr (!CHUNKED)
      load<1, 1, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
          weight_ptr, 0, 0, 0, weight_stride_n, 1, 1, n, 1, n, w, 0, 0, 1);

    for (int row = blockIdx.x * TILE_M; row < m; row += gridDim.x * TILE_M) {
      for (int c = 0; c < n; c += kSpan) {
        DTYPE sum[TILE_M][kPacks][kPackElems];
        all_gather<AllGatherAlgorithm::columns_pull, Protocol::simple,
                   QUANTIZE_INT4, DTYPE, WORLD, TILE_M, TILE_N, BLOCK_X,
                   BLOCK_Y, WAVE_SIZE, PACK_BYTES, true, TAIL>(
            partials_ptr_array, partials_ptr_array_size, partials_stride_m,
            partials_stride_n, partials_size_m, partials_size_n, sum, 0, 0, m,
            n, row, c, 0, 0, slice, nullptr, 0, nullptr, 0, 0, 0, 0, 0, 0, 0,
            nullptr, nullptr, 0, 0, 0, 0, 0, 0, nullptr, 0, 0, 0, 0, 0, 0, 0,
            nullptr, 0, 0, rank, 0);

        if constexpr (CHUNKED)
          store<1, TILE_M, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
              sum, output_ptr, 0, 0, output_stride_m, output_stride_n, 1, m, n,
              m, n, row, c, 1);
        else
          add_rms_norm_rows<RESIDUAL, BLOCK_X, BLOCK_Y, WAVE_SIZE>(
              row, sum, w, m, n, residual_ptr, residual_stride_m,
              residual_stride_n, output_ptr, output_stride_m, output_stride_n,
              residual_output_ptr, residual_output_stride_m,
              residual_output_stride_n, eps);
      }
      if constexpr (CHUNKED)
        add_rms_norm_rows<RESIDUAL, DTYPE, WEIGHT_DTYPE, TILE_M, TILE_N,
                          BLOCK_X, BLOCK_Y, WAVE_SIZE, PACK_BYTES, CHUNKED>(
            row, output_ptr, output_stride_m, output_stride_n, m, n,
            residual_ptr, residual_stride_m, residual_stride_n, weight_ptr,
            weight_stride_n, output_ptr, output_stride_m, output_stride_n,
            residual_output_ptr, residual_output_stride_m,
            residual_output_stride_n, eps);
    }
  }
}

template <bool RESIDUAL, bool QUANTIZE_INT4, typename DTYPE,
          typename WEIGHT_DTYPE, int WORLD, int TILE_M, int TILE_N, int BLOCK_X,
          int BLOCK_Y, int WAVE_SIZE, int PACK_BYTES, bool CHUNKED, bool TAIL>
DINLINE void all_reduce_add_rms_norm_two_shot_push(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* const* input_ptr_array,
    int input_ptr_array_size, int64_t input_ptr_array_stride_m,
    int64_t input_ptr_array_stride_n, const DTYPE* residual_ptr,
    int64_t residual_stride_m, int64_t residual_stride_n,
    const WEIGHT_DTYPE* weight_ptr, int64_t weight_stride_n, DTYPE* output_ptr,
    int64_t output_stride_m, int64_t output_stride_n,
    DTYPE* residual_output_ptr, int64_t residual_output_stride_m,
    int64_t residual_output_stride_n, char* const* partials_ptr_array,
    int partials_ptr_array_size, int64_t partials_stride_m,
    int64_t partials_stride_n, int partials_size_m, int partials_size_n,
    char* const* staging_ptr_array, int staging_ptr_array_size,
    int64_t staging_stride_m, int64_t staging_stride_n, int staging_size_m,
    int staging_size_n, uint32_t* const* signals_ptr_array,
    int signals_ptr_array_size, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    uint32_t* status_ptr, int64_t status_stride_field, int status_size_field,
    int rank, uint64_t timeout_ticks, float eps) {
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
  constexpr int kPacks =
      (TILE_N + kPackElems * BLOCK_X - 1) / (kPackElems * BLOCK_X);
  constexpr int kSpan = kPacks * kPackElems * BLOCK_X;
  const int m = input_size_m, n = input_size_n;

  reduce_scatter<ReduceScatterAlgorithm::columns_push, Protocol::simple,
                 QUANTIZE_INT4, WORLD, TILE_M, TILE_N, BLOCK_X, BLOCK_Y,
                 WAVE_SIZE, PACK_BYTES, true, TAIL>(
      input_ptr, input_stride_m, input_stride_n, m, n, input_ptr_array,
      input_ptr_array_size, input_ptr_array_stride_m, input_ptr_array_stride_n,
      staging_ptr_array, staging_ptr_array_size, staging_stride_m,
      staging_stride_n, staging_size_m, staging_size_n, partials_ptr_array,
      partials_ptr_array_size, partials_stride_m, partials_stride_n,
      partials_size_m, partials_size_n, signals_ptr_array,
      signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);

  WEIGHT_DTYPE w[1][kPacks][kPackElems];
  if constexpr (!CHUNKED)
    load<1, 1, BLOCK_X, WAVE_SIZE, false, Layout::tile>(
        weight_ptr, 0, 0, 0, weight_stride_n, 1, 1, n, 1, n, w, 0, 0, 1);

  for (int row = blockIdx.x * TILE_M; row < m; row += gridDim.x * TILE_M) {
    for (int c = 0; c < n; c += kSpan) {
      DTYPE sum[TILE_M][kPacks][kPackElems];
      recv<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE, QUANTIZE_INT4,
           Protocol::simple, Layout::tile, TAIL>(
          partials_ptr_array, partials_ptr_array_size, rank, 0, 0,
          partials_stride_m, partials_stride_n, 1, partials_size_m,
          partials_size_n, m, n, sum, row, c, 1);

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
        const DTYPE* const* __restrict__ input_ptr_array,
        int input_ptr_array_size, int64_t input_ptr_array_stride_m,
        int64_t input_ptr_array_stride_n, const DTYPE* residual_ptr,
        int64_t residual_stride_m, int64_t residual_stride_n,
        int residual_size_m, int residual_size_n,
        const WEIGHT_DTYPE* __restrict__ weight_ptr, int64_t weight_stride_n,
        int weight_size_n, DTYPE* output_ptr, int64_t output_stride_m,
        int64_t output_stride_n, int output_size_m, int output_size_n,
        DTYPE* residual_output_ptr, int64_t residual_output_stride_m,
        int64_t residual_output_stride_n, int residual_output_size_m,
        int residual_output_size_n,
        char* const* __restrict__ partials_ptr_array,
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
        uint64_t timeout_ticks, float eps) {
  // Its stages synchronize with barriers until they move to a protocol; the
  // one-shot has.
  static_assert(
      PROTOCOL == Protocol::simple ||
          ALGORITHM == AllReduceAddRmsNormKernelAlgorithm::one_shot_pull ||
          ALGORITHM ==
              AllReduceAddRmsNormKernelAlgorithm::two_shot_pull_columns,
      "the fused add_rms_norm's other stages are Simple's still");
  if constexpr (ALGORITHM == AllReduceAddRmsNormKernelAlgorithm::one_shot_pull)
    all_reduce_add_rms_norm_one_shot_pull<
        PROTOCOL, RESIDUAL, QUANTIZE_INT4, DTYPE, WEIGHT_DTYPE, WORLD, TILE_M,
        TILE_N, BLOCK_X, BLOCK_Y, WAVE_SIZE, PACK_BYTES, CHUNKED, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m,
        input_ptr_array_stride_n, residual_ptr, residual_stride_m,
        residual_stride_n, weight_ptr, weight_stride_n, output_ptr,
        output_stride_m, output_stride_n, residual_output_ptr,
        residual_output_stride_m, residual_output_stride_n, staging_ptr_array,
        staging_ptr_array_size, inbox_ptr_array, inbox_ptr_array_size,
        inbox_stride_slot, inbox_stride_m, inbox_stride_n, inbox_size_slot,
        inbox_size_m, inbox_size_n, connection_ptr, connection_stride_block,
        connection_stride_word, connection_size_block, connection_size_word,
        counters_ptr_array, counters_ptr_array_size, counters_stride_kind,
        counters_stride_block, counters_stride_peer, counters_size_block,
        counters_size_peer, signals_ptr_array, signals_ptr_array_size,
        signals_stride_parity, signals_stride_block, signals_stride_writer,
        signals_size_parity, signals_size_block, signals_size_writer,
        status_ptr, status_stride_field, status_size_field, rank, timeout_ticks,
        eps);
  else if constexpr (ALGORITHM ==
                     AllReduceAddRmsNormKernelAlgorithm::two_shot_pull_rows)
    all_reduce_add_rms_norm_two_shot_pull_rows<
        RESIDUAL, QUANTIZE_INT4, DTYPE, WEIGHT_DTYPE, WORLD, TILE_M, TILE_N,
        BLOCK_X, BLOCK_Y, WAVE_SIZE, PACK_BYTES, CHUNKED, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m,
        input_ptr_array_stride_n, residual_ptr, residual_stride_m,
        residual_stride_n, weight_ptr, weight_stride_n, output_ptr,
        output_stride_m, output_stride_n, residual_output_ptr,
        residual_output_stride_m, residual_output_stride_n, partials_ptr_array,
        partials_ptr_array_size, partials_stride_m, partials_stride_n,
        partials_size_m, partials_size_n, staging_ptr_array,
        staging_ptr_array_size, staging_stride_m, staging_stride_n,
        staging_size_m, staging_size_n, signals_ptr_array,
        signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks, eps);
  else if constexpr (ALGORITHM ==
                     AllReduceAddRmsNormKernelAlgorithm::two_shot_pull_columns)
    all_reduce_add_rms_norm_two_shot_pull_columns<
        PROTOCOL, RESIDUAL, QUANTIZE_INT4, DTYPE, WEIGHT_DTYPE, WORLD, TILE_M,
        TILE_N, BLOCK_X, BLOCK_Y, WAVE_SIZE, PACK_BYTES, CHUNKED, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m,
        input_ptr_array_stride_n, residual_ptr, residual_stride_m,
        residual_stride_n, weight_ptr, weight_stride_n, output_ptr,
        output_stride_m, output_stride_n, residual_output_ptr,
        residual_output_stride_m, residual_output_stride_n, partials_ptr_array,
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
        eps);
  else if constexpr (ALGORITHM ==
                     AllReduceAddRmsNormKernelAlgorithm::two_shot_push)
    all_reduce_add_rms_norm_two_shot_push<
        RESIDUAL, QUANTIZE_INT4, DTYPE, WEIGHT_DTYPE, WORLD, TILE_M, TILE_N,
        BLOCK_X, BLOCK_Y, WAVE_SIZE, PACK_BYTES, CHUNKED, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m,
        input_ptr_array_stride_n, residual_ptr, residual_stride_m,
        residual_stride_n, weight_ptr, weight_stride_n, output_ptr,
        output_stride_m, output_stride_n, residual_output_ptr,
        residual_output_stride_m, residual_output_stride_n, partials_ptr_array,
        partials_ptr_array_size, partials_stride_m, partials_stride_n,
        partials_size_m, partials_size_n, staging_ptr_array,
        staging_ptr_array_size, staging_stride_m, staging_stride_n,
        staging_size_m, staging_size_n, signals_ptr_array,
        signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks, eps);
}

}  // namespace hip_custom_communicator
