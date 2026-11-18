#pragma once

// reduce: a tile's registers summed or maxed across the block.

#include "types.cuh"

namespace hip_custom_communicator {

namespace ReduceAxis {
constexpr int all = 0;
constexpr int m = 1;
}  // namespace ReduceAxis

template <int TILE_M, int BLOCK_X, int WAVE_SIZE, int AXIS, typename T,
          typename U, int K, int P>
DINLINE void reduce(const T (&input)[K][P], U (&output)[K][P]) {
  static_assert(AXIS == ReduceAxis::all || AXIS == ReduceAxis::m,
                "AXIS is all or m");

#if defined(__AMDGCN_WAVEFRONT_SIZE)
  static_assert(__AMDGCN_WAVEFRONT_SIZE == WAVE_SIZE,
                "the compiler's wave is not the one handed");
#endif

  constexpr int kThreadsM = TILE_M;
  constexpr int kThreadsN = BLOCK_X / TILE_M;
  static_assert(kThreadsM * kThreadsN == BLOCK_X,
                "the block is TILE_M whole rows of threads");
  if constexpr (AXIS == ReduceAxis::all) {
    static_assert(TILE_M == 1, "AXIS all is one row of the block's threads");
    constexpr int kValues = K * P;
    constexpr int kWaves = (kThreadsN + WAVE_SIZE - 1) / WAVE_SIZE;
    __shared__ float partial[kValues][kWaves];
    const int lane = threadIdx.x % WAVE_SIZE;
    const int wave = threadIdx.x / WAVE_SIZE;
    float v[kValues];
#pragma unroll
    for (int n = 0; n < kValues; ++n)
      v[n] = static_cast<float>(input[n / P][n % P]);

    static_assert(WAVE_SIZE == 64, "the DPP sequence is for 64-lane waves");
    const auto bits = [](float x) { return __builtin_bit_cast(int, x); };
    const auto real = [](int x) { return __builtin_bit_cast(float, x); };
    constexpr int kIdentity = 0;
#pragma unroll
    for (int n = 0; n < kValues; ++n) {
      float x = v[n];

      x = real(__builtin_amdgcn_mov_dpp(bits(x), 0xb1, 0xf, 0xf, false)) + x;
      x = real(__builtin_amdgcn_mov_dpp(bits(x), 0x4e, 0xf, 0xf, false)) + x;
      x = real(__builtin_amdgcn_mov_dpp(bits(x), 0x141, 0xf, 0xf, false)) + x;
      x = real(__builtin_amdgcn_mov_dpp(bits(x), 0x140, 0xf, 0xf, false)) + x;

      x = real(__builtin_amdgcn_update_dpp(kIdentity, bits(x), 0x142, 0xa, 0xf,
                                           false)) +
          x;
      x = real(__builtin_amdgcn_update_dpp(kIdentity, bits(x), 0x143, 0xc, 0xf,
                                           false)) +
          x;
      v[n] = real(__builtin_amdgcn_readlane(bits(x), 63));
    }

    if (lane == 0) {
#pragma unroll
      for (int n = 0; n < kValues; ++n) partial[n][wave] = v[n];
    }
    __syncthreads();
#pragma unroll
    for (int n = 0; n < kValues; ++n) {
      float x = partial[n][0];
#pragma unroll
      for (int w = 1; w < kWaves; ++w) x = x + partial[n][w];
      output[n / P][n % P] = static_cast<U>(x);
    }
    __syncthreads();
  } else {
    using V = vec<T, P>;
    __shared__ V part[kThreadsM][K][kThreadsN];
    const int mine = static_cast<int>(threadIdx.x) % kThreadsN;
#pragma unroll
    for (int k = 0; k < K; ++k)
      __builtin_memcpy(
          &part[static_cast<int>(threadIdx.x) / kThreadsN][k][mine], input[k],
          sizeof(V));
    __syncthreads();

    if (kThreadsM == 1 || static_cast<int>(threadIdx.x) < kThreadsN) {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        float x[P];
        V p = part[0][k][mine];
#pragma unroll
        for (int j = 0; j < P; ++j)
          x[j] = static_cast<float>(reinterpret_cast<const T*>(&p)[j]);
#pragma unroll
        for (int r = 1; r < kThreadsM; ++r) {
          p = part[r][k][mine];
#pragma unroll
          for (int j = 0; j < P; ++j)
            x[j] = x[j] + static_cast<float>(reinterpret_cast<const T*>(&p)[j]);
        }
#pragma unroll
        for (int j = 0; j < P; ++j) output[k][j] = static_cast<U>(x[j]);
      }
    }
    __syncthreads();
  }
}

}  // namespace hip_custom_communicator
