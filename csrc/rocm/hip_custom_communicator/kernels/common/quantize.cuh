#pragma once

// quantize: values to INT4 packs, a group's words and its fp16 scale.

#include "types.cuh"

namespace hip_custom_communicator {

template <int LOWEST = kInt4Lowest, int HIGHEST = kInt4Highest, int PACKS,
          typename T, int P>
DINLINE void quantize(const T (&input)[PACKS][P],
                      uint32_t (&encoded)[PACKS][2]) {
  constexpr int THREAD_PACKS = PACKS;
  uint32_t words[PACKS];
  float scale;
  static_assert(P * 4 == 32, "a pack is one word of 8 values");
  static_assert(THREAD_PACKS == 1 || THREAD_PACKS == kInt4GroupPacks,
                "a pack a thread (its quad's group) or a group a thread");

  float x[THREAD_PACKS][P];
  uint32_t key = 0;
#pragma unroll
  for (int k = 0; k < THREAD_PACKS; ++k)
#pragma unroll
    for (int j = 0; j < P; ++j) {
      x[k][j] = static_cast<float>(input[k][j]);
      const uint32_t bits = __builtin_bit_cast(uint32_t, x[k][j]);
      key = max(key, (bits << 1) | (bits >> 31));
    }
  if constexpr (THREAD_PACKS == 1) {
    key = max(key, static_cast<uint32_t>(__builtin_amdgcn_mov_dpp(
                       static_cast<int>(key), 0xb1, 0xf, 0xf, false)));
    key = max(key, static_cast<uint32_t>(__builtin_amdgcn_mov_dpp(
                       static_cast<int>(key), 0x4e, 0xf, 0xf, false)));
  }
  const float extreme = __builtin_bit_cast(float, (key >> 1) | (key << 31));

  // The group's signed extreme lands on the lowest level, so it is never
  // clipped.
  scale = static_cast<float>(static_cast<__fp16>(
      fminf(fmaxf(extreme * (1.0f / LOWEST), -65504.0f), 65504.0f)));
  const float inv = scale != 0.0f ? 1.0f / scale : 0.0f;

  const auto word = [](const float (&v)[P], float inverse) {
    constexpr float kLowest = 12582912.0f;
    constexpr float kBias = -LOWEST;
    constexpr float kLevels = HIGHEST - LOWEST;
    uint32_t q[P];
#pragma unroll
    for (int p = 0; p < P / 2; ++p) {
#pragma clang fp contract(off)
      const f32x2 y = f32x2{v[2 * p], v[2 * p + 1]} * f32x2{inverse, inverse} +
                      f32x2{kLowest + kBias, kLowest + kBias};
#pragma unroll
      for (int e = 0; e < 2; ++e)
        q[2 * p + e] = __builtin_bit_cast(
            uint32_t,
            __builtin_amdgcn_fmed3f(y[e], kLowest, kLowest + kLevels));
    }

    const auto nibbles = [](uint32_t high, uint32_t low) {
      uint32_t byte;

      asm("v_lshl_or_b32 %0, %1, 4, %2" : "=v"(byte) : "v"(high), "v"(low));
      return byte;
    };
    const uint32_t even_low = nibbles(q[2], q[0]),
                   even_high = nibbles(q[6], q[4]);
    const uint32_t odd_low = nibbles(q[3], q[1]),
                   odd_high = nibbles(q[7], q[5]);
    return __builtin_amdgcn_perm(
        __builtin_amdgcn_perm(odd_high, odd_low, 0x0c0c0400u),
        __builtin_amdgcn_perm(even_high, even_low, 0x0c0c0400u), 0x05040100u);
  };
#pragma unroll
  for (int k = 0; k < THREAD_PACKS; ++k) words[k] = word(x[k], inv);

  const uint32_t bits =
      __builtin_bit_cast(uint16_t, static_cast<__fp16>(scale));
#pragma unroll
  for (int k = 0; k < PACKS; ++k) {
    encoded[k][0] = words[k];
    encoded[k][1] = bits;
  }
}

template <typename T, int K, int P>
DINLINE void quantize(const T (&input)[1][K][P], uint32_t (&encoded)[1][K][2]) {
  static_assert(K % kInt4GroupPacks == 0, "a thread holds whole groups");
#pragma unroll
  for (int jg = 0; jg < K / kInt4GroupPacks; ++jg)
    quantize(reinterpret_cast<const T(&)[kInt4GroupPacks][P]>(
                 input[0][jg * kInt4GroupPacks]),
             reinterpret_cast<uint32_t (&)[kInt4GroupPacks][2]>(
                 encoded[0][jg * kInt4GroupPacks]));
}

}  // namespace hip_custom_communicator
