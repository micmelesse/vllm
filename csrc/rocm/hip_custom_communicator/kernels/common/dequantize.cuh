#pragma once

// dequantize: an INT4 pack, or several ranks' packs summed, back to values.

#include "types.cuh"

namespace hip_custom_communicator {

// An encoded INT4 pack is a word of 8 values and its group's fp16 scale.
// Only quantize and dequantize read inside it.
template <int LOWEST = kInt4Lowest, int HIGHEST = kInt4Highest, typename U,
          int P>
DINLINE void dequantize(const uint32_t (&encoded)[2], U (&output)[P]) {
  const uint32_t word = encoded[0];
  const float scale = static_cast<float>(
      __builtin_bit_cast(__fp16, static_cast<uint16_t>(encoded[1])));
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
      ((encoded[1] >> 16) != 0 || !__builtin_isfinite(scale)))
    __builtin_trap();
  static_assert(P == 8, "a word is 8 values");
  static_assert(std::is_same_v<U, float> || sizeof(U) == 2,
                "fp32 terms or a 16-bit dtype");
  const float scale16 = scale * 0.0625f;

  const float bias = scale * -(1024.0f - LOWEST);
  const float bias16 = scale * -(64.0f - LOWEST);
  float term[P];
  uint32_t p0, p1, p2, p3;
  // 0x6400 is 1024 in fp16. OR-ing a nibble into it gives 1024 + q exactly.
  asm volatile(
      "v_and_or_b32 %[p0], %[w], %[m0], %[magic]\n\t"
      "v_and_or_b32 %[p1], %[w], %[m1], %[magic]\n\t"
      "v_lshrrev_b32 %[p2], 8, %[w]\n\t"
      "v_and_or_b32 %[p3], %[p2], %[m1], %[magic]\n\t"
      "v_and_or_b32 %[p2], %[p2], %[m0], %[magic]\n\t"
      "v_fma_mix_f32 %[t0], %[p0], %[s], %[b] op_sel_hi:[1,0,0]\n\t"
      "v_fma_mix_f32 %[t1], %[p0], %[s], %[b] op_sel:[1,0,0] "
      "op_sel_hi:[1,0,0]\n\t"
      "v_fma_mix_f32 %[t2], %[p1], %[s16], %[b16] op_sel_hi:[1,0,0]\n\t"
      "v_fma_mix_f32 %[t3], %[p1], %[s16], %[b16] op_sel:[1,0,0] "
      "op_sel_hi:[1,0,0]\n\t"
      "v_fma_mix_f32 %[t4], %[p2], %[s], %[b] op_sel_hi:[1,0,0]\n\t"
      "v_fma_mix_f32 %[t5], %[p2], %[s], %[b] op_sel:[1,0,0] "
      "op_sel_hi:[1,0,0]\n\t"
      "v_fma_mix_f32 %[t6], %[p3], %[s16], %[b16] op_sel_hi:[1,0,0]\n\t"
      "v_fma_mix_f32 %[t7], %[p3], %[s16], %[b16] op_sel:[1,0,0] "
      "op_sel_hi:[1,0,0]"
      : [p0] "=&v"(p0), [p1] "=&v"(p1), [p2] "=&v"(p2), [p3] "=&v"(p3),
        [t0] "=&v"(term[0]), [t1] "=&v"(term[1]), [t2] "=&v"(term[2]),
        [t3] "=&v"(term[3]), [t4] "=&v"(term[4]), [t5] "=&v"(term[5]),
        [t6] "=&v"(term[6]), [t7] "=&v"(term[7])
      : [w] "v"(word), [m0] "s"(0x000F000Fu), [m1] "s"(0x00F000F0u),
        [magic] "v"(0x64006400u), [s] "v"(scale), [b] "v"(bias),
        [s16] "v"(scale16), [b16] "v"(bias16));
  if constexpr (std::is_same_v<U, float>) {
#pragma unroll
    for (int j = 0; j < P; ++j) output[j] = term[j];
  } else {
    u32x4 packed;
#pragma unroll
    for (int w = 0; w < P / 2; ++w) {
      const float a = term[2 * w], b = term[2 * w + 1];
      if constexpr (kBf16<U>)
        packed[w] = __builtin_bit_cast(
            uint32_t, __builtin_convertvector(f32x2{a, b}, bf16x2));
      else
        packed[w] = static_cast<uint32_t>(
                        __builtin_bit_cast(uint16_t, static_cast<U>(a))) |
                    static_cast<uint32_t>(
                        __builtin_bit_cast(uint16_t, static_cast<U>(b)))
                        << 16;
    }
    __builtin_memcpy(output, &packed, sizeof(packed));
  }
}

template <typename DTYPE, int LOWEST = kInt4Lowest, int HIGHEST = kInt4Highest,
          int SLOTS, int K, int P>
DINLINE void dequantize(const uint32_t (&encoded)[SLOTS][1][K][2],
                        float (&sum)[1][K][P]) {
  const auto& words = encoded;

  const auto scale_of = [&](int w, int k) {
    if constexpr (K == kInt4GroupPacks)
      return words[w][0][0][1];
    else
      return words[w][0][k / kInt4GroupPacks * kInt4GroupPacks][1];
  };
#pragma unroll
  for (int k = 0; k < K; ++k) {
#pragma unroll
    for (int j = 0; j < P; ++j) sum[0][k][j] = 0.0f;
    int w = 0;
    if constexpr (kBf16<DTYPE>) {
      const bf16x2 ones = __builtin_bit_cast(bf16x2, 0x3f803f80u);
#pragma unroll
      for (; w + 1 < SLOTS; w += 2) {
        float a[P], b[P];
        const uint32_t pack_a[2] = {words[w][0][k][0], scale_of(w, k)};
        dequantize<LOWEST, HIGHEST>(pack_a, a);
        const uint32_t pack_b[2] = {words[w + 1][0][k][0], scale_of(w + 1, k)};
        dequantize<LOWEST, HIGHEST>(pack_b, b);
#pragma unroll
        for (int j = 0; j < P; ++j)
          sum[0][k][j] = __builtin_amdgcn_fdot2_f32_bf16(
              __builtin_convertvector(f32x2{a[j], b[j]}, bf16x2), ones,
              sum[0][k][j], false);

#pragma unroll
        for (int j = 0; j < P; ++j) asm volatile("" ::"v"(sum[0][k][j]));
      }
    }
#pragma unroll
    for (; w < SLOTS; ++w) {
      float a[P];
      const uint32_t pack_a[2] = {words[w][0][k][0], scale_of(w, k)};
      dequantize<LOWEST, HIGHEST>(pack_a, a);
#pragma unroll
      for (int j = 0; j < P; ++j)
        sum[0][k][j] += static_cast<float>(static_cast<DTYPE>(a[j]));
#pragma unroll
      for (int j = 0; j < P; ++j) asm volatile("" ::"v"(sum[0][k][j]));
    }
  }
}

}  // namespace hip_custom_communicator
