#pragma once

// send: registers into a peer's memory, by its protocol (send_simple, send_ll,
// send_lamport).

#include "quantize.cuh"
#include "types.cuh"

namespace hip_custom_communicator {

namespace internal {

// A Simple send: a store into the peer's memory, in any layout.
template <int WORLD, int TILE_M, int BLOCK_X, int WAVE_SIZE, bool QUANTIZE_INT4,
          Layout LAYOUT, bool TAIL, typename INPUT, typename PTR>
DINLINE void send_simple(const INPUT& input, PTR const* output_ptr_array,
                         int output_ptr_array_size, int output_rank,
                         int output_slot, int64_t output_stride_slot,
                         int64_t output_stride_m, int64_t output_stride_n,
                         int output_size_slot, int output_size_m,
                         int output_size_n, int clip_m, int clip_n, int offs_m,
                         int offs_n, int row_step) {
  constexpr int kPacks = std::extent_v<INPUT, 1>;
  if constexpr ((LAYOUT == Layout::group_run || LAYOUT == Layout::group_flat) &&
                kPacks != kInt4GroupPacks) {
    static_assert(std::extent_v<INPUT, 0> == 1 && kPacks % kInt4GroupPacks == 0,
                  "a thread holds whole groups");
    constexpr int kPackElems = std::extent_v<INPUT, 2>;
    using U = std::remove_all_extents_t<INPUT>;
#pragma unroll
    for (int jg = 0; jg < kPacks / kInt4GroupPacks; ++jg) {
      U group[1][kInt4GroupPacks][kPackElems];
#pragma unroll
      for (int k = 0; k < kInt4GroupPacks; ++k)
#pragma unroll
        for (int j = 0; j < kPackElems; ++j)
          group[0][k][j] = input[0][jg * kInt4GroupPacks + k][j];
      send_simple<WORLD, TILE_M, BLOCK_X, WAVE_SIZE, QUANTIZE_INT4, LAYOUT,
                  TAIL>(
          group, output_ptr_array, output_ptr_array_size, output_rank,
          output_slot, output_stride_slot, output_stride_m, output_stride_n,
          output_size_slot, output_size_m, output_size_n, clip_m, clip_n,
          offs_m, offs_n + jg * (BLOCK_X / TILE_M) * kInt4Group, row_step);
    }
  } else {
    const auto* const entries =
        (decltype(+output_ptr_array[0]) const*)output_ptr_array;
    const int first = __builtin_amdgcn_readfirstlane(output_rank);
    if (__builtin_amdgcn_ballot_w64(output_rank != first) == 0) {
      // One peer for the wave: its memory's address is scalar.
      void* const output = ptr_array_at(entries, output_ptr_array_size, first);
#if defined(__AMDGCN_WAVEFRONT_SIZE)
      static_assert(__AMDGCN_WAVEFRONT_SIZE == WAVE_SIZE,
                    "the compiler's wave is not the one handed");
#endif
      constexpr int kRows = std::extent_v<INPUT, 0>;
      constexpr int kPacks = std::extent_v<INPUT, 1>;
      constexpr int kPackElems = std::extent_v<INPUT, 2>;
      using U = std::remove_all_extents_t<INPUT>;
      constexpr bool kRaw = std::is_same_v<U, uint32_t>;
      static_assert(
          !kRaw || (QUANTIZE_INT4 && kPackElems == 2),
          "uint32_t registers are an INT4 payload's words and scales");
      constexpr int kThreadsM = TILE_M / kRows;
      constexpr int kThreadsN = BLOCK_X / kThreadsM;
      static_assert(kThreadsM * kRows == TILE_M,
                    "a tile is whole rows a thread");
      static_assert(kThreadsM * kThreadsN == BLOCK_X,
                    "the layout is the block");
      using V = vec<U, kPackElems>;
      using E = std::conditional_t<!QUANTIZE_INT4, U, uint8_t>;
      if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
          (output_slot < 0 || output_slot >= output_size_slot))
        __builtin_trap();
      [[maybe_unused]] E* const at_copy =
          reinterpret_cast<E*>(output) + output_slot * output_stride_slot;
      const int tid = static_cast<int>(threadIdx.x);
      const auto row_at = [&](int i) {
        return offs_m + (kThreadsM == 1 ? i : tid / kThreadsN + i * kThreadsM) *
                            row_step;
      };
      const auto col_at = [&](int k) {
        return offs_n / kPackElems + tid % kThreadsN + k * kThreadsN;
      };
      [[maybe_unused]] const auto element_at = [](auto* ptr, int64_t stride_m,
                                                  int64_t stride_n, int64_t e,
                                                  int n) {
        if (stride_m == int64_t{n} * stride_n) return ptr + e * stride_n;
        const int64_t row =
            e < (int64_t{1} << 31)
                ? int64_t{static_cast<uint32_t>(e) / static_cast<uint32_t>(n)}
                : e / n;
        return ptr + row * stride_m + (e - row * n) * stride_n;
      };
      constexpr bool kGroup =
          LAYOUT == Layout::group_run || LAYOUT == Layout::group_flat;
      constexpr bool kRun =
          LAYOUT == Layout::run || LAYOUT == Layout::group_run;
      if constexpr (LAYOUT != Layout::tile && !kGroup) {
        static_assert(!QUANTIZE_INT4 && kRows == 1,
                      "the flat walk writes one exact copy a row of threads");

        constexpr int kLanes = BLOCK_X / WORLD;
        static_assert(kLanes * WORLD == BLOCK_X,
                      "a block is a group of threads a rank");
        U* const to =
            reinterpret_cast<U*>(output) + int64_t{offs_m} * output_stride_m;
        const int64_t first = int64_t{output_slot} * output_stride_slot;
        U* const run = to + first * output_stride_n;
#pragma unroll
        for (int k = 0; k < kPacks; ++k) {
          const int c = offs_n / kPackElems + tid % kLanes + k * kLanes;
          if (c >= clip_n / kPackElems) continue;
          V* p;
          if constexpr (kRun) {
            uint32_t offset = static_cast<uint32_t>(c) * uint32_t{sizeof(V)};
            asm volatile("" : "+v"(offset));
            p = reinterpret_cast<V*>(reinterpret_cast<char*>(run) + offset);
          } else {
            p = reinterpret_cast<V*>(
                element_at(to, output_stride_m, output_stride_n,
                           first + int64_t{c} * kPackElems, output_size_n));
          }
          V v;
          __builtin_memcpy(&v, input[0][k], sizeof(v));
          {
            using W = std::conditional_t<
                sizeof(V) == 16, u32x4,
                std::conditional_t<
                    sizeof(V) == 8, u32x2,
                    std::conditional_t<sizeof(V) == 4, uint32_t, uint16_t>>>;
            W raw;
            __builtin_memcpy(&raw, &v, sizeof(V));
            *(__attribute__((address_space(1))) W*)(p) = raw;
          }
        }
      } else if constexpr (kGroup) {
        static_assert(
            kRows == 1 && kPacks == kInt4GroupPacks,
            "a group a thread here: several are load's, recv's, store's "
            "and send's loop");
        const int g = offs_n / kInt4Group + tid % kThreadsN;
        if constexpr (!QUANTIZE_INT4) {
          if constexpr (TAIL) {
            if (g >= int4_groups(clip_n)) return;
            U* const to = reinterpret_cast<U*>(output) +
                          int64_t{offs_m} * output_stride_m;
            const int64_t first = int64_t{output_slot} * output_stride_slot;

            [[maybe_unused]] V* group = nullptr;
            if constexpr (kRun) {
              uint32_t offset =
                  static_cast<uint32_t>(g) * (kInt4Group * sizeof(U));
              asm volatile("" : "+v"(offset));
              group = reinterpret_cast<V*>(
                  reinterpret_cast<char*>(to + first * output_stride_n) +
                  offset);
            }
            const auto pack_at = [&](int k) -> V* {
              if constexpr (kRun)
                return group + k;
              else
                return reinterpret_cast<V*>(element_at(
                    to, output_stride_m, output_stride_n,
                    first + int64_t{g} * kInt4Group + int64_t{k} * kPackElems,
                    output_size_n));
            };
#pragma unroll
            for (int k = 0; k < kInt4GroupPacks; ++k) {
              if (g * kInt4GroupPacks + k >= clip_n / kPackElems) continue;
              u32x4 raw;
              __builtin_memcpy(&raw, input[0][k], sizeof(raw));
              *(global_u32x4*)(pack_at(k)) = raw;
            }
          } else {
            if (g >= clip_n / kInt4Group) return;
            U* const to = reinterpret_cast<U*>(output) +
                          int64_t{offs_m} * output_stride_m;
            const int64_t first = int64_t{output_slot} * output_stride_slot;
            V* group;
            if constexpr (kRun) {
              uint32_t offset =
                  static_cast<uint32_t>(g) * (kInt4Group * sizeof(U));
              asm volatile("" : "+v"(offset));
              group = reinterpret_cast<V*>(
                  reinterpret_cast<char*>(to + first * output_stride_n) +
                  offset);
            } else
              group = reinterpret_cast<V*>(
                  element_at(to, output_stride_m, output_stride_n,
                             first + int64_t{g} * kInt4Group, output_size_n));
#pragma unroll
            for (int k = 0; k < kInt4GroupPacks; ++k) {
              u32x4 raw;
              __builtin_memcpy(&raw, input[0][k], sizeof(raw));
              *(global_u32x4*)(group + k) = raw;
            }
          }
        } else {
          static_assert(kRun, "an INT4 payload is dense");
          uint32_t encoded[kInt4GroupPacks][2];
          if constexpr (kRaw) {
#pragma unroll
            for (int k = 0; k < kInt4GroupPacks; ++k) {
              encoded[k][0] = input[0][k][0];
              encoded[k][1] = input[0][k][1];
            }
          } else {
            quantize(input[0], encoded);
          }
          if (g < (TAIL ? static_cast<int>(int4_groups(clip_n))
                        : clip_n / kInt4Group)) {
            const int64_t slot = int64_t{output_slot} * output_stride_slot;
            uint8_t* const values =
                reinterpret_cast<uint8_t*>(output) + slot / 2;
            __fp16* const scales =
                reinterpret_cast<__fp16*>(
                    reinterpret_cast<uint8_t*>(output) +
                    int4_scales_offset(output_size_m, output_size_n)) +
                slot / kInt4Group;
            uint32_t value_at = static_cast<uint32_t>(g) * 16;
            uint32_t scale_at = static_cast<uint32_t>(g) * kInt4ScaleBytes;
            asm volatile("" : "+v"(value_at));
            asm volatile("" : "+v"(scale_at));

            const int words_in =
                clip_n / (kInt4Group / kInt4GroupPacks) - g * kInt4GroupPacks;
            if (!TAIL || words_in >= kInt4GroupPacks) {
              *(global_u32x4*)(values + value_at) = u32x4{
                  encoded[0][0], encoded[1][0], encoded[2][0], encoded[3][0]};
            } else {
#pragma unroll
              for (int k = 0; k < kInt4GroupPacks; ++k)
                if (k < words_in)
                  *(global_u32*)(values + value_at + k * kInt4WordBytes) =
                      encoded[k][0];
            }
            *(global_f16*)(reinterpret_cast<char*>(scales) + scale_at) =
                __builtin_bit_cast(__fp16,
                                   static_cast<uint16_t>(encoded[0][1]));
          }
        }
      } else if constexpr (!QUANTIZE_INT4) {
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
          if (row_at(i) >= clip_m) continue;
#pragma unroll
          for (int k = 0; k < kPacks; ++k) {
            if (col_at(k) >= clip_n / kPackElems) continue;
            V v;
            __builtin_memcpy(&v, input[i][k], sizeof(v));

            uint32_t offset =
                static_cast<uint32_t>(col_at(k)) * uint32_t{sizeof(V)};
            asm volatile("" : "+v"(offset));
            V* row = reinterpret_cast<V*>(at_copy) +
                     int64_t{row_at(i)} * (output_stride_m / kPackElems);
            V* p = reinterpret_cast<V*>(reinterpret_cast<char*>(row) + offset);
            if constexpr (sizeof(V) == 16) {
              u32x4 raw;
              __builtin_memcpy(&raw, &v, 16);
              *(global_u32x4*)(p) = raw;
            } else {
              *p = v;
            }
          }
        }
      } else {
        static_assert(kThreadsN % kInt4GroupPacks == 0,
                      "a group is 4 lanes' packs");
        constexpr int kInt4PackElems = kInt4Group / kInt4GroupPacks;
        uint32_t encoded[kRows][kPacks][2];
#pragma unroll
        for (int i = 0; i < kRows; ++i)
#pragma unroll
          for (int k = 0; k < kPacks; ++k) {
            if constexpr (kRaw) {
              encoded[i][k][0] = input[i][k][0];
              encoded[i][k][1] = input[i][k][1];
            } else {
              quantize(reinterpret_cast<const U(&)[1][kPackElems]>(input[i][k]),
                       reinterpret_cast<uint32_t (&)[1][2]>(encoded[i][k]));
            }
          }
        const int64_t values_stride_m = output_size_n / 2;
        const int64_t scales_stride_m =
            TAIL ? int4_groups(output_size_n) : output_size_n / kInt4Group;
        const int64_t scales_at =
            int4_scales_offset(output_size_m, output_size_n);
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
          if (row_at(i) >= clip_m) continue;
          const int64_t row = row_at(i);
#pragma unroll
          for (int k = 0; k < kPacks; ++k) {
            const int pack =
                offs_n / kInt4PackElems + tid % kThreadsN + k * kThreadsN;
            if (pack >= clip_n / kInt4PackElems) continue;
            uint32_t value_at = static_cast<uint32_t>(row * values_stride_m) +
                                static_cast<uint32_t>(pack) * kInt4WordBytes;
            uint32_t scale_at = static_cast<uint32_t>(
                (row * scales_stride_m + pack / kInt4GroupPacks) *
                kInt4ScaleBytes);
            asm volatile("" : "+v"(value_at));
            asm volatile("" : "+v"(scale_at));
            *(global_u32*)(at_copy + value_at) = encoded[i][k][0];
            if (pack % kInt4GroupPacks == 0)
              *(global_f16*)(at_copy + scales_at + scale_at) =
                  __builtin_bit_cast(__fp16,
                                     static_cast<uint16_t>(encoded[i][k][1]));
          }
        }
      }
    } else {
      // A peer a lane.
      void* const output = lane_ptr_array_at<void, WORLD>(
          entries, output_ptr_array_size, output_rank);
#if defined(__AMDGCN_WAVEFRONT_SIZE)
      static_assert(__AMDGCN_WAVEFRONT_SIZE == WAVE_SIZE,
                    "the compiler's wave is not the one handed");
#endif
      constexpr int kRows = std::extent_v<INPUT, 0>;
      constexpr int kPacks = std::extent_v<INPUT, 1>;
      constexpr int kPackElems = std::extent_v<INPUT, 2>;
      using U = std::remove_all_extents_t<INPUT>;
      constexpr bool kRaw = std::is_same_v<U, uint32_t>;
      static_assert(
          !kRaw || (QUANTIZE_INT4 && kPackElems == 2),
          "uint32_t registers are an INT4 payload's words and scales");
      constexpr int kThreadsM = TILE_M / kRows;
      constexpr int kThreadsN = BLOCK_X / kThreadsM;
      static_assert(kThreadsM * kRows == TILE_M,
                    "a tile is whole rows a thread");
      static_assert(kThreadsM * kThreadsN == BLOCK_X,
                    "the layout is the block");
      using V = vec<U, kPackElems>;
      using E = std::conditional_t<!QUANTIZE_INT4, U, uint8_t>;
      if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
          (output_slot < 0 || output_slot >= output_size_slot))
        __builtin_trap();
      [[maybe_unused]] E* const at_copy =
          reinterpret_cast<E*>(output) + output_slot * output_stride_slot;
      const int tid = static_cast<int>(threadIdx.x);
      const auto row_at = [&](int i) {
        return offs_m + (kThreadsM == 1 ? i : tid / kThreadsN + i * kThreadsM) *
                            row_step;
      };
      const auto col_at = [&](int k) {
        return offs_n / kPackElems + tid % kThreadsN + k * kThreadsN;
      };
      [[maybe_unused]] const auto element_at = [](auto* ptr, int64_t stride_m,
                                                  int64_t stride_n, int64_t e,
                                                  int n) {
        if (stride_m == int64_t{n} * stride_n) return ptr + e * stride_n;
        const int64_t row =
            e < (int64_t{1} << 31)
                ? int64_t{static_cast<uint32_t>(e) / static_cast<uint32_t>(n)}
                : e / n;
        return ptr + row * stride_m + (e - row * n) * stride_n;
      };
      constexpr bool kGroup =
          LAYOUT == Layout::group_run || LAYOUT == Layout::group_flat;
      constexpr bool kRun =
          LAYOUT == Layout::run || LAYOUT == Layout::group_run;
      if constexpr (LAYOUT != Layout::tile && !kGroup) {
        static_assert(!QUANTIZE_INT4 && kRows == 1,
                      "the flat walk writes one exact copy a row of threads");

        constexpr int kLanes = BLOCK_X / WORLD;
        static_assert(kLanes * WORLD == BLOCK_X,
                      "a block is a group of threads a rank");
        U* const to =
            reinterpret_cast<U*>(output) + int64_t{offs_m} * output_stride_m;
        const int64_t first = int64_t{output_slot} * output_stride_slot;
        U* const run = to + first * output_stride_n;
#pragma unroll
        for (int k = 0; k < kPacks; ++k) {
          const int c = offs_n / kPackElems + tid % kLanes + k * kLanes;
          if (c >= clip_n / kPackElems) continue;
          V* p;
          if constexpr (kRun) {
            uint32_t offset = static_cast<uint32_t>(c) * uint32_t{sizeof(V)};
            asm volatile("" : "+v"(offset));
            p = reinterpret_cast<V*>(reinterpret_cast<char*>(run) + offset);
          } else {
            p = reinterpret_cast<V*>(
                element_at(to, output_stride_m, output_stride_n,
                           first + int64_t{c} * kPackElems, output_size_n));
          }
          V v;
          __builtin_memcpy(&v, input[0][k], sizeof(v));
          {
            using W = std::conditional_t<
                sizeof(V) == 16, u32x4,
                std::conditional_t<
                    sizeof(V) == 8, u32x2,
                    std::conditional_t<sizeof(V) == 4, uint32_t, uint16_t>>>;
            W raw;
            __builtin_memcpy(&raw, &v, sizeof(V));
            *(__attribute__((address_space(1))) W*)(p) = raw;
          }
        }
      } else if constexpr (kGroup) {
        static_assert(
            kRows == 1 && kPacks == kInt4GroupPacks,
            "a group a thread here: several are load's, recv's, store's "
            "and send's loop");
        const int g = offs_n / kInt4Group + tid % kThreadsN;
        if constexpr (!QUANTIZE_INT4) {
          if constexpr (TAIL) {
            if (g >= int4_groups(clip_n)) return;
            U* const to = reinterpret_cast<U*>(output) +
                          int64_t{offs_m} * output_stride_m;
            const int64_t first = int64_t{output_slot} * output_stride_slot;

            [[maybe_unused]] V* group = nullptr;
            if constexpr (kRun) {
              uint32_t offset =
                  static_cast<uint32_t>(g) * (kInt4Group * sizeof(U));
              asm volatile("" : "+v"(offset));
              group = reinterpret_cast<V*>(
                  reinterpret_cast<char*>(to + first * output_stride_n) +
                  offset);
            }
            const auto pack_at = [&](int k) -> V* {
              if constexpr (kRun)
                return group + k;
              else
                return reinterpret_cast<V*>(element_at(
                    to, output_stride_m, output_stride_n,
                    first + int64_t{g} * kInt4Group + int64_t{k} * kPackElems,
                    output_size_n));
            };
#pragma unroll
            for (int k = 0; k < kInt4GroupPacks; ++k) {
              if (g * kInt4GroupPacks + k >= clip_n / kPackElems) continue;
              u32x4 raw;
              __builtin_memcpy(&raw, input[0][k], sizeof(raw));
              *(global_u32x4*)(pack_at(k)) = raw;
            }
          } else {
            if (g >= clip_n / kInt4Group) return;
            U* const to = reinterpret_cast<U*>(output) +
                          int64_t{offs_m} * output_stride_m;
            const int64_t first = int64_t{output_slot} * output_stride_slot;
            V* group;
            if constexpr (kRun) {
              uint32_t offset =
                  static_cast<uint32_t>(g) * (kInt4Group * sizeof(U));
              asm volatile("" : "+v"(offset));
              group = reinterpret_cast<V*>(
                  reinterpret_cast<char*>(to + first * output_stride_n) +
                  offset);
            } else
              group = reinterpret_cast<V*>(
                  element_at(to, output_stride_m, output_stride_n,
                             first + int64_t{g} * kInt4Group, output_size_n));
#pragma unroll
            for (int k = 0; k < kInt4GroupPacks; ++k) {
              u32x4 raw;
              __builtin_memcpy(&raw, input[0][k], sizeof(raw));
              *(global_u32x4*)(group + k) = raw;
            }
          }
        } else {
          static_assert(kRun, "an INT4 payload is dense");
          uint32_t encoded[kInt4GroupPacks][2];
          if constexpr (kRaw) {
#pragma unroll
            for (int k = 0; k < kInt4GroupPacks; ++k) {
              encoded[k][0] = input[0][k][0];
              encoded[k][1] = input[0][k][1];
            }
          } else {
            quantize(input[0], encoded);
          }
          if (g < (TAIL ? static_cast<int>(int4_groups(clip_n))
                        : clip_n / kInt4Group)) {
            const int64_t slot = int64_t{output_slot} * output_stride_slot;
            uint8_t* const values =
                reinterpret_cast<uint8_t*>(output) + slot / 2;
            __fp16* const scales =
                reinterpret_cast<__fp16*>(
                    reinterpret_cast<uint8_t*>(output) +
                    int4_scales_offset(output_size_m, output_size_n)) +
                slot / kInt4Group;
            uint32_t value_at = static_cast<uint32_t>(g) * 16;
            uint32_t scale_at = static_cast<uint32_t>(g) * kInt4ScaleBytes;
            asm volatile("" : "+v"(value_at));
            asm volatile("" : "+v"(scale_at));

            const int words_in =
                clip_n / (kInt4Group / kInt4GroupPacks) - g * kInt4GroupPacks;
            if (!TAIL || words_in >= kInt4GroupPacks) {
              *(global_u32x4*)(values + value_at) = u32x4{
                  encoded[0][0], encoded[1][0], encoded[2][0], encoded[3][0]};
            } else {
#pragma unroll
              for (int k = 0; k < kInt4GroupPacks; ++k)
                if (k < words_in)
                  *(global_u32*)(values + value_at + k * kInt4WordBytes) =
                      encoded[k][0];
            }
            *(global_f16*)(reinterpret_cast<char*>(scales) + scale_at) =
                __builtin_bit_cast(__fp16,
                                   static_cast<uint16_t>(encoded[0][1]));
          }
        }
      } else if constexpr (!QUANTIZE_INT4) {
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
          if (row_at(i) >= clip_m) continue;
#pragma unroll
          for (int k = 0; k < kPacks; ++k) {
            if (col_at(k) >= clip_n / kPackElems) continue;
            V v;
            __builtin_memcpy(&v, input[i][k], sizeof(v));

            uint32_t offset =
                static_cast<uint32_t>(col_at(k)) * uint32_t{sizeof(V)};
            asm volatile("" : "+v"(offset));
            V* row = reinterpret_cast<V*>(at_copy) +
                     int64_t{row_at(i)} * (output_stride_m / kPackElems);
            V* p = reinterpret_cast<V*>(reinterpret_cast<char*>(row) + offset);
            if constexpr (sizeof(V) == 16) {
              u32x4 raw;
              __builtin_memcpy(&raw, &v, 16);
              *(global_u32x4*)(p) = raw;
            } else {
              *p = v;
            }
          }
        }
      } else {
        static_assert(kThreadsN % kInt4GroupPacks == 0,
                      "a group is 4 lanes' packs");
        constexpr int kInt4PackElems = kInt4Group / kInt4GroupPacks;
        uint32_t encoded[kRows][kPacks][2];
#pragma unroll
        for (int i = 0; i < kRows; ++i)
#pragma unroll
          for (int k = 0; k < kPacks; ++k) {
            if constexpr (kRaw) {
              encoded[i][k][0] = input[i][k][0];
              encoded[i][k][1] = input[i][k][1];
            } else {
              quantize(reinterpret_cast<const U(&)[1][kPackElems]>(input[i][k]),
                       reinterpret_cast<uint32_t (&)[1][2]>(encoded[i][k]));
            }
          }
        const int64_t values_stride_m = output_size_n / 2;
        const int64_t scales_stride_m =
            TAIL ? int4_groups(output_size_n) : output_size_n / kInt4Group;
        const int64_t scales_at =
            int4_scales_offset(output_size_m, output_size_n);
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
          if (row_at(i) >= clip_m) continue;
          const int64_t row = row_at(i);
#pragma unroll
          for (int k = 0; k < kPacks; ++k) {
            const int pack =
                offs_n / kInt4PackElems + tid % kThreadsN + k * kThreadsN;
            if (pack >= clip_n / kInt4PackElems) continue;
            uint32_t value_at = static_cast<uint32_t>(row * values_stride_m) +
                                static_cast<uint32_t>(pack) * kInt4WordBytes;
            uint32_t scale_at = static_cast<uint32_t>(
                (row * scales_stride_m + pack / kInt4GroupPacks) *
                kInt4ScaleBytes);
            asm volatile("" : "+v"(value_at));
            asm volatile("" : "+v"(scale_at));
            *(global_u32*)(at_copy + value_at) = encoded[i][k][0];
            if (pack % kInt4GroupPacks == 0)
              *(global_f16*)(at_copy + scales_at + scale_at) =
                  __builtin_bit_cast(__fp16,
                                     static_cast<uint16_t>(encoded[i][k][1]));
          }
        }
      }
    }
  }
}

// An LL send: half of each line the data, half its row's epoch.
template <int WORLD, int TILE_M, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE,
          bool QUANTIZE_INT4, Layout LAYOUT, bool TAIL, typename INPUT,
          typename PTR>
DINLINE void send_ll(const INPUT& input, PTR const* output_ptr_array,
                     int output_ptr_array_size, int output_rank,
                     int output_slot, int64_t output_stride_slot,
                     int64_t output_stride_m, int64_t output_stride_n,
                     int output_size_slot, int output_size_m, int output_size_n,
                     int clip_m, int clip_n, int offs_m, int offs_n,
                     int row_step, const uint32_t* flags) {
  // A block of BLOCK_Y rows of threads sends to output_rank from the row that
  // receives its copy (recv's rows): row output_rank / (WORLD / BLOCK_Y).
  if constexpr (BLOCK_Y > 1)
    if (output_rank / (WORLD / BLOCK_Y) !=
        __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.y)))
      return;
  // One peer for the wave: its memory's address is scalar.
  const auto* const entries =
      (decltype(+output_ptr_array[0]) const*)output_ptr_array;
  const int owner = __builtin_amdgcn_readfirstlane(output_rank);
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
      __builtin_amdgcn_ballot_w64(output_rank != owner) != 0)
    __builtin_trap();
  void* const output = ptr_array_at(entries, output_ptr_array_size, owner);
#if defined(__AMDGCN_WAVEFRONT_SIZE)
  static_assert(__AMDGCN_WAVEFRONT_SIZE == WAVE_SIZE,
                "the compiler's wave is not the one handed");
#endif
  constexpr int kRows = std::extent_v<INPUT, 0>;
  constexpr int kPacks = std::extent_v<INPUT, 1>;
  constexpr int kPackElems = std::extent_v<INPUT, 2>;
  using U = std::remove_all_extents_t<INPUT>;
  constexpr bool kRaw = std::is_same_v<U, uint32_t>;
  static_assert(!kRaw || (QUANTIZE_INT4 && kPackElems == 2),
                "uint32_t registers are an INT4 payload's words and scales");
  constexpr int kThreadsM = TILE_M / kRows;
  constexpr int kThreadsN = BLOCK_X / kThreadsM;
  static_assert(kThreadsM * kRows == TILE_M, "a tile is whole rows a thread");
  static_assert(kThreadsM * kThreadsN == BLOCK_X, "the layout is the block");
  using V = vec<U, kPackElems>;
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
      (output_slot < 0 || output_slot >= output_size_slot))
    __builtin_trap();
  const int tid = static_cast<int>(threadIdx.x);
  const auto row_at = [&](int i) {
    return offs_m +
           (kThreadsM == 1 ? i : tid / kThreadsN + i * kThreadsM) * row_step;
  };
  const auto col_at = [&](int k) {
    return offs_n / kPackElems + tid % kThreadsN + k * kThreadsN;
  };
  {
    static_assert(LAYOUT == Layout::tile, "LL lines sit in a tile");
    constexpr int kLines = QUANTIZE_INT4 || sizeof(V) <= kLinePayloadBytes
                               ? 1
                               : sizeof(V) / kLinePayloadBytes;
    static_assert(
        (QUANTIZE_INT4 ? kRaw || kPackElems == kInt4Group / kInt4GroupPacks
                       : sizeof(V) <= kLinePayloadBytes ||
                             sizeof(V) % kLinePayloadBytes == 0) &&
            kLines >= 1 && kLines <= 2 && sizeof(u32x4) == kLineBytes,
        "a pack is one or two lines, an INT4 pack a word of 8 values");
    constexpr int kLineScale = kLineBytes / kLinePayloadBytes;
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG && flags == nullptr) __builtin_trap();

    [[maybe_unused]] uint32_t payload[kRows][kPacks][2];
    if constexpr (QUANTIZE_INT4) {
#pragma unroll
      for (int i = 0; i < kRows; ++i)
#pragma unroll
        for (int k = 0; k < kPacks; ++k) {
          if constexpr (kRaw) {
            payload[i][k][0] = input[i][k][0];
            payload[i][k][1] = input[i][k][1];
          } else {
            quantize(reinterpret_cast<const U(&)[1][kPackElems]>(input[i][k]),
                     reinterpret_cast<uint32_t (&)[1][2]>(payload[i][k]));
          }
        }
    }
    const int bank_rows = output_size_m / 2;

    constexpr int64_t kElemBytes = kRaw ? 2 : int64_t{sizeof(U)};
    char* const slot =
        reinterpret_cast<char*>(output) +
        int64_t{output_slot} * output_stride_slot * kElemBytes * kLineScale;
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      if (!(kRows == 1 && kThreadsM == 1) && row_at(i) >= clip_m) continue;

      const int row = row_at(i) + static_cast<int>(flags[i] & 1u) * bank_rows;
      if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
          (row >= output_size_m || flags[i] == 0))
        __builtin_trap();
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        constexpr int kElems = kRaw ? kInt4Group / kInt4GroupPacks : kPackElems;

        if constexpr (QUANTIZE_INT4) {
          if (col_at(k) >= clip_n / kElems) continue;
        }
        const int col = min(col_at(k), clip_n / kElems - 1);
        uint32_t words[2 * kLines] = {};
        if constexpr (QUANTIZE_INT4) {
          words[0] = payload[i][k][0];
          words[1] = payload[i][k][1];
        } else {
          __builtin_memcpy(words, input[i][k], sizeof(V));
        }

        const uint32_t half =
            static_cast<uint32_t>(output_stride_m * kElemBytes);
        uint32_t at = static_cast<uint32_t>(row) * half * kLineScale +
                      static_cast<uint32_t>(col) * uint32_t{kLineBytes};
        asm volatile("" : "+v"(at));
#pragma unroll
        for (int h = 0; h < kLines; ++h)
          __builtin_nontemporal_store(
              // The flag is in both 8-byte halves, so a reader that sees both
              // has the whole line.
              u32x4{words[2 * h], flags[i], words[2 * h + 1], flags[i]},
              (global_u32x4*)(slot + at + h * half));
      }
    }
  }
}

// A Lamport send: each pack whole, a sentinel element made 0x7fff.
template <int WORLD, int TILE_M, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE,
          bool QUANTIZE_INT4, Layout LAYOUT, bool TAIL, typename INPUT,
          typename PTR>
DINLINE void send_lamport(const INPUT& input, PTR const* output_ptr_array,
                          int output_ptr_array_size, int output_rank,
                          int output_slot, int64_t output_stride_slot,
                          int64_t output_stride_m, int64_t output_stride_n,
                          int output_size_slot, int output_size_m,
                          int output_size_n, int clip_m, int clip_n, int offs_m,
                          int offs_n, int row_step, const uint32_t* flags) {
  // A block of BLOCK_Y rows of threads sends to output_rank from the row that
  // receives its copy (recv's rows): row output_rank / (WORLD / BLOCK_Y).
  if constexpr (BLOCK_Y > 1)
    if (output_rank / (WORLD / BLOCK_Y) !=
        __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.y)))
      return;
  // One peer for the wave: its memory's address is scalar.
  const auto* const entries =
      (decltype(+output_ptr_array[0]) const*)output_ptr_array;
  const int owner = __builtin_amdgcn_readfirstlane(output_rank);
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
      __builtin_amdgcn_ballot_w64(output_rank != owner) != 0)
    __builtin_trap();
  void* const output = ptr_array_at(entries, output_ptr_array_size, owner);
#if defined(__AMDGCN_WAVEFRONT_SIZE)
  static_assert(__AMDGCN_WAVEFRONT_SIZE == WAVE_SIZE,
                "the compiler's wave is not the one handed");
#endif
  constexpr int kRows = std::extent_v<INPUT, 0>;
  constexpr int kPacks = std::extent_v<INPUT, 1>;
  constexpr int kPackElems = std::extent_v<INPUT, 2>;
  using U = std::remove_all_extents_t<INPUT>;
  constexpr bool kRaw = std::is_same_v<U, uint32_t>;
  static_assert(!kRaw || (QUANTIZE_INT4 && kPackElems == 2),
                "uint32_t registers are an INT4 payload's words and scales");
  constexpr int kThreadsM = TILE_M / kRows;
  constexpr int kThreadsN = BLOCK_X / kThreadsM;
  static_assert(kThreadsM * kRows == TILE_M, "a tile is whole rows a thread");
  static_assert(kThreadsM * kThreadsN == BLOCK_X, "the layout is the block");
  using V = vec<U, kPackElems>;
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
      (output_slot < 0 || output_slot >= output_size_slot))
    __builtin_trap();
  const int tid = static_cast<int>(threadIdx.x);
  const auto row_at = [&](int i) {
    return offs_m +
           (kThreadsM == 1 ? i : tid / kThreadsN + i * kThreadsM) * row_step;
  };
  const auto col_at = [&](int k) {
    return offs_n / kPackElems + tid % kThreadsN + k * kThreadsN;
  };
  {
    static_assert(LAYOUT == Layout::tile, "Lamport lines sit in a tile");
    static_assert(!QUANTIZE_INT4 && sizeof(U) == 2,
                  "the sentinel is a 2-byte NaN");
    using W = raw_t<sizeof(V)>;
    constexpr int kHalves = sizeof(V) / 2;
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG && flags == nullptr) __builtin_trap();
    const int bank_rows = output_size_m / 2;
    char* const slot =
        reinterpret_cast<char*>(output) +
        int64_t{output_slot} * output_stride_slot * int64_t{sizeof(U)};
    const uint32_t row_bytes =
        static_cast<uint32_t>(output_stride_m * int64_t{sizeof(U)});
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      if (!(kRows == 1 && kThreadsM == 1) && row_at(i) >= clip_m) continue;
      const int row = row_at(i) + static_cast<int>(flags[i] & 1u) * bank_rows;
      if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
          (row >= output_size_m || flags[i] == 0))
        __builtin_trap();
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        if (col_at(k) >= clip_n / kPackElems) continue;
        uint16_t h[kHalves];
        __builtin_memcpy(h, input[i][k], sizeof(h));
#pragma unroll
        for (int j = 0; j < kHalves; ++j)
          if (h[j] == kLamportEmpty) h[j] = 0x7fff;
        W w;
        __builtin_memcpy(&w, h, sizeof(w));
        uint32_t at = static_cast<uint32_t>(row) * row_bytes +
                      static_cast<uint32_t>(col_at(k)) * uint32_t{sizeof(V)};
        asm volatile("" : "+v"(at));
        __builtin_nontemporal_store(
            w, (__attribute__((address_space(1))) W*)(slot + at));
      }
    }
  }
}

}  // namespace internal

template <int WORLD, int TILE_M, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE,
          bool QUANTIZE_INT4, Protocol PROTOCOL, Layout LAYOUT,
          bool TAIL = false, typename INPUT, typename PTR>
DINLINE void send(const INPUT& input, PTR const* output_ptr_array,
                  int output_ptr_array_size, int output_rank, int output_slot,
                  int64_t output_stride_slot, int64_t output_stride_m,
                  int64_t output_stride_n, int output_size_slot,
                  int output_size_m, int output_size_n, int clip_m, int clip_n,
                  int offs_m, int offs_n, int row_step,
                  const uint32_t* flags = nullptr) {
  if constexpr (PROTOCOL == Protocol::simple) {
    static_assert(BLOCK_Y == 1, "a Simple send is one row of threads");
    internal::send_simple<WORLD, TILE_M, BLOCK_X, WAVE_SIZE, QUANTIZE_INT4,
                          LAYOUT, TAIL>(
        input, output_ptr_array, output_ptr_array_size, output_rank,
        output_slot, output_stride_slot, output_stride_m, output_stride_n,
        output_size_slot, output_size_m, output_size_n, clip_m, clip_n, offs_m,
        offs_n, row_step);
  } else if constexpr (PROTOCOL == Protocol::ll) {
    internal::send_ll<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE, QUANTIZE_INT4,
                      LAYOUT, TAIL>(
        input, output_ptr_array, output_ptr_array_size, output_rank,
        output_slot, output_stride_slot, output_stride_m, output_stride_n,
        output_size_slot, output_size_m, output_size_n, clip_m, clip_n, offs_m,
        offs_n, row_step, flags);
  } else {
    internal::send_lamport<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE,
                           QUANTIZE_INT4, LAYOUT, TAIL>(
        input, output_ptr_array, output_ptr_array_size, output_rank,
        output_slot, output_stride_slot, output_stride_m, output_stride_n,
        output_size_slot, output_size_m, output_size_n, clip_m, clip_n, offs_m,
        offs_n, row_step, flags);
  }
}

}  // namespace hip_custom_communicator
