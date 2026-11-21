#pragma once

// send: registers into a peer's memory, by its protocol (send_ll, send_lamport,
// send_simple_write, send_simple_read; send_simple moves their data).

#include "quantize.cuh"
#include "types.cuh"

namespace hip_custom_communicator {

namespace internal {

// A Simple send: a store into the peer's memory, in any layout.
template <int WORLD, int TILE_M, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE,
          bool QUANTIZE_INT4, Layout LAYOUT, bool TAIL, typename INPUT,
          typename PTR>
DINLINE void send_simple(const INPUT& input, PTR const* output_ptr_array,
                         int output_ptr_array_size, int output_rank,
                         int output_slot, int64_t output_stride_slot,
                         int64_t output_stride_m, int64_t output_stride_n,
                         int output_size_slot, int output_size_m,
                         int output_size_n, int clip_m, int clip_n, int offs_m,
                         int offs_n, int row_step) {
  // One row of threads (BLOCK_Y = 1): a row split of Simple is not written yet.
  static_assert(BLOCK_Y == 1, "a Simple send is one row of threads");
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
      send_simple<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE, QUANTIZE_INT4,
                  LAYOUT, TAIL>(
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
        internal::check_failed();
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
            const int64_t slot = int64_t{output_slot} * output_stride_slot +
                                 int64_t{offs_m} * output_stride_m;
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
        internal::check_failed();
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
            const int64_t slot = int64_t{output_slot} * output_stride_slot +
                                 int64_t{offs_m} * output_stride_m;
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

// An LL send: half of each line the data, half its row's step.
template <int WORLD, int TILE_M, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE,
          bool QUANTIZE_INT4, Layout LAYOUT, bool TAIL, typename INPUT,
          typename PTR>
DINLINE void send_ll(const INPUT& input, PTR const* output_ptr_array,
                     int output_ptr_array_size, int output_rank,
                     int output_slot, int64_t output_stride_slot,
                     int64_t output_stride_m, int64_t output_stride_n,
                     int output_size_slot, int output_size_m, int output_size_n,
                     int clip_m, int clip_n, int offs_m, int offs_n,
                     int row_step, Connection& connection) {
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
    internal::check_failed();
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
    internal::check_failed();
  const int tid = static_cast<int>(threadIdx.x);
  const auto row_at = [&](int i) {
    return offs_m +
           (kThreadsM == 1 ? i : tid / kThreadsN + i * kThreadsM) * row_step;
  };
  const auto col_at = [&](int k) {
    return offs_n / kPackElems + tid % kThreadsN + k * kThreadsN;
  };
  // The tile's step (CONNECTION, types.cuh): the next after the one the block's
  // last tile left.
  static_assert(TILE_M == 1, "a connection's step is one inbox row's");
  uint32_t flags[kRows];
#pragma unroll
  for (int i = 0; i < kRows; ++i) {
    uint32_t last = static_cast<uint32_t>(connection[0][0][0]);
    // Left in a vector register: a readfirstlane is convergent, so a send to
    // each rank would keep its own, the sends' stores could not share a
    // branch, and each rank's address load would wait in its own.
    flags[i] = row_at(i) < clip_m ? (last == kLastStep ? 3u : last + 1u) : last;
  }
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
    const int bank_rows = output_size_m / kLlBanks;

    constexpr int64_t kElemBytes = kRaw ? 2 : int64_t{sizeof(U)};
    char* const slot =
        reinterpret_cast<char*>(output) +
        int64_t{output_slot} * output_stride_slot * kElemBytes * kLineScale;
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      if (!(kRows == 1 && kThreadsM == 1) && row_at(i) >= clip_m) continue;

      const int row =
          row_at(i) + static_cast<int>(flags[i] % kLlBanks) * bank_rows;
      if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
          (row >= output_size_m || flags[i] == 0))
        internal::check_failed();
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
                          int offs_n, int row_step, Connection& connection) {
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
    internal::check_failed();
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
    internal::check_failed();
  const int tid = static_cast<int>(threadIdx.x);
  const auto row_at = [&](int i) {
    return offs_m +
           (kThreadsM == 1 ? i : tid / kThreadsN + i * kThreadsM) * row_step;
  };
  const auto col_at = [&](int k) {
    return offs_n / kPackElems + tid % kThreadsN + k * kThreadsN;
  };
  // The tile's step (CONNECTION, types.cuh): the next after the one the block's
  // last tile left.
  static_assert(TILE_M == 1, "a connection's step is one inbox row's");
  uint32_t flags[kRows];
#pragma unroll
  for (int i = 0; i < kRows; ++i) {
    uint32_t last = static_cast<uint32_t>(connection[0][0][0]);
    // Left in a vector register: a readfirstlane is convergent, so a send to
    // each rank would keep its own, the sends' stores could not share a
    // branch, and each rank's address load would wait in its own.
    flags[i] = row_at(i) < clip_m ? (last == kLastStep ? 3u : last + 1u) : last;
  }
  {
    static_assert(LAYOUT == Layout::tile, "Lamport lines sit in a tile");
    static_assert(!QUANTIZE_INT4 && sizeof(U) == 2,
                  "the sentinel is a 2-byte NaN");
    using W = raw_t<sizeof(V)>;
    constexpr int kHalves = sizeof(V) / 2;
    const int bank_rows = output_size_m / kLamportBanks;
    char* const slot =
        reinterpret_cast<char*>(output) +
        int64_t{output_slot} * output_stride_slot * int64_t{sizeof(U)};
    const uint32_t row_bytes =
        static_cast<uint32_t>(output_stride_m * int64_t{sizeof(U)});
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      if (!(kRows == 1 && kThreadsM == 1) && row_at(i) >= clip_m) continue;
      const int row =
          row_at(i) + static_cast<int>(flags[i] % kLamportBanks) * bank_rows;
      if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
          (row >= output_size_m || flags[i] == 0))
        internal::check_failed();
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

// A simple_write send: once the receiver has read the last round (head), the
// data into its staging, then the round posted (tail, COUNTERS).
template <int WORLD, int TILE_M, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE,
          bool QUANTIZE_INT4, Layout LAYOUT, bool TAIL, typename INPUT,
          typename PTR>
DINLINE void send_simple_write(
    const INPUT& input, PTR const* output_ptr_array, int output_ptr_array_size,
    int output_rank, int output_slot, int64_t output_stride_slot,
    int64_t output_stride_m, int64_t output_stride_n, int output_size_slot,
    int output_size_m, int output_size_n, int clip_m, int clip_n, int offs_m,
    int offs_n, int row_step, Connection& connection,
    uint32_t* const* counters_ptr_array, int counters_ptr_array_size,
    int64_t counters_stride_kind, int64_t counters_stride_block,
    int64_t counters_stride_peer, int counters_size_block,
    int counters_size_peer, uint32_t* status_ptr, int64_t status_stride_field,
    int status_size_field, uint64_t timeout_ticks) {
  if constexpr (LAYOUT != Layout::tile) {
    // A flat walk: a peer a lane group, this lane's output_rank its group's
    // receiver; this rank is output_slot.
    static_assert(LAYOUT == Layout::run || LAYOUT == Layout::flat ||
                      LAYOUT == Layout::group_run ||
                      LAYOUT == Layout::group_flat,
                  "a flat walk: run or flat, or their INT4 groups");
    static_assert(BLOCK_Y == 1, "a flat walk is one row of threads");
    constexpr int kLanes = BLOCK_X / WORLD;
    static_assert(kLanes * WORLD == BLOCK_X,
                  "a block is a group of threads a rank");
    // The round: the next after the block's last (CONNECTION).
    const uint32_t last = static_cast<uint32_t>(connection[0][0][0]);
    const uint32_t next = last == kLastStep ? 3u : last + 1u;
    const int64_t at_block = int64_t{blockIdx.x} * counters_stride_block;
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
        (static_cast<int>(blockIdx.x) >= counters_size_block ||
         output_rank < 0 || output_rank >= counters_size_peer))
      internal::check_failed();
    // Each lane waits until its group's receiver has read the last round
    // (its head, in this rank's counters).
    uint32_t* const head =
        ptr_array_at(counters_ptr_array, counters_ptr_array_size, output_slot) +
        static_cast<int64_t>(Counter::head) * counters_stride_kind + at_block +
        int64_t{output_rank} * counters_stride_peer;
    const uint64_t t0 = wall_clock64();
    for (uint32_t n = 1;; ++n) {
      const uint32_t seen =
          __scoped_atomic_load_n(head, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
      if (HIP_CUSTOM_COMMUNICATOR_DEBUG && seen != last && seen != last - 1u &&
          !(last == 3u && seen == kLastStep))
        internal::check_failed();
      if (__builtin_amdgcn_ballot_w64(seen != last) == 0) break;
      if ((n & 255u) != 0) continue;
      const uint64_t waited = wall_clock64() - t0;
      const bool aborted =
          __scoped_atomic_load_n(
              status_ptr + static_cast<int64_t>(StatusField::abort) *
                               status_stride_field,
              __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM) != 0;
      if (__builtin_amdgcn_ballot_w64(aborted) == 0 && waited <= timeout_ticks)
        continue;
      if (!aborted)
        give_up(status_ptr, status_stride_field, status_size_field,
                Where::start, output_slot, static_cast<uint32_t>(output_rank),
                seen, last, waited);
      __builtin_amdgcn_endpgm();
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    // The data into the receiver's inbox, at this rank's slot.
    send_simple<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE, QUANTIZE_INT4,
                LAYOUT, TAIL>(input, output_ptr_array, output_ptr_array_size,
                              output_rank, output_slot, output_stride_slot,
                              output_stride_m, output_stride_n,
                              output_size_slot, output_size_m, output_size_n,
                              clip_m, clip_n, offs_m, offs_n, row_step);
    // The block's stores, then each group's round posted (its receiver's
    // tail from this rank), by the group's first lane.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    if constexpr (BLOCK_X > WAVE_SIZE) __syncthreads();
    if (static_cast<int>(threadIdx.x) % kLanes == 0)
      __scoped_atomic_store_n(
          lane_ptr_array_at<uint32_t, WORLD>(
              counters_ptr_array, counters_ptr_array_size, output_rank) +
              static_cast<int64_t>(Counter::tail) * counters_stride_kind +
              at_block + int64_t{output_slot} * counters_stride_peer,
          next, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
  } else {
    static_assert(
        LAYOUT == Layout::tile && !QUANTIZE_INT4 && TILE_M == 1 && BLOCK_Y == 1,
        "a simple_write send is an exact tile of one row, from one "
        "row of threads");
    static_assert(BLOCK_X % WAVE_SIZE == 0, "a block is whole waves");
    const auto* const entries =
        (decltype(+output_ptr_array[0]) const*)output_ptr_array;
    const int owner = __builtin_amdgcn_readfirstlane(output_rank);
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
        __builtin_amdgcn_ballot_w64(output_rank != owner) != 0)
      internal::check_failed();
    void* const output = ptr_array_at(entries, output_ptr_array_size, owner);
#if defined(__AMDGCN_WAVEFRONT_SIZE)
    static_assert(__AMDGCN_WAVEFRONT_SIZE == WAVE_SIZE,
                  "the compiler's wave is not the one handed");
#endif
    constexpr int kRows = std::extent_v<INPUT, 0>;
    constexpr int kPacks = std::extent_v<INPUT, 1>;
    constexpr int kPackElems = std::extent_v<INPUT, 2>;
    using U = std::remove_all_extents_t<INPUT>;
    constexpr int kThreadsM = TILE_M / kRows;
    constexpr int kThreadsN = BLOCK_X / kThreadsM;
    static_assert(kThreadsM * kRows == TILE_M, "a tile is whole rows a thread");
    static_assert(kThreadsM * kThreadsN == BLOCK_X, "the layout is the block");
    using V = vec<U, kPackElems>;
    using W = raw_t<sizeof(V)>;
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
        (output_slot < 0 || output_slot >= output_size_slot))
      internal::check_failed();
    const int tid = static_cast<int>(threadIdx.x);
    const auto row_at = [&](int i) {
      return offs_m +
             (kThreadsM == 1 ? i : tid / kThreadsN + i * kThreadsM) * row_step;
    };
    const auto col_at = [&](int k) {
      return offs_n / kPackElems + tid % kThreadsN + k * kThreadsN;
    };
    // The round: the next after the block's last (CONNECTION).
    const uint32_t last = static_cast<uint32_t>(connection[0][0][0]);
    const uint32_t next = last == kLastStep ? 3u : last + 1u;
    const auto counter = [&](int at, Counter kind, int peer) {
      if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
          (static_cast<int>(blockIdx.x) >= counters_size_block || peer < 0 ||
           peer >= counters_size_peer))
        internal::check_failed();
      return ptr_array_at(counters_ptr_array, counters_ptr_array_size, at) +
             static_cast<int64_t>(kind) * counters_stride_kind +
             int64_t{blockIdx.x} * counters_stride_block +
             int64_t{peer} * counters_stride_peer;
    };
    // Each wave waits until the receiver has read the last round (its head
    // here, in this rank's counters).
    uint32_t* const head = counter(output_slot, Counter::head, owner);
    const uint64_t t0 = wall_clock64();
    for (uint32_t n = 1;; ++n) {
      const uint32_t seen =
          __scoped_atomic_load_n(head, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
      if (HIP_CUSTOM_COMMUNICATOR_DEBUG && seen != last && seen != last - 1u &&
          !(last == 3u && seen == kLastStep))
        internal::check_failed();
      if (__builtin_amdgcn_ballot_w64(seen != last) == 0) break;
      if ((n & 255u) != 0) continue;
      const uint64_t waited = wall_clock64() - t0;
      const bool aborted =
          __scoped_atomic_load_n(
              status_ptr + static_cast<int64_t>(StatusField::abort) *
                               status_stride_field,
              __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM) != 0;
      if (__builtin_amdgcn_ballot_w64(aborted) == 0 && waited <= timeout_ticks)
        continue;
      if (!aborted)
        give_up(status_ptr, status_stride_field, status_size_field,
                Where::start, output_slot, static_cast<uint32_t>(owner), seen,
                last, waited);
      __builtin_amdgcn_endpgm();
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    char* const slot =
        reinterpret_cast<char*>(output) +
        int64_t{output_slot} * output_stride_slot * int64_t{sizeof(U)};
    const uint32_t row_bytes =
        static_cast<uint32_t>(output_stride_m * int64_t{sizeof(U)});
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      if (!(kRows == 1 && kThreadsM == 1) && row_at(i) >= clip_m) continue;
      if (HIP_CUSTOM_COMMUNICATOR_DEBUG && row_at(i) >= output_size_m)
        internal::check_failed();
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        if (col_at(k) >= clip_n / kPackElems) continue;
        W w;
        __builtin_memcpy(&w, input[i][k], sizeof(w));
        uint32_t at = static_cast<uint32_t>(row_at(i)) * row_bytes +
                      static_cast<uint32_t>(col_at(k)) * uint32_t{sizeof(V)};
        asm volatile("" : "+v"(at));
        *(__attribute__((address_space(1))) W*)(slot + at) = w;
      }
    }
    // The block's stores, then the round posted (the receiver's tail from this
    // rank), as barrier orders them.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    if constexpr (BLOCK_X > WAVE_SIZE) __syncthreads();
    if (threadIdx.x == 0)
      __scoped_atomic_store_n(counter(owner, Counter::tail, output_slot), next,
                              __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
  }
}

// A simple_read send: once every receiver has read the last round (their
// heads), the data into this rank's own inbox, at each receiver's slot, then
// the round posted to each (their tails, COUNTERS).
template <int WORLD, int TILE_M, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE,
          bool QUANTIZE_INT4, Layout LAYOUT, bool TAIL, typename INPUT,
          typename PTR>
DINLINE void send_simple_read(
    const INPUT& input, PTR const* output_ptr_array, int output_ptr_array_size,
    int output_rank, int output_rank_count, int output_slot,
    int64_t output_stride_slot, int64_t output_stride_m,
    int64_t output_stride_n, int output_size_slot, int output_size_m,
    int output_size_n, int clip_m, int clip_n, int offs_m, int offs_n,
    int row_step, Connection& connection, uint32_t* const* counters_ptr_array,
    int counters_ptr_array_size, int64_t counters_stride_kind,
    int64_t counters_stride_block, int64_t counters_stride_peer,
    int counters_size_block, int counters_size_peer, uint32_t* status_ptr,
    int64_t status_stride_field, int status_size_field,
    uint64_t timeout_ticks) {
  if constexpr (LAYOUT != Layout::tile) {
    // A flat walk: a peer a lane group, this lane's output_rank its group's
    // receiver; this rank is output_slot.
    static_assert(LAYOUT == Layout::run || LAYOUT == Layout::flat ||
                      LAYOUT == Layout::group_run ||
                      LAYOUT == Layout::group_flat,
                  "a flat walk: run or flat, or their INT4 groups");
    static_assert(BLOCK_Y == 1, "a flat walk is one row of threads");
    constexpr int kLanes = BLOCK_X / WORLD;
    static_assert(kLanes * WORLD == BLOCK_X,
                  "a block is a group of threads a rank");
    // The round: the next after the block's last (CONNECTION).
    const uint32_t last = static_cast<uint32_t>(connection[0][0][0]);
    const uint32_t next = last == kLastStep ? 3u : last + 1u;
    const int64_t at_block = int64_t{blockIdx.x} * counters_stride_block;
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
        (static_cast<int>(blockIdx.x) >= counters_size_block ||
         output_rank < 0 || output_rank >= counters_size_peer))
      internal::check_failed();
    // Each lane waits until its group's receiver has read the last round
    // (its head, in this rank's counters).
    uint32_t* const head =
        ptr_array_at(counters_ptr_array, counters_ptr_array_size, output_slot) +
        static_cast<int64_t>(Counter::head) * counters_stride_kind + at_block +
        int64_t{output_rank} * counters_stride_peer;
    const uint64_t t0 = wall_clock64();
    for (uint32_t n = 1;; ++n) {
      const uint32_t seen =
          __scoped_atomic_load_n(head, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
      if (HIP_CUSTOM_COMMUNICATOR_DEBUG && seen != last && seen != last - 1u &&
          !(last == 3u && seen == kLastStep))
        internal::check_failed();
      if (__builtin_amdgcn_ballot_w64(seen != last) == 0) break;
      if ((n & 255u) != 0) continue;
      const uint64_t waited = wall_clock64() - t0;
      const bool aborted =
          __scoped_atomic_load_n(
              status_ptr + static_cast<int64_t>(StatusField::abort) *
                               status_stride_field,
              __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM) != 0;
      if (__builtin_amdgcn_ballot_w64(aborted) == 0 && waited <= timeout_ticks)
        continue;
      if (!aborted)
        give_up(status_ptr, status_stride_field, status_size_field,
                Where::start, output_slot, static_cast<uint32_t>(output_rank),
                seen, last, waited);
      __builtin_amdgcn_endpgm();
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    // The data into this rank's own inbox, at the receiver's slot.
    send_simple<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE, QUANTIZE_INT4,
                LAYOUT, TAIL>(input, output_ptr_array, output_ptr_array_size,
                              output_slot, output_rank, output_stride_slot,
                              output_stride_m, output_stride_n,
                              output_size_slot, output_size_m, output_size_n,
                              clip_m, clip_n, offs_m, offs_n, row_step);
    // The block's stores, then each group's round posted (its receiver's
    // tail from this rank), by the group's first lane.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    if constexpr (BLOCK_X > WAVE_SIZE) __syncthreads();
    if (static_cast<int>(threadIdx.x) % kLanes == 0)
      __scoped_atomic_store_n(
          lane_ptr_array_at<uint32_t, WORLD>(
              counters_ptr_array, counters_ptr_array_size, output_rank) +
              static_cast<int64_t>(Counter::tail) * counters_stride_kind +
              at_block + int64_t{output_slot} * counters_stride_peer,
          next, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
  } else {
    static_assert(
        LAYOUT == Layout::tile && !QUANTIZE_INT4 && TILE_M == 1 && BLOCK_Y == 1,
        "a simple_read send is an exact tile of one row, from one "
        "row of threads");
    static_assert(BLOCK_X % WAVE_SIZE == 0, "a block is whole waves");
    const auto* const entries =
        (decltype(+output_ptr_array[0]) const*)output_ptr_array;
    // This rank's own staging: output_slot is this rank.
    void* const output =
        ptr_array_at(entries, output_ptr_array_size, output_slot);
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
        (output_rank_count < 1 || output_rank_count > WORLD))
      internal::check_failed();
#if defined(__AMDGCN_WAVEFRONT_SIZE)
    static_assert(__AMDGCN_WAVEFRONT_SIZE == WAVE_SIZE,
                  "the compiler's wave is not the one handed");
#endif
    constexpr int kRows = std::extent_v<INPUT, 0>;
    constexpr int kPacks = std::extent_v<INPUT, 1>;
    constexpr int kPackElems = std::extent_v<INPUT, 2>;
    using U = std::remove_all_extents_t<INPUT>;
    constexpr int kThreadsM = TILE_M / kRows;
    constexpr int kThreadsN = BLOCK_X / kThreadsM;
    static_assert(kThreadsM * kRows == TILE_M, "a tile is whole rows a thread");
    static_assert(kThreadsM * kThreadsN == BLOCK_X, "the layout is the block");
    using V = vec<U, kPackElems>;
    using W = raw_t<sizeof(V)>;
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
        (output_slot < 0 || output_slot >= output_size_slot))
      internal::check_failed();
    const int tid = static_cast<int>(threadIdx.x);
    const auto row_at = [&](int i) {
      return offs_m +
             (kThreadsM == 1 ? i : tid / kThreadsN + i * kThreadsM) * row_step;
    };
    const auto col_at = [&](int k) {
      return offs_n / kPackElems + tid % kThreadsN + k * kThreadsN;
    };
    // The round: the next after the block's last (CONNECTION).
    const uint32_t last = static_cast<uint32_t>(connection[0][0][0]);
    const uint32_t next = last == kLastStep ? 3u : last + 1u;
    const auto counter = [&](int at, Counter kind, int peer) {
      if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
          (static_cast<int>(blockIdx.x) >= counters_size_block || peer < 0 ||
           peer >= counters_size_peer))
        internal::check_failed();
      return ptr_array_at(counters_ptr_array, counters_ptr_array_size, at) +
             static_cast<int64_t>(kind) * counters_stride_kind +
             int64_t{blockIdx.x} * counters_stride_block +
             int64_t{peer} * counters_stride_peer;
    };
    // Each wave waits until every receiver has read the last round (their
    // heads here, in this rank's counters).
    const uint64_t t0 = wall_clock64();
    for (uint32_t n = 1;; ++n) {
      bool missing = false;
      uint32_t seen = last, peer = 0;
#pragma unroll
      for (int j = 0; j < WORLD; ++j) {
        if (j >= output_rank_count) break;
        const int reader = (output_rank + j) % WORLD;
        const uint32_t v =
            __scoped_atomic_load_n(counter(output_slot, Counter::head, reader),
                                   __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
        if (HIP_CUSTOM_COMMUNICATOR_DEBUG && v != last && v != last - 1u &&
            !(last == 3u && v == kLastStep))
          internal::check_failed();
        if (v != last) missing = true, seen = v, peer = reader;
      }
      if (__builtin_amdgcn_ballot_w64(missing) == 0) break;
      if ((n & 255u) != 0) continue;
      const uint64_t waited = wall_clock64() - t0;
      const bool aborted =
          __scoped_atomic_load_n(
              status_ptr + static_cast<int64_t>(StatusField::abort) *
                               status_stride_field,
              __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM) != 0;
      if (__builtin_amdgcn_ballot_w64(aborted) == 0 && waited <= timeout_ticks)
        continue;
      if (!aborted)
        give_up(status_ptr, status_stride_field, status_size_field,
                Where::start, output_slot, peer, seen, last, waited);
      __builtin_amdgcn_endpgm();
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    // A region a receiver, in this rank's own inbox: the slot is the receiver's
    // (a connection a pair of ranks, NCCL's), so a send to several stores a
    // copy for each.
    const uint32_t row_bytes =
        static_cast<uint32_t>(output_stride_m * int64_t{sizeof(U)});
#pragma unroll
    for (int j = 0; j < WORLD; ++j) {
      if (j >= output_rank_count) break;
      const int reader = (output_rank + j) % WORLD;
      char* const slot =
          reinterpret_cast<char*>(output) +
          int64_t{reader} * output_stride_slot * int64_t{sizeof(U)};
#pragma unroll
      for (int i = 0; i < kRows; ++i) {
        if (!(kRows == 1 && kThreadsM == 1) && row_at(i) >= clip_m) continue;
        if (HIP_CUSTOM_COMMUNICATOR_DEBUG && row_at(i) >= output_size_m)
          internal::check_failed();
#pragma unroll
        for (int k = 0; k < kPacks; ++k) {
          if (col_at(k) >= clip_n / kPackElems) continue;
          W w;
          __builtin_memcpy(&w, input[i][k], sizeof(w));
          uint32_t at = static_cast<uint32_t>(row_at(i)) * row_bytes +
                        static_cast<uint32_t>(col_at(k)) * uint32_t{sizeof(V)};
          asm volatile("" : "+v"(at));
          *(__attribute__((address_space(1))) W*)(slot + at) = w;
        }
      }
    }
    // The block's stores, then the round posted to each receiver (its tail from
    // this rank), as barrier orders them.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    if constexpr (BLOCK_X > WAVE_SIZE) __syncthreads();
    if (static_cast<int>(threadIdx.x) < output_rank_count) {
      const int reader = (output_rank + static_cast<int>(threadIdx.x)) % WORLD;
      uint32_t* const theirs = lane_ptr_array_at<uint32_t, WORLD>(
          counters_ptr_array, counters_ptr_array_size, reader);
      __scoped_atomic_store_n(
          theirs + static_cast<int64_t>(Counter::tail) * counters_stride_kind +
              int64_t{blockIdx.x} * counters_stride_block +
              int64_t{output_slot} * counters_stride_peer,
          next, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    }
  }
}

}  // namespace internal

template <int WORLD, int TILE_M, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE,
          bool QUANTIZE_INT4, Protocol PROTOCOL, Layout LAYOUT,
          bool TAIL = false, typename INPUT>
DINLINE void send(const INPUT& input, const ProtocolBuffers& protocol_buffers,
                  int output_rank, int output_rank_count, int rows, int cols) {
  // This rank's slot, and the block's own rows of it: a tile's TILE_M from the
  // block's first, a flat walk's one. Of them, `rows` exist, of `cols`
  // columns.
  const int output_slot = protocol_buffers.rank;
  const int offs_m =
      static_cast<int>(blockIdx.x) * (LAYOUT == Layout::tile ? TILE_M : 1);
  const int clip_m = offs_m + rows, clip_n = cols, offs_n = 0, row_step = 1;
  // The protocol's values, as the leaves take them.
  char* const* const output_ptr_array = protocol_buffers.inbox_ptr_array;
  const int output_ptr_array_size = protocol_buffers.inbox_ptr_array_size;
  const int64_t output_stride_slot = protocol_buffers.inbox_stride_slot,
                output_stride_m = protocol_buffers.inbox_stride_m,
                output_stride_n = protocol_buffers.inbox_stride_n;
  // An INT4 group walk's rows are every slot's: its scales follow them all.
  constexpr bool kGroups =
      LAYOUT == Layout::group_run || LAYOUT == Layout::group_flat;
  const int output_size_slot = protocol_buffers.inbox_size_slot,
            output_size_m = kGroups ? protocol_buffers.inbox_size_slot *
                                          protocol_buffers.inbox_size_m
                                    : protocol_buffers.inbox_size_m,
            output_size_n = protocol_buffers.inbox_size_n;
  uint32_t* const* const counters_ptr_array =
      protocol_buffers.counters_ptr_array;
  const int counters_ptr_array_size = protocol_buffers.counters_ptr_array_size;
  const int64_t counters_stride_kind = protocol_buffers.counters_stride_kind,
                counters_stride_block = protocol_buffers.counters_stride_block,
                counters_stride_peer = protocol_buffers.counters_stride_peer;
  const int counters_size_block = protocol_buffers.counters_size_block,
            counters_size_peer = protocol_buffers.counters_size_peer;
  uint32_t* const status_ptr = protocol_buffers.status_ptr;
  const int64_t status_stride_field = protocol_buffers.status_stride_field;
  const int status_size_field = protocol_buffers.status_size_field;
  const uint64_t timeout_ticks = protocol_buffers.timeout_ticks;
  // This wave's step (STEPS), as the leaves take it.
  constexpr int kWaves = BLOCK_X * BLOCK_Y / WAVE_SIZE;
  static_assert(kWaves >= 1 && kWaves <= kStepWaves,
                "a block's waves have their steps");
  const int wave = __builtin_amdgcn_readfirstlane(
      static_cast<int>(threadIdx.y * BLOCK_X + threadIdx.x) / WAVE_SIZE);
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
      (static_cast<int>(blockIdx.x) >= protocol_buffers.steps_size_block ||
       protocol_buffers.steps_size_wave != kStepWaves))
    internal::check_failed();
  uint32_t* const block_steps =
      protocol_buffers.steps_ptr +
      int64_t{blockIdx.x} * protocol_buffers.steps_stride_block;
  uint32_t* const step_at =
      block_steps + int64_t{wave} * protocol_buffers.steps_stride_wave;
  Connection step{};
  step[0][0][0] = static_cast<int32_t>(
      __scoped_atomic_load_n(step_at, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM));
  Connection* const connection = &step;
  // simple_read waits on and posts to every receiver at once: the leaf takes
  // the set.
  if constexpr (PROTOCOL == Protocol::simple_read) {
    internal::send_simple_read<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE,
                               QUANTIZE_INT4, LAYOUT, TAIL>(
        input, output_ptr_array, output_ptr_array_size, output_rank,
        output_rank_count, output_slot, output_stride_slot, output_stride_m,
        output_stride_n, output_size_slot, output_size_m, output_size_n, clip_m,
        clip_n, offs_m, offs_n, row_step, *connection, counters_ptr_array,
        counters_ptr_array_size, counters_stride_kind, counters_stride_block,
        counters_stride_peer, counters_size_block, counters_size_peer,
        status_ptr, status_stride_field, status_size_field, timeout_ticks);
  } else {
    // The ranks output_rank on, output_rank_count of them, a leaf call each.
#pragma unroll
    for (int j = 0; j < output_rank_count; ++j) {
      const int r = j == 0 ? output_rank : (output_rank + j) % WORLD;
      if constexpr (PROTOCOL == Protocol::ll) {
        internal::send_ll<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE,
                          QUANTIZE_INT4, LAYOUT, TAIL>(
            input, output_ptr_array, output_ptr_array_size, r, output_slot,
            output_stride_slot, output_stride_m, output_stride_n,
            output_size_slot, output_size_m, output_size_n, clip_m, clip_n,
            offs_m, offs_n, row_step, *connection);
      } else if constexpr (PROTOCOL == Protocol::simple_write) {
        internal::send_simple_write<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE,
                                    QUANTIZE_INT4, LAYOUT, TAIL>(
            input, output_ptr_array, output_ptr_array_size, r, output_slot,
            output_stride_slot, output_stride_m, output_stride_n,
            output_size_slot, output_size_m, output_size_n, clip_m, clip_n,
            offs_m, offs_n, row_step, *connection, counters_ptr_array,
            counters_ptr_array_size, counters_stride_kind,
            counters_stride_block, counters_stride_peer, counters_size_block,
            counters_size_peer, status_ptr, status_stride_field,
            status_size_field, timeout_ticks);
      } else {
        internal::send_lamport<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE,
                               QUANTIZE_INT4, LAYOUT, TAIL>(
            input, output_ptr_array, output_ptr_array_size, r, output_slot,
            output_stride_slot, output_stride_m, output_stride_n,
            output_size_slot, output_size_m, output_size_n, clip_m, clip_n,
            offs_m, offs_n, row_step, *connection);
      }
    }
  }
}

}  // namespace hip_custom_communicator
