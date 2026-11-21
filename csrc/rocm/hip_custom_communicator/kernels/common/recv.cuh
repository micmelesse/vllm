#pragma once

// recv: a peer's data into registers, by its protocol (recv_ll, recv_lamport,
// recv_simple_write, recv_simple_read; recv_simple moves their data).

#include "dequantize.cuh"
#include "types.cuh"

namespace hip_custom_communicator {

namespace internal {

// A Simple recv: a load from the peer's memory, in any layout.
template <int WORLD, int TILE_M, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE,
          bool QUANTIZE_INT4, Layout LAYOUT, bool TAIL, typename PTR,
          typename OUTPUT>
DINLINE void recv_simple(PTR const* input_ptr_array, int input_ptr_array_size,
                         int input_rank, int input_slot,
                         int64_t input_stride_slot, int64_t input_stride_m,
                         int64_t input_stride_n, int input_size_slot,
                         int input_size_m, int input_size_n, int clip_m,
                         int clip_n, OUTPUT& output, int offs_m, int offs_n,
                         int row_step) {
  // One row of threads (BLOCK_Y = 1): a row split of Simple is not written yet.
  static_assert(BLOCK_Y == 1, "a Simple recv is one row of threads");
  constexpr bool kSlots = std::rank_v<OUTPUT> == 4;
  constexpr int kPacks = std::extent_v < OUTPUT, kSlots ? 2 : 1 > ;
  if constexpr ((LAYOUT == Layout::group_run || LAYOUT == Layout::group_flat) &&
                kPacks != kInt4GroupPacks) {
    static_assert(kPacks % kInt4GroupPacks == 0, "a thread holds whole groups");
    constexpr int kCopies = kSlots ? std::extent_v<OUTPUT, 0> : 1;
    constexpr int kPackElems = std::extent_v < OUTPUT, kSlots ? 3 : 2 > ;
    using U = std::remove_all_extents_t<OUTPUT>;
    using Group =
        std::conditional_t<kSlots, U[kCopies][1][kInt4GroupPacks][kPackElems],
                           U[1][kInt4GroupPacks][kPackElems]>;
#pragma unroll
    for (int jg = 0; jg < kPacks / kInt4GroupPacks; ++jg) {
      Group group;
      recv_simple<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE, QUANTIZE_INT4,
                  LAYOUT, TAIL>(
          input_ptr_array, input_ptr_array_size, input_rank, input_slot,
          input_stride_slot, input_stride_m, input_stride_n, input_size_slot,
          input_size_m, input_size_n, clip_m, clip_n, group, offs_m,
          offs_n + jg * (BLOCK_X / TILE_M) * kInt4Group, row_step);
#pragma unroll
      for (int c = 0; c < kCopies; ++c)
#pragma unroll
        for (int k = 0; k < kInt4GroupPacks; ++k)
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) {
            if constexpr (kSlots)
              output[c][0][jg * kInt4GroupPacks + k][j] = group[c][0][k][j];
            else
              output[0][jg * kInt4GroupPacks + k][j] = group[0][k][j];
          }
    }
  } else {
    const auto* const entries =
        (decltype(+input_ptr_array[0]) const*)input_ptr_array;
    const int first = __builtin_amdgcn_readfirstlane(input_rank);
    if ((__builtin_constant_p(input_ptr_array_size) &&
         input_ptr_array_size == 1) ||
        __builtin_amdgcn_ballot_w64(input_rank != first) == 0) {
      // One peer for the wave: its memory's address is scalar.
      const void* const input =
          __builtin_constant_p(input_ptr_array_size) &&
                  input_ptr_array_size == 1
              ? entries[0]
              : ptr_array_at(entries, input_ptr_array_size, first);
#if defined(__AMDGCN_WAVEFRONT_SIZE)
      static_assert(__AMDGCN_WAVEFRONT_SIZE == WAVE_SIZE,
                    "the compiler's wave is not the one handed");
#endif
      constexpr bool kSlots = std::rank_v<OUTPUT> == 4;
      constexpr int kCopies = kSlots ? std::extent_v<OUTPUT, 0> : 1;
      constexpr int kRows = std::extent_v < OUTPUT, kSlots ? 1 : 0 > ;
      constexpr int kPacks = std::extent_v < OUTPUT, kSlots ? 2 : 1 > ;
      constexpr int kPackElems = std::extent_v < OUTPUT, kSlots ? 3 : 2 > ;
      using U = std::remove_all_extents_t<OUTPUT>;
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
      using E = std::conditional_t<!QUANTIZE_INT4, const U, const uint8_t>;
      const auto base = [&](int c) -> E* {
        const int slot = kSlots ? (input_slot + c) % WORLD : input_slot;
        if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
            (slot < 0 || slot >= input_size_slot))
          internal::check_failed();
        return reinterpret_cast<E*>(input) + slot * input_stride_slot;
      };
      const int tid = static_cast<int>(threadIdx.x);
      const auto row_at = [&](int i) {
        return offs_m + (kThreadsM == 1 ? i : tid / kThreadsN + i * kThreadsM) *
                            row_step;
      };
      const auto row_of = [&](int i) {
        return row_at(i) < clip_m ? row_at(i) : clip_m - 1;
      };
      const auto skips = [&](int i) {
        return !(kRows == 1 && kThreadsM == 1) && row_at(i) >= clip_m;
      };
      constexpr int kElems = kRaw ? kInt4Group / kInt4GroupPacks : kPackElems;
      const auto pack_of = [&](int k) {
        const int n = offs_n / kElems + tid % kThreadsN + k * kThreadsN;
        return n < clip_n / kElems ? n : clip_n / kElems - 1;
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
        static_assert(!QUANTIZE_INT4 && !kSlots && kRows == 1,
                      "the flat walk reads one exact copy a row of threads");

        constexpr int kLanes = BLOCK_X / WORLD;
        static_assert(kLanes * WORLD == BLOCK_X,
                      "a block is a group of threads a rank");
        const U* const from = reinterpret_cast<const U*>(input) +
                              int64_t{offs_m} * input_stride_m;
        const int64_t first = int64_t{input_slot} * input_stride_slot;
        if constexpr (kRun) {
#pragma unroll
          for (int k = 0; k < kPacks; ++k) {
            const int last = clip_n / kPackElems - 1;
            const V* run =
                reinterpret_cast<const V*>(from + first * input_stride_n);
            const int n = offs_n / kPackElems + tid % kLanes + k * kLanes;
            const int col = n < clip_n / kPackElems ? n : last;
            uint32_t offset = static_cast<uint32_t>(col) * uint32_t{sizeof(V)};
            asm volatile("" : "+v"(offset));
            V v;
            {
              using W = std::conditional_t<
                  sizeof(V) == 16, u32x4,
                  std::conditional_t<
                      sizeof(V) == 8, u32x2,
                      std::conditional_t<sizeof(V) == 4, uint32_t, uint16_t>>>;
              const W raw = *(const __attribute__((address_space(1)))
                              W*)(reinterpret_cast<const char*>(run) + offset);
              __builtin_memcpy(&v, &raw, sizeof(V));
            }
            __builtin_memcpy(output[0][k], &v, sizeof(v));
          }
        } else {
#pragma unroll
          for (int k = 0; k < kPacks; ++k) {
            const int n = offs_n / kPackElems + tid % kLanes + k * kLanes;
            const int col =
                n < clip_n / kPackElems ? n : clip_n / kPackElems - 1;
            V v;
            {
              using W = std::conditional_t<
                  sizeof(V) == 16, u32x4,
                  std::conditional_t<
                      sizeof(V) == 8, u32x2,
                      std::conditional_t<sizeof(V) == 4, uint32_t, uint16_t>>>;
              const W raw =
                  *(const __attribute__((address_space(1))) W*)(element_at(
                      from, input_stride_m, input_stride_n,
                      first + int64_t{col} * kPackElems, input_size_n));
              __builtin_memcpy(&v, &raw, sizeof(V));
            }
            __builtin_memcpy(output[0][k], &v, sizeof(v));
          }
        }

        asm volatile("" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
      } else if constexpr (kGroup) {
        static_assert(
            kRows == 1 && kPacks == kInt4GroupPacks,
            "a group a thread here: several are load's, recv's, store's "
            "and send's loop");

        const int g = offs_n / kInt4Group + tid % kThreadsN;

        const int at = TAIL ? min(g, static_cast<int>(int4_groups(clip_n)) - 1)
                            : min(g, clip_n / kInt4Group - 1);
        if constexpr (!QUANTIZE_INT4) {
          static_assert(!kSlots, "an exact group is one copy");
          const U* const from = reinterpret_cast<const U*>(input) +
                                int64_t{offs_m} * input_stride_m;
          const int64_t first = int64_t{input_slot} * input_stride_slot;
          if constexpr (TAIL) {
            [[maybe_unused]] const V* group = nullptr;
            if constexpr (kRun) {
              uint32_t offset =
                  static_cast<uint32_t>(at) * (kInt4Group * sizeof(U));
              asm volatile("" : "+v"(offset));
              group = reinterpret_cast<const V*>(
                  reinterpret_cast<const char*>(from + first * input_stride_n) +
                  offset);
            }
            const auto pack_at = [&](int kk) -> const V* {
              if constexpr (kRun)
                return group + kk;
              else
                return reinterpret_cast<const V*>(element_at(
                    from, input_stride_m, input_stride_n,
                    first + int64_t{at} * kInt4Group + int64_t{kk} * kPackElems,
                    input_size_n));
            };

            static_assert(sizeof(V) % sizeof(u32x4) == 0,
                          "a pack is whole 16-byte loads");
            const int last = clip_n / kPackElems - 1 - at * kInt4GroupPacks;
#pragma unroll
            for (int k = 0; k < kInt4GroupPacks; ++k) {
              const int kk = k < last ? k : last;
              if constexpr (sizeof(V) == sizeof(u32x4)) {
                const u32x4 raw = *(const global_u32x4*)(pack_at(kk));
                __builtin_memcpy(output[0][k], &raw, sizeof(raw));
              } else {
#pragma unroll
                for (int h = 0; h < static_cast<int>(sizeof(V) / sizeof(u32x4));
                     ++h) {
                  const u32x4 raw =
                      *(const global_u32x4*)(reinterpret_cast<const char*>(
                                                 pack_at(kk)) +
                                             h * sizeof(u32x4));
                  __builtin_memcpy(
                      reinterpret_cast<char*>(output[0][k]) + h * sizeof(u32x4),
                      &raw, sizeof(raw));
                }
              }
            }
#pragma unroll
            for (int k = 0; k < kInt4GroupPacks; ++k)
              if (k > last)
#pragma unroll
                for (int j = 0; j < kPackElems; ++j)
                  output[0][k][j] = static_cast<U>(0.0f);
          } else {
            const V* group;
            if constexpr (kRun) {
              uint32_t offset =
                  static_cast<uint32_t>(at) * (kInt4Group * sizeof(U));
              asm volatile("" : "+v"(offset));
              group = reinterpret_cast<const V*>(
                  reinterpret_cast<const char*>(from + first * input_stride_n) +
                  offset);
            } else
              group = reinterpret_cast<const V*>(
                  element_at(from, input_stride_m, input_stride_n,
                             first + int64_t{at} * kInt4Group, input_size_n));

            static_assert(sizeof(V) % sizeof(u32x4) == 0,
                          "a pack is whole 16-byte loads");
#pragma unroll
            for (int k = 0; k < kInt4GroupPacks; ++k) {
              if constexpr (sizeof(V) == sizeof(u32x4)) {
                const u32x4 raw = *(const global_u32x4*)(group + k);
                __builtin_memcpy(output[0][k], &raw, sizeof(raw));
              } else {
#pragma unroll
                for (int h = 0; h < static_cast<int>(sizeof(V) / sizeof(u32x4));
                     ++h) {
                  const u32x4 raw =
                      *(const global_u32x4*)(reinterpret_cast<const char*>(
                                                 group + k) +
                                             h * sizeof(u32x4));
                  __builtin_memcpy(
                      reinterpret_cast<char*>(output[0][k]) + h * sizeof(u32x4),
                      &raw, sizeof(raw));
                }
              }
            }
          }
        } else {
          static_assert(kRaw && kRun,
                        "an INT4 group is read raw from a dense payload");
          const int64_t scales_offset =
              int4_scales_offset(input_size_m, input_size_n);
          const uint8_t* values[kCopies];
          const __fp16* scales[kCopies];
#pragma unroll
          for (int c = 0; c < kCopies; ++c) {
            const int64_t slot =
                int64_t{kSlots ? (input_slot + c) % WORLD : input_slot} *
                    input_stride_slot +
                int64_t{offs_m} * input_stride_m;
            values[c] = reinterpret_cast<const uint8_t*>(input) + slot / 2;
            scales[c] =
                reinterpret_cast<const __fp16*>(
                    reinterpret_cast<const uint8_t*>(input) + scales_offset) +
                slot / kInt4Group;
          }
          uint32_t value_at = static_cast<uint32_t>(at) * 16;
          uint32_t scale_at = static_cast<uint32_t>(at) * kInt4ScaleBytes;
          asm volatile("" : "+v"(value_at));
          asm volatile("" : "+v"(scale_at));
#pragma unroll
          for (int c = 0; c < kCopies; ++c) {
            const u32x4 words = *(const global_u32x4*)(values[c] + value_at);

            const uint32_t scale = __builtin_bit_cast(
                uint16_t,
                *(const global_f16*)(reinterpret_cast<const char*>(scales[c]) +
                                     scale_at));
            auto& out = [&]() -> U(&)[kPacks][kPackElems] {
              if constexpr (kSlots)
                return output[c][0];
              else
                return output[0];
            }();

            const int last = clip_n / (kInt4Group / kInt4GroupPacks) - 1 -
                             at * kInt4GroupPacks;
#pragma unroll
            for (int k = 0; k < kInt4GroupPacks; ++k) {
              out[k][0] = TAIL && k > last ? kInt4ZeroWord : words[k];
              out[k][1] = scale;
            }
          }

          asm volatile("" ::: "memory");
          __builtin_amdgcn_sched_barrier(0);
        }
      } else if constexpr (!QUANTIZE_INT4 && !kSlots) {
        const V* at = reinterpret_cast<const V*>(base(0));
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
          constexpr bool kLaneRow = !(kRows == 1 && kThreadsM == 1);
          uint32_t lane_offset[kPacks];
          if constexpr (kLaneRow) {
            if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
                int64_t{row_of(i)} * input_stride_m * int64_t{sizeof(U)} +
                        input_size_n * int64_t{sizeof(U)} >
                    (int64_t{1} << 32))
              internal::check_failed();
            const uint32_t row_bytes =
                static_cast<uint32_t>(row_of(i)) *
                static_cast<uint32_t>(input_stride_m * int64_t{sizeof(U)});
#pragma unroll
            for (int k = 0; k < kPacks; ++k) {
              lane_offset[k] = row_bytes + static_cast<uint32_t>(pack_of(k)) *
                                               uint32_t{sizeof(V)};
              asm("" : "+v"(lane_offset[k]));
            }
          }
          if (skips(i)) continue;
          const V* row = kLaneRow ? at
                                  : at + int64_t{row_of(i)} *
                                             (input_stride_m / kPackElems);
#pragma unroll
          for (int k = 0; k < kPacks; ++k) {
            uint32_t offset = 0;
            if constexpr (kLaneRow)
              offset = lane_offset[k];
            else {
              offset = static_cast<uint32_t>(pack_of(k)) * uint32_t{sizeof(V)};
              asm volatile("" : "+v"(offset));
            }
            const V* p = reinterpret_cast<const V*>(
                reinterpret_cast<const char*>(row) + offset);
            V v;
            if constexpr (sizeof(V) == 16) {
              const u32x4 raw = *(const global_u32x4*)(p);
              __builtin_memcpy(&v, &raw, 16);
            } else {
              v = *p;
            }
            __builtin_memcpy(output[i][k], &v, sizeof(v));
          }
        }

        asm volatile("" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
      } else if constexpr (!QUANTIZE_INT4) {
        const U* at_copy[kCopies];
#pragma unroll
        for (int c = 0; c < kCopies; ++c) at_copy[c] = base(c);
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
          if (skips(i)) continue;
#pragma unroll
          for (int k = 0; k < kPacks; ++k) {
            int64_t at =
                int64_t{row_of(i)} * (input_stride_m / kPackElems) + pack_of(k);
            asm volatile("" : "+v"(at));
#pragma unroll
            for (int c = 0; c < kCopies; ++c) {
              const V* p = reinterpret_cast<const V*>(at_copy[c]) + at;
              V v;
              if constexpr (sizeof(V) == 16) {
                const u32x4 raw = *(const global_u32x4*)(p);
                __builtin_memcpy(&v, &raw, 16);
              } else {
                v = *p;
              }
              __builtin_memcpy(output[c][i][k], &v, sizeof(v));
            }
          }
        }
        asm volatile("" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
      } else {
        static_assert(kThreadsN % kInt4GroupPacks == 0,
                      "a group is 4 lanes' packs");

        const int64_t values_stride_m = input_size_n / 2;
        const int64_t scales_stride_m =
            TAIL ? int4_groups(input_size_n) : input_size_n / kInt4Group;
        const int64_t scales_at =
            int4_scales_offset(input_size_m, input_size_n);

        const uint8_t* values[kCopies];
        const char* scales[kCopies];
#pragma unroll
        for (int c = 0; c < kCopies; ++c) values[c] = base(c);
#pragma unroll
        for (int c = 0; c < kCopies; ++c)
          scales[c] = reinterpret_cast<const char*>(values[c] + scales_at);
        uint32_t words[kCopies][kRows][kPacks];
        uint32_t group_scales[kCopies][kRows][kPacks];
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
          if (skips(i)) continue;
#pragma unroll
          for (int k = 0; k < kPacks; ++k) {
            const int64_t row = row_of(i);
            const int pack = pack_of(k);
            uint32_t value_at = static_cast<uint32_t>(row * values_stride_m) +
                                static_cast<uint32_t>(pack) * kInt4WordBytes;
            uint32_t scale_at = static_cast<uint32_t>(
                (row * scales_stride_m + pack / kInt4GroupPacks) *
                kInt4ScaleBytes);
            asm volatile("" : "+v"(value_at));
            asm volatile("" : "+v"(scale_at));
#pragma unroll
            for (int c = 0; c < kCopies; ++c) {
              words[c][i][k] = *(const global_u32*)(values[c] + value_at);
              group_scales[c][i][k] = __builtin_bit_cast(
                  uint16_t, *(const global_f16*)(scales[c] + scale_at));
            }
          }
        }
        asm volatile("" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int c = 0; c < kCopies; ++c)
#pragma unroll
          for (int i = 0; i < kRows; ++i)
#pragma unroll
            for (int k = 0; k < kPacks; ++k) {
              auto& out = [&]() -> U(&)[kPackElems] {
                if constexpr (kSlots)
                  return output[c][i][k];
                else
                  return output[i][k];
              }();
              const uint32_t encoded[2] = {words[c][i][k],
                                           group_scales[c][i][k]};
              if constexpr (kRaw) {
                out[0] = encoded[0];
                out[1] = encoded[1];
              } else {
                dequantize(encoded, out);
              }
            }
      }
    } else {
      // A peer a lane.
      const void* const input = lane_ptr_array_at<void, WORLD>(
          entries, input_ptr_array_size, input_rank);
#if defined(__AMDGCN_WAVEFRONT_SIZE)
      static_assert(__AMDGCN_WAVEFRONT_SIZE == WAVE_SIZE,
                    "the compiler's wave is not the one handed");
#endif
      constexpr bool kSlots = std::rank_v<OUTPUT> == 4;
      constexpr int kCopies = kSlots ? std::extent_v<OUTPUT, 0> : 1;
      constexpr int kRows = std::extent_v < OUTPUT, kSlots ? 1 : 0 > ;
      constexpr int kPacks = std::extent_v < OUTPUT, kSlots ? 2 : 1 > ;
      constexpr int kPackElems = std::extent_v < OUTPUT, kSlots ? 3 : 2 > ;
      using U = std::remove_all_extents_t<OUTPUT>;
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
      using E = std::conditional_t<!QUANTIZE_INT4, const U, const uint8_t>;
      const auto base = [&](int c) -> E* {
        const int slot = kSlots ? (input_slot + c) % WORLD : input_slot;
        if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
            (slot < 0 || slot >= input_size_slot))
          internal::check_failed();
        return reinterpret_cast<E*>(input) + slot * input_stride_slot;
      };
      const int tid = static_cast<int>(threadIdx.x);
      const auto row_at = [&](int i) {
        return offs_m + (kThreadsM == 1 ? i : tid / kThreadsN + i * kThreadsM) *
                            row_step;
      };
      const auto row_of = [&](int i) {
        return row_at(i) < clip_m ? row_at(i) : clip_m - 1;
      };
      const auto skips = [&](int i) {
        return !(kRows == 1 && kThreadsM == 1) && row_at(i) >= clip_m;
      };
      constexpr int kElems = kRaw ? kInt4Group / kInt4GroupPacks : kPackElems;
      const auto pack_of = [&](int k) {
        const int n = offs_n / kElems + tid % kThreadsN + k * kThreadsN;
        return n < clip_n / kElems ? n : clip_n / kElems - 1;
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
        static_assert(!QUANTIZE_INT4 && !kSlots && kRows == 1,
                      "the flat walk reads one exact copy a row of threads");

        constexpr int kLanes = BLOCK_X / WORLD;
        static_assert(kLanes * WORLD == BLOCK_X,
                      "a block is a group of threads a rank");
        const U* const from = reinterpret_cast<const U*>(input) +
                              int64_t{offs_m} * input_stride_m;
        const int64_t first = int64_t{input_slot} * input_stride_slot;
        if constexpr (kRun) {
#pragma unroll
          for (int k = 0; k < kPacks; ++k) {
            const int last = clip_n / kPackElems - 1;
            const V* run =
                reinterpret_cast<const V*>(from + first * input_stride_n);
            const int n = offs_n / kPackElems + tid % kLanes + k * kLanes;
            const int col = n < clip_n / kPackElems ? n : last;
            uint32_t offset = static_cast<uint32_t>(col) * uint32_t{sizeof(V)};
            asm volatile("" : "+v"(offset));
            V v;
            {
              using W = std::conditional_t<
                  sizeof(V) == 16, u32x4,
                  std::conditional_t<
                      sizeof(V) == 8, u32x2,
                      std::conditional_t<sizeof(V) == 4, uint32_t, uint16_t>>>;
              const W raw = *(const __attribute__((address_space(1)))
                              W*)(reinterpret_cast<const char*>(run) + offset);
              __builtin_memcpy(&v, &raw, sizeof(V));
            }
            __builtin_memcpy(output[0][k], &v, sizeof(v));
          }
        } else {
#pragma unroll
          for (int k = 0; k < kPacks; ++k) {
            const int n = offs_n / kPackElems + tid % kLanes + k * kLanes;
            const int col =
                n < clip_n / kPackElems ? n : clip_n / kPackElems - 1;
            V v;
            {
              using W = std::conditional_t<
                  sizeof(V) == 16, u32x4,
                  std::conditional_t<
                      sizeof(V) == 8, u32x2,
                      std::conditional_t<sizeof(V) == 4, uint32_t, uint16_t>>>;
              const W raw =
                  *(const __attribute__((address_space(1))) W*)(element_at(
                      from, input_stride_m, input_stride_n,
                      first + int64_t{col} * kPackElems, input_size_n));
              __builtin_memcpy(&v, &raw, sizeof(V));
            }
            __builtin_memcpy(output[0][k], &v, sizeof(v));
          }
        }

        asm volatile("" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
      } else if constexpr (kGroup) {
        static_assert(
            kRows == 1 && kPacks == kInt4GroupPacks,
            "a group a thread here: several are load's, recv's, store's "
            "and send's loop");

        const int g = offs_n / kInt4Group + tid % kThreadsN;

        const int at = TAIL ? min(g, static_cast<int>(int4_groups(clip_n)) - 1)
                            : min(g, clip_n / kInt4Group - 1);
        if constexpr (!QUANTIZE_INT4) {
          static_assert(!kSlots, "an exact group is one copy");
          const U* const from = reinterpret_cast<const U*>(input) +
                                int64_t{offs_m} * input_stride_m;
          const int64_t first = int64_t{input_slot} * input_stride_slot;
          if constexpr (TAIL) {
            [[maybe_unused]] const V* group = nullptr;
            if constexpr (kRun) {
              uint32_t offset =
                  static_cast<uint32_t>(at) * (kInt4Group * sizeof(U));
              asm volatile("" : "+v"(offset));
              group = reinterpret_cast<const V*>(
                  reinterpret_cast<const char*>(from + first * input_stride_n) +
                  offset);
            }
            const auto pack_at = [&](int kk) -> const V* {
              if constexpr (kRun)
                return group + kk;
              else
                return reinterpret_cast<const V*>(element_at(
                    from, input_stride_m, input_stride_n,
                    first + int64_t{at} * kInt4Group + int64_t{kk} * kPackElems,
                    input_size_n));
            };

            static_assert(sizeof(V) % sizeof(u32x4) == 0,
                          "a pack is whole 16-byte loads");
            const int last = clip_n / kPackElems - 1 - at * kInt4GroupPacks;
#pragma unroll
            for (int k = 0; k < kInt4GroupPacks; ++k) {
              const int kk = k < last ? k : last;
              if constexpr (sizeof(V) == sizeof(u32x4)) {
                const u32x4 raw = *(const global_u32x4*)(pack_at(kk));
                __builtin_memcpy(output[0][k], &raw, sizeof(raw));
              } else {
#pragma unroll
                for (int h = 0; h < static_cast<int>(sizeof(V) / sizeof(u32x4));
                     ++h) {
                  const u32x4 raw =
                      *(const global_u32x4*)(reinterpret_cast<const char*>(
                                                 pack_at(kk)) +
                                             h * sizeof(u32x4));
                  __builtin_memcpy(
                      reinterpret_cast<char*>(output[0][k]) + h * sizeof(u32x4),
                      &raw, sizeof(raw));
                }
              }
            }
#pragma unroll
            for (int k = 0; k < kInt4GroupPacks; ++k)
              if (k > last)
#pragma unroll
                for (int j = 0; j < kPackElems; ++j)
                  output[0][k][j] = static_cast<U>(0.0f);
          } else {
            const V* group;
            if constexpr (kRun) {
              uint32_t offset =
                  static_cast<uint32_t>(at) * (kInt4Group * sizeof(U));
              asm volatile("" : "+v"(offset));
              group = reinterpret_cast<const V*>(
                  reinterpret_cast<const char*>(from + first * input_stride_n) +
                  offset);
            } else
              group = reinterpret_cast<const V*>(
                  element_at(from, input_stride_m, input_stride_n,
                             first + int64_t{at} * kInt4Group, input_size_n));

            static_assert(sizeof(V) % sizeof(u32x4) == 0,
                          "a pack is whole 16-byte loads");
#pragma unroll
            for (int k = 0; k < kInt4GroupPacks; ++k) {
              if constexpr (sizeof(V) == sizeof(u32x4)) {
                const u32x4 raw = *(const global_u32x4*)(group + k);
                __builtin_memcpy(output[0][k], &raw, sizeof(raw));
              } else {
#pragma unroll
                for (int h = 0; h < static_cast<int>(sizeof(V) / sizeof(u32x4));
                     ++h) {
                  const u32x4 raw =
                      *(const global_u32x4*)(reinterpret_cast<const char*>(
                                                 group + k) +
                                             h * sizeof(u32x4));
                  __builtin_memcpy(
                      reinterpret_cast<char*>(output[0][k]) + h * sizeof(u32x4),
                      &raw, sizeof(raw));
                }
              }
            }
          }
        } else {
          static_assert(kRaw && kRun,
                        "an INT4 group is read raw from a dense payload");
          const int64_t scales_offset =
              int4_scales_offset(input_size_m, input_size_n);
          const uint8_t* values[kCopies];
          const __fp16* scales[kCopies];
#pragma unroll
          for (int c = 0; c < kCopies; ++c) {
            const int64_t slot =
                int64_t{kSlots ? (input_slot + c) % WORLD : input_slot} *
                    input_stride_slot +
                int64_t{offs_m} * input_stride_m;
            values[c] = reinterpret_cast<const uint8_t*>(input) + slot / 2;
            scales[c] =
                reinterpret_cast<const __fp16*>(
                    reinterpret_cast<const uint8_t*>(input) + scales_offset) +
                slot / kInt4Group;
          }
          uint32_t value_at = static_cast<uint32_t>(at) * 16;
          uint32_t scale_at = static_cast<uint32_t>(at) * kInt4ScaleBytes;
          asm volatile("" : "+v"(value_at));
          asm volatile("" : "+v"(scale_at));
#pragma unroll
          for (int c = 0; c < kCopies; ++c) {
            const u32x4 words = *(const global_u32x4*)(values[c] + value_at);

            const uint32_t scale = __builtin_bit_cast(
                uint16_t,
                *(const global_f16*)(reinterpret_cast<const char*>(scales[c]) +
                                     scale_at));
            auto& out = [&]() -> U(&)[kPacks][kPackElems] {
              if constexpr (kSlots)
                return output[c][0];
              else
                return output[0];
            }();

            const int last = clip_n / (kInt4Group / kInt4GroupPacks) - 1 -
                             at * kInt4GroupPacks;
#pragma unroll
            for (int k = 0; k < kInt4GroupPacks; ++k) {
              out[k][0] = TAIL && k > last ? kInt4ZeroWord : words[k];
              out[k][1] = scale;
            }
          }

          asm volatile("" ::: "memory");
          __builtin_amdgcn_sched_barrier(0);
        }
      } else if constexpr (!QUANTIZE_INT4 && !kSlots) {
        const V* at = reinterpret_cast<const V*>(base(0));
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
          constexpr bool kLaneRow = !(kRows == 1 && kThreadsM == 1);
          uint32_t lane_offset[kPacks];
          if constexpr (kLaneRow) {
            if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
                int64_t{row_of(i)} * input_stride_m * int64_t{sizeof(U)} +
                        input_size_n * int64_t{sizeof(U)} >
                    (int64_t{1} << 32))
              internal::check_failed();
            const uint32_t row_bytes =
                static_cast<uint32_t>(row_of(i)) *
                static_cast<uint32_t>(input_stride_m * int64_t{sizeof(U)});
#pragma unroll
            for (int k = 0; k < kPacks; ++k) {
              lane_offset[k] = row_bytes + static_cast<uint32_t>(pack_of(k)) *
                                               uint32_t{sizeof(V)};
              asm("" : "+v"(lane_offset[k]));
            }
          }
          if (skips(i)) continue;
          const V* row = kLaneRow ? at
                                  : at + int64_t{row_of(i)} *
                                             (input_stride_m / kPackElems);
#pragma unroll
          for (int k = 0; k < kPacks; ++k) {
            uint32_t offset = 0;
            if constexpr (kLaneRow)
              offset = lane_offset[k];
            else {
              offset = static_cast<uint32_t>(pack_of(k)) * uint32_t{sizeof(V)};
              asm volatile("" : "+v"(offset));
            }
            const V* p = reinterpret_cast<const V*>(
                reinterpret_cast<const char*>(row) + offset);
            V v;
            if constexpr (sizeof(V) == 16) {
              const u32x4 raw = *(const global_u32x4*)(p);
              __builtin_memcpy(&v, &raw, 16);
            } else {
              v = *p;
            }
            __builtin_memcpy(output[i][k], &v, sizeof(v));
          }
        }

        asm volatile("" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
      } else if constexpr (!QUANTIZE_INT4) {
        const U* at_copy[kCopies];
#pragma unroll
        for (int c = 0; c < kCopies; ++c) at_copy[c] = base(c);
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
          if (skips(i)) continue;
#pragma unroll
          for (int k = 0; k < kPacks; ++k) {
            int64_t at =
                int64_t{row_of(i)} * (input_stride_m / kPackElems) + pack_of(k);
            asm volatile("" : "+v"(at));
#pragma unroll
            for (int c = 0; c < kCopies; ++c) {
              const V* p = reinterpret_cast<const V*>(at_copy[c]) + at;
              V v;
              if constexpr (sizeof(V) == 16) {
                const u32x4 raw = *(const global_u32x4*)(p);
                __builtin_memcpy(&v, &raw, 16);
              } else {
                v = *p;
              }
              __builtin_memcpy(output[c][i][k], &v, sizeof(v));
            }
          }
        }
        asm volatile("" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
      } else {
        static_assert(kThreadsN % kInt4GroupPacks == 0,
                      "a group is 4 lanes' packs");

        const int64_t values_stride_m = input_size_n / 2;
        const int64_t scales_stride_m =
            TAIL ? int4_groups(input_size_n) : input_size_n / kInt4Group;
        const int64_t scales_at =
            int4_scales_offset(input_size_m, input_size_n);

        const uint8_t* values[kCopies];
        const char* scales[kCopies];
#pragma unroll
        for (int c = 0; c < kCopies; ++c) values[c] = base(c);
#pragma unroll
        for (int c = 0; c < kCopies; ++c)
          scales[c] = reinterpret_cast<const char*>(values[c] + scales_at);
        uint32_t words[kCopies][kRows][kPacks];
        uint32_t group_scales[kCopies][kRows][kPacks];
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
          if (skips(i)) continue;
#pragma unroll
          for (int k = 0; k < kPacks; ++k) {
            const int64_t row = row_of(i);
            const int pack = pack_of(k);
            uint32_t value_at = static_cast<uint32_t>(row * values_stride_m) +
                                static_cast<uint32_t>(pack) * kInt4WordBytes;
            uint32_t scale_at = static_cast<uint32_t>(
                (row * scales_stride_m + pack / kInt4GroupPacks) *
                kInt4ScaleBytes);
            asm volatile("" : "+v"(value_at));
            asm volatile("" : "+v"(scale_at));
#pragma unroll
            for (int c = 0; c < kCopies; ++c) {
              words[c][i][k] = *(const global_u32*)(values[c] + value_at);
              group_scales[c][i][k] = __builtin_bit_cast(
                  uint16_t, *(const global_f16*)(scales[c] + scale_at));
            }
          }
        }
        asm volatile("" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int c = 0; c < kCopies; ++c)
#pragma unroll
          for (int i = 0; i < kRows; ++i)
#pragma unroll
            for (int k = 0; k < kPacks; ++k) {
              auto& out = [&]() -> U(&)[kPackElems] {
                if constexpr (kSlots)
                  return output[c][i][k];
                else
                  return output[i][k];
              }();
              const uint32_t encoded[2] = {words[c][i][k],
                                           group_scales[c][i][k]};
              if constexpr (kRaw) {
                out[0] = encoded[0];
                out[1] = encoded[1];
              } else {
                dequantize(encoded, out);
              }
            }
      }
    }
  }
}

// An LL recv: each line polled until it carries its row's step.
template <int WORLD, int TILE_M, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE,
          bool QUANTIZE_INT4, Layout LAYOUT, bool TAIL, typename PTR,
          typename OUTPUT>
DINLINE void recv_ll(PTR const* input_ptr_array, int input_ptr_array_size,
                     int input_rank, int input_slot, int64_t input_stride_slot,
                     int64_t input_stride_m, int64_t input_stride_n,
                     int input_size_slot, int input_size_m, int input_size_n,
                     int clip_m, int clip_n, OUTPUT& output, int offs_m,
                     int offs_n, int row_step, Connection& connection,
                     uint32_t* status_ptr, int64_t status_stride_field,
                     int status_size_field, uint64_t timeout_ticks) {
  // A block of BLOCK_Y rows of threads, each the same tile: row y reads the
  // copies of writers input_slot + y * kRanks on, then the rows exchange them,
  // and every row ends with every writer's copy.
  static_assert(std::rank_v<OUTPUT> == 4 && std::extent_v<OUTPUT, 0> == WORLD &&
                    WORLD % BLOCK_Y == 0,
                "rows of threads read every writer's copy, whole rows each");
  constexpr int kRanks = WORLD / BLOCK_Y;
  using Got =
      std::remove_all_extents_t<OUTPUT>[kRanks][std::extent_v<OUTPUT, 1>]
                                       [std::extent_v<OUTPUT, 2>]
                                       [std::extent_v<OUTPUT, 3>];
  const int y =
      BLOCK_Y == 1
          ? 0
          : __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.y));
  input_slot += y * kRanks;
  Got rows;
  Got& into = BLOCK_Y == 1 ? reinterpret_cast<Got&>(output) : rows;
  // One peer for the wave: its memory's address is scalar.
  const auto* const entries =
      (decltype(+input_ptr_array[0]) const*)input_ptr_array;
  const int owner = __builtin_amdgcn_readfirstlane(input_rank);
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
      __builtin_amdgcn_ballot_w64(input_rank != owner) != 0)
    internal::check_failed();
  const void* const input =
      __builtin_constant_p(input_ptr_array_size) && input_ptr_array_size == 1
          ? entries[0]
          : ptr_array_at(entries, input_ptr_array_size, owner);
#if defined(__AMDGCN_WAVEFRONT_SIZE)
  static_assert(__AMDGCN_WAVEFRONT_SIZE == WAVE_SIZE,
                "the compiler's wave is not the one handed");
#endif
  constexpr bool kSlots = std::rank_v<Got> == 4;
  constexpr int kCopies = kSlots ? std::extent_v<Got, 0> : 1;
  constexpr int kRows = std::extent_v < Got, kSlots ? 1 : 0 > ;
  constexpr int kPacks = std::extent_v < Got, kSlots ? 2 : 1 > ;
  constexpr int kPackElems = std::extent_v < Got, kSlots ? 3 : 2 > ;
  using U = std::remove_all_extents_t<Got>;
  constexpr bool kRaw = std::is_same_v<U, uint32_t>;
  static_assert(!kRaw || (QUANTIZE_INT4 && kPackElems == 2),
                "uint32_t registers are an INT4 payload's words and scales");
  constexpr int kThreadsM = TILE_M / kRows;
  constexpr int kThreadsN = BLOCK_X / kThreadsM;
  static_assert(kThreadsM * kRows == TILE_M, "a tile is whole rows a thread");
  static_assert(kThreadsM * kThreadsN == BLOCK_X, "the layout is the block");
  using V = vec<U, kPackElems>;
  const int tid = static_cast<int>(threadIdx.x);
  const auto row_at = [&](int i) {
    return offs_m +
           (kThreadsM == 1 ? i : tid / kThreadsN + i * kThreadsM) * row_step;
  };
  constexpr int kElems = kRaw ? kInt4Group / kInt4GroupPacks : kPackElems;
  const auto pack_of = [&](int k) {
    const int n = offs_n / kElems + tid % kThreadsN + k * kThreadsN;
    return n < clip_n / kElems ? n : clip_n / kElems - 1;
  };
  // The tile's step (CONNECTION, types.cuh): the next after the one the block's
  // last tile left.
  static_assert(TILE_M == 1, "a connection's step is one inbox row's");
  uint32_t flags[kRows];
#pragma unroll
  for (int i = 0; i < kRows; ++i) {
    uint32_t last = static_cast<uint32_t>(connection[0][0][0]);
    // Left in a vector register, as send's: a readfirstlane here, after the
    // sends' lane-masked stores, would wait for those stores to complete.
    flags[i] = row_at(i) < clip_m ? (last == kLastStep ? 3u : last + 1u) : last;
  }
  {
    static_assert(LAYOUT == Layout::tile, "LL lines sit in a tile");
    // kCopies writers' copies, those of writers input_slot on.
    static_assert(kSlots && kCopies <= WORLD, "a line read is writers' copies");

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
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
        (input_slot < 0 || input_slot + kCopies > input_size_slot))
      internal::check_failed();

    const char* const first = reinterpret_cast<const char*>(input) +
                              int64_t{input_slot} * input_stride_slot *
                                  int64_t{sizeof(U)} * kLineScale;
    const uint32_t slot_bytes = static_cast<uint32_t>(
        input_stride_slot * int64_t{sizeof(U)} * kLineScale);

    const uint32_t half =
        static_cast<uint32_t>(input_stride_m * int64_t{sizeof(U)});
    const int bank_rows = input_size_m / kLlBanks;
    uint32_t line_at[kRows][kPacks];
    bool waits[kRows][kPacks];
#pragma unroll
    for (int i = 0; i < kRows; ++i)
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        const int row =
            row_at(i) + static_cast<int>(flags[i] % kLlBanks) * bank_rows;
        if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
            (row < 0 || row >= input_size_m || flags[i] == 0))
          internal::check_failed();
        line_at[i][k] =
            static_cast<uint32_t>(row) *
                static_cast<uint32_t>(input_stride_m * int64_t{sizeof(U)}) *
                kLineScale +
            static_cast<uint32_t>(pack_of(k)) * uint32_t{kLineBytes};
        waits[i][k] =
            row_at(i) < clip_m &&
            offs_n / kElems + tid % kThreadsN + k * kThreadsN < clip_n / kElems;
      }
    u32x4 got[kCopies][kRows][kPacks][kLines];
    const uint64_t t0 = wall_clock64();
    for (uint32_t n = 1;; ++n) {
#pragma unroll
      for (int c = 0; c < kCopies; ++c)
#pragma unroll
        for (int i = 0; i < kRows; ++i)
#pragma unroll
          for (int k = 0; k < kPacks; ++k)
#pragma unroll
            for (int h = 0; h < kLines; ++h) {
              uint32_t at = static_cast<uint32_t>(c) * slot_bytes +
                            line_at[i][k] + h * half;
              asm volatile("" : "+v"(at));
              got[c][i][k][h] =
                  __builtin_nontemporal_load((const global_u32x4*)(first + at));
            }

      asm volatile("" ::: "memory");
      bool missing = false;
#pragma unroll
      for (int c = 0; c < kCopies; ++c)
#pragma unroll
        for (int i = 0; i < kRows; ++i)
#pragma unroll
          for (int k = 0; k < kPacks; ++k)
#pragma unroll
            for (int h = 0; h < kLines; ++h)
              // & not &&: no branch, so each load is waited for here and not
              // again after the loop, behind the stores that follow it.
              missing |= waits[i][k] & (got[c][i][k][h][1] != flags[i] ||
                                        got[c][i][k][h][3] != flags[i]);

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
      if (!aborted) {
        uint32_t peer = 0, seen = flags[0], want = flags[0];
#pragma unroll
        for (int c = 0; c < kCopies; ++c)
#pragma unroll
          for (int i = 0; i < kRows; ++i)
#pragma unroll
            for (int k = 0; k < kPacks; ++k)
#pragma unroll
              for (int h = 0; h < kLines; ++h)
                if (waits[i][k] && (got[c][i][k][h][1] != flags[i] ||
                                    got[c][i][k][h][3] != flags[i])) {
                  peer = input_slot + c;
                  seen = got[c][i][k][h][1];
                  want = flags[i];
                }
        give_up(status_ptr, status_stride_field, status_size_field,
                Where::start, input_rank, peer, seen, want, waited);
      }
      __builtin_amdgcn_endpgm();
    }

#pragma unroll
    for (int c = 0; c < kCopies; ++c)
#pragma unroll
      for (int i = 0; i < kRows; ++i)
#pragma unroll
        for (int k = 0; k < kPacks; ++k) {
          if constexpr (QUANTIZE_INT4 && kRaw) {
            into[c][i][k][0] = got[c][i][k][0][0];
            into[c][i][k][1] = got[c][i][k][0][2];
          } else if constexpr (QUANTIZE_INT4) {
            uint32_t line[4];
            // Copy the payload out whole. Reading it as vector elements decoded
            // the wrong word.
            __builtin_memcpy(line, &got[c][i][k][0], sizeof(line));
            const uint32_t encoded[2] = {line[0], line[2]};
            dequantize(encoded, into[c][i][k]);
          } else {
            uint32_t words[2 * kLines];
#pragma unroll
            for (int h = 0; h < kLines; ++h) {
              words[2 * h] = got[c][i][k][h][0];
              words[2 * h + 1] = got[c][i][k][h][2];
            }
            __builtin_memcpy(into[c][i][k], words, sizeof(V));
          }
        }
  }
  if constexpr (BLOCK_Y > 1) {
    using U = std::remove_all_extents_t<OUTPUT>;
    __shared__ U exchanged[WORLD][BLOCK_X][std::extent_v<OUTPUT, 1>]
                          [std::extent_v<OUTPUT, 2>][std::extent_v<OUTPUT, 3>];
    const int t = static_cast<int>(threadIdx.x);
#pragma unroll
    for (int c = 0; c < kRanks; ++c)
      __builtin_memcpy(exchanged[y * kRanks + c][t], into[c], sizeof(into[c]));
    __syncthreads();
#pragma unroll
    for (int c = 0; c < WORLD; ++c)
      __builtin_memcpy(output[c], exchanged[c][t], sizeof(output[c]));
    // Read before any row writes the next call's.
    __syncthreads();
  }
  // The block's next tile starts from this one's step (CONNECTION).
  connection[0][0][0] = static_cast<int32_t>(flags[0]);
}

// A Lamport recv: each pack polled until no element is the sentinel, then
// re-armed.
template <int WORLD, int TILE_M, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE,
          bool QUANTIZE_INT4, Layout LAYOUT, bool TAIL, typename PTR,
          typename OUTPUT>
DINLINE void recv_lamport(PTR const* input_ptr_array, int input_ptr_array_size,
                          int input_rank, int input_slot,
                          int64_t input_stride_slot, int64_t input_stride_m,
                          int64_t input_stride_n, int input_size_slot,
                          int input_size_m, int input_size_n, int clip_m,
                          int clip_n, OUTPUT& output, int offs_m, int offs_n,
                          int row_step, Connection& connection,
                          uint32_t* status_ptr, int64_t status_stride_field,
                          int status_size_field, uint64_t timeout_ticks) {
  // A block of BLOCK_Y rows of threads, each the same tile: row y reads the
  // copies of writers input_slot + y * kRanks on, then the rows exchange them,
  // and every row ends with every writer's copy.
  static_assert(std::rank_v<OUTPUT> == 4 && std::extent_v<OUTPUT, 0> == WORLD &&
                    WORLD % BLOCK_Y == 0,
                "rows of threads read every writer's copy, whole rows each");
  constexpr int kRanks = WORLD / BLOCK_Y;
  using Got =
      std::remove_all_extents_t<OUTPUT>[kRanks][std::extent_v<OUTPUT, 1>]
                                       [std::extent_v<OUTPUT, 2>]
                                       [std::extent_v<OUTPUT, 3>];
  const int y =
      BLOCK_Y == 1
          ? 0
          : __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.y));
  input_slot += y * kRanks;
  Got rows;
  Got& into = BLOCK_Y == 1 ? reinterpret_cast<Got&>(output) : rows;
  // One peer for the wave: its memory's address is scalar.
  const auto* const entries =
      (decltype(+input_ptr_array[0]) const*)input_ptr_array;
  const int owner = __builtin_amdgcn_readfirstlane(input_rank);
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
      __builtin_amdgcn_ballot_w64(input_rank != owner) != 0)
    internal::check_failed();
  const void* const input =
      __builtin_constant_p(input_ptr_array_size) && input_ptr_array_size == 1
          ? entries[0]
          : ptr_array_at(entries, input_ptr_array_size, owner);
#if defined(__AMDGCN_WAVEFRONT_SIZE)
  static_assert(__AMDGCN_WAVEFRONT_SIZE == WAVE_SIZE,
                "the compiler's wave is not the one handed");
#endif
  constexpr bool kSlots = std::rank_v<Got> == 4;
  constexpr int kCopies = kSlots ? std::extent_v<Got, 0> : 1;
  constexpr int kRows = std::extent_v < Got, kSlots ? 1 : 0 > ;
  constexpr int kPacks = std::extent_v < Got, kSlots ? 2 : 1 > ;
  constexpr int kPackElems = std::extent_v < Got, kSlots ? 3 : 2 > ;
  using U = std::remove_all_extents_t<Got>;
  constexpr bool kRaw = std::is_same_v<U, uint32_t>;
  static_assert(!kRaw || (QUANTIZE_INT4 && kPackElems == 2),
                "uint32_t registers are an INT4 payload's words and scales");
  constexpr int kThreadsM = TILE_M / kRows;
  constexpr int kThreadsN = BLOCK_X / kThreadsM;
  static_assert(kThreadsM * kRows == TILE_M, "a tile is whole rows a thread");
  static_assert(kThreadsM * kThreadsN == BLOCK_X, "the layout is the block");
  using V = vec<U, kPackElems>;
  const int tid = static_cast<int>(threadIdx.x);
  const auto row_at = [&](int i) {
    return offs_m +
           (kThreadsM == 1 ? i : tid / kThreadsN + i * kThreadsM) * row_step;
  };
  constexpr int kElems = kRaw ? kInt4Group / kInt4GroupPacks : kPackElems;
  const auto pack_of = [&](int k) {
    const int n = offs_n / kElems + tid % kThreadsN + k * kThreadsN;
    return n < clip_n / kElems ? n : clip_n / kElems - 1;
  };
  // The tile's step (CONNECTION, types.cuh): the next after the one the block's
  // last tile left.
  static_assert(TILE_M == 1, "a connection's step is one inbox row's");
  uint32_t flags[kRows];
#pragma unroll
  for (int i = 0; i < kRows; ++i) {
    uint32_t last = static_cast<uint32_t>(connection[0][0][0]);
    // Left in a vector register, as send's: a readfirstlane here, after the
    // sends' lane-masked stores, would wait for those stores to complete.
    flags[i] = row_at(i) < clip_m ? (last == kLastStep ? 3u : last + 1u) : last;
  }
  {
    static_assert(LAYOUT == Layout::tile, "Lamport lines sit in a tile");
    static_assert(kSlots && kCopies <= WORLD,
                  "a Lamport read is writers' copies");
    static_assert(!QUANTIZE_INT4 && sizeof(U) == 2,
                  "the sentinel is a 2-byte NaN");
    using W = raw_t<sizeof(V)>;
    constexpr int kHalves = sizeof(V) / 2;
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
        (input_slot < 0 || input_slot + kCopies > input_size_slot))
      internal::check_failed();

    char* const first =
        const_cast<char*>(reinterpret_cast<const char*>(input)) +
        int64_t{input_slot} * input_stride_slot * int64_t{sizeof(U)};
    const uint32_t slot_bytes =
        static_cast<uint32_t>(input_stride_slot * int64_t{sizeof(U)});
    const uint32_t row_bytes =
        static_cast<uint32_t>(input_stride_m * int64_t{sizeof(U)});
    const int bank_rows = input_size_m / kLamportBanks;
    uint32_t at[kRows][kPacks];
    bool waits[kRows][kPacks];
#pragma unroll
    for (int i = 0; i < kRows; ++i)
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        const int row =
            row_at(i) + static_cast<int>(flags[i] % kLamportBanks) * bank_rows;
        if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
            (row < 0 || row >= input_size_m || flags[i] == 0))
          internal::check_failed();
        at[i][k] = static_cast<uint32_t>(row) * row_bytes +
                   static_cast<uint32_t>(pack_of(k)) * uint32_t{sizeof(V)};
        waits[i][k] =
            row_at(i) < clip_m &&
            offs_n / kElems + tid % kThreadsN + k * kThreadsN < clip_n / kElems;
      }
    const auto empty = [](const W& w) {
      uint16_t h[kHalves];
      __builtin_memcpy(h, &w, sizeof(h));
      bool any = false;
#pragma unroll
      for (int j = 0; j < kHalves; ++j) any |= h[j] == kLamportEmpty;
      return any;
    };
    W got[kCopies][kRows][kPacks];
    const uint64_t t0 = wall_clock64();
    for (uint32_t n = 1;; ++n) {
#pragma unroll
      for (int c = 0; c < kCopies; ++c)
#pragma unroll
        for (int i = 0; i < kRows; ++i)
#pragma unroll
          for (int k = 0; k < kPacks; ++k) {
            uint32_t a = static_cast<uint32_t>(c) * slot_bytes + at[i][k];
            asm volatile("" : "+v"(a));
            got[c][i][k] = __builtin_nontemporal_load(
                (const __attribute__((address_space(1))) W*)(first + a));
          }
      asm volatile("" ::: "memory");
      bool missing = false;
#pragma unroll
      for (int c = 0; c < kCopies; ++c)
#pragma unroll
        for (int i = 0; i < kRows; ++i)
#pragma unroll
          for (int k = 0; k < kPacks; ++k)
            // & not &&, as LL's.
            missing |= waits[i][k] & empty(got[c][i][k]);

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
      if (!aborted) {
        uint32_t peer = 0;
#pragma unroll
        for (int c = 0; c < kCopies; ++c)
#pragma unroll
          for (int i = 0; i < kRows; ++i)
#pragma unroll
            for (int k = 0; k < kPacks; ++k)
              if (waits[i][k] && empty(got[c][i][k])) peer = input_slot + c;
        give_up(status_ptr, status_stride_field, status_size_field,
                Where::start, input_rank, peer, kLamportEmpty, flags[0],
                waited);
      }
      __builtin_amdgcn_endpgm();
    }

#pragma unroll
    for (int c = 0; c < kCopies; ++c)
#pragma unroll
      for (int i = 0; i < kRows; ++i)
#pragma unroll
        for (int k = 0; k < kPacks; ++k)
          __builtin_memcpy(into[c][i][k], &got[c][i][k], sizeof(V));

    // Re-arm what was read. A peer writes this bank again only after this
    // rank's sends two tiles on (three banks, CONNECTION), and those come
    // after this wave's next recv has polled: a wave's loads and stores
    // complete in the order it issues them, so a poll that returns has these
    // stores behind it.
    // No wait here or before the sends.
    W cleared;
    __builtin_memset(&cleared, 0xff, sizeof(cleared));
#pragma unroll
    for (int c = 0; c < kCopies; ++c)
#pragma unroll
      for (int i = 0; i < kRows; ++i)
#pragma unroll
        for (int k = 0; k < kPacks; ++k)
          if (waits[i][k])
            __builtin_nontemporal_store(
                cleared,
                (__attribute__((address_space(1)))
                 W*)(first + static_cast<uint32_t>(c) * slot_bytes + at[i][k]));
  }
  if constexpr (BLOCK_Y > 1) {
    using U = std::remove_all_extents_t<OUTPUT>;
    __shared__ U exchanged[WORLD][BLOCK_X][std::extent_v<OUTPUT, 1>]
                          [std::extent_v<OUTPUT, 2>][std::extent_v<OUTPUT, 3>];
    const int t = static_cast<int>(threadIdx.x);
#pragma unroll
    for (int c = 0; c < kRanks; ++c)
      __builtin_memcpy(exchanged[y * kRanks + c][t], into[c], sizeof(into[c]));
    __syncthreads();
#pragma unroll
    for (int c = 0; c < WORLD; ++c)
      __builtin_memcpy(output[c], exchanged[c][t], sizeof(output[c]));
    // Read before any row writes the next call's.
    __syncthreads();
  }
  // The block's next tile starts from this one's step (CONNECTION).
  connection[0][0][0] = static_cast<int32_t>(flags[0]);
}

// A simple_write recv: each writer's round waited for (its tail), the copies
// read from this rank's staging, then each writer told (its head, COUNTERS).
template <int WORLD, int TILE_M, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE,
          bool QUANTIZE_INT4, Layout LAYOUT, bool TAIL, typename PTR,
          typename OUTPUT>
DINLINE void recv_simple_write(
    PTR const* input_ptr_array, int input_ptr_array_size, int input_rank,
    int input_slot, int64_t input_stride_slot, int64_t input_stride_m,
    int64_t input_stride_n, int input_size_slot, int input_size_m,
    int input_size_n, int clip_m, int clip_n, OUTPUT& output, int offs_m,
    int offs_n, int row_step, Connection& connection,
    uint32_t* const* counters_ptr_array, int counters_ptr_array_size,
    int64_t counters_stride_kind, int64_t counters_stride_block,
    int64_t counters_stride_peer, int counters_size_block,
    int counters_size_peer, uint32_t* status_ptr, int64_t status_stride_field,
    int status_size_field, uint64_t timeout_ticks) {
  if constexpr (LAYOUT != Layout::tile) {
    // A flat walk: a peer a lane group, this lane's input_slot its group's
    // writer; this rank is input_rank.
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
         input_slot < 0 || input_slot >= counters_size_peer))
      internal::check_failed();
    // Each lane waits for its group's writer's round (its tail, in this
    // rank's counters).
    uint32_t* const tail =
        ptr_array_at(counters_ptr_array, counters_ptr_array_size, input_rank) +
        static_cast<int64_t>(Counter::tail) * counters_stride_kind + at_block +
        int64_t{input_slot} * counters_stride_peer;
    const uint64_t t0 = wall_clock64();
    for (uint32_t n = 1;; ++n) {
      const uint32_t seen =
          __scoped_atomic_load_n(tail, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
      if (HIP_CUSTOM_COMMUNICATOR_DEBUG && seen != last && seen != next)
        internal::check_failed();
      if (__builtin_amdgcn_ballot_w64(seen != next) == 0) break;
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
                Where::start, input_rank, static_cast<uint32_t>(input_slot),
                seen, next, waited);
      __builtin_amdgcn_endpgm();
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    // The data from this rank's inbox, at the writer's slot.
    recv_simple<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE, QUANTIZE_INT4,
                LAYOUT, TAIL>(input_ptr_array, input_ptr_array_size, input_rank,
                              input_slot, input_stride_slot, input_stride_m,
                              input_stride_n, input_size_slot, input_size_m,
                              input_size_n, clip_m, clip_n, output, offs_m,
                              offs_n, row_step);
    // The block's reads, then each group's writer told (its head from this
    // rank), by the group's first lane.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    if constexpr (BLOCK_X > WAVE_SIZE) __syncthreads();
    if (static_cast<int>(threadIdx.x) % kLanes == 0)
      __scoped_atomic_store_n(
          lane_ptr_array_at<uint32_t, WORLD>(
              counters_ptr_array, counters_ptr_array_size, input_slot) +
              static_cast<int64_t>(Counter::head) * counters_stride_kind +
              at_block + int64_t{input_rank} * counters_stride_peer,
          next, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    connection[0][0][0] = static_cast<int32_t>(next);
  } else {
    static_assert(
        LAYOUT == Layout::tile && !QUANTIZE_INT4 && TILE_M == 1 && BLOCK_Y == 1,
        "a simple_write recv is an exact tile of one row, into one "
        "row of threads");
    static_assert(BLOCK_X % WAVE_SIZE == 0, "a block is whole waves");
    static_assert(std::rank_v<OUTPUT> == 4, "a recv here is writers' copies");
    const auto* const entries =
        (decltype(+input_ptr_array[0]) const*)input_ptr_array;
    const int owner = __builtin_amdgcn_readfirstlane(input_rank);
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
        __builtin_amdgcn_ballot_w64(input_rank != owner) != 0)
      internal::check_failed();
    const void* const input =
        ptr_array_at(entries, input_ptr_array_size, owner);
#if defined(__AMDGCN_WAVEFRONT_SIZE)
    static_assert(__AMDGCN_WAVEFRONT_SIZE == WAVE_SIZE,
                  "the compiler's wave is not the one handed");
#endif
    constexpr int kCopies = std::extent_v<OUTPUT, 0>;
    constexpr int kRows = std::extent_v<OUTPUT, 1>;
    constexpr int kPacks = std::extent_v<OUTPUT, 2>;
    constexpr int kPackElems = std::extent_v<OUTPUT, 3>;
    using U = std::remove_all_extents_t<OUTPUT>;
    static_assert(kCopies <= WORLD, "a read is writers' copies");
    constexpr int kThreadsM = TILE_M / kRows;
    constexpr int kThreadsN = BLOCK_X / kThreadsM;
    static_assert(kThreadsM * kRows == TILE_M, "a tile is whole rows a thread");
    static_assert(kThreadsM * kThreadsN == BLOCK_X, "the layout is the block");
    using V = vec<U, kPackElems>;
    using W = raw_t<sizeof(V)>;
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
        (input_slot < 0 || input_slot + kCopies > input_size_slot))
      internal::check_failed();
    const int tid = static_cast<int>(threadIdx.x);
    const auto row_at = [&](int i) {
      return offs_m +
             (kThreadsM == 1 ? i : tid / kThreadsN + i * kThreadsM) * row_step;
    };
    const auto pack_of = [&](int k) {
      const int n = offs_n / kPackElems + tid % kThreadsN + k * kThreadsN;
      return n < clip_n / kPackElems ? n : clip_n / kPackElems - 1;
    };
    // The round: the next after the block's last (CONNECTION).
    const uint32_t last = static_cast<uint32_t>(connection[0][0][0]);
    const uint32_t next = last == kLastStep ? 3u : last + 1u;
    const auto offset = [&](Counter kind, int peer) {
      if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
          (static_cast<int>(blockIdx.x) >= counters_size_block || peer < 0 ||
           peer >= counters_size_peer))
        internal::check_failed();
      return static_cast<int64_t>(kind) * counters_stride_kind +
             int64_t{blockIdx.x} * counters_stride_block +
             int64_t{peer} * counters_stride_peer;
    };
    // Each wave waits for every writer's round (their tails, in this rank's
    // counters).
    uint32_t* const mine =
        ptr_array_at(counters_ptr_array, counters_ptr_array_size, owner);
    const uint64_t t0 = wall_clock64();
    for (uint32_t n = 1;; ++n) {
      bool missing = false;
      uint32_t seen = next;
#pragma unroll
      for (int c = 0; c < kCopies; ++c) {
        const uint32_t v =
            __scoped_atomic_load_n(mine + offset(Counter::tail, input_slot + c),
                                   __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
        if (HIP_CUSTOM_COMMUNICATOR_DEBUG && v != last && v != next)
          internal::check_failed();
        if (v != next) missing = true, seen = v;
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
                Where::start, owner, kCopies, seen, next, waited);
      __builtin_amdgcn_endpgm();
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    const char* const first =
        reinterpret_cast<const char*>(input) +
        int64_t{input_slot} * input_stride_slot * int64_t{sizeof(U)};
    const uint32_t slot_bytes =
        static_cast<uint32_t>(input_stride_slot * int64_t{sizeof(U)});
    const uint32_t row_bytes =
        static_cast<uint32_t>(input_stride_m * int64_t{sizeof(U)});
#pragma unroll
    for (int c = 0; c < kCopies; ++c)
#pragma unroll
      for (int i = 0; i < kRows; ++i) {
        const int row = row_at(i) < clip_m ? row_at(i) : clip_m - 1;
        if (HIP_CUSTOM_COMMUNICATOR_DEBUG && row >= input_size_m)
          internal::check_failed();
#pragma unroll
        for (int k = 0; k < kPacks; ++k) {
          uint32_t at = static_cast<uint32_t>(c) * slot_bytes +
                        static_cast<uint32_t>(row) * row_bytes +
                        static_cast<uint32_t>(pack_of(k)) * uint32_t{sizeof(V)};
          asm volatile("" : "+v"(at));
          const W w = *(const __attribute__((address_space(1))) W*)(first + at);
          __builtin_memcpy(output[c][i][k], &w, sizeof(V));
        }
      }
    // The block's reads, then each writer told it may write the next round
    // (its head from this rank, in its counters).
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    if constexpr (BLOCK_X > WAVE_SIZE) __syncthreads();
    if (threadIdx.x < kCopies) {
      const int writer = input_slot + static_cast<int>(threadIdx.x);
      uint32_t* const theirs = lane_ptr_array_at<uint32_t, WORLD>(
          counters_ptr_array, counters_ptr_array_size, writer);
      __scoped_atomic_store_n(theirs + offset(Counter::head, owner), next,
                              __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    }
    // The block's next tile starts from this one's step (CONNECTION).
    connection[0][0][0] = static_cast<int32_t>(next);
  }
}

// A simple_read recv: each writer's round waited for (its tail), each copy
// loaded from its writer's own inbox at this rank's slot, then each writer
// told (its head, COUNTERS).
template <int WORLD, int TILE_M, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE,
          bool QUANTIZE_INT4, Layout LAYOUT, bool TAIL, typename PTR,
          typename OUTPUT>
DINLINE void recv_simple_read(
    PTR const* input_ptr_array, int input_ptr_array_size, int input_rank,
    int input_slot, int64_t input_stride_slot, int64_t input_stride_m,
    int64_t input_stride_n, int input_size_slot, int input_size_m,
    int input_size_n, int clip_m, int clip_n, OUTPUT& output, int offs_m,
    int offs_n, int row_step, Connection& connection,
    uint32_t* const* counters_ptr_array, int counters_ptr_array_size,
    int64_t counters_stride_kind, int64_t counters_stride_block,
    int64_t counters_stride_peer, int counters_size_block,
    int counters_size_peer, uint32_t* status_ptr, int64_t status_stride_field,
    int status_size_field, uint64_t timeout_ticks) {
  if constexpr (LAYOUT != Layout::tile) {
    // A flat walk: a peer a lane group, this lane's input_slot its group's
    // writer; this rank is input_rank.
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
         input_slot < 0 || input_slot >= counters_size_peer))
      internal::check_failed();
    // Each lane waits for its group's writer's round (its tail, in this
    // rank's counters).
    uint32_t* const tail =
        ptr_array_at(counters_ptr_array, counters_ptr_array_size, input_rank) +
        static_cast<int64_t>(Counter::tail) * counters_stride_kind + at_block +
        int64_t{input_slot} * counters_stride_peer;
    const uint64_t t0 = wall_clock64();
    for (uint32_t n = 1;; ++n) {
      const uint32_t seen =
          __scoped_atomic_load_n(tail, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
      if (HIP_CUSTOM_COMMUNICATOR_DEBUG && seen != last && seen != next)
        internal::check_failed();
      if (__builtin_amdgcn_ballot_w64(seen != next) == 0) break;
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
                Where::start, input_rank, static_cast<uint32_t>(input_slot),
                seen, next, waited);
      __builtin_amdgcn_endpgm();
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    // The data from the writer's own inbox, at this rank's slot.
    recv_simple<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE, QUANTIZE_INT4,
                LAYOUT, TAIL>(input_ptr_array, input_ptr_array_size, input_slot,
                              input_rank, input_stride_slot, input_stride_m,
                              input_stride_n, input_size_slot, input_size_m,
                              input_size_n, clip_m, clip_n, output, offs_m,
                              offs_n, row_step);
    // The block's reads, then each group's writer told (its head from this
    // rank), by the group's first lane.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    if constexpr (BLOCK_X > WAVE_SIZE) __syncthreads();
    if (static_cast<int>(threadIdx.x) % kLanes == 0)
      __scoped_atomic_store_n(
          lane_ptr_array_at<uint32_t, WORLD>(
              counters_ptr_array, counters_ptr_array_size, input_slot) +
              static_cast<int64_t>(Counter::head) * counters_stride_kind +
              at_block + int64_t{input_rank} * counters_stride_peer,
          next, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    connection[0][0][0] = static_cast<int32_t>(next);
  } else {
    static_assert(
        LAYOUT == Layout::tile && !QUANTIZE_INT4 && TILE_M == 1 && BLOCK_Y == 1,
        "a simple_read recv is an exact tile of one row, into one "
        "row of threads");
    static_assert(BLOCK_X % WAVE_SIZE == 0, "a block is whole waves");
    static_assert(std::rank_v<OUTPUT> == 4, "a recv here is writers' copies");
    const auto* const entries =
        (decltype(+input_ptr_array[0]) const*)input_ptr_array;
    const int owner = __builtin_amdgcn_readfirstlane(input_rank);
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
        __builtin_amdgcn_ballot_w64(input_rank != owner) != 0)
      internal::check_failed();
#if defined(__AMDGCN_WAVEFRONT_SIZE)
    static_assert(__AMDGCN_WAVEFRONT_SIZE == WAVE_SIZE,
                  "the compiler's wave is not the one handed");
#endif
    constexpr int kCopies = std::extent_v<OUTPUT, 0>;
    constexpr int kRows = std::extent_v<OUTPUT, 1>;
    constexpr int kPacks = std::extent_v<OUTPUT, 2>;
    constexpr int kPackElems = std::extent_v<OUTPUT, 3>;
    using U = std::remove_all_extents_t<OUTPUT>;
    static_assert(kCopies <= WORLD, "a read is writers' copies");
    constexpr int kThreadsM = TILE_M / kRows;
    constexpr int kThreadsN = BLOCK_X / kThreadsM;
    static_assert(kThreadsM * kRows == TILE_M, "a tile is whole rows a thread");
    static_assert(kThreadsM * kThreadsN == BLOCK_X, "the layout is the block");
    using V = vec<U, kPackElems>;
    using W = raw_t<sizeof(V)>;
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
        (input_slot < 0 || input_slot + kCopies > input_size_slot))
      internal::check_failed();
    const int tid = static_cast<int>(threadIdx.x);
    const auto row_at = [&](int i) {
      return offs_m +
             (kThreadsM == 1 ? i : tid / kThreadsN + i * kThreadsM) * row_step;
    };
    const auto pack_of = [&](int k) {
      const int n = offs_n / kPackElems + tid % kThreadsN + k * kThreadsN;
      return n < clip_n / kPackElems ? n : clip_n / kPackElems - 1;
    };
    // The round: the next after the block's last (CONNECTION).
    const uint32_t last = static_cast<uint32_t>(connection[0][0][0]);
    const uint32_t next = last == kLastStep ? 3u : last + 1u;
    const auto offset = [&](Counter kind, int peer) {
      if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
          (static_cast<int>(blockIdx.x) >= counters_size_block || peer < 0 ||
           peer >= counters_size_peer))
        internal::check_failed();
      return static_cast<int64_t>(kind) * counters_stride_kind +
             int64_t{blockIdx.x} * counters_stride_block +
             int64_t{peer} * counters_stride_peer;
    };
    // Each wave waits for every writer's round (their tails, in this rank's
    // counters).
    uint32_t* const mine =
        ptr_array_at(counters_ptr_array, counters_ptr_array_size, owner);
    const uint64_t t0 = wall_clock64();
    for (uint32_t n = 1;; ++n) {
      bool missing = false;
      uint32_t seen = next;
#pragma unroll
      for (int c = 0; c < kCopies; ++c) {
        const uint32_t v =
            __scoped_atomic_load_n(mine + offset(Counter::tail, input_slot + c),
                                   __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
        if (HIP_CUSTOM_COMMUNICATOR_DEBUG && v != last && v != next)
          internal::check_failed();
        if (v != next) missing = true, seen = v;
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
                Where::start, owner, kCopies, seen, next, waited);
      __builtin_amdgcn_endpgm();
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    const uint32_t slot_bytes =
        static_cast<uint32_t>(input_stride_slot * int64_t{sizeof(U)});
    const uint32_t row_bytes =
        static_cast<uint32_t>(input_stride_m * int64_t{sizeof(U)});
#pragma unroll
    for (int c = 0; c < kCopies; ++c) {
      // Writer input_slot + c's copy, in its own inbox, at this rank's slot.
      const char* const first =
          reinterpret_cast<const char*>(
              ptr_array_at(entries, input_ptr_array_size, input_slot + c)) +
          static_cast<int64_t>(owner) * slot_bytes;
#pragma unroll
      for (int i = 0; i < kRows; ++i) {
        const int row = row_at(i) < clip_m ? row_at(i) : clip_m - 1;
        if (HIP_CUSTOM_COMMUNICATOR_DEBUG && row >= input_size_m)
          internal::check_failed();
#pragma unroll
        for (int k = 0; k < kPacks; ++k) {
          uint32_t at = static_cast<uint32_t>(row) * row_bytes +
                        static_cast<uint32_t>(pack_of(k)) * uint32_t{sizeof(V)};
          asm volatile("" : "+v"(at));
          const W w = *(const __attribute__((address_space(1))) W*)(first + at);
          __builtin_memcpy(output[c][i][k], &w, sizeof(V));
        }
      }
    }
    // The block's reads, then each writer told it may write the next round
    // (its head from this rank, in its counters).
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    if constexpr (BLOCK_X > WAVE_SIZE) __syncthreads();
    if (threadIdx.x < kCopies) {
      const int writer = input_slot + static_cast<int>(threadIdx.x);
      uint32_t* const theirs = lane_ptr_array_at<uint32_t, WORLD>(
          counters_ptr_array, counters_ptr_array_size, writer);
      __scoped_atomic_store_n(theirs + offset(Counter::head, owner), next,
                              __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    }
    // The block's next tile starts from this one's step (CONNECTION).
    connection[0][0][0] = static_cast<int32_t>(next);
  }
}

}  // namespace internal

template <int WORLD, int TILE_M, int BLOCK_X, int BLOCK_Y, int WAVE_SIZE,
          bool QUANTIZE_INT4, Protocol PROTOCOL, Layout LAYOUT,
          bool TAIL = false, typename OUTPUT>
DINLINE void recv(const ProtocolBuffers& protocol_buffers, int input_rank,
                  int rows, int cols, OUTPUT& output) {
  // The writers from input_rank on, read as this rank (`self`; a leaf takes
  // the reader, then the first writer), at the block's own rows: a tile's
  // TILE_M from the block's first, a flat walk's one. Of them, `rows` exist,
  // of `cols` columns.
  const int self = protocol_buffers.rank;
  const int offs_m =
      static_cast<int>(blockIdx.x) * (LAYOUT == Layout::tile ? TILE_M : 1);
  const int clip_m = offs_m + rows, clip_n = cols, offs_n = 0, row_step = 1;
  // The protocol's values, as the leaves take them.
  char* const* const input_ptr_array = protocol_buffers.inbox_ptr_array;
  const int input_ptr_array_size = protocol_buffers.inbox_ptr_array_size;
  const int64_t input_stride_slot = protocol_buffers.inbox_stride_slot,
                input_stride_m = protocol_buffers.inbox_stride_m,
                input_stride_n = protocol_buffers.inbox_stride_n;
  // An INT4 group walk's rows are every slot's: its scales follow them all.
  constexpr bool kGroups =
      LAYOUT == Layout::group_run || LAYOUT == Layout::group_flat;
  const int input_size_slot = protocol_buffers.inbox_size_slot,
            input_size_m = kGroups ? protocol_buffers.inbox_size_slot *
                                         protocol_buffers.inbox_size_m
                                   : protocol_buffers.inbox_size_m,
            input_size_n = protocol_buffers.inbox_size_n;
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
  if constexpr (PROTOCOL == Protocol::ll) {
    internal::recv_ll<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE, QUANTIZE_INT4,
                      LAYOUT, TAIL>(
        input_ptr_array, input_ptr_array_size, self, input_rank,
        input_stride_slot, input_stride_m, input_stride_n, input_size_slot,
        input_size_m, input_size_n, clip_m, clip_n, output, offs_m, offs_n,
        row_step, *connection, status_ptr, status_stride_field,
        status_size_field, timeout_ticks);
  } else if constexpr (PROTOCOL == Protocol::simple_write) {
    internal::recv_simple_write<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE,
                                QUANTIZE_INT4, LAYOUT, TAIL>(
        input_ptr_array, input_ptr_array_size, self, input_rank,
        input_stride_slot, input_stride_m, input_stride_n, input_size_slot,
        input_size_m, input_size_n, clip_m, clip_n, output, offs_m, offs_n,
        row_step, *connection, counters_ptr_array, counters_ptr_array_size,
        counters_stride_kind, counters_stride_block, counters_stride_peer,
        counters_size_block, counters_size_peer, status_ptr,
        status_stride_field, status_size_field, timeout_ticks);
  } else if constexpr (PROTOCOL == Protocol::simple_read) {
    internal::recv_simple_read<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE,
                               QUANTIZE_INT4, LAYOUT, TAIL>(
        input_ptr_array, input_ptr_array_size, self, input_rank,
        input_stride_slot, input_stride_m, input_stride_n, input_size_slot,
        input_size_m, input_size_n, clip_m, clip_n, output, offs_m, offs_n,
        row_step, *connection, counters_ptr_array, counters_ptr_array_size,
        counters_stride_kind, counters_stride_block, counters_stride_peer,
        counters_size_block, counters_size_peer, status_ptr,
        status_stride_field, status_size_field, timeout_ticks);
  } else {
    internal::recv_lamport<WORLD, TILE_M, BLOCK_X, BLOCK_Y, WAVE_SIZE,
                           QUANTIZE_INT4, LAYOUT, TAIL>(
        input_ptr_array, input_ptr_array_size, self, input_rank,
        input_stride_slot, input_stride_m, input_stride_n, input_size_slot,
        input_size_m, input_size_n, clip_m, clip_n, output, offs_m, offs_n,
        row_step, *connection, status_ptr, status_stride_field,
        status_size_field, timeout_ticks);
  }
  // This wave's slot, and wave 0 the slots of the waves the block does not
  // have (a lane each: no wave reads them in this launch).
  const uint32_t advanced = static_cast<uint32_t>(step[0][0][0]);
  __scoped_atomic_store_n(step_at, advanced, __ATOMIC_RELAXED,
                          __MEMORY_SCOPE_SYSTEM);
  if constexpr (kWaves < kStepWaves) {
    const int lane = static_cast<int>(threadIdx.x) % WAVE_SIZE;
    if (wave == 0 && lane >= kWaves && lane < kStepWaves)
      __scoped_atomic_store_n(
          block_steps + int64_t{lane} * protocol_buffers.steps_stride_wave,
          advanced, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

}  // namespace hip_custom_communicator
