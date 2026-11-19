#pragma once

// load: this rank's own memory into registers, in a layout.

#include "dequantize.cuh"
#include "types.cuh"

namespace hip_custom_communicator {

template <int WORLD, int TILE_M, int BLOCK_X, int WAVE_SIZE, bool QUANTIZE_INT4,
          Layout LAYOUT, bool TAIL = false, typename T, typename OUTPUT>
DINLINE void load(const T* input, int input_slot, int64_t input_stride_slot,
                  int64_t input_stride_m, int64_t input_stride_n,
                  int input_size_slot, int input_size_m, int input_size_n,
                  int clip_m, int clip_n, OUTPUT& output, int offs_m,
                  int offs_n, int row_step) {
  static_assert(!std::is_pointer_v<T>,
                "a pointer array is other ranks' memory: send or recv it");
  constexpr bool kSlots = std::rank_v<OUTPUT> == 4;
  constexpr int kPacks = std::extent_v < OUTPUT, kSlots ? 2 : 1 > ;
  if constexpr (LAYOUT == Layout::block) {
    static_assert(!QUANTIZE_INT4 && !kSlots && std::extent_v<OUTPUT, 0> == 1 &&
                      kPacks == 1,
                  "a block's line is one row of one pack");
    constexpr int kElems = std::extent_v<OUTPUT, 2>;
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
        (input_slot < 0 || input_slot >= input_size_slot || offs_n < 0 ||
         offs_n + kElems > clip_n || clip_n > input_size_n || clip_m < 1 ||
         clip_m > input_size_m))
      internal::check_failed();
    // A row past the clip reads the last, as the tile walk's do.
    const int row = offs_m < clip_m ? offs_m : clip_m - 1;
    const T* const at = input + int64_t{input_slot} * input_stride_slot +
                        int64_t{row} * input_stride_m +
                        int64_t{offs_n} * input_stride_n;
    // Vector loads, though the address is the block's: they wait with the
    // tile's other vector loads, where scalar loads would wait in line
    // behind the kernel's argument loads.
#pragma unroll
    for (int j = 0; j < kElems; ++j) {
      int64_t e = int64_t{j} * input_stride_n;
      asm volatile("" : "+v"(e));
      output[0][0][j] = at[e];
    }
  } else if constexpr ((LAYOUT == Layout::group_run ||
                        LAYOUT == Layout::group_flat) &&
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
      load<WORLD, TILE_M, BLOCK_X, WAVE_SIZE, QUANTIZE_INT4, LAYOUT, TAIL>(
          input, input_slot, input_stride_slot, input_stride_m, input_stride_n,
          input_size_slot, input_size_m, input_size_n, clip_m, clip_n, group,
          offs_m, offs_n + jg * (BLOCK_X / TILE_M) * kInt4Group, row_step);
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
    static_assert(!kRaw || (QUANTIZE_INT4 && kPackElems == 2),
                  "uint32_t registers are an INT4 payload's words and scales");
    constexpr int kThreadsM = TILE_M / kRows;
    constexpr int kThreadsN = BLOCK_X / kThreadsM;
    static_assert(kThreadsM * kRows == TILE_M, "a tile is whole rows a thread");
    static_assert(kThreadsM * kThreadsN == BLOCK_X, "the layout is the block");
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
      return offs_m +
             (kThreadsM == 1 ? i : tid / kThreadsN + i * kThreadsM) * row_step;
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
    [[maybe_unused]] const auto element_at =
        [](auto* ptr, int64_t stride_m, int64_t stride_n, int64_t e, int n) {
          if (stride_m == int64_t{n} * stride_n) return ptr + e * stride_n;
          const int64_t row =
              e < (int64_t{1} << 31)
                  ? int64_t{static_cast<uint32_t>(e) / static_cast<uint32_t>(n)}
                  : e / n;
          return ptr + row * stride_m + (e - row * n) * stride_n;
        };
    constexpr bool kGroup =
        LAYOUT == Layout::group_run || LAYOUT == Layout::group_flat;
    constexpr bool kRun = LAYOUT == Layout::run || LAYOUT == Layout::group_run;
    if constexpr (LAYOUT != Layout::tile && !kGroup) {
      static_assert(!QUANTIZE_INT4 && !kSlots && kRows == 1,
                    "the flat walk reads one exact copy a row of threads");

      constexpr int kLanes = BLOCK_X / WORLD;
      static_assert(kLanes * WORLD == BLOCK_X,
                    "a block is a group of threads a rank");
      const U* const from =
          reinterpret_cast<const U*>(input) + int64_t{offs_m} * input_stride_m;
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
          const int col = n < clip_n / kPackElems ? n : clip_n / kPackElems - 1;
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
                    *(const global_u32x4*)(reinterpret_cast<const char*>(group +
                                                                         k) +
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
              input_stride_slot;
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
        const V* row =
            kLaneRow ? at
                     : at + int64_t{row_of(i)} * (input_stride_m / kPackElems);
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
      const int64_t scales_at = int4_scales_offset(input_size_m, input_size_n);

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
            const uint32_t encoded[2] = {words[c][i][k], group_scales[c][i][k]};
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

}  // namespace hip_custom_communicator
