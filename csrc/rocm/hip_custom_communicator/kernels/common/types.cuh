#pragma once

// What the primitives (common/*.cuh) and their callers share: register and
// memory types, the INT4 and line formats, the status record, the layouts and
// protocols, and the internals more than one primitive calls.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace hip_custom_communicator {

#define DINLINE __device__ __forceinline__

template <typename DTYPE, int VEC_SIZE>
struct __align__(sizeof(DTYPE) * VEC_SIZE) vec {
  DTYPE d[VEC_SIZE];
};

template <typename DTYPE, int PACK_BYTES>
struct traits {
  static constexpr int N = PACK_BYTES / sizeof(DTYPE);
};

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(1))) u32x4 global_u32x4;
typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));

// A pack's raw bytes, moved in one access.
template <int BYTES>
using raw_t = std::conditional_t<
    BYTES == 16, u32x4,
    std::conditional_t<BYTES == 8, u32x2,
                       std::conditional_t<BYTES == 4, uint32_t, uint16_t>>>;

// Lamport's sentinel: an inbox position holds it until its data arrives. It is
// a NaN in bf16 and fp16, and a sent 0xffff becomes 0x7fff, also a NaN, so
// data never carries it.
constexpr uint16_t kLamportEmpty = 0xffff;

// CONNECTION (NCCL's word for what a send and a recv coordinate through): a
// block's few words of state for the peer buffer it talks through, a register
// array. The stage loads it before its first tile and stores it after its
// last; send and recv read and advance it, and each protocol decides what the
// words are. Every wave keeps its own copy and every copy advances the same
// way, and the next read is the next launch's, so no wave waits for another
// to save it. Each peer buffer has its own connections (the LL inbox's, the
// Lamport inbox's, Simple's staging), a block's at its block index.
// LL and Lamport: word 0 is the step of the block's inbox row (a tile is one
// row). A tile's step is the next after the last (0 is none): LL writes it in
// every line and waits for it, Lamport takes its bank from it. LL alternates
// two banks; Lamport three, so a bank it re-armed is written again only after
// this rank's sends two tiles later, and the next tile's recv polls before
// those: a poll that returns has the wave's earlier stores, the re-arm, behind
// it (recv_lamport), so nothing waits for the re-arm.
// The wrap skips 0 and keeps consecutive steps in different banks of two and of
// three (0xfffffffe, then 3).
constexpr int kConnectionWords = 4;
using Connection = int32_t[1][1][kConnectionWords];
constexpr int kLlBanks = 2;
constexpr int kLamportBanks = 3;
constexpr uint32_t kLastStep = 0xfffffffeu;

typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));

#ifndef HIP_CUSTOM_COMMUNICATOR_DEBUG
  #define HIP_CUSTOM_COMMUNICATOR_DEBUG 0
#endif

constexpr int kInt4Group = 32;
constexpr int kInt4Lowest = -8;
constexpr int kInt4Highest = 7;
constexpr int kInt4WordBytes = 4;
constexpr int kInt4ScaleBytes = 2;
constexpr uint32_t kInt4ZeroWord = 0x88888888u;

constexpr int kInt4GroupPacks = kInt4Group / (2 * kInt4WordBytes);

__host__ __device__ constexpr int64_t int4_groups(int64_t n) {
  return (n + kInt4Group - 1) / kInt4Group;
}

constexpr int64_t kInt4ScalesAlign = 256;
__host__ __device__ constexpr int64_t int4_scales_offset(int64_t m, int64_t n) {
  return (m * n / 2 + kInt4ScalesAlign - 1) / kInt4ScalesAlign *
         kInt4ScalesAlign;
}

__host__ __device__ constexpr int64_t int4_payload_bytes(int64_t m, int64_t n) {
  return int4_scales_offset(m, n) + m * int4_groups(n) * kInt4ScaleBytes;
}

constexpr int kLineBytes = 16;
constexpr int kLinePayloadBytes = kLineBytes / 2;

enum class Where : uint32_t {
  start = 1,
  peer_barrier,
};

enum class FlagState : uint32_t {
  clear = 0,
  arrived = 1,
};

enum class FaultState : uint32_t {
  none = 0,
  writing = 1,
  written = 2,
};

enum class StatusField : int {
  abort,
  state,
  where,
  rank,
  block,
  peer,
  count,
  want,
  elapsed_low,

  elapsed_high,
};

constexpr int kStatusFields = static_cast<int>(StatusField::elapsed_high) + 1;

typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));

template <typename U>
constexpr bool kBf16 = sizeof(U) == 2 && std::numeric_limits<U>::digits == 8;

typedef __attribute__((address_space(1))) uint32_t global_u32;
typedef __attribute__((address_space(1))) __fp16 global_f16;

// block: the block's one line (a row of one pack), every thread the same copy,
// at an address uniform over the block, so it is read with scalar loads, which
// wait behind no vector store (a block's connection, CONNECTION).
enum class Layout { tile, run, flat, group_run, group_flat, block };

// How a receiver knows a peer's data has arrived. Simple: the data, then a
// barrier. LL: a flag in every line, half the line payload.
enum class Protocol { simple, ll, lamport };

namespace internal {

// A checked build's failed check: it says where, from the wave's first active
// lane, then ends the wave. A trap halts the GPU on some machines (a hang that
// says nothing); this way the launch finishes, a peer waiting on this rank
// gives up the same way (give_up), and the results the wave did not write
// fail the caller's test.
[[noreturn]] inline __device__ __attribute__((noinline)) void check_failed(
    const char* file = __builtin_FILE(), int line = __builtin_LINE()) {
  const uint64_t active = __builtin_amdgcn_read_exec();
  const uint32_t lane =
      __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
  if (lane == static_cast<uint32_t>(__builtin_ctzll(active)))
    printf(
        "hip_custom_communicator check failed: %s:%d block %u thread (%u, "
        "%u)\n",
        file, line, blockIdx.x, threadIdx.x, threadIdx.y);
  __builtin_amdgcn_endpgm();
}

template <typename U = void, typename T>
DINLINE std::conditional_t<std::is_void_v<U>, T, U>* ptr_array_at(
    T* const* x_ptr_array, int x_ptr_array_size, int r) {
  r = __builtin_amdgcn_readfirstlane(r);
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG && (r < 0 || r >= x_ptr_array_size))
    internal::check_failed();
  typedef T* Entry;
  return reinterpret_cast<std::conditional_t<std::is_void_v<U>, T, U>*>(
      ((const __attribute__((address_space(4))) Entry*)x_ptr_array)[r]);
}

template <typename U, int WORLD, typename T>
DINLINE std::conditional_t<std::is_void_v<U>, T, U>* lane_ptr_array_at(
    T* const* x_ptr_array, int x_ptr_array_size, int r) {
  static_assert(WORLD >= 1, "a world has a rank");
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
      (WORLD - 1 < 0 || WORLD - 1 >= x_ptr_array_size))
    internal::check_failed();
  const auto scalar = [](T* p) {
    const auto u = reinterpret_cast<uint64_t>(p);
    const uint32_t lo =
        __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(u));
    const uint32_t hi =
        __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(u >> 32));
    return (uint64_t{hi} << 32) | lo;
  };
  uint64_t at = scalar(ptr_array_at(x_ptr_array, x_ptr_array_size, 0));
#pragma unroll
  for (int k = 1; k < WORLD; ++k) {
    const uint64_t p = scalar(ptr_array_at(x_ptr_array, x_ptr_array_size, k));
    at = r == k ? p : at;
  }
  auto* const global =
      reinterpret_cast<__attribute__((address_space(1))) char*>(at);
  return reinterpret_cast<std::conditional_t<std::is_void_v<U>, T, U>*>(
      (char*)global);
}

DINLINE void give_up(uint32_t* status_ptr, int64_t status_stride_field,
                     int status_size_field, Where where, int rank,
                     uint32_t peer, uint32_t count, uint32_t want,
                     uint64_t waited) {
  const auto word = [&](StatusField f) {
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
        (static_cast<int64_t>(f) < 0 ||
         static_cast<int64_t>(f) >= status_size_field))
      internal::check_failed();
    return status_ptr + static_cast<int64_t>(f) * status_stride_field;
  };
  uint32_t none = static_cast<uint32_t>(FaultState::none);
  if (__scoped_atomic_compare_exchange_n(
          word(StatusField::state), &none,
          static_cast<uint32_t>(FaultState::writing), false, __ATOMIC_RELAXED,
          __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM)) {
    *word(StatusField::where) = static_cast<uint32_t>(where);
    *word(StatusField::rank) = static_cast<uint32_t>(rank);
    *word(StatusField::block) = blockIdx.x;
    *word(StatusField::peer) = peer;
    *word(StatusField::count) = count;
    *word(StatusField::want) = want;
    *word(StatusField::elapsed_low) = static_cast<uint32_t>(waited);
    *word(StatusField::elapsed_high) = static_cast<uint32_t>(waited >> 32);
    __scoped_atomic_store_n(word(StatusField::state),
                            static_cast<uint32_t>(FaultState::written),
                            __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
  }
  // A checked build says what it waited for, a line a wave, before it stops:
  // the trap alone says only that something did.
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG && (threadIdx.x & 63) == 0)
    printf(
        "hip_custom_communicator give_up: where %u rank %d block %u thread "
        "(%u, %u) peer %u count %u want %u waited %llu ticks\n",
        static_cast<uint32_t>(where), rank, blockIdx.x, threadIdx.x,
        threadIdx.y, peer, count, want,
        static_cast<unsigned long long>(waited));
  // A checked build's launch finishes (check_failed); a normal build stops.
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG) __builtin_amdgcn_endpgm();
  __builtin_trap();
}

}  // namespace internal

}  // namespace hip_custom_communicator
