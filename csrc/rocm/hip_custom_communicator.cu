// SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc.
// All rights reserved.
//
// Our HIP collectives as torch ops, built into `_rocm_C`: the host side in one
// file, the kernels in hip_custom_communicator/kernels/. In three parts: 1. the
// types and tables (the machine, the build, the errors, each op's builds, the
// layout, the state between Python and C++); 2. the helpers the API calls; 3.
// the API, a function a torch op, each taking what Python passes and returning
// the Error's number it was refused for, or none (it raises nothing): its
// preconditions, its kernel's arguments, its launch, its postcondition.
// Self-contained: torch and the HIP runtime, nothing from aiter.

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <hip/hip_runtime.h>
#include <dirent.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>
#include "ops.h"
#include <limits>

#include "hip_custom_communicator/kernels/all_reduce.cuh"
#include "hip_custom_communicator/kernels/all_reduce_add_attn_res_rms_norm.cuh"
#include "hip_custom_communicator/kernels/all_reduce_add_rms_norm.cuh"
#include "hip_custom_communicator/kernels/common.cuh"
#include "hip_custom_communicator/kernels/reduce_scatter.cuh"

// THE DECLARATIONS torch_bindings.cpp registers, so a definition here that
// differs is a compile error, not a mismatch at run time.
#include "ops.h"

// =============================================================================
// 1. THE TYPES AND TABLES: the machine, the build, the errors, each op's
//    builds, the layout's constants, and the state between Python and C++.
// =============================================================================

namespace hip_custom_communicator {

// WHAT A CONFIG MOVES ACROSS THE LINKS, the host's alone: a config's
// `quantization` field. A kernel takes QUANTIZE_INT4 (bool), which the host
// derives per call from it and quantize_from_bytes.
namespace Quantization {
constexpr int exact = 0;
constexpr int int4 = 1;
}  // namespace Quantization

// THE BUILD: a PACK is the widest load (global_load_dwordx4), the unit every
// kernel loads, sums and stores, as in vLLM's and aiter's custom all-reduce;
// the dtypes and worlds dispatch instantiates, and the check refuses the rest.
// The exchange is sized at open from the largest call, never a constant; a
// build's block is its config's (any the device launches).
constexpr int kPackBytes = 16;
constexpr std::array<at::ScalarType, 2> kDtypes = {at::ScalarType::Half,
                                                   at::ScalarType::BFloat16};
constexpr std::array<int, 3> kWorlds = {2, 4, 8};

// =============================================================================
// TYPES
// =============================================================================

// THE VOCABULARY, every type the tables and the handle name: why a call cannot
// run (`Error`), and what is compiled in (each op's build, its algorithm its op
// file's). `at::ScalarType` is common.cuh's.

// WHY C++ REFUSED. C++ never raises: every function returns its Error as a
// value (none: it ran), and Python names it by number from its mirror of this
// enum.
enum class Error : int {
  not_contiguous = 0,
  dtype_not_built = 1,
  world_not_built = 2,
  row_not_packs = 3,
  partials_too_small = 4,
  grid_not_resident = 5,
  staging_too_small = 6,
  device_not_built = 7,
  weight_not_built = 9,
  no_such_kernel = 10,
  not_built = 12,
  launch_out_of_range = 13,
  field_not_positive = 14,
  field_not_this_templates = 15,
  inner_stride_not_one = 17,
  row_stride_not_packs = 18,
  no_such_quantization = 19,
  quantization_not_supported = 20,
  block_too_large = 21,
  lds_too_large = 22,
  scratch_too_large = 23,
  unfenced_too_small = 24,
  pack_out_of_range = 25,
  hip_failed = 26,
  not_on_device = 27,
  shape_mismatch = 28,
  dtype_mismatch = 29,
  not_aligned = 30,
  inputs_overlap_outputs = 31,
  field_out_of_range = 32,
  view_past_region = 33,
  tables_full = 34,
  ranks_disagree = 35,
  not_open = 36,
};

// A KERNEL'S CONFIG, in two parts and every field always set: its BUILD, what
// is compiled in (its algorithm first, which of its kernel's bodies it is), and
// its
// LAUNCH, what is chosen at each call. A tuned kernel holds both, a forced call
// gives both, and select hands both to the launch; no field is ever 0 for "left
// to someone else".
//
// A LAUNCH DIMENSION, the kernel's dim3: x, y and z. Every kernel here indexes
// x alone, so its configs hold y = z = 1; one that indexed y or z would widen
// its space, nothing else.
struct Dim3 {
  int x = 1;
  int y = 1;
  int z = 1;
  bool operator==(const Dim3&) const = default;
};

// EACH OP'S BUILD, Triton's autotune config less the grid: the block, its
// threads in each dimension (x: Triton's num_warps x 64), the waves a SIMD (an
// EU) the compiler fits its registers to (__launch_bounds__'s second value,
// Triton's waves_per_eu; 1 asks nothing), and the tile, TILE_M rows x TILE_N
// columns in elements; TILE_K is the reduced dimension a step (AttnRes's
// sources a step); `quantization`, what crosses the links (common.cuh's
// QUANTIZATION): a two-shot's build is listed as it is and moving INT4 (two
// configs of the table), forced like any field or named by a tuned point;
// `pack_bytes`, a pack's bytes, the unit every load, sum and store moves
// (PACK_BYTES: 16, the widest load, or 8, 4 or 2, a row's that is not whole
// wider packs, so every row has a build whose loads are aligned: PyTorch's and
// vLLM's vector width by alignment, tuned here like the tile). ONE TYPE AN OP,
// so a config cannot cross ops: its op (`op`), the algorithms its kernel has by
// name (`algorithms`, the table's and Python's), and its fields. A norm on this
// rank's rows has one algorithm (it reads no peer), so its build has none.
template <typename A>
struct Named {
  A algorithm;
  const char* name;
};

// all_reduce: a tile a work item, the grid striding over the buffer's.
struct AllReduceBuildConfig {
  static constexpr Named<AllReduceAlgorithm> algorithms[] = {
      {AllReduceAlgorithm::one_shot_pull, "one_shot_pull"},
      {AllReduceAlgorithm::one_shot_push, "one_shot_push"},
      {AllReduceAlgorithm::two_shot_pull, "two_shot_pull"},
      {AllReduceAlgorithm::two_shot_push, "two_shot_push"},
      {AllReduceAlgorithm::one_shot_push_staged, "one_shot_push_staged"}};
  AllReduceAlgorithm algorithm;
  Dim3 block;
  int waves_per_eu;
  int tile_m;
  int tile_n;
  int quantization = Quantization::exact;  // a Quantization
  int pack_bytes = kPackBytes;             // a pack's bytes (PACK_BYTES)
  bool operator==(const AllReduceBuildConfig&) const = default;
};

// all_reduce_rms_norm.
struct AllReduceRmsNormBuildConfig {
  static constexpr Named<AllReduceRmsNormAlgorithm> algorithms[] = {
      {AllReduceRmsNormAlgorithm::one_shot_pull, "one_shot_pull"},
      {AllReduceRmsNormAlgorithm::two_shot_pull, "two_shot_pull"},
      {AllReduceRmsNormAlgorithm::two_shot_push, "two_shot_push"}};
  AllReduceRmsNormAlgorithm algorithm;
  Dim3 block;
  int waves_per_eu;
  int tile_m;
  int tile_n;
  int quantization = Quantization::exact;  // a Quantization
  int pack_bytes = kPackBytes;             // a pack's bytes (PACK_BYTES)
  bool operator==(const AllReduceRmsNormBuildConfig&) const = default;
};

// all_reduce_add_rms_norm.
struct AllReduceAddRmsNormBuildConfig {
  static constexpr Named<AllReduceAddRmsNormAlgorithm> algorithms[] = {
      {AllReduceAddRmsNormAlgorithm::one_shot_pull, "one_shot_pull"},
      {AllReduceAddRmsNormAlgorithm::two_shot_pull, "two_shot_pull"},
      {AllReduceAddRmsNormAlgorithm::two_shot_push, "two_shot_push"}};
  AllReduceAddRmsNormAlgorithm algorithm;
  Dim3 block;
  int waves_per_eu;
  int tile_m;
  int tile_n;
  int quantization = Quantization::exact;  // a Quantization
  int pack_bytes = kPackBytes;             // a pack's bytes (PACK_BYTES)
  bool operator==(const AllReduceAddRmsNormBuildConfig&) const = default;
};

// all_reduce_add_attn_res_rms_norm: TILE_K sources a step.
struct AllReduceAddAttnResRmsNormBuildConfig {
  static constexpr Named<AllReduceAddAttnResRmsNormAlgorithm> algorithms[] = {
      {AllReduceAddAttnResRmsNormAlgorithm::one_shot_pull, "one_shot_pull"},
      {AllReduceAddAttnResRmsNormAlgorithm::two_shot_pull, "two_shot_pull"},
      {AllReduceAddAttnResRmsNormAlgorithm::two_shot_push, "two_shot_push"}};
  AllReduceAddAttnResRmsNormAlgorithm algorithm;
  Dim3 block;
  int waves_per_eu;
  int tile_m;
  int tile_n;
  int tile_k;
  int quantization = Quantization::exact;  // a Quantization
  int pack_bytes = kPackBytes;             // a pack's bytes (PACK_BYTES)
  bool operator==(const AllReduceAddAttnResRmsNormBuildConfig&) const = default;
};

// rms_norm (its kernel rms_norm).
struct RmsNormBuildConfig {
  Dim3 block;
  int waves_per_eu;
  int tile_m;
  int tile_n;
  int quantization = Quantization::exact;  // a Quantization
  int pack_bytes = kPackBytes;             // a pack's bytes (PACK_BYTES)
  bool operator==(const RmsNormBuildConfig&) const = default;
};

// add_rms_norm (its kernel add_rms_norm).
struct AddRmsNormBuildConfig {
  Dim3 block;
  int waves_per_eu;
  int tile_m;
  int tile_n;
  int quantization = Quantization::exact;  // a Quantization
  int pack_bytes = kPackBytes;             // a pack's bytes (PACK_BYTES)
  bool operator==(const AddRmsNormBuildConfig&) const = default;
};

// add_attn_res_rms_norm (its kernel add_attn_res_rms_norm): TILE_K sources a
// step.
struct AddAttnResRmsNormBuildConfig {
  Dim3 block;
  int waves_per_eu;
  int tile_m;
  int tile_n;
  int tile_k;
  int quantization = Quantization::exact;  // a Quantization
  int pack_bytes = kPackBytes;             // a pack's bytes (PACK_BYTES)
  bool operator==(const AddAttnResRmsNormBuildConfig&) const = default;
};

// =============================================================================
// THE COMMUNICATOR'S MEMORY
// =============================================================================

// C++ OWNS THE MEMORY, in CommsState: `open` allocates every buffer, each its
// own allocation, and `free` frees them.
//  - THE EXCHANGE'S REGIONS (exported over legacy HIP IPC, opened by every peer
//    at connect): the signals, the staging, the partials, the scales, the
//    pushed one-shot's inbox. A peer's is at whatever address its opening gave,
//    so a kernel takes each region as its own DEVICE ARRAY of every rank's
//    address of it.
//  - THIS RANK'S ALONE: those device arrays, the pushed one-shot's epochs, the
//    input tables, the status a kernel writes before it traps.
// AN EXACT CALL'S INPUT is read by its peers IN PLACE (aiter's gfx950 pattern),
// through an INPUT TABLE (InputTables): a captured call's table points at every
// rank's own tensor, written when the capture exits; an eager call's input is
// copied into this rank's staging on the launch's stream, and its table (slot
// 0, written at connect) points at every rank's staging. An INT4 call's kernel
// quantizes its input into the staging itself, before its first barrier.

// A HIP CALL'S FAILURE, returned as hip_failed by the function it is in (every
// caller of HIP here returns an Error); the HIP error itself is cleared.
#define HIP_CHECK(expr)         \
  do {                          \
    if ((expr) != hipSuccess) { \
      (void)hipGetLastError();  \
      return Error::hip_failed; \
    }                           \
  } while (0)

// =============================================================================
// THE COMPILED KERNELS: every instance of every kernel, a row of its kernel's
// table, which a launch looks its kernel up in.
// =============================================================================

// EACH NORM ON ROWS' ONE ALGORITHM, its op's own (the table's "rows"; its
// kernel has no algorithm parameter).
enum class RmsNormAlgorithm : int { rows };
enum class AddRmsNormAlgorithm : int { rows };
enum class AddAttnResRmsNormAlgorithm : int { rows };

// A COMPILED INSTANCE OF A KERNEL, a row of its table: the build it was
// compiled at (a tuned config, its grid aside), the world (0 for a norm on
// rows, which has none), its wave's lanes and its dtypes, which of the build's
// instances it is (contiguous: the plain two-shots' contiguous walk; chunked: a
// row wider than its tile; tail: an INT4 row with a short last group;
// quantize_int4: what it moves), and its address. GENERATED (builds.py, at
// configure), one table a kernel, `k<Kernel>Instances`: every row the table's
// configs reach, so a config of the table is always a row, and a call finds its
// kernel or is refused (not_built).
template <typename ALGORITHM>
struct Instance {
  ALGORITHM algorithm;
  int block_x, block_y, block_z, waves_per_eu, tile_m, tile_n, tile_k;
  int quantization, pack_bytes;
  int world, wave_size;
  at::ScalarType dtype, weight_dtype;
  bool contiguous, chunked, tail, quantize_int4;
  void* kernel;
};

// EACH KERNEL'S TYPE at its dtypes, T the input's and W the weight's (a
// kernel's parameters are the same at every build: any instance's type is every
// instance's), what a launch casts a row's address to.
template <typename T, typename W>
using AllReduceKernel =
    decltype(&all_reduce<AllReduceAlgorithm{}, false, T, 2, 1, 1, 64, 64, 16, 1,
                         false, false>);
template <typename T, typename W>
using AllReduceRmsNormKernel =
    decltype(&all_reduce_rms_norm<AllReduceRmsNormAlgorithm{}, false, T, W, 2,
                                  1, 64, 64, 16, 1, false, false>);
template <typename T, typename W>
using AllReduceAddRmsNormKernel =
    decltype(&all_reduce_add_rms_norm<AllReduceAddRmsNormAlgorithm{}, false, T,
                                      W, 2, 1, 64, 64, 16, 1, false, false>);
template <typename T, typename W>
using AllReduceAddAttnResRmsNormKernel =
    decltype(&all_reduce_add_attn_res_rms_norm<
             AllReduceAddAttnResRmsNormAlgorithm{}, false, T, 2, 1, 1, 1, 64,
             64, 16, 1, false, false>);
template <typename T, typename W>
using RmsNormKernel = decltype(&rms_norm<T, W, 1, 1, 64, 64, 16, 1, false>);
template <typename T, typename W>
using AddRmsNormKernel =
    decltype(&add_rms_norm<T, W, 1, 1, 64, 64, 16, 1, false>);
template <typename T, typename W>
using AddAttnResRmsNormKernel =
    decltype(&add_attn_res_rms_norm<T, 1, 1, 1, 64, 64, 16, 1, false>);

// EVERY KERNEL INSTANCE IS COMPILED IN ITS BUILD'S OWN FILE (builds.py):
// declared here, so this file instantiates none of them; then the tables.
#include "hip_custom_communicator_instances.inc"
#include "hip_custom_communicator_builds.inc"

// THE INPUT TABLES: where an exact pull reads every rank's input in place, so a
// captured call copies nothing. A table is `world` addresses, one pointer its
// kernel takes (aiter's RankData*). Slot 0, the eager table, points at every
// rank's staging (written by connect; an eager call copies its input there
// first). Every later slot is a captured launch's: the launch reserves it
// (`next_slot`) and records its input (`pending`); when the capture exits,
// `graph_ipc_meta` exports the pending inputs and `register_graph_buffers`
// opens every peer's (`opened`) and writes their tables. A slot is never reused
// (a graph keeps its address). An input may be overwritten once its launch
// ends: every peer's last read of it precedes a barrier the launch waits on.
struct InputTables {
  // OWNED, device: `slots` tables of `world` addresses.
  void** device = nullptr;
  int64_t slots = 0;
  // The next captured launch's slot.
  int64_t next_slot = 1;
  // (slot, input): captured, not yet registered.
  std::vector<std::pair<int64_t, const void*>> pending;
  // BORROWED: a peer's input allocation, by its IPC handle.
  std::map<std::vector<char>, char*> opened;
};

// THE SYMMETRIC MEMORY (the standard term: OpenSHMEM's symmetric heap,
// NVSHMEM's, PyTorch's SymmetricMemory): the buffers every rank allocates
// alike, the same sizes and layout, and exchanges, every peer opening every
// other's over legacy HIP IPC. Allocated by `open` (each region its own
// uncached allocation, its IPC handle in open's answer), its peers opened and
// its device arrays written by `connect`, the peers closed by `close_peers`,
// freed by `free`. Nothing else is exchanged.
struct SymmetricMemory {
  // THE LAYOUT, the same on every rank (open says why each is what it is): the
  // signals' blocks (a parity's slots, every block the device holds at once),
  // the unfenced region's blocks and a row's payload bytes.
  int64_t max_blocks = 0, unfenced_blocks = 0, unfenced_slot_bytes = 0;
  // THE REGIONS' BYTES: the staging's, the largest call's input or payload; the
  // partials', the most any two-shot keeps.
  int64_t signals_bytes = 0, staging_bytes = 0, partials_bytes = 0,
          scales_bytes = 0, unfenced_bytes = 0;
  // THE REGIONS, OWNED: the signals (the barrier flags), the staging (what this
  // rank publishes: an eager call's copy, an INT4 payload, a pushed two-shot's
  // inbox), the partials (its share's sums), the scales (the add-RMSNorm pull's
  // INT4 row scales), the unfenced region (the pushed one-shot's inbox).
  uint32_t* signals = nullptr;
  char* staging = nullptr;
  char* partials = nullptr;
  float* scales = nullptr;
  char* unfenced = nullptr;
  // EACH REGION'S DEVICE ARRAY of every rank's address of it (OWNED, this
  // rank's alone, written by connect): what every launch hands its kernel.
  uint32_t* const* signals_ptr_array = nullptr;
  char* const* staging_ptr_array = nullptr;
  char* const* partials_ptr_array = nullptr;
  float* const* scales_ptr_array = nullptr;
  char* const* unfenced_ptr_array = nullptr;
  // BORROWED: every peer's regions as connect opened them, for close_peers.
  std::vector<char*> peers;
};

// THE DEVICE A COMMUNICATOR RUNS ON, as open asked it once: its index, its arch
// (gcnArchName's, `gfx950`: the tuned table Python reads), its wave's lanes (a
// launch runs an instance compiled at them), its wall clock (a wait's timeout
// is counted in its ticks), and the limits every launch is checked against: a
// block's LDS, its compute units, the threads a CU holds, and a lane's scratch
// (open's share of the device's).
struct DeviceProperties {
  int index = 0;
  std::string arch;
  int wave_size = 0;
  int64_t clock_khz = 0;
  int64_t lds_bytes = 0, compute_units = 0, threads_a_cu = 0,
          scratch_a_lane = 0;
};

// THE STATE BETWEEN PYTHON AND C++, one a communicator: what it was given (this
// rank, the world, the device, the largest call, a wait's bound), what open
// asked of the device, the symmetric memory, this rank's own buffers (the
// pushed one-shot's epochs, the status), and the input tables. Made by `open`,
// completed by `connect`, released by `close_peers` then `free`; Python holds
// it as a handle. A launch reads it, and a captured in-place launch reserves
// its table slot in it: its one mutation during a call.
struct CommsState {
  int rank = 0;
  int world = 0;
  int64_t max_rows = 0;
  uint64_t timeout_ticks = 0;
  DeviceProperties device;
  SymmetricMemory symmetric;
  // OWNED, this rank's alone: the pushed one-shot's epochs (a block's
  // `unfenced_epoch_words` int32 words), and a kernel's fault record, written
  // before it traps.
  int32_t* unfenced_epochs = nullptr;
  int64_t unfenced_epoch_words = 0;
  uint32_t* status_device = nullptr;
  InputTables input_tables;
};

}  // namespace hip_custom_communicator

// WHAT CROSSES THE TORCH BOUNDARY, as the tuples an op schema can return.
using Names = std::vector<std::string>;
using OpenWire =
    std::tuple<std::optional<int64_t>, int64_t, std::vector<int64_t>,
               std::string, std::vector<int64_t>>;
using GraphMetaWire = std::tuple<std::optional<int64_t>, std::vector<int64_t>,
                                 std::vector<int64_t>>;
// AN ERROR CODE ACROSS THE BOUNDARY: none when the call ran, else the Error's
// number (a torch op returns no enum).
using ErrorCode = std::optional<int64_t>;

// =============================================================================
// 2. THE HELPERS: what the API functions call, and the compile-time checks of
//    the tables.
// =============================================================================

namespace hip_custom_communicator {

// WHETHER THE BUILD HOLDS a dtype or a world: dispatch instantiates exactly
// these.
template <typename ELEMENT, size_t NUM_BUILT>
constexpr bool built_in(const std::array<ELEMENT, NUM_BUILT>& built,
                        ELEMENT x) {
  for (const ELEMENT& b : built)
    if (b == x) return true;
  return false;
}

constexpr bool dtype_built(at::ScalarType d) { return built_in(kDtypes, d); }

constexpr bool world_built(int world) {
  return built_in(kWorlds, world);
}  // numbered from 1

// AN ALGORITHM'S PARTS, where a host decision needs only one: whether it is a
// one-shot or a two-shot, and whether it pushes (only the plain all-reduce
// pushes a one-shot: as lines, or staged).
template <typename A>
constexpr bool is_one_shot(A a) {
  if constexpr (std::is_same_v<A, AllReduceAlgorithm>)
    return a == A::one_shot_pull || a == A::one_shot_push ||
           a == A::one_shot_push_staged;
  else
    return a == A::one_shot_pull;
}

template <typename A>
constexpr bool is_two_shot(A a) {
  return a == A::two_shot_pull || a == A::two_shot_push;
}

template <typename A>
constexpr bool is_push(A a) {
  if constexpr (std::is_same_v<A, AllReduceAlgorithm>)
    return a == A::one_shot_push || a == A::one_shot_push_staged ||
           a == A::two_shot_push;
  else
    return a == A::two_shot_push;
}

constexpr int64_t volume(const Dim3& d) {
  return static_cast<int64_t>(d.x) * d.y * d.z;
}

// Whether `d` is x alone: y = z = 1, every kernel's today.
constexpr bool x_alone(const Dim3& d) { return d.y == 1 && d.z == 1; }

constexpr int elem_bytes(at::ScalarType d) {
  return d == at::ScalarType::Float ? 4 : 2;
}

// (A launch's input table: `input_table`, below, after CommsState.)

// WHY `kernel` CANNOT LAUNCH HERE, `grid` blocks of `block` threads and
// `shared_bytes` of dynamic LDS (a wide AttnRes row's logits), if it cannot:
// every check of a launch, in one place, what the compiled kernel asks
// (hipFuncGetAttributes, HIP's occupancy API) against what this device gives.
//  - block_too_large: more threads than the kernel takes (its launch bounds).
//  - lds_too_large: its LDS and the launch's dynamic LDS, past the device's a
//    block.
//  - scratch_too_large: its scratch a lane past the state's scratch_a_lane
//    (open).
//  - grid_not_resident, where the kernel waits on its peers (`waits_on_peers`;
//    a norm on rows waits on none, and any grid runs): more blocks than the
//    device holds at once, the textbook query (CUDA's
//    cudaOccupancyMaxActiveBlocksPerMultiprocessor: blocks a CU from the
//    kernel's registers, LDS and the CU's wave slots) times the CUs. A peer
//    barrier spins until the same block on every peer arrives, so a grid past
//    it can hang.
// HIP refuses the first two at the launch; the scratch only when the dispatch
// runs, aborting the queue and the process with it (the tune's crashes,
// 2026-10-24: AttnRes at 64 threads, tile 4x16384x4, 21 KiB a lane); the grid
// never (it hangs). The device's side is the state's.
inline std::optional<Error> check_kernel(const CommsState& s,
                                         const void* kernel, const Dim3& grid,
                                         const Dim3& block, int shared_bytes,
                                         bool waits_on_peers) {
  const int threads = static_cast<int>(volume(block));
  hipFuncAttributes f;
  HIP_CHECK(hipFuncGetAttributes(&f, kernel));
  if (threads > f.maxThreadsPerBlock) return Error::block_too_large;
  if (static_cast<int64_t>(f.sharedSizeBytes) + shared_bytes >
      s.device.lds_bytes)
    return Error::lds_too_large;
  if (static_cast<int64_t>(f.localSizeBytes) > s.device.scratch_a_lane)
    return Error::scratch_too_large;
  int per_cu = 0;  // asked of a kernel that launches alone
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &per_cu, kernel, threads, shared_bytes));
  if (waits_on_peers && volume(grid) > int64_t{per_cu} * s.device.compute_units)
    return Error::grid_not_resident;
  return std::nullopt;
}

// =============================================================================
// CONFIG
// =============================================================================

// Built kernels, tuned tables, and the complete configuration of each call.

// EACH OP'S ALGORITHMS, what the host sizes and checks by: how its two-shot
// splits the work (SLICES COLUMNS: by columns, every block keeping every row,
// the pulled AttnRes and every pushed fused op; else by rows, or the plain
// all-reduce's flat walk), and whether its INT4 is WHOLE GROUPS A THREAD (the
// pulled norms, all_reduce_add_rms_norm.cuh: a block a row, TILE_N / (32 *
// THREADS) groups a thread; the tile does not size the payload, so any build
// sizes alike).
constexpr bool slices_columns(AllReduceAlgorithm) { return false; }
constexpr bool slices_columns(AllReduceRmsNormAlgorithm a) {
  return a == AllReduceRmsNormAlgorithm::two_shot_push;
}

constexpr bool slices_columns(AllReduceAddRmsNormAlgorithm a) {
  return a == AllReduceAddRmsNormAlgorithm::two_shot_push;
}

constexpr bool slices_columns(AllReduceAddAttnResRmsNormAlgorithm a) {
  return is_two_shot(a);
}

constexpr bool int4_groups_a_thread(AllReduceAlgorithm) { return false; }
constexpr bool int4_groups_a_thread(AllReduceRmsNormAlgorithm a) {
  return a == AllReduceRmsNormAlgorithm::two_shot_pull;
}

constexpr bool int4_groups_a_thread(AllReduceAddRmsNormAlgorithm a) {
  return a == AllReduceAddRmsNormAlgorithm::two_shot_pull;
}

constexpr bool int4_groups_a_thread(AllReduceAddAttnResRmsNormAlgorithm a) {
  return a == AllReduceAddAttnResRmsNormAlgorithm::two_shot_pull;
}

// AN INT4 ROW WITH A TAIL, the INT4 two-shots' TAIL instance: the flat walk
// (the plain two-shots) where the call is not whole groups; the plain
// one-shots, a groups-a-thread pulled norm, where a row is not; a column
// two-shot where a row is not whole groups for every rank.
template <typename A>
constexpr bool int4_tail(A algorithm, int64_t m, int64_t n, int world) {
  if constexpr (std::is_same_v<A, AllReduceAlgorithm>)
    return is_two_shot(algorithm) ? m * n % kInt4Group != 0
                                  : n % kInt4Group != 0;
  else if (int4_groups_a_thread(algorithm))
    return n % kInt4Group != 0;
  else
    return n % kInt4Group != 0 || n / kInt4Group % world != 0;
}

// THE INT4 FLAT WALK'S SHARE (reduce_scatter.cuh, the plain push two-shot's
// INT4): an [m, n] input's elements numbered row-major, a rank's share whole
// groups, the last rank's to the end. Its payloads lie on views of their own,
// [1, world * share] in the inbox and [1, share] in the partials: on the
// input's [m, n] the rounded-up shares would run past its values.
constexpr int64_t int4_flat_share(int64_t m, int64_t n, int world) {
  return ((m * n + kInt4Group - 1) / kInt4Group + world - 1) / world *
         kInt4Group;
}

// =============================================================================
// EACH OP'S KERNELS.
// =============================================================================

// EVERY BUILD THE TABLE NAMES IS COMPILED AND TAKES ANY CALL ITS KERNEL CAN RUN
// (Michael, October 25: the build makes what it is told): each kernel walks any
// tile at any block, its tails its own; what a call cannot do is refused by
// name at the call (`refused`), its capacity (the regions, the unfenced slot),
// the hardware's (`check_kernel`), or an algorithm's (INT4 on a one-shot).

// A ROW OF THE STAGING AND PARTIALS AS A KERNEL VIEWS IT: n, or a
// groups-a-thread INT4 norm's payload row, whole groups (its kernels' n_pad:
// every row's groups start on a group).
template <typename A>
constexpr int64_t payload_row(A algorithm, int quantization, int64_t n) {
  return int4_groups_a_thread(algorithm) && quantization == Quantization::int4
             ? (n + kInt4Group - 1) / kInt4Group * kInt4Group
             : n;
}

// THE STAGING'S ROWS AS A KERNEL VIEWS IT: the input's m, or a groups-a-thread
// INT4 norm's inbox, WORLD slots of an owner's ceil(m / WORLD) rows.
template <typename A>
constexpr int64_t staged_rows(A algorithm, int quantization, int64_t m,
                              int world) {
  return int4_groups_a_thread(algorithm) && quantization == Quantization::int4
             ? world * ((m + world - 1) / world)
             : m;
}

// =============================================================================
// AN OP'S ALGORITHM BY NAME: the call names it, among its kernel's.
// =============================================================================

// AN ALGORITHM OF B'S KERNEL by its name, or none where the kernel has no such
// algorithm.
template <typename B>
constexpr std::optional<decltype(B::algorithm)> algorithm_named(
    std::string_view name) {
  for (const auto& a : B::algorithms)
    if (name == a.name) return a.algorithm;
  return std::nullopt;
}

// =============================================================================
// THE CHECK: the first Error config `c` meets on the call here, in this order,
// or none. Every Error is a capability (a kernel that cannot take the input),
// never "the unfused ops would be faster". Residency, which needs the compiled
// kernel, is each select's last check.
// =============================================================================

// A KERNEL'S PARTIALS, row-major, the whole call's: the plain two-shot its
// slice of every row; a column two-shot the whole reduced tensor (each rank's
// columns at their place); a row two-shot its rank's rows, twice where it
// leaves two results (out and the residual). A one-shot reads the inputs and
// keeps nothing; `packs` a row's in packs of `pack` bytes (the build's). THE
// ADD-RMSNORM ROW TWO-SHOT's rows each carry their scale in a slot of
// kScaleSlotBytes before the row (common.cuh;
// kernels/all_reduce_add_rms_norm.cuh, its exact body: a row with its scale,
// one buffer), its row whole kScaleSlotBytes so the next row's scale is aligned
// (scale_slot_row).
template <typename A>
constexpr int64_t partials_need(A algorithm, int64_t m, int64_t packs,
                                int world, int64_t pack) {
  if (!is_two_shot(algorithm)) return 0;
  if constexpr (std::is_same_v<A, AllReduceAlgorithm>) {
    return m * ((packs + world - 1) / world) * pack;
  } else {
    if (slices_columns(algorithm)) return m * packs * pack;
    const int64_t row_bytes =
        std::is_same_v<A, AllReduceAddRmsNormAlgorithm> && !is_push(algorithm)
            ? scale_slot_row(packs * pack) + kScaleSlotBytes
            : packs * pack;
    return (m + world - 1) / world * row_bytes;
  }
}

// EVERY ALGORITHM OF EVERY KERNEL, `f` handed each (a norm on rows has none).
template <typename F>
constexpr void every_algorithm(F&& f) {
  for (const auto& a : AllReduceBuildConfig::algorithms) f(a.algorithm);
  for (const auto& a : AllReduceRmsNormBuildConfig::algorithms) f(a.algorithm);
  for (const auto& a : AllReduceAddRmsNormBuildConfig::algorithms)
    f(a.algorithm);
  for (const auto& a : AllReduceAddAttnResRmsNormBuildConfig::algorithms)
    f(a.algorithm);
}

// THE CHECK OF A CALL'S CONFIG (its build, of its op's type B, its grid, and
// the bytes it quantizes from) on a call of `m` rows of `n`: the world and the
// dtype built; the config's ranges, each the hardware's (STANDARDS 3.2): the
// grid and the block x alone (every kernel's one dimension), the block whole
// waves up to the device's workgroup (a wider block's register trade is the
// tune's to measure), the grid at most the blocks the signals have slots for
// (whether this kernel's are resident is grid_not_resident), the waves a EU the
// CU holds, the tile positive, a pack a load the hardware has,
// quantize_from_bytes an INT4 build's alone; then what the build cannot run;
// then what the regions cannot hold. With no state (`collective` false) the
// checks that need peers are left out.
template <typename B>
std::optional<Error> refused(bool collective, int world, int64_t staging_bytes,
                             int64_t partials_bytes, int64_t max_blocks,
                             const B& build, Dim3 grid,
                             std::optional<int64_t> quantize_from_bytes,
                             at::ScalarType dtype, int64_t m, int64_t n) {
  if (collective && !world_built(world)) return Error::world_not_built;
  if (!dtype_built(dtype)) return Error::dtype_not_built;
  if (!x_alone(grid) || grid.x < 1 || grid.x > max_blocks)
    return Error::launch_out_of_range;
  int tile_k = 1;  // a kernel with no K: its identity
  if constexpr (requires { build.tile_k; }) tile_k = build.tile_k;
  if (build.tile_m < 1 || build.tile_n < 1 || tile_k < 1)
    return Error::field_not_positive;
  if (build.quantization == Quantization::exact && quantize_from_bytes)
    return Error::field_not_this_templates;
  if (quantize_from_bytes.value_or(0) < 0) return Error::field_not_positive;
  // A PACK A LOAD THE HARDWARE HAS: 2, 4, 8 or 16 bytes (its widest).
  if (build.pack_bytes < 2 || build.pack_bytes > kPackBytes ||
      (build.pack_bytes & (build.pack_bytes - 1)) != 0)
    return Error::pack_out_of_range;
  const int64_t quantize_from = quantize_from_bytes.value_or(0);
  constexpr bool kPlain = std::is_same_v<B, AllReduceBuildConfig>;
  // THE BUILD'S PACK: at least an element, and whole packs a row (an aligned
  // load of the pack's width: a row that is not whole packs starts the next off
  // them; a narrower build's takes it).
  const int64_t pack = build.pack_bytes;
  if (pack < elem_bytes(dtype)) return Error::pack_out_of_range;
  if (n * elem_bytes(dtype) % pack != 0) return Error::row_not_packs;
  // INT4 ON A NORM ON ROWS (it moves nothing) OR A FUSED ONE-SHOT (it reads its
  // peers' inputs as they are): named before the build lookup, which lists no
  // such build; INT4 AT A NARROWER PACK than the widest: the format's word is a
  // pack of 8 values (common.cuh's QUANTIZATION).
  const bool int4_build = build.quantization == Quantization::int4;
  if constexpr (requires { build.algorithm; }) {
    if (int4_build &&
        ((!kPlain && is_one_shot(build.algorithm)) || pack != kPackBytes))
      return Error::quantization_not_supported;
  } else if (int4_build) {
    return Error::quantization_not_supported;
  }
  // A FUSED OP'S INT4 BUILD ALWAYS QUANTIZES: its kernels have no exact
  // instance of their own.
  if (int4_build && !kPlain && quantize_from != 0)
    return Error::field_not_this_templates;
  // A BUILD THE KERNEL DOES NOT LIST, named by what differs: its block, its
  // waves, its tile.
  if constexpr (!requires { build.algorithm; }) {
    return std::nullopt;  // a norm on rows reads no peer and no region
  } else {
    // WHAT THE CALL MOVES: INT4 where its INT4 build quantizes it (the call's
    // bytes from its quantize_from_bytes on), else exact (the INT4 build's
    // instance passing it exact).
    const bool int4 = int4_build && m * n * elem_bytes(dtype) >= quantize_from;
    if (!collective) return std::nullopt;
    const int64_t packs =
        (n * elem_bytes(dtype) + pack - 1) / pack;  // the last short
    // An INT4 payload lies in the region that holds it exact (common.cuh's INT4
    // payload layout): the input's in the inputs, the sums' in the partials, at
    // most one row of the partials's elements (its every view's bound); the
    // flat walk's on its shares' own views (int4_flat_share).
    const bool flat = kPlain && is_two_shot(build.algorithm);
    const int64_t share = int4_flat_share(m, n, world);
    const int64_t row = payload_row(build.algorithm, Quantization::int4, n);
    if (int4 &&
        (flat ? int4_payload_bytes(1, world * share)
              : int4_payload_bytes(
                    staged_rows(build.algorithm, Quantization::int4, m, world),
                    row)) > staging_bytes)
      return Error::staging_too_small;
    const int64_t partials =
        partials_need(build.algorithm, m, packs, world, pack);
    if (int4 && (flat ? int4_payload_bytes(1, share)
                 : int4_groups_a_thread(build.algorithm)
                     ? int4_payload_bytes((m + world - 1) / world, row)
                     : int4_payload_bytes(1, partials / elem_bytes(dtype))) >
                    partials_bytes)
      return Error::partials_too_small;
    // A COLLECTIVE RUNS THE WHOLE CALL IN ONE PASS: the regions are sized from
    // the largest call the server makes, so a call past them is refused here,
    // never cut into bands. Its input is in the staging whole: an eager exact
    // call's host copy (INPUT TABLES; a captured one's is read where it is,
    // refused here alike, so the eager and captured calls of one shape agree),
    // or a pushed input.
    if (partials > partials_bytes) return Error::partials_too_small;
    if (m * n * elem_bytes(dtype) > staging_bytes)
      return Error::staging_too_small;
    return std::nullopt;
  }
}

// A LAUNCH'S INPUT TABLE (INPUT TABLES): eager, the input copied into this
// rank's staging on `stream` and the eager table, slot 0, rows `n` apart;
// captured, the next slot reserved for it (filled by `register_graph_buffers`
// when the capture exits), rows at the input's own stride.
inline std::optional<Error> input_table(CommsState& s, const void* input,
                                        int64_t stride_m, int64_t m, int64_t n,
                                        int64_t elem_bytes, hipStream_t stream,
                                        void* const*& table,
                                        int64_t& table_stride_m) {
  hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
  HIP_CHECK(hipStreamIsCapturing(stream, &capture));
  const auto tables = s.input_tables.device;
  if (capture == hipStreamCaptureStatusActive) {
    if (s.input_tables.next_slot >= s.input_tables.slots)
      return Error::tables_full;
    s.input_tables.pending.emplace_back(s.input_tables.next_slot, input);
    table = tables + s.input_tables.next_slot * s.world;
    table_stride_m = stride_m;
    ++s.input_tables.next_slot;
    return std::nullopt;
  }
  const int64_t row = n * elem_bytes;
  if (m * row > s.symmetric.staging_bytes) return Error::staging_too_small;
  HIP_CHECK(hipMemcpy2DAsync(s.symmetric.staging, static_cast<size_t>(row),
                             input, static_cast<size_t>(stride_m * elem_bytes),
                             static_cast<size_t>(row), static_cast<size_t>(m),
                             hipMemcpyDeviceToDevice, stream));
  table = tables;
  table_stride_m = n;
  return std::nullopt;
}

// A VIEW INSIDE ITS REGION: `slots` x `rows` x `cols` elements of `elem` bytes
// at their strides end within `region_bytes` (an empty view is inside any).
// Every launch's views checked before it launches (October 26: n11 rebooted
// seven times timing the fused two-shots at size; the kernels check their
// indices only against the sizes they are handed).
constexpr bool view_fits(int64_t slots, int64_t stride_slot, int64_t rows,
                         int64_t stride_m, int64_t cols, int64_t stride_n,
                         int64_t elem, int64_t region_bytes) {
  if (slots <= 0 || rows <= 0 || cols <= 0) return true;
  const int64_t last =
      (slots - 1) * stride_slot + (rows - 1) * stride_m + (cols - 1) * stride_n;
  return stride_slot >= 0 && stride_m >= 0 && stride_n >= 0 &&
         (last + 1) * elem <= region_bytes;
}

// A LAUNCH DIMENSION AS HIP TAKES IT.
inline dim3 dim3_of(const Dim3& d) {
  return dim3(static_cast<unsigned>(d.x), static_cast<unsigned>(d.y),
              static_cast<unsigned>(d.z));
}

}  // namespace hip_custom_communicator

// =============================================================================
// THE TORCH OP BOUNDARY. Nothing crosses but values and one handle: the
// communicator's state (CommsState, C++'s: its memory, its peers, its captured
// inputs), made, connected and freed through the functions below; Python does
// only the exchange over its CPU group. EVERY ERROR IS A VALUE: no function
// here raises; each returns the Error's number it was refused for (none: it
// ran), and Python names it.
// =============================================================================

// EACH OP TAKES ITS STATE (a handle, below), its tensors and values, then its
// config's fields as plain values, its op's own (an algorithm where its kernel
// has several, tile_k where it has K). Nothing is picked here: Python fills
// what a caller leaves out and hands its table's pick the same way; the API
// checks it.

namespace {
using hip_custom_communicator::Error;

// EVERY CONFIG FIELD AN INT: the API takes ints.
bool all_ints(std::initializer_list<int64_t> vs) {
  for (const int64_t v : vs)
    if (v < std::numeric_limits<int>::min() ||
        v > std::numeric_limits<int>::max())
      return false;
  return true;
}

// The ops' tensors: on device; their layouts are their strides, checked behind
// the interface.
bool on_device(std::initializer_list<const torch::Tensor*> ts) {
  for (const torch::Tensor* t : ts)
    if (!t->is_cuda()) return false;
  return true;
}

}  // namespace

ErrorCode error_code(std::optional<Error> e) {
  return e ? ErrorCode(static_cast<int64_t>(*e)) : std::nullopt;
}

namespace {
// NO INPUT OVERLAPS AN OUTPUT (STANDARDS): the kernels take each as its own
// `__restrict__` pointer, so two that share bytes would be undefined behavior.
// A missing tensor (null) overlaps nothing.
//
// EXACT FOR VIEWS OF ONE STACK, conservative otherwise. AttnRes reads
// `blocks[:, :num_blocks]` and writes `blocks[:, write_index]`: interleaved,
// every row the sources then the slot, so their first-to-last spans overlap
// though no element does (the span check refused Kimi's own append,
// 2026-10-13). A view whose dimensions after the first are one contiguous run
// is ROWS: a run of `width` bytes every `stride` bytes. Two rows views with the
// same stride are disjoint exactly when their runs do not meet within one
// period; anything else is compared by its span.
struct Span {
  uintptr_t first, last;    // the first byte and one past the last
  bool rows;                // whether it is rows (below)
  uintptr_t stride, width;  // a row's period and its run, in bytes
};

Span span_of(const torch::Tensor& t) {
  const uintptr_t first = reinterpret_cast<uintptr_t>(t.data_ptr());
  const int64_t elem = t.element_size();
  int64_t last = 0;
  for (int64_t d = 0; d < t.dim(); ++d) last += (t.size(d) - 1) * t.stride(d);
  Span s{first, first + static_cast<uintptr_t>((last + 1) * elem), false, 0, 0};
  if (t.dim() < 2 || t.stride(t.dim() - 1) != 1) return s;
  int64_t run = 1;
  for (int64_t d = t.dim() - 1; d >= 1; --d) {
    if (t.stride(d) != run) return s;
    run *= t.size(d);
  }
  if (t.stride(0) < run) return s;
  s.rows = true;
  s.stride = static_cast<uintptr_t>(t.stride(0) * elem);
  s.width = static_cast<uintptr_t>(run * elem);
  return s;
}

bool overlap(const Span& a, const Span& b) {
  if (a.last <= b.first || b.last <= a.first) return false;
  if (!(a.rows && b.rows && a.stride == b.stride)) return true;
  // Within one period: a's run at offset `at`, b's at `bt`, on a circle of
  // `stride` bytes.
  const uintptr_t at = a.first % a.stride, bt = b.first % b.stride;
  const auto meets = [&](uintptr_t x, uintptr_t xw, uintptr_t y) {
    return (y + a.stride - x) % a.stride <
           xw;  // y lands inside [x, x + xw) on the circle
  };
  return meets(at, a.width, bt) || meets(bt, b.width, at);
}

// THE SAME TENSOR EXACTLY: one start, one shape, one set of strides.
bool same_tensor(const torch::Tensor& a, const torch::Tensor& b) {
  return a.data_ptr() == b.data_ptr() && a.sizes() == b.sizes() &&
         a.strides() == b.strides();
}

// AN OUTPUT MAY BE AN INPUT ITSELF (`in_place`) only where the kernel is
// correct with that exact alias: the norms on rows, whose every element is
// written by the thread that read it, after it read it (STANDARDS 3.5: in place
// is the caller's explicit choice). A partial overlap never is.
bool overlapping(std::initializer_list<const torch::Tensor*> inputs,
                 std::initializer_list<const torch::Tensor*> outputs,
                 bool in_place = false) {
  for (const torch::Tensor* i : inputs)
    for (const torch::Tensor* o : outputs) {
      if (i == nullptr || o == nullptr || i->numel() == 0 || o->numel() == 0)
        continue;
      if (in_place && same_tensor(*i, *o)) continue;
      if (overlap(span_of(*i), span_of(*o))) return true;
    }
  return false;
}

}  // namespace

namespace {
using hip_custom_communicator::Error;

// A NORM CALL'S TENSORS, or the Error they are refused for: input [tokens,
// hidden], output (and with a residual, residual and residual_output) its shape
// and dtype, the weight 1-D of hidden at a pack (its dtype is the API's to
// refuse), no input sharing an output's memory.
std::optional<Error> norm_tensors(const torch::Tensor& input,
                                  const torch::Tensor* residual,
                                  const torch::Tensor& weight,
                                  const torch::Tensor& output,
                                  const torch::Tensor* residual_output,
                                  bool in_place) {
  if (!on_device({&input, &weight, &output})) return Error::not_on_device;
  if (input.dim() != 2 || output.sizes() != input.sizes() ||
      weight.dim() != 1 || weight.numel() != input.size(1))
    return Error::shape_mismatch;
  if (output.scalar_type() != input.scalar_type()) return Error::dtype_mismatch;
  // The weight is read a pack at a time, a pack's worth of input's elements per
  // load.
  const int64_t weight_pack = hip_custom_communicator::kPackBytes /
                              input.element_size() * weight.element_size();
  if (reinterpret_cast<uintptr_t>(weight.data_ptr()) % weight_pack != 0)
    return Error::not_aligned;
  if (residual != nullptr) {
    if (!on_device({residual, residual_output})) return Error::not_on_device;
    if (residual->sizes() != input.sizes() ||
        residual_output->sizes() != input.sizes())
      return Error::shape_mismatch;
    if (residual->scalar_type() != input.scalar_type() ||
        residual_output->scalar_type() != input.scalar_type())
      return Error::dtype_mismatch;
  }
  if (overlapping({&input, residual}, {&output, residual_output}, in_place) ||
      overlapping({&weight}, {&output, residual_output}))
    return Error::inputs_overlap_outputs;
  return std::nullopt;
}

// AN ATTNRES CALL'S TENSORS, as every AttnRes op takes them: `input` (the
// partial sum, or the delta) [tokens, hidden]; `prefix` (none: the sum starts
// it), `prefix_output` and `output` its shape; `blocks` [tokens, sources,
// hidden], its first `num_blocks` the sources read; `block_output` (none: no
// block written) a [tokens, hidden] slot, by the standard not one of the
// sources; every one input's dtype. Or the Error they are refused for.
std::optional<Error> attn_res_tensors(
    const torch::Tensor& input, const std::optional<torch::Tensor>& prefix,
    const torch::Tensor& blocks, const torch::Tensor& norm_weight,
    const torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& output_norm_weight,
    const torch::Tensor& prefix_output, const torch::Tensor& output,
    const std::optional<torch::Tensor>& block_output, int64_t num_blocks) {
  if (!on_device(
          {&input, &norm_weight, &qk_weight, &prefix_output, &output, &blocks}))
    return Error::not_on_device;
  if (input.dim() != 2) return Error::shape_mismatch;
  const int64_t n = input.size(1);
  if (prefix_output.sizes() != input.sizes() ||
      output.sizes() != input.sizes() || blocks.dim() != 3 ||
      blocks.size(0) != input.size(0) || blocks.size(2) != n ||
      blocks.stride(2) != 1 || num_blocks < 0 || num_blocks > blocks.size(1) ||
      norm_weight.dim() != 1 || norm_weight.numel() != n ||
      qk_weight.dim() != 1 || qk_weight.numel() != n)
    return Error::shape_mismatch;
  std::vector<const torch::Tensor*> same = {&blocks, &norm_weight, &qk_weight,
                                            &prefix_output, &output};
  for (const std::optional<torch::Tensor>* o :
       {&prefix, &output_norm_weight, &block_output})
    if (*o) {
      if (!on_device({&**o})) return Error::not_on_device;
      same.push_back(&**o);
    }
  for (const torch::Tensor* t : same)
    if (t->scalar_type() != input.scalar_type()) return Error::dtype_mismatch;
  if ((prefix && prefix->sizes() != input.sizes()) ||
      (block_output && (block_output->sizes() != input.sizes() ||
                        block_output->stride(1) != 1)) ||
      (output_norm_weight &&
       (output_norm_weight->dim() != 1 || output_norm_weight->numel() != n)))
    return Error::shape_mismatch;
  // Its rows' and sources' strides are the API's to refuse (whole packs); its
  // start is here.
  if (reinterpret_cast<uintptr_t>(blocks.data_ptr()) %
          hip_custom_communicator::kPackBytes !=
      0)
    return Error::not_aligned;
  // The sources read are the first num_blocks; block_output may be a later slot
  // of the same stack.
  const torch::Tensor sources = blocks.narrow(1, 0, num_blocks);
  if (overlapping(
          {&input, prefix ? &*prefix : nullptr, &sources, &norm_weight,
           &qk_weight, output_norm_weight ? &*output_norm_weight : nullptr},
          {&prefix_output, &output, block_output ? &*block_output : nullptr}))
    return Error::inputs_overlap_outputs;
  return std::nullopt;
}

}  // namespace

// =============================================================================
// 3. THE API: a function a torch op, what Python passes in, its Error's number
//    (or none) out.
// =============================================================================

// =============================================================================
// THE STATE, AS A HANDLE: the CommsState, made by `open` and handed to Python
// as its address (an int, as vLLM's custom all-reduce hands its own), passed
// back to every call, and freed by `free`. Each returns its Error's number, or
// none.
// =============================================================================

// OPEN: the state of a communicator at `rank` of `world` on `device`, serving
// `max_rows` rows of at most `max_cols` columns, a wait giving up after
// `timeout_seconds`: (the Error it is refused for, or none; the handle, 0 where
// refused; this rank's regions' IPC handles, their bytes; the device's arch;
// the capacity: max_rows, max_cols, the partials' and staging's bytes, then
// each op's most bytes a call). A device the build does not run on is refused,
// nothing allocated: the caller does not run here.
OpenWire hip_custom_communicator_open(int64_t rank, int64_t world,
                                      int64_t device, int64_t max_rows,
                                      int64_t max_cols,
                                      double timeout_seconds) {
  using namespace hip_custom_communicator;
  // ---- PRECONDITIONS: the call's values, the world built, the device the
  // build's target with our kernels' code object on it (a kernel's attributes
  // answer exactly when it does). ----
  if (!all_ints({rank, world, device}))
    return {error_code(Error::field_out_of_range), 0, {}, "", {}};
  if (rank < 0 || rank >= world || !world_built(static_cast<int>(world)))
    return {error_code(Error::world_not_built), 0, {}, "", {}};
  if (max_rows < 1 || max_cols < 1)
    return {error_code(Error::field_not_positive), 0, {}, "", {}};
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, static_cast<int>(device)) != hipSuccess) {
    (void)hipGetLastError();
    return {error_code(Error::device_not_built), 0, {}, "", {}};
  }
  const std::string name = prop.gcnArchName;
  const std::string arch =
      name.substr(0, name.find(':'));  // gfx950:sramecc+:xnack-
  if (hipSetDevice(static_cast<int>(device)) != hipSuccess)
    return {error_code(Error::hip_failed), 0, {}, "", {}};
  // THE BUILD RUNS HERE where the device has our code object: a compiled
  // kernel's attributes answer exactly then.
  hipFuncAttributes attrs;
  const hipError_t found =
      hipFuncGetAttributes(&attrs, kAllReduceInstances[0].kernel);
  (void)hipGetLastError();
  if (found != hipSuccess)
    return {error_code(Error::device_not_built), 0, {}, "", {}};
  int khz = 0, lds = 0, cus = 0, threads_a_cu = 0, wave = 0;
  size_t scratch = 0;
  if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate,
                            static_cast<int>(device)) != hipSuccess ||
      hipDeviceGetAttribute(&wave, hipDeviceAttributeWarpSize,
                            static_cast<int>(device)) != hipSuccess ||
      hipDeviceGetAttribute(&lds, hipDeviceAttributeMaxSharedMemoryPerBlock,
                            static_cast<int>(device)) != hipSuccess ||
      hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount,
                            static_cast<int>(device)) != hipSuccess ||
      hipDeviceGetAttribute(&threads_a_cu,
                            hipDeviceAttributeMaxThreadsPerMultiProcessor,
                            static_cast<int>(device)) != hipSuccess ||
      hipDeviceGetLimit(&scratch, hipExtLimitScratchMax) != hipSuccess)
    return {error_code(Error::hip_failed), 0, {}, "", {}};

  // ---- THE EXCHANGE, SIZED FROM THE PROBLEM (STANDARDS, *A kernel's own
  // buffers are sized from the problem it serves*): each region holds its
  // largest payload at the largest call, in every built dtype and either
  // encoding (exact or INT4: common.cuh's INT4 payload layout): the staging the
  // whole input, the pushed two-shot's whole-call inbox, or their INT4
  // payloads; the partials the most any two-shot keeps (`partials_need`), or
  // its INT4 payload. Each grows with the rows and the row's width, so a call
  // inside the box fits by construction. Each a whole pack, at least one. ----
  const int64_t pack = kPackBytes;
  const auto whole = [pack](int64_t bytes) {
    return bytes < pack ? pack : (bytes + pack - 1) / pack * pack;
  };
  int64_t partials_bytes = 0,
          staging_bytes = int4_payload_bytes(max_rows, max_cols);
  for (const at::ScalarType d : kDtypes) {
    const int64_t packs = (max_cols * elem_bytes(d) + pack - 1) / pack;
    every_algorithm([&](auto a) {
      const int64_t need =
          partials_need(a, max_rows, packs, static_cast<int>(world), pack);
      if (need > partials_bytes) partials_bytes = need;
      const int64_t int4 = int4_payload_bytes(1, need / elem_bytes(d));
      if (need > 0 && int4 > partials_bytes) partials_bytes = int4;
    });
    staging_bytes =
        std::max(staging_bytes, max_rows * max_cols * elem_bytes(d));
    // THE EXACT PUSHED TWO-SHOT'S INBOX, the whole call: WORLD slots of a share
    // each.
    staging_bytes = std::max(
        staging_bytes, world * ((max_rows * packs + world - 1) / world) * pack);
  }
  // THE TWO-SHOTS' INT4 INBOX (the flat walk): WORLD shares, each rounded up to
  // whole groups.
  staging_bytes = std::max(
      staging_bytes,
      int4_payload_bytes(1, world * int4_flat_share(max_rows, max_cols,
                                                    static_cast<int>(world))));
  partials_bytes = whole(partials_bytes);
  staging_bytes = whole(staging_bytes);

  // ---- THE LAYOUT, decided here, the same on every rank. ----
  // THE SIGNALS: two flag sets (a barrier's parity) of a slot a block by a slot
  // a writer, ONE SLOT A BLOCK THE DEVICE CAN HOLD RESIDENT (every CU full of
  // one-wave blocks), so the signals never rule out a grid; whether one
  // kernel's grid is resident is its occupancy (grid_not_resident).
  const int64_t max_blocks = int64_t{cus} * (threads_a_cu / wave);
  const int64_t signals_bytes =
      2 * max_blocks * world * int64_t{sizeof(uint32_t)};
  // THE UNFENCED REGION: the pushed one-shot's inbox, written with the flag in
  // the line and fenced by no barrier, so its own region (a stale word a fenced
  // kernel left could carry an epoch's value): [writer, bank * row] rows, two
  // banks of `unfenced_blocks` rows, a grid past them refused; a row
  // `unfenced_slot_bytes` of payload, its lines twice that (a line's flag
  // beside its payload). 128 blocks of 2 KiB: the one-shot's grids at the sizes
  // it serves (a tile a block, rows of up to 8192 bf16 values a line pack). Its
  // epochs are rank-local: a block's a pack (its own line), in int32 words.
  const int64_t unfenced_blocks = 128, unfenced_slot_bytes = 2048;
  const int64_t unfenced_epoch_words = kPackBytes / int64_t{sizeof(int32_t)};
  const int64_t unfenced_bytes = 2 * unfenced_blocks * world *
                                 unfenced_slot_bytes *
                                 (kLineBytes / kLinePayloadBytes);
  // THE INPUT TABLES' SLOTS, fixed here (no allocation during a capture): 2^18,
  // 16 MiB at 8 ranks (2^14 was too few for Kimi-K3 at c48).
  const int64_t input_table_slots = int64_t{1} << 18;
  // THE SCRATCH A KERNEL MAY ASK: the device's scratch over its every lane, by
  // 16. A dispatch that asks too much fails only when it runs, aborting the
  //    process; the share is a margin for the queues a process holds (16 full
  //    dispatches; 4 KiB a lane at gfx950).
  const int64_t scratch_shares = 16;

  // ---- THE STATE. ----
  auto* s = new CommsState();
  s->rank = static_cast<int>(rank);
  s->world = static_cast<int>(world);
  s->device.index = static_cast<int>(device);
  s->device.arch = arch;
  s->device.wave_size = wave;
  s->device.clock_khz = khz;
  s->max_rows = max_rows;
  s->timeout_ticks =
      static_cast<uint64_t>(timeout_seconds * s->device.clock_khz * 1000);
  s->device.lds_bytes = lds;
  s->device.compute_units = cus;
  s->device.threads_a_cu = threads_a_cu;
  s->device.scratch_a_lane = static_cast<int64_t>(scratch) /
                             (int64_t{cus} * threads_a_cu) / scratch_shares;
  SymmetricMemory& symmetric = s->symmetric;
  symmetric.max_blocks = max_blocks;
  symmetric.unfenced_blocks = unfenced_blocks;
  symmetric.unfenced_slot_bytes = unfenced_slot_bytes;
  symmetric.signals_bytes = signals_bytes;
  symmetric.staging_bytes = staging_bytes;
  symmetric.partials_bytes = partials_bytes;
  symmetric.scales_bytes = max_rows * int64_t{sizeof(float)};
  symmetric.unfenced_bytes = unfenced_bytes;
  s->unfenced_epoch_words = unfenced_epoch_words;
  s->input_tables.slots = input_table_slots;

  // ---- THE BUFFERS, each its own allocation, zeroed before any stream can use
  // it; freed by `free`. THE SYMMETRIC MEMORY'S REGIONS are
  // hipExtMallocWithFlags(hipDeviceMallocUncached), exported with
  // hipIpcGetMemHandle, opened by each peer with hipIpcOpenMemHandle: the
  // convention of every IPC communicator on ROCm (vLLM's custom all-reduce and
  // quick reduce, aiter's). Correct cached or not under the barrier's memory
  // model (common.cuh); uncached keeps peer reads off every L2. Not VMM:
  // exchanging more than ~256 MiB of VMM allocations across 8 processes left
  // the driver in an eviction storm on ROCm 10.0. A region holds either
  // encoding, sized for the larger: a call publishes exact or INT4, never both,
  // and every reuse is ordered by a barrier the previous call's readers passed.
  // The rest is this rank's alone (hipMalloc). ----
  bool made = true;
  const auto allocate = [&](auto*& buffer, int64_t bytes, bool exchanged) {
    void* p = nullptr;
    made =
        made &&
        (exchanged ? hipExtMallocWithFlags(&p, static_cast<size_t>(bytes),
                                           hipDeviceMallocUncached)
                   : hipMalloc(&p, static_cast<size_t>(bytes))) == hipSuccess &&
        hipMemset(p, 0, static_cast<size_t>(bytes)) == hipSuccess;
    buffer = static_cast<std::remove_reference_t<decltype(buffer)>>(p);
  };
  const int64_t ptr_array_bytes = world * int64_t{sizeof(char*)};
  allocate(symmetric.signals, symmetric.signals_bytes, true);
  allocate(symmetric.staging, symmetric.staging_bytes, true);
  allocate(symmetric.partials, symmetric.partials_bytes, true);
  allocate(symmetric.scales, symmetric.scales_bytes, true);
  allocate(symmetric.unfenced, symmetric.unfenced_bytes, true);
  allocate(symmetric.signals_ptr_array, ptr_array_bytes, false);
  allocate(symmetric.staging_ptr_array, ptr_array_bytes, false);
  allocate(symmetric.partials_ptr_array, ptr_array_bytes, false);
  allocate(symmetric.scales_ptr_array, ptr_array_bytes, false);
  allocate(symmetric.unfenced_ptr_array, ptr_array_bytes, false);
  allocate(s->unfenced_epochs,
           unfenced_blocks * unfenced_epoch_words * int64_t{sizeof(int32_t)},
           false);
  allocate(s->status_device, kStatusFields * int64_t{sizeof(uint32_t)}, false);
  allocate(s->input_tables.device,
           input_table_slots * world * int64_t{sizeof(void*)}, false);
  // THE REGIONS' IPC HANDLES, their bytes, in connect's order: the signals, the
  // staging, the partials, the scales, the unfenced region.
  std::vector<int64_t> handle;
  for (const void* region : {static_cast<const void*>(symmetric.signals),
                             static_cast<const void*>(symmetric.staging),
                             static_cast<const void*>(symmetric.partials),
                             static_cast<const void*>(symmetric.scales),
                             static_cast<const void*>(symmetric.unfenced)}) {
    hipIpcMemHandle_t h{};
    made =
        made && hipIpcGetMemHandle(&h, const_cast<void*>(region)) == hipSuccess;
    for (size_t i = 0; i < sizeof(h); ++i)
      handle.push_back(
          static_cast<uint8_t>(reinterpret_cast<const char*>(&h)[i]));
  }
  if (!made || hipDeviceSynchronize() != hipSuccess) {
    (void)hipGetLastError();
    (void)hip_custom_communicator_free(reinterpret_cast<int64_t>(s));
    return {error_code(Error::hip_failed), 0, {}, "", {}};
  }

  // ---- THE ANSWER: the handle, the regions' IPC handles' bytes, the arch, the
  // capacity: the box, the regions, and each op's most bytes a call (its
  // partials a call's bytes at most, its staging the whole call). ----
  std::vector<int64_t> capacity{max_rows, max_cols, partials_bytes,
                                staging_bytes};
  // EACH OP'S MOST BYTES A CALL, in Python's Op order: an all-reduce's the most
  // its partials hold at its algorithms' widest keep a byte, at most its
  // staging; a norm on rows', which keeps nothing, its staging.
  const auto most_bytes = [&](const auto& algorithms) {
    int64_t per_byte = 0;
    for (const auto& a : algorithms)
      per_byte = std::max(
          per_byte, partials_need(a.algorithm, 1, 1, static_cast<int>(world),
                                  kPackBytes) /
                        kPackBytes);
    return per_byte == 0 ? staging_bytes
                         : std::min(partials_bytes / per_byte, staging_bytes);
  };
  capacity.push_back(most_bytes(AllReduceBuildConfig::algorithms));
  capacity.push_back(most_bytes(AllReduceRmsNormBuildConfig::algorithms));
  capacity.push_back(most_bytes(AllReduceAddRmsNormBuildConfig::algorithms));
  capacity.push_back(
      most_bytes(AllReduceAddAttnResRmsNormBuildConfig::algorithms));
  capacity.push_back(staging_bytes);  // rms_norm
  capacity.push_back(staging_bytes);  // add_rms_norm
  capacity.push_back(staging_bytes);  // add_attn_res_rms_norm
  return {std::nullopt, reinterpret_cast<int64_t>(s), handle, s->device.arch,
          capacity};
}

// CONNECT: the symmetric memory joined: every peer's regions opened from their
// IPC handles (`handles`: a rank's five, as open answered them, rank after
// rank; this rank's ignored), then each region's device array written (every
// rank's address of it), and the eager input table, slot 0: every rank's
// staging.
ErrorCode hip_custom_communicator_connect(int64_t state,
                                          const std::vector<int64_t>& handles) {
  using namespace hip_custom_communicator;
  if (state == 0) return error_code(Error::not_open);
  CommsState& s = *reinterpret_cast<CommsState*>(state);
  constexpr int kRegions = 5;  // open's handles: signals, staging, partials,
                               // scales, unfenced
  constexpr size_t kHandleBytes = sizeof(hipIpcMemHandle_t);
  if (handles.size() != s.world * kRegions * kHandleBytes)
    return error_code(Error::world_not_built);
  SymmetricMemory& symmetric = s.symmetric;
  char* const own[kRegions] = {reinterpret_cast<char*>(symmetric.signals),
                               symmetric.staging, symmetric.partials,
                               reinterpret_cast<char*>(symmetric.scales),
                               symmetric.unfenced};
  const void* const arrays[kRegions] = {
      symmetric.signals_ptr_array, symmetric.staging_ptr_array,
      symmetric.partials_ptr_array, symmetric.scales_ptr_array,
      symmetric.unfenced_ptr_array};
  std::vector<char*> staging(s.world);
  for (int region = 0; region < kRegions; ++region) {
    std::vector<char*> addresses(s.world);
    for (int r = 0; r < s.world; ++r) {
      if (r == s.rank) {
        addresses[r] = own[region];
        continue;
      }
      hipIpcMemHandle_t h;
      for (size_t i = 0; i < kHandleBytes; ++i)
        reinterpret_cast<char*>(&h)[i] = static_cast<char>(
            handles[(r * kRegions + region) * kHandleBytes + i]);
      void* base = nullptr;
      if (hipIpcOpenMemHandle(&base, h, hipIpcMemLazyEnablePeerAccess) !=
          hipSuccess) {
        (void)hipGetLastError();
        return error_code(Error::hip_failed);
      }
      addresses[r] = static_cast<char*>(base);
      symmetric.peers.push_back(addresses[r]);
    }
    if (hipMemcpy(const_cast<void*>(arrays[region]), addresses.data(),
                  s.world * sizeof(char*),
                  hipMemcpyHostToDevice) != hipSuccess) {
      (void)hipGetLastError();
      return error_code(Error::hip_failed);
    }
    if (region == 1) staging = addresses;
  }
  if (hipMemcpy(s.input_tables.device, staging.data(), s.world * sizeof(char*),
                hipMemcpyHostToDevice) != hipSuccess) {
    (void)hipGetLastError();
    return error_code(Error::hip_failed);
  }
  return std::nullopt;
}

// THE CAPTURED LAUNCHES' INPUTS FOR THE PEERS (vLLM's
// get_graph_buffer_ipc_meta): (the Error or none; each pending input's
// allocation's IPC handle's bytes, an input after another, in the order
// reserved; each input's offset in its allocation).
GraphMetaWire hip_custom_communicator_graph_ipc_meta(int64_t state) {
  using namespace hip_custom_communicator;
  if (state == 0) return {error_code(Error::not_open), {}, {}};
  const CommsState& s = *reinterpret_cast<const CommsState*>(state);
  std::vector<int64_t> bytes, offsets;
  for (const auto& [slot, input] : s.input_tables.pending) {
    void* base = nullptr;
    hipIpcMemHandle_t h;
    if (hipPointerGetAttribute(&base, HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                               reinterpret_cast<hipDeviceptr_t>(
                                   const_cast<void*>(input))) != hipSuccess ||
        hipIpcGetMemHandle(&h, base) != hipSuccess) {
      (void)hipGetLastError();
      return {error_code(Error::hip_failed), {}, {}};
    }
    for (size_t i = 0; i < sizeof(h); ++i)
      bytes.push_back(
          static_cast<uint8_t>(reinterpret_cast<const char*>(&h)[i]));
    offsets.push_back(static_cast<const char*>(input) -
                      static_cast<const char*>(base));
  }
  return {std::nullopt, bytes, offsets};
}

// THE CAPTURED LAUNCHES' TABLES WRITTEN (vLLM's register_graph_buffers): every
// rank's `graph_ipc_meta` (`handles`' bytes and `offsets`, a rank's after
// another's, each as many as this rank has pending), each peer allocation
// opened once (cached by its handle), every pending slot written, none pending
// after. Ranks that captured different counts are refused (ranks_disagree).
ErrorCode hip_custom_communicator_register_graph_buffers(
    int64_t state, const std::vector<int64_t>& handles,
    const std::vector<int64_t>& offsets) {
  using namespace hip_custom_communicator;
  if (state == 0) return error_code(Error::not_open);
  CommsState& s = *reinterpret_cast<CommsState*>(state);
  const size_t count = s.input_tables.pending.size();
  if (offsets.size() != count * s.world ||
      handles.size() != offsets.size() * sizeof(hipIpcMemHandle_t))
    return error_code(Error::ranks_disagree);
  for (size_t i = 0; i < count; ++i) {
    const auto& [slot, input] = s.input_tables.pending[i];
    std::vector<const void*> table(s.world, nullptr);
    for (int r = 0; r < s.world; ++r) {
      if (r == s.rank) {
        table[r] = input;
        continue;
      }
      const size_t at = r * count + i;
      std::vector<char> key(
          handles.begin() + at * sizeof(hipIpcMemHandle_t),
          handles.begin() + (at + 1) * sizeof(hipIpcMemHandle_t));
      auto it = s.input_tables.opened.find(key);
      if (it == s.input_tables.opened.end()) {
        hipIpcMemHandle_t h;
        std::memcpy(&h, key.data(), sizeof(h));
        void* base = nullptr;
        if (hipIpcOpenMemHandle(&base, h, hipIpcMemLazyEnablePeerAccess) !=
            hipSuccess) {
          (void)hipGetLastError();
          return error_code(Error::hip_failed);
        }
        it = s.input_tables.opened.emplace(key, static_cast<char*>(base)).first;
      }
      table[r] = it->second + offsets[at];
    }
    if (hipMemcpy(s.input_tables.device + slot * s.world, table.data(),
                  table.size() * sizeof(void*),
                  hipMemcpyHostToDevice) != hipSuccess) {
      (void)hipGetLastError();
      return error_code(Error::hip_failed);
    }
  }
  s.input_tables.pending.clear();
  return std::nullopt;
}

// CLOSE THE PEERS: every BORROWED mapping closed (the peers' regions, their
// captured inputs), every one tried: the first failure's Error. Every launch
// finished first, and every rank stopped (close is a collective).
ErrorCode hip_custom_communicator_close_peers(int64_t state) {
  using namespace hip_custom_communicator;
  if (state == 0) return error_code(Error::not_open);
  CommsState& s = *reinterpret_cast<CommsState*>(state);
  std::optional<Error> first;
  std::vector<char*> borrowed = s.symmetric.peers;
  for (const auto& [key, base] : s.input_tables.opened)
    borrowed.push_back(base);
  for (char* base : borrowed)
    if (hipIpcCloseMemHandle(base) != hipSuccess) {
      (void)hipGetLastError();
      first = first.value_or(Error::hip_failed);
    }
  s.symmetric.peers.clear();
  s.input_tables.opened.clear();
  return error_code(first);
}

// FREE: this rank's OWNED buffers freed, every one tried (the first failure's
// Error), and the state with them. Once no peer has them open (after every
// rank's close_peers).
ErrorCode hip_custom_communicator_free(int64_t state) {
  using namespace hip_custom_communicator;
  if (state == 0) return error_code(Error::not_open);
  auto* s = reinterpret_cast<CommsState*>(state);
  std::optional<Error> first;
  for (const void* p :
       {static_cast<const void*>(s->symmetric.signals),
        static_cast<const void*>(s->symmetric.staging),
        static_cast<const void*>(s->symmetric.partials),
        static_cast<const void*>(s->symmetric.scales),
        static_cast<const void*>(s->symmetric.unfenced),
        static_cast<const void*>(s->symmetric.signals_ptr_array),
        static_cast<const void*>(s->symmetric.staging_ptr_array),
        static_cast<const void*>(s->symmetric.partials_ptr_array),
        static_cast<const void*>(s->symmetric.scales_ptr_array),
        static_cast<const void*>(s->symmetric.unfenced_ptr_array),
        static_cast<const void*>(s->unfenced_epochs),
        static_cast<const void*>(s->status_device),
        static_cast<const void*>(s->input_tables.device)})
    if (p != nullptr && hipFree(const_cast<void*>(p)) != hipSuccess) {
      (void)hipGetLastError();
      first = first.value_or(Error::hip_failed);
    }
  delete s;
  return error_code(first);
}

// all_reduce: at its config. One function: what Python passes, its tensors
// stripped to pointers and strides, the preconditions, the kernel's arguments,
// the launch, the postcondition.
ErrorCode hip_custom_communicator_all_reduce(
    int64_t state, torch::Tensor& input, torch::Tensor& output,
    const std::string& algorithm_name, int64_t grid_x, int64_t grid_y,
    int64_t grid_z, int64_t block_x, int64_t block_y, int64_t block_z,
    int64_t waves_per_eu, int64_t tile_m, int64_t tile_n,
    const std::string& quantization_name, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  using namespace hip_custom_communicator;
  // ---- THE CALL FROM TORCH: its state, its tensors stripped to pointers and
  // strides, its config's ints. ----
  if (state == 0) return error_code(Error::not_open);
  CommsState& s = *reinterpret_cast<CommsState*>(state);
  if (!input.is_cuda() || !output.is_cuda())
    return error_code(Error::not_on_device);
  if (input.dim() != 2 || output.sizes() != input.sizes())
    return error_code(Error::shape_mismatch);
  if (output.scalar_type() != input.scalar_type())
    return error_code(Error::dtype_mismatch);
  for (const int64_t v : {waves_per_eu, tile_m, tile_n, pack_bytes, grid_x,
                          grid_y, grid_z, block_x, block_y, block_z})
    if (v < std::numeric_limits<int>::min() ||
        v > std::numeric_limits<int>::max())
      return error_code(Error::field_out_of_range);
  const void* input_ptr = input.data_ptr();
  const int64_t input_stride_m = input.stride(0),
                input_stride_n = input.stride(1);
  const int64_t input_size_m = input.size(0), input_size_n = input.size(1);
  void* output_ptr = output.data_ptr();
  const int64_t output_stride_m = output.stride(0),
                output_stride_n = output.stride(1);
  const int64_t output_size_m = output.size(0), output_size_n = output.size(1);
  const at::ScalarType dtype = input.scalar_type();
  const int64_t e = input.element_size();
  // THE PROBLEM'S SHAPE: the input's.
  const int64_t m = input_size_m, n = input_size_n;
  const Dim3 grid{static_cast<int>(grid_x), static_cast<int>(grid_y),
                  static_cast<int>(grid_z)};
  const Dim3 block{static_cast<int>(block_x), static_cast<int>(block_y),
                   static_cast<int>(block_z)};
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  // ---- PRECONDITIONS, every Error the call can be refused for, in this order:
  // the config alone; the call against the config; the regions; the compiled
  // kernel on this device. Past this section the call runs. ----
  // THE CONFIG'S NAMES.
  AllReduceAlgorithm algorithm;
  if (algorithm_name == "one_shot_pull")
    algorithm = AllReduceAlgorithm::one_shot_pull;
  else if (algorithm_name == "one_shot_push")
    algorithm = AllReduceAlgorithm::one_shot_push;
  else if (algorithm_name == "one_shot_push_staged")
    algorithm = AllReduceAlgorithm::one_shot_push_staged;
  else if (algorithm_name == "two_shot_pull")
    algorithm = AllReduceAlgorithm::two_shot_pull;
  else if (algorithm_name == "two_shot_push")
    algorithm = AllReduceAlgorithm::two_shot_push;
  else
    return error_code(Error::no_such_kernel);
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_code(Error::no_such_quantization);
  const bool int4_config = quantization_name == "int4";
  // THE GRID: x alone (the kernel's one dimension), at least a block, at most
  // the blocks the signals have slots for. The block and the waves are the
  // build's (not_built, below).
  if (grid_y != 1 || grid_z != 1 || grid_x < 1 ||
      grid_x > s.symmetric.max_blocks)
    return error_code(Error::launch_out_of_range);
  if (tile_m < 1 || tile_n < 1) return error_code(Error::field_not_positive);
  // QUANTIZE_FROM_BYTES: an INT4 config's alone, and not negative.
  if (!int4_config && quantize_from_bytes)
    return error_code(Error::field_not_this_templates);
  const int64_t quantize_from = quantize_from_bytes.value_or(0);
  if (quantize_from < 0) return error_code(Error::field_not_positive);
  // A PACK A LOAD THE HARDWARE HAS (2, 4, 8 or 16 bytes); INT4 at the widest
  // (the format's word is a pack of 8 values: common.cuh's QUANTIZATION).
  if (pack_bytes < 2 || pack_bytes > kPackBytes ||
      (pack_bytes & (pack_bytes - 1)) != 0)
    return error_code(Error::pack_out_of_range);
  if (int4_config && pack_bytes != kPackBytes)
    return error_code(Error::quantization_not_supported);
  // THE CALL'S DTYPE BUILT (the world is: open refuses the rest), and its
  // elements whole packs: at least an element a pack, whole packs a row (an
  // aligned load of the pack's width).
  if (std::find(kDtypes.begin(), kDtypes.end(), dtype) == kDtypes.end())
    return error_code(Error::dtype_not_built);
  if (pack_bytes < e) return error_code(Error::pack_out_of_range);
  if (n * e % pack_bytes != 0) return error_code(Error::row_not_packs);
  // A LAYOUT THE PACKED LOADS CAN READ: the innermost stride 1, every row on
  // whole packs.
  if (input_stride_n != 1 || output_stride_n != 1)
    return error_code(Error::inner_stride_not_one);
  if (input_stride_m * e % pack_bytes != 0 ||
      output_stride_m * e % pack_bytes != 0)
    return error_code(Error::row_stride_not_packs);
  // THE OUTPUT APART FROM THE INPUT: their bytes do not meet (an output that
  // aliases the input is refused, interleaved rows included).
  if (m > 0 && n > 0) {
    const char* input_first = static_cast<const char*>(input_ptr);
    const char* output_first = static_cast<const char*>(output_ptr);
    const char* input_last =
        input_first +
        ((m - 1) * input_stride_m + (n - 1) * input_stride_n + 1) * e;
    const char* output_last =
        output_first +
        ((m - 1) * output_stride_m + (n - 1) * output_stride_n + 1) * e;
    if (input_first < output_last && output_first < input_last)
      return error_code(Error::inputs_overlap_outputs);
  }
  // QUANTIZE_INT4, WHAT THIS LAUNCH MOVES: INT4 where the config's quantization
  // is int4 and the call's bytes reach its quantize_from_bytes, else exact (the
  // kernel's one template parameter).
  const bool quantize_int4 = int4_config && m * n * e >= quantize_from;
  // IN PLACE: the exact pull, every rank's input read where its table says.
  const bool in_place = (algorithm == AllReduceAlgorithm::one_shot_pull ||
                         algorithm == AllReduceAlgorithm::two_shot_pull) &&
                        !quantize_int4;
  const int shared_memory_bytes = 0;
  const bool contiguous = input_stride_m == n * input_stride_n &&
                          output_stride_m == n * output_stride_n;
  // AN INT4 ROW WITH A TAIL runs the build's TAIL instance (int4_tail).
  const bool tail = quantize_int4 && int4_tail(algorithm, m, n, s.world);
  // THE PUSHED ONE-SHOT'S UNFENCED REGION HOLDS ITS TILES: a row of a block's
  // tile in a slot's lines, a line a pack of at most kLinePayloadBytes (a
  // narrower pack its line alone: kernels/all_reduce.cuh), INT4 a line a word
  // of its tile's whole groups; a block's TILE_M rows of a bank's
  // unfenced_blocks, a block its own slots (a grid past them is refused, never
  // cut).
  if (algorithm == AllReduceAlgorithm::one_shot_push) {
    const int64_t line_pack = std::min<int64_t>(pack_bytes, kLinePayloadBytes);
    const int64_t row_lines =
        quantize_int4 ? (tile_n + kInt4Group - 1) / kInt4Group * kInt4GroupPacks
                      : (tile_n * e + line_pack - 1) / line_pack;
    if (row_lines > s.symmetric.unfenced_slot_bytes / kLinePayloadBytes ||
        grid_x * tile_m > s.symmetric.unfenced_blocks)
      return error_code(Error::unfenced_too_small);
  }
  void* kernel = nullptr;
  // THE INSTANCE: the kernel compiled at this config, on this device, for this
  // call (builds.py's kAllReduceInstances, a row each); a config the table did
  // not build is refused here.
  for (const auto& i : kAllReduceInstances)
    if (i.algorithm == algorithm && i.block_x == block_x &&
        i.block_y == block_y && i.block_z == block_z &&
        i.waves_per_eu == waves_per_eu && i.tile_m == tile_m &&
        i.tile_n == tile_n &&
        i.quantization ==
            (int4_config ? Quantization::int4 : Quantization::exact) &&
        i.pack_bytes == pack_bytes && i.world == s.world &&
        i.wave_size == s.device.wave_size && i.dtype == dtype &&
        i.contiguous == contiguous && i.tail == tail &&
        i.quantize_int4 == quantize_int4)
      kernel = i.kernel;
  if (kernel == nullptr) return error_code(Error::not_built);
  // THE KERNEL ON THIS DEVICE, as its compiler left it. HIP refuses the block
  // and the LDS at the launch; the scratch only when the dispatch runs,
  // aborting the queue and the process with it (the tune's crashes, 2026-10-24:
  // AttnRes at 64 threads, tile 4x16384x4, 21 KiB a lane).
  hipFuncAttributes attributes;
  if (hipFuncGetAttributes(&attributes, kernel) != hipSuccess)
    return error_code(Error::hip_failed);
  if (block_x > attributes.maxThreadsPerBlock)
    return error_code(Error::block_too_large);
  if (static_cast<int64_t>(attributes.sharedSizeBytes) + shared_memory_bytes >
      s.device.lds_bytes)
    return error_code(Error::lds_too_large);
  if (static_cast<int64_t>(attributes.localSizeBytes) > s.device.scratch_a_lane)
    return error_code(Error::scratch_too_large);
  // THE GRID RESIDENT AT ONCE: a peer barrier spins until the same block on
  // every peer arrives, so a grid past the blocks the device holds at once can
  // hang (HIP never refuses it). The blocks a CU holds from the kernel's
  // registers, LDS and the CU's wave slots, times the CUs.
  int blocks_a_cu = 0;
  if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_a_cu, kernel, static_cast<int>(block_x),
          shared_memory_bytes) != hipSuccess)
    return error_code(Error::hip_failed);
  if (grid_x > blocks_a_cu * s.device.compute_units)
    return error_code(Error::grid_not_resident);
  // ---- ITS VIEWS OF THE REGIONS, by its algorithm, in elements: a view [m, n]
  // at strides [stride_m, stride_n] (the line push's inbox [slot, m, n]); an
  // empty view (size 0) where it reads none. The kernels find a one-shot's
  // payload in the staging from the input's sizes.
  // THE WHOLE CALL IN ONE PASS: the regions are sized from the largest call the
  // server makes, so a call past them is refused, never cut into bands. Every
  // exact kernel's input fits the staging whole (an eager call's host copy,
  // INPUT TABLES; a captured call's is read where it is, refused alike, so the
  // eager and captured calls of one shape agree). ----
  // The two-shots' exact share is the flat walk's, whole packs, rounded up: a
  // rank's slice of a row in the partials, and the pushed two-shot's inbox
  // [WORLD, slot] in the staging, its slot a share of the whole call.
  const int64_t packs = n * e / pack_bytes;
  const int64_t slice_bytes = (packs + s.world - 1) / s.world * pack_bytes;
  const int64_t slot = (m * packs + s.world - 1) / s.world * pack_bytes / e;
  const int64_t share =
      int4_flat_share(m, n, s.world);  // the INT4 flat walk's share
  int64_t staging_need = quantize_int4 ? 0 : m * n * e, partials_need = 0;
  int64_t partials_stride_m = slice_bytes / e, partials_stride_n = 1,
          partials_size_m = 0, partials_size_n = slice_bytes / e;
  int64_t staging_stride_m = 0, staging_stride_n = 0, staging_size_m = 0,
          staging_size_n = 0;
  int64_t unfenced_stride_slot = 0, unfenced_stride_m = 0,
          unfenced_stride_n = 0, unfenced_size_slot = 0, unfenced_size_m = 0,
          unfenced_size_n = 0;
  switch (algorithm) {
    case AllReduceAlgorithm::two_shot_pull:
    case AllReduceAlgorithm::two_shot_push:
      staging_need = m * n * e, partials_need = m * slice_bytes;
      if (quantize_int4) {
        // THE INT4 FLAT WALK: this rank's share of the call's groups in the
        // partials, and every rank's share in the staging.
        partials_stride_m = share, partials_size_m = 1, partials_size_n = share;
        staging_stride_m = s.world * share, staging_stride_n = 1,
        staging_size_m = 1, staging_size_n = s.world * share;
        staging_need =
            std::max(staging_need, int4_payload_bytes(1, s.world * share));
        partials_need = std::max(partials_need, int4_payload_bytes(1, share));
      } else {
        // EXACT: this rank's slice of every row in the partials; the pushed
        // two-shot's inbox in the staging, WORLD slots of a share of the call
        // (the pull reads every rank's input in place, through the input
        // table).
        partials_size_m = m;
        if (algorithm == AllReduceAlgorithm::two_shot_push) {
          staging_stride_m = slot, staging_stride_n = 1,
          staging_size_m = s.world, staging_size_n = slot;
          staging_need = std::max(staging_need, s.world * slot * e);
        }
      }
      break;
    case AllReduceAlgorithm::one_shot_pull:
      // INT4: its [m, n] payload in the staging, no partials (exact: every
      // rank's input in place).
      if (quantize_int4) {
        partials_stride_m = 0, partials_size_n = 0;
        staging_stride_n = 1, staging_size_m = 1,
        staging_size_n = (int4_payload_bytes(m, n) + e - 1) / e;
        staging_need = int4_payload_bytes(m, n);
      }
      break;
    case AllReduceAlgorithm::one_shot_push:
      // THE UNFENCED REGION: [writer, row, n] payload elements, 2 *
      // unfenced_blocks rows (bank, row) of unfenced_slot_bytes of payload each
      // whatever the build (so a row is the same bytes at every build, its
      // epoch its own), a writer's rows contiguous.
      unfenced_stride_slot = 2 * s.symmetric.unfenced_blocks *
                             (s.symmetric.unfenced_slot_bytes / e);
      unfenced_stride_m = s.symmetric.unfenced_slot_bytes / e,
      unfenced_stride_n = 1;
      unfenced_size_slot = s.world,
      unfenced_size_m = 2 * s.symmetric.unfenced_blocks,
      unfenced_size_n = s.symmetric.unfenced_slot_bytes / e;
      break;
    case AllReduceAlgorithm::one_shot_push_staged: {
      // WORLD SLOTS OF AN INPUT in the staging (INT4: its payload, a slot whole
      // kInt4ScalesAlign), no partials.
      const int64_t payload =
          quantize_int4 ? int4_payload_bytes(m, n) : m * n * e;
      const int64_t slot_bytes = quantize_int4
                                     ? (payload + kInt4ScalesAlign - 1) /
                                           kInt4ScalesAlign * kInt4ScalesAlign
                                     : payload;
      partials_stride_m = 0, partials_size_n = 0;
      staging_stride_n = 1, staging_size_m = 1,
      staging_size_n = (s.world * slot_bytes + e - 1) / e;
      staging_need = s.world * slot_bytes;
      break;
    }
  }
  // THE REGIONS HOLD THE CALL: what the kernel stages and keeps, and each
  // view's end, its last element's (an empty view reads nothing). The kernels
  // check their indices only against the sizes handed them (October 26: n11
  // rebooted seven times on views past their regions). The unfenced views are
  // that region's whole layout (open's): they fit by construction.
  const int64_t staging_end =
      staging_size_m == 0 || staging_size_n == 0
          ? 0
          : ((staging_size_m - 1) * staging_stride_m +
             (staging_size_n - 1) * staging_stride_n + 1) *
                e;
  const int64_t partials_end =
      partials_size_m == 0 || partials_size_n == 0
          ? 0
          : ((partials_size_m - 1) * partials_stride_m +
             (partials_size_n - 1) * partials_stride_n + 1) *
                e;
  if (std::max(staging_need, staging_end) > s.symmetric.staging_bytes)
    return error_code(Error::staging_too_small);
  if (std::max(partials_need, partials_end) > s.symmetric.partials_bytes)
    return error_code(Error::partials_too_small);
  // EVERY RANK'S INPUT, where the kernel reads it: in place through an input
  // table (the exact pull: `world` addresses), else in every rank's staging (no
  // table, the staging's array); its strides either way. Captured: the next
  // table slot, reserved for this input (register_graph_buffers fills it with
  // every rank's address when the capture exits), its rows at the input's
  // stride; nothing is copied. Eager: the input copied into this rank's staging
  // (it fits: the staging_need above), read through the eager table, slot 0,
  // its rows n apart.
  void* const* peer_inputs = nullptr;
  int peer_inputs_size = 0;
  int64_t peer_inputs_stride_m = staging_stride_m,
          peer_inputs_stride_n = staging_stride_n;
  if (in_place) {
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(stream, &capture) != hipSuccess)
      return error_code(Error::hip_failed);
    InputTables& tables = s.input_tables;
    if (capture == hipStreamCaptureStatusActive) {
      if (tables.next_slot >= tables.slots)
        return error_code(Error::tables_full);
      tables.pending.emplace_back(tables.next_slot, input_ptr);
      peer_inputs = tables.device + tables.next_slot * s.world;
      peer_inputs_stride_m = input_stride_m;
      ++tables.next_slot;
    } else {
      if (hipMemcpy2DAsync(s.symmetric.staging, static_cast<size_t>(n * e),
                           input_ptr, static_cast<size_t>(input_stride_m * e),
                           static_cast<size_t>(n * e), static_cast<size_t>(m),
                           hipMemcpyDeviceToDevice, stream) != hipSuccess)
        return error_code(Error::hip_failed);
      peer_inputs = tables.device;
      peer_inputs_stride_m = n;
    }
    peer_inputs_size = s.world, peer_inputs_stride_n = 1;
  }
  // THE LAUNCH: the instance's kernel, at the call's dtypes.
  const auto launch = [&]<typename T, typename W>() {
    const auto typed = reinterpret_cast<AllReduceKernel<T, W>>(kernel);
    typed<<<dim3_of(grid), dim3_of(block), shared_memory_bytes, stream>>>(
        static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
        static_cast<int>(input_size_m), static_cast<int>(input_size_n),
        reinterpret_cast<const T* const*>(peer_inputs), peer_inputs_size,
        peer_inputs_stride_m, peer_inputs_stride_n, static_cast<T*>(output_ptr),
        output_stride_m, output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), s.symmetric.partials_ptr_array,
        s.world, partials_stride_m, partials_stride_n,
        static_cast<int>(partials_size_m), static_cast<int>(partials_size_n),
        s.symmetric.staging_ptr_array, s.world, staging_stride_m,
        staging_stride_n, static_cast<int>(staging_size_m),
        static_cast<int>(staging_size_n), s.symmetric.unfenced_ptr_array,
        s.world, unfenced_stride_slot, unfenced_stride_m, unfenced_stride_n,
        static_cast<int>(unfenced_size_slot), static_cast<int>(unfenced_size_m),
        static_cast<int>(unfenced_size_n), s.unfenced_epochs,
        s.unfenced_epoch_words, int64_t{1},
        static_cast<int>(s.symmetric.unfenced_blocks),
        static_cast<int>(s.unfenced_epoch_words), s.symmetric.signals_ptr_array,
        s.world, s.symmetric.max_blocks * s.world, int64_t{s.world}, int64_t{1},
        int64_t{2}, s.symmetric.max_blocks, int64_t{s.world}, s.status_device,
        int64_t{1}, int64_t{kStatusFields}, s.rank, s.timeout_ticks);
  };
  if (dtype == at::ScalarType::BFloat16)
    launch.template operator()<c10::BFloat16, c10::BFloat16>();
  else
    launch.template operator()<c10::Half, c10::Half>();
  // POSTCONDITION: the launch went in.
  if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
  return std::nullopt;
}

// all_reduce_rms_norm: at its config. One function: what Python passes, its
// tensors stripped to pointers and strides, the preconditions, the kernel's
// arguments, the launch, the postcondition.
ErrorCode hip_custom_communicator_all_reduce_rms_norm(
    int64_t state, torch::Tensor& input, torch::Tensor& weight,
    torch::Tensor& output, double eps, const std::string& algorithm_name,
    int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t block_x,
    int64_t block_y, int64_t block_z, int64_t waves_per_eu, int64_t tile_m,
    int64_t tile_n, const std::string& quantization_name, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  using namespace hip_custom_communicator;
  // THE CALL FROM TORCH: its state, its tensors, its config's ints.
  if (state == 0) return error_code(Error::not_open);
  if (const std::optional<Error> e = norm_tensors(
          input, nullptr, weight, output, nullptr, /*in_place=*/false))
    return error_code(e);
  if (!all_ints({waves_per_eu, tile_m, tile_n, pack_bytes, grid_x, grid_y,
                 grid_z, block_x, block_y, block_z}))
    return error_code(Error::field_out_of_range);
  CommsState& s = *reinterpret_cast<CommsState*>(state);
  const void* input_ptr = input.data_ptr();
  const int64_t input_stride_m = input.stride(0);
  const int64_t input_stride_n = input.stride(1);
  const int64_t input_size_m = input.size(0);
  const int64_t input_size_n = input.size(1);
  const void* weight_ptr = weight.data_ptr();
  const int64_t weight_stride_n = weight.stride(0);
  const int64_t weight_size_n = weight.size(0);
  void* output_ptr = output.data_ptr();
  const int64_t output_stride_m = output.stride(0);
  const int64_t output_stride_n = output.stride(1);
  const int64_t output_size_m = output.size(0);
  const int64_t output_size_n = output.size(1);
  const at::ScalarType dtype = input.scalar_type();
  const at::ScalarType weight_dtype = weight.scalar_type();
  const Dim3 grid{static_cast<int>(grid_x), static_cast<int>(grid_y),
                  static_cast<int>(grid_z)};
  const Dim3 block{static_cast<int>(block_x), static_cast<int>(block_y),
                   static_cast<int>(block_z)};
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  // ---- PRECONDITIONS: the config, the call, then its instance. The first
  // Error refuses the call; past this section the call runs. ----
  const std::optional<AllReduceRmsNormAlgorithm> named =
      algorithm_named<AllReduceRmsNormBuildConfig>(algorithm_name);
  if (!named) return error_code(Error::no_such_kernel);
  const AllReduceRmsNormAlgorithm algorithm = *named;
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_code(Error::no_such_quantization);
  const AllReduceRmsNormBuildConfig build{
      .algorithm = algorithm,
      .block = block,
      .waves_per_eu = static_cast<int>(waves_per_eu),
      .tile_m = static_cast<int>(tile_m),
      .tile_n = static_cast<int>(tile_n),
      .quantization = quantization_name == "int4" ? Quantization::int4
                                                  : Quantization::exact,
      .pack_bytes = static_cast<int>(pack_bytes)};
  const int64_t m = input_size_m,
                n = input_size_n;  // the problem's shape: the input's
  if (const std::optional<Error> e =
          refused(true, s.world, s.symmetric.staging_bytes,
                  s.symmetric.partials_bytes, s.symmetric.max_blocks, build,
                  grid, quantize_from_bytes, dtype, m, n))
    return error_code(e);
  // A LAYOUT THE PACKED LOADS CAN READ: every buffer's innermost stride 1, its
  // rows on whole packs of the build's.
  if (output_stride_n != 1 || input_stride_n != 1)
    return error_code(Error::inner_stride_not_one);
  if (output_stride_m * elem_bytes(dtype) % build.pack_bytes != 0 ||
      input_stride_m * elem_bytes(dtype) % build.pack_bytes != 0)
    return error_code(Error::row_stride_not_packs);
  if (weight_stride_n != 1) return error_code(Error::inner_stride_not_one);
  // THE PARTIALS'S ROWS, what partials_need sized it for (a one-shot keeps
  // none).
  const int64_t partials_rows =
      n > 0 ? partials_need(algorithm, m,
                            n * elem_bytes(dtype) / build.pack_bytes, s.world,
                            build.pack_bytes) /
                  (n * elem_bytes(dtype))
            : 0;
  const int shared_memory_bytes = 0;
  // A ROW WIDER THAN THE TILE runs the build's CHUNKED instance.
  const bool chunked = n > build.tile_n;
  // AN INT4 ROW WITH A TAIL runs the build's TAIL instance (int4_tail).
  const bool tail = build.quantization == Quantization::int4 &&
                    int4_tail(algorithm, m, n, s.world);
  void* kernel = nullptr;
  // THE INSTANCE: the kernel compiled at this build, on this device, for this
  // call (builds.py's kAllReduceRmsNormInstances, a row each).
  for (const auto& i : kAllReduceRmsNormInstances)
    if (i.algorithm == build.algorithm && i.block_x == build.block.x &&
        i.block_y == build.block.y && i.block_z == build.block.z &&
        i.waves_per_eu == build.waves_per_eu && i.tile_m == build.tile_m &&
        i.tile_n == build.tile_n && i.tile_k == 1 &&
        i.quantization == build.quantization &&
        i.pack_bytes == build.pack_bytes && i.world == s.world &&
        i.wave_size == s.device.wave_size && i.dtype == dtype &&
        i.weight_dtype == weight_dtype && i.contiguous == false &&
        i.chunked == chunked && i.tail == tail &&
        i.quantize_int4 == (build.quantization == Quantization::int4 &&
                            build.pack_bytes == kPackBytes))
      kernel = i.kernel;
  if (kernel == nullptr) return error_code(Error::not_built);
  if (const std::optional<Error> e =
          check_kernel(s, kernel, grid, block, shared_memory_bytes, true))
    return error_code(e);

  // ---- THE LAUNCH: the buffers, the views of their regions, and the instance.
  // ----
  // ITS VIEWS OF THE REGIONS, each [m, n] (the pushed one-shot's inbox [slot,
  // m, n]), in elements: zeros where it reads none.
  int64_t partials_stride_m = 0, partials_stride_n = 0, partials_size_m = 0,
          partials_size_n = 0, staging_stride_m = 0, staging_stride_n = 0,
          staging_size_m = 0, staging_size_n = 0;
  bool in_place = false;
  const int64_t row_n = payload_row(algorithm, build.quantization, n);
  partials_stride_m = row_n;
  partials_stride_n = 1;
  partials_size_m = partials_rows;
  partials_size_n = row_n;
  staging_stride_m = row_n;
  staging_stride_n = 1;
  staging_size_m = staged_rows(algorithm, build.quantization, m, s.world);
  staging_size_n = row_n;
  in_place = !is_push(algorithm) && build.quantization == Quantization::exact;
  // EVERY VIEW INSIDE ITS REGION (view_fits).
  if (!view_fits(1, 0, partials_size_m, partials_stride_m, partials_size_n,
                 partials_stride_n, elem_bytes(dtype),
                 s.symmetric.partials_bytes) ||
      !view_fits(1, 0, staging_size_m, staging_stride_m, staging_size_n,
                 staging_stride_n, elem_bytes(dtype),
                 s.symmetric.staging_bytes))
    return error_code(Error::view_past_region);
  // THE INPUT TABLE (INPUT TABLES): where the kernel reads every rank's input
  // in place (an exact pull), or none where it reads the staging.
  void* const* table = nullptr;
  int64_t table_stride_m = staging_stride_m;
  if (in_place)
    if (const std::optional<Error> e = input_table(
            s, input_ptr, input_stride_m, input_size_m, input_size_n,
            elem_bytes(dtype), stream, table, table_stride_m))
      return error_code(e);
  const int table_size = in_place ? s.world : 0;
  const int64_t table_stride_n = in_place ? 1 : staging_stride_n;
  // THE LAUNCH: the instance's kernel, at the call's dtypes.
  const auto launch = [&]<typename T, typename W>() {
    const auto typed = reinterpret_cast<AllReduceRmsNormKernel<T, W>>(kernel);
    typed<<<dim3_of(grid), dim3_of(block), shared_memory_bytes, stream>>>(
        static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
        static_cast<int>(input_size_m), static_cast<int>(input_size_n),
        reinterpret_cast<const T* const*>(table), table_size, table_stride_m,
        table_stride_n, static_cast<const W*>(weight_ptr), weight_stride_n,
        static_cast<int>(weight_size_n), static_cast<T*>(output_ptr),
        output_stride_m, output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), s.symmetric.partials_ptr_array,
        s.world, partials_stride_m, partials_stride_n,
        static_cast<int>(partials_size_m), static_cast<int>(partials_size_n),
        s.symmetric.staging_ptr_array, s.world, staging_stride_m,
        staging_stride_n, static_cast<int>(staging_size_m),
        static_cast<int>(staging_size_n), s.symmetric.signals_ptr_array,
        s.world, s.symmetric.max_blocks * s.world, int64_t{s.world}, int64_t{1},
        int64_t{2}, s.symmetric.max_blocks, int64_t{s.world}, s.status_device,
        int64_t{1}, int64_t{kStatusFields}, s.rank, s.timeout_ticks, eps);
  };
  if (dtype == at::ScalarType::BFloat16 &&
      weight_dtype == at::ScalarType::Float)
    launch.template operator()<c10::BFloat16, float>();
  else if (dtype == at::ScalarType::BFloat16)
    launch.template operator()<c10::BFloat16, c10::BFloat16>();
  else if (weight_dtype == at::ScalarType::Float)
    launch.template operator()<c10::Half, float>();
  else
    launch.template operator()<c10::Half, c10::Half>();
  // POSTCONDITION: the launch went in.
  if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
  return std::nullopt;
}

// all_reduce_add_rms_norm: at its config. all_reduce_add_rms_norm: out,
// residual_out = fused_add_rms_norm(all_reduce(inp), residual, weight), vLLM's
// roundings exactly. One function: what Python passes, its tensors stripped to
// pointers and strides, the preconditions, the kernel's arguments, the launch,
// the postcondition.
ErrorCode hip_custom_communicator_all_reduce_add_rms_norm(
    int64_t state, torch::Tensor& input, torch::Tensor& residual,
    torch::Tensor& weight, torch::Tensor& output,
    torch::Tensor& residual_output, double eps,
    const std::string& algorithm_name, int64_t grid_x, int64_t grid_y,
    int64_t grid_z, int64_t block_x, int64_t block_y, int64_t block_z,
    int64_t waves_per_eu, int64_t tile_m, int64_t tile_n,
    const std::string& quantization_name, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  using namespace hip_custom_communicator;
  // THE CALL FROM TORCH: its state, its tensors, its config's ints.
  if (state == 0) return error_code(Error::not_open);
  if (const std::optional<Error> e =
          norm_tensors(input, &residual, weight, output, &residual_output,
                       /*in_place=*/false))
    return error_code(e);
  if (!all_ints({waves_per_eu, tile_m, tile_n, pack_bytes, grid_x, grid_y,
                 grid_z, block_x, block_y, block_z}))
    return error_code(Error::field_out_of_range);
  CommsState& s = *reinterpret_cast<CommsState*>(state);
  const void* input_ptr = input.data_ptr();
  const int64_t input_stride_m = input.stride(0);
  const int64_t input_stride_n = input.stride(1);
  const int64_t input_size_m = input.size(0);
  const int64_t input_size_n = input.size(1);
  const void* residual_ptr = residual.data_ptr();
  const int64_t residual_stride_m = residual.stride(0);
  const int64_t residual_stride_n = residual.stride(1);
  const int64_t residual_size_m = residual.size(0);
  const int64_t residual_size_n = residual.size(1);
  const void* weight_ptr = weight.data_ptr();
  const int64_t weight_stride_n = weight.stride(0);
  const int64_t weight_size_n = weight.size(0);
  void* output_ptr = output.data_ptr();
  const int64_t output_stride_m = output.stride(0);
  const int64_t output_stride_n = output.stride(1);
  const int64_t output_size_m = output.size(0);
  const int64_t output_size_n = output.size(1);
  void* residual_output_ptr = residual_output.data_ptr();
  const int64_t residual_output_stride_m = residual_output.stride(0);
  const int64_t residual_output_stride_n = residual_output.stride(1);
  const int64_t residual_output_size_m = residual_output.size(0);
  const int64_t residual_output_size_n = residual_output.size(1);
  const at::ScalarType dtype = input.scalar_type();
  const at::ScalarType weight_dtype = weight.scalar_type();
  const Dim3 grid{static_cast<int>(grid_x), static_cast<int>(grid_y),
                  static_cast<int>(grid_z)};
  const Dim3 block{static_cast<int>(block_x), static_cast<int>(block_y),
                   static_cast<int>(block_z)};
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  // ---- PRECONDITIONS: the config, the call, then its instance. The first
  // Error refuses the call; past this section the call runs. ----
  const std::optional<AllReduceAddRmsNormAlgorithm> named =
      algorithm_named<AllReduceAddRmsNormBuildConfig>(algorithm_name);
  if (!named) return error_code(Error::no_such_kernel);
  const AllReduceAddRmsNormAlgorithm algorithm = *named;
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_code(Error::no_such_quantization);
  const AllReduceAddRmsNormBuildConfig build{
      .algorithm = algorithm,
      .block = block,
      .waves_per_eu = static_cast<int>(waves_per_eu),
      .tile_m = static_cast<int>(tile_m),
      .tile_n = static_cast<int>(tile_n),
      .quantization = quantization_name == "int4" ? Quantization::int4
                                                  : Quantization::exact,
      .pack_bytes = static_cast<int>(pack_bytes)};
  const int64_t m = input_size_m,
                n = input_size_n;  // the problem's shape: the input's
  if (const std::optional<Error> e =
          refused(true, s.world, s.symmetric.staging_bytes,
                  s.symmetric.partials_bytes, s.symmetric.max_blocks, build,
                  grid, quantize_from_bytes, dtype, m, n))
    return error_code(e);
  // A LAYOUT THE PACKED LOADS CAN READ: every buffer's innermost stride 1, its
  // rows on whole packs of the build's.
  if (output_stride_n != 1 || residual_output_stride_n != 1 ||
      input_stride_n != 1 || residual_stride_n != 1)
    return error_code(Error::inner_stride_not_one);
  if (output_stride_m * elem_bytes(dtype) % build.pack_bytes != 0 ||
      residual_output_stride_m * elem_bytes(dtype) % build.pack_bytes != 0 ||
      input_stride_m * elem_bytes(dtype) % build.pack_bytes != 0 ||
      residual_stride_m * elem_bytes(dtype) % build.pack_bytes != 0)
    return error_code(Error::row_stride_not_packs);
  if (weight_stride_n != 1) return error_code(Error::inner_stride_not_one);
  // THE PARTIALS'S ROWS, what partials_need sized it for (a one-shot keeps
  // none), at its row stride: the exact pull two-shot's row is its scale's pack
  // then the row (partials_need); the INT4 build's rows are the row alone, its
  // scales in the scales region.
  const bool scale_slot =
      algorithm == AllReduceAddRmsNormAlgorithm::two_shot_pull &&
      build.quantization == Quantization::exact;
  // The scale slot's row whole slots (scale_slot_row), so every row's scale is
  // aligned at any pack.
  const int64_t slot = scale_slot ? kScaleSlotBytes / elem_bytes(dtype) : 0;
  const int64_t row = payload_row(algorithm, build.quantization, n);
  const int64_t partials_stride_m =
      scale_slot
          ? scale_slot_row(row * elem_bytes(dtype)) / elem_bytes(dtype) + slot
          : row;
  const int64_t partials_rows =
      n > 0 ? partials_need(algorithm, m,
                            n * elem_bytes(dtype) / build.pack_bytes, s.world,
                            build.pack_bytes) /
                  (partials_stride_m * elem_bytes(dtype))
            : 0;
  const int shared_memory_bytes = 0;
  // A ROW WIDER THAN THE TILE runs the build's CHUNKED instance.
  const bool chunked = n > build.tile_n;
  // AN INT4 ROW WITH A TAIL runs the build's TAIL instance (int4_tail).
  const bool tail = build.quantization == Quantization::int4 &&
                    int4_tail(algorithm, m, n, s.world);
  // THE INT4 PULL TWO-SHOT'S SCALES, a rank's rows of them, in the scales
  // region (max_rows rows).
  if (algorithm == AllReduceAddRmsNormAlgorithm::two_shot_pull && !scale_slot &&
      (m + s.world - 1) / s.world > s.max_rows)
    return error_code(Error::partials_too_small);
  void* kernel = nullptr;
  // THE INSTANCE: the kernel compiled at this build, on this device, for this
  // call (builds.py's kAllReduceAddRmsNormInstances, a row each).
  for (const auto& i : kAllReduceAddRmsNormInstances)
    if (i.algorithm == build.algorithm && i.block_x == build.block.x &&
        i.block_y == build.block.y && i.block_z == build.block.z &&
        i.waves_per_eu == build.waves_per_eu && i.tile_m == build.tile_m &&
        i.tile_n == build.tile_n && i.tile_k == 1 &&
        i.quantization == build.quantization &&
        i.pack_bytes == build.pack_bytes && i.world == s.world &&
        i.wave_size == s.device.wave_size && i.dtype == dtype &&
        i.weight_dtype == weight_dtype && i.contiguous == false &&
        i.chunked == chunked && i.tail == tail &&
        i.quantize_int4 == (build.quantization == Quantization::int4 &&
                            build.pack_bytes == kPackBytes))
      kernel = i.kernel;
  if (kernel == nullptr) return error_code(Error::not_built);
  if (const std::optional<Error> e =
          check_kernel(s, kernel, grid, block, shared_memory_bytes, true))
    return error_code(e);

  // ---- THE LAUNCH: the buffers, the views of their regions, and the instance.
  // ----
  // ITS VIEWS OF THE REGIONS, each [m, n] (the pushed one-shot's inbox [slot,
  // m, n]), in elements: zeros where it reads none.
  int64_t partials_stride_n = 0, partials_size_m = 0, partials_size_n = 0,
          staging_stride_m = 0, staging_stride_n = 0, staging_size_m = 0,
          staging_size_n = 0, scales_stride_m = 0, scales_size_m = 0;
  bool in_place = false;
  partials_stride_n = 1;
  partials_size_m = partials_rows;
  partials_size_n = partials_stride_m;
  scales_stride_m = 1;
  scales_size_m =
      algorithm == AllReduceAddRmsNormAlgorithm::two_shot_pull && !scale_slot
          ? (m + s.world - 1) / s.world
          : 0;
  staging_stride_m = payload_row(algorithm, build.quantization, n);
  staging_stride_n = 1;
  staging_size_m = staged_rows(algorithm, build.quantization, m, s.world);
  staging_size_n = payload_row(algorithm, build.quantization, n);
  in_place = !is_push(algorithm) && build.quantization == Quantization::exact;
  // EVERY VIEW INSIDE ITS REGION (view_fits).
  if (!view_fits(1, 0, partials_size_m, partials_stride_m, partials_size_n,
                 partials_stride_n, elem_bytes(dtype),
                 s.symmetric.partials_bytes) ||
      !view_fits(1, 0, staging_size_m, staging_stride_m, staging_size_n,
                 staging_stride_n, elem_bytes(dtype),
                 s.symmetric.staging_bytes) ||
      !view_fits(1, 0, scales_size_m, scales_stride_m, 1, 1, 4,
                 s.max_rows * int64_t{sizeof(float)}))
    return error_code(Error::view_past_region);
  // THE INPUT TABLE (INPUT TABLES): where the kernel reads every rank's input
  // in place (an exact pull), or none where it reads the staging.
  void* const* table = nullptr;
  int64_t table_stride_m = staging_stride_m;
  if (in_place)
    if (const std::optional<Error> e = input_table(
            s, input_ptr, input_stride_m, input_size_m, input_size_n,
            elem_bytes(dtype), stream, table, table_stride_m))
      return error_code(e);
  const int table_size = in_place ? s.world : 0;
  const int64_t table_stride_n = in_place ? 1 : staging_stride_n;
  // THE LAUNCH: the instance's kernel, at the call's dtypes.
  const auto launch = [&]<typename T, typename W>() {
    const auto typed =
        reinterpret_cast<AllReduceAddRmsNormKernel<T, W>>(kernel);
    typed<<<dim3_of(grid), dim3_of(block), shared_memory_bytes, stream>>>(
        static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
        static_cast<int>(input_size_m), static_cast<int>(input_size_n),
        reinterpret_cast<const T* const*>(table), table_size, table_stride_m,
        table_stride_n, static_cast<const T*>(residual_ptr), residual_stride_m,
        residual_stride_n, static_cast<int>(residual_size_m),
        static_cast<int>(residual_size_n), static_cast<const W*>(weight_ptr),
        weight_stride_n, static_cast<int>(weight_size_n),
        static_cast<T*>(output_ptr), output_stride_m, output_stride_n,
        static_cast<int>(output_size_m), static_cast<int>(output_size_n),
        static_cast<T*>(residual_output_ptr), residual_output_stride_m,
        residual_output_stride_n, static_cast<int>(residual_output_size_m),
        static_cast<int>(residual_output_size_n),
        s.symmetric.partials_ptr_array, s.world, partials_stride_m,
        partials_stride_n, static_cast<int>(partials_size_m),
        static_cast<int>(partials_size_n), s.symmetric.scales_ptr_array,
        s.world, scales_stride_m, static_cast<int>(scales_size_m),
        s.symmetric.staging_ptr_array, s.world, staging_stride_m,
        staging_stride_n, static_cast<int>(staging_size_m),
        static_cast<int>(staging_size_n), s.symmetric.signals_ptr_array,
        s.world, s.symmetric.max_blocks * s.world, int64_t{s.world}, int64_t{1},
        int64_t{2}, s.symmetric.max_blocks, int64_t{s.world}, s.status_device,
        int64_t{1}, int64_t{kStatusFields}, s.rank, s.timeout_ticks, eps);
  };
  if (dtype == at::ScalarType::BFloat16 &&
      weight_dtype == at::ScalarType::Float)
    launch.template operator()<c10::BFloat16, float>();
  else if (dtype == at::ScalarType::BFloat16)
    launch.template operator()<c10::BFloat16, c10::BFloat16>();
  else if (weight_dtype == at::ScalarType::Float)
    launch.template operator()<c10::Half, float>();
  else
    launch.template operator()<c10::Half, c10::Half>();
  // POSTCONDITION: the launch went in.
  if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
  return std::nullopt;
}

// all_reduce_add_attn_res_rms_norm: at its config. One function: what Python
// passes, its tensors stripped to pointers and strides, the preconditions, the
// kernel's arguments, the launch, the postcondition.
ErrorCode hip_custom_communicator_all_reduce_add_attn_res_rms_norm(
    int64_t state, torch::Tensor& input,
    const std::optional<torch::Tensor>& prefix, torch::Tensor& blocks,
    torch::Tensor& norm_weight, torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& output_norm_weight,
    torch::Tensor& prefix_output, torch::Tensor& output,
    const std::optional<torch::Tensor>& block_output, int64_t num_blocks,
    double eps, double output_eps, const std::string& algorithm_name,
    int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t block_x,
    int64_t block_y, int64_t block_z, int64_t waves_per_eu, int64_t tile_m,
    int64_t tile_n, int64_t tile_k, const std::string& quantization_name,
    int64_t pack_bytes, std::optional<int64_t> quantize_from_bytes) {
  using namespace hip_custom_communicator;
  // THE CALL FROM TORCH: its state, its tensors, its config's ints.
  if (state == 0) return error_code(Error::not_open);
  if (const std::optional<Error> e = attn_res_tensors(
          input, prefix, blocks, norm_weight, qk_weight, output_norm_weight,
          prefix_output, output, block_output, num_blocks))
    return error_code(e);
  if (!all_ints({waves_per_eu, tile_m, tile_n, tile_k, pack_bytes, grid_x,
                 grid_y, grid_z, block_x, block_y, block_z}))
    return error_code(Error::field_out_of_range);
  CommsState& s = *reinterpret_cast<CommsState*>(state);
  const void* input_ptr = input.data_ptr();
  const int64_t input_stride_m = input.stride(0);
  const int64_t input_stride_n = input.stride(1);
  const int64_t input_size_m = input.size(0);
  const int64_t input_size_n = input.size(1);
  const void* prefix_ptr = prefix ? prefix->data_ptr() : nullptr;
  const int64_t prefix_stride_m = prefix ? prefix->stride(0) : 0;
  const int64_t prefix_stride_n = prefix ? prefix->stride(1) : 0;
  const int64_t prefix_size_m = prefix ? prefix->size(0) : 0;
  const int64_t prefix_size_n = prefix ? prefix->size(1) : 0;
  const void* blocks_ptr = blocks.data_ptr();
  const int64_t blocks_stride_m = blocks.stride(0);
  const int64_t blocks_stride_r = blocks.stride(1);
  const int64_t blocks_stride_n = blocks.stride(2);
  const int64_t blocks_size_m = blocks.size(0);
  const int64_t blocks_size_r = blocks.size(1);
  const int64_t blocks_size_n = blocks.size(2);
  const void* norm_weight_ptr = norm_weight.data_ptr();
  const int64_t norm_weight_stride_n = norm_weight.stride(0);
  const int64_t norm_weight_size_n = norm_weight.size(0);
  const void* qk_weight_ptr = qk_weight.data_ptr();
  const int64_t qk_weight_stride_n = qk_weight.stride(0);
  const int64_t qk_weight_size_n = qk_weight.size(0);
  const void* output_norm_weight_ptr =
      output_norm_weight ? output_norm_weight->data_ptr() : nullptr;
  const int64_t output_norm_weight_stride_n =
      output_norm_weight ? output_norm_weight->stride(0) : 0;
  const int64_t output_norm_weight_size_n =
      output_norm_weight ? output_norm_weight->size(0) : 0;
  void* prefix_output_ptr = prefix_output.data_ptr();
  const int64_t prefix_output_stride_m = prefix_output.stride(0);
  const int64_t prefix_output_stride_n = prefix_output.stride(1);
  const int64_t prefix_output_size_m = prefix_output.size(0);
  const int64_t prefix_output_size_n = prefix_output.size(1);
  void* output_ptr = output.data_ptr();
  const int64_t output_stride_m = output.stride(0);
  const int64_t output_stride_n = output.stride(1);
  const int64_t output_size_m = output.size(0);
  const int64_t output_size_n = output.size(1);
  void* block_output_ptr = block_output ? block_output->data_ptr() : nullptr;
  const int64_t block_output_stride_m =
      block_output ? block_output->stride(0) : 0;
  const int64_t block_output_stride_n =
      block_output ? block_output->stride(1) : 0;
  const int64_t block_output_size_m = block_output ? block_output->size(0) : 0;
  const int64_t block_output_size_n = block_output ? block_output->size(1) : 0;
  const at::ScalarType dtype = input.scalar_type();
  const Dim3 grid{static_cast<int>(grid_x), static_cast<int>(grid_y),
                  static_cast<int>(grid_z)};
  const Dim3 block{static_cast<int>(block_x), static_cast<int>(block_y),
                   static_cast<int>(block_z)};
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  // ---- PRECONDITIONS: the config, the call, then its instance. The first
  // Error refuses the call; past this section the call runs. ----
  const std::optional<AllReduceAddAttnResRmsNormAlgorithm> named =
      algorithm_named<AllReduceAddAttnResRmsNormBuildConfig>(algorithm_name);
  if (!named) return error_code(Error::no_such_kernel);
  const AllReduceAddAttnResRmsNormAlgorithm algorithm = *named;
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_code(Error::no_such_quantization);
  const AllReduceAddAttnResRmsNormBuildConfig build{
      .algorithm = algorithm,
      .block = block,
      .waves_per_eu = static_cast<int>(waves_per_eu),
      .tile_m = static_cast<int>(tile_m),
      .tile_n = static_cast<int>(tile_n),
      .tile_k = static_cast<int>(tile_k),
      .quantization = quantization_name == "int4" ? Quantization::int4
                                                  : Quantization::exact,
      .pack_bytes = static_cast<int>(pack_bytes)};
  const int64_t m = input_size_m,
                n = input_size_n;  // the problem's shape: the input's
  if (const std::optional<Error> e =
          refused(true, s.world, s.symmetric.staging_bytes,
                  s.symmetric.partials_bytes, s.symmetric.max_blocks, build,
                  grid, quantize_from_bytes, dtype, m, n))
    return error_code(e);
  // A LAYOUT THE PACKED LOADS CAN READ: every buffer's innermost stride 1, its
  // rows on whole packs of the build's.
  if (input_stride_n != 1 || blocks_stride_n != 1 ||
      norm_weight_stride_n != 1 || qk_weight_stride_n != 1 ||
      prefix_output_stride_n != 1 || output_stride_n != 1)
    return error_code(Error::inner_stride_not_one);
  if (input_stride_m * elem_bytes(dtype) % build.pack_bytes != 0 ||
      blocks_stride_m * elem_bytes(dtype) % build.pack_bytes != 0 ||
      blocks_stride_r * elem_bytes(dtype) % build.pack_bytes != 0 ||
      prefix_output_stride_m * elem_bytes(dtype) % build.pack_bytes != 0 ||
      output_stride_m * elem_bytes(dtype) % build.pack_bytes != 0)
    return error_code(Error::row_stride_not_packs);
  // THE OPTIONAL BUFFERS, checked only when given: a missing one is a null
  // pointer with zero strides, which is no layout at all (a prefix-less call
  // was refused as `inner_stride_not_one`, 2026-10-13).
  if (prefix_ptr != nullptr && prefix_stride_n != 1)
    return error_code(Error::inner_stride_not_one);
  if (prefix_ptr != nullptr &&
      prefix_stride_m * elem_bytes(dtype) % build.pack_bytes != 0)
    return error_code(Error::row_stride_not_packs);
  if (output_norm_weight_ptr != nullptr && output_norm_weight_stride_n != 1)
    return error_code(Error::inner_stride_not_one);
  if (block_output_ptr != nullptr && block_output_stride_n != 1)
    return error_code(Error::inner_stride_not_one);
  if (block_output_ptr != nullptr &&
      block_output_stride_m * elem_bytes(dtype) % build.pack_bytes != 0)
    return error_code(Error::row_stride_not_packs);
  // THE PARTIALS'S ROWS, what partials_need sized it for (a one-shot keeps
  // none).
  const int64_t partials_rows =
      n > 0 ? partials_need(algorithm, m,
                            n * elem_bytes(dtype) / build.pack_bytes, s.world,
                            build.pack_bytes) /
                  (n * elem_bytes(dtype))
            : 0;
  // A ROW WIDER THAN THE TILE keeps its rows' logits in the launch's LDS
  // (attn_res.cuh's attn_res_wide): [tile_m][num_blocks + 1] floats.
  const int shared_memory_bytes =
      n > build.tile_n
          ? static_cast<int>(build.tile_m * (num_blocks + 1) * sizeof(float))
          : 0;
  // A ROW WIDER THAN THE TILE runs the build's CHUNKED instance.
  const bool chunked = n > build.tile_n;
  // AN INT4 ROW WITH A TAIL runs the build's TAIL instance (int4_tail).
  const bool tail = build.quantization == Quantization::int4 &&
                    int4_tail(algorithm, m, n, s.world);
  void* kernel = nullptr;
  // THE INSTANCE: the kernel compiled at this build, on this device, for this
  // call (builds.py's kAllReduceAddAttnResRmsNormInstances, a row each).
  for (const auto& i : kAllReduceAddAttnResRmsNormInstances)
    if (i.algorithm == build.algorithm && i.block_x == build.block.x &&
        i.block_y == build.block.y && i.block_z == build.block.z &&
        i.waves_per_eu == build.waves_per_eu && i.tile_m == build.tile_m &&
        i.tile_n == build.tile_n && i.tile_k == build.tile_k &&
        i.quantization == build.quantization &&
        i.pack_bytes == build.pack_bytes && i.world == s.world &&
        i.wave_size == s.device.wave_size && i.dtype == dtype &&
        i.weight_dtype == dtype && i.contiguous == false &&
        i.chunked == chunked && i.tail == tail &&
        i.quantize_int4 == (build.quantization == Quantization::int4 &&
                            build.pack_bytes == kPackBytes))
      kernel = i.kernel;
  if (kernel == nullptr) return error_code(Error::not_built);
  if (const std::optional<Error> e =
          check_kernel(s, kernel, grid, block, shared_memory_bytes, true))
    return error_code(e);

  // ---- THE LAUNCH: the buffers, the views of their regions, and the instance.
  // ----
  // ITS VIEWS OF THE REGIONS, each [m, n] (the pushed one-shot's inbox [slot,
  // m, n]), in elements: zeros where it reads none.
  int64_t partials_stride_m = 0, partials_stride_n = 0, partials_size_m = 0,
          partials_size_n = 0, staging_stride_m = 0, staging_stride_n = 0,
          staging_size_m = 0, staging_size_n = 0;
  bool in_place = false;
  const int64_t row_n = payload_row(algorithm, build.quantization, n);
  partials_stride_m = row_n;
  partials_stride_n = 1;
  partials_size_m = partials_rows;
  partials_size_n = row_n;
  staging_stride_m = row_n;
  staging_stride_n = 1;
  staging_size_m = staged_rows(algorithm, build.quantization, m, s.world);
  staging_size_n = row_n;
  in_place = !is_push(algorithm) && build.quantization == Quantization::exact;
  // EVERY VIEW INSIDE ITS REGION (view_fits).
  if (!view_fits(1, 0, partials_size_m, partials_stride_m, partials_size_n,
                 partials_stride_n, elem_bytes(dtype),
                 s.symmetric.partials_bytes) ||
      !view_fits(1, 0, staging_size_m, staging_stride_m, staging_size_n,
                 staging_stride_n, elem_bytes(dtype),
                 s.symmetric.staging_bytes))
    return error_code(Error::view_past_region);
  // THE INPUT TABLE (INPUT TABLES): where the kernel reads every rank's input
  // in place (an exact pull), or none where it reads the staging.
  void* const* table = nullptr;
  int64_t table_stride_m = staging_stride_m;
  if (in_place)
    if (const std::optional<Error> e = input_table(
            s, input_ptr, input_stride_m, input_size_m, input_size_n,
            elem_bytes(dtype), stream, table, table_stride_m))
      return error_code(e);
  const int table_size = in_place ? s.world : 0;
  const int64_t table_stride_n = in_place ? 1 : staging_stride_n;
  // THE LAUNCH: the instance's kernel, at the call's dtypes.
  const auto launch = [&]<typename T, typename W>() {
    const auto typed =
        reinterpret_cast<AllReduceAddAttnResRmsNormKernel<T, W>>(kernel);
    typed<<<dim3_of(grid), dim3_of(block), shared_memory_bytes, stream>>>(
        static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
        static_cast<int>(input_size_m), static_cast<int>(input_size_n),
        reinterpret_cast<const T* const*>(table), table_size, table_stride_m,
        table_stride_n, static_cast<const T*>(prefix_ptr), prefix_stride_m,
        prefix_stride_n, static_cast<int>(prefix_size_m),
        static_cast<int>(prefix_size_n), static_cast<const T*>(blocks_ptr),
        blocks_stride_m, blocks_stride_r, blocks_stride_n,
        static_cast<int>(blocks_size_m), static_cast<int>(blocks_size_r),
        static_cast<int>(blocks_size_n), static_cast<const T*>(norm_weight_ptr),
        norm_weight_stride_n, static_cast<int>(norm_weight_size_n),
        static_cast<const T*>(qk_weight_ptr), qk_weight_stride_n,
        static_cast<int>(qk_weight_size_n),
        static_cast<const T*>(output_norm_weight_ptr),
        output_norm_weight_stride_n,
        static_cast<int>(output_norm_weight_size_n),
        static_cast<T*>(prefix_output_ptr), prefix_output_stride_m,
        prefix_output_stride_n, static_cast<int>(prefix_output_size_m),
        static_cast<int>(prefix_output_size_n), static_cast<T*>(output_ptr),
        output_stride_m, output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), static_cast<T*>(block_output_ptr),
        block_output_stride_m, block_output_stride_n,
        static_cast<int>(block_output_size_m),
        static_cast<int>(block_output_size_n), s.symmetric.partials_ptr_array,
        s.world, partials_stride_m, partials_stride_n,
        static_cast<int>(partials_size_m), static_cast<int>(partials_size_n),
        s.symmetric.staging_ptr_array, s.world, staging_stride_m,
        staging_stride_n, static_cast<int>(staging_size_m),
        static_cast<int>(staging_size_n), s.symmetric.signals_ptr_array,
        s.world, s.symmetric.max_blocks * s.world, int64_t{s.world}, int64_t{1},
        int64_t{2}, s.symmetric.max_blocks, int64_t{s.world}, s.status_device,
        int64_t{1}, int64_t{kStatusFields}, s.rank, s.timeout_ticks, num_blocks,
        eps, output_eps);
  };
  if (dtype == at::ScalarType::BFloat16)
    launch.template operator()<c10::BFloat16, c10::BFloat16>();
  else
    launch.template operator()<c10::Half, c10::Half>();
  // POSTCONDITION: the launch went in.
  if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
  return std::nullopt;
}

// rms_norm: this rank's rows normed (the second kernel of an op Python
// composes), at its config. rms_norm: out = rms_norm(inp, weight), vLLM's
// roundings exactly, on this rank's rows; the weight `weight_dtype`, dtype or
// f32. One function: what Python passes, its tensors stripped to pointers and
// strides, the preconditions, the kernel's arguments, the launch, the
// postcondition.
ErrorCode hip_custom_communicator_rms_norm(
    int64_t state, torch::Tensor& input, torch::Tensor& weight,
    torch::Tensor& output, double eps, int64_t grid_x, int64_t grid_y,
    int64_t grid_z, int64_t block_x, int64_t block_y, int64_t block_z,
    int64_t waves_per_eu, int64_t tile_m, int64_t tile_n,
    const std::string& quantization_name, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  using namespace hip_custom_communicator;
  // THE CALL FROM TORCH: its state, its tensors, its config's ints.
  if (state == 0) return error_code(Error::not_open);
  if (const std::optional<Error> e = norm_tensors(
          input, nullptr, weight, output, nullptr, /*in_place=*/true))
    return error_code(e);
  if (!all_ints({waves_per_eu, tile_m, tile_n, pack_bytes, grid_x, grid_y,
                 grid_z, block_x, block_y, block_z}))
    return error_code(Error::field_out_of_range);
  CommsState& s = *reinterpret_cast<CommsState*>(state);
  const void* input_ptr = input.data_ptr();
  const int64_t input_stride_m = input.stride(0);
  const int64_t input_stride_n = input.stride(1);
  const int64_t input_size_m = input.size(0);
  const int64_t input_size_n = input.size(1);
  const void* weight_ptr = weight.data_ptr();
  const int64_t weight_stride_n = weight.stride(0);
  const int64_t weight_size_n = weight.size(0);
  void* output_ptr = output.data_ptr();
  const int64_t output_stride_m = output.stride(0);
  const int64_t output_stride_n = output.stride(1);
  const int64_t output_size_m = output.size(0);
  const int64_t output_size_n = output.size(1);
  const at::ScalarType dtype = input.scalar_type();
  const at::ScalarType weight_dtype = weight.scalar_type();
  const Dim3 grid{static_cast<int>(grid_x), static_cast<int>(grid_y),
                  static_cast<int>(grid_z)};
  const Dim3 block{static_cast<int>(block_x), static_cast<int>(block_y),
                   static_cast<int>(block_z)};
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  // ---- PRECONDITIONS: the config, the call, then its instance. The first
  // Error refuses the call; past this section the call runs. ----
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_code(Error::no_such_quantization);
  const RmsNormBuildConfig build{.block = block,
                                 .waves_per_eu = static_cast<int>(waves_per_eu),
                                 .tile_m = static_cast<int>(tile_m),
                                 .tile_n = static_cast<int>(tile_n),
                                 .quantization = quantization_name == "int4"
                                                     ? Quantization::int4
                                                     : Quantization::exact,
                                 .pack_bytes = static_cast<int>(pack_bytes)};
  const int64_t m = input_size_m,
                n = input_size_n;  // the problem's shape: the input's
  if (const std::optional<Error> e =
          refused(false, s.world, 0, 0, s.symmetric.max_blocks, build, grid,
                  quantize_from_bytes, dtype, m, n))
    return error_code(e);
  // A LAYOUT THE PACKED LOADS CAN READ: every buffer's innermost stride 1, its
  // rows on whole packs of the build's.
  if (output_stride_n != 1 || input_stride_n != 1)
    return error_code(Error::inner_stride_not_one);
  if (output_stride_m * elem_bytes(dtype) % build.pack_bytes != 0 ||
      input_stride_m * elem_bytes(dtype) % build.pack_bytes != 0)
    return error_code(Error::row_stride_not_packs);
  if (weight_stride_n != 1) return error_code(Error::inner_stride_not_one);
  const int shared_memory_bytes = 0;
  // A ROW WIDER THAN THE TILE runs the build's CHUNKED instance.
  const bool chunked = n > build.tile_n;
  void* kernel = nullptr;
  // THE INSTANCE: the kernel compiled at this build, on this device, for this
  // call (builds.py's kRmsNormInstances, a row each).
  for (const auto& i : kRmsNormInstances)
    if (i.algorithm == RmsNormAlgorithm::rows && i.block_x == build.block.x &&
        i.block_y == build.block.y && i.block_z == build.block.z &&
        i.waves_per_eu == build.waves_per_eu && i.tile_m == build.tile_m &&
        i.tile_n == build.tile_n && i.tile_k == 1 &&
        i.quantization == build.quantization &&
        i.pack_bytes == build.pack_bytes && i.world == 0 &&
        i.wave_size == s.device.wave_size && i.dtype == dtype &&
        i.weight_dtype == weight_dtype && i.contiguous == false &&
        i.chunked == chunked && i.tail == false &&
        i.quantize_int4 == (build.quantization == Quantization::int4 &&
                            build.pack_bytes == kPackBytes))
      kernel = i.kernel;
  if (kernel == nullptr) return error_code(Error::not_built);
  if (const std::optional<Error> e =
          check_kernel(s, kernel, grid, block, shared_memory_bytes, false))
    return error_code(e);

  // ---- THE LAUNCH: the buffers, the views of their regions, and the instance.
  // ----
  // THE LAUNCH: the instance's kernel, at the call's dtypes.
  const auto launch = [&]<typename T, typename W>() {
    const auto typed = reinterpret_cast<RmsNormKernel<T, W>>(kernel);
    typed<<<dim3_of(grid), dim3_of(block), shared_memory_bytes, stream>>>(
        static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
        static_cast<int>(input_size_m), static_cast<int>(input_size_n),
        static_cast<const W*>(weight_ptr), weight_stride_n,
        static_cast<int>(weight_size_n), static_cast<T*>(output_ptr),
        output_stride_m, output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), eps);
  };
  if (dtype == at::ScalarType::BFloat16 &&
      weight_dtype == at::ScalarType::Float)
    launch.template operator()<c10::BFloat16, float>();
  else if (dtype == at::ScalarType::BFloat16)
    launch.template operator()<c10::BFloat16, c10::BFloat16>();
  else if (weight_dtype == at::ScalarType::Float)
    launch.template operator()<c10::Half, float>();
  else
    launch.template operator()<c10::Half, c10::Half>();
  // POSTCONDITION: the launch went in.
  if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
  return std::nullopt;
}

// add_rms_norm: this rank's rows added and normed (the second kernel of an op
// Python composes), at its config. add_rms_norm: out, residual_out =
// fused_add_rms_norm(inp, residual, weight), vLLM's roundings exactly, on this
// rank's rows. One function: what Python passes, its tensors stripped to
// pointers and strides, the preconditions, the kernel's arguments, the launch,
// the postcondition.
ErrorCode hip_custom_communicator_add_rms_norm(
    int64_t state, torch::Tensor& input, torch::Tensor& residual,
    torch::Tensor& weight, torch::Tensor& output,
    torch::Tensor& residual_output, double eps, int64_t grid_x, int64_t grid_y,
    int64_t grid_z, int64_t block_x, int64_t block_y, int64_t block_z,
    int64_t waves_per_eu, int64_t tile_m, int64_t tile_n,
    const std::string& quantization_name, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  using namespace hip_custom_communicator;
  // THE CALL FROM TORCH: its state, its tensors, its config's ints.
  if (state == 0) return error_code(Error::not_open);
  if (const std::optional<Error> e =
          norm_tensors(input, &residual, weight, output, &residual_output,
                       /*in_place=*/true))
    return error_code(e);
  if (!all_ints({waves_per_eu, tile_m, tile_n, pack_bytes, grid_x, grid_y,
                 grid_z, block_x, block_y, block_z}))
    return error_code(Error::field_out_of_range);
  CommsState& s = *reinterpret_cast<CommsState*>(state);
  const void* input_ptr = input.data_ptr();
  const int64_t input_stride_m = input.stride(0);
  const int64_t input_stride_n = input.stride(1);
  const int64_t input_size_m = input.size(0);
  const int64_t input_size_n = input.size(1);
  const void* residual_ptr = residual.data_ptr();
  const int64_t residual_stride_m = residual.stride(0);
  const int64_t residual_stride_n = residual.stride(1);
  const int64_t residual_size_m = residual.size(0);
  const int64_t residual_size_n = residual.size(1);
  const void* weight_ptr = weight.data_ptr();
  const int64_t weight_stride_n = weight.stride(0);
  const int64_t weight_size_n = weight.size(0);
  void* output_ptr = output.data_ptr();
  const int64_t output_stride_m = output.stride(0);
  const int64_t output_stride_n = output.stride(1);
  const int64_t output_size_m = output.size(0);
  const int64_t output_size_n = output.size(1);
  void* residual_output_ptr = residual_output.data_ptr();
  const int64_t residual_output_stride_m = residual_output.stride(0);
  const int64_t residual_output_stride_n = residual_output.stride(1);
  const int64_t residual_output_size_m = residual_output.size(0);
  const int64_t residual_output_size_n = residual_output.size(1);
  const at::ScalarType dtype = input.scalar_type();
  const at::ScalarType weight_dtype = weight.scalar_type();
  const Dim3 grid{static_cast<int>(grid_x), static_cast<int>(grid_y),
                  static_cast<int>(grid_z)};
  const Dim3 block{static_cast<int>(block_x), static_cast<int>(block_y),
                   static_cast<int>(block_z)};
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  // ---- PRECONDITIONS: the config, the call, then its instance. The first
  // Error refuses the call; past this section the call runs. ----
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_code(Error::no_such_quantization);
  const AddRmsNormBuildConfig build{
      .block = block,
      .waves_per_eu = static_cast<int>(waves_per_eu),
      .tile_m = static_cast<int>(tile_m),
      .tile_n = static_cast<int>(tile_n),
      .quantization = quantization_name == "int4" ? Quantization::int4
                                                  : Quantization::exact,
      .pack_bytes = static_cast<int>(pack_bytes)};
  const int64_t m = input_size_m,
                n = input_size_n;  // the problem's shape: the input's
  if (const std::optional<Error> e =
          refused(false, s.world, 0, 0, s.symmetric.max_blocks, build, grid,
                  quantize_from_bytes, dtype, m, n))
    return error_code(e);
  // A LAYOUT THE PACKED LOADS CAN READ: every buffer's innermost stride 1, its
  // rows on whole packs of the build's.
  if (output_stride_n != 1 || residual_output_stride_n != 1 ||
      input_stride_n != 1 || residual_stride_n != 1)
    return error_code(Error::inner_stride_not_one);
  if (output_stride_m * elem_bytes(dtype) % build.pack_bytes != 0 ||
      residual_output_stride_m * elem_bytes(dtype) % build.pack_bytes != 0 ||
      input_stride_m * elem_bytes(dtype) % build.pack_bytes != 0 ||
      residual_stride_m * elem_bytes(dtype) % build.pack_bytes != 0)
    return error_code(Error::row_stride_not_packs);
  if (weight_stride_n != 1) return error_code(Error::inner_stride_not_one);
  const int shared_memory_bytes = 0;
  // A ROW WIDER THAN THE TILE runs the build's CHUNKED instance.
  const bool chunked = n > build.tile_n;
  void* kernel = nullptr;
  // THE INSTANCE: the kernel compiled at this build, on this device, for this
  // call (builds.py's kAddRmsNormInstances, a row each).
  for (const auto& i : kAddRmsNormInstances)
    if (i.algorithm == AddRmsNormAlgorithm::rows &&
        i.block_x == build.block.x && i.block_y == build.block.y &&
        i.block_z == build.block.z && i.waves_per_eu == build.waves_per_eu &&
        i.tile_m == build.tile_m && i.tile_n == build.tile_n && i.tile_k == 1 &&
        i.quantization == build.quantization &&
        i.pack_bytes == build.pack_bytes && i.world == 0 &&
        i.wave_size == s.device.wave_size && i.dtype == dtype &&
        i.weight_dtype == weight_dtype && i.contiguous == false &&
        i.chunked == chunked && i.tail == false &&
        i.quantize_int4 == (build.quantization == Quantization::int4 &&
                            build.pack_bytes == kPackBytes))
      kernel = i.kernel;
  if (kernel == nullptr) return error_code(Error::not_built);
  if (const std::optional<Error> e =
          check_kernel(s, kernel, grid, block, shared_memory_bytes, false))
    return error_code(e);

  // ---- THE LAUNCH: the buffers, the views of their regions, and the instance.
  // ----
  // THE LAUNCH: the instance's kernel, at the call's dtypes.
  const auto launch = [&]<typename T, typename W>() {
    const auto typed = reinterpret_cast<AddRmsNormKernel<T, W>>(kernel);
    typed<<<dim3_of(grid), dim3_of(block), shared_memory_bytes, stream>>>(
        static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
        static_cast<int>(input_size_m), static_cast<int>(input_size_n),
        static_cast<const T*>(residual_ptr), residual_stride_m,
        residual_stride_n, static_cast<int>(residual_size_m),
        static_cast<int>(residual_size_n), static_cast<const W*>(weight_ptr),
        weight_stride_n, static_cast<int>(weight_size_n),
        static_cast<T*>(output_ptr), output_stride_m, output_stride_n,
        static_cast<int>(output_size_m), static_cast<int>(output_size_n),
        static_cast<T*>(residual_output_ptr), residual_output_stride_m,
        residual_output_stride_n, static_cast<int>(residual_output_size_m),
        static_cast<int>(residual_output_size_n), eps);
  };
  if (dtype == at::ScalarType::BFloat16 &&
      weight_dtype == at::ScalarType::Float)
    launch.template operator()<c10::BFloat16, float>();
  else if (dtype == at::ScalarType::BFloat16)
    launch.template operator()<c10::BFloat16, c10::BFloat16>();
  else if (weight_dtype == at::ScalarType::Float)
    launch.template operator()<c10::Half, float>();
  else
    launch.template operator()<c10::Half, c10::Half>();
  // POSTCONDITION: the launch went in.
  if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
  return std::nullopt;
}

// add_attn_res_rms_norm: AttnRes on this rank's rows (the second kernel of the
// op Python composes), at its config. add_attn_res_rms_norm: Kimi-K3's AttnRes
// and its RMSNorm on each of this rank's rows
// (all_reduce_add_attn_res_rms_norm's, its input the sum), `num_blocks`
// sources. One function: what Python passes, its tensors stripped to pointers
// and strides, the preconditions, the kernel's arguments, the launch, the
// postcondition.
ErrorCode hip_custom_communicator_add_attn_res_rms_norm(
    int64_t state, torch::Tensor& input,
    const std::optional<torch::Tensor>& prefix, torch::Tensor& blocks,
    torch::Tensor& norm_weight, torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& output_norm_weight,
    torch::Tensor& prefix_output, torch::Tensor& output,
    const std::optional<torch::Tensor>& block_output, int64_t num_blocks,
    double eps, double output_eps, int64_t grid_x, int64_t grid_y,
    int64_t grid_z, int64_t block_x, int64_t block_y, int64_t block_z,
    int64_t waves_per_eu, int64_t tile_m, int64_t tile_n, int64_t tile_k,
    const std::string& quantization_name, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  using namespace hip_custom_communicator;
  // THE CALL FROM TORCH: its state, its tensors, its config's ints.
  if (state == 0) return error_code(Error::not_open);
  if (const std::optional<Error> e = attn_res_tensors(
          input, prefix, blocks, norm_weight, qk_weight, output_norm_weight,
          prefix_output, output, block_output, num_blocks))
    return error_code(e);
  if (!all_ints({waves_per_eu, tile_m, tile_n, tile_k, pack_bytes, grid_x,
                 grid_y, grid_z, block_x, block_y, block_z}))
    return error_code(Error::field_out_of_range);
  CommsState& s = *reinterpret_cast<CommsState*>(state);
  const void* input_ptr = input.data_ptr();
  const int64_t input_stride_m = input.stride(0);
  const int64_t input_stride_n = input.stride(1);
  const int64_t input_size_m = input.size(0);
  const int64_t input_size_n = input.size(1);
  const void* prefix_ptr = prefix ? prefix->data_ptr() : nullptr;
  const int64_t prefix_stride_m = prefix ? prefix->stride(0) : 0;
  const int64_t prefix_stride_n = prefix ? prefix->stride(1) : 0;
  const int64_t prefix_size_m = prefix ? prefix->size(0) : 0;
  const int64_t prefix_size_n = prefix ? prefix->size(1) : 0;
  const void* blocks_ptr = blocks.data_ptr();
  const int64_t blocks_stride_m = blocks.stride(0);
  const int64_t blocks_stride_r = blocks.stride(1);
  const int64_t blocks_stride_n = blocks.stride(2);
  const int64_t blocks_size_m = blocks.size(0);
  const int64_t blocks_size_r = blocks.size(1);
  const int64_t blocks_size_n = blocks.size(2);
  const void* norm_weight_ptr = norm_weight.data_ptr();
  const int64_t norm_weight_stride_n = norm_weight.stride(0);
  const int64_t norm_weight_size_n = norm_weight.size(0);
  const void* qk_weight_ptr = qk_weight.data_ptr();
  const int64_t qk_weight_stride_n = qk_weight.stride(0);
  const int64_t qk_weight_size_n = qk_weight.size(0);
  const void* output_norm_weight_ptr =
      output_norm_weight ? output_norm_weight->data_ptr() : nullptr;
  const int64_t output_norm_weight_stride_n =
      output_norm_weight ? output_norm_weight->stride(0) : 0;
  const int64_t output_norm_weight_size_n =
      output_norm_weight ? output_norm_weight->size(0) : 0;
  void* prefix_output_ptr = prefix_output.data_ptr();
  const int64_t prefix_output_stride_m = prefix_output.stride(0);
  const int64_t prefix_output_stride_n = prefix_output.stride(1);
  const int64_t prefix_output_size_m = prefix_output.size(0);
  const int64_t prefix_output_size_n = prefix_output.size(1);
  void* output_ptr = output.data_ptr();
  const int64_t output_stride_m = output.stride(0);
  const int64_t output_stride_n = output.stride(1);
  const int64_t output_size_m = output.size(0);
  const int64_t output_size_n = output.size(1);
  void* block_output_ptr = block_output ? block_output->data_ptr() : nullptr;
  const int64_t block_output_stride_m =
      block_output ? block_output->stride(0) : 0;
  const int64_t block_output_stride_n =
      block_output ? block_output->stride(1) : 0;
  const int64_t block_output_size_m = block_output ? block_output->size(0) : 0;
  const int64_t block_output_size_n = block_output ? block_output->size(1) : 0;
  const at::ScalarType dtype = input.scalar_type();
  const Dim3 grid{static_cast<int>(grid_x), static_cast<int>(grid_y),
                  static_cast<int>(grid_z)};
  const Dim3 block{static_cast<int>(block_x), static_cast<int>(block_y),
                   static_cast<int>(block_z)};
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  // ---- PRECONDITIONS: the config, the call, then its instance. The first
  // Error refuses the call; past this section the call runs. ----
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_code(Error::no_such_quantization);
  const AddAttnResRmsNormBuildConfig build{
      .block = block,
      .waves_per_eu = static_cast<int>(waves_per_eu),
      .tile_m = static_cast<int>(tile_m),
      .tile_n = static_cast<int>(tile_n),
      .tile_k = static_cast<int>(tile_k),
      .quantization = quantization_name == "int4" ? Quantization::int4
                                                  : Quantization::exact,
      .pack_bytes = static_cast<int>(pack_bytes)};
  const int64_t m = input_size_m,
                n = input_size_n;  // the problem's shape: the input's
  if (const std::optional<Error> e =
          refused(false, s.world, 0, 0, s.symmetric.max_blocks, build, grid,
                  quantize_from_bytes, dtype, m, n))
    return error_code(e);
  // A LAYOUT THE PACKED LOADS CAN READ: every buffer's innermost stride 1, its
  // rows on whole packs of the build's.
  if (input_stride_n != 1 || blocks_stride_n != 1 ||
      norm_weight_stride_n != 1 || qk_weight_stride_n != 1 ||
      prefix_output_stride_n != 1 || output_stride_n != 1)
    return error_code(Error::inner_stride_not_one);
  if (input_stride_m * elem_bytes(dtype) % build.pack_bytes != 0 ||
      blocks_stride_m * elem_bytes(dtype) % build.pack_bytes != 0 ||
      blocks_stride_r * elem_bytes(dtype) % build.pack_bytes != 0 ||
      prefix_output_stride_m * elem_bytes(dtype) % build.pack_bytes != 0 ||
      output_stride_m * elem_bytes(dtype) % build.pack_bytes != 0)
    return error_code(Error::row_stride_not_packs);
  // THE OPTIONAL BUFFERS, checked only when given
  // (all_reduce_add_attn_res_rms_norm's).
  if (prefix_ptr != nullptr && prefix_stride_n != 1)
    return error_code(Error::inner_stride_not_one);
  if (prefix_ptr != nullptr &&
      prefix_stride_m * elem_bytes(dtype) % build.pack_bytes != 0)
    return error_code(Error::row_stride_not_packs);
  if (output_norm_weight_ptr != nullptr && output_norm_weight_stride_n != 1)
    return error_code(Error::inner_stride_not_one);
  if (block_output_ptr != nullptr && block_output_stride_n != 1)
    return error_code(Error::inner_stride_not_one);
  if (block_output_ptr != nullptr &&
      block_output_stride_m * elem_bytes(dtype) % build.pack_bytes != 0)
    return error_code(Error::row_stride_not_packs);
  // A ROW WIDER THAN THE TILE keeps its rows' logits in the launch's LDS
  // (attn_res.cuh's attn_res_wide): [tile_m][num_blocks + 1] floats.
  const int shared_memory_bytes =
      n > tile_n ? static_cast<int>(tile_m * (num_blocks + 1) * sizeof(float))
                 : 0;
  // A ROW WIDER THAN THE TILE runs the build's CHUNKED instance.
  const bool chunked = n > build.tile_n;
  void* kernel = nullptr;
  // THE INSTANCE: the kernel compiled at this build, on this device, for this
  // call (builds.py's kAddAttnResRmsNormInstances, a row each).
  for (const auto& i : kAddAttnResRmsNormInstances)
    if (i.algorithm == AddAttnResRmsNormAlgorithm::rows &&
        i.block_x == build.block.x && i.block_y == build.block.y &&
        i.block_z == build.block.z && i.waves_per_eu == build.waves_per_eu &&
        i.tile_m == build.tile_m && i.tile_n == build.tile_n &&
        i.tile_k == build.tile_k && i.quantization == build.quantization &&
        i.pack_bytes == build.pack_bytes && i.world == 0 &&
        i.wave_size == s.device.wave_size && i.dtype == dtype &&
        i.weight_dtype == dtype && i.contiguous == false &&
        i.chunked == chunked && i.tail == false &&
        i.quantize_int4 == (build.quantization == Quantization::int4 &&
                            build.pack_bytes == kPackBytes))
      kernel = i.kernel;
  if (kernel == nullptr) return error_code(Error::not_built);
  if (const std::optional<Error> e =
          check_kernel(s, kernel, grid, block, shared_memory_bytes, false))
    return error_code(e);

  // ---- THE LAUNCH: the buffers, the views of their regions, and the instance.
  // ----
  // THE LAUNCH: the instance's kernel, at the call's dtypes.
  const auto launch = [&]<typename T, typename W>() {
    const auto typed = reinterpret_cast<AddAttnResRmsNormKernel<T, W>>(kernel);
    typed<<<dim3_of(grid), dim3_of(block), shared_memory_bytes, stream>>>(
        static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
        static_cast<int>(input_size_m), static_cast<int>(input_size_n),
        static_cast<const T*>(prefix_ptr), prefix_stride_m, prefix_stride_n,
        static_cast<int>(prefix_size_m), static_cast<int>(prefix_size_n),
        static_cast<const T*>(blocks_ptr), blocks_stride_m, blocks_stride_r,
        blocks_stride_n, static_cast<int>(blocks_size_m),
        static_cast<int>(blocks_size_r), static_cast<int>(blocks_size_n),
        static_cast<const T*>(norm_weight_ptr), norm_weight_stride_n,
        static_cast<int>(norm_weight_size_n),
        static_cast<const T*>(qk_weight_ptr), qk_weight_stride_n,
        static_cast<int>(qk_weight_size_n),
        static_cast<const T*>(output_norm_weight_ptr),
        output_norm_weight_stride_n,
        static_cast<int>(output_norm_weight_size_n),
        static_cast<T*>(prefix_output_ptr), prefix_output_stride_m,
        prefix_output_stride_n, static_cast<int>(prefix_output_size_m),
        static_cast<int>(prefix_output_size_n), static_cast<T*>(output_ptr),
        output_stride_m, output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), static_cast<T*>(block_output_ptr),
        block_output_stride_m, block_output_stride_n,
        static_cast<int>(block_output_size_m),
        static_cast<int>(block_output_size_n), num_blocks, eps, output_eps);
  };
  if (dtype == at::ScalarType::BFloat16)
    launch.template operator()<c10::BFloat16, c10::BFloat16>();
  else
    launch.template operator()<c10::Half, c10::Half>();
  // POSTCONDITION: the launch went in.
  if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
  return std::nullopt;
}
