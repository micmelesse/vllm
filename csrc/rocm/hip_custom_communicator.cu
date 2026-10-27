// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Our HIP collectives as torch ops, built into `_rocm_C`: the host side in one file, the kernels
// in hip_custom_communicator/kernels/. In three parts: 1. the types and tables (the machine, the
// build, the errors, each op's builds, the layout, the state between Python and C++); 2. the
// helpers the API calls; 3. the API, a function a torch op, each taking what Python passes and
// returning the Error's number it was refused for, or none (it raises nothing): its preconditions,
// its kernel's arguments, its launch, its postcondition. Self-contained: torch and the HIP runtime,
// nothing from aiter.

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

// THE DECLARATIONS torch_bindings.cpp registers, so a definition here that differs is a compile
// error, not a mismatch at run time.
#include "ops.h"

// =================================================================================================
// 1. THE TYPES AND TABLES: the machine, the build, the errors, each op's builds, the layout's
// constants, and the state between Python and C++.
// =================================================================================================

namespace hip_custom_communicator {


// WHAT A CONFIG MOVES ACROSS THE LINKS, the host's alone: a config's `quantization` field. A kernel
// takes QUANTIZE_INT4 (bool), which the host derives per call from it and quantize_from_bytes.
namespace Quantization {
constexpr int exact = 0;
constexpr int int4 = 1;
}  // namespace Quantization

// THE TARGET, gfx950 (AMD Instinct MI355X; MI350X the same): the one arch the build compiles device
// code for (CMakeLists.txt) and the host tunes for; `open` refuses every other. Its facts, from
// `rocminfo` on n11 (2026-09-29) and ROCm's gpu-arch-specs.
constexpr const char* kTargetArch = "gfx950";
constexpr int kWaveSize = 64;       // lanes a wave (gfx9 runs 64-lane waves only)
constexpr int kMaxWorkgroup = 1024;  // threads a block the device allows
constexpr int kComputeUnits = 256;   // 32 active per XCD
constexpr int kMaxWavesPerCu = 32;   // resident at once
constexpr int kSimdsPerCu = 4;       // a wave runs on one SIMD
constexpr int kXgmiLinks = 7;        // one to each peer in an 8-GPU node


// THE BUILD: a PACK is the widest load (global_load_dwordx4), the unit every kernel loads, sums and
// stores, as in vLLM's and aiter's custom all-reduce; the dtypes and worlds dispatch instantiates,
// and the check refuses the rest. The exchange is sized at open from the largest call, never a
// constant; a build's block is its config's (any the device launches).
constexpr int kPackBytes = 16;
constexpr std::array<at::ScalarType, 2> kDtypes = {at::ScalarType::Half, at::ScalarType::BFloat16};
constexpr std::array<int, 3> kWorlds = {2, 4, 8};

// THE PEERS' LIMITS, for the layouts host and device share (the signal block, the state): at most
// a rank a link plus this one (the full xGMI mesh of one node); ONE SIGNAL SLOT A BLOCK THE DEVICE
// CAN HOLD RESIDENT (every CU full of one-wave blocks), so the signal block never rules out a grid.
// Whether one kernel's grid is resident is its occupancy (check_kernel).
constexpr int kMaxRanks = kXgmiLinks + 1;
constexpr int kMaxBlocks = kComputeUnits * kMaxWavesPerCu;


// =================================================================================================
// THE FAULT SITES
// =================================================================================================


// THE FAULT SITES' NAMES (common.cuh's Where, the status's `where`), for the host's reports.
constexpr int kWhereCount = 2;


// =================================================================================================
// TYPES
// =================================================================================================

// THE VOCABULARY, every type the tables and the handle name: why a call cannot run (`Error`),
// what a caller asks for (`OpType`), and what is compiled in (each op's build, its algorithm its
// op file's). `at::ScalarType` is common.cuh's.


// WHY C++ REFUSED. C++ never raises: every function returns its Error as a value (none: it ran),
// and Python names it by number from its mirror of this enum.
enum class Error : int {
  not_contiguous = 0,
  dtype_not_built = 1,
  world_not_built = 2,
  row_not_packs = 3,
  partials_too_small = 4,
  grid_not_resident = 5,
  staging_too_small = 6,
  device_not_built = 7,
  device_not_tuned = 8,
  weight_not_built = 9,
  no_such_kernel = 10,
  threads_not_built = 11,
  tile_not_built = 12,
  launch_out_of_range = 13,
  field_not_positive = 14,
  field_not_this_templates = 15,
  waves_not_built = 16,
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
constexpr int kNumErrors = 37;

// WHICH OP the caller asked for: an all-reduce, alone or with what it fuses, or a norm alone on
// this rank's rows (the second kernel of an op Python composes). EACH OP IS ONE `__global__`
// TEMPLATE (its kernel, kOps), a family of instances, one per set of template arguments; its
// ALGORITHM (its op file's enum: a one-shot or a two-shot, pulled or pushed) is a template
// parameter and its build's field, as its tile is.
enum class OpType : int {
  all_reduce = 0,
  all_reduce_rms_norm = 1,
  all_reduce_add_rms_norm = 2,
  all_reduce_add_attn_res_rms_norm = 3,
  rms_norm = 4,
  add_rms_norm = 5,
  add_attn_res_rms_norm = 6,
};

// A KERNEL'S CONFIG, in two parts and every field always set: its BUILD, what
// is compiled in (one of the builds its op lists, kOps: its algorithm
// first, which of the op's kernel's bodies it is), and its
// LAUNCH, what is chosen at each call. A tuned kernel holds both, a forced call
// gives both, and select hands both to the launch; no field is ever 0 for "left
// to someone else".
//
// A LAUNCH DIMENSION, the kernel's dim3: x, y and z. Every kernel here indexes x
// alone, so its configs hold y = z = 1; one that indexed y or z would widen its
// space, nothing else.
struct Dim3 {
  int x = 1;
  int y = 1;
  int z = 1;
  bool operator==(const Dim3&) const = default;
};

// EACH OP'S BUILD, Triton's autotune config less the grid: the block, its
// threads in each dimension (x: Triton's num_warps x 64), the waves a SIMD (an EU) the compiler fits its
// registers to (__launch_bounds__'s second value, Triton's waves_per_eu; 1 asks
// nothing), and the tile, TILE_M rows x TILE_N columns in elements; TILE_K is
// the reduced dimension a step (AttnRes's sources a step); `quantization`, what
// crosses the links (common.cuh's QUANTIZATION): a two-shot's build is listed
// as it is and moving INT4 (two configs of the table), forced like any field or
// named by a tuned point; `pack_bytes`, a pack's bytes, the unit every load, sum and store moves
// (PACK_BYTES: 16, the widest load, or 8, 4 or 2, a row's that is not whole wider packs, so
// every row has a build whose loads are aligned: PyTorch's and vLLM's vector width by alignment,
// tuned here like the tile). ONE TYPE AN OP, so a config cannot cross ops: its op (`op`), the
// algorithms its kernel has by name (`algorithms`, the table's and Python's), and
// its fields. A norm on this rank's rows has one algorithm (it reads no peer), so its build has
// none.
template <typename A>
struct Named {
  A algorithm;
  const char* name;
};
// all_reduce: a tile a work item, the grid striding over the buffer's.
struct AllReduceBuildConfig {
  static constexpr OpType op = OpType::all_reduce;
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
  int pack_bytes = kPackBytes;  // a pack's bytes (PACK_BYTES)
  bool operator==(const AllReduceBuildConfig&) const = default;
};
// all_reduce_rms_norm.
struct AllReduceRmsNormBuildConfig {
  static constexpr OpType op = OpType::all_reduce_rms_norm;
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
  int pack_bytes = kPackBytes;  // a pack's bytes (PACK_BYTES)
  bool operator==(const AllReduceRmsNormBuildConfig&) const = default;
};
// all_reduce_add_rms_norm.
struct AllReduceAddRmsNormBuildConfig {
  static constexpr OpType op = OpType::all_reduce_add_rms_norm;
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
  int pack_bytes = kPackBytes;  // a pack's bytes (PACK_BYTES)
  bool operator==(const AllReduceAddRmsNormBuildConfig&) const = default;
};
// all_reduce_add_attn_res_rms_norm: TILE_K sources a step.
struct AllReduceAddAttnResRmsNormBuildConfig {
  static constexpr OpType op = OpType::all_reduce_add_attn_res_rms_norm;
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
  int pack_bytes = kPackBytes;  // a pack's bytes (PACK_BYTES)
  bool operator==(const AllReduceAddAttnResRmsNormBuildConfig&) const = default;
};
// rms_norm (its kernel rms_norm_rows).
struct RmsNormBuildConfig {
  static constexpr OpType op = OpType::rms_norm;
  Dim3 block;
  int waves_per_eu;
  int tile_m;
  int tile_n;
  int quantization = Quantization::exact;  // a Quantization
  int pack_bytes = kPackBytes;  // a pack's bytes (PACK_BYTES)
  bool operator==(const RmsNormBuildConfig&) const = default;
};
// add_rms_norm (its kernel add_rms_norm_rows).
struct AddRmsNormBuildConfig {
  static constexpr OpType op = OpType::add_rms_norm;
  Dim3 block;
  int waves_per_eu;
  int tile_m;
  int tile_n;
  int quantization = Quantization::exact;  // a Quantization
  int pack_bytes = kPackBytes;  // a pack's bytes (PACK_BYTES)
  bool operator==(const AddRmsNormBuildConfig&) const = default;
};
// add_attn_res_rms_norm (its kernel add_attn_res_rms_norm_rows): TILE_K sources a step.
struct AddAttnResRmsNormBuildConfig {
  static constexpr OpType op = OpType::add_attn_res_rms_norm;
  Dim3 block;
  int waves_per_eu;
  int tile_m;
  int tile_n;
  int tile_k;
  int quantization = Quantization::exact;  // a Quantization
  int pack_bytes = kPackBytes;  // a pack's bytes (PACK_BYTES)
  bool operator==(const AddAttnResRmsNormBuildConfig&) const = default;
};
// IN OpType'S ORDER: a build's index is its op.
using BuildConfig =
    std::variant<AllReduceBuildConfig, AllReduceRmsNormBuildConfig, AllReduceAddRmsNormBuildConfig,
                 AllReduceAddAttnResRmsNormBuildConfig, RmsNormBuildConfig, AddRmsNormBuildConfig,
                 AddAttnResRmsNormBuildConfig>;
template <OpType OP>
using build_t = std::variant_alternative_t<static_cast<size_t>(OP), BuildConfig>;
// A BUILD WITH AN ALGORITHM: every kernel's but a norm's on rows, whose one algorithm the table
// names kRowsAlgorithm.
template <typename B>
concept HasAlgorithm = requires(B b) { b.algorithm; };
constexpr std::string_view kRowsAlgorithm = "rows";


// =================================================================================================
// THE COMMUNICATOR'S MEMORY
// =================================================================================================

// NO STATE IN C++: Python owns every buffer, as addresses, and hands each op what it reads. THREE
// BUFFERS a rank, each for a physical reason:
//  - SHARED (the arena): one hipExtMallocWithFlags (uncached), exported over legacy HIP IPC and
//    opened by every peer: the signals, the staging, the partials, the scales, the pushed one-shot's
//    inbox, a guard after each (`arena_layout`). A peer's is at whatever address its opening gave,
//    so a kernel takes each region as its own DEVICE ARRAY of every rank's address of it.
//  - PRIVATE: one hipMalloc read by this rank's kernels only: those device arrays, the pushed
//    one-shot's epochs, the input tables (kPrivate*).
//  - HOST: the status words, host-mapped, which the kernels write and the host polls.
// AN EXACT CALL'S INPUT is read by its peers IN PLACE (aiter's gfx950 pattern), through an INPUT
// TABLE (INPUT TABLES, below): a captured call's table points at every rank's own tensor, written
// by Python when the capture exits; an eager call's input is copied into this rank's staging on
// the launch's stream, and its table (slot 0, written at open) points at every rank's staging. An
// INT4 call's kernel quantizes its input into the staging itself, before its first barrier.

// A HIP CALL'S FAILURE, returned as hip_failed by the function it is in (every caller of HIP here
// returns an Error); the HIP error itself is cleared.
#define HIP_CHECK(expr)                \
  do {                                 \
    if ((expr) != hipSuccess) {        \
      (void)hipGetLastError();         \
      return Error::hip_failed;        \
    }                                  \
  } while (0)


// THE SIGNALS' BYTES: two flag sets of kMaxBlocks blocks by kMaxRanks writers,
// rounded to 128 bytes; their region in the arena is padded to 4 KiB
// (kSignalsRegionBytes), so every region after it starts aligned.
constexpr int64_t kSignalsBytes =
    (2 * int64_t{kMaxBlocks} * kMaxRanks * int64_t{sizeof(uint32_t)} + 127) /
    128 * 128;
constexpr int64_t kSignalsRegionBytes = (kSignalsBytes + 4095) / 4096 * 4096;

// HOW THE ARENA IS BACKED: hipExtMallocWithFlags(hipDeviceMallocUncached), exported with
// hipIpcGetMemHandle, opened by each peer with hipIpcOpenMemHandle: the convention of every IPC
// communicator on ROCm (vLLM's custom all-reduce and quick reduce, aiter's). Correct cached or
// not under the barrier's memory model (common.cuh); uncached keeps peer reads off every L2. Not
// VMM: exchanging more than ~256 MiB of VMM allocations across 8 processes left the driver in an
// eviction storm on ROCm 10.0.
constexpr unsigned kArenaMallocFlags = hipDeviceMallocUncached;

// THE ARENA'S LAYOUT, the same offsets on every rank (`arena_layout`, from the sizes Python opens
// it at): the signals (the barrier flags), the staging (what this rank publishes of its input: an
// eager call's copy, an INT4 payload, or the pushed all-reduce's inbox), the partials (its share's
// sums for the all-gather; the add-RMSNorm pull's INT4 row scales at their end), then the unfenced
// region. A region holds either encoding, sized for the larger: a call
// publishes exact or INT4, never both, and every reuse is ordered by a barrier the previous
// call's readers passed.
//
// THE UNFENCED REGION: the pushed one-shot's inbox, written with the flag in the line and fenced by
// no barrier, so its own region (a stale word a fenced kernel left could carry an epoch's value):
// [writer, bank * row] rows of kUnfencedSlotBytes of payload, two banks of kUnfencedBlocks rows,
// a grid past them refused. Its epochs are rank-local and cached, in the private buffer.
constexpr int kUnfencedBlocks = 128;
constexpr int64_t kUnfencedSlotBytes = 2048;
constexpr int64_t kUnfencedEpochsBytes = int64_t{kUnfencedBlocks} * kPackBytes;
// THE STRIDES the kernels take for the epochs (a block's, in int32 words) and the signals (a
// parity's: every block's every writer).
constexpr int64_t kUnfencedEpochsStrideBlock = kPackBytes / int64_t{sizeof(int32_t)};
constexpr int64_t kSignalsStrideParity = int64_t{kMaxBlocks} * kMaxRanks;


// INPUT TABLES: an exact call's view of every rank's input, a table of kMaxRanks addresses in the
// private buffer, one pointer its kernel takes (aiter's RankData*). Slot 0, the eager table, points
// at every rank's staging (an eager call copies its input there first); every later slot is a
// captured launch's, reserved by the launch (CommsState's next_slot) and filled when the capture
// exits (`register_graph_buffers`). A slot is
// never reused (a graph keeps its address). An input may be overwritten once its launch ends:
// every peer's last read of it precedes a barrier the launch waits on. The slots are fixed at open
// (no allocation during a capture): 2^18, 16 MiB (2^14 was too few for Kimi-K3 at c48).
constexpr int64_t kInputTableSlots = int64_t{1} << 18;

// THE SCRATCH A KERNEL MAY ASK: the device's scratch over its every lane, by kScratchShares. A
// dispatch that asks too much fails only when it runs, aborting the process; the share is a
// margin for the queues a process holds (16 full dispatches; 4 KiB a lane at gfx950).
constexpr int64_t kScratchShares = 16;

// THE REGIONS' DEVICE ARRAYS in the private buffer, kRegionPtrArrays of kMaxRanks addresses, in
// this order (Python's `_REGION_ARRAYS`): the signals, the staging, the partials, the scales, the
// unfenced region.
constexpr int kSignalsArray = 0, kStagingArray = 1, kPartialsArray = 2, kScalesArray = 3,
          kUnfencedArray = 4, kRegionPtrArrays = 5;

// THE PRIVATE BUFFER'S LAYOUT (byte offsets): the regions' device arrays, the pushed one-shot's
// epochs, the input tables. 256-byte aligned each. CommsState's (open, connect, free).
constexpr int64_t kPrivateRegionPtrArrays = 0;
constexpr int64_t kPrivateUnfencedEpochs =
    (int64_t{kRegionPtrArrays} * kMaxRanks * int64_t{sizeof(char*)} + 255) / 256 * 256;
constexpr int64_t kPrivateTables =
    kPrivateUnfencedEpochs + (kUnfencedEpochsBytes + 255) / 256 * 256;
constexpr int64_t kPrivateBytes =
    kPrivateTables + kInputTableSlots * kMaxRanks * int64_t{sizeof(void*)};

// =================================================================================================
// THE OPS, in OpType's order: each one's name (as Python calls it), its kernel, what a call of it
// must be to run (which `build_info` hands Python's `should`), and the builds its kernel is
// compiled at. What each runs at a size is Python's (its tuned table), handed whole.
// =================================================================================================

// AN OP'S BUILDS, its BuildConfig each: what dispatch instantiates, what the check accepts and
// what a tuner searches. The launch (the grid) is not a build's: a config gives its own. tile_n
// counts the elements of a 16-bit dtype, the only ones built. A build that spills shows in its code
// object (gp report isa) and loses on the clock; none is ruled out by policy. GENERATED FROM THE
// TUNED TABLE (builds.py, at configure): each kernel's `kBuilds_<kernel>`, the build half of every
// config the table lists (an INT4 config a build as any other). The tune chooses them.
#include "hip_custom_communicator_builds.inc"
// EVERY KERNEL INSTANCE IS COMPILED IN ITS BUILD'S OWN FILE (builds.py): declared here, so the
// launches below instantiate none of them in this file.
#include "hip_custom_communicator_instances.inc"

// A NORM'S WEIGHT, besides its input's dtype: fp32 (by_weight compiles exactly
// these).
constexpr at::ScalarType kNormWeights[] = {at::ScalarType::Float};

struct Op {
  OpType type;
  const char* name;
  // Its kernel's name (the table's `kernel`).
  const char* kernel;
  // The dtypes its input may have.
  std::span<const at::ScalarType> dtypes;
  // The dtypes its weight may have besides its input's; none for an op with no
  // weight.
  std::span<const at::ScalarType> weights;
  // Whether its input is [rows, cols] as given (2-D), not any shape read as
  // rows of its last dimension.
  bool two_d;
  // Its kernel's builds.
  std::span<const BuildConfig> builds;
};

// AN OP, ITS NAME AND ITS KERNEL'S FROM TOKENS, so neither name can differ from the enum's and
// the builds are its kernel's.
#define HIP_CUSTOM_COMMUNICATOR_NAMED(o, k) OpType::o, #o, #k
#define HIP_CUSTOM_COMMUNICATOR_BUILDS(k) kBuilds_##k

constexpr Op kOps[] = {
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce, all_reduce), kDtypes, {}, true,
     HIP_CUSTOM_COMMUNICATOR_BUILDS(all_reduce)},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_rms_norm, all_reduce_rms_norm),
     kDtypes, kNormWeights, true,
     HIP_CUSTOM_COMMUNICATOR_BUILDS(all_reduce_rms_norm)},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_add_rms_norm, all_reduce_add_rms_norm),
     kDtypes, kNormWeights, true,
     HIP_CUSTOM_COMMUNICATOR_BUILDS(all_reduce_add_rms_norm)},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_add_attn_res_rms_norm,
                                   all_reduce_add_attn_res_rms_norm),
     kDtypes, {}, true,
     HIP_CUSTOM_COMMUNICATOR_BUILDS(all_reduce_add_attn_res_rms_norm)},
    // THE NORMS ON THIS RANK'S ROWS (STANDARDS 3.2, *We win at the op level*): the second kernel
    // of an op Python composes as two kernels of ours, our all-reduce and then this, on its own
    // grid at its own occupancy.
    {HIP_CUSTOM_COMMUNICATOR_NAMED(rms_norm, rms_norm_rows), kDtypes,
     kNormWeights, true, HIP_CUSTOM_COMMUNICATOR_BUILDS(rms_norm_rows)},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(add_rms_norm, add_rms_norm_rows), kDtypes,
     kNormWeights, true, HIP_CUSTOM_COMMUNICATOR_BUILDS(add_rms_norm_rows)},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(add_attn_res_rms_norm, add_attn_res_rms_norm_rows),
     kDtypes, {}, true,
     HIP_CUSTOM_COMMUNICATOR_BUILDS(add_attn_res_rms_norm_rows)},
};
#undef HIP_CUSTOM_COMMUNICATOR_BUILDS
#undef HIP_CUSTOM_COMMUNICATOR_NAMED
constexpr int kNumOps = sizeof(kOps) / sizeof(Op);


// AN INSTANCE OF AN OP'S KERNEL, every template argument a constant: its build (B, as compiled:
// compiled_quantization), what the call picks (the world, the dtypes, the layout's instance), and
// THE MACHINE'S WAVE, the host's fact and no call's (HARDWARE), every kernel's WAVE_SIZE. The
// template parameter of every launch (`kernel_pointer<I>`).
template <typename B>
struct Instance {
  B build;
  int world;
  at::ScalarType dtype;
  at::ScalarType weight_dtype;
  bool contiguous;
  bool chunked;
  bool tail;
  bool quantize_int4;
  int wave_size = kWaveSize;
};

template <typename WRAPPED>
struct type {
  using t = WRAPPED;
};

template <int INT_VALUE>
using constant = std::integral_constant<int, INT_VALUE>;


template <auto LISTED_BUILD>
struct build_constant {
  static constexpr auto value = LISTED_BUILD;
};




// THE STATE BETWEEN PYTHON AND C++, one a communicator: what it was given (this rank, the world,
// the device, the largest call, a wait's bound), what follows from it (the exchange's sizes, the
// arena's layout), what it holds (its buffers, OWNED; each peer's shared buffer as this process
// opened it, BORROWED), and the captured launches' input tables (the next free slot, those awaiting
// registration, the peers' inputs opened for them). Made by `open_state`, completed by
// `connect_state`, released by `close_peers` then `free_state`; Python holds it as a handle. A launch reads it, and a
// captured in-place launch reserves its slot in it: its one mutation during a call.
struct CommsState {
  int rank = 0;
  int world = 0;
  int device = 0;
  int64_t max_rows = 0;
  uint64_t timeout_ticks = 0;
  int64_t staging_bytes = 0;   // the staging region's: the largest call's input or payload
  int64_t partials_bytes = 0;  // the partials region's: the most any two-shot keeps
  // THE ARENA'S LAYOUT, byte offsets: the signals (the barrier flags), the staging (what this
  // rank publishes: an eager call's copy, an INT4 payload, a pushed two-shot's inbox), the partials
  // (its share's sums; the add-RMSNorm pull's INT4 row scales at their end), the unfenced region
  // (the pushed one-shot's inbox), and the arena's bytes. The same on every rank.
  int64_t signals = 0, staging = 0, partials = 0, scales = 0, unfenced = 0, arena_bytes = 0;
  // EACH REGION'S DEVICE ARRAY of every rank's address of it (in the private buffer, written by
  // connect), and the pushed one-shot's epochs: what every launch hands its kernel.
  uint32_t* const* signals_ptr_array = nullptr;
  char* const* staging_ptr_array = nullptr;
  char* const* partials_ptr_array = nullptr;
  float* const* scales_ptr_array = nullptr;
  char* const* unfenced_ptr_array = nullptr;
  int32_t* unfenced_epochs = nullptr;
  char* shared_buffer = nullptr;   // OWNED: the arena, exported to the peers
  char* private_buffer = nullptr;  // OWNED: the region arrays, the epochs, the input tables
  uint32_t* status_device = nullptr;  // OWNED: a kernel's fault record, written before it traps
  std::vector<char*> peers;        // BORROWED: each rank's arena as opened here (its own: ours)
  int64_t next_slot = 1;
  std::vector<std::pair<int64_t, const void*>> pending;  // (slot, input), captured, unregistered
  std::map<std::vector<char>, char*> opened_inputs;      // BORROWED: a peer allocation, by handle
};
}  // namespace hip_custom_communicator

  // namespace

// WHAT CROSSES THE TORCH BOUNDARY, as the tuples an op schema can return.
using Names = std::vector<std::string>;
using OpenWire = std::tuple<std::optional<int64_t>, int64_t, std::vector<int64_t>, std::string,
                            std::vector<int64_t>>;
using GraphMetaWire = std::tuple<std::optional<int64_t>, std::vector<int64_t>, std::vector<int64_t>>;
using BuildInfoWire =
    std::tuple<Names, std::vector<int64_t>, int64_t, Names, Names, Names, Names,
               Names, Names, std::vector<int64_t>, std::vector<int64_t>, Names,
               Names, std::vector<int64_t>, int64_t, Names>;
// AN ERROR CODE ACROSS THE BOUNDARY: none when the call ran, else the Error's number (a torch op
// returns no enum).
using ErrorCode = std::optional<int64_t>;

// =================================================================================================
// 2. THE HELPERS: what the API functions call, and the compile-time checks of the tables.
// =================================================================================================

namespace hip_custom_communicator {



// A DTYPE BY THE NAME PYTHON'S BUILD INFO GIVES IT (its `_DTYPES`).
constexpr const char* to_string(at::ScalarType d) {
  switch (d) {
    case at::ScalarType::Half:
      return "f16";
    case at::ScalarType::BFloat16:
      return "bf16";
    case at::ScalarType::Float:
      return "f32";
    default:
      return "unknown";
  }
}

// WHETHER THE BUILD HOLDS a dtype or a world: dispatch instantiates exactly
// these.
template <typename ELEMENT, size_t NUM_BUILT>
constexpr bool built_in(const std::array<ELEMENT, NUM_BUILT>& built,
                        ELEMENT x) {
  for (const ELEMENT& b : built)
    if (b == x) return true;
  return false;
}
constexpr bool dtype_built(at::ScalarType d) {
  return built_in(kDtypes, d);
}
constexpr bool world_built(int world) {
  return built_in(kWorlds, world);
}  // numbered from 1
constexpr const char* to_string(Where w) {
  switch (w) {
    case Where::start:
      return "start";
    case Where::peer_barrier:
      return "peer_barrier";
  }
  return "?";
}

constexpr const char* to_string(Error e) {
  switch (e) {
    case Error::not_contiguous:
      return "not_contiguous: the input is not contiguous";
    case Error::dtype_not_built:
      return "dtype_not_built: only float16 and bfloat16 are built";
    case Error::world_not_built:
      return "world_not_built: the world size is not 2, 4 or 8";
    case Error::row_not_packs:
      return "row_not_packs: the row is not whole packs of the config's pack_bytes (a "
             "build at a narrower pack takes it)";
    case Error::partials_too_small:
      return "partials_too_small: the two-shot's published sums, exact or INT4, "
             "exceed the partials";
    case Error::grid_not_resident:
      return "grid_not_resident: the grid exceeds the blocks the GPU holds "
             "resident";
    case Error::staging_too_small:
      return "staging_too_small: the input this kernel publishes whole, exact "
             "or INT4, exceeds the staging";
    case Error::device_not_built:
      return "device_not_built: this build holds no code for the device";
    case Error::device_not_tuned:
      return "device_not_tuned: the device is not the one select is calibrated "
             "for";
    case Error::weight_not_built:
      return "weight_not_built: a norm's weight is in the call's dtype or fp32";
    case Error::no_such_kernel:
      return "no_such_kernel: the op has no kernel of that name and algorithm";
    case Error::threads_not_built:
      return "threads_not_built: no build of the template runs at that block "
             "size";
    case Error::waves_not_built:
      return "waves_not_built: no build of the template at that block size has "
             "those waves a EU";
    case Error::tile_not_built:
      return "tile_not_built: no build of the template has that TILE_M or "
             "TILE_N";
    case Error::launch_out_of_range:
      return "launch_out_of_range: the block is whole waves up to the device's "
             "workgroup and the grid at least 1 and at most the resident maximum "
             "(the pushed one-shot's, its unfenced region's blocks), each over the "
             "dimensions its kernel indexes (x alone: y and z 1)";
    case Error::field_not_positive:
      return "field_not_positive: a forced config field is positive";
    case Error::field_not_this_templates:
      return "field_not_this_templates: a field its kernel does not have (tile_k on "
             "a kernel with no K, quantize_from_bytes on an exact build)";
    case Error::inner_stride_not_one:
      return "inner_stride_not_one: a buffer's innermost stride is not 1 (the "
             "loads are packs)";
    case Error::row_stride_not_packs:
      return "row_stride_not_packs: a buffer's row stride is not whole packs of the "
             "config's pack_bytes (a build at a narrower pack takes it)";
    case Error::no_such_quantization:
      return "no_such_quantization: the quantization is exact or int4";
    case Error::quantization_not_supported:
      return "quantization_not_supported: INT4 is not supported for the selected "
             "kernel";
    case Error::block_too_large:
      return "block_too_large: the block has more threads than the compiled kernel "
             "takes (its launch bounds, hipFuncGetAttributes' maxThreadsPerBlock)";
    case Error::lds_too_large:
      return "lds_too_large: the compiled kernel's LDS and the launch's dynamic LDS "
             "together are more than the device gives a block";
    case Error::scratch_too_large:
      return "scratch_too_large: the compiled kernel's scratch a lane is more than "
             "the device's scratch (hipExtLimitScratchMax) over its every lane, by "
             "kScratchShares (gfx950: 4 KiB a lane)";
    case Error::unfenced_too_small:
      return "unfenced_too_small: a pushed one-shot's tile does not fit the unfenced "
             "region: a row of it past a slot's kUnfencedSlotBytes of payload, or its "
             "grid's rows past a bank's kUnfencedBlocks";
    case Error::pack_out_of_range:
      return "pack_out_of_range: a config's pack_bytes is not 2, 4, 8 or 16 (a load the "
             "hardware has, up to its widest), or is less than an element of the dtype";
    case Error::hip_failed:
      return "hip_failed: a HIP runtime call failed (an allocation, an IPC handle, a copy, a "
             "device query)";
    case Error::not_on_device:
      return "not_on_device: a tensor is not on the device";
    case Error::shape_mismatch:
      return "shape_mismatch: a tensor's shape is not the op's (an output or a residual not the "
             "input's, a weight not 1-D of the row, blocks not [tokens, sources, hidden], "
             "num_blocks past its sources)";
    case Error::dtype_mismatch:
      return "dtype_mismatch: the tensors' dtypes differ where the op takes one";
    case Error::not_aligned:
      return "not_aligned: a weight or the blocks do not start at a pack";
    case Error::inputs_overlap_outputs:
      return "inputs_overlap_outputs: an input and an output share memory (not as the same "
             "tensor, where the op takes that)";
    case Error::field_out_of_range:
      return "field_out_of_range: a config field past an int";
    case Error::view_past_region:
      return "view_past_region: a view the launch would read or write ends past its region "
             "(the regions are sized from the largest call; a bug, refused before it launches)";
    case Error::tables_full:
      return "tables_full: every input table slot is taken (a captured exact pull a slot)";
    case Error::ranks_disagree:
      return "ranks_disagree: the ranks captured different numbers of in-place launches";
    case Error::not_open:
      return "not_open: a call on no communicator state (closed, or never opened)";
  }
  return "unknown";
}

// AN ALGORITHM'S PARTS, where a host decision needs only one: whether it is a one-shot or a
// two-shot, and whether it pushes (only the plain all-reduce pushes a one-shot: as lines, or
// staged).
template <typename A>
constexpr bool is_one_shot(A a) {
  if constexpr (std::is_same_v<A, AllReduceAlgorithm>)
    return a == A::one_shot_pull || a == A::one_shot_push || a == A::one_shot_push_staged;
  else return a == A::one_shot_pull;
}
template <typename A>
constexpr bool is_two_shot(A a) {
  return a == A::two_shot_pull || a == A::two_shot_push;
}
template <typename A>
constexpr bool is_push(A a) {
  if constexpr (std::is_same_v<A, AllReduceAlgorithm>)
    return a == A::one_shot_push || a == A::one_shot_push_staged || a == A::two_shot_push;
  else return a == A::two_shot_push;
}
constexpr int64_t volume(const Dim3& d) {
  return static_cast<int64_t>(d.x) * d.y * d.z;
}
// Whether `d` is x alone: y = z = 1, every kernel's today.
constexpr bool x_alone(const Dim3& d) { return d.y == 1 && d.z == 1; }
static_assert(build_t<OpType::all_reduce>::op == OpType::all_reduce &&
              build_t<OpType::all_reduce_rms_norm>::op == OpType::all_reduce_rms_norm &&
              build_t<OpType::all_reduce_add_rms_norm>::op == OpType::all_reduce_add_rms_norm &&
              build_t<OpType::all_reduce_add_attn_res_rms_norm>::op ==
                  OpType::all_reduce_add_attn_res_rms_norm &&
              build_t<OpType::rms_norm>::op == OpType::rms_norm &&
              build_t<OpType::add_rms_norm>::op == OpType::add_rms_norm &&
              build_t<OpType::add_attn_res_rms_norm>::op == OpType::add_attn_res_rms_norm,
              "BuildConfig's alternatives are not in OpType's order");


// THE FIELDS EVERY BUILD HAS.
// Its algorithm's name, as its kernel's `algorithms` names it.
constexpr std::string_view algorithm_name(const BuildConfig& b) {
  return std::visit([](const auto& f) -> std::string_view {
    if constexpr (HasAlgorithm<std::decay_t<decltype(f)>>) {
      for (const auto& a : f.algorithms)
        if (a.algorithm == f.algorithm) return a.name;
      return "";
    } else {
      return kRowsAlgorithm;
    }
  }, b);
}
// Its algorithm's parts (is_one_shot, is_two_shot, is_push); a norm on rows is none of them.
constexpr bool is_one_shot(const BuildConfig& b) {
  return std::visit([](const auto& f) {
    if constexpr (HasAlgorithm<std::decay_t<decltype(f)>>) return is_one_shot(f.algorithm);
    else return false;
  }, b);
}
constexpr bool is_two_shot(const BuildConfig& b) {
  return std::visit([](const auto& f) {
    if constexpr (HasAlgorithm<std::decay_t<decltype(f)>>) return is_two_shot(f.algorithm);
    else return false;
  }, b);
}
constexpr bool is_push(const BuildConfig& b) {
  return std::visit([](const auto& f) {
    if constexpr (HasAlgorithm<std::decay_t<decltype(f)>>) return is_push(f.algorithm);
    else return false;
  }, b);
}
constexpr Dim3 block_of(const BuildConfig& b) {
  return std::visit([](const auto& f) { return f.block; }, b);
}
constexpr int waves_of(const BuildConfig& b) {
  return std::visit([](const auto& f) { return f.waves_per_eu; }, b);
}
// WHAT A BUILD'S INSTANCE IS COMPILED MOVING: its quantization, but an INT4 build at a pack
// narrower than the widest compiles its exact body (INT4's word is a pack of 8 values,
// common.cuh's QUANTIZATION), which never runs: the call is refused by name
// (quantization_not_supported), as an INT4 one-shot is. builds.py's instances alike.
template <typename B>
constexpr int compiled_quantization(const B& b) {
  return b.pack_bytes == kPackBytes ? b.quantization : Quantization::exact;
}

constexpr int elem_bytes(at::ScalarType d) { return d == at::ScalarType::Float ? 4 : 2; }
constexpr int64_t unfenced_bytes(int world) {
  return 2 * int64_t{kUnfencedBlocks} * world * kUnfencedSlotBytes * (kLineBytes / kLinePayloadBytes);
}

// (A launch's input table: `input_table`, below, after CommsState.)

// WHY `kernel` CANNOT LAUNCH HERE, `grid` blocks of `block` threads and `shared_bytes` of dynamic
// LDS (a wide AttnRes row's logits), if it cannot: every check of a launch, in one place, what
// the compiled kernel asks (hipFuncGetAttributes, HIP's occupancy API) against what this device
// gives.
//  - block_too_large: more threads than the kernel takes (its launch bounds).
//  - lds_too_large: its LDS and the launch's dynamic LDS, past the device's a block.
//  - scratch_too_large: its scratch a lane past the device's scratch over its every lane (the
//    CUs times the waves a CU holds times a wave's lanes), by kScratchShares (gfx950: 32 GiB
//    over 256 x 32 x 64 lanes, 64 KiB a lane; by 16, 4 KiB). See kScratchShares.
//  - grid_not_resident, where the kernel waits on its peers (`waits_on_peers`; a norm on rows
//    waits on none, and any grid runs): more blocks than the device holds at once, the
//    textbook query (CUDA's cudaOccupancyMaxActiveBlocksPerMultiprocessor: blocks a CU from
//    the kernel's registers, LDS and the CU's wave slots) times the CUs. A peer barrier spins
//    until the same block on every peer arrives, so a grid past it can hang.
// HIP refuses the first two at the launch; the scratch only when the dispatch runs, aborting
// the queue and the process with it (the tune's crashes, 2026-10-24: AttnRes at 64 threads,
// tile 4x16384x4, 21 KiB a lane); the grid never (it hangs). Asked once per kernel, block size
// and dynamic LDS, then cached; the grid compared each call.
inline std::optional<Error> check_kernel(const void* kernel, const Dim3& grid, const Dim3& block,
                                  int shared_bytes, bool waits_on_peers) {
  const int threads = static_cast<int>(volume(block));
  // ASKED ONCE A PROCESS for each device, kernel, block size and dynamic LDS: a function of them.
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  static std::mutex mutex;
  static std::map<std::tuple<int, const void*, int, int>, std::pair<std::optional<Error>, int64_t>>
      occupancy;
  std::lock_guard<std::mutex> g(mutex);
  const std::tuple<int, const void*, int, int> key{device, kernel, threads, shared_bytes};
  auto it = occupancy.find(key);
  if (it == occupancy.end()) {
    hipFuncAttributes f;
    HIP_CHECK(hipFuncGetAttributes(&f, kernel));
    int lds = 0, cus = 0, threads_a_cu = 0;
    HIP_CHECK(hipDeviceGetAttribute(&lds, hipDeviceAttributeMaxSharedMemoryPerBlock, device));
    HIP_CHECK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, device));
    HIP_CHECK(hipDeviceGetAttribute(&threads_a_cu,
                                    hipDeviceAttributeMaxThreadsPerMultiProcessor, device));
    size_t budget = 0;  // the current device's
    HIP_CHECK(hipDeviceGetLimit(&budget, hipExtLimitScratchMax));
    const int64_t lanes = static_cast<int64_t>(cus) * threads_a_cu;  // every wave slot's lanes
    const int64_t scratch_a_lane = static_cast<int64_t>(budget) / lanes / kScratchShares;
    const std::optional<Error> refused =
        threads > f.maxThreadsPerBlock ? std::optional<Error>(Error::block_too_large)
        : static_cast<int64_t>(f.sharedSizeBytes) + shared_bytes > lds
            ? std::optional<Error>(Error::lds_too_large)
        : static_cast<int64_t>(f.localSizeBytes) > scratch_a_lane
            ? std::optional<Error>(Error::scratch_too_large)
            : std::nullopt;
    int per_cu = 0;  // asked of a kernel that launches alone
    if (!refused)
      HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, kernel, threads,
                                                             shared_bytes));
    it = occupancy.emplace(key, std::pair{refused, int64_t{per_cu} * cus}).first;
  }
  const auto& [refused, resident] = it->second;
  if (refused) return refused;
  if (waits_on_peers && volume(grid) > resident) return Error::grid_not_resident;
  return std::nullopt;
}


// =================================================================================================
// CONFIG
// =================================================================================================

// Built kernels, tuned tables, and the complete configuration of each call.


// EACH OP'S ALGORITHMS, what the host sizes and checks by: how its two-shot splits the work
// (SLICES COLUMNS: by columns, every block keeping every row, the pulled AttnRes and every pushed
// fused op; else by rows, or the plain all-reduce's flat walk), and whether its INT4 is WHOLE
// GROUPS A THREAD (the pulled norms, all_reduce_add_rms_norm.cuh: a block a row, TILE_N /
// (32 * THREADS) groups a thread; the tile does not size the payload, so any build sizes alike).
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

// AN INT4 ROW WITH A TAIL, the INT4 two-shots' TAIL instance: the flat walk (the plain two-shots)
// where the call is not whole groups; the plain one-shots, a groups-a-thread pulled norm, where a
// row is not; a column two-shot where a row is not whole groups for every rank.
template <typename A>
constexpr bool int4_tail(A algorithm, int64_t m, int64_t n, int world) {
  if constexpr (std::is_same_v<A, AllReduceAlgorithm>)
    return is_two_shot(algorithm) ? m * n % kInt4Group != 0 : n % kInt4Group != 0;
  else if (int4_groups_a_thread(algorithm)) return n % kInt4Group != 0;
  else return n % kInt4Group != 0 || n / kInt4Group % world != 0;
}

// THE INT4 FLAT WALK'S SHARE (reduce_scatter.cuh, the plain push two-shot's INT4): an [m, n]
// input's elements numbered row-major, a rank's share whole groups, the last rank's to the end.
// Its payloads lie on views of their own, [1, world * share] in the inbox and [1, share] in the
// partials: on the input's [m, n] the rounded-up shares would run past its values.
constexpr int64_t int4_flat_share(int64_t m, int64_t n, int world) {
  return ((m * n + kInt4Group - 1) / kInt4Group + world - 1) / world *
         kInt4Group;
}

constexpr const Op& op(OpType t) { return kOps[static_cast<int>(t)]; }
constexpr const char* to_string(OpType t) { return op(t).name; }

constexpr bool ops_in_order() {
  for (int i = 0; i < kNumOps; ++i)
    if (static_cast<int>(kOps[i].type) != i) return false;
  return true;
}
static_assert(ops_in_order(), "kOps must list every OpType in its order");

constexpr std::span<const BuildConfig> builds_of(OpType o) { return op(o).builds; }
constexpr bool every_op_built() {
  for (const Op& o : kOps)
    if (o.builds.empty()) return false;
  return true;
}
static_assert(every_op_built(), "an op's kernel lists no builds");
// EVERY BUILD OF AN OP IS ITS OWN TYPE (BuildConfig's index is its op).
constexpr bool builds_their_ops() {
  for (const Op& o : kOps)
    for (const BuildConfig& b : o.builds)
      if (std::visit([](const auto& f) { return f.op; }, b) != o.type) return false;
  return true;
}
static_assert(builds_their_ops(), "an op lists another op's build");

// Whether op `o`'s kernel builds `b`.
constexpr bool built(OpType o, const BuildConfig& b) {
  for (const BuildConfig& listed : builds_of(o))
    if (listed == b) return true;
  return false;
}
// Whether two builds are of one op at one algorithm.
constexpr bool same_algorithm(const BuildConfig& a, const BuildConfig& b) {
  return a.index() == b.index() && std::visit([&b](const auto& f) {
    using B = std::decay_t<decltype(f)>;
    if constexpr (HasAlgorithm<B>) return f.algorithm == std::get<B>(b).algorithm;
    else return true;
  }, a);
}
// Whether any build of `o` is at `b`'s algorithm and block (and its waves a EU): the check's
// names for a build that is not listed.
constexpr bool built_at_block(OpType o, const BuildConfig& b) {
  for (const BuildConfig& listed : builds_of(o))
    if (same_algorithm(listed, b) && block_of(listed) == block_of(b)) return true;
  return false;
}
constexpr bool built_at_waves(OpType o, const BuildConfig& b) {
  for (const BuildConfig& listed : builds_of(o))
    if (same_algorithm(listed, b) && block_of(listed) == block_of(b) &&
        waves_of(listed) == waves_of(b))
      return true;
  return false;
}
// EVERY BUILD'S BLOCK IS x ALONE: every kernel indexes x alone (types.cuh's
// Dim3), so a build with threads in y or z would launch threads it never uses.
constexpr bool every_build_x_alone() {
  for (const Op& o : kOps)
    for (const BuildConfig& b : o.builds)
      if (!x_alone(block_of(b))) return false;
  return true;
}
static_assert(every_build_x_alone(), "a build's block indexes y or z");

// =================================================================================================
// EACH OP'S KERNELS.
// =================================================================================================

// EVERY BUILD THE TABLE NAMES IS COMPILED AND TAKES ANY CALL ITS KERNEL CAN RUN (Michael, October
// 25: the build makes what it is told): each kernel walks any tile at any block, its tails its own;
// what a call cannot do is refused by name at the call (`refused`), its capacity (the regions, the
// unfenced slot), the hardware's (`check_kernel`), or an algorithm's (INT4 on a one-shot).

// A ROW OF THE STAGING AND PARTIALS AS A KERNEL VIEWS IT: n, or a groups-a-thread INT4 norm's
// payload row, whole groups (its kernels' n_pad: every row's groups start on a group).
template <typename A>
constexpr int64_t payload_row(A algorithm, int quantization, int64_t n) {
  return int4_groups_a_thread(algorithm) && quantization == Quantization::int4
             ? (n + kInt4Group - 1) / kInt4Group * kInt4Group
             : n;
}
// THE STAGING'S ROWS AS A KERNEL VIEWS IT: the input's m, or a groups-a-thread
// INT4 norm's inbox, WORLD slots of an owner's ceil(m / WORLD) rows.
template <typename A>
constexpr int64_t staged_rows(A algorithm, int quantization, int64_t m, int world) {
  return int4_groups_a_thread(algorithm) && quantization == Quantization::int4
             ? world * ((m + world - 1) / world)
             : m;
}


// =================================================================================================
// AN OP'S ALGORITHM BY NAME: the call names it, among its kernel's.
// =================================================================================================

// AN ALGORITHM OF B'S KERNEL by its name, or none where the kernel has no such algorithm.
template <typename B>
constexpr std::optional<decltype(B::algorithm)> algorithm_named(std::string_view name) {
  for (const auto& a : B::algorithms)
    if (name == a.name) return a.algorithm;
  return std::nullopt;
}

// =================================================================================================
// THE CHECK: the first Error config `c` meets on the call here, in
// this order, or none. Every Error is a capability (a kernel that cannot take
// the input), never "the unfused ops would be faster". Residency, which needs
// the compiled kernel, is each select's last check.
// =================================================================================================

// A KERNEL'S PARTIALS, row-major, the whole call's: the plain two-shot its slice of every row; a
// column two-shot the whole reduced tensor (each rank's columns at their
// place); a row two-shot its rank's rows, twice where it leaves two results
// (out and the residual). A one-shot reads the inputs and keeps nothing; `packs` a
// row's in packs of `pack` bytes (the build's). THE ADD-RMSNORM ROW TWO-SHOT's rows each
// carry their scale in a slot of kScaleSlotBytes before the row (common.cuh;
// kernels/all_reduce_add_rms_norm.cuh, its exact body: a row with its scale, one
// buffer), its row whole kScaleSlotBytes so the next row's scale is aligned
// (scale_slot_row).
template <typename A>
constexpr int64_t partials_need(A algorithm, int64_t m, int64_t packs, int world, int64_t pack) {
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
// EVERY ALGORITHM OF EVERY KERNEL, `f` handed each with its kernel (a norm on rows has none).
template <typename F>
constexpr void every_algorithm(F&& f) {
  for (const auto& a : AllReduceBuildConfig::algorithms)
    f(AllReduceBuildConfig::op, a.algorithm);
  for (const auto& a : AllReduceRmsNormBuildConfig::algorithms)
    f(AllReduceRmsNormBuildConfig::op, a.algorithm);
  for (const auto& a : AllReduceAddRmsNormBuildConfig::algorithms)
    f(AllReduceAddRmsNormBuildConfig::op, a.algorithm);
  for (const auto& a : AllReduceAddAttnResRmsNormBuildConfig::algorithms)
    f(AllReduceAddAttnResRmsNormBuildConfig::op, a.algorithm);
}

// A LAYOUT THE PACKED LOADS CAN READ: every buffer's innermost stride 1 and its
// rows starting on whole packs of `pack` bytes, the build's ({stride_m, stride_n} a
// buffer, in elements of `dtype`; a vector's stride_m 0).
inline std::optional<Error> layout_refused(
    at::ScalarType dtype, int64_t pack, std::initializer_list<std::pair<int64_t, int64_t>> strides) {
  for (const auto& [m, n] : strides) {
    if (n != 1) return Error::inner_stride_not_one;
    if (m * elem_bytes(dtype) % pack != 0)
      return Error::row_stride_not_packs;
  }
  return std::nullopt;
}

// THE CHECK OF A CALL'S CONFIG (its build, of its op's type B, its grid, and the bytes it
// quantizes from) on a call of `m` rows of `n`: the world and the dtype built; the config's ranges,
// each the hardware's (STANDARDS 3.2): the grid and the block x alone (every kernel's one
// dimension), the block whole waves up to the device's workgroup (a wider block's register trade
// is the tune's to measure), the grid at most the blocks the signals have slots for (whether this
// kernel's are resident is grid_not_resident), the waves a EU the CU holds, the tile positive, a
// pack a load the hardware has, quantize_from_bytes an INT4 build's alone; then what the build
// cannot run; then what the regions cannot hold. With no state (`collective` false) the checks
// that need peers are left out.
template <typename B>
std::optional<Error> refused(bool collective, int world, int64_t staging_bytes,
                             int64_t partials_bytes, const B& build, Dim3 grid,
                             std::optional<int64_t> quantize_from_bytes, at::ScalarType dtype, int64_t m,
                             int64_t n) {
  if (collective && !world_built(world)) return Error::world_not_built;
  if (!dtype_built(dtype)) return Error::dtype_not_built;
  if (!x_alone(build.block) || !x_alone(grid) || build.block.x < kWaveSize ||
      build.block.x > kMaxWorkgroup || build.block.x % kWaveSize != 0 ||
      grid.x < 1 || grid.x > kMaxBlocks || build.waves_per_eu < 1 ||
      build.waves_per_eu > kMaxWavesPerCu / kSimdsPerCu)
    return Error::launch_out_of_range;
  int tile_k = 1;  // a kernel with no K: its identity
  if constexpr (requires { build.tile_k; }) tile_k = build.tile_k;
  if (build.tile_m < 1 || build.tile_n < 1 || tile_k < 1) return Error::field_not_positive;
  if (build.quantization == Quantization::exact && quantize_from_bytes)
    return Error::field_not_this_templates;
  if (quantize_from_bytes.value_or(0) < 0) return Error::field_not_positive;
  // A PACK A LOAD THE HARDWARE HAS: 2, 4, 8 or 16 bytes (its widest).
  if (build.pack_bytes < 2 || build.pack_bytes > kPackBytes ||
      (build.pack_bytes & (build.pack_bytes - 1)) != 0)
    return Error::pack_out_of_range;
  const int64_t quantize_from = quantize_from_bytes.value_or(0);
  constexpr bool kPlain = std::is_same_v<B, AllReduceBuildConfig>;
  // THE BUILD'S PACK: at least an element, and whole packs a row (an aligned load of the
  // pack's width: a row that is not whole packs starts the next off them; a narrower
  // build's takes it).
  const int64_t pack = build.pack_bytes;
  if (pack < elem_bytes(dtype)) return Error::pack_out_of_range;
  if (n * elem_bytes(dtype) % pack != 0) return Error::row_not_packs;
  // INT4 ON A NORM ON ROWS (it moves nothing) OR A FUSED ONE-SHOT (it reads its peers' inputs as
  // they are): named before the build lookup, which lists no such build; INT4 AT A NARROWER PACK
  // than the widest: the format's word is a pack of 8 values (common.cuh's QUANTIZATION).
  const bool int4_build = build.quantization == Quantization::int4;
  if constexpr (HasAlgorithm<B>) {
    if (int4_build && ((!kPlain && is_one_shot(build.algorithm)) || pack != kPackBytes))
      return Error::quantization_not_supported;
  } else if (int4_build) {
    return Error::quantization_not_supported;
  }
  // A FUSED OP'S INT4 BUILD ALWAYS QUANTIZES: its kernels have no exact instance of their own.
  if (int4_build && !kPlain && quantize_from != 0)
    return Error::field_not_this_templates;
  // A BUILD THE KERNEL DOES NOT LIST, named by what differs: its block, its waves, its tile.
  if (!built_at_block(B::op, build)) return Error::threads_not_built;
  if (!built_at_waves(B::op, build)) return Error::waves_not_built;
  if (!built(B::op, build)) return Error::tile_not_built;
  if constexpr (!HasAlgorithm<B>) {
    return std::nullopt;  // a norm on rows reads no peer and no region
  } else {
    // WHAT THE CALL MOVES: INT4 where its INT4 build quantizes it (the call's bytes from its
    // quantize_from_bytes on), else exact (the INT4 build's instance passing it exact).
    const bool int4 = int4_build && m * n * elem_bytes(dtype) >= quantize_from;
    if constexpr (kPlain) {
      // A PUSHED ONE-SHOT IN THE UNFENCED REGION, its capacity (not a build's rule: any tile and
      // grid is built): a row of a block's tile in a slot's lines, a line a pack of at most
      // kLinePayloadBytes (a narrower pack its line alone: kernels/all_reduce.cuh), INT4 a line
      // a word of its tile's whole groups, a slot kUnfencedSlotBytes / kLinePayloadBytes lines;
      // a block's TILE_M rows of a bank's kUnfencedBlocks.
      const int64_t line_pack = pack < kLinePayloadBytes ? pack : kLinePayloadBytes;
      const int64_t row_lines =
          int4 ? (build.tile_n + kInt4Group - 1) / kInt4Group * kInt4GroupPacks
               : (build.tile_n * elem_bytes(dtype) + line_pack - 1) / line_pack;
      if (build.algorithm == AllReduceAlgorithm::one_shot_push &&
          (row_lines > kUnfencedSlotBytes / kLinePayloadBytes ||
           int64_t{grid.x} * build.tile_m > kUnfencedBlocks))
        return Error::unfenced_too_small;
    }
    if (!collective) return std::nullopt;
    if constexpr (kPlain) {
      // THE PLAIN ALL-REDUCE'S ONE-SHOTS IN THE STAGING (kernels/all_reduce.cuh): the INT4 pull's
      // [m, n] payload; the staged push's inbox, WORLD slots of an input (INT4: its payload, whole
      // kInt4ScalesAlign). The line push stages nothing (INT4 or exact; exact's eager input
      // aside, below).
      if (is_one_shot(build.algorithm) &&
          (build.algorithm == AllReduceAlgorithm::one_shot_push_staged || int4)) {
        const int64_t payload = int4 ? int4_payload_bytes(m, n) : m * n * elem_bytes(dtype);
        const int64_t slot =
            int4 ? (payload + kInt4ScalesAlign - 1) / kInt4ScalesAlign * kInt4ScalesAlign
                 : payload;
        const int64_t staged = build.algorithm == AllReduceAlgorithm::one_shot_push_staged
                                   ? world * slot
                               : build.algorithm == AllReduceAlgorithm::one_shot_pull ? payload
                                                                                 : 0;
        if (staged > staging_bytes) return Error::staging_too_small;
        return std::nullopt;
      }
    }
    const int64_t packs = (n * elem_bytes(dtype) + pack - 1) / pack;  // the last short
    // An INT4 payload lies in the region that holds it exact (common.cuh's
    // INT4 payload layout): the input's in the inputs, the sums' in the partials, at
    // most one row of the partials's elements (its every view's bound); the flat walk's
    // on its shares' own views (int4_flat_share).
    const bool flat = kPlain && is_two_shot(build.algorithm);
    const int64_t share = int4_flat_share(m, n, world);
    const int64_t row = payload_row(build.algorithm, Quantization::int4, n);
    if (int4 && (flat ? int4_payload_bytes(1, world * share)
                      : int4_payload_bytes(
                            staged_rows(build.algorithm, Quantization::int4, m, world), row)) >
                    staging_bytes)
      return Error::staging_too_small;
    const int64_t partials = partials_need(build.algorithm, m, packs, world, pack);
    if (int4 && (flat ? int4_payload_bytes(1, share)
                 : int4_groups_a_thread(build.algorithm)
                     ? int4_payload_bytes((m + world - 1) / world, row)
                     : int4_payload_bytes(1, partials / elem_bytes(dtype))) > partials_bytes)
      return Error::partials_too_small;
    // A COLLECTIVE RUNS THE WHOLE CALL IN ONE PASS: the regions are sized from the largest call
    // the server makes, so a call past them is refused here, never cut into bands. Its input is in
    // the staging whole: an eager exact call's host copy (INPUT TABLES; a captured one's is read
    // where it is, refused here alike, so the eager and captured calls of one shape agree), or a
    // pushed input.
    if (partials > partials_bytes) return Error::partials_too_small;
    if (m * n * elem_bytes(dtype) > staging_bytes) return Error::staging_too_small;
    return std::nullopt;
  }
}


// THE WEIGHT a norm takes: its input's dtype, or one of its op's `weights`.
constexpr std::optional<Error> weight_refused(OpType o, at::ScalarType dtype,
                                              at::ScalarType weight_dtype) {
  if (weight_dtype == dtype) return std::nullopt;
  for (const at::ScalarType w : op(o).weights)
    if (weight_dtype == w) return std::nullopt;
  return Error::weight_not_built;
}

// OVER THE BUILT LISTS (kWorlds, kDtypes), so what is compiled is what they say.
template <typename VISITOR, size_t... INDICES>
bool by_world_in(int world, VISITOR& f, std::index_sequence<INDICES...>) {
  constexpr auto& built = kWorlds;
  return ((world == built[INDICES] && (f(constant<built[INDICES]>{}), true)) || ...);
}

template <typename VISITOR>
bool by_world(int world, VISITOR&& f) {
  return by_world_in(world, f,
              std::make_index_sequence<kWorlds.size()>{});
}

template <typename VISITOR, size_t... INDICES>
bool by_dtype_in(at::ScalarType d, VISITOR& f, std::index_sequence<INDICES...>) {
  constexpr auto& built = kDtypes;
  return ((d == built[INDICES] && (f(type<c10::impl::ScalarTypeToCPPTypeT<built[INDICES]>>{}), true)) ||
          ...);
}

template <typename VISITOR>
bool by_dtype(at::ScalarType d, VISITOR&& f) {
  return by_dtype_in(d, f, std::make_index_sequence<kDtypes.size()>{});
}

// A norm's weight: T itself, or fp32.
template <typename DTYPE, typename VISITOR>
bool by_weight(at::ScalarType weight, at::ScalarType dtype, VISITOR&& f) {
  if (weight == dtype) {
    f(type<DTYPE>{});
    return true;
  }
  if (weight == at::ScalarType::Float) {
    f(type<float>{});
    return true;
  }
  return false;
}

// AN OP'S BUILD, compiled in: the one of op OP's builds (kOps) `b` is, and only those. f is
// handed it as build_constant<B>, B its op's build, every field a constant expression.
template <OpType OP, size_t BUILD_INDEX>
constexpr build_t<OP> listed_build = std::get<build_t<OP>>(builds_of(OP)[BUILD_INDEX]);
template <OpType OP, typename VISITOR, size_t... INDICES>
bool by_build_in(const BuildConfig& b, VISITOR& f, std::index_sequence<INDICES...>) {
  return ((b == builds_of(OP)[INDICES] && (f(build_constant<listed_build<OP, INDICES>>{}), true)) ||
          ...);
}

template <OpType OP, typename VISITOR>
bool by_build(const BuildConfig& b, VISITOR&& f) {
  return by_build_in<OP>(b, f, std::make_index_sequence<builds_of(OP).size()>{});
}

// OP `OP`'S INSTANCE for a call, compiled in: its build, the world, the dtypes, and which of
// the build's instances the call runs (contiguous, chunked, with a tail, quantizing). `f` is handed
// the Instance as its template argument, every field a constant. Whether the build compiled that
// instance (a call validated first always finds it).
template <OpType OP, typename F>
bool with_instance(const build_t<OP>& build, int world, at::ScalarType dtype, at::ScalarType weight_dtype,
                   bool contiguous, bool chunked, bool tail, bool quantize_int4, F&& f) {
  bool found = false;
  by_world(world, [&](auto world_constant) {
    by_dtype(dtype, [&](auto dtype_constant) {
      using T = typename decltype(dtype_constant)::t;
      // Only the norms instantiate a separate weight dtype.
      const auto weight = [&](auto w) {
        using W = typename decltype(w)::t;
        by_build<OP>(BuildConfig{build}, [&](auto b) {
          constexpr build_t<OP> B = decltype(b)::value;
          constexpr int WORLD = decltype(world_constant)::value;
          const auto layout = [&]<bool CONTIGUOUS, bool CHUNKED, bool TAIL, bool QUANTIZE_INT4>() {
            constexpr Instance<build_t<OP>> I = [=] {
              build_t<OP> compiled = B;
              compiled.quantization = compiled_quantization(B);
              return Instance<build_t<OP>>{compiled, WORLD, c10::CppTypeToScalarType<T>::value, c10::CppTypeToScalarType<W>::value,
                                           CONTIGUOUS, CHUNKED, TAIL, QUANTIZE_INT4};
            }();
            f.template operator()<I>();
            found = true;
          };
          // THE INSTANCES a build has: contiguous or strided (the plain two-shots), chunked or not
          // (the norms and AttnRes), with a tail or not (the INT4 two-shots), and the plain
          // all-reduce's INT4 build quantizing (its INT4 body) or passing the call exact (its
          // exact body); a fused op's INT4 build always quantizes.
          constexpr bool kTwoShot = is_two_shot(BuildConfig{B});
          const auto tailed = [&]<bool CONTIGUOUS, bool CHUNKED>() {
            constexpr bool kInt4 = compiled_quantization(B) == Quantization::int4;
            if constexpr (kInt4 && OP == OpType::all_reduce) {
              if (!quantize_int4) layout.template operator()<CONTIGUOUS, CHUNKED, false, false>();
              else if (tail) layout.template operator()<CONTIGUOUS, CHUNKED, true, true>();
              else layout.template operator()<CONTIGUOUS, CHUNKED, false, true>();
            } else if constexpr (kInt4 && kTwoShot) {
              if (tail) layout.template operator()<CONTIGUOUS, CHUNKED, true, true>();
              else layout.template operator()<CONTIGUOUS, CHUNKED, false, true>();
            } else layout.template operator()<CONTIGUOUS, CHUNKED, false, kInt4>();
          };
          if constexpr (OP == OpType::all_reduce && kTwoShot) {
            if (contiguous) tailed.template operator()<true, false>();
            else tailed.template operator()<false, false>();
          } else if constexpr (OP != OpType::all_reduce) {
            if (chunked) tailed.template operator()<false, true>();
            else tailed.template operator()<false, false>();
          } else tailed.template operator()<false, false>();
        });
      };
      if constexpr (OP == OpType::all_reduce_rms_norm || OP == OpType::all_reduce_add_rms_norm ||
                    OP == OpType::rms_norm || OP == OpType::add_rms_norm)
        by_weight<T>(weight_dtype, dtype, weight);
      else weight(type<T>{});
    });
  });
  return found;
}

// AN INSTANCE'S KERNEL: every kernel's template arguments, the one list of them (builds.py reads
// it to write the instances).
template <Instance I>
const void* kernel_pointer() {
  constexpr auto B = I.build;
  using T = c10::impl::ScalarTypeToCPPTypeT<I.dtype>;
  using W = c10::impl::ScalarTypeToCPPTypeT<I.weight_dtype>;
  if constexpr (B.op == OpType::all_reduce)
    return reinterpret_cast<const void*>(
        all_reduce<B.algorithm, I.quantize_int4, T, I.world, B.tile_m, B.tile_n, B.block.x,
            I.wave_size, B.pack_bytes, B.waves_per_eu, I.contiguous, I.tail>);
  if constexpr (B.op == OpType::all_reduce_rms_norm)
    return reinterpret_cast<const void*>(
        all_reduce_rms_norm<B.algorithm, I.quantize_int4, T, W, I.world, B.tile_n, B.block.x,
            I.wave_size, B.pack_bytes, B.waves_per_eu, I.chunked, I.tail>);
  if constexpr (B.op == OpType::rms_norm)
    return reinterpret_cast<const void*>(
        rms_norm_rows<T, W, B.tile_m, B.tile_n, B.block.x, I.wave_size, B.pack_bytes,
            B.waves_per_eu, I.chunked>);
  if constexpr (B.op == OpType::all_reduce_add_rms_norm)
    return reinterpret_cast<const void*>(
        all_reduce_add_rms_norm<B.algorithm, I.quantize_int4, T, W, I.world, B.tile_n, B.block.x,
            I.wave_size, B.pack_bytes, B.waves_per_eu, I.chunked, I.tail>);
  if constexpr (B.op == OpType::add_rms_norm)
    return reinterpret_cast<const void*>(
        add_rms_norm_rows<T, W, B.tile_m, B.tile_n, B.block.x, I.wave_size, B.pack_bytes,
            B.waves_per_eu, I.chunked>);
  if constexpr (B.op == OpType::all_reduce_add_attn_res_rms_norm)
    return reinterpret_cast<const void*>(
        all_reduce_add_attn_res_rms_norm<B.algorithm, I.quantize_int4, T, I.world, B.tile_m,
            B.tile_n, B.tile_k, B.block.x, I.wave_size, B.pack_bytes, B.waves_per_eu, I.chunked,
            I.tail>);
  if constexpr (B.op == OpType::add_attn_res_rms_norm)
    return reinterpret_cast<const void*>(
        add_attn_res_rms_norm_rows<T, B.tile_m, B.tile_n, B.tile_k, B.block.x, I.wave_size,
            B.pack_bytes, B.waves_per_eu, I.chunked>);
}

// A LAUNCH'S INPUT TABLE (INPUT TABLES): eager, the input copied into this rank's staging on
// `stream` and the eager table, slot 0, rows `n` apart; captured, the next slot reserved for it
// (filled by `register_graph_buffers` when the capture exits), rows at the input's own stride.
inline std::optional<Error> input_table(CommsState& s, const void* input, int64_t stride_m,
                                        int64_t m, int64_t n, int64_t elem_bytes,
                                        hipStream_t stream, void* const*& table,
                                        int64_t& table_stride_m) {
  hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
  HIP_CHECK(hipStreamIsCapturing(stream, &capture));
  const auto tables = reinterpret_cast<void* const*>(s.private_buffer + kPrivateTables);
  if (capture == hipStreamCaptureStatusActive) {
    if (s.next_slot >= kInputTableSlots) return Error::tables_full;
    s.pending.emplace_back(s.next_slot, input);
    table = tables + s.next_slot * kMaxRanks;
    table_stride_m = stride_m;
    ++s.next_slot;
    return std::nullopt;
  }
  const int64_t row = n * elem_bytes;
  if (m * row > s.staging_bytes) return Error::staging_too_small;
  HIP_CHECK(hipMemcpy2DAsync(s.shared_buffer + s.staging, static_cast<size_t>(row), input,
                             static_cast<size_t>(stride_m * elem_bytes), static_cast<size_t>(row),
                             static_cast<size_t>(m), hipMemcpyDeviceToDevice, stream));
  table = tables;
  table_stride_m = n;
  return std::nullopt;
}

// A VIEW INSIDE ITS REGION: `slots` x `rows` x `cols` elements of `elem` bytes at their strides end
// within `region_bytes` (an empty view is inside any). Every launch's views checked before it
// launches (October 26: n11 rebooted seven times timing the fused two-shots at size; the kernels
// check their indices only against the sizes they are handed).
constexpr bool view_fits(int64_t slots, int64_t stride_slot, int64_t rows, int64_t stride_m,
                         int64_t cols, int64_t stride_n, int64_t elem, int64_t region_bytes) {
  if (slots <= 0 || rows <= 0 || cols <= 0) return true;
  const int64_t last = (slots - 1) * stride_slot + (rows - 1) * stride_m + (cols - 1) * stride_n;
  return stride_slot >= 0 && stride_m >= 0 && stride_n >= 0 && (last + 1) * elem <= region_bytes;
}

// A LAUNCH DIMENSION AS HIP TAKES IT.
inline dim3 dim3_of(const Dim3& d) {
  return dim3(static_cast<unsigned>(d.x), static_cast<unsigned>(d.y),
              static_cast<unsigned>(d.z));
}
}  // namespace hip_custom_communicator


// =================================================================================
// THE TORCH OP BOUNDARY. Nothing crosses but values and one handle: the communicator's state
// (CommsState, C++'s: its memory, its peers, its captured inputs), made, connected and freed through
// the functions below; Python does only the exchange over its CPU group. EVERY ERROR IS A VALUE:
// no function here raises; each returns the Error's number it was refused for (none: it ran), and
// Python names it.
// =================================================================================

// EACH OP TAKES ITS STATE (a handle, below), its tensors and values, then its config's fields as
// plain values, its op's own (an algorithm where its kernel has several, tile_k
// where it has K). Nothing is picked here: Python fills what a caller leaves out and hands its
// table's pick the same way; the API checks it.

namespace {
using hip_custom_communicator::Error;

// EVERY CONFIG FIELD AN INT: the API takes ints.
bool all_ints(std::initializer_list<int64_t> vs) {
  for (const int64_t v : vs)
    if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) return false;
  return true;
}


// The ops' tensors: on device; their layouts are their strides, checked behind
// the interface.
bool on_device(std::initializer_list<const torch::Tensor*> ts) {
  for (const torch::Tensor* t : ts)
    if (!t->is_cuda()) return false;
  return true;
}
// A COLLECTIVE'S TENSOR AS [rows, cols]: its last dimension a row, every
// dimension before it the rows. A view, so its strides are the tensor's own;
// none where those dimensions cannot merge.
std::optional<torch::Tensor> as_rows(const torch::Tensor& t) {
  if (t.dim() == 0) return std::nullopt;
  try {
    return t.view({-1, t.size(-1)});
  } catch (const c10::Error&) {
    return std::nullopt;
  }
}
}
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
  return a.data_ptr() == b.data_ptr() && a.sizes() == b.sizes() && a.strides() == b.strides();
}
// AN OUTPUT MAY BE AN INPUT ITSELF (`in_place`) only where the kernel is correct with that exact
// alias: the norms on rows, whose every element is written by the thread that read it, after it
// read it (STANDARDS 3.5: in place is the caller's explicit choice). A partial overlap never is.
bool overlapping(std::initializer_list<const torch::Tensor*> inputs,
                 std::initializer_list<const torch::Tensor*> outputs, bool in_place = false) {
  for (const torch::Tensor* i : inputs)
    for (const torch::Tensor* o : outputs) {
      if (i == nullptr || o == nullptr || i->numel() == 0 || o->numel() == 0) continue;
      if (in_place && same_tensor(*i, *o)) continue;
      if (overlap(span_of(*i), span_of(*o))) return true;
    }
  return false;
}
}  // namespace

namespace {
using hip_custom_communicator::Error;

// A NORM CALL'S TENSORS, or the Error they are refused for: input [tokens, hidden], output (and
// with a residual, residual and residual_output) its shape and dtype, the weight 1-D of hidden at
// a pack (its dtype is the API's to refuse), no input sharing an output's memory.
std::optional<Error> norm_tensors(const torch::Tensor& input, const torch::Tensor* residual,
                                  const torch::Tensor& weight, const torch::Tensor& output,
                                  const torch::Tensor* residual_output, bool in_place) {
  if (!on_device({&input, &weight, &output})) return Error::not_on_device;
  if (input.dim() != 2 || output.sizes() != input.sizes() || weight.dim() != 1 ||
      weight.numel() != input.size(1))
    return Error::shape_mismatch;
  if (output.scalar_type() != input.scalar_type()) return Error::dtype_mismatch;
  // The weight is read a pack at a time, a pack's worth of input's elements per load.
  const int64_t weight_pack = hip_custom_communicator::kPackBytes /
                              input.element_size() * weight.element_size();
  if (reinterpret_cast<uintptr_t>(weight.data_ptr()) % weight_pack != 0) return Error::not_aligned;
  if (residual != nullptr) {
    if (!on_device({residual, residual_output})) return Error::not_on_device;
    if (residual->sizes() != input.sizes() || residual_output->sizes() != input.sizes())
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
    const torch::Tensor& blocks, const torch::Tensor& norm_weight, const torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& output_norm_weight, const torch::Tensor& prefix_output,
    const torch::Tensor& output, const std::optional<torch::Tensor>& block_output,
    int64_t num_blocks) {
  if (!on_device({&input, &norm_weight, &qk_weight, &prefix_output, &output, &blocks}))
    return Error::not_on_device;
  if (input.dim() != 2) return Error::shape_mismatch;
  const int64_t n = input.size(1);
  if (prefix_output.sizes() != input.sizes() || output.sizes() != input.sizes() ||
      blocks.dim() != 3 || blocks.size(0) != input.size(0) || blocks.size(2) != n ||
      blocks.stride(2) != 1 || num_blocks < 0 || num_blocks > blocks.size(1) ||
      norm_weight.dim() != 1 || norm_weight.numel() != n || qk_weight.dim() != 1 ||
      qk_weight.numel() != n)
    return Error::shape_mismatch;
  std::vector<const torch::Tensor*> same = {&blocks, &norm_weight, &qk_weight, &prefix_output,
                                            &output};
  for (const std::optional<torch::Tensor>* o : {&prefix, &output_norm_weight, &block_output})
    if (*o) {
      if (!on_device({&**o})) return Error::not_on_device;
      same.push_back(&**o);
    }
  for (const torch::Tensor* t : same)
    if (t->scalar_type() != input.scalar_type()) return Error::dtype_mismatch;
  if ((prefix && prefix->sizes() != input.sizes()) ||
      (block_output && (block_output->sizes() != input.sizes() || block_output->stride(1) != 1)) ||
      (output_norm_weight && (output_norm_weight->dim() != 1 || output_norm_weight->numel() != n)))
    return Error::shape_mismatch;
  // Its rows' and sources' strides are the API's to refuse (whole packs); its start is here.
  if (reinterpret_cast<uintptr_t>(blocks.data_ptr()) %
          hip_custom_communicator::kPackBytes != 0)
    return Error::not_aligned;
  // The sources read are the first num_blocks; block_output may be a later slot of the same stack.
  const torch::Tensor sources = blocks.narrow(1, 0, num_blocks);
  if (overlapping({&input, prefix ? &*prefix : nullptr, &sources, &norm_weight, &qk_weight,
                   output_norm_weight ? &*output_norm_weight : nullptr},
                  {&prefix_output, &output, block_output ? &*block_output : nullptr}))
    return Error::inputs_overlap_outputs;
  return std::nullopt;
}

}  // namespace

// =================================================================================================
// 3. THE API: a function a torch op, what Python passes in, its Error's number (or none) out.
// =================================================================================================

// WHAT THE BUILD HOLDS, kBuild's projection for Python: its dtypes by name, its
// worlds, a pack's bytes, its ops, errors and fault places by
// name in their enums' order (each error's name without its reason), which
// Python's `Op`, `Error` and `Where` are held to, its templates, and each op's
// plain call, which Python's `should` admits.
BuildInfoWire hip_custom_communicator_build_info() {
  using namespace hip_custom_communicator;
  Names dtypes, ops, errors, where_names;
  for (const at::ScalarType d : kDtypes) dtypes.push_back(to_string(d));
  const std::vector<int64_t> worlds(kWorlds.begin(),
                                    kWorlds.end());
  for (int i = 0; i < kNumOps; ++i)
    ops.push_back(to_string(static_cast<OpType>(i)));
  for (int i = 0; i < kNumErrors; ++i) {
    const std::string s = to_string(static_cast<Error>(i));
    errors.push_back(s.substr(0, s.find(':')));
  }
  for (int i = 1; i <= kWhereCount; ++i)
    where_names.push_back(to_string(static_cast<Where>(i)));
  // Each op's kernel, and its builds (what dispatch instantiates: the tuner's
  // search space, each one a forcing can name), flat: a build's fields in
  // `fields`' order, 0 where its op's BuildConfig has none (every field it has
  // is set), its quantization its number; its
  // block a field a dimension; and its algorithm's name, a build each (`algorithms`). The
  // launch is not a build's: a forced call gives its own grid.
  const Names fields = {"tile_m",  "tile_n",  "tile_k",       "block_x",      "block_y",
                        "block_z", "waves_per_eu", "quantization", "pack_bytes"};
  Names templates, template_ops, algorithms;
  std::vector<int64_t> configs, counts;
  for (const Op& o : kOps) {
    templates.push_back(o.kernel);
    template_ops.push_back(o.name);
    for (const BuildConfig& b : o.builds) {
      algorithms.emplace_back(algorithm_name(b));
      std::visit(
          [&](const auto& f) {
            int64_t k = 1;  // a kernel with no K: its identity
            if constexpr (requires { f.tile_k; }) k = f.tile_k;
            configs.insert(configs.end(),
                           {f.tile_m, f.tile_n, k, f.block.x, f.block.y, f.block.z,
                            f.waves_per_eu, f.quantization, f.pack_bytes});
          },
          b);
    }
    counts.push_back(static_cast<int64_t>(o.builds.size()));
  }
  // EACH OP'S PLAIN CALL (kOps, config.cuh), in `ops`' order: its dtypes and
  // its weight's besides its input's, flat, and four ints an op: those two
  // counts and two_d. What a call may hold is a communicator's (`hip_custom_communicator_capacity`),
  // not the build's.
  Names op_dtypes, op_weights;
  std::vector<int64_t> op_facts;
  for (const Op& o : kOps) {
    for (const at::ScalarType d : o.dtypes) op_dtypes.push_back(to_string(d));
    for (const at::ScalarType d : o.weights) op_weights.push_back(to_string(d));
    op_facts.insert(op_facts.end(), {static_cast<int64_t>(o.dtypes.size()),
                                     static_cast<int64_t>(o.weights.size()),
                                     o.two_d});
  }
  return {dtypes,     worlds,       kPackBytes,
          ops,        errors,       where_names,
          templates,  template_ops, fields,
          configs,    counts,       op_dtypes,
          op_weights, op_facts,   kUnfencedBlocks,
          algorithms};
}

// =================================================================================================
// THE STATE, AS A HANDLE: the CommsState, made by `open` and handed to Python as its address (an
// int, as vLLM's custom all-reduce hands its own), passed back to every call, and freed by `free`.
// Each returns its Error's number, or none.
// =================================================================================================

// OPEN: the state of a communicator at `rank` of `world` on `device`, serving `max_rows` rows of
// at most `max_cols` columns, a wait giving up after `timeout_seconds`: (the Error it is refused
// for, or none; the handle, 0 where refused; this rank's arena's IPC handle, its bytes; the
// device's arch; the capacity: max_rows, max_cols, the partials' and staging's bytes, then each
// op's most bytes a call). A device the build does not run on is refused, nothing allocated: the
// caller does not run here.
OpenWire hip_custom_communicator_open(int64_t rank, int64_t world, int64_t device,
                                      int64_t max_rows, int64_t max_cols,
                                      double timeout_seconds) {
  using namespace hip_custom_communicator;
  // ---- PRECONDITIONS: the call's values, the world built, the device the build's target with
  // our kernels' code object on it (a kernel's attributes answer exactly when it does). ----
  if (!all_ints({rank, world, device})) return {error_code(Error::field_out_of_range), 0, {}, "", {}};
  if (rank < 0 || rank >= world || !world_built(static_cast<int>(world)))
    return {error_code(Error::world_not_built), 0, {}, "", {}};
  if (max_rows < 1 || max_cols < 1) return {error_code(Error::field_not_positive), 0, {}, "", {}};
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, static_cast<int>(device)) != hipSuccess) {
    (void)hipGetLastError();
    return {error_code(Error::device_not_built), 0, {}, "", {}};
  }
  const std::string name = prop.gcnArchName;
  const std::string arch = name.substr(0, name.find(':'));  // gfx950:sramecc+:xnack-
  if (hipSetDevice(static_cast<int>(device)) != hipSuccess)
    return {error_code(Error::hip_failed), 0, {}, "", {}};
  hipFuncAttributes attrs;
  const hipError_t found = hipFuncGetAttributes(
      &attrs, reinterpret_cast<const void*>(
                  all_reduce<AllReduceAlgorithm::one_shot_pull, false, c10::BFloat16, 2, 1,
                             kWaveSize * traits<c10::BFloat16, kPackBytes>::N, kWaveSize,
                             kWaveSize, kPackBytes, 1, false, false>));
  (void)hipGetLastError();
  if (found != hipSuccess) return {error_code(Error::device_not_built), 0, {}, "", {}};
  if (arch != kTargetArch) return {error_code(Error::device_not_tuned), 0, {}, "", {}};
  int khz = 0;
  if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, static_cast<int>(device)) !=
      hipSuccess)
    return {error_code(Error::hip_failed), 0, {}, "", {}};

  // ---- THE EXCHANGE, SIZED FROM THE PROBLEM (STANDARDS, *A kernel's own buffers are sized from
  // the problem it serves*): each region holds its largest payload at the largest call, in every
  // built dtype and either encoding (exact or INT4: common.cuh's INT4 payload layout): the staging
  // the whole input, the pushed two-shot's whole-call inbox, or their INT4 payloads; the partials
  // the most any two-shot keeps (`partials_need`), or its INT4 payload. Each grows with the rows
  // and the row's width, so a call inside the box fits by construction. Each a whole pack, at
  // least one. ----
  const int64_t pack = kPackBytes;
  const auto whole = [pack](int64_t bytes) {
    return bytes < pack ? pack : (bytes + pack - 1) / pack * pack;
  };
  int64_t partials_bytes = 0, staging_bytes = int4_payload_bytes(max_rows, max_cols);
  for (const at::ScalarType d : kDtypes) {
    const int64_t packs = (max_cols * elem_bytes(d) + pack - 1) / pack;
    every_algorithm([&](OpType, auto a) {
      const int64_t need = partials_need(a, max_rows, packs, static_cast<int>(world), pack);
      if (need > partials_bytes) partials_bytes = need;
      const int64_t int4 = int4_payload_bytes(1, need / elem_bytes(d));
      if (need > 0 && int4 > partials_bytes) partials_bytes = int4;
    });
    staging_bytes = std::max(staging_bytes, max_rows * max_cols * elem_bytes(d));
    // THE EXACT PUSHED TWO-SHOT'S INBOX, the whole call: WORLD slots of a share each.
    staging_bytes = std::max(staging_bytes, world * ((max_rows * packs + world - 1) / world) * pack);
  }
  // THE TWO-SHOTS' INT4 INBOX (the flat walk): WORLD shares, each rounded up to whole groups.
  staging_bytes = std::max(
      staging_bytes,
      int4_payload_bytes(1, world * int4_flat_share(max_rows, max_cols, static_cast<int>(world))));
  partials_bytes = whole(partials_bytes);
  staging_bytes = whole(staging_bytes);

  // ---- THE STATE AND ITS BUFFERS: the arena's regions back to back (4 KiB aligned each), the
  // private buffer, the status; every one zeroed before any stream can use it. ----
  auto* s = new CommsState();
  s->rank = static_cast<int>(rank);
  s->world = static_cast<int>(world);
  s->device = static_cast<int>(device);
  s->max_rows = max_rows;
  s->timeout_ticks = static_cast<uint64_t>(timeout_seconds * khz * 1000);
  s->staging_bytes = staging_bytes;
  s->partials_bytes = partials_bytes;
  const auto aligned = [](int64_t bytes) { return (bytes + 4095) / 4096 * 4096; };
  s->signals = 0;
  s->staging = s->signals + kSignalsRegionBytes;
  s->partials = aligned(s->staging + staging_bytes);
  s->scales = s->partials + partials_bytes;
  s->unfenced = aligned(s->scales + max_rows * int64_t{sizeof(float)});
  s->arena_bytes = aligned(s->unfenced + unfenced_bytes(s->world));
  void* shared = nullptr;
  void* priv = nullptr;
  void* status = nullptr;
  const bool made =
      hipExtMallocWithFlags(&shared, static_cast<size_t>(s->arena_bytes), kArenaMallocFlags) ==
          hipSuccess &&
      hipMalloc(&priv, static_cast<size_t>(kPrivateBytes)) == hipSuccess &&
      hipMalloc(&status, kStatusFields * sizeof(uint32_t)) == hipSuccess;
  s->shared_buffer = static_cast<char*>(shared);
  s->private_buffer = static_cast<char*>(priv);
  s->status_device = static_cast<uint32_t*>(status);
  hipIpcMemHandle_t h{};
  const bool ready =
      made && hipMemset(shared, 0, static_cast<size_t>(s->arena_bytes)) == hipSuccess &&
      hipMemset(priv, 0, static_cast<size_t>(kPrivateBytes)) == hipSuccess &&
      hipMemset(status, 0, kStatusFields * sizeof(uint32_t)) == hipSuccess &&
      hipDeviceSynchronize() == hipSuccess && hipIpcGetMemHandle(&h, shared) == hipSuccess;
  if (!ready) {
    (void)hipGetLastError();
    (void)hip_custom_communicator_free(reinterpret_cast<int64_t>(s));
    return {error_code(Error::hip_failed), 0, {}, "", {}};
  }

  // ---- THE ANSWER: the handle, this arena's IPC handle's bytes, the arch, the capacity: the box,
  // the regions, and each op's most bytes a call (its partials a call's bytes at most, its staging
  // the whole call). ----
  std::vector<int64_t> handle;
  for (size_t i = 0; i < sizeof(h); ++i)
    handle.push_back(static_cast<uint8_t>(reinterpret_cast<const char*>(&h)[i]));
  std::vector<int64_t> capacity{max_rows, max_cols, partials_bytes, staging_bytes};
  for (const auto& o : kOps) {
    int64_t per_byte = 0;
    every_algorithm([&](OpType of, auto a) {
      if (of != o.type) return;
      per_byte = std::max(per_byte, partials_need(a, 1, 1, kMaxRanks, kPackBytes) / kPackBytes);
    });
    capacity.push_back(per_byte == 0 ? staging_bytes
                                     : std::min(partials_bytes / per_byte, staging_bytes));
  }
  return {std::nullopt, reinterpret_cast<int64_t>(s), handle, arch, capacity};
}

// CONNECT: every rank's arena opened from its IPC handle (`handles`, their bytes, rank after rank;
// this rank's ignored), then the private buffer written: each region's array of every rank's
// address of it (0 past the world), and the eager input table, slot 0: every rank's staging.
ErrorCode hip_custom_communicator_connect(int64_t state, const std::vector<int64_t>& handles) {
  using namespace hip_custom_communicator;
  if (state == 0) return error_code(Error::not_open);
  CommsState& s = *reinterpret_cast<CommsState*>(state);
  if (handles.size() != s.world * sizeof(hipIpcMemHandle_t))
    return error_code(Error::world_not_built);
  s.peers.assign(s.world, nullptr);
  for (int r = 0; r < s.world; ++r) {
    if (r == s.rank) {
      s.peers[r] = s.shared_buffer;
      continue;
    }
    hipIpcMemHandle_t h;
    for (size_t i = 0; i < sizeof(h); ++i)
      reinterpret_cast<char*>(&h)[i] = static_cast<char>(handles[r * sizeof(h) + i]);
    void* base = nullptr;
    if (hipIpcOpenMemHandle(&base, h, hipIpcMemLazyEnablePeerAccess) != hipSuccess) {
      (void)hipGetLastError();
      return error_code(Error::hip_failed);
    }
    s.peers[r] = static_cast<char*>(base);
  }
  const int64_t offsets[kRegionPtrArrays] = {s.signals, s.staging, s.partials, s.scales,
                                             s.unfenced};
  std::vector<char*> arrays(kRegionPtrArrays * kMaxRanks, nullptr);
  for (int a = 0; a < kRegionPtrArrays; ++a)
    for (int r = 0; r < s.world; ++r) arrays[a * kMaxRanks + r] = s.peers[r] + offsets[a];
  std::vector<char*> eager(kMaxRanks, nullptr);
  for (int r = 0; r < s.world; ++r) eager[r] = s.peers[r] + s.staging;
  if (hipMemcpy(s.private_buffer + kPrivateRegionPtrArrays, arrays.data(),
                arrays.size() * sizeof(char*), hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(s.private_buffer + kPrivateTables, eager.data(), eager.size() * sizeof(char*),
                hipMemcpyHostToDevice) != hipSuccess) {
    (void)hipGetLastError();
    return error_code(Error::hip_failed);
  }
  char* const* const region_ptr_arrays =
      reinterpret_cast<char* const*>(s.private_buffer + kPrivateRegionPtrArrays);
  s.signals_ptr_array =
      reinterpret_cast<uint32_t* const*>(region_ptr_arrays + kSignalsArray * kMaxRanks);
  s.staging_ptr_array = region_ptr_arrays + kStagingArray * kMaxRanks;
  s.partials_ptr_array = region_ptr_arrays + kPartialsArray * kMaxRanks;
  s.scales_ptr_array = reinterpret_cast<float* const*>(region_ptr_arrays + kScalesArray * kMaxRanks);
  s.unfenced_ptr_array = region_ptr_arrays + kUnfencedArray * kMaxRanks;
  s.unfenced_epochs = reinterpret_cast<int32_t*>(s.private_buffer + kPrivateUnfencedEpochs);
  return std::nullopt;
}

// THE CAPTURED LAUNCHES' INPUTS FOR THE PEERS (vLLM's get_graph_buffer_ipc_meta): (the Error or
// none; each pending input's allocation's IPC handle's bytes, an input after another, in the order
// reserved; each input's offset in its allocation).
GraphMetaWire hip_custom_communicator_graph_ipc_meta(int64_t state) {
  using namespace hip_custom_communicator;
  if (state == 0) return {error_code(Error::not_open), {}, {}};
  const CommsState& s = *reinterpret_cast<const CommsState*>(state);
  std::vector<int64_t> bytes, offsets;
  for (const auto& [slot, input] : s.pending) {
    void* base = nullptr;
    hipIpcMemHandle_t h;
    if (hipPointerGetAttribute(&base, HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                               reinterpret_cast<hipDeviceptr_t>(const_cast<void*>(input))) !=
            hipSuccess ||
        hipIpcGetMemHandle(&h, base) != hipSuccess) {
      (void)hipGetLastError();
      return {error_code(Error::hip_failed), {}, {}};
    }
    for (size_t i = 0; i < sizeof(h); ++i)
      bytes.push_back(static_cast<uint8_t>(reinterpret_cast<const char*>(&h)[i]));
    offsets.push_back(static_cast<const char*>(input) - static_cast<const char*>(base));
  }
  return {std::nullopt, bytes, offsets};
}

// THE CAPTURED LAUNCHES' TABLES WRITTEN (vLLM's register_graph_buffers): every rank's
// `graph_ipc_meta` (`handles`' bytes and `offsets`, a rank's after another's, each as many as this
// rank has pending), each peer allocation opened once (cached by its handle), every pending slot
// written, none pending after. Ranks that captured different counts are refused (ranks_disagree).
ErrorCode hip_custom_communicator_register_graph_buffers(int64_t state,
                                                       const std::vector<int64_t>& handles,
                                                       const std::vector<int64_t>& offsets) {
  using namespace hip_custom_communicator;
  if (state == 0) return error_code(Error::not_open);
  CommsState& s = *reinterpret_cast<CommsState*>(state);
  const size_t count = s.pending.size();
  if (offsets.size() != count * s.world ||
      handles.size() != offsets.size() * sizeof(hipIpcMemHandle_t))
    return error_code(Error::ranks_disagree);
  const auto tables = s.private_buffer + kPrivateTables;
  for (size_t i = 0; i < count; ++i) {
    const auto& [slot, input] = s.pending[i];
    std::vector<const void*> table(kMaxRanks, nullptr);
    for (int r = 0; r < s.world; ++r) {
      if (r == s.rank) {
        table[r] = input;
        continue;
      }
      const size_t at = r * count + i;
      std::vector<char> key(handles.begin() + at * sizeof(hipIpcMemHandle_t),
                            handles.begin() + (at + 1) * sizeof(hipIpcMemHandle_t));
      auto it = s.opened_inputs.find(key);
      if (it == s.opened_inputs.end()) {
        hipIpcMemHandle_t h;
        std::memcpy(&h, key.data(), sizeof(h));
        void* base = nullptr;
        if (hipIpcOpenMemHandle(&base, h, hipIpcMemLazyEnablePeerAccess) != hipSuccess) {
          (void)hipGetLastError();
          return error_code(Error::hip_failed);
        }
        it = s.opened_inputs.emplace(key, static_cast<char*>(base)).first;
      }
      table[r] = it->second + offsets[at];
    }
    if (hipMemcpy(tables + slot * kMaxRanks * int64_t{sizeof(void*)}, table.data(),
                  table.size() * sizeof(void*), hipMemcpyHostToDevice) != hipSuccess) {
      (void)hipGetLastError();
      return error_code(Error::hip_failed);
    }
  }
  s.pending.clear();
  return std::nullopt;
}

// CLOSE THE PEERS: every BORROWED mapping closed (the peers' arenas, their captured inputs), every
// one tried: the first failure's Error. Every launch finished first, and every rank stopped (close
// is a collective).
ErrorCode hip_custom_communicator_close_peers(int64_t state) {
  using namespace hip_custom_communicator;
  if (state == 0) return error_code(Error::not_open);
  CommsState& s = *reinterpret_cast<CommsState*>(state);
  std::optional<Error> first;
  std::vector<char*> borrowed;
  for (int r = 0; r < static_cast<int>(s.peers.size()); ++r)
    if (r != s.rank && s.peers[r] != nullptr) borrowed.push_back(s.peers[r]);
  for (const auto& [key, base] : s.opened_inputs) borrowed.push_back(base);
  for (char* base : borrowed)
    if (hipIpcCloseMemHandle(base) != hipSuccess) {
      (void)hipGetLastError();
      first = first.value_or(Error::hip_failed);
    }
  s.peers.clear();
  s.opened_inputs.clear();
  return error_code(first);
}

// FREE: this rank's OWNED buffers freed, every one tried (the first failure's Error), and the
// state with them. Once no peer has them open (after every rank's close_peers).
ErrorCode hip_custom_communicator_free(int64_t state) {
  using namespace hip_custom_communicator;
  if (state == 0) return error_code(Error::not_open);
  auto* s = reinterpret_cast<CommsState*>(state);
  std::optional<Error> first;
  for (void* p : {static_cast<void*>(s->shared_buffer), static_cast<void*>(s->private_buffer),
                  static_cast<void*>(s->status_device)})
    if (p != nullptr && hipFree(p) != hipSuccess) {
      (void)hipGetLastError();
      first = first.value_or(Error::hip_failed);
    }
  delete s;
  return error_code(first);
}

// all_reduce: at its config.
// One function: what Python passes, its tensors stripped to pointers and strides, the
// preconditions, the kernel's arguments, the launch, the postcondition.
ErrorCode hip_custom_communicator_all_reduce(int64_t state, torch::Tensor& input,
    torch::Tensor& output, const std::string& algorithm_name, int64_t grid_x, int64_t grid_y,
    int64_t grid_z, int64_t block_x, int64_t block_y, int64_t block_z, int64_t waves_per_eu,
    int64_t tile_m, int64_t tile_n, const std::string& quantization_name, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  using namespace hip_custom_communicator;
  // THE CALL FROM TORCH: its state, its tensors, its config's ints.
  if (state == 0) return error_code(Error::not_open);
  if (!on_device({&input, &output})) return error_code(Error::not_on_device);
  if (output.sizes() != input.sizes()) return error_code(Error::shape_mismatch);
  if (output.scalar_type() != input.scalar_type()) return error_code(Error::dtype_mismatch);
  if (overlapping({&input}, {&output})) return error_code(Error::inputs_overlap_outputs);
  const std::optional<torch::Tensor> input_rows = as_rows(input),
                                     output_rows = as_rows(output);
  if (!input_rows || !output_rows)
    return error_code(Error::not_contiguous);
  if (!all_ints({waves_per_eu, tile_m, tile_n, pack_bytes, grid_x, grid_y, grid_z, block_x, block_y, block_z})) return error_code(Error::field_out_of_range);
  CommsState& s = *reinterpret_cast<CommsState*>(state);
  const void* input_ptr = input_rows->data_ptr();
  const int64_t input_stride_m = input_rows->stride(0);
  const int64_t input_stride_n = input_rows->stride(1);
  const int64_t input_size_m = input_rows->size(0);
  const int64_t input_size_n = input_rows->size(1);
  void* output_ptr = output_rows->data_ptr();
  const int64_t output_stride_m = output_rows->stride(0);
  const int64_t output_stride_n = output_rows->stride(1);
  const int64_t output_size_m = output_rows->size(0);
  const int64_t output_size_n = output_rows->size(1);
  const at::ScalarType dtype = input.scalar_type();
  const Dim3 grid{static_cast<int>(grid_x), static_cast<int>(grid_y), static_cast<int>(grid_z)};
  const Dim3 block{static_cast<int>(block_x), static_cast<int>(block_y), static_cast<int>(block_z)};
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  // ---- PRECONDITIONS: the config, the call, then its instance. The first Error refuses the
  // call; past this section the call runs. ----
  const std::optional<AllReduceAlgorithm> named = algorithm_named<AllReduceBuildConfig>(algorithm_name);
  if (!named) return error_code(Error::no_such_kernel);
  const AllReduceAlgorithm algorithm = *named;
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_code(Error::no_such_quantization);
  const AllReduceBuildConfig build{.algorithm = algorithm,
      .block = block,
      .waves_per_eu = static_cast<int>(waves_per_eu),
      .tile_m = static_cast<int>(tile_m),
      .tile_n = static_cast<int>(tile_n),
      .quantization = quantization_name == "int4" ? Quantization::int4 : Quantization::exact,
      .pack_bytes = static_cast<int>(pack_bytes)};
  const int64_t m = input_size_m, n = input_size_n;  // the problem's shape: the input's
  if (const std::optional<Error> e = refused(true, s.world, s.staging_bytes, s.partials_bytes, build,
                                             grid, quantize_from_bytes, dtype, m, n))
    return error_code(e);
  const int64_t quantize_from = quantize_from_bytes.value_or(0);
  if (const std::optional<Error> e =
          layout_refused(dtype, build.pack_bytes, {{output_stride_m, output_stride_n},
                                 {input_stride_m, input_stride_n}}))
    return error_code(e);
  // QUANTIZE_INT4, WHAT THIS LAUNCH MOVES: INT4 where the config's quantization is int4 and the
  // call's bytes reach its quantize_from_bytes, else exact (the kernel's one template parameter).
  const bool quantize_int4 = build.quantization == Quantization::int4 &&
                           input_size_m * input_size_n * elem_bytes(dtype) >=
                               quantize_from;
  const int q = quantize_int4 ? Quantization::int4 : Quantization::exact;
  // IN PLACE: the exact pull, every rank's input read where its table says.
  const bool in_place = !is_push(algorithm) && q == Quantization::exact;
  // THE UNFENCED REGION: the pushed one-shot pushes lines into its own inbox there, a tile a
  // block a pass, nothing staged. ITS GRID IS THE ONE GIVEN: a block its own slots, so at most the
  // unfenced region's blocks (more is refused, never cut); a block past the tiles has no pass, and leaves its
  // epoch as it was.
  const bool unfenced = algorithm == AllReduceAlgorithm::one_shot_push;
  if (unfenced && volume(grid) > kUnfencedBlocks) return error_code(Error::launch_out_of_range);
  // THE WHOLE CALL IN ONE PASS (the regions are sized from the largest call; refused() refuses a
  // call past them). The two-shots' exact share is the flat walk's, whole packs, rounded up: a
  // rank's slice of a row in the partials, and the pushed two-shot's inbox [WORLD, slot] in the
  // staging, its slot a share of the whole call.
  const int64_t pack = build.pack_bytes;  // the build's
  const int64_t packs = n * elem_bytes(dtype) / pack;
  const int64_t slice_bytes = (packs + s.world - 1) / s.world * pack;
  const int64_t slot = (m * packs + s.world - 1) / s.world * pack / elem_bytes(dtype);
  // THE EXACT PUSHED TWO-SHOT'S INBOX, its slots rounded up to whole packs.
  const bool inbox = !quantize_int4 && algorithm == AllReduceAlgorithm::two_shot_push;
  if (inbox && s.world * slot * elem_bytes(dtype) > s.staging_bytes) return error_code(Error::staging_too_small);
  const int shared_memory_bytes = 0;
  const bool contiguous = input_stride_m == n * input_stride_n &&
      output_stride_m == n * output_stride_n;
  // AN INT4 ROW WITH A TAIL runs the build's TAIL instance (int4_tail).
  const bool tail = quantize_int4 && int4_tail(algorithm, m, n, s.world);
  const void* kernel = nullptr;
  if (!with_instance<OpType::all_reduce>(build, s.world, dtype, dtype, contiguous, /*chunked=*/false, tail, quantize_int4, [&]<Instance I>() { kernel = kernel_pointer<I>(); }))
    return error_code(Error::tile_not_built);
  if (const std::optional<Error> e = check_kernel(kernel, grid, block, shared_memory_bytes, true))
    return error_code(e);

  // ---- THE LAUNCH: the buffers, the views of their regions, and the instance. ----
  // ---- ITS VIEWS OF THE REGIONS, by its algorithm, in elements: a view [m, n] at strides
  // [stride_m, stride_n] (the line push's inbox [slot, m, n]); an empty view (size 0) where it
  // reads none. The kernels find a one-shot's payload in the staging from the input's sizes. ----
  const int64_t e = elem_bytes(dtype);
  const int64_t share = int4_flat_share(m, n, s.world);  // the INT4 flat walk's share
  int64_t partials_stride_m = slice_bytes / e, partials_stride_n = 1, partials_size_m = 0,
          partials_size_n = slice_bytes / e;
  int64_t staging_stride_m = 0, staging_stride_n = 0, staging_size_m = 0, staging_size_n = 0;
  int64_t unfenced_stride_slot = 0, unfenced_stride_m = 0, unfenced_stride_n = 0,
          unfenced_size_slot = 0, unfenced_size_m = 0, unfenced_size_n = 0;
  switch (algorithm) {
    case AllReduceAlgorithm::two_shot_pull:
    case AllReduceAlgorithm::two_shot_push:
      if (quantize_int4) {
        // THE INT4 FLAT WALK: this rank's share of the call's groups in the partials, and every
        // rank's share in the staging.
        partials_stride_m = share, partials_size_m = 1, partials_size_n = share;
        staging_stride_m = s.world * share, staging_stride_n = 1, staging_size_m = 1,
        staging_size_n = s.world * share;
      } else {
        // EXACT: this rank's slice of every row in the partials; the pushed two-shot's inbox in
        // the staging, WORLD slots of a share of the call (the pull reads every rank's input in
        // place, through the input table).
        partials_size_m = m;
        if (algorithm == AllReduceAlgorithm::two_shot_push)
          staging_stride_m = slot, staging_stride_n = 1, staging_size_m = s.world,
          staging_size_n = slot;
      }
      break;
    case AllReduceAlgorithm::one_shot_pull:
      // INT4: its [m, n] payload in the staging, no partials (exact: every rank's input in place).
      if (quantize_int4) {
        partials_stride_m = 0, partials_size_n = 0;
        staging_stride_n = 1, staging_size_m = 1,
        staging_size_n = (int4_payload_bytes(m, n) + e - 1) / e;
      }
      break;
    case AllReduceAlgorithm::one_shot_push:
      // THE UNFENCED REGION: [writer, row, n] payload elements, 2 * kUnfencedBlocks rows (bank,
      // row) of kUnfencedSlotBytes of payload each whatever the build (so a row is the same bytes
      // at every build, its epoch its own), a writer's rows contiguous.
      unfenced_stride_slot = 2 * kUnfencedBlocks * (kUnfencedSlotBytes / e);
      unfenced_stride_m = kUnfencedSlotBytes / e, unfenced_stride_n = 1;
      unfenced_size_slot = s.world, unfenced_size_m = 2 * kUnfencedBlocks,
      unfenced_size_n = kUnfencedSlotBytes / e;
      break;
    case AllReduceAlgorithm::one_shot_push_staged: {
      // WORLD SLOTS OF AN INPUT in the staging (INT4: its payload, a slot whole kInt4ScalesAlign),
      // no partials.
      const int64_t payload = quantize_int4 ? int4_payload_bytes(m, n) : m * n * e;
      const int64_t slot_bytes =
          quantize_int4 ? (payload + kInt4ScalesAlign - 1) / kInt4ScalesAlign * kInt4ScalesAlign
                        : payload;
      partials_stride_m = 0, partials_size_n = 0;
      staging_stride_n = 1, staging_size_m = 1,
      staging_size_n = (s.world * slot_bytes + e - 1) / e;
      break;
    }
  }
  // EVERY VIEW INSIDE ITS REGION (view_fits).
  if (!view_fits(1, 0, partials_size_m, partials_stride_m, partials_size_n, partials_stride_n, elem_bytes(dtype), s.partials_bytes) ||
      !view_fits(1, 0, staging_size_m, staging_stride_m, staging_size_n, staging_stride_n, elem_bytes(dtype), s.staging_bytes) ||
      !view_fits(unfenced_size_slot, unfenced_stride_slot, unfenced_size_m, unfenced_stride_m, unfenced_size_n, unfenced_stride_n, elem_bytes(dtype) * (kLineBytes / kLinePayloadBytes), unfenced_bytes(s.world)) ||
      !view_fits(1, 0, int64_t{kUnfencedBlocks}, kUnfencedEpochsStrideBlock, kUnfencedEpochsStrideBlock, int64_t{1}, 4, kUnfencedEpochsBytes))
    return error_code(Error::view_past_region);
  // THE INPUT TABLE (INPUT TABLES): where the kernel reads every rank's input in place (an exact
  // pull), or none where it reads the staging.
  void* const* table = nullptr;
  int64_t table_stride_m = staging_stride_m;
  if (in_place)
    if (const std::optional<Error> e = input_table(s, input_ptr, input_stride_m, input_size_m,
                                                   input_size_n, elem_bytes(dtype), stream, table,
                                                   table_stride_m))
      return error_code(e);
  const int table_size = in_place ? s.world : 0;
  const int64_t table_stride_n = in_place ? 1 : staging_stride_n;
  with_instance<OpType::all_reduce>(build, s.world, dtype, dtype, contiguous, /*chunked=*/false, tail, quantize_int4, [&]<Instance I>() {
    constexpr auto B = I.build;
    using T = c10::impl::ScalarTypeToCPPTypeT<I.dtype>;
    using W = c10::impl::ScalarTypeToCPPTypeT<I.weight_dtype>;
    all_reduce<B.algorithm, I.quantize_int4, T, I.world, B.tile_m, B.tile_n, B.block.x,
        I.wave_size, B.pack_bytes, B.waves_per_eu, I.contiguous, I.tail>
        <<<dim3_of(grid), dim3_of(B.block), shared_memory_bytes, stream>>>(
        static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
        static_cast<int>(input_size_m), static_cast<int>(input_size_n),
        reinterpret_cast<const T* const*>(table), table_size, table_stride_m, table_stride_n,
        static_cast<T*>(output_ptr), output_stride_m, output_stride_n,
        static_cast<int>(output_size_m), static_cast<int>(output_size_n),
        s.partials_ptr_array, s.world, partials_stride_m,
        partials_stride_n, static_cast<int>(partials_size_m),
        static_cast<int>(partials_size_n), s.staging_ptr_array,
        s.world, staging_stride_m, staging_stride_n,
        static_cast<int>(staging_size_m), static_cast<int>(staging_size_n),
        s.unfenced_ptr_array, s.world, unfenced_stride_slot,
        unfenced_stride_m, unfenced_stride_n,
        static_cast<int>(unfenced_size_slot), static_cast<int>(unfenced_size_m),
        static_cast<int>(unfenced_size_n), s.unfenced_epochs,
        kUnfencedEpochsStrideBlock, int64_t{1},
        static_cast<int>(kUnfencedBlocks),
        static_cast<int>(kUnfencedEpochsStrideBlock), s.signals_ptr_array,
        s.world, kSignalsStrideParity, int64_t{kMaxRanks},
        int64_t{1}, int64_t{2}, int64_t{kMaxBlocks},
        int64_t{kMaxRanks}, s.status_device, int64_t{1},
        int64_t{kStatusFields}, s.rank, s.timeout_ticks);
  });
  // POSTCONDITION: the launch went in.
  if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
  return std::nullopt;
}  // namespace

// all_reduce_rms_norm: at its config.
// One function: what Python passes, its tensors stripped to pointers and strides, the
// preconditions, the kernel's arguments, the launch, the postcondition.
ErrorCode hip_custom_communicator_all_reduce_rms_norm(int64_t state,
    torch::Tensor& input, torch::Tensor& weight, torch::Tensor& output, double eps,
    const std::string& algorithm_name, int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t block_x,
    int64_t block_y, int64_t block_z, int64_t waves_per_eu, int64_t tile_m, int64_t tile_n,
    const std::string& quantization_name, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  using namespace hip_custom_communicator;
  // THE CALL FROM TORCH: its state, its tensors, its config's ints.
  if (state == 0) return error_code(Error::not_open);
  if (const std::optional<Error> e = norm_tensors(input, nullptr, weight, output, nullptr, /*in_place=*/false)) return error_code(e);
  if (!all_ints({waves_per_eu, tile_m, tile_n, pack_bytes, grid_x, grid_y, grid_z, block_x, block_y, block_z})) return error_code(Error::field_out_of_range);
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
  const Dim3 grid{static_cast<int>(grid_x), static_cast<int>(grid_y), static_cast<int>(grid_z)};
  const Dim3 block{static_cast<int>(block_x), static_cast<int>(block_y), static_cast<int>(block_z)};
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  // ---- PRECONDITIONS: the config, the call, then its instance. The first Error refuses the
  // call; past this section the call runs. ----
  const std::optional<AllReduceRmsNormAlgorithm> named = algorithm_named<AllReduceRmsNormBuildConfig>(algorithm_name);
  if (!named) return error_code(Error::no_such_kernel);
  const AllReduceRmsNormAlgorithm algorithm = *named;
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_code(Error::no_such_quantization);
  const AllReduceRmsNormBuildConfig build{.algorithm = algorithm,
      .block = block,
      .waves_per_eu = static_cast<int>(waves_per_eu),
      .tile_m = static_cast<int>(tile_m),
      .tile_n = static_cast<int>(tile_n),
      .quantization = quantization_name == "int4" ? Quantization::int4 : Quantization::exact,
      .pack_bytes = static_cast<int>(pack_bytes)};
  const int64_t m = input_size_m, n = input_size_n;  // the problem's shape: the input's
  if (const std::optional<Error> e = refused(true, s.world, s.staging_bytes, s.partials_bytes, build,
                                             grid, quantize_from_bytes, dtype, m, n))
    return error_code(e);
  if (const std::optional<Error> e =
          layout_refused(dtype, build.pack_bytes, {{output_stride_m, output_stride_n},
                                 {input_stride_m, input_stride_n}}))
    return error_code(e);
  if (const std::optional<Error> e =
          layout_refused(weight_dtype, build.pack_bytes, {{0, weight_stride_n}}))
    return error_code(e);
  const OpType op = OpType::all_reduce_rms_norm;
  if (const std::optional<Error> e = weight_refused(op, dtype, weight_dtype)) return error_code(e);
  // THE PARTIALS'S ROWS, what partials_need sized it for (a one-shot keeps none).
  const int64_t partials_rows =
      n > 0 ? partials_need(algorithm, m,
                           n * elem_bytes(dtype) / build.pack_bytes, s.world,
                           build.pack_bytes) /
                  (n * elem_bytes(dtype))
            : 0;
  const int shared_memory_bytes = 0;
  const bool contiguous = input_stride_m == n * input_stride_n &&
      output_stride_m == n * output_stride_n;
  // A ROW WIDER THAN THE TILE runs the build's CHUNKED instance.
  const bool chunked = n > build.tile_n;
  // AN INT4 ROW WITH A TAIL runs the build's TAIL instance (int4_tail).
  const bool tail = build.quantization == Quantization::int4 && int4_tail(algorithm, m, n, s.world);
  const void* kernel = nullptr;
  if (!with_instance<OpType::all_reduce_rms_norm>(build, s.world, dtype, weight_dtype, contiguous, chunked, tail, /*quantize_int4=*/false, [&]<Instance I>() { kernel = kernel_pointer<I>(); }))
    return error_code(Error::tile_not_built);
  if (const std::optional<Error> e = check_kernel(kernel, grid, block, shared_memory_bytes, true))
    return error_code(e);

  // ---- THE LAUNCH: the buffers, the views of their regions, and the instance. ----
  // ITS VIEWS OF THE REGIONS, each [m, n] (the pushed one-shot's inbox [slot, m, n]), in
  // elements: zeros where it reads none.
  int64_t partials_stride_m = 0, partials_stride_n = 0, partials_size_m = 0, partials_size_n = 0, staging_stride_m = 0, staging_stride_n = 0, staging_size_m = 0, staging_size_n = 0;
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
  in_place = !is_push(algorithm) &&
      build.quantization == Quantization::exact;
  // EVERY VIEW INSIDE ITS REGION (view_fits).
  if (!view_fits(1, 0, partials_size_m, partials_stride_m, partials_size_n, partials_stride_n, elem_bytes(dtype), s.partials_bytes) ||
      !view_fits(1, 0, staging_size_m, staging_stride_m, staging_size_n, staging_stride_n, elem_bytes(dtype), s.staging_bytes))
    return error_code(Error::view_past_region);
  // THE INPUT TABLE (INPUT TABLES): where the kernel reads every rank's input in place (an exact
  // pull), or none where it reads the staging.
  void* const* table = nullptr;
  int64_t table_stride_m = staging_stride_m;
  if (in_place)
    if (const std::optional<Error> e = input_table(s, input_ptr, input_stride_m, input_size_m,
                                                   input_size_n, elem_bytes(dtype), stream, table,
                                                   table_stride_m))
      return error_code(e);
  const int table_size = in_place ? s.world : 0;
  const int64_t table_stride_n = in_place ? 1 : staging_stride_n;
  with_instance<OpType::all_reduce_rms_norm>(build, s.world, dtype, weight_dtype, contiguous, chunked, tail, /*quantize_int4=*/false, [&]<Instance I>() {
    constexpr auto B = I.build;
    using T = c10::impl::ScalarTypeToCPPTypeT<I.dtype>;
    using W = c10::impl::ScalarTypeToCPPTypeT<I.weight_dtype>;
    all_reduce_rms_norm<B.algorithm, I.quantize_int4, T, W, I.world, B.tile_n, B.block.x,
        I.wave_size, B.pack_bytes, B.waves_per_eu, I.chunked, I.tail>
        <<<dim3_of(grid), dim3_of(B.block), shared_memory_bytes, stream>>>(
        static_cast<const T*>(input_ptr),
        input_stride_m, input_stride_n, static_cast<int>(input_size_m),
        static_cast<int>(input_size_n), reinterpret_cast<const T* const*>(table),
            table_size, table_stride_m, table_stride_n, static_cast<const W*>(weight_ptr),
            weight_stride_n,
        static_cast<int>(weight_size_n), static_cast<T*>(output_ptr), output_stride_m,
        output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), s.partials_ptr_array, s.world,
         partials_stride_m,
        partials_stride_n, static_cast<int>(partials_size_m),
        static_cast<int>(partials_size_n), s.staging_ptr_array,
            s.world, staging_stride_m,
        staging_stride_n, static_cast<int>(staging_size_m),
        static_cast<int>(staging_size_n), s.signals_ptr_array,
            s.world, kSignalsStrideParity,
        int64_t{kMaxRanks}, int64_t{1},
        int64_t{2}, int64_t{kMaxBlocks},
        int64_t{kMaxRanks}, s.status_device, int64_t{1},
        int64_t{kStatusFields}, s.rank, s.timeout_ticks, eps);
  });
  // POSTCONDITION: the launch went in.
  if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
  return std::nullopt;
}

// all_reduce_add_rms_norm: at its config.
// all_reduce_add_rms_norm: out, residual_out =
// fused_add_rms_norm(all_reduce(inp), residual, weight), vLLM's roundings
// exactly.
// One function: what Python passes, its tensors stripped to pointers and strides, the
// preconditions, the kernel's arguments, the launch, the postcondition.
ErrorCode hip_custom_communicator_all_reduce_add_rms_norm(int64_t state,
    torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, torch::Tensor& output,
    torch::Tensor& residual_output, double eps, const std::string& algorithm_name, int64_t grid_x,
    int64_t grid_y, int64_t grid_z, int64_t block_x, int64_t block_y, int64_t block_z,
    int64_t waves_per_eu, int64_t tile_m, int64_t tile_n, const std::string& quantization_name,
    int64_t pack_bytes, std::optional<int64_t> quantize_from_bytes) {
  using namespace hip_custom_communicator;
  // THE CALL FROM TORCH: its state, its tensors, its config's ints.
  if (state == 0) return error_code(Error::not_open);
  if (const std::optional<Error> e = norm_tensors(input, &residual, weight, output, &residual_output, /*in_place=*/false)) return error_code(e);
  if (!all_ints({waves_per_eu, tile_m, tile_n, pack_bytes, grid_x, grid_y, grid_z, block_x, block_y, block_z})) return error_code(Error::field_out_of_range);
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
  const Dim3 grid{static_cast<int>(grid_x), static_cast<int>(grid_y), static_cast<int>(grid_z)};
  const Dim3 block{static_cast<int>(block_x), static_cast<int>(block_y), static_cast<int>(block_z)};
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  // ---- PRECONDITIONS: the config, the call, then its instance. The first Error refuses the
  // call; past this section the call runs. ----
  const std::optional<AllReduceAddRmsNormAlgorithm> named = algorithm_named<AllReduceAddRmsNormBuildConfig>(algorithm_name);
  if (!named) return error_code(Error::no_such_kernel);
  const AllReduceAddRmsNormAlgorithm algorithm = *named;
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_code(Error::no_such_quantization);
  const AllReduceAddRmsNormBuildConfig build{.algorithm = algorithm,
      .block = block,
      .waves_per_eu = static_cast<int>(waves_per_eu),
      .tile_m = static_cast<int>(tile_m),
      .tile_n = static_cast<int>(tile_n),
      .quantization = quantization_name == "int4" ? Quantization::int4 : Quantization::exact,
      .pack_bytes = static_cast<int>(pack_bytes)};
  const int64_t m = input_size_m, n = input_size_n;  // the problem's shape: the input's
  if (const std::optional<Error> e = refused(true, s.world, s.staging_bytes, s.partials_bytes, build,
                                             grid, quantize_from_bytes, dtype, m, n))
    return error_code(e);
  if (const std::optional<Error> e = layout_refused(
          dtype, build.pack_bytes, {{output_stride_m, output_stride_n},
                  {residual_output_stride_m, residual_output_stride_n},
                  {input_stride_m, input_stride_n},
                  {residual_stride_m, residual_stride_n}}))
    return error_code(e);
  if (const std::optional<Error> e =
          layout_refused(weight_dtype, build.pack_bytes, {{0, weight_stride_n}}))
    return error_code(e);
  const OpType op = OpType::all_reduce_add_rms_norm;
  if (const std::optional<Error> e = weight_refused(op, dtype, weight_dtype)) return error_code(e);
  // THE PARTIALS'S ROWS, what partials_need sized it for (a one-shot keeps none),
  // at its row stride: the exact pull two-shot's row is its scale's pack then the
  // row (partials_need); the INT4 build's rows are the row alone, its scales in
  // the scales region.
  const bool scale_slot = algorithm == AllReduceAddRmsNormAlgorithm::two_shot_pull &&
                          build.quantization == Quantization::exact;
  // The scale slot's row whole slots (scale_slot_row), so every row's scale is aligned at any pack.
  const int64_t slot = scale_slot ? kScaleSlotBytes / elem_bytes(dtype) : 0;
  const int64_t row = payload_row(algorithm, build.quantization, n);
  const int64_t partials_stride_m =
      scale_slot ? scale_slot_row(row * elem_bytes(dtype)) / elem_bytes(dtype) + slot : row;
  const int64_t partials_rows =
      n > 0 ? partials_need(algorithm, m,
                           n * elem_bytes(dtype) / build.pack_bytes, s.world, build.pack_bytes) /
                  (partials_stride_m * elem_bytes(dtype))
            : 0;
  const int shared_memory_bytes = 0;
  const bool contiguous = input_stride_m == n * input_stride_n &&
      output_stride_m == n * output_stride_n;
  // A ROW WIDER THAN THE TILE runs the build's CHUNKED instance.
  const bool chunked = n > build.tile_n;
  // AN INT4 ROW WITH A TAIL runs the build's TAIL instance (int4_tail).
  const bool tail = build.quantization == Quantization::int4 && int4_tail(algorithm, m, n, s.world);
  // THE INT4 PULL TWO-SHOT'S SCALES, a rank's rows of them, in the scales region (max_rows rows).
  if (algorithm == AllReduceAddRmsNormAlgorithm::two_shot_pull && !scale_slot &&
      (m + s.world - 1) / s.world > s.max_rows)
    return error_code(Error::partials_too_small);
  const void* kernel = nullptr;
  if (!with_instance<OpType::all_reduce_add_rms_norm>(build, s.world, dtype, weight_dtype, contiguous, chunked, tail, /*quantize_int4=*/false, [&]<Instance I>() { kernel = kernel_pointer<I>(); }))
    return error_code(Error::tile_not_built);
  if (const std::optional<Error> e = check_kernel(kernel, grid, block, shared_memory_bytes, true))
    return error_code(e);

  // ---- THE LAUNCH: the buffers, the views of their regions, and the instance. ----
  // ITS VIEWS OF THE REGIONS, each [m, n] (the pushed one-shot's inbox [slot, m, n]), in
  // elements: zeros where it reads none.
  int64_t partials_stride_n = 0, partials_size_m = 0, partials_size_n = 0, staging_stride_m = 0, staging_stride_n = 0, staging_size_m = 0, staging_size_n = 0, scales_stride_m = 0, scales_size_m = 0;
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
  in_place = !is_push(algorithm) &&
      build.quantization == Quantization::exact;
  // EVERY VIEW INSIDE ITS REGION (view_fits).
  if (!view_fits(1, 0, partials_size_m, partials_stride_m, partials_size_n, partials_stride_n, elem_bytes(dtype), s.partials_bytes) ||
      !view_fits(1, 0, staging_size_m, staging_stride_m, staging_size_n, staging_stride_n, elem_bytes(dtype), s.staging_bytes) ||
      !view_fits(1, 0, scales_size_m, scales_stride_m, 1, 1, 4, s.max_rows * int64_t{sizeof(float)}))
    return error_code(Error::view_past_region);
  // THE INPUT TABLE (INPUT TABLES): where the kernel reads every rank's input in place (an exact
  // pull), or none where it reads the staging.
  void* const* table = nullptr;
  int64_t table_stride_m = staging_stride_m;
  if (in_place)
    if (const std::optional<Error> e = input_table(s, input_ptr, input_stride_m, input_size_m,
                                                   input_size_n, elem_bytes(dtype), stream, table,
                                                   table_stride_m))
      return error_code(e);
  const int table_size = in_place ? s.world : 0;
  const int64_t table_stride_n = in_place ? 1 : staging_stride_n;
  with_instance<OpType::all_reduce_add_rms_norm>(build, s.world, dtype, weight_dtype, contiguous, chunked, tail, /*quantize_int4=*/false, [&]<Instance I>() {
    constexpr auto B = I.build;
    using T = c10::impl::ScalarTypeToCPPTypeT<I.dtype>;
    using W = c10::impl::ScalarTypeToCPPTypeT<I.weight_dtype>;
    all_reduce_add_rms_norm<B.algorithm, I.quantize_int4, T, W, I.world, B.tile_n, B.block.x,
        I.wave_size, B.pack_bytes, B.waves_per_eu, I.chunked, I.tail>
        <<<dim3_of(grid), dim3_of(B.block), shared_memory_bytes, stream>>>(
        static_cast<const T*>(input_ptr),
        input_stride_m, input_stride_n, static_cast<int>(input_size_m),
        static_cast<int>(input_size_n), reinterpret_cast<const T* const*>(table),
            table_size, table_stride_m, table_stride_n, static_cast<const T*>(residual_ptr),
            residual_stride_m,
        residual_stride_n, static_cast<int>(residual_size_m),
        static_cast<int>(residual_size_n), static_cast<const W*>(weight_ptr), weight_stride_n,
        static_cast<int>(weight_size_n), static_cast<T*>(output_ptr), output_stride_m,
        output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), static_cast<T*>(residual_output_ptr),
        residual_output_stride_m, residual_output_stride_n,
        static_cast<int>(residual_output_size_m),
        static_cast<int>(residual_output_size_n), s.partials_ptr_array,
            s.world,
         partials_stride_m,
        partials_stride_n, static_cast<int>(partials_size_m),
        static_cast<int>(partials_size_n), s.scales_ptr_array,
        s.world, scales_stride_m, static_cast<int>(scales_size_m),
        s.staging_ptr_array, s.world, staging_stride_m,
        staging_stride_n, static_cast<int>(staging_size_m),
        static_cast<int>(staging_size_n), s.signals_ptr_array,
            s.world, kSignalsStrideParity,
        int64_t{kMaxRanks}, int64_t{1},
        int64_t{2}, int64_t{kMaxBlocks},
        int64_t{kMaxRanks}, s.status_device, int64_t{1},
        int64_t{kStatusFields}, s.rank, s.timeout_ticks, eps);
  });
  // POSTCONDITION: the launch went in.
  if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
  return std::nullopt;
}  // namespace

// all_reduce_add_attn_res_rms_norm: at its config.
// One function: what Python passes, its tensors stripped to pointers and strides, the
// preconditions, the kernel's arguments, the launch, the postcondition.
ErrorCode hip_custom_communicator_all_reduce_add_attn_res_rms_norm(int64_t state,
    torch::Tensor& input, const std::optional<torch::Tensor>& prefix, torch::Tensor& blocks,
    torch::Tensor& norm_weight, torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& output_norm_weight, torch::Tensor& prefix_output,
    torch::Tensor& output, const std::optional<torch::Tensor>& block_output, int64_t num_blocks,
    double eps, double output_eps, const std::string& algorithm_name, int64_t grid_x, int64_t grid_y,
    int64_t grid_z, int64_t block_x, int64_t block_y, int64_t block_z, int64_t waves_per_eu,
    int64_t tile_m, int64_t tile_n, int64_t tile_k, const std::string& quantization_name,
    int64_t pack_bytes, std::optional<int64_t> quantize_from_bytes) {
  using namespace hip_custom_communicator;
  // THE CALL FROM TORCH: its state, its tensors, its config's ints.
  if (state == 0) return error_code(Error::not_open);
  if (const std::optional<Error> e = attn_res_tensors(input, prefix, blocks, norm_weight, qk_weight,
                   output_norm_weight, prefix_output, output, block_output,
                   num_blocks))
    return error_code(e);
  if (!all_ints({waves_per_eu, tile_m, tile_n, tile_k, pack_bytes, grid_x, grid_y, grid_z, block_x, block_y, block_z})) return error_code(Error::field_out_of_range);
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
  const void* output_norm_weight_ptr = output_norm_weight ? output_norm_weight->data_ptr() : nullptr;
  const int64_t output_norm_weight_stride_n = output_norm_weight ? output_norm_weight->stride(0) : 0;
  const int64_t output_norm_weight_size_n = output_norm_weight ? output_norm_weight->size(0) : 0;
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
  const int64_t block_output_stride_m = block_output ? block_output->stride(0) : 0;
  const int64_t block_output_stride_n = block_output ? block_output->stride(1) : 0;
  const int64_t block_output_size_m = block_output ? block_output->size(0) : 0;
  const int64_t block_output_size_n = block_output ? block_output->size(1) : 0;
  const at::ScalarType dtype = input.scalar_type();
  const Dim3 grid{static_cast<int>(grid_x), static_cast<int>(grid_y), static_cast<int>(grid_z)};
  const Dim3 block{static_cast<int>(block_x), static_cast<int>(block_y), static_cast<int>(block_z)};
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  // ---- PRECONDITIONS: the config, the call, then its instance. The first Error refuses the
  // call; past this section the call runs. ----
  const std::optional<AllReduceAddAttnResRmsNormAlgorithm> named = algorithm_named<AllReduceAddAttnResRmsNormBuildConfig>(algorithm_name);
  if (!named) return error_code(Error::no_such_kernel);
  const AllReduceAddAttnResRmsNormAlgorithm algorithm = *named;
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_code(Error::no_such_quantization);
  const AllReduceAddAttnResRmsNormBuildConfig build{.algorithm = algorithm,
      .block = block,
      .waves_per_eu = static_cast<int>(waves_per_eu),
      .tile_m = static_cast<int>(tile_m),
      .tile_n = static_cast<int>(tile_n),
      .tile_k = static_cast<int>(tile_k),
      .quantization = quantization_name == "int4" ? Quantization::int4 : Quantization::exact,
      .pack_bytes = static_cast<int>(pack_bytes)};
  const int64_t m = input_size_m, n = input_size_n;  // the problem's shape: the input's
  if (const std::optional<Error> e = refused(true, s.world, s.staging_bytes, s.partials_bytes, build,
                                             grid, quantize_from_bytes, dtype, m, n))
    return error_code(e);
  if (const std::optional<Error> e = layout_refused(
          dtype, build.pack_bytes, {{input_stride_m, input_stride_n},
                  {blocks_stride_m, blocks_stride_n},
                  {blocks_stride_r, blocks_stride_n},
                  {0, norm_weight_stride_n},
                  {0, qk_weight_stride_n},
                  {prefix_output_stride_m, prefix_output_stride_n},
                  {output_stride_m, output_stride_n}}))
    return error_code(e);
  // THE OPTIONAL BUFFERS, checked only when given: a missing one is a null
  // pointer with zero strides, which is no layout at all (a prefix-less call
  // was refused as `inner_stride_not_one`, 2026-10-13).
  if (prefix_ptr != nullptr)
    if (const std::optional<Error> e =
            layout_refused(dtype, build.pack_bytes, {{prefix_stride_m, prefix_stride_n}}))
      return error_code(e);
  if (output_norm_weight_ptr != nullptr)
    if (const std::optional<Error> e =
            layout_refused(dtype, build.pack_bytes, {{0, output_norm_weight_stride_n}}))
      return error_code(e);
  if (block_output_ptr != nullptr)
    if (const std::optional<Error> e = layout_refused(
            dtype, build.pack_bytes, {{block_output_stride_m, block_output_stride_n}}))
      return error_code(e);
  // THE PARTIALS'S ROWS, what partials_need sized it for (a one-shot keeps none).
  const int64_t partials_rows =
      n > 0 ? partials_need(algorithm, m,
                           n * elem_bytes(dtype) / build.pack_bytes, s.world,
                           build.pack_bytes) /
                  (n * elem_bytes(dtype))
            : 0;
  // A ROW WIDER THAN THE TILE keeps its rows' logits in the launch's LDS (attn_res.cuh's
  // attn_res_wide): [tile_m][num_blocks + 1] floats.
  const int shared_memory_bytes =
      n > build.tile_n ? static_cast<int>(build.tile_m * (num_blocks + 1) * sizeof(float)) : 0;
  const bool contiguous = input_stride_m == n * input_stride_n &&
      output_stride_m == n * output_stride_n;
  // A ROW WIDER THAN THE TILE runs the build's CHUNKED instance.
  const bool chunked = n > build.tile_n;
  // AN INT4 ROW WITH A TAIL runs the build's TAIL instance (int4_tail).
  const bool tail = build.quantization == Quantization::int4 && int4_tail(algorithm, m, n, s.world);
  const void* kernel = nullptr;
  if (!with_instance<OpType::all_reduce_add_attn_res_rms_norm>(build, s.world, dtype, dtype, contiguous, chunked, tail, /*quantize_int4=*/false, [&]<Instance I>() { kernel = kernel_pointer<I>(); }))
    return error_code(Error::tile_not_built);
  if (const std::optional<Error> e = check_kernel(kernel, grid, block, shared_memory_bytes, true))
    return error_code(e);

  // ---- THE LAUNCH: the buffers, the views of their regions, and the instance. ----
  // ITS VIEWS OF THE REGIONS, each [m, n] (the pushed one-shot's inbox [slot, m, n]), in
  // elements: zeros where it reads none.
  int64_t partials_stride_m = 0, partials_stride_n = 0, partials_size_m = 0, partials_size_n = 0, staging_stride_m = 0, staging_stride_n = 0, staging_size_m = 0, staging_size_n = 0;
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
  in_place = !is_push(algorithm) &&
      build.quantization == Quantization::exact;
  // EVERY VIEW INSIDE ITS REGION (view_fits).
  if (!view_fits(1, 0, partials_size_m, partials_stride_m, partials_size_n, partials_stride_n, elem_bytes(dtype), s.partials_bytes) ||
      !view_fits(1, 0, staging_size_m, staging_stride_m, staging_size_n, staging_stride_n, elem_bytes(dtype), s.staging_bytes))
    return error_code(Error::view_past_region);
  // THE INPUT TABLE (INPUT TABLES): where the kernel reads every rank's input in place (an exact
  // pull), or none where it reads the staging.
  void* const* table = nullptr;
  int64_t table_stride_m = staging_stride_m;
  if (in_place)
    if (const std::optional<Error> e = input_table(s, input_ptr, input_stride_m, input_size_m,
                                                   input_size_n, elem_bytes(dtype), stream, table,
                                                   table_stride_m))
      return error_code(e);
  const int table_size = in_place ? s.world : 0;
  const int64_t table_stride_n = in_place ? 1 : staging_stride_n;
  with_instance<OpType::all_reduce_add_attn_res_rms_norm>(build, s.world, dtype, dtype, contiguous, chunked, tail, /*quantize_int4=*/false, [&]<Instance I>() {
    constexpr auto B = I.build;
    using T = c10::impl::ScalarTypeToCPPTypeT<I.dtype>;
    using W = c10::impl::ScalarTypeToCPPTypeT<I.weight_dtype>;
    all_reduce_add_attn_res_rms_norm<B.algorithm, I.quantize_int4, T, I.world, B.tile_m, B.tile_n,
        B.tile_k, B.block.x, I.wave_size, B.pack_bytes, B.waves_per_eu, I.chunked, I.tail>
        <<<dim3_of(grid), dim3_of(B.block), shared_memory_bytes, stream>>>(
        static_cast<const T*>(input_ptr),
        input_stride_m, input_stride_n, static_cast<int>(input_size_m),
        static_cast<int>(input_size_n), reinterpret_cast<const T* const*>(table),
            table_size, table_stride_m, table_stride_n, static_cast<const T*>(prefix_ptr),
            prefix_stride_m,
        prefix_stride_n, static_cast<int>(prefix_size_m),
        static_cast<int>(prefix_size_n), static_cast<const T*>(blocks_ptr), blocks_stride_m,
        blocks_stride_r, blocks_stride_n, static_cast<int>(blocks_size_m),
        static_cast<int>(blocks_size_r), static_cast<int>(blocks_size_n),
        static_cast<const T*>(norm_weight_ptr), norm_weight_stride_n,
        static_cast<int>(norm_weight_size_n), static_cast<const T*>(qk_weight_ptr),
        qk_weight_stride_n, static_cast<int>(qk_weight_size_n),
        static_cast<const T*>(output_norm_weight_ptr), output_norm_weight_stride_n,
        static_cast<int>(output_norm_weight_size_n), static_cast<T*>(prefix_output_ptr),
        prefix_output_stride_m, prefix_output_stride_n,
        static_cast<int>(prefix_output_size_m),
        static_cast<int>(prefix_output_size_n), static_cast<T*>(output_ptr), output_stride_m,
        output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), static_cast<T*>(block_output_ptr),
        block_output_stride_m, block_output_stride_n,
        static_cast<int>(block_output_size_m),
        static_cast<int>(block_output_size_n), s.partials_ptr_array,
        s.world, partials_stride_m, partials_stride_n,
        static_cast<int>(partials_size_m), static_cast<int>(partials_size_n),
        s.staging_ptr_array, s.world, staging_stride_m,
        staging_stride_n, static_cast<int>(staging_size_m),
        static_cast<int>(staging_size_n), s.signals_ptr_array,
            s.world, kSignalsStrideParity,
        int64_t{kMaxRanks}, int64_t{1},
        int64_t{2}, int64_t{kMaxBlocks},
        int64_t{kMaxRanks}, s.status_device, int64_t{1},
        int64_t{kStatusFields}, s.rank, s.timeout_ticks, num_blocks, eps,
        output_eps);
  });
  // POSTCONDITION: the launch went in.
  if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
  return std::nullopt;
}

// rms_norm: this rank's rows normed (the second kernel of an op Python composes), at its config.
// rms_norm: out = rms_norm(inp, weight), vLLM's roundings exactly, on this rank's rows; the
// weight `weight_dtype`, dtype or f32.
// One function: what Python passes, its tensors stripped to pointers and strides, the
// preconditions, the kernel's arguments, the launch, the postcondition.
ErrorCode hip_custom_communicator_rms_norm(int64_t state, torch::Tensor& input, torch::Tensor& weight,
    torch::Tensor& output, double eps, int64_t grid_x, int64_t grid_y, int64_t grid_z,
    int64_t block_x, int64_t block_y, int64_t block_z, int64_t waves_per_eu, int64_t tile_m,
    int64_t tile_n, const std::string& quantization_name, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  using namespace hip_custom_communicator;
  // THE CALL FROM TORCH: its state, its tensors, its config's ints.
  if (state == 0) return error_code(Error::not_open);
  if (const std::optional<Error> e = norm_tensors(input, nullptr, weight, output, nullptr, /*in_place=*/true)) return error_code(e);
  if (!all_ints({waves_per_eu, tile_m, tile_n, pack_bytes, grid_x, grid_y, grid_z, block_x, block_y, block_z})) return error_code(Error::field_out_of_range);
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
  const Dim3 grid{static_cast<int>(grid_x), static_cast<int>(grid_y), static_cast<int>(grid_z)};
  const Dim3 block{static_cast<int>(block_x), static_cast<int>(block_y), static_cast<int>(block_z)};
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  // ---- PRECONDITIONS: the config, the call, then its instance. The first Error refuses the
  // call; past this section the call runs. ----
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_code(Error::no_such_quantization);
  const RmsNormBuildConfig build{.block = block,
      .waves_per_eu = static_cast<int>(waves_per_eu),
      .tile_m = static_cast<int>(tile_m),
      .tile_n = static_cast<int>(tile_n),
      .quantization = quantization_name == "int4" ? Quantization::int4 : Quantization::exact,
      .pack_bytes = static_cast<int>(pack_bytes)};
  const int64_t m = input_size_m, n = input_size_n;  // the problem's shape: the input's
  if (const std::optional<Error> e =
          refused(false, s.world, 0, 0, build, grid, quantize_from_bytes, dtype, m, n))
    return error_code(e);
  if (const std::optional<Error> e =
          layout_refused(dtype, build.pack_bytes, {{output_stride_m, output_stride_n},
                                 {input_stride_m, input_stride_n}}))
    return error_code(e);
  if (const std::optional<Error> e =
          layout_refused(weight_dtype, build.pack_bytes, {{0, weight_stride_n}}))
    return error_code(e);
  const OpType op = OpType::rms_norm;
  if (const std::optional<Error> e = weight_refused(op, dtype, weight_dtype)) return error_code(e);
  const int shared_memory_bytes = 0;
  // A ROW WIDER THAN THE TILE runs the build's CHUNKED instance.
  const bool chunked = n > build.tile_n;
  const void* kernel = nullptr;
  if (!with_instance<OpType::rms_norm>(build, s.world, dtype, weight_dtype, /*contiguous=*/false, chunked, /*tail=*/false, /*quantize_int4=*/false, [&]<Instance I>() { kernel = kernel_pointer<I>(); }))
    return error_code(Error::tile_not_built);
  if (const std::optional<Error> e = check_kernel(kernel, grid, block, shared_memory_bytes, false))
    return error_code(e);

  // ---- THE LAUNCH: the buffers, the views of their regions, and the instance. ----
  with_instance<OpType::rms_norm>(build, s.world, dtype, weight_dtype, /*contiguous=*/false, chunked, /*tail=*/false, /*quantize_int4=*/false, [&]<Instance I>() {
    constexpr auto B = I.build;
    using T = c10::impl::ScalarTypeToCPPTypeT<I.dtype>;
    using W = c10::impl::ScalarTypeToCPPTypeT<I.weight_dtype>;
    rms_norm_rows<T, W, B.tile_m, B.tile_n, B.block.x, I.wave_size, B.pack_bytes, B.waves_per_eu,
        I.chunked>
        <<<dim3_of(grid), dim3_of(B.block), shared_memory_bytes, stream>>>(
        static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
        static_cast<int>(input_size_m), static_cast<int>(input_size_n),
        static_cast<const W*>(weight_ptr), weight_stride_n, static_cast<int>(weight_size_n),
        static_cast<T*>(output_ptr), output_stride_m, output_stride_n,
        static_cast<int>(output_size_m), static_cast<int>(output_size_n), eps);
  });
  // POSTCONDITION: the launch went in.
  if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
  return std::nullopt;
}

// add_rms_norm: this rank's rows added and normed (the second kernel of an op Python composes),
// at its config.
// add_rms_norm: out, residual_out = fused_add_rms_norm(inp, residual, weight), vLLM's roundings
// exactly, on this rank's rows.
// One function: what Python passes, its tensors stripped to pointers and strides, the
// preconditions, the kernel's arguments, the launch, the postcondition.
ErrorCode hip_custom_communicator_add_rms_norm(int64_t state, torch::Tensor& input,
    torch::Tensor& residual, torch::Tensor& weight, torch::Tensor& output,
    torch::Tensor& residual_output, double eps, int64_t grid_x, int64_t grid_y, int64_t grid_z,
    int64_t block_x, int64_t block_y, int64_t block_z, int64_t waves_per_eu, int64_t tile_m,
    int64_t tile_n, const std::string& quantization_name, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  using namespace hip_custom_communicator;
  // THE CALL FROM TORCH: its state, its tensors, its config's ints.
  if (state == 0) return error_code(Error::not_open);
  if (const std::optional<Error> e = norm_tensors(input, &residual, weight, output, &residual_output, /*in_place=*/true)) return error_code(e);
  if (!all_ints({waves_per_eu, tile_m, tile_n, pack_bytes, grid_x, grid_y, grid_z, block_x, block_y, block_z})) return error_code(Error::field_out_of_range);
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
  const Dim3 grid{static_cast<int>(grid_x), static_cast<int>(grid_y), static_cast<int>(grid_z)};
  const Dim3 block{static_cast<int>(block_x), static_cast<int>(block_y), static_cast<int>(block_z)};
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  // ---- PRECONDITIONS: the config, the call, then its instance. The first Error refuses the
  // call; past this section the call runs. ----
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_code(Error::no_such_quantization);
  const AddRmsNormBuildConfig build{.block = block,
      .waves_per_eu = static_cast<int>(waves_per_eu),
      .tile_m = static_cast<int>(tile_m),
      .tile_n = static_cast<int>(tile_n),
      .quantization = quantization_name == "int4" ? Quantization::int4 : Quantization::exact,
      .pack_bytes = static_cast<int>(pack_bytes)};
  const int64_t m = input_size_m, n = input_size_n;  // the problem's shape: the input's
  if (const std::optional<Error> e =
          refused(false, s.world, 0, 0, build, grid, quantize_from_bytes, dtype, m, n))
    return error_code(e);
  if (const std::optional<Error> e = layout_refused(
          dtype, build.pack_bytes, {{output_stride_m, output_stride_n},
                  {residual_output_stride_m, residual_output_stride_n},
                  {input_stride_m, input_stride_n},
                  {residual_stride_m, residual_stride_n}}))
    return error_code(e);
  if (const std::optional<Error> e =
          layout_refused(weight_dtype, build.pack_bytes, {{0, weight_stride_n}}))
    return error_code(e);
  const OpType op = OpType::add_rms_norm;
  if (const std::optional<Error> e = weight_refused(op, dtype, weight_dtype)) return error_code(e);
  const int shared_memory_bytes = 0;
  // A ROW WIDER THAN THE TILE runs the build's CHUNKED instance.
  const bool chunked = n > build.tile_n;
  const void* kernel = nullptr;
  if (!with_instance<OpType::add_rms_norm>(build, s.world, dtype, weight_dtype, /*contiguous=*/false, chunked, /*tail=*/false, /*quantize_int4=*/false, [&]<Instance I>() { kernel = kernel_pointer<I>(); }))
    return error_code(Error::tile_not_built);
  if (const std::optional<Error> e = check_kernel(kernel, grid, block, shared_memory_bytes, false))
    return error_code(e);

  // ---- THE LAUNCH: the buffers, the views of their regions, and the instance. ----
  with_instance<OpType::add_rms_norm>(build, s.world, dtype, weight_dtype, /*contiguous=*/false, chunked, /*tail=*/false, /*quantize_int4=*/false, [&]<Instance I>() {
    constexpr auto B = I.build;
    using T = c10::impl::ScalarTypeToCPPTypeT<I.dtype>;
    using W = c10::impl::ScalarTypeToCPPTypeT<I.weight_dtype>;
    add_rms_norm_rows<T, W, B.tile_m, B.tile_n, B.block.x, I.wave_size, B.pack_bytes,
        B.waves_per_eu, I.chunked>
        <<<dim3_of(grid), dim3_of(B.block), shared_memory_bytes, stream>>>(
        static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
        static_cast<int>(input_size_m), static_cast<int>(input_size_n),
        static_cast<const T*>(residual_ptr), residual_stride_m, residual_stride_n,
        static_cast<int>(residual_size_m), static_cast<int>(residual_size_n),
        static_cast<const W*>(weight_ptr), weight_stride_n, static_cast<int>(weight_size_n),
        static_cast<T*>(output_ptr), output_stride_m, output_stride_n,
        static_cast<int>(output_size_m), static_cast<int>(output_size_n),
        static_cast<T*>(residual_output_ptr), residual_output_stride_m, residual_output_stride_n,
        static_cast<int>(residual_output_size_m), static_cast<int>(residual_output_size_n), eps);
  });
  // POSTCONDITION: the launch went in.
  if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
  return std::nullopt;
}

// add_attn_res_rms_norm: AttnRes on this rank's rows (the second kernel of the op Python
// composes), at its config.
// add_attn_res_rms_norm: Kimi-K3's AttnRes and its RMSNorm on each of this rank's rows
// (all_reduce_add_attn_res_rms_norm's, its input the sum), `num_blocks` sources.
// One function: what Python passes, its tensors stripped to pointers and strides, the
// preconditions, the kernel's arguments, the launch, the postcondition.
ErrorCode hip_custom_communicator_add_attn_res_rms_norm(int64_t state, torch::Tensor& input,
    const std::optional<torch::Tensor>& prefix, torch::Tensor& blocks, torch::Tensor& norm_weight,
    torch::Tensor& qk_weight, const std::optional<torch::Tensor>& output_norm_weight,
    torch::Tensor& prefix_output, torch::Tensor& output,
    const std::optional<torch::Tensor>& block_output, int64_t num_blocks, double eps,
    double output_eps, int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t block_x,
    int64_t block_y, int64_t block_z, int64_t waves_per_eu, int64_t tile_m, int64_t tile_n,
    int64_t tile_k, const std::string& quantization_name, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  using namespace hip_custom_communicator;
  // THE CALL FROM TORCH: its state, its tensors, its config's ints.
  if (state == 0) return error_code(Error::not_open);
  if (const std::optional<Error> e = attn_res_tensors(input, prefix, blocks, norm_weight, qk_weight,
                   output_norm_weight, prefix_output, output, block_output,
                   num_blocks))
    return error_code(e);
  if (!all_ints({waves_per_eu, tile_m, tile_n, tile_k, pack_bytes, grid_x, grid_y, grid_z, block_x, block_y, block_z})) return error_code(Error::field_out_of_range);
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
  const void* output_norm_weight_ptr = output_norm_weight ? output_norm_weight->data_ptr() : nullptr;
  const int64_t output_norm_weight_stride_n = output_norm_weight ? output_norm_weight->stride(0) : 0;
  const int64_t output_norm_weight_size_n = output_norm_weight ? output_norm_weight->size(0) : 0;
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
  const int64_t block_output_stride_m = block_output ? block_output->stride(0) : 0;
  const int64_t block_output_stride_n = block_output ? block_output->stride(1) : 0;
  const int64_t block_output_size_m = block_output ? block_output->size(0) : 0;
  const int64_t block_output_size_n = block_output ? block_output->size(1) : 0;
  const at::ScalarType dtype = input.scalar_type();
  const Dim3 grid{static_cast<int>(grid_x), static_cast<int>(grid_y), static_cast<int>(grid_z)};
  const Dim3 block{static_cast<int>(block_x), static_cast<int>(block_y), static_cast<int>(block_z)};
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  // ---- PRECONDITIONS: the config, the call, then its instance. The first Error refuses the
  // call; past this section the call runs. ----
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_code(Error::no_such_quantization);
  const AddAttnResRmsNormBuildConfig build{.block = block,
      .waves_per_eu = static_cast<int>(waves_per_eu),
      .tile_m = static_cast<int>(tile_m),
      .tile_n = static_cast<int>(tile_n),
      .tile_k = static_cast<int>(tile_k),
      .quantization = quantization_name == "int4" ? Quantization::int4 : Quantization::exact,
      .pack_bytes = static_cast<int>(pack_bytes)};
  const int64_t m = input_size_m, n = input_size_n;  // the problem's shape: the input's
  if (const std::optional<Error> e =
          refused(false, s.world, 0, 0, build, grid, quantize_from_bytes, dtype, m, n))
    return error_code(e);
  if (const std::optional<Error> e = layout_refused(
          dtype, build.pack_bytes, {{input_stride_m, input_stride_n},
                  {blocks_stride_m, blocks_stride_n},
                  {blocks_stride_r, blocks_stride_n},
                  {0, norm_weight_stride_n},
                  {0, qk_weight_stride_n},
                  {prefix_output_stride_m, prefix_output_stride_n},
                  {output_stride_m, output_stride_n}}))
    return error_code(e);
  // THE OPTIONAL BUFFERS, checked only when given (all_reduce_add_attn_res_rms_norm's).
  if (prefix_ptr != nullptr)
    if (const std::optional<Error> e =
            layout_refused(dtype, build.pack_bytes, {{prefix_stride_m, prefix_stride_n}}))
      return error_code(e);
  if (output_norm_weight_ptr != nullptr)
    if (const std::optional<Error> e =
            layout_refused(dtype, build.pack_bytes, {{0, output_norm_weight_stride_n}}))
      return error_code(e);
  if (block_output_ptr != nullptr)
    if (const std::optional<Error> e = layout_refused(
            dtype, build.pack_bytes, {{block_output_stride_m, block_output_stride_n}}))
      return error_code(e);
  // A ROW WIDER THAN THE TILE keeps its rows' logits in the launch's LDS (attn_res.cuh's
  // attn_res_wide): [tile_m][num_blocks + 1] floats.
  const int shared_memory_bytes = n > tile_n ? static_cast<int>(tile_m * (num_blocks + 1) * sizeof(float)) : 0;
  // A ROW WIDER THAN THE TILE runs the build's CHUNKED instance.
  const bool chunked = n > build.tile_n;
  const void* kernel = nullptr;
  if (!with_instance<OpType::add_attn_res_rms_norm>(build, s.world, dtype, dtype, /*contiguous=*/false, chunked, /*tail=*/false, /*quantize_int4=*/false, [&]<Instance I>() { kernel = kernel_pointer<I>(); }))
    return error_code(Error::tile_not_built);
  if (const std::optional<Error> e = check_kernel(kernel, grid, block, shared_memory_bytes, false))
    return error_code(e);

  // ---- THE LAUNCH: the buffers, the views of their regions, and the instance. ----
  with_instance<OpType::add_attn_res_rms_norm>(build, s.world, dtype, dtype, /*contiguous=*/false, chunked, /*tail=*/false, /*quantize_int4=*/false, [&]<Instance I>() {
    constexpr auto B = I.build;
    using T = c10::impl::ScalarTypeToCPPTypeT<I.dtype>;
    add_attn_res_rms_norm_rows<T, B.tile_m, B.tile_n, B.tile_k, B.block.x, I.wave_size,
        B.pack_bytes, B.waves_per_eu, I.chunked>
        <<<dim3_of(grid), dim3_of(B.block), shared_memory_bytes, stream>>>(
        static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
        static_cast<int>(input_size_m), static_cast<int>(input_size_n),
        static_cast<const T*>(prefix_ptr), prefix_stride_m, prefix_stride_n,
        static_cast<int>(prefix_size_m), static_cast<int>(prefix_size_n),
        static_cast<const T*>(blocks_ptr), blocks_stride_m, blocks_stride_r, blocks_stride_n,
        static_cast<int>(blocks_size_m), static_cast<int>(blocks_size_r),
        static_cast<int>(blocks_size_n), static_cast<const T*>(norm_weight_ptr),
        norm_weight_stride_n, static_cast<int>(norm_weight_size_n),
        static_cast<const T*>(qk_weight_ptr), qk_weight_stride_n, static_cast<int>(qk_weight_size_n),
        static_cast<const T*>(output_norm_weight_ptr), output_norm_weight_stride_n,
        static_cast<int>(output_norm_weight_size_n), static_cast<T*>(prefix_output_ptr),
        prefix_output_stride_m, prefix_output_stride_n, static_cast<int>(prefix_output_size_m),
        static_cast<int>(prefix_output_size_n), static_cast<T*>(output_ptr), output_stride_m,
        output_stride_n, static_cast<int>(output_size_m), static_cast<int>(output_size_n),
        static_cast<T*>(block_output_ptr), block_output_stride_m, block_output_stride_n,
        static_cast<int>(block_output_size_m), static_cast<int>(block_output_size_n), num_blocks,
        eps, output_eps);
  });
  // POSTCONDITION: the launch went in.
  if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
  return std::nullopt;
}
