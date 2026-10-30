// SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc.
// All rights reserved.
//
// THE TYPES the API functions (api/) share: the machine, the build, the errors,
// each op's kernels and their compiled instances, the layout's constants, and
// the state between Python and C++.
#pragma once

// HIP's spelling: hipify rewrites only the .cu, never the headers it includes.
#include <ATen/hip/HIPContext.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "kernels/all_reduce.cuh"
#include "kernels/all_reduce_add_attn_res_rms_norm.cuh"
#include "kernels/all_reduce_add_rms_norm.cuh"
#include "kernels/common.cuh"
#include "kernels/reduce_scatter.cuh"

namespace hip_custom_communicator {

// WHAT A CONFIG MOVES ACROSS THE LINKS, the host's alone: a config's
// `quantization` field. A kernel takes QUANTIZE_INT4 (bool): the config's
// quantization, every call (an INT4 config always quantizes, an exact one
// never).
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
  field_missing = 37,
  no_such_group = 38,
  exchange_failed = 39,
  peer_not_accessible = 40,
  peers_not_opened = 41,
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

// =============================================================================
// THE COMPILED KERNELS: every instance of every kernel, a row of its kernel's
// table, which a launch looks its kernel up in.
// =============================================================================

// ALL_REDUCE_ADD_RMS_NORM'S ALGORITHMS, the op's (what the tuner picks among):
// a fused kernel's body (the norm kernel's or the add-norm kernel's, by the
// residual), or a SPLIT, our all-reduce kernel at that body then the norm (or
// add-norm) on rows. The kernels' own enum is their bodies alone
// (AllReduceAddRmsNormKernelAlgorithm).
enum class AllReduceAddRmsNormAlgorithm : int {
  one_shot_pull,
  two_shot_pull_rows,
  two_shot_pull_columns,
  two_shot_push,
  one_shot_pull_split,
  one_shot_push_split,
  one_shot_push_staged_split,
  two_shot_pull_split,
  two_shot_push_split,
};

// EACH NORM ON ROWS' ONE ALGORITHM, its op's own (the table's "rows"; its
// kernel has no algorithm parameter): add_rms_norm's both kernels' (rms_norm
// and add_rms_norm, by the residual).
enum class AddRmsNormAlgorithm : int { rows };
enum class AddAttnResRmsNormAlgorithm : int { rows };

// A COMPILED INSTANCE OF A KERNEL, a row of its table: the build it was
// compiled at (a tuned config, its grid aside), the world (0 for a norm on
// rows, which has none), its wave's lanes and its dtypes, which of the build's
// instances it is (contiguous: the plain two-shots' contiguous walk; chunked: a
// row wider than its tile; tail: an INT4 row with a short last group;
// quantize_int4: what it moves; residual: which of a norm op's two kernels, the
// add-norm or the norm), and its address. GENERATED (builds.py, at configure),
// one table a kernel, `k<Kernel>Instances`: every row the table's configs
// reach, so a config of the table is always a row, and a call finds its kernel
// or is refused (not_built).
template <typename ALGORITHM>
struct Instance {
  ALGORITHM algorithm;
  int block_x, block_y, block_z, waves_per_eu, tile_m, tile_n, tile_k;
  int quantization, pack_bytes;
  int world, wave_size;
  at::ScalarType dtype, weight_dtype;
  bool contiguous, chunked, tail, quantize_int4, residual;
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
using AllReduceAddRmsNormKernel =
    decltype(&all_reduce_add_rms_norm<AllReduceAddRmsNormKernelAlgorithm{},
                                      false, false, T, W, 2, 1, 1, 64, 64, 16,
                                      1, false, false>);
template <typename T, typename W>
using AllReduceAddAttnResRmsNormKernel =
    decltype(&all_reduce_add_attn_res_rms_norm<
             AllReduceAddAttnResRmsNormAlgorithm{}, false, T, 2, 1, 1, 1, 64,
             64, 16, 1, false, false>);
template <typename T, typename W>
using AddRmsNormKernel =
    decltype(&add_rms_norm<false, T, W, 1, 1, 64, 64, 16, 1, false>);
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

// THE STATE BETWEEN PYTHON AND C++, one a communicator: what it was given (its
// group, this rank, the world, the device, the largest call, a wait's bound),
// what open asked of the device, the symmetric memory, this rank's own buffers
// (the pushed one-shot's epochs, the status), and the input tables. Made by
// `open`, released by `close`; Python holds it as a handle. A launch reads it,
// and a captured in-place launch reserves its table slot in it: its one
// mutation during a call.
struct CommsState {
  // THE CPU GROUP the exchanges go round (open's, register_graph's, close's
  // handshakes), each giving up on a peer after `timeout_seconds`.
  c10::intrusive_ptr<c10d::ProcessGroup> group;
  double timeout_seconds = 0;
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

// WHAT CROSSES THE TORCH BOUNDARY, as the tuple an op schema can return: open's
// (the Error or none, the handle, the device's arch).
using OpenWire = std::tuple<std::optional<int64_t>, int64_t, std::string>;
// AN ERROR CODE ACROSS THE BOUNDARY: none when the call ran, else the Error's
// number (a torch op returns no enum).
using ErrorCode = std::optional<int64_t>;
