// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Our HIP collectives as torch ops, built into `_rocm_C`: this file only strips
// torch. Each op checks its tensors and passes their pointers, shapes and
// dtypes, its names as enums and its numbers as ints, to its function in
// hip_custom_communicator/hip_custom_communicator.cuh (the API, torch-free),
// and turns the answer back: an op returns the Error's number it was refused
// for, or none when it ran. Self-contained: torch and the HIP runtime, nothing
// from aiter.

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

// THE DECLARATIONS torch_bindings.cpp registers, so a definition here that
// differs is a compile error, not a mismatch at run time.
#include "ops.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <variant>
#include <string>
#include <vector>

#include "hip_custom_communicator/hip_custom_communicator.cuh"

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
using hip_custom_communicator::DType;
using hip_custom_communicator::Error;

// A TENSOR'S DTYPE, where the build has it.
std::optional<DType> dtype_of_tensor(const torch::Tensor& t) {
  const at::ScalarType s = t.scalar_type();
  if (s == at::ScalarType::Half) return DType::f16;
  if (s == at::ScalarType::BFloat16) return DType::bf16;
  if (s == at::ScalarType::Float) return DType::f32;
  return std::nullopt;
}

// EVERY CONFIG FIELD AN INT: the API takes ints.
bool all_ints(std::initializer_list<int64_t> vs) {
  for (const int64_t v : vs)
    if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) return false;
  return true;
}

hipStream_t current_stream() { return at::cuda::getCurrentCUDAStream(); }

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
}  // namespace

// WHAT CROSSES THE TORCH BOUNDARY, as the tuples an op schema can return.
using Names = std::vector<std::string>;
using OpenWire = std::tuple<std::optional<int64_t>, int64_t, std::vector<int64_t>, std::string,
                            std::vector<int64_t>>;
using GraphMetaWire = std::tuple<std::optional<int64_t>, std::vector<int64_t>, std::vector<int64_t>>;
using BuildInfoWire =
    std::tuple<Names, std::vector<int64_t>, int64_t, Names, Names, Names, Names,
               Names, Names, std::vector<int64_t>, std::vector<int64_t>, Names,
               Names, std::vector<int64_t>, int64_t, Names>;
// A REFUSAL ACROSS THE BOUNDARY: none when it ran, else the Error's number (a torch op returns no
// enum).
using Refused = std::optional<int64_t>;
Refused error_number(std::optional<Error> e) {
  return e ? Refused(static_cast<int64_t>(*e)) : std::nullopt;
}

// WHAT THE BUILD HOLDS, kBuild's projection for Python: its dtypes by name, its
// worlds, a pack's bytes, its ops, errors and fault places by
// name in their enums' order (each error's name without its reason), which
// Python's `Op`, `Error` and `Where` are held to, its templates, and each op's
// plain call, which Python's `should` admits.
BuildInfoWire hip_custom_communicator_build_info() {
  using namespace hip_custom_communicator;
  Names dtypes, ops, errors, where_names;
  for (const DType d : kBuild.supports.dtypes) dtypes.push_back(to_string(d));
  const std::vector<int64_t> worlds(kBuild.supports.worlds.begin(),
                                    kBuild.supports.worlds.end());
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
    for (const DType d : o.dtypes) op_dtypes.push_back(to_string(d));
    for (const DType d : o.weights) op_weights.push_back(to_string(d));
    op_facts.insert(op_facts.end(), {static_cast<int64_t>(o.dtypes.size()),
                                     static_cast<int64_t>(o.weights.size()),
                                     o.two_d});
  }
  return {dtypes,     worlds,       kBuild.memory.pack_bytes,
          ops,        errors,       where_names,
          templates,  template_ops, fields,
          configs,    counts,       op_dtypes,
          op_weights, op_facts,   kUnfencedBlocks,
          algorithms};
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

// =================================================================================================
// THE STATE, AS A HANDLE: the API's CommsState, made by `open` and handed to Python as its address
// (an int, as vLLM's custom all-reduce hands its own), passed back to every call, and freed by
// `free`. Each returns its Error's number, or none.
// =================================================================================================

namespace {
hip_custom_communicator::CommsState* state_at(int64_t state) {
  return reinterpret_cast<hip_custom_communicator::CommsState*>(state);
}
hipIpcMemHandle_t ipc_from(const std::vector<int64_t>& bytes, size_t at) {
  hipIpcMemHandle_t h;
  for (size_t i = 0; i < sizeof(h); ++i)
    reinterpret_cast<char*>(&h)[i] = static_cast<char>(bytes[at * sizeof(h) + i]);
  return h;
}
void ipc_into(const hipIpcMemHandle_t& h, std::vector<int64_t>& bytes) {
  for (size_t i = 0; i < sizeof(h); ++i)
    bytes.push_back(static_cast<uint8_t>(reinterpret_cast<const char*>(&h)[i]));
}
}  // namespace

// OPEN: the state of a communicator at `rank` of `world` on `device`, serving `max_rows` rows of
// at most `max_cols` columns, a wait giving up after `timeout_seconds`: (the Error it is refused
// for, or none; the handle, 0 where refused; this rank's arena's IPC handle, its bytes; the
// device's arch; the capacity: max_rows, max_cols, the partials' and staging's bytes, then each
// op's most bytes a call). A device the build does not run on is refused (nothing allocated): the
// caller does not run here.
OpenWire hip_custom_communicator_open(int64_t rank, int64_t world, int64_t device,
                                      int64_t max_rows, int64_t max_cols,
                                      double timeout_seconds) {
  using namespace hip_custom_communicator;
  if (!all_ints({rank, world, device}))
    return {error_number(Error::field_out_of_range), 0, {}, "", {}};
  auto* s = new CommsState();
  std::optional<Error> e = open_state(*s, static_cast<int>(rank), static_cast<int>(world),
                                      static_cast<int>(device), max_rows, max_cols,
                                      timeout_seconds);
  hipIpcMemHandle_t h{};
  if (!e) e = ipc_handle(*s, h);
  if (e) {
    (void)free_state(*s);
    delete s;
    return {error_number(e), 0, {}, "", {}};
  }
  std::vector<int64_t> handle;
  ipc_into(h, handle);
  std::vector<int64_t> capacity{max_rows, max_cols, s->partials_bytes, s->staging_bytes};
  for (const auto& o : kOps)
    capacity.push_back(max_bytes(o.type, s->partials_bytes, s->staging_bytes));
  return {std::nullopt, reinterpret_cast<int64_t>(s), handle, kTargetArch, capacity};
}

// CONNECT: every rank's arena's IPC handle (`handles`, their bytes, rank after rank) opened.
Refused hip_custom_communicator_connect(int64_t state, const std::vector<int64_t>& handles) {
  if (state == 0) return error_number(Error::not_open);
  hip_custom_communicator::CommsState& s = *state_at(state);
  if (handles.size() != s.world * sizeof(hipIpcMemHandle_t)) return error_number(Error::world_not_built);
  std::vector<hipIpcMemHandle_t> by_rank;
  for (int r = 0; r < s.world; ++r) by_rank.push_back(ipc_from(handles, r));
  return error_number(hip_custom_communicator::connect_state(s, by_rank));
}

// THE CAPTURED LAUNCHES' INPUTS (vLLM's get_graph_buffer_ipc_meta): (the Error or none; their
// allocations' IPC handles' bytes, an input after another; each input's offset in its own).
GraphMetaWire hip_custom_communicator_graph_ipc_meta(int64_t state) {
  if (state == 0) return {error_number(Error::not_open), {}, {}};
  std::vector<hipIpcMemHandle_t> handles;
  std::vector<int64_t> offsets;
  if (const std::optional<Error> e =
          hip_custom_communicator::graph_ipc_meta(*state_at(state), handles, offsets))
    return {error_number(e), {}, {}};
  std::vector<int64_t> bytes;
  for (const hipIpcMemHandle_t& h : handles) ipc_into(h, bytes);
  return {std::nullopt, bytes, offsets};
}

// THE CAPTURED LAUNCHES' TABLES (vLLM's register_graph_buffers): every rank's
// `graph_ipc_meta`, rank after rank.
Refused hip_custom_communicator_register_graph_buffers(int64_t state,
                                                       const std::vector<int64_t>& handles,
                                                       const std::vector<int64_t>& offsets) {
  if (state == 0) return error_number(Error::not_open);
  if (handles.size() != offsets.size() * sizeof(hipIpcMemHandle_t))
    return error_number(Error::ranks_disagree);
  std::vector<hipIpcMemHandle_t> by_input;
  for (size_t i = 0; i < offsets.size(); ++i) by_input.push_back(ipc_from(handles, i));
  return error_number(
      hip_custom_communicator::register_graph_buffers(*state_at(state), by_input, offsets));
}

// CLOSE THE PEERS: every borrowed mapping closed (every launch finished, every rank stopped).
Refused hip_custom_communicator_close_peers(int64_t state) {
  if (state == 0) return error_number(Error::not_open);
  return error_number(hip_custom_communicator::close_peers(*state_at(state)));
}

// FREE: this rank's buffers freed and the state with them (no peer has them open).
Refused hip_custom_communicator_free(int64_t state) {
  if (state == 0) return error_number(Error::not_open);
  hip_custom_communicator::CommsState* s = state_at(state);
  const std::optional<Error> e = hip_custom_communicator::free_state(*s);
  delete s;
  return error_number(e);
}

// all_reduce: at its config.
// One function: what Python passes, its tensors stripped to pointers and strides, the
// preconditions, the kernel's arguments, the launch, the postcondition.
Refused hip_custom_communicator_all_reduce(int64_t state, torch::Tensor& input,
    torch::Tensor& output, const std::string& algorithm_name, int64_t grid_x, int64_t grid_y,
    int64_t grid_z, int64_t block_x, int64_t block_y, int64_t block_z, int64_t waves_per_eu,
    int64_t tile_m, int64_t tile_n, const std::string& quantization_name, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  using namespace hip_custom_communicator;
  // THE CALL FROM TORCH: its state, its tensors, its config's ints.
  if (state == 0) return error_number(Error::not_open);
  if (!on_device({&input, &output})) return error_number(Error::not_on_device);
  if (output.sizes() != input.sizes()) return error_number(Error::shape_mismatch);
  if (output.scalar_type() != input.scalar_type()) return error_number(Error::dtype_mismatch);
  if (overlapping({&input}, {&output})) return error_number(Error::inputs_overlap_outputs);
  const std::optional<torch::Tensor> input_rows = as_rows(input),
                                     output_rows = as_rows(output);
  if (!input_rows || !output_rows)
    return error_number(Error::not_contiguous);
  const std::optional<DType> found_dtype = dtype_of_tensor(input);
  if (!found_dtype) return error_number(Error::dtype_not_built);
  if (!all_ints({waves_per_eu, tile_m, tile_n, pack_bytes, grid_x, grid_y, grid_z, block_x, block_y, block_z})) return error_number(Error::field_out_of_range);
  CommsState& s = *state_at(state);
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
  const DType dtype = *found_dtype;
  const Dim3 grid{static_cast<int>(grid_x), static_cast<int>(grid_y), static_cast<int>(grid_z)};
  const Dim3 block{static_cast<int>(block_x), static_cast<int>(block_y), static_cast<int>(block_z)};
  const hipStream_t stream = current_stream();
  // ---- PRECONDITIONS: the config, the call, then its instance. The first Error refuses the
  // call; past this section the call runs. ----
  const std::optional<AllReduceAlgorithm> named = algorithm_named<AllReduceBuildConfig>(algorithm_name);
  if (!named) return error_number(Error::no_such_kernel);
  const AllReduceAlgorithm algorithm = *named;
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_number(Error::no_such_quantization);
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
    return error_number(e);
  const int64_t quantize_from = quantize_from_bytes.value_or(0);
  if (const std::optional<Error> e =
          layout_refused(dtype, build.pack_bytes, {{output_stride_m, output_stride_n},
                                 {input_stride_m, input_stride_n}}))
    return error_number(e);
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
  if (unfenced && volume(grid) > kUnfencedBlocks) return error_number(Error::launch_out_of_range);
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
  if (inbox && s.world * slot * elem_bytes(dtype) > s.staging_bytes) return error_number(Error::staging_too_small);
  const int shared_memory_bytes = 0;
  const bool contiguous = input_stride_m == n * input_stride_n &&
      output_stride_m == n * output_stride_n;
  // AN INT4 ROW WITH A TAIL runs the build's TAIL instance (int4_tail).
  const bool tail = quantize_int4 && int4_tail(algorithm, m, n, s.world);
  const void* kernel = nullptr;
  if (!with_instance<OpType::all_reduce>(build, s.world, dtype, dtype, contiguous, /*chunked=*/false, tail, quantize_int4, [&]<Instance I>() { kernel = kernel_pointer<I>(); }))
    return error_number(Error::tile_not_built);
  if (const std::optional<Error> e = check_kernel(kernel, grid, block, shared_memory_bytes, true))
    return error_number(e);

  // ---- THE LAUNCH: the buffers, the views of their regions, and the instance. ----
  // THIS CALL'S BUFFERS, from its arguments: the shared buffer's regions (`arena_layout`)
  // and the private buffer's arrays, epochs and tables (kPrivate*).
  char* const* const region_ptr_arrays =
      reinterpret_cast<char* const*>(s.private_buffer + kPrivateRegionPtrArrays);
  const auto signals_ptr_array =
      reinterpret_cast<uint32_t* const*>(region_ptr_arrays + kSignalsArray * kMaxRanks);
  char* const* const staging_ptr_array = region_ptr_arrays + kStagingArray * kMaxRanks;
  char* const* const partials_ptr_array = region_ptr_arrays + kPartialsArray * kMaxRanks;
  char* const* const unfenced_ptr_array = region_ptr_arrays + kUnfencedArray * kMaxRanks;
  const auto unfenced_epochs =
      reinterpret_cast<int32_t*>(s.private_buffer + kPrivateUnfencedEpochs);
  // ITS VIEWS OF THE REGIONS, each [m, n] (the pushed one-shot's inbox [slot, m, n]), in
  // elements: zeros where it reads none.
  int64_t partials_stride_m = 0, partials_stride_n = 0, partials_size_m = 0, partials_size_n = 0, staging_stride_m = 0, staging_stride_n = 0, staging_size_m = 0, staging_size_n = 0, unfenced_stride_slot = 0, unfenced_stride_m = 0, unfenced_stride_n = 0, unfenced_size_slot = 0, unfenced_size_m = 0, unfenced_size_n = 0;
  // THE INT4 FLAT WALK'S PAYLOADS on views of their own (int4_flat_share): both two-shots'.
  const bool int4_flat = quantize_int4 && is_two_shot(algorithm);
  const int64_t share = int4_flat_share(m, n, s.world);
  partials_stride_m = int4_flat ? share : slice_bytes / elem_bytes(dtype);
  partials_stride_n = 1;
  // THE ONE-SHOTS USE NO PARTIALS: an empty view.
  partials_size_m = !is_two_shot(algorithm) ? 0 : int4_flat ? 1 : m;
  partials_size_n = int4_flat ? share : slice_bytes / elem_bytes(dtype);
  staging_stride_m = in_place || unfenced ? 0 : int4_flat ? s.world * share : inbox ? slot : n;
  staging_stride_n = in_place || unfenced ? 0 : 1;
  staging_size_m = in_place || unfenced ? 0 : int4_flat ? 1 : inbox ? s.world : m;
  staging_size_n = in_place || unfenced ? 0 : int4_flat ? s.world * share : inbox ? slot : n;
  // THE UNFENCED REGION: [writer, row, n] payload elements, 2 * kUnfencedBlocks rows (bank, row) of
  // kUnfencedSlotBytes of payload each whatever the build (so a row is the same bytes at every
  // build, its epoch its own), a writer's rows contiguous.
  unfenced_stride_slot = unfenced ? 2 * kUnfencedBlocks * (kUnfencedSlotBytes / elem_bytes(dtype)) : 0;
  unfenced_stride_m = unfenced ? kUnfencedSlotBytes / elem_bytes(dtype) : 0;
  unfenced_stride_n = unfenced ? 1 : 0;
  unfenced_size_slot = unfenced ? s.world : 0;
  unfenced_size_m = unfenced ? 2 * kUnfencedBlocks : 0;
  unfenced_size_n = unfenced ? kUnfencedSlotBytes / elem_bytes(dtype) : 0;
  // THE ONE-SHOTS IN THE STAGING (kernels/all_reduce.cuh), as bytes over the dtype's elements
  // (what the launch's region checks read): the INT4 pull's [m, n] payload; the staged push's
  // WORLD slots of an input (INT4: its payload); no partials. The kernels find them from the
  // input's sizes. (The line push's inbox is the unfenced region, above.)
  if (is_one_shot(algorithm) && algorithm != AllReduceAlgorithm::one_shot_push &&
      (quantize_int4 || algorithm == AllReduceAlgorithm::one_shot_push_staged)) {
    const int64_t payload = quantize_int4 ? int4_payload_bytes(m, n) : m * n * elem_bytes(dtype);
    const int64_t slot = quantize_int4 ? (payload + kInt4ScalesAlign - 1) / kInt4ScalesAlign *
                                             kInt4ScalesAlign
                                       : payload;
    const int64_t staged =
        algorithm == AllReduceAlgorithm::one_shot_push_staged ? s.world * slot : payload;
    staging_stride_m = 0;
    staging_stride_n = 1;
    staging_size_m = 1;
    staging_size_n = (staged + elem_bytes(dtype) - 1) / elem_bytes(dtype);
    partials_stride_m = 0;
    partials_stride_n = 1;
    partials_size_m = 0;
    partials_size_n = 0;
  }
  // EVERY VIEW INSIDE ITS REGION (view_fits).
  if (!view_fits(1, 0, partials_size_m, partials_stride_m, partials_size_n, partials_stride_n, elem_bytes(dtype), s.partials_bytes) ||
      !view_fits(1, 0, staging_size_m, staging_stride_m, staging_size_n, staging_stride_n, elem_bytes(dtype), s.staging_bytes) ||
      !view_fits(unfenced_size_slot, unfenced_stride_slot, unfenced_size_m, unfenced_stride_m, unfenced_size_n, unfenced_stride_n, elem_bytes(dtype) * (kLineBytes / kLinePayloadBytes), unfenced_bytes(s.world)) ||
      !view_fits(1, 0, int64_t{kUnfencedBlocks}, kUnfencedEpochsStrideBlock, kUnfencedEpochsStrideBlock, int64_t{1}, 4, kUnfencedEpochsBytes))
    return error_number(Error::view_past_region);
  // THE INPUT TABLE (INPUT TABLES): where the kernel reads every rank's input in place (an exact
  // pull), or none where it reads the staging.
  void* const* table = nullptr;
  int64_t table_stride_m = staging_stride_m;
  if (in_place)
    if (const std::optional<Error> e = input_table(s, input_ptr, input_stride_m, input_size_m,
                                                   input_size_n, elem_bytes(dtype), stream, table,
                                                   table_stride_m))
      return error_number(e);
  const int table_size = in_place ? s.world : 0;
  const int64_t table_stride_n = in_place ? 1 : staging_stride_n;
  with_instance<OpType::all_reduce>(build, s.world, dtype, dtype, contiguous, /*chunked=*/false, tail, quantize_int4, [&]<Instance I>() {
    constexpr auto B = I.build;
    using T = typename of_dtype<I.dtype>::t;
    using W = typename of_dtype<I.weight_dtype>::t;
    all_reduce<B.algorithm, I.quantize_int4, T, I.world, B.tile_m, B.tile_n, B.block.x,
        I.wave_size, B.pack_bytes, B.waves_per_eu, I.contiguous, I.tail>
        <<<dim3_of(grid), dim3_of(B.block), shared_memory_bytes, stream>>>(
        static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
        static_cast<int>(input_size_m), static_cast<int>(input_size_n),
        reinterpret_cast<const T* const*>(table), table_size, table_stride_m, table_stride_n,
        static_cast<T*>(output_ptr), output_stride_m, output_stride_n,
        static_cast<int>(output_size_m), static_cast<int>(output_size_n),
        partials_ptr_array, s.world, partials_stride_m,
        partials_stride_n, static_cast<int>(partials_size_m),
        static_cast<int>(partials_size_n), staging_ptr_array,
        s.world, staging_stride_m, staging_stride_n,
        static_cast<int>(staging_size_m), static_cast<int>(staging_size_n),
        unfenced_ptr_array, s.world, unfenced_stride_slot,
        unfenced_stride_m, unfenced_stride_n,
        static_cast<int>(unfenced_size_slot), static_cast<int>(unfenced_size_m),
        static_cast<int>(unfenced_size_n), unfenced_epochs,
        kUnfencedEpochsStrideBlock, int64_t{1},
        static_cast<int>(kUnfencedBlocks),
        static_cast<int>(kUnfencedEpochsStrideBlock), signals_ptr_array,
        s.world, kSignalsStrideParity, int64_t{kMaxRanks},
        int64_t{1}, int64_t{2}, int64_t{kMaxBlocks},
        int64_t{kMaxRanks}, s.status_device, int64_t{1},
        int64_t{kStatusFields}, s.rank, s.timeout_ticks);
  });
  // POSTCONDITION: the launch went in.
  if (hipGetLastError() != hipSuccess) return error_number(Error::hip_failed);
  return std::nullopt;
}

namespace {
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
  const int64_t weight_pack = hip_custom_communicator::kBuild.memory.pack_bytes /
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
}  // namespace

// all_reduce_rms_norm: at its config.
// One function: what Python passes, its tensors stripped to pointers and strides, the
// preconditions, the kernel's arguments, the launch, the postcondition.
Refused hip_custom_communicator_all_reduce_rms_norm(int64_t state,
    torch::Tensor& input, torch::Tensor& weight, torch::Tensor& output, double eps,
    const std::string& algorithm_name, int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t block_x,
    int64_t block_y, int64_t block_z, int64_t waves_per_eu, int64_t tile_m, int64_t tile_n,
    const std::string& quantization_name, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  using namespace hip_custom_communicator;
  // THE CALL FROM TORCH: its state, its tensors, its config's ints.
  if (state == 0) return error_number(Error::not_open);
  if (const std::optional<Error> e = norm_tensors(input, nullptr, weight, output, nullptr, /*in_place=*/false)) return error_number(e);
  const std::optional<DType> found_dtype = dtype_of_tensor(input);
  if (!found_dtype) return error_number(Error::dtype_not_built);
  const std::optional<DType> found_weight_dtype = dtype_of_tensor(weight);
  if (!found_weight_dtype) return error_number(Error::weight_not_built);
  if (!all_ints({waves_per_eu, tile_m, tile_n, pack_bytes, grid_x, grid_y, grid_z, block_x, block_y, block_z})) return error_number(Error::field_out_of_range);
  CommsState& s = *state_at(state);
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
  const DType dtype = *found_dtype;
  const DType weight_dtype = *found_weight_dtype;
  const Dim3 grid{static_cast<int>(grid_x), static_cast<int>(grid_y), static_cast<int>(grid_z)};
  const Dim3 block{static_cast<int>(block_x), static_cast<int>(block_y), static_cast<int>(block_z)};
  const hipStream_t stream = current_stream();
  // ---- PRECONDITIONS: the config, the call, then its instance. The first Error refuses the
  // call; past this section the call runs. ----
  const std::optional<AllReduceRmsNormAlgorithm> named = algorithm_named<AllReduceRmsNormBuildConfig>(algorithm_name);
  if (!named) return error_number(Error::no_such_kernel);
  const AllReduceRmsNormAlgorithm algorithm = *named;
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_number(Error::no_such_quantization);
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
    return error_number(e);
  if (const std::optional<Error> e =
          layout_refused(dtype, build.pack_bytes, {{output_stride_m, output_stride_n},
                                 {input_stride_m, input_stride_n}}))
    return error_number(e);
  if (const std::optional<Error> e =
          layout_refused(weight_dtype, build.pack_bytes, {{0, weight_stride_n}}))
    return error_number(e);
  const OpType op = OpType::all_reduce_rms_norm;
  if (const std::optional<Error> e = weight_refused(op, dtype, weight_dtype)) return error_number(e);
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
    return error_number(Error::tile_not_built);
  if (const std::optional<Error> e = check_kernel(kernel, grid, block, shared_memory_bytes, true))
    return error_number(e);

  // ---- THE LAUNCH: the buffers, the views of their regions, and the instance. ----
  // THIS CALL'S BUFFERS, from its arguments: the shared buffer's regions (`arena_layout`)
  // and the private buffer's arrays, epochs and tables (kPrivate*).
  char* const* const region_ptr_arrays =
      reinterpret_cast<char* const*>(s.private_buffer + kPrivateRegionPtrArrays);
  const auto signals_ptr_array =
      reinterpret_cast<uint32_t* const*>(region_ptr_arrays + kSignalsArray * kMaxRanks);
  char* const* const staging_ptr_array = region_ptr_arrays + kStagingArray * kMaxRanks;
  char* const* const partials_ptr_array = region_ptr_arrays + kPartialsArray * kMaxRanks;
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
    return error_number(Error::view_past_region);
  // THE INPUT TABLE (INPUT TABLES): where the kernel reads every rank's input in place (an exact
  // pull), or none where it reads the staging.
  void* const* table = nullptr;
  int64_t table_stride_m = staging_stride_m;
  if (in_place)
    if (const std::optional<Error> e = input_table(s, input_ptr, input_stride_m, input_size_m,
                                                   input_size_n, elem_bytes(dtype), stream, table,
                                                   table_stride_m))
      return error_number(e);
  const int table_size = in_place ? s.world : 0;
  const int64_t table_stride_n = in_place ? 1 : staging_stride_n;
  with_instance<OpType::all_reduce_rms_norm>(build, s.world, dtype, weight_dtype, contiguous, chunked, tail, /*quantize_int4=*/false, [&]<Instance I>() {
    constexpr auto B = I.build;
    using T = typename of_dtype<I.dtype>::t;
    using W = typename of_dtype<I.weight_dtype>::t;
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
        static_cast<int>(output_size_n), partials_ptr_array, s.world,
         partials_stride_m,
        partials_stride_n, static_cast<int>(partials_size_m),
        static_cast<int>(partials_size_n), staging_ptr_array,
            s.world, staging_stride_m,
        staging_stride_n, static_cast<int>(staging_size_m),
        static_cast<int>(staging_size_n), signals_ptr_array,
            s.world, kSignalsStrideParity,
        int64_t{kMaxRanks}, int64_t{1},
        int64_t{2}, int64_t{kMaxBlocks},
        int64_t{kMaxRanks}, s.status_device, int64_t{1},
        int64_t{kStatusFields}, s.rank, s.timeout_ticks, eps);
  });
  // POSTCONDITION: the launch went in.
  if (hipGetLastError() != hipSuccess) return error_number(Error::hip_failed);
  return std::nullopt;
}

// all_reduce_add_rms_norm: at its config.
// all_reduce_add_rms_norm: out, residual_out =
// fused_add_rms_norm(all_reduce(inp), residual, weight), vLLM's roundings
// exactly.
// One function: what Python passes, its tensors stripped to pointers and strides, the
// preconditions, the kernel's arguments, the launch, the postcondition.
Refused hip_custom_communicator_all_reduce_add_rms_norm(int64_t state,
    torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, torch::Tensor& output,
    torch::Tensor& residual_output, double eps, const std::string& algorithm_name, int64_t grid_x,
    int64_t grid_y, int64_t grid_z, int64_t block_x, int64_t block_y, int64_t block_z,
    int64_t waves_per_eu, int64_t tile_m, int64_t tile_n, const std::string& quantization_name,
    int64_t pack_bytes, std::optional<int64_t> quantize_from_bytes) {
  using namespace hip_custom_communicator;
  // THE CALL FROM TORCH: its state, its tensors, its config's ints.
  if (state == 0) return error_number(Error::not_open);
  if (const std::optional<Error> e = norm_tensors(input, &residual, weight, output, &residual_output, /*in_place=*/false)) return error_number(e);
  const std::optional<DType> found_dtype = dtype_of_tensor(input);
  if (!found_dtype) return error_number(Error::dtype_not_built);
  const std::optional<DType> found_weight_dtype = dtype_of_tensor(weight);
  if (!found_weight_dtype) return error_number(Error::weight_not_built);
  if (!all_ints({waves_per_eu, tile_m, tile_n, pack_bytes, grid_x, grid_y, grid_z, block_x, block_y, block_z})) return error_number(Error::field_out_of_range);
  CommsState& s = *state_at(state);
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
  const DType dtype = *found_dtype;
  const DType weight_dtype = *found_weight_dtype;
  const Dim3 grid{static_cast<int>(grid_x), static_cast<int>(grid_y), static_cast<int>(grid_z)};
  const Dim3 block{static_cast<int>(block_x), static_cast<int>(block_y), static_cast<int>(block_z)};
  const hipStream_t stream = current_stream();
  // ---- PRECONDITIONS: the config, the call, then its instance. The first Error refuses the
  // call; past this section the call runs. ----
  const std::optional<AllReduceAddRmsNormAlgorithm> named = algorithm_named<AllReduceAddRmsNormBuildConfig>(algorithm_name);
  if (!named) return error_number(Error::no_such_kernel);
  const AllReduceAddRmsNormAlgorithm algorithm = *named;
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_number(Error::no_such_quantization);
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
    return error_number(e);
  if (const std::optional<Error> e = layout_refused(
          dtype, build.pack_bytes, {{output_stride_m, output_stride_n},
                  {residual_output_stride_m, residual_output_stride_n},
                  {input_stride_m, input_stride_n},
                  {residual_stride_m, residual_stride_n}}))
    return error_number(e);
  if (const std::optional<Error> e =
          layout_refused(weight_dtype, build.pack_bytes, {{0, weight_stride_n}}))
    return error_number(e);
  const OpType op = OpType::all_reduce_add_rms_norm;
  if (const std::optional<Error> e = weight_refused(op, dtype, weight_dtype)) return error_number(e);
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
    return error_number(Error::partials_too_small);
  const void* kernel = nullptr;
  if (!with_instance<OpType::all_reduce_add_rms_norm>(build, s.world, dtype, weight_dtype, contiguous, chunked, tail, /*quantize_int4=*/false, [&]<Instance I>() { kernel = kernel_pointer<I>(); }))
    return error_number(Error::tile_not_built);
  if (const std::optional<Error> e = check_kernel(kernel, grid, block, shared_memory_bytes, true))
    return error_number(e);

  // ---- THE LAUNCH: the buffers, the views of their regions, and the instance. ----
  // THIS CALL'S BUFFERS, from its arguments: the shared buffer's regions (`arena_layout`)
  // and the private buffer's arrays, epochs and tables (kPrivate*).
  char* const* const region_ptr_arrays =
      reinterpret_cast<char* const*>(s.private_buffer + kPrivateRegionPtrArrays);
  const auto signals_ptr_array =
      reinterpret_cast<uint32_t* const*>(region_ptr_arrays + kSignalsArray * kMaxRanks);
  char* const* const staging_ptr_array = region_ptr_arrays + kStagingArray * kMaxRanks;
  char* const* const partials_ptr_array = region_ptr_arrays + kPartialsArray * kMaxRanks;
  const auto scales_ptr_array =
      reinterpret_cast<float* const*>(region_ptr_arrays + kScalesArray * kMaxRanks);
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
    return error_number(Error::view_past_region);
  // THE INPUT TABLE (INPUT TABLES): where the kernel reads every rank's input in place (an exact
  // pull), or none where it reads the staging.
  void* const* table = nullptr;
  int64_t table_stride_m = staging_stride_m;
  if (in_place)
    if (const std::optional<Error> e = input_table(s, input_ptr, input_stride_m, input_size_m,
                                                   input_size_n, elem_bytes(dtype), stream, table,
                                                   table_stride_m))
      return error_number(e);
  const int table_size = in_place ? s.world : 0;
  const int64_t table_stride_n = in_place ? 1 : staging_stride_n;
  with_instance<OpType::all_reduce_add_rms_norm>(build, s.world, dtype, weight_dtype, contiguous, chunked, tail, /*quantize_int4=*/false, [&]<Instance I>() {
    constexpr auto B = I.build;
    using T = typename of_dtype<I.dtype>::t;
    using W = typename of_dtype<I.weight_dtype>::t;
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
        static_cast<int>(residual_output_size_n), partials_ptr_array,
            s.world,
         partials_stride_m,
        partials_stride_n, static_cast<int>(partials_size_m),
        static_cast<int>(partials_size_n), scales_ptr_array,
        s.world, scales_stride_m, static_cast<int>(scales_size_m),
        staging_ptr_array, s.world, staging_stride_m,
        staging_stride_n, static_cast<int>(staging_size_m),
        static_cast<int>(staging_size_n), signals_ptr_array,
            s.world, kSignalsStrideParity,
        int64_t{kMaxRanks}, int64_t{1},
        int64_t{2}, int64_t{kMaxBlocks},
        int64_t{kMaxRanks}, s.status_device, int64_t{1},
        int64_t{kStatusFields}, s.rank, s.timeout_ticks, eps);
  });
  // POSTCONDITION: the launch went in.
  if (hipGetLastError() != hipSuccess) return error_number(Error::hip_failed);
  return std::nullopt;
}

namespace {
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
          hip_custom_communicator::kBuild.memory.pack_bytes != 0)
    return Error::not_aligned;
  // The sources read are the first num_blocks; block_output may be a later slot of the same stack.
  const torch::Tensor sources = blocks.narrow(1, 0, num_blocks);
  if (overlapping({&input, prefix ? &*prefix : nullptr, &sources, &norm_weight, &qk_weight,
                   output_norm_weight ? &*output_norm_weight : nullptr},
                  {&prefix_output, &output, block_output ? &*block_output : nullptr}))
    return Error::inputs_overlap_outputs;
  return std::nullopt;
}

// A [rows, cols] TENSOR'S ARGUMENTS, none's a null pointer and zeros.
struct Rows {
  void* ptr;
  int64_t stride_m, stride_n, size_m, size_n;
};
Rows rows_of(const std::optional<torch::Tensor>& t) {
  if (!t) return {nullptr, 0, 0, 0, 0};
  return {t->data_ptr(), t->stride(0), t->stride(1), t->size(0), t->size(1)};
}
}  // namespace

// all_reduce_add_attn_res_rms_norm: at its config.
// One function: what Python passes, its tensors stripped to pointers and strides, the
// preconditions, the kernel's arguments, the launch, the postcondition.
Refused hip_custom_communicator_all_reduce_add_attn_res_rms_norm(int64_t state,
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
  if (state == 0) return error_number(Error::not_open);
  if (const std::optional<Error> e = attn_res_tensors(input, prefix, blocks, norm_weight, qk_weight,
                   output_norm_weight, prefix_output, output, block_output,
                   num_blocks))
    return error_number(e);
  const Rows old = rows_of(prefix), block = rows_of(block_output);
  const std::optional<DType> found_dtype = dtype_of_tensor(input);
  if (!found_dtype) return error_number(Error::dtype_not_built);
  if (!all_ints({waves_per_eu, tile_m, tile_n, tile_k, pack_bytes, grid_x, grid_y, grid_z, block_x, block_y, block_z})) return error_number(Error::field_out_of_range);
  CommsState& s = *state_at(state);
  const void* input_ptr = input.data_ptr();
  const int64_t input_stride_m = input.stride(0);
  const int64_t input_stride_n = input.stride(1);
  const int64_t input_size_m = input.size(0);
  const int64_t input_size_n = input.size(1);
  const void* prefix_ptr = old.ptr;
  const int64_t prefix_stride_m = old.stride_m;
  const int64_t prefix_stride_n = old.stride_n;
  const int64_t prefix_size_m = old.size_m;
  const int64_t prefix_size_n = old.size_n;
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
  void* block_output_ptr = block.ptr;
  const int64_t block_output_stride_m = block.stride_m;
  const int64_t block_output_stride_n = block.stride_n;
  const int64_t block_output_size_m = block.size_m;
  const int64_t block_output_size_n = block.size_n;
  const DType dtype = *found_dtype;
  const Dim3 grid{static_cast<int>(grid_x), static_cast<int>(grid_y), static_cast<int>(grid_z)};
  const Dim3 block{static_cast<int>(block_x), static_cast<int>(block_y), static_cast<int>(block_z)};
  const hipStream_t stream = current_stream();
  // ---- PRECONDITIONS: the config, the call, then its instance. The first Error refuses the
  // call; past this section the call runs. ----
  const std::optional<AllReduceAddAttnResRmsNormAlgorithm> named = algorithm_named<AllReduceAddAttnResRmsNormBuildConfig>(algorithm_name);
  if (!named) return error_number(Error::no_such_kernel);
  const AllReduceAddAttnResRmsNormAlgorithm algorithm = *named;
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_number(Error::no_such_quantization);
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
    return error_number(e);
  if (const std::optional<Error> e = layout_refused(
          dtype, build.pack_bytes, {{input_stride_m, input_stride_n},
                  {blocks_stride_m, blocks_stride_n},
                  {blocks_stride_r, blocks_stride_n},
                  {0, norm_weight_stride_n},
                  {0, qk_weight_stride_n},
                  {prefix_output_stride_m, prefix_output_stride_n},
                  {output_stride_m, output_stride_n}}))
    return error_number(e);
  // THE OPTIONAL BUFFERS, checked only when given: a missing one is a null
  // pointer with zero strides, which is no layout at all (a prefix-less call
  // was refused as `inner_stride_not_one`, 2026-10-13).
  if (prefix_ptr != nullptr)
    if (const std::optional<Error> e =
            layout_refused(dtype, build.pack_bytes, {{prefix_stride_m, prefix_stride_n}}))
      return error_number(e);
  if (output_norm_weight_ptr != nullptr)
    if (const std::optional<Error> e =
            layout_refused(dtype, build.pack_bytes, {{0, output_norm_weight_stride_n}}))
      return error_number(e);
  if (block_output_ptr != nullptr)
    if (const std::optional<Error> e = layout_refused(
            dtype, build.pack_bytes, {{block_output_stride_m, block_output_stride_n}}))
      return error_number(e);
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
    return error_number(Error::tile_not_built);
  if (const std::optional<Error> e = check_kernel(kernel, grid, block, shared_memory_bytes, true))
    return error_number(e);

  // ---- THE LAUNCH: the buffers, the views of their regions, and the instance. ----
  // THIS CALL'S BUFFERS, from its arguments: the shared buffer's regions (`arena_layout`)
  // and the private buffer's arrays, epochs and tables (kPrivate*).
  char* const* const region_ptr_arrays =
      reinterpret_cast<char* const*>(s.private_buffer + kPrivateRegionPtrArrays);
  const auto signals_ptr_array =
      reinterpret_cast<uint32_t* const*>(region_ptr_arrays + kSignalsArray * kMaxRanks);
  char* const* const staging_ptr_array = region_ptr_arrays + kStagingArray * kMaxRanks;
  char* const* const partials_ptr_array = region_ptr_arrays + kPartialsArray * kMaxRanks;
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
    return error_number(Error::view_past_region);
  // THE INPUT TABLE (INPUT TABLES): where the kernel reads every rank's input in place (an exact
  // pull), or none where it reads the staging.
  void* const* table = nullptr;
  int64_t table_stride_m = staging_stride_m;
  if (in_place)
    if (const std::optional<Error> e = input_table(s, input_ptr, input_stride_m, input_size_m,
                                                   input_size_n, elem_bytes(dtype), stream, table,
                                                   table_stride_m))
      return error_number(e);
  const int table_size = in_place ? s.world : 0;
  const int64_t table_stride_n = in_place ? 1 : staging_stride_n;
  with_instance<OpType::all_reduce_add_attn_res_rms_norm>(build, s.world, dtype, dtype, contiguous, chunked, tail, /*quantize_int4=*/false, [&]<Instance I>() {
    constexpr auto B = I.build;
    using T = typename of_dtype<I.dtype>::t;
    using W = typename of_dtype<I.weight_dtype>::t;
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
        static_cast<int>(block_output_size_n), partials_ptr_array,
        s.world, partials_stride_m, partials_stride_n,
        static_cast<int>(partials_size_m), static_cast<int>(partials_size_n),
        staging_ptr_array, s.world, staging_stride_m,
        staging_stride_n, static_cast<int>(staging_size_m),
        static_cast<int>(staging_size_n), signals_ptr_array,
            s.world, kSignalsStrideParity,
        int64_t{kMaxRanks}, int64_t{1},
        int64_t{2}, int64_t{kMaxBlocks},
        int64_t{kMaxRanks}, s.status_device, int64_t{1},
        int64_t{kStatusFields}, s.rank, s.timeout_ticks, num_blocks, eps,
        output_eps);
  });
  // POSTCONDITION: the launch went in.
  if (hipGetLastError() != hipSuccess) return error_number(Error::hip_failed);
  return std::nullopt;
}

// rms_norm: this rank's rows normed (the second kernel of an op Python composes), at its config.
// rms_norm: out = rms_norm(inp, weight), vLLM's roundings exactly, on this rank's rows; the
// weight `weight_dtype`, dtype or f32.
// One function: what Python passes, its tensors stripped to pointers and strides, the
// preconditions, the kernel's arguments, the launch, the postcondition.
Refused hip_custom_communicator_rms_norm(int64_t state, torch::Tensor& input, torch::Tensor& weight,
    torch::Tensor& output, double eps, int64_t grid_x, int64_t grid_y, int64_t grid_z,
    int64_t block_x, int64_t block_y, int64_t block_z, int64_t waves_per_eu, int64_t tile_m,
    int64_t tile_n, const std::string& quantization_name, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  using namespace hip_custom_communicator;
  // THE CALL FROM TORCH: its state, its tensors, its config's ints.
  if (state == 0) return error_number(Error::not_open);
  if (const std::optional<Error> e = norm_tensors(input, nullptr, weight, output, nullptr, /*in_place=*/true)) return error_number(e);
  const std::optional<DType> found_dtype = dtype_of_tensor(input);
  if (!found_dtype) return error_number(Error::dtype_not_built);
  const std::optional<DType> found_weight_dtype = dtype_of_tensor(weight);
  if (!found_weight_dtype) return error_number(Error::weight_not_built);
  if (!all_ints({waves_per_eu, tile_m, tile_n, pack_bytes, grid_x, grid_y, grid_z, block_x, block_y, block_z})) return error_number(Error::field_out_of_range);
  CommsState& s = *state_at(state);
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
  const DType dtype = *found_dtype;
  const DType weight_dtype = *found_weight_dtype;
  const Dim3 grid{static_cast<int>(grid_x), static_cast<int>(grid_y), static_cast<int>(grid_z)};
  const Dim3 block{static_cast<int>(block_x), static_cast<int>(block_y), static_cast<int>(block_z)};
  const hipStream_t stream = current_stream();
  // ---- PRECONDITIONS: the config, the call, then its instance. The first Error refuses the
  // call; past this section the call runs. ----
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_number(Error::no_such_quantization);
  const RmsNormBuildConfig build{.block = block,
      .waves_per_eu = static_cast<int>(waves_per_eu),
      .tile_m = static_cast<int>(tile_m),
      .tile_n = static_cast<int>(tile_n),
      .quantization = quantization_name == "int4" ? Quantization::int4 : Quantization::exact,
      .pack_bytes = static_cast<int>(pack_bytes)};
  const int64_t m = input_size_m, n = input_size_n;  // the problem's shape: the input's
  if (const std::optional<Error> e =
          refused(false, s.world, 0, 0, build, grid, quantize_from_bytes, dtype, m, n))
    return error_number(e);
  if (const std::optional<Error> e =
          layout_refused(dtype, build.pack_bytes, {{output_stride_m, output_stride_n},
                                 {input_stride_m, input_stride_n}}))
    return error_number(e);
  if (const std::optional<Error> e =
          layout_refused(weight_dtype, build.pack_bytes, {{0, weight_stride_n}}))
    return error_number(e);
  const OpType op = OpType::rms_norm;
  if (const std::optional<Error> e = weight_refused(op, dtype, weight_dtype)) return error_number(e);
  const int shared_memory_bytes = 0;
  // A ROW WIDER THAN THE TILE runs the build's CHUNKED instance.
  const bool chunked = n > build.tile_n;
  const void* kernel = nullptr;
  if (!with_instance<OpType::rms_norm>(build, s.world, dtype, weight_dtype, /*contiguous=*/false, chunked, /*tail=*/false, /*quantize_int4=*/false, [&]<Instance I>() { kernel = kernel_pointer<I>(); }))
    return error_number(Error::tile_not_built);
  if (const std::optional<Error> e = check_kernel(kernel, grid, block, shared_memory_bytes, false))
    return error_number(e);

  // ---- THE LAUNCH: the buffers, the views of their regions, and the instance. ----
  with_instance<OpType::rms_norm>(build, s.world, dtype, weight_dtype, /*contiguous=*/false, chunked, /*tail=*/false, /*quantize_int4=*/false, [&]<Instance I>() {
    constexpr auto B = I.build;
    using T = typename of_dtype<I.dtype>::t;
    using W = typename of_dtype<I.weight_dtype>::t;
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
  if (hipGetLastError() != hipSuccess) return error_number(Error::hip_failed);
  return std::nullopt;
}

// add_rms_norm: this rank's rows added and normed (the second kernel of an op Python composes),
// at its config.
// add_rms_norm: out, residual_out = fused_add_rms_norm(inp, residual, weight), vLLM's roundings
// exactly, on this rank's rows.
// One function: what Python passes, its tensors stripped to pointers and strides, the
// preconditions, the kernel's arguments, the launch, the postcondition.
Refused hip_custom_communicator_add_rms_norm(int64_t state, torch::Tensor& input,
    torch::Tensor& residual, torch::Tensor& weight, torch::Tensor& output,
    torch::Tensor& residual_output, double eps, int64_t grid_x, int64_t grid_y, int64_t grid_z,
    int64_t block_x, int64_t block_y, int64_t block_z, int64_t waves_per_eu, int64_t tile_m,
    int64_t tile_n, const std::string& quantization_name, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  using namespace hip_custom_communicator;
  // THE CALL FROM TORCH: its state, its tensors, its config's ints.
  if (state == 0) return error_number(Error::not_open);
  if (const std::optional<Error> e = norm_tensors(input, &residual, weight, output, &residual_output, /*in_place=*/true)) return error_number(e);
  const std::optional<DType> found_dtype = dtype_of_tensor(input);
  if (!found_dtype) return error_number(Error::dtype_not_built);
  const std::optional<DType> found_weight_dtype = dtype_of_tensor(weight);
  if (!found_weight_dtype) return error_number(Error::weight_not_built);
  if (!all_ints({waves_per_eu, tile_m, tile_n, pack_bytes, grid_x, grid_y, grid_z, block_x, block_y, block_z})) return error_number(Error::field_out_of_range);
  CommsState& s = *state_at(state);
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
  const DType dtype = *found_dtype;
  const DType weight_dtype = *found_weight_dtype;
  const Dim3 grid{static_cast<int>(grid_x), static_cast<int>(grid_y), static_cast<int>(grid_z)};
  const Dim3 block{static_cast<int>(block_x), static_cast<int>(block_y), static_cast<int>(block_z)};
  const hipStream_t stream = current_stream();
  // ---- PRECONDITIONS: the config, the call, then its instance. The first Error refuses the
  // call; past this section the call runs. ----
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_number(Error::no_such_quantization);
  const AddRmsNormBuildConfig build{.block = block,
      .waves_per_eu = static_cast<int>(waves_per_eu),
      .tile_m = static_cast<int>(tile_m),
      .tile_n = static_cast<int>(tile_n),
      .quantization = quantization_name == "int4" ? Quantization::int4 : Quantization::exact,
      .pack_bytes = static_cast<int>(pack_bytes)};
  const int64_t m = input_size_m, n = input_size_n;  // the problem's shape: the input's
  if (const std::optional<Error> e =
          refused(false, s.world, 0, 0, build, grid, quantize_from_bytes, dtype, m, n))
    return error_number(e);
  if (const std::optional<Error> e = layout_refused(
          dtype, build.pack_bytes, {{output_stride_m, output_stride_n},
                  {residual_output_stride_m, residual_output_stride_n},
                  {input_stride_m, input_stride_n},
                  {residual_stride_m, residual_stride_n}}))
    return error_number(e);
  if (const std::optional<Error> e =
          layout_refused(weight_dtype, build.pack_bytes, {{0, weight_stride_n}}))
    return error_number(e);
  const OpType op = OpType::add_rms_norm;
  if (const std::optional<Error> e = weight_refused(op, dtype, weight_dtype)) return error_number(e);
  const int shared_memory_bytes = 0;
  // A ROW WIDER THAN THE TILE runs the build's CHUNKED instance.
  const bool chunked = n > build.tile_n;
  const void* kernel = nullptr;
  if (!with_instance<OpType::add_rms_norm>(build, s.world, dtype, weight_dtype, /*contiguous=*/false, chunked, /*tail=*/false, /*quantize_int4=*/false, [&]<Instance I>() { kernel = kernel_pointer<I>(); }))
    return error_number(Error::tile_not_built);
  if (const std::optional<Error> e = check_kernel(kernel, grid, block, shared_memory_bytes, false))
    return error_number(e);

  // ---- THE LAUNCH: the buffers, the views of their regions, and the instance. ----
  with_instance<OpType::add_rms_norm>(build, s.world, dtype, weight_dtype, /*contiguous=*/false, chunked, /*tail=*/false, /*quantize_int4=*/false, [&]<Instance I>() {
    constexpr auto B = I.build;
    using T = typename of_dtype<I.dtype>::t;
    using W = typename of_dtype<I.weight_dtype>::t;
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
  if (hipGetLastError() != hipSuccess) return error_number(Error::hip_failed);
  return std::nullopt;
}

// add_attn_res_rms_norm: AttnRes on this rank's rows (the second kernel of the op Python
// composes), at its config.
// add_attn_res_rms_norm: Kimi-K3's AttnRes and its RMSNorm on each of this rank's rows
// (all_reduce_add_attn_res_rms_norm's, its input the sum), `num_blocks` sources.
// One function: what Python passes, its tensors stripped to pointers and strides, the
// preconditions, the kernel's arguments, the launch, the postcondition.
Refused hip_custom_communicator_add_attn_res_rms_norm(int64_t state, torch::Tensor& input,
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
  if (state == 0) return error_number(Error::not_open);
  if (const std::optional<Error> e = attn_res_tensors(input, prefix, blocks, norm_weight, qk_weight,
                   output_norm_weight, prefix_output, output, block_output,
                   num_blocks))
    return error_number(e);
  const Rows old = rows_of(prefix), block = rows_of(block_output);
  const std::optional<DType> found_dtype = dtype_of_tensor(input);
  if (!found_dtype) return error_number(Error::dtype_not_built);
  if (!all_ints({waves_per_eu, tile_m, tile_n, tile_k, pack_bytes, grid_x, grid_y, grid_z, block_x, block_y, block_z})) return error_number(Error::field_out_of_range);
  CommsState& s = *state_at(state);
  const void* input_ptr = input.data_ptr();
  const int64_t input_stride_m = input.stride(0);
  const int64_t input_stride_n = input.stride(1);
  const int64_t input_size_m = input.size(0);
  const int64_t input_size_n = input.size(1);
  const void* prefix_ptr = old.ptr;
  const int64_t prefix_stride_m = old.stride_m;
  const int64_t prefix_stride_n = old.stride_n;
  const int64_t prefix_size_m = old.size_m;
  const int64_t prefix_size_n = old.size_n;
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
  void* block_output_ptr = block.ptr;
  const int64_t block_output_stride_m = block.stride_m;
  const int64_t block_output_stride_n = block.stride_n;
  const int64_t block_output_size_m = block.size_m;
  const int64_t block_output_size_n = block.size_n;
  const DType dtype = *found_dtype;
  const Dim3 grid{static_cast<int>(grid_x), static_cast<int>(grid_y), static_cast<int>(grid_z)};
  const Dim3 block{static_cast<int>(block_x), static_cast<int>(block_y), static_cast<int>(block_z)};
  const hipStream_t stream = current_stream();
  // ---- PRECONDITIONS: the config, the call, then its instance. The first Error refuses the
  // call; past this section the call runs. ----
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_number(Error::no_such_quantization);
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
    return error_number(e);
  if (const std::optional<Error> e = layout_refused(
          dtype, build.pack_bytes, {{input_stride_m, input_stride_n},
                  {blocks_stride_m, blocks_stride_n},
                  {blocks_stride_r, blocks_stride_n},
                  {0, norm_weight_stride_n},
                  {0, qk_weight_stride_n},
                  {prefix_output_stride_m, prefix_output_stride_n},
                  {output_stride_m, output_stride_n}}))
    return error_number(e);
  // THE OPTIONAL BUFFERS, checked only when given (all_reduce_add_attn_res_rms_norm's).
  if (prefix_ptr != nullptr)
    if (const std::optional<Error> e =
            layout_refused(dtype, build.pack_bytes, {{prefix_stride_m, prefix_stride_n}}))
      return error_number(e);
  if (output_norm_weight_ptr != nullptr)
    if (const std::optional<Error> e =
            layout_refused(dtype, build.pack_bytes, {{0, output_norm_weight_stride_n}}))
      return error_number(e);
  if (block_output_ptr != nullptr)
    if (const std::optional<Error> e = layout_refused(
            dtype, build.pack_bytes, {{block_output_stride_m, block_output_stride_n}}))
      return error_number(e);
  // A ROW WIDER THAN THE TILE keeps its rows' logits in the launch's LDS (attn_res.cuh's
  // attn_res_wide): [tile_m][num_blocks + 1] floats.
  const int shared_memory_bytes = n > tile_n ? static_cast<int>(tile_m * (num_blocks + 1) * sizeof(float)) : 0;
  // A ROW WIDER THAN THE TILE runs the build's CHUNKED instance.
  const bool chunked = n > build.tile_n;
  const void* kernel = nullptr;
  if (!with_instance<OpType::add_attn_res_rms_norm>(build, s.world, dtype, dtype, /*contiguous=*/false, chunked, /*tail=*/false, /*quantize_int4=*/false, [&]<Instance I>() { kernel = kernel_pointer<I>(); }))
    return error_number(Error::tile_not_built);
  if (const std::optional<Error> e = check_kernel(kernel, grid, block, shared_memory_bytes, false))
    return error_number(e);

  // ---- THE LAUNCH: the buffers, the views of their regions, and the instance. ----
  with_instance<OpType::add_attn_res_rms_norm>(build, s.world, dtype, dtype, /*contiguous=*/false, chunked, /*tail=*/false, /*quantize_int4=*/false, [&]<Instance I>() {
    constexpr auto B = I.build;
    using T = typename of_dtype<I.dtype>::t;
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
  if (hipGetLastError() != hipSuccess) return error_number(Error::hip_failed);
  return std::nullopt;
}
