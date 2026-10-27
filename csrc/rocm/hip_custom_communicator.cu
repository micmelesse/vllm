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
std::optional<DType> dtype_of(const torch::Tensor& t) {
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
Refused refused(std::optional<Error> e) {
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
    return {refused(Error::field_out_of_range), 0, {}, "", {}};
  auto* s = new CommsState();
  std::optional<Error> e = open_state(*s, static_cast<int>(rank), static_cast<int>(world),
                                      static_cast<int>(device), max_rows, max_cols,
                                      timeout_seconds);
  hipIpcMemHandle_t h{};
  if (!e) e = ipc_handle(*s, h);
  if (e) {
    (void)free_state(*s);
    delete s;
    return {refused(e), 0, {}, "", {}};
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
  if (state == 0) return refused(Error::not_open);
  hip_custom_communicator::CommsState& s = *state_at(state);
  if (handles.size() != s.world * sizeof(hipIpcMemHandle_t)) return refused(Error::world_not_built);
  std::vector<hipIpcMemHandle_t> by_rank;
  for (int r = 0; r < s.world; ++r) by_rank.push_back(ipc_from(handles, r));
  return refused(hip_custom_communicator::connect_state(s, by_rank));
}

// THE CAPTURED LAUNCHES' INPUTS (vLLM's get_graph_buffer_ipc_meta): (the Error or none; their
// allocations' IPC handles' bytes, an input after another; each input's offset in its own).
GraphMetaWire hip_custom_communicator_graph_ipc_meta(int64_t state) {
  if (state == 0) return {refused(Error::not_open), {}, {}};
  std::vector<hipIpcMemHandle_t> handles;
  std::vector<int64_t> offsets;
  if (const std::optional<Error> e =
          hip_custom_communicator::graph_ipc_meta(*state_at(state), handles, offsets))
    return {refused(e), {}, {}};
  std::vector<int64_t> bytes;
  for (const hipIpcMemHandle_t& h : handles) ipc_into(h, bytes);
  return {std::nullopt, bytes, offsets};
}

// THE CAPTURED LAUNCHES' TABLES (vLLM's register_graph_buffers): every rank's
// `graph_ipc_meta`, rank after rank.
Refused hip_custom_communicator_register_graph_buffers(int64_t state,
                                                       const std::vector<int64_t>& handles,
                                                       const std::vector<int64_t>& offsets) {
  if (state == 0) return refused(Error::not_open);
  if (handles.size() != offsets.size() * sizeof(hipIpcMemHandle_t))
    return refused(Error::ranks_disagree);
  std::vector<hipIpcMemHandle_t> by_input;
  for (size_t i = 0; i < offsets.size(); ++i) by_input.push_back(ipc_from(handles, i));
  return refused(
      hip_custom_communicator::register_graph_buffers(*state_at(state), by_input, offsets));
}

// CLOSE THE PEERS: every borrowed mapping closed (every launch finished, every rank stopped).
Refused hip_custom_communicator_close_peers(int64_t state) {
  if (state == 0) return refused(Error::not_open);
  return refused(hip_custom_communicator::close_peers(*state_at(state)));
}

// FREE: this rank's buffers freed and the state with them (no peer has them open).
Refused hip_custom_communicator_free(int64_t state) {
  if (state == 0) return refused(Error::not_open);
  hip_custom_communicator::CommsState* s = state_at(state);
  const std::optional<Error> e = hip_custom_communicator::free_state(*s);
  delete s;
  return refused(e);
}

// all_reduce: at its config.
Refused hip_custom_communicator_all_reduce(int64_t state, torch::Tensor& input,
    torch::Tensor& output, const std::string& algorithm, int64_t grid_x, int64_t grid_y,
    int64_t grid_z, int64_t block_x, int64_t block_y, int64_t block_z, int64_t waves_per_eu,
    int64_t tile_m, int64_t tile_n, const std::string& quantization, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  if (state == 0) return refused(Error::not_open);
  hip_custom_communicator::CommsState& s = *state_at(state);
  if (!on_device({&input, &output})) return refused(Error::not_on_device);
  if (output.sizes() != input.sizes()) return refused(Error::shape_mismatch);
  if (output.scalar_type() != input.scalar_type()) return refused(Error::dtype_mismatch);
  if (overlapping({&input}, {&output})) return refused(Error::inputs_overlap_outputs);
  const std::optional<torch::Tensor> input_rows = as_rows(input),
                                     output_rows = as_rows(output);
  if (!input_rows || !output_rows)
    return refused(Error::not_contiguous);
  const std::optional<DType> dtype = dtype_of(input);
  if (!dtype) return refused(Error::dtype_not_built);
  if (!all_ints({waves_per_eu, tile_m, tile_n, pack_bytes, grid_x, grid_y, grid_z, block_x, block_y, block_z})) return refused(Error::field_out_of_range);
  const auto got = hip_custom_communicator::all_reduce(
      s, input_rows->data_ptr(), input_rows->stride(0),
      input_rows->stride(1), input_rows->size(0), input_rows->size(1),
      output_rows->data_ptr(), output_rows->stride(0), output_rows->stride(1),
      output_rows->size(0), output_rows->size(1), *dtype,
      algorithm, {static_cast<int>(grid_x), static_cast<int>(grid_y), static_cast<int>(grid_z)}, {static_cast<int>(block_x), static_cast<int>(block_y), static_cast<int>(block_z)}, static_cast<int>(waves_per_eu), static_cast<int>(tile_m), static_cast<int>(tile_n), quantization, static_cast<int>(pack_bytes), quantize_from_bytes,
      current_stream());
  return refused(got);
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
Refused hip_custom_communicator_all_reduce_rms_norm(int64_t state,
    torch::Tensor& input, torch::Tensor& weight, torch::Tensor& output, double eps,
    const std::string& algorithm, int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t block_x,
    int64_t block_y, int64_t block_z, int64_t waves_per_eu, int64_t tile_m, int64_t tile_n,
    const std::string& quantization, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  if (state == 0) return refused(Error::not_open);
  hip_custom_communicator::CommsState& s = *state_at(state);
  if (const std::optional<Error> e = norm_tensors(input, nullptr, weight, output, nullptr, /*in_place=*/false)) return refused(e);
  const std::optional<DType> dtype = dtype_of(input);
  if (!dtype) return refused(Error::dtype_not_built);
  const std::optional<DType> weight_dtype = *weight_dtype;
  if (!weight_dtype) return refused(Error::weight_not_built);
  if (!all_ints({waves_per_eu, tile_m, tile_n, pack_bytes, grid_x, grid_y, grid_z, block_x, block_y, block_z})) return refused(Error::field_out_of_range);
  const auto got = hip_custom_communicator::all_reduce_rms_norm(
      s, input.data_ptr(), input.stride(0), input.stride(1),
      input.size(0), input.size(1), weight.data_ptr(), weight.stride(0),
      weight.size(0), output.data_ptr(), output.stride(0), output.stride(1),
      output.size(0), output.size(1), *dtype, *weight_dtype,
      static_cast<float>(eps),
      algorithm, {static_cast<int>(grid_x), static_cast<int>(grid_y), static_cast<int>(grid_z)}, {static_cast<int>(block_x), static_cast<int>(block_y), static_cast<int>(block_z)}, static_cast<int>(waves_per_eu), static_cast<int>(tile_m), static_cast<int>(tile_n), quantization, static_cast<int>(pack_bytes), quantize_from_bytes,
      current_stream());
  return refused(got);
}

// all_reduce_add_rms_norm: at its config.
Refused hip_custom_communicator_all_reduce_add_rms_norm(int64_t state,
    torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, torch::Tensor& output,
    torch::Tensor& residual_output, double eps, const std::string& algorithm, int64_t grid_x,
    int64_t grid_y, int64_t grid_z, int64_t block_x, int64_t block_y, int64_t block_z,
    int64_t waves_per_eu, int64_t tile_m, int64_t tile_n, const std::string& quantization,
    int64_t pack_bytes, std::optional<int64_t> quantize_from_bytes) {
  if (state == 0) return refused(Error::not_open);
  hip_custom_communicator::CommsState& s = *state_at(state);
  if (const std::optional<Error> e = norm_tensors(input, &residual, weight, output, &residual_output, /*in_place=*/false)) return refused(e);
  const std::optional<DType> dtype = dtype_of(input);
  if (!dtype) return refused(Error::dtype_not_built);
  const std::optional<DType> weight_dtype = *weight_dtype;
  if (!weight_dtype) return refused(Error::weight_not_built);
  if (!all_ints({waves_per_eu, tile_m, tile_n, pack_bytes, grid_x, grid_y, grid_z, block_x, block_y, block_z})) return refused(Error::field_out_of_range);
  const auto got = hip_custom_communicator::all_reduce_add_rms_norm(
      s, input.data_ptr(), input.stride(0), input.stride(1),
      input.size(0), input.size(1), residual.data_ptr(), residual.stride(0),
      residual.stride(1), residual.size(0), residual.size(1), weight.data_ptr(),
      weight.stride(0), weight.size(0), output.data_ptr(), output.stride(0),
      output.stride(1), output.size(0), output.size(1),
      residual_output.data_ptr(), residual_output.stride(0),
      residual_output.stride(1), residual_output.size(0),
      residual_output.size(1), *dtype, *weight_dtype,
      static_cast<float>(eps),
      algorithm, {static_cast<int>(grid_x), static_cast<int>(grid_y), static_cast<int>(grid_z)}, {static_cast<int>(block_x), static_cast<int>(block_y), static_cast<int>(block_z)}, static_cast<int>(waves_per_eu), static_cast<int>(tile_m), static_cast<int>(tile_n), quantization, static_cast<int>(pack_bytes), quantize_from_bytes,
      current_stream());
  return refused(got);
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
Refused hip_custom_communicator_all_reduce_add_attn_res_rms_norm(int64_t state,
    torch::Tensor& input, const std::optional<torch::Tensor>& prefix, torch::Tensor& blocks,
    torch::Tensor& norm_weight, torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& output_norm_weight, torch::Tensor& prefix_output,
    torch::Tensor& output, const std::optional<torch::Tensor>& block_output, int64_t num_blocks,
    double eps, double output_eps, const std::string& algorithm, int64_t grid_x, int64_t grid_y,
    int64_t grid_z, int64_t block_x, int64_t block_y, int64_t block_z, int64_t waves_per_eu,
    int64_t tile_m, int64_t tile_n, int64_t tile_k, const std::string& quantization,
    int64_t pack_bytes, std::optional<int64_t> quantize_from_bytes) {
  if (state == 0) return refused(Error::not_open);
  hip_custom_communicator::CommsState& s = *state_at(state);
  if (const std::optional<Error> e = attn_res_tensors(input, prefix, blocks, norm_weight, qk_weight,
                   output_norm_weight, prefix_output, output, block_output,
                   num_blocks))
    return refused(e);
  const Rows old = rows_of(prefix), block = rows_of(block_output);
  const std::optional<DType> dtype = dtype_of(input);
  if (!dtype) return refused(Error::dtype_not_built);
  if (!all_ints({waves_per_eu, tile_m, tile_n, tile_k, pack_bytes, grid_x, grid_y, grid_z, block_x, block_y, block_z})) return refused(Error::field_out_of_range);
  const auto got = hip_custom_communicator::all_reduce_add_attn_res_rms_norm(
      s, input.data_ptr(), input.stride(0), input.stride(1),
      input.size(0), input.size(1), old.ptr, old.stride_m, old.stride_n,
      old.size_m, old.size_n, blocks.data_ptr(), blocks.stride(0),
      blocks.stride(1), blocks.stride(2), blocks.size(0), blocks.size(1),
      blocks.size(2), norm_weight.data_ptr(), norm_weight.stride(0),
      norm_weight.size(0), qk_weight.data_ptr(), qk_weight.stride(0),
      qk_weight.size(0),
      output_norm_weight ? output_norm_weight->data_ptr() : nullptr,
      output_norm_weight ? output_norm_weight->stride(0) : 0,
      output_norm_weight ? output_norm_weight->size(0) : 0,
      prefix_output.data_ptr(), prefix_output.stride(0),
      prefix_output.stride(1), prefix_output.size(0), prefix_output.size(1),
      output.data_ptr(), output.stride(0), output.stride(1), output.size(0),
      output.size(1), block.ptr, block.stride_m, block.stride_n, block.size_m,
      block.size_n, *dtype, static_cast<int>(num_blocks),
      static_cast<float>(eps), static_cast<float>(output_eps),
      algorithm, {static_cast<int>(grid_x), static_cast<int>(grid_y), static_cast<int>(grid_z)}, {static_cast<int>(block_x), static_cast<int>(block_y), static_cast<int>(block_z)}, static_cast<int>(waves_per_eu), static_cast<int>(tile_m), static_cast<int>(tile_n), static_cast<int>(tile_k), quantization, static_cast<int>(pack_bytes), quantize_from_bytes,
      current_stream());
  return refused(got);
}

// rms_norm: this rank's rows normed (the second kernel of an op Python composes), at its config.
Refused hip_custom_communicator_rms_norm(int64_t state, torch::Tensor& input, torch::Tensor& weight,
    torch::Tensor& output, double eps, int64_t grid_x, int64_t grid_y, int64_t grid_z,
    int64_t block_x, int64_t block_y, int64_t block_z, int64_t waves_per_eu, int64_t tile_m,
    int64_t tile_n, const std::string& quantization, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  if (state == 0) return refused(Error::not_open);
  hip_custom_communicator::CommsState& s = *state_at(state);
  if (const std::optional<Error> e = norm_tensors(input, nullptr, weight, output, nullptr, /*in_place=*/true)) return refused(e);
  const std::optional<DType> dtype = dtype_of(input);
  if (!dtype) return refused(Error::dtype_not_built);
  const std::optional<DType> weight_dtype = *weight_dtype;
  if (!weight_dtype) return refused(Error::weight_not_built);
  if (!all_ints({waves_per_eu, tile_m, tile_n, pack_bytes, grid_x, grid_y, grid_z, block_x, block_y, block_z})) return refused(Error::field_out_of_range);
  const auto got = hip_custom_communicator::rms_norm(
      s, input.data_ptr(), input.stride(0), input.stride(1),
      input.size(0), input.size(1), weight.data_ptr(), weight.stride(0),
      weight.size(0), output.data_ptr(), output.stride(0), output.stride(1),
      output.size(0), output.size(1), *dtype, *weight_dtype,
      static_cast<float>(eps),
      {static_cast<int>(grid_x), static_cast<int>(grid_y), static_cast<int>(grid_z)}, {static_cast<int>(block_x), static_cast<int>(block_y), static_cast<int>(block_z)}, static_cast<int>(waves_per_eu), static_cast<int>(tile_m), static_cast<int>(tile_n), quantization, static_cast<int>(pack_bytes), quantize_from_bytes,
      current_stream());
  return refused(got);
}

// add_rms_norm: this rank's rows added and normed (the second kernel of an op Python composes),
// at its config.
Refused hip_custom_communicator_add_rms_norm(int64_t state, torch::Tensor& input,
    torch::Tensor& residual, torch::Tensor& weight, torch::Tensor& output,
    torch::Tensor& residual_output, double eps, int64_t grid_x, int64_t grid_y, int64_t grid_z,
    int64_t block_x, int64_t block_y, int64_t block_z, int64_t waves_per_eu, int64_t tile_m,
    int64_t tile_n, const std::string& quantization, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  if (state == 0) return refused(Error::not_open);
  hip_custom_communicator::CommsState& s = *state_at(state);
  if (const std::optional<Error> e = norm_tensors(input, &residual, weight, output, &residual_output, /*in_place=*/true)) return refused(e);
  const std::optional<DType> dtype = dtype_of(input);
  if (!dtype) return refused(Error::dtype_not_built);
  const std::optional<DType> weight_dtype = *weight_dtype;
  if (!weight_dtype) return refused(Error::weight_not_built);
  if (!all_ints({waves_per_eu, tile_m, tile_n, pack_bytes, grid_x, grid_y, grid_z, block_x, block_y, block_z})) return refused(Error::field_out_of_range);
  const auto got = hip_custom_communicator::add_rms_norm(
      s, input.data_ptr(), input.stride(0), input.stride(1),
      input.size(0), input.size(1), residual.data_ptr(), residual.stride(0),
      residual.stride(1), residual.size(0), residual.size(1), weight.data_ptr(),
      weight.stride(0), weight.size(0), output.data_ptr(), output.stride(0),
      output.stride(1), output.size(0), output.size(1),
      residual_output.data_ptr(), residual_output.stride(0),
      residual_output.stride(1), residual_output.size(0),
      residual_output.size(1), *dtype, *weight_dtype,
      static_cast<float>(eps),
      {static_cast<int>(grid_x), static_cast<int>(grid_y), static_cast<int>(grid_z)}, {static_cast<int>(block_x), static_cast<int>(block_y), static_cast<int>(block_z)}, static_cast<int>(waves_per_eu), static_cast<int>(tile_m), static_cast<int>(tile_n), quantization, static_cast<int>(pack_bytes), quantize_from_bytes,
      current_stream());
  return refused(got);
}

// add_attn_res_rms_norm: AttnRes on this rank's rows (the second kernel of the op Python
// composes), at its config.
Refused hip_custom_communicator_add_attn_res_rms_norm(int64_t state, torch::Tensor& input,
    const std::optional<torch::Tensor>& prefix, torch::Tensor& blocks, torch::Tensor& norm_weight,
    torch::Tensor& qk_weight, const std::optional<torch::Tensor>& output_norm_weight,
    torch::Tensor& prefix_output, torch::Tensor& output,
    const std::optional<torch::Tensor>& block_output, int64_t num_blocks, double eps,
    double output_eps, int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t block_x,
    int64_t block_y, int64_t block_z, int64_t waves_per_eu, int64_t tile_m, int64_t tile_n,
    int64_t tile_k, const std::string& quantization, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  if (state == 0) return refused(Error::not_open);
  hip_custom_communicator::CommsState& s = *state_at(state);
  if (const std::optional<Error> e = attn_res_tensors(input, prefix, blocks, norm_weight, qk_weight,
                   output_norm_weight, prefix_output, output, block_output,
                   num_blocks))
    return refused(e);
  const Rows old = rows_of(prefix), block = rows_of(block_output);
  const std::optional<DType> dtype = dtype_of(input);
  if (!dtype) return refused(Error::dtype_not_built);
  if (!all_ints({waves_per_eu, tile_m, tile_n, tile_k, pack_bytes, grid_x, grid_y, grid_z, block_x, block_y, block_z})) return refused(Error::field_out_of_range);
  const auto got = hip_custom_communicator::add_attn_res_rms_norm(
      s, input.data_ptr(), input.stride(0), input.stride(1),
      input.size(0), input.size(1), old.ptr, old.stride_m, old.stride_n,
      old.size_m, old.size_n, blocks.data_ptr(), blocks.stride(0),
      blocks.stride(1), blocks.stride(2), blocks.size(0), blocks.size(1),
      blocks.size(2), norm_weight.data_ptr(), norm_weight.stride(0),
      norm_weight.size(0), qk_weight.data_ptr(), qk_weight.stride(0),
      qk_weight.size(0),
      output_norm_weight ? output_norm_weight->data_ptr() : nullptr,
      output_norm_weight ? output_norm_weight->stride(0) : 0,
      output_norm_weight ? output_norm_weight->size(0) : 0,
      prefix_output.data_ptr(), prefix_output.stride(0),
      prefix_output.stride(1), prefix_output.size(0), prefix_output.size(1),
      output.data_ptr(), output.stride(0), output.stride(1), output.size(0),
      output.size(1), block.ptr, block.stride_m, block.stride_n, block.size_m,
      block.size_n, *dtype, static_cast<int>(num_blocks),
      static_cast<float>(eps), static_cast<float>(output_eps),
      {static_cast<int>(grid_x), static_cast<int>(grid_y), static_cast<int>(grid_z)}, {static_cast<int>(block_x), static_cast<int>(block_y), static_cast<int>(block_z)}, static_cast<int>(waves_per_eu), static_cast<int>(tile_m), static_cast<int>(tile_n), static_cast<int>(tile_k), quantization, static_cast<int>(pack_bytes), quantize_from_bytes,
      current_stream());
  return refused(got);
}
