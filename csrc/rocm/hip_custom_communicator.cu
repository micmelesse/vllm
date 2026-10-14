// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Our HIP collectives as torch ops, built into `_rocm_C`: this file only strips
// torch. Each op checks its tensors and passes their pointers, shapes and
// dtypes, its names as enums and its numbers as ints, to its function in
// hip_custom_communicator/interface.cuh (the API, torch-free), and turns the
// answer back: an op returns the Error's number it was refused for, or none
// when it ran. Self-contained: torch and the HIP runtime, nothing from aiter.

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>

// THE DECLARATIONS torch_bindings.cpp registers, so a definition here that
// differs is a compile error, not a mismatch at run time.
#include "ops.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <variant>
#include <string>
#include <vector>

#include "hip_custom_communicator/hip_custom_communicator.cuh"

// =================================================================================
// THE TORCH OP BOUNDARY. `Handle` (handle.cuh) is a stateful C++ object and a
// torch op is a free function over schema types, so the object crosses as an
// opaque pointer -- the same `fptr_t = int64_t` vLLM's custom all-reduce and
// quick-reduce use. The IPC handles never cross: the Handle gathers them over
// the process group this boundary resolves by name.
// =================================================================================

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

// EACH OP TAKES ITS OWN FORCING LAST, as interface.cuh's does: the algorithm
// and direction of the template it forces, and only the config fields that op's
// kernels have, each none for select's. The model passes none of them; the
// bench and the tests force with them.

namespace {
using ProcessGroupPtr = c10::intrusive_ptr<c10d::ProcessGroup>;

// THE PROCESS GROUP registered as `name`, or none.
std::optional<ProcessGroupPtr> resolved(const std::string& name) {
  try {
    return c10d::resolve_process_group(name);
  } catch (const c10::Error&) {
    return std::nullopt;
  }
}

// EVERY RANK'S `mine`, in rank order, over `pg` (a byte string the same length
// on every rank): the IPC handles go round here, not in Python.
std::vector<std::string> all_gathered(const ProcessGroupPtr& pg,
                                      const std::string& mine) {
  const auto as_bytes = torch::TensorOptions().dtype(torch::kUInt8);
  std::vector<at::Tensor> in{
      torch::empty({static_cast<int64_t>(mine.size())}, as_bytes)};
  std::memcpy(in[0].data_ptr(), mine.data(), mine.size());
  std::vector<std::vector<at::Tensor>> output(1);
  for (int r = 0; r < pg->getSize(); ++r)
    output[0].push_back(torch::empty_like(in[0]));
  pg->allgather(output, in)->wait();
  std::vector<std::string> got;
  for (const auto& t : output[0])
    got.emplace_back(static_cast<const char*>(t.data_ptr()),
                     static_cast<size_t>(t.numel()));
  return got;
}

// THE HANDLE'S COLLECTIVE, over `pg`: what the IPC handles go round in.
hip_custom_communicator::Gather gather_over(const ProcessGroupPtr& pg) {
  return [pg](const std::string& mine) { return all_gathered(pg, mine); };
}
}  // namespace

namespace {
// THE HANDLE CROSSES TORCH AS AN INT (a schema has no object type), as vLLM's
// custom all-reduce passes its own: hip_custom_communicator_init returns it,
// every op takes it first.
hip_custom_communicator::Handle& handle_of(fptr_t handle_ptr) {
  return *reinterpret_cast<hip_custom_communicator::Handle*>(handle_ptr);
}

hip_custom_communicator::DType dtype_of(const torch::Tensor& t) {
  const at::ScalarType s = t.scalar_type();
  if (s == at::ScalarType::Half) return hip_custom_communicator::DType::f16;
  if (s == at::ScalarType::BFloat16)
    return hip_custom_communicator::DType::bf16;
  TORCH_CHECK(s == at::ScalarType::Float, "hip_custom_communicator: dtype ", s,
              " not built");
  return hip_custom_communicator::DType::f32;
}

// THE FORCING ACROSS THE BOUNDARY, as the interface takes it: a name to its
// enum (another name is the caller's mistake, raised), and a field narrowed to
// an int.
std::optional<hip_custom_communicator::Algorithm> algorithm_from(
    const std::optional<std::string>& s) {
  if (!s) return std::nullopt;
  if (*s == "one_shot") return hip_custom_communicator::Algorithm::one_shot;
  TORCH_CHECK(*s == "two_shot",
              "hip_custom_communicator: algorithm is one_shot or two_shot");
  return hip_custom_communicator::Algorithm::two_shot;
}
std::optional<hip_custom_communicator::Direction> direction_from(
    const std::optional<std::string>& s) {
  if (!s) return std::nullopt;
  if (*s == "pull") return hip_custom_communicator::Direction::pull;
  TORCH_CHECK(*s == "push",
              "hip_custom_communicator: direction is pull or push");
  return hip_custom_communicator::Direction::push;
}
std::optional<int> narrowed(std::optional<int64_t> v) {
  if (!v) return std::nullopt;
  TORCH_CHECK(*v >= std::numeric_limits<int>::min() &&
                  *v <= std::numeric_limits<int>::max(),
              "hip_custom_communicator: a config field must fit an int");
  return static_cast<int>(*v);
}

hipStream_t current_stream() { return at::cuda::getCurrentCUDAStream(); }

// AN ERROR RAISED, at the torch boundary: a torch op can only return or raise.
[[noreturn]] void raise(hip_custom_communicator::Error e) {
  TORCH_CHECK(false, "hip_custom_communicator: ",
              hip_custom_communicator::to_string(e));
  __builtin_unreachable();
}

void check_device_contiguous(std::initializer_list<const torch::Tensor*> ts) {
  for (const torch::Tensor* t : ts) {
    TORCH_CHECK(t->is_cuda(), "every tensor must be on device");
    TORCH_CHECK(t->is_contiguous(), "every tensor must be contiguous");
  }
}
// The ops' tensors: on device; their layouts are their strides, checked behind
// the interface.
void check_device(std::initializer_list<const torch::Tensor*> ts) {
  for (const torch::Tensor* t : ts)
    TORCH_CHECK(t->is_cuda(), "every tensor must be on device");
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
using SupportedWire =
    std::tuple<std::optional<std::string>, std::optional<int64_t>>;
using OpenWire = std::tuple<std::optional<int64_t>, std::optional<int64_t>>;
using BuildInfoWire =
    std::tuple<Names, std::vector<int64_t>, int64_t, int64_t, Names, Names,
               Names, Names, Names, Names, std::vector<int64_t>,
               std::vector<int64_t>, Names, Names, std::vector<int64_t>>;
// AN OP'S ANSWER: the Error's number it was refused for, or none when it ran.
using Refusal = std::optional<int64_t>;

// THE COMMUNICATOR OPENED on `device` over the process groups named `cpu_group`
// (which carries the handles) and `device_group`, a collective: this rank's
// symmetric memory, made and owned here, its handle gathered with every rank's,
// and the peers' opened. `timeout_seconds`: how long a kernel waits on a peer
// before it reports a fault. A torch op cannot return a variant, so it is two
// optionals and exactly one is set, the handle or the Error's number.
OpenWire hip_custom_communicator_open(const std::string& cpu_group,
                                      const std::string& device_group,
                                      int64_t device, double timeout_seconds) {
  const auto refused = [](hip_custom_communicator::Error e) {
    return OpenWire{std::nullopt, static_cast<int64_t>(e)};
  };
  const auto pg = resolved(cpu_group), dg = resolved(device_group);
  if (!pg || !dg) return refused(hip_custom_communicator::Error::no_such_group);
  if ((*pg)->getSize() != (*dg)->getSize() ||
      (*pg)->getRank() != (*dg)->getRank())
    return refused(hip_custom_communicator::Error::groups_disagree);
  const int world = (*pg)->getSize();
  const auto ok =
      hip_custom_communicator::supported(static_cast<int>(device), world);
  if (const auto* e = std::get_if<hip_custom_communicator::Error>(&ok))
    return refused(*e);
  auto* handle = new hip_custom_communicator::Handle(
      (*pg)->getRank(), world, gather_over(*pg), timeout_seconds);
  return {reinterpret_cast<fptr_t>(handle), std::nullopt};
}

// SUPPORTED AT THE TORCH BOUNDARY: a torch op cannot return a variant, so it is
// two optionals and exactly one is set, the arch or the Error's number.
SupportedWire hip_custom_communicator_supported(int64_t device, int64_t world) {
  const auto got = hip_custom_communicator::supported(static_cast<int>(device),
                                                      static_cast<int>(world));
  if (const auto* e = std::get_if<hip_custom_communicator::Error>(&got))
    return {std::nullopt, static_cast<int64_t>(*e)};
  return {std::get<hip_custom_communicator::Supported>(got).arch, std::nullopt};
}

// WHAT THE BUILD HOLDS, kBuild's projection for Python: its dtypes by name, its
// worlds, a pack's bytes and a staging's, its ops, errors and fault places by
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
  // Each template's op, and its builds (what dispatch instantiates: the tuner's
  // search space), flat: a build's fields in `fields`' order, 0 where its op's
  // BuildConfig has none (every field it has is set). The launch is not a
  // build's: a forced call gives its own grid.
  const Names fields = {"tile_m", "tile_n", "tile_k", "threads_per_block",
                        "waves_per_eu"};
  Names templates, template_ops;
  std::vector<int64_t> configs, counts;
  for (const TemplateInfo& t : kTemplates) {
    templates.push_back(t.name);
    template_ops.push_back(to_string(op_of(t.fn)));
    counts.push_back(static_cast<int64_t>(t.builds.size()));
    for (const BuildConfig& b : t.builds)
      std::visit(
          [&](const auto& f) {
            int64_t k = 0;
            if constexpr (requires { f.tile_k; }) k = f.tile_k;
            configs.insert(
                configs.end(),
                {f.tile_m, f.tile_n, k, f.threads_per_block, f.waves_per_eu});
          },
          b);
  }
  // EACH OP'S PLAIN CALL (kOps, select.cuh), in `ops`' order: its dtypes and
  // its weight's besides its input's, flat, and five ints an op: those two
  // counts, widest_row, max_bytes and two_d.
  Names op_dtypes, op_weights;
  std::vector<int64_t> op_facts;
  for (const Op& o : kOps) {
    for (const DType d : o.dtypes) op_dtypes.push_back(to_string(d));
    for (const DType d : o.weights) op_weights.push_back(to_string(d));
    op_facts.insert(op_facts.end(),
                    {static_cast<int64_t>(o.dtypes.size()),
                     static_cast<int64_t>(o.weights.size()), widest_row(o.type),
                     max_bytes(o.type), o.two_d});
  }
  return {dtypes,
          worlds,
          kBuild.memory.pack_bytes,
          kBuild.memory.staging_bytes,
          ops,
          errors,
          where_names,
          templates,
          template_ops,
          fields,
          configs,
          counts,
          op_dtypes,
          op_weights,
          op_facts};
}

void hip_custom_communicator_dispose(fptr_t handle_ptr) {
  delete &handle_of(handle_ptr);
}

// THE STATUS AS PYTHON READS IT, every field in StatusField's order, `elapsed`
// its two words joined: polled by the communicator's watchdog, so it never
// waits on the device.
std::vector<int64_t> hip_custom_communicator_status(fptr_t handle_ptr) {
  using hip_custom_communicator::StatusField;
  const auto words = handle_of(handle_ptr).status();
  const auto word = [&](StatusField s) { return words[static_cast<int>(s)]; };
  const auto as_int = [](uint32_t w) {
    return static_cast<int64_t>(static_cast<int32_t>(w));
  };
  return {word(StatusField::abort),
          word(StatusField::state),
          word(StatusField::where),
          as_int(word(StatusField::rank)),
          as_int(word(StatusField::block)),
          as_int(word(StatusField::peer)),
          word(StatusField::count),
          word(StatusField::want),
          static_cast<int64_t>(uint64_t{word(StatusField::elapsed_high)} << 32 |
                               word(StatusField::elapsed_low))};
}
// THIS RANK'S SIGNALS, every flag, flattened four a flag: parity, block,
// writer, value. Every value zero between launches; read once they have
// finished.
std::vector<int64_t> hip_custom_communicator_signals(fptr_t handle_ptr) {
  std::vector<int64_t> flat;
  for (const auto& f : handle_of(handle_ptr).signals())
    flat.insert(flat.end(), f.begin(), f.end());
  return flat;
}
void hip_custom_communicator_abort(fptr_t handle_ptr) {
  handle_of(handle_ptr).abort();
}

// THE PER-PHASE STAMPS SINCE THE LAST READ, [block][phase] device clock ticks
// (100 MHz), zero where no block stamped, then zeroed. Only a
// HIP_CUSTOM_COMMUNICATOR_STAMPS build writes them (block_stamp).
torch::Tensor hip_custom_communicator_stamps() {
  auto output = torch::zeros({hip_custom_communicator::kStampBlocks,
                              hip_custom_communicator::kStampPhases},
                             torch::TensorOptions().dtype(torch::kInt64));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpyFromSymbol(output.data_ptr<int64_t>(),
                                HIP_SYMBOL(hip_custom_communicator::g_stamps),
                                sizeof(hip_custom_communicator::g_stamps)));
  // Zeroed after reading, so the next launch's table holds only its own blocks.
  void* table = nullptr;
  HIP_CHECK(hipGetSymbolAddress(&table,
                                HIP_SYMBOL(hip_custom_communicator::g_stamps)));
  HIP_CHECK(hipMemset(table, 0, sizeof(hip_custom_communicator::g_stamps)));
  return output;
}

// `buffer` registered with every rank's over the process group named `group`, a
// collective: the peers read it where it is until it is forgotten, which must
// come before it is freed.
void hip_custom_communicator_register_buffer(fptr_t handle_ptr,
                                             torch::Tensor& buffer,
                                             const std::string& group) {
  check_device_contiguous({&buffer});
  const auto pg = resolved(group);
  if (!pg) raise(hip_custom_communicator::Error::no_such_group);
  handle_of(handle_ptr).register_buffer(buffer.data_ptr(), gather_over(*pg));
}
void hip_custom_communicator_forget_buffer(fptr_t handle_ptr,
                                           torch::Tensor& buffer) {
  handle_of(handle_ptr).forget_buffer(buffer.data_ptr());
}

// The device clock's rate, kHz: what its ticks are per ms.
int64_t hip_custom_communicator_wall_clock_khz(int64_t device) {
  int khz = 0;
  HIP_CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate,
                                  static_cast<int>(device)));
  return khz;
}

// THE BUFFERS A CAPTURE RECORDED, registered over the process group named
// `group`, every rank together (Handle::register_captured).
void hip_custom_communicator_register_captured(fptr_t handle_ptr,
                                               const std::string& group) {
  const auto pg = resolved(group);
  if (!pg) raise(hip_custom_communicator::Error::no_such_group);
  if (!handle_of(handle_ptr).register_captured(gather_over(*pg)))
    raise(hip_custom_communicator::Error::ranks_disagree);
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
void refuse_overlap(std::initializer_list<const torch::Tensor*> inputs,
                    std::initializer_list<const torch::Tensor*> outputs) {
  for (const torch::Tensor* i : inputs)
    for (const torch::Tensor* o : outputs) {
      if (i == nullptr || o == nullptr || i->numel() == 0 || o->numel() == 0)
        continue;
      TORCH_CHECK(
          !overlap(span_of(*i), span_of(*o)),
          "an input and an output share memory; pass a separate output");
    }
}
}  // namespace

Refusal hip_custom_communicator_all_reduce(
    fptr_t handle_ptr, torch::Tensor& input, torch::Tensor& output,
    std::optional<std::string> algorithm, std::optional<std::string> direction,
    std::optional<int64_t> tile_m, std::optional<int64_t> tile_n,
    std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid,
    std::optional<int64_t> waves_per_eu) {
  check_device({&input, &output});
  TORCH_CHECK(output.sizes() == input.sizes(),
              "output and input must have the same shape");
  TORCH_CHECK(output.scalar_type() == input.scalar_type(),
              "output and input must share a dtype");
  refuse_overlap({&input}, {&output});
  const std::optional<torch::Tensor> input_rows = as_rows(input),
                                     output_rows = as_rows(output);
  if (!input_rows || !output_rows)
    return static_cast<int64_t>(hip_custom_communicator::Error::not_contiguous);
  const auto got = hip_custom_communicator::all_reduce(
      handle_of(handle_ptr), input_rows->data_ptr(), input_rows->stride(0),
      input_rows->stride(1), input_rows->size(0), input_rows->size(1),
      output_rows->data_ptr(), output_rows->stride(0), output_rows->stride(1),
      output_rows->size(0), output_rows->size(1), dtype_of(input),
      algorithm_from(algorithm), direction_from(direction), narrowed(tile_m),
      narrowed(tile_n), narrowed(threads_per_block), narrowed(blocks_per_grid),
      narrowed(waves_per_eu), current_stream());
  if (const auto* e = std::get_if<hip_custom_communicator::Error>(&got))
    return static_cast<int64_t>(*e);
  return std::nullopt;
}

namespace {
// A NORM CALL'S TENSORS: input [tokens, hidden], output (and with a residual,
// residual and residual_output) its shape and dtype, the weight 1-D of hidden
// (its dtype is select's to refuse).
void norm_tensors(const torch::Tensor& input, const torch::Tensor* residual,
                  const torch::Tensor& weight, const torch::Tensor& output,
                  const torch::Tensor* residual_output) {
  check_device({&input, &weight, &output});
  TORCH_CHECK(input.dim() == 2, "input must be 2-D [tokens, hidden]; got ",
              input.dim(), "-D");
  TORCH_CHECK(output.sizes() == input.sizes(),
              "output must have input's shape");
  TORCH_CHECK(output.scalar_type() == input.scalar_type(),
              "output must share input's dtype");
  TORCH_CHECK(weight.dim() == 1 && weight.numel() == input.size(1),
              "weight must be 1-D of hidden=", input.size(1));
  // The weight is read a pack at a time, a pack's worth of input's elements per
  // load.
  const int64_t weight_pack =
      hip_custom_communicator::kBuild.memory.pack_bytes / input.element_size() *
      weight.element_size();
  TORCH_CHECK(reinterpret_cast<uintptr_t>(weight.data_ptr()) % weight_pack == 0,
              "weight must be aligned to ", weight_pack, " bytes");
  if (residual != nullptr) {
    check_device({residual, residual_output});
    TORCH_CHECK(residual->sizes() == input.sizes() &&
                    residual_output->sizes() == input.sizes(),
                "residual and residual_output must have input's shape");
    TORCH_CHECK(residual->scalar_type() == input.scalar_type() &&
                    residual_output->scalar_type() == input.scalar_type(),
                "every tensor must share input's dtype");
  }
  refuse_overlap({&input, residual, &weight}, {&output, residual_output});
}
}  // namespace

Refusal hip_custom_communicator_all_reduce_rms_norm(
    fptr_t handle_ptr, torch::Tensor& input, torch::Tensor& weight,
    torch::Tensor& output, double eps, std::optional<std::string> algorithm,
    std::optional<std::string> direction, std::optional<int64_t> tile_m,
    std::optional<int64_t> tile_n, std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid,
    std::optional<int64_t> waves_per_eu) {
  norm_tensors(input, nullptr, weight, output, nullptr);
  const auto got = hip_custom_communicator::all_reduce_rms_norm(
      handle_of(handle_ptr), input.data_ptr(), input.stride(0), input.stride(1),
      input.size(0), input.size(1), weight.data_ptr(), weight.stride(0),
      weight.size(0), output.data_ptr(), output.stride(0), output.stride(1),
      output.size(0), output.size(1), dtype_of(input), dtype_of(weight),
      static_cast<float>(eps), algorithm_from(algorithm),
      direction_from(direction), narrowed(tile_m), narrowed(tile_n),
      narrowed(threads_per_block), narrowed(blocks_per_grid),
      narrowed(waves_per_eu), current_stream());
  if (const auto* e = std::get_if<hip_custom_communicator::Error>(&got))
    return static_cast<int64_t>(*e);
  return std::nullopt;
}

Refusal hip_custom_communicator_all_reduce_add_rms_norm(
    fptr_t handle_ptr, torch::Tensor& input, torch::Tensor& residual,
    torch::Tensor& weight, torch::Tensor& output,
    torch::Tensor& residual_output, double eps,
    std::optional<std::string> algorithm, std::optional<std::string> direction,
    std::optional<int64_t> tile_m, std::optional<int64_t> tile_n,
    std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid,
    std::optional<int64_t> waves_per_eu) {
  norm_tensors(input, &residual, weight, output, &residual_output);
  const auto got = hip_custom_communicator::all_reduce_add_rms_norm(
      handle_of(handle_ptr), input.data_ptr(), input.stride(0), input.stride(1),
      input.size(0), input.size(1), residual.data_ptr(), residual.stride(0),
      residual.stride(1), residual.size(0), residual.size(1), weight.data_ptr(),
      weight.stride(0), weight.size(0), output.data_ptr(), output.stride(0),
      output.stride(1), output.size(0), output.size(1),
      residual_output.data_ptr(), residual_output.stride(0),
      residual_output.stride(1), residual_output.size(0),
      residual_output.size(1), dtype_of(input), dtype_of(weight),
      static_cast<float>(eps), algorithm_from(algorithm),
      direction_from(direction), narrowed(tile_m), narrowed(tile_n),
      narrowed(threads_per_block), narrowed(blocks_per_grid),
      narrowed(waves_per_eu), current_stream());
  if (const auto* e = std::get_if<hip_custom_communicator::Error>(&got))
    return static_cast<int64_t>(*e);
  return std::nullopt;
}

namespace {
// AN ATTNRES CALL'S TENSORS, as every AttnRes op takes them: `input` (the
// partial sum, or the delta) [tokens, hidden]; `prefix` (none: the sum starts
// it), `prefix_output` and `output` its shape; `blocks` [tokens, sources,
// hidden], its first `num_blocks` the sources read; `block_output` (none: no
// block written) a [tokens, hidden] slot, by the standard not one of the
// sources; every one input's dtype.
void attn_res_tensors(
    const torch::Tensor& input, const std::optional<torch::Tensor>& prefix,
    const torch::Tensor& blocks, const torch::Tensor& norm_weight,
    const torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& output_norm_weight,
    const torch::Tensor& prefix_output, const torch::Tensor& output,
    const std::optional<torch::Tensor>& block_output, int64_t num_blocks) {
  check_device({&input, &norm_weight, &qk_weight, &prefix_output, &output});
  TORCH_CHECK(input.dim() == 2, "input must be 2-D [tokens, hidden]; got ",
              input.dim(), "-D");
  const int64_t n = input.size(1);
  TORCH_CHECK(
      prefix_output.sizes() == input.sizes() && output.sizes() == input.sizes(),
      "prefix_output and output must have input's shape");
  TORCH_CHECK(
      blocks.is_cuda() && blocks.dim() == 3 &&
          blocks.size(0) == input.size(0) && blocks.size(2) == n &&
          blocks.stride(2) == 1,
      "blocks must be [tokens, sources, hidden] with a unit hidden stride");
  TORCH_CHECK(num_blocks >= 0 && num_blocks <= blocks.size(1),
              "num_blocks must be in [0, ", blocks.size(1), "]");
  std::vector<const torch::Tensor*> same = {&blocks, &norm_weight, &qk_weight,
                                            &prefix_output, &output};
  for (const std::optional<torch::Tensor>* o :
       {&prefix, &output_norm_weight, &block_output})
    if (*o) {
      check_device({&**o});
      same.push_back(&**o);
    }
  for (const torch::Tensor* t : same)
    TORCH_CHECK(t->scalar_type() == input.scalar_type(),
                "every tensor must share input's dtype");
  if (prefix)
    TORCH_CHECK(prefix->sizes() == input.sizes(),
                "prefix must have input's shape");
  if (block_output)
    TORCH_CHECK(
        block_output->sizes() == input.sizes() && block_output->stride(1) == 1,
        "block_output must be [tokens, hidden] with a unit hidden stride");
  for (const torch::Tensor* t : {&norm_weight, &qk_weight})
    TORCH_CHECK(t->dim() == 1 && t->numel() == n,
                "weights must be 1-D of hidden=", n);
  if (output_norm_weight)
    TORCH_CHECK(
        output_norm_weight->dim() == 1 && output_norm_weight->numel() == n,
        "output_norm_weight must be 1-D of hidden=", n);
  // Its rows' and sources' strides are the interface's to refuse (whole packs);
  // its start is here.
  TORCH_CHECK(reinterpret_cast<uintptr_t>(blocks.data_ptr()) %
                      hip_custom_communicator::kBuild.memory.pack_bytes ==
                  0,
              "blocks must start 16-byte aligned");
  // The sources read are the first num_blocks; block_output may be a later slot
  // of the same stack.
  const torch::Tensor sources = blocks.narrow(1, 0, num_blocks);
  refuse_overlap(
      {&input, prefix ? &*prefix : nullptr, &sources, &norm_weight, &qk_weight,
       output_norm_weight ? &*output_norm_weight : nullptr},
      {&prefix_output, &output, block_output ? &*block_output : nullptr});
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

Refusal hip_custom_communicator_all_reduce_add_attn_res_rms_norm(
    fptr_t handle_ptr, torch::Tensor& input,
    const std::optional<torch::Tensor>& prefix, torch::Tensor& blocks,
    torch::Tensor& norm_weight, torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& output_norm_weight,
    torch::Tensor& prefix_output, torch::Tensor& output,
    const std::optional<torch::Tensor>& block_output, int64_t num_blocks,
    double eps, double output_eps, std::optional<std::string> algorithm,
    std::optional<std::string> direction, std::optional<int64_t> tile_m,
    std::optional<int64_t> tile_n, std::optional<int64_t> tile_k,
    std::optional<int64_t> reduce_scatter_blocks,
    std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid,
    std::optional<int64_t> waves_per_eu) {
  attn_res_tensors(input, prefix, blocks, norm_weight, qk_weight,
                   output_norm_weight, prefix_output, output, block_output,
                   num_blocks);
  const Rows old = rows_of(prefix), block = rows_of(block_output);
  const auto got = hip_custom_communicator::all_reduce_add_attn_res_rms_norm(
      handle_of(handle_ptr), input.data_ptr(), input.stride(0), input.stride(1),
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
      block.size_n, dtype_of(input), static_cast<int>(num_blocks),
      static_cast<float>(eps), static_cast<float>(output_eps),
      algorithm_from(algorithm), direction_from(direction), narrowed(tile_m),
      narrowed(tile_n), narrowed(tile_k), narrowed(reduce_scatter_blocks),
      narrowed(threads_per_block), narrowed(blocks_per_grid),
      narrowed(waves_per_eu), current_stream());
  if (const auto* e = std::get_if<hip_custom_communicator::Error>(&got))
    return static_cast<int64_t>(*e);
  return std::nullopt;
}
