// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Our HIP collectives as torch ops, built into `_rocm_C`: this file only strips torch. Each op
// checks its tensors and passes their pointers, shapes and dtypes, its names as enums and its
// numbers as ints, to its function in rocm_comms/interface.cuh (the API, torch-free), and turns
// the answer back: an Error raised, or a plan's kernel as its schema's tuple.
// Self-contained: torch and the HIP runtime, nothing from aiter.

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>

// THE DECLARATIONS torch_bindings.cpp registers, so a definition here that differs is a compile
// error, not a mismatch at run time.
#include "ops.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <variant>
#include <string>
#include <vector>

#include "rocm_comms/rocm_comms.cuh"

// =================================================================================
// THE TORCH OP BOUNDARY. `Handle` (handle.cuh) is a stateful C++ object and a torch op is a
// free function over schema types, so the object crosses as an opaque pointer -- the same
// `fptr_t = int64_t` vLLM's custom all-reduce and quick-reduce use. The IPC handles never cross:
// the Handle gathers them over the process group this boundary resolves by name.
// =================================================================================

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

// EACH OP TAKES ITS OWN FORCING LAST, as interface.cuh's does: the algorithm and direction of the
// template it forces, and only the config fields that op's kernels have, each none for select's.
// The model passes none of them; the bench and the tests force with them.

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

// EVERY RANK'S `mine`, in rank order, over `pg` (a byte string the same length on every rank): the
// IPC handles go round here, not in Python.
std::vector<std::string> all_gathered(const ProcessGroupPtr& pg, const std::string& mine) {
  const auto as_bytes = torch::TensorOptions().dtype(torch::kUInt8);
  std::vector<at::Tensor> in{torch::empty({static_cast<int64_t>(mine.size())}, as_bytes)};
  std::memcpy(in[0].data_ptr(), mine.data(), mine.size());
  std::vector<std::vector<at::Tensor>> out(1);
  for (int r = 0; r < pg->getSize(); ++r) out[0].push_back(torch::empty_like(in[0]));
  pg->allgather(out, in)->wait();
  std::vector<std::string> got;
  for (const auto& t : out[0])
    got.emplace_back(static_cast<const char*>(t.data_ptr()), static_cast<size_t>(t.numel()));
  return got;
}

// THE HANDLE'S COLLECTIVE, over `pg`: what the IPC handles go round in.
hip_comms::Gather gather_over(const ProcessGroupPtr& pg) {
  return [pg](const std::string& mine) { return all_gathered(pg, mine); };
}
}  // namespace

namespace {
// THE HANDLE CROSSES TORCH AS AN INT (a schema has no object type), as vLLM's custom all-reduce
// passes its own: rocm_comms_init returns it, every op takes it first.
hip_comms::Handle& handle_of(fptr_t handle_ptr) {
  return *reinterpret_cast<hip_comms::Handle*>(handle_ptr);
}

hip_comms::DType dtype_of(const torch::Tensor& t) {
  const at::ScalarType s = t.scalar_type();
  if (s == at::ScalarType::Half) return hip_comms::DType::f16;
  if (s == at::ScalarType::BFloat16) return hip_comms::DType::bf16;
  TORCH_CHECK(s == at::ScalarType::Float, "hip_comms: dtype ", s, " not built");
  return hip_comms::DType::f32;
}

// THE FORCING ACROSS THE BOUNDARY, as the interface takes it: a name to its enum (another name is
// the caller's mistake, raised), and a field narrowed to an int.
std::optional<hip_comms::Algorithm> algorithm_from(const std::optional<std::string>& s) {
  if (!s) return std::nullopt;
  if (*s == "one_shot") return hip_comms::Algorithm::one_shot;
  TORCH_CHECK(*s == "two_shot", "hip_comms: algorithm is one_shot or two_shot");
  return hip_comms::Algorithm::two_shot;
}
std::optional<hip_comms::Direction> direction_from(const std::optional<std::string>& s) {
  if (!s) return std::nullopt;
  if (*s == "pull") return hip_comms::Direction::pull;
  TORCH_CHECK(*s == "push", "hip_comms: direction is pull or push");
  return hip_comms::Direction::push;
}
std::optional<int> narrowed(std::optional<int64_t> v) {
  if (!v) return std::nullopt;
  TORCH_CHECK(*v >= std::numeric_limits<int>::min() && *v <= std::numeric_limits<int>::max(),
              "hip_comms: a config field must fit an int");
  return static_cast<int>(*v);
}

hipStream_t current_stream() { return at::cuda::getCurrentCUDAStream(); }

// AN ERROR RAISED, at the torch boundary: a torch op can only return or raise.
[[noreturn]] void raise(hip_comms::Error e) {
  TORCH_CHECK(false, "hip_comms: ", hip_comms::to_string(e));
  __builtin_unreachable();
}

void check_device_contiguous(std::initializer_list<const torch::Tensor*> ts) {
  for (const torch::Tensor* t : ts) {
    TORCH_CHECK(t->is_cuda(), "every tensor must be on device");
    TORCH_CHECK(t->is_contiguous(), "every tensor must be contiguous");
  }
}
}  // namespace

// AN OP'S ERROR RAISED.
template <typename Launch>
void ran(const std::variant<Launch, hip_comms::Error>& result) {
  if (const hip_comms::Error* e = std::get_if<hip_comms::Error>(&result)) raise(*e);
}

// WHAT CROSSES THE TORCH BOUNDARY, as the tuples an op schema can return.
using Names         = std::vector<std::string>;
using SupportedWire = std::tuple<std::optional<std::string>, std::optional<int64_t>>;
using OpenWire      = std::tuple<std::optional<int64_t>, std::optional<int64_t>>;
using ProbeWire     = std::tuple<std::vector<double>, Names, std::vector<double>>;
using BuildInfoWire = std::tuple<Names, std::vector<int64_t>, int64_t, int64_t, Names, Names, Names,
                                 Names, Names, std::vector<int64_t>, std::vector<int64_t>>;

// A torch dtype as ours, or none for one ours has no name for.
std::optional<hip_comms::DType> dtype_from(at::ScalarType s) {
  if (s == at::ScalarType::Half) return hip_comms::DType::f16;
  if (s == at::ScalarType::BFloat16) return hip_comms::DType::bf16;
  if (s == at::ScalarType::Float) return hip_comms::DType::f32;
  return std::nullopt;
}

// WHAT RUNS A CALL, or the first Error it meets: `hip_comms::plan`, one planner
// per op family, each handed the call's own tensors and reading their facts
// here, with the op's own forcing. A torch op cannot return a variant, so each
// returns the kernel's template (by name) and its config as that op's own
// fields, or the Error's number: those or the one. The arguments a call's
// kernel reads are not, so they are null.
namespace {
// AN INPUT AS OURS: its dtype, or the Error it meets first (not contiguous, not
// 2-D where the op reads rows, a dtype ours has no name for).
std::variant<hip_comms::DType, hip_comms::Error> admitted(
    const torch::Tensor& inp, bool rows) {
  if (!inp.is_contiguous()) return hip_comms::Error::not_contiguous;
  if (rows && inp.dim() != 2) return hip_comms::Error::not_two_d;
  const std::optional<hip_comms::DType> d = dtype_from(inp.scalar_type());
  if (!d) return hip_comms::Error::dtype_not_built;
  return *d;
}

std::optional<std::string> name_of(hip_comms::Algorithm a) {
  return std::string(a == hip_comms::Algorithm::two_shot ? "two_shot" : "one_shot");
}
std::optional<std::string> name_of(hip_comms::Direction d) {
  return std::string(d == hip_comms::Direction::push ? "push" : "pull");
}
std::optional<int64_t> number_of(hip_comms::Error e) { return static_cast<int64_t>(e); }

// EACH OP'S ANSWER, as its schema returns it: the launch's fields, or the Error's number.
RocmCommsAllReducePlan all_reduce_plan_of(
    const std::variant<hip_comms::AllReduceLaunch, hip_comms::Error>& p) {
  if (const auto* e = std::get_if<hip_comms::Error>(&p))
    return {std::nullopt, std::nullopt, std::nullopt, std::nullopt, number_of(*e)};
  const auto& l = std::get<hip_comms::AllReduceLaunch>(p);
  return {name_of(l.algorithm), name_of(l.direction), l.threads_per_block, l.blocks_per_grid,
          std::nullopt};
}
template <typename Launch>
RocmCommsRowPlan row_plan_of(const std::variant<Launch, hip_comms::Error>& p) {
  if (const auto* e = std::get_if<hip_comms::Error>(&p))
    return {std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, number_of(*e)};
  const Launch& l = std::get<Launch>(p);
  return {name_of(l.algorithm), name_of(l.direction), l.tile_n, l.threads_per_block,
          l.blocks_per_grid, std::nullopt};
}
// The one-shot and push report no TILE_M and no reduce-scatter blocks (their families have none).
RocmCommsAttnResPlan attn_res_plan_of(
    const std::variant<hip_comms::AllReduceAddAttnResRmsNormLaunch, hip_comms::Error>& p) {
  if (const auto* e = std::get_if<hip_comms::Error>(&p))
    return {std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
            std::nullopt, std::nullopt, std::nullopt, number_of(*e)};
  const auto& l   = std::get<hip_comms::AllReduceAddAttnResRmsNormLaunch>(p);
  const bool pull = l.algorithm == hip_comms::Algorithm::two_shot &&
                    l.direction == hip_comms::Direction::pull;
  return {name_of(l.algorithm),
          name_of(l.direction),
          pull ? std::optional<int64_t>{l.tile_m} : std::nullopt,
          l.tile_n,
          l.tile_k,
          pull ? std::optional<int64_t>{l.reduce_scatter_blocks} : std::nullopt,
          l.threads_per_block,
          l.blocks_per_grid,
          std::nullopt};
}
template <typename Launch>
RocmCommsGemmPlan gemm_plan_of(const std::variant<Launch, hip_comms::Error>& p) {
  if (const auto* e = std::get_if<hip_comms::Error>(&p))
    return {std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
            std::nullopt, std::nullopt, std::nullopt, number_of(*e)};
  const Launch& l = std::get<Launch>(p);
  return {name_of(l.algorithm), name_of(l.direction), l.tile_m, l.tile_n, l.tile_k, l.slice_k,
          l.threads_per_block, l.blocks_per_grid, std::nullopt};
}
}  // namespace

RocmCommsAllReducePlan rocm_comms_plan_all_reduce(
    fptr_t handle_ptr, const torch::Tensor& inp, std::optional<std::string> algorithm,
    std::optional<std::string> direction, std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid) {
  const auto alg = algorithm_from(algorithm);
  const auto dir = direction_from(direction);
  const auto d = admitted(inp, false);
  if (const auto* e = std::get_if<hip_comms::Error>(&d)) return all_reduce_plan_of(*e);
  return all_reduce_plan_of(hip_comms::select_all_reduce(
      handle_of(handle_ptr), nullptr, nullptr, inp.numel() * inp.element_size(),
      std::get<hip_comms::DType>(d), alg, dir, narrowed(threads_per_block),
      narrowed(blocks_per_grid), current_stream()));
}

// `add`: fused_add_rms_norm's; otherwise rms_norm's.
RocmCommsRowPlan rocm_comms_plan_all_reduce_rms_norm(
    fptr_t handle_ptr, const torch::Tensor& inp, const torch::Tensor& weight, bool add,
    std::optional<std::string> algorithm, std::optional<std::string> direction,
    std::optional<int64_t> tile_n, std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid) {
  const auto alg = algorithm_from(algorithm);
  const auto dir = direction_from(direction);
  const auto d = admitted(inp, true);
  if (const auto* e = std::get_if<hip_comms::Error>(&d))
    return row_plan_of<hip_comms::AllReduceRmsNormLaunch>(*e);
  const std::optional<hip_comms::DType> w = dtype_from(weight.scalar_type());
  if (!w)
    return row_plan_of<hip_comms::AllReduceRmsNormLaunch>(hip_comms::Error::weight_not_built);
  auto& h = handle_of(handle_ptr);
  const hip_comms::DType dt = std::get<hip_comms::DType>(d);
  const auto tn = narrowed(tile_n), tpb = narrowed(threads_per_block);
  const auto bpg = narrowed(blocks_per_grid);
  if (add)
    return row_plan_of(hip_comms::select_all_reduce_add_rms_norm(
        h, nullptr, nullptr, nullptr, nullptr, nullptr, dt, *w, inp.size(0), inp.size(1), 0.f,
        alg, dir, tn, tpb, bpg, current_stream()));
  return row_plan_of(hip_comms::select_all_reduce_rms_norm(
      h, nullptr, nullptr, nullptr, dt, *w, inp.size(0), inp.size(1), 0.f, alg, dir, tn, tpb,
      bpg, current_stream()));
}

RocmCommsAttnResPlan rocm_comms_plan_all_reduce_add_attn_res_rms_norm(
    fptr_t handle_ptr, const torch::Tensor& inp, std::optional<std::string> algorithm,
    std::optional<std::string> direction, std::optional<int64_t> tile_m,
    std::optional<int64_t> tile_n, std::optional<int64_t> tile_k,
    std::optional<int64_t> reduce_scatter_blocks, std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid) {
  const auto alg = algorithm_from(algorithm);
  const auto dir = direction_from(direction);
  const auto d = admitted(inp, true);
  if (const auto* e = std::get_if<hip_comms::Error>(&d)) return attn_res_plan_of(*e);
  return attn_res_plan_of(hip_comms::select_all_reduce_add_attn_res_rms_norm(
      handle_of(handle_ptr), nullptr, nullptr, nullptr, nullptr, 0, 0, nullptr, nullptr, nullptr,
      std::get<hip_comms::DType>(d), inp.size(0), inp.size(1), 0, -1, 0.f, 0.f, false, alg, dir,
      narrowed(tile_m), narrowed(tile_n), narrowed(tile_k), narrowed(reduce_scatter_blocks),
      narrowed(threads_per_block), narrowed(blocks_per_grid), current_stream()));
}

// `add`: the GEMM added into the output's; otherwise written. `gemm_weight` is [N, hidden].
RocmCommsGemmPlan rocm_comms_plan_all_reduce_rms_norm_gemm(
    fptr_t handle_ptr, const torch::Tensor& inp, const torch::Tensor& gemm_weight, bool add,
    std::optional<std::string> algorithm, std::optional<std::string> direction,
    std::optional<int64_t> tile_m, std::optional<int64_t> tile_n, std::optional<int64_t> tile_k,
    std::optional<int64_t> slice_k, std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid) {
  const auto alg = algorithm_from(algorithm);
  const auto dir = direction_from(direction);
  const auto d = admitted(inp, true);
  if (const auto* e = std::get_if<hip_comms::Error>(&d))
    return gemm_plan_of<hip_comms::AllReduceRmsNormGemmLaunch>(*e);
  if (gemm_weight.dim() != 2)
    return gemm_plan_of<hip_comms::AllReduceRmsNormGemmLaunch>(
        hip_comms::Error::output_not_two_d);
  auto& h = handle_of(handle_ptr);
  const hip_comms::DType dt = std::get<hip_comms::DType>(d);
  const auto tm = narrowed(tile_m), tn = narrowed(tile_n), tk = narrowed(tile_k);
  const auto sk = narrowed(slice_k), tpb = narrowed(threads_per_block);
  const auto bpg = narrowed(blocks_per_grid);
  if (add)
    return gemm_plan_of(hip_comms::select_all_reduce_rms_norm_gemm_add(
        h, nullptr, 0, nullptr, nullptr, 0.f, nullptr, gemm_weight.size(0), nullptr, dt,
        inp.size(0), inp.size(1), alg, dir, tm, tn, tk, sk, tpb, bpg, current_stream()));
  return gemm_plan_of(hip_comms::select_all_reduce_rms_norm_gemm(
      h, nullptr, 0, nullptr, nullptr, 0.f, nullptr, gemm_weight.size(0), nullptr, dt,
      inp.size(0), inp.size(1), alg, dir, tm, tn, tk, sk, tpb, bpg, current_stream()));
}

// `inp`'s row is [shared | projected | latent] and `out` [rows, hidden]: the latent is what is
// left.
RocmCommsRowPlan rocm_comms_plan_all_reduce_rms_scale_add(
    fptr_t handle_ptr, const torch::Tensor& inp, const torch::Tensor& out,
    std::optional<std::string> algorithm, std::optional<std::string> direction,
    std::optional<int64_t> tile_n, std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid) {
  const auto alg = algorithm_from(algorithm);
  const auto dir = direction_from(direction);
  const auto d = admitted(inp, true);
  using L = hip_comms::AllReduceRmsScaleAddLaunch;
  if (const auto* e = std::get_if<hip_comms::Error>(&d)) return row_plan_of<L>(*e);
  if (out.dim() != 2) return row_plan_of<L>(hip_comms::Error::output_not_two_d);
  const int64_t hidden = out.size(1), latent = inp.size(1) - 2 * hidden;
  if (latent < 1) return row_plan_of<L>(hip_comms::Error::row_not_wider_than_output);
  return row_plan_of(hip_comms::select_all_reduce_rms_scale_add(
      handle_of(handle_ptr), nullptr, nullptr, std::get<hip_comms::DType>(d), inp.size(0), hidden,
      latent, 0.f, alg, dir, narrowed(tile_n), narrowed(threads_per_block),
      narrowed(blocks_per_grid), current_stream()));
}

// THE COMMUNICATOR OPENED on `device` over the process groups named `cpu_group` (which carries the
// handles) and `device_group`, a collective: this rank's symmetric memory, made and owned here, its
// handle gathered with every rank's, and the peers' opened. A torch op cannot return a variant, so
// it is two optionals and exactly one is set, the handle or the Error's number.
OpenWire rocm_comms_open(const std::string& cpu_group, const std::string& device_group,
                         int64_t device) {
  const auto refused = [](hip_comms::Error e) {
    return OpenWire{std::nullopt, static_cast<int64_t>(e)};
  };
  const auto pg = resolved(cpu_group), dg = resolved(device_group);
  if (!pg || !dg) return refused(hip_comms::Error::no_such_group);
  if ((*pg)->getSize() != (*dg)->getSize() || (*pg)->getRank() != (*dg)->getRank())
    return refused(hip_comms::Error::groups_disagree);
  const int world = (*pg)->getSize();
  const auto ok   = hip_comms::supported(static_cast<int>(device), world);
  if (const auto* e = std::get_if<hip_comms::Error>(&ok)) return refused(*e);
  auto* handle = new hip_comms::Handle((*pg)->getRank(), world, gather_over(*pg));
  return {reinterpret_cast<fptr_t>(handle), std::nullopt};
}

// SUPPORTED AT THE TORCH BOUNDARY: a torch op cannot return a variant, so it is two optionals and
// exactly one is set, the arch or the Error's number.
SupportedWire rocm_comms_supported(int64_t device, int64_t world) {
  const auto got = hip_comms::supported(static_cast<int>(device), static_cast<int>(world));
  if (const auto* e = std::get_if<hip_comms::Error>(&got))
    return {std::nullopt, static_cast<int64_t>(*e)};
  return {std::get<hip_comms::Supported>(got).arch, std::nullopt};
}

// WHAT THE BUILD HOLDS, kBuild's projection for Python: its dtypes by name, its worlds, a pack's
// bytes and a staging's, its ops and errors by name in their enums' order (each error's name
// without its reason), which Python's `Op` and `Error` are held to, and its templates.
BuildInfoWire rocm_comms_build_info() {
  using namespace hip_comms;
  Names dtypes, ops, errors;
  for (const DType d : kBuild.supports.dtypes) dtypes.push_back(to_string(d));
  const std::vector<int64_t> worlds(kBuild.supports.worlds.begin(), kBuild.supports.worlds.end());
  for (int i = 0; i < kNumOps; ++i) ops.push_back(to_string(static_cast<OpType>(i)));
  for (int i = 0; i < kNumErrors; ++i) {
    const std::string s = to_string(static_cast<Error>(i));
    errors.push_back(s.substr(0, s.find(':')));
  }
  // Each template's op, and its configs (what dispatch instantiates: the
  // tuner's search space), flat: a config's fields in `fields`' order, 0 where its family has
  // none.
  const Names fields = {"tile_m",  "tile_n", "tile_k", "slice_k", "reduce_scatter_blocks",
                        "threads_per_block", "blocks_per_grid"};
  Names templates, template_ops;
  std::vector<int64_t> configs, counts;
  for (const TemplateInfo& t : kTemplates) {
    templates.push_back(t.name);
    template_ops.push_back(to_string(op_of(t.fn)));
    counts.push_back(static_cast<int64_t>(t.configs.size()));
    for (const KernelConfig& c : t.configs)
      std::visit(
          [&](const auto& f) {
            int64_t m = 0, n = 0, k = 0, sk = 0, rs = 0;
            if constexpr (requires { f.tile_m; }) m = f.tile_m;
            if constexpr (requires { f.tile_n; }) n = f.tile_n;
            if constexpr (requires { f.tile_k; }) k = f.tile_k;
            if constexpr (requires { f.slice_k; }) sk = f.slice_k;
            if constexpr (requires { f.reduce_scatter_blocks; }) rs = f.reduce_scatter_blocks;
            configs.insert(configs.end(), {m, n, k, sk, rs, f.launch.threads_per_block,
                                           f.launch.blocks_per_grid});
          },
          c);
  }
  return {dtypes, worlds, kBuild.memory.pack_bytes, kBuild.memory.staging_bytes, ops, errors,
          templates, template_ops, fields, configs, counts};
}

void rocm_comms_dispose(fptr_t handle_ptr) { delete &handle_of(handle_ptr); }



// THE PER-PHASE STAMPS SINCE THE LAST READ, [block][phase] device clock ticks (100 MHz), zero where
// no block stamped, then zeroed. Only a HIP_COMMS_STAMPS build writes them (block_stamp).
torch::Tensor rocm_comms_stamps() {
  auto out = torch::zeros({hip_comms::kStampBlocks, hip_comms::kStampPhases},
                          torch::TensorOptions().dtype(torch::kInt64));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpyFromSymbol(out.data_ptr<int64_t>(), HIP_SYMBOL(hip_comms::g_stamps),
                                sizeof(hip_comms::g_stamps)));
  // Zeroed after reading, so the next launch's table holds only its own blocks.
  void* table = nullptr;
  HIP_CHECK(hipGetSymbolAddress(&table, HIP_SYMBOL(hip_comms::g_stamps)));
  HIP_CHECK(hipMemset(table, 0, sizeof(hip_comms::g_stamps)));
  return out;
}

// THE PROBE, every rank together over the process group named `group` (experimental::probe).
ProbeWire rocm_comms_probe(fptr_t handle_ptr, const std::string& group, int64_t bytes,
                           int64_t ping_iters, int64_t traffic_iters, int64_t trials) {
  const auto pg = resolved(group);
  if (!pg) raise(hip_comms::Error::no_such_group);
  const auto got = hip_comms::experimental::probe(
      handle_of(handle_ptr), gather_over(*pg), bytes, *narrowed(ping_iters),
      *narrowed(traffic_iters), *narrowed(trials), current_stream());
  if (const auto* e = std::get_if<hip_comms::Error>(&got)) raise(*e);
  const auto& r = std::get<hip_comms::ProbeResult>(got);
  return {r.ping_ns, r.names, r.gbytes_per_s};
}

// THE BUFFERS A CAPTURE RECORDED, registered over the process group named `group`, every rank
// together (Handle::register_captured).
void rocm_comms_register_captured(fptr_t handle_ptr, const std::string& group) {
  const auto pg = resolved(group);
  if (!pg) raise(hip_comms::Error::no_such_group);
  if (!handle_of(handle_ptr).register_captured(gather_over(*pg)))
    raise(hip_comms::Error::ranks_disagree);
}

void rocm_comms_all_reduce(fptr_t handle_ptr, torch::Tensor& out, torch::Tensor& inp,
                           std::optional<std::string> algorithm,
                           std::optional<std::string> direction,
                           std::optional<int64_t> threads_per_block,
                           std::optional<int64_t> blocks_per_grid) {
  check_device_contiguous({&out, &inp});
  TORCH_CHECK(out.sizes() == inp.sizes(), "out and inp must have the same shape");
  TORCH_CHECK(out.scalar_type() == inp.scalar_type(), "out and inp must share a dtype");
  ran(hip_comms::all_reduce(handle_of(handle_ptr), out.data_ptr(), inp.data_ptr(),
                            inp.numel() * inp.element_size(), dtype_of(inp),
                            algorithm_from(algorithm), direction_from(direction),
                            narrowed(threads_per_block), narrowed(blocks_per_grid),
                            current_stream()));
}

namespace {
// A NORM CALL'S TENSORS: inp [tokens, hidden], out (and with a residual, residual and
// residual_out) its shape and dtype, the weight 1-D of hidden in inp's dtype or fp32.
void norm_tensors(const torch::Tensor& out, const torch::Tensor& inp, const torch::Tensor& weight,
                  const torch::Tensor* residual, const torch::Tensor* residual_out) {
  check_device_contiguous({&out, &inp, &weight});
  TORCH_CHECK(inp.dim() == 2, "inp must be 2-D [tokens, hidden]; got ", inp.dim(), "-D");
  TORCH_CHECK(out.sizes() == inp.sizes(), "out must have inp's shape");
  TORCH_CHECK(out.scalar_type() == inp.scalar_type(), "out must share inp's dtype");
  // THE WEIGHT IN ITS OWN DTYPE: inp's, or fp32, the two a norm's weight is kept in.
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::Float ||
                  weight.scalar_type() == inp.scalar_type(),
              "weight must be inp's dtype or float32; got ", weight.scalar_type());
  TORCH_CHECK(weight.dim() == 1 && weight.numel() == inp.size(1),
              "weight must be 1-D of hidden=", inp.size(1));
  // The weight is read a pack at a time, a pack's worth of inp's elements per load.
  const int64_t weight_pack =
      hip_comms::kBuild.memory.pack_bytes / inp.element_size() * weight.element_size();
  TORCH_CHECK(reinterpret_cast<uintptr_t>(weight.data_ptr()) % weight_pack == 0,
              "weight must be aligned to ", weight_pack, " bytes");
  if (residual != nullptr) {
    check_device_contiguous({residual, residual_out});
    TORCH_CHECK(residual->sizes() == inp.sizes() && residual_out->sizes() == inp.sizes(),
                "residual and residual_out must have inp's shape");
    TORCH_CHECK(residual->scalar_type() == inp.scalar_type() &&
                    residual_out->scalar_type() == inp.scalar_type(),
                "every tensor must share inp's dtype");
  }
}
}  // namespace

void rocm_comms_all_reduce_rms_norm(fptr_t handle_ptr, torch::Tensor& out, torch::Tensor& inp,
                                    torch::Tensor& weight, double eps,
                                    std::optional<std::string> algorithm,
                                    std::optional<std::string> direction,
                                    std::optional<int64_t> tile_n,
                                    std::optional<int64_t> threads_per_block,
                                    std::optional<int64_t> blocks_per_grid) {
  norm_tensors(out, inp, weight, nullptr, nullptr);
  ran(hip_comms::all_reduce_rms_norm(
      handle_of(handle_ptr), out.data_ptr(), inp.data_ptr(), weight.data_ptr(), dtype_of(inp),
      dtype_of(weight), inp.size(0), inp.size(1), static_cast<float>(eps),
      algorithm_from(algorithm), direction_from(direction), narrowed(tile_n),
      narrowed(threads_per_block), narrowed(blocks_per_grid), current_stream()));
}

void rocm_comms_all_reduce_add_rms_norm(
    fptr_t handle_ptr, torch::Tensor& out, torch::Tensor& residual_out, torch::Tensor& inp,
    torch::Tensor& residual, torch::Tensor& weight, double eps,
    std::optional<std::string> algorithm, std::optional<std::string> direction,
    std::optional<int64_t> tile_n, std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid) {
  norm_tensors(out, inp, weight, &residual, &residual_out);
  ran(hip_comms::all_reduce_add_rms_norm(
      handle_of(handle_ptr), out.data_ptr(), residual_out.data_ptr(), inp.data_ptr(),
      residual.data_ptr(), weight.data_ptr(), dtype_of(inp), dtype_of(weight), inp.size(0),
      inp.size(1), static_cast<float>(eps), algorithm_from(algorithm), direction_from(direction),
      narrowed(tile_n), narrowed(threads_per_block), narrowed(blocks_per_grid),
      current_stream()));
}

namespace {
// AN ATTNRES CALL'S TENSORS, as every AttnRes op takes them: `inp` (the partial sum, or the delta)
// [tokens, hidden], prefix and out its shape, blocks [tokens, sources, hidden], every one its
// dtype.
void attn_res_tensors(const torch::Tensor& prefix, const torch::Tensor& out,
                      const torch::Tensor& inp, const torch::Tensor& blocks,
                      const torch::Tensor& norm_weight, const torch::Tensor& qk_weight,
                      const std::optional<torch::Tensor>& out_norm_weight, int64_t num_blocks,
                      int64_t write_idx) {
  check_device_contiguous({&prefix, &out, &inp, &norm_weight, &qk_weight});
  TORCH_CHECK(inp.dim() == 2, "inp must be 2-D [tokens, hidden]; got ", inp.dim(), "-D");
  const int64_t hidden = inp.size(1);
  TORCH_CHECK(prefix.sizes() == inp.sizes() && out.sizes() == inp.sizes(),
              "prefix and out must have inp's shape");
  TORCH_CHECK(blocks.is_cuda() && blocks.dim() == 3 && blocks.size(0) == inp.size(0) &&
                  blocks.size(2) == hidden && blocks.stride(2) == 1,
              "blocks must be [tokens, sources, hidden] with a unit hidden stride");
  TORCH_CHECK(num_blocks >= 0 && num_blocks <= blocks.size(1), "num_blocks must be in [0, ",
              blocks.size(1), "]");
  TORCH_CHECK(write_idx < blocks.size(1), "write_idx must be < ", blocks.size(1));
  std::vector<const torch::Tensor*> same = {&prefix, &out, &blocks, &norm_weight, &qk_weight};
  if (out_norm_weight) {
    check_device_contiguous({&*out_norm_weight});
    same.push_back(&*out_norm_weight);
  }
  for (const torch::Tensor* t : same)
    TORCH_CHECK(t->scalar_type() == inp.scalar_type(), "every tensor must share inp's dtype");
  for (const torch::Tensor* t : {&norm_weight, &qk_weight})
    TORCH_CHECK(t->dim() == 1 && t->numel() == hidden, "weights must be 1-D of hidden=", hidden);
  if (out_norm_weight)
    TORCH_CHECK(out_norm_weight->dim() == 1 && out_norm_weight->numel() == hidden,
                "out_norm_weight must be 1-D of hidden=", hidden);
  const int64_t lanes = hip_comms::kBuild.memory.pack_bytes / inp.element_size();
  TORCH_CHECK(blocks.stride(0) % lanes == 0 && blocks.stride(1) % lanes == 0 &&
                  reinterpret_cast<uintptr_t>(blocks.data_ptr()) %
                          hip_comms::kBuild.memory.pack_bytes ==
                      0,
              "blocks must be 16-byte aligned in every row and source");
}
}  // namespace

void rocm_comms_all_reduce_add_attn_res_rms_norm(
    fptr_t handle_ptr, torch::Tensor& prefix, torch::Tensor& out, torch::Tensor& inp,
    torch::Tensor& blocks, torch::Tensor& norm_weight, torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& out_norm_weight, int64_t num_blocks, int64_t write_idx,
    double eps, double out_eps, bool has_prefix, std::optional<std::string> algorithm,
    std::optional<std::string> direction, std::optional<int64_t> tile_m,
    std::optional<int64_t> tile_n, std::optional<int64_t> tile_k,
    std::optional<int64_t> reduce_scatter_blocks, std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid) {
  attn_res_tensors(prefix, out, inp, blocks, norm_weight, qk_weight, out_norm_weight,
                   num_blocks, write_idx);
  ran(hip_comms::all_reduce_add_attn_res_rms_norm(
      handle_of(handle_ptr), prefix.data_ptr(), out.data_ptr(), inp.data_ptr(),
      blocks.data_ptr(), blocks.stride(0), blocks.stride(1), norm_weight.data_ptr(),
      qk_weight.data_ptr(), out_norm_weight ? out_norm_weight->data_ptr() : nullptr,
      dtype_of(inp), inp.size(0), inp.size(1), static_cast<int>(num_blocks),
      static_cast<int>(write_idx), static_cast<float>(eps), static_cast<float>(out_eps),
      has_prefix, algorithm_from(algorithm), direction_from(direction), narrowed(tile_m),
      narrowed(tile_n), narrowed(tile_k), narrowed(reduce_scatter_blocks),
      narrowed(threads_per_block), narrowed(blocks_per_grid), current_stream()));
}

namespace {
// A GEMM-TAIL CALL'S TENSORS: inp [rows, hidden]; workspace its shape and dtype; gemm_weight
// [N, hidden] with contiguous rows; out [rows, N] with a unit column stride (a column slice of a
// wider buffer allowed); every one inp's dtype.
void gemm_tensors(const torch::Tensor& out, const torch::Tensor& inp,
                  const torch::Tensor& norm_weight, const torch::Tensor& gemm_weight,
                  const torch::Tensor& workspace) {
  check_device_contiguous({&inp, &norm_weight, &workspace});
  TORCH_CHECK(inp.dim() == 2, "inp must be 2-D");
  const int64_t rows = inp.size(0), hidden = inp.size(1);
  TORCH_CHECK(workspace.sizes() == inp.sizes() && workspace.scalar_type() == inp.scalar_type(),
              "workspace must be of inp's shape and dtype");
  TORCH_CHECK(gemm_weight.is_cuda() && gemm_weight.dim() == 2 && gemm_weight.size(1) == hidden &&
                  gemm_weight.stride(1) == 1 && gemm_weight.stride(0) == hidden,
              "gemm_weight must be [N, hidden] with contiguous rows");
  TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.size(0) == rows &&
                  out.size(1) == gemm_weight.size(0) && out.stride(1) == 1,
              "out must be [rows, N] with a unit column stride");
  TORCH_CHECK(norm_weight.dim() == 1 && norm_weight.numel() == hidden,
              "norm_weight must be 1-D of hidden=", hidden);
  for (const torch::Tensor* t : {&out, &norm_weight, &gemm_weight})
    TORCH_CHECK(t->scalar_type() == inp.scalar_type(), "every tensor must share inp's dtype");
}
}  // namespace

void rocm_comms_all_reduce_rms_norm_gemm(
    fptr_t handle_ptr, torch::Tensor& out, torch::Tensor& inp, torch::Tensor& norm_weight,
    double eps, torch::Tensor& gemm_weight, torch::Tensor& workspace,
    std::optional<std::string> algorithm, std::optional<std::string> direction,
    std::optional<int64_t> tile_m, std::optional<int64_t> tile_n, std::optional<int64_t> tile_k,
    std::optional<int64_t> slice_k, std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid) {
  gemm_tensors(out, inp, norm_weight, gemm_weight, workspace);
  ran(hip_comms::all_reduce_rms_norm_gemm(
      handle_of(handle_ptr), out.data_ptr(), out.stride(0), inp.data_ptr(),
      norm_weight.data_ptr(), static_cast<float>(eps), gemm_weight.data_ptr(),
      gemm_weight.size(0), workspace.data_ptr(), dtype_of(inp), inp.size(0), inp.size(1),
      algorithm_from(algorithm), direction_from(direction), narrowed(tile_m), narrowed(tile_n),
      narrowed(tile_k), narrowed(slice_k), narrowed(threads_per_block),
      narrowed(blocks_per_grid), current_stream()));
}

void rocm_comms_all_reduce_rms_norm_gemm_add(
    fptr_t handle_ptr, torch::Tensor& out, torch::Tensor& inp, torch::Tensor& norm_weight,
    double eps, torch::Tensor& gemm_weight, torch::Tensor& workspace,
    std::optional<std::string> algorithm, std::optional<std::string> direction,
    std::optional<int64_t> tile_m, std::optional<int64_t> tile_n, std::optional<int64_t> tile_k,
    std::optional<int64_t> slice_k, std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid) {
  gemm_tensors(out, inp, norm_weight, gemm_weight, workspace);
  ran(hip_comms::all_reduce_rms_norm_gemm_add(
      handle_of(handle_ptr), out.data_ptr(), out.stride(0), inp.data_ptr(),
      norm_weight.data_ptr(), static_cast<float>(eps), gemm_weight.data_ptr(),
      gemm_weight.size(0), workspace.data_ptr(), dtype_of(inp), inp.size(0), inp.size(1),
      algorithm_from(algorithm), direction_from(direction), narrowed(tile_m), narrowed(tile_n),
      narrowed(tile_k), narrowed(slice_k), narrowed(threads_per_block),
      narrowed(blocks_per_grid), current_stream()));
}

// out [rows, hidden] = shared + projected * rsqrt(mean(latent^2) + eps), inp's row [shared |
// projected | latent] summed over the ranks first: the widths are out's, out's again, and the
// rest.
void rocm_comms_all_reduce_rms_scale_add(
    fptr_t handle_ptr, torch::Tensor& out, torch::Tensor& inp, double eps,
    std::optional<std::string> algorithm, std::optional<std::string> direction,
    std::optional<int64_t> tile_n, std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid) {
  check_device_contiguous({&out, &inp});
  TORCH_CHECK(inp.dim() == 2 && out.dim() == 2 && out.size(0) == inp.size(0),
              "inp and out must be 2-D with the same rows");
  const int64_t hidden = out.size(1);
  const int64_t latent = inp.size(1) - 2 * hidden;
  TORCH_CHECK(latent > 0,
              "inp's row must be wider than twice out's: [shared | projected | latent]");
  TORCH_CHECK(out.scalar_type() == inp.scalar_type(), "out must share inp's dtype");
  ran(hip_comms::all_reduce_rms_scale_add(
      handle_of(handle_ptr), out.data_ptr(), inp.data_ptr(), dtype_of(inp), inp.size(0), hidden,
      latent, static_cast<float>(eps), algorithm_from(algorithm), direction_from(direction),
      narrowed(tile_n), narrowed(threads_per_block), narrowed(blocks_per_grid),
      current_stream()));
}

// EXPERIMENTAL: AttnRes on a local `delta` (no all-reduce), prefix updated in place to
// prefix + delta.
void rocm_comms_add_attn_res_rms_norm(
    torch::Tensor& prefix, torch::Tensor& out, torch::Tensor& delta, torch::Tensor& blocks,
    torch::Tensor& norm_weight, torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& out_norm_weight, int64_t num_blocks, int64_t write_idx,
    double eps, double out_eps, std::optional<int64_t> tile_n, std::optional<int64_t> tile_k,
    std::optional<int64_t> threads_per_block, std::optional<int64_t> blocks_per_grid) {
  attn_res_tensors(prefix, out, delta, blocks, norm_weight, qk_weight, out_norm_weight,
                   num_blocks, write_idx);
  ran(hip_comms::experimental::add_attn_res_rms_norm(
      prefix.data_ptr(), out.data_ptr(), delta.data_ptr(), blocks.data_ptr(), blocks.stride(0),
      blocks.stride(1), norm_weight.data_ptr(), qk_weight.data_ptr(),
      out_norm_weight ? out_norm_weight->data_ptr() : nullptr, dtype_of(delta), delta.size(0),
      delta.size(1), static_cast<int>(num_blocks), static_cast<int>(write_idx),
      static_cast<float>(eps), static_cast<float>(out_eps), narrowed(tile_n), narrowed(tile_k),
      narrowed(threads_per_block), narrowed(blocks_per_grid), current_stream()));
}
