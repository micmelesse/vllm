// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Our HIP collectives as torch ops, built into `_rocm_C`: this file is only the wrapper. Each op
// checks its tensors, turns them into the op's Args and its trailing integers into Options, and
// calls the op in rocm_comms/rocm_comms.cuh, which is torch-free and holds everything else.
// Self-contained: torch and the HIP runtime, nothing from aiter.

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <variant>
#include <string>
#include <vector>

#include "rocm_comms/probe.cuh"
#include "rocm_comms/rocm_comms.cuh"

// =================================================================================
// THE TORCH OP BOUNDARY. `Handle` (handle.cuh) is a stateful C++ object and a torch op is a
// free function over schema types, so the object crosses as an opaque pointer -- the same
// `fptr_t = int64_t` vLLM's custom all-reduce and quick-reduce use. The IPC handles never cross:
// the Handle gathers them over the process group this boundary resolves by name.
// =================================================================================

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

// EVERY OP TAKES THE SAME FOUR VALUES LAST, its Options: quant_bits, a lossy precision (none:
// exact), then a forced template by name with its blocks and threads (none: select's). The model
// passes none of them.

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

// A FORCED KERNEL, as Options carries it: a template by name, and with it a KernelConfig
// (threads_per_block and grid together, tile_m and tile_n each where given, else the template's
// own), a schema having no struct. A name that names no template is the caller's Error; the rest
// are its mistakes, raised.
std::variant<hip_comms::Options, hip_comms::Error> options_of(
    std::optional<int64_t> quant_bits, const std::optional<std::string>& template_,
    std::optional<int64_t> tile_m, std::optional<int64_t> tile_n, std::optional<int64_t> tile_k,
    std::optional<int64_t> slice_k, std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid) {
  TORCH_CHECK(!quant_bits || *quant_bits == 8 || *quant_bits == 4,
              "quant_bits is 8 or 4, or none (exact)");
  const std::optional<hip_comms::QuantBits> bits =
      quant_bits ? std::optional(static_cast<hip_comms::QuantBits>(*quant_bits)) : std::nullopt;
  const bool config = blocks_per_grid.has_value();
  TORCH_CHECK(
      config == threads_per_block.has_value(),
      "hip_comms: a forced KernelConfig gives its threads_per_block and blocks_per_grid together");
  TORCH_CHECK(config || (!tile_m && !tile_n && !tile_k && !slice_k),
              "hip_comms: a forced tile needs its launch");
  TORCH_CHECK(!config || template_, "hip_comms: a forced KernelConfig needs its template");
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (!template_) return hip_comms::Options{bits, std::nullopt, std::nullopt, stream};
  const std::optional<hip_comms::Template> t = hip_comms::template_named(*template_);
  if (!t) return hip_comms::Error::no_such_template;
  if (!config) return hip_comms::Options{bits, t, std::nullopt, stream};
  TORCH_CHECK(*blocks_per_grid > 0 && *blocks_per_grid <= hip_comms::p2p::kMaxBlocks,
              "blocks_per_grid must be in [1, ", hip_comms::p2p::kMaxBlocks, "]");
  TORCH_CHECK(*threads_per_block > 0 &&
                  *threads_per_block <= hip_comms::kBuild.kernels.max_threads &&
                  *threads_per_block % hip_comms::kWaveSize == 0,
              "threads_per_block must be a multiple of ", hip_comms::kWaveSize, " up to ",
              hip_comms::kBuild.kernels.max_threads);
  TORCH_CHECK(tile_m.value_or(1) > 0 && tile_n.value_or(1) > 0 && tile_k.value_or(1) > 0 &&
                  slice_k.value_or(1) > 0,
              "tile_m, tile_n, tile_k and slice_k are positive");
  return hip_comms::Options{
      bits, t,
      hip_comms::KernelConfig{
          static_cast<int>(tile_m.value_or(0)), static_cast<int>(tile_n.value_or(0)),
          static_cast<int>(tile_k.value_or(0)), static_cast<int>(slice_k.value_or(0)),
          static_cast<int>(*threads_per_block), static_cast<int>(*blocks_per_grid)},
      stream};
}

// AN ERROR RAISED, at the torch boundary: a torch op can only return or raise.
[[noreturn]] void raise(hip_comms::Error e) {
  TORCH_CHECK(false, "hip_comms: ", hip_comms::to_string(e));
  __builtin_unreachable();
}

// An op's options, its Error raised: a torch op can only return or raise.
hip_comms::Options options_or_raise(std::optional<int64_t> quant_bits,
                                    const std::optional<std::string>& template_,
                                    std::optional<int64_t> tile_m, std::optional<int64_t> tile_n,
                                    std::optional<int64_t> tile_k, std::optional<int64_t> slice_k,
                                    std::optional<int64_t> threads_per_block,
                                    std::optional<int64_t> blocks_per_grid) {
  auto o = options_of(quant_bits, template_, tile_m, tile_n, tile_k, slice_k, threads_per_block,
                      blocks_per_grid);
  if (const auto* e = std::get_if<hip_comms::Error>(&o)) raise(*e);
  return std::get<hip_comms::Options>(o);
}

void check_device_contiguous(std::initializer_list<const torch::Tensor*> ts) {
  for (const torch::Tensor* t : ts) {
    TORCH_CHECK(t->is_cuda(), "every tensor must be on device");
    TORCH_CHECK(t->is_contiguous(), "every tensor must be contiguous");
  }
}
}  // namespace

// AN OP'S ERROR RAISED.
static void ran(const std::variant<hip_comms::Kernel, hip_comms::Error>& result) {
  if (const hip_comms::Error* e = std::get_if<hip_comms::Error>(&result)) raise(*e);
}

// WHAT CROSSES THE TORCH BOUNDARY, as the tuples an op schema can return.
using Names         = std::vector<std::string>;
using PlanWire =
    std::tuple<std::optional<std::string>, std::optional<int64_t>, std::optional<int64_t>,
               std::optional<int64_t>, std::optional<int64_t>, std::optional<int64_t>,
               std::optional<int64_t>, std::optional<int64_t>>;
using SupportedWire = std::tuple<std::optional<std::string>, std::optional<int64_t>>;
using OpenWire      = std::tuple<std::optional<int64_t>, std::optional<int64_t>>;
using ProbeWire     = std::tuple<std::vector<double>, Names, std::vector<double>>;
using BuildInfoWire = std::tuple<Names, std::vector<int64_t>, int64_t, int64_t, Names, Names>;

// A torch dtype as ours, or none for one ours has no name for.
std::optional<hip_comms::DType> dtype_from(at::ScalarType s) {
  if (s == at::ScalarType::Half) return hip_comms::DType::f16;
  if (s == at::ScalarType::BFloat16) return hip_comms::DType::bf16;
  if (s == at::ScalarType::Float) return hip_comms::DType::f32;
  return std::nullopt;
}

// WHAT RUNS A CALL, or the first Error it meets: `hip_comms::plan`, one planner per op family,
// each handed the call's own tensors and reading their facts here. A torch op cannot return a
// variant, so each returns the kernel's template (by name) and its KernelConfig's six fields, or
// the Error's number: the seven or the one. The arguments a call's kernel reads are not, so they
// are null.
namespace {
PlanWire error_wire(hip_comms::Error e) {
  return PlanWire{std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                  std::nullopt, std::nullopt, std::nullopt, static_cast<int64_t>(e)};
}

// AN INPUT AS OURS: its dtype, or the Error it meets first (not contiguous, not 2-D where the op
// reads rows, a dtype ours has no name for).
std::variant<hip_comms::DType, hip_comms::Error> admitted(const torch::Tensor& inp, bool rows) {
  if (!inp.is_contiguous()) return hip_comms::Error::not_contiguous;
  if (rows && inp.dim() != 2) return hip_comms::Error::not_two_d;
  const std::optional<hip_comms::DType> d = dtype_from(inp.scalar_type());
  if (!d) return hip_comms::Error::dtype_not_built;
  return *d;
}

template <typename Args>
PlanWire planned(fptr_t handle_ptr, const Args& a, std::optional<int64_t> quant_bits,
                 std::optional<std::string> template_, std::optional<int64_t> tile_m,
                 std::optional<int64_t> tile_n, std::optional<int64_t> tile_k,
                 std::optional<int64_t> slice_k, std::optional<int64_t> threads_per_block,
                 std::optional<int64_t> blocks_per_grid) {
  const auto forced = options_of(quant_bits, template_, tile_m, tile_n, tile_k, slice_k,
                                 threads_per_block, blocks_per_grid);
  if (const auto* e = std::get_if<hip_comms::Error>(&forced)) return error_wire(*e);
  const std::variant<hip_comms::Kernel, hip_comms::Error> p =
      hip_comms::plan(handle_of(handle_ptr), a, std::get<hip_comms::Options>(forced));
  if (const auto* e = std::get_if<hip_comms::Error>(&p)) return error_wire(*e);
  const auto& k = std::get<hip_comms::Kernel>(p);
  return PlanWire{std::string(hip_comms::to_string(k.fn)),
                  k.config.tile_m,
                  k.config.tile_n,
                  k.config.tile_k,
                  k.config.slice_k,
                  k.config.threads_per_block,
                  k.config.blocks_per_grid,
                  std::nullopt};
}
}  // namespace

PlanWire rocm_comms_plan_all_reduce(fptr_t handle_ptr, const torch::Tensor& inp,
                                    std::optional<int64_t> quant_bits,
                                    std::optional<std::string> template_,
                                    std::optional<int64_t> tile_m, std::optional<int64_t> tile_n,
                                    std::optional<int64_t> tile_k, std::optional<int64_t> slice_k,
                                    std::optional<int64_t> threads_per_block,
                                    std::optional<int64_t> blocks_per_grid) {
  const auto d = admitted(inp, false);
  if (const auto* e = std::get_if<hip_comms::Error>(&d)) return error_wire(*e);
  const hip_comms::DType dtype = std::get<hip_comms::DType>(d);
  return planned(
      handle_ptr,
      hip_comms::AllReduceArgs{nullptr, nullptr, inp.numel() * inp.element_size(), dtype},
      quant_bits, template_, tile_m, tile_n, tile_k, slice_k, threads_per_block, blocks_per_grid);
}

// `add`: fused_add_rms_norm; otherwise rms_norm.
PlanWire rocm_comms_plan_all_reduce_rms_norm(
    fptr_t handle_ptr, const torch::Tensor& inp, const torch::Tensor& weight, bool add,
    std::optional<int64_t> quant_bits, std::optional<std::string> template_,
    std::optional<int64_t> tile_m, std::optional<int64_t> tile_n, std::optional<int64_t> tile_k,
    std::optional<int64_t> slice_k, std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid) {
  const auto d = admitted(inp, true);
  if (const auto* e = std::get_if<hip_comms::Error>(&d)) return error_wire(*e);
  const std::optional<hip_comms::DType> w = dtype_from(weight.scalar_type());
  if (!w) return error_wire(hip_comms::Error::weight_not_built);
  return planned(handle_ptr,
                 hip_comms::NormArgs{add, nullptr, nullptr, nullptr, std::get<hip_comms::DType>(d),
                                     *w, inp.size(0), inp.size(1), 0.f, nullptr, nullptr},
                 quant_bits, template_, tile_m, tile_n, tile_k, slice_k, threads_per_block,
                 blocks_per_grid);
}

PlanWire rocm_comms_plan_all_reduce_add_attn_res_rms_norm(
    fptr_t handle_ptr, const torch::Tensor& inp, std::optional<int64_t> quant_bits,
    std::optional<std::string> template_, std::optional<int64_t> tile_m,
    std::optional<int64_t> tile_n, std::optional<int64_t> tile_k, std::optional<int64_t> slice_k,
    std::optional<int64_t> threads_per_block, std::optional<int64_t> blocks_per_grid) {
  const auto d = admitted(inp, true);
  if (const auto* e = std::get_if<hip_comms::Error>(&d)) return error_wire(*e);
  return planned(handle_ptr,
                 hip_comms::AttnResArgs{nullptr, nullptr, nullptr, nullptr, 0, 0, nullptr, nullptr,
                                        nullptr, std::get<hip_comms::DType>(d), inp.size(0),
                                        inp.size(1), 0, -1, 0.f, 0.f, false},
                 quant_bits, template_, tile_m, tile_n, tile_k, slice_k, threads_per_block,
                 blocks_per_grid);
}

// `add`: the GEMM added into the output; otherwise written. `gemm_weight` is [N, hidden].
PlanWire rocm_comms_plan_all_reduce_rms_norm_gemm(
    fptr_t handle_ptr, const torch::Tensor& inp, const torch::Tensor& gemm_weight, bool add,
    std::optional<int64_t> quant_bits, std::optional<std::string> template_,
    std::optional<int64_t> tile_m, std::optional<int64_t> tile_n, std::optional<int64_t> tile_k,
    std::optional<int64_t> slice_k, std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid) {
  const auto d = admitted(inp, true);
  if (const auto* e = std::get_if<hip_comms::Error>(&d)) return error_wire(*e);
  if (gemm_weight.dim() != 2) return error_wire(hip_comms::Error::output_not_two_d);
  return planned(handle_ptr,
                 hip_comms::GemmTailArgs{.add = add,
                                         .n_cols = gemm_weight.size(0),
                                         .dtype = std::get<hip_comms::DType>(d),
                                         .rows = inp.size(0),
                                         .hidden = inp.size(1)},
                 quant_bits, template_, tile_m, tile_n, tile_k, slice_k, threads_per_block,
                 blocks_per_grid);
}

// `inp`'s row is [shared | projected | latent] and `out` [rows, hidden]: the latent is what is
// left.
PlanWire rocm_comms_plan_all_reduce_rms_scale_add(
    fptr_t handle_ptr, const torch::Tensor& inp, const torch::Tensor& out,
    std::optional<int64_t> quant_bits, std::optional<std::string> template_,
    std::optional<int64_t> tile_m, std::optional<int64_t> tile_n, std::optional<int64_t> tile_k,
    std::optional<int64_t> slice_k, std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid) {
  const auto d = admitted(inp, true);
  if (const auto* e = std::get_if<hip_comms::Error>(&d)) return error_wire(*e);
  if (out.dim() != 2) return error_wire(hip_comms::Error::output_not_two_d);
  const int64_t hidden = out.size(1), latent = inp.size(1) - 2 * hidden;
  if (latent < 1) return error_wire(hip_comms::Error::row_not_wider_than_output);
  return planned(handle_ptr,
                 hip_comms::ScaleAddArgs{nullptr, nullptr, std::get<hip_comms::DType>(d),
                                         inp.size(0), hidden, latent, 0.f},
                 quant_bits, template_, tile_m, tile_n, tile_k, slice_k, threads_per_block,
                 blocks_per_grid);
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
// bytes and a staging's, and its ops
// and errors by name in their enums' order (each error's name without its reason), which Python's
// `Op` and `Error` are held to.
BuildInfoWire rocm_comms_build_info() {
  using namespace hip_comms;
  Names dtypes, ops, errors;
  for (const DType d : kBuild.supports.dtypes) dtypes.push_back(to_string(d));
  const std::vector<int64_t> worlds(kBuild.supports.worlds.begin(), kBuild.supports.worlds.end());
  for (int i = 0; i < kNumOps; ++i) ops.push_back(to_string(static_cast<Op>(i)));
  for (int i = 0; i < kNumErrors; ++i) {
    const std::string s = to_string(static_cast<Error>(i));
    errors.push_back(s.substr(0, s.find(':')));
  }
  return {dtypes, worlds, kBuild.memory.pack_bytes, kBuild.memory.staging_bytes, ops, errors};
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

// THE PROBE, every rank together over the process group named `group`: the round trip to each peer
// (ns, by peer, this rank's 0), then GB/s by name, `<buffer>_<mode>`: the buffer the staging
// (uncached) or a registered allocation (cached); the mode pulled from one peer (rank ^ 1) or every
// peer, pushed into every peer, or both at once (each way), the blocks split or each doing both.
// Each the median of `trials`, a probe barrier before each; `bytes` a peer's share, at most the
// staging, `traffic_iters` launches a trial, `ping_iters` round trips a trial.
ProbeWire rocm_comms_probe(fptr_t handle_ptr, const std::string& group, int64_t bytes,
                           int64_t ping_iters, int64_t traffic_iters, int64_t trials) {
  auto& h = handle_of(handle_ptr);
  TORCH_CHECK(ping_iters > 0 && traffic_iters > 0 && trials > 0 && bytes >= 16,
              "iters and trials must be positive and bytes at least a pack");
  const auto pg = resolved(group);
  if (!pg) raise(hip_comms::Error::no_such_group);
  bytes             = std::min<int64_t>(bytes, h.staging_bytes()) / 16 * 16;
  const int world   = h.world_size(), rank = h.rank();
  auto stream       = at::cuda::getCurrentCUDAStream();
  const auto on_dev = torch::TensorOptions().device(torch::kCUDA, c10::cuda::current_device());
  int device = 0, khz = 0;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, device));
  void (*barrier)(hip_comms::p2p::DevComm)                                     = nullptr;
  void (*traffic)(hip_comms::p2p::DevComm, int, int, int, int64_t, uint32_t*)  = nullptr;
  hip_comms::impl::by_world(world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    barrier          = hip_comms::probe_barrier<NG>;
    traffic          = hip_comms::link_traffic<c10::BFloat16, NG>;
  });
  const auto together = [&]() {
    barrier<<<dim3(1), dim3(64), 0, stream>>>(h.dev_comm());
  };
  const auto median = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };

  // THE ROUND TRIP to each peer: round d pairs this rank with rank ^ d, both calling together.
  std::vector<double> ping(world, 0.0);
  auto ticks = torch::empty({1}, on_dev.dtype(torch::kInt64));
  for (int d = 1; d < world; ++d) {
    const int peer = rank ^ d;
    std::vector<double> got;
    for (int64_t t = 0; t < trials; ++t) {
      together();
      const uint32_t base =
          h.take_flags(peer, hip_comms::ping_pong_flags(static_cast<int>(ping_iters)));
      hip_comms::ping_pong<<<dim3(1), dim3(64), 0, stream>>>(
          h.dev_comm(), peer, base, static_cast<int>(ping_iters),
          reinterpret_cast<uint64_t*>(ticks.data_ptr<int64_t>()));
      got.push_back(static_cast<double>(ticks.item<int64_t>()) * 1e6 / khz /
                    static_cast<double>(ping_iters));
    }
    ping[peer] = median(got);
  }

  // THE LINKS: GB/s in and out of this rank, the whole device streaming, over the staging (the
  // symmetric memory, uncached) and over an ordinary allocation every rank registers (cached).
  void* cached = nullptr;
  HIP_CHECK(hipMalloc(&cached, static_cast<size_t>(bytes)));
  HIP_CHECK(hipMemset(cached, 0, static_cast<size_t>(bytes)));
  h.register_buffer(cached, gather_over(*pg));
  auto sink = torch::empty({1}, on_dev.dtype(torch::kInt32));
  uint32_t* s = reinterpret_cast<uint32_t*>(sink.data_ptr<int32_t>());
  const dim3 block(hip_comms::kBuild.kernels.max_threads);
  const auto gbytes = [&](const void* over, hip_comms::Traffic mode, int peer,
                          int blocks = hip_comms::kTarget.compute_units, int pullers = -1) {
    const hip_comms::p2p::DevComm p = h.dev_comm(over, bytes, stream);
    const dim3 grid(blocks);
    const int how = static_cast<int>(mode);
    if (pullers < 0) pullers = blocks / 2;
    auto launch = [&]() {
      traffic<<<grid, block, 0, stream>>>(p, how, peer, pullers, bytes / 16, s);
    };
    std::vector<double> got;
    for (int64_t t = 0; t < trials; ++t) {
      together();
      launch();  // untimed: the first touch
      hipEvent_t start, stop;
      HIP_CHECK(hipEventCreate(&start));
      HIP_CHECK(hipEventCreate(&stop));
      HIP_CHECK(hipEventRecord(start, stream));
      for (int64_t i = 0; i < traffic_iters; ++i) launch();
      HIP_CHECK(hipEventRecord(stop, stream));
      HIP_CHECK(hipEventSynchronize(stop));
      float ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
      HIP_CHECK(hipEventDestroy(start));
      HIP_CHECK(hipEventDestroy(stop));
      const int peers = peer >= 0 ? 1 : world - 1;
      got.push_back(static_cast<double>(bytes) * peers * traffic_iters / (ms * 1e-3) / 1e9);
    }
    return median(got);
  };
  using hip_comms::Traffic;
  std::vector<std::string> names;
  std::vector<double> gbps;
  for (const auto& [over, where] : {std::pair<const void*, const char*>{h.staging(), "staging"},
                                    std::pair<const void*, const char*>{cached, "cached"}}) {
    for (const auto& [mode, peer, what] :
         {std::tuple<Traffic, int, const char*>{Traffic::pull, rank ^ 1, "pull_one"},
          {Traffic::pull, -1, "pull"},
          {Traffic::push, -1, "push"},
          {Traffic::split, -1, "split"},
          {Traffic::each, -1, "each"}}) {
      names.push_back(std::string(where) + "_" + what);
      gbps.push_back(gbytes(over, mode, peer));
    }
  }
  // THE GRID: how many blocks the links take before reads queue (the kernels' reduce runs on dozens).
  for (const int blocks : {16, 32, 48, 64, 96, 128}) {
    names.push_back("cached_pull_b" + std::to_string(blocks));
    gbps.push_back(gbytes(cached, Traffic::pull, -1, blocks));
    names.push_back("cached_push_b" + std::to_string(blocks));
    gbps.push_back(gbytes(cached, Traffic::push, -1, blocks));
  }
  // BOTH AT ONCE AT SANE GRIDS: `pullers` blocks pull, the rest push; GB/s each way.
  for (const auto& [pullers, pushers] : {std::pair<int, int>{32, 32}, {32, 64}, {32, 128},
                                         {48, 48}, {48, 144}, {64, 64}}) {
    names.push_back("cached_split_p" + std::to_string(pullers) + "_q" + std::to_string(pushers));
    gbps.push_back(gbytes(cached, Traffic::split, -1, pullers + pushers, pullers));
  }
  // Every rank done reading every other's before any frees its copy.
  together();
  HIP_CHECK(hipStreamSynchronize(stream));
  (void)all_gathered(*pg, std::string(1, '\0'));
  h.forget_buffer(cached);
  HIP_CHECK(hipFree(cached));
  return {ping, names, gbps};
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
                           std::optional<int64_t> quant_bits, std::optional<std::string> template_,
                           std::optional<int64_t> tile_m, std::optional<int64_t> tile_n,
                           std::optional<int64_t> tile_k, std::optional<int64_t> slice_k,
                           std::optional<int64_t> threads_per_block,
                           std::optional<int64_t> blocks_per_grid) {
  check_device_contiguous({&out, &inp});
  TORCH_CHECK(out.sizes() == inp.sizes(), "out and inp must have the same shape");
  TORCH_CHECK(out.scalar_type() == inp.scalar_type(), "out and inp must share a dtype");
  ran(hip_comms::all_reduce(
      handle_of(handle_ptr),
      {out.data_ptr(), inp.data_ptr(), inp.numel() * inp.element_size(), dtype_of(inp)},
      options_or_raise(quant_bits, template_, tile_m, tile_n, tile_k, slice_k, threads_per_block,
                       blocks_per_grid)));
}

namespace {
// rms_norm, or fused_add_rms_norm when `residual` is given (with `residual_out`).
void all_reduce_rms_norm(fptr_t handle_ptr, torch::Tensor& out, torch::Tensor* residual_out,
                         torch::Tensor& inp, const torch::Tensor* residual,
                         torch::Tensor& weight, double eps, const hip_comms::Options& o) {
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
  ran(hip_comms::all_reduce_rms_norm(
      handle_of(handle_ptr),
      {residual != nullptr, out.data_ptr(), inp.data_ptr(), weight.data_ptr(), dtype_of(inp),
       dtype_of(weight),
       inp.size(0), inp.size(1), static_cast<float>(eps),
       residual_out ? residual_out->data_ptr() : nullptr,
       residual ? residual->data_ptr() : nullptr},
      o));
}
}  // namespace

void rocm_comms_all_reduce_rms_norm(fptr_t handle_ptr, torch::Tensor& out, torch::Tensor& inp,
                                    torch::Tensor& weight, double eps,
                                    std::optional<int64_t> quant_bits,
                                    std::optional<std::string> template_,
                                    std::optional<int64_t> tile_m, std::optional<int64_t> tile_n,
                                    std::optional<int64_t> tile_k, std::optional<int64_t> slice_k,
                                    std::optional<int64_t> threads_per_block,
                                    std::optional<int64_t> blocks_per_grid) {
  all_reduce_rms_norm(handle_ptr, out, nullptr, inp, nullptr, weight, eps,
                      options_or_raise(quant_bits, template_, tile_m, tile_n, tile_k, slice_k,
                                       threads_per_block, blocks_per_grid));
}

void rocm_comms_all_reduce_add_rms_norm(
    fptr_t handle_ptr, torch::Tensor& out, torch::Tensor& residual_out, torch::Tensor& inp,
    torch::Tensor& residual, torch::Tensor& weight, double eps, std::optional<int64_t> quant_bits,
    std::optional<std::string> template_, std::optional<int64_t> tile_m,
    std::optional<int64_t> tile_n, std::optional<int64_t> tile_k, std::optional<int64_t> slice_k,
    std::optional<int64_t> threads_per_block, std::optional<int64_t> blocks_per_grid) {
  all_reduce_rms_norm(handle_ptr, out, &residual_out, inp, &residual, weight, eps,
                      options_or_raise(quant_bits, template_, tile_m, tile_n, tile_k, slice_k,
                                       threads_per_block, blocks_per_grid));
}

// With `has_prefix` the sum is added to `prefix` in place; without, the sum IS the new prefix.
void rocm_comms_all_reduce_add_attn_res_rms_norm(
    fptr_t handle_ptr, torch::Tensor& prefix, torch::Tensor& out, torch::Tensor& inp,
    torch::Tensor& blocks, torch::Tensor& norm_weight, torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& out_norm_weight, int64_t num_blocks, int64_t write_idx,
    double eps, double out_eps, bool has_prefix, std::optional<int64_t> quant_bits,
    std::optional<std::string> template_, std::optional<int64_t> tile_m,
    std::optional<int64_t> tile_n, std::optional<int64_t> tile_k, std::optional<int64_t> slice_k,
    std::optional<int64_t> threads_per_block, std::optional<int64_t> blocks_per_grid) {
  check_device_contiguous({&prefix, &out, &inp, &norm_weight, &qk_weight});
  TORCH_CHECK(inp.dim() == 2, "inp must be 2-D [tokens, hidden]; got ", inp.dim(), "-D");
  const int64_t hidden = inp.size(1);
  TORCH_CHECK(prefix.sizes() == inp.sizes() && out.sizes() == inp.sizes(),
              "prefix and out must have inp's shape");
  TORCH_CHECK(blocks.is_cuda() && blocks.dim() == 3 && blocks.size(0) == inp.size(0) &&
                  blocks.size(2) == hidden && blocks.stride(2) == 1,
              "blocks must be [tokens, sources, hidden] with a unit hidden stride");
  TORCH_CHECK(num_blocks >= 0 && num_blocks <= blocks.size(1),
              "num_blocks must be in [0, ", blocks.size(1), "]");
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
  TORCH_CHECK(
      blocks.stride(0) % lanes == 0 && blocks.stride(1) % lanes == 0 &&
          reinterpret_cast<uintptr_t>(blocks.data_ptr()) % hip_comms::kBuild.memory.pack_bytes == 0,
      "blocks must be 16-byte aligned in every row and source");
  ran(hip_comms::all_reduce_add_attn_res_rms_norm(
      handle_of(handle_ptr),
      {prefix.data_ptr(), out.data_ptr(), inp.data_ptr(), blocks.data_ptr(), blocks.stride(0),
       blocks.stride(1), norm_weight.data_ptr(), qk_weight.data_ptr(),
       out_norm_weight ? out_norm_weight->data_ptr() : nullptr, dtype_of(inp), inp.size(0), hidden,
       static_cast<int>(num_blocks), static_cast<int>(write_idx), static_cast<float>(eps),
       static_cast<float>(out_eps), has_prefix},
      options_or_raise(quant_bits, template_, tile_m, tile_n, tile_k, slice_k, threads_per_block,
                       blocks_per_grid)));
}

namespace {
// out = rms_norm(all_reduce(inp)) @ gemm_weight^T (out [rows, N], a column slice of a wider
// buffer allowed), or += with `add`; `workspace`
// holds the normed rows, inp's shape and dtype.
void all_reduce_rms_norm_gemm(fptr_t handle_ptr, bool add, torch::Tensor& out,
                              torch::Tensor& inp, torch::Tensor& norm_weight, double eps,
                              torch::Tensor& gemm_weight, torch::Tensor& workspace,
                              const hip_comms::Options& o) {
  check_device_contiguous({&inp, &norm_weight, &workspace});
  TORCH_CHECK(inp.dim() == 2, "inp must be 2-D");
  const int64_t rows = inp.size(0), hidden = inp.size(1);
  TORCH_CHECK(workspace.sizes() == inp.sizes() && workspace.scalar_type() == inp.scalar_type(),
              "workspace must be of inp's shape and dtype");
  TORCH_CHECK(gemm_weight.is_cuda() && gemm_weight.dim() == 2 && gemm_weight.size(1) == hidden &&
                  gemm_weight.stride(1) == 1 && gemm_weight.stride(0) == hidden,
              "gemm_weight must be [N, hidden] with contiguous rows");
  const int64_t n_cols = gemm_weight.size(0);
  TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.size(0) == rows && out.size(1) == n_cols &&
                  out.stride(1) == 1,
              "out must be [rows, N] with a unit column stride");
  TORCH_CHECK(norm_weight.dim() == 1 && norm_weight.numel() == hidden,
              "norm_weight must be 1-D of hidden=", hidden);
  for (const torch::Tensor* t : {&out, &norm_weight, &gemm_weight})
    TORCH_CHECK(t->scalar_type() == inp.scalar_type(), "every tensor must share inp's dtype");
  const hip_comms::GemmTailArgs a{add, out.data_ptr(), out.stride(0), inp.data_ptr(), norm_weight.data_ptr(), static_cast<float>(eps),
                                  gemm_weight.data_ptr(), n_cols, workspace.data_ptr(),
                                  dtype_of(inp), rows, hidden};
  ran(add ? hip_comms::all_reduce_rms_norm_gemm_add(handle_of(handle_ptr), a, o)
          : hip_comms::all_reduce_rms_norm_gemm(handle_of(handle_ptr), a, o));
}
}  // namespace

void rocm_comms_all_reduce_rms_norm_gemm(
    fptr_t handle_ptr, torch::Tensor& out, torch::Tensor& inp, torch::Tensor& norm_weight,
    double eps, torch::Tensor& gemm_weight, torch::Tensor& workspace,
    std::optional<int64_t> quant_bits, std::optional<std::string> template_,
    std::optional<int64_t> tile_m, std::optional<int64_t> tile_n, std::optional<int64_t> tile_k,
    std::optional<int64_t> slice_k, std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid) {
  all_reduce_rms_norm_gemm(handle_ptr, false, out, inp, norm_weight, eps, gemm_weight, workspace,
                           options_or_raise(quant_bits, template_, tile_m, tile_n, tile_k, slice_k,
                                            threads_per_block, blocks_per_grid));
}

void rocm_comms_all_reduce_rms_norm_gemm_add(
    fptr_t handle_ptr, torch::Tensor& out, torch::Tensor& inp, torch::Tensor& norm_weight,
    double eps, torch::Tensor& gemm_weight, torch::Tensor& workspace,
    std::optional<int64_t> quant_bits, std::optional<std::string> template_,
    std::optional<int64_t> tile_m, std::optional<int64_t> tile_n, std::optional<int64_t> tile_k,
    std::optional<int64_t> slice_k, std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid) {
  all_reduce_rms_norm_gemm(handle_ptr, true, out, inp, norm_weight, eps, gemm_weight, workspace,
                           options_or_raise(quant_bits, template_, tile_m, tile_n, tile_k, slice_k,
                                            threads_per_block, blocks_per_grid));
}

// out [rows, hidden] = shared + projected * rsqrt(mean(latent^2) + eps), inp's row [shared |
// projected | latent] summed over the ranks first: the widths are out's, out's again, and the rest.
void rocm_comms_all_reduce_rms_scale_add(
    fptr_t handle_ptr, torch::Tensor& out, torch::Tensor& inp, double eps,
    std::optional<int64_t> quant_bits, std::optional<std::string> template_,
    std::optional<int64_t> tile_m, std::optional<int64_t> tile_n, std::optional<int64_t> tile_k,
    std::optional<int64_t> slice_k, std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid) {
  check_device_contiguous({&out, &inp});
  TORCH_CHECK(inp.dim() == 2 && out.dim() == 2 && out.size(0) == inp.size(0),
              "inp and out must be 2-D with the same rows");
  const int64_t hidden = out.size(1);
  const int64_t latent = inp.size(1) - 2 * hidden;
  TORCH_CHECK(latent > 0, "inp's row must be wider than twice out's: [shared | projected | latent]");
  TORCH_CHECK(out.scalar_type() == inp.scalar_type(), "out must share inp's dtype");
  ran(hip_comms::all_reduce_rms_scale_add(
      handle_of(handle_ptr),
      {out.data_ptr(), inp.data_ptr(), dtype_of(inp), inp.size(0), hidden, latent,
       static_cast<float>(eps)},
      options_or_raise(quant_bits, template_, tile_m, tile_n, tile_k, slice_k, threads_per_block,
                       blocks_per_grid)));
}
