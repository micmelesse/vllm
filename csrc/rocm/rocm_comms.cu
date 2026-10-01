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

#include <algorithm>
#include <cstdint>
#include <optional>
#include <variant>
#include <string>
#include <vector>

#include "rocm_comms/probes/peer_read.cuh"
#include "rocm_comms/probes/ping_pong.cuh"
#include "rocm_comms/rocm_comms.cuh"

// =================================================================================
// THE TORCH OP BOUNDARY. `p2p::host::Group` is a stateful C++ object and a torch op is a
// free function over schema types, so the object crosses as an opaque handle -- the same
// `fptr_t = int64_t` vLLM's custom all-reduce and quick-reduce use. IPC handles cross as
// `int[]` for the same reason they do there: a schema has no bytes type.
// =================================================================================

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

// EVERY OP TAKES THE SAME FOUR INTEGERS LAST, its Options: quant_bits, the precision the caller
// accepts (16: exact, what the model passes), then a forced template and its launch, kernel,
// launch_blocks and launch_threads (-1 and zeros: select's, what the model passes).

namespace {
std::string bytes_of(const std::vector<int64_t>& xs) {
  std::string out;
  out.reserve(xs.size());
  for (int64_t x : xs) out.push_back(static_cast<char>(x));
  return out;
}

std::vector<std::string> bytes_of(const std::vector<std::vector<int64_t>>& xss) {
  std::vector<std::string> out;
  out.reserve(xss.size());
  for (const auto& xs : xss) out.push_back(bytes_of(xs));
  return out;
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

hip_comms::Options options_of(int64_t quant_bits, int64_t kernel, int64_t blocks,
                              int64_t threads) {
  using hip_comms::Template;
  TORCH_CHECK(quant_bits == 16 || quant_bits == 8 || quant_bits == 4,
              "quant_bits must be 16 (exact), 8 or 4");
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (kernel < 0) return {static_cast<int>(quant_bits), std::nullopt, stream};
  TORCH_CHECK(kernel < hip_comms::kNumTemplates, "hip_comms: no template ", kernel);
  TORCH_CHECK(blocks > 0 && blocks <= hip_comms::p2p::kMaxBlocks, "blocks must be in [1, ",
              hip_comms::p2p::kMaxBlocks, "]");
  TORCH_CHECK(threads > 0 && threads <= hip_comms::kMaxThreads &&
                  threads % hip_comms::kWaveSize == 0,
              "threads must be a multiple of ", hip_comms::kWaveSize, " up to ",
              hip_comms::kMaxThreads);
  return {static_cast<int>(quant_bits),
          hip_comms::Forced{static_cast<Template>(kernel), static_cast<int>(blocks),
                            static_cast<int>(threads)},
          stream};
}

void check_device_contiguous(std::initializer_list<const torch::Tensor*> ts) {
  for (const torch::Tensor* t : ts) {
    TORCH_CHECK(t->is_cuda(), "every tensor must be on device");
    TORCH_CHECK(t->is_contiguous(), "every tensor must be contiguous");
  }
}
}  // namespace

// AN OP'S ERROR RAISED, at the torch boundary: a torch op can only return or raise.
static void ran(const std::variant<hip_comms::Kernel, hip_comms::Error>& result) {
  if (const hip_comms::Error* e = std::get_if<hip_comms::Error>(&result))
    TORCH_CHECK(false, "hip_comms: ", hip_comms::to_string(*e));
}

int64_t rocm_comms_alloc(int64_t scratch_bytes) {
  return static_cast<int64_t>(
      hip_comms::p2p::host::alloc_memory(scratch_bytes, hip_comms::kStagingBytes));
}

fptr_t rocm_comms_init(int64_t rank, int64_t world_size, int64_t self_memory,
                       const std::vector<std::vector<int64_t>>& signal_handles,
                       const std::vector<int64_t>& signal_offsets, int64_t max_buffers,
                       int64_t scratch_bytes, double sync_timeout_s) {
  auto* handle = new hip_comms::Handle(
      static_cast<int>(rank), static_cast<int>(world_size),
      static_cast<uintptr_t>(self_memory), bytes_of(signal_handles), signal_offsets, max_buffers,
      scratch_bytes, hip_comms::kStagingBytes, sync_timeout_s);
  return reinterpret_cast<fptr_t>(handle);
}



// THE FIRST ERROR OUR KERNELS MEET ON THIS CALL, as its number, or none: every rule is here or in
// `hip_comms::check`, and Python only passes the call's facts. `cols` is an output's columns, for
// the ops that have one.
std::optional<int64_t> rocm_comms_check(fptr_t handle_ptr, int64_t op,
                                        const std::vector<int64_t>& shape, at::ScalarType dtype,
                                        bool contiguous, std::optional<int64_t> cols,
                                        int64_t quant_bits, int64_t kernel, int64_t launch_blocks,
                                        int64_t launch_threads) {
  using hip_comms::Error;
  constexpr auto kLastOp = static_cast<int64_t>(hip_comms::Op::all_reduce_rms_scale_add);
  TORCH_CHECK(op >= 0 && op <= kLastOp, "hip_comms: no op ", op);
  const auto which    = static_cast<hip_comms::Op>(op);
  const bool has_cols = hip_comms::gemms(which) || which == hip_comms::Op::all_reduce_rms_scale_add;
  TORCH_CHECK(has_cols || !cols, "hip_comms: only an op with an output's columns takes cols");
  const auto error = [](Error e) { return std::optional<int64_t>(static_cast<int64_t>(e)); };
  if (!contiguous) return error(Error::not_contiguous);
  hip_comms::DType d;
  if (dtype == at::ScalarType::Half) d = hip_comms::DType::f16;
  else if (dtype == at::ScalarType::BFloat16) d = hip_comms::DType::bf16;
  else if (dtype == at::ScalarType::Float) d = hip_comms::DType::f32;
  else return error(Error::dtype_not_built);
  int64_t numel = 1;
  for (const int64_t n : shape) numel *= n;
  if (which != hip_comms::Op::all_reduce && shape.size() != 2) return error(Error::not_two_d);
  if (has_cols && !cols) return error(Error::output_not_two_d);
  // THE CALL, WITHOUT ITS TENSORS: what select reads of it, its pointers null. A norm's weight is
  // taken in the call's dtype and AttnRes without a prefix: neither changes whether a call runs.
  const int64_t rows  = which == hip_comms::Op::all_reduce ? 1 : shape[0];
  const int64_t width = which == hip_comms::Op::all_reduce ? numel : shape[1];
  auto& h             = handle_of(handle_ptr);
  const auto o        = options_of(quant_bits, kernel, launch_blocks, launch_threads);
  const float eps     = 0.f;
  const auto of = [&](const auto& args) {
    const std::variant<hip_comms::Kernel, Error> p = hip_comms::plan(h, args, o);
    const Error* e = std::get_if<Error>(&p);
    return e ? error(*e) : std::nullopt;
  };
  switch (which) {
    case hip_comms::Op::all_reduce:
      return of(hip_comms::AllReduceArgs{nullptr, nullptr, numel * hip_comms::elem_bytes(d), d});
    case hip_comms::Op::all_reduce_rms_norm:
    case hip_comms::Op::all_reduce_add_rms_norm:
      return of(hip_comms::NormArgs{which == hip_comms::Op::all_reduce_add_rms_norm, nullptr,
                                    nullptr, nullptr, d, d, rows, width, eps, nullptr, nullptr});
    case hip_comms::Op::all_reduce_add_attn_res_rms_norm:
      return of(hip_comms::AttnResArgs{nullptr, nullptr, nullptr, nullptr, 0, 0, nullptr, nullptr,
                                       nullptr, d, rows, width, 0, -1, eps, eps, false});
    case hip_comms::Op::all_reduce_rms_norm_gemm:
    case hip_comms::Op::all_reduce_rms_norm_gemm_add:
      return of(hip_comms::GemmTailArgs{which == hip_comms::Op::all_reduce_rms_norm_gemm_add,
                                        nullptr, 0, 0, nullptr, nullptr, eps, nullptr, *cols,
                                        nullptr, d, rows, width});
    // The input's row is [shared | projected | latent], and `cols` the output's width: the
    // latent is what is left.
    case hip_comms::Op::all_reduce_rms_scale_add:
      if (width - 2 * *cols < 1) return error(Error::row_not_wider_than_output);
      return of(hip_comms::ScaleAddArgs{nullptr, nullptr, d, rows, *cols, width - 2 * *cols, eps});
  }
  return error(Error::no_such_op);
}

// THE ERROR NAMES, by number: Python's mirror (rocm_comms.Error) is tested against them.
std::vector<std::string> rocm_comms_error_names() {
  std::vector<std::string> names;
  for (int i = 0; i < hip_comms::kNumErrors; ++i) {
    const std::string s = hip_comms::to_string(static_cast<hip_comms::Error>(i));
    names.push_back(s.substr(0, s.find(':')));
  }
  return names;
}

void rocm_comms_dispose(fptr_t handle_ptr) { delete &handle_of(handle_ptr); }

// GB/S INTO THIS RANK reading `bytes` of every peer's staging (`peer` -1) or one peer's, `iters`
// times: every rank calls it together, as an all-reduce reads.
double rocm_comms_peer_read(fptr_t handle_ptr, int64_t peer, int64_t bytes, int64_t iters) {
  auto& group = handle_of(handle_ptr);
  TORCH_CHECK(peer == -1 || (peer >= 0 && peer < group.world_size() && peer != group.rank()),
              "peer must be another rank, or -1 for every other rank");
  TORCH_CHECK(iters > 0 && bytes >= 16, "iters must be positive and bytes at least a pack");
  bytes = std::min<int64_t>(bytes, group.staging_bytes()) / 16 * 16;
  auto stream = at::cuda::getCurrentCUDAStream();
  const hip_comms::p2p::DevComm p = group.dev_comm(group.staging(), bytes, stream);
  auto sink = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt32).device(
                                   torch::kCUDA, c10::cuda::current_device()));
  // THE WHOLE DEVICE READING, as a large all-reduce's grid would.
  void (*kernel)(hip_comms::p2p::DevComm, int, int64_t, uint32_t*) = nullptr;
  switch (group.world_size()) {
    case 2: kernel = hip_comms::peer_read<c10::BFloat16, 2>; break;
    case 4: kernel = hip_comms::peer_read<c10::BFloat16, 4>; break;
    case 8: kernel = hip_comms::peer_read<c10::BFloat16, 8>; break;
    default: TORCH_CHECK(false, "world size must be 2, 4 or 8");
  }
  const dim3 grid(hip_comms::kTarget.compute_units), block(hip_comms::kMaxThreads);
  uint32_t* s = reinterpret_cast<uint32_t*>(sink.data_ptr<int32_t>());
  const int who = static_cast<int>(peer);
  auto launch   = [&]() { kernel<<<grid, block, 0, stream>>>(p, who, bytes / 16, s); };
  launch();  // untimed: the first touch
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start, stream));
  for (int64_t i = 0; i < iters; ++i) launch();
  HIP_CHECK(hipEventRecord(stop, stream));
  HIP_CHECK(hipEventSynchronize(stop));
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  const int64_t read = peer >= 0 ? 1 : group.world_size() - 1;
  return static_cast<double>(bytes) * read * iters / (ms * 1e-3) / 1e9;
}

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

// NANOSECONDS PER ROUND TRIP to `peer`, over `iters`: both ranks of the pair call it together.
double rocm_comms_ping_pong(fptr_t handle_ptr, int64_t peer, int64_t iters) {
  auto& group = handle_of(handle_ptr);
  TORCH_CHECK(peer >= 0 && peer < group.world_size() && peer != group.rank(),
              "peer must be another rank");
  TORCH_CHECK(iters > 0, "iters must be positive");
  auto ticks  = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt64).device(
                                     torch::kCUDA, c10::cuda::current_device()));
  auto stream = at::cuda::getCurrentCUDAStream();
  const uint32_t base =
      group.take_flags(static_cast<int>(peer), hip_comms::ping_pong_flags(static_cast<int>(iters)));
  hip_comms::ping_pong<<<dim3(1), dim3(64), 0, stream>>>(
      group.dev_comm(), static_cast<int>(peer), base, static_cast<int>(iters),
      reinterpret_cast<uint64_t*>(ticks.data_ptr<int64_t>()));
  const double t = static_cast<double>(ticks.item<int64_t>());
  int device = 0, khz = 0;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, device));
  return t * 1e6 / khz / static_cast<double>(iters);
}


std::vector<int64_t> rocm_comms_pending_graph_buffers(fptr_t handle_ptr) {
  auto pending = handle_of(handle_ptr).pending_graph_buffers();
  return std::vector<int64_t>(pending.begin(), pending.end());
}

// ONE ENTRY PER PENDING BUFFER, each the WORLD'S handles for it laid end to end: a
// schema nests two deep and this needs three (buffer, rank, byte), so the innermost
// level is split back out here by the handle size, which is fixed.
void rocm_comms_register_graph_buffers(
    fptr_t handle_ptr, const std::vector<std::vector<int64_t>>& handles,
    const std::vector<std::vector<int64_t>>& offsets) {
  const size_t stride = sizeof(hip_comms::p2p::host::IpcHandle);
  std::vector<std::vector<std::string>> bytes;
  bytes.reserve(handles.size());
  for (const auto& joined : handles) {
    TORCH_CHECK(joined.size() % stride == 0,
                "rocm_comms: ", joined.size(),
                " handle bytes is not a whole number of ", stride, "-byte handles");
    std::string all = bytes_of(joined);
    std::vector<std::string> per_rank;
    per_rank.reserve(all.size() / stride);
    for (size_t at = 0; at < all.size(); at += stride)
      per_rank.push_back(all.substr(at, stride));
    bytes.push_back(std::move(per_rank));
  }
  handle_of(handle_ptr).register_graph_buffers(bytes, offsets);
}


void rocm_comms_all_reduce(fptr_t handle_ptr, torch::Tensor& out, torch::Tensor& inp,
                           int64_t quant_bits, int64_t kernel, int64_t launch_blocks,
                           int64_t launch_threads) {
  check_device_contiguous({&out, &inp});
  TORCH_CHECK(out.sizes() == inp.sizes(), "out and inp must have the same shape");
  TORCH_CHECK(out.scalar_type() == inp.scalar_type(), "out and inp must share a dtype");
  ran(hip_comms::all_reduce(
      handle_of(handle_ptr),
      {out.data_ptr(), inp.data_ptr(), inp.numel() * inp.element_size(), dtype_of(inp)},
      options_of(quant_bits, kernel, launch_blocks, launch_threads)));
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
      hip_comms::kPackBytes / inp.element_size() * weight.element_size();
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
                                    torch::Tensor& weight, double eps, int64_t quant_bits,
                                    int64_t kernel, int64_t launch_blocks,
                                    int64_t launch_threads) {
  all_reduce_rms_norm(handle_ptr, out, nullptr, inp, nullptr, weight, eps,
                      options_of(quant_bits, kernel, launch_blocks, launch_threads));
}

void rocm_comms_all_reduce_add_rms_norm(fptr_t handle_ptr, torch::Tensor& out,
                                        torch::Tensor& residual_out, torch::Tensor& inp,
                                        torch::Tensor& residual, torch::Tensor& weight,
                                        double eps, int64_t quant_bits, int64_t kernel,
                                        int64_t launch_blocks, int64_t launch_threads) {
  all_reduce_rms_norm(handle_ptr, out, &residual_out, inp, &residual, weight, eps,
                      options_of(quant_bits, kernel, launch_blocks, launch_threads));
}

// With `has_prefix` the sum is added to `prefix` in place; without, the sum IS the new prefix.
void rocm_comms_all_reduce_add_attn_res_rms_norm(
    fptr_t handle_ptr, torch::Tensor& prefix, torch::Tensor& out, torch::Tensor& inp,
    torch::Tensor& blocks, torch::Tensor& norm_weight, torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& out_norm_weight, int64_t num_blocks,
    int64_t write_idx, double eps, double out_eps, bool has_prefix, int64_t quant_bits,
    int64_t kernel, int64_t launch_blocks, int64_t launch_threads) {
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
  const int64_t lanes = hip_comms::kPackBytes / inp.element_size();
  TORCH_CHECK(blocks.stride(0) % lanes == 0 && blocks.stride(1) % lanes == 0 &&
                  reinterpret_cast<uintptr_t>(blocks.data_ptr()) % hip_comms::kPackBytes == 0,
              "blocks must be 16-byte aligned in every row and source");
  ran(hip_comms::all_reduce_add_attn_res_rms_norm(
      handle_of(handle_ptr),
      {prefix.data_ptr(), out.data_ptr(), inp.data_ptr(), blocks.data_ptr(), blocks.stride(0),
       blocks.stride(1), norm_weight.data_ptr(), qk_weight.data_ptr(),
       out_norm_weight ? out_norm_weight->data_ptr() : nullptr, dtype_of(inp), inp.size(0),
       hidden, static_cast<int>(num_blocks), static_cast<int>(write_idx),
       static_cast<float>(eps), static_cast<float>(out_eps), has_prefix},
      options_of(quant_bits, kernel, launch_blocks, launch_threads)));
}

namespace {
// out[:, col0:col0+N] = rms_norm(all_reduce(inp)) @ gemm_weight^T, or += with `add`; `workspace`
// holds the normed rows, inp's shape and dtype.
void all_reduce_rms_norm_gemm(fptr_t handle_ptr, bool add, torch::Tensor& out, int64_t out_col0,
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
  TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.size(0) == rows && out.stride(1) == 1 &&
                  out_col0 >= 0 && out_col0 + n_cols <= out.size(1),
              "out must be [rows, >= col0 + N] with a unit column stride");
  TORCH_CHECK(norm_weight.dim() == 1 && norm_weight.numel() == hidden,
              "norm_weight must be 1-D of hidden=", hidden);
  for (const torch::Tensor* t : {&out, &norm_weight, &gemm_weight})
    TORCH_CHECK(t->scalar_type() == inp.scalar_type(), "every tensor must share inp's dtype");
  const hip_comms::GemmTailArgs a{add, out.data_ptr(), out.stride(0), static_cast<int>(out_col0),
                                  inp.data_ptr(), norm_weight.data_ptr(), static_cast<float>(eps),
                                  gemm_weight.data_ptr(), n_cols, workspace.data_ptr(),
                                  dtype_of(inp), rows, hidden};
  ran(add ? hip_comms::all_reduce_rms_norm_gemm_add(handle_of(handle_ptr), a, o)
          : hip_comms::all_reduce_rms_norm_gemm(handle_of(handle_ptr), a, o));
}
}  // namespace

void rocm_comms_all_reduce_rms_norm_gemm(
    fptr_t handle_ptr, torch::Tensor& out, int64_t out_col0, torch::Tensor& inp,
    torch::Tensor& norm_weight, double eps, torch::Tensor& gemm_weight,
    torch::Tensor& workspace, int64_t quant_bits, int64_t kernel, int64_t launch_blocks,
    int64_t launch_threads) {
  all_reduce_rms_norm_gemm(handle_ptr, false, out, out_col0, inp, norm_weight, eps, gemm_weight,
                           workspace, options_of(quant_bits, kernel, launch_blocks, launch_threads));
}

void rocm_comms_all_reduce_rms_norm_gemm_add(
    fptr_t handle_ptr, torch::Tensor& out, int64_t out_col0, torch::Tensor& inp,
    torch::Tensor& norm_weight, double eps, torch::Tensor& gemm_weight,
    torch::Tensor& workspace, int64_t quant_bits, int64_t kernel, int64_t launch_blocks,
    int64_t launch_threads) {
  all_reduce_rms_norm_gemm(handle_ptr, true, out, out_col0, inp, norm_weight, eps, gemm_weight,
                           workspace, options_of(quant_bits, kernel, launch_blocks, launch_threads));
}

// out [rows, hidden] = shared + projected * rsqrt(mean(latent^2) + eps), inp's row [shared |
// projected | latent] summed over the ranks first: the widths are out's, out's again, and the rest.
void rocm_comms_all_reduce_rms_scale_add(fptr_t handle_ptr, torch::Tensor& out,
                                         torch::Tensor& inp, double eps, int64_t quant_bits,
                                         int64_t kernel, int64_t launch_blocks,
                                         int64_t launch_threads) {
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
      options_of(quant_bits, kernel, launch_blocks, launch_threads)));
}

std::tuple<std::vector<int64_t>, int64_t> rocm_comms_handle_and_offset(int64_t ptr) {
  auto [handle, offset] =
      hip_comms::p2p::host::handle_and_offset(static_cast<uintptr_t>(ptr));
  std::vector<int64_t> bytes(handle.begin(), handle.end());
  return std::make_tuple(bytes, offset);
}

