// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Our HIP collectives. Self-contained: torch + the HIP runtime, nothing from aiter's
// csrc. Built into `_rocm_C` with the other ROCm sources.
//
// THE CALLER NAMES AN OP; THIS PICKS THE KERNEL. Which kernel runs and how wide is decided
// once, in tune.cuh, from the target's facts (hardware.cuh) and the input: Python never sees
// an algorithm or a geometry. The one way to force a choice is the launch every op takes
// last, which the sweep and the tests pass and the model leaves to tune.cuh. No getenv.
//
// Compile-time vs runtime is the one distinction that shapes everything. `ngpus` and the
// dtype must be constexpr to unroll and vectorize, so they are template parameters and
// the instantiation list below is the finite menu; asking for one that was not built
// RAISES rather than falling back, because a silent substitution produces a number about
// the wrong thing.
//
// rocm_comms/ holds the layers as headers, all included here so the device code stays in
// this one translation unit and needs no -fgpu-rdc: common/ (primitives: the pack, how it is
// loaded and stored, a thread's share of a row, the sums), p2p/ (the peer layer, host and
// device, behind its one interface p2p/p2p.cuh), hardware.cuh (the target's facts and its
// calibration), launch.cuh (the kernels there are), tune.cuh (the picker), probes/ (the
// calibration kernels), then one all_reduce_pull_<shot>[_<fusion>].cuh per kernel, named as
// its Kernel is, each holding its whole algorithm. Design rules: CONTEXT.md, "Code design".

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "rocm_comms/all_reduce_pull_one_shot.cuh"
#include "rocm_comms/all_reduce_pull_one_shot_add_attn_res_rms_norm.cuh"
#include "rocm_comms/all_reduce_pull_one_shot_add_rms_norm.cuh"
#include "rocm_comms/all_reduce_pull_one_shot_rms_norm_gemm_add.cuh"
#include "rocm_comms/all_reduce_pull_two_shot.cuh"
#include "rocm_comms/all_reduce_pull_two_shot_add_attn_res_rms_norm.cuh"
#include "rocm_comms/all_reduce_pull_two_shot_add_rms_norm.cuh"
#include "rocm_comms/all_reduce_pull_two_shot_rms_norm_gemm_add.cuh"
#include "rocm_comms/p2p/p2p.cuh"
#include "rocm_comms/probes/peer_read.cuh"
#include "rocm_comms/probes/ping_pong.cuh"
#include "rocm_comms/launch.cuh"
#include "rocm_comms/tune.cuh"

namespace hip_comms {

// EVERY TUNED LAUNCH STAYS WITHIN WHAT THE KERNELS WERE BUILT FOR, for every op at the smallest
// input and a large one: a config past a capability is a compile error, not a kernel that overruns
// its signal slots or register arrays. A decline is fine.
// EVERY TUNED LAUNCH IS A KERNEL THAT FITS: a tune_<op> never declines, since a fusion that is on
// runs its fused op; the only no is admits' for an input no kernel can run.
constexpr bool fits(const Launch& l) {
  if (l.kernel == Kernel::none) return false;
  if (l.grid < 1 || l.grid > p2p::kMaxBlocks) return false;
  if (has_row_packs(l.kernel) && l.row_packs == 0) return false;
  return l.threads >= kWaveSize && l.threads <= kMaxThreads && l.threads % kWaveSize == 0;
}
constexpr bool tuned_launches_fit() {
  for (int op = 0; op <= static_cast<int>(Op::all_reduce_rms_norm_gemm_add); ++op)
    for (const Input in :
         {Input{1, 8, 2, 0, 16, 2}, Input{4096, 7168, 2, 0, 16, p2p::kMaxRanks}})
      if (!fits(tune(static_cast<Op>(op), in, kTarget, kTargetCalibration))) return false;
  return true;
}
static_assert(tuned_launches_fit(),
              "a tune_<op> in tune.cuh declines, or exceeds a kernel capability");

// A FORCED LAUNCH, passed with every call (the sweep's and the tests'; the model passes none): a
// kernel at a grid and block. `kernel` none: tune.cuh picks everything.
struct Forced {
  Kernel kernel;
  int blocks;
  int threads;
};

// WHAT A CALL ASKS: the launch it forced, and the precision it accepts on the wire.
struct Request {
  Forced forced;
  int quant_bits;
};

// The four integers every op takes last, checked: the precision, then the launch (kernel -1: not
// forced, tune.cuh picks).
Request request_of(int64_t quant_bits, int64_t kernel, int64_t blocks, int64_t threads) {
  TORCH_CHECK(quant_bits == 16 || quant_bits == 8 || quant_bits == 4,
              "quant_bits must be 16 (exact), 8 or 4");
  if (kernel < 0) return {{Kernel::none, 0, 0}, static_cast<int>(quant_bits)};
  TORCH_CHECK(kernel < kNumKernels, "hip_comms: no kernel ", kernel);
  TORCH_CHECK(blocks > 0 && blocks <= p2p::kMaxBlocks, "blocks must be in [1, ",
              p2p::kMaxBlocks, "]");
  TORCH_CHECK(threads > 0 && threads <= kMaxThreads && threads % kWaveSize == 0,
              "threads must be a multiple of ", kWaveSize, " up to ", kMaxThreads);
  return {{static_cast<Kernel>(kernel), static_cast<int>(blocks), static_cast<int>(threads)},
          static_cast<int>(quant_bits)};
}

// THE CALL as tune.cuh reads it.
Input input_of(const p2p::host::Group& group, const Request& req, int64_t rows, int64_t hidden,
               int64_t elem, int64_t cols) {
  return {rows, hidden, static_cast<int>(elem), cols, req.quant_bits, group.world_size()};
}

// What runs: tune.cuh's pick, or the forced kernel at the forced grid and block.
Launch launch_for(const Request& req, Op op, Input in) {
  const Forced& f = req.forced;
  if (f.kernel == Kernel::none) return tune(op, in, kTarget, kTargetCalibration);
  TORCH_CHECK(op_of(f.kernel) == op, "hip_comms: the forced kernel ",
              static_cast<int>(f.kernel), " is not one of op ", static_cast<int>(op), "'s");
  // A forced GEMM tail runs at the calibrated lanes a column: a launch's grid and block are forced,
  // what is inside the kernel is not.
  const int lanes =
      op == Op::all_reduce_rms_norm_gemm_add ? kTargetCalibration.gemm_lanes_per_col : 0;
  return at(f.kernel, in, f.blocks, f.threads, lanes);
}

int64_t scratch_need(const Launch& l, int64_t rows, int64_t packs, int64_t flat,
                     int world) {
  // A KERNEL'S SCRATCH is this rank's slice, row-major: the plain two-shot's packs, a fused
  // two-shot's rows, twice where it leaves two results (out and the residual or the prefix). A
  // one-shot reads the inputs and keeps nothing.
  if (!is_two_shot(l.kernel)) return 0;
  const Op op         = op_of(l.kernel);
  const int64_t slice = op == Op::all_reduce ? (flat + world - 1) / world
                                             : (rows + world - 1) / world * packs;
  const bool two      = op == Op::all_reduce_add_rms_norm ||
                   op == Op::all_reduce_add_attn_res_rms_norm;
  return slice * (two ? 2 : 1) * 16;
}

// WHY `op` over [rows, hidden] of this element size CANNOT RUN here, or empty when it can: every
// reason is a capability (a kernel that cannot take the input), never "the unfused ops would be
// faster" -- a tune_<op> never declines. The ops raise with it; `admits` is it as a yes or no, for
// vLLM's choice of all-reduce backend.
std::string why_not(const p2p::host::Group& group, const Request& req, Op op, int64_t rows,
                    int64_t hidden, int64_t elem, int64_t cols) {
  const int64_t lanes = 16 / elem;
  if (hidden % lanes != 0) return "the row is not a whole number of 16-byte packs";
  const int64_t packs = hidden / lanes;
  const Launch l      = launch_for(req, op, input_of(group, req, rows, hidden, elem, cols));
  if (has_row_packs(l.kernel) && l.row_packs == 0)
    return "the row is wider than the op's widest build holds at this block (max_row_packs)";
  // TWO-SHOT'S BLOCK IS ONE WAVE PER PEER, so anything else would leave a peer unread.
  if (l.kernel == Kernel::all_reduce_pull_two_shot &&
      l.threads % (group.world_size() * kWaveSize) != 0)
    return "a two-shot block must be one wave per peer";
  if (l.quant_bits != 16) return "no kernel quantizes yet";
  if (op == Op::all_reduce_rms_norm_gemm_add &&
      l.threads > gemm_max_threads(l.gemm_lanes_per_col))
    return "the GEMM tail's block exceeds what its LDS holds";
  if (scratch_need(l, rows, packs, rows * packs, group.world_size()) > group.scratch_bytes())
    return "its two-shot slice exceeds the scratch (raise scratch_bytes)";
  return "";
}

bool admits(const p2p::host::Group& group, const Request& req, Op op, int64_t rows,
            int64_t hidden, int64_t elem, int64_t cols) {
  return why_not(group, req, op, rows, hidden, elem, cols).empty();
}

// The launch for a call, or an error naming why it cannot run.
Launch checked_launch(const p2p::host::Group& group, const Request& req, Op op,
                      int64_t rows, int64_t hidden, int64_t elem, int64_t cols) {
  const std::string why = why_not(group, req, op, rows, hidden, elem, cols);
  TORCH_CHECK(why.empty(), "hip_comms: op ", static_cast<int>(op), " over [", rows, ", ", hidden,
              "] cannot run: ", why);
  return launch_for(req, op, input_of(group, req, rows, hidden, elem, cols));
}

#define BY_NGPUS(world, LAUNCH)                                                          \
  switch (world) {                                                                       \
    case 2: LAUNCH(2); return;                                                           \
    case 4: LAUNCH(4); return;                                                           \
    case 8: LAUNCH(8); return;                                                           \
    default: break;                                                                      \
  }

[[noreturn]] void not_built(int world) {
  throw std::runtime_error("hip_comms: world_size " + std::to_string(world) +
                           " not built. Built: 2, 4, 8.");
}

[[noreturn]] void dtype_not_built() {
  throw std::runtime_error("hip_comms: dtype not built. Built: float16, bfloat16.");
}

[[noreturn]] void not_this_ops(Kernel k) {
  throw std::runtime_error("hip_comms: kernel " + std::to_string(static_cast<int>(k)) +
                           " is not this op's");
}

// A row kernel's row packs as a template argument: f(std::integral_constant<int, k>), one case
// per kRowPacksBuilt up to the op's max_row_packs, so no build past what fits is instantiated.
template <int kMax, typename F>
void by_row_packs(int k, F&& f) {
  switch (k) {
    case 1: f(std::integral_constant<int, 1>{}); return;
    case 2: f(std::integral_constant<int, 2>{}); return;
    case 4: if constexpr (kMax >= 4) { f(std::integral_constant<int, 4>{}); return; } break;
    case 8: if constexpr (kMax >= 8) { f(std::integral_constant<int, 8>{}); return; } break;
    default: break;
  }
  TORCH_CHECK(false, "hip_comms: no row kernel built for ", k, " packs a thread");
}

// ONE CASE PER KERNEL: `CASE_PULL(kernel, args, template args...)` launches the kernel's
// function, named as its Kernel is; `CASE_PULL_ROWS` the same at the launch's row packs, which
// the template arguments name as kR. Every op's launch is one switch over its kernels. The launch
// configuration is spelled out: hipify parses `<<<...>>>` as text.
#define CASE_PULL(KERNEL, ARGS, ...)                                                     \
  case Kernel::KERNEL:                                                                   \
    KERNEL<__VA_ARGS__><<<dim3(l.grid), dim3(l.threads), 0, stream>>>(ARGS);            \
    break;
#define CASE_PULL_ROWS(KERNEL, ARGS, ...)                                                \
  case Kernel::KERNEL:                                                                   \
    by_row_packs<max_row_packs(op_of(Kernel::KERNEL))>(l.row_packs, [&](auto r) {        \
      constexpr int kR = decltype(r)::value;                                             \
      KERNEL<__VA_ARGS__>                                                                \
          <<<dim3(l.grid), dim3(l.threads), 0, stream>>>(ARGS);                          \
    });                                                                                  \
    break;

// PLAIN ALL-REDUCE over the flat buffer.
void all_reduce(p2p::host::Group& group, const Request& req, torch::Tensor& out,
                torch::Tensor& inp) {
  TORCH_CHECK(out.is_cuda() && inp.is_cuda(), "out and inp must be on device");
  TORCH_CHECK(out.is_contiguous() && inp.is_contiguous(), "out and inp must be contiguous");
  TORCH_CHECK(out.sizes() == inp.sizes(), "out and inp must have the same shape");
  TORCH_CHECK(out.scalar_type() == inp.scalar_type(), "out and inp must share a dtype");
  const Launch l =
      checked_launch(group, req, Op::all_reduce, 1, inp.numel(), inp.element_size(), 0);
  const int n    = static_cast<int>(inp.numel() * inp.element_size() / 16);
  const p2p::DevComm p = group.dev_comm(inp);
  auto stream        = at::cuda::getCurrentCUDAStream();

#define ALL_REDUCE_ARGS(T) p, out.data_ptr<T>(), n
#define LAUNCH_ALL_REDUCE(T, NG)                                                         \
  switch (l.kernel) {                                                                    \
    CASE_PULL(all_reduce_pull_one_shot, ALL_REDUCE_ARGS(T), T, NG)                       \
    CASE_PULL(all_reduce_pull_two_shot, ALL_REDUCE_ARGS(T), T, NG)                       \
    default: not_this_ops(l.kernel);                                                     \
  }
#define ALL_REDUCE_HALF(NG) LAUNCH_ALL_REDUCE(at::Half, NG)
#define ALL_REDUCE_BF16(NG) LAUNCH_ALL_REDUCE(at::BFloat16, NG)

  switch (out.scalar_type()) {
    case at::ScalarType::Half: BY_NGPUS(group.world_size(), ALL_REDUCE_HALF) break;
    case at::ScalarType::BFloat16:
      BY_NGPUS(group.world_size(), ALL_REDUCE_BF16) break;
    default: dtype_not_built();
  }
  not_built(group.world_size());
#undef ALL_REDUCE_BF16
#undef ALL_REDUCE_HALF
#undef LAUNCH_ALL_REDUCE
#undef ALL_REDUCE_ARGS
}

// FUSED: all-reduce, then vLLM's `rms_norm`, or `fused_add_rms_norm` when `residual` is
// given (and then `residual_out` too). Exact to those ops' roundings; see
// `all_reduce_pull_one_shot_add_rms_norm_body`.
void all_reduce_add_rms_norm(p2p::host::Group& group, const Request& req,
                             torch::Tensor& out, torch::Tensor* residual_out,
                             torch::Tensor& inp, const torch::Tensor* residual,
                             torch::Tensor& weight, double eps) {
  const bool add = residual != nullptr;
  TORCH_CHECK(add == (residual_out != nullptr),
              "residual and residual_out come together or not at all");
  TORCH_CHECK(out.is_cuda() && inp.is_cuda() && weight.is_cuda(),
              "every tensor must be on device");
  TORCH_CHECK(out.is_contiguous() && inp.is_contiguous() && weight.is_contiguous(),
              "every tensor must be contiguous");
  TORCH_CHECK(out.sizes() == inp.sizes(), "out must have inp's shape");
  TORCH_CHECK(out.scalar_type() == inp.scalar_type(), "out must share inp's dtype");
  // THE WEIGHT IN ITS OWN DTYPE: inp's, or fp32, the two a norm's weight is kept in. The
  // kernel rounds as the reference does for either, so nothing casts it on the way in.
  const bool fp32_weight = weight.scalar_type() == at::ScalarType::Float;
  TORCH_CHECK(fp32_weight || weight.scalar_type() == inp.scalar_type(),
              "weight must be inp's dtype or float32; got ", weight.scalar_type());
  if (add) {
    TORCH_CHECK(residual->is_cuda() && residual_out->is_cuda(),
                "every tensor must be on device");
    TORCH_CHECK(residual->is_contiguous() && residual_out->is_contiguous(),
                "every tensor must be contiguous");
    TORCH_CHECK(residual->sizes() == inp.sizes() && residual_out->sizes() == inp.sizes(),
                "residual and residual_out must have inp's shape");
    TORCH_CHECK(residual->scalar_type() == inp.scalar_type() &&
                    residual_out->scalar_type() == inp.scalar_type(),
                "every tensor must share inp's dtype");
  }
  TORCH_CHECK(inp.dim() == 2, "inp must be 2-D [tokens, hidden]; got ", inp.dim(), "-D");
  TORCH_CHECK(weight.dim() == 1 && weight.numel() == inp.size(1),
              "weight must be 1-D of hidden=", inp.size(1));
  const int lanes = 16 / static_cast<int>(inp.element_size());
  // The weight is read a pack at a time, `lanes` of its elements per load.
  const int64_t weight_pack = lanes * weight.element_size();
  TORCH_CHECK(reinterpret_cast<uintptr_t>(weight.data_ptr()) % weight_pack == 0,
              "weight must be aligned to ", weight_pack, " bytes");
  const int rows  = static_cast<int>(inp.size(0));
  const int packs = static_cast<int>(inp.size(1) / lanes);
  const Op op     = add ? Op::all_reduce_add_rms_norm : Op::all_reduce_rms_norm;
  const Launch l  = checked_launch(group, req, op, rows, inp.size(1), inp.element_size(), 0);
  const p2p::DevComm p = group.dev_comm(inp);
  auto stream        = at::cuda::getCurrentCUDAStream();
  const float feps   = static_cast<float>(eps);

#define NORM_ARGS(T, W) p, out.data_ptr<T>(), weight.data_ptr<W>(), feps, rows, packs
#define ADD_NORM_ARGS(T, W)                                                              \
  p, out.data_ptr<T>(), residual_out->data_ptr<T>(), residual->data_ptr<T>(),            \
      weight.data_ptr<W>(), feps, rows, packs
#define LAUNCH_NORM(T, W, NG)                                                            \
  switch (l.kernel) {                                                                    \
    CASE_PULL_ROWS(all_reduce_pull_one_shot_rms_norm, NORM_ARGS(T, W), T, W, NG, kR)     \
    CASE_PULL_ROWS(all_reduce_pull_two_shot_rms_norm, NORM_ARGS(T, W), T, W, NG, kR)     \
    CASE_PULL_ROWS(all_reduce_pull_one_shot_add_rms_norm, ADD_NORM_ARGS(T, W), T, W,     \
              NG, kR)                                                                    \
    CASE_PULL_ROWS(all_reduce_pull_two_shot_add_rms_norm, ADD_NORM_ARGS(T, W), T, W,     \
              NG, kR)                                                                    \
    default: not_this_ops(l.kernel);                                                     \
  }
#define NORM_HALF(NG) LAUNCH_NORM(at::Half, at::Half, NG)
#define NORM_HALF_F32(NG) LAUNCH_NORM(at::Half, float, NG)
#define NORM_BF16(NG) LAUNCH_NORM(at::BFloat16, at::BFloat16, NG)
#define NORM_BF16_F32(NG) LAUNCH_NORM(at::BFloat16, float, NG)

  const int world = group.world_size();
  switch (inp.scalar_type()) {
    case at::ScalarType::Half:
      if (fp32_weight) {
        BY_NGPUS(world, NORM_HALF_F32)
      } else {
        BY_NGPUS(world, NORM_HALF)
      }
      break;
    case at::ScalarType::BFloat16:
      if (fp32_weight) {
        BY_NGPUS(world, NORM_BF16_F32)
      } else {
        BY_NGPUS(world, NORM_BF16)
      }
      break;
    default: dtype_not_built();
  }
  not_built(world);
#undef NORM_BF16_F32
#undef NORM_BF16
#undef NORM_HALF_F32
#undef NORM_HALF
#undef LAUNCH_NORM
#undef ADD_NORM_ARGS
#undef NORM_ARGS
}

// FUSED: all-reduce, then add into the prefix, then Kimi-K3's AttnRes and its RMSNorm on each
// row (the kernels spell it out). With `has_prefix` the sum is added to `prefix` in place;
// without, the sum IS the new prefix and is written there.
void all_reduce_add_attn_res_rms_norm(p2p::host::Group& group, const Request& req,
                                      torch::Tensor& prefix, torch::Tensor& out,
                                      torch::Tensor& inp,
                                      torch::Tensor& blocks, torch::Tensor& norm_weight,
                                      torch::Tensor& qk_weight,
                                      const torch::Tensor* out_norm_weight,
                                      int64_t num_blocks, int64_t write_idx, double eps,
                                      double out_eps, bool has_prefix) {
  TORCH_CHECK(inp.dim() == 2, "inp must be 2-D [tokens, hidden]; got ", inp.dim(), "-D");
  const int64_t hidden = inp.size(1);
  for (const torch::Tensor* t : {&prefix, &out}) {
    TORCH_CHECK(t->sizes() == inp.sizes(), "prefix and out must have inp's shape");
    TORCH_CHECK(t->is_contiguous(), "prefix and out must be contiguous");
  }
  TORCH_CHECK(inp.is_contiguous(), "inp must be contiguous");
  TORCH_CHECK(blocks.dim() == 3 && blocks.size(0) == inp.size(0) &&
                  blocks.size(2) == hidden && blocks.stride(2) == 1,
              "blocks must be [tokens, sources, hidden] with a unit hidden stride");
  TORCH_CHECK(num_blocks >= 0 && num_blocks <= blocks.size(1),
              "num_blocks must be in [0, ", blocks.size(1), "]");
  TORCH_CHECK(num_blocks < hip_comms::kAttnResMaxSources,
              "num_blocks must be below ", hip_comms::kAttnResMaxSources,
              ", the most sources a row mixes");
  TORCH_CHECK(write_idx < blocks.size(1), "write_idx must be < ", blocks.size(1));
  std::vector<const torch::Tensor*> same = {&prefix, &out, &blocks, &norm_weight,
                                            &qk_weight};
  if (out_norm_weight != nullptr) same.push_back(out_norm_weight);
  for (const torch::Tensor* t : same) {
    TORCH_CHECK(t->is_cuda(), "every tensor must be on device");
    TORCH_CHECK(t->scalar_type() == inp.scalar_type(),
                "every tensor must share inp's dtype");
  }
  for (const torch::Tensor* t : {&norm_weight, &qk_weight}) {
    TORCH_CHECK(t->dim() == 1 && t->numel() == hidden && t->is_contiguous(),
                "weights must be contiguous 1-D of hidden=", hidden);
  }
  if (out_norm_weight != nullptr)
    TORCH_CHECK(out_norm_weight->dim() == 1 && out_norm_weight->numel() == hidden &&
                    out_norm_weight->is_contiguous(),
                "out_norm_weight must be contiguous 1-D of hidden=", hidden);
  const int lanes = 16 / static_cast<int>(inp.element_size());
  TORCH_CHECK(blocks.stride(0) % lanes == 0 && blocks.stride(1) % lanes == 0 &&
                  reinterpret_cast<uintptr_t>(blocks.data_ptr()) % 16 == 0,
              "blocks must be 16-byte aligned in every row and source");
  const int rows  = static_cast<int>(inp.size(0));
  const int packs = static_cast<int>(hidden / lanes);
  const Launch l  = checked_launch(group, req, Op::all_reduce_add_attn_res_rms_norm, rows, hidden,
                                   inp.element_size(), 0);
  const p2p::DevComm p = group.dev_comm(inp);
  auto stream        = at::cuda::getCurrentCUDAStream();

#define ATTN_RES_ARGS(T)                                                                 \
  p, prefix.data_ptr<T>(), blocks.data_ptr<T>(), blocks.stride(0), blocks.stride(1),     \
      norm_weight.data_ptr<T>(), qk_weight.data_ptr<T>(),                                \
      out_norm_weight ? out_norm_weight->data_ptr<T>() : nullptr, out.data_ptr<T>(),     \
      static_cast<int>(num_blocks), static_cast<int>(write_idx), static_cast<float>(eps),\
      static_cast<float>(out_eps), rows, packs
#define LAUNCH_ATTN_RES(T, NG, PRE)                                                      \
  switch (l.kernel) {                                                                    \
    CASE_PULL_ROWS(all_reduce_pull_one_shot_add_attn_res_rms_norm, ATTN_RES_ARGS(T), T, NG,\
              PRE, kR)                                                                   \
    CASE_PULL_ROWS(all_reduce_pull_two_shot_add_attn_res_rms_norm, ATTN_RES_ARGS(T), T, NG,\
              PRE, kR)                                                                   \
    default: not_this_ops(l.kernel);                                                     \
  }
#define ATTN_RES_BY_PREFIX(T, NG)                                                        \
  if (has_prefix) {                                                                      \
    LAUNCH_ATTN_RES(T, NG, true);                                                        \
  } else {                                                                               \
    LAUNCH_ATTN_RES(T, NG, false);                                                       \
  }
#define ATTN_RES_HALF(NG) ATTN_RES_BY_PREFIX(at::Half, NG)
#define ATTN_RES_BF16(NG) ATTN_RES_BY_PREFIX(at::BFloat16, NG)

  const int world = group.world_size();
  switch (inp.scalar_type()) {
    case at::ScalarType::Half: BY_NGPUS(world, ATTN_RES_HALF) break;
    case at::ScalarType::BFloat16: BY_NGPUS(world, ATTN_RES_BF16) break;
    default: dtype_not_built();
  }
  not_built(world);
#undef ATTN_RES_BF16
#undef ATTN_RES_HALF
#undef ATTN_RES_BY_PREFIX
#undef LAUNCH_ATTN_RES
#undef ATTN_RES_ARGS
}

// FUSED: all-reduce, then RMSNorm, then `out[:, col0:col0+N] += normed @ gemm_w^T` -- the
// latent MoE tail. The normed rows go in scratch, and a sync separates the norm from the
// GEMM (see the kernels).
void all_reduce_rms_norm_gemm_add(p2p::host::Group& group, const Request& req,
                                  torch::Tensor& out, int64_t out_col0, torch::Tensor& inp,
                                  torch::Tensor& norm_weight, double eps,
                                  torch::Tensor& gemm_weight, torch::Tensor& workspace) {
  TORCH_CHECK(inp.dim() == 2 && inp.is_contiguous(), "inp must be contiguous 2-D");
  const int64_t rows = inp.size(0), hidden = inp.size(1);
  // The normed rows, which the GEMM reads over and over; inp's shape and dtype.
  TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous() &&
                  workspace.sizes() == inp.sizes() &&
                  workspace.scalar_type() == inp.scalar_type(),
              "workspace must be a contiguous device tensor of inp's shape and dtype");
  TORCH_CHECK(gemm_weight.dim() == 2 && gemm_weight.size(1) == hidden &&
                  gemm_weight.stride(1) == 1 && gemm_weight.stride(0) == hidden,
              "gemm_weight must be [N, hidden] with contiguous rows");
  const int64_t n_cols = gemm_weight.size(0);
  TORCH_CHECK(out.dim() == 2 && out.size(0) == rows && out.stride(1) == 1 &&
                  out_col0 >= 0 && out_col0 + n_cols <= out.size(1),
              "out must be [rows, >= col0 + N] with a unit column stride");
  TORCH_CHECK(norm_weight.dim() == 1 && norm_weight.numel() == hidden &&
                  norm_weight.is_contiguous(),
              "norm_weight must be contiguous 1-D of hidden=", hidden);
  for (const torch::Tensor* t : {&out, &norm_weight, &gemm_weight}) {
    TORCH_CHECK(t->is_cuda(), "every tensor must be on device");
    TORCH_CHECK(t->scalar_type() == inp.scalar_type(),
                "every tensor must share inp's dtype");
  }
  const int lanes = 16 / static_cast<int>(inp.element_size());
  const int packs = static_cast<int>(hidden / lanes);
  const Launch l =
      checked_launch(group, req, Op::all_reduce_rms_norm_gemm_add, rows, hidden,
                     inp.element_size(), n_cols);
  TORCH_CHECK(l.threads % kWaveSize == 0, "the GEMM phase needs whole waves; threads ",
              l.threads);
  const p2p::DevComm p = group.dev_comm(inp);
  auto stream        = at::cuda::getCurrentCUDAStream();
  // EVERY BLOCK, not one per row: the GEMM phase spreads the columns over the whole grid.

#define GEMM_ADD_ARGS(T)                                                                 \
  p, norm_weight.data_ptr<T>(), static_cast<float>(eps), gemm_weight.data_ptr<T>(),      \
      static_cast<int>(n_cols), out.data_ptr<T>(), out.stride(0),                        \
      static_cast<int>(out_col0), workspace.data_ptr<T>(), static_cast<int>(rows), packs
#define GEMM_ADD_SHOT(T, NG, LPC)                                                        \
  switch (l.kernel) {                                                                    \
    CASE_PULL_ROWS(all_reduce_pull_one_shot_rms_norm_gemm_add, GEMM_ADD_ARGS(T), T, NG,  \
              LPC, kR)                                                                   \
    CASE_PULL_ROWS(all_reduce_pull_two_shot_rms_norm_gemm_add, GEMM_ADD_ARGS(T), T, NG,  \
              LPC, kR)                                                                   \
    default: not_this_ops(l.kernel);                                                     \
  }
#define LAUNCH_GEMM_ADD(T, NG)                                                           \
  switch (l.gemm_lanes_per_col) {                                                        \
    case 1: GEMM_ADD_SHOT(T, NG, 1); break;                                              \
    case 2: GEMM_ADD_SHOT(T, NG, 2); break;                                              \
    case 4: GEMM_ADD_SHOT(T, NG, 4); break;                                              \
    case 8: GEMM_ADD_SHOT(T, NG, 8); break;                                              \
    default:                                                                             \
      TORCH_CHECK(false, "hip_comms: no GEMM lanes per column ", l.gemm_lanes_per_col);  \
  }
#define GEMM_ADD_HALF(NG) LAUNCH_GEMM_ADD(at::Half, NG)
#define GEMM_ADD_BF16(NG) LAUNCH_GEMM_ADD(at::BFloat16, NG)

  const int world = group.world_size();
  switch (inp.scalar_type()) {
    case at::ScalarType::Half: BY_NGPUS(world, GEMM_ADD_HALF) break;
    case at::ScalarType::BFloat16: BY_NGPUS(world, GEMM_ADD_BF16) break;
    default: dtype_not_built();
  }
  not_built(world);
#undef GEMM_ADD_BF16
#undef GEMM_ADD_HALF
#undef LAUNCH_GEMM_ADD
#undef GEMM_ADD_SHOT
#undef GEMM_ADD_ARGS
}

#undef BY_NGPUS
#undef CASE_PULL

}  // namespace hip_comms

// =================================================================================
// THE TORCH OP BOUNDARY. `p2p::host::Group` is a stateful C++ object and a torch op is a
// free function over schema types, so the object crosses as an opaque handle -- the same
// `fptr_t = int64_t` vLLM's custom all-reduce and quick-reduce use. IPC handles cross as
// `int[]` for the same reason they do there: a schema has no bytes type.
// =================================================================================

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

// EVERY OP TAKES THE SAME FOUR INTEGERS LAST (see `hip_comms::Request`): quant_bits, the
// precision the caller accepts (16: exact, what the model passes), then its launch, kernel,
// launch_blocks and launch_threads (-1 and zeros: tune.cuh's, what the model passes).

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
hip_comms::p2p::host::Group& comms_of(fptr_t comms) {
  return *reinterpret_cast<hip_comms::p2p::host::Group*>(comms);
}
}  // namespace

int64_t rocm_comms_alloc(int64_t scratch_bytes, int64_t staging_bytes) {
  return static_cast<int64_t>(hip_comms::p2p::host::alloc_memory(scratch_bytes, staging_bytes));
}

fptr_t rocm_comms_init(int64_t rank, int64_t world_size, int64_t self_memory,
                       const std::vector<std::vector<int64_t>>& signal_handles,
                       const std::vector<int64_t>& signal_offsets, int64_t max_buffers,
                       int64_t scratch_bytes, int64_t staging_bytes, double sync_timeout_s) {
  auto* comms = new hip_comms::p2p::host::Group(
      static_cast<int>(rank), static_cast<int>(world_size),
      static_cast<uintptr_t>(self_memory), bytes_of(signal_handles), signal_offsets, max_buffers,
      scratch_bytes, staging_bytes, sync_timeout_s);
  return reinterpret_cast<fptr_t>(comms);
}

// The staging as a byte tensor, a view the Group owns: Python copies an eager input into it.
torch::Tensor rocm_comms_staging(fptr_t comms) {
  auto& group = comms_of(comms);
  return torch::from_blob(group.staging(), {group.staging_bytes()},
                          torch::TensorOptions().dtype(torch::kUInt8).device(
                              torch::kCUDA, c10::cuda::current_device()));
}


bool rocm_comms_admits(fptr_t comms, int64_t op, int64_t rows, int64_t hidden,
                       int64_t element_size, int64_t cols, int64_t quant_bits, int64_t kernel,
                       int64_t launch_blocks, int64_t launch_threads) {
  constexpr auto kGemmTail = static_cast<int64_t>(hip_comms::Op::all_reduce_rms_norm_gemm_add);
  TORCH_CHECK(op >= 0 && op <= kGemmTail, "hip_comms: no op ", op);
  TORCH_CHECK(element_size == 2, "hip_comms: only 2-byte dtypes are built");
  TORCH_CHECK(cols >= 0 && (cols > 0) == (op == kGemmTail),
              "hip_comms: cols is the GEMM tail's output columns, and only it has them");
  const auto req = hip_comms::request_of(quant_bits, kernel, launch_blocks, launch_threads);
  return hip_comms::admits(comms_of(comms), req, static_cast<hip_comms::Op>(op), rows,
                           hidden, element_size, cols);
}

void rocm_comms_dispose(fptr_t comms) { delete &comms_of(comms); }

// GB/S INTO THIS RANK reading `bytes` of every peer's staging (`peer` -1) or one peer's, `iters`
// times: every rank calls it together, as an all-reduce reads.
double rocm_comms_peer_read(fptr_t comms, int64_t peer, int64_t bytes, int64_t iters) {
  auto& group = comms_of(comms);
  TORCH_CHECK(peer == -1 || (peer >= 0 && peer < group.world_size() && peer != group.rank()),
              "peer must be another rank, or -1 for every other rank");
  TORCH_CHECK(iters > 0 && bytes >= 16, "iters must be positive and bytes at least a pack");
  bytes = std::min<int64_t>(bytes, group.staging_bytes()) / 16 * 16;
  auto staging = torch::from_blob(group.staging(), {bytes},
                                  torch::TensorOptions().dtype(torch::kUInt8).device(
                                      torch::kCUDA, c10::cuda::current_device()));
  const hip_comms::p2p::DevComm p = group.dev_comm(staging);
  auto sink   = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt32).device(
                                     torch::kCUDA, c10::cuda::current_device()));
  auto stream = at::cuda::getCurrentCUDAStream();
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
double rocm_comms_ping_pong(fptr_t comms, int64_t peer, int64_t iters) {
  auto& group = comms_of(comms);
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


std::vector<int64_t> rocm_comms_pending_graph_buffers(fptr_t comms) {
  auto pending = comms_of(comms).pending_graph_buffers();
  return std::vector<int64_t>(pending.begin(), pending.end());
}

// ONE ENTRY PER PENDING BUFFER, each the WORLD'S handles for it laid end to end: a
// schema nests two deep and this needs three (buffer, rank, byte), so the innermost
// level is split back out here by the handle size, which is fixed.
void rocm_comms_register_graph_buffers(
    fptr_t comms, const std::vector<std::vector<int64_t>>& handles,
    const std::vector<std::vector<int64_t>>& offsets) {
  const size_t stride = sizeof(hip_comms::p2p::host::Handle);
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
  comms_of(comms).register_graph_buffers(bytes, offsets);
}


void rocm_comms_all_reduce(fptr_t comms, torch::Tensor& out, torch::Tensor& inp,
                           int64_t quant_bits, int64_t kernel, int64_t launch_blocks,
                           int64_t launch_threads) {
  const auto req = hip_comms::request_of(quant_bits, kernel, launch_blocks, launch_threads);
  hip_comms::all_reduce(comms_of(comms), req, out, inp);
}

void rocm_comms_all_reduce_rms_norm(fptr_t comms, torch::Tensor& out, torch::Tensor& inp,
                                    torch::Tensor& weight, double eps, int64_t quant_bits,
                                    int64_t kernel, int64_t launch_blocks,
                                    int64_t launch_threads) {
  const auto req = hip_comms::request_of(quant_bits, kernel, launch_blocks, launch_threads);
  hip_comms::all_reduce_add_rms_norm(comms_of(comms), req, out, nullptr, inp, nullptr,
                                     weight, eps);
}

void rocm_comms_all_reduce_add_rms_norm(fptr_t comms, torch::Tensor& out,
                                        torch::Tensor& residual_out, torch::Tensor& inp,
                                        torch::Tensor& residual, torch::Tensor& weight,
                                        double eps, int64_t quant_bits, int64_t kernel,
                                        int64_t launch_blocks, int64_t launch_threads) {
  const auto req = hip_comms::request_of(quant_bits, kernel, launch_blocks, launch_threads);
  hip_comms::all_reduce_add_rms_norm(comms_of(comms), req, out, &residual_out, inp,
                                     &residual, weight, eps);
}

void rocm_comms_all_reduce_add_attn_res_rms_norm(
    fptr_t comms, torch::Tensor& prefix, torch::Tensor& out, torch::Tensor& inp,
    torch::Tensor& blocks, torch::Tensor& norm_weight, torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& out_norm_weight, int64_t num_blocks,
    int64_t write_idx, double eps, double out_eps, bool has_prefix, int64_t quant_bits,
    int64_t kernel, int64_t launch_blocks, int64_t launch_threads) {
  const auto req = hip_comms::request_of(quant_bits, kernel, launch_blocks, launch_threads);
  hip_comms::all_reduce_add_attn_res_rms_norm(
      comms_of(comms), req, prefix, out, inp, blocks, norm_weight, qk_weight,
      out_norm_weight ? &*out_norm_weight : nullptr, num_blocks, write_idx, eps, out_eps,
      has_prefix);
}

void rocm_comms_all_reduce_rms_norm_gemm_add(
    fptr_t comms, torch::Tensor& out, int64_t out_col0, torch::Tensor& inp,
    torch::Tensor& norm_weight, double eps, torch::Tensor& gemm_weight,
    torch::Tensor& workspace, int64_t quant_bits, int64_t kernel, int64_t launch_blocks,
    int64_t launch_threads) {
  const auto req = hip_comms::request_of(quant_bits, kernel, launch_blocks, launch_threads);
  hip_comms::all_reduce_rms_norm_gemm_add(comms_of(comms), req, out, out_col0, inp,
                                          norm_weight, eps, gemm_weight, workspace);
}

std::tuple<std::vector<int64_t>, int64_t> rocm_comms_handle_and_offset(int64_t ptr) {
  auto [handle, offset] =
      hip_comms::p2p::host::handle_and_offset(static_cast<uintptr_t>(ptr));
  std::vector<int64_t> bytes(handle.begin(), handle.end());
  return std::make_tuple(bytes, offset);
}

