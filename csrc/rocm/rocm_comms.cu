// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Our HIP collectives. Self-contained: torch + the HIP runtime, nothing from aiter's
// csrc. Built into `_rocm_C` with the other ROCm sources.
//
// THE CALLER NAMES AN OP; THIS PICKS THE KERNEL. Which kernel runs and how wide is decided
// once, in launch.cuh, for gfx950: Python never sees an algorithm or a geometry. The
// one way to force a choice is `rocm_comms_set_launch_override`, which the sweep uses.
// No getenv.
//
// Compile-time vs runtime is the one distinction that shapes everything. `ngpus` and the
// dtype must be constexpr to unroll and vectorize, so they are template parameters and
// the instantiation list below is the finite menu; asking for one that was not built
// RAISES rather than falling back, because a silent substitution produces a number about
// the wrong thing.
//
// rocm_comms/ holds the layers as headers, all included here so the device code stays in
// this one translation unit and needs no -fgpu-rdc: ipc.cuh (peer memory and sync),
// launch.cuh (the picker), utils.cuh (what the kernels share), then one allreduce_*.cuh
// per kernel.

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rocm_comms/allreduce_one_shot.cuh"
#include "rocm_comms/allreduce_one_shot_add_attn_res_rms_norm.cuh"
#include "rocm_comms/allreduce_one_shot_add_rms_norm.cuh"
#include "rocm_comms/allreduce_one_shot_rms_norm_gemm_add.cuh"
#include "rocm_comms/allreduce_two_shot.cuh"
#include "rocm_comms/allreduce_two_shot_add_attn_res_rms_norm.cuh"
#include "rocm_comms/allreduce_two_shot_add_rms_norm.cuh"
#include "rocm_comms/allreduce_two_shot_rms_norm_gemm_add.cuh"
#include "rocm_comms/ipc.cuh"
#include "rocm_comms/launch.cuh"

namespace hip_comms {

// What Python holds: the peers, and the launch the sweep forces, if any.
struct Comms {
  template <typename... A>
  explicit Comms(A&&... a) : group(std::forward<A>(a)...) {}
  ipc::Group group;
  Launch forced{Kernel::none, 0, 0};
};

Launch launch_for(const Comms& comms, Op op, int64_t rows, int64_t bytes) {
  if (comms.forced.kernel == Kernel::none) return pick(op, rows, bytes);
  TORCH_CHECK(op_of(comms.forced.kernel) == op, "hip_comms: the forced kernel ",
              static_cast<int>(comms.forced.kernel), " is not one of op ",
              static_cast<int>(op), "'s");
  return comms.forced;
}

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

// The scratch a kernel needs on each rank, in bytes. `packs` is a row's, `flat` the whole
// buffer's (plain all-reduce).
int64_t scratch_need(Kernel k, int64_t rows, int64_t packs, int64_t flat, int world) {
  switch (k) {
    case Kernel::two_shot: return ceil_div(flat, world) * 16;
    case Kernel::two_shot_rms_norm: return ceil_div(rows, world) * packs * 16;
    case Kernel::two_shot_add_rms_norm:
    case Kernel::two_shot_add_attn_res_rms_norm:
      return 2 * ceil_div(rows, world) * packs * 16;
    case Kernel::one_shot_rms_norm_gemm_add: return rows * packs * 16;
    case Kernel::two_shot_rms_norm_gemm_add: return ceil_div(rows, world) * packs * 16;
    default: return 0;
  }
}

// Whether `op` over [rows, hidden] of this element size runs a kernel here: something
// was picked, a row fits in registers at the picked width, and its scratch fits.
// Python asks this before every fused call and runs the unfused ops on a no.
bool admits(const Comms& comms, Op op, int64_t rows, int64_t hidden, int64_t elem) {
  const int64_t lanes = 16 / elem;
  if (hidden % lanes != 0) return false;
  const int64_t packs = hidden / lanes;
  const Launch l      = launch_for(comms, op, rows, rows * hidden * elem);
  if (l.kernel == Kernel::none) return false;
  if (op != Op::all_reduce && packs > kMaxRowPacks * l.threads) return false;
  if (l.kernel == Kernel::one_shot_rms_norm_gemm_add && rows > kGemmRows) return false;
  return scratch_need(l.kernel, rows, packs, rows * packs, comms.group.world_size()) <=
         comms.group.scratch_bytes();
}

// The picked launch for a call, refused where `admits` would have said no.
Launch checked_launch(const Comms& comms, Op op, int64_t rows, int64_t hidden,
                      int64_t elem) {
  TORCH_CHECK(admits(comms, op, rows, hidden, elem), "hip_comms: op ",
              static_cast<int>(op), " over [", rows, ", ", hidden,
              "] is declined here; ask admits first");
  return launch_for(comms, op, rows, rows * hidden * elem);
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

// PLAIN ALL-REDUCE over the flat buffer.
void all_reduce(Comms& comms, torch::Tensor& out, torch::Tensor& inp) {
  TORCH_CHECK(out.is_cuda() && inp.is_cuda(), "out and inp must be on device");
  TORCH_CHECK(out.is_contiguous() && inp.is_contiguous(), "out and inp must be contiguous");
  TORCH_CHECK(out.sizes() == inp.sizes(), "out and inp must have the same shape");
  TORCH_CHECK(out.scalar_type() == inp.scalar_type(), "out and inp must share a dtype");
  const Launch l =
      checked_launch(comms, Op::all_reduce, 1, inp.numel(), inp.element_size());
  const int n    = static_cast<int>(inp.numel() * inp.element_size() / 16);
  const ipc::Peers p = comms.group.peers(inp);
  auto stream        = at::cuda::getCurrentCUDAStream();
  const bool two     = is_two_shot(l.kernel);

#define LAUNCH_ALL_REDUCE(T, NG)                                                         \
  if (two)                                                                               \
    allreduce_two_shot<T, NG><<<dim3(l.blocks), dim3(l.threads), 0, stream>>>(           \
        p, out.data_ptr<T>(), n);                                                        \
  else                                                                                   \
    allreduce_one_shot<T, NG><<<dim3(l.blocks), dim3(l.threads), 0, stream>>>(           \
        p, out.data_ptr<T>(), n)
#define ALL_REDUCE_HALF(NG) LAUNCH_ALL_REDUCE(at::Half, NG)
#define ALL_REDUCE_BF16(NG) LAUNCH_ALL_REDUCE(at::BFloat16, NG)

  switch (out.scalar_type()) {
    case at::ScalarType::Half: BY_NGPUS(comms.group.world_size(), ALL_REDUCE_HALF) break;
    case at::ScalarType::BFloat16:
      BY_NGPUS(comms.group.world_size(), ALL_REDUCE_BF16) break;
    default: dtype_not_built();
  }
  not_built(comms.group.world_size());
#undef ALL_REDUCE_BF16
#undef ALL_REDUCE_HALF
#undef LAUNCH_ALL_REDUCE
}

// Every row op's grid: one block per row for one-shot, capped by what was picked, since a
// block past the last row has nothing to do and still pays the barriers; two-shot keeps
// every block for its gather.
int grid_for(const Launch& l, int rows) {
  return is_two_shot(l.kernel) ? l.blocks : std::min(l.blocks, rows);
}

// FUSED: all-reduce, then vLLM's `rms_norm`, or `fused_add_rms_norm` when `residual` is
// given (and then `residual_out` too). Exact to those ops' roundings; see
// `add_rms_norm_row`.
void all_reduce_add_rms_norm(Comms& comms, torch::Tensor& out, torch::Tensor* residual_out,
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
  const Launch l  = checked_launch(comms, add ? Op::add_rms_norm : Op::rms_norm, rows,
                                   inp.size(1), inp.element_size());
  const ipc::Peers p = comms.group.peers(inp);
  auto stream        = at::cuda::getCurrentCUDAStream();
  const int grid     = grid_for(l, rows);
  const bool two     = is_two_shot(l.kernel);
  const float feps   = static_cast<float>(eps);

#define LAUNCH_NORM(T, W, NG)                                                            \
  if (add) {                                                                             \
    if (two)                                                                             \
      allreduce_two_shot_add_rms_norm<T, W, NG>                                          \
          <<<dim3(grid), dim3(l.threads), 0, stream>>>(                                  \
              p, out.data_ptr<T>(), residual_out->data_ptr<T>(),                         \
              residual->data_ptr<T>(), weight.data_ptr<W>(), feps, rows, packs);         \
    else                                                                                 \
      allreduce_one_shot_add_rms_norm<T, W, NG>                                          \
          <<<dim3(grid), dim3(l.threads), 0, stream>>>(                                  \
              p, out.data_ptr<T>(), residual_out->data_ptr<T>(),                         \
              residual->data_ptr<T>(), weight.data_ptr<W>(), feps, rows, packs);         \
  } else {                                                                               \
    if (two)                                                                             \
      allreduce_two_shot_rms_norm<T, W, NG><<<dim3(grid), dim3(l.threads), 0, stream>>>( \
          p, out.data_ptr<T>(), weight.data_ptr<W>(), feps, rows, packs);                \
    else                                                                                 \
      allreduce_one_shot_rms_norm<T, W, NG><<<dim3(grid), dim3(l.threads), 0, stream>>>( \
          p, out.data_ptr<T>(), weight.data_ptr<W>(), feps, rows, packs);                \
  }
#define NORM_HALF(NG) LAUNCH_NORM(at::Half, at::Half, NG)
#define NORM_HALF_F32(NG) LAUNCH_NORM(at::Half, float, NG)
#define NORM_BF16(NG) LAUNCH_NORM(at::BFloat16, at::BFloat16, NG)
#define NORM_BF16_F32(NG) LAUNCH_NORM(at::BFloat16, float, NG)

  const int world = comms.group.world_size();
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
}

// FUSED: all-reduce, then add into the prefix, then Kimi-K3's AttnRes and its RMSNorm
// on each row (see `add_attn_res_rms_norm_row`). With `has_prefix` the sum is added to
// `prefix` in place; without, the sum IS the new prefix and is written there.
void all_reduce_add_attn_res_rms_norm(Comms& comms, torch::Tensor& prefix,
                                      torch::Tensor& out, torch::Tensor& inp,
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
  const Launch l  = checked_launch(comms, Op::add_attn_res_rms_norm, rows, hidden,
                                   inp.element_size());
  const ipc::Peers p = comms.group.peers(inp);
  auto stream        = at::cuda::getCurrentCUDAStream();
  const int grid     = grid_for(l, rows);
  const bool two     = is_two_shot(l.kernel);

#define ATTN_RES_ARGS(T)                                                                 \
  p, prefix.data_ptr<T>(), blocks.data_ptr<T>(), blocks.stride(0), blocks.stride(1),     \
      norm_weight.data_ptr<T>(), qk_weight.data_ptr<T>(),                                \
      out_norm_weight ? out_norm_weight->data_ptr<T>() : nullptr, out.data_ptr<T>(),     \
      static_cast<int>(num_blocks), static_cast<int>(write_idx), static_cast<float>(eps), \
      static_cast<float>(out_eps), rows, packs
#define LAUNCH_ATTN_RES(T, NG, PRE)                                                      \
  if (two)                                                                               \
    allreduce_two_shot_add_attn_res_rms_norm<T, NG, PRE>                                 \
        <<<dim3(grid), dim3(l.threads), 0, stream>>>(ATTN_RES_ARGS(T));                  \
  else                                                                                   \
    allreduce_one_shot_add_attn_res_rms_norm<T, NG, PRE>                                 \
        <<<dim3(grid), dim3(l.threads), 0, stream>>>(ATTN_RES_ARGS(T))
#define ATTN_RES_BY_PREFIX(T, NG)                                                        \
  if (has_prefix) {                                                                      \
    LAUNCH_ATTN_RES(T, NG, true);                                                        \
  } else {                                                                               \
    LAUNCH_ATTN_RES(T, NG, false);                                                       \
  }
#define ATTN_RES_HALF(NG) ATTN_RES_BY_PREFIX(at::Half, NG)
#define ATTN_RES_BF16(NG) ATTN_RES_BY_PREFIX(at::BFloat16, NG)

  const int world = comms.group.world_size();
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
void all_reduce_rms_norm_gemm_add(Comms& comms, torch::Tensor& out, int64_t out_col0,
                                  torch::Tensor& inp, torch::Tensor& norm_weight,
                                  double eps, torch::Tensor& gemm_weight) {
  TORCH_CHECK(inp.dim() == 2 && inp.is_contiguous(), "inp must be contiguous 2-D");
  const int64_t rows = inp.size(0), hidden = inp.size(1);
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
      checked_launch(comms, Op::rms_norm_gemm_add, rows, hidden, inp.element_size());
  TORCH_CHECK(l.threads % 64 == 0, "the GEMM phase needs whole waves; threads ", l.threads);
  const ipc::Peers p = comms.group.peers(inp);
  auto stream        = at::cuda::getCurrentCUDAStream();
  // EVERY BLOCK, not one per row: the GEMM phase spreads the columns over the whole grid.
  const bool two = is_two_shot(l.kernel);

#define LAUNCH_GEMM_ADD(T, NG)                                                           \
  if (two)                                                                               \
    allreduce_two_shot_rms_norm_gemm_add<T, NG>                                          \
        <<<dim3(l.blocks), dim3(l.threads), 0, stream>>>(                                \
            p, norm_weight.data_ptr<T>(), static_cast<float>(eps),                       \
            gemm_weight.data_ptr<T>(), static_cast<int>(n_cols), out.data_ptr<T>(),      \
            out.stride(0), static_cast<int>(out_col0), static_cast<int>(rows), packs);   \
  else                                                                                   \
    allreduce_one_shot_rms_norm_gemm_add<T, NG>                                          \
        <<<dim3(l.blocks), dim3(l.threads), 0, stream>>>(                                \
            p, norm_weight.data_ptr<T>(), static_cast<float>(eps),                       \
            gemm_weight.data_ptr<T>(), static_cast<int>(n_cols), out.data_ptr<T>(),      \
            out.stride(0), static_cast<int>(out_col0), static_cast<int>(rows), packs)
#define GEMM_ADD_HALF(NG) LAUNCH_GEMM_ADD(at::Half, NG)
#define GEMM_ADD_BF16(NG) LAUNCH_GEMM_ADD(at::BFloat16, NG)

  const int world = comms.group.world_size();
  switch (inp.scalar_type()) {
    case at::ScalarType::Half: BY_NGPUS(world, GEMM_ADD_HALF) break;
    case at::ScalarType::BFloat16: BY_NGPUS(world, GEMM_ADD_BF16) break;
    default: dtype_not_built();
  }
  not_built(world);
#undef GEMM_ADD_BF16
#undef GEMM_ADD_HALF
#undef LAUNCH_GEMM_ADD
}

#undef BY_NGPUS

}  // namespace hip_comms

// =================================================================================
// THE TORCH OP BOUNDARY. `ipc::Group` is a stateful C++ object and a torch op is a free
// function over schema types, so the object crosses as an opaque handle -- the same
// `fptr_t = int64_t` vLLM's custom all-reduce and quick-reduce use. IPC handles cross
// as `int[]` for the same reason they do there: a schema has no bytes type.
// =================================================================================

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

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
hip_comms::Comms& comms_of(fptr_t comms) {
  return *reinterpret_cast<hip_comms::Comms*>(comms);
}
}  // namespace

fptr_t rocm_comms_init(int64_t rank, int64_t world_size, int64_t self_signal,
                       const std::vector<std::vector<int64_t>>& signal_handles,
                       const std::vector<int64_t>& signal_offsets, int64_t peer_slab,
                       int64_t peer_slab_bytes, int64_t scratch_bytes,
                       double sync_timeout_s) {
  auto* comms = new hip_comms::Comms(
      static_cast<int>(rank), static_cast<int>(world_size),
      static_cast<uintptr_t>(self_signal), bytes_of(signal_handles), signal_offsets,
      static_cast<uintptr_t>(peer_slab), peer_slab_bytes, scratch_bytes, sync_timeout_s);
  return reinterpret_cast<fptr_t>(comms);
}

void rocm_comms_set_checked(fptr_t comms, bool checked) {
  comms_of(comms).group.set_checked(checked);
}

// THE SWEEP'S ONE HANDLE: every later launch of that kernel's op runs `kernel` at this
// geometry, and a launch of any other op is refused. `kernel` -1 clears it.
void rocm_comms_set_launch_override(fptr_t comms, int64_t kernel, int64_t blocks,
                                    int64_t threads) {
  using hip_comms::Kernel;
  if (kernel < 0) {
    comms_of(comms).forced = {Kernel::none, 0, 0};
    return;
  }
  TORCH_CHECK(kernel <= static_cast<int64_t>(Kernel::two_shot_rms_norm_gemm_add),
              "hip_comms: no kernel ", kernel);
  TORCH_CHECK(blocks > 0 && blocks <= hip_comms::ipc::kMaxBlocks, "blocks must be in [1, ",
              hip_comms::ipc::kMaxBlocks, "]");
  TORCH_CHECK(threads > 0 && threads <= 512 && threads % 64 == 0,
              "threads must be a multiple of 64 in [64, 512]");
  comms_of(comms).forced = {static_cast<Kernel>(kernel), static_cast<int>(blocks),
                            static_cast<int>(threads)};
}

bool rocm_comms_admits(fptr_t comms, int64_t op, int64_t rows, int64_t hidden,
                       int64_t element_size) {
  TORCH_CHECK(op >= 0 && op <= static_cast<int64_t>(hip_comms::Op::rms_norm_gemm_add),
              "hip_comms: no op ", op);
  TORCH_CHECK(element_size == 2, "hip_comms: only 2-byte dtypes are built");
  return hip_comms::admits(comms_of(comms), static_cast<hip_comms::Op>(op), rows, hidden,
                           element_size);
}

void rocm_comms_dispose(fptr_t comms) { delete &comms_of(comms); }

void rocm_comms_register_buffer(fptr_t comms,
                                const std::vector<std::vector<int64_t>>& handles,
                                const std::vector<int64_t>& offsets, int64_t self_ptr) {
  comms_of(comms).group.register_buffer(bytes_of(handles), offsets,
                                        static_cast<uintptr_t>(self_ptr));
}

std::vector<int64_t> rocm_comms_pending_graph_buffers(fptr_t comms) {
  auto pending = comms_of(comms).group.pending_graph_buffers();
  return std::vector<int64_t>(pending.begin(), pending.end());
}

// ONE ENTRY PER PENDING BUFFER, each the WORLD'S handles for it laid end to end: a
// schema nests two deep and this needs three (buffer, rank, byte), so the innermost
// level is split back out here by the handle size, which is fixed.
void rocm_comms_register_graph_buffers(
    fptr_t comms, const std::vector<std::vector<int64_t>>& handles,
    const std::vector<std::vector<int64_t>>& offsets) {
  const size_t stride = sizeof(hip_comms::ipc::Handle);
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
  comms_of(comms).group.register_graph_buffers(bytes, offsets);
}

int64_t rocm_comms_pending_count(fptr_t comms) {
  return comms_of(comms).group.pending_count();
}

void rocm_comms_all_reduce(fptr_t comms, torch::Tensor& out, torch::Tensor& inp) {
  hip_comms::all_reduce(comms_of(comms), out, inp);
}

void rocm_comms_all_reduce_rms_norm(fptr_t comms, torch::Tensor& out, torch::Tensor& inp,
                                    torch::Tensor& weight, double eps) {
  hip_comms::all_reduce_add_rms_norm(comms_of(comms), out, nullptr, inp, nullptr, weight,
                                     eps);
}

void rocm_comms_all_reduce_add_rms_norm(fptr_t comms, torch::Tensor& out,
                                        torch::Tensor& residual_out, torch::Tensor& inp,
                                        torch::Tensor& residual, torch::Tensor& weight,
                                        double eps) {
  hip_comms::all_reduce_add_rms_norm(comms_of(comms), out, &residual_out, inp, &residual,
                                     weight, eps);
}

void rocm_comms_all_reduce_add_attn_res_rms_norm(
    fptr_t comms, torch::Tensor& prefix, torch::Tensor& out, torch::Tensor& inp,
    torch::Tensor& blocks, torch::Tensor& norm_weight, torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& out_norm_weight, int64_t num_blocks,
    int64_t write_idx, double eps, double out_eps, bool has_prefix) {
  hip_comms::all_reduce_add_attn_res_rms_norm(
      comms_of(comms), prefix, out, inp, blocks, norm_weight, qk_weight,
      out_norm_weight ? &*out_norm_weight : nullptr, num_blocks, write_idx, eps, out_eps,
      has_prefix);
}

void rocm_comms_all_reduce_rms_norm_gemm_add(fptr_t comms, torch::Tensor& out,
                                             int64_t out_col0, torch::Tensor& inp,
                                             torch::Tensor& norm_weight, double eps,
                                             torch::Tensor& gemm_weight) {
  hip_comms::all_reduce_rms_norm_gemm_add(comms_of(comms), out, out_col0, inp, norm_weight,
                                          eps, gemm_weight);
}

std::tuple<std::vector<int64_t>, int64_t> rocm_comms_handle_and_offset(int64_t ptr) {
  auto [handle, offset] = hip_comms::ipc::handle_and_offset(static_cast<uintptr_t>(ptr));
  std::vector<int64_t> bytes(handle.begin(), handle.end());
  return std::make_tuple(bytes, offset);
}

// The sizes Python needs to allocate the signal block and the peer slab, and the bounds it
// checks a world size and a launch against. Constants of the kernel, so they are asked for
// rather than restated.
std::vector<int64_t> rocm_comms_sizes() {
  return {static_cast<int64_t>(sizeof(hip_comms::ipc::Signal)),
          static_cast<int64_t>(sizeof(hip_comms::ipc::PeerPtrs)),
          static_cast<int64_t>(hip_comms::ipc::kMaxBlocks),
          static_cast<int64_t>(hip_comms::ipc::kMaxRanks),
          static_cast<int64_t>(sizeof(hip_comms::ipc::Handle)),
          static_cast<int64_t>(hip_comms::kMaxRowPacks)};
}
