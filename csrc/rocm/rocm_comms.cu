// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Our HIP collectives. Self-contained: torch + the HIP runtime, nothing from aiter's
// csrc, no build system. hip_comms.py compiles this file.
//
// THIS FILE HOLDS NO POLICY. It has no size heuristics, no getenv, and no tuned
// constants: which algorithm, how many blocks and how many threads all arrive as
// arguments, chosen in Python. A decision made here would be invisible and untestable,
// and the escape hatch it eventually needs is how you end up with an env var deciding
// how a kernel runs. What lives here is mechanism -- the peer handshake, the barrier,
// the reduction -- plus a dispatch table over the instantiations that exist.
//
// Compile-time vs runtime is the one distinction that shapes everything. `ngpus` and the
// dtype must be constexpr to unroll and vectorize, so they are template parameters and
// the instantiation list below is the finite menu Python may pick from; asking for one
// that was not built RAISES rather than falling back, because a silent substitution
// produces a number about the wrong thing.
//
// rocm_comms/ holds the layers as headers, all included here so the device code stays in
// this one translation unit and needs no -fgpu-rdc: ipc.cuh (peer memory and sync), then
// one allreduce_*.cuh per variant built on it.

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "rocm_comms/allreduce_one_shot.cuh"
#include "rocm_comms/allreduce_one_shot_rmsnorm.cuh"
#include "rocm_comms/allreduce_two_shot.cuh"
#include "rocm_comms/allreduce_two_shot_rmsnorm.cuh"
#include "rocm_comms/ipc.cuh"

namespace hip_comms {

constexpr int64_t kAlgoOneShot = 0;
constexpr int64_t kAlgoTwoShot = 1;

void check_algo(int64_t algo) {
  if (algo != kAlgoOneShot && algo != kAlgoTwoShot)
    throw std::runtime_error("hip_comms: algo " + std::to_string(algo) +
                             " is not built. Built: 0 (one_shot), 1 (two_shot).");
}

// THE instantiation menu. Every combination that exists is named here exactly once, so
// an unsupported request is a listed refusal rather than a wrong kernel.
void dispatch(ipc::Group& group, torch::Tensor& out, void* input, int64_t algo,
              int64_t blocks, int64_t threads, int n) {
  auto stream        = at::cuda::getCurrentCUDAStream();
  const ipc::Peers p = group.peers(input);
#define LAUNCH(T, NG)                                                                   \
  do {                                                                                  \
    if (algo == kAlgoTwoShot)                                                           \
      allreduce_two_shot<T, NG><<<dim3(blocks), dim3(threads), 0, stream>>>(            \
          p, out.data_ptr<T>(), n);                                                     \
    else                                                                                \
      allreduce_one_shot<T, NG><<<dim3(blocks), dim3(threads), 0, stream>>>(            \
          p, out.data_ptr<T>(), n);                                                     \
  } while (0)

#define BY_NGPUS(T)                                                                     \
  switch (group.world_size()) {                                                         \
    case 2: LAUNCH(T, 2); return;                                                       \
    case 4: LAUNCH(T, 4); return;                                                       \
    case 8: LAUNCH(T, 8); return;                                                       \
    default: break;                                                                     \
  }

  check_algo(algo);
  switch (out.scalar_type()) {
    case at::ScalarType::Half: BY_NGPUS(at::Half) break;
    case at::ScalarType::BFloat16: BY_NGPUS(at::BFloat16) break;
    default:
      throw std::runtime_error("hip_comms: dtype not built. Built: float16, bfloat16.");
  }
  throw std::runtime_error("hip_comms: world_size " + std::to_string(group.world_size()) +
                           " not built. Built: 2, 4, 8.");
#undef BY_NGPUS
#undef LAUNCH
}

// `algo`, `blocks` and `threads` are the CALLER's decision. Nothing here inspects the
// size to pick them.
void all_reduce(ipc::Group& group, torch::Tensor& out, torch::Tensor& inp, int64_t algo,
                int64_t blocks, int64_t threads) {
  TORCH_CHECK(out.is_cuda() && inp.is_cuda(), "out and inp must be on device");
  TORCH_CHECK(out.is_contiguous() && inp.is_contiguous(), "out and inp must be contiguous");
  TORCH_CHECK(out.sizes() == inp.sizes(), "out and inp must have the same shape");
  TORCH_CHECK(out.scalar_type() == inp.scalar_type(), "out and inp must share a dtype");
  TORCH_CHECK(blocks > 0 && blocks <= ipc::kMaxBlocks, "blocks must be in [1, ",
              ipc::kMaxBlocks, "]");
  TORCH_CHECK(threads > 0 && threads <= 512, "threads must be in [1, 512]");

  const int lanes = 16 / static_cast<int>(inp.element_size());
  TORCH_CHECK(inp.numel() % lanes == 0, "numel ", inp.numel(),
              " must be a multiple of ", lanes, " for 16-byte vectorized access");
  const int n = static_cast<int>(inp.numel() / lanes);

  // TWO-SHOT PUBLISHES ITS SLICE IN THE SCRATCH, so the scratch bounds the buffer it can
  // reduce: ceil(n/ngpus) vectors of 16 bytes. Checked HERE, where the tensor can be named,
  // rather than discovered as a peer reading past the end of an IPC mapping.
  if (algo == kAlgoTwoShot) {
    const int world_size  = group.world_size();
    const int64_t scratch = group.scratch_bytes();
    const int64_t chunk   = (static_cast<int64_t>(n) + world_size - 1) / world_size;
    TORCH_CHECK(chunk * 16 <= scratch, "hip_comms: two_shot needs ", chunk * 16,
                " scratch bytes for a ", inp.numel() * inp.element_size(),
                "-byte buffer across ", world_size, " ranks, but only ", scratch,
                " were allocated. Raise HipTunables.scratch_bytes.");
  }
  dispatch(group, out, inp.data_ptr(), algo, blocks, threads, n);
}

// FUSED: sum across ranks, add the residual, normalise. `out` is the normed result and
// `residual_out` the sum-plus-residual the next block needs -- both are real outputs, so
// nothing here is scratch.
//
// `algo` picks one-shot or two-shot, as for the plain all-reduce.
void all_reduce_rmsnorm(ipc::Group& group, torch::Tensor& out,
                        torch::Tensor& residual_out, torch::Tensor& inp,
                        torch::Tensor& residual, torch::Tensor& weight, double eps,
                        int64_t algo, int64_t blocks, int64_t threads) {
  check_algo(algo);
  TORCH_CHECK(out.is_cuda() && inp.is_cuda() && residual.is_cuda() && weight.is_cuda(),
              "every tensor must be on device");
  TORCH_CHECK(out.is_contiguous() && residual_out.is_contiguous() &&
                  inp.is_contiguous() && residual.is_contiguous() &&
                  weight.is_contiguous(),
              "every tensor must be contiguous");
  TORCH_CHECK(out.sizes() == inp.sizes() && residual_out.sizes() == inp.sizes() &&
                  residual.sizes() == inp.sizes(),
              "out, residual_out and residual must have inp's shape");
  TORCH_CHECK(out.scalar_type() == inp.scalar_type() &&
                  residual_out.scalar_type() == inp.scalar_type() &&
                  residual.scalar_type() == inp.scalar_type() &&
                  weight.scalar_type() == inp.scalar_type(),
              "every tensor must share inp's dtype");
  TORCH_CHECK(inp.dim() == 2, "inp must be 2-D [tokens, hidden]; got ", inp.dim(), "-D");
  TORCH_CHECK(weight.dim() == 1 && weight.numel() == inp.size(1),
              "weight must be 1-D of hidden=", inp.size(1));
  TORCH_CHECK(blocks > 0 && blocks <= ipc::kMaxBlocks, "blocks must be in [1, ",
              ipc::kMaxBlocks, "]");
  TORCH_CHECK(threads > 0 && threads <= 512, "threads must be in [1, 512]");

  const int lanes = 16 / static_cast<int>(inp.element_size());
  const int64_t hidden = inp.size(1);
  // A BLOCK OWNS A ROW, so a row has to divide into whole 16-byte packs. Refused here,
  // where the tensor can be named, rather than by a partial pack read past the end.
  TORCH_CHECK(hidden % lanes == 0, "hidden ", hidden, " must be a multiple of ", lanes,
              " for 16-byte vectorized access");
  const int rows  = static_cast<int>(inp.size(0));
  const int packs = static_cast<int>(hidden / lanes);

  // TWO-SHOT PUBLISHES BOTH OUTPUTS FOR ITS ROWS in the scratch: 2 x ceil(rows/ngpus) rows.
  if (algo == kAlgoTwoShot) {
    const int world_size  = group.world_size();
    const int64_t scratch = group.scratch_bytes();
    const int64_t need =
        2 * ((static_cast<int64_t>(rows) + world_size - 1) / world_size) * packs * 16;
    TORCH_CHECK(need <= scratch, "hip_comms: two_shot rmsnorm needs ", need,
                " scratch bytes for ", rows, " rows of ", hidden, " across ", world_size,
                " ranks, but only ", scratch,
                " were allocated. Raise HipTunables.scratch_bytes.");
  }

  const ipc::Peers p = group.peers(inp.data_ptr());
  auto stream        = at::cuda::getCurrentCUDAStream();
  // ONE-SHOT: ONE BLOCK PER ROW, capped by what was asked for, since a block past the last
  // row has nothing to do and still pays both barriers. Two-shot keeps every block: its
  // gather is grid-stride over the flat buffer.
  const int grid = algo == kAlgoTwoShot
                       ? static_cast<int>(blocks)
                       : static_cast<int>(std::min<int64_t>(blocks, rows));

#define LAUNCH_FUSED(T, NG)                                                             \
  do {                                                                                  \
    if (algo == kAlgoTwoShot)                                                           \
      allreduce_two_shot_rmsnorm<T, NG><<<dim3(grid), dim3(threads), 0, stream>>>(      \
          p, out.data_ptr<T>(), residual_out.data_ptr<T>(), residual.data_ptr<T>(),     \
          weight.data_ptr<T>(), static_cast<float>(eps), rows, packs);                  \
    else                                                                                \
      allreduce_one_shot_rmsnorm<T, NG><<<dim3(grid), dim3(threads), 0, stream>>>(      \
          p, out.data_ptr<T>(), residual_out.data_ptr<T>(), residual.data_ptr<T>(),     \
          weight.data_ptr<T>(), static_cast<float>(eps), rows, packs);                  \
  } while (0)

#define FUSED_BY_NGPUS(T)                                                               \
  switch (group.world_size()) {                                                         \
    case 2: LAUNCH_FUSED(T, 2); return;                                                 \
    case 4: LAUNCH_FUSED(T, 4); return;                                                 \
    case 8: LAUNCH_FUSED(T, 8); return;                                                 \
    default: break;                                                                     \
  }

  switch (inp.scalar_type()) {
    case at::ScalarType::Half: FUSED_BY_NGPUS(at::Half) break;
    case at::ScalarType::BFloat16: FUSED_BY_NGPUS(at::BFloat16) break;
    default:
      throw std::runtime_error("hip_comms: dtype not built. Built: float16, bfloat16.");
  }
  throw std::runtime_error("hip_comms: world_size " + std::to_string(group.world_size()) +
                           " not built. Built: 2, 4, 8.");
#undef FUSED_BY_NGPUS
#undef LAUNCH_FUSED
}

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

fptr_t rocm_comms_init(int64_t rank, int64_t world_size, int64_t self_signal,
                       const std::vector<std::vector<int64_t>>& signal_handles,
                       const std::vector<int64_t>& signal_offsets, int64_t peer_slab,
                       int64_t peer_slab_bytes, int64_t scratch_bytes) {
  auto* comms = new hip_comms::ipc::Group(
      static_cast<int>(rank), static_cast<int>(world_size),
      static_cast<uintptr_t>(self_signal), bytes_of(signal_handles), signal_offsets,
      static_cast<uintptr_t>(peer_slab), peer_slab_bytes, scratch_bytes);
  return reinterpret_cast<fptr_t>(comms);
}

void rocm_comms_dispose(fptr_t comms) {
  delete reinterpret_cast<hip_comms::ipc::Group*>(comms);
}

void rocm_comms_register_buffer(fptr_t comms,
                                const std::vector<std::vector<int64_t>>& handles,
                                const std::vector<int64_t>& offsets, int64_t self_ptr) {
  reinterpret_cast<hip_comms::ipc::Group*>(comms)->register_buffer(
      bytes_of(handles), offsets, static_cast<uintptr_t>(self_ptr));
}

std::vector<int64_t> rocm_comms_pending_graph_buffers(fptr_t comms) {
  auto pending =
      reinterpret_cast<hip_comms::ipc::Group*>(comms)->pending_graph_buffers();
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
  reinterpret_cast<hip_comms::ipc::Group*>(comms)->register_graph_buffers(bytes, offsets);
}

int64_t rocm_comms_pending_count(fptr_t comms) {
  return reinterpret_cast<hip_comms::ipc::Group*>(comms)->pending_count();
}

void rocm_comms_all_reduce(fptr_t comms, torch::Tensor& out, torch::Tensor& inp,
                           int64_t algo, int64_t blocks, int64_t threads) {
  hip_comms::all_reduce(*reinterpret_cast<hip_comms::ipc::Group*>(comms), out, inp, algo,
                        blocks, threads);
}

void rocm_comms_all_reduce_rmsnorm(fptr_t comms, torch::Tensor& out,
                                   torch::Tensor& residual_out, torch::Tensor& inp,
                                   torch::Tensor& residual, torch::Tensor& weight,
                                   double eps, int64_t algo, int64_t blocks,
                                   int64_t threads) {
  hip_comms::all_reduce_rmsnorm(*reinterpret_cast<hip_comms::ipc::Group*>(comms), out,
                                residual_out, inp, residual, weight, eps, algo, blocks,
                                threads);
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
          static_cast<int64_t>(sizeof(hip_comms::ipc::Handle))};
}
