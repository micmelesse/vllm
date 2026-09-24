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

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace hip_comms {

constexpr int kMaxRanks  = 8;
constexpr int kMaxBlocks = 36;

#define HIP_CHECK(expr)                                                             \
  do {                                                                              \
    hipError_t _e = (expr);                                                         \
    if (_e != hipSuccess) {                                                         \
      throw std::runtime_error(std::string("hip_comms: ") + #expr + " -> " +         \
                               hipGetErrorString(_e));                              \
    }                                                                               \
  } while (0)

// ---------------------------------------------------------------------------------
// Shared layout. One IPC allocation per rank holds the signal block AND the scratch the
// two-stage algorithm needs, so adding that algorithm changes no layout: scratch is
// simply the bytes after the struct.
// ---------------------------------------------------------------------------------

// TWO counter arrays, not one. A peer block can reach the second barrier while this one
// is still at the first, and with a single array the peer would write counter+1 while we
// busy-wait on counter. `seq` is the per-block monotonic sequence number.
struct Signal {
  alignas(128) uint32_t start[kMaxBlocks][kMaxRanks];
  alignas(128) uint32_t end[kMaxBlocks][kMaxRanks];
  alignas(128) uint32_t seq[kMaxBlocks];
};

struct __align__(16) PeerPtrs { void* p[kMaxRanks]; };
struct __align__(16) PeerSignals { Signal* s[kMaxRanks]; };

__device__ __forceinline__ void* scratch_of(Signal* sig) { return sig + 1; }

#define DINLINE __device__ __forceinline__

// ---------------------------------------------------------------------------------
// Barrier. System scope to REACH a peer's memory, device scope to poll our own.
// ---------------------------------------------------------------------------------

template <int ngpus>
DINLINE void barrier_start(PeerSignals sigs, Signal* self, int rank) {
  uint32_t f = self->seq[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    __scoped_atomic_store_n(&sigs.s[threadIdx.x]->start[blockIdx.x][rank], f,
                            __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    while (__scoped_atomic_load_n(&self->start[blockIdx.x][threadIdx.x],
                                  __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) < f);
  }
  __syncthreads();
  if (threadIdx.x == 0) self->seq[blockIdx.x] = f;
}

// `final_sync` drops the release/acquire pair: nothing after the last barrier reads what
// this kernel wrote, so ordering costs without buying anything.
template <int ngpus, bool final_sync>
DINLINE void barrier_end(PeerSignals sigs, Signal* self, int rank) {
  __syncthreads();
  uint32_t f = self->seq[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    __scoped_atomic_store_n(&sigs.s[threadIdx.x]->end[blockIdx.x][rank], f,
                            final_sync ? __ATOMIC_RELAXED : __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
    while (__scoped_atomic_load_n(&self->end[blockIdx.x][threadIdx.x],
                                  final_sync ? __ATOMIC_RELAXED : __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_DEVICE) < f);
  }
  if constexpr (!final_sync) __syncthreads();
  if (threadIdx.x == 0) self->seq[blockIdx.x] = f;
}

// ---------------------------------------------------------------------------------
// Vectorized reduce. 16 bytes per thread, accumulated in fp32 so a bf16 sum of 8 values
// rounds once at the end rather than eight times along the way.
// ---------------------------------------------------------------------------------

template <typename T, int N>
struct __align__(sizeof(T) * N) vec {
  T d[N];
};

template <typename T>
struct traits {
  static constexpr int N = 16 / sizeof(T);
  using V = vec<T, N>;
};

template <typename T, int ngpus>
DINLINE typename traits<T>::V reduce_at(const typename traits<T>::V* const ptrs[],
                                        int idx) {
  constexpr int N = traits<T>::N;
  float acc[N];
  auto v0 = ptrs[0][idx];
#pragma unroll
  for (int j = 0; j < N; ++j) acc[j] = static_cast<float>(v0.d[j]);
#pragma unroll
  for (int i = 1; i < ngpus; ++i) {
    auto v = ptrs[i][idx];
#pragma unroll
    for (int j = 0; j < N; ++j) acc[j] += static_cast<float>(v.d[j]);
  }
  typename traits<T>::V out;
#pragma unroll
  for (int j = 0; j < N; ++j) out.d[j] = static_cast<T>(acc[j]);
  return out;
}

// ---------------------------------------------------------------------------------
// Kernels. One per algorithm; `algo` selects among them at dispatch.
// ---------------------------------------------------------------------------------

// ONE-SHOT: every rank reads every peer's input over the whole buffer and writes its own
// output. Moves ngpus x the bytes of a two-stage, so it is the wrong algorithm at 1 MiB
// and the right one to land first: it exercises the peer handshake end to end and is
// correct, which makes two-stage a pure performance change against a working baseline.
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    one_shot_all_reduce(const PeerPtrs* peers, PeerSignals sigs, Signal* self,
                        T* __restrict__ out, int rank, int size) {
  using V = typename traits<T>::V;
  // ROTATED by rank, so the ranks do not all hammer rank 0's buffer first. The consequence is
  // worth knowing: every rank sums the same ngpus values in a DIFFERENT order, so the outputs
  // agree only to within one ULP of T rather than bitwise. `reduce_at` accumulates in float and
  // rounds once, which bounds it there. Rank 0's order is 0..ngpus-1, so rank 0 alone matches a
  // sequential reference exactly -- a test that reads only rank 0 will call this bit-exact.
  const V* ptrs[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; ++i)
    ptrs[i] = reinterpret_cast<const V*>(peers->p[(rank + i) % ngpus]);

  barrier_start<ngpus>(sigs, self, rank);
  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int idx = tid; idx < size; idx += stride)
    reinterpret_cast<V*>(out)[idx] = reduce_at<T, ngpus>(ptrs, idx);
  // Required, not defensive: without it a rank can return and let its INPUT be reused
  // while a peer is still reading that input.
  barrier_end<ngpus, true>(sigs, self, rank);
}

// ONE-SHOT, FUSED WITH RMSNorm. The sum is already in registers when one-shot is about to
// store it, so normalising there costs one HBM round trip less than an all-reduce kernel
// followed by a norm kernel, and one launch less.
//
// THE DECOMPOSITION IS THE WHOLE COST, and it is why this is a separate kernel rather than a
// flag on the one above. RMSNorm needs the sum of squares across a WHOLE ROW, so a block has
// to own a row; plain one-shot is grid-stride over the flat buffer and a block owns whatever
// it lands on. So this is one block per row, striding over rows, and the reduction inside a
// block is what the norm needs. (aiter reaches the same constraint from the other side and
// spells it `hidden_dim / pack_size <= 1024` -- their gate for the 1-stage fused path.)
//
// TWO PASSES OVER THE ROW, NOT ONE, and the second reads what the first wrote. Holding the
// row in registers would avoid it, but only up to a hidden size the register file allows,
// and the bound would then be a silent wrong answer rather than a refusal. `residual_out` has
// to be written anyway, so the re-read is of a 16 KB row this block wrote microseconds ago --
// L2, not HBM. If profiling says otherwise, a register-resident variant is the next step.
//
// THE VARIANCE IS AN fp32 SUM OF fp32 SQUARES, matching `vllm.ir.ops.fused_add_rms_norm`.
// What differs from it: the second pass reads back the ROUNDED residual, where the reference
// scales the unrounded fp32 value. One rounding, inside the norm's own tolerance, and the
// equivalence test is what says so rather than this comment.
DINLINE float block_sum(float v) {
  // 8 = 512 / 64, the launch bound over the wavefront. A block wider than the bound cannot
  // be launched, so this cannot be overrun.
  __shared__ float partial[8];
  __shared__ float total;
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  for (int off = warpSize / 2; off > 0; off >>= 1) v += __shfl_down(v, off, warpSize);
  if (lane == 0) partial[warp] = v;
  __syncthreads();
  const int warps = (blockDim.x + warpSize - 1) / warpSize;
  if (warp == 0) {
    v = (lane < warps) ? partial[lane] : 0.0f;
    for (int off = warpSize / 2; off > 0; off >>= 1) v += __shfl_down(v, off, warpSize);
    if (lane == 0) total = v;
  }
  __syncthreads();
  return total;
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) one_shot_all_reduce_rmsnorm(
    const PeerPtrs* peers, PeerSignals sigs, Signal* self, T* __restrict__ out,
    T* __restrict__ residual_out, const T* __restrict__ residual,
    const T* __restrict__ weight, float eps, int rank, int rows, int packs) {
  using V          = typename traits<T>::V;
  constexpr int NL = traits<T>::N;
  const V* ptrs[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; ++i)
    ptrs[i] = reinterpret_cast<const V*>(peers->p[(rank + i) % ngpus]);

  barrier_start<ngpus>(sigs, self, rank);

  const V* res_in = reinterpret_cast<const V*>(residual);
  V* res_out      = reinterpret_cast<V*>(residual_out);
  const V* w      = reinterpret_cast<const V*>(weight);
  V* o            = reinterpret_cast<V*>(out);
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);

  // A BLOCK OWNS A ROW AT A TIME. Uniform across the block, so every `__syncthreads`
  // below is reached by every thread in it.
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int base = row * packs;
    float acc      = 0.0f;
    for (int i = threadIdx.x; i < packs; i += blockDim.x) {
      V sum = reduce_at<T, ngpus>(ptrs, base + i);
      V r   = res_in[base + i];
#pragma unroll
      for (int j = 0; j < NL; ++j) {
        const float s = static_cast<float>(sum.d[j]) + static_cast<float>(r.d[j]);
        sum.d[j]      = static_cast<T>(s);
        acc += s * s;
      }
      res_out[base + i] = sum;
    }
    const float scale = rsqrtf(block_sum(acc) * inv_hidden + eps);
    for (int i = threadIdx.x; i < packs; i += blockDim.x) {
      V r  = res_out[base + i];
      V wv = w[i];
#pragma unroll
      for (int j = 0; j < NL; ++j)
        r.d[j] = static_cast<T>(static_cast<float>(r.d[j]) * scale *
                                static_cast<float>(wv.d[j]));
      o[base + i] = r;
    }
    // Before the next row reuses `block_sum`'s shared slots.
    __syncthreads();
  }
  // Same reason as one-shot's: a rank that returns lets its INPUT be reused while a peer
  // is still reading it.
  barrier_end<ngpus, true>(sigs, self, rank);
}

// TWO-SHOT: reduce-scatter, then all-gather. Every rank owns one slice of the buffer,
// reduces ONLY that slice by reading every peer's copy of it, publishes the result in its
// own scratch, and then every rank copies all ngpus slices back out.
//
// WHY IT EXISTS: bytes. One-shot moves (ngpus-1) x N per rank in one pass; this moves
// (ngpus-1)/ngpus x N twice, so 1.75N against 7N at ngpus=8 -- a 4x reduction, which is
// exactly the ratio measured between vLLM's two-stage and our one-shot on a real capture.
// It costs one more barrier, so it is the WRONG algorithm below the crossover where that
// barrier dominates and the right one above it. Neither is universally better and the
// caller picks: `algo` is the caller's decision, as it is for blocks and threads.
//
// THE SLICE IS ceil(size/ngpus) AND THE LAST RANK TAKES WHAT IS LEFT, so a buffer that
// does not divide by ngpus is still reduced exactly once everywhere -- no padding, no
// element summed twice, and a rank whose slice is empty still runs both barriers.
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    two_shot_all_reduce(const PeerPtrs* peers, PeerSignals sigs, Signal* self,
                        T* __restrict__ out, int rank, int size) {
  using V = typename traits<T>::V;

  // ROTATED by rank, for the reason one-shot rotates: the ranks do not all read rank 0
  // first. The same consequence follows -- each rank sums in a different order, so the
  // slices agree to within one ULP rather than bitwise. Unlike one-shot, EVERY element of
  // the output here was summed by exactly one rank, so all ranks see identical bytes;
  // what differs is only which order that one rank used.
  const V* ptrs[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; ++i)
    ptrs[i] = reinterpret_cast<const V*>(peers->p[(rank + i) % ngpus]);

  const int chunk = (size + ngpus - 1) / ngpus;
  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  barrier_start<ngpus>(sigs, self, rank);

  // PHASE 1 -- reduce-scatter. Our slice, summed across every rank, into our own scratch.
  {
    const int begin = rank * chunk;
    const int end   = begin + chunk < size ? begin + chunk : size;
    V* mine = reinterpret_cast<V*>(scratch_of(self));
    for (int idx = begin + tid; idx < end; idx += stride)
      mine[idx - begin] = reduce_at<T, ngpus>(ptrs, idx);
  }

  // NOT `final_sync`: the peers are about to READ what we just wrote, so this barrier has
  // to carry the release/acquire pair that the last one is allowed to drop.
  barrier_end<ngpus, false>(sigs, self, rank);

  // PHASE 2 -- all-gather. Slice i is finished and sitting in rank i's scratch; every rank
  // copies all ngpus of them into its own output.
  {
    V* dst = reinterpret_cast<V*>(out);
#pragma unroll
    for (int i = 0; i < ngpus; ++i) {
      const int begin = i * chunk;
      const int end   = begin + chunk < size ? begin + chunk : size;
      const V* src = reinterpret_cast<const V*>(scratch_of(sigs.s[i]));
      for (int idx = begin + tid; idx < end; idx += stride)
        dst[idx] = src[idx - begin];
    }
  }

  // Required for the same reason one-shot's is: without it a rank can return and let its
  // INPUT be reused while a peer is still reading that input.
  barrier_end<ngpus, true>(sigs, self, rank);
}

// ---------------------------------------------------------------------------------
// The context: peer memory, registration, dispatch. One per process group.
// ---------------------------------------------------------------------------------

using Handle = hipIpcMemHandle_t;

static Handle handle_from(const std::string& bytes) {
  if (bytes.size() != sizeof(Handle))
    throw std::runtime_error("hip_comms: ipc handle is " + std::to_string(bytes.size()) +
                             " bytes, expected " + std::to_string(sizeof(Handle)));
  Handle h;
  std::memcpy(&h, bytes.data(), sizeof(Handle));
  return h;
}

// hipIpcGetMemHandle must be given the BASE of an allocation, but a torch tensor sits at
// an offset inside one -- so the handle names the allocation and the offset locates the
// tensor within it. Returning both together is what keeps the two from drifting apart.
static hipPointer_attribute range_start_attr = HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR;

// std::string as a BYTE BUFFER, not text: an IPC handle is arbitrary binary. It leaves
// this file as `int[]` at the op boundary below, which is how vLLM's other all-reduces
// carry handle bytes through a schema that has no bytes type.
static std::pair<std::string, int64_t> ipc_handle_and_offset(uintptr_t ptr) {
  void* base = nullptr;
  HIP_CHECK(hipPointerGetAttribute(&base, range_start_attr,
                                   reinterpret_cast<hipDeviceptr_t>(ptr)));
  Handle h;
  HIP_CHECK(hipIpcGetMemHandle(&h, base));
  return {std::string(reinterpret_cast<const char*>(&h), sizeof(Handle)),
          reinterpret_cast<char*>(ptr) - static_cast<char*>(base)};
}

class Comms {
 public:
  // `signal_handles`/`signal_offsets` are the whole world's handles for their own signal
  // allocation, gathered in PYTHON -- the collective that exchanges them belongs to the
  // process group, which C++ has no business knowing about.
  Comms(int rank, int world_size, uintptr_t self_signal,
        const std::vector<std::string>& signal_handles,
        const std::vector<int64_t>& signal_offsets, uintptr_t peer_slab,
        int64_t peer_slab_bytes, int64_t scratch_bytes)
      : rank_(rank),
        world_size_(world_size),
        self_signal_(reinterpret_cast<Signal*>(self_signal)),
        scratch_bytes_(scratch_bytes),
        slab_(reinterpret_cast<PeerPtrs*>(peer_slab)),
        slab_end_(reinterpret_cast<PeerPtrs*>(peer_slab) +
                  peer_slab_bytes / sizeof(PeerPtrs)),
        cursor_(reinterpret_cast<PeerPtrs*>(peer_slab)) {
    if (world_size_ < 2 || world_size_ > kMaxRanks)
      throw std::runtime_error("hip_comms: world_size " + std::to_string(world_size_) +
                               " outside [2, " + std::to_string(kMaxRanks) + "]");
    if (signal_handles.size() != static_cast<size_t>(world_size_) ||
        signal_offsets.size() != static_cast<size_t>(world_size_))
      throw std::runtime_error("hip_comms: expected one signal handle+offset per rank");
    auto opened = open_peers(signal_handles, signal_offsets, self_signal);
    for (int i = 0; i < world_size_; ++i)
      peer_signals_.s[i] = reinterpret_cast<Signal*>(opened[i]);
  }

  ~Comms() {
    for (const auto& kv : opened_) hipIpcCloseMemHandle(kv.second);
  }

  // A buffer whose address is known ahead of time. The eager path.
  void register_buffer(const std::vector<std::string>& handles,
                       const std::vector<int64_t>& offsets, uintptr_t self_ptr) {
    auto ptrs = open_peers(handles, offsets, self_ptr);
    registered_[reinterpret_cast<void*>(self_ptr)] = commit(ptrs);
  }

  // The CAPTURE path, in two halves. During capture the input address is not registered
  // yet, so `all_reduce` reserves a slab slot and remembers the pointer; afterwards
  // Python gathers handles for everything remembered and this fills the slots in. Sound
  // because a captured address is fixed for the graph's life -- the kernel reads a
  // PeerPtrs populated AFTER the capture that recorded the launch.
  std::vector<uintptr_t> pending_graph_buffers() const {
    std::vector<uintptr_t> out;
    out.reserve(pending_.size());
    for (void* p : pending_) out.push_back(reinterpret_cast<uintptr_t>(p));
    return out;
  }

  void register_graph_buffers(const std::vector<std::vector<std::string>>& handles,
                              const std::vector<std::vector<int64_t>>& offsets) {
    if (handles.size() != pending_.size() || offsets.size() != pending_.size())
      throw std::runtime_error("hip_comms: got handles for " +
                               std::to_string(handles.size()) + " buffers, " +
                               std::to_string(pending_.size()) + " are pending");
    // The slots are filled in; nothing goes into `registered_`. That map means "an address
    // this object keeps alive", and a captured buffer is the opposite -- it dies with its
    // graph. Nothing needs it there either: the slot pointer is baked into the launch the
    // capture recorded, so a replay never looks the address up. `registered_` therefore holds
    // exactly what `register_buffer` was called for, which is what Python's `_registered`
    // tracks.
    for (size_t i = 0; i < pending_.size(); ++i) {
      auto ptrs = open_peers(handles[i], offsets[i],
                             reinterpret_cast<uintptr_t>(pending_[i]));
      write_slot(pending_slots_[i], ptrs);
    }
    pending_.clear();
    pending_slots_.clear();
  }

  int64_t pending_count() const { return static_cast<int64_t>(pending_.size()); }

  // `algo`, `blocks` and `threads` are the CALLER's decision. Nothing here inspects the
  // size to pick them.
  void all_reduce(torch::Tensor& out, torch::Tensor& inp, int64_t algo, int64_t blocks,
                  int64_t threads) {
    TORCH_CHECK(out.is_cuda() && inp.is_cuda(), "out and inp must be on device");
    TORCH_CHECK(out.is_contiguous() && inp.is_contiguous(), "out and inp must be contiguous");
    TORCH_CHECK(out.sizes() == inp.sizes(), "out and inp must have the same shape");
    TORCH_CHECK(out.scalar_type() == inp.scalar_type(), "out and inp must share a dtype");
    TORCH_CHECK(blocks > 0 && blocks <= kMaxBlocks, "blocks must be in [1, ", kMaxBlocks, "]");
    TORCH_CHECK(threads > 0 && threads <= 512, "threads must be in [1, 512]");

    const int lanes = 16 / static_cast<int>(inp.element_size());
    TORCH_CHECK(inp.numel() % lanes == 0, "numel ", inp.numel(),
                " must be a multiple of ", lanes, " for 16-byte vectorized access");
    const int n = static_cast<int>(inp.numel() / lanes);

    // TWO-SHOT PUBLISHES ITS SLICE IN THE SCRATCH, so the scratch bounds the buffer it can
    // reduce: ceil(n/ngpus) vectors of 16 bytes. Checked HERE, where the tensor can be named,
    // rather than discovered as a peer reading past the end of an IPC mapping.
    if (algo == kAlgoTwoShot) {
      const int64_t chunk = (static_cast<int64_t>(n) + world_size_ - 1) / world_size_;
      TORCH_CHECK(chunk * 16 <= scratch_bytes_, "hip_comms: two_shot needs ", chunk * 16,
                  " scratch bytes for a ", inp.numel() * inp.element_size(),
                  "-byte buffer across ", world_size_, " ranks, but only ", scratch_bytes_,
                  " were allocated. Raise HipTunables.scratch_bytes.");
    }
    PeerPtrs* slot = slot_for(inp.data_ptr());
    dispatch(out, algo, blocks, threads, slot, n);
  }

  // `out` must be rank-major with ngpus x inp.numel() elements; the caller reshapes.
  // FUSED: sum across ranks, add the residual, normalise. `out` is the normed result and
  // `residual_out` the sum-plus-residual the next block needs -- both are real outputs, so
  // nothing here is scratch.
  //
  // ONE-SHOT ONLY, and it says so rather than silently picking. Two-shot leaves each rank
  // holding one SLICE of the row after its reduce-scatter, and a row's variance needs the
  // whole row -- so a fused two-shot wants another reduction of the partial sums of squares
  // between the two phases. That is a real design and it is not this one.
  void all_reduce_rmsnorm(torch::Tensor& out, torch::Tensor& residual_out,
                          torch::Tensor& inp, torch::Tensor& residual,
                          torch::Tensor& weight, double eps, int64_t blocks,
                          int64_t threads) {
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
    TORCH_CHECK(blocks > 0 && blocks <= kMaxBlocks, "blocks must be in [1, ", kMaxBlocks, "]");
    TORCH_CHECK(threads > 0 && threads <= 512, "threads must be in [1, 512]");

    const int lanes = 16 / static_cast<int>(inp.element_size());
    const int64_t hidden = inp.size(1);
    // A BLOCK OWNS A ROW, so a row has to divide into whole 16-byte packs. Refused here,
    // where the tensor can be named, rather than by a partial pack read past the end.
    TORCH_CHECK(hidden % lanes == 0, "hidden ", hidden, " must be a multiple of ", lanes,
                " for 16-byte vectorized access");
    const int rows  = static_cast<int>(inp.size(0));
    const int packs = static_cast<int>(hidden / lanes);

    PeerPtrs* slot = slot_for(inp.data_ptr());
    auto stream    = at::cuda::getCurrentCUDAStream();
    // ONE BLOCK PER ROW, capped by what was asked for: a grid wider than the rows leaves
    // blocks with nothing to do and still pays both barriers.
    const int grid = static_cast<int>(std::min<int64_t>(blocks, rows));

#define LAUNCH_FUSED(T, NG)                                                             \
  one_shot_all_reduce_rmsnorm<T, NG><<<dim3(grid), dim3(threads), 0, stream>>>(         \
      slot, peer_signals_, self_signal_, out.data_ptr<T>(),                             \
      residual_out.data_ptr<T>(), residual.data_ptr<T>(), weight.data_ptr<T>(),         \
      static_cast<float>(eps), rank_, rows, packs)

#define FUSED_BY_NGPUS(T)                                                               \
  switch (world_size_) {                                                                \
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
    throw std::runtime_error("hip_comms: world_size " + std::to_string(world_size_) +
                             " not built. Built: 2, 4, 8.");
#undef FUSED_BY_NGPUS
#undef LAUNCH_FUSED
  }

 private:
  // Open every rank's handle into a local pointer. Our OWN handle is never opened --
  // hipIpcOpenMemHandle refuses a self-handle -- so the local pointer is used directly.
  std::vector<void*> open_peers(const std::vector<std::string>& handles,
                                const std::vector<int64_t>& offsets, uintptr_t self) {
    std::vector<void*> out(world_size_, nullptr);
    for (int i = 0; i < world_size_; ++i) {
      if (i == rank_) {
        out[i] = reinterpret_cast<void*>(self);
        continue;
      }
      // Once per distinct handle. A capture-heavy run exchanges the same peer BASES over
      // and over -- one per graph, per rank -- and hipIpcOpenMemHandle on a handle this
      // process already mapped is not a second mapping to close later. vLLM's
      // CustomAllreduce keeps the same cache, keyed the same way.
      auto it = opened_.find(handles[i]);
      if (it == opened_.end()) {
        void* base = nullptr;
        HIP_CHECK(hipIpcOpenMemHandle(&base, handle_from(handles[i]),
                                      hipIpcMemLazyEnablePeerAccess));
        it = opened_.emplace(handles[i], base).first;
      }
      out[i] = static_cast<char*>(it->second) + offsets[i];
    }
    return out;
  }

  PeerPtrs* commit(const std::vector<void*>& ptrs) {
    PeerPtrs* slot = next_slot();
    write_slot(slot, ptrs);
    return slot;
  }

  PeerPtrs* next_slot() {
    if (cursor_ >= slab_end_)
      throw std::runtime_error(
          "hip_comms: peer-pointer slab is full; allocate a larger one");
    return cursor_++;
  }

  void write_slot(PeerPtrs* slot, const std::vector<void*>& ptrs) {
    PeerPtrs host{};
    for (int i = 0; i < world_size_; ++i) host.p[i] = ptrs[i];
    HIP_CHECK(hipMemcpy(slot, &host, sizeof(PeerPtrs), hipMemcpyHostToDevice));
  }

  PeerPtrs* slot_for(void* input) {
    hipStreamCaptureStatus status;
    HIP_CHECK(hipStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &status));
    if (status == hipStreamCaptureStatusActive) {
      // A fresh slot ALWAYS, even for an address `registered_` already knows. An address is
      // only as durable as the allocation under it: a graph's buffers are freed when the
      // graph dies and the allocator hands the same address back for the next one. Skipping
      // the record here would make the recorded COUNT depend on that luck -- and what
      // follows a capture is a COLLECTIVE exchange, so ranks that skip differently do not
      // merely disagree, they exchange the wrong number of handles.
      PeerPtrs* slot = next_slot();
      pending_.push_back(input);
      pending_slots_.push_back(slot);
      return slot;
    }
    auto it = registered_.find(input);
    if (it == registered_.end()) {
      std::ostringstream os;
      os << "hip_comms: buffer " << input
         << " is not registered. Register it, or call inside a cudagraph capture (where "
            "registration is deferred until after capture).";
      throw std::runtime_error(os.str());
    }
    return it->second;
  }

  // THE instantiation menu. Every combination that exists is named here exactly once, so
  // an unsupported request is a listed refusal rather than a wrong kernel.
  void dispatch(torch::Tensor& out, int64_t algo, int64_t blocks, int64_t threads,
                PeerPtrs* slot, int n) {
    auto stream = at::cuda::getCurrentCUDAStream();
  // THE ALGO PICKS THE KERNEL, and there is nothing else it picks: this backend does
  // all-reduce. all_gather was here and moved to `torch.distributed` (2026-09-24) --
  // it was a second op to keep correct for something nothing in the decode path we
  // study ever called, and the two-shot phase that gathers is internal to that kernel.
#define LAUNCH(T, NG)                                                                   \
  do {                                                                                  \
    if (algo == kAlgoTwoShot)                                                           \
      two_shot_all_reduce<T, NG><<<dim3(blocks), dim3(threads), 0, stream>>>(           \
          slot, peer_signals_, self_signal_, out.data_ptr<T>(), rank_, n);              \
    else                                                                                \
      one_shot_all_reduce<T, NG><<<dim3(blocks), dim3(threads), 0, stream>>>(           \
          slot, peer_signals_, self_signal_, out.data_ptr<T>(), rank_, n);              \
  } while (0)

#define BY_NGPUS(T)                                                                     \
  switch (world_size_) {                                                                \
    case 2: LAUNCH(T, 2); return;                                                       \
    case 4: LAUNCH(T, 4); return;                                                       \
    case 8: LAUNCH(T, 8); return;                                                       \
    default: break;                                                                     \
  }

    if (algo != kAlgoOneShot && algo != kAlgoTwoShot)
      throw std::runtime_error("hip_comms: algo " + std::to_string(algo) +
                               " is not built. Built: 0 (one_shot), 1 (two_shot).");
    switch (out.scalar_type()) {
      case at::ScalarType::Half: BY_NGPUS(at::Half) break;
      case at::ScalarType::BFloat16: BY_NGPUS(at::BFloat16) break;
      default:
        throw std::runtime_error("hip_comms: dtype not built. Built: float16, bfloat16.");
    }
    throw std::runtime_error("hip_comms: world_size " + std::to_string(world_size_) +
                             " not built. Built: 2, 4, 8.");
#undef BY_NGPUS
#undef LAUNCH
  }

  static constexpr int64_t kAlgoOneShot = 0;
  static constexpr int64_t kAlgoTwoShot = 1;

  int rank_;
  int world_size_;
  Signal* self_signal_;
  int64_t scratch_bytes_;
  PeerSignals peer_signals_{};
  PeerPtrs* slab_;
  PeerPtrs* slab_end_;
  PeerPtrs* cursor_;
  std::unordered_map<void*, PeerPtrs*> registered_;
  std::vector<void*> pending_;
  std::vector<PeerPtrs*> pending_slots_;
  std::unordered_map<std::string, void*> opened_;
};

}  // namespace hip_comms

// =================================================================================
// THE TORCH OP BOUNDARY. `Comms` is a stateful C++ object and a torch op is a free
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
  auto* comms = new hip_comms::Comms(
      static_cast<int>(rank), static_cast<int>(world_size),
      static_cast<uintptr_t>(self_signal), bytes_of(signal_handles), signal_offsets,
      static_cast<uintptr_t>(peer_slab), peer_slab_bytes, scratch_bytes);
  return reinterpret_cast<fptr_t>(comms);
}

void rocm_comms_dispose(fptr_t comms) {
  delete reinterpret_cast<hip_comms::Comms*>(comms);
}

void rocm_comms_register_buffer(fptr_t comms,
                                const std::vector<std::vector<int64_t>>& handles,
                                const std::vector<int64_t>& offsets, int64_t self_ptr) {
  reinterpret_cast<hip_comms::Comms*>(comms)->register_buffer(
      bytes_of(handles), offsets, static_cast<uintptr_t>(self_ptr));
}

std::vector<int64_t> rocm_comms_pending_graph_buffers(fptr_t comms) {
  auto pending =
      reinterpret_cast<hip_comms::Comms*>(comms)->pending_graph_buffers();
  return std::vector<int64_t>(pending.begin(), pending.end());
}

// ONE ENTRY PER PENDING BUFFER, each the WORLD'S handles for it laid end to end: a
// schema nests two deep and this needs three (buffer, rank, byte), so the innermost
// level is split back out here by the handle size, which is fixed.
void rocm_comms_register_graph_buffers(
    fptr_t comms, const std::vector<std::vector<int64_t>>& handles,
    const std::vector<std::vector<int64_t>>& offsets) {
  const size_t stride = sizeof(hip_comms::Handle);
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
  reinterpret_cast<hip_comms::Comms*>(comms)->register_graph_buffers(bytes, offsets);
}

int64_t rocm_comms_pending_count(fptr_t comms) {
  return reinterpret_cast<hip_comms::Comms*>(comms)->pending_count();
}

void rocm_comms_all_reduce(fptr_t comms, torch::Tensor& out, torch::Tensor& inp,
                           int64_t algo, int64_t blocks, int64_t threads) {
  reinterpret_cast<hip_comms::Comms*>(comms)->all_reduce(out, inp, algo, blocks, threads);
}

void rocm_comms_all_reduce_rmsnorm(fptr_t comms, torch::Tensor& out,
                                   torch::Tensor& residual_out, torch::Tensor& inp,
                                   torch::Tensor& residual, torch::Tensor& weight,
                                   double eps, int64_t blocks, int64_t threads) {
  reinterpret_cast<hip_comms::Comms*>(comms)->all_reduce_rmsnorm(
      out, residual_out, inp, residual, weight, eps, blocks, threads);
}

std::tuple<std::vector<int64_t>, int64_t> rocm_comms_handle_and_offset(int64_t ptr) {
  auto [handle, offset] = hip_comms::ipc_handle_and_offset(static_cast<uintptr_t>(ptr));
  std::vector<int64_t> bytes(handle.begin(), handle.end());
  return std::make_tuple(bytes, offset);
}

// The sizes Python needs to allocate the signal block and the peer slab, and the bounds it
// checks a world size and a launch against. Constants of the kernel, so they are asked for
// rather than restated.
std::vector<int64_t> rocm_comms_sizes() {
  return {static_cast<int64_t>(sizeof(hip_comms::Signal)),
          static_cast<int64_t>(sizeof(hip_comms::PeerPtrs)),
          static_cast<int64_t>(hip_comms::kMaxBlocks),
          static_cast<int64_t>(hip_comms::kMaxRanks),
          static_cast<int64_t>(sizeof(hipIpcMemHandle_t))};
}
