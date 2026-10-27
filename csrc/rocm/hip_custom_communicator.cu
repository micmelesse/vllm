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
// THE TORCH OP BOUNDARY. Nothing crosses but values and handles: Python owns the communicator's
// memory (three buffers a rank: shared, private, host) as addresses, and its state and its configs
// (plain structs) as handles, makes and frees each through the functions below, and hands each op
// its state, its config and its tensors. C++ keeps nothing between calls Python holds no handle
// to.
// =================================================================================

// EACH OP TAKES ITS COMMUNICATOR'S STATE AND ITS CONFIG AS HANDLES (below), then its tensors.
// Nothing is optional and nothing is picked here: Python fills what a caller leaves out, makes the
// config, and hands its table's pick the same way.

namespace {
hip_custom_communicator::DType dtype_of(const torch::Tensor& t) {
  const at::ScalarType s = t.scalar_type();
  if (s == at::ScalarType::Half) return hip_custom_communicator::DType::f16;
  if (s == at::ScalarType::BFloat16)
    return hip_custom_communicator::DType::bf16;
  TORCH_CHECK(s == at::ScalarType::Float, "hip_custom_communicator: dtype ", s,
              " not built");
  return hip_custom_communicator::DType::f32;
}

// THE CONFIG ACROSS THE BOUNDARY, every field required (optionality is Python's): a name to its
// enum (another name is the caller's mistake, raised), and a field narrowed to an int.
int quantization_from(const std::string& s) {
  if (s == "exact") return hip_custom_communicator::Quantization::exact;
  TORCH_CHECK(s == "int4", "hip_custom_communicator: quantization is exact or int4");
  return hip_custom_communicator::Quantization::int4;
}
int narrowed(int64_t v) {
  TORCH_CHECK(v >= std::numeric_limits<int>::min() && v <= std::numeric_limits<int>::max(),
              "hip_custom_communicator: a config field must fit an int");
  return static_cast<int>(v);
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
using BuildInfoWire =
    std::tuple<Names, std::vector<int64_t>, int64_t, Names, Names, Names, Names,
               Names, Names, std::vector<int64_t>, std::vector<int64_t>, Names,
               Names, std::vector<int64_t>, int64_t, Names>;
// AN OP'S ANSWER: -1 when it ran, else the Error's number it was refused for. What
// ran is what Python passed: C++ runs the config it is given, or refuses it.
using Answer = int64_t;
Answer refusal(hip_custom_communicator::Error e) { return static_cast<int64_t>(e); }
Answer answer(const std::variant<hip_custom_communicator::Config,
                                 hip_custom_communicator::Error>& got) {
  if (const auto* e = std::get_if<hip_custom_communicator::Error>(&got))
    return refusal(*e);
  return -1;
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

// WHAT A COMMUNICATOR SERVING `max_rows` ROWS OF AT MOST `max_cols` COLUMNS AT
// `world` RANKS HOLDS, for Python's `should`: the box, its exchange's partials
// and inputs bytes (`exchange_sizes`, what the opener allocates). A pure function of its
// arguments, so a communicator that is not open can be asked.
std::vector<int64_t> hip_custom_communicator_capacity(int64_t max_rows,
                                                      int64_t max_cols,
                                                      int64_t world) {
  const auto sizes = hip_custom_communicator::exchange_sizes(
      max_rows, max_cols, static_cast<int>(world));
  std::vector<int64_t> out{max_rows, max_cols, sizes.partials_bytes,
                           sizes.staging_bytes};
  for (const auto& o : hip_custom_communicator::kOps)
    out.push_back(hip_custom_communicator::max_bytes(
        o.type, sizes.partials_bytes, sizes.staging_bytes));
  return out;
}

// =================================================================================
// THE MEMORY, Python's: allocated, exchanged, read, written and freed here, every
// address an int Python holds. Nothing here keeps one.
// =================================================================================

// THE LAYOUT of a communicator serving `max_rows` rows of at most `max_cols` columns at `world`
// ranks, flat, in this order (Python's `Layout`): the staging and partials bytes
// (`exchange_sizes`); the shared buffer's bytes and its regions' offsets (`arena_layout`): signals,
// staging, partials, scales, unfenced, and each guard's [begin, end); the signals' bytes and the
// guard word; the private buffer's bytes and its offsets (kPrivate*): the region arrays, the epochs,
// the tables; the region arrays' count, kMaxRanks, kMaxBlocks, the table slots, the status words.
std::vector<int64_t> hip_custom_communicator_layout(int64_t max_rows, int64_t max_cols,
                                                    int64_t world) {
  using namespace hip_custom_communicator;
  TORCH_CHECK(max_rows > 0 && max_cols > 0,
              "the largest call a communicator serves has rows and columns: max_rows=", max_rows,
              " max_cols=", max_cols);
  const auto sizes = exchange_sizes(max_rows, max_cols, static_cast<int>(world));
  const ArenaLayout l = arena_layout(sizes.staging_bytes, sizes.partials_bytes,
                                     max_rows * int64_t{sizeof(float)},
                                     unfenced_bytes(static_cast<int>(world)));
  return {sizes.staging_bytes, sizes.partials_bytes,
          l.end, l.signals, l.staging, l.partials, l.scales, l.unfenced,
          l.staging_guard, l.staging_guard_end, l.partials_guard, l.partials_guard_end,
          l.unfenced_guard, l.unfenced_guard_end,
          kSignalsBytes, int64_t{kGuardWord},
          kPrivateBytes, kPrivateRegionPtrArrays, kPrivateUnfencedEpochs, kPrivateTables,
          int64_t{kRegionPtrArrays}, int64_t{kMaxRanks}, int64_t{kMaxBlocks}, kInputTableSlots,
          int64_t{kStatusFields}};
}

// A DEVICE BUFFER of `bytes` on the current device, zeroed before any stream can use it: SHARED
// (`shared`: uncached, kArenaMallocFlags, for IPC) or PRIVATE (hipMalloc). Its address.
int64_t hip_custom_communicator_alloc(int64_t bytes, bool shared) {
  void* p = nullptr;
  if (shared)
    HIP_CHECK(hipExtMallocWithFlags(&p, static_cast<size_t>(bytes),
                                    hip_custom_communicator::kArenaMallocFlags));
  else
    HIP_CHECK(hipMalloc(&p, static_cast<size_t>(bytes)));
  HIP_CHECK(hipMemset(p, 0, static_cast<size_t>(bytes)));
  HIP_CHECK(hipDeviceSynchronize());
  return reinterpret_cast<int64_t>(p);
}
// A device buffer freed: 0, or the hipError_t (a teardown carries on past one).
int64_t hip_custom_communicator_free(int64_t ptr) {
  const hipError_t e = hipFree(reinterpret_cast<void*>(ptr));
  if (e != hipSuccess) (void)hipGetLastError();
  return static_cast<int64_t>(e);
}
// HOST MEMORY of `bytes`, mapped and coherent (the kernels write it with system-scope atomics while
// the host polls it), zeroed: [its host address, its device address].
std::vector<int64_t> hip_custom_communicator_alloc_host(int64_t bytes) {
  void* host = nullptr;
  HIP_CHECK(hipHostMalloc(&host, static_cast<size_t>(bytes),
                          hipHostMallocMapped | hipHostMallocCoherent));
  std::memset(host, 0, static_cast<size_t>(bytes));
  void* device = nullptr;
  HIP_CHECK(hipHostGetDevicePointer(&device, host, 0));
  return {reinterpret_cast<int64_t>(host), reinterpret_cast<int64_t>(device)};
}
int64_t hip_custom_communicator_free_host(int64_t host) {
  const hipError_t e = hipHostFree(reinterpret_cast<void*>(host));
  if (e != hipSuccess) (void)hipGetLastError();
  return static_cast<int64_t>(e);
}

// THE IPC HANDLE of the allocation `ptr` lies in, and `ptr`'s offset in it: the handle's bytes, one
// an int, then the offset (hipIpcGetMemHandle takes an allocation's base; a torch tensor sits at an
// offset in one).
std::vector<int64_t> hip_custom_communicator_ipc_handle(int64_t ptr) {
  void* base = nullptr;
  HIP_CHECK(hipPointerGetAttribute(&base, HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                                   static_cast<hipDeviceptr_t>(reinterpret_cast<void*>(ptr))));
  hipIpcMemHandle_t h;
  HIP_CHECK(hipIpcGetMemHandle(&h, base));
  std::vector<int64_t> out;
  for (size_t i = 0; i < sizeof(h); ++i)
    out.push_back(static_cast<uint8_t>(reinterpret_cast<const char*>(&h)[i]));
  out.push_back(ptr - reinterpret_cast<int64_t>(base));
  return out;
}
// A PEER'S ALLOCATION opened from its handle (`ipc_handle`'s bytes, its offset dropped): its base as
// this process reaches it. BORROWED: closed by `ipc_close`, never freed.
int64_t hip_custom_communicator_ipc_open(const std::vector<int64_t>& handle) {
  hipIpcMemHandle_t h;
  TORCH_CHECK(handle.size() >= sizeof(h), "an IPC handle is ", sizeof(h), " bytes");
  for (size_t i = 0; i < sizeof(h); ++i)
    reinterpret_cast<char*>(&h)[i] = static_cast<char>(handle[i]);
  void* base = nullptr;
  HIP_CHECK(hipIpcOpenMemHandle(&base, h, hipIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<int64_t>(base);
}
int64_t hip_custom_communicator_ipc_close(int64_t base) {
  const hipError_t e = hipIpcCloseMemHandle(reinterpret_cast<void*>(base));
  if (e != hipSuccess) (void)hipGetLastError();
  return static_cast<int64_t>(e);
}
// The device `ptr` is memory of, where it is the base of its own device allocation; -1 otherwise.
int64_t hip_custom_communicator_pointer_device(int64_t ptr) {
  hipPointerAttribute_t a{};
  const hipError_t e = hipPointerGetAttributes(&a, reinterpret_cast<void*>(ptr));
  if (e != hipSuccess) {
    (void)hipGetLastError();
    return -1;
  }
  if (a.type != hipMemoryTypeDevice || a.devicePointer != reinterpret_cast<void*>(ptr)) return -1;
  return a.device;
}

// DEVICE WORDS: 64-bit `words` written at `ptr` (the region arrays, a table); `count` 32-bit words
// at `ptr` set to `word` (the guards); `count` 32-bit words read back (the guards, the signals).
// Each synchronous: the caller has every launch that touches them finished, or none started.
void hip_custom_communicator_write_words(int64_t ptr, const std::vector<int64_t>& words) {
  HIP_CHECK(hipMemcpy(reinterpret_cast<void*>(ptr), words.data(), words.size() * sizeof(int64_t),
                      hipMemcpyHostToDevice));
}
void hip_custom_communicator_fill_words(int64_t ptr, int64_t word, int64_t count) {
  HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(ptr), static_cast<int>(word),
                         static_cast<size_t>(count)));
  HIP_CHECK(hipDeviceSynchronize());
}
std::vector<int64_t> hip_custom_communicator_read_words(int64_t ptr, int64_t count) {
  std::vector<uint32_t> words(static_cast<size_t>(count));
  HIP_CHECK(hipMemcpy(words.data(), reinterpret_cast<void*>(ptr), words.size() * sizeof(uint32_t),
                      hipMemcpyDeviceToHost));
  return std::vector<int64_t>(words.begin(), words.end());
}
// HOST WORDS, while kernels run: `count` 32-bit words at `host` read (acquire: the status, the fault
// whole once `state` reads written), and one set (release: the abort every wait leaves at).
std::vector<int64_t> hip_custom_communicator_read_host(int64_t host, int64_t count) {
  const auto* words = reinterpret_cast<const uint32_t*>(host);
  std::vector<int64_t> out;
  for (int64_t i = 0; i < count; ++i) out.push_back(__atomic_load_n(&words[i], __ATOMIC_ACQUIRE));
  return out;
}
void hip_custom_communicator_write_host(int64_t host, int64_t index, int64_t word) {
  __atomic_store_n(&reinterpret_cast<uint32_t*>(host)[index], static_cast<uint32_t>(word),
                   __ATOMIC_RELEASE);
}

// The device clock's rate, kHz: what its ticks are per ms.
int64_t hip_custom_communicator_wall_clock_khz(int64_t device) {
  int khz = 0;
  HIP_CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate,
                                  static_cast<int>(device)));
  return khz;
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
void refuse_overlap(std::initializer_list<const torch::Tensor*> inputs,
                    std::initializer_list<const torch::Tensor*> outputs,
                    bool in_place = false) {
  for (const torch::Tensor* i : inputs)
    for (const torch::Tensor* o : outputs) {
      if (i == nullptr || o == nullptr || i->numel() == 0 || o->numel() == 0)
        continue;
      if (in_place && same_tensor(*i, *o)) continue;
      TORCH_CHECK(
          !overlap(span_of(*i), span_of(*o)),
          in_place ? "an input and an output share memory, not as the same tensor; pass it "
                     "whole or a separate output"
                   : "an input and an output share memory; pass a separate output");
    }
}
}  // namespace

// =================================================================================================
// THE STATE AND THE CONFIGS, AS HANDLES: each a plain struct of the API's, made here and handed to
// Python as its address (an int, as vLLM's custom all-reduce hands its own), passed back to every
// op's call, and freed here when Python is done with it. C++ keeps none: Python owns every
// lifetime. Each op's config is its own type, made by its own function from its own fields.
// =================================================================================================

// THE COMMUNICATOR'S STATE (the API's CommsState), made once its buffers are: its handle.
int64_t hip_custom_communicator_state(int64_t rank, int64_t world, int64_t shared_buffer,
                                      int64_t private_buffer, int64_t status_device,
                                      int64_t timeout_ticks, int64_t max_rows,
                                      int64_t staging_bytes, int64_t partials_bytes,
                                      bool checked_after_launch) {
  TORCH_CHECK(world >= 1 && rank >= 0 && rank < world,
              "hip_custom_communicator: rank ", rank, " is not one of world ", world, "'s");
  return reinterpret_cast<int64_t>(new hip_custom_communicator::CommsState{
      .rank = narrowed(rank),
      .world = narrowed(world),
      .shared_buffer = reinterpret_cast<char*>(shared_buffer),
      .private_buffer = reinterpret_cast<char*>(private_buffer),
      .status_device = reinterpret_cast<uint32_t*>(status_device),
      .timeout_ticks = static_cast<uint64_t>(timeout_ticks),
      .max_rows = max_rows,
      .staging_bytes = staging_bytes,
      .partials_bytes = partials_bytes,
      .checked_after_launch = checked_after_launch});
}
void hip_custom_communicator_free_state(int64_t state) {
  delete reinterpret_cast<hip_custom_communicator::CommsState*>(state);
}

// A CONFIG AT THE TORCH BOUNDARY: a torch op cannot return a variant, so it is two optionals and
// exactly one is set, the config's handle or the Error's number it is refused for.
using ConfigWire = std::tuple<std::optional<int64_t>, std::optional<int64_t>>;

namespace {
hip_custom_communicator::Dim3 dim3_from(int64_t x, int64_t y, int64_t z) {
  return {narrowed(x), narrowed(y), narrowed(z)};
}
ConfigWire refused_config(hip_custom_communicator::Error e) {
  return {std::nullopt, static_cast<int64_t>(e)};
}
// A CHECKED CONFIG's handle (a copy of it, Python's to free), or the Error it is refused for.
template <typename B>
ConfigWire handle_of(
    const std::variant<hip_custom_communicator::OpConfig<B>, hip_custom_communicator::Error>&
        got) {
  if (const auto* e = std::get_if<hip_custom_communicator::Error>(&got))
    return refused_config(*e);
  return {reinterpret_cast<int64_t>(new hip_custom_communicator::OpConfig<B>(
              std::get<hip_custom_communicator::OpConfig<B>>(got))),
          std::nullopt};
}
}  // namespace

// all_reduce's config: its algorithm by name, the grid, the block and its waves a EU, the tile,
// what it moves, the pack, and the bytes an INT4 build quantizes from.
ConfigWire hip_custom_communicator_all_reduce_config(
    const std::string& algorithm, int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t block_x,
    int64_t block_y, int64_t block_z, int64_t waves_per_eu, int64_t tile_m, int64_t tile_n,
    const std::string& quantization, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  using hip_custom_communicator::AllReduceBuildConfig;
  const auto named = hip_custom_communicator::algorithm_named<AllReduceBuildConfig>(algorithm);
  if (!named) return refused_config(hip_custom_communicator::Error::no_such_kernel);
  const AllReduceBuildConfig build{.algorithm = *named,
                                   .block = dim3_from(block_x, block_y, block_z),
                                   .waves_per_eu = narrowed(waves_per_eu),
                                   .tile_m = narrowed(tile_m),
                                   .tile_n = narrowed(tile_n),
                                   .quantization = quantization_from(quantization),
                                   .pack_bytes = narrowed(pack_bytes)};
  return handle_of(hip_custom_communicator::config_of(
      build, dim3_from(grid_x, grid_y, grid_z), quantize_from_bytes));
}
void hip_custom_communicator_free_all_reduce_config(int64_t config) {
  delete reinterpret_cast<hip_custom_communicator::AllReduceConfig*>(config);
}

// all_reduce_rms_norm's config: all_reduce's fields.
ConfigWire hip_custom_communicator_all_reduce_rms_norm_config(
    const std::string& algorithm, int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t block_x,
    int64_t block_y, int64_t block_z, int64_t waves_per_eu, int64_t tile_m, int64_t tile_n,
    const std::string& quantization, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  using hip_custom_communicator::AllReduceRmsNormBuildConfig;
  const auto named =
      hip_custom_communicator::algorithm_named<AllReduceRmsNormBuildConfig>(algorithm);
  if (!named) return refused_config(hip_custom_communicator::Error::no_such_kernel);
  const AllReduceRmsNormBuildConfig build{.algorithm = *named,
                                          .block = dim3_from(block_x, block_y, block_z),
                                          .waves_per_eu = narrowed(waves_per_eu),
                                          .tile_m = narrowed(tile_m),
                                          .tile_n = narrowed(tile_n),
                                          .quantization = quantization_from(quantization),
                                          .pack_bytes = narrowed(pack_bytes)};
  return handle_of(hip_custom_communicator::config_of(
      build, dim3_from(grid_x, grid_y, grid_z), quantize_from_bytes));
}
void hip_custom_communicator_free_all_reduce_rms_norm_config(int64_t config) {
  delete reinterpret_cast<hip_custom_communicator::AllReduceRmsNormConfig*>(config);
}

// all_reduce_add_rms_norm's config: all_reduce's fields.
ConfigWire hip_custom_communicator_all_reduce_add_rms_norm_config(
    const std::string& algorithm, int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t block_x,
    int64_t block_y, int64_t block_z, int64_t waves_per_eu, int64_t tile_m, int64_t tile_n,
    const std::string& quantization, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  using hip_custom_communicator::AllReduceAddRmsNormBuildConfig;
  const auto named =
      hip_custom_communicator::algorithm_named<AllReduceAddRmsNormBuildConfig>(algorithm);
  if (!named) return refused_config(hip_custom_communicator::Error::no_such_kernel);
  const AllReduceAddRmsNormBuildConfig build{.algorithm = *named,
                                             .block = dim3_from(block_x, block_y, block_z),
                                             .waves_per_eu = narrowed(waves_per_eu),
                                             .tile_m = narrowed(tile_m),
                                             .tile_n = narrowed(tile_n),
                                             .quantization = quantization_from(quantization),
                                             .pack_bytes = narrowed(pack_bytes)};
  return handle_of(hip_custom_communicator::config_of(
      build, dim3_from(grid_x, grid_y, grid_z), quantize_from_bytes));
}
void hip_custom_communicator_free_all_reduce_add_rms_norm_config(int64_t config) {
  delete reinterpret_cast<hip_custom_communicator::AllReduceAddRmsNormConfig*>(config);
}

// all_reduce_add_attn_res_rms_norm's config: all_reduce's fields and tile_k, its sources a step.
ConfigWire hip_custom_communicator_all_reduce_add_attn_res_rms_norm_config(
    const std::string& algorithm, int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t block_x,
    int64_t block_y, int64_t block_z, int64_t waves_per_eu, int64_t tile_m, int64_t tile_n,
    int64_t tile_k, const std::string& quantization, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  using hip_custom_communicator::AllReduceAddAttnResRmsNormBuildConfig;
  const auto named =
      hip_custom_communicator::algorithm_named<AllReduceAddAttnResRmsNormBuildConfig>(algorithm);
  if (!named) return refused_config(hip_custom_communicator::Error::no_such_kernel);
  const AllReduceAddAttnResRmsNormBuildConfig build{
      .algorithm = *named,
      .block = dim3_from(block_x, block_y, block_z),
      .waves_per_eu = narrowed(waves_per_eu),
      .tile_m = narrowed(tile_m),
      .tile_n = narrowed(tile_n),
      .tile_k = narrowed(tile_k),
      .quantization = quantization_from(quantization),
      .pack_bytes = narrowed(pack_bytes)};
  return handle_of(hip_custom_communicator::config_of(
      build, dim3_from(grid_x, grid_y, grid_z), quantize_from_bytes));
}
void hip_custom_communicator_free_all_reduce_add_attn_res_rms_norm_config(int64_t config) {
  delete reinterpret_cast<hip_custom_communicator::AllReduceAddAttnResRmsNormConfig*>(config);
}

// rms_norm's config (its kernel on this rank's rows, one algorithm: none named): the grid, the
// block and its waves a EU, the tile, what it moves, the pack, and the bytes it quantizes from.
ConfigWire hip_custom_communicator_rms_norm_config(
    int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t block_x, int64_t block_y,
    int64_t block_z, int64_t waves_per_eu, int64_t tile_m, int64_t tile_n,
    const std::string& quantization, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  const hip_custom_communicator::RmsNormBuildConfig build{
      .block = dim3_from(block_x, block_y, block_z),
      .waves_per_eu = narrowed(waves_per_eu),
      .tile_m = narrowed(tile_m),
      .tile_n = narrowed(tile_n),
      .quantization = quantization_from(quantization),
      .pack_bytes = narrowed(pack_bytes)};
  return handle_of(hip_custom_communicator::config_of(
      build, dim3_from(grid_x, grid_y, grid_z), quantize_from_bytes));
}
void hip_custom_communicator_free_rms_norm_config(int64_t config) {
  delete reinterpret_cast<hip_custom_communicator::RmsNormConfig*>(config);
}

// add_rms_norm's config: rms_norm's fields.
ConfigWire hip_custom_communicator_add_rms_norm_config(
    int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t block_x, int64_t block_y,
    int64_t block_z, int64_t waves_per_eu, int64_t tile_m, int64_t tile_n,
    const std::string& quantization, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  const hip_custom_communicator::AddRmsNormBuildConfig build{
      .block = dim3_from(block_x, block_y, block_z),
      .waves_per_eu = narrowed(waves_per_eu),
      .tile_m = narrowed(tile_m),
      .tile_n = narrowed(tile_n),
      .quantization = quantization_from(quantization),
      .pack_bytes = narrowed(pack_bytes)};
  return handle_of(hip_custom_communicator::config_of(
      build, dim3_from(grid_x, grid_y, grid_z), quantize_from_bytes));
}
void hip_custom_communicator_free_add_rms_norm_config(int64_t config) {
  delete reinterpret_cast<hip_custom_communicator::AddRmsNormConfig*>(config);
}

// add_attn_res_rms_norm's config: rms_norm's fields and tile_k, its sources a step.
ConfigWire hip_custom_communicator_add_attn_res_rms_norm_config(
    int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t block_x, int64_t block_y,
    int64_t block_z, int64_t waves_per_eu, int64_t tile_m, int64_t tile_n, int64_t tile_k,
    const std::string& quantization, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes) {
  const hip_custom_communicator::AddAttnResRmsNormBuildConfig build{
      .block = dim3_from(block_x, block_y, block_z),
      .waves_per_eu = narrowed(waves_per_eu),
      .tile_m = narrowed(tile_m),
      .tile_n = narrowed(tile_n),
      .tile_k = narrowed(tile_k),
      .quantization = quantization_from(quantization),
      .pack_bytes = narrowed(pack_bytes)};
  return handle_of(hip_custom_communicator::config_of(
      build, dim3_from(grid_x, grid_y, grid_z), quantize_from_bytes));
}
void hip_custom_communicator_free_add_attn_res_rms_norm_config(int64_t config) {
  delete reinterpret_cast<hip_custom_communicator::AddAttnResRmsNormConfig*>(config);
}

// all_reduce: at its config.
Answer hip_custom_communicator_all_reduce(int64_t state, int64_t config, int64_t table_slot,
    torch::Tensor& input, torch::Tensor& output) {
  TORCH_CHECK(state != 0 && config != 0,
              "hip_custom_communicator: a call takes its communicator's state and its op's config");
  const auto& s = *reinterpret_cast<const hip_custom_communicator::CommsState*>(state);
  const auto& c = *reinterpret_cast<const hip_custom_communicator::AllReduceConfig*>(config);
  check_device({&input, &output});
  TORCH_CHECK(output.sizes() == input.sizes(),
              "output and input must have the same shape");
  TORCH_CHECK(output.scalar_type() == input.scalar_type(),
              "output and input must share a dtype");
  refuse_overlap({&input}, {&output});
  const std::optional<torch::Tensor> input_rows = as_rows(input),
                                     output_rows = as_rows(output);
  if (!input_rows || !output_rows)
    return refusal(hip_custom_communicator::Error::not_contiguous);
  const auto got = hip_custom_communicator::all_reduce(
      s, c, table_slot, input_rows->data_ptr(), input_rows->stride(0),
      input_rows->stride(1), input_rows->size(0), input_rows->size(1),
      output_rows->data_ptr(), output_rows->stride(0), output_rows->stride(1),
      output_rows->size(0), output_rows->size(1), dtype_of(input),
      current_stream());
  return answer(got);
}

namespace {
// A NORM CALL'S TENSORS: input [tokens, hidden], output (and with a residual,
// residual and residual_output) its shape and dtype, the weight 1-D of hidden
// (its dtype is select's to refuse).
void norm_tensors(const torch::Tensor& input, const torch::Tensor* residual,
                  const torch::Tensor& weight, const torch::Tensor& output,
                  const torch::Tensor* residual_output, bool in_place) {
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
  refuse_overlap({&input, residual}, {&output, residual_output}, in_place);
  refuse_overlap({&weight}, {&output, residual_output});
}
}  // namespace

// all_reduce_rms_norm: at its config.
Answer hip_custom_communicator_all_reduce_rms_norm(int64_t state, int64_t config,
    int64_t table_slot, torch::Tensor& input, torch::Tensor& weight, torch::Tensor& output,
    double eps) {
  TORCH_CHECK(state != 0 && config != 0,
              "hip_custom_communicator: a call takes its communicator's state and its op's config");
  const auto& s = *reinterpret_cast<const hip_custom_communicator::CommsState*>(state);
  const auto& c = *reinterpret_cast<const hip_custom_communicator::AllReduceRmsNormConfig*>(config);
  norm_tensors(input, nullptr, weight, output, nullptr, /*in_place=*/false);
  const auto got = hip_custom_communicator::all_reduce_rms_norm(
      s, c, table_slot, input.data_ptr(), input.stride(0), input.stride(1),
      input.size(0), input.size(1), weight.data_ptr(), weight.stride(0),
      weight.size(0), output.data_ptr(), output.stride(0), output.stride(1),
      output.size(0), output.size(1), dtype_of(input), dtype_of(weight),
      static_cast<float>(eps),
      current_stream());
  return answer(got);
}

// all_reduce_add_rms_norm: at its config.
Answer hip_custom_communicator_all_reduce_add_rms_norm(int64_t state, int64_t config,
    int64_t table_slot, torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight,
    torch::Tensor& output, torch::Tensor& residual_output, double eps) {
  TORCH_CHECK(state != 0 && config != 0,
              "hip_custom_communicator: a call takes its communicator's state and its op's config");
  const auto& s = *reinterpret_cast<const hip_custom_communicator::CommsState*>(state);
  const auto& c = *reinterpret_cast<const hip_custom_communicator::AllReduceAddRmsNormConfig*>(config);
  norm_tensors(input, &residual, weight, output, &residual_output, /*in_place=*/false);
  const auto got = hip_custom_communicator::all_reduce_add_rms_norm(
      s, c, table_slot, input.data_ptr(), input.stride(0), input.stride(1),
      input.size(0), input.size(1), residual.data_ptr(), residual.stride(0),
      residual.stride(1), residual.size(0), residual.size(1), weight.data_ptr(),
      weight.stride(0), weight.size(0), output.data_ptr(), output.stride(0),
      output.stride(1), output.size(0), output.size(1),
      residual_output.data_ptr(), residual_output.stride(0),
      residual_output.stride(1), residual_output.size(0),
      residual_output.size(1), dtype_of(input), dtype_of(weight),
      static_cast<float>(eps),
      current_stream());
  return answer(got);
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

// all_reduce_add_attn_res_rms_norm: at its config.
Answer hip_custom_communicator_all_reduce_add_attn_res_rms_norm(int64_t state, int64_t config,
    int64_t table_slot, torch::Tensor& input, const std::optional<torch::Tensor>& prefix,
    torch::Tensor& blocks, torch::Tensor& norm_weight, torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& output_norm_weight, torch::Tensor& prefix_output,
    torch::Tensor& output, const std::optional<torch::Tensor>& block_output, int64_t num_blocks,
    double eps, double output_eps) {
  TORCH_CHECK(state != 0 && config != 0,
              "hip_custom_communicator: a call takes its communicator's state and its op's config");
  const auto& s = *reinterpret_cast<const hip_custom_communicator::CommsState*>(state);
  const auto& c = *reinterpret_cast<const hip_custom_communicator::AllReduceAddAttnResRmsNormConfig*>(config);
  attn_res_tensors(input, prefix, blocks, norm_weight, qk_weight,
                   output_norm_weight, prefix_output, output, block_output,
                   num_blocks);
  const Rows old = rows_of(prefix), block = rows_of(block_output);
  const auto got = hip_custom_communicator::all_reduce_add_attn_res_rms_norm(
      s, c, table_slot, input.data_ptr(), input.stride(0), input.stride(1),
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
      current_stream());
  return answer(got);
}

// rms_norm: this rank's rows normed (the second kernel of an op Python composes), at its config.
Answer hip_custom_communicator_rms_norm(int64_t state, int64_t config, torch::Tensor& input,
    torch::Tensor& weight, torch::Tensor& output, double eps) {
  TORCH_CHECK(state != 0 && config != 0,
              "hip_custom_communicator: a call takes its communicator's state and its op's config");
  const auto& s = *reinterpret_cast<const hip_custom_communicator::CommsState*>(state);
  const auto& c = *reinterpret_cast<const hip_custom_communicator::RmsNormConfig*>(config);
  norm_tensors(input, nullptr, weight, output, nullptr, /*in_place=*/true);
  const auto got = hip_custom_communicator::rms_norm(
      s, c, input.data_ptr(), input.stride(0), input.stride(1),
      input.size(0), input.size(1), weight.data_ptr(), weight.stride(0),
      weight.size(0), output.data_ptr(), output.stride(0), output.stride(1),
      output.size(0), output.size(1), dtype_of(input), dtype_of(weight),
      static_cast<float>(eps),
      current_stream());
  return answer(got);
}

// add_rms_norm: this rank's rows added and normed (the second kernel of an op Python composes),
// at its config.
Answer hip_custom_communicator_add_rms_norm(int64_t state, int64_t config, torch::Tensor& input,
    torch::Tensor& residual, torch::Tensor& weight, torch::Tensor& output,
    torch::Tensor& residual_output, double eps) {
  TORCH_CHECK(state != 0 && config != 0,
              "hip_custom_communicator: a call takes its communicator's state and its op's config");
  const auto& s = *reinterpret_cast<const hip_custom_communicator::CommsState*>(state);
  const auto& c = *reinterpret_cast<const hip_custom_communicator::AddRmsNormConfig*>(config);
  norm_tensors(input, &residual, weight, output, &residual_output, /*in_place=*/true);
  const auto got = hip_custom_communicator::add_rms_norm(
      s, c, input.data_ptr(), input.stride(0), input.stride(1),
      input.size(0), input.size(1), residual.data_ptr(), residual.stride(0),
      residual.stride(1), residual.size(0), residual.size(1), weight.data_ptr(),
      weight.stride(0), weight.size(0), output.data_ptr(), output.stride(0),
      output.stride(1), output.size(0), output.size(1),
      residual_output.data_ptr(), residual_output.stride(0),
      residual_output.stride(1), residual_output.size(0),
      residual_output.size(1), dtype_of(input), dtype_of(weight),
      static_cast<float>(eps),
      current_stream());
  return answer(got);
}

// add_attn_res_rms_norm: AttnRes on this rank's rows (the second kernel of the op Python
// composes), at its config.
Answer hip_custom_communicator_add_attn_res_rms_norm(int64_t state, int64_t config,
    torch::Tensor& input, const std::optional<torch::Tensor>& prefix, torch::Tensor& blocks,
    torch::Tensor& norm_weight, torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& output_norm_weight, torch::Tensor& prefix_output,
    torch::Tensor& output, const std::optional<torch::Tensor>& block_output, int64_t num_blocks,
    double eps, double output_eps) {
  TORCH_CHECK(state != 0 && config != 0,
              "hip_custom_communicator: a call takes its communicator's state and its op's config");
  const auto& s = *reinterpret_cast<const hip_custom_communicator::CommsState*>(state);
  const auto& c = *reinterpret_cast<const hip_custom_communicator::AddAttnResRmsNormConfig*>(config);
  attn_res_tensors(input, prefix, blocks, norm_weight, qk_weight,
                   output_norm_weight, prefix_output, output, block_output,
                   num_blocks);
  const Rows old = rows_of(prefix), block = rows_of(block_output);
  const auto got = hip_custom_communicator::add_attn_res_rms_norm(
      s, c, input.data_ptr(), input.stride(0), input.stride(1),
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
      current_stream());
  return answer(got);
}
