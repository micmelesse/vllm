// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE RUNTIME STATE, one `Handle` per communicator: everything the running
// program learns, as common.cuh's `BuildInfo` is everything fixed at
// compile time. OUR OWN MEMORY (the signals and the exchange) is ONE device
// allocation per rank, the arena, shared over legacy HIP IPC: each rank
// exports its arena's base (hipIpcGetMemHandle), the handles go round over the
// caller's CPU group (`Gather`), and each rank opens every peer's
// (hipIpcOpenMemHandle). A peer's arena is at whatever address its opening
// gave, so a kernel takes each region as its own DEVICE ARRAY of every rank's
// address of it, written once at open (`signals_ptr_array`, `staging_ptr_array`,
// `partials_ptr_array`, each with its size, the world); the arena stays the
// host's, for the regions' lifetimes and IPC. AN
// EXACT CALL'S INPUT is read by its peers IN PLACE (aiter's gfx950 pattern),
// through an INPUT TABLE in device memory (INPUT TABLES, below): a captured
// call's table points at every rank's own tensor, registered when the capture
// exits (`register_captured`, one batched exchange over the CPU group); an
// eager call's input is copied by the host into this rank's inputs region on
// the launch's stream and its table, fixed at open, points at every rank's
// inputs region. An INT4 call's kernel quantizes its input into the inputs'
// INT4 payload itself (an encoding, not a copy), before its first barrier.
// The collective
// that exchanges the handles is the caller's (`Gather`), so this names no
// process group.

#pragma once

#include <c10/util/BFloat16.h>
#include <hip/hip_runtime.h>
#include <dirent.h>
#include <unistd.h>

#include <array>
#include <fstream>
#include <map>
#include <memory>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <utility>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "kernels/common.cuh"
#include "log.cuh"
#include "types.cuh"
#include "kernels/all_reduce.cuh"

#define HIP_CHECK(expr)                                                   \
  do {                                                                    \
    hipError_t _e = (expr);                                               \
    if (_e != hipSuccess) {                                               \
      throw std::runtime_error(std::string("hip_custom_communicator: ") + \
                               #expr + " -> " + hipGetErrorString(_e));   \
    }                                                                     \
  } while (0)

namespace hip_custom_communicator {

// EVERY RANK'S `mine`, in rank order, `mine` the same length on every rank: a
// collective, so every rank calls it together.
using Gather = std::function<std::vector<std::string>(const std::string&)>;

// THE SIGNALS' BYTES: two flag sets of kMaxBlocks blocks by kMaxRanks writers,
// rounded to 128 bytes; their region in the arena is padded to 4 KiB
// (kSignalsRegionBytes), so every region after it starts aligned.
constexpr int64_t kSignalsBytes =
    (2 * int64_t{kMaxBlocks} * kMaxRanks * int64_t{sizeof(uint32_t)} + 127) /
    128 * 128;
constexpr int64_t kSignalsRegionBytes = (kSignalsBytes + 4095) / 4096 * 4096;

// HOW THE ARENA IS BACKED: hipExtMallocWithFlags(hipDeviceMallocUncached),
// exported with hipIpcGetMemHandle, opened by each peer with
// hipIpcOpenMemHandle(hipIpcMemLazyEnablePeerAccess). The gfx950 convention,
// which every IPC communicator on ROCm follows:
//   - vLLM's custom all-reduce: allocate_shared_buffer_and_handle
//     (csrc/libtorch_stable/custom_all_reduce.cu) backs EVERY shared buffer,
//     the signals with the reduce's scratch (meta_size() + max_size) and the
//     registered data buffer alike, with hipDeviceMallocUncached ("data
//     buffers need to be uncached for signal on MI200").
//   - vLLM's quick reduce (csrc/quickreduce/quick_reduce.h): ONE
//     hipDeviceMallocUncached allocation per rank, its flags then its data at
//     an offset: our arena's shape exactly.
//   - aiter's custom all-reduce (csrc/kernels/custom_all_reduce.cu,
//     allocate_meta_buffer; aiter/dist/.../custom_all_reduce.py): the meta
//     buffer, its signals and the 2 * max_size it reduces through, uncached;
//     only its input pool (what callers' tensors are copied into, read once
//     per call) cacheable (hipMalloc or torch's allocator).
// So signals and the data peers exchange share one uncached allocation by
// convention; aiter's one cacheable buffer is an input pool, which our inline
// copy-in into the inputs region replaces. ONE ARENA OF EITHER KIND IS CORRECT
// under our kernels' memory model, LLVM's AMDGPUUsage for memory shared across
// agents ("Memory Model GFX942"): every flag store a system-scope atomic,
// released (buffer_wbl2 sc0 sc1) when it publishes data; every flag wait a
// system-scope load, then a system-scope acquire (buffer_inv sc0 sc1) before
// the data is read (common.cuh's BARRIER, the one place a kernel synchronizes
// with a peer). Uncached is the convention, keeps peer reads off every L2 (no
// stale line for an acquire to drop), and does not depend on how a runtime
// backs the flag (hipDeviceMallocUncached was fine-grained under ROCm 7.2 and,
// at 2 MiB and up, coarse-grained under HIP 7.15: aiter 0.1.24's
// custom_all_reduce.py), since the model holds for cacheable memory too.
// WHY NOT VMM (until 2026-10-16 the arena was hipMemCreate'd and mapped):
// on ROCm 10.0 / amdgpu-dkms 6.16.6 (MI355X), exchanging VMM allocations
// totalling more than ~256 MiB per process across 8 processes leaves the
// driver in an eviction storm for every later process until a reboot (healthy
// at 256 MiB, degraded at 288 MiB and at two 192 MiB allocations; legacy IPC
// at 449 MiB healthy): hip.md's VMM section.
constexpr unsigned kArenaMallocFlags = hipDeviceMallocUncached;

// THE ARENA'S LAYOUT, the same offsets on every rank (bytes from the arena's
// base; `arena_layout` computes it from the sizes open is given, config.cuh's
// `exchange_sizes`). A signal pad, then ONE PUBLISHED REGION PER PHASE, whose
// data peers may still read while this rank writes the next phase's:
//
//   region     offset                     bytes (each region kRegionAlign-aligned)
//   signals    0                          kSignalsRegionBytes: the barrier
//                                         flags (torch SymmetricMemory's
//                                         "signal pad")
//   staging    signals end                what this rank publishes of its
//                                         input: an eager exact call's host
//                                         copy (read through the eager input
//                                         table), an INT4 call's payload, or
//                                         the pushed all-reduce's inbox
//   partials   staging end                what this rank publishes of its
//                                         reduced slice for the all-gather
//                                         (two-shot only)
//   scales     partials end               the add-RMS-norm pull two-shot's row
//                                         scales, a float a row (max_rows)
//
// Rank r's region is at its arena's base (wherever its IPC opening put it)
// plus the region's offset, written into the region's device array at open. INT4 IS AN ENCODING, NOT A
// REGION: each region is sized for the largest encoding it may hold
// (config.cuh's `exchange_sizes`: the larger of the exact bytes and the INT4
// values plus scales at the largest problem), and the encoding lays out the
// inside (common.cuh's INT4 payload layout: values, then scales at an offset from
// the call's m and n, the same on every rank).
//
// WHY ONE REGION SERVES EITHER ENCODING: a call publishes its input exact or
// INT4, never both, and its sums exact or INT4, never both. Across calls: a rank writes inputs (its own,
// or, the plain all-reduce, its slot of every peer's inbox) before its first
// barrier, and any rank reads inputs (a peer's, or its own inbox) only before
// its barrier that every peer's sums are visible (or, a one-shot, its last),
// which this rank passed in the previous call only once every peer arrived,
// done reading; partials are
// written (by their rank, or a push kernel's peers) only after the writer's
// first barrier, which every peer reaches only once it finished the previous
// call, its all-gather's reads included. Not every launch ENDS with a barrier
// (a two-shot ends with its all-gather): the next launch's first barrier is
// the one that orders it. So the previous call's readers are done whichever
// encoding either call used.
// An INT4 kernel quantizes its input into its inputs' payload, and the pushed
// all-reduce pushes it into its peers' inputs (each rank's inbox, a slot a
// writer), before its first barrier, which is ordered, every rank's at its
// base and the region's offset. An exact call's eager copy is the host's,
// before its launch; its peers read it through the eager input table only
// after its first barrier, and its launch ends only after every peer's last
// read (INPUT TABLES).
constexpr int64_t kRegionAlign = 4096;
constexpr int64_t region_aligned(int64_t bytes) {
  return (bytes + kRegionAlign - 1) / kRegionAlign * kRegionAlign;
}
struct ArenaLayout {
  int64_t signals = 0;
  int64_t staging = 0;
  int64_t partials = 0;
  int64_t scales = 0;
  int64_t end = 0;  // the bytes the regions use: the arena's size
};
inline ArenaLayout arena_layout(int64_t staging_bytes, int64_t partials_bytes,
                                int64_t scales_bytes) {
  ArenaLayout l;
  l.signals = 0;
  l.staging = l.signals + kSignalsRegionBytes;
  l.partials = l.staging + region_aligned(staging_bytes);
  l.scales = l.partials + region_aligned(partials_bytes);
  l.end = l.scales + region_aligned(scales_bytes);
  return l;
}

// WHAT EACH RANK SAYS ABOUT ITSELF, gathered in the one gather open makes: its
// arena's IPC handle, what open compares across ranks (warnings) and how a
// peer's GPU is found in this process (its PCI bus id). Fixed width, as the
// gather moves equal lengths.
struct RankInfo {
  hipIpcMemHandle_t handle;
  char pci_bus_id[32];
  char host[64];
  int32_t pid;
  int32_t device;
};

// EVERY RANK'S RECORD, `me` at this rank, over `gather` (a collective: every
// rank calls it together).
inline std::vector<RankInfo> gather_ranks(int world, const Gather& gather,
                                          const RankInfo& me, Log& log) {
  log.phase(OpenPhase::handle_exchange_begun,
            static_cast<uint32_t>(world - 1));
  const std::vector<std::string> gathered =
      gather(std::string(reinterpret_cast<const char*>(&me), sizeof(me)));
  if (static_cast<int>(gathered.size()) != world)
    throw std::runtime_error(
        "hip_custom_communicator: the gather returned " +
        std::to_string(gathered.size()) + " records for " +
        std::to_string(world) + " ranks");
  std::vector<RankInfo> all(world);
  for (int p = 0; p < world; ++p) {
    if (gathered[p].size() != sizeof(RankInfo))
      throw std::runtime_error(
          "hip_custom_communicator: a gathered rank record of the wrong size");
    std::memcpy(&all[p], gathered[p].data(), sizeof(RankInfo));
    all[p].pci_bus_id[sizeof(all[p].pci_bus_id) - 1] = 0;
    all[p].host[sizeof(all[p].host) - 1] = 0;
  }
  log.phase(OpenPhase::handle_exchange_done, static_cast<uint32_t>(world - 1),
            static_cast<uint32_t>(world - 1));
  return all;
}

// THE ARENA: all of this rank's communicator memory, ONE allocation made once
// at open and freed as a unit, and every peer's opened over IPC: `base(r)` is
// rank r's arena as this process reaches it (ours at our rank). Built in steps
// around the handle exchange (the Handle's): the constructor makes and zeroes
// this rank's (before any peer can see it); `exported` gives its IPC handle;
// then `open_peers`.
//
// OWNERSHIP (STANDARDS 3.5): this rank's allocation (`bases_[rank]`,
// hipExtMallocWithFlags) is OWNED here and freed last, by `release_own`. Each
// peer's opened mapping is BORROWED: ours to close (`release_peers`,
// hipIpcCloseMemHandle), never the memory behind it, which its owner frees. A
// peer's mapping is closed before our own memory is freed (the close
// protocol's order, and the destructor's).
class Arena {
 public:
  Arena(int rank, int world, int64_t bytes)
      : rank_(rank), world_(world), bytes_(bytes), bases_(world, nullptr) {
    // A constructor that throws runs no destructor: what it made goes here.
    try {
      make();
    } catch (...) {
      (void)release_own();
      throw;
    }
  }

 private:
  void make() {
    // OWNED: this rank's memory, the one allocation of the communicator.
    void* own = nullptr;
    HIP_CHECK(hipExtMallocWithFlags(&own, static_cast<size_t>(bytes_),
                                    kArenaMallocFlags));
    bases_[rank_] = static_cast<char*>(own);
    HIP_CHECK(hipMemset(own, 0, static_cast<size_t>(bytes_)));
    HIP_CHECK(hipDeviceSynchronize());
  }

 public:
  // This rank's arena as an IPC handle its peers open.
  hipIpcMemHandle_t exported() const {
    hipIpcMemHandle_t h{};
    HIP_CHECK(hipIpcGetMemHandle(&h, bases_[rank_]));
    return h;
  }

  // Every peer's arena opened from its handle (`ranks[r].handle`, rank r's);
  // our own is not opened. Peer access is enabled lazily, by the open.
  void open_peers(const std::vector<RankInfo>& ranks) {
    for (int r = 0; r < world_; ++r) {
      if (r == rank_) continue;
      void* p = nullptr;
      HIP_CHECK(hipIpcOpenMemHandle(&p, ranks[r].handle,
                                    hipIpcMemLazyEnablePeerAccess));
      bases_[r] = static_cast<char*>(p);
    }
  }

  // OPEN'S ASSERTS ON THE ARENA, each logged (Log::check): every rank's arena
  // present (ours allocated, every peer's opened), and hipPointerGetAttributes
  // on each base sane (device memory, the base its own device pointer). The
  // device it reports is an observation (OpenWarning::pointer_device), against
  // `devices[r]`, rank r's GPU in this process (-1: not visible, not compared).
  // False when a check failed (the log names it).
  bool verify(Log& log, const std::vector<int>& devices) const {
    bool ok = true;
    for (int r = 0; r < world_; ++r) {
      const bool present = bases_[r] != nullptr;
      log.check(OpenCheck::peer_opened, present, r);
      ok = ok && present;
      if (!present) continue;
      hipPointerAttribute_t a{};
      const hipError_t e = hipPointerGetAttributes(&a, bases_[r]);
      if (e != hipSuccess) (void)hipGetLastError();
      const bool sane = e == hipSuccess && a.type == hipMemoryTypeDevice &&
                        a.devicePointer == static_cast<void*>(bases_[r]);
      log.check(OpenCheck::pointer_sane, sane, r, static_cast<uint32_t>(a.type),
                static_cast<uint32_t>(e));
      ok = ok && sane;
      if (sane && devices[r] >= 0 && a.device != devices[r])
        log.warn(OpenWarning::pointer_device, r,
                 static_cast<uint32_t>(a.device),
                 static_cast<uint32_t>(devices[r]));
    }
    return ok;
  }

  // WHAT A RELEASE DID: how many peers' handles it closed and whether it freed
  // our arena, and the first error it met (it carries on past one: a release
  // that stops half-way leaves more behind).
  struct Released {
    uint32_t closed = 0;
    uint32_t freed = 0;
    hipError_t error = hipSuccess;
  };

  // EVERY BORROWED PEER MAPPING closed: the first half of teardown, before our
  // own memory goes. Idempotent.
  Released release_peers() {
    Released out;
    for (int r = 0; r < world_; ++r) {
      if (r == rank_ || bases_[r] == nullptr) continue;
      note(out, hipIpcCloseMemHandle(bases_[r]));
      bases_[r] = nullptr;
      ++out.closed;
    }
    return out;
  }
  // OUR OWN arena freed: the second half, once no peer has it open (the close
  // protocol's second handshake). Idempotent.
  Released release_own() {
    Released out;
    if (bases_[rank_] != nullptr) {
      note(out, hipFree(bases_[rank_]));
      bases_[rank_] = nullptr;
      ++out.freed;
    }
    return out;
  }
  bool peers_released() const {
    for (int r = 0; r < world_; ++r)
      if (r != rank_ && bases_[r] != nullptr) return false;
    return true;
  }
  bool released() const { return bases_[rank_] == nullptr; }

  // Borrowed first, then owned, whatever the path that got here.
  ~Arena() {
    (void)release_peers();
    (void)release_own();
  }
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  // Rank r's arena as this process reaches it.
  char* base(int r) const { return bases_[r]; }
  int64_t bytes() const { return bytes_; }

 private:
  static void note(Released& out, hipError_t e) {
    if (e != hipSuccess && out.error == hipSuccess) out.error = e;
    if (e != hipSuccess) (void)hipGetLastError();
  }

  int rank_;
  int world_;
  int64_t bytes_;
  std::vector<char*> bases_;
};

// OTHER PROCESSES HOLDING DEVICE MEMORY ON `device`, read from the KFD's sysfs
// (/sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>): how many, and their MiB. The
// GPU's KFD id is the topology node whose PCI location is the device's. {0, 0}
// when sysfs cannot be read: an observation, never a failure.
struct OtherProcesses {
  uint32_t count = 0;
  uint32_t mib = 0;
};
inline OtherProcesses other_processes_on(int device) {
  OtherProcesses out;
  int domain = 0, bus = 0, dev = 0;
  if (hipDeviceGetAttribute(&domain, hipDeviceAttributePciDomainID, device) !=
          hipSuccess ||
      hipDeviceGetAttribute(&bus, hipDeviceAttributePciBusId, device) !=
          hipSuccess ||
      hipDeviceGetAttribute(&dev, hipDeviceAttributePciDeviceId, device) !=
          hipSuccess) {
    (void)hipGetLastError();
    return out;
  }
  const std::string topology = "/sys/class/kfd/kfd/topology/nodes";
  std::string gpu_id;
  if (DIR* d = ::opendir(topology.c_str())) {
    while (dirent* e = ::readdir(d)) {
      if (e->d_name[0] == '.') continue;
      const std::string node = topology + "/" + e->d_name;
      std::ifstream props(node + "/properties");
      std::string key;
      long long value = 0, node_domain = -1, location = -1;
      while (props >> key >> value) {
        if (key == "domain") node_domain = value;
        if (key == "location_id") location = value;
      }
      if (location < 0 || node_domain != domain || (location >> 8) != bus ||
          ((location >> 3) & 0x1f) != dev)
        continue;
      std::ifstream id(node + "/gpu_id");
      id >> gpu_id;
      break;
    }
    ::closedir(d);
  }
  if (gpu_id.empty() || gpu_id == "0") return out;
  const std::string procs = "/sys/class/kfd/kfd/proc";
  const std::string self = std::to_string(::getpid());
  if (DIR* d = ::opendir(procs.c_str())) {
    while (dirent* e = ::readdir(d)) {
      if (e->d_name[0] == '.' || self == e->d_name) continue;
      std::ifstream vram(procs + "/" + e->d_name + "/vram_" + gpu_id);
      unsigned long long bytes = 0;
      if (vram >> bytes && bytes > 0) {
        ++out.count;
        out.mib += static_cast<uint32_t>(bytes >> 20);
      }
    }
    ::closedir(d);
  }
  return out;
}

// INPUT TABLES: an exact call's view of every rank's input (every op's), a
// TABLE of kMaxRanks addresses (rank r's input at [r], null past the world) in
// this rank's device memory, one pointer its kernel takes (aiter's RankData*).
// The tables live in one SLAB (hipMalloc: read by this rank's kernels only),
// kInputTableSlots of them:
//   - slot 0, THE EAGER TABLE, written at open: every rank's inputs region
//     (its arena base, as this process opened it, plus the region's offset).
//     An eager call copies its input there first (hipMemcpy2DAsync on the
//     launch's stream, dense rows), so its kernel, stream-ordered after the
//     copy, finds every rank's at the table, each rank having copied its own.
//   - every later slot, A CAPTURED LAUNCH'S: reserved at capture (the launch
//     records the slot's address, fixed for the graph's life) with the input's
//     address remembered; filled when the capture exits, before any replay,
//     by `register_captured`: one batched exchange of every recorded input's
//     IPC handle and offset over the CPU group, each peer's allocation opened
//     once (cached by its handle: the caching allocator hands every graph the
//     same few segments) and the slots written.
// A slot is never reused: a graph's addresses die with it, and the count each
// rank records must match for the exchange (a collective) to pair them.
// THE ORDER THAT LETS AN INPUT BE OVERWRITTEN: every peer's last read of this
// rank's input precedes a barrier this rank's launch waits on before it ends
// (a one-shot's `read`, a two-shot's second, after its reduce-scatter), so the
// next eager copy, or the caller's next write, comes after every read.
// THE SLAB'S SIZE, fixed at open: a slot's address is baked into the graph that
// captured it, and torch captures in global mode, where no allocation may
// happen, so the slab cannot grow during a capture. 262144 slots of kMaxRanks
// pointers is 16 MiB. 16384 was too few for Kimi-K3 at InferenceX's c48 point
// (capture sizes to 48, a speculative drafter's layers too; run
// 2026-10-17T03-18-18Z, hip-bench_inferencex_c48).
constexpr int64_t kInputTableSlots = int64_t{1} << 18;

// A LAUNCH'S INPUT TABLE: the device array of every rank's input address
// (`ptr_array`, the kernels' `inputs_ptr_array`) with its size (the world; 0
// with no array), and the strides every rank's input has there (the caller's
// for a captured call; dense rows, its copy's, for an eager one).
struct InputTable {
  const void* ptr_array;
  int ptr_array_size;
  int64_t stride_m;
  int64_t stride_n;
};

class Handle {
 public:
  // OPENED on every rank together: this rank's arena allocated (the signals,
  // the staging, the partials, the scales: `arena_layout`), its IPC handle exchanged with
  // every rank's, and the peers' opened.
  // `world_size` is one the build holds (the opener checks it).
  // `timeout_seconds`: how long a kernel waits on a peer before it records a
  // fault and leaves (common.cuh's BARRIER), the caller's: the process group's
  // own timeout, as every other collective on it has. THE EXCHANGE'S SIZES are
  // the caller's too, computed from the largest call it serves, `max_rows` rows
  // of at most `max_cols` columns (config.cuh's `exchange_sizes`): kept, so
  // what the buffers were sized for can be asked. `quantization`: what the tuned path moves
  // across the links, a Quantization (config.cuh's `quantizes`).
  // `device`: the device the communicator was given (open checks it is the
  // current one). `log_mode`: what the log records (log.cuh). `rccl_ready`:
  // whether the caller found the device group's RCCL communicator set up (1),
  // not (0), or could not tell (-1): observed and logged, never acted on.
  Handle(int rank, int world_size, int device, const Gather& gather,
         double timeout_seconds, int64_t max_rows, int64_t max_cols,
         int64_t staging_bytes, int64_t partials_bytes,
         int quantization, LogMode log_mode, int rccl_ready)
      : rank_(rank),
        world_size_(world_size),
        device_(device),
        max_rows_(max_rows),
        max_cols_(max_cols),
        staging_bytes_(staging_bytes),
        partials_bytes_(partials_bytes),
        quantization_(quantization),
        log_(std::make_shared<Log>(log_mode, rank, world_size, device)),
        layout_(arena_layout(staging_bytes, partials_bytes,
                             max_rows * static_cast<int64_t>(sizeof(float)))) {
    // THE LOG FIRST, and kept as the process's last (log.cuh's `last_log`):
    // an open that fails or hangs is read there.
    set_last_log(log_);
    try {
      open(gather, timeout_seconds, rccl_ready);
    } catch (...) {
      // THE PHASE AFTER THE LAST REACHED is the one that failed.
      const uint32_t reached = log_->get(LogHeader::open_phase);
      if (reached < static_cast<uint32_t>(OpenPhase::open_done))
        log_->phase(next_open_phase(static_cast<OpenPhase>(reached)), 0, 0, 1);
      // AN OPEN THAT FAILED tears down what it made, alone (its peers may
      // never have opened it, or may be failing too), borrowed mappings first.
      if (arena_) abandon(kAbandonWaitSeconds);
      throw;
    }
    enlist();
  }

  // THE LAST OWNER'S TEARDOWN, local: whatever close or abandon did not reach,
  // in the same order (every launch finished, borrowed mappings, then our own
  // memory). A handle that was closed has nothing left here.
  ~Handle() {
    delist();
    std::lock_guard<std::mutex> g(teardown_);
    if (!arena_ || arena_->released()) return;
    (void)hipDeviceSynchronize();
    Arena::Released peers = arena_->release_peers();
    add(peers, release_inputs());
    log_->close_phase(ClosePhase::peers_released, false, peers.closed, 0,
                      static_cast<uint32_t>(peers.error));
    Arena::Released own = arena_->release_own();
    add(own, release_tables());
    add(own, release_sums());
    add(own, release_region_ptr_arrays());
    log_->close_phase(ClosePhase::released, false, own.freed, 0,
                      static_cast<uint32_t>(own.error));
  }

  // A CHECK ON EVERY RANK'S ARRIVAL, the caller's (over the CPU group): true
  // when every rank arrived within the caller's bound, false when one did not
  // or the group failed. Never unbounded: a peer that died must not hang us.
  using Barrier = std::function<bool()>;

  // CLOSE, A COLLECTIVE (STANDARDS 3.5, "lifetimes are collective"): every
  // rank calls it together. (a) every launch on this device finished, so
  // nothing of ours reads or writes a peer's memory; (b) handshake: every rank
  // has stopped; (c) every peer's IPC mapping closed (hipIpcCloseMemHandle);
  // (d) handshake: every rank has closed its mapping of ours; (e) our own
  // arena freed. No rank frees memory another still has open. A handshake that fails (a peer that did
  // not arrive within the bound) is logged and the teardown carries on alone.
  // Returns 0, or the first ClosePhase that failed. Idempotent; steps an
  // abandon already ran are not run again.
  int close(const Barrier& barrier) {
    std::lock_guard<std::mutex> g(teardown_);
    if (!arena_ || arena_->released()) return 0;
    int failed = 0;
    const auto fail = [&](ClosePhase p) {
      if (failed == 0) failed = static_cast<int>(p);
    };
    log_->close_phase(ClosePhase::closing, true);
    // (a) Unbounded: every kernel's wait has its own timeout (common.cuh's
    // BARRIER), so a launch on a dead peer leaves by itself.
    const hipError_t synced = hipDeviceSynchronize();
    if (synced != hipSuccess) (void)hipGetLastError();
    log_->close_phase(ClosePhase::synchronized, true, 1, 0,
                      static_cast<uint32_t>(synced));
    if (synced != hipSuccess) fail(ClosePhase::synchronized);
    // (b)
    const bool stopped = barrier();
    log_->close_phase(ClosePhase::stopped, true, 0, 0, stopped ? 0 : 1);
    if (!stopped) fail(ClosePhase::stopped);
    // (c) The arenas and every registered input opened from a peer.
    Arena::Released peers = arena_->release_peers();
    add(peers, release_inputs());
    log_->close_phase(ClosePhase::peers_released, true, peers.closed, 0,
                      static_cast<uint32_t>(peers.error));
    if (peers.error != hipSuccess) fail(ClosePhase::peers_released);
    // (d) Skipped when (b) failed: a peer that missed the first handshake
    // would pair this one with its own first.
    const bool done = stopped && barrier();
    log_->close_phase(ClosePhase::peers_done, true, 0, 0, done ? 0 : 1);
    if (!done) fail(ClosePhase::peers_done);
    // (e) Our arena, the slab of input tables and the arena's device array.
    Arena::Released own = arena_->release_own();
    add(own, release_tables());
    add(own, release_sums());
    add(own, release_region_ptr_arrays());
    log_->close_phase(ClosePhase::released, true, own.freed, 0,
                      static_cast<uint32_t>(own.error));
    if (own.error != hipSuccess) fail(ClosePhase::released);
    log_->close_phase(ClosePhase::close_done, true, 0, 0,
                      static_cast<uint32_t>(failed));
    return failed;
  }

  // ABANDON: the teardown ALONE, for an error path, a signal or process exit
  // (STANDARDS 3.5, "abnormal exit is designed for"): a process should not
  // die holding peers' memory open, nor free its own while peers read it. So,
  // as far as it can without a peer: every wait on this rank told to leave
  // (`abort`), the device given `wait_seconds` to finish (never more: a peer
  // may be dead), then every BORROWED peer mapping closed
  // (hipIpcCloseMemHandle), and, only when the device finished, our own arena
  // freed (freeing memory a running kernel still reads would fault it). Never
  // waits on a peer. Returns 0, or the first ClosePhase that failed.
  // Idempotent, and a no-op while another teardown runs (it returns
  // ClosePhase::closing then).
  int abandon(double wait_seconds) {
    std::unique_lock<std::mutex> g(teardown_, std::try_to_lock);
    if (!g.owns_lock()) return static_cast<int>(ClosePhase::closing);
    if (!arena_ || arena_->peers_released()) return 0;
    int failed = 0;
    log_->close_phase(ClosePhase::closing, false);
    abort();
    const bool finished = synchronize_within(wait_seconds);
    log_->close_phase(ClosePhase::synchronized, false, finished ? 1 : 0, 0,
                      finished ? 0 : 1);
    if (!finished) failed = static_cast<int>(ClosePhase::synchronized);
    Arena::Released peers = arena_->release_peers();
    add(peers, release_inputs());
    log_->close_phase(ClosePhase::peers_released, false, peers.closed, 0,
                      static_cast<uint32_t>(peers.error));
    if (peers.error != hipSuccess && failed == 0)
      failed = static_cast<int>(ClosePhase::peers_released);
    if (finished) {
      Arena::Released own = arena_->release_own();
      add(own, release_tables());
      add(own, release_sums());
      add(own, release_region_ptr_arrays());
      log_->close_phase(ClosePhase::released, false, own.freed, 0,
                        static_cast<uint32_t>(own.error));
      if (own.error != hipSuccess && failed == 0)
        failed = static_cast<int>(ClosePhase::released);
    }
    log_->close_phase(ClosePhase::close_done, false, 0, 0,
                      static_cast<uint32_t>(failed));
    return failed;
  }

  // How long an abandon waits for the device, when its caller does not say
  // (a failed open, process exit): long enough for every wait to see `abort`.
  static constexpr double kAbandonWaitSeconds = 2.0;

  // EVERY HANDLE STILL OPEN IN THIS PROCESS, abandoned: process exit's hook
  // (`open_handles`' std::atexit), and Python's atexit and signal handlers.
  // Held under the registry's lock, so no handle is deleted meanwhile.
  static void abandon_all(double wait_seconds) {
    Registry& r = registry();
    std::lock_guard<std::mutex> g(r.mutex);
    for (Handle* h : r.open) (void)h->abandon(wait_seconds);
  }

  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;

  int rank() const { return rank_; }
  int world_size() const { return world_size_; }
  int64_t max_rows() const { return max_rows_; }
  int64_t max_cols() const { return max_cols_; }
  // Each region's bytes, what its largest payload in either encoding takes.
  int64_t staging_bytes() const { return staging_bytes_; }
  int64_t partials_bytes() const { return partials_bytes_; }
  // What the tuned path moves across the links, a Quantization.
  int quantization() const { return quantization_; }
  // THIS RANK'S LOCAL SUMS: a split op's all-reduced rows (config.cuh's
  // Algorithm::split), written by its plain all-reduce and read by its own
  // epilogue kernel after it on the stream. Never a peer's (not in the arena:
  // hipMalloc, cacheable), so nothing orders it but the stream. Sized as the
  // inputs region, which holds any call's whole input (select refuses one it
  // does not), so it holds any call's sum.
  void* sums() const { return sums_; }
  int64_t sums_bytes() const { return staging_bytes_; }

  // EVERY RANK'S SIGNALS, a device array of every rank's signals region with
  // its size, the world (written at open, with the others in
  // `region_ptr_arrays_`).
  uint32_t* const* signals_ptr_array() const {
    return reinterpret_cast<uint32_t* const*>(region_ptr_arrays_ +
                                              kSignalsArray * kMaxRanks);
  }
  int signals_ptr_array_size() const { return world_size_; }
  // EVERY RANK'S STAGING (the arena's `inputs` region: an eager copy, an INT4
  // payload, the pushed all-reduce's inbox) and EVERY RANK'S PARTIALS, a device
  // array each of every rank's address of that region with its size, the world
  // (written at open, with the others in `region_ptr_arrays_`). A kernel that
  // takes them never sees the arena or an offset into it.
  char* const* staging_ptr_array() const {
    return region_ptr_arrays_ + kStagingArray * kMaxRanks;
  }
  int staging_ptr_array_size() const { return world_size_; }
  char* const* partials_ptr_array() const {
    return region_ptr_arrays_ + kPartialsArray * kMaxRanks;
  }
  int partials_ptr_array_size() const { return world_size_; }
  // EVERY RANK'S ROW SCALES (the add-RMS-norm pull two-shot's ADD_RESIDUAL: a float a row it owns,
  // which its peers read while gathering), [max_rows] floats a rank, its own region: a device array
  // of every rank's address of it with its size, the world.
  float* const* scales_ptr_array() const {
    return reinterpret_cast<float* const*>(region_ptr_arrays_ + kScalesArray * kMaxRanks);
  }
  int scales_ptr_array_size() const { return world_size_; }
  int64_t scales_size_m() const { return max_rows_; }
  // The synchronization state: every rank's signal block and how long a wait
  // may last before it gives up. Each rank's signals, [2, kMaxBlocks,
  // kMaxRanks] (parity, block, writer) at signals_ptr_array()[r]: the
  // strides and the sizes, in elements.
  static constexpr int64_t signals_stride_parity() {
    return int64_t{kMaxBlocks} * kMaxRanks;
  }
  static constexpr int64_t signals_stride_block() { return kMaxRanks; }
  static constexpr int64_t signals_stride_writer() { return 1; }
  static constexpr int64_t signals_size_parity() { return 2; }
  static constexpr int64_t signals_size_block() { return kMaxBlocks; }
  static constexpr int64_t signals_size_writer() { return kMaxRanks; }
  // This rank's status, at its device address: one word a StatusField.
  uint32_t* status_ptr() const { return device_status_; }
  static constexpr int64_t status_stride_field() { return 1; }
  static constexpr int64_t status_size_field() { return kStatusFields; }
  uint64_t timeout_ticks() const { return timeout_ticks_; }

  // THE STATUS, every field's word, read while kernels run (host-mapped: it
  // never waits on the device). The fault's fields are whole only once `state`
  // is FaultState::written; the reader checks.
  std::array<uint32_t, kStatusFields> status() const {
    std::array<uint32_t, kStatusFields> words{};
    words[static_cast<int>(StatusField::state)] = __atomic_load_n(
        &status_[static_cast<int>(StatusField::state)], __ATOMIC_ACQUIRE);
    for (int f = 0; f < kStatusFields; ++f)
      if (f != static_cast<int>(StatusField::state)) words[f] = status_[f];
    return words;
  }
  // THIS RANK'S SIGNALS, every flag, as (parity, block, writer, value), the
  // value a FlagState unless the memory is corrupt. Between launches every one
  // is `clear` (common.cuh's BARRIER); read once every launch has finished,
  // which this waits for itself.
  std::vector<std::array<int64_t, 4>> signals() const {
    std::vector<uint32_t> flags(
        static_cast<size_t>(kSignalsBytes / sizeof(uint32_t)));
    // EVERY LAUNCH FINISHED FIRST, here and not at the callers: read
    // mid-launch, a barrier shows its peers' flags set and this rank's not yet
    // cleared, which reads as a launch that left its signals set (a test that
    // synchronized on one of its two paths, 2026-10-15). torch's streams are
    // non-blocking, so the copy below would not wait for them. A hung launch
    // hangs this query too: status() is the one readable while a kernel waits.
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(flags.data(), arena_->base(rank_) + layout_.signals,
                        static_cast<size_t>(kSignalsBytes),
                        hipMemcpyDeviceToHost));
    std::vector<std::array<int64_t, 4>> all;
    for (int64_t p = 0; p < signals_size_parity(); ++p)
      for (int64_t b = 0; b < signals_size_block(); ++b)
        for (int64_t w = 0; w < signals_size_writer(); ++w)
          all.push_back(
              {p, b, w,
               flags[p * signals_stride_parity() + b * signals_stride_block() +
                     w * signals_stride_writer()]});
    return all;
  }
  // THE LOG (log.cuh): every word of it, as the host reads it now, and the
  // number the last launch was given.
  std::vector<uint32_t> log_words() const { return log_->words(); }
  uint32_t launch_number() const { return log_->launches(); }

  // A LAUNCH ABOUT TO BE ISSUED: its number, in the header always (the
  // kernel echoes it from the next cycle on), and in everything mode a record
  // of what it is. A captured launch is numbered at capture; its replays are
  // not counted (a graph replay runs none of our host code).
  uint32_t begin_launch(OpType op, const void* kernel, int64_t m, int64_t n,
                        DType dtype, hipStream_t stream) {
    const uint32_t launch = log_->next_launch();
    if (!log_->everything()) return launch;
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    HIP_CHECK(hipStreamIsCapturing(stream, &capture));
    const uint8_t flags = static_cast<uint8_t>(
        capture == hipStreamCaptureStatusActive ? 1 : 0);
    log_->write({static_cast<uint8_t>(LogKind::launch_issued), flags,
                 stream_index(stream), launch, kernel_index(kernel),
                 static_cast<uint16_t>(op), static_cast<uint32_t>(m),
                 static_cast<uint32_t>(n), static_cast<uint32_t>(dtype),
                 host_ns()});
    return launch;
  }

  // THE INPUT TABLE FOR A LAUNCH over `input` ([m, n] of `elem_bytes`
  // elements, rows `stride_m` apart, dense within) on `stream` (INPUT TABLES):
  // capturing, a fresh slot, filled at capture exit (`register_captured`);
  // eager, the input copied into this rank's inputs region on the stream
  // (logged: copy_in_issued) and the eager table. Select refuses an eager call
  // the inputs region does not hold, so the throw is never a call's answer.
  InputTable input_table(const void* input, int64_t stride_m, int64_t m,
                         int64_t n, int64_t elem_bytes, hipStream_t stream) {
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    HIP_CHECK(hipStreamIsCapturing(stream, &capture));
    if (capture == hipStreamCaptureStatusActive) {
      // A FRESH SLOT ALWAYS, even for an address seen before: a graph's
      // buffers die with it and the allocator hands the address back, and the
      // exchange at capture exit pairs the ranks' records by count.
      if (next_slot_ >= kInputTableSlots)
        throw std::runtime_error(
            "hip_custom_communicator: the input-table slab is full (" +
            std::to_string(kInputTableSlots) + " captured launches)");
      void** slot = tables_ + next_slot_ * kMaxRanks;
      ++next_slot_;
      pending_.push_back(input);
      pending_slots_.push_back(slot);
      return {slot, world_size_, stride_m, 1};
    }
    const int64_t row = n * elem_bytes;
    if (m * row > staging_bytes_)
      throw std::runtime_error(
          "hip_custom_communicator: an eager " + std::to_string(m * row) +
          "-byte input exceeds the " + std::to_string(staging_bytes_) +
          "-byte inputs region");
    HIP_CHECK(hipMemcpy2DAsync(arena_->base(rank_) + layout_.staging,
                               static_cast<size_t>(row), input,
                               static_cast<size_t>(stride_m * elem_bytes),
                               static_cast<size_t>(row),
                               static_cast<size_t>(m), hipMemcpyDeviceToDevice,
                               stream));
    if (log_->everything()) {
      const auto bytes = static_cast<uint64_t>(m * row);
      log_->write({static_cast<uint8_t>(LogKind::copy_in_issued), 0,
                   stream_index(stream), log_->launches(), 0, 0,
                   static_cast<uint32_t>(bytes),
                   static_cast<uint32_t>(bytes >> 32), 0, host_ns()});
    }
    return {tables_, world_size_, n, 1};
  }

  // THE INPUTS A CAPTURE RECORDED, registered, every rank together (a
  // collective over `gather`, even with none recorded, or the other ranks wait
  // in it): the ranks' counts compared, then one exchange of every recorded
  // input's IPC handle and offset, each peer's allocation opened (once: cached
  // by its handle) and every slot written. Sound because a captured address is
  // fixed for the graph's life and the slots are filled before any replay.
  // False, nothing registered, when the ranks recorded different counts (every
  // rank must capture the same launches). Logged: registration, begun and done.
  bool register_captured(const Gather& gather) {
    const auto count = static_cast<int64_t>(pending_.size());
    if (log_->everything())
      log_->write({static_cast<uint8_t>(LogKind::registration), 0, 0,
                   log_->launches(), 0, 0, static_cast<uint32_t>(count), 0, 0,
                   host_ns()});
    for (const std::string& c : gather(std::string(
             reinterpret_cast<const char*>(&count), sizeof(count)))) {
      int64_t theirs = -1;
      if (c.size() == sizeof(theirs)) std::memcpy(&theirs, c.data(), sizeof(theirs));
      if (theirs != count) {
        log_->write({static_cast<uint8_t>(LogKind::registration), 2, 0,
                     log_->launches(), 0, 0, static_cast<uint32_t>(count),
                     static_cast<uint32_t>(theirs), 1, host_ns()});
        pending_.clear();
        pending_slots_.clear();
        return false;
      }
    }
    const size_t opened_before = opened_.size();
    if (count > 0) {
      std::string mine;
      for (const void* p : pending_) mine += ipc_bytes(p);
      const size_t each = mine.size() / pending_.size();
      const std::vector<std::string> all = gather(mine);
      std::vector<void*> host(kMaxRanks, nullptr);
      for (size_t i = 0; i < pending_.size(); ++i) {
        for (int r = 0; r < world_size_; ++r)
          host[r] = r == rank_ ? const_cast<void*>(pending_[i])
                               : open_input(all[r].substr(i * each, each));
        HIP_CHECK(hipMemcpy(pending_slots_[i], host.data(),
                            kMaxRanks * sizeof(void*), hipMemcpyHostToDevice));
      }
    }
    if (log_->everything())
      log_->write({static_cast<uint8_t>(LogKind::registration), 1, 0,
                   log_->launches(), 0, 0, static_cast<uint32_t>(count),
                   static_cast<uint32_t>(opened_.size() - opened_before), 0,
                   host_ns()});
    pending_.clear();
    pending_slots_.clear();
    return true;
  }
  // How many captured launches await registration, and how many slots are
  // taken (the eager table's included).
  int64_t pending_inputs() const { return static_cast<int64_t>(pending_.size()); }
  int64_t table_slots_used() const { return next_slot_; }

  // Every wait on this rank leaves at its next check: for a communicator the
  // host has given up on.
  // The log's words, not `status_`: an open that failed is abandoned before
  // `status_` is set.
  void abort() {
    __atomic_store_n(&log_->status_host()[static_cast<int>(StatusField::abort)],
                     uint32_t{1}, __ATOMIC_RELEASE);
  }

  // THE MOST BLOCKS OF `kernel`, `threads` A BLOCK, RESIDENT AT ONCE ON THIS
  // DEVICE: the textbook query, HIP's occupancy API (CUDA's
  // cudaOccupancyMaxActiveBlocksPerMultiprocessor), blocks a CU from the
  // compiled kernel's registers, LDS and the CU's wave slots, times the
  // device's CUs. A peer barrier spins until the same block on every peer
  // arrives, so a grid past it can hang (select refuses it,
  // grid_not_resident). Asked once per kernel and block size, then cached; no
  // dynamic LDS (Config selects zero shared memory).
  int resident_blocks(const void* kernel, int threads) const {
    const std::pair<const void*, int> key{kernel, threads};
    if (const auto it = resident_.find(key); it != resident_.end())
      return it->second;
    int per_cu = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, kernel,
                                                           threads, 0));
    int cus = 0;
    HIP_CHECK(hipDeviceGetAttribute(
        &cus, hipDeviceAttributeMultiprocessorCount, device_));
    return resident_[key] = per_cu * cus;
  }

 private:
  // OPEN, every phase logged (log.cuh's OpenPhase), each internal assert a
  // logged OpenCheck that fails open naming it, each observation of the
  // broader state a logged OpenWarning. No collective beyond the CPU group's
  // gather the handle exchange already makes.
  void open(const Gather& gather, double timeout_seconds, int rccl_ready) {
    Log& log = *log_;
    log.phase(OpenPhase::log_ready, static_cast<uint32_t>(kLogCapacity));
    int current = -1;
    HIP_CHECK(hipGetDevice(&current));
    log.check(OpenCheck::current_device, current == device_, rank_,
              static_cast<uint32_t>(current), static_cast<uint32_t>(device_));
    if (current != device_)
      fail(OpenCheck::current_device,
           "the current device is " + std::to_string(current) +
               ", not the communicator's " + std::to_string(device_));
    // THE BROADER STATE, observed: RCCL's lazy setup, other processes on this
    // GPU, and whether what open allocates fits.
    if (rccl_ready == 0)
      log.warn(OpenWarning::rccl_not_ready);
    else if (rccl_ready < 0)
      log.warn(OpenWarning::rccl_unknown);
    const OtherProcesses others = other_processes_on(device_);
    if (others.count > 0)
      log.warn(OpenWarning::other_processes, 0, others.count, others.mib);
    {
      size_t free = 0, total = 0;
      HIP_CHECK(hipMemGetInfo(&free, &total));
      const int64_t needed = layout_.end;
      if (static_cast<int64_t>(free) < needed)
        log.warn(OpenWarning::low_memory, 0, static_cast<uint32_t>(free >> 20),
                 static_cast<uint32_t>(needed >> 20));
    }
    // THE ARENA, the one allocation (its layout: `arena_layout`), made and
    // zeroed here, then every rank's IPC handle exchanged (one collective) and
    // every peer's opened.
    arena_.emplace(rank_, world_size_, layout_.end);
    log.phase(OpenPhase::arena_allocated,
              static_cast<uint32_t>(arena_->bytes() >> 10));
    {
      // THIS RANK'S SIGNALS READ ZERO after the memset, before any peer can
      // write them: a few words, host-side.
      std::array<uint32_t, 64> words{};
      HIP_CHECK(hipMemcpy(words.data(), arena_->base(rank_) + layout_.signals,
                          sizeof(words), hipMemcpyDeviceToHost));
      uint32_t nonzero = 0;
      for (uint32_t w : words) nonzero += w != 0;
      log.check(OpenCheck::signals_zero, nonzero == 0, rank_, nonzero, 0);
      if (nonzero != 0)
        fail(OpenCheck::signals_zero, std::to_string(nonzero) +
                                          " of the first 64 signal words are "
                                          "not zero after the memset");
      log.phase(OpenPhase::zeroed);
    }
    RankInfo me{};
    me.handle = arena_->exported();
    log.phase(OpenPhase::exported);
    HIP_CHECK(
        hipDeviceGetPCIBusId(me.pci_bus_id, sizeof(me.pci_bus_id), device_));
    (void)::gethostname(me.host, sizeof(me.host) - 1);
    me.pid = static_cast<int32_t>(::getpid());
    me.device = device_;
    const std::vector<RankInfo> ranks =
        gather_ranks(world_size_, gather, me, log);
    observe_peers(ranks);
    arena_->open_peers(ranks);
    log.phase(OpenPhase::opened, static_cast<uint32_t>(world_size_ - 1));
    // EVERY PEER'S GPU in this process (by its PCI bus id), and whether this
    // device reaches it: a peer that cannot be reached is a definite error.
    std::vector<int> peer_devices(world_size_, -1);
    for (int r = 0; r < world_size_; ++r) {
      int d = -1;
      if (hipDeviceGetByPCIBusId(&d, ranks[r].pci_bus_id) != hipSuccess) {
        (void)hipGetLastError();
        log.warn(OpenWarning::peer_not_visible, r);
        continue;
      }
      peer_devices[r] = d;
      if (r == rank_ || d == device_) continue;
      int can = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&can, device_, d));
      log.check(OpenCheck::can_access_peer, can != 0, r,
                static_cast<uint32_t>(device_), static_cast<uint32_t>(d));
      if (can == 0)
        fail(OpenCheck::can_access_peer,
             "device " + std::to_string(device_) + " cannot access rank " +
                 std::to_string(r) + "'s device " + std::to_string(d));
    }
    if (!arena_->verify(log, peer_devices))
      fail(static_cast<OpenCheck>(log.get(LogHeader::failed)),
           "the peers' arenas");
    // EACH REGION'S DEVICE ARRAY of every rank's address of it (its arena's
    // base, where this process reaches it, plus the region's offset), kMaxRanks
    // entries each (null past the world), all in one allocation, written once,
    // here: the kernels take each region's as its own array with its size, the
    // world (`signals_ptr_array`, ...), and never an arena base or an offset.
    // The arena's own bases never reach the device: they are the host's, for
    // the regions' lifetimes and IPC.
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&region_ptr_arrays_),
                        kRegionPtrArrays * kMaxRanks * sizeof(char*)));
    {
      std::vector<char*> at(kRegionPtrArrays * kMaxRanks, nullptr);
      for (int r = 0; r < world_size_; ++r) {
        char* const base = arena_->base(r);
        at[kSignalsArray * kMaxRanks + r] = base + layout_.signals;
        at[kStagingArray * kMaxRanks + r] = base + layout_.staging;
        at[kPartialsArray * kMaxRanks + r] = base + layout_.partials;
        at[kScalesArray * kMaxRanks + r] = base + layout_.scales;
      }
      HIP_CHECK(hipMemcpy(region_ptr_arrays_, at.data(),
                          at.size() * sizeof(char*), hipMemcpyHostToDevice));
    }
    // THE INPUT TABLES' SLAB (INPUT TABLES), zeroed, and its slot 0, the
    // eager table: every rank's inputs region.
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&tables_),
                        static_cast<size_t>(kInputTableSlots) * kMaxRanks *
                            sizeof(void*)));
    HIP_CHECK(hipMemset(tables_, 0,
                        static_cast<size_t>(kInputTableSlots) * kMaxRanks *
                            sizeof(void*)));
    {
      std::vector<void*> eager(kMaxRanks, nullptr);
      for (int r = 0; r < world_size_; ++r)
        eager[r] = arena_->base(r) + layout_.staging;
      HIP_CHECK(hipMemcpy(tables_, eager.data(), kMaxRanks * sizeof(void*),
                          hipMemcpyHostToDevice));
      next_slot_ = 1;
    }
    log.phase(OpenPhase::tables_written,
              static_cast<uint32_t>(kInputTableSlots));
    // THE LOCAL SUMS (`sums`): hipMalloc, cacheable, this rank's alone.
    HIP_CHECK(hipMalloc(&sums_, static_cast<size_t>(staging_bytes_)));
    // THE STATUS, host-mapped and coherent: the log's first words
    // (log.cuh). A wait that gives up writes its fault with system-scope
    // atomics while the host polls it, and the host sets its `abort`
    // (common.cuh's BARRIER). The kernels take its device address as their
    // own argument, a buffer of kStatusFields words.
    status_ = log.status_host();
    device_status_ = log.status_device();
    // The device wall clock is fixed-rate, in kHz; the kernels count the
    // timeout in it.
    int khz = 0;
    HIP_CHECK(
        hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, device_));
    timeout_ticks_ = static_cast<uint64_t>(timeout_seconds * khz * 1000.0);
    log.set_timeout(timeout_ticks_, khz);
    log.phase(OpenPhase::checked);
    log.phase(OpenPhase::open_done);
  }

  // AN INPUT AS THE GATHER CARRIES IT: its allocation's IPC handle, then its
  // offset in that allocation (hipIpcGetMemHandle takes an allocation's BASE,
  // and a torch tensor sits at an offset inside one).
  static std::string ipc_bytes(const void* ptr) {
    void* base = nullptr;
    HIP_CHECK(hipPointerGetAttribute(
        &base, HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
        reinterpret_cast<hipDeviceptr_t>(const_cast<void*>(ptr))));
    hipIpcMemHandle_t h;
    HIP_CHECK(hipIpcGetMemHandle(&h, base));
    const int64_t offset =
        static_cast<const char*>(ptr) - static_cast<char*>(base);
    return std::string(reinterpret_cast<const char*>(&h), sizeof(h)) +
           std::string(reinterpret_cast<const char*>(&offset), sizeof(offset));
  }

  // A PEER'S INPUT (its `ipc_bytes`) as this process reaches it: its
  // allocation opened once (cached by its handle, as vLLM's CustomAllreduce
  // does), plus the offset. BORROWED: closed by `release_inputs`.
  void* open_input(const std::string& bytes) {
    constexpr size_t kIpc = sizeof(hipIpcMemHandle_t);
    if (bytes.size() != kIpc + sizeof(int64_t))
      throw std::runtime_error(
          "hip_custom_communicator: a gathered input is an IPC handle and an "
          "offset");
    const std::string handle = bytes.substr(0, kIpc);
    int64_t offset = 0;
    std::memcpy(&offset, bytes.data() + kIpc, sizeof(offset));
    auto it = opened_.find(handle);
    if (it == opened_.end()) {
      hipIpcMemHandle_t h;
      std::memcpy(&h, handle.data(), kIpc);
      void* base = nullptr;
      HIP_CHECK(hipIpcOpenMemHandle(&base, h, hipIpcMemLazyEnablePeerAccess));
      it = opened_.emplace(handle, static_cast<char*>(base)).first;
    }
    return it->second + offset;
  }

  // EVERY PEER INPUT OPENED AT REGISTRATION closed (borrowed, as a peer's
  // arena is): with the peers' arenas, in teardown's first half. Idempotent.
  Arena::Released release_inputs() {
    Arena::Released out;
    for (auto& kv : opened_) {
      const hipError_t e = hipIpcCloseMemHandle(kv.second);
      if (e != hipSuccess) {
        (void)hipGetLastError();
        if (out.error == hipSuccess) out.error = e;
      }
      ++out.closed;
    }
    opened_.clear();
    return out;
  }
  // THE REGIONS' DEVICE ARRAYS freed (ours, local): with our arena, in
  // teardown's second half. Idempotent.
  Arena::Released release_region_ptr_arrays() {
    Arena::Released out;
    if (region_ptr_arrays_ != nullptr) {
      const hipError_t e = hipFree(region_ptr_arrays_);
      if (e != hipSuccess) {
        (void)hipGetLastError();
        out.error = e;
      }
      region_ptr_arrays_ = nullptr;
    }
    return out;
  }
  // THE LOCAL SUMS freed (ours, local): with our arena, in teardown's second
  // half. Idempotent.
  Arena::Released release_sums() {
    Arena::Released out;
    if (sums_ != nullptr) {
      const hipError_t e = hipFree(sums_);
      if (e != hipSuccess) {
        (void)hipGetLastError();
        out.error = e;
      }
      sums_ = nullptr;
    }
    return out;
  }
  // THE SLAB OF INPUT TABLES freed (ours, local): with our arena, in
  // teardown's second half. Idempotent.
  Arena::Released release_tables() {
    Arena::Released out;
    if (tables_ != nullptr) {
      const hipError_t e = hipFree(tables_);
      if (e != hipSuccess) {
        (void)hipGetLastError();
        out.error = e;
      }
      tables_ = nullptr;
    }
    return out;
  }
  static void add(Arena::Released& into, const Arena::Released& more) {
    into.closed += more.closed;
    into.freed += more.freed;
    if (into.error == hipSuccess) into.error = more.error;
  }

  // WHAT THE RANKS SAID ABOUT THEMSELVES, compared: two ranks on one GPU, a
  // rank on another host. Warnings only.
  void observe_peers(const std::vector<RankInfo>& ranks) {
    const RankInfo& me = ranks[rank_];
    for (int r = 0; r < world_size_; ++r) {
      if (r == rank_) continue;
      const RankInfo& p = ranks[r];
      const bool same_host =
          std::strncmp(p.host, me.host, sizeof(me.host)) == 0;
      if (!same_host) log_->warn(OpenWarning::host_mismatch, r);
      if (same_host &&
          std::strncmp(p.pci_bus_id, me.pci_bus_id, sizeof(me.pci_bus_id)) == 0)
        log_->warn(OpenWarning::duplicate_device, r,
                   static_cast<uint32_t>(p.pid), static_cast<uint32_t>(me.pid));
    }
  }

  // OPEN FAILS, naming the check (already logged) and the phase reached.
  [[noreturn]] void fail(OpenCheck c, const std::string& why) const {
    throw std::runtime_error(
        std::string("hip_custom_communicator: open check failed: ") +
        to_string(c) + ": " + why + " (rank " + std::to_string(rank_) + ")");
  }

  // EVERY LAUNCH ON THIS DEVICE FINISHED within `seconds`, or false: the
  // synchronize runs on its own thread, left behind when it does not return
  // (the process is going; a peer may be dead).
  bool synchronize_within(double seconds) const {
    auto done = std::make_shared<std::promise<hipError_t>>();
    std::future<hipError_t> finished = done->get_future();
    const int device = device_;
    std::thread([done, device] {
      (void)hipSetDevice(device);
      done->set_value(hipDeviceSynchronize());
    }).detach();
    if (finished.wait_for(std::chrono::duration<double>(seconds)) !=
        std::future_status::ready)
      return false;
    const hipError_t e = finished.get();
    if (e != hipSuccess) (void)hipGetLastError();
    return e == hipSuccess;
  }

  // THE PROCESS'S OPEN HANDLES, for its exit hook: a handle enlists once open
  // and delists when it goes. Never destroyed (it is read at exit).
  struct Registry {
    std::mutex mutex;
    std::vector<Handle*> open;
  };
  static Registry& registry() {
    static auto* r = new Registry;
    return *r;
  }
  // PROCESS EXIT ABANDONS what is still open (registered once, after the HIP
  // runtime started, so it runs before the runtime's own teardown).
  void enlist() {
    Registry& r = registry();
    std::lock_guard<std::mutex> g(r.mutex);
    r.open.push_back(this);
    static std::once_flag hooked;
    std::call_once(hooked, [] {
      std::atexit([] { abandon_all(kAbandonWaitSeconds); });
    });
  }
  void delist() {
    Registry& r = registry();
    std::lock_guard<std::mutex> g(r.mutex);
    r.open.erase(std::remove(r.open.begin(), r.open.end(), this),
                 r.open.end());
  }

  // A STREAM'S AND A KERNEL'S INDEX in the log, recorded the first time each
  // is seen with its address (stream_seen, kernel_seen).
  uint16_t stream_index(hipStream_t stream) {
    for (size_t i = 0; i < streams_seen_.size(); ++i)
      if (streams_seen_[i] == stream) return static_cast<uint16_t>(i);
    streams_seen_.push_back(stream);
    const uint16_t i = static_cast<uint16_t>(streams_seen_.size() - 1);
    const auto a = reinterpret_cast<uintptr_t>(stream);
    log_->write({static_cast<uint8_t>(LogKind::stream_seen), 0, i, 0, 0, 0,
                 static_cast<uint32_t>(a), static_cast<uint32_t>(a >> 32), 0,
                 host_ns()});
    return i;
  }
  uint16_t kernel_index(const void* kernel) {
    for (size_t i = 0; i < kernels_seen_.size(); ++i)
      if (kernels_seen_[i] == kernel) return static_cast<uint16_t>(i);
    kernels_seen_.push_back(kernel);
    const uint16_t i = static_cast<uint16_t>(kernels_seen_.size() - 1);
    const auto a = reinterpret_cast<uintptr_t>(kernel);
    log_->write({static_cast<uint8_t>(LogKind::kernel_seen), 0, 0, 0, i, 0,
                 static_cast<uint32_t>(a), static_cast<uint32_t>(a >> 32), 0,
                 host_ns()});
    return i;
  }

  int rank_;
  int world_size_;
  int device_;
  int64_t max_rows_;
  int64_t max_cols_;
  int64_t staging_bytes_;
  int64_t partials_bytes_;
  int quantization_;
  std::shared_ptr<Log> log_;
  // THE ARENA'S REGIONS (`arena_layout`), fixed at construction.
  ArenaLayout layout_;
  // KERNELS AND STREAMS BY INDEX, as the log names them (kernel_seen,
  // stream_seen): first seen, first numbered.
  std::vector<const void*> kernels_seen_;
  std::vector<hipStream_t> streams_seen_;
  // The arena itself, made at open, released by close, abandon or the
  // destructor.
  std::optional<Arena> arena_;
  // One teardown at a time: close, abandon (a signal, exit) and the
  // destructor.
  std::mutex teardown_;
  uint32_t* status_ = nullptr;
  uint32_t* device_status_ = nullptr;
  uint64_t timeout_ticks_ = 0;
  mutable std::map<std::pair<const void*, int>, int> resident_;
  // THE INPUT TABLES (INPUT TABLES): the slab, the next free slot, the
  // captured launches awaiting registration (input, slot), and every peer
  // allocation opened for one, by its IPC handle's bytes.
  void** tables_ = nullptr;
  // The regions' device arrays, kRegionPtrArrays of kMaxRanks entries, in one
  // allocation: the signals, the inputs, the partials.
  static constexpr int kSignalsArray = 0, kStagingArray = 1, kPartialsArray = 2, kScalesArray = 3,
                       kRegionPtrArrays = 4;
  char** region_ptr_arrays_ = nullptr;
  // THE LOCAL SUMS (`sums`), staging_bytes_ of them.
  void* sums_ = nullptr;
  int64_t next_slot_ = 0;
  std::vector<const void*> pending_;
  std::vector<void**> pending_slots_;
  std::unordered_map<std::string, char*> opened_;
};

// =================================================================================================
// WHETHER A HANDLE CAN EXIST HERE (`supported`): whether the library runs on a
// device and world, asked once before anything is opened. The build answers,
// not a list: a device with no code object for our kernels is not built, and
// one that is but is not select's target would run on another device's tuning.
// =================================================================================================

// WHAT THE LIBRARY RUNS ON, once `supported` finds it can: the device's arch,
// as HIP names it.
struct Supported {
  std::string arch;
};

inline std::variant<Supported, Error> supported(int device, int world) {
  if (!world_built(world)) return Error::world_not_built;
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, device) != hipSuccess)
    return Error::device_not_built;
  const std::string name = prop.gcnArchName;
  const std::string arch =
      name.substr(0, name.find(':'));  // gfx950:sramecc+:xnack-
  // A CODE OBJECT FOR THIS DEVICE: any of our kernels has one exactly when the
  // build covered it.
  int was = 0;
  HIP_CHECK(hipGetDevice(&was));
  HIP_CHECK(hipSetDevice(device));
  hipFuncAttributes attrs;
  const hipError_t found = hipFuncGetAttributes(
      &attrs, reinterpret_cast<const void*>(
                  all_reduce_pull_one_shot<
                      c10::BFloat16, 2, 1, kWaveSize * traits<c10::BFloat16>::N,
                      kWaveSize, 1, Quantization::none>));
  (void)hipGetLastError();
  HIP_CHECK(hipSetDevice(was));
  if (found != hipSuccess) return Error::device_not_built;
  if (arch != kTargetArch) return Error::device_not_tuned;
  return Supported{arch};
}

}  // namespace hip_custom_communicator
