// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE HOST SIDE, one file: the state log, the types, the handle (the communicator's state and
// its open), the configs (the build catalog, each op's config from a whole config point), and the
// API (each op checks its call and its config, and launches its kernel). No table and no pick:
// Python's (the tuned table and its lookup), handing C++ every field. The device
// side is kernels/. Each section below was a file of its own, in this order (their includes).

#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <hip/hip_runtime.h>

#include <dirent.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "kernels/all_reduce.cuh"
#include "kernels/all_reduce_add_attn_res_rms_norm.cuh"
#include "kernels/all_reduce_add_rms_norm.cuh"
#include "kernels/common.cuh"
#include "kernels/reduce_scatter.cuh"

// =================================================================================================
// HARDWARE (was hardware.cuh)
// =================================================================================================

// THE MACHINE AND THE BUILD, host facts (comms.md's DECIDED, H8): the target's documented
// facts (HARDWARE), what the build compiles for it (BUILD), and the limits the host sizes the
// peers' buffers by (kMaxRanks, kMaxBlocks). The device code holds none of them: every kernel is
// handed the wave's lanes (kTarget.wave_size) and the pack's bytes (kBuild.memory.pack_bytes) as
// template parameters (a Config's wave_size and pack_bytes, CONFIG).




namespace hip_custom_communicator {

// =================================================================================================
// HARDWARE
//
// THE HARDWARE: each target's facts, as the device and AMD's docs report them.
// No decision lives here: BUILD derives what a build is from them,
// config.cuh a launch from them, the input and each op's tuned kernels
// (measured, each citing its sweep). A new target is one more `Hardware`;
// `kTarget` is the one the host tunes for.
// =================================================================================================

constexpr int64_t kKiB = 1024;
constexpr int64_t kMiB = 1024 * kKiB;
constexpr int64_t kGiB = 1024 * kMiB;

struct Hardware {
  // The compute units.
  int wave_size;         // lanes per wave
  int max_workgroup;     // threads per block the device allows
  int compute_units;     // across the device
  int max_waves_per_cu;  // resident at once
  int xcds;              // compute dies; each has its own L2
  int simds_per_cu;      // a wave runs on one SIMD
  // Per CU.
  int64_t vgpr_file_bytes;  // vector registers, split over the SIMDs (4-byte
                            // registers, per lane)
  int64_t sgpr_file_bytes;  // scalar registers
  int64_t lds_bytes;        // shared memory
  int64_t l1_bytes;         // vector L1
  // The memory system.
  int cache_line_bytes;
  int64_t l2_bytes_per_xcd;  // not shared across XCDs: a release to peers
                             // writes it back
  int64_t infinity_cache_bytes;
  int64_t hbm_bytes;
  double hbm_gbytes_per_s;  // peak
  // The fabric to the peers.
  int xgmi_links;                  // one to each peer in an 8-GPU node
  double xgmi_gbytes_per_s_a_way;  // peak, one link, one direction
  // The instruction set.
  int max_load_bytes;  // the widest vector load or store a lane issues
  int arch_vgprs;      // vector registers an instruction names (v0 up)
  int acc_vgprs;       // accumulation registers beside them (a0 up), where the
                       // compiler spills first
  int vgpr_granule;  // a wave's vector registers are allocated in multiples of
                     // it
};

// gfx950, AMD Instinct MI355X (MI350X is the same). Sources: `rocminfo` on n11
// (2026-09-29, all 8 GPUs) for the compute units; ROCm's GPU hardware
// specifications table (gpu-arch-specs) for the register files, caches and LDS;
// ROCm's MI350 architecture page (gpu-arch/mi350) for the XCDs, HBM and fabric.
// A link is 16 lanes at 38.4 Gbps: 76.8 GB/s a way, 7 of them "over 1 TB/s".
constexpr Hardware kGfx950 = {
    64,             // wave_size (gfx9 runs 64-lane waves only)
    1024,           // max_workgroup
    256,            // compute_units (32 active per XCD)
    32,             // max_waves_per_cu
    8,              // xcds
    4,              // simds_per_cu
    512 * kKiB,     // vgpr_file_bytes
    12800,          // sgpr_file_bytes (12.5 KiB)
    160 * kKiB,     // lds_bytes
    32 * kKiB,      // l1_bytes
    128,            // cache_line_bytes
    4 * kMiB,       // l2_bytes_per_xcd
    256 * kMiB,     // infinity_cache_bytes
    288 * kGiB,     // hbm_bytes (8 stacks of 36)
    8000.0,         // hbm_gbytes_per_s
    7,              // xgmi_links
    16 * 38.4 / 8,  // xgmi_gbytes_per_s_a_way
    16,             // max_load_bytes (global_load_dwordx4; CDNA4 ISA guide)
    256,            // arch_vgprs
    256,            // acc_vgprs
    8,              // vgpr_granule (LLVM AMDGPUUsage, gfx90a and later, wave64)
};
static_assert(kGfx950.compute_units % kGfx950.xcds == 0,
              "every XCD has the same CUs");

// THE TARGET: the one arch the build compiles device code for (gfx950,
// CMakeLists.txt) and the host tunes for; `supported` refuses every other.
constexpr const Hardware& kTarget = kGfx950;
constexpr const char* kTargetArch = "gfx950";

// THE MOST COMPUTE UNITS ON THE TARGET: for a layout host and device share
// (the peers' signal block).
constexpr int kMaxComputeUnits = kTarget.compute_units;

// THE MOST BLOCKS RESIDENT AT ONCE ON THE TARGET: every CU full of
// one-wave blocks, so the peers' signal block never rules out a grid. Whether
// one kernel's grid is resident is its occupancy, which only the compiled
// kernel knows (HIP's occupancy API, Handle::check_kernel, checked by
// select).
constexpr int resident_waves(const Hardware& hw) {
  return hw.compute_units * hw.max_waves_per_cu;
}
constexpr int kMaxResidentBlocks = resident_waves(kTarget);

// THE MOST GPUS THAT CAN READ EACH OTHER'S MEMORY DIRECTLY: a link to every
// peer (the full xGMI mesh of one node), so a peer group is at most the links
// plus one. Beyond it there is no load path, only a network collective.
constexpr int kMaxPeers = kTarget.xgmi_links + 1;


// =================================================================================================
// BUILD
//
// THE BUILD: every number fixed at compile time, derived from the device's
// `Hardware` (HARDWARE) and nothing else; what depends on the input is
// select's (config.cuh), at run time. A number neither gives is a policy: one
// named line in `derive`.
// =================================================================================================

// A TENSOR'S ELEMENT TYPE, as the build names it. f32 is a norm weight's, not a
// call's.
enum class DType { f16, bf16, f32 };

constexpr const char* to_string(DType d) {
  switch (d) {
    case DType::f16:
      return "f16";
    case DType::bf16:
      return "bf16";
    case DType::f32:
      return "f32";
  }
  return "unknown";
}

// EVERYTHING THE BUILD FIXES, in one value: what is compiled, the memory and
// its unit, and the kernels' launch geometry. Python reads a projection of it
// (`build_info`).
struct BuildInfo {
  struct Supports {
    std::array<DType, 2>
        dtypes;  // a call's; dispatch instantiates exactly these
    std::array<int, 3> worlds;
  };
  struct Memory {
    int pack_bytes;  // a pack: the unit every kernel loads, sums and stores in
  };
  Supports supports;
  Memory memory;
};

constexpr BuildInfo derive(const Hardware& hw) {
  BuildInfo info{};
  // WHAT IS COMPILED, one list each: dispatch instantiates exactly these, check
  // refuses the rest.
  info.supports = {{DType::f16, DType::bf16}, {2, 4, 8}};

  BuildInfo::Memory& m = info.memory;
  // A PACK IS THE WIDEST LOAD, as in vLLM's and aiter's custom all-reduce and
  // NCCL.
  m.pack_bytes = hw.max_load_bytes;
  // THE EXCHANGE (partials and inputs) IS NOT HERE: it is sized at open from
  // the largest call the communicator serves (config.cuh's `exchange_sizes`),
  // never a constant.
  // THE BLOCK IS NOT HERE EITHER: a build's block is its config's, any the device
  // launches (the forcing's range, max_workgroup). A wide block is a register
  // trade (at 1024 threads a gfx950 thread gets 128 and the row kernels spill:
  // AttnRes at 2 packs, ISA 2026-09-29T23-48-40Z), which the tune measures; it
  // is no refusal (STANDARDS 3.2).

  return info;
}

// THIS COMPILE PASS'S BUILD: the device code for its own target, the host for
// the tuning target.
constexpr BuildInfo kBuild = derive(kTarget);

// WHETHER THE BUILD HOLDS a dtype or a world: dispatch instantiates exactly
// these.
template <typename ELEMENT, size_t NUM_BUILT>
constexpr bool built_in(const std::array<ELEMENT, NUM_BUILT>& built,
                        ELEMENT x) {
  for (const ELEMENT& b : built)
    if (b == x) return true;
  return false;
}
constexpr bool dtype_built(DType d) {
  return built_in(kBuild.supports.dtypes, d);
}
constexpr bool world_built(int world) {
  return built_in(kBuild.supports.worlds, world);
}



// THE PEERS' LIMITS, for the layouts host and device share (the signal block, handle.cuh): at
// most kMaxPeers ranks; ONE SIGNAL SLOT PER BLOCK THE HARDWARE CAN HOLD RESIDENT, so the signal
// block never rules out a grid. Which grid is fast is select's (config.cuh).
constexpr int kMaxRanks = kMaxPeers;
constexpr int kMaxBlocks = kMaxResidentBlocks;

}  // namespace hip_custom_communicator


// =================================================================================================
// LOG (was log.cuh)
// =================================================================================================

// THE STATE LOG, one per communicator: the status words the kernels write
// today (common.cuh's StatusField, unchanged at the front), then a header, then
// a ring of fixed 32-byte records. Host-mapped and coherent, so the host reads
// it while a kernel spins, and it outlives a failed open (`last_log`) so the
// reason an open failed, or where it hangs, can be read.
//
//   words [0, kStatusFields)   the kernels' status: abort, the fault (as today)
//   words [16, 32)             the header (LogHeader)
//   bytes [kLogRecordsOffset, +kLogCapacity * 32)  the ring of LogRecords
//
// A record's slot is claimed by an atomic add on the cursor, so writers never
// share a slot; the record's `kind` is stored last, released, so a reader that
// sees a kind sees its fields (until the ring wraps over it). The ring keeps
// the newest kLogCapacity records; `dropped` counts the ones overwritten. The
// host orders host records by their clock (CLOCK_MONOTONIC ns) and, from the
// next cycle, kernel records by (launch, clock).
//
// TWO MODES. errors (serving's default): only failures and open's warnings are
// recorded; the header (phase, launch counter) is written always. everything:
// every kind. KERNEL RECORDS (the kinds from `entered` on) are reserved here
// for the next cycle: a kernel claims a slot the same way (an atomic add on
// the header's cursor at system scope) and writes the same layout, its clock
// the device's wall clock.





namespace hip_custom_communicator {

// THE FAULT SITES' NAMES (common.cuh's Where, the status's `where`), for the host's reports.
constexpr int kWhereCount = 2;  // numbered from 1
constexpr const char* to_string(Where w) {
  switch (w) {
    case Where::start:
      return "start";
    case Where::peer_barrier:
      return "peer_barrier";
  }
  return "?";
}

enum class LogMode : uint32_t { errors = 0, everything = 1 };

constexpr uint32_t kLogMagic = 0x474F4C48;  // "HLOG", little-endian
constexpr uint32_t kLogVersion = 1;

// THE HEADER'S WORDS, after the kernels' status words (which end below 16).
enum class LogHeader : int {
  magic = 16,
  version,
  mode,  // a LogMode
  rank,
  world,
  device,             // this rank's device, as this process numbers it
  timeout_ticks_low,  // what the kernels are given, in the wall clock's ticks
  timeout_ticks_high,
  clock_khz,       // the device wall clock's rate
  open_phase,      // the last OpenPhase reached
  launches,        // the launch counter: host-written before each launch
  cursor,          // records ever claimed
  dropped,         // records overwritten by the ring
  capacity,        // the ring's records
  records_offset,  // the ring's byte offset
  failed,          // 0, or the OpenCheck that failed open
  close_phase,     // the last ClosePhase reached: the memory's teardown
};
static_assert(kStatusFields <= 16, "the status words end before the header");
constexpr int kLogHeaderEnd = 33;
constexpr int64_t kLogRecordsOffset = 256;  // bytes
constexpr int64_t kLogCapacity = 4096;      // records

// WHAT A RECORD IS. Host kinds below 16, kernel kinds from 16 (next cycle).
enum class LogKind : uint8_t {
  empty = 0,
  open_phase = 1,      // barrier: the OpenPhase; seen, want: the phase's
                       // numbers; extra: 0, or the hipError_t it failed with
  open_check = 2,      // barrier: the OpenCheck; peer; seen, want; extra: 1 if
                       // it failed (open fails), else 0
  open_warning = 3,    // barrier: the OpenWarning; peer; seen, want
  launch_issued = 4,   // launch; kernel: the kernel's index (kernel_seen);
                       // block: the OpType; barrier: bit 0 capturing;
                       // peer: the stream's index (stream_seen);
                       // seen, want: m, n; extra: the DType
  copy_in_issued = 5,  // an eager plain all-reduce's host copy of its input
                       // into the inputs region; launch: the one it feeds;
                       // peer: the stream's index; seen, want: its bytes'
                       // low and high words
  stream_seen = 6,     // peer: the stream's index; seen, want: its handle's
                       // low and high words
  kernel_seen = 7,     // kernel: its index; seen, want: its address's words
  deadline_expired = 8,  // reserved: a host deadline (what: extra)
  close_phase = 9,       // barrier: the ClosePhase; peer: 1 collective (close),
                         // 0 alone (abandon: an error, a signal, exit);
                         // seen, want: the phase's numbers; extra: 0, or what
                         // it failed with (a hipError_t, or 1)
  registration = 10,     // the captured inputs registered at capture exit
                         // (Handle::register_captured); barrier: 0 begun,
                         // 1 done, 2 refused (the ranks' counts differ);
                         // seen: inputs recorded; want: peer allocations
                         // opened (done), or a peer's count (refused);
                         // extra: 1 refused
  // Kernel kinds, the next cycle's.
  entered = 16,
  barrier_signal = 17,
  barrier_passed = 18,
  wait_timeout = 19,
  aborted = 20,
  exited = 21,
};

// THE PHASES OF OPEN, in order; the header holds the last reached.
enum class OpenPhase : uint8_t {
  none = 0,
  log_ready,           // seen: the ring's capacity
  arena_allocated,     // the one allocation (hipExtMallocWithFlags); seen:
                       // its KiB
  exchange_allocated,  // retired (one arena); kept so the numbers hold
  zeroed,              // this rank's signals read back zero
  exported,            // hipIpcGetMemHandle on the arena's base
  handle_exchange_begun,  // seen: peers expected
  handle_exchange_done,   // seen: peers' records gathered
  opened,                 // seen: peers' handles opened (hipIpcOpenMemHandle)
  mapped,          // retired (VMM's map; IPC opens in one step)
  access_set,      // retired (VMM's access; IPC's is the lazy peer flag)
  tables_written,  // the input tables' slab allocated and its eager table
                   // (every rank's inputs region) written; seen: its slots
  checked,  // every OpenCheck passed; seen: warnings
  open_done,
};

// THE PHASE AFTER `p`, skipping retired ones: where an open that reached `p`
// stopped.
constexpr OpenPhase next_open_phase(OpenPhase p) {
  auto n = static_cast<uint8_t>(static_cast<uint8_t>(p) + 1);
  while (n == static_cast<uint8_t>(OpenPhase::exchange_allocated) ||
         n == static_cast<uint8_t>(OpenPhase::mapped) ||
         n == static_cast<uint8_t>(OpenPhase::access_set))
    ++n;
  return static_cast<OpenPhase>(n);
}

// THE PHASES OF THE MEMORY'S TEARDOWN, in order (STANDARDS 3.5); the header
// holds the last reached. CLOSE (collective, every rank): closing ->
// synchronized -> stopped (handshake 1: every rank has stopped using the
// memory) -> peers_released (our mappings of peers gone) -> peers_done
// (handshake 2: every rank's are) -> released (our own memory freed) ->
// close_done. ABANDON (alone: an error path, a signal, process exit;
// never waits on a peer): closing -> synchronized (bounded; failed when the
// device did not finish) -> peers_released -> released (only when the device
// finished) -> close_done. A failed handshake (a peer that did not come within
// the timeout) is recorded and the teardown carries on alone.
enum class ClosePhase : uint8_t {
  none = 0,
  closing,         // seen: 1 collective, 0 abandoned
  synchronized,    // every launch on this device finished; failed: it did not
                   // within the bounded wait (abandon)
  stopped,         // handshake 1 passed; failed: a peer did not arrive
  peers_released,  // seen: peers' handles closed (hipIpcCloseMemHandle);
                   // extra: the first hipError_t
  peers_done,      // handshake 2 passed; failed: a peer did not arrive
  released,        // seen: 1 when our arena was freed (hipFree); extra: the
                   // first hipError_t
  close_done,
};

// OPEN'S INTERNAL ASSERTS: each a definite error when it fails, and open
// fails naming it. Never a collective: each reads this process's own state.
enum class OpenCheck : uint8_t {
  none = 0,
  current_device,   // the current device is the one the communicator was
                    // given (seen: current, want: given)
  peer_opened,      // every rank's arena present: ours allocated, every
                    // peer's IPC handle opened
  pointer_sane,     // hipPointerGetAttributes on every rank's base: device
                    // memory, the base its device pointer (seen: the type,
                    // want: the hipError_t)
  geometry,         // retired (VMM's granule and stride); kept so the
                    // numbers hold
  can_access_peer,  // hipDeviceCanAccessPeer(this device, the peer's)
  signals_zero,     // this rank's signals read zero after the memset
};

// WHAT OPEN OBSERVES and only warns about.
enum class OpenWarning : uint8_t {
  none = 0,
  rccl_not_ready,        // the device group's RCCL communicator is not set up
  rccl_unknown,          // it could not be told
  other_processes,       // seen: other processes holding this GPU
  duplicate_device,      // peer: a rank on the same GPU as this one
  host_mismatch,         // peer: a rank on another host
  granularity_mismatch,  // retired (VMM's granularity); kept so the
                         // numbering holds
  low_memory,            // seen: free MiB, want: MiB open allocates
  pointer_device,        // peer; seen: the device hipPointerGetAttributes
                         // reports for its opened arena, want: its GPU here
  peer_not_visible,      // peer: its GPU (by PCI bus id) is not visible here,
                         // so its device checks are skipped
};

constexpr const char* to_string(OpenCheck c) {
  switch (c) {
    case OpenCheck::none:
      return "none";
    case OpenCheck::current_device:
      return "current_device";
    case OpenCheck::peer_opened:
      return "peer_opened";
    case OpenCheck::pointer_sane:
      return "pointer_sane";
    case OpenCheck::geometry:
      return "geometry";
    case OpenCheck::can_access_peer:
      return "can_access_peer";
    case OpenCheck::signals_zero:
      return "signals_zero";
  }
  return "?";
}

struct LogRecord {
  uint8_t kind;     // a LogKind; stored last
  uint8_t barrier;  // a barrier index, or a phase/check/warning
  uint16_t peer;
  uint32_t launch;
  uint16_t kernel;
  uint16_t block;
  uint32_t seen;
  uint32_t want;
  uint32_t extra;
  uint64_t clock;
};
static_assert(sizeof(LogRecord) == 32, "a record is 32 bytes");

inline uint64_t host_ns() {
  timespec t{};
  clock_gettime(CLOCK_MONOTONIC, &t);
  return uint64_t(t.tv_sec) * 1000000000ull + uint64_t(t.tv_nsec);
}

class Log {
 public:
  Log(LogMode mode, int rank, int world, int device) : mode_(mode) {
    const size_t bytes = kLogRecordsOffset + kLogCapacity * sizeof(LogRecord);
    const hipError_t e =
        hipHostMalloc(reinterpret_cast<void**>(&words_), bytes,
                      hipHostMallocMapped | hipHostMallocCoherent);
    if (e != hipSuccess)
      throw std::runtime_error(
          std::string("hip_custom_communicator: the log's hipHostMalloc -> ") +
          hipGetErrorString(e));
    std::memset(words_, 0, bytes);
    if (hipHostGetDevicePointer(reinterpret_cast<void**>(&device_words_),
                                words_, 0) != hipSuccess) {
      (void)hipHostFree(words_);
      throw std::runtime_error(
          "hip_custom_communicator: the log's hipHostGetDevicePointer failed");
    }
    set(LogHeader::mode, static_cast<uint32_t>(mode));
    set(LogHeader::rank, static_cast<uint32_t>(rank));
    set(LogHeader::world, static_cast<uint32_t>(world));
    set(LogHeader::device, static_cast<uint32_t>(device));
    set(LogHeader::capacity, static_cast<uint32_t>(kLogCapacity));
    set(LogHeader::records_offset, static_cast<uint32_t>(kLogRecordsOffset));
    set(LogHeader::version, kLogVersion);
    __atomic_store_n(&words_[static_cast<int>(LogHeader::magic)], kLogMagic,
                     __ATOMIC_RELEASE);
  }
  ~Log() { (void)hipHostFree(words_); }
  Log(const Log&) = delete;
  Log& operator=(const Log&) = delete;

  bool everything() const { return mode_ == LogMode::everything; }
  // The kernels' status buffer: the log's first words, at their device address.
  uint32_t* status_host() const { return words_; }
  uint32_t* status_device() const { return device_words_; }

  void set(LogHeader h, uint32_t v) {
    __atomic_store_n(&words_[static_cast<int>(h)], v, __ATOMIC_RELEASE);
  }
  uint32_t get(LogHeader h) const {
    return __atomic_load_n(&words_[static_cast<int>(h)], __ATOMIC_ACQUIRE);
  }
  void set_timeout(uint64_t ticks, int khz) {
    set(LogHeader::timeout_ticks_low, static_cast<uint32_t>(ticks));
    set(LogHeader::timeout_ticks_high, static_cast<uint32_t>(ticks >> 32));
    set(LogHeader::clock_khz, static_cast<uint32_t>(khz));
  }
  // The next launch's number, written before it is issued.
  uint32_t next_launch() {
    return __atomic_add_fetch(&words_[static_cast<int>(LogHeader::launches)],
                              1u, __ATOMIC_ACQ_REL);
  }
  uint32_t launches() const { return get(LogHeader::launches); }

  // A record, claimed and written whole, `kind` last. Callers filter by mode.
  void write(LogRecord r) {
    const uint32_t n = __atomic_fetch_add(
        &words_[static_cast<int>(LogHeader::cursor)], 1u, __ATOMIC_ACQ_REL);
    if (n >= kLogCapacity)
      __atomic_fetch_add(&words_[static_cast<int>(LogHeader::dropped)], 1u,
                         __ATOMIC_RELAXED);
    LogRecord* slot = records() + (n % kLogCapacity);
    const uint8_t kind = r.kind;
    r.kind = 0;
    __atomic_store_n(&slot->kind, uint8_t{0}, __ATOMIC_RELAXED);
    std::memcpy(slot, &r, sizeof(r));
    __atomic_store_n(&slot->kind, kind, __ATOMIC_RELEASE);
  }

  // OPEN'S PHASE `p` reached (`failed`: the hipError_t it failed with), always
  // in the header, a record in everything mode or when it failed.
  void phase(OpenPhase p, uint32_t seen = 0, uint32_t want = 0,
             uint32_t failed = 0) {
    if (failed == 0) set(LogHeader::open_phase, static_cast<uint32_t>(p));
    if (everything() || failed != 0)
      write({static_cast<uint8_t>(LogKind::open_phase), static_cast<uint8_t>(p),
             0, 0, 0, 0, seen, want, failed, host_ns()});
  }
  // THE TEARDOWN'S PHASE `p` (`failed`: what it failed with), in the header
  // when it did not fail, and ALWAYS a record: a teardown is rare and its
  // steps are what an abnormal exit is read by. `collective`: close (1) or
  // abandon (0).
  void close_phase(ClosePhase p, bool collective, uint32_t seen = 0,
                   uint32_t want = 0, uint32_t failed = 0) {
    if (failed == 0) set(LogHeader::close_phase, static_cast<uint32_t>(p));
    write({static_cast<uint8_t>(LogKind::close_phase), static_cast<uint8_t>(p),
           static_cast<uint16_t>(collective ? 1 : 0), 0, 0, 0, seen, want,
           failed, host_ns()});
  }
  void check(OpenCheck c, bool ok, int peer = 0, uint32_t seen = 0,
             uint32_t want = 0) {
    if (!ok) set(LogHeader::failed, static_cast<uint32_t>(c));
    if (everything() || !ok)
      write({static_cast<uint8_t>(LogKind::open_check), static_cast<uint8_t>(c),
             static_cast<uint16_t>(peer), 0, 0, 0, seen, want, ok ? 0u : 1u,
             host_ns()});
  }
  void warn(OpenWarning w, int peer = 0, uint32_t seen = 0, uint32_t want = 0) {
    write({static_cast<uint8_t>(LogKind::open_warning), static_cast<uint8_t>(w),
           static_cast<uint16_t>(peer), 0, 0, 0, seen, want, 0, host_ns()});
  }

  // Every word, header and ring, as the host reads it now.
  std::vector<uint32_t> words() const {
    const size_t n = (kLogRecordsOffset + kLogCapacity * sizeof(LogRecord)) / 4;
    std::vector<uint32_t> out(n);
    for (size_t i = 0; i < n; ++i)
      out[i] = __atomic_load_n(&words_[i], __ATOMIC_ACQUIRE);
    return out;
  }

 private:
  LogRecord* records() const {
    return reinterpret_cast<LogRecord*>(reinterpret_cast<char*>(words_) +
                                        kLogRecordsOffset);
  }
  LogMode mode_;
  uint32_t* words_ = nullptr;
  uint32_t* device_words_ = nullptr;
};

// THE LAST LOG THIS PROCESS MADE, kept past a failed open (the Handle is
// gone, the log is not) and readable while an open is still running. Never
// destroyed: freeing host memory after the runtime's teardown at exit is not
// safe.
inline std::mutex& last_log_mutex() {
  static auto* m = new std::mutex;
  return *m;
}
inline std::shared_ptr<Log>& last_log_slot() {
  static auto* p = new std::shared_ptr<Log>;
  return *p;
}
inline void set_last_log(std::shared_ptr<Log> log) {
  std::lock_guard<std::mutex> g(last_log_mutex());
  last_log_slot() = std::move(log);
}
inline std::shared_ptr<Log> last_log() {
  std::lock_guard<std::mutex> g(last_log_mutex());
  return last_log_slot();
}

}  // namespace hip_custom_communicator


// =================================================================================================
// TYPES (was types.cuh)
// =================================================================================================

// THE VOCABULARY, every type the tables and the handle name: why a call
// cannot run (`Error`), what is tuned and forced (`Template`, and each family's
// build, `KernelConfig`, its algorithm common.cuh's `Algorithm`), and what a caller
// asks for (`OpType`). `DType` is common.cuh's; what
// runs (`Config`) is config.cuh's.





namespace hip_custom_communicator {

// WHY A CALL CANNOT RUN, every reason there is. The numbers cross to Python
// (hip_custom_communicator.Error), so a reason is only ever added at the end.
// `disabled` and `no_such_op` are the communicator's own.
enum class Error : int {
  disabled = 0,
  no_such_op = 1,
  not_contiguous = 2,
  not_two_d = 3,
  dtype_not_built = 4,
  world_not_built = 5,
  row_not_packs = 6,
  template_not_this_ops = 7,
  row_too_wide = 8,  // retired: every kernel takes a row wider than its tile, in chunks
  block_not_a_wave_per_peer = 9,
  partials_too_small = 10,
  grid_not_resident = 11,
  staging_too_small = 12,
  device_not_built = 13,
  device_not_tuned = 14,
  weight_not_built = 15,
  no_such_template = 16,
  no_such_group = 17,
  ranks_disagree = 18,
  groups_disagree = 19,
  threads_not_built = 20,
  tile_not_built = 21,
  launch_out_of_range = 22,
  field_not_positive = 23,
  field_not_this_templates = 24,  // a forced tile_k other than 1 where the kernel has no K
  waves_not_built = 25,
  inner_stride_not_one = 26,
  row_stride_not_packs = 27,
  row_narrower_than_world = 28,  // retired: every two-shot takes a row narrower than a
                                // pack a rank (the late ranks' slices empty)
  forced_incomplete = 29,
  no_such_quantization = 30,
  quantization_not_supported = 31,
  row_not_int4_groups = 32,  // retired: every INT4 kernel takes a row of whole packs, its last
                             // group short
  quantized_too_small = 33,  // retired: INT4 is an encoding inside the
                             // regions (staging_too_small, partials_too_small)
  block_too_large = 34,
  lds_too_large = 35,
  scratch_too_large = 36,
  unfenced_too_small = 37,
  pack_out_of_range = 38,
};
constexpr int kNumErrors = 39;

constexpr const char* to_string(Error e) {
  switch (e) {
    case Error::disabled:
      return "disabled: the communicator is disabled";
    case Error::no_such_op:
      return "no_such_op: the backend has no such op";
    case Error::not_contiguous:
      return "not_contiguous: the input is not contiguous";
    case Error::not_two_d:
      return "not_two_d: an op takes a 2-D input, [rows, cols]";
    case Error::dtype_not_built:
      return "dtype_not_built: only float16 and bfloat16 are built";
    case Error::world_not_built:
      return "world_not_built: the world size is not 2, 4 or 8";
    case Error::row_not_packs:
      return "row_not_packs: the row is not whole packs of the config's pack_bytes (a "
             "build at a narrower pack takes it)";
    case Error::template_not_this_ops:
      return "template_not_this_ops: the forced template is not this op's";
    case Error::row_too_wide:
      return "row_too_wide: the row is wider than the template's widest build "
             "holds";
    case Error::block_not_a_wave_per_peer:
      return "block_not_a_wave_per_peer: a two-shot block must be one wave per "
             "peer";
    case Error::partials_too_small:
      return "partials_too_small: the two-shot's published sums, exact or INT4, "
             "exceed the partials";
    case Error::grid_not_resident:
      return "grid_not_resident: the grid exceeds the blocks the GPU holds "
             "resident";
    case Error::staging_too_small:
      return "staging_too_small: the input this kernel publishes whole, exact "
             "or INT4, exceeds the staging";
    case Error::device_not_built:
      return "device_not_built: this build holds no code for the device";
    case Error::device_not_tuned:
      return "device_not_tuned: the device is not the one select is calibrated "
             "for";
    case Error::weight_not_built:
      return "weight_not_built: a norm's weight is in the call's dtype or fp32";
    case Error::no_such_template:
      return "no_such_template: the op has no kernel of the forced algorithm";
    case Error::no_such_group:
      return "no_such_group: no process group is registered by that name";
    case Error::ranks_disagree:
      return "ranks_disagree: the ranks captured different numbers of buffers";
    case Error::groups_disagree:
      return "groups_disagree: the CPU and device groups differ in size or in "
             "this rank";
    case Error::threads_not_built:
      return "threads_not_built: no build of the template runs at that block "
             "size";
    case Error::waves_not_built:
      return "waves_not_built: no build of the template at that block size has "
             "those waves a EU";
    case Error::tile_not_built:
      return "tile_not_built: no build of the template has that TILE_M or "
             "TILE_N";
    case Error::launch_out_of_range:
      return "launch_out_of_range: the block is whole waves up to the device's "
             "workgroup and the grid at least 1 and at most the resident maximum "
             "(the pushed one-shot's, its unfenced region's blocks), each over the "
             "dimensions its kernel indexes (x alone: y and z 1)";
    case Error::field_not_positive:
      return "field_not_positive: a forced config field is positive";
    case Error::field_not_this_templates:
      return "field_not_this_templates: a forced tile_k other than 1 on a kernel "
             "with no K";
    case Error::inner_stride_not_one:
      return "inner_stride_not_one: a buffer's innermost stride is not 1 (the "
             "loads are packs)";
    case Error::row_stride_not_packs:
      return "row_stride_not_packs: a buffer's row stride is not whole packs of the "
             "config's pack_bytes (a build at a narrower pack takes it)";
    case Error::row_narrower_than_world:
      return "row_narrower_than_world: a two-shot row has fewer packs than "
             "ranks, so a rank's "
             "slice is empty";
    case Error::forced_incomplete:
      return "forced_incomplete: a forced call gives every field of its config "
             "or none (Python's refusal: C++ takes every field)";
    case Error::no_such_quantization:
      return "no_such_quantization: the quantization is exact or int4";
    case Error::quantization_not_supported:
      return "quantization_not_supported: INT4 is not supported for the selected "
             "kernel";
    case Error::row_not_int4_groups:
      return "row_not_int4_groups: an INT4 row is whole groups of 32 for every "
             "rank";
    case Error::quantized_too_small:
      return "quantized_too_small: retired";
    case Error::block_too_large:
      return "block_too_large: the block has more threads than the compiled kernel "
             "takes (its launch bounds, hipFuncGetAttributes' maxThreadsPerBlock)";
    case Error::lds_too_large:
      return "lds_too_large: the compiled kernel's LDS and the launch's dynamic LDS "
             "together are more than the device gives a block";
    case Error::scratch_too_large:
      return "scratch_too_large: the compiled kernel's scratch a lane is more than "
             "the device's scratch (hipExtLimitScratchMax) over its every lane, by "
             "kScratchShares (gfx950: 4 KiB a lane)";
    case Error::unfenced_too_small:
      return "unfenced_too_small: a pushed one-shot's tile does not fit the unfenced "
             "region: a row of it past a slot's kUnfencedSlotBytes of payload, or its "
             "grid's rows past a bank's kUnfencedBlocks";
    case Error::pack_out_of_range:
      return "pack_out_of_range: a config's pack_bytes is not 2, 4, 8 or 16 (a load the "
             "hardware has, up to its widest), or is less than an element of the dtype";
  }
  return "unknown";
}

// EVERY `__global__` TEMPLATE THERE IS, ONE AN OP, named for what it computes: each op's one
// kernel, a family of instances, one per set of template arguments. Its ALGORITHM (common.cuh's
// `Algorithm`: a one-shot or a two-shot, pulled or pushed, or rows) is a template parameter and a
// build's field, as its tile is: the algorithms a kernel has a body for are listed beside it
// (kAlgorithms_<kernel>, below).
enum class Template : int {
  all_reduce = 0,
  all_reduce_rms_norm = 1,
  all_reduce_add_rms_norm = 2,
  all_reduce_add_attn_res_rms_norm = 3,
  rms_norm_rows = 4,
  add_rms_norm_rows = 5,
  add_attn_res_rms_norm_rows = 6,
};

// AN ALGORITHM'S PARTS, where a host decision needs only one: whether it is a one-shot or a
// two-shot, and whether it pushes (a norm on rows pulls from no one).
constexpr bool is_one_shot(int algorithm) {
  return algorithm == Algorithm::one_shot_pull || algorithm == Algorithm::one_shot_push;
}
constexpr bool is_two_shot(int algorithm) {
  return algorithm == Algorithm::two_shot_pull || algorithm == Algorithm::two_shot_push;
}
constexpr bool is_push(int algorithm) {
  return algorithm == Algorithm::one_shot_push || algorithm == Algorithm::two_shot_push;
}

// WHICH OP the caller asked for: an all-reduce, alone or with what it fuses, or a norm alone on
// this rank's rows (the second kernel of an op Python composes).
enum class OpType : int {
  all_reduce = 0,
  all_reduce_rms_norm = 1,
  all_reduce_add_rms_norm = 2,
  all_reduce_add_attn_res_rms_norm = 3,
  rms_norm = 4,
  add_rms_norm = 5,
  add_attn_res_rms_norm = 6,
};

// A KERNEL'S CONFIG, in two parts and every field always set: its BUILD, what
// is compiled in (one of the builds its template lists, config.cuh: its algorithm
// first, which of the op's kernel's bodies it is), and its
// LAUNCH, what is chosen at each call. A tuned kernel holds both, a forced call
// gives both, and select hands both to the launch; no field is ever 0 for "left
// to someone else".
//
// A LAUNCH DIMENSION, the kernel's dim3: x, y and z. Every kernel here indexes x
// alone, so its configs hold y = z = 1; one that indexed y or z would widen its
// space, nothing else.
struct Dim3 {
  int x = 1;
  int y = 1;
  int z = 1;
  bool operator==(const Dim3&) const = default;
};
constexpr int64_t volume(const Dim3& d) {
  return static_cast<int64_t>(d.x) * d.y * d.z;
}
// Whether `d` is x alone: y = z = 1, every kernel's today.
constexpr bool x_alone(const Dim3& d) { return d.y == 1 && d.z == 1; }

// EACH OP'S BUILD, Triton's autotune config less the grid: the block, its
// threads in each dimension (x: Triton's num_warps x 64), the waves a SIMD (an EU) the compiler fits its
// registers to (__launch_bounds__'s second value, Triton's waves_per_eu; 1 asks
// nothing), and the tile, TILE_M rows x TILE_N columns in elements; TILE_K is
// the reduced dimension a step (AttnRes's sources a step); `quantization`, what
// crosses the links (common.cuh's QUANTIZATION): a two-shot's build is listed
// as it is and moving INT4 (two configs of the table), forced like any field or
// named by a tuned point; `pack_bytes`, a pack's bytes, the unit every load, sum and store moves
// (PACK_BYTES: 16, the widest load, or 8, 4 or 2, a row's that is not whole wider packs, so
// every row has a build whose loads are aligned: PyTorch's and vLLM's vector width by alignment,
// tuned here like the tile). One type an op, so a config
// cannot cross ops, though three hold the same fields. all_reduce: a tile a
// work item, the grid striding over the buffer's.
struct AllReduceBuildConfig {
  int algorithm;  // an Algorithm
  Dim3 block;
  int waves_per_eu;
  int tile_m;
  int tile_n;
  int quantization = Quantization::exact;  // a Quantization
  int pack_bytes = kBuild.memory.pack_bytes;  // a pack's bytes (PACK_BYTES)
  bool operator==(const AllReduceBuildConfig&) const = default;
};
// all_reduce_rms_norm.
struct AllReduceRmsNormBuildConfig {
  int algorithm;  // an Algorithm
  Dim3 block;
  int waves_per_eu;
  int tile_m;
  int tile_n;
  int quantization = Quantization::exact;  // a Quantization
  int pack_bytes = kBuild.memory.pack_bytes;  // a pack's bytes (PACK_BYTES)
  bool operator==(const AllReduceRmsNormBuildConfig&) const = default;
};
// all_reduce_add_rms_norm.
struct AllReduceAddRmsNormBuildConfig {
  int algorithm;  // an Algorithm
  Dim3 block;
  int waves_per_eu;
  int tile_m;
  int tile_n;
  int quantization = Quantization::exact;  // a Quantization
  int pack_bytes = kBuild.memory.pack_bytes;  // a pack's bytes (PACK_BYTES)
  bool operator==(const AllReduceAddRmsNormBuildConfig&) const = default;
};
// all_reduce_add_attn_res_rms_norm: TILE_K sources a step.
struct AllReduceAddAttnResRmsNormBuildConfig {
  int algorithm;  // an Algorithm
  Dim3 block;
  int waves_per_eu;
  int tile_m;
  int tile_n;
  int tile_k;
  int quantization = Quantization::exact;  // a Quantization
  int pack_bytes = kBuild.memory.pack_bytes;  // a pack's bytes (PACK_BYTES)
  bool operator==(const AllReduceAddAttnResRmsNormBuildConfig&) const = default;
};
using BuildConfig =
    std::variant<AllReduceBuildConfig, AllReduceRmsNormBuildConfig,
                 AllReduceAddRmsNormBuildConfig,
                 AllReduceAddAttnResRmsNormBuildConfig>;

// A KERNEL'S LAUNCH: its grid, blocks in each dimension, never compiled in.
struct LaunchConfig {
  Dim3 grid;
};

// THE COMPLETE CONFIG: the build and the launch.
struct KernelConfig {
  BuildConfig build;
  LaunchConfig launch;
};

// THE FIELDS EVERY BUILD HAS.
constexpr int algorithm_of(const BuildConfig& b) {
  return std::visit([](const auto& f) { return f.algorithm; }, b);
}
constexpr Dim3 block_of(const BuildConfig& b) {
  return std::visit([](const auto& f) { return f.block; }, b);
}
// A block's threads, every dimension's.
constexpr int threads_of(const BuildConfig& b) {
  return static_cast<int>(volume(block_of(b)));
}
constexpr int waves_of(const BuildConfig& b) {
  return std::visit([](const auto& f) { return f.waves_per_eu; }, b);
}
constexpr int tile_m_of(const BuildConfig& b) {
  return std::visit([](const auto& f) { return f.tile_m; }, b);
}
constexpr int tile_n_of(const BuildConfig& b) {
  return std::visit([](const auto& f) { return f.tile_n; }, b);
}
constexpr int quantization_of(const BuildConfig& b) {
  return std::visit([](const auto& f) { return f.quantization; }, b);
}
constexpr int pack_bytes_of(const BuildConfig& b) {
  return std::visit([](const auto& f) { return f.pack_bytes; }, b);
}
// WHAT A BUILD'S INSTANCE IS COMPILED MOVING: its quantization, but an INT4 build at a pack
// narrower than the widest compiles its exact body (INT4's word is a pack of 8 values,
// common.cuh's QUANTIZATION), which never runs: the call is refused by name
// (quantization_not_supported), as an INT4 one-shot is. builds.py's instances alike.
template <typename B>
constexpr int compiled_quantization(const B& b) {
  return b.pack_bytes == kBuild.memory.pack_bytes ? b.quantization : Quantization::exact;
}
// The same build, moving `q` across the links.
constexpr BuildConfig quantized(BuildConfig b, int q) {
  std::visit([q](auto& f) { f.quantization = q; }, b);
  return b;
}

constexpr int elem_bytes(DType d) { return d == DType::f32 ? 4 : 2; }
// A row of `elems` of `dtype` in packs of `pack` bytes, the last short where the row is not
// whole packs (a build whose pack does not divide a call's row is refused, row_not_packs).
constexpr int64_t packs_of(int64_t elems, DType dtype, int64_t pack) {
  return (elems * elem_bytes(dtype) + pack - 1) / pack;
}

}  // namespace hip_custom_communicator


// =================================================================================================
// HANDLE (was handle.cuh)
// =================================================================================================

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
//                                         (two-shot only), then, at its end, the
//                                         add-RMS-norm pull two-shot's INT4 row
//                                         scales, a float a row (max_rows): a
//                                         view of the partials (`scales`), fenced
//                                         as they are
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
//
// THE UNFENCED REGION AND ITS EPOCHS: memory no barrier fences, each write carrying its own
// flag (the pushed one-shot's: kernels/all_reduce.cuh, the flag in the line), one more
// region after the partials:
//   unfenced   the INBOX peers push lines into: [writer, bank * row] rows of the blocks'
//              tiles, kUnfencedSlotBytes of payload a row, each payload byte two in lines
//              (common.cuh's Layout::line), two banks: world * 2 * kUnfencedBlocks rows.
//              Its own region, never a fenced kernel's: a stale word a fenced kernel left
//              could carry an epoch's value, and the reader would take it for data.
// THE EPOCHS are not a region: each row of a bank's last epoch, a 16-byte pack a row (its
// word at column 0), kUnfencedBlocks of them, read only by their own rank, so CACHED
// rank-local memory (hipMalloc, zeroed at open; aiter's block_flags): from the
// uncached arena each launch began with an uncached load (~2k cycles before any
// send, ATT 2026-10-19T04-09-17Z). The lines start zeroed with the arena (no line
// carries epoch 0, the first is 1); both are written only by that kernel. THE CAP: aiter's LL serves at most 128 KiB a call
// (kLLPocSlotCapPk, 16384 packets) and refuses more; ours takes any size, a pass of a
// tile a block, so its cap is the grid's, kUnfencedBlocks blocks (a grid past it is refused,
// launch_out_of_range): 128 blocks of 1024 elements, 256 KiB in one pass. 8 MiB at eight ranks.
constexpr int kUnfencedBlocks = 128;
constexpr int64_t kUnfencedSlotBytes = 2048;  // a block's tile's payload, at most
constexpr int64_t unfenced_bytes(int world) {
  return 2 * int64_t{kUnfencedBlocks} * world * kUnfencedSlotBytes * (kLineBytes / kLinePayloadBytes);
}
constexpr int64_t kUnfencedEpochsBytes = int64_t{kUnfencedBlocks} * kBuild.memory.pack_bytes;

constexpr int64_t kRegionAlign = 4096;
constexpr int64_t region_aligned(int64_t bytes) {
  return (bytes + kRegionAlign - 1) / kRegionAlign * kRegionAlign;
}
// THE GUARDS: after each region's last used byte, up to the next region, kGuardBytes at least,
// every word kGuardWord, written at open and never by a kernel: a write past a region (this rank's
// or a peer's into it) changes one, and a checked launch (VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_CHECKED)
// reads them after the kernel (October 26: n11 rebooted seven times timing the fused two-shots at
// large sizes, every result right and every index inside the sizes the host passed).
constexpr int64_t kGuardBytes = 4096;
constexpr uint32_t kGuardWord = 0xdeadbeefu;
struct ArenaLayout {
  int64_t signals = 0;
  int64_t staging = 0;
  int64_t partials = 0;
  int64_t scales = 0;  // inside the partials, at their end
  int64_t unfenced = 0;
  int64_t end = 0;  // the bytes the regions use: the arena's size
  // Each region's guard, [begin, end) bytes: after the staging, after the partials and their
  // scales, after the unfenced region.
  int64_t staging_guard = 0, staging_guard_end = 0;
  int64_t partials_guard = 0, partials_guard_end = 0;
  int64_t unfenced_guard = 0, unfenced_guard_end = 0;
};
inline ArenaLayout arena_layout(int64_t staging_bytes, int64_t partials_bytes,
                                int64_t scales_bytes, int64_t unfenced_bytes) {
  ArenaLayout l;
  l.signals = 0;
  l.staging = l.signals + kSignalsRegionBytes;
  l.staging_guard = l.staging + staging_bytes;
  l.partials = region_aligned(l.staging_guard + kGuardBytes);
  l.staging_guard_end = l.partials;
  l.scales = l.partials + partials_bytes;  // whole packs: the floats aligned
  l.partials_guard = l.scales + scales_bytes;
  l.unfenced = region_aligned(l.partials_guard + kGuardBytes);
  l.partials_guard_end = l.unfenced;
  l.unfenced_guard = l.unfenced + unfenced_bytes;
  l.end = region_aligned(l.unfenced_guard + kGuardBytes);
  l.unfenced_guard_end = l.end;
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

// THE SCRATCH A KERNEL MAY ASK, as a share of the device's: the device's scratch
// (hipExtLimitScratchMax) over its every lane, by this. A dispatch that asks too much fails only
// when it runs, aborting its queue and the process with it (the tune's crashes, 2026-10-24); HIP
// refuses nothing at the launch. ROCr gives a dispatch scratch for every wave slot of the device,
// a hardware queue keeps the most it was ever given, and the process's queues share the budget
// (measured on n11, one process: one queue ran 32 KiB a lane, not 64 -- the budget over every
// lane; at 16 KiB a lane the fourth queue failed, at 24 KiB the third). How many queues a process
// holds HIP does not say (a gd worker's held more than four: AttnRes at 14-16 KiB a lane still
// crashed), so the share is a margin, not a count: 16 queues at a full dispatch each before the
// budget runs out, past HIP's pools (four queues a stream priority, three priorities). At gfx950,
// 4 KiB a lane: the crashes spilled 14-25 KiB; the tuned table's build spills at most 1.8 KiB
// (all_reduce_rms_norm at 8192 a tile, 128 threads); of the coarse build's 6,974 kernels about
// 400 heavy spillers are refused.
constexpr int64_t kScratchShares = 16;

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
  // what the buffers were sized for can be asked. `device`: the device the communicator was given (open checks it is the
  // current one). `log_mode`: what the log records (log.cuh). `rccl_ready`:
  // whether the caller found the device group's RCCL communicator set up (1),
  // not (0), or could not tell (-1): observed and logged, never acted on.
  Handle(int rank, int world_size, int device, const Gather& gather,
         double timeout_seconds, int64_t max_rows, int64_t max_cols,
         int64_t staging_bytes, int64_t partials_bytes,
         LogMode log_mode, int rccl_ready)
      : rank_(rank),
        world_size_(world_size),
        device_(device),
        max_rows_(max_rows),
        max_cols_(max_cols),
        staging_bytes_(staging_bytes),
        partials_bytes_(partials_bytes),
        log_(std::make_shared<Log>(log_mode, rank, world_size, device)),
        layout_(arena_layout(staging_bytes, partials_bytes,
                             max_rows * static_cast<int64_t>(sizeof(float)),
                             unfenced_bytes(world_size))) {
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
    add(own, release_unfenced_epochs());
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
    add(own, release_unfenced_epochs());
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
      add(own, release_unfenced_epochs());
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
  bool checked() const { return checked_after_launch_; }
  // THIS RANK'S GUARDS, read back after a launch (the caller synchronized): "" where every word is
  // kGuardWord, else the first region whose guard changed and where.
  std::string guards_broken() const {
    for (const auto& [name, b, e] :
         {std::tuple{"staging", layout_.staging_guard, layout_.staging_guard_end},
          std::tuple{"partials", layout_.partials_guard, layout_.partials_guard_end},
          std::tuple{"unfenced", layout_.unfenced_guard, layout_.unfenced_guard_end}}) {
      std::vector<uint32_t> words(static_cast<size_t>((e - b) / 4));
      HIP_CHECK(hipMemcpy(words.data(), arena_->base(rank_) + b, words.size() * 4,
                          hipMemcpyDeviceToHost));
      for (size_t i = 0; i < words.size(); ++i)
        if (words[i] != kGuardWord)
          return std::string("the ") + name + " guard's word " + std::to_string(i) + " of " +
                 std::to_string(words.size()) + " is " + std::to_string(words[i]) +
                 " (a write past the " + name + " region)";
    }
    return "";
  }
  int64_t unfenced_region_bytes() const { return layout_.unfenced_guard - layout_.unfenced; }
  int64_t scales_region_bytes() const { return layout_.partials_guard - layout_.scales; }
  int world_size() const { return world_size_; }
  int64_t max_rows() const { return max_rows_; }
  int64_t max_cols() const { return max_cols_; }
  // Each region's bytes, what its largest payload in either encoding takes.
  int64_t staging_bytes() const { return staging_bytes_; }
  int64_t partials_bytes() const { return partials_bytes_; }

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
  // which its peers read while gathering), [max_rows] floats a rank at the partials' end (a view of
  // that region, past what `partials_bytes` counts): a device array of every rank's address of it
  // with its size, the world.
  float* const* scales_ptr_array() const {
    return reinterpret_cast<float* const*>(region_ptr_arrays_ + kScalesArray * kMaxRanks);
  }
  int scales_ptr_array_size() const { return world_size_; }
  int64_t scales_size_m() const { return max_rows_; }
  // EVERY RANK'S UNFENCED REGION (the pushed one-shot's inbox of lines), a device array of every rank's address of
  // that region with its size, the world, as [writer, row, n] (config.cuh's); and THIS RANK'S EPOCHS
  // (its per-block counters, read only here), their own pointer, as [block, n], a pack of int32 a
  // block (THE UNFENCED REGION AND ITS EPOCHS, above).
  char* const* unfenced_ptr_array() const { return region_ptr_arrays_ + kUnfencedArray * kMaxRanks; }
  int unfenced_ptr_array_size() const { return world_size_; }
  int32_t* unfenced_epochs_ptr() const { return unfenced_epochs_; }
  static constexpr int64_t unfenced_epochs_stride_block() {
    return kBuild.memory.pack_bytes / int64_t{sizeof(int32_t)};
  }
  static constexpr int64_t unfenced_epochs_stride_n() { return 1; }
  static constexpr int64_t unfenced_epochs_size_block() { return kUnfencedBlocks; }
  static constexpr int64_t unfenced_epochs_size_n() { return unfenced_epochs_stride_block(); }
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

  // WHY `kernel` CANNOT LAUNCH HERE, `grid` blocks of `block` threads and `shared_bytes` of dynamic
  // LDS (a wide AttnRes row's logits), if it cannot: every check of a launch, in one place, what
  // the compiled kernel asks (hipFuncGetAttributes, HIP's occupancy API) against what this device
  // gives.
  //  - block_too_large: more threads than the kernel takes (its launch bounds).
  //  - lds_too_large: its LDS and the launch's dynamic LDS, past the device's a block.
  //  - scratch_too_large: its scratch a lane past the device's scratch over its every lane (the
  //    CUs times the waves a CU holds times a wave's lanes), by kScratchShares (gfx950: 32 GiB
  //    over 256 x 32 x 64 lanes, 64 KiB a lane; by 16, 4 KiB). See kScratchShares.
  //  - grid_not_resident, where the kernel waits on its peers (`waits_on_peers`; a norm on rows
  //    waits on none, and any grid runs): more blocks than the device holds at once, the
  //    textbook query (CUDA's cudaOccupancyMaxActiveBlocksPerMultiprocessor: blocks a CU from
  //    the kernel's registers, LDS and the CU's wave slots) times the CUs. A peer barrier spins
  //    until the same block on every peer arrives, so a grid past it can hang.
  // HIP refuses the first two at the launch; the scratch only when the dispatch runs, aborting
  // the queue and the process with it (the tune's crashes, 2026-10-24: AttnRes at 64 threads,
  // tile 4x16384x4, 21 KiB a lane); the grid never (it hangs). Asked once per kernel, block size
  // and dynamic LDS, then cached; the grid compared each call.
  std::optional<Error> check_kernel(const void* kernel, const Dim3& grid, const Dim3& block,
                                    int shared_bytes, bool waits_on_peers) const {
    const int threads = static_cast<int>(volume(block));
    const std::tuple<const void*, int, int> key{kernel, threads, shared_bytes};
    auto it = checked_.find(key);
    if (it == checked_.end()) {
      hipFuncAttributes f;
      HIP_CHECK(hipFuncGetAttributes(&f, kernel));
      int lds = 0, cus = 0, threads_a_cu = 0;
      HIP_CHECK(hipDeviceGetAttribute(&lds, hipDeviceAttributeMaxSharedMemoryPerBlock, device_));
      HIP_CHECK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, device_));
      HIP_CHECK(hipDeviceGetAttribute(&threads_a_cu,
                                      hipDeviceAttributeMaxThreadsPerMultiProcessor, device_));
      size_t budget = 0;  // the current device's: this one's, for the query
      int was = 0;
      HIP_CHECK(hipGetDevice(&was));
      HIP_CHECK(hipSetDevice(device_));
      HIP_CHECK(hipDeviceGetLimit(&budget, hipExtLimitScratchMax));
      HIP_CHECK(hipSetDevice(was));
      const int64_t lanes = static_cast<int64_t>(cus) * threads_a_cu;  // every wave slot's lanes
      const int64_t scratch_a_lane = static_cast<int64_t>(budget) / lanes / kScratchShares;
      const std::optional<Error> refused =
          threads > f.maxThreadsPerBlock ? std::optional<Error>(Error::block_too_large)
          : static_cast<int64_t>(f.sharedSizeBytes) + shared_bytes > lds
              ? std::optional<Error>(Error::lds_too_large)
          : static_cast<int64_t>(f.localSizeBytes) > scratch_a_lane
              ? std::optional<Error>(Error::scratch_too_large)
              : std::nullopt;
      int per_cu = 0;  // asked of a kernel that launches alone
      if (!refused)
        HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, kernel, threads,
                                                               shared_bytes));
      it = checked_.emplace(key, std::pair{refused, int64_t{per_cu} * cus}).first;
    }
    const auto& [refused, resident] = it->second;
    if (refused) return refused;
    if (waits_on_peers && volume(grid) > resident) return Error::grid_not_resident;
    return std::nullopt;
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
    // THE GUARDS, every word kGuardWord (`kGuardBytes`): before any peer opens the arena.
    for (const auto& [b, e] : {std::pair{layout_.staging_guard, layout_.staging_guard_end},
                               std::pair{layout_.partials_guard, layout_.partials_guard_end},
                               std::pair{layout_.unfenced_guard, layout_.unfenced_guard_end}}) {
      if (b % 4 != 0 || e % 4 != 0 || e - b < kGuardBytes)
        throw std::runtime_error("hip_custom_communicator: a guard is not whole words or too short");
      HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(arena_->base(rank_) + b),
                             static_cast<int>(kGuardWord), static_cast<size_t>((e - b) / 4)));
    }
    HIP_CHECK(hipDeviceSynchronize());
    // A CHECKED COMMUNICATOR (VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_CHECKED=1, read here once): every
    // launch outside a capture synchronizes and checks what it promises after the kernel.
    {
      const char* checked = std::getenv("VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_CHECKED");
      checked_after_launch_ = checked != nullptr && std::string(checked) == "1";
    }
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
        at[kUnfencedArray * kMaxRanks + r] = base + layout_.unfenced;
      }
      // THE EPOCHS: this rank's own hipMalloc (no peer reads them; a pointer of their own).
      HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&unfenced_epochs_), static_cast<size_t>(kUnfencedEpochsBytes)));
      HIP_CHECK(hipMemset(unfenced_epochs_, 0, static_cast<size_t>(kUnfencedEpochsBytes)));
      // Zero before any launch on the caller's streams, which do not wait for the null stream.
      HIP_CHECK(hipDeviceSynchronize());
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
  // THE EPOCHS freed (ours, local): with our arena, in teardown's second half. Idempotent.
  Arena::Released release_unfenced_epochs() {
    Arena::Released out;
    if (unfenced_epochs_ != nullptr) {
      const hipError_t e = hipFree(unfenced_epochs_);
      if (e != hipSuccess) {
        (void)hipGetLastError();
        out.error = e;
      }
      unfenced_epochs_ = nullptr;
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
  std::shared_ptr<Log> log_;
  // THE ARENA'S REGIONS (`arena_layout`), fixed at construction.
  ArenaLayout layout_;
  bool checked_after_launch_ = false;  // VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_CHECKED: each launch checked after
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
  // check_kernel's, by kernel, block size and dynamic LDS: why it cannot launch at all (if it
  // cannot), and the most of its blocks resident at once.
  mutable std::map<std::tuple<const void*, int, int>, std::pair<std::optional<Error>, int64_t>>
      checked_;
  // THE INPUT TABLES (INPUT TABLES): the slab, the next free slot, the
  // captured launches awaiting registration (input, slot), and every peer
  // allocation opened for one, by its IPC handle's bytes.
  void** tables_ = nullptr;
  // The regions' device arrays, kRegionPtrArrays of kMaxRanks entries, in one
  // allocation: the signals, the inputs, the partials.
  static constexpr int kSignalsArray = 0, kStagingArray = 1, kPartialsArray = 2, kScalesArray = 3,
                       kUnfencedArray = 4, kRegionPtrArrays = 5;
  char** region_ptr_arrays_ = nullptr;
  // THE EPOCHS (THE UNFENCED REGION AND ITS EPOCHS), kUnfencedEpochsBytes, this rank's alone.
  int32_t* unfenced_epochs_ = nullptr;
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
                  all_reduce<
                      c10::BFloat16, 2, 1,
                      kTarget.wave_size * traits<c10::BFloat16, kBuild.memory.pack_bytes>::N,
                      kTarget.wave_size, kTarget.wave_size, kBuild.memory.pack_bytes, 1,
                      Quantization::exact, Algorithm::one_shot_pull, false, false>));
  (void)hipGetLastError();
  HIP_CHECK(hipSetDevice(was));
  if (found != hipSuccess) return Error::device_not_built;
  if (arch != kTargetArch) return Error::device_not_tuned;
  return Supported{arch};
}

}  // namespace hip_custom_communicator


// =================================================================================================
// CONFIG (was config.cuh)
// =================================================================================================

// Built kernels, tuned tables, and the complete configuration of each call.





namespace hip_custom_communicator {

// THE FUSED OPS' KERNELS: an all-reduce and the norm or AttnRes after it, in one kernel.
constexpr bool fuses(Template k) {
  return k == Template::all_reduce_rms_norm || k == Template::all_reduce_add_rms_norm ||
         k == Template::all_reduce_add_attn_res_rms_norm;
}

// THE PULLED NORMS' INT4 IS WHOLE GROUPS A THREAD, A ROW SLICE A RANK (their INT4 body,
// all_reduce_add_rms_norm.cuh): a block a row, TILE_N / (32 * THREADS) groups a thread; the tile
// does not size the payload (a row's whole groups, below), so any build of theirs sizes alike.
constexpr bool int4_groups_a_thread(Template k, int algorithm) {
  return fuses(k) && algorithm == Algorithm::two_shot_pull;
}

// AN INT4 ROW WITH A TAIL, the INT4 two-shots' TAIL instance: the flat walk (the plain push
// two-shot) where the band is not whole groups; a groups-a-thread pulled norm where a row is not;
// a column two-shot where a row is not whole groups for every rank.
constexpr bool int4_tail(Template k, int algorithm, int64_t m, int64_t n, int world) {
  if (k == Template::all_reduce && algorithm == Algorithm::two_shot_push)
    return m * n % kInt4Group != 0;
  if (int4_groups_a_thread(k, algorithm)) return n % kInt4Group != 0;
  return n % kInt4Group != 0 || n / kInt4Group % world != 0;
}

// THE INT4 FLAT WALK'S SHARE (reduce_scatter.cuh, the plain push two-shot's INT4): an [m, n]
// input's elements numbered row-major, a rank's share whole groups, the last rank's to the end.
// Its payloads lie on views of their own, [1, world * share] in the inbox and [1, share] in the
// partials: on the input's [m, n] the rounded-up shares would run past its values.
constexpr int64_t int4_flat_share(int64_t m, int64_t n, int world) {
  return ((m * n + kInt4Group - 1) / kInt4Group + world - 1) / world *
         kInt4Group;
}

// =================================================================================================
// THE CATALOG: each template, and the BuildConfigs it is built for.
// =================================================================================================

// A TEMPLATE'S BUILDS, each of its op's BuildConfig (types.cuh): what dispatch
// instantiates, what the check accepts and what a tuner searches. The launch
// (the grid) is not a build's: each tuned kernel (below)
// gives its own, and a forced call gives its own. tile_n counts the elements of
// a 16-bit dtype, the only ones built. A build that spills shows in its code
// object (gp report isa) and loses on the clock; none is ruled out by
// policy. The sweeps cited beside a list set its builds and the grids its tuned
// kernels carry.

// THE PUSHED ONE-SHOT'S PACK: a line's payload (kernels/all_reduce.cuh), so its

// THE BUILDS, GENERATED FROM THE TUNED TABLE (builds.py, at configure): each kernel's,
// `kBuilds_<kernel>`, the build half of every config the table lists. A two-shot's INT4
// build is a config moving int4, listed as any other. WHY THE TABLE HOLDS THESE BUILDS, the
// sweeps that chose them (the lists they were, kept as their history):
//
// The norms: one-shot, the push two-shot (a column split), the pull two-shot (a
// row split). Not swept: select cuts it to the rows, so it matters only past 16
// rows. The plain one-shot: a tile a row high, a group a thread wide, at each
// block size a launch may take. The grid is the size's (kAllReduceKernels,
// which picks among these).
//
// THE PUSHED ONE-SHOT (the flag in the line): a tile a row high, a pack a
// thread; aiter's ar_ll_gfx9 runs 256 threads of one 8-byte packet each, 1024
// elements a block (7 blocks at 7168), which {128, 1, 1, 1024} matches block for
// block at a 16-byte pack a thread. Every build's tile fits a lines slot
// (handle.cuh's kUnfencedSlotBytes, checked below). Not swept yet.
//
// The plain two-shot: 512 threads divided into whole waves per peer. The
// eight-rank tile is 512 elements; four and two ranks need 1024 and 2048
// elements respectively so every lane owns a complete pack. Keep the tuned
// eight-rank build first; fallback picks use the smallest valid tile below.
//
// 256 the best of 48-256 at 192-256 tokens (15.17 and 18.04 against 15.55
// and 18.11 at 128; 2026-10-01T03-26-44Z); a row a block below that.
//
// Pipelined, 48 the best of 36-96 at 2048-4096 tokens (79.3 and 145.4
// against 81.0 and 147.6 at 36), within 0.8 of 36 below (2026-10-01T02-59-52Z).
// INT4 WHOLE GROUPS A THREAD (the kernel's INT4 body): a block a row's chunk of TILE_N columns,
// TILE_N / (32 * THREADS) groups a thread (a multiple of 32 * THREADS, any block).
//
// Not swept: rms_norm's.
//
// 256 the best of 48-256 at 192 tokens (15.48 against 16.75 at 128;
// 2026-10-01T03-26-44Z).
//
// 48 the best of 36-96 at every size from 512 to 4096 tokens (80.6 and 149.7 us
// at 2048 and 4096 against 84.3 and 154.7 at 36; 2026-10-01T02-59-52Z).
// INT4 whole groups a thread, as rms_norm's.
//
// AttnRes, one source a step (TILE_K): at 4 (Triton's tile) the row got
// slower, 6.48 -> 8.16 us a row and 147.5 -> 171.0 at 4096 tokens, with 100 ->
// 166 VGPRs (stamps 2026-10-01T06-26-55Z against 04-15-48Z). The one-shot's
// grid not swept: the norms'.
//
// 256 the best of 32-256 at 256-1024 tokens (34.58, 70.78, 143.98
// against 39.93, 83.15, 159.33 at 128), a row a block below that
// (2026-10-01T03-57-23Z). 256 threads a block: Triton's program for AttnRes,
// and more blocks resident at once (AttnRes from local memory took 192.8 /
// 109.6 / 86.0 us on 192 / 384 / 512 blocks, stamps 2026-10-02T20-36-56Z); its
// grid not swept yet.
//
// The column split wants a wide grid (AttnRes is compute a row): at 7168, 192
// is within about 5% of the best of 16-256 from 512 to 4096 tokens; 4096 at
// 447.2 us against 1160.7 at the 36 it had (2026-10-01T22-56-58Z,
// 2026-10-01T23-00-47Z). TILE_M 2 and 4 lost at every grid, best 480.0 us at
// 128 blocks against 412.8 at 1 on 192 (4096 x 7168, 2026-10-02T21-19-22Z). Its
// reduce-scatter once ran on the grid's first 32 blocks: its reads queue behind
// the links past a few dozen blocks while AttnRes is compute a row and wants
// every block. At 4096 tokens it took 146.6 us on 32 blocks against 218.9 on
// 192, AttnRes 1110.5 against 199.4 (stamps, 2026-10-01T23-45-31Z and
// 2026-10-01T23-50-54Z). The kernel no longer splits, and its knob
// (reduce_scatter_blocks, read by nothing) is gone (2026-10-21): a split that
// returns is a config field the kernel reads.
// 256 threads as the push's, its grid doubled (not swept yet). TILE_M 2 and 4
// are no longer built: they lost at every grid, and 4 spilled 25-30 VGPRs.
//
// OUR SEPARATE NORM KERNELS (rms_norm_rows and add_rms_norm_rows, the second kernel of a norm op
// Python composes after our all-reduce): a row a tile, Layout::tile, a block's threads along the
// row. Exact: no peer, nothing crosses a link.
//
// OUR SEPARATE ATTNRES KERNEL (add_attn_res_rms_norm_rows, the second kernel of AttnRes as Python
// composes it): a row a tile, the fused pull's exact tile (TILE_N a row's packs over the threads,
// Layout::tile), at Triton's AttnRes program (256 threads, a source a step: the reference's prefill
// config) and the neighbours a sweep picks among. Exact, as the norms'.
#include "hip_custom_communicator_builds.inc"
// EVERY KERNEL INSTANCE IS COMPILED IN ITS BUILD'S OWN FILE (builds.py): declared here, so the
// launches below instantiate none of them in this file.
#include "hip_custom_communicator_instances.inc"

// EACH KERNEL'S ALGORITHMS, in its op file's order: the bodies its template compiles, and what
// builds.py builds every shape of the tuned table at (it reads these lists, so they are one list).
constexpr int kAlgorithms_all_reduce[] = {Algorithm::one_shot_pull, Algorithm::two_shot_pull,
                                          Algorithm::one_shot_push, Algorithm::two_shot_push};
constexpr int kAlgorithms_all_reduce_rms_norm[] = {
    Algorithm::one_shot_pull, Algorithm::two_shot_pull, Algorithm::two_shot_push};
constexpr int kAlgorithms_all_reduce_add_rms_norm[] = {
    Algorithm::one_shot_pull, Algorithm::two_shot_pull, Algorithm::two_shot_push};
constexpr int kAlgorithms_all_reduce_add_attn_res_rms_norm[] = {
    Algorithm::one_shot_pull, Algorithm::two_shot_pull, Algorithm::two_shot_push};
constexpr int kAlgorithms_rms_norm_rows[] = {Algorithm::rows};
constexpr int kAlgorithms_add_rms_norm_rows[] = {Algorithm::rows};
constexpr int kAlgorithms_add_attn_res_rms_norm_rows[] = {Algorithm::rows};

// What each template is, in Template's order: its op, its algorithms, and its builds.
struct TemplateInfo {
  Template fn;
  const char* name;
  OpType op;
  std::span<const int> algorithms;
  std::span<const BuildConfig> builds;
};

// A TEMPLATE AND ITS NAME FROM ONE TOKEN, so the name cannot differ from the
// enum's, and its lists named by it.
#define HIP_CUSTOM_COMMUNICATOR_NAMED(t) Template::t, #t
#define HIP_CUSTOM_COMMUNICATOR_LISTS(t) kAlgorithms_##t, kBuilds_##t

constexpr TemplateInfo kTemplates[] = {
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce), OpType::all_reduce,
     HIP_CUSTOM_COMMUNICATOR_LISTS(all_reduce)},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_rms_norm), OpType::all_reduce_rms_norm,
     HIP_CUSTOM_COMMUNICATOR_LISTS(all_reduce_rms_norm)},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_add_rms_norm), OpType::all_reduce_add_rms_norm,
     HIP_CUSTOM_COMMUNICATOR_LISTS(all_reduce_add_rms_norm)},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_add_attn_res_rms_norm),
     OpType::all_reduce_add_attn_res_rms_norm,
     HIP_CUSTOM_COMMUNICATOR_LISTS(all_reduce_add_attn_res_rms_norm)},
    // THE NORMS ON THIS RANK'S ROWS (Algorithm::rows; STANDARDS 3.2, *We win at the op level*):
    // the second kernel of an op Python composes as two kernels of ours, our all-reduce and then
    // this, on its own grid at its own occupancy.
    {HIP_CUSTOM_COMMUNICATOR_NAMED(rms_norm_rows), OpType::rms_norm,
     HIP_CUSTOM_COMMUNICATOR_LISTS(rms_norm_rows)},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(add_rms_norm_rows), OpType::add_rms_norm,
     HIP_CUSTOM_COMMUNICATOR_LISTS(add_rms_norm_rows)},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(add_attn_res_rms_norm_rows), OpType::add_attn_res_rms_norm,
     HIP_CUSTOM_COMMUNICATOR_LISTS(add_attn_res_rms_norm_rows)},
};
#undef HIP_CUSTOM_COMMUNICATOR_LISTS
#undef HIP_CUSTOM_COMMUNICATOR_NAMED
constexpr int kNumTemplates = sizeof(kTemplates) / sizeof(TemplateInfo);

constexpr bool templates_in_order() {
  for (int i = 0; i < kNumTemplates; ++i)
    if (static_cast<int>(kTemplates[i].fn) != i) return false;
  return true;
}
static_assert(templates_in_order(),
              "kTemplates must list every Template in its order");

constexpr const TemplateInfo& info(Template k) {
  return kTemplates[static_cast<int>(k)];
}
constexpr const char* to_string(Template k) { return info(k).name; }

constexpr std::span<const BuildConfig> builds_of(Template k) {
  return info(k).builds;
}
constexpr bool every_template_built() {
  for (const TemplateInfo& t : kTemplates)
    if (t.builds.empty()) return false;
  return true;
}
static_assert(every_template_built(), "a template lists no builds");

// Whether template `k` has a body for `algorithm`.
constexpr bool has_algorithm(Template k, int algorithm) {
  for (const int a : info(k).algorithms)
    if (a == algorithm) return true;
  return false;
}
// EVERY BUILD IS AT ONE OF ITS KERNEL'S ALGORITHMS: a table config naming one its kernel has no
// body for is a compile error here, not a launch of nothing.
constexpr bool every_build_an_algorithm() {
  for (const TemplateInfo& t : kTemplates)
    for (const BuildConfig& b : t.builds)
      if (!has_algorithm(t.fn, algorithm_of(b))) return false;
  return true;
}
static_assert(every_build_an_algorithm(), "a build at an algorithm its kernel lacks");

// Whether template `k` builds `b`.
constexpr bool built(Template k, const BuildConfig& b) {
  for (const BuildConfig& listed : builds_of(k))
    if (listed == b) return true;
  return false;
}
// Whether any build of `k` at `algorithm` runs at `block` (and `waves_per_eu`): the check's names
// for a build that is not listed.
constexpr bool built_at(Template k, int algorithm, const Dim3& block) {
  for (const BuildConfig& listed : builds_of(k))
    if (algorithm_of(listed) == algorithm && block_of(listed) == block) return true;
  return false;
}
constexpr bool built_at(Template k, int algorithm, const Dim3& block, int waves_per_eu) {
  for (const BuildConfig& listed : builds_of(k))
    if (algorithm_of(listed) == algorithm && block_of(listed) == block &&
        waves_of(listed) == waves_per_eu)
      return true;
  return false;
}
// EVERY BUILD'S BLOCK IS x ALONE: every kernel indexes x alone (types.cuh's
// Dim3), so a build with threads in y or z would launch threads it never uses.
constexpr bool every_build_x_alone() {
  for (const TemplateInfo& t : kTemplates)
    for (const BuildConfig& b : t.builds)
      if (!x_alone(block_of(b))) return false;
  return true;
}
static_assert(every_build_x_alone(), "a build's block indexes y or z");
// THE BUILD TYPE A TEMPLATE'S BUILDS ARE (BuildConfig's index): its op's, every
// entry the same.
constexpr size_t family_of(Template k) { return builds_of(k)[0].index(); }
constexpr bool builds_one_family() {
  for (const TemplateInfo& t : kTemplates)
    for (const BuildConfig& b : t.builds)
      if (b.index() != family_of(t.fn)) return false;
  return true;
}
static_assert(builds_one_family(),
              "a template's builds are not all of one family");

constexpr OpType op_of(Template k) { return info(k).op; }
// EVERY OP HAS ONE TEMPLATE.
constexpr bool one_template_an_op() {
  for (const TemplateInfo& t : kTemplates)
    for (const TemplateInfo& u : kTemplates)
      if (t.fn != u.fn && t.op == u.op) return false;
  return true;
}
static_assert(one_template_an_op(), "an op has two templates");

// =================================================================================================
// OVER WHAT IS COMPILED, at compile time: the template arguments are constexpr,
// so each value (the world, a dtype, a built config) is turned into its
// template parameter here; one that was not built raises.
// =================================================================================================


// =================================================================================================
// EACH OP'S KERNELS.
// =================================================================================================

// EVERY BUILD THE TABLE NAMES IS COMPILED AND TAKES ANY CALL ITS KERNEL CAN RUN (Michael, October
// 25: the build makes what it is told): each kernel walks any tile at any block, its tails its own;
// what a call cannot do is refused by name at the call (`refused`), its capacity (the regions, the
// unfenced slot), the hardware's (`check_kernel`), or an algorithm's (INT4 on a one-shot).

// A ROW OF THE STAGING AND PARTIALS AS A KERNEL VIEWS IT: n, or a groups-a-thread INT4 norm's
// payload row, whole groups (its kernels' n_pad: every row's groups start on a group).
constexpr int64_t payload_row(Template k, int algorithm, int quantization, int64_t n) {
  return int4_groups_a_thread(k, algorithm) && quantization == Quantization::int4
             ? (n + kInt4Group - 1) / kInt4Group * kInt4Group
             : n;
}
// THE STAGING'S ROWS AS A KERNEL VIEWS IT: the input's m, or a groups-a-thread
// INT4 norm's inbox, WORLD slots of an owner's ceil(m / WORLD) rows.
constexpr int64_t staged_rows(Template k, int algorithm, int quantization, int64_t m,
                              int world) {
  return int4_groups_a_thread(k, algorithm) && quantization == Quantization::int4
             ? world * ((m + world - 1) / world)
             : m;
}

// =================================================================================================
// THE OPS, in OpType's order: each one's name (as Python calls it) and what a call of it must be
// to run, which `build_info` hands Python's `should`. What each runs at a size is Python's (its
// tuned table), handed whole.
// =================================================================================================

// A NORM'S WEIGHT, besides its input's dtype: fp32 (by_weight compiles exactly
// these).
constexpr DType kNormWeights[] = {DType::f32};

struct Op {
  OpType type;
  const char* name;
  // The dtypes its input may have.
  std::span<const DType> dtypes;
  // The dtypes its weight may have besides its input's; none for an op with no
  // weight.
  std::span<const DType> weights;
  // Whether its input is [rows, cols] as given (2-D), not any shape read as
  // rows of its last dimension.
  bool two_d;
};

// AN OP AND ITS NAME FROM ONE TOKEN, so the name cannot differ from the enum's.
#define HIP_CUSTOM_COMMUNICATOR_NAMED(o) OpType::o, #o

constexpr Op kOps[] = {
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce),
     kBuild.supports.dtypes,
     {},
     true},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_rms_norm),
     kBuild.supports.dtypes, kNormWeights, true},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_add_rms_norm),
     kBuild.supports.dtypes, kNormWeights, true},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(all_reduce_add_attn_res_rms_norm),
     kBuild.supports.dtypes,
     {},
     true},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(rms_norm), kBuild.supports.dtypes, kNormWeights, true},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(add_rms_norm), kBuild.supports.dtypes, kNormWeights, true},
    {HIP_CUSTOM_COMMUNICATOR_NAMED(add_attn_res_rms_norm), kBuild.supports.dtypes, {}, true},
};
#undef HIP_CUSTOM_COMMUNICATOR_NAMED
constexpr int kNumOps = sizeof(kOps) / sizeof(Op);

constexpr const Op& op(OpType t) { return kOps[static_cast<int>(t)]; }
constexpr const char* to_string(OpType t) { return op(t).name; }

constexpr bool ops_in_order() {
  for (int i = 0; i < kNumOps; ++i)
    if (static_cast<int>(kOps[i].type) != i) return false;
  return true;
}
static_assert(ops_in_order(), "kOps must list every OpType in its order");

// EVERY OP HAS KERNELS, each one of its own templates, its build listed and its
// launch one the forcing would accept (a grid of 1 to kMaxBlocks blocks in x
// alone, every kernel's one dimension).
constexpr bool launch_fits(const LaunchConfig& l) {
  return x_alone(l.grid) && l.grid.x >= 1 && l.grid.x <= kMaxBlocks;
}

// A COLUMN TWO-SHOT reduce-scatters by columns, so every block keeps every row;
// a row two-shot gives each rank its rows: the pulled AttnRes and every pushed fused op.
constexpr bool slices_columns(Template k, int algorithm) {
  return (k == Template::all_reduce_add_attn_res_rms_norm &&
          algorithm == Algorithm::two_shot_pull) ||
         (fuses(k) && algorithm == Algorithm::two_shot_push);
}

// =================================================================================================
// 1. THE FORCING: a whole config point, every field given (the C++ API takes no optional; what a
// caller may leave out is Python's to fill): the template, its algorithm, the grid,
// the block and its waves a EU, the tile, the quantization. A value out of range is its Error; a
// build the template does not list is the check's to name. Every call is forced: Python hands its
// table's pick the same way.
// =================================================================================================

// The kernel a call runs: its template and its complete config.
struct Chosen {
  Template fn;
  KernelConfig config;
};

// THE TEMPLATE BY ITS NAME (`build_info`'s, the config's `kernel`), or none.
inline std::optional<Template> template_named(std::string_view name) {
  for (const TemplateInfo& t : kTemplates)
    if (name == t.name) return t.fn;
  return std::nullopt;
}

// THE FORCED CONFIG of op `o`: template `fn` (its op's) at one of its algorithms, the grid and
// the block (x alone, every
// kernel's one dimension: y and z 1) of whole waves up to the device's workgroup (the hardware's
// limit; a wider block's register trade is the tune's to measure), the grid at most the blocks
// the signals have slots for (every block the device holds resident: whether this kernel's are is
// grid_not_resident), waves a EU the CU holds, the tile positive (tile_k 1 where the kernel has
// no K), the quantization exact or int4. Each limit is the hardware's (STANDARDS 3.2).
inline std::variant<Chosen, Error> forced(OpType o, Template fn, int algorithm, Dim3 grid,
                                          Dim3 block, int waves_per_eu, int tile_m, int tile_n,
                                          int tile_k, int quantization, int pack_bytes) {
  if (op_of(fn) != o || !has_algorithm(fn, algorithm)) return Error::no_such_template;
  if (quantization != Quantization::exact && quantization != Quantization::int4)
    return Error::no_such_quantization;
  if (!x_alone(block) || !x_alone(grid) || block.x < kTarget.wave_size ||
      block.x > kTarget.max_workgroup || block.x % kTarget.wave_size != 0 || grid.x < 1 ||
      grid.x > kMaxBlocks || waves_per_eu < 1 ||
      waves_per_eu > kTarget.max_waves_per_cu / kTarget.simds_per_cu)
    return Error::launch_out_of_range;
  if (tile_m < 1 || tile_n < 1 || tile_k < 1) return Error::field_not_positive;
  const bool has_k = o == OpType::all_reduce_add_attn_res_rms_norm ||
                     o == OpType::add_attn_res_rms_norm;
  if (!has_k && tile_k != 1) return Error::field_not_this_templates;
  // A PACK A LOAD THE HARDWARE HAS: 2, 4, 8 or 16 bytes (its widest); an element's at least is
  // the call's (refused): the dtype is not known here.
  if (pack_bytes < 2 || pack_bytes > kBuild.memory.pack_bytes ||
      (pack_bytes & (pack_bytes - 1)) != 0)
    return Error::pack_out_of_range;
  BuildConfig b = AllReduceBuildConfig{algorithm, block, waves_per_eu, tile_m, tile_n,
                                       quantization, pack_bytes};
  if (o == OpType::all_reduce_rms_norm || o == OpType::rms_norm)
    b = AllReduceRmsNormBuildConfig{algorithm, block, waves_per_eu, tile_m, tile_n,
                                    quantization, pack_bytes};
  if (o == OpType::all_reduce_add_rms_norm || o == OpType::add_rms_norm)
    b = AllReduceAddRmsNormBuildConfig{algorithm, block, waves_per_eu, tile_m, tile_n,
                                       quantization, pack_bytes};
  if (has_k)
    b = AllReduceAddAttnResRmsNormBuildConfig{algorithm, block, waves_per_eu, tile_m,
                                              tile_n, tile_k, quantization, pack_bytes};
  return Chosen{fn, {b, {grid}}};
}

// =================================================================================================
// 2. THE CHECK: the first Error kernel `fn` at `c` meets on the call here, in
// this order, or none. Every Error is a capability (a kernel that cannot take
// the input), never "the unfused ops would be faster". Residency, which needs
// the compiled kernel, is each select's last check.
// =================================================================================================

// A KERNEL'S PARTIALS, row-major (a build in passes needs none: it runs in passes
// of what the partials holds): the plain two-shot its slice of every row; a
// column two-shot the whole reduced tensor (each rank's columns at their
// place); a row two-shot its rank's rows, twice where it leaves two results
// (out and the residual). A one-shot reads the inputs and keeps nothing; `packs` a
// row's in packs of `pack` bytes (the build's). THE ADD-RMSNORM ROW TWO-SHOT's rows each
// carry their scale in a slot of kScaleSlotBytes before the row (common.cuh;
// kernels/all_reduce_add_rms_norm.cuh, its exact body: a row with its scale, one
// buffer), its row whole kScaleSlotBytes so the next row's scale is aligned
// (scale_slot_row).
constexpr int64_t partials_need(Template t, int algorithm, int64_t m, int64_t packs,
                                int world, int64_t pack) {
  if (!is_two_shot(algorithm)) return 0;
  if (t == Template::all_reduce)
    return m * ((packs + world - 1) / world) * pack;
  if (slices_columns(t, algorithm)) return m * packs * pack;
  const int64_t row_bytes =
      t == Template::all_reduce_add_rms_norm && !is_push(algorithm)
          ? scale_slot_row(packs * pack) + kScaleSlotBytes
          : packs * pack;
  return (m + world - 1) / world * row_bytes;
}

// A LAYOUT THE PACKED LOADS CAN READ: every buffer's innermost stride 1 and its
// rows starting on whole packs of `pack` bytes, the build's ({stride_m, stride_n} a
// buffer, in elements of `dtype`; a vector's stride_m 0).
inline std::optional<Error> layout_refused(
    DType dtype, int64_t pack, std::initializer_list<std::pair<int64_t, int64_t>> strides) {
  for (const auto& [m, n] : strides) {
    if (n != 1) return Error::inner_stride_not_one;
    if (m * elem_bytes(dtype) % pack != 0)
      return Error::row_stride_not_packs;
  }
  return std::nullopt;
}

// `row_elems`: the row the kernel reduces, in elements; `own`: the op's own
// Error on its own arguments, checked after the dtype. With no Handle the
// checks that need peers are left out.
inline std::optional<Error> refused(const Handle* h, Template fn,
                                    const KernelConfig& c, DType dtype,
                                    int64_t m, int64_t n,
                                    std::optional<Error> own_error,
                                    bool in_passes) {
  const int world = h ? h->world_size() : 1;
  if (h && !world_built(world)) return Error::world_not_built;
  if (!dtype_built(dtype)) return Error::dtype_not_built;
  if (own_error) return own_error;
  // THE BUILD'S PACK: at least an element, and whole packs a row (an aligned load of the
  // pack's width: a row that is not whole packs starts the next off them; a narrower
  // build's takes it).
  const int64_t pack = pack_bytes_of(c.build);
  if (pack < elem_bytes(dtype)) return Error::pack_out_of_range;
  if (n * elem_bytes(dtype) % pack != 0) return Error::row_not_packs;
  // INT4 ON A ONE-SHOT, which reads its peers' inputs as they are: named before
  // the build lookup, which lists no such build; INT4 AT A NARROWER PACK than the
  // widest: the format's word is a pack of 8 values (common.cuh's QUANTIZATION).
  const bool int4 = quantization_of(c.build) == Quantization::int4;
  const int algorithm = algorithm_of(c.build);
  if (int4 && (is_one_shot(algorithm) || algorithm == Algorithm::rows ||
               (fn == Template::all_reduce && !is_push(algorithm)) ||
               pack != kBuild.memory.pack_bytes))
    return Error::quantization_not_supported;
  // A BUILD THE TEMPLATE DOES NOT LIST, named by what differs: its block, its
  // waves, its tile.
  const Dim3 block = block_of(c.build);
  if (c.build.index() != family_of(fn)) return Error::tile_not_built;
  if (!built_at(fn, algorithm, block)) return Error::threads_not_built;
  if (!built_at(fn, algorithm, block, waves_of(c.build)))
    return Error::waves_not_built;
  if (!built(fn, c.build)) return Error::tile_not_built;
  // A PUSHED ONE-SHOT IN THE UNFENCED REGION, its capacity (not a build's rule: any tile and
  // grid is built): a row of a block's tile in a slot's lines, a line a pack of at most
  // kLinePayloadBytes (a narrower pack its line alone: kernels/all_reduce.cuh), a slot
  // kUnfencedSlotBytes / kLinePayloadBytes lines; a block's TILE_M rows of a bank's
  // kUnfencedBlocks.
  const int64_t line_pack = pack < kLinePayloadBytes ? pack : kLinePayloadBytes;
  if (algorithm == Algorithm::one_shot_push &&
      ((tile_n_of(c.build) * elem_bytes(dtype) + line_pack - 1) / line_pack >
           kUnfencedSlotBytes / kLinePayloadBytes ||
       int64_t{c.launch.grid.x} * tile_m_of(c.build) > kUnfencedBlocks))
    return Error::unfenced_too_small;
  if (!h) return std::nullopt;
  const int64_t packs = packs_of(n, dtype, pack);
  // An INT4 payload lies in the region that holds it exact (common.cuh's
  // INT4 payload layout): the input's in the inputs, the sums' in the partials, at
  // most one row of the partials's elements (its every view's bound); the flat walk's
  // on its shares' own views (int4_flat_share).
  const bool flat = fn == Template::all_reduce && algorithm == Algorithm::two_shot_push;
  const int64_t share = int4_flat_share(m, n, world);
  const int64_t row = payload_row(fn, algorithm, Quantization::int4, n);
  if (int4 && (flat ? int4_payload_bytes(1, world * share)
                    : int4_payload_bytes(staged_rows(fn, algorithm,
                                                     Quantization::int4, m, world),
                                         row)) > h->staging_bytes())
    return Error::staging_too_small;
  const int64_t partials = partials_need(fn, algorithm, m, packs, world, pack);
  if (int4 && (flat ? int4_payload_bytes(1, share)
               : int4_groups_a_thread(fn, algorithm)
                   ? int4_payload_bytes((m + world - 1) / world, row)
                   : int4_payload_bytes(1, partials / elem_bytes(dtype))) >
                  h->partials_bytes())
    return Error::partials_too_small;
  if (!in_passes && partials > h->partials_bytes())
    return Error::partials_too_small;
  // A BUILD THAT DOES NOT RUN IN PASSES has its input in the inputs whole:
  // an eager exact call's host copy (handle.cuh's INPUT TABLES; a captured
  // one's is read where it is, refused here alike, so the eager and captured
  // calls of one shape agree), or an INT4 call's quantized payload (the INT4
  // all-reduce: pushed).
  const int64_t bytes = m * n * elem_bytes(dtype);
  if (!in_passes && bytes > h->staging_bytes())
    return Error::staging_too_small;
  return std::nullopt;
}

// THE WEIGHT a norm takes: its input's dtype, or one of its op's `weights`.
constexpr std::optional<Error> weight_refused(OpType o, DType dtype,
                                              DType weight_dtype) {
  if (weight_dtype == dtype) return std::nullopt;
  for (const DType w : op(o).weights)
    if (weight_dtype == w) return std::nullopt;
  return Error::weight_not_built;
}

// =================================================================================================
// EACH OP'S SELECT: the op's launch, everything decided, from the call's
// primitives and its forcing, or the first Error the call meets. Each in four
// steps: 1. the kernel (the forced template and config, else the op's tuned
// kernel, its grid cut to the work), 2. the checks, 3. its compiled
// instance, 4. the launch.
// =================================================================================================

// A KERNEL'S INSTANCE AND ITS LAUNCH, everything decided: the template (`kernel`), the build's
// fields, and what the call adds (its dtype, world, layout and regions' views). The template
// parameter of every instance's dispatch (`kernel_pointer<C>`), so each field is a constant there.
struct Config {
  Template kernel{};
  int algorithm{};  // an Algorithm
  int world{};
  int tile_m{};
  int tile_n{};
  int tile_k{1};  // 1 where the kernel has no K, its identity
  Dim3 block{};
  Dim3 grid{};
  int waves_per_eu{};
  int quantization{};
  int shared_memory_bytes{};
  DType dtype{};
  DType weight_dtype{};
  bool contiguous{};
  bool in_place{};
  // CHUNKED: a norm's or AttnRes's row wider than the build's tile, walked in tile-wide chunks in
  // passes (the kernels' CHUNKED instance); a row the tile holds runs the one-pass instance.
  bool chunked{};
  // TAIL: an INT4 row that is not whole groups for every rank (its last group short, or its groups
  // not split evenly over the ranks), the INT4 two-shots' TAIL instance (int4_tail); whole groups
  // run the instance without the tail's masks.
  bool tail{};
  int64_t partials_stride_m{};
  int64_t partials_stride_n{};
  int64_t partials_size_m{};
  int64_t partials_size_n{};
  int64_t staging_stride_m{};
  int64_t staging_stride_n{};
  int64_t staging_size_m{};
  int64_t staging_size_n{};
  // THE PUSHED ONE-SHOT'S INBOX IN THE UNFENCED REGION, [slot, m, n] in payload elements
  // (kernels/all_reduce.cuh; THE UNFENCED REGION): zeros for every other kernel.
  int64_t unfenced_stride_slot{};
  int64_t unfenced_stride_m{};
  int64_t unfenced_stride_n{};
  int64_t unfenced_size_slot{};
  int64_t unfenced_size_m{};
  int64_t unfenced_size_n{};
  int64_t scales_stride_m{};
  int64_t scales_size_m{};
  // THE MACHINE'S WAVE, the host's fact and no call's (HARDWARE), handed to every kernel as its
  // WAVE_SIZE; the BUILD'S PACK, its bytes, as its PACK_BYTES (kernel_pointer and each launch),
  // each op's config setting it from its build. NO DEFAULT PACK: left at 16, an op that did not set
  // it ran the 16-byte build of its other fields, wrong on a row not whole 16-byte packs
  // (2026-10-25T18-32-31Z); left at 0, it finds no build.
  int wave_size{kTarget.wave_size};
  int pack_bytes{};
};

template <typename WRAPPED>
struct type {
  using t = WRAPPED;
};

template <int INT_VALUE>
using constant = std::integral_constant<int, INT_VALUE>;

[[noreturn]] inline void not_built(const std::string& what) {
  throw std::runtime_error("hip_custom_communicator: " + what + " not built");
}

// THE C++ TYPE OF A BUILT DTYPE.
template <DType DTYPE_ID>
struct of_dtype;
template <>
struct of_dtype<DType::f16> {
  using t = c10::Half;
};
template <>
struct of_dtype<DType::bf16> {
  using t = c10::BFloat16;
};

// OVER THE BUILT LISTS (kBuild.supports), so what is compiled is what they say.
template <typename VISITOR, size_t... INDICES>
void by_world_in(int world, VISITOR& f, std::index_sequence<INDICES...>) {
  constexpr auto& built = kBuild.supports.worlds;
  if (!((world == built[INDICES] && (f(constant<built[INDICES]>{}), true)) ||
        ...))
    not_built("world size " + std::to_string(world));
}

template <typename VISITOR>
void by_world(int world, VISITOR&& f) {
  by_world_in(world, f,
              std::make_index_sequence<kBuild.supports.worlds.size()>{});
}

template <typename VISITOR, size_t... INDICES>
void by_dtype_in(DType d, VISITOR& f, std::index_sequence<INDICES...>) {
  constexpr auto& built = kBuild.supports.dtypes;
  if (!((d == built[INDICES] &&
         (f(type<typename of_dtype<built[INDICES]>::t>{}), true)) ||
        ...))
    not_built("dtype");
}

template <typename VISITOR>
void by_dtype(DType d, VISITOR&& f) {
  by_dtype_in(d, f, std::make_index_sequence<kBuild.supports.dtypes.size()>{});
}

// A norm's weight: T itself, or fp32.
template <typename DTYPE, typename VISITOR>
void by_weight(DType weight, DType dtype, VISITOR&& f) {
  if (weight == dtype) return f(type<DTYPE>{});
  if (weight == DType::f32) return f(type<float>{});
  not_built("weight dtype");
}

// A TEMPLATE'S BUILD, compiled in: the one of template K's builds (the catalog
// above) `b` is, and only those. f is handed it as build_constant<B>, B its
// op's BuildConfig, every field a constant expression.
template <Template TEMPLATE>
using family_t = std::variant_alternative_t<family_of(TEMPLATE), BuildConfig>;
template <Template TEMPLATE, size_t BUILD_INDEX>
constexpr family_t<TEMPLATE> listed_build =
    std::get<family_t<TEMPLATE>>(builds_of(TEMPLATE)[BUILD_INDEX]);
template <auto LISTED_BUILD>
struct build_constant {
  static constexpr auto value = LISTED_BUILD;
};

template <Template TEMPLATE, typename VISITOR, size_t... INDICES>
void by_build_in(const BuildConfig& b, VISITOR& f,
                 std::index_sequence<INDICES...>) {
  if (!((b == builds_of(TEMPLATE)[INDICES] &&
         (f(build_constant<listed_build<TEMPLATE, INDICES>>{}), true)) ||
        ...))
    not_built(std::string("template ") + to_string(TEMPLATE) +
              " at that build");
}

template <Template TEMPLATE, typename VISITOR>
void by_build(const BuildConfig& b, VISITOR&& f) {
  by_build_in<TEMPLATE>(b, f,
                        std::make_index_sequence<builds_of(TEMPLATE).size()>{});
}


template <>
struct of_dtype<DType::f32> { using t = float; };

template <typename T>
constexpr DType dtype_of = std::is_same_v<T, c10::Half> ? DType::f16 : std::is_same_v<T,
    c10::BFloat16> ? DType::bf16 : DType::f32;

template <Template FN, typename F>
void with_config_build(const Config& config, F& f) {
  by_world(config.world, [&](auto world) {
    by_dtype(config.dtype, [&](auto dtype) {
      using T = typename decltype(dtype)::t;
      // Only the norm families instantiate a separate weight dtype.
      const auto weight = [&](auto w) {
        using W = typename decltype(w)::t;
        by_build<FN>([&]() -> BuildConfig {
          using B = family_t<FN>;
          B b{};
          b.algorithm = config.algorithm;
          b.block = config.block;
          b.waves_per_eu = config.waves_per_eu;
          b.tile_m = config.tile_m;
          b.tile_n = config.tile_n;
          b.quantization = config.quantization;
          b.pack_bytes = config.pack_bytes;
          if constexpr (requires { b.tile_k; }) b.tile_k = config.tile_k;
          return b;
        }(), [&](auto b) {
          constexpr auto B = decltype(b)::value;
          constexpr int WORLD = decltype(world)::value;
          {
            const auto layout = [&]<bool CONTIGUOUS, bool CHUNKED, bool TAIL>() {
              constexpr Config C = [=] {
                Config c{};
                c.kernel = FN;
                c.algorithm = B.algorithm;
                c.world = WORLD;
                c.dtype = dtype_of<T>;
                c.weight_dtype = dtype_of<W>;
                c.tile_m = B.tile_m;
                c.tile_n = B.tile_n;
                if constexpr (requires { B.tile_k; }) c.tile_k = B.tile_k;
                c.block = B.block;
                c.waves_per_eu = B.waves_per_eu;
                c.quantization = compiled_quantization(B);
                c.pack_bytes = B.pack_bytes;
                c.contiguous = CONTIGUOUS;
                c.chunked = CHUNKED;
                c.tail = TAIL;
                return c;
              }();
              f.template operator()<C>();
            };
            // THE INSTANCES a build has: contiguous or strided (the plain two-shots), chunked or not
            // (the norms and AttnRes), with a tail or not (the INT4 two-shots).
            const auto tailed = [&]<bool CONTIGUOUS, bool CHUNKED>() {
              if constexpr (compiled_quantization(B) == Quantization::int4 &&
                            is_two_shot(B.algorithm)) {
                if (config.tail) layout.template operator()<CONTIGUOUS, CHUNKED, true>();
                else layout.template operator()<CONTIGUOUS, CHUNKED, false>();
              } else layout.template operator()<CONTIGUOUS, CHUNKED, false>();
            };
            if constexpr (FN == Template::all_reduce && is_two_shot(B.algorithm)) {
              if (config.contiguous) tailed.template operator()<true, false>();
              else tailed.template operator()<false, false>();
            } else if constexpr (FN != Template::all_reduce) {
              if (config.chunked) tailed.template operator()<false, true>();
              else tailed.template operator()<false, false>();
            } else tailed.template operator()<false, false>();
          }
        });
      };
      if constexpr (op_of(FN) == OpType::all_reduce_rms_norm ||
                    op_of(FN) == OpType::all_reduce_add_rms_norm ||
                    op_of(FN) == OpType::rms_norm || op_of(FN) == OpType::add_rms_norm)
        by_weight<T>(config.weight_dtype, config.dtype, weight);
      else weight(type<T>{});
    });
  });
}

// OP `OP`'S INSTANCE AT `config`, compiled in: its one template's (`f` is handed the Config of
// the instance as its template argument, every field a constant).
template <OpType OP, typename F>
void with_config(const Config& config, F&& f) {
  constexpr Template FN = [] {
    for (const TemplateInfo& t : kTemplates)
      if (t.op == OP) return t.fn;
    return Template::all_reduce;  // never: every op has its template (one_template_an_op)
  }();
  if (config.kernel != FN) not_built("kernel config");
  with_config_build<FN>(config, f);
}

template <Config C>
const void* kernel_pointer() {
  using T = typename of_dtype<C.dtype>::t;
  using W = typename of_dtype<C.weight_dtype>::t;
  if constexpr (C.kernel == Template::all_reduce)
    return reinterpret_cast<const void*>(
        all_reduce<T, C.world, C.tile_m, C.tile_n, C.block.x, C.wave_size, C.pack_bytes,
            C.waves_per_eu, C.quantization, C.algorithm, C.contiguous, C.tail>);
  if constexpr (C.kernel == Template::all_reduce_rms_norm)
    return reinterpret_cast<const void*>(
        all_reduce_rms_norm<T, W, C.world, C.tile_n, C.block.x, C.wave_size, C.pack_bytes,
            C.waves_per_eu, C.quantization, C.algorithm, C.chunked, C.tail>);
  if constexpr (C.kernel == Template::rms_norm_rows)
    return reinterpret_cast<const void*>(
        rms_norm_rows<T, W, C.tile_m, C.tile_n, C.block.x, C.wave_size, C.pack_bytes,
            C.waves_per_eu, C.chunked>);
  if constexpr (C.kernel == Template::all_reduce_add_rms_norm)
    return reinterpret_cast<const void*>(
        all_reduce_add_rms_norm<T, W, C.world, C.tile_n, C.block.x, C.wave_size, C.pack_bytes,
            C.waves_per_eu, C.quantization, C.algorithm, C.chunked, C.tail>);
  if constexpr (C.kernel == Template::add_rms_norm_rows)
    return reinterpret_cast<const void*>(
        add_rms_norm_rows<T, W, C.tile_m, C.tile_n, C.block.x, C.wave_size, C.pack_bytes,
            C.waves_per_eu, C.chunked>);
  if constexpr (C.kernel == Template::all_reduce_add_attn_res_rms_norm)
    return reinterpret_cast<const void*>(
        all_reduce_add_attn_res_rms_norm<T, C.world, C.tile_m, C.tile_n, C.tile_k, C.block.x,
            C.wave_size, C.pack_bytes, C.waves_per_eu, C.quantization, C.algorithm,
            C.chunked, C.tail>);
  if constexpr (C.kernel == Template::add_attn_res_rms_norm_rows)
    return reinterpret_cast<const void*>(
        add_attn_res_rms_norm_rows<T, C.tile_m, C.tile_n, C.tile_k, C.block.x, C.wave_size,
            C.pack_bytes, C.waves_per_eu, C.chunked>);
}

inline std::variant<Config, Error> all_reduce_config(
    const Handle& h, const void* input, int64_t input_stride_m,
    int64_t input_stride_n, int64_t input_size_m, int64_t input_size_n,
    void* output, int64_t output_stride_m, int64_t output_stride_n,
    int64_t output_size_m, int64_t output_size_n, DType dtype, const Chosen& k, hipStream_t stream) {
  const int64_t m = input_size_m,
                n = input_size_n;  // the problem's shape: the input's
  const int world = h.world_size();
  const Template fn = k.fn;
  const int algorithm = algorithm_of(k.config.build);
  const KernelConfig& c = k.config;
  const AllReduceBuildConfig& a = std::get<AllReduceBuildConfig>(c.build);
  // IN PLACE: the exact pull, every rank's input read where its table says.
  const bool in_place = !is_push(algorithm) &&
                        a.quantization == Quantization::exact;
  // THE UNFENCED REGION: the pushed one-shot pushes lines into its own inbox there, a tile a
  // block a pass, nothing staged. ITS GRID IS THE ONE GIVEN: a block its own slots, so at most the
  // unfenced region's blocks (more is refused, never cut); a block past the tiles has no pass, and leaves its
  // epoch as it was.
  const bool unfenced = algorithm == Algorithm::one_shot_push;
  if (unfenced && volume(c.launch.grid) > kUnfencedBlocks) return Error::launch_out_of_range;
  // IN PASSES (none): a band of rows a pass (the in-place one-shot's one band,
  // the whole input). INT4 pushes its input's shares quantized, whole,
  // instead.
  const bool in_passes = a.quantization == Quantization::exact;
  // THE BAND: the rows a pass, what the inbox (the inputs region) holds in
  // WORLD slots. The one-shot's slot is the band's rows, [WORLD * band, n];
  // the two-shot's is one rank's share of the band (the flat walk's, whole
  // packs, rounded up), [WORLD, slot], whose band the partials also hold (a
  // row of its slice a row). Rounding up costs a two-shot at most one pack a
  // slot, so the band one row less than the inputs' rows always fits.
  const int64_t row_bytes = n * elem_bytes(dtype);
  const int64_t pack = a.pack_bytes;  // the build's
  const int64_t packs = n * elem_bytes(dtype) / pack;
  const int64_t slice_bytes = (packs + world - 1) / world * pack;
  const auto slot_packs = [&](int64_t rows) {
    return (rows * packs + world - 1) / world;
  };
  int64_t band = 0;
  if ((in_place && !is_two_shot(algorithm)) || unfenced) {
    band = m;  // nothing staged: one pass (the lines' passes are the kernel's)
  } else if (in_place) {
    // THE IN-PLACE TWO-SHOT'S BAND: what the partials hold, a row of its
    // slice a row; the input is read where it is.
    band = slice_bytes > 0 ? h.partials_bytes() / slice_bytes : 0;
  } else if (row_bytes > 0 && !is_two_shot(algorithm)) {
    band = h.staging_bytes() / (world * row_bytes);
  } else if (row_bytes > 0) {
    band = h.staging_bytes() / row_bytes;
    if (band > 0 && world * slot_packs(band) * pack > h.staging_bytes()) --band;
    if (slice_bytes > 0 && h.partials_bytes() / slice_bytes < band)
      band = h.partials_bytes() / slice_bytes;
  }
  const int64_t rows = band < m ? band : m;  // a pass's rows
  if (const std::optional<Error> e =
          refused(&h, fn, c, dtype, m, n,
                  std::nullopt, in_passes))
    return *e;
  if (in_place && is_two_shot(algorithm) && band < 1)
    return Error::partials_too_small;
  if (in_passes && band < 1) return Error::staging_too_small;
  // AN EAGER IN-PLACE CALL'S INPUT IS COPIED WHOLE into the inputs region
  // (its table, handle.cuh's INPUT TABLES); a captured one is read where it
  // is, registered at capture exit.
  if (in_place) {
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    HIP_CHECK(hipStreamIsCapturing(stream, &capture));
    if (capture != hipStreamCaptureStatusActive &&
        m * n * elem_bytes(dtype) > h.staging_bytes())
      return Error::staging_too_small;
  }
  Config config{};
  config.shared_memory_bytes = 0;
  config.weight_dtype = dtype;
  config.kernel = fn;
  config.algorithm = algorithm;
  config.world = world;
  config.tile_m = a.tile_m;
  config.tile_n = a.tile_n;
  config.block = a.block;
  config.grid = c.launch.grid;
  config.waves_per_eu = a.waves_per_eu;
  config.quantization = a.quantization;
  config.pack_bytes = a.pack_bytes;  // the build's: the instance dispatch finds
  // THE INT4 FLAT WALK'S PAYLOADS on views of their own (int4_flat_share).
  const bool int4_flat = !in_passes && is_two_shot(algorithm);
  const int64_t share = int4_flat_share(m, n, world);
  config.partials_stride_m = int4_flat ? share : slice_bytes / elem_bytes(dtype);
  config.partials_stride_n = 1;
  config.partials_size_m = int4_flat ? 1 : in_passes ? rows : m;
  config.partials_size_n = int4_flat ? share : slice_bytes / elem_bytes(dtype);
  config.staging_stride_m = in_place || unfenced                ? 0
                         : int4_flat                       ? world * share
                         : !in_passes || !is_two_shot(algorithm) ? n
                             : slot_packs(rows) * pack / elem_bytes(dtype);
  config.staging_stride_n = in_place || unfenced ? 0 : 1;
  config.staging_size_m = in_place || unfenced ? 0
                       : int4_flat ? 1
                       : !in_passes ? m
                       : is_two_shot(algorithm) ? world
                                         : world * rows;
  config.staging_size_n = in_place || unfenced                ? 0
                       : int4_flat                       ? world * share
                       : !in_passes || !is_two_shot(algorithm) ? n
                           : slot_packs(rows) * pack / elem_bytes(dtype);
  // THE UNFENCED REGION: [writer, row, n] payload elements, 2 * kUnfencedBlocks rows (bank, row) of
  // kUnfencedSlotBytes of payload each whatever the build (so a row is the same bytes at every
  // build, its epoch its own), a writer's rows contiguous.
  config.unfenced_stride_slot = unfenced ? 2 * kUnfencedBlocks * (kUnfencedSlotBytes / elem_bytes(dtype)) : 0;
  config.unfenced_stride_m = unfenced ? kUnfencedSlotBytes / elem_bytes(dtype) : 0;
  config.unfenced_stride_n = unfenced ? 1 : 0;
  config.unfenced_size_slot = unfenced ? world : 0;
  config.unfenced_size_m = unfenced ? 2 * kUnfencedBlocks : 0;
  config.unfenced_size_n = unfenced ? kUnfencedSlotBytes / elem_bytes(dtype) : 0;
  config.dtype = dtype;
  config.contiguous = input_stride_m == n * input_stride_n &&
      output_stride_m == n * output_stride_n;
  config.in_place = !is_push(config.algorithm) &&
      config.quantization == Quantization::exact;
  // AN INT4 ROW WITH A TAIL runs the build's TAIL instance (int4_tail).
  config.tail = config.quantization == Quantization::int4 && int4_tail(fn, algorithm, m, n, world);
  const void* kernel = nullptr;
  with_config<OpType::all_reduce>(config, [&]<Config C>() { kernel = kernel_pointer<C>(); });
  if (!kernel) return Error::tile_not_built;
  if (const std::optional<Error> e = h.check_kernel(kernel, config.grid, config.block,
                                                     config.shared_memory_bytes, true))
    return *e;
  return config;
}

// all_reduce_rms_norm: out = rms_norm(all_reduce(inp), weight), vLLM's
// roundings exactly. inp [rows, hidden]; the weight `weight_dtype`, dtype or
// f32.
inline std::variant<Config, Error> all_reduce_rms_norm_config(
    const Handle& h, const void* input, int64_t input_stride_m,
    int64_t input_stride_n, int64_t input_size_m, int64_t input_size_n,
    const void* weight, int64_t weight_stride_n, int64_t weight_size_n,
    void* output, int64_t output_stride_m, int64_t output_stride_n,
    int64_t output_size_m, int64_t output_size_n, DType dtype,
    DType weight_dtype, float eps, const Chosen& k, hipStream_t stream) {
  const int64_t m = input_size_m,
                n = input_size_n;  // the problem's shape: the input's
  using K = Template;
  const OpType op = OpType::all_reduce_rms_norm;
  const int world = h.world_size();
  const Template fn = k.fn;
  const int algorithm = algorithm_of(k.config.build);
  const KernelConfig& c = k.config;
  if (const std::optional<Error> e = refused(
          &h, fn, c, dtype, m, n,
          weight_refused(op, dtype, weight_dtype), false))
    return *e;
  const AllReduceRmsNormBuildConfig& r =
      std::get<AllReduceRmsNormBuildConfig>(c.build);
  // THE PARTIALS'S ROWS, what partials_need sized it for (a one-shot keeps none).
  const int64_t partials_rows =
      n > 0 ? partials_need(fn, algorithm, m,
                           n * elem_bytes(dtype) / pack_bytes_of(c.build), world,
                           pack_bytes_of(c.build)) /
                  (n * elem_bytes(dtype))
            : 0;
  Config config{};
  config.shared_memory_bytes = 0;
  config.weight_dtype = dtype;
  config.kernel = fn;
  config.algorithm = algorithm;
  config.world = world;
  config.tile_m = r.tile_m;
  config.tile_n = r.tile_n;
  config.block = r.block;
  config.grid = c.launch.grid;
  config.waves_per_eu = r.waves_per_eu;
  config.quantization = r.quantization;
  config.pack_bytes = r.pack_bytes;  // the build's: the instance dispatch finds
  const int64_t row_n = payload_row(fn, algorithm, r.quantization, n);
  config.partials_stride_m = row_n;
  config.partials_stride_n = 1;
  config.partials_size_m = partials_rows;
  config.partials_size_n = row_n;
  config.staging_stride_m = row_n;
  config.staging_stride_n = 1;
  config.staging_size_m = staged_rows(fn, algorithm, r.quantization, m, world);
  config.staging_size_n = row_n;
  config.dtype = dtype;
  config.weight_dtype = weight_dtype;
  config.contiguous = input_stride_m == n * input_stride_n &&
      output_stride_m == n * output_stride_n;
  config.in_place = !is_push(config.algorithm) &&
      config.quantization == Quantization::exact;
  // A ROW WIDER THAN THE TILE runs the build's CHUNKED instance.
  config.chunked = n > config.tile_n;
  // AN INT4 ROW WITH A TAIL runs the build's TAIL instance (int4_tail).
  config.tail = config.quantization == Quantization::int4 && int4_tail(fn, algorithm, m, n, world);
  const void* kernel = nullptr;
  with_config<op>(config, [&]<Config C>() { kernel = kernel_pointer<C>(); });
  if (!kernel) return Error::tile_not_built;
  if (const std::optional<Error> e = h.check_kernel(kernel, config.grid, config.block,
                                                     config.shared_memory_bytes, true))
    return *e;
  return config;
}

// all_reduce_add_rms_norm: out, residual_out =
// fused_add_rms_norm(all_reduce(inp), residual, weight), vLLM's roundings
// exactly.
inline std::variant<Config, Error>
all_reduce_add_rms_norm_config(
    const Handle& h, const void* input, int64_t input_stride_m,
    int64_t input_stride_n, int64_t input_size_m, int64_t input_size_n,
    const void* residual, int64_t residual_stride_m, int64_t residual_stride_n,
    int64_t residual_size_m, int64_t residual_size_n, const void* weight,
    int64_t weight_stride_n, int64_t weight_size_n, void* output,
    int64_t output_stride_m, int64_t output_stride_n, int64_t output_size_m,
    int64_t output_size_n, void* residual_output,
    int64_t residual_output_stride_m, int64_t residual_output_stride_n,
    int64_t residual_output_size_m, int64_t residual_output_size_n, DType dtype,
    DType weight_dtype, float eps, const Chosen& k, hipStream_t stream) {
  const int64_t m = input_size_m,
                n = input_size_n;  // the problem's shape: the input's
  using K = Template;
  const OpType op = OpType::all_reduce_add_rms_norm;
  const int world = h.world_size();
  const Template fn = k.fn;
  const int algorithm = algorithm_of(k.config.build);
  const KernelConfig& c = k.config;
  if (const std::optional<Error> e = refused(
          &h, fn, c, dtype, m, n,
          weight_refused(op, dtype, weight_dtype), false))
    return *e;
  const AllReduceAddRmsNormBuildConfig& r =
      std::get<AllReduceAddRmsNormBuildConfig>(c.build);
  // THE PARTIALS'S ROWS, what partials_need sized it for (a one-shot keeps none),
  // at its row stride: the exact pull two-shot's row is its scale's pack then the
  // row (partials_need); the INT4 build's rows are the row alone, its scales in
  // the scales region.
  const bool scale_slot = algorithm == Algorithm::two_shot_pull &&
                          r.quantization == Quantization::exact;
  // The scale slot's row whole slots (scale_slot_row), so every row's scale is aligned at any pack.
  const int64_t slot = scale_slot ? kScaleSlotBytes / elem_bytes(dtype) : 0;
  const int64_t row = payload_row(fn, algorithm, r.quantization, n);
  const int64_t partials_stride_m =
      scale_slot ? scale_slot_row(row * elem_bytes(dtype)) / elem_bytes(dtype) + slot : row;
  const int64_t partials_rows =
      n > 0 ? partials_need(fn, algorithm, m,
                           n * elem_bytes(dtype) / r.pack_bytes, world, r.pack_bytes) /
                  (partials_stride_m * elem_bytes(dtype))
            : 0;
  Config config{};
  config.shared_memory_bytes = 0;
  config.weight_dtype = dtype;
  config.kernel = fn;
  config.algorithm = algorithm;
  config.world = world;
  config.tile_m = r.tile_m;
  config.tile_n = r.tile_n;
  config.block = r.block;
  config.grid = c.launch.grid;
  config.waves_per_eu = r.waves_per_eu;
  config.quantization = r.quantization;
  config.pack_bytes = r.pack_bytes;  // the build's: the instance dispatch finds
  config.partials_stride_m = partials_stride_m;
  config.partials_stride_n = 1;
  config.partials_size_m = partials_rows;
  config.partials_size_n = partials_stride_m;
  config.scales_stride_m = 1;
  config.scales_size_m =
      algorithm == Algorithm::two_shot_pull && !scale_slot
          ? (m + world - 1) / world
          : 0;
  config.staging_stride_m = payload_row(fn, algorithm, r.quantization, n);
  config.staging_stride_n = 1;
  config.staging_size_m = staged_rows(fn, algorithm, r.quantization, m, world);
  config.staging_size_n = payload_row(fn, algorithm, r.quantization, n);
  config.dtype = dtype;
  config.weight_dtype = weight_dtype;
  config.contiguous = input_stride_m == n * input_stride_n &&
      output_stride_m == n * output_stride_n;
  config.in_place = !is_push(config.algorithm) &&
      config.quantization == Quantization::exact;
  // A ROW WIDER THAN THE TILE runs the build's CHUNKED instance.
  config.chunked = n > config.tile_n;
  // AN INT4 ROW WITH A TAIL runs the build's TAIL instance (int4_tail).
  config.tail = config.quantization == Quantization::int4 && int4_tail(fn, algorithm, m, n, world);
  const void* kernel = nullptr;
  with_config<op>(config, [&]<Config C>() { kernel = kernel_pointer<C>(); });
  if (!kernel) return Error::tile_not_built;
  if (config.scales_size_m > h.scales_size_m()) return Error::partials_too_small;
  if (const std::optional<Error> e = h.check_kernel(kernel, config.grid, config.block,
                                                     config.shared_memory_bytes, true))
    return *e;
  return config;
}

// all_reduce_add_attn_res_rms_norm: Kimi-K3's AttnRes and its RMSNorm on each
// row of the all-reduced sum of `input`. With a `prefix` the sum is added to
// it, into `prefix_output`; without (null), the sum IS the new prefix.
// `block_output`, when not null, gets the new prefix too. The pull two-shot's
// config family has TILE_M and its reduce-scatter blocks; the one-shot's and
// push's (a row a tile) have neither.
inline std::variant<Config, Error>
all_reduce_add_attn_res_rms_norm_config(
    const Handle& h, const void* input, int64_t input_stride_m,
    int64_t input_stride_n, int64_t input_size_m, int64_t input_size_n,
    const void* prefix, int64_t prefix_stride_m, int64_t prefix_stride_n,
    int64_t prefix_size_m, int64_t prefix_size_n, const void* blocks,
    int64_t blocks_stride_m, int64_t blocks_stride_r, int64_t blocks_stride_n,
    int64_t blocks_size_m, int64_t blocks_size_r, int64_t blocks_size_n,
    const void* norm_weight, int64_t norm_weight_stride_n,
    int64_t norm_weight_size_n, const void* qk_weight,
    int64_t qk_weight_stride_n, int64_t qk_weight_size_n,
    const void* output_norm_weight, int64_t output_norm_weight_stride_n,
    int64_t output_norm_weight_size_n, void* prefix_output,
    int64_t prefix_output_stride_m, int64_t prefix_output_stride_n,
    int64_t prefix_output_size_m, int64_t prefix_output_size_n, void* output,
    int64_t output_stride_m, int64_t output_stride_n, int64_t output_size_m,
    int64_t output_size_n, void* block_output, int64_t block_output_stride_m,
    int64_t block_output_stride_n, int64_t block_output_size_m,
    int64_t block_output_size_n, DType dtype, int num_blocks, float eps,
    float output_eps, const Chosen& k, hipStream_t stream) {
  const int64_t m = input_size_m,
                n = input_size_n;  // the problem's shape: the input's
  using K = Template;
  const OpType op = OpType::all_reduce_add_attn_res_rms_norm;
  const int world = h.world_size();
  const Template fn = k.fn;
  const int algorithm = algorithm_of(k.config.build);
  const KernelConfig& c = k.config;
  if (const std::optional<Error> e =
          refused(&h, fn, c, dtype, m, n, std::nullopt,
                  false))
    return *e;
  const AllReduceAddAttnResRmsNormBuildConfig& a =
      std::get<AllReduceAddAttnResRmsNormBuildConfig>(c.build);
  const int tm = a.tile_m, tk = a.tile_k;
  // THE PARTIALS'S ROWS, what partials_need sized it for (a one-shot keeps none).
  const int64_t partials_rows =
      n > 0 ? partials_need(fn, algorithm, m,
                           n * elem_bytes(dtype) / pack_bytes_of(c.build), world,
                           pack_bytes_of(c.build)) /
                  (n * elem_bytes(dtype))
            : 0;
  Config config{};
  // A ROW WIDER THAN THE TILE keeps its rows' logits in the launch's LDS (attn_res.cuh's
  // attn_res_wide): [tile_m][num_blocks + 1] floats.
  config.shared_memory_bytes =
      n > a.tile_n ? static_cast<int>(a.tile_m * (num_blocks + 1) * sizeof(float)) : 0;
  config.weight_dtype = dtype;
  config.kernel = fn;
  config.algorithm = algorithm;
  config.world = world;
  config.tile_m = tm;
  config.tile_n = a.tile_n;
  config.tile_k = tk;
  config.block = a.block;
  config.grid = c.launch.grid;
  config.waves_per_eu = a.waves_per_eu;
  config.quantization = a.quantization;
  config.pack_bytes = a.pack_bytes;  // the build's: the instance dispatch finds
  const int64_t row_n = payload_row(fn, algorithm, a.quantization, n);
  config.partials_stride_m = row_n;
  config.partials_stride_n = 1;
  config.partials_size_m = partials_rows;
  config.partials_size_n = row_n;
  config.staging_stride_m = row_n;
  config.staging_stride_n = 1;
  config.staging_size_m = staged_rows(fn, algorithm, a.quantization, m, world);
  config.staging_size_n = row_n;
  config.dtype = dtype;
  config.contiguous = input_stride_m == n * input_stride_n &&
      output_stride_m == n * output_stride_n;
  config.in_place = !is_push(config.algorithm) &&
      config.quantization == Quantization::exact;
  // A ROW WIDER THAN THE TILE runs the build's CHUNKED instance.
  config.chunked = n > config.tile_n;
  // AN INT4 ROW WITH A TAIL runs the build's TAIL instance (int4_tail).
  config.tail = config.quantization == Quantization::int4 && int4_tail(fn, algorithm, m, n, world);
  const void* kernel = nullptr;
  with_config<op>(config, [&]<Config C>() { kernel = kernel_pointer<C>(); });
  if (!kernel) return Error::tile_not_built;
  if (const std::optional<Error> e = h.check_kernel(kernel, config.grid, config.block,
                                                     config.shared_memory_bytes, true))
    return *e;
  return config;
}

// =================================================================================================
// THE NORMS ON THIS RANK'S ROWS (Algorithm::rows): an op Python composes as two kernels of ours runs
// our all-reduce, then one of these on its sum. A norm on rows waits on no peer: its check reads
// nothing a peer holds (no Handle), and any grid runs.
// =================================================================================================

// A NORM ON ROWS' CONFIG: the build and grid `k` names, on rows `n` wide of `dtype` (its weight
// `weight_dtype`), its launch's LDS `shared_memory_bytes`. Its instances are one at every world.
inline Config rows_config(const Handle& h, const Chosen& k, int64_t n, DType dtype,
                          DType weight_dtype, int shared_memory_bytes) {
  const BuildConfig& b = k.config.build;
  Config config{};
  config.shared_memory_bytes = shared_memory_bytes;
  config.kernel = k.fn;
  config.algorithm = algorithm_of(b);
  config.world = h.world_size();
  config.tile_m = tile_m_of(b);
  config.tile_n = tile_n_of(b);
  config.tile_k = std::visit([](const auto& f) -> int {
    if constexpr (requires { f.tile_k; }) return f.tile_k;
    else return 1;
  }, b);
  config.block = block_of(b);
  config.grid = k.config.launch.grid;
  config.waves_per_eu = waves_of(b);
  config.quantization = quantization_of(b);
  config.pack_bytes = pack_bytes_of(b);
  config.dtype = dtype;
  config.weight_dtype = weight_dtype;
  // A ROW WIDER THAN THE TILE runs the build's CHUNKED instance.
  config.chunked = n > config.tile_n;
  return config;
}

// rms_norm: out = rms_norm(inp, weight), vLLM's roundings exactly, on this rank's `m` rows of `n`;
// the weight `weight_dtype`, dtype or f32.
inline std::variant<Config, Error> rms_norm_config(const Handle& h, int64_t m, int64_t n,
                                                   DType dtype, DType weight_dtype,
                                                   const Chosen& k) {
  const OpType op = OpType::rms_norm;
  if (const std::optional<Error> e = refused(nullptr, k.fn, k.config, dtype, m, n,
                                             weight_refused(op, dtype, weight_dtype), true))
    return *e;
  const Config config = rows_config(h, k, n, dtype, weight_dtype, 0);
  const void* kernel = nullptr;
  with_config<op>(config, [&]<Config C>() { kernel = kernel_pointer<C>(); });
  if (!kernel) return Error::tile_not_built;
  if (const std::optional<Error> e = h.check_kernel(kernel, config.grid, config.block,
                                                     config.shared_memory_bytes, false))
    return *e;
  return config;
}

// add_rms_norm: out, residual_out = fused_add_rms_norm(inp, residual, weight), vLLM's roundings
// exactly, on this rank's `m` rows of `n`.
inline std::variant<Config, Error> add_rms_norm_config(const Handle& h, int64_t m, int64_t n,
                                                       DType dtype, DType weight_dtype,
                                                       const Chosen& k) {
  const OpType op = OpType::add_rms_norm;
  if (const std::optional<Error> e = refused(nullptr, k.fn, k.config, dtype, m, n,
                                             weight_refused(op, dtype, weight_dtype), true))
    return *e;
  const Config config = rows_config(h, k, n, dtype, weight_dtype, 0);
  const void* kernel = nullptr;
  with_config<op>(config, [&]<Config C>() { kernel = kernel_pointer<C>(); });
  if (!kernel) return Error::tile_not_built;
  if (const std::optional<Error> e = h.check_kernel(kernel, config.grid, config.block,
                                                     config.shared_memory_bytes, false))
    return *e;
  return config;
}

// add_attn_res_rms_norm: Kimi-K3's AttnRes and its RMSNorm on each of this rank's `m` rows of `n`
// (all_reduce_add_attn_res_rms_norm's, its input the sum), `num_blocks` sources.
inline std::variant<Config, Error> add_attn_res_rms_norm_config(const Handle& h, int64_t m,
                                                                int64_t n, DType dtype,
                                                                int num_blocks, const Chosen& k) {
  const OpType op = OpType::add_attn_res_rms_norm;
  if (const std::optional<Error> e =
          refused(nullptr, k.fn, k.config, dtype, m, n, std::nullopt, true))
    return *e;
  // A ROW WIDER THAN THE TILE keeps its rows' logits in the launch's LDS (attn_res.cuh's
  // attn_res_wide): [tile_m][num_blocks + 1] floats.
  const int tile_m = tile_m_of(k.config.build), tile_n = tile_n_of(k.config.build);
  const Config config = rows_config(
      h, k, n, dtype, dtype,
      n > tile_n ? static_cast<int>(tile_m * (num_blocks + 1) * sizeof(float)) : 0);
  const void* kernel = nullptr;
  with_config<op>(config, [&]<Config C>() { kernel = kernel_pointer<C>(); });
  if (!kernel) return Error::tile_not_built;
  if (const std::optional<Error> e = h.check_kernel(kernel, config.grid, config.block,
                                                     config.shared_memory_bytes, false))
    return *e;
  return config;
}

// =================================================================================================
// EVERY TUNED SELECTION FITS, for every op at the smallest call (8 columns on 2
// ranks: narrower than any slicing kernel takes) and a large one: an op never
// declines (a fusion that is on runs its fused op), and a kernel past a
// capability is a compile error, not one that overruns its signal slots.
// =================================================================================================

// =================================================================================================
// A PLAIN CALL'S FACTS, an op's (nothing forced, contiguous, whole packs): its
// row and its bytes up to these always run. `build_info` hands them to Python's
// `should`, which admits nothing past them.
// =================================================================================================

// THE EXCHANGE, SIZED FROM THE PROBLEM (STANDARDS, *A kernel's own buffers are
// sized from the problem it serves*): the largest call a communicator serves is
// `max_rows` rows of at most `max_cols` columns (for a model, the server's
// tokens a batch by its hidden size), and each region holds its largest
// payload at that call, in every built dtype and in either encoding (exact or
// INT4: common.cuh's INT4 payload layout): the inputs the whole input (an eager
// exact call's host copy, read through the eager input table; the pushed
// all-reduce's inbox of WORLD slots, a band of rows a pass, so at least a row
// a slot), or its INT4 payload; the partials the most any two-shot keeps
// (`partials_need`), or its INT4 payload (bounded as one row of its elements).
// Each need grows with the rows and the row's width, so a call inside the box
// fits by construction. Each rounded to a whole pack, and at least one.
struct ExchangeSizes {
  int64_t partials_bytes;
  int64_t staging_bytes;
};
constexpr ExchangeSizes exchange_sizes(int64_t max_rows, int64_t max_cols,
                                       int world) {
  // THE WIDEST PACK, a row's packs rounded up: a narrower build's need is at most its.
  const int64_t pack = kBuild.memory.pack_bytes;
  const auto whole = [pack](int64_t bytes) {
    return bytes < pack ? pack : (bytes + pack - 1) / pack * pack;
  };
  int64_t partials = 0, staging = int4_payload_bytes(max_rows, max_cols);
  for (const DType d : kBuild.supports.dtypes) {
    const int64_t packs = packs_of(max_cols, d, pack);
    for (const TemplateInfo& t : kTemplates)
      for (const int a : t.algorithms) {
        const int64_t need = partials_need(t.fn, a, max_rows, packs, world, pack);
        if (need > partials) partials = need;
        const int64_t int4 = int4_payload_bytes(1, need / elem_bytes(d));
        if (need > 0 && int4 > partials) partials = int4;
      }
    const int64_t input = max_rows * max_cols * elem_bytes(d);
    if (input > staging) staging = input;
    // THE PUSHED TWO-SHOT'S INBOX (forced) takes a pass of at least one row:
    // WORLD slots of a row each covers its WORLD shares of one row rounded up
    // to whole packs. The in-place pull's eager copy is the input whole,
    // above. (The pushed one-shot's inbox is its own region, handle.cuh's
    // UNFENCED REGION.)
    const int64_t one_row_each = world * max_cols * elem_bytes(d);
    if (one_row_each > staging) staging = one_row_each;
  }
  return {whole(partials), whole(staging)};
}

// =================================================================================================
// WHAT A CALL MAY HOLD: what Python's `should` admits by.
// =================================================================================================

// THE MOST BYTES A CALL OF AN OP RUNS ON THESE BUFFERS, whatever its shape: what the inputs holds
// (an eager input is copied in whole), and what the partials holds at the most partials a byte of
// its input needs among the op's algorithms (any may be a config's): at one row of one pack, where
// rounding up is all of it. Python admits a call by it or by the box the exchange was sized for,
// whichever is larger.
constexpr int64_t max_bytes(OpType o, int64_t partials_bytes, int64_t staging_bytes) {
  int64_t per_byte = 0;
  for (const TemplateInfo& t : kTemplates)
    if (t.op == o)
      for (const int a : t.algorithms) {
        const int64_t need = partials_need(t.fn, a, 1, 1, kMaxRanks, kBuild.memory.pack_bytes) /
                             kBuild.memory.pack_bytes;
        if (need > per_byte) per_byte = need;
      }
  if (per_byte == 0) return staging_bytes;
  const int64_t partials = partials_bytes / per_byte;
  return partials < staging_bytes ? partials : staging_bytes;
}

}  // namespace hip_custom_communicator


// =================================================================================================
// HIP CUSTOM COMMUNICATOR (was hip_custom_communicator.cuh)
// =================================================================================================

// Each op checks its call, obtains a complete config, and launches its kernels.





namespace hip_custom_communicator {

// A LAUNCH DIMENSION AS HIP TAKES IT.
inline dim3 dim3_of(const Dim3& d) {
  return dim3(static_cast<unsigned>(d.x), static_cast<unsigned>(d.y),
              static_cast<unsigned>(d.z));
}

// =================================================================================================
// all_reduce: the sum of every rank's `input` into `output`, each at its own
// strides and sizes.
// =================================================================================================

inline std::variant<Config, Error> all_reduce(
    Handle& h, const void* input, int64_t input_stride_m,
    int64_t input_stride_n, int64_t input_size_m, int64_t input_size_n,
    void* output, int64_t output_stride_m, int64_t output_stride_n,
    int64_t output_size_m, int64_t output_size_n, DType dtype, const Chosen& k, hipStream_t stream) {
  if (const std::optional<Error> e =
          layout_refused(dtype, pack_bytes_of(k.config.build), {{output_stride_m, output_stride_n},
                                 {input_stride_m, input_stride_n}}))
    return *e;
  std::variant<Config, Error> checked = all_reduce_config(
      h, input, input_stride_m, input_stride_n, input_size_m, input_size_n,
      output, output_stride_m, output_stride_n, output_size_m, output_size_n,
      dtype, k, stream);
  if (const Error* e = std::get_if<Error>(&checked)) return *e;
  const Config config = std::get<Config>(checked);
  with_config<OpType::all_reduce>(config, [&]<Config C>() {
    using T = typename of_dtype<C.dtype>::t;
    using W = typename of_dtype<C.weight_dtype>::t;
    // BEFORE THE LAUNCH, every expectation of what all_reduce is handed (October 26: n11 rebooted
    // seven times timing the fused two-shots at size; the kernels check their indices only
    // against these sizes, so a size past its region was never caught): each view of a region
    // inside the region, and the grid inside the signals' slots.
    const auto fits = [&](const char* region, int64_t slots, int64_t stride_slot, int64_t rows,
                          int64_t stride_m, int64_t cols, int64_t stride_n, int64_t elem,
                          int64_t region_bytes) {
      if (slots <= 0 || rows <= 0 || cols <= 0) return;
      const int64_t last = (slots - 1) * stride_slot + (rows - 1) * stride_m + (cols - 1) * stride_n;
      if (stride_slot < 0 || stride_m < 0 || stride_n < 0 || (last + 1) * elem > region_bytes)
        throw std::runtime_error(std::string("hip_custom_communicator all_reduce before its launch: the ") +
            region + " view (" + std::to_string(slots) + " slots of " + std::to_string(rows) +
            " rows of " + std::to_string(cols) + ", strides " + std::to_string(stride_slot) + ", " +
            std::to_string(stride_m) + ", " + std::to_string(stride_n) + ") ends at byte " +
            std::to_string((last + 1) * elem) + ", past the region's " + std::to_string(region_bytes));
    };
    fits("partials", 1, 0, config.partials_size_m, config.partials_stride_m, config.partials_size_n,
         config.partials_stride_n, elem_bytes(dtype), h.partials_bytes());
    fits("staging", 1, 0, config.staging_size_m, config.staging_stride_m, config.staging_size_n,
         config.staging_stride_n, elem_bytes(dtype), h.staging_bytes());
    fits("unfenced", config.unfenced_size_slot, config.unfenced_stride_slot, config.unfenced_size_m,
         config.unfenced_stride_m, config.unfenced_size_n, config.unfenced_stride_n,
         elem_bytes(dtype) * (kLineBytes / kLinePayloadBytes), h.unfenced_region_bytes());
    fits("unfenced epochs", 1, 0, h.unfenced_epochs_size_block(), h.unfenced_epochs_stride_block(),
         h.unfenced_epochs_size_n(), h.unfenced_epochs_stride_n(), 4, kUnfencedEpochsBytes);
    if (volume(config.grid) > h.signals_size_block())
      throw std::runtime_error("hip_custom_communicator all_reduce before its launch: a grid of " +
          std::to_string(volume(config.grid)) + " blocks past the signals' " +
          std::to_string(h.signals_size_block()));
    h.begin_launch(OpType::all_reduce, kernel_pointer<C>(), input_size_m, input_size_n, dtype,
        stream);
    const InputTable t = config.in_place
        ? h.input_table(input, input_stride_m, input_size_m, input_size_n, elem_bytes(dtype),
            stream)
        : InputTable{nullptr, 0, config.staging_stride_m, config.staging_stride_n};
    all_reduce<T, C.world, C.tile_m, C.tile_n, C.block.x, C.wave_size, C.pack_bytes,
        C.waves_per_eu, C.quantization, C.algorithm, C.contiguous, C.tail>
        <<<dim3_of(config.grid), dim3_of(C.block), config.shared_memory_bytes, stream>>>(
        static_cast<const T*>(input), input_stride_m, input_stride_n,
        static_cast<int>(input_size_m), static_cast<int>(input_size_n),
        reinterpret_cast<const T* const*>(t.ptr_array), t.ptr_array_size, t.stride_m, t.stride_n,
        static_cast<T*>(output), output_stride_m, output_stride_n,
        static_cast<int>(output_size_m), static_cast<int>(output_size_n),
        h.partials_ptr_array(), h.partials_ptr_array_size(), config.partials_stride_m,
        config.partials_stride_n, static_cast<int>(config.partials_size_m),
        static_cast<int>(config.partials_size_n), h.staging_ptr_array(),
        h.staging_ptr_array_size(), config.staging_stride_m, config.staging_stride_n,
        static_cast<int>(config.staging_size_m), static_cast<int>(config.staging_size_n),
        h.unfenced_ptr_array(), h.unfenced_ptr_array_size(), config.unfenced_stride_slot,
        config.unfenced_stride_m, config.unfenced_stride_n,
        static_cast<int>(config.unfenced_size_slot), static_cast<int>(config.unfenced_size_m),
        static_cast<int>(config.unfenced_size_n), h.unfenced_epochs_ptr(),
        h.unfenced_epochs_stride_block(), h.unfenced_epochs_stride_n(),
        static_cast<int>(h.unfenced_epochs_size_block()),
        static_cast<int>(h.unfenced_epochs_size_n()), h.signals_ptr_array(),
        h.signals_ptr_array_size(), h.signals_stride_parity(), h.signals_stride_block(),
        h.signals_stride_writer(), h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks());
    // AFTER THE LAUNCH, checked (VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_CHECKED=1) and not captured: the
    // kernel done, every expectation of all_reduce after it: no write past a region (the guards), no
    // fault, every output finite where the input is.
    if (h.checked()) {
      hipStreamCaptureStatus capturing = hipStreamCaptureStatusNone;
      HIP_CHECK(hipStreamIsCapturing(stream, &capturing));
      if (capturing == hipStreamCaptureStatusNone) {
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(stream));
        if (const std::string broken = h.guards_broken(); !broken.empty())
          throw std::runtime_error("hip_custom_communicator all_reduce after its launch: " + broken);
        const auto status = h.status();
        if (status[static_cast<int>(StatusField::state)] != 0 ||
            status[static_cast<int>(StatusField::abort)] != 0)
          throw std::runtime_error("hip_custom_communicator all_reduce after its launch: a fault or an "
                                   "abort in the status");
        // EVERY SIGNAL CLEAR is every rank's promise once every rank finished (a faster peer's next
        // launch sets its first barrier's flags here before this rank looks): the caller's, between
        // two barriers of every rank (the communicator's check_after, the tests' _assert_clean).
        const auto finite = [&](const void* ptr, int64_t stride_m, int64_t rows, int64_t cols) {
          std::vector<T> host(static_cast<size_t>(rows * cols));
          HIP_CHECK(hipMemcpy2D(host.data(), cols * sizeof(T), ptr, stride_m * sizeof(T),
                                cols * sizeof(T), rows, hipMemcpyDeviceToHost));
          for (const T& x : host)
            if (!std::isfinite(static_cast<float>(x))) return false;
          return true;
        };
        if (finite(input, input_stride_m, input_size_m, input_size_n)) {
          if (!finite(output, output_stride_m, output_size_m, output_size_n))
            throw std::runtime_error("hip_custom_communicator all_reduce after its launch: output not "
                                     "finite where the input is");
        }
      }
    }
  });
  HIP_CHECK(hipGetLastError());
  return config;
}

// =================================================================================================
// THE NORMS: output = rms_norm(all_reduce(input), weight), or with a residual
// vLLM's fused_add_rms_norm (output and residual_output), its roundings
// exactly. input [rows, hidden]; the weight `weight_dtype`, dtype or f32.
// =================================================================================================

inline std::variant<Config, Error> all_reduce_rms_norm(
    Handle& h, const void* input, int64_t input_stride_m,
    int64_t input_stride_n, int64_t input_size_m, int64_t input_size_n,
    const void* weight, int64_t weight_stride_n, int64_t weight_size_n,
    void* output, int64_t output_stride_m, int64_t output_stride_n,
    int64_t output_size_m, int64_t output_size_n, DType dtype,
    DType weight_dtype, float eps, const Chosen& k, hipStream_t stream) {
  if (const std::optional<Error> e =
          layout_refused(dtype, pack_bytes_of(k.config.build), {{output_stride_m, output_stride_n},
                                 {input_stride_m, input_stride_n}}))
    return *e;
  if (const std::optional<Error> e =
          layout_refused(weight_dtype, pack_bytes_of(k.config.build), {{0, weight_stride_n}}))
    return *e;
  std::variant<Config, Error> checked = all_reduce_rms_norm_config(
      h, input, input_stride_m, input_stride_n, input_size_m, input_size_n,
      weight, weight_stride_n, weight_size_n, output, output_stride_m,
      output_stride_n, output_size_m, output_size_n, dtype, weight_dtype, eps,
      k, stream);
  if (const Error* e = std::get_if<Error>(&checked)) return *e;
  const Config config = std::get<Config>(checked);
  with_config<OpType::all_reduce_rms_norm>(config, [&]<Config C>() {
    using T = typename of_dtype<C.dtype>::t;
    using W = typename of_dtype<C.weight_dtype>::t;
    // BEFORE THE LAUNCH, every expectation of what all_reduce_rms_norm is handed (October 26: n11 rebooted
    // seven times timing the fused two-shots at size; the kernels check their indices only
    // against these sizes, so a size past its region was never caught): each view of a region
    // inside the region, and the grid inside the signals' slots.
    const auto fits = [&](const char* region, int64_t slots, int64_t stride_slot, int64_t rows,
                          int64_t stride_m, int64_t cols, int64_t stride_n, int64_t elem,
                          int64_t region_bytes) {
      if (slots <= 0 || rows <= 0 || cols <= 0) return;
      const int64_t last = (slots - 1) * stride_slot + (rows - 1) * stride_m + (cols - 1) * stride_n;
      if (stride_slot < 0 || stride_m < 0 || stride_n < 0 || (last + 1) * elem > region_bytes)
        throw std::runtime_error(std::string("hip_custom_communicator all_reduce_rms_norm before its launch: the ") +
            region + " view (" + std::to_string(slots) + " slots of " + std::to_string(rows) +
            " rows of " + std::to_string(cols) + ", strides " + std::to_string(stride_slot) + ", " +
            std::to_string(stride_m) + ", " + std::to_string(stride_n) + ") ends at byte " +
            std::to_string((last + 1) * elem) + ", past the region's " + std::to_string(region_bytes));
    };
    fits("partials", 1, 0, config.partials_size_m, config.partials_stride_m, config.partials_size_n,
         config.partials_stride_n, elem_bytes(dtype), h.partials_bytes());
    fits("staging", 1, 0, config.staging_size_m, config.staging_stride_m, config.staging_size_n,
         config.staging_stride_n, elem_bytes(dtype), h.staging_bytes());
    if (volume(config.grid) > h.signals_size_block())
      throw std::runtime_error("hip_custom_communicator all_reduce_rms_norm before its launch: a grid of " +
          std::to_string(volume(config.grid)) + " blocks past the signals' " +
          std::to_string(h.signals_size_block()));
    h.begin_launch(OpType::all_reduce_rms_norm, kernel_pointer<C>(), input_size_m, input_size_n,
        dtype, stream);
    const InputTable t = config.quantization != Quantization::exact
        ? InputTable{nullptr, 0, config.staging_stride_m, config.staging_stride_n}
        : h.input_table(input, input_stride_m, input_size_m, input_size_n, elem_bytes(dtype),
            stream);
    all_reduce_rms_norm<T, W, C.world, C.tile_n, C.block.x, C.wave_size, C.pack_bytes,
        C.waves_per_eu, C.quantization, C.algorithm, C.chunked, C.tail>
        <<<dim3_of(config.grid), dim3_of(C.block), config.shared_memory_bytes, stream>>>(
        static_cast<const T*>(input),
        input_stride_m, input_stride_n, static_cast<int>(input_size_m),
        static_cast<int>(input_size_n), reinterpret_cast<const T* const*>(t.ptr_array),
            t.ptr_array_size, t.stride_m, t.stride_n, static_cast<const W*>(weight),
            weight_stride_n,
        static_cast<int>(weight_size_n), static_cast<T*>(output), output_stride_m,
        output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), h.partials_ptr_array(), h.partials_ptr_array_size(),
         config.partials_stride_m,
        config.partials_stride_n, static_cast<int>(config.partials_size_m),
        static_cast<int>(config.partials_size_n), h.staging_ptr_array(),
            h.staging_ptr_array_size(), config.staging_stride_m,
        config.staging_stride_n, static_cast<int>(config.staging_size_m),
        static_cast<int>(config.staging_size_n), h.signals_ptr_array(),
            h.signals_ptr_array_size(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), eps);
    // AFTER THE LAUNCH, checked (VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_CHECKED=1) and not captured: the
    // kernel done, every expectation of all_reduce_rms_norm after it: no write past a region (the guards), no
    // fault, every output finite where the input is.
    if (h.checked()) {
      hipStreamCaptureStatus capturing = hipStreamCaptureStatusNone;
      HIP_CHECK(hipStreamIsCapturing(stream, &capturing));
      if (capturing == hipStreamCaptureStatusNone) {
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(stream));
        if (const std::string broken = h.guards_broken(); !broken.empty())
          throw std::runtime_error("hip_custom_communicator all_reduce_rms_norm after its launch: " + broken);
        const auto status = h.status();
        if (status[static_cast<int>(StatusField::state)] != 0 ||
            status[static_cast<int>(StatusField::abort)] != 0)
          throw std::runtime_error("hip_custom_communicator all_reduce_rms_norm after its launch: a fault or an "
                                   "abort in the status");
        // EVERY SIGNAL CLEAR is every rank's promise once every rank finished (a faster peer's next
        // launch sets its first barrier's flags here before this rank looks): the caller's, between
        // two barriers of every rank (the communicator's check_after, the tests' _assert_clean).
        const auto finite = [&](const void* ptr, int64_t stride_m, int64_t rows, int64_t cols) {
          std::vector<T> host(static_cast<size_t>(rows * cols));
          HIP_CHECK(hipMemcpy2D(host.data(), cols * sizeof(T), ptr, stride_m * sizeof(T),
                                cols * sizeof(T), rows, hipMemcpyDeviceToHost));
          for (const T& x : host)
            if (!std::isfinite(static_cast<float>(x))) return false;
          return true;
        };
        if (finite(input, input_stride_m, input_size_m, input_size_n)) {
          if (!finite(output, output_stride_m, output_size_m, output_size_n))
            throw std::runtime_error("hip_custom_communicator all_reduce_rms_norm after its launch: output not "
                                     "finite where the input is");
        }
      }
    }
  });
  HIP_CHECK(hipGetLastError());
  return config;
}

inline std::variant<Config, Error> all_reduce_add_rms_norm(
    Handle& h, const void* input, int64_t input_stride_m,
    int64_t input_stride_n, int64_t input_size_m, int64_t input_size_n,
    const void* residual, int64_t residual_stride_m, int64_t residual_stride_n,
    int64_t residual_size_m, int64_t residual_size_n, const void* weight,
    int64_t weight_stride_n, int64_t weight_size_n, void* output,
    int64_t output_stride_m, int64_t output_stride_n, int64_t output_size_m,
    int64_t output_size_n, void* residual_output,
    int64_t residual_output_stride_m, int64_t residual_output_stride_n,
    int64_t residual_output_size_m, int64_t residual_output_size_n, DType dtype,
    DType weight_dtype, float eps, const Chosen& k, hipStream_t stream) {
  if (const std::optional<Error> e = layout_refused(
          dtype, pack_bytes_of(k.config.build), {{output_stride_m, output_stride_n},
                  {residual_output_stride_m, residual_output_stride_n},
                  {input_stride_m, input_stride_n},
                  {residual_stride_m, residual_stride_n}}))
    return *e;
  if (const std::optional<Error> e =
          layout_refused(weight_dtype, pack_bytes_of(k.config.build), {{0, weight_stride_n}}))
    return *e;
  std::variant<Config, Error> checked =
      all_reduce_add_rms_norm_config(
          h, input, input_stride_m, input_stride_n, input_size_m, input_size_n,
          residual, residual_stride_m, residual_stride_n, residual_size_m,
          residual_size_n, weight, weight_stride_n, weight_size_n, output,
          output_stride_m, output_stride_n, output_size_m, output_size_n,
          residual_output, residual_output_stride_m, residual_output_stride_n,
          residual_output_size_m, residual_output_size_n, dtype, weight_dtype,
          eps, k, stream);
  if (const Error* e = std::get_if<Error>(&checked)) return *e;
  const Config config = std::get<Config>(checked);
  with_config<OpType::all_reduce_add_rms_norm>(config, [&]<Config C>() {
    using T = typename of_dtype<C.dtype>::t;
    using W = typename of_dtype<C.weight_dtype>::t;
    // BEFORE THE LAUNCH, every expectation of what all_reduce_add_rms_norm is handed (October 26: n11 rebooted
    // seven times timing the fused two-shots at size; the kernels check their indices only
    // against these sizes, so a size past its region was never caught): each view of a region
    // inside the region, and the grid inside the signals' slots.
    const auto fits = [&](const char* region, int64_t slots, int64_t stride_slot, int64_t rows,
                          int64_t stride_m, int64_t cols, int64_t stride_n, int64_t elem,
                          int64_t region_bytes) {
      if (slots <= 0 || rows <= 0 || cols <= 0) return;
      const int64_t last = (slots - 1) * stride_slot + (rows - 1) * stride_m + (cols - 1) * stride_n;
      if (stride_slot < 0 || stride_m < 0 || stride_n < 0 || (last + 1) * elem > region_bytes)
        throw std::runtime_error(std::string("hip_custom_communicator all_reduce_add_rms_norm before its launch: the ") +
            region + " view (" + std::to_string(slots) + " slots of " + std::to_string(rows) +
            " rows of " + std::to_string(cols) + ", strides " + std::to_string(stride_slot) + ", " +
            std::to_string(stride_m) + ", " + std::to_string(stride_n) + ") ends at byte " +
            std::to_string((last + 1) * elem) + ", past the region's " + std::to_string(region_bytes));
    };
    fits("partials", 1, 0, config.partials_size_m, config.partials_stride_m, config.partials_size_n,
         config.partials_stride_n, elem_bytes(dtype), h.partials_bytes());
    fits("staging", 1, 0, config.staging_size_m, config.staging_stride_m, config.staging_size_n,
         config.staging_stride_n, elem_bytes(dtype), h.staging_bytes());
    fits("scales", 1, 0, config.scales_size_m, config.scales_stride_m, 1, 1, 4, h.scales_region_bytes());
    if (volume(config.grid) > h.signals_size_block())
      throw std::runtime_error("hip_custom_communicator all_reduce_add_rms_norm before its launch: a grid of " +
          std::to_string(volume(config.grid)) + " blocks past the signals' " +
          std::to_string(h.signals_size_block()));
    h.begin_launch(OpType::all_reduce_add_rms_norm, kernel_pointer<C>(), input_size_m,
        input_size_n, dtype, stream);
    const InputTable t = config.quantization != Quantization::exact
        ? InputTable{nullptr, 0, config.staging_stride_m, config.staging_stride_n}
        : h.input_table(input, input_stride_m, input_size_m, input_size_n, elem_bytes(dtype),
            stream);
    all_reduce_add_rms_norm<T, W, C.world, C.tile_n, C.block.x, C.wave_size, C.pack_bytes,
        C.waves_per_eu, C.quantization, C.algorithm, C.chunked, C.tail>
        <<<dim3_of(config.grid), dim3_of(C.block), config.shared_memory_bytes, stream>>>(
        static_cast<const T*>(input),
        input_stride_m, input_stride_n, static_cast<int>(input_size_m),
        static_cast<int>(input_size_n), reinterpret_cast<const T* const*>(t.ptr_array),
            t.ptr_array_size, t.stride_m, t.stride_n, static_cast<const T*>(residual),
            residual_stride_m,
        residual_stride_n, static_cast<int>(residual_size_m),
        static_cast<int>(residual_size_n), static_cast<const W*>(weight), weight_stride_n,
        static_cast<int>(weight_size_n), static_cast<T*>(output), output_stride_m,
        output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), static_cast<T*>(residual_output),
        residual_output_stride_m, residual_output_stride_n,
        static_cast<int>(residual_output_size_m),
        static_cast<int>(residual_output_size_n), h.partials_ptr_array(),
            h.partials_ptr_array_size(),
         config.partials_stride_m,
        config.partials_stride_n, static_cast<int>(config.partials_size_m),
        static_cast<int>(config.partials_size_n), h.scales_ptr_array(),
        h.scales_ptr_array_size(), config.scales_stride_m, static_cast<int>(config.scales_size_m),
        h.staging_ptr_array(), h.staging_ptr_array_size(), config.staging_stride_m,
        config.staging_stride_n, static_cast<int>(config.staging_size_m),
        static_cast<int>(config.staging_size_n), h.signals_ptr_array(),
            h.signals_ptr_array_size(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), eps);
    // AFTER THE LAUNCH, checked (VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_CHECKED=1) and not captured: the
    // kernel done, every expectation of all_reduce_add_rms_norm after it: no write past a region (the guards), no
    // fault, every output finite where the input is.
    if (h.checked()) {
      hipStreamCaptureStatus capturing = hipStreamCaptureStatusNone;
      HIP_CHECK(hipStreamIsCapturing(stream, &capturing));
      if (capturing == hipStreamCaptureStatusNone) {
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(stream));
        if (const std::string broken = h.guards_broken(); !broken.empty())
          throw std::runtime_error("hip_custom_communicator all_reduce_add_rms_norm after its launch: " + broken);
        const auto status = h.status();
        if (status[static_cast<int>(StatusField::state)] != 0 ||
            status[static_cast<int>(StatusField::abort)] != 0)
          throw std::runtime_error("hip_custom_communicator all_reduce_add_rms_norm after its launch: a fault or an "
                                   "abort in the status");
        // EVERY SIGNAL CLEAR is every rank's promise once every rank finished (a faster peer's next
        // launch sets its first barrier's flags here before this rank looks): the caller's, between
        // two barriers of every rank (the communicator's check_after, the tests' _assert_clean).
        const auto finite = [&](const void* ptr, int64_t stride_m, int64_t rows, int64_t cols) {
          std::vector<T> host(static_cast<size_t>(rows * cols));
          HIP_CHECK(hipMemcpy2D(host.data(), cols * sizeof(T), ptr, stride_m * sizeof(T),
                                cols * sizeof(T), rows, hipMemcpyDeviceToHost));
          for (const T& x : host)
            if (!std::isfinite(static_cast<float>(x))) return false;
          return true;
        };
        if (finite(input, input_stride_m, input_size_m, input_size_n)) {
          if (!finite(output, output_stride_m, output_size_m, output_size_n))
            throw std::runtime_error("hip_custom_communicator all_reduce_add_rms_norm after its launch: output not "
                                     "finite where the input is");
          if (!finite(residual_output, residual_output_stride_m, residual_output_size_m, residual_output_size_n))
            throw std::runtime_error("hip_custom_communicator all_reduce_add_rms_norm after its launch: residual_output not "
                                     "finite where the input is");
        }
      }
    }
  });
  HIP_CHECK(hipGetLastError());
  return config;
}

// =================================================================================================
// all_reduce_add_attn_res_rms_norm: Kimi-K3's AttnRes and its RMSNorm on each
// row of the all-reduced sum of `input` (the kernels spell it out). With a
// `prefix` the sum is added to it, into `prefix_output`; without (null), the
// sum IS the new prefix. `block_output`, when not null, gets the new prefix
// too: the slot of `blocks` the caller appends to.
// =================================================================================================

inline std::variant<Config, Error>
all_reduce_add_attn_res_rms_norm(
    Handle& h, const void* input, int64_t input_stride_m,
    int64_t input_stride_n, int64_t input_size_m, int64_t input_size_n,
    const void* prefix, int64_t prefix_stride_m, int64_t prefix_stride_n,
    int64_t prefix_size_m, int64_t prefix_size_n, const void* blocks,
    int64_t blocks_stride_m, int64_t blocks_stride_r, int64_t blocks_stride_n,
    int64_t blocks_size_m, int64_t blocks_size_r, int64_t blocks_size_n,
    const void* norm_weight, int64_t norm_weight_stride_n,
    int64_t norm_weight_size_n, const void* qk_weight,
    int64_t qk_weight_stride_n, int64_t qk_weight_size_n,
    const void* output_norm_weight, int64_t output_norm_weight_stride_n,
    int64_t output_norm_weight_size_n, void* prefix_output,
    int64_t prefix_output_stride_m, int64_t prefix_output_stride_n,
    int64_t prefix_output_size_m, int64_t prefix_output_size_n, void* output,
    int64_t output_stride_m, int64_t output_stride_n, int64_t output_size_m,
    int64_t output_size_n, void* block_output, int64_t block_output_stride_m,
    int64_t block_output_stride_n, int64_t block_output_size_m,
    int64_t block_output_size_n, DType dtype, int num_blocks, float eps,
    float output_eps, const Chosen& k, hipStream_t stream) {
  if (const std::optional<Error> e = layout_refused(
          dtype, pack_bytes_of(k.config.build), {{input_stride_m, input_stride_n},
                  {blocks_stride_m, blocks_stride_n},
                  {blocks_stride_r, blocks_stride_n},
                  {0, norm_weight_stride_n},
                  {0, qk_weight_stride_n},
                  {prefix_output_stride_m, prefix_output_stride_n},
                  {output_stride_m, output_stride_n}}))
    return *e;
  // THE OPTIONAL BUFFERS, checked only when given: a missing one is a null
  // pointer with zero strides, which is no layout at all (a prefix-less call
  // was refused as `inner_stride_not_one`, 2026-10-13).
  if (prefix != nullptr)
    if (const std::optional<Error> e =
            layout_refused(dtype, pack_bytes_of(k.config.build), {{prefix_stride_m, prefix_stride_n}}))
      return *e;
  if (output_norm_weight != nullptr)
    if (const std::optional<Error> e =
            layout_refused(dtype, pack_bytes_of(k.config.build), {{0, output_norm_weight_stride_n}}))
      return *e;
  if (block_output != nullptr)
    if (const std::optional<Error> e = layout_refused(
            dtype, pack_bytes_of(k.config.build), {{block_output_stride_m, block_output_stride_n}}))
      return *e;
  std::variant<Config, Error> checked =
      all_reduce_add_attn_res_rms_norm_config(
          h, input, input_stride_m, input_stride_n, input_size_m, input_size_n,
          prefix, prefix_stride_m, prefix_stride_n, prefix_size_m,
          prefix_size_n, blocks, blocks_stride_m, blocks_stride_r,
          blocks_stride_n, blocks_size_m, blocks_size_r, blocks_size_n,
          norm_weight, norm_weight_stride_n, norm_weight_size_n, qk_weight,
          qk_weight_stride_n, qk_weight_size_n, output_norm_weight,
          output_norm_weight_stride_n, output_norm_weight_size_n, prefix_output,
          prefix_output_stride_m, prefix_output_stride_n, prefix_output_size_m,
          prefix_output_size_n, output, output_stride_m, output_stride_n,
          output_size_m, output_size_n, block_output, block_output_stride_m,
          block_output_stride_n, block_output_size_m, block_output_size_n,
          dtype, num_blocks, eps, output_eps, k, stream);
  if (const Error* e = std::get_if<Error>(&checked)) return *e;
  const Config config = std::get<Config>(checked);
  with_config<OpType::all_reduce_add_attn_res_rms_norm>(config, [&]<Config C>() {
    using T = typename of_dtype<C.dtype>::t;
    using W = typename of_dtype<C.weight_dtype>::t;
    // BEFORE THE LAUNCH, every expectation of what all_reduce_add_attn_res_rms_norm is handed (October 26: n11 rebooted
    // seven times timing the fused two-shots at size; the kernels check their indices only
    // against these sizes, so a size past its region was never caught): each view of a region
    // inside the region, and the grid inside the signals' slots.
    const auto fits = [&](const char* region, int64_t slots, int64_t stride_slot, int64_t rows,
                          int64_t stride_m, int64_t cols, int64_t stride_n, int64_t elem,
                          int64_t region_bytes) {
      if (slots <= 0 || rows <= 0 || cols <= 0) return;
      const int64_t last = (slots - 1) * stride_slot + (rows - 1) * stride_m + (cols - 1) * stride_n;
      if (stride_slot < 0 || stride_m < 0 || stride_n < 0 || (last + 1) * elem > region_bytes)
        throw std::runtime_error(std::string("hip_custom_communicator all_reduce_add_attn_res_rms_norm before its launch: the ") +
            region + " view (" + std::to_string(slots) + " slots of " + std::to_string(rows) +
            " rows of " + std::to_string(cols) + ", strides " + std::to_string(stride_slot) + ", " +
            std::to_string(stride_m) + ", " + std::to_string(stride_n) + ") ends at byte " +
            std::to_string((last + 1) * elem) + ", past the region's " + std::to_string(region_bytes));
    };
    fits("partials", 1, 0, config.partials_size_m, config.partials_stride_m, config.partials_size_n,
         config.partials_stride_n, elem_bytes(dtype), h.partials_bytes());
    fits("staging", 1, 0, config.staging_size_m, config.staging_stride_m, config.staging_size_n,
         config.staging_stride_n, elem_bytes(dtype), h.staging_bytes());
    if (volume(config.grid) > h.signals_size_block())
      throw std::runtime_error("hip_custom_communicator all_reduce_add_attn_res_rms_norm before its launch: a grid of " +
          std::to_string(volume(config.grid)) + " blocks past the signals' " +
          std::to_string(h.signals_size_block()));
    h.begin_launch(OpType::all_reduce_add_attn_res_rms_norm, kernel_pointer<C>(), input_size_m,
        input_size_n, dtype, stream);
    const InputTable t = config.quantization != Quantization::exact
        ? InputTable{nullptr, 0, config.staging_stride_m, config.staging_stride_n}
        : h.input_table(input, input_stride_m, input_size_m, input_size_n, elem_bytes(dtype),
            stream);
    all_reduce_add_attn_res_rms_norm<T, C.world, C.tile_m, C.tile_n, C.tile_k, C.block.x,
        C.wave_size, C.pack_bytes, C.waves_per_eu, C.quantization, C.algorithm,
        C.chunked, C.tail>
        <<<dim3_of(config.grid), dim3_of(C.block), config.shared_memory_bytes, stream>>>(
        static_cast<const T*>(input),
        input_stride_m, input_stride_n, static_cast<int>(input_size_m),
        static_cast<int>(input_size_n), reinterpret_cast<const T* const*>(t.ptr_array),
            t.ptr_array_size, t.stride_m, t.stride_n, static_cast<const T*>(prefix),
            prefix_stride_m,
        prefix_stride_n, static_cast<int>(prefix_size_m),
        static_cast<int>(prefix_size_n), static_cast<const T*>(blocks), blocks_stride_m,
        blocks_stride_r, blocks_stride_n, static_cast<int>(blocks_size_m),
        static_cast<int>(blocks_size_r), static_cast<int>(blocks_size_n),
        static_cast<const T*>(norm_weight), norm_weight_stride_n,
        static_cast<int>(norm_weight_size_n), static_cast<const T*>(qk_weight),
        qk_weight_stride_n, static_cast<int>(qk_weight_size_n),
        static_cast<const T*>(output_norm_weight), output_norm_weight_stride_n,
        static_cast<int>(output_norm_weight_size_n), static_cast<T*>(prefix_output),
        prefix_output_stride_m, prefix_output_stride_n,
        static_cast<int>(prefix_output_size_m),
        static_cast<int>(prefix_output_size_n), static_cast<T*>(output), output_stride_m,
        output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), static_cast<T*>(block_output),
        block_output_stride_m, block_output_stride_n,
        static_cast<int>(block_output_size_m),
        static_cast<int>(block_output_size_n), h.partials_ptr_array(),
        h.partials_ptr_array_size(), config.partials_stride_m, config.partials_stride_n,
        static_cast<int>(config.partials_size_m), static_cast<int>(config.partials_size_n),
        h.staging_ptr_array(), h.staging_ptr_array_size(), config.staging_stride_m,
        config.staging_stride_n, static_cast<int>(config.staging_size_m),
        static_cast<int>(config.staging_size_n), h.signals_ptr_array(),
            h.signals_ptr_array_size(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), num_blocks, eps,
        output_eps);
    // AFTER THE LAUNCH, checked (VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_CHECKED=1) and not captured: the
    // kernel done, every expectation of all_reduce_add_attn_res_rms_norm after it: no write past a region (the guards), no
    // fault, every output finite where the input is.
    if (h.checked()) {
      hipStreamCaptureStatus capturing = hipStreamCaptureStatusNone;
      HIP_CHECK(hipStreamIsCapturing(stream, &capturing));
      if (capturing == hipStreamCaptureStatusNone) {
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(stream));
        if (const std::string broken = h.guards_broken(); !broken.empty())
          throw std::runtime_error("hip_custom_communicator all_reduce_add_attn_res_rms_norm after its launch: " + broken);
        const auto status = h.status();
        if (status[static_cast<int>(StatusField::state)] != 0 ||
            status[static_cast<int>(StatusField::abort)] != 0)
          throw std::runtime_error("hip_custom_communicator all_reduce_add_attn_res_rms_norm after its launch: a fault or an "
                                   "abort in the status");
        // EVERY SIGNAL CLEAR is every rank's promise once every rank finished (a faster peer's next
        // launch sets its first barrier's flags here before this rank looks): the caller's, between
        // two barriers of every rank (the communicator's check_after, the tests' _assert_clean).
        const auto finite = [&](const void* ptr, int64_t stride_m, int64_t rows, int64_t cols) {
          std::vector<T> host(static_cast<size_t>(rows * cols));
          HIP_CHECK(hipMemcpy2D(host.data(), cols * sizeof(T), ptr, stride_m * sizeof(T),
                                cols * sizeof(T), rows, hipMemcpyDeviceToHost));
          for (const T& x : host)
            if (!std::isfinite(static_cast<float>(x))) return false;
          return true;
        };
        if (finite(input, input_stride_m, input_size_m, input_size_n)) {
          if (!finite(output, output_stride_m, output_size_m, output_size_n))
            throw std::runtime_error("hip_custom_communicator all_reduce_add_attn_res_rms_norm after its launch: output not "
                                     "finite where the input is");
        }
      }
    }
  });
  HIP_CHECK(hipGetLastError());
  return config;
}

// =================================================================================================
// THE NORMS ON THIS RANK'S ROWS: each checks its call and its config, and launches its kernel on
// the stream after the all-reduce Python ran before it.
// =================================================================================================

inline std::variant<Config, Error> rms_norm(
    Handle& h, const void* input, int64_t input_stride_m, int64_t input_stride_n,
    int64_t input_size_m, int64_t input_size_n, const void* weight, int64_t weight_stride_n,
    int64_t weight_size_n, void* output, int64_t output_stride_m, int64_t output_stride_n,
    int64_t output_size_m, int64_t output_size_n, DType dtype, DType weight_dtype, float eps,
    const Chosen& k, hipStream_t stream) {
  if (const std::optional<Error> e =
          layout_refused(dtype, pack_bytes_of(k.config.build), {{output_stride_m, output_stride_n},
                                 {input_stride_m, input_stride_n}}))
    return *e;
  if (const std::optional<Error> e =
          layout_refused(weight_dtype, pack_bytes_of(k.config.build), {{0, weight_stride_n}}))
    return *e;
  std::variant<Config, Error> checked =
      rms_norm_config(h, input_size_m, input_size_n, dtype, weight_dtype, k);
  if (const Error* e = std::get_if<Error>(&checked)) return *e;
  const Config config = std::get<Config>(checked);
  with_config<OpType::rms_norm>(config, [&]<Config C>() {
    using T = typename of_dtype<C.dtype>::t;
    using W = typename of_dtype<C.weight_dtype>::t;
    // BEFORE THE LAUNCH, every expectation of what rms_norm is handed (October 26: n11 rebooted
    // seven times timing the fused two-shots at size; the kernels check their indices only
    // against these sizes, so a size past its region was never caught): each view of a region
    // inside the region, and the grid inside the signals' slots.
    if (volume(config.grid) > h.signals_size_block())
      throw std::runtime_error("hip_custom_communicator rms_norm before its launch: a grid of " +
          std::to_string(volume(config.grid)) + " blocks past the signals' " +
          std::to_string(h.signals_size_block()));
    h.begin_launch(OpType::rms_norm, kernel_pointer<C>(), input_size_m, input_size_n, dtype,
        stream);
    rms_norm_rows<T, W, C.tile_m, C.tile_n, C.block.x, C.wave_size, C.pack_bytes, C.waves_per_eu,
        C.chunked>
        <<<dim3_of(config.grid), dim3_of(C.block), config.shared_memory_bytes, stream>>>(
        static_cast<const T*>(input), input_stride_m, input_stride_n,
        static_cast<int>(input_size_m), static_cast<int>(input_size_n),
        static_cast<const W*>(weight), weight_stride_n, static_cast<int>(weight_size_n),
        static_cast<T*>(output), output_stride_m, output_stride_n,
        static_cast<int>(output_size_m), static_cast<int>(output_size_n), eps);
    // AFTER THE LAUNCH, checked (VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_CHECKED=1) and not captured: the
    // kernel done, every expectation of rms_norm after it: no write past a region (the guards), no
    // fault, every output finite where the input is.
    if (h.checked()) {
      hipStreamCaptureStatus capturing = hipStreamCaptureStatusNone;
      HIP_CHECK(hipStreamIsCapturing(stream, &capturing));
      if (capturing == hipStreamCaptureStatusNone) {
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(stream));
        if (const std::string broken = h.guards_broken(); !broken.empty())
          throw std::runtime_error("hip_custom_communicator rms_norm after its launch: " + broken);
        const auto status = h.status();
        if (status[static_cast<int>(StatusField::state)] != 0 ||
            status[static_cast<int>(StatusField::abort)] != 0)
          throw std::runtime_error("hip_custom_communicator rms_norm after its launch: a fault or an "
                                   "abort in the status");
        // EVERY SIGNAL CLEAR is every rank's promise once every rank finished (a faster peer's next
        // launch sets its first barrier's flags here before this rank looks): the caller's, between
        // two barriers of every rank (the communicator's check_after, the tests' _assert_clean).
        const auto finite = [&](const void* ptr, int64_t stride_m, int64_t rows, int64_t cols) {
          std::vector<T> host(static_cast<size_t>(rows * cols));
          HIP_CHECK(hipMemcpy2D(host.data(), cols * sizeof(T), ptr, stride_m * sizeof(T),
                                cols * sizeof(T), rows, hipMemcpyDeviceToHost));
          for (const T& x : host)
            if (!std::isfinite(static_cast<float>(x))) return false;
          return true;
        };
        if (finite(input, input_stride_m, input_size_m, input_size_n)) {
          if (!finite(output, output_stride_m, output_size_m, output_size_n))
            throw std::runtime_error("hip_custom_communicator rms_norm after its launch: output not "
                                     "finite where the input is");
        }
      }
    }
  });
  HIP_CHECK(hipGetLastError());
  return config;
}

inline std::variant<Config, Error> add_rms_norm(
    Handle& h, const void* input, int64_t input_stride_m, int64_t input_stride_n,
    int64_t input_size_m, int64_t input_size_n, const void* residual, int64_t residual_stride_m,
    int64_t residual_stride_n, int64_t residual_size_m, int64_t residual_size_n,
    const void* weight, int64_t weight_stride_n, int64_t weight_size_n, void* output,
    int64_t output_stride_m, int64_t output_stride_n, int64_t output_size_m,
    int64_t output_size_n, void* residual_output, int64_t residual_output_stride_m,
    int64_t residual_output_stride_n, int64_t residual_output_size_m,
    int64_t residual_output_size_n, DType dtype, DType weight_dtype, float eps,
    const Chosen& k, hipStream_t stream) {
  if (const std::optional<Error> e = layout_refused(
          dtype, pack_bytes_of(k.config.build), {{output_stride_m, output_stride_n},
                  {residual_output_stride_m, residual_output_stride_n},
                  {input_stride_m, input_stride_n},
                  {residual_stride_m, residual_stride_n}}))
    return *e;
  if (const std::optional<Error> e =
          layout_refused(weight_dtype, pack_bytes_of(k.config.build), {{0, weight_stride_n}}))
    return *e;
  std::variant<Config, Error> checked =
      add_rms_norm_config(h, input_size_m, input_size_n, dtype, weight_dtype, k);
  if (const Error* e = std::get_if<Error>(&checked)) return *e;
  const Config config = std::get<Config>(checked);
  with_config<OpType::add_rms_norm>(config, [&]<Config C>() {
    using T = typename of_dtype<C.dtype>::t;
    using W = typename of_dtype<C.weight_dtype>::t;
    // BEFORE THE LAUNCH, every expectation of what add_rms_norm is handed (October 26: n11 rebooted
    // seven times timing the fused two-shots at size; the kernels check their indices only
    // against these sizes, so a size past its region was never caught): each view of a region
    // inside the region, and the grid inside the signals' slots.
    if (volume(config.grid) > h.signals_size_block())
      throw std::runtime_error("hip_custom_communicator add_rms_norm before its launch: a grid of " +
          std::to_string(volume(config.grid)) + " blocks past the signals' " +
          std::to_string(h.signals_size_block()));
    h.begin_launch(OpType::add_rms_norm, kernel_pointer<C>(), input_size_m, input_size_n, dtype,
        stream);
    add_rms_norm_rows<T, W, C.tile_m, C.tile_n, C.block.x, C.wave_size, C.pack_bytes,
        C.waves_per_eu, C.chunked>
        <<<dim3_of(config.grid), dim3_of(C.block), config.shared_memory_bytes, stream>>>(
        static_cast<const T*>(input), input_stride_m, input_stride_n,
        static_cast<int>(input_size_m), static_cast<int>(input_size_n),
        static_cast<const T*>(residual), residual_stride_m, residual_stride_n,
        static_cast<int>(residual_size_m), static_cast<int>(residual_size_n),
        static_cast<const W*>(weight), weight_stride_n, static_cast<int>(weight_size_n),
        static_cast<T*>(output), output_stride_m, output_stride_n,
        static_cast<int>(output_size_m), static_cast<int>(output_size_n),
        static_cast<T*>(residual_output), residual_output_stride_m, residual_output_stride_n,
        static_cast<int>(residual_output_size_m), static_cast<int>(residual_output_size_n), eps);
    // AFTER THE LAUNCH, checked (VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_CHECKED=1) and not captured: the
    // kernel done, every expectation of add_rms_norm after it: no write past a region (the guards), no
    // fault, every output finite where the input is.
    if (h.checked()) {
      hipStreamCaptureStatus capturing = hipStreamCaptureStatusNone;
      HIP_CHECK(hipStreamIsCapturing(stream, &capturing));
      if (capturing == hipStreamCaptureStatusNone) {
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(stream));
        if (const std::string broken = h.guards_broken(); !broken.empty())
          throw std::runtime_error("hip_custom_communicator add_rms_norm after its launch: " + broken);
        const auto status = h.status();
        if (status[static_cast<int>(StatusField::state)] != 0 ||
            status[static_cast<int>(StatusField::abort)] != 0)
          throw std::runtime_error("hip_custom_communicator add_rms_norm after its launch: a fault or an "
                                   "abort in the status");
        // EVERY SIGNAL CLEAR is every rank's promise once every rank finished (a faster peer's next
        // launch sets its first barrier's flags here before this rank looks): the caller's, between
        // two barriers of every rank (the communicator's check_after, the tests' _assert_clean).
        const auto finite = [&](const void* ptr, int64_t stride_m, int64_t rows, int64_t cols) {
          std::vector<T> host(static_cast<size_t>(rows * cols));
          HIP_CHECK(hipMemcpy2D(host.data(), cols * sizeof(T), ptr, stride_m * sizeof(T),
                                cols * sizeof(T), rows, hipMemcpyDeviceToHost));
          for (const T& x : host)
            if (!std::isfinite(static_cast<float>(x))) return false;
          return true;
        };
        if (finite(input, input_stride_m, input_size_m, input_size_n)) {
          if (!finite(output, output_stride_m, output_size_m, output_size_n))
            throw std::runtime_error("hip_custom_communicator add_rms_norm after its launch: output not "
                                     "finite where the input is");
          if (!finite(residual_output, residual_output_stride_m, residual_output_size_m, residual_output_size_n))
            throw std::runtime_error("hip_custom_communicator add_rms_norm after its launch: residual_output not "
                                     "finite where the input is");
        }
      }
    }
  });
  HIP_CHECK(hipGetLastError());
  return config;
}

inline std::variant<Config, Error> add_attn_res_rms_norm(
    Handle& h, const void* input, int64_t input_stride_m, int64_t input_stride_n,
    int64_t input_size_m, int64_t input_size_n, const void* prefix, int64_t prefix_stride_m,
    int64_t prefix_stride_n, int64_t prefix_size_m, int64_t prefix_size_n, const void* blocks,
    int64_t blocks_stride_m, int64_t blocks_stride_r, int64_t blocks_stride_n,
    int64_t blocks_size_m, int64_t blocks_size_r, int64_t blocks_size_n,
    const void* norm_weight, int64_t norm_weight_stride_n, int64_t norm_weight_size_n,
    const void* qk_weight, int64_t qk_weight_stride_n, int64_t qk_weight_size_n,
    const void* output_norm_weight, int64_t output_norm_weight_stride_n,
    int64_t output_norm_weight_size_n, void* prefix_output, int64_t prefix_output_stride_m,
    int64_t prefix_output_stride_n, int64_t prefix_output_size_m, int64_t prefix_output_size_n,
    void* output, int64_t output_stride_m, int64_t output_stride_n, int64_t output_size_m,
    int64_t output_size_n, void* block_output, int64_t block_output_stride_m,
    int64_t block_output_stride_n, int64_t block_output_size_m, int64_t block_output_size_n,
    DType dtype, int num_blocks, float eps, float output_eps, const Chosen& k,
    hipStream_t stream) {
  if (const std::optional<Error> e = layout_refused(
          dtype, pack_bytes_of(k.config.build), {{input_stride_m, input_stride_n},
                  {blocks_stride_m, blocks_stride_n},
                  {blocks_stride_r, blocks_stride_n},
                  {0, norm_weight_stride_n},
                  {0, qk_weight_stride_n},
                  {prefix_output_stride_m, prefix_output_stride_n},
                  {output_stride_m, output_stride_n}}))
    return *e;
  // THE OPTIONAL BUFFERS, checked only when given (all_reduce_add_attn_res_rms_norm's).
  if (prefix != nullptr)
    if (const std::optional<Error> e =
            layout_refused(dtype, pack_bytes_of(k.config.build), {{prefix_stride_m, prefix_stride_n}}))
      return *e;
  if (output_norm_weight != nullptr)
    if (const std::optional<Error> e =
            layout_refused(dtype, pack_bytes_of(k.config.build), {{0, output_norm_weight_stride_n}}))
      return *e;
  if (block_output != nullptr)
    if (const std::optional<Error> e = layout_refused(
            dtype, pack_bytes_of(k.config.build), {{block_output_stride_m, block_output_stride_n}}))
      return *e;
  std::variant<Config, Error> checked =
      add_attn_res_rms_norm_config(h, input_size_m, input_size_n, dtype, num_blocks, k);
  if (const Error* e = std::get_if<Error>(&checked)) return *e;
  const Config config = std::get<Config>(checked);
  with_config<OpType::add_attn_res_rms_norm>(config, [&]<Config C>() {
    using T = typename of_dtype<C.dtype>::t;
    // BEFORE THE LAUNCH, every expectation of what add_attn_res_rms_norm is handed (October 26: n11 rebooted
    // seven times timing the fused two-shots at size; the kernels check their indices only
    // against these sizes, so a size past its region was never caught): each view of a region
    // inside the region, and the grid inside the signals' slots.
    if (volume(config.grid) > h.signals_size_block())
      throw std::runtime_error("hip_custom_communicator add_attn_res_rms_norm before its launch: a grid of " +
          std::to_string(volume(config.grid)) + " blocks past the signals' " +
          std::to_string(h.signals_size_block()));
    h.begin_launch(OpType::add_attn_res_rms_norm, kernel_pointer<C>(), input_size_m,
        input_size_n, dtype, stream);
    add_attn_res_rms_norm_rows<T, C.tile_m, C.tile_n, C.tile_k, C.block.x, C.wave_size,
        C.pack_bytes, C.waves_per_eu, C.chunked>
        <<<dim3_of(config.grid), dim3_of(C.block), config.shared_memory_bytes, stream>>>(
        static_cast<const T*>(input), input_stride_m, input_stride_n,
        static_cast<int>(input_size_m), static_cast<int>(input_size_n),
        static_cast<const T*>(prefix), prefix_stride_m, prefix_stride_n,
        static_cast<int>(prefix_size_m), static_cast<int>(prefix_size_n),
        static_cast<const T*>(blocks), blocks_stride_m, blocks_stride_r, blocks_stride_n,
        static_cast<int>(blocks_size_m), static_cast<int>(blocks_size_r),
        static_cast<int>(blocks_size_n), static_cast<const T*>(norm_weight),
        norm_weight_stride_n, static_cast<int>(norm_weight_size_n),
        static_cast<const T*>(qk_weight), qk_weight_stride_n, static_cast<int>(qk_weight_size_n),
        static_cast<const T*>(output_norm_weight), output_norm_weight_stride_n,
        static_cast<int>(output_norm_weight_size_n), static_cast<T*>(prefix_output),
        prefix_output_stride_m, prefix_output_stride_n, static_cast<int>(prefix_output_size_m),
        static_cast<int>(prefix_output_size_n), static_cast<T*>(output), output_stride_m,
        output_stride_n, static_cast<int>(output_size_m), static_cast<int>(output_size_n),
        static_cast<T*>(block_output), block_output_stride_m, block_output_stride_n,
        static_cast<int>(block_output_size_m), static_cast<int>(block_output_size_n), num_blocks,
        eps, output_eps);
    // AFTER THE LAUNCH, checked (VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_CHECKED=1) and not captured: the
    // kernel done, every expectation of add_attn_res_rms_norm after it: no write past a region (the guards), no
    // fault, every output finite where the input is.
    if (h.checked()) {
      hipStreamCaptureStatus capturing = hipStreamCaptureStatusNone;
      HIP_CHECK(hipStreamIsCapturing(stream, &capturing));
      if (capturing == hipStreamCaptureStatusNone) {
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(stream));
        if (const std::string broken = h.guards_broken(); !broken.empty())
          throw std::runtime_error("hip_custom_communicator add_attn_res_rms_norm after its launch: " + broken);
        const auto status = h.status();
        if (status[static_cast<int>(StatusField::state)] != 0 ||
            status[static_cast<int>(StatusField::abort)] != 0)
          throw std::runtime_error("hip_custom_communicator add_attn_res_rms_norm after its launch: a fault or an "
                                   "abort in the status");
        // EVERY SIGNAL CLEAR is every rank's promise once every rank finished (a faster peer's next
        // launch sets its first barrier's flags here before this rank looks): the caller's, between
        // two barriers of every rank (the communicator's check_after, the tests' _assert_clean).
        const auto finite = [&](const void* ptr, int64_t stride_m, int64_t rows, int64_t cols) {
          std::vector<T> host(static_cast<size_t>(rows * cols));
          HIP_CHECK(hipMemcpy2D(host.data(), cols * sizeof(T), ptr, stride_m * sizeof(T),
                                cols * sizeof(T), rows, hipMemcpyDeviceToHost));
          for (const T& x : host)
            if (!std::isfinite(static_cast<float>(x))) return false;
          return true;
        };
        if (finite(input, input_stride_m, input_size_m, input_size_n)) {
          if (!finite(output, output_stride_m, output_size_m, output_size_n))
            throw std::runtime_error("hip_custom_communicator add_attn_res_rms_norm after its launch: output not "
                                     "finite where the input is");
        }
      }
    }
  });
  HIP_CHECK(hipGetLastError());
  return config;
}

}  // namespace hip_custom_communicator
