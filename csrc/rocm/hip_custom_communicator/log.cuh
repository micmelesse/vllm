// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
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

#pragma once

#include <hip/hip_runtime.h>
#include <time.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/common.cuh"

namespace hip_custom_communicator {

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
};
static_assert(kStatusFields <= 16, "the status words end before the header");
constexpr int kLogHeaderEnd = 32;
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
  copy_in_issued = 5,  // retired (the kernels copy their input into the
                       // staging themselves); kept so the numbers hold
  stream_seen = 6,     // peer: the stream's index; seen, want: its handle's
                       // low and high words
  kernel_seen = 7,     // kernel: its index; seen, want: its address's words
  deadline_expired = 8,  // reserved: a host deadline (what: extra)
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
  signals_allocated,   // seen, want: bytes requested, stride
  exchange_allocated,  // seen, want: bytes requested (MiB), stride (MiB)
  zeroed,              // this rank's signals read back zero
  exported,
  socket_exchange_begun,  // seen: peers expected
  socket_exchange_done,   // seen: peers that sent
  imported,               // seen: peers imported, a buffer each
  mapped,
  access_set,
  tables_written,  // retired (no peer tables); kept so the numbers hold
  checked,  // every OpenCheck passed; seen: warnings
  open_done,
};

// OPEN'S INTERNAL ASSERTS: each a definite error when it fails, and open
// fails naming it. Never a collective: each reads this process's own state.
enum class OpenCheck : uint8_t {
  none = 0,
  current_device,   // the current device is the one the communicator was
                    // given (seen: current, want: given)
  peer_imported,    // every peer's descriptor imported and mapped
  access_rw,        // hipMemGetAccess from this device: read-write, every slice
  geometry,         // stride a granule multiple, at least the bytes asked;
                    // base granule-aligned
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
  granularity_mismatch,  // peer; seen: its granularity, want: ours
  low_memory,            // seen: free MiB, want: MiB open allocates
  pointer_device,        // retired: hipPointerGetAttributes on an imported
                         // VMM slice reports the importing device, on every
                         // slice, so it was noise; kept so the numbering holds
  peer_not_visible,      // peer: its GPU (by PCI bus id) is not visible here,
                         // so its device checks are skipped
};

constexpr const char* to_string(OpenCheck c) {
  switch (c) {
    case OpenCheck::none:
      return "none";
    case OpenCheck::current_device:
      return "current_device";
    case OpenCheck::peer_imported:
      return "peer_imported";
    case OpenCheck::access_rw:
      return "access_rw";
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
