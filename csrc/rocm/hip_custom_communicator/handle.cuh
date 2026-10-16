// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE RUNTIME STATE, one `Handle` per communicator: everything the running
// program learns, as common.cuh's `BuildInfo` is everything fixed at
// compile time. OUR OWN MEMORY (the signals and the exchange) is HIP virtual
// memory: each rank creates its physical memory with explicit properties,
// exports it as a POSIX file descriptor, and every rank maps every rank's into
// ONE reserved virtual range, rank r's at base + r * stride, so a kernel
// reaches a peer's buffer by a pointer and a rank stride. THE CALLER'S INPUT
// is never read by a peer where it is: every kernel copies this rank's input
// into this rank's staging itself (INT4: quantized into the staging values),
// before its first barrier, which is ordered; the peers then read the staging
// at ptr + r * stride. One path, eager and captured alike: no host copy, no
// pointer table, nothing registered or opened over IPC. The collective
// that exchanges the socket names is the caller's (`Gather`), so this names no
// process group.

#pragma once

#include <c10/util/BFloat16.h>
#include <hip/hip_runtime.h>
#include <dirent.h>
#include <poll.h>
#include <iomanip>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <array>
#include <fstream>
#include <memory>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <random>
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

// THE SIGNALS' BYTES, their own allocation: two flag sets of kMaxBlocks blocks
// by kMaxRanks writers, rounded to 128 bytes.
constexpr int64_t kSignalsBytes =
    (2 * int64_t{kMaxBlocks} * kMaxRanks * int64_t{sizeof(uint32_t)} + 127) /
    128 * 128;

// HOW EACH SYMMETRIC ALLOCATION IS BACKED, stated here and not left to how a
// runtime backs a malloc flag (hipDeviceMallocUncached was fine-grained under
// ROCm 7.2 and, at 2 MiB and up, coarse-grained under HIP 7.15: aiter 0.1.24's
// custom_all_reduce.py). hipMemAllocationTypeUncached: the memory is mapped
// uncached on EVERY GPU that maps it, this rank's and its peers', so no L2
// holds a line of it: a store is in memory (or the memory-side infinity cache,
// which every agent sees alike) once it has completed, and a load reads memory.
//   SIGNALS: uncached. A flag is written by a peer over the fabric and polled
//   by its owner; an L2 copy of it in the owner would never see the peer's
//   store (an agent's L2 is not coherent with a remote agent's writes to its
//   memory, and a device-scope acquire invalidates only non-local lines).
//   EXCHANGE: uncached, for the same reason: a push kernel stores into its
//   peers' scratch and then reads its own; a pull kernel reads a peer's scratch
//   the peer wrote in its own L2's place. Coarse-grained (cacheable) memory
//   would need every reader's L2 invalidated at the barrier and every writer's
//   written back, which the barrier's device-scope acquire does not do for
//   local lines (common.cuh's BARRIER). On HIP 7.15 a
//   hipDeviceMallocFinegrained exchange was 4-10% slower than the uncached flag
//   at 16-256 tokens and the same at 1024-4096 (2026-10-16T15-36-54Z); this
//   choice is measured against both on n11.
constexpr hipMemAllocationType kSignalsAllocation =
    hipMemAllocationTypeUncached;
constexpr hipMemAllocationType kExchangeAllocation =
    hipMemAllocationTypeUncached;

// A FILE DESCRIPTOR WE OWN, closed when it goes.
class Fd {
  int fd_ = -1;

 public:
  Fd() = default;
  explicit Fd(int fd) : fd_(fd) {}
  Fd(Fd&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }
  Fd& operator=(Fd&& o) noexcept {
    if (this != &o) {
      reset();
      fd_ = o.fd_;
      o.fd_ = -1;
    }
    return *this;
  }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  ~Fd() { reset(); }
  int get() const { return fd_; }
  void reset() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
  }
};

[[noreturn]] inline void sys_fail(const std::string& what) {
  throw std::runtime_error("hip_custom_communicator: " + what + ": " +
                           std::strerror(errno));
}

// WHAT EACH RANK SAYS ABOUT ITSELF, gathered with the socket names (the one
// gather open already makes): what open compares across ranks (warnings) and
// how a peer's GPU is found in this process (its PCI bus id). Fixed width, as
// the gather moves equal lengths.
struct RankInfo {
  char socket[64];
  char pci_bus_id[32];
  char host[64];
  int32_t pid;
  int32_t device;
  int64_t granularity;
};

// EVERY RANK'S FILE DESCRIPTORS, `mine` sent to every peer and theirs received,
// over Unix sockets (a descriptor crosses processes only as SCM_RIGHTS), a
// collective: every rank calls it together. Each rank listens on an abstract
// socket (no file to clean up; the ranks share a network namespace, as they
// share the GPUs) named by its pid and a random word, the names gathered over
// `gather`; then each connects to every peer and sends its rank and its
// descriptors in one message, and accepts one connection from every peer. No
// deadlock: a connection to a listening socket completes and its message is
// buffered before the peer accepts it (the backlog holds every peer).
// `theirs[r]` is rank r's, `mine` again at this rank. A peer that has not sent
// within `timeout_seconds` is an error naming how many did.
// `me` goes round with the names; `all[r]` is rank r's, `me` at this rank.
inline std::vector<std::vector<Fd>> exchange_fds(
    int rank, int world, const Gather& gather, const std::vector<int>& mine,
    double timeout_seconds, RankInfo me, Log& log, std::vector<RankInfo>& all) {
  const size_t n = mine.size();
  Fd listener(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
  if (listener.get() < 0) sys_fail("socket");
  std::random_device entropy;
  std::ostringstream os;
  // FIXED WIDTH: the gather moves a byte string the same length on every rank
  // (all_gathered), and pids and entropy vary in digits; a ragged name breaks
  // the CPU group's allgather (gloo: "21 vs 20", f9a8dc6f00 on n11).
  os << "hip_custom_communicator/" << std::setfill('0') << std::setw(10)
     << ::getpid() << "/" << std::setw(10) << entropy() << "/" << std::setw(5)
     << rank;
  const std::string name = os.str();
  if (name.size() + 1 > sizeof(me.socket))
    throw std::runtime_error("hip_custom_communicator: socket name too long");
  std::memset(me.socket, 0, sizeof(me.socket));
  std::memcpy(me.socket, name.data(), name.size());
  const auto address = [](const std::string& s, sockaddr_un& a) {
    std::memset(&a, 0, sizeof(a));
    a.sun_family = AF_UNIX;
    if (s.size() + 1 > sizeof(a.sun_path))
      throw std::runtime_error("hip_custom_communicator: socket name too long");
    std::memcpy(a.sun_path + 1, s.data(),
                s.size());  // sun_path[0] = 0: abstract
    return static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + 1 +
                                  s.size());
  };
  sockaddr_un a;
  socklen_t len = address(name, a);
  if (::bind(listener.get(), reinterpret_cast<sockaddr*>(&a), len) != 0)
    sys_fail("bind " + name);
  if (::listen(listener.get(), world) != 0) sys_fail("listen");
  log.phase(OpenPhase::socket_exchange_begun, static_cast<uint32_t>(world - 1));
  const std::vector<std::string> gathered =
      gather(std::string(reinterpret_cast<const char*>(&me), sizeof(me)));
  all.assign(world, RankInfo{});
  std::vector<std::string> names(world);
  for (int p = 0; p < world; ++p) {
    if (gathered[p].size() != sizeof(RankInfo))
      throw std::runtime_error(
          "hip_custom_communicator: a gathered rank record of the wrong size");
    std::memcpy(&all[p], gathered[p].data(), sizeof(RankInfo));
    all[p].socket[sizeof(all[p].socket) - 1] = 0;
    names[p] = all[p].socket;
  }
  // A rank's message: its rank, then its descriptors as SCM_RIGHTS.
  for (int p = 0; p < world; ++p) {
    if (p == rank) continue;
    Fd c(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
    if (c.get() < 0) sys_fail("socket");
    len = address(names[p], a);
    if (::connect(c.get(), reinterpret_cast<sockaddr*>(&a), len) != 0)
      sys_fail("connect to rank " + std::to_string(p));
    int32_t payload = rank;
    iovec io{&payload, sizeof(payload)};
    std::vector<char> control(CMSG_SPACE(n * sizeof(int)), 0);
    msghdr m{};
    m.msg_iov = &io;
    m.msg_iovlen = 1;
    m.msg_control = control.data();
    m.msg_controllen = control.size();
    cmsghdr* c0 = CMSG_FIRSTHDR(&m);
    c0->cmsg_level = SOL_SOCKET;
    c0->cmsg_type = SCM_RIGHTS;
    c0->cmsg_len = CMSG_LEN(n * sizeof(int));
    std::memcpy(CMSG_DATA(c0), mine.data(), n * sizeof(int));
    if (::sendmsg(c.get(), &m, MSG_NOSIGNAL) != sizeof(payload))
      sys_fail("send to rank " + std::to_string(p));
  }
  std::vector<std::vector<Fd>> theirs(world);
  for (int got = 0; got < world - 1; ++got) {
    pollfd pf{listener.get(), POLLIN, 0};
    const int ready = ::poll(&pf, 1, static_cast<int>(timeout_seconds * 1000));
    if (ready <= 0) {
      if (ready == 0) {
        log.phase(OpenPhase::socket_exchange_done, static_cast<uint32_t>(got),
                  static_cast<uint32_t>(world - 1), 1);
        throw std::runtime_error(
            "hip_custom_communicator: " + std::to_string(got) + " of " +
            std::to_string(world - 1) +
            " peers sent their memory's descriptors before the timeout");
      }
      sys_fail("poll");
    }
    Fd c(::accept4(listener.get(), nullptr, nullptr, SOCK_CLOEXEC));
    if (c.get() < 0) sys_fail("accept");
    int32_t payload = -1;
    iovec io{&payload, sizeof(payload)};
    std::vector<char> control(CMSG_SPACE(n * sizeof(int)), 0);
    msghdr m{};
    m.msg_iov = &io;
    m.msg_iovlen = 1;
    m.msg_control = control.data();
    m.msg_controllen = control.size();
    if (::recvmsg(c.get(), &m, MSG_CMSG_CLOEXEC) != sizeof(payload))
      sys_fail("receive");
    cmsghdr* c0 = CMSG_FIRSTHDR(&m);
    if (payload < 0 || payload >= world || payload == rank ||
        !theirs[payload].empty() || c0 == nullptr ||
        c0->cmsg_type != SCM_RIGHTS ||
        c0->cmsg_len != CMSG_LEN(n * sizeof(int)))
      throw std::runtime_error(
          "hip_custom_communicator: a malformed descriptor message (rank " +
          std::to_string(payload) + ")");
    std::vector<int> fds(n);
    std::memcpy(fds.data(), CMSG_DATA(c0), n * sizeof(int));
    for (int fd : fds) theirs[payload].emplace_back(fd);
  }
  log.phase(OpenPhase::socket_exchange_done, static_cast<uint32_t>(world - 1),
            static_cast<uint32_t>(world - 1));
  return theirs;
}

// ONE SYMMETRIC ALLOCATION: `bytes` of this rank's device memory, created with
// `type` (kSignalsAllocation, kExchangeAllocation), and every rank's mapped
// into one reserved virtual range of this process, rank r's at
// base() + r * stride_bytes(), every local GPU given read-write access to the
// whole range. The stride is `bytes` rounded up to the allocation granularity
// (the largest of the minimum and recommended: 4 KiB on ROCm 7.2, 2 MiB on HIP
// 7.15), so every rank's buffer starts at the same alignment. Built in two
// steps around the descriptor exchange (`exchange_fds`), the Handle's: `create`
// makes and zeroes this rank's (before any peer can see it) and exports it;
// `map_peers` maps every peer's.
class SymmetricMemory {
 public:
  SymmetricMemory(int rank, int world, int64_t bytes, hipMemAllocationType type)
      : rank_(rank), world_(world), bytes_(bytes), peers_(world) {
    HIP_CHECK(hipGetDevice(&device_));
    hipMemAllocationProp prop{};
    prop.type = type;
    prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device_;
    size_t minimum = 0, recommended = 0;
    HIP_CHECK(hipMemGetAllocationGranularity(
        &minimum, &prop, hipMemAllocationGranularityMinimum));
    HIP_CHECK(hipMemGetAllocationGranularity(
        &recommended, &prop, hipMemAllocationGranularityRecommended));
    granularity_ = static_cast<int64_t>(std::max(minimum, recommended));
    stride_ = (bytes_ + granularity_ - 1) / granularity_ * granularity_;
    HIP_CHECK(
        hipMemCreate(&peers_[rank_], static_cast<size_t>(stride_), &prop, 0));
    HIP_CHECK(
        hipMemAddressReserve(&base_, static_cast<size_t>(stride_ * world_),
                             static_cast<size_t>(granularity_), nullptr, 0));
    map(rank_);
    allow_access(rank_);
    HIP_CHECK(hipMemset(at(rank_), 0, static_cast<size_t>(stride_)));
    HIP_CHECK(hipDeviceSynchronize());
  }

  // This rank's memory as a descriptor its peers import (they own their copy).
  Fd exported() const {
    int fd = -1;
    HIP_CHECK(hipMemExportToShareableHandle(
        &fd, peers_[rank_], hipMemHandleTypePosixFileDescriptor, 0));
    return Fd(fd);
  }

  // Every peer's memory, imported from its descriptor (`fds[r]`, rank r's),
  // then mapped at its place, then made accessible: three steps, so open's log
  // says which one it reached.
  void import_peers(const std::vector<int>& fds) {
    for (int r = 0; r < world_; ++r) {
      if (r == rank_) continue;
      HIP_CHECK(hipMemImportFromShareableHandle(
          &peers_[r], reinterpret_cast<void*>(static_cast<intptr_t>(fds[r])),
          hipMemHandleTypePosixFileDescriptor));
    }
  }
  void map_peers() {
    for (int r = 0; r < world_; ++r)
      if (r != rank_) map(r);
  }
  void allow_peers() {
    for (int r = 0; r < world_; ++r)
      if (r != rank_) allow_access(r);
  }

  // OPEN'S ASSERTS ON THIS ALLOCATION, each logged (Log::check): every
  // peer's imported and mapped, every slice read-write from this device, the
  // geometry what was asked. False when one failed (the log names it).
  // `peer_devices[r]`: rank r's GPU in this process, -1 if not visible; a
  // slice whose hipPointerGetAttributes names another device is a warning,
  // not a failure (what the runtime reports for an imported VMM slice is not
  // established).
  bool verify(Log& log, const std::vector<int>& peer_devices) const {
    bool ok = true;
    const bool geometry =
        granularity_ > 0 && stride_ % granularity_ == 0 && stride_ >= bytes_ &&
        reinterpret_cast<uintptr_t>(base_) % granularity_ == 0;
    log.check(OpenCheck::geometry, geometry, rank_,
              static_cast<uint32_t>(stride_ / 1024),
              static_cast<uint32_t>(granularity_ / 1024));
    ok = ok && geometry;
    for (int r = 0; r < world_; ++r) {
      const bool present = peers_[r] != nullptr &&
                           mapped_.size() == size_t(world_) && mapped_[r];
      log.check(OpenCheck::peer_imported, present, r);
      ok = ok && present;
      if (!present) continue;
      hipMemLocation here{};
      here.type = hipMemLocationTypeDevice;
      here.id = device_;
      unsigned long long flags = 0;
      const hipError_t e = hipMemGetAccess(&flags, &here, at(r));
      const bool rw =
          e == hipSuccess && flags == hipMemAccessFlagsProtReadWrite;
      log.check(OpenCheck::access_rw, rw, r, static_cast<uint32_t>(flags),
                static_cast<uint32_t>(e));
      ok = ok && rw;
      hipPointerAttribute_t attrs{};
      if (hipPointerGetAttributes(&attrs, at(r)) == hipSuccess) {
        if (peer_devices[r] >= 0 && attrs.device != peer_devices[r])
          log.warn(OpenWarning::pointer_device, r,
                   static_cast<uint32_t>(attrs.device),
                   static_cast<uint32_t>(peer_devices[r]));
      } else {
        (void)hipGetLastError();
        log.warn(OpenWarning::pointer_device, r, 0xFFFFFFFFu,
                 static_cast<uint32_t>(peer_devices[r]));
      }
    }
    return ok;
  }

  ~SymmetricMemory() {
    for (int r = 0; r < world_; ++r) {
      if (mapped_.size() > static_cast<size_t>(r) && mapped_[r])
        (void)hipMemUnmap(at(r), static_cast<size_t>(stride_));
      if (peers_[r] != nullptr) (void)hipMemRelease(peers_[r]);
    }
    if (base_ != nullptr)
      (void)hipMemAddressFree(base_, static_cast<size_t>(stride_ * world_));
  }
  SymmetricMemory(const SymmetricMemory&) = delete;
  SymmetricMemory& operator=(const SymmetricMemory&) = delete;

  // Rank 0's buffer: rank r's is stride_bytes() * r past it.
  char* base() const { return static_cast<char*>(base_); }
  int64_t stride_bytes() const { return stride_; }
  int64_t bytes() const { return bytes_; }
  int64_t granularity() const { return granularity_; }
  char* at(int r) const { return base() + int64_t{r} * stride_; }

 private:
  void map(int r) {
    HIP_CHECK(hipMemMap(at(r), static_cast<size_t>(stride_), 0, peers_[r], 0));
    if (mapped_.empty()) mapped_.assign(world_, false);
    mapped_[r] = true;
  }
  // EVERY LOCAL GPU may read and write rank r's buffer through this range: the
  // kernels run on this rank's, and a mapping no device may access faults.
  void allow_access(int r) {
    int devices = 0;
    HIP_CHECK(hipGetDeviceCount(&devices));
    std::vector<hipMemAccessDesc> access(devices);
    for (int d = 0; d < devices; ++d) {
      access[d].location.type = hipMemLocationTypeDevice;
      access[d].location.id = d;
      access[d].flags = hipMemAccessFlagsProtReadWrite;
    }
    HIP_CHECK(hipMemSetAccess(at(r), static_cast<size_t>(stride_),
                              access.data(), access.size()));
  }

  int rank_;
  int world_;
  int device_ = 0;
  int64_t bytes_;
  int64_t granularity_ = 0;
  int64_t stride_ = 0;
  void* base_ = nullptr;
  std::vector<hipMemGenericAllocationHandle_t> peers_;
  std::vector<bool> mapped_;
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

class Handle {
 public:
  // OPENED on every rank together: this rank's symmetric memory allocated, two
  // allocations, the signals and the exchange (the scratch, then the staging),
  // each one's IPC handle gathered with every rank's, and the peers' opened.
  // `world_size` is one the build holds (the opener checks it).
  // `timeout_seconds`: how long a kernel waits on a peer before it records a
  // fault and leaves (common.cuh's BARRIER), the caller's: the process group's
  // own timeout, as every other collective on it has. THE EXCHANGE'S SIZES are
  // the caller's too, computed from the largest call it serves, `max_rows` rows
  // of at most `max_cols` columns (select.cuh's `exchange_sizes`): kept, so
  // what the buffers were sized for can be asked. `quantized_values_bytes` and
  // `quantized_scales_bytes`: one quantized buffer's values and scales at that
  // box (common.cuh's QUANTIZATION). `quantization`: what the tuned path moves
  // across the links, a Quantization (select.cuh's `quantizes`).
  // `device`: the device the communicator was given (open checks it is the
  // current one). `log_mode`: what the log records (log.cuh). `rccl_ready`:
  // whether the caller found the device group's RCCL communicator set up (1),
  // not (0), or could not tell (-1): observed and logged, never acted on.
  Handle(int rank, int world_size, int device, const Gather& gather,
         double timeout_seconds, int64_t max_rows, int64_t max_cols,
         int64_t scratch_bytes, int64_t quantized_values_bytes,
         int64_t quantized_scales_bytes, int64_t staging_bytes,
         int quantization, LogMode log_mode, int rccl_ready)
      : rank_(rank),
        world_size_(world_size),
        device_(device),
        max_rows_(max_rows),
        max_cols_(max_cols),
        scratch_bytes_(scratch_bytes),
        quantized_values_bytes_(quantized_values_bytes),
        quantized_scales_bytes_(quantized_scales_bytes),
        staging_bytes_(staging_bytes),
        quantization_(quantization),
        log_(std::make_shared<Log>(log_mode, rank, world_size, device)) {
    // THE LOG FIRST, and kept as the process's last (log.cuh's `last_log`):
    // an open that fails or hangs is read there.
    set_last_log(log_);
    try {
      open(gather, timeout_seconds, rccl_ready);
    } catch (...) {
      // THE PHASE AFTER THE LAST REACHED is the one that failed.
      const uint32_t reached = log_->get(LogHeader::open_phase);
      if (reached < static_cast<uint32_t>(OpenPhase::open_done))
        log_->phase(static_cast<OpenPhase>(reached + 1), 0, 0, 1);
      throw;
    }
  }

  // The symmetric memory unmaps and releases itself (SymmetricMemory), after
  // every launch on it has finished.
  ~Handle() { (void)hipDeviceSynchronize(); }

  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;

  int rank() const { return rank_; }
  int world_size() const { return world_size_; }
  int64_t max_rows() const { return max_rows_; }
  int64_t max_cols() const { return max_cols_; }
  int64_t scratch_bytes() const { return scratch_bytes_; }
  // One quantized buffer's values and scales, the most each region holds.
  int64_t quantized_values_bytes() const { return quantized_values_bytes_; }
  int64_t quantized_scales_bytes() const { return quantized_scales_bytes_; }
  // Where every kernel copies its input for its peers to read, and how many
  // bytes it holds.
  void* staging() const { return exchange_->at(rank_) + staging_offset(); }
  int64_t staging_bytes() const { return staging_bytes_; }
  // What the tuned path moves across the links, a Quantization.
  int quantization() const { return quantization_; }

  // EVERY RANK'S SCRATCH AND STAGING, and the quantized forms of the scratch
  // and of the staged input, fixed for the handle's life: each region is rank
  // 0's address, rank r's exchange_stride_rank_bytes() * r past it (a kernel
  // takes the stride in its buffer's elements: the launch divides).
  void* scratch_ptr() const { return exchange_->base(); }
  void* scratch_values_ptr() const {
    return exchange_->base() + scratch_bytes_;
  }
  void* scratch_scales_ptr() const {
    return exchange_->base() + scratch_bytes_ + quantized_values_bytes_;
  }
  void* staging_values_ptr() const {
    return exchange_->base() + scratch_bytes_ + quantized_values_bytes_ +
           quantized_scales_bytes_;
  }
  void* staging_scales_ptr() const {
    return exchange_->base() + scratch_bytes_ + 2 * quantized_values_bytes_ +
           quantized_scales_bytes_;
  }
  void* staging_ptr() const { return exchange_->base() + staging_offset(); }
  int64_t exchange_stride_rank_bytes() const {
    return exchange_->stride_bytes();
  }
  // The synchronization state: every rank's signal block and how long a wait
  // may last before it gives up. Every rank's signals, [R, 2, kMaxBlocks,
  // kMaxRanks] (owner, parity, block, writer): rank 0's address, then the
  // strides and the inner sizes, in elements (R is the kernel's WORLD).
  uint32_t* signals_ptr() const {
    return reinterpret_cast<uint32_t*>(signals_->base());
  }
  int64_t signals_stride_rank() const {
    return signals_->stride_bytes() / static_cast<int64_t>(sizeof(uint32_t));
  }
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
    HIP_CHECK(hipMemcpy(flags.data(), signals_->at(rank_),
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

  // Every wait on this rank leaves at its next check: for a communicator the
  // host has given up on.
  void abort() {
    __atomic_store_n(&status_[static_cast<int>(StatusField::abort)],
                     uint32_t{1}, __ATOMIC_RELEASE);
  }

  // What a compiled kernel uses, read once from the code object loaded on this
  // device: only the compiler knows it.
  Resources resources_of(const void* kernel) const {
    if (const auto it = resources_.find(kernel); it != resources_.end())
      return it->second;
    hipFuncAttributes attrs{};
    HIP_CHECK(hipFuncGetAttributes(&attrs, kernel));
    return resources_[kernel] = {attrs.numRegs,
                                 static_cast<int64_t>(attrs.sharedSizeBytes)};
  }

 private:
  // OPEN, every phase logged (log.cuh's OpenPhase), each internal assert a
  // logged OpenCheck that fails open naming it, each observation of the
  // broader state a logged OpenWarning. No collective beyond the CPU group's
  // gather the descriptor exchange already makes.
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
      const int64_t needed = kSignalsBytes + staging_offset() + staging_bytes_;
      if (static_cast<int64_t>(free) < needed)
        log.warn(OpenWarning::low_memory, 0, static_cast<uint32_t>(free >> 20),
                 static_cast<uint32_t>(needed >> 20));
    }
    // THE SIGNALS APART FROM THE DATA: an overrun of the scratch or the staging
    // runs off its own allocation, never into barrier state. Each made and
    // zeroed here, then every rank's descriptors exchanged (one collective) and
    // every peer's mapped beside this rank's.
    signals_.emplace(rank_, world_size_, kSignalsBytes, kSignalsAllocation);
    log.phase(OpenPhase::signals_allocated,
              static_cast<uint32_t>(kSignalsBytes),
              static_cast<uint32_t>(signals_->stride_bytes()));
    exchange_.emplace(rank_, world_size_, staging_offset() + staging_bytes_,
                      kExchangeAllocation);
    log.phase(OpenPhase::exchange_allocated,
              static_cast<uint32_t>(exchange_->bytes() >> 20),
              static_cast<uint32_t>(exchange_->stride_bytes() >> 20));
    {
      // THIS RANK'S SIGNALS READ ZERO after the memset, before any peer can
      // write them: a few words, host-side.
      std::array<uint32_t, 64> words{};
      HIP_CHECK(hipMemcpy(words.data(), signals_->at(rank_), sizeof(words),
                          hipMemcpyDeviceToHost));
      uint32_t nonzero = 0;
      for (uint32_t w : words) nonzero += w != 0;
      log.check(OpenCheck::signals_zero, nonzero == 0, rank_, nonzero, 0);
      if (nonzero != 0)
        fail(OpenCheck::signals_zero, std::to_string(nonzero) +
                                          " of the first 64 signal words are "
                                          "not zero after the memset");
      log.phase(OpenPhase::zeroed);
    }
    std::vector<RankInfo> ranks;
    {
      const Fd signals_fd = signals_->exported();
      const Fd exchange_fd = exchange_->exported();
      log.phase(OpenPhase::exported);
      RankInfo me{};
      HIP_CHECK(
          hipDeviceGetPCIBusId(me.pci_bus_id, sizeof(me.pci_bus_id), device_));
      (void)::gethostname(me.host, sizeof(me.host) - 1);
      me.pid = static_cast<int32_t>(::getpid());
      me.device = device_;
      me.granularity = exchange_->granularity();
      const std::vector<std::vector<Fd>> theirs = exchange_fds(
          rank_, world_size_, gather, {signals_fd.get(), exchange_fd.get()},
          timeout_seconds, me, log, ranks);
      observe_peers(ranks);
      std::vector<int> signals_fds(world_size_, -1),
          exchange_fds_of(world_size_, -1);
      for (int r = 0; r < world_size_; ++r) {
        if (r == rank_) continue;
        signals_fds[r] = theirs[r][0].get();
        exchange_fds_of[r] = theirs[r][1].get();
      }
      signals_->import_peers(signals_fds);
      exchange_->import_peers(exchange_fds_of);
      log.phase(OpenPhase::imported, static_cast<uint32_t>(world_size_ - 1), 2);
    }
    signals_->map_peers();
    exchange_->map_peers();
    log.phase(OpenPhase::mapped);
    signals_->allow_peers();
    exchange_->allow_peers();
    log.phase(OpenPhase::access_set);
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
    if (!signals_->verify(log, peer_devices))
      fail(static_cast<OpenCheck>(log.get(LogHeader::failed)),
           "the signals' mapping");
    if (!exchange_->verify(log, peer_devices))
      fail(static_cast<OpenCheck>(log.get(LogHeader::failed)),
           "the exchange's mapping");
    // THE STATUS, host-mapped and coherent: the log's first words
    // (log.cuh). A wait that gives up writes its fault with system-scope
    // atomics while the host polls it, and the host sets its `abort`
    // (common.cuh's BARRIER). The kernels take its device address as their
    // own argument, a buffer of kStatusFields words.
    status_ = log.status_host();
    device_status_ = log.status_device();
    // EACH RANK'S EXCHANGE is [ scratch | scratch values | scratch scales |
    // staging values | staging scales | staging ], at the same offsets on every
    // rank, so each region is rank 0's address and the exchange's rank stride
    // (scratch_ptr(), exchange_stride_rank_bytes()). The STAGED INPUT'S LIES
    // PAST EVERY KERNEL'S SCRATCH, since an INT4 two-shot writes it before its
    // first barrier, while a peer may still read this rank's scratch after its
    // last. Every kernel copies its input into its staging (or, INT4, its
    // staging values and scales) before its first barrier, which is ordered,
    // and its peers read it there, at rank 0's address and the rank stride:
    // no table.
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

  // WHAT THE RANKS SAID ABOUT THEMSELVES, compared: two ranks on one GPU,
  // a rank on another host, granularities that differ. Warnings only.
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
      if (p.granularity != me.granularity)
        log_->warn(OpenWarning::granularity_mismatch, r,
                   static_cast<uint32_t>(p.granularity),
                   static_cast<uint32_t>(me.granularity));
    }
  }

  // OPEN FAILS, naming the check (already logged) and the phase reached.
  [[noreturn]] void fail(OpenCheck c, const std::string& why) const {
    throw std::runtime_error(
        std::string("hip_custom_communicator: open check failed: ") +
        to_string(c) + ": " + why + " (rank " + std::to_string(rank_) + ")");
  }

  // The staging's offset in the exchange: past the scratch and the quantized
  // regions.
  int64_t staging_offset() const {
    return scratch_bytes_ +
           2 * (quantized_values_bytes_ + quantized_scales_bytes_);
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
  int64_t scratch_bytes_;
  int64_t quantized_values_bytes_;
  int64_t quantized_scales_bytes_;
  int64_t staging_bytes_;
  int quantization_;
  std::shared_ptr<Log> log_;
  // KERNELS AND STREAMS BY INDEX, as the log names them (kernel_seen,
  // stream_seen): first seen, first numbered.
  std::vector<const void*> kernels_seen_;
  std::vector<hipStream_t> streams_seen_;
  std::optional<SymmetricMemory> signals_;
  std::optional<SymmetricMemory> exchange_;
  uint32_t* status_ = nullptr;
  uint32_t* device_status_ = nullptr;
  uint64_t timeout_ticks_ = 0;
  mutable std::unordered_map<const void*, Resources> resources_;
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
                  all_reduce_pull_one_shot_staged<
                      c10::BFloat16, 2, 1, kWaveSize * traits<c10::BFloat16>::N,
                      kWaveSize, 1>));
  (void)hipGetLastError();
  HIP_CHECK(hipSetDevice(was));
  if (found != hipSuccess) return Error::device_not_built;
  if (arch != kTargetArch) return Error::device_not_tuned;
  return Supported{arch};
}

}  // namespace hip_custom_communicator
