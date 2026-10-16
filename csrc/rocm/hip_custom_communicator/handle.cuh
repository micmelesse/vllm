// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE RUNTIME STATE, one `Handle` per communicator: everything the running
// program learns, as common.cuh's `BuildInfo` is everything fixed at
// compile time. OUR OWN MEMORY (the signals and the exchange) is HIP virtual
// memory: each rank creates its physical memory with explicit properties,
// exports it as a POSIX file descriptor, and every rank maps every rank's into
// ONE reserved virtual range, rank r's at base + r * stride, so a kernel reaches
// a peer's buffer by a pointer and a rank stride. The CALLER'S buffers
// (registered or captured inputs: torch's, not ours) are opened over legacy HIP
// IPC handles, a device table of every rank's address each. The collective
// that exchanges the handles and the socket names is the caller's (`Gather`),
// so this names no process group.

#pragma once

#include <c10/util/BFloat16.h>
#include <hip/hip_runtime.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <array>
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
//   local lines (common.cuh's BARRIER). On HIP 7.15 a hipDeviceMallocFinegrained
//   exchange was 4-10% slower than the uncached flag at 16-256 tokens and the
//   same at 1024-4096 (2026-10-16T15-36-54Z); this choice is measured against
//   both on n11.
constexpr hipMemAllocationType kSignalsAllocation = hipMemAllocationTypeUncached;
constexpr hipMemAllocationType kExchangeAllocation = hipMemAllocationTypeUncached;

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
inline std::vector<std::vector<Fd>> exchange_fds(int rank, int world,
                                                 const Gather& gather,
                                                 const std::vector<int>& mine,
                                                 double timeout_seconds) {
  const size_t n = mine.size();
  Fd listener(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
  if (listener.get() < 0) sys_fail("socket");
  std::random_device entropy;
  std::ostringstream os;
  os << "hip_custom_communicator/" << ::getpid() << "/" << entropy() << "/"
     << rank;
  const std::string name = os.str();
  const auto address = [](const std::string& s, sockaddr_un& a) {
    std::memset(&a, 0, sizeof(a));
    a.sun_family = AF_UNIX;
    if (s.size() + 1 > sizeof(a.sun_path))
      throw std::runtime_error("hip_custom_communicator: socket name too long");
    std::memcpy(a.sun_path + 1, s.data(), s.size());  // sun_path[0] = 0: abstract
    return static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + 1 + s.size());
  };
  sockaddr_un a;
  socklen_t len = address(name, a);
  if (::bind(listener.get(), reinterpret_cast<sockaddr*>(&a), len) != 0)
    sys_fail("bind " + name);
  if (::listen(listener.get(), world) != 0) sys_fail("listen");
  const std::vector<std::string> names = gather(name);
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
      if (ready == 0)
        throw std::runtime_error(
            "hip_custom_communicator: " + std::to_string(got) + " of " +
            std::to_string(world - 1) +
            " peers sent their memory's descriptors before the timeout");
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
        c0->cmsg_type != SCM_RIGHTS || c0->cmsg_len != CMSG_LEN(n * sizeof(int)))
      throw std::runtime_error(
          "hip_custom_communicator: a malformed descriptor message (rank " +
          std::to_string(payload) + ")");
    std::vector<int> fds(n);
    std::memcpy(fds.data(), CMSG_DATA(c0), n * sizeof(int));
    for (int fd : fds) theirs[payload].emplace_back(fd);
  }
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
    HIP_CHECK(hipMemCreate(&peers_[rank_], static_cast<size_t>(stride_), &prop, 0));
    HIP_CHECK(hipMemAddressReserve(&base_, static_cast<size_t>(stride_ * world_),
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

  // Every peer's memory, imported from its descriptor (`fds[r]`, rank r's) and
  // mapped at its place.
  void map_peers(const std::vector<int>& fds) {
    for (int r = 0; r < world_; ++r) {
      if (r == rank_) continue;
      HIP_CHECK(hipMemImportFromShareableHandle(
          &peers_[r], reinterpret_cast<void*>(static_cast<intptr_t>(fds[r])),
          hipMemHandleTypePosixFileDescriptor));
      map(r);
      allow_access(r);
    }
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
  Handle(int rank, int world_size, const Gather& gather, double timeout_seconds,
         int64_t max_rows, int64_t max_cols, int64_t scratch_bytes,
         int64_t quantized_values_bytes, int64_t quantized_scales_bytes,
         int64_t staging_bytes, int quantization,
         const BuildInfo& build = kBuild)
      : rank_(rank),
        world_size_(world_size),
        max_rows_(max_rows),
        max_cols_(max_cols),
        scratch_bytes_(scratch_bytes),
        quantized_values_bytes_(quantized_values_bytes),
        quantized_scales_bytes_(quantized_scales_bytes),
        staging_bytes_(staging_bytes),
        quantization_(quantization) {
    // THE SIGNALS APART FROM THE DATA: an overrun of the scratch or the staging
    // runs off its own allocation, never into barrier state. Each made and
    // zeroed here, then every rank's descriptors exchanged (one collective) and
    // every peer's mapped beside this rank's.
    signals_.emplace(rank_, world_size_, kSignalsBytes, kSignalsAllocation);
    exchange_.emplace(rank_, world_size_, staging_offset() + staging_bytes_,
                      kExchangeAllocation);
    {
      const Fd signals_fd = signals_->exported();
      const Fd exchange_fd = exchange_->exported();
      const std::vector<std::vector<Fd>> theirs = exchange_fds(
          rank_, world_size_, gather, {signals_fd.get(), exchange_fd.get()},
          timeout_seconds);
      std::vector<int> signals_fds(world_size_, -1), exchange_fds_of(world_size_, -1);
      for (int r = 0; r < world_size_; ++r) {
        if (r == rank_) continue;
        signals_fds[r] = theirs[r][0].get();
        exchange_fds_of[r] = theirs[r][1].get();
      }
      signals_->map_peers(signals_fds);
      exchange_->map_peers(exchange_fds_of);
    }
    // THE STATUS, host-mapped and coherent: a wait that gives up writes its
    // fault with system-scope atomics while the host polls it, and the host
    // sets its `abort` (common.cuh's BARRIER). The kernels take its device
    // address as their own argument, a buffer of kStatusFields words.
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&status_),
                            kStatusFields * sizeof(uint32_t),
                            hipHostMallocMapped | hipHostMallocCoherent));
    std::memset(status_, 0, kStatusFields * sizeof(uint32_t));
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&device_status_),
                                      status_, 0));
    // THE SLAB the launches' peer tables live in, kMaxRanks addresses a table:
    // read by this rank's kernels only.
    const int64_t slots = build.memory.peer_ptr_slots;
    HIP_CHECK(hipMalloc(
        &slab_, static_cast<size_t>(slots) * kMaxRanks * sizeof(void*)));
    slab_end_ = slab_ + slots * kMaxRanks;
    cursor_ = slab_;
    // EACH RANK'S EXCHANGE is [ scratch | scratch values | scratch scales |
    // staging values | staging scales | staging ], at the same offsets on every
    // rank, so each region is rank 0's address and the exchange's rank stride
    // (scratch_ptr(), exchange_stride_rank_bytes()). The STAGED INPUT'S LIES
    // PAST EVERY KERNEL'S SCRATCH, since an INT4 two-shot writes it before its
    // first barrier, while a peer may still read this rank's scratch after its
    // last (common.cuh's publish_staged). AN EAGER INPUT IS READ AS AN INPUT,
    // through a table like any registered one (`input_ptrs`), so the staging
    // also has its table, written once, registered here.
    std::vector<void*> staging(world_size_);
    for (int i = 0; i < world_size_; ++i)
      staging[i] = exchange_->at(i) + staging_offset();
    staging_ptrs_ = next_slot();
    write_slot(staging_ptrs_, staging);
    registered_[staging[rank_]] = staging_ptrs_;
    // The device wall clock is fixed-rate, in kHz; the kernels count the
    // timeout in it.
    int device = 0, khz = 0;
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(
        hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, device));
    timeout_ticks_ = static_cast<uint64_t>(timeout_seconds * khz * 1000.0);
  }

  // The symmetric memory unmaps and releases itself (SymmetricMemory), after
  // every launch on it has finished.
  ~Handle() {
    (void)hipDeviceSynchronize();
    for (const auto& kv : opened_) hipIpcCloseMemHandle(kv.second);
    hipFree(slab_);
    hipHostFree(status_);
  }

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
  // Where an eager input is copied for its peers to read, and how many bytes it
  // holds.
  void* staging() const { return exchange_->at(rank_) + staging_offset(); }
  int64_t staging_bytes() const { return staging_bytes_; }
  // What the tuned path moves across the links, a Quantization.
  int quantization() const { return quantization_; }

  // THE BUFFERS A CAPTURE RECORDED, registered, every rank together. During
  // capture an input's address is not registered yet, so `inp_ptrs` reserves a
  // slot and remembers the pointer; this gathers every rank's handle for each
  // and fills the slots in. Sound because a captured address is fixed for the
  // graph's life: the kernel reads a slot filled AFTER the capture that
  // recorded the launch. False, with nothing registered, when the ranks
  // captured different numbers (every rank must run the same graphs); a
  // collective even with none, or the other ranks wait in it.
  bool register_captured(const Gather& gather) {
    const int64_t mine = static_cast<int64_t>(pending_.size());
    for (const std::string& c : gather(
             std::string(reinterpret_cast<const char*>(&mine), sizeof(mine)))) {
      int64_t n = 0;
      std::memcpy(&n, c.data(), sizeof(n));
      if (n != mine) return false;
    }
    if (pending_.empty()) return true;
    std::string all;
    for (void* p : pending_) all += ipc_bytes(p);
    const size_t each = all.size() / pending_.size();
    std::vector<std::vector<std::string>> theirs(pending_.size());
    for (const std::string& rank_bytes : gather(all))
      for (size_t i = 0; i < pending_.size(); ++i)
        theirs[i].push_back(rank_bytes.substr(i * each, each));
    // Nothing goes into `registered_`, which means "an address this object
    // keeps alive": a captured buffer dies with its graph, and its slot is
    // baked into the recorded launch.
    for (size_t i = 0; i < pending_.size(); ++i)
      write_slot(pending_slots_[i], open_peers(theirs[i], pending_[i]));
    pending_.clear();
    pending_slots_.clear();
    return true;
  }

  // A BUFFER EVERY RANK ALLOCATED, registered every rank together: launches
  // over `self` read the peers' copies where they are. `forget_buffer` undoes
  // it before the buffer is freed, or a later allocation at the same address
  // would read the dead buffer's peers.
  void register_buffer(void* self, const Gather& gather) {
    const std::vector<std::string> theirs = gather(ipc_bytes(self));
    void** slot = next_slot();
    write_slot(slot, open_peers(theirs, self));
    registered_[self] = slot;
    buffer_handles_[self] = theirs;
  }
  void forget_buffer(void* self) {
    registered_.erase(self);
    const auto it = buffer_handles_.find(self);
    if (it == buffer_handles_.end()) return;
    for (int i = 0; i < world_size_; ++i) {
      if (i == rank_) continue;
      if (auto o =
              opened_.find(it->second[i].substr(0, sizeof(hipIpcMemHandle_t)));
          o != opened_.end()) {
        HIP_CHECK(hipIpcCloseMemHandle(o->second));
        opened_.erase(o);
      }
    }
    buffer_handles_.erase(it);
  }

  // WHAT A KERNEL IS HANDED, each its own argument, a peer buffer as a device
  // table of every rank's address. Every rank's input for a launch over `input`
  // (`bytes` long) on `stream`: filled after the capture for a captured launch
  // (an eager input is copied into the staging, and the table is the
  // staging's).
  const void* const* input_ptrs(const void* input, int64_t bytes,
                                hipStream_t stream) {
    return slot_for(const_cast<void*>(input), bytes, stream);
  }
  // EVERY RANK'S SCRATCH AND STAGING, and the quantized forms of the scratch
  // and of the staged input, fixed for the handle's life: each region is rank
  // 0's address, rank r's exchange_stride_rank_bytes() * r past it (a kernel
  // takes the stride in its buffer's elements: the launch divides).
  void* scratch_ptr() const { return exchange_->base(); }
  void* scratch_values_ptr() const { return exchange_->base() + scratch_bytes_; }
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
  int64_t exchange_stride_rank_bytes() const { return exchange_->stride_bytes(); }
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
  // Every wait on this rank leaves at its next check: for a communicator the
  // host has given up on.
  void abort() {
    __atomic_store_n(&status_[static_cast<int>(StatusField::abort)],
                     uint32_t{1}, __ATOMIC_RELEASE);
  }

  // Whether the peers can read `input` where it is, on `stream`: registered, or
  // captured (it is registered at capture exit, before any replay). Otherwise
  // it goes through the staging.
  bool reads_in_place(const void* input, hipStream_t stream) const {
    hipStreamCaptureStatus status;
    HIP_CHECK(hipStreamIsCapturing(stream, &status));
    return status == hipStreamCaptureStatusActive ||
           registered_.count(const_cast<void*>(input)) != 0;
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
  // The staging's offset in the exchange: past the scratch and the quantized
  // regions.
  int64_t staging_offset() const {
    return scratch_bytes_ + 2 * (quantized_values_bytes_ + quantized_scales_bytes_);
  }

  // A BUFFER AS THE GATHER CARRIES IT: its allocation's IPC handle, then its
  // offset in that allocation. hipIpcGetMemHandle must be given an allocation's
  // BASE, and a torch tensor sits at an offset inside one.
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

  // Every rank's `ipc_bytes` opened into a local pointer, `self` for this
  // rank's: our OWN handle is never opened (hipIpcOpenMemHandle refuses a
  // self-handle). Once per distinct allocation: a capture-heavy run gathers the
  // same peer BASES over and over, and vLLM's CustomAllreduce keeps the same
  // cache, keyed the same way.
  std::vector<void*> open_peers(const std::vector<std::string>& theirs,
                                void* self) {
    constexpr size_t kIpc = sizeof(hipIpcMemHandle_t);
    std::vector<void*> output(world_size_, nullptr);
    for (int i = 0; i < world_size_; ++i) {
      if (i == rank_) {
        output[i] = self;
        continue;
      }
      if (theirs[i].size() != kIpc + sizeof(int64_t))
        throw std::runtime_error(
            "hip_custom_communicator: a gathered buffer is an IPC handle and "
            "an offset");
      const std::string handle = theirs[i].substr(0, kIpc);
      int64_t offset = 0;
      std::memcpy(&offset, theirs[i].data() + kIpc, sizeof(offset));
      auto it = opened_.find(handle);
      if (it == opened_.end()) {
        hipIpcMemHandle_t h;
        std::memcpy(&h, handle.data(), kIpc);
        void* base = nullptr;
        HIP_CHECK(hipIpcOpenMemHandle(&base, h, hipIpcMemLazyEnablePeerAccess));
        it = opened_.emplace(handle, base).first;
      }
      output[i] = static_cast<char*>(it->second) + offset;
    }
    return output;
  }

  // The peers' view of `input`: a capture's deferred slot, a registered
  // buffer's, or the staging's with the input copied in.
  void** slot_for(void* input, int64_t bytes, hipStream_t stream) {
    hipStreamCaptureStatus status;
    HIP_CHECK(hipStreamIsCapturing(stream, &status));
    if (status == hipStreamCaptureStatusActive) {
      // A fresh slot ALWAYS, even for an address `registered_` already knows: a
      // graph's buffers are freed with the graph and the allocator hands the
      // same address back. Skipping the record would make the recorded COUNT
      // depend on that luck, and the exchange after capture is COLLECTIVE, so
      // ranks would exchange different counts.
      void** slot = next_slot();
      pending_.push_back(input);
      pending_slots_.push_back(slot);
      return slot;
    }
    if (auto it = registered_.find(input); it != registered_.end())
      return it->second;
    // AN EAGER INPUT the peers cannot read (the caching allocator's, borrowed
    // for the call): copied into the staging, which they mapped once, on the
    // launch's stream.
    if (bytes > staging_bytes_) {
      std::ostringstream os;
      os << "hip_custom_communicator: an eager " << bytes
         << "-byte input exceeds the " << staging_bytes_ << "-byte staging";
      throw std::runtime_error(os.str());
    }
    HIP_CHECK(hipMemcpyAsync(staging(), input, static_cast<size_t>(bytes),
                             hipMemcpyDeviceToDevice, stream));
    return registered_.at(staging());
  }

  // A TABLE, kMaxRanks addresses of the slab.
  void** next_slot() {
    if (cursor_ >= slab_end_)
      throw std::runtime_error(
          "hip_custom_communicator: peer-pointer slab is full; allocate a "
          "larger one");
    void** slot = cursor_;
    cursor_ += kMaxRanks;
    return slot;
  }

  void write_slot(void** slot, const std::vector<void*>& ptrs) {
    std::vector<void*> host(kMaxRanks, nullptr);
    for (int i = 0; i < world_size_; ++i) host[i] = ptrs[i];
    HIP_CHECK(hipMemcpy(slot, host.data(), kMaxRanks * sizeof(void*),
                        hipMemcpyHostToDevice));
  }

  int rank_;
  int world_size_;
  int64_t max_rows_;
  int64_t max_cols_;
  int64_t scratch_bytes_;
  int64_t quantized_values_bytes_;
  int64_t quantized_scales_bytes_;
  int64_t staging_bytes_;
  int quantization_;
  std::optional<SymmetricMemory> signals_;
  std::optional<SymmetricMemory> exchange_;
  uint32_t* status_ = nullptr;
  uint32_t* device_status_ = nullptr;
  uint64_t timeout_ticks_ = 0;
  void** staging_ptrs_ = nullptr;
  void** slab_ = nullptr;
  void** slab_end_ = nullptr;
  void** cursor_ = nullptr;
  std::unordered_map<void*, void**> registered_;
  std::unordered_map<void*, std::vector<std::string>> buffer_handles_;
  mutable std::unordered_map<const void*, Resources> resources_;
  std::vector<void*> pending_;
  std::vector<void**> pending_slots_;
  std::unordered_map<std::string, void*> opened_;
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
                  all_reduce_pull_one_shot<c10::BFloat16, 2, 1,
                                           kWaveSize * traits<c10::BFloat16>::N,
                                           kWaveSize, 1>));
  (void)hipGetLastError();
  HIP_CHECK(hipSetDevice(was));
  if (found != hipSuccess) return Error::device_not_built;
  if (arch != kTargetArch) return Error::device_not_tuned;
  return Supported{arch};
}

}  // namespace hip_custom_communicator
