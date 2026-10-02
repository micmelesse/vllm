// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE RUNTIME STATE, one `Handle` per communicator: everything the running program learns, as
// build.cuh's `BuildInfo` is everything fixed at compile time. It maps every peer's signal block,
// scratch and registered buffers once over HIP IPC handles, and hands a launch the `DevComm` its
// kernel reads. The collective that exchanges the handles is the caller's (`Gather`), so this
// names no process group.

#pragma once

#include <c10/util/BFloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "build.cuh"
#include "error.cuh"
#include "kernels/all_reduce_pull_one_shot.cuh"
#include "p2p/p2p.cuh"

#define HIP_CHECK(expr)                                                     \
  do {                                                                      \
    hipError_t _e = (expr);                                                 \
    if (_e != hipSuccess) {                                                 \
      throw std::runtime_error(std::string("hip_comms: ") + #expr + " -> " + \
                               hipGetErrorString(_e));                      \
    }                                                                       \
  } while (0)

namespace hip_comms {

// EVERY RANK'S `mine`, in rank order, `mine` the same length on every rank: a collective, so every
// rank calls it together.
using Gather = std::function<std::vector<std::string>(const std::string&)>;

class Handle {
 public:
  // OPENED on every rank together: this rank's symmetric memory (the signal block, the scratch,
  // then the staging) allocated, its IPC handle gathered with every rank's, and the peers' opened.
  // `world_size` is one the build holds (the opener checks it).
  Handle(int rank, int world_size, const Gather& gather, const BuildInfo& build = kBuild)
      : rank_(rank),
        world_size_(world_size),
        scratch_bytes_(build.memory.scratch_bytes),
        staging_bytes_(build.memory.staging_bytes) {
    self_signal_ = static_cast<p2p::Signal*>(alloc_symmetric());
    // THE SLAB the launches' peer-pointer tables live in: read by this rank's kernels only.
    const int64_t slots = build.memory.peer_ptr_slots;
    HIP_CHECK(hipMalloc(&slab_, static_cast<size_t>(slots) * sizeof(p2p::PeerPtrs)));
    slab_end_         = slab_ + slots;
    cursor_           = slab_;
    const auto opened = open_peers(gather(ipc_bytes(self_signal_)), self_signal_);
    for (int i = 0; i < world_size_; ++i)
      signals_.s[i] = static_cast<p2p::Signal*>(opened[i]);
    // THE STAGING IS REGISTERED HERE: every rank's lies at the same offset in its allocation.
    std::vector<void*> staging(world_size_);
    for (int i = 0; i < world_size_; ++i)
      staging[i] = static_cast<char*>(opened[i]) + sizeof(p2p::Signal) + scratch_bytes_;
    p2p::PeerPtrs* slot = next_slot();
    write_slot(slot, staging);
    registered_[staging[rank_]] = slot;
    // The device wall clock is fixed-rate, in kHz; the kernels count the timeout in it.
    int device = 0, khz = 0;
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, device));
    timeout_ticks_ = static_cast<uint64_t>(build.kernels.sync_timeout_seconds * khz * 1000.0);
  }

  ~Handle() {
    for (const auto& kv : opened_) hipIpcCloseMemHandle(kv.second);
    hipFree(slab_);
    hipFree(self_signal_);
  }

  Handle(const Handle&)            = delete;
  Handle& operator=(const Handle&) = delete;

  int rank() const { return rank_; }
  int world_size() const { return world_size_; }
  int64_t scratch_bytes() const { return scratch_bytes_; }
  // Where an eager input is copied for its peers to read, and how many bytes it holds.
  void* staging() const {
    return reinterpret_cast<char*>(self_signal_) + sizeof(p2p::Signal) + scratch_bytes_;
  }
  int64_t staging_bytes() const { return staging_bytes_; }

  // THE BUFFERS A CAPTURE RECORDED, registered, every rank together. During capture an input's
  // address is not registered yet, so `dev_comm` reserves a slot and remembers the pointer; this
  // gathers every rank's handle for each and fills the slots in. Sound because a captured address
  // is fixed for the graph's life: the kernel reads a slot filled AFTER the capture that recorded
  // the launch. False, with nothing registered, when the ranks captured different numbers (every
  // rank must run the same graphs); a collective even with none, or the other ranks wait in it.
  bool register_captured(const Gather& gather) {
    const int64_t mine = static_cast<int64_t>(pending_.size());
    for (const std::string& c :
         gather(std::string(reinterpret_cast<const char*>(&mine), sizeof(mine)))) {
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
    // Nothing goes into `registered_`, which means "an address this object keeps alive": a
    // captured buffer dies with its graph, and its slot is baked into the recorded launch.
    for (size_t i = 0; i < pending_.size(); ++i)
      write_slot(pending_slots_[i], open_peers(theirs[i], pending_[i]));
    pending_.clear();
    pending_slots_.clear();
    return true;
  }

  // A BUFFER EVERY RANK ALLOCATED, registered every rank together: launches over `self` read the
  // peers' copies where they are. `forget_buffer` undoes it before the buffer is freed, or a later
  // allocation at the same address would read the dead buffer's peers.
  void register_buffer(void* self, const Gather& gather) {
    const std::vector<std::string> theirs = gather(ipc_bytes(self));
    p2p::PeerPtrs* slot                   = next_slot();
    write_slot(slot, open_peers(theirs, self));
    registered_[self]     = slot;
    buffer_handles_[self] = theirs;
  }
  void forget_buffer(void* self) {
    registered_.erase(self);
    const auto it = buffer_handles_.find(self);
    if (it == buffer_handles_.end()) return;
    for (int i = 0; i < world_size_; ++i) {
      if (i == rank_) continue;
      if (auto o = opened_.find(it->second[i].substr(0, sizeof(hipIpcMemHandle_t)));
          o != opened_.end()) {
        HIP_CHECK(hipIpcCloseMemHandle(o->second));
        opened_.erase(o);
      }
    }
    buffer_handles_.erase(it);
  }

  // What a launch over `input` (`bytes` long) on `stream` passes to its kernel.
  p2p::DevComm dev_comm(const void* input, int64_t bytes, hipStream_t stream) {
    p2p::DevComm p = dev_comm();
    p.inputs       = slot_for(const_cast<void*>(input), bytes, stream);
    p.input_packs  = bytes / 16;
    return p;
  }

  // What a STAGED kernel's launch passes: no peer reads its input where it is, so it needs no
  // slot; the kernel takes its own input and the staging's size as arguments and copies it in a
  // pass at a time, any size.
  p2p::DevComm dev_comm_staged(int64_t bytes) const {
    p2p::DevComm p = dev_comm();
    p.input_packs  = bytes / 16;
    return p;
  }

  // A launch with no input: only the signals, for a kernel that moves no data.
  p2p::DevComm dev_comm() const {
    return p2p::DevComm{rank_, nullptr, signals_, self_signal_, 0, scratch_bytes_ / 16,
                        timeout_ticks_};
  }

  // Whether the peers can read `input` where it is, on `stream`: registered, or captured (it is
  // registered at capture exit, before any replay). Otherwise it goes through the staging.
  bool reads_in_place(const void* input, hipStream_t stream) const {
    hipStreamCaptureStatus status;
    HIP_CHECK(hipStreamIsCapturing(stream, &status));
    return status == hipStreamCaptureStatusActive ||
           registered_.count(const_cast<void*>(input)) != 0;
  }

  // What a compiled kernel uses, read once from the code object loaded on this device: only the
  // compiler knows it.
  Resources resources_of(const void* kernel) const {
    if (const auto it = resources_.find(kernel); it != resources_.end()) return it->second;
    hipFuncAttributes attrs{};
    HIP_CHECK(hipFuncGetAttributes(&attrs, kernel));
    return resources_[kernel] = {attrs.numRegs, static_cast<int64_t>(attrs.sharedSizeBytes)};
  }

  // The first of `n` flag values for `peer`, the rest reserved: flags only grow, so each use starts
  // past the last (p2p::write_flag).
  uint32_t take_flags(int peer, uint32_t n) {
    const uint32_t base = flags_used_[peer];
    flags_used_[peer] += n;
    return base;
  }

 private:
  // THIS RANK'S SYMMETRIC MEMORY, one allocation, zeroed: the signal block, the scratch, then the
  // staging. UNCACHED, as aiter and vLLM's custom all-reduce allocate theirs: peers read what this
  // rank wrote, and cached, those writes sit dirty in L2 for the barrier's writeback to flush.
  void* alloc_symmetric() const {
    void* p            = nullptr;
    const size_t bytes = sizeof(p2p::Signal) + static_cast<size_t>(scratch_bytes_ + staging_bytes_);
    HIP_CHECK(hipExtMallocWithFlags(&p, bytes, hipDeviceMallocUncached));
    HIP_CHECK(hipMemset(p, 0, bytes));
    HIP_CHECK(hipDeviceSynchronize());
    return p;
  }

  // A BUFFER AS THE GATHER CARRIES IT: its allocation's IPC handle, then its offset in that
  // allocation. hipIpcGetMemHandle must be given an allocation's BASE, and a torch tensor sits at
  // an offset inside one.
  static std::string ipc_bytes(const void* ptr) {
    void* base = nullptr;
    HIP_CHECK(hipPointerGetAttribute(&base, HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                                     reinterpret_cast<hipDeviceptr_t>(const_cast<void*>(ptr))));
    hipIpcMemHandle_t h;
    HIP_CHECK(hipIpcGetMemHandle(&h, base));
    const int64_t offset = static_cast<const char*>(ptr) - static_cast<char*>(base);
    return std::string(reinterpret_cast<const char*>(&h), sizeof(h)) +
           std::string(reinterpret_cast<const char*>(&offset), sizeof(offset));
  }

  // Every rank's `ipc_bytes` opened into a local pointer, `self` for this rank's: our OWN handle is
  // never opened (hipIpcOpenMemHandle refuses a self-handle). Once per distinct allocation: a
  // capture-heavy run gathers the same peer BASES over and over, and vLLM's CustomAllreduce keeps
  // the same cache, keyed the same way.
  std::vector<void*> open_peers(const std::vector<std::string>& theirs, void* self) {
    constexpr size_t kIpc = sizeof(hipIpcMemHandle_t);
    std::vector<void*> out(world_size_, nullptr);
    for (int i = 0; i < world_size_; ++i) {
      if (i == rank_) {
        out[i] = self;
        continue;
      }
      if (theirs[i].size() != kIpc + sizeof(int64_t))
        throw std::runtime_error("hip_comms: a gathered buffer is an IPC handle and an offset");
      const std::string handle = theirs[i].substr(0, kIpc);
      int64_t offset           = 0;
      std::memcpy(&offset, theirs[i].data() + kIpc, sizeof(offset));
      auto it = opened_.find(handle);
      if (it == opened_.end()) {
        hipIpcMemHandle_t h;
        std::memcpy(&h, handle.data(), kIpc);
        void* base = nullptr;
        HIP_CHECK(hipIpcOpenMemHandle(&base, h, hipIpcMemLazyEnablePeerAccess));
        it = opened_.emplace(handle, base).first;
      }
      out[i] = static_cast<char*>(it->second) + offset;
    }
    return out;
  }

  // The peers' view of `input`: a capture's deferred slot, a registered buffer's, or the staging's
  // with the input copied in.
  p2p::PeerPtrs* slot_for(void* input, int64_t bytes, hipStream_t stream) {
    hipStreamCaptureStatus status;
    HIP_CHECK(hipStreamIsCapturing(stream, &status));
    if (status == hipStreamCaptureStatusActive) {
      // A fresh slot ALWAYS, even for an address `registered_` already knows: a graph's
      // buffers are freed with the graph and the allocator hands the same address back.
      // Skipping the record would make the recorded COUNT depend on that luck, and the
      // exchange after capture is COLLECTIVE, so ranks would exchange different counts.
      p2p::PeerPtrs* slot = next_slot();
      pending_.push_back(input);
      pending_slots_.push_back(slot);
      return slot;
    }
    if (auto it = registered_.find(input); it != registered_.end()) return it->second;
    // AN EAGER INPUT the peers cannot read (the caching allocator's, borrowed for the call): copied
    // into the staging, which they mapped once, on the launch's stream.
    if (bytes > staging_bytes_) {
      std::ostringstream os;
      os << "hip_comms: an eager " << bytes << "-byte input exceeds the " << staging_bytes_
         << "-byte staging";
      throw std::runtime_error(os.str());
    }
    HIP_CHECK(hipMemcpyAsync(staging(), input, static_cast<size_t>(bytes),
                             hipMemcpyDeviceToDevice, stream));
    return registered_.at(staging());
  }

  p2p::PeerPtrs* next_slot() {
    if (cursor_ >= slab_end_)
      throw std::runtime_error("hip_comms: peer-pointer slab is full; allocate a larger one");
    return cursor_++;
  }

  void write_slot(p2p::PeerPtrs* slot, const std::vector<void*>& ptrs) {
    p2p::PeerPtrs host{};
    for (int i = 0; i < world_size_; ++i) host.p[i] = ptrs[i];
    HIP_CHECK(hipMemcpy(slot, &host, sizeof(p2p::PeerPtrs), hipMemcpyHostToDevice));
  }

  int rank_;
  int world_size_;
  int64_t scratch_bytes_;
  int64_t staging_bytes_;
  p2p::Signal* self_signal_ = nullptr;
  uint64_t timeout_ticks_   = 0;
  p2p::PeerSignals signals_{};
  p2p::PeerPtrs* slab_     = nullptr;
  p2p::PeerPtrs* slab_end_ = nullptr;
  p2p::PeerPtrs* cursor_   = nullptr;
  std::unordered_map<void*, p2p::PeerPtrs*> registered_;
  std::unordered_map<void*, std::vector<std::string>> buffer_handles_;
  mutable std::unordered_map<const void*, Resources> resources_;
  std::vector<void*> pending_;
  std::vector<p2p::PeerPtrs*> pending_slots_;
  std::unordered_map<std::string, void*> opened_;
  uint32_t flags_used_[p2p::kMaxRanks] = {};
};

// =================================================================================================
// WHETHER A HANDLE CAN EXIST HERE (`supported`): whether the library runs on a device and world,
// asked once before anything is opened. The build answers, not a list: a device with no code
// object for our kernels is not built, and one that is but is not select's target would run on
// another device's tuning.
// =================================================================================================

// WHAT THE LIBRARY RUNS ON, once `supported` finds it can: the device's arch, as HIP names it.
struct Supported {
  std::string arch;
};

inline std::variant<Supported, Error> supported(int device, int world) {
  if (!world_built(world)) return Error::world_not_built;
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, device) != hipSuccess) return Error::device_not_built;
  const std::string name = prop.gcnArchName;
  const std::string arch = name.substr(0, name.find(':'));  // gfx950:sramecc+:xnack-
  // A CODE OBJECT FOR THIS DEVICE: any of our kernels has one exactly when the build covered it.
  int was = 0;
  HIP_CHECK(hipGetDevice(&was));
  HIP_CHECK(hipSetDevice(device));
  hipFuncAttributes attrs;
  const hipError_t found = hipFuncGetAttributes(
      &attrs, reinterpret_cast<const void*>(all_reduce_pull_one_shot<c10::BFloat16, 2, false>));
  (void)hipGetLastError();
  HIP_CHECK(hipSetDevice(was));
  if (found != hipSuccess) return Error::device_not_built;
  if (arch != kTargetArch) return Error::device_not_tuned;
  return Supported{arch};
}

}  // namespace hip_comms
