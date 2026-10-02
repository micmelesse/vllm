// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// p2p::host, behind p2p.cuh: `Group` maps every peer's signal block, scratch and
// registered buffers once over HIP IPC handles, and hands a launch the `DevComm` it passes
// to its kernel. The one object with a lifetime, on purpose: the mappings must outlive
// every launch.

#pragma once

#ifndef HIP_COMMS_P2P_INTERFACE
#error "include p2p/p2p.cuh, p2p's one interface, not its parts"
#endif

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "peers.cuh"

#define HIP_CHECK(expr)                                                             \
  do {                                                                              \
    hipError_t _e = (expr);                                                         \
    if (_e != hipSuccess) {                                                         \
      throw std::runtime_error(std::string("hip_comms: ") + #expr + " -> " +         \
                               hipGetErrorString(_e));                              \
    }                                                                               \
  } while (0)

namespace hip_comms::p2p::host {

using IpcHandle = hipIpcMemHandle_t;

inline IpcHandle handle_from(const std::string& bytes) {
  if (bytes.size() != sizeof(IpcHandle))
    throw std::runtime_error("hip_comms: ipc handle is " + std::to_string(bytes.size()) +
                             " bytes, expected " + std::to_string(sizeof(IpcHandle)));
  IpcHandle h;
  std::memcpy(&h, bytes.data(), sizeof(IpcHandle));
  return h;
}

// hipIpcGetMemHandle must be given the BASE of an allocation, but a torch tensor sits at
// an offset inside one -- so the handle names the allocation and the offset locates the
// tensor within it. The handle is a std::string used as a byte buffer.
inline std::pair<std::string, int64_t> handle_and_offset(uintptr_t ptr) {
  void* base = nullptr;
  HIP_CHECK(hipPointerGetAttribute(&base, HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                                   reinterpret_cast<hipDeviceptr_t>(ptr)));
  IpcHandle h;
  HIP_CHECK(hipIpcGetMemHandle(&h, base));
  return {std::string(reinterpret_cast<const char*>(&h), sizeof(IpcHandle)),
          reinterpret_cast<char*>(ptr) - static_cast<char*>(base)};
}

// THIS RANK'S PEER MEMORY, one allocation, zeroed: the signal block, the scratch, then the
// staging an eager input is copied into. UNCACHED, as aiter and vLLM's custom all-reduce allocate
// theirs: peers read what this rank wrote, and cached, those writes sit dirty in L2 for the
// barrier's writeback to flush. The Group it is passed to owns it.
inline uintptr_t alloc_memory(int64_t scratch_bytes, int64_t staging_bytes) {
  void* p = nullptr;
  const size_t bytes = sizeof(Signal) + static_cast<size_t>(scratch_bytes + staging_bytes);
  HIP_CHECK(hipExtMallocWithFlags(&p, bytes, hipDeviceMallocUncached));
  HIP_CHECK(hipMemset(p, 0, bytes));
  HIP_CHECK(hipDeviceSynchronize());
  return reinterpret_cast<uintptr_t>(p);
}

class Group {
 public:
  // `self_memory` is this rank's `alloc_memory`, which the Group now owns; `signal_handles` and
  // `signal_offsets` are the whole world's handles for theirs, gathered in PYTHON -- the collective
  // that exchanges them belongs to the process group. `max_buffers` sizes the peer-pointer slab.
  Group(int rank, int world_size, uintptr_t self_memory,
        const std::vector<std::string>& signal_handles,
        const std::vector<int64_t>& signal_offsets, int64_t max_buffers, int64_t scratch_bytes,
        int64_t staging_bytes, double sync_timeout_s)
      : rank_(rank),
        world_size_(world_size),
        self_signal_(reinterpret_cast<Signal*>(self_memory)),
        scratch_bytes_(scratch_bytes),
        staging_bytes_(staging_bytes) {
    if (world_size_ < 2 || world_size_ > kMaxRanks)
      throw std::runtime_error("hip_comms: world_size " + std::to_string(world_size_) +
                               " outside [2, " + std::to_string(kMaxRanks) + "]");
    if (signal_handles.size() != static_cast<size_t>(world_size_) ||
        signal_offsets.size() != static_cast<size_t>(world_size_))
      throw std::runtime_error("hip_comms: expected one signal handle+offset per rank");
    // THE SLAB the launches' peer-pointer tables live in: read by this rank's kernels only.
    HIP_CHECK(hipMalloc(&slab_, static_cast<size_t>(max_buffers) * sizeof(PeerPtrs)));
    slab_end_ = slab_ + max_buffers;
    cursor_   = slab_;
    auto opened = open_peers(signal_handles, signal_offsets, self_memory);
    for (int i = 0; i < world_size_; ++i)
      signals_.s[i] = reinterpret_cast<Signal*>(opened[i]);
    // THE STAGING IS REGISTERED HERE: every rank's lies at the same offset in its allocation.
    std::vector<void*> staging(world_size_);
    for (int i = 0; i < world_size_; ++i)
      staging[i] = static_cast<char*>(opened[i]) + sizeof(Signal) + scratch_bytes_;
    PeerPtrs* slot = next_slot();
    write_slot(slot, staging);
    registered_[staging[rank_]] = slot;
    // The device wall clock is fixed-rate, in kHz; the kernels count the timeout in it.
    int device = 0, khz = 0;
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, device));
    timeout_ticks_ = static_cast<uint64_t>(sync_timeout_s * khz * 1000.0);
  }

  ~Group() {
    for (const auto& kv : opened_) hipIpcCloseMemHandle(kv.second);
    hipFree(slab_);
    hipFree(self_signal_);
  }

  int world_size() const { return world_size_; }
  int64_t scratch_bytes() const { return scratch_bytes_; }
  // Where an eager input is copied for its peers to read, and how many bytes it holds.
  void* staging() const {
    return reinterpret_cast<char*>(self_signal_) + sizeof(Signal) + scratch_bytes_;
  }
  int64_t staging_bytes() const { return staging_bytes_; }

  // The CAPTURE path, in two halves. During capture the input address is not registered
  // yet, so `dev_comm` reserves a slab slot and remembers the pointer; afterwards Python
  // gathers handles for everything remembered and this fills the slots in. Sound because
  // a captured address is fixed for the graph's life -- the kernel reads a PeerPtrs
  // populated AFTER the capture that recorded the launch.
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
    // The slots are filled in; nothing goes into `registered_`, which means "an address
    // this object keeps alive". A captured buffer dies with its graph, and its slot pointer
    // is baked into the recorded launch, so a replay never looks the address up.
    for (size_t i = 0; i < pending_.size(); ++i) {
      auto ptrs = open_peers(handles[i], offsets[i],
                             reinterpret_cast<uintptr_t>(pending_[i]));
      write_slot(pending_slots_[i], ptrs);
    }
    pending_.clear();
    pending_slots_.clear();
  }

  // What a launch over `input` (`bytes` long) on `stream` passes to its kernel.
  DevComm dev_comm(const void* input, int64_t bytes, hipStream_t stream) {
    DevComm p     = dev_comm();
    p.inputs      = slot_for(const_cast<void*>(input), bytes, stream);
    p.input_packs = bytes / 16;
    return p;
  }

  // What a STAGED kernel's launch passes: no peer reads its input where it is, so it needs no
  // slot; the kernel takes its own input and the staging's size as arguments and copies it in a
  // pass at a time, any size.
  DevComm dev_comm_staged(int64_t bytes) const {
    DevComm p     = dev_comm();
    p.input_packs = bytes / 16;
    return p;
  }

  // Whether the peers can read `input` where it is, on `stream`: registered, or captured (it is
  // registered at capture exit, before any replay). Otherwise it goes through the staging.
  bool reads_in_place(const void* input, hipStream_t stream) const {
    hipStreamCaptureStatus status;
    HIP_CHECK(hipStreamIsCapturing(stream, &status));
    return status == hipStreamCaptureStatusActive ||
           registered_.count(const_cast<void*>(input)) != 0;
  }

  // A launch with no input: only the signals, for a kernel that moves no data.
  DevComm dev_comm() const {
    return DevComm{rank_, nullptr, signals_, self_signal_, 0, scratch_bytes_ / 16, timeout_ticks_};
  }
  int rank() const { return rank_; }
  // The first of `n` flag values for `peer`, the rest reserved: flags only grow, so each use starts
  // past the last (p2p::write_flag).
  uint32_t take_flags(int peer, uint32_t n) {
    const uint32_t base = flags_used_[peer];
    flags_used_[peer] += n;
    return base;
  }

 private:
  // The peers' view of `input`: a capture's deferred slot, a registered buffer's, or the staging's
  // with the input copied in.
  PeerPtrs* slot_for(void* input, int64_t bytes, hipStream_t stream) {
    hipStreamCaptureStatus status;
    HIP_CHECK(hipStreamIsCapturing(stream, &status));
    if (status == hipStreamCaptureStatusActive) {
      // A fresh slot ALWAYS, even for an address `registered_` already knows: a graph's
      // buffers are freed with the graph and the allocator hands the same address back.
      // Skipping the record would make the recorded COUNT depend on that luck, and the
      // exchange after capture is COLLECTIVE, so ranks would exchange different counts.
      PeerPtrs* slot = next_slot();
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
      // Once per distinct handle: a capture-heavy run exchanges the same peer BASES over
      // and over, and vLLM's CustomAllreduce keeps the same cache, keyed the same way.
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

  int rank_;
  int world_size_;
  Signal* self_signal_;
  int64_t scratch_bytes_;
  int64_t staging_bytes_;
  uint64_t timeout_ticks_ = 0;
  PeerSignals signals_{};
  PeerPtrs* slab_     = nullptr;
  PeerPtrs* slab_end_ = nullptr;
  PeerPtrs* cursor_   = nullptr;
  std::unordered_map<void*, PeerPtrs*> registered_;
  std::vector<void*> pending_;
  std::vector<PeerPtrs*> pending_slots_;
  std::unordered_map<std::string, void*> opened_;
  uint32_t flags_used_[kMaxRanks] = {};
};

}  // namespace hip_comms::p2p::host
