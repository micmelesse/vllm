// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Setting it up: IPC export and open, the signals, buffer and graph registration.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "peer.cuh"

namespace hip_comms {

#define HIP_CHECK(expr)                                                             \
  do {                                                                              \
    hipError_t _e = (expr);                                                         \
    if (_e != hipSuccess) {                                                         \
      throw std::runtime_error(std::string("hip_comms: ") + #expr + " -> " +         \
                               hipGetErrorString(_e));                              \
    }                                                                               \
  } while (0)

// ---------------------------------------------------------------------------------
// The context: peer memory and registration. One per process group.
// ---------------------------------------------------------------------------------

using Handle = hipIpcMemHandle_t;

inline Handle handle_from(const std::string& bytes) {
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
inline hipPointer_attribute range_start_attr = HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR;

// std::string as a BYTE BUFFER, not text: an IPC handle is arbitrary binary. It leaves
// this file as `int[]` at the op boundary below, which is how vLLM's other all-reduces
// carry handle bytes through a schema that has no bytes type.
inline std::pair<std::string, int64_t> ipc_handle_and_offset(uintptr_t ptr) {
  void* base = nullptr;
  HIP_CHECK(hipPointerGetAttribute(&base, range_start_attr,
                                   reinterpret_cast<hipDeviceptr_t>(ptr)));
  Handle h;
  HIP_CHECK(hipIpcGetMemHandle(&h, base));
  return {std::string(reinterpret_cast<const char*>(&h), sizeof(Handle)),
          reinterpret_cast<char*>(ptr) - static_cast<char*>(base)};
}

class PeerContext {
 public:
  // `signal_handles`/`signal_offsets` are the whole world's handles for their own signal
  // allocation, gathered in PYTHON -- the collective that exchanges them belongs to the
  // process group, which C++ has no business knowing about.
  PeerContext(int rank, int world_size, uintptr_t self_signal,
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

  ~PeerContext() {
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

  int rank() const { return rank_; }
  int world_size() const { return world_size_; }
  Signal* self_signal() const { return self_signal_; }
  PeerSignals peer_signals() const { return peer_signals_; }
  int64_t scratch_bytes() const { return scratch_bytes_; }

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
