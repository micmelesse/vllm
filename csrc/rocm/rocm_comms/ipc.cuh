// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// The layer every collective is built on: peer memory and synchronisation over HIP IPC.
// `Peers` is what a kernel gets; `Group` is the host object that maps the peers and
// hands out a `Peers` per launch.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils.cuh"

#define HIP_CHECK(expr)                                                             \
  do {                                                                              \
    hipError_t _e = (expr);                                                         \
    if (_e != hipSuccess) {                                                         \
      throw std::runtime_error(std::string("hip_comms: ") + #expr + " -> " +         \
                               hipGetErrorString(_e));                              \
    }                                                                               \
  } while (0)

namespace hip_comms::ipc {

constexpr int kMaxRanks  = 8;
constexpr int kMaxBlocks = 36;

// One IPC allocation per rank holds the signal block AND the scratch: scratch is simply
// the bytes after the struct.
//
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

// =================================================================================
// DEVICE SIDE. Everything a collective kernel may do with its peers: read their input,
// read and write scratch, and synchronise.
// =================================================================================

class Peers {
 public:
  Peers(const PeerPtrs* inputs, PeerSignals signals, Signal* self, int rank)
      : rank_(rank), inputs_(inputs), signals_(signals), self_(self) {}

  template <typename V>
  DINLINE const V* input(int r) const {
    return reinterpret_cast<const V*>(inputs_->p[r]);
  }

  template <typename V>
  DINLINE V* scratch(int r) const {
    return reinterpret_cast<V*>(signals_.s[r] + 1);
  }

  // System scope to REACH a peer's memory, device scope to poll our own.
  template <int ngpus>
  DINLINE void barrier_start() const {
    uint32_t f = self_->seq[blockIdx.x] + 1;
    if (threadIdx.x < ngpus) {
      __scoped_atomic_store_n(&signals_.s[threadIdx.x]->start[blockIdx.x][rank_], f,
                              __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
      while (__scoped_atomic_load_n(&self_->start[blockIdx.x][threadIdx.x],
                                    __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) < f);
    }
    __syncthreads();
    if (threadIdx.x == 0) self_->seq[blockIdx.x] = f;
  }

  // `final_sync` drops the release/acquire pair: nothing after the last barrier reads what
  // this kernel wrote, so ordering costs without buying anything.
  template <int ngpus, bool final_sync>
  DINLINE void barrier_end() const {
    __syncthreads();
    uint32_t f = self_->seq[blockIdx.x] + 1;
    if (threadIdx.x < ngpus) {
      __scoped_atomic_store_n(&signals_.s[threadIdx.x]->end[blockIdx.x][rank_], f,
                              final_sync ? __ATOMIC_RELAXED : __ATOMIC_RELEASE,
                              __MEMORY_SCOPE_SYSTEM);
      while (__scoped_atomic_load_n(&self_->end[blockIdx.x][threadIdx.x],
                                    final_sync ? __ATOMIC_RELAXED : __ATOMIC_ACQUIRE,
                                    __MEMORY_SCOPE_DEVICE) < f);
    }
    if constexpr (!final_sync) __syncthreads();
    if (threadIdx.x == 0) self_->seq[blockIdx.x] = f;
  }

  DINLINE int rank() const { return rank_; }

 private:
  int rank_;
  const PeerPtrs* inputs_;
  PeerSignals signals_;
  Signal* self_;
};

// =================================================================================
// HOST SIDE. Maps the peers' memory once, registers buffers, and turns an input
// pointer into the `Peers` a launch passes.
// =================================================================================

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
// tensor within it. The handle is a std::string used as a byte buffer.
inline std::pair<std::string, int64_t> handle_and_offset(uintptr_t ptr) {
  void* base = nullptr;
  HIP_CHECK(hipPointerGetAttribute(&base, HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                                   reinterpret_cast<hipDeviceptr_t>(ptr)));
  Handle h;
  HIP_CHECK(hipIpcGetMemHandle(&h, base));
  return {std::string(reinterpret_cast<const char*>(&h), sizeof(Handle)),
          reinterpret_cast<char*>(ptr) - static_cast<char*>(base)};
}

class Group {
 public:
  // `signal_handles`/`signal_offsets` are the whole world's handles for their own signal
  // allocation, gathered in PYTHON -- the collective that exchanges them belongs to the
  // process group, which C++ has no business knowing about.
  Group(int rank, int world_size, uintptr_t self_signal,
        const std::vector<std::string>& signal_handles,
        const std::vector<int64_t>& signal_offsets, uintptr_t peer_slab,
        int64_t peer_slab_bytes, int64_t scratch_bytes)
      : rank_(rank),
        world_size_(world_size),
        self_signal_(reinterpret_cast<Signal*>(self_signal)),
        scratch_bytes_(scratch_bytes),
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
      signals_.s[i] = reinterpret_cast<Signal*>(opened[i]);
  }

  ~Group() {
    for (const auto& kv : opened_) hipIpcCloseMemHandle(kv.second);
  }

  int world_size() const { return world_size_; }
  int64_t scratch_bytes() const { return scratch_bytes_; }

  // A buffer whose address is known ahead of time. The eager path.
  void register_buffer(const std::vector<std::string>& handles,
                       const std::vector<int64_t>& offsets, uintptr_t self_ptr) {
    auto ptrs = open_peers(handles, offsets, self_ptr);
    PeerPtrs* slot = next_slot();
    write_slot(slot, ptrs);
    registered_[reinterpret_cast<void*>(self_ptr)] = slot;
  }

  // The CAPTURE path, in two halves. During capture the input address is not registered
  // yet, so `peers` reserves a slab slot and remembers the pointer; afterwards Python
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

  int64_t pending_count() const { return static_cast<int64_t>(pending_.size()); }

  // What a launch over `input` passes to its kernel.
  Peers peers(void* input) {
    return Peers(slot_for(input), signals_, self_signal_, rank_);
  }

 private:
  PeerPtrs* slot_for(void* input) {
    hipStreamCaptureStatus status;
    HIP_CHECK(hipStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &status));
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
  PeerSignals signals_{};
  PeerPtrs* slab_end_;
  PeerPtrs* cursor_;
  std::unordered_map<void*, PeerPtrs*> registered_;
  std::vector<void*> pending_;
  std::vector<PeerPtrs*> pending_slots_;
  std::unordered_map<std::string, void*> opened_;
};

}  // namespace hip_comms::ipc
