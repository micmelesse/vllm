// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// The layer every collective is built on: peer memory and synchronisation over HIP IPC.
// A kernel is passed `Peers` and does everything through `Comm`, built from it; `Group`
// is the host object that maps the peers and hands out a `Peers` per launch.

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
constexpr int kMaxBlocks = 64;

// One IPC allocation per rank holds the signal block AND the scratch: scratch is simply
// the bytes after the struct.
//
// TWO counter arrays, not one. A peer block can reach the second barrier while this one
// is still at the first, and with a single array the peer would write counter+1 while we
// busy-wait on counter. `seq` is the per-block monotonic sequence number.
//
// The barriers use the rest: `peer[r]` is the last world barrier rank r posted here,
// `arrive` and `gen` the grid barrier on this device, `epoch` the syncs this rank has
// completed.
struct Signal {
  alignas(128) uint32_t start[kMaxBlocks][kMaxRanks];
  alignas(128) uint32_t end[kMaxBlocks][kMaxRanks];
  alignas(128) uint32_t seq[kMaxBlocks];
  alignas(128) uint32_t peer[kMaxRanks];
  alignas(128) uint32_t arrive;
  alignas(128) uint32_t gen;
  alignas(128) uint32_t epoch;
};

struct __align__(16) PeerPtrs { void* p[kMaxRanks]; };
struct __align__(16) PeerSignals { Signal* s[kMaxRanks]; };

// =================================================================================
// DEVICE SIDE.
// =================================================================================

template <typename T, int ngpus>
class Comm;

// What a launch passes: opaque to the kernel, which hands it to `Comm`.
class Peers {
 public:
  Peers(const PeerPtrs* inputs, PeerSignals signals, Signal* self, int rank,
        int64_t input_packs, int64_t scratch_packs, uint64_t timeout_ticks, bool checked)
      : rank_(rank),
        checked_(checked),
        inputs_(inputs),
        signals_(signals),
        self_(self),
        input_packs_(input_packs),
        scratch_packs_(scratch_packs),
        timeout_ticks_(timeout_ticks) {}

 private:
  template <typename, int>
  friend class Comm;

  int rank_;
  bool checked_;
  const PeerPtrs* inputs_;
  PeerSignals signals_;
  Signal* self_;
  int64_t input_packs_;
  int64_t scratch_packs_;
  uint64_t timeout_ticks_;
};

// =================================================================================
// THE KERNEL'S WHOLE VIEW OF ITS PEERS. Every rank's input is read only through `sum`;
// every rank's scratch (its own included) through `put` and `get`; the barriers order it.
// Indices are in 16-byte packs of T.
//
//   Comm c(p);             returns once every peer has launched: their inputs are ready
//   c.sum(idx)             input pack idx summed over ranks, fp32, rounded once
//   c.put(peer, idx, v)    into peer's scratch
//   c.get(peer, idx)       from peer's scratch (put/get_float: a float, idx in floats)
//   c.mine(idx)            this rank's own input pack
//   c.ptr(peer, idx, n)    a direct pointer to n packs of it, checked once, for hot loops
//   c.world_barrier()      every put before it, by any block of any rank, is visible to
//                          every get after it
//   c.peer_block_barrier() the same between this block and its same-numbered peers only
//   c.reduce_flat          a flat range summed over ranks, batched, handed to a store
//   c.gather_flat/_rows    after a peer_block_barrier, read back what those peers put
//   c.grid_barrier()       the same for the blocks of this rank and its own scratch
//   c.block_barrier()      the threads of this block
//   c.close()              last; after it this rank's input may be reused (unneeded when
//                          the input is last read before a peer_block_barrier)
//
// A wait that outlives the timeout prints where it was and traps, so a hang is an error.
// Checked (the tests), every index is bounds-checked and every wait is skewed by a
// random per-block delay, so a race shows on every run.
// =================================================================================

template <typename T, int ngpus>
class Comm {
 public:
  using V = typename traits<T>::V;

  DINLINE explicit Comm(Peers p) : p_(p) {
    // ROTATED by rank, so the ranks do not all read rank 0 first. Each rank then sums in
    // a different order, so one-shot outputs agree to one ULP of T, not bitwise.
#pragma unroll
    for (int i = 0; i < ngpus; ++i) {
      in_[i]      = reinterpret_cast<const V*>(p.inputs_->p[(p.rank_ + i) % ngpus]);
      scratch_[i] = reinterpret_cast<V*>(p.signals_.s[i] + 1);
    }
    skew();
    pair_blocks<false>(true);
  }

  DINLINE int rank() const { return p_.rank_; }

  DINLINE V sum(int64_t idx) const {
    V out[1];
    sum<1>(idx, 0, idx + 1, out);
    return out[0];
  }

  // kBatch packs at once, idx + u * stride for u < kBatch (those at or past `limit` are
  // skipped): every load from every peer is issued before any is added, so kBatch x ngpus
  // are in flight rather than ngpus.
  template <int kBatch>
  DINLINE void sum(int64_t idx, int64_t stride, int64_t limit, V (&out)[kBatch]) const {
    constexpr int N = traits<T>::N;
    V raw[kBatch][ngpus];
#pragma unroll
    for (int u = 0; u < kBatch; ++u) {
      const int64_t at = idx + u * stride;
      if (at < limit) {
        check(at < p_.input_packs_, "sum", -1, at, p_.input_packs_);
#pragma unroll
        for (int i = 0; i < ngpus; ++i) raw[u][i] = load_global(in_[i] + at);
      }
    }
#pragma unroll
    for (int u = 0; u < kBatch; ++u) {
      float acc[N];
#pragma unroll
      for (int j = 0; j < N; ++j) acc[j] = static_cast<float>(raw[u][0].d[j]);
#pragma unroll
      for (int i = 1; i < ngpus; ++i)
#pragma unroll
        for (int j = 0; j < N; ++j) acc[j] += static_cast<float>(raw[u][i].d[j]);
#pragma unroll
      for (int j = 0; j < N; ++j) out[u].d[j] = static_cast<T>(acc[j]);
    }
  }

  // This rank's own input pack: the quantized two-shot encodes it before sending it.
  DINLINE V mine(int64_t idx) const {
    check(idx < p_.input_packs_, "mine", -1, idx, p_.input_packs_);
    return load_global(in_[0] + idx);
  }

  // A float in peer's scratch, `idx` in floats: the quantized two-shot's scales.
  DINLINE void put(int peer, int64_t idx, float v) const {
    check(peer >= 0 && peer < ngpus && idx < 4 * p_.scratch_packs_, "put", peer, idx,
          4 * p_.scratch_packs_);
    __scoped_atomic_store_n(reinterpret_cast<float*>(scratch_of(peer)) + idx, v,
                            __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }

  DINLINE float get_float(int peer, int64_t idx) const {
    check(peer >= 0 && peer < ngpus && idx < 4 * p_.scratch_packs_, "get", peer, idx,
          4 * p_.scratch_packs_);
    return __scoped_atomic_load_n(reinterpret_cast<const float*>(scratch_of(peer)) + idx,
                                  __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }

  DINLINE void put(int peer, int64_t idx, const V& v) const {
    check(peer >= 0 && peer < ngpus && idx < p_.scratch_packs_, "put", peer, idx,
          p_.scratch_packs_);
    store_global(scratch_of(peer) + idx, v);
  }

  DINLINE V get(int peer, int64_t idx) const {
    check(peer >= 0 && peer < ngpus && idx < p_.scratch_packs_, "get", peer, idx,
          p_.scratch_packs_);
    return load_global(scratch_of(peer) + idx);
  }

  // SHMEM's `shmem_ptr`: a hot loop reads through this rather than paying `get`'s check on
  // every load, which keeps the loads free to issue back to back.
  DINLINE const V* ptr(int peer, int64_t idx, int64_t n) const {
    check(peer >= 0 && peer < ngpus && idx + n <= p_.scratch_packs_, "ptr", peer, idx + n,
          p_.scratch_packs_);
    return scratch_of(peer) + idx;
  }

  DINLINE void world_barrier() { barrier<true>(); }

  // Every block of THIS rank's kernel: puts to our own scratch before it are visible to our
  // own gets after it. Cheaper than `world_barrier`, and wrong for anything a peer put.
  DINLINE void grid_barrier() { barrier<false>(); }

  // The threads of this block.
  DINLINE void block_barrier() const { __syncthreads(); }

  // THE FLAT REDUCE: positions [begin, end) summed over ranks, grid-strided, kSumBatch
  // packs a thread all loaded before any is stored. store(position, v). A gather_flat after
  // a peer_block_barrier reads back exactly these positions, thread for thread.
  template <typename Store>
  DINLINE void reduce_flat(int begin, int end, Store store) const {
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int idx = begin + tid; idx < end; idx += stride * kSumBatch) {
      V v[kSumBatch];
      sum<kSumBatch>(idx, stride, end, v);
#pragma unroll
      for (int u = 0; u < kSumBatch; ++u)
        if (idx + u * stride < end) store(idx + u * stride, v[u]);
    }
  }

  // THE TWO-SHOT GATHERS, after a peer_block_barrier: each reads back exactly what the
  // same-numbered block on every peer put, every peer at once (index outer, peer inner,
  // so a load is in flight on every link), and hands each pack to `store`.
  //
  // gather_flat: a flat buffer sliced `chunk` packs per rank; each thread takes the
  // positions tid, tid + grid, ... it reduced. store(position in the whole buffer, v).
  template <typename Store>
  DINLINE void gather_flat(int chunk, int size, Store store) const {
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int k = tid; k < chunk; k += stride) {
      V g[ngpus];
#pragma unroll
      for (int i = 0; i < ngpus; ++i)
        if (i * chunk + k < size) g[i] = get(i, k);
#pragma unroll
      for (int i = 0; i < ngpus; ++i)
        if (i * chunk + k < size) store(i * chunk + k, g[i]);
    }
  }

  // gather_rows: `chunk` whole rows per rank; block b takes local rows b, b + grid, ...
  // as it reduced them. kRegions regions, region_packs apart in the scratch.
  // store(region, row, pack within the row, v).
  template <int kRegions, typename Store>
  DINLINE void gather_rows(int chunk, int rows, int packs, int region_packs,
                           Store store) const {
    for (int lr = blockIdx.x; lr < chunk; lr += gridDim.x) {
      for (int k = threadIdx.x; k < packs; k += blockDim.x) {
        const int at = lr * packs + k;
        V g[kRegions][ngpus];
#pragma unroll
        for (int i = 0; i < ngpus; ++i)
          if (i * chunk + lr < rows)
#pragma unroll
            for (int r = 0; r < kRegions; ++r)
              g[r][i] = get(i, r * region_packs + at);
#pragma unroll
        for (int i = 0; i < ngpus; ++i)
          if (i * chunk + lr < rows)
#pragma unroll
            for (int r = 0; r < kRegions; ++r)
              store(r, i * chunk + lr, k, g[r][i]);
      }
    }
  }

  // This block and the same-numbered block on every peer, and no other block: one peer
  // write each, where `world_barrier` waits for the whole grid. A get after it may read
  // ONLY what the same-numbered block on that peer put, so both phases must give each
  // block the same indices (vLLM's custom all-reduce, and the rule its two-stage kernel
  // states).
  DINLINE void peer_block_barrier() {
    skew();
    pair_blocks<true>(false);
  }

  DINLINE void close() {
    skew();
    pair_blocks<false>(false);
  }

 private:
  // The grid on this device, then (kPeers) one exchange with the peers by the last block
  // to arrive, then the grid released. ONE FENCE PER BLOCK, by thread 0 after the block
  // barrier: the barrier waits for every wave's stores, and a release writes back the whole
  // L2, so one covers the block where one per thread wrote it back 512 times.
  template <bool kPeers>
  DINLINE void barrier() {
    constexpr int kScope = kPeers ? __MEMORY_SCOPE_SYSTEM : __MEMORY_SCOPE_DEVICE;
    skew();
    __syncthreads();
    if (threadIdx.x == 0) {
      fence<__ATOMIC_RELEASE, kScope>();
      Signal* self     = p_.self_;
      const uint32_t g = __scoped_atomic_load_n(&self->gen, __ATOMIC_ACQUIRE, kScope);
      if (__scoped_atomic_fetch_add(&self->arrive, 1u, __ATOMIC_ACQ_REL,
                                    __MEMORY_SCOPE_DEVICE) == gridDim.x - 1) {
        __scoped_atomic_store_n(&self->arrive, 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
        if constexpr (kPeers) {
          const uint32_t e = self->epoch + 1;
          self->epoch      = e;
          fence<__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM>();
#pragma unroll
          for (int i = 0; i < ngpus; ++i)
            __scoped_atomic_store_n(&p_.signals_.s[i]->peer[p_.rank_], e, __ATOMIC_RELAXED,
                                    __MEMORY_SCOPE_SYSTEM);
#pragma unroll
          for (int i = 0; i < ngpus; ++i)
            wait<true, __MEMORY_SCOPE_SYSTEM>(&self->peer[i], e, "world_barrier: peer", i);
        }
        __scoped_atomic_fetch_add(&self->gen, 1u, __ATOMIC_RELEASE, kScope);
      } else {
        wait<true, kScope>(&self->gen, g + 1, kPeers ? "world_barrier" : "grid_barrier",
                           -1);
      }
    }
    __syncthreads();
  }

  // The builtin takes the ordering as a literal, so each case is spelled out.
  template <int kOrder, int kScope>
  DINLINE void fence() const {
    constexpr bool system = kScope == __MEMORY_SCOPE_SYSTEM;
    if constexpr (kOrder == __ATOMIC_RELEASE) {
      if constexpr (system) __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
      else __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    } else {
      if constexpr (system) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
      else __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
  }

  // BY SELECT, NOT `scratch_[peer]`: a runtime index into a register array moves the
  // array to scratch memory, and every put and get would load its pointer from there
  // first.
  DINLINE V* scratch_of(int peer) const {
    V* at = scratch_[0];
#pragma unroll
    for (int i = 1; i < ngpus; ++i)
      if (peer == i) at = scratch_[i];
    return at;
  }

  // Block b waits for block b on every rank, and for no other block: enough at the ends,
  // where it says "every peer has launched" or "every peer is done reading me", and not
  // enough between phases, which is what `world_barrier` is for. kOrdered: the store
  // releases and the wait acquires, so what the block put before is visible to its peers'
  // same-numbered block after (a peer_block_barrier). Unordered, it only says when
  // (start: every peer has launched; close: every peer is done reading us).
  template <bool kOrdered>
  DINLINE void pair_blocks(bool start) const {
    if (!start) __syncthreads();
    Signal* self     = p_.self_;
    const uint32_t f = self->seq[blockIdx.x] + 1;
    if (threadIdx.x < ngpus) {
      uint32_t* theirs = start ? &p_.signals_.s[threadIdx.x]->start[blockIdx.x][p_.rank_]
                               : &p_.signals_.s[threadIdx.x]->end[blockIdx.x][p_.rank_];
      uint32_t* mine   = start ? &self->start[blockIdx.x][threadIdx.x]
                               : &self->end[blockIdx.x][threadIdx.x];
      __scoped_atomic_store_n(theirs, f, kOrdered ? __ATOMIC_RELEASE : __ATOMIC_RELAXED,
                              __MEMORY_SCOPE_SYSTEM);
      wait<kOrdered, __MEMORY_SCOPE_DEVICE>(mine, f, start ? "start" : "peer barrier",
                                            threadIdx.x);
    }
    __syncthreads();
    if (threadIdx.x == 0) self->seq[blockIdx.x] = f;
  }

  // Spins relaxed and acquires once, after: an acquire per poll would invalidate the
  // caches on every iteration of every spinning block.
  template <bool kAcquire, int kScope>
  DINLINE void wait(const uint32_t* flag, uint32_t want, const char* what, int peer) const {
    const uint64_t t0 = wall_clock64();
    uint32_t seen;
    while ((seen = __scoped_atomic_load_n(flag, __ATOMIC_RELAXED, kScope)) < want) {
      if (wall_clock64() - t0 > p_.timeout_ticks_) {
        printf("rocm_comms: rank %d block %d timed out in %s, peer %d: flag %u, want %u\n",
               p_.rank_, blockIdx.x, what, peer, seen, want);
        __builtin_trap();
      }
    }
    if constexpr (kAcquire) fence<__ATOMIC_ACQUIRE, kScope>();
  }

  DINLINE void check(bool ok, const char* what, int peer, int64_t idx,
                     int64_t limit) const {
    if (p_.checked_ && (!ok || idx < 0)) {
      printf("rocm_comms: rank %d block %d thread %d: %s(peer %d, idx %lld) outside "
             "[0, %lld)\n",
             p_.rank_, blockIdx.x, threadIdx.x, what, peer, static_cast<long long>(idx),
             static_cast<long long>(limit));
      __builtin_trap();
    }
  }

  // Checked only: up to ~32 x 8K cycles, different per rank, block and call.
  DINLINE void skew() {
    ++calls_;
    if (!p_.checked_) return;
    uint32_t h = static_cast<uint32_t>(p_.rank_) * 73856093u ^ blockIdx.x * 19349663u ^
                 calls_ * 83492791u;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    for (uint32_t n = (h ^ (h >> 15)) % 32; n > 0; --n) __builtin_amdgcn_s_sleep(127);
  }

  const Peers p_;
  const V* in_[ngpus];
  V* scratch_[ngpus];
  uint32_t calls_ = 0;
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
        int64_t peer_slab_bytes, int64_t scratch_bytes, double sync_timeout_s)
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
    // The device wall clock is fixed-rate, in kHz; the kernels count the timeout in it.
    int device = 0, khz = 0;
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, device));
    timeout_ticks_ = static_cast<uint64_t>(sync_timeout_s * khz * 1000.0);
  }

  ~Group() {
    for (const auto& kv : opened_) hipIpcCloseMemHandle(kv.second);
  }

  int world_size() const { return world_size_; }
  // Bounds checks and random skew in every kernel launched after this. The tests' mode.
  void set_checked(bool checked) { checked_ = checked; }
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
  Peers peers(const torch::Tensor& input) {
    return Peers(slot_for(input.data_ptr()), signals_, self_signal_, rank_,
                 input.numel() * input.element_size() / 16, scratch_bytes_ / 16,
                 timeout_ticks_, checked_);
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
  uint64_t timeout_ticks_ = 0;
  bool checked_           = false;
  PeerSignals signals_{};
  PeerPtrs* slab_end_;
  PeerPtrs* cursor_;
  std::unordered_map<void*, PeerPtrs*> registered_;
  std::vector<void*> pending_;
  std::vector<PeerPtrs*> pending_slots_;
  std::unordered_map<std::string, void*> opened_;
};

}  // namespace hip_comms::ipc
