// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE BUILD: every number fixed at compile time, derived from the device's `Hardware`
// (hardware.cuh) and nothing else; what depends on the input is select's (select.cuh), at run
// time. A number neither gives is a policy: one named line in `derive`.

#pragma once

#include <array>
#include <cstdint>

#include "hardware.cuh"

namespace hip_comms {

// A TENSOR'S ELEMENT TYPE, as the build names it. f32 is a norm weight's, not a call's.
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

// EVERYTHING THE BUILD FIXES, in one value: what is compiled, the memory and its unit, and the
// kernels' launch geometry. Python reads a projection of it (`build_info`).
struct BuildInfo {
  struct Supports {
    std::array<DType, 2> dtypes;  // a call's; dispatch instantiates exactly these
    std::array<int, 3> worlds;
  };
  struct Memory {
    int pack_bytes;          // a pack: the unit every kernel loads, sums and stores in
    int64_t staging_bytes;   // where an eager input is copied for its peers
    int64_t scratch_bytes;   // a two-shot's partial sums, per rank
    int64_t peer_ptr_slots;  // one per registered or captured buffer: its peers' addresses
  };
  struct Kernels {
    int max_threads;              // the widest block, and every kernel's __launch_bounds__
    int max_waves;                // its waves
    double sync_timeout_seconds;  // how long a kernel waits on a peer before it traps
  };
  Supports supports;
  Memory memory;
  Kernels kernels;
};

constexpr BuildInfo derive(const Hardware& hw) {
  BuildInfo info{};
  // WHAT IS COMPILED, one list each: dispatch instantiates exactly these, check refuses the rest.
  info.supports = {{DType::f16, DType::bf16}, {2, 4, 8}};

  BuildInfo::Memory& m = info.memory;
  // A PACK IS THE WIDEST LOAD, as in vLLM's and aiter's custom all-reduce and NCCL.
  m.pack_bytes = hw.max_load_bytes;
  // THE STAGING an eager input is copied into for its peers: policy. A staged build takes any
  // size through it a pass at a time; an in-place build on an eager input (the fused ops) needs it
  // whole, up to Kimi-K3's widest row (the one-all-reduce tail's 4096 x 17920 bf16, 147 MB).
  m.staging_bytes = 256 * kMiB;
  // THE TWO-SHOT SCRATCH, per rank, after the signal block: policy. It holds one rank's slice, so
  // it caps a two-shot buffer at scratch_bytes x ngpus (1 GiB at 8 ranks); a quantized two-shot
  // holds every rank's slice at half width, padded to whole grid strides (Kimi-K3's 4096 x 7168
  // bf16 prefill needs 74 MB at 36 blocks). check refuses a call past it (scratch_too_small).
  m.scratch_bytes = 128 * kMiB;
  // THE PEER-POINTER SLOTS, one per captured launch (capture sizes x collectives a forward): the
  // size vLLM gives the same array, 8 MB of slots. Registration past it fails at capture.
  m.peer_ptr_slots = 131072;

  BuildInfo::Kernels& b = info.kernels;

  // THE WIDEST BLOCK WHOSE THREADS KEEP EVERY ARCHITECTURAL REGISTER. A bound is a register trade:
  // at 1024 threads a gfx950 thread gets 128 and the row kernels spill (AttnRes at 2 packs, the
  // GEMM tail at any; ISA 2026-09-29T23-48-40Z).
  b.max_threads = hw.wave_size;
  for (int t = hw.wave_size; t <= hw.max_workgroup; t += hw.wave_size)
    if (vgprs_per_thread(hw, t) >= hw.arch_vgprs) b.max_threads = t;
  b.max_waves = b.max_threads / hw.wave_size;

  // HOW LONG A KERNEL WAITS ON A PEER before it prints where it was and traps.
  b.sync_timeout_seconds = 10.0;
  return info;
}

// THIS COMPILE PASS'S BUILD: the device code for its own target, the host for the tuning target.
constexpr BuildInfo kBuild = derive(kDevice);

// WHETHER THE BUILD HOLDS a dtype or a world: dispatch instantiates exactly these.
template <typename T, size_t N>
constexpr bool built_in(const std::array<T, N>& built, T x) {
  for (const T& b : built)
    if (b == x) return true;
  return false;
}
constexpr bool dtype_built(DType d) { return built_in(kBuild.supports.dtypes, d); }
constexpr bool world_built(int world) { return built_in(kBuild.supports.worlds, world); }

static_assert(kBuild.kernels.max_threads <= kDevice.max_workgroup &&
                  kBuild.kernels.max_threads % kWaveSize == 0,
              "the block limit must be whole waves the device can launch");

}  // namespace hip_comms
