// SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc.
// All rights reserved.
//
// OPEN, a torch op: the communicator's state made, its peers reached.
#pragma once

#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>

#include "../../ops.h"
#include "../types.cuh"
#include "../utils.h"

// OPEN: the state of a communicator over the CPU group registered as `group`
// (this rank its rank, the world its size) on `device`, serving `max_rows` rows
// of at most `max_cols` columns, a wait giving up after `timeout_seconds`,
// every peer's regions opened: (the Error it is refused for, or none; the
// handle, 0 where refused; the device's arch). A device the build does not run
// on is refused, nothing allocated: the caller does not run here. A rank
// refused before the exchange leaves its peers to give up on it after
// `timeout_seconds` (exchange_failed); after it, every rank agrees.
OpenWire hip_custom_communicator_open(const std::string& group, int64_t device,
                                      int64_t max_rows, int64_t max_cols,
                                      double timeout_seconds) {
  using namespace hip_custom_communicator;
  using namespace hip_custom_communicator::internal;
  // ---- PRECONDITIONS: the group, the call's values, the world built, the
  // device the build's target with our kernels' code object on it (a kernel's
  // attributes answer exactly when it does). ----
  c10::intrusive_ptr<c10d::ProcessGroup> process_group;
  try {
    process_group = c10d::resolve_process_group(group);
  } catch (const std::exception&) {
    return {error_code(Error::no_such_group), 0, ""};
  }
  const int64_t rank = process_group->getRank();
  const int64_t world = process_group->getSize();
  for (const int64_t v : {rank, world, device})
    if (v < std::numeric_limits<int>::min() ||
        v > std::numeric_limits<int>::max())
      return {error_code(Error::field_out_of_range), 0, ""};
  if (rank < 0 || rank >= world ||
      std::find(kWorlds.begin(), kWorlds.end(), world) == kWorlds.end())
    return {error_code(Error::world_not_built), 0, ""};
  if (max_rows < 1 || max_cols < 1)
    return {error_code(Error::field_not_positive), 0, ""};
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, static_cast<int>(device)) != hipSuccess) {
    (void)hipGetLastError();
    return {error_code(Error::device_not_built), 0, ""};
  }
  const std::string name = prop.gcnArchName;
  const std::string arch =
      name.substr(0, name.find(':'));  // gfx950:sramecc+:xnack-
  if (hipSetDevice(static_cast<int>(device)) != hipSuccess)
    return {error_code(Error::hip_failed), 0, ""};
  // THE BUILD RUNS HERE where the device has our code object: a compiled
  // kernel's attributes answer exactly then.
  hipFuncAttributes attrs;
  const hipError_t found =
      hipFuncGetAttributes(&attrs, kAllReduceInstances[0].kernel);
  (void)hipGetLastError();
  if (found != hipSuccess) return {error_code(Error::device_not_built), 0, ""};
  int khz = 0, lds = 0, cus = 0, threads_a_cu = 0, wave = 0;
  size_t scratch = 0;
  if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate,
                            static_cast<int>(device)) != hipSuccess ||
      hipDeviceGetAttribute(&wave, hipDeviceAttributeWarpSize,
                            static_cast<int>(device)) != hipSuccess ||
      hipDeviceGetAttribute(&lds, hipDeviceAttributeMaxSharedMemoryPerBlock,
                            static_cast<int>(device)) != hipSuccess ||
      hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount,
                            static_cast<int>(device)) != hipSuccess ||
      hipDeviceGetAttribute(&threads_a_cu,
                            hipDeviceAttributeMaxThreadsPerMultiProcessor,
                            static_cast<int>(device)) != hipSuccess ||
      hipDeviceGetLimit(&scratch, hipExtLimitScratchMax) != hipSuccess)
    return {error_code(Error::hip_failed), 0, ""};
  hipUUID uuid{};
  if (hipDeviceGetUuid(&uuid, static_cast<int>(device)) != hipSuccess)
    return {error_code(Error::hip_failed), 0, ""};

  // ---- THE EXCHANGE, SIZED FROM THE PROBLEM (STANDARDS, *A kernel's own
  // buffers are sized from the problem it serves*): each region holds its
  // largest payload at the largest call, in every built dtype and either
  // encoding (exact or INT4: common.cuh's INT4 payload layout): the staging the
  // whole input, the pushed two-shot's whole-call inbox, or their INT4
  // payloads; the partials the most any two-shot keeps, or its INT4 payload.
  // Each grows with the rows and the row's width, so a call inside the box fits
  // by construction. Each a whole pack, at least one. ----
  const int64_t pack = kPackBytes;
  const auto whole = [pack](int64_t bytes) {
    return bytes < pack ? pack : (bytes + pack - 1) / pack * pack;
  };
  int64_t partials_bytes = 0,
          staging_bytes = int4_payload_bytes(max_rows, max_cols);
  for (const at::ScalarType d : kDtypes) {
    const int64_t e = static_cast<int64_t>(c10::elementSize(d));
    const int64_t packs = (max_cols * e + pack - 1) / pack;
    // THE MOST ANY TWO-SHOT KEEPS: a fused two-shot's whole sum, every row on
    // whole packs (the plain two-shot keeps a slice of every row, less), or its
    // INT4 payload.
    const int64_t partials = max_rows * packs * pack;
    partials_bytes =
        std::max(partials_bytes,
                 std::max(partials, int4_payload_bytes(1, partials / e)));
    staging_bytes = std::max(staging_bytes, max_rows * max_cols * e);
    // THE EXACT PUSHED TWO-SHOT'S INBOX, the whole call: WORLD slots of a share
    // each.
    staging_bytes = std::max(
        staging_bytes, world * ((max_rows * packs + world - 1) / world) * pack);
  }
  // THE TWO-SHOTS' INT4 INBOX (the flat walk): WORLD shares, each a rank's
  // whole groups of the call's, rounded up.
  const int64_t int4_share =
      ((max_rows * max_cols + kInt4Group - 1) / kInt4Group + world - 1) /
      world * kInt4Group;
  staging_bytes =
      std::max(staging_bytes, int4_payload_bytes(1, world * int4_share));
  partials_bytes = whole(partials_bytes);
  staging_bytes = whole(staging_bytes);

  // ---- THE LAYOUT, decided here, the same on every rank. ----
  // THE SIGNALS: two flag sets (a barrier's parity) of a slot a block by a slot
  // a writer, ONE SLOT A BLOCK THE DEVICE CAN HOLD RESIDENT (every CU full of
  // one-wave blocks), so the signals never rule out a grid; whether one
  // kernel's grid is resident is its occupancy (grid_not_resident).
  const int64_t max_blocks = int64_t{cus} * (threads_a_cu / wave);
  const int64_t signals_bytes =
      2 * max_blocks * world * int64_t{sizeof(uint32_t)};
  // THE UNFENCED REGION: the pushed one-shot's inbox, written with the flag in
  // the line and fenced by no barrier, so its own region (a stale word a fenced
  // kernel left could carry an epoch's value): [writer, bank * row] rows, two
  // banks of `unfenced_blocks` rows, a grid past them refused; a row
  // `unfenced_slot_bytes` of payload, its lines twice that (a line's flag
  // beside its payload). 128 blocks of 2 KiB: the one-shot's grids at the sizes
  // it serves (a tile a block, rows of up to 8192 bf16 values a line pack). Its
  // epochs are rank-local: a block's a pack (its own line), in int32 words.
  const int64_t unfenced_blocks = 128, unfenced_slot_bytes = 2048;
  const int64_t unfenced_epoch_words = kPackBytes / int64_t{sizeof(int32_t)};
  const int64_t unfenced_bytes = 2 * unfenced_blocks * world *
                                 unfenced_slot_bytes *
                                 (kLineBytes / kLinePayloadBytes);
  // THE INPUT TABLES' SLOTS, fixed here (no allocation during a capture): 2^18,
  // 16 MiB at 8 ranks (2^14 was too few for Kimi-K3 at c48).
  const int64_t input_table_slots = int64_t{1} << 18;
  // THE SCRATCH A KERNEL MAY ASK: the device's scratch over its every lane, by
  // 16. A dispatch that asks too much fails only when it runs, aborting the
  //    process; the share is a margin for the queues a process holds (16 full
  //    dispatches; 4 KiB a lane at gfx950).
  const int64_t scratch_shares = 16;

  // ---- THE STATE. ----
  auto* s = new CommsState();
  s->group = process_group;
  s->timeout_seconds = timeout_seconds;
  s->rank = static_cast<int>(rank);
  s->world = static_cast<int>(world);
  s->device.index = static_cast<int>(device);
  s->device.arch = arch;
  s->device.wave_size = wave;
  s->device.clock_khz = khz;
  s->max_rows = max_rows;
  s->timeout_ticks =
      static_cast<uint64_t>(timeout_seconds * s->device.clock_khz * 1000);
  s->device.lds_bytes = lds;
  s->device.compute_units = cus;
  s->device.threads_a_cu = threads_a_cu;
  s->device.scratch_a_lane = static_cast<int64_t>(scratch) /
                             (int64_t{cus} * threads_a_cu) / scratch_shares;
  SymmetricMemory& symmetric = s->symmetric;
  symmetric.max_blocks = max_blocks;
  symmetric.unfenced_blocks = unfenced_blocks;
  symmetric.unfenced_slot_bytes = unfenced_slot_bytes;
  symmetric.signals_bytes = signals_bytes;
  symmetric.staging_bytes = staging_bytes;
  symmetric.partials_bytes = partials_bytes;
  symmetric.scales_bytes = max_rows * int64_t{sizeof(float)};
  symmetric.unfenced_bytes = unfenced_bytes;
  s->unfenced_epoch_words = unfenced_epoch_words;
  s->input_tables.slots = input_table_slots;

  // ---- THE BUFFERS, each its own allocation, zeroed before any stream can use
  // it; freed by `free`. THE SYMMETRIC MEMORY'S REGIONS are
  // hipExtMallocWithFlags(hipDeviceMallocUncached), exported with
  // hipIpcGetMemHandle, opened by each peer with hipIpcOpenMemHandle: the
  // convention of every IPC communicator on ROCm (vLLM's custom all-reduce and
  // quick reduce, aiter's). Correct cached or not under the barrier's memory
  // model (common.cuh); uncached keeps peer reads off every L2. Not VMM:
  // exchanging more than ~256 MiB of VMM allocations across 8 processes left
  // the driver in an eviction storm on ROCm 10.0. A region holds either
  // encoding, sized for the larger: a call publishes exact or INT4, never both,
  // and every reuse is ordered by a barrier the previous call's readers passed.
  // The rest is this rank's alone (hipMalloc). ----
  bool made = true;
  const auto allocate = [&](auto*& buffer, int64_t bytes, bool exchanged) {
    void* p = nullptr;
    made =
        made &&
        (exchanged ? hipExtMallocWithFlags(&p, static_cast<size_t>(bytes),
                                           hipDeviceMallocUncached)
                   : hipMalloc(&p, static_cast<size_t>(bytes))) == hipSuccess &&
        hipMemset(p, 0, static_cast<size_t>(bytes)) == hipSuccess;
    buffer = static_cast<std::remove_reference_t<decltype(buffer)>>(p);
  };
  const int64_t ptr_array_bytes = world * int64_t{sizeof(char*)};
  allocate(symmetric.signals, symmetric.signals_bytes, true);
  allocate(symmetric.staging, symmetric.staging_bytes, true);
  allocate(symmetric.partials, symmetric.partials_bytes, true);
  allocate(symmetric.scales, symmetric.scales_bytes, true);
  allocate(symmetric.unfenced, symmetric.unfenced_bytes, true);
  allocate(symmetric.signals_ptr_array, ptr_array_bytes, false);
  allocate(symmetric.staging_ptr_array, ptr_array_bytes, false);
  allocate(symmetric.partials_ptr_array, ptr_array_bytes, false);
  allocate(symmetric.scales_ptr_array, ptr_array_bytes, false);
  allocate(symmetric.unfenced_ptr_array, ptr_array_bytes, false);
  allocate(s->unfenced_epochs,
           unfenced_blocks * unfenced_epoch_words * int64_t{sizeof(int32_t)},
           false);
  allocate(s->status_device, kStatusFields * int64_t{sizeof(uint32_t)}, false);
  allocate(s->input_tables.device,
           input_table_slots * world * int64_t{sizeof(void*)}, false);
  // WHAT THIS RANK SENDS ITS PEERS: its device's UUID, then its regions' IPC
  // handles in this order: the signals, the staging, the partials, the scales,
  // the unfenced region.
  constexpr int kRegions = 5;
  constexpr size_t kHandleBytes = sizeof(hipIpcMemHandle_t);
  std::string mine(reinterpret_cast<const char*>(&uuid), sizeof(uuid));
  char* const own[kRegions] = {reinterpret_cast<char*>(symmetric.signals),
                               symmetric.staging, symmetric.partials,
                               reinterpret_cast<char*>(symmetric.scales),
                               symmetric.unfenced};
  for (char* region : own) {
    hipIpcMemHandle_t h{};
    made = made && hipIpcGetMemHandle(&h, region) == hipSuccess;
    mine.append(reinterpret_cast<const char*>(&h), kHandleBytes);
  }
  if (!made || hipDeviceSynchronize() != hipSuccess) {
    (void)hipGetLastError();
    (void)hip_custom_communicator_close(reinterpret_cast<int64_t>(s), true);
    return {error_code(Error::hip_failed), 0, ""};
  }

  // ---- THE EXCHANGE, over the group: every rank's UUID and handles. ----
  const std::optional<std::vector<std::string>> everyone =
      all_gathered(*process_group, mine, timeout_seconds);
  if (!everyone) {
    (void)hip_custom_communicator_close(reinterpret_cast<int64_t>(s), true);
    return {error_code(Error::exchange_failed), 0, ""};
  }

  // ---- EVERY PEER REACHED: a peer's device this process sees (by its UUID)
  // reachable from ours, every peer's regions opened, each region's device
  // array written (every rank's address of it), and the eager input table,
  // slot 0: every rank's staging. The first failure is this rank's. ----
  std::optional<Error> failed;
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess) failed = Error::hip_failed;
  for (int r = 0; r < world && !failed; ++r)
    for (int d = 0; d < devices && r != rank && !failed; ++d) {
      hipUUID theirs{};
      int reachable = 0;
      if (d == device || hipDeviceGetUuid(&theirs, d) != hipSuccess ||
          std::memcmp(&theirs, (*everyone)[r].data(), sizeof(theirs)) != 0)
        continue;
      if (hipDeviceCanAccessPeer(&reachable, static_cast<int>(device), d) !=
              hipSuccess ||
          reachable == 0)
        failed = Error::peer_not_accessible;
    }
  const void* const arrays[kRegions] = {
      symmetric.signals_ptr_array, symmetric.staging_ptr_array,
      symmetric.partials_ptr_array, symmetric.scales_ptr_array,
      symmetric.unfenced_ptr_array};
  for (int region = 0; region < kRegions && !failed; ++region) {
    std::vector<char*> addresses(world);
    for (int r = 0; r < world && !failed; ++r) {
      if (r == rank) {
        addresses[r] = own[region];
        continue;
      }
      hipIpcMemHandle_t h;
      std::memcpy(&h,
                  (*everyone)[r].data() + sizeof(uuid) + region * kHandleBytes,
                  kHandleBytes);
      void* base = nullptr;
      if (hipIpcOpenMemHandle(&base, h, hipIpcMemLazyEnablePeerAccess) !=
          hipSuccess) {
        failed = Error::peers_not_opened;
        continue;
      }
      addresses[r] = static_cast<char*>(base);
      symmetric.peers.push_back(addresses[r]);
    }
    if (!failed &&
        hipMemcpy(const_cast<void*>(arrays[region]), addresses.data(),
                  world * sizeof(char*), hipMemcpyHostToDevice) != hipSuccess)
      failed = Error::hip_failed;
    if (!failed && region == 1 &&
        hipMemcpy(s->input_tables.device, addresses.data(),
                  world * sizeof(char*), hipMemcpyHostToDevice) != hipSuccess)
      failed = Error::hip_failed;
  }
  (void)hipGetLastError();

  // ---- EVERY RANK OPENED, or every rank closes (a flag a rank, over the
  // group): this rank's failure, else a peer's (ranks_disagree). ----
  const std::optional<std::vector<std::string>> opened = all_gathered(
      *process_group, std::string(1, failed ? '0' : '1'), timeout_seconds);
  const bool everyone_opened =
      opened && std::count(opened->begin(), opened->end(), "1") == world;
  if (!everyone_opened) {
    (void)hip_custom_communicator_close(reinterpret_cast<int64_t>(s), !opened);
    return {error_code(failed    ? *failed
                       : !opened ? Error::exchange_failed
                                 : Error::ranks_disagree),
            0, ""};
  }

  // ---- THE ANSWER: the handle, the arch. ----
  return {std::nullopt, reinterpret_cast<int64_t>(s), s->device.arch};
}
