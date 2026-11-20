#pragma once

// Opens the communicator: allocates its buffers and maps every peer's over IPC.

#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>

#include "../../ops.h"
#include "../types.cuh"
#include "../utils.h"

OpenWire hip_custom_communicator_open(const std::string& group, int64_t device,
                                      int64_t max_rows, int64_t max_cols,
                                      double timeout_seconds) {
  using namespace hip_custom_communicator;
  using namespace hip_custom_communicator::internal;

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
  const std::string arch = name.substr(0, name.find(':'));
  if (hipSetDevice(static_cast<int>(device)) != hipSuccess)
    return {error_code(Error::hip_failed), 0, ""};

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

  const int64_t max_blocks = int64_t{cus} * (threads_a_cu / wave);
  // simple_write's counters, then simple_read's, the same shape.
  const int64_t signals_bytes =
      2 * 2 * max_blocks * world * int64_t{sizeof(uint32_t)};

  const int64_t unfenced_blocks = 128, unfenced_slot_bytes = 2048;
  const int64_t unfenced_bytes = 2 * unfenced_blocks * world *
                                 unfenced_slot_bytes *
                                 (kLineBytes / kLinePayloadBytes);
  // Lamport's lines are all data, so its inbox affords more rows: a block row
  // each for a grid that covers 64 tokens of 7168 at a tile of 1024.
  const int64_t lamport_blocks = 1024;
  const int64_t lamport_bytes =
      kLamportBanks * lamport_blocks * world * unfenced_slot_bytes;

  const int64_t scratch_shares = 16;

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
  // A checked build gives up within seconds, so a hang is a crash that says
  // where it waited, not a stall until the process is killed.
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG)
    s->timeout_ticks = std::min<uint64_t>(
        s->timeout_ticks,
        static_cast<uint64_t>(5.0 * s->device.clock_khz * 1000));
  s->device.lds_bytes = lds;
  s->device.compute_units = cus;
  s->device.threads_a_cu = threads_a_cu;
  s->device.scratch_a_lane = static_cast<int64_t>(scratch) /
                             (int64_t{cus} * threads_a_cu) / scratch_shares;
  SymmetricMemory& symmetric = s->symmetric;
  symmetric.max_blocks = max_blocks;
  symmetric.unfenced_blocks = unfenced_blocks;
  symmetric.unfenced_slot_bytes = unfenced_slot_bytes;
  symmetric.lamport_blocks = lamport_blocks;
  symmetric.signals_bytes = signals_bytes;
  symmetric.unfenced_bytes = unfenced_bytes;
  symmetric.lamport_bytes = lamport_bytes;
  symmetric.inbox_bytes = lamport_blocks * world * unfenced_slot_bytes;
  s->step_blocks = lamport_blocks;
  s->step_waves = kStepWaves;

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
  allocate(symmetric.unfenced, symmetric.unfenced_bytes, true);
  allocate(symmetric.lamport, symmetric.lamport_bytes, true);
  allocate(symmetric.simple_write_inbox, symmetric.inbox_bytes, true);
  allocate(symmetric.simple_read_inbox, symmetric.inbox_bytes, true);
  made = made &&
         hipMemset(symmetric.lamport, kLamportEmpty & 0xff,
                   static_cast<size_t>(symmetric.lamport_bytes)) == hipSuccess;
  // A checked build fills the Simple protocols' inboxes with 0xff, a NaN in
  // bf16 and fp16 and in an INT4 group's scale, where zeros would read as
  // finite data: a read of what no rank wrote fails a check or a test on its
  // first call. The signals, the steps and the other inboxes keep the
  // start their protocols need.
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG)
    made =
        made &&
        hipMemset(symmetric.simple_write_inbox, 0xff,
                  static_cast<size_t>(symmetric.inbox_bytes)) == hipSuccess &&
        hipMemset(symmetric.simple_read_inbox, 0xff,
                  static_cast<size_t>(symmetric.inbox_bytes)) == hipSuccess;
  allocate(symmetric.signals_ptr_array, ptr_array_bytes, false);
  allocate(symmetric.simple_write_counters_ptr_array, ptr_array_bytes, false);
  allocate(symmetric.simple_read_counters_ptr_array, ptr_array_bytes, false);
  allocate(symmetric.unfenced_ptr_array, ptr_array_bytes, false);
  allocate(symmetric.lamport_ptr_array, ptr_array_bytes, false);
  allocate(symmetric.simple_write_inbox_ptr_array, ptr_array_bytes, false);
  allocate(symmetric.simple_read_inbox_ptr_array, ptr_array_bytes, false);
  allocate(s->steps,
           4 * lamport_blocks * kStepWaves * int64_t{sizeof(uint32_t)}, false);
  allocate(s->status_device, kStatusFields * int64_t{sizeof(uint32_t)}, false);

  constexpr int kRegions = 5;
  constexpr size_t kHandleBytes = sizeof(hipIpcMemHandle_t);
  std::string mine(reinterpret_cast<const char*>(&uuid), sizeof(uuid));
  char* const own[kRegions] = {reinterpret_cast<char*>(symmetric.signals),
                               symmetric.unfenced, symmetric.lamport,
                               symmetric.simple_write_inbox,
                               symmetric.simple_read_inbox};
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

  const std::optional<std::vector<std::string>> everyone =
      all_gathered(*process_group, mine, timeout_seconds);
  if (!everyone) {
    (void)hip_custom_communicator_close(reinterpret_cast<int64_t>(s), true);
    return {error_code(Error::exchange_failed), 0, ""};
  }

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
      symmetric.signals_ptr_array, symmetric.unfenced_ptr_array,
      symmetric.lamport_ptr_array, symmetric.simple_write_inbox_ptr_array,
      symmetric.simple_read_inbox_ptr_array};
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
    if (!failed && region == 0) {
      // Each Simple protocol's counters, a region each.
      uint32_t* const* const tables[2] = {
          symmetric.simple_write_counters_ptr_array,
          symmetric.simple_read_counters_ptr_array};
      for (int t = 0; t < 2 && !failed; ++t) {
        std::vector<uint32_t*> counters(world);
        for (int r = 0; r < world; ++r)
          counters[r] = reinterpret_cast<uint32_t*>(addresses[r]) +
                        t * 2 * max_blocks * world;
        if (hipMemcpy(const_cast<uint32_t**>(tables[t]), counters.data(),
                      world * sizeof(uint32_t*),
                      hipMemcpyHostToDevice) != hipSuccess)
          failed = Error::hip_failed;
      }
    }
  }
  (void)hipGetLastError();

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

  return {std::nullopt, reinterpret_cast<int64_t>(s), s->device.arch};
}

// For the tests: views of the state the protocols keep from launch to launch,
// this rank's Lamport inbox (bytes) and the steps ([protocol][block][wave],
// STEPS), so a test can check what every launch must leave: the inbox all the
// sentinel, the steps equal on every rank. No copy; the
// views live as long as the communicator.
std::vector<torch::Tensor> hip_custom_communicator_inspect(int64_t state) {
  using namespace hip_custom_communicator;
  using namespace hip_custom_communicator::internal;
  TORCH_CHECK(state != 0, "hip_custom_communicator_inspect: not open");
  CommsState& s = *reinterpret_cast<CommsState*>(state);
  const auto on = torch::TensorOptions().device(torch::kCUDA, s.device.index);
  return {
      torch::from_blob(s.symmetric.lamport, {s.symmetric.lamport_bytes},
                       on.dtype(torch::kUInt8)),
      torch::from_blob(s.steps, {4, s.step_blocks, s.step_waves},
                       on.dtype(torch::kInt32)),
  };
}
