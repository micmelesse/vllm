#pragma once

#include <torch/all.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "types.cuh"

namespace hip_custom_communicator::internal {

template <typename ERROR>
std::optional<int64_t> error_code(ERROR error) {
  return static_cast<int64_t>(error);
}

inline std::optional<std::vector<std::string>> all_gathered(
    c10d::ProcessGroup& group, const std::string& mine,
    double timeout_seconds) {
  try {
    const auto bytes = torch::TensorOptions().dtype(torch::kUInt8);
    std::vector<at::Tensor> in{
        torch::empty({static_cast<int64_t>(mine.size())}, bytes)};
    std::memcpy(in[0].data_ptr(), mine.data(), mine.size());
    std::vector<std::vector<at::Tensor>> out(1);
    for (int r = 0; r < group.getSize(); ++r)
      out[0].push_back(torch::empty_like(in[0]));
    if (!group.allgather(out, in)->wait(std::chrono::milliseconds(
            static_cast<int64_t>(timeout_seconds * 1000.0))))
      return std::nullopt;
    std::vector<std::string> everyone;
    for (const at::Tensor& t : out[0])
      everyone.emplace_back(static_cast<const char*>(t.data_ptr()),
                            static_cast<size_t>(t.numel()));
    return everyone;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

// A launch's protocol buffers (ProtocolBuffers, kernels/common/types.cuh): the
// protocol's inbox laid out for a message of `rows` rows of `row_elems`
// elements of `elem_bytes` each (a block's rows are its own), its counters and
// its steps; none when the inbox cannot hold the message. Each protocol
// has one layout. LL's and Lamport's are fixed lines, a row a block, so a row
// keeps its place from launch to launch (their arrival signal is in the
// data); the Simple protocols' are packed, a slot a writer, a row a block.
inline std::optional<ProtocolBuffers> layout_protocol_buffers(
    const CommsState& s, Protocol protocol, int64_t rows, int64_t row_elems,
    int64_t elem_bytes, int64_t pack_bytes, bool quantize_int4) {
  const SymmetricMemory& m = s.symmetric;
  const int64_t e = elem_bytes;
  int64_t stride_slot = 0, stride_m = 0, size_m = 0, size_n = 0;
  if (protocol == Protocol::ll || protocol == Protocol::lamport) {
    const bool ll = protocol == Protocol::ll;
    if (ll) {
      // An LL line is half data.
      const int64_t line_pack =
          std::min<int64_t>(pack_bytes, kLinePayloadBytes);
      const int64_t row_lines =
          quantize_int4
              ? (row_elems + kInt4Group - 1) / kInt4Group * kInt4GroupPacks
              : (row_elems * e + line_pack - 1) / line_pack;
      if (row_lines > m.unfenced_slot_bytes / kLinePayloadBytes ||
          rows > m.unfenced_blocks)
        return std::nullopt;
    } else if ((row_elems * e + pack_bytes - 1) / pack_bytes * pack_bytes >
                   m.unfenced_slot_bytes ||
               rows > m.lamport_blocks) {
      return std::nullopt;
    }
    const int64_t blocks = ll ? m.unfenced_blocks : m.lamport_blocks;
    const int64_t banks = ll ? kLlBanks : kLamportBanks;
    stride_slot = banks * blocks * (m.unfenced_slot_bytes / e);
    stride_m = m.unfenced_slot_bytes / e;
    size_m = banks * blocks, size_n = m.unfenced_slot_bytes / e;
  } else {
    if (s.world * rows * row_elems * e > m.inbox_bytes ||
        rows > s.step_blocks || rows > m.max_blocks)
      return std::nullopt;
    stride_slot = rows * row_elems, stride_m = row_elems;
    size_m = rows, size_n = row_elems;
  }
  char* const* const inbox_ptr_array =
      protocol == Protocol::ll             ? m.unfenced_ptr_array
      : protocol == Protocol::lamport      ? m.lamport_ptr_array
      : protocol == Protocol::simple_write ? m.simple_write_inbox_ptr_array
                                           : m.simple_read_inbox_ptr_array;
  uint32_t* const* const counters_ptr_array =
      protocol == Protocol::simple_read ? m.simple_read_counters_ptr_array
                                        : m.simple_write_counters_ptr_array;
  uint32_t* const steps =
      s.steps + static_cast<int>(protocol) * s.step_blocks * kStepWaves;
  return ProtocolBuffers{s.rank,
                         inbox_ptr_array,
                         s.world,
                         stride_slot,
                         stride_m,
                         int64_t{1},
                         s.world,
                         static_cast<int>(size_m),
                         static_cast<int>(size_n),
                         counters_ptr_array,
                         s.world,
                         m.max_blocks * s.world,
                         int64_t{s.world},
                         int64_t{1},
                         static_cast<int>(m.max_blocks),
                         s.world,
                         steps,
                         int64_t{kStepWaves},
                         int64_t{1},
                         static_cast<int>(s.step_blocks),
                         kStepWaves,
                         s.status_device,
                         int64_t{1},
                         kStatusFields,
                         s.timeout_ticks};
}

}  // namespace hip_custom_communicator::internal
