// SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc.
// All rights reserved.
//
// THE HELPERS hip_custom_communicator.cu's API shares: internal, not the API.
#pragma once

#include <torch/all.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace hip_custom_communicator::internal {

// AN ERROR AS A TORCH OP RETURNS IT: the Error's number (a torch op returns no
// enum; none, std::nullopt, is a call that ran).
template <typename ERROR>
std::optional<int64_t> error_code(ERROR error) {
  return static_cast<int64_t>(error);
}

// EVERY RANK'S `mine`, in rank order, over `group` (the CPU group), or none
// where a rank did not arrive within `timeout_seconds` or the group failed
// (c10d raises; caught here, as C++ raises nothing). `mine` is the same length
// on every rank: gloo's allgather moves equal lengths.
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

}  // namespace hip_custom_communicator::internal
