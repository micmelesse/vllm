// SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc.
// All rights reserved.
//
// THE HELPERS hip_custom_communicator.cu's API shares: internal, not the API.
#pragma once

#include <cstdint>
#include <optional>

namespace hip_custom_communicator::internal {

// AN ERROR AS A TORCH OP RETURNS IT: the Error's number (a torch op returns no
// enum; none, std::nullopt, is a call that ran).
template <typename ERROR>
std::optional<int64_t> error_code(ERROR error) {
  return static_cast<int64_t>(error);
}

}  // namespace hip_custom_communicator::internal
