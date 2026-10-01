// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// A KERNEL'S TWO BUILDS, in place and staged (`kStaged`): what the staged build takes beyond the
// in-place build's arguments, appended last and empty in place, so the in-place build's arguments
// and instructions are what they were before it had a staged twin (ISA 2026-10-01T21-01-19Z).

#pragma once

#include <cstdint>
#include <type_traits>

namespace hip_comms {

// A staged build's own input (it copies it into its staging a pass at a time) and the packs a
// staging holds; nothing in place.
template <typename T, bool kStaged>
struct Staged {};

template <typename T>
struct Staged<T, true> {
  const T* own_input;
  int64_t stage_packs;
};

// The packs a build counts: 64-bit staged (any size), the in-place build's int.
template <bool kStaged>
using PackCount = std::conditional_t<kStaged, int64_t, int>;

}  // namespace hip_comms
