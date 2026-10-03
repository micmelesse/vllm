// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ROCM_COMMS, THE ONE INTERFACE: our collectives, torch-free, and an index of their parts. An OP is
// what a caller asks for (an API call); a KERNEL is what runs, one compiled instruction sequence
// (vllm CONTEXT's lingo). Every op is select (everything decided: the op's launch, or the first
// Error the call meets) then launch (which decides nothing).
//
// What a build is:
//   error.cuh       Error: every reason a call cannot run here
//   build.cuh       kBuild: what is compiled, the memory, the kernels' geometry (compile time)
//   handle.cuh      Handle: the peers' memory, mapped once (run time; kBuild's twin), and
//                   supported(device, world): whether one can exist there
//   types.cuh       the vocabulary: Template, each family's KernelConfig, OpType, Algorithm,
//                   Direction, and each op's launch (the normal form select returns)
// How a call runs:
//   launch.cuh      what is built (each template's configs), a launch's compiled instance, and
//                   each op's launch_<op>
//   select.cuh      what was tuned (each op's kernels), and the steps every select takes: forced or
//                   picked, fitted, checked
//   interface.cuh   THE API: each op's select_<op>, and the op (select then launch)

#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>

#include "p2p/p2p.cuh"
#include "build.cuh"
#include "error.cuh"
#include "handle.cuh"
#include "types.cuh"
#include "machine/hardware.cuh"

#define HIP_COMMS_INTERFACE
#include "launch.cuh"
#include "select.cuh"
#include "interface.cuh"
#undef HIP_COMMS_INTERFACE
