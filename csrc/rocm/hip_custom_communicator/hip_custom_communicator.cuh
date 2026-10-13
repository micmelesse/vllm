// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// HIP_CUSTOM_COMMUNICATOR, THE ONE INTERFACE: our collectives, torch-free, and an index of their parts. An OP is
// what a caller asks for (an API call); a KERNEL is what runs, one compiled instruction sequence
// (vllm CONTEXT's lingo). Every op is select (everything decided: the op's launch, or the first
// Error the call meets) then launch (which decides nothing).
//
// What a build is:
//   common/         everything a kernel uses and the machine under it: the device (hardware.cuh:
//                   its Hardware, and the machine model over it), the build (build.cuh: kBuild,
//                   what is compiled for it, fixed at compile time), tiles, the peers' memory and
//                   barriers (common/interface.cuh lists it all)
//   handle.cuh      Handle: the peers' memory, mapped once (run time; kBuild's twin), and
//                   supported(device, world): whether one can exist there
//   types.cuh       the vocabulary: Error, Template, each op's BuildConfig, LaunchConfig and
//                   KernelConfig, OpType, Algorithm, Direction, each op's kernel signatures, and
//                   each op's launch (the normal form select returns)
// How a call runs:
//   select.cuh      what is built (each template's builds) and what was tuned (each op's
//                   kernels), and each op's select_<op>: forced whole or picked, its grid cut to
//                   the work, checked, its compiled kernel found; everything decided
//   launch.cuh      each op's launch_<op>: the launch's kernel run, nothing decided
//   interface.cuh   THE API: each op's select_<op>, and the op (select then launch)

#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>

#include "common/interface.cuh"
#include "handle.cuh"
#include "types.cuh"

#define HIP_CUSTOM_COMMUNICATOR_INTERFACE
#include "select.cuh"
#include "launch.cuh"
#include "interface.cuh"
#undef HIP_CUSTOM_COMMUNICATOR_INTERFACE
