// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ROCM_COMMS, THE ONE INTERFACE: our collectives, torch-free, and an index of their parts. An OP is
// what a caller asks for (an API call); a KERNEL is what runs, one compiled instruction sequence
// (vllm CONTEXT's lingo). Every op is plan (select's kernel, or the first Error check meets),
// then launch (which decides nothing); it returns the kernel it launched, or the Error and
// launches nothing.
//
// What a build is, and what runs:
//   error.cuh       Error: every reason a call cannot run here
//   build.cuh       kBuild: what is compiled, the memory, the kernels' geometry (compile time)
//   handle.cuh      Handle: the peers' memory, mapped once (run time; kBuild's twin), and
//                   supported(device, world): whether one can exist there
//   kernel.cuh      Kernel: a Template, its arguments, its family's KernelConfig (LaunchConfig and
//                   the family's fields)
//   args.cuh        each op's call (AllReduceArgs ... ScaleAddArgs), and the Options it runs under
// How a call becomes a kernel:
//   op.cuh          the ops: the template catalog (each Template's built KernelConfigs), and each
//                   op's name, templates, tuned kernels, and entry point
//   select.cuh      select: the op's tuned kernel for the call (or the forced one), fitted to it
//   dispatch.cuh    a Kernel to its compiled instance
//   check.cuh       check and plan: the first Error a kernel meets on a call, or none
//   launch.cuh      launch: the kernel on the stream

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
#include "kernel.cuh"
#include "args.cuh"
#include "machine/hardware.cuh"

#define HIP_COMMS_INTERFACE
#include "op.cuh"
#include "select.cuh"
#include "dispatch.cuh"
#include "check.cuh"
#include "launch.cuh"
#undef HIP_COMMS_INTERFACE
