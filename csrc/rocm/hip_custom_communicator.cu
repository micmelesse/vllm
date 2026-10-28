// SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc.
// All rights reserved.
//
// Our HIP collectives as torch ops, built into `_rocm_C`: one file an API
// function, each a torch op bound in torch_bindings.cpp
// (hip_custom_communicator/api/), on the types they share
// (hip_custom_communicator/types.cuh), the helpers they share
// (hip_custom_communicator/utils.h) and the kernels
// (hip_custom_communicator/kernels/). Self-contained: torch and the HIP
// runtime, nothing from aiter.
//
// Nothing crosses the torch boundary but values and one handle: the
// communicator's state (CommsState, C++'s: its memory, its peers, its captured
// inputs), made by `open` and handed to Python as its address (an int, as
// vLLM's custom all-reduce hands its own), passed back to every call, and
// released by `close`. Open, register_graph and close are COLLECTIVES: every
// rank calls them, and they exchange over the state's CPU group themselves.
// EVERY ERROR IS A VALUE: no function raises; each returns the Error's number
// it was refused for (none: it ran), and Python names it.
//
// EACH OP TAKES ITS STATE, its tensors and values, then its config's fields as
// plain values, its op's own (an algorithm where its kernel has several, tile_k
// where it has K). Nothing is picked here: Python fills what a caller leaves
// out and hands its table's pick the same way; the op checks it, then runs it:
// its preconditions, its kernel's arguments, its launch, its postcondition,
// written out in its file.

// THE DECLARATIONS torch_bindings.cpp registers, so a definition that differs
// is a compile error, not a mismatch at run time.
#include "ops.h"

// THE STATE'S.
#include "hip_custom_communicator/api/open.cuh"
#include "hip_custom_communicator/api/register_graph.cuh"
#include "hip_custom_communicator/api/close.cuh"

// THE OPS.
#include "hip_custom_communicator/api/all_reduce.cuh"
#include "hip_custom_communicator/api/all_reduce_add_rms_norm.cuh"
#include "hip_custom_communicator/api/all_reduce_add_attn_res_rms_norm.cuh"
