// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// LAUNCH, NO DECISIONS: the peers' view of the input, then the kernel select named (its compiled
// function from dispatch.cuh) at its grid and block.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <hip/hip_runtime.h>

#include <tuple>

namespace hip_comms {

// The kernel at its grid and block, on the stream. hipify reads `<<<...>>>` as text, so it
// is spelled out.
template <typename... P, typename... A>
void start(void (*kernel)(P...), const Kernel& k, hipStream_t stream, A&&... args) {
  const LaunchConfig& l = launch_of(k.config);
  kernel<<<dim3(l.blocks_per_grid), dim3(l.threads_per_block), 0, stream>>>(
      std::forward<A>(args)...);
}

template <typename Args>
void launch_on(const Kernel& k, const Args& a, const p2p::DevComm& p, hipStream_t s) {
  dispatch(k, a, [&](auto kernel, const auto& bind) {
    std::apply([&](auto&&... xs) { start(kernel, k, s, xs...); }, bind(p));
  });
}

// A STAGED BUILD copies the input in itself, so no peer reads it where it is.
inline void launch(Handle& h, const Kernel& k, const AllReduceArgs& a, hipStream_t s) {
  const p2p::DevComm p =
      is_staged(k) ? h.dev_comm_staged(a.bytes) : h.dev_comm(a.inp, a.bytes, s);
  launch_on(k, a, p, s);
}

inline void launch(Handle& h, const Kernel& k, const NormArgs& a, hipStream_t s) {
  launch_on(k, a, h.dev_comm(a.inp, a.rows * a.hidden * elem_bytes(a.dtype), s), s);
}

inline void launch(Handle& h, const Kernel& k, const AttnResArgs& a, hipStream_t s) {
  launch_on(k, a, h.dev_comm(a.inp, a.rows * a.hidden * elem_bytes(a.dtype), s), s);
}

inline void launch(Handle& h, const Kernel& k, const GemmTailArgs& a, hipStream_t s) {
  launch_on(k, a, h.dev_comm(a.inp, a.rows * a.hidden * elem_bytes(a.dtype), s), s);
}

inline void launch(Handle& h, const Kernel& k, const ScaleAddArgs& a, hipStream_t s) {
  launch_on(k, a, h.dev_comm(a.inp, bytes_of(a), s), s);
}

// EXPERIMENTAL, no peers.
inline void launch(const Kernel& k, const AddAttnResArgs& a, hipStream_t s) {
  dispatch(k, a, [&](auto kernel, const auto& bind) {
    std::apply([&](auto&&... xs) { start(kernel, k, s, xs...); }, bind());
  });
}

}  // namespace hip_comms
