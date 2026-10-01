// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// LAUNCH, NO DECISIONS: the peers' view of the input, then the kernel select named (its compiled
// function from impl/dispatch.cuh) at its grid and block.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <hip/hip_runtime.h>

#include <tuple>

namespace hip_comms {

namespace impl {

// The kernel at its grid and block, on the stream. hipify reads `<<<...>>>` as text, so it
// is spelled out.
template <typename... P, typename... A>
void start(void (*kernel)(P...), const Kernel& k, hipStream_t stream, A&&... args) {
  kernel<<<dim3(k.grid), dim3(k.threads), 0, stream>>>(std::forward<A>(args)...);
}

template <typename Args>
void run(Handle& h, const Kernel& k, const Args& a, const p2p::DevComm& p, hipStream_t s) {
  dispatch(k, a, [&](auto kernel, const auto& bind) {
    std::apply([&](auto&&... xs) { start(kernel, k, s, xs...); }, bind(p));
  });
}

}  // namespace impl

// A STAGED BUILD copies the input in itself, so no peer reads it where it is.
inline void launch(Handle& h, const Kernel& k, const AllReduceArgs& a, hipStream_t s) {
  const p2p::DevComm p =
      is_staged(k.fn) ? h.dev_comm_staged(a.bytes) : h.dev_comm(a.inp, a.bytes, s);
  impl::run(h, k, a, p, s);
}

inline void launch(Handle& h, const Kernel& k, const NormArgs& a, hipStream_t s) {
  impl::run(h, k, a, h.dev_comm(a.inp, a.rows * a.hidden * elem_bytes(a.dtype), s), s);
}

inline void launch(Handle& h, const Kernel& k, const AttnResArgs& a, hipStream_t s) {
  impl::run(h, k, a, h.dev_comm(a.inp, a.rows * a.hidden * elem_bytes(a.dtype), s), s);
}

inline void launch(Handle& h, const Kernel& k, const GemmTailArgs& a, hipStream_t s) {
  impl::run(h, k, a, h.dev_comm(a.inp, a.rows * a.hidden * elem_bytes(a.dtype), s), s);
}

inline void launch(Handle& h, const Kernel& k, const ScaleAddArgs& a, hipStream_t s) {
  impl::run(h, k, a, h.dev_comm(a.inp, bytes_of(a), s), s);
}

}  // namespace hip_comms
