// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE OPS: each is select, validate, launch.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

namespace hip_comms {

namespace impl {

template <typename Args>
void run(Handle& h, const Args& a, const Options& o) {
  const KernelSpec k = select(h, a, o);
  validate(h, k, a, o);
  launch(h, k, a, o.stream);
}

}  // namespace impl

inline void all_reduce(Handle& h, const AllReduceArgs& a, const Options& o) { impl::run(h, a, o); }

// rms_norm(all_reduce(inp)), or with a residual fused_add_rms_norm: vLLM's roundings exactly.
inline void all_reduce_rms_norm(Handle& h, const NormArgs& a, const Options& o) {
  impl::run(h, a, o);
}

// Kimi-K3's AttnRes and its RMSNorm on each row of the all-reduced sum (the kernels spell it out).
inline void all_reduce_add_attn_res_rms_norm(Handle& h, const AttnResArgs& a, const Options& o) {
  impl::run(h, a, o);
}

// The latent MoE tail: RMSNorm of the sum, then out[:, col0:col0+N] += normed @ gemm_weight^T.
inline void all_reduce_rms_norm_gemm_add(Handle& h, const GemmTailArgs& a, const Options& o) {
  impl::run(h, a, o);
}

}  // namespace hip_comms
