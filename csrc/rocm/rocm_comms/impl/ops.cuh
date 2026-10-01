// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// THE OPS: each is select, validate, launch.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <variant>

namespace hip_comms {

namespace impl {

// The plan's kernel launched, or its Error and nothing launched.
template <typename Args>
std::variant<Kernel, Error> run(Handle& h, const Args& a, const Options& o) {
  std::variant<Kernel, Error> p = plan(h, a, o);
  if (const Kernel* k = std::get_if<Kernel>(&p)) launch(h, *k, a, o.stream);
  return p;
}

}  // namespace impl

inline std::variant<Kernel, Error> all_reduce(Handle& h, const AllReduceArgs& a,
                                                 const Options& o) {
  return impl::run(h, a, o);
}

// rms_norm(all_reduce(inp)), or with a residual fused_add_rms_norm: vLLM's roundings exactly.
inline std::variant<Kernel, Error> all_reduce_rms_norm(Handle& h, const NormArgs& a,
                                                 const Options& o) {
  return impl::run(h, a, o);
}

// Kimi-K3's AttnRes and its RMSNorm on each row of the all-reduced sum (the kernels spell it out).
inline std::variant<Kernel, Error> all_reduce_add_attn_res_rms_norm(Handle& h, const AttnResArgs& a,
                                                 const Options& o) {
  return impl::run(h, a, o);
}

// RMSNorm of the sum, then out[:, col0:col0+N] = normed @ gemm_weight^T.
inline std::variant<Kernel, Error> all_reduce_rms_norm_gemm(Handle& h, const GemmTailArgs& a,
                                                 const Options& o) {
  return impl::run(h, a, o);
}

// The latent MoE tail: RMSNorm of the sum, then out[:, col0:col0+N] += normed @ gemm_weight^T.
inline std::variant<Kernel, Error> all_reduce_rms_norm_gemm_add(Handle& h, const GemmTailArgs& a,
                                                 const Options& o) {
  return impl::run(h, a, o);
}

// Kimi-K3's latent MoE tail with one all-reduce: out = shared + projected * 1/rms(latent), all
// three summed over the ranks.
inline std::variant<Kernel, Error> all_reduce_rms_scale_add(Handle& h, const ScaleAddArgs& a,
                                                 const Options& o) {
  return impl::run(h, a, o);
}

}  // namespace hip_comms
