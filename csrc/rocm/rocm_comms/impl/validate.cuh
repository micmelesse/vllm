// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// VALIDATE, THE ONLY NO: why a kernel cannot run a call here, or empty. Every reason is a
// capability (a kernel that cannot take the input), never "the unfused ops would be faster": a
// tune_<op> never declines. The ops raise with it; `why_not` is it before any launch, for vLLM's
// choice of all-reduce backend.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace hip_comms {

// A KERNEL'S SCRATCH, row-major: the plain two-shot's slice of packs; a column two-shot the whole
// reduced tensor (each rank's columns at their place); a row two-shot its rank's rows, twice where
// it leaves two results (out and the residual). A one-shot reads the inputs and keeps nothing.
inline int64_t scratch_need(Template t, int64_t rows, int64_t packs, int world) {
  if (!is_two_shot(t)) return 0;
  const Op op = op_of(t);
  if (op == Op::all_reduce) return (rows * packs + world - 1) / world * kPackBytes;
  if (slices_columns(t)) return rows * packs * kPackBytes;
  const bool two = op == Op::all_reduce_add_rms_norm;
  return (rows + world - 1) / world * packs * (two ? 2 : 1) * kPackBytes;
}

// Why kernel `k` cannot hold its whole grid resident, or empty: only the compiled kernel knows
// what it uses.
template <typename Args>
std::string why_not_resident(const Kernel& k, const Args& a) {
  std::string why;
  dispatch(k, a, [&](auto kernel, const auto&) {
    const Resources r  = resources_of(kernel);
    const int resident = resident_blocks(kTarget, r, k.threads);
    if (k.grid > resident)
      why = "its grid of " + std::to_string(k.grid) + " exceeds the " + std::to_string(resident) +
            " blocks it holds resident (" + std::to_string(r.vgprs) + " VGPRs, " +
            std::to_string(r.lds_bytes) + " B of LDS at " + std::to_string(k.threads) +
            " threads)";
  });
  return why;
}

// Why kernel `k` cannot run call `a` here, or empty.
template <typename Args>
std::string why_not(const Handle& h, const Kernel& k, const Args& a, const Options& o) {
  const int world = h.world_size();
  const int e     = elem_bytes(a.dtype);
  if (world != 2 && world != 4 && world != 8)
    return "world size " + std::to_string(world) + " is not built (2, 4, 8)";
  if (e != 2) return "only 2-byte dtypes are built (float16, bfloat16)";
  if (hidden_of(a) * e % kPackBytes != 0) return "the row is not a whole number of 16-byte packs";
  if constexpr (std::is_same_v<Args, ScaleAddArgs>)
    if (a.hidden * e % kPackBytes != 0 || a.latent * e % kPackBytes != 0 || a.latent < 1)
      return "the hidden and the latent must each be a whole number of 16-byte packs";
  if (op_of(k.fn) != op_of(a)) return "the forced template is not this op's";
  if (has_row_packs(k.fn) && !row_packs_of(k.args))
    return "the row is wider than the template's widest build holds at this block (max_row_packs)";
  // TWO-SHOT'S BLOCK IS ONE WAVE PER PEER, so anything else would leave a peer unread.
  if (k.fn == Template::all_reduce_pull_two_shot && k.threads % (world * kWaveSize) != 0)
    return "a two-shot block must be one wave per peer";
  if (o.quant_bits != 16) return "no kernel quantizes yet";
  if (gemms(op_of(a)) && k.threads > gemm_max_threads(kBuild.gemm_lanes))
    return "the GEMM tail's block exceeds what its LDS holds";
  if (scratch_need(k.fn, rows_of(a), packs_of(a), world) > h.scratch_bytes())
    return "its two-shot scratch exceeds the scratch (raise scratch_bytes)";
  return why_not_resident(k, a);
}

// Why call `a` cannot run with these options, before any launch: `admits` for vLLM.
template <typename Args>
std::string why_not(const Handle& h, const Args& a, const Options& o) {
  return why_not(h, select(a, h.world_size(), o), a, o);
}

template <typename Args>
void validate(const Handle& h, const Kernel& k, const Args& a, const Options& o) {
  const std::string why = why_not(h, k, a, o);
  if (!why.empty())
    throw std::runtime_error("hip_comms: op " + std::to_string(static_cast<int>(op_of(a))) +
                             " over [" + std::to_string(rows_of(a)) + ", " +
                             std::to_string(hidden_of(a)) + "] cannot run: " + why);
}

}  // namespace hip_comms
