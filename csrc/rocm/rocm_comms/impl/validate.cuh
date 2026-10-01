// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// VALIDATE, THE ONLY NO: why a spec cannot run an input here, or empty. Every reason is a
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

namespace hip_comms {

// A KERNEL'S SCRATCH, row-major: the plain two-shot's slice of packs; a column two-shot the whole
// reduced tensor (each rank's columns at their place); a row two-shot its rank's rows, twice where
// it leaves two results (out and the residual). A one-shot reads the inputs and keeps nothing.
inline int64_t scratch_need(Kernel k, Input in) {
  if (!is_two_shot(k)) return 0;
  const Op op         = op_of(k);
  const int64_t packs = in.hidden * in.elem_bytes / kPackBytes;
  if (op == Op::all_reduce) return (in.rows * packs + in.world - 1) / in.world * kPackBytes;
  if (slices_columns(k)) return in.rows * packs * kPackBytes;
  const bool two = op == Op::all_reduce_add_rms_norm;
  return (in.rows + in.world - 1) / in.world * packs * (two ? 2 : 1) * kPackBytes;
}

inline std::string why_not(const Handle& h, Op op, const KernelSpec& k, Input in,
                           const Options& o) {
  if (in.world != 2 && in.world != 4 && in.world != 8)
    return "world size " + std::to_string(in.world) + " is not built (2, 4, 8)";
  if (in.elem_bytes != 2) return "only 2-byte dtypes are built (float16, bfloat16)";
  if (in.hidden * in.elem_bytes % kPackBytes != 0)
    return "the row is not a whole number of 16-byte packs";
  if (op_of(k.kernel) != op) return "the forced kernel is not this op's";
  if (has_row_packs(k.kernel) &&
      row_packs_for(k.kernel, in.hidden * in.elem_bytes / kPackBytes, k.threads) == 0)
    return "the row is wider than the kernel's widest build holds at this block (max_row_packs)";
  // TWO-SHOT'S BLOCK IS ONE WAVE PER PEER, so anything else would leave a peer unread.
  if (k.kernel == Kernel::all_reduce_pull_two_shot && k.threads % (in.world * kWaveSize) != 0)
    return "a two-shot block must be one wave per peer";
  if (o.quant_bits != 16) return "no kernel quantizes yet";
  if (gemms(op) && k.threads > gemm_max_threads(kBuild.gemm_lanes))
    return "the GEMM tail's block exceeds what its LDS holds";
  if (scratch_need(k.kernel, in) > h.scratch_bytes())
    return "its two-shot scratch exceeds the scratch (raise scratch_bytes)";
  return "";
}

// Why `op` over `in` cannot run with these options, before any launch.
inline std::string why_not(const Handle& h, Op op, Input in, const Options& o) {
  return why_not(h, op, select(op, in, o), in, o);
}

template <typename Args>
void validate(const Handle& h, const KernelSpec& k, const Args& a, const Options& o) {
  const Input in        = input_of(h, a);
  const std::string why = why_not(h, op_of(a), k, in, o);
  if (!why.empty())
    throw std::runtime_error("hip_comms: op " + std::to_string(static_cast<int>(op_of(a))) +
                             " over [" + std::to_string(in.rows) + ", " +
                             std::to_string(in.hidden) + "] cannot run: " + why);
}

}  // namespace hip_comms
