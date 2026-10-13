// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// LAUNCH, NO DECISIONS: each op's launch_<op> runs the kernel its launch names (select chose it) on
// the launch's grid, block and stream, with the kernel's arguments from the launch and the peers'
// view of its input. A branch here follows a field of the launch; it chooses nothing.

#pragma once

#ifndef HIP_CUSTOM_COMMUNICATOR_INTERFACE
#error "include hip_custom_communicator.cuh, the one interface, not its parts"
#endif

#include <hip/hip_runtime.h>

#include <cstdint>

namespace hip_custom_communicator {

// A KERNEL CALLED THROUGH ITS OP'S SIGNATURE (types.cuh): each argument is converted to its
// parameter, so a wrong one is a compile error, then launched as `kernel<<<blocks, threads, 0,
// stream>>>(args...)` launches it.
template <typename SIGNATURE>
struct Call;
template <typename... PARAMS>
struct Call<void (*)(PARAMS...)> {
  static void run(const void* kernel, int blocks, int threads, hipStream_t stream, PARAMS... args) {
    void* argv[] = {static_cast<void*>(&args)...};
    HIP_CHECK(hipLaunchKernel(kernel, dim3(blocks), dim3(threads), argv, 0, stream));
  }
};

// EVERY LAUNCH OVER THE PEERS hands its kernel, in this order: every rank's input (a device table:
// a captured launch's are filled after the capture; an eager input's is the staging's, copied in
// on the launch's stream), every rank's scratch or staging where the kernel uses it, then every
// rank's signal block, this rank's, its rank and the wait limit; then the kernel's own arguments.
// A ONE-SHOT reads only the inputs; a TWO-SHOT also every rank's scratch.

// A [rows, cols] BUFFER'S BYTES at its row stride, first element to last: what an eager input's
// copy into the registered buffer moves, so the peers read it at its own layout.
inline int64_t extent_bytes(int64_t m, int64_t n, int64_t stride_m, DType dtype) {
  return m > 0 ? ((m - 1) * stride_m + n) * elem_bytes(dtype) : 0;
}

// STAGED: the kernel copies its own input through its staging a pass at a time, so no peer reads it
// where it is; in place, the peers read it through its input table.
inline void launch_all_reduce(Handle& h, const AllReduceLaunch& l) {
  const int inp_size_m = static_cast<int>(l.inp_size_m);
  const int inp_size_n = static_cast<int>(l.inp_size_n);
  const bool one_shot = l.algorithm == Algorithm::one_shot;
  if (l.staged && one_shot)
    Call<AllReduceOneShotStagedKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, h.staging_ptrs(),
        l.staging_stride_m, l.staging_stride_n, h.signals_ptrs(), h.signals_stride_parity(), h.signals_stride_block(),
        h.signals_stride_writer(), h.fault_ptr(), h.rank(),
        h.timeout_ticks(), l.out, l.out_stride_m, l.out_stride_n, l.inp, l.inp_stride_m,
        l.inp_stride_n, inp_size_m, inp_size_n, l.band_m);
  else if (l.staged)
    Call<AllReduceTwoShotStagedKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, h.scratch_ptrs(),
        l.scratch_stride_m, l.scratch_stride_n, h.staging_ptrs(), l.staging_stride_m,
        l.staging_stride_n, h.signals_ptrs(), h.signals_stride_parity(), h.signals_stride_block(),
        h.signals_stride_writer(), h.fault_ptr(), h.rank(), h.timeout_ticks(),
        l.out, l.out_stride_m, l.out_stride_n, l.inp, l.inp_stride_m, l.inp_stride_n, inp_size_m,
        inp_size_n, l.band_m);
  else {
    const void* const* inp_ptrs =
        h.inp_ptrs(l.inp, extent_bytes(l.inp_size_m, l.inp_size_n, l.inp_stride_m, l.dtype),
                   l.stream);
    if (one_shot)
      Call<AllReduceOneShotKernel>::run(l.kernel, l.blocks_per_grid, l.threads_per_block,
                                        l.stream, inp_ptrs, l.inp_stride_m, l.inp_stride_n,
                                        h.signals_ptrs(), h.signals_stride_parity(), h.signals_stride_block(),
                                        h.signals_stride_writer(), h.fault_ptr(), h.rank(),
                                        h.timeout_ticks(), l.out, l.out_stride_m, l.out_stride_n,
                                        inp_size_m, inp_size_n);
    else
      Call<AllReduceTwoShotKernel>::run(l.kernel, l.blocks_per_grid, l.threads_per_block,
                                        l.stream, inp_ptrs, l.inp_stride_m, l.inp_stride_n,
                                        h.scratch_ptrs(), l.scratch_stride_m, l.scratch_stride_n,
                                        h.signals_ptrs(), h.signals_stride_parity(), h.signals_stride_block(),
                                        h.signals_stride_writer(), h.fault_ptr(), h.rank(),
                                        h.timeout_ticks(), l.out, l.out_stride_m, l.out_stride_n,
                                        inp_size_m, inp_size_n);
  }
}

inline void launch_all_reduce_rms_norm(Handle& h, const AllReduceRmsNormLaunch& l) {
  const int64_t bytes = extent_bytes(l.inp_size_m, l.inp_size_n, l.inp_stride_m, l.dtype);
  const int inp_size_m = static_cast<int>(l.inp_size_m);
  const int inp_size_n = static_cast<int>(l.inp_size_n);
  const void* const* inp_ptrs = h.inp_ptrs(l.inp, bytes, l.stream);
  if (l.algorithm == Algorithm::one_shot)
    Call<AllReduceRmsNormOneShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, inp_ptrs, l.inp_stride_m,
        l.inp_stride_n, h.signals_ptrs(), h.signals_stride_parity(), h.signals_stride_block(),
        h.signals_stride_writer(), h.fault_ptr(), h.rank(), h.timeout_ticks(), l.out,
        l.out_stride_m, l.out_stride_n, l.weight, l.weight_stride_n, l.eps, inp_size_m, inp_size_n);
  else
    Call<AllReduceRmsNormTwoShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, inp_ptrs, l.inp_stride_m,
        l.inp_stride_n, h.scratch_ptrs(), l.scratch_stride_m, l.scratch_stride_n,
        h.signals_ptrs(), h.signals_stride_parity(), h.signals_stride_block(),
        h.signals_stride_writer(), h.fault_ptr(), h.rank(), h.timeout_ticks(), l.out, l.out_stride_m,
        l.out_stride_n, l.weight, l.weight_stride_n, l.eps, inp_size_m, inp_size_n);
}

inline void launch_all_reduce_add_rms_norm(Handle& h, const AllReduceAddRmsNormLaunch& l) {
  const int64_t bytes = extent_bytes(l.inp_size_m, l.inp_size_n, l.inp_stride_m, l.dtype);
  const int inp_size_m = static_cast<int>(l.inp_size_m);
  const int inp_size_n = static_cast<int>(l.inp_size_n);
  const void* const* inp_ptrs = h.inp_ptrs(l.inp, bytes, l.stream);
  if (l.algorithm == Algorithm::one_shot)
    Call<AllReduceAddRmsNormOneShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, inp_ptrs, l.inp_stride_m,
        l.inp_stride_n, h.signals_ptrs(), h.signals_stride_parity(), h.signals_stride_block(),
        h.signals_stride_writer(), h.fault_ptr(), h.rank(), h.timeout_ticks(), l.out,
        l.out_stride_m, l.out_stride_n, l.residual_out, l.residual_out_stride_m,
        l.residual_out_stride_n, l.residual, l.residual_stride_m, l.residual_stride_n, l.weight,
        l.weight_stride_n, l.eps, inp_size_m, inp_size_n);
  else
    Call<AllReduceAddRmsNormTwoShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, inp_ptrs, l.inp_stride_m,
        l.inp_stride_n, h.scratch_ptrs(), l.scratch_stride_m, l.scratch_stride_n,
        h.signals_ptrs(), h.signals_stride_parity(), h.signals_stride_block(),
        h.signals_stride_writer(), h.fault_ptr(), h.rank(), h.timeout_ticks(), l.out, l.out_stride_m,
        l.out_stride_n, l.residual_out, l.residual_out_stride_m, l.residual_out_stride_n,
        l.residual, l.residual_stride_m, l.residual_stride_n, l.weight, l.weight_stride_n, l.eps,
        inp_size_m, inp_size_n);
}

// THE PULL TWO-SHOT takes its reduce-scatter blocks last.
inline void launch_all_reduce_add_attn_res_rms_norm(Handle& h,
                                                    const AllReduceAddAttnResRmsNormLaunch& l) {
  const int64_t bytes = extent_bytes(l.inp_size_m, l.inp_size_n, l.inp_stride_m, l.dtype);
  const int inp_size_m = static_cast<int>(l.inp_size_m);
  const int inp_size_n = static_cast<int>(l.inp_size_n);
  const void* const* inp_ptrs = h.inp_ptrs(l.inp, bytes, l.stream);
  if (l.algorithm == Algorithm::one_shot)
    Call<AllReduceAddAttnResRmsNormOneShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, inp_ptrs, l.inp_stride_m,
        l.inp_stride_n, h.signals_ptrs(), h.signals_stride_parity(), h.signals_stride_block(),
        h.signals_stride_writer(), h.fault_ptr(), h.rank(), h.timeout_ticks(),
        l.prefix, l.prefix_stride_m, l.prefix_stride_n, l.blocks, l.blocks_stride_m,
        l.blocks_stride_r, l.blocks_stride_n, l.norm_weight, l.norm_weight_stride_n, l.qk_weight,
        l.qk_weight_stride_n, l.out_norm_weight, l.out_norm_weight_stride_n, l.out, l.out_stride_m,
        l.out_stride_n, l.num_blocks, l.write_idx, l.eps, l.out_eps, inp_size_m, inp_size_n);
  else if (l.direction == Direction::push)
    Call<AllReduceAddAttnResRmsNormPushKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, inp_ptrs, l.inp_stride_m,
        l.inp_stride_n, h.scratch_ptrs(), l.scratch_stride_m, l.scratch_stride_n,
        h.signals_ptrs(), h.signals_stride_parity(), h.signals_stride_block(),
        h.signals_stride_writer(), h.fault_ptr(), h.rank(), h.timeout_ticks(),
        l.prefix, l.prefix_stride_m, l.prefix_stride_n, l.blocks, l.blocks_stride_m,
        l.blocks_stride_r, l.blocks_stride_n, l.norm_weight, l.norm_weight_stride_n, l.qk_weight,
        l.qk_weight_stride_n, l.out_norm_weight, l.out_norm_weight_stride_n, l.out, l.out_stride_m,
        l.out_stride_n, l.num_blocks, l.write_idx, l.eps, l.out_eps, inp_size_m, inp_size_n);
  else
    Call<AllReduceAddAttnResRmsNormPullKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, inp_ptrs, l.inp_stride_m,
        l.inp_stride_n, h.scratch_ptrs(), l.scratch_stride_m, l.scratch_stride_n,
        h.signals_ptrs(), h.signals_stride_parity(), h.signals_stride_block(),
        h.signals_stride_writer(), h.fault_ptr(), h.rank(), h.timeout_ticks(),
        l.prefix, l.prefix_stride_m, l.prefix_stride_n, l.blocks, l.blocks_stride_m,
        l.blocks_stride_r, l.blocks_stride_n, l.norm_weight, l.norm_weight_stride_n, l.qk_weight,
        l.qk_weight_stride_n, l.out_norm_weight, l.out_norm_weight_stride_n, l.out, l.out_stride_m,
        l.out_stride_n, l.num_blocks, l.write_idx, l.eps, l.out_eps, inp_size_m, inp_size_n,
        l.reduce_scatter_blocks);
}

}  // namespace hip_custom_communicator
