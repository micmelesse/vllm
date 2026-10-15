// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// LAUNCH, NO DECISIONS: each op's launch_<op> runs the kernel its launch names
// (select chose it) on the launch's grid, block and stream, with the kernel's
// arguments from the launch and the peers' view of its input. A branch here
// follows a field of the launch; it chooses nothing.

#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "handle.cuh"
#include "select.cuh"
#include "types.cuh"

namespace hip_custom_communicator {

// A KERNEL CALLED THROUGH ITS OP'S SIGNATURE (types.cuh): each argument is
// converted to its parameter, so a wrong one is a compile error, then launched
// as `kernel<<<blocks, threads, 0, stream>>>(args...)` launches it.
template <typename SIGNATURE>
struct Call;
template <typename... PARAMS>
struct Call<void (*)(PARAMS...)> {
  static void run(const void* kernel, int blocks, int threads,
                  hipStream_t stream, PARAMS... args) {
    void* argv[] = {static_cast<void*>(&args)...};
    HIP_CHECK(
        hipLaunchKernel(kernel, dim3(blocks), dim3(threads), argv, 0, stream));
  }
};

// EVERY LAUNCH OVER THE PEERS hands its kernel, in this order: every rank's
// input (a device table: a captured launch's are filled after the capture; an
// eager input's is the staging's, copied in on the launch's stream), every
// rank's scratch or staging where the kernel uses it, then every rank's signal
// block, this rank's, its rank and the wait limit; then the kernel's own
// arguments. A ONE-SHOT reads only the inputs; a TWO-SHOT also every rank's
// scratch, and the quantized forms of the scratch and of the staged input
// (null with none: the kernel reads neither).

// A TWO-SHOT'S QUANTIZED BUFFERS' TABLES: the handle's, or null with none.
struct QuantizedTables {
  void* const* scratch_values;
  void* const* scratch_scales;
  void* const* staging_values;
  void* const* staging_scales;
};
inline QuantizedTables quantized_tables(const Handle& h, int q) {
  if (q == Quantization::none) return {nullptr, nullptr, nullptr, nullptr};
  return {h.scratch_values_ptrs(), h.scratch_scales_ptrs(),
          h.staging_values_ptrs(), h.staging_scales_ptrs()};
}

// A [rows, cols] BUFFER'S BYTES at its row stride, first element to last: what
// an eager input's copy into the registered buffer moves, so the peers read it
// at its own layout.
inline int64_t extent_bytes(int64_t m, int64_t n, int64_t stride_m,
                            DType dtype) {
  return m > 0 ? ((m - 1) * stride_m + n) * elem_bytes(dtype) : 0;
}

// STAGED: the kernel copies its own input through its staging a pass at a time,
// so no peer reads it where it is; in place, the peers read it through its
// input table.
inline void launch_all_reduce(Handle& h, const AllReduceLaunch& l) {
  const bool one_shot = l.algorithm == Algorithm::one_shot;
  if (l.staged && one_shot)
    Call<AllReduceOneShotStagedKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, l.input,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), h.staging_ptrs(), l.staging_stride_m,
        l.staging_stride_n, static_cast<int>(l.staging_size_m),
        static_cast<int>(l.staging_size_n), h.signals_ptrs(),
        h.signals_stride_parity(), h.signals_stride_block(),
        h.signals_stride_writer(), h.signals_size_parity(),
        h.signals_size_block(), h.signals_size_writer(), h.status_ptr(),
        h.status_stride_field(), h.status_size_field(), h.rank(),
        h.timeout_ticks());
  else if (l.staged)
    Call<AllReduceTwoShotStagedKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, l.input,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), h.scratch_ptrs(), l.scratch_stride_m,
        l.scratch_stride_n, static_cast<int>(l.scratch_size_m),
        static_cast<int>(l.scratch_size_n), h.staging_ptrs(),
        l.staging_stride_m, l.staging_stride_n,
        static_cast<int>(l.staging_size_m), static_cast<int>(l.staging_size_n),
        h.signals_ptrs(), h.signals_stride_parity(), h.signals_stride_block(),
        h.signals_stride_writer(), h.signals_size_parity(),
        h.signals_size_block(), h.signals_size_writer(), h.status_ptr(),
        h.status_stride_field(), h.status_size_field(), h.rank(),
        h.timeout_ticks());
  else {
    const QuantizedTables q = quantized_tables(h, l.quantization);
    const void* const* input_ptrs = h.input_ptrs(
        l.input,
        extent_bytes(l.input_size_m, l.input_size_n, l.input_stride_m, l.dtype),
        l.stream);
    if (one_shot)
      Call<AllReduceOneShotKernel>::run(
          l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream,
          input_ptrs, l.input_stride_m, l.input_stride_n,
          static_cast<int>(l.input_size_m), static_cast<int>(l.input_size_n),
          l.output, l.output_stride_m, l.output_stride_n,
          static_cast<int>(l.output_size_m), static_cast<int>(l.output_size_n),
          h.signals_ptrs(), h.signals_stride_parity(), h.signals_stride_block(),
          h.signals_stride_writer(), h.signals_size_parity(),
          h.signals_size_block(), h.signals_size_writer(), h.status_ptr(),
          h.status_stride_field(), h.status_size_field(), h.rank(),
          h.timeout_ticks());
    else
      Call<AllReduceTwoShotKernel>::run(
          l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream,
          input_ptrs, l.input_stride_m, l.input_stride_n,
          static_cast<int>(l.input_size_m), static_cast<int>(l.input_size_n),
          l.output, l.output_stride_m, l.output_stride_n,
          static_cast<int>(l.output_size_m), static_cast<int>(l.output_size_n),
          h.scratch_ptrs(), l.scratch_stride_m, l.scratch_stride_n,
          static_cast<int>(l.scratch_size_m),
          static_cast<int>(l.scratch_size_n), q.scratch_values, l.scratch_values_stride_m,
          l.scratch_values_stride_n, static_cast<int>(l.scratch_values_size_m),
          static_cast<int>(l.scratch_values_size_n), q.scratch_scales,
          l.scratch_scales_stride_m, l.scratch_scales_stride_n,
          static_cast<int>(l.scratch_scales_size_m),
          static_cast<int>(l.scratch_scales_size_n), q.staging_values,
          l.staging_values_stride_m, l.staging_values_stride_n,
          static_cast<int>(l.staging_values_size_m),
          static_cast<int>(l.staging_values_size_n), q.staging_scales,
          l.staging_scales_stride_m, l.staging_scales_stride_n,
          static_cast<int>(l.staging_scales_size_m),
          static_cast<int>(l.staging_scales_size_n),
          h.signals_ptrs(),
          h.signals_stride_parity(), h.signals_stride_block(),
          h.signals_stride_writer(), h.signals_size_parity(),
          h.signals_size_block(), h.signals_size_writer(), h.status_ptr(),
          h.status_stride_field(), h.status_size_field(), h.rank(),
          h.timeout_ticks());
  }
}

inline void launch_all_reduce_rms_norm(Handle& h,
                                       const AllReduceRmsNormLaunch& l) {
  const int64_t bytes =
      extent_bytes(l.input_size_m, l.input_size_n, l.input_stride_m, l.dtype);
  const QuantizedTables q = quantized_tables(h, l.quantization);
  const void* const* input_ptrs = h.input_ptrs(l.input, bytes, l.stream);
  if (l.algorithm == Algorithm::one_shot)
    Call<AllReduceRmsNormOneShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, input_ptrs,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), l.weight, l.weight_stride_n,
        static_cast<int>(l.weight_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), h.signals_ptrs(),
        h.signals_stride_parity(), h.signals_stride_block(),
        h.signals_stride_writer(), h.signals_size_parity(),
        h.signals_size_block(), h.signals_size_writer(), h.status_ptr(),
        h.status_stride_field(), h.status_size_field(), h.rank(),
        h.timeout_ticks(), l.eps);
  else
    Call<AllReduceRmsNormTwoShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, input_ptrs,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), l.weight, l.weight_stride_n,
        static_cast<int>(l.weight_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), h.scratch_ptrs(), l.scratch_stride_m,
        l.scratch_stride_n, static_cast<int>(l.scratch_size_m),
        static_cast<int>(l.scratch_size_n), q.scratch_values, l.scratch_values_stride_m,
        l.scratch_values_stride_n, static_cast<int>(l.scratch_values_size_m),
        static_cast<int>(l.scratch_values_size_n), q.scratch_scales,
        l.scratch_scales_stride_m, l.scratch_scales_stride_n,
        static_cast<int>(l.scratch_scales_size_m),
        static_cast<int>(l.scratch_scales_size_n), q.staging_values,
        l.staging_values_stride_m, l.staging_values_stride_n,
        static_cast<int>(l.staging_values_size_m),
        static_cast<int>(l.staging_values_size_n), q.staging_scales,
        l.staging_scales_stride_m, l.staging_scales_stride_n,
        static_cast<int>(l.staging_scales_size_m),
        static_cast<int>(l.staging_scales_size_n),
        h.signals_ptrs(),
        h.signals_stride_parity(), h.signals_stride_block(),
        h.signals_stride_writer(), h.signals_size_parity(),
        h.signals_size_block(), h.signals_size_writer(), h.status_ptr(),
        h.status_stride_field(), h.status_size_field(), h.rank(),
        h.timeout_ticks(), l.eps);
}

inline void launch_all_reduce_add_rms_norm(Handle& h,
                                           const AllReduceAddRmsNormLaunch& l) {
  const int64_t bytes =
      extent_bytes(l.input_size_m, l.input_size_n, l.input_stride_m, l.dtype);
  const QuantizedTables q = quantized_tables(h, l.quantization);
  const void* const* input_ptrs = h.input_ptrs(l.input, bytes, l.stream);
  if (l.algorithm == Algorithm::one_shot)
    Call<AllReduceAddRmsNormOneShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, input_ptrs,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), l.residual, l.residual_stride_m,
        l.residual_stride_n, static_cast<int>(l.residual_size_m),
        static_cast<int>(l.residual_size_n), l.weight, l.weight_stride_n,
        static_cast<int>(l.weight_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), l.residual_output,
        l.residual_output_stride_m, l.residual_output_stride_n,
        static_cast<int>(l.residual_output_size_m),
        static_cast<int>(l.residual_output_size_n), h.signals_ptrs(),
        h.signals_stride_parity(), h.signals_stride_block(),
        h.signals_stride_writer(), h.signals_size_parity(),
        h.signals_size_block(), h.signals_size_writer(), h.status_ptr(),
        h.status_stride_field(), h.status_size_field(), h.rank(),
        h.timeout_ticks(), l.eps);
  else
    Call<AllReduceAddRmsNormTwoShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, input_ptrs,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), l.residual, l.residual_stride_m,
        l.residual_stride_n, static_cast<int>(l.residual_size_m),
        static_cast<int>(l.residual_size_n), l.weight, l.weight_stride_n,
        static_cast<int>(l.weight_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), l.residual_output,
        l.residual_output_stride_m, l.residual_output_stride_n,
        static_cast<int>(l.residual_output_size_m),
        static_cast<int>(l.residual_output_size_n), h.scratch_ptrs(),
        l.scratch_stride_m, l.scratch_stride_n,
        static_cast<int>(l.scratch_size_m), static_cast<int>(l.scratch_size_n),
        q.scratch_values, l.scratch_values_stride_m,
        l.scratch_values_stride_n, static_cast<int>(l.scratch_values_size_m),
        static_cast<int>(l.scratch_values_size_n), q.scratch_scales,
        l.scratch_scales_stride_m, l.scratch_scales_stride_n,
        static_cast<int>(l.scratch_scales_size_m),
        static_cast<int>(l.scratch_scales_size_n), q.staging_values,
        l.staging_values_stride_m, l.staging_values_stride_n,
        static_cast<int>(l.staging_values_size_m),
        static_cast<int>(l.staging_values_size_n), q.staging_scales,
        l.staging_scales_stride_m, l.staging_scales_stride_n,
        static_cast<int>(l.staging_scales_size_m),
        static_cast<int>(l.staging_scales_size_n),
        h.signals_ptrs(), h.signals_stride_parity(), h.signals_stride_block(),
        h.signals_stride_writer(), h.signals_size_parity(),
        h.signals_size_block(), h.signals_size_writer(), h.status_ptr(),
        h.status_stride_field(), h.status_size_field(), h.rank(),
        h.timeout_ticks(), l.eps);
}

// THE PULL TWO-SHOT takes its reduce-scatter blocks last.
inline void launch_all_reduce_add_attn_res_rms_norm(
    Handle& h, const AllReduceAddAttnResRmsNormLaunch& l) {
  const int64_t bytes =
      extent_bytes(l.input_size_m, l.input_size_n, l.input_stride_m, l.dtype);
  const QuantizedTables q = quantized_tables(h, l.quantization);
  const void* const* input_ptrs = h.input_ptrs(l.input, bytes, l.stream);
  if (l.algorithm == Algorithm::one_shot)
    Call<AllReduceAddAttnResRmsNormOneShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, input_ptrs,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), l.prefix, l.prefix_stride_m,
        l.prefix_stride_n, static_cast<int>(l.prefix_size_m),
        static_cast<int>(l.prefix_size_n), l.blocks, l.blocks_stride_m,
        l.blocks_stride_r, l.blocks_stride_n, static_cast<int>(l.blocks_size_m),
        static_cast<int>(l.blocks_size_r), static_cast<int>(l.blocks_size_n),
        l.norm_weight, l.norm_weight_stride_n,
        static_cast<int>(l.norm_weight_size_n), l.qk_weight,
        l.qk_weight_stride_n, static_cast<int>(l.qk_weight_size_n),
        l.output_norm_weight, l.output_norm_weight_stride_n,
        static_cast<int>(l.output_norm_weight_size_n), l.prefix_output,
        l.prefix_output_stride_m, l.prefix_output_stride_n,
        static_cast<int>(l.prefix_output_size_m),
        static_cast<int>(l.prefix_output_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), l.block_output,
        l.block_output_stride_m, l.block_output_stride_n,
        static_cast<int>(l.block_output_size_m),
        static_cast<int>(l.block_output_size_n), h.signals_ptrs(),
        h.signals_stride_parity(), h.signals_stride_block(),
        h.signals_stride_writer(), h.signals_size_parity(),
        h.signals_size_block(), h.signals_size_writer(), h.status_ptr(),
        h.status_stride_field(), h.status_size_field(), h.rank(),
        h.timeout_ticks(), l.num_blocks, l.eps, l.output_eps);
  else if (l.direction == Direction::push)
    Call<AllReduceAddAttnResRmsNormPushKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, input_ptrs,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), l.prefix, l.prefix_stride_m,
        l.prefix_stride_n, static_cast<int>(l.prefix_size_m),
        static_cast<int>(l.prefix_size_n), l.blocks, l.blocks_stride_m,
        l.blocks_stride_r, l.blocks_stride_n, static_cast<int>(l.blocks_size_m),
        static_cast<int>(l.blocks_size_r), static_cast<int>(l.blocks_size_n),
        l.norm_weight, l.norm_weight_stride_n,
        static_cast<int>(l.norm_weight_size_n), l.qk_weight,
        l.qk_weight_stride_n, static_cast<int>(l.qk_weight_size_n),
        l.output_norm_weight, l.output_norm_weight_stride_n,
        static_cast<int>(l.output_norm_weight_size_n), l.prefix_output,
        l.prefix_output_stride_m, l.prefix_output_stride_n,
        static_cast<int>(l.prefix_output_size_m),
        static_cast<int>(l.prefix_output_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), l.block_output,
        l.block_output_stride_m, l.block_output_stride_n,
        static_cast<int>(l.block_output_size_m),
        static_cast<int>(l.block_output_size_n), h.scratch_ptrs(),
        l.scratch_stride_m, l.scratch_stride_n,
        static_cast<int>(l.scratch_size_m), static_cast<int>(l.scratch_size_n),
        q.scratch_values, l.scratch_values_stride_m,
        l.scratch_values_stride_n, static_cast<int>(l.scratch_values_size_m),
        static_cast<int>(l.scratch_values_size_n), q.scratch_scales,
        l.scratch_scales_stride_m, l.scratch_scales_stride_n,
        static_cast<int>(l.scratch_scales_size_m),
        static_cast<int>(l.scratch_scales_size_n), q.staging_values,
        l.staging_values_stride_m, l.staging_values_stride_n,
        static_cast<int>(l.staging_values_size_m),
        static_cast<int>(l.staging_values_size_n), q.staging_scales,
        l.staging_scales_stride_m, l.staging_scales_stride_n,
        static_cast<int>(l.staging_scales_size_m),
        static_cast<int>(l.staging_scales_size_n),
        h.signals_ptrs(), h.signals_stride_parity(), h.signals_stride_block(),
        h.signals_stride_writer(), h.signals_size_parity(),
        h.signals_size_block(), h.signals_size_writer(), h.status_ptr(),
        h.status_stride_field(), h.status_size_field(), h.rank(),
        h.timeout_ticks(), l.num_blocks, l.eps, l.output_eps);
  else
    Call<AllReduceAddAttnResRmsNormPullKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, input_ptrs,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), l.prefix, l.prefix_stride_m,
        l.prefix_stride_n, static_cast<int>(l.prefix_size_m),
        static_cast<int>(l.prefix_size_n), l.blocks, l.blocks_stride_m,
        l.blocks_stride_r, l.blocks_stride_n, static_cast<int>(l.blocks_size_m),
        static_cast<int>(l.blocks_size_r), static_cast<int>(l.blocks_size_n),
        l.norm_weight, l.norm_weight_stride_n,
        static_cast<int>(l.norm_weight_size_n), l.qk_weight,
        l.qk_weight_stride_n, static_cast<int>(l.qk_weight_size_n),
        l.output_norm_weight, l.output_norm_weight_stride_n,
        static_cast<int>(l.output_norm_weight_size_n), l.prefix_output,
        l.prefix_output_stride_m, l.prefix_output_stride_n,
        static_cast<int>(l.prefix_output_size_m),
        static_cast<int>(l.prefix_output_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), l.block_output,
        l.block_output_stride_m, l.block_output_stride_n,
        static_cast<int>(l.block_output_size_m),
        static_cast<int>(l.block_output_size_n), h.scratch_ptrs(),
        l.scratch_stride_m, l.scratch_stride_n,
        static_cast<int>(l.scratch_size_m), static_cast<int>(l.scratch_size_n),
        q.scratch_values, l.scratch_values_stride_m,
        l.scratch_values_stride_n, static_cast<int>(l.scratch_values_size_m),
        static_cast<int>(l.scratch_values_size_n), q.scratch_scales,
        l.scratch_scales_stride_m, l.scratch_scales_stride_n,
        static_cast<int>(l.scratch_scales_size_m),
        static_cast<int>(l.scratch_scales_size_n), q.staging_values,
        l.staging_values_stride_m, l.staging_values_stride_n,
        static_cast<int>(l.staging_values_size_m),
        static_cast<int>(l.staging_values_size_n), q.staging_scales,
        l.staging_scales_stride_m, l.staging_scales_stride_n,
        static_cast<int>(l.staging_scales_size_m),
        static_cast<int>(l.staging_scales_size_n),
        h.signals_ptrs(), h.signals_stride_parity(), h.signals_stride_block(),
        h.signals_stride_writer(), h.signals_size_parity(),
        h.signals_size_block(), h.signals_size_writer(), h.status_ptr(),
        h.status_stride_field(), h.status_size_field(), h.rank(),
        h.timeout_ticks(), l.num_blocks, l.eps, l.output_eps,
        l.reduce_scatter_blocks);
}

}  // namespace hip_custom_communicator
