// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// LAUNCH, NO DECISIONS: each op's launch_<op> runs the kernel its launch names
// (select chose it) on the launch's grid, block and stream, with the kernel's
// arguments from the launch and the handle's buffers. A branch here
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

// EVERY LAUNCH OVER THE PEERS hands its kernel, in this order: this rank's
// input (its own pointer: no table), the outputs, every rank's arena base (by
// value: `PeerBases`, in the kernel arguments) and the partials and inputs
// where the kernel uses them, then the signals, this rank's status, its rank
// and the wait limit; then the kernel's own arguments. Our regions (handle.cuh's
// arena layout) are each its byte offset in the arena, the same on every rank.
// THE ONE MEMORY PATH: every kernel copies this rank's input into this rank's
// inputs (INT4: quantized into their INT4 payload, which the kernel finds
// itself from the region and the call's sizes: common.cuh's `int4_payload`)
// inside the kernel, before its first barrier, which is ordered, so the peers
// read only our arena, never the caller's tensor; eager and captured alike, no
// host copy. A ONE-SHOT reads every rank's inputs; a TWO-SHOT its slice of them
// and every rank's partials.

// THE PLAIN ALL-REDUCE: the kernel copies its input through its inputs a band
// of rows a pass (none); the INT4 two-shot quantizes it whole into the inputs's
// INT4 payload.
inline void launch_all_reduce(Handle& h, const AllReduceLaunch& l) {
  h.begin_launch(OpType::all_reduce, l.kernel, l.input_size_m, l.input_size_n,
                 l.dtype, l.stream);
  if (l.algorithm == Algorithm::one_shot)
    Call<AllReduceOneShotStagedKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, l.input,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), h.bases(),
        h.inputs_offset(), l.inputs_stride_m,
        l.inputs_stride_n, static_cast<int>(l.inputs_size_m),
        static_cast<int>(l.inputs_size_n), h.signals_offset(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks());
  else if (l.quantization == Quantization::none)
    Call<AllReduceTwoShotStagedKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, l.input,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), h.bases(),
        h.partials_offset(), l.partials_stride_m,
        l.partials_stride_n, static_cast<int>(l.partials_size_m),
        static_cast<int>(l.partials_size_n), h.inputs_offset(), l.inputs_stride_m,
        l.inputs_stride_n, static_cast<int>(l.inputs_size_m),
        static_cast<int>(l.inputs_size_n), h.signals_offset(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks());
  else
    Call<AllReduceTwoShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, l.input,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), h.bases(),
        h.partials_offset(), l.partials_stride_m,
        l.partials_stride_n, static_cast<int>(l.partials_size_m),
        static_cast<int>(l.partials_size_n), h.inputs_offset(), l.inputs_stride_m,
        l.inputs_stride_n, static_cast<int>(l.inputs_size_m),
        static_cast<int>(l.inputs_size_n), h.signals_offset(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks());
}

inline void launch_all_reduce_rms_norm(Handle& h,
                                       const AllReduceRmsNormLaunch& l) {
  h.begin_launch(OpType::all_reduce_rms_norm, l.kernel, l.input_size_m,
                 l.input_size_n, l.dtype, l.stream);
  if (l.algorithm == Algorithm::one_shot)
    Call<AllReduceRmsNormOneShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, l.input,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), l.weight, l.weight_stride_n,
        static_cast<int>(l.weight_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), h.bases(), h.inputs_offset(), l.inputs_stride_m,
        l.inputs_stride_n, static_cast<int>(l.inputs_size_m),
        static_cast<int>(l.inputs_size_n), h.signals_offset(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), l.eps);
  else
    Call<AllReduceRmsNormTwoShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, l.input,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), l.weight, l.weight_stride_n,
        static_cast<int>(l.weight_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), h.bases(),
        h.partials_offset(), l.partials_stride_m,
        l.partials_stride_n, static_cast<int>(l.partials_size_m),
        static_cast<int>(l.partials_size_n), h.inputs_offset(), l.inputs_stride_m,
        l.inputs_stride_n, static_cast<int>(l.inputs_size_m),
        static_cast<int>(l.inputs_size_n), h.signals_offset(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), l.eps);
}

inline void launch_all_reduce_add_rms_norm(Handle& h,
                                           const AllReduceAddRmsNormLaunch& l) {
  h.begin_launch(OpType::all_reduce_add_rms_norm, l.kernel, l.input_size_m,
                 l.input_size_n, l.dtype, l.stream);
  if (l.algorithm == Algorithm::one_shot)
    Call<AllReduceAddRmsNormOneShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, l.input,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), l.residual, l.residual_stride_m,
        l.residual_stride_n, static_cast<int>(l.residual_size_m),
        static_cast<int>(l.residual_size_n), l.weight, l.weight_stride_n,
        static_cast<int>(l.weight_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), l.residual_output,
        l.residual_output_stride_m, l.residual_output_stride_n,
        static_cast<int>(l.residual_output_size_m),
        static_cast<int>(l.residual_output_size_n), h.bases(), h.inputs_offset(), l.inputs_stride_m,
        l.inputs_stride_n, static_cast<int>(l.inputs_size_m),
        static_cast<int>(l.inputs_size_n), h.signals_offset(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), l.eps);
  else
    Call<AllReduceAddRmsNormTwoShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, l.input,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), l.residual, l.residual_stride_m,
        l.residual_stride_n, static_cast<int>(l.residual_size_m),
        static_cast<int>(l.residual_size_n), l.weight, l.weight_stride_n,
        static_cast<int>(l.weight_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), l.residual_output,
        l.residual_output_stride_m, l.residual_output_stride_n,
        static_cast<int>(l.residual_output_size_m),
        static_cast<int>(l.residual_output_size_n), h.bases(),
        h.partials_offset(), l.partials_stride_m,
        l.partials_stride_n, static_cast<int>(l.partials_size_m),
        static_cast<int>(l.partials_size_n), h.inputs_offset(), l.inputs_stride_m,
        l.inputs_stride_n, static_cast<int>(l.inputs_size_m),
        static_cast<int>(l.inputs_size_n), h.signals_offset(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), l.eps);
}

// THE PULL TWO-SHOT takes its reduce-scatter blocks last.
inline void launch_all_reduce_add_attn_res_rms_norm(
    Handle& h, const AllReduceAddAttnResRmsNormLaunch& l) {
  h.begin_launch(OpType::all_reduce_add_attn_res_rms_norm, l.kernel,
                 l.input_size_m, l.input_size_n, l.dtype, l.stream);
  if (l.algorithm == Algorithm::one_shot)
    Call<AllReduceAddAttnResRmsNormOneShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, l.input,
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
        static_cast<int>(l.block_output_size_n), h.bases(), h.inputs_offset(), l.inputs_stride_m,
        l.inputs_stride_n, static_cast<int>(l.inputs_size_m),
        static_cast<int>(l.inputs_size_n), h.signals_offset(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), l.num_blocks, l.eps,
        l.output_eps);
  else if (l.direction == Direction::push)
    Call<AllReduceAddAttnResRmsNormPushKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, l.input,
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
        static_cast<int>(l.block_output_size_n), h.bases(),
        h.partials_offset(), l.partials_stride_m,
        l.partials_stride_n, static_cast<int>(l.partials_size_m),
        static_cast<int>(l.partials_size_n), h.inputs_offset(), l.inputs_stride_m,
        l.inputs_stride_n, static_cast<int>(l.inputs_size_m),
        static_cast<int>(l.inputs_size_n), h.signals_offset(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), l.num_blocks, l.eps,
        l.output_eps);
  else
    Call<AllReduceAddAttnResRmsNormPullKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, l.input,
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
        static_cast<int>(l.block_output_size_n), h.bases(),
        h.partials_offset(), l.partials_stride_m,
        l.partials_stride_n, static_cast<int>(l.partials_size_m),
        static_cast<int>(l.partials_size_n), h.inputs_offset(), l.inputs_stride_m,
        l.inputs_stride_n, static_cast<int>(l.inputs_size_m),
        static_cast<int>(l.inputs_size_n), h.signals_offset(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), l.num_blocks, l.eps,
        l.output_eps, l.reduce_scatter_blocks);
}

}  // namespace hip_custom_communicator
