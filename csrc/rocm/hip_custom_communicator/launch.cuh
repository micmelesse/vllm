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
// input (its own pointer) and its INPUT TABLE (the plain all-reduce's in-place
// pull: the table only), each a device array of pointers with its size, the
// outputs, the partials and the staging where the kernel uses them (each
// region's own device array of every rank's address, with its size, strides
// and sizes; never the arena or an offset), then the signals, this rank's
// status, its rank and the wait limit; then the kernel's own arguments.
// THE MEMORY PATH: an exact call's peers read its input IN PLACE through its
// input table (Handle::input_table: a captured call's tensor, registered at
// capture exit; an eager call's host copy in the inputs region, made on the
// stream before the launch), at the table's strides. An INT4 call's kernel
// quantizes its input into the inputs' INT4 payload itself (which it finds
// from the region and the call's sizes: common.cuh's INT4 payload layout) before
// its first barrier, and the peers read that. The pushed all-reduce (forced)
// pushes its input into the peers' inboxes. A ONE-SHOT reads every rank's
// input; a TWO-SHOT its slice of them and every rank's partials.

// THE PLAIN ALL-REDUCE: PULLED IN PLACE (exact, the default), the kernel takes
// the launch's INPUT TABLE (Handle::input_table: a captured call's slot,
// registered at capture exit, or, eager, the input copied into the inputs
// region on the stream and the eager table) and reads every rank's input
// there. INT4: the two-shot pushes the shares quantized, whole. PUSHED
// (forced): the input pushed into the peers' inboxes, a band a pass.
inline void launch_all_reduce(Handle& h, const AllReduceLaunch& l) {
  const bool in_place = l.direction == Direction::pull &&
                        l.quantization == Quantization::none;
  h.begin_launch(OpType::all_reduce, l.kernel, l.input_size_m, l.input_size_n,
                 l.dtype, l.stream);
  // THE TABLE ONCE THE LAUNCH IS NUMBERED: an eager copy is logged under the
  // launch it feeds.
  const InputTable t =
      in_place ? h.input_table(l.input, l.input_stride_m, l.input_size_m,
                               l.input_size_n, elem_bytes(l.dtype), l.stream)
               : InputTable{nullptr, 0, 0, 0};
  if (in_place && l.algorithm == Algorithm::one_shot)
    Call<AllReduceOneShotInPlaceKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, t.ptr_array, t.ptr_array_size,
        h.signals_ptr_array(), h.signals_ptr_array_size(), h.rank(),
        h.signals_stride_parity(), h.signals_stride_block(), h.signals_stride_writer(),
        t.stride_m, t.stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), h.signals_size_parity(),
        h.signals_size_block(), h.signals_size_writer(), h.status_ptr(),
        h.status_stride_field(), h.status_size_field(), h.timeout_ticks());
  else if (in_place)
    Call<AllReduceTwoShotInPlaceKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, t.ptr_array, t.ptr_array_size,
        h.signals_ptr_array(), h.signals_ptr_array_size(), h.partials_ptr_array(),
        h.partials_ptr_array_size(), h.rank(),
        h.signals_stride_parity(), h.signals_stride_block(), h.signals_stride_writer(),
        t.stride_m, t.stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), l.partials_stride_m, l.partials_stride_n,
        static_cast<int>(l.partials_size_m),
        static_cast<int>(l.partials_size_n), h.signals_size_parity(),
        h.signals_size_block(), h.signals_size_writer(), h.status_ptr(),
        h.status_stride_field(), h.status_size_field(), h.timeout_ticks());
  else if (l.algorithm == Algorithm::one_shot)
    Call<AllReduceOneShotPushedKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, l.input,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), h.staging_ptr_array(), h.staging_ptr_array_size(),
        l.staging_stride_m,
        l.staging_stride_n, static_cast<int>(l.staging_size_m),
        static_cast<int>(l.staging_size_n), h.signals_ptr_array(), h.signals_ptr_array_size(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks());
  else if (l.quantization == Quantization::none)
    Call<AllReduceTwoShotPushedKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, l.input,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), h.partials_ptr_array(), h.partials_ptr_array_size(),
        l.partials_stride_m,
        l.partials_stride_n, static_cast<int>(l.partials_size_m),
        static_cast<int>(l.partials_size_n), h.staging_ptr_array(), h.staging_ptr_array_size(), l.staging_stride_m,
        l.staging_stride_n, static_cast<int>(l.staging_size_m),
        static_cast<int>(l.staging_size_n), h.signals_ptr_array(), h.signals_ptr_array_size(), h.signals_stride_parity(),
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
        static_cast<int>(l.output_size_n), h.partials_ptr_array(), h.partials_ptr_array_size(),
        l.partials_stride_m,
        l.partials_stride_n, static_cast<int>(l.partials_size_m),
        static_cast<int>(l.partials_size_n), h.staging_ptr_array(), h.staging_ptr_array_size(), l.staging_stride_m,
        l.staging_stride_n, static_cast<int>(l.staging_size_m),
        static_cast<int>(l.staging_size_n), h.signals_ptr_array(), h.signals_ptr_array_size(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks());
}

// A FUSED LAUNCH'S INPUT, exact: every rank's read in place through its input
// table (Handle::input_table: a captured call's slot, or the eager table with
// the input copied into the inputs region on the stream), at the table's
// strides. INT4: no table (the kernel quantizes its input into the inputs'
// INT4 payload itself), the inputs' strides as select laid them out.
template <typename LAUNCH>
inline InputTable fused_input(Handle& h, const LAUNCH& l) {
  if (l.quantization != Quantization::none)
    return {nullptr, 0, l.staging_stride_m, l.staging_stride_n};
  return h.input_table(l.input, l.input_stride_m, l.input_size_m,
                       l.input_size_n, elem_bytes(l.dtype), l.stream);
}

// THE NORMS' SPLIT (their `all_reduce` set): that launch first, into this rank's local sums,
// then our norm kernel on them, stream-ordered after it, nothing over the peers.
inline void launch_all_reduce_rms_norm(Handle& h,
                                       const AllReduceRmsNormLaunch& l) {
  if (l.all_reduce) {
    const AllReduceLaunch& r = *l.all_reduce;
    launch_all_reduce(h, r);
    h.begin_launch(OpType::all_reduce_rms_norm, l.kernel, l.input_size_m,
                   l.input_size_n, l.dtype, l.stream);
    Call<RmsNormRowsKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, r.output,
        r.output_stride_m, r.output_stride_n, static_cast<int>(r.output_size_m),
        static_cast<int>(r.output_size_n), l.weight, l.weight_stride_n,
        static_cast<int>(l.weight_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), l.eps);
    return;
  }
  h.begin_launch(OpType::all_reduce_rms_norm, l.kernel, l.input_size_m,
                 l.input_size_n, l.dtype, l.stream);
  const InputTable t = fused_input(h, l);
  if (l.algorithm == Algorithm::one_shot)
    Call<AllReduceRmsNormOneShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, l.input,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), t.ptr_array, t.ptr_array_size, t.stride_m, t.stride_n, l.weight, l.weight_stride_n,
        static_cast<int>(l.weight_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), h.staging_ptr_array(), h.staging_ptr_array_size(), l.staging_stride_m,
        l.staging_stride_n, static_cast<int>(l.staging_size_m),
        static_cast<int>(l.staging_size_n), h.signals_ptr_array(), h.signals_ptr_array_size(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), l.eps);
  else
    Call<AllReduceRmsNormTwoShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, l.input,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), t.ptr_array, t.ptr_array_size, t.stride_m, t.stride_n, l.weight, l.weight_stride_n,
        static_cast<int>(l.weight_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), h.partials_ptr_array(), h.partials_ptr_array_size(),
         l.partials_stride_m,
        l.partials_stride_n, static_cast<int>(l.partials_size_m),
        static_cast<int>(l.partials_size_n), h.staging_ptr_array(), h.staging_ptr_array_size(), l.staging_stride_m,
        l.staging_stride_n, static_cast<int>(l.staging_size_m),
        static_cast<int>(l.staging_size_n), h.signals_ptr_array(), h.signals_ptr_array_size(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), l.eps);
}

inline void launch_all_reduce_add_rms_norm(Handle& h,
                                           const AllReduceAddRmsNormLaunch& l) {
  if (l.all_reduce) {
    const AllReduceLaunch& r = *l.all_reduce;
    launch_all_reduce(h, r);
    h.begin_launch(OpType::all_reduce_add_rms_norm, l.kernel, l.input_size_m,
                   l.input_size_n, l.dtype, l.stream);
    Call<AddRmsNormRowsKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, r.output,
        r.output_stride_m, r.output_stride_n, static_cast<int>(r.output_size_m),
        static_cast<int>(r.output_size_n), l.residual, l.residual_stride_m,
        l.residual_stride_n, static_cast<int>(l.residual_size_m),
        static_cast<int>(l.residual_size_n), l.weight, l.weight_stride_n,
        static_cast<int>(l.weight_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), l.residual_output,
        l.residual_output_stride_m, l.residual_output_stride_n,
        static_cast<int>(l.residual_output_size_m),
        static_cast<int>(l.residual_output_size_n), l.eps);
    return;
  }
  h.begin_launch(OpType::all_reduce_add_rms_norm, l.kernel, l.input_size_m,
                 l.input_size_n, l.dtype, l.stream);
  const InputTable t = fused_input(h, l);
  if (l.algorithm == Algorithm::one_shot)
    Call<AllReduceAddRmsNormOneShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, l.input,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), t.ptr_array, t.ptr_array_size, t.stride_m, t.stride_n, l.residual, l.residual_stride_m,
        l.residual_stride_n, static_cast<int>(l.residual_size_m),
        static_cast<int>(l.residual_size_n), l.weight, l.weight_stride_n,
        static_cast<int>(l.weight_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), l.residual_output,
        l.residual_output_stride_m, l.residual_output_stride_n,
        static_cast<int>(l.residual_output_size_m),
        static_cast<int>(l.residual_output_size_n), h.staging_ptr_array(), h.staging_ptr_array_size(), l.staging_stride_m,
        l.staging_stride_n, static_cast<int>(l.staging_size_m),
        static_cast<int>(l.staging_size_n), h.signals_ptr_array(), h.signals_ptr_array_size(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), l.eps);
  else
    Call<AllReduceAddRmsNormTwoShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, l.input,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), t.ptr_array, t.ptr_array_size, t.stride_m, t.stride_n, l.residual, l.residual_stride_m,
        l.residual_stride_n, static_cast<int>(l.residual_size_m),
        static_cast<int>(l.residual_size_n), l.weight, l.weight_stride_n,
        static_cast<int>(l.weight_size_n), l.output, l.output_stride_m,
        l.output_stride_n, static_cast<int>(l.output_size_m),
        static_cast<int>(l.output_size_n), l.residual_output,
        l.residual_output_stride_m, l.residual_output_stride_n,
        static_cast<int>(l.residual_output_size_m),
        static_cast<int>(l.residual_output_size_n), h.partials_ptr_array(), h.partials_ptr_array_size(),
         l.partials_stride_m,
        l.partials_stride_n, static_cast<int>(l.partials_size_m),
        static_cast<int>(l.partials_size_n), h.scales_ptr_array(),
        h.scales_ptr_array_size(), l.scales_stride_m, static_cast<int>(l.scales_size_m),
        h.staging_ptr_array(), h.staging_ptr_array_size(), l.staging_stride_m,
        l.staging_stride_n, static_cast<int>(l.staging_size_m),
        static_cast<int>(l.staging_size_n), h.signals_ptr_array(), h.signals_ptr_array_size(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), l.eps);
}

// THE PULL TWO-SHOT takes its reduce-scatter blocks last. THE SPLIT (its
// `all_reduce` set): that launch first, into this rank's local sums, then our
// AttnRes kernel on them, stream-ordered after it, nothing over the peers.
inline void launch_all_reduce_add_attn_res_rms_norm(
    Handle& h, const AllReduceAddAttnResRmsNormLaunch& l) {
  if (l.all_reduce) {
    const AllReduceLaunch& r = *l.all_reduce;
    launch_all_reduce(h, r);
    h.begin_launch(OpType::all_reduce_add_attn_res_rms_norm, l.kernel,
                   l.input_size_m, l.input_size_n, l.dtype, l.stream);
    Call<AddAttnResRmsNormKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, r.output,
        r.output_stride_m, r.output_stride_n, static_cast<int>(r.output_size_m),
        static_cast<int>(r.output_size_n), l.prefix, l.prefix_stride_m,
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
        static_cast<int>(l.block_output_size_n), l.num_blocks, l.eps,
        l.output_eps);
    return;
  }
  h.begin_launch(OpType::all_reduce_add_attn_res_rms_norm, l.kernel,
                 l.input_size_m, l.input_size_n, l.dtype, l.stream);
  const InputTable t = fused_input(h, l);
  if (l.algorithm == Algorithm::one_shot)
    Call<AllReduceAddAttnResRmsNormOneShotKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, l.input,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), t.ptr_array, t.ptr_array_size, t.stride_m, t.stride_n, l.prefix, l.prefix_stride_m,
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
        static_cast<int>(l.block_output_size_n), h.signals_ptr_array(), h.signals_ptr_array_size(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), l.num_blocks, l.eps,
        l.output_eps);
  else if (l.direction == Direction::push)
    Call<AllReduceAddAttnResRmsNormPushKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, l.input,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), t.ptr_array, t.ptr_array_size, t.stride_m, t.stride_n, l.prefix, l.prefix_stride_m,
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
        static_cast<int>(l.block_output_size_n), h.partials_ptr_array(),
        h.partials_ptr_array_size(), l.partials_stride_m, l.partials_stride_n,
        static_cast<int>(l.partials_size_m), static_cast<int>(l.partials_size_n),
        h.staging_ptr_array(), h.staging_ptr_array_size(), l.staging_stride_m,
        l.staging_stride_n, static_cast<int>(l.staging_size_m),
        static_cast<int>(l.staging_size_n), h.signals_ptr_array(), h.signals_ptr_array_size(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), l.num_blocks, l.eps,
        l.output_eps);
  else
    Call<AllReduceAddAttnResRmsNormPullKernel>::run(
        l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, l.input,
        l.input_stride_m, l.input_stride_n, static_cast<int>(l.input_size_m),
        static_cast<int>(l.input_size_n), t.ptr_array, t.ptr_array_size, t.stride_m, t.stride_n, l.prefix, l.prefix_stride_m,
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
        static_cast<int>(l.block_output_size_n), h.partials_ptr_array(),
        h.partials_ptr_array_size(), l.partials_stride_m, l.partials_stride_n,
        static_cast<int>(l.partials_size_m), static_cast<int>(l.partials_size_n),
        h.staging_ptr_array(), h.staging_ptr_array_size(), l.staging_stride_m,
        l.staging_stride_n, static_cast<int>(l.staging_size_m),
        static_cast<int>(l.staging_size_n), h.signals_ptr_array(), h.signals_ptr_array_size(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), l.num_blocks, l.eps,
        l.output_eps, l.reduce_scatter_blocks);
}

}  // namespace hip_custom_communicator
