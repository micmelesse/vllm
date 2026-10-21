// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Each op checks its call, obtains a complete config, and launches its kernels.

#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <optional>
#include <variant>

#include "handle.cuh"
#include "kernels/common.cuh"
#include "config.cuh"
#include "types.cuh"

namespace hip_custom_communicator {

// A LAUNCH DIMENSION AS HIP TAKES IT.
inline dim3 dim3_of(const Dim3& d) {
  return dim3(static_cast<unsigned>(d.x), static_cast<unsigned>(d.y),
              static_cast<unsigned>(d.z));
}

// =================================================================================================
// all_reduce: the sum of every rank's `input` into `output`, each at its own
// strides and sizes.
// =================================================================================================

inline std::variant<Config, Error> all_reduce(
    Handle& h, const void* input, int64_t input_stride_m,
    int64_t input_stride_n, int64_t input_size_m, int64_t input_size_n,
    void* output, int64_t output_stride_m, int64_t output_stride_n,
    int64_t output_size_m, int64_t output_size_n, DType dtype,
    std::optional<Algorithm> algorithm, std::optional<Direction> direction,
    std::optional<int> tile_m, std::optional<int> tile_n,
    std::optional<int> block_x, std::optional<int> block_y, std::optional<int> block_z,
    std::optional<int> grid_x, std::optional<int> grid_y, std::optional<int> grid_z,
    std::optional<int> waves_per_eu, std::optional<int> quantization,
    hipStream_t stream) {
  if (const std::optional<Error> e =
          layout_refused(dtype, {{output_stride_m, output_stride_n},
                                 {input_stride_m, input_stride_n}}))
    return *e;
  std::variant<Config, Error> checked = all_reduce_config(
      h, input, input_stride_m, input_stride_n, input_size_m, input_size_n,
      output, output_stride_m, output_stride_n, output_size_m, output_size_n,
      dtype, algorithm, direction, tile_m, tile_n, block_x, block_y, block_z,
      grid_x, grid_y, grid_z, waves_per_eu, quantization, stream);
  if (const Error* e = std::get_if<Error>(&checked)) return *e;
  const Config config = std::get<Config>(checked);
  with_config<OpType::all_reduce>(config, [&]<Config C>() {
    using T = typename of_dtype<C.dtype>::t;
    using W = typename of_dtype<C.weight_dtype>::t;
    h.begin_launch(OpType::all_reduce, kernel_pointer<C>(), input_size_m, input_size_n, dtype,
        stream);
    const InputTable t = config.in_place
        ? h.input_table(input, input_stride_m, input_size_m, input_size_n, elem_bytes(dtype),
            stream)
        : InputTable{nullptr, 0, config.staging_stride_m, config.staging_stride_n};
    if constexpr (C.kernel == Kernel::all_reduce_pull_one_shot) {
      all_reduce_pull_one_shot<T, C.world, C.tile_m, C.tile_n, C.block.x, C.waves_per_eu,
          C.quantization>
          <<<dim3_of(config.grid), dim3_of(C.block), config.shared_memory_bytes, stream>>>(
        reinterpret_cast<const T* const*>(t.ptr_array), t.ptr_array_size,
        t.stride_m, t.stride_n, static_cast<int>(input_size_m),
        static_cast<int>(input_size_n), static_cast<T*>(output), output_stride_m,
        output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), h.signals_ptr_array(), h.signals_ptr_array_size(),
        h.signals_stride_parity(), h.signals_stride_block(),
        h.signals_stride_writer(), h.signals_size_parity(),
        h.signals_size_block(), h.signals_size_writer(), h.status_ptr(),
        h.status_stride_field(), h.status_size_field(), h.rank(),
        h.timeout_ticks());
    }
    if constexpr (C.kernel == Kernel::all_reduce_pull_two_shot) {
      all_reduce_pull_two_shot<T, C.world, C.tile_m, C.tile_n, C.block.x, C.waves_per_eu,
          C.contiguous, C.quantization>
          <<<dim3_of(config.grid), dim3_of(C.block), config.shared_memory_bytes, stream>>>(
        reinterpret_cast<const T* const*>(t.ptr_array), t.ptr_array_size,
        t.stride_m, t.stride_n, static_cast<int>(input_size_m),
        static_cast<int>(input_size_n), static_cast<T*>(output), output_stride_m,
        output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), h.partials_ptr_array(), h.partials_ptr_array_size(),
        config.partials_stride_m, config.partials_stride_n,
        static_cast<int>(config.partials_size_m),
        static_cast<int>(config.partials_size_n), h.signals_ptr_array(), h.signals_ptr_array_size(),
        h.signals_stride_parity(), h.signals_stride_block(),
        h.signals_stride_writer(), h.signals_size_parity(),
        h.signals_size_block(), h.signals_size_writer(), h.status_ptr(),
        h.status_stride_field(), h.status_size_field(), h.rank(),
        h.timeout_ticks());
    }
    if constexpr (C.kernel == Kernel::all_reduce_push_one_shot) {
      all_reduce_push_one_shot<T, C.world, C.tile_m, C.tile_n, C.block.x, C.waves_per_eu,
          C.quantization>
          <<<dim3_of(config.grid), dim3_of(C.block), config.shared_memory_bytes, stream>>>(
        static_cast<const T*>(input),
        input_stride_m, input_stride_n, static_cast<int>(input_size_m),
        static_cast<int>(input_size_n), static_cast<T*>(output), output_stride_m,
        output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), h.lines_ptr_array(), h.lines_ptr_array_size(),
        config.lines_stride_slot, config.lines_stride_m, config.lines_stride_n,
        static_cast<int>(config.lines_size_slot), static_cast<int>(config.lines_size_m),
        static_cast<int>(config.lines_size_n), h.epochs_ptr(), h.epochs_stride_block(),
        h.epochs_stride_n(), static_cast<int>(h.epochs_size_block()),
        static_cast<int>(h.epochs_size_n()), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks());
    }
    if constexpr (C.kernel == Kernel::all_reduce_push_two_shot) {
      all_reduce_push_two_shot<T, C.world, C.tile_m, C.tile_n, C.block.x, C.waves_per_eu,
          C.contiguous, C.quantization, C.tail>
          <<<dim3_of(config.grid), dim3_of(C.block), config.shared_memory_bytes, stream>>>(
        static_cast<const T*>(input),
        input_stride_m, input_stride_n, static_cast<int>(input_size_m),
        static_cast<int>(input_size_n), static_cast<T*>(output), output_stride_m,
        output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), h.partials_ptr_array(), h.partials_ptr_array_size(),
        config.partials_stride_m,
        config.partials_stride_n, static_cast<int>(config.partials_size_m),
        static_cast<int>(config.partials_size_n), h.staging_ptr_array(),
            h.staging_ptr_array_size(), config.staging_stride_m,
        config.staging_stride_n, static_cast<int>(config.staging_size_m),
        static_cast<int>(config.staging_size_n), h.signals_ptr_array(),
            h.signals_ptr_array_size(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks());
    }
  });
  HIP_CHECK(hipGetLastError());
  return config;
}

// =================================================================================================
// THE NORMS: output = rms_norm(all_reduce(input), weight), or with a residual
// vLLM's fused_add_rms_norm (output and residual_output), its roundings
// exactly. input [rows, hidden]; the weight `weight_dtype`, dtype or f32.
// =================================================================================================

inline std::variant<Config, Error> all_reduce_rms_norm(
    Handle& h, const void* input, int64_t input_stride_m,
    int64_t input_stride_n, int64_t input_size_m, int64_t input_size_n,
    const void* weight, int64_t weight_stride_n, int64_t weight_size_n,
    void* output, int64_t output_stride_m, int64_t output_stride_n,
    int64_t output_size_m, int64_t output_size_n, DType dtype,
    DType weight_dtype, float eps, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> tile_m,
    std::optional<int> tile_n, std::optional<int> block_x, std::optional<int> block_y, std::optional<int> block_z,
    std::optional<int> grid_x, std::optional<int> grid_y, std::optional<int> grid_z, std::optional<int> waves_per_eu,
    std::optional<int> quantization, hipStream_t stream) {
  if (const std::optional<Error> e =
          layout_refused(dtype, {{output_stride_m, output_stride_n},
                                 {input_stride_m, input_stride_n}}))
    return *e;
  if (const std::optional<Error> e =
          layout_refused(weight_dtype, {{0, weight_stride_n}}))
    return *e;
  std::variant<Config, Error> checked = all_reduce_rms_norm_config(
      h, input, input_stride_m, input_stride_n, input_size_m, input_size_n,
      weight, weight_stride_n, weight_size_n, output, output_stride_m,
      output_stride_n, output_size_m, output_size_n, dtype, weight_dtype, eps,
      algorithm, direction, tile_m, tile_n, block_x, block_y, block_z, grid_x, grid_y, grid_z,
      waves_per_eu, quantization, stream);
  if (const Error* e = std::get_if<Error>(&checked)) return *e;
  const Config config = std::get<Config>(checked);
  if (config.split) {
    const int64_t sums_stride_n = config.sums_stride_n;
    const Config reduced{config.all_reduce};
    with_config<OpType::all_reduce>(reduced, [&]<Config C>() {
      using T = typename of_dtype<C.dtype>::t;
      using W = typename of_dtype<C.weight_dtype>::t;
      h.begin_launch(OpType::all_reduce, kernel_pointer<C>(), input_size_m, input_size_n, dtype,
          stream);
      const InputTable t = reduced.in_place
          ? h.input_table(input, input_stride_m, input_size_m, input_size_n, elem_bytes(dtype),
              stream)
          : InputTable{nullptr, 0, reduced.staging_stride_m, reduced.staging_stride_n};
      if constexpr (C.kernel == Kernel::all_reduce_pull_one_shot) {
        all_reduce_pull_one_shot<T, C.world, C.tile_m, C.tile_n, C.block.x, C.waves_per_eu,
            C.quantization>
            <<<dim3_of(reduced.grid), dim3_of(C.block), reduced.shared_memory_bytes, stream>>>(
          reinterpret_cast<const T* const*>(t.ptr_array), t.ptr_array_size,
          t.stride_m, t.stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), static_cast<T*>(h.sums()), input_size_n,
          sums_stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), h.signals_ptr_array(), h.signals_ptr_array_size(),
          h.signals_stride_parity(), h.signals_stride_block(),
          h.signals_stride_writer(), h.signals_size_parity(),
          h.signals_size_block(), h.signals_size_writer(), h.status_ptr(),
          h.status_stride_field(), h.status_size_field(), h.rank(),
          h.timeout_ticks());
      }
      if constexpr (C.kernel == Kernel::all_reduce_pull_two_shot) {
        all_reduce_pull_two_shot<T, C.world, C.tile_m, C.tile_n, C.block.x, C.waves_per_eu,
            C.contiguous, C.quantization>
            <<<dim3_of(reduced.grid), dim3_of(C.block), reduced.shared_memory_bytes, stream>>>(
          reinterpret_cast<const T* const*>(t.ptr_array), t.ptr_array_size,
          t.stride_m, t.stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), static_cast<T*>(h.sums()), input_size_n,
          sums_stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), h.partials_ptr_array(), h.partials_ptr_array_size(),
          reduced.partials_stride_m, reduced.partials_stride_n,
          static_cast<int>(reduced.partials_size_m),
          static_cast<int>(reduced.partials_size_n), h.signals_ptr_array(),
              h.signals_ptr_array_size(),
          h.signals_stride_parity(), h.signals_stride_block(),
          h.signals_stride_writer(), h.signals_size_parity(),
          h.signals_size_block(), h.signals_size_writer(), h.status_ptr(),
          h.status_stride_field(), h.status_size_field(), h.rank(),
          h.timeout_ticks());
      }
      if constexpr (C.kernel == Kernel::all_reduce_push_one_shot) {
        all_reduce_push_one_shot<T, C.world, C.tile_m, C.tile_n, C.block.x, C.waves_per_eu,
            C.quantization>
            <<<dim3_of(reduced.grid), dim3_of(C.block), reduced.shared_memory_bytes, stream>>>(
          static_cast<const T*>(input),
          input_stride_m, input_stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), static_cast<T*>(h.sums()), input_size_n,
          sums_stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), h.lines_ptr_array(), h.lines_ptr_array_size(),
          reduced.lines_stride_slot, reduced.lines_stride_m, reduced.lines_stride_n,
          static_cast<int>(reduced.lines_size_slot), static_cast<int>(reduced.lines_size_m),
          static_cast<int>(reduced.lines_size_n), h.epochs_ptr(), h.epochs_stride_block(),
          h.epochs_stride_n(), static_cast<int>(h.epochs_size_block()),
          static_cast<int>(h.epochs_size_n()), h.status_ptr(), h.status_stride_field(),
          h.status_size_field(), h.rank(), h.timeout_ticks());
      }
      if constexpr (C.kernel == Kernel::all_reduce_push_two_shot) {
        all_reduce_push_two_shot<T, C.world, C.tile_m, C.tile_n, C.block.x, C.waves_per_eu,
            C.contiguous, C.quantization, C.tail>
            <<<dim3_of(reduced.grid), dim3_of(C.block), reduced.shared_memory_bytes, stream>>>(
          static_cast<const T*>(input),
          input_stride_m, input_stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), static_cast<T*>(h.sums()), input_size_n,
          sums_stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), h.partials_ptr_array(), h.partials_ptr_array_size(),
          reduced.partials_stride_m,
          reduced.partials_stride_n, static_cast<int>(reduced.partials_size_m),
          static_cast<int>(reduced.partials_size_n), h.staging_ptr_array(),
              h.staging_ptr_array_size(), reduced.staging_stride_m,
          reduced.staging_stride_n, static_cast<int>(reduced.staging_size_m),
          static_cast<int>(reduced.staging_size_n), h.signals_ptr_array(),
              h.signals_ptr_array_size(), h.signals_stride_parity(),
          h.signals_stride_block(), h.signals_stride_writer(),
          h.signals_size_parity(), h.signals_size_block(),
          h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
          h.status_size_field(), h.rank(), h.timeout_ticks());
      }
    });
    HIP_CHECK(hipGetLastError());
    with_config<OpType::all_reduce_rms_norm>(config, [&]<Config C>() {
      using T = typename of_dtype<C.dtype>::t;
      using W = typename of_dtype<C.weight_dtype>::t;
      h.begin_launch(OpType::all_reduce_rms_norm, kernel_pointer<C>(), input_size_m, input_size_n,
          dtype, stream);
      if constexpr (C.kernel == Kernel::rms_norm_rows) {
        rms_norm_rows<T, W, C.tile_m, C.tile_n, C.block.x, C.waves_per_eu, C.chunked>
            <<<dim3_of(config.grid), dim3_of(C.block), config.shared_memory_bytes, stream>>>(
          static_cast<const T*>(h.sums()),
          input_size_n, config.sums_stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), static_cast<const W*>(weight), weight_stride_n,
          static_cast<int>(weight_size_n), static_cast<T*>(output), output_stride_m,
          output_stride_n, static_cast<int>(output_size_m),
          static_cast<int>(output_size_n), eps);
      }
    });
    HIP_CHECK(hipGetLastError());
    return config;
  }
  with_config<OpType::all_reduce_rms_norm>(config, [&]<Config C>() {
    using T = typename of_dtype<C.dtype>::t;
    using W = typename of_dtype<C.weight_dtype>::t;
    h.begin_launch(OpType::all_reduce_rms_norm, kernel_pointer<C>(), input_size_m, input_size_n,
        dtype, stream);
    const InputTable t = config.quantization != Quantization::none
        ? InputTable{nullptr, 0, config.staging_stride_m, config.staging_stride_n}
        : h.input_table(input, input_stride_m, input_size_m, input_size_n, elem_bytes(dtype),
            stream);
    if constexpr (C.kernel == Kernel::all_reduce_pull_one_shot_rms_norm) {
      all_reduce_pull_one_shot_rms_norm<T, W, C.world, C.tile_n, C.block.x, C.waves_per_eu,
          C.quantization, C.chunked>
          <<<dim3_of(config.grid), dim3_of(C.block), config.shared_memory_bytes, stream>>>(
        static_cast<const T*>(input),
        input_stride_m, input_stride_n, static_cast<int>(input_size_m),
        static_cast<int>(input_size_n), reinterpret_cast<const T* const*>(t.ptr_array),
            t.ptr_array_size, t.stride_m, t.stride_n, static_cast<const W*>(weight),
            weight_stride_n,
        static_cast<int>(weight_size_n), static_cast<T*>(output), output_stride_m,
        output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), h.staging_ptr_array(), h.staging_ptr_array_size(),
            config.staging_stride_m,
        config.staging_stride_n, static_cast<int>(config.staging_size_m),
        static_cast<int>(config.staging_size_n), h.signals_ptr_array(),
            h.signals_ptr_array_size(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), eps);
    }
    if constexpr (C.kernel == Kernel::all_reduce_pull_two_shot_rms_norm) {
      all_reduce_pull_two_shot_rms_norm<T, W, C.world, C.tile_n, C.block.x, C.waves_per_eu,
          C.quantization, C.chunked, C.tail>
          <<<dim3_of(config.grid), dim3_of(C.block), config.shared_memory_bytes, stream>>>(
        static_cast<const T*>(input),
        input_stride_m, input_stride_n, static_cast<int>(input_size_m),
        static_cast<int>(input_size_n), reinterpret_cast<const T* const*>(t.ptr_array),
            t.ptr_array_size, t.stride_m, t.stride_n, static_cast<const W*>(weight),
            weight_stride_n,
        static_cast<int>(weight_size_n), static_cast<T*>(output), output_stride_m,
        output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), h.partials_ptr_array(), h.partials_ptr_array_size(),
         config.partials_stride_m,
        config.partials_stride_n, static_cast<int>(config.partials_size_m),
        static_cast<int>(config.partials_size_n), h.staging_ptr_array(),
            h.staging_ptr_array_size(), config.staging_stride_m,
        config.staging_stride_n, static_cast<int>(config.staging_size_m),
        static_cast<int>(config.staging_size_n), h.signals_ptr_array(),
            h.signals_ptr_array_size(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), eps);
    }
    if constexpr (C.kernel == Kernel::all_reduce_push_two_shot_rms_norm) {
      all_reduce_push_two_shot_rms_norm<T, W, C.world, C.tile_n, C.block.x, C.waves_per_eu,
          C.quantization, C.chunked, C.tail>
          <<<dim3_of(config.grid), dim3_of(C.block), config.shared_memory_bytes, stream>>>(
        static_cast<const T*>(input),
        input_stride_m, input_stride_n, static_cast<int>(input_size_m),
        static_cast<int>(input_size_n), reinterpret_cast<const T* const*>(t.ptr_array),
            t.ptr_array_size, t.stride_m, t.stride_n, static_cast<const W*>(weight),
            weight_stride_n,
        static_cast<int>(weight_size_n), static_cast<T*>(output), output_stride_m,
        output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), h.partials_ptr_array(), h.partials_ptr_array_size(),
         config.partials_stride_m,
        config.partials_stride_n, static_cast<int>(config.partials_size_m),
        static_cast<int>(config.partials_size_n), h.staging_ptr_array(),
            h.staging_ptr_array_size(), config.staging_stride_m,
        config.staging_stride_n, static_cast<int>(config.staging_size_m),
        static_cast<int>(config.staging_size_n), h.signals_ptr_array(),
            h.signals_ptr_array_size(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), eps);
    }
  });
  HIP_CHECK(hipGetLastError());
  return config;
}

inline std::variant<Config, Error> all_reduce_add_rms_norm(
    Handle& h, const void* input, int64_t input_stride_m,
    int64_t input_stride_n, int64_t input_size_m, int64_t input_size_n,
    const void* residual, int64_t residual_stride_m, int64_t residual_stride_n,
    int64_t residual_size_m, int64_t residual_size_n, const void* weight,
    int64_t weight_stride_n, int64_t weight_size_n, void* output,
    int64_t output_stride_m, int64_t output_stride_n, int64_t output_size_m,
    int64_t output_size_n, void* residual_output,
    int64_t residual_output_stride_m, int64_t residual_output_stride_n,
    int64_t residual_output_size_m, int64_t residual_output_size_n, DType dtype,
    DType weight_dtype, float eps, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> tile_m,
    std::optional<int> tile_n, std::optional<int> block_x, std::optional<int> block_y, std::optional<int> block_z,
    std::optional<int> grid_x, std::optional<int> grid_y, std::optional<int> grid_z, std::optional<int> waves_per_eu,
    std::optional<int> quantization, hipStream_t stream) {
  if (const std::optional<Error> e = layout_refused(
          dtype, {{output_stride_m, output_stride_n},
                  {residual_output_stride_m, residual_output_stride_n},
                  {input_stride_m, input_stride_n},
                  {residual_stride_m, residual_stride_n}}))
    return *e;
  if (const std::optional<Error> e =
          layout_refused(weight_dtype, {{0, weight_stride_n}}))
    return *e;
  std::variant<Config, Error> checked =
      all_reduce_add_rms_norm_config(
          h, input, input_stride_m, input_stride_n, input_size_m, input_size_n,
          residual, residual_stride_m, residual_stride_n, residual_size_m,
          residual_size_n, weight, weight_stride_n, weight_size_n, output,
          output_stride_m, output_stride_n, output_size_m, output_size_n,
          residual_output, residual_output_stride_m, residual_output_stride_n,
          residual_output_size_m, residual_output_size_n, dtype, weight_dtype,
          eps, algorithm, direction, tile_m, tile_n, block_x, block_y, block_z,
          grid_x, grid_y, grid_z, waves_per_eu, quantization, stream);
  if (const Error* e = std::get_if<Error>(&checked)) return *e;
  const Config config = std::get<Config>(checked);
  if (config.split) {
    const int64_t sums_stride_n = config.sums_stride_n;
    const Config reduced{config.all_reduce};
    with_config<OpType::all_reduce>(reduced, [&]<Config C>() {
      using T = typename of_dtype<C.dtype>::t;
      using W = typename of_dtype<C.weight_dtype>::t;
      h.begin_launch(OpType::all_reduce, kernel_pointer<C>(), input_size_m, input_size_n, dtype,
          stream);
      const InputTable t = reduced.in_place
          ? h.input_table(input, input_stride_m, input_size_m, input_size_n, elem_bytes(dtype),
              stream)
          : InputTable{nullptr, 0, reduced.staging_stride_m, reduced.staging_stride_n};
      if constexpr (C.kernel == Kernel::all_reduce_pull_one_shot) {
        all_reduce_pull_one_shot<T, C.world, C.tile_m, C.tile_n, C.block.x, C.waves_per_eu,
            C.quantization>
            <<<dim3_of(reduced.grid), dim3_of(C.block), reduced.shared_memory_bytes, stream>>>(
          reinterpret_cast<const T* const*>(t.ptr_array), t.ptr_array_size,
          t.stride_m, t.stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), static_cast<T*>(h.sums()), input_size_n,
          sums_stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), h.signals_ptr_array(), h.signals_ptr_array_size(),
          h.signals_stride_parity(), h.signals_stride_block(),
          h.signals_stride_writer(), h.signals_size_parity(),
          h.signals_size_block(), h.signals_size_writer(), h.status_ptr(),
          h.status_stride_field(), h.status_size_field(), h.rank(),
          h.timeout_ticks());
      }
      if constexpr (C.kernel == Kernel::all_reduce_pull_two_shot) {
        all_reduce_pull_two_shot<T, C.world, C.tile_m, C.tile_n, C.block.x, C.waves_per_eu,
            C.contiguous, C.quantization>
            <<<dim3_of(reduced.grid), dim3_of(C.block), reduced.shared_memory_bytes, stream>>>(
          reinterpret_cast<const T* const*>(t.ptr_array), t.ptr_array_size,
          t.stride_m, t.stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), static_cast<T*>(h.sums()), input_size_n,
          sums_stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), h.partials_ptr_array(), h.partials_ptr_array_size(),
          reduced.partials_stride_m, reduced.partials_stride_n,
          static_cast<int>(reduced.partials_size_m),
          static_cast<int>(reduced.partials_size_n), h.signals_ptr_array(),
              h.signals_ptr_array_size(),
          h.signals_stride_parity(), h.signals_stride_block(),
          h.signals_stride_writer(), h.signals_size_parity(),
          h.signals_size_block(), h.signals_size_writer(), h.status_ptr(),
          h.status_stride_field(), h.status_size_field(), h.rank(),
          h.timeout_ticks());
      }
      if constexpr (C.kernel == Kernel::all_reduce_push_one_shot) {
        all_reduce_push_one_shot<T, C.world, C.tile_m, C.tile_n, C.block.x, C.waves_per_eu,
            C.quantization>
            <<<dim3_of(reduced.grid), dim3_of(C.block), reduced.shared_memory_bytes, stream>>>(
          static_cast<const T*>(input),
          input_stride_m, input_stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), static_cast<T*>(h.sums()), input_size_n,
          sums_stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), h.lines_ptr_array(), h.lines_ptr_array_size(),
          reduced.lines_stride_slot, reduced.lines_stride_m, reduced.lines_stride_n,
          static_cast<int>(reduced.lines_size_slot), static_cast<int>(reduced.lines_size_m),
          static_cast<int>(reduced.lines_size_n), h.epochs_ptr(), h.epochs_stride_block(),
          h.epochs_stride_n(), static_cast<int>(h.epochs_size_block()),
          static_cast<int>(h.epochs_size_n()), h.status_ptr(), h.status_stride_field(),
          h.status_size_field(), h.rank(), h.timeout_ticks());
      }
      if constexpr (C.kernel == Kernel::all_reduce_push_two_shot) {
        all_reduce_push_two_shot<T, C.world, C.tile_m, C.tile_n, C.block.x, C.waves_per_eu,
            C.contiguous, C.quantization, C.tail>
            <<<dim3_of(reduced.grid), dim3_of(C.block), reduced.shared_memory_bytes, stream>>>(
          static_cast<const T*>(input),
          input_stride_m, input_stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), static_cast<T*>(h.sums()), input_size_n,
          sums_stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), h.partials_ptr_array(), h.partials_ptr_array_size(),
          reduced.partials_stride_m,
          reduced.partials_stride_n, static_cast<int>(reduced.partials_size_m),
          static_cast<int>(reduced.partials_size_n), h.staging_ptr_array(),
              h.staging_ptr_array_size(), reduced.staging_stride_m,
          reduced.staging_stride_n, static_cast<int>(reduced.staging_size_m),
          static_cast<int>(reduced.staging_size_n), h.signals_ptr_array(),
              h.signals_ptr_array_size(), h.signals_stride_parity(),
          h.signals_stride_block(), h.signals_stride_writer(),
          h.signals_size_parity(), h.signals_size_block(),
          h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
          h.status_size_field(), h.rank(), h.timeout_ticks());
      }
    });
    HIP_CHECK(hipGetLastError());
    with_config<OpType::all_reduce_add_rms_norm>(config, [&]<Config C>() {
      using T = typename of_dtype<C.dtype>::t;
      using W = typename of_dtype<C.weight_dtype>::t;
      h.begin_launch(OpType::all_reduce_add_rms_norm, kernel_pointer<C>(), input_size_m,
          input_size_n, dtype, stream);
      if constexpr (C.kernel == Kernel::add_rms_norm_rows) {
        add_rms_norm_rows<T, W, C.tile_m, C.tile_n, C.block.x, C.waves_per_eu, C.chunked>
            <<<dim3_of(config.grid), dim3_of(C.block), config.shared_memory_bytes, stream>>>(
          static_cast<const T*>(h.sums()),
          input_size_n, config.sums_stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), static_cast<const T*>(residual), residual_stride_m,
          residual_stride_n, static_cast<int>(residual_size_m),
          static_cast<int>(residual_size_n), static_cast<const W*>(weight), weight_stride_n,
          static_cast<int>(weight_size_n), static_cast<T*>(output), output_stride_m,
          output_stride_n, static_cast<int>(output_size_m),
          static_cast<int>(output_size_n), static_cast<T*>(residual_output),
          residual_output_stride_m, residual_output_stride_n,
          static_cast<int>(residual_output_size_m),
          static_cast<int>(residual_output_size_n), eps);
      }
    });
    HIP_CHECK(hipGetLastError());
    return config;
  }
  with_config<OpType::all_reduce_add_rms_norm>(config, [&]<Config C>() {
    using T = typename of_dtype<C.dtype>::t;
    using W = typename of_dtype<C.weight_dtype>::t;
    h.begin_launch(OpType::all_reduce_add_rms_norm, kernel_pointer<C>(), input_size_m,
        input_size_n, dtype, stream);
    const InputTable t = config.quantization != Quantization::none
        ? InputTable{nullptr, 0, config.staging_stride_m, config.staging_stride_n}
        : h.input_table(input, input_stride_m, input_size_m, input_size_n, elem_bytes(dtype),
            stream);
    if constexpr (C.kernel == Kernel::all_reduce_pull_one_shot_add_rms_norm) {
      all_reduce_pull_one_shot_add_rms_norm<T, W, C.world, C.tile_n, C.block.x, C.waves_per_eu,
          C.quantization, C.chunked>
          <<<dim3_of(config.grid), dim3_of(C.block), config.shared_memory_bytes, stream>>>(
        static_cast<const T*>(input),
        input_stride_m, input_stride_n, static_cast<int>(input_size_m),
        static_cast<int>(input_size_n), reinterpret_cast<const T* const*>(t.ptr_array),
            t.ptr_array_size, t.stride_m, t.stride_n, static_cast<const T*>(residual),
            residual_stride_m,
        residual_stride_n, static_cast<int>(residual_size_m),
        static_cast<int>(residual_size_n), static_cast<const W*>(weight), weight_stride_n,
        static_cast<int>(weight_size_n), static_cast<T*>(output), output_stride_m,
        output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), static_cast<T*>(residual_output),
        residual_output_stride_m, residual_output_stride_n,
        static_cast<int>(residual_output_size_m),
        static_cast<int>(residual_output_size_n), h.staging_ptr_array(),
            h.staging_ptr_array_size(), config.staging_stride_m,
        config.staging_stride_n, static_cast<int>(config.staging_size_m),
        static_cast<int>(config.staging_size_n), h.signals_ptr_array(),
            h.signals_ptr_array_size(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), eps);
    }
    if constexpr (C.kernel == Kernel::all_reduce_pull_two_shot_add_rms_norm) {
      all_reduce_pull_two_shot_add_rms_norm<T, W, C.world, C.tile_n, C.block.x, C.waves_per_eu,
          C.quantization, C.chunked, C.tail>
          <<<dim3_of(config.grid), dim3_of(C.block), config.shared_memory_bytes, stream>>>(
        static_cast<const T*>(input),
        input_stride_m, input_stride_n, static_cast<int>(input_size_m),
        static_cast<int>(input_size_n), reinterpret_cast<const T* const*>(t.ptr_array),
            t.ptr_array_size, t.stride_m, t.stride_n, static_cast<const T*>(residual),
            residual_stride_m,
        residual_stride_n, static_cast<int>(residual_size_m),
        static_cast<int>(residual_size_n), static_cast<const W*>(weight), weight_stride_n,
        static_cast<int>(weight_size_n), static_cast<T*>(output), output_stride_m,
        output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), static_cast<T*>(residual_output),
        residual_output_stride_m, residual_output_stride_n,
        static_cast<int>(residual_output_size_m),
        static_cast<int>(residual_output_size_n), h.partials_ptr_array(),
            h.partials_ptr_array_size(),
         config.partials_stride_m,
        config.partials_stride_n, static_cast<int>(config.partials_size_m),
        static_cast<int>(config.partials_size_n), h.scales_ptr_array(),
        h.scales_ptr_array_size(), config.scales_stride_m, static_cast<int>(config.scales_size_m),
        h.staging_ptr_array(), h.staging_ptr_array_size(), config.staging_stride_m,
        config.staging_stride_n, static_cast<int>(config.staging_size_m),
        static_cast<int>(config.staging_size_n), h.signals_ptr_array(),
            h.signals_ptr_array_size(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), eps);
    }
    if constexpr (C.kernel == Kernel::all_reduce_push_two_shot_add_rms_norm) {
      all_reduce_push_two_shot_add_rms_norm<T, W, C.world, C.tile_n, C.block.x, C.waves_per_eu,
          C.quantization, C.chunked, C.tail>
          <<<dim3_of(config.grid), dim3_of(C.block), config.shared_memory_bytes, stream>>>(
        static_cast<const T*>(input),
        input_stride_m, input_stride_n, static_cast<int>(input_size_m),
        static_cast<int>(input_size_n), reinterpret_cast<const T* const*>(t.ptr_array),
            t.ptr_array_size, t.stride_m, t.stride_n, static_cast<const T*>(residual),
            residual_stride_m,
        residual_stride_n, static_cast<int>(residual_size_m),
        static_cast<int>(residual_size_n), static_cast<const W*>(weight), weight_stride_n,
        static_cast<int>(weight_size_n), static_cast<T*>(output), output_stride_m,
        output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), static_cast<T*>(residual_output),
        residual_output_stride_m, residual_output_stride_n,
        static_cast<int>(residual_output_size_m),
        static_cast<int>(residual_output_size_n), h.partials_ptr_array(),
            h.partials_ptr_array_size(),
         config.partials_stride_m,
        config.partials_stride_n, static_cast<int>(config.partials_size_m),
        static_cast<int>(config.partials_size_n), h.scales_ptr_array(),
        h.scales_ptr_array_size(), config.scales_stride_m, static_cast<int>(config.scales_size_m),
        h.staging_ptr_array(), h.staging_ptr_array_size(), config.staging_stride_m,
        config.staging_stride_n, static_cast<int>(config.staging_size_m),
        static_cast<int>(config.staging_size_n), h.signals_ptr_array(),
            h.signals_ptr_array_size(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), eps);
    }
  });
  HIP_CHECK(hipGetLastError());
  return config;
}

// =================================================================================================
// all_reduce_add_attn_res_rms_norm: Kimi-K3's AttnRes and its RMSNorm on each
// row of the all-reduced sum of `input` (the kernels spell it out). With a
// `prefix` the sum is added to it, into `prefix_output`; without (null), the
// sum IS the new prefix. `block_output`, when not null, gets the new prefix
// too: the slot of `blocks` the caller appends to.
// =================================================================================================

inline std::variant<Config, Error>
all_reduce_add_attn_res_rms_norm(
    Handle& h, const void* input, int64_t input_stride_m,
    int64_t input_stride_n, int64_t input_size_m, int64_t input_size_n,
    const void* prefix, int64_t prefix_stride_m, int64_t prefix_stride_n,
    int64_t prefix_size_m, int64_t prefix_size_n, const void* blocks,
    int64_t blocks_stride_m, int64_t blocks_stride_r, int64_t blocks_stride_n,
    int64_t blocks_size_m, int64_t blocks_size_r, int64_t blocks_size_n,
    const void* norm_weight, int64_t norm_weight_stride_n,
    int64_t norm_weight_size_n, const void* qk_weight,
    int64_t qk_weight_stride_n, int64_t qk_weight_size_n,
    const void* output_norm_weight, int64_t output_norm_weight_stride_n,
    int64_t output_norm_weight_size_n, void* prefix_output,
    int64_t prefix_output_stride_m, int64_t prefix_output_stride_n,
    int64_t prefix_output_size_m, int64_t prefix_output_size_n, void* output,
    int64_t output_stride_m, int64_t output_stride_n, int64_t output_size_m,
    int64_t output_size_n, void* block_output, int64_t block_output_stride_m,
    int64_t block_output_stride_n, int64_t block_output_size_m,
    int64_t block_output_size_n, DType dtype, int num_blocks, float eps,
    float output_eps, std::optional<Algorithm> algorithm,
    std::optional<Direction> direction, std::optional<int> tile_m,
    std::optional<int> tile_n, std::optional<int> tile_k,
    std::optional<int> reduce_scatter_blocks,
    std::optional<int> block_x, std::optional<int> block_y, std::optional<int> block_z,
    std::optional<int> grid_x, std::optional<int> grid_y, std::optional<int> grid_z,
    std::optional<int> waves_per_eu, std::optional<int> quantization,
    hipStream_t stream) {
  if (const std::optional<Error> e = layout_refused(
          dtype, {{input_stride_m, input_stride_n},
                  {blocks_stride_m, blocks_stride_n},
                  {blocks_stride_r, blocks_stride_n},
                  {0, norm_weight_stride_n},
                  {0, qk_weight_stride_n},
                  {prefix_output_stride_m, prefix_output_stride_n},
                  {output_stride_m, output_stride_n}}))
    return *e;
  // THE OPTIONAL BUFFERS, checked only when given: a missing one is a null
  // pointer with zero strides, which is no layout at all (a prefix-less call
  // was refused as `inner_stride_not_one`, 2026-10-13).
  if (prefix != nullptr)
    if (const std::optional<Error> e =
            layout_refused(dtype, {{prefix_stride_m, prefix_stride_n}}))
      return *e;
  if (output_norm_weight != nullptr)
    if (const std::optional<Error> e =
            layout_refused(dtype, {{0, output_norm_weight_stride_n}}))
      return *e;
  if (block_output != nullptr)
    if (const std::optional<Error> e = layout_refused(
            dtype, {{block_output_stride_m, block_output_stride_n}}))
      return *e;
  std::variant<Config, Error> checked =
      all_reduce_add_attn_res_rms_norm_config(
          h, input, input_stride_m, input_stride_n, input_size_m, input_size_n,
          prefix, prefix_stride_m, prefix_stride_n, prefix_size_m,
          prefix_size_n, blocks, blocks_stride_m, blocks_stride_r,
          blocks_stride_n, blocks_size_m, blocks_size_r, blocks_size_n,
          norm_weight, norm_weight_stride_n, norm_weight_size_n, qk_weight,
          qk_weight_stride_n, qk_weight_size_n, output_norm_weight,
          output_norm_weight_stride_n, output_norm_weight_size_n, prefix_output,
          prefix_output_stride_m, prefix_output_stride_n, prefix_output_size_m,
          prefix_output_size_n, output, output_stride_m, output_stride_n,
          output_size_m, output_size_n, block_output, block_output_stride_m,
          block_output_stride_n, block_output_size_m, block_output_size_n,
          dtype, num_blocks, eps, output_eps, algorithm, direction, tile_m,
          tile_n, tile_k, reduce_scatter_blocks, block_x, block_y, block_z,
          grid_x, grid_y, grid_z, waves_per_eu, quantization, stream);
  if (const Error* e = std::get_if<Error>(&checked)) return *e;
  const Config config = std::get<Config>(checked);
  if (config.split) {
    const int64_t sums_stride_n = config.sums_stride_n;
    const Config reduced{config.all_reduce};
    with_config<OpType::all_reduce>(reduced, [&]<Config C>() {
      using T = typename of_dtype<C.dtype>::t;
      using W = typename of_dtype<C.weight_dtype>::t;
      h.begin_launch(OpType::all_reduce, kernel_pointer<C>(), input_size_m, input_size_n, dtype,
          stream);
      const InputTable t = reduced.in_place
          ? h.input_table(input, input_stride_m, input_size_m, input_size_n, elem_bytes(dtype),
              stream)
          : InputTable{nullptr, 0, reduced.staging_stride_m, reduced.staging_stride_n};
      if constexpr (C.kernel == Kernel::all_reduce_pull_one_shot) {
        all_reduce_pull_one_shot<T, C.world, C.tile_m, C.tile_n, C.block.x, C.waves_per_eu,
            C.quantization>
            <<<dim3_of(reduced.grid), dim3_of(C.block), reduced.shared_memory_bytes, stream>>>(
          reinterpret_cast<const T* const*>(t.ptr_array), t.ptr_array_size,
          t.stride_m, t.stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), static_cast<T*>(h.sums()), input_size_n,
          sums_stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), h.signals_ptr_array(), h.signals_ptr_array_size(),
          h.signals_stride_parity(), h.signals_stride_block(),
          h.signals_stride_writer(), h.signals_size_parity(),
          h.signals_size_block(), h.signals_size_writer(), h.status_ptr(),
          h.status_stride_field(), h.status_size_field(), h.rank(),
          h.timeout_ticks());
      }
      if constexpr (C.kernel == Kernel::all_reduce_pull_two_shot) {
        all_reduce_pull_two_shot<T, C.world, C.tile_m, C.tile_n, C.block.x, C.waves_per_eu,
            C.contiguous, C.quantization>
            <<<dim3_of(reduced.grid), dim3_of(C.block), reduced.shared_memory_bytes, stream>>>(
          reinterpret_cast<const T* const*>(t.ptr_array), t.ptr_array_size,
          t.stride_m, t.stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), static_cast<T*>(h.sums()), input_size_n,
          sums_stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), h.partials_ptr_array(), h.partials_ptr_array_size(),
          reduced.partials_stride_m, reduced.partials_stride_n,
          static_cast<int>(reduced.partials_size_m),
          static_cast<int>(reduced.partials_size_n), h.signals_ptr_array(),
              h.signals_ptr_array_size(),
          h.signals_stride_parity(), h.signals_stride_block(),
          h.signals_stride_writer(), h.signals_size_parity(),
          h.signals_size_block(), h.signals_size_writer(), h.status_ptr(),
          h.status_stride_field(), h.status_size_field(), h.rank(),
          h.timeout_ticks());
      }
      if constexpr (C.kernel == Kernel::all_reduce_push_one_shot) {
        all_reduce_push_one_shot<T, C.world, C.tile_m, C.tile_n, C.block.x, C.waves_per_eu,
            C.quantization>
            <<<dim3_of(reduced.grid), dim3_of(C.block), reduced.shared_memory_bytes, stream>>>(
          static_cast<const T*>(input),
          input_stride_m, input_stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), static_cast<T*>(h.sums()), input_size_n,
          sums_stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), h.lines_ptr_array(), h.lines_ptr_array_size(),
          reduced.lines_stride_slot, reduced.lines_stride_m, reduced.lines_stride_n,
          static_cast<int>(reduced.lines_size_slot), static_cast<int>(reduced.lines_size_m),
          static_cast<int>(reduced.lines_size_n), h.epochs_ptr(), h.epochs_stride_block(),
          h.epochs_stride_n(), static_cast<int>(h.epochs_size_block()),
          static_cast<int>(h.epochs_size_n()), h.status_ptr(), h.status_stride_field(),
          h.status_size_field(), h.rank(), h.timeout_ticks());
      }
      if constexpr (C.kernel == Kernel::all_reduce_push_two_shot) {
        all_reduce_push_two_shot<T, C.world, C.tile_m, C.tile_n, C.block.x, C.waves_per_eu,
            C.contiguous, C.quantization, C.tail>
            <<<dim3_of(reduced.grid), dim3_of(C.block), reduced.shared_memory_bytes, stream>>>(
          static_cast<const T*>(input),
          input_stride_m, input_stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), static_cast<T*>(h.sums()), input_size_n,
          sums_stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), h.partials_ptr_array(), h.partials_ptr_array_size(),
          reduced.partials_stride_m,
          reduced.partials_stride_n, static_cast<int>(reduced.partials_size_m),
          static_cast<int>(reduced.partials_size_n), h.staging_ptr_array(),
              h.staging_ptr_array_size(), reduced.staging_stride_m,
          reduced.staging_stride_n, static_cast<int>(reduced.staging_size_m),
          static_cast<int>(reduced.staging_size_n), h.signals_ptr_array(),
              h.signals_ptr_array_size(), h.signals_stride_parity(),
          h.signals_stride_block(), h.signals_stride_writer(),
          h.signals_size_parity(), h.signals_size_block(),
          h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
          h.status_size_field(), h.rank(), h.timeout_ticks());
      }
    });
    HIP_CHECK(hipGetLastError());
    with_config<OpType::all_reduce_add_attn_res_rms_norm>(config, [&]<Config C>() {
      using T = typename of_dtype<C.dtype>::t;
      using W = typename of_dtype<C.weight_dtype>::t;
      h.begin_launch(OpType::all_reduce_add_attn_res_rms_norm, kernel_pointer<C>(), input_size_m,
          input_size_n, dtype, stream);
      if constexpr (C.kernel == Kernel::add_attn_res_rms_norm_rows) {
        add_attn_res_rms_norm_rows<T, C.tile_m, C.tile_n, C.tile_k, C.block.x, C.waves_per_eu, C.chunked>
            <<<dim3_of(config.grid), dim3_of(C.block), config.shared_memory_bytes, stream>>>(
          static_cast<const T*>(h.sums()),
          input_size_n, config.sums_stride_n, static_cast<int>(input_size_m),
          static_cast<int>(input_size_n), static_cast<const T*>(prefix), prefix_stride_m,
          prefix_stride_n, static_cast<int>(prefix_size_m),
          static_cast<int>(prefix_size_n), static_cast<const T*>(blocks), blocks_stride_m,
          blocks_stride_r, blocks_stride_n, static_cast<int>(blocks_size_m),
          static_cast<int>(blocks_size_r), static_cast<int>(blocks_size_n),
          static_cast<const T*>(norm_weight), norm_weight_stride_n,
          static_cast<int>(norm_weight_size_n), static_cast<const T*>(qk_weight),
          qk_weight_stride_n, static_cast<int>(qk_weight_size_n),
          static_cast<const T*>(output_norm_weight), output_norm_weight_stride_n,
          static_cast<int>(output_norm_weight_size_n), static_cast<T*>(prefix_output),
          prefix_output_stride_m, prefix_output_stride_n,
          static_cast<int>(prefix_output_size_m),
          static_cast<int>(prefix_output_size_n), static_cast<T*>(output), output_stride_m,
          output_stride_n, static_cast<int>(output_size_m),
          static_cast<int>(output_size_n), static_cast<T*>(block_output),
          block_output_stride_m, block_output_stride_n,
          static_cast<int>(block_output_size_m),
          static_cast<int>(block_output_size_n), num_blocks, eps,
          output_eps);
      }
    });
    HIP_CHECK(hipGetLastError());
    return config;
  }
  with_config<OpType::all_reduce_add_attn_res_rms_norm>(config, [&]<Config C>() {
    using T = typename of_dtype<C.dtype>::t;
    using W = typename of_dtype<C.weight_dtype>::t;
    h.begin_launch(OpType::all_reduce_add_attn_res_rms_norm, kernel_pointer<C>(), input_size_m,
        input_size_n, dtype, stream);
    const InputTable t = config.quantization != Quantization::none
        ? InputTable{nullptr, 0, config.staging_stride_m, config.staging_stride_n}
        : h.input_table(input, input_stride_m, input_size_m, input_size_n, elem_bytes(dtype),
            stream);
    if constexpr (C.kernel == Kernel::all_reduce_pull_one_shot_add_attn_res_rms_norm) {
      all_reduce_pull_one_shot_add_attn_res_rms_norm<T, C.world, C.tile_n, C.tile_k, C.block.x,
          C.waves_per_eu, C.quantization, C.chunked>
          <<<dim3_of(config.grid), dim3_of(C.block), config.shared_memory_bytes, stream>>>(
        static_cast<const T*>(input),
        input_stride_m, input_stride_n, static_cast<int>(input_size_m),
        static_cast<int>(input_size_n), reinterpret_cast<const T* const*>(t.ptr_array),
            t.ptr_array_size, t.stride_m, t.stride_n, static_cast<const T*>(prefix),
            prefix_stride_m,
        prefix_stride_n, static_cast<int>(prefix_size_m),
        static_cast<int>(prefix_size_n), static_cast<const T*>(blocks), blocks_stride_m,
        blocks_stride_r, blocks_stride_n, static_cast<int>(blocks_size_m),
        static_cast<int>(blocks_size_r), static_cast<int>(blocks_size_n),
        static_cast<const T*>(norm_weight), norm_weight_stride_n,
        static_cast<int>(norm_weight_size_n), static_cast<const T*>(qk_weight),
        qk_weight_stride_n, static_cast<int>(qk_weight_size_n),
        static_cast<const T*>(output_norm_weight), output_norm_weight_stride_n,
        static_cast<int>(output_norm_weight_size_n), static_cast<T*>(prefix_output),
        prefix_output_stride_m, prefix_output_stride_n,
        static_cast<int>(prefix_output_size_m),
        static_cast<int>(prefix_output_size_n), static_cast<T*>(output), output_stride_m,
        output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), static_cast<T*>(block_output),
        block_output_stride_m, block_output_stride_n,
        static_cast<int>(block_output_size_m),
        static_cast<int>(block_output_size_n), h.signals_ptr_array(), h.signals_ptr_array_size(),
            h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), num_blocks, eps,
        output_eps);
    }
    if constexpr (C.kernel == Kernel::all_reduce_push_two_shot_add_attn_res_rms_norm) {
      all_reduce_push_two_shot_add_attn_res_rms_norm<T, C.world, C.tile_n, C.tile_k, C.block.x,
          C.waves_per_eu, C.quantization, C.chunked, C.tail>
          <<<dim3_of(config.grid), dim3_of(C.block), config.shared_memory_bytes, stream>>>(
        static_cast<const T*>(input),
        input_stride_m, input_stride_n, static_cast<int>(input_size_m),
        static_cast<int>(input_size_n), reinterpret_cast<const T* const*>(t.ptr_array),
            t.ptr_array_size, t.stride_m, t.stride_n, static_cast<const T*>(prefix),
            prefix_stride_m,
        prefix_stride_n, static_cast<int>(prefix_size_m),
        static_cast<int>(prefix_size_n), static_cast<const T*>(blocks), blocks_stride_m,
        blocks_stride_r, blocks_stride_n, static_cast<int>(blocks_size_m),
        static_cast<int>(blocks_size_r), static_cast<int>(blocks_size_n),
        static_cast<const T*>(norm_weight), norm_weight_stride_n,
        static_cast<int>(norm_weight_size_n), static_cast<const T*>(qk_weight),
        qk_weight_stride_n, static_cast<int>(qk_weight_size_n),
        static_cast<const T*>(output_norm_weight), output_norm_weight_stride_n,
        static_cast<int>(output_norm_weight_size_n), static_cast<T*>(prefix_output),
        prefix_output_stride_m, prefix_output_stride_n,
        static_cast<int>(prefix_output_size_m),
        static_cast<int>(prefix_output_size_n), static_cast<T*>(output), output_stride_m,
        output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), static_cast<T*>(block_output),
        block_output_stride_m, block_output_stride_n,
        static_cast<int>(block_output_size_m),
        static_cast<int>(block_output_size_n), h.partials_ptr_array(),
        h.partials_ptr_array_size(), config.partials_stride_m, config.partials_stride_n,
        static_cast<int>(config.partials_size_m), static_cast<int>(config.partials_size_n),
        h.staging_ptr_array(), h.staging_ptr_array_size(), config.staging_stride_m,
        config.staging_stride_n, static_cast<int>(config.staging_size_m),
        static_cast<int>(config.staging_size_n), h.signals_ptr_array(),
            h.signals_ptr_array_size(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), num_blocks, eps,
        output_eps);
    }
    if constexpr (C.kernel == Kernel::all_reduce_pull_two_shot_add_attn_res_rms_norm) {
      all_reduce_pull_two_shot_add_attn_res_rms_norm<T, C.world, C.tile_m, C.tile_n, C.tile_k,
          C.block.x, C.waves_per_eu, C.quantization, C.chunked, C.tail>
          <<<dim3_of(config.grid), dim3_of(C.block), config.shared_memory_bytes, stream>>>(
        static_cast<const T*>(input),
        input_stride_m, input_stride_n, static_cast<int>(input_size_m),
        static_cast<int>(input_size_n), reinterpret_cast<const T* const*>(t.ptr_array),
            t.ptr_array_size, t.stride_m, t.stride_n, static_cast<const T*>(prefix),
            prefix_stride_m,
        prefix_stride_n, static_cast<int>(prefix_size_m),
        static_cast<int>(prefix_size_n), static_cast<const T*>(blocks), blocks_stride_m,
        blocks_stride_r, blocks_stride_n, static_cast<int>(blocks_size_m),
        static_cast<int>(blocks_size_r), static_cast<int>(blocks_size_n),
        static_cast<const T*>(norm_weight), norm_weight_stride_n,
        static_cast<int>(norm_weight_size_n), static_cast<const T*>(qk_weight),
        qk_weight_stride_n, static_cast<int>(qk_weight_size_n),
        static_cast<const T*>(output_norm_weight), output_norm_weight_stride_n,
        static_cast<int>(output_norm_weight_size_n), static_cast<T*>(prefix_output),
        prefix_output_stride_m, prefix_output_stride_n,
        static_cast<int>(prefix_output_size_m),
        static_cast<int>(prefix_output_size_n), static_cast<T*>(output), output_stride_m,
        output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), static_cast<T*>(block_output),
        block_output_stride_m, block_output_stride_n,
        static_cast<int>(block_output_size_m),
        static_cast<int>(block_output_size_n), h.partials_ptr_array(),
        h.partials_ptr_array_size(), config.partials_stride_m, config.partials_stride_n,
        static_cast<int>(config.partials_size_m), static_cast<int>(config.partials_size_n),
        h.staging_ptr_array(), h.staging_ptr_array_size(), config.staging_stride_m,
        config.staging_stride_n, static_cast<int>(config.staging_size_m),
        static_cast<int>(config.staging_size_n), h.signals_ptr_array(),
            h.signals_ptr_array_size(), h.signals_stride_parity(),
        h.signals_stride_block(), h.signals_stride_writer(),
        h.signals_size_parity(), h.signals_size_block(),
        h.signals_size_writer(), h.status_ptr(), h.status_stride_field(),
        h.status_size_field(), h.rank(), h.timeout_ticks(), num_blocks, eps,
        output_eps, config.reduce_scatter_blocks);
    }
  });
  HIP_CHECK(hipGetLastError());
  return config;
}

}  // namespace hip_custom_communicator
