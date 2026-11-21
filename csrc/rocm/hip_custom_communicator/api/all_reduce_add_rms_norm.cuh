#pragma once

// Host side of the all-reduce with RMSNorm: checks the call, then launches its
// kernel.

#include "../../ops.h"
#include "../types.cuh"
#include "../utils.h"
#include "../kernels/all_reduce.cuh"
#include "../kernels/all_reduce_add_rms_norm.cuh"

ErrorCode hip_custom_communicator_all_reduce_add_rms_norm(
    int64_t state, torch::Tensor& input,
    const std::optional<torch::Tensor>& residual, torch::Tensor& weight,
    torch::Tensor& output, const std::optional<torch::Tensor>& residual_output,
    double eps, const std::string& algorithm_name,
    const std::string& protocol_name, int64_t grid_x, int64_t grid_y,
    int64_t grid_z, int64_t block_x, int64_t block_y, int64_t block_z,
    int64_t waves_per_eu, int64_t tile_m, int64_t tile_n,
    const std::string& quantization_name, int64_t pack_bytes) {
  using namespace hip_custom_communicator;
  using namespace hip_custom_communicator::internal;

  if (state == 0) return error_code(Error::not_open);
  CommsState& s = *reinterpret_cast<CommsState*>(state);
  if (!input.is_cuda() || !weight.is_cuda() || !output.is_cuda())
    return error_code(Error::not_on_device);

  if (input.dim() != 2 || input.size(0) < 1 || input.size(1) < 1 ||
      output.sizes() != input.sizes() || weight.dim() != 1 ||
      weight.numel() != input.size(1))
    return error_code(Error::shape_mismatch);
  if (output.scalar_type() != input.scalar_type())
    return error_code(Error::dtype_mismatch);

  const bool add = residual.has_value();
  if (residual_output.has_value() != add)
    return error_code(Error::shape_mismatch);
  if (add && (!residual->is_cuda() || !residual_output->is_cuda()))
    return error_code(Error::not_on_device);
  if (add && (residual->sizes() != input.sizes() ||
              residual_output->sizes() != input.sizes()))
    return error_code(Error::shape_mismatch);
  if (add && (residual->scalar_type() != input.scalar_type() ||
              residual_output->scalar_type() != input.scalar_type()))
    return error_code(Error::dtype_mismatch);
  for (const int64_t v : {waves_per_eu, tile_m, tile_n, pack_bytes, grid_x,
                          grid_y, grid_z, block_x, block_y, block_z})
    if (v < std::numeric_limits<int>::min() ||
        v > std::numeric_limits<int>::max())
      return error_code(Error::field_out_of_range);

  Protocol protocol;
  if (protocol_name == "ll")
    protocol = Protocol::ll;
  else if (protocol_name == "lamport")
    protocol = Protocol::lamport;
  else if (protocol_name == "simple_write")
    protocol = Protocol::simple_write;
  else if (protocol_name == "simple_read")
    protocol = Protocol::simple_read;
  else
    return error_code(Error::no_such_protocol);

  AllReduceAddRmsNormKernelAlgorithm algorithm{};
  if (algorithm_name == "one_shot")
    algorithm = AllReduceAddRmsNormKernelAlgorithm::one_shot;
  else if (algorithm_name == "two_shot")
    algorithm = AllReduceAddRmsNormKernelAlgorithm::two_shot;
  else
    return error_code(Error::no_such_kernel);
  const void* input_ptr = input.data_ptr();
  const int64_t input_stride_m = input.stride(0),
                input_stride_n = input.stride(1);
  const int64_t input_size_m = input.size(0), input_size_n = input.size(1);
  const void* weight_ptr = weight.data_ptr();
  const int64_t weight_stride_n = weight.stride(0),
                weight_size_n = weight.size(0);

  const void* residual_ptr = add ? residual->data_ptr() : nullptr;
  const int64_t residual_stride_m = add ? residual->stride(0) : 0,
                residual_stride_n = add ? residual->stride(1) : 0,
                residual_size_m = add ? residual->size(0) : 0,
                residual_size_n = add ? residual->size(1) : 0;
  void* residual_output_ptr = add ? residual_output->data_ptr() : nullptr;
  const int64_t residual_output_stride_m = add ? residual_output->stride(0) : 0,
                residual_output_stride_n = add ? residual_output->stride(1) : 0,
                residual_output_size_m = add ? residual_output->size(0) : 0,
                residual_output_size_n = add ? residual_output->size(1) : 0;
  void* output_ptr = output.data_ptr();
  const int64_t output_stride_m = output.stride(0),
                output_stride_n = output.stride(1);
  const int64_t output_size_m = output.size(0), output_size_n = output.size(1);
  const at::ScalarType dtype = input.scalar_type();
  const at::ScalarType weight_dtype = weight.scalar_type();
  const int64_t e = input.element_size();

  const int64_t m = input_size_m, n = input_size_n;
  const dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y),
                  static_cast<unsigned>(grid_z));
  const dim3 block(static_cast<unsigned>(block_x),
                   static_cast<unsigned>(block_y),
                   static_cast<unsigned>(block_z));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();

  if (quantization_name != "exact" && quantization_name != "int4")
    return error_code(Error::no_such_quantization);
  const bool int4_config = quantization_name == "int4";

  if (grid_y != 1 || grid_z != 1 || grid_x < 1 ||
      grid_x > s.symmetric.max_blocks)
    return error_code(Error::launch_out_of_range);
  if (tile_m < 1 || tile_n < 1) return error_code(Error::field_not_positive);

  if (pack_bytes < 2 || pack_bytes > kPackBytes ||
      (pack_bytes & (pack_bytes - 1)) != 0)
    return error_code(Error::pack_out_of_range);
  if (int4_config && pack_bytes != kPackBytes)
    return error_code(Error::quantization_not_supported);

  if (std::find(kDtypes.begin(), kDtypes.end(), dtype) == kDtypes.end())
    return error_code(Error::dtype_not_built);
  if (pack_bytes < e) return error_code(Error::pack_out_of_range);
  if (n * e % pack_bytes != 0) return error_code(Error::row_not_packs);

  if (reinterpret_cast<uintptr_t>(weight_ptr) %
          (kPackBytes / e * weight.element_size()) !=
      0)
    return error_code(Error::not_aligned);

  if (input_stride_n != 1 || output_stride_n != 1 || weight_stride_n != 1 ||
      (add && (residual_stride_n != 1 || residual_output_stride_n != 1)))
    return error_code(Error::inner_stride_not_one);
  if (input_stride_m * e % pack_bytes != 0 ||
      output_stride_m * e % pack_bytes != 0 ||
      (add && (residual_stride_m * e % pack_bytes != 0 ||
               residual_output_stride_m * e % pack_bytes != 0)))
    return error_code(Error::row_stride_not_packs);

  if (m > 0 && n > 0) {
    const auto first = [](const void* p) {
      return static_cast<const char*>(p);
    };
    const auto last = [&](const void* p, int64_t stride_m, int64_t stride_n,
                          int64_t rows, int64_t elem) {
      return first(p) + ((rows - 1) * stride_m + (n - 1) * stride_n + 1) * elem;
    };
    const auto meet = [&](const void* a, const char* a_last, const void* b,
                          const char* b_last) {
      return a != nullptr && b != nullptr && first(a) < b_last &&
             first(b) < a_last;
    };
    const char* input_last =
        last(input_ptr, input_stride_m, input_stride_n, m, e);
    const char* weight_last =
        last(weight_ptr, 0, weight_stride_n, 1, weight.element_size());
    const char* residual_last =
        add ? last(residual_ptr, residual_stride_m, residual_stride_n, m, e)
            : nullptr;
    const char* output_last =
        last(output_ptr, output_stride_m, output_stride_n, m, e);
    const char* residual_output_last =
        add ? last(residual_output_ptr, residual_output_stride_m,
                   residual_output_stride_n, m, e)
            : nullptr;
    for (const auto& [out, out_last] :
         {std::pair{static_cast<const void*>(output_ptr), output_last},
          std::pair{static_cast<const void*>(residual_output_ptr),
                    residual_output_last}})
      if (meet(input_ptr, input_last, out, out_last) ||
          meet(weight_ptr, weight_last, out, out_last) ||
          meet(residual_ptr, residual_last, out, out_last))
        return error_code(Error::inputs_overlap_outputs);
    if (meet(output_ptr, output_last, residual_output_ptr,
             residual_output_last))
      return error_code(Error::inputs_overlap_outputs);
  }

  const bool quantize_int4 = int4_config;
  const int shared_memory_bytes = 0;

  void* kernel = nullptr;
  const bool chunked = n > tile_n;
  const bool tail =
      quantize_int4 && (n % kInt4Group != 0 || n / kInt4Group % s.world != 0);

  for (const auto& i : kAllReduceAddRmsNormInstances)
    if (i.algorithm == algorithm && i.protocol == protocol &&
        i.residual == add && i.block_x == block_x && i.block_y == block_y &&
        i.block_z == block_z && i.waves_per_eu == waves_per_eu &&
        i.tile_m == tile_m && i.tile_n == tile_n &&
        i.quantization ==
            (int4_config ? Quantization::int4 : Quantization::exact) &&
        i.pack_bytes == pack_bytes && i.world == s.world &&
        i.wave_size == s.device.wave_size && i.dtype == dtype &&
        i.weight_dtype == weight_dtype && i.chunked == chunked &&
        i.tail == tail && i.quantize_int4 == quantize_int4)
      kernel = i.kernel;
  if (kernel == nullptr) return error_code(Error::not_built);

  hipFuncAttributes attributes;
  if (hipFuncGetAttributes(&attributes, kernel) != hipSuccess)
    return error_code(Error::hip_failed);
  if (block_x * block_y * block_z > attributes.maxThreadsPerBlock)
    return error_code(Error::block_too_large);
  if (static_cast<int64_t>(attributes.sharedSizeBytes) + shared_memory_bytes >
      s.device.lds_bytes)
    return error_code(Error::lds_too_large);
  if (static_cast<int64_t>(attributes.localSizeBytes) > s.device.scratch_a_lane)
    return error_code(Error::scratch_too_large);

  int blocks_a_cu = 0;
  if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_a_cu, kernel, static_cast<int>(block_x * block_y * block_z),
          shared_memory_bytes) != hipSuccess)
    return error_code(Error::hip_failed);
  if (grid_x > blocks_a_cu * s.device.compute_units)
    return error_code(Error::grid_not_resident);

  // The Simple protocols' inbox: a slot a pair of ranks, a row a block, each
  // the tile's span (the one-shot's) or a chunk of a share (the two-shot's).
  // The message a block sends a round, in elements: the one-shot's tile
  // row, the two-shot's share (exact, the tile; INT4, a group a lane).
  const int64_t elems = pack_bytes / e;
  const int64_t span =
      algorithm == AllReduceAddRmsNormKernelAlgorithm::one_shot
          ? (tile_n + elems * block_x - 1) / (elems * block_x) * elems * block_x
      : quantize_int4 ? block_x / s.world * kInt4Group
                      : (tile_n + elems - 1) / elems * elems;
  const std::optional<ProtocolBuffers> protocol_buffers =
      layout_protocol_buffers(s, protocol, grid_x, span, e, pack_bytes,
                              quantize_int4);
  if (!protocol_buffers) return error_code(Error::unfenced_too_small);
  const auto launch = [&]<typename T, typename W>() {
    const auto typed =
        reinterpret_cast<AllReduceAddRmsNormKernel<T, W>>(kernel);
    typed<<<grid, block, shared_memory_bytes, stream>>>(
        static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
        static_cast<int>(input_size_m), static_cast<int>(input_size_n),
        static_cast<const T*>(residual_ptr), residual_stride_m,
        residual_stride_n, static_cast<int>(residual_size_m),
        static_cast<int>(residual_size_n), static_cast<const W*>(weight_ptr),
        weight_stride_n, static_cast<int>(weight_size_n),
        static_cast<T*>(output_ptr), output_stride_m, output_stride_n,
        static_cast<int>(output_size_m), static_cast<int>(output_size_n),
        static_cast<T*>(residual_output_ptr), residual_output_stride_m,
        residual_output_stride_n, static_cast<int>(residual_output_size_m),
        static_cast<int>(residual_output_size_n), *protocol_buffers, eps);
  };
  if (dtype == at::ScalarType::BFloat16 &&
      weight_dtype == at::ScalarType::Float)
    launch.template operator()<c10::BFloat16, float>();
  else if (dtype == at::ScalarType::BFloat16)
    launch.template operator()<c10::BFloat16, c10::BFloat16>();
  else if (weight_dtype == at::ScalarType::Float)
    launch.template operator()<c10::Half, float>();
  else
    launch.template operator()<c10::Half, c10::Half>();

  if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
  return std::nullopt;
}
