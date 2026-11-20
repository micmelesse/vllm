#pragma once

// Host side of the all-reduce: checks the call, then launches its kernel.

#include "../../ops.h"
#include "../types.cuh"
#include "../utils.h"
#include "../kernels/all_reduce.cuh"

ErrorCode hip_custom_communicator_all_reduce(
    int64_t state, torch::Tensor& input, torch::Tensor& output,
    const std::string& algorithm_name, const std::string& protocol_name,
    int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t block_x,
    int64_t block_y, int64_t block_z, int64_t waves_per_eu, int64_t tile_m,
    int64_t tile_n, const std::string& quantization_name, int64_t pack_bytes) {
  using namespace hip_custom_communicator;
  using namespace hip_custom_communicator::internal;

  if (state == 0) return error_code(Error::not_open);
  CommsState& s = *reinterpret_cast<CommsState*>(state);
  if (!input.is_cuda() || !output.is_cuda())
    return error_code(Error::not_on_device);

  if (input.dim() != 2 || input.size(0) < 1 || input.size(1) < 1 ||
      output.sizes() != input.sizes())
    return error_code(Error::shape_mismatch);
  if (output.scalar_type() != input.scalar_type())
    return error_code(Error::dtype_mismatch);
  for (const int64_t v : {waves_per_eu, tile_m, tile_n, pack_bytes, grid_x,
                          grid_y, grid_z, block_x, block_y, block_z})
    if (v < std::numeric_limits<int>::min() ||
        v > std::numeric_limits<int>::max())
      return error_code(Error::field_out_of_range);
  const void* input_ptr = input.data_ptr();
  const int64_t input_stride_m = input.stride(0),
                input_stride_n = input.stride(1);
  const int64_t input_size_m = input.size(0), input_size_n = input.size(1);
  void* output_ptr = output.data_ptr();
  const int64_t output_stride_m = output.stride(0),
                output_stride_n = output.stride(1);
  const int64_t output_size_m = output.size(0), output_size_n = output.size(1);
  const at::ScalarType dtype = input.scalar_type();
  const int64_t e = input.element_size();

  const int64_t m = input_size_m, n = input_size_n;
  const dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y),
                  static_cast<unsigned>(grid_z));
  const dim3 block(static_cast<unsigned>(block_x),
                   static_cast<unsigned>(block_y),
                   static_cast<unsigned>(block_z));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();

  AllReduceAlgorithm algorithm;
  if (algorithm_name == "one_shot")
    algorithm = AllReduceAlgorithm::one_shot;
  else if (algorithm_name == "two_shot")
    algorithm = AllReduceAlgorithm::two_shot;
  else
    return error_code(Error::no_such_kernel);
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

  if (input_stride_n != 1 || output_stride_n != 1)
    return error_code(Error::inner_stride_not_one);
  if (input_stride_m * e % pack_bytes != 0 ||
      output_stride_m * e % pack_bytes != 0)
    return error_code(Error::row_stride_not_packs);

  if (m > 0 && n > 0) {
    const char* input_first = static_cast<const char*>(input_ptr);
    const char* output_first = static_cast<const char*>(output_ptr);
    const char* input_last =
        input_first +
        ((m - 1) * input_stride_m + (n - 1) * input_stride_n + 1) * e;
    const char* output_last =
        output_first +
        ((m - 1) * output_stride_m + (n - 1) * output_stride_n + 1) * e;
    if (input_first < output_last && output_first < input_last)
      return error_code(Error::inputs_overlap_outputs);
  }

  const bool quantize_int4 = int4_config;

  const int shared_memory_bytes = 0;

  const bool two_shot = algorithm == AllReduceAlgorithm::two_shot;
  const bool contiguous = two_shot && input_stride_m == n * input_stride_n &&
                          output_stride_m == n * output_stride_n;
  const bool tail = quantize_int4 &&
                    (two_shot ? m * n % kInt4Group != 0 : n % kInt4Group != 0);

  // The message a block sends a round, in elements: the two-shot's chunk
  // (exact, the tile; INT4, a group a lane), the one-shot's tile row.
  if (two_shot && protocol != Protocol::simple_write &&
      protocol != Protocol::simple_read)
    return error_code(Error::not_built);
  const int64_t elems = pack_bytes / e;
  const bool simple_protocol =
      protocol == Protocol::simple_write || protocol == Protocol::simple_read;
  const int64_t row_elems =
      two_shot ? (quantize_int4 ? block_x / s.world * kInt4Group
                                : (tile_n + elems - 1) / elems * elems)
      : simple_protocol
          ? (tile_n + elems * block_x - 1) / (elems * block_x) * elems * block_x
          : tile_n;
  const std::optional<ProtocolBuffers> protocol_buffers =
      layout_protocol_buffers(s, protocol, two_shot ? grid_x : grid_x * tile_m,
                              row_elems, e, pack_bytes, quantize_int4);
  if (!protocol_buffers) return error_code(Error::unfenced_too_small);
  void* kernel = nullptr;

  for (const auto& i : kAllReduceInstances)
    if (i.algorithm == algorithm && i.protocol == protocol &&
        i.block_x == block_x && i.block_y == block_y && i.block_z == block_z &&
        i.waves_per_eu == waves_per_eu && i.tile_m == tile_m &&
        i.tile_n == tile_n &&
        i.quantization ==
            (int4_config ? Quantization::int4 : Quantization::exact) &&
        i.pack_bytes == pack_bytes && i.world == s.world &&
        i.wave_size == s.device.wave_size && i.dtype == dtype &&
        i.contiguous == contiguous && i.tail == tail &&
        i.quantize_int4 == quantize_int4)
      kernel = i.kernel;
  if (kernel == nullptr) return error_code(Error::not_built);

  hipFuncAttributes attributes;
  if (hipFuncGetAttributes(&attributes, kernel) != hipSuccess)
    return error_code(Error::hip_failed);
  const int64_t threads = block_x * block_y * block_z;
  if (threads > attributes.maxThreadsPerBlock)
    return error_code(Error::block_too_large);
  if (static_cast<int64_t>(attributes.sharedSizeBytes) + shared_memory_bytes >
      s.device.lds_bytes)
    return error_code(Error::lds_too_large);
  if (static_cast<int64_t>(attributes.localSizeBytes) > s.device.scratch_a_lane)
    return error_code(Error::scratch_too_large);

  int blocks_a_cu = 0;
  if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_a_cu, kernel, static_cast<int>(threads),
          shared_memory_bytes) != hipSuccess)
    return error_code(Error::hip_failed);
  if (grid_x > blocks_a_cu * s.device.compute_units)
    return error_code(Error::grid_not_resident);

  const auto launch = [&]<typename T, typename W>() {
    const auto typed = reinterpret_cast<AllReduceKernel<T, W>>(kernel);
    typed<<<grid, block, shared_memory_bytes, stream>>>(
        static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
        static_cast<int>(input_size_m), static_cast<int>(input_size_n),
        static_cast<T*>(output_ptr), output_stride_m, output_stride_n,
        static_cast<int>(output_size_m), static_cast<int>(output_size_n),
        *protocol_buffers, s.rank);
  };
  if (dtype == at::ScalarType::BFloat16)
    launch.template operator()<c10::BFloat16, c10::BFloat16>();
  else
    launch.template operator()<c10::Half, c10::Half>();

  if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
  return std::nullopt;
}
