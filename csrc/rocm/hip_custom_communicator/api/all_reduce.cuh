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

  if (protocol == Protocol::ll) {
    const int64_t line_pack = std::min<int64_t>(pack_bytes, kLinePayloadBytes);
    const int64_t row_lines =
        quantize_int4 ? (tile_n + kInt4Group - 1) / kInt4Group * kInt4GroupPacks
                      : (tile_n * e + line_pack - 1) / line_pack;
    if (row_lines > s.symmetric.unfenced_slot_bytes / kLinePayloadBytes ||
        grid_x * tile_m > s.symmetric.unfenced_blocks)
      return error_code(Error::unfenced_too_small);
  }
  if (protocol == Protocol::lamport &&
      ((tile_n * e + pack_bytes - 1) / pack_bytes * pack_bytes >
           s.symmetric.unfenced_slot_bytes ||
       grid_x * tile_m > s.symmetric.lamport_blocks))
    return error_code(Error::unfenced_too_small);
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

  int64_t unfenced_stride_slot = 0, unfenced_stride_m = 0,
          unfenced_stride_n = 0, unfenced_size_slot = 0, unfenced_size_m = 0,
          unfenced_size_n = 0;
  if (algorithm == AllReduceAlgorithm::two_shot) {
    // The two-shot is the Simple protocols': a slot a pair of ranks, a row a
    // block, each a chunk (exact, the tile; INT4, a group a lane, as the
    // stage's).
    if (protocol != Protocol::simple_write && protocol != Protocol::simple_read)
      return error_code(Error::not_built);
    const int64_t elems = pack_bytes / e;
    const int64_t lanes = block_x / s.world;
    const int64_t row = quantize_int4 ? lanes * kInt4Group
                                      : (tile_n + elems - 1) / elems * elems;
    if (s.world * grid_x * row * e > s.symmetric.inbox_bytes ||
        grid_x > s.symmetric.max_blocks)
      return error_code(Error::unfenced_too_small);
    unfenced_stride_slot = grid_x * row, unfenced_stride_m = row,
    unfenced_stride_n = 1;
    unfenced_size_slot = s.world, unfenced_size_m = grid_x,
    unfenced_size_n = row;
  } else if (protocol == Protocol::simple_write ||
             protocol == Protocol::simple_read) {
    // The one-shot's simple_write and simple_read inbox: a slot a pair of
    // ranks (the writer's in the reader's inbox, or the reader's in the
    // writer's), a row a block, each the tile.
    if (grid_x * tile_m > s.symmetric.lamport_blocks ||
        (tile_n * e + pack_bytes - 1) / pack_bytes * pack_bytes >
            s.symmetric.unfenced_slot_bytes)
      return error_code(Error::unfenced_too_small);
    const int64_t blocks = s.symmetric.lamport_blocks;
    unfenced_stride_slot = blocks * (s.symmetric.unfenced_slot_bytes / e);
    unfenced_stride_m = s.symmetric.unfenced_slot_bytes / e,
    unfenced_stride_n = 1;
    unfenced_size_slot = s.world, unfenced_size_m = blocks,
    unfenced_size_n = s.symmetric.unfenced_slot_bytes / e;
  } else {
    // LL's and Lamport's inboxes have the same rows; LL's lines are twice
    // as wide, half data.
    const int64_t blocks = protocol == Protocol::lamport
                               ? s.symmetric.lamport_blocks
                               : s.symmetric.unfenced_blocks;
    const int64_t banks =
        protocol == Protocol::lamport ? kLamportBanks : kLlBanks;
    unfenced_stride_slot =
        banks * blocks * (s.symmetric.unfenced_slot_bytes / e);
    unfenced_stride_m = s.symmetric.unfenced_slot_bytes / e,
    unfenced_stride_n = 1;
    unfenced_size_slot = s.world, unfenced_size_m = banks * blocks,
    unfenced_size_n = s.symmetric.unfenced_slot_bytes / e;
  }

  char* const* const inbox_ptr_array =
      protocol == Protocol::lamport ? s.symmetric.lamport_ptr_array
      : protocol == Protocol::simple_write
          ? s.symmetric.simple_write_inbox_ptr_array
      : protocol == Protocol::simple_read
          ? s.symmetric.simple_read_inbox_ptr_array
          : s.symmetric.unfenced_ptr_array;
  // This protocol's connections and counters (CONNECTION, COUNTERS).
  int32_t* const connections = s.connections + static_cast<int>(protocol) *
                                                   s.connection_blocks *
                                                   s.connection_words;
  uint32_t* const* const counters_ptr_array =
      protocol == Protocol::simple_read
          ? s.symmetric.simple_read_counters_ptr_array
          : s.symmetric.simple_write_counters_ptr_array;
  const auto launch = [&]<typename T, typename W>() {
    const auto typed = reinterpret_cast<AllReduceKernel<T, W>>(kernel);
    typed<<<grid, block, shared_memory_bytes, stream>>>(
        static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
        static_cast<int>(input_size_m), static_cast<int>(input_size_n),
        static_cast<T*>(output_ptr), output_stride_m, output_stride_n,
        static_cast<int>(output_size_m), static_cast<int>(output_size_n),
        inbox_ptr_array, s.world, unfenced_stride_slot, unfenced_stride_m,
        unfenced_stride_n, static_cast<int>(unfenced_size_slot),
        static_cast<int>(unfenced_size_m), static_cast<int>(unfenced_size_n),
        connections, s.connection_words, int64_t{1},
        static_cast<int>(s.connection_blocks),
        static_cast<int>(s.connection_words), counters_ptr_array, s.world,
        int64_t{s.symmetric.max_blocks * s.world}, int64_t{s.world}, int64_t{1},
        static_cast<int>(s.symmetric.max_blocks), s.world, s.status_device,
        int64_t{1}, int64_t{kStatusFields}, s.rank, s.timeout_ticks);
  };
  if (dtype == at::ScalarType::BFloat16)
    launch.template operator()<c10::BFloat16, c10::BFloat16>();
  else
    launch.template operator()<c10::Half, c10::Half>();

  if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
  return std::nullopt;
}
