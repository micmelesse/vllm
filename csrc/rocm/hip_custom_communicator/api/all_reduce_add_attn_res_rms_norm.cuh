#pragma once

// Host side of the all-reduce with AttnRes: checks the call, then launches its
// kernel.

#include "../../ops.h"
#include "../types.cuh"
#include "../utils.h"
#include "../kernels/all_reduce.cuh"
#include "../kernels/all_reduce_add_attn_res_rms_norm.cuh"

ErrorCode hip_custom_communicator_all_reduce_add_attn_res_rms_norm(
    int64_t state, torch::Tensor& input,
    const std::optional<torch::Tensor>& prefix, torch::Tensor& blocks,
    torch::Tensor& norm_weight, torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& output_norm_weight,
    torch::Tensor& prefix_output, torch::Tensor& output,
    const std::optional<torch::Tensor>& block_output, int64_t num_blocks,
    double eps, double output_eps, const std::string& algorithm_name,
    const std::string& protocol_name, int64_t grid_x, int64_t grid_y,
    int64_t grid_z, int64_t block_x, int64_t block_y, int64_t block_z,
    int64_t waves_per_eu, int64_t tile_m, int64_t tile_n, int64_t tile_k,
    const std::string& quantization_name, int64_t pack_bytes,
    std::optional<int64_t> add_attn_res_rms_norm_grid_x,
    std::optional<int64_t> add_attn_res_rms_norm_grid_y,
    std::optional<int64_t> add_attn_res_rms_norm_grid_z,
    std::optional<int64_t> add_attn_res_rms_norm_block_x,
    std::optional<int64_t> add_attn_res_rms_norm_block_y,
    std::optional<int64_t> add_attn_res_rms_norm_block_z,
    std::optional<int64_t> add_attn_res_rms_norm_waves_per_eu,
    std::optional<int64_t> add_attn_res_rms_norm_tile_m,
    std::optional<int64_t> add_attn_res_rms_norm_tile_n,
    std::optional<int64_t> add_attn_res_rms_norm_tile_k,
    std::optional<int64_t> add_attn_res_rms_norm_pack_bytes) {
  using namespace hip_custom_communicator;
  using namespace hip_custom_communicator::internal;

  if (state == 0) return error_code(Error::not_open);

  if (!input.is_cuda() || !norm_weight.is_cuda() || !qk_weight.is_cuda() ||
      !prefix_output.is_cuda() || !output.is_cuda() || !blocks.is_cuda())
    return error_code(Error::not_on_device);

  if (input.dim() != 2 || input.size(0) < 1 || input.size(1) < 1)
    return error_code(Error::shape_mismatch);
  const int64_t width = input.size(1);
  if (prefix_output.sizes() != input.sizes() ||
      output.sizes() != input.sizes() || blocks.dim() != 3 ||
      blocks.size(0) != input.size(0) || blocks.size(2) != width ||
      blocks.stride(2) != 1 || num_blocks < 0 || num_blocks > blocks.size(1) ||
      norm_weight.dim() != 1 || norm_weight.numel() != width ||
      qk_weight.dim() != 1 || qk_weight.numel() != width)
    return error_code(Error::shape_mismatch);
  std::vector<const torch::Tensor*> same = {&blocks, &norm_weight, &qk_weight,
                                            &prefix_output, &output};
  for (const std::optional<torch::Tensor>* o :
       {&prefix, &output_norm_weight, &block_output})
    if (*o) {
      if (!(*o)->is_cuda()) return error_code(Error::not_on_device);
      same.push_back(&**o);
    }
  for (const torch::Tensor* t : same)
    if (t->scalar_type() != input.scalar_type())
      return error_code(Error::dtype_mismatch);
  if ((prefix && prefix->sizes() != input.sizes()) ||
      (block_output && (block_output->sizes() != input.sizes() ||
                        block_output->stride(1) != 1)) ||
      (output_norm_weight && (output_norm_weight->dim() != 1 ||
                              output_norm_weight->numel() != width)))
    return error_code(Error::shape_mismatch);

  if (reinterpret_cast<uintptr_t>(blocks.data_ptr()) % kPackBytes != 0)
    return error_code(Error::not_aligned);

  const torch::Tensor sources = blocks.narrow(1, 0, num_blocks);

  struct Span {
    uintptr_t first, last;
    bool rows;
    uintptr_t stride, width;
  };
  const auto span_of = [](const torch::Tensor& t) {
    const uintptr_t first = reinterpret_cast<uintptr_t>(t.data_ptr());
    const int64_t elem = t.element_size();
    int64_t last = 0;
    for (int64_t d = 0; d < t.dim(); ++d) last += (t.size(d) - 1) * t.stride(d);
    Span span{first, first + static_cast<uintptr_t>((last + 1) * elem), false,
              0, 0};
    if (t.dim() < 2 || t.stride(t.dim() - 1) != 1) return span;
    int64_t run = 1;
    for (int64_t d = t.dim() - 1; d >= 1; --d) {
      if (t.stride(d) != run) return span;
      run *= t.size(d);
    }
    if (t.stride(0) < run) return span;
    span.rows = true;
    span.stride = static_cast<uintptr_t>(t.stride(0) * elem);
    span.width = static_cast<uintptr_t>(run * elem);
    return span;
  };
  for (const torch::Tensor* i : std::initializer_list<const torch::Tensor*>{
           &input, prefix ? &*prefix : nullptr, &sources, &norm_weight,
           &qk_weight, output_norm_weight ? &*output_norm_weight : nullptr})
    for (const torch::Tensor* o : std::initializer_list<const torch::Tensor*>{
             &prefix_output, &output,
             block_output ? &*block_output : nullptr}) {
      if (i == nullptr || o == nullptr || i->numel() == 0 || o->numel() == 0)
        continue;
      const Span a = span_of(*i), b = span_of(*o);
      if (a.last <= b.first || b.last <= a.first) continue;
      if (!(a.rows && b.rows && a.stride == b.stride))
        return error_code(Error::inputs_overlap_outputs);

      const uintptr_t at = a.first % a.stride, bt = b.first % b.stride;
      const auto meets = [&](uintptr_t x, uintptr_t xw, uintptr_t y) {
        return (y + a.stride - x) % a.stride < xw;
      };
      if (meets(at, a.width, bt) || meets(bt, b.width, at))
        return error_code(Error::inputs_overlap_outputs);
    }
  for (const int64_t v : {waves_per_eu, tile_m, tile_n, tile_k, pack_bytes,
                          grid_x, grid_y, grid_z, block_x, block_y, block_z})
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

  // The splits' norm-kernel fields: no split is built.
  for (const std::optional<int64_t>& field :
       {add_attn_res_rms_norm_grid_x, add_attn_res_rms_norm_grid_y,
        add_attn_res_rms_norm_grid_z, add_attn_res_rms_norm_block_x,
        add_attn_res_rms_norm_block_y, add_attn_res_rms_norm_block_z,
        add_attn_res_rms_norm_waves_per_eu, add_attn_res_rms_norm_tile_m,
        add_attn_res_rms_norm_tile_n, add_attn_res_rms_norm_tile_k,
        add_attn_res_rms_norm_pack_bytes}) {
    if (field) return error_code(Error::field_not_this_templates);
  }
  CommsState& s = *reinterpret_cast<CommsState*>(state);
  const void* input_ptr = input.data_ptr();
  const int64_t input_stride_m = input.stride(0);
  const int64_t input_stride_n = input.stride(1);
  const int64_t input_size_m = input.size(0);
  const int64_t input_size_n = input.size(1);
  const void* prefix_ptr = prefix ? prefix->data_ptr() : nullptr;
  const int64_t prefix_stride_m = prefix ? prefix->stride(0) : 0;
  const int64_t prefix_stride_n = prefix ? prefix->stride(1) : 0;
  const int64_t prefix_size_m = prefix ? prefix->size(0) : 0;
  const int64_t prefix_size_n = prefix ? prefix->size(1) : 0;
  const void* blocks_ptr = blocks.data_ptr();
  const int64_t blocks_stride_m = blocks.stride(0);
  const int64_t blocks_stride_r = blocks.stride(1);
  const int64_t blocks_stride_n = blocks.stride(2);
  const int64_t blocks_size_m = blocks.size(0);
  const int64_t blocks_size_r = blocks.size(1);
  const int64_t blocks_size_n = blocks.size(2);
  const void* norm_weight_ptr = norm_weight.data_ptr();
  const int64_t norm_weight_stride_n = norm_weight.stride(0);
  const int64_t norm_weight_size_n = norm_weight.size(0);
  const void* qk_weight_ptr = qk_weight.data_ptr();
  const int64_t qk_weight_stride_n = qk_weight.stride(0);
  const int64_t qk_weight_size_n = qk_weight.size(0);
  const void* output_norm_weight_ptr =
      output_norm_weight ? output_norm_weight->data_ptr() : nullptr;
  const int64_t output_norm_weight_stride_n =
      output_norm_weight ? output_norm_weight->stride(0) : 0;
  const int64_t output_norm_weight_size_n =
      output_norm_weight ? output_norm_weight->size(0) : 0;
  void* prefix_output_ptr = prefix_output.data_ptr();
  const int64_t prefix_output_stride_m = prefix_output.stride(0);
  const int64_t prefix_output_stride_n = prefix_output.stride(1);
  const int64_t prefix_output_size_m = prefix_output.size(0);
  const int64_t prefix_output_size_n = prefix_output.size(1);
  void* output_ptr = output.data_ptr();
  const int64_t output_stride_m = output.stride(0);
  const int64_t output_stride_n = output.stride(1);
  const int64_t output_size_m = output.size(0);
  const int64_t output_size_n = output.size(1);
  void* block_output_ptr = block_output ? block_output->data_ptr() : nullptr;
  const int64_t block_output_stride_m =
      block_output ? block_output->stride(0) : 0;
  const int64_t block_output_stride_n =
      block_output ? block_output->stride(1) : 0;
  const int64_t block_output_size_m = block_output ? block_output->size(0) : 0;
  const int64_t block_output_size_n = block_output ? block_output->size(1) : 0;
  const at::ScalarType dtype = input.scalar_type();
  const dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y),
                  static_cast<unsigned>(grid_z));
  const dim3 block(static_cast<unsigned>(block_x),
                   static_cast<unsigned>(block_y),
                   static_cast<unsigned>(block_z));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  AllReduceAddAttnResRmsNormAlgorithm algorithm;
  if (algorithm_name == "one_shot_pull")
    algorithm = AllReduceAddAttnResRmsNormAlgorithm::one_shot_pull;
  else if (algorithm_name == "two_shot_pull_columns")
    algorithm = AllReduceAddAttnResRmsNormAlgorithm::two_shot_pull_columns;
  else
    return error_code(Error::no_such_kernel);
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_code(Error::no_such_quantization);
  const int quantization =
      quantization_name == "int4" ? Quantization::int4 : Quantization::exact;
  const int64_t m = input_size_m, n = input_size_n;
  const int64_t e = input.element_size();
  const int64_t pack = pack_bytes;

  if (std::find(kDtypes.begin(), kDtypes.end(), dtype) == kDtypes.end())
    return error_code(Error::dtype_not_built);
  if (grid_y != 1 || grid_z != 1 || grid_x < 1 ||
      grid_x > s.symmetric.max_blocks)
    return error_code(Error::launch_out_of_range);
  if (tile_m < 1 || tile_n < 1 || tile_k < 1)
    return error_code(Error::field_not_positive);
  if (pack < 2 || pack > kPackBytes || (pack & (pack - 1)) != 0 || pack < e)
    return error_code(Error::pack_out_of_range);
  if (n * e % pack != 0) return error_code(Error::row_not_packs);
  const bool int4 = quantization == Quantization::int4;
  if (int4 && pack != kPackBytes)
    return error_code(Error::quantization_not_supported);

  if (input_stride_n != 1 || blocks_stride_n != 1 ||
      norm_weight_stride_n != 1 || qk_weight_stride_n != 1 ||
      prefix_output_stride_n != 1 || output_stride_n != 1)
    return error_code(Error::inner_stride_not_one);
  if (input_stride_m * e % pack != 0 || blocks_stride_m * e % pack != 0 ||
      blocks_stride_r * e % pack != 0 ||
      prefix_output_stride_m * e % pack != 0 || output_stride_m * e % pack != 0)
    return error_code(Error::row_stride_not_packs);

  if (prefix_ptr != nullptr && prefix_stride_n != 1)
    return error_code(Error::inner_stride_not_one);
  if (prefix_ptr != nullptr && prefix_stride_m * e % pack != 0)
    return error_code(Error::row_stride_not_packs);
  if (output_norm_weight_ptr != nullptr && output_norm_weight_stride_n != 1)
    return error_code(Error::inner_stride_not_one);
  if (block_output_ptr != nullptr && block_output_stride_n != 1)
    return error_code(Error::inner_stride_not_one);
  if (block_output_ptr != nullptr && block_output_stride_m * e % pack != 0)
    return error_code(Error::row_stride_not_packs);

  // The Simple protocols' columns two-shot always normalizes the row
  // chunked, from the output.
  const bool simple_two_shot =
      (protocol == Protocol::simple_write ||
       protocol == Protocol::simple_read) &&
      algorithm == AllReduceAddAttnResRmsNormAlgorithm::two_shot_pull_columns;
  const int shared_memory_bytes =
      n > tile_n || simple_two_shot
          ? static_cast<int>(tile_m * (num_blocks + 1) * sizeof(float))
          : 0;

  const bool chunked = n > tile_n;

  const bool tail = quantization == Quantization::int4 &&
                    (n % kInt4Group != 0 || n / kInt4Group % s.world != 0);
  void* kernel = nullptr;

  for (const auto& i : kAllReduceAddAttnResRmsNormInstances)
    if (i.algorithm == algorithm && i.protocol == protocol &&
        i.block_x == block_x && i.block_y == block_y && i.block_z == block_z &&
        i.waves_per_eu == waves_per_eu && i.tile_m == tile_m &&
        i.tile_n == tile_n && i.tile_k == tile_k &&
        i.quantization == quantization && i.pack_bytes == pack &&
        i.world == s.world && i.wave_size == s.device.wave_size &&
        i.dtype == dtype && i.weight_dtype == dtype && i.contiguous == false &&
        i.chunked == chunked && i.tail == tail &&
        i.quantize_int4 ==
            (quantization == Quantization::int4 && pack == kPackBytes))
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
  int64_t inbox_stride_slot = 0, inbox_stride_m = 0, inbox_size_m = 0;
  if (protocol == Protocol::simple_write || protocol == Protocol::simple_read) {
    const int64_t elems = pack / e;
    const int64_t span = !simple_two_shot
                             ? (tile_n + elems * block_x - 1) /
                                   (elems * block_x) * elems * block_x
                         : quantization == Quantization::int4
                             ? block_x / s.world * kInt4Group
                             : (tile_n + elems - 1) / elems * elems;
    if (s.world * grid_x * span * e > s.symmetric.inbox_bytes ||
        grid_x > s.connection_blocks || grid_x > s.symmetric.max_blocks)
      return error_code(Error::unfenced_too_small);
    inbox_stride_slot = grid_x * span, inbox_stride_m = span,
    inbox_size_m = grid_x;
  }
  char* const* const inbox_ptr_array =
      protocol == Protocol::simple_read
          ? s.symmetric.simple_read_inbox_ptr_array
          : s.symmetric.simple_write_inbox_ptr_array;
  uint32_t* const* const counters_ptr_array =
      protocol == Protocol::simple_read
          ? s.symmetric.simple_read_counters_ptr_array
          : s.symmetric.simple_write_counters_ptr_array;
  int32_t* const connections = s.connections + static_cast<int>(protocol) *
                                                   s.connection_blocks *
                                                   s.connection_words;
  const auto launch = [&]<typename T, typename W>() {
    const auto typed =
        reinterpret_cast<AllReduceAddAttnResRmsNormKernel<T, W>>(kernel);
    typed<<<grid, block, shared_memory_bytes, stream>>>(
        static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
        static_cast<int>(input_size_m), static_cast<int>(input_size_n),
        static_cast<const T* const*>(nullptr), 0, 0, 0,
        static_cast<const T*>(prefix_ptr), prefix_stride_m, prefix_stride_n,
        static_cast<int>(prefix_size_m), static_cast<int>(prefix_size_n),
        static_cast<const T*>(blocks_ptr), blocks_stride_m, blocks_stride_r,
        blocks_stride_n, static_cast<int>(blocks_size_m),
        static_cast<int>(blocks_size_r), static_cast<int>(blocks_size_n),
        static_cast<const T*>(norm_weight_ptr), norm_weight_stride_n,
        static_cast<int>(norm_weight_size_n),
        static_cast<const T*>(qk_weight_ptr), qk_weight_stride_n,
        static_cast<int>(qk_weight_size_n),
        static_cast<const T*>(output_norm_weight_ptr),
        output_norm_weight_stride_n,
        static_cast<int>(output_norm_weight_size_n),
        static_cast<T*>(prefix_output_ptr), prefix_output_stride_m,
        prefix_output_stride_n, static_cast<int>(prefix_output_size_m),
        static_cast<int>(prefix_output_size_n), static_cast<T*>(output_ptr),
        output_stride_m, output_stride_n, static_cast<int>(output_size_m),
        static_cast<int>(output_size_n), static_cast<T*>(block_output_ptr),
        block_output_stride_m, block_output_stride_n,
        static_cast<int>(block_output_size_m),
        static_cast<int>(block_output_size_n), s.symmetric.partials_ptr_array,
        s.world, 0, 0, 0, 0, s.symmetric.staging_ptr_array, s.world, 0, 0, 0, 0,
        inbox_ptr_array, s.world, inbox_stride_slot, inbox_stride_m, int64_t{1},
        s.world, static_cast<int>(inbox_size_m),
        static_cast<int>(inbox_stride_m), connections, s.connection_words,
        int64_t{1}, static_cast<int>(s.connection_blocks),
        static_cast<int>(s.connection_words), counters_ptr_array, s.world,
        int64_t{s.symmetric.max_blocks * s.world}, int64_t{s.world}, int64_t{1},
        static_cast<int>(s.symmetric.max_blocks), s.world,
        s.symmetric.signals_ptr_array, s.world,
        s.symmetric.max_blocks * s.world, int64_t{s.world}, int64_t{1},
        int64_t{2}, s.symmetric.max_blocks, int64_t{s.world}, s.status_device,
        int64_t{1}, int64_t{kStatusFields}, s.rank, s.timeout_ticks, num_blocks,
        eps, output_eps);
  };
  if (dtype == at::ScalarType::BFloat16)
    launch.template operator()<c10::BFloat16, c10::BFloat16>();
  else
    launch.template operator()<c10::Half, c10::Half>();

  if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
  return std::nullopt;
}
