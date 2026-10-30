// SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc.
// All rights reserved.
//
// ALL_REDUCE_ADD_ATTN_RES_RMS_NORM, a torch op: the all-reduce and AttnRes at a
// config.
#pragma once

#include "../../ops.h"
#include "../types.cuh"
#include "../utils.h"
#include "../kernels/all_reduce.cuh"
#include "../kernels/all_reduce_add_attn_res_rms_norm.cuh"

// all_reduce_add_attn_res_rms_norm: at its config. One function: what Python
// passes, its tensors stripped to pointers and strides, the preconditions, the
// kernel's arguments, the launch, the postcondition.
ErrorCode hip_custom_communicator_all_reduce_add_attn_res_rms_norm(
    int64_t state, torch::Tensor& input,
    const std::optional<torch::Tensor>& prefix, torch::Tensor& blocks,
    torch::Tensor& norm_weight, torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& output_norm_weight,
    torch::Tensor& prefix_output, torch::Tensor& output,
    const std::optional<torch::Tensor>& block_output, int64_t num_blocks,
    double eps, double output_eps, const std::string& algorithm_name,
    int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t block_x,
    int64_t block_y, int64_t block_z, int64_t waves_per_eu, int64_t tile_m,
    int64_t tile_n, int64_t tile_k, const std::string& quantization_name,
    int64_t pack_bytes, std::optional<int64_t> add_attn_res_rms_norm_grid_x,
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
  // THE CALL FROM TORCH: its state, its tensors, its config's ints.
  if (state == 0) return error_code(Error::not_open);
  // ITS TENSORS: on the device, their shapes and dtypes the input's, the
  // blocks' start on a pack, no input sharing an output's memory.
  if (!input.is_cuda() || !norm_weight.is_cuda() || !qk_weight.is_cuda() ||
      !prefix_output.is_cuda() || !output.is_cuda() || !blocks.is_cuda())
    return error_code(Error::not_on_device);
  if (input.dim() != 2) return error_code(Error::shape_mismatch);
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
  // Its rows' and sources' strides are the API's to refuse (whole packs); its
  // start is here.
  if (reinterpret_cast<uintptr_t>(blocks.data_ptr()) % kPackBytes != 0)
    return error_code(Error::not_aligned);
  // The sources read are the first num_blocks; block_output may be a later slot
  // of the same stack.
  const torch::Tensor sources = blocks.narrow(1, 0, num_blocks);
  // NO INPUT OVERLAPS AN OUTPUT: the kernels take each as its own
  // `__restrict__` pointer, so two that share bytes would be undefined
  // behavior. A missing tensor (null) overlaps nothing. EXACT FOR VIEWS OF ONE
  // STACK, conservative otherwise: the sources and block_output interleave,
  // every row the sources then the slot, so their first-to-last spans overlap
  // though no element does (the span check refused Kimi's own append,
  // 2026-10-13). A view whose dimensions after the first are one contiguous
  // run is ROWS: a run of `width` bytes every `stride` bytes. Two rows views
  // with the same stride are disjoint exactly when their runs do not meet
  // within one period; anything else is compared by its span.
  struct Span {
    uintptr_t first, last;    // the first byte and one past the last
    bool rows;                // whether it is rows
    uintptr_t stride, width;  // a row's period and its run, in bytes
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
      // Within one period: a's run at offset `at`, b's at `bt`, on a circle of
      // `stride` bytes; `meets`: y lands inside [x, x + xw) on the circle.
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
  // THE ALGORITHM: the fused kernel's body, or `<body>_split`, the split: our
  // all-reduce kernel at that body (at this config's fields) into this rank's
  // sum, then AttnRes on its rows (at the add_attn_res_rms_norm_ fields), back
  // to back on the stream (the split's path, at the end). Every refusal, either
  // path's, before any launch. The add_attn_res_rms_norm_ fields are a split's
  // alone: given to a fused algorithm, or missing from a split, refused.
  const bool split = algorithm_name.ends_with("_split");
  for (const std::optional<int64_t>& field :
       {add_attn_res_rms_norm_grid_x, add_attn_res_rms_norm_grid_y,
        add_attn_res_rms_norm_grid_z, add_attn_res_rms_norm_block_x,
        add_attn_res_rms_norm_block_y, add_attn_res_rms_norm_block_z,
        add_attn_res_rms_norm_waves_per_eu, add_attn_res_rms_norm_tile_m,
        add_attn_res_rms_norm_tile_n, add_attn_res_rms_norm_tile_k,
        add_attn_res_rms_norm_pack_bytes}) {
    if (!split && field) return error_code(Error::field_not_this_templates);
    if (split && !field) return error_code(Error::field_missing);
    if (field && (*field < std::numeric_limits<int>::min() ||
                  *field > std::numeric_limits<int>::max()))
      return error_code(Error::field_out_of_range);
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
  if (!split) {
    // ---- THE FUSED KERNEL ----
    // ---- PRECONDITIONS: the config, the call, then its instance. The first
    // Error refuses the call; past this section the call runs. ----
    AllReduceAddAttnResRmsNormAlgorithm algorithm;
    if (algorithm_name == "one_shot_pull")
      algorithm = AllReduceAddAttnResRmsNormAlgorithm::one_shot_pull;
    else if (algorithm_name == "two_shot_pull_rows")
      algorithm = AllReduceAddAttnResRmsNormAlgorithm::two_shot_pull_rows;
    else if (algorithm_name == "two_shot_pull_columns")
      algorithm = AllReduceAddAttnResRmsNormAlgorithm::two_shot_pull_columns;
    else if (algorithm_name == "two_shot_push")
      algorithm = AllReduceAddAttnResRmsNormAlgorithm::two_shot_push;
    else
      return error_code(Error::no_such_kernel);
    if (quantization_name != "exact" && quantization_name != "int4")
      return error_code(Error::no_such_quantization);
    const int quantization =
        quantization_name == "int4" ? Quantization::int4 : Quantization::exact;
    const int64_t m = input_size_m,
                  n = input_size_n;  // the problem's shape: the input's
    const int64_t e = input.element_size();
    const int64_t pack = pack_bytes;
    // THE DTYPE BUILT (the world is: open refuses the rest); THE CONFIG'S
    // RANGES, each the hardware's
    // (STANDARDS 3.2): the grid x alone (the kernel's one dimension), at least
    // a block, at most the blocks the signals have slots for; the tile
    // positive; a pack a load the hardware has (2, 4, 8 or 16 bytes), at least
    // an element, whole packs a row (an aligned load of the pack's width); INT4
    // at the widest pack (the format's word is a pack of 8 values: common.cuh's
    // QUANTIZATION).
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
    // WHAT THE REGIONS MUST HOLD. An INT4 payload lies in the region that holds
    // it exact (common.cuh's INT4 payload layout): the input's in the staging,
    // the sums' in the partials. A collective runs the whole call in one pass:
    // the regions are sized from the largest call the server makes, so a call
    // past them is refused here, never cut into bands; its input is in the
    // staging whole (an eager exact call's host copy, INPUT TABLES; a captured
    // one's is read where it is, refused alike, so the eager and captured calls
    // of one shape agree).
    // A TWO-SHOT KEEPS THE WHOLE SUM in the partials, [m, n] (each rank's rows
    // or columns at their place); a one-shot keeps none.
    const bool two_shot =
        algorithm != AllReduceAddAttnResRmsNormAlgorithm::one_shot_pull;
    const int64_t partials = two_shot ? m * n * e : 0;
    if (int4 && int4_payload_bytes(m, n) > s.symmetric.staging_bytes)
      return error_code(Error::staging_too_small);
    if (int4 && two_shot &&
        int4_payload_bytes(1, m * n) > s.symmetric.partials_bytes)
      return error_code(Error::partials_too_small);
    if (partials > s.symmetric.partials_bytes)
      return error_code(Error::partials_too_small);
    if (m * n * e > s.symmetric.staging_bytes)
      return error_code(Error::staging_too_small);
    // A LAYOUT THE PACKED LOADS CAN READ: every buffer's innermost stride 1,
    // its rows on whole packs of the build's.
    if (input_stride_n != 1 || blocks_stride_n != 1 ||
        norm_weight_stride_n != 1 || qk_weight_stride_n != 1 ||
        prefix_output_stride_n != 1 || output_stride_n != 1)
      return error_code(Error::inner_stride_not_one);
    if (input_stride_m * e % pack != 0 || blocks_stride_m * e % pack != 0 ||
        blocks_stride_r * e % pack != 0 ||
        prefix_output_stride_m * e % pack != 0 ||
        output_stride_m * e % pack != 0)
      return error_code(Error::row_stride_not_packs);
    // THE OPTIONAL BUFFERS, checked only when given: a missing one is a null
    // pointer with zero strides, which is no layout at all (a prefix-less call
    // was refused as `inner_stride_not_one`, 2026-10-13).
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
    // A ROW WIDER THAN THE TILE keeps its rows' logits in the launch's LDS
    // (attn_res.cuh's attn_res_wide): [tile_m][num_blocks + 1] floats.
    const int shared_memory_bytes =
        n > tile_n ? static_cast<int>(tile_m * (num_blocks + 1) * sizeof(float))
                   : 0;
    // A ROW WIDER THAN THE TILE runs the build's CHUNKED instance.
    const bool chunked = n > tile_n;
    // AN INT4 ROW WITH A TAIL runs the build's TAIL instance: a row not whole
    // groups, or not whole groups for every rank.
    const bool tail = quantization == Quantization::int4 &&
                      (n % kInt4Group != 0 || n / kInt4Group % s.world != 0);
    void* kernel = nullptr;
    // THE INSTANCE: the kernel compiled at this build, on this device, for this
    // call (builds.py's kAllReduceAddAttnResRmsNormInstances, a row each).
    for (const auto& i : kAllReduceAddAttnResRmsNormInstances)
      if (i.algorithm == algorithm && i.block_x == block_x &&
          i.block_y == block_y && i.block_z == block_z &&
          i.waves_per_eu == waves_per_eu && i.tile_m == tile_m &&
          i.tile_n == tile_n && i.tile_k == tile_k &&
          i.quantization == quantization && i.pack_bytes == pack &&
          i.world == s.world && i.wave_size == s.device.wave_size &&
          i.dtype == dtype && i.weight_dtype == dtype &&
          i.contiguous == false && i.chunked == chunked && i.tail == tail &&
          i.quantize_int4 ==
              (quantization == Quantization::int4 && pack == kPackBytes))
        kernel = i.kernel;
    if (kernel == nullptr) return error_code(Error::not_built);
    // THE KERNEL ON THIS DEVICE, as its compiler left it. HIP refuses the block
    // and the LDS at the launch; the scratch only when the dispatch runs,
    // aborting the queue and the process with it (the tune's crashes,
    // 2026-10-24: AttnRes at 64 threads, tile 4x16384x4, 21 KiB a lane).
    hipFuncAttributes attributes;
    if (hipFuncGetAttributes(&attributes, kernel) != hipSuccess)
      return error_code(Error::hip_failed);
    if (block_x > attributes.maxThreadsPerBlock)
      return error_code(Error::block_too_large);
    if (static_cast<int64_t>(attributes.sharedSizeBytes) + shared_memory_bytes >
        s.device.lds_bytes)
      return error_code(Error::lds_too_large);
    if (static_cast<int64_t>(attributes.localSizeBytes) >
        s.device.scratch_a_lane)
      return error_code(Error::scratch_too_large);
    // THE GRID RESIDENT AT ONCE: a peer barrier spins until the same block on
    // every peer arrives, so a grid past the blocks the device holds at once
    // can hang (HIP never refuses it).
    int blocks_a_cu = 0;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_a_cu, kernel, static_cast<int>(block_x),
            shared_memory_bytes) != hipSuccess)
      return error_code(Error::hip_failed);
    if (grid_x > blocks_a_cu * s.device.compute_units)
      return error_code(Error::grid_not_resident);

    // ---- THE LAUNCH: the buffers, the views of their regions, and the
    // instance.
    // ----
    // ITS VIEWS OF THE REGIONS, in elements, [m, n] at stride n: the staging
    // the input (an eager call's copy; INT4 its payload), a two-shot's partials
    // the whole sum (a one-shot reads none). Inside their regions by the checks
    // above.
    const int64_t partials_stride_m = n, partials_stride_n = 1,
                  partials_size_m = two_shot && n > 0 ? m : 0,
                  partials_size_n = n;
    const int64_t staging_stride_m = n, staging_stride_n = 1,
                  staging_size_m = m, staging_size_n = n;
    // IN PLACE, every exact algorithm: the pushed two-shot's reduce-scatter
    // reads its peers' inputs through the input table too (reduce_scatter.cuh's
    // columns_push), as the norm's does.
    const bool in_place = quantization == Quantization::exact;
    // THE INPUT TABLE (INPUT TABLES): where the kernel reads every rank's input
    // in place (an exact pull), or none where it reads the staging. Captured:
    // the next table slot, reserved for this input (register_graph_buffers
    // fills it with every rank's address when the capture exits), its rows at
    // the input's stride. Eager: the input copied into this rank's staging (it
    // fits: checked above), read through the eager table, slot 0, its rows n
    // apart.
    void* const* table = nullptr;
    int table_size = 0;
    int64_t table_stride_m = staging_stride_m,
            table_stride_n = staging_stride_n;
    if (in_place) {
      hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
      if (hipStreamIsCapturing(stream, &capture) != hipSuccess)
        return error_code(Error::hip_failed);
      InputTables& tables = s.input_tables;
      if (capture == hipStreamCaptureStatusActive) {
        if (tables.next_slot >= tables.slots)
          return error_code(Error::tables_full);
        tables.pending.emplace_back(tables.next_slot, input_ptr);
        table = tables.device + tables.next_slot * s.world;
        table_stride_m = input_stride_m;
        ++tables.next_slot;
      } else {
        if (hipMemcpy2DAsync(s.symmetric.staging, static_cast<size_t>(n * e),
                             input_ptr, static_cast<size_t>(input_stride_m * e),
                             static_cast<size_t>(n * e), static_cast<size_t>(m),
                             hipMemcpyDeviceToDevice, stream) != hipSuccess)
          return error_code(Error::hip_failed);
        table = tables.device;
        table_stride_m = n;
      }
      table_size = s.world, table_stride_n = 1;
    }
    // THE LAUNCH: the instance's kernel, at the call's dtypes.
    const auto launch = [&]<typename T, typename W>() {
      const auto typed =
          reinterpret_cast<AllReduceAddAttnResRmsNormKernel<T, W>>(kernel);
      typed<<<grid, block, shared_memory_bytes, stream>>>(
          static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
          static_cast<int>(input_size_m), static_cast<int>(input_size_n),
          reinterpret_cast<const T* const*>(table), table_size, table_stride_m,
          table_stride_n, static_cast<const T*>(prefix_ptr), prefix_stride_m,
          prefix_stride_n, static_cast<int>(prefix_size_m),
          static_cast<int>(prefix_size_n), static_cast<const T*>(blocks_ptr),
          blocks_stride_m, blocks_stride_r, blocks_stride_n,
          static_cast<int>(blocks_size_m), static_cast<int>(blocks_size_r),
          static_cast<int>(blocks_size_n),
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
          s.world, partials_stride_m, partials_stride_n,
          static_cast<int>(partials_size_m), static_cast<int>(partials_size_n),
          s.symmetric.staging_ptr_array, s.world, staging_stride_m,
          staging_stride_n, static_cast<int>(staging_size_m),
          static_cast<int>(staging_size_n), s.symmetric.signals_ptr_array,
          s.world, s.symmetric.max_blocks * s.world, int64_t{s.world},
          int64_t{1}, int64_t{2}, s.symmetric.max_blocks, int64_t{s.world},
          s.status_device, int64_t{1}, int64_t{kStatusFields}, s.rank,
          s.timeout_ticks, num_blocks, eps, output_eps);
    };
    if (dtype == at::ScalarType::BFloat16)
      launch.template operator()<c10::BFloat16, c10::BFloat16>();
    else
      launch.template operator()<c10::Half, c10::Half>();
    // POSTCONDITION: the launch went in.
    if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
    return std::nullopt;
  }

  // ---- THE SPLIT: our all-reduce at this config's fields into this rank's
  // sum, then AttnRes on its rows at the add_attn_res_rms_norm_ fields. Both
  // kernels' preconditions first, then the launches. ----
  const std::string body = algorithm_name.substr(0, algorithm_name.size() - 6);
  AllReduceAlgorithm all_reduce_algorithm;
  if (body == "one_shot_pull")
    all_reduce_algorithm = AllReduceAlgorithm::one_shot_pull;
  else if (body == "one_shot_push")
    all_reduce_algorithm = AllReduceAlgorithm::one_shot_push;
  else if (body == "one_shot_push_staged")
    all_reduce_algorithm = AllReduceAlgorithm::one_shot_push_staged;
  else if (body == "two_shot_pull")
    all_reduce_algorithm = AllReduceAlgorithm::two_shot_pull;
  else if (body == "two_shot_push")
    all_reduce_algorithm = AllReduceAlgorithm::two_shot_push;
  else
    return error_code(Error::no_such_kernel);
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_code(Error::no_such_quantization);
  const bool int4_config = quantization_name == "int4";
  const bool quantize_int4 = int4_config;
  const int64_t m = input_size_m, n = input_size_n;
  const int64_t e = input.element_size();
  // THE SPLIT'S SUM, this rank's [m, n]: the all-reduce's output, AttnRes's
  // input.
  torch::Tensor summed = torch::empty_like(input);
  void* summed_ptr = summed.data_ptr();
  const int64_t summed_stride_m = summed.stride(0),
                summed_stride_n = summed.stride(1);
  const int64_t summed_size_m = summed.size(0), summed_size_n = summed.size(1);
  // THE ALL-REDUCE'S CONFIG: the dtype built (the world is: open refuses the
  // rest); the grid x alone,
  // at least a block, at most the blocks the signals have slots for; the tile
  // positive; a pack a load the hardware has, at least an element, whole packs
  // a row; INT4 at the widest pack; the input's rows on whole packs.
  if (std::find(kDtypes.begin(), kDtypes.end(), dtype) == kDtypes.end())
    return error_code(Error::dtype_not_built);
  if (grid_y != 1 || grid_z != 1 || grid_x < 1 ||
      grid_x > s.symmetric.max_blocks)
    return error_code(Error::launch_out_of_range);
  if (tile_m < 1 || tile_n < 1) return error_code(Error::field_not_positive);
  if (pack_bytes < 2 || pack_bytes > kPackBytes ||
      (pack_bytes & (pack_bytes - 1)) != 0 || pack_bytes < e)
    return error_code(Error::pack_out_of_range);
  if (int4_config && pack_bytes != kPackBytes)
    return error_code(Error::quantization_not_supported);
  if (n * e % pack_bytes != 0) return error_code(Error::row_not_packs);
  if (input_stride_n != 1) return error_code(Error::inner_stride_not_one);
  if (input_stride_m * e % pack_bytes != 0)
    return error_code(Error::row_stride_not_packs);
  void* kernel = nullptr;
  bool in_place = false;
  int64_t staging_need = 0, partials_need = 0;
  int64_t partials_stride_m = 0, partials_stride_n = 0, partials_size_m = 0,
          partials_size_n = 0;
  int64_t staging_stride_m = 0, staging_stride_n = 0, staging_size_m = 0,
          staging_size_n = 0;
  int64_t unfenced_stride_slot = 0, unfenced_stride_m = 0,
          unfenced_stride_n = 0, unfenced_size_slot = 0, unfenced_size_m = 0,
          unfenced_size_n = 0;
  // IN PLACE: the exact pull, every rank's input read where its table says.
  in_place = (all_reduce_algorithm == AllReduceAlgorithm::one_shot_pull ||
              all_reduce_algorithm == AllReduceAlgorithm::two_shot_pull) &&
             !quantize_int4;
  // THE PLAIN TWO-SHOTS WALK THE CALL FLAT: CONTIGUOUS, their one instance
  // that varies by layout (every other algorithm's is the strided one); an
  // INT4 call not whole groups runs the build's TAIL instance, as does a
  // one-shot's whose row is not.
  const bool two_shot =
      all_reduce_algorithm == AllReduceAlgorithm::two_shot_pull ||
      all_reduce_algorithm == AllReduceAlgorithm::two_shot_push;
  const bool contiguous = two_shot && input_stride_m == n * input_stride_n &&
                          summed_stride_m == n * summed_stride_n;
  const bool tail = quantize_int4 &&
                    (two_shot ? m * n % kInt4Group != 0 : n % kInt4Group != 0);
  // THE PUSHED ONE-SHOT'S UNFENCED REGION HOLDS ITS TILES: a row of a block's
  // tile in a slot's lines, a line a pack of at most kLinePayloadBytes (a
  // narrower pack its line alone: kernels/all_reduce.cuh), INT4 a line a word
  // of its tile's whole groups; a block's TILE_M rows of a bank's
  // unfenced_blocks, a block its own slots (a grid past them is refused, never
  // cut).
  if (all_reduce_algorithm == AllReduceAlgorithm::one_shot_push) {
    const int64_t line_pack = std::min<int64_t>(pack_bytes, kLinePayloadBytes);
    const int64_t row_lines =
        quantize_int4 ? (tile_n + kInt4Group - 1) / kInt4Group * kInt4GroupPacks
                      : (tile_n * e + line_pack - 1) / line_pack;
    if (row_lines > s.symmetric.unfenced_slot_bytes / kLinePayloadBytes ||
        grid_x * tile_m > s.symmetric.unfenced_blocks)
      return error_code(Error::unfenced_too_small);
  }
  // THE INSTANCE: the kernel compiled at this config, on this device, for this
  // call (builds.py's kAllReduceInstances, a row each); a config the table did
  // not build is refused here.
  for (const auto& i : kAllReduceInstances)
    if (i.algorithm == all_reduce_algorithm && i.block_x == block_x &&
        i.block_y == block_y && i.block_z == block_z &&
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
  // THE KERNEL ON THIS DEVICE, as its compiler left it. HIP refuses the block
  // and the LDS at the launch; the scratch only when the dispatch runs,
  // aborting the queue and the process with it (the tune's crashes, 2026-10-24:
  // AttnRes at 64 threads, tile 4x16384x4, 21 KiB a lane).
  hipFuncAttributes attributes;
  if (hipFuncGetAttributes(&attributes, kernel) != hipSuccess)
    return error_code(Error::hip_failed);
  if (block_x > attributes.maxThreadsPerBlock)
    return error_code(Error::block_too_large);
  if (static_cast<int64_t>(attributes.sharedSizeBytes) + 0 > s.device.lds_bytes)
    return error_code(Error::lds_too_large);
  if (static_cast<int64_t>(attributes.localSizeBytes) > s.device.scratch_a_lane)
    return error_code(Error::scratch_too_large);
  // THE GRID RESIDENT AT ONCE: a peer barrier spins until the same block on
  // every peer arrives, so a grid past the blocks the device holds at once can
  // hang (HIP never refuses it). The blocks a CU holds from the kernel's
  // registers, LDS and the CU's wave slots, times the CUs.
  int blocks_a_cu = 0;
  if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_a_cu, kernel, static_cast<int>(block_x), 0) != hipSuccess)
    return error_code(Error::hip_failed);
  if (grid_x > blocks_a_cu * s.device.compute_units)
    return error_code(Error::grid_not_resident);
  // ---- ITS VIEWS OF THE REGIONS, by its all_reduce_algorithm, in elements: a
  // view [m, n] at strides [stride_m, stride_n] (the line push's inbox [slot,
  // m, n]); an empty view (size 0) where it reads none. The kernels find a
  // one-shot's payload in the staging from the input's sizes.
  // THE WHOLE CALL IN ONE PASS: the regions are sized from the largest call the
  // server makes, so a call past them is refused, never cut into bands. Every
  // exact kernel's input fits the staging whole (an eager call's host copy,
  // INPUT TABLES; a captured call's is read where it is, refused alike, so the
  // eager and captured calls of one shape agree). ----
  // The two-shots' exact share is the flat walk's, whole packs, rounded up: a
  // rank's slice of a row in the partials, and the pushed two-shot's inbox
  // [WORLD, slot] in the staging, its slot a share of the whole call.
  const int64_t packs = n * e / pack_bytes;
  const int64_t slice_bytes = (packs + s.world - 1) / s.world * pack_bytes;
  const int64_t slot = (m * packs + s.world - 1) / s.world * pack_bytes / e;
  // THE INT4 FLAT WALK'S SHARE: a rank's whole groups of the call's, rounded
  // up, the last rank's to the end.
  const int64_t share = ((m * n + kInt4Group - 1) / kInt4Group + s.world - 1) /
                        s.world * kInt4Group;
  staging_need = quantize_int4 ? 0 : m * n * e;
  partials_stride_m = slice_bytes / e, partials_stride_n = 1,
  partials_size_n = slice_bytes / e;
  switch (all_reduce_algorithm) {
    case AllReduceAlgorithm::two_shot_pull:
    case AllReduceAlgorithm::two_shot_push:
      staging_need = m * n * e, partials_need = m * slice_bytes;
      if (quantize_int4) {
        // THE INT4 FLAT WALK: this rank's share of the call's groups in the
        // partials, and every rank's share in the staging.
        partials_stride_m = share, partials_size_m = 1, partials_size_n = share;
        staging_stride_m = s.world * share, staging_stride_n = 1,
        staging_size_m = 1, staging_size_n = s.world * share;
        staging_need =
            std::max(staging_need, int4_payload_bytes(1, s.world * share));
        partials_need = std::max(partials_need, int4_payload_bytes(1, share));
      } else {
        // EXACT: this rank's slice of every row in the partials; the pushed
        // two-shot's inbox in the staging, WORLD slots of a share of the call
        // (the pull reads every rank's input in place, through the input
        // table).
        partials_size_m = m;
        if (all_reduce_algorithm == AllReduceAlgorithm::two_shot_push) {
          staging_stride_m = slot, staging_stride_n = 1,
          staging_size_m = s.world, staging_size_n = slot;
          staging_need = std::max(staging_need, s.world * slot * e);
        }
      }
      break;
    case AllReduceAlgorithm::one_shot_pull:
      // INT4: its [m, n] payload in the staging, no partials (exact: every
      // rank's input in place).
      if (quantize_int4) {
        partials_stride_m = 0, partials_size_n = 0;
        staging_stride_n = 1, staging_size_m = 1,
        staging_size_n = (int4_payload_bytes(m, n) + e - 1) / e;
        staging_need = int4_payload_bytes(m, n);
      }
      break;
    case AllReduceAlgorithm::one_shot_push:
      // THE UNFENCED REGION: [writer, row, n] payload elements, 2 *
      // unfenced_blocks rows (bank, row) of unfenced_slot_bytes of payload each
      // whatever the build (so a row is the same bytes at every build, its
      // epoch its own), a writer's rows contiguous.
      unfenced_stride_slot = 2 * s.symmetric.unfenced_blocks *
                             (s.symmetric.unfenced_slot_bytes / e);
      unfenced_stride_m = s.symmetric.unfenced_slot_bytes / e,
      unfenced_stride_n = 1;
      unfenced_size_slot = s.world,
      unfenced_size_m = 2 * s.symmetric.unfenced_blocks,
      unfenced_size_n = s.symmetric.unfenced_slot_bytes / e;
      break;
    case AllReduceAlgorithm::one_shot_push_staged: {
      // WORLD SLOTS OF AN INPUT in the staging (INT4: its payload, a slot whole
      // kInt4ScalesAlign), no partials.
      const int64_t payload =
          quantize_int4 ? int4_payload_bytes(m, n) : m * n * e;
      const int64_t slot_bytes = quantize_int4
                                     ? (payload + kInt4ScalesAlign - 1) /
                                           kInt4ScalesAlign * kInt4ScalesAlign
                                     : payload;
      partials_stride_m = 0, partials_size_n = 0;
      staging_stride_n = 1, staging_size_m = 1,
      staging_size_n = (s.world * slot_bytes + e - 1) / e;
      staging_need = s.world * slot_bytes;
      break;
    }
  }
  // THE REGIONS HOLD THE CALL: what the kernel stages and keeps, and each
  // view's end, its last element's (an empty view reads nothing). The kernels
  // check their indices only against the sizes handed them (October 26: n11
  // rebooted seven times on views past their regions). The unfenced views are
  // that region's whole layout (open's): they fit by construction.
  const int64_t staging_end =
      staging_size_m == 0 || staging_size_n == 0
          ? 0
          : ((staging_size_m - 1) * staging_stride_m +
             (staging_size_n - 1) * staging_stride_n + 1) *
                e;
  const int64_t partials_end =
      partials_size_m == 0 || partials_size_n == 0
          ? 0
          : ((partials_size_m - 1) * partials_stride_m +
             (partials_size_n - 1) * partials_stride_n + 1) *
                e;
  if (std::max(staging_need, staging_end) > s.symmetric.staging_bytes)
    return error_code(Error::staging_too_small);
  if (std::max(partials_need, partials_end) > s.symmetric.partials_bytes)
    return error_code(Error::partials_too_small);
  // ATTNRES ON THIS RANK'S ROWS, at the add_attn_res_rms_norm_ fields: its
  // config (the grid x alone, at least a block; the tile positive; a pack a
  // load the hardware has, at least an element, whole packs a row; it moves
  // nothing), every buffer's layout on its packs, its compiled kernel on this
  // device (it waits on no peer: no residency).
  const dim3 second_grid(static_cast<unsigned>(*add_attn_res_rms_norm_grid_x),
                         static_cast<unsigned>(*add_attn_res_rms_norm_grid_y),
                         static_cast<unsigned>(*add_attn_res_rms_norm_grid_z));
  const dim3 second_block(
      static_cast<unsigned>(*add_attn_res_rms_norm_block_x),
      static_cast<unsigned>(*add_attn_res_rms_norm_block_y),
      static_cast<unsigned>(*add_attn_res_rms_norm_block_z));
  const int64_t second_pack = *add_attn_res_rms_norm_pack_bytes;
  if (*add_attn_res_rms_norm_grid_y != 1 ||
      *add_attn_res_rms_norm_grid_z != 1 || *add_attn_res_rms_norm_grid_x < 1)
    return error_code(Error::launch_out_of_range);
  if (*add_attn_res_rms_norm_tile_m < 1 || *add_attn_res_rms_norm_tile_n < 1 ||
      *add_attn_res_rms_norm_tile_k < 1)
    return error_code(Error::field_not_positive);
  if (second_pack < 2 || second_pack > kPackBytes ||
      (second_pack & (second_pack - 1)) != 0 || second_pack < e)
    return error_code(Error::pack_out_of_range);
  if (n * e % second_pack != 0) return error_code(Error::row_not_packs);
  if (blocks_stride_n != 1 || norm_weight_stride_n != 1 ||
      qk_weight_stride_n != 1 || prefix_output_stride_n != 1 ||
      output_stride_n != 1 || (prefix_ptr != nullptr && prefix_stride_n != 1) ||
      (output_norm_weight_ptr != nullptr && output_norm_weight_stride_n != 1) ||
      (block_output_ptr != nullptr && block_output_stride_n != 1))
    return error_code(Error::inner_stride_not_one);
  if (summed_stride_m * e % second_pack != 0 ||
      blocks_stride_m * e % second_pack != 0 ||
      blocks_stride_r * e % second_pack != 0 ||
      prefix_output_stride_m * e % second_pack != 0 ||
      output_stride_m * e % second_pack != 0 ||
      (prefix_ptr != nullptr && prefix_stride_m * e % second_pack != 0) ||
      (block_output_ptr != nullptr &&
       block_output_stride_m * e % second_pack != 0))
    return error_code(Error::row_stride_not_packs);
  // A ROW WIDER THAN THE TILE keeps its rows' logits in the launch's LDS
  // (attn_res.cuh's attn_res_wide): [tile_m][num_blocks + 1] floats, and runs
  // the build's CHUNKED instance.
  const bool second_chunked = n > *add_attn_res_rms_norm_tile_n;
  const int second_shared_bytes =
      second_chunked ? static_cast<int>(*add_attn_res_rms_norm_tile_m *
                                        (num_blocks + 1) * sizeof(float))
                     : 0;
  void* second_kernel = nullptr;
  for (const auto& i : kAddAttnResRmsNormInstances)
    if (i.algorithm == AddAttnResRmsNormAlgorithm::rows &&
        i.block_x == *add_attn_res_rms_norm_block_x &&
        i.block_y == *add_attn_res_rms_norm_block_y &&
        i.block_z == *add_attn_res_rms_norm_block_z &&
        i.waves_per_eu == *add_attn_res_rms_norm_waves_per_eu &&
        i.tile_m == *add_attn_res_rms_norm_tile_m &&
        i.tile_n == *add_attn_res_rms_norm_tile_n &&
        i.tile_k == *add_attn_res_rms_norm_tile_k &&
        i.quantization == Quantization::exact && i.pack_bytes == second_pack &&
        i.world == 0 && i.wave_size == s.device.wave_size && i.dtype == dtype &&
        i.weight_dtype == dtype && i.contiguous == false &&
        i.chunked == second_chunked && i.tail == false &&
        i.quantize_int4 == false)
      second_kernel = i.kernel;
  if (second_kernel == nullptr) return error_code(Error::not_built);
  // THE KERNEL ON THIS DEVICE, as its compiler left it (it waits on no peer:
  // no residency).
  hipFuncAttributes second_attributes;
  if (hipFuncGetAttributes(&second_attributes, second_kernel) != hipSuccess)
    return error_code(Error::hip_failed);
  if (*add_attn_res_rms_norm_block_x > second_attributes.maxThreadsPerBlock)
    return error_code(Error::block_too_large);
  if (static_cast<int64_t>(second_attributes.sharedSizeBytes) +
          second_shared_bytes >
      s.device.lds_bytes)
    return error_code(Error::lds_too_large);
  if (static_cast<int64_t>(second_attributes.localSizeBytes) >
      s.device.scratch_a_lane)
    return error_code(Error::scratch_too_large);
  // EVERY RANK'S INPUT, where the all-reduce reads it: in place through an
  // input table (the exact pulls), else in every rank's staging (INPUT TABLES).
  void* const* peer_inputs = nullptr;
  int peer_inputs_size = 0;
  int64_t peer_inputs_stride_m = staging_stride_m,
          peer_inputs_stride_n = staging_stride_n;
  if (in_place) {
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(stream, &capture) != hipSuccess)
      return error_code(Error::hip_failed);
    InputTables& tables = s.input_tables;
    if (capture == hipStreamCaptureStatusActive) {
      if (tables.next_slot >= tables.slots)
        return error_code(Error::tables_full);
      tables.pending.emplace_back(tables.next_slot, input_ptr);
      peer_inputs = tables.device + tables.next_slot * s.world;
      peer_inputs_stride_m = input_stride_m;
      ++tables.next_slot;
    } else {
      if (hipMemcpy2DAsync(s.symmetric.staging, static_cast<size_t>(n * e),
                           input_ptr, static_cast<size_t>(input_stride_m * e),
                           static_cast<size_t>(n * e), static_cast<size_t>(m),
                           hipMemcpyDeviceToDevice, stream) != hipSuccess)
        return error_code(Error::hip_failed);
      peer_inputs = tables.device;
      peer_inputs_stride_m = n;
    }
    peer_inputs_size = s.world, peer_inputs_stride_n = 1;
  }
  // ---- THE LAUNCHES: the all-reduce into the sum, then AttnRes on its rows.
  // ----
  // THE LAUNCH: the instance's kernel, at the call's dtypes.
  const auto launch_all_reduce = [&]<typename T, typename W>() {
    const auto typed = reinterpret_cast<AllReduceKernel<T, W>>(kernel);
    typed<<<grid, block, 0, stream>>>(
        static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
        static_cast<int>(input_size_m), static_cast<int>(input_size_n),
        reinterpret_cast<const T* const*>(peer_inputs), peer_inputs_size,
        peer_inputs_stride_m, peer_inputs_stride_n, static_cast<T*>(summed_ptr),
        summed_stride_m, summed_stride_n, static_cast<int>(summed_size_m),
        static_cast<int>(summed_size_n), s.symmetric.partials_ptr_array,
        s.world, partials_stride_m, partials_stride_n,
        static_cast<int>(partials_size_m), static_cast<int>(partials_size_n),
        s.symmetric.staging_ptr_array, s.world, staging_stride_m,
        staging_stride_n, static_cast<int>(staging_size_m),
        static_cast<int>(staging_size_n), s.symmetric.unfenced_ptr_array,
        s.world, unfenced_stride_slot, unfenced_stride_m, unfenced_stride_n,
        static_cast<int>(unfenced_size_slot), static_cast<int>(unfenced_size_m),
        static_cast<int>(unfenced_size_n), s.unfenced_epochs,
        s.unfenced_epoch_words, int64_t{1},
        static_cast<int>(s.symmetric.unfenced_blocks),
        static_cast<int>(s.unfenced_epoch_words), s.symmetric.signals_ptr_array,
        s.world, s.symmetric.max_blocks * s.world, int64_t{s.world}, int64_t{1},
        int64_t{2}, s.symmetric.max_blocks, int64_t{s.world}, s.status_device,
        int64_t{1}, int64_t{kStatusFields}, s.rank, s.timeout_ticks);
  };
  if (dtype == at::ScalarType::BFloat16)
    launch_all_reduce.template operator()<c10::BFloat16, c10::BFloat16>();
  else
    launch_all_reduce.template operator()<c10::Half, c10::Half>();
  const auto launch_attn_res = [&]<typename T, typename W>() {
    const auto typed =
        reinterpret_cast<AddAttnResRmsNormKernel<T, W>>(second_kernel);
    typed<<<second_grid, second_block, second_shared_bytes, stream>>>(
        static_cast<const T*>(summed_ptr), summed_stride_m, summed_stride_n,
        static_cast<int>(summed_size_m), static_cast<int>(summed_size_n),
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
        static_cast<int>(block_output_size_n), num_blocks, eps, output_eps);
  };

  if (dtype == at::ScalarType::BFloat16)
    launch_attn_res.template operator()<c10::BFloat16, c10::BFloat16>();
  else
    launch_attn_res.template operator()<c10::Half, c10::Half>();
  // POSTCONDITION: the launches went in.
  if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
  return std::nullopt;
}
