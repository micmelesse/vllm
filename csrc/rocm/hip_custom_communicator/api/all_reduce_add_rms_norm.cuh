// SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc.
// All rights reserved.
//
// ALL_REDUCE_ADD_RMS_NORM, a torch op: the all-reduce and the (add-)RMSNorm at
// a config.
#pragma once

#include "../../ops.h"
#include "../types.cuh"
#include "../utils.h"
#include "../kernels/all_reduce.cuh"
#include "../kernels/all_reduce_add_rms_norm.cuh"

// all_reduce_add_rms_norm: at its config. One function: what Python passes, its
// tensors stripped to pointers and strides, the preconditions, the kernel's
// arguments, the launch, the postcondition.
ErrorCode hip_custom_communicator_all_reduce_add_rms_norm(
    int64_t state, torch::Tensor& input,
    const std::optional<torch::Tensor>& residual, torch::Tensor& weight,
    torch::Tensor& output, const std::optional<torch::Tensor>& residual_output,
    double eps, const std::string& algorithm_name, int64_t grid_x,
    int64_t grid_y, int64_t grid_z, int64_t block_x, int64_t block_y,
    int64_t block_z, int64_t waves_per_eu, int64_t tile_m, int64_t tile_n,
    const std::string& quantization_name, int64_t pack_bytes,
    std::optional<int64_t> add_rms_norm_grid_x,
    std::optional<int64_t> add_rms_norm_grid_y,
    std::optional<int64_t> add_rms_norm_grid_z,
    std::optional<int64_t> add_rms_norm_block_x,
    std::optional<int64_t> add_rms_norm_block_y,
    std::optional<int64_t> add_rms_norm_block_z,
    std::optional<int64_t> add_rms_norm_waves_per_eu,
    std::optional<int64_t> add_rms_norm_tile_m,
    std::optional<int64_t> add_rms_norm_tile_n,
    std::optional<int64_t> add_rms_norm_pack_bytes) {
  using namespace hip_custom_communicator;
  using namespace hip_custom_communicator::internal;
  // ---- THE CALL FROM TORCH: its state, its tensors stripped to pointers and
  // strides, its config's ints. ----
  if (state == 0) return error_code(Error::not_open);
  CommsState& s = *reinterpret_cast<CommsState*>(state);
  if (!input.is_cuda() || !weight.is_cuda() || !output.is_cuda())
    return error_code(Error::not_on_device);
  if (input.dim() != 2 || output.sizes() != input.sizes() ||
      weight.dim() != 1 || weight.numel() != input.size(1))
    return error_code(Error::shape_mismatch);
  if (output.scalar_type() != input.scalar_type())
    return error_code(Error::dtype_mismatch);
  // THE RESIDUAL, given with its output or neither: with it the op is vLLM's
  // fused_add_rms_norm on the sum (out = rms_norm(sum + residual), residual_out
  // = sum + residual), without it its rms_norm (out = rms_norm(sum)).
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
  // THE ALGORITHM, the op's (AllReduceAddRmsNormAlgorithm): one fused kernel's
  // body, or `<body>_split`: our all-reduce kernel at that body (at this
  // config's fields, its checks and views written out here as all_reduce's
  // are), then the norm on this rank's rows in place on its output (at the
  // add_rms_norm_ fields; with a residual, the add and the norm), back to back
  // on the stream. Every refusal, either path's, before any launch. The
  // add_rms_norm_ fields are a split's alone: given to a fused algorithm, or
  // missing from a split, refused.
  AllReduceAddRmsNormAlgorithm op_algorithm;
  if (algorithm_name == "one_shot_pull")
    op_algorithm = AllReduceAddRmsNormAlgorithm::one_shot_pull;
  else if (algorithm_name == "two_shot_pull_rows")
    op_algorithm = AllReduceAddRmsNormAlgorithm::two_shot_pull_rows;
  else if (algorithm_name == "two_shot_pull_columns")
    op_algorithm = AllReduceAddRmsNormAlgorithm::two_shot_pull_columns;
  else if (algorithm_name == "two_shot_push")
    op_algorithm = AllReduceAddRmsNormAlgorithm::two_shot_push;
  else if (algorithm_name == "one_shot_pull_split")
    op_algorithm = AllReduceAddRmsNormAlgorithm::one_shot_pull_split;
  else if (algorithm_name == "one_shot_push_split")
    op_algorithm = AllReduceAddRmsNormAlgorithm::one_shot_push_split;
  else if (algorithm_name == "one_shot_push_staged_split")
    op_algorithm = AllReduceAddRmsNormAlgorithm::one_shot_push_staged_split;
  else if (algorithm_name == "two_shot_pull_split")
    op_algorithm = AllReduceAddRmsNormAlgorithm::two_shot_pull_split;
  else if (algorithm_name == "two_shot_push_split")
    op_algorithm = AllReduceAddRmsNormAlgorithm::two_shot_push_split;
  else
    return error_code(Error::no_such_kernel);
  // WHAT IT LAUNCHES: the fused kernel's body, or the split's all-reduce's.
  bool split = false;
  AllReduceAddRmsNormKernelAlgorithm algorithm{};
  AllReduceAlgorithm all_reduce_algorithm{};
  switch (op_algorithm) {
    case AllReduceAddRmsNormAlgorithm::one_shot_pull:
      algorithm = AllReduceAddRmsNormKernelAlgorithm::one_shot_pull;
      break;
    case AllReduceAddRmsNormAlgorithm::two_shot_pull_rows:
      algorithm = AllReduceAddRmsNormKernelAlgorithm::two_shot_pull_rows;
      break;
    case AllReduceAddRmsNormAlgorithm::two_shot_pull_columns:
      algorithm = AllReduceAddRmsNormKernelAlgorithm::two_shot_pull_columns;
      break;
    case AllReduceAddRmsNormAlgorithm::two_shot_push:
      algorithm = AllReduceAddRmsNormKernelAlgorithm::two_shot_push;
      break;
    case AllReduceAddRmsNormAlgorithm::one_shot_pull_split:
      split = true, all_reduce_algorithm = AllReduceAlgorithm::one_shot_pull;
      break;
    case AllReduceAddRmsNormAlgorithm::one_shot_push_split:
      split = true, all_reduce_algorithm = AllReduceAlgorithm::one_shot_push;
      break;
    case AllReduceAddRmsNormAlgorithm::one_shot_push_staged_split:
      split = true,
      all_reduce_algorithm = AllReduceAlgorithm::one_shot_push_staged;
      break;
    case AllReduceAddRmsNormAlgorithm::two_shot_pull_split:
      split = true, all_reduce_algorithm = AllReduceAlgorithm::two_shot_pull;
      break;
    case AllReduceAddRmsNormAlgorithm::two_shot_push_split:
      split = true, all_reduce_algorithm = AllReduceAlgorithm::two_shot_push;
      break;
  }
  for (const std::optional<int64_t>& field :
       {add_rms_norm_grid_x, add_rms_norm_grid_y, add_rms_norm_grid_z,
        add_rms_norm_block_x, add_rms_norm_block_y, add_rms_norm_block_z,
        add_rms_norm_waves_per_eu, add_rms_norm_tile_m, add_rms_norm_tile_n,
        add_rms_norm_pack_bytes}) {
    if (!split && field) return error_code(Error::field_not_this_templates);
    if (split && !field) return error_code(Error::field_missing);
  }
  if (split)
    for (const int64_t v :
         {*add_rms_norm_grid_x, *add_rms_norm_grid_y, *add_rms_norm_grid_z,
          *add_rms_norm_block_x, *add_rms_norm_block_y, *add_rms_norm_block_z,
          *add_rms_norm_waves_per_eu, *add_rms_norm_tile_m,
          *add_rms_norm_tile_n, *add_rms_norm_pack_bytes})
      if (v < std::numeric_limits<int>::min() ||
          v > std::numeric_limits<int>::max())
        return error_code(Error::field_out_of_range);
  const void* input_ptr = input.data_ptr();
  const int64_t input_stride_m = input.stride(0),
                input_stride_n = input.stride(1);
  const int64_t input_size_m = input.size(0), input_size_n = input.size(1);
  const void* weight_ptr = weight.data_ptr();
  const int64_t weight_stride_n = weight.stride(0),
                weight_size_n = weight.size(0);
  // The residual's and its output's (none without a residual).
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
  // THE PROBLEM'S SHAPE: the input's.
  const int64_t m = input_size_m, n = input_size_n;
  const dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y),
                  static_cast<unsigned>(grid_z));
  const dim3 block(static_cast<unsigned>(block_x),
                   static_cast<unsigned>(block_y),
                   static_cast<unsigned>(block_z));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  // ---- PRECONDITIONS, every Error the call can be refused for, in this order:
  // the config alone; the call against the config; the compiled kernel on this
  // device; the regions. Past this section the call runs. ----
  // THE CONFIG'S QUANTIZATION, by name.
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_code(Error::no_such_quantization);
  const bool int4_config = quantization_name == "int4";
  // THE GRID: x alone (the kernel's one dimension), at least a block, at most
  // the blocks the signals have slots for. The block and the waves are the
  // build's (not_built, below).
  if (grid_y != 1 || grid_z != 1 || grid_x < 1 ||
      grid_x > s.symmetric.max_blocks)
    return error_code(Error::launch_out_of_range);
  if (tile_m < 1 || tile_n < 1) return error_code(Error::field_not_positive);
  // A PACK A LOAD THE HARDWARE HAS (2, 4, 8 or 16 bytes); INT4 at the widest
  // (the format's word is a pack of 8 values: common.cuh's QUANTIZATION).
  if (pack_bytes < 2 || pack_bytes > kPackBytes ||
      (pack_bytes & (pack_bytes - 1)) != 0)
    return error_code(Error::pack_out_of_range);
  if (int4_config && pack_bytes != kPackBytes)
    return error_code(Error::quantization_not_supported);
  // THE CALL'S DTYPE BUILT (the world is: open refuses the rest), and its
  // elements whole packs: at least an element a pack, whole packs a row (an
  // aligned load of the pack's width).
  if (std::find(kDtypes.begin(), kDtypes.end(), dtype) == kDtypes.end())
    return error_code(Error::dtype_not_built);
  if (pack_bytes < e) return error_code(Error::pack_out_of_range);
  if (n * e % pack_bytes != 0) return error_code(Error::row_not_packs);
  // THE WEIGHT read a pack at a time, a pack's worth of the input's elements a
  // load: its address on such a load.
  if (reinterpret_cast<uintptr_t>(weight_ptr) %
          (kPackBytes / e * weight.element_size()) !=
      0)
    return error_code(Error::not_aligned);
  // A LAYOUT THE PACKED LOADS CAN READ: every buffer's innermost stride 1, its
  // rows on whole packs.
  if (input_stride_n != 1 || output_stride_n != 1 || weight_stride_n != 1 ||
      (add && (residual_stride_n != 1 || residual_output_stride_n != 1)))
    return error_code(Error::inner_stride_not_one);
  if (input_stride_m * e % pack_bytes != 0 ||
      output_stride_m * e % pack_bytes != 0 ||
      (add && (residual_stride_m * e % pack_bytes != 0 ||
               residual_output_stride_m * e % pack_bytes != 0)))
    return error_code(Error::row_stride_not_packs);
  // THE OUTPUTS APART FROM THE INPUTS: no output's bytes meet an input's (the
  // input, the residual, the weight), nor each other's.
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
  // WHAT THE CALL MOVES: the config's quantization.
  const bool quantize_int4 = int4_config;
  const int shared_memory_bytes = 0;
  // ITS KERNELS, ITS VIEWS OF THE REGIONS AND WHAT THE REGIONS MUST HOLD, by
  // its path: the fused kernel's, or the split's all-reduce's and its norm's.
  void* kernel = nullptr;
  void* rms_norm_kernel = nullptr;
  dim3 rms_norm_grid, rms_norm_block;
  bool in_place = false;
  int64_t staging_need = 0, partials_need = 0;
  int64_t partials_stride_m = 0, partials_stride_n = 0, partials_size_m = 0,
          partials_size_n = 0;
  int64_t staging_stride_m = 0, staging_stride_n = 0, staging_size_m = 0,
          staging_size_n = 0;
  int64_t unfenced_stride_slot = 0, unfenced_stride_m = 0,
          unfenced_stride_n = 0, unfenced_size_slot = 0, unfenced_size_m = 0,
          unfenced_size_n = 0;
  if (split) {
    // ---- THE SPLIT'S ALL-REDUCE, at this config's fields: all_reduce's checks
    // and views, its own. ----
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
                            output_stride_m == n * output_stride_n;
    const bool tail = quantize_int4 && (two_shot ? m * n % kInt4Group != 0
                                                 : n % kInt4Group != 0);
    // THE PUSHED ONE-SHOT'S UNFENCED REGION HOLDS ITS TILES: a row of a block's
    // tile in a slot's lines, a line a pack of at most kLinePayloadBytes (a
    // narrower pack its line alone: kernels/all_reduce.cuh), INT4 a line a word
    // of its tile's whole groups; a block's TILE_M rows of a bank's
    // unfenced_blocks, a block its own slots (a grid past them is refused,
    // never cut).
    if (all_reduce_algorithm == AllReduceAlgorithm::one_shot_push) {
      const int64_t line_pack =
          std::min<int64_t>(pack_bytes, kLinePayloadBytes);
      const int64_t row_lines =
          quantize_int4
              ? (tile_n + kInt4Group - 1) / kInt4Group * kInt4GroupPacks
              : (tile_n * e + line_pack - 1) / line_pack;
      if (row_lines > s.symmetric.unfenced_slot_bytes / kLinePayloadBytes ||
          grid_x * tile_m > s.symmetric.unfenced_blocks)
        return error_code(Error::unfenced_too_small);
    }
    // THE INSTANCE: the kernel compiled at this config, on this device, for
    // this call (builds.py's kAllReduceInstances, a row each); a config the
    // table did not build is refused here.
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
    // can hang (HIP never refuses it). The blocks a CU holds from the kernel's
    // registers, LDS and the CU's wave slots, times the CUs.
    int blocks_a_cu = 0;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_a_cu, kernel, static_cast<int>(block_x),
            shared_memory_bytes) != hipSuccess)
      return error_code(Error::hip_failed);
    if (grid_x > blocks_a_cu * s.device.compute_units)
      return error_code(Error::grid_not_resident);
    // ---- ITS VIEWS OF THE REGIONS, by its all_reduce_algorithm, in elements:
    // a view [m, n] at strides [stride_m, stride_n] (the line push's inbox
    // [slot, m, n]); an empty view (size 0) where it reads none. The kernels
    // find a one-shot's payload in the staging from the input's sizes.
    // THE WHOLE CALL IN ONE PASS: the regions are sized from the largest call
    // the server makes, so a call past them is refused, never cut into bands.
    // Every exact kernel's input fits the staging whole (an eager call's host
    // copy,
    // INPUT TABLES; a captured call's is read where it is, refused alike, so
    // the eager and captured calls of one shape agree). ----
    // The two-shots' exact share is the flat walk's, whole packs, rounded up: a
    // rank's slice of a row in the partials, and the pushed two-shot's inbox
    // [WORLD, slot] in the staging, its slot a share of the whole call.
    const int64_t packs = n * e / pack_bytes;
    const int64_t slice_bytes = (packs + s.world - 1) / s.world * pack_bytes;
    const int64_t slot = (m * packs + s.world - 1) / s.world * pack_bytes / e;
    // THE INT4 FLAT WALK'S SHARE: a rank's whole groups of the call's, rounded
    // up, the last rank's to the end.
    const int64_t share =
        ((m * n + kInt4Group - 1) / kInt4Group + s.world - 1) / s.world *
        kInt4Group;
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
          partials_stride_m = share, partials_size_m = 1,
          partials_size_n = share;
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
        // unfenced_blocks rows (bank, row) of unfenced_slot_bytes of payload
        // each whatever the build (so a row is the same bytes at every build,
        // its epoch its own), a writer's rows contiguous.
        unfenced_stride_slot = 2 * s.symmetric.unfenced_blocks *
                               (s.symmetric.unfenced_slot_bytes / e);
        unfenced_stride_m = s.symmetric.unfenced_slot_bytes / e,
        unfenced_stride_n = 1;
        unfenced_size_slot = s.world,
        unfenced_size_m = 2 * s.symmetric.unfenced_blocks,
        unfenced_size_n = s.symmetric.unfenced_slot_bytes / e;
        break;
      case AllReduceAlgorithm::one_shot_push_staged: {
        // WORLD SLOTS OF AN INPUT in the staging (INT4: its payload, a slot
        // whole kInt4ScalesAlign), no partials.
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
    rms_norm_grid = dim3(static_cast<unsigned>(*add_rms_norm_grid_x),
                         static_cast<unsigned>(*add_rms_norm_grid_y),
                         static_cast<unsigned>(*add_rms_norm_grid_z));
    rms_norm_block = dim3(static_cast<unsigned>(*add_rms_norm_block_x),
                          static_cast<unsigned>(*add_rms_norm_block_y),
                          static_cast<unsigned>(*add_rms_norm_block_z));
    // THE SPLIT'S NORM ON ROWS, at the add_rms_norm_ fields (with a residual,
    // the add and the norm): its config, the call against it, its compiled
    // kernel on this device.
    // THE GRID: x alone, at least a block (it waits on no peer: any grid runs).
    if (*add_rms_norm_grid_y != 1 || *add_rms_norm_grid_z != 1 ||
        *add_rms_norm_grid_x < 1)
      return error_code(Error::launch_out_of_range);
    if (*add_rms_norm_tile_m < 1 || *add_rms_norm_tile_n < 1)
      return error_code(Error::field_not_positive);
    // A PACK A LOAD THE HARDWARE HAS (2, 4, 8 or 16 bytes), at least an
    // element, whole packs a row.
    if (*add_rms_norm_pack_bytes < 2 || *add_rms_norm_pack_bytes > kPackBytes ||
        (*add_rms_norm_pack_bytes & (*add_rms_norm_pack_bytes - 1)) != 0 ||
        *add_rms_norm_pack_bytes < e)
      return error_code(Error::pack_out_of_range);
    if (n * e % *add_rms_norm_pack_bytes != 0)
      return error_code(Error::row_not_packs);
    // THE WEIGHT read a pack at a time, a pack's worth of the output's elements
    // a load: its address on such a load.
    if (reinterpret_cast<uintptr_t>(weight_ptr) %
            (kPackBytes / e * weight.element_size()) !=
        0)
      return error_code(Error::not_aligned);
    // A LAYOUT THE PACKED LOADS CAN READ: the innermost stride 1, the output's
    // rows on whole packs.
    if (output_stride_n != 1 || weight_stride_n != 1)
      return error_code(Error::inner_stride_not_one);
    if (output_stride_m * e % *add_rms_norm_pack_bytes != 0 ||
        (add && (residual_stride_m * e % *add_rms_norm_pack_bytes != 0 ||
                 residual_output_stride_m * e % *add_rms_norm_pack_bytes != 0)))
      return error_code(Error::row_stride_not_packs);
    // THE INSTANCE: the norm compiled at this config, on this device, for this
    // call (builds.py's kAddRmsNormInstances: with a residual the add-norm,
    // else the norm); a row wider than the tile its CHUNKED instance.
    const bool rms_norm_chunked = n > *add_rms_norm_tile_n;
    for (const auto& i : kAddRmsNormInstances)
      if (i.residual == add && i.block_x == *add_rms_norm_block_x &&
          i.block_y == *add_rms_norm_block_y &&
          i.block_z == *add_rms_norm_block_z &&
          i.waves_per_eu == *add_rms_norm_waves_per_eu &&
          i.tile_m == *add_rms_norm_tile_m &&
          i.tile_n == *add_rms_norm_tile_n &&
          i.quantization == Quantization::exact &&
          i.pack_bytes == *add_rms_norm_pack_bytes &&
          i.wave_size == s.device.wave_size && i.dtype == dtype &&
          i.weight_dtype == weight_dtype && i.chunked == rms_norm_chunked)
        rms_norm_kernel = i.kernel;
    if (rms_norm_kernel == nullptr) return error_code(Error::not_built);
    // THE KERNEL ON THIS DEVICE, as its compiler left it (it waits on no peer:
    // no residency).
    hipFuncAttributes rms_norm_attributes;
    if (hipFuncGetAttributes(&rms_norm_attributes, rms_norm_kernel) !=
        hipSuccess)
      return error_code(Error::hip_failed);
    if (*add_rms_norm_block_x > rms_norm_attributes.maxThreadsPerBlock)
      return error_code(Error::block_too_large);
    if (static_cast<int64_t>(rms_norm_attributes.sharedSizeBytes) +
            shared_memory_bytes >
        s.device.lds_bytes)
      return error_code(Error::lds_too_large);
    if (static_cast<int64_t>(rms_norm_attributes.localSizeBytes) >
        s.device.scratch_a_lane)
      return error_code(Error::scratch_too_large);
  } else {
    // ---- THE FUSED KERNEL ----
    // WHAT THE CALL PICKS: in place, every exact algorithm (the pushed
    // two-shot's reduce-scatter reads its peers' inputs through the input table
    // too: reduce_scatter.cuh's columns_push); a row wider than the tile, the
    // build's CHUNKED instance; an INT4 row with a tail (not whole groups, or
    // not whole groups for every rank), its TAIL instance; with a residual its
    // RESIDUAL instance.
    in_place = !quantize_int4;
    const bool chunked = n > tile_n;
    const bool tail =
        quantize_int4 && (n % kInt4Group != 0 || n / kInt4Group % s.world != 0);
    // THE INSTANCE: the kernel compiled at this config, on this device, for
    // this call (builds.py's kAllReduceAddRmsNormInstances, a row each); a
    // config the table did not build, or a weight dtype no build has, is
    // refused here.
    for (const auto& i : kAllReduceAddRmsNormInstances)
      if (i.algorithm == algorithm && i.residual == add &&
          i.block_x == block_x && i.block_y == block_y &&
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
    // THE KERNEL ON THIS DEVICE, as its compiler left it.
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
    // THE GRID RESIDENT AT ONCE (a peer barrier spins on every peer's block).
    int blocks_a_cu = 0;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_a_cu, kernel, static_cast<int>(block_x),
            shared_memory_bytes) != hipSuccess)
      return error_code(Error::hip_failed);
    if (grid_x > blocks_a_cu * s.device.compute_units)
      return error_code(Error::grid_not_resident);
    // ---- ITS VIEWS OF THE REGIONS, in elements, [m, n] at stride n: the
    // staging holds the input (an eager call's copy; INT4 its payload), a
    // two-shot's partials the whole sum (each rank's rows or columns at their
    // place; INT4 its payload). ----
    const bool two_shot =
        algorithm != AllReduceAddRmsNormKernelAlgorithm::one_shot_pull;
    staging_stride_m = n, staging_stride_n = 1, staging_size_m = m,
    staging_size_n = n;
    partials_stride_m = n, partials_stride_n = 1,
    partials_size_m = two_shot ? m : 0, partials_size_n = n;
    // WHAT THE REGIONS MUST HOLD: the input whole and a two-shot's sum; INT4,
    // their payloads.
    staging_need = m * n * e, partials_need = two_shot ? m * n * e : 0;
    if (quantize_int4) {
      staging_need = std::max(staging_need, int4_payload_bytes(m, n));
      partials_need = std::max(partials_need, int4_payload_bytes(m, n));
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
  // EVERY RANK'S INPUT, where the kernel reads it: in place through an input
  // table (the exact pulls: `world` addresses), else in every rank's staging;
  // its strides either way. Captured: the next table slot, reserved for this
  // input (register_graph_buffers fills it when the capture exits), its rows at
  // the input's stride. Eager: the input copied into this rank's staging (it
  // fits: the staging_need above), read through the eager table, slot 0.
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
  // ---- THE LAUNCH, by its path: the split's all-reduce, then its norm (or
  // add-norm) on the sum in place, back to back; or the fused norm kernel, or
  // the fused add-norm kernel. ----
  if (split) {
    // THE LAUNCH: the instance's kernel, at the call's dtypes.
    const auto launch = [&]<typename T, typename W>() {
      const auto typed = reinterpret_cast<AllReduceKernel<T, W>>(kernel);
      typed<<<grid, block, shared_memory_bytes, stream>>>(
          static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
          static_cast<int>(input_size_m), static_cast<int>(input_size_n),
          reinterpret_cast<const T* const*>(peer_inputs), peer_inputs_size,
          peer_inputs_stride_m, peer_inputs_stride_n,
          static_cast<T*>(output_ptr), output_stride_m, output_stride_n,
          static_cast<int>(output_size_m), static_cast<int>(output_size_n),
          s.symmetric.partials_ptr_array, s.world, partials_stride_m,
          partials_stride_n, static_cast<int>(partials_size_m),
          static_cast<int>(partials_size_n), s.symmetric.staging_ptr_array,
          s.world, staging_stride_m, staging_stride_n,
          static_cast<int>(staging_size_m), static_cast<int>(staging_size_n),
          s.symmetric.unfenced_ptr_array, s.world, unfenced_stride_slot,
          unfenced_stride_m, unfenced_stride_n,
          static_cast<int>(unfenced_size_slot),
          static_cast<int>(unfenced_size_m), static_cast<int>(unfenced_size_n),
          s.unfenced_epochs, s.unfenced_epoch_words, int64_t{1},
          static_cast<int>(s.symmetric.unfenced_blocks),
          static_cast<int>(s.unfenced_epoch_words),
          s.symmetric.signals_ptr_array, s.world,
          s.symmetric.max_blocks * s.world, int64_t{s.world}, int64_t{1},
          int64_t{2}, s.symmetric.max_blocks, int64_t{s.world}, s.status_device,
          int64_t{1}, int64_t{kStatusFields}, s.rank, s.timeout_ticks);
    };
    if (dtype == at::ScalarType::BFloat16)
      launch.template operator()<c10::BFloat16, c10::BFloat16>();
    else
      launch.template operator()<c10::Half, c10::Half>();
    // ---- THE NORM (with a residual, the add and the norm), on the sum in
    // place: every element written by the thread that read it, after it read
    // it; without a residual its arguments null and empty. ----
    const auto launch_add_rms_norm = [&]<typename T, typename W>() {
      const auto typed =
          reinterpret_cast<AddRmsNormKernel<T, W>>(rms_norm_kernel);
      typed<<<rms_norm_grid, rms_norm_block, shared_memory_bytes, stream>>>(
          static_cast<const T*>(output_ptr), output_stride_m, output_stride_n,
          static_cast<int>(output_size_m), static_cast<int>(output_size_n),
          static_cast<const T*>(residual_ptr), residual_stride_m,
          residual_stride_n, static_cast<int>(residual_size_m),
          static_cast<int>(residual_size_n), static_cast<const W*>(weight_ptr),
          weight_stride_n, static_cast<int>(weight_size_n),
          static_cast<T*>(output_ptr), output_stride_m, output_stride_n,
          static_cast<int>(output_size_m), static_cast<int>(output_size_n),
          static_cast<T*>(residual_output_ptr), residual_output_stride_m,
          residual_output_stride_n, static_cast<int>(residual_output_size_m),
          static_cast<int>(residual_output_size_n), eps);
    };
    if (dtype == at::ScalarType::BFloat16 &&
        weight_dtype == at::ScalarType::Float)
      launch_add_rms_norm.template operator()<c10::BFloat16, float>();
    else if (dtype == at::ScalarType::BFloat16)
      launch_add_rms_norm.template operator()<c10::BFloat16, c10::BFloat16>();
    else if (weight_dtype == at::ScalarType::Float)
      launch_add_rms_norm.template operator()<c10::Half, float>();
    else
      launch_add_rms_norm.template operator()<c10::Half, c10::Half>();
  } else {
    // THE FUSED KERNEL'S LAUNCH, at the call's dtypes (without a residual its
    // arguments null and empty: the kernel reads none).
    const auto launch = [&]<typename T, typename W>() {
      const auto typed =
          reinterpret_cast<AllReduceAddRmsNormKernel<T, W>>(kernel);
      typed<<<grid, block, shared_memory_bytes, stream>>>(
          static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
          static_cast<int>(input_size_m), static_cast<int>(input_size_n),
          reinterpret_cast<const T* const*>(peer_inputs), peer_inputs_size,
          peer_inputs_stride_m, peer_inputs_stride_n,
          static_cast<const T*>(residual_ptr), residual_stride_m,
          residual_stride_n, static_cast<int>(residual_size_m),
          static_cast<int>(residual_size_n), static_cast<const W*>(weight_ptr),
          weight_stride_n, static_cast<int>(weight_size_n),
          static_cast<T*>(output_ptr), output_stride_m, output_stride_n,
          static_cast<int>(output_size_m), static_cast<int>(output_size_n),
          static_cast<T*>(residual_output_ptr), residual_output_stride_m,
          residual_output_stride_n, static_cast<int>(residual_output_size_m),
          static_cast<int>(residual_output_size_n),
          s.symmetric.partials_ptr_array, s.world, partials_stride_m,
          partials_stride_n, static_cast<int>(partials_size_m),
          static_cast<int>(partials_size_n), s.symmetric.staging_ptr_array,
          s.world, staging_stride_m, staging_stride_n,
          static_cast<int>(staging_size_m), static_cast<int>(staging_size_n),
          s.symmetric.signals_ptr_array, s.world,
          s.symmetric.max_blocks * s.world, int64_t{s.world}, int64_t{1},
          int64_t{2}, s.symmetric.max_blocks, int64_t{s.world}, s.status_device,
          int64_t{1}, int64_t{kStatusFields}, s.rank, s.timeout_ticks, eps);
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
  }
  // POSTCONDITION: the launches went in.
  if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
  return std::nullopt;
}
