#include "core/registration.h"
#include "rocm/ops.h"

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, rocm_ops) {
  // vLLM custom ops for rocm

// skinny_gemms.cu (LLMM1/wvSplitK/wvSplitKrc/wvSplitKQ) is excluded on gfx1250
// (gfx9/gfx11 ISA, unsupported there); skip these registrations to avoid
// undefined symbols. vLLM uses default/Triton GEMM for these ops on gfx1250.
#ifndef VLLM_SKIP_SKINNY_GEMMS
  // Custom gemm op for matrix-vector multiplication
  rocm_ops.def("LLMM1(Tensor in_a, Tensor in_b, int rows_per_block) -> Tensor");
  rocm_ops.impl("LLMM1", torch::kCUDA, &LLMM1);

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitK(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitK", torch::kCUDA, &wvSplitK);

  // W4A16 grouped skinny GEMM: packed int4 weights, per-group scales,
  // optional zero points [M/8, K/group_size] int32 for asymmetric
  // quantization
  rocm_ops.def(
      "wvSplitK_int4_g(Tensor in_a, Tensor in_b, Tensor in_scale, Tensor? "
      "in_zero_points, "
      "Tensor? in_bias, int CuCount, int group_size) -> Tensor");
  rocm_ops.impl("wvSplitK_int4_g", torch::kCUDA, &wvSplitK_int4_g);

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitKrc(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitKrc", torch::kCUDA, &wvSplitKrc);

  // wvSplitK for fp8
  rocm_ops.def(
      "wvSplitKQ(Tensor in_a, Tensor in_b, Tensor? in_bias, Tensor! out_c, "
      "Tensor scale_a, "
      "          Tensor scale_b, int CuCount) -> ()");
  rocm_ops.impl("wvSplitKQ", torch::kCUDA, &wvSplitKQ);
#endif  // VLLM_SKIP_SKINNY_GEMMS

#ifdef VLLM_ROCM_GFX1100
  // W4A16 GPTQ kernels for AMD RDNA3 (gfx1100).
  rocm_ops.def(
      "gptq_gemm_rdna3(Tensor a, Tensor b_q_weight, Tensor b_qzeros, Tensor "
      "b_scales, "
      "bool use_v2_format) -> Tensor");
  rocm_ops.impl("gptq_gemm_rdna3", torch::kCUDA, &gptq_gemm_rdna3);

  rocm_ops.def(
      "gptq_gemm_rdna3_wmma(Tensor a, Tensor b_q_weight, Tensor b_qzeros, "
      "Tensor b_scales, "
      "bool use_v2_format) -> Tensor");
  rocm_ops.impl("gptq_gemm_rdna3_wmma", torch::kCUDA, &gptq_gemm_rdna3_wmma);

  rocm_ops.def(
      "moe_gptq_gemm_rdna3(Tensor a, Tensor! c, Tensor b_q_weight, Tensor "
      "b_scales, "
      "Tensor b_qzeros, Tensor topk_weights, Tensor sorted_token_ids, Tensor "
      "expert_ids, "
      "Tensor num_tokens_post_padded, int top_k, int block_size_m, bool "
      "mul_topk_weight, "
      "int output_topk) -> ()");
  rocm_ops.impl("moe_gptq_gemm_rdna3", torch::kCUDA, &moe_gptq_gemm_rdna3);
#endif

  // Custom attention op
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  rocm_ops.def(
      "paged_attention(Tensor! out, Tensor exp_sums,                Tensor "
      "max_logits, "
      "Tensor tmp_out,                Tensor query, Tensor key_cache, "
      "               Tensor value_cache, int num_kv_heads,                "
      "float scale, "
      "Tensor block_tables,                Tensor seq_lens, "
      "               Tensor? query_start_loc,                int block_size, "
      "               int max_seq_len,                Tensor? alibi_slopes, "
      "               str kv_cache_dtype,                Tensor k_scale, "
      "Tensor v_scale, "
      "               Tensor? fp8_out_scale,                str mfma_type) -> "
      "()");
  rocm_ops.impl("paged_attention", torch::kCUDA, &paged_attention);

  // hip_custom_communicator.cu is built only with gfx950 in the arch list
  // (CMakeLists.txt); without it its ops are not registered.
#ifndef VLLM_SKIP_HIP_CUSTOM_COMMUNICATOR
  // ROCm TP collectives. Only the ops that take tensors get a schema (the
  // collectives also a device impl). THE REST TAKE NONE, and an op with no
  // tensor argument has nothing for the dispatcher to select a backend from --
  // a `kCPU` impl is then unreachable and the call raises "no fallback function
  // is registered". So they are bound directly, which is what vLLM's
  // quick-reduce does with its own handle ops, and their schema is INFERRED
  // from the C++ signature: one source, nothing to keep in step (a hand-written
  // schema that disagreed with its function failed the library load,
  // 2026-10-14).
  rocm_ops.def("hip_custom_communicator_layout",
               &hip_custom_communicator_layout);
  rocm_ops.def("hip_custom_communicator_capacity",
               &hip_custom_communicator_capacity);
  rocm_ops.def("hip_custom_communicator_alloc",
               &hip_custom_communicator_alloc);
  rocm_ops.def("hip_custom_communicator_free",
               &hip_custom_communicator_free);
  rocm_ops.def("hip_custom_communicator_alloc_host",
               &hip_custom_communicator_alloc_host);
  rocm_ops.def("hip_custom_communicator_free_host",
               &hip_custom_communicator_free_host);
  rocm_ops.def("hip_custom_communicator_ipc_handle",
               &hip_custom_communicator_ipc_handle);
  rocm_ops.def("hip_custom_communicator_ipc_open",
               &hip_custom_communicator_ipc_open);
  rocm_ops.def("hip_custom_communicator_ipc_close",
               &hip_custom_communicator_ipc_close);
  rocm_ops.def("hip_custom_communicator_pointer_device",
               &hip_custom_communicator_pointer_device);
  rocm_ops.def("hip_custom_communicator_write_words",
               &hip_custom_communicator_write_words);
  rocm_ops.def("hip_custom_communicator_fill_words",
               &hip_custom_communicator_fill_words);
  rocm_ops.def("hip_custom_communicator_read_words",
               &hip_custom_communicator_read_words);
  rocm_ops.def("hip_custom_communicator_read_host",
               &hip_custom_communicator_read_host);
  rocm_ops.def("hip_custom_communicator_write_host",
               &hip_custom_communicator_write_host);
  rocm_ops.def("hip_custom_communicator_wall_clock_khz",
               &hip_custom_communicator_wall_clock_khz);
  rocm_ops.def("hip_custom_communicator_supported",
               &hip_custom_communicator_supported);
  rocm_ops.def("hip_custom_communicator_build_info",
               &hip_custom_communicator_build_info);
  // THE LAUNCH STATE, made and freed. Its maker's schema is written out, so Python passes its fields by
  // name (an inferred schema names none); a schema that disagrees with its function fails the
  // library load. The free takes its handle alone, inferred.
  rocm_ops.def(
      "hip_custom_communicator_launch_state(int rank, int world, int shared_buffer, int private_buffer, int status_device, int timeout_ticks, int max_rows, int staging_bytes, int partials_bytes, bool checked_after_launch) -> int",
      &hip_custom_communicator_launch_state);
  rocm_ops.def("hip_custom_communicator_free_launch_state",
               &hip_custom_communicator_free_launch_state);

  // EACH OP ONCE, on its communicator's state (a handle, above), its tensors and values, then its
  // config's fields. `eps` is a float in the schema because torch has no `double` there; the kernel
  // narrows it.

  rocm_ops.def(
      "hip_custom_communicator_all_reduce(int state, int table_slot, Tensor input, Tensor! output, str algorithm, int grid_x, int grid_y, int grid_z, int block_x, int block_y, int block_z, int waves_per_eu, int tile_m, int tile_n, str quantization, int pack_bytes, int? quantize_from_bytes) -> int");
  rocm_ops.impl("hip_custom_communicator_all_reduce", torch::kCUDA,
                &hip_custom_communicator_all_reduce);

  rocm_ops.def(
      "hip_custom_communicator_all_reduce_rms_norm(int state, int table_slot, Tensor input, Tensor weight, Tensor! output, float eps, str algorithm, int grid_x, int grid_y, int grid_z, int block_x, int block_y, int block_z, int waves_per_eu, int tile_m, int tile_n, str quantization, int pack_bytes, int? quantize_from_bytes) -> int");
  rocm_ops.impl("hip_custom_communicator_all_reduce_rms_norm", torch::kCUDA,
                &hip_custom_communicator_all_reduce_rms_norm);

  rocm_ops.def(
      "hip_custom_communicator_all_reduce_add_rms_norm(int state, int table_slot, Tensor input, Tensor residual, Tensor weight, Tensor! output, Tensor! residual_output, float eps, str algorithm, int grid_x, int grid_y, int grid_z, int block_x, int block_y, int block_z, int waves_per_eu, int tile_m, int tile_n, str quantization, int pack_bytes, int? quantize_from_bytes) -> int");
  rocm_ops.impl("hip_custom_communicator_all_reduce_add_rms_norm", torch::kCUDA,
                &hip_custom_communicator_all_reduce_add_rms_norm);

  rocm_ops.def(
      "hip_custom_communicator_all_reduce_add_attn_res_rms_norm(int state, int table_slot, Tensor input, Tensor? prefix, Tensor blocks, Tensor norm_weight, Tensor qk_weight, Tensor? output_norm_weight, Tensor! prefix_output, Tensor! output, Tensor!? block_output, int num_blocks, float eps, float output_eps, str algorithm, int grid_x, int grid_y, int grid_z, int block_x, int block_y, int block_z, int waves_per_eu, int tile_m, int tile_n, int tile_k, str quantization, int pack_bytes, int? quantize_from_bytes) -> int");
  rocm_ops.impl("hip_custom_communicator_all_reduce_add_attn_res_rms_norm", torch::kCUDA,
                &hip_custom_communicator_all_reduce_add_attn_res_rms_norm);

  rocm_ops.def(
      "hip_custom_communicator_rms_norm(int state, Tensor input, Tensor weight, Tensor! output, float eps, int grid_x, int grid_y, int grid_z, int block_x, int block_y, int block_z, int waves_per_eu, int tile_m, int tile_n, str quantization, int pack_bytes, int? quantize_from_bytes) -> int");
  rocm_ops.impl("hip_custom_communicator_rms_norm", torch::kCUDA,
                &hip_custom_communicator_rms_norm);

  rocm_ops.def(
      "hip_custom_communicator_add_rms_norm(int state, Tensor input, Tensor residual, Tensor weight, Tensor! output, Tensor! residual_output, float eps, int grid_x, int grid_y, int grid_z, int block_x, int block_y, int block_z, int waves_per_eu, int tile_m, int tile_n, str quantization, int pack_bytes, int? quantize_from_bytes) -> int");
  rocm_ops.impl("hip_custom_communicator_add_rms_norm", torch::kCUDA,
                &hip_custom_communicator_add_rms_norm);

  rocm_ops.def(
      "hip_custom_communicator_add_attn_res_rms_norm(int state, Tensor input, Tensor? prefix, Tensor blocks, Tensor norm_weight, Tensor qk_weight, Tensor? output_norm_weight, Tensor! prefix_output, Tensor! output, Tensor!? block_output, int num_blocks, float eps, float output_eps, int grid_x, int grid_y, int grid_z, int block_x, int block_y, int block_z, int waves_per_eu, int tile_m, int tile_n, int tile_k, str quantization, int pack_bytes, int? quantize_from_bytes) -> int");
  rocm_ops.impl("hip_custom_communicator_add_attn_res_rms_norm", torch::kCUDA,
                &hip_custom_communicator_add_attn_res_rms_norm);
#endif  // VLLM_SKIP_HIP_CUSTOM_COMMUNICATOR
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
