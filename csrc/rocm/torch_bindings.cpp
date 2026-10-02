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
  rocm_ops.def(
      "LLMM1(Tensor in_a, Tensor in_b, int rows_per_block) -> "
      "Tensor");
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
      "wvSplitK_int4_g(Tensor in_a, Tensor in_b, Tensor in_scale, "
      "Tensor? in_zero_points, Tensor? in_bias, int CuCount, "
      "int group_size) -> Tensor");
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
      "gptq_gemm_rdna3(Tensor a, Tensor b_q_weight, Tensor b_qzeros, "
      "Tensor b_scales, bool use_v2_format) -> Tensor");
  rocm_ops.impl("gptq_gemm_rdna3", torch::kCUDA, &gptq_gemm_rdna3);

  rocm_ops.def(
      "gptq_gemm_rdna3_wmma(Tensor a, Tensor b_q_weight, Tensor b_qzeros, "
      "Tensor b_scales, bool use_v2_format) -> Tensor");
  rocm_ops.impl("gptq_gemm_rdna3_wmma", torch::kCUDA, &gptq_gemm_rdna3_wmma);

  rocm_ops.def(
      "moe_gptq_gemm_rdna3(Tensor a, Tensor! c, Tensor b_q_weight, "
      "Tensor b_scales, Tensor b_qzeros, Tensor topk_weights, "
      "Tensor sorted_token_ids, Tensor expert_ids, "
      "Tensor num_tokens_post_padded, "
      "int top_k, int block_size_m, bool mul_topk_weight, "
      "int output_topk) -> ()");
  rocm_ops.impl("moe_gptq_gemm_rdna3", torch::kCUDA, &moe_gptq_gemm_rdna3);
#endif

  // Custom attention op
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  rocm_ops.def(
      "paged_attention(Tensor! out, Tensor exp_sums,"
      "                Tensor max_logits, Tensor tmp_out,"
      "                Tensor query, Tensor key_cache,"
      "                Tensor value_cache, int num_kv_heads,"
      "                float scale, Tensor block_tables,"
      "                Tensor seq_lens,"
      "                Tensor? query_start_loc,"
      "                int block_size,"
      "                int max_seq_len,"
      "                Tensor? alibi_slopes,"
      "                str kv_cache_dtype,"
      "                Tensor k_scale, Tensor v_scale,"
      "                Tensor? fp8_out_scale,"
      "                str mfma_type) -> ()");
  rocm_ops.impl("paged_attention", torch::kCUDA, &paged_attention);

  // ROCm TP collectives. Only the two collectives take tensors, so only they get a
  // schema plus a device impl. THE REST TAKE NONE, and an op with no tensor argument
  // has nothing for the dispatcher to select a backend from -- a `kCPU` impl is then
  // unreachable and the call raises "no fallback function is registered". So they are
  // bound directly, which is what vLLM's quick-reduce does with its own handle ops.
  rocm_ops.def("rocm_comms_open", &rocm_comms_open);
  rocm_ops.def("rocm_comms_stamps", &rocm_comms_stamps);
  rocm_ops.def("rocm_comms_dispose", &rocm_comms_dispose);
  rocm_ops.def(
      "rocm_comms_probe(int handle_ptr, int bytes, int ping_iters, int traffic_iters, "
      "int trials) -> (float[], float, float, float, float)",
      &rocm_comms_probe);
  // The planners: what runs a call, or the Error it meets.
  rocm_ops.def("rocm_comms_plan_all_reduce(int handle_ptr, Tensor inp, "
               "int? quant_bits, str? template_, int? launch_blocks, "
               "int? launch_threads) -> (str?, int?, int?, int?)",
               &rocm_comms_plan_all_reduce);
  rocm_ops.def("rocm_comms_plan_all_reduce_rms_norm(int handle_ptr, Tensor inp, Tensor weight, "
               "bool add, int? quant_bits, str? template_, int? launch_blocks, "
               "int? launch_threads) -> (str?, int?, int?, int?)",
               &rocm_comms_plan_all_reduce_rms_norm);
  rocm_ops.def("rocm_comms_plan_all_reduce_add_attn_res_rms_norm(int handle_ptr, Tensor inp, "
               "int? quant_bits, str? template_, int? launch_blocks, "
               "int? launch_threads) -> (str?, int?, int?, int?)",
               &rocm_comms_plan_all_reduce_add_attn_res_rms_norm);
  rocm_ops.def("rocm_comms_plan_all_reduce_rms_norm_gemm(int handle_ptr, Tensor inp, "
               "Tensor gemm_weight, bool add, int? quant_bits, str? template_, int? launch_blocks, "
               "int? launch_threads) -> (str?, int?, int?, int?)",
               &rocm_comms_plan_all_reduce_rms_norm_gemm);
  rocm_ops.def("rocm_comms_plan_all_reduce_rms_scale_add(int handle_ptr, Tensor inp, Tensor out, "
               "int? quant_bits, str? template_, int? launch_blocks, "
               "int? launch_threads) -> (str?, int?, int?, int?)",
               &rocm_comms_plan_all_reduce_rms_scale_add);
  rocm_ops.def("rocm_comms_supported(int device, int world) -> (str?, int?)",
               &rocm_comms_supported);
  rocm_ops.def("rocm_comms_build_info() -> (str[], int[], int, int, str[], str[])",
               &rocm_comms_build_info);
  rocm_ops.def("rocm_comms_register_captured", &rocm_comms_register_captured);

  rocm_ops.def(
      "rocm_comms_all_reduce(int handle_ptr, Tensor! out, Tensor inp, "
      "int? quant_bits, str? template_, int? launch_blocks, int? launch_threads) -> ()");
  rocm_ops.impl("rocm_comms_all_reduce", torch::kCUDA, &rocm_comms_all_reduce);

  // FUSED: all-reduce then vLLM's `rms_norm`, and all-reduce then `fused_add_rms_norm`
  // (which also returns the sum plus residual). `eps` is a float in the schema because
  // torch has no `double` there; the kernel narrows it.
  rocm_ops.def(
      "rocm_comms_all_reduce_rms_norm(int handle_ptr, Tensor! out, Tensor inp, "
      "Tensor weight, float eps, "
      "int? quant_bits, str? template_, int? launch_blocks, int? launch_threads) -> ()");
  rocm_ops.impl("rocm_comms_all_reduce_rms_norm", torch::kCUDA,
                &rocm_comms_all_reduce_rms_norm);
  rocm_ops.def(
      "rocm_comms_all_reduce_add_rms_norm(int handle_ptr, Tensor! out, "
      "Tensor! residual_out, Tensor inp, Tensor residual, Tensor weight, float eps, "
      "int? quant_bits, str? template_, int? launch_blocks, int? launch_threads) -> ()");
  rocm_ops.impl("rocm_comms_all_reduce_add_rms_norm", torch::kCUDA,
                &rocm_comms_all_reduce_add_rms_norm);

  rocm_ops.def(
      "rocm_comms_all_reduce_add_attn_res_rms_norm(int handle_ptr, Tensor! prefix, Tensor! out, "
      "Tensor inp, "
      "Tensor! blocks, Tensor norm_weight, Tensor qk_weight, Tensor? out_norm_weight, "
      "int num_blocks, int write_idx, float eps, float out_eps, bool has_prefix, "
      "int? quant_bits, str? template_, int? launch_blocks, int? launch_threads) -> ()");
  rocm_ops.impl("rocm_comms_all_reduce_add_attn_res_rms_norm", torch::kCUDA,
                &rocm_comms_all_reduce_add_attn_res_rms_norm);

  rocm_ops.def(
      "rocm_comms_all_reduce_rms_norm_gemm(int handle_ptr, Tensor! out, int out_col0, "
      "Tensor inp, Tensor norm_weight, float eps, Tensor gemm_weight, Tensor! workspace, "
      "int? quant_bits, str? template_, int? launch_blocks, int? launch_threads) -> ()");
  rocm_ops.impl("rocm_comms_all_reduce_rms_norm_gemm", torch::kCUDA,
                &rocm_comms_all_reduce_rms_norm_gemm);

  rocm_ops.def(
      "rocm_comms_all_reduce_rms_scale_add(int handle_ptr, Tensor! out, Tensor inp, float eps, "
      "int? quant_bits, str? template_, int? launch_blocks, int? launch_threads) -> ()");
  rocm_ops.impl("rocm_comms_all_reduce_rms_scale_add", torch::kCUDA,
                &rocm_comms_all_reduce_rms_scale_add);

  rocm_ops.def(
      "rocm_comms_all_reduce_rms_norm_gemm_add(int handle_ptr, Tensor! out, int out_col0, "
      "Tensor inp, Tensor norm_weight, float eps, Tensor gemm_weight, Tensor! workspace, "
      "int? quant_bits, str? template_, int? launch_blocks, int? launch_threads) -> ()");
  rocm_ops.impl("rocm_comms_all_reduce_rms_norm_gemm_add", torch::kCUDA,
                &rocm_comms_all_reduce_rms_norm_gemm_add);

}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
