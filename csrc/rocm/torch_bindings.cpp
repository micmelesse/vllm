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

#ifdef VLLM_ROCM_GFX1100
  // W4A16 GPTQ kernels for AMD RDNA3 (gfx1100).
  rocm_ops.def(
      "gptq_gemm_rdna3(Tensor a, Tensor b_q_weight, Tensor b_qzeros, "
      "Tensor b_scales, Tensor b_g_idx, bool use_v2_format) -> Tensor");
  rocm_ops.impl("gptq_gemm_rdna3", torch::kCUDA, &gptq_gemm_rdna3);

  rocm_ops.def(
      "gptq_gemm_rdna3_wmma(Tensor a, Tensor b_q_weight, Tensor b_qzeros, "
      "Tensor b_scales, Tensor b_g_idx, bool use_v2_format) -> Tensor");
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

  // ROCm TP collectives. The context is a stateful object behind an opaque handle, so
  // everything but the two collectives is a CPU-side call on that handle; only
  // all_reduce/all_gather touch tensors and therefore carry a device impl.
  rocm_ops.def(
      "rocm_comms_init(int rank, int world_size, int self_signal, "
      "int[][] signal_handles, int[] signal_offsets, int peer_slab, "
      "int peer_slab_bytes) -> int");
  rocm_ops.impl("rocm_comms_init", torch::kCPU, &rocm_comms_init);

  rocm_ops.def("rocm_comms_dispose(int comms) -> ()");
  rocm_ops.impl("rocm_comms_dispose", torch::kCPU, &rocm_comms_dispose);

  rocm_ops.def(
      "rocm_comms_register_buffer(int comms, int[][] handles, int[] offsets, "
      "int self_ptr) -> ()");
  rocm_ops.impl("rocm_comms_register_buffer", torch::kCPU, &rocm_comms_register_buffer);

  rocm_ops.def("rocm_comms_pending_graph_buffers(int comms) -> int[]");
  rocm_ops.impl("rocm_comms_pending_graph_buffers", torch::kCPU,
                &rocm_comms_pending_graph_buffers);

  rocm_ops.def(
      "rocm_comms_register_graph_buffers(int comms, int[][] handles, "
      "int[][] offsets) -> ()");
  rocm_ops.impl("rocm_comms_register_graph_buffers", torch::kCPU,
                &rocm_comms_register_graph_buffers);

  rocm_ops.def("rocm_comms_pending_count(int comms) -> int");
  rocm_ops.impl("rocm_comms_pending_count", torch::kCPU, &rocm_comms_pending_count);

  rocm_ops.def(
      "rocm_comms_all_reduce(int comms, Tensor! out, Tensor inp, int algo, "
      "int blocks, int threads) -> ()");
  rocm_ops.impl("rocm_comms_all_reduce", torch::kCUDA, &rocm_comms_all_reduce);

  rocm_ops.def(
      "rocm_comms_all_gather(int comms, Tensor! out, Tensor inp, int algo, "
      "int blocks, int threads) -> ()");
  rocm_ops.impl("rocm_comms_all_gather", torch::kCUDA, &rocm_comms_all_gather);

  rocm_ops.def("rocm_comms_handle_and_offset(int ptr) -> (int[], int)");
  rocm_ops.impl("rocm_comms_handle_and_offset", torch::kCPU,
                &rocm_comms_handle_and_offset);

  rocm_ops.def("rocm_comms_sizes() -> int[]");
  rocm_ops.impl("rocm_comms_sizes", torch::kCPU, &rocm_comms_sizes);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
