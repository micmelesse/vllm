#pragma once

#include <torch/all.h>

torch::Tensor LLMM1(at::Tensor& in_a, at::Tensor& in_b,
                    const int64_t rows_per_block);

torch::Tensor wvSplitK(const at::Tensor& in_a, const at::Tensor& in_b,
                       const std::optional<at::Tensor>& in_bias,
                       const int64_t CuCount);

torch::Tensor wvSplitK_int4_g(const at::Tensor& in_a, const at::Tensor& in_b,
                              const at::Tensor& in_scale,
                              const std::optional<at::Tensor>& in_zero_points,
                              const std::optional<at::Tensor>& in_bias,
                              const int64_t CuCount, const int64_t group_size);

torch::Tensor wvSplitKrc(const at::Tensor& in_a, const at::Tensor& in_b,
                         const std::optional<at::Tensor>& in_bias,
                         const int64_t CuCount);

void wvSplitKQ(const at::Tensor& in_a, const at::Tensor& in_b,
               const std::optional<at::Tensor>& in_bias, at::Tensor& out_c,
               const at::Tensor& scale_a, const at::Tensor& scale_b,
               const int64_t CuCount);

torch::Tensor gptq_gemm_rdna3(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_qzeros, torch::Tensor b_scales,
                              bool use_v2_format);

torch::Tensor gptq_gemm_rdna3_wmma(torch::Tensor a, torch::Tensor b_q_weight,
                                   torch::Tensor b_qzeros,
                                   torch::Tensor b_scales, bool use_v2_format);

void moe_gptq_gemm_rdna3(torch::Tensor a, torch::Tensor c,
                         torch::Tensor b_q_weight, torch::Tensor b_scales,
                         torch::Tensor b_qzeros, torch::Tensor topk_weights,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor num_tokens_post_padded, int64_t top_k,
                         int64_t block_size_m, bool mul_topk_weight,
                         int64_t output_topk);

void paged_attention(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& query_start_loc, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const std::optional<torch::Tensor>& fp8_out_scale,
    const std::string& mfma_type);

// ROCm TP collectives (vllm/distributed/device_communicators/rocm_comms). The context is a
// stateful object, so it crosses as an opaque handle the way custom all-reduce's does.
using fptr_t = int64_t;

fptr_t rocm_comms_init(int64_t rank, int64_t world_size, int64_t self_signal,
                       const std::vector<std::vector<int64_t>>& signal_handles,
                       const std::vector<int64_t>& signal_offsets, int64_t peer_slab,
                       int64_t peer_slab_bytes, int64_t scratch_bytes);

void rocm_comms_dispose(fptr_t comms);

void rocm_comms_register_buffer(fptr_t comms,
                                const std::vector<std::vector<int64_t>>& handles,
                                const std::vector<int64_t>& offsets, int64_t self_ptr);

std::vector<int64_t> rocm_comms_pending_graph_buffers(fptr_t comms);

void rocm_comms_register_graph_buffers(
    fptr_t comms, const std::vector<std::vector<int64_t>>& handles,
    const std::vector<std::vector<int64_t>>& offsets);

int64_t rocm_comms_pending_count(fptr_t comms);

void rocm_comms_all_reduce(fptr_t comms, torch::Tensor& out, torch::Tensor& inp,
                           int64_t algo, int64_t small_limit, int64_t blocks,
                           int64_t threads);

void rocm_comms_all_reduce_rms_norm(fptr_t comms, torch::Tensor& out, torch::Tensor& inp,
                                    torch::Tensor& weight, double eps, int64_t algo,
                                    int64_t small_limit, int64_t blocks, int64_t threads);

void rocm_comms_all_reduce_fused_add_rms_norm(fptr_t comms, torch::Tensor& out,
                                              torch::Tensor& residual_out,
                                              torch::Tensor& inp, torch::Tensor& residual,
                                              torch::Tensor& weight, double eps,
                                              int64_t algo, int64_t small_limit,
                                              int64_t blocks, int64_t threads);


std::tuple<std::vector<int64_t>, int64_t> rocm_comms_handle_and_offset(int64_t ptr);

std::vector<int64_t> rocm_comms_sizes();
