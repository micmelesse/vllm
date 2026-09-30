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

int64_t rocm_comms_alloc(int64_t scratch_bytes, int64_t staging_bytes);
fptr_t rocm_comms_init(int64_t rank, int64_t world_size, int64_t self_memory,
                       const std::vector<std::vector<int64_t>>& signal_handles,
                       const std::vector<int64_t>& signal_offsets, int64_t max_buffers,
                       int64_t scratch_bytes, int64_t staging_bytes, double sync_timeout_s);
torch::Tensor rocm_comms_staging(fptr_t handle_ptr);
double rocm_comms_ping_pong(fptr_t handle_ptr, int64_t peer, int64_t iters);
torch::Tensor rocm_comms_stamps();
double rocm_comms_peer_read(fptr_t handle_ptr, int64_t peer, int64_t bytes, int64_t iters);


void rocm_comms_dispose(fptr_t handle_ptr);


std::vector<int64_t> rocm_comms_pending_graph_buffers(fptr_t handle_ptr);

void rocm_comms_register_graph_buffers(
    fptr_t handle_ptr, const std::vector<std::vector<int64_t>>& handles,
    const std::vector<std::vector<int64_t>>& offsets);


// Every op takes the same four integers last: quant_bits, the precision it accepts (16:
// exact), then its launch: kernel, launch_blocks, launch_threads (-1 and zeros: tune.cuh's).
bool rocm_comms_admits(fptr_t handle_ptr, int64_t op, int64_t rows, int64_t hidden,
                       int64_t element_size, int64_t cols, int64_t quant_bits, int64_t kernel,
                       int64_t launch_blocks, int64_t launch_threads);

void rocm_comms_all_reduce(fptr_t handle_ptr, torch::Tensor& out, torch::Tensor& inp,
                           int64_t quant_bits, int64_t kernel, int64_t launch_blocks,
                           int64_t launch_threads);

void rocm_comms_all_reduce_rms_norm(fptr_t handle_ptr, torch::Tensor& out, torch::Tensor& inp,
                                    torch::Tensor& weight, double eps, int64_t quant_bits,
                                    int64_t kernel, int64_t launch_blocks,
                                    int64_t launch_threads);

void rocm_comms_all_reduce_add_rms_norm(fptr_t handle_ptr, torch::Tensor& out,
                                        torch::Tensor& residual_out, torch::Tensor& inp,
                                        torch::Tensor& residual, torch::Tensor& weight,
                                        double eps, int64_t quant_bits, int64_t kernel,
                                        int64_t launch_blocks, int64_t launch_threads);

void rocm_comms_all_reduce_rms_norm_gemm_add(
    fptr_t handle_ptr, torch::Tensor& out, int64_t out_col0, torch::Tensor& inp,
    torch::Tensor& norm_weight, double eps, torch::Tensor& gemm_weight,
    torch::Tensor& workspace, int64_t quant_bits, int64_t kernel, int64_t launch_blocks,
    int64_t launch_threads);

void rocm_comms_all_reduce_add_attn_res_rms_norm(
    fptr_t handle_ptr, torch::Tensor& prefix, torch::Tensor& out, torch::Tensor& inp,
    torch::Tensor& blocks, torch::Tensor& norm_weight, torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& out_norm_weight, int64_t num_blocks,
    int64_t write_idx, double eps, double out_eps, bool has_prefix, int64_t quant_bits,
    int64_t kernel, int64_t launch_blocks, int64_t launch_threads);

std::tuple<std::vector<int64_t>, int64_t> rocm_comms_handle_and_offset(int64_t ptr);

