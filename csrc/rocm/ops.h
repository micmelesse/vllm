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

// Opened on `device` over the process groups named `cpu_group` and `device_group`, a collective:
// the handle or the Error's number. The IPC handles go round in C++.
std::tuple<std::optional<int64_t>, std::optional<int64_t>> rocm_comms_open(
    const std::string& cpu_group, const std::string& device_group, int64_t device);
// The buffers a capture recorded, registered over `group`, a collective.
void rocm_comms_register_captured(fptr_t handle_ptr, const std::string& group);
// The probe, every rank together: the round trip to each peer (ns), then GB/s pulled from one
// peer, pulled from every peer, pushed into every peer, and both at once (each way).
std::tuple<std::vector<double>, double, double, double, double> rocm_comms_probe(
    fptr_t handle_ptr, int64_t bytes, int64_t ping_iters, int64_t traffic_iters, int64_t trials);
torch::Tensor rocm_comms_stamps();


void rocm_comms_dispose(fptr_t handle_ptr);




// Every op takes the same four values last, its options: quant_bits, a lossy precision (none:
// exact), then a forced template by name with its blocks and threads (none: select's).

// The planners, one per op family, each given the call's own tensors. A variant at the torch
// boundary: the kernel's template name, grid and threads, or the Error's number.
std::tuple<std::optional<std::string>, std::optional<int64_t>, std::optional<int64_t>,
           std::optional<int64_t>>
rocm_comms_plan_all_reduce(
    fptr_t handle_ptr, const torch::Tensor& inp,
    std::optional<int64_t> quant_bits, std::optional<std::string> template_,
    std::optional<int64_t> launch_blocks, std::optional<int64_t> launch_threads);
std::tuple<std::optional<std::string>, std::optional<int64_t>, std::optional<int64_t>,
           std::optional<int64_t>>
rocm_comms_plan_all_reduce_rms_norm(
    fptr_t handle_ptr, const torch::Tensor& inp, const torch::Tensor& weight, bool add,
    std::optional<int64_t> quant_bits, std::optional<std::string> template_,
    std::optional<int64_t> launch_blocks, std::optional<int64_t> launch_threads);
std::tuple<std::optional<std::string>, std::optional<int64_t>, std::optional<int64_t>,
           std::optional<int64_t>>
rocm_comms_plan_all_reduce_add_attn_res_rms_norm(
    fptr_t handle_ptr, const torch::Tensor& inp,
    std::optional<int64_t> quant_bits, std::optional<std::string> template_,
    std::optional<int64_t> launch_blocks, std::optional<int64_t> launch_threads);
std::tuple<std::optional<std::string>, std::optional<int64_t>, std::optional<int64_t>,
           std::optional<int64_t>>
rocm_comms_plan_all_reduce_rms_norm_gemm(
    fptr_t handle_ptr, const torch::Tensor& inp, const torch::Tensor& gemm_weight, bool add,
    std::optional<int64_t> quant_bits, std::optional<std::string> template_,
    std::optional<int64_t> launch_blocks, std::optional<int64_t> launch_threads);
std::tuple<std::optional<std::string>, std::optional<int64_t>, std::optional<int64_t>,
           std::optional<int64_t>>
rocm_comms_plan_all_reduce_rms_scale_add(
    fptr_t handle_ptr, const torch::Tensor& inp, const torch::Tensor& out,
    std::optional<int64_t> quant_bits, std::optional<std::string> template_,
    std::optional<int64_t> launch_blocks, std::optional<int64_t> launch_threads);
// A variant at the torch boundary: the arch or the Error's number, exactly one set.
std::tuple<std::optional<std::string>, std::optional<int64_t>> rocm_comms_supported(
    int64_t device, int64_t world);
// The build's dtypes by name, its worlds, a pack's bytes and a staging's, and its ops' and errors'
// names in their enums' order.
std::tuple<std::vector<std::string>, std::vector<int64_t>, int64_t, int64_t,
           std::vector<std::string>, std::vector<std::string>>
rocm_comms_build_info();

void rocm_comms_all_reduce(fptr_t handle_ptr, torch::Tensor& out, torch::Tensor& inp,
                           std::optional<int64_t> quant_bits, std::optional<std::string> template_,
                           std::optional<int64_t> launch_blocks,
                           std::optional<int64_t> launch_threads);

void rocm_comms_all_reduce_rms_norm(fptr_t handle_ptr, torch::Tensor& out, torch::Tensor& inp,
                                    torch::Tensor& weight, double eps,
                                    std::optional<int64_t> quant_bits,
                                    std::optional<std::string> template_,
                                    std::optional<int64_t> launch_blocks,
                                    std::optional<int64_t> launch_threads);

void rocm_comms_all_reduce_add_rms_norm(fptr_t handle_ptr, torch::Tensor& out,
                                        torch::Tensor& residual_out, torch::Tensor& inp,
                                        torch::Tensor& residual, torch::Tensor& weight,
                                        double eps, std::optional<int64_t> quant_bits,
                                        std::optional<std::string> template_,
                                        std::optional<int64_t> launch_blocks,
                                        std::optional<int64_t> launch_threads);

void rocm_comms_all_reduce_rms_norm_gemm(
    fptr_t handle_ptr, torch::Tensor& out, int64_t out_col0, torch::Tensor& inp,
    torch::Tensor& norm_weight, double eps, torch::Tensor& gemm_weight,
    torch::Tensor& workspace, std::optional<int64_t> quant_bits,
    std::optional<std::string> template_,
    std::optional<int64_t> launch_blocks, std::optional<int64_t> launch_threads);

void rocm_comms_all_reduce_rms_scale_add(fptr_t handle_ptr, torch::Tensor& out,
                                         torch::Tensor& inp, double eps,
                                         std::optional<int64_t> quant_bits,
                                         std::optional<std::string> template_,
                                         std::optional<int64_t> launch_blocks,
                                         std::optional<int64_t> launch_threads);

void rocm_comms_all_reduce_rms_norm_gemm_add(
    fptr_t handle_ptr, torch::Tensor& out, int64_t out_col0, torch::Tensor& inp,
    torch::Tensor& norm_weight, double eps, torch::Tensor& gemm_weight,
    torch::Tensor& workspace, std::optional<int64_t> quant_bits,
    std::optional<std::string> template_,
    std::optional<int64_t> launch_blocks, std::optional<int64_t> launch_threads);

void rocm_comms_all_reduce_add_attn_res_rms_norm(
    fptr_t handle_ptr, torch::Tensor& prefix, torch::Tensor& out, torch::Tensor& inp,
    torch::Tensor& blocks, torch::Tensor& norm_weight, torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& out_norm_weight, int64_t num_blocks,
    int64_t write_idx, double eps, double out_eps, bool has_prefix,
    std::optional<int64_t> quant_bits,
    std::optional<std::string> template_,
    std::optional<int64_t> launch_blocks, std::optional<int64_t> launch_threads);


