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

// ROCm TP collectives
// (vllm/distributed/device_communicators/hip_custom_communicator). The context
// is a stateful object, so it crosses as an opaque handle the way custom
// all-reduce's does.
using fptr_t = int64_t;

// Opened on `device` over the process groups named `cpu_group` and
// `device_group`, a collective: the handle or the Error's number. The IPC
// handles go round in C++.
std::tuple<std::optional<int64_t>, std::optional<int64_t>>
hip_custom_communicator_open(const std::string& cpu_group,
                             const std::string& device_group, int64_t device,
                             double timeout_seconds, int64_t max_rows,
                             int64_t max_cols, int64_t quantization,
                             int64_t log_mode, int64_t rccl_ready,
                             const std::vector<int64_t>& points);
// What an opened communicator launches for an op's call: a plain call's pick, or, a launch named,
// that kernel's tuned config (hip_custom_communicator.cu).
std::vector<int64_t> hip_custom_communicator_tuned(
    fptr_t handle_ptr, const std::string& op_name, int64_t rows, int64_t cols,
    const std::string& dtype, const std::string& quantization,
    std::optional<std::string> algorithm,
    std::optional<std::string> direction);
// What a communicator serving that box at `world` holds: [max_rows, max_cols,
// partials_bytes, inputs_bytes].
std::vector<int64_t> hip_custom_communicator_capacity(int64_t max_rows,
                                                      int64_t max_cols,
                                                      int64_t world);
int64_t hip_custom_communicator_wall_clock_khz(int64_t device);

// The collective close, its handshakes over `cpu_group` bounded by
// `timeout_seconds`: 0, or the ClosePhase that failed. Then dispose.
int64_t hip_custom_communicator_close(fptr_t handle_ptr,
                                      const std::string& cpu_group,
                                      double timeout_seconds);
// The inputs a capture recorded, registered over `cpu_group` (a collective,
// at capture exit): 0, 1 the ranks' counts differ, 2 no such group.
int64_t hip_custom_communicator_register_captured(fptr_t handle_ptr,
                                                  const std::string& cpu_group);
// Every open communicator torn down alone (exit, a signal): never waits on a
// peer.
void hip_custom_communicator_abandon_all(double wait_seconds);
void hip_custom_communicator_dispose(fptr_t handle_ptr);
// The rank's fault record, [where, rank, block, peer, count, want,
// elapsed_ticks], or empty while no wait has given up; and the abort every wait
// on the rank then leaves at.
std::vector<int64_t> hip_custom_communicator_status(fptr_t handle_ptr);
std::vector<int64_t> hip_custom_communicator_signals(fptr_t handle_ptr);
// The state log's every word (log.cuh): the handle's, or with 0 the last log
// this process made (an open that failed or is still running).
std::vector<int64_t> hip_custom_communicator_log(fptr_t handle_ptr);
void hip_custom_communicator_abort(fptr_t handle_ptr);

// A variant at the torch boundary: the arch or the Error's number, exactly one
// set.
std::tuple<std::optional<std::string>, std::optional<int64_t>>
hip_custom_communicator_supported(int64_t device, int64_t world);
// The build's dtypes by name, its worlds, a pack's bytes, its
// ops', errors' and fault places' names in their enums' order, and each
// template's name, op and BuildConfigs (flat, a build's fields in the names'
// order, 0 where its op's BuildConfig has none, with each template's count),
// and each op's plain call (its dtypes and its weight's besides its input's,
// flat, and four ints an op).
std::tuple<std::vector<std::string>, std::vector<int64_t>, int64_t,
           std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>, std::vector<std::string>,
           std::vector<int64_t>, std::vector<int64_t>, std::vector<std::string>,
           std::vector<std::string>, std::vector<int64_t>>
hip_custom_communicator_build_info();

// Each op takes its own forcing last: the template by name, every field of one
// of its builds, and the launch, all or none (none: select's), the build's
// quantization ("none" or "int4") optional within it. It returns the
// Error's number it was refused for, or -1 and what ran
// (hip_custom_communicator.cu's Answer).

std::vector<int64_t> hip_custom_communicator_all_reduce(
    fptr_t handle_ptr, torch::Tensor& input, torch::Tensor& output,
    std::optional<std::string> algorithm, std::optional<std::string> direction,
    std::optional<int64_t> tile_m, std::optional<int64_t> tile_n,
    std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid, std::optional<int64_t> waves_per_eu,
    std::optional<std::string> quantization);
std::vector<int64_t> hip_custom_communicator_all_reduce_rms_norm(
    fptr_t handle_ptr, torch::Tensor& input, torch::Tensor& weight,
    torch::Tensor& output, double eps, std::optional<std::string> algorithm,
    std::optional<std::string> direction, std::optional<int64_t> tile_m,
    std::optional<int64_t> tile_n, std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid, std::optional<int64_t> waves_per_eu,
    std::optional<std::string> quantization);
std::vector<int64_t> hip_custom_communicator_all_reduce_add_rms_norm(
    fptr_t handle_ptr, torch::Tensor& input, torch::Tensor& residual,
    torch::Tensor& weight, torch::Tensor& output,
    torch::Tensor& residual_output, double eps,
    std::optional<std::string> algorithm, std::optional<std::string> direction,
    std::optional<int64_t> tile_m, std::optional<int64_t> tile_n,
    std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid, std::optional<int64_t> waves_per_eu,
    std::optional<std::string> quantization);
std::vector<int64_t> hip_custom_communicator_all_reduce_add_attn_res_rms_norm(
    fptr_t handle_ptr, torch::Tensor& input,
    const std::optional<torch::Tensor>& prefix, torch::Tensor& blocks,
    torch::Tensor& norm_weight, torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& output_norm_weight,
    torch::Tensor& prefix_output, torch::Tensor& output,
    const std::optional<torch::Tensor>& block_output, int64_t num_blocks,
    double eps, double output_eps, std::optional<std::string> algorithm,
    std::optional<std::string> direction, std::optional<int64_t> tile_m,
    std::optional<int64_t> tile_n, std::optional<int64_t> tile_k,
    std::optional<int64_t> reduce_scatter_blocks,
    std::optional<int64_t> threads_per_block,
    std::optional<int64_t> blocks_per_grid, std::optional<int64_t> waves_per_eu,
    std::optional<std::string> quantization);
