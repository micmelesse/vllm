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
                             int64_t max_cols, int64_t log_mode,
                             int64_t rccl_ready);
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
// flat, and four ints an op), and the unfenced region's blocks (the pushed
// one-shot's most blocks).
std::tuple<std::vector<std::string>, std::vector<int64_t>, int64_t,
           std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>, std::vector<std::string>,
           std::vector<int64_t>, std::vector<int64_t>, std::vector<std::string>,
           std::vector<std::string>, std::vector<int64_t>, int64_t>
hip_custom_communicator_build_info();

// Each op once, every argument required (optionality is Python's, and so is the pick: Python's
// tuned table): its config point last in the essential order (the kernel by name, its algorithm
// "one_shot", "two_shot" or "rows" and direction "pull" or "push", the grid, the block and its
// waves a EU, the tile, the quantization "exact" or "int4"). The norms on rows are
// the second kernel of an op Python composes after our all-reduce. Each returns the Error's
// number it was refused for, or -1 when it ran (hip_custom_communicator.cu's Answer).

int64_t hip_custom_communicator_all_reduce(
    fptr_t handle_ptr, torch::Tensor& input,
    torch::Tensor& output,
    const std::string& kernel,
    const std::string& algorithm,
    const std::string& direction,
    int64_t grid_x,
    int64_t grid_y,
    int64_t grid_z,
    int64_t block_x,
    int64_t block_y,
    int64_t block_z,
    int64_t waves_per_eu,
    int64_t tile_m,
    int64_t tile_n,
    int64_t tile_k,
    const std::string& quantization);
int64_t hip_custom_communicator_all_reduce_rms_norm(
    fptr_t handle_ptr, torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& output,
    double eps,
    const std::string& kernel,
    const std::string& algorithm,
    const std::string& direction,
    int64_t grid_x,
    int64_t grid_y,
    int64_t grid_z,
    int64_t block_x,
    int64_t block_y,
    int64_t block_z,
    int64_t waves_per_eu,
    int64_t tile_m,
    int64_t tile_n,
    int64_t tile_k,
    const std::string& quantization);
int64_t hip_custom_communicator_all_reduce_add_rms_norm(
    fptr_t handle_ptr, torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    torch::Tensor& output,
    torch::Tensor& residual_output,
    double eps,
    const std::string& kernel,
    const std::string& algorithm,
    const std::string& direction,
    int64_t grid_x,
    int64_t grid_y,
    int64_t grid_z,
    int64_t block_x,
    int64_t block_y,
    int64_t block_z,
    int64_t waves_per_eu,
    int64_t tile_m,
    int64_t tile_n,
    int64_t tile_k,
    const std::string& quantization);
int64_t hip_custom_communicator_all_reduce_add_attn_res_rms_norm(
    fptr_t handle_ptr, torch::Tensor& input,
    const std::optional<torch::Tensor>& prefix,
    torch::Tensor& blocks,
    torch::Tensor& norm_weight,
    torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& output_norm_weight,
    torch::Tensor& prefix_output,
    torch::Tensor& output,
    const std::optional<torch::Tensor>& block_output,
    int64_t num_blocks,
    double eps,
    double output_eps,
    const std::string& kernel,
    const std::string& algorithm,
    const std::string& direction,
    int64_t grid_x,
    int64_t grid_y,
    int64_t grid_z,
    int64_t block_x,
    int64_t block_y,
    int64_t block_z,
    int64_t waves_per_eu,
    int64_t tile_m,
    int64_t tile_n,
    int64_t tile_k,
    const std::string& quantization);
int64_t hip_custom_communicator_rms_norm(
    fptr_t handle_ptr, torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& output,
    double eps,
    const std::string& kernel,
    const std::string& algorithm,
    const std::string& direction,
    int64_t grid_x,
    int64_t grid_y,
    int64_t grid_z,
    int64_t block_x,
    int64_t block_y,
    int64_t block_z,
    int64_t waves_per_eu,
    int64_t tile_m,
    int64_t tile_n,
    int64_t tile_k,
    const std::string& quantization);
int64_t hip_custom_communicator_add_rms_norm(
    fptr_t handle_ptr, torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    torch::Tensor& output,
    torch::Tensor& residual_output,
    double eps,
    const std::string& kernel,
    const std::string& algorithm,
    const std::string& direction,
    int64_t grid_x,
    int64_t grid_y,
    int64_t grid_z,
    int64_t block_x,
    int64_t block_y,
    int64_t block_z,
    int64_t waves_per_eu,
    int64_t tile_m,
    int64_t tile_n,
    int64_t tile_k,
    const std::string& quantization);
int64_t hip_custom_communicator_add_attn_res_rms_norm(
    fptr_t handle_ptr, torch::Tensor& input,
    const std::optional<torch::Tensor>& prefix,
    torch::Tensor& blocks,
    torch::Tensor& norm_weight,
    torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& output_norm_weight,
    torch::Tensor& prefix_output,
    torch::Tensor& output,
    const std::optional<torch::Tensor>& block_output,
    int64_t num_blocks,
    double eps,
    double output_eps,
    const std::string& kernel,
    const std::string& algorithm,
    const std::string& direction,
    int64_t grid_x,
    int64_t grid_y,
    int64_t grid_z,
    int64_t block_x,
    int64_t block_y,
    int64_t block_z,
    int64_t waves_per_eu,
    int64_t tile_m,
    int64_t tile_n,
    int64_t tile_k,
    const std::string& quantization);
