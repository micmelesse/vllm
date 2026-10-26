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
// (vllm/distributed/device_communicators/hip_custom_communicator). No state crosses: Python owns
// the communicator's memory as addresses, through these, and hands every op what it reads.
std::vector<int64_t> hip_custom_communicator_layout(int64_t max_rows, int64_t max_cols,
                                                    int64_t world);
// What a communicator serving that box at `world` holds: [max_rows, max_cols,
// partials_bytes, inputs_bytes].
std::vector<int64_t> hip_custom_communicator_capacity(int64_t max_rows,
                                                      int64_t max_cols,
                                                      int64_t world);
int64_t hip_custom_communicator_wall_clock_khz(int64_t device);
int64_t hip_custom_communicator_alloc(int64_t bytes, bool shared);
int64_t hip_custom_communicator_free(int64_t ptr);
std::vector<int64_t> hip_custom_communicator_alloc_host(int64_t bytes);
int64_t hip_custom_communicator_free_host(int64_t host);
std::vector<int64_t> hip_custom_communicator_ipc_handle(int64_t ptr);
int64_t hip_custom_communicator_ipc_open(const std::vector<int64_t>& handle);
int64_t hip_custom_communicator_ipc_close(int64_t base);
int64_t hip_custom_communicator_pointer_device(int64_t ptr);
void hip_custom_communicator_write_words(int64_t ptr, const std::vector<int64_t>& words);
void hip_custom_communicator_fill_words(int64_t ptr, int64_t word, int64_t count);
std::vector<int64_t> hip_custom_communicator_read_words(int64_t ptr, int64_t count);
std::vector<int64_t> hip_custom_communicator_read_host(int64_t host, int64_t count);
void hip_custom_communicator_write_host(int64_t host, int64_t index, int64_t word);

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
// "one_shot_pull", "one_shot_push", "two_shot_pull", "two_shot_push" or "rows", the grid, the
// block and its waves a EU, the tile, the quantization "exact" or "int4"). The norms on rows are
// the second kernel of an op Python composes after our all-reduce. Each returns the Error's
// number it was refused for, or -1 when it ran (hip_custom_communicator.cu's Answer).

int64_t hip_custom_communicator_all_reduce(
    int64_t rank, int64_t world, int64_t shared_buffer, int64_t private_buffer,
    int64_t status_device, int64_t timeout_ticks, int64_t max_rows, int64_t staging_bytes,
    int64_t partials_bytes, int64_t table_slot, bool checked_after_launch, torch::Tensor& input,
    torch::Tensor& output,
    const std::string& kernel,
    const std::string& algorithm,
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
    const std::string& quantization, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes);
int64_t hip_custom_communicator_all_reduce_rms_norm(
    int64_t rank, int64_t world, int64_t shared_buffer, int64_t private_buffer,
    int64_t status_device, int64_t timeout_ticks, int64_t max_rows, int64_t staging_bytes,
    int64_t partials_bytes, int64_t table_slot, bool checked_after_launch, torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& output,
    double eps,
    const std::string& kernel,
    const std::string& algorithm,
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
    const std::string& quantization, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes);
int64_t hip_custom_communicator_all_reduce_add_rms_norm(
    int64_t rank, int64_t world, int64_t shared_buffer, int64_t private_buffer,
    int64_t status_device, int64_t timeout_ticks, int64_t max_rows, int64_t staging_bytes,
    int64_t partials_bytes, int64_t table_slot, bool checked_after_launch, torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    torch::Tensor& output,
    torch::Tensor& residual_output,
    double eps,
    const std::string& kernel,
    const std::string& algorithm,
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
    const std::string& quantization, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes);
int64_t hip_custom_communicator_all_reduce_add_attn_res_rms_norm(
    int64_t rank, int64_t world, int64_t shared_buffer, int64_t private_buffer,
    int64_t status_device, int64_t timeout_ticks, int64_t max_rows, int64_t staging_bytes,
    int64_t partials_bytes, int64_t table_slot, bool checked_after_launch, torch::Tensor& input,
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
    const std::string& quantization, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes);
int64_t hip_custom_communicator_rms_norm(
    int64_t world, bool checked_after_launch, torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& output,
    double eps,
    const std::string& kernel,
    const std::string& algorithm,
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
    const std::string& quantization, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes);
int64_t hip_custom_communicator_add_rms_norm(
    int64_t world, bool checked_after_launch, torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    torch::Tensor& output,
    torch::Tensor& residual_output,
    double eps,
    const std::string& kernel,
    const std::string& algorithm,
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
    const std::string& quantization, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes);
int64_t hip_custom_communicator_add_attn_res_rms_norm(
    int64_t world, bool checked_after_launch, torch::Tensor& input,
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
    const std::string& quantization, int64_t pack_bytes,
    std::optional<int64_t> quantize_from_bytes);
