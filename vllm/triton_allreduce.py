# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused AllReduce + RMSNorm + Quantization for ROCm.

This module provides fused operations that combine:
1. All-reduce across tensor parallel GPUs
2. Optional residual addition
3. RMS normalization
4. FP8 per-tensor quantization

The fusion reduces memory bandwidth by avoiding intermediate writes.

Two implementations are provided:
1. `fused_allreduce_rms_quant` - Graph-level fusion using existing vLLM ops
   (used by RocmAiterAllReduceFusionPass)
2. `triton_allreduce` - Iris-based kernel fusion (experimental)
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


# ============================================================================
# Fused AllReduce + RMSNorm + FP8 Quant (Graph-level fusion)
# ============================================================================
# These functions are used by RocmAiterAllReduceFusionPass to replace the pattern:
#     all_reduce → rms_norm → per_tensor_quant
# Currently implements "graph fusion" (3 separate kernel calls wrapped in one op).
# TODO: Replace with true kernel fusion using AITER's all_reduce_rmsnorm_quant().


def fused_allreduce_rms_quant(
    input: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    quant_scale: torch.Tensor,
    quant_dtype: torch.dtype,
    group_name: str,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """
    Fused AllReduce + RMSNorm + FP8 Per-Tensor Quant.
    
    Args:
        input: Input tensor to all-reduce
        rms_weight: RMSNorm weight
        rms_eps: RMSNorm epsilon
        quant_scale: Quantization scale (can be None for dynamic)
        quant_dtype: Target quantization dtype (e.g., torch.float8_e4m3fn)
        group_name: TP group name for all-reduce
        residual: Optional residual tensor for fused add
        
    Returns: (allreduce_out, rms_out, residual_out, quant_out, quant_scale_out)
             residual_out is None if residual is None
    """
    # Step 1: All-reduce
    allreduce_out = torch.ops.vllm.all_reduce(input, group_name=group_name)

    # Step 2: RMSNorm (with or without residual add)
    if residual is not None:
        rms_out, residual_out = torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add(
            allreduce_out, residual, rms_weight, rms_eps
        )
    else:
        rms_out = torch.ops.vllm.rocm_aiter_rms_norm(
            allreduce_out, rms_weight, rms_eps
        )
        residual_out = None

    # Step 3: FP8 Quant
    quant_out, quant_scale_out = torch.ops.vllm.rocm_aiter_per_tensor_quant(
        rms_out, quant_dtype, quant_scale
    )

    return allreduce_out, rms_out, residual_out, quant_out, quant_scale_out


# Wrapper implementations for torch custom op registration


def _rocm_aiter_fused_allreduce_rms_quant_impl(
    input: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    quant_scale: torch.Tensor,
    quant_dtype: torch.dtype,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused AllReduce + RMSNorm + FP8 Quant (no residual).
    
    Returns: (allreduce_out, rms_out, quant_out, quant_scale_out)
    """
    allreduce_out, rms_out, _, quant_out, quant_scale_out = fused_allreduce_rms_quant(
        input=input,
        rms_weight=rms_weight,
        rms_eps=rms_eps,
        quant_scale=quant_scale,
        quant_dtype=quant_dtype,
        group_name=group_name,
        residual=None,
    )
    return allreduce_out, rms_out, quant_out, quant_scale_out


def _rocm_aiter_fused_allreduce_rms_quant_fake(
    input: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    quant_scale: torch.Tensor,
    quant_dtype: torch.dtype,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake impl for torch.compile - returns empty tensors with correct shapes."""
    allreduce_out = torch.empty_like(input)
    rms_out = torch.empty_like(input)
    quant_out = torch.empty_like(input, dtype=quant_dtype)
    quant_scale_out = torch.empty(1, device=input.device, dtype=torch.float32)
    return allreduce_out, rms_out, quant_out, quant_scale_out


def _rocm_aiter_fused_allreduce_add_rms_quant_impl(
    input: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    quant_scale: torch.Tensor,
    quant_dtype: torch.dtype,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused AllReduce + RMSNorm with Add + FP8 Quant (with residual).
    
    Returns: (allreduce_out, rms_out, residual_out, quant_out, quant_scale_out)
    """
    allreduce_out, rms_out, residual_out, quant_out, quant_scale_out = fused_allreduce_rms_quant(
        input=input,
        rms_weight=rms_weight,
        rms_eps=rms_eps,
        quant_scale=quant_scale,
        quant_dtype=quant_dtype,
        group_name=group_name,
        residual=residual,
    )
    # residual_out is guaranteed to be non-None when residual is provided
    assert residual_out is not None
    return allreduce_out, rms_out, residual_out, quant_out, quant_scale_out


def _rocm_aiter_fused_allreduce_add_rms_quant_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    quant_scale: torch.Tensor,
    quant_dtype: torch.dtype,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake impl for torch.compile - returns empty tensors with correct shapes."""
    allreduce_out = torch.empty_like(input)
    rms_out = torch.empty_like(input)
    residual_out = torch.empty_like(input)
    quant_out = torch.empty_like(input, dtype=quant_dtype)
    quant_scale_out = torch.empty(1, device=input.device, dtype=torch.float32)
    return allreduce_out, rms_out, residual_out, quant_out, quant_scale_out


# ============================================================================
# Iris CCL Baseline Implementation (Experimental)
# ============================================================================
# This follows the exact pattern from the Iris CCL example:
# https://github.com/micmelesse/iris/blob/main/iris/ccl/all_reduce.py
#
# Key points:
# - Allocates Iris context and buffers per-call (not cached)
# - Calls all_reduce_preamble() and all_reduce() with same exact buffers
# - Works correctly for any M, N size
# - Not optimized for CUDA graph capture (but works for eager mode)

try:
    import iris
    from iris.ccl import Config
    IRIS_AVAILABLE = True
except ImportError:
    IRIS_AVAILABLE = False
    logger.debug("Iris not available, triton_allreduce will not work")


def _ccl_baseline(
    input_: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CCL baseline: per-call Iris allocation following Iris example exactly.
    
    This is the reference implementation that always works correctly.
    """
    if not IRIS_AVAILABLE:
        raise RuntimeError("Iris not available for triton_allreduce")
    
    M, N = input_.shape
    cur_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    
    # Feature flags
    do_residual = residual is not None
    do_rmsnorm = weight is not None
    
    # Allocate outputs
    output = torch.empty_like(input_)
    residual_out = torch.empty_like(input_)
    
    # Check if capturing - barriers not allowed during capture
    is_capturing = torch.cuda.is_current_stream_capturing()
    
    logger.info(f"triton_allreduce [ccl_baseline]: rank={cur_rank}, M={M}, N={N}, capturing={is_capturing}")
    
    # Use large heap like Iris tests (8GB)
    heap_size = 2**33  # 8GB
    
    # Create fresh Iris instance for this call (exactly like the Iris example)
    shmem = iris.iris(heap_size)
    
    # Allocate input and output tensors on symmetric heap (exactly like Iris example)
    iris_input = shmem.zeros((M, N), dtype=input_.dtype)
    iris_output = shmem.zeros((M, N), dtype=input_.dtype)
    
    # Copy input to symmetric heap tensor
    iris_input.copy_(input_)
    
    # Barrier to ensure all ranks have copied input (skip during graph capture)
    if not is_capturing:
        shmem.barrier()
    
    logger.info(f"rank={cur_rank} iris_input sample: {iris_input[0, :5].tolist()}")
    
    # Use Config with two_shot variant (default and most tested)
    config = Config(all_reduce_variant="two_shot")
    
    # Following exact Iris test pattern:
    # 1. all_reduce_preamble with config
    workspace = shmem.ccl.all_reduce_preamble(iris_output, iris_input, config=config)
    
    # 2. barrier to ensure all ranks complete preamble (skip during graph capture)
    if not is_capturing:
        shmem.barrier()
    
    # 3. all_reduce with config and workspace
    shmem.ccl.all_reduce(iris_output, iris_input, config=config, workspace=workspace)
    
    # 4. synchronize
    torch.cuda.synchronize()
    
    logger.info(f"rank={cur_rank} iris_output sample: {iris_output[0, :5].tolist()}")
    
    # Now iris_output contains the all-reduced result on all ranks
    # Add residual and compute RMSNorm if needed
    if do_residual:
        assert residual is not None
        result = iris_output + residual
    else:
        result = iris_output
    
    # Store to residual_out
    residual_out.copy_(result)
    
    if do_rmsnorm:
        assert weight is not None
        # Simple RMSNorm: out = (x / sqrt(mean(x^2) + eps)) * weight
        variance = (result.float() ** 2).mean(dim=-1, keepdim=True)
        rrms = torch.rsqrt(variance + eps)
        normed = (result.float() * rrms * weight.float()).to(input_.dtype)
        output.copy_(normed)
    else:
        output.copy_(result)
    
    logger.info(f"triton_allreduce [ccl_baseline]: rank={cur_rank}, done")
    
    return output, residual_out


# ============================================================================
# Iris-based triton_allreduce Implementation
# ============================================================================


def _triton_allreduce_impl(
    input_: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Internal implementation of fused all-reduce with optional residual add and RMS normalization.
    
    Args:
        input_: Input tensor to all-reduce, shape (M, N)
        residual: Optional residual tensor to add after all-reduce
        weight: Optional RMSNorm weight (if None, skip RMSNorm)
        eps: RMSNorm epsilon
    
    Returns:
        output: All-reduced (and optionally normalized) output
        residual_out: all_reduce(input) + residual (or just all_reduce(input) if no residual)
    """
    assert input_.dim() == 2, f"Expected 2D input, got {input_.dim()}D"
    
    M, N = input_.shape
    
    if residual is not None:
        assert residual.dim() == 2, f"Expected 2D residual, got {residual.dim()}D"
        assert input_.shape == residual.shape, f"Shape mismatch: {input_.shape} vs {residual.shape}"
    
    if weight is not None:
        assert weight.shape == (N,), f"Weight shape mismatch: {weight.shape} vs ({N},)"
    
    return _ccl_baseline(input_, residual, weight, eps)


def _triton_allreduce_fake(
    input_: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for torch.compile tracing."""
    return torch.empty_like(input_), torch.empty_like(input_)


# Register as torch custom op for torch.compile compatibility
try:
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(
        op_name="triton_allreduce",
        op_func=_triton_allreduce_impl,
        mutates_args=[],
        fake_impl=_triton_allreduce_fake,
    )
    
    # Register the fused allreduce ops (used by RocmAiterAllReduceFusionPass)
    direct_register_custom_op(
        op_name="rocm_aiter_fused_allreduce_rms_quant",
        op_func=_rocm_aiter_fused_allreduce_rms_quant_impl,
        mutates_args=[],
        fake_impl=_rocm_aiter_fused_allreduce_rms_quant_fake,
    )
    direct_register_custom_op(
        op_name="rocm_aiter_fused_allreduce_add_rms_quant",
        op_func=_rocm_aiter_fused_allreduce_add_rms_quant_impl,
        mutates_args=[],
        fake_impl=_rocm_aiter_fused_allreduce_add_rms_quant_fake,
    )
except Exception as e:
    logger.warning(f"Failed to register triton_allreduce custom op: {e}")


# ============================================================================
# Public API
# ============================================================================


def triton_allreduce(
    input_: torch.Tensor,
    residual: torch.Tensor | None = None,
    norm: "RMSNorm | None" = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    All-reduce with optional fused residual add and RMS normalization.
    
    Uses Iris symmetric memory for efficient inter-GPU communication on ROCm.
    
    Args:
        input_: Input tensor from RowParallelLinear (pre-reduce partial sums)
                Shape: (num_tokens, hidden_dim) e.g., (M, 8192)
        residual: Optional residual tensor for skip connection
                  Shape: (num_tokens, hidden_dim) e.g., (M, 8192)
        norm: Optional RMSNorm layer with:
              - norm.weight: (hidden_dim,) e.g., (8192,)
              - norm.variance_epsilon: scalar e.g., 1e-5
        
    Returns:
        output: All-reduced (and optionally normalized) output, shape (num_tokens, hidden_dim)
        residual_out: all_reduce(input) + residual if residual provided, else just all_reduce(input)
    """
    weight = norm.weight if norm is not None else None
    eps = norm.variance_epsilon if norm is not None else 1e-5
    
    return torch.ops.vllm.triton_allreduce(
        input_,
        residual,
        weight,
        eps,
    )
