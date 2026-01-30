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

Two implementations are available via the `impl` parameter:
1. "baseline" (default) - Graph-level fusion using existing vLLM ops
2. "iris" - Iris CCL-based implementation (experimental)
"""

from typing import Optional, Tuple

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


# ============================================================================
# Iris CCL Implementation (Experimental)
# ============================================================================

try:
    import iris
    from iris.ccl import Config
    IRIS_AVAILABLE = True
except ImportError:
    IRIS_AVAILABLE = False
    logger.debug("Iris not available, 'iris' impl will not work")


def _fused_allreduce_add_rms_quant_iris(
    input: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    quant_scale: torch.Tensor,
    quant_dtype: torch.dtype,
    group_name: str,
    residual: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Iris CCL implementation of fused AllReduce + Add + RMSNorm + FP8 Quant.
    
    Uses Iris symmetric memory for all-reduce, then applies RMSNorm and quantization.
    """
    if not IRIS_AVAILABLE:
        raise RuntimeError("Iris not available for 'iris' impl")
    
    M, N = input.shape
    cur_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    
    # Check if capturing - barriers not allowed during capture
    is_capturing = torch.cuda.is_current_stream_capturing()
    
    logger.debug(f"fused_allreduce_add_rms_quant [iris]: rank={cur_rank}, M={M}, N={N}")
    
    # Use large heap like Iris tests (8GB)
    heap_size = 2**33  # 8GB
    
    # Create fresh Iris instance for this call
    shmem = iris.iris(heap_size)
    
    # Allocate input and output tensors on symmetric heap
    iris_input = shmem.zeros((M, N), dtype=input.dtype)
    iris_output = shmem.zeros((M, N), dtype=input.dtype)
    
    # Copy input to symmetric heap tensor
    iris_input.copy_(input)
    
    # Barrier to ensure all ranks have copied input (skip during graph capture)
    if not is_capturing:
        shmem.barrier()
    
    # Use Config with two_shot variant
    config = Config(all_reduce_variant="two_shot")
    
    # All-reduce
    workspace = shmem.ccl.all_reduce_preamble(iris_output, iris_input, config=config)
    if not is_capturing:
        shmem.barrier()
    shmem.ccl.all_reduce(iris_output, iris_input, config=config, workspace=workspace)
    torch.cuda.synchronize()
    
    # Copy result back to regular tensor
    allreduce_out = torch.empty_like(input)
    allreduce_out.copy_(iris_output)
    
    # RMSNorm (with or without residual add)
    if residual is not None:
        # Add residual first
        residual_out = allreduce_out + residual
        # Simple RMSNorm
        variance = (residual_out.float() ** 2).mean(dim=-1, keepdim=True)
        rrms = torch.rsqrt(variance + rms_eps)
        rms_out = (residual_out.float() * rrms * rms_weight.float()).to(input.dtype)
    else:
        residual_out = None
        # Simple RMSNorm
        variance = (allreduce_out.float() ** 2).mean(dim=-1, keepdim=True)
        rrms = torch.rsqrt(variance + rms_eps)
        rms_out = (allreduce_out.float() * rrms * rms_weight.float()).to(input.dtype)
    
    # FP8 Quant - simple per-tensor quantization
    # Note: scale must be float32 for torch._scaled_mm
    abs_max = rms_out.float().abs().max()
    if quant_dtype == torch.float8_e4m3fn:
        fp8_max = 448.0  # max value for e4m3
    else:
        fp8_max = 57344.0  # max value for e5m2
    quant_scale_out = (abs_max / fp8_max).to(torch.float32)
    quant_out = (rms_out.float() / quant_scale_out).to(quant_dtype)
    quant_scale_out = quant_scale_out.view(1)
    
    return allreduce_out, rms_out, residual_out, quant_out, quant_scale_out


# ============================================================================
# Baseline Implementation (Graph-level fusion using vLLM ops)
# ============================================================================


def _fused_allreduce_add_rms_quant_baseline(
    input: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    quant_scale: torch.Tensor,
    quant_dtype: torch.dtype,
    group_name: str,
    residual: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Baseline implementation using existing vLLM/AITER ops.
    
    This is "graph fusion" - 3 separate kernel calls wrapped in one op.
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


# ============================================================================
# Main Entry Point
# ============================================================================


def fused_allreduce_add_rms_quant(
    input: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    quant_scale: torch.Tensor,
    quant_dtype: torch.dtype,
    group_name: str,
    residual: Optional[torch.Tensor] = None,
    impl: str = "iris",
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Fused AllReduce + (optional) Add + RMSNorm + FP8 Per-Tensor Quant.
    
    Args:
        input: Input tensor to all-reduce
        rms_weight: RMSNorm weight
        rms_eps: RMSNorm epsilon
        quant_scale: Quantization scale (can be None for dynamic)
        quant_dtype: Target quantization dtype (e.g., torch.float8_e4m3fn)
        group_name: TP group name for all-reduce
        residual: Optional residual tensor for fused add
        impl: Implementation to use - "baseline" (default) or "iris"
        
    Returns: (allreduce_out, rms_out, residual_out, quant_out, quant_scale_out)
             residual_out is None if residual is None
    """
    if impl == "baseline":
        return _fused_allreduce_add_rms_quant_baseline(
            input, rms_weight, rms_eps, quant_scale, quant_dtype, group_name, residual
        )
    elif impl == "iris":
        return _fused_allreduce_add_rms_quant_iris(
            input, rms_weight, rms_eps, quant_scale, quant_dtype, group_name, residual
        )
    else:
        raise ValueError(f"Unknown impl '{impl}', expected 'baseline' or 'iris'")


# Wrapper implementations for torch custom op registration


def _rocm_aiter_fused_allreduce_rms_quant_impl(
    input: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    quant_scale: torch.Tensor,
    quant_dtype: torch.dtype,
    group_name: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused AllReduce + RMSNorm + FP8 Quant (no residual).
    
    Returns: (allreduce_out, rms_out, quant_out, quant_scale_out)
    """
    allreduce_out, rms_out, _, quant_out, quant_scale_out = fused_allreduce_add_rms_quant(
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused AllReduce + RMSNorm with Add + FP8 Quant (with residual).
    
    Returns: (allreduce_out, rms_out, residual_out, quant_out, quant_scale_out)
    """
    allreduce_out, rms_out, residual_out, quant_out, quant_scale_out = fused_allreduce_add_rms_quant(
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake impl for torch.compile - returns empty tensors with correct shapes."""
    allreduce_out = torch.empty_like(input)
    rms_out = torch.empty_like(input)
    residual_out = torch.empty_like(input)
    quant_out = torch.empty_like(input, dtype=quant_dtype)
    quant_scale_out = torch.empty(1, device=input.device, dtype=torch.float32)
    return allreduce_out, rms_out, residual_out, quant_out, quant_scale_out


# Register as torch custom ops for torch.compile compatibility
try:
    from vllm.utils.torch_utils import direct_register_custom_op
    
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
    logger.warning(f"Failed to register fused allreduce custom ops: {e}")
