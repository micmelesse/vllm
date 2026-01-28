# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused AllReduce + RMSNorm for ROCm using Iris.

This module provides a fused operation that combines:
1. All-reduce across tensor parallel GPUs (via Iris symmetric memory)
2. Residual addition
3. RMS normalization

The fusion reduces memory bandwidth by avoiding intermediate writes.

Current status:
    - Reference implementation (ccl_baseline) using Iris CCL works in eager mode
    - CUDA graph capture is NOT supported (Iris barriers forbidden during capture)
    - This is currently enabled via VLLM_ROCM_ALLREDUCE=1 env var
    
TODO: Integrate with vLLM's fusion pass system instead of env var switching.
      See AllReduceFusionPass in vllm/compilation/collective_fusion.py for reference.
"""

import iris
import torch
from iris.ccl import Config

from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm

logger = init_logger(__name__)


# ============================================================================
# CCL Baseline Implementation
# ============================================================================
# This follows the exact pattern from the Iris CCL example:
# https://github.com/micmelesse/iris/blob/main/iris/ccl/all_reduce.py
#
# Key points:
# - Allocates Iris context and buffers per-call (not cached)
# - Calls all_reduce_preamble() and all_reduce() with same exact buffers
# - Works correctly for any M, N size
# - Not optimized for CUDA graph capture (but works for eager mode)


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
# Main Implementation Dispatch
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
except Exception as e:
    logger.warning(f"Failed to register triton_allreduce custom op: {e}")


# ============================================================================
# Public API
# ============================================================================


def triton_allreduce(
    input_: torch.Tensor,
    residual: torch.Tensor | None = None,
    norm: RMSNorm | None = None,
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
