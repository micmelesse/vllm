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

Four implementations are available via VLLM_TRITON_ALLREDUCE_IMPL:
1. "vllm" (default) - Pure torch math with dist.all_reduce (see vllm_allreduce.py)
2. "iris" - Iris CCL-based implementation (see iris_ccl_allreduce.py)
3. "iris_opt" - Iris with inlined two-shot kernel (see iris_opt_allreduce.py)
4. "torch" - Pure torch reference implementation (see torch_allreduce.py)
"""

import os
from typing import Optional, Tuple

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ALLREDUCE_IMPL = os.environ.get("VLLM_TRITON_ALLREDUCE_IMPL", "vllm")
logger.info(f"AllReduce impl: {ALLREDUCE_IMPL}")


def _get_tp_process_group():
    """Resolve the TP group name to a torch.distributed ProcessGroup."""
    from vllm.distributed.parallel_state import get_tp_group
    return get_tp_group().device_group


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
    impl: str = ALLREDUCE_IMPL,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """Fused AllReduce + (optional) Add + RMSNorm + FP8 Per-Tensor Quant.

    Args:
        input: Input tensor to all-reduce
        rms_weight: RMSNorm weight
        rms_eps: RMSNorm epsilon
        quant_scale: Quantization scale (can be None for dynamic)
        quant_dtype: Target quantization dtype (e.g., torch.float8_e4m3fn)
        group_name: TP group name for all-reduce
        residual: Optional residual tensor for fused add
        impl: Implementation to use - "vllm" (default, CUDA graph compatible),
              "iris" (Iris CCL, experimental), "iris_opt" (Iris inlined
              two-shot), or "torch" (pure torch reference)

    Returns: (allreduce_out, rms_out, residual_out, quant_out, quant_scale_out)
             residual_out is None if residual is None
    """
    if impl == "vllm":
        from vllm.vllm_allreduce import fused_allreduce_add_rms_quant_vllm

        return fused_allreduce_add_rms_quant_vllm(
            input, rms_weight, rms_eps, quant_scale, quant_dtype, group_name,
            residual,
        )
    elif impl == "iris":
        from vllm.iris_ccl_allreduce import (
            fused_allreduce_add_rms_quant_iris,
        )

        return fused_allreduce_add_rms_quant_iris(
            input, rms_weight, rms_eps, quant_scale, quant_dtype, group_name,
            residual,
        )
    elif impl == "iris_opt":
        from vllm.iris_opt_allreduce import (
            fused_allreduce_add_rms_quant_iris_opt,
        )

        return fused_allreduce_add_rms_quant_iris_opt(
            input, rms_weight, rms_eps, quant_scale, quant_dtype, group_name,
            residual,
        )
    elif impl == "torch":
        from vllm.torch_allreduce import (
            fused_allreduce_add_rms_quant_torch,
        )

        group = _get_tp_process_group()
        return fused_allreduce_add_rms_quant_torch(
            input, rms_weight, rms_eps, quant_scale, quant_dtype, group,
            residual,
        )
    else:
        raise ValueError(
            f"Unknown impl '{impl}', expected 'vllm', 'iris', 'iris_opt',"
            f" or 'torch'"
        )


# ============================================================================
# Torch custom op registration (for torch.compile pattern matching)
# ============================================================================


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
    allreduce_out, rms_out, _, quant_out, quant_scale_out = (
        fused_allreduce_add_rms_quant(
            input=input,
            rms_weight=rms_weight,
            rms_eps=rms_eps,
            quant_scale=quant_scale,
            quant_dtype=quant_dtype,
            group_name=group_name,
            residual=None,
        )
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
    quant_scale_out = torch.empty(
        1, device=input.device, dtype=torch.float32
    )
    return allreduce_out, rms_out, quant_out, quant_scale_out


def _rocm_aiter_fused_allreduce_add_rms_quant_impl(
    input: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    quant_scale: torch.Tensor,
    quant_dtype: torch.dtype,
    group_name: str,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Fused AllReduce + RMSNorm with Add + FP8 Quant (with residual).

    Returns: (allreduce_out, rms_out, residual_out, quant_out, quant_scale_out)
    """
    allreduce_out, rms_out, residual_out, quant_out, quant_scale_out = (
        fused_allreduce_add_rms_quant(
            input=input,
            rms_weight=rms_weight,
            rms_eps=rms_eps,
            quant_scale=quant_scale,
            quant_dtype=quant_dtype,
            group_name=group_name,
            residual=residual,
        )
    )
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
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Fake impl for torch.compile - returns empty tensors with correct shapes."""
    allreduce_out = torch.empty_like(input)
    rms_out = torch.empty_like(input)
    residual_out = torch.empty_like(input)
    quant_out = torch.empty_like(input, dtype=quant_dtype)
    quant_scale_out = torch.empty(
        1, device=input.device, dtype=torch.float32
    )
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
