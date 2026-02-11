# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM all-reduce implementation (default).

Graph-level fusion using existing vLLM/AITER ops. Three separate kernel
calls (all_reduce, RMSNorm, per_tensor_quant) wrapped in one custom op
for the compiler to pattern-match and fuse.

Compatible with CUDA graph capture.
"""

from typing import Optional, Tuple

import torch

__all__ = ["fused_allreduce_add_rms_quant_vllm"]


def fused_allreduce_add_rms_quant_vllm(
    input: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    quant_scale: torch.Tensor,
    quant_dtype: torch.dtype,
    group_name: str,
    residual: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """vLLM implementation using existing AITER ops.

    This is "graph fusion" - 3 separate kernel calls wrapped in one op.
    The compiler pass replaces the unfused pattern with this fused op.
    """
    # Step 1: All-reduce
    allreduce_out = torch.ops.vllm.all_reduce(input, group_name=group_name)

    # Step 2: RMSNorm (with or without residual add)
    if residual is not None:
        rms_out, residual_out = (
            torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add(
                allreduce_out, residual, rms_weight, rms_eps
            )
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
