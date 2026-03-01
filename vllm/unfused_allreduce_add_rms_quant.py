# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unfused vLLM production path: 4 separate kernel launches.

all_reduce -> aiter RMSNorm -> aiter FP8 per-tensor quant -> scaled_mm.

This is the exact sequence that exists in the model graph before the fusion
pass runs. Used as the baseline for benchmarking fused implementations and
as the reference for correctness tests.
"""

from typing import Optional

import torch

__all__ = ["unfused_allreduce_add_rms_quant_gemm"]


def unfused_allreduce_add_rms_quant_gemm(
    input: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    quant_scale: torch.Tensor,
    quant_dtype: torch.dtype,
    group_name: str,
    gemm_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    residual: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Unfused AllReduce + (optional) Add + RMSNorm + FP8 Quant + scaled_mm.

    Four separate kernel launches using the same vLLM ops that the
    production model graph contains before the fusion pass runs.

    Returns (gemm_out, residual_out). residual_out is None when residual
    is None.
    """
    ar_out = torch.ops.vllm.all_reduce(input, group_name=group_name)

    if residual is not None:
        rms_out, res_out = torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add(
            ar_out, residual, rms_weight, rms_eps)
    else:
        rms_out = torch.ops.vllm.rocm_aiter_rms_norm(
            ar_out, rms_weight, rms_eps)
        res_out = None

    q_out, qs_out = torch.ops.vllm.rocm_aiter_per_tensor_quant(
        rms_out, quant_dtype, quant_scale)

    gemm_out = torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl(
        q_out, gemm_weight, out_dtype, qs_out, weight_scale, bias,
    )

    return gemm_out, res_out
