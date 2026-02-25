# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unfused vLLM production path: 3 separate kernel launches.

all_reduce -> aiter RMSNorm -> aiter FP8 per-tensor quant.

This is the exact sequence that exists in the model graph before the fusion
pass runs. Used as the baseline for benchmarking fused implementations and
as the reference for correctness tests.
"""

from typing import Optional, Tuple

import torch

__all__ = ["unfused_allreduce_add_rms_quant"]


def unfused_allreduce_add_rms_quant(
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
    """Unfused AllReduce + (optional) Add + RMSNorm + FP8 Per-Tensor Quant.

    Three separate kernel launches using the same vLLM ops that the
    production model graph contains before the fusion pass runs.
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

    return ar_out, rms_out, res_out, q_out, qs_out
