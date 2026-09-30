# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3's AttnRes fused with the all-reduce that feeds it, when
VLLM_KIMI_K3_FUSED_ATTN_RES is set and the rocm_comms backend is live. Fused when the
backend admits the input, else the all-reduce then the model's own AttnRes. Here and not
in a fusion pass: Kimi-K3 is not torch.compiled.
"""

from typing import Any

import torch

import vllm.envs as envs
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.models.kimi_k3.amd.ops.attn_res import attn_res as triton_attn_res


def _comm() -> Any | None:
    """The rocm_comms backend, when live with TP above one; None otherwise."""
    if get_tensor_model_parallel_world_size() <= 1:
        return None
    comm = getattr(get_tp_group().device_communicator, "rocm_comm", None)
    return None if comm is None or comm.disabled else comm


def fused_attn_res() -> bool:
    """Whether a layer leaves its outputs unreduced for `attn_res` to reduce: the flag
    on, the backend live, and no pipeline split, whose stage boundary needs the sum."""
    return (
        envs.VLLM_KIMI_K3_FUSED_ATTN_RES
        and get_pp_group().world_size == 1
        and _comm() is not None
    )


def attn_res(
    prefix_sum: torch.Tensor | None,
    block_residual: torch.Tensor,
    proj: ReplicatedLinear,
    norm: RMSNorm,
    num_valid_blocks: int,
    *,
    delta: torch.Tensor | None = None,
    output_norm: RMSNorm | None = None,
    block_write_idx: int = -1,
    reduce_delta: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AttnRes over `prefix_sum + delta`, `delta` all-reduced first when `reduce_delta`.
    A None `prefix_sum` means `delta` starts the prefix. Returns (prefix, output)."""
    weights = (
        norm.weight,
        proj.weight.squeeze(0),
        None if output_norm is None else output_norm.weight,
    )
    eps = norm.variance_epsilon
    out_eps = 0.0 if output_norm is None else output_norm.variance_epsilon
    if reduce_delta and delta is not None:
        comm = _comm()
        if comm is not None and comm.should_allreduce_add_attn_res_rms_norm(delta):
            return comm.all_reduce_add_attn_res_rms_norm(
                delta,
                prefix_sum,
                block_residual,
                *weights,
                num_valid_blocks,
                block_write_idx,
                eps,
                out_eps,
            )
        delta = tensor_model_parallel_all_reduce(delta)
    if prefix_sum is None:
        assert delta is not None, "a new prefix needs a delta to start it"
        prefix_sum, delta = delta, None
    out = triton_attn_res(
        prefix_sum,
        delta,
        block_residual,
        *weights,
        num_valid_blocks,
        block_write_idx,
        eps,
        out_eps,
    )
    return prefix_sum, out
