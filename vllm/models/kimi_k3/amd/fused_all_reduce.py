# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3's all-reduces fused with the ops that consume them, when the rocm_comms
backend is live. Every decision is here; the model calls these and nothing else.

Each op is fused when the backend admits the input, else the model's own ops in the
same order. Here and not in a fusion pass: Kimi-K3 is not torch.compiled.
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


def fusion_enabled() -> bool:
    """Whether a layer should leave its outputs unreduced for `attn_res` to reduce:
    the backend live and no pipeline split, whose stage boundary needs the sum."""
    return get_pp_group().world_size == 1 and _comm() is not None


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


def latent_tail(
    fused_output: torch.Tensor,
    shared_output: torch.Tensor,
    norm: RMSNorm | None,
    up_proj_shard: torch.Tensor,
    shard_start: int,
) -> None:
    """`shared_output[:, shard] += norm(all_reduce(fused_output)) @ up_proj_shard.T`:
    the whole tail in one kernel, else the all-reduce and norm in one, else each op."""
    comm = _comm()
    if (
        comm is not None
        and norm is not None
        and norm.weight.dtype == fused_output.dtype
        and comm.should_allreduce_rms_norm_gemm_add(fused_output, up_proj_shard)
    ):
        comm.all_reduce_rms_norm_gemm_add(
            fused_output,
            norm.weight,
            norm.variance_epsilon,
            up_proj_shard,
            shared_output,
            shard_start,
        )
        return
    if (
        comm is not None
        and norm is not None
        and comm.should_allreduce_rms_norm(fused_output)
    ):
        latent = comm.all_reduce_rms_norm(
            fused_output, norm.weight, norm.variance_epsilon
        )
    else:
        latent = tensor_model_parallel_all_reduce(fused_output)
        if norm is not None:
            latent = norm(latent)
    hidden_shard = shared_output.narrow(-1, shard_start, up_proj_shard.shape[0])
    # hidden_shard += latent @ up_proj_shard.T, accumulated in the GEMM's
    # beta-add epilogue so folding in the shared partial costs no kernel.
    hidden_shard.addmm_(latent, up_proj_shard.t())


def moe_tail_one_all_reduce() -> bool:
    """Whether the latent-MoE tail runs as one all-reduce (`latent_tail_one_all_reduce`)
    rather than two (`latent_tail`, then the output's)."""
    return envs.VLLM_KIMI_K3_MOE_TAIL == "one_all_reduce"


def latent_tail_one_all_reduce(
    fused_output: torch.Tensor,
    shared_output: torch.Tensor,
    norm: RMSNorm | None,
    up_proj: torch.Tensor,
) -> torch.Tensor:
    """`all_reduce(shared_output + up_proj(norm(all_reduce(fused_output))))` in ONE
    all-reduce, returned reduced. `up_proj` is the whole [hidden, latent] weight.

    RMSNorm's only nonlinear part is a per-row scale, 1 / rms(L) with L the reduced
    latent, so it moves past the GEMM: up(norm(L)) = ((g * L) @ W^T) / rms(L), and the
    GEMM is linear, so (g * L) @ W^T = sum over ranks of (g * l_r) @ W^T. Each rank
    projects its own partial latent l_r with the WHOLE up-projection (it is replicated),
    one all-reduce carries [shared partial | projected partial | latent partial], and
    out = S + P / rms(L) per row. The latent travels because rms(L) needs the true sum;
    the ranks' partial sums of squares miss the cross terms."""
    hidden = shared_output.shape[-1]
    if norm is None:
        return tensor_model_parallel_all_reduce(
            torch.addmm(shared_output, fused_output, up_proj.t())
        )
    projected = (fused_output * norm.weight.to(fused_output.dtype)) @ up_proj.t()
    reduced = tensor_model_parallel_all_reduce(
        torch.cat([shared_output, projected, fused_output], dim=-1)
    )
    shared, proj, latent = reduced.split(
        [hidden, hidden, fused_output.shape[-1]], dim=-1
    )
    inv_rms = torch.rsqrt(
        latent.float().pow(2).mean(dim=-1, keepdim=True) + norm.variance_epsilon
    )
    return (shared.float() + proj.float() * inv_rms).to(shared_output.dtype)
