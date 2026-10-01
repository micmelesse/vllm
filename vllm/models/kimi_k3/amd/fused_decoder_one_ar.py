# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The one_ar path: the latent MoE tail's two all-reduces as one
(VLLM_KIMI_K3_FUSED_DECODER=one_ar)."""

from collections.abc import Callable

import torch

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.models.kimi_k3.amd.fused_decoder import comm
from vllm.models.kimi_k3.amd.latent_moe_runner import ROCmLatentMoERunner
from vllm.models.kimi_k3.amd.linear import KimiDecoderLayer


def _one_all_reduce_tail(
    reduce: Callable[[torch.Tensor], torch.Tensor],
    fused_output: torch.Tensor,
    shared_output: torch.Tensor,
    norm: RMSNorm | None,
    up_proj: torch.Tensor,
) -> torch.Tensor:
    """`all_reduce(shared + up_proj(norm(all_reduce(fused))))` in ONE all-reduce.

    RMSNorm's only nonlinear part is a per-row scale, 1 / rms(L) with L the reduced
    latent, so it moves past the GEMM: up(norm(L)) = ((g * L) @ W^T) / rms(L), and
    (g * L) @ W^T is the sum over ranks of (g * l_r) @ W^T. Each rank projects its
    own partial latent with the WHOLE up-projection, one all-reduce carries
    [shared | projected | latent], and out = S + P / rms(L) per row; the latent
    travels because rms(L) needs the true sum."""
    if norm is None:
        return reduce(torch.addmm(shared_output, fused_output, up_proj.t()))
    hidden = shared_output.shape[-1]
    projected = (fused_output * norm.weight.to(fused_output.dtype)) @ up_proj.t()
    reduced = reduce(torch.cat([shared_output, projected, fused_output], dim=-1))
    shared, proj, latent = reduced.split(
        [hidden, hidden, fused_output.shape[-1]], dim=-1
    )
    inv_rms = torch.rsqrt(
        latent.float().pow(2).mean(dim=-1, keepdim=True) + norm.variance_epsilon
    )
    return (shared.float() + proj.float() * inv_rms).to(shared_output.dtype)


class ROCmLatentMoERunnerOneAR(ROCmLatentMoERunner):
    """The latent tail as one all-reduce on rocm_comms. A batch whose row the
    backend cannot hold (a large prefill: the row is 2.5 hidden rows) takes the two
    all-reduces."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not self._tail_shardable:
            raise NotImplementedError(
                "the one_ar path needs the latent tail's shared experts and "
                "up-projection, no sequence parallelism and a unit routed scale"
            )
        assert not self.moe_config.skip_final_all_reduce, (
            "the one_ar tail reduces the whole output; nothing may skip it"
        )

    def _shard_up_proj_tail(
        self,
        fused_output: torch.Tensor,
        shared_output: torch.Tensor,
        trunc_size: int | None,
    ) -> torch.Tensor:
        backend = comm()
        assert backend is not None, "the one_ar path runs only with rocm_comms live"
        width = 2 * shared_output.shape[-1] + fused_output.shape[-1]
        row = fused_output.new_empty(fused_output.shape[0], width)
        if not backend.should_allreduce(row):
            return super()._shard_up_proj_tail(fused_output, shared_output, trunc_size)
        transform = self.routed_output_transform
        assert transform is not None
        out = _one_all_reduce_tail(
            backend.all_reduce,
            fused_output,
            shared_output,
            transform.norm,
            transform.up_proj.weight,
        )
        return out[..., :trunc_size] if trunc_size is not None else out


class KimiDecoderLayerOneAR(KimiDecoderLayer):
    latent_runner_cls = ROCmLatentMoERunnerOneAR
