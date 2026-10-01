# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The one_ar path (VLLM_KIMI_K3_FUSED_DECODER=one_ar): decoder.py's layer with
the latent MoE tail's two all-reduces as one."""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention as KimiLinearGatedDeltaNetAttention,
)
from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention
from vllm.models.kimi_k3.amd.latent_moe_runner import ROCmLatentMoERunner
from vllm.models.kimi_k3.amd.linear import (
    KimiMLAAttention,
    KimiMLP,
    KimiMoE,
    _apply_attn_res,
)
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.utils.math_utils import cdiv


class KimiDecoderLayerOneAR(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = int(prefix.rsplit(".", 1)[1])

        self.is_moe = config.is_moe
        layer_idx = self.layer_idx
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        if config.is_kda_layer(layer_idx):
            # Kimi-K3 sets use_full_rank_gate and uses the ROCm-specific K3 KDA
            # layer; Kimi-Linear keeps the shared low-rank-gate implementation.
            kda_config = config.linear_attn_config
            assert kda_config is not None
            if kda_config.get("use_full_rank_gate", False):
                self.self_attn = KimiK3DeltaAttention(
                    config,
                    vllm_config,
                    prefix=f"{prefix}.self_attn",
                )
            else:
                self.self_attn = KimiLinearGatedDeltaNetAttention(
                    config,
                    vllm_config,
                    prefix=f"{prefix}.self_attn",
                )
        else:
            qk_nope_head_dim = config.qk_nope_head_dim
            qk_rope_head_dim = config.qk_rope_head_dim
            v_head_dim = config.v_head_dim
            kv_lora_rank = config.kv_lora_rank
            mla_use_nope = config.mla_use_nope
            assert qk_nope_head_dim is not None
            assert qk_rope_head_dim is not None
            assert v_head_dim is not None
            assert kv_lora_rank is not None
            assert mla_use_nope is not None
            self.self_attn = KimiMLAAttention(
                layer_idx=layer_idx,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                quant_config=quant_config,
                cache_config=cache_config,
                model_config=model_config,
                prefix=f"{prefix}.self_attn",
                config=config,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                use_nope=mla_use_nope,
            )

        if (
            self.is_moe
            and config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.block_sparse_moe = KimiMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe",
                layer_idx=layer_idx,
                latent_runner_cls=ROCmLatentMoERunnerOneAR,
            )
            self.mlp = self.block_sparse_moe
        else:
            self.mlp = KimiMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                activation_situ_beta=config.activation_situ_beta,
                activation_situ_linear_beta=config.activation_situ_linear_beta,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        attn_res_block_size = config.attn_res_block_size
        self.use_attn_residuals = attn_res_block_size is not None
        if attn_res_block_size is not None:
            self.attn_res_block_size = attn_res_block_size
            self.is_block_write_layer = layer_idx % self.attn_res_block_size == 0
            self.block_write_idx = layer_idx // self.attn_res_block_size
            self.prev_valid_blocks = cdiv(layer_idx, self.attn_res_block_size)
            self.self_attention_res_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.mlp_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attention_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.self_attention_res_proj",
            )
            self.mlp_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.mlp_res_proj",
            )

    def _run_self_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        prefix_delta: torch.Tensor | None = None,
        **kwargs,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        if self.use_attn_residuals:
            assert residual is not None
            return self.forward_attn_residual(
                positions, hidden_states, residual, prefix_delta
            )

        assert prefix_delta is None
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self._run_self_attn(positions, hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def forward_attn_residual(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        prefix_delta: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix_sum = hidden_states
        hidden_states = _apply_attn_res(
            prefix_sum,
            block_residual,
            self.self_attention_res_proj,
            self.self_attention_res_norm,
            self.prev_valid_blocks,
            delta=prefix_delta,
            output_norm=self.input_layernorm,
            block_write_idx=(self.block_write_idx if self.is_block_write_layer else -1),
        )

        if self.is_block_write_layer:
            prefix_sum = None

        hidden_states = self._run_self_attn(positions, hidden_states)

        if prefix_sum is None:
            prefix_sum = hidden_states
            prefix_delta = None
        else:
            prefix_delta = hidden_states

        mlp_valid_blocks = self.prev_valid_blocks + (
            1 if self.is_block_write_layer else 0
        )
        hidden_states = _apply_attn_res(
            prefix_sum,
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
            mlp_valid_blocks,
            delta=prefix_delta,
            output_norm=self.post_attention_layernorm,
        )

        hidden_states = self.mlp(hidden_states)
        return prefix_sum, block_residual, hidden_states


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
        backend = _comm()
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


def _comm() -> Any | None:
    """The rocm_comms backend, when live with TP above one; None otherwise."""
    if get_tensor_model_parallel_world_size() <= 1:
        return None
    backend = getattr(get_tp_group().device_communicator, "rocm_comm", None)
    return None if backend is None or backend.disabled else backend
