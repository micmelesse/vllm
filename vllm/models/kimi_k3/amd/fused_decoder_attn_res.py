# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The attn_res path (VLLM_KIMI_K3_FUSED_DECODER=attn_res): decoder.py's layer
with each AttnRes fused with the all-reduce that feeds it and the RMSNorm after
it. Attention and the MLP leave their outputs unreduced; each AttnRes reduces its
delta in one kernel."""

from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention
from vllm.models.kimi_k3.amd.linear import (
    KimiMLAAttention,
    KimiMLP,
    KimiMoE,
    _apply_attn_res,
)
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.utils.math_utils import cdiv


class KimiDecoderLayerAttnRes(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if config.attn_res_block_size is None:
            raise ValueError("the attn_res path needs a model with AttnRes")
        if get_pp_group().world_size != 1:
            raise NotImplementedError(
                "the attn_res path needs no pipeline split: a stage boundary "
                "passes the reduced sum"
            )
        spec = vllm_config.speculative_config
        if spec is not None and spec.method == "eagle3":
            raise NotImplementedError(
                "EAGLE-3's aux hidden states need reduced layer outputs"
            )
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
                    reduce_results=False,
                )
            else:
                raise NotImplementedError(
                    "Kimi-Linear's delta attention reduces its own output"
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
                reduce_results=False,
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
                reduce_results=False,
            )
            self.mlp = self.block_sparse_moe
            assert self.mlp.experts.moe_config.skip_final_all_reduce, (
                "the MoE ignored reduce_results=False; it would reduce twice"
            )
        else:
            self.mlp = KimiMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                activation_situ_beta=config.activation_situ_beta,
                activation_situ_linear_beta=config.activation_situ_linear_beta,
                reduce_results=False,
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
        # The model's final AttnRes is unfused, so the last layer reduces its output.
        self.is_last_layer = layer_idx == config.num_hidden_layers - 1

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
        # `prefix_delta` is the previous layer's MLP output, unreduced; the first
        # layer has none, and its input (the embedding) is already whole.
        prefix_sum: torch.Tensor | None = hidden_states
        block_write_idx = self.block_write_idx if self.is_block_write_layer else -1
        if prefix_delta is None:
            hidden_states = _apply_attn_res(
                hidden_states,
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.prev_valid_blocks,
                output_norm=self.input_layernorm,
                block_write_idx=block_write_idx,
            )
        else:
            prefix_sum, hidden_states = _fused_attn_res(
                prefix_sum,
                prefix_delta,
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
                self.prev_valid_blocks,
                block_write_idx,
            )

        if self.is_block_write_layer:
            prefix_sum = None

        attn_out = self._run_self_attn(positions, hidden_states)

        mlp_valid_blocks = self.prev_valid_blocks + (
            1 if self.is_block_write_layer else 0
        )
        prefix, hidden_states = _fused_attn_res(
            prefix_sum,
            attn_out,
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
            self.post_attention_layernorm,
            mlp_valid_blocks,
        )

        hidden_states = self.mlp(hidden_states)
        if self.is_last_layer:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        return prefix, block_residual, hidden_states


def _fused_attn_res(
    prefix_sum: torch.Tensor | None,
    delta: torch.Tensor,
    block_residual: torch.Tensor,
    proj: ReplicatedLinear,
    norm: RMSNorm,
    output_norm: RMSNorm,
    num_valid_blocks: int,
    block_write_idx: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AttnRes over `prefix_sum + all_reduce(delta)` in one kernel; a None
    `prefix_sum` is started by the reduced delta. Returns (prefix, output). The op
    raises, naming why, for an input it cannot run."""
    backend = _comm()
    assert backend is not None, "the attn_res path runs only with rocm_comms live"
    return backend.all_reduce_add_attn_res_rms_norm(
        delta,
        prefix_sum,
        block_residual,
        norm.weight,
        proj.weight.squeeze(0),
        output_norm.weight,
        num_valid_blocks,
        block_write_idx,
        norm.variance_epsilon,
        output_norm.variance_epsilon,
    )


def _comm() -> Any | None:
    """The rocm_comms backend, when live with TP above one; None otherwise."""
    if get_tensor_model_parallel_world_size() <= 1:
        return None
    backend = getattr(get_tp_group().device_communicator, "rocm_comm", None)
    return None if backend is None or backend.disabled else backend
