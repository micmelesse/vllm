# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The attn_res path: each AttnRes fused with the all-reduce that feeds it and the
RMSNorm after it (VLLM_KIMI_K3_FUSED_DECODER=attn_res). Attention and the MLP leave
their outputs unreduced; each AttnRes reduces its delta in one kernel."""

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, tensor_model_parallel_all_reduce
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.models.kimi_k3.amd.fused_decoder import comm
from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention
from vllm.models.kimi_k3.amd.linear import (
    KimiDecoderLayer,
    KimiMLAAttention,
    KimiMoE,
    _apply_attn_res,
)
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig


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
    backend = comm()
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


class KimiDecoderLayerAttnRes(KimiDecoderLayer):
    reduce_results = False

    def __init__(
        self, config: KimiLinearConfig, vllm_config: VllmConfig, prefix: str = ""
    ) -> None:
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
        super().__init__(config, vllm_config, prefix)
        # EVERY SUBLAYER MUST HONOR reduce_results, or the fused op reduces twice.
        if not isinstance(self.self_attn, KimiK3DeltaAttention | KimiMLAAttention):
            raise NotImplementedError(
                f"{type(self.self_attn).__name__} reduces its own output"
            )
        if isinstance(self.mlp, KimiMoE):
            assert self.mlp.experts.moe_config.skip_final_all_reduce, (
                "the MoE ignored reduce_results=False; it would reduce twice"
            )
        # The model's final AttnRes is unfused, so the last layer reduces its output.
        self.is_last_layer = self.layer_idx == config.num_hidden_layers - 1

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
