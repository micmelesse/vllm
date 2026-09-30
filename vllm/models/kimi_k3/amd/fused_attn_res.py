# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3's decoder layer with each AttnRes fused with the all-reduce that feeds it,
chosen over `KimiDecoderLayer` when VLLM_KIMI_K3_FUSED_ATTN_RES is set and the
rocm_comms backend is live. Attention and the MLP leave their outputs unreduced; each
AttnRes reduces its delta, in one kernel when the backend admits it, else the
all-reduce then the model's own AttnRes. Here and not in a fusion pass: Kimi-K3 is not
torch.compiled.
"""

from typing import Any

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.models.kimi_k3.amd.linear import KimiDecoderLayer, KimiMoE
from vllm.models.kimi_k3.amd.ops.attn_res import attn_res as triton_attn_res
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig


def _comm() -> Any | None:
    """The rocm_comms backend, when live with TP above one; None otherwise."""
    if get_tensor_model_parallel_world_size() <= 1:
        return None
    comm = getattr(get_tp_group().device_communicator, "rocm_comm", None)
    return None if comm is None or comm.disabled else comm


def enabled(config: KimiLinearConfig) -> bool:
    """The flag on, a model with AttnRes, the backend live, and no pipeline split
    (a stage boundary needs the reduced sum)."""
    return (
        envs.VLLM_KIMI_K3_FUSED_ATTN_RES
        and config.attn_res_block_size is not None
        and get_pp_group().world_size == 1
        and _comm() is not None
    )


def _attn_res(
    prefix_sum: torch.Tensor | None,
    delta: torch.Tensor,
    block_residual: torch.Tensor,
    proj: ReplicatedLinear,
    norm: RMSNorm,
    output_norm: RMSNorm,
    num_valid_blocks: int,
    block_write_idx: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AttnRes over `prefix_sum + all_reduce(delta)`; a None `prefix_sum` is started by
    the reduced delta. Returns (prefix, output)."""
    args = (
        block_residual,
        norm.weight,
        proj.weight.squeeze(0),
        output_norm.weight,
        num_valid_blocks,
        block_write_idx,
        norm.variance_epsilon,
        output_norm.variance_epsilon,
    )
    comm = _comm()
    if comm is not None and comm.should_allreduce_add_attn_res_rms_norm(delta):
        return comm.all_reduce_add_attn_res_rms_norm(delta, prefix_sum, *args)
    delta = tensor_model_parallel_all_reduce(delta)
    if prefix_sum is None:
        return delta, triton_attn_res(delta, None, *args)
    return prefix_sum, triton_attn_res(prefix_sum, delta, *args)


class KimiDecoderLayerFusedAttnRes(KimiDecoderLayer):
    reduce_results = False

    def __init__(
        self, config: KimiLinearConfig, vllm_config: VllmConfig, prefix: str = ""
    ) -> None:
        spec = vllm_config.speculative_config
        if spec is not None and spec.method == "eagle3":
            raise NotImplementedError(
                "EAGLE-3's aux hidden states need reduced layer outputs; unset "
                "VLLM_KIMI_K3_FUSED_ATTN_RES"
            )
        super().__init__(config, vllm_config, prefix)
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
        # layer has none.
        prefix_sum = hidden_states
        block_write_idx = self.block_write_idx if self.is_block_write_layer else -1
        if prefix_delta is None:
            hidden_states = triton_attn_res(
                prefix_sum,
                None,
                block_residual,
                self.self_attention_res_norm.weight,
                self.self_attention_res_proj.weight.squeeze(0),
                self.input_layernorm.weight,
                self.prev_valid_blocks,
                block_write_idx,
                self.self_attention_res_norm.variance_epsilon,
                self.input_layernorm.variance_epsilon,
            )
        else:
            prefix_sum, hidden_states = _attn_res(
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
        prefix_sum, hidden_states = _attn_res(
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
        return prefix_sum, block_residual, hidden_states
