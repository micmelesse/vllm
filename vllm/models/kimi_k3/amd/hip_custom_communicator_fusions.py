# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3's all-reduces fused into the ops that consume them, when
HipCustomCommunicator owns the TP all-reduce: the MoE latent's all-reduce with
its RMSNorm (`all_reduce_add_rms_norm`, no residual, in the latent tail), and each
AttnRes with the all-reduce of the delta it adds
(`all_reduce_add_attn_res_rms_norm`).
Attention and the MLP then leave their outputs unreduced; one all-reduce at each
point, as before. `KimiLinearModel` decides once (`fuse_all_reduces`) and its
decoder layers take these paths; the MTP block's layer never does."""

import torch

from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.distributed.device_communicators.hip_custom_communicator import (
    HipCustomCommunicator,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.models.kimi_k3.amd.latent_moe_runner import ROCmLatentMoERunner
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

logger = init_logger(__name__)


def fuse_all_reduces(config: KimiLinearConfig, vllm_config: VllmConfig) -> bool:
    """Whether the decoder layers fuse their all-reduces: HipCustomCommunicator live
    with TP above one, on a model with AttnRes and the K3 delta attention, served
    without a pipeline split (a stage boundary passes the reduced sum) or EAGLE-3 (its
    aux hidden states need reduced layer outputs). Otherwise the standard layers run."""
    if _communicator() is None:
        return False
    kda_config = config.linear_attn_config
    why = (
        "the model has no AttnRes"
        if config.attn_res_block_size is None
        else "Kimi-Linear's delta attention reduces its own output"
        if kda_config is not None and not kda_config.get("use_full_rank_gate", False)
        else "a pipeline split passes the reduced sum"
        if get_pp_group().world_size != 1
        else "EAGLE-3's aux hidden states need reduced layer outputs"
        if vllm_config.speculative_config is not None
        and vllm_config.speculative_config.method == "eagle3"
        else None
    )
    if why is not None:
        logger.info("Kimi-K3 keeps its all-reduces unfused: %s", why)
        return False
    return True


class ROCmLatentMoERunnerAllReduceRmsNorm(ROCmLatentMoERunner):
    """The latent tail with its all-reduce and RMSNorm in one kernel, always: the
    fused op runs or the call fails, never the unfused ops. Otherwise the tail as
    `ROCmLatentMoERunner` runs it."""

    def _shard_up_proj_tail(
        self,
        fused_output: torch.Tensor,
        shared_output: torch.Tensor,
        trunc_size: int | None,
    ) -> torch.Tensor:
        backend = _communicator()
        assert backend is not None, (
            "the all_reduce_add_rms_norm path runs only with HipCustomCommunicator live"
        )
        transform = self.routed_output_transform
        assert transform is not None
        if not isinstance(transform.norm, RMSNorm):
            raise RuntimeError(
                f"the fused latent tail needs an RMSNorm: {transform.norm!r}"
            )
        # The op raises, naming why, for an input it cannot run.
        latent = backend.all_reduce_add_rms_norm(
            fused_output, transform.norm.weight, transform.norm.variance_epsilon
        )
        assert isinstance(latent, torch.Tensor)

        shard_size = self._up_proj_shard_size
        shard_start = get_tensor_model_parallel_rank() * shard_size
        weight = transform.up_proj.weight
        up_proj_shard = (
            weight
            if self._up_proj_preshard
            else weight.narrow(0, shard_start, shard_size)
        )
        hidden_shard = shared_output.narrow(-1, shard_start, shard_size)
        # hidden_shard += latent @ up_proj_shard.T, in the GEMM's beta-add epilogue.
        hidden_shard.addmm_(latent, up_proj_shard.t())

        return self._maybe_reduce_final_output(
            shared_output, trunc_size, output_is_reduced=False
        )


def fused_attn_res(
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
    backend = _communicator()
    assert backend is not None, (
        "the all_reduce_add_attn_res_rms_norm path runs only with "
        "HipCustomCommunicator live"
    )
    prefix, out = backend.all_reduce_add_attn_res_rms_norm(
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
    return prefix, out


def _communicator() -> HipCustomCommunicator | None:
    """HipCustomCommunicator, when live with TP above one; None otherwise."""
    if get_tensor_model_parallel_world_size() <= 1:
        return None
    backend = getattr(
        get_tp_group().device_communicator, "hip_custom_communicator", None
    )
    return None if backend is None or backend.disabled else backend
