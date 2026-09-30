# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3's decoder layer with its all-reduces fused into the ops that consume
them, chosen over `KimiDecoderLayer` when VLLM_KIMI_K3_FUSED_DECODER is set and the
rocm_comms backend is live. Today it fuses the latent MoE tail's all-reduce with its
RMSNorm; a fusion runs when the backend admits the input, else the all-reduce then
the model's own op. Here and not in a fusion pass: Kimi-K3 is not torch.compiled.
"""

from typing import Any

import torch

import vllm.envs as envs
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.models.kimi_k3.amd.latent_moe_runner import ROCmLatentMoERunner
from vllm.models.kimi_k3.amd.linear import KimiDecoderLayer


def _comm() -> Any | None:
    """The rocm_comms backend, when live with TP above one; None otherwise."""
    if get_tensor_model_parallel_world_size() <= 1:
        return None
    comm = getattr(get_tp_group().device_communicator, "rocm_comm", None)
    return None if comm is None or comm.disabled else comm


def enabled() -> bool:
    """The flag on and the backend live."""
    return envs.VLLM_KIMI_K3_FUSED_DECODER and _comm() is not None


class ROCmLatentMoERunnerFused(ROCmLatentMoERunner):
    """The latent tail's all-reduce and RMSNorm in one kernel, when admitted."""

    def _all_reduce_norm(
        self, fused_output: torch.Tensor, norm: torch.nn.Module | None
    ) -> torch.Tensor:
        comm = _comm()
        if (
            comm is not None
            and isinstance(norm, RMSNorm)
            and comm.should_allreduce_rms_norm(fused_output)
        ):
            return comm.all_reduce_rms_norm(
                fused_output, norm.weight, norm.variance_epsilon
            )
        return super()._all_reduce_norm(fused_output, norm)


class KimiDecoderLayerFused(KimiDecoderLayer):
    latent_runner_cls = ROCmLatentMoERunnerFused
