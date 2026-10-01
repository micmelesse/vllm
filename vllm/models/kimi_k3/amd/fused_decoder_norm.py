# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The norm path: the ladder grown from the all-reduce, starting with the latent MoE
tail's all-reduce and RMSNorm in one kernel (VLLM_KIMI_K3_FUSED_DECODER=norm)."""

import torch

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.models.kimi_k3.amd.fused_decoder import comm as _comm
from vllm.models.kimi_k3.amd.latent_moe_runner import ROCmLatentMoERunner
from vllm.models.kimi_k3.amd.linear import KimiDecoderLayer


class ROCmLatentMoERunnerNorm(ROCmLatentMoERunner):
    """The latent tail's all-reduce and RMSNorm in one kernel, always: with the
    fused decoder on, the fused op runs or the call fails, never the unfused ops."""

    def _all_reduce_norm(
        self, fused_output: torch.Tensor, norm: torch.nn.Module | None
    ) -> torch.Tensor:
        comm = _comm()
        assert comm is not None, "the fused decoder runs only with rocm_comms live"
        if not isinstance(norm, RMSNorm):
            raise RuntimeError(f"the fused latent tail needs an RMSNorm: {norm!r}")
        # The op raises, naming why, for an input it cannot run.
        return comm.all_reduce_rms_norm(
            fused_output, norm.weight, norm.variance_epsilon
        )


class KimiDecoderLayerNorm(KimiDecoderLayer):
    latent_runner_cls = ROCmLatentMoERunnerNorm
