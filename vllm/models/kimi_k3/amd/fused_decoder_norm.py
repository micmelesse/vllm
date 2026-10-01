# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The norm path: the ladder grown from the all-reduce, starting with the latent MoE
tail's all-reduce and RMSNorm in one kernel (VLLM_KIMI_K3_FUSED_DECODER=norm)."""

import torch

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.models.kimi_k3.amd.fused_decoder import comm
from vllm.models.kimi_k3.amd.latent_moe_runner import ROCmLatentMoERunner
from vllm.models.kimi_k3.amd.linear import KimiDecoderLayer


class ROCmLatentMoERunnerNorm(ROCmLatentMoERunner):
    """The latent tail with its all-reduce and RMSNorm in one kernel, always: the
    fused op runs or the call fails, never the unfused ops. Otherwise the tail as
    `ROCmLatentMoERunner` runs it."""

    def _shard_up_proj_tail(
        self,
        fused_output: torch.Tensor,
        shared_output: torch.Tensor,
        trunc_size: int | None,
    ) -> torch.Tensor:
        backend = comm()
        assert backend is not None, "the norm path runs only with rocm_comms live"
        transform = self.routed_output_transform
        assert transform is not None
        if not isinstance(transform.norm, RMSNorm):
            raise RuntimeError(
                f"the fused latent tail needs an RMSNorm: {transform.norm!r}"
            )
        # The op raises, naming why, for an input it cannot run.
        latent = backend.all_reduce_rms_norm(
            fused_output, transform.norm.weight, transform.norm.variance_epsilon
        )

        shard_size = self._up_proj_shard_size
        shard_start = get_tensor_model_parallel_rank() * shard_size
        up_proj_shard = transform.up_proj.weight.narrow(0, shard_start, shard_size)
        hidden_shard = shared_output.narrow(-1, shard_start, shard_size)
        # hidden_shard += latent @ up_proj_shard.T, in the GEMM's beta-add epilogue.
        hidden_shard.addmm_(latent, up_proj_shard.t())

        return self._maybe_reduce_final_output(
            shared_output, trunc_size, output_is_reduced=False
        )


class KimiDecoderLayerNorm(KimiDecoderLayer):
    latent_runner_cls = ROCmLatentMoERunnerNorm
