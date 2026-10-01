# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3's fused decoder layers: one class per fusion path, each in its own
file, chosen once by VLLM_KIMI_K3_FUSED_DECODER. A path never builds on another's;
one that combines two is a class of its own. A chosen path runs its fused ops or
raises, never the unfused ops. Not a fusion pass: Kimi-K3 is not torch.compiled.
"""

from typing import Any

import vllm.envs as envs
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.models.kimi_k3.amd.linear import KimiDecoderLayer


def comm() -> Any | None:
    """The rocm_comms backend, when live with TP above one; None otherwise."""
    if get_tensor_model_parallel_world_size() <= 1:
        return None
    backend = getattr(get_tp_group().device_communicator, "rocm_comm", None)
    return None if backend is None or backend.disabled else backend


def layer_class() -> type[KimiDecoderLayer]:
    """The decoder layer VLLM_KIMI_K3_FUSED_DECODER names."""
    path = envs.VLLM_KIMI_K3_FUSED_DECODER
    if path == "none":
        return KimiDecoderLayer
    if comm() is None:
        raise RuntimeError(
            f"VLLM_KIMI_K3_FUSED_DECODER={path} needs the rocm_comms backend live "
            "with TP above one"
        )
    if path == "norm":
        from vllm.models.kimi_k3.amd.fused_decoder_norm import KimiDecoderLayerNorm

        return KimiDecoderLayerNorm
    if path == "attn_res":
        from vllm.models.kimi_k3.amd.fused_decoder_attn_res import (
            KimiDecoderLayerAttnRes,
        )

        return KimiDecoderLayerAttnRes
    if path == "one_ar":
        from vllm.models.kimi_k3.amd.fused_decoder_one_ar import KimiDecoderLayerOneAR

        return KimiDecoderLayerOneAR
    raise ValueError(f"VLLM_KIMI_K3_FUSED_DECODER={path} names no fused decoder")
