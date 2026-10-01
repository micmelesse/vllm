# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which fused decoder layer VLLM_KIMI_K3_FUSED_DECODER names. Each path is its own
file, a copy of decoder.py's layer with that path's change, sharing nothing with
the others; a path that combines two is a file of its own. A chosen path runs its
fused ops or raises, never the unfused ops. Not a fusion pass: Kimi-K3 is not
torch.compiled.
"""

from torch import nn

from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group


def layer_class(path: str) -> type[nn.Module]:
    """The decoder layer of fusion path `path`."""
    backend = (
        getattr(get_tp_group().device_communicator, "rocm_comm", None)
        if get_tensor_model_parallel_world_size() > 1
        else None
    )
    if backend is None or backend.disabled:
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
