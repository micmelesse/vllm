# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
import torch

from tests.conftest import VllmRunner
from tests.utils import create_new_process_for_each_test
from vllm import SamplingParams, TokensPrompt
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="HipCustomCommunicator is ROCm-only"
)

HIDDEN = 7168

TINY_KIMI_K3 = {
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
    "vocab_size": 256,
    "hidden_size": HIDDEN,
    "head_dim": 32,
    "intermediate_size": 512,
    "num_hidden_layers": 4,
    "attn_res_block_size": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "num_experts": None,
    "num_experts_per_token": None,
    "num_shared_experts": 0,
    "q_lora_rank": 128,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "mla_use_nope": True,
    "mla_use_output_gate": True,
    "linear_attn_config": {
        "full_attn_layers": [4],
        "head_dim": 128,
        "kda_layers": [1, 2, 3],
        "num_heads": 8,
        "short_conv_kernel_size": 4,
        "use_full_rank_gate": True,
    },
    "max_position_embeddings": 512,
    "model_max_length": 512,
}


@create_new_process_for_each_test()
@pytest.mark.distributed(num_gpus=2)
def test_kimi_k3_fused_capture_correctness(
    monkeypatch: pytest.MonkeyPatch,
    vllm_runner: type[VllmRunner],
    tmp_path: Path,
):
    """Fused layers capture the same hidden states as unfused ones."""
    from vllm.models.kimi_k3.amd.linear import KimiLinearModel

    if torch.accelerator.device_count() < 2:
        pytest.skip("Not enough GPUs to run the test.")
    model = tmp_path / "tiny-kimi-k3"
    KimiK3Config(
        text_config=TINY_KIMI_K3,
        vision_config={
            "vt_num_attention_heads": 2,
            "vt_num_hidden_layers": 1,
            "vt_hidden_size": 32,
            "vt_intermediate_size": 64,
            "qkv_hidden_size": 48,
        },
        architectures=["KimiK3ForConditionalGeneration"],
    ).save_pretrained(model)
    layers = [1, 2, 3, 4]
    prompts = [
        TokensPrompt(prompt_token_ids=[3 + (i * 7 + n) % 251 for i in range(n)])
        for n in (1, 17, 130)
    ]
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    captured = {}
    for fused in [True, False]:
        monkeypatch.setenv("VLLM_ROCM_USE_HIP_CUSTOM_COMMUNICATOR", str(int(fused)))
        with vllm_runner(
            model_name=str(model),
            skip_tokenizer_init=True,
            load_format="dummy",
            language_model_only=True,
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
            dtype="bfloat16",
            seed=0,
            enforce_eager=True,
            max_model_len=512,
            max_num_seqs=4,
            gpu_memory_utilization=0.5,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            speculative_config={
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {"eagle_aux_hidden_state_layer_ids": layers}
                },
            },
            kv_transfer_config={
                "kv_connector": "ExampleHiddenStatesConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "shared_storage_path": str(tmp_path / f"fused_{fused}")
                },
            },
        ) as runner:
            assert runner.apply_model(
                lambda m: any(
                    k.fused for k in m.modules() if isinstance(k, KimiLinearModel)
                )
            ) == [fused, fused]
            outputs = runner.llm.generate(
                prompts, SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False
            )
        captured[fused] = [
            example_hidden_states_connector.load_hidden_states(
                o.kv_transfer_params["hidden_states_path"]
            )["hidden_states"]
            for o in outputs
        ]

    for got, want in zip(captured[True], captured[False], strict=True):
        assert got.shape == want.shape == (got.shape[0], len(layers), HIDDEN)
        for i, layer in enumerate(layers):
            a, b = got[:, i].float(), want[:, i].float()
            assert torch.isfinite(a).all(), layer
            err = ((a - b).norm() / b.norm()).item()
            assert err <= 1e-2, (layer, err)
