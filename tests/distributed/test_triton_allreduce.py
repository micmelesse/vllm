# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import ray
import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_tp_group, graph_capture
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.triton_allreduce import triton_allreduce
from vllm.platforms import current_platform

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

random.seed(42)
# (M, N) pairs representing (num_tokens, hidden_dim)
test_sizes = [(32, 4096), (128, 4096), (32, 8192)]


@ray.remote(num_gpus=1, max_calls=1)
def graph_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group().device_group

        # A small all_reduce for warmup.
        # this is needed because device communicators might be created lazily
        # (e.g. NCCL). This will ensure that the communicator is initialized
        # before any communication happens, so that this group can be used for
        # graph capture immediately.
        data = torch.zeros(1)
        data = data.to(device=device)
        torch.distributed.all_reduce(data, group=group)
        torch.cuda.synchronize()
        del data

        for M, N in test_sizes:
            for dtype in [torch.float16, torch.bfloat16]:
                with graph_capture(device=device) as graph_capture_context:
                    # use integers so result matches NCCL exactly
                    inp = torch.randint(
                        1, 16, (M, N), dtype=dtype, device=torch.cuda.current_device()
                    )
                    residual = torch.randint(
                        1, 16, (M, N), dtype=dtype, device=torch.cuda.current_device()
                    )
                    norm = RMSNorm(N, eps=1e-5).to(device=device, dtype=dtype)
                    max_m = M * 2

                    # compute reference: all_reduce(inp) + residual
                    ref_inp = inp.clone()
                    dist.all_reduce(ref_inp, group=group)
                    ref_res_out = ref_inp + residual

                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                        # triton fused all_reduce + residual + rmsnorm
                        out, res_out = triton_allreduce(inp, residual, norm, max_m)
                graph.replay()
                torch.testing.assert_close(res_out, ref_res_out, rtol=5e-2, atol=5e-2)


@ray.remote(num_gpus=1, max_calls=1)
def eager_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group().device_group

        # A small all_reduce for warmup.
        data = torch.zeros(1)
        data = data.to(device=device)
        torch.distributed.all_reduce(data, group=group)
        torch.cuda.synchronize()
        del data

        for M, N in test_sizes:
            for dtype in [torch.float16, torch.bfloat16]:
                # use integers so result matches NCCL exactly
                inp = torch.randint(
                    1, 16, (M, N), dtype=dtype, device=torch.cuda.current_device()
                )
                residual = torch.randint(
                    1, 16, (M, N), dtype=dtype, device=torch.cuda.current_device()
                )
                norm = RMSNorm(N, eps=1e-5).to(device=device, dtype=dtype)
                max_m = M * 2

                # compute reference: all_reduce(inp) + residual
                ref_inp = inp.clone()
                dist.all_reduce(ref_inp, group=group)
                ref_res_out = ref_inp + residual

                # triton fused all_reduce + residual + rmsnorm
                out, res_out = triton_allreduce(inp, residual, norm, max_m)
                torch.testing.assert_close(res_out, ref_res_out, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Triton allreduce with Iris is ROCm-only"
)
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1])
@pytest.mark.parametrize("test_target", [eager_allreduce, graph_allreduce])
def test_triton_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pipeline_parallel_size,
    test_target,
):
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.cuda.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    monkeypatch.setenv("VLLM_ROCM_TRITON_ALLREDUCE", "1")
    monkeypatch.setenv("VLLM_TRITON_ALLREDUCE_IMPL", "simple_allreduce")
    multi_process_parallel(monkeypatch, tp_size, pipeline_parallel_size, test_target)
