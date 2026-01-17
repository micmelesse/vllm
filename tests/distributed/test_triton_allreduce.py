# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton AllReduce tests.

Tests triton_allreduce in both eager and graph capture modes.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.parallel_state import get_tp_group, graph_capture
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.triton_allreduce import triton_allreduce
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

# (M, N) = (num_tokens, hidden_dim)
TEST_SIZES = [(32, 4096), (128, 4096), (32, 8192)]


def _init_worker(rank: int, world_size: int, port: int):
    """Initialize vLLM distributed environment."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
        )
        ensure_model_parallel_initialized(world_size, 1)  # tp_size, pp_size=1

    return device


def _worker_eager(rank: int, world_size: int, port: int):
    """Eager mode test."""
    device = _init_worker(rank, world_size, port)
    group = get_tp_group().device_group

    # Warmup all-reduce (needed for NCCL initialization)
    warmup = torch.zeros(1, device=device)
    dist.all_reduce(warmup, group=group)
    torch.cuda.synchronize()
    del warmup

    for M, N in TEST_SIZES:
        for dtype in [torch.float16, torch.bfloat16]:
            # Use integers so result matches NCCL exactly
            inp = torch.randint(1, 16, (M, N), dtype=dtype, device=device)
            residual = torch.randint(1, 16, (M, N), dtype=dtype, device=device)
            norm = RMSNorm(N, eps=1e-5).to(device=device, dtype=dtype)
            max_m = M * 2

            # Reference: all_reduce(inp) + residual
            ref_inp = inp.clone()
            dist.all_reduce(ref_inp, group=group)
            ref_res_out = ref_inp + residual

            # Test: triton fused all-reduce + residual + rmsnorm
            out, res_out = triton_allreduce(inp, residual, norm, max_m)
            torch.testing.assert_close(res_out, ref_res_out, rtol=1e-2, atol=1e-2)


def _worker_graph(rank: int, world_size: int, port: int):
    """Graph capture mode test."""
    device = _init_worker(rank, world_size, port)
    group = get_tp_group().device_group

    # Warmup all-reduce (required before graph capture)
    warmup = torch.zeros(1, device=device)
    dist.all_reduce(warmup, group=group)
    torch.cuda.synchronize()
    del warmup

    for M, N in TEST_SIZES:
        for dtype in [torch.float16, torch.bfloat16]:
            with graph_capture(device=device) as graph_capture_context:
                # Use integers so result matches NCCL exactly
                inp = torch.randint(1, 16, (M, N), dtype=dtype, device=device)
                residual = torch.randint(1, 16, (M, N), dtype=dtype, device=device)
                norm = RMSNorm(N, eps=1e-5).to(device=device, dtype=dtype)
                max_m = M * 2

                # Reference: all_reduce(inp) + residual
                ref_inp = inp.clone()
                dist.all_reduce(ref_inp, group=group)
                ref_res_out = ref_inp + residual

                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                    # Triton fused all-reduce + residual + rmsnorm
                    out, res_out = triton_allreduce(inp, residual, norm, max_m)

            graph.replay()
            torch.testing.assert_close(res_out, ref_res_out, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Triton allreduce with Iris is ROCm-only"
)
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_triton_allreduce(tp_size: int, mode: str):
    if tp_size > torch.cuda.device_count():
        pytest.skip("Not enough GPUs")

    os.environ["VLLM_ROCM_TRITON_ALLREDUCE"] = "1"
    os.environ["VLLM_TRITON_ALLREDUCE_IMPL"] = "simple_allreduce"

    port = get_open_port()
    worker = _worker_eager if mode == "eager" else _worker_graph
    mp.spawn(worker, args=(tp_size, port), nprocs=tp_size, join=True)
