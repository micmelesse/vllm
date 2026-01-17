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


def _worker_eager(rank: int, world_size: int, port: int, 
                  inp_cpu: torch.Tensor, residual_cpu: torch.Tensor | None,
                  weight_cpu: torch.Tensor | None, max_m: int):
    """Eager mode test worker."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
        )
        ensure_model_parallel_initialized(world_size, 1)
        group = get_tp_group().device_group

        # Warmup all-reduce (needed for NCCL initialization)
        warmup = torch.zeros(1, device=device)
        dist.all_reduce(warmup, group=group)
        torch.cuda.synchronize()
        del warmup

        # Move tensors to device
        M, N = inp_cpu.shape
        dtype = inp_cpu.dtype
        inp = inp_cpu.to(device)
        residual = residual_cpu.to(device) if residual_cpu is not None else None
        
        # Create RMSNorm if weight provided
        norm = None
        if weight_cpu is not None:
            norm = RMSNorm(N, eps=1e-5).to(device=device, dtype=dtype)
            norm.weight.data.copy_(weight_cpu.to(device))

        # Reference: all_reduce(inp) + residual (if provided)
        ref_out = inp.clone()
        dist.all_reduce(ref_out, group=group)
        if residual is not None:
            ref_res_out = ref_out + residual
        else:
            ref_res_out = ref_out.clone()

        print(f"\n[Rank {rank}] Input: {inp}")
        print(f"[Rank {rank}] Expected all_reduce(inp): {ref_out}")

        # Test: triton all-reduce
        out, res_out = triton_allreduce(inp, max_m, residual=residual, norm=norm)
        torch.cuda.synchronize()

        print(f"[Rank {rank}] Actual out: {out}")
        print(f"[Rank {rank}] Actual res_out: {res_out}")

        # Check res_out matches expected
        if torch.allclose(res_out, ref_res_out, rtol=1e-2, atol=1e-2):
            print(f"[Rank {rank}] SUCCESS: res_out matches!")
        else:
            diff = (res_out - ref_res_out).abs()
            print(f"[Rank {rank}] FAILED: res_out mismatch!")
            print(f"[Rank {rank}] Max diff: {diff.max()}, Mean diff: {diff.mean()}")
            torch.testing.assert_close(res_out, ref_res_out, rtol=1e-2, atol=1e-2)


def _worker_graph(rank: int, world_size: int, port: int,
                  inp_cpu: torch.Tensor, residual_cpu: torch.Tensor | None,
                  weight_cpu: torch.Tensor | None, max_m: int):
    """Graph capture mode test worker."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
        )
        ensure_model_parallel_initialized(world_size, 1)
        group = get_tp_group().device_group

        # Warmup all-reduce (needed for NCCL initialization)
        warmup = torch.zeros(1, device=device)
        dist.all_reduce(warmup, group=group)
        torch.cuda.synchronize()
        del warmup

        # Move tensors to device
        M, N = inp_cpu.shape
        dtype = inp_cpu.dtype
        inp = inp_cpu.to(device)
        residual = residual_cpu.to(device) if residual_cpu is not None else None
        
        # Create RMSNorm if weight provided
        norm = None
        if weight_cpu is not None:
            norm = RMSNorm(N, eps=1e-5).to(device=device, dtype=dtype)
            norm.weight.data.copy_(weight_cpu.to(device))

        # Warmup call to initialize Iris buffers BEFORE graph capture
        _ = triton_allreduce(inp, max_m, residual=residual, norm=norm)
        torch.cuda.synchronize()

        # Reference: all_reduce(inp) + residual (if provided)
        ref_out = inp.clone()
        dist.all_reduce(ref_out, group=group)
        if residual is not None:
            ref_res_out = ref_out + residual
        else:
            ref_res_out = ref_out.clone()

        print(f"\n[Rank {rank}] Graph mode - Input: {inp}")
        print(f"[Rank {rank}] Graph mode - Expected: {ref_res_out}")

        with graph_capture(device=device) as graph_capture_context:
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                out, res_out = triton_allreduce(inp, max_m, residual=residual, norm=norm)

        graph.replay()
        torch.cuda.synchronize()

        print(f"[Rank {rank}] Graph mode - Actual res_out: {res_out}")

        if torch.allclose(res_out, ref_res_out, rtol=5e-2, atol=5e-2):
            print(f"[Rank {rank}] Graph mode SUCCESS!")
        else:
            diff = (res_out - ref_res_out).abs()
            print(f"[Rank {rank}] Graph mode FAILED! Max diff: {diff.max()}")
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

    # ===== Create test inputs on CPU (shared across all ranks) =====
    # Debug case: small 2x4 tensor with all ones
    # Expected: all_reduce(1) = 2 (for 2 ranks)
    M, N = 2, 4
    dtype = torch.float16
    max_m = 8
    
    inp_cpu = torch.ones((M, N), dtype=dtype)
    residual_cpu = None  # No residual for debug
    weight_cpu = None    # No RMSNorm for debug

    port = get_open_port()
    worker = _worker_eager if mode == "eager" else _worker_graph
    mp.spawn(
        worker, 
        args=(tp_size, port, inp_cpu, residual_cpu, weight_cpu, max_m), 
        nprocs=tp_size, 
        join=True
    )
    )
