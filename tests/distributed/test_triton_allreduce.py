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

# Set to True for verbose debug output
DEBUG = True


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

        # Reference calculation:
        # 1. all_reduce(inp)
        # 2. + residual (if provided) -> this is ref_res_out (intermediate)
        # 3. RMSNorm (if provided) -> this is ref_out (final)
        ref_allreduced = inp.clone()
        dist.all_reduce(ref_allreduced, group=group)
        
        # ref_res_out = all_reduce + residual (BEFORE norm)
        if residual is not None:
            ref_res_out = ref_allreduced + residual
        else:
            ref_res_out = ref_allreduced.clone()
        
        # ref_out = after RMSNorm (if provided), else same as ref_res_out
        if norm is not None:
            ref_out = norm(ref_res_out)
        else:
            ref_out = ref_res_out.clone()

        if DEBUG:
            print(f"\n[Rank {rank}] Eager mode - shape={inp.shape}, residual={residual is not None}, norm={norm is not None}")
            print(f"[Rank {rank}] inp sample: {inp[0, :5].tolist()}")
            print(f"[Rank {rank}] ref_allreduced sample: {ref_allreduced[0, :5].tolist()}")
            if residual is not None:
                print(f"[Rank {rank}] residual sample: {residual[0, :5].tolist()}")
            print(f"[Rank {rank}] ref_res_out sample: {ref_res_out[0, :5].tolist()}")
            print(f"[Rank {rank}] ref_out sample: {ref_out[0, :5].tolist()}")

        # Test: triton all-reduce
        # out = final output (after all-reduce + residual + RMSNorm if applicable)
        # res_out = intermediate (after all-reduce + residual, BEFORE RMSNorm)
        out, res_out = triton_allreduce(inp, max_m, residual=residual, norm=norm)
        torch.cuda.synchronize()

        if DEBUG:
            print(f"[Rank {rank}] res_out sample: {res_out[0, :5].tolist()}")
            print(f"[Rank {rank}] out sample: {out[0, :5].tolist()}")

        # Check for NaN first
        if torch.isnan(res_out).any():
            print(f"[Rank {rank}] ERROR: res_out contains NaN!")
        if torch.isnan(out).any():
            print(f"[Rank {rank}] ERROR: out contains NaN!")

        # Check res_out matches ref_res_out (pre-norm)
        res_out_ok = torch.allclose(res_out, ref_res_out, rtol=1e-2, atol=1e-2)
        # Check out matches ref_out (post-norm)
        out_ok = torch.allclose(out, ref_out, rtol=1e-2, atol=1e-2)
        
        if res_out_ok and out_ok:
            print(f"[Rank {rank}] Eager mode SUCCESS!")
        else:
            if not res_out_ok:
                diff = (res_out - ref_res_out).abs()
                print(f"[Rank {rank}] res_out FAILED! Max diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")
            if not out_ok:
                diff = (out - ref_out).abs()
                print(f"[Rank {rank}] out FAILED! Max diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")
            # Assert on res_out first (simpler to debug)
            torch.testing.assert_close(res_out, ref_res_out, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)


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

        # Reference calculation:
        # 1. all_reduce(inp)
        # 2. + residual (if provided) -> this is ref_res_out (intermediate)
        # 3. RMSNorm (if provided) -> this is ref_out (final)
        ref_allreduced = inp.clone()
        dist.all_reduce(ref_allreduced, group=group)
        
        # ref_res_out = all_reduce + residual (BEFORE norm)
        if residual is not None:
            ref_res_out = ref_allreduced + residual
        else:
            ref_res_out = ref_allreduced.clone()
        
        # ref_out = after RMSNorm (if provided), else same as ref_res_out
        if norm is not None:
            ref_out = norm(ref_res_out)
        else:
            ref_out = ref_res_out.clone()

        if DEBUG:
            print(f"\n[Rank {rank}] Graph mode - shape={inp.shape}, residual={residual is not None}, norm={norm is not None}")

        with graph_capture(device=device) as graph_capture_context:
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                out, res_out = triton_allreduce(inp, max_m, residual=residual, norm=norm)

        graph.replay()
        torch.cuda.synchronize()

        if DEBUG:
            print(f"[Rank {rank}] res_out sample: {res_out[0, :5].tolist()}")
            print(f"[Rank {rank}] out sample: {out[0, :5].tolist()}")

        # Check for NaN first
        if torch.isnan(res_out).any():
            print(f"[Rank {rank}] ERROR: res_out contains NaN!")
        if torch.isnan(out).any():
            print(f"[Rank {rank}] ERROR: out contains NaN!")

        # Check res_out matches ref_res_out (pre-norm)
        res_out_ok = torch.allclose(res_out, ref_res_out, rtol=5e-2, atol=5e-2)
        # Check out matches ref_out (post-norm)
        out_ok = torch.allclose(out, ref_out, rtol=5e-2, atol=5e-2)
        
        if res_out_ok and out_ok:
            print(f"[Rank {rank}] Graph mode SUCCESS!")
        else:
            if not res_out_ok:
                diff = (res_out - ref_res_out).abs()
                print(f"[Rank {rank}] res_out FAILED! Max diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")
            if not out_ok:
                diff = (out - ref_out).abs()
                print(f"[Rank {rank}] out FAILED! Max diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")
            # Assert on res_out first (simpler to debug)
            torch.testing.assert_close(res_out, ref_res_out, rtol=5e-2, atol=5e-2)
            torch.testing.assert_close(out, ref_out, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Triton allreduce with Iris is ROCm-only"
)
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("mode", ["eager"])  # Start with eager only for debugging
@pytest.mark.parametrize("with_residual", [False])  # Disable residual for now
@pytest.mark.parametrize("with_norm", [False])  # Disable norm for now
@pytest.mark.parametrize("M,N", [
    (1, 128),      # Single token - small
    (32, 128),     # Small batch
    (128, 4096),   # Medium batch, larger hidden dim
    (256, 8192),   # Larger batch
])
def test_triton_allreduce(tp_size: int, mode: str, 
                          with_residual: bool, with_norm: bool,
                          M: int, N: int):
    if tp_size > torch.cuda.device_count():
        pytest.skip("Not enough GPUs")

    os.environ["VLLM_ROCM_TRITON_ALLREDUCE"] = "1"

    # ===== Create test inputs on CPU (shared across all ranks) =====
    # Use random inputs with fixed seed for reproducibility
    torch.manual_seed(42)
    dtype = torch.float16
    max_m = max(M * 2, 64)  # Ensure max_m > M with some headroom
    
    inp_cpu = torch.randn((M, N), dtype=dtype)
    residual_cpu = torch.randn((M, N), dtype=dtype) if with_residual else None
    weight_cpu = torch.randn(N, dtype=dtype).abs() + 0.1 if with_norm else None  # Positive weights

    # Share memory so spawned processes get identical data
    inp_cpu.share_memory_()
    if residual_cpu is not None:
        residual_cpu.share_memory_()
    if weight_cpu is not None:
        weight_cpu.share_memory_()

    port = get_open_port()
    worker = _worker_eager if mode == "eager" else _worker_graph
    mp.spawn(
        worker, 
        args=(tp_size, port, inp_cpu, residual_cpu, weight_cpu, max_m), 
        nprocs=tp_size, 
        join=True
    )
