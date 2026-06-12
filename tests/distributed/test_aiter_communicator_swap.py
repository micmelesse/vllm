# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Communicator-swap correctness: aiter vs the NCCL reference, eager + cudagraph.

A fast, model-free check of the *swap* we make in serving — routing the TP
collectives through the aiter (iris/gluon) communicator instead of
CustomAllreduce/NCCL. It exercises the real vLLM dispatch
(`tensor_model_parallel_all_reduce` / `_all_gather` through `CudaCommunicator`)
under vLLM's own cudagraph capture, and asserts the aiter result matches a raw
`torch.distributed` reference (the ground truth: sum across ranks for all_reduce,
rank-ordered concat for all_gather).

This compartmentalizes the gsm8k-in-serving failure: the aiter op-level unit test
(`aiter .../test_aiter_communicator.py`) passes, but the corruption appears in the
vLLM serving cudagraph regime. The eager case here is the baseline; if eager
passes and the cudagraph case fails (wrong values for all_reduce, or a CUDA fault
for all_gather), the bug is capture-regime-specific — caught in seconds, not a
30-minute serve+gsm8k loop.
"""

import os

import pytest
import ray
import torch
import torch.distributed as dist

# Ray gives each worker HIP_VISIBLE_DEVICES (its assigned GPU) but
# CUDA_VISIBLE_DEVICES is inherited from the driver (all GPUs); vLLM's ROCm
# platform raises on the mismatch at import. Ray imports this whole module in the
# worker, so reconcile the env HERE, before any vllm import — the worker then
# selects its device by rank (set_device_index below), like the other dist tests.
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.environ.pop("HIP_VISIBLE_DEVICES", None)

from vllm._aiter_ops import rocm_aiter_ops  # noqa: E402
from vllm.distributed.communication_op import (  # noqa: E402
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import (  # noqa: E402
    get_tp_group,
    graph_capture,
)

from ..utils import (  # noqa: E402
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

# Small enough to route to the aiter communicator (it handles small AR on ROCm);
# bf16/fp16 hidden-state-shaped tensors like the serving decode all-reduce.
SHAPES = [(64, 8192), (256, 8192)]
DTYPES = [torch.bfloat16, torch.float16]


def _diff(out, ref):
    """One-line magnitude summary of how `out` differs from the reference.
    Inputs are integers (exactly representable), so any delta is a real error,
    not fp noise."""
    d = (out.float() - ref.float()).abs()
    n = int((d > 0).sum())
    return f"mismatched {n}/{out.numel()}, max|delta|={d.max().item():.4g}"


def _block_diff(out, ref, m, world, my_rank):
    """All-gather pattern: which rank's row-block [r*m:(r+1)*m) is correct. The
    push kernel writes each rank's slice to every peer, so a remote-write race
    shows as this rank's OWN block clean but peers' blocks stale — that pattern
    points at a memory-ordering bug, not a math bug."""
    parts = []
    for r in range(world):
        blk = slice(r * m, (r + 1) * m)
        own = "*" if r == my_rank else ""
        if torch.equal(out[blk], ref[blk]):
            parts.append(f"r{r}{own}=ok")
        else:
            md = (out[blk].float() - ref[blk].float()).abs().max().item()
            parts.append(f"r{r}{own}=BAD({md:.3g})")
    return "blocks[" + " ".join(parts) + "]  (*=own slice, written locally)"


def _setup(monkeypatch, tp_size, pp_size, rank, port):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    # Enable the aiter communicator so the high-level ops dispatch to it. The
    # flags are captured as class attrs when vllm._aiter_ops is imported (before
    # this setenv runs), so reload them — vLLM's sanctioned hook for exactly the
    # monkeypatch-in-a-test case — before the communicator is created below.
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_COMMS", "1")
    rocm_aiter_ops.refresh_env_variables()
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, port)
    ensure_model_parallel_initialized(tp_size, pp_size)
    comm = get_tp_group().device_communicator
    assert comm.aiter_comm is not None and not comm.aiter_comm.disabled, (
        "aiter communicator not active — VLLM_ROCM_USE_AITER_COMMS not honored"
    )
    group = get_tp_group().device_group
    # Warmup so the NCCL communicator is initialized before graph capture.
    warm = torch.zeros(1, device=device)
    dist.all_reduce(warm, group=group)
    torch.accelerator.synchronize()
    return device, group


@ray.remote(num_gpus=1, max_calls=1)
def allreduce_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    with monkeypatch.context() as m:
        device, group = _setup(m, tp_size, pp_size, rank, distributed_init_port)
        for shape in SHAPES:
            for dtype in DTYPES:
                # integers => the cross-rank sum is exact for every dtype
                inp = torch.randint(1, 16, shape, dtype=dtype, device=device)
                ref = inp.clone()
                dist.all_reduce(ref, group=group)

                eager = tensor_model_parallel_all_reduce(inp)
                if not torch.equal(eager, ref):
                    raise AssertionError(
                        f"eager all_reduce {shape} {dtype}: {_diff(eager, ref)}"
                    )

                with graph_capture(device=device) as cc:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=cc.stream):
                        graphed = tensor_model_parallel_all_reduce(inp)
                graph.replay()
                torch.accelerator.synchronize()
                if not torch.equal(graphed, ref):
                    raise AssertionError(
                        f"cudagraph all_reduce {shape} {dtype}: {_diff(graphed, ref)}"
                    )


@ray.remote(num_gpus=1, max_calls=1)
def allgather_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    with monkeypatch.context() as m:
        device, group = _setup(m, tp_size, pp_size, rank, distributed_init_port)
        for shape in SHAPES:
            for dtype in DTYPES:
                inp = torch.randint(1, 16, shape, dtype=dtype, device=device)
                gathered = [torch.empty_like(inp) for _ in range(tp_size)]
                dist.all_gather(gathered, inp, group=group)
                ref = torch.cat(gathered, dim=0)

                eager = tensor_model_parallel_all_gather(inp, dim=0)
                if not torch.equal(eager, ref):
                    raise AssertionError(
                        f"eager all_gather {shape} {dtype}: {_diff(eager, ref)}\n"
                        f"{_block_diff(eager, ref, shape[0], tp_size, rank)}"
                    )

                with graph_capture(device=device) as cc:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=cc.stream):
                        graphed = tensor_model_parallel_all_gather(inp, dim=0)
                graph.replay()
                torch.accelerator.synchronize()
                if not torch.equal(graphed, ref):
                    raise AssertionError(
                        f"cudagraph all_gather {shape} {dtype}: {_diff(graphed, ref)}\n"
                        f"{_block_diff(graphed, ref, shape[0], tp_size, rank)}"
                    )


@pytest.mark.skipif(torch.version.hip is None, reason="aiter communicator is ROCm-only")
@pytest.mark.parametrize("tp_size", [8])
@pytest.mark.parametrize("test_target", [allreduce_worker, allgather_worker])
def test_aiter_communicator_swap(monkeypatch, tp_size, test_target):
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    multi_process_parallel(monkeypatch, tp_size, 1, test_target)
