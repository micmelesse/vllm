#!/usr/bin/env python3
# SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""Latency of the rocm_comms collectives: the benchmark twin of
`tests/distributed/test_rocm_comms.py`. Same communicators, built the same way, timed
instead of checked.

Each case is a [tokens, hidden] all-reduce, captured N times into one cudagraph (how
vLLM decodes) and replayed; the reported latency is per op, the slowest rank's. Every
case is also checked once against RCCL, so a fast wrong kernel shows as `ok=False`.

Arms:
    rccl                 vLLM's PyNccl all-reduce, the floor
    aiter                aiter's custom all-reduce, what the baseline arm runs
    hip-<algo>           our kernel, per --algos, swept over --blocks x --threads
    with --fused, also:
    <arm>+norm           the arm, then vLLM's fused_add_rms_norm (the unfused pair)
    hip-<algo>-fused     our all_reduce_rmsnorm

Usage:
    torchrun --nproc_per_node=8 benchmarks/kernels/benchmark_rocm_comms.py \\
        --hidden 7168 --tokens 1 8 32 128 512 2048 4096 --fused --output out.jsonl
"""

import argparse
import json
import os
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, replace
from itertools import product

import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.rocm_comms import make_communicator
from vllm.distributed.device_communicators.rocm_comms.hip import (
    Algo,
    HipCommunicator,
)
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
EPS = 1e-6
# Decode batch sizes up to Kimi-K3's max_num_seqs, then prefill chunks up to its
# max_num_batched_tokens.
DEFAULT_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


@dataclass(frozen=True)
class Result:
    arm: str
    tokens: int
    hidden: int
    dtype: str
    world: int
    bytes: int
    us: float
    busbw_gbs: float
    ok: bool
    blocks: int | None = None
    threads: int | None = None


@dataclass(frozen=True)
class Case:
    """One timed thing: `run(x, residual, weight)` returns what gets checked."""

    arm: str
    run: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    capture: Callable[[], AbstractContextManager]
    fused: bool
    blocks: int | None = None
    threads: int | None = None


def _hip_cases(
    comm: HipCommunicator,
    algos: list[Algo],
    blocks: list[int],
    threads: list[int],
    fused: bool,
) -> Iterator[Case]:
    for algo, b, t in product(algos, blocks, threads):
        tunables = replace(comm.hip_tunables, algo=algo, blocks=b, threads=t)
        suffix = "" if len(blocks) == len(threads) == 1 else f"-b{b}-t{t}"

        def tuned(fn, tunables=tunables):
            def call(x, r, w):
                comm.hip_tunables = tunables
                return fn(x, r, w)

            return call

        yield Case(
            f"hip-{algo}{suffix}",
            tuned(lambda x, r, w: comm.all_reduce(x)),
            comm.capture,
            False,
            b,
            t,
        )
        if fused:
            yield Case(
                f"hip-{algo}{suffix}+norm",
                tuned(lambda x, r, w: _then_norm(comm.all_reduce(x), r, w)),
                comm.capture,
                True,
                b,
                t,
            )
            yield Case(
                f"hip-{algo}{suffix}-fused",
                tuned(lambda x, r, w: comm.all_reduce_rmsnorm(x, r, w, EPS)[0]),
                comm.capture,
                True,
                b,
                t,
            )


def _then_norm(summed: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor):
    """The in-place fused_add_rms_norm vLLM runs, on copies so inputs stay reusable."""
    out, res = summed.clone(), residual.clone()
    ops.fused_add_rms_norm(out, res, weight, EPS)
    return out


def _reference(
    pynccl: PyNcclCommunicator, x: torch.Tensor, r: torch.Tensor, w: torch.Tensor, fused
) -> torch.Tensor:
    summed = pynccl.all_reduce(x.clone())
    return _then_norm(summed, r, w) if fused else summed


def _time(
    case: Case,
    x: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    ops_per_graph: int,
    warmup: int,
    trials: int,
) -> float:
    """Microseconds per op on this rank: `ops_per_graph` launches in one graph."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            case.run(x, r, w)
        graph = torch.cuda.CUDAGraph()
        with case.capture(), torch.cuda.graph(graph, stream=stream):
            for _ in range(ops_per_graph):
                case.run(x, r, w)
    torch.cuda.synchronize()
    for _ in range(warmup):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(trials):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / (trials * ops_per_graph)


def _slowest(value: float, device: torch.device) -> float:
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=get_tp_group().device_group)
    return t.item()


def _all_ranks(ok: bool, device: torch.device) -> bool:
    t = torch.tensor([int(ok)], dtype=torch.int32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=get_tp_group().device_group)
    return bool(t.item())


def _print_table(results: list[Result]) -> None:
    arms = list(dict.fromkeys(r.arm for r in results))
    cell = {(r.arm, r.tokens): r for r in results}
    width = max(12, *(len(a) for a in arms)) + 2
    print("\nus per op (slowest rank); * = disagrees with rccl")
    print(f"{'tokens':>8} {'MB':>8} " + "".join(f"{a:>{width}}" for a in arms))
    for tokens in sorted({r.tokens for r in results}):
        row = [cell.get((a, tokens)) for a in arms]
        mb = next(r for r in row if r is not None).bytes / 1e6
        cols = "".join(
            f"{'-':>{width}}"
            if c is None
            else f"{c.us:>{width - 1}.1f}{' ' if c.ok else '*'}"
            for c in row
        )
        print(f"{tokens:>8} {mb:>8.2f} {cols}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--hidden", type=int, default=7168, help="Kimi-K3 is 7168")
    p.add_argument("--tokens", type=int, nargs="+", default=DEFAULT_TOKENS)
    p.add_argument("--dtype", choices=DTYPES, default="bf16")
    p.add_argument(
        "--algos", nargs="+", default=["one_shot", "two_shot", "mixed"], type=str
    )
    p.add_argument("--blocks", type=int, nargs="+", default=[16])
    p.add_argument("--threads", type=int, nargs="+", default=[512])
    p.add_argument("--fused", action="store_true", help="also time + rmsnorm")
    p.add_argument("--no-baselines", action="store_true", help="hip arms only")
    p.add_argument("--ops-per-graph", type=int, default=10)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--output", help="JSON Lines, one Result per case")
    args = p.parse_args()

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    init_distributed_environment(world, rank, "env://", local_rank)
    with set_current_vllm_config(VllmConfig()):
        ensure_model_parallel_initialized(world, 1)
    cpu_group = get_tp_group().cpu_group
    dtype = DTYPES[args.dtype]

    pynccl = PyNcclCommunicator(group=cpu_group, device=device)
    hip = make_communicator(cpu_group, get_tp_group().device_group, device, "hip")
    assert isinstance(hip, HipCommunicator) and not hip.disabled, "hip unavailable"

    cases: list[Case] = []
    aiter = None
    if not args.no_baselines:
        cases.append(
            Case("rccl", lambda x, r, w: pynccl.all_reduce(x), nullcontext, False)
        )
        if args.fused:
            cases.append(
                Case(
                    "rccl+norm",
                    lambda x, r, w: _then_norm(pynccl.all_reduce(x), r, w),
                    nullcontext,
                    True,
                )
            )
        if rocm_aiter_ops.is_custom_all_reduce_enabled():
            from vllm.distributed.device_communicators.aiter_custom_all_reduce import (
                AiterCustomAllreduce,
            )

            widest = max(args.tokens) * args.hidden * dtype.itemsize
            aiter = AiterCustomAllreduce(cpu_group, device, max_size=2 * widest + 1)
            if not aiter.disabled:
                cases.append(
                    Case(
                        "aiter",
                        lambda x, r, w: aiter.custom_all_reduce(x),
                        aiter.capture,
                        False,
                    )
                )
                if args.fused:
                    cases.append(
                        Case(
                            "aiter+norm",
                            lambda x, r, w: _then_norm(
                                aiter.custom_all_reduce(x), r, w
                            ),
                            aiter.capture,
                            True,
                        )
                    )
    cases += _hip_cases(hip, args.algos, args.blocks, args.threads, args.fused)

    results: list[Result] = []
    for tokens in args.tokens:
        gen = torch.Generator(device=device).manual_seed(rank)
        shape = (tokens, args.hidden)
        x = torch.randn(shape, dtype=dtype, device=device, generator=gen)
        # The residual and weight are replicated across ranks, as in the model.
        gen.manual_seed(1234)
        r = torch.randn(shape, dtype=dtype, device=device, generator=gen)
        w = torch.randn(args.hidden, dtype=dtype, device=device, generator=gen)
        nbytes = x.numel() * x.element_size()
        for case in cases:
            want = _reference(pynccl, x, r, w, case.fused)
            got = case.run(x, r, w)
            ok = _all_ranks(
                torch.allclose(got.float(), want.float(), atol=0.1, rtol=0.05), device
            )
            us = _slowest(
                _time(case, x, r, w, args.ops_per_graph, args.warmup, args.trials),
                device,
            )
            busbw = nbytes / (us * 1e-6) * 2 * (world - 1) / world / 1e9
            results.append(
                Result(
                    case.arm,
                    tokens,
                    args.hidden,
                    args.dtype,
                    world,
                    nbytes,
                    us,
                    busbw,
                    ok,
                    case.blocks,
                    case.threads,
                )
            )

    if rank == 0:
        _print_table(results)
        if args.output:
            with open(args.output, "w") as f:
                for res in results:
                    f.write(json.dumps(asdict(res)) + "\n")
            print(f"\nwrote {len(results)} results to {args.output}")

    hip.close()
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
