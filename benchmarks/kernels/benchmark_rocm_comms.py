#!/usr/bin/env python3
# SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""Latency of the rocm_comms ops: the benchmark twin of
`tests/distributed/test_rocm_comms.py`. Same communicators, built the same way, timed
instead of checked.

Each case is one op over [tokens, hidden], captured N times into one cudagraph (how
vLLM decodes) and replayed; the reported latency is per op, the slowest rank's. Every
case is also checked once against RCCL followed by the model's own ops, so a fast wrong
kernel shows as `ok=False`.

Ops (--op), named by the communicator call they time, each an all-reduce then the ops
named, in order:
    all_reduce                          the plain collective
    all_reduce_rms_norm                 then vllm.ir.ops.rms_norm
    all_reduce_add_rms_norm             then vllm.ir.ops.fused_add_rms_norm
    all_reduce_add_attn_res_rms_norm    then Kimi-K3's Triton attn_res, with its output norm
    all_reduce_rms_norm_gemm_add        then rms_norm, then addmm_ into a 1/world column shard

Arms:
    rccl / aiter             that all-reduce, then the ops (for all_reduce, just it)
    aiter-fused              aiter's fused all-reduce + RMSNorm (the norm ops only)
    hip                      our all-reduce, then the ops
    hip-fused                our fused op, the kernel C++ picks
    hip-<kernel>[-bB-tT-lL][-qQ]  our op forced to one kernel, per --blocks x --threads
                             (x --gemm-lanes-per-col for the GEMM tail, x --quant-bits
                             for a push kernel)

Usage (Kimi-K3's decode shapes):
    torchrun --nproc_per_node=8 benchmarks/kernels/benchmark_rocm_comms.py \\
        --op all_reduce_add_attn_res_rms_norm --hidden 7168 --output attn_res.jsonl
"""

import argparse
import json
import os
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, replace
from itertools import product
from typing import cast, get_args

import torch
import torch.distributed as dist

import vllm.ir.ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.rocm_comms import make_communicator
from vllm.distributed.device_communicators.rocm_comms.base import FusedOp
from vllm.distributed.device_communicators.rocm_comms.hip import HipCommunicator
from vllm.distributed.device_communicators.rocm_comms.launch import Kernel, Launch
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
EPS = 1e-5
OPS = ("all_reduce", *get_args(FusedOp))
# THE NAME AN OP IS ASKED FOR AND REPORTED BY: the communicator method it times.
API = {op if op == "all_reduce" else f"all_reduce_{op}": op for op in OPS}
# Decode batch sizes up to Kimi-K3's max_num_seqs, then prefill chunks up to its
# max_num_batched_tokens.
DEFAULT_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
# Kimi-K3 mid-model: 4 stored AttnRes blocks of the 10 a row holds.
ATTN_RES_SOURCES, ATTN_RES_VALID = 10, 4

AllReduce = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Result:
    op: str
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
class Inputs:
    """Everything one op reads, for one token count. `x` is this rank's partial; the
    rest is replicated across ranks, as in the model."""

    x: torch.Tensor
    residual: torch.Tensor
    weight: torch.Tensor
    prefix: torch.Tensor
    blocks: torch.Tensor
    qk_weight: torch.Tensor
    shared: torch.Tensor
    up_proj: torch.Tensor
    col0: int

    def fresh(self) -> "Inputs":
        """The tensors the ops write into, copied: prefix, blocks and shared."""
        return replace(
            self,
            prefix=self.prefix.clone(),
            blocks=self.blocks.clone(),
            shared=self.shared.clone(),
        )


@dataclass(frozen=True)
class Case:
    """One timed thing: `run(inputs)` returns what gets checked."""

    arm: str
    run: Callable[[Inputs], torch.Tensor]
    capture: Callable[[], AbstractContextManager]
    blocks: int | None = None
    threads: int | None = None


def _inputs(op: str, tokens: int, hidden: int, dtype, world, rank, device) -> Inputs:
    gen = torch.Generator(device=device).manual_seed(rank)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(shape, dtype=dtype, device=device, generator=gen)

    x = randn(tokens, hidden)
    gen.manual_seed(1234)
    # The GEMM tail up-projects the latent `hidden` into a row `world` shards wide.
    out_hidden = hidden * 2 if op == "rms_norm_gemm_add" else hidden
    shard = out_hidden // world
    return Inputs(
        x=x,
        residual=randn(tokens, hidden),
        weight=randn(hidden),
        prefix=randn(tokens, hidden),
        blocks=randn(tokens, ATTN_RES_SOURCES, hidden),
        qk_weight=randn(hidden),
        shared=randn(tokens, out_hidden),
        up_proj=randn(shard, hidden) / hidden**0.5,
        col0=rank * shard,
    )


def _tail(op: str, summed: torch.Tensor, t: Inputs) -> torch.Tensor:
    """The model's own ops after the all-reduce, as the unfused path runs them."""
    if op == "all_reduce":
        return summed
    if op == "rms_norm":
        return vllm.ir.ops.rms_norm(summed, t.weight, EPS)
    if op == "add_rms_norm":
        return vllm.ir.ops.fused_add_rms_norm(summed, t.residual, t.weight, EPS)[0]
    if op == "add_attn_res_rms_norm":
        from vllm.models.kimi_k3.amd.ops.attn_res import attn_res

        return attn_res(
            t.prefix,
            summed,
            t.blocks,
            t.weight,
            t.qk_weight,
            t.weight,
            ATTN_RES_VALID,
            -1,
            EPS,
            EPS,
        )
    latent = vllm.ir.ops.rms_norm(summed, t.weight, EPS)
    t.shared.narrow(-1, t.col0, t.up_proj.shape[0]).addmm_(latent, t.up_proj.t())
    return t.shared


def _fused(
    comm: HipCommunicator, op: FusedOp, t: Inputs, launch: Launch | None = None
) -> torch.Tensor:
    if op == "rms_norm":
        return comm.all_reduce_rms_norm(t.x, t.weight, EPS, launch=launch)
    if op == "add_rms_norm":
        return comm.all_reduce_add_rms_norm(
            t.x, t.residual, t.weight, EPS, launch=launch
        )[0]
    if op == "add_attn_res_rms_norm":
        return comm.all_reduce_add_attn_res_rms_norm(
            t.x,
            t.prefix,
            t.blocks,
            t.weight,
            t.qk_weight,
            t.weight,
            ATTN_RES_VALID,
            -1,
            EPS,
            EPS,
            launch=launch,
        )[1]
    comm.all_reduce_rms_norm_gemm_add(
        t.x, t.weight, EPS, t.up_proj, t.shared, t.col0, launch=launch
    )
    return t.shared


def _run(
    comm: HipCommunicator, op: str, launch: Launch | None
) -> Callable[[Inputs], torch.Tensor]:
    """Our op over the inputs, at `launch` (None: as C++ picks)."""
    if op == "all_reduce":
        return lambda t: comm.all_reduce(t.x, launch=launch)
    return lambda t: _fused(comm, cast(FusedOp, op), t, launch)


def _admitted(
    comm: HipCommunicator, op: str, x: torch.Tensor, launch: Launch | None
) -> bool:
    name = "should_allreduce" if op == "all_reduce" else f"should_allreduce_{op}"
    return getattr(comm, name)(x, launch)


def _unfused_case(arm: str, op: str, all_reduce: AllReduce, capture) -> Case:
    return Case(arm, lambda t: _tail(op, all_reduce(t.x), t), capture)


def _hip_cases(
    comm: HipCommunicator,
    op: str,
    x: torch.Tensor,
    blocks,
    threads,
    lanes_per_col,
    quant_bits,
) -> list[Case]:
    cases = [_unfused_case("hip", op, comm.all_reduce, comm.capture)]
    if op != "rms_norm_gemm_add":
        lanes_per_col = [0]
    sweep = len(blocks) * len(threads) * len(lanes_per_col) > 1
    # (kernel, codec bits): a push kernel once per --quant-bits, a pull kernel once (0).
    suffix = "" if op == "all_reduce" else f"_{op}"
    kernels: list[tuple[Kernel, int]] = []
    for shot in ("one_shot", "two_shot"):
        kernels.append((cast(Kernel, f"{shot}_pull{suffix}"), 0))
        kernels += [(cast(Kernel, f"{shot}_push{suffix}"), q) for q in quant_bits]
    if op != "all_reduce" and _admitted(comm, op, x, None):
        cases.append(Case("hip-fused", _run(comm, op, None), comm.capture))
    for (kernel, q), b, tr, v in product(kernels, blocks, threads, lanes_per_col):
        launch = Launch(kernel, b, tr, v, q)
        if _admitted(comm, op, x, launch):
            arm = f"hip-{kernel}" + (f"-b{b}-t{tr}" if sweep else "")
            arm += (f"-l{v}" if v else "") + (f"-q{q}" if q else "")
            cases.append(Case(arm, _run(comm, op, launch), comm.capture, b, tr))
    return cases


def _time(case: Case, t: Inputs, ops_per_graph: int, warmup: int, trials: int) -> float:
    """Microseconds per op on this rank: `ops_per_graph` launches in one graph."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            case.run(t)
        graph = torch.cuda.CUDAGraph()
        with case.capture(), torch.cuda.graph(graph, stream=stream):
            for _ in range(ops_per_graph):
                case.run(t)
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
    print(f"\n{results[0].op}: us per op (slowest rank); * = disagrees with rccl")
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
    p.add_argument("--op", choices=tuple(API), default="all_reduce")
    p.add_argument("--hidden", type=int, default=7168, help="Kimi-K3 is 7168")
    p.add_argument("--tokens", type=int, nargs="+", default=DEFAULT_TOKENS)
    p.add_argument("--dtype", choices=DTYPES, default="bf16")
    p.add_argument("--blocks", type=int, nargs="+", default=[16])
    p.add_argument("--threads", type=int, nargs="+", default=[512])
    p.add_argument(
        "--gemm-lanes-per-col", type=int, nargs="+", default=[0], help="0: the table's"
    )
    p.add_argument(
        "--quant-bits",
        type=int,
        nargs="+",
        default=[16, 8, 4],
        help="a push kernel's codec: 16 (unquantized), 8, 4",
    )
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
    op = API[args.op]

    pynccl = PyNcclCommunicator(group=cpu_group, device=device)
    hip = make_communicator(cpu_group, get_tp_group().device_group, device, "hip")
    assert isinstance(hip, HipCommunicator) and not hip.disabled, "hip unavailable"

    baselines: list[Case] = []
    if not args.no_baselines:
        baselines.append(
            _unfused_case("rccl", op, lambda x: pynccl.all_reduce(x), nullcontext)
        )
        if rocm_aiter_ops.is_custom_all_reduce_enabled():
            from vllm.distributed.device_communicators.aiter_custom_all_reduce import (
                AiterCustomAllreduce,
            )

            widest = max(args.tokens) * args.hidden * dtype.itemsize
            aiter = AiterCustomAllreduce(cpu_group, device, max_size=2 * widest + 1)
            if not aiter.disabled:
                baselines.append(
                    _unfused_case("aiter", op, aiter.custom_all_reduce, aiter.capture)
                )
        # The fused op the baseline's aiter pass rewrites to, on the aiter all-reduce
        # vLLM itself built on the TP communicator.
        vllm_aiter = rocm_aiter_ops.get_aiter_allreduce()
        if op in ("rms_norm", "add_rms_norm") and vllm_aiter is not None:
            aiter_fused = rocm_aiter_ops.get_fused_allreduce_rmsnorm_op()
            if op == "rms_norm":
                run = lambda t: aiter_fused(t.x, torch.zeros_like(t.x), t.weight, EPS)[
                    0
                ]  # noqa: E731
            else:
                run = lambda t: aiter_fused(t.x, t.residual, t.weight, EPS)[0]  # noqa: E731
            baselines.append(Case("aiter-fused", run, vllm_aiter.capture))

    results: list[Result] = []
    for tokens in args.tokens:
        t = _inputs(op, tokens, args.hidden, dtype, world, rank, device)
        cases = baselines + _hip_cases(
            hip,
            op,
            t.x,
            args.blocks,
            args.threads,
            args.gemm_lanes_per_col,
            args.quant_bits,
        )
        nbytes = t.x.numel() * t.x.element_size()
        want = _tail(op, pynccl.all_reduce(t.x.clone()), t.fresh())
        for case in cases:
            got = case.run(t.fresh())
            ok = _all_ranks(
                torch.allclose(got.float(), want.float(), atol=0.1, rtol=0.05), device
            )
            us = _slowest(
                _time(case, t.fresh(), args.ops_per_graph, args.warmup, args.trials),
                device,
            )
            busbw = nbytes / (us * 1e-6) * 2 * (world - 1) / world / 1e9
            results.append(
                Result(
                    args.op,
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
