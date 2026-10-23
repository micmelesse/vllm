# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HipCustomCommunicator at Kimi-K3's configuration: TP=8, the hidden row (7168)
and the latent MoE row (3584), decode- and prefill-sized batches.

Each op runs eager and under CUDA graphs (several layers a graph, every capture
size under one capture, replayed with fresh inputs), with the backend's choice of
kernel and with each of its kernels forced, against the ops it replaces. Each
two-shot also runs forced onto INT4 against an emulation of the format. Every
call must leave the communicator open and its barrier flags clear.
"""

import copy
import json
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from functools import partial
from multiprocessing.queues import Queue
from typing import TypeAlias, TypeVar, get_args

import pytest
import torch
import torch.distributed as dist

from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.device_communicators.hip_custom_communicator import (
    _TABLE,
    ABANDON_WAIT_S,
    TUNED_OPS,
    Algorithm,
    Error,
    FlagState,
    HipCustomCommunicator,
    LogKind,
    LogMode,
    Op,
    Open,
    Quantization,
    Supported,
    TunedTable,
    Where,
    abandon_open,
    build_info,
    check_table,
    last_log,
    open_handle,
    supported,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

from ..utils import init_test_distributed_environment, multi_gpu_marks

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-only HipCustomCommunicator tests"
)

WORLD_SIZE = 8
# The largest call Kimi-K3's server makes (max_num_batched_tokens 16384 of the
# hidden row), which the communicator's exchange is sized for.
MAX_TOKENS = 16384
HIDDEN = 7168
LATENT = 3584
# Decode batches, fewer rows than ranks among them, and a prefill chunk.
TOKENS = (1, 4, 16, 2048)
DTYPES: dict[Op, tuple[torch.dtype, ...]] = {
    "all_reduce": (torch.bfloat16, torch.float16),
    "all_reduce_rms_norm": (torch.bfloat16, torch.float16),
    "all_reduce_add_rms_norm": (torch.bfloat16, torch.float16),
    "all_reduce_add_attn_res_rms_norm": (torch.bfloat16,),
}
WIDTHS: dict[Op, tuple[int, ...]] = {
    "all_reduce": (HIDDEN, LATENT),
    "all_reduce_rms_norm": (HIDDEN, LATENT),
    "all_reduce_add_rms_norm": (HIDDEN, LATENT),
    "all_reduce_add_attn_res_rms_norm": (HIDDEN,),
}
# AttnRes layers as Kimi-K3 runs them, (has_prefix, num_blocks, write_index): one
# that updates the prefix, and a block-write layer where the sum starts it.
AttnResLayer = tuple[bool, int, int]
ATTN_RES_LAYERS: tuple[AttnResLayer, ...] = ((True, 4, -1), (False, 4, 4))
ATTN_RES_SOURCES = 10
EPS = 1e-5
ATTN_RES_EPS = 1e-6
GRAPH_LAYERS = 2
GRAPH_REPLAYS = 3
KERNEL_TIMEOUT_S = 30.0
PROCESS_TIMEOUT_S = 900
# The basic-correctness (smoke) test's world: a broken basic fails in minutes,
# not a PROCESS_TIMEOUT_S.
# 30 s, as every other test: on 2026-10-16 one rank's first 2048-row launch ran
# ~6 s late (a first launch can load its code object), and a 5 s wait gave up.
SMOKE_KERNEL_TIMEOUT_S = 30.0
SMOKE_PROCESS_TIMEOUT_S = 240
# How long a hung rank has to print its state log before the harness terminates it,
# and how long it then has to tear its communicator down alone (SIGTERM, `abandon_open`)
# before it is killed.
DUMP_GRACE_S = 5.0
TERM_GRACE_S = ABANDON_WAIT_S + 5.0

Forced = dict[str, str | int]
# AN OP'S ONE KERNEL AT ONE OF ITS ALGORITHMS (a config's first two fields).
Variant = tuple[str, Algorithm]
# AN OP AS TWO KERNELS OF OURS, forced: our all-reduce's forcing, then its norm's.
Composed = tuple[Forced, Forced]
Tolerance = tuple[float, float]
F = TypeVar("F", bound=Callable[..., None])


def on_eight_gpus(f: F) -> F:
    for mark in multi_gpu_marks(num_gpus=WORLD_SIZE):
        f = mark(f)
    return f


def _tolerance(dtype: torch.dtype, normed: bool) -> Tolerance:
    """(atol, rtol): a sum lands within a rounding of the input dtype of RCCL's
    fp32 sum; a normed output carries that rounding through the norm."""
    if normed:
        return (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-2, 2e-3)
    return (1e-1, 2e-2) if dtype == torch.bfloat16 else (1e-2, 2e-3)


# --------------------------------------------------------------------------------
# Running a function on every rank.
# --------------------------------------------------------------------------------


# This rank's communicator once open, for the state log a failure or a hang prints.
_COMM: HipCustomCommunicator | None = None


def _state_log(rank: int) -> str:
    """This rank's state log as text: its communicator's, or, before one is open,
    the last log this process made (an open that failed or still runs)."""
    if _COMM is not None:
        return f"rank {rank}: {_COMM._log_text()}"
    try:
        got = last_log()
    except RuntimeError as e:
        return f"rank {rank}: no state log: {e}"
    return f"rank {rank}: {got if got is not None else 'no state log made yet'}"


def _dump_on_signal(rank: int) -> None:
    """Waits for the harness's signals to a hung rank: SIGUSR1, print the state log;
    SIGTERM, tear the communicator down alone (`abandon_open`: borrowed peer mappings
    first, never waiting on a peer), print the log with its close phases, and exit,
    before the SIGKILL that would leave the driver to untangle live peer mappings
    (what wedged n11, 2026-10-16). Its own thread, so it runs while the main thread
    is stuck in a call that released the GIL (a synchronize, an op)."""
    while True:
        got = signal.sigwait({signal.SIGUSR1, signal.SIGTERM})
        if got == signal.SIGUSR1:
            print(f"rank {rank}: hung; its state log:\n{_state_log(rank)}", flush=True)
            continue
        print(f"rank {rank}: terminated; abandoning its communicator", flush=True)
        try:
            abandon_open()
        except RuntimeError as e:
            print(f"rank {rank}: abandon failed: {e}", flush=True)
        print(f"rank {rank}: after abandon:\n{_state_log(rank)}", flush=True)
        os._exit(1)


@contextmanager
def _step(rank: int, name: str) -> Iterator[None]:
    """Prints the step a rank enters, so the captured output of a hung world names
    it, and names it in any failure it raises, with the rank's state log."""
    print(f"rank {rank}: {name}", flush=True)
    try:
        yield
    except Exception as e:
        print(f"rank {rank}: step {name!r} failed\n{_state_log(rank)}", flush=True)
        raise AssertionError(f"rank {rank}: step {name!r} failed: {e}") from e


def _rank_main(
    rank: int,
    port: int,
    target: Callable[..., None],
    kwargs: dict[str, str],
    failures: "Queue[str]",
    kernel_timeout_s: float,
    log_mode: LogMode,
    table: TunedTable | None,
) -> None:
    global _COMM
    # SIGUSR1 AND SIGTERM TO THE DUMPER ONLY: blocked here before any thread starts
    # (threads inherit it), taken by `_dump_on_signal`'s sigwait.
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1, signal.SIGTERM})
    threading.Thread(target=_dump_on_signal, args=(rank,), daemon=True).start()
    try:
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(WORLD_SIZE, 1, rank, str(port), rank)
        tp = get_tp_group()
        # RCCL UP BEFORE THE COMMUNICATOR OPENS, as vLLM's own startup does (and the
        # old harness did): torch builds its RCCL communicator lazily, at the first
        # collective, and on 2026-10-16 RCCL's setup hung when that came after our
        # VMM open (the reference all-reduce in the first call).
        with _step(rank, "rccl warm-up"):
            torch.distributed.all_reduce(
                torch.zeros(1, device=device), group=tp.device_group
            )
            torch.accelerator.synchronize()
        with _step(rank, "open"):
            comm = HipCustomCommunicator(
                tp.cpu_group,
                tp.device_group,
                device,
                timeout_seconds=kernel_timeout_s,
                max_rows=MAX_TOKENS,
                max_cols=HIDDEN,
                log_mode=log_mode,
                # EXACT, whatever the image bakes: the hip image sets
                # VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION=INT4, and a call past
                # the crossover would then be checked, lossy, against the exact
                # tolerance (the e2e smoke of 2026-10-17). INT4 is tested by asking.
                quantization="exact",
                table=table,
            )
        _COMM = comm
        try:
            assert not comm.disabled, f"rank {rank}: open: disabled: {comm.state}"
            target(comm, rank, **kwargs)
        finally:
            _COMM = None
            comm.close()
    except BaseException:
        failures.put(f"rank {rank}:\n{traceback.format_exc()}")
        raise
    finally:
        cleanup_dist_env_and_memory()


def _on_every_rank(
    target: Callable[..., None],
    process_timeout_s: float = PROCESS_TIMEOUT_S,
    kernel_timeout_s: float = KERNEL_TIMEOUT_S,
    log_mode: LogMode = "errors",
    table: TunedTable | None = None,
    **kwargs: str,
) -> None:
    """`target(comm, rank, **kwargs)` on WORLD_SIZE spawned ranks, each holding a
    communicator sized for Kimi-K3's largest call, its state log in `log_mode`, its
    tuned table `table` (None: the file's). A
    rank still running at the deadline is told to print its state log (SIGUSR1),
    given DUMP_GRACE_S, told to tear its communicator down alone (SIGTERM), given
    TERM_GRACE_S, then killed."""
    ctx = mp.get_context("spawn")
    failures = ctx.Queue()
    port = get_open_port()
    procs = [
        ctx.Process(
            target=_rank_main,
            args=(r, port, target, kwargs, failures, kernel_timeout_s, log_mode, table),
        )
        for r in range(WORLD_SIZE)
    ]
    for p in procs:
        p.start()
    deadline = time.monotonic() + process_timeout_s
    for p in procs:
        p.join(timeout=max(0.0, deadline - time.monotonic()))
    hung = [r for r, p in enumerate(procs) if p.is_alive()]
    for r in hung:
        pid = procs[r].pid
        if pid is not None:
            os.kill(pid, signal.SIGUSR1)
    grace = time.monotonic() + DUMP_GRACE_S
    for r in hung:
        procs[r].join(timeout=max(0.0, grace - time.monotonic()))
    for r in hung:
        pid = procs[r].pid
        if pid is not None and procs[r].is_alive():
            os.kill(pid, signal.SIGTERM)
    grace = time.monotonic() + TERM_GRACE_S
    for r in hung:
        procs[r].join(timeout=max(0.0, grace - time.monotonic()))
    for p in procs:
        if p.is_alive():
            p.kill()
            p.join()
    errors = []
    while True:
        try:
            errors.append(failures.get(timeout=1))
        except queue.Empty:
            break
    assert not errors, "\n".join(errors)
    assert not hung, (
        f"ranks {hung} still running after {process_timeout_s} s; each rank's last "
        "printed step is where it hung, and its state log follows it"
    )
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]


def _assert_clean(comm: HipCustomCommunicator, where: str) -> None:
    """The calls so far left the communicator open and every barrier flag clear.
    Every rank looks BETWEEN CALLS: a CPU barrier after this rank's work is done
    and before it reads its signals, another after, so a faster peer cannot have
    started the next call and set its first barrier's flags here (the race seen
    as "96 flags left set", 2026-10-16T19-40-10Z)."""
    torch.accelerator.synchronize()
    dist.barrier(group=comm.cpu_group)
    state = comm.state
    left = [f for f in comm.signals() if f.state is not FlagState.clear]
    dist.barrier(group=comm.cpu_group)
    assert isinstance(state, Open), f"{where}: {state}"
    assert not left, f"{where}: {len(left)} flags left set, {left[:4]}"


def _assert_close(
    where: str,
    got: Sequence[torch.Tensor],
    want: Sequence[torch.Tensor],
    tolerances: Sequence[Tolerance],
) -> None:
    assert len(got) == len(want), where
    for i, (a, b, (atol, rtol)) in enumerate(zip(got, want, tolerances)):
        torch.testing.assert_close(
            a.float(), b.float(), atol=atol, rtol=rtol, msg=f"{where}, output {i}"
        )


# --------------------------------------------------------------------------------
# Each op: its arguments, its call, and the ops it replaces.
# --------------------------------------------------------------------------------


def _randn(shape: tuple[int, ...], dtype: torch.dtype, seed: int) -> torch.Tensor:
    device = torch.device("cuda", torch.accelerator.current_device_index())
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape, generator=g, device=device).to(dtype)


@dataclass(frozen=True)
class Args:
    """An op's arguments besides its input, the same on every rank."""

    weight: torch.Tensor | None = None
    residual: torch.Tensor | None = None
    prefix: torch.Tensor | None = None
    blocks: torch.Tensor | None = None
    norm_weight: torch.Tensor | None = None
    qk_weight: torch.Tensor | None = None
    output_norm_weight: torch.Tensor | None = None
    num_blocks: int = 0
    write_index: int = -1

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            f.name: v
            for f in fields(self)
            if isinstance(v := getattr(self, f.name), torch.Tensor)
        }

    def rows(self, n: int) -> "Args":
        """The first `n` rows of every per-token argument."""
        per_token = ("residual", "prefix", "blocks")
        return replace(
            self, **{k: v[:n] for k, v in self.tensors().items() if k in per_token}
        )

    def clone(self) -> "Args":
        return replace(self, **{k: v.clone() for k, v in self.tensors().items()})

    def copy_(self, other: "Args") -> None:
        for k, v in self.tensors().items():
            v.copy_(getattr(other, k))


def _make(
    op: Op,
    rows: int,
    hidden: int,
    dtype: torch.dtype,
    seed: int,
    rank: int,
    layer: AttnResLayer | None = None,
) -> tuple[torch.Tensor, Args]:
    """This rank's input and the op's other arguments."""
    x = _randn((rows, hidden), dtype, seed * WORLD_SIZE + rank + 1)
    shared = seed * 101 + 7
    if op in ("all_reduce_rms_norm", "all_reduce_add_rms_norm"):
        residual = None
        if op == "all_reduce_add_rms_norm":
            residual = _randn((rows, hidden), dtype, shared + 1)
        return x, Args(weight=_randn((hidden,), dtype, shared), residual=residual)
    if op == "all_reduce_add_attn_res_rms_norm":
        assert layer is not None
        has_prefix, num_blocks, write_index = layer
        return x, Args(
            prefix=_randn((rows, hidden), dtype, shared + 1) if has_prefix else None,
            blocks=_randn((rows, ATTN_RES_SOURCES, hidden), dtype, shared + 2),
            norm_weight=_randn((hidden,), dtype, shared + 3),
            qk_weight=_randn((hidden,), dtype, shared + 4),
            output_norm_weight=_randn((hidden,), dtype, shared + 5),
            num_blocks=num_blocks,
            write_index=write_index,
        )
    return x, Args()


def _call(
    comm: HipCustomCommunicator,
    op: Op,
    x: torch.Tensor,
    args: Args,
    forced: Forced | Composed,
) -> list[torch.Tensor]:
    """The op's outputs. AttnRes also writes its block into `args.blocks`. A
    composition runs our all-reduce, then the op's norm on rows on its sum, each its
    public op, as a caller composes it."""
    if isinstance(forced, tuple):
        summed, rows = comm.all_reduce(x, **forced[0]), forced[1]
        if op == "all_reduce_rms_norm":
            assert args.weight is not None
            return [comm.rms_norm(summed, args.weight, EPS, **rows)]
        if op == "all_reduce_add_rms_norm":
            assert args.weight is not None and args.residual is not None
            return list(
                comm.add_rms_norm(summed, args.residual, args.weight, EPS, **rows)
            )
        assert args.blocks is not None
        assert args.norm_weight is not None and args.qk_weight is not None
        prefix, out = comm.add_attn_res_rms_norm(
            summed,
            args.prefix,
            args.blocks,
            args.norm_weight,
            args.qk_weight,
            args.output_norm_weight,
            args.num_blocks,
            args.write_index,
            ATTN_RES_EPS,
            EPS,
            **rows,
        )
        return [out, prefix, args.blocks]
    if op == "all_reduce":
        return [comm.all_reduce(x, **forced)]
    if op == "all_reduce_rms_norm":
        assert args.weight is not None
        return [comm.all_reduce_rms_norm(x, args.weight, EPS, **forced)]
    if op == "all_reduce_add_rms_norm":
        assert args.weight is not None and args.residual is not None
        return list(
            comm.all_reduce_add_rms_norm(x, args.residual, args.weight, EPS, **forced)
        )
    assert args.blocks is not None
    assert args.norm_weight is not None and args.qk_weight is not None
    prefix, out = comm.all_reduce_add_attn_res_rms_norm(
        x,
        args.prefix,
        args.blocks,
        args.norm_weight,
        args.qk_weight,
        args.output_norm_weight,
        args.num_blocks,
        args.write_index,
        ATTN_RES_EPS,
        EPS,
        **forced,
    )
    return [out, prefix, args.blocks]


def _sum(x: torch.Tensor, comm: HipCustomCommunicator) -> torch.Tensor:
    """RCCL's fp32 sum of every rank's `x`, landed in its dtype."""
    total = x.float().contiguous()
    dist.all_reduce(total, group=comm.device_group)
    return total.to(x.dtype)


def _reference(
    op: Op, summed: torch.Tensor, args: Args
) -> tuple[list[torch.Tensor], list[Tolerance]]:
    """What the op replaces, on the exact sum, and each output's tolerance."""
    import vllm.ir.ops
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res

    sums = _tolerance(summed.dtype, normed=False)
    normed = _tolerance(summed.dtype, normed=True)
    if op == "all_reduce":
        return [summed], [sums]
    if op == "all_reduce_rms_norm":
        return [vllm.ir.ops.rms_norm(summed, args.weight, EPS)], [normed]
    if op == "all_reduce_add_rms_norm":
        assert args.residual is not None
        out, residual = vllm.ir.ops.fused_add_rms_norm(
            summed, args.residual.clone(), args.weight, EPS
        )
        return [out, residual], [normed, sums]
    assert args.blocks is not None
    prefix = summed.clone() if args.prefix is None else args.prefix.clone()
    blocks = args.blocks.clone()
    out = attn_res(
        prefix,
        None if args.prefix is None else summed,
        blocks,
        args.norm_weight,
        args.qk_weight,
        args.output_norm_weight,
        args.num_blocks,
        args.write_index,
        ATTN_RES_EPS,
        EPS,
    )
    return [out, prefix, blocks], [normed, sums, sums]


def _should(comm: HipCustomCommunicator, op: Op, x: torch.Tensor, args: Args) -> bool:
    if op == "all_reduce":
        return comm.should_allreduce(x)
    if op == "all_reduce_add_attn_res_rms_norm":
        return comm.should_allreduce_add_attn_res_rms_norm(x)
    assert args.weight is not None
    if op == "all_reduce_rms_norm":
        return comm.should_allreduce_rms_norm(x, args.weight)
    return comm.should_allreduce_add_rms_norm(x, args.weight)


def _forced(
    variant: Variant, cols: int = 0, quantization: Quantization = "exact"
) -> Forced | None:
    """`variant` forced whole: the kernel by its name, its first build at the variant's
    algorithm moving `quantization` whose tile holds `cols` columns, and a fixed grid.
    None when no build's tile holds them."""
    kernel, algorithm = variant
    build = next(
        (
            b
            for b in build_info().templates[kernel].builds
            if b["algorithm"] == algorithm
            and b["quantization"] == quantization
            and int(b["tile_n"]) >= cols
        ),
        None,
    )
    if build is None:
        return None
    return {"kernel": kernel, "grid_x": 16, "grid_y": 1, "grid_z": 1, **build}


def _forced_whole(
    variant: Variant, cols: int = 0, quantization: Quantization = "exact"
) -> Forced:
    forced = _forced(variant, cols, quantization)
    assert forced is not None, f"{variant} has no {quantization} build for {cols}"
    return forced


def _error_of(e: RuntimeError) -> Error:
    """THE ERROR a refusal names: its message's first word."""
    return Error[str(e).split(":")[0]]


def _variants(op: Op) -> list[Variant]:
    """Every variant of `op`'s kernel the build has, in its builds' order."""
    return list(
        dict.fromkeys(
            (name, b["algorithm"])
            for name, t in build_info().templates.items()
            if t.op == op
            for b in t.builds
        )
    )


# EACH FUSED OP'S NORM ON ROWS, the second kernel of it as two kernels of ours.
ROWS: dict[Op, Op] = {
    "all_reduce_rms_norm": "rms_norm",
    "all_reduce_add_rms_norm": "add_rms_norm",
    "all_reduce_add_attn_res_rms_norm": "add_attn_res_rms_norm",
}


def _tile_cols(op: Op, hidden: int) -> int:
    """The columns a forced build's tile must hold: the row, where the op's tile
    holds it."""
    return 0 if op == "all_reduce" else hidden


# --------------------------------------------------------------------------------
# The basics, first and fast: one short-deadline world, each step named.
# --------------------------------------------------------------------------------


def _smoke_on_rank(comm: HipCustomCommunicator, rank: int) -> None:
    bf16 = torch.bfloat16
    with _step(rank, f"all_reduce 1x{HIDDEN} eager"):
        _eager(comm, rank, "all_reduce", 1, HIDDEN, bf16, {}, None, 1)
    for op in TUNED_OPS:
        if op != "all_reduce":
            layer = ATTN_RES_LAYERS[0] if "attn_res" in op else None
            with _step(rank, f"{op} 1x{HIDDEN} eager"):
                _eager(comm, rank, op, 1, HIDDEN, bf16, {}, layer, 2)
    with _step(rank, f"all_reduce 2048x{HIDDEN} eager"):
        _eager(comm, rank, "all_reduce", 2048, HIDDEN, bf16, {}, None, 3)
    with _step(rank, f"all_reduce 1x{HIDDEN} graph capture and replay"):
        _graphs(comm, rank, "all_reduce", (1,), HIDDEN, bf16, {}, None)
    with _step(rank, "the capture's inputs registered, the eager inputs copied"):
        log = comm.log()
        assert log is not None
        registered = [
            r for r in log.records if r.kind is LogKind.registration and r.barrier == 1
        ]
        assert registered and registered[-1].seen == GRAPH_LAYERS, registered
        assert any(r.kind is LogKind.copy_in_issued for r in log.records)


@on_eight_gpus
def test_basic_correctness() -> None:
    """Open, each op once at a decode row, a prefill-sized all-reduce and one
    CUDA graph, against the ops they replace, every call leaving the communicator
    open and its flags clear: a broken basic fails in minutes, naming its step."""
    _on_every_rank(
        _smoke_on_rank,
        process_timeout_s=SMOKE_PROCESS_TIMEOUT_S,
        kernel_timeout_s=SMOKE_KERNEL_TIMEOUT_S,
        log_mode="everything",
    )


# --------------------------------------------------------------------------------
# Every op, eager and under CUDA graphs.
# --------------------------------------------------------------------------------


def _eager(
    comm: HipCustomCommunicator,
    rank: int,
    op: Op,
    rows: int,
    hidden: int,
    dtype: torch.dtype,
    forced: Forced,
    layer: AttnResLayer | None,
    seed: int,
) -> bool:
    """One eager call against the reference. False when the forced kernel refused
    it; a plain call the communicator admits must run."""
    where = f"{op} {rows}x{hidden} {dtype} forced={forced or None} eager"
    x, args = _make(op, rows, hidden, dtype, seed, rank, layer)
    summed = _sum(x, comm)
    if not forced:
        assert _should(comm, op, x, args), f"{where}: not admitted"
    try:
        got = _call(comm, op, x, args.clone(), forced)
    except RuntimeError as no:
        assert forced, f"{where}: a plain call was refused: {no}"
        _assert_clean(comm, where)
        return False
    _assert_clean(comm, where)
    _assert_close(where, got, *_reference(op, summed, args))
    return True


def _graphs(
    comm: HipCustomCommunicator,
    rank: int,
    op: Op,
    tokens: Sequence[int],
    hidden: int,
    dtype: torch.dtype,
    forced: Forced,
    layer: AttnResLayer | None,
) -> None:
    """As vLLM captures: GRAPH_LAYERS calls a graph, every capture size warmed up
    and captured under one capture, each reading a prefix of the same static
    buffers; replayed with fresh inputs, an eager call between replays."""
    where = f"{op} {hidden} {dtype} forced={forced or None} graph"
    statics = [
        _make(op, max(tokens), hidden, dtype, 1000 + i, rank, layer)
        for i in range(GRAPH_LAYERS)
    ]
    graphs: dict[int, torch.cuda.CUDAGraph] = {}
    outs: dict[int, list[list[torch.Tensor]]] = {}
    with comm.capture():
        for t in tokens:
            for x, args in statics:  # the warmup, which launches nothing
                _call(comm, op, x[:t], args.rows(t), forced)
            graphs[t] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[t]):
                outs[t] = [
                    _call(comm, op, x[:t], args.rows(t), forced) for x, args in statics
                ]
    for replay in range(GRAPH_REPLAYS):
        for t in tokens:
            fresh = [
                _make(op, t, hidden, dtype, 2000 + 10 * replay + i, rank, layer)
                for i in range(GRAPH_LAYERS)
            ]
            for (x, args), (fresh_x, fresh_args) in zip(statics, fresh):
                x[:t].copy_(fresh_x)
                args.rows(t).copy_(fresh_args)
            graphs[t].replay()
            for i, (fresh_x, fresh_args) in enumerate(fresh):
                _assert_close(
                    f"{where} {t} tokens, layer {i}, replay {replay}",
                    outs[t][i],
                    *_reference(op, _sum(fresh_x, comm), fresh_args),
                )
            _assert_clean(comm, f"{where} {t} tokens, replay {replay}")
        _eager(comm, rank, op, tokens[0], hidden, dtype, forced, layer, 3000 + replay)


def _op_on_rank(comm: HipCustomCommunicator, rank: int, op: Op) -> None:
    layers = ATTN_RES_LAYERS if op == "all_reduce_add_attn_res_rms_norm" else (None,)
    for dtype in DTYPES[op]:
        for hidden in WIDTHS[op]:
            forcings: list[Forced | Composed] = [
                {},
                *(
                    forced
                    for t in _variants(op)
                    if (forced := _forced(t, _tile_cols(op, hidden))) is not None
                ),
                # AS TWO KERNELS OF OURS: the plain all-reduce, then each norm on rows.
                *(
                    ({}, forced)
                    for t in (_variants(ROWS[op]) if op in ROWS else [])
                    if (forced := _forced(t, hidden)) is not None
                ),
            ]
            for layer in layers:
                for forced in forcings:
                    ran = [
                        t
                        for t in TOKENS
                        if _eager(comm, rank, op, t, hidden, dtype, forced, layer, t)
                    ]
                    if ran:
                        _graphs(comm, rank, op, ran, hidden, dtype, forced, layer)
                    elif rank == 0:
                        print(f"{op} {hidden} {dtype} {forced}: refused at every size")


@on_eight_gpus
@pytest.mark.parametrize("op", TUNED_OPS)
def test_op_matches_the_ops_it_replaces(op: Op) -> None:
    _on_every_rank(_op_on_rank, op=op)


# --------------------------------------------------------------------------------
# INT4 across the links: each two-shot forced onto its INT4 build, against an
# emulation of quick reduce's format. Each rank's input quantized and landed in
# its dtype, summed in fp32 in the kernel's rank order, the sum quantized, then
# the op's reference on it (the pull norms split rows, sum unquantized and
# quantize what they gather: the owner's normed rows, or its residual rows beside
# their RMS scales).
# --------------------------------------------------------------------------------

INT4_GROUP = 32
INT4_SHAPES = ((1, LATENT), (16, HIDDEN), (96, HIDDEN), (2048, HIDDEN))
# Near every element as the emulation computes it, the rest a quantization step
# off. INT4's own error against the exact op is the format's, judged end to end
# (the gsm8k gate), not here: a kernel is right when it computes what the format
# computes.
INT4_TIGHT = 0.995
INT4_LANDED_TOLERANCE = (1e-6, 2**-7)
INT4_SEED = 20260615


def _int4_input(rank: int, k: int, shape: tuple[int, ...]) -> torch.Tensor:
    """Built on the CPU, so every rank holds every rank's input."""
    g = torch.Generator().manual_seed(INT4_SEED + rank * 1_000_003 + k)
    return torch.randn(shape, generator=g).to(torch.bfloat16)


def _int4_round_trip(x: torch.Tensor) -> torch.Tensor:
    """Through quick reduce's INT4 and back, in fp32: each group's scale its largest
    magnitude / 7 rounded to fp16, each value the nearest of -8..7 times it."""
    groups = x.float().reshape(*x.shape[:-1], -1, INT4_GROUP)
    amax = groups.abs().amax(dim=-1, keepdim=True)
    scale = (
        torch.clamp(amax * torch.tensor(1.0 / 7.0, dtype=torch.float32), max=65504.0)
        .to(torch.float16)
        .float()
    )
    inv = torch.where(scale > 0, 1.0 / scale, torch.zeros_like(scale))
    q = torch.clamp(torch.round(groups * inv), -8.0, 7.0)
    return (q * scale).reshape(x.shape)


def _int4_landed(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return _int4_round_trip(x).to(dtype)


def _int4_sum(terms: Sequence[torch.Tensor], owner: torch.Tensor) -> torch.Tensor:
    """The fp32 sum in the kernel's order: each element from its `owner` on."""
    stacked = torch.stack([t.float() for t in terms])
    acc = torch.zeros_like(stacked[0])
    for k in range(WORLD_SIZE):
        acc += stacked.gather(0, ((owner + k) % WORLD_SIZE).unsqueeze(0)).squeeze(0)
    return acc


def _int4_owner(op: Op, variant: Variant, shape: tuple[int, int]) -> torch.Tensor:
    """Which rank sums each element: the plain two-shot's flat share, the AttnRes
    pull's row slice (its INT4 build whole groups a thread, each owner's rows summed from
    its inbox), or rank order (the push kernels and the pull norms)."""
    rows, hidden = shape
    if op == "all_reduce":
        flat = torch.arange(rows * hidden).reshape(rows, hidden)
        return flat // (rows * hidden // WORLD_SIZE)
    if variant[1] == "two_shot_pull" and "attn_res" in op:
        slice_m = -(-rows // WORLD_SIZE)
        return (torch.arange(rows) // slice_m).unsqueeze(1).expand(rows, hidden)
    return torch.zeros(rows, hidden, dtype=torch.long)


def _int4_verdict(
    where: str,
    got: torch.Tensor,
    want: torch.Tensor,
    tolerance: Tolerance,
) -> None:
    a, b = got.float().cpu(), want.float().cpu()
    assert a.shape == b.shape, f"{where}: {tuple(a.shape)}, not {tuple(b.shape)}"
    assert torch.isfinite(a).all(), f"{where} is not finite"
    atol, rtol = tolerance
    tight = ((a - b).abs() <= atol + rtol * b.abs()).float().mean().item()
    assert tight >= INT4_TIGHT, (
        f"{where}: {tight:.3%} of elements match the INT4 emulation "
        f"(worst |diff| {(a - b).abs().max().item():.4g})"
    )


def _int4_case(
    comm: HipCustomCommunicator,
    rank: int,
    op: Op,
    variant: Variant,
    shape: tuple[int, int],
) -> None:
    import vllm.ir.ops
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res

    where = f"{variant} {shape} int4"
    device, dtype = comm.device, torch.bfloat16
    rows, hidden = shape
    inputs = [_int4_input(r, 0, shape) for r in range(WORLD_SIZE)]
    landed = [_int4_landed(x, dtype) for x in inputs]
    total = _int4_sum(landed, _int4_owner(op, variant, shape))
    summed = _int4_landed(total, dtype)
    pull_norm = (
        variant[1] == "two_shot_pull"
        and "rms_norm" in op
        and "attn_res" not in op
    )
    forced = _forced_whole(variant, _tile_cols(op, hidden), "int4")
    mine = inputs[rank].to(device)
    normed = _tolerance(dtype, normed=True)
    checks: list[tuple[str, torch.Tensor, torch.Tensor, Tolerance]]
    if op == "all_reduce":
        got = comm.all_reduce(mine, **forced)
        checks = [("out", got, summed, INT4_LANDED_TOLERANCE)]
    elif op in ("all_reduce_rms_norm", "all_reduce_add_rms_norm"):
        residual = _int4_input(WORLD_SIZE, 1, shape)
        weight = _int4_input(WORLD_SIZE + 1, 2, (hidden,))
        if op == "all_reduce_add_rms_norm":
            got, got_residual = comm.all_reduce_add_rms_norm(
                mine, residual.to(device), weight.to(device), EPS, **forced
            )
            if pull_norm:
                # A row split: the owner adds the residual to the unquantized sum
                # and takes its RMS scale; the residual row crosses quantized, and
                # every rank norms the landed row by the owner's scale.
                s = total.to(dtype).float() + residual.float()
                scale = torch.rsqrt(s.pow(2).mean(dim=-1, keepdim=True) + EPS)
                want_residual = _int4_landed(s.to(dtype), dtype)
                want = ((want_residual.float() * scale).to(dtype) * weight).to(dtype)
            else:
                want, want_residual = vllm.ir.ops.fused_add_rms_norm(
                    summed, residual, weight, EPS
                )
            checks = [
                ("out", got, want, normed),
                ("residual", got_residual, want_residual, INT4_LANDED_TOLERANCE),
            ]
        else:
            got = comm.all_reduce_rms_norm(mine, weight.to(device), EPS, **forced)
            if pull_norm:
                # The owner norms the unquantized sum; the normed row crosses.
                want = _int4_landed(
                    vllm.ir.ops.rms_norm(total.to(dtype), weight, EPS), dtype
                )
            else:
                want = vllm.ir.ops.rms_norm(summed, weight, EPS)
            checks = [("out", got, want, normed)]
    else:
        # The block-write layer: the sum starts the prefix and is written as the
        # block, so both are the landed sum.
        blocks = torch.stack(
            [
                _int4_input(WORLD_SIZE + 2 + s, 3, shape)
                for s in range(ATTN_RES_SOURCES)
            ],
            dim=1,
        ).contiguous()
        norm_w = _int4_input(WORLD_SIZE + 20, 4, (hidden,)).to(device)
        qk_w = _int4_input(WORLD_SIZE + 21, 5, (hidden,)).to(device)
        out_w = _int4_input(WORLD_SIZE + 22, 6, (hidden,)).to(device)
        num_blocks = write_index = 4

        def reference(
            the_sum: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            prefix = the_sum.to(device).clone()
            ref_blocks = blocks.to(device).clone()
            out = attn_res(
                prefix,
                None,
                ref_blocks,
                norm_w,
                qk_w,
                out_w,
                num_blocks,
                write_index,
                ATTN_RES_EPS,
                EPS,
            )
            return out, prefix, ref_blocks

        want, want_prefix, want_blocks = reference(summed)
        got_blocks = blocks.to(device).clone()
        got_prefix, got = comm.all_reduce_add_attn_res_rms_norm(
            mine,
            None,
            got_blocks,
            norm_w,
            qk_w,
            out_w,
            num_blocks,
            write_index,
            ATTN_RES_EPS,
            EPS,
            **forced,
        )
        checks = [
            ("out", got, want, normed),
            ("prefix", got_prefix, want_prefix, INT4_LANDED_TOLERANCE),
            ("blocks", got_blocks, want_blocks, INT4_LANDED_TOLERANCE),
        ]
    _assert_clean(comm, where)
    for name, a, b, tolerance in checks:
        _int4_verdict(f"{where} {name}", a, b, tolerance)


def _unquantized_case(comm: HipCustomCommunicator, rank: int) -> None:
    """A call that does not ask for INT4 moves the input as it is: a plain call on
    a communicator whose quantization is exact (the default), whose table's pick moves
    nothing quantized, and a forced exact two-shot."""
    forced = _forced_whole(("all_reduce", "two_shot_pull"))
    ((*_, quantization),) = comm.tuned("all_reduce", 2048, HIDDEN)
    assert quantization == "exact" and forced["quantization"] == "exact"
    for name, forcing in (("plain", {}), ("forced", forced)):
        x = _randn((2048, HIDDEN), torch.bfloat16, 77 * WORLD_SIZE + rank)
        summed = _sum(x, comm)
        got = comm.all_reduce(x, **forcing)
        _assert_clean(comm, name)
        _assert_close(name, [got], [summed], [_tolerance(x.dtype, normed=False)])


def _int4_on_rank(comm: HipCustomCommunicator, rank: int) -> None:
    for op in TUNED_OPS:
        for variant in _variants(op):
            kernel, algorithm = variant
            if any(
                (b["algorithm"], b["quantization"]) == (algorithm, "int4")
                for b in build_info().templates[kernel].builds
            ):
                for shape in INT4_SHAPES:
                    _int4_case(comm, rank, op, variant, shape)
    _unquantized_case(comm, rank)


@on_eight_gpus
def test_int4_two_shot_matches_the_format_it_moves() -> None:
    # Every rank's slice of each row is whole INT4 groups.
    assert all(cols % (INT4_GROUP * WORLD_SIZE) == 0 for _, cols in INT4_SHAPES)
    _on_every_rank(_int4_on_rank)


# --------------------------------------------------------------------------------
# What it refuses, and that a refused call leaves it working.
# --------------------------------------------------------------------------------

Outputs: TypeAlias = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


def _refused(
    comm: HipCustomCommunicator, rank: int, where: str, call: Callable[[], Outputs]
) -> Error | None:
    """The Error `call` was refused for, None when it ran; and after it the
    communicator is open, its flags are clear and the next all-reduce is right."""
    try:
        call()
        error = None
    except RuntimeError as no:
        error = _error_of(no)
    _assert_clean(comm, where)
    _eager(comm, rank, "all_reduce", 16, HIDDEN, torch.bfloat16, {}, None, 99)
    return error


def _refusals_on_rank(comm: HipCustomCommunicator, rank: int) -> None:
    bf16 = torch.bfloat16

    # A two-shot row with fewer 16-byte packs than ranks runs (the late ranks'
    # slices empty), as the plain call does.
    narrow = _randn((16, 8), bf16, rank)
    two_shot = _forced_whole(("all_reduce", "two_shot_pull"))
    error = _refused(
        comm, rank, "narrow row", lambda: comm.all_reduce(narrow, **two_shot)
    )
    assert error is None, error
    _eager(comm, rank, "all_reduce", 16, 8, bf16, {}, None, 5)

    # An input whose innermost stride is not 1: declined, and refused if called.
    strided = _randn((16, 2 * HIDDEN), bf16, rank)[:, ::2]
    assert not comm.should_allreduce(strided)
    error = _refused(comm, rank, "strided", lambda: comm.all_reduce(strided))
    assert error is Error.inner_stride_not_one, error

    # A norm weight in neither the input's dtype nor fp32.
    x = _randn((16, HIDDEN), bf16, rank)
    half = torch.ones(HIDDEN, dtype=torch.float16, device=comm.device)
    assert not comm.should_allreduce_rms_norm(x, half)
    assert not comm.should_allreduce_add_rms_norm(x, half)
    error = _refused(
        comm, rank, "fp16 weight", lambda: comm.all_reduce_rms_norm(x, half, EPS)
    )
    assert error is Error.weight_not_built, error
    error = _refused(
        comm,
        rank,
        "fp16 weight, add",
        lambda: comm.all_reduce_add_rms_norm(x, x.clone(), half, EPS),
    )
    assert error is Error.weight_not_built, error

    # Out of capacity: one row wider than the staging, which no pass holds.
    wide = torch.ones((1, comm.capacity.staging_bytes), dtype=bf16, device=comm.device)
    for shot in ("one_shot_pull", "two_shot_pull"):
        forced = _forced_whole(("all_reduce", shot))
        error = _refused(comm, rank, shot, partial(comm.all_reduce, wide, **forced))
        assert error is Error.staging_too_small, (shot, error)

    # Out of capacity: past the rows the exchange was sized for, a fused call the
    # communicator does not admit is refused by name or runs right.
    op: Op = "all_reduce_add_rms_norm"
    big, args = _make(op, MAX_TOKENS + 1, HIDDEN, bf16, 9, rank)
    admitted = _should(comm, op, big, args)
    summed = _sum(big, comm)
    try:
        got = _call(comm, op, big, args.clone(), {})
    except RuntimeError as no:
        assert not admitted, f"an admitted call was refused: {no}"
        assert _error_of(no) in (Error.partials_too_small, Error.staging_too_small), no
    else:
        _assert_close("past the box", got, *_reference(op, summed, args))
    _assert_clean(comm, "past the box")


@on_eight_gpus
def test_refused_calls_name_why_and_fall_back_cleanly() -> None:
    _on_every_rank(_refusals_on_rank)


def _warmup_on_rank(comm: HipCustomCommunicator, rank: int) -> None:
    """Inside `capture()` on a stream that is not recording, rank 0 alone calls
    every op: a launch would wait on peers that never come."""
    x = _randn((16, HIDDEN), torch.bfloat16, rank)
    weight = torch.ones(HIDDEN, device=comm.device, dtype=torch.bfloat16)
    done = torch.cuda.Event()
    outs: list[torch.Tensor] = []
    with comm.capture(), torch.cuda.stream(torch.cuda.Stream()):
        if rank == 0:
            outs = [
                comm.all_reduce(x),
                comm.all_reduce_rms_norm(x, weight, EPS),
                *comm.all_reduce_add_rms_norm(x, x, weight, EPS),
            ]
            done.record()
    if rank == 0:
        deadline = time.monotonic() + 10.0
        while not done.query():
            assert time.monotonic() < deadline, "a warmup call launched"
            time.sleep(0.01)
        assert all(o.shape == x.shape and o.dtype == x.dtype for o in outs)
    dist.barrier(group=comm.cpu_group)
    _assert_clean(comm, "warmup")


@on_eight_gpus
def test_capture_warmup_launches_nothing() -> None:
    """A capture's warmup, which vLLM runs on the capture stream and discards,
    returns outputs of the right shape and launches nothing."""
    _on_every_rank(_warmup_on_rank)


# --------------------------------------------------------------------------------
# The opener, supported and build_info, in one process.
# --------------------------------------------------------------------------------


def test_python_names_match_the_build() -> None:
    """`Error` and `Where` are C++'s, name for name in order, and `Op` names C++'s
    ops."""
    import vllm._rocm_C  # noqa: F401

    built = build_info()
    assert tuple(e.name for e in Error) == built.error_names
    assert tuple(w.name for w in Where) == built.where_names
    assert [w.value for w in Where] == list(range(1, len(built.where_names) + 1))
    assert set(get_args(Op)) == set(built.op_names)


def test_build_info_lists_every_template_with_its_builds() -> None:
    """Each template (an op's one kernel) names one of the ops, one template an op, and
    lists builds, each at one of its algorithms, a tile and no launch; INT4 only on a
    two-shot and not the plain pull, the fused pulls' INT4 builds whole groups a thread
    (TILE_N a multiple of 32 * THREADS, one row high); a norm on rows its one algorithm,
    rows, moving nothing (no peer)."""
    import vllm._rocm_C  # noqa: F401

    built = build_info()
    assert built.templates
    assert sorted(t.op for t in built.templates.values()) == sorted(built.op_names)
    for name, t in built.templates.items():
        assert t.op in built.op_names, name
        for b in t.builds:
            assert b["algorithm"] in get_args(Algorithm), (name, b)
            assert (b["algorithm"] == "rows") == (t.op not in TUNED_OPS), (name, b)
            assert b["block_x"] > 0 and (b["block_y"], b["block_z"]) == (1, 1), (
                name,
                b,
            )
            assert b["waves_per_eu"] >= 1, (name, b)
            assert b["tile_m"] >= 1 and b["tile_n"] > 0, (name, b)
            assert b["quantization"] in get_args(Quantization), (name, b)
            assert not {"grid_x", "grid_y", "grid_z"} & b.keys()
            assert b["tile_k"] >= 1 and ("attn_res" in name or b["tile_k"] == 1), (
                name,
                b,
            )
        for b in t.builds:
            if b["quantization"] != "int4":
                continue
            assert b["algorithm"] in ("two_shot_pull", "two_shot_push"), (name, b)
            if b["algorithm"] == "two_shot_pull":
                assert name != "all_reduce", (name, b)
                assert b["tile_n"] % (b["block_x"] * INT4_GROUP) == 0, (name, b)
                assert b["tile_m"] == 1, (name, b)


# EACH KERNEL'S ALGORITHMS, the bodies its op file has, in its order (the host's
# kAlgorithms_<kernel>).
_FUSED_ALGORITHMS = ("one_shot_pull", "two_shot_pull", "two_shot_push")
ALGORITHMS: dict[str, tuple[str, ...]] = {
    "all_reduce": ("one_shot_pull", "two_shot_pull", "one_shot_push", "two_shot_push"),
    "all_reduce_rms_norm": _FUSED_ALGORITHMS,
    "all_reduce_add_rms_norm": _FUSED_ALGORITHMS,
    "all_reduce_add_attn_res_rms_norm": _FUSED_ALGORITHMS,
    "rms_norm_rows": ("rows",),
    "add_rms_norm_rows": ("rows",),
    "add_attn_res_rms_norm_rows": ("rows",),
}


def _takes(kernel: str, b: dict[str, int | str]) -> bool:
    """Whether `kernel` at build `b`'s algorithm is one it builds: INT4 a two-shot's and
    not the plain pull's (a one-shot reads its peers' inputs as they are); the pushed
    one-shot a row no wider than its unfenced slot (2048 bytes of 16-bit elements); a
    fused pull's INT4 whole groups a thread, a row high; a plain two-shot a row high (the
    flat walk)."""
    algorithm = b["algorithm"]
    if b["quantization"] == "int4" and not (
        algorithm == "two_shot_push" or (algorithm == "two_shot_pull" and kernel != "all_reduce")
    ):
        return False
    if kernel == "all_reduce" and algorithm == "one_shot_push":
        return b["tile_m"] == 1 and int(b["tile_n"]) * 2 <= 2048
    if kernel == "all_reduce" and algorithm in ("two_shot_pull", "two_shot_push"):
        return b["tile_m"] == 1
    if b["quantization"] == "int4" and algorithm == "two_shot_pull":
        return int(b["tile_n"]) % (int(b["block_x"]) * INT4_GROUP) == 0 and b["tile_m"] == 1
    return True


def test_the_builds_are_every_algorithm_of_the_tables_shapes() -> None:
    """What the build compiled is the build half of every config the tuned table
    lists on gfx950, each kernel's in the order the file first names them, then the
    kernel at every algorithm it has at every shape the table holds of it (its block,
    waves, tile and quantization) that the algorithm takes (builds.py, at configure): no
    build without a shape of the table, no config without its build, no algorithm of a
    shape left out, so a pruned table never drops a path a call can be forced to take."""
    import vllm._rocm_C  # noqa: F401

    raw = json.loads(_TABLE.read_text())
    variant = ("algorithm",)
    shape = ("block_x", "block_y", "block_z", "waves_per_eu")
    shape += ("tile_m", "tile_n", "tile_k", "quantization")
    want: dict[str, list[dict[str, int | str]]] = {}
    for op in TUNED_OPS:
        for c in raw[op]["configs"]["gfx950"]:
            b = {f: c[f] for f in variant + shape}
            if b not in want.setdefault(c["kernel"], []):
                want[c["kernel"]].append(b)
    for kernel, bs in want.items():
        shapes = list(dict.fromkeys(tuple(b[f] for f in shape) for b in bs))
        for s in shapes:
            for a in ALGORITHMS[kernel]:
                b = dict(zip(variant + shape, (a, *s)))
                if b not in bs and _takes(kernel, b):
                    bs.append(b)
    got = {name: list(t.builds) for name, t in build_info().templates.items()}
    assert got == want


def test_the_tuned_table_is_the_arch_file() -> None:
    """The arch's tuned table (`hip_custom_communicator.json`) checks: every tuned op's
    inputs, each running one kernel of its op or our all-reduce then its norm on
    rows, every config a build its kernel lists; an arch the table has no configs on
    has none, named; an unsound table raises, naming the entry."""
    import vllm._rocm_C  # noqa: F401

    raw = json.loads(_TABLE.read_text())
    assert check_table(raw, "gfx950") is None
    assert set(raw) == {"format", "version", *TUNED_OPS}
    built = build_info()
    for op in TUNED_OPS:
        t = raw[op]
        configs = t["configs"]["gfx950"]
        assert t["inputs"] and configs, op
        for p in t["inputs"]:
            ops = tuple(
                built.templates[configs[j]["kernel"]].op for j in p["config"]["gfx950"]
            )
            assert ops in ((op,), ("all_reduce", ROWS.get(op))), (op, p)
    assert any(
        len(p["config"]["gfx950"]) == 2 for op in TUNED_OPS for p in raw[op]["inputs"]
    )
    assert isinstance(check_table(raw, "gfx0"), str)
    # AN UNSOUND TABLE, named: a build its kernel does not list, an exact input moving
    # INT4.
    bad = copy.deepcopy(raw)
    bad["all_reduce"]["configs"]["gfx950"][0]["tile_n"] = 100
    with pytest.raises(ValueError, match="all_reduce.configs.gfx950.0.*no build"):
        check_table(bad, "gfx950")
    bad = copy.deepcopy(raw)
    (exact,) = [p for p in bad["all_reduce"]["inputs"] if p["quantization"] == "exact"][
        :1
    ]
    int4 = next(
        i
        for i, c in enumerate(bad["all_reduce"]["configs"]["gfx950"])
        if c["quantization"] == "int4"
    )
    exact["config"]["gfx950"] = [int4]
    with pytest.raises(ValueError, match="an exact input moves INT4"):
        check_table(bad, "gfx950")


# THE HANDED-IN TABLE'S ONE INPUT: the plain all-reduce at the hidden row, a decode
# step of 8 tokens, where the file names one config and this test another.
_HANDED_TOKENS = 8


def _one_input_table(arch: str) -> tuple[TunedTable, dict[str, str | int]]:
    """THE FILE'S TABLE WITH ONE PLAIN ALL-REDUCE INPUT RE-POINTED: the exact inputs at
    WORLD_SIZE, the hidden row, `_HANDED_TOKENS`, naming a config the file does not
    run there (the file's 1-token pick at that width, a config it already runs on
    this world); every other input kept, as the check wants an input at every built
    world. Its config too."""
    raw = json.loads(_TABLE.read_text())
    entry = raw["all_reduce"]

    def named(tokens: int) -> int:
        ((index,),) = [
            p["config"][arch]
            for p in entry["inputs"]
            if (p["quantization"], p["world"], p["hidden"], p["tokens"])
            == ("exact", WORLD_SIZE, HIDDEN, tokens)
        ]
        return index

    handed = named(1)
    assert handed != named(_HANDED_TOKENS), "the file runs that config there already"
    table = copy.deepcopy(raw)
    for p in table["all_reduce"]["inputs"]:
        if (p["quantization"], p["world"], p["hidden"], p["tokens"]) == (
            "exact",
            WORLD_SIZE,
            HIDDEN,
            _HANDED_TOKENS,
        ):
            p["config"] = {arch: [handed]}
    return table, entry["configs"][arch][handed]


def _handed_table_on_rank(comm: HipCustomCommunicator, rank: int, config: str) -> None:
    """The communicator's pick at the handed-in input is its config, and a plain call
    there runs."""
    want = tuple(json.loads(config).values())
    rows, bf16 = _HANDED_TOKENS, torch.bfloat16
    (got,) = comm.tuned("all_reduce", rows, HIDDEN)
    assert got == want, f"rank {rank}: tuned {got}, the table names {want}"
    comm.all_reduce(_randn((rows, HIDDEN), bf16, rank))
    _assert_clean(comm, "the handed-in table")


@on_eight_gpus
def test_a_table_handed_in_is_the_one_it_runs() -> None:
    """A communicator opened with a `table` of the file's shape runs from it, not
    the file: the file's table with one plain all-reduce input re-pointed to a config
    the file does not run there, and its `tuned` answers that config, which a plain
    call runs."""
    # example-based: the decode step the tuner's candidate is checked at (gp tune)
    import vllm._rocm_C  # noqa: F401

    found = supported(torch.device("cuda:0"), WORLD_SIZE)
    assert not isinstance(found, Error), found
    table, config = _one_input_table(found.arch)
    _on_every_rank(_handed_table_on_rank, table=table, config=json.dumps(config))


def test_open_refuses_a_group_no_one_registered() -> None:
    import vllm._rocm_C  # noqa: F401

    got = open_handle("no-such", "no-such", 0, KERNEL_TIMEOUT_S, MAX_TOKENS, HIDDEN)
    assert got is Error.no_such_group


@pytest.mark.parametrize("size", [1, 2, 3, 4, 8, 16])
def test_supported_is_the_builds_answer(size: int) -> None:
    """This device runs exactly the worlds `build_info` says are built."""
    import vllm._rocm_C  # noqa: F401

    got = supported(torch.device("cuda:0"), size)
    if size in build_info().worlds:
        assert isinstance(got, Supported), got
        assert got.arch in torch.cuda.get_device_properties(0).gcnArchName
    else:
        assert got is Error.world_not_built


@pytest.mark.parametrize(
    ("variable", "given", "want"),
    [
        ("NONE", None, "exact"),
        ("INT4", None, "int4"),
        ("NONE", "int4", "int4"),
        ("INT4", "exact", "exact"),
    ],
)
def test_quantization_is_the_given_else_the_environments(
    variable: str,
    given: Quantization | None,
    want: Quantization,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A communicator's quantization (the table its plain calls pick from) is the one
    given, else what VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION asks; quick
    reduce's variable does not turn it on."""
    import vllm.distributed.device_communicators.hip_custom_communicator as module

    monkeypatch.setenv("VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION", variable)
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", "INT4")
    opened: list[bool] = []

    def fake_open_handle(*args: str | int | float) -> Error:
        opened.append(True)
        return Error.no_such_group

    monkeypatch.setattr(module, "open_handle", fake_open_handle)
    monkeypatch.setattr(module, "capacity_of", lambda rows, cols, world: None)

    class Group:
        group_name = "fake"

        def size(self) -> int:
            return WORLD_SIZE

    comm = HipCustomCommunicator(
        Group(),
        Group(),
        "cuda:0",
        timeout_seconds=KERNEL_TIMEOUT_S,
        max_rows=MAX_TOKENS,
        max_cols=HIDDEN,
        quantization=given,
    )
    assert opened == [True] and comm.quantization == want
    assert comm.disabled
