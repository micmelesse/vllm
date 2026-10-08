# SPDX-License-Identifier: MIT Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All
# rights reserved.
"""Does each communicator backend work?

One test per (backend, mode) and a chosen set of dtypes and shape groups. `exercise`
puts ONE communicator through the whole API -- every dtype and shape the case names --
because that is what vLLM does. Every cell is four steps: gen_inputs -> run_collective
-> expected_outputs -> compare.

The modes localise a failure rather than covering different ground: `eager` asks whether
the collective is right at all, `graph` whether capture/replay preserves that across
many replays with fresh input, `vllm` whether the real pattern works: a collective per
layer, EVERY shape captured together under one registration -- vLLM's capture-size
ladder, sharing one set of buffers -- replayed round-robin, with an eager fallback in
the middle.

ONE WORLD FOR THE SESSION. Bringing up eight ranks -- spawn, import, NCCL, model
parallel -- costs 30-60 s and a case's kernels cost milliseconds, so the ranks are
spawned once (`ranks`) and serve every case: a case is a module-level function run in
every rank as `fn(ctx, **kwargs)`, returning that rank's `(value, err)`. A communicator
is built the first time a case asks for its backend and kept for the world's life, as
vLLM keeps one. A case that errors, hangs or kills a rank ends the world, and the next
case starts a fresh one: a failed collective can leave peer state inconsistent.

TWO TIERS. The default (`-m "not full"`) is what vLLM runs, at Kimi-K3's shapes, plus
the boundaries that decide which path is taken. `full` is everything else: fp16, the
other backends, eager, the remaining shapes and the kernels tune
declines today.
"""

import logging
import math
import multiprocessing as mp
import queue
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import partial
from itertools import product
from multiprocessing import set_start_method
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from typing import Concatenate, Literal, ParamSpec, TypeVar, cast, get_args

import pytest
import torch
import torch.distributed as dist
from _pytest.mark import ParameterSet
from torch.distributed import ProcessGroup

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.device_communicators.rocm_comms import (
    Communicator,
    Error,
    make_communicator,
)
from vllm.distributed.device_communicators.rocm_comms.base import (
    Broken,
    Capturing,
    Closed,
    Disabled,
    Op,
    Open,
    Refused,
    State,
    Supported,
    build_info,
    supported,
)
from vllm.distributed.device_communicators.rocm_comms.experimental import ExperimentalOp
from vllm.distributed.device_communicators.rocm_comms.hip import HipCommunicator
from vllm.distributed.device_communicators.rocm_comms.iris import (
    IrisCommunicator,
)
from vllm.distributed.device_communicators.rocm_comms.torch import TorchCommunicator
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
)
from vllm.utils.network_utils import get_distributed_init_method, get_open_port

# WHAT A DTYPE NAME MEANS, stated here rather than imported: two names, and the table it
# came from carried thirty more that this test has no opinion about.
D_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

logger = logging.getLogger(__name__)

set_start_method("spawn", force=True)

# Back-to-back with no inter-replay sync is what stresses an elided end barrier; a
# per-replay sync would hide the race. 200 because each replay also keeps a snapshot,
# and a stale read shows early. It is the budget for a whole GROUP: shapes captured
# together divide it rather than each taking 200.
GRAPH_REPLAYS = 200


CASE_TIMEOUT_S = 600
# How long a world's ranks get to tear their groups down before they are killed:
# `destroy_process_group` is the call that has hung on us before.
STOP_TIMEOUT_S = 60

# Deterministic per-(rank, replay) seed base for the varying-input check.
_INPUT_SEED = 20260615

# CustomAllreduce's admission, RECORDED not imported: it is the path these backends
# replace, so it is the envelope they must match. Importing it would make this test
# FOLLOW a change to that envelope silently; written out, a change upstream fails here
# and someone decides whether our backends should follow. From `custom_all_reduce.py`,
# `should_custom_ar`: inp_size % 16 == 0, is_weak_contiguous(inp), inp_size <
# self.max_size.
BASELINE_MAX_SIZE = 8 * 1024 * 1024  # CustomAllreduce's default max_size
BASELINE_ALIGNMENT = 16  # "input byte size to be multiples of 16"


# THE COMMUNICATOR'S API, and the test is a sweep over it. A name is both the collective
# to call and, with the underscores dropped, the gate to ask (`all_reduce` /
# `should_allreduce`), so `getattr` needs no table -- and renaming the API breaks this
# loudly instead of testing something else quietly.
# ALL-REDUCE AND ITS SHOTS, AND NOTHING ELSE. all_gather left the package on
# 2026-09-24: a second collective to keep correct, instantiate and measure, for
# something the decode path this work studies never calls. The gathering that
# matters is a PHASE INSIDE two-shot and is covered by two-shot's own cases.
OPS = ("all_reduce",)


def _say(rank: int, line: str) -> None:
    """Print from rank 0 only: eight ranks saying the same thing is one fact, eight
    times.

    Rank 0 is not a NEUTRAL sample, though. `one_shot_all_reduce` rotates its peer read
    order by rank, so rank 0 sums 0..world-1 -- the reference's own order -- and is the
    one rank that can be bit-exact. `run_communicator` prints the spread across ranks
    for exactly this reason.
    """
    if rank == 0:
        print(line, flush=True)


def _expected(op_name: str, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
    """What every rank should hold after `op_name` -- the one thing not derivable from
    the API.

    all_reduce sums in fp32 so the reference does not itself eat bf16 rounding.
    """
    assert op_name == "all_reduce", op_name
    acc = torch.zeros_like(inputs[0], dtype=torch.float32)
    for x in inputs:
        acc += x.to(torch.float32)
    return acc.to(inputs[0].dtype)


# The RELATIVE half of the tolerance. One value for every case, where `atol` is per (op,
# dtype) -- but PASSED alongside it rather than reached for inside `compare`, so the
# whole tolerance arrives the same way and the line can report what was actually
# applied. It has to be reported: a bf16 reduce of eight values lands ~0.125 off a
# sequential fp32 reference, so `worst|diff|` routinely exceeds `atol` on its own and a
# line quoting only `atol` reads as a failure that passed.
RTOL = 0.01


def _atol(op_name: str, dtype: torch.dtype) -> float:
    """all_reduce sums world_size values, and bf16's 7-bit mantissa (ULP ~8x fp16's)
    makes tree-vs-sequential accumulation diverge by a few ULPs -- benign, but a CORRECT
    bf16 reduce needs a dtype-aware tolerance or it reads as a failure.
    A real bug is orders of magnitude past this."""
    assert op_name == "all_reduce", op_name
    return 0.1 if dtype == torch.bfloat16 else 0.01


# The interface, all three impls and the `make_communicator` selector live in vLLM's
# `communicator.py` -- exactly what the serving path runs, so this drives that selector
# directly.


# What each name MUST construct, stated independently of the factory's if-chain: a
# branch wired to the wrong class is silent -- you ask for iris, get something else, and
# the number looks plausible. CONTROL FIRST, since the run order and the control both
# derive from this one list.
_BACKEND_CLASS = {
    "torch": TorchCommunicator,  # the known-good reference
    "hip": HipCommunicator,
    "iris": IrisCommunicator,  # someone else's kernel; expected to fail, see below
}

# WHAT WE ALREADY KNOW IS BROKEN, and why it stays in the matrix anyway. A backend left
# out of the list is a backend nobody measures; one left in and red is a suite nobody
# trusts. Naming it here is the third option: the case appears in the report, with the
# reason, and nothing is spent running it.
#
# iris serves and computes WRONG ANSWERS -- gsm8k 0.000 against baseline's 0.805 on
# 2026-09-14 -- and on 2026-09-17 it stopped serving at all, raising inside all_gather.
# It is not ours; we measure against it.
#
# SKIP AND NOT XFAIL, which is what this was. `xfail` RUNS the case: it spawns eight
# ranks, initialises NCCL and waits for the failure it already expects -- a third of
# this suite's wall clock spent confirming a dated line above. The suite gates every
# run, so that time is charged to every run.
#
# WHAT IS GIVEN UP: the day iris starts working, `xfail` would have said XPASS and skip
# says nothing. That is the trade, and it is the right way round while the thing is
# known broken. Turn it back into an xfail when there is a reason to think it passes.
DISABLED = {
    "iris": "iris computes wrong answers (gsm8k 0.000, 2026-09-14) and has since "
    "stopped serving (all_gather raises, 2026-09-17)",
}

# Control FIRST in `COMMUNICATOR_CASES`, because it is the first case to run: if torch
# is red, nothing after it means anything.
# hip as C++ picks, and each all_reduce kernel forced; the other backends have one
# kernel each. A SHOT names every op's kernel, `f"{shot}_{op}"` (the plain
# all_reduce's is the shot itself).
Shot = Literal["all_reduce_pull_one_shot", "all_reduce_pull_two_shot"]
SHOTS: tuple[Shot, ...] = get_args(Shot)
# The fast tier forces each, whatever tune would pick.
PULL_SHOTS: tuple[Shot, ...] = SHOTS
ALL_REDUCE_KERNELS: tuple[str, ...] = SHOTS
# THE FUSED OPS' SHOTS: the pull shots, and the push two-shot (a column split) that the
# norms and AttnRes have.
FusedShot = Literal[
    "all_reduce_pull_one_shot", "all_reduce_pull_two_shot", "all_reduce_push_two_shot"
]
FUSED_SHOTS: tuple[FusedShot, ...] = get_args(FusedShot)
BACKEND_KERNELS = tuple(
    (name, kernel)
    for name in _BACKEND_CLASS
    for kernel in ((None, *ALL_REDUCE_KERNELS) if name == "hip" else (None,))
)


# Enumerated rather than property-generated: a shrinking framework cannot drive across
# spawned ranks, and each candidate would be a full 8-process run.

DTYPES = ("fp16", "bf16")
# [tokens, 8192] is what vLLM hands a TP=8 all-reduce. 511/512 straddle the 8 MiB
# admission bound, where the communicator starts declining and vLLM falls back; 4088 is
# deliberately not a power of two.
SHAPES = ((4, 8192), (128, 8192), (256, 8192), (511, 8192), (512, 8192), (4088, 8192))
# Kimi-K3's rows: the latent MoE row (3584) and the hidden row (7168), at decode token
# counts and one prefill chunk. What the fast tier runs the plain all-reduce at.
KIMI_TOKENS = (1, 2, 8, 16, 2048)
KIMI_WIDTHS = (3584, 7168)
# A GROUP is the shapes one `vllm` capture covers, and a case sweeps whole groups. The
# forced-kernel group is a small and a large shape, so a forced kernel is exercised at
# both ends whatever tune would pick.
SHAPE_GROUPS: dict[str, tuple[tuple[int, int], ...]] = {
    **{f"h{w}": tuple((t, w) for t in KIMI_TOKENS) for w in KIMI_WIDTHS},
    "h7168-ends": ((16, 7168), (2048, 7168)),
    "h8192": SHAPES,
}
# Each group differs in the LEADING dimension only, asserted here because here is where
# the literals are. A group of shapes shares one set of buffers sized to the tallest and
# each reads a PREFIX (see `run_collective`), which is valid only under this. vLLM's
# capture sizes differ in tokens and share the hidden size, so the ladder is faithful
# exactly while this holds -- and at import, before a process is spawned or a card
# touched, is the cheapest place to find out that it stopped.
for _name, _group in SHAPE_GROUPS.items():
    assert len({sh[1:] for sh in _group}) == 1, (
        f"group {_name} may differ only in the leading dim: {_group}"
    )

# `vllm` is a superset of `graph`: one collective per layer, and every shape captured
# TOGETHER -- vLLM's capture-size ladder -- rather than a capture per shape.
VLLM_LAYERS = 8

# The share of one card a single case may need for its inputs and snapshots. Multi-GPU
# runs OWN the machine (coordinated beforehand), so a case may take most of a card --
# but not so much that the framework's own allocations turn a valid case into an OOM.
MEMORY_BUDGET = 0.70


@dataclass(frozen=True)
class Schedule:
    buffers: int  # distinct input buffers the collective is called on, per replay
    # shapes sharing ONE capture; 1 gives each its own, as vLLM does not, and 0 puts the
    # whole group under one, as vLLM does
    shapes: int
    replays: int  # body launches, SHARED by every shape in the group
    # recorded into cudagraphs -- one per shape -- and replayed, or run eagerly
    captured: bool

    def slots(self, admitted: int) -> int:
        """Collectives ONE shape sees, and the index space its inputs and its judging
        share.

        `replays` is the budget for the whole GROUP, so the shapes that share a capture
        share it too: adding a shape to the ladder must not multiply the snapshots,
        which is what decides whether the group fits on a card at all. Truncated, so
        every shape gets the SAME count and the
        arithmetic is identical for all of them.
        """
        return self.buffers * max(1, self.replays // admitted)


SCHEDULES = {
    "eager": Schedule(buffers=1, shapes=1, replays=1, captured=False),
    "graph": Schedule(buffers=1, shapes=1, replays=GRAPH_REPLAYS, captured=True),
    "vllm": Schedule(
        buffers=VLLM_LAYERS, shapes=0, replays=GRAPH_REPLAYS, captured=True
    ),
}
MODES = tuple(SCHEDULES)


@dataclass(frozen=True)
class Measurement:
    """What a case that RAN produced: pure numbers.

    No error field -- not because errors do not matter, but because this is where they
    do NOT live. A function that can fail returns `(value, err)`, so the error track is
    the tuple's second slot and this type is the value slot. A `failed` field here would
    give an error two homes, and the one nobody checks is the one that hides a red run.
    """

    # The ranks' own allclose verdict, STORED rather than re-derived. Deriving it as
    # `worst_diff <= atol` would be absolute-only, and a correct large-magnitude fp16
    # reduce exceeds a fixed atol through rounding alone -- so the derived form would
    # fail cases the ranks passed.
    within_tolerance: bool
    worst_diff: float
    worst_slot: int  # slot index of the worst divergence; -1 if exact
    atol: float
    rtol: float  # STORED, not read from the constant: the line reports what was applied

    def worse_of(self, other: "Measurement") -> "Measurement":
        """The worse of two, so a sweep folds without unpacking.

        Passing requires BOTH to pass -- that is the whole verdict, and it is an AND
        with no shortcut: an earlier form returned one side outright when its divergence
        was larger, which threw away the OTHER side's failure. The NUMBERS come from a
        failing side if either failed, and otherwise from the larger divergence: `atol`
        is per (op, dtype), so the bigger `worst_diff` is not always the one that broke
        its tolerance, and printing a passing cell's numbers beside a failing verdict
        names the wrong cell.
        """
        if self.within_tolerance != other.within_tolerance:
            keep = other if self.within_tolerance else self
        else:
            keep = self if self.worst_diff >= other.worst_diff else other
        return Measurement(
            within_tolerance=self.within_tolerance and other.within_tolerance,
            worst_diff=keep.worst_diff,
            worst_slot=keep.worst_slot,
            atol=keep.atol,
            rtol=keep.rtol,
        )

    def __str__(self) -> str:
        # The VERDICT, not just the numbers: `worst|diff|` alone is not readable against
        # a two-part tolerance, so a line without it looks the same whether the cell
        # passed or failed.
        at = f" @slot {self.worst_slot}" if self.worst_slot >= 0 else ""
        ok = "ok  " if self.within_tolerance else "FAIL"
        return (
            f"{ok} worst|diff|={self.worst_diff:g} atol={self.atol:g} "
            f"rtol={self.rtol:g}{at}"
        )


def _build_communicator(
    backend: str,
    cpu_group: ProcessGroup,
    device_group: ProcessGroup,
    device: torch.device,
) -> Communicator:
    comm = make_communicator(cpu_group, device_group, device, backend=backend)
    # Every test funnels through here, so this one assertion covers the mapping at every
    # world size, dtype, shape and op the suite runs -- there is no separate test to
    # remember to extend when a backend is added.
    expected = _BACKEND_CLASS.get(backend)
    if expected is None:
        raise RuntimeError(f"test does not know what backend {backend!r} should build")
    if type(comm) is not expected:
        raise RuntimeError(
            f"asked for backend {backend!r} and got {type(comm).__name__}, "
            f"expected {expected.__name__} -- the factory is wired to the wrong class"
        )
    if comm.disabled:
        raise RuntimeError(f"{backend} communicator disabled")
    return comm


# ---------------------------------------------------------------------------------
# THE SESSION'S WORLD. Every case used to spawn its own eight ranks and bring up NCCL
# and model parallel for a few milliseconds of kernels; now the ranks come up once and
# serve cases until a case leaves them in doubt.
# ---------------------------------------------------------------------------------

T = TypeVar("T")
P = ParamSpec("P")

# Every value a case hands back, named for the wire: a sweep's worst `Measurement`, a
# fused case's verdict. `run` gives the caller
# back its own case's type.
CaseValue = Measurement | bool | tuple[float, str] | None
# To a rank: the case's sequence number, the function and its arguments. None stops it.
Order = tuple[int, Callable[..., tuple[CaseValue, str | None]], tuple, dict] | None
# From a rank: the sequence number it answers, the rank, and its `(value, err)`.
Reply = tuple[int, int, tuple[CaseValue, str | None]]


@dataclass(frozen=True)
class RankContext:
    """What a case gets in each rank: who it is, its groups, and the world's
    communicators."""

    rank: int
    world: int
    device: torch.device
    cpu_group: ProcessGroup
    device_group: ProcessGroup
    # A dict in a frozen dataclass: the fields cannot be rebound, the cache still fills.
    _comms: dict[str, Communicator] = field(default_factory=dict, init=False)

    def comm(self, backend: str) -> Communicator:
        """The world's communicator for `backend`, built the first time a case asks and
        kept for the world's life, as vLLM keeps one.

        Building is a collective, so every rank must ask in the same case -- which it
        does, since every rank runs every case. A build that raises (not installed,
        disabled) is not cached: the case errors, as it did when each case built its
        own, and the error replaces the world.
        """
        if backend not in self._comms:
            self._comms[backend] = _build_communicator(
                backend, self.cpu_group, self.device_group, self.device
            )
        return self._comms[backend]

    def close(self) -> None:
        """Close every communicator this world built. Before the process group goes:
        `close` is local, but it releases IPC handles the group exchanged."""
        for comm in self._comms.values():
            comm.close()
        self._comms.clear()


def _bring_up(rank: int, world: int, init_method: str) -> RankContext:
    """ONE rank's process group and model parallel, once per world."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_distributed_environment(
        world_size=world, rank=rank, distributed_init_method=init_method
    )
    # A CONFIG CONTEXT, because `ensure_model_parallel_initialized` builds vLLM's device
    # communicators and those instantiate CustomOps, which read the current config. The
    # aiter version of this test built its groups with plain `torch.distributed` and
    # never touched `parallel_state`; the port does, and without it every rank
    # dies with "Current vLLM config is not set" before a single collective runs.
    # HERE AND NOT A FIXTURE: each rank is its own process, so a parent fixture is
    # not in scope where the config is read.
    with set_current_vllm_config(VllmConfig()):
        ensure_model_parallel_initialized(world, 1)
    cpu_group, group = get_tp_group().cpu_group, get_tp_group().device_group
    dist.all_reduce(
        torch.zeros(1).cuda(), group=group
    )  # force comm init before we measure
    torch.cuda.synchronize()
    return RankContext(rank, world, device, cpu_group, group)


def _tear_down(ctx: RankContext) -> None:
    """Communicators first, then the groups they were built on. Each in its OWN try, so
    a teardown that fails cannot stop the next one -- `destroy_process_group` is exactly
    the call that has hung on us before."""
    try:
        ctx.close()
    except BaseException:
        logger.exception("rank %d: closing the communicators failed", ctx.rank)
    try:
        if dist.is_initialized():
            destroy_model_parallel()
            destroy_distributed_environment()
        torch.cuda.empty_cache()
    except BaseException:
        logger.exception("rank %d: teardown failed", ctx.rank)


def _serve(
    rank: int,
    world: int,
    init_method: str,
    inbox: "Queue[Order]",
    outbox: "Queue[Reply]",
) -> None:
    """ONE rank of the session's world: come up once, then run every case sent until
    told to stop.

    Every rank rebuilds every rank's input from a seed, so it judges its own result and
    only scalars cross the process boundary. `err` is set when this rank failed, and it
    is a VALUE rather than an exception so the parent gets EVERY rank's verdict.
    """
    try:
        ctx = _bring_up(rank, world, init_method)
    except Exception as e:
        logger.exception("rank %d failed to come up", rank)
        outbox.put((0, rank, (None, f"did not come up: {type(e).__name__}: {e}")))
        return
    outbox.put((0, rank, (None, None)))
    # FINALLY: a rank that dies with its group still up leaves its peers waiting on a
    # socket rather than seeing a clean disconnect, turning one rank's error into
    # everyone's 600-second timeout.
    try:
        while (order := inbox.get()) is not None:
            seq, fn, args, kwargs = order
            try:
                outcome = fn(ctx, *args, **kwargs)
            except Exception as e:
                # Logged HERE, with the rank and the full traceback, before anything
                # crosses the process boundary -- then RETURNED, not re-raised, and the
                # rank keeps serving. Re-raising surfaces one rank to the parent and on
                # eight ranks the one it picks is not always the informative one.
                # `Exception`, not `BaseException`: a Ctrl-C is not this rank's verdict
                # and has to keep unwinding.
                logger.exception("rank %d failed %s %s", rank, fn.__name__, kwargs)
                outcome = (None, f"{type(e).__name__}: {e}")
            outbox.put((seq, rank, outcome))
    finally:
        _tear_down(ctx)


class World:
    """The session's ranks: started by the first case that needs them, and replaced
    after any case that leaves them in doubt.

    A case that ERRORS ends the world, even when every rank answered: a failed
    collective can leave peer state -- a flag, a registered buffer, a half-written
    slice -- that the next case would inherit and fail on for no reason of its own. A
    passing, failing-on-tolerance or declined case keeps it. A rank that hangs or dies
    ends it at once, killed rather than asked, since it will not answer.
    """

    def __init__(self, size: int) -> None:
        self.size = size
        self._procs: list[BaseProcess] = []
        self._inboxes: list[Queue[Order]] = []
        self._outbox: Queue[Reply] | None = None
        self._seq = 0

    def run(
        self,
        fn: Callable[Concatenate[RankContext, P], tuple[T, str | None]],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> list[tuple[T | None, str | None]]:
        """`fn(ctx, *args, **kwargs)` in every rank: each rank's `(value, err)`, in rank
        order. A rank that did not answer has `(None, why)`, so the caller reports it
        with the rest."""
        if not self._procs:
            failed = self._start()
            if failed is not None:
                return failed
        self._seq += 1
        for inbox in self._inboxes:
            inbox.put((self._seq, fn, args, kwargs))
        got, answered = self._collect(self._seq, stop_on_error=False)
        if not answered:
            self.stop(graceful=False)
        elif any(err is not None and err != NO_FUSED_KERNEL for _, err in got):
            self.stop(graceful=True)
        return cast(list[tuple[T | None, str | None]], got)

    def _start(self) -> list[tuple[None, str | None]] | None:
        """Spawn the ranks and wait for every one to come up. None when they did, and
        otherwise every rank's `(None, why)`."""
        spawn = mp.get_context("spawn")
        # A FRESH port per world: the worlds are sequential and each tears its group
        # down, but two runs on a shared box must not collide.
        init = get_distributed_init_method("127.0.0.1", get_open_port())
        self._outbox = spawn.Queue()
        self._inboxes = [spawn.Queue() for _ in range(self.size)]
        self._procs = [
            spawn.Process(
                target=_serve,
                args=(r, self.size, init, self._inboxes[r], self._outbox),
                name=f"rocm-comms-rank{r}",
                daemon=True,
            )
            for r in range(self.size)
        ]
        for p in self._procs:
            p.start()
        self._seq = 0
        got, answered = self._collect(0, stop_on_error=True)
        if answered and all(err is None for _, err in got):
            return None
        self.stop(graceful=False)
        return [
            (None, err if err is not None else f"rank {r} came up; the world did not")
            for r, (_, err) in enumerate(got)
        ]

    def _collect(
        self, seq: int, stop_on_error: bool
    ) -> tuple[list[tuple[CaseValue, str | None]], bool]:
        """Every rank's reply to `seq`, and whether every rank gave one.

        Under the deadline, never an unbounded wait: a deadlocked rank would otherwise
        stall the whole run with nothing printed. A rank whose process has exited is
        not waited for at all. A rank that raised has already logged its traceback in
        its own process, so a missing reply here only has to name the rank.
        """
        assert self._outbox is not None
        deadline = time.monotonic() + CASE_TIMEOUT_S
        got: dict[int, tuple[CaseValue, str | None]] = {}
        dead: list[int] = []
        while len(got) < self.size and time.monotonic() < deadline:
            try:
                s, r, outcome = self._outbox.get(timeout=1.0)
            except queue.Empty:
                dead = [
                    r
                    for r, p in enumerate(self._procs)
                    if r not in got and not p.is_alive()
                ]
                if dead:
                    break
                continue
            if s != seq:
                continue  # an answer to a case already given up on
            got[r] = outcome
            if stop_on_error and outcome[1] is not None:
                break
        n = len(got)
        for r in range(self.size):
            if r in got:
                continue
            if r in dead:
                why = f"rank {r} died (exit code {self._procs[r].exitcode})"
            elif time.monotonic() >= deadline:
                why = f"rank {r} still running after {CASE_TIMEOUT_S}s -- a deadlock"
            else:
                why = f"rank {r} not waited for: another rank failed to come up"
            got[r] = (None, f"{why}; {n} of {self.size} ranks returned")
        return [got[r] for r in range(self.size)], n == self.size

    def stop(self, graceful: bool = True) -> None:
        """End the world. GRACEFUL asks each rank to tear its groups down and kills
        whichever has not within `STOP_TIMEOUT_S`; otherwise every rank is killed
        outright, which frees the GPUs whatever state they were left in."""
        if not self._procs:
            return
        if graceful:
            for inbox in self._inboxes:
                inbox.put(None)
            deadline = time.monotonic() + STOP_TIMEOUT_S
            for p in self._procs:
                p.join(timeout=max(0.0, deadline - time.monotonic()))
        for r, p in enumerate(self._procs):
            if p.is_alive():
                if graceful:
                    logger.warning("rank %d did not stop; killing it", r)
                p.terminate()
        for p in self._procs:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
                p.join()
        # A queue a dead rank never drained would otherwise block this process's exit
        # on flushing it.
        for q in (*self._inboxes, self._outbox):
            if q is not None:
                q.cancel_join_thread()
                q.close()
        self._procs, self._inboxes, self._outbox = [], [], None


INPUT_POOL = 16


def gen_inputs(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    world: int,
    slots: int,
    device: torch.device,
) -> list[list[torch.Tensor]]:
    """THE one source of input: every rank's tensor for every slot, as `[rank][slot]`.

    A cycled POOL of `INPUT_POOL` distinct tensors, not one per slot: what a replay must
    detect is a read of the PREVIOUS slot's data, so consecutive slots differing is what
    matters, not all 400 being unique. Per-slot interface, per-pool memory -- and it is
    per SHAPE, so the whole `vllm` ladder's inputs cost 16 tensors a shape rather than
    400, which is what lets every shape run in every mode.
    """
    pool = min(slots, INPUT_POOL)
    made = [
        [_one_input(r, j, shape, dtype).to(device) for j in range(pool)]
        for r in range(world)
    ]
    return [[made[r][j % pool] for j in range(slots)] for r in range(world)]


def _one_input(
    rank: int, k: int, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    """Deterministic input for (rank, k), generated on CPU so every rank's process
    builds a bit-identical
    copy of everyone's input and can compute the reference locally -- no tensors cross
    the boundary."""
    g = torch.Generator().manual_seed(_INPUT_SEED + rank * 1_000_003 + k)
    return torch.randn(shape, generator=g).to(dtype)


def _chunks(
    shapes: Sequence[tuple[int, ...]], size: int
) -> list[list[tuple[int, ...]]]:
    """`shapes` in groups of `size`, a group being what ONE capture covers. `size=1` is
    a capture per
    shape; `size=len(shapes)` is vLLM's ladder, every size captured under one
    registration."""
    return [list(shapes[i : i + size]) for i in range(0, len(shapes), size)]


# A FORCED CALL'S KEYWORD ARGUMENTS, as every op takes them.
Forced = dict[str, str | int]


def _forced(kernel: str) -> Forced:
    """The kernel a case names (`all_reduce_push_two_shot...`) forced, by its algorithm
    and direction, at a launch every template admits (16 blocks of 512 threads), its
    own tile, for a case that forces one only to check it."""
    return {
        "algorithm": "two_shot" if "two_shot" in kernel else "one_shot",
        "direction": "push" if "_push_" in kernel else "pull",
        "threads_per_block": 512,
        "blocks_per_grid": 16,
    }


def declined(
    comm: Communicator,
    op_name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    forced: Forced | None,
) -> str | None:
    """Why the communicator will not take `shape`, or None if it will.

    A bare Optional, NOT `(value, err)`: there is no value, and a refusal is the ANSWER
    rather than the failure of one -- `(512, 8192)` is in the list precisely to be
    refused. `(value, err)` is for a function that was asked to produce something and
    could not; this one was asked a question.
    """
    one = torch.empty(shape, dtype=dtype)
    should = getattr(comm, f"should_{op_name.replace('_', '')}")
    if not should(one, **(forced or {})):
        return "declined by the communicator"
    return None


def precheck(
    op_name: str,
    shapes: Sequence[tuple[int, ...]],
    dtype: torch.dtype,
    world: int,
    sched: Schedule,
    budget: int,
) -> tuple[None, str | None]:
    """`(None, why)` if this group will not run, `(None, None)` if it will. Not a
    failure either way.

    ALWAYS the tuple, even with no value to return: the pair IS the signal that a
    function can fail. A bare `-> Optional[str]` cannot say that -- a reader has to open
    the docstring to learn the string is an error rather than a result. Python has no
    void, so `None` fills the value slot. Contrast `declined` above, where the string is
    the answer and the bare Optional is right.

    A GUARD, not a contract: `need` mirrors what the steps allocate -- per shape a
    snapshot per slot, a pool each for inputs and expectations and one live output set
    per graph, plus ONE set of statics shared by the whole group -- so it duplicates
    their knowledge and can go stale. Under-counting OOMs, over-counting skips a group
    that would have fit; counting two of the terms is how it once passed a cell that
    then OOM'd.
    """
    item = torch.empty(0, dtype=dtype).element_size()
    probe = torch.empty(shapes[0], dtype=dtype)
    fan = _expected(op_name, [probe] * world).numel() // probe.numel()
    slots = sched.slots(len(shapes))
    pool = min(slots, INPUT_POOL)
    need = sum(
        item
        * math.prod(sh)
        * (slots * fan + pool * world + pool * fan + sched.buffers * fan)
        for sh in shapes
    )
    # The statics are SHARED by the group -- one set sized to the tallest shape, which
    # every other shape reads a prefix of. See `run_collective`.
    need += sched.buffers * item * math.prod(max(shapes, key=lambda sh: sh[0]))
    if need > budget:
        return None, f"needs {need / 2**30:.0f}G, budget {budget / 2**30:.0f}G"
    return None, None


def run_collective(
    comm: Communicator,
    op_name: str,
    mine: Sequence[Sequence[torch.Tensor]],
    sched: Schedule,
    forced: Forced | None,
) -> list[list[torch.Tensor]]:
    """One output per (shape, SLOT): `mine[s][j]` in, the result out. Slot j is replay
    `j // buffers`
    on buffer `j % buffers`.

    The shapes of one call SHARE their input buffers -- `buffers` allocations sized to
    the tallest, each shape's launch reading a PREFIX, which `SHAPES` is asserted to
    permit. That is vLLM's arrangement rather than a saving: its capture sizes are
    slices of one persistent activation buffer, so every graph records a launch on the
    SAME address with a different length. Allocating per shape instead would test
    something vLLM never does, and would hide the case where one address is registered
    once per graph.

    The graphs are LOCALS and die here; a live one makes `destroy_process_group` block
    forever. The warmup runs EAGER on the same communicator, which registers the
    statics. Only a snapshot sits between replays, so they stay back-to-back; an elided
    end barrier needs that to race.
    """
    shapes = [tuple(m[0].shape) for m in mine]
    n = len(shapes)
    each = (
        sched.slots(n) // sched.buffers
    )  # replays THIS shape gets, the same for all of them
    ref = mine[0][0]
    statics = [
        torch.zeros(
            (max(sh[0] for sh in shapes), *shapes[0][1:]),
            dtype=ref.dtype,
            device=ref.device,
        )
        for _ in range(sched.buffers)
    ]
    # `partial` on the API itself, over the prefix VIEWS. Nothing re-checks admission:
    # the communicator enforces its own envelope and raises, so a check here would
    # restate what the callee guarantees.
    ops = [
        [
            partial(getattr(comm, op_name), st[: sh[0]], **(forced or {}))
            for st in statics
        ]
        for sh in shapes
    ]

    def body(s: int) -> list[torch.Tensor]:
        # Each op returns its output and what ran.
        return [op()[0] for op in ops[s]]

    def feed(s: int, replay: int) -> None:
        for m, st in enumerate(statics):
            st[: shapes[s][0]].copy_(mine[s][replay * sched.buffers + m])

    if not sched.captured:
        out: list[list[torch.Tensor]] = [[] for _ in shapes]
        for k in range(each * n):
            s, replay = k % n, k // n
            feed(s, replay)
            out[s] += [o.clone() for o in body(s)]
        return out

    for _ in range(
        3
    ):  # eager warmup: first-call allocations, and the statics' registration
        for s in range(n):
            body(s)
    torch.cuda.synchronize()
    # ONE capture context over EVERY graph, the way vLLM's covers every capture size:
    # the deferred registration then happens once for all of them, which is the path a
    # real startup takes and the only place a registration wider than a single graph is
    # exercised. The graphs are NOT given a shared pool -- what the communicator sees
    # are the statics, allocated before any capture, so sharing would change torch's
    # bookkeeping and nothing the communicator observes.
    graphs = [torch.cuda.CUDAGraph() for _ in shapes]
    outs = []
    with comm.capture():
        for s, g in enumerate(graphs):
            with torch.cuda.graph(g):
                outs.append(body(s))
    snaps = [
        [torch.empty((each, *o.shape), dtype=o.dtype, device=o.device) for o in outs[s]]
        for s in range(n)
    ]
    for k in range(each * n):
        s, replay = k % n, k // n
        feed(s, replay)
        # ROUND-ROBIN over the shapes, not one graph to exhaustion: vLLM replays
        # whichever graph the incoming batch matches. Alternating is what shows each
        # graph kept its OWN peer pointers, and it puts other shapes' work between a
        # shape's consecutive replays -- a stale read has to survive that to go
        # unnoticed.
        graphs[s].replay()
        for m, o in enumerate(outs[s]):
            snaps[s][m][replay].copy_(o)
        if n > 1 and k == (each * n) // 2:
            # An EAGER collective mid-replay, result DISCARDED. vLLM falls back to eager
            # for a batch no graph matches, on this same communicator and after its
            # graph buffers are registered. What is checked is not this result but that
            # every replay after it is still right -- so it belongs to the mode that
            # models vLLM, and `graph` stays the clean isolator.
            body(s)
    torch.cuda.synchronize()
    return [
        [snaps[s][m][replay] for replay in range(each) for m in range(sched.buffers)]
        for s in range(n)
    ]


def expected_outputs(
    op_name: str, inputs: Sequence[Sequence[torch.Tensor]], slots: int
) -> list[torch.Tensor]:
    """What this rank should hold after each slot: the collective applied to every
    rank's input.

    POOLED like the inputs, and for the same reason -- only `INPUT_POOL` inputs are
    distinct, so only that many answers are. Building one per slot would have cost as
    much as the snapshots (100 GiB at `vllm`'s largest cell, on top of the snapshots'
    100) and OOM'd inside the memory budget.
    """
    pool = min(slots, INPUT_POOL)
    made = [
        _expected(op_name, [inputs[r][j] for r in range(len(inputs))])
        for j in range(pool)
    ]
    return [made[j % pool] for j in range(slots)]


def compare(
    got: Sequence[torch.Tensor],
    expected: Sequence[torch.Tensor],
    atol: float,
    rtol: float,
) -> Measurement:
    """Every slot against its OWN expectation.

    allclose (atol + rtol*|ref|), not absolute-only: a correct large-magnitude fp16
    reduce exceeds a fixed 0.01 through rounding alone, which the torch control caught.
    Diverging at slot 0 means capture is wrong, at 1+ means staleness between replays.
    """
    ok, worst_diff, worst_slot = True, 0.0, -1
    for j, (mine, ref) in enumerate(zip(got, expected)):
        a, b = mine.to(torch.float32), ref.to(torch.float32)
        d = (a - b).abs().max().item()
        if d > worst_diff:
            worst_diff, worst_slot = d, j
        if not torch.allclose(a, b, atol=atol, rtol=rtol):
            ok = False
    return Measurement(
        within_tolerance=ok,
        worst_diff=worst_diff,
        worst_slot=worst_slot,
        atol=atol,
        rtol=rtol,
    )


def exercise(
    ctx: RankContext,
    backend: str,
    mode: str,
    groups: Sequence[str],
    dtypes: Sequence[str],
    kernel: str | None = None,
) -> tuple[Measurement | None, str | None]:
    """Exercise the world's communicator for `backend` over its whole API. `(worst
    verdict, None)`, or `(None, why nothing was measured)`.

    TWO LEVELS of error track, carrying different things. A CELL's `err` means that cell
    did not run -- declined, or over budget -- which is not a failure and does not stop
    the sweep. This function's `err` means the sweep produced NO measurement at all. A
    cell that RAISES is neither: it is a real failure and it propagates, to be logged
    with its traceback and turned into an error track by `_serve`.

    The communicator is the WORLD's, not this case's: built on the first case that asks
    and closed when the world ends, as vLLM holds one for its whole life. Construction
    and teardown are still two of the ways a communicator fails -- building one is a
    collective, and teardown releases IPC handles -- and they still run, once per world.
    A declined shape, or a group past the memory budget, is reported and skipped:
    declining is correct behaviour. Each group is scoped so its tensors die with the
    frame.

    A GROUP is the shapes one capture covers -- one shape for `eager` and `graph`, all
    of them for `vllm`. They run together and are JUDGED APART, so the log keeps a line
    per (op, dtype, shape) whichever mode produced it.
    """
    rank, world, device = ctx.rank, ctx.world, ctx.device
    sched = SCHEDULES[mode]
    budget = int(torch.cuda.get_device_properties(device).total_memory * MEMORY_BUDGET)
    worst = Measurement(
        within_tolerance=True, worst_diff=0.0, worst_slot=-1, atol=0.0, rtol=0.0
    )
    ran = 0
    forced = None if kernel is None else _forced(kernel)
    comm = ctx.comm(backend)
    for op_name, dtype_name, group in product(OPS, dtypes, groups):
        dtype = D_DTYPES[dtype_name]
        atol = _atol(op_name, dtype)
        group_shapes = SHAPE_GROUPS[group]
        for chunk in _chunks(group_shapes, sched.shapes or len(group_shapes)):
            # Admission FIRST and per shape, because a group is what the capture
            # covers: a refused shape is dropped from it, not a reason to skip the
            # ones that were admitted.
            shapes = []
            for sh in chunk:
                why = declined(comm, op_name, sh, dtype, forced)
                if why is None:
                    shapes.append(sh)
                else:
                    _say(
                        rank,
                        f"      - {op_name:11} {dtype_name:5} {str(sh):12} {why}",
                    )
            if not shapes:
                continue

            # THE LOOP VARIABLES ARE BOUND AT DEFINITION, as defaults. `cell` is
            # called immediately below so late binding cannot bite today, but a
            # closure over a loop variable is one edit away from doing so.
            def cell(
                op_name: str = op_name,
                shapes: Sequence[tuple[int, int]] = shapes,
                dtype: torch.dtype = dtype,
                atol: float = atol,
            ) -> tuple[list[Measurement] | None, str | None]:
                # THE ERROR TRACK is the second element, and it is the ONLY thing we
                # test: `err is not None` means we have an error. Never the value --
                # when `err` is set the value slot is not to be read.
                _, err = precheck(op_name, shapes, dtype, world, sched, budget)
                if err is not None:
                    return None, err
                slots = sched.slots(len(shapes))
                inputs = [gen_inputs(sh, dtype, world, slots, device) for sh in shapes]
                got = run_collective(
                    comm, op_name, [i[rank] for i in inputs], sched, forced
                )
                expected = [expected_outputs(op_name, i, slots) for i in inputs]
                return [compare(g, e, atol, RTOL) for g, e in zip(got, expected)], None

            # Same check, same track: `err is not None` means an error, and the
            # value is only read once we know there was none.
            got, err = cell()
            if err is not None:
                _say(
                    rank,
                    f"      - {op_name:11} {dtype_name:5} "
                    f"{len(shapes)} shapes together  {err}",
                )
                continue
            for sh, m in zip(shapes, got):
                _say(rank, f"      {op_name:11} {dtype_name:5} {str(sh):12} {m}")
                worst = worst.worse_of(m)
                ran += 1
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    # COUNTED, because the seed verdict passes: a communicator that declined everything
    # would otherwise fold to `ok worst|diff|=0` and report green having measured
    # nothing.
    if ran == 0:
        return (
            None,
            "no cell ran -- every one was declined or over budget, so nothing was "
            "measured",
        )
    return worst, None


def run_communicator(
    ranks: World,
    backend: str,
    mode: str,
    groups: Sequence[str],
    dtypes: Sequence[str],
    kernel: str | None = None,
) -> tuple[Measurement | None, str | None]:
    """Run `exercise` in every rank, fold to the WORST rank's numbers -- a collective's
    bug is often visible on only a subset, so it passes only if EVERY rank passed."""
    per_rank = ranks.run(
        exercise,
        backend=backend,
        mode=mode,
        groups=groups,
        dtypes=dtypes,
        kernel=kernel,
    )

    # EVERY failing rank, not the first: they usually fail for one reason, and the ranks
    # that did NOT fail are half of what a count like `[1,1,1,0,0,1,1,1]` tells you.
    failed = [f"rank {r}: {e}" for r, (_, e) in enumerate(per_rank) if e is not None]
    if failed:
        return None, "; ".join(failed)

    ms = [cast(Measurement, m) for m, _ in per_rank]
    # The SPREAD, whenever the ranks disagree. The cell lines above come from rank 0
    # only, and for `hip` that is the rank whose summation order matches the reference
    # -- so its `worst|diff|=0` says nothing about the other seven. A rank-rotated read
    # order (see `one_shot_all_reduce`) puts them within an ULP rather than bitwise, and
    # printing it is what stops the next reader chasing the gap between a cell line and
    # this fold as if it were a bug.
    if len({m.worst_diff for m in ms}) > 1:
        print(
            "      per-rank worst|diff|: "
            + "  ".join(f"{r}:{m.worst_diff:g}" for r, m in enumerate(ms)),
            flush=True,
        )
    worst = ms[0]
    for m in ms[1:]:
        worst = worst.worse_of(m)
    return worst, None


@pytest.fixture(scope="session")
def world() -> int:
    """Every GPU on the box. A read of the machine, so it is a fixture rather than a
    constant."""
    return torch.cuda.device_count()


@pytest.fixture(scope="session")
def ranks(world: int) -> Iterator[World]:
    """The session's world. Nothing is spawned until a case runs, so a session whose
    cases all skip or are deselected never pays for one."""
    ranks = World(world)
    try:
        yield ranks
    finally:
        ranks.stop()


# One of each STATE variant: get_args(State) is every variant, so a new one fails here
# until it has an instance.
_A_STATE: dict[type, State] = {
    Disabled: Disabled("not supported"),
    Open: Open(),
    Capturing: Capturing(),
    Broken: Broken("a capture raised"),
    Closed: Closed(),
}


@pytest.mark.parametrize("variant", get_args(State))
def test_capture_only_from_open(variant: type) -> None:
    """A capture is entered from Open only; from any other state it raises before
    touching the backend. Every variant, so a new one is covered when it is added."""
    comm = object.__new__(TorchCommunicator)
    state = _A_STATE[variant]
    comm.state = state
    if isinstance(state, Open):
        with comm.capture():
            assert isinstance(comm.state, Capturing)
        assert isinstance(comm.state, Open)
        return
    with pytest.raises(RuntimeError), comm.capture():
        pass
    assert comm.state == state


def test_a_failed_capture_leaves_it_broken() -> None:
    """A capture that raises leaves the communicator broken, and every call after
    raises (through `disabled`, which every op checks first) naming why, rather than
    falling back while its peers hold half a registration."""
    comm = object.__new__(TorchCommunicator)
    comm.state = Open()
    with pytest.raises(ValueError), comm.capture():
        raise ValueError("graph capture failed")
    assert isinstance(comm.state, Broken)
    with pytest.raises(RuntimeError, match="graph capture failed"):
        _ = comm.disabled
    with pytest.raises(RuntimeError):
        comm.capture().__enter__()


# FULL: it spawns no ranks, but it checks a rule that moves rarely, against a baseline
# that moves never.
@pytest.mark.full
def test_admission_matches_the_baseline() -> None:
    """Every backend admits exactly what vLLM's CustomAllreduce does -- the precondition
    for the matrix
    meaning anything, since arms that route different work compare routing rather than
    kernels. Needs no GPU. fp32 is excluded: `hip_comms.cu` has fp16/bf16 instantiations
    only, so closing it needs a kernel.
    """
    ours = object.__new__(TorchCommunicator)
    ours.state = Open()
    # Every power of two across the range PLUS the bound and one element either side.
    # Bounds alone are the edges of the rule AS IT IS, so a wrong rule that diverges in
    # the band between two of them shows up on neither: a bounds-only grid missed a real
    # bound at world 2 and 4.
    edges = tuple(1 << k for k in range(4, 28)) + (BASELINE_MAX_SIZE,)
    for dtype in (torch.float16, torch.bfloat16):
        es = torch.empty(0, dtype=dtype).element_size()
        for nbytes in sorted({e + d for e in edges for d in (-es, 0, es) if e + d > 0}):
            if nbytes % es:
                continue
            t = torch.empty(nbytes // es, dtype=dtype)
            where = f"{dtype} {nbytes}B"
            aligned = nbytes % BASELINE_ALIGNMENT == 0
            # THE ONE TERM LEFT IN THE ENVELOPE IS ALIGNMENT. Size refuses nothing,
            # and the dtype grid here is fp16/bf16, so what `should_*` still refuses is
            # a tensor the kernel cannot vectorise: `vec` is 16 bytes wide. The grid
            # REACHES those on purpose: an edge +/- one element is 14 and 18 bytes off
            # the 16-byte edge.
            assert ours.should_allreduce(t) is aligned, (
                f"all_reduce admission is not the alignment rule: {where}"
            )


def _communicator_param(
    backend: str,
    kernel: str | None,
    mode: str,
    dtypes: tuple[str, ...],
    groups: tuple[str, ...],
    full: bool,
) -> ParameterSet:
    """One `test_communicator` case, marked with its tier and, for a backend known to
    be broken, the reason it is skipped."""
    name = backend if kernel is None else f"{backend}-{kernel}"
    marks = [pytest.mark.full] if full else []
    if backend in DISABLED:
        marks.append(pytest.mark.skip(reason=DISABLED[backend]))
    return pytest.param(
        backend,
        kernel,
        mode,
        dtypes,
        groups,
        id=f"{name}-{mode}-{'+'.join(dtypes)}-{'+'.join(groups)}",
        marks=marks,
    )


# CHOSEN, not gridded. FAST is what vLLM runs: hip with no forced launch (select
# picks), captured the way vLLM captures, at Kimi-K3's widths in bf16; and each PULL
# kernel forced at a small and a large shape, so both code paths run whatever tune
# picks. FULL is the rest: the control and iris, every forced kernel and every mode on
# the 8192 sweep with its admission boundaries, and hip's own choice in fp16 and eager
# at Kimi-K3's widths. Control FIRST: the list's order is the run order, and if torch
# is red, nothing after it means anything.
_KIMI_GROUPS = tuple(f"h{w}" for w in KIMI_WIDTHS)
COMMUNICATOR_CASES = (
    *(
        _communicator_param("torch", None, mode, DTYPES, ("h8192",), full=True)
        for mode in MODES
    ),
    *(
        _communicator_param("hip", None, mode, ("bf16",), (group,), full=False)
        for mode in ("graph", "vllm")
        for group in _KIMI_GROUPS
    ),
    *(
        _communicator_param(
            "hip", kernel, "graph", ("bf16",), ("h7168-ends",), full=False
        )
        for kernel in PULL_SHOTS
    ),
    *(
        _communicator_param("hip", None, mode, ("fp16",), _KIMI_GROUPS, full=True)
        for mode in ("graph", "vllm")
    ),
    _communicator_param("hip", None, "eager", DTYPES, _KIMI_GROUPS, full=True),
    *(
        _communicator_param(backend, kernel, mode, DTYPES, ("h8192",), full=True)
        for backend, kernel in BACKEND_KERNELS
        if backend != "torch"
        for mode in MODES
    ),
)


@pytest.mark.parametrize(
    ("backend", "kernel", "mode", "dtypes", "groups"), COMMUNICATOR_CASES
)
def test_communicator(
    backend: str,
    kernel: str | None,
    mode: str,
    dtypes: tuple[str, ...],
    groups: tuple[str, ...],
    world: int,
    ranks: World,
) -> None:
    """Does this communicator work? One instance, its whole API, in one mode.

    Not parameterised per collective or shape -- `exercise` sweeps those on ONE
    communicator, as vLLM does, and prints each cell so a failure names itself. BACKEND
    is outermost so the control runs first; MODE is the ladder that localises a `vllm`
    failure to the pattern rather than the arithmetic.
    """
    if world < 2:
        pytest.skip("a collective needs at least two ranks")
    name = backend if kernel is None else f"{backend}-{kernel}"
    print(f"\n  {name} / {mode} / {'+'.join(dtypes)} / {'+'.join(groups)}", flush=True)
    got, err = run_communicator(ranks, backend, mode, groups, dtypes, kernel=kernel)
    # THE ERROR TRACK, checked explicitly against None: set means the case produced no
    # measurement at all, and `got` is not to be read. Only then is there a verdict to
    # assert on.
    if err is not None:
        pytest.fail(f"{name}/{mode}: {err}")
    print(f"      => {got}", flush=True)
    assert got.within_tolerance, f"{name}/{mode}: outside tolerance (cells above)"


# ---------------------------------------------------------------------------------
# THE FUSED SHOTS. Not in `OPS` above: each takes more than one tensor, and what it
# must be judged against is THE TWO OPS A FUSION PASS REPLACED, not a reference written
# here. So they get their own runner, compared against `vllm.ir.ops` itself.
# ---------------------------------------------------------------------------------

# What "no kernel" looks like coming back from a rank, so the parent can SKIP rather
# than pass: the fallback IS the reference, and would pass forever.
NO_FUSED_KERNEL = "no fused kernel"

FUSED_EPS = 1e-5
# all_reduce -> rms_norm, and all_reduce -> fused_add_rms_norm.
FORMS = ("rms_norm", "add_rms_norm")
# Kimi-K3's latent MoE row (3584) and hidden row (7168), then the sweep's 8192. Four
# rows is fewer than the ranks, which two-shot must still get right. The full tier's.
FUSED_SHAPES = (
    (4, 3584),
    (128, 3584),
    (4, 7168),
    (128, 7168),
    (4088, 7168),
    (512, 8192),
)
# The fast tier's: Kimi-K3's decode rows at both widths. One row is fewer than the
# ranks too.
FUSED_FAST_SHAPES = ((1, 3584), (16, 3584), (16, 7168))


def _shape_id(shape: tuple[int, ...]) -> str:
    return "x".join(map(str, shape))


def _fused_reference(
    form: str,
    inputs: Sequence[torch.Tensor],
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The all-reduce, landed in the input dtype as an all-reduce would land it, then
    the exact op the pass rewrote away."""
    import vllm.ir.ops

    acc = torch.zeros_like(inputs[0], dtype=torch.float32)
    for x in inputs:
        acc += x.to(torch.float32)
    summed = acc.to(inputs[0].dtype)
    if form == "rms_norm":
        return vllm.ir.ops.rms_norm(summed, weight, FUSED_EPS), None
    return vllm.ir.ops.fused_add_rms_norm(summed, residual, weight, FUSED_EPS)


def _fused_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    """`(atol, rtol)` for the normed output. The kernel matches the reference's
    roundings; what is left is the sum's order, which the rank rotation changes by an
    ULP of the input dtype, and the norm carries through."""
    return (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-2, 2e-3)


def run_fused_rank(
    ctx: RankContext,
    form: str,
    shape: tuple[int, int],
    dtype_name: str,
    shot: FusedShot,
    weight_dtype: torch.dtype | None = None,
) -> tuple[bool, str | None]:
    """ONE rank: run the fused op and the two ops it replaces, and say whether they
    agree. `weight_dtype` None is the input's dtype. Returns `(agreed, err)`; `err`
    is `NO_FUSED_KERNEL` when this backend has none, which is a skip and not a
    failure."""
    rank, world, device = ctx.rank, ctx.world, ctx.device
    dtype = D_DTYPES[dtype_name]
    # Every rank rebuilds every rank's input from the seed, as the sweep above does,
    # so the reference is computed locally and no tensor crosses a process boundary.
    inputs = [_one_input(r, 0, shape, dtype) for r in range(world)]
    residual = _one_input(world, 1, shape, dtype)
    weight = _one_input(world + 1, 2, (shape[1],), dtype).to(weight_dtype or dtype)
    forced = _forced(f"{shot}_{form}")
    comm = ctx.comm("hip")
    mine = inputs[rank].to(device)
    if form == "rms_norm":
        if not comm.should_allreduce_rms_norm(mine, weight, **forced):
            return False, NO_FUSED_KERNEL
        got, _ = comm.all_reduce_rms_norm(mine, weight.to(device), FUSED_EPS, **forced)
        got_residual = None
    else:
        if not comm.should_allreduce_add_rms_norm(mine, weight, **forced):
            return False, NO_FUSED_KERNEL
        got, got_residual, _ = comm.all_reduce_add_rms_norm(
            mine,
            residual.to(device),
            weight.to(device),
            FUSED_EPS,
            **forced,
        )
    want, want_residual = _fused_reference(form, inputs, residual, weight)
    atol, rtol = _fused_tolerance(dtype)
    checks = [("out", got.cpu(), want, atol)]
    if got_residual is not None:
        # The residual is the raw sum plus the incoming residual, unnormalised, so
        # it gets the raw reduce's tolerance and not the normed one.
        checks.append(
            (
                "residual",
                got_residual.cpu(),
                want_residual,
                _atol("all_reduce", dtype),
            )
        )
    for name, a, b, tol in checks:
        worst = (a.to(torch.float32) - b.to(torch.float32)).abs().max().item()
        if not torch.allclose(
            a.to(torch.float32), b.to(torch.float32), atol=tol, rtol=rtol
        ):
            return False, f"{name} differs: worst|diff|={worst:.4g} atol={tol}"
    return True, None


def _fused_param(
    form: str, shape: tuple[int, int], dtype_name: str, shot: FusedShot, full: bool
) -> ParameterSet:
    return pytest.param(
        form,
        shape,
        dtype_name,
        shot,
        id=f"{form}-{_shape_id(shape)}-{dtype_name}-{shot}",
        marks=[pytest.mark.full] if full else [],
    )


# FAST: the pull kernels at Kimi-K3's decode shapes in bf16. FULL: both kernels of each
# direction at every shape, forced: what C++ would pick is one of them.
FUSED_CASES = (
    *(
        _fused_param(form, shape, "bf16", shot, full=False)
        for form in FORMS
        for shape in FUSED_FAST_SHAPES
        for shot in FUSED_SHOTS
    ),
    *(
        _fused_param(form, shape, dtype_name, shot, full=True)
        for form in FORMS
        for shape in FUSED_SHAPES
        for dtype_name in DTYPES
        for shot in FUSED_SHOTS
    ),
)


@pytest.mark.parametrize(("form", "shape", "dtype_name", "shot"), FUSED_CASES)
def test_all_reduce_rms_norm_matches_the_two_ops_it_replaces(
    form: str,
    shape: tuple[int, int],
    dtype_name: str,
    shot: FusedShot,
    world: int,
    ranks: World,
) -> None:
    """Each fused op against all_reduce then the `vllm.ir.ops` norm it replaces.

    Enumerated, not property-generated: a shrinking framework cannot drive across
    spawned ranks, and each candidate is a full eight-process run.
    """
    _fused_case(form, shape, dtype_name, shot, None, world, ranks)


# AN FP32 WEIGHT, in its own dtype: the reference rounds the normed row to the WEIGHT's
# dtype, so this is a different rounding from a weight in the input's, not the same one
# with a cast. Kimi-K3's latent row and its hidden row, both kernels: the rounding is
# per element, so more shapes would say nothing new. FAST is one case per form, the
# pull one-shot at the latent row in bf16; FULL is the rest.
FP32_WEIGHT_FAST = ((4, 3584), "bf16", "all_reduce_pull_one_shot")
FP32_WEIGHT_CASES = tuple(
    _fused_param(
        form,
        shape,
        dtype_name,
        shot,
        full=(shape, dtype_name, shot) != FP32_WEIGHT_FAST,
    )
    for form in FORMS
    for shape in ((4, 3584), (128, 7168))
    for dtype_name in DTYPES
    for shot in FUSED_SHOTS
)


@pytest.mark.parametrize(("form", "shape", "dtype_name", "shot"), FP32_WEIGHT_CASES)
def test_all_reduce_rms_norm_takes_an_fp32_weight(
    form: str,
    shape: tuple[int, int],
    dtype_name: str,
    shot: FusedShot,
    world: int,
    ranks: World,
) -> None:
    _fused_case(form, shape, dtype_name, shot, torch.float32, world, ranks)


def run_weight_plan_rank(ctx: RankContext) -> tuple[bool, str | None]:
    """ONE rank: a norm's weight in neither the call's dtype nor fp32 is C++'s refusal,
    `weight_not_built`; in either, the op runs."""
    comm = ctx.comm("hip")
    x = torch.randn(16, 7168, device=ctx.device).to(torch.bfloat16)
    for add in (False, True):
        for dtype, want in (
            (torch.float16, Error.weight_not_built),
            (torch.bfloat16, None),
            (torch.float32, None),
        ):
            weight = torch.ones(7168, dtype=dtype, device=ctx.device)
            got: Error | None = None
            try:
                if add:
                    comm.all_reduce_add_rms_norm(x, torch.zeros_like(x), weight, 1e-5)
                else:
                    comm.all_reduce_rms_norm(x, weight, 1e-5)
            except Refused as refused:
                got = refused.error
            if got is not want:
                return False, f"add={add} with a {dtype} weight: {got}, not {want}"
    return True, None


def test_a_norm_weight_c_does_not_build_is_refused(world: int, ranks: World) -> None:
    """The weight's dtype is part of the call C++ plans: the rule that a norm's weight
    is the call's dtype or fp32 is C++'s alone."""
    # example-based: the three weight dtypes against one input dtype are the whole rule
    if world < 2:
        pytest.skip("a collective needs at least two ranks")
    bad = [err for _, err in ranks.run(run_weight_plan_rank) if err is not None]
    assert not bad, "; ".join(bad)


def _fused_case(
    form: str,
    shape: tuple[int, int],
    dtype_name: str,
    shot: FusedShot,
    weight_dtype: torch.dtype | None,
    world: int,
    ranks: World,
) -> None:
    """One fused case across every rank, judged by `run_fused_rank`."""
    if world < 2:
        pytest.skip("a collective needs at least two ranks")
    got = ranks.run(
        run_fused_rank,
        form=form,
        shape=shape,
        dtype_name=dtype_name,
        shot=shot,
        weight_dtype=weight_dtype,
    )
    where = f"{form} {shot} {shape} {dtype_name} weight={weight_dtype or dtype_name}"
    if any(err == NO_FUSED_KERNEL for _, err in got):
        pytest.skip(f"hip has no fused all_reduce_{form} for {where}")
    bad = [err for _, err in got if err is not None]
    assert not bad, f"{where}: " + "; ".join(bad)
    assert all(agreed for agreed, _ in got), f"{where}: ranks disagreed"


# ---------------------------------------------------------------------------------
# ALL-REDUCE + ATTNRES, judged against the model's own Triton `attn_res` applied to the
# all-reduced sum: the op the fusion replaces, not a reference written here.
# ---------------------------------------------------------------------------------

# (shape, has_prefix, num_blocks, write_idx, output_norm). Example-based: each case is a
# full eight-process run, so the cases are the ones Kimi-K3 runs -- decode rows, 0 to 9
# stored blocks, the block-write layer where the sum starts the prefix, with and without
# the output norm -- plus prefill-sized row counts, a block-write one among them, that
# the two-shot kernel is for. Each case runs on every kernel.
ADD_ATTN_RES_RMS_NORM_CASES = (
    ((4, 7168), True, 0, -1, True),
    ((16, 7168), True, 4, -1, True),
    ((16, 7168), True, 9, -1, False),
    ((16, 7168), False, 4, 4, True),
    ((128, 7168), True, 9, -1, True),
    ((100, 7168), False, 4, 4, True),
)
ATTN_RES_SOURCES = 10
# FAST: the decode cases on the kernels decode runs, the pull one-shot and the push
# two-shot. FULL: the prefill-sized cases, and every case on the pull two-shot.
ATTN_RES_FAST_MAX_ROWS = 16
ATTN_RES_PARAMS = tuple(
    pytest.param(
        case,
        shot,
        id=f"case{i}-{shot}",
        marks=[]
        if shot != "all_reduce_pull_two_shot" and case[0][0] <= ATTN_RES_FAST_MAX_ROWS
        else [pytest.mark.full],
    )
    for i, case in enumerate(ADD_ATTN_RES_RMS_NORM_CASES)
    for shot in FUSED_SHOTS
)


def run_add_attn_res_rms_norm_rank(
    ctx: RankContext,
    case: tuple[tuple[int, int], bool, int, int, bool],
    shot: FusedShot,
) -> tuple[bool, str | None]:
    """ONE rank: the fused op against the two it replaces, on every output it writes."""
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res

    rank, world, device = ctx.rank, ctx.world, ctx.device
    shape, has_prefix, num_blocks, write_idx, output_norm = case
    dtype = torch.bfloat16
    rows, hidden = shape
    inputs = [_one_input(r, 0, shape, dtype) for r in range(world)]
    prefix = _one_input(world, 1, shape, dtype)
    blocks = torch.stack(
        [_one_input(world + 2 + s, 3, shape, dtype) for s in range(ATTN_RES_SOURCES)],
        dim=1,
    ).contiguous()
    norm_w = _one_input(world + 20, 4, (hidden,), dtype)
    qk_w = _one_input(world + 21, 5, (hidden,), dtype)
    out_w = _one_input(world + 22, 6, (hidden,), dtype) if output_norm else None

    # THE REFERENCE: the sum as an all-reduce lands it, then the model's kernel.
    acc = torch.zeros(shape, dtype=torch.float32)
    for x in inputs:
        acc += x.to(torch.float32)
    summed = acc.to(dtype).to(device)
    ref_prefix = prefix.to(device).clone() if has_prefix else summed.clone()
    ref_blocks = blocks.to(device).clone()
    want = attn_res(
        ref_prefix,
        summed if has_prefix else None,
        ref_blocks,
        norm_w.to(device),
        qk_w.to(device),
        None if out_w is None else out_w.to(device),
        num_blocks,
        write_idx,
        1e-6,
        1e-5,
    )
    torch.cuda.synchronize()

    forced = _forced(f"{shot}_add_attn_res_rms_norm")
    comm = ctx.comm("hip")
    mine = inputs[rank].to(device)
    if not comm.should_allreduce_add_attn_res_rms_norm(mine, **forced):
        return False, NO_FUSED_KERNEL
    got_blocks = blocks.to(device).clone()
    got_prefix, got, _ = comm.all_reduce_add_attn_res_rms_norm(
        mine,
        prefix.to(device).clone() if has_prefix else None,
        got_blocks,
        norm_w.to(device),
        qk_w.to(device),
        None if out_w is None else out_w.to(device),
        num_blocks,
        write_idx,
        1e-6,
        1e-5,
        **forced,
    )
    torch.cuda.synchronize()
    atol, rtol = _fused_tolerance(dtype)
    sum_tol = _atol("all_reduce", dtype)
    checks = [
        ("out", got, want, atol),
        ("prefix", got_prefix, ref_prefix, sum_tol),
        ("blocks", got_blocks, ref_blocks, sum_tol),
    ]
    for name, a, b, tol in checks:
        a32, b32 = a.float().cpu(), b.float().cpu()
        if not torch.allclose(a32, b32, atol=tol, rtol=rtol):
            worst = (a32 - b32).abs().max().item()
            return False, f"{name} differs: worst|diff|={worst:.4g} atol={tol}"
    return True, None


@pytest.mark.parametrize(("case", "shot"), ATTN_RES_PARAMS)
def test_all_reduce_add_attn_res_rms_norm_matches_the_two_ops_it_replaces(
    case: tuple[tuple[int, int], bool, int, int, bool],
    shot: FusedShot,
    world: int,
    ranks: World,
) -> None:
    """The fused op against all_reduce then Kimi-K3's `attn_res`: the output, the prefix
    it updates or starts, and the block it writes."""
    if world < 2:
        pytest.skip("a collective needs at least two ranks")
    got = ranks.run(run_add_attn_res_rms_norm_rank, case=case, shot=shot)
    if any(err == NO_FUSED_KERNEL for _, err in got):
        pytest.skip(f"hip declines {shot} all_reduce_add_attn_res_rms_norm at {case}")
    bad = [err for _, err in got if err is not None]
    assert not bad, f"{case}: " + "; ".join(bad)
    assert all(agreed for agreed, _ in got), f"{case}: ranks disagreed"


# ---------------------------------------------------------------------------------
# ALL-REDUCE + RMSNORM + GEMM, WRITTEN OR ADDED, judged against the ops Kimi-K3's latent
# MoE tail runs: the all-reduce, `vllm.ir.ops.rms_norm`, and `addmm_` into this rank's
# column shard of the shared output (the add op), or the product written there (the
# other). FAST: Kimi-K3's decode shape at 4 rows on the one-shot, each op; FULL: the
# rest.
# ---------------------------------------------------------------------------------

# (rows, latent, hidden, shard). Example-based: each case is a full eight-process run,
# so the cases are Kimi-K3's decode tail (latent 3584 -> hidden 7168, a 1/8 shard) at
# 1, 4 and 16 rows, a shard that is not a multiple of the kernel's column tile, and row
# counts past one GEMM pass that only the two-shot kernel takes (one-shot declines).
RMS_NORM_GEMM_CASES = (
    (1, 3584, 7168, 896),
    (4, 3584, 7168, 896),
    (16, 3584, 7168, 896),
    (16, 3584, 7168, 30),
    (33, 3584, 7168, 30),
    (64, 3584, 7168, 896),
)


def run_rms_norm_gemm_rank(
    ctx: RankContext,
    case: tuple[int, int, int, int],
    shot: Shot,
    add: bool,
) -> tuple[bool, str | None]:
    """ONE rank: the fused op against the ops it replaces, over the whole output."""
    import vllm.ir.ops

    rank, world, device = ctx.rank, ctx.world, ctx.device
    rows, latent, hidden, shard = case
    dtype = torch.bfloat16
    inputs = [_one_input(r, 0, (rows, latent), dtype) for r in range(world)]
    norm_w = _one_input(world, 1, (latent,), dtype).to(device)
    # A NARROWED VIEW of the full weight, as the model passes its up_proj shard.
    full_w = (_one_input(world + 1, 2, (hidden, latent), dtype) / latent**0.5).to(
        device
    )
    col0 = (rank * shard) % (hidden - shard + 1)
    gemm_w = full_w.narrow(0, col0, shard)
    shared = _one_input(world + 2, 3, (rows, hidden), dtype).to(device)

    acc = torch.zeros((rows, latent), dtype=torch.float32)
    for x in inputs:
        acc += x.to(torch.float32)
    normed = vllm.ir.ops.rms_norm(acc.to(dtype).to(device), norm_w, FUSED_EPS)
    want = shared.clone()
    if add:
        want.narrow(-1, col0, shard).addmm_(normed, gemm_w.t())
    else:
        want.narrow(-1, col0, shard).copy_(normed @ gemm_w.t())
    torch.cuda.synchronize()

    op = "rms_norm_gemm_add" if add else "rms_norm_gemm"
    forced = _forced(f"{shot}_{op}")
    comm = ctx.comm("hip")
    mine = inputs[rank].to(device)
    admits = (
        comm.should_allreduce_rms_norm_gemm_add
        if add
        else comm.should_allreduce_rms_norm_gemm
    )
    if not admits(mine, gemm_w, **forced):
        return False, NO_FUSED_KERNEL
    got = shared.clone()
    run = comm.all_reduce_rms_norm_gemm_add if add else comm.all_reduce_rms_norm_gemm
    run(mine, norm_w, FUSED_EPS, gemm_w, got.narrow(-1, col0, shard), **forced)
    torch.cuda.synchronize()
    atol, rtol = _fused_tolerance(dtype)
    a32, b32 = got.float().cpu(), want.float().cpu()
    if not torch.allclose(a32, b32, atol=atol, rtol=rtol):
        worst = (a32 - b32).abs().max().item()
        return False, f"out differs: worst|diff|={worst:.4g} atol={atol}"
    return True, None


RMS_NORM_GEMM_FAST = ((4, 3584, 7168, 896), "all_reduce_pull_one_shot")
RMS_NORM_GEMM_PARAMS = tuple(
    pytest.param(
        case,
        shot,
        add,
        id=f"{'add' if add else 'write'}-{case}-{shot}",
        marks=[] if (case, shot) == RMS_NORM_GEMM_FAST else [pytest.mark.full],
    )
    for add in (False, True)
    for case in RMS_NORM_GEMM_CASES
    for shot in SHOTS
)


@pytest.mark.parametrize(("case", "shot", "add"), RMS_NORM_GEMM_PARAMS)
def test_all_reduce_rms_norm_gemm_matches_the_ops_it_replaces(
    case: tuple[int, int, int, int],
    shot: Shot,
    add: bool,
    world: int,
    ranks: World,
) -> None:
    """The fused op against all_reduce, rms_norm and the GEMM into a column shard: the
    shard gets the GEMM written (or added), every other column is left as it was."""
    if world < 2:
        pytest.skip("a collective needs at least two ranks")
    got = ranks.run(run_rms_norm_gemm_rank, case=case, shot=shot, add=add)
    if any(err == NO_FUSED_KERNEL for _, err in got):
        pytest.skip(f"hip declines {shot} rms_norm_gemm (add={add}) at {case}")
    bad = [err for _, err in got if err is not None]
    assert not bad, f"{case}: " + "; ".join(bad)
    assert all(agreed for agreed, _ in got), f"{case}: ranks disagreed"


# (rows, hidden, latent): Kimi-K3's decode and prefill, a row wider than a block, a
# narrow one.
RMS_SCALE_ADD_CASES = (
    (1, 7168, 3584),
    (16, 7168, 3584),
    (33, 7168, 3584),
    (256, 7168, 3584),
    (4, 512, 64),
)
# Kimi-K3 decode, and an uneven split (the last rank has fewer rows).
RMS_SCALE_ADD_FAST = ((16, 7168, 3584), (33, 7168, 3584))


ScaleAddShot = Literal[
    "all_reduce_pull_one_shot_rms_scale_add", "all_reduce_pull_two_shot_rms_scale_add"
]


def run_rms_scale_add_rank(
    ctx: RankContext, case: tuple[int, int, int], shot: ScaleAddShot
) -> tuple[bool, str | None]:
    """ONE rank: the fused op against all_reduce then the scale and add in fp32."""
    rank, world, device = ctx.rank, ctx.world, ctx.device
    rows, hidden, latent = case
    dtype = torch.bfloat16
    width = 2 * hidden + latent
    inputs = [_one_input(r, 0, (rows, width), dtype) for r in range(world)]
    acc = torch.zeros((rows, width), dtype=torch.float32)
    for x in inputs:
        acc += x.to(torch.float32)
    s = acc.to(dtype).float()
    shared, proj, lat = s.split([hidden, hidden, latent], dim=-1)
    inv_rms = torch.rsqrt(lat.pow(2).mean(dim=-1, keepdim=True) + FUSED_EPS)
    want = (shared + proj * inv_rms).to(dtype)

    comm = ctx.comm("hip")
    mine = inputs[rank].to(device)
    got = torch.empty(rows, hidden, dtype=dtype, device=device)
    forced = _forced(shot)
    if not comm.should_allreduce_rms_scale_add(mine, got, **forced):
        return False, NO_FUSED_KERNEL
    comm.all_reduce_rms_scale_add(mine, got, FUSED_EPS, **forced)
    torch.cuda.synchronize()
    atol, rtol = _fused_tolerance(dtype)
    a32, b32 = got.float().cpu(), want.float()
    if not torch.allclose(a32, b32, atol=atol, rtol=rtol):
        worst = (a32 - b32).abs().max().item()
        return False, f"out differs: worst|diff|={worst:.4g} atol={atol}"
    return True, None


@pytest.mark.parametrize(
    ("case", "shot"),
    [
        pytest.param(
            c,
            shot,
            marks=[] if c in RMS_SCALE_ADD_FAST else [pytest.mark.full],
            id=f"{c}-{shot}",
        )
        for c in RMS_SCALE_ADD_CASES
        for shot in get_args(ScaleAddShot)
    ],
)
def test_all_reduce_rms_scale_add_matches_the_ops_it_replaces(
    case: tuple[int, int, int], shot: ScaleAddShot, world: int, ranks: World
) -> None:
    """The one-all-reduce latent MoE tail's epilogue, each shot forced, against
    all_reduce then shared + projected * 1/rms(latent)."""
    if world < 2:
        pytest.skip("a collective needs at least two ranks")
    got = ranks.run(run_rms_scale_add_rank, case=case, shot=shot)
    if any(err == NO_FUSED_KERNEL for _, err in got):
        pytest.skip(f"hip declines {shot} at {case}")
    bad = [err for _, err in got if err is not None]
    assert not bad, f"{case} {shot}: " + "; ".join(bad)
    assert all(agreed for agreed, _ in got), f"{case}: ranks disagreed"


def test_python_names_cpps_errors_and_ops() -> None:
    """Python's `Error` is C++'s, name for name in order, so an Error's number names the
    same reason on both sides; and its `Op` names exactly C++'s ops, which cross by
    name."""
    # example-based: fixed tables against fixed tables, nothing to vary
    import vllm._rocm_C  # noqa: F401  (registers torch.ops._rocm_C)

    built = build_info()
    assert tuple(e.name for e in Error) == built.error_names
    assert set(get_args(Op)) | set(get_args(ExperimentalOp)) == set(built.op_names)


def test_build_info_lists_every_template_with_its_configs() -> None:
    """Every template the build holds names one of its ops and lists at least one config at a
    legal block size, each with a tile; every op's but the plain all-reduce's has a grid (its grid
    is the call's)."""
    # example-based: one fixed catalog
    import vllm._rocm_C  # noqa: F401  (registers torch.ops._rocm_C)

    built = build_info()
    assert built.templates
    for name, t in built.templates.items():
        assert t.op in built.op_names, name
        assert t.configs, name
        for c in t.configs:
            assert c["threads_per_block"] > 0 and c["waves_per_eu"] >= 1, (name, c)
            assert c["tile_m"] >= 1 and c["tile_n"] > 0, (name, c)
            assert t.op == "all_reduce" or c["blocks_per_grid"] > 0, (name, c)


def test_open_refuses_a_group_no_one_registered() -> None:
    """Opening over a name no group is registered as is an Error, not a raise."""
    # example-based: the lookup has one outcome for every unregistered name
    import vllm._rocm_C  # noqa: F401  (registers torch.ops._rocm_C)

    handle, err = torch.ops._rocm_C.rocm_comms_open("no-such", "no-such", 0)
    assert handle is None and Error(err) is Error.no_such_group


@pytest.mark.parametrize("size", [1, 2, 3, 4, 8, 16])
def test_supported_is_the_builds_answer(size: int) -> None:
    """`supported` is the build's answer: this device (one the build covers, the tuning
    target) runs exactly the worlds `build_info` says are built, and refuses the rest by
    name."""
    import vllm._rocm_C  # noqa: F401  (registers torch.ops._rocm_C)

    got = supported(torch.device("cuda:0"), size)
    if size in build_info().worlds:
        assert isinstance(got, Supported), got
        assert got.arch in torch.cuda.get_device_properties(0).gcnArchName
    else:
        assert got is Error.world_not_built


def run_eager_beyond_staging_rank(
    ctx: RankContext, shot: Shot
) -> tuple[bool, str | None]:
    """ONE rank: an eager all-reduce of two and a half stagings, forced at `shot`,
    against RCCL's fp32 sum. Each rank draws its own input on the device: the input is
    too large to rebuild every rank's on the CPU."""
    n = build_info().staging_bytes * 5 // 2 // 2  # bf16 elements
    g = torch.Generator(device=ctx.device).manual_seed(_INPUT_SEED + ctx.rank)
    x = torch.randn(n, generator=g, device=ctx.device).to(torch.bfloat16)
    want = x.float()
    dist.all_reduce(want, group=ctx.device_group)
    comm = ctx.comm("hip")
    forced = _forced(shot)
    if not comm.should_allreduce(x, **forced):
        return False, "refused"
    got = comm.all_reduce(x, **forced)[0].float()
    torch.cuda.synchronize()
    atol, rtol = _fused_tolerance(torch.bfloat16)
    if not torch.allclose(got, want, atol=atol, rtol=rtol):
        worst = (got - want).abs().max().item()
        return False, f"out differs: worst|diff|={worst:.4g} atol={atol}"
    return True, None


@pytest.mark.parametrize("shot", SHOTS)
def test_an_eager_all_reduce_wider_than_the_staging_runs_in_passes(
    shot: Shot, world: int, ranks: World
) -> None:
    """An eager input runs on the shot's staged build, which copies it into its staging
    a pass at a time, so an input past the staging runs on our kernels, in one launch,
    and sums right: every pass, the short last one too."""
    # example-based: the size is the point (past the staging, a partial last pass)
    if world < 2:
        pytest.skip("a collective needs at least two ranks")
    got = ranks.run(run_eager_beyond_staging_rank, shot=shot)
    bad = [err for _, err in got if err is not None]
    assert not bad, f"{shot}: " + "; ".join(bad)


def run_warmup_rank(ctx: RankContext) -> tuple[bool, str | None]:
    """ONE rank: inside `capture()` on a stream that is not recording, a capture's
    warmup, rank 0 alone calls every op. Launching would wait on peers that never
    come, so finishing proves nothing launched; the outputs are the right shape."""
    comm = ctx.comm("hip")
    x = torch.randn(16, 7168, device=ctx.device).to(torch.bfloat16)
    weight = torch.ones(7168, device=ctx.device, dtype=torch.bfloat16)
    done = torch.cuda.Event()
    with comm.capture(), torch.cuda.stream(torch.cuda.Stream()):
        if ctx.rank == 0:
            # Each op's outputs, without what ran.
            outs = [
                comm.all_reduce(x)[0],
                comm.all_reduce_rms_norm(x, weight, 1e-6)[0],
                *comm.all_reduce_add_rms_norm(x, x, weight, 1e-6)[:2],
            ]
            done.record()
    if ctx.rank == 0:
        deadline = time.monotonic() + 10.0
        while not done.query():
            if time.monotonic() > deadline:
                return False, "a warmup call launched: it waited on its peers"
            time.sleep(0.01)
        if any(o.shape != x.shape or o.dtype != x.dtype for o in outs):
            shapes = [tuple(o.shape) for o in outs]
            return False, f"outputs {shapes}, not {tuple(x.shape)}"
    dist.barrier(group=ctx.cpu_group)
    return True, None


def test_a_capture_warmup_launches_nothing(world: int, ranks: World) -> None:
    """A capture's warmup: vLLM runs the forward on the capture stream before recording
    and discards the outputs. An op there returns outputs of the right shape and
    launches nothing, as vLLM's and aiter's custom all-reduce do."""
    # example-based: the point is one state (a capture's warmup), not a domain
    if world < 2:
        pytest.skip("a collective needs at least two ranks")
    got = ranks.run(run_warmup_rank)
    bad = [err for _, err in got if err is not None]
    assert not bad, "; ".join(bad)
