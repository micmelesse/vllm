# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: MIT Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""HipCustomCommunicator: vLLM's custom all-reduce (`custom_all_reduce.py`'s
CustomAllreduce) extended to every collective Kimi-K3 needs -- the all-reduce and the
all-reduces fused with the ops after them -- over our HIP kernels in `_rocm_C`
(`csrc/rocm/hip_custom_communicator.cu`).

The caller names an op and Python looks its kernels up in the tuned table
(`get_nearest_configs`: the inputs tuned at the communicator's `quantization`, INT4
where it pays): one kernel or, where the op is faster as two kernels of ours, our
all-reduce and then the op's norm on this rank's rows (`rms_norm`, `add_rms_norm`,
`add_attn_res_rms_norm`, each a public op). The keyword arguments an op takes after its
own (a config in its essential order: the kernel by name, its algorithm, the grid, the block and its waves a EU, the tile, the quantization) are the one way to
force one, for the tuner and
the tests: every one or none. C++ takes every field of every call, checks the config,
and answers -1 or the Error it refused it for: it holds no table and picks nothing.

THE MEMORY (C++'s `Handle`): the communicator's arena is one allocation per rank,
which every rank opened at open over legacy HIP IPC. Every exact call reads every
rank's input IN PLACE through an input table (aiter's gfx950 pattern): a captured
call's tensors, registered when `capture()` exits (one exchange of their IPC handles
over the CPU group, a collective), or an eager call's copy, which C++ makes into the
arena's staging region on the call's stream. An INT4 call's kernel quantizes its input
into the staging region itself.

The C++ context crosses as an opaque `int` handle, so nothing frees it for us:
`close()` has to run.
"""

import atexit
import functools
import json
import os
import signal
import threading
import warnings
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import IntEnum
from pathlib import Path
from types import FrameType
from typing import Any, ClassVar, Literal, final, get_args

import torch
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)


# ---- THE BUILD'S FACTS, as the communicator and its callers name them: the ops, the
# Errors and fault places C++ answers with (number for number; a test holds them equal),
# the build's contents (`build_info`), and whether a device runs it (`supported`). ----

# EVERY OP, as C++ names them (`enum class Op`; a test holds them equal) and as its
# method is named: the all-reduce, then each all-reduce and the ops it fuses, then each
# norm on this rank's rows (the second kernel of an op composed as two), in order.
Op = Literal[
    "all_reduce",
    "all_reduce_rms_norm",
    "all_reduce_add_rms_norm",
    "all_reduce_add_attn_res_rms_norm",
    "rms_norm",
    "add_rms_norm",
    "add_attn_res_rms_norm",
]
# THE OPS A PLAIN CALL OF RUNS ITS TABLE'S PICK (the tuned table's ops), and each one's
# norm on rows, which a pick of two kernels runs after our all-reduce.
TUNED_OPS: tuple[Op, ...] = (
    "all_reduce",
    "all_reduce_rms_norm",
    "all_reduce_add_rms_norm",
    "all_reduce_add_attn_res_rms_norm",
)
_ROWS: Mapping[Op, Op] = {
    "all_reduce_rms_norm": "rms_norm",
    "all_reduce_add_rms_norm": "add_rms_norm",
    "all_reduce_add_attn_res_rms_norm": "add_attn_res_rms_norm",
}

# WHAT CROSSES THE LINKS, C++'s `Quantization` by name (number for number): exact
# values, or INT4. A communicator's setting for its plain calls, and a config's field.
Quantization = Literal["exact", "int4"]
_QUANTIZATIONS: tuple[Quantization, ...] = ("exact", "int4")
# WHICH BODY OF AN OP'S ONE KERNEL a config runs, C++'s `Algorithm` by name (number for
# number): a one-shot or a two-shot, pulled or pushed, or a norm on this rank's rows. Exactly
# the ones that exist. A config's field.
Algorithm = Literal["one_shot_pull", "one_shot_push", "two_shot_pull", "two_shot_push", "rows"]
_ALGORITHMS: tuple[Algorithm, ...] = get_args(Algorithm)


class Error(IntEnum):
    """WHY A CALL CANNOT RUN: C++'s `hip_custom_communicator::Error`, number for
    number (a test holds them equal)."""

    disabled = 0
    no_such_op = 1
    not_contiguous = 2
    not_two_d = 3
    dtype_not_built = 4
    world_not_built = 5
    row_not_packs = 6
    template_not_this_ops = 7
    row_too_wide = 8
    block_not_a_wave_per_peer = 9
    partials_too_small = 10
    grid_not_resident = 11
    staging_too_small = 12
    device_not_built = 13
    device_not_tuned = 14
    weight_not_built = 15
    no_such_template = 16
    no_such_group = 17
    ranks_disagree = 18
    groups_disagree = 19
    threads_not_built = 20
    tile_not_built = 21
    launch_out_of_range = 22
    field_not_positive = 23
    field_not_this_templates = 24
    waves_not_built = 25
    inner_stride_not_one = 26
    row_stride_not_packs = 27
    row_narrower_than_world = 28
    forced_incomplete = 29
    no_such_quantization = 30
    quantization_not_supported = 31
    row_not_int4_groups = 32
    quantized_too_small = 33  # retired: INT4 lies inside staging and partials
    block_too_large = 34
    lds_too_large = 35
    scratch_too_large = 36


class Where(IntEnum):
    """WHERE A KERNEL'S WAIT GAVE UP: C++'s `hip_custom_communicator::Where`, number
    for number (a test holds them equal)."""

    start = 1  # a launch's first barrier: the peer had not launched
    peer_barrier = 2  # a later barrier


class FlagState(IntEnum):
    """A FLAG'S STATES: C++'s `hip_custom_communicator::FlagState`, value for value,
    the only values a barrier writes. Any other value is corrupt memory, and decoding
    it raises."""

    clear = 0  # the writer has not reached this barrier, or the owner has seen it
    arrived = 1  # the writer reached it; the owner has not yet seen it


class FaultState(IntEnum):
    """THE FAULT'S STATES, the status's `state`: C++'s `FaultState`, value for value."""

    none = 0
    writing = 1
    written = 2  # the fault's fields are whole


@dataclass(frozen=True)
class Flag:
    """ONE FLAG of this rank's signals: the slot block `block` on rank `writer` sets
    at a barrier of `parity`, and its state. Every one is `clear` between launches; one
    `arrived` is state a launch left behind."""

    parity: int
    block: int
    writer: int
    state: FlagState


@dataclass(frozen=True)
class Fault:
    """A RANK'S FAULT RECORD: the first of its kernels' waits to outlive the timeout,
    which then left the kernel (its output garbage) and made every other wait on the
    rank leave."""

    where: Where
    rank: int
    block: int
    peer: int  # the rank it waited for
    count: int  # the flag it read last
    want: int  # the value it waited for
    waited_s: float

    def __str__(self) -> str:
        return (
            f"rank {self.rank} block {self.block} gave up in {self.where.name}"
            f" for rank {self.peer} after {self.waited_s:.1f} s: count {self.count}, "
            f"want {self.want}"
        )


@dataclass(frozen=True)
class Status:
    """A RANK'S STATUS, as C++ reads it: whether its waits were told to leave, and
    the fault its first wait to give up recorded (None when none has)."""

    abort: bool
    fault: Fault | None


# --------------------------------------------------------------------------------
# THE STATE LOG (C++'s log.cuh): the kernels' status words, a header, and a ring of
# 32-byte records, in host-mapped memory. Each enum is C++'s, number for number.
# --------------------------------------------------------------------------------

LogMode = Literal["errors", "everything"]
_LOG_MODES: tuple[LogMode, ...] = ("errors", "everything")


class LogKind(IntEnum):
    empty = 0
    open_phase = 1
    open_check = 2
    open_warning = 3
    launch_issued = 4
    copy_in_issued = 5  # an eager all-reduce's host copy into the staging
    stream_seen = 6
    kernel_seen = 7
    deadline_expired = 8
    close_phase = 9
    registration = 10  # captured inputs registered at capture exit
    entered = 16
    barrier_signal = 17
    barrier_passed = 18
    wait_timeout = 19
    aborted = 20
    exited = 21


class OpenPhase(IntEnum):
    none = 0
    log_ready = 1
    arena_allocated = 2
    exchange_allocated = 3  # retired: one arena; kept so the numbers hold
    zeroed = 4
    exported = 5
    handle_exchange_begun = 6
    handle_exchange_done = 7
    opened = 8
    mapped = 9  # retired: VMM's map; IPC opens in one step
    access_set = 10  # retired: VMM's access
    tables_written = 11  # the input tables' slab and its eager table
    checked = 12
    open_done = 13


class ClosePhase(IntEnum):
    """THE MEMORY'S TEARDOWN, in order (C++'s `ClosePhase`, log.cuh): a close runs
    every phase with its two handshakes; an abandon (an error, a signal, exit) runs
    closing, synchronized, peers_released, released (only once the device
    finished) and close_done, never waiting on a peer."""

    none = 0
    closing = 1
    synchronized = 2
    stopped = 3
    peers_released = 4
    peers_done = 5
    released = 6
    close_done = 7


class OpenCheck(IntEnum):
    none = 0
    current_device = 1
    peer_opened = 2
    pointer_sane = 3
    geometry = 4  # retired: VMM's granule and stride
    can_access_peer = 5
    signals_zero = 6


class OpenWarning(IntEnum):
    none = 0
    rccl_not_ready = 1
    rccl_unknown = 2
    other_processes = 3
    duplicate_device = 4
    host_mismatch = 5
    granularity_mismatch = 6  # retired: VMM's granularity
    low_memory = 7
    pointer_device = 8
    peer_not_visible = 9


_LOG_MAGIC = 0x474F4C48
_HEADER = 16  # the header's first word
_CLOSE_PHASE = 32  # the header's close_phase word
_DTYPE_NAMES = ("f16", "bf16", "f32")


@dataclass(frozen=True)
class LogRecord:
    """ONE RECORD of the ring, its fields as C++'s `LogRecord` holds them."""

    kind: LogKind
    barrier: int  # a barrier index, or the phase, check or warning
    peer: int
    launch: int
    kernel: int
    block: int
    seen: int
    want: int
    extra: int
    clock: int  # host records: CLOCK_MONOTONIC ns

    @staticmethod
    def of(w: Sequence[int]) -> "LogRecord":
        return LogRecord(
            kind=LogKind(w[0] & 0xFF),
            barrier=(w[0] >> 8) & 0xFF,
            peer=w[0] >> 16,
            launch=w[1],
            kernel=w[2] & 0xFFFF,
            block=w[2] >> 16,
            seen=w[3],
            want=w[4],
            extra=w[5],
            clock=w[6] | (w[7] << 32),
        )

    def warning(self) -> str:
        """An open_warning record as the sentence it stands for."""
        w = OpenWarning(self.barrier)
        match w:
            case OpenWarning.rccl_not_ready:
                return (
                    "RCCL for this group is not set up yet; its lazy setup will "
                    "run after our open (on 2026-10-16, under VMM, that hung)"
                )
            case OpenWarning.rccl_unknown:
                return "whether RCCL for this group is set up could not be told"
            case OpenWarning.other_processes:
                return (
                    f"{self.seen} other processes hold {self.want} MiB of device "
                    "memory on this GPU"
                )
            case OpenWarning.duplicate_device:
                return (
                    f"rank {self.peer} (pid {self.seen}) is on this rank's GPU "
                    f"(pid {self.want})"
                )
            case OpenWarning.host_mismatch:
                return f"rank {self.peer} is on another host"
            case OpenWarning.pointer_device:
                return (
                    f"rank {self.peer}'s opened arena reports device {self.seen}, "
                    f"its GPU here is {self.want}"
                )
            case OpenWarning.low_memory:
                return f"{self.seen} MiB free, open allocates {self.want} MiB"
            case OpenWarning.peer_not_visible:
                return (
                    f"rank {self.peer}'s GPU is not visible in this process; its "
                    "device checks were skipped"
                )
        return w.name

    def __str__(self) -> str:
        k = self.kind
        if k is LogKind.open_phase:
            failed = f" FAILED (error {self.extra})" if self.extra else ""
            return (
                f"open_phase {OpenPhase(self.barrier).name} seen={self.seen} "
                f"want={self.want}{failed}"
            )
        if k is LogKind.open_check:
            verdict = "FAILED" if self.extra else "ok"
            return (
                f"open_check {OpenCheck(self.barrier).name} rank {self.peer} "
                f"{verdict} seen={self.seen} want={self.want}"
            )
        if k is LogKind.open_warning:
            return f"open_warning {OpenWarning(self.barrier).name}: {self.warning()}"
        if k is LogKind.close_phase:
            how = "close" if self.peer else "abandon"
            failed = f" FAILED ({self.extra})" if self.extra else ""
            return (
                f"close_phase {ClosePhase(self.barrier).name} ({how}) "
                f"seen={self.seen} want={self.want}{failed}"
            )
        if k is LogKind.launch_issued:
            dtype = (
                _DTYPE_NAMES[self.extra]
                if self.extra < len(_DTYPE_NAMES)
                else str(self.extra)
            )
            mode = "captured" if self.barrier & 1 else "eager"
            ops = get_args(Op)
            op = ops[self.block] if self.block < len(ops) else str(self.block)
            return (
                f"launch_issued #{self.launch} {op} kernel "
                f"{self.kernel} {self.seen}x{self.want} {dtype} stream {self.peer} "
                f"{mode}"
            )
        if k is LogKind.stream_seen:
            return f"stream_seen {self.peer} = {self.seen | (self.want << 32):#x}"
        if k is LogKind.kernel_seen:
            return f"kernel_seen {self.kernel} = {self.seen | (self.want << 32):#x}"
        return (
            f"{k.name} launch {self.launch} kernel {self.kernel} block {self.block} "
            f"barrier {self.barrier} peer {self.peer} seen {self.seen} want "
            f"{self.want} extra {self.extra} clock {self.clock}"
        )


@dataclass(frozen=True)
class Log:
    """A RANK'S STATE LOG, the host's fold of C++'s header and ring: what open
    reached, how many launches were issued, the fault if one was recorded, and the
    records still in the ring, oldest first."""

    mode: LogMode
    rank: int
    world: int
    device: int
    timeout_ticks: int
    clock_khz: int
    phase: OpenPhase
    close_phase: ClosePhase
    launches: int
    abort: bool
    cursor: int
    dropped: int
    failed: OpenCheck
    fault: "Fault | None"
    records: tuple[LogRecord, ...]

    @staticmethod
    def of(words: Sequence[int]) -> "Log | None":
        """The log from its words, None when there are none (no log was made)."""
        if len(words) <= _HEADER or words[_HEADER] != _LOG_MAGIC:
            return None
        h = words[_HEADER : _HEADER + 16]
        capacity, offset = h[13], h[14] // 4
        cursor = h[11]
        first = max(0, cursor - capacity)
        records = []
        for n in range(first, cursor):
            at = offset + 8 * (n % capacity)
            r = LogRecord.of(words[at : at + 8])
            if r.kind is not LogKind.empty:
                records.append(r)
        khz = h[8]
        return Log(
            mode=_LOG_MODES[h[2]],
            rank=h[3],
            world=h[4],
            device=h[5],
            timeout_ticks=h[6] | (h[7] << 32),
            clock_khz=khz,
            phase=OpenPhase(h[9]),
            close_phase=ClosePhase(words[_CLOSE_PHASE]),
            launches=h[10],
            abort=words[0] != 0,
            cursor=cursor,
            dropped=h[12],
            failed=OpenCheck(h[15]),
            fault=_fault_of(words, khz),
            records=tuple(records),
        )

    def warnings(self) -> list[str]:
        return [r.warning() for r in self.records if r.kind is LogKind.open_warning]

    def lines(self, last: int = 40) -> list[str]:
        timeout_s = (
            self.timeout_ticks / (self.clock_khz * 1000.0) if self.clock_khz else 0.0
        )
        out = [
            f"hip_custom_communicator log: rank {self.rank} of {self.world}, device "
            f"{self.device}, mode {self.mode}, open phase {self.phase.name}, "
            f"close phase {self.close_phase.name}, launches {self.launches}, "
            f"timeout {timeout_s:.2f} s "
            f"({self.timeout_ticks} ticks at {self.clock_khz} kHz), abort "
            f"{int(self.abort)}, records {self.cursor} (dropped {self.dropped})"
        ]
        if self.failed is not OpenCheck.none:
            out.append(f"open failed its check: {self.failed.name}")
        out.append(f"fault: {self.fault}" if self.fault else "fault: none")
        shown = self.records[-last:]
        if shown:
            t0 = shown[0].clock
            out.append(f"last {len(shown)} of {len(self.records)} records:")
            out += [f"  +{(r.clock - t0) / 1e9:10.6f} s  {r}" for r in shown]
        return out

    def __str__(self) -> str:
        return "\n".join(self.lines())


def _fault_of(words: Sequence[int], khz: int) -> "Fault | None":
    """The kernels' fault record, from the status words at the log's front."""
    if FaultState(words[1]) is not FaultState.written:
        return None

    def signed(w: int) -> int:
        return w - (1 << 32) if w >= 1 << 31 else w

    ticks = words[8] | (words[9] << 32)
    return Fault(
        where=Where(words[2]),
        rank=signed(words[3]),
        block=signed(words[4]),
        peer=signed(words[5]),
        count=words[6],
        want=words[7],
        waited_s=ticks / (khz * 1000.0) if khz else 0.0,
    )


def last_log() -> Log | None:
    """THE LAST LOG THIS PROCESS MADE, kept by C++ past a failed open and readable
    while an open still runs: where an open failed or hangs."""
    return Log.of(torch.ops._rocm_C.hip_custom_communicator_log(0))


# ---- ABNORMAL EXIT (STANDARDS 3.5, "abnormal exit is designed for"): a process should
# not die holding its peers' memory open, nor free its own while they read it. So an
# exit, SIGTERM and SIGINT tear every open communicator down ALONE (C++'s
# `Handle::abandon`): every wait told to leave, the device given a bounded wait, every
# BORROWED peer IPC mapping closed, then our own arena freed once the device finished,
# each step in the state log.
# It never waits on a peer: one may be dead. C++ also runs it from its own process-exit
# hook; it is idempotent. ----

# How long an abandon gives a device to finish its launches, a communicator.
ABANDON_WAIT_S = 2.0
# What the signal handlers cover: what a supervisor and a terminal send.
_EXIT_SIGNALS = (signal.SIGTERM, signal.SIGINT)
_exit_hooks_installed = False
SignalHandler = Callable[[int, FrameType | None], None] | int | None


def abandon_open(wait_s: float = ABANDON_WAIT_S) -> None:
    """EVERY COMMUNICATOR STILL OPEN IN THIS PROCESS torn down alone, borrowed peer
    mappings first: for an exit, a signal, a harness about to kill this process.
    Never waits on a peer; waits at most `wait_s` a communicator for its device. A
    communicator abandoned here can still be `close()`d (it does what is left)."""
    torch.ops._rocm_C.hip_custom_communicator_abandon_all(wait_s)
    for communicator in list(_open_communicators):
        communicator._abandoned()


def _on_exit_signal(
    previous: SignalHandler, signum: int, frame: FrameType | None
) -> None:
    """Abandon, then what was there before: the previous handler, or the default
    action (re-raised with the default restored), or nothing when it was ignored.
    Python runs this on the main thread between bytecodes, so a main thread stuck
    in a C++ call (a synchronize) reaches it only when that returns; a harness that
    must not wait calls `abandon_open` from its own thread."""
    abandon_open()
    if callable(previous):
        previous(signum, frame)
    elif previous != signal.SIG_IGN:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


def _install_exit_hooks() -> None:
    """Once a process, at the first open: atexit, and SIGTERM and SIGINT handlers
    chained in front of whatever is there (from the main thread only: Python sets
    handlers nowhere else)."""
    global _exit_hooks_installed
    if _exit_hooks_installed:
        return
    _exit_hooks_installed = True
    atexit.register(abandon_open)
    if threading.current_thread() is not threading.main_thread():
        logger.warning(
            "HipCustomCommunicator opened off the main thread: no SIGTERM/SIGINT "
            "teardown (process exit still runs it)"
        )
        return
    for sig in _EXIT_SIGNALS:
        signal.signal(sig, functools.partial(_on_exit_signal, signal.getsignal(sig)))


def configured_log_mode() -> LogMode:
    """What `VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_LOG` asks (errors by default)."""
    import vllm.envs as envs

    return (
        "everything"
        if envs.VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_LOG == "everything"
        else "errors"
    )


def rccl_ready(group: ProcessGroup, device: torch.device) -> int:
    """Whether `group`'s RCCL communicator is set up already (1), not yet (0), or
    cannot be told (-1). torch's ProcessGroupNCCL builds it lazily, at the first
    collective; `_is_initialized` reads that without starting it."""
    try:
        backend = group._get_backend(device)
        is_initialized = getattr(backend, "_is_initialized", None)
        if is_initialized is None:
            return -1
        return 1 if is_initialized() else 0
    except (RuntimeError, AttributeError, ValueError):
        return -1


# C++'s `DType` names, as torch's dtypes.
_DTYPES: Mapping[str, torch.dtype] = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}


@dataclass(frozen=True)
class TemplateBuild:
    """One C++ template as built (an op's one kernel): its op, and the builds dispatch
    instantiates (its catalog list), each its fields by name as a forcing passes them
    (algorithm, block_x, block_y, block_z, waves_per_eu, its tile and its
    quantization; never the launch, which a forced call gives itself)."""

    op: str
    builds: tuple[dict[str, int | str], ...]


@dataclass(frozen=True)
class OpRecord:
    """ONE OP'S PLAIN CALL as built (C++'s `kOps`): what a call of it with nothing
    forced must be to run, which `should` reads."""

    dtypes: frozenset[torch.dtype]  # its input's
    weight_dtypes: frozenset[torch.dtype]  # its weight's besides its input's; none
    two_d: bool  # its input is [rows, cols], not any shape read as rows of the last


@dataclass(frozen=True)
class BuildInfo:
    """What the build holds, the same on every device: C++'s `kBuild.supports`, its
    pack bytes, its ops', errors' and fault places' names, its templates, and each
    op's plain call. What a call may hold is a communicator's (`Capacity`)."""

    dtypes: frozenset[torch.dtype]
    worlds: frozenset[int]
    pack_bytes: int
    # C++'s `Op`, `Error` and `Where` members by name, each in its enum's order: an op
    # crosses by name, an Error and a Where by number.
    op_names: tuple[str, ...]
    error_names: tuple[str, ...]
    where_names: tuple[str, ...]
    # Each template by name, in C++'s `Template` order: what a tuner searches.
    templates: dict[str, TemplateBuild]
    # Each op by name, in C++'s `Op` order: its plain call.
    ops: dict[str, OpRecord]
    # The pushed one-shot's most blocks: a block's own slots in the unfenced region.
    unfenced_blocks: int


@functools.cache
def build_info() -> BuildInfo:
    """The build's facts, read once: they are fixed when it is compiled."""
    (
        dtypes,
        worlds,
        pack,
        ops,
        errors,
        where_names,
        names,
        template_ops,
        fields,
        flat,
        counts,
        op_dtypes,
        op_weights,
        op_facts,
        unfenced_blocks,
    ) = torch.ops._rocm_C.hip_custom_communicator_build_info()
    templates: dict[str, TemplateBuild] = {}
    at, width = 0, len(fields)
    for name, op, count in zip(names, template_ops, counts):
        builds: list[dict[str, int | str]] = []
        for _ in range(count):
            given = dict(zip(fields, flat[at : at + width]))
            variant = {"algorithm": _ALGORITHMS[given.pop("algorithm")]}
            q = _QUANTIZATIONS[given.pop("quantization")]
            fields_set = {f: v for f, v in given.items() if v != 0}
            builds.append(variant | fields_set | {"quantization": q})
            at += width
        templates[name] = TemplateBuild(op, tuple(builds))
    records: dict[str, OpRecord] = {}
    at_dtype = at_weight = 0
    for op, at in zip(ops, range(0, len(op_facts), 3)):
        num_dtypes, num_weights, two_d = op_facts[at : at + 3]
        dtypes_in = op_dtypes[at_dtype : at_dtype + num_dtypes]
        weights_in = op_weights[at_weight : at_weight + num_weights]
        at_dtype += num_dtypes
        at_weight += num_weights
        records[op] = OpRecord(
            frozenset(_DTYPES[d] for d in dtypes_in),
            frozenset(_DTYPES[d] for d in weights_in),
            bool(two_d),
        )
    return BuildInfo(
        frozenset(_DTYPES[d] for d in dtypes),
        frozenset(worlds),
        pack,
        tuple(ops),
        tuple(errors),
        tuple(where_names),
        templates,
        records,
        unfenced_blocks,
    )


@dataclass(frozen=True)
class Capacity:
    """WHAT ONE COMMUNICATOR HOLDS (C++'s `hip_custom_communicator_capacity`): the
    largest call its exchange was sized for, `max_rows` rows of at most `max_cols`
    elements (for a model, the server's tokens a batch by its hidden size), the
    exchange's partials and staging bytes, and each op's most bytes on them whatever
    a call's shape."""

    max_rows: int
    max_cols: int
    partials_bytes: int
    staging_bytes: int
    max_bytes: dict[str, int]


def capacity_of(max_rows: int, max_cols: int, world: int) -> Capacity:
    """What a communicator serving `max_rows` rows of at most `max_cols` columns at
    `world` ranks holds: C++'s answer, the same function its opener allocates by."""
    rows, cols, partials, staging, *most = (
        torch.ops._rocm_C.hip_custom_communicator_capacity(max_rows, max_cols, world)
    )
    return Capacity(
        rows, cols, partials, staging, dict(zip(build_info().op_names, most))
    )


def problem_size() -> tuple[int, int] | None:
    """THE LARGEST CALL A MODEL'S COMMUNICATOR SERVES, from the server's config: every
    all-reduce is at most a batch of tokens, `max_num_batched_tokens`, and a row is
    at most the hidden size, the target's or a speculative draft model's (STANDARDS,
    *A kernel's own buffers are sized from the problem it serves*). None where there
    is no server config to say it (vLLM building its communicators in a test or a
    tool): a communicator is never sized by a guess."""
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    if config is None or config.model_config is None:
        return None
    widest = config.model_config.get_hidden_size()
    spec = config.speculative_config
    draft = getattr(spec, "draft_model_config", None) if spec is not None else None
    if draft is not None:
        widest = max(widest, draft.get_hidden_size())
    return config.scheduler_config.max_num_batched_tokens, widest


def should(
    op: Op,
    input: torch.Tensor,
    capacity: Capacity,
    weight: torch.Tensor | None = None,
) -> bool:
    """Whether a PLAIN call of `op` (nothing forced) on `input`, and a norm's `weight`,
    runs on an open communicator holding `capacity`: inside the op's record
    (`build_info().ops`), whole packs a row, and inside what its exchange holds,
    either the box it was sized for (rows and row width) or the op's most bytes.
    Python alone, no C++ call, as vLLM's `should_custom_ar`; C++ refuses no call this
    admits."""
    built = build_info()
    record = built.ops[op]
    if (
        input.dtype not in record.dtypes
        or not input.is_contiguous()
        or input.numel() == 0
    ):
        return False
    if input.dim() == 0 or (record.two_d and input.dim() != 2):
        return False
    row = input.shape[-1]
    if weight is not None and weight.dtype not in record.weight_dtypes | {input.dtype}:
        return False
    rows = input.numel() // row
    in_box = rows <= capacity.max_rows and row <= capacity.max_cols
    return row * input.element_size() % built.pack_bytes == 0 and (
        in_box or input.numel() * input.element_size() <= capacity.max_bytes[op]
    )


@dataclass(frozen=True)
class Supported:
    """What the library runs on, once `supported` finds it can: the device's arch."""

    arch: str


# THE TUNED TABLES (`hip_custom_communicator.json`, beside this file, written by gd's
# tune: gd's `TunedTables`), a table an arch, each tuned on its own: its `inputs` (the
# input points tuned: the op, qualified `hip_custom_communicator.<op>`, the
# communicator's mode, `exact` or `quantized`, the world, the width, the tokens and the
# op's own arguments, which the lookup does not read), its `configs` (each its op and
# its fields: what the build compiles there) and its `picks` (an input's index, and its
# configs' in the order a plain call launches them: the op's one kernel, or our
# all-reduce then the op's norm on rows). No format or version: it changes with this
# file, in one commit. Read once, the arch's table checked at open (`check_table`); a
# plain call launches its nearest input's pick (`get_nearest_configs`). Every tuned
# number is here, none in C++. A communicator opened with `tables` of the file's shape
# runs those instead (the tuner's candidate).
_TABLE = Path(__file__).with_name("hip_custom_communicator.json")
_TABLE_KEYS = ("inputs", "configs", "picks")
# AN ARCH'S TUNED TABLE AS THE FILE HOLDS IT, parsed: its three lists; and the file, a
# table an arch.
TunedTable = dict[str, Any]
TunedTables = dict[str, TunedTable]
# AN OP AS THE TABLE NAMES IT, and the communicator's quantization as the table's input
# names it (the mode the communicator was opened in).
_QUALIFIED = "hip_custom_communicator."
_MODES: Mapping[Quantization, str] = {"exact": "exact", "int4": "quantized"}
# A CONFIG POINT'S FIELDS, in their order (C++'s arguments'): the kernel, its algorithm,
# its grid, its block and the block's waves a EU (__launch_bounds__), its
# tile (tile_k 1 where the kernel has no K), then what it moves. Every config of every
# op carries each; all but the kernel and the grid are its build.
_CONFIG_KEYS = (
    "kernel",
    "algorithm",
    "grid_x",
    "grid_y",
    "grid_z",
    "block_x",
    "block_y",
    "block_z",
    "waves_per_eu",
    "tile_m",
    "tile_n",
    "tile_k",
    "quantization",
)


@dataclass(frozen=True)
class Config:
    """ONE KERNEL'S CONFIG, a field each (`_CONFIG_KEYS`, C++'s arguments, in their
    order): the kernel, its algorithm, its grid, its block and the block's waves a EU,
    its tile, what it moves. Every op's C++ call is handed each field by name, a line
    each, never a tuple or a dict unpacked: a field one side lacks is an error naming it
    (STANDARDS 3)."""

    kernel: str
    algorithm: Algorithm
    grid_x: int
    grid_y: int
    grid_z: int
    block_x: int
    block_y: int
    block_z: int
    waves_per_eu: int
    tile_m: int
    tile_n: int
    tile_k: int
    quantization: Quantization


def _config(c: Mapping[str, Any]) -> Config:
    """A table's config record (its op, then a field a key, `_CONFIG_KEYS`) as a
    `Config`."""
    return Config(
        kernel=c["kernel"],
        algorithm=c["algorithm"],
        grid_x=c["grid_x"],
        grid_y=c["grid_y"],
        grid_z=c["grid_z"],
        block_x=c["block_x"],
        block_y=c["block_y"],
        block_z=c["block_z"],
        waves_per_eu=c["waves_per_eu"],
        tile_m=c["tile_m"],
        tile_n=c["tile_n"],
        tile_k=c["tile_k"],
        quantization=c["quantization"],
    )


def check_table(raw: TunedTable) -> None:
    """THE TUNED TABLE `raw` (the arch's file, or one handed in at open), checked: one
    that is not sound raises, naming the entry. Its keys `_TABLE_KEYS`; every config its
    op's (qualified) and every field `_CONFIG_KEYS` names in that order, a picked one a
    kernel the build has at a build it lists (its grid is C++'s to refuse at the call; a
    candidate, a config no pick names, need not be built); every input
    one of `TUNED_OPS`, in a mode, at a built world, with a width and tokens; every pick
    one input's, its configs one kernel of the input's op, or our all-reduce then the
    op's norm on rows, an exact input's moving nothing quantized; every tuned op an
    input in each mode at every built world (the lookup reads one there)."""
    built = build_info()
    if list(raw) != list(_TABLE_KEYS):
        raise ValueError(
            f"the tuned table's keys are {list(raw)}, not {list(_TABLE_KEYS)}"
        )
    configs, inputs = raw["configs"], raw["inputs"]
    # A CANDIDATE (a config no pick names: a tune's) need not be built: builds.py leaves
    # out what no kernel takes, and a forced call of it is refused (tile_not_built).
    picked = {j for k in raw["picks"] for j in k.get("configs", [])}
    ops_of_configs = []
    for i, c in enumerate(configs):
        where = f"the tuned table: configs[{i}]"
        if list(c) != ["op", *_CONFIG_KEYS]:
            raise ValueError(f"{where} has keys {list(c)}, not {['op', *_CONFIG_KEYS]}")
        template = built.templates.get(c["kernel"])
        build = {k: c[k] for k in _CONFIG_KEYS if k != "kernel" and k[:5] != "grid_"}
        if i in picked and (template is None or build not in template.builds):
            raise ValueError(f"{where} ({c['kernel']}): no build of it lists this")
        if template is not None and c["op"] != _QUALIFIED + template.op:
            raise ValueError(
                f"{where}: {c['kernel']} is {template.op}'s, not {c['op']}'s"
            )
        ops_of_configs.append(template.op if template is not None else None)
    for i, e in enumerate(inputs):
        where = f"the tuned table: inputs[{i}]"
        name = str(e.get("op", ""))
        if not name.startswith(_QUALIFIED) or name[len(_QUALIFIED) :] not in TUNED_OPS:
            raise ValueError(f"{where}: {name} is no tuned op of the communicator's")
        if e.get("mode") not in _MODES.values() or e.get("world") not in built.worlds:
            raise ValueError(f"{where}: no such mode or world built: {e}")
        if e.get("hidden", 0) < 1 or e.get("tokens", 0) < 1:
            raise ValueError(f"{where}: no width or no tokens")
    picked = set()
    for i, k in enumerate(raw["picks"]):
        where = f"the tuned table: picks[{i}]"
        at = k.get("configs")
        if not (
            isinstance(k.get("input"), int)
            and 0 <= k["input"] < len(inputs)
            and k["input"] not in picked
            and isinstance(at, list)
            and at
            and all(isinstance(j, int) and 0 <= j < len(configs) for j in at)
        ):
            raise ValueError(f"{where}: {k} names no one input's configs")
        picked.add(k["input"])
        e = inputs[k["input"]]
        name = e["op"].removeprefix(_QUALIFIED)
        ops = [ops_of_configs[j] for j in at]
        if ops not in ([name], ["all_reduce", _ROWS.get(name)]):
            raise ValueError(
                f"{where}: its configs are {ops}'s: one of {name}'s, or our all-reduce "
                f"then {_ROWS.get(name)}"
            )
        moved = {configs[j]["quantization"] for j in at}
        if e["mode"] == "exact" and moved != {"exact"}:
            raise ValueError(f"{where}: an exact input moves INT4")
    for name in TUNED_OPS:
        for world in sorted(built.worlds):
            for mode in _MODES.values():
                if not any(
                    inputs[k["input"]]["op"] == _QUALIFIED + name
                    and inputs[k["input"]]["world"] == world
                    and inputs[k["input"]]["mode"] == mode
                    for k in raw["picks"]
                ):
                    raise ValueError(
                        f"the tuned table: {name}: no pick at world {world}, {mode}"
                    )


def get_given_config(
    kernel: str | None,
    algorithm: Algorithm | None,
    grid_x: int | None,
    grid_y: int | None,
    grid_z: int | None,
    block_x: int | None,
    block_y: int | None,
    block_z: int | None,
    waves_per_eu: int | None,
    tile_m: int | None,
    tile_n: int | None,
    tile_k: int | None,
    quantization: Quantization | None,
) -> Config | None:
    """AN OP'S CONFIG AS GIVEN IN ITS ARGUMENTS: every field given (the config), or none
    (None: a plain call, the table's pick). Some but not all is no config, refused,
    forced_incomplete: C++ takes every field."""
    if (
        kernel is not None
        and algorithm is not None
        and grid_x is not None
        and grid_y is not None
        and grid_z is not None
        and block_x is not None
        and block_y is not None
        and block_z is not None
        and waves_per_eu is not None
        and tile_m is not None
        and tile_n is not None
        and tile_k is not None
        and quantization is not None
    ):
        return Config(
            kernel=kernel,
            algorithm=algorithm,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_z=grid_z,
            block_x=block_x,
            block_y=block_y,
            block_z=block_z,
            waves_per_eu=waves_per_eu,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            quantization=quantization,
        )
    if (
        kernel is None
        and algorithm is None
        and grid_x is None
        and grid_y is None
        and grid_z is None
        and block_x is None
        and block_y is None
        and block_z is None
        and waves_per_eu is None
        and tile_m is None
        and tile_n is None
        and tile_k is None
        and quantization is None
    ):
        return None
    raise RuntimeError(
        "forced_incomplete: a forced call gives kernel, algorithm, grid x y z, block x "
        "y z, waves_per_eu, tile m n k and quantization"
    )


def get_nearest_configs(
    table: TunedTable,
    op: Op,
    rows: int,
    cols: int,
    quantization: Quantization,
    world: int,
) -> list[Config]:
    """WHAT A PLAIN CALL OF `op` LAUNCHES on `rows` x `cols`, from the arch's `table`:
    the pick of the nearest input of `op` tuned in the communicator's mode
    (`quantization`) at `world`, its configs in the order they launch. The nearest
    width, then the nearest size (rows x width), the fewer tokens on a tie."""
    inputs = table["inputs"]
    picks = [
        k
        for k in table["picks"]
        if inputs[k["input"]]["op"] == _QUALIFIED + op
        and inputs[k["input"]]["mode"] == _MODES[quantization]
        and inputs[k["input"]]["world"] == world
    ]
    width = min(
        (inputs[k["input"]]["hidden"] for k in picks), key=lambda h: abs(h - cols)
    )
    nearest = min(
        (k for k in picks if inputs[k["input"]]["hidden"] == width),
        key=lambda k: (
            abs(inputs[k["input"]]["tokens"] * width - rows * cols),
            inputs[k["input"]]["tokens"],
        ),
    )
    return [_config(table["configs"][j]) for j in nearest["configs"]]


def get_kernel_config(
    raw: TunedTable,
    rows: int,
    cols: int,
    kernel: str,
    algorithm: Algorithm,
    quantization: Quantization,
    world: int,
) -> Config | None:
    """KERNEL `kernel`'S OWN CONFIG AT `algorithm` FOR `rows` x `cols`
    MOVING `quantization` (a forced launch at the tuned config: gp's kernel lines, the
    tests): the config of it moving that which its nearest input at `world` launches,
    of any op, else the first such config of the table's; moving INT4 where it
    has no INT4 config, its exact config moved INT4 (a build it lists). None where it
    has neither."""
    builds = build_info().templates[kernel].builds
    variant = {"kernel": kernel, "algorithm": algorithm}

    def takes(c: dict[str, Any], moving: Quantization) -> bool:
        build = {k: c[k] for k in _CONFIG_KEYS if k != "kernel" and k[:5] != "grid_"}
        return (
            {k: c[k] for k in variant} == variant
            and c["quantization"] == moving
            and build | {"quantization": quantization} in builds
        )

    for moving in (
        (quantization, "exact") if quantization == "int4" else (quantization,)
    ):
        found = []
        for k in raw["picks"]:
            p = raw["inputs"][k["input"]]
            if p["world"] == world:
                c = next((raw["configs"][j] for j in k["configs"]
                          if takes(raw["configs"][j], moving)), None)  # fmt: skip
                if c is not None:
                    found.append((p, c))
        if found:
            width = min((p["hidden"] for p, _ in found), key=lambda h: abs(h - cols))
            _, c = min(
                ((p, c) for p, c in found if p["hidden"] == width),
                key=lambda pc: (
                    abs(pc[0]["tokens"] * pc[0]["hidden"] - rows * cols),
                    pc[0]["tokens"],
                ),
            )
        else:
            c = next((c for c in raw["configs"] if takes(c, moving)), None)
        if c is not None:
            return replace(_config(c), quantization=quantization)
    return None


def open_handle(
    cpu_group: str,
    device_group: str,
    device: int,
    timeout_seconds: float,
    max_rows: int,
    max_cols: int,
    log_mode: LogMode = "errors",
    rccl_ready: int = -1,
) -> int | Error:
    """THE PEER MEMORY OPENED over the process groups named `cpu_group` and
    `device_group`, a collective (C++ gathers every rank's IPC handle over the CPU
    group): the handle, or the Error it was refused for. The one place Python calls
    C++'s opener, so its arguments are written once. `log_mode`: what the state log
    records. `rccl_ready`: whether the device group's RCCL communicator was found set
    up (`rccl_ready`), logged as an observation. An open that fails one of its checks
    raises, naming it, with the log appended."""
    try:
        handle, err = torch.ops._rocm_C.hip_custom_communicator_open(
            cpu_group,
            device_group,
            device,
            timeout_seconds,
            max_rows,
            max_cols,
            _LOG_MODES.index(log_mode),
            rccl_ready,
        )
    except RuntimeError as e:
        raise RuntimeError(f"{e}\n{last_log() or 'no log was made'}") from e
    return Error(err) if err is not None else handle


def configured_quantization() -> Quantization:
    """What `VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION` asks (exact by default:
    INT4 is lossy, so opt-in)."""
    import vllm.envs as envs

    asked = envs.VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION
    return "int4" if asked == "INT4" else "exact"


def supported(device: torch.device, world: int) -> Supported | Error:
    """Whether our kernels run on `device` in a world of `world`, as the C++ build
    answers it (`hip_custom_communicator::supported`): the Supported, or the Error.
    A build without gfx950 has no communicator: device_not_built."""
    if not hasattr(torch.ops._rocm_C, "hip_custom_communicator_supported"):
        return Error.device_not_built
    arch, err = torch.ops._rocm_C.hip_custom_communicator_supported(device.index, world)
    return Error(err) if err is not None else Supported(arch)


def is_weak_contiguous(input: torch.Tensor) -> bool:
    return input.is_contiguous() or (
        input.storage().nbytes() - input.storage_offset() * input.element_size()
        == input.numel() * input.element_size()
    )


# TORCH'S DEFAULT for a collective on a GPU process group (`default_pg_nccl_timeout`):
# how long a kernel waits on a peer before it reports a fault, where the group does not
# say its own.
TORCH_COLLECTIVE_TIMEOUT_SECONDS = 600.0


def collective_timeout_seconds(group: ProcessGroup, device: torch.device) -> float:
    """The group's own timeout, so a wait in our kernels gives up when every other
    collective on the group would: torch's 10 minutes unless the group was made with
    another. Read through torch's backend options, which are private, so where they
    are not there it is torch's default."""
    try:
        return group._get_backend(device).options._timeout.total_seconds()
    except (AttributeError, RuntimeError):
        return TORCH_COLLECTIVE_TIMEOUT_SECONDS


# ---- THE COMMUNICATOR. ----


def _as_device(device: int | str | torch.device) -> torch.device:
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    if isinstance(device, str):
        return torch.device(device)
    assert isinstance(device, torch.device)
    return device


# A COMMUNICATOR'S LIFE, each state holding what is true in it alone: `__init__` -> Open
# | Disabled(why); Open <-> Capturing; Capturing -> Broken(cause) when a capture raises,
# and Open or Capturing -> Broken(cause) when the watchdog finds a fault; Open,
# Capturing or Broken -> Abandoned when the process exits (`abandon_open`); any ->
# Closed.
# Only `HipCustomCommunicator` moves it, under its lock (the watchdog is another
# thread). `holds` is whether the communicator still holds peer mappings and buffers
# that only `close()` releases: what `__del__` warns about, read without a global
# lookup since collection can run at interpreter shutdown, when globals are None.
@dataclass(frozen=True)
class Disabled:
    why: str  # the backend's reason it cannot run here (not an error)
    holds: ClassVar[bool] = False


@dataclass(frozen=True)
class Open:
    holds: ClassVar[bool] = True


@dataclass(frozen=True)
class Capturing:
    holds: ClassVar[bool] = True


@dataclass(frozen=True)
class Broken:
    cause: str  # what left the peers' state half-done
    holds: ClassVar[bool] = True


@dataclass(frozen=True)
class Abandoned:
    """The process is exiting: the exit hook tore its peer mappings down alone, without
    the peers. Every call is refused; `close()` still does what is left."""

    holds: ClassVar[bool] = False


@dataclass(frozen=True)
class Closed:
    holds: ClassVar[bool] = False


State = Disabled | Open | Capturing | Broken | Abandoned | Closed

# EVERY COMMUNICATOR OPENED IN THIS PROCESS AND NOT YET COLLECTED, for the exit hook to
# move to Abandoned.
_open_communicators: "weakref.WeakSet[HipCustomCommunicator]" = weakref.WeakSet()


# How often the watchdog reads the fault record: a fault is reported within this of the
# wait that wrote it giving up.
_WATCH_S = 0.1


@functools.cache
def _file_tables() -> TunedTables:
    """The tuned tables' file, read once a process."""
    return json.loads(_TABLE.read_text())


class HipCustomCommunicator:
    """vLLM's TP all-reduce and its fused ops over our HIP kernels, behind
    CudaCommunicator. One per process group, because peer pointers are group-scoped.

    Each public op is its whole call, in its own body: what it is refused for, its
    configs (given, or the table's), and the C++ calls, every argument passed as it
    is. The IPC handle exchange is done over the gloo `cpu_group`; C++ only opens the
    handles it is handed. Missing ops are an error, not a disable: falling back quietly
    would let vLLM run its own all-reduce and report the arm ready. Collectives are
    out-of-place.
    """

    state: State = Disabled("not opened")

    @final
    @property
    def disabled(self) -> bool:
        """The flag every vLLM communicator has: no call may run. A BROKEN one
        raises instead: its peers' state is half-done (a capture that failed part
        way), so a quiet fallback would leave the next call on a rank to hang."""
        match self.state:
            case Broken(cause):
                raise RuntimeError(
                    f"{type(self).__name__} is broken: {cause}. Close it and build "
                    f"a new one on every rank."
                )
            case Disabled() | Abandoned() | Closed():
                return True
            case Open() | Capturing():
                return False

    @final
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device_group: ProcessGroup,
        device: int | str | torch.device,
        *,
        timeout_seconds: float | None = None,
        max_rows: int | None = None,
        max_cols: int | None = None,
        quantization: Quantization | None = None,
        log_mode: LogMode | None = None,
        tables: TunedTables | None = None,
    ) -> None:
        """Keep the groups and the device, then `_open`, which checks the backend's
        availability. Unavailability is not an error: the caller checks `.disabled`.
        `timeout_seconds`: how long a kernel waits on a peer before it reports a
        fault; None is the device group's own (`collective_timeout_seconds`).
        `max_rows` and `max_cols`: the largest call it serves, which its exchange is
        sized for; None is the server's (`problem_size`). `quantization`: what the
        plain calls move across the links, from each op's crossover up (their table:
        the quantized one where INT4); None is what the environment asks
        (`configured_quantization`). `log_mode`: what the
        state log records (`log()`); None is what the environment asks
        (`configured_log_mode`). `tables`: the tuned tables its plain calls run from
        (the arch's), of the file's shape (the tuner's candidate); None is the file's
        (`hip_custom_communicator.json`), read once."""
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.device = _as_device(device)
        self.timeout_seconds = (
            collective_timeout_seconds(device_group, self.device)
            if timeout_seconds is None
            else timeout_seconds
        )
        if max_rows is None or max_cols is None:
            served = problem_size()
            if served is None:
                # NOTHING TO SIZE THE EXCHANGE FROM: disabled, naming why, so vLLM
                # takes its own path.
                self._own_fields()
                why = (
                    "no server config to size the exchange from "
                    "(max_num_batched_tokens, the hidden size); "
                    "pass max_rows and max_cols"
                )
                logger.info("%s disabled: %s", type(self).__name__, why)
                self.state = Disabled(why)
                return
            served_rows, served_cols = served
            max_rows = served_rows if max_rows is None else max_rows
            max_cols = served_cols if max_cols is None else max_cols
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.quantization: Quantization = (
            configured_quantization() if quantization is None else quantization
        )
        self.log_mode: LogMode = configured_log_mode() if log_mode is None else log_mode
        self._given_tables = tables
        self.capacity = capacity_of(max_rows, max_cols, device_group.size())
        self._own_fields()
        why = self._open()
        if why is None:
            self.state = Open()
            _open_communicators.add(self)
            if self._faults:
                self._watchdog = threading.Thread(
                    target=self._watch,
                    name=f"{type(self).__name__}-watchdog",
                    daemon=True,
                )
                self._watchdog.start()
        else:
            logger.info("%s disabled: %s", type(self).__name__, why)
            self.state = Disabled(why)

    @final
    def _own_fields(self, state: State | None = None) -> None:
        """THE BASE'S OWN FIELDS: the lock its state moves under and the watchdog's.
        `__init__` sets them first; a test's bare instance (no process group) sets them
        here too, with its `state`."""
        self._lock = threading.Lock()
        self._stop_watching = threading.Event()
        self._watchdog = None
        if state is not None:
            self.state = state

    # ---- What the CALLER uses. Concrete: this class owns the order. ----
    #
    # Each `should_` asks the module's `should` of the op's own tensors: whether a
    # PLAIN call (nothing forced) runs, read from the build's record of the op, with no
    # C++ call -- as vLLM's `should_custom_ar` is. A caller asks it, then calls the op.
    # Each op takes its own tensors and values, then, keyword-only, what the tuner and
    # the tests force: a config, every field given or none (`get_given_config`). None
    # is a plain call: the table's nearest input's configs (`get_nearest_configs`), one
    # kernel, or our all-reduce then the op's norm on rows. The norms on rows have no
    # table: a call of one gives its config. A forced call asks no `should_`: C++
    # answers it. An op raises a RuntimeError naming the Error it is refused for; a
    # call its `should_` admitted is never refused.

    @final
    def tuned(self, op: Op, rows: int, cols: int) -> list[Config]:
        """WHAT A PLAIN CALL OF `op` ON `rows` x `cols` LAUNCHES here, in order: its
        table's nearest input's configs (`get_nearest_configs`)."""
        return get_nearest_configs(
            self.table,
            op,
            rows,
            cols,
            self.quantization,
            self.device_group.size(),
        )

    @final
    def kernel_config(
        self,
        kernel: str,
        algorithm: Algorithm,
        rows: int,
        cols: int,
        quantization: Quantization = "exact",
    ) -> Config | None:
        """KERNEL `kernel`'S OWN TUNED CONFIG at `algorithm` for `rows` x `cols` moving `quantization` (`get_kernel_config`), or None."""
        return get_kernel_config(
            self.table,
            rows,
            cols,
            kernel,
            algorithm,
            quantization,
            self.device_group.size(),
        )

    @final
    def should_allreduce(self, input: torch.Tensor) -> bool:
        """Whether `all_reduce(input)` runs here."""
        return not self.disabled and should("all_reduce", input, self.capacity)

    @final
    def all_reduce(
        self,
        input: torch.Tensor,
        *,
        kernel: str | None = None,
        algorithm: Algorithm | None = None,
        grid_x: int | None = None,
        grid_y: int | None = None,
        grid_z: int | None = None,
        block_x: int | None = None,
        block_y: int | None = None,
        block_z: int | None = None,
        waves_per_eu: int | None = None,
        tile_m: int | None = None,
        tile_n: int | None = None,
        tile_k: int | None = None,
        quantization: Quantization | None = None,
    ) -> torch.Tensor:
        """The sum of every rank's `input`, at the config given (the tuner's, the
        tests'), else at the table's pick for its rows and width. Each config argument
        is C++'s of the same name."""
        # THE CALL.
        logger.debug(
            "all_reduce %s %s%s",
            tuple(input.shape),
            input.dtype,
            " (captured)" if torch.cuda.is_current_stream_capturing() else "",
        )

        # WHAT IT IS REFUSED FOR BEFORE C++ IS ASKED: a communicator that runs nothing,
        # an input that is not [rows, cols], a forcing that is not whole (C++ takes
        # every field).
        if self.disabled:
            raise RuntimeError(f"disabled: {type(self).__name__} runs no call")
        if input.dim() != 2:
            raise RuntimeError(f"not_two_d: [rows, cols], not {input.shape}")
        given_config = get_given_config(
            kernel=kernel,
            algorithm=algorithm,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_z=grid_z,
            block_x=block_x,
            block_y=block_y,
            block_z=block_z,
            waves_per_eu=waves_per_eu,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            quantization=quantization,
        )

        # A CAPTURE'S WARMUP: vLLM discards its outputs, so nothing launches.
        output = torch.empty_like(input)
        if (
            isinstance(self.state, Capturing)
            and not torch.cuda.is_current_stream_capturing()
        ):
            return output

        # THE KERNELS IT LAUNCHES, in order: the config given, else the table's nearest
        # input's (`get_nearest_configs`).
        if given_config is not None:
            configs = [given_config]
        else:
            rows, cols = input.shape
            configs = get_nearest_configs(
                self.table,
                "all_reduce",
                rows,
                cols,
                self.quantization,
                self.device_group.size(),
            )
        logger.debug("all_reduce launches %s", configs)

        match configs:
            case [config]:
                error = torch.ops._rocm_C.hip_custom_communicator_all_reduce(
                    handle_ptr=self._handle,
                    input=input,
                    output=output,
                    kernel=config.kernel,
                    algorithm=config.algorithm,
                    grid_x=config.grid_x,
                    grid_y=config.grid_y,
                    grid_z=config.grid_z,
                    block_x=config.block_x,
                    block_y=config.block_y,
                    block_z=config.block_z,
                    waves_per_eu=config.waves_per_eu,
                    tile_m=config.tile_m,
                    tile_n=config.tile_n,
                    tile_k=config.tile_k,
                    quantization=config.quantization,
                )
                if error != -1:
                    raise RuntimeError(
                        f"{Error(error).name}: {config} on {input.shape}"
                    )
            case _:
                raise RuntimeError(f"no_such_template: all_reduce launches {configs}")
        return output

    @final
    def should_allreduce_rms_norm(
        self, input: torch.Tensor, weight: torch.Tensor
    ) -> bool:
        """Whether `all_reduce_rms_norm(input, weight, ...)` runs here."""
        return not self.disabled and should(
            "all_reduce_rms_norm", input, self.capacity, weight
        )

    @final
    def all_reduce_rms_norm(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        *,
        kernel: str | None = None,
        algorithm: Algorithm | None = None,
        grid_x: int | None = None,
        grid_y: int | None = None,
        grid_z: int | None = None,
        block_x: int | None = None,
        block_y: int | None = None,
        block_z: int | None = None,
        waves_per_eu: int | None = None,
        tile_m: int | None = None,
        tile_n: int | None = None,
        tile_k: int | None = None,
        quantization: Quantization | None = None,
    ) -> torch.Tensor:
        """`vllm.ir.ops.rms_norm(all_reduce(input), weight, eps)`, at the config given
        (the tuner's, the tests'), else at the table's pick for its rows and width: one
        kernel, or our all-reduce then the norm on its rows. Each config argument is
        C++'s of the same name. Returns the normed sum."""
        # THE CALL.
        logger.debug(
            "all_reduce_rms_norm %s %s%s",
            tuple(input.shape),
            input.dtype,
            " (captured)" if torch.cuda.is_current_stream_capturing() else "",
        )

        # WHAT IT IS REFUSED FOR BEFORE C++ IS ASKED: a communicator that runs nothing,
        # an input that is not [rows, cols], a forcing that is not whole (C++ takes
        # every field).
        if self.disabled:
            raise RuntimeError(f"disabled: {type(self).__name__} runs no call")
        if input.dim() != 2:
            raise RuntimeError(f"not_two_d: [rows, cols], not {input.shape}")
        given_config = get_given_config(
            kernel=kernel,
            algorithm=algorithm,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_z=grid_z,
            block_x=block_x,
            block_y=block_y,
            block_z=block_z,
            waves_per_eu=waves_per_eu,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            quantization=quantization,
        )

        # A CAPTURE'S WARMUP: vLLM discards its outputs, so nothing launches.
        output = torch.empty_like(input)
        if (
            isinstance(self.state, Capturing)
            and not torch.cuda.is_current_stream_capturing()
        ):
            return output

        # THE KERNELS IT LAUNCHES, in order: the config given, else the table's nearest
        # input's (`get_nearest_configs`).
        if given_config is not None:
            configs = [given_config]
        else:
            rows, cols = input.shape
            configs = get_nearest_configs(
                self.table,
                "all_reduce_rms_norm",
                rows,
                cols,
                self.quantization,
                self.device_group.size(),
            )
        logger.debug("all_reduce_rms_norm launches %s", configs)

        match configs:
            case [fused_config]:
                # ONE KERNEL, the fused op.
                error = torch.ops._rocm_C.hip_custom_communicator_all_reduce_rms_norm(
                    handle_ptr=self._handle,
                    input=input,
                    weight=weight,
                    output=output,
                    eps=eps,
                    kernel=fused_config.kernel,
                    algorithm=fused_config.algorithm,
                    grid_x=fused_config.grid_x,
                    grid_y=fused_config.grid_y,
                    grid_z=fused_config.grid_z,
                    block_x=fused_config.block_x,
                    block_y=fused_config.block_y,
                    block_z=fused_config.block_z,
                    waves_per_eu=fused_config.waves_per_eu,
                    tile_m=fused_config.tile_m,
                    tile_n=fused_config.tile_n,
                    tile_k=fused_config.tile_k,
                    quantization=fused_config.quantization,
                )
                if error != -1:
                    raise RuntimeError(
                        f"{Error(error).name}: {fused_config} on {input.shape}"
                    )
            case [reduce_config, norm_config]:
                # TWO KERNELS OF OURS: our all-reduce into this rank's sum, then the
                # norm on its rows.
                summed = torch.empty_like(input)
                error = torch.ops._rocm_C.hip_custom_communicator_all_reduce(
                    handle_ptr=self._handle,
                    input=input,
                    output=summed,
                    kernel=reduce_config.kernel,
                    algorithm=reduce_config.algorithm,
                    grid_x=reduce_config.grid_x,
                    grid_y=reduce_config.grid_y,
                    grid_z=reduce_config.grid_z,
                    block_x=reduce_config.block_x,
                    block_y=reduce_config.block_y,
                    block_z=reduce_config.block_z,
                    waves_per_eu=reduce_config.waves_per_eu,
                    tile_m=reduce_config.tile_m,
                    tile_n=reduce_config.tile_n,
                    tile_k=reduce_config.tile_k,
                    quantization=reduce_config.quantization,
                )
                if error != -1:
                    raise RuntimeError(
                        f"{Error(error).name}: {reduce_config} on {input.shape}"
                    )
                error = torch.ops._rocm_C.hip_custom_communicator_rms_norm(
                    handle_ptr=self._handle,
                    input=summed,
                    weight=weight,
                    output=output,
                    eps=eps,
                    kernel=norm_config.kernel,
                    algorithm=norm_config.algorithm,
                    grid_x=norm_config.grid_x,
                    grid_y=norm_config.grid_y,
                    grid_z=norm_config.grid_z,
                    block_x=norm_config.block_x,
                    block_y=norm_config.block_y,
                    block_z=norm_config.block_z,
                    waves_per_eu=norm_config.waves_per_eu,
                    tile_m=norm_config.tile_m,
                    tile_n=norm_config.tile_n,
                    tile_k=norm_config.tile_k,
                    quantization=norm_config.quantization,
                )
                if error != -1:
                    raise RuntimeError(
                        f"{Error(error).name}: {norm_config} on {input.shape}"
                    )
            case _:
                raise RuntimeError(
                    f"no_such_template: all_reduce_rms_norm launches {configs}"
                )
        return output

    @final
    def should_allreduce_add_rms_norm(
        self, input: torch.Tensor, weight: torch.Tensor
    ) -> bool:
        """Whether `all_reduce_add_rms_norm(input, ..., weight, ...)` runs here."""
        return not self.disabled and should(
            "all_reduce_add_rms_norm", input, self.capacity, weight
        )

    @final
    def all_reduce_add_rms_norm(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        *,
        kernel: str | None = None,
        algorithm: Algorithm | None = None,
        grid_x: int | None = None,
        grid_y: int | None = None,
        grid_z: int | None = None,
        block_x: int | None = None,
        block_y: int | None = None,
        block_z: int | None = None,
        waves_per_eu: int | None = None,
        tile_m: int | None = None,
        tile_n: int | None = None,
        tile_k: int | None = None,
        quantization: Quantization | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`vllm.ir.ops.fused_add_rms_norm(all_reduce(input), residual, weight, eps)`,
        at the config given (the tuner's, the tests'), else at the table's pick for its
        rows and width: one kernel, or our all-reduce then the add and norm on its rows.
        Each config argument is C++'s of the same name. Returns the normed result and
        the sum plus residual."""
        # THE CALL.
        logger.debug(
            "all_reduce_add_rms_norm %s %s%s",
            tuple(input.shape),
            input.dtype,
            " (captured)" if torch.cuda.is_current_stream_capturing() else "",
        )

        # WHAT IT IS REFUSED FOR BEFORE C++ IS ASKED: a communicator that runs nothing,
        # an input that is not [rows, cols], a forcing that is not whole (C++ takes
        # every field).
        if self.disabled:
            raise RuntimeError(f"disabled: {type(self).__name__} runs no call")
        if input.dim() != 2:
            raise RuntimeError(f"not_two_d: [rows, cols], not {input.shape}")
        given_config = get_given_config(
            kernel=kernel,
            algorithm=algorithm,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_z=grid_z,
            block_x=block_x,
            block_y=block_y,
            block_z=block_z,
            waves_per_eu=waves_per_eu,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            quantization=quantization,
        )

        # A CAPTURE'S WARMUP: vLLM discards its outputs, so nothing launches.
        output, residual_output = torch.empty_like(input), torch.empty_like(input)
        if (
            isinstance(self.state, Capturing)
            and not torch.cuda.is_current_stream_capturing()
        ):
            return output, residual_output

        # THE KERNELS IT LAUNCHES, in order: the config given, else the table's nearest
        # input's (`get_nearest_configs`).
        if given_config is not None:
            configs = [given_config]
        else:
            rows, cols = input.shape
            configs = get_nearest_configs(
                self.table,
                "all_reduce_add_rms_norm",
                rows,
                cols,
                self.quantization,
                self.device_group.size(),
            )
        logger.debug("all_reduce_add_rms_norm launches %s", configs)

        match configs:
            case [fused_config]:
                # ONE KERNEL, the fused op.
                error = (
                    torch.ops._rocm_C.hip_custom_communicator_all_reduce_add_rms_norm(
                        handle_ptr=self._handle,
                        input=input,
                        residual=residual,
                        weight=weight,
                        output=output,
                        residual_output=residual_output,
                        eps=eps,
                        kernel=fused_config.kernel,
                        algorithm=fused_config.algorithm,
                        grid_x=fused_config.grid_x,
                        grid_y=fused_config.grid_y,
                        grid_z=fused_config.grid_z,
                        block_x=fused_config.block_x,
                        block_y=fused_config.block_y,
                        block_z=fused_config.block_z,
                        waves_per_eu=fused_config.waves_per_eu,
                        tile_m=fused_config.tile_m,
                        tile_n=fused_config.tile_n,
                        tile_k=fused_config.tile_k,
                        quantization=fused_config.quantization,
                    )
                )
                if error != -1:
                    raise RuntimeError(
                        f"{Error(error).name}: {fused_config} on {input.shape}"
                    )
            case [reduce_config, norm_config]:
                # TWO KERNELS OF OURS: our all-reduce into this rank's sum, then the
                # add and norm on its rows.
                summed = torch.empty_like(input)
                error = torch.ops._rocm_C.hip_custom_communicator_all_reduce(
                    handle_ptr=self._handle,
                    input=input,
                    output=summed,
                    kernel=reduce_config.kernel,
                    algorithm=reduce_config.algorithm,
                    grid_x=reduce_config.grid_x,
                    grid_y=reduce_config.grid_y,
                    grid_z=reduce_config.grid_z,
                    block_x=reduce_config.block_x,
                    block_y=reduce_config.block_y,
                    block_z=reduce_config.block_z,
                    waves_per_eu=reduce_config.waves_per_eu,
                    tile_m=reduce_config.tile_m,
                    tile_n=reduce_config.tile_n,
                    tile_k=reduce_config.tile_k,
                    quantization=reduce_config.quantization,
                )
                if error != -1:
                    raise RuntimeError(
                        f"{Error(error).name}: {reduce_config} on {input.shape}"
                    )
                error = torch.ops._rocm_C.hip_custom_communicator_add_rms_norm(
                    handle_ptr=self._handle,
                    input=summed,
                    residual=residual,
                    weight=weight,
                    output=output,
                    residual_output=residual_output,
                    eps=eps,
                    kernel=norm_config.kernel,
                    algorithm=norm_config.algorithm,
                    grid_x=norm_config.grid_x,
                    grid_y=norm_config.grid_y,
                    grid_z=norm_config.grid_z,
                    block_x=norm_config.block_x,
                    block_y=norm_config.block_y,
                    block_z=norm_config.block_z,
                    waves_per_eu=norm_config.waves_per_eu,
                    tile_m=norm_config.tile_m,
                    tile_n=norm_config.tile_n,
                    tile_k=norm_config.tile_k,
                    quantization=norm_config.quantization,
                )
                if error != -1:
                    raise RuntimeError(
                        f"{Error(error).name}: {norm_config} on {input.shape}"
                    )
            case _:
                raise RuntimeError(
                    f"no_such_template: all_reduce_add_rms_norm launches {configs}"
                )
        return output, residual_output

    @final
    def should_allreduce_add_attn_res_rms_norm(self, input: torch.Tensor) -> bool:
        """Whether `all_reduce_add_attn_res_rms_norm(input, ...)` runs here."""
        return not self.disabled and should(
            "all_reduce_add_attn_res_rms_norm", input, self.capacity
        )

    @final
    def all_reduce_add_attn_res_rms_norm(
        self,
        input: torch.Tensor,
        prefix: torch.Tensor | None,
        blocks: torch.Tensor,
        norm_weight: torch.Tensor,
        qk_weight: torch.Tensor,
        output_norm_weight: torch.Tensor | None,
        num_blocks: int,
        write_index: int,
        eps: float,
        output_eps: float,
        *,
        kernel: str | None = None,
        algorithm: Algorithm | None = None,
        grid_x: int | None = None,
        grid_y: int | None = None,
        grid_z: int | None = None,
        block_x: int | None = None,
        block_y: int | None = None,
        block_z: int | None = None,
        waves_per_eu: int | None = None,
        tile_m: int | None = None,
        tile_n: int | None = None,
        tile_k: int | None = None,
        quantization: Quantization | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`attn_res(prefix, all_reduce(input), blocks, ...)`, at the config given (the
        tuner's, the tests'), else at the table's pick for its rows and width: one
        kernel, or our all-reduce then AttnRes on its rows; with no `prefix` the sum
        starts one. Each config argument is C++'s of the same name. Returns the new
        prefix and the AttnRes output, each a new tensor: `prefix` is only read.
        `write_index` >= 0 also stores the new prefix as that block of `blocks`, which
        must not be one of the `num_blocks` sources the call reads."""
        # THE CALL.
        logger.debug(
            "all_reduce_add_attn_res_rms_norm %s %s%s",
            tuple(input.shape),
            input.dtype,
            " (captured)" if torch.cuda.is_current_stream_capturing() else "",
        )

        # WHAT IT IS REFUSED FOR BEFORE C++ IS ASKED: a communicator that runs nothing,
        # an input that is not [rows, cols], a forcing that is not whole (C++ takes
        # every field).
        if self.disabled:
            raise RuntimeError(f"disabled: {type(self).__name__} runs no call")
        if input.dim() != 2:
            raise RuntimeError(f"not_two_d: [rows, cols], not {input.shape}")
        given_config = get_given_config(
            kernel=kernel,
            algorithm=algorithm,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_z=grid_z,
            block_x=block_x,
            block_y=block_y,
            block_z=block_z,
            waves_per_eu=waves_per_eu,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            quantization=quantization,
        )

        # A CAPTURE'S WARMUP: vLLM discards its outputs, so nothing launches.
        prefix_output, output = torch.empty_like(input), torch.empty_like(input)
        if (
            isinstance(self.state, Capturing)
            and not torch.cuda.is_current_stream_capturing()
        ):
            return prefix_output, output

        # THE KERNELS IT LAUNCHES, in order: the config given, else the table's nearest
        # input's (`get_nearest_configs`).
        if given_config is not None:
            configs = [given_config]
        else:
            rows, cols = input.shape
            configs = get_nearest_configs(
                self.table,
                "all_reduce_add_attn_res_rms_norm",
                rows,
                cols,
                self.quantization,
                self.device_group.size(),
            )
        logger.debug("all_reduce_add_attn_res_rms_norm launches %s", configs)

        # THE ONE SLOT WRITTEN, a view of `blocks`; the sources read are the first
        # `num_blocks`, so the two never overlap (C++ refuses a call where they do).
        block_output = blocks[:, write_index] if write_index >= 0 else None
        match configs:
            case [fused_config]:
                # ONE KERNEL, the fused op.
                error = torch.ops._rocm_C.hip_custom_communicator_all_reduce_add_attn_res_rms_norm(  # noqa: E501
                    handle_ptr=self._handle,
                    input=input,
                    prefix=prefix,
                    blocks=blocks,
                    norm_weight=norm_weight,
                    qk_weight=qk_weight,
                    output_norm_weight=output_norm_weight,
                    prefix_output=prefix_output,
                    output=output,
                    block_output=block_output,
                    num_blocks=num_blocks,
                    eps=eps,
                    output_eps=output_eps,
                    kernel=fused_config.kernel,
                    algorithm=fused_config.algorithm,
                    grid_x=fused_config.grid_x,
                    grid_y=fused_config.grid_y,
                    grid_z=fused_config.grid_z,
                    block_x=fused_config.block_x,
                    block_y=fused_config.block_y,
                    block_z=fused_config.block_z,
                    waves_per_eu=fused_config.waves_per_eu,
                    tile_m=fused_config.tile_m,
                    tile_n=fused_config.tile_n,
                    tile_k=fused_config.tile_k,
                    quantization=fused_config.quantization,
                )
                if error != -1:
                    raise RuntimeError(
                        f"{Error(error).name}: {fused_config} on {input.shape}"
                    )
            case [reduce_config, norm_config]:
                # TWO KERNELS OF OURS: our all-reduce into this rank's sum, then
                # AttnRes on its rows.
                summed = torch.empty_like(input)
                error = torch.ops._rocm_C.hip_custom_communicator_all_reduce(
                    handle_ptr=self._handle,
                    input=input,
                    output=summed,
                    kernel=reduce_config.kernel,
                    algorithm=reduce_config.algorithm,
                    grid_x=reduce_config.grid_x,
                    grid_y=reduce_config.grid_y,
                    grid_z=reduce_config.grid_z,
                    block_x=reduce_config.block_x,
                    block_y=reduce_config.block_y,
                    block_z=reduce_config.block_z,
                    waves_per_eu=reduce_config.waves_per_eu,
                    tile_m=reduce_config.tile_m,
                    tile_n=reduce_config.tile_n,
                    tile_k=reduce_config.tile_k,
                    quantization=reduce_config.quantization,
                )
                if error != -1:
                    raise RuntimeError(
                        f"{Error(error).name}: {reduce_config} on {input.shape}"
                    )
                error = torch.ops._rocm_C.hip_custom_communicator_add_attn_res_rms_norm(
                    handle_ptr=self._handle,
                    input=summed,
                    prefix=prefix,
                    blocks=blocks,
                    norm_weight=norm_weight,
                    qk_weight=qk_weight,
                    output_norm_weight=output_norm_weight,
                    prefix_output=prefix_output,
                    output=output,
                    block_output=block_output,
                    num_blocks=num_blocks,
                    eps=eps,
                    output_eps=output_eps,
                    kernel=norm_config.kernel,
                    algorithm=norm_config.algorithm,
                    grid_x=norm_config.grid_x,
                    grid_y=norm_config.grid_y,
                    grid_z=norm_config.grid_z,
                    block_x=norm_config.block_x,
                    block_y=norm_config.block_y,
                    block_z=norm_config.block_z,
                    waves_per_eu=norm_config.waves_per_eu,
                    tile_m=norm_config.tile_m,
                    tile_n=norm_config.tile_n,
                    tile_k=norm_config.tile_k,
                    quantization=norm_config.quantization,
                )
                if error != -1:
                    raise RuntimeError(
                        f"{Error(error).name}: {norm_config} on {input.shape}"
                    )
            case _:
                raise RuntimeError(
                    f"no_such_template: all_reduce_add_attn_res_rms_norm launches "
                    f"{configs}"
                )
        return prefix_output, output

    # ---- THE NORMS ON THIS RANK'S ROWS, no peer: the second kernel of an op as two
    # kernels of ours, a caller's to compose (our all-reduce, then one of these on its
    # sum). No table: a call gives its config, whole. ----

    @final
    def rms_norm(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        *,
        kernel: str | None = None,
        algorithm: Algorithm | None = None,
        grid_x: int | None = None,
        grid_y: int | None = None,
        grid_z: int | None = None,
        block_x: int | None = None,
        block_y: int | None = None,
        block_z: int | None = None,
        waves_per_eu: int | None = None,
        tile_m: int | None = None,
        tile_n: int | None = None,
        tile_k: int | None = None,
        quantization: Quantization | None = None,
    ) -> torch.Tensor:
        """`vllm.ir.ops.rms_norm(input, weight, eps)` on this rank's rows, `input` a
        sum, at the config given. Returns the normed rows."""
        # THE CALL.
        logger.debug(
            "rms_norm %s %s%s",
            tuple(input.shape),
            input.dtype,
            " (captured)" if torch.cuda.is_current_stream_capturing() else "",
        )

        # WHAT IT IS REFUSED FOR BEFORE C++ IS ASKED: a communicator that runs nothing,
        # an input that is not [rows, cols], a forcing that is not whole (C++ takes
        # every field).
        if self.disabled:
            raise RuntimeError(f"disabled: {type(self).__name__} runs no call")
        if input.dim() != 2:
            raise RuntimeError(f"not_two_d: [rows, cols], not {input.shape}")
        given_config = get_given_config(
            kernel=kernel,
            algorithm=algorithm,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_z=grid_z,
            block_x=block_x,
            block_y=block_y,
            block_z=block_z,
            waves_per_eu=waves_per_eu,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            quantization=quantization,
        )

        # A CAPTURE'S WARMUP: vLLM discards its outputs, so nothing launches.
        output = torch.empty_like(input)
        if (
            isinstance(self.state, Capturing)
            and not torch.cuda.is_current_stream_capturing()
        ):
            return output

        # THE KERNEL IT LAUNCHES: the config given (a norm on rows has no table).
        if given_config is None:
            raise RuntimeError("forced_incomplete: a norm on rows takes its config")
        config = given_config
        logger.debug("rms_norm launches %s", config)

        error = torch.ops._rocm_C.hip_custom_communicator_rms_norm(
            handle_ptr=self._handle,
            input=input,
            weight=weight,
            output=output,
            eps=eps,
            kernel=config.kernel,
            algorithm=config.algorithm,
            grid_x=config.grid_x,
            grid_y=config.grid_y,
            grid_z=config.grid_z,
            block_x=config.block_x,
            block_y=config.block_y,
            block_z=config.block_z,
            waves_per_eu=config.waves_per_eu,
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            tile_k=config.tile_k,
            quantization=config.quantization,
        )
        if error != -1:
            raise RuntimeError(f"{Error(error).name}: {config} on {input.shape}")
        return output

    @final
    def add_rms_norm(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        *,
        kernel: str | None = None,
        algorithm: Algorithm | None = None,
        grid_x: int | None = None,
        grid_y: int | None = None,
        grid_z: int | None = None,
        block_x: int | None = None,
        block_y: int | None = None,
        block_z: int | None = None,
        waves_per_eu: int | None = None,
        tile_m: int | None = None,
        tile_n: int | None = None,
        tile_k: int | None = None,
        quantization: Quantization | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`vllm.ir.ops.fused_add_rms_norm(input, residual, weight, eps)` on this rank's
        rows, `input` a sum, at the config given. Returns the normed result and the sum
        plus residual."""
        # THE CALL.
        logger.debug(
            "add_rms_norm %s %s%s",
            tuple(input.shape),
            input.dtype,
            " (captured)" if torch.cuda.is_current_stream_capturing() else "",
        )

        # WHAT IT IS REFUSED FOR BEFORE C++ IS ASKED: a communicator that runs nothing,
        # an input that is not [rows, cols], a forcing that is not whole (C++ takes
        # every field).
        if self.disabled:
            raise RuntimeError(f"disabled: {type(self).__name__} runs no call")
        if input.dim() != 2:
            raise RuntimeError(f"not_two_d: [rows, cols], not {input.shape}")
        given_config = get_given_config(
            kernel=kernel,
            algorithm=algorithm,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_z=grid_z,
            block_x=block_x,
            block_y=block_y,
            block_z=block_z,
            waves_per_eu=waves_per_eu,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            quantization=quantization,
        )

        # A CAPTURE'S WARMUP: vLLM discards its outputs, so nothing launches.
        output, residual_output = torch.empty_like(input), torch.empty_like(input)
        if (
            isinstance(self.state, Capturing)
            and not torch.cuda.is_current_stream_capturing()
        ):
            return output, residual_output

        # THE KERNEL IT LAUNCHES: the config given (a norm on rows has no table).
        if given_config is None:
            raise RuntimeError("forced_incomplete: a norm on rows takes its config")
        config = given_config
        logger.debug("add_rms_norm launches %s", config)

        error = torch.ops._rocm_C.hip_custom_communicator_add_rms_norm(
            handle_ptr=self._handle,
            input=input,
            residual=residual,
            weight=weight,
            output=output,
            residual_output=residual_output,
            eps=eps,
            kernel=config.kernel,
            algorithm=config.algorithm,
            grid_x=config.grid_x,
            grid_y=config.grid_y,
            grid_z=config.grid_z,
            block_x=config.block_x,
            block_y=config.block_y,
            block_z=config.block_z,
            waves_per_eu=config.waves_per_eu,
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            tile_k=config.tile_k,
            quantization=config.quantization,
        )
        if error != -1:
            raise RuntimeError(f"{Error(error).name}: {config} on {input.shape}")
        return output, residual_output

    @final
    def add_attn_res_rms_norm(
        self,
        input: torch.Tensor,
        prefix: torch.Tensor | None,
        blocks: torch.Tensor,
        norm_weight: torch.Tensor,
        qk_weight: torch.Tensor,
        output_norm_weight: torch.Tensor | None,
        num_blocks: int,
        write_index: int,
        eps: float,
        output_eps: float,
        *,
        kernel: str | None = None,
        algorithm: Algorithm | None = None,
        grid_x: int | None = None,
        grid_y: int | None = None,
        grid_z: int | None = None,
        block_x: int | None = None,
        block_y: int | None = None,
        block_z: int | None = None,
        waves_per_eu: int | None = None,
        tile_m: int | None = None,
        tile_n: int | None = None,
        tile_k: int | None = None,
        quantization: Quantization | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`attn_res(prefix, input, blocks, ...)` on this rank's rows, `input` a sum,
        its arguments `all_reduce_add_attn_res_rms_norm`'s, at the config given.
        Returns the new prefix and the AttnRes output."""
        # THE CALL.
        logger.debug(
            "add_attn_res_rms_norm %s %s%s",
            tuple(input.shape),
            input.dtype,
            " (captured)" if torch.cuda.is_current_stream_capturing() else "",
        )

        # WHAT IT IS REFUSED FOR BEFORE C++ IS ASKED: a communicator that runs nothing,
        # an input that is not [rows, cols], a forcing that is not whole (C++ takes
        # every field).
        if self.disabled:
            raise RuntimeError(f"disabled: {type(self).__name__} runs no call")
        if input.dim() != 2:
            raise RuntimeError(f"not_two_d: [rows, cols], not {input.shape}")
        given_config = get_given_config(
            kernel=kernel,
            algorithm=algorithm,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_z=grid_z,
            block_x=block_x,
            block_y=block_y,
            block_z=block_z,
            waves_per_eu=waves_per_eu,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            quantization=quantization,
        )

        # A CAPTURE'S WARMUP: vLLM discards its outputs, so nothing launches.
        prefix_output, output = torch.empty_like(input), torch.empty_like(input)
        if (
            isinstance(self.state, Capturing)
            and not torch.cuda.is_current_stream_capturing()
        ):
            return prefix_output, output

        # THE KERNEL IT LAUNCHES: the config given (a norm on rows has no table).
        if given_config is None:
            raise RuntimeError("forced_incomplete: a norm on rows takes its config")
        config = given_config
        logger.debug("add_attn_res_rms_norm launches %s", config)

        block_output = blocks[:, write_index] if write_index >= 0 else None
        error = torch.ops._rocm_C.hip_custom_communicator_add_attn_res_rms_norm(
            handle_ptr=self._handle,
            input=input,
            prefix=prefix,
            blocks=blocks,
            norm_weight=norm_weight,
            qk_weight=qk_weight,
            output_norm_weight=output_norm_weight,
            prefix_output=prefix_output,
            output=output,
            block_output=block_output,
            num_blocks=num_blocks,
            eps=eps,
            output_eps=output_eps,
            kernel=config.kernel,
            algorithm=config.algorithm,
            grid_x=config.grid_x,
            grid_y=config.grid_y,
            grid_z=config.grid_z,
            block_x=config.block_x,
            block_y=config.block_y,
            block_z=config.block_z,
            waves_per_eu=config.waves_per_eu,
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            tile_k=config.tile_k,
            quantization=config.quantization,
        )
        if error != -1:
            raise RuntimeError(f"{Error(error).name}: {config} on {input.shape}")
        return prefix_output, output

    @final
    def close(self) -> None:
        """Release what this communicator holds, NOW. Idempotent, and safe to call on a
        disabled one.

        Deterministic release rather than waiting to be collected, because the
        collection is what cannot be relied on: dropping the last reference frees a
        backend's IPC handles and device buffers through its destructor, but an
        exception TRACEBACK holds the frames that hold the communicator -- so a failing
        call is exactly when `del` does not release. This does.

        A COLLECTIVE, every rank calls it (STANDARDS 3.5: lifetimes are collective):
        every rank's device finishes, a handshake over the CPU group, every rank
        closes its peers' IPC mappings, a second handshake, then each frees its own
        (C++'s `Handle::close`; each step a `ClosePhase` in `log()`). Each handshake
        is bounded by `timeout_seconds`, so a rank closing alone, or a dead peer,
        costs at most that, never a hang: the failed handshake is logged and the
        teardown carries on alone. Ordering that matters is the caller's -- close
        before the process group is destroyed, and after any captured graph is gone.
        """
        if isinstance(self.state, Closed):
            return
        if isinstance(self.state, Capturing):
            raise RuntimeError(
                f"{type(self).__name__}.close() inside `capture()`: the graph being "
                f"recorded would "
                f"replay against released buffers. Leave the capture first."
            )
        # THE WATCHDOG STOPS FIRST: it reads the backend's handle, which `_on_close`
        # frees.
        self._stop_watching.set()
        if self._watchdog is not None:
            self._watchdog.join()
        with self._lock:
            self.state = Closed()
        self._on_close()

    @final
    def __enter__(self) -> "HipCustomCommunicator":
        return self

    @final
    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        # WARNS, never releases. Doing GPU or IPC work here would run at an arbitrary
        # moment -- interpreter shutdown included, where the runtime may already be gone
        # -- so this only reports that a release was left to chance. Same bargain as an
        # unclosed file's ResourceWarning. At shutdown the exit hook has already moved
        # it to Abandoned, which holds nothing.
        if self.state.holds:
            warnings.warn(
                f"{type(self).__name__} was never closed; its peer handles and buffers "
                f"were left to garbage collection. Use `with make_communicator(...) as "
                f"comm:` "
                f"or call `close()`.",
                ResourceWarning,
                stacklevel=2,
            )

    @final
    @contextmanager
    def capture(self) -> Iterator[None]:
        """Enter around a cudagraph capture. Required: a captured launch records an
        address that is not valid yet, so a backend has to be told. From `open`
        only (not nested, not closed); a capture that raises leaves it broken,
        since its peers may have registered what this rank did not."""
        with self._lock:
            if not isinstance(self.state, Open):
                raise RuntimeError(
                    f"{type(self).__name__}.capture() while {type(self.state).__name__}"
                )
            self.state = Capturing()
        try:
            with self._on_capture():
                yield
        except BaseException as e:
            self._break(f"a capture raised {type(e).__name__}: {e}")
            raise
        # Open again, unless the watchdog broke it meanwhile.
        with self._lock:
            if isinstance(self.state, Capturing):
                self.state = Open()

    @final
    def _abandoned(self) -> None:
        """Abandoned, from the exit hook once C++ has torn its mappings down: the
        watchdog told to stop (not joined: a signal handler may be running this)."""
        with self._lock:
            if not self.state.holds:
                return
            self.state = Abandoned()
        self._stop_watching.set()

    # ---- A FAULT, AND WHAT IS LOGGED WHEN ONE BREAKS IT. ----

    def _watch(self) -> None:
        """THE WATCHDOG, its own thread while open: reads the backend's fault record
        and breaks the communicator on the first. A graph replay runs none of our host
        code, so this is the only place a fault in one surfaces."""
        while not self._stop_watching.wait(_WATCH_S):
            fault = self._fault()
            if fault is not None:
                self._break(f"a kernel's wait gave up: {fault}")
                return

    @final
    def _break(self, cause: str) -> None:
        """Broken, once: every later call raises (`disabled`), every wait still on the
        device leaves (`_abort`), and the cause and the state log go to the log."""
        with self._lock:
            if isinstance(self.state, Broken | Abandoned | Closed):
                return
            self.state = Broken(cause)
        self._abort()
        logger.error("%s broken: %s.\n%s", type(self).__name__, cause, self._log_text())

    def _log_text(self) -> str:
        """The state log as text, or why there is none (a test's bare instance)."""
        try:
            got = self.log()
        except (RuntimeError, AttributeError) as e:
            return f"(no log: {e})"
        return str(got) if got is not None else "(no log)"

    # ---- THE PEER MEMORY'S LIFE. ----

    # Declared here so a disabled communicator is still safe to hold and close.
    _handle: int | None = None
    # The state log as close left it.
    _closed_log: Log | None = None
    # The thread that reads the fault record, while the communicator is open.
    _watchdog: threading.Thread | None = None
    # The device wall clock's rate, which the fault record counts its wait in.
    _khz: int = 0
    capacity: Capacity
    # The device's arch, whose tuned table the communicator runs.
    arch: str
    # The tuned tables handed in at construction (the tuner's candidate), or None: the
    # file's.
    _given_tables: TunedTables | None
    # The tuned table the plain calls look up, set at open.
    table: TunedTable
    # Whether there is a fault record for the watchdog to read (`_fault`).
    _faults: ClassVar[bool] = True

    def _open(self) -> str | None:
        """Open the peer memory. A collective, so every rank must reach it.

        Eager, and before any capture: doing this inside a cudagraph capture is not
        recoverable.
        """
        # THE TUNED TABLE for this device's arch, checked: none, and the communicator is
        # disabled, naming why, as on a device it does not support.
        found = supported(self.device, self.device_group.size())
        if isinstance(found, Error):
            return found.name
        tables = _file_tables() if self._given_tables is None else self._given_tables
        table = tables.get(found.arch)
        if table is None:
            return f"no tuned table on {found.arch}"
        check_table(table)
        self.table = table
        self.arch = found.arch
        # THE PEER MEMORY, made, sized and owned by C++, which checks the groups and the
        # device and gathers every rank's IPC handle over the CPU group (a collective).
        handle = open_handle(
            self.cpu_group.group_name,
            self.device_group.group_name,
            self.device.index,
            self.timeout_seconds,
            self.max_rows,
            self.max_cols,
            self.log_mode,
            rccl_ready(self.device_group, self.device),
        )
        if isinstance(handle, Error):
            return handle.name
        self._handle = handle
        _install_exit_hooks()
        # WHAT OPEN OBSERVED of the broader state: warnings, never a failure.
        opened = self.log()
        for warning in opened.warnings() if opened is not None else []:
            logger.warning("HipCustomCommunicator open: %s", warning)
        self._khz = torch.ops._rocm_C.hip_custom_communicator_wall_clock_khz(
            self.device.index
        )
        logger.info(
            "HipCustomCommunicator ready: exchange sized for %d rows of %d "
            "(partials %.1f MiB, staging %.1f MiB), quantization=%s",
            self.capacity.max_rows,
            self.capacity.max_cols,
            self.capacity.partials_bytes / 2**20,
            self.capacity.staging_bytes / 2**20,
            self.quantization,
        )
        return None

    @contextmanager
    def _on_capture(self) -> Iterator[None]:
        """Wrap a cudagraph capture. Each captured exact call reserves an input
        table, filled on exit: every rank exchanges the IPC handles of the inputs
        its capture recorded over the CPU group (a collective, so every rank exits
        its capture together) and opens its peers'. Before any replay."""
        yield
        failed = torch.ops._rocm_C.hip_custom_communicator_register_captured(
            self._handle, self.cpu_group.group_name
        )
        if failed:
            raise RuntimeError(
                "hip_custom_communicator: registering the captured inputs failed "
                + (
                    "(the ranks captured different numbers of all-reduces)"
                    if failed == 1
                    else "(the CPU group no longer resolves)"
                )
            )

    def _on_close(self) -> None:
        """Release the peer memory now, collectively (`close`), then the handle; the
        log as it stood is kept for `log()`. Idempotent. Dropping the object does not
        release what the C++ side opened; this does."""
        if self._handle is None:
            return
        failed = ClosePhase(
            torch.ops._rocm_C.hip_custom_communicator_close(
                self._handle, self.cpu_group.group_name, self.timeout_seconds
            )
        )
        self._closed_log = self.log()
        torch.ops._rocm_C.hip_custom_communicator_dispose(self._handle)
        self._handle = None
        if failed is not ClosePhase.none:
            logger.error(
                "HipCustomCommunicator close failed at %s (it finished alone):\n%s",
                failed.name,
                self._closed_log,
            )

    def status(self) -> Status:
        """This rank's status, read from host-mapped memory, so it never waits on the
        device: valid even while a kernel is hung."""
        if self._handle is None:
            return Status(abort=False, fault=None)
        abort, state, where, rank, block, peer, count, want, ticks = (
            torch.ops._rocm_C.hip_custom_communicator_status(self._handle)
        )
        fault = None
        if FaultState(state) is FaultState.written:
            fault = Fault(
                where=Where(where),
                rank=rank,
                block=block,
                peer=peer,
                count=count,
                want=want,
                waited_s=ticks / (self._khz * 1000.0),
            )
        return Status(abort=abort != 0, fault=fault)

    def log(self) -> Log | None:
        """This rank's state log (`Log`), read from host-mapped memory, so it never
        waits on the device: readable while a kernel is hung. Once closed, the log as
        the close left it (its close phases); None when it never opened."""
        if self._handle is None:
            return self._closed_log
        return Log.of(torch.ops._rocm_C.hip_custom_communicator_log(self._handle))

    def signals(self) -> list[Flag]:
        """Every flag of this rank's signals, each `clear` between launches. C++
        waits for every launch on the device to finish before it reads them, so a
        caller need not synchronize; a hung launch hangs this query too (`status()`
        is readable while a kernel waits)."""
        if self._handle is None:
            return []
        flat = torch.ops._rocm_C.hip_custom_communicator_signals(self._handle)
        return [
            Flag(flat[i], flat[i + 1], flat[i + 2], FlagState(flat[i + 3]))
            for i in range(0, len(flat), 4)
        ]

    def _fault(self) -> Fault | None:
        """The fault this rank's status records, for the watchdog."""
        return self.status().fault

    def _abort(self) -> None:
        if self._handle is not None:
            torch.ops._rocm_C.hip_custom_communicator_abort(self._handle)
