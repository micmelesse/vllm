# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: MIT Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""HipCustomCommunicator: vLLM's custom all-reduce (`custom_all_reduce.py`'s
CustomAllreduce) extended to every collective Kimi-K3 needs -- the all-reduce and the
all-reduces fused with the ops after them -- over our HIP kernels in `_rocm_C`
(`csrc/rocm/hip_custom_communicator.cu`).

The caller names an op and C++ picks the kernel and its launch (`select.cuh`), moving
INT4 where the communicator's `quantization` asks; the keyword arguments an op takes
after its own (the algorithm, the direction, every field of one build and the launch,
all or none) are the one way to force one, for the sweep and the tests. Every one is
passed through as it came: C++ builds the config, checks it, and answers.

THE MEMORY (C++'s `Handle`): the communicator's arena is one allocation per rank,
which every rank opened at open over legacy HIP IPC. Every exact call reads every
rank's input IN PLACE through an input table (aiter's gfx950 pattern): a captured
call's tensors, registered when `capture()` exits (one exchange of their IPC handles
over the CPU group, a collective), or an eager call's copy, which C++ makes into the
arena's inputs region on the call's stream. An INT4 call's kernel quantizes its input
into the inputs region itself.

The C++ context crosses as an opaque `int` handle, so nothing frees it for us:
`close()` has to run.
"""

import atexit
import functools
import os
import signal
import threading
import time
import warnings
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from types import FrameType
from typing import ClassVar, Literal, final, get_args

import torch
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)


# ---- THE BUILD'S FACTS, as the communicator and its callers name them: the ops, the
# Errors and fault places C++ answers with (number for number; a test holds them equal),
# the build's contents (`build_info`), whether a device runs it (`supported`), and the
# exception an op raises when it does not take a call (`Refused`). ----

# EVERY OP, as C++ names them (`enum class Op`; a test holds them equal) and as its
# method is named: the all-reduce, then each all-reduce and the ops it fuses, in order.
Op = Literal[
    "all_reduce",
    "all_reduce_rms_norm",
    "all_reduce_add_rms_norm",
    "all_reduce_add_attn_res_rms_norm",
]

# WHAT CROSSES THE LINKS, C++'s `Quantization` by name: a communicator's setting for
# the calls it picks, and a build field a forcing names.
Quantization = Literal["none", "int4"]


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
    inputs_too_small = 12
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
    quantization_not_two_shot = 31
    row_not_int4_groups = 32
    quantized_too_small = 33  # retired: INT4 lies inside inputs and partials


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
    copy_in_issued = 5  # an eager all-reduce's host copy into the inputs
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
    """One C++ template as built: its op, and the builds dispatch instantiates (its
    catalog list), each its fields by name as a forcing passes them
    (threads_per_block, waves_per_eu, its tile and its quantization; never the launch,
    which a forced call gives itself)."""

    op: str
    builds: tuple[dict[str, int | str], ...]


@dataclass(frozen=True)
class OpRecord:
    """ONE OP'S PLAIN CALL as built (C++'s `kOps`): what a call of it with nothing
    forced must be to run, which `should` reads."""

    dtypes: frozenset[torch.dtype]  # its input's
    weight_dtypes: frozenset[torch.dtype]  # its weight's besides its input's; none
    widest_row: int  # elements a row at most; 0 for any (its tiles cross the row)
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
    ) = torch.ops._rocm_C.hip_custom_communicator_build_info()
    templates: dict[str, TemplateBuild] = {}
    at, width = 0, len(fields)
    for name, op, count in zip(names, template_ops, counts):
        builds: list[dict[str, int | str]] = []
        for _ in range(count):
            given = dict(zip(fields, flat[at : at + width]))
            q = _QUANTIZATIONS[given.pop("quantization")]
            fields_set = {f: v for f, v in given.items() if v != 0}
            builds.append(fields_set | {"quantization": q})
            at += width
        templates[name] = TemplateBuild(op, tuple(builds))
    records: dict[str, OpRecord] = {}
    at_dtype = at_weight = 0
    for op, at in zip(ops, range(0, len(op_facts), 4)):
        num_dtypes, num_weights, widest_row, two_d = op_facts[at : at + 4]
        dtypes_in = op_dtypes[at_dtype : at_dtype + num_dtypes]
        weights_in = op_weights[at_weight : at_weight + num_weights]
        at_dtype += num_dtypes
        at_weight += num_weights
        records[op] = OpRecord(
            frozenset(_DTYPES[d] for d in dtypes_in),
            frozenset(_DTYPES[d] for d in weights_in),
            widest_row,
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
    )


@dataclass(frozen=True)
class Capacity:
    """WHAT ONE COMMUNICATOR HOLDS (C++'s `hip_custom_communicator_capacity`): the
    largest call its exchange was sized for, `max_rows` rows of at most `max_cols`
    elements (for a model, the server's tokens a batch by its hidden size), the
    exchange's partials and inputs bytes, and each op's most bytes on them whatever
    a call's shape."""

    max_rows: int
    max_cols: int
    partials_bytes: int
    inputs_bytes: int
    max_bytes: dict[str, int]


def capacity_of(max_rows: int, max_cols: int, world: int) -> Capacity:
    """What a communicator serving `max_rows` rows of at most `max_cols` columns at
    `world` ranks holds: C++'s answer, the same function its opener allocates by."""
    rows, cols, partials, inputs, *most = (
        torch.ops._rocm_C.hip_custom_communicator_capacity(max_rows, max_cols, world)
    )
    return Capacity(
        rows, cols, partials, inputs, dict(zip(build_info().op_names, most))
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
    runs on a communicator holding `capacity`: inside the op's record
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
    return (
        row * input.element_size() % built.pack_bytes == 0
        and (record.widest_row == 0 or row <= record.widest_row)
        and (in_box or input.numel() * input.element_size() <= capacity.max_bytes[op])
    )


class Refused(RuntimeError):
    """A call this backend does not take, raised by the op with the Error C++ refused
    it for, and the input it was asked to run on. A call `should` admits never is."""

    def __init__(self, error: Error, input: torch.Tensor) -> None:
        super().__init__(
            f"{error.name}: shape={tuple(input.shape)} dtype={input.dtype}"
        )
        self.error = error


@dataclass(frozen=True)
class Supported:
    """What the library runs on, once `supported` finds it can: the device's arch."""

    arch: str


def open_handle(
    cpu_group: str,
    device_group: str,
    device: int,
    timeout_seconds: float,
    max_rows: int,
    max_cols: int,
    quantization: Quantization,
    log_mode: LogMode = "errors",
    rccl_ready: int = -1,
) -> int | Error:
    """THE PEER MEMORY OPENED over the process groups named `cpu_group` and
    `device_group`, a collective (C++ gathers every rank's IPC handle over the CPU
    group): the handle, or the Error it was refused for. The one place Python calls
    C++'s opener, so its arguments are written once. `quantization`: what the calls C++
    picks move across the links (each op's crossover is C++'s, `select.cuh`).
    `log_mode`: what the state log records. `rccl_ready`: whether the device group's
    RCCL communicator was found set up (`rccl_ready`), logged as an observation. An
    open that fails one of its checks raises, naming it, with the log appended."""
    try:
        handle, err = torch.ops._rocm_C.hip_custom_communicator_open(
            cpu_group,
            device_group,
            device,
            timeout_seconds,
            max_rows,
            max_cols,
            _QUANTIZATIONS.index(quantization),
            _LOG_MODES.index(log_mode),
            rccl_ready,
        )
    except RuntimeError as e:
        raise RuntimeError(f"{e}\n{last_log() or 'no log was made'}") from e
    return Error(err) if err is not None else handle


def configured_quantization() -> Quantization:
    """What `VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION` asks (NONE by default:
    INT4 is lossy, so opt-in)."""
    import vllm.envs as envs

    asked = envs.VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION
    return "int4" if asked == "INT4" else "none"


def supported(device: torch.device, world: int) -> Supported | Error:
    """Whether our kernels run on `device` in a world of `world`, as the C++ build
    answers it (`hip_custom_communicator::supported`): the Supported, or the Error."""
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
# and Open or Capturing -> Broken(cause) when the watchdog finds a fault; any -> Closed.
# Only `HipCustomCommunicator` moves it, under its lock (the watchdog is another
# thread).
@dataclass(frozen=True)
class Disabled:
    why: str  # the backend's reason it cannot run here (not an error)


@dataclass(frozen=True)
class Open:
    pass


@dataclass(frozen=True)
class Capturing:
    pass


@dataclass(frozen=True)
class Broken:
    cause: str  # what left the peers' state half-done


@dataclass(frozen=True)
class Closed:
    pass


State = Disabled | Open | Capturing | Broken | Closed


# AN OP'S ANSWER, C++'s wire (`hip_custom_communicator.cu`'s Answer): the Error's
# number it was refused for, or -1 when it ran, then what ran.
Answer = list[int]
# C++'s `Algorithm`, `Direction` and `Quantization`, number for number.
_ALGORITHMS = ("one_shot", "two_shot")
_DIRECTIONS = ("pull", "push")
_QUANTIZATIONS: tuple[Quantization, ...] = ("none", "int4")


@dataclass(frozen=True)
class Ran:
    """WHAT A LAUNCH RAN, as C++ answered it: the kernel's algorithm and direction,
    its tile and threads, its grid; `tile_k` and `reduce_scatter_blocks` 0 where
    the op has none; `quantization`, what crossed the links ("none" or "int4")."""

    algorithm: str
    direction: str
    tile_m: int
    tile_n: int
    threads_per_block: int
    blocks_per_grid: int
    waves_per_eu: int
    tile_k: int
    reduce_scatter_blocks: int
    quantization: Quantization

    @classmethod
    def of(cls, answer: Answer) -> "Ran | None":
        """From an answer that ran; None for one that says nothing of what ran (the
        test's torch control answers [-1])."""
        if len(answer) < 11:
            return None
        a = answer[1:]
        return cls(
            _ALGORITHMS[a[0]],
            _DIRECTIONS[a[1]],
            a[2],
            a[3],
            a[4],
            a[5],
            a[6],
            a[7],
            a[8],
            _QUANTIZATIONS[a[9]],
        )

    def __str__(self) -> str:
        how = " int4" if self.quantization == "int4" else ""
        return (
            f"{self.direction} {self.algorithm}{how} tile {self.tile_m}x{self.tile_n}"
            f" threads {self.threads_per_block} grid {self.blocks_per_grid}"
        )


@dataclass(frozen=True)
class Call:
    """ONE LAUNCH, as the flight recorder keeps it, with what it ran. A graph's
    replays run none of this code, so a captured launch is recorded once, at its
    capture."""

    seq: int  # this communicator's launches before it
    op: Op
    shape: tuple[int, ...]
    dtype: torch.dtype
    captured: bool
    at_s: float  # time.monotonic() at the launch
    ran: Ran | None

    def __str__(self) -> str:
        how = " (captured)" if self.captured else ""
        what = f" {self.ran}" if self.ran is not None else ""
        return (
            f"#{self.seq} {self.op} {list(self.shape)} {self.dtype}{how}{what} "
            f"at {self.at_s:.3f}"
        )


# THE FLIGHT RECORDER'S LENGTH: the launches a broken communicator logs, the last first.
_RECORDED = 64
# How often the watchdog reads the fault record: a fault is reported within this of the
# wait that wrote it giving up.
_WATCH_S = 0.1


class HipCustomCommunicator:
    """vLLM's TP all-reduce and its fused ops over our HIP kernels, behind
    CudaCommunicator. One per process group, because peer pointers are group-scoped.

    The public ops own the call SEQUENCE -- the admission gate, the capture invariant
    and the flight recorder -- and the private hooks at the bottom are the kernels. The
    IPC handle exchange is done over the gloo `cpu_group`; C++ only opens the handles it
    is handed. Missing ops are an error, not a disable: falling back quietly would let
    vLLM run its own all-reduce and report the arm ready. Collectives are out-of-place.
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
            case Disabled() | Closed():
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
    ) -> None:
        """Keep the groups and the device, then `_open`, which checks the backend's
        availability. Unavailability is not an error: the caller checks `.disabled`.
        `timeout_seconds`: how long a kernel waits on a peer before it reports a
        fault; None is the device group's own (`collective_timeout_seconds`).
        `max_rows` and `max_cols`: the largest call it serves, which its exchange is
        sized for; None is the server's (`problem_size`). `quantization`: what the
        calls C++ picks move across the links, from each op's crossover up; None is
        what the environment asks (`configured_quantization`). `log_mode`: what the
        state log records (`log()`); None is what the environment asks
        (`configured_log_mode`)."""
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
        self.capacity = capacity_of(max_rows, max_cols, device_group.size())
        self._own_fields()
        why = self._open()
        if why is None:
            self.state = Open()
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
        """THE BASE'S OWN FIELDS: the lock its state moves under, the flight recorder
        and the watchdog's. `__init__` sets them first; a test's bare instance (no
        process group) sets them here too, with its `state`."""
        self._lock = threading.Lock()
        self._calls: deque[Call] = deque(maxlen=_RECORDED)
        self._launches = 0
        self._stop_watching = threading.Event()
        self._watchdog = None
        if state is not None:
            self.state = state

    # ---- What the CALLER uses. Concrete: this class owns the order. ----
    #
    # Each `should_` asks the module's `should` of the op's own tensors: whether a
    # PLAIN call (nothing forced) runs, read from the build's record of the op, with no
    # C++ call -- as vLLM's `should_custom_ar` is. A caller asks it, then calls the op.
    # Each op takes its own tensors and values, then, keyword-only, what the sweep and
    # the tests force: ALL OR NOTHING, the algorithm (`one_shot` or `two_shot`) and the
    # direction (`pull` or `push`) where the op has more than one template, every field
    # of one of its builds (`build_info().templates`), and the launch (blocks_per_grid,
    # and the AttnRes op's reduce_scatter_blocks); some but not all is
    # `forced_incomplete`. The build's `quantization` alone may be left out, and is
    # then none, never the communicator's: a forced call runs what it names. All None
    # is the backend's choice; the model forces none. A forced call asks no `should_`:
    # C++ answers it. An op raises Refused with the Error C++ refused it for; a call
    # its `should_` admitted is never refused.

    @final
    def should_allreduce(self, input: torch.Tensor) -> bool:
        """Whether `all_reduce(input)` runs here."""
        return not self.disabled and should("all_reduce", input, self.capacity)

    @final
    def all_reduce(
        self,
        input: torch.Tensor,
        *,
        algorithm: str | None = None,
        direction: str | None = None,
        tile_m: int | None = None,
        tile_n: int | None = None,
        threads_per_block: int | None = None,
        blocks_per_grid: int | None = None,
        waves_per_eu: int | None = None,
        quantization: Quantization | None = None,
    ) -> torch.Tensor:
        """EVERY all-reduce this backend's kernel can compile for, at any SIZE: size
        picks among a backend's own paths (hip's in C++), never whether it is ours.
        Returns the sum."""
        self._admit(input)
        output = torch.empty_like(input)
        if self._warming_up():
            return output
        answer = self._all_reduce(
            input,
            output,
            algorithm,
            direction,
            tile_m,
            tile_n,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
            quantization,
        )
        self._launched("all_reduce", input, answer)
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
        algorithm: str | None = None,
        direction: str | None = None,
        tile_m: int | None = None,
        tile_n: int | None = None,
        threads_per_block: int | None = None,
        blocks_per_grid: int | None = None,
        waves_per_eu: int | None = None,
        quantization: Quantization | None = None,
    ) -> torch.Tensor:
        """`vllm.ir.ops.rms_norm(all_reduce(input), weight, eps)` in one kernel. Returns
        the normed sum."""
        self._admit(input)
        output = torch.empty_like(input)
        if self._warming_up():
            return output
        answer = self._all_reduce_rms_norm(
            input,
            weight,
            output,
            eps,
            algorithm,
            direction,
            tile_m,
            tile_n,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
            quantization,
        )
        self._launched("all_reduce_rms_norm", input, answer)
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
        algorithm: str | None = None,
        direction: str | None = None,
        tile_m: int | None = None,
        tile_n: int | None = None,
        threads_per_block: int | None = None,
        blocks_per_grid: int | None = None,
        waves_per_eu: int | None = None,
        quantization: Quantization | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`vllm.ir.ops.fused_add_rms_norm(all_reduce(input), residual, weight, eps)` in
        one kernel. Returns the normed result and the sum plus residual."""
        self._admit(input)
        output, residual_output = torch.empty_like(input), torch.empty_like(input)
        if self._warming_up():
            return output, residual_output
        answer = self._all_reduce_add_rms_norm(
            input,
            residual,
            weight,
            output,
            residual_output,
            eps,
            algorithm,
            direction,
            tile_m,
            tile_n,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
            quantization,
        )
        self._launched("all_reduce_add_rms_norm", input, answer)
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
        algorithm: str | None = None,
        direction: str | None = None,
        tile_m: int | None = None,
        tile_n: int | None = None,
        tile_k: int | None = None,
        reduce_scatter_blocks: int | None = None,
        threads_per_block: int | None = None,
        blocks_per_grid: int | None = None,
        waves_per_eu: int | None = None,
        quantization: Quantization | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`attn_res(prefix, all_reduce(input), blocks, ...)` in one kernel, or with no
        `prefix` the sum starting one. Returns the new prefix and the AttnRes output,
        each a new tensor: `prefix` is only read. `write_index` >= 0 also stores the new
        prefix as that block of `blocks`, which must not be one of the `num_blocks`
        sources the call reads."""
        self._admit(input)
        prefix_output = torch.empty_like(input)
        output = torch.empty_like(input)
        if self._warming_up():
            return prefix_output, output
        # THE ONE SLOT WRITTEN, a view of `blocks`; the sources read are the first
        # `num_blocks`, so the two never overlap (C++ refuses a call where they do).
        block_output = blocks[:, write_index] if write_index >= 0 else None
        answer = self._all_reduce_add_attn_res_rms_norm(
            input,
            prefix,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight,
            prefix_output,
            output,
            block_output,
            num_blocks,
            eps,
            output_eps,
            algorithm,
            direction,
            tile_m,
            tile_n,
            tile_k,
            reduce_scatter_blocks,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
            quantization,
        )
        self._launched("all_reduce_add_attn_res_rms_norm", input, answer)
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
        # unclosed file's ResourceWarning.
        if isinstance(self.state, Open):
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
        device leaves (`_abort`), and the cause and the last launches go to the log."""
        with self._lock:
            if isinstance(self.state, Broken | Closed):
                return
            self.state = Broken(cause)
        self._abort()
        calls = "\n".join(f"  {c}" for c in reversed(self._calls))
        logger.error(
            "%s broken: %s. Its last %d launches, the last first:\n%s\n%s",
            type(self).__name__,
            cause,
            len(self._calls),
            calls or "  (none)",
            self._log_text(),
        )

    def _log_text(self) -> str:
        """The state log as text, or why there is none (a test's bare instance)."""
        try:
            got = self.log()
        except (RuntimeError, AttributeError) as e:
            return f"(no log: {e})"
        return str(got) if got is not None else "(no log)"

    @final
    def _admit(self, input: torch.Tensor) -> None:
        """Before an op: refused when no call may run (`disabled`), and raise on a
        capture outside `capture()`."""
        if self.disabled:
            raise Refused(Error.disabled, input)

    @final
    def _launched(self, op: Op, input: torch.Tensor, answer: Answer) -> None:
        """After an op's kernel hook: Refused with the Error's number it answered, or
        the launch, and what it ran, into the flight recorder."""
        if answer[0] >= 0:
            raise Refused(Error(answer[0]), input)
        self._record(op, input, Ran.of(answer))

    @final
    def _record(self, op: Op, input: torch.Tensor, ran: Ran | None = None) -> None:
        """Into the flight recorder, as a launch goes out."""
        self._calls.append(
            Call(
                seq=self._launches,
                op=op,
                shape=tuple(input.shape),
                dtype=input.dtype,
                captured=torch.cuda.is_current_stream_capturing(),
                at_s=time.monotonic(),
                ran=ran,
            )
        )
        self._launches += 1

    # ---- The rules a caller can get wrong, enforced once. ----

    def _warming_up(self) -> bool:
        """Whether this is a capture's WARMUP: inside `capture()`, the stream not
        recording. vLLM discards its outputs, so an op returns unwritten ones of the
        right shape and launches nothing, as vLLM's and aiter's custom all-reduce do."""
        return (
            isinstance(self.state, Capturing)
            and not torch.cuda.is_current_stream_capturing()
        )

    # ---- THE KERNELS: each op's launch (its `Answer`: the Error's number C++
    # refused it for, or -1 and what ran), and the peer memory's life. ----

    # Declared here so a disabled communicator is still safe to hold and close.
    _handle: int | None = None
    # The state log as close left it.
    _closed_log: Log | None = None
    # The thread that reads the fault record, while the communicator is open.
    _watchdog: threading.Thread | None = None
    # The device wall clock's rate, which the fault record counts its wait in.
    _khz: int = 0
    capacity: Capacity
    # Whether there is a fault record for the watchdog to read (`_fault`).
    _faults: ClassVar[bool] = True

    def _open(self) -> str | None:
        """Open the peer memory. A collective, so every rank must reach it.

        Eager, and before any capture: doing this inside a cudagraph capture is not
        recoverable.
        """
        # THE PEER MEMORY, made, sized and owned by C++, which checks the groups and the
        # device and gathers every rank's IPC handle over the CPU group (a collective).
        handle = open_handle(
            self.cpu_group.group_name,
            self.device_group.group_name,
            self.device.index,
            self.timeout_seconds,
            self.max_rows,
            self.max_cols,
            self.quantization,
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
            "(partials %.1f MiB, inputs %.1f MiB)",
            self.capacity.max_rows,
            self.capacity.max_cols,
            self.capacity.partials_bytes / 2**20,
            self.capacity.inputs_bytes / 2**20,
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

    def _all_reduce(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        algorithm: str | None,
        direction: str | None,
        tile_m: int | None,
        tile_n: int | None,
        threads_per_block: int | None,
        blocks_per_grid: int | None,
        waves_per_eu: int | None,
        quantization: Quantization | None,
    ) -> Answer:
        """Sum `input` across the TP ranks into `output`; the Error's number C++ refused
        the call for, or -1 and what ran (as every kernel hook answers: `Answer`)."""
        return torch.ops._rocm_C.hip_custom_communicator_all_reduce(
            self._handle,
            input,
            output,
            algorithm,
            direction,
            tile_m,
            tile_n,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
            quantization,
        )

    def _all_reduce_rms_norm(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        eps: float,
        algorithm: str | None,
        direction: str | None,
        tile_m: int | None,
        tile_n: int | None,
        threads_per_block: int | None,
        blocks_per_grid: int | None,
        waves_per_eu: int | None,
        quantization: Quantization | None,
    ) -> Answer:
        return torch.ops._rocm_C.hip_custom_communicator_all_reduce_rms_norm(
            self._handle,
            input,
            weight,
            output,
            eps,
            algorithm,
            direction,
            tile_m,
            tile_n,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
            quantization,
        )

    def _all_reduce_add_rms_norm(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        residual_output: torch.Tensor,
        eps: float,
        algorithm: str | None,
        direction: str | None,
        tile_m: int | None,
        tile_n: int | None,
        threads_per_block: int | None,
        blocks_per_grid: int | None,
        waves_per_eu: int | None,
        quantization: Quantization | None,
    ) -> Answer:
        """The normed result into `output`; the sum plus residual into
        `residual_output`."""
        return torch.ops._rocm_C.hip_custom_communicator_all_reduce_add_rms_norm(
            self._handle,
            input,
            residual,
            weight,
            output,
            residual_output,
            eps,
            algorithm,
            direction,
            tile_m,
            tile_n,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
            quantization,
        )

    def _all_reduce_add_attn_res_rms_norm(
        self,
        input: torch.Tensor,
        prefix: torch.Tensor | None,
        blocks: torch.Tensor,
        norm_weight: torch.Tensor,
        qk_weight: torch.Tensor,
        output_norm_weight: torch.Tensor | None,
        prefix_output: torch.Tensor,
        output: torch.Tensor,
        block_output: torch.Tensor | None,
        num_blocks: int,
        eps: float,
        output_eps: float,
        algorithm: str | None,
        direction: str | None,
        tile_m: int | None,
        tile_n: int | None,
        tile_k: int | None,
        reduce_scatter_blocks: int | None,
        threads_per_block: int | None,
        blocks_per_grid: int | None,
        waves_per_eu: int | None,
        quantization: Quantization | None,
    ) -> Answer:
        launch = (
            torch.ops._rocm_C.hip_custom_communicator_all_reduce_add_attn_res_rms_norm
        )
        return launch(
            self._handle,
            input,
            prefix,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight,
            prefix_output,
            output,
            block_output,
            num_blocks,
            eps,
            output_eps,
            algorithm,
            direction,
            tile_m,
            tile_n,
            tile_k,
            reduce_scatter_blocks,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
            quantization,
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
