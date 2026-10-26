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

THE MEMORY is this class's, as addresses: three buffers a rank, allocated, exchanged and
freed through C++'s small functions (`hip_custom_communicator_alloc`, `_ipc_handle`,
...), and handed to every op, which keeps nothing: SHARED (the arena: signals, staging,
partials, scales, the pushed one-shot's inbox, a guard after each), which every rank
opened at open over legacy HIP IPC; PRIVATE (the regions' device arrays, the pushed
one-shot's epochs, the input tables); HOST (the status words the kernels write and the
watchdog polls). Every exact call reads every rank's input IN PLACE through an input
table (aiter's gfx950 pattern): a captured call's tensors, registered when `capture()`
exits (one exchange of their IPC handles over the CPU group, a collective), or an eager
call's copy, which C++ makes into the arena's staging region on the call's stream. An
INT4 call's kernel quantizes its input into the staging region itself.

Nothing frees the buffers for us: `close()` has to run.
"""

import atexit
import functools
import json
import os
import signal
import socket
import threading
import warnings
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
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
Algorithm = Literal[
    "one_shot_pull", "one_shot_push", "two_shot_pull", "two_shot_push", "rows"
]
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
    unfenced_too_small = 37
    pack_out_of_range = 38


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


# THE STATUS WORDS (C++'s `StatusField`, common.cuh), in order: abort, state, where,
# rank, block, peer, count, want, and the wait's ticks, low then high.
_STATUS_ABORT = 0


def _fault_of(words: Sequence[int], khz: int) -> "Fault | None":
    """The kernels' fault record, from the status words."""
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


# ---- ABNORMAL EXIT (STANDARDS 3.5, "abnormal exit is designed for"): a process should
# not die holding its peers' memory open, nor free its own while they read it. So an
# exit, SIGTERM and SIGINT tear every open communicator down ALONE (this class's
# `_abandoned`): every wait told to leave, the device given a bounded wait, every
# BORROWED peer IPC mapping closed, then our own buffers freed once the device finished.
# It never waits on a peer: one may be dead. It is idempotent. ----

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
    for communicator in list(_open_communicators):
        communicator._abandoned(wait_s)


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


def configured_debug() -> bool:
    """Whether `VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_DEBUG` asks for debug mode (off by
    default)."""
    import vllm.envs as envs

    return bool(envs.VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_DEBUG)


def debug_path() -> Path:
    """WHERE A PROCESS'S CALLS GO IN DEBUG MODE: vLLM's cache root (VLLM_CACHE_ROOT,
    where it keeps its other per-machine files), `hip_custom_communicator/`, a file a
    process (`calls-<host>-<pid>.jsonl`), so ranks and runs never share one."""
    import socket

    import vllm.envs as envs

    return (
        Path(envs.VLLM_CACHE_ROOT)
        / "hip_custom_communicator"
        / f"calls-{socket.gethostname()}-{os.getpid()}.jsonl"
    )


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


# THE PRIVATE BUFFER'S REGION ARRAYS, in C++'s order (kSignalsArray, kStagingArray,
# ...): each the device array of every rank's address of that shared region.
_REGION_ARRAYS = ("signals", "staging", "partials", "scales", "unfenced")


@dataclass(frozen=True)
class Layout:
    """A COMMUNICATOR'S BUFFERS (C++'s `hip_custom_communicator_layout`, field for
    field): the exchange's staging and partials bytes; the SHARED buffer's bytes and its
    regions' offsets, each guard's [begin, end); the signals' bytes and the guard word;
    the PRIVATE buffer's bytes and its offsets; the region arrays' count, the most ranks
    and blocks, the input table slots and the status words."""

    staging_bytes: int
    partials_bytes: int
    shared_bytes: int
    signals: int
    staging: int
    partials: int
    scales: int
    unfenced: int
    staging_guard: int
    staging_guard_end: int
    partials_guard: int
    partials_guard_end: int
    unfenced_guard: int
    unfenced_guard_end: int
    signals_bytes: int
    guard_word: int
    private_bytes: int
    private_region_ptr_arrays: int
    private_unfenced_epochs: int
    private_tables: int
    region_ptr_arrays: int
    max_ranks: int
    max_blocks: int
    table_slots: int
    status_fields: int


def layout_of(max_rows: int, max_cols: int, world: int) -> Layout:
    """The buffers of a communicator serving `max_rows` rows of at most `max_cols` at
    `world`."""
    return Layout(
        *torch.ops._rocm_C.hip_custom_communicator_layout(max_rows, max_cols, world)
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
# A TUNED INPUT'S DTYPE's element bytes, as the table names it.
_DTYPE_BYTES: Mapping[str, int] = {"bf16": 2, "f16": 2, "fp16": 2, "f32": 4}
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
    "pack_bytes",
    "quantize_from_bytes",
)


@dataclass(frozen=True)
class Config:
    """ONE KERNEL'S CONFIG, a field each (`_CONFIG_KEYS`, C++'s arguments, in their
    order): the kernel, its algorithm, its grid, its block and the block's waves a EU,
    its tile, what it moves, its pack, and the call's bytes from which an INT4 config
    quantizes (`quantize_from_bytes`: a launch's, the tune's pick; below them its INT4
    build passes the call exact; 0, it always quantizes; an exact config's 0). Every
    op's C++ call is handed each field by name, a line each, never a tuple or a dict
    unpacked: a field one side lacks is an error naming it (STANDARDS 3)."""

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
    pack_bytes: int
    # None: an exact config's (it never quantizes), or an INT4 one's left out of a
    # forcing (C++'s 0, it always quantizes).
    quantize_from_bytes: int | None = None


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
        pack_bytes=c["pack_bytes"],
        quantize_from_bytes=c["quantize_from_bytes"],
    )


def _moves_exact(config: Config, input: torch.Tensor) -> bool:
    """WHETHER A CALL OF `config` ON `input` MOVES EXACT VALUES: an exact config, or an
    INT4 one below its quantize_from_bytes (its build's instance passing the call
    exact)."""
    return config.quantization == "exact" or input.numel() * input.element_size() < (
        config.quantize_from_bytes or 0
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
        build = {
            k: c[k]
            for k in _CONFIG_KEYS
            if k not in ("kernel", "quantize_from_bytes") and k[:5] != "grid_"
        }
        q = c["quantize_from_bytes"]
        if (c["quantization"] == "exact") != (q is None) or (
            q is not None and (not isinstance(q, int) or q < 0)
        ):
            raise ValueError(
                f"{where}: quantize_from_bytes is null for an exact config, an INT4 "
                "one's count of bytes"
            )
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
        # THE QUANTIZED ALL-REDUCE RUNS ITS INT4 KERNELS, every pick (each quantizing
        # where its config says): no exact kernel in quantized mode.
        if e["mode"] == "quantized" and name == "all_reduce" and moved != {"int4"}:
            raise ValueError(
                f"{where}: a quantized all-reduce input picks an exact kernel"
            )
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
    pack_bytes: int | None,
    quantize_from_bytes: int | None = None,
) -> Config | None:
    """AN OP'S CONFIG AS GIVEN IN ITS ARGUMENTS: every field given (the config), or none
    (None: a plain call, the table's pick). Some but not all is no config, refused,
    forced_incomplete: C++ takes every field. `quantize_from_bytes`, a launch's, may be
    left out of a forcing: None (an INT4 config then always quantizes; an exact one has
    none)."""
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
        and pack_bytes is not None
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
            pack_bytes=pack_bytes,
            quantize_from_bytes=quantize_from_bytes,
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
        and pack_bytes is None
        and quantize_from_bytes is None
    ):
        return None
    raise RuntimeError(
        "forced_incomplete: a forced call gives kernel, algorithm, grid x y z, block x "
        "y z, waves_per_eu, tile m n k, quantization and pack_bytes"
    )


def get_nearest_pick(
    table: TunedTable,
    op: Op,
    rows: int,
    cols: int,
    quantization: Quantization,
    world: int,
) -> dict[str, Any]:
    """THE PICK A PLAIN CALL OF `op` ON `rows` x `cols` RUNS, from the arch's `table`:
    its record (`input`, the table input it was tuned at; `configs`, what it launches),
    the nearest input of `op` tuned in the communicator's mode (`quantization`) at
    `world`. Among the picks whose every config's pack fits the row (whole packs of
    `pack_bytes`, at the tuned input's dtype; a config the row is not whole packs of is
    refused, row_not_packs), where any does; then the nearest width, then the nearest
    size (rows x width), the fewer tokens on a tie."""
    inputs = table["inputs"]
    picks = [
        k
        for k in table["picks"]
        if inputs[k["input"]]["op"] == _QUALIFIED + op
        and inputs[k["input"]]["mode"] == _MODES[quantization]
        and inputs[k["input"]]["world"] == world
    ]
    fitting = [
        k
        for k in picks
        if all(
            cols
            * _DTYPE_BYTES[inputs[k["input"]]["dtype"]]
            % table["configs"][j]["pack_bytes"]
            == 0
            for j in k["configs"]
        )
    ]
    picks = fitting or picks
    width = min(
        (inputs[k["input"]]["hidden"] for k in picks), key=lambda h: abs(h - cols)
    )
    return min(
        (k for k in picks if inputs[k["input"]]["hidden"] == width),
        key=lambda k: (
            abs(inputs[k["input"]]["tokens"] * width - rows * cols),
            inputs[k["input"]]["tokens"],
        ),
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
    its pick's configs (`get_nearest_pick`), in the order they launch."""
    return picked_configs(
        table, get_nearest_pick(table, op, rows, cols, quantization, world)
    )


def picked_configs(table: TunedTable, pick: Mapping[str, Any]) -> list[Config]:
    """A pick's configs (its record in the arch's `table`), in the order they launch."""
    return [_config(table["configs"][j]) for j in pick["configs"]]


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
        build = {
            k: c[k]
            for k in _CONFIG_KEYS
            if k not in ("kernel", "quantize_from_bytes") and k[:5] != "grid_"
        }
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
            return replace(
                _config(c),
                quantization=quantization,
                quantize_from_bytes=None if quantization == "exact" else 0,
            )
    return None


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
        tables: TunedTables | None = None,
        debug: bool | None = None,
    ) -> None:
        """Keep the groups and the device, then `_open`, which checks the backend's
        availability. Unavailability is not an error: the caller checks `.disabled`.
        `timeout_seconds`: how long a kernel waits on a peer before it reports a fault;
        None is the device group's own (`collective_timeout_seconds`). `max_rows` and
        `max_cols`: the largest call it serves, which its exchange is sized for; None is
        the server's (`problem_size`). `quantization`: what the plain calls move across
        the links, from each op's crossover up (their table: the quantized one where
        INT4); None is what the environment asks (`configured_quantization`). `tables`:
        the tuned tables its plain calls run from (the arch's), of the file's shape (the
        tuner's candidate); None is the file's (`hip_custom_communicator.json`), read
        once. `debug`: debug mode, every call printed and appended before it launches, a
        JSON line each (`_debug`), to this process's file under vLLM's cache root
        (`debug_path`), its path printed at open; None is what the environment asks
        (`configured_debug`), off by default."""
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
        debugging = configured_debug() if debug is None else debug
        self._given_tables = tables
        self.capacity = capacity_of(max_rows, max_cols, device_group.size())
        self._own_fields()
        if debugging:
            calls = debug_path()
            calls.parent.mkdir(parents=True, exist_ok=True)
            # LINE-BUFFERED, APPENDED: each call's line one write.
            self._debug_out = open(calls, "a", buffering=1)  # noqa: SIM115
            logger.info("%s debug: every call to %s", type(self).__name__, calls)
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
        # One teardown at a time: close, and abandon (a signal, exit).
        self._teardown = threading.Lock()
        # Nothing held yet: no peer mapping, no captured launch.
        self._peers: dict[int, int] = {}
        self._pending: list[tuple[int, int]] = []
        self._opened_inputs: dict[tuple[int, ...], int] = {}
        self._stop_watching = threading.Event()
        self._watchdog = None
        self._debug_out = None
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
    def _debug(
        self,
        op: str,
        tensors: Mapping[str, torch.Tensor | None],
        values: Mapping[str, int | float],
        configs: Sequence[Config],
        pick: Mapping[str, Any] | None,
    ) -> None:
        """THE CALL'S LINE IN DEBUG MODE (`debug`), printed and appended to the
        process's file (`debug_path`) before it launches: when, the rank and the world,
        the op, the communicator's quantization, whether a graph is being captured,
        every tensor's shape, dtype and
        strides (null where one is not given), the call's values, the configs it
        launches in order (each field by name), and the table input their pick was
        tuned at (null for a forced call). Off, nothing: no work on the call's path."""
        if self._debug_out is None:
            return
        line = {
            "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "rank": self.device_group.rank(),
            "world": self.device_group.size(),
            "op": op,
            "quantization": self.quantization,
            "captured": torch.cuda.is_current_stream_capturing(),
            "tensors": {
                name: None
                if t is None
                else {
                    "shape": list(t.shape),
                    "dtype": str(t.dtype).removeprefix("torch."),
                    "stride": list(t.stride()),
                }
                for name, t in tensors.items()
            },
            "values": dict(values),
            "configs": [asdict(c) for c in configs],
            "table_input": (
                None if pick is None else self.table["inputs"][pick["input"]]
            ),
        }
        logger.info(
            "%s rank %d: %s %s %s launches %s (its pick tuned at %s)",
            type(self).__name__,
            line["rank"],
            op,
            self.quantization,
            line["tensors"],
            line["configs"],
            line["table_input"],
        )
        self._debug_out.write(json.dumps(line) + "\n")

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
        pack_bytes: int | None = None,
        quantize_from_bytes: int | None = None,
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
            pack_bytes=pack_bytes,
            quantize_from_bytes=quantize_from_bytes,
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
        pick = None
        if given_config is not None:
            configs = [given_config]
        else:
            rows, cols = input.shape
            pick = get_nearest_pick(
                self.table,
                "all_reduce",
                rows,
                cols,
                self.quantization,
                self.device_group.size(),
            )
            configs = picked_configs(self.table, pick)
        logger.debug("all_reduce launches %s", configs)
        self._debug(
            "all_reduce",
            {"input": input},
            {},
            configs,
            pick,
        )

        match configs:
            case [config]:
                error = torch.ops._rocm_C.hip_custom_communicator_all_reduce(
                    rank=self._rank,
                    world=self._world,
                    shared_buffer=self._shared,
                    private_buffer=self._private,
                    status_device=self._status_device,
                    timeout_ticks=self._timeout_ticks,
                    max_rows=self.max_rows,
                    staging_bytes=self._layout.staging_bytes,
                    partials_bytes=self._layout.partials_bytes,
                    table_slot=self._table_slot(
                        input,
                        _moves_exact(config, input)
                        and not config.algorithm.endswith("_push"),
                    ),
                    checked_after_launch=self._checked,
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
                    pack_bytes=config.pack_bytes,
                    quantize_from_bytes=config.quantize_from_bytes,
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
        pack_bytes: int | None = None,
        quantize_from_bytes: int | None = None,
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
            pack_bytes=pack_bytes,
            quantize_from_bytes=quantize_from_bytes,
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
        pick = None
        if given_config is not None:
            configs = [given_config]
        else:
            rows, cols = input.shape
            pick = get_nearest_pick(
                self.table,
                "all_reduce_rms_norm",
                rows,
                cols,
                self.quantization,
                self.device_group.size(),
            )
            configs = picked_configs(self.table, pick)
        logger.debug("all_reduce_rms_norm launches %s", configs)
        self._debug(
            "all_reduce_rms_norm",
            {"input": input, "weight": weight},
            {"eps": eps},
            configs,
            pick,
        )

        match configs:
            case [fused_config]:
                # ONE KERNEL, the fused op.
                error = torch.ops._rocm_C.hip_custom_communicator_all_reduce_rms_norm(
                    rank=self._rank,
                    world=self._world,
                    shared_buffer=self._shared,
                    private_buffer=self._private,
                    status_device=self._status_device,
                    timeout_ticks=self._timeout_ticks,
                    max_rows=self.max_rows,
                    staging_bytes=self._layout.staging_bytes,
                    partials_bytes=self._layout.partials_bytes,
                    table_slot=self._table_slot(
                        input, fused_config.quantization == "exact"
                    ),
                    checked_after_launch=self._checked,
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
                    pack_bytes=fused_config.pack_bytes,
                    quantize_from_bytes=fused_config.quantize_from_bytes,
                )
                if error != -1:
                    raise RuntimeError(
                        f"{Error(error).name}: {fused_config} on {input.shape}"
                    )
            case [reduce_config, norm_config]:
                # TWO KERNELS OF OURS: our all-reduce into this rank's sum, then the
                # norm on its rows. THE SUM IS THE OUTPUT ITSELF (ours, fresh): the norm writes
                # its rows over them in place (STANDARDS 3.5), no tensor of its own.
                error = torch.ops._rocm_C.hip_custom_communicator_all_reduce(
                    rank=self._rank,
                    world=self._world,
                    shared_buffer=self._shared,
                    private_buffer=self._private,
                    status_device=self._status_device,
                    timeout_ticks=self._timeout_ticks,
                    max_rows=self.max_rows,
                    staging_bytes=self._layout.staging_bytes,
                    partials_bytes=self._layout.partials_bytes,
                    table_slot=self._table_slot(
                        input,
                        _moves_exact(reduce_config, input)
                        and not reduce_config.algorithm.endswith("_push"),
                    ),
                    checked_after_launch=self._checked,
                    input=input,
                    output=output,
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
                    pack_bytes=reduce_config.pack_bytes,
                    quantize_from_bytes=reduce_config.quantize_from_bytes,
                )
                if error != -1:
                    raise RuntimeError(
                        f"{Error(error).name}: {reduce_config} on {input.shape}"
                    )
                error = torch.ops._rocm_C.hip_custom_communicator_rms_norm(
                    world=self._world,
                    checked_after_launch=self._checked,
                    input=output,
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
                    pack_bytes=norm_config.pack_bytes,
                    quantize_from_bytes=norm_config.quantize_from_bytes,
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
        pack_bytes: int | None = None,
        quantize_from_bytes: int | None = None,
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
            pack_bytes=pack_bytes,
            quantize_from_bytes=quantize_from_bytes,
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
        pick = None
        if given_config is not None:
            configs = [given_config]
        else:
            rows, cols = input.shape
            pick = get_nearest_pick(
                self.table,
                "all_reduce_add_rms_norm",
                rows,
                cols,
                self.quantization,
                self.device_group.size(),
            )
            configs = picked_configs(self.table, pick)
        logger.debug("all_reduce_add_rms_norm launches %s", configs)
        self._debug(
            "all_reduce_add_rms_norm",
            {"input": input, "residual": residual, "weight": weight},
            {"eps": eps},
            configs,
            pick,
        )

        match configs:
            case [fused_config]:
                # ONE KERNEL, the fused op.
                error = (
                    torch.ops._rocm_C.hip_custom_communicator_all_reduce_add_rms_norm(
                        rank=self._rank,
                        world=self._world,
                        shared_buffer=self._shared,
                        private_buffer=self._private,
                        status_device=self._status_device,
                        timeout_ticks=self._timeout_ticks,
                        max_rows=self.max_rows,
                        staging_bytes=self._layout.staging_bytes,
                        partials_bytes=self._layout.partials_bytes,
                        table_slot=self._table_slot(
                            input, fused_config.quantization == "exact"
                        ),
                        checked_after_launch=self._checked,
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
                        pack_bytes=fused_config.pack_bytes,
                        quantize_from_bytes=fused_config.quantize_from_bytes,
                    )
                )
                if error != -1:
                    raise RuntimeError(
                        f"{Error(error).name}: {fused_config} on {input.shape}"
                    )
            case [reduce_config, norm_config]:
                # TWO KERNELS OF OURS: our all-reduce into this rank's sum, then the
                # add and norm on its rows. THE SUM IS THE OUTPUT ITSELF (ours, fresh): the norm writes
                # its rows over them in place (STANDARDS 3.5), no tensor of its own.
                error = torch.ops._rocm_C.hip_custom_communicator_all_reduce(
                    rank=self._rank,
                    world=self._world,
                    shared_buffer=self._shared,
                    private_buffer=self._private,
                    status_device=self._status_device,
                    timeout_ticks=self._timeout_ticks,
                    max_rows=self.max_rows,
                    staging_bytes=self._layout.staging_bytes,
                    partials_bytes=self._layout.partials_bytes,
                    table_slot=self._table_slot(
                        input,
                        _moves_exact(reduce_config, input)
                        and not reduce_config.algorithm.endswith("_push"),
                    ),
                    checked_after_launch=self._checked,
                    input=input,
                    output=output,
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
                    pack_bytes=reduce_config.pack_bytes,
                    quantize_from_bytes=reduce_config.quantize_from_bytes,
                )
                if error != -1:
                    raise RuntimeError(
                        f"{Error(error).name}: {reduce_config} on {input.shape}"
                    )
                error = torch.ops._rocm_C.hip_custom_communicator_add_rms_norm(
                    world=self._world,
                    checked_after_launch=self._checked,
                    input=output,
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
                    pack_bytes=norm_config.pack_bytes,
                    quantize_from_bytes=norm_config.quantize_from_bytes,
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
        pack_bytes: int | None = None,
        quantize_from_bytes: int | None = None,
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
            pack_bytes=pack_bytes,
            quantize_from_bytes=quantize_from_bytes,
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
        pick = None
        if given_config is not None:
            configs = [given_config]
        else:
            rows, cols = input.shape
            pick = get_nearest_pick(
                self.table,
                "all_reduce_add_attn_res_rms_norm",
                rows,
                cols,
                self.quantization,
                self.device_group.size(),
            )
            configs = picked_configs(self.table, pick)
        logger.debug("all_reduce_add_attn_res_rms_norm launches %s", configs)
        self._debug(
            "all_reduce_add_attn_res_rms_norm",
            {
                "input": input,
                "prefix": prefix,
                "blocks": blocks,
                "norm_weight": norm_weight,
                "qk_weight": qk_weight,
                "output_norm_weight": output_norm_weight,
            },
            {
                "num_blocks": num_blocks,
                "write_index": write_index,
                "eps": eps,
                "output_eps": output_eps,
            },
            configs,
            pick,
        )

        # THE ONE SLOT WRITTEN, a view of `blocks`; the sources read are the first
        # `num_blocks`, so the two never overlap (C++ refuses a call where they do).
        block_output = blocks[:, write_index] if write_index >= 0 else None
        match configs:
            case [fused_config]:
                # ONE KERNEL, the fused op.
                error = torch.ops._rocm_C.hip_custom_communicator_all_reduce_add_attn_res_rms_norm(  # noqa: E501
                    rank=self._rank,
                    world=self._world,
                    shared_buffer=self._shared,
                    private_buffer=self._private,
                    status_device=self._status_device,
                    timeout_ticks=self._timeout_ticks,
                    max_rows=self.max_rows,
                    staging_bytes=self._layout.staging_bytes,
                    partials_bytes=self._layout.partials_bytes,
                    table_slot=self._table_slot(
                        input, fused_config.quantization == "exact"
                    ),
                    checked_after_launch=self._checked,
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
                    pack_bytes=fused_config.pack_bytes,
                    quantize_from_bytes=fused_config.quantize_from_bytes,
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
                    rank=self._rank,
                    world=self._world,
                    shared_buffer=self._shared,
                    private_buffer=self._private,
                    status_device=self._status_device,
                    timeout_ticks=self._timeout_ticks,
                    max_rows=self.max_rows,
                    staging_bytes=self._layout.staging_bytes,
                    partials_bytes=self._layout.partials_bytes,
                    table_slot=self._table_slot(
                        input,
                        _moves_exact(reduce_config, input)
                        and not reduce_config.algorithm.endswith("_push"),
                    ),
                    checked_after_launch=self._checked,
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
                    pack_bytes=reduce_config.pack_bytes,
                    quantize_from_bytes=reduce_config.quantize_from_bytes,
                )
                if error != -1:
                    raise RuntimeError(
                        f"{Error(error).name}: {reduce_config} on {input.shape}"
                    )
                error = torch.ops._rocm_C.hip_custom_communicator_add_attn_res_rms_norm(
                    world=self._world,
                    checked_after_launch=self._checked,
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
                    pack_bytes=norm_config.pack_bytes,
                    quantize_from_bytes=norm_config.quantize_from_bytes,
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
        pack_bytes: int | None = None,
        quantize_from_bytes: int | None = None,
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
            pack_bytes=pack_bytes,
            quantize_from_bytes=quantize_from_bytes,
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
        self._debug(
            "rms_norm",
            {"input": input, "weight": weight},
            {"eps": eps},
            [config],
            None,
        )

        error = torch.ops._rocm_C.hip_custom_communicator_rms_norm(
            world=self._world,
            checked_after_launch=self._checked,
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
            pack_bytes=config.pack_bytes,
            quantize_from_bytes=config.quantize_from_bytes,
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
        pack_bytes: int | None = None,
        quantize_from_bytes: int | None = None,
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
            pack_bytes=pack_bytes,
            quantize_from_bytes=quantize_from_bytes,
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
        self._debug(
            "add_rms_norm",
            {"input": input, "residual": residual, "weight": weight},
            {"eps": eps},
            [config],
            None,
        )

        error = torch.ops._rocm_C.hip_custom_communicator_add_rms_norm(
            world=self._world,
            checked_after_launch=self._checked,
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
            pack_bytes=config.pack_bytes,
            quantize_from_bytes=config.quantize_from_bytes,
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
        pack_bytes: int | None = None,
        quantize_from_bytes: int | None = None,
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
            pack_bytes=pack_bytes,
            quantize_from_bytes=quantize_from_bytes,
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
        self._debug(
            "add_attn_res_rms_norm",
            {
                "input": input,
                "prefix": prefix,
                "blocks": blocks,
                "norm_weight": norm_weight,
                "qk_weight": qk_weight,
                "output_norm_weight": output_norm_weight,
            },
            {
                "num_blocks": num_blocks,
                "write_index": write_index,
                "eps": eps,
                "output_eps": output_eps,
            },
            [config],
            None,
        )

        block_output = blocks[:, write_index] if write_index >= 0 else None
        error = torch.ops._rocm_C.hip_custom_communicator_add_attn_res_rms_norm(
            world=self._world,
            checked_after_launch=self._checked,
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
            pack_bytes=config.pack_bytes,
            quantize_from_bytes=config.quantize_from_bytes,
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
        (`_on_close`). Each handshake
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
        # THE WATCHDOG STOPS FIRST: it reads the status words, which `_on_close`
        # frees.
        self._stop_watching.set()
        if self._watchdog is not None:
            self._watchdog.join()
        with self._lock:
            self.state = Closed()
        self._on_close()
        if self._debug_out is not None:
            self._debug_out.close()
            self._debug_out = None

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
    def _abandoned(self, wait_s: float = ABANDON_WAIT_S) -> None:
        """ABANDONED, from the exit hook (`abandon_open`): torn down ALONE, never
        waiting on a peer (STANDARDS 3.5). The watchdog told to stop (not joined: a
        signal handler may be running this); every wait on this rank told to leave
        (`_abort`); the device given `wait_s` to finish; every BORROWED peer mapping
        closed; and, only once the device finished, our own shared and private buffers
        freed (freeing memory a running kernel reads would fault it). The status words
        stay: the watchdog may still read them. A no-op while another teardown runs."""
        with self._lock:
            if not self.state.holds:
                return
            self.state = Abandoned()
        self._stop_watching.set()
        if self._shared is None and self._private is None:
            return
        if not self._teardown.acquire(blocking=False):
            return
        try:
            self._abort()
            finished = _synchronized_within(self.device, wait_s)
            self._release_peers()
            if finished:
                self._release_own()
        finally:
            self._teardown.release()

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
        device leaves (`_abort`), and the cause and the status go to the log."""
        with self._lock:
            if isinstance(self.state, Broken | Abandoned | Closed):
                return
            self.state = Broken(cause)
        self._abort()
        logger.error("%s broken: %s. %s", type(self).__name__, cause, self.status())

    # ---- THE PEER MEMORY'S LIFE: three buffers a rank, this class's, as addresses.
    # ----

    # Declared here so a disabled communicator is still safe to hold and close: None,
    # nothing held. HOST: the status words, as the host and the device reach them.
    _status_host: int | None = None
    _status_device: int | None = None
    # SHARED: this rank's (OWNED), and each peer's as its IPC opening gave it
    # (BORROWED), by rank.
    _shared: int | None = None
    # PRIVATE: this rank's (OWNED).
    _private: int | None = None
    # THE CAPTURED LAUNCHES: the next free table slot, those awaiting registration
    # (slot, input address), and every peer allocation opened for one, by its IPC handle
    # (BORROWED).
    _next_slot: int = 1
    # What every op is handed: this rank, the world, the buffers' layout, a wait's bound
    # in the device wall clock's ticks, and whether each launch is checked after
    # (VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_CHECKED=1).
    _rank: int = 0
    _world: int = 1
    _layout: Layout
    _timeout_ticks: int = 0
    _checked: bool = False
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
        recoverable. An open that raises frees what it made, alone, and raises.
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
        # THE GROUPS: one world, this rank the same in both.
        if (
            self.cpu_group.size() != self.device_group.size()
            or self.cpu_group.rank() != self.device_group.rank()
        ):
            return Error.groups_disagree.name
        if torch.cuda.current_device() != self.device.index:
            raise RuntimeError(
                f"hip_custom_communicator open: the current device is "
                f"{torch.cuda.current_device()}, not the communicator's "
                f"{self.device.index}"
            )
        self._rank = self.device_group.rank()
        self._world = self.device_group.size()
        self._layout = layout_of(self.max_rows, self.max_cols, self._world)
        self._khz = torch.ops._rocm_C.hip_custom_communicator_wall_clock_khz(
            self.device.index
        )
        self._timeout_ticks = int(self.timeout_seconds * self._khz * 1000)
        checked = os.environ.get("VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_CHECKED", "")
        self._checked = checked == "1"
        self._next_slot = 1
        try:
            self._open_buffers()
        except BaseException:
            self._release_peers()
            self._release_own()
            self._release_status()
            raise
        _install_exit_hooks()
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

    def _open_buffers(self) -> None:
        """THE THREE BUFFERS made, the shared ones exchanged (a collective over the CPU
        group), and the private one written. Each step an assert that raises naming
        it."""
        ops = torch.ops._rocm_C
        layout = self._layout
        # HOST: the status words, zeroed.
        self._status_host, self._status_device = ops.hip_custom_communicator_alloc_host(
            4 * layout.status_fields
        )
        # SHARED: zeroed, then each region's guard set, before any peer can see it; this
        # rank's signals read back zero.
        self._shared = ops.hip_custom_communicator_alloc(layout.shared_bytes, True)
        for begin, end in (
            (layout.staging_guard, layout.staging_guard_end),
            (layout.partials_guard, layout.partials_guard_end),
            (layout.unfenced_guard, layout.unfenced_guard_end),
        ):
            ops.hip_custom_communicator_fill_words(
                self._shared + begin, layout.guard_word, (end - begin) // 4
            )
        nonzero = sum(
            w != 0
            for w in ops.hip_custom_communicator_read_words(
                self._shared + layout.signals, 64
            )
        )
        if nonzero:
            raise RuntimeError(
                f"hip_custom_communicator open: {nonzero} of this rank's first 64 "
                "signal words are not zero after the memset"
            )
        # PRIVATE: zeroed.
        self._private = ops.hip_custom_communicator_alloc(layout.private_bytes, False)
        # THE EXCHANGE, one collective over the CPU group: every rank's shared buffer's
        # IPC handle, its host, its GPU and its pid.
        props = torch.cuda.get_device_properties(self.device)
        me = {
            "handle": ops.hip_custom_communicator_ipc_handle(self._shared),
            "host": socket.gethostname(),
            "gpu": str(getattr(props, "uuid", "")),
            "pid": os.getpid(),
        }
        everyone: list[Any] = [None] * self._world
        torch.distributed.all_gather_object(everyone, me, group=self.cpu_group)
        # EVERY PEER'S GPU, where this process sees it (by its UUID), reachable from
        # ours: a rank that cannot reach a peer's GPU is refused here, by name, not at
        # its first call.
        here = {
            str(getattr(torch.cuda.get_device_properties(d), "uuid", "")): d
            for d in range(torch.cuda.device_count())
        }
        for r, peer in enumerate(everyone):
            d = here.get(peer["gpu"])
            if r == self._rank or d is None or d == self.device.index:
                continue
            if not torch.cuda.can_device_access_peer(self.device.index, d):
                raise RuntimeError(
                    f"hip_custom_communicator open: device {self.device.index} cannot "
                    f"access rank {r}'s device {d}"
                )
        for r, peer in enumerate(everyone):
            if r == self._rank:
                continue
            if peer["host"] != me["host"]:
                logger.warning(
                    "HipCustomCommunicator open: rank %d is on another host", r
                )
            elif peer["gpu"] and peer["gpu"] == me["gpu"]:
                logger.warning(
                    "HipCustomCommunicator open: rank %d (pid %d) is on this rank's "
                    "GPU",
                    r,
                    peer["pid"],
                )
        # EVERY PEER'S SHARED BUFFER opened (BORROWED); ours is not. Each base, as this
        # process reaches it, the base of device memory.
        for r, peer in enumerate(everyone):
            if peer["handle"][-1] != 0:
                raise RuntimeError(
                    f"hip_custom_communicator open: rank {r}'s shared buffer is not "
                    f"the base of its allocation (offset {peer['handle'][-1]})"
                )
            if r != self._rank:
                self._peers[r] = ops.hip_custom_communicator_ipc_open(peer["handle"])
        bases = [
            self._shared if r == self._rank else self._peers[r]
            for r in range(self._world)
        ]
        for r, base in enumerate(bases):
            if ops.hip_custom_communicator_pointer_device(base) < 0:
                raise RuntimeError(
                    f"hip_custom_communicator open: rank {r}'s shared buffer, as this "
                    "process opened it, is not the base of device memory"
                )
        # THE PRIVATE BUFFER WRITTEN: each region's device array of every rank's address
        # of it (0 past the world), and the input tables' slot 0, the eager table: every
        # rank's staging.
        if len(_REGION_ARRAYS) != layout.region_ptr_arrays:
            raise RuntimeError(
                f"hip_custom_communicator open: C++ has {layout.region_ptr_arrays} "
                f"region arrays, Python names {len(_REGION_ARRAYS)}"
            )
        arrays = [0] * (layout.region_ptr_arrays * layout.max_ranks)
        for a, region in enumerate(_REGION_ARRAYS):
            for r, base in enumerate(bases):
                arrays[a * layout.max_ranks + r] = base + getattr(layout, region)
        ops.hip_custom_communicator_write_words(
            self._private + layout.private_region_ptr_arrays, arrays
        )
        eager = [base + layout.staging for base in bases]
        ops.hip_custom_communicator_write_words(
            self._private + layout.private_tables,
            eager + [0] * (layout.max_ranks - self._world),
        )

    def _table_slot(self, input: torch.Tensor, reads_table: bool) -> int:
        """THE INPUT TABLE SLOT a collective's launch takes (C++'s INPUT TABLES): 0,
        eager, or where its kernel does not read the table (`reads_table`: C++'s rule,
        an exact pulled all-reduce and an exact fused op); capturing, a fresh one, never
        reused (a graph's addresses die with it), its input remembered for `_on_capture`
        to register. C++ raises where the two disagree."""
        if not reads_table or not torch.cuda.is_current_stream_capturing():
            return 0
        slot = self._next_slot
        if slot >= self._layout.table_slots:
            raise RuntimeError(
                f"hip_custom_communicator: the input tables are full "
                f"({self._layout.table_slots - 1} captured launches)"
            )
        self._next_slot += 1
        self._pending.append((slot, input.data_ptr()))
        return slot

    @contextmanager
    def _on_capture(self) -> Iterator[None]:
        """Wrap a cudagraph capture. Each captured collective reserved an input table
        slot (`_table_slot`), filled on exit, before any replay: the ranks' counts
        compared, then every rank's recorded inputs' IPC handles exchanged over the CPU
        group (a collective, so every rank exits its capture together), each peer
        allocation opened once (cached by its handle: the caching allocator hands every
        graph the same few segments) and every slot written."""
        yield
        if self._private is None:
            return
        ops = torch.ops._rocm_C
        layout = self._layout
        pending, self._pending = self._pending, []
        counts: list[Any] = [None] * self._world
        torch.distributed.all_gather_object(counts, len(pending), group=self.cpu_group)
        if any(c != len(pending) for c in counts):
            raise RuntimeError(
                "hip_custom_communicator: registering the captured inputs failed (the "
                f"ranks captured different numbers of collectives: {counts})"
            )
        mine = [ops.hip_custom_communicator_ipc_handle(ptr) for _, ptr in pending]
        everyone: list[Any] = [None] * self._world
        torch.distributed.all_gather_object(everyone, mine, group=self.cpu_group)
        for i, (slot, ptr) in enumerate(pending):
            table = [0] * layout.max_ranks
            for r in range(self._world):
                if r == self._rank:
                    table[r] = ptr
                    continue
                handle = everyone[r][i]
                key = tuple(handle[:-1])
                if key not in self._opened_inputs:
                    self._opened_inputs[key] = ops.hip_custom_communicator_ipc_open(
                        handle
                    )
                table[r] = self._opened_inputs[key] + handle[-1]
            ops.hip_custom_communicator_write_words(
                self._private + layout.private_tables + slot * layout.max_ranks * 8,
                table,
            )

    def _barrier(self) -> bool:
        """EVERY RANK ARRIVED over the CPU group within `timeout_seconds`, or False (a
        peer that did not, or a group that failed): never unbounded, a peer may be
        dead."""
        try:
            torch.distributed.barrier(group=self.cpu_group, async_op=True).wait(
                timedelta(seconds=self.timeout_seconds)
            )
        except RuntimeError:
            return False
        return True

    def _release_peers(self) -> None:
        """EVERY BORROWED MAPPING closed: the peers' shared buffers and their captured
        inputs. Idempotent; carries on past an error."""
        ops = torch.ops._rocm_C
        for base in [*self._peers.values(), *self._opened_inputs.values()]:
            if e := ops.hip_custom_communicator_ipc_close(base):
                logger.error(
                    "HipCustomCommunicator: closing a peer mapping: hipError %d", e
                )
        self._peers = {}
        self._opened_inputs = {}

    def _release_own(self) -> None:
        """OUR OWN shared and private buffers freed: once no peer has them open.
        Idempotent."""
        ops = torch.ops._rocm_C
        for name in ("_shared", "_private"):
            ptr = getattr(self, name)
            if ptr is not None:
                if e := ops.hip_custom_communicator_free(ptr):
                    logger.error(
                        "HipCustomCommunicator: freeing %s: hipError %d", name, e
                    )
                setattr(self, name, None)

    def _release_status(self) -> None:
        """THE STATUS WORDS freed: last, once nothing reads them (the watchdog
        joined)."""
        if self._status_host is not None:
            if e := torch.ops._rocm_C.hip_custom_communicator_free_host(
                self._status_host
            ):
                logger.error(
                    "HipCustomCommunicator: freeing the status: hipError %d", e
                )
            self._status_host = None
            self._status_device = None

    def _on_close(self) -> None:
        """CLOSE, A COLLECTIVE (STANDARDS 3.5, "lifetimes are collective"): every rank
        calls it together. (a) every launch on this device finished, so nothing of ours
        reads or writes a peer's memory; (b) handshake: every rank has stopped; (c)
        every borrowed mapping closed; (d) handshake: every rank has closed its mappings
        of ours (skipped when (b) failed: a peer that missed it would pair this one with
        its own first); (e) our own buffers freed. A handshake that fails (a peer that
        did not arrive within the bound) is logged and the teardown carries on alone.
        After an abandon, only what it left. Idempotent."""
        with self._teardown:
            if self._shared is None and self._private is None:
                self._release_status()
                return
            try:
                torch.cuda.synchronize(self.device)
            except RuntimeError as e:
                # A KERNEL FAULT IS STICKY: logged, and the teardown carries on (a
                # process must not die holding its peers' memory open).
                logger.error(
                    "HipCustomCommunicator close: the device did not finish: %s", e
                )
            stopped = self._barrier()
            self._release_peers()
            done = stopped and self._barrier()
            self._release_own()
            self._release_status()
        if not (stopped and done):
            logger.error(
                "HipCustomCommunicator close: a peer did not arrive at the %s "
                "handshake within %.0f s; it finished alone",
                "first" if not stopped else "second",
                self.timeout_seconds,
            )

    def status(self) -> Status:
        """This rank's status, read from host-mapped memory, so it never waits on the
        device: valid even while a kernel is hung."""
        if self._status_host is None:
            return Status(abort=False, fault=None)
        words = torch.ops._rocm_C.hip_custom_communicator_read_host(
            self._status_host, self._layout.status_fields
        )
        return Status(
            abort=words[_STATUS_ABORT] != 0, fault=_fault_of(words, self._khz)
        )

    def signals(self) -> list[Flag]:
        """Every flag of this rank's signals, each `clear` between launches. Every
        launch on the device finished first, here (read mid-launch, a barrier shows its
        peers' flags set and this rank's not yet cleared); a hung launch hangs this
        query too (`status()` is readable while a kernel waits)."""
        if self._shared is None:
            return []
        torch.cuda.synchronize(self.device)
        layout = self._layout
        words = torch.ops._rocm_C.hip_custom_communicator_read_words(
            self._shared + layout.signals, layout.signals_bytes // 4
        )
        parity = layout.max_blocks * layout.max_ranks
        return [
            Flag(p, b, w, FlagState(words[p * parity + b * layout.max_ranks + w]))
            for p in range(2)
            for b in range(layout.max_blocks)
            for w in range(layout.max_ranks)
        ]

    def _guards_broken(self) -> str:
        """THIS RANK'S GUARDS, every launch finished: "" where every word is the guard
        word, else the first region a write went past, and where."""
        if self._shared is None:
            return ""
        torch.cuda.synchronize(self.device)
        layout = self._layout
        for name, begin, end in (
            ("staging", layout.staging_guard, layout.staging_guard_end),
            ("partials", layout.partials_guard, layout.partials_guard_end),
            ("unfenced", layout.unfenced_guard, layout.unfenced_guard_end),
        ):
            words = torch.ops._rocm_C.hip_custom_communicator_read_words(
                self._shared + begin, (end - begin) // 4
            )
            for i, w in enumerate(words):
                if w != layout.guard_word:
                    return (
                        f"the {name} guard's word {i} of {len(words)} is {w} (a "
                        f"write past the {name} region)"
                    )
        return ""

    def check_after(self, what: str) -> None:
        """EVERY EXPECTATION AFTER LAUNCHES, as C++ checks after each launch in its checked mode
        (VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_CHECKED=1), for launches it could not check
        itself: a CUDA graph's replays (`what` names them). Every launch on the device
        finished, then no write past a region of this rank's shared buffer (its guards),
        no fault or abort in its status, every signal clear (read between two barriers
        of every rank: a collective, every rank calls it). Raises RuntimeError naming
        the first that fails."""
        if self._shared is None:
            return
        broken = self._guards_broken()
        if broken:
            raise RuntimeError(f"hip_custom_communicator after {what}: {broken}")
        status = self.status()
        if status.abort or status.fault is not None:
            raise RuntimeError(f"hip_custom_communicator after {what}: {status}")
        # EVERY SIGNAL CLEAR once every rank finished and none started its next launch: read
        # between two barriers of every rank (a faster peer's next launch would set its first
        # barrier's flags here: the tests' _assert_clean).
        torch.distributed.barrier(group=self.cpu_group)
        left = [f for f in self.signals() if f.state is not FlagState.clear]
        torch.distributed.barrier(group=self.cpu_group)
        if left:
            raise RuntimeError(
                f"hip_custom_communicator after {what}: {len(left)} signals left set, "
                f"{left[:4]}"
            )

    def _fault(self) -> Fault | None:
        """The fault this rank's status records, for the watchdog."""
        return self.status().fault

    def _abort(self) -> None:
        """Every wait on this rank leaves at its next check: for a communicator the host
        has given up on."""
        if self._status_host is not None:
            torch.ops._rocm_C.hip_custom_communicator_write_host(
                self._status_host, _STATUS_ABORT, 1
            )


def _synchronized_within(device: torch.device, seconds: float) -> bool:
    """EVERY LAUNCH ON `device` FINISHED within `seconds`, or False: the synchronize
    runs on its own thread, left behind when it does not return (the process is going; a
    peer may be dead)."""
    failed: list[BaseException] = []

    def synchronize() -> None:
        try:
            torch.cuda.synchronize(device)
        except BaseException as e:  # noqa: BLE001 -- reported as not finished
            failed.append(e)

    t = threading.Thread(target=synchronize, daemon=True)
    t.start()
    t.join(seconds)
    return not t.is_alive() and not failed
