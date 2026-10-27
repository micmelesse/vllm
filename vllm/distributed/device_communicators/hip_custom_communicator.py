# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: MIT Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""HipCustomCommunicator: vLLM's custom all-reduce (`custom_all_reduce.py`'s
CustomAllreduce) extended to every collective Kimi-K3 needs -- the all-reduce and the
all-reduces fused with the ops after them -- over our HIP kernels in `_rocm_C`
(`csrc/rocm/hip_custom_communicator.cu`).

The caller names an op and Python looks its kernels up in the tuned table
(`TunedConfigs.nearest`: the inputs tuned at the communicator's `quantization`, INT4
where it pays): one kernel or, where the op is faster as two kernels of ours, our
all-reduce and then the op's norm on this rank's rows (`rms_norm`, `add_rms_norm`,
`add_attn_res_rms_norm`, each a public op). The keyword arguments an op takes after its
own: `config`, its op's config class, the one way to force one, for the tuner and
the tests. C++ takes every field of every call, checks the config, and returns the Error
it refused it for, or none: it raises nothing, holds no table and picks nothing.

THE MEMORY is C++'s, in its CommsState (a handle this class holds), each buffer its own
allocation: the exchange's regions (signals, staging, partials, scales, the pushed
one-shot's inbox), which every rank opens at open over legacy HIP IPC after this class
exchanges the handles over the CPU group; the regions' device arrays, the pushed
one-shot's epochs, the input tables; the status the kernels write before a wait that
gave up traps. Every exact pull
reads every rank's input IN PLACE through an input table (aiter's gfx950 pattern): an
eager call's copy, which C++ makes into its staging on the call's stream, or a
captured call's tensors, each launch reserving its slot in the state and every rank's
registered when `capture()` exits (vLLM's register_graph_buffers). An INT4 call's kernel
quantizes its input into the staging itself.

Nothing frees the buffers for us: `close()` has to run."""

import atexit
import functools
import json
import os
import signal
import socket
import threading
import warnings
import weakref
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
from datetime import timedelta
from enum import IntEnum
from pathlib import Path
from types import FrameType
from typing import Any, ClassVar, Literal, cast, final, get_args

import torch
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)


# ---- THE BUILD'S FACTS, as the communicator and its callers name them: the ops, the
# Errors C++ answers with (number for number), and whether a device runs it
# (`supported`). What is compiled is the tuned table's configs (builds.py). ----

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
# WHICH BODY OF AN OP'S ONE KERNEL a config runs, each op's by the names its build
# gives them (C++'s `algorithms`, its op file's enum): a one-shot or a two-shot,
# pulled or pushed. A norm on this rank's rows has one, which the table names "rows".
AllReduceAlgorithm = Literal[
    "one_shot_pull",
    "one_shot_push",
    "two_shot_pull",
    "two_shot_push",
    "one_shot_push_staged",
]
AllReduceRmsNormAlgorithm = Literal["one_shot_pull", "two_shot_pull", "two_shot_push"]
AllReduceAddRmsNormAlgorithm = Literal[
    "one_shot_pull", "two_shot_pull", "two_shot_push"
]
AllReduceAddAttnResRmsNormAlgorithm = Literal[
    "one_shot_pull", "two_shot_pull", "two_shot_push"
]
# THE ALGORITHMS THAT PUSH (their peers store into this rank's buffers): no input of
# theirs is read in place.
_PUSHED = ("one_shot_push", "two_shot_push", "one_shot_push_staged")
# Any op's, by name: what a lookup across ops (`TunedConfigs.kernel_config`) asks.
Algorithm = Literal[
    "one_shot_pull",
    "one_shot_push",
    "two_shot_pull",
    "two_shot_push",
    "one_shot_push_staged",
    "rows",
]


class Error(IntEnum):
    """EVERY WAY THE COMMUNICATOR FAILS. C++'s `hip_custom_communicator::Error` first,
    number for number (a test holds them equal): C++ returns its number, never raises;
    Python's own after them, from 100."""

    not_contiguous = 0
    dtype_not_built = 1
    world_not_built = 2
    row_not_packs = 3
    partials_too_small = 4
    grid_not_resident = 5
    staging_too_small = 6
    device_not_built = 7
    weight_not_built = 9
    no_such_kernel = 10
    not_built = 12
    launch_out_of_range = 13
    field_not_positive = 14
    field_not_this_templates = 15
    inner_stride_not_one = 17
    row_stride_not_packs = 18
    no_such_quantization = 19
    quantization_not_supported = 20
    block_too_large = 21
    lds_too_large = 22
    scratch_too_large = 23
    unfenced_too_small = 24
    pack_out_of_range = 25
    hip_failed = 26
    not_on_device = 27
    shape_mismatch = 28
    dtype_mismatch = 29
    not_aligned = 30
    inputs_overlap_outputs = 31
    field_out_of_range = 32
    view_past_region = 33
    tables_full = 34
    ranks_disagree = 35
    not_open = 36

    # PYTHON'S OWN, after C++'s. Why it is disabled:
    no_problem_size = 100  # no server config to size the exchange from
    no_tuned_table = 101  # the shipped tables hold none for this arch
    groups_disagree = 102  # the CPU and device groups differ in size or in this rank
    # A call refused before C++ is asked:
    disabled = 103  # the communicator runs no call: disabled, abandoned or closed
    broken = 104  # a fault or a failed capture left the peers' state half-done
    not_two_d = 105  # an op takes [rows, cols]
    forced_incomplete = 106  # a forced call gives every config field or none
    # The lifecycle out of order:
    capture_not_open = 107  # capture() on a communicator that is not open
    close_while_capturing = 108  # close() inside capture()
    # Opening failed:
    wrong_device = 109  # the current device is not the communicator's
    peer_not_accessible = 111  # this device cannot reach a peer's
    capture_raised = 119  # a capture raised part way


class CommunicatorError(RuntimeError):
    """A FAILURE OF THE COMMUNICATOR: which one (`error`), and what it was."""

    def __init__(self, error: Error, detail: str) -> None:
        super().__init__(f"{error.name}: {detail}")
        self.error = error


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


# C++'s `DType` names, as torch's dtypes.
_DTYPES: Mapping[str, torch.dtype] = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}


@dataclass(frozen=True)
class Build:
    """ONE BUILD OF AN OP'S KERNEL, what dispatch instantiates (C++'s BuildConfig): its
    algorithm (a norm on rows' "rows"), its block and the block's waves a EU, its tile
    (tile_k 1 where the kernel has no K), what it moves and its pack. A config is a
    build and a launch (its grid, quantize_from_bytes)."""

    algorithm: str
    block_x: int
    block_y: int
    block_z: int
    waves_per_eu: int
    tile_m: int
    tile_n: int
    tile_k: int
    quantization: Quantization
    pack_bytes: int

    @classmethod
    def of(cls, config: "Config") -> "Build":
        """`config`'s build: its fields less the launch."""
        return cls(
            algorithm=config.algorithm,
            block_x=config.block_x,
            block_y=config.block_y,
            block_z=config.block_z,
            waves_per_eu=config.waves_per_eu,
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            tile_k=config.tile_k,
            quantization=config.quantization,
            pack_bytes=config.pack_bytes,
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


# WHAT THE KERNELS ARE COMPILED AT (builds.py's DTYPES and WEIGHT_DTYPE, the host's
# kDtypes and kPackBytes): an input in a 16-bit dtype, a norm's weight in its input's
# dtype or fp32, every config's pack at most the widest load's 16 bytes.
_INPUT_DTYPES = (torch.float16, torch.bfloat16)
_WEIGHT_DTYPES = (torch.float32,)
_PACK_BYTES = 16


def should(
    op: Op,
    input: torch.Tensor,
    capacity: Capacity,
    weight: torch.Tensor | None = None,
) -> bool:
    """Whether a PLAIN call of `op` (nothing forced) on `input`, and a norm's `weight`,
    runs on an open communicator holding `capacity`: [rows, cols] in a dtype the
    kernels are compiled at, a norm's weight in its dtype or fp32, whole packs a row,
    and inside what its exchange holds, either the box it was sized for (rows and row
    width) or the op's most bytes. Python alone, no C++ call, as vLLM's
    `should_custom_ar`; C++ refuses no call this admits."""
    if (
        input.dtype not in _INPUT_DTYPES
        or not input.is_contiguous()
        or input.numel() == 0
        or input.dim() != 2
    ):
        return False
    if weight is not None and weight.dtype not in (input.dtype, *_WEIGHT_DTYPES):
        return False
    rows, row = input.shape
    in_box = rows <= capacity.max_rows and row <= capacity.max_cols
    return row * input.element_size() % _PACK_BYTES == 0 and (
        in_box or input.numel() * input.element_size() <= capacity.max_bytes[op]
    )


# THE TUNED TABLES (`hip_custom_communicator.json`, beside this file, written by gd's
# tune: gd's `TunedTables`), a table an arch, each tuned on its own: its `inputs` (the
# input points tuned: the op, qualified `hip_custom_communicator.<op>`, the
# communicator's mode, `exact` or `quantized`, the world, the width, the tokens and the
# op's own arguments, which the lookup does not read), its `configs` (each its op and
# its fields: what the build compiles there) and its `picks` (an input's index, and its
# configs' in the order a plain call launches them: the op's one kernel, or our
# all-reduce then the op's norm on rows). No format or version: it changes with this
# file, in one commit. The arch's table is parsed and checked at open
# (`parse_tuned_configs`); a plain call launches its nearest input's pick
# (`TunedConfigs.nearest`). Every tuned number is here, none in C++. A communicator
# opened with another file's path runs that instead (the tuner's candidate).
_TABLE = Path(__file__).with_name("hip_custom_communicator.json")
_TABLE_KEYS = ("inputs", "configs", "picks")
# AN OP AS THE TABLE NAMES IT, and the communicator's quantization as the table's input
# names it (the mode the communicator was opened in).
_QUALIFIED = "hip_custom_communicator."
_MODES: Mapping[Quantization, str] = {"exact": "exact", "int4": "quantized"}
# A TUNED INPUT'S DTYPE's element bytes, as the table names it.
_DTYPE_BYTES: Mapping[str, int] = {"bf16": 2, "f16": 2, "fp16": 2, "f32": 4}
# A CONFIG POINT'S FIELDS, in their order (the table's): the kernel, its algorithm,
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


# EACH OP'S CONFIG, ONE CLASS AN OP, a field each (in `_CONFIG_KEYS`' order): its
# algorithm (its own), its grid, its block and the block's waves a EU, its
# tile, what it moves, its pack, and the call's bytes from which an INT4 config
# quantizes (`quantize_from_bytes`: a launch's, the tune's pick; below them its INT4
# build passes the call exact; 0, it always quantizes; an exact config's None). Its
# kernel, its algorithms and what it does not choose (a norm on rows' one algorithm, a
# kernel without K's tile_k) are the class's: the table and C++ name every field of
# every op, so a config reads them alike, but a call cannot force them. Every op's C++
# call is handed its own fields by name, a line each.
@dataclass(frozen=True)
class AllReduceConfig:
    """THE ALL-REDUCE'S CONFIG."""

    kernel: ClassVar[str] = "all_reduce"
    algorithms: ClassVar[tuple[str, ...]] = get_args(AllReduceAlgorithm)
    tile_k: ClassVar[int] = 1  # no K: the table's and C++'s identity
    algorithm: AllReduceAlgorithm
    grid_x: int
    grid_y: int
    grid_z: int
    block_x: int
    block_y: int
    block_z: int
    waves_per_eu: int
    tile_m: int
    tile_n: int
    quantization: Quantization
    pack_bytes: int
    # None: an exact config's (it never quantizes), or an INT4 one's left out of a
    # forcing (C++'s 0, it always quantizes).
    quantize_from_bytes: int | None = None


@dataclass(frozen=True)
class AllReduceRmsNormConfig:
    """THE ALL-REDUCE THEN RMSNORM'S CONFIG, one kernel."""

    kernel: ClassVar[str] = "all_reduce_rms_norm"
    algorithms: ClassVar[tuple[str, ...]] = get_args(AllReduceRmsNormAlgorithm)
    tile_k: ClassVar[int] = 1  # no K: the table's and C++'s identity
    algorithm: AllReduceRmsNormAlgorithm
    grid_x: int
    grid_y: int
    grid_z: int
    block_x: int
    block_y: int
    block_z: int
    waves_per_eu: int
    tile_m: int
    tile_n: int
    quantization: Quantization
    pack_bytes: int
    # None: an exact config's (it never quantizes), or an INT4 one's left out of a
    # forcing (C++'s 0, it always quantizes).
    quantize_from_bytes: int | None = None


@dataclass(frozen=True)
class AllReduceAddRmsNormConfig:
    """THE ALL-REDUCE THEN ADD-RMSNORM'S CONFIG, one kernel."""

    kernel: ClassVar[str] = "all_reduce_add_rms_norm"
    algorithms: ClassVar[tuple[str, ...]] = get_args(AllReduceAddRmsNormAlgorithm)
    tile_k: ClassVar[int] = 1  # no K: the table's and C++'s identity
    algorithm: AllReduceAddRmsNormAlgorithm
    grid_x: int
    grid_y: int
    grid_z: int
    block_x: int
    block_y: int
    block_z: int
    waves_per_eu: int
    tile_m: int
    tile_n: int
    quantization: Quantization
    pack_bytes: int
    # None: an exact config's (it never quantizes), or an INT4 one's left out of a
    # forcing (C++'s 0, it always quantizes).
    quantize_from_bytes: int | None = None


@dataclass(frozen=True)
class AllReduceAddAttnResRmsNormConfig:
    """THE ALL-REDUCE THEN ATTNRES AND ITS RMSNORM'S CONFIG, one kernel: TILE_K
    sources a step."""

    kernel: ClassVar[str] = "all_reduce_add_attn_res_rms_norm"
    algorithms: ClassVar[tuple[str, ...]] = get_args(
        AllReduceAddAttnResRmsNormAlgorithm
    )
    algorithm: AllReduceAddAttnResRmsNormAlgorithm
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


@dataclass(frozen=True)
class RmsNormConfig:
    """THE RMSNORM ON THIS RANK'S ROWS' CONFIG."""

    kernel: ClassVar[str] = "rms_norm"
    algorithms: ClassVar[tuple[str, ...]] = ("rows",)
    algorithm: ClassVar[str] = "rows"  # its one
    tile_k: ClassVar[int] = 1  # no K: the table's and C++'s identity
    grid_x: int
    grid_y: int
    grid_z: int
    block_x: int
    block_y: int
    block_z: int
    waves_per_eu: int
    tile_m: int
    tile_n: int
    quantization: Quantization
    pack_bytes: int
    # None: an exact config's (it never quantizes), or an INT4 one's left out of a
    # forcing (C++'s 0, it always quantizes).
    quantize_from_bytes: int | None = None


@dataclass(frozen=True)
class AddRmsNormConfig:
    """THE ADD-RMSNORM ON THIS RANK'S ROWS' CONFIG."""

    kernel: ClassVar[str] = "add_rms_norm"
    algorithms: ClassVar[tuple[str, ...]] = ("rows",)
    algorithm: ClassVar[str] = "rows"  # its one
    tile_k: ClassVar[int] = 1  # no K: the table's and C++'s identity
    grid_x: int
    grid_y: int
    grid_z: int
    block_x: int
    block_y: int
    block_z: int
    waves_per_eu: int
    tile_m: int
    tile_n: int
    quantization: Quantization
    pack_bytes: int
    # None: an exact config's (it never quantizes), or an INT4 one's left out of a
    # forcing (C++'s 0, it always quantizes).
    quantize_from_bytes: int | None = None


@dataclass(frozen=True)
class AddAttnResRmsNormConfig:
    """ATTNRES AND ITS RMSNORM ON THIS RANK'S ROWS' CONFIG: TILE_K sources a step."""

    kernel: ClassVar[str] = "add_attn_res_rms_norm"
    algorithms: ClassVar[tuple[str, ...]] = ("rows",)
    algorithm: ClassVar[str] = "rows"  # its one
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


Config = (
    AllReduceConfig
    | AllReduceRmsNormConfig
    | AllReduceAddRmsNormConfig
    | AllReduceAddAttnResRmsNormConfig
    | RmsNormConfig
    | AddRmsNormConfig
    | AddAttnResRmsNormConfig
)
# EACH KERNEL'S CONFIG CLASS, by the kernel's name (the table's `kernel`).
_CONFIGS: Mapping[str, type[Config]] = {
    c.kernel: c
    for c in (
        AllReduceConfig,
        AllReduceRmsNormConfig,
        AllReduceAddRmsNormConfig,
        AllReduceAddAttnResRmsNormConfig,
        RmsNormConfig,
        AddRmsNormConfig,
        AddAttnResRmsNormConfig,
    )
}


# A FUSED OP AS TWO KERNELS OF OURS, its algorithm "split": our all-reduce into this
# rank's sum, then the op on this rank's rows (`rows`, its kernel on rows). The tune
# picks it where it is faster than the op's one kernel.
@dataclass(frozen=True)
class AllReduceRmsNormSplitConfig:
    algorithm: ClassVar[str] = "split"
    all_reduce: AllReduceConfig
    rows: RmsNormConfig


@dataclass(frozen=True)
class AllReduceAddRmsNormSplitConfig:
    algorithm: ClassVar[str] = "split"
    all_reduce: AllReduceConfig
    rows: AddRmsNormConfig


@dataclass(frozen=True)
class AllReduceAddAttnResRmsNormSplitConfig:
    algorithm: ClassVar[str] = "split"
    all_reduce: AllReduceConfig
    rows: AddAttnResRmsNormConfig


SplitConfig = (
    AllReduceRmsNormSplitConfig
    | AllReduceAddRmsNormSplitConfig
    | AllReduceAddAttnResRmsNormSplitConfig
)


def config_of_record(c: Mapping[str, Any]) -> Config:
    """A table's config record (its op, then a field a key, `_CONFIG_KEYS`) as its
    kernel's config: its algorithm one of the kernel's, and a field the class holds
    (a norm on rows' algorithm, tile_k without K) the class's."""
    cls = _CONFIGS[c["kernel"]]
    if c["algorithm"] not in cls.algorithms:
        raise ValueError(f"{c['kernel']} has no algorithm {c['algorithm']}")
    own = {f.name for f in fields(cls)}
    for k in ("algorithm", "tile_k"):
        if k not in own and c[k] != getattr(cls, k):
            raise ValueError(f"{c['kernel']}'s {k} is {getattr(cls, k)}, not {c[k]}")
    return cls(**{k: c[k] for k in _CONFIG_KEYS if k in own})


def record_refused(op: Op, c: Mapping[str, Any]) -> Error | None:
    """WHAT A CONFIG AS THE TABLE HOLDS IT (`c`, `_CONFIG_KEYS`) FORCED ON `op` IS
    REFUSED FOR, or None (a caller holding configs flat, gd's): another op's kernel or
    an algorithm its kernel lacks, no_such_kernel; tile_k on a kernel without K,
    field_not_this_templates."""
    cls = _CONFIGS.get(c["kernel"])
    if cls is None or c["kernel"] != op or c["algorithm"] not in cls.algorithms:
        return Error.no_such_kernel
    if "tile_k" not in {f.name for f in fields(cls)} and c["tile_k"] != cls.tile_k:
        return Error.field_not_this_templates
    return None


@dataclass(frozen=True)
class TunedInput:
    """ONE INPUT THE TABLE WAS TUNED AT (its `inputs` record, the fields the lookup
    reads): the op, the communicator's mode (`exact` or `quantized`), the world, the
    width, the tokens and the dtype."""

    op: Op
    mode: str
    world: int
    hidden: int
    tokens: int
    dtype: str


@dataclass(frozen=True)
class TunedConfigs:
    """THE ARCH'S TUNED TABLE, parsed and checked (`parse_tuned_configs`): each tuned
    input and the config a plain call runs there (the op's one kernel, or its split:
    our all-reduce then the op on rows), and every config the file lists (its picks'
    and a tune's candidates)."""

    picks: dict[TunedInput, Config | SplitConfig]
    configs: tuple[Config, ...]

    def nearest(
        self, op: Op, quantization: Quantization, world: int, rows: int, cols: int
    ) -> tuple[TunedInput, Config | SplitConfig]:
        """WHAT A PLAIN CALL OF `op` ON `rows` x `cols` RUNS, and the input it was
        tuned at: the nearest input of `op` tuned in the communicator's mode at `world`.
        Among the inputs whose every config's pack fits the row (whole packs at the
        input's dtype; a config the row is not whole packs of is refused,
        row_not_packs), where any does; then the nearest width, then the nearest size
        (rows x width), the fewer tokens on a tie."""
        tuned = [
            (i, c)
            for i, c in self.picks.items()
            if i.op == op and i.mode == _MODES[quantization] and i.world == world
        ]
        fitting = [
            (i, c)
            for i, c in tuned
            if all(
                cols * _DTYPE_BYTES[i.dtype] % x.pack_bytes == 0
                for x in (
                    (c.all_reduce, c.rows) if isinstance(c, SplitConfig) else (c,)
                )
            )
        ]
        tuned = fitting or tuned
        width = min((i.hidden for i, _ in tuned), key=lambda h: abs(h - cols))
        return min(
            ((i, c) for i, c in tuned if i.hidden == width),
            key=lambda ic: (abs(ic[0].tokens * width - rows * cols), ic[0].tokens),
        )

    def kernel_config(
        self,
        kernel: str,
        algorithm: Algorithm,
        rows: int,
        cols: int,
        quantization: Quantization,
        world: int,
    ) -> Config | None:
        """KERNEL `kernel`'S OWN CONFIG AT `algorithm` FOR `rows` x `cols` MOVING
        `quantization` (a forced launch at the tuned config: gp's kernel lines, the
        tests): the config of it moving that which its nearest input at `world`
        launches, of any op, else the first such config of the table's; moving INT4
        where it has no INT4 config, its exact config moved INT4 (a build it lists).
        None where it has neither."""
        assert kernel in _CONFIGS, f"no op's kernel is {kernel}"
        # ITS BUILDS: the table's configs of it (builds.py compiles exactly those).
        builds = {Build.of(c) for c in self.configs if c.kernel == kernel}

        def takes(c: Config, moving: Quantization) -> bool:
            return (
                c.kernel == kernel
                and c.algorithm == algorithm
                and c.quantization == moving
                and replace(Build.of(c), quantization=quantization) in builds
            )

        for moving in (
            (quantization, "exact") if quantization == "int4" else (quantization,)
        ):
            found = []
            for i, pick in self.picks.items():
                if i.world == world:
                    launches = (
                        (pick.all_reduce, pick.rows)
                        if isinstance(pick, SplitConfig)
                        else (pick,)
                    )
                    c = next((c for c in launches if takes(c, moving)), None)
                    if c is not None:
                        found.append((i, c))
            if found:
                width = min((i.hidden for i, _ in found), key=lambda h: abs(h - cols))
                _, config = min(
                    ((i, c) for i, c in found if i.hidden == width),
                    key=lambda ic: (
                        abs(ic[0].tokens * ic[0].hidden - rows * cols),
                        ic[0].tokens,
                    ),
                )
            else:
                first = next((c for c in self.configs if takes(c, moving)), None)
                if first is None:
                    continue
                config = first
            if config is not None:
                return replace(
                    config,
                    quantization=quantization,
                    quantize_from_bytes=None if quantization == "exact" else 0,
                )
        return None


def parse_tuned_configs(path: Path, arch: str) -> TunedConfigs | None:
    """THE TUNED TABLE AT `path` ON `arch` (the file, a table an arch: gd's
    `TunedTables`), parsed and checked, or None where the file holds none for it. A
    table that is not sound raises, naming the entry: its keys `_TABLE_KEYS`; every
    config its op's (qualified), every field `_CONFIG_KEYS` names in that order, read as
    its op's config (`config_of_record`; builds.py compiles every config, and C++
    refuses a config it did not, not_built); every input one of `TUNED_OPS`, in a
    mode, at a world, with a width and tokens, tuned once; every pick one input's, its
    configs one kernel of the input's op, or our all-reduce then the op's norm on rows,
    an exact input's moving
    nothing quantized, a quantized one's all-reduce INT4; every tuned op an input in
    each mode at every world the table tunes (the lookup reads one there)."""
    tables = json.loads(path.read_text())
    if arch not in tables:
        return None
    raw = tables[arch]
    if list(raw) != list(_TABLE_KEYS):
        raise ValueError(
            f"the tuned table's keys are {list(raw)}, not {list(_TABLE_KEYS)}"
        )
    configs: list[Config] = []
    for i, c in enumerate(raw["configs"]):
        where = f"the tuned table: configs[{i}]"
        if list(c) != ["op", *_CONFIG_KEYS]:
            raise ValueError(f"{where} has keys {list(c)}, not {['op', *_CONFIG_KEYS]}")
        if c["kernel"] not in _CONFIGS or c["op"] != _QUALIFIED + c["kernel"]:
            raise ValueError(f"{where}: {c['kernel']} is no kernel of {c['op']}")
        q = c["quantize_from_bytes"]
        if (c["quantization"] == "exact") != (q is None) or (
            q is not None and (not isinstance(q, int) or q < 0)
        ):
            raise ValueError(
                f"{where}: quantize_from_bytes is null for an exact config, an INT4 "
                "one's count of bytes"
            )
        try:
            config = config_of_record(c)
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from None
        configs.append(config)
    picks: dict[TunedInput, Config | SplitConfig] = {}
    for i, k in enumerate(raw["picks"]):
        where = f"the tuned table: picks[{i}]"
        at = k.get("configs")
        if not (
            isinstance(k.get("input"), int)
            and 0 <= k["input"] < len(raw["inputs"])
            and isinstance(at, list)
            and at
            and all(isinstance(j, int) and 0 <= j < len(configs) for j in at)
        ):
            raise ValueError(f"{where}: {k} names no one input's configs")
        e = raw["inputs"][k["input"]]
        name = str(e.get("op", "")).removeprefix(_QUALIFIED)
        if not str(e.get("op", "")).startswith(_QUALIFIED) or name not in TUNED_OPS:
            raise ValueError(
                f"{where}: {e.get('op')} is no tuned op of the communicator's"
            )
        if e.get("mode") not in _MODES.values() or not isinstance(e.get("world"), int):
            raise ValueError(f"{where}: no such mode, or no world: {e}")
        if e.get("hidden", 0) < 1 or e.get("tokens", 0) < 1:
            raise ValueError(f"{where}: no width or no tokens")
        tuned = TunedInput(
            op=cast(Op, name),
            mode=e["mode"],
            world=e["world"],
            hidden=e["hidden"],
            tokens=e["tokens"],
            dtype=e["dtype"],
        )
        if tuned in picks:
            raise ValueError(f"{where}: {tuned} is tuned twice")
        launches = tuple(configs[j] for j in at)
        ops = [c.kernel for c in launches]
        if ops not in ([name], ["all_reduce", _ROWS.get(name)]):
            raise ValueError(
                f"{where}: its configs are {ops}'s: one of {name}'s, or our all-reduce "
                f"then {_ROWS.get(name)}"
            )
        if tuned.mode == "exact" and {c.quantization for c in launches} != {"exact"}:
            raise ValueError(f"{where}: an exact input moves INT4")
        # THE QUANTIZED MODE RUNS THE ALL-REDUCE'S INT4 KERNELS, every pick (each
        # passing a call exact itself below its quantize_from_bytes): no exact
        # all-reduce kernel in quantized mode, the plain op's or a split's first.
        if tuned.mode == "quantized" and any(
            c.kernel == "all_reduce" and c.quantization != "int4" for c in launches
        ):
            raise ValueError(
                f"{where}: a quantized input picks the exact all-reduce kernel"
            )
        # ONE KERNEL, or the op's split: our all-reduce, then its kernel on rows (the
        # ops checked above).
        if len(launches) == 1:
            picks[tuned] = launches[0]
            continue
        first, second = launches
        assert isinstance(first, AllReduceConfig), where
        if isinstance(second, RmsNormConfig):
            picks[tuned] = AllReduceRmsNormSplitConfig(first, second)
        elif isinstance(second, AddRmsNormConfig):
            picks[tuned] = AllReduceAddRmsNormSplitConfig(first, second)
        else:
            assert isinstance(second, AddAttnResRmsNormConfig), where
            picks[tuned] = AllReduceAddAttnResRmsNormSplitConfig(first, second)
    for name in TUNED_OPS:
        for world in sorted({i.world for i in picks}):
            for mode in _MODES.values():
                if not any(
                    i.op == name and i.world == world and i.mode == mode for i in picks
                ):
                    raise ValueError(
                        f"the tuned table: {name}: no pick at world {world}, {mode}"
                    )
    return TunedConfigs(picks, tuple(configs))


def configured_quantization() -> Quantization:
    """What `VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION` asks (exact by default:
    INT4 is lossy, so opt-in)."""
    import vllm.envs as envs

    asked = envs.VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION
    return "int4" if asked == "INT4" else "exact"


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


# A COMMUNICATOR'S LIFE: `__init__` -> open | disabled (its error); open <-> capturing;
# capturing -> broken when a capture raises, and open or capturing -> broken when the
# capture's registration fails (its error and detail); open, capturing or broken ->
# abandoned when the process exits (`abandon_open`); any -> closed. Every call is
# refused but in open and capturing; a broken one raises naming why.
Status = Literal["disabled", "open", "capturing", "broken", "abandoned", "closed"]


@dataclass
class Lifecycle:
    """THE COMMUNICATOR'S LIFE: its status, and why where it is disabled or broken; the
    lock every move of the status takes; one teardown at a time (close, and the exit
    hook's abandon)."""

    status: Status = "closed"
    error: Error | None = None  # why: a disabled one's, a broken one's
    detail: str = ""  # what left a broken one's peers half-done
    lock: threading.Lock = field(default_factory=threading.Lock)
    teardown: threading.Lock = field(default_factory=threading.Lock)

    def holds(self) -> bool:
        """Whether it still holds peer mappings and buffers only `close()` releases:
        what `__del__` warns about, read without a global lookup (collection can run at
        interpreter shutdown, when globals are None)."""
        return self.status in ("open", "capturing", "broken")


@dataclass(frozen=True)
class CommsState:
    """THE STATE BETWEEN PYTHON AND C++: C++'s CommsState (`handle`: its memory, its
    peers, its captured inputs), and what Python reads of it once at open: the device's
    arch, what a communicator of its box holds (`should`'s capacity), and the arch's
    tuned table."""

    handle: int
    arch: str
    capacity: Capacity
    tuned_configs: TunedConfigs


@dataclass(frozen=True, eq=False)
class Arguments:
    """WHAT THE COMMUNICATOR WAS CONSTRUCTED WITH, every None filled (`__init__`): the
    groups, the device, how long a kernel waits on a peer, the largest call served (rows
    of at most `max_cols`), what the plain calls move, and the tuned tables' file. What
    follows from them is a method each."""

    cpu_group: ProcessGroup
    device_group: ProcessGroup
    device: torch.device
    timeout_seconds: float
    max_rows: int
    max_cols: int
    quantization: Quantization
    tuned_path: Path

    @functools.cached_property
    def world(self) -> int:
        return self.device_group.size()

    @functools.cached_property
    def rank(self) -> int:
        return self.device_group.rank()


# EVERY COMMUNICATOR OPENED IN THIS PROCESS AND NOT YET COLLECTED, for the exit hook to
# move to Abandoned.
_open_communicators: "weakref.WeakSet[HipCustomCommunicator]" = weakref.WeakSet()


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

    # ---- ITS FIELDS. A disabled one has its arguments where it got that far, its
    # lifecycle, and no state. ----

    # What it was constructed with, resolved; what follows from them is its methods.
    arguments: Arguments
    lifecycle: Lifecycle  # its status, its locks
    state: CommsState | None = None  # the state between Python and C++, once open

    # ---- MAKING IT: the constructor, and the open it runs. ----

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
        tuned_path: Path = _TABLE,
    ) -> None:
        """Keep the groups and the device, then `_open`, which checks the backend's
        availability. Unavailability is not an error: the caller checks `.disabled`.
        `timeout_seconds`: how long a kernel waits on a peer before it reports a fault;
        None is the device group's own (`collective_timeout_seconds`). `max_rows` and
        `max_cols`: the largest call it serves, which its exchange is sized for; None is
        the server's (`problem_size`). `quantization`: what the plain calls move across
        the links, from each op's crossover up (their table: the quantized one where
        INT4); None is what the environment asks (`configured_quantization`).
        `tuned_path`: the tuned tables' file its plain calls run from (the arch's table,
        `parse_tuned_configs`): the shipped one (`hip_custom_communicator.json`), or a
        tuner's candidate."""
        self._own_fields()
        device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        if max_rows is None or max_cols is None:
            served = problem_size()
            if served is None:
                # NOTHING TO SIZE THE EXCHANGE FROM (max_num_batched_tokens, the
                # hidden size; or max_rows and max_cols): disabled, so vLLM takes its
                # own path.
                self._disable(Error.no_problem_size)
                return
            served_rows, served_cols = served
            max_rows = served_rows if max_rows is None else max_rows
            max_cols = served_cols if max_cols is None else max_cols
        self.arguments = Arguments(
            cpu_group=cpu_group,
            device_group=device_group,
            device=device,
            timeout_seconds=(
                collective_timeout_seconds(device_group, device)
                if timeout_seconds is None
                else timeout_seconds
            ),
            max_rows=max_rows,
            max_cols=max_cols,
            quantization=(
                configured_quantization() if quantization is None else quantization
            ),
            tuned_path=tuned_path,
        )
        error = self._open()
        if error is not None:
            self._disable(error)
            return
        self.lifecycle.status = "open"
        _open_communicators.add(self)

    @final
    def _own_fields(
        self, status: Status = "closed", error: Error | None = None, detail: str = ""
    ) -> None:
        """WHAT NOTHING IS GIVEN TO MAKE: its lifecycle at `status`, and no state yet.
        `__init__` makes them first; a test's bare instance (no process group) makes
        them here too, at its status."""
        self.lifecycle = Lifecycle(status=status, error=error, detail=detail)
        self.state = None

    @final
    def _disable(self, error: Error) -> None:
        """Disabled, for `error`: it runs nothing, and the caller takes its own path."""
        self.lifecycle.status, self.lifecycle.error = "disabled", error
        logger.info("%s disabled: %s", type(self).__name__, error.name)

    def _open(self) -> Error | None:
        """OPEN, or the Error it cannot open for (it is then disabled): C++'s state made
        (the device supported, its buffers allocated), the tuned table read for its
        arch, and every peer's regions opened, after the exchange over the CPU group (a
        collective, so every rank must reach it).

        Eager, and before any capture: doing this inside a cudagraph capture is not
        recoverable. An open that raises frees what it made, alone, and raises."""
        arguments = self.arguments
        # THE GROUPS: one world, this rank the same in both.
        if (
            arguments.cpu_group.size() != arguments.device_group.size()
            or arguments.cpu_group.rank() != arguments.device_group.rank()
        ):
            return Error.groups_disagree
        if torch.cuda.current_device() != arguments.device.index:
            raise CommunicatorError(
                Error.wrong_device,
                f"the current device is {torch.cuda.current_device()}, not the "
                f"communicator's {arguments.device.index}",
            )
        ops = torch.ops._rocm_C
        error, handle, ipc_handle, arch, capacity = ops.hip_custom_communicator_open(
            rank=arguments.rank,
            world=arguments.world,
            device=arguments.device.index,
            max_rows=arguments.max_rows,
            max_cols=arguments.max_cols,
            timeout_seconds=arguments.timeout_seconds,
        )
        if error is not None:
            return Error(error)
        tuned_configs = parse_tuned_configs(arguments.tuned_path, arch)
        if tuned_configs is None:
            ops.hip_custom_communicator_free(state=handle)
            return Error.no_tuned_table
        rows, cols, partials, staging, *most = capacity
        self.state = CommsState(
            handle=handle,
            arch=arch,
            capacity=Capacity(
                rows, cols, partials, staging, dict(zip(get_args(Op), most))
            ),
            tuned_configs=tuned_configs,
        )
        try:
            # THE EXCHANGE, one collective over the CPU group: every rank's regions' IPC
            # handles, its host, its GPU and its pid.
            props = torch.cuda.get_device_properties(arguments.device)
            me = {
                "handle": ipc_handle,
                "host": socket.gethostname(),
                "gpu": str(getattr(props, "uuid", "")),
                "pid": os.getpid(),
            }
            everyone: list[Any] = [None] * arguments.world
            torch.distributed.all_gather_object(everyone, me, group=arguments.cpu_group)
            # EVERY PEER'S GPU, where this process sees it (by its UUID), reachable from
            # ours: a rank that cannot reach a peer's GPU is refused here, by name, not
            # at its first call.
            here = {
                str(getattr(torch.cuda.get_device_properties(d), "uuid", "")): d
                for d in range(torch.cuda.device_count())
            }
            for r, peer in enumerate(everyone):
                d = here.get(peer["gpu"])
                if r == arguments.rank or d is None or d == arguments.device.index:
                    continue
                if not torch.cuda.can_device_access_peer(arguments.device.index, d):
                    raise CommunicatorError(
                        Error.peer_not_accessible,
                        f"device {arguments.device.index} cannot access rank {r}'s "
                        f"device {d}",
                    )
            for r, peer in enumerate(everyone):
                if r == arguments.rank:
                    continue
                if peer["host"] != me["host"]:
                    logger.warning(
                        "HipCustomCommunicator open: rank %d is on another host", r
                    )
                elif peer["gpu"] and peer["gpu"] == me["gpu"]:
                    logger.warning(
                        "HipCustomCommunicator open: rank %d (pid %d) is on this "
                        "rank's GPU",
                        r,
                        peer["pid"],
                    )
            error = ops.hip_custom_communicator_connect(
                state=handle, handles=[b for peer in everyone for b in peer["handle"]]
            )
            if error is not None:
                raise CommunicatorError(Error(error), "opening the peers' regions")
        except BaseException:
            self._close_peers()
            self._free()
            raise
        _install_exit_hooks()
        logger.info(
            "HipCustomCommunicator ready: exchange sized for %d rows of %d "
            "(partials %.1f MiB, staging %.1f MiB), quantization=%s",
            rows,
            cols,
            partials / 2**20,
            staging / 2**20,
            arguments.quantization,
        )
        return None

    # ---- WHAT THE CALLER USES, in its order. Concrete: this class owns the order. ----
    #
    # Each `should_` asks the module's `should` of the op's own tensors: whether a
    # PLAIN call (nothing forced) runs, read from the build's record of the op, with no
    # C++ call -- as vLLM's `should_custom_ar` is. A caller asks it, then calls the op.
    # Each op takes its own tensors and values, then, keyword-only, what the tuner and
    # the tests force: `config`, the op's config class (a fused op's split too). None is
    # a plain call: the table's nearest input's config (`TunedConfigs.nearest`). The
    # norms on rows have no table: a call of one gives its config. A forced call asks no
    # `should_`: C++ answers it. An op raises a RuntimeError naming the Error it is
    # refused for; a call its `should_` admitted is never refused.

    @final
    @property
    def disabled(self) -> bool:
        """The flag every vLLM communicator has: no call may run. A BROKEN one
        raises instead: its peers' state is half-done (a capture that failed part
        way), so a quiet fallback would leave the next call on a rank to hang."""
        life = self.lifecycle
        if life.status == "broken":
            assert life.error is not None
            raise CommunicatorError(
                Error.broken,
                f"{life.error.name}: {life.detail}. Close it and build a new one on "
                f"every rank.",
            )
        return life.status not in ("open", "capturing")

    @final
    def should_allreduce(self, input: torch.Tensor) -> bool:
        """Whether `all_reduce(input)` runs here."""
        return (
            not self.disabled
            and self.state is not None
            and should("all_reduce", input, self.state.capacity)
        )

    @final
    def all_reduce(
        self, input: torch.Tensor, *, config: AllReduceConfig | None = None
    ) -> torch.Tensor:
        """The sum of every rank's `input`, at `config` (the tuner's, the tests'), else
        at the tuned table's pick for its rows and width."""
        # REFUSED BEFORE C++ IS ASKED: a communicator that runs nothing, an input that
        # is not [rows, cols].
        if self.disabled:
            raise CommunicatorError(
                Error.disabled, f"{type(self).__name__} runs no call"
            )
        assert self.state is not None, "an open communicator has its state"
        if input.dim() != 2:
            raise CommunicatorError(Error.not_two_d, f"not {tuple(input.shape)}")

        # A CAPTURE'S WARMUP: vLLM discards its outputs, so nothing launches.
        output = torch.empty_like(input)
        capturing = torch.cuda.is_current_stream_capturing()
        if self.lifecycle.status == "capturing" and not capturing:
            return output

        # THE CONFIG: the one given, else the tuned table's nearest input's.
        if config is None:
            rows, cols = input.shape
            _, picked = self.state.tuned_configs.nearest(
                "all_reduce",
                self.arguments.quantization,
                self.arguments.world,
                rows,
                cols,
            )
            assert isinstance(picked, AllReduceConfig), picked
            config = picked
        # WHAT IT MOVES IS THE COMMUNICATOR'S: an exact one moves no INT4, a quantized
        # one runs the INT4 kernel (which passes a call below its quantize_from_bytes
        # exact itself).
        if self.arguments.quantization == "exact" and config.quantization != "exact":
            raise AssertionError(
                f"an exact communicator given an INT4 config: {config}"
            )
        if self.arguments.quantization == "int4" and config.quantization != "int4":
            raise AssertionError(
                f"a quantized communicator given an exact config: {config}"
            )

        error = torch.ops._rocm_C.hip_custom_communicator_all_reduce(
            state=self.state.handle,
            input=input,
            output=output,
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
            quantization=config.quantization,
            pack_bytes=config.pack_bytes,
            quantize_from_bytes=config.quantize_from_bytes,
        )
        if error is not None:
            raise CommunicatorError(Error(error), f"{config} on {input.shape}")
        return output

    @final
    def should_allreduce_rms_norm(
        self, input: torch.Tensor, weight: torch.Tensor
    ) -> bool:
        """Whether `all_reduce_rms_norm(input, weight, ...)` runs here."""
        return (
            not self.disabled
            and self.state is not None
            and should("all_reduce_rms_norm", input, self.state.capacity, weight)
        )

    @final
    def all_reduce_rms_norm(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        *,
        config: AllReduceRmsNormConfig | AllReduceRmsNormSplitConfig | None = None,
    ) -> torch.Tensor:
        """`vllm.ir.ops.rms_norm(all_reduce(input), weight, eps)`, at the config given
        (the tuner's, the tests'), else at the table's pick for its rows and width: one
        kernel, or our all-reduce then the norm on its rows. Returns the normed sum."""
        # REFUSED BEFORE C++ IS ASKED: a communicator that runs nothing, an input that
        # is not [rows, cols].
        if self.disabled:
            raise CommunicatorError(
                Error.disabled, f"{type(self).__name__} runs no call"
            )
        assert self.state is not None, "an open communicator has its state"
        if input.dim() != 2:
            raise CommunicatorError(Error.not_two_d, f"not {tuple(input.shape)}")

        # A CAPTURE'S WARMUP: vLLM discards its outputs, so nothing launches.
        output = torch.empty_like(input)
        if (
            self.lifecycle.status == "capturing"
            and not torch.cuda.is_current_stream_capturing()
        ):
            return output

        # THE CONFIG: the one given, else the tuned table's nearest input's (one
        # kernel, or the split: our all-reduce, then the op on this rank's rows).
        if config is None:
            rows, cols = input.shape
            _, picked = self.state.tuned_configs.nearest(
                "all_reduce_rms_norm",
                self.arguments.quantization,
                self.arguments.world,
                rows,
                cols,
            )
            assert isinstance(
                picked, AllReduceRmsNormConfig | AllReduceRmsNormSplitConfig
            ), picked
            config = picked
        launches: list[Config] = (
            [config.all_reduce, config.rows]
            if isinstance(config, AllReduceRmsNormSplitConfig)
            else [config]
        )
        # WHAT IT MOVES IS THE COMMUNICATOR'S: an exact one moves no INT4; a quantized
        # one's all-reduce kernel (the split's first) is the INT4 one, which passes a
        # call below its quantize_from_bytes exact itself.
        for c in launches:
            if self.arguments.quantization == "exact" and c.quantization != "exact":
                raise AssertionError(f"an exact communicator given an INT4 config: {c}")
            if (
                self.arguments.quantization == "int4"
                and c.kernel == "all_reduce"
                and c.quantization != "int4"
            ):
                raise AssertionError(
                    f"a quantized communicator given an exact config: {c}"
                )

        match config:
            case AllReduceRmsNormConfig():
                # ONE KERNEL, the fused op.
                error = torch.ops._rocm_C.hip_custom_communicator_all_reduce_rms_norm(
                    state=self.state.handle,
                    input=input,
                    weight=weight,
                    output=output,
                    eps=eps,
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
                    quantization=config.quantization,
                    pack_bytes=config.pack_bytes,
                    quantize_from_bytes=config.quantize_from_bytes,
                )
                if error is not None:
                    raise CommunicatorError(Error(error), f"{config} on {input.shape}")
            case AllReduceRmsNormSplitConfig():
                # TWO KERNELS OF OURS: our all-reduce into this rank's sum, then the
                # norm on its rows. THE SUM IS THE OUTPUT ITSELF (ours, fresh): the norm
                # writes its rows over them in place (STANDARDS 3.5), no tensor of its
                # own.
                error = torch.ops._rocm_C.hip_custom_communicator_all_reduce(
                    state=self.state.handle,
                    input=input,
                    output=output,
                    algorithm=config.all_reduce.algorithm,
                    grid_x=config.all_reduce.grid_x,
                    grid_y=config.all_reduce.grid_y,
                    grid_z=config.all_reduce.grid_z,
                    block_x=config.all_reduce.block_x,
                    block_y=config.all_reduce.block_y,
                    block_z=config.all_reduce.block_z,
                    waves_per_eu=config.all_reduce.waves_per_eu,
                    tile_m=config.all_reduce.tile_m,
                    tile_n=config.all_reduce.tile_n,
                    quantization=config.all_reduce.quantization,
                    pack_bytes=config.all_reduce.pack_bytes,
                    quantize_from_bytes=config.all_reduce.quantize_from_bytes,
                )
                if error is not None:
                    raise CommunicatorError(
                        Error(error), f"{config.all_reduce} on {input.shape}"
                    )
                error = torch.ops._rocm_C.hip_custom_communicator_rms_norm(
                    state=self.state.handle,
                    input=output,
                    weight=weight,
                    output=output,
                    eps=eps,
                    grid_x=config.rows.grid_x,
                    grid_y=config.rows.grid_y,
                    grid_z=config.rows.grid_z,
                    block_x=config.rows.block_x,
                    block_y=config.rows.block_y,
                    block_z=config.rows.block_z,
                    waves_per_eu=config.rows.waves_per_eu,
                    tile_m=config.rows.tile_m,
                    tile_n=config.rows.tile_n,
                    quantization=config.rows.quantization,
                    pack_bytes=config.rows.pack_bytes,
                    quantize_from_bytes=config.rows.quantize_from_bytes,
                )
                if error is not None:
                    raise CommunicatorError(
                        Error(error), f"{config.rows} on {input.shape}"
                    )
        return output

    @final
    def should_allreduce_add_rms_norm(
        self, input: torch.Tensor, weight: torch.Tensor
    ) -> bool:
        """Whether `all_reduce_add_rms_norm(input, ..., weight, ...)` runs here."""
        return (
            not self.disabled
            and self.state is not None
            and should("all_reduce_add_rms_norm", input, self.state.capacity, weight)
        )

    @final
    def all_reduce_add_rms_norm(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        *,
        config: AllReduceAddRmsNormConfig
        | AllReduceAddRmsNormSplitConfig
        | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`vllm.ir.ops.fused_add_rms_norm(all_reduce(input), residual, weight, eps)`,
        at the config given (the tuner's, the tests'), else at the table's pick for its
        rows and width: one kernel, or our all-reduce then the add and norm on its rows.
        Returns the normed result and
        the sum plus residual."""
        # REFUSED BEFORE C++ IS ASKED: a communicator that runs nothing, an input that
        # is not [rows, cols].
        if self.disabled:
            raise CommunicatorError(
                Error.disabled, f"{type(self).__name__} runs no call"
            )
        assert self.state is not None, "an open communicator has its state"
        if input.dim() != 2:
            raise CommunicatorError(Error.not_two_d, f"not {tuple(input.shape)}")

        # A CAPTURE'S WARMUP: vLLM discards its outputs, so nothing launches.
        output, residual_output = torch.empty_like(input), torch.empty_like(input)
        if (
            self.lifecycle.status == "capturing"
            and not torch.cuda.is_current_stream_capturing()
        ):
            return output, residual_output

        # THE CONFIG: the one given, else the tuned table's nearest input's (one
        # kernel, or the split: our all-reduce, then the op on this rank's rows).
        if config is None:
            rows, cols = input.shape
            _, picked = self.state.tuned_configs.nearest(
                "all_reduce_add_rms_norm",
                self.arguments.quantization,
                self.arguments.world,
                rows,
                cols,
            )
            assert isinstance(
                picked, AllReduceAddRmsNormConfig | AllReduceAddRmsNormSplitConfig
            ), picked
            config = picked
        launches: list[Config] = (
            [config.all_reduce, config.rows]
            if isinstance(config, AllReduceAddRmsNormSplitConfig)
            else [config]
        )
        # WHAT IT MOVES IS THE COMMUNICATOR'S: an exact one moves no INT4; a quantized
        # one's all-reduce kernel (the split's first) is the INT4 one, which passes a
        # call below its quantize_from_bytes exact itself.
        for c in launches:
            if self.arguments.quantization == "exact" and c.quantization != "exact":
                raise AssertionError(f"an exact communicator given an INT4 config: {c}")
            if (
                self.arguments.quantization == "int4"
                and c.kernel == "all_reduce"
                and c.quantization != "int4"
            ):
                raise AssertionError(
                    f"a quantized communicator given an exact config: {c}"
                )

        match config:
            case AllReduceAddRmsNormConfig():
                # ONE KERNEL, the fused op.
                error = (
                    torch.ops._rocm_C.hip_custom_communicator_all_reduce_add_rms_norm(
                        state=self.state.handle,
                        input=input,
                        residual=residual,
                        weight=weight,
                        output=output,
                        residual_output=residual_output,
                        eps=eps,
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
                        quantization=config.quantization,
                        pack_bytes=config.pack_bytes,
                        quantize_from_bytes=config.quantize_from_bytes,
                    )
                )
                if error is not None:
                    raise CommunicatorError(Error(error), f"{config} on {input.shape}")
            case AllReduceAddRmsNormSplitConfig():
                # TWO KERNELS OF OURS: our all-reduce into this rank's sum, then the add
                # and norm on its rows. THE SUM IS THE OUTPUT ITSELF (ours, fresh): the
                # norm writes its rows over them in place (STANDARDS 3.5), no tensor of
                # its own.
                error = torch.ops._rocm_C.hip_custom_communicator_all_reduce(
                    state=self.state.handle,
                    input=input,
                    output=output,
                    algorithm=config.all_reduce.algorithm,
                    grid_x=config.all_reduce.grid_x,
                    grid_y=config.all_reduce.grid_y,
                    grid_z=config.all_reduce.grid_z,
                    block_x=config.all_reduce.block_x,
                    block_y=config.all_reduce.block_y,
                    block_z=config.all_reduce.block_z,
                    waves_per_eu=config.all_reduce.waves_per_eu,
                    tile_m=config.all_reduce.tile_m,
                    tile_n=config.all_reduce.tile_n,
                    quantization=config.all_reduce.quantization,
                    pack_bytes=config.all_reduce.pack_bytes,
                    quantize_from_bytes=config.all_reduce.quantize_from_bytes,
                )
                if error is not None:
                    raise CommunicatorError(
                        Error(error), f"{config.all_reduce} on {input.shape}"
                    )
                error = torch.ops._rocm_C.hip_custom_communicator_add_rms_norm(
                    state=self.state.handle,
                    input=output,
                    residual=residual,
                    weight=weight,
                    output=output,
                    residual_output=residual_output,
                    eps=eps,
                    grid_x=config.rows.grid_x,
                    grid_y=config.rows.grid_y,
                    grid_z=config.rows.grid_z,
                    block_x=config.rows.block_x,
                    block_y=config.rows.block_y,
                    block_z=config.rows.block_z,
                    waves_per_eu=config.rows.waves_per_eu,
                    tile_m=config.rows.tile_m,
                    tile_n=config.rows.tile_n,
                    quantization=config.rows.quantization,
                    pack_bytes=config.rows.pack_bytes,
                    quantize_from_bytes=config.rows.quantize_from_bytes,
                )
                if error is not None:
                    raise CommunicatorError(
                        Error(error), f"{config.rows} on {input.shape}"
                    )
        return output, residual_output

    @final
    def should_allreduce_add_attn_res_rms_norm(self, input: torch.Tensor) -> bool:
        """Whether `all_reduce_add_attn_res_rms_norm(input, ...)` runs here."""
        return (
            not self.disabled
            and self.state is not None
            and should("all_reduce_add_attn_res_rms_norm", input, self.state.capacity)
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
        config: AllReduceAddAttnResRmsNormConfig
        | AllReduceAddAttnResRmsNormSplitConfig
        | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`attn_res(prefix, all_reduce(input), blocks, ...)`, at the config given (the
        tuner's, the tests'), else at the table's pick for its rows and width: one
        kernel, or our all-reduce then AttnRes on its rows; with no `prefix` the sum
        starts one. Returns the new
        prefix and the AttnRes output, each a new tensor: `prefix` is only read.
        `write_index` >= 0 also stores the new prefix as that block of `blocks`, which
        must not be one of the `num_blocks` sources the call reads."""
        # REFUSED BEFORE C++ IS ASKED: a communicator that runs nothing, an input that
        # is not [rows, cols].
        if self.disabled:
            raise CommunicatorError(
                Error.disabled, f"{type(self).__name__} runs no call"
            )
        assert self.state is not None, "an open communicator has its state"
        if input.dim() != 2:
            raise CommunicatorError(Error.not_two_d, f"not {tuple(input.shape)}")

        # A CAPTURE'S WARMUP: vLLM discards its outputs, so nothing launches.
        prefix_output, output = torch.empty_like(input), torch.empty_like(input)
        if (
            self.lifecycle.status == "capturing"
            and not torch.cuda.is_current_stream_capturing()
        ):
            return prefix_output, output

        # THE CONFIG: the one given, else the tuned table's nearest input's (one
        # kernel, or the split: our all-reduce, then the op on this rank's rows).
        if config is None:
            rows, cols = input.shape
            _, picked = self.state.tuned_configs.nearest(
                "all_reduce_add_attn_res_rms_norm",
                self.arguments.quantization,
                self.arguments.world,
                rows,
                cols,
            )
            assert isinstance(
                picked,
                AllReduceAddAttnResRmsNormConfig
                | AllReduceAddAttnResRmsNormSplitConfig,
            ), picked
            config = picked
        launches: list[Config] = (
            [config.all_reduce, config.rows]
            if isinstance(config, AllReduceAddAttnResRmsNormSplitConfig)
            else [config]
        )
        # WHAT IT MOVES IS THE COMMUNICATOR'S: an exact one moves no INT4; a quantized
        # one's all-reduce kernel (the split's first) is the INT4 one, which passes a
        # call below its quantize_from_bytes exact itself.
        for c in launches:
            if self.arguments.quantization == "exact" and c.quantization != "exact":
                raise AssertionError(f"an exact communicator given an INT4 config: {c}")
            if (
                self.arguments.quantization == "int4"
                and c.kernel == "all_reduce"
                and c.quantization != "int4"
            ):
                raise AssertionError(
                    f"a quantized communicator given an exact config: {c}"
                )

        # THE ONE SLOT WRITTEN, a view of `blocks`; the sources read are the first
        # `num_blocks`, so the two never overlap (C++ refuses a call where they do).
        block_output = blocks[:, write_index] if write_index >= 0 else None
        match config:
            case AllReduceAddAttnResRmsNormConfig():
                # ONE KERNEL, the fused op.
                error = torch.ops._rocm_C.hip_custom_communicator_all_reduce_add_attn_res_rms_norm(  # noqa: E501
                    state=self.state.handle,
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
                if error is not None:
                    raise CommunicatorError(Error(error), f"{config} on {input.shape}")
            case AllReduceAddAttnResRmsNormSplitConfig():
                # TWO KERNELS OF OURS: our all-reduce into this rank's sum, then
                # AttnRes on its rows.
                summed = torch.empty_like(input)
                error = torch.ops._rocm_C.hip_custom_communicator_all_reduce(
                    state=self.state.handle,
                    input=input,
                    output=summed,
                    algorithm=config.all_reduce.algorithm,
                    grid_x=config.all_reduce.grid_x,
                    grid_y=config.all_reduce.grid_y,
                    grid_z=config.all_reduce.grid_z,
                    block_x=config.all_reduce.block_x,
                    block_y=config.all_reduce.block_y,
                    block_z=config.all_reduce.block_z,
                    waves_per_eu=config.all_reduce.waves_per_eu,
                    tile_m=config.all_reduce.tile_m,
                    tile_n=config.all_reduce.tile_n,
                    quantization=config.all_reduce.quantization,
                    pack_bytes=config.all_reduce.pack_bytes,
                    quantize_from_bytes=config.all_reduce.quantize_from_bytes,
                )
                if error is not None:
                    raise CommunicatorError(
                        Error(error), f"{config.all_reduce} on {input.shape}"
                    )
                error = torch.ops._rocm_C.hip_custom_communicator_add_attn_res_rms_norm(
                    state=self.state.handle,
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
                    grid_x=config.rows.grid_x,
                    grid_y=config.rows.grid_y,
                    grid_z=config.rows.grid_z,
                    block_x=config.rows.block_x,
                    block_y=config.rows.block_y,
                    block_z=config.rows.block_z,
                    waves_per_eu=config.rows.waves_per_eu,
                    tile_m=config.rows.tile_m,
                    tile_n=config.rows.tile_n,
                    tile_k=config.rows.tile_k,
                    quantization=config.rows.quantization,
                    pack_bytes=config.rows.pack_bytes,
                    quantize_from_bytes=config.rows.quantize_from_bytes,
                )
                if error is not None:
                    raise CommunicatorError(
                        Error(error), f"{config.rows} on {input.shape}"
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
        config: RmsNormConfig,
    ) -> torch.Tensor:
        """`vllm.ir.ops.rms_norm(input, weight, eps)` on this rank's rows, `input` a
        sum, at the config given. Returns the normed rows."""
        # REFUSED BEFORE C++ IS ASKED: a communicator that runs nothing, an input that
        # is not [rows, cols].
        if self.disabled:
            raise CommunicatorError(
                Error.disabled, f"{type(self).__name__} runs no call"
            )
        assert self.state is not None, "an open communicator has its state"
        if input.dim() != 2:
            raise CommunicatorError(Error.not_two_d, f"not {tuple(input.shape)}")

        # A CAPTURE'S WARMUP: vLLM discards its outputs, so nothing launches.
        output = torch.empty_like(input)
        if (
            self.lifecycle.status == "capturing"
            and not torch.cuda.is_current_stream_capturing()
        ):
            return output

        # WHAT IT MOVES IS THE COMMUNICATOR'S: a norm on rows moves nothing quantized.
        if config.quantization != "exact":
            raise AssertionError(f"a norm on rows given an INT4 config: {config}")

        error = torch.ops._rocm_C.hip_custom_communicator_rms_norm(
            state=self.state.handle,
            input=input,
            weight=weight,
            output=output,
            eps=eps,
            grid_x=config.grid_x,
            grid_y=config.grid_y,
            grid_z=config.grid_z,
            block_x=config.block_x,
            block_y=config.block_y,
            block_z=config.block_z,
            waves_per_eu=config.waves_per_eu,
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            quantization=config.quantization,
            pack_bytes=config.pack_bytes,
            quantize_from_bytes=config.quantize_from_bytes,
        )
        if error is not None:
            raise CommunicatorError(Error(error), f"{config} on {input.shape}")
        return output

    @final
    def add_rms_norm(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        *,
        config: AddRmsNormConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`vllm.ir.ops.fused_add_rms_norm(input, residual, weight, eps)` on this rank's
        rows, `input` a sum, at the config given. Returns the normed result and the sum
        plus residual."""
        # REFUSED BEFORE C++ IS ASKED: a communicator that runs nothing, an input that
        # is not [rows, cols].
        if self.disabled:
            raise CommunicatorError(
                Error.disabled, f"{type(self).__name__} runs no call"
            )
        assert self.state is not None, "an open communicator has its state"
        if input.dim() != 2:
            raise CommunicatorError(Error.not_two_d, f"not {tuple(input.shape)}")

        # A CAPTURE'S WARMUP: vLLM discards its outputs, so nothing launches.
        output, residual_output = torch.empty_like(input), torch.empty_like(input)
        if (
            self.lifecycle.status == "capturing"
            and not torch.cuda.is_current_stream_capturing()
        ):
            return output, residual_output

        # WHAT IT MOVES IS THE COMMUNICATOR'S: a norm on rows moves nothing quantized.
        if config.quantization != "exact":
            raise AssertionError(f"a norm on rows given an INT4 config: {config}")

        error = torch.ops._rocm_C.hip_custom_communicator_add_rms_norm(
            state=self.state.handle,
            input=input,
            residual=residual,
            weight=weight,
            output=output,
            residual_output=residual_output,
            eps=eps,
            grid_x=config.grid_x,
            grid_y=config.grid_y,
            grid_z=config.grid_z,
            block_x=config.block_x,
            block_y=config.block_y,
            block_z=config.block_z,
            waves_per_eu=config.waves_per_eu,
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            quantization=config.quantization,
            pack_bytes=config.pack_bytes,
            quantize_from_bytes=config.quantize_from_bytes,
        )
        if error is not None:
            raise CommunicatorError(Error(error), f"{config} on {input.shape}")
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
        config: AddAttnResRmsNormConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`attn_res(prefix, input, blocks, ...)` on this rank's rows, `input` a sum,
        its arguments `all_reduce_add_attn_res_rms_norm`'s, at the config given.
        Returns the new prefix and the AttnRes output."""
        # REFUSED BEFORE C++ IS ASKED: a communicator that runs nothing, an input that
        # is not [rows, cols].
        if self.disabled:
            raise CommunicatorError(
                Error.disabled, f"{type(self).__name__} runs no call"
            )
        assert self.state is not None, "an open communicator has its state"
        if input.dim() != 2:
            raise CommunicatorError(Error.not_two_d, f"not {tuple(input.shape)}")

        # A CAPTURE'S WARMUP: vLLM discards its outputs, so nothing launches.
        prefix_output, output = torch.empty_like(input), torch.empty_like(input)
        if (
            self.lifecycle.status == "capturing"
            and not torch.cuda.is_current_stream_capturing()
        ):
            return prefix_output, output

        # WHAT IT MOVES IS THE COMMUNICATOR'S: a norm on rows moves nothing quantized.
        if config.quantization != "exact":
            raise AssertionError(f"a norm on rows given an INT4 config: {config}")

        block_output = blocks[:, write_index] if write_index >= 0 else None
        error = torch.ops._rocm_C.hip_custom_communicator_add_attn_res_rms_norm(
            state=self.state.handle,
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
        if error is not None:
            raise CommunicatorError(Error(error), f"{config} on {input.shape}")
        return prefix_output, output

    # ---- A CAPTURE, and what tools and tests read: what a plain call launches, a
    # kernel's tuned config, the status, the signals, the check after a call. ----

    @final
    @contextmanager
    def capture(self) -> Iterator[None]:
        """Enter around a cudagraph capture. Required: a captured launch records an
        address that is not valid yet, so a backend has to be told. From `open`
        only (not nested, not closed); a capture that raises leaves it broken,
        since its peers may have registered what this rank did not."""
        with self.lifecycle.lock:
            if self.lifecycle.status != "open":
                raise CommunicatorError(
                    Error.capture_not_open,
                    f"capture() while {self.lifecycle.status}",
                )
            self.lifecycle.status = "capturing"
        try:
            yield
            # THE CAPTURED IN-PLACE LAUNCHES' INPUTS, registered before any replay
            # (vLLM's register_graph_buffers): every rank's IPC handles exchanged over
            # the CPU group (a collective, so every rank exits its capture together),
            # then C++ opens each peer allocation once and writes every slot its
            # launches reserved.
            assert self.state is not None
            ops = torch.ops._rocm_C
            error, handles, offsets = ops.hip_custom_communicator_graph_ipc_meta(
                state=self.state.handle
            )
            if error is not None:
                raise CommunicatorError(
                    Error(error), "the captured inputs' IPC handles"
                )
            everyone: list[Any] = [None] * self.arguments.world
            torch.distributed.all_gather_object(
                everyone, (handles, offsets), group=self.arguments.cpu_group
            )
            error = ops.hip_custom_communicator_register_graph_buffers(
                state=self.state.handle,
                handles=[h for rank_handles, _ in everyone for h in rank_handles],
                offsets=[o for _, rank_offsets in everyone for o in rank_offsets],
            )
            if error is not None:
                raise CommunicatorError(
                    Error(error),
                    f"captured inputs a rank: {[len(o) for _, o in everyone]}",
                )
        except BaseException as e:
            self._break(Error.capture_raised, f"{type(e).__name__}: {e}")
            raise
        with self.lifecycle.lock:
            if self.lifecycle.status == "capturing":
                self.lifecycle.status = "open"

    @final
    def tuned(self, op: Op, rows: int, cols: int) -> list[Config]:
        """WHAT A PLAIN CALL OF `op` ON `rows` x `cols` LAUNCHES here, in order: its
        table's nearest input's configs (`TunedConfigs.nearest`)."""
        assert self.state is not None, "an open communicator has its state"
        _, config = self.state.tuned_configs.nearest(
            op, self.arguments.quantization, self.arguments.world, rows, cols
        )
        if isinstance(config, SplitConfig):
            return [config.all_reduce, config.rows]
        return [config]

    @final
    def kernel_config(
        self,
        kernel: str,
        algorithm: Algorithm,
        rows: int,
        cols: int,
        quantization: Quantization = "exact",
    ) -> Config | None:
        """KERNEL `kernel`'S OWN TUNED CONFIG at `algorithm` for `rows` x `cols` moving
        `quantization` (`TunedConfigs.kernel_config`), or None."""
        assert self.state is not None, "an open communicator has its state"
        return self.state.tuned_configs.kernel_config(
            kernel, algorithm, rows, cols, quantization, self.arguments.world
        )

    # ---- THE HELPERS, after their callers: the handshake, a break. ----

    def _barrier(self) -> bool:
        """EVERY RANK ARRIVED over the CPU group within `timeout_seconds`, or False (a
        peer that did not, or a group that failed): never unbounded, a peer may be
        dead."""
        try:
            torch.distributed.barrier(
                group=self.arguments.cpu_group, async_op=True
            ).wait(timedelta(seconds=self.arguments.timeout_seconds))
        except RuntimeError:
            return False
        return True

    @final
    def _break(self, error: Error, detail: str) -> None:
        """Broken, once: every later call raises (`disabled`), and the cause goes to
        the log."""
        life = self.lifecycle
        with life.lock:
            if life.status in ("broken", "abandoned", "closed"):
                return
            life.status, life.error, life.detail = "broken", error, detail
        logger.error("%s broken: %s: %s", type(self).__name__, error.name, detail)

    # ---- ITS END: close, a collective, and what it frees; the abandon at exit; the
    # context manager's. ----

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
        if self.lifecycle.status == "closed":
            return
        if self.lifecycle.status == "capturing":
            raise CommunicatorError(
                Error.close_while_capturing,
                "the graph being recorded would replay against released buffers; "
                "leave the capture first",
            )
        with self.lifecycle.lock:
            self.lifecycle.status = "closed"
        self._on_close()

    def _on_close(self) -> None:
        """CLOSE, A COLLECTIVE (STANDARDS 3.5, "lifetimes are collective"): every rank
        calls it together. (a) every launch on this device finished, so nothing of ours
        reads or writes a peer's memory; (b) handshake: every rank has stopped; (c)
        every borrowed mapping closed; (d) handshake: every rank has closed its mappings
        of ours (skipped when (b) failed: a peer that missed it would pair this one with
        its own first); (e) our own buffers freed. A handshake that fails (a peer that
        did not arrive within the bound) is logged and the teardown carries on alone.
        After an abandon, only what it left. Idempotent."""
        with self.lifecycle.teardown:
            if self.state is None:
                return
            try:
                torch.cuda.synchronize(self.arguments.device)
            except RuntimeError as e:
                # A KERNEL FAULT IS STICKY: logged, and the teardown carries on (a
                # process must not die holding its peers' memory open).
                logger.error(
                    "HipCustomCommunicator close: the device did not finish: %s", e
                )
            stopped = self._barrier()
            self._close_peers()
            done = stopped and self._barrier()
            self._free()
        if not (stopped and done):
            logger.error(
                "HipCustomCommunicator close: a peer did not arrive at the %s "
                "handshake within %.0f s; it finished alone",
                "first" if not stopped else "second",
                self.arguments.timeout_seconds,
            )

    def _close_peers(self) -> None:
        """EVERY BORROWED MAPPING closed (C++'s close_peers: the peers' regions, their
        captured inputs). Idempotent; carries on past an error, logged."""
        if self.state is None:
            return
        error = torch.ops._rocm_C.hip_custom_communicator_close_peers(
            state=self.state.handle
        )
        if error is not None:
            logger.error(
                "HipCustomCommunicator: closing the peers: %s", Error(error).name
            )

    def _free(self) -> None:
        """OUR OWN buffers freed, and C++'s state with them (C++'s free): once no peer
        has them open. Idempotent; an error is logged."""
        if self.state is None:
            return
        error = torch.ops._rocm_C.hip_custom_communicator_free(state=self.state.handle)
        self.state = None
        if error is not None:
            logger.error("HipCustomCommunicator: freeing: %s", Error(error).name)

    @final
    def _abandoned(self, wait_s: float = ABANDON_WAIT_S) -> None:
        """ABANDONED, from the exit hook (`abandon_open`): torn down ALONE, never
        waiting on a peer (STANDARDS 3.5). The device given `wait_s` to finish (a wait
        on a dead peer traps at its timeout); every BORROWED peer mapping closed; and,
        only once the device finished, our own buffers freed (freeing memory a running
        kernel reads would fault it). A no-op while another teardown runs."""
        with self.lifecycle.lock:
            if not self.lifecycle.holds():
                return
            self.lifecycle.status = "abandoned"
        if self.state is None:
            return
        if not self.lifecycle.teardown.acquire(blocking=False):
            return
        try:
            finished = _synchronized_within(self.arguments.device, wait_s)
            self._close_peers()
            if finished:
                self._free()
        finally:
            self.lifecycle.teardown.release()

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
        if self.lifecycle.holds():
            warnings.warn(
                f"{type(self).__name__} was never closed; its peer handles and buffers "
                f"were left to garbage collection. Use `with make_communicator(...) as "
                f"comm:` "
                f"or call `close()`.",
                ResourceWarning,
                stacklevel=2,
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
