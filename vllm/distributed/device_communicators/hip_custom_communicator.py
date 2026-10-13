# SPDX-License-Identifier: MIT Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""HipCustomCommunicator: vLLM's custom all-reduce (`custom_all_reduce.py`'s
CustomAllreduce) extended to every collective Kimi-K3 needs -- the all-reduce and the
all-reduces fused with the ops after them -- over our HIP kernels in `_rocm_C`
(`csrc/rocm/hip_custom_communicator.cu`).

The caller names an op and C++ picks the kernel and its launch (`select.cuh`); the
keyword arguments an op takes after its own (the algorithm, the direction, every field
of one build and the launch, all or none) are the one way to force one, for the sweep
and the tests. Every one is passed through as it came: C++ builds the config, checks
it, and answers.

TWO MEMORY PATHS, split by lifetime, both C++'s (`Handle::dev_comm`). A captured buffer
is held by vLLM for the graph's life, so it is registered once at capture exit and read
in place. An eager input is the caching allocator's, borrowed for the call, so C++
copies it into a staging buffer it owns.

The C++ context crosses as an opaque `int` handle, so nothing frees it for us:
`close()` has to run.
"""

import functools
import threading
import time
import warnings
from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar, Literal, final

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
    scratch_too_small = 10
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


class Where(IntEnum):
    """WHERE A KERNEL'S WAIT GAVE UP: C++'s `hip_custom_communicator::Where`, number
    for number (a test holds them equal)."""

    start = 1  # a launch's first barrier: the peer had not launched
    peer_barrier = 2  # a later barrier


@dataclass(frozen=True)
class Fault:
    """A RANK'S FAULT RECORD: the first of its kernels' waits to outlive the timeout,
    which then left the kernel (its output garbage) and made every other wait on the
    rank leave."""

    where: Where
    rank: int
    block: int
    peer: int  # the rank it waited for, -1 for none (a wait on this device)
    count: int  # the counter it read last
    want: int  # the value it waited for
    waited_s: float

    def __str__(self) -> str:
        waited_for = "" if self.peer < 0 else f" for rank {self.peer}"
        return (
            f"rank {self.rank} block {self.block} gave up in {self.where.name}"
            f"{waited_for} after {self.waited_s:.1f} s: count {self.count}, "
            f"want {self.want}"
        )


# C++'s `DType` names, as torch's dtypes.
_DTYPES: Mapping[str, torch.dtype] = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}


@dataclass(frozen=True)
class TemplateBuild:
    """One C++ template as built: its op, and the builds dispatch instantiates (its
    catalog list), each its fields by name (threads_per_block, waves_per_eu and its
    tile; never the launch, which a forced call gives itself)."""

    op: str
    builds: tuple[dict[str, int], ...]


@dataclass(frozen=True)
class OpRecord:
    """ONE OP'S PLAIN CALL as built (C++'s `kOps`): what a call of it with nothing
    forced must be to run, which `should` reads."""

    dtypes: frozenset[torch.dtype]  # its input's
    weight_dtypes: frozenset[torch.dtype]  # its weight's besides its input's; none
    widest_row: int  # elements a row at most; 0 for any (its tiles cross the row)
    max_bytes: int  # its input's at most
    two_d: bool  # its input is [rows, cols], not any shape read as rows of the last


@dataclass(frozen=True)
class BuildInfo:
    """What the build holds, the same on every device: C++'s `kBuild.supports`, its
    pack and staging bytes, its ops', errors' and fault places' names, its templates,
    and each op's plain call."""

    dtypes: frozenset[torch.dtype]
    worlds: frozenset[int]
    pack_bytes: int
    staging_bytes: int
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
        staging,
        ops,
        errors,
        wheres,
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
        builds = []
        for _ in range(count):
            values = flat[at : at + width]
            builds.append({f: v for f, v in zip(fields, values) if v != 0})
            at += width
        templates[name] = TemplateBuild(op, tuple(builds))
    records: dict[str, OpRecord] = {}
    at_dtype = at_weight = 0
    for op, at in zip(ops, range(0, len(op_facts), 5)):
        num_dtypes, num_weights, widest_row, max_bytes, two_d = op_facts[at : at + 5]
        dtypes_in = op_dtypes[at_dtype : at_dtype + num_dtypes]
        weights_in = op_weights[at_weight : at_weight + num_weights]
        at_dtype += num_dtypes
        at_weight += num_weights
        records[op] = OpRecord(
            frozenset(_DTYPES[d] for d in dtypes_in),
            frozenset(_DTYPES[d] for d in weights_in),
            widest_row,
            max_bytes,
            bool(two_d),
        )
    return BuildInfo(
        frozenset(_DTYPES[d] for d in dtypes),
        frozenset(worlds),
        pack,
        staging,
        tuple(ops),
        tuple(errors),
        tuple(wheres),
        templates,
        records,
    )


def should(op: Op, inp: torch.Tensor, weight: torch.Tensor | None = None) -> bool:
    """Whether a PLAIN call of `op` (nothing forced) on `inp`, and a norm's `weight`,
    runs: inside the op's record (`build_info().ops`) and whole packs a row. Python
    alone, no C++ call, as vLLM's `should_custom_ar`; C++ refuses no call this
    admits."""
    built = build_info()
    record = built.ops[op]
    if inp.dtype not in record.dtypes or not inp.is_contiguous() or inp.numel() == 0:
        return False
    if inp.dim() == 0 or (record.two_d and inp.dim() != 2):
        return False
    row = inp.shape[-1]
    if weight is not None and weight.dtype not in record.weight_dtypes | {inp.dtype}:
        return False
    return (
        row * inp.element_size() % built.pack_bytes == 0
        and (record.widest_row == 0 or row <= record.widest_row)
        and inp.numel() * inp.element_size() <= record.max_bytes
    )


class Refused(RuntimeError):
    """A call this backend does not take, raised by the op with the Error C++ refused
    it for, and the input it was asked to run on. A call `should` admits never is."""

    def __init__(self, error: Error, inp: torch.Tensor) -> None:
        super().__init__(f"{error.name}: shape={tuple(inp.shape)} dtype={inp.dtype}")
        self.error = error


@dataclass(frozen=True)
class Supported:
    """What the library runs on, once `supported` finds it can: the device's arch."""

    arch: str


def supported(device: torch.device, world: int) -> Supported | Error:
    """Whether our kernels run on `device` in a world of `world`, as the C++ build
    answers it (`hip_custom_communicator::supported`): the Supported, or the Error."""
    arch, err = torch.ops._rocm_C.hip_custom_communicator_supported(device.index, world)
    return Error(err) if err is not None else Supported(arch)


def is_weak_contiguous(inp: torch.Tensor) -> bool:
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


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


@dataclass(frozen=True)
class Call:
    """ONE LAUNCH, as the flight recorder keeps it. A graph's replays run none of this
    code, so a captured launch is recorded once, at its capture."""

    seq: int  # this communicator's launches before it
    op: Op
    shape: tuple[int, ...]
    dtype: torch.dtype
    captured: bool
    at_s: float  # time.monotonic() at the launch

    def __str__(self) -> str:
        how = " (captured)" if self.captured else ""
        return (
            f"#{self.seq} {self.op} {list(self.shape)} {self.dtype}{how} "
            f"at {self.at_s:.3f}"
        )


# THE FLIGHT RECORDER'S LENGTH: the launches a broken communicator logs, the last first.
_RECORDED = 64
# How often the watchdog reads the fault record: a fault is reported within this of the
# wait that wrote it giving up.
_WATCH_S = 0.1


def _require_capture(state: State) -> None:
    """Raise on a launch being recorded into a cudagraph outside `capture()`: peers are
    registered when that context exits, and a graph captured without it replays against
    addresses that were never registered."""
    if torch.cuda.is_current_stream_capturing() and not isinstance(state, Capturing):
        raise RuntimeError(
            "HipCustomCommunicator is being captured into a cudagraph without "
            "`capture()`. "
            "Use `with comm.capture(), torch.cuda.graph(g): ...`"
        )


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
    ) -> None:
        """Keep the groups and the device, then `_open`, which checks the backend's
        availability. Unavailability is not an error: the caller checks `.disabled`."""
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.device = _as_device(device)
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
        self._watchdog: threading.Thread | None = None
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
    # `forced_incomplete`. All None is the backend's choice; the model forces none. A
    # forced call asks no `should_`: C++ answers it. An op raises Refused with the
    # Error C++ refused it for; a call its `should_` admitted is never refused.

    @final
    def should_allreduce(self, inp: torch.Tensor) -> bool:
        """Whether `all_reduce(inp)` runs here."""
        return not self.disabled and should("all_reduce", inp)

    @final
    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        algorithm: str | None = None,
        direction: str | None = None,
        tile_m: int | None = None,
        tile_n: int | None = None,
        threads_per_block: int | None = None,
        blocks_per_grid: int | None = None,
        waves_per_eu: int | None = None,
    ) -> torch.Tensor:
        """EVERY all-reduce this backend's kernel can compile for, at any SIZE: size
        picks among a backend's own paths (hip's in C++), never whether it is ours.
        Returns the sum."""
        self._admit(inp)
        out = torch.empty_like(inp)
        if self._warming_up():
            return out
        refused = self._all_reduce(
            out,
            inp,
            algorithm,
            direction,
            tile_m,
            tile_n,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
        )
        self._launched("all_reduce", inp, refused)
        return out

    @final
    def should_allreduce_rms_norm(
        self, inp: torch.Tensor, weight: torch.Tensor
    ) -> bool:
        """Whether `all_reduce_rms_norm(inp, weight, ...)` runs here."""
        return not self.disabled and should("all_reduce_rms_norm", inp, weight)

    @final
    def all_reduce_rms_norm(
        self,
        inp: torch.Tensor,
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
    ) -> torch.Tensor:
        """`vllm.ir.ops.rms_norm(all_reduce(inp), weight, eps)` in one kernel. Returns
        the normed sum."""
        self._admit(inp)
        out = torch.empty_like(inp)
        if self._warming_up():
            return out
        refused = self._all_reduce_rms_norm(
            out,
            inp,
            weight,
            eps,
            algorithm,
            direction,
            tile_m,
            tile_n,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
        )
        self._launched("all_reduce_rms_norm", inp, refused)
        return out

    @final
    def should_allreduce_add_rms_norm(
        self, inp: torch.Tensor, weight: torch.Tensor
    ) -> bool:
        """Whether `all_reduce_add_rms_norm(inp, ..., weight, ...)` runs here."""
        return not self.disabled and should("all_reduce_add_rms_norm", inp, weight)

    @final
    def all_reduce_add_rms_norm(
        self,
        inp: torch.Tensor,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`vllm.ir.ops.fused_add_rms_norm(all_reduce(inp), residual, weight, eps)` in
        one kernel. Returns the normed result and the sum plus residual."""
        self._admit(inp)
        out, residual_out = torch.empty_like(inp), torch.empty_like(inp)
        if self._warming_up():
            return out, residual_out
        refused = self._all_reduce_add_rms_norm(
            out,
            residual_out,
            inp,
            residual,
            weight,
            eps,
            algorithm,
            direction,
            tile_m,
            tile_n,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
        )
        self._launched("all_reduce_add_rms_norm", inp, refused)
        return out, residual_out

    @final
    def should_allreduce_add_attn_res_rms_norm(self, inp: torch.Tensor) -> bool:
        """Whether `all_reduce_add_attn_res_rms_norm(inp, ...)` runs here."""
        return not self.disabled and should("all_reduce_add_attn_res_rms_norm", inp)

    @final
    def all_reduce_add_attn_res_rms_norm(
        self,
        inp: torch.Tensor,
        prefix: torch.Tensor | None,
        blocks: torch.Tensor,
        norm_weight: torch.Tensor,
        qk_weight: torch.Tensor,
        out_norm_weight: torch.Tensor | None,
        num_blocks: int,
        write_idx: int,
        eps: float,
        out_eps: float,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`attn_res(prefix, all_reduce(inp), blocks, ...)` in one kernel, or with no
        `prefix` the sum starting one. Returns the prefix (updated in place when given)
        and the AttnRes output. `write_idx` >= 0 also stores the prefix as that
        block."""
        self._admit(inp)
        # THE PREFIX IS UPDATED IN PLACE when given; with none, the sum starts one.
        prefix_out = torch.empty_like(inp) if prefix is None else prefix
        out = torch.empty_like(inp)
        if self._warming_up():
            return prefix_out, out
        refused = self._all_reduce_add_attn_res_rms_norm(
            prefix_out,
            out,
            inp,
            prefix is not None,
            blocks,
            norm_weight,
            qk_weight,
            out_norm_weight,
            num_blocks,
            write_idx,
            eps,
            out_eps,
            algorithm,
            direction,
            tile_m,
            tile_n,
            tile_k,
            reduce_scatter_blocks,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
        )
        self._launched("all_reduce_add_attn_res_rms_norm", inp, refused)
        return prefix_out, out

    @final
    def close(self) -> None:
        """Release what this communicator holds, NOW. Idempotent, and safe to call on a
        disabled one.

        Deterministic release rather than waiting to be collected, because the
        collection is what cannot be relied on: dropping the last reference frees a
        backend's IPC handles and device buffers through its destructor, but an
        exception TRACEBACK holds the frames that hold the communicator -- so a failing
        call is exactly when `del` does not release. This does.

        LOCAL only, deliberately: nothing here coordinates across ranks, so a rank
        closing early or late cannot hang its peers. Ordering that does matter is the
        caller's -- close before the process group is destroyed, and after any captured
        graph is gone.
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
            "%s broken: %s. Its last %d launches, the last first:\n%s",
            type(self).__name__,
            cause,
            len(self._calls),
            calls or "  (none)",
        )

    @final
    def _admit(self, inp: torch.Tensor) -> None:
        """Before an op: refused when no call may run (`disabled`), and raise on a
        capture outside `capture()`."""
        if self.disabled:
            raise Refused(Error.disabled, inp)
        _require_capture(self.state)

    @final
    def _launched(self, op: Op, inp: torch.Tensor, refused: int | None) -> None:
        """After an op's kernel hook: Refused with the Error's number it answered, or
        the launch into the flight recorder."""
        if refused is not None:
            raise Refused(Error(refused), inp)
        self._record(op, inp)

    @final
    def _record(self, op: Op, inp: torch.Tensor) -> None:
        """Into the flight recorder, as a launch goes out."""
        self._calls.append(
            Call(
                seq=self._launches,
                op=op,
                shape=tuple(inp.shape),
                dtype=inp.dtype,
                captured=torch.cuda.is_current_stream_capturing(),
                at_s=time.monotonic(),
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

    # ---- THE KERNELS: each op's launch (the Error's number C++ refused it for, or
    # None), and the peer memory's life. ----

    # Declared here so a disabled communicator is still safe to hold and close.
    _handle: int | None = None
    # The device wall clock's rate, which the fault record counts its wait in.
    _khz: int = 0
    # Whether there is a fault record for the watchdog to read (`_fault`).
    _faults: ClassVar[bool] = True

    def _open(self) -> str | None:
        """Open the peer memory. A collective, so every rank must reach it.

        Eager, and before any capture: doing this inside a cudagraph capture is not
        recoverable.
        """
        # THE PEER MEMORY, made, sized and owned by C++, which checks the groups and the
        # device and gathers every rank's IPC handle over the CPU group (a collective).
        handle, err = torch.ops._rocm_C.hip_custom_communicator_open(
            self.cpu_group.group_name, self.device_group.group_name, self.device.index
        )
        if err is not None:
            return Error(err).name
        self._handle = handle
        self._khz = torch.ops._rocm_C.hip_custom_communicator_wall_clock_khz(
            self.device.index
        )
        logger.info("HipCustomCommunicator ready")
        return None

    @contextmanager
    def _on_capture(self) -> Iterator[None]:
        """Wrap a cudagraph capture; the buffers its launches recorded are registered
        on exit, by C++, a collective over the CPU group. A captured address is fixed
        for the graph's life, so filling the slots after capture is sound."""
        try:
            yield
        finally:
            # Even on error: unfilled slots fault later on a null peer pointer.
            torch.ops._rocm_C.hip_custom_communicator_register_captured(
                self._handle, self.cpu_group.group_name
            )

    def _all_reduce(
        self,
        out: torch.Tensor,
        inp: torch.Tensor,
        algorithm: str | None,
        direction: str | None,
        tile_m: int | None,
        tile_n: int | None,
        threads_per_block: int | None,
        blocks_per_grid: int | None,
        waves_per_eu: int | None,
    ) -> int | None:
        """Sum `inp` across the TP ranks into `out`; the Error's number C++ refused
        the call for, or None when it ran (as every kernel hook answers)."""
        return torch.ops._rocm_C.hip_custom_communicator_all_reduce(
            self._handle,
            out,
            inp,
            algorithm,
            direction,
            tile_m,
            tile_n,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
        )

    def _all_reduce_rms_norm(
        self,
        out: torch.Tensor,
        inp: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        algorithm: str | None,
        direction: str | None,
        tile_m: int | None,
        tile_n: int | None,
        threads_per_block: int | None,
        blocks_per_grid: int | None,
        waves_per_eu: int | None,
    ) -> int | None:
        return torch.ops._rocm_C.hip_custom_communicator_all_reduce_rms_norm(
            self._handle,
            out,
            inp,
            weight,
            eps,
            algorithm,
            direction,
            tile_m,
            tile_n,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
        )

    def _all_reduce_add_rms_norm(
        self,
        out: torch.Tensor,
        residual_out: torch.Tensor,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        algorithm: str | None,
        direction: str | None,
        tile_m: int | None,
        tile_n: int | None,
        threads_per_block: int | None,
        blocks_per_grid: int | None,
        waves_per_eu: int | None,
    ) -> int | None:
        """The normed result into `out`, the sum plus residual into `residual_out`."""
        return torch.ops._rocm_C.hip_custom_communicator_all_reduce_add_rms_norm(
            self._handle,
            out,
            residual_out,
            inp,
            residual,
            weight,
            eps,
            algorithm,
            direction,
            tile_m,
            tile_n,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
        )

    def _all_reduce_add_attn_res_rms_norm(
        self,
        prefix_out: torch.Tensor,
        out: torch.Tensor,
        inp: torch.Tensor,
        has_prefix: bool,
        blocks: torch.Tensor,
        norm_weight: torch.Tensor,
        qk_weight: torch.Tensor,
        out_norm_weight: torch.Tensor | None,
        num_blocks: int,
        write_idx: int,
        eps: float,
        out_eps: float,
        algorithm: str | None,
        direction: str | None,
        tile_m: int | None,
        tile_n: int | None,
        tile_k: int | None,
        reduce_scatter_blocks: int | None,
        threads_per_block: int | None,
        blocks_per_grid: int | None,
        waves_per_eu: int | None,
    ) -> int | None:
        launch = (
            torch.ops._rocm_C.hip_custom_communicator_all_reduce_add_attn_res_rms_norm
        )
        return launch(
            self._handle,
            prefix_out,
            out,
            inp,
            blocks,
            norm_weight,
            qk_weight,
            out_norm_weight,
            num_blocks,
            write_idx,
            eps,
            out_eps,
            has_prefix,
            algorithm,
            direction,
            tile_m,
            tile_n,
            tile_k,
            reduce_scatter_blocks,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
        )

    def _on_close(self) -> None:
        """Release the peer memory now. Idempotent. Dropping the object does not close
        the IPC handles the C++ side opened; this does."""
        if self._handle is None:
            return
        torch.ops._rocm_C.hip_custom_communicator_dispose(self._handle)
        self._handle = None

    def _fault(self) -> Fault | None:
        """The C++ handle's fault record (host-mapped, so this never waits on the
        device)."""
        if self._handle is None:
            return None
        got = torch.ops._rocm_C.hip_custom_communicator_fault(self._handle)
        if not got:
            return None
        where, rank, block, peer, count, want, ticks = got
        return Fault(
            where=Where(where),
            rank=rank,
            block=block,
            peer=peer,
            count=count,
            want=want,
            waited_s=ticks / (self._khz * 1000.0),
        )

    def _abort(self) -> None:
        if self._handle is not None:
            torch.ops._rocm_C.hip_custom_communicator_abort(self._handle)
