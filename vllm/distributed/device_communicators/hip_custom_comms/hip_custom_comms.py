# SPDX-License-Identifier: MIT Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""HipCustomComms: vLLM's custom all-reduce (`custom_all_reduce.py`'s CustomAllreduce)
extended to every collective Kimi-K3 needs -- the all-reduce and the all-reduces fused
with the ops after them -- over our HIP kernels in `_rocm_C`
(`csrc/rocm/hip_custom_comms.cu`).

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

import threading
import time
import warnings
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import ClassVar, final

import torch
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

from .build import Error, Fault, Op, Ran, Refused, Where

logger = init_logger(__name__)


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
# Only `HipCustomComms` moves it, under its lock (the watchdog is another thread).
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
    ran: "Ran"
    captured: bool
    at_s: float  # time.monotonic() at the launch

    def __str__(self) -> str:
        how = " (captured)" if self.captured else ""
        return (
            f"#{self.seq} {self.op} {list(self.shape)} {self.dtype} "
            f"ran {self.ran}{how} at {self.at_s:.3f}"
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
            "HipCustomComms is being captured into a cudagraph without `capture()`. "
            "Use `with comm.capture(), torch.cuda.graph(g): ...`"
        )


def _answer(got: tuple[str | int | None, ...]) -> Ran | Error:
    """A C++ planner's answer: what would run (the algorithm, the direction and the op's
    config fields), or its last value, the Error's number."""
    err = got[-1]
    if err is not None:
        assert isinstance(err, int)
        return Error(err)
    return tuple(got[:-1])


class HipCustomComms:
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
    # Each op and its `should_` take the same arguments: the op's own tensors and
    # values, then, keyword-only, what the sweep and the tests force: ALL OR NOTHING,
    # the algorithm (`one_shot` or `two_shot`) and the direction (`pull` or `push`)
    # where the op has more than one template, every field of one of its builds
    # (`build_info().templates`), and the launch (blocks_per_grid, and the AttnRes
    # op's reduce_scatter_blocks); some but not all is `forced_incomplete`. All None
    # is the backend's choice; the model forces none.
    # `should_` is the op's check; the op runs its check, raises Refused on an Error,
    # and returns what ran.

    @final
    def should_allreduce(
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
    ) -> bool:
        """Whether `all_reduce(inp)` runs here."""
        ran = self._check_all_reduce(
            inp,
            algorithm,
            direction,
            tile_m,
            tile_n,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
        )
        return not isinstance(ran, Error)

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
    ) -> tuple[torch.Tensor, Ran]:
        """EVERY all-reduce this backend's kernel can compile for, at any SIZE: size
        picks among a backend's own paths (hip's in C++), never whether it is ours.
        Returns the sum and what ran."""
        ran = self._check_all_reduce(
            inp,
            algorithm,
            direction,
            tile_m,
            tile_n,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
        )
        if isinstance(ran, Error):
            raise Refused(ran, inp)
        out = torch.empty_like(inp)
        if not self._warming_up():
            self._record("all_reduce", inp, ran)
            self._all_reduce(
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
        return out, ran

    @final
    def should_allreduce_rms_norm(
        self,
        inp: torch.Tensor,
        weight: torch.Tensor,
        *,
        algorithm: str | None = None,
        direction: str | None = None,
        tile_m: int | None = None,
        tile_n: int | None = None,
        threads_per_block: int | None = None,
        blocks_per_grid: int | None = None,
        waves_per_eu: int | None = None,
    ) -> bool:
        """Whether `all_reduce_rms_norm(inp, weight, ...)` runs here."""
        ran = self._check_all_reduce_rms_norm(
            inp,
            weight,
            False,
            algorithm,
            direction,
            tile_m,
            tile_n,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
        )
        return not isinstance(ran, Error)

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
    ) -> tuple[torch.Tensor, Ran]:
        """`vllm.ir.ops.rms_norm(all_reduce(inp), weight, eps)` in one kernel. Returns
        the normed sum and what ran."""
        ran = self._check_all_reduce_rms_norm(
            inp,
            weight,
            False,
            algorithm,
            direction,
            tile_m,
            tile_n,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
        )
        if isinstance(ran, Error):
            raise Refused(ran, inp)
        out = torch.empty_like(inp)
        if not self._warming_up():
            self._record("all_reduce_rms_norm", inp, ran)
            self._all_reduce_rms_norm(
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
        return out, ran

    @final
    def should_allreduce_add_rms_norm(
        self,
        inp: torch.Tensor,
        weight: torch.Tensor,
        *,
        algorithm: str | None = None,
        direction: str | None = None,
        tile_m: int | None = None,
        tile_n: int | None = None,
        threads_per_block: int | None = None,
        blocks_per_grid: int | None = None,
        waves_per_eu: int | None = None,
    ) -> bool:
        """Whether `all_reduce_add_rms_norm(inp, ..., weight, ...)` runs here."""
        ran = self._check_all_reduce_rms_norm(
            inp,
            weight,
            True,
            algorithm,
            direction,
            tile_m,
            tile_n,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
        )
        return not isinstance(ran, Error)

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
    ) -> tuple[torch.Tensor, torch.Tensor, Ran]:
        """`vllm.ir.ops.fused_add_rms_norm(all_reduce(inp), residual, weight, eps)` in
        one kernel. Returns the normed result, the sum plus residual, and what ran."""
        ran = self._check_all_reduce_rms_norm(
            inp,
            weight,
            True,
            algorithm,
            direction,
            tile_m,
            tile_n,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
        )
        if isinstance(ran, Error):
            raise Refused(ran, inp)
        out, residual_out = torch.empty_like(inp), torch.empty_like(inp)
        if not self._warming_up():
            self._record("all_reduce_add_rms_norm", inp, ran)
            self._all_reduce_add_rms_norm(
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
        return out, residual_out, ran

    @final
    def should_allreduce_add_attn_res_rms_norm(
        self,
        inp: torch.Tensor,
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
    ) -> bool:
        """Whether `all_reduce_add_attn_res_rms_norm(inp, ...)` runs here."""
        ran = self._check_all_reduce_add_attn_res_rms_norm(
            inp,
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
        return not isinstance(ran, Error)

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
    ) -> tuple[torch.Tensor, torch.Tensor, Ran]:
        """`attn_res(prefix, all_reduce(inp), blocks, ...)` in one kernel, or with no
        `prefix` the sum starting one. Returns the prefix (updated in place when given),
        the AttnRes output, and what ran. `write_idx` >= 0 also stores the prefix as
        that block."""
        ran = self._check_all_reduce_add_attn_res_rms_norm(
            inp,
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
        if isinstance(ran, Error):
            raise Refused(ran, inp)
        # THE PREFIX IS UPDATED IN PLACE when given; with none, the sum starts one.
        prefix_out = torch.empty_like(inp) if prefix is None else prefix
        out = torch.empty_like(inp)
        if self._warming_up():
            return prefix_out, out, ran
        self._record("all_reduce_add_attn_res_rms_norm", inp, ran)
        self._all_reduce_add_attn_res_rms_norm(
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
        return prefix_out, out, ran

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
    def __enter__(self) -> "HipCustomComms":
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
    def _record(self, op: Op, inp: torch.Tensor, ran: Ran) -> None:
        """Into the flight recorder, as a launch goes out."""
        self._calls.append(
            Call(
                seq=self._launches,
                op=op,
                shape=tuple(inp.shape),
                dtype=inp.dtype,
                ran=ran,
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

    # ---- THE KERNELS: each op's check (what C++ would run, or the Error) and launch,
    # and the peer memory's life. ----

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
        handle, err = torch.ops._rocm_C.hip_custom_comms_open(
            self.cpu_group.group_name, self.device_group.group_name, self.device.index
        )
        if err is not None:
            return Error(err).name
        self._handle = handle
        self._khz = torch.ops._rocm_C.hip_custom_comms_wall_clock_khz(self.device.index)
        logger.info("HipCustomComms ready")
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
            torch.ops._rocm_C.hip_custom_comms_register_captured(
                self._handle, self.cpu_group.group_name
            )

    def _check_all_reduce(
        self,
        inp: torch.Tensor,
        algorithm: str | None,
        direction: str | None,
        tile_m: int | None,
        tile_n: int | None,
        threads_per_block: int | None,
        blocks_per_grid: int | None,
        waves_per_eu: int | None,
    ) -> Ran | Error:
        if self.disabled:
            return Error.disabled
        _require_capture(self.state)
        return _answer(
            torch.ops._rocm_C.hip_custom_comms_plan_all_reduce(
                self._handle,
                inp,
                algorithm,
                direction,
                tile_m,
                tile_n,
                threads_per_block,
                blocks_per_grid,
                waves_per_eu,
            )
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
    ) -> None:
        """Sum `inp` across the TP ranks into `out`."""
        torch.ops._rocm_C.hip_custom_comms_all_reduce(
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

    def _check_all_reduce_rms_norm(
        self,
        inp: torch.Tensor,
        weight: torch.Tensor,
        add: bool,
        algorithm: str | None,
        direction: str | None,
        tile_m: int | None,
        tile_n: int | None,
        threads_per_block: int | None,
        blocks_per_grid: int | None,
        waves_per_eu: int | None,
    ) -> Ran | Error:
        if self.disabled:
            return Error.disabled
        _require_capture(self.state)
        return _answer(
            torch.ops._rocm_C.hip_custom_comms_plan_all_reduce_rms_norm(
                self._handle,
                inp,
                weight,
                add,
                algorithm,
                direction,
                tile_m,
                tile_n,
                threads_per_block,
                blocks_per_grid,
                waves_per_eu,
            )
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
    ) -> None:
        torch.ops._rocm_C.hip_custom_comms_all_reduce_rms_norm(
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
    ) -> None:
        """The normed result into `out`, the sum plus residual into `residual_out`."""
        torch.ops._rocm_C.hip_custom_comms_all_reduce_add_rms_norm(
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

    def _check_all_reduce_add_attn_res_rms_norm(
        self,
        inp: torch.Tensor,
        algorithm: str | None,
        direction: str | None,
        tile_m: int | None,
        tile_n: int | None,
        tile_k: int | None,
        reduce_scatter_blocks: int | None,
        threads_per_block: int | None,
        blocks_per_grid: int | None,
        waves_per_eu: int | None,
    ) -> Ran | Error:
        if self.disabled:
            return Error.disabled
        _require_capture(self.state)
        return _answer(
            torch.ops._rocm_C.hip_custom_comms_plan_all_reduce_add_attn_res_rms_norm(
                self._handle,
                inp,
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
    ) -> None:
        torch.ops._rocm_C.hip_custom_comms_all_reduce_add_attn_res_rms_norm(
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
        torch.ops._rocm_C.hip_custom_comms_dispose(self._handle)
        self._handle = None

    def _fault(self) -> Fault | None:
        """The C++ handle's fault record (host-mapped, so this never waits on the
        device)."""
        if self._handle is None:
            return None
        got = torch.ops._rocm_C.hip_custom_comms_fault(self._handle)
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
            torch.ops._rocm_C.hip_custom_comms_abort(self._handle)
