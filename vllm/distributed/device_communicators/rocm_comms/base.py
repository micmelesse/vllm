# SPDX-License-Identifier: MIT Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All
# rights reserved.

"""The interface every ROCm TP collective backend implements, and the two rules it
enforces once.

One file per backend beside this one: `hip`, `iris`, `torch`. This holds only what they
share -- the abstract surface `CudaCommunicator` calls, and the admission checks that
decide whether a tensor is one of ours at all.
"""

import functools
import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar, Literal

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .launch import Launch
from .tunables import Tunables

logger = logging.getLogger(__name__)

# ---- THE CAPABILITIES EVERY BACKEND HERE IS BOUND BY. Not tunables: a tunable is a
# number you may change with the code still correct, and changing one of these means
# changing a kernel. They are declared ONCE, here, for the same reason the admission
# envelope is: torch is the control, and a control that serves a superset is answering a
# different question than the backends it is a control for. ----

# EVERY OP, as C++ names them (`enum class Op`; a test holds them equal) and as its
# method is named: the all-reduce, then each all-reduce and the ops it fuses, in order.
Op = Literal[
    "all_reduce",
    "all_reduce_rms_norm",
    "all_reduce_add_rms_norm",
    "all_reduce_add_attn_res_rms_norm",
    "all_reduce_rms_norm_gemm_add",
    "all_reduce_rms_norm_gemm",
    "all_reduce_rms_scale_add",
]


def _as_device(device: int | str | torch.device) -> torch.device:
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    if isinstance(device, str):
        return torch.device(device)
    assert isinstance(device, torch.device)
    return device


class Error(IntEnum):
    """WHY A CALL CANNOT RUN: C++'s `hip_comms::Error`, number for number (a test
    holds them equal). Every backend answers with these; ours from its C++."""

    disabled = 0
    no_such_op = 1
    not_contiguous = 2
    not_two_d = 3
    output_not_two_d = 4
    dtype_not_built = 5
    world_not_built = 6
    row_not_packs = 7
    widths_not_packs = 8
    row_not_wider_than_output = 9
    template_not_this_ops = 10
    row_too_wide = 11
    block_not_a_wave_per_peer = 12
    quantized_not_built = 13
    block_exceeds_lds = 14
    scratch_too_small = 15
    grid_not_resident = 16
    staging_too_small = 17
    device_not_built = 18
    device_not_tuned = 19
    weight_not_built = 20
    no_such_template = 21


# C++'s `DType` names, as torch's dtypes.
_DTYPES: Mapping[str, torch.dtype] = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}


@dataclass(frozen=True)
class BuildInfo:
    """What the build holds, the same on every device (C++'s `kDTypesBuilt`,
    `kWorldsBuilt`, `kPackBytes`, `kStagingBytes`, its ops' and errors' names)."""

    dtypes: frozenset[torch.dtype]
    worlds: frozenset[int]
    pack_bytes: int
    staging_bytes: int
    # C++'s `Op` and `Error` members by name, each in its enum's order: an op crosses
    # by name, an Error by its number.
    op_names: tuple[str, ...]
    error_names: tuple[str, ...]


@functools.cache
def build_info() -> BuildInfo:
    """The build's facts, read once: they are fixed when it is compiled."""
    dtypes, worlds, pack, staging, ops, errors = (
        torch.ops._rocm_C.rocm_comms_build_info()
    )
    return BuildInfo(
        frozenset(_DTYPES[d] for d in dtypes),
        frozenset(worlds),
        pack,
        staging,
        tuple(ops),
        tuple(errors),
    )


@dataclass(frozen=True)
class Plan:
    """WHAT RUNS on a call this backend takes: hip's template, grid and block, or None
    for a backend with no kernels to choose (torch, iris)."""

    launch: Launch | None


@dataclass(frozen=True)
class Supported:
    """What the library runs on, once `supported` finds it can: the device's arch."""

    arch: str


def supported(device: torch.device, world: int) -> Supported | Error:
    """Whether our kernels run on `device` in a world of `world`, as the C++ build
    answers it (`hip_comms::supported`): the Supported, or the Error."""
    arch, err = torch.ops._rocm_C.rocm_comms_supported(device.index, world)
    return Error(err) if err is not None else Supported(arch)


def _cols(t: torch.Tensor, dim: int) -> int | None:
    """An output's columns, its `dim`, when it is 2-D; else None (refused)."""
    return t.shape[dim] if t.dim() == 2 else None


def _is_weak_contiguous(inp: torch.Tensor) -> bool:
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


class Communicator(ABC):
    """A TP all-reduce / all-gather backend behind vLLM's CudaCommunicator.

    This class owns the call SEQUENCE -- the admission gate, the capture invariant, and
    delegation -- and a backend supplies only the hooks at the bottom. Collectives are
    out-of-place.
    """

    disabled: bool
    tunables: Tunables
    world_size: int

    # THE OPS THIS BACKEND RUNS; every other is `no_such_op`.
    OPS: ClassVar[frozenset[Op]] = frozenset({"all_reduce"})

    _capturing: bool = False
    _closed: bool = False

    # WHAT THE BASE OWNS, and therefore what a backend may not override -- checked
    # when the class is DEFINED. A backend's own rules are its `_plan`.
    _OWNED = (
        "__init__",
        "should_allreduce",
        "should_allreduce_rms_norm",
        "should_allreduce_add_rms_norm",
        "should_allreduce_add_attn_res_rms_norm",
        "should_allreduce_rms_norm_gemm",
        "should_allreduce_rms_norm_gemm_add",
        "should_allreduce_rms_scale_add",
        "all_reduce",
        "all_reduce_rms_norm",
        "all_reduce_add_rms_norm",
        "all_reduce_add_attn_res_rms_norm",
        "all_reduce_rms_norm_gemm",
        "all_reduce_rms_norm_gemm_add",
        "all_reduce_rms_scale_add",
        "capture",
        "plan",
        "close",
        "__enter__",
        "__exit__",
    )

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        taken = [n for n in Communicator._OWNED if n in cls.__dict__]
        if taken:
            raise TypeError(
                f"{cls.__name__} overrides {taken}, which `Communicator` owns -- an "
                f"override skips the capture invariant and the shared envelope. "
                f"Supply `_all_reduce`, `_on_capture` or "
                f"`_on_close` instead -- what the kernel supports and how long it "
                f"lives are not a backend's to redefine."
            )

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device_group: ProcessGroup,
        device: int | str | torch.device,
        tunables: Tunables,
    ) -> None:
        """Every backend's construction, done ONCE here.

        It was three copies of the same prelude -- normalise the device, keep the two
        groups, read the world size, gate on the hardware -- and a backend now supplies
        only `_open`, the part that is actually its own.

        DISABLED FIRST, so every early return leaves a safe object rather than one whose
        flag depends on how far this got. Unavailability is not an error: the caller
        checks `.disabled`.
        """
        self.disabled = True
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.device = _as_device(device)
        self.tunables = tunables
        self.world_size = dist.get_world_size(device_group)

        # THE BOX, NOT THE BACKEND, asked of the build once for every backend. torch is
        # gated too: it is the CONTROL, and a control available where no backend is has
        # nothing to be a control for.
        got = supported(self.device, self.world_size)
        if isinstance(got, Error):
            logger.info("%s disabled: %s", type(self).__name__, got.name)
            return
        self.disabled = not self._open()

    # ---- What the CALLER uses. Concrete: this class owns the order. ----
    #
    # Every op and every `should_*` takes `launch`, a forced kernel and geometry, per
    # call. The sweep and the tests pass one (to hip, the one backend with kernels to
    # choose); the model passes None, the backend's own choice. And `quant_bits`, the
    # precision the caller accepts on the wire: 16 (exact) unless it says otherwise; a
    # backend without lossy kernels refuses anything else.

    def plan(
        self,
        op: Op,
        inp: torch.Tensor,
        launch: Launch | None = None,
        quant_bits: int = 16,
        cols: int | None = None,
        weight_dtype: torch.dtype | None = None,
    ) -> Plan | Error:
        """WHAT RUNS `op` over `inp` on this backend, or the Error it meets: the one
        place a refusal is decided. `cols` is the output's columns of an op that has
        them (the GEMM tails', the scale-add's), `weight_dtype` a norm's weight dtype
        (the norm ops'); None for the others. Every `should_*` is this, a Plan."""
        if self.disabled:
            return Error.disabled
        if op not in self.OPS:
            return Error.no_such_op
        return self._plan(op, inp, launch, quant_bits, cols, weight_dtype)

    def should_allreduce(
        self, inp: torch.Tensor, launch: Launch | None = None, quant_bits: int = 16
    ) -> bool:
        """Whether this backend takes `inp`: `plan` is a Plan."""
        return isinstance(self.plan("all_reduce", inp, launch, quant_bits), Plan)

    def should_allreduce_rms_norm(
        self,
        inp: torch.Tensor,
        weight_dtype: torch.dtype,
        launch: Launch | None = None,
        quant_bits: int = 16,
    ) -> bool:
        """Whether this backend can all-reduce then `rms_norm` `inp`, its weight in
        `weight_dtype`, in one kernel: `plan` is a Plan."""
        got = self.plan(
            "all_reduce_rms_norm", inp, launch, quant_bits, weight_dtype=weight_dtype
        )
        return isinstance(got, Plan)

    def should_allreduce_add_rms_norm(
        self,
        inp: torch.Tensor,
        weight_dtype: torch.dtype,
        launch: Launch | None = None,
        quant_bits: int = 16,
    ) -> bool:
        """As `should_allreduce_rms_norm`, for all-reduce then `fused_add_rms_norm`."""
        got = self.plan(
            "all_reduce_add_rms_norm",
            inp,
            launch,
            quant_bits,
            weight_dtype=weight_dtype,
        )
        return isinstance(got, Plan)

    def _is_small(self, inp: torch.Tensor) -> bool:
        """Whether `inp` is under `small_limit` -- the line that used to pick between
        this backend and QuickReduce, and now picks between this backend's OWN paths.
        Nothing calls it to refuse work; it is the switch for when a second kernel
        exists.
        """
        return inp.numel() * inp.element_size() < self.tunables.small_limit

    def all_reduce(
        self, inp: torch.Tensor, launch: Launch | None = None, quant_bits: int = 16
    ) -> torch.Tensor:
        """EVERY all-reduce this backend's kernel can compile for, at any SIZE.

        The size used to decide whether the caller kept it or handed it to QuickReduce,
        so an arm named for a backend was that backend under the limit and something
        else above it. Now the collective is ours and `_is_small` picks which of OUR
        paths it takes.

        SIZE IS NOT DECIDED HERE, though `_is_small` and `small_limit` live on this
        class because the LINE is shared. What a backend does on either side of it is
        the backend's, so the classification is offered and not applied.
        """
        self._require("all_reduce", inp, launch, quant_bits)
        out = torch.empty_like(inp)
        if not self._warming_up("all_reduce"):
            self._all_reduce(out, inp, launch, quant_bits)
        return out

    def all_reduce_rms_norm(
        self,
        inp: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        launch: Launch | None = None,
        quant_bits: int = 16,
    ) -> torch.Tensor:
        """`vllm.ir.ops.rms_norm(all_reduce(inp), weight, eps)` in one kernel.

        A variant of the collective, so it goes through the same two doors: the capture
        check and the admission check."""
        self._require(
            "all_reduce_rms_norm", inp, launch, quant_bits, weight_dtype=weight.dtype
        )
        out = torch.empty_like(inp)
        if not self._warming_up("all_reduce_rms_norm"):
            self._all_reduce_rms_norm(out, inp, weight, eps, launch, quant_bits)
        return out

    def all_reduce_add_rms_norm(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        launch: Launch | None = None,
        quant_bits: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`vllm.ir.ops.fused_add_rms_norm(all_reduce(inp), residual, weight, eps)` in
        one kernel. Returns the normed result, then the sum plus residual."""
        self._require(
            "all_reduce_add_rms_norm",
            inp,
            launch,
            quant_bits,
            weight_dtype=weight.dtype,
        )
        out, residual_out = torch.empty_like(inp), torch.empty_like(inp)
        if not self._warming_up("all_reduce_add_rms_norm"):
            self._all_reduce_add_rms_norm(
                out, residual_out, inp, residual, weight, eps, launch, quant_bits
            )
        return out, residual_out

    def should_allreduce_add_attn_res_rms_norm(
        self, inp: torch.Tensor, launch: Launch | None = None, quant_bits: int = 16
    ) -> bool:
        """As `should_allreduce_rms_norm`, for all-reduce then Kimi-K3's AttnRes."""
        got = self.plan("all_reduce_add_attn_res_rms_norm", inp, launch, quant_bits)
        return isinstance(got, Plan)

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
        launch: Launch | None = None,
        quant_bits: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`attn_res(prefix, all_reduce(inp), blocks, ...)` in one kernel, or with no
        `prefix` the sum starting one. Returns the prefix (updated in place when given)
        and the AttnRes output. `write_idx` >= 0 also stores the prefix as that
        block."""
        self._require("all_reduce_add_attn_res_rms_norm", inp, launch, quant_bits)
        # THE PREFIX IS UPDATED IN PLACE when given; with none, the sum starts one.
        prefix_out = torch.empty_like(inp) if prefix is None else prefix
        out = torch.empty_like(inp)
        if self._warming_up("all_reduce_add_attn_res_rms_norm"):
            return prefix_out, out
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
            launch,
            quant_bits,
        )
        return prefix_out, out

    def should_allreduce_rms_norm_gemm(
        self,
        inp: torch.Tensor,
        gemm_weight: torch.Tensor,
        launch: Launch | None = None,
        quant_bits: int = 16,
    ) -> bool:
        """As `should_allreduce_rms_norm`, for all-reduce then RMSNorm then a GEMM
        written into an output; `gemm_weight` is [N, hidden], and its N shapes the
        launch."""
        got = self.plan(
            "all_reduce_rms_norm_gemm",
            inp,
            launch,
            quant_bits,
            _cols(gemm_weight, 0),
        )
        return isinstance(got, Plan)

    def all_reduce_rms_norm_gemm(
        self,
        inp: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        gemm_weight: torch.Tensor,
        out: torch.Tensor,
        out_col0: int,
        launch: Launch | None = None,
        quant_bits: int = 16,
    ) -> None:
        """`out[:, out_col0:out_col0 + N] = rms_norm(all_reduce(inp), norm_weight, eps)
        @ gemm_weight.T` in one kernel, `gemm_weight` being [N, hidden]."""
        self._require(
            "all_reduce_rms_norm_gemm", inp, launch, quant_bits, _cols(gemm_weight, 0)
        )
        if self._warming_up("all_reduce_rms_norm_gemm"):
            return
        self._all_reduce_rms_norm_gemm(
            inp, norm_weight, eps, gemm_weight, out, out_col0, launch, quant_bits
        )

    def should_allreduce_rms_norm_gemm_add(
        self,
        inp: torch.Tensor,
        gemm_weight: torch.Tensor,
        launch: Launch | None = None,
        quant_bits: int = 16,
    ) -> bool:
        """As `should_allreduce_rms_norm`, for all-reduce then RMSNorm then a GEMM added
        into an output; `gemm_weight` is [N, hidden], and its N shapes the launch."""
        got = self.plan(
            "all_reduce_rms_norm_gemm_add",
            inp,
            launch,
            quant_bits,
            _cols(gemm_weight, 0),
        )
        return isinstance(got, Plan)

    def all_reduce_rms_norm_gemm_add(
        self,
        inp: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        gemm_weight: torch.Tensor,
        out: torch.Tensor,
        out_col0: int,
        launch: Launch | None = None,
        quant_bits: int = 16,
    ) -> None:
        """`out[:, out_col0:out_col0 + N] += rms_norm(all_reduce(inp), norm_weight, eps)
        @ gemm_weight.T` in one kernel, `gemm_weight` being [N, hidden]."""
        self._require(
            "all_reduce_rms_norm_gemm_add",
            inp,
            launch,
            quant_bits,
            _cols(gemm_weight, 0),
        )
        if self._warming_up("all_reduce_rms_norm_gemm_add"):
            return
        self._all_reduce_rms_norm_gemm_add(
            inp, norm_weight, eps, gemm_weight, out, out_col0, launch, quant_bits
        )

    def should_allreduce_rms_scale_add(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        launch: Launch | None = None,
        quant_bits: int = 16,
    ) -> bool:
        """As `should_allreduce_rms_norm`, for writing `out` [rows, hidden] from `inp`'s
        row [shared | projected | latent], the widths out's, out's and the rest."""
        got = self.plan(
            "all_reduce_rms_scale_add", inp, launch, quant_bits, _cols(out, 1)
        )
        return isinstance(got, Plan)

    def all_reduce_rms_scale_add(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        eps: float,
        launch: Launch | None = None,
        quant_bits: int = 16,
    ) -> None:
        """`s = all_reduce(inp)` split [shared | projected | latent], then `out = shared
        + projected * rsqrt(mean(latent^2) + eps)` in one kernel."""
        self._require(
            "all_reduce_rms_scale_add", inp, launch, quant_bits, _cols(out, 1)
        )
        if self._warming_up("all_reduce_rms_scale_add"):
            return
        self._all_reduce_rms_scale_add(inp, out, eps, launch, quant_bits)

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
        if self._closed:
            return
        if self._capturing:
            raise RuntimeError(
                f"{type(self).__name__}.close() inside `capture()`: the graph being "
                f"recorded would "
                f"replay against released buffers. Leave the capture first."
            )
        self._closed = True
        self._on_close()

    def __enter__(self) -> "Communicator":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        # WARNS, never releases. Doing GPU or IPC work here would run at an arbitrary
        # moment -- interpreter shutdown included, where the runtime may already be gone
        # -- so this only reports that a release was left to chance. Same bargain as an
        # unclosed file's ResourceWarning.
        if not self._closed and not getattr(self, "disabled", True):
            warnings.warn(
                f"{type(self).__name__} was never closed; its peer handles and buffers "
                f"were left to garbage collection. Use `with make_communicator(...) as "
                f"comm:` "
                f"or call `close()`.",
                ResourceWarning,
                stacklevel=2,
            )

    @contextmanager
    def capture(self) -> Iterator[None]:
        """Enter around a cudagraph capture. Required: a captured launch records an
        address that is not valid yet, so a backend has to be told."""
        self._capturing = True
        try:
            with self._on_capture():
                yield
        finally:
            self._capturing = False

    # ---- The two rules a caller can get wrong, enforced once. ----

    def _require(
        self,
        op: Op,
        inp: torch.Tensor,
        launch: Launch | None,
        quant_bits: int,
        cols: int | None = None,
        weight_dtype: torch.dtype | None = None,
    ) -> None:
        """Raise unless this backend runs `op` over `inp`: `plan`'s Error."""
        got = self.plan(op, inp, launch, quant_bits, cols, weight_dtype)
        if isinstance(got, Error):
            raise RuntimeError(self._rejected(op, inp, got))

    def _warming_up(self, op: str) -> bool:
        """Whether this is a capture's WARMUP: inside `capture()`, the stream not
        recording. vLLM discards its outputs, so an op returns unwritten ones of the
        right shape and launches nothing, as vLLM's and aiter's custom all-reduce do.
        Raises on a recording outside `capture()`."""
        recording = torch.cuda.is_current_stream_capturing()
        if recording and not self._capturing:
            raise RuntimeError(
                f"{type(self).__name__}.{op} is being captured into a cudagraph "
                f"without `capture()`. Use `with comm.capture(), torch.cuda.graph(g): "
                f"...` -- a backend may defer peer registration until that context "
                f"exits, and a graph captured without it "
                f"replays against addresses that were never registered."
            )
        return self._capturing and not recording

    def _rejected(self, op: str, inp: torch.Tensor, err: Error) -> str:
        """A refused call's message: the op, the input, and `check`'s Error."""
        return (
            f"{type(self).__name__} cannot run {op} over shape={tuple(inp.shape)} "
            f"dtype={inp.dtype}: {err.name}"
        )

    # ---- What a BACKEND supplies. ----

    @abstractmethod
    def _all_reduce(
        self,
        out: torch.Tensor,
        inp: torch.Tensor,
        launch: Launch | None = None,
        quant_bits: int = 16,
    ) -> None:
        """SUM across ranks into `out`, which the base allocated: input untouched.
        Assume `inp` is admitted -- the base checked. Every op's outputs are the base's,
        so a capture's warmup returns them without calling here."""

    # The fused variants. Not abstract: a backend without them is one the fusion pass
    # leaves alone, and overriding one is how a backend says it has it.

    def _all_reduce_rms_norm(
        self,
        out: torch.Tensor,
        inp: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        launch: Launch | None = None,
        quant_bits: int = 16,
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} has no fused all-reduce + rms_norm; "
            f"ask should_allreduce_rms_norm first."
        )

    def _all_reduce_add_rms_norm(
        self,
        out: torch.Tensor,
        residual_out: torch.Tensor,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        launch: Launch | None = None,
        quant_bits: int = 16,
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} has no fused all-reduce + fused_add_rms_norm; "
            f"ask should_allreduce_add_rms_norm first."
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
        launch: Launch | None = None,
        quant_bits: int = 16,
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} has no fused all-reduce + AttnRes; "
            f"ask should_allreduce_add_attn_res_rms_norm first."
        )

    def _all_reduce_rms_norm_gemm(
        self,
        inp: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        gemm_weight: torch.Tensor,
        out: torch.Tensor,
        out_col0: int,
        launch: Launch | None = None,
        quant_bits: int = 16,
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} has no fused all-reduce + rms_norm + gemm; "
            f"ask should_allreduce_rms_norm_gemm first."
        )

    def _all_reduce_rms_norm_gemm_add(
        self,
        inp: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        gemm_weight: torch.Tensor,
        out: torch.Tensor,
        out_col0: int,
        launch: Launch | None = None,
        quant_bits: int = 16,
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} has no fused all-reduce + rms_norm + gemm + add; "
            f"ask should_allreduce_rms_norm_gemm_add first."
        )

    def _all_reduce_rms_scale_add(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        eps: float,
        launch: Launch | None = None,
        quant_bits: int = 16,
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} has no fused all-reduce + rms scale + add; "
            f"ask should_allreduce_rms_scale_add first."
        )

    def _plan(
        self,
        op: Op,
        inp: torch.Tensor,
        launch: Launch | None = None,
        quant_bits: int = 16,
        cols: int | None = None,
        weight_dtype: torch.dtype | None = None,
    ) -> Plan | Error:
        """A backend's own rules: what runs `op` over `inp`, or the Error it meets. By
        default (torch, iris) the build's envelope, so a control admits what our kernels
        do: weak-contiguous, whole packs, a dtype built; no launch to choose and no
        lossy kernel. Ours overrides it with its C++'s answer."""
        self._refuse_launch(launch)
        self._refuse_lossy(quant_bits)
        if not _is_weak_contiguous(inp):
            return Error.not_contiguous
        built = build_info()
        if inp.numel() * inp.element_size() % built.pack_bytes != 0:
            return Error.row_not_packs
        if inp.dtype not in built.dtypes:
            return Error.dtype_not_built
        return Plan(None)

    def _refuse_lossy(self, quant_bits: int) -> None:
        """A LOSSY PRECISION has no kernel yet; a backend that cannot run one says so
        rather than returning an exact sum the caller did not ask to time."""
        if quant_bits != 16:
            raise ValueError(
                f"{type(self).__name__} runs exact only; got quant_bits={quant_bits}"
            )

    def _refuse_launch(self, launch: Launch | None) -> None:
        """A LAUNCH names hip's kernels; a backend without them says so rather than
        ignoring it, which would time something other than what was asked."""
        if launch is not None:
            raise ValueError(f"{type(self).__name__} takes no launch; got {launch!r}")

    def _open(self) -> bool:
        """Bring this backend up. True when it is usable; False leaves it disabled,
        which is not an error.

        NOTHING, by default -- torch needs no setup. It is where a backend's OWN
        availability checks go, the ones only it could run.
        """
        return True

    def _on_capture(self) -> AbstractContextManager[None]:
        """What this backend needs around a capture. Nothing, by default."""
        return nullcontext()

    def _on_close(self) -> None:  # noqa: B027 -- an optional hook, not an abstract one
        """What this backend has to release. Nothing, by default -- torch.distributed
        holds nothing
        of ours, and a backend that does drops it here so its destructor runs at a known
        moment."""
