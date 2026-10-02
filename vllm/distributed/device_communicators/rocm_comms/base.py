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
from torch.distributed import ProcessGroup

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


# A LOSSY PRECISION on the wire, in bits; none is exact.
QuantBits = Literal[8, 4]


@dataclass(frozen=True)
class Options:
    """HOW A CALL RUNS, beside what it computes: C++'s `hip_comms::Options`. The model
    passes none. `template` (a C++ template's name, `kTemplates` in
    `csrc/rocm/rocm_comms/impl/templates.cuh`), `blocks` and `threads` force a launch,
    all three or none (C++ refuses the rest); none is select's choice."""

    quant_bits: QuantBits | None = None
    template: str | None = None
    blocks: int | None = None
    threads: int | None = None


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
    no_such_group = 22
    ranks_disagree = 23
    groups_disagree = 24


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
    """WHAT RUNS on a call this backend takes: hip's template, grid and block; none of
    them for a backend with no kernels to choose (torch, iris)."""

    template: str | None = None
    blocks: int | None = None
    threads: int | None = None


# A CALL'S INPUTS AND OUTPUTS, one type per op family, as C++'s `AllReduceArgs`,
# `NormArgs`, `AttnResArgs`, `GemmTailArgs` and `ScaleAddArgs`: what `plan` is asked
# about. Each holds the tensors the decision reads, and the backend reads their facts.


@dataclass(frozen=True)
class AllReduceArgs:
    inp: torch.Tensor

    @property
    def op(self) -> Op:
        return "all_reduce"


@dataclass(frozen=True)
class NormArgs:
    """All-reduce then `rms_norm`, or `fused_add_rms_norm` with `add`."""

    inp: torch.Tensor
    weight: torch.Tensor
    add: bool

    @property
    def op(self) -> Op:
        return "all_reduce_add_rms_norm" if self.add else "all_reduce_rms_norm"


@dataclass(frozen=True)
class AttnResArgs:
    inp: torch.Tensor

    @property
    def op(self) -> Op:
        return "all_reduce_add_attn_res_rms_norm"


@dataclass(frozen=True)
class GemmTailArgs:
    """All-reduce, `rms_norm`, then a GEMM by `gemm_weight` [N, hidden], written or
    added (`add`)."""

    inp: torch.Tensor
    gemm_weight: torch.Tensor
    add: bool

    @property
    def op(self) -> Op:
        return (
            "all_reduce_rms_norm_gemm_add" if self.add else "all_reduce_rms_norm_gemm"
        )


@dataclass(frozen=True)
class ScaleAddArgs:
    """`inp`'s row [shared | projected | latent] into `out` [rows, hidden]."""

    inp: torch.Tensor
    out: torch.Tensor

    @property
    def op(self) -> Op:
        return "all_reduce_rms_scale_add"


Args = AllReduceArgs | NormArgs | AttnResArgs | GemmTailArgs | ScaleAddArgs


@dataclass(frozen=True)
class Supported:
    """What the library runs on, once `supported` finds it can: the device's arch."""

    arch: str


def supported(device: torch.device, world: int) -> Supported | Error:
    """Whether our kernels run on `device` in a world of `world`, as the C++ build
    answers it (`hip_comms::supported`): the Supported, or the Error."""
    arch, err = torch.ops._rocm_C.rocm_comms_supported(device.index, world)
    return Error(err) if err is not None else Supported(arch)


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
    ) -> None:
        """Keep the groups and the device, then `_open`, which checks the backend's
        availability. Unavailability is not an error: the caller checks `.disabled`."""
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.device = _as_device(device)
        self.disabled = not self._open()

    # ---- What the CALLER uses. Concrete: this class owns the order. ----
    #
    # Every op and every `should_*` takes `options`, per call, none meaning the
    # defaults: a forced template and geometry, which the sweep and the tests pass (to
    # hip, the one backend with kernels to choose) and the model leaves to the backend;
    # and a lossy `quant_bits`, which a backend without lossy kernels refuses.

    def plan(self, args: Args, options: Options | None = None) -> Plan | Error:
        """WHAT RUNS the call `args` on this backend, or the Error it meets: the one
        place a refusal is decided, the options' included. Every `should_*` is this, a
        Plan."""
        if self.disabled:
            return Error.disabled
        if args.op not in self.OPS:
            return Error.no_such_op
        return self._plan(args, Options() if options is None else options)

    def should_allreduce(
        self, inp: torch.Tensor, options: Options | None = None
    ) -> bool:
        """Whether this backend takes `inp`: `plan` is a Plan."""
        return isinstance(self.plan(AllReduceArgs(inp), options), Plan)

    def should_allreduce_rms_norm(
        self,
        inp: torch.Tensor,
        weight: torch.Tensor,
        options: Options | None = None,
    ) -> bool:
        """Whether this backend can all-reduce then `rms_norm` `inp` by `weight` in one
        kernel: `plan` is a Plan."""
        return isinstance(self.plan(NormArgs(inp, weight, add=False), options), Plan)

    def should_allreduce_add_rms_norm(
        self,
        inp: torch.Tensor,
        weight: torch.Tensor,
        options: Options | None = None,
    ) -> bool:
        """As `should_allreduce_rms_norm`, for all-reduce then `fused_add_rms_norm`."""
        return isinstance(self.plan(NormArgs(inp, weight, add=True), options), Plan)

    def all_reduce(
        self, inp: torch.Tensor, options: Options | None = None
    ) -> torch.Tensor:
        """EVERY all-reduce this backend's kernel can compile for, at any SIZE: size
        picks among a backend's own paths (hip's in C++), never whether it is ours."""
        options = self._require(AllReduceArgs(inp), options)
        out = torch.empty_like(inp)
        if not self._warming_up("all_reduce"):
            self._all_reduce(out, inp, options)
        return out

    def all_reduce_rms_norm(
        self,
        inp: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        options: Options | None = None,
    ) -> torch.Tensor:
        """`vllm.ir.ops.rms_norm(all_reduce(inp), weight, eps)` in one kernel.

        A variant of the collective, so it goes through the same two doors: the capture
        check and the admission check."""
        options = self._require(NormArgs(inp, weight, add=False), options)
        out = torch.empty_like(inp)
        if not self._warming_up("all_reduce_rms_norm"):
            self._all_reduce_rms_norm(out, inp, weight, eps, options)
        return out

    def all_reduce_add_rms_norm(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        options: Options | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`vllm.ir.ops.fused_add_rms_norm(all_reduce(inp), residual, weight, eps)` in
        one kernel. Returns the normed result, then the sum plus residual."""
        options = self._require(NormArgs(inp, weight, add=True), options)
        out, residual_out = torch.empty_like(inp), torch.empty_like(inp)
        if not self._warming_up("all_reduce_add_rms_norm"):
            self._all_reduce_add_rms_norm(
                out, residual_out, inp, residual, weight, eps, options
            )
        return out, residual_out

    def should_allreduce_add_attn_res_rms_norm(
        self, inp: torch.Tensor, options: Options | None = None
    ) -> bool:
        """As `should_allreduce_rms_norm`, for all-reduce then Kimi-K3's AttnRes."""
        return isinstance(self.plan(AttnResArgs(inp), options), Plan)

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
        options: Options | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`attn_res(prefix, all_reduce(inp), blocks, ...)` in one kernel, or with no
        `prefix` the sum starting one. Returns the prefix (updated in place when given)
        and the AttnRes output. `write_idx` >= 0 also stores the prefix as that
        block."""
        options = self._require(AttnResArgs(inp), options)
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
            options,
        )
        return prefix_out, out

    def should_allreduce_rms_norm_gemm(
        self,
        inp: torch.Tensor,
        gemm_weight: torch.Tensor,
        options: Options | None = None,
    ) -> bool:
        """As `should_allreduce_rms_norm`, for all-reduce then RMSNorm then a GEMM
        written into an output; `gemm_weight` is [N, hidden], and its N shapes the
        launch."""
        args = GemmTailArgs(inp, gemm_weight, add=False)
        return isinstance(self.plan(args, options), Plan)

    def all_reduce_rms_norm_gemm(
        self,
        inp: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        gemm_weight: torch.Tensor,
        out: torch.Tensor,
        out_col0: int,
        options: Options | None = None,
    ) -> None:
        """`out[:, out_col0:out_col0 + N] = rms_norm(all_reduce(inp), norm_weight, eps)
        @ gemm_weight.T` in one kernel, `gemm_weight` being [N, hidden]."""
        options = self._require(GemmTailArgs(inp, gemm_weight, add=False), options)
        if self._warming_up("all_reduce_rms_norm_gemm"):
            return
        self._all_reduce_rms_norm_gemm(
            inp, norm_weight, eps, gemm_weight, out, out_col0, options
        )

    def should_allreduce_rms_norm_gemm_add(
        self,
        inp: torch.Tensor,
        gemm_weight: torch.Tensor,
        options: Options | None = None,
    ) -> bool:
        """As `should_allreduce_rms_norm`, for all-reduce then RMSNorm then a GEMM added
        into an output; `gemm_weight` is [N, hidden], and its N shapes the launch."""
        args = GemmTailArgs(inp, gemm_weight, add=True)
        return isinstance(self.plan(args, options), Plan)

    def all_reduce_rms_norm_gemm_add(
        self,
        inp: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        gemm_weight: torch.Tensor,
        out: torch.Tensor,
        out_col0: int,
        options: Options | None = None,
    ) -> None:
        """`out[:, out_col0:out_col0 + N] += rms_norm(all_reduce(inp), norm_weight, eps)
        @ gemm_weight.T` in one kernel, `gemm_weight` being [N, hidden]."""
        options = self._require(GemmTailArgs(inp, gemm_weight, add=True), options)
        if self._warming_up("all_reduce_rms_norm_gemm_add"):
            return
        self._all_reduce_rms_norm_gemm_add(
            inp, norm_weight, eps, gemm_weight, out, out_col0, options
        )

    def should_allreduce_rms_scale_add(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        options: Options | None = None,
    ) -> bool:
        """As `should_allreduce_rms_norm`, for writing `out` [rows, hidden] from `inp`'s
        row [shared | projected | latent], the widths out's, out's and the rest."""
        return isinstance(self.plan(ScaleAddArgs(inp, out), options), Plan)

    def all_reduce_rms_scale_add(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        eps: float,
        options: Options | None = None,
    ) -> None:
        """`s = all_reduce(inp)` split [shared | projected | latent], then `out = shared
        + projected * rsqrt(mean(latent^2) + eps)` in one kernel."""
        options = self._require(ScaleAddArgs(inp, out), options)
        if self._warming_up("all_reduce_rms_scale_add"):
            return
        self._all_reduce_rms_scale_add(inp, out, eps, options)

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

    def _require(self, args: Args, options: Options | None) -> Options:
        """The call's options, resolved, unless this backend refuses `args`: then
        `plan`'s Error, raised."""
        got = self.plan(args, options)
        if isinstance(got, Error):
            raise RuntimeError(self._rejected(args, got))
        return Options() if options is None else options

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

    def _rejected(self, args: Args, err: Error) -> str:
        """A refused call's message: the op, its input, and `plan`'s Error."""
        return (
            f"{type(self).__name__} cannot run {args.op} over "
            f"shape={tuple(args.inp.shape)} dtype={args.inp.dtype}: {err.name}"
        )

    # ---- What a BACKEND supplies. ----

    @abstractmethod
    def _all_reduce(
        self,
        out: torch.Tensor,
        inp: torch.Tensor,
        options: Options,
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
        options: Options,
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
        options: Options,
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
        options: Options,
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
        options: Options,
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
        options: Options,
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
        options: Options,
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} has no fused all-reduce + rms scale + add; "
            f"ask should_allreduce_rms_scale_add first."
        )

    def _plan(self, args: Args, options: Options) -> Plan | Error:
        """A backend's own rules, the options among them: what runs the call `args`, or
        the Error it meets. By default (torch, iris) the build's envelope, so a control
        admits what our kernels do: weak-contiguous, whole packs, a dtype built; and no
        options, having no lossy kernel and no template to force. Ours overrides it with
        its C++'s answer."""
        if options.quant_bits is not None:
            return Error.quantized_not_built
        if (options.template, options.blocks, options.threads) != (None, None, None):
            return Error.no_such_template
        inp = args.inp
        if not _is_weak_contiguous(inp):
            return Error.not_contiguous
        built = build_info()
        if inp.numel() * inp.element_size() % built.pack_bytes != 0:
            return Error.row_not_packs
        if inp.dtype not in built.dtypes:
            return Error.dtype_not_built
        return Plan()

    @abstractmethod
    def _open(self) -> bool:
        """Bring this backend up, its availability checks included. True when it is
        usable; False leaves it disabled, which is not an error."""

    def _on_capture(self) -> AbstractContextManager[None]:
        """What this backend needs around a capture. Nothing, by default."""
        return nullcontext()

    def _on_close(self) -> None:  # noqa: B027 -- an optional hook, not an abstract one
        """What this backend has to release. Nothing, by default -- torch.distributed
        holds nothing
        of ours, and a backend that does drops it here so its destructor runs at a known
        moment."""
