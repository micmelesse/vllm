# SPDX-License-Identifier: MIT Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All
# rights reserved.

"""The interface every ROCm TP collective backend implements, and the two rules it
enforces once.

One file per backend beside this one: `hip`, `iris`, `torch`. This holds only what they
share -- the abstract surface `CudaCommunicator` calls, and the admission checks that
decide whether a tensor is one of ours at all.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext

import torch

from .tunables import Tunables

logger = logging.getLogger(__name__)


def _rocm_arch_available() -> bool:
    try:
        props = torch.cuda.get_device_properties(0)
        gcn_arch = getattr(props, "gcnArchName", "")
        return any(gfx in gcn_arch for gfx in ["gfx94", "gfx95"])
    except Exception:
        return False


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

    # The admission envelope, shared by EVERY backend including torch. Uniform on
    # purpose: torch is the control, so a control that admits a superset is comparing
    # against a different question -- a shape outside the envelope would run on torch
    # and fall back on the others.
    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

    # Class attributes, so a backend needs no cooperating `__init__` to get either
    # invariant.
    _capturing: bool = False
    _closed: bool = False

    # WHAT THE BASE OWNS, and therefore what a backend may not override -- checked
    # when the class is DEFINED. `_is_supported` is private and still belongs here: a
    # backend redefining it would change what the shared envelope means.
    _OWNED = (
        "should_allreduce",
        "should_allgather",
        "all_reduce",
        "all_gather",
        "capture",
        "_is_supported",
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
                f"Supply `_all_reduce`, `_all_gather`, `_on_capture` or "
                f"`_on_close` instead -- what the kernel supports and how long it "
                f"lives are not a backend's to redefine."
            )

    # ---- What the CALLER uses. Concrete: this class owns the order. ----

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        """Whether this backend takes `inp` -- every SIZE, because it owns the
        collective. The only no is a tensor the kernel cannot compile for, or being
        disabled, and a no sends the caller elsewhere, so this stays public."""
        return not self.disabled and self._is_supported(inp)

    def should_allgather(self, inp: torch.Tensor) -> bool:
        """Whether this backend takes `inp`. Every SIZE too; same two noes."""
        return not self.disabled and self._is_supported(inp)

    def _is_supported(self, inp: torch.Tensor) -> bool:
        """CAN THIS KERNEL TAKE THIS TENSOR AT ALL -- its shape and its dtype.

        Transcribed from `custom_all_reduce.should_custom_ar`, the path these backends
        replace: a 16-byte multiple and weak-contiguous. DTYPE is the one addition --
        our kernels are instantiated for fp16 and bf16 only.

        SIZE IS NOT HERE; it is `_is_small`. Splitting them is what lets an all-reduce
        of ANY size be ours while a dtype we cannot compile for is still refused.
        """
        nbytes = inp.numel() * inp.element_size()
        return (
            _is_weak_contiguous(inp)
            and nbytes % 16 == 0
            and inp.dtype in self._SUPPORTED_DTYPES
        )

    def _is_small(self, inp: torch.Tensor) -> bool:
        """Whether `inp` is under `small_limit` -- the line that used to pick between
        this backend and QuickReduce, and now picks between this backend's OWN paths.
        Nothing calls it to refuse work; it is the switch for when a second kernel
        exists.
        """
        return inp.numel() * inp.element_size() < self.tunables.small_limit

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        """EVERY all-reduce this backend's kernel can compile for, at any SIZE.

        The size used to decide whether the caller kept it or handed it to QuickReduce,
        so an arm named for a backend was that backend under the limit and something
        else above it. Now the collective is ours and `_is_small` picks which of OUR
        paths it takes.

        BOTH ARMS RUN THE SAME KERNEL TODAY, and the branch is here anyway: it is where
        the two paths part, and a split that exists only in a comment is one the next
        person has to rediscover.
        """
        self._check_capture("all_reduce")
        if not self.should_allreduce(inp):
            raise RuntimeError(self._rejected("all_reduce", inp))
        if self._is_small(inp):
            return self._all_reduce(inp)
        else:
            # OVER `small_limit`. Used to be QuickReduce's; ours now, and its own kernel
            # when there is one.
            return self._all_reduce(inp)

    def all_gather(self, inp: torch.Tensor, dim: int = -1) -> torch.Tensor:
        self._check_capture("all_gather")
        if not self.should_allgather(inp):
            raise RuntimeError(self._rejected("all_gather", inp))
        return self._all_gather(inp, dim)

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

    def _check_capture(self, op: str) -> None:
        """Refuse a collective recorded into a graph outside `capture()`: the stream
        says it is capturing and this object says nobody entered the context, so the
        caller is wrong."""
        if torch.cuda.is_current_stream_capturing() and not self._capturing:
            raise RuntimeError(
                f"{type(self).__name__}.{op} is being captured into a cudagraph "
                f"without `capture()`. Use `with comm.capture(), torch.cuda.graph(g): "
                f"...` -- a backend may defer peer registration until that context "
                f"exits, and a graph captured without it "
                f"replays against addresses that were never registered."
            )

    def _rejected(self, op: str, inp: torch.Tensor) -> str:
        return (
            f"{type(self).__name__} rejected {op}: shape={tuple(inp.shape)} "
            f"dtype={inp.dtype} disabled={self.disabled}. Ask "
            f"should_{op.replace('_', '')} first and fall back "
            f"when it says no."
        )

    # ---- What a BACKEND supplies. ----

    @abstractmethod
    def _all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        """SUM across ranks, out of place: input untouched, new tensor returned. Assume
        `inp` is admitted -- the base checked."""

    @abstractmethod
    def _all_gather(self, inp: torch.Tensor, dim: int) -> torch.Tensor:
        """The per-rank inputs concatenated along `dim`, rank-ordered."""

    def _on_capture(self) -> AbstractContextManager[None]:
        """What this backend needs around a capture. Nothing, by default."""
        return nullcontext()

    def _on_close(self) -> None:  # noqa: B027 -- an optional hook, not an abstract one
        """What this backend has to release. Nothing, by default -- torch.distributed
        holds nothing
        of ours, and a backend that does drops it here so its destructor runs at a known
        moment."""
