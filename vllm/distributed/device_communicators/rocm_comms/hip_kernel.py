# SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc. All
# rights reserved.
"""Python interface to our HIP collectives, and the only place they are TUNED.

The kernel is `csrc/rocm/rocm_comms.cu`, compiled into vLLM's `_rocm_C` extension with
the rest of the ROCm sources. This module calls the ops it registers; it compiles
nothing and there is no cache to warm.

THE SPLIT. The `.cu` is mechanism and holds no policy; every decision -- which
algorithm, how many blocks, how many threads -- is made here, by `config_for`, and
passed down. Same shape as a Triton kernel with `@triton.autotune`: the kernel body has
no heuristics and the meta-parameters are chosen outside it. Two consequences worth
knowing:

- A tuning decision is a pure Python function, so it is testable without a GPU. A `if
(size < N)` in the `.cu` would be a decision nobody can see, and the env var it
eventually grows is how you end up with `VLLM_CUSTOM_ALLREDUCE_ALGO`.
- It costs nothing where it matters. vLLM captures cudagraphs, so `config_for` runs ONCE
during capture and replay is pure kernel launch with no Python at all.

`ngpus` and the dtype have to be compile-time to unroll and vectorize, so they select a
template instantiation rather than being passed; the `.cu`'s dispatch names every
combination that exists and REFUSES anything else rather than substituting.

THE CONTEXT IS AN OPAQUE HANDLE. A torch op is a free function over schema types, so
the C++ object crosses as an `int` -- the same shape vLLM's custom all-reduce uses. The
consequence is that nothing frees it for us: `close()` has to run, and dropping the last
Python reference does not.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

# Algorithms, matching the `.cu`'s dispatch. One today.
ALGO_ONE_SHOT = 0


# =================================================================================
# TUNING. The whole tuning surface is this dataclass and the function under it.
# =================================================================================


@dataclass(frozen=True)
class LaunchConfig:
    """How to run one collective. Pure data, chosen in Python, passed to the kernel."""

    algo: int = ALGO_ONE_SHOT
    blocks: int = 16
    threads: int = 512


# A single default, deliberately. Every field here is a guess until we have a
# measurement, and a tuning table invented before the first number is a wrong
# abstraction held confidently. `blocks=16` is vLLM's tuned value on the same hardware,
# carried over on the grounds that a measured constant beats an unmeasured one -- their
# comment is that too many SMs contend on the interconnect. When we have numbers this
# becomes a lookup keyed by (op, dtype, numel, world_size); nothing outside this
# function needs to change.
_DEFAULT = LaunchConfig()


def config_for(
    op: str, dtype: torch.dtype, numel: int, world_size: int
) -> LaunchConfig:
    """The tuning decision, and the ONLY place one is made."""
    return _DEFAULT


# =================================================================================
# CONTEXT. Peer memory and registration, one per process group.
# =================================================================================


def _all_gather_object(group: ProcessGroup, obj: Any) -> list[Any]:
    out: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(out, obj, group=group)
    return out


class HipComms:
    """The peer-memory context: IPC handshake, signal block, scratch, registration.

    ONE per process group, like vLLM's `CustomAllreduce`, because peer pointers are
    group-scoped. It knows nothing about which collective runs over it, which is the
    property that keeps a new workload from touching it.

    The handle EXCHANGE happens here rather than in C++: it is a collective over the
    gloo `cpu_group`, and a process group is not something the kernel layer should know
    about. C++ only opens the handles it is handed.
    """

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device,
        *,
        scratch_bytes: int = 8 << 20,
        # One slot per CAPTURED LAUNCH over this object's life, not per distinct
        # address: a capture always records (see `slot_for`). vLLM captures one graph
        # per batch size and a collective per layer, so the count is capture_sizes x
        # layers -- thousands. 131072 slots is 8MB, the size vLLM gives the same array.
        max_buffers: int = 131072,
        max_size: int = 8 << 20,
    ) -> None:
        # The kernel's own constants, asked for rather than restated here.
        signal_bytes, peer_ptrs_bytes, _blocks, _ranks, _handle = (
            torch.ops._rocm_C.rocm_comms_sizes()
        )
        self.cpu_group = cpu_group
        self.device = device
        self.rank = dist.get_rank(cpu_group)
        self.world_size = dist.get_world_size(cpu_group)

        # ONE allocation per rank holds the signal block and the scratch after it, so
        # the two-stage algorithm needs no new buffer or handshake -- only a kernel.
        self._signal = torch.zeros(
            signal_bytes + scratch_bytes, dtype=torch.uint8, device=device
        )
        # Device-side array of peer-pointer sets, one slot per registered buffer.
        self._slab = torch.zeros(
            peer_ptrs_bytes * max_buffers, dtype=torch.uint8, device=device
        )
        # A pre-registered buffer for the EAGER path. The kernel reads peer pointers, so
        # an input has to be registered -- and eagerly the caller hands us whatever the
        # allocator gave it. Staging into this buffer is the copy the capture path
        # exists to avoid, and that asymmetry is the point: the copy lives ONLY on the
        # path nobody measures. vLLM's CustomAllreduce does the same thing for the same
        # reason.
        self._staging = torch.zeros(max_size, dtype=torch.uint8, device=device)
        self._registered: set = set()

        self.max_size = max_size
        handles, offsets = self._exchange(self._signal.data_ptr())
        self._comms = torch.ops._rocm_C.rocm_comms_init(
            self.rank,
            self.world_size,
            self._signal.data_ptr(),
            handles,
            offsets,
            self._slab.data_ptr(),
            self._slab.numel(),
        )
        self.register(self._staging)
        # Say what will actually be launched, once. Otherwise a run tells you the answer
        # was wrong but not what was asked for, and "which config produced this" is the
        # first question every time.
        cfg = config_for("all_reduce", torch.bfloat16, 0, self.world_size)
        print(
            f"[hip_comms] rank {self.rank}/{self.world_size} ready: algo={cfg.algo} "
            f"blocks={cfg.blocks} threads={cfg.threads} "
            f"staging={self._staging.numel()}B slots={max_buffers}",
            flush=True,
        )

    def _exchange(self, ptr: int) -> tuple[list[list[int]], list[int]]:
        """Every rank's IPC handle + offset for its own `ptr`, in rank order. A handle
        is a list of byte values: an op schema has no bytes type, which is how vLLM's
        other all-reduces carry theirs too."""
        mine = torch.ops._rocm_C.rocm_comms_handle_and_offset(ptr)
        gathered = _all_gather_object(self.cpu_group, mine)
        return [h for h, _ in gathered], [o for _, o in gathered]

    def register(self, tensor: torch.Tensor) -> None:
        """Make `tensor` usable as a collective INPUT. Collective: every rank must call
        it for its own tensor, in the same order."""
        handles, offsets = self._exchange(tensor.data_ptr())
        torch.ops._rocm_C.rocm_comms_register_buffer(
            self._comms, handles, offsets, tensor.data_ptr()
        )
        self._registered.add(tensor.data_ptr())

    @contextmanager
    def capture(self) -> Iterator[None]:
        """Wrap a cudagraph capture. Buffers used inside are registered on EXIT.

        While capturing, an input's address is not registered yet, so the kernel layer
        reserves a slot and records the pointer; here we exchange handles for everything
        recorded and fill those slots in. Sound because a captured address is fixed for
        the graph's life -- the replayed kernel reads a peer-pointer set populated after
        the capture that recorded its launch.
        """
        try:
            yield
        finally:
            # FINALLY, so a capture that raises still registers what it recorded.
            # Without it the slots stay null and the next launch faults on a null peer
            # pointer -- the original error is then buried under a GPU memory fault that
            # names nothing.
            self.flush_pending()

    def flush_pending(self) -> None:
        """Register whatever the capture deferred.

        ALWAYS one collective, even with nothing pending -- a rank that returned early
        here would leave the others waiting in the gather. Nothing is registered when
        nothing is pending; the exchange still has to happen."""
        pending: Sequence[int] = torch.ops._rocm_C.rocm_comms_pending_graph_buffers(
            self._comms
        )
        # ONE collective for ALL of them, not one each. A capture records a buffer per
        # collective in the graph -- a layer each, in vLLM -- so per-buffer exchanges
        # are how a graph-heavy startup becomes thousands of round trips.
        mine = [torch.ops._rocm_C.rocm_comms_handle_and_offset(p) for p in pending]
        gathered: list[list[tuple[list[int], int]]] = _all_gather_object(
            self.cpu_group, mine
        )
        # Every rank must agree on the count. The same gather that carries the handles
        # proves it, and it has to be proven: the transpose below reads slot `i` from
        # every rank, so a short list there is a peer set silently missing a rank.
        counts = [len(g) for g in gathered]
        if len(set(counts)) != 1:
            raise RuntimeError(
                f"hip_comms: ranks captured different numbers of buffers ({counts}); "
                "every rank must run the same graph."
            )
        if not pending:
            return
        # ONE ENTRY PER BUFFER, the world's handles laid end to end: the op splits them
        # back by handle size, because a schema nests two deep and this needs three.
        torch.ops._rocm_C.rocm_comms_register_graph_buffers(
            self._comms,
            [[b for g in gathered for b in g[i][0]] for i in range(len(pending))],
            [[g[i][1] for g in gathered] for i in range(len(pending))],
        )

    def _as_input(self, inp: torch.Tensor) -> torch.Tensor:
        """The tensor the kernel may read as an input: `inp` when it is registered or we
        are capturing (registration is deferred then), else a staged copy."""
        if inp.data_ptr() in self._registered:
            return inp
        if torch.cuda.is_current_stream_capturing():
            return inp
        nbytes = inp.numel() * inp.element_size()
        if nbytes > self._staging.numel():
            raise RuntimeError(
                f"hip_comms: {nbytes} bytes exceeds the {self._staging.numel()}-byte "
                f"staging buffer. Register the tensor, or raise max_size."
            )
        staged = self._staging[:nbytes].view(inp.dtype).view_as(inp)
        staged.copy_(inp)
        return staged

    def all_reduce(
        self, out: torch.Tensor, inp: torch.Tensor, cfg: LaunchConfig | None = None
    ) -> None:
        """Sum `inp` across every rank into `out`, in place."""
        if cfg is None:
            cfg = config_for("all_reduce", inp.dtype, inp.numel(), self.world_size)
        torch.ops._rocm_C.rocm_comms_all_reduce(
            self._comms, out, self._as_input(inp), cfg.algo, cfg.blocks, cfg.threads
        )

    def all_gather(
        self, inp: torch.Tensor, dim: int = -1, cfg: LaunchConfig | None = None
    ) -> torch.Tensor:
        """Concatenate every rank's `inp` along `dim`, rank-ordered.

        The kernel fills a RANK-MAJOR `(world, *inp.shape)` buffer and the axis is moved
        here, which is what the torch path did -- so the two agree bit for bit and the
        reference in the correctness suite covers both. The `movedim`+`reshape` copy is
        a known cost and a later perf item, not a correctness one.
        """
        if cfg is None:
            cfg = config_for("all_gather", inp.dtype, inp.numel(), self.world_size)
        if dim < 0:
            dim += inp.dim()
        shape = tuple(inp.size())
        staged = torch.empty(
            (self.world_size,) + shape, dtype=inp.dtype, device=inp.device
        )
        torch.ops._rocm_C.rocm_comms_all_gather(
            self._comms, staged, self._as_input(inp), cfg.algo, cfg.blocks, cfg.threads
        )
        return staged.movedim(0, dim).reshape(
            shape[:dim] + (self.world_size * shape[dim],) + shape[dim + 1 :]
        )

    def close(self) -> None:
        """Release the context, NOW. Idempotent.

        THE HANDLE IS AN `int`, so nothing collects it: dropping this object frees the
        tensors and leaves the C++ side holding every peer handle it opened. The
        communicator above calls this from its own `close`, which vLLM's teardown calls.
        """
        if getattr(self, "_comms", None) is None:
            return
        torch.ops._rocm_C.rocm_comms_dispose(self._comms)
        self._comms = None
