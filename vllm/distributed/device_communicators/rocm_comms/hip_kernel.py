# SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc. All
# rights reserved.
"""Python interface to our HIP collectives, and the only place they are TUNED.

The kernel is `csrc/rocm/rocm_comms.cu`, compiled into vLLM's `_rocm_C` extension with
the rest of the ROCm sources. This module calls the ops it registers; it compiles
nothing and there is no cache to warm.

THE SPLIT. The `.cu` is mechanism and holds no policy; every decision -- which
algorithm, how many blocks, how many threads, how big a buffer -- is `HipConfig`, below.
Same shape as a Triton kernel with `@triton.autotune`: the kernel body has no heuristics
and the meta-parameters are chosen outside it. A `if (size < N)` in the `.cu` would be a
decision nobody can see, and the env var it eventually grows is how you end up with
`VLLM_CUSTOM_ALLREDUCE_ALGO`.

`HipConfig` is hip's alone. The tunables every backend shares are `config.Config`.

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


@dataclass(frozen=True)
class HipConfig:
    """Every arbitrary number this backend has, in one place.

    A SINGLE DEFAULT, deliberately: each field is a guess until there is a measurement,
    and a tuning table invented before the first number is a wrong abstraction held
    confidently. When there are numbers this grows a heuristic keyed on the device and
    the problem; nothing outside this file changes when it does.
    """

    algo: int = ALGO_ONE_SHOT
    # vLLM's tuned value on this hardware, carried over because a measured constant
    # beats an unmeasured one -- their note is that too many SMs contend on the
    # interconnect.
    blocks: int = 16
    threads: int = 512
    # Scratch after the signal block in one allocation, so a two-stage algorithm needs a
    # kernel and neither a new buffer nor a new handshake.
    scratch_bytes: int = 8 << 20
    # Peer-pointer slots: one per CAPTURED LAUNCH over a context's life (a capture
    # always records -- see `capture`) and one per registered eager buffer. vLLM
    # captures a graph per batch size and a collective per layer, so the count is
    # capture_sizes x layers. 131072 slots is 8MB, the size vLLM gives the same array.
    max_buffers: int = 131072


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
        config: HipConfig,
    ) -> None:
        # The kernel's own constants, asked for rather than restated here.
        signal_bytes, peer_ptrs_bytes, _blocks, _ranks, _handle = (
            torch.ops._rocm_C.rocm_comms_sizes()
        )
        self.cpu_group = cpu_group
        self.device = device
        self.config = config
        self.rank = dist.get_rank(cpu_group)
        self.world_size = dist.get_world_size(cpu_group)

        # ONE allocation per rank holds the signal block and the scratch after it, so
        # the two-stage algorithm needs no new buffer or handshake -- only a kernel.
        self._signal = torch.zeros(
            signal_bytes + config.scratch_bytes, dtype=torch.uint8, device=device
        )
        # Device-side array of peer-pointer sets, one slot per registered buffer.
        self._slab = torch.zeros(
            peer_ptrs_bytes * config.max_buffers, dtype=torch.uint8, device=device
        )
        # EVERY input is registered; nothing is ever copied. See `_as_input`. The value
        # is the input's STORAGE, held on purpose -- see `register`.
        self._registered: dict[int, torch.UntypedStorage] = {}

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
        # Say what will actually be launched, once. Otherwise a run tells you the answer
        # was wrong but not what was asked for, and "which config produced this" is the
        # first question every time.
        print(
            f"[hip_comms] rank {self.rank}/{self.world_size} ready: {config}",
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
        """Make `tensor` usable as a collective INPUT, permanently.

        COLLECTIVE AND ORDER-SENSITIVE: every rank must call this for its own tensor, in
        the same order. It rests on the assumption `flush_pending` already states --
        every rank runs the same collectives on the same shapes -- since that is what
        makes the first-touch miss in `_as_input` happen on all of them at once. A rank
        that misses when its peers hit HANGS in the gather below rather than failing:
        the pointers compared there are local, and only the identical call sequence
        makes the comparison agree. That is this path's standing hazard.

        IT HOLDS THE STORAGE. A registration hands peers a raw address, so if the
        caching allocator were free to recycle that block the peers would read whatever
        landed there next -- silent corruption. Keeping a reference makes the block
        un-recyclable, which is the invalidation problem answered by not having one.
        The cost is retention, and `HipConfig.max_buffers` bounds it.
        """
        ptr = tensor.data_ptr()
        if len(self._registered) >= self.config.max_buffers:
            raise RuntimeError(
                f"hip_comms: {len(self._registered)} registered buffers hits the "
                f"{self.config.max_buffers} limit. Every collective input is "
                f"registered and held, so the caller allocates fresh buffers per step "
                f"rather than reusing them; raise max_buffers or reuse."
            )
        # ONE gather, carrying the handle and the signature together: it is the
        # rendezvous this path pays for, and it also proves the ranks agree on WHAT they
        # are registering. It cannot prove they agree on WHETHER to -- that disagreement
        # is this same gather, hanging.
        mine = torch.ops._rocm_C.rocm_comms_handle_and_offset(ptr)
        signature = (tensor.numel() * tensor.element_size(), str(tensor.dtype))
        gathered = _all_gather_object(self.cpu_group, (mine, signature))
        seen = {sig for _, sig in gathered}
        if len(seen) != 1:
            raise RuntimeError(
                f"hip_comms: ranks registered different tensors ({sorted(seen)}); "
                f"every rank must register in the same order with the same shapes."
            )
        torch.ops._rocm_C.rocm_comms_register_buffer(
            self._comms,
            [h for (h, _), _ in gathered],
            [o for (_, o), _ in gathered],
            ptr,
        )
        self._registered[ptr] = tensor.untyped_storage()

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
        """The tensor the kernel reads as an input: ALWAYS `inp` itself.

        REGISTRATION IS THE ONLY MEMORY PATH; there is no staging buffer and no copy, at
        any size. Staging cost a full device copy of the message on every collective and
        never got cheaper, because the copy WAS the mechanism; a registration costs one
        CPU rendezvous the first time a buffer is seen and nothing afterwards, and vLLM
        reuses its activation buffers every step. That is also what lifts the old 8 MiB
        bound -- it was the staging buffer's, never the kernel's.

        CAPTURING is the deferred case: the address is not valid yet, so the kernel
        layer records it and `capture()` registers the batch on exit.
        """
        if inp.data_ptr() in self._registered:
            return inp
        if torch.cuda.is_current_stream_capturing():
            return inp
        self.register(inp)
        return inp

    def all_reduce(self, out: torch.Tensor, inp: torch.Tensor) -> None:
        """Sum `inp` across every rank into `out`, in place."""
        cfg = self.config
        torch.ops._rocm_C.rocm_comms_all_reduce(
            self._comms, out, self._as_input(inp), cfg.algo, cfg.blocks, cfg.threads
        )

    def all_gather(self, inp: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Concatenate every rank's `inp` along `dim`, rank-ordered.

        The kernel fills a RANK-MAJOR `(world, *inp.shape)` buffer and the axis is moved
        here, which is what the torch path did -- so the two agree bit for bit and the
        reference in the correctness suite covers both. The `movedim`+`reshape` copy is
        a known cost and a later perf item, not a correctness one.
        """
        cfg = self.config
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
