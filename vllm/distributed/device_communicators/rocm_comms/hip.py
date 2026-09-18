# SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""The HIP backend: our own kernel in `_rocm_C`, and the peer memory it runs over.

The kernel is `csrc/rocm/rocm_comms.cu`, compiled into vLLM's `_rocm_C` extension with
the rest of the ROCm sources. This module calls the ops it registers; it compiles
nothing and there is no cache to warm.

ONE CLASS, like every other backend here. The peer memory -- the IPC handshake, the
signal block, the registrations -- was a `HipCommsImpl` the communicator delegated to,
which bought nothing once `base` took over construction: the outer class was four
forwarding methods, each opening with an assert that the inner one existed, and the two
carried four same-named methods with different contracts (`all_reduce(out, inp)` against
`all_reduce(inp)`). iris and torch each hold their own machinery in one class.

THE SPLIT THAT DOES MATTER is between this file and the `.cu`. The `.cu` is mechanism
and holds no policy; every decision -- which algorithm, how many blocks, how many
threads, how big a buffer -- is `HipTunables`. Same shape as a Triton kernel with
`@triton.autotune`: the kernel body has no heuristics and the meta-parameters are chosen
outside it. A `if (size < N)` in the `.cu` would be a decision nobody can see, and the
env var it eventually grows is how you end up with `VLLM_CUSTOM_ALLREDUCE_ALGO`.

`HipTunables` is hip's alone. What every backend shares is `tunables.Tunables`.

TWO MEMORY PATHS, AND THE SPLIT IS LIFETIME, not size. A CAPTURED buffer is vLLM's and
it holds it for the graph's life, so it is registered once at capture exit and read in
place. An EAGER input is the caching allocator's, borrowed for the call, so it is copied
into a staging buffer we own. Registering an eager input instead means holding it, and
holding one 2 GiB activation per layer OOM'd the 70B profile run (2026-09-17); not
holding it means peers read recycled memory. `CustomAllreduce` splits the same way for
the same reason. See `_as_input`.

`ngpus` and the dtype have to be compile-time to unroll and vectorize, so they select a
template instantiation rather than being passed; the `.cu`'s dispatch names every
combination that exists and REFUSES anything else rather than substituting.

THE ONE THING IT READS FROM VLLM is the current config, to size that staging buffer --
see `_staging_bytes`. The rest of the package knows nothing about the program around it.

THE CONTEXT IS AN OPAQUE HANDLE. A torch op is a free function over schema types, so
the C++ object crosses as an `int` -- the same shape vLLM's custom all-reduce uses. The
consequence is that nothing frees it for us: `close()` has to run, and dropping the last
Python reference does not.
"""

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .base import Communicator
from .utils import widest_input_bytes

logger = logging.getLogger(__name__)


# Algorithms, matching the `.cu`'s dispatch. One today.
ALGO_ONE_SHOT = 0


@dataclass(frozen=True)
class HipTunables:
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
    # Peer-pointer slots, one per CAPTURED LAUNCH over a communicator's life (a capture
    # always records). vLLM captures a graph per batch size and a collective per layer,
    # so the count is capture_sizes x layers. 131072 slots is 8MB, the size vLLM gives
    # the same array.
    max_buffers: int = 131072
    # A FLOOR on the eager staging buffer, not its size: `_staging_bytes` derives that
    # from the workload. It matters only when there is no vLLM config to derive from --
    # the correctness suite, say -- where the shapes are the caller's own business and
    # 128 MiB covers anything a test has reason to try.
    staging_floor_bytes: int = 128 << 20


# =================================================================================
# CONTEXT. Peer memory and registration, one per process group.
# =================================================================================


def _all_gather_object(group: ProcessGroup, obj: Any) -> list[Any]:
    out: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(out, obj, group=group)
    return out


def _staging_bytes(floor: int) -> int:
    """How big the eager staging buffer has to be: ONE all-reduce input at the largest
    batch vLLM will build.

    FROM THE WORKLOAD, NOT FROM A CONSTANT. QuickReduce takes the other route and
    allocates `INT32_MAX + 1` because its indices are 32-bit -- a ceiling that is right
    by accident. `widest_input_bytes` is the same number for this model and says why.

    NO CONFIG IS NOT AN ERROR: this package is usable without vLLM around it, and the
    floor is what it gets then.
    """
    return max(floor, widest_input_bytes())


class HipCommunicator(Communicator):
    """Communicator over HIP collectives we own, and the peer memory they run on.

    ONE per process group, like vLLM's `CustomAllreduce`, because peer pointers are
    group-scoped.

    The handle EXCHANGE happens in Python rather than in C++: it is a collective over
    the gloo `cpu_group`, and a process group is not something the kernel layer should
    know about. C++ only opens the handles it is handed.

    It does NOT self-disable when the ops are missing: that means a build without them,
    and falling back quietly would let vLLM use its own all-reduce and report the run
    READY.
    """

    hip_tunables: HipTunables = HipTunables()

    # Set in `_open`, which runs only once the shared gates pass. Declared here so a
    # DISABLED communicator is still a safe object to hold and close.
    _handle: int | None = None
    # POINTERS ONLY. It held each input's storage for a while, to stop the caching
    # allocator recycling a block a peer points into -- which pinned one 2 GiB
    # activation per layer and OOM'd the 70B profile run (2026-09-17). Nothing here
    # needs pinning now: the staging buffer is ours and we hold it, and a captured
    # buffer is held by vLLM for the graph's life.
    _registered: set[int]
    _staging: torch.Tensor

    def _open(self) -> bool:
        """Open the peer memory. A COLLECTIVE -- it all-gathers IPC handles -- so every
        rank must reach it.

        EAGER, and after the shared gates: doing this inside vLLM's cudagraph capture is
        not recoverable, and a box that cannot run this backend should not pay for it.
        The gates are uniform across a TP group in practice (same arch, same world
        size), but if they ever were not, the ranks that got here would HANG waiting for
        the ones that returned rather than failing. Worth knowing: a deadlock is far
        worse than an error.
        """
        tunables = self.hip_tunables
        # The kernel's own constants, asked for rather than restated here.
        signal_bytes, peer_ptrs_bytes, _blocks, _ranks, _handle_bytes = (
            torch.ops._rocm_C.rocm_comms_sizes()
        )
        self.rank = dist.get_rank(self.cpu_group)
        # ONE allocation per rank holds the signal block and the scratch after it, so
        # the two-stage algorithm needs no new buffer or handshake -- only a kernel.
        self._signal = torch.zeros(
            signal_bytes + tunables.scratch_bytes, dtype=torch.uint8, device=self.device
        )
        # Device-side array of peer-pointer sets, one slot per registered buffer.
        self._slab = torch.zeros(
            peer_ptrs_bytes * tunables.max_buffers,
            dtype=torch.uint8,
            device=self.device,
        )
        self._registered = set()
        # THE EAGER PATH'S PEER-VISIBLE MEMORY, allocated ONCE, here, before vLLM
        # profiles: a buffer that appeared later would change the memory the profile run
        # measures, and one that grew during a capture would be worse than that.
        self._staging = torch.zeros(
            _staging_bytes(tunables.staging_floor_bytes),
            dtype=torch.uint8,
            device=self.device,
        )

        handles, offsets = self._exchange(self._signal.data_ptr())
        self._handle = torch.ops._rocm_C.rocm_comms_init(
            self.rank,
            self.world_size,
            self._signal.data_ptr(),
            handles,
            offsets,
            self._slab.data_ptr(),
            self._slab.numel(),
        )
        # ONE COLLECTIVE, AT STARTUP, with every rank here in the same order. That is
        # the whole of eager registration now, which is why nothing in this class has to
        # reason about ranks disagreeing about whether to register.
        self._register(self._staging)
        # Say what will actually be launched, once. Otherwise a run tells you the answer
        # was wrong but not what was asked for, and "what was it tuned to" is the first
        # question every time.
        logger.info(
            "HipCommunicator ready: rank %d/%d, small_limit=%dMB, staging=%dMB, %s",
            self.rank,
            self.world_size,
            self.tunables.small_limit >> 20,
            self._staging.numel() >> 20,
            tunables,
        )
        return True

    # No admission of its own. A two-stage reduce-scatter will need the count to divide
    # the ranks; the shipped kernel is one-shot and does not, and the baseline does not
    # check it either, so adding it would refuse tensors both we and the path we replace
    # can handle.

    def _exchange(self, ptr: int) -> tuple[list[list[int]], list[int]]:
        """Every rank's IPC handle + offset for its own `ptr`, in rank order. A handle
        is a list of byte values: an op schema has no bytes type, which is how vLLM's
        other all-reduces carry theirs too."""
        mine = torch.ops._rocm_C.rocm_comms_handle_and_offset(ptr)
        gathered = _all_gather_object(self.cpu_group, mine)
        return [h for h, _ in gathered], [o for _, o in gathered]

    def _register(self, tensor: torch.Tensor) -> None:
        """Make `tensor` usable as a collective INPUT, permanently.

        COLLECTIVE. Called from exactly two places and NEVER from a collective: `_open`,
        for the staging buffer, and `_flush_pending`, for what a capture recorded. Both
        are reached by every rank in the same order by construction -- one is startup,
        the other is a graph every rank captures -- so there is no rule here about ranks
        agreeing, because there is no decision for them to disagree about. It used to be
        called per eager call on a pointer test that was local to each rank, which could
        have hung; that is gone with staging's return.

        IT DOES NOT HOLD THE TENSOR. Whoever owns the buffer keeps it alive: we hold the
        staging buffer ourselves, and vLLM holds a captured graph's buffers for the
        graph's life. Holding it here instead is what OOM'd the 70B profile run.
        """
        ptr = tensor.data_ptr()
        if len(self._registered) >= self.hip_tunables.max_buffers:
            raise RuntimeError(
                f"hip_comms: {len(self._registered)} registered buffers hits the "
                f"{self.hip_tunables.max_buffers} limit; raise max_buffers."
            )
        # ONE gather, carrying the handle and the signature together: it is the
        # rendezvous this pays for, and it also proves the ranks agree on what they are
        # registering.
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
            self._handle,
            [h for (h, _), _ in gathered],
            [o for (_, o), _ in gathered],
            ptr,
        )
        self._registered.add(ptr)

    @contextmanager
    def _on_capture(self) -> Iterator[None]:
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
            self._flush_pending()

    def _flush_pending(self) -> None:
        """Register whatever the capture deferred.

        ALWAYS one collective, even with nothing pending -- a rank that returned early
        here would leave the others waiting in the gather. Nothing is registered when
        nothing is pending; the exchange still has to happen."""
        pending: Sequence[int] = torch.ops._rocm_C.rocm_comms_pending_graph_buffers(
            self._handle
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
            self._handle,
            [[b for g in gathered for b in g[i][0]] for i in range(len(pending))],
            [[g[i][1] for g in gathered] for i in range(len(pending))],
        )

    def _as_input(self, inp: torch.Tensor) -> torch.Tensor:
        """An address the KERNEL may read as an input. `inp` itself, or a copy of it in
        memory the peers can already see.

        THE TWO PATHS ARE THE TWO LIFETIMES, and that is the whole of the distinction:
        whether anyone guarantees this address outlives the collective. A captured
        buffer is vLLM's and it holds it for the graph's life, so registering costs one
        rendezvous at capture exit and nothing per call. An eager input is the caching
        allocator's, borrowed for the duration of the call, so registering it would mean
        either holding it (a 2 GiB leak per layer) or letting peers read recycled memory
        (silent corruption). It is copied instead.
        """
        return inp if self._visible_to_peers(inp) else self._staged(inp)

    def _visible_to_peers(self, inp: torch.Tensor) -> bool:
        """Whether the peers can read `inp` BY THE TIME THIS LAUNCH RUNS, which is the
        only moment that matters and is not always now.

        Registered is visible now. CAPTURING is the other case: nothing runs while a
        graph is recorded, the kernel layer reserves a slot and notes the address, and
        `_flush_pending` registers it on the way out -- before any replay.
        """
        return (
            inp.data_ptr() in self._registered
            or torch.cuda.is_current_stream_capturing()
        )

    def _staged(self, inp: torch.Tensor) -> torch.Tensor:
        """`inp` copied into the staging buffer, which IS registered.

        THE COPY IS THE EAGER PATH'S WHOLE COST, and it is paid where nothing is
        measured: vLLM serves decode from cudagraphs, so this runs during profiling,
        warmup and any shape outside a capture size. It is also what the incumbent does
        -- `CustomAllreduce` stages eagerly and registers only what a capture records.

        THE BOUND IS THIS BUFFER'S, NOT THE KERNEL'S. The kernel is grid-stride and
        takes any size; `_staging_bytes` is sized so a vLLM workload cannot exceed it,
        and exceeding it is a configuration error rather than something to work around.
        """
        nbytes = inp.numel() * inp.element_size()
        if nbytes > self._staging.numel():
            raise RuntimeError(
                f"hip_comms: an eager {nbytes}-byte collective exceeds the "
                f"{self._staging.numel()}-byte staging buffer, which was sized for the "
                f"widest batch this workload declared. Raise staging_floor_bytes, or "
                f"ask why a collective is larger than max_num_batched_tokens allows."
            )
        staged = self._staging[:nbytes].view(inp.dtype).view_as(inp)
        staged.copy_(inp)
        return staged

    def _all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        """Sum `inp` across the TP ranks, out of place."""
        cfg = self.hip_tunables
        out = torch.empty_like(inp)
        torch.ops._rocm_C.rocm_comms_all_reduce(
            self._handle, out, self._as_input(inp), cfg.algo, cfg.blocks, cfg.threads
        )
        return out

    def _all_gather(self, inp: torch.Tensor, dim: int) -> torch.Tensor:
        """Concatenate every rank's `inp` along `dim`, rank-ordered.

        The kernel fills a RANK-MAJOR `(world, *inp.shape)` buffer and the axis is moved
        here, which is what the torch path did -- so the two agree bit for bit and the
        reference in the correctness suite covers both. The `movedim`+`reshape` copy is
        a known cost and a later perf item, not a correctness one.
        """
        cfg = self.hip_tunables
        inp = inp.contiguous()
        if dim < 0:
            dim += inp.dim()
        shape = tuple(inp.size())
        staged = torch.empty(
            (self.world_size,) + shape, dtype=inp.dtype, device=inp.device
        )
        torch.ops._rocm_C.rocm_comms_all_gather(
            self._handle, staged, self._as_input(inp), cfg.algo, cfg.blocks, cfg.threads
        )
        return staged.movedim(0, dim).reshape(
            shape[:dim] + (self.world_size * shape[dim],) + shape[dim + 1 :]
        )

    def _on_close(self) -> None:
        """Release the peer memory, NOW. Idempotent.

        CALLED, not collected. THE HANDLE IS AN `int`, so dropping this object frees the
        tensors and leaves the C++ side holding every peer handle it opened; this is
        what runs `~Comms()` and its `hipIpcCloseMemHandle` on every peer base. The
        handles are a per-process resource, and a construct/destroy cycle that leaks
        them fails later and elsewhere.
        """
        self.disabled = True
        if self._handle is None:
            return
        torch.ops._rocm_C.rocm_comms_dispose(self._handle)
        self._handle = None
