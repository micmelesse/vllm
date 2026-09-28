# SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""The HIP backend: our kernels in `_rocm_C` (`csrc/rocm/rocm_comms.cu`) and the peer
memory they run over.

The caller names an op and C++ picks the kernel and its launch geometry
(`csrc/rocm/rocm_comms/launch.cuh`); `set_launch_override` is the one way to force one,
for the sweep and the tests.

TWO MEMORY PATHS, split by lifetime. A captured buffer is held by vLLM for the graph's
life, so it is registered once at capture exit and read in place. An eager input is
the caching allocator's, borrowed for the call, so it is copied into a staging buffer
we own. See `_as_input`.

The C++ context crosses as an opaque `int` handle, so nothing frees it for us:
`close()` has to run.
"""

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .base import AdmitOp, Communicator

logger = logging.getLogger(__name__)


# EVERY KERNEL THERE IS, as C++ numbers them (`enum class Kernel` in launch.cuh), named
# by shot, direction (pull reads peers, push writes into them) and what it fuses. Named
# only to force one through `set_launch_override`; nothing else picks.
Kernel = Literal[
    "one_shot_pull",
    "one_shot_push",
    "two_shot_pull",
    "two_shot_push",
    "one_shot_pull_rms_norm",
    "one_shot_push_rms_norm",
    "two_shot_pull_rms_norm",
    "two_shot_push_rms_norm",
    "one_shot_pull_add_rms_norm",
    "one_shot_push_add_rms_norm",
    "two_shot_pull_add_rms_norm",
    "two_shot_push_add_rms_norm",
    "one_shot_pull_add_attn_res_rms_norm",
    "one_shot_push_add_attn_res_rms_norm",
    "two_shot_pull_add_attn_res_rms_norm",
    "two_shot_push_add_attn_res_rms_norm",
    "one_shot_pull_rms_norm_gemm_add",
    "one_shot_push_rms_norm_gemm_add",
    "two_shot_pull_rms_norm_gemm_add",
    "two_shot_push_rms_norm_gemm_add",
]
_KERNEL_WIRE: Mapping[Kernel, int] = {
    k: i
    for i, k in enumerate(Kernel.__args__)  # type: ignore[attr-defined]
}

# The ops as C++ numbers them (`enum class Op`).
_OP_WIRE: Mapping[AdmitOp, int] = {
    "all_reduce": 0,
    "rms_norm": 1,
    "add_rms_norm": 2,
    "add_attn_res_rms_norm": 3,
    "rms_norm_gemm_add": 4,
}


@dataclass(frozen=True)
class HipTunables:
    """Every arbitrary number this backend has. One default each until measured."""

    # Two-shot's scratch, after the signal block, per rank. It holds one rank's slice,
    # so it caps a two-shot buffer at `scratch_bytes` x ngpus (1.07 GB at 8 ranks); the
    # INT8 two-shot holds every rank's slice at half width, padded to whole grid
    # strides (Kimi-K3's 4096 x 7168 bf16 prefill needs 74 MB at 36 blocks).
    scratch_bytes: int = 128 << 20
    # Peer-pointer slots, one per captured launch: capture_sizes x layers. 8 MB, the
    # size vLLM gives the same array.
    max_buffers: int = 131072
    # A floor on the eager staging buffer, for when there is no vLLM config to size it
    # from (the correctness suite).
    staging_floor_bytes: int = 128 << 20
    # How long a kernel waits on a peer before it prints where it was and traps.
    sync_timeout_s: float = 10.0


def _all_gather_object(group: ProcessGroup, obj: Any) -> list[Any]:
    out: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(out, obj, group=group)
    return out


class HipCommunicator(Communicator):
    """Communicator over our HIP kernels. One per process group, because peer pointers
    are group-scoped.

    The IPC handle exchange is done here over the gloo `cpu_group`; C++ only opens the
    handles it is handed. Missing ops are an error, not a disable: falling back quietly
    would let vLLM run its own all-reduce and report the arm ready.
    """

    hip_tunables: HipTunables = HipTunables()

    # Declared here so a disabled communicator is still safe to hold and close.
    _handle: int | None = None
    # Pointers only: owners keep the buffers alive (we hold staging, vLLM holds a
    # graph's buffers).
    _registered: set[int]
    _staging: torch.Tensor

    def _staging_bytes(self) -> int:
        """One all-reduce input at the widest batch vLLM will build, or the floor."""
        return max(self.hip_tunables.staging_floor_bytes, self.widest_input_bytes())

    def _open(self) -> bool:
        """Open the peer memory. A collective, so every rank must reach it.

        Eager, and before any capture: doing this inside a cudagraph capture is not
        recoverable.
        """
        tunables = self.hip_tunables
        signal_bytes, peer_ptrs_bytes, _blocks, _ranks, _handle_bytes, _row_packs = (
            torch.ops._rocm_C.rocm_comms_sizes()
        )
        self.rank = dist.get_rank(self.cpu_group)
        # One allocation per rank: the signal block, then the scratch.
        self._signal = torch.zeros(
            signal_bytes + tunables.scratch_bytes, dtype=torch.uint8, device=self.device
        )
        self._slab = torch.zeros(
            peer_ptrs_bytes * tunables.max_buffers,
            dtype=torch.uint8,
            device=self.device,
        )
        self._registered = set()
        # Allocated once, here, so vLLM's memory profile sees it.
        self._staging = torch.zeros(
            self._staging_bytes(),
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
            tunables.scratch_bytes,
            tunables.sync_timeout_s,
        )
        self._register(self._staging)
        logger.info(
            "HipCommunicator ready: rank %d/%d, staging=%dMB, %s",
            self.rank,
            self.world_size,
            self._staging.numel() >> 20,
            tunables,
        )
        return True

    def _exchange(self, ptr: int) -> tuple[list[list[int]], list[int]]:
        """Every rank's IPC handle and offset for its own `ptr`, in rank order. A handle
        is a list of byte values, since an op schema has no bytes type."""
        mine = torch.ops._rocm_C.rocm_comms_handle_and_offset(ptr)
        gathered = _all_gather_object(self.cpu_group, mine)
        return [h for h, _ in gathered], [o for _, o in gathered]

    def _register(self, tensor: torch.Tensor) -> None:
        """Make `tensor` usable as a collective input, permanently.

        A collective. Only called at startup (`_open`) and after a capture
        (`_flush_pending`), which every rank reaches in the same order.
        """
        ptr = tensor.data_ptr()
        if len(self._registered) >= self.hip_tunables.max_buffers:
            raise RuntimeError(
                f"hip_comms: {len(self._registered)} registered buffers hits the "
                f"{self.hip_tunables.max_buffers} limit; raise max_buffers."
            )
        # The same gather checks that the ranks are registering the same thing.
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
        """Wrap a cudagraph capture; buffers used inside are registered on exit.

        During capture the C++ side reserves a slot per launch and records the pointer.
        A captured address is fixed for the graph's life, so filling the slots after
        capture is sound.
        """
        try:
            yield
        finally:
            # Even on error: unfilled slots fault later on a null peer pointer.
            self._flush_pending()

    def _flush_pending(self) -> None:
        """Register whatever the capture deferred. Always one collective, even with
        nothing pending, or the other ranks wait in the gather."""
        pending: Sequence[int] = torch.ops._rocm_C.rocm_comms_pending_graph_buffers(
            self._handle
        )
        mine = [torch.ops._rocm_C.rocm_comms_handle_and_offset(p) for p in pending]
        gathered: list[list[tuple[list[int], int]]] = _all_gather_object(
            self.cpu_group, mine
        )
        # The transpose below reads slot `i` from every rank, so the counts must agree.
        counts = [len(g) for g in gathered]
        if len(set(counts)) != 1:
            raise RuntimeError(
                f"hip_comms: ranks captured different numbers of buffers ({counts}); "
                "every rank must run the same graph."
            )
        if not pending:
            return
        # One entry per buffer, the world's handles laid end to end; the op splits them.
        torch.ops._rocm_C.rocm_comms_register_graph_buffers(
            self._handle,
            [[b for g in gathered for b in g[i][0]] for i in range(len(pending))],
            [[g[i][1] for g in gathered] for i in range(len(pending))],
        )

    def set_launch_override(
        self,
        kernel: Kernel | None,
        blocks: int = 16,
        threads: int = 512,
        gemm_lanes_per_col: int = 0,
        quant_bits: int = 0,
    ) -> None:
        """Force `kernel` at this geometry, the GEMM tail's lanes per column and a push
        kernel's codec bits (16: unquantized, 8, 4; each 0: the table's), for every
        later launch of its op, refusing any other op, until cleared with None. The
        sweep's and the tests' handle; the model never calls it."""
        wire = -1 if kernel is None else _KERNEL_WIRE[kernel]
        torch.ops._rocm_C.rocm_comms_set_launch_override(
            self._handle, wire, blocks, threads, gemm_lanes_per_col, quant_bits
        )

    def set_checked(self, checked: bool) -> None:
        """Bounds checks and random skew in every later kernel: the tests' mode, which
        turns a race into a failure on every run."""
        torch.ops._rocm_C.rocm_comms_set_checked(self._handle, checked)

    def _as_input(self, inp: torch.Tensor) -> torch.Tensor:
        """`inp` if the peers can read it, otherwise a copy in the staging buffer.

        Registering an eager input would mean holding it (it OOMs) or letting peers
        read recycled memory, so it is copied.
        """
        return inp if self._visible_to_peers(inp) else self._staged(inp)

    def _visible_to_peers(self, inp: torch.Tensor) -> bool:
        """Registered, or captured: a captured address is registered on capture exit,
        before any replay runs."""
        return (
            inp.data_ptr() in self._registered
            or torch.cuda.is_current_stream_capturing()
        )

    def _staged(self, inp: torch.Tensor) -> torch.Tensor:
        """`inp` copied into the registered staging buffer. Eager only, so decode
        (served from cudagraphs) never pays the copy."""
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
        out = torch.empty_like(inp)
        torch.ops._rocm_C.rocm_comms_all_reduce(self._handle, out, self._as_input(inp))
        return out

    def _admits(self, op: AdmitOp, inp: torch.Tensor) -> bool:
        """What C++ picks for this shape runs here: it has a kernel for it, the row fits
        in registers at that kernel's width, and its scratch fits (a push kernel holds
        every rank's slice). The plain all-reduce is one flat row, as C++ launches
        it."""
        if op == "all_reduce":
            rows, hidden = 1, inp.numel()
        elif inp.dim() == 2:
            rows, hidden = inp.shape
        else:
            return False
        return torch.ops._rocm_C.rocm_comms_admits(
            self._handle, _OP_WIRE[op], rows, hidden, inp.element_size()
        )

    def _all_reduce_rms_norm(
        self, inp: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> torch.Tensor:
        out = torch.empty_like(inp)
        torch.ops._rocm_C.rocm_comms_all_reduce_rms_norm(
            self._handle, out, self._as_input(inp), weight, eps
        )
        return out

    def _all_reduce_rms_norm_gemm_add(
        self,
        inp: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        gemm_weight: torch.Tensor,
        out: torch.Tensor,
        out_col0: int,
    ) -> None:
        torch.ops._rocm_C.rocm_comms_all_reduce_rms_norm_gemm_add(
            self._handle,
            out,
            out_col0,
            self._as_input(inp),
            norm_weight,
            eps,
            gemm_weight,
        )

    def _all_reduce_add_attn_res_rms_norm(
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        started = prefix is None
        prefix_out = torch.empty_like(inp) if started else prefix
        out = torch.empty_like(inp)
        torch.ops._rocm_C.rocm_comms_all_reduce_add_attn_res_rms_norm(
            self._handle,
            prefix_out,
            out,
            self._as_input(inp),
            blocks,
            norm_weight,
            qk_weight,
            out_norm_weight,
            num_blocks,
            write_idx,
            eps,
            out_eps,
            not started,
        )
        return prefix_out, out

    def _all_reduce_add_rms_norm(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the normed result, then the sum plus residual."""
        out = torch.empty_like(inp)
        residual_out = torch.empty_like(inp)
        torch.ops._rocm_C.rocm_comms_all_reduce_add_rms_norm(
            self._handle, out, residual_out, self._as_input(inp), residual, weight, eps
        )
        return out, residual_out

    def _on_close(self) -> None:
        """Release the peer memory now. Idempotent. Dropping the object does not close
        the IPC handles the C++ side opened; this does."""
        self.disabled = True
        if self._handle is None:
            return
        torch.ops._rocm_C.rocm_comms_dispose(self._handle)
        self._handle = None
