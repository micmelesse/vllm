# SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""The HIP backend: our kernels in `_rocm_C` (`csrc/rocm/rocm_comms.cu`) and the peer
memory they run over.

The caller names an op and C++ picks the kernel and its launch geometry
(`csrc/rocm/rocm_comms/rocm_comms.cuh`); a `Launch` passed with a call is the one way to
force one, for the sweep and the tests.

TWO MEMORY PATHS, split by lifetime, both C++'s (`p2p::host::Group::dev_comm`). A
captured buffer is held by vLLM for the graph's life, so it is registered once at
capture exit and read in place. An eager input is the caching allocator's, borrowed for
the call, so C++ copies it into a staging buffer it owns.

The C++ context crosses as an opaque `int` handle, so nothing frees it for us:
`close()` has to run.
"""

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .base import AdmitOp, Communicator
from .launch import Launch, launch_wire

logger = logging.getLogger(__name__)

# The ops as C++ numbers them (`enum class Op`).
_OP_WIRE: Mapping[AdmitOp, int] = {
    "all_reduce": 0,
    "all_reduce_rms_norm": 1,
    "all_reduce_add_rms_norm": 2,
    "all_reduce_add_attn_res_rms_norm": 3,
    "all_reduce_rms_norm_gemm_add": 4,
    "all_reduce_rms_norm_gemm": 5,
    "all_reduce_rms_scale_add": 6,
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

    def _open(self) -> bool:
        """Open the peer memory. A collective, so every rank must reach it.

        Eager, and before any capture: doing this inside a cudagraph capture is not
        recoverable.
        """
        tunables = self.hip_tunables
        self.rank = dist.get_rank(self.cpu_group)
        # THIS RANK'S PEER MEMORY, made and owned by C++ (signal block, scratch,
        # staging); Python only exchanges its handle, a process-group collective.
        memory = torch.ops._rocm_C.rocm_comms_alloc(tunables.scratch_bytes)
        handles, offsets = self._exchange(memory)
        self._handle = torch.ops._rocm_C.rocm_comms_init(
            self.rank,
            self.world_size,
            memory,
            handles,
            offsets,
            tunables.max_buffers,
            tunables.scratch_bytes,
            tunables.sync_timeout_s,
        )
        logger.info(
            "HipCommunicator ready: rank %d/%d, %s",
            self.rank,
            self.world_size,
            tunables,
        )
        return True

    def _exchange(self, ptr: int) -> tuple[list[list[int]], list[int]]:
        """Every rank's IPC handle and offset for its own `ptr`, in rank order. A handle
        is a list of byte values, since an op schema has no bytes type."""
        mine = torch.ops._rocm_C.rocm_comms_handle_and_offset(ptr)
        gathered = _all_gather_object(self.cpu_group, mine)
        return [h for h, _ in gathered], [o for _, o in gathered]

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

    def _all_reduce(
        self, inp: torch.Tensor, launch: Launch | None = None, quant_bits: int = 16
    ) -> torch.Tensor:
        """Sum `inp` across the TP ranks, out of place."""
        out = torch.empty_like(inp)
        torch.ops._rocm_C.rocm_comms_all_reduce(
            self._handle, out, inp, quant_bits, *launch_wire(launch)
        )
        return out

    def _admits(
        self,
        op: AdmitOp,
        inp: torch.Tensor,
        launch: Launch | None = None,
        quant_bits: int = 16,
        cols: int = 0,
    ) -> bool:
        """What C++ picks for this shape runs here: it has a kernel for it, the row fits
        in registers at that kernel's width, and its scratch fits. The plain all-reduce
        is one flat row, as C++ launches it; `cols` is the GEMM tail's or the
        scale-add's output columns."""
        if op == "all_reduce":
            rows, hidden = 1, inp.numel()
        elif inp.dim() == 2:
            rows, hidden = inp.shape
        else:
            return False
        return torch.ops._rocm_C.rocm_comms_admits(
            self._handle,
            _OP_WIRE[op],
            rows,
            hidden,
            inp.element_size(),
            cols,
            quant_bits,
            *launch_wire(launch),
        )

    def _all_reduce_rms_norm(
        self,
        inp: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        launch: Launch | None = None,
        quant_bits: int = 16,
    ) -> torch.Tensor:
        out = torch.empty_like(inp)
        torch.ops._rocm_C.rocm_comms_all_reduce_rms_norm(
            self._handle,
            out,
            inp,
            weight,
            eps,
            quant_bits,
            *launch_wire(launch),
        )
        return out

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
        torch.ops._rocm_C.rocm_comms_all_reduce_rms_norm_gemm(
            self._handle,
            out,
            out_col0,
            inp,
            norm_weight,
            eps,
            gemm_weight,
            # The normed rows, which the GEMM reads over and over.
            torch.empty_like(inp),
            quant_bits,
            *launch_wire(launch),
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
        torch.ops._rocm_C.rocm_comms_all_reduce_rms_norm_gemm_add(
            self._handle,
            out,
            out_col0,
            inp,
            norm_weight,
            eps,
            gemm_weight,
            # The normed rows, which the GEMM reads over and over.
            torch.empty_like(inp),
            quant_bits,
            *launch_wire(launch),
        )

    def _all_reduce_rms_scale_add(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        eps: float,
        launch: Launch | None = None,
        quant_bits: int = 16,
    ) -> None:
        torch.ops._rocm_C.rocm_comms_all_reduce_rms_scale_add(
            self._handle,
            out,
            inp,
            eps,
            quant_bits,
            *launch_wire(launch),
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
        launch: Launch | None = None,
        quant_bits: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        started = prefix is None
        prefix_out = torch.empty_like(inp) if started else prefix
        out = torch.empty_like(inp)
        torch.ops._rocm_C.rocm_comms_all_reduce_add_attn_res_rms_norm(
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
            not started,
            quant_bits,
            *launch_wire(launch),
        )
        return prefix_out, out

    def _all_reduce_add_rms_norm(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        launch: Launch | None = None,
        quant_bits: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the normed result, then the sum plus residual."""
        out = torch.empty_like(inp)
        residual_out = torch.empty_like(inp)
        torch.ops._rocm_C.rocm_comms_all_reduce_add_rms_norm(
            self._handle,
            out,
            residual_out,
            inp,
            residual,
            weight,
            eps,
            quant_bits,
            *launch_wire(launch),
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
