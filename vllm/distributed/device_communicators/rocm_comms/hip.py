# SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""The HIP backend: our kernels in `_rocm_C` (`csrc/rocm/rocm_comms.cu`) and the peer
memory they run over.

The caller names an op and C++ picks the kernel and its launch geometry
(`csrc/rocm/rocm_comms/rocm_comms.cuh`); `Options` passed with a call are the one way to
force one, for the sweep and the tests.

TWO MEMORY PATHS, split by lifetime, both C++'s (`p2p::host::Group::dev_comm`). A
captured buffer is held by vLLM for the graph's life, so it is registered once at
capture exit and read in place. An eager input is the caching allocator's, borrowed for
the call, so C++ copies it into a staging buffer it owns.

The C++ context crosses as an opaque `int` handle, so nothing frees it for us:
`close()` has to run.
"""

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar, get_args

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .base import (
    AllReduceArgs,
    Args,
    AttnResArgs,
    Communicator,
    Error,
    GemmTailArgs,
    NormArgs,
    Op,
    Options,
    Plan,
    ScaleAddArgs,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HipTunables:
    """Every arbitrary number this backend has. One default each until measured."""

    # Peer-pointer slots, one per captured launch: capture_sizes x layers. 8 MB, the
    # size vLLM gives the same array.
    max_buffers: int = 131072
    # How long a kernel waits on a peer before it prints where it was and traps.
    sync_timeout_s: float = 10.0


def _wire(options: Options) -> tuple[int | None, str | None, int | None, int | None]:
    """The options as our torch ops take them, their last four values."""
    return options.quant_bits, options.template, options.blocks, options.threads


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

    OPS: ClassVar[frozenset[Op]] = frozenset(get_args(Op))
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
        memory = torch.ops._rocm_C.rocm_comms_alloc()
        handles, offsets = self._exchange(memory)
        self._handle = torch.ops._rocm_C.rocm_comms_init(
            self.rank,
            self.world_size,
            memory,
            handles,
            offsets,
            tunables.max_buffers,
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
        self,
        out: torch.Tensor,
        inp: torch.Tensor,
        options: Options,
    ) -> None:
        """Sum `inp` across the TP ranks into `out`."""
        torch.ops._rocm_C.rocm_comms_all_reduce(self._handle, out, inp, *_wire(options))

    def _plan(self, args: Args, options: Options) -> Plan | Error:
        """C++'s answer (`hip_comms::plan`): the op family's planner, handed the call's
        own tensors. Every rule about what our kernels run is there, none here."""
        ops, wire = torch.ops._rocm_C, _wire(options)
        if isinstance(args, AllReduceArgs):
            got = ops.rocm_comms_plan_all_reduce(self._handle, args.inp, *wire)
        elif isinstance(args, NormArgs):
            got = ops.rocm_comms_plan_all_reduce_rms_norm(
                self._handle, args.inp, args.weight, args.add, *wire
            )
        elif isinstance(args, AttnResArgs):
            got = ops.rocm_comms_plan_all_reduce_add_attn_res_rms_norm(
                self._handle, args.inp, *wire
            )
        elif isinstance(args, GemmTailArgs):
            got = ops.rocm_comms_plan_all_reduce_rms_norm_gemm(
                self._handle, args.inp, args.gemm_weight, args.add, *wire
            )
        elif isinstance(args, ScaleAddArgs):
            got = ops.rocm_comms_plan_all_reduce_rms_scale_add(
                self._handle, args.inp, args.out, *wire
            )
        else:
            raise AssertionError(f"{type(args).__name__} is an Args with no planner")
        template, grid, threads, err = got
        if err is not None:
            return Error(err)
        return Plan(template, grid, threads)

    def _all_reduce_rms_norm(
        self,
        out: torch.Tensor,
        inp: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        options: Options,
    ) -> None:
        torch.ops._rocm_C.rocm_comms_all_reduce_rms_norm(
            self._handle,
            out,
            inp,
            weight,
            eps,
            *_wire(options),
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
            *_wire(options),
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
            *_wire(options),
        )

    def _all_reduce_rms_scale_add(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        eps: float,
        options: Options,
    ) -> None:
        torch.ops._rocm_C.rocm_comms_all_reduce_rms_scale_add(
            self._handle,
            out,
            inp,
            eps,
            *_wire(options),
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
            has_prefix,
            *_wire(options),
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
        """The normed result into `out`, the sum plus residual into `residual_out`."""
        torch.ops._rocm_C.rocm_comms_all_reduce_add_rms_norm(
            self._handle,
            out,
            residual_out,
            inp,
            residual,
            weight,
            eps,
            *_wire(options),
        )

    def _on_close(self) -> None:
        """Release the peer memory now. Idempotent. Dropping the object does not close
        the IPC handles the C++ side opened; this does."""
        self.disabled = True
        if self._handle is None:
            return
        torch.ops._rocm_C.rocm_comms_dispose(self._handle)
        self._handle = None
