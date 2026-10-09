# SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""The HIP backend: our kernels in `_rocm_C` (`csrc/rocm/rocm_comms.cu`) and the peer
memory they run over.

The caller names an op and C++ picks the kernel and its launch (`select.cuh`); the
keyword arguments an op takes after its own (the algorithm, the direction, that op's
config fields) are the one way to force one, for the sweep and the tests. Every one is
passed through as it came: C++ builds the config, checks it, and answers.

TWO MEMORY PATHS, split by lifetime, both C++'s (`Handle::dev_comm`). A captured buffer
is held by vLLM for the graph's life, so it is registered once at capture exit and read
in place. An eager input is the caching allocator's, borrowed for the call, so C++
copies it into a staging buffer it owns.

The C++ context crosses as an opaque `int` handle, so nothing frees it for us:
`close()` has to run.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from vllm.logger import init_logger

from .base import Capturing, Communicator, Error, Fault, Ran, State, Where

logger = init_logger(__name__)


def _require_capture(state: State) -> None:
    """Raise on a launch being recorded into a cudagraph outside `capture()`: peers are
    registered when that context exits, and a graph captured without it replays against
    addresses that were never registered."""
    if torch.cuda.is_current_stream_capturing() and not isinstance(state, Capturing):
        raise RuntimeError(
            "HipCommunicator is being captured into a cudagraph without `capture()`. "
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


class HipCommunicator(Communicator):
    """Communicator over our HIP kernels. One per process group, because peer pointers
    are group-scoped.

    The IPC handle exchange is done here over the gloo `cpu_group`; C++ only opens the
    handles it is handed. Missing ops are an error, not a disable: falling back quietly
    would let vLLM run its own all-reduce and report the arm ready.
    """

    # Declared here so a disabled communicator is still safe to hold and close.
    _handle: int | None = None
    # The device wall clock's rate, which the fault record counts its wait in.
    _khz: int = 0
    _faults = True

    def _open(self) -> str | None:
        """Open the peer memory. A collective, so every rank must reach it.

        Eager, and before any capture: doing this inside a cudagraph capture is not
        recoverable.
        """
        # THE PEER MEMORY, made, sized and owned by C++, which checks the groups and the
        # device and gathers every rank's IPC handle over the CPU group (a collective).
        handle, err = torch.ops._rocm_C.rocm_comms_open(
            self.cpu_group.group_name, self.device_group.group_name, self.device.index
        )
        if err is not None:
            return Error(err).name
        self._handle = handle
        self._khz = torch.ops._rocm_C.rocm_comms_wall_clock_khz(self.device.index)
        logger.info("HipCommunicator ready")
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
            torch.ops._rocm_C.rocm_comms_register_captured(
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
            torch.ops._rocm_C.rocm_comms_plan_all_reduce(
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
        torch.ops._rocm_C.rocm_comms_all_reduce(
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
            torch.ops._rocm_C.rocm_comms_plan_all_reduce_rms_norm(
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
        torch.ops._rocm_C.rocm_comms_all_reduce_rms_norm(
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
        torch.ops._rocm_C.rocm_comms_all_reduce_add_rms_norm(
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
            torch.ops._rocm_C.rocm_comms_plan_all_reduce_add_attn_res_rms_norm(
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

    def _check_all_reduce_rms_norm_gemm(
        self,
        inp: torch.Tensor,
        gemm_weight: torch.Tensor,
        add: bool,
        algorithm: str | None,
        direction: str | None,
        tile_m: int | None,
        tile_n: int | None,
        tile_k: int | None,
        slice_k: int | None,
        threads_per_block: int | None,
        blocks_per_grid: int | None,
        waves_per_eu: int | None,
    ) -> Ran | Error:
        if self.disabled:
            return Error.disabled
        _require_capture(self.state)
        return _answer(
            torch.ops._rocm_C.rocm_comms_plan_all_reduce_rms_norm_gemm(
                self._handle,
                inp,
                gemm_weight,
                add,
                algorithm,
                direction,
                tile_m,
                tile_n,
                tile_k,
                slice_k,
                threads_per_block,
                blocks_per_grid,
                waves_per_eu,
            )
        )

    def _all_reduce_rms_norm_gemm(
        self,
        add: bool,
        inp: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        gemm_weight: torch.Tensor,
        out: torch.Tensor,
        algorithm: str | None,
        direction: str | None,
        tile_m: int | None,
        tile_n: int | None,
        tile_k: int | None,
        slice_k: int | None,
        threads_per_block: int | None,
        blocks_per_grid: int | None,
        waves_per_eu: int | None,
    ) -> None:
        op = (
            torch.ops._rocm_C.rocm_comms_all_reduce_rms_norm_gemm_add
            if add
            else torch.ops._rocm_C.rocm_comms_all_reduce_rms_norm_gemm
        )
        op(
            self._handle,
            out,
            inp,
            norm_weight,
            eps,
            gemm_weight,
            # The normed rows, which the GEMM reads over and over.
            torch.empty_like(inp),
            algorithm,
            direction,
            tile_m,
            tile_n,
            tile_k,
            slice_k,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
        )

    def _check_all_reduce_rms_scale_add(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
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
            torch.ops._rocm_C.rocm_comms_plan_all_reduce_rms_scale_add(
                self._handle,
                inp,
                out,
                algorithm,
                direction,
                tile_m,
                tile_n,
                threads_per_block,
                blocks_per_grid,
                waves_per_eu,
            )
        )

    def _all_reduce_rms_scale_add(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        eps: float,
        algorithm: str | None,
        direction: str | None,
        tile_m: int | None,
        tile_n: int | None,
        threads_per_block: int | None,
        blocks_per_grid: int | None,
        waves_per_eu: int | None,
    ) -> None:
        torch.ops._rocm_C.rocm_comms_all_reduce_rms_scale_add(
            self._handle,
            out,
            inp,
            eps,
            algorithm,
            direction,
            tile_m,
            tile_n,
            threads_per_block,
            blocks_per_grid,
            waves_per_eu,
        )

    def _on_close(self) -> None:
        """Release the peer memory now. Idempotent. Dropping the object does not close
        the IPC handles the C++ side opened; this does."""
        if self._handle is None:
            return
        torch.ops._rocm_C.rocm_comms_dispose(self._handle)
        self._handle = None

    def _fault(self) -> Fault | None:
        """The C++ handle's fault record (host-mapped, so this never waits on the
        device)."""
        if self._handle is None:
            return None
        got = torch.ops._rocm_C.rocm_comms_fault(self._handle)
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
            torch.ops._rocm_C.rocm_comms_abort(self._handle)
