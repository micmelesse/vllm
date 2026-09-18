# SPDX-License-Identifier: MIT Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All
# rights reserved.

"""The iris backend: GPU-initiated collectives over iris's own symmetric heap."""

import logging
from dataclasses import dataclass

import torch

from .base import Communicator
from .utils import widest_input_bytes

logger = logging.getLogger(__name__)


def _heap_bytes(floor: int) -> int:
    """How big iris's symmetric heap has to be: what iris itself keeps there, plus room
    for the inputs it will be handed.

    THE FLOOR IS IRIS'S OWN and the rest is the workload's. The OOM that grounded this
    arm reported 6.8 GiB of an 8 GiB heap already in use before a 2 GiB input arrived,
    so the floor covers the first part and the input size is added to it. FOUR of them,
    because a collective holds more than one at once and the exact number is iris's
    business -- this is headroom chosen from a measured failure, not a derivation, and
    the next run is what checks it.
    """
    return floor + 4 * widest_input_bytes()


@dataclass(frozen=True)
class IrisTunables:
    """Every arbitrary number this backend has, in one place."""

    # A FLOOR on the symmetric heap, not its size: `_heap_bytes` adds what the workload
    # will actually hand us. 8 GiB was this backend's whole heap while admission capped
    # inputs at `small_limit`; owning every size means whole activations arrive, and a
    # 2 GiB input against 1.16 GiB free is what iris refused (2026-09-17).
    heap_floor_bytes: int = 2**33  # 8 GiB
    slab_bytes: int = 2**25  # all-gather, 32 MB per rank
    use_gluon: bool = True


def _iris_available() -> bool:
    try:
        import iris  # noqa: F401

        return True
    except ImportError:
        return False


class IrisCommunicator(Communicator):
    """Communicator using Iris CCL GPU-initiated communication.

    Iris drives its own GPU-initiated CCL over a symmetric heap, so it uses neither
    torch group for collectives; it accepts both for interface parity with the other
    backends (and any future CPU-side coordination).

    THE HEAP IS ALLOCATED IN `_open`, which runs only once the shared gates pass -- 8 GB
    is not something to take on a box whose width we do not serve.
    """

    # Set in `_open`. Class attributes so a DISABLED communicator is still a safe
    # object to hold and close.
    iris: IrisTunables = IrisTunables()
    _shmem = None
    _gluon_config = None
    _workspace = None
    _input_buf = None
    _buf_shape = None
    _buf_dtype = None
    _ag_input_slab = None
    _ag_output_slab = None

    def _open(self) -> bool:
        if not _iris_available():
            logger.warning("IrisCommunicator disabled: the iris package is not here")
            return False
        try:
            import iris
            from iris.ccl.config import Config as CclConfig

            self._heap = _heap_bytes(self.iris.heap_floor_bytes)
            self._shmem = iris.iris(heap_size=self._heap)
            self._gluon_config = CclConfig(use_gluon=self.iris.use_gluon)
        except Exception as e:
            logger.warning("IrisCommunicator disabled: iris failed to start: %s", e)
            return False

        # ITS RANKS AND OURS MUST AGREE. iris counts its own, and every buffer below is
        # sized by `self.world_size`, which came from the device group. They match in
        # any arrangement we run; a mismatch would be a silently wrong shape, so it is
        # a disable and not an assumption.
        if self._shmem.num_ranks != self.world_size:
            logger.warning(
                "IrisCommunicator disabled: iris has %d ranks, the device group %d",
                self._shmem.num_ranks,
                self.world_size,
            )
            return False

        # A floor on the CONFIGURATION: the heap has to back at least the small path.
        # It is no longer an upper bound on a tensor -- admission stopped gating on size
        # -- so a large enough input can still exhaust the heap at call time.
        small = self.tunables.small_limit
        if small * 2 > self._heap or small > self.iris.slab_bytes:
            logger.warning(
                "IrisCommunicator disabled: heap=%dGB / slab=%dMB cannot back a "
                "%dMB small path",
                self._heap >> 30,
                self.iris.slab_bytes >> 20,
                small >> 20,
            )
            return False
        logger.info(
            "IrisCommunicator ready: world_size=%d heap=%dGB small_limit=%dMB",
            self.world_size,
            self._heap >> 30,
            small >> 20,
        )
        return True

    def _get_buffers(self, shape, dtype):
        if self._buf_shape != shape or self._buf_dtype != dtype:
            assert self._shmem is not None
            self._input_buf = self._shmem.empty(shape, dtype=dtype)
            self._buf_shape = shape
            self._buf_dtype = dtype
            self._workspace = None
        return self._input_buf

    def _all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        assert self._shmem is not None
        try:
            out = torch.empty_like(inp)
            input_buf = self._get_buffers(inp.shape, inp.dtype)
            input_buf.copy_(inp)

            if self._workspace is None:
                self._workspace = self._shmem.ccl.all_reduce_preamble(
                    out, input_buf, config=self._gluon_config
                )
            self._workspace = self._shmem.ccl.all_reduce(
                out,
                input_buf,
                workspace=self._workspace,
                config=self._gluon_config,
                async_op=True,
            )

            return out
        except Exception as e:
            logger.error(
                "IrisCommunicator.all_reduce failed: shape=%s dtype=%s "
                "capturing=%s err=%s",
                tuple(inp.shape),
                inp.dtype,
                torch.cuda.is_current_stream_capturing(),
                e,
            )
            raise

    def _get_allgather_buffers(self, numel, dtype):
        # Fixed byte slabs allocated once; per-call views avoid heap churn (the
        # symmetric heap never frees).
        if self._ag_input_slab is None:
            assert self._shmem is not None
            world_size = self.world_size
            self._ag_input_slab = self._shmem.empty(
                (self.iris.slab_bytes,), dtype=torch.uint8
            )
            self._ag_output_slab = self._shmem.empty(
                (world_size, self.iris.slab_bytes), dtype=torch.uint8
            )
        input_buf = self._ag_input_slab.view(dtype)[:numel].view(1, numel)
        output_buf = self._ag_output_slab.view(dtype)[:, :numel]
        return input_buf, output_buf

    def _all_gather(self, inp: torch.Tensor, dim: int) -> torch.Tensor:
        assert self._shmem is not None
        try:
            if dim < 0:
                dim += inp.dim()
            world_size = self.world_size
            input_size = inp.size()

            input_buf, output_buf = self._get_allgather_buffers(inp.numel(), inp.dtype)
            input_buf.view(-1).copy_(inp.reshape(-1))

            self._shmem.ccl.all_gather(
                output_buf,
                input_buf,
                config=self._gluon_config,
                async_op=True,
            )

            # Same reshape contract as vLLM's DeviceCommunicatorBase.all_gather.
            # output_buf is a non-contiguous slab view, so reshape always copies; the
            # result never aliases the symmetric heap.
            output = output_buf.reshape((world_size,) + input_size).movedim(0, dim)
            return output.reshape(
                input_size[:dim]
                + (world_size * input_size[dim],)
                + input_size[dim + 1 :]
            )

        except Exception as e:
            logger.error(
                "IrisCommunicator.all_gather failed: shape=%s dtype=%s "
                "capturing=%s err=%s",
                tuple(inp.shape),
                inp.dtype,
                torch.cuda.is_current_stream_capturing(),
                e,
            )
            raise

    def _on_close(self) -> None:
        # The symmetric heap NEVER frees, so its slabs live as long as it does --
        # dropping the heap is the only way to give the memory back, and it has to go
        # last.
        self._buf_shape = self._buf_dtype = None
        self._ag_input_slab = self._ag_output_slab = None
        self._shmem = None
        self.disabled = True

    # No `_on_capture`: iris needs nothing around a capture.
