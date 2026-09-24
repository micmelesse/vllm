# SPDX-License-Identifier: MIT Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All
# rights reserved.

"""The iris backend: GPU-initiated collectives over iris's own symmetric heap."""

import logging
from dataclasses import dataclass

import torch

from .base import Communicator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IrisTunables:
    """Every arbitrary number this backend has, in one place."""

    # THE WHOLE HEAP, as one number. Two measured failures bracket it: at 8 GiB iris had
    # 6.8 GiB in use and refused a 2 GiB input against 1.16 GiB free (2026-09-17); at 16 GiB
    # the card ran out, because the heap is allocated OUTSIDE vLLM's accounting and
    # gpu-memory-utilization 0.90 left no room for it (2026-09-19, 4 MB free, 0/128 served).
    # So the need is about 9 GiB and the ceiling is under 16, and this is the flat number in
    # between. A CONSTANT AND NOT A FORMULA: the one it replaces derived headroom from
    # `max_num_batched_tokens`, a scheduler CEILING of 131072 that no decode batch comes near,
    # so it sized the heap off a number that had nothing to do with what arrives.
    heap_bytes: int = 12 * 2**30  # 12 GiB
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

    def _heap_bytes(self) -> int:
        """How big iris's symmetric heap is. See `heap_bytes` for why it is a constant."""
        return self.iris.heap_bytes

    def _open(self) -> bool:
        if not _iris_available():
            logger.warning("IrisCommunicator disabled: the iris package is not here")
            return False
        try:
            import iris
            from iris.ccl.config import Config as CclConfig

            self._heap = self._heap_bytes()
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

    def _on_close(self) -> None:
        # The symmetric heap NEVER frees, so its slabs live as long as it does --
        # dropping the heap is the only way to give the memory back, and it has to go
        # last.
        self._buf_shape = self._buf_dtype = None
        self._ag_input_slab = self._ag_output_slab = None
        self._shmem = None
        self.disabled = True

    # No `_on_capture`: iris needs nothing around a capture.
