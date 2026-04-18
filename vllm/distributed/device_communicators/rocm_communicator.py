# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from typing import Optional, Protocol

import torch
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)

# Match CustomAllreduce default (8 MB).
_DEFAULT_MAX_SIZE = 8192 * 1024


class RocmCommunicator(Protocol):
    """Interface for ROCm allreduce backends.

    Mirrors CustomAllreduce:
      - should_allreduce(inp) gates on dtype/contiguity/16B alignment/max_size
      - all_reduce(inp) is out-of-place (returns a fresh tensor; inp untouched)
      - capture() is a context manager toggling _IS_CAPTURING for graph capture
    """

    disabled: bool
    backend_name: str
    max_size: int

    def should_allreduce(self, inp: torch.Tensor) -> bool: ...

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor: ...

    @contextmanager
    def capture(self): ...


def _ca_size_gates(inp: torch.Tensor, max_size: int) -> bool:
    """The size gates CustomAllreduce applies in should_custom_ar."""
    inp_size = inp.numel() * inp.element_size()
    # Custom allreduce requires input byte size to be a multiple of 16.
    if inp_size % 16 != 0:
        return False
    if inp_size >= max_size:
        return False
    return True


class AiterRocmCommunicator:
    backend_name = "aiter"

    def __init__(
        self,
        device: torch.device,
        max_size: int = _DEFAULT_MAX_SIZE,
    ) -> None:
        from vllm._aiter_ops import create_aiter_communicator

        self._impl = create_aiter_communicator(device=device)
        self.disabled = self._impl is None or self._impl.disabled
        self.max_size = max_size
        self._IS_CAPTURING = False

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if not _ca_size_gates(inp, self.max_size):
            return False
        return self._impl.should_allreduce(inp)

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        # Out-of-place: stage input into a fresh tensor, let the underlying
        # impl reduce in place on the staging buffer, return it. inp is
        # never mutated, matching CustomAllreduce.all_reduce semantics.
        out = torch.empty_like(inp)
        out.copy_(inp)
        self._impl.all_reduce(out)
        return out

    @contextmanager
    def capture(self):
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False


class QuickReduceRocmCommunicator:
    backend_name = "quick_reduce"

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_size: int = _DEFAULT_MAX_SIZE,
    ) -> None:
        from vllm.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )

        self._impl = QuickAllReduce(group=group, device=device)
        self.disabled = self._impl.disabled
        self.max_size = max_size
        self._IS_CAPTURING = False

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if not _ca_size_gates(inp, self.max_size):
            return False
        return self._impl.should_quick_allreduce(inp)

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        return self._impl.quick_all_reduce(inp)

    @contextmanager
    def capture(self):
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False


def create_rocm_communicator(
    group: ProcessGroup, device: torch.device
) -> Optional[RocmCommunicator]:
    """Build the ROCm allreduce backend selected by env vars.

    Returns None if no backend is available (caller falls through to the next
    layer, e.g. CustomAllreduce).
    """
    from vllm._aiter_ops import rocm_aiter_ops

    comm: RocmCommunicator
    if rocm_aiter_ops.is_comms_enabled():
        comm = AiterRocmCommunicator(device=device)
    else:
        comm = QuickReduceRocmCommunicator(group=group, device=device)

    if comm.disabled:
        return None

    logger.info("ROCm allreduce backend: %s", comm.backend_name)
    return comm
