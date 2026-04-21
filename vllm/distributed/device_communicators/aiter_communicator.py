# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Match CustomAllreduce default (8 MB).
_DEFAULT_MAX_SIZE = 8192 * 1024


def _ca_size_gates(inp: torch.Tensor, max_size: int) -> bool:
    """The size gates CustomAllreduce applies in should_custom_ar."""
    inp_size = inp.numel() * inp.element_size()
    if inp_size % 16 != 0:
        return False
    if inp_size >= max_size:
        return False
    return True


class AiterCommunicator:
    """Aiter-backed communicator, gated by VLLM_ROCM_USE_AITER_COMMS.

    Single class encapsulating the aiter comms path. Mirrors the
    CustomAllreduce API (`should_allreduce`, `all_reduce`, `capture`, plus
    `disabled` / `max_size`) so it slots into CudaCommunicator's
    fall-through chain alongside ca_comm / qr_comm / fi_ar_comm.

    Allreduce is the only op wired up today; additional aiter comms ops can
    be added here as methods rather than introducing a new wrapper.
    """

    def __init__(
        self,
        device: int | str | torch.device,
        max_size: int = _DEFAULT_MAX_SIZE,
    ) -> None:
        self.disabled = True
        self.max_size = max_size
        self._impl = None
        self._IS_CAPTURING = False

        from vllm._aiter_ops import rocm_aiter_ops

        if not rocm_aiter_ops.is_comms_enabled():
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)

        from vllm._aiter_ops import create_aiter_communicator

        impl = create_aiter_communicator(device=device)
        if impl is None or impl.disabled:
            return

        self._impl = impl
        self.disabled = False

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if not inp.is_contiguous():
            return False
        if not _ca_size_gates(inp, self.max_size):
            return False
        assert self._impl is not None
        return self._impl.should_allreduce(inp)

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        # Out-of-place: stage into a fresh tensor, reduce in place on the
        # staging buffer, return it. Matches CustomAllreduce semantics.
        assert self._impl is not None
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
