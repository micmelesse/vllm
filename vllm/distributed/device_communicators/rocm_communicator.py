# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)


class RocmCommunicator:
    """ROCm allreduce communicator that selects the best available backend.

    VLLM_ROCM_USE_AITER_COMMS=1 (+ parent VLLM_ROCM_USE_AITER=1): AiterCommunicator.
    Otherwise: QuickAllReduce.
    """

    def __init__(self, group: ProcessGroup, device: torch.device) -> None:
        from vllm.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )

        self.disabled = True
        self._backend = None
        self._backend_name = "none"

        from vllm._aiter_ops import rocm_aiter_ops

        if rocm_aiter_ops.is_comms_enabled():
            from vllm._aiter_ops import create_aiter_communicator

            aiter = create_aiter_communicator(group=group, device=device)
            if aiter is not None and not aiter.disabled:
                self._backend = aiter
                self._backend_name = "aiter"
                self.disabled = False
                logger.info("ROCm allreduce backend: aiter")
        else:
            qr = QuickAllReduce(group=group, device=device)
            if not qr.disabled:
                self._backend = qr
                self._backend_name = "quick_reduce"
                self.disabled = False
                logger.info("ROCm allreduce backend: quick_reduce")

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        if self.disabled or self._backend is None:
            return False
        if self._backend_name == "quick_reduce":
            return self._backend.should_quick_allreduce(inp)
        return self._backend.should_allreduce(inp)

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        if self._backend_name == "quick_reduce":
            return self._backend.quick_all_reduce(inp)
        return self._backend.all_reduce(inp)
