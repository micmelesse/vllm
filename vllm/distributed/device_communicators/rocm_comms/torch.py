# SPDX-License-Identifier: MIT Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All
# rights reserved.

"""The reference backend: torch.distributed over the device group.

NOT A FAST PATH. It is the oracle the other backends are checked against -- same
interface, same admission, an implementation nobody doubts.
"""

import logging

import torch
import torch.distributed as dist

from .base import Communicator, Error, supported

logger = logging.getLogger(__name__)


class TorchCommunicator(Communicator):
    """torch.distributed reference: the known-good control the other backends are
    checked against.

    Collectives run over `device_group` (nccl/rccl); the gloo `cpu_group` is accepted
    for interface parity and unused. It admits exactly what the others admit, taking the
    shared envelope unchanged.

    Supplies no `_on_capture`: there is nothing to do around a capture.
    """

    def _open(self) -> bool:
        """Nothing to bring up, but gated on the device and world like hip: a control
        is only wanted where the backends run."""
        got = supported(self.device, dist.get_world_size(self.device_group))
        if isinstance(got, Error):
            logger.info("TorchCommunicator disabled: %s", got.name)
            return False
        return True

    def _all_reduce(
        self,
        out: torch.Tensor,
        inp: torch.Tensor,
        algorithm: str | None,
        direction: str | None,
        threads_per_block: int | None,
        blocks_per_grid: int | None,
        waves_per_eu: int | None,
    ) -> None:
        out.copy_(inp)
        dist.all_reduce(out, group=self.device_group)  # SUM
