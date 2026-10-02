# SPDX-License-Identifier: MIT Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All
# rights reserved.

"""The reference backend: torch.distributed over the device group.

NOT A FAST PATH. It is the oracle the other backends are checked against -- same
interface, same admission, an implementation nobody doubts.
"""

import logging

import torch
import torch.distributed as dist

from .base import Communicator
from .launch import Launch

logger = logging.getLogger(__name__)


class TorchCommunicator(Communicator):
    """torch.distributed reference: the known-good control the other backends are
    checked against.

    Collectives run over `device_group` (nccl/rccl); the gloo `cpu_group` is accepted
    for interface parity and unused. It admits exactly what the others admit, taking the
    shared envelope unchanged.

    Supplies no `_open` and no `_on_capture`: there is nothing to bring up and nothing
    to do around a capture.

    It is gated on arch and width like the others even though torch.distributed is
    neither -- see `base.__init__`. A control is only wanted where the backends run.
    """

    def _all_reduce(
        self,
        out: torch.Tensor,
        inp: torch.Tensor,
        launch: Launch | None = None,
        quant_bits: int = 16,
    ) -> None:
        self._refuse_launch(launch)
        self._refuse_lossy(quant_bits)
        out.copy_(inp)
        dist.all_reduce(out, group=self.device_group)  # SUM
