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

    def _all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        out = inp.clone()
        dist.all_reduce(out, group=self.device_group)  # SUM
        return out
