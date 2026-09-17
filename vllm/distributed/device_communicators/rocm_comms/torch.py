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

    NEITHER LIMIT IS ITS. torch.distributed runs on any arch at any width, so the two
    gates `base.__init__` enforces are switched off here rather than inherited. A
    control that disabled itself where the backends do would have nothing to compare
    against exactly where the comparison is wanted.

    Supplies no `_open` and no `_on_capture`: there is nothing to bring up, and nothing
    to do around a capture.
    """

    _NEEDS_OUR_ARCH = False
    _WORLD_SIZES = ()

    def _all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        out = inp.clone()
        dist.all_reduce(out, group=self.device_group)  # SUM
        return out

    def _all_gather(self, inp: torch.Tensor, dim: int) -> torch.Tensor:
        if dim < 0:
            dim += inp.dim()
        input_size = inp.size()
        out = torch.empty(
            (self.world_size,) + tuple(input_size),
            dtype=inp.dtype,
            device=inp.device,
        )
        dist.all_gather_into_tensor(out, inp.contiguous(), group=self.device_group)
        return out.movedim(0, dim).reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )
