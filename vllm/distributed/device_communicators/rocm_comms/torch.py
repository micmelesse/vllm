# SPDX-License-Identifier: MIT Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All
# rights reserved.

"""The reference backend: torch.distributed over the device group.

NOT A FAST PATH. It is the oracle the other backends are checked against -- same
interface, same admission, an implementation nobody doubts.
"""

import logging

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .base import Communicator
from .config import Config

logger = logging.getLogger(__name__)


class TorchCommunicator(Communicator):
    """torch.distributed reference: the known-good control the other backends are
    checked against.

    Collectives run over `device_group` (nccl/rccl); the gloo `cpu_group` is accepted
    for interface parity and unused. It admits exactly what the others admit, taking the
    shared envelope unchanged.
    """

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device_group: ProcessGroup,
        device: int | str | torch.device,
        config: Config,
    ) -> None:
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.device = device
        self.config = config
        self.world_size = dist.get_world_size(device_group)
        self.disabled = False

    # Supplies neither `_admits_*` nor `_on_capture`: it takes the shared envelope
    # unchanged -- the control has to admit exactly what it is a control for -- and
    # needs no capture handling.

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
