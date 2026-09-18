# SPDX-License-Identifier: MIT Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All
# rights reserved.

"""ROCm TP collective backends: one interface, one file per implementation.

    tunables  the tunable parameters every backend shares
    base      the Communicator interface, and the admission rules every backend shares
    hip       our own kernel, built into `_rocm_C` (csrc/rocm/rocm_comms.cu)
    iris      iris's GPU-initiated collectives
    torch     torch.distributed, the oracle the others are measured against

WHICH ONE IS A CHOICE THE CALLER MAKES AND PASSES IN: this package takes process groups,
a device and a backend name, and reads no environment variable. The one thing it asks
vLLM for is the current config, in `hip._staging_bytes`, to size a buffer against the
widest batch the workload declared -- a number nobody else has and a constant would get
wrong.
"""

import logging
from typing import Literal, get_args

import torch
from torch.distributed import ProcessGroup

from .base import Communicator
from .tunables import Tunables

logger = logging.getLogger(__name__)

# THE BACKENDS THERE ARE, one file each beside this one. `vllm.envs` states the same set
# for the env var it reads; the two meet where the value is passed in, so a name vLLM
# admits and this package does not is a type error at that call rather than a surprise.
Backend = Literal["hip", "iris", "torch"]

__all__ = ["Backend", "Communicator", "make_communicator"]


def make_communicator(
    cpu_group: ProcessGroup,
    device_group: ProcessGroup,
    device: int | str | torch.device,
    backend: Backend,
) -> Communicator:
    """Construct the TP collective backend at the one branching point.

    Takes both of vLLM's process groups and each backend uses what it needs. The backend
    comes from the `backend` argument or `VLLM_ROCM_COMMS_BACKEND`, with NO default, so
    it is always an explicit choice; a missing or unknown one raises.

    Unavailability does NOT raise -- the caller checks `.disabled`. A config error does.

    IMPORTED ONE AT A TIME: `iris` needs the iris package installed and `hip` compiles a
    kernel on first use, so naming one backend must not cost the others.
    """
    if backend not in get_args(Backend):
        raise ValueError(
            f"unknown communicator backend {backend!r}; there is "
            f"{', '.join(get_args(Backend))}"
        )
    # FILLED IN HERE, not asked for. The tunables are this package's own business, so a
    # caller picks a backend and nothing else.
    tunables = Tunables()
    logger.info(
        "rocm_comms make_communicator: backend=%s tunables=%s", backend, tunables
    )
    if backend == "iris":
        from .iris import IrisCommunicator

        return IrisCommunicator(cpu_group, device_group, device, tunables)
    if backend == "torch":
        from .torch import TorchCommunicator

        return TorchCommunicator(cpu_group, device_group, device, tunables)
    if backend == "hip":
        from .hip import HipCommunicator

        return HipCommunicator(cpu_group, device_group, device, tunables)
    raise AssertionError(f"backend {backend!r} is in Backend and has no branch here")
