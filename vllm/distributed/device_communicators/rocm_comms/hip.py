# SPDX-License-Identifier: MIT Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All
# rights reserved.

"""The HIP backend: our own kernel in `_rocm_C`, tuned and driven by `hip_kernel`."""

import logging
from contextlib import AbstractContextManager

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from . import hip_kernel
from .base import Communicator, _rocm_arch_available
from .config import Config

logger = logging.getLogger(__name__)


class HipCommunicator(Communicator):
    """Communicator over HIP collectives we own: `csrc/rocm/rocm_comms.cu`, built into
    `_rocm_C` and driven by `hip_kernel.py`.

    Self-disables on an unsupported arch or world size. It does NOT self-disable when
    the ops are missing: that means a build without them, and falling back quietly would
    let vLLM use its own all-reduce and report the run READY.
    """

    # Matches IrisCommunicator: a two-stage reduce-scatter needs the element count
    # divisible by the world size, and these are the TP widths we actually run.
    _SUPPORTED_WORLD_SIZES = [2, 4, 8]

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device_group: ProcessGroup,
        device: int | str | torch.device,
        config: Config,
        hip: hip_kernel.HipConfig | None = None,
    ) -> None:
        # Disabled FIRST, so every early return below leaves a safe object rather than
        # one whose disabled flag depends on how far __init__ got.
        self.disabled = True
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.device = device
        self.config = config
        self.hip = hip or hip_kernel.HipConfig()
        self.world_size = dist.get_world_size(device_group)

        if not _rocm_arch_available():
            logger.info("HipCommunicator disabled: unsupported ROCm arch")
            return
        if self.world_size not in self._SUPPORTED_WORLD_SIZES:
            logger.info(
                "HipCommunicator disabled: world_size=%d not in %s",
                self.world_size,
                self._SUPPORTED_WORLD_SIZES,
            )
            return

        # EAGER, and after the disable checks: compiling inside vLLM's cudagraph capture
        # is not recoverable, and a box that cannot run this backend should not pay a
        # build. The context owns the peer handshake and is built once per group, like
        # CustomAllreduce; it knows nothing about which collective runs over it.  NOTE
        # this line is a COLLECTIVE (it all-gathers IPC handles), so every rank must
        # reach it. The disable checks above are uniform across a TP group in practice
        # -- same arch, same world size -- but if they ever were not, the ranks that got
        # here would HANG waiting for the ones that returned, rather than failing. Worth
        # knowing because a deadlock is far worse than an error.
        self._comms = hip_kernel.HipComms(cpu_group, self.device, self.hip)
        self.disabled = False
        logger.info(
            "HipCommunicator ready: world_size=%d small_limit=%dMB",
            self.world_size,
            self.config.small_limit >> 20,
        )

    # No admission of its own. A two-stage reduce-scatter will need the count to divide
    # the ranks; the shipped kernel is one-shot and does not, and the baseline does not
    # check it either, so adding it would refuse tensors both we and the path we replace
    # can handle.

    def _all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        """Sum `inp` across the TP ranks. The launch config is chosen in `hip_comms`."""
        out = torch.empty_like(inp)
        self._comms.all_reduce(out, inp)
        return out

    def _all_gather(self, inp: torch.Tensor, dim: int) -> torch.Tensor:
        """Concatenate every rank's `inp` along `dim`, rank-ordered."""
        return self._comms.all_gather(inp.contiguous(), dim)

    def _on_capture(self) -> AbstractContextManager[None]:
        # A captured input's address is not registered when the launch is recorded, so
        # the context reserves a slot during capture and exchanges the IPC handles on
        # the way out.
        return self._comms.capture()

    def _on_close(self) -> None:
        # CALLED, not collected. The context lives behind an opaque handle now, so
        # dropping this reference frees nothing: `close()` is what runs `~Comms()` and
        # its `hipIpcCloseMemHandle` on every peer base it opened. The handles are a
        # per-process resource, and a construct/destroy cycle that leaks them fails
        # later and elsewhere.
        if self._comms is not None:
            self._comms.close()
        self._comms = None
        self.disabled = True
