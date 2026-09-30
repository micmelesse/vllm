# SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""WHICH KERNEL, AND HOW WIDE, when a caller forces it: the Python mirror of
`csrc/rocm/rocm_comms/launch.cuh`. The model never forces one; the sweep and the tests
pass a `Launch` with a call."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

# EVERY KERNEL THERE IS, as C++ numbers them (`enum class Kernel` in launch.cuh), named
# by shot and what it fuses; all of them pull (a rank reads its peers). Named only to
# force one through a `Launch`; nothing else picks.
Kernel = Literal[
    "all_reduce_pull_one_shot",
    "all_reduce_pull_two_shot",
    "all_reduce_pull_one_shot_rms_norm",
    "all_reduce_pull_two_shot_rms_norm",
    "all_reduce_pull_one_shot_add_rms_norm",
    "all_reduce_pull_two_shot_add_rms_norm",
    "all_reduce_pull_one_shot_add_attn_res_rms_norm",
    "all_reduce_pull_two_shot_add_attn_res_rms_norm",
    "all_reduce_pull_one_shot_rms_norm_gemm_add",
    "all_reduce_pull_two_shot_rms_norm_gemm_add",
]
_KERNEL_WIRE: Mapping[Kernel, int] = {
    k: i
    for i, k in enumerate(Kernel.__args__)  # type: ignore[attr-defined]
}


@dataclass(frozen=True)
class Launch:
    """A FORCED LAUNCH, passed per call by the sweep and the tests (the model passes
    None, and tune.cuh picks): `kernel` at this grid and block. The defaults are a
    width every kernel admits, for a test that forces a kernel only to check it."""

    kernel: Kernel
    blocks: int = 16
    threads: int = 512


def launch_wire(launch: Launch | None) -> tuple[int, int, int]:
    """The launch's three integers, last on every op: -1 and zeros for tune.cuh's."""
    if launch is None:
        return (-1, 0, 0)
    return (_KERNEL_WIRE[launch.kernel], launch.blocks, launch.threads)
