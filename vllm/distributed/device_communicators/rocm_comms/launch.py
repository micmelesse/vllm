# SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""WHICH KERNEL, AND HOW WIDE, when a caller forces it: the Python mirror of
`csrc/rocm/rocm_comms/launch.cuh`. The model never forces one; the sweep and the tests
pass a `Launch` with a call."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

# EVERY KERNEL THERE IS, as C++ numbers them (`enum class Kernel` in launch.cuh), named
# by shot, direction (pull reads peers, push writes into them) and what it fuses. Named
# only to force one through a `Launch`; nothing else picks.
Kernel = Literal[
    "one_shot_pull",
    "one_shot_push",
    "two_shot_pull",
    "two_shot_push",
    "one_shot_pull_rms_norm",
    "one_shot_push_rms_norm",
    "two_shot_pull_rms_norm",
    "two_shot_push_rms_norm",
    "one_shot_pull_add_rms_norm",
    "one_shot_push_add_rms_norm",
    "two_shot_pull_add_rms_norm",
    "two_shot_push_add_rms_norm",
    "one_shot_pull_add_attn_res_rms_norm",
    "one_shot_push_add_attn_res_rms_norm",
    "two_shot_pull_add_attn_res_rms_norm",
    "two_shot_push_add_attn_res_rms_norm",
    "one_shot_pull_rms_norm_gemm_add",
    "one_shot_push_rms_norm_gemm_add",
    "two_shot_pull_rms_norm_gemm_add",
    "two_shot_push_rms_norm_gemm_add",
]
_KERNEL_WIRE: Mapping[Kernel, int] = {
    k: i
    for i, k in enumerate(Kernel.__args__)  # type: ignore[attr-defined]
}


@dataclass(frozen=True)
class Launch:
    """A FORCED LAUNCH, passed per call by the sweep and the tests (the model passes
    None, and C++ picks): `kernel` at this grid and block, the GEMM tail's lanes per
    column and a push kernel's codec bits (each 0: the table's)."""

    kernel: Kernel
    blocks: int = 16
    threads: int = 512
    gemm_lanes_per_col: int = 0
    quant_bits: int = 0


def launch_wire(launch: Launch | None) -> tuple[int, int, int, int, int]:
    """The five integers every op takes last: -1 and zeros for the table's."""
    if launch is None:
        return (-1, 0, 0, 0, 0)
    return (
        _KERNEL_WIRE[launch.kernel],
        launch.blocks,
        launch.threads,
        launch.gemm_lanes_per_col,
        launch.quant_bits,
    )
