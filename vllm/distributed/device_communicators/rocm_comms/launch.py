# SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""WHICH KERNEL, AND HOW WIDE, when a caller forces it: the Python mirror of
`csrc/rocm/rocm_comms/rocm_comms.cuh`. The model never forces one; the sweep and the
tests pass a `Launch` with a call."""

from dataclasses import dataclass
from typing import Literal, get_args

# EVERY TEMPLATE THERE IS, in C++'s order (`enum class Template` in rocm_comms.cuh; a
# test holds them equal), named by how it moves data (pull: a rank reads its peers;
# push: it also writes into them), its shot and what it fuses. Named only to force one
# through a `Launch`.
Template = Literal[
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
    "all_reduce_push_two_shot_rms_norm",
    "all_reduce_push_two_shot_add_rms_norm",
    "all_reduce_push_two_shot_add_attn_res_rms_norm",
    "all_reduce_pull_one_shot_rms_norm_gemm",
    "all_reduce_pull_two_shot_rms_norm_gemm",
    "all_reduce_pull_one_shot_rms_scale_add",
    "all_reduce_pull_two_shot_rms_scale_add",
]


@dataclass(frozen=True)
class Launch:
    """A FORCED LAUNCH, passed per call by the sweep and the tests (the model passes
    None, and select picks): `template` at this grid and block. The defaults are a
    width every template admits, for a test that forces one only to check it."""

    template: Template
    blocks: int = 16
    threads: int = 512


def launch_wire(launch: Launch | None) -> list[int] | None:
    """The launch as C++ takes it, last on every op: [template, blocks, threads], or
    None for select's."""
    if launch is None:
        return None
    return [get_args(Template).index(launch.template), launch.blocks, launch.threads]
