# SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""WHICH KERNEL, AND HOW WIDE, when a caller forces it: a C++ template by name
(`kTemplates` in `csrc/rocm/rocm_comms/impl/templates.cuh`). The model never forces one;
the sweep and the tests pass a `Launch` with a call."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Launch:
    """A FORCED LAUNCH, passed per call by the sweep and the tests (the model passes
    None, and select picks): `template` at this grid and block. The defaults are a
    width every template admits, for a test that forces one only to check it."""

    template: str  # a C++ template's name; one it does not know is its Error
    blocks: int = 16
    threads: int = 512


def launch_wire(
    launch: Launch | None,
) -> tuple[str, int, int] | tuple[None, None, None]:
    """The launch as C++ takes it, the last three values of every op: its template's
    name, its blocks and its threads, or none of them for select's."""
    if launch is None:
        return None, None, None
    return launch.template, launch.blocks, launch.threads
