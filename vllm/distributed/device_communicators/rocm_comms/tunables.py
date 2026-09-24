# SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""The tunable parameters EVERY backend reads. One field today.

NOT A CALLER'S SURFACE. Nobody outside this package sets these; `make_communicator`
fills them in. They are collected into a type so that the arbitrary numbers can be SEEN
TOGETHER and changed in one place, which is the whole of what this file is for.

NOT `Constants` EITHER, for the opposite reason: a constant is a value that does not
change, and every one of these is expected to move with the hardware. `Tunables` is the
standing term for an internal knob meant to be tuned (`GLIBC_TUNABLES`, Kokkos Tuning).

A tunable is a number you may change with the code still correct; a CAPABILITY is one
where changing it means changing a kernel, and those stay beside the thing they describe
-- the supported world sizes and dtypes name the `.cu`'s template instantiations, the
16-byte rule is its `vec` alignment, the gfx list is what the extension is built for.

WHAT IS NOT HERE: a backend's own numbers. Those live in that backend's file, in its own
`HipTunables` / `IrisTunables`, because nothing else can use them -- a shared type
holding `iris_heap_bytes` would make a caller tuning hip hold iris's numbers, and would
be edited by every backend added.
"""

from dataclasses import dataclass

__all__ = ["Tunables"]


@dataclass(frozen=True)
class Tunables:
    """What `base` asks of every backend uniformly."""

    # Where a collective stops being small. It was CustomAllreduce's `max_size`, the
    # point at which vLLM handed the work to QuickReduce; we own every size now, so it
    # picks which of OUR kernels runs and caps nothing.
    small_limit: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.small_limit <= 0:
            raise ValueError(f"small_limit must be positive, not {self.small_limit}")
