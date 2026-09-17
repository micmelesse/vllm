# SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""The tunable parameters EVERY backend reads. One field today.

A tunable is a number you may change with the code still correct; a CAPABILITY is one
where changing it means changing a kernel, and those stay beside the thing they describe
-- the supported world sizes and dtypes name the `.cu`'s template instantiations, the
16-byte rule is its `vec` alignment, the gfx list is what the extension is built for.

WHAT IS NOT HERE: a backend's own numbers. Those live in that backend's file, in its own
`HipConfig` / `IrisConfig`, because nothing else can use them -- a shared type holding
`iris_heap_bytes` would make a caller tuning hip hold iris's numbers, and would be
edited by every backend added.
"""

from dataclasses import dataclass

__all__ = ["Config"]


@dataclass(frozen=True)
class Config:
    """What `base` asks of every backend uniformly."""

    # Where a collective stops being small. It was CustomAllreduce's `max_size`, the
    # point at which vLLM handed the work to QuickReduce; we own every size now, so it
    # picks which of OUR kernels runs and caps nothing.
    small_limit: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.small_limit <= 0:
            raise ValueError(f"small_limit must be positive, not {self.small_limit}")
