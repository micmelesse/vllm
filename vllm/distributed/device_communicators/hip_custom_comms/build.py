# SPDX-License-Identifier: MIT Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""THE BUILD'S FACTS, as `HipCustomComms` and its callers name them: the ops, the Errors
and fault places C++ answers with (number for number; a test holds them equal), the
build's contents (`build_info`), whether a device runs it (`supported`), and the
exception an op raises when it does not take a call (`Refused`)."""

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

import torch

# EVERY OP, as C++ names them (`enum class Op`; a test holds them equal) and as its
# method is named: the all-reduce, then each all-reduce and the ops it fuses, in order.
Op = Literal[
    "all_reduce",
    "all_reduce_rms_norm",
    "all_reduce_add_rms_norm",
    "all_reduce_add_attn_res_rms_norm",
]


class Error(IntEnum):
    """WHY A CALL CANNOT RUN: C++'s `hip_comms::Error`, number for number (a test
    holds them equal). Every backend answers with these; ours from its C++."""

    disabled = 0
    no_such_op = 1
    not_contiguous = 2
    not_two_d = 3
    dtype_not_built = 4
    world_not_built = 5
    row_not_packs = 6
    template_not_this_ops = 7
    row_too_wide = 8
    block_not_a_wave_per_peer = 9
    scratch_too_small = 10
    grid_not_resident = 11
    staging_too_small = 12
    device_not_built = 13
    device_not_tuned = 14
    weight_not_built = 15
    no_such_template = 16
    no_such_group = 17
    ranks_disagree = 18
    groups_disagree = 19
    threads_not_built = 20
    tile_not_built = 21
    launch_out_of_range = 22
    field_not_positive = 23
    field_not_this_templates = 24
    waves_not_built = 25
    inner_stride_not_one = 26
    row_stride_not_packs = 27
    row_narrower_than_world = 28
    forced_incomplete = 29


class Where(IntEnum):
    """WHERE A KERNEL'S WAIT GAVE UP: C++'s `hip_comms::Where`, number for number (a
    test holds them equal)."""

    start = 1  # a launch's first barrier: the peer had not launched
    peer_barrier = 2  # a later peers barrier
    world_peer = 3  # a world barrier's exchange: the peer had not posted
    world = 4  # a world barrier on this device
    grid = 5  # a grid barrier on this device
    flag = 6  # a flag the peer had not written


@dataclass(frozen=True)
class Fault:
    """A RANK'S FAULT RECORD: the first of its kernels' waits to outlive the timeout,
    which then left the kernel (its output garbage) and made every other wait on the
    rank leave."""

    where: Where
    rank: int
    block: int
    peer: int  # the rank it waited for, -1 for none (a wait on this device)
    count: int  # the counter it read last
    want: int  # the value it waited for
    waited_s: float

    def __str__(self) -> str:
        waited_for = "" if self.peer < 0 else f" for rank {self.peer}"
        return (
            f"rank {self.rank} block {self.block} gave up in {self.where.name}"
            f"{waited_for} after {self.waited_s:.1f} s: count {self.count}, "
            f"want {self.want}"
        )


# C++'s `DType` names, as torch's dtypes.
_DTYPES: Mapping[str, torch.dtype] = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}


@dataclass(frozen=True)
class TemplateBuild:
    """One C++ template as built: its op, and the builds dispatch instantiates (its
    catalog list), each its fields by name (threads_per_block, waves_per_eu and its
    tile; never the launch, which a forced call gives itself)."""

    op: str
    builds: tuple[dict[str, int], ...]


@dataclass(frozen=True)
class BuildInfo:
    """What the build holds, the same on every device: C++'s `kBuild.supports`, its
    pack and staging bytes, its ops', errors' and fault places' names, and its
    templates."""

    dtypes: frozenset[torch.dtype]
    worlds: frozenset[int]
    pack_bytes: int
    staging_bytes: int
    # C++'s `Op`, `Error` and `Where` members by name, each in its enum's order: an op
    # crosses by name, an Error and a Where by number.
    op_names: tuple[str, ...]
    error_names: tuple[str, ...]
    where_names: tuple[str, ...]
    # Each template by name, in C++'s `Template` order: what a tuner searches.
    templates: dict[str, TemplateBuild]


@functools.cache
def build_info() -> BuildInfo:
    """The build's facts, read once: they are fixed when it is compiled."""
    (
        dtypes,
        worlds,
        pack,
        staging,
        ops,
        errors,
        wheres,
        names,
        template_ops,
        fields,
        flat,
        counts,
    ) = torch.ops._rocm_C.hip_custom_comms_build_info()
    templates: dict[str, TemplateBuild] = {}
    at, width = 0, len(fields)
    for name, op, count in zip(names, template_ops, counts):
        builds = []
        for _ in range(count):
            values = flat[at : at + width]
            builds.append({f: v for f, v in zip(fields, values) if v != 0})
            at += width
        templates[name] = TemplateBuild(op, tuple(builds))
    return BuildInfo(
        frozenset(_DTYPES[d] for d in dtypes),
        frozenset(worlds),
        pack,
        staging,
        tuple(ops),
        tuple(errors),
        tuple(wheres),
        templates,
    )


# WHAT RAN: the algorithm and direction, then the op's config fields as its kernel had
# them, in the op's argument order; all None from a backend with no kernels to choose
# (torch).
Ran = tuple[str | int | None, ...]


class Refused(RuntimeError):
    """A call this backend does not take, raised by the op (its `should_` said so
    first): the Error, and the input it was asked to run on."""

    def __init__(self, error: Error, inp: torch.Tensor) -> None:
        super().__init__(f"{error.name}: shape={tuple(inp.shape)} dtype={inp.dtype}")
        self.error = error


@dataclass(frozen=True)
class Supported:
    """What the library runs on, once `supported` finds it can: the device's arch."""

    arch: str


def supported(device: torch.device, world: int) -> Supported | Error:
    """Whether our kernels run on `device` in a world of `world`, as the C++ build
    answers it (`hip_comms::supported`): the Supported, or the Error."""
    arch, err = torch.ops._rocm_C.hip_custom_comms_supported(device.index, world)
    return Error(err) if err is not None else Supported(arch)


def is_weak_contiguous(inp: torch.Tensor) -> bool:
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )
