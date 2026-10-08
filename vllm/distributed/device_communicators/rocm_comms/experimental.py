# SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""EXPERIMENTAL OPS: for measuring and trying, with no stability promise; each is
promoted out of here or deleted. They run on one rank with no communicator (C++'s
`hip_comms::experimental`, in `csrc/rocm/rocm_comms/interface.cuh`), and pass every
argument through to C++, which raises on a call it cannot run.
"""

from typing import Literal

import torch

# EVERY EXPERIMENTAL OP, as C++ names them (beside `base.Op`; a test holds the two
# equal to C++'s ops).
ExperimentalOp = Literal["add_attn_res_rms_norm"]


def add_attn_res_rms_norm(
    prefix: torch.Tensor,
    delta: torch.Tensor,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
    tile_m: int | None = None,
    tile_n: int | None = None,
    tile_k: int | None = None,
    threads_per_block: int | None = None,
    blocks_per_grid: int | None = None,
    waves_per_eu: int | None = None,
) -> torch.Tensor:
    """Triton's `attn_res` with a delta (`vllm/models/kimi_k3/amd/ops/attn_res.py`), on
    our AttnRes kernel: `prefix += delta` in place, then the output. The config fields
    force a kernel, a launch first."""
    import vllm._rocm_C  # noqa: F401  (registers torch.ops._rocm_C)

    out = torch.empty_like(prefix)
    torch.ops._rocm_C.rocm_comms_add_attn_res_rms_norm(
        prefix,
        out,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        block_write_idx,
        eps,
        output_norm_eps,
        tile_m,
        tile_n,
        tile_k,
        threads_per_block,
        blocks_per_grid,
        waves_per_eu,
    )
    return out
