# SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""The fused custom ops a torch.compile pass rewrites the graph to call.

    rocm_comms_all_reduce_rms_norm            all_reduce -> ir.ops.rms_norm
    rocm_comms_all_reduce_fused_add_rms_norm  all_reduce -> ir.ops.fused_add_rms_norm

Free functions over schema types, not communicator methods: they find the live backend
themselves (`rocm_comm` on the TP device communicator), so one set of peer buffers
serves both the plain collective and these.

EACH OP IS TOTAL. Once the pass has rewritten a graph there is no unfused path left to
fall back to, so whatever the backend's kernel will not take is answered here by
running exactly the two ops that were replaced.
"""

import logging
from typing import Any

import torch

import vllm.ir.ops
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.utils.torch_utils import direct_register_custom_op

logger = logging.getLogger(__name__)


def _rocm_comm() -> Any | None:
    """The live backend, or None where this process is not running one. Not cached: a
    handle kept across a process that re-forms its groups points at freed memory."""
    try:
        group = get_tp_group()
    except Exception:
        return None
    return getattr(getattr(group, "device_communicator", None), "rocm_comm", None)


def _summed(input_: torch.Tensor) -> torch.Tensor:
    """The all-reduce half of the fallback, through our backend where it takes it."""
    comm = _rocm_comm()
    if comm is not None and not comm.disabled and comm.should_allreduce(input_):
        return comm.all_reduce(input_)
    return tensor_model_parallel_all_reduce(input_)


def _weight_fits(input_: torch.Tensor, weight: torch.Tensor) -> bool:
    """The kernel takes the weight in its own dtype: the input's, or fp32, rounding as
    `vllm.ir.ops` does for either. Any other goes to the two ops, uncast."""
    return weight.dtype in (input_.dtype, torch.float32)


_WARNED: set[str] = set()


def _warn_unfused(op: str) -> None:
    if op in _WARNED:
        return
    _WARNED.add(op)
    logger.warning(
        "%s has no fused kernel for this input on this backend: running the two ops "
        "it replaced. Correct, and not what the fusion is for.",
        op,
    )


def _all_reduce_rms_norm_impl(
    input_: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    comm = _rocm_comm()
    if (
        comm is not None
        and comm.should_allreduce_rms_norm(input_)
        and _weight_fits(input_, weight)
    ):
        return comm.all_reduce_rms_norm(input_, weight, epsilon)
    _warn_unfused("rocm_comms_all_reduce_rms_norm")
    return vllm.ir.ops.rms_norm(_summed(input_), weight, epsilon)


def _all_reduce_rms_norm_fake(
    input_: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    return torch.empty_like(input_)


def _all_reduce_fused_add_rms_norm_impl(
    input_: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    comm = _rocm_comm()
    if (
        comm is not None
        and comm.should_allreduce_fused_add_rms_norm(input_)
        and _weight_fits(input_, weight)
    ):
        return comm.all_reduce_fused_add_rms_norm(input_, residual, weight, epsilon)
    _warn_unfused("rocm_comms_all_reduce_fused_add_rms_norm")
    return vllm.ir.ops.fused_add_rms_norm(_summed(input_), residual, weight, epsilon)


def _all_reduce_fused_add_rms_norm_fake(
    input_: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(input_), torch.empty_like(input_)


direct_register_custom_op(
    op_name="rocm_comms_all_reduce_rms_norm",
    op_func=_all_reduce_rms_norm_impl,
    fake_impl=_all_reduce_rms_norm_fake,
)
direct_register_custom_op(
    op_name="rocm_comms_all_reduce_fused_add_rms_norm",
    op_func=_all_reduce_fused_add_rms_norm_impl,
    fake_impl=_all_reduce_fused_add_rms_norm_fake,
)

ALL_REDUCE_RMS_NORM_OP = torch.ops.vllm.rocm_comms_all_reduce_rms_norm.default
ALL_REDUCE_FUSED_ADD_RMS_NORM_OP = (
    torch.ops.vllm.rocm_comms_all_reduce_fused_add_rms_norm.default
)
