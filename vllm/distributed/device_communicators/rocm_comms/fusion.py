# SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""THE FUSED ENTRY POINT: all-reduce and normalise without landing the sum in HBM
between the two.

NOT A COMMUNICATOR METHOD, and that is the whole shape of it. `Communicator.all_reduce`
is what vLLM's dispatch chain CALLS at runtime; this is what a torch.compile pass
REWRITES a graph to call. So it is a free function over schema types rather than a
method, it finds the live communicator itself, and the two entry points share one kernel
and one set of peer buffers. Nothing in `hip.py` changes.

WE REACH THE LIVE COMMUNICATOR, WHICH AITER CANNOT. Its pass stands up a SECOND
custom-allreduce with its own IPC handles and tears it down on every bail-out so the two
do not race -- it has to, being a library outside vLLM with no handle on the
communicator in use. `rocm_comm` hangs off `CudaCommunicator`, so there is one
registration here and nothing to tear down.

ONE OP SERVES BOTH PATTERNS. It returns `(normed, residual_out)`, where `residual_out`
is the all-reduce output plus the incoming residual. The pattern with no residual passes
zeros, so `residual_out` is then the all-reduce output unchanged -- which is why there
is one op here and not two.

A BACKEND THAT DOES NOT FUSE IS NOT AN ERROR. `should_allreduce_rmsnorm` says whether
this one has the kernel -- derived from the override, so it cannot claim what it has
not got -- and a no runs the two ops this is meant to replace. That is also what makes
the REWRITE testable on a backend with no kernel: does the pattern match, is the
rewritten graph legal, do the numbers agree.

THE FALLBACK IS ALSO THE SAFETY NET, and it stays after the kernel lands. A rewritten
graph has no way back: once the pass has replaced `all_reduce -> rms_norm` there is no
unfused path left to fall to at runtime. So the op must be TOTAL, and anything the
kernel will not take -- a dtype, a hidden dim too wide for one block, a size past the
backend's envelope -- is answered here rather than by a crash inside a captured graph.
"""

import logging
from typing import Any

import torch

import vllm.ir.ops
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.utils.torch_utils import direct_register_custom_op

logger = logging.getLogger(__name__)

OP_NAME = "rocm_comms_fused_allreduce_rmsnorm"


def _rocm_comm() -> Any | None:
    """The live backend, or None where this process is not running one.

    THROUGH THE SAME DOOR THE PASS USES for `ca_comm`: a field on the device
    communicator. Nothing is constructed here and nothing is cached -- a handle kept
    across a process that re-forms its groups is a handle to freed peer memory."""
    try:
        group = get_tp_group()
    except Exception:
        return None
    return getattr(getattr(group, "device_communicator", None), "rocm_comm", None)


def _summed(input_: torch.Tensor) -> torch.Tensor:
    """The all-reduce half of the reference path, through OUR backend where it will take
    it. Falling straight to `tensor_model_parallel_all_reduce` would measure vLLM's
    dispatch chain rather than ours, and the point of the reference is to isolate what
    fusion changes."""
    comm = _rocm_comm()
    if comm is not None and not comm.disabled and comm.should_allreduce(input_):
        return comm.all_reduce(input_)
    return tensor_model_parallel_all_reduce(input_)


_SAID = False


def _said_unfused() -> None:
    """Say ONCE that this is the unfused path, then stop.

    A PLAIN BOOL AND NOT `warning_once`. vLLM patches that onto `logging.Logger`, but
    nothing else in this package leans on the patch, and this runs once per norm per
    layer per step -- so the cheap check is the one that belongs in it."""
    global _SAID
    if _SAID:
        return
    _SAID = True
    logger.warning(
        "%s has no fused kernel on this backend: running all_reduce + "
        "fused_add_rms_norm instead. Correct, and not what the fusion is for.",
        OP_NAME,
    )


def _fused_allreduce_rmsnorm_impl(
    input_: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    comm = _rocm_comm()
    if comm is not None and comm.should_allreduce_rmsnorm(input_):
        return comm.all_reduce_rmsnorm(input_, residual, weight, epsilon)
    # THE TWO OPS THE PASS REPLACED, IN THE ORDER IT REPLACED THEM. Equivalent by
    # construction rather than by a tolerance: this is the same `fused_add_rms_norm`
    # the pattern matched on.
    _said_unfused()
    return vllm.ir.ops.fused_add_rms_norm(_summed(input_), residual, weight, epsilon)


def _fused_allreduce_rmsnorm_fake(
    input_: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """WHAT TRACING SEES. Shapes and dtypes only -- both outputs have the input's, since
    the norm is elementwise and the residual is the sum of two tensors of that shape."""
    return torch.empty_like(input_), torch.empty_like(input_)


direct_register_custom_op(
    op_name=OP_NAME,
    op_func=_fused_allreduce_rmsnorm_impl,
    fake_impl=_fused_allreduce_rmsnorm_fake,
)

FUSED_AR_RMSNORM_OP = torch.ops.vllm.rocm_comms_fused_allreduce_rmsnorm.default
