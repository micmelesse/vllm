# SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""AR + RMSNorm FUSION FOR THE `hip` BACKEND: our kernel, and nobody's library.

NOTHING STRUCTURAL IN FUSION IS AITER'S. `direct_register_custom_op`, `BasePattern`,
`VllmPatternReplacement`, `VllmFusionPatternMatcherPass`, the ops the pattern matches --
all of it is vLLM core. AITER supplies exactly two things to its own pass: the op behind
the replacement, and an init that builds the peer memory that op needs. Both are ours
here, so `RocmAiterAllReduceFusionPass` is a TEMPLATE for this file and not a dependency
of it. Nothing below imports aiter.

AND NO SECOND REGISTRATION. The aiter pass calls `initialize_aiter_allreduce` to stand
up its own custom-allreduce, then `destroy_aiter_allreduce` on every bail-out so its IPC
handles do not race vLLM's -- unavoidable for a library that cannot see vLLM's
communicator. Ours is `rocm_comm` on the device communicator, already built, already
registered, so this pass allocates nothing and has no `__del__`.

THE GATE IS THE SWITCH THAT ALREADY PICKS THE BACKEND. `VLLM_ROCM_COMMS_BACKEND` names
which collective runs, with no default, so it names which fusion runs too. No second
flag, and an arm that turns fusion on stays an arm that changed ONE thing.

WHY A PASS AND NOT A CALL SITE. vLLM PR #57920 fuses by editing `latent_moe_runner.py`
directly, which works for one model's tail and no other. A pattern rewrite reaches every
model that emits `all_reduce -> rms_norm`, which is all of them, and costs no model
code at all.
"""

from typing import Any

import torch
import torch.fx as fx

import vllm.ir.ops
from vllm import envs
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.distributed.device_communicators.rocm_comms.fusion import (
    ALL_REDUCE_FUSED_ADD_RMS_NORM_OP,
    ALL_REDUCE_RMS_NORM_OP,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.logger import init_logger

from ..vllm_inductor_pass import (
    VllmFusionPatternMatcherPass,
    VllmInductorPass,
    VllmPatternMatcherPass,
    VllmPatternReplacement,
)

logger = init_logger(__name__)


class BasePattern:
    """The dtype/device a pattern is traced at.

    TWELVE LINES, COPIED RATHER THAN IMPORTED. The identical class sits in
    `allreduce_rms_fusion`, and that module imports `rocm_aiter_ops` at its top -- so
    importing one name from it would pull aiter's module in behind our backs, which is
    the one thing this file is for not doing. Twelve lines is cheaper than the coupling,
    and if it ever moves somewhere neutral we import it from there."""

    def __init__(self, dtype: torch.dtype, device: str | None) -> None:
        self.dtype = dtype
        self.device = device
        self.tp = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()

    def empty(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.empty(*args, dtype=self.dtype, device=self.device, **kwargs)


class HipAllReduceRMSNormPattern(BasePattern, VllmPatternReplacement):
    """`all_reduce` then `rms_norm`, no residual (e.g. Kimi-K3's latent MoE tail).

    Returns only the norm, so it matches only where the all-reduce output has no other
    user: exactly where skipping it is legal."""

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str | None) -> None:
        super().__init__(dtype, device)
        self.epsilon = epsilon

    def get_inputs(self) -> list[torch.Tensor]:
        return [self.empty(5, 16), self.empty(16)]

    @property
    def pattern(self):
        def _pattern(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            allreduce_output = tensor_model_parallel_all_reduce(input)
            return vllm.ir.ops.rms_norm(allreduce_output, weight, self.epsilon)

        return _pattern

    @property
    def replacement(self):
        def _replacement(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return ALL_REDUCE_RMS_NORM_OP(
                input_=input, weight=weight.to(input.dtype), epsilon=self.epsilon
            )

        return _replacement


class HipAllReduceFusedAddRMSNormPattern(BasePattern, VllmPatternReplacement):
    """`all_reduce` then `fused_add_rms_norm`: the per-layer form in a standard
    decoder."""

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str | None) -> None:
        super().__init__(dtype, device)
        self.epsilon = epsilon

    def get_inputs(self) -> list[torch.Tensor]:
        # residual, input, weight
        return [self.empty(5, 16), self.empty(5, 16), self.empty(16)]

    @property
    def pattern(self):
        def _pattern(
            residual: torch.Tensor, input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            allreduce_output = tensor_model_parallel_all_reduce(input)
            rms, residual = vllm.ir.ops.fused_add_rms_norm(
                allreduce_output, residual, weight, self.epsilon
            )
            return rms, residual

        return _pattern

    @property
    def replacement(self):
        def _replacement(
            residual: torch.Tensor, input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            fused = ALL_REDUCE_FUSED_ADD_RMS_NORM_OP(
                input_=input,
                residual=residual,
                weight=weight.to(input.dtype),
                epsilon=self.epsilon,
            )
            return fused[0], fused[1]

        return _replacement


class RocmHipAllReduceFusionPass(VllmFusionPatternMatcherPass):
    """Rewrite `all_reduce -> rms_norm` to the one op, wherever the graph has it.

    IT REFUSES BY SAYING SO. Every reason to be off is logged once and leaves
    `self.disabled` set; a pass that quietly registers nothing is a pass whose absence
    looks like a graph that had no matches."""

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, "rocm_hip_allreduce_fusion_pass")
        self.disabled = True

        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size <= 1:
            logger.warning_once("hip AllReduce fusion is off for tp_size <= 1.")
            return

        if config.model_config is None:
            logger.warning_once("hip AllReduce fusion needs a model_config.")
            return

        # THE SAME DOOR THE OP USES AT RUNTIME. If the pass and the op disagree about
        # which communicator is live, the rewrite is for a backend that is not there.
        device_comm = get_tp_group().device_communicator
        comm = getattr(device_comm, "rocm_comm", None)
        if comm is None or comm.disabled:
            logger.warning_once(
                "hip AllReduce fusion needs a live rocm_comms backend; "
                "VLLM_ROCM_COMMS_BACKEND is %r.",
                envs.VLLM_ROCM_COMMS_BACKEND,
            )
            return

        # A BOUND FOR `is_applicable_for_range`, NOT A CORRECTNESS GATE. The op is
        # total -- it answers anything the kernel will not take with the unfused pair --
        # so a range past this is slow rather than wrong, and that is what lets the
        # rewrite be unconditional once it has matched.
        self.max_token_num = config.scheduler_config.max_num_batched_tokens

        for epsilon in [1e-5, 1e-6]:
            # THE FUSED-ADD FORM FIRST. It is the larger subgraph, and registering the
            # smaller one first lets it consume the `all_reduce` node and strand the
            # trailing add as its own kernel -- the ordering aiter's pass learned.
            self.register(
                HipAllReduceFusedAddRMSNormPattern(
                    epsilon, self.model_dtype, self.device
                )
            )
            self.register(
                HipAllReduceRMSNormPattern(epsilon, self.model_dtype, self.device)
            )
            # The pattern matcher caches by traced graph, and two epsilons trace alike;
            # without this the second registration is silently dropped.
            torch._inductor.pattern_matcher._seen_patterns.clear()

        self.disabled = False
        self.dump_patterns(config, self.pm_pass)

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        if self.disabled:
            return False
        return bool(compile_range.end <= self.max_token_num)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.pm_pass.apply(graph)
        VllmPatternMatcherPass.match_table[self.pass_name] += self.matched_count
        logger.debug(
            "%s replaced %s patterns", self.__class__.__name__, self.matched_count
        )
