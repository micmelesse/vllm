# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton-based AllReduce + RMSNorm + FP8 Quant fusion pass for ROCm.

This is the ROCm equivalent of AllReduceFusionPass which uses flashinfer on CUDA.
It matches patterns like:
    torch.ops.vllm.all_reduce → torch.ops.vllm.rocm_aiter_rms_norm 
        → torch.ops.vllm.rocm_aiter_per_tensor_quant
And replaces them with a fused Triton kernel.

See AllReduceFusionPass in collective_fusion.py for reference.
"""
from __future__ import annotations

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .inductor_pass import enable_fake_mode
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)


# =============================================================================
# Custom Op Registration
# =============================================================================
# Register the fused op that will replace the 3 separate ops.
# Initially this just calls the original ops - will be replaced with
# actual Triton kernel later.


def _triton_fused_allreduce_rms_quant_impl(
    input: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    quant_scale: torch.Tensor,
    quant_dtype: torch.dtype,
    group_name: str,
    # Outputs (mutated)
    allreduce_out: torch.Tensor,
    rms_out: torch.Tensor,
    quant_out: torch.Tensor,
) -> None:
    """
    Fused AllReduce + RMSNorm + FP8 Quant implementation.
    
    Currently just calls the 3 separate ops as a stub.
    TODO: Replace with actual fused Triton kernel.
    """
    # Step 1: AllReduce (vLLM's custom op, not torch.distributed)
    allreduce_result = torch.ops.vllm.all_reduce(input, group_name=group_name)
    allreduce_out.copy_(allreduce_result)
    
    # Step 2: RMSNorm
    rms_result = torch.ops.vllm.rocm_aiter_rms_norm(
        allreduce_out, rms_weight, rms_eps
    )
    rms_out.copy_(rms_result)
    
    # Step 3: FP8 Quant  
    quant_result, _ = torch.ops.vllm.rocm_aiter_per_tensor_quant(
        rms_out, quant_dtype, quant_scale
    )
    quant_out.copy_(quant_result)


def _triton_fused_allreduce_rms_quant_fake(
    input: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    quant_scale: torch.Tensor,
    quant_dtype: torch.dtype,
    group_name: str,
    # Outputs (mutated)
    allreduce_out: torch.Tensor,
    rms_out: torch.Tensor,
    quant_out: torch.Tensor,
) -> None:
    """Fake implementation for torch.compile tracing."""
    pass


# Only register if we're on ROCm
if current_platform.is_rocm():
    direct_register_custom_op(
        op_name="triton_fused_allreduce_rms_quant",
        op_func=_triton_fused_allreduce_rms_quant_impl,
        mutates_args=["allreduce_out", "rms_out", "quant_out"],
        fake_impl=_triton_fused_allreduce_rms_quant_fake,
    )
    triton_fused_allreduce_rms_quant = (
        torch.ops.vllm.triton_fused_allreduce_rms_quant.default
    )
else:
    triton_fused_allreduce_rms_quant = None


# =============================================================================
# Pattern Classes
# =============================================================================


class TritonAllReduceRMSNormQuantPattern:
    """
    Pattern to match: all_reduce → rocm_aiter_rms_norm → rocm_aiter_per_tensor_quant
    
    This pattern applies to the first Transformer block where there's no residual.
    """

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
    ) -> None:
        self.epsilon = epsilon
        self.dtype = dtype
        self.device = device
        self.quant_dtype = torch.float8_e4m3fn
        self.tp = get_tp_group()

    def get_inputs(self) -> list[torch.Tensor]:
        # Input tensor that goes through all_reduce
        input = torch.empty([16, 16], device=self.device, dtype=self.dtype)
        # RMSNorm weight
        weight = torch.empty([16], device=self.device, dtype=self.dtype)
        # Quant scale
        scale = torch.empty([1], device=self.device, dtype=torch.float32)
        return [input, weight, scale]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Match: all_reduce → rms_norm → per_tensor_quant
            allreduce_out = tensor_model_parallel_all_reduce(input)
            rms_out = torch.ops.vllm.rocm_aiter_rms_norm(
                allreduce_out, weight, self.epsilon
            )
            quant_result = torch.ops.vllm.rocm_aiter_per_tensor_quant(
                rms_out, self.quant_dtype, scale
            )
            # quant_result is tuple (quant_out, scale_out)
            return quant_result[0], allreduce_out

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Allocate output tensors
            allreduce_out = torch.empty_like(input)
            rms_out = torch.empty_like(input)
            quant_out = torch.empty_like(input, dtype=self.quant_dtype)
            
            # Call fused op
            fused_result = auto_functionalized(
                triton_fused_allreduce_rms_quant,
                input=input,
                rms_weight=weight,
                rms_eps=self.epsilon,
                quant_scale=scale,
                quant_dtype=self.quant_dtype,
                group_name=self.tp.unique_name,
                allreduce_out=allreduce_out,
                rms_out=rms_out,
                quant_out=quant_out,
            )
            # Return quant_out and allreduce_out
            # fused_result indices: [0]=token, [1-6]=inputs, [7]=allreduce_out, 
            #                       [8]=rms_out, [9]=quant_out
            return fused_result[9], fused_result[7]

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


# =============================================================================
# Fusion Pass
# =============================================================================


class TritonAllReduceFusionPass(VllmPatternMatcherPass):
    """
    ROCm-specific fusion pass that fuses AllReduce + RMSNorm + FP8 Quant.
    
    This is the ROCm equivalent of AllReduceFusionPass (which uses flashinfer).
    Uses Triton-based kernels instead of flashinfer.
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.disabled = True
        self.tp_size = get_tensor_model_parallel_world_size()
        
        if self.tp_size <= 1:
            logger.warning_once(
                "TritonAllReduceFusionPass is disabled for tp_size <= 1."
            )
            return
        
        if not current_platform.is_rocm():
            logger.warning_once(
                "TritonAllReduceFusionPass is only supported on ROCm."
            )
            return
            
        if triton_fused_allreduce_rms_quant is None:
            logger.warning_once(
                "triton_fused_allreduce_rms_quant op not registered."
            )
            return
        
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="triton_allreduce_fusion_pass"
        )
        
        if config.model_config is None:
            logger.warning_once(
                "TritonAllReduceFusionPass is disabled for missing model_config."
            )
            return
            
        self.hidden_dim = config.model_config.get_hidden_size()
        self.rank = get_tensor_model_parallel_rank()
        
        self.register_patterns()
        self.dump_patterns(config, self.patterns)

    @enable_fake_mode
    def register_patterns(self) -> None:
        """Register all pattern variants."""
        for epsilon in [1e-5, 1e-6]:
            TritonAllReduceRMSNormQuantPattern(
                epsilon,
                self.model_dtype,
                self.device,
            ).register(self.patterns)
            
            # Clear pattern cache to allow multiple epsilon values
            torch._inductor.pattern_matcher._seen_patterns.clear()

        self.disabled = False

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        if self.disabled:
            logger.warning_once("TritonAllReduceFusionPass is disabled.")
            return False
        # For now, apply to all ranges
        # TODO: Add size-based gating like flashinfer
        return True

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        if self.disabled:
            logger.debug("TritonAllReduceFusionPass disabled")
            return

        self.matched_count = self.patterns.apply(graph)
        logger.debug(
            "TritonAllReduceFusionPass replaced %s patterns", self.matched_count
        )
