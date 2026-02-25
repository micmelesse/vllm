# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._ops import OpOverload

import vllm.model_executor.layers.quantization.utils.fp8_utils  # noqa: F401
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
)
from vllm.platforms import current_platform

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .act_quant_fusion import ActivationQuantPattern
from .matcher_utils import (
    MatcherFusedAddRMSNorm,
    MatcherFusedAddRMSNormAiter,
    MatcherPerTensorQuantAiter,
    MatcherQuantFP8,
    MatcherRMSNorm,
    MatcherRMSNormAiter,
    MatcherSiluAndMul,
)
from .rms_quant_fusion import (
    FusedRMSQuantKey,
)

logger = init_logger(__name__)
FP8_DTYPE = current_platform.fp8_dtype()


class AiterRMSNormQuantPattern:
    def __init__(
        self, epsilon: float, key: FusedRMSQuantKey, match_aiter_quant: bool = True
    ):
        self.epsilon = epsilon
        self.quant_dtype = key.quant.dtype

        self.rmsnorm_matcher = (
            MatcherRMSNorm(epsilon, match_rocm_aiter=True)
            if not key.fused_add
            else MatcherFusedAddRMSNorm(epsilon, match_rocm_aiter=True)
        )
        self.quant_matcher = MatcherQuantFP8(
            key.quant,
            match_rocm_aiter=match_aiter_quant,
        )


class AiterRMSNormDynamicQuantPattern(AiterRMSNormQuantPattern):
    """AITER RMSNorm + Dynamic Quantization pattern."""

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_fused_dynamic_quant_op()

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        match_aiter_quant: bool = True,
        group_shape: GroupShape = GroupShape.PER_TOKEN,
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )

        super().__init__(epsilon, key, match_aiter_quant)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_rms = self.rmsnorm_matcher(input, weight)
            result, scale = self.quant_matcher(result_rms)
            return result, scale

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result = self.FUSED_OP(
                x=input,
                weight=weight,
                epsilon=self.epsilon,
                quant_dtype=self.quant_dtype,
            )

            return result[0], result[1]

        pm.register_replacement(
            pattern,
            replacement,
            self.rmsnorm_matcher.inputs(),
            pm.fwd_only,
            pm_pass,
        )


class AiterFusedAddRMSNormDynamicQuantPattern(AiterRMSNormQuantPattern):
    """AITER RMSNorm Fused Add + Dynamic Quantization pattern."""

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_fused_add_dynamic_quant_op()

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        match_aiter_quant: bool = True,
        group_shape: GroupShape = GroupShape.PER_TOKEN,
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )

        super().__init__(epsilon, key, match_aiter_quant)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result_rms, residual_out = self.rmsnorm_matcher(input, weight, residual)
            result, scale = self.quant_matcher(result_rms)

            return result, residual_out, scale

        def replacement(
            input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result = self.FUSED_OP(
                x=input,
                residual=residual,
                weight=weight,
                epsilon=self.epsilon,
                quant_dtype=self.quant_dtype,
            )

            return result[0], result[1], result[2]

        pm.register_replacement(
            pattern,
            replacement,
            self.rmsnorm_matcher.inputs(),
            pm.fwd_only,
            pm_pass,
        )


class AiterRMSFp8GroupQuantPattern(AiterRMSNormQuantPattern):
    """
    This pattern fuses aiter rms_norm & group fp8 quant custom
    ops into an aiter rms_norm_group_fp8_quant op.
    """

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_group_fused_quant_op()

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape,
        match_aiter_quant: bool = True,
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )

        super().__init__(epsilon, key, match_aiter_quant)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_rms = self.rmsnorm_matcher(input, weight)
            result, scale = self.quant_matcher(result_rms)
            return result, scale

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            at = self.FUSED_OP(
                x=input,
                weight=weight,
                variance_epsilon=self.epsilon,
                group_size=128,
            )

            return at[0], at[1]

        pm.register_replacement(
            pattern, replacement, self.rmsnorm_matcher.inputs(), pm.fwd_only, pm_pass
        )


class AiterFusedAddRMSFp8GroupQuantPattern(AiterRMSNormQuantPattern):
    """
    This pattern fuses aiter rms_norm_with_add & group fp8 quant custom ops
    into a aiter rms_norm_with_add_group_fp8_quant op.
    """

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_group_add_fused_quant_op()

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape,
        match_aiter_quant: bool = True,
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )

        super().__init__(epsilon, key, match_aiter_quant)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result_rms, residual_out = self.rmsnorm_matcher(input, weight, residual)
            result, scale = self.quant_matcher(result_rms)

            return result, residual_out, scale

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            at = self.FUSED_OP(
                x=input,
                residual=residual,
                weight=weight,
                variance_epsilon=self.epsilon,
                group_size=128,
            )

            # result, scale, residual
            return at[0], at[1], at[2]

        pm.register_replacement(
            pattern, replacement, self.rmsnorm_matcher.inputs(), pm.fwd_only, pm_pass
        )


class RocmAiterRMSNormQuantFusionPass(VllmPatternMatcherPass):
    """
    This pass fuses aiter rms_norm & vllm/aiter quant custom ops
    into a fused rms_norm_quant op.
    It also supports fused_add_rms_norm.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rocm_aiter_rms_norm_quant_fusion_pass"
        )

        # Make sure fused add patterns are before simple rms norm,
        # as the latter is a subset of the former in torch ops
        for epsilon in [1e-5, 1e-6]:
            #  Fuse aiter rms_norm + aiter dynamic group fp8 quant
            AiterRMSFp8GroupQuantPattern(
                epsilon, FP8_DTYPE, GroupShape(1, 128)
            ).register(self.patterns)

            # Fuse aiter fused_add_rms_norm + aiter dynamic group fp8 quant
            AiterFusedAddRMSFp8GroupQuantPattern(
                epsilon, FP8_DTYPE, GroupShape(1, 128)
            ).register(self.patterns)

            for match_aiter_quant in [True, False]:
                # Fuse aiter rms_norm + (aiter / vllm built-in)
                # dynamic per-token fp8 quant
                AiterRMSNormDynamicQuantPattern(
                    epsilon, FP8_DTYPE, match_aiter_quant=match_aiter_quant
                ).register(self.patterns)

                # Fuse aiter fused_add_rms_norm + (aiter / vllm built-in)
                # dynamic per-token fp8 quant
                AiterFusedAddRMSNormDynamicQuantPattern(
                    epsilon, FP8_DTYPE, match_aiter_quant=match_aiter_quant
                ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> str:
        fusion_patterns = [
            AiterRMSNormDynamicQuantPattern,
            AiterFusedAddRMSNormDynamicQuantPattern,
            AiterRMSFp8GroupQuantPattern,
            AiterFusedAddRMSFp8GroupQuantPattern,
        ]
        return self.hash_source(self, *fusion_patterns)


class AiterSiluMulFp8GroupQuantPattern(ActivationQuantPattern):
    """
    This pattern fuses aiter silu_and_mul & group fp8 quant custom
    ops into an aiter silu_and_mul_group_fp8_quant op.
    """

    FUSED_SILU_MUL_QUANT_OP = rocm_aiter_ops.get_act_mul_fused_fp8_group_quant_op()

    def __init__(self, quant_op: OpOverload) -> None:
        self.silu_and_mul_matcher = MatcherSiluAndMul()
        self.quant_op = quant_op

    def get_inputs(self) -> list[torch.Tensor]:
        return [
            self.silu_and_mul_matcher.inputs()[0],
        ]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            at1 = self.silu_and_mul_matcher(input)
            at2 = self.quant_op(at1, 128)
            return at2[0], at2[1]

        def replacement(
            input: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            at = self.FUSED_SILU_MUL_QUANT_OP(x=input, group_size=128)
            return at[0], at[1]

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class RocmAiterSiluMulFp8GroupQuantFusionPass(VllmPatternMatcherPass):
    """
    This pass fuses a pre-defined set of custom ops into fused ops.
    It uses the torch pattern matcher to find the patterns and replace them.

    Because patterns can only be registered once, the pass is a singleton.
    This will be addressed in a future version of PyTorch:
    https://github.com/pytorch/pytorch/pull/139321#issuecomment-2452354980
    """

    AITER_GROUP_FP8_QUANT_OP = rocm_aiter_ops.get_group_quant_op()
    TRITON_GROUP_FP8_QUANT_OP = torch.ops.vllm.triton_per_token_group_quant_fp8.default

    QUANT_OPS = [AITER_GROUP_FP8_QUANT_OP, TRITON_GROUP_FP8_QUANT_OP]

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rocm_aiter_silu_mul_fp8_group_quant_fusion_pass"
        )

        for quant_op in self.QUANT_OPS:
            AiterSiluMulFp8GroupQuantPattern(quant_op).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> str:
        fusion_patterns = [
            ActivationQuantPattern,
            AiterSiluMulFp8GroupQuantPattern,
        ]
        return VllmInductorPass.hash_source(self, *fusion_patterns)


class AddAiterRMSNormPadPattern:
    """
    This pattern replaces an aiter_rmsnorm_with_add & a pad op
    with a custom triton_add_rmsnorm_pad op from AITER.
    """

    AITER_TRITON_ADD_RMSNORM_PAD_OP = rocm_aiter_ops.get_triton_add_rmsnorm_pad_op()

    def __init__(
        self,
        epsilon: float,
        hidden_size: int,
        x_pad_to_multiple: int,
    ):
        self.epsilon = epsilon
        self.hidden_size = hidden_size
        self.x_pad_to_multiple = x_pad_to_multiple
        self.rmsnorm_matcher = MatcherFusedAddRMSNorm(epsilon, match_rocm_aiter=True)

    def get_inputs(self) -> list[torch.Tensor]:
        input, weight, residual = self.rmsnorm_matcher.inputs()
        router_weight = torch.empty([8, 16], dtype=weight.dtype, device=weight.device)
        router_bias = torch.empty([8], dtype=weight.dtype, device=weight.device)
        return [input, weight, residual, router_weight, router_bias]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            router_weight: torch.Tensor,
            router_bias: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            pad_size = self.x_pad_to_multiple - (
                self.hidden_size % self.x_pad_to_multiple
            )
            result_rms, residual_out = self.rmsnorm_matcher(input, weight, residual)
            router_logits = torch.ops.vllm.rocm_unquantized_gemm(
                result_rms, router_weight, router_bias
            )
            result = torch.nn.functional.pad(
                result_rms, (0, pad_size), mode="constant", value=0.0
            )
            return result, residual_out, router_logits

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            router_weight: torch.Tensor,
            router_bias: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            at = self.AITER_TRITON_ADD_RMSNORM_PAD_OP(
                x=input,
                weight=weight,
                variance_epsilon=self.epsilon,
                residual=residual,
                x_pad_to_multiple=self.x_pad_to_multiple,
            )
            result_padded = at[0]
            router_logits = torch.ops.vllm.rocm_unquantized_gemm(
                result_padded[:, : self.hidden_size], router_weight, router_bias
            )
            residual_out = at[1]
            return result_padded, residual_out, router_logits

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class RocmAiterTritonAddRMSNormPadFusionPass(VllmPatternMatcherPass):
    """
    This pass replaces an AITER CK RMSNorm + residual add and a pad op
    with an triton_add_rmsnorm_pad op from AITER.
    """

    def __init__(self, config: VllmConfig):
        super().__init__(config)
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rocm_aiter_triton_add_rmsnorm_pad_fusion_pass"
        )

        # gpt-oss has hidden size 2880
        # padded to a multiple of 128 on gfx942 and 256 on gfx950 respectively
        hidden_size = 2880
        for epsilon in [1e-5, 1e-6]:
            for x_pad_to_multiple in [128, 256]:
                AddAiterRMSNormPadPattern(
                    epsilon, hidden_size, x_pad_to_multiple
                ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(self, AddAiterRMSNormPadPattern)


# =============================================================================
# AllReduce + RMSNorm + FP8 Quant Fusion (ROCm equivalent of flashinfer)
# =============================================================================
# This is the ROCm equivalent of AllReduceFusionPass which uses flashinfer on CUDA.
# It matches the allreduce -> rmsnorm -> quant pattern in the FX graph and
# replaces it with a fused kernel. Matches aiter custom ops directly
# (rocm_aiter_rms_norm, rocm_aiter_per_tensor_quant) which bypass vLLM's
# CustomOp decomposition and appear as opaque nodes in the graph.

class RocmAiterAllReduceRMSNormQuantGemmPattern:
    """
    Pattern: all_reduce -> rocm_aiter_rms_norm -> rocm_aiter_per_tensor_quant
             -> rocm_per_tensor_float_w8a8_scaled_mm_impl

    Applies to the first Transformer block where there's no residual.
    Matches the aiter custom ops that appear in the graph when
    VLLM_ROCM_USE_AITER=1, extended through the scaled_mm GEMM.
    The fused op takes BF16 in and produces BF16 GEMM output.
    All FP8 is internal.
    """

    FUSED_OP = rocm_aiter_ops.get_fused_allreduce_rms_quant_op()

    def __init__(self, epsilon: float, out_dtype: torch.dtype) -> None:
        self.epsilon = epsilon
        self.quant_dtype = current_platform.fp8_dtype()
        self.out_dtype = out_dtype
        self.tp = get_tp_group()
        self.rmsnorm_matcher = MatcherRMSNormAiter(epsilon)
        self.quant_matcher = MatcherPerTensorQuantAiter()

    def register(self, pm_pass: PatternMatcherPass) -> None:
        fp8_dtype = self.quant_dtype
        N = 16  # dummy dimension for example inputs

        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
            gemm_weight: torch.Tensor,
            weight_scale: torch.Tensor,
        ) -> torch.Tensor:
            allreduce_out = tensor_model_parallel_all_reduce(input)
            rms_out = self.rmsnorm_matcher(allreduce_out, weight)
            quant_out, quant_scale = self.quant_matcher(rms_out, scale)
            gemm_out = torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl(
                quant_out, gemm_weight, self.out_dtype,
                quant_scale, weight_scale, None,
            )
            return gemm_out

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
            gemm_weight: torch.Tensor,
            weight_scale: torch.Tensor,
        ) -> torch.Tensor:
            fused_result = self.FUSED_OP(
                input=input,
                rms_weight=weight,
                rms_eps=self.epsilon,
                quant_dtype=self.quant_dtype,
                group_name=self.tp.unique_name,
                gemm_weight=gemm_weight,
                weight_scale=weight_scale,
                out_dtype=self.out_dtype,
            )
            return fused_result[0]

        inputs = [
            *self.rmsnorm_matcher.inputs(),       # input (M,N), rms_weight (N,)
            self.quant_matcher.inputs()[1],        # quant_scale (1,)
            torch.empty(
                [N, N], device="meta", dtype=fp8_dtype,
            ).contiguous().t(),                    # gemm_weight (transposed)
            torch.empty(
                [1], device="meta", dtype=torch.float32,
            ),                                     # weight_scale
        ]

        pm.register_replacement(
            pattern, replacement, inputs, pm.fwd_only, pm_pass
        )


class RocmAiterAllReduceAddRMSNormQuantGemmPattern:
    """
    Pattern: all_reduce -> rocm_aiter_rmsnorm2d_fwd_with_add
             -> rocm_aiter_per_tensor_quant
             -> rocm_per_tensor_float_w8a8_scaled_mm_impl

    Applies to layers after the first where there's a residual connection.
    Matches the aiter custom ops that appear in the graph when
    VLLM_ROCM_USE_AITER=1, extended through the scaled_mm GEMM.
    The fused op takes BF16 in and produces BF16 GEMM output + updated
    residual. All FP8 is internal.
    """

    FUSED_OP = rocm_aiter_ops.get_fused_allreduce_add_rms_quant_op()

    def __init__(self, epsilon: float, out_dtype: torch.dtype) -> None:
        self.epsilon = epsilon
        self.quant_dtype = current_platform.fp8_dtype()
        self.out_dtype = out_dtype
        self.tp = get_tp_group()
        self.rmsnorm_matcher = MatcherFusedAddRMSNormAiter(epsilon)
        self.quant_matcher = MatcherPerTensorQuantAiter()

    def register(self, pm_pass: PatternMatcherPass) -> None:
        fp8_dtype = self.quant_dtype
        N = 16  # dummy dimension for example inputs

        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            scale: torch.Tensor,
            gemm_weight: torch.Tensor,
            weight_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            allreduce_out = tensor_model_parallel_all_reduce(input)
            rms_out, new_residual = self.rmsnorm_matcher(
                allreduce_out, weight, residual
            )
            quant_out, quant_scale = self.quant_matcher(rms_out, scale)
            gemm_out = torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl(
                quant_out, gemm_weight, self.out_dtype,
                quant_scale, weight_scale, None,
            )
            return gemm_out, new_residual

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            scale: torch.Tensor,
            gemm_weight: torch.Tensor,
            weight_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            fused_result = self.FUSED_OP(
                input=input,
                residual=residual,
                rms_weight=weight,
                rms_eps=self.epsilon,
                quant_dtype=self.quant_dtype,
                group_name=self.tp.unique_name,
                gemm_weight=gemm_weight,
                weight_scale=weight_scale,
                out_dtype=self.out_dtype,
            )
            # Return gemm_out, residual_out
            return fused_result[0], fused_result[1]

        inputs = [
            *self.rmsnorm_matcher.inputs(),       # input (M,N), rms_weight (N,), residual (M,N)
            self.quant_matcher.inputs()[1],        # quant_scale (1,)
            torch.empty(
                [N, N], device="meta", dtype=fp8_dtype,
            ).contiguous().t(),                    # gemm_weight (transposed)
            torch.empty(
                [1], device="meta", dtype=torch.float32,
            ),                                     # weight_scale
        ]

        pm.register_replacement(
            pattern, replacement, inputs, pm.fwd_only, pm_pass
        )


class RocmAiterAllReduceFusionPass(VllmPatternMatcherPass):
    """
    ROCm-specific fusion pass that fuses AllReduce + RMSNorm + FP8 Quant + GEMM.

    Matches the pattern: all_reduce -> rmsnorm -> per_tensor_quant -> scaled_mm
    and replaces with a single fused op. All FP8 is internal to the fusion;
    the op takes BF16 in and produces BF16 GEMM output.

    This is the ROCm equivalent of AllReduceFusionPass (which uses flashinfer).
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.disabled = True
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size <= 1:
            logger.warning_once(
                "RocmAiterAllReduceFusionPass is disabled for tp_size <= 1."
            )
            return

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rocm_aiter_allreduce_fusion_pass"
        )

        if config.model_config is None:
            logger.warning_once(
                "RocmAiterAllReduceFusionPass is disabled: missing model_config."
            )
            return

        self.hidden_dim = config.model_config.get_hidden_size()
        self.rank = get_tensor_model_parallel_rank()
        self.max_token_num = (
            config.compilation_config.pass_config
            .allreduce_rms_fusion_max_token_num
        )

        self.register_patterns()
        self.dump_patterns(config, self.patterns)

    @enable_fake_mode
    def register_patterns(self) -> None:
        """Register all pattern variants."""
        for epsilon in [1e-5, 1e-6]:
            for out_dtype in [torch.bfloat16, torch.float16]:
                # Pattern 1: No residual (first layer)
                RocmAiterAllReduceRMSNormQuantGemmPattern(
                    epsilon, out_dtype=out_dtype,
                ).register(self.patterns)

                # Pattern 2: With residual (layers after first)
                RocmAiterAllReduceAddRMSNormQuantGemmPattern(
                    epsilon, out_dtype=out_dtype,
                ).register(self.patterns)

                # Clear pattern cache to allow multiple combinations
                torch._inductor.pattern_matcher._seen_patterns.clear()

        self.disabled = False

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        if self.disabled:
            logger.warning_once("RocmAiterAllReduceFusionPass is disabled.")
            return False
        return compile_range.end <= self.max_token_num

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        if self.disabled:
            logger.debug("RocmAiterAllReduceFusionPass disabled")
            return

        self.matched_count = self.patterns.apply(graph)
        logger.debug(
            "RocmAiterAllReduceFusionPass replaced %s patterns", self.matched_count
        )
