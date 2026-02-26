# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for RocmAiterAllReduceFusionPass.

Verifies that the fusion pass correctly replaces:
    all_reduce -> rocm_aiter_rms_norm -> rocm_aiter_per_tensor_quant
with the fused op, and that the fused output matches the unfused output.
"""

import pytest
import torch

from tests.compile.backend import TestBackend
from tests.utils import multi_gpu_test
from vllm._aiter_ops import rocm_aiter_ops  # noqa: F401 (registers ops)
from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
    RocmAiterAllReduceFusionPass,
)
from vllm.compilation.passes.utility.fix_functionalization import (
    FixFunctionalizationPass,
)
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    DeviceConfig,
    ModelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    get_tp_group,
    graph_capture as vllm_graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.unfused_allreduce_add_rms_quant import unfused_allreduce_add_rms_quant
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed


class AllReduceFusionModel(torch.nn.Module):
    """Model with all_reduce -> RMSNorm -> per_tensor_quant blocks.

    Mimics a transformer with 4 blocks. Block 1 always uses plain rms_norm
    (no residual). With use_residual=True, blocks 2-4 use
    fused_add_rms_norm with a residual connection, matching real transformer
    layers after the first.

    Args:
        use_residual: If True, blocks 2-4 use rocm_aiter_rmsnorm2d_fwd_with_add.
            If False, all blocks use rocm_aiter_rms_norm.
    """

    def __init__(self, hidden_size: int = 16, token_num: int = 16,
                 eps: float = 1e-5, use_residual: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.use_residual = use_residual
        self.w = [torch.rand(hidden_size, hidden_size) for _ in range(4)]
        self.rms_weight = [
            torch.rand(hidden_size, dtype=torch.float16) for _ in range(4)
        ]
        self.scale = [
            torch.rand(1, dtype=torch.float32) for _ in range(4)
        ]
        self.quant_dtype = current_platform.fp8_dtype()

    def _block_no_residual(
        self, x: torch.Tensor, idx: int,
    ) -> tuple[torch.Tensor, None]:
        ar = tensor_model_parallel_all_reduce(x)
        rms = torch.ops.vllm.rocm_aiter_rms_norm(
            ar, self.rms_weight[idx], self.eps)
        q, s = torch.ops.vllm.rocm_aiter_per_tensor_quant(
            rms, self.quant_dtype, self.scale[idx])
        return q, None

    def _block_residual(
        self, x: torch.Tensor, resid: torch.Tensor, idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ar = tensor_model_parallel_all_reduce(x)
        rms, resid = torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add(
            ar, resid, self.rms_weight[idx], self.eps)
        q, s = torch.ops.vllm.rocm_aiter_per_tensor_quant(
            rms, self.quant_dtype, self.scale[idx])
        return q, resid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(x)

        if self.use_residual:
            # Block 1: plain rms_norm (creates initial residual from ar output)
            ar1 = tensor_model_parallel_all_reduce(z)
            resid = ar1
            rms1 = torch.ops.vllm.rocm_aiter_rms_norm(
                ar1, self.rms_weight[0], self.eps)
            q1, _ = torch.ops.vllm.rocm_aiter_per_tensor_quant(
                rms1, self.quant_dtype, self.scale[0])
            z2 = torch.mm(q1.to(x.dtype), self.w[0])

            # Blocks 2-4: fused_add_rms_norm with residual
            q2, resid = self._block_residual(z2, resid, 1)
            z3 = torch.mm(q2.to(x.dtype), self.w[1])
            q3, resid = self._block_residual(z3, resid, 2)
            z4 = torch.mm(q3.to(x.dtype), self.w[2])
            q4, resid = self._block_residual(z4, resid, 3)
        else:
            # All blocks: all_reduce -> rms_norm -> quant -> mm
            q1, _ = self._block_no_residual(z, 0)
            z2 = torch.mm(q1.to(x.dtype), self.w[0])
            q2, _ = self._block_no_residual(z2, 1)
            z3 = torch.mm(q2.to(x.dtype), self.w[1])
            q3, _ = self._block_no_residual(z3, 2)
            z4 = torch.mm(q3.to(x.dtype), self.w[2])
            q4, _ = self._block_no_residual(z4, 3)

        return q4.to(x.dtype)

    def ops_in_model_before(self) -> list:
        if self.use_residual:
            return [
                torch.ops.vllm.all_reduce.default,
                torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add.default,
                torch.ops.vllm.rocm_aiter_per_tensor_quant.default,
            ]
        else:
            return [
                torch.ops.vllm.all_reduce.default,
                torch.ops.vllm.rocm_aiter_rms_norm.default,
                torch.ops.vllm.rocm_aiter_per_tensor_quant.default,
            ]

    def ops_in_model_after(self) -> list:
        if self.use_residual:
            return [
                torch.ops.vllm.rocm_aiter_fused_allreduce_add_rms_quant.default,
            ]
        else:
            return [
                torch.ops.vllm.rocm_aiter_fused_allreduce_rms_quant.default,
            ]


# ============================================================================
# Worker functions
# ============================================================================


def _run_fusion_compile_test(
    local_rank: int,
    world_size: int,
    use_residual: bool,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    model_name: str,
) -> None:
    """Worker for test_rocm_aiter_allreduce_fusion_compile."""
    set_random_seed(0)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    update_environment_variables({
        "RANK": str(local_rank),
        "LOCAL_RANK": str(local_rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "12345",
    })

    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    try:
        vllm_config = VllmConfig(
            compilation_config=CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
                custom_ops=["+rms_norm"],
            )
        )
        vllm_config.compilation_config.pass_config = PassConfig(
            eliminate_noops=True,
        )
        vllm_config.device_config = DeviceConfig(
            device=torch.device("cuda"))
        vllm_config.parallel_config.rank = local_rank

        vllm_config.model_config = ModelConfig(
            model=model_name, trust_remote_code=True, dtype=dtype, seed=42
        )

        with set_current_vllm_config(vllm_config):
            allreduce_fusion_pass = RocmAiterAllReduceFusionPass(
                vllm_config)
            noop_pass = NoOpEliminationPass(vllm_config)
            func_pass = FixFunctionalizationPass(vllm_config)
            cleanup_pass = PostCleanupPass(vllm_config)

            backend_fused = TestBackend(
                noop_pass, allreduce_fusion_pass, func_pass, cleanup_pass
            )
            backend_unfused = TestBackend(
                noop_pass, func_pass, cleanup_pass
            )

            model = AllReduceFusionModel(
                hidden_size, num_tokens, use_residual=use_residual)

            hidden_states = torch.randn(
                (num_tokens, hidden_size), requires_grad=False
            )

            # Compile and run with fusion
            model_fused = torch.compile(model, backend=backend_fused)
            result_fused = model_fused(hidden_states)

            # Verify pattern matching and op replacement
            assert allreduce_fusion_pass.matched_count > 0, (
                f"Expected fusion matches, got "
                f"{allreduce_fusion_pass.matched_count}"
            )
            backend_fused.check_before_ops(
                model.ops_in_model_before(), fully_replaced=False
            )
            backend_fused.check_after_ops(model.ops_in_model_after())

            # Compile and run without fusion, compare outputs
            torch._dynamo.reset()
            model_unfused = torch.compile(model, backend=backend_unfused)
            result_unfused = model_unfused(hidden_states)

            if dtype == torch.float16:
                ATOL, RTOL = (2e-3, 2e-3)
            else:
                ATOL, RTOL = (1e-2, 1e-2)

            torch.testing.assert_close(
                result_fused, result_unfused, atol=ATOL, rtol=RTOL
            )

    finally:
        cleanup_dist_env_and_memory()


def _run_fused_op_correctness_test(
    local_rank: int,
    world_size: int,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    mode: str = "eager",
    iterations: int = 1,
) -> None:
    """Worker for test_rocm_aiter_fused_op_correctness.

    Compares the registered torch ops against unfused individual ops.

    Args:
        mode: "eager" for direct execution, "graph" for CUDA graph
            capture/replay.
        iterations: Number of times to invoke the fused op. Values > 1 test
            buffer reuse and barrier correctness across consecutive calls.
    """
    set_random_seed(0)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    update_environment_variables({
        "RANK": str(local_rank),
        "LOCAL_RANK": str(local_rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "12347",
    })

    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    try:
        quant_dtype = current_platform.fp8_dtype()
        group_name = get_tp_group().unique_name
        rms_weight = torch.rand(hidden_size, dtype=dtype, device=device)
        rms_eps = 1e-5
        quant_scale = torch.rand(1, dtype=torch.float32, device=device)

        if dtype == torch.float16:
            ATOL, RTOL = (2e-3, 2e-3)
        else:
            ATOL, RTOL = (1e-2, 1e-2)

        for use_residual in [False, True]:
            input_base = torch.randn(
                (num_tokens, hidden_size), dtype=dtype, device=device
            )
            residual_base = (
                torch.randn((num_tokens, hidden_size), dtype=dtype,
                            device=device)
                if use_residual else None
            )

            # Run reference: individual unfused ops
            (ar_ref, rms_ref, res_ref, q_ref, qs_ref) = (
                unfused_allreduce_add_rms_quant(
                    input_base.clone(), rms_weight, rms_eps, quant_scale,
                    quant_dtype, group_name,
                    residual_base.clone() if residual_base is not None
                    else None,
                )
            )

            tag = f"residual={use_residual}, mode={mode}"

            checks = []

            if mode == "eager":
                for i in range(iterations):
                    # Generate fresh input each iteration to stress barriers
                    if i > 0:
                        input_base = torch.randn(
                            (num_tokens, hidden_size), dtype=dtype,
                            device=device,
                        )
                        if use_residual:
                            residual_base = torch.randn(
                                (num_tokens, hidden_size), dtype=dtype,
                                device=device,
                            )

                    if use_residual:
                        result = (
                            torch.ops.vllm
                            .rocm_aiter_fused_allreduce_add_rms_quant(
                                input_base.clone(), residual_base.clone(),
                                rms_weight, rms_eps, quant_scale, quant_dtype,
                                group_name,
                            )
                        )
                        ar_out, rms_out, res_out, q_out, qs_out = result
                    else:
                        result = (
                            torch.ops.vllm
                            .rocm_aiter_fused_allreduce_rms_quant(
                                input_base.clone(), rms_weight, rms_eps,
                                quant_scale, quant_dtype, group_name,
                            )
                        )
                        ar_out, rms_out, q_out, qs_out = result
                        res_out = None

                # Recompute reference for the last iteration's input
                if iterations > 1:
                    (ar_ref, rms_ref, res_ref, q_ref, qs_ref) = (
                        unfused_allreduce_add_rms_quant(
                            input_base.clone(), rms_weight, rms_eps,
                            quant_scale, quant_dtype, group_name,
                            residual_base.clone()
                            if residual_base is not None else None,
                        )
                    )

                checks.append(
                    ((ar_out, rms_out, res_out, q_out, qs_out),
                     (ar_ref, rms_ref, res_ref, q_ref, qs_ref),
                     tag))

            elif mode == "graph":
                # Warmup pass (eager)
                if use_residual:
                    torch.ops.vllm.rocm_aiter_fused_allreduce_add_rms_quant(
                        input_base.clone(), residual_base.clone(),
                        rms_weight, rms_eps, quant_scale, quant_dtype,
                        group_name,
                    )
                else:
                    torch.ops.vllm.rocm_aiter_fused_allreduce_rms_quant(
                        input_base.clone(), rms_weight, rms_eps,
                        quant_scale, quant_dtype, group_name,
                    )
                torch.cuda.synchronize()

                # Capture
                input_capture = input_base.clone()
                residual_capture = (
                    residual_base.clone() if residual_base is not None
                    else None
                )

                with vllm_graph_capture(device=device) as ctx:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=ctx.stream):
                        if use_residual:
                            cap = (
                                torch.ops.vllm
                                .rocm_aiter_fused_allreduce_add_rms_quant(
                                    input_capture, residual_capture,
                                    rms_weight, rms_eps, quant_scale,
                                    quant_dtype, group_name,
                                )
                            )
                            cap_ar, cap_rms, cap_res, cap_q, cap_qs = cap
                        else:
                            cap = (
                                torch.ops.vllm
                                .rocm_aiter_fused_allreduce_rms_quant(
                                    input_capture, rms_weight, rms_eps,
                                    quant_scale, quant_dtype, group_name,
                                )
                            )
                            cap_ar, cap_rms, cap_q, cap_qs = cap
                            cap_res = None

                    # Replay iterations times with fresh data each time,
                    # check correctness on the last replay.
                    for i in range(iterations):
                        input_fresh = torch.randn(
                            (num_tokens, hidden_size), dtype=dtype,
                            device=device,
                        )
                        input_capture.copy_(input_fresh)
                        if residual_base is not None:
                            residual_fresh = torch.randn(
                                (num_tokens, hidden_size), dtype=dtype,
                                device=device,
                            )
                            residual_capture.copy_(residual_fresh)
                        else:
                            residual_fresh = None

                        graph.replay()
                        torch.cuda.synchronize()

                    # Check correctness on the last replay
                    last_ref = unfused_allreduce_add_rms_quant(
                        input_fresh.clone(), rms_weight, rms_eps,
                        quant_scale, quant_dtype, group_name,
                        residual_fresh.clone()
                        if residual_fresh is not None else None,
                    )
                    checks.append(
                        ((cap_ar, cap_rms, cap_res, cap_q, cap_qs),
                         last_ref,
                         tag + f" (replay {iterations})"))

            # Epilogue: compare outputs against reference
            for (ar_out, rms_out, res_out, q_out, qs_out), \
                (ar_ref_, rms_ref_, res_ref_, q_ref_, qs_ref_), \
                    check_tag in checks:
                torch.testing.assert_close(
                    ar_out, ar_ref_, atol=ATOL, rtol=RTOL,
                    msg=f"allreduce_out mismatch ({check_tag})",
                )
                torch.testing.assert_close(
                    rms_out, rms_ref_, atol=ATOL, rtol=RTOL,
                    msg=f"rms_out mismatch ({check_tag})",
                )
                if use_residual:
                    assert res_out is not None and res_ref_ is not None
                    torch.testing.assert_close(
                        res_out, res_ref_, atol=ATOL, rtol=RTOL,
                        msg=f"residual_out mismatch ({check_tag})",
                    )
                else:
                    assert res_out is None and res_ref_ is None
                q_out_deq = q_out.to(torch.float32) * qs_out
                q_ref_deq = q_ref_.to(torch.float32) * qs_ref_
                torch.testing.assert_close(
                    q_out_deq, q_ref_deq, atol=ATOL, rtol=RTOL,
                    msg=f"quant_out dequantized mismatch ({check_tag})",
                )

    finally:
        cleanup_dist_env_and_memory()


# ============================================================================
# Tests
# ============================================================================


@multi_gpu_test(num_gpus=8)
@pytest.mark.parametrize("model_name", [
    "amd/Llama-3.3-70B-Instruct-FP8-KV",
])
@pytest.mark.parametrize("use_residual", [False, True])
@pytest.mark.parametrize("num_tokens,hidden_size", [
    (1, 2048),       # single token decode, Llama 1B
    (16, 4096),      # small batch, Llama 8B
    (17, 7168),      # odd token count, DeepSeek V3
    (32, 8192),      # larger batch, Llama 70B
    (1, 8192),       # single token decode, Llama 70B hidden dim
    (512, 8192),     # production decode batch
])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.hip),
    reason="ROCm AITER fusion pass only runs on ROCm (HIP)",
)
def test_rocm_aiter_allreduce_fusion_compile(
    model_name: str,
    use_residual: bool,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    num_processes: int = 8
):
    """Verify the fusion pass matches patterns and produces correct output.

    Compiles the model with and without the fusion pass, checks that:
    1. The pass matched and replaced the expected ops
    2. Fused output matches unfused output
    """
    torch.multiprocessing.spawn(
        _run_fusion_compile_test,
        args=(num_processes, use_residual, num_tokens, hidden_size, dtype,
              model_name),
        nprocs=num_processes,
    )


@multi_gpu_test(num_gpus=8)
@pytest.mark.parametrize("num_tokens,hidden_size", [
    (1, 2048),       # single token, Llama 1B
    (16, 4096),      # small batch, Llama 8B
    (17, 7168),      # odd token count, DeepSeek V3
    (32, 8192),      # larger batch, Llama 70B
    (1, 8192),       # single token decode, Llama 70B hidden dim
    (512, 8192),     # production decode batch
])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("mode", ["eager", "graph"])
@pytest.mark.parametrize("iterations", [1, 50])
@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.hip),
    reason="ROCm AITER fusion pass only runs on ROCm (HIP)",
)
def test_rocm_aiter_fused_op_correctness(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    mode: str,
    iterations: int,
    num_processes: int = 8
):
    """Compare fused torch ops against unfused individual ops.

    Tests both rocm_aiter_fused_allreduce_rms_quant (no residual) and
    rocm_aiter_fused_allreduce_add_rms_quant (with residual) in eager
    and CUDA graph modes.
    """
    torch.multiprocessing.spawn(
        _run_fused_op_correctness_test,
        args=(
            num_processes,
            num_tokens,
            hidden_size,
            dtype,
            mode,
            iterations,
        ),
        nprocs=num_processes,
    )
