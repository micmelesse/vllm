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

from vllm.compilation.fix_functionalization import FixFunctionalizationPass
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.compilation.post_cleanup import PostCleanupPass
from vllm.compilation.rocm_aiter_fusion import RocmAiterAllReduceFusionPass
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
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.fused_allreduce_add_rms_quant import fused_allreduce_add_rms_quant
from vllm.platforms import current_platform
from vllm.unfused_allreduce_add_rms_quant import unfused_allreduce_add_rms_quant
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed

from ...utils import multi_gpu_test
from ..backend import TestBackend


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

@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("use_residual", [False, True])
@pytest.mark.parametrize("num_tokens,hidden_size", [
    (1, 2048),       # single token decode, Llama 1B
    (16, 4096),      # small batch, Llama 8B
    (17, 7168),      # odd token count, DeepSeek V3
    (32, 8192),      # larger batch, Llama 70B
])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.hip),
    reason="ROCm AITER fusion pass only runs on ROCm (HIP)",
)
def test_rocm_aiter_allreduce_fusion_compile(
    use_residual: bool,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    num_processes: int = 2
):
    """Verify the fusion pass matches patterns and produces correct output.

    Compiles the model with and without the fusion pass, checks that:
    1. The pass matched and replaced the expected ops
    2. Fused output matches unfused output
    """

    def _run_fusion_compile_test(
        local_rank: int,
        world_size: int,
        use_residual: bool,
        num_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
    ) -> None:
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

            model_name = "RedHatAI/Llama-3.2-1B-Instruct-FP8"
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

    torch.multiprocessing.spawn(
        _run_fusion_compile_test,
        args=(num_processes, use_residual, num_tokens, hidden_size, dtype),
        nprocs=num_processes,
    )

@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("impl", ["torch", "iris_ccl", "iris_inline",
                                   "iris_opt"])
@pytest.mark.parametrize("num_tokens,hidden_size", [
    (1, 2048),       # single token, Llama 1B
    (16, 4096),      # small batch, Llama 8B
    (17, 7168),      # odd token count, DeepSeek V3
    (32, 8192),      # larger batch, Llama 70B
])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("mode", ["eager", "graph"])
@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.hip),
    reason="ROCm AITER fusion pass only runs on ROCm (HIP)",
)
def test_rocm_aiter_allreduce_impl_correctness(
    impl: str,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    mode: str,
    num_processes: int = 2
):
    """Compare impl against the unfused individual ops.

    Runs in both eager mode (direct execution) and graph mode (CUDA graph
    capture/replay). iris_ccl uses host barriers and is not graph-capturable.
    """

    def _run_impl_correctness_test(
        local_rank: int,
        world_size: int,
        impl: str,
        num_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        mode: str = "eager",
    ) -> None:
        """Compare a single impl against the unfused individual ops as baseline.

        Args:
            mode: "eager" for direct execution, "graph" for CUDA graph
                capture/replay. Graph mode does a warmup pass (eager), then
                captures with torch.cuda.CUDAGraph, then replays with fresh
                data and verifies correctness.
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

        # assert helper
        def _assert_impl_outputs(
            ar_impl, rms_impl, res_impl, q_impl, qs_impl,
            ar_ref, rms_ref, res_ref, q_ref, qs_ref,
            use_residual, tag, atol, rtol,
        ):
            """Assert that impl outputs match reference outputs."""
            torch.testing.assert_close(
                ar_impl, ar_ref, atol=atol, rtol=rtol,
                msg=f"allreduce_out mismatch ({tag})",
            )
            torch.testing.assert_close(
                rms_impl, rms_ref, atol=atol, rtol=rtol,
                msg=f"rms_out mismatch ({tag})",
            )

            if use_residual:
                assert res_impl is not None and res_ref is not None
                torch.testing.assert_close(
                    res_impl, res_ref, atol=atol, rtol=rtol,
                    msg=f"residual_out mismatch ({tag})",
                )
            else:
                assert res_impl is None and res_ref is None

            # Compare dequantized quant outputs
            q_impl_deq = q_impl.to(torch.float32) * qs_impl
            q_ref_deq = q_ref.to(torch.float32) * qs_ref
            torch.testing.assert_close(
                q_impl_deq, q_ref_deq, atol=atol, rtol=rtol,
                msg=f"quant_out dequantized mismatch ({tag})",
            )

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

                tag = f"impl={impl}, residual={use_residual}, mode={mode}"

                if mode == "eager":
                    (ar_impl, rms_impl, res_impl, q_impl, qs_impl) = (
                        fused_allreduce_add_rms_quant(
                            input_base.clone(), rms_weight, rms_eps,
                            quant_scale, quant_dtype, group_name,
                            residual_base.clone() if residual_base is not None
                            else None,
                            impl=impl,
                        )
                    )

                    _assert_impl_outputs(
                        ar_impl, rms_impl, res_impl, q_impl, qs_impl,
                        ar_ref, rms_ref, res_ref, q_ref, qs_ref,
                        use_residual, tag, ATOL, RTOL,
                    )

                elif mode == "graph":
                    # --- Warmup pass (eager) ---
                    # This allocates buffers, JITs Triton kernels, and
                    # initializes device_barrier flag tensors.
                    input_warmup = input_base.clone()
                    residual_warmup = (
                        residual_base.clone() if residual_base is not None
                        else None
                    )
                    fused_allreduce_add_rms_quant(
                        input_warmup, rms_weight, rms_eps,
                        quant_scale, quant_dtype, group_name,
                        residual_warmup, impl=impl,
                    )
                    torch.cuda.synchronize()

                    # --- Capture ---
                    # Use fixed input/residual tensors for capture. The graph
                    # records the kernel launches with these GPU addresses.
                    input_capture = input_base.clone()
                    residual_capture = (
                        residual_base.clone() if residual_base is not None
                        else None
                    )

                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        (ar_cap, rms_cap, res_cap, q_cap, qs_cap) = (
                            fused_allreduce_add_rms_quant(
                                input_capture, rms_weight, rms_eps,
                                quant_scale, quant_dtype, group_name,
                                residual_capture, impl=impl,
                            )
                        )

                    # --- Replay with capture data ---
                    # First replay uses the same data as capture.
                    graph.replay()
                    torch.cuda.synchronize()

                    _assert_impl_outputs(
                        ar_cap, rms_cap, res_cap, q_cap, qs_cap,
                        ar_ref, rms_ref, res_ref, q_ref, qs_ref,
                        use_residual, tag + " (replay 1)", ATOL, RTOL,
                    )

                    # --- Replay with fresh data ---
                    # Copy new data into the captured input tensors and replay.
                    input_fresh = torch.randn(
                        (num_tokens, hidden_size), dtype=dtype, device=device
                    )
                    input_capture.copy_(input_fresh)
                    if residual_base is not None:
                        residual_fresh = torch.randn(
                            (num_tokens, hidden_size), dtype=dtype, device=device
                        )
                        residual_capture.copy_(residual_fresh)
                    else:
                        residual_fresh = None

                    # Compute fresh reference
                    (ar_ref2, rms_ref2, res_ref2, q_ref2, qs_ref2) = (
                        unfused_allreduce_add_rms_quant(
                            input_fresh.clone(), rms_weight, rms_eps,
                            quant_scale, quant_dtype, group_name,
                            residual_fresh.clone() if residual_fresh is not None
                            else None,
                        )
                    )

                    graph.replay()
                    torch.cuda.synchronize()

                    _assert_impl_outputs(
                        ar_cap, rms_cap, res_cap, q_cap, qs_cap,
                        ar_ref2, rms_ref2, res_ref2, q_ref2, qs_ref2,
                        use_residual, tag + " (replay 2)", ATOL, RTOL,
                    )

        finally:
            cleanup_dist_env_and_memory()

    # launch n processes
    torch.multiprocessing.spawn(
        _run_impl_correctness_test,
        args=(
            num_processes,
            impl,
            num_tokens,
            hidden_size,
            dtype,
            mode,
        ),
        nprocs=num_processes,
    )
