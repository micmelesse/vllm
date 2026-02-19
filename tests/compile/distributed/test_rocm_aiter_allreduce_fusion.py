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


class AllReduceRMSNormPerTensorQuantModel(torch.nn.Module):
    """Model with all_reduce -> RMSNorm -> per_tensor_quant (no residual).

    Mimics the first transformer block pattern.
    """

    def __init__(self, hidden_size: int = 16, token_num: int = 16,
                 eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.w = [torch.rand(hidden_size, hidden_size) for _ in range(4)]
        self.rms_weight = [
            torch.rand(hidden_size, dtype=torch.float16) for _ in range(4)
        ]
        self.scale = [
            torch.rand(1, dtype=torch.float32) for _ in range(4)
        ]
        self.quant_dtype = current_platform.fp8_dtype()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(x)

        # Block 1: all_reduce -> rms_norm -> per_tensor_quant -> mm
        ar1 = tensor_model_parallel_all_reduce(z)
        rms1 = torch.ops.vllm.rocm_aiter_rms_norm(
            ar1, self.rms_weight[0], self.eps)
        q1, s1 = torch.ops.vllm.rocm_aiter_per_tensor_quant(
            rms1, self.quant_dtype, self.scale[0])

        z2 = torch.mm(q1.to(x.dtype), self.w[0])

        # Block 2
        ar2 = tensor_model_parallel_all_reduce(z2)
        rms2 = torch.ops.vllm.rocm_aiter_rms_norm(
            ar2, self.rms_weight[1], self.eps)
        q2, s2 = torch.ops.vllm.rocm_aiter_per_tensor_quant(
            rms2, self.quant_dtype, self.scale[1])

        z3 = torch.mm(q2.to(x.dtype), self.w[1])

        # Block 3
        ar3 = tensor_model_parallel_all_reduce(z3)
        rms3 = torch.ops.vllm.rocm_aiter_rms_norm(
            ar3, self.rms_weight[2], self.eps)
        q3, s3 = torch.ops.vllm.rocm_aiter_per_tensor_quant(
            rms3, self.quant_dtype, self.scale[2])

        z4 = torch.mm(q3.to(x.dtype), self.w[2])

        # Block 4
        ar4 = tensor_model_parallel_all_reduce(z4)
        rms4 = torch.ops.vllm.rocm_aiter_rms_norm(
            ar4, self.rms_weight[3], self.eps)
        q4, s4 = torch.ops.vllm.rocm_aiter_per_tensor_quant(
            rms4, self.quant_dtype, self.scale[3])

        return q4.to(x.dtype)

    def ops_in_model_before(self) -> list:
        return [
            torch.ops.vllm.all_reduce.default,
            torch.ops.vllm.rocm_aiter_rms_norm.default,
            torch.ops.vllm.rocm_aiter_per_tensor_quant.default,
        ]

    def ops_in_model_after(self) -> list:
        return [
            torch.ops.vllm.rocm_aiter_fused_allreduce_rms_quant.default,
        ]


class AllReduceAddRMSNormPerTensorQuantModel(torch.nn.Module):
    """Model with all_reduce -> fused_add_rms_norm -> per_tensor_quant.

    Mimics transformer blocks after the first (with residual connections).
    """

    def __init__(self, hidden_size: int = 16, token_num: int = 16,
                 eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.w = [torch.rand(hidden_size, hidden_size) for _ in range(4)]
        self.rms_weight = [
            torch.rand(hidden_size, dtype=torch.float16) for _ in range(4)
        ]
        self.scale = [
            torch.rand(1, dtype=torch.float32) for _ in range(4)
        ]
        self.quant_dtype = current_platform.fp8_dtype()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(x)

        # Block 1: all_reduce (creates initial residual)
        ar1 = tensor_model_parallel_all_reduce(z)
        resid = ar1
        rms1 = torch.ops.vllm.rocm_aiter_rms_norm(
            ar1, self.rms_weight[0], self.eps)
        q1, s1 = torch.ops.vllm.rocm_aiter_per_tensor_quant(
            rms1, self.quant_dtype, self.scale[0])

        z2 = torch.mm(q1.to(x.dtype), self.w[0])

        # Block 2: all_reduce -> fused_add_rms_norm (with residual)
        ar2 = tensor_model_parallel_all_reduce(z2)
        rms2, resid = torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add(
            ar2, resid, self.rms_weight[1], self.eps)
        q2, s2 = torch.ops.vllm.rocm_aiter_per_tensor_quant(
            rms2, self.quant_dtype, self.scale[1])

        z3 = torch.mm(q2.to(x.dtype), self.w[1])

        # Block 3
        ar3 = tensor_model_parallel_all_reduce(z3)
        rms3, resid = torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add(
            ar3, resid, self.rms_weight[2], self.eps)
        q3, s3 = torch.ops.vllm.rocm_aiter_per_tensor_quant(
            rms3, self.quant_dtype, self.scale[2])

        z4 = torch.mm(q3.to(x.dtype), self.w[2])

        # Block 4
        ar4 = tensor_model_parallel_all_reduce(z4)
        rms4, resid = torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add(
            ar4, resid, self.rms_weight[3], self.eps)
        q4, s4 = torch.ops.vllm.rocm_aiter_per_tensor_quant(
            rms4, self.quant_dtype, self.scale[3])

        return q4.to(x.dtype)

    def ops_in_model_before(self) -> list:
        return [
            torch.ops.vllm.all_reduce.default,
            torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add.default,
            torch.ops.vllm.rocm_aiter_per_tensor_quant.default,
        ]

    def ops_in_model_after(self) -> list:
        return [
            torch.ops.vllm.rocm_aiter_fused_allreduce_add_rms_quant.default,
        ]

@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("test_model", [
    AllReduceRMSNormPerTensorQuantModel,
    AllReduceAddRMSNormPerTensorQuantModel,
])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [8])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.hip),
    reason="ROCm AITER fusion pass only runs on ROCm (HIP)",
)
def test_rocm_aiter_allreduce_fusion_pass(
    test_model: type,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    num_processes: int = 2
):
    def _run_fusion_pass_test(
        local_rank: int,
        world_size: int,
        test_model_cls: type,
        batch_size: int,
        seq_len: int,
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


        vllm_config = VllmConfig(
            compilation_config=CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
                custom_ops=["+rms_norm"],
            )
        )
        vllm_config.compilation_config.pass_config = PassConfig(
            eliminate_noops=True,
        )
        vllm_config.device_config = DeviceConfig(device=torch.device("cuda"))
        vllm_config.parallel_config.rank = local_rank

        model_name = "RedHatAI/Llama-3.2-1B-Instruct-FP8"
        vllm_config.model_config = ModelConfig(
            model=model_name, trust_remote_code=True, dtype=dtype, seed=42
        )

        with set_current_vllm_config(vllm_config):
            allreduce_fusion_pass = RocmAiterAllReduceFusionPass(vllm_config)
            noop_pass = NoOpEliminationPass(vllm_config)
            func_pass = FixFunctionalizationPass(vllm_config)
            cleanup_pass = PostCleanupPass(vllm_config)

            backend = TestBackend(
                noop_pass, allreduce_fusion_pass, func_pass, cleanup_pass
            )

            token_num = batch_size * seq_len
            model = test_model_cls(hidden_size, token_num)

            hidden_states = torch.randn(
                (token_num, hidden_size), requires_grad=False
            )

            compiled_model = torch.compile(model, backend=backend)
            compiled_model(hidden_states)

            # Verify pattern matching occurred
            assert allreduce_fusion_pass.matched_count > 0, (
                f"Expected fusion matches, got {allreduce_fusion_pass.matched_count}"
            )

            # Verify unfused ops existed before and fused ops exist after
            backend.check_before_ops(
                model.ops_in_model_before(), fully_replaced=False
            )
            backend.check_after_ops(model.ops_in_model_after())

    # launch n instance
    torch.multiprocessing.spawn(
        _run_fusion_pass_test,
        args=(
            num_processes,
            test_model,
            batch_size,
            seq_len,
            hidden_size,
            dtype,
        ),
        nprocs=num_processes,
    )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("test_model", [
    AllReduceRMSNormPerTensorQuantModel,
    AllReduceAddRMSNormPerTensorQuantModel,
])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [8])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.hip),
    reason="ROCm AITER fusion pass only runs on ROCm (HIP)",
)
def test_rocm_aiter_allreduce_fusion_correctness(
    test_model: type,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    num_processes: int = 2
):
    """Verify fused ops produce same output as unfused sequence."""

    def _run_fusion_correctness_test(
        local_rank: int,
        world_size: int,
        test_model_cls: type,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        dtype: torch.dtype,
    ) -> None:
        """Dual-backend test: compile with and without fusion, compare outputs."""
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
            "MASTER_PORT": "12346",
        })

        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=world_size)


        vllm_config = VllmConfig(
            compilation_config=CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
                custom_ops=["+rms_norm"],
            )
        )
        vllm_config.compilation_config.pass_config = PassConfig(
            eliminate_noops=True,
        )
        vllm_config.device_config = DeviceConfig(device=torch.device("cuda"))
        vllm_config.parallel_config.rank = local_rank

        model_name = "RedHatAI/Llama-3.2-1B-Instruct-FP8"
        vllm_config.model_config = ModelConfig(
            model=model_name, trust_remote_code=True, dtype=dtype, seed=42
        )

        with set_current_vllm_config(vllm_config):
            allreduce_fusion_pass = RocmAiterAllReduceFusionPass(vllm_config)
            noop_pass = NoOpEliminationPass(vllm_config)
            func_pass = FixFunctionalizationPass(vllm_config)
            cleanup_pass = PostCleanupPass(vllm_config)

            # Backend WITH fusion pass
            backend_fused = TestBackend(
                noop_pass, allreduce_fusion_pass, func_pass, cleanup_pass
            )
            # Backend WITHOUT fusion pass
            backend_unfused = TestBackend(
                noop_pass, func_pass, cleanup_pass
            )

            token_num = batch_size * seq_len
            model = test_model_cls(hidden_size, token_num)

            hidden_states = torch.randn(
                (token_num, hidden_size), requires_grad=False
            )

            # Run fused model
            model_fused = torch.compile(model, backend=backend_fused)
            result_fused = model_fused(hidden_states)

            # Reset dynamo between compilations
            torch._dynamo.reset()

            # Run unfused model with the same input
            model_unfused = torch.compile(model, backend=backend_unfused)
            result_unfused = model_unfused(hidden_states)

            # Compare outputs
            if dtype == torch.float16:
                ATOL, RTOL = (2e-3, 2e-3)
            else:
                ATOL, RTOL = (1e-2, 1e-2)

            torch.testing.assert_close(
                result_fused, result_unfused, atol=ATOL, rtol=RTOL
            )

            # Also verify pattern matching occurred
            assert allreduce_fusion_pass.matched_count > 0, (
                f"Expected fusion matches, got "
                f"{allreduce_fusion_pass.matched_count}"
            )

            backend_fused.check_before_ops(
                model.ops_in_model_before(), fully_replaced=False
            )
            backend_fused.check_after_ops(model.ops_in_model_after())

    # launch n processes
    torch.multiprocessing.spawn(
        _run_fusion_correctness_test,
        args=(
            num_processes,
            test_model,
            batch_size,
            seq_len,
            hidden_size,
            dtype,
        ),
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
