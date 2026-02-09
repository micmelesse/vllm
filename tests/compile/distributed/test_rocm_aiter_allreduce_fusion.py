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

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
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
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed

from ...utils import multi_gpu_test
from ..backend import TestBackend


class TestAllReduceRMSNormPerTensorQuantModel(torch.nn.Module):
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


class TestAllReduceAddRMSNormPerTensorQuantModel(torch.nn.Module):
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

    # Import to register fused custom ops
    import vllm.fused_allreduce_add_rms_quant  # noqa: F401

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

    # Import to register fused custom ops
    import vllm.fused_allreduce_add_rms_quant  # noqa: F401

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


def _run_torch_reference_test(
    local_rank: int,
    world_size: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> None:
    """Direct op-level test: compare vllm impl against pure torch reference."""
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        get_tp_group,
    )
    from vllm.fused_allreduce_add_rms_quant import (
        fused_allreduce_add_rms_quant,
    )

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

        for use_residual in [False, True]:
            # Create identical inputs for both impls
            input_vllm = torch.randn(
                (16, hidden_size), dtype=dtype, device=device
            )
            input_torch = input_vllm.clone()

            if use_residual:
                residual_vllm = torch.randn(
                    (16, hidden_size), dtype=dtype, device=device
                )
                residual_torch = residual_vllm.clone()
            else:
                residual_vllm = None
                residual_torch = None

            # Run vllm impl
            (ar_vllm, rms_vllm, res_vllm, q_vllm, qs_vllm) = (
                fused_allreduce_add_rms_quant(
                    input_vllm, rms_weight, rms_eps, quant_scale,
                    quant_dtype, group_name, residual_vllm, impl="vllm",
                )
            )

            # Run torch reference impl
            (ar_torch, rms_torch, res_torch, q_torch, qs_torch) = (
                fused_allreduce_add_rms_quant(
                    input_torch, rms_weight, rms_eps, quant_scale,
                    quant_dtype, group_name, residual_torch, impl="torch",
                )
            )

            if dtype == torch.float16:
                ATOL, RTOL = (2e-3, 2e-3)
            else:
                ATOL, RTOL = (1e-2, 1e-2)

            # Compare allreduce outputs
            torch.testing.assert_close(
                ar_vllm, ar_torch, atol=ATOL, rtol=RTOL,
                msg=f"allreduce_out mismatch (residual={use_residual})",
            )

            # Compare RMSNorm outputs
            torch.testing.assert_close(
                rms_vllm, rms_torch, atol=ATOL, rtol=RTOL,
                msg=f"rms_out mismatch (residual={use_residual})",
            )

            # Compare residual outputs
            if use_residual:
                assert res_vllm is not None and res_torch is not None
                torch.testing.assert_close(
                    res_vllm, res_torch, atol=ATOL, rtol=RTOL,
                    msg="residual_out mismatch",
                )
            else:
                assert res_vllm is None and res_torch is None

            # Compare dequantized quant outputs
            q_vllm_deq = q_vllm.to(torch.float32) * qs_vllm
            q_torch_deq = q_torch.to(torch.float32) * qs_torch
            torch.testing.assert_close(
                q_vllm_deq, q_torch_deq, atol=ATOL, rtol=RTOL,
                msg=f"quant_out dequantized mismatch "
                    f"(residual={use_residual})",
            )

    finally:
        cleanup_dist_env_and_memory()


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("test_model", [
    TestAllReduceRMSNormPerTensorQuantModel,
    TestAllReduceAddRMSNormPerTensorQuantModel,
])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [8])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["rocm"],
    reason="ROCm AITER fusion pass only runs on ROCm",
)
def test_rocm_aiter_allreduce_fusion_pass(
    test_model: type,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
):
    num_processes = 2

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
    TestAllReduceRMSNormPerTensorQuantModel,
    TestAllReduceAddRMSNormPerTensorQuantModel,
])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [8])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["rocm"],
    reason="ROCm AITER fusion pass only runs on ROCm",
)
def test_rocm_aiter_allreduce_fusion_correctness(
    test_model: type,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
):
    """Verify fused ops produce same output as unfused sequence."""
    num_processes = 2

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
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["rocm"],
    reason="ROCm AITER fusion pass only runs on ROCm",
)
def test_rocm_aiter_allreduce_torch_reference(
    hidden_size: int,
    dtype: torch.dtype,
):
    """Compare vllm impl against pure torch reference at the op level."""
    num_processes = 2

    torch.multiprocessing.spawn(
        _run_torch_reference_test,
        args=(
            num_processes,
            hidden_size,
            dtype,
        ),
        nprocs=num_processes,
    )
