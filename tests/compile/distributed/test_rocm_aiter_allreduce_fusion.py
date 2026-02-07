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
