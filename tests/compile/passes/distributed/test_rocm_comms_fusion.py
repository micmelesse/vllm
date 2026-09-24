# SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc. All
# rights reserved.
"""Does `RocmHipAllReduceFusionPass` rewrite exactly the all-reduces it should?

Three all-reduces in one graph:
    a -> rms_norm only               fused: rocm_comms_all_reduce_rms_norm
    b -> rms_norm, and b used again  NOT fused: skipping b would lose that use
    c -> fused_add_rms_norm          fused: rocm_comms_all_reduce_fused_add_rms_norm
and the compiled model must agree with the eager one.
"""

import pytest
import torch

from tests.compile.backend import TestBackend
from tests.utils import multi_gpu_test
from vllm.compilation.passes.fusion.rocm_comms_fusion import (
    RocmHipAllReduceFusionPass,
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
from vllm.distributed.device_communicators.rocm_comms.fusion import (
    ALL_REDUCE_FUSED_ADD_RMS_NORM_OP,
    ALL_REDUCE_RMS_NORM_OP,
)
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed

EPS = 1e-5


class ThreeAllReduces(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = [RMSNorm(hidden_size, EPS) for _ in range(3)]
        self.w = [torch.rand(hidden_size, hidden_size) / hidden_size for _ in range(2)]

    def forward(self, x: torch.Tensor):
        # relu first, so no graph input is a pattern argument directly
        a = tensor_model_parallel_all_reduce(torch.relu(x))
        y = self.norm[0](a)

        b = tensor_model_parallel_all_reduce(torch.mm(y, self.w[0]))
        y2 = self.norm[1](b) + b

        c = tensor_model_parallel_all_reduce(torch.mm(y2, self.w[1]))
        y3, resid = self.norm[2](c, y)
        return y3, resid


def _run(local_rank: int, world_size: int, port: int, hidden_size: int, tokens: int):
    set_random_seed(0)
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)
    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(port),
            "VLLM_ROCM_COMMS_BACKEND": "hip",
        }
    )
    init_distributed_environment()

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    )
    vllm_config.compilation_config.pass_config = PassConfig(
        fuse_allreduce_rms=True, eliminate_noops=True
    )
    vllm_config.device_config = DeviceConfig(device=torch.device("cuda"))
    vllm_config.parallel_config.rank = local_rank
    # A model name only to build a model config; nothing is loaded.
    vllm_config.model_config = ModelConfig(
        model="RedHatAI/Llama-3.2-1B-Instruct-FP8",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        seed=42,
    )
    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        fusion_pass = RocmHipAllReduceFusionPass(vllm_config)
        assert not fusion_pass.disabled, "the hip fusion pass refused to register"
        backend = TestBackend(
            NoOpEliminationPass(vllm_config),
            fusion_pass,
            FixFunctionalizationPass(vllm_config),
            PostCleanupPass(vllm_config),
        )
        model = ThreeAllReduces(hidden_size)
        x = torch.randn((tokens, hidden_size))
        compiled = torch.compile(model, backend=backend)
        compiled(x)

        eager = model(x)
        fused = compiled(x)
        for e, f in zip(eager, fused):
            torch.testing.assert_close(e, f, atol=2e-2, rtol=2e-2)

        assert fusion_pass.matched_count == 2, f"{fusion_pass.matched_count=}"
        backend.check_before_ops(
            [torch.ops.vllm.all_reduce.default], fully_replaced=False
        )
        backend.check_after_ops(
            [ALL_REDUCE_RMS_NORM_OP, ALL_REDUCE_FUSED_ADD_RMS_NORM_OP]
        )


@multi_gpu_test(num_gpus=2)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="rocm_comms is ROCm-only")
@pytest.mark.parametrize("hidden_size", [3584, 7168])
def test_rocm_hip_all_reduce_fusion_pass_replace(hidden_size: int) -> None:
    world_size = 2
    torch.multiprocessing.spawn(
        _run,
        args=(world_size, get_open_port(), hidden_size, 64),
        nprocs=world_size,
    )
