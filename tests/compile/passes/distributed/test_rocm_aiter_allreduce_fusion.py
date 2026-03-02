# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for RocmAiterAllReduceFusionPass.

Verifies that the fusion pass correctly replaces:
    all_reduce -> rocm_aiter_rms_norm -> rocm_aiter_per_tensor_quant
        -> rocm_per_tensor_float_w8a8_scaled_mm_impl
with the fused op, and that the fused output matches the unfused output.
"""

import pytest
import torch

from tests.compile.backend import TestBackend
from tests.utils import multi_gpu_test
from vllm._aiter_ops import rocm_aiter_ops  # noqa: F401 (registers ops)
import vllm.model_executor.kernels.linear.scaled_mm.rocm  # noqa: F401 (registers scaled_mm op)
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
from vllm.unfused_allreduce_add_rms_quant import (
    unfused_allreduce_add_rms_quant_gemm,
)
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed


class AllReduceFusionModel(torch.nn.Module):
    """Model with all_reduce -> RMSNorm -> per_tensor_quant -> scaled_mm.

    Mimics a transformer with 4 blocks. Block 1 always uses plain rms_norm
    (no residual). With use_residual=True, blocks 2-4 use
    fused_add_rms_norm with a residual connection, matching real transformer
    layers after the first.

    The pattern now extends through the scaled_mm GEMM, matching the
    fused op boundary.

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
        self.quant_dtype = current_platform.fp8_dtype()
        self.out_dtype = torch.float16

        # FP8 GEMM weights (N, K) transposed and per-tensor weight scales
        self.gemm_weight = [
            torch.rand(hidden_size, hidden_size,
                       dtype=torch.float32).to(self.quant_dtype).contiguous().t()
            for _ in range(4)
        ]
        self.weight_scale = [
            torch.rand(1, dtype=torch.float32) for _ in range(4)
        ]
        self.rms_weight = [
            torch.rand(hidden_size, dtype=torch.float16) for _ in range(4)
        ]
        self.scale = [
            torch.rand(1, dtype=torch.float32) for _ in range(4)
        ]

    def _block_no_residual(
        self, x: torch.Tensor, idx: int,
    ) -> tuple[torch.Tensor, None]:
        ar = tensor_model_parallel_all_reduce(x)
        rms = torch.ops.vllm.rocm_aiter_rms_norm(
            ar, self.rms_weight[idx], self.eps)
        q, s = torch.ops.vllm.rocm_aiter_per_tensor_quant(
            rms, self.quant_dtype, self.scale[idx])
        gemm = torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl(
            q, self.gemm_weight[idx], self.out_dtype,
            s, self.weight_scale[idx], None,
        )
        return gemm, None

    def _block_residual(
        self, x: torch.Tensor, resid: torch.Tensor, idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ar = tensor_model_parallel_all_reduce(x)
        rms, resid = torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add(
            ar, resid, self.rms_weight[idx], self.eps)
        q, s = torch.ops.vllm.rocm_aiter_per_tensor_quant(
            rms, self.quant_dtype, self.scale[idx])
        gemm = torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl(
            q, self.gemm_weight[idx], self.out_dtype,
            s, self.weight_scale[idx], None,
        )
        return gemm, resid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(x)

        if self.use_residual:
            # Block 1: plain rms_norm (creates initial residual from ar output)
            ar1 = tensor_model_parallel_all_reduce(z)
            resid = ar1
            rms1 = torch.ops.vllm.rocm_aiter_rms_norm(
                ar1, self.rms_weight[0], self.eps)
            q1, s1 = torch.ops.vllm.rocm_aiter_per_tensor_quant(
                rms1, self.quant_dtype, self.scale[0])
            z2 = torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl(
                q1, self.gemm_weight[0], self.out_dtype,
                s1, self.weight_scale[0], None,
            )

            # Blocks 2-4: fused_add_rms_norm with residual
            z3, resid = self._block_residual(z2, resid, 1)
            z4, resid = self._block_residual(z3, resid, 2)
            z5, resid = self._block_residual(z4, resid, 3)
        else:
            # All blocks: all_reduce -> rms_norm -> quant -> scaled_mm
            z2, _ = self._block_no_residual(z, 0)
            z3, _ = self._block_no_residual(z2, 1)
            z4, _ = self._block_no_residual(z3, 2)
            z5, _ = self._block_no_residual(z4, 3)

        return z5

    def ops_in_model_before(self) -> list:
        if self.use_residual:
            return [
                torch.ops.vllm.all_reduce.default,
                torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add.default,
                torch.ops.vllm.rocm_aiter_per_tensor_quant.default,
                torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl.default,
            ]
        else:
            return [
                torch.ops.vllm.all_reduce.default,
                torch.ops.vllm.rocm_aiter_rms_norm.default,
                torch.ops.vllm.rocm_aiter_per_tensor_quant.default,
                torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl.default,
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


def _get_batch_sizes(num_tokens: int) -> list[int]:
    """Return the graph capture schedule for a given num_tokens.

    Mirrors the production schedule from vllm/config/vllm.py
    (VLLM_CUDAGRAPH_BATCH_SIZES), filtered to sizes <= num_tokens.
    If num_tokens itself is not in the schedule, it is appended.
    """
    schedule = (
        [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16))
    )
    batch_sizes = [s for s in schedule if s <= num_tokens]
    if not batch_sizes or batch_sizes[-1] != num_tokens:
        batch_sizes.append(num_tokens)
    return batch_sizes


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
    hidden_size: int,
    dtype: torch.dtype,
    mode: str,
    batch_sizes: list[int],
) -> None:
    """Worker for test_rocm_aiter_fused_op_correctness.

    Exercises the fused op across the production batch size schedule,
    matching how vLLM actually uses these ops at runtime.

    The fused op now includes the GEMM: it takes BF16 in and produces
    BF16 GEMM output. The reference (unfused) does the same 4 ops
    separately.

    Args:
        mode: "eager" runs the fused op at each batch size sequentially,
            matching the warmup phase. "graph" does an eager warmup at
            max batch size, then captures one CUDA graph per batch size
            in sequence, then replays and verifies each -- matching the
            production CUDA graph capture loop.
        batch_sizes: List of batch sizes (token counts) to test.
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
        out_dtype = dtype

        rms_weight = torch.rand(hidden_size, dtype=dtype, device=device)
        rms_eps = 1e-5
        quant_scale = torch.rand(1, dtype=torch.float32, device=device)

        # GEMM params: weight (N, K) transposed, per-tensor weight scale
        gemm_weight = torch.rand(
            hidden_size, hidden_size, dtype=torch.float32, device=device,
        ).to(quant_dtype).contiguous().t()
        weight_scale = torch.rand(1, dtype=torch.float32, device=device)

        if dtype == torch.float16:
            ATOL, RTOL = (2e-3, 2e-3)
        else:
            ATOL, RTOL = (1e-2, 1e-2)

        for use_residual in [False, True]:
            checks: list[tuple] = []

            if mode == "eager":
                for M in batch_sizes:
                    input_data = torch.randn(
                        (M, hidden_size), dtype=dtype, device=device)
                    residual_data = (
                        torch.randn((M, hidden_size), dtype=dtype,
                                    device=device)
                        if use_residual else None
                    )

                    if use_residual:
                        result = (
                            torch.ops.vllm
                            .rocm_aiter_fused_allreduce_add_rms_quant(
                                input_data.clone(), residual_data.clone(),
                                rms_weight, rms_eps,
                                quant_dtype, group_name,
                                gemm_weight, weight_scale, out_dtype,
                            )
                        )
                        gemm_out, res_out = result
                    else:
                        result = (
                            torch.ops.vllm
                            .rocm_aiter_fused_allreduce_rms_quant(
                                input_data.clone(), rms_weight, rms_eps,
                                quant_dtype, group_name,
                                gemm_weight, weight_scale, out_dtype,
                            )
                        )
                        gemm_out = result[0]
                        res_out = None

                    ref_gemm, ref_res = unfused_allreduce_add_rms_quant_gemm(
                        input_data.clone(), rms_weight, rms_eps,
                        quant_scale, quant_dtype, group_name,
                        gemm_weight, weight_scale, out_dtype,
                        residual=residual_data.clone()
                        if residual_data is not None else None,
                    )
                    tag = f"M={M}, residual={use_residual}, mode=eager"
                    checks.append((
                        (gemm_out, res_out),
                        (ref_gemm, ref_res), tag
                    ))

            elif mode == "graph":
                max_M = max(batch_sizes)

                # Warmup at max batch size (eager), matching production
                warmup_input = torch.randn(
                    (max_M, hidden_size), dtype=dtype, device=device)
                warmup_residual = (
                    torch.randn((max_M, hidden_size), dtype=dtype,
                                device=device)
                    if use_residual else None
                )
                if use_residual:
                    torch.ops.vllm \
                        .rocm_aiter_fused_allreduce_add_rms_quant(
                            warmup_input, warmup_residual,
                            rms_weight, rms_eps,
                            quant_dtype, group_name,
                            gemm_weight, weight_scale, out_dtype,
                        )
                else:
                    torch.ops.vllm.rocm_aiter_fused_allreduce_rms_quant(
                        warmup_input, rms_weight, rms_eps,
                        quant_dtype, group_name,
                        gemm_weight, weight_scale, out_dtype,
                    )
                torch.cuda.synchronize()

                # Sequential graph capture at each batch size
                captured = []

                with vllm_graph_capture(device=device) as ctx:
                    for M in batch_sizes:
                        input_capture = torch.randn(
                            (M, hidden_size), dtype=dtype, device=device)
                        residual_capture = (
                            torch.randn((M, hidden_size), dtype=dtype,
                                        device=device)
                            if use_residual else None
                        )

                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=ctx.stream):
                            if use_residual:
                                cap = (
                                    torch.ops.vllm
                                    .rocm_aiter_fused_allreduce_add_rms_quant(
                                        input_capture, residual_capture,
                                        rms_weight, rms_eps,
                                        quant_dtype, group_name,
                                        gemm_weight, weight_scale, out_dtype,
                                    )
                                )
                                cap_gemm, cap_res = cap
                            else:
                                cap = (
                                    torch.ops.vllm
                                    .rocm_aiter_fused_allreduce_rms_quant(
                                        input_capture, rms_weight, rms_eps,
                                        quant_dtype, group_name,
                                        gemm_weight, weight_scale, out_dtype,
                                    )
                                )
                                cap_gemm = cap[0]
                                cap_res = None

                        captured.append((
                            graph, input_capture, residual_capture,
                            (cap_gemm, cap_res), M
                        ))

                # Replay and verify each captured graph multiple
                # times with fresh data to test that the inlined
                # barrier epoch advances correctly across repeated
                # CUDA graph replays (production replays many times).
                graph_replays = 10
                for idx, (graph, input_t, residual_t, outputs, M) in \
                        enumerate(captured):
                    cap_gemm, cap_res = outputs

                    for replay_i in range(graph_replays):
                        input_fresh = torch.randn(
                            (M, hidden_size), dtype=dtype, device=device)
                        input_t.copy_(input_fresh)
                        if use_residual:
                            residual_fresh = torch.randn(
                                (M, hidden_size), dtype=dtype,
                                device=device)
                            residual_t.copy_(residual_fresh)
                        else:
                            residual_fresh = None

                        graph.replay()
                        torch.cuda.synchronize()

                        ref_gemm, ref_res = \
                            unfused_allreduce_add_rms_quant_gemm(
                                input_fresh.clone(), rms_weight, rms_eps,
                                quant_scale, quant_dtype, group_name,
                                gemm_weight, weight_scale, out_dtype,
                                residual=residual_fresh.clone()
                                if residual_fresh is not None else None,
                            )
                        tag = (
                            f"graph {idx + 1}/{len(captured)}, "
                            f"replay {replay_i + 1}/{graph_replays}, "
                            f"M={M}, residual={use_residual}"
                        )
                        checks.append((
                            (cap_gemm, cap_res),
                            (ref_gemm, ref_res), tag
                        ))

            # Epilogue: compare outputs against reference
            for (gemm_out, res_out), \
                (ref_gemm, ref_res), \
                    check_tag in checks:
                M_out = gemm_out.shape[0]
                N_out = gemm_out.shape[1]

                # GEMM output shape
                assert gemm_out.shape == ref_gemm.shape, (
                    f"gemm_out shape {gemm_out.shape} != "
                    f"ref shape {ref_gemm.shape} ({check_tag})"
                )

                torch.testing.assert_close(
                    gemm_out, ref_gemm, atol=ATOL, rtol=RTOL,
                    msg=f"gemm_out mismatch ({check_tag})",
                )

                if use_residual:
                    assert res_out is not None and ref_res is not None
                    assert res_out.shape == (M_out, hidden_size), (
                        f"residual_out shape {res_out.shape}, "
                        f"expected ({M_out}, {hidden_size}) ({check_tag})"
                    )
                    torch.testing.assert_close(
                        res_out, ref_res, atol=ATOL, rtol=RTOL,
                        msg=f"residual_out mismatch ({check_tag})",
                    )
                else:
                    assert res_out is None and ref_res is None

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
@pytest.mark.timeout(120)
@pytest.mark.parametrize("mode", ["eager", "graph"])
@pytest.mark.parametrize("hidden_size", [2048, 4096, 7168, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("num_tokens", [1, 32, 128, 512, 576])
@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.hip),
    reason="ROCm AITER fusion pass only runs on ROCm (HIP)",
)
def test_rocm_aiter_fused_op_correctness(
    mode: str,
    hidden_size: int,
    dtype: torch.dtype,
    num_tokens: int,
    num_processes: int = 8,
):
    """Compare fused torch ops against unfused individual ops.

    Exercises the fused op (which now includes the GEMM) across the
    CUDA graph capture schedule that vLLM would generate for the given
    num_tokens, matching real usage. Both residual and non-residual
    variants are tested within each invocation.

    In eager mode, the op is called at each batch size sequentially.
    In graph mode, an eager warmup runs at the max batch size, then
    one CUDA graph is captured per batch size in sequence, and each
    graph is replayed 10 times with fresh data to verify barrier
    epoch correctness across repeated replays (matching production).
    """
    batch_sizes = _get_batch_sizes(num_tokens)

    torch.multiprocessing.spawn(
        _run_fused_op_correctness_test,
        args=(
            num_processes,
            hidden_size,
            dtype,
            mode,
            batch_sizes,
        ),
        nprocs=num_processes,
    )
