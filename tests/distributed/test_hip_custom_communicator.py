# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HipCustomCommunicator (ROCm): the TP all-reduce and the ops it fuses, at
Kimi-K3's calls (the hidden row, decode batches and a prefill chunk), eager and
under a CUDA graph, exact and INT4, against RCCL's all-reduce and vLLM's own
norms and AttnRes."""

import pytest
import ray
import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.hip_custom_communicator import (
    _TABLE,
    AllReduceAddAttnResRmsNormConfig,
    AllReduceAddRmsNormConfig,
    HipCustomCommunicator,
    TunedTable,
    parse_tuned_table,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.platforms import current_platform

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-only HipCustomCommunicator tests"
)

HIDDEN = 7168
# Kimi-K3's calls: decode batches (fewer rows than ranks among them) and a prefill
# chunk, past where the quantized table moves INT4.
NUM_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 512, 2048]
MAX_TOKENS = 2048
EPS = 1e-5
ATTN_RES_EPS = 1e-6
# INT4's error against the exact op, end to end (relative RMS): the format's
# rounding (a group's largest value / 7 a step) summed over the ranks. A one-shot
# quantizes once (each rank's input); a two-shot twice (the inputs for the
# reduce-scatter, the sums for the all-gather), as QuickReduce's two-shot does:
# its bar the looser (measured: AttnRes's two-shots 0.19 at 16 tokens, world 8,
# its one-shot 0.10).
INT4_ONE_SHOT_RELATIVE_RMS = 0.15
INT4_TWO_SHOT_RELATIVE_RMS = 0.25


def no_table_scans(m: pytest.MonkeyPatch) -> None:
    """After open, a plain call reads its config off the picks decided at open
    (`PlainPicks`): scanning the tuned table on a call fails the test (2026-10-29: a
    scan on every call cost a worker ~1 s of host time in a 65 s window)."""

    def scanned(*args, **kwargs):
        raise AssertionError("a plain call scanned the tuned table")

    m.setattr(TunedTable, "nearest", scanned)


def int4_bar(config) -> float:
    """The INT4 bar of a call that ran `config`: a two-shot's where it moved INT4 in
    two shots (a split's all-reduce among them), else a one-shot's."""
    two_shot = config.quantization == "int4" and config.algorithm.startswith("two_shot")
    return INT4_TWO_SHOT_RELATIVE_RMS if two_shot else INT4_ONE_SHOT_RELATIVE_RMS


@ray.remote(num_gpus=1, max_calls=1)
def all_reduce_ranks(monkeypatch: pytest.MonkeyPatch, tp_size, pp_size, rank, port):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group()
        # RCCL up first, as vLLM's startup does.
        dist.all_reduce(torch.zeros(1, device=device), group=group.device_group)
        torch.accelerator.synchronize()
        comm = HipCustomCommunicator(
            group.cpu_group,
            group.device_group,
            device,
            max_rows=MAX_TOKENS,
            max_cols=HIDDEN,
        )
        assert not comm.disabled, comm.lifecycle
        no_table_scans(m)
        try:
            for num_tokens in NUM_TOKENS:
                for dtype in [torch.bfloat16, torch.float16]:
                    # Small integers: every sum is exact in the dtype, so the exact
                    # all-reduce matches RCCL's bit for bit.
                    g = torch.Generator(device=device).manual_seed(rank)
                    x = torch.randint(
                        -8, 8, (num_tokens, HIDDEN), generator=g, device=device
                    ).to(dtype)
                    want = x.clone()
                    dist.all_reduce(want, group=group.device_group)
                    assert comm.should_allreduce(x)

                    # Eager.
                    out = comm.all_reduce(x)
                    if comm.arguments.quantization == "int4":
                        err = (out.float() - want.float()).square().mean().sqrt()
                        rms = want.float().square().mean().sqrt()
                        assert err / rms <= int4_bar(
                            comm.tuned("all_reduce", num_tokens, HIDDEN, None)
                        ), (num_tokens, dtype)
                    else:
                        assert torch.equal(out, want), (num_tokens, dtype)

                    # A CUDA graph, replayed on fresh inputs.
                    graph = torch.cuda.CUDAGraph()
                    with comm.capture():
                        comm.all_reduce(x)  # vLLM's warmup: launches nothing
                        with torch.cuda.graph(graph):
                            out = comm.all_reduce(x)
                    x.copy_(x + 1)
                    want = x.clone()
                    dist.all_reduce(want, group=group.device_group)
                    graph.replay()
                    torch.accelerator.synchronize()
                    if comm.arguments.quantization == "int4":
                        err = (out.float() - want.float()).square().mean().sqrt()
                        rms = want.float().square().mean().sqrt()
                        assert err / rms <= int4_bar(
                            comm.tuned("all_reduce", num_tokens, HIDDEN, None)
                        ), (num_tokens, dtype)
                    else:
                        assert torch.equal(out, want), (num_tokens, dtype)
        finally:
            comm.close()


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("quantization", ["NONE", "INT4"])
def test_all_reduce(monkeypatch: pytest.MonkeyPatch, tp_size, quantization):
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    monkeypatch.setenv("VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION", quantization)
    multi_process_parallel(monkeypatch, tp_size, 1, all_reduce_ranks)


@ray.remote(num_gpus=1, max_calls=1)
def all_reduce_add_rms_norm_no_residual_ranks(
    monkeypatch: pytest.MonkeyPatch, tp_size, pp_size, rank, port
):
    from vllm.ir.ops import rms_norm

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group()
        dist.all_reduce(torch.zeros(1, device=device), group=group.device_group)
        torch.accelerator.synchronize()
        comm = HipCustomCommunicator(
            group.cpu_group,
            group.device_group,
            device,
            max_rows=MAX_TOKENS,
            max_cols=HIDDEN,
        )
        assert not comm.disabled, comm.lifecycle
        no_table_scans(m)
        try:
            for num_tokens in NUM_TOKENS:
                for dtype in [torch.bfloat16, torch.float16]:
                    g = torch.Generator(device=device).manual_seed(rank)
                    x = torch.randint(
                        -8, 8, (num_tokens, HIDDEN), generator=g, device=device
                    ).to(dtype)
                    weight = torch.empty(HIDDEN, dtype=dtype, device=device)
                    weight.normal_(mean=1.0, std=0.1, generator=g.manual_seed(0))
                    summed = x.clone()
                    dist.all_reduce(summed, group=group.device_group)
                    want = rms_norm(summed, weight, EPS)
                    assert comm.should_allreduce_add_rms_norm(x, weight)

                    # Eager: the table's pick, then each algorithm of the fused
                    # kernel forced at the table's build of it in this mode (one
                    # it has no build of is skipped).
                    forced = [
                        comm.kernel_config(
                            "all_reduce_add_rms_norm",
                            algorithm,
                            num_tokens,
                            HIDDEN,
                            comm.arguments.quantization,
                        )
                        for algorithm in AllReduceAddRmsNormConfig.algorithms
                    ]
                    for config in [None, *(c for c in forced if c is not None)]:
                        out = comm.all_reduce_add_rms_norm(
                            x, weight, EPS, config=config
                        )
                        if comm.arguments.quantization == "int4":
                            err = (out.float() - want.float()).square().mean().sqrt()
                            rms = want.float().square().mean().sqrt()
                            assert err / rms <= int4_bar(
                                config
                                if config is not None
                                else comm.tuned(
                                    "all_reduce_add_rms_norm", num_tokens, HIDDEN, False
                                )
                            ), (
                                num_tokens,
                                dtype,
                                config,
                            )
                        else:
                            torch.testing.assert_close(
                                out, want, atol=2e-2, rtol=2e-2, msg=str(config)
                            )

                    # A CUDA graph, replayed on fresh inputs.
                    graph = torch.cuda.CUDAGraph()
                    with comm.capture():
                        comm.all_reduce_add_rms_norm(x, weight, EPS)
                        with torch.cuda.graph(graph):
                            out = comm.all_reduce_add_rms_norm(x, weight, EPS)
                    x.copy_(x + 1)
                    summed = x.clone()
                    dist.all_reduce(summed, group=group.device_group)
                    want = rms_norm(summed, weight, EPS)
                    graph.replay()
                    torch.accelerator.synchronize()
                    if comm.arguments.quantization == "int4":
                        err = (out.float() - want.float()).square().mean().sqrt()
                        rms = want.float().square().mean().sqrt()
                        assert err / rms <= int4_bar(
                            comm.tuned(
                                "all_reduce_add_rms_norm", num_tokens, HIDDEN, False
                            )
                        ), (num_tokens, dtype)
                    else:
                        torch.testing.assert_close(out, want, atol=2e-2, rtol=2e-2)
        finally:
            comm.close()


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("quantization", ["NONE", "INT4"])
def test_all_reduce_add_rms_norm_no_residual(
    monkeypatch: pytest.MonkeyPatch, tp_size, quantization
):
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    monkeypatch.setenv("VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION", quantization)
    multi_process_parallel(
        monkeypatch, tp_size, 1, all_reduce_add_rms_norm_no_residual_ranks
    )


@ray.remote(num_gpus=1, max_calls=1)
def all_reduce_add_rms_norm_ranks(
    monkeypatch: pytest.MonkeyPatch, tp_size, pp_size, rank, port
):
    from vllm.ir.ops import fused_add_rms_norm

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group()
        dist.all_reduce(torch.zeros(1, device=device), group=group.device_group)
        torch.accelerator.synchronize()
        comm = HipCustomCommunicator(
            group.cpu_group,
            group.device_group,
            device,
            max_rows=MAX_TOKENS,
            max_cols=HIDDEN,
        )
        assert not comm.disabled, comm.lifecycle
        no_table_scans(m)
        try:
            for num_tokens in NUM_TOKENS:
                for dtype in [torch.bfloat16, torch.float16]:
                    g = torch.Generator(device=device).manual_seed(rank)
                    x = torch.randint(
                        -8, 8, (num_tokens, HIDDEN), generator=g, device=device
                    ).to(dtype)
                    # The residual and the weight the same on every rank.
                    g.manual_seed(0)
                    residual = torch.randint(
                        -8, 8, (num_tokens, HIDDEN), generator=g, device=device
                    ).to(dtype)
                    weight = torch.empty(HIDDEN, dtype=dtype, device=device)
                    weight.normal_(mean=1.0, std=0.1, generator=g)
                    # The reference first: fused_add_rms_norm writes its residual.
                    summed = x.clone()
                    dist.all_reduce(summed, group=group.device_group)
                    want_out, want_residual = fused_add_rms_norm(
                        summed, residual.clone(), weight, EPS
                    )
                    assert comm.should_allreduce_add_rms_norm(x, weight, residual)

                    # Eager: the table's pick, then each algorithm of the fused
                    # kernel forced at the table's build of it in this mode (one
                    # it has no build of is skipped).
                    forced = [
                        comm.kernel_config(
                            "all_reduce_add_rms_norm",
                            algorithm,
                            num_tokens,
                            HIDDEN,
                            comm.arguments.quantization,
                        )
                        for algorithm in AllReduceAddRmsNormConfig.algorithms
                    ]
                    for config in [None, *(c for c in forced if c is not None)]:
                        out, residual_out = comm.all_reduce_add_rms_norm(
                            x, weight, EPS, residual=residual, config=config
                        )
                        for got, want in [
                            (out, want_out),
                            (residual_out, want_residual),
                        ]:
                            if comm.arguments.quantization == "int4":
                                err = (
                                    (got.float() - want.float()).square().mean().sqrt()
                                )
                                rms = want.float().square().mean().sqrt()
                                assert err / rms <= int4_bar(
                                    config
                                    if config is not None
                                    else comm.tuned(
                                        "all_reduce_add_rms_norm",
                                        num_tokens,
                                        HIDDEN,
                                        True,
                                    )
                                ), (
                                    num_tokens,
                                    dtype,
                                    config,
                                )
                            else:
                                torch.testing.assert_close(
                                    got, want, atol=2e-2, rtol=2e-2, msg=str(config)
                                )

                    # A CUDA graph, replayed on fresh inputs.
                    graph = torch.cuda.CUDAGraph()
                    with comm.capture():
                        comm.all_reduce_add_rms_norm(x, weight, EPS, residual=residual)
                        with torch.cuda.graph(graph):
                            out, residual_out = comm.all_reduce_add_rms_norm(
                                x, weight, EPS, residual=residual
                            )
                    x.copy_(x + 1)
                    summed = x.clone()
                    dist.all_reduce(summed, group=group.device_group)
                    want_out, want_residual = fused_add_rms_norm(
                        summed, residual.clone(), weight, EPS
                    )
                    graph.replay()
                    torch.accelerator.synchronize()
                    for got, want in [(out, want_out), (residual_out, want_residual)]:
                        if comm.arguments.quantization == "int4":
                            err = (got.float() - want.float()).square().mean().sqrt()
                            rms = want.float().square().mean().sqrt()
                            assert err / rms <= int4_bar(
                                comm.tuned(
                                    "all_reduce_add_rms_norm", num_tokens, HIDDEN, True
                                )
                            ), (num_tokens, dtype)
                        else:
                            torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)
        finally:
            comm.close()


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("quantization", ["NONE", "INT4"])
def test_all_reduce_add_rms_norm(
    monkeypatch: pytest.MonkeyPatch, tp_size, quantization
):
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    monkeypatch.setenv("VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION", quantization)
    multi_process_parallel(monkeypatch, tp_size, 1, all_reduce_add_rms_norm_ranks)


@ray.remote(num_gpus=1, max_calls=1)
def all_reduce_add_attn_res_rms_norm_ranks(
    monkeypatch: pytest.MonkeyPatch, tp_size, pp_size, rank, port
):
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group()
        dist.all_reduce(torch.zeros(1, device=device), group=group.device_group)
        torch.accelerator.synchronize()
        comm = HipCustomCommunicator(
            group.cpu_group,
            group.device_group,
            device,
            max_rows=MAX_TOKENS,
            max_cols=HIDDEN,
        )
        assert not comm.disabled, comm.lifecycle
        no_table_scans(m)
        dtype = torch.bfloat16
        try:
            for num_tokens in NUM_TOKENS:
                # Kimi-K3's AttnRes layers: 10 source blocks, one that updates the
                # prefix (4 blocks) and a block-write layer whose sum starts it.
                for has_prefix, num_blocks, write_index in [
                    (True, 4, -1),
                    (False, 4, 4),
                ]:
                    g = torch.Generator(device=device).manual_seed(rank)
                    x = torch.randint(
                        -8, 8, (num_tokens, HIDDEN), generator=g, device=device
                    ).to(dtype)
                    # Everything else the same on every rank.
                    g.manual_seed(0)
                    prefix = (
                        torch.randint(
                            -8, 8, (num_tokens, HIDDEN), generator=g, device=device
                        ).to(dtype)
                        if has_prefix
                        else None
                    )
                    blocks = torch.randint(
                        -8, 8, (num_tokens, 10, HIDDEN), generator=g, device=device
                    ).to(dtype)
                    norm_weight = torch.empty(HIDDEN, dtype=dtype, device=device)
                    norm_weight.normal_(mean=1.0, std=0.1, generator=g)
                    qk_weight = torch.empty(HIDDEN, dtype=dtype, device=device)
                    qk_weight.normal_(mean=0.0, std=0.1, generator=g)
                    output_norm_weight = torch.empty(HIDDEN, dtype=dtype, device=device)
                    output_norm_weight.normal_(mean=1.0, std=0.1, generator=g)

                    # The reference first: AttnRes writes its prefix and its block.
                    summed = x.clone()
                    dist.all_reduce(summed, group=group.device_group)
                    want_prefix = summed.clone() if prefix is None else prefix.clone()
                    want_blocks = blocks.clone()
                    want_out = attn_res(
                        want_prefix,
                        None if prefix is None else summed,
                        want_blocks,
                        norm_weight,
                        qk_weight,
                        output_norm_weight,
                        num_blocks,
                        write_index,
                        ATTN_RES_EPS,
                        EPS,
                    )
                    assert comm.should_allreduce_add_attn_res_rms_norm(x)

                    # Eager: the table's pick, then each algorithm of the fused
                    # kernel forced at the table's build of it in this mode (one
                    # it has no build of is skipped), each on its own copy of the
                    # blocks (AttnRes writes one).
                    forced = [
                        comm.kernel_config(
                            "all_reduce_add_attn_res_rms_norm",
                            algorithm,
                            num_tokens,
                            HIDDEN,
                            comm.arguments.quantization,
                        )
                        for algorithm in AllReduceAddAttnResRmsNormConfig.algorithms
                    ]
                    original_blocks = blocks.clone()
                    for config in [None, *(c for c in forced if c is not None)]:
                        blocks.copy_(original_blocks)
                        out_prefix, out = comm.all_reduce_add_attn_res_rms_norm(
                            x,
                            prefix,
                            blocks,
                            norm_weight,
                            qk_weight,
                            output_norm_weight,
                            num_blocks,
                            write_index,
                            ATTN_RES_EPS,
                            EPS,
                            config=config,
                        )
                        for got, want in [
                            (out, want_out),
                            (out_prefix, want_prefix),
                            (blocks, want_blocks),
                        ]:
                            if comm.arguments.quantization == "int4":
                                err = (
                                    (got.float() - want.float()).square().mean().sqrt()
                                )
                                rms = want.float().square().mean().sqrt()
                                assert err / rms <= int4_bar(
                                    config
                                    if config is not None
                                    else comm.tuned(
                                        "all_reduce_add_attn_res_rms_norm",
                                        num_tokens,
                                        HIDDEN,
                                        None,
                                    )
                                ), (
                                    num_tokens,
                                    dtype,
                                    config,
                                )
                            else:
                                torch.testing.assert_close(
                                    got, want, atol=2e-2, rtol=2e-2, msg=str(config)
                                )
                    blocks.copy_(original_blocks)

                    # A CUDA graph, replayed on fresh inputs.
                    graph = torch.cuda.CUDAGraph()
                    with comm.capture():
                        comm.all_reduce_add_attn_res_rms_norm(
                            x,
                            prefix,
                            blocks,
                            norm_weight,
                            qk_weight,
                            output_norm_weight,
                            num_blocks,
                            write_index,
                            ATTN_RES_EPS,
                            EPS,
                        )
                        with torch.cuda.graph(graph):
                            out_prefix, out = comm.all_reduce_add_attn_res_rms_norm(
                                x,
                                prefix,
                                blocks,
                                norm_weight,
                                qk_weight,
                                output_norm_weight,
                                num_blocks,
                                write_index,
                                ATTN_RES_EPS,
                                EPS,
                            )
                    x.copy_(x + 1)
                    summed = x.clone()
                    dist.all_reduce(summed, group=group.device_group)
                    want_prefix = summed.clone() if prefix is None else prefix.clone()
                    want_blocks = blocks.clone()
                    want_out = attn_res(
                        want_prefix,
                        None if prefix is None else summed,
                        want_blocks,
                        norm_weight,
                        qk_weight,
                        output_norm_weight,
                        num_blocks,
                        write_index,
                        ATTN_RES_EPS,
                        EPS,
                    )
                    graph.replay()
                    torch.accelerator.synchronize()
                    for got, want in [
                        (out, want_out),
                        (out_prefix, want_prefix),
                        (blocks, want_blocks),
                    ]:
                        if comm.arguments.quantization == "int4":
                            err = (got.float() - want.float()).square().mean().sqrt()
                            rms = want.float().square().mean().sqrt()
                            assert err / rms <= int4_bar(
                                comm.tuned(
                                    "all_reduce_add_attn_res_rms_norm",
                                    num_tokens,
                                    HIDDEN,
                                    None,
                                )
                            ), (num_tokens, dtype)
                        else:
                            torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)
        finally:
            comm.close()


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("quantization", ["NONE", "INT4"])
def test_all_reduce_add_attn_res_rms_norm(
    monkeypatch: pytest.MonkeyPatch, tp_size, quantization
):
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    monkeypatch.setenv("VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION", quantization)
    multi_process_parallel(
        monkeypatch, tp_size, 1, all_reduce_add_attn_res_rms_norm_ranks
    )


@ray.remote(num_gpus=1, max_calls=1)
def refused_calls_ranks(monkeypatch: pytest.MonkeyPatch, tp_size, pp_size, rank, port):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group()
        dist.all_reduce(torch.zeros(1, device=device), group=group.device_group)
        torch.accelerator.synchronize()
        comm = HipCustomCommunicator(
            group.cpu_group,
            group.device_group,
            device,
            max_rows=MAX_TOKENS,
            max_cols=HIDDEN,
        )
        assert not comm.disabled, comm.lifecycle
        try:
            # Far past the largest call it was sized for: vLLM takes another path.
            past = torch.ones(
                16 * MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device
            )
            assert not comm.should_allreduce(past)

            # Not [rows, cols]: refused by name before anything launches.
            with pytest.raises(RuntimeError, match="^not_two_d:"):
                comm.all_reduce(
                    torch.ones(2, 4, HIDDEN, dtype=torch.bfloat16, device=device)
                )

            # A strided view of the hidden row: run where admitted, and summed right.
            x = torch.ones(4, 2 * HIDDEN, dtype=torch.bfloat16, device=device)[
                :, :HIDDEN
            ]
            want = x.clone()
            dist.all_reduce(want, group=group.device_group)
            if comm.should_allreduce(x):
                assert torch.equal(comm.all_reduce(x), want)
        finally:
            comm.close()


def test_refused_calls(monkeypatch: pytest.MonkeyPatch):
    if torch.accelerator.device_count() < 2:
        pytest.skip("Not enough GPUs to run the test.")
    monkeypatch.setenv("VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION", "NONE")
    multi_process_parallel(monkeypatch, 2, 1, refused_calls_ranks)


@ray.remote(num_gpus=1, max_calls=1)
def tuned_table_ranks(monkeypatch: pytest.MonkeyPatch, tp_size, pp_size, rank, port):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group()
        dist.all_reduce(torch.zeros(1, device=device), group=group.device_group)
        torch.accelerator.synchronize()
        # THE SHIPPED TABLE, as a communicator reads it at open (its device's arch),
        # in each mode: every tuned op has a pick at Kimi-K3's decode and prefill
        # sizes at this world.
        for quantization in ["exact", "int4"]:
            comm = HipCustomCommunicator(
                group.cpu_group,
                group.device_group,
                device,
                max_rows=MAX_TOKENS,
                max_cols=HIDDEN,
                quantization=quantization,
            )
            assert not comm.disabled, comm.lifecycle
            try:
                # The norm op's picks with a residual and without.
                for op, residual in [
                    ("all_reduce", None),
                    ("all_reduce_add_rms_norm", False),
                    ("all_reduce_add_rms_norm", True),
                    ("all_reduce_add_attn_res_rms_norm", None),
                ]:
                    for num_tokens in [1, MAX_TOKENS]:
                        config = comm.tuned(op, num_tokens, HIDDEN, residual)
                        assert config is not None, (
                            op,
                            residual,
                            quantization,
                            num_tokens,
                        )
            finally:
                comm.close()


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_tuned_table(monkeypatch: pytest.MonkeyPatch, tp_size):
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    multi_process_parallel(monkeypatch, tp_size, 1, tuned_table_ranks)


def test_plain_picks_are_nearest():
    """Every row count's pick decided at open (`TunedTable.by_rows`) is the one a scan
    of the table finds (`TunedTable.nearest`), in each mode at each world the shipped
    table holds, for each op, at its tuned widths and at widths it was not tuned at."""
    table = parse_tuned_table(_TABLE, "gfx950")
    assert isinstance(table, TunedTable), table
    max_rows = 4096
    for quantization, mode in [("exact", "exact"), ("int4", "quantized")]:
        for world in sorted({i.world for i in table.picks}):
            for op, residual in [
                ("all_reduce", None),
                ("all_reduce_add_rms_norm", False),
                ("all_reduce_add_rms_norm", True),
                ("all_reduce_add_attn_res_rms_norm", None),
            ]:
                tuned = {
                    i.hidden
                    for i in table.picks
                    if i.op == op
                    and i.residual == residual
                    and i.mode == mode
                    and i.world == world
                }
                if not tuned:
                    continue
                for cols in sorted(tuned | {4096, 5120}):
                    picks = table.by_rows(
                        op, residual, quantization, world, cols, max_rows
                    )
                    assert len(picks) == max_rows + 1
                    for rows in [*range(0, 130), *range(130, max_rows + 1, 37)]:
                        want = table.nearest(
                            op, residual, quantization, world, rows, cols
                        )[1]
                        assert picks[rows] == want, (op, residual, world, cols, rows)
