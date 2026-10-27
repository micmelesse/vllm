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
    CommunicatorError,
    Error,
    HipCustomCommunicator,
    parse_tuned_configs,
    supported,
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
TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 512, 2048)
MAX_TOKENS = 2048
EPS = 1e-5
ATTN_RES_EPS = 1e-6
# AttnRes as Kimi-K3 runs it: 10 source blocks, a layer that updates the prefix
# and a block-write layer whose sum starts it.
ATTN_RES_SOURCES = 10
ATTN_RES_LAYERS = ((True, 4, -1), (False, 4, 4))
# INT4's error against the exact op, end to end: the format's rounding (a group's
# largest value / 7 a step) summed over the ranks.
INT4_RELATIVE_RMS = 0.15
OPS = (
    "all_reduce",
    "all_reduce_rms_norm",
    "all_reduce_add_rms_norm",
    "all_reduce_add_attn_res_rms_norm",
)


def _ints(shape: tuple[int, ...], dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Small integers, so every rank's sum is exact in the dtype: the exact
    all-reduce matches RCCL's bit for bit."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(-8, 8, shape, generator=g, device="cuda").to(dtype)


def _weight(dtype: torch.dtype, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.rand(HIDDEN, generator=g, device="cuda").to(dtype)


def _inputs(op: str, rows: int, dtype: torch.dtype, rank: int, layer) -> dict:
    """This rank's input `x` and the op's other arguments (the same on every rank)."""
    args = {"x": _ints((rows, HIDDEN), dtype, 1000 * rows + rank)}
    if op in ("all_reduce_rms_norm", "all_reduce_add_rms_norm"):
        args["weight"] = _weight(dtype, 1)
    if op == "all_reduce_add_rms_norm":
        args["residual"] = _ints((rows, HIDDEN), dtype, 7)
    if op == "all_reduce_add_attn_res_rms_norm":
        has_prefix, num_blocks, write_index = layer
        args["prefix"] = _ints((rows, HIDDEN), dtype, 8) if has_prefix else None
        args["blocks"] = _ints((rows, ATTN_RES_SOURCES, HIDDEN), dtype, 9)
        args["norm_weight"] = _weight(dtype, 2)
        args["qk_weight"] = _weight(dtype, 3)
        args["output_norm_weight"] = _weight(dtype, 4)
        args["num_blocks"] = num_blocks
        args["write_index"] = write_index
    return args


def _call(comm: HipCustomCommunicator, op: str, a: dict) -> list[torch.Tensor]:
    """The op as vLLM calls it (nothing forced: the tuned table's pick)."""
    if op == "all_reduce":
        return [comm.all_reduce(a["x"])]
    if op == "all_reduce_rms_norm":
        return [comm.all_reduce_rms_norm(a["x"], a["weight"], EPS)]
    if op == "all_reduce_add_rms_norm":
        return list(
            comm.all_reduce_add_rms_norm(a["x"], a["residual"], a["weight"], EPS)
        )
    prefix, out = comm.all_reduce_add_attn_res_rms_norm(
        a["x"],
        a["prefix"],
        a["blocks"],
        a["norm_weight"],
        a["qk_weight"],
        a["output_norm_weight"],
        a["num_blocks"],
        a["write_index"],
        ATTN_RES_EPS,
        EPS,
    )
    return [out, prefix, a["blocks"]]


def _should(comm: HipCustomCommunicator, op: str, a: dict) -> bool:
    if op == "all_reduce":
        return comm.should_allreduce(a["x"])
    if op == "all_reduce_rms_norm":
        return comm.should_allreduce_rms_norm(a["x"], a["weight"])
    if op == "all_reduce_add_rms_norm":
        return comm.should_allreduce_add_rms_norm(a["x"], a["weight"])
    return comm.should_allreduce_add_attn_res_rms_norm(a["x"])


def _reference(op: str, summed: torch.Tensor, a: dict) -> list[torch.Tensor]:
    """What the op replaces, on RCCL's sum: vLLM's own norms and AttnRes."""
    import vllm.ir.ops
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res

    if op == "all_reduce":
        return [summed]
    if op == "all_reduce_rms_norm":
        return [vllm.ir.ops.rms_norm(summed, a["weight"], EPS)]
    if op == "all_reduce_add_rms_norm":
        out, residual = vllm.ir.ops.fused_add_rms_norm(
            summed, a["residual"].clone(), a["weight"], EPS
        )
        return [out, residual]
    prefix = summed.clone() if a["prefix"] is None else a["prefix"].clone()
    blocks = a["blocks"].clone()
    out = attn_res(
        prefix,
        None if a["prefix"] is None else summed,
        blocks,
        a["norm_weight"],
        a["qk_weight"],
        a["output_norm_weight"],
        a["num_blocks"],
        a["write_index"],
        ATTN_RES_EPS,
        EPS,
    )
    return [out, prefix, blocks]


def _check(op: str, quantized: bool, outs, refs, where: str) -> None:
    """Exact: the all-reduce bit for bit, a fused op's outputs within a rounding of
    the dtype carried through its norm. INT4: within the format's error."""
    for got, want in zip(outs, refs):
        if quantized:
            err = (got.float() - want.float()).square().mean().sqrt()
            scale = want.float().square().mean().sqrt().clamp_min(1e-6)
            assert err / scale <= INT4_RELATIVE_RMS, f"{where}: {err / scale:.3f}"
        elif op == "all_reduce":
            assert torch.equal(got, want), where
        else:
            torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2, msg=where)


def _cases(dtypes: dict[str, tuple[torch.dtype, ...]]):
    for op in OPS:
        layers = ATTN_RES_LAYERS if "attn_res" in op else (None,)
        for dtype in dtypes[op]:
            for layer in layers:
                for rows in TOKENS:
                    yield op, dtype, layer, rows


def _open(rank: int, tp_size: int, pp_size: int, port: int) -> HipCustomCommunicator:
    """This rank's communicator over the TP group, sized for the largest call,
    quantizing as the environment asks (VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION)."""
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, port)
    ensure_model_parallel_initialized(tp_size, pp_size)
    group = get_tp_group()
    # RCCL first, as vLLM's startup does: torch builds it at its first collective.
    dist.all_reduce(torch.zeros(1, device=device), group=group.device_group)
    torch.accelerator.synchronize()
    comm = HipCustomCommunicator(
        group.cpu_group,
        group.device_group,
        device,
        max_rows=MAX_TOKENS,
        max_cols=HIDDEN,
    )
    assert not comm.disabled, comm.state
    return comm


DTYPES = {
    "all_reduce": (torch.bfloat16, torch.float16),
    "all_reduce_rms_norm": (torch.bfloat16, torch.float16),
    "all_reduce_add_rms_norm": (torch.bfloat16, torch.float16),
    "all_reduce_add_attn_res_rms_norm": (torch.bfloat16,),
}


@ray.remote(num_gpus=1, max_calls=1)
def eager_ops(monkeypatch: pytest.MonkeyPatch, tp_size, pp_size, rank, port):
    with monkeypatch.context() as m:
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
        ):
            m.delenv(name, raising=False)
        comm = _open(rank, tp_size, pp_size, port)
        quantized = comm.quantization == "int4"
        try:
            for op, dtype, layer, rows in _cases(DTYPES):
                where = f"{op} {dtype} {layer} {rows} x {HIDDEN}"
                a = _inputs(op, rows, dtype, rank, layer)
                assert _should(comm, op, a), where
                refs_args = {k: v.clone() if isinstance(v, torch.Tensor) else v
                             for k, v in a.items()}  # fmt: skip
                outs = _call(comm, op, a)
                summed = a["x"].clone()
                dist.all_reduce(summed, group=comm.device_group)
                _check(op, quantized, outs, _reference(op, summed, refs_args), where)
        finally:
            comm.close()


@ray.remote(num_gpus=1, max_calls=1)
def graph_ops(monkeypatch: pytest.MonkeyPatch, tp_size, pp_size, rank, port):
    with monkeypatch.context() as m:
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
        ):
            m.delenv(name, raising=False)
        comm = _open(rank, tp_size, pp_size, port)
        quantized = comm.quantization == "int4"
        try:
            for op, dtype, layer, rows in _cases(DTYPES):
                where = f"graph {op} {dtype} {layer} {rows} x {HIDDEN}"
                static = _inputs(op, rows, dtype, rank, layer)
                graph = torch.cuda.CUDAGraph()
                with comm.capture():
                    _call(comm, op, static)  # vLLM's warmup: launches nothing
                    with torch.cuda.graph(graph):
                        outs = _call(comm, op, static)
                # Replayed on fresh inputs, written into the captured ones.
                fresh = _inputs(op, rows, dtype, rank + 100, layer)
                for k, v in static.items():
                    if isinstance(v, torch.Tensor):
                        v.copy_(fresh[k])
                graph.replay()
                torch.accelerator.synchronize()
                summed = fresh["x"].clone()
                dist.all_reduce(summed, group=comm.device_group)
                _check(op, quantized, outs, _reference(op, summed, fresh), where)
        finally:
            comm.close()


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("test_target", [eager_ops, graph_ops])
@pytest.mark.parametrize("quantization", ["NONE", "INT4"])
def test_hip_custom_communicator(
    monkeypatch: pytest.MonkeyPatch, tp_size, test_target, quantization
):
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    monkeypatch.setenv("VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION", quantization)
    multi_process_parallel(monkeypatch, tp_size, 1, test_target)


@ray.remote(num_gpus=1, max_calls=1)
def refused_calls(monkeypatch: pytest.MonkeyPatch, tp_size, pp_size, rank, port):
    with monkeypatch.context() as m:
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
        ):
            m.delenv(name, raising=False)
        comm = _open(rank, tp_size, pp_size, port)
        try:
            # Past the largest call it was sized for: vLLM takes another path.
            past = _ints((16 * MAX_TOKENS, HIDDEN), torch.bfloat16, rank)
            assert not comm.should_allreduce(past)
            # Not [rows, cols]: refused by name before anything launches.
            with pytest.raises(CommunicatorError) as e:
                comm.all_reduce(_ints((2, 4, HIDDEN), torch.bfloat16, rank))
            assert e.value.error == Error.not_two_d
            # A strided view of the hidden row runs, and sums right.
            wide = _ints((4, 2 * HIDDEN), torch.bfloat16, rank)
            x = wide[:, :HIDDEN]
            summed = x.clone()
            dist.all_reduce(summed, group=comm.device_group)
            if comm.should_allreduce(x):
                assert torch.equal(comm.all_reduce(x), summed)
        finally:
            comm.close()


def test_hip_custom_communicator_refuses_by_name(monkeypatch: pytest.MonkeyPatch):
    if torch.accelerator.device_count() < 2:
        pytest.skip("Not enough GPUs to run the test.")
    monkeypatch.setenv("VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION", "NONE")
    multi_process_parallel(monkeypatch, 2, 1, refused_calls)


def test_hip_custom_communicator_tuned_table():
    """The shipped tuned table parses and checks on this device's arch, and every
    tuned op has a pick at Kimi-K3's decode and prefill sizes."""
    import vllm._rocm_C  # noqa: F401

    found = supported(torch.device("cuda:0"), 2)
    if isinstance(found, Error):
        pytest.skip(f"not a supported device: {found.name}")
    tuned = parse_tuned_configs(_TABLE, found.arch)
    assert tuned is not None, f"no tuned table for {found.arch}"
    for op in OPS:
        for quantization in ("exact", "int4"):
            for rows in (1, 2048):
                _, configs = tuned.nearest(op, quantization, 8, rows, HIDDEN)
                assert configs, (op, quantization, rows)
