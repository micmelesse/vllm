# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark: Unfused vLLM ops vs iris_opt fused Triton kernel.

Compares two paths (with residual, FP8 per-tensor quant):
  Unfused:  tensor_model_parallel_all_reduce -> RMSNorm (fused_add) -> QuantFP8 (static per-tensor)
  Fused:    fused_allreduce_add_rms_quant_iris_opt  (single Triton kernel)

Eager mode only (no CUDA graph capture) to avoid ROCm hipErrorStreamCaptureUnsupported.
Uses CUDA events for GPU-side timing.

Usage:
    torchrun --nproc_per_node=8 benchmarks/kernels/benchmark_fused_collective_triton.py
    torchrun --nproc_per_node=8 benchmarks/kernels/benchmark_fused_collective_triton.py \
        --num-tokens 1 16 128 512 1024 2048
"""

import argparse
import os
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist

from vllm.config.vllm import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform

from vllm.iris_opt_allreduce import (
    fused_allreduce_add_rms_quant_iris_opt,
    initialize_iris_opt,
)

logger = init_logger(__name__)

FP8_DTYPE = current_platform.fp8_dtype()


# ── Benchmark variant definition ────────────────────────────────────────────

@dataclass
class BenchVariant:
    """A self-contained benchmark variant.

    Each variant owns its tensors and callable — no shared state between
    variants.
    """
    name: str
    make_fn: Callable[
        [int, int, torch.dtype, torch.device, str],
        Callable[[], None],
    ]
    """Factory: (num_tokens, hidden_dim, dtype, device, group_name) -> run_fn"""


def _make_unfused(
    num_tokens: int,
    hidden_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    group_name: str,
) -> Callable[[], None]:
    """Build an unfused run_fn with its own tensors and layers."""
    with set_current_vllm_config(VllmConfig()):
        rms_norm = RMSNorm(hidden_dim, eps=1e-6, dtype=dtype)
        fp8_quant = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)

    input_tensor = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
    residual = torch.randn_like(input_tensor)
    scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    def run():
        inp = input_tensor.clone()
        res = residual.clone()
        ar_out = tensor_model_parallel_all_reduce(inp)
        rms_out, _residual_out = rms_norm(ar_out, res)
        fp8_quant(rms_out, scale)

    return run


def _make_iris_opt(
    num_tokens: int,
    hidden_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    group_name: str,
) -> Callable[[], None]:
    """Build an iris_opt run_fn with its own tensors."""
    input_tensor = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
    residual = torch.randn_like(input_tensor)
    rms_weight = torch.ones(hidden_dim, dtype=dtype, device=device)
    scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    def run():
        inp = input_tensor.clone()
        res = residual.clone()
        fused_allreduce_add_rms_quant_iris_opt(
            inp, rms_weight, 1e-6, scale, FP8_DTYPE,
            group_name, residual=res,
        )

    return run


VARIANTS = [
    BenchVariant(name="unfused_vllm", make_fn=_make_unfused),
    BenchVariant(name="iris_opt",     make_fn=_make_iris_opt),
]


# ── Benchmark helpers ────────────────────────────────────────────────────────

def benchmark_eager(fn: Callable[[], None], warmup: int, trials: int) -> float:
    """Benchmark *fn()* in eager mode using CUDA events.

    Returns average time in milliseconds.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(trials):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return sum(times) / len(times)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark unfused vLLM ops vs iris_opt fused Triton kernel"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        help="Token counts to benchmark",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=8192, help="Hidden dimension"
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Warmup iterations"
    )
    parser.add_argument(
        "--trials", type=int, default=50, help="Benchmark trials"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional output markdown file (rank 0 only)",
    )
    args = parser.parse_args()

    # ── Distributed setup ────────────────────────────────────────────────
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "Must run with torchrun. "
            "Example: torchrun --nproc_per_node=8 "
            "benchmarks/kernels/benchmark_fused_collective_triton.py"
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    if world_size <= 1:
        raise ValueError(
            "World size must be > 1 for collective operations. "
            f"Got world_size={world_size}."
        )

    # ── iris_opt init ────────────────────────────────────────────────────
    initialize_iris_opt()
    if rank == 0:
        logger.info(
            "iris_opt initialized. world_size=%d, hidden_dim=%d",
            world_size, args.hidden_dim,
        )

    group_name = get_tp_group().unique_name

    # ── Run benchmarks ───────────────────────────────────────────────────
    dtype = torch.bfloat16
    # {num_tokens: {variant_name: time_ms}}
    all_results: dict[int, dict[str, float]] = {}

    for num_tokens in args.num_tokens:
        timings: dict[str, float] = {}

        for variant in VARIANTS:
            run_fn = variant.make_fn(
                num_tokens, args.hidden_dim, dtype, device, group_name,
            )
            timings[variant.name] = benchmark_eager(
                run_fn, args.warmup, args.trials,
            )

        all_results[num_tokens] = timings

        if rank == 0:
            baseline = timings[VARIANTS[0].name]
            parts = [f"tokens={num_tokens:>5d}"]
            for v in VARIANTS:
                t = timings[v.name]
                speedup = baseline / t if t > 0 else float("inf")
                parts.append(f"{v.name}={t:.3f} ms ({speedup:.2f}x)")
            print("  ".join(parts))

    # ── Print summary table (rank 0) ─────────────────────────────────────
    if rank == 0:
        baseline_name = VARIANTS[0].name
        variant_names = [v.name for v in VARIANTS]

        hdr = (
            f"\n{'='*70}\n"
            f"Benchmark: {' vs '.join(variant_names)}\n"
            f"world_size={world_size}  hidden_dim={args.hidden_dim}  "
            f"dtype={dtype}  warmup={args.warmup}  trials={args.trials}\n"
            f"baseline={baseline_name}\n"
            f"{'='*70}"
        )
        print(hdr)

        # Header row
        col_headers = ["Tokens"]
        for name in variant_names:
            col_headers.append(f"{name} (ms)")
            if name != baseline_name:
                col_headers.append("Speedup")
        print("  ".join(f"{h:>15s}" for h in col_headers))
        print("-" * (17 * len(col_headers)))

        for num_tokens, timings in all_results.items():
            cols = [f"{num_tokens:>15d}"]
            baseline_ms = timings[baseline_name]
            for name in variant_names:
                t = timings[name]
                cols.append(f"{t:>15.3f}")
                if name != baseline_name:
                    speedup = baseline_ms / t if t > 0 else float("inf")
                    cols.append(f"{speedup:>14.2f}x")
            print("".join(cols))
        print()

    # ── Save markdown (rank 0) ───────────────────────────────────────────
    if args.output_file and rank == 0:
        baseline_name = VARIANTS[0].name
        variant_names = [v.name for v in VARIANTS]

        lines = [
            f"# Benchmark: {' vs '.join(variant_names)}",
            "",
            f"**World Size:** {world_size}  ",
            f"**Hidden Dimension:** {args.hidden_dim}  ",
            f"**dtype:** {dtype}  ",
            f"**Warmup:** {args.warmup}  ",
            f"**Trials:** {args.trials}  ",
            f"**Baseline:** {baseline_name}  ",
            "",
        ]

        # Build markdown table header
        md_cols = ["Tokens"]
        for name in variant_names:
            md_cols.append(f"{name} (ms)")
            if name != baseline_name:
                md_cols.append("Speedup")
        lines.append("| " + " | ".join(md_cols) + " |")
        lines.append("|" + "|".join("---:" for _ in md_cols) + "|")

        for num_tokens, timings in all_results.items():
            baseline_ms = timings[baseline_name]
            row = [str(num_tokens)]
            for name in variant_names:
                t = timings[name]
                row.append(f"{t:.3f}")
                if name != baseline_name:
                    speedup = baseline_ms / t if t > 0 else float("inf")
                    row.append(f"{speedup:.2f}x")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")
        with open(args.output_file, "w") as f:
            f.write("\n".join(lines))
        logger.info("Results saved to %s", args.output_file)

    dist.barrier()


if __name__ == "__main__":
    main()
