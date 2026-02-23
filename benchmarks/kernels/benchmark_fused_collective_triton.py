# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark: fused allreduce+rmsnorm+quant vs unfused baseline.

Compares unfused baseline (3 separate kernel launches) against the fused
torch op (rocm_aiter_fused_allreduce_add_rms_quant), with residual and
FP8 per-tensor quant.

Eager mode only (no CUDA graph capture) to avoid ROCm
hipErrorStreamCaptureUnsupported. Uses CUDA events for GPU-side timing.

Usage:
    torchrun --nproc_per_node=8 benchmarks/kernels/benchmark_fused_collective_triton.py
    torchrun --nproc_per_node=8 benchmarks/kernels/benchmark_fused_collective_triton.py \
        --num-tokens 1 4 1024
"""

import argparse
import os
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist

from vllm._aiter_ops import rocm_aiter_ops  # noqa: F401 (registers torch.ops.vllm.* ops)
from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.unfused_allreduce_add_rms_quant import unfused_allreduce_add_rms_quant

logger = init_logger(__name__)

FP8_DTYPE = current_platform.fp8_dtype()


# ── Benchmark variant ───────────────────────────────────────────────────────

@dataclass
class BenchVariant:
    """A self-contained benchmark variant.

    Each variant owns its tensors and callable. No shared state between
    variants.
    """
    name: str
    make_fn: Callable[
        [int, int, torch.dtype, torch.device, str],
        Callable[[], None],
    ]


def _make_unfused_variant():
    """Return a factory that creates a run_fn for unfused baseline."""

    def make_fn(
        num_tokens: int,
        hidden_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        group_name: str,
    ) -> Callable[[], None]:
        input_tensor = torch.randn(
            num_tokens, hidden_dim, dtype=dtype, device=device,
        )
        residual = torch.randn_like(input_tensor)
        rms_weight = torch.ones(hidden_dim, dtype=dtype, device=device)
        scale = torch.tensor(1.0, dtype=torch.float32, device=device)

        def run():
            inp = input_tensor.clone()
            res = residual.clone()
            unfused_allreduce_add_rms_quant(
                inp, rms_weight, 1e-6, scale, FP8_DTYPE,
                group_name, residual=res,
            )

        return run

    return make_fn


def _make_fused_variant():
    """Return a factory that creates a run_fn for fused torch op."""

    def make_fn(
        num_tokens: int,
        hidden_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        group_name: str,
    ) -> Callable[[], None]:
        input_tensor = torch.randn(
            num_tokens, hidden_dim, dtype=dtype, device=device,
        )
        residual = torch.randn_like(input_tensor)
        rms_weight = torch.ones(hidden_dim, dtype=dtype, device=device)
        scale = torch.tensor(1.0, dtype=torch.float32, device=device)

        def run():
            inp = input_tensor.clone()
            res = residual.clone()
            torch.ops.vllm.rocm_aiter_fused_allreduce_add_rms_quant(
                inp, res, rms_weight, 1e-6, scale, FP8_DTYPE, group_name,
            )

        return run

    return make_fn


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
        description="Benchmark fused allreduce+rmsnorm+quant vs unfused"
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
    parser.add_argument(
        "--variant",
        type=str,
        choices=["all", "unfused", "fused"],
        default="all",
        help="Which variant(s) to run (default: all)",
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

    if rank == 0:
        logger.info(
            "Benchmarking unfused vs fused  world_size=%d  hidden_dim=%d",
            world_size, args.hidden_dim,
        )

    group_name = get_tp_group().unique_name

    # ── Build variants ───────────────────────────────────────────────────
    all_variants = [
        BenchVariant(name="unfused", make_fn=_make_unfused_variant()),
        BenchVariant(name="fused", make_fn=_make_fused_variant()),
    ]
    if args.variant == "all":
        variants = all_variants
    else:
        variants = [v for v in all_variants if v.name == args.variant]

    # ── Run benchmarks ───────────────────────────────────────────────────
    dtype = torch.bfloat16
    all_results: dict[int, dict[str, float]] = {}

    for num_tokens in args.num_tokens:
        timings: dict[str, float] = {}

        for variant in variants:
            run_fn = variant.make_fn(
                num_tokens, args.hidden_dim, dtype, device, group_name,
            )
            timings[variant.name] = benchmark_eager(
                run_fn, args.warmup, args.trials,
            )

        all_results[num_tokens] = timings

        if rank == 0:
            baseline = timings["unfused"]
            parts = [f"tokens={num_tokens:>5d}"]
            for v in variants:
                t = timings[v.name]
                speedup = baseline / t if t > 0 else float("inf")
                parts.append(f"{v.name}={t:.3f} ms ({speedup:.2f}x)")
            print("  ".join(parts))

    # ── Print summary table (rank 0) ─────────────────────────────────────
    if rank == 0:
        variant_names = [v.name for v in variants]

        hdr = (
            f"\n{'='*70}\n"
            f"Benchmark: unfused vs fused\n"
            f"world_size={world_size}  hidden_dim={args.hidden_dim}  "
            f"dtype={dtype}  warmup={args.warmup}  trials={args.trials}\n"
            f"baseline=unfused\n"
            f"{'='*70}"
        )
        print(hdr)

        col_headers = ["Tokens"]
        for name in variant_names:
            col_headers.append(f"{name} (ms)")
            if name != "unfused":
                col_headers.append("Speedup")
        print("  ".join(f"{h:>15s}" for h in col_headers))
        print("-" * (17 * len(col_headers)))

        for num_tokens, timings in all_results.items():
            cols = [f"{num_tokens:>15d}"]
            baseline_ms = timings["unfused"]
            for name in variant_names:
                t = timings[name]
                cols.append(f"{t:>15.3f}")
                if name != "unfused":
                    speedup = baseline_ms / t if t > 0 else float("inf")
                    cols.append(f"{speedup:>14.2f}x")
            print("".join(cols))
        print()

    # ── Save markdown (rank 0) ───────────────────────────────────────────
    if args.output_file and rank == 0:
        variant_names = [v.name for v in variants]

        lines = [
            f"# Benchmark: unfused vs fused",
            "",
            f"**World Size:** {world_size}  ",
            f"**Hidden Dimension:** {args.hidden_dim}  ",
            f"**dtype:** {dtype}  ",
            f"**Warmup:** {args.warmup}  ",
            f"**Trials:** {args.trials}  ",
            f"**Baseline:** unfused  ",
            "",
        ]

        md_cols = ["Tokens"]
        for name in variant_names:
            md_cols.append(f"{name} (ms)")
            if name != "unfused":
                md_cols.append("Speedup")
        lines.append("| " + " | ".join(md_cols) + " |")
        lines.append("|" + "|".join("---:" for _ in md_cols) + "|")

        for num_tokens, timings in all_results.items():
            baseline_ms = timings["unfused"]
            row = [str(num_tokens)]
            for name in variant_names:
                t = timings[name]
                row.append(f"{t:.3f}")
                if name != "unfused":
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
