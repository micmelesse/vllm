# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark: fused allreduce+rmsnorm+quant+gemm vs unfused baseline.

Compares unfused baseline (4 separate kernel launches) against the fused
torch op (rocm_aiter_fused_allreduce_add_rms_quant), with residual and
FP8 per-row quant + inlined GEMM.

Uses CUDA graph capture + replay for timing.  All barriers are inlined
in the fused kernel, so graph capture works on ROCm.  CUDA events around
graph.replay() measure pure GPU execution time with zero CPU dispatch
overhead — the same execution model as vLLM inference.

Usage:
    torchrun --nproc_per_node=8 benchmarks/kernels/benchmark_fused_collective_triton.py
    torchrun --nproc_per_node=8 benchmarks/kernels/benchmark_fused_collective_triton.py \\
        --num-tokens 1 4 1024
"""

import argparse
import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.distributed as dist

from vllm._aiter_ops import rocm_aiter_ops  # noqa: F401 (registers torch.ops.vllm.* ops)
import vllm.model_executor.kernels.linear.scaled_mm.rocm  # noqa: F401 (registers scaled_mm op)
from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.unfused_allreduce_add_rms_quant import (
    unfused_allreduce_add_rms_quant_gemm,
)

logger = init_logger(__name__)

FP8_DTYPE = current_platform.fp8_dtype()


# -- Benchmark variant ---------------------------------------------------

@dataclass
class BenchVariant:
    """A self-contained benchmark variant.

    Each variant owns its tensors and callable. No shared state between
    variants. Set is_baseline=True on the variant that other variants
    are compared against.
    """
    name: str
    make_fn: Callable[
        [int, int, torch.dtype, torch.device, str],
        Callable[[], None],
    ]
    is_baseline: bool = False


@dataclass
class BenchResults:
    """Raw benchmark data collected from a run."""
    variant_names: list[str]
    baseline_name: Optional[str]
    timings: dict[int, dict[str, float]]  # {num_tokens: {variant_name: ms}}
    world_size: int
    hidden_dim: int
    dtype: torch.dtype
    warmup: int
    trials: int


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
        gemm_weight = torch.rand(
            hidden_dim, hidden_dim, dtype=torch.float32, device=device,
        ).to(FP8_DTYPE).contiguous().t()
        weight_scale = torch.tensor(
            1.0, dtype=torch.float32, device=device,
        ).unsqueeze(0)

        def run():
            inp = input_tensor.clone()
            res = residual.clone()
            unfused_allreduce_add_rms_quant_gemm(
                inp, rms_weight, 1e-6, scale, FP8_DTYPE,
                group_name, gemm_weight, weight_scale, dtype,
                residual=res,
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
        gemm_weight = torch.rand(
            hidden_dim, hidden_dim, dtype=torch.float32, device=device,
        ).to(FP8_DTYPE).contiguous().t()
        weight_scale = torch.tensor(
            1.0, dtype=torch.float32, device=device,
        ).unsqueeze(0)

        def run():
            inp = input_tensor.clone()
            res = residual.clone()
            torch.ops.vllm.rocm_aiter_fused_allreduce_add_rms_quant(
                inp, res, rms_weight, 1e-6,
                FP8_DTYPE, group_name,
                gemm_weight, weight_scale, dtype,
            )

        return run

    return make_fn


# -- Data collection ------------------------------------------------------

def _capture_graph(
    fn: Callable[[], None],
    warmup: int,
) -> torch.cuda.CUDAGraph:
    """Warm up *fn* in eager mode, then capture into a CUDA graph.

    The warmup runs ensure all lazy allocations, JIT compilations, and
    autotuning are complete before graph capture begins.
    """
    stream = torch.cuda.current_stream()

    # Eager warmup (outside capture)
    for _ in range(warmup):
        fn()
    stream.synchronize()

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    stream.synchronize()

    return graph


def benchmark_graph(
    graph: torch.cuda.CUDAGraph,
    trials: int,
) -> float:
    """Replay a captured CUDA graph, return median time in milliseconds.

    CUDA events around graph.replay() measure pure GPU execution time
    with zero CPU launch overhead.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []

    for _ in range(trials):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    mid = len(times) // 2
    if len(times) % 2 == 0:
        return (times[mid - 1] + times[mid]) / 2
    return times[mid]


def collect(
    variants: list[BenchVariant],
    token_counts: list[int],
    hidden_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    group_name: str,
    warmup: int,
    trials: int,
    world_size: int,
    profile_dir: Optional[str] = None,
) -> BenchResults:
    """Capture CUDA graphs for all variants, time replays, return data.

    Each (variant, token_count) pair gets its own CUDA graph.  Warmup
    runs happen in eager mode before capture.  Profiling, when enabled,
    wraps the timed replays.
    """
    rank = device.index or 0

    # Build run functions and capture graphs.
    all_run_fns: dict[int, dict[str, Callable]] = {}
    all_graphs: dict[int, dict[str, torch.cuda.CUDAGraph]] = {}
    for num_tokens in token_counts:
        all_run_fns[num_tokens] = {}
        all_graphs[num_tokens] = {}
        for variant in variants:
            run_fn = variant.make_fn(
                num_tokens, hidden_dim, dtype, device, group_name,
            )
            all_run_fns[num_tokens][variant.name] = run_fn
            graph = _capture_graph(run_fn, warmup)
            all_graphs[num_tokens][variant.name] = graph
            if rank == 0:
                logger.info(
                    "Captured CUDA graph: %s, tokens=%d",
                    variant.name, num_tokens,
                )

    # Start profiler after capture so traces only contain replay calls.
    profiler: Optional[torch.profiler.profile] = None
    if profile_dir is not None:
        os.makedirs(profile_dir, exist_ok=True)
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        )
        profiler.start()
        if rank == 0:
            logger.info("Profiler enabled, traces will be saved to %s",
                        profile_dir)

    # Timed graph replays.
    all_timings: dict[int, dict[str, float]] = {}
    for num_tokens in token_counts:
        timings: dict[str, float] = {}
        for variant in variants:
            graph = all_graphs[num_tokens][variant.name]
            timings[variant.name] = benchmark_graph(graph, trials)
        all_timings[num_tokens] = timings

    if profiler is not None:
        profiler.stop()
        if rank == 0:
            print(profiler.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=30,
            ))
        trace_path = os.path.join(profile_dir, f"trace_rank{rank}.json")
        profiler.export_chrome_trace(trace_path)
        if rank == 0:
            logger.info("Traces saved to %s/trace_rank*.json", profile_dir)

    baseline = next((v for v in variants if v.is_baseline), None)
    return BenchResults(
        variant_names=[v.name for v in variants],
        baseline_name=baseline.name if baseline else None,
        timings=all_timings,
        world_size=world_size,
        hidden_dim=hidden_dim,
        dtype=dtype,
        warmup=warmup,
        trials=trials,
    )


# -- Presentation ---------------------------------------------------------

def _speedup(baseline_ms: Optional[float], ms: float) -> Optional[float]:
    if baseline_ms is None or ms <= 0:
        return None
    return baseline_ms / ms


def print_results(results: BenchResults) -> None:
    """Print formatted summary table to stdout."""
    bl = results.baseline_name

    hdr = (
        f"\n{'='*70}\n"
        f"Benchmark: {' vs '.join(results.variant_names)}\n"
        f"world_size={results.world_size}  hidden_dim={results.hidden_dim}  "
        f"dtype={results.dtype}  warmup={results.warmup}  "
        f"trials={results.trials}  mode=cudagraph\n"
    )
    if bl:
        hdr += f"baseline={bl}\n"
    hdr += f"{'='*70}"
    print(hdr)

    col_headers = ["Tokens"]
    for name in results.variant_names:
        col_headers.append(f"{name} (ms)")
        if bl and name != bl:
            col_headers.append("Speedup")
    print("  ".join(f"{h:>15s}" for h in col_headers))
    print("-" * (17 * len(col_headers)))

    for num_tokens, timings in results.timings.items():
        cols = [f"{num_tokens:>15d}"]
        baseline_ms = timings.get(bl) if bl else None
        for name in results.variant_names:
            t = timings[name]
            cols.append(f"{t:>15.3f}")
            if bl and name != bl:
                s = _speedup(baseline_ms, t)
                cols.append(f"{s:>14.2f}x" if s else f"{'N/A':>15s}")
        print("".join(cols))
    print()


def save_markdown(results: BenchResults, path: str) -> None:
    """Save results as a markdown table."""
    bl = results.baseline_name

    lines = [
        f"# Benchmark: {' vs '.join(results.variant_names)}",
        "",
        f"**World Size:** {results.world_size}  ",
        f"**Hidden Dimension:** {results.hidden_dim}  ",
        f"**dtype:** {results.dtype}  ",
        f"**Warmup:** {results.warmup}  ",
        f"**Trials:** {results.trials}  ",
        "**Mode:** cudagraph  ",
    ]
    if bl:
        lines.append(f"**Baseline:** {bl}  ")
    lines.append("")

    md_cols = ["Tokens"]
    for name in results.variant_names:
        md_cols.append(f"{name} (ms)")
        if bl and name != bl:
            md_cols.append("Speedup")
    lines.append("| " + " | ".join(md_cols) + " |")
    lines.append("|" + "|".join("---:" for _ in md_cols) + "|")

    for num_tokens, timings in results.timings.items():
        baseline_ms = timings.get(bl) if bl else None
        row = [str(num_tokens)]
        for name in results.variant_names:
            t = timings[name]
            row.append(f"{t:.3f}")
            if bl and name != bl:
                s = _speedup(baseline_ms, t)
                row.append(f"{s:.2f}x" if s else "N/A")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Results saved to %s", path)


# -- Main -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark fused allreduce+rmsnorm+quant+gemm vs unfused"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        help="Token counts to benchmark",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=8192, help="Hidden dimension",
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Warmup iterations before graph capture",
    )
    parser.add_argument(
        "--trials", type=int, default=50, help="Graph replay trials",
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
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        metavar="DIR",
        help="Enable torch.profiler and save traces to DIR",
    )
    args = parser.parse_args()

    # -- Distributed setup ------------------------------------------------
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

    group_name = get_tp_group().unique_name

    # -- Build variants ---------------------------------------------------
    all_variants = [
        BenchVariant(name="unfused", make_fn=_make_unfused_variant(), is_baseline=True),
        BenchVariant(name="fused", make_fn=_make_fused_variant()),
    ]
    if args.variant == "all":
        variants = all_variants
    else:
        variants = [v for v in all_variants if v.name == args.variant]

    # -- Collect data -----------------------------------------------------
    results = collect(
        variants=variants,
        token_counts=args.num_tokens,
        hidden_dim=args.hidden_dim,
        dtype=torch.bfloat16,
        device=device,
        group_name=group_name,
        warmup=args.warmup,
        trials=args.trials,
        world_size=world_size,
        profile_dir=args.profile,
    )

    # -- Present results (rank 0) -----------------------------------------
    if rank == 0:
        print_results(results)
        if args.output_file:
            save_markdown(results, args.output_file)

    dist.barrier()


if __name__ == "__main__":
    main()
