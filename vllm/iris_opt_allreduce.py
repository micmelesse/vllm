# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Iris optimized all-reduce implementation.

Inlines the Iris two-shot all-reduce Triton kernel and all supporting
code (config, workspace, group info, chiplet transform) directly in
this file. No imports from iris.ccl. Only depends on the iris runtime
for symmetric heap allocation (iris.iris) and the iris Triton language
extensions (iris.load, iris.store).

This gives us full control over the kernel for future fusion of
RMSNorm + FP8 Quant into the all-reduce's store phase.

Not compatible with CUDA graph capture. Requires --enforce-eager.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import iris
import torch
import triton
import triton.language as tl

import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Inlined from iris.ccl.utils
# ============================================================================


@triton.jit()
def chiplet_transform_chunked(
        pid,
        num_workgroups: tl.constexpr,
        num_xcds: tl.constexpr,
        chunk_size: tl.constexpr,
    ):
        """Redistribute workgroups across XCDs in chunks."""
        if pid > (num_workgroups // (num_xcds * chunk_size)) * (
            num_xcds * chunk_size
        ):
            return pid

        local_pid = pid // num_xcds
        chunk_idx = local_pid // chunk_size
        pos_in_chunk = local_pid % chunk_size

        xcd = pid % num_xcds
        new_pid = (
            chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk
        )
        return new_pid


def extract_group_info(
    shmem: Any,
) -> Tuple[int, int, int, int, int]:
    """Extract rank/group info from iris shmem context.

    Returns: (rank_in_group, rank_global, world_size, rank_start, rank_stride)
    """
    rank_global = shmem.get_rank()
    rank_in_group = rank_global
    world_size = shmem.get_num_ranks()
    rank_start = 0
    rank_stride = 1
    return rank_in_group, rank_global, world_size, rank_start, rank_stride


# ============================================================================
# Inlined from iris.ccl.config (two-shot relevant fields only)
# ============================================================================


@dataclass
class TwoShotConfig:
    """Config for the two-shot all-reduce kernel."""

    block_size_m: int = 32
    block_size_n: int = 64
    swizzle_size: int = 4
    comm_sms: int = 64
    num_xcds: Optional[int] = None
    chunk_size: Optional[int] = None
    all_reduce_distribution: int = 1

    def __post_init__(self) -> None:
        if self.num_xcds is None:
            self.num_xcds = iris.hip.get_num_xcc()
        if self.chunk_size is None:
            self.chunk_size = self.swizzle_size * self.swizzle_size
            self.chunk_size = min(
                self.chunk_size, self.comm_sms // self.num_xcds
            )


# ============================================================================
# Inlined from iris.ccl.all_reduce (two-shot kernel only)
# ============================================================================


@triton.jit
def persistent_all_reduce_two_shot(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DISTRIBUTION: tl.constexpr,
):
    """Two-shot all-reduce: reduce assigned tiles, broadcast to all peers.

    Phase 1 (reduce): Each rank reads its assigned tiles from all ranks
    via iris.load and sums them locally.
    Phase 2 (broadcast): Each rank writes the reduced result to all
    other ranks via iris.store.
    """
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    acc_dtype = (
        tl.float32
        if output_ptr.type.element_ty != tl.int8
        else tl.int32
    )

    tiles_per_rank = tl.cdiv(total_tiles, world_size)
    if DISTRIBUTION == 0:
        start_tile = group_rank
        stride = world_size
        remaining = total_tiles - start_tile
        remaining = tl.maximum(remaining, 0)
        max_tile_offset = tl.cdiv(remaining, stride)
    else:
        start_tile = group_rank * tiles_per_rank
        stride = 1
        remaining = total_tiles - start_tile
        remaining = tl.maximum(remaining, 0)
        max_tile_offset = tl.minimum(tiles_per_rank, remaining)

    for tile_offset in range(pid, max_tile_offset, COMM_SMS):
        tile_id = start_tile + tile_offset * stride

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (
            (tile_id % num_pid_in_group) % group_size_m
        )
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N

        is_full = (rm_base + BLOCK_SIZE_M <= M) & (
            rn_base + BLOCK_SIZE_N <= N
        )

        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)

        rm = tl.max_contiguous(
            tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M
        )
        rn = tl.max_contiguous(
            tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N
        )

        input_offset = (
            rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        )
        output_offset = (
            rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
        )

        base_ptr = input_ptr + input_offset
        out_ptr = output_ptr + output_offset

        # Fast path: full tiles (no masking needed)
        if is_full:
            mask = (rm[:, None] < M) & (rn[None, :] < N)

            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            acc = iris.load(
                base_ptr, iris_rank, start_rank_global, heap_bases
            ).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                acc += iris.load(
                    base_ptr, iris_rank, remote_rank, heap_bases
                ).to(acc_dtype)

            reduced = acc.to(output_ptr.type.element_ty)
            tl.store(out_ptr, reduced, cache_modifier=".wt")

            for i in tl.static_range(0, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                if remote_rank_idx != group_rank:
                    iris.store(
                        out_ptr,
                        reduced,
                        iris_rank,
                        remote_rank,
                        heap_bases,
                    )

        # Slow path: boundary tiles (masked)
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)

            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            acc = iris.load(
                base_ptr,
                iris_rank,
                start_rank_global,
                heap_bases,
                mask=mask,
            ).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                acc += iris.load(
                    base_ptr,
                    iris_rank,
                    remote_rank,
                    heap_bases,
                    mask=mask,
                ).to(acc_dtype)

            reduced = acc.to(output_ptr.type.element_ty)
            tl.store(out_ptr, reduced, mask=mask, cache_modifier=".wt")

            for i in tl.static_range(0, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                if remote_rank_idx != group_rank:
                    iris.store(
                        out_ptr,
                        reduced,
                        iris_rank,
                        remote_rank,
                        heap_bases,
                        mask=mask,
                    )


# ============================================================================
# Manager and public API
# ============================================================================


class IrisOptManager:
    """Singleton manager for Iris with inlined two-shot all-reduce.

    Calls the two-shot Triton kernel directly instead of going through
    shmem.ccl.all_reduce(). This gives us control over the kernel for
    future fusion.
    """

    _instance: Optional["IrisOptManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "IrisOptManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if IrisOptManager._initialized:
            return
        IrisOptManager._initialized = True

        self._shmem: Any = None
        self._heap_size: int = 2**33  # 8GB default
        self._config: Optional[TwoShotConfig] = None

        # Buffer cache: (M, N, dtype) -> (iris_input, iris_output)
        self._buffer_cache: dict[
            tuple[int, int, torch.dtype], tuple[Any, Any]
        ] = {}
        self._max_cached_shapes: int = 16

    def initialize(self, heap_size: Optional[int] = None) -> None:
        """Initialize Iris symmetric heap (call once at startup)."""
        if self._shmem is not None:
            logger.debug("Iris already initialized, skipping")
            return

        if heap_size is not None:
            self._heap_size = heap_size

        cur_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        logger.info(
            "Initializing Iris (opt) symmetric heap: "
            f"rank={cur_rank}, heap_size={self._heap_size / 2**30:.1f}GB"
        )

        self._shmem = iris.iris(self._heap_size)
        self._config = TwoShotConfig()

        logger.info(f"Iris (opt) initialized successfully on rank {cur_rank}")

    @property
    def shmem(self) -> Any:
        """Get the Iris symmetric memory instance (auto-initializes)."""
        if self._shmem is None:
            self.initialize()
        return self._shmem

    @property
    def config(self) -> TwoShotConfig:
        """Get the two-shot config."""
        if self._config is None:
            self.initialize()
        assert self._config is not None
        return self._config

    def _get_or_create_buffers(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
    ) -> tuple[Any, Any]:
        """Get cached buffers or create new ones."""
        cache_key = (M, N, dtype)

        if cache_key in self._buffer_cache:
            return self._buffer_cache[cache_key]

        if len(self._buffer_cache) >= self._max_cached_shapes:
            logger.info(
                f"Iris buffer cache full ({len(self._buffer_cache)} shapes), "
                "clearing"
            )
            self._buffer_cache.clear()

        shmem = self.shmem

        iris_input = shmem.zeros((M, N), dtype=dtype)
        iris_output = shmem.zeros((M, N), dtype=dtype)

        # Two-shot preamble is a no-op (no workspace needed)
        shmem.barrier()

        self._buffer_cache[cache_key] = (iris_input, iris_output)

        cur_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        logger.info(
            f"Iris (opt): created buffers for shape ({M}, {N}), "
            f"dtype={dtype}, rank={cur_rank}"
        )

        return iris_input, iris_output

    def all_reduce(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Perform all-reduce using the inlined two-shot kernel.

        Args:
            input_tensor: Input tensor (M, N) on GPU

        Returns:
            All-reduced tensor (M, N)
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Iris requires --enforce-eager")

        shmem = self.shmem
        config = self.config

        M, N = input_tensor.shape

        iris_input, iris_output = self._get_or_create_buffers(
            M, N, input_tensor.dtype
        )

        # Copy input to symmetric heap
        iris_input.copy_(input_tensor)
        shmem.barrier()

        # Extract group info
        rank_in_group, rank_global, world_size, rank_start, rank_stride = (
            extract_group_info(shmem)
        )
        heap_bases = shmem.get_heap_bases()

        stride_in_m, stride_in_n = (
            iris_input.stride(0),
            iris_input.stride(1),
        )
        stride_out_m, stride_out_n = (
            iris_output.stride(0),
            iris_output.stride(1),
        )

        # Launch the two-shot kernel directly
        persistent_all_reduce_two_shot[(config.comm_sms,)](
            iris_input,
            iris_output,
            M,
            N,
            stride_in_m,
            stride_in_n,
            stride_out_m,
            stride_out_n,
            heap_bases,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
            config.all_reduce_distribution,
            num_warps=8,
            num_stages=1,
            waves_per_eu=1,
        )

        shmem.barrier()

        # Copy result back from symmetric heap
        output = torch.empty_like(input_tensor)
        output.copy_(iris_output)

        return output


_iris_opt_manager: Optional[IrisOptManager] = None


def get_iris_opt_manager() -> IrisOptManager:
    """Get the global Iris opt manager instance."""
    global _iris_opt_manager
    if _iris_opt_manager is None:
        _iris_opt_manager = IrisOptManager()
    return _iris_opt_manager


def initialize_iris_opt(heap_size: Optional[int] = None) -> None:
    """Initialize Iris for optimized all-reduce operations.

    Call this once at model load time before any forward passes.

    Args:
        heap_size: Size of symmetric heap in bytes (default: 8GB)
    """
    get_iris_opt_manager().initialize(heap_size)


def fused_allreduce_add_rms_quant_iris_opt(
    input: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    quant_scale: torch.Tensor,
    quant_dtype: torch.dtype,
    group_name: str,
    residual: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """Iris optimized AllReduce + Add + RMSNorm + FP8 Quant.

    Calls the inlined two-shot kernel directly. RMSNorm and quant are
    still separate ops for now. The next step is to fuse them into the
    kernel's store phase.
    """
    iris_mgr = get_iris_opt_manager()

    # Step 1: All-reduce using inlined two-shot kernel
    allreduce_out = iris_mgr.all_reduce(input)

    # Step 2: RMSNorm (with or without residual add)
    if residual is not None:
        residual_out = allreduce_out + residual
        variance = (residual_out.float() ** 2).mean(dim=-1, keepdim=True)
        rrms = torch.rsqrt(variance + rms_eps)
        rms_out = (residual_out.float() * rrms * rms_weight.float()).to(
            input.dtype
        )
    else:
        residual_out = None
        variance = (allreduce_out.float() ** 2).mean(dim=-1, keepdim=True)
        rrms = torch.rsqrt(variance + rms_eps)
        rms_out = (allreduce_out.float() * rrms * rms_weight.float()).to(
            input.dtype
        )

    # Step 3: FP8 Quant - simple per-tensor quantization
    abs_max = rms_out.float().abs().max()
    if quant_dtype == torch.float8_e4m3fn:
        fp8_max = 448.0
    else:
        fp8_max = 57344.0
    quant_scale_out = (abs_max / fp8_max).to(torch.float32)
    quant_out = (rms_out.float() / quant_scale_out).to(quant_dtype)
    quant_scale_out = quant_scale_out.view(1)

    return allreduce_out, rms_out, residual_out, quant_out, quant_scale_out
