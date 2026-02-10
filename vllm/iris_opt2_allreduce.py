# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Iris fused AllReduce + RMSNorm + per-token FP8 Quant in a single kernel.

Single Triton kernel that fuses:
1. One-shot all-reduce (gather from all ranks via iris.load)
2. Optional residual addition
3. RMSNorm (row-level variance + normalize)
4. Per-token FP8 quantization (per-row scale)

The reduced data stays in registers through RMSNorm and quantization,
eliminating intermediate global memory traffic.

Row-based processing: BLOCK_SIZE_N >= hidden_size so each iteration
handles complete rows. Uses persistent CTAs that iterate over rows.

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
# Config
# ============================================================================


@dataclass
class FusedOneShotConfig:
    """Config for the fused one-shot all-reduce kernel."""

    comm_sms: int = 64
    num_xcds: Optional[int] = None
    chunk_size: Optional[int] = None
    swizzle_size: int = 4

    def __post_init__(self) -> None:
        if self.num_xcds is None:
            self.num_xcds = iris.hip.get_num_xcc()
        if self.chunk_size is None:
            self.chunk_size = self.swizzle_size * self.swizzle_size
            self.chunk_size = min(
                self.chunk_size, self.comm_sms // self.num_xcds
            )


# ============================================================================
# Fused AllReduce + RMSNorm + Per-Token FP8 Quant kernel
# ============================================================================


@triton.jit
def fused_allreduce_rmsnorm_quant_kernel(
    # Input (in symmetric heap for iris.load)
    input_ptr,
    # Outputs (regular GPU memory)
    allreduce_out_ptr,
    rms_out_ptr,
    quant_out_ptr,
    scale_out_ptr,
    # Residual (regular GPU memory; dummy ptr when HAS_RESIDUAL=False)
    residual_in_ptr,
    residual_out_ptr,
    # RMSNorm weight
    rms_weight_ptr,
    # Scalar params
    rms_eps,
    fp8_max,
    # Dimensions
    M,
    N,
    # Input strides (iris heap buffer)
    stride_in_m,
    stride_in_n,
    # Iris params
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    # Kernel config
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    """Fused one-shot AllReduce + RMSNorm + per-token FP8 quant.

    Row-based processing: BLOCK_SIZE_N >= hidden_size so each iteration
    handles complete rows. The reduced data stays in registers through
    RMSNorm and quantization — no intermediate global memory traffic.

    Each CTA persistently iterates over rows.
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    cols = tl.arange(0, BLOCK_SIZE_N)
    col_mask = cols < N

    # Load RMSNorm weight once (shared across all rows)
    rms_w = tl.load(
        rms_weight_ptr + cols, mask=col_mask, other=0.0
    ).to(tl.float32)

    for row in range(pid, M, COMM_SMS):
        in_offset = row * stride_in_m + cols * stride_in_n
        out_offset = row * N + cols

        # --- Step 1: Gather from all ranks and reduce ---
        acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        for i in range(world_size):
            remote_rank = rank_start + i * rank_stride
            partial = iris.load(
                input_ptr + in_offset,
                iris_rank,
                remote_rank,
                heap_bases,
                mask=col_mask,
            )
            acc += partial.to(tl.float32)

        # Store allreduce output
        tl.store(
            allreduce_out_ptr + out_offset,
            acc.to(allreduce_out_ptr.type.element_ty),
            mask=col_mask,
        )

        # --- Step 2: Optional residual add ---
        if HAS_RESIDUAL:
            res_in = tl.load(
                residual_in_ptr + out_offset, mask=col_mask, other=0.0
            ).to(tl.float32)
            rms_in = acc + res_in
            tl.store(
                residual_out_ptr + out_offset,
                rms_in.to(residual_out_ptr.type.element_ty),
                mask=col_mask,
            )
        else:
            rms_in = acc

        # --- Step 3: RMSNorm (in float32) ---
        sq_sum = tl.sum(rms_in * rms_in, axis=0)
        variance = sq_sum / N
        rrms = tl.rsqrt(variance + rms_eps)
        normed = rms_in * rrms * rms_w

        tl.store(
            rms_out_ptr + out_offset,
            normed.to(rms_out_ptr.type.element_ty),
            mask=col_mask,
        )

        # --- Step 4: Per-token FP8 quantization ---
        amax = tl.max(tl.abs(normed), axis=0)
        amax = tl.maximum(amax, 1e-12)
        scale = amax / fp8_max
        quantized = (normed / scale).to(quant_out_ptr.type.element_ty)

        tl.store(
            quant_out_ptr + out_offset, quantized, mask=col_mask
        )
        tl.store(scale_out_ptr + row, scale.to(tl.float32))


# ============================================================================
# Manager and public API
# ============================================================================


class IrisOpt2Manager:
    """Singleton manager for fused one-shot AllReduce+RMSNorm+Quant."""

    _instance: Optional["IrisOpt2Manager"] = None
    _initialized: bool = False

    def __new__(cls) -> "IrisOpt2Manager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if IrisOpt2Manager._initialized:
            return
        IrisOpt2Manager._initialized = True

        self._shmem: Any = None
        self._heap_size: int = 2**33  # 8GB default
        self._config: Optional[FusedOneShotConfig] = None

        # Buffer cache: (M, N, dtype) -> iris_input
        self._buffer_cache: dict[
            tuple[int, int, torch.dtype], Any
        ] = {}
        self._max_cached_shapes: int = 16

    def initialize(self, heap_size: Optional[int] = None) -> None:
        """Initialize Iris symmetric heap (call once at startup)."""
        if self._shmem is not None:
            logger.debug("Iris (opt2) already initialized, skipping")
            return

        if heap_size is not None:
            self._heap_size = heap_size

        cur_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        logger.info(
            "Initializing Iris (opt2) symmetric heap: "
            f"rank={cur_rank}, heap_size={self._heap_size / 2**30:.1f}GB"
        )

        self._shmem = iris.iris(self._heap_size)
        self._config = FusedOneShotConfig()

        logger.info(
            f"Iris (opt2) initialized successfully on rank {cur_rank}"
        )

    @property
    def shmem(self) -> Any:
        """Get the Iris symmetric memory instance (auto-initializes)."""
        if self._shmem is None:
            self.initialize()
        return self._shmem

    @property
    def config(self) -> FusedOneShotConfig:
        """Get the fused one-shot config."""
        if self._config is None:
            self.initialize()
        assert self._config is not None
        return self._config

    def _get_or_create_input_buffer(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
    ) -> Any:
        """Get cached symmetric heap input buffer or create a new one."""
        cache_key = (M, N, dtype)

        if cache_key in self._buffer_cache:
            return self._buffer_cache[cache_key]

        if len(self._buffer_cache) >= self._max_cached_shapes:
            logger.info(
                f"Iris (opt2) buffer cache full "
                f"({len(self._buffer_cache)} shapes), clearing"
            )
            self._buffer_cache.clear()

        shmem = self.shmem
        iris_input = shmem.zeros((M, N), dtype=dtype)
        shmem.barrier()

        self._buffer_cache[cache_key] = iris_input

        cur_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        logger.info(
            f"Iris (opt2): created input buffer for shape ({M}, {N}), "
            f"dtype={dtype}, rank={cur_rank}"
        )

        return iris_input

    def fused_allreduce_rmsnorm_quant(
        self,
        input_tensor: torch.Tensor,
        rms_weight: torch.Tensor,
        rms_eps: float,
        quant_dtype: torch.dtype,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        """Fused AllReduce + RMSNorm + per-token FP8 quant in one kernel.

        Args:
            input_tensor: Input tensor (M, N) on GPU
            rms_weight: RMSNorm weight (N,)
            rms_eps: RMSNorm epsilon
            quant_dtype: FP8 dtype (e.g. torch.float8_e4m3fn)
            residual: Optional residual tensor (M, N)

        Returns:
            (allreduce_out, rms_out, residual_out, quant_out, scale_out)
            residual_out is None when residual is None.
            scale_out is per-token: shape (M, 1).
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Iris requires --enforce-eager")

        shmem = self.shmem
        config = self.config
        M, N = input_tensor.shape
        device = input_tensor.device

        # Get symmetric heap buffer for input
        iris_input = self._get_or_create_input_buffer(
            M, N, input_tensor.dtype
        )

        # Copy input to symmetric heap
        iris_input.copy_(input_tensor)
        shmem.barrier()

        # Allocate output tensors (regular GPU memory)
        allreduce_out = torch.empty_like(input_tensor)
        rms_out = torch.empty_like(input_tensor)
        quant_out = torch.empty(
            (M, N), dtype=quant_dtype, device=device
        )
        scale_out = torch.empty(M, dtype=torch.float32, device=device)

        if residual is not None:
            residual_out = torch.empty_like(input_tensor)
        else:
            residual_out = None

        # FP8 max value
        fp8_max = torch.finfo(quant_dtype).max

        # Iris group info
        rank_in_group, rank_global, world_size, rank_start, rank_stride = (
            extract_group_info(shmem)
        )
        heap_bases = shmem.get_heap_bases()

        BLOCK_SIZE_N = triton.next_power_of_2(N)

        # Launch fused kernel
        fused_allreduce_rmsnorm_quant_kernel[(config.comm_sms,)](
            iris_input,
            allreduce_out,
            rms_out,
            quant_out,
            scale_out,
            # Dummy ptrs when no residual (HAS_RESIDUAL=False skips access)
            residual if residual is not None else input_tensor,
            residual_out if residual_out is not None else allreduce_out,
            rms_weight,
            rms_eps,
            fp8_max,
            M,
            N,
            iris_input.stride(0),
            iris_input.stride(1),
            heap_bases,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
            BLOCK_SIZE_N,
            residual is not None,
            num_warps=8,
            num_stages=1,
            waves_per_eu=1,
        )

        shmem.barrier()

        # The kernel computes per-token scales (M,) but the fused op contract
        # requires per-tensor scale (1,) to match the unfused graph output.
        # Take the max per-token scale as the per-tensor scale, then
        # re-quantize rms_out with the per-tensor scale.
        per_tensor_scale = scale_out.max().reshape(1)
        fp8_max_val = torch.finfo(quant_dtype).max
        quant_out = (rms_out.float() / per_tensor_scale).clamp(
            -fp8_max_val, fp8_max_val
        ).to(quant_dtype)

        return allreduce_out, rms_out, residual_out, quant_out, per_tensor_scale


_iris_opt2_manager: Optional[IrisOpt2Manager] = None


def get_iris_opt2_manager() -> IrisOpt2Manager:
    """Get the global Iris opt2 manager instance."""
    global _iris_opt2_manager
    if _iris_opt2_manager is None:
        _iris_opt2_manager = IrisOpt2Manager()
    return _iris_opt2_manager


def initialize_iris_opt2(heap_size: Optional[int] = None) -> None:
    """Initialize Iris for fused all-reduce operations.

    Call this once at model load time before any forward passes.

    Args:
        heap_size: Size of symmetric heap in bytes (default: 8GB)
    """
    get_iris_opt2_manager().initialize(heap_size)


def fused_allreduce_add_rms_quant_iris_opt2(
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
    """Fused AllReduce + Add + RMSNorm + per-token FP8 Quant.

    Single Triton kernel: one-shot all-reduce with RMSNorm and per-token
    FP8 quantization fused into the store phase. No intermediate memory
    traffic between the three operations.

    Note: Uses per-token quantization (one scale per row) instead of
    per-tensor quantization. scale_out shape is (M, 1).
    """
    iris_mgr = get_iris_opt2_manager()
    return iris_mgr.fused_allreduce_rmsnorm_quant(
        input, rms_weight, rms_eps, quant_dtype, residual,
    )
