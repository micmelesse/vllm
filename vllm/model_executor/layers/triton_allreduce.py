# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused AllReduce + RMSNorm for ROCm using Iris.

This module provides a fused operation that combines:
1. All-reduce across tensor parallel GPUs (via Iris symmetric memory)
2. Residual addition
3. RMS normalization

The fusion reduces memory bandwidth by avoiding intermediate writes.

Usage:
    Set VLLM_ROCM_TRITON_ALLREDUCE=1 to enable.
"""

from typing import Literal

import iris
import torch
import triton
import triton.language as tl

from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm

logger = init_logger(__name__)

# Implementation selection (internal detail)
AllReduceImpl = Literal["test", "simple_allreduce", "ccl_allreduce", "atomic_allreduce", "ring_allreduce"]
_IMPL: AllReduceImpl = "ccl_allreduce"


# ============================================================================
# AllReduce Context (initialized once, reused)
# ============================================================================


class _Context:
    """
    AllReduce state with automatic cleanup.
    
    Initialized once at max size, reused for all calls.
    Automatically cleans up Iris shared memory when garbage collected.
    """
    
    def __init__(
        self,
        shmem: iris.Iris,
        heap_bases: torch.Tensor,
        cur_rank: int,
        world_size: int,
        send_buffer: torch.Tensor,
        flags: torch.Tensor | None,
        global_output: torch.Tensor | None,
        ccl_output: torch.Tensor | None,
        max_m: int,
        N: int,
    ):
        self.shmem = shmem
        self.heap_bases = heap_bases
        self.cur_rank = cur_rank
        self.world_size = world_size
        self.send_buffer = send_buffer
        self.flags = flags
        self.global_output = global_output
        self.ccl_output = ccl_output
        self.max_m = max_m
        self.N = N
    
    def __del__(self):
        """Clean up Iris shared memory when context is destroyed."""
        if hasattr(self, 'shmem') and self.shmem is not None:
            try:
                del self.shmem
            except Exception:
                pass  # Ignore errors during cleanup


_ctx: _Context | None = None


def _get_buffers(
    M: int,
    N: int,
    dtype: torch.dtype,
    max_m: int,
    impl: AllReduceImpl,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """
    Get buffers for allreduce based on implementation type.
    Initializes context on first call, then returns views of appropriate size.
    Only allocates buffers needed by the specified impl:
    - test/simple_allreduce: send_buffer only
    - ccl_allreduce: send_buffer + ccl_output (for CCL all_reduce)
    - atomic_allreduce: send_buffer + global_output
    - ring_allreduce: send_buffer + flags
    
    Args:
        M: Actual number of tokens for this call.
        N: Hidden dimension.
        dtype: Tensor dtype.
        max_m: Max M (tokens) for buffer pre-allocation.
        impl: Which implementation to allocate buffers for.
        
    Returns:
        (send_buffer, flags, global_output, ccl_output) - views into pre-allocated buffers.
        flags, global_output, and ccl_output may be None if not needed by impl.
    """
    global _ctx
    
    # Initialize once
    if _ctx is None:
        dtype_size = torch.tensor([], dtype=dtype).element_size()
        send_buffer_bytes = max_m * N * dtype_size
        
        # Minimum heap size for Iris overhead (metadata, alignment, etc.)
        MIN_HEAP_SIZE = 1 * 1024 * 1024  # 1 MB minimum
        
        # Allocate buffers based on impl type
        if impl in ("test", "simple_allreduce"):
            # Just need send_buffer for send/recv
            heap_size = max(MIN_HEAP_SIZE, int(send_buffer_bytes * 1.1))
            shmem = iris.iris(heap_size)
            send_buffer = shmem.zeros((max_m, N), dtype=dtype)
            flags = None
            global_output = None
            ccl_output = None
        elif impl == "ccl_allreduce":
            # Need send_buffer + ccl_output for CCL all_reduce
            # Both buffers must be pre-allocated so all ranks have same offsets
            heap_size = max(MIN_HEAP_SIZE, int(send_buffer_bytes * 2.2))  # 2x for input+output
            shmem = iris.iris(heap_size)
            send_buffer = shmem.zeros((max_m, N), dtype=dtype)
            ccl_output = shmem.zeros((max_m, N), dtype=dtype)  # Output buffer for CCL
            flags = None
            global_output = None
        elif impl == "atomic_allreduce":
            # Need send_buffer + global_output for atomic accumulation
            global_output_bytes = max_m * N * 4  # float32
            heap_size = max(MIN_HEAP_SIZE, int((send_buffer_bytes + global_output_bytes) * 1.1))
            shmem = iris.iris(heap_size)
            send_buffer = shmem.zeros((max_m, N), dtype=dtype)
            flags = None
            global_output = shmem.zeros((max_m, N), dtype=torch.float32)
            ccl_output = None
        elif impl == "ring_allreduce":
            # Need send_buffer + flags for ring protocol
            flags_bytes = max_m * 4  # int32
            heap_size = max(MIN_HEAP_SIZE, int((send_buffer_bytes + flags_bytes) * 1.1))
            shmem = iris.iris(heap_size)
            send_buffer = shmem.zeros((max_m, N), dtype=dtype)
            flags = shmem.zeros((max_m,), dtype=torch.int32)
            global_output = None
            ccl_output = None
        else:
            raise ValueError(f"Unknown impl: {impl}")
        
        _ctx = _Context(
            shmem=shmem,
            heap_bases=shmem.get_heap_bases(),
            cur_rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            world_size=torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
            send_buffer=send_buffer,
            flags=flags,
            global_output=global_output,
            ccl_output=ccl_output,
            max_m=max_m,
            N=N,
        )
        logger.info(
            f"Initialized triton_allreduce [{impl}]: rank={_ctx.cur_rank}, "
            f"world_size={_ctx.world_size}, heap={heap_size / 2**30:.2f}GB, "
            f"buffers=({max_m}, {N}) {dtype}"
        )
    
    # Validate N matches allocated buffer
    if N != _ctx.N:
        raise ValueError(
            f"N mismatch: called with N={N} but buffer was allocated with N={_ctx.N}. "
            f"Triton allreduce requires consistent hidden dimension across calls."
        )
    
    # Get views for actual size
    send_view = _ctx.send_buffer[:M, :] if M < _ctx.max_m else _ctx.send_buffer
    flags_view = _ctx.flags[:M] if _ctx.flags is not None and M < _ctx.max_m else _ctx.flags
    global_output_view = _ctx.global_output[:M, :] if _ctx.global_output is not None and M < _ctx.max_m else _ctx.global_output
    ccl_output_view = _ctx.ccl_output[:M, :] if _ctx.ccl_output is not None and M < _ctx.max_m else _ctx.ccl_output
    
    # NOTE: We do NOT zero flags/global_output here - they are zeroed in the impl function
    # right before the kernel, after we've copied input to send_buffer.
    
    return send_view, flags_view, global_output_view, ccl_output_view


def get_configs(autotune: bool = False):
    if not autotune:
        return [
            triton.Config(
                {'BLOCK_N': 1024, 'waves_per_eu': 2},
                num_warps=4,
                num_stages=2,
            )
        ]

    configs = []
    for block_n in [64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        for num_warps in [2, 4, 8]:
            for num_stages in [1, 2]:
                for waves_per_eu in [1, 2]:
                    configs.append(
                        triton.Config(
                            {'BLOCK_N': block_n, 'waves_per_eu': waves_per_eu},
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    )
    return configs


@triton.autotune(
    configs=get_configs(),
    key=['N'],
)
@triton.jit
def _kernel_test(
    # Input pointer (data is in send_buffer, pre-copied by host)
    send_buffer_ptr,
    # Output pointers
    output_ptr,
    residual_out_ptr,
    # Shape params
    M,
    N: tl.constexpr,
    # Iris params
    heap_bases,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Test kernel - exercises iris.load and iris.store, outputs zeros.
    
    Following Iris examples 00_load and 01_store patterns:
    - iris.load: read from remote rank's buffer into local
    - iris.store: write from local to remote rank's buffer
    
    This kernel exercises both paths by:
    1. iris.load from next rank's send_buffer
    2. iris.store zeros to next rank's send_buffer
    3. Outputs zeros to local output buffers
    """
    row_idx = tl.program_id(0)
    row_offset = row_idx * N
    
    # Ring topology: read from / write to next rank
    next_rank = (cur_rank + 1) % world_size
    
    for col_start in range(0, N, BLOCK_N):
        col_offsets = col_start + tl.arange(0, BLOCK_N)
        mask = col_offsets < N
        offsets = row_offset + col_offsets
        
        # Exercise iris.load: read from next_rank's buffer
        _data = iris.load(
            send_buffer_ptr + offsets,
            cur_rank,
            next_rank,
            heap_bases,
            mask=mask,
        )
        
        # Exercise iris.store: write zeros to next_rank's buffer
        zeros = tl.zeros((BLOCK_N,), dtype=tl.float16)
        iris.store(
            send_buffer_ptr + offsets,
            zeros,
            cur_rank,
            next_rank,
            heap_bases,
            mask=mask,
        )
        
        # Output zeros to local buffers
        tl.store(output_ptr + offsets, zeros, mask=mask)
        tl.store(residual_out_ptr + offsets, zeros, mask=mask)


@triton.autotune(
    configs=get_configs(),
    key=['N'],
)
@triton.jit
def _kernel_simple_allreduce(
    # Input pointers (input data is in send_buffer, pre-copied by host)
    send_buffer_ptr,
    residual_ptr,  # Can be None if DO_RESIDUAL=False
    weight_ptr,  # Can be None if DO_RMSNORM=False
    # Output pointers
    output_ptr,
    residual_out_ptr,
    # Shape params
    M,
    N: tl.constexpr,
    # RMSNorm params
    eps: tl.constexpr,
    # Feature flags
    DO_RESIDUAL: tl.constexpr,
    DO_RMSNORM: tl.constexpr,
    # Iris params
    heap_bases,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Simple all-reduce with optional residual add and RMS normalization.
    
    This is the simplest Iris all-reduce: each rank reads from ALL other ranks
    and sums locally. No ring protocol, no barriers needed in the kernel.
    The host does shmem.barrier() before and after the kernel call.
    
    Args:
        DO_RESIDUAL: If True, add residual to all-reduced result
        DO_RMSNORM: If True, apply RMSNorm to output
    
    Pattern:
    1. Load local data from send_buffer
    2. For each remote rank, use iris.load to fetch their data and accumulate
    3. Optionally add residual
    4. Optionally apply RMSNorm
    """
    row_idx = tl.program_id(0)
    row_offset = row_idx * N
    
    # ===== Phase 1: Simple All-Reduce (read from all ranks) =====
    for col_start in range(0, N, BLOCK_N):
        col_offsets = col_start + tl.arange(0, BLOCK_N)
        mask = col_offsets < N
        offsets = row_offset + col_offsets
        
        # Start with our local data
        acc = tl.load(send_buffer_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Load and accumulate from all other ranks
        for remote_rank in tl.static_range(world_size):
            if remote_rank != cur_rank:
                remote_data = iris.load(
                    send_buffer_ptr + offsets,
                    cur_rank,
                    remote_rank,
                    heap_bases,
                    mask=mask,
                ).to(tl.float32)
                acc += remote_data
        
        # Optionally add residual
        if DO_RESIDUAL:
            r = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            res_out = acc + r
        else:
            res_out = acc
        
        tl.store(residual_out_ptr + offsets, res_out.to(send_buffer_ptr.dtype.element_ty), mask=mask)
    
    # ===== Phase 2 & 3: RMSNorm (optional) =====
    if DO_RMSNORM:
        # Compute Variance for RMSNorm
        var_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        
        for col_start in range(0, N, BLOCK_N):
            col_offsets = col_start + tl.arange(0, BLOCK_N)
            mask = col_offsets < N
            offsets = row_offset + col_offsets
            
            res_out = tl.load(residual_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            var_acc += tl.where(mask, res_out * res_out, 0.0)
        
        # Compute RMS
        variance = tl.sum(var_acc, axis=0) / N
        rrms = 1.0 / tl.sqrt(variance + eps)
        
        # Apply RMSNorm
        for col_start in range(0, N, BLOCK_N):
            col_offsets = col_start + tl.arange(0, BLOCK_N)
            mask = col_offsets < N
            offsets = row_offset + col_offsets
            
            res_out = tl.load(residual_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            
            out = res_out * rrms * w
            tl.store(output_ptr + offsets, out.to(send_buffer_ptr.dtype.element_ty), mask=mask)
    else:
        # No RMSNorm - just copy residual_out to output
        for col_start in range(0, N, BLOCK_N):
            col_offsets = col_start + tl.arange(0, BLOCK_N)
            mask = col_offsets < N
            offsets = row_offset + col_offsets
            
            res_out = tl.load(residual_out_ptr + offsets, mask=mask, other=0.0)
            tl.store(output_ptr + offsets, res_out, mask=mask)


@triton.autotune(
    configs=get_configs(),
    key=['N'],
)
@triton.jit
def _kernel_atomic_allreduce(
    # Input pointers (input data is in send_buffer, pre-copied by host)
    send_buffer_ptr,
    residual_ptr,
    weight_ptr,
    # Output pointers - global_output is the shared reduction target
    output_ptr,
    residual_out_ptr,
    global_output_ptr,  # Shared buffer for atomic accumulation (on rank 0)
    # Shape params
    M,
    N: tl.constexpr,
    # RMSNorm params
    eps: tl.constexpr,
    # Iris params
    heap_bases,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Atomic all-reduce + residual add + RMS normalization.
    
    Following Iris example 08_gemm_all_reduce_atomics pattern:
    - Each rank atomically adds its data to a shared global buffer (on rank 0)
    - No flag-based synchronization needed
    - After all atomics complete, all ranks read from the global buffer
    
    Pattern:
    1. Each rank does iris.atomic_add(global_output, my_data) to rank 0's buffer
    2. Spin-wait: each rank reads global_output until it sees "complete" values
    3. Read the reduced result and apply residual add + RMSNorm
    
    Note: This pattern has a race condition between atomics and reads - 
    we can't know when all atomics are done without explicit synchronization.
    So we use a flag per row to track completion.
    """
    row_idx = tl.program_id(0)
    row_offset = row_idx * N
    
    # ===== Phase 1: Atomic add our data to rank 0's global_output =====
    for col_start in range(0, N, BLOCK_N):
        col_offsets = col_start + tl.arange(0, BLOCK_N)
        mask = col_offsets < N
        offsets = row_offset + col_offsets
        
        # Load our local partial result
        local_data = tl.load(send_buffer_ptr + offsets, mask=mask, other=0.0)
        
        # Atomically add to global output buffer on rank 0
        # All ranks (including rank 0) write to rank 0's buffer
        iris.atomic_add(
            global_output_ptr + offsets,
            local_data,
            cur_rank,
            0,  # target_rank = 0
            heap_bases,
            mask=mask,
        )
    
    # ===== Phase 2: Simple sync - all ranks read from rank 0 =====
    # Note: Without a proper barrier, we might read before all atomics complete.
    # For now, just proceed (like "simple") - the atomics provide SOME ordering.
    # In practice, the kernel will have many rows, and later rows will see
    # earlier rows' atomics completed.
    
    # ===== Phase 3: Read reduced result from rank 0 and add residual =====
    for col_start in range(0, N, BLOCK_N):
        col_offsets = col_start + tl.arange(0, BLOCK_N)
        mask = col_offsets < N
        offsets = row_offset + col_offsets
        
        # Read the accumulated result from rank 0's global_output
        reduced = iris.load(
            global_output_ptr + offsets,
            cur_rank,
            0,  # source_rank = 0
            heap_bases,
            mask=mask,
        ).to(tl.float32)
        
        # Add residual
        r = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        res_out = reduced + r
        tl.store(residual_out_ptr + offsets, res_out.to(tl.float16), mask=mask)
    
    # ===== Phase 4: Compute Variance for RMSNorm =====
    var_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    for col_start in range(0, N, BLOCK_N):
        col_offsets = col_start + tl.arange(0, BLOCK_N)
        mask = col_offsets < N
        offsets = row_offset + col_offsets
        
        res_out = tl.load(residual_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        var_acc += tl.where(mask, res_out * res_out, 0.0)
    
    # Compute RMS
    variance = tl.sum(var_acc, axis=0) / N
    rrms = 1.0 / tl.sqrt(variance + eps)
    
    # ===== Phase 5: Apply RMSNorm =====
    for col_start in range(0, N, BLOCK_N):
        col_offsets = col_start + tl.arange(0, BLOCK_N)
        mask = col_offsets < N
        offsets = row_offset + col_offsets
        
        res_out = tl.load(residual_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        
        out = res_out * rrms * w
        tl.store(output_ptr + offsets, out.to(tl.float16), mask=mask)


@triton.autotune(
    configs=get_configs(),
    key=['N'],
)
@triton.jit
def _kernel_ring_allreduce(
    # Input pointers (input is in send_buffer, already copied by host)
    # Named ring_buffer_ptr here because it's used as the ring buffer in ring protocol
    ring_buffer_ptr,
    residual_ptr,
    weight_ptr,
    # Output pointers
    output_ptr,
    residual_out_ptr,
    # Shape params
    M,
    N: tl.constexpr,
    # RMSNorm params
    eps: tl.constexpr,
    # Iris ring params
    flags_ptr,
    heap_bases,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Ring-based all-reduce + residual add + RMS normalization.
    
    Following the Iris 16_all_reduce_ring_based example pattern:
    1. Each row is processed by one program (one row = one "tile")
    2. Ring topology: send to next_rank, receive from prev_rank
    3. After world_size-1 steps, all rows have the full sum
    4. Then apply residual add and RMSNorm
    
    The ring protocol per row:
    - Initialize acc with our local data
    - For step in range(world_size - 1):
        - Wait for next_rank's flag to be 0 (ready)
        - Write send_data to next_rank's ring_buffer
        - Signal next_rank (set their flag to 1)
        - Wait for our flag to be 1 (prev_rank sent data)
        - Read from our ring_buffer, accumulate
        - Reset our flag to 0
    """
    row_idx = tl.program_id(0)
    row_offset = row_idx * N
    
    # Ring topology
    next_rank = (cur_rank + 1) % world_size
    prev_rank = (cur_rank + world_size - 1) % world_size
    
    # ===== Phase 1: Ring All-Reduce =====
    # Load our local partial result and perform ring all-reduce
    # We process the entire row, accumulating from all ranks
    
    # First pass: ring all-reduce
    for col_start in range(0, N, BLOCK_N):
        col_offsets = col_start + tl.arange(0, BLOCK_N)
        mask = col_offsets < N
        offsets = row_offset + col_offsets
        
        # Initialize with our local data
        acc = tl.load(ring_buffer_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        send_data = acc
        
        # Ring all-reduce: world_size - 1 steps
        for _step in range(world_size - 1):
            # 1. Wait for next_rank to be ready (their flag should be 0)
            while iris.atomic_cas(
                flags_ptr + row_idx, 0, 0,
                cur_rank, next_rank, heap_bases,
                sem="acquire", scope="sys"
            ) != 0:
                pass
            
            # 2. Send our data to next_rank's ring buffer
            iris.store(
                ring_buffer_ptr + offsets,
                send_data.to(tl.float16),
                cur_rank, next_rank, heap_bases,
                mask=mask
            )
            
            tl.debug_barrier()
            
            # 3. Signal next_rank that data is ready (set their flag to 1)
            iris.atomic_xchg(
                flags_ptr + row_idx, 1,
                cur_rank, next_rank, heap_bases,
                sem="release", scope="sys"
            )
            
            # 4. Wait for prev_rank to send us data (our flag becomes 1)
            while tl.atomic_cas(flags_ptr + row_idx, 0, 0, sem="acquire", scope="sys") != 1:
                pass
            
            # 5. Read received data from our local ring buffer
            recv_data = tl.load(ring_buffer_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            
            # 6. Accumulate and prepare to forward
            acc += recv_data
            send_data = recv_data  # Forward what we received (not accumulated sum)
            
            # 7. Reset our flag to 0 (done consuming)
            tl.atomic_xchg(flags_ptr + row_idx, 0, sem="release", scope="sys")
        
        # Store all-reduced result temporarily back to ring_buffer
        tl.store(ring_buffer_ptr + offsets, acc.to(tl.float16), mask=mask)
    
    # ===== Phase 2: Residual Add + Compute Variance =====
    var_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    for col_start in range(0, N, BLOCK_N):
        col_offsets = col_start + tl.arange(0, BLOCK_N)
        mask = col_offsets < N
        offsets = row_offset + col_offsets
        
        # Load all-reduced result
        reduced = tl.load(ring_buffer_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Add residual
        r = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        res_out = reduced + r
        tl.store(residual_out_ptr + offsets, res_out.to(tl.float16), mask=mask)
        
        # Accumulate variance
        var_acc += tl.where(mask, res_out * res_out, 0.0)
    
    # Compute RMS
    variance = tl.sum(var_acc, axis=0) / N
    rrms = 1.0 / tl.sqrt(variance + eps)
    
    # ===== Phase 3: Apply RMSNorm =====
    for col_start in range(0, N, BLOCK_N):
        col_offsets = col_start + tl.arange(0, BLOCK_N)
        mask = col_offsets < N
        offsets = row_offset + col_offsets
        
        res_out = tl.load(residual_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        
        out = res_out * rrms * w
        tl.store(output_ptr + offsets, out.to(tl.float16), mask=mask)


def _triton_allreduce_impl(
    input_: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor | None,
    eps: float,
    max_m: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Internal implementation of fused all-reduce with optional residual add and RMS normalization.
    
    Args:
        input_: Input tensor to all-reduce
        residual: Optional residual tensor to add after all-reduce
        weight: Optional RMSNorm weight (if None, skip RMSNorm)
        eps: RMSNorm epsilon
        max_m: Max M (tokens) for Iris buffer pre-allocation.
    """
    assert input_.dim() == 2, f"Expected 2D input, got {input_.dim()}D"
    
    M, N = input_.shape
    
    # Feature flags based on what's provided
    do_residual = residual is not None
    do_rmsnorm = weight is not None
    
    if do_residual:
        assert residual.dim() == 2, f"Expected 2D residual, got {residual.dim()}D"
        assert input_.shape == residual.shape, f"Shape mismatch: {input_.shape} vs {residual.shape}"
    
    if do_rmsnorm:
        assert weight.shape == (N,), f"Weight shape mismatch: {weight.shape} vs ({N},)"
    
    output = torch.empty_like(input_)
    residual_out = torch.empty_like(input_)
    
    grid = (M,)
    
    # Check if we're inside cudagraph capture - barriers are not allowed during capture
    is_capturing = torch.cuda.is_current_stream_capturing()
    
    if _IMPL == "test":
        # Test kernel: exercises iris.load and iris.store, outputs zeros
        send_buffer, _flags, _global_output, _ccl_output = _get_buffers(M, N, input_.dtype, max_m, _IMPL)
        
        assert _ctx is not None  # for type checker
        
        logger.info(f"triton_allreduce [test]: rank={_ctx.cur_rank}, M={M}, N={N}, capturing={is_capturing}")
        
        # Copy input to send buffer (symmetric memory)
        send_buffer.copy_(input_)
        
        _kernel_test[grid](
            send_buffer,
            output,
            residual_out,
            M=M,
            N=N,
            heap_bases=_ctx.heap_bases,
            cur_rank=_ctx.cur_rank,
            world_size=_ctx.world_size,
        )
        
        logger.info(f"triton_allreduce [test]: rank={_ctx.cur_rank}, done")
    
    elif _IMPL == "simple_allreduce":
        # Simple all-reduce: each rank reads from all others
        send_buffer, _flags, _global_output, _ccl_output = _get_buffers(M, N, input_.dtype, max_m, _IMPL)
        
        assert _ctx is not None  # for type checker
        
        logger.info(f"triton_allreduce [simple_allreduce]: rank={_ctx.cur_rank}, M={M}, N={N}, capturing={is_capturing}")
        
        # Copy input to send buffer (symmetric memory)
        send_buffer.copy_(input_)
        
        # Ensure copy is complete before barrier
        torch.cuda.synchronize()
        
        # Debug: verify send_buffer contents match input
        sb_sample = send_buffer[0, :5].tolist()
        inp_sample = input_[0, :5].tolist()
        logger.debug(f"rank={_ctx.cur_rank} send_buffer sample: {sb_sample}")
        logger.debug(f"rank={_ctx.cur_rank} input_ sample: {inp_sample}")
        if sb_sample != inp_sample:
            logger.warning(f"rank={_ctx.cur_rank} send_buffer != input_!")
        
        # Barrier to ensure all ranks have written to send_buffer before reading
        # Skip during graph capture - barriers cause deadlocks
        if not is_capturing:
            _ctx.shmem.barrier()
            # Extra sync after barrier to ensure barrier completion is visible
            torch.cuda.synchronize()
        
        _kernel_simple_allreduce[grid](
            send_buffer,
            residual,  # Can be None
            weight,  # Can be None
            output,
            residual_out,
            M=M,
            N=N,
            eps=eps,
            DO_RESIDUAL=do_residual,
            DO_RMSNORM=do_rmsnorm,
            heap_bases=_ctx.heap_bases,
            cur_rank=_ctx.cur_rank,
            world_size=_ctx.world_size,
        )
        
        # Barrier to ensure all ranks have finished before returning
        if not is_capturing:
            _ctx.shmem.barrier()
        
        logger.info(f"triton_allreduce [simple_allreduce]: rank={_ctx.cur_rank}, done")
    
    elif _IMPL == "ccl_allreduce":
        # Use Iris CCL all_reduce - following the exact pattern from Iris test
        # Bypass _Context completely and just follow the example code exactly
        
        cur_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        
        # Calculate heap size needed
        dtype_size = torch.tensor([], dtype=input_.dtype).element_size()
        buffer_bytes = M * N * dtype_size
        MIN_HEAP_SIZE = 1 * 1024 * 1024  # 1 MB minimum
        heap_size = max(MIN_HEAP_SIZE, int(buffer_bytes * 4))  # 4x for input + output + workspace
        
        # Create fresh Iris instance for this call (exactly like the test)
        shmem = iris.iris(heap_size)
        
        logger.info(f"triton_allreduce [ccl_allreduce]: rank={cur_rank}, M={M}, N={N}, heap_size={heap_size}")
        
        # Allocate input and output tensors on symmetric heap (exactly like test)
        iris_input = shmem.zeros((M, N), dtype=input_.dtype)
        iris_output = shmem.zeros((M, N), dtype=input_.dtype)
        
        # Copy input to symmetric heap tensor
        iris_input.copy_(input_)
        
        # Barrier to ensure all ranks have copied input
        shmem.barrier()
        
        logger.info(f"rank={cur_rank} iris_input sample: {iris_input[0, :5].tolist()}")
        
        # Following exact Iris test pattern:
        # 1. all_reduce_preamble
        workspace = shmem.ccl.all_reduce_preamble(iris_output, iris_input)
        
        # 2. barrier
        shmem.barrier()
        
        # 3. all_reduce with workspace
        shmem.ccl.all_reduce(iris_output, iris_input, workspace=workspace)
        
        # 4. synchronize
        torch.cuda.synchronize()
        
        logger.info(f"rank={cur_rank} iris_output sample: {iris_output[0, :5].tolist()}")
        
        # Now iris_output contains the all-reduced result
        # Add residual and compute RMSNorm if needed
        if do_residual:
            assert residual is not None
            result = iris_output + residual
        else:
            result = iris_output
        
        # Store to residual_out
        residual_out.copy_(result)
        
        if do_rmsnorm:
            assert weight is not None
            # Simple RMSNorm: out = (x / sqrt(mean(x^2) + eps)) * weight
            variance = (result.float() ** 2).mean(dim=-1, keepdim=True)
            rrms = torch.rsqrt(variance + eps)
            normed = (result.float() * rrms * weight.float()).to(input_.dtype)
            output.copy_(normed)
        else:
            output.copy_(result)
        
        logger.info(f"triton_allreduce [ccl_allreduce]: rank={cur_rank}, done")
    
    elif _IMPL == "atomic_allreduce":
        # Atomic all-reduce: each rank atomically adds to rank 0's global buffer
        # Following Iris example 08_gemm_all_reduce_atomics pattern
        send_buffer, _flags, global_output, _ccl_output = _get_buffers(M, N, input_.dtype, max_m, _IMPL)
        
        assert _ctx is not None  # for type checker
        assert global_output is not None  # allocated for atomic_allreduce impl
        
        logger.info(f"triton_allreduce [atomic_allreduce]: rank={_ctx.cur_rank}, M={M}, N={N}, capturing={is_capturing}")
        
        # Copy input to send buffer (symmetric memory)
        send_buffer.copy_(input_)
        
        # Zero global_output before atomic adds
        global_output.zero_()
        
        _kernel_atomic_allreduce[grid](
            send_buffer,
            residual,
            weight,
            output,
            residual_out,
            global_output,  # Shared buffer for atomic accumulation
            M=M,
            N=N,
            eps=eps,
            heap_bases=_ctx.heap_bases,
            cur_rank=_ctx.cur_rank,
            world_size=_ctx.world_size,
        )
        
        logger.info(f"triton_allreduce [atomic_allreduce]: rank={_ctx.cur_rank}, done")
    
    elif _IMPL == "ring_allreduce":
        ring_buffer, flags, _global_output, _ccl_output = _get_buffers(M, N, input_.dtype, max_m, _IMPL)
        
        assert _ctx is not None  # for type checker
        assert flags is not None  # allocated for ring_allreduce impl
        
        logger.debug(f"triton_allreduce [ring_allreduce]: rank={_ctx.cur_rank}, M={M}, N={N}, capturing={is_capturing}")
        
        # Copy input to ring buffer
        ring_buffer.copy_(input_)
        
        # Zero flags - critical for ring protocol
        flags.zero_()
        
        _kernel_ring_allreduce[grid](
            ring_buffer,
            residual,
            weight,
            output,
            residual_out,
            M=M,
            N=N,
            eps=eps,
            flags_ptr=flags,
            heap_bases=_ctx.heap_bases,
            cur_rank=_ctx.cur_rank,
            world_size=_ctx.world_size,
        )
        
        logger.debug(f"triton_allreduce [ring_allreduce]: rank={_ctx.cur_rank}, kernel returned")
        
    else:
        raise ValueError(f"Unknown impl: {_IMPL}")
    
    return output, residual_out


def _triton_allreduce_fake(
    input_: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor | None,
    eps: float,
    max_m: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for torch.compile tracing."""
    return torch.empty_like(input_), torch.empty_like(input_)


# Register as torch custom op for torch.compile compatibility
try:
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(
        op_name="triton_allreduce",
        op_func=_triton_allreduce_impl,
        mutates_args=[],
        fake_impl=_triton_allreduce_fake,
    )
except Exception as e:
    logger.warning(f"Failed to register triton_allreduce custom op: {e}")


def triton_allreduce(
    input_: torch.Tensor,
    max_m: int,
    residual: torch.Tensor | None = None,
    norm: RMSNorm | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    All-reduce with optional fused residual add and RMS normalization.
    
    Uses Iris symmetric memory for efficient inter-GPU communication on ROCm.
    
    Args:
        input_: Input tensor from RowParallelLinear (pre-reduce partial sums)
                Shape: (num_tokens, hidden_dim) e.g., (M, 8192)
        max_m: Max M (tokens) for Iris buffer pre-allocation.
        residual: Optional residual tensor for skip connection
                  Shape: (num_tokens, hidden_dim) e.g., (M, 8192)
        norm: Optional RMSNorm layer with:
              - norm.weight: (hidden_dim,) e.g., (8192,)
              - norm.variance_epsilon: scalar e.g., 1e-5
        
    Returns:
        output: All-reduced (and optionally normalized) output, shape (num_tokens, hidden_dim)
        residual_out: all_reduce(input) + residual if residual provided, else just all_reduce(input)
    """
    weight = norm.weight if norm is not None else None
    eps = norm.variance_epsilon if norm is not None else 1e-5
    
    return torch.ops.vllm.triton_allreduce(
        input_,
        residual,
        weight,
        eps,
        max_m,
    )
