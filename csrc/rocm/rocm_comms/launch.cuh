// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// LAUNCH, NO DECISIONS: each op's launch_<op> runs the kernel its launch names (select chose it) on
// the launch's grid, block and stream, with the kernel's arguments from the launch and the peers'
// view of its input. A branch here follows a field of the launch; it chooses nothing.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <hip/hip_runtime.h>

#include <cstdint>

namespace hip_comms {

// A KERNEL CALLED THROUGH ITS OP'S SIGNATURE (types.cuh): each argument is converted to its
// parameter, so a wrong one is a compile error, then launched as `kernel<<<blocks, threads, 0,
// stream>>>(args...)` launches it.
template <typename Signature>
struct Call;
template <typename... P>
struct Call<void (*)(P...)> {
  static void run(const void* kernel, int blocks, int threads, hipStream_t stream, P... args) {
    void* argv[] = {static_cast<void*>(&args)...};
    HIP_CHECK(hipLaunchKernel(kernel, dim3(blocks), dim3(threads), argv, 0, stream));
  }
};

// STAGED: the kernel copies its own input through its staging a pass at a time, so no peer reads
// it where it is; in place, the peers read it through its slot.
inline void launch_all_reduce(Handle& h, const AllReduceLaunch& l) {
  const int64_t packs = l.bytes / kBuild.memory.pack_bytes;
  if (l.staged) {
    const int64_t stage_packs = kBuild.memory.staging_bytes / kBuild.memory.pack_bytes;
    Call<AllReduceStagedKernel>::run(l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream,
                                     h.dev_comm_staged(l.bytes), l.out, packs, l.inp,
                                     stage_packs);
  } else {
    Call<AllReduceKernel>::run(l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream,
                               h.dev_comm(l.inp, l.bytes, l.stream), l.out,
                               static_cast<int>(packs));
  }
}

inline void launch_all_reduce_rms_norm(Handle& h, const AllReduceRmsNormLaunch& l) {
  const int64_t bytes = l.rows * l.hidden * elem_bytes(l.dtype);
  Call<RmsNormKernel>::run(l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream,
                           h.dev_comm(l.inp, bytes, l.stream), l.out, l.weight, l.eps,
                           static_cast<int>(l.rows),
                           static_cast<int>(packs_of(l.hidden, l.dtype)));
}

inline void launch_all_reduce_add_rms_norm(Handle& h, const AllReduceAddRmsNormLaunch& l) {
  const int64_t bytes = l.rows * l.hidden * elem_bytes(l.dtype);
  Call<AddRmsNormKernel>::run(l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream,
                              h.dev_comm(l.inp, bytes, l.stream), l.out, l.residual_out,
                              l.residual, l.weight, l.eps, static_cast<int>(l.rows),
                              static_cast<int>(packs_of(l.hidden, l.dtype)));
}

// THE PULL TWO-SHOT takes its reduce-scatter blocks last; the one-shot and push do not.
inline void launch_all_reduce_add_attn_res_rms_norm(Handle& h,
                                                    const AllReduceAddAttnResRmsNormLaunch& l) {
  const int64_t bytes  = l.rows * l.hidden * elem_bytes(l.dtype);
  const p2p::DevComm p = h.dev_comm(l.inp, bytes, l.stream);
  const int rows = static_cast<int>(l.rows), packs = static_cast<int>(packs_of(l.hidden, l.dtype));
  if (l.algorithm == Algorithm::two_shot && l.direction == Direction::pull) {
    Call<AttnResPullKernel>::run(l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, p,
                                 l.prefix, l.blocks, l.block_stride_m, l.block_stride_r,
                                 l.norm_weight, l.qk_weight, l.out_norm_weight, l.out,
                                 l.num_blocks, l.write_idx, l.eps, l.out_eps, rows, packs,
                                 l.reduce_scatter_blocks);
  } else {
    Call<AttnResKernel>::run(l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream, p,
                             l.prefix, l.blocks, l.block_stride_m, l.block_stride_r,
                             l.norm_weight, l.qk_weight, l.out_norm_weight, l.out, l.num_blocks,
                             l.write_idx, l.eps, l.out_eps, rows, packs);
  }
}

inline void launch_all_reduce_rms_norm_gemm(Handle& h, const AllReduceRmsNormGemmLaunch& l) {
  const int64_t bytes = l.rows * l.hidden * elem_bytes(l.dtype);
  Call<GemmTailKernel>::run(l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream,
                            h.dev_comm(l.inp, bytes, l.stream), l.norm_weight, l.eps,
                            l.gemm_weight, static_cast<int>(l.n_cols), l.out, l.out_stride,
                            l.workspace, static_cast<int>(l.rows),
                            static_cast<int>(packs_of(l.hidden, l.dtype)));
}

inline void launch_all_reduce_rms_norm_gemm_add(Handle& h, const AllReduceRmsNormGemmAddLaunch& l) {
  const int64_t bytes = l.rows * l.hidden * elem_bytes(l.dtype);
  Call<GemmTailKernel>::run(l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream,
                            h.dev_comm(l.inp, bytes, l.stream), l.norm_weight, l.eps,
                            l.gemm_weight, static_cast<int>(l.n_cols), l.out, l.out_stride,
                            l.workspace, static_cast<int>(l.rows),
                            static_cast<int>(packs_of(l.hidden, l.dtype)));
}

// inp's row is [shared | projected | latent].
inline void launch_all_reduce_rms_scale_add(Handle& h, const AllReduceRmsScaleAddLaunch& l) {
  const int64_t bytes = l.rows * (2 * l.hidden + l.latent) * elem_bytes(l.dtype);
  Call<RmsScaleAddKernel>::run(l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream,
                               h.dev_comm(l.inp, bytes, l.stream), l.out, l.eps,
                               static_cast<int>(l.rows),
                               static_cast<int>(packs_of(l.hidden, l.dtype)),
                               static_cast<int>(packs_of(l.latent, l.dtype)));
}

namespace experimental {

// No peers: its own tensors only.
inline void launch_add_attn_res_rms_norm(const AddAttnResRmsNormLaunch& l) {
  Call<AddAttnResKernel>::run(l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream,
                              l.prefix, l.delta, l.blocks, l.block_stride_m, l.block_stride_r,
                              l.norm_weight, l.qk_weight, l.out_norm_weight, l.out, l.num_blocks,
                              l.write_idx, l.eps, l.out_eps, static_cast<int>(l.rows),
                              static_cast<int>(packs_of(l.hidden, l.dtype)));
}

// The probe's: a barrier over the signals only; the ping-pong's flags from this pair's next ones
// (flags only grow); the link traffic over the peers' view of its buffer.
inline void launch_probe_barrier(Handle& h, const ProbeBarrierLaunch& l) {
  Call<ProbeBarrierKernel>::run(l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream,
                                h.dev_comm());
}

inline void launch_ping_pong(Handle& h, const PingPongLaunch& l) {
  const uint32_t base = h.take_flags(l.peer, ping_pong_flags(l.iters));
  Call<PingPongKernel>::run(l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream,
                            h.dev_comm(), l.peer, base, l.iters, l.ticks);
}

inline void launch_link_traffic(Handle& h, const LinkTrafficLaunch& l) {
  Call<LinkTrafficKernel>::run(l.kernel, l.blocks_per_grid, l.threads_per_block, l.stream,
                               h.dev_comm(l.buffer, l.bytes, l.stream), static_cast<int>(l.mode),
                               l.peer, l.pullers, l.bytes / kBuild.memory.pack_bytes, l.sink);
}

}  // namespace experimental

}  // namespace hip_comms
