// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// One-shot pull all-reduce, then RMSNorm, then a GEMM whose result is added into an
// output: the tail of Kimi-K3's latent MoE (`fused_all_reduce.latent_tail`).

#pragma once

#include "p2p/p2p.cuh"
#include "common/memory.cuh"
#include "common/reduce.cuh"
#include "common/row.cuh"
#include "launch.cuh"

namespace hip_comms {

// Every rank reduces and norms every row into `workspace` ([rows, packs] of its own); a
// grid barrier; the GEMM over every row. At most kGemmRows rows: one GEMM pass.
// kLanesPerCol is the GEMM's lanes per column (tune.cuh).
template <typename T, int ngpus, int kLanesPerCol, int kRowPacks>
__global__ void __launch_bounds__(gemm_max_threads(kLanesPerCol), 1)
    all_reduce_pull_one_shot_rms_norm_gemm_add(
    p2p::DevComm p, const T* __restrict__ norm_w, float eps, const T* __restrict__ gemm_w,
    int n_cols, T* __restrict__ out, int64_t out_stride, int out_col0,
    T* __restrict__ workspace, int rows, int packs) {
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  const V* weight        = reinterpret_cast<const V*>(norm_w);
  V* normed              = reinterpret_cast<V*>(workspace);
  const auto sh          = share<kRowPacks>(packs);
  // The GEMM's layout: a column's kLanesPerCol lanes split its K, tiles of kTile columns.
  constexpr int kTile = kWaveSize / kLanesPerCol;
  static_assert(kTile * kLanesPerCol == kWaveSize, "a column's lanes must divide a wave");
  __shared__ float partial[gemm_max_waves(kLanesPerCol)][kGemmRows][kTile];
  __shared__ V xs[kGemmRows][kGemmChunk];
  const int lane   = threadIdx.x % kWaveSize;
  const int wave   = threadIdx.x / kWaveSize;
  const int waves  = blockDim.x / kWaveSize;
  const int column = lane % kTile;
  const int splits = waves * kLanesPerCol;
  const int split  = wave * kLanesPerCol + lane / kTile;
  const V* wv      = reinterpret_cast<const V*>(gemm_w);
  const int tiles  = (n_cols + kTile - 1) / kTile;

  // 1. Every rank's buffers, then wait until every peer has launched, so its input is ready.
  const auto peers = p2p::peers<T, ngpus>(p);
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);
  const auto read = [&](int r, int64_t i) { return p2p::read_input(peers[r], i); };

  // 2. Each of this block's rows: read it from every rank in rank order, sum, norm, into the
  //    workspace.
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    V sum[kRowPacks];
    sum_row<T, ngpus>(read, row, packs, sh, sum);
    // The norm, rounding as vLLM's reference rms_norm does (weight in T):
    //   out = T(T(s * rsqrt(mean(s^2) + eps)) * float(w)), s = float(T(sum over ranks))
    float s[kRowPacks][NL];
    float ss[1] = {0.0f};
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k) {
      unpack<T>(sum[k], s[k]);
#pragma unroll
      for (int j = 0; j < NL; ++j) ss[0] += sh.in[k] * s[k][j] * s[k][j];
    }
    block_sum(ss);
    const float scale = rsqrtf(ss[0] * inv_hidden + eps);
    V w[kRowPacks], normed_row[kRowPacks];
    load(weight, sh, w);
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k)
#pragma unroll
      for (int j = 0; j < NL; ++j) {
        const float x      = static_cast<float>(static_cast<T>(s[k][j] * scale));
        normed_row[k].d[j] = static_cast<T>(x * static_cast<float>(w[k].d[j]));
      }
    store(normed + int64_t{row} * packs, sh, normed_row);
  }

  // 3. The GEMM reads rows other blocks of this rank wrote.
  p2p::barrier<ngpus, p2p::Among::grid, p2p::Ensure::visible>(p);

  // 4. The GEMM over every row, one pass. THE GEMM, a skinny one: out[r, out_col0 + n] =
  // T(float(out[r, out_col0 + n]) + sum_k x[r][k] * w[n][k]) for the pass's rows, the sum in fp32
  // and rounded once. x is staged in LDS a K-chunk at a time (coalesced, once per block per chunk),
  // so the hot loop's row reads are LDS reads. A lane keeps one column's row sums in registers; K
  // is split over the kLanesPerCol lanes of a column and over the waves; shuffles and an LDS pass
  // add the splits; blocks stride over tiles of kWaveSize / kLanesPerCol columns. The order of the
  // sum differs from hipBLASLt's, so a result agrees to the rounding of the last bits, not bitwise.
  {
    const int pass_rows = rows;
    T* pass_out         = out;
    for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
      const int n   = tile * kTile + column;
      const V* wrow = wv + static_cast<int64_t>(n < n_cols ? n : 0) * packs;
      float acc[kGemmRows];
#pragma unroll
      for (int r = 0; r < kGemmRows; ++r) acc[r] = 0.0f;
      for (int k0 = 0; k0 < packs; k0 += kGemmChunk) {
        const int chunk = min(kGemmChunk, packs - k0);
        // Rows past the pass's are never staged; their sums read stale LDS and are never stored.
        for (int i = threadIdx.x; i < pass_rows * chunk; i += blockDim.x)
          xs[i / chunk][i % chunk] = (normed + (i / chunk) * packs)[k0 + i % chunk];
        __syncthreads();
        for (int k = split; k < chunk; k += splits) {
          const V wx = wrow[k0 + k];
          float wf[NL];
          unpack<T>(wx, wf);
#pragma unroll
          for (int r = 0; r < kGemmRows; ++r) {
            const V xr = xs[r][k];
#pragma unroll
            for (int j = 0; j < NL; ++j) acc[r] += static_cast<float>(xr.d[j]) * wf[j];
          }
        }
        // Before the next chunk overwrites `xs`.
        __syncthreads();
      }
      // A column's lanes are kTile apart in the wave.
#pragma unroll
      for (int r = 0; r < kGemmRows; ++r)
#pragma unroll
        for (int d = kTile; d < kWaveSize; d <<= 1) acc[r] += __shfl_xor(acc[r], d, kWaveSize);
      if (lane < kTile) {
#pragma unroll
        for (int r = 0; r < kGemmRows; ++r) partial[wave][r][column] = acc[r];
      }
      __syncthreads();
      for (int i = threadIdx.x; i < kGemmRows * kTile; i += blockDim.x) {
        const int r   = i / kTile;
        const int col = tile * kTile + i % kTile;
        if (r < pass_rows && col < n_cols) {
          float v = 0.0f;
          for (int q = 0; q < waves; ++q) v += partial[q][r][i % kTile];
          T* at = pass_out + r * out_stride + out_col0 + col;
          *at   = static_cast<T>(static_cast<float>(*at) + v);
        }
      }
      // Before the next tile overwrites `partial`.
      __syncthreads();
    }
  }


  // 5. No rank may overwrite its input until every peer has read it.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::read>(p);
}

}  // namespace hip_comms
