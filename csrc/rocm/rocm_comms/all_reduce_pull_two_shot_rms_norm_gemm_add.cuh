// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Two-shot pull all-reduce, then RMSNorm, then a GEMM whose result is added into an
// output: the tail of Kimi-K3's latent MoE (`fused_all_reduce.latent_tail`).

#pragma once

#include "p2p/p2p.cuh"
#include "common/memory.cuh"
#include "common/reduce.cuh"
#include "common/row.cuh"
#include "launch.cuh"

namespace hip_comms {

// Each rank reduces and norms the rows it owns into its scratch, row-major; after the sync
// every rank copies every normed row into `workspace` ([rows, packs] of its own); a grid sync;
// the GEMM over every row, kGemmRows per pass. THE SAME BLOCK AND THREAD INDEX A PACK IN
// BOTH PHASES: after the sync a block may read only what the same block on a peer wrote.
template <typename T, int ngpus, int kLanesPerCol, int kRowPacks>
__global__ void __launch_bounds__(gemm_max_threads(kLanesPerCol), 1)
    all_reduce_pull_two_shot_rms_norm_gemm_add(
    p2p::DevComm p, const T* __restrict__ norm_w, float eps, const T* __restrict__ gemm_w,
    int n_cols, T* __restrict__ out, int64_t out_stride, int out_col0,
    T* __restrict__ workspace, int rows, int packs) {
  using V                = typename traits<T>::V;
  constexpr int NL       = traits<T>::N;
  const float inv_hidden = 1.0f / static_cast<float>(packs * NL);
  const V* weight        = reinterpret_cast<const V*>(norm_w);
  V* normed              = reinterpret_cast<V*>(workspace);
  const int slice_rows   = (rows + ngpus - 1) / ngpus;
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

  // 1. Wait until every peer has launched, so its input is ready.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::launched>(p);

  // THE RANKS' POINTERS AFTER THE BARRIER here: held across it, the 8-pack build keeps 68 B of
  // scratch (the ISA gate, 2026-09-30).
  const auto peers = p2p::peers<T, ngpus>(p);
  const auto read = [&](int r, int64_t i) { return p2p::read_input(peers[r], i); };
  const auto self = p2p::self<T, ngpus>(p);

  // 2. This rank's rows: read each from every rank in rank order, sum, norm, into this rank's
  //    scratch.
  const int first = p.rank * slice_rows;
  const int last  = min(first + slice_rows, rows);
  for (int row = first + blockIdx.x; row < last; row += gridDim.x) {
    V sum[kRowPacks];
    sum_row<T, ngpus>(read, row, packs, sh, sum);
    const int64_t at = int64_t{row - first} * packs;
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
#pragma unroll
    for (int k = 0; k < kRowPacks; ++k)
      if (sh.in[k] != 0.0f) p2p::write_scratch(self, at + sh.at[k], normed_row[k]);
  }

  // 3. Every rank's normed rows are visible to its peers.
  p2p::barrier<ngpus, p2p::Among::peers, p2p::Ensure::visible>(p);

  // 4. Every owner's normed rows out of its scratch, into the workspace.
  for (int l = blockIdx.x; l < slice_rows; l += gridDim.x) {
    for (int i = threadIdx.x; i < packs; i += blockDim.x) {
    // EVERY OWNER'S PACK LOADED BEFORE ANY IS STORED: the compiler cannot prove the output
    // and the peers' scratch apart, so a store between two loads holds the next load back
    // until the store is done, and the eight owners' round trips run one after another.
      // Every load unconditional: each rank's scratch holds slice_rows rows, so a slot past the
      // last row is real, and a load under an `if` waits on the one before.
      V got[ngpus];
#pragma unroll
      for (int r = 0; r < ngpus; ++r) got[r] = p2p::read_scratch(peers[r], int64_t{l} * packs + i);
#pragma unroll
      for (int r = 0; r < ngpus; ++r) {
        const int row = r * slice_rows + l;
        if (row < rows) store_global(normed + int64_t{row} * packs + i, got[r]);
      }
    }
  }

  // 5. The GEMM reads rows other blocks of this rank copied.
  p2p::barrier<ngpus, p2p::Among::grid, p2p::Ensure::visible>(p);

  // 6. The GEMM over every row, kGemmRows per pass.
  for (int r0 = 0; r0 < rows; r0 += kGemmRows)
    // THE GEMM, a skinny one: out[r, out_col0 + n] = T(float(out[r, out_col0 + n]) + sum_k x[r][k]
    // * w[n][k]) for the pass's rows, the sum in fp32 and rounded once. x is staged in LDS a
    // K-chunk at a time (coalesced, once per block per chunk), so the hot loop's row reads are LDS
    // reads. A lane keeps one column's row sums in registers; K is split over the kLanesPerCol
    // lanes of a column and over the waves; shuffles and an LDS pass add the splits; blocks stride
    // over tiles of kWaveSize / kLanesPerCol columns. The order of the sum differs from
    // hipBLASLt's, so a result agrees to the rounding of the last bits, not bitwise.
    {
      const int pass_rows = min(kGemmRows, rows - r0);
      T* pass_out         = out + r0 * out_stride;
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
            xs[i / chunk][i % chunk] = (normed + (r0 + i / chunk) * packs)[k0 + i % chunk];
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
}

}  // namespace hip_comms
