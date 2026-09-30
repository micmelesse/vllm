// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// A THREAD'S SHARE OF A ROW, and the primitives a kernel reads and writes it with. Each one exists
// to prevent a footgun that cost a run to find:
//   - A load under a runtime `if` cannot be hoisted past the branch, so loads meant to be in
//     flight together wait one at a time. `load` never branches: every address is clamped to a
//     real pack, and a pack past the row is weighted zero (`Share::in`), not skipped.
//   - A runtime index into a local array puts the array in scratch. Nothing here takes one.
// Stores do not hold up loads, so `store` is guarded.

#pragma once

#include "memory.cuh"
#include "pack.cuh"

namespace hip_comms {

// Packs threadIdx.x + k * blockDim.x, k < K, of a row `packs` long: `at` clamped into the row, `in`
// 1 for a pack inside it and 0 past its end.
template <int K>
struct Share {
  int at[K];
  float in[K];
};

template <int K>
DINLINE Share<K> share(int packs) {
  Share<K> sh;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    sh.at[k]    = i < packs ? i : packs - 1;
    sh.in[k]    = i < packs ? 1.0f : 0.0f;
  }
  return sh;
}

// This thread's packs of `row`, every load issued (a pack past the row reads the last one). Any
// element type: an fp32 weight's pack is 32 bytes.
template <int K, typename V>
DINLINE void load(const V* row, const Share<K>& sh, V (&out)[K]) {
#pragma unroll
  for (int k = 0; k < K; ++k) out[k] = row[sh.at[k]];
}

// This thread's packs into `row`, those inside it.
template <int K, typename V>
DINLINE void store(V* row, const Share<K>& sh, const V (&v)[K]) {
#pragma unroll
  for (int k = 0; k < K; ++k)
    if (sh.in[k] != 0.0f) row[sh.at[k]] = v[k];
}

// A pack as fp32, and fp32 rounded once back to a pack of T.
template <typename T>
DINLINE void unpack(const typename traits<T>::V& v, float (&f)[traits<T>::N]) {
#pragma unroll
  for (int j = 0; j < traits<T>::N; ++j) f[j] = static_cast<float>(v.d[j]);
}

template <typename T>
DINLINE typename traits<T>::V round_pack(const float (&f)[traits<T>::N]) {
  typename traits<T>::V v;
#pragma unroll
  for (int j = 0; j < traits<T>::N; ++j) v.d[j] = static_cast<T>(f[j]);
  return v;
}

}  // namespace hip_comms
