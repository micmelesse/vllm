// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// The codec, behind p2p.cuh: how a group of kCodecGroupPacks packs looks on the wire -- T itself
// at 16 bits, or QuickReduce's symmetric integers (8, 4) with one fp32 scale per group. No kernel
// uses it today; it is the format a quantized kernel will send.

#pragma once

#ifndef HIP_COMMS_P2P_INTERFACE
#error "include p2p/p2p.cuh, p2p's one interface, not its parts"
#endif

#include "../../common/common.cuh"

namespace hip_comms::p2p {

// A CODEC GROUP: 32 values of a 2-byte T under one fp32 scale (QuickReduce's).
constexpr int kCodecGroupPacks = 4;

namespace impl {

// A group's kCodecGroupPacks packs as floats and back, rounding to T: what a quantizing kernel
// reduces and encodes in, against the packs a row helper takes.
template <typename DTYPE>
DINLINE void floats_of(const typename traits<DTYPE>::V (&v)[kCodecGroupPacks],
                       float (&x)[kCodecGroupPacks * traits<DTYPE>::N]) {
  constexpr int N = traits<DTYPE>::N;
#pragma unroll
  for (int u = 0; u < kCodecGroupPacks; ++u)
#pragma unroll
    for (int j = 0; j < N; ++j) x[u * N + j] = static_cast<float>(v[u].d[j]);
}

template <typename DTYPE>
DINLINE void packs_of(const float (&x)[kCodecGroupPacks * traits<DTYPE>::N],
                      typename traits<DTYPE>::V (&v)[kCodecGroupPacks]) {
  constexpr int N = traits<DTYPE>::N;
#pragma unroll
  for (int u = 0; u < kCodecGroupPacks; ++u)
#pragma unroll
    for (int j = 0; j < N; ++j) v[u].d[j] = static_cast<DTYPE>(x[u * N + j]);
}

// ---------------------------------------------------------------------------------
// THE PUSH KERNELS' CODEC: what a group of kCodecGroupPacks packs (one thread's batch, 32 values
// of a 2-byte T) looks like on the wire. kBits 16 is T itself (no scale); 8 and 4 are
// QuickReduce's symmetric integers with one fp32 scale per group.
// ---------------------------------------------------------------------------------

template <typename DTYPE, int NUM_BITS>
struct Codec {
  using V                    = typename traits<DTYPE>::V;
  static constexpr int N     = traits<DTYPE>::N;
  static constexpr int kVals = kCodecGroupPacks * N;
  static_assert(sizeof(DTYPE) == 2, "the codec is built for 2-byte DTYPE");
  static_assert(NUM_BITS == 16 || NUM_BITS == 8 || NUM_BITS == 4, "16 (DTYPE), INT8 and INT4 are built");
  static constexpr bool kScaled = NUM_BITS < 16;
  // The payload of one group, in 16-byte packs: 32 values x NUM_BITS.
  static constexpr int kPayloadPacks = kVals * NUM_BITS / 8 / 16;
  static constexpr int kMax          = kScaled ? (1 << (NUM_BITS - 1)) - 1 : 0;

  // x -> payload; returns the scale, absmax / kMax (0 for an all-zero group; unused at 16).
  static DINLINE float encode(const float (&x)[kVals], V (&payload)[kPayloadPacks]) {
    if constexpr (!kScaled) {
#pragma unroll
      for (int i = 0; i < kVals; ++i) payload[i / N].d[i % N] = static_cast<DTYPE>(x[i]);
      return 1.0f;
    } else {
      float amax = 0.0f;
#pragma unroll
      for (int i = 0; i < kVals; ++i) amax = fmaxf(amax, fabsf(x[i]));
      const float scale = amax / kMax;
      const float inv   = amax > 0.0f ? kMax / amax : 0.0f;
      unsigned char bytes[kPayloadPacks * 16];
#pragma unroll
      for (int i = 0; i < kVals; ++i) {
        const int q =
            static_cast<int>(fminf(fmaxf(rintf(x[i] * inv), -kMax - 1.0f), kMax));
        if constexpr (NUM_BITS == 8) {
          bytes[i] = static_cast<unsigned char>(q & 0xFF);
        } else if (i % 2 == 0) {
          bytes[i / 2] = static_cast<unsigned char>(q & 0xF);
        } else {
          bytes[i / 2] |= static_cast<unsigned char>((q & 0xF) << 4);
        }
      }
      __builtin_memcpy(payload, bytes, sizeof(bytes));
      return scale;
    }
  }

  // payload, scale -> x.
  static DINLINE void decode(const V (&payload)[kPayloadPacks], float scale,
                             float (&x)[kVals]) {
    if constexpr (!kScaled) {
#pragma unroll
      for (int i = 0; i < kVals; ++i) x[i] = static_cast<float>(payload[i / N].d[i % N]);
    } else {
      unsigned char bytes[kPayloadPacks * 16];
      __builtin_memcpy(bytes, payload, sizeof(bytes));
#pragma unroll
      for (int i = 0; i < kVals; ++i) {
        int q;
        if constexpr (NUM_BITS == 8) {
          q = static_cast<signed char>(bytes[i]);
        } else {
          const int nib = (bytes[i / 2] >> (4 * (i % 2))) & 0xF;
          q             = nib >= 8 ? nib - 16 : nib;
        }
        x[i] = static_cast<float>(q) * scale;
      }
    }
  }
};

}  // namespace impl

}  // namespace hip_comms::p2p
