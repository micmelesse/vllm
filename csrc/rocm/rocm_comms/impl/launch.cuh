// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// LAUNCH, NO DECISIONS: a spec and an op's args to the kernel instance that runs them. What a
// kernel is compiled for is constexpr, so it is a template parameter, and each one is resolved
// here from the call: the world from the handle, T (and a norm's weight type W) from the args, a
// row kernel's packs a thread from its row and block, the GEMM tail's lanes from the build. Every
// combination below is compiled; one that was not raises.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <hip/hip_runtime.h>

#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../kernels/all_reduce_pull_one_shot.cuh"
#include "../kernels/all_reduce_pull_one_shot_add_attn_res_rms_norm.cuh"
#include "../kernels/all_reduce_pull_one_shot_add_rms_norm.cuh"
#include "../kernels/all_reduce_pull_one_shot_rms_norm_gemm_add.cuh"
#include "../kernels/all_reduce_pull_two_shot.cuh"
#include "../kernels/all_reduce_pull_two_shot_add_attn_res_rms_norm.cuh"
#include "../kernels/all_reduce_pull_two_shot_add_rms_norm.cuh"
#include "../kernels/all_reduce_pull_two_shot_rms_norm_gemm_add.cuh"
#include "../kernels/all_reduce_push_two_shot_add_attn_res_rms_norm.cuh"
#include "../kernels/all_reduce_push_two_shot_add_rms_norm.cuh"

namespace hip_comms {

namespace impl {

template <typename T>
struct type {
  using t = T;
};

template <int N>
using constant = std::integral_constant<int, N>;

[[noreturn]] inline void not_built(const std::string& what) {
  throw std::runtime_error("hip_comms: " + what + " not built");
}

template <typename F>
void by_world(int world, F&& f) {
  switch (world) {
    case 2: return f(constant<2>{});
    case 4: return f(constant<4>{});
    case 8: return f(constant<8>{});
    default: not_built("world size " + std::to_string(world));
  }
}

template <typename F>
void by_dtype(DType d, F&& f) {
  switch (d) {
    case DType::f16: return f(type<c10::Half>{});
    case DType::bf16: return f(type<c10::BFloat16>{});
    default: not_built("dtype");
  }
}

// A norm's weight: T itself, or fp32.
template <typename T, typename F>
void by_weight(DType weight, DType dtype, F&& f) {
  if (weight == dtype) return f(type<T>{});
  if (weight == DType::f32) return f(type<float>{});
  not_built("weight dtype");
}

// A row kernel's packs a thread, up to the op's build (kMax), so no build past it is compiled.
template <int kMax, typename F>
void by_row_packs(int k, F&& f) {
  switch (k) {
    case 1: return f(constant<1>{});
    case 2: return f(constant<2>{});
    case 4: if constexpr (kMax >= 4) return f(constant<4>{}); break;
    case 8: if constexpr (kMax >= 8) return f(constant<8>{}); break;
    default: break;
  }
  not_built("row kernel of " + std::to_string(k) + " packs a thread");
}

// A row kernel's packs a thread, up to its own build (the catalog's max_row_packs).
template <Kernel K, typename F>
void at_row_packs(int k, F&& f) {
  by_row_packs<max_row_packs(K)>(k, std::forward<F>(f));
}

[[noreturn]] inline void not_this_ops(Kernel k) {
  throw std::runtime_error("hip_comms: kernel " + std::to_string(static_cast<int>(k)) +
                           " is not this op's");
}

// THE MOST BLOCKS OF `kernel` RESIDENT AT ONCE on this device at `threads`: its occupancy (from
// its compiled registers and LDS) on every CU. A peer barrier spins until the same block on every
// peer arrives, so a grid past it can hang. Asked once a kernel and block.
inline int resident_blocks(const void* kernel, int threads) {
  thread_local std::map<std::tuple<int, const void*, int>, int> known;
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  const auto key = std::make_tuple(device, kernel, threads);
  if (const auto it = known.find(key); it != known.end()) return it->second;
  int per_cu = 0, cus = 0;
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, kernel, threads, 0));
  HIP_CHECK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, device));
  return known[key] = per_cu * cus;
}

// The kernel at the spec's grid and block, on the stream, once its whole grid can be resident.
// hipify reads `<<<...>>>` as text, so it is spelled out.
template <typename... P, typename... A>
void start(void (*kernel)(P...), const KernelSpec& k, hipStream_t stream, A&&... args) {
  const int resident = resident_blocks(reinterpret_cast<const void*>(kernel), k.threads);
  if (k.grid > resident)
    throw std::runtime_error("hip_comms: kernel " + std::to_string(static_cast<int>(k.kernel)) +
                             " at " + std::to_string(k.threads) + " threads holds " +
                             std::to_string(resident) + " blocks resident, not " +
                             std::to_string(k.grid));
  kernel<<<dim3(k.grid), dim3(k.threads), 0, stream>>>(std::forward<A>(args)...);
}

inline int row_packs(const KernelSpec& k, int64_t hidden, DType d) {
  return row_packs_for(k.kernel, hidden * elem_bytes(d) / kPackBytes, k.threads);
}

}  // namespace impl

inline void launch(Handle& h, const KernelSpec& k, const AllReduceArgs& a, hipStream_t s) {
  const p2p::DevComm p = h.dev_comm(a.inp, a.bytes, s);
  const int n          = static_cast<int>(a.bytes / kPackBytes);
  impl::by_world(h.world_size(), [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    impl::by_dtype(a.dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      T* out  = static_cast<T*>(a.out);
      switch (k.kernel) {
        case Kernel::all_reduce_pull_one_shot:
          return impl::start(all_reduce_pull_one_shot<T, NG>, k, s, p, out, n);
        case Kernel::all_reduce_pull_two_shot:
          return impl::start(all_reduce_pull_two_shot<T, NG>, k, s, p, out, n);
        default: impl::not_this_ops(k.kernel);
      }
    });
  });
}

inline void launch(Handle& h, const KernelSpec& k, const NormArgs& a, hipStream_t s) {
  const p2p::DevComm p = h.dev_comm(a.inp, a.rows * a.hidden * elem_bytes(a.dtype), s);
  const int rows       = static_cast<int>(a.rows);
  const int packs      = static_cast<int>(a.hidden * elem_bytes(a.dtype) / kPackBytes);
  const int r          = impl::row_packs(k, a.hidden, a.dtype);
  impl::by_world(h.world_size(), [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    impl::by_dtype(a.dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      impl::by_weight<T>(a.weight_dtype, a.dtype, [&](auto w) {
        using W         = typename decltype(w)::t;
        T* out          = static_cast<T*>(a.out);
        T* res_out      = static_cast<T*>(a.residual_out);
        const T* res    = static_cast<const T*>(a.residual);
        const W* weight = static_cast<const W*>(a.weight);
        using K = Kernel;
        switch (k.kernel) {
          case K::all_reduce_pull_one_shot_rms_norm:
            return impl::at_row_packs<K::all_reduce_pull_one_shot_rms_norm>(r, [&](auto rp) {
              constexpr int R = decltype(rp)::value;
              impl::start(all_reduce_pull_one_shot_rms_norm<T, W, NG, R>, k, s, p, out,
                          weight, a.eps, rows, packs);
            });
          case K::all_reduce_pull_two_shot_rms_norm:
            return impl::at_row_packs<K::all_reduce_pull_two_shot_rms_norm>(r, [&](auto rp) {
              constexpr int R = decltype(rp)::value;
              impl::start(all_reduce_pull_two_shot_rms_norm<T, W, NG, R>, k, s, p, out,
                          weight, a.eps, rows, packs);
            });
          case K::all_reduce_push_two_shot_rms_norm:
            return impl::at_row_packs<K::all_reduce_push_two_shot_rms_norm>(r, [&](auto rp) {
              constexpr int R = decltype(rp)::value;
              impl::start(all_reduce_push_two_shot_rms_norm<T, W, NG, R>, k, s, p, out,
                          weight, a.eps, rows, packs);
            });
          case K::all_reduce_pull_one_shot_add_rms_norm:
            return impl::at_row_packs<K::all_reduce_pull_one_shot_add_rms_norm>(r, [&](auto rp) {
              constexpr int R = decltype(rp)::value;
              impl::start(all_reduce_pull_one_shot_add_rms_norm<T, W, NG, R>, k, s, p,
                          out, res_out, res, weight, a.eps, rows, packs);
            });
          case K::all_reduce_pull_two_shot_add_rms_norm:
            return impl::at_row_packs<K::all_reduce_pull_two_shot_add_rms_norm>(r, [&](auto rp) {
              constexpr int R = decltype(rp)::value;
              impl::start(all_reduce_pull_two_shot_add_rms_norm<T, W, NG, R>, k, s, p,
                          out, res_out, res, weight, a.eps, rows, packs);
            });
          case K::all_reduce_push_two_shot_add_rms_norm:
            return impl::at_row_packs<K::all_reduce_push_two_shot_add_rms_norm>(r, [&](auto rp) {
              constexpr int R = decltype(rp)::value;
              impl::start(all_reduce_push_two_shot_add_rms_norm<T, W, NG, R>, k, s, p,
                          out, res_out, res, weight, a.eps, rows, packs);
            });
          default: impl::not_this_ops(k.kernel);
        }
      });
    });
  });
}

// AttnRes's and the GEMM tail's two kernels share their row builds, so one cap serves both.
static_assert(max_row_packs(Kernel::all_reduce_pull_one_shot_add_attn_res_rms_norm) ==
                  max_row_packs(Kernel::all_reduce_pull_two_shot_add_attn_res_rms_norm) &&
              max_row_packs(Kernel::all_reduce_pull_one_shot_add_attn_res_rms_norm) ==
                  max_row_packs(Kernel::all_reduce_push_two_shot_add_attn_res_rms_norm) &&
              max_row_packs(Kernel::all_reduce_pull_one_shot_rms_norm_gemm_add) ==
                  max_row_packs(Kernel::all_reduce_pull_two_shot_rms_norm_gemm_add) &&
              max_row_packs(Kernel::all_reduce_pull_one_shot_rms_norm_gemm_add) ==
                  max_row_packs(Kernel::all_reduce_pull_one_shot_rms_norm_gemm) &&
              max_row_packs(Kernel::all_reduce_pull_one_shot_rms_norm_gemm_add) ==
                  max_row_packs(Kernel::all_reduce_pull_two_shot_rms_norm_gemm),
              "a kernel with its own row builds needs its own case");

inline void launch(Handle& h, const KernelSpec& k, const AttnResArgs& a, hipStream_t s) {
  const p2p::DevComm p = h.dev_comm(a.inp, a.rows * a.hidden * elem_bytes(a.dtype), s);
  const int rows       = static_cast<int>(a.rows);
  const int packs      = static_cast<int>(a.hidden * elem_bytes(a.dtype) / kPackBytes);
  impl::by_world(h.world_size(), [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    impl::by_dtype(a.dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      impl::at_row_packs<Kernel::all_reduce_pull_one_shot_add_attn_res_rms_norm>(
          impl::row_packs(k, a.hidden, a.dtype), [&](auto r) {
        constexpr int R = decltype(r)::value;
        const auto args = std::make_tuple(
            p, static_cast<T*>(a.prefix), static_cast<T*>(a.blocks), a.block_stride_m,
            a.block_stride_r, static_cast<const T*>(a.norm_weight),
            static_cast<const T*>(a.qk_weight), static_cast<const T*>(a.out_norm_weight),
            static_cast<T*>(a.out), a.num_blocks, a.write_idx, a.eps, a.out_eps, rows, packs);
        const auto run = [&](auto kernel) {
          std::apply([&](auto&&... xs) { impl::start(kernel, k, s, xs...); }, args);
        };
        const bool prefix = a.has_prefix;
        switch (k.kernel) {
          case Kernel::all_reduce_pull_one_shot_add_attn_res_rms_norm:
            return prefix ? run(all_reduce_pull_one_shot_add_attn_res_rms_norm<T, NG, true, R>)
                          : run(all_reduce_pull_one_shot_add_attn_res_rms_norm<T, NG, false, R>);
          case Kernel::all_reduce_pull_two_shot_add_attn_res_rms_norm:
            return prefix ? run(all_reduce_pull_two_shot_add_attn_res_rms_norm<T, NG, true, R>)
                          : run(all_reduce_pull_two_shot_add_attn_res_rms_norm<T, NG, false, R>);
          case Kernel::all_reduce_push_two_shot_add_attn_res_rms_norm:
            return prefix ? run(all_reduce_push_two_shot_add_attn_res_rms_norm<T, NG, true, R>)
                          : run(all_reduce_push_two_shot_add_attn_res_rms_norm<T, NG, false, R>);
          default: impl::not_this_ops(k.kernel);
        }
      });
    });
  });
}

inline void launch(Handle& h, const KernelSpec& k, const GemmTailArgs& a, hipStream_t s) {
  const p2p::DevComm p = h.dev_comm(a.inp, a.rows * a.hidden * elem_bytes(a.dtype), s);
  const int rows       = static_cast<int>(a.rows);
  const int packs      = static_cast<int>(a.hidden * elem_bytes(a.dtype) / kPackBytes);
  constexpr int L      = kBuild.gemm_lanes;
  impl::by_world(h.world_size(), [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    impl::by_dtype(a.dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      impl::at_row_packs<Kernel::all_reduce_pull_one_shot_rms_norm_gemm_add>(
          impl::row_packs(k, a.hidden, a.dtype), [&](auto r) {
        constexpr int R = decltype(r)::value;
        const auto args = std::make_tuple(
            p, static_cast<const T*>(a.norm_weight), a.eps, static_cast<const T*>(a.gemm_weight),
            static_cast<int>(a.n_cols), static_cast<T*>(a.out), a.out_stride, a.out_col0,
            static_cast<T*>(a.workspace), rows, packs);
        const auto run = [&](auto kernel) {
          std::apply([&](auto&&... xs) { impl::start(kernel, k, s, xs...); }, args);
        };
        switch (k.kernel) {
          case Kernel::all_reduce_pull_one_shot_rms_norm_gemm_add:
            return run(all_reduce_pull_one_shot_rms_norm_gemm_add<T, NG, L, R>);
          case Kernel::all_reduce_pull_two_shot_rms_norm_gemm_add:
            return run(all_reduce_pull_two_shot_rms_norm_gemm_add<T, NG, L, R>);
          case Kernel::all_reduce_pull_one_shot_rms_norm_gemm:
            return run(all_reduce_pull_one_shot_rms_norm_gemm<T, NG, L, R>);
          case Kernel::all_reduce_pull_two_shot_rms_norm_gemm:
            return run(all_reduce_pull_two_shot_rms_norm_gemm<T, NG, L, R>);
          default: impl::not_this_ops(k.kernel);
        }
      });
    });
  });
}

}  // namespace hip_comms
