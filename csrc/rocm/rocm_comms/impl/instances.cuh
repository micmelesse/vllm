// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// INSTANCES, NO DECISIONS: a spec and an op's args to the one compiled kernel they name. What a
// kernel is compiled for is constexpr, so it is a template parameter, resolved here from the call:
// the world from the handle, T (and a norm's weight type W) from the args, a row kernel's packs a
// thread from the spec, the GEMM tail's lanes from the build. Every combination below is compiled;
// one that was not raises. with_kernel(h, spec, args, f) hands f the kernel and `bind`, which
// makes the kernel's arguments from the peers' view: validate reads what the kernel uses, launch
// runs it.

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

}  // namespace impl

// What a compiled kernel uses, from its code object; read once a kernel.
template <typename... P>
Resources resources_of(void (*kernel)(P...)) {
  thread_local std::map<const void*, Resources> known;
  const void* at = reinterpret_cast<const void*>(kernel);
  if (const auto it = known.find(at); it != known.end()) return it->second;
  hipFuncAttributes attrs{};
  HIP_CHECK(hipFuncGetAttributes(&attrs, at));
  return known[at] = {attrs.numRegs, static_cast<int64_t>(attrs.sharedSizeBytes)};
}

template <typename F>
void with_kernel(const Handle& h, const KernelSpec& k, const AllReduceArgs& a, F&& f) {
  const int n = static_cast<int>(a.bytes / kPackBytes);
  impl::by_world(h.world_size(), [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    impl::by_dtype(a.dtype, [&](auto t) {
      using T         = typename decltype(t)::t;
      const auto bind = [&](const p2p::DevComm& p) {
        return std::make_tuple(p, static_cast<T*>(a.out), n);
      };
      switch (k.kernel) {
        case Kernel::all_reduce_pull_one_shot:
          return f(all_reduce_pull_one_shot<T, NG>, bind);
        case Kernel::all_reduce_pull_two_shot:
          return f(all_reduce_pull_two_shot<T, NG>, bind);
        default: impl::not_this_ops(k.kernel);
      }
    });
  });
}

template <typename F>
void with_kernel(const Handle& h, const KernelSpec& k, const NormArgs& a, F&& f) {
  const int rows  = static_cast<int>(a.rows);
  const int packs = static_cast<int>(a.hidden * elem_bytes(a.dtype) / kPackBytes);
  impl::by_world(h.world_size(), [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    impl::by_dtype(a.dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      impl::by_weight<T>(a.weight_dtype, a.dtype, [&](auto w) {
        using W         = typename decltype(w)::t;
        const auto norm = [&](const p2p::DevComm& p) {
          return std::make_tuple(p, static_cast<T*>(a.out), static_cast<const W*>(a.weight), a.eps,
                                 rows, packs);
        };
        const auto add_norm = [&](const p2p::DevComm& p) {
          return std::make_tuple(p, static_cast<T*>(a.out), static_cast<T*>(a.residual_out),
                                 static_cast<const T*>(a.residual),
                                 static_cast<const W*>(a.weight), a.eps, rows, packs);
        };
        using K = Kernel;
        switch (k.kernel) {
          case K::all_reduce_pull_one_shot_rms_norm:
            return impl::at_row_packs<K::all_reduce_pull_one_shot_rms_norm>(
                k.row_packs, [&](auto rp) {
                  constexpr int R = decltype(rp)::value;
                  f(all_reduce_pull_one_shot_rms_norm<T, W, NG, R>, norm);
                });
          case K::all_reduce_pull_two_shot_rms_norm:
            return impl::at_row_packs<K::all_reduce_pull_two_shot_rms_norm>(
                k.row_packs, [&](auto rp) {
                  constexpr int R = decltype(rp)::value;
                  f(all_reduce_pull_two_shot_rms_norm<T, W, NG, R>, norm);
                });
          case K::all_reduce_push_two_shot_rms_norm:
            return impl::at_row_packs<K::all_reduce_push_two_shot_rms_norm>(
                k.row_packs, [&](auto rp) {
                  constexpr int R = decltype(rp)::value;
                  f(all_reduce_push_two_shot_rms_norm<T, W, NG, R>, norm);
                });
          case K::all_reduce_pull_one_shot_add_rms_norm:
            return impl::at_row_packs<K::all_reduce_pull_one_shot_add_rms_norm>(
                k.row_packs, [&](auto rp) {
                  constexpr int R = decltype(rp)::value;
                  f(all_reduce_pull_one_shot_add_rms_norm<T, W, NG, R>, add_norm);
                });
          case K::all_reduce_pull_two_shot_add_rms_norm:
            return impl::at_row_packs<K::all_reduce_pull_two_shot_add_rms_norm>(
                k.row_packs, [&](auto rp) {
                  constexpr int R = decltype(rp)::value;
                  f(all_reduce_pull_two_shot_add_rms_norm<T, W, NG, R>, add_norm);
                });
          case K::all_reduce_push_two_shot_add_rms_norm:
            return impl::at_row_packs<K::all_reduce_push_two_shot_add_rms_norm>(
                k.row_packs, [&](auto rp) {
                  constexpr int R = decltype(rp)::value;
                  f(all_reduce_push_two_shot_add_rms_norm<T, W, NG, R>, add_norm);
                });
          default: impl::not_this_ops(k.kernel);
        }
      });
    });
  });
}

// AttnRes's and the GEMM tail's kernels share their row builds, so one cap serves each op's.
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

template <typename F>
void with_kernel(const Handle& h, const KernelSpec& k, const AttnResArgs& a, F&& f) {
  const int rows  = static_cast<int>(a.rows);
  const int packs = static_cast<int>(a.hidden * elem_bytes(a.dtype) / kPackBytes);
  impl::by_world(h.world_size(), [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    impl::by_dtype(a.dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      impl::at_row_packs<Kernel::all_reduce_pull_one_shot_add_attn_res_rms_norm>(
          k.row_packs, [&](auto r) {
            constexpr int R = decltype(r)::value;
            const auto bind = [&](const p2p::DevComm& p) {
              return std::make_tuple(
                  p, static_cast<T*>(a.prefix), static_cast<T*>(a.blocks), a.block_stride_m,
                  a.block_stride_r, static_cast<const T*>(a.norm_weight),
                  static_cast<const T*>(a.qk_weight), static_cast<const T*>(a.out_norm_weight),
                  static_cast<T*>(a.out), a.num_blocks, a.write_idx, a.eps, a.out_eps, rows,
                  packs);
            };
            const bool prefix = a.has_prefix;
            switch (k.kernel) {
              case Kernel::all_reduce_pull_one_shot_add_attn_res_rms_norm:
                return prefix
                           ? f(all_reduce_pull_one_shot_add_attn_res_rms_norm<T, NG, true, R>, bind)
                           : f(all_reduce_pull_one_shot_add_attn_res_rms_norm<T, NG, false, R>,
                               bind);
              case Kernel::all_reduce_pull_two_shot_add_attn_res_rms_norm:
                return prefix
                           ? f(all_reduce_pull_two_shot_add_attn_res_rms_norm<T, NG, true, R>, bind)
                           : f(all_reduce_pull_two_shot_add_attn_res_rms_norm<T, NG, false, R>,
                               bind);
              case Kernel::all_reduce_push_two_shot_add_attn_res_rms_norm:
                return prefix
                           ? f(all_reduce_push_two_shot_add_attn_res_rms_norm<T, NG, true, R>, bind)
                           : f(all_reduce_push_two_shot_add_attn_res_rms_norm<T, NG, false, R>,
                               bind);
              default: impl::not_this_ops(k.kernel);
            }
          });
    });
  });
}

template <typename F>
void with_kernel(const Handle& h, const KernelSpec& k, const GemmTailArgs& a, F&& f) {
  const int rows  = static_cast<int>(a.rows);
  const int packs = static_cast<int>(a.hidden * elem_bytes(a.dtype) / kPackBytes);
  constexpr int L = kBuild.gemm_lanes;
  impl::by_world(h.world_size(), [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    impl::by_dtype(a.dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      impl::at_row_packs<Kernel::all_reduce_pull_one_shot_rms_norm_gemm_add>(
          k.row_packs, [&](auto r) {
            constexpr int R = decltype(r)::value;
            const auto bind = [&](const p2p::DevComm& p) {
              return std::make_tuple(
                  p, static_cast<const T*>(a.norm_weight), a.eps,
                  static_cast<const T*>(a.gemm_weight), static_cast<int>(a.n_cols),
                  static_cast<T*>(a.out), a.out_stride, a.out_col0, static_cast<T*>(a.workspace),
                  rows, packs);
            };
            switch (k.kernel) {
              case Kernel::all_reduce_pull_one_shot_rms_norm_gemm_add:
                return f(all_reduce_pull_one_shot_rms_norm_gemm_add<T, NG, L, R>, bind);
              case Kernel::all_reduce_pull_two_shot_rms_norm_gemm_add:
                return f(all_reduce_pull_two_shot_rms_norm_gemm_add<T, NG, L, R>, bind);
              case Kernel::all_reduce_pull_one_shot_rms_norm_gemm:
                return f(all_reduce_pull_one_shot_rms_norm_gemm<T, NG, L, R>, bind);
              case Kernel::all_reduce_pull_two_shot_rms_norm_gemm:
                return f(all_reduce_pull_two_shot_rms_norm_gemm<T, NG, L, R>, bind);
              default: impl::not_this_ops(k.kernel);
            }
          });
    });
  });
}

}  // namespace hip_comms
