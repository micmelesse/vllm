// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// DISPATCH, NO DECISIONS: a Kernel (select's: the template and its arguments) to its compiled
// function, as torch's dispatcher takes an op to its kernel. The template's arguments are
// constexpr, so each is a template parameter, resolved here from the Kernel alone; every
// combination below is compiled, and one that was not raises. dispatch(kernel, args, f) hands f
// the compiled function and `bind`, which makes its arguments from the call's pointers and the
// peers' view: validate reads what the function uses, launch runs it.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <hip/hip_runtime.h>

#include <map>
#include <optional>
#include <variant>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../kernels/all_reduce_pull_one_shot.cuh"
#include "../kernels/all_reduce_pull_one_shot_add_attn_res_rms_norm.cuh"
#include "../kernels/all_reduce_pull_one_shot_add_rms_norm.cuh"
#include "../kernels/all_reduce_pull_one_shot_rms_norm_gemm_add.cuh"
#include "../kernels/all_reduce_pull_one_shot_rms_scale_add.cuh"
#include "../kernels/all_reduce_pull_one_shot_staged.cuh"
#include "../kernels/all_reduce_pull_two_shot.cuh"
#include "../kernels/all_reduce_pull_two_shot_add_attn_res_rms_norm.cuh"
#include "../kernels/all_reduce_pull_two_shot_add_rms_norm.cuh"
#include "../kernels/all_reduce_pull_two_shot_rms_norm_gemm_add.cuh"
#include "../kernels/all_reduce_pull_two_shot_staged.cuh"
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
template <Template K, typename F>
void at_row_packs(int k, F&& f) {
  by_row_packs<max_row_packs(K)>(k, std::forward<F>(f));
}

[[noreturn]] inline void not_this_ops(Template k) {
  throw std::runtime_error("hip_comms: template " + std::to_string(static_cast<int>(k)) +
                           " is not this op's");
}

// A row template's build, which select left empty when none holds the row (validate refuses it).
inline int row_build(std::optional<int> row_packs) {
  if (!row_packs) not_built("a row build that holds the row");
  return *row_packs;
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
void dispatch(const Kernel& k, const AllReduceArgs& a, F&& f) {
  const auto& args = std::get<AllReduceTemplateArgs>(k.args);
  const int n      = static_cast<int>(a.bytes / kPackBytes);
  impl::by_world(args.world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    impl::by_dtype(args.dtype, [&](auto t) {
      using T         = typename decltype(t)::t;
      const auto bind = [&](const p2p::DevComm& p) {
        return std::make_tuple(p, static_cast<T*>(a.out), n);
      };
      // A staged build takes its own input and the staging's size, and counts in 64 bits.
      const auto staged_bind = [&](const p2p::DevComm& p) {
        return std::make_tuple(p, static_cast<T*>(a.out), static_cast<const T*>(a.inp),
                               int64_t{a.bytes / kPackBytes}, kStagingBytes / kPackBytes);
      };
      switch (k.fn) {
        case Template::all_reduce_pull_one_shot:
          return f(all_reduce_pull_one_shot<T, NG>, bind);
        case Template::all_reduce_pull_two_shot:
          return f(all_reduce_pull_two_shot<T, NG>, bind);
        case Template::all_reduce_pull_one_shot_staged:
          return f(all_reduce_pull_one_shot_staged<T, NG>, staged_bind);
        case Template::all_reduce_pull_two_shot_staged:
          return f(all_reduce_pull_two_shot_staged<T, NG>, staged_bind);
        default: impl::not_this_ops(k.fn);
      }
    });
  });
}

template <typename F>
void dispatch(const Kernel& k, const NormArgs& a, F&& f) {
  const auto& args = std::get<NormTemplateArgs>(k.args);
  const int rows   = static_cast<int>(a.rows);
  const int packs  = static_cast<int>(packs_of(a));
  const int r      = impl::row_build(args.row_packs);
  impl::by_world(args.world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    impl::by_dtype(args.dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      impl::by_weight<T>(args.weight, args.dtype, [&](auto w) {
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
        using K = Template;
        switch (k.fn) {
          case K::all_reduce_pull_one_shot_rms_norm:
            return impl::at_row_packs<K::all_reduce_pull_one_shot_rms_norm>(
                r, [&](auto rp) {
                  constexpr int R = decltype(rp)::value;
                  f(all_reduce_pull_one_shot_rms_norm<T, W, NG, R>, norm);
                });
          case K::all_reduce_pull_two_shot_rms_norm:
            return impl::at_row_packs<K::all_reduce_pull_two_shot_rms_norm>(
                r, [&](auto rp) {
                  constexpr int R = decltype(rp)::value;
                  f(all_reduce_pull_two_shot_rms_norm<T, W, NG, R>, norm);
                });
          case K::all_reduce_push_two_shot_rms_norm:
            return impl::at_row_packs<K::all_reduce_push_two_shot_rms_norm>(
                r, [&](auto rp) {
                  constexpr int R = decltype(rp)::value;
                  f(all_reduce_push_two_shot_rms_norm<T, W, NG, R>, norm);
                });
          case K::all_reduce_pull_one_shot_add_rms_norm:
            return impl::at_row_packs<K::all_reduce_pull_one_shot_add_rms_norm>(
                r, [&](auto rp) {
                  constexpr int R = decltype(rp)::value;
                  f(all_reduce_pull_one_shot_add_rms_norm<T, W, NG, R>, add_norm);
                });
          case K::all_reduce_pull_two_shot_add_rms_norm:
            return impl::at_row_packs<K::all_reduce_pull_two_shot_add_rms_norm>(
                r, [&](auto rp) {
                  constexpr int R = decltype(rp)::value;
                  f(all_reduce_pull_two_shot_add_rms_norm<T, W, NG, R>, add_norm);
                });
          case K::all_reduce_push_two_shot_add_rms_norm:
            return impl::at_row_packs<K::all_reduce_push_two_shot_add_rms_norm>(
                r, [&](auto rp) {
                  constexpr int R = decltype(rp)::value;
                  f(all_reduce_push_two_shot_add_rms_norm<T, W, NG, R>, add_norm);
                });
          default: impl::not_this_ops(k.fn);
        }
      });
    });
  });
}

// AttnRes's and the GEMM tail's kernels share their row builds, so one cap serves each op's.
static_assert(max_row_packs(Template::all_reduce_pull_one_shot_add_attn_res_rms_norm) ==
                  max_row_packs(Template::all_reduce_pull_two_shot_add_attn_res_rms_norm) &&
              max_row_packs(Template::all_reduce_pull_one_shot_add_attn_res_rms_norm) ==
                  max_row_packs(Template::all_reduce_push_two_shot_add_attn_res_rms_norm) &&
              max_row_packs(Template::all_reduce_pull_one_shot_rms_norm_gemm_add) ==
                  max_row_packs(Template::all_reduce_pull_two_shot_rms_norm_gemm_add) &&
              max_row_packs(Template::all_reduce_pull_one_shot_rms_norm_gemm_add) ==
                  max_row_packs(Template::all_reduce_pull_one_shot_rms_norm_gemm) &&
              max_row_packs(Template::all_reduce_pull_one_shot_rms_norm_gemm_add) ==
                  max_row_packs(Template::all_reduce_pull_two_shot_rms_norm_gemm),
              "a kernel with its own row builds needs its own case");

template <typename F>
void dispatch(const Kernel& k, const AttnResArgs& a, F&& f) {
  const auto& args = std::get<AttnResTemplateArgs>(k.args);
  const int rows   = static_cast<int>(a.rows);
  const int packs  = static_cast<int>(packs_of(a));
  impl::by_world(args.world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    impl::by_dtype(args.dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      impl::at_row_packs<Template::all_reduce_pull_one_shot_add_attn_res_rms_norm>(
          impl::row_build(args.row_packs), [&](auto r) {
            constexpr int R = decltype(r)::value;
            const auto bind = [&](const p2p::DevComm& p) {
              return std::make_tuple(
                  p, static_cast<T*>(a.prefix), static_cast<T*>(a.blocks), a.block_stride_m,
                  a.block_stride_r, static_cast<const T*>(a.norm_weight),
                  static_cast<const T*>(a.qk_weight), static_cast<const T*>(a.out_norm_weight),
                  static_cast<T*>(a.out), a.num_blocks, a.write_idx, a.eps, a.out_eps, rows,
                  packs);
            };
            const bool prefix = args.prefix;
            switch (k.fn) {
              case Template::all_reduce_pull_one_shot_add_attn_res_rms_norm:
                return prefix
                           ? f(all_reduce_pull_one_shot_add_attn_res_rms_norm<T, NG, true, R>, bind)
                           : f(all_reduce_pull_one_shot_add_attn_res_rms_norm<T, NG, false, R>,
                               bind);
              case Template::all_reduce_pull_two_shot_add_attn_res_rms_norm:
                return prefix
                           ? f(all_reduce_pull_two_shot_add_attn_res_rms_norm<T, NG, true, R>, bind)
                           : f(all_reduce_pull_two_shot_add_attn_res_rms_norm<T, NG, false, R>,
                               bind);
              case Template::all_reduce_push_two_shot_add_attn_res_rms_norm:
                return prefix
                           ? f(all_reduce_push_two_shot_add_attn_res_rms_norm<T, NG, true, R>, bind)
                           : f(all_reduce_push_two_shot_add_attn_res_rms_norm<T, NG, false, R>,
                               bind);
              default: impl::not_this_ops(k.fn);
            }
          });
    });
  });
}

template <typename F>
void dispatch(const Kernel& k, const GemmTailArgs& a, F&& f) {
  const auto& args = std::get<GemmTemplateArgs>(k.args);
  const int rows   = static_cast<int>(a.rows);
  const int packs  = static_cast<int>(packs_of(a));
  constexpr int L  = kBuild.gemm_lanes;  // one build of it
  if (args.lanes != L) impl::not_built("the GEMM tail at " + std::to_string(args.lanes) + " lanes");
  impl::by_world(args.world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    impl::by_dtype(args.dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      impl::at_row_packs<Template::all_reduce_pull_one_shot_rms_norm_gemm_add>(
          impl::row_build(args.row_packs), [&](auto r) {
            constexpr int R = decltype(r)::value;
            const auto bind = [&](const p2p::DevComm& p) {
              return std::make_tuple(
                  p, static_cast<const T*>(a.norm_weight), a.eps,
                  static_cast<const T*>(a.gemm_weight), static_cast<int>(a.n_cols),
                  static_cast<T*>(a.out), a.out_stride, a.out_col0, static_cast<T*>(a.workspace),
                  rows, packs);
            };
            switch (k.fn) {
              case Template::all_reduce_pull_one_shot_rms_norm_gemm_add:
                return f(all_reduce_pull_one_shot_rms_norm_gemm_add<T, NG, L, R>, bind);
              case Template::all_reduce_pull_two_shot_rms_norm_gemm_add:
                return f(all_reduce_pull_two_shot_rms_norm_gemm_add<T, NG, L, R>, bind);
              case Template::all_reduce_pull_one_shot_rms_norm_gemm:
                return f(all_reduce_pull_one_shot_rms_norm_gemm<T, NG, L, R>, bind);
              case Template::all_reduce_pull_two_shot_rms_norm_gemm:
                return f(all_reduce_pull_two_shot_rms_norm_gemm<T, NG, L, R>, bind);
              default: impl::not_this_ops(k.fn);
            }
          });
    });
  });
}

template <typename F>
void dispatch(const Kernel& k, const ScaleAddArgs& a, F&& f) {
  const auto& args = std::get<ScaleAddTemplateArgs>(k.args);
  const int e      = elem_bytes(a.dtype);
  const int rows   = static_cast<int>(a.rows);
  const int hp     = static_cast<int>(a.hidden * e / kPackBytes);
  const int lp     = static_cast<int>(a.latent * e / kPackBytes);
  impl::by_world(args.world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    impl::by_dtype(args.dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      constexpr Template K = Template::all_reduce_pull_one_shot_rms_scale_add;
      impl::at_row_packs<K>(impl::row_build(args.row_packs), [&](auto r) {
        constexpr int R = decltype(r)::value;
        const auto bind = [&](const p2p::DevComm& p) {
          return std::make_tuple(p, static_cast<T*>(a.out), a.eps, rows, hp, lp, args.splits);
        };
        switch (k.fn) {
          case K: return f(all_reduce_pull_one_shot_rms_scale_add<T, NG, R>, bind);
          default: impl::not_this_ops(k.fn);
        }
      });
    });
  });
}

}  // namespace hip_comms
