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

#include <optional>
#include <variant>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "kernels/all_reduce_pull_one_shot.cuh"
#include "kernels/all_reduce_pull_one_shot_add_attn_res_rms_norm.cuh"
#include "kernels/all_reduce_pull_one_shot_add_rms_norm.cuh"
#include "kernels/all_reduce_pull_one_shot_rms_norm_gemm_add.cuh"
#include "kernels/all_reduce_pull_one_shot_rms_scale_add.cuh"
#include "kernels/all_reduce_pull_two_shot.cuh"
#include "kernels/all_reduce_pull_two_shot_add_attn_res_rms_norm.cuh"
#include "kernels/all_reduce_pull_two_shot_add_rms_norm.cuh"
#include "kernels/all_reduce_pull_two_shot_rms_norm_gemm_add.cuh"
#include "kernels/all_reduce_pull_two_shot_rms_scale_add.cuh"
#include "kernels/all_reduce_push_two_shot_add_attn_res_rms_norm.cuh"
#include "kernels/all_reduce_push_two_shot_add_rms_norm.cuh"

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

// THE C++ TYPE OF A BUILT DTYPE.
template <DType D>
struct of_dtype;
template <>
struct of_dtype<DType::f16> {
  using t = c10::Half;
};
template <>
struct of_dtype<DType::bf16> {
  using t = c10::BFloat16;
};

// OVER THE BUILT LISTS (kBuild.supports), so what is compiled is what they say.
template <typename F, size_t... I>
void by_world_in(int world, F& f, std::index_sequence<I...>) {
  constexpr auto& built = kBuild.supports.worlds;
  if (!((world == built[I] && (f(constant<built[I]>{}), true)) || ...))
    not_built("world size " + std::to_string(world));
}

template <typename F>
void by_world(int world, F&& f) {
  by_world_in(world, f, std::make_index_sequence<kBuild.supports.worlds.size()>{});
}

template <typename F, size_t... I>
void by_dtype_in(DType d, F& f, std::index_sequence<I...>) {
  constexpr auto& built = kBuild.supports.dtypes;
  if (!((d == built[I] && (f(type<typename of_dtype<built[I]>::t>{}), true)) || ...))
    not_built("dtype");
}

template <typename F>
void by_dtype(DType d, F&& f) {
  by_dtype_in(d, f, std::make_index_sequence<kBuild.supports.dtypes.size()>{});
}

// A norm's weight: T itself, or fp32.
template <typename T, typename F>
void by_weight(DType weight, DType dtype, F&& f) {
  if (weight == dtype) return f(type<T>{});
  if (weight == DType::f32) return f(type<float>{});
  not_built("weight dtype");
}

// A TEMPLATE'S BUILD, compiled in: the one of template K's configs (op.cuh) `c` names (all but
// its launch's grid and reduce_scatter_blocks, which are run time), and only those. f is handed
// it as config_constant<C>, C its family's config, every field a constant expression.
template <Template K>
using family_t = std::variant_alternative_t<family_of(K), KernelConfig>;
template <Template K, size_t I>
constexpr family_t<K> built_config = std::get<family_t<K>>(configs_of(K)[I]);
template <auto C>
struct config_constant {
  static constexpr auto value = C;
};

template <Template K, typename F, size_t... I>
void by_config_in(const KernelConfig& c, F& f, std::index_sequence<I...>) {
  if (!((same_build(c, configs_of(K)[I]) && (f(config_constant<built_config<K, I>>{}), true)) ||
        ...))
    not_built(std::string("template ") + to_string(K) + " at that tile and threads");
}

template <Template K, typename F>
void by_config(const KernelConfig& c, F&& f) {
  by_config_in<K>(c, f, std::make_index_sequence<configs_of(K).size()>{});
}

[[noreturn]] inline void not_this_ops(Template k) {
  throw std::runtime_error("hip_comms: template " + std::to_string(static_cast<int>(k)) +
                           " is not this op's");
}

}  // namespace impl

template <typename F>
void dispatch(const Kernel& k, const AllReduceArgs& a, F&& f) {
  const auto& args = std::get<AllReduceTemplateArgs>(k.args);
  const int n = static_cast<int>(a.bytes / kBuild.memory.pack_bytes);
  impl::by_world(args.world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    impl::by_dtype(args.dtype, [&](auto t) {
      using T         = typename decltype(t)::t;
      // IN PLACE: the input's packs, its int. STAGED: in 64 bits, with its own input and the packs
      // a staging holds.
      const auto in_place = [&](const p2p::DevComm& p) {
        return std::make_tuple(p, static_cast<T*>(a.out), n, Staged<T, false>{});
      };
      const auto staged = [&](const p2p::DevComm& p) {
        return std::make_tuple(
            p, static_cast<T*>(a.out), int64_t{a.bytes / kBuild.memory.pack_bytes},
            Staged<T, true>{static_cast<const T*>(a.inp),
                            kBuild.memory.staging_bytes / kBuild.memory.pack_bytes});
      };
      switch (k.fn) {
        case Template::all_reduce_pull_one_shot:
          return args.staged ? f(all_reduce_pull_one_shot<T, NG, true>, staged)
                             : f(all_reduce_pull_one_shot<T, NG, false>, in_place);
        case Template::all_reduce_pull_two_shot:
          return args.staged ? f(all_reduce_pull_two_shot<T, NG, true>, staged)
                             : f(all_reduce_pull_two_shot<T, NG, false>, in_place);
        default: impl::not_this_ops(k.fn);
      }
    });
  });
}

template <typename F>
void dispatch(const Kernel& k, const NormArgs& a, F&& f) {
  const auto& args = std::get<NormTemplateArgs>(k.args);
  const int rows   = static_cast<int>(a.rows);
  const int packs = static_cast<int>(packs_of(a));
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
            return impl::by_config<K::all_reduce_pull_one_shot_rms_norm>(k.config, [&](auto c) {
                  constexpr RowConfig C = decltype(c)::value;
                  f(all_reduce_pull_one_shot_rms_norm<T, W, NG, C.tile_n,
                                                      C.launch.threads_per_block>,
                    norm);
                });
          case K::all_reduce_pull_two_shot_rms_norm:
            return impl::by_config<K::all_reduce_pull_two_shot_rms_norm>(k.config, [&](auto c) {
                  constexpr RowConfig C = decltype(c)::value;
                  f(all_reduce_pull_two_shot_rms_norm<T, W, NG, C.tile_n,
                                                      C.launch.threads_per_block>,
                    norm);
                });
          case K::all_reduce_push_two_shot_rms_norm:
            return impl::by_config<K::all_reduce_push_two_shot_rms_norm>(k.config, [&](auto c) {
                  constexpr RowConfig C = decltype(c)::value;
                  f(all_reduce_push_two_shot_rms_norm<T, W, NG, C.tile_n,
                                                      C.launch.threads_per_block>,
                    norm);
                });
          case K::all_reduce_pull_one_shot_add_rms_norm:
            return impl::by_config<K::all_reduce_pull_one_shot_add_rms_norm>(k.config, [&](auto c) {
                  constexpr RowConfig C = decltype(c)::value;
                  f(all_reduce_pull_one_shot_add_rms_norm<T, W, NG, C.tile_n,
                                                          C.launch.threads_per_block>,
                    add_norm);
                });
          case K::all_reduce_pull_two_shot_add_rms_norm:
            return impl::by_config<K::all_reduce_pull_two_shot_add_rms_norm>(k.config, [&](auto c) {
                  constexpr RowConfig C = decltype(c)::value;
                  f(all_reduce_pull_two_shot_add_rms_norm<T, W, NG, C.tile_n,
                                                          C.launch.threads_per_block>,
                    add_norm);
                });
          case K::all_reduce_push_two_shot_add_rms_norm:
            return impl::by_config<K::all_reduce_push_two_shot_add_rms_norm>(k.config, [&](auto c) {
                  constexpr RowConfig C = decltype(c)::value;
                  f(all_reduce_push_two_shot_add_rms_norm<T, W, NG, C.tile_n,
                                                          C.launch.threads_per_block>,
                    add_norm);
                });
          default: impl::not_this_ops(k.fn);
        }
      });
    });
  });
}

template <typename F>
void dispatch(const Kernel& k, const AttnResArgs& a, F&& f) {
  const auto& args = std::get<AttnResTemplateArgs>(k.args);
  const int rows   = static_cast<int>(a.rows);
  const int packs  = static_cast<int>(packs_of(a));
  impl::by_world(args.world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    impl::by_dtype(args.dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      const auto bind = [&](const p2p::DevComm& p) {
        return std::make_tuple(
            p, static_cast<T*>(a.prefix), static_cast<T*>(a.blocks), a.block_stride_m,
            a.block_stride_r, static_cast<const T*>(a.norm_weight),
            static_cast<const T*>(a.qk_weight), static_cast<const T*>(a.out_norm_weight),
            static_cast<T*>(a.out), a.num_blocks, a.write_idx, a.eps, a.out_eps, rows, packs);
      };
      // The pull's reduce-scatter blocks, a launch's choice: run time, after the rest.
      const auto bind_pull = [&](const p2p::DevComm& p) {
        return std::tuple_cat(bind(p), std::make_tuple(std::get<AttnResPullConfig>(k.config)
                                                           .reduce_scatter_blocks));
      };
      const bool prefix = args.prefix;
      using K = Template;
      switch (k.fn) {
        case K::all_reduce_pull_one_shot_add_attn_res_rms_norm:
          return impl::by_config<K::all_reduce_pull_one_shot_add_attn_res_rms_norm>(
              k.config, [&](auto c) {
                constexpr AttnResConfig C = decltype(c)::value;
                constexpr int TN = C.tile_n, TK = C.tile_k, TPB = C.launch.threads_per_block;
                prefix
                    ? f(all_reduce_pull_one_shot_add_attn_res_rms_norm<T, NG, true, TN, TK, TPB>,
                        bind)
                    : f(all_reduce_pull_one_shot_add_attn_res_rms_norm<T, NG, false, TN, TK, TPB>,
                        bind);
              });
        case K::all_reduce_pull_two_shot_add_attn_res_rms_norm:
          return impl::by_config<K::all_reduce_pull_two_shot_add_attn_res_rms_norm>(
              k.config, [&](auto c) {
                constexpr AttnResPullConfig C = decltype(c)::value;
                constexpr int TM = C.tile_m, TN = C.tile_n, TK = C.tile_k,
                              TPB = C.launch.threads_per_block;
                prefix ? f(all_reduce_pull_two_shot_add_attn_res_rms_norm<T, NG, true, TM, TN, TK,
                                                                           TPB>,
                           bind_pull)
                       : f(all_reduce_pull_two_shot_add_attn_res_rms_norm<T, NG, false, TM, TN, TK,
                                                                           TPB>,
                           bind_pull);
              });
        case K::all_reduce_push_two_shot_add_attn_res_rms_norm:
          return impl::by_config<K::all_reduce_push_two_shot_add_attn_res_rms_norm>(
              k.config, [&](auto c) {
                constexpr AttnResConfig C = decltype(c)::value;
                constexpr int TN = C.tile_n, TK = C.tile_k, TPB = C.launch.threads_per_block;
                prefix
                    ? f(all_reduce_push_two_shot_add_attn_res_rms_norm<T, NG, true, TN, TK, TPB>,
                        bind)
                    : f(all_reduce_push_two_shot_add_attn_res_rms_norm<T, NG, false, TN, TK, TPB>,
                        bind);
              });
        default:
          impl::not_this_ops(k.fn);
      }
    });
  });
}

template <typename F>
void dispatch(const Kernel& k, const GemmTailArgs& a, F&& f) {
  const auto& args = std::get<GemmTemplateArgs>(k.args);
  const int rows   = static_cast<int>(a.rows);
  const int packs = static_cast<int>(packs_of(a));
  impl::by_world(args.world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    impl::by_dtype(args.dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      // The four GEMM-tail templates share one list, so one lookup serves each.
      static_assert(configs_of(Template::all_reduce_pull_one_shot_rms_norm_gemm_add).data() ==
                        configs_of(Template::all_reduce_pull_two_shot_rms_norm_gemm_add).data() &&
                    configs_of(Template::all_reduce_pull_one_shot_rms_norm_gemm_add).data() ==
                        configs_of(Template::all_reduce_pull_one_shot_rms_norm_gemm).data() &&
                    configs_of(Template::all_reduce_pull_one_shot_rms_norm_gemm_add).data() ==
                        configs_of(Template::all_reduce_pull_two_shot_rms_norm_gemm).data());
      impl::by_config<Template::all_reduce_pull_one_shot_rms_norm_gemm_add>(
          k.config, [&](auto c) {
            constexpr GemmConfig C = decltype(c)::value;
            constexpr int TM = C.tile_m, TN = C.tile_n, TK = C.tile_k, SK = C.slice_k,
                          TPB = C.launch.threads_per_block;
            const auto bind = [&](const p2p::DevComm& p) {
              return std::make_tuple(
                  p, static_cast<const T*>(a.norm_weight), a.eps,
                  static_cast<const T*>(a.gemm_weight), static_cast<int>(a.n_cols),
                  static_cast<T*>(a.out), a.out_stride, static_cast<T*>(a.workspace),
                  rows, packs);
            };
            switch (k.fn) {
              case Template::all_reduce_pull_one_shot_rms_norm_gemm_add:
                return f(all_reduce_pull_one_shot_rms_norm_gemm_add<T, NG, TM, TN, TK, SK, TPB>,
                         bind);
              case Template::all_reduce_pull_two_shot_rms_norm_gemm_add:
                return f(all_reduce_pull_two_shot_rms_norm_gemm_add<T, NG, TM, TN, TK, SK, TPB>,
                         bind);
              case Template::all_reduce_pull_one_shot_rms_norm_gemm:
                return f(all_reduce_pull_one_shot_rms_norm_gemm<T, NG, TM, TN, TK, SK, TPB>, bind);
              case Template::all_reduce_pull_two_shot_rms_norm_gemm:
                return f(all_reduce_pull_two_shot_rms_norm_gemm<T, NG, TM, TN, TK, SK, TPB>, bind);
              default:
                impl::not_this_ops(k.fn);
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
  const int hp = static_cast<int>(a.hidden * e / kBuild.memory.pack_bytes);
  const int lp = static_cast<int>(a.latent * e / kBuild.memory.pack_bytes);
  impl::by_world(args.world, [&](auto ng) {
    constexpr int NG = decltype(ng)::value;
    impl::by_dtype(args.dtype, [&](auto t) {
      using T = typename decltype(t)::t;
      // The one-shot's and the two-shot's builds are one list.
      constexpr Template K = Template::all_reduce_pull_one_shot_rms_scale_add;
      static_assert(same_builds(K, Template::all_reduce_pull_two_shot_rms_scale_add));
      impl::by_config<K>(k.config, [&](auto c) {
        constexpr RowConfig C = decltype(c)::value;
        constexpr int BN = C.tile_n, NT = C.launch.threads_per_block;
        const auto bind = [&](const p2p::DevComm& p) {
          return std::make_tuple(p, static_cast<T*>(a.out), a.eps, rows, hp, lp);
        };
        switch (k.fn) {
          case K:
            return f(all_reduce_pull_one_shot_rms_scale_add<T, NG, BN, NT>, bind);
          case Template::all_reduce_pull_two_shot_rms_scale_add:
            return f(all_reduce_pull_two_shot_rms_scale_add<T, NG, BN, NT>, bind);
          default:
            impl::not_this_ops(k.fn);
        }
      });
    });
  });
}

}  // namespace hip_comms
