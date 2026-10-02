// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// CHECK, THE ONLY NO: `check(handle, kernel, args, options)` is the first Error kernel `k` meets on
// call `a` here, or none; `plan(handle, args, options)` is select's kernel or that Error. Every
// Error is a capability (a kernel that cannot take the input), never "the unfused ops would be
// faster": a tune_<op> never declines.

#pragma once

#ifndef HIP_COMMS_INTERFACE
#error "include rocm_comms.cuh, the one interface, not its parts"
#endif

#include <cstdint>
#include <optional>
#include <type_traits>
#include <variant>

namespace hip_comms {

// A KERNEL'S SCRATCH, row-major (a staged build needs none: it runs in passes of what the
// scratch holds): the plain two-shot's slice of packs; a column two-shot the whole
// reduced tensor (each rank's columns at their place); a row two-shot its rank's rows, twice where
// it leaves two results (out and the residual). A one-shot reads the inputs and keeps nothing.
inline int64_t scratch_need(Template t, int64_t rows, int64_t packs, int world) {
  if (!is_two_shot(t)) return 0;
  const Op op = op_of(t);
  if (op == Op::all_reduce) return (rows * packs + world - 1) / world * kBuild.memory.pack_bytes;
  if (slices_columns(t)) return rows * packs * kBuild.memory.pack_bytes;
  const bool two = op == Op::all_reduce_add_rms_norm;
  return (rows + world - 1) / world * packs * (two ? 2 : 1) * kBuild.memory.pack_bytes;
}

// Whether kernel `k` holds its whole grid resident: only the compiled kernel knows what it uses.
template <typename Args>
bool resident(const Handle& h, const Kernel& k, const Args& a) {
  bool fits = true;
  dispatch(k, a, [&](auto kernel, const auto&) {
    const Resources used = h.resources_of(reinterpret_cast<const void*>(kernel));
    fits = k.config.blocks_per_grid <= resident_blocks(kTarget, used, k.config.threads_per_block);
  });
  return fits;
}

// The first Error kernel `k` meets on call `a` here, in this order, or none.
template <typename Args>
std::optional<Error> check(const Handle& h, const Kernel& k, const Args& a, const Options& o) {
  const int world = h.world_size();
  const int e     = elem_bytes(a.dtype);
  if (!world_built(world)) return Error::world_not_built;
  if (!dtype_built(a.dtype)) return Error::dtype_not_built;
  if constexpr (std::is_same_v<Args, NormArgs>)
    if (a.weight_dtype != a.dtype && a.weight_dtype != DType::f32) return Error::weight_not_built;
  if (hidden_of(a) * e % kBuild.memory.pack_bytes != 0) return Error::row_not_packs;
  if constexpr (std::is_same_v<Args, ScaleAddArgs>)
    if (a.hidden * e % kBuild.memory.pack_bytes != 0 ||
        a.latent * e % kBuild.memory.pack_bytes != 0 || a.latent < 1)
      return Error::widths_not_packs;
  if (op_of(k.fn) != op_of(a)) return Error::template_not_this_ops;
  if (has_tiles(k.fn) && !built_at(k.fn, k.config.threads_per_block))
    return Error::threads_not_built;
  if (has_tiles(k.fn) && k.config.tile_n == 0) return Error::row_too_wide;
  if (has_tiles(k.fn) && !built(k.fn, k.config)) return Error::tile_not_built;
  if (has_tiles(k.fn) && k.config.tile_n < tile_cols(a)) return Error::row_too_wide;
  // TWO-SHOT'S BLOCK IS ONE WAVE PER PEER, so anything else would leave a peer unread.
  if (k.fn == Template::all_reduce_pull_two_shot &&
      k.config.threads_per_block % (world * kWaveSize) != 0)
    return Error::block_not_a_wave_per_peer;
  if (o.quant_bits) return Error::quantized_not_built;
  if (gemms(op_of(a)) && k.config.threads_per_block > gemm_max_threads(kBuild.kernels.gemm_lanes))
    return Error::block_exceeds_lds;
  if (!is_staged(k) && scratch_need(k.fn, rows_of(a), packs_of(a), world) > h.scratch_bytes())
    return Error::scratch_too_small;
  // AN IN-PLACE BUILD ON AN EAGER INPUT reads it through the staging, copied in whole first.
  if (!is_staged(k) && !h.reads_in_place(a.inp, o.stream) && bytes_of(a) > h.staging_bytes())
    return Error::staging_too_small;
  if (!resident(h, k, a)) return Error::grid_not_resident;
  return std::nullopt;
}

// THE ONE DECISION: select's kernel for call `a`, or the Error it meets.
template <typename Args>
std::variant<Kernel, Error> plan(const Handle& h, const Args& a, const Options& o) {
  Kernel k = select(a, h.world_size(), o);
  // THE BUILD, from what the handle knows (a forced template too): in place when the peers can
  // read the input where it is, otherwise its staged build, where there is one.
  if (auto* args = std::get_if<AllReduceTemplateArgs>(&k.args))
    args->staged = !h.reads_in_place(a.inp, o.stream);
  if (const std::optional<Error> err = check(h, k, a, o)) return *err;
  return k;
}

}  // namespace hip_comms
