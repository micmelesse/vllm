// SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc.
// All rights reserved.
//
// CLOSE, a torch op: the communicator's state released.
#pragma once

#include "../../ops.h"
#include "../types.cuh"
#include "../utils.h"

// CLOSE: every BORROWED mapping closed (the peers' regions, their captured
// inputs), then this rank's OWNED buffers freed, and the state with them; every
// step tried, the first failure's Error. Every launch on the device finished
// first (the caller's). A COLLECTIVE unless `alone`: a handshake over the group
// once every rank stopped, its peers' mappings closed, a second once every
// rank closed its mappings of ours, then ours freed. A handshake a peer misses
// within the state's timeout is exchange_failed, and the close carries on
// alone (a peer that missed the first would pair this rank's second with its
// own first). ALONE (an exit, a signal, a failed open): no handshake, never
// waiting on a peer.
ErrorCode hip_custom_communicator_close(int64_t state, bool alone) {
  using namespace hip_custom_communicator;
  using namespace hip_custom_communicator::internal;
  if (state == 0) return error_code(Error::not_open);
  auto* s = reinterpret_cast<CommsState*>(state);
  ErrorCode first;
  if (!alone && !all_gathered(*s->group, "1", s->timeout_seconds)) {
    first = error_code(Error::exchange_failed);
    alone = true;
  }
  std::vector<char*> borrowed = s->symmetric.peers;
  for (const auto& [key, base] : s->input_tables.opened)
    borrowed.push_back(base);
  for (char* base : borrowed)
    if (hipIpcCloseMemHandle(base) != hipSuccess) {
      (void)hipGetLastError();
      if (!first) first = error_code(Error::hip_failed);
    }
  if (!alone && !all_gathered(*s->group, "1", s->timeout_seconds) && !first)
    first = error_code(Error::exchange_failed);
  for (const void* p :
       {static_cast<const void*>(s->symmetric.signals),
        static_cast<const void*>(s->symmetric.staging),
        static_cast<const void*>(s->symmetric.partials),
        static_cast<const void*>(s->symmetric.scales),
        static_cast<const void*>(s->symmetric.unfenced),
        static_cast<const void*>(s->symmetric.signals_ptr_array),
        static_cast<const void*>(s->symmetric.staging_ptr_array),
        static_cast<const void*>(s->symmetric.partials_ptr_array),
        static_cast<const void*>(s->symmetric.scales_ptr_array),
        static_cast<const void*>(s->symmetric.unfenced_ptr_array),
        static_cast<const void*>(s->unfenced_epochs),
        static_cast<const void*>(s->status_device),
        static_cast<const void*>(s->input_tables.device)})
    if (p != nullptr && hipFree(const_cast<void*>(p)) != hipSuccess) {
      (void)hipGetLastError();
      if (!first) first = error_code(Error::hip_failed);
    }
  delete s;
  return first;
}
