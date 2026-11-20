#pragma once

// Closes the communicator and frees its buffers.

#include "../../ops.h"
#include "../types.cuh"
#include "../utils.h"

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
  for (char* base : s->symmetric.peers)
    if (hipIpcCloseMemHandle(base) != hipSuccess) {
      (void)hipGetLastError();
      if (!first) first = error_code(Error::hip_failed);
    }
  if (!alone && !all_gathered(*s->group, "1", s->timeout_seconds) && !first)
    first = error_code(Error::exchange_failed);
  for (const void* p :
       {static_cast<const void*>(s->symmetric.signals),
        static_cast<const void*>(s->symmetric.unfenced),
        static_cast<const void*>(s->symmetric.lamport),
        static_cast<const void*>(s->symmetric.simple_write_inbox),
        static_cast<const void*>(s->symmetric.simple_read_inbox),
        static_cast<const void*>(s->symmetric.signals_ptr_array),
        static_cast<const void*>(s->symmetric.simple_write_counters_ptr_array),
        static_cast<const void*>(s->symmetric.simple_read_counters_ptr_array),
        static_cast<const void*>(s->symmetric.unfenced_ptr_array),
        static_cast<const void*>(s->symmetric.lamport_ptr_array),
        static_cast<const void*>(s->symmetric.simple_write_inbox_ptr_array),
        static_cast<const void*>(s->symmetric.simple_read_inbox_ptr_array),
        static_cast<const void*>(s->steps),
        static_cast<const void*>(s->status_device)})
    if (p != nullptr && hipFree(const_cast<void*>(p)) != hipSuccess) {
      (void)hipGetLastError();
      if (!first) first = error_code(Error::hip_failed);
    }
  delete s;
  return first;
}
