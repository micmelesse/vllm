#pragma once

// address: one rank's entry of a pointer table, read from memory.

#include "types.cuh"

namespace hip_custom_communicator {

// A buffer's pointer table holds every rank's address of it. A graph's inputs
// are known only after capture (register_graph fills their table), so a kernel
// reads the entry at run time; a caller that reads it before a barrier has it
// when the barrier ends, where send and recv would read it after. The
// symmetric buffers' tables are read the same way.
template <typename T>
DINLINE T* address(T* const* x_ptr_array, int x_ptr_array_size, int x_rank) {
  return internal::ptr_array_at(x_ptr_array, x_ptr_array_size, x_rank);
}

}  // namespace hip_custom_communicator
