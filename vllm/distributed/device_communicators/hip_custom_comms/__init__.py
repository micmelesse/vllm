# SPDX-License-Identifier: MIT Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""HipCustomComms: vLLM's TP all-reduce and the all-reduces fused with the ops after
them, over our HIP kernels (csrc/rocm/hip_custom_comms.cu).

    hip_custom_comms  the communicator
    build             the build's facts: its ops, Errors, contents and support

IMPORTED ON USE: the kernels are compiled into `_rocm_C`, so importing this package
costs nothing.
"""

from .build import Error, Refused, build_info
from .hip_custom_comms import HipCustomComms

__all__ = ["Error", "HipCustomComms", "Refused", "build_info"]
