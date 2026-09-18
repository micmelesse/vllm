# SPDX-License-Identifier: MIT Copyright (C) 2026, Advanced Micro Devices, Inc. All
# rights reserved.

"""The small things more than one backend needs and none of them owns.

NOT `tunables.py`, WHICH IS THE KNOBS. A tunable is a number you may change with the
code still correct; what is here is DERIVED -- nobody sets it, and changing it would
just be wrong. They read alike and are opposites.
"""

import torch

from vllm.config import get_current_vllm_config_or_none

__all__ = ["widest_input_bytes"]


def widest_input_bytes() -> int:
    """ONE ALL-REDUCE INPUT at the largest batch vLLM will build, or 0 if there is no
    vLLM around us.

    A FACT ABOUT THE WORKLOAD AND NOT ABOUT A BACKEND, which is why neither owns it: hip
    sizes its staging buffer by it and iris its heap, and two copies of
    `max_num_batched_tokens x hidden x itemsize` would be two places to be wrong about
    the biggest thing we will be handed.

    ZERO AND NOT AN ERROR: this package is usable without vLLM around it -- the
    correctness suite runs it that way -- and a caller adds its own floor.
    """
    config = get_current_vllm_config_or_none()
    try:
        assert config is not None
        widest = config.scheduler_config.max_num_batched_tokens
        row = config.model_config.get_hidden_size()
        item = torch.empty(0, dtype=config.model_config.dtype).element_size()
        return int(widest) * int(row) * int(item)
    except Exception:
        return 0
