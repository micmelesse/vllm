# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""rocm_comms's experimental ops, on one GPU, against the ops they would replace."""

import pytest
import torch

pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from vllm.distributed.device_communicators.rocm_comms.base import build_info

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the kernels run on a GPU"
)

# KIMI-K3'S SHAPES: its two hidden widths, and up to ten stored sources a row.
HIDDENS = (3584, 7168)
SOURCES = 10


def _randn(seed: int, shape: tuple[int, ...]) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g).to(torch.bfloat16).cuda()


# (rows, hidden, num_blocks, write_idx, output norm, a forced config or none)
AttnResCall = tuple[int, int, int, int, bool, dict[str, int]]


@st.composite
def _attn_res_calls(draw: st.DrawFn) -> AttnResCall:
    hidden = draw(st.sampled_from(HIDDENS))
    rows = draw(st.integers(1, 600))
    num_blocks = draw(st.integers(0, SOURCES - 1))
    write_idx = draw(st.sampled_from((-1, num_blocks)))
    configs = [
        c
        for c in build_info().templates["add_attn_res_rms_norm"].configs
        if c["tile_n"] >= hidden
    ]
    forced = draw(st.sampled_from([{}, *configs]))
    if forced:
        forced = {**forced, "blocks_per_grid": draw(st.integers(1, 1024))}
    return rows, hidden, num_blocks, write_idx, draw(st.booleans()), forced


@settings(max_examples=40, deadline=None)
@given(call=_attn_res_calls())
def test_add_attn_res_rms_norm_matches_triton_attn_res(call: AttnResCall) -> None:
    """Every output our AttnRes writes (out, the updated prefix, the written block)
    against Triton's `attn_res` with a delta, at any rows, sources and config."""
    import vllm._rocm_C  # noqa: F401  (registers torch.ops._rocm_C)
    from vllm.distributed.device_communicators.rocm_comms.experimental import (
        add_attn_res_rms_norm,
    )
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res

    rows, hidden, num_blocks, write_idx, output_norm, forced = call
    shape = (rows, hidden)
    prefix, delta = _randn(0, shape), _randn(1, shape)
    blocks = _randn(2, (rows, SOURCES, hidden))
    out_w = _randn(5, (hidden,)) if output_norm else None
    args = (_randn(3, (hidden,)), _randn(4, (hidden,)), out_w, num_blocks, write_idx)

    want_prefix, want_blocks = prefix.clone(), blocks.clone()
    want = attn_res(want_prefix, delta, want_blocks, *args, 1e-6, 1e-5)
    got_prefix, got_blocks = prefix.clone(), blocks.clone()
    got = add_attn_res_rms_norm(
        got_prefix, delta, got_blocks, *args, 1e-6, 1e-5, **forced
    )
    torch.accelerator.synchronize()

    # The prefix and the block are one rounding of prefix + delta in both; the output's
    # sums run in another order.
    torch.testing.assert_close(got_prefix, want_prefix, atol=0, rtol=0)
    torch.testing.assert_close(got_blocks, want_blocks, atol=0, rtol=0)
    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)
