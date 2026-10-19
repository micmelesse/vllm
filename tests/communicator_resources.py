# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resource selection for HipCustomCommunicator tests only."""

from collections import Counter

import pytest

RESOURCES = ("cpu", "gpu", "multi_gpu")
TIERS = ("cpu", "gpu", "gpus2", "gpus4", "gpus8")
KINDS = ("smoke", "example", "regression", "property", "stress", "fixture")


def _communicator(item):
    return item.path.name == "test_hip_custom_communicator.py"


def _required(item):
    mark = item.get_closest_marker("multi_gpu")
    if mark is not None:
        if len(mark.args) != 1 or mark.args[0] not in (2, 4, 8) or mark.kwargs:
            raise ValueError("multi_gpu requires one positional world size: 2, 4, or 8")
        return mark.args[0]
    return 1 if item.get_closest_marker("gpu") else 0


def pytest_addoption(parser):
    parser.addoption(
        "--on",
        choices=TIERS,
        default="cpu",
        help="Communicator resources: cpu (default), gpu (includes cpu), "
        "gpus2, gpus4, gpus8 (cumulative GPU world sizes)",
    )


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_collection_modifyitems(config, items):
    limit = (0, 1, 2, 4, 8)[TIERS.index(config.getoption("--on"))]
    selected, deselected, errors = [], [], []
    for item in items:
        if not _communicator(item):
            selected.append(item)
            continue
        marks = [m.name for m in item.iter_markers() if m.name in RESOURCES]
        if len(marks) != 1:
            errors.append(f"{item.nodeid}: expected exactly one resource, got {marks}")
        else:
            try:
                required = _required(item)
            except ValueError as error:
                errors.append(f"{item.nodeid}: {error}")
                continue
            (selected if required <= limit else deselected).append(item)
    if errors:
        raise pytest.UsageError("\n".join(errors))
    items[:] = selected
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    return (yield)


def pytest_collection_finish(session):
    items = [item for item in session.items if _communicator(item)]
    if not items:
        return
    counts = Counter(m.name for item in items for m in item.iter_markers())
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter:
        reporter.write_line(
            "Communicator collection resources: "
            + ", ".join(f"{name}={counts[name]}" for name in RESOURCES)
        )
        reporter.write_line(
            "Communicator collection kinds: "
            + ", ".join(f"{name}={counts[name]}" for name in KINDS)
        )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    if not _communicator(item) or item.get_closest_marker("cpu"):
        return
    # This runs only for executed tests, never while collecting.
    import torch

    count = torch.cuda.device_count()
    required = _required(item)
    if count < required:
        pytest.fail(
            f"{item.nodeid}: requires at least {required} visible GPUs, got {count}"
        )
