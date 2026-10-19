# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resource selection for HipCustomCommunicator tests only."""

from collections import Counter

import pytest

RESOURCES = ("cpu", "gpu", "multi_gpu")
KINDS = ("smoke", "example", "regression", "property", "stress", "fixture")


def _communicator(item):
    return item.path.name == "test_hip_custom_communicator.py"


def pytest_addoption(parser):
    parser.addoption(
        "--on",
        choices=RESOURCES,
        default="cpu",
        help="Communicator resources: cpu (default), gpu (includes cpu), "
        "multi_gpu (all; requires exactly 8 visible GPUs at execution)",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    allowed = RESOURCES[: RESOURCES.index(config.getoption("--on")) + 1]
    selected, deselected, errors = [], [], []
    for item in items:
        if not _communicator(item):
            selected.append(item)
            continue
        marks = [m.name for m in item.iter_markers() if m.name in RESOURCES]
        if len(marks) != 1:
            errors.append(f"{item.nodeid}: expected exactly one resource, got {marks}")
        elif marks[0] in allowed:
            selected.append(item)
        else:
            deselected.append(item)
    if errors:
        raise pytest.UsageError("\n".join(errors))
    items[:] = selected
    if deselected:
        config.hook.pytest_deselected(items=deselected)


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
    if item.get_closest_marker("multi_gpu"):
        if count != 8:
            pytest.fail(f"{item.nodeid}: requires exactly 8 visible GPUs, got {count}")
    elif count < 1:
        pytest.fail(f"{item.nodeid}: requires at least one visible GPU, got {count}")
