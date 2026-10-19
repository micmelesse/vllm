# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only subprocess checks of the communicator collection policy.

Can run without vLLM dependencies with pytest --noconftest and this file.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent


@pytest.fixture
def policy_suite(tmp_path):
    (tmp_path / "conftest.py").write_text(
        (TESTS / "communicator_resources.py").read_text()
    )
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n cpu\n gpu\n multi_gpu\n smoke\n example\n"
    )
    (tmp_path / "test_hip_custom_communicator.py").write_text(
        "import pytest\n"
        "@pytest.mark.cpu\n@pytest.mark.smoke\ndef test_cpu(): pass\n"
        "@pytest.mark.gpu\n@pytest.mark.example\ndef test_gpu(): pass\n"
        "@pytest.mark.multi_gpu\n@pytest.mark.example\ndef test_multi(): pass\n"
    )
    (tmp_path / "test_other.py").write_text("def test_unrelated(): pass\n")
    # Any accidental torch import during collection fails, even on GPU machines.
    (tmp_path / "torch.py").write_text("raise AssertionError('GPU queried')\n")

    def run(*args):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *args],
            cwd=tmp_path,
            env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode, result.stdout + result.stderr

    return tmp_path, run


@pytest.mark.cpu
@pytest.mark.example
@pytest.mark.parametrize(
    ("resource", "counts"),
    [
        ("cpu", "cpu=1, gpu=0, multi_gpu=0"),
        ("gpu", "cpu=1, gpu=1, multi_gpu=0"),
        ("multi_gpu", "cpu=1, gpu=1, multi_gpu=1"),
    ],
)
def test_resource_selection_is_cumulative_without_device_queries(
    policy_suite, resource, counts
):
    _, run = policy_suite
    code, output = run("--collect-only", "--on", resource)
    assert code == 0, output
    assert f"Communicator collection resources: {counts}" in output
    assert "test_other.py::test_unrelated" in output
    examples = ("cpu", "gpu", "multi_gpu").index(resource)
    assert f"smoke=1, example={examples}, regression=0" in output


@pytest.mark.cpu
@pytest.mark.example
def test_default_executes_only_cpu_and_leaves_unrelated_tests_alone(policy_suite):
    _, run = policy_suite
    code, output = run()
    assert code == 0, output
    assert "2 passed, 2 deselected" in output


@pytest.mark.cpu
@pytest.mark.example
def test_invalid_resources_fail_before_marker_filtering(policy_suite):
    path, run = policy_suite
    with (path / "test_hip_custom_communicator.py").open("a") as f:
        f.write(
            "\ndef test_missing(): pass\n"
            "@pytest.mark.cpu\n@pytest.mark.gpu\ndef test_double(): pass\n"
        )
    code, output = run("--collect-only", "-m", "smoke")
    assert code == pytest.ExitCode.USAGE_ERROR, output
    for name in ("missing", "double"):
        assert (
            f"test_hip_custom_communicator.py::test_{name}: "
            "expected exactly one resource"
        ) in output


@pytest.mark.cpu
@pytest.mark.example
@pytest.mark.parametrize("visible", [0, 1, 4, 7, 8, 9])
def test_multi_gpu_execution_refuses_any_world_other_than_eight(policy_suite, visible):
    path, run = policy_suite
    (path / "torch.py").write_text(
        "from types import SimpleNamespace\n"
        f"cuda = SimpleNamespace(device_count=lambda: {visible})\n"
    )
    code, output = run("--on", "multi_gpu", "-k", "test_multi")
    if visible == 8:
        assert code == 0, output
        assert "1 passed" in output
    else:
        assert code == pytest.ExitCode.TESTS_FAILED, output
        assert f"requires exactly 8 visible GPUs, got {visible}" in output


@pytest.mark.cpu
@pytest.mark.example
def test_multi_gpu_helper_supplies_resource_without_querying_devices():
    # Compile the actual helper alone to avoid importing unrelated model/test
    # dependencies. No platform object exists: querying it would fail here.
    tree = ast.parse((TESTS / "utils.py").read_text())
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "multi_gpu_marks"
    )
    namespace = {"pytest": pytest}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), "tests/utils.py", "exec"),
        namespace,
    )
    marks = namespace["multi_gpu_marks"](num_gpus=8)
    assert [mark.name for mark in marks] == ["multi_gpu", "distributed", "skipif"]
    assert marks[1].kwargs == {"num_gpus": 8}
