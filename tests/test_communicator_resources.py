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
        "@pytest.mark.multi_gpu(2)\n@pytest.mark.example\ndef test_multi2(): pass\n"
        "@pytest.mark.multi_gpu(4)\n@pytest.mark.example\ndef test_multi4(): pass\n"
        "@pytest.mark.multi_gpu(8)\n@pytest.mark.example\ndef test_multi8(): pass\n"
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
        ("gpus2", "cpu=1, gpu=1, multi_gpu=1"),
        ("gpus4", "cpu=1, gpu=1, multi_gpu=2"),
        ("gpus8", "cpu=1, gpu=1, multi_gpu=3"),
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
    examples = ("cpu", "gpu", "gpus2", "gpus4", "gpus8").index(resource)
    assert f"smoke=1, example={examples}, regression=0" in output


@pytest.mark.cpu
@pytest.mark.example
def test_default_executes_only_cpu_and_leaves_unrelated_tests_alone(policy_suite):
    _, run = policy_suite
    code, output = run()
    assert code == 0, output
    assert "2 passed, 4 deselected" in output


@pytest.mark.cpu
@pytest.mark.example
def test_invalid_resources_fail_before_marker_filtering(policy_suite):
    path, run = policy_suite
    with (path / "test_hip_custom_communicator.py").open("a") as f:
        f.write(
            "\ndef test_missing(): pass\n"
            "@pytest.mark.cpu\n@pytest.mark.gpu\ndef test_double(): pass\n"
            "@pytest.mark.multi_gpu(3)\ndef test_invalid_world(): pass\n"
        )
    code, output = run("--collect-only", "-m", "smoke")
    assert code == pytest.ExitCode.USAGE_ERROR, output
    assert "test_invalid_world: multi_gpu requires one positional world size" in output
    for name in ("missing", "double"):
        assert (
            f"test_hip_custom_communicator.py::test_{name}: "
            "expected exactly one resource"
        ) in output


@pytest.mark.cpu
@pytest.mark.example
@pytest.mark.parametrize("visible", [0, 1, 4, 7, 8, 9])
@pytest.mark.parametrize("world", [2, 4, 8])
def test_multi_gpu_execution_requires_its_declared_world(policy_suite, visible, world):
    path, run = policy_suite
    (path / "torch.py").write_text(
        "from types import SimpleNamespace\n"
        f"cuda = SimpleNamespace(device_count=lambda: {visible})\n"
    )
    code, output = run("--on", "gpus8", "-k", f"test_multi{world}")
    if visible >= world:
        assert code == 0, output
        assert "1 passed" in output
    else:
        assert code == pytest.ExitCode.TESTS_FAILED, output
        assert f"requires at least {world} visible GPUs, got {visible}" in output


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
    assert marks[0].args == (8,)
    assert marks[1].kwargs == {"num_gpus": 8}


@pytest.mark.cpu
@pytest.mark.example
def test_actual_world_parameters_reach_workers_without_global_mutation():
    tree = ast.parse(
        (TESTS / "distributed/test_hip_custom_communicator.py").read_text()
    )
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    worker = ast.unparse(functions["_rank_main"])
    assert "init_test_distributed_environment(world_size, 1, rank" in worker
    launcher = ast.unparse(functions["_on_every_rank"])
    assert "for r in range(world_size)" in launcher
    assert "args=(r, world_size, port" in launcher
    assert "WORLD_SIZE" not in ast.unparse(tree)
    for name in (
        "test_op_matches_the_ops_it_replaces",
        "test_int4_two_shot_matches_the_format_it_moves",
        "test_refused_calls_name_why_and_fall_back_cleanly",
        "test_capture_warmup_launches_nothing",
    ):
        function = functions[name]
        assert "worlds" in [ast.unparse(d) for d in function.decorator_list]
        assert "world_size" in [arg.arg for arg in function.args.args]
        call = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_on_every_rank"
        )
        assert ast.unparse(call.args[1]) == "world_size"
