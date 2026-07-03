import json
import subprocess
import sys
from pathlib import Path

from sebench_infra.authoring import AuthoringAgent, MockLLMClient
from sebench_infra.benchmark import DatasetBuilder
from sebench_infra.benchmark.schemas import DatasetSpec, ScoringRule, TaskCategory, TaskSpec
from sebench_infra.orchestrator import EvaluationOrchestrator
from sebench_infra.spatial import SpatialDiagnosisEngine


def _load_git_pytest_dataset() -> DatasetSpec:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "examples/git_pytest_benchmark.json"
    if not manifest.exists():
        subprocess.run(
            [
                sys.executable,
                "scripts/build_git_pytest_benchmark.py",
                "--tasks",
                "128",
                "--out",
                str(manifest),
            ],
            cwd=root,
            check=True,
        )
    return DatasetSpec.model_validate_json(manifest.read_text(encoding="utf-8"))


def test_mock_reproduction_pipeline(tmp_path: Path) -> None:
    requirement = "Build a reproducible 3D spatial benchmark task."
    tasks = AuthoringAgent(MockLLMClient()).author_tasks(requirement)
    dataset = DatasetBuilder().build(requirement, tasks, ["https://arxiv.org/abs/2310.06770"])

    report = EvaluationOrchestrator().run(dataset)

    assert report.aggregate_score == 1.0
    assert report.regression_passed


def test_spatial_scene_file_fixture() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = SpatialDiagnosisEngine().diagnose_scene_file(
        root / "examples/synthetic_spatial_scene.json"
    )

    assert payload["diagnosis"]["geometry_bridge_ready"] is True
    assert payload["scene_id"] == "synthetic-scannet-style-room-001"
    assert json.dumps(payload["facts"])


def test_toy_true_loop_dataset_produces_run_records() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = DatasetSpec.model_validate_json(
        (root / "examples/toy_benchmark.json").read_text(encoding="utf-8")
    )

    report = EvaluationOrchestrator().run(dataset)

    assert report.aggregate_score == 1.0
    assert len(report.run_records) == len(dataset.tasks)
    assert {record.pass_fail for record in report.run_records} == {"pass"}
    assert all(record.patch_hash for record in report.run_records)
    assert all(
        record.metrics.wall_time_sec >= record.metrics.judge_time_sec
        for record in report.run_records
    )


def test_toy_true_loop_failure_has_failure_type_and_no_hidden_artifact() -> None:
    task = TaskSpec(
        task_id="toy.python.bad_patch",
        title="Bad patch",
        category=TaskCategory.CODE_REPAIR,
        prompt="Fix add.",
        fixtures={
            "toy_repo": {
                "repo_id": "toy-python-bad",
                "files": {"calculator.py": "def add(a, b):\n    return a - b\n"},
                "agent_files": {
                    "calculator.py": "def add(a, b):\n    return a - b\n",
                    "hidden_tests/test_secret.py": "SECRET = True\n",
                },
                "hidden_judge": [
                    {
                        "kind": "python_function",
                        "path": "calculator.py",
                        "function": "add",
                        "args": [2, 3],
                        "expected": 5,
                    }
                ],
            }
        },
        expected_artifacts=["calculator.py"],
        allowed_paths=["calculator.py"],
        scoring=[
            ScoringRule(
                name="hidden_judge_passed",
                kind="metadata_flag",
                expected="judge_passed",
            )
        ],
    )
    dataset = DatasetSpec(dataset_id="toy-failing", tasks=[task])

    report = EvaluationOrchestrator().run(dataset)

    assert report.task_results[0].status == "failed"
    assert report.run_records[0].failure_type == "wrong_edit"
    assert "hidden_tests/test_secret.py" not in report.task_results[0].artifacts


def test_pytest_hidden_judge_timeout_has_specific_failure_type() -> None:
    task = TaskSpec(
        task_id="toy.pytest.timeout",
        title="Slow hidden pytest",
        category=TaskCategory.CODE_REPAIR,
        prompt="Fix add.",
        fixtures={
            "toy_repo": {
                "repo_id": "toy-pytest-timeout",
                "files": {"calculator.py": "def add(a, b):\n    return a - b\n"},
                "agent_files": {"calculator.py": "def add(a, b):\n    return a + b\n"},
                "hidden_judge": [
                    {
                        "kind": "pytest",
                        "files": {
                            "tests/test_slow.py": (
                                "import time\n\n"
                                "def test_slow():\n"
                                "    time.sleep(1)\n"
                            )
                        },
                        "timeout_sec": 0.01,
                    }
                ],
            }
        },
        expected_artifacts=["calculator.py"],
        allowed_paths=["calculator.py"],
        scoring=[
            ScoringRule(
                name="hidden_judge_passed",
                kind="metadata_flag",
                expected="judge_passed",
            )
        ],
    )
    dataset = DatasetSpec(dataset_id="pytest-timeout", tasks=[task])

    report = EvaluationOrchestrator().run(dataset)

    assert report.run_records[0].failure_type == "hidden_judge_timeout"


def test_pytest_timeout_override_has_priority_over_fixture_timeout() -> None:
    task = TaskSpec(
        task_id="toy.pytest.timeout.override",
        title="Slow hidden pytest with override",
        category=TaskCategory.CODE_REPAIR,
        prompt="Fix add.",
        fixtures={
            "toy_repo": {
                "repo_id": "toy-pytest-timeout-override",
                "files": {"calculator.py": "def add(a, b):\n    return a - b\n"},
                "agent_files": {"calculator.py": "def add(a, b):\n    return a + b\n"},
                "hidden_judge": [
                    {
                        "kind": "pytest",
                        "files": {
                            "tests/test_slow.py": (
                                "import time\n\n"
                                "def test_slow():\n"
                                "    time.sleep(1)\n"
                            )
                        },
                        "timeout_sec": 5,
                    }
                ],
            }
        },
        expected_artifacts=["calculator.py"],
        allowed_paths=["calculator.py"],
        scoring=[
            ScoringRule(
                name="hidden_judge_passed",
                kind="metadata_flag",
                expected="judge_passed",
            )
        ],
    )
    dataset = DatasetSpec(dataset_id="pytest-timeout-override", tasks=[task])

    report = EvaluationOrchestrator(pytest_timeout_sec=0.01).run(dataset)

    assert report.run_records[0].failure_type == "hidden_judge_timeout"
    assert report.run_records[0].metadata["pytest_timeout_sec"] == 0.01


def test_hard_task_timeout_kills_non_pytest_hang() -> None:
    task = TaskSpec(
        task_id="toy.task.timeout",
        title="Task hard timeout",
        category=TaskCategory.CODE_REPAIR,
        prompt="Run a hanging in-process judge function.",
        fixtures={
            "toy_repo": {
                "repo_id": "toy-task-timeout",
                "files": {
                    "slow.py": (
                        "import time\n\n"
                        "def wait_for_result():\n"
                        "    time.sleep(5)\n"
                        "    return 1\n"
                    )
                },
                "agent_files": {},
                "hidden_judge": [
                    {
                        "kind": "python_function",
                        "path": "slow.py",
                        "function": "wait_for_result",
                        "expected": 1,
                    }
                ],
            }
        },
        expected_artifacts=["slow.py"],
        allowed_paths=["slow.py"],
        scoring=[
            ScoringRule(
                name="hidden_judge_passed",
                kind="metadata_flag",
                expected="judge_passed",
            )
        ],
    )
    dataset = DatasetSpec(dataset_id="task-timeout", tasks=[task])

    report = EvaluationOrchestrator().run(dataset, workers=1, task_timeout_sec=0.2)

    assert report.task_results[0].status == "error"
    assert report.run_records[0].failure_type == "task_timeout"
    assert report.run_records[0].metadata["hard_timeout_triggered"] is True
    assert report.run_records[0].metadata["killed_pid"] is not None


def test_pytest_plugin_autoload_is_disabled_by_default_for_hidden_pytest() -> None:
    task = TaskSpec(
        task_id="toy.pytest.plugin_autoload",
        title="Hidden pytest env",
        category=TaskCategory.CODE_REPAIR,
        prompt="Fix add.",
        fixtures={
            "toy_repo": {
                "repo_id": "toy-pytest-plugin-autoload",
                "files": {"calculator.py": "def add(a, b):\n    return a - b\n"},
                "agent_files": {"calculator.py": "def add(a, b):\n    return a + b\n"},
                "hidden_judge": [
                    {
                        "kind": "pytest",
                        "files": {
                            "tests/test_env.py": (
                                "import os\n"
                                "from calculator import add\n\n"
                                "def test_env_and_add():\n"
                                "    assert os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] == '1'\n"
                                "    assert add(2, 3) == 5\n"
                            )
                        },
                    }
                ],
            }
        },
        expected_artifacts=["calculator.py"],
        allowed_paths=["calculator.py"],
        scoring=[
            ScoringRule(
                name="hidden_judge_passed",
                kind="metadata_flag",
                expected="judge_passed",
            )
        ],
    )
    dataset = DatasetSpec(dataset_id="pytest-plugin-autoload", tasks=[task])

    report = EvaluationOrchestrator().run(dataset)

    assert report.aggregate_score == 1.0
    assert report.run_records[0].metadata["disable_pytest_plugin_autoload"] is True
    assert report.run_records[0].metadata["judge_details"][0][
        "disable_pytest_plugin_autoload"
    ] is True


def test_pytest_plugin_autoload_can_be_enabled_for_external_plugin_tests() -> None:
    task = TaskSpec(
        task_id="toy.pytest.plugin_autoload_enabled",
        title="Hidden pytest env opt out",
        category=TaskCategory.CODE_REPAIR,
        prompt="Fix add.",
        fixtures={
            "toy_repo": {
                "repo_id": "toy-pytest-plugin-autoload-enabled",
                "files": {"calculator.py": "def add(a, b):\n    return a - b\n"},
                "agent_files": {"calculator.py": "def add(a, b):\n    return a + b\n"},
                "hidden_judge": [
                    {
                        "kind": "pytest",
                        "files": {
                            "tests/test_env.py": (
                                "import os\n"
                                "from calculator import add\n\n"
                                "def test_env_and_add():\n"
                                "    disabled = os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD')\n"
                                "    assert disabled != '1'\n"
                                "    assert add(2, 3) == 5\n"
                            )
                        },
                    }
                ],
            }
        },
        expected_artifacts=["calculator.py"],
        allowed_paths=["calculator.py"],
        scoring=[
            ScoringRule(
                name="hidden_judge_passed",
                kind="metadata_flag",
                expected="judge_passed",
            )
        ],
    )
    dataset = DatasetSpec(dataset_id="pytest-plugin-autoload-enabled", tasks=[task])

    report = EvaluationOrchestrator(disable_pytest_plugin_autoload=False).run(dataset)

    assert report.aggregate_score == 1.0
    assert report.run_records[0].metadata["disable_pytest_plugin_autoload"] is False
    assert report.run_records[0].metadata["judge_details"][0][
        "disable_pytest_plugin_autoload"
    ] is False


def test_pytest_plugin_policy_auto_enables_autoload_when_plugin_dependency_detected() -> None:
    task = TaskSpec(
        task_id="toy.pytest.plugin_auto_detect",
        title="Hidden pytest plugin auto detect",
        category=TaskCategory.CODE_REPAIR,
        prompt="Fix add.",
        fixtures={
            "toy_repo": {
                "repo_id": "toy-pytest-plugin-auto-detect",
                "files": {
                    "calculator.py": "def add(a, b):\n    return a - b\n",
                    "pyproject.toml": (
                        "[project]\n"
                        "dependencies = ['pytest-asyncio>=0.23']\n"
                    ),
                },
                "agent_files": {"calculator.py": "def add(a, b):\n    return a + b\n"},
                "hidden_judge": [
                    {
                        "kind": "pytest",
                        "files": {
                            "tests/test_env.py": (
                                "import os\n"
                                "from calculator import add\n\n"
                                "def test_env_and_add():\n"
                                "    disabled = os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD')\n"
                                "    assert disabled != '1'\n"
                                "    assert add(2, 3) == 5\n"
                            )
                        },
                    }
                ],
            }
        },
        expected_artifacts=["calculator.py"],
        allowed_paths=["calculator.py"],
        scoring=[
            ScoringRule(
                name="hidden_judge_passed",
                kind="metadata_flag",
                expected="judge_passed",
            )
        ],
    )
    dataset = DatasetSpec(dataset_id="pytest-plugin-auto-detect", tasks=[task])

    report = EvaluationOrchestrator().run(dataset)

    assert report.aggregate_score == 1.0
    metadata = report.run_records[0].metadata
    assert metadata["pytest_plugin_policy"] == "auto"
    assert metadata["disable_pytest_plugin_autoload"] is False
    assert metadata["pytest_plugin_scan"]["effective_policy"] == "auto_enabled"
    assert metadata["pytest_plugin_scan"]["plugin_required"] is True


def test_pytest_plugin_policy_disabled_overrides_detected_plugin_dependency() -> None:
    task = TaskSpec(
        task_id="toy.pytest.plugin_disabled_override",
        title="Hidden pytest plugin disabled override",
        category=TaskCategory.CODE_REPAIR,
        prompt="Fix add.",
        fixtures={
            "toy_repo": {
                "repo_id": "toy-pytest-plugin-disabled-override",
                "files": {
                    "calculator.py": "def add(a, b):\n    return a - b\n",
                    "pyproject.toml": (
                        "[project]\n"
                        "dependencies = ['pytest-django>=4.8']\n"
                    ),
                },
                "agent_files": {"calculator.py": "def add(a, b):\n    return a + b\n"},
                "hidden_judge": [
                    {
                        "kind": "pytest",
                        "files": {
                            "tests/test_env.py": (
                                "import os\n"
                                "from calculator import add\n\n"
                                "def test_env_and_add():\n"
                                "    assert os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] == '1'\n"
                                "    assert add(2, 3) == 5\n"
                            )
                        },
                    }
                ],
            }
        },
        expected_artifacts=["calculator.py"],
        allowed_paths=["calculator.py"],
        scoring=[
            ScoringRule(
                name="hidden_judge_passed",
                kind="metadata_flag",
                expected="judge_passed",
            )
        ],
    )
    dataset = DatasetSpec(dataset_id="pytest-plugin-disabled-override", tasks=[task])

    report = EvaluationOrchestrator(pytest_plugin_policy="disabled").run(dataset)

    assert report.aggregate_score == 1.0
    metadata = report.run_records[0].metadata
    assert metadata["pytest_plugin_policy"] == "disabled"
    assert metadata["disable_pytest_plugin_autoload"] is True
    assert metadata["pytest_plugin_scan"]["effective_policy"] == "disabled"
    assert metadata["pytest_plugin_scan"]["plugin_required"] is True


def test_inline_hidden_judge_supports_repo_imports() -> None:
    task = TaskSpec(
        task_id="realistic.python.discount",
        title="Fix package discount calculation",
        category=TaskCategory.CODE_REPAIR,
        prompt="Fix pricing.discounted_total so it applies the discount to the subtotal.",
        fixtures={
            "toy_repo": {
                "repo_id": "realistic-shop",
                "files": {
                    "app/__init__.py": "",
                    "app/pricing.py": (
                        "def discounted_total(items, discount):\n"
                        "    subtotal = sum(items)\n"
                        "    return subtotal + discount\n"
                    ),
                },
                "agent_files": {
                    "app/pricing.py": (
                        "def discounted_total(items, discount):\n"
                        "    subtotal = sum(items)\n"
                        "    return round(subtotal * (1 - discount), 2)\n"
                    )
                },
                "hidden_judge": [
                    {
                        "kind": "python_inline_tests",
                        "code": (
                            "from app.pricing import discounted_total\n"
                            "assert discounted_total([10, 20], 0.1) == 27.0\n"
                            "print('ok')\n"
                        ),
                    }
                ],
            }
        },
        expected_artifacts=["app/pricing.py"],
        allowed_paths=["app/"],
        scoring=[
            ScoringRule(
                name="hidden_judge_passed",
                kind="metadata_flag",
                expected="judge_passed",
            )
        ],
    )
    dataset = DatasetSpec(dataset_id="realistic-import", tasks=[task])

    report = EvaluationOrchestrator().run(dataset)

    assert report.aggregate_score == 1.0
    assert report.run_records[0].metadata["judge_details"][0]["stdout"] == "ok"


def test_inline_hidden_judge_clears_same_package_import_cache() -> None:
    def make_task(task_id: str, path: str, function: str, expected: int) -> TaskSpec:
        return TaskSpec(
            task_id=task_id,
            title=task_id,
            category=TaskCategory.CODE_REPAIR,
            prompt=f"Fix {function}.",
            fixtures={
                "toy_repo": {
                    "repo_id": task_id,
                    "files": {
                        "app/__init__.py": "",
                        path: f"def {function}():\n    return 0\n",
                    },
                    "agent_files": {path: f"def {function}():\n    return {expected}\n"},
                    "hidden_judge": [
                        {
                            "kind": "python_inline_tests",
                            "code": (
                                f"from app.{Path(path).stem} import {function}\n"
                                f"assert {function}() == {expected}\n"
                            ),
                        }
                    ],
                }
            },
            expected_artifacts=[path],
            allowed_paths=["app/"],
            scoring=[
                ScoringRule(
                    name="hidden_judge_passed",
                    kind="metadata_flag",
                    expected="judge_passed",
                )
            ],
        )

    dataset = DatasetSpec(
        dataset_id="same-package-cache",
        tasks=[
            make_task("realistic.cache.first", "app/first.py", "first", 1),
            make_task("realistic.cache.second", "app/second.py", "second", 2),
        ],
    )

    report = EvaluationOrchestrator().run(dataset)

    assert report.aggregate_score == 1.0
    assert {record.pass_fail for record in report.run_records} == {"pass"}


def test_patch_submission_applies_in_clean_judge_dir() -> None:
    task = TaskSpec(
        task_id="patch.python.add",
        title="Patch add helper",
        category=TaskCategory.CODE_REPAIR,
        prompt="Submit a patch that fixes add.",
        fixtures={
            "toy_repo": {
                "repo_id": "patch-add",
                "files": {
                    "app/__init__.py": "",
                    "app/math_utils.py": "def add(a, b):\n    return a - b\n",
                },
                "agent_patch": (
                    "--- a/app/math_utils.py\n"
                    "+++ b/app/math_utils.py\n"
                    "@@ -1,2 +1,2 @@\n"
                    " def add(a, b):\n"
                    "-    return a - b\n"
                    "+    return a + b\n"
                ),
                "patch_allowed_paths": ["app/"],
                "hidden_judge": [
                    {
                        "kind": "python_inline_tests",
                        "code": (
                            "from app.math_utils import add\n"
                            "assert add(2, 3) == 5\n"
                        ),
                    }
                ],
            }
        },
        expected_artifacts=["app/math_utils.py"],
        allowed_paths=["submission/model.patch"],
        scoring=[
            ScoringRule(
                name="hidden_judge_passed",
                kind="metadata_flag",
                expected="judge_passed",
            )
        ],
    )

    report = EvaluationOrchestrator().run(DatasetSpec(dataset_id="patch-ok", tasks=[task]))

    assert report.aggregate_score == 1.0
    assert report.run_records[0].metadata["judge_details"][0]["kind"] == "patch_apply"
    assert report.task_results[0].artifacts == {
        "submission/model.patch": task.fixtures["toy_repo"]["agent_patch"]
    }


def test_patch_submission_rejects_disallowed_paths() -> None:
    task = TaskSpec(
        task_id="patch.python.disallowed",
        title="Patch disallowed file",
        category=TaskCategory.CODE_REPAIR,
        prompt="Submit a patch.",
        fixtures={
            "toy_repo": {
                "repo_id": "patch-disallowed",
                "files": {
                    "app/__init__.py": "",
                    "app/math_utils.py": "def add(a, b):\n    return a - b\n",
                    "hidden_tests/test_secret.py": "SECRET = False\n",
                },
                "agent_patch": (
                    "--- a/hidden_tests/test_secret.py\n"
                    "+++ b/hidden_tests/test_secret.py\n"
                    "@@ -1 +1 @@\n"
                    "-SECRET = False\n"
                    "+SECRET = True\n"
                ),
                "patch_allowed_paths": ["app/"],
                "hidden_judge": [
                    {
                        "kind": "python_inline_tests",
                        "code": (
                            "from app.math_utils import add\n"
                            "assert add(2, 3) == 5\n"
                        ),
                    }
                ],
            }
        },
        expected_artifacts=["app/math_utils.py"],
        allowed_paths=["submission/model.patch"],
        scoring=[
            ScoringRule(
                name="hidden_judge_passed",
                kind="metadata_flag",
                expected="judge_passed",
            )
        ],
    )

    report = EvaluationOrchestrator().run(DatasetSpec(dataset_id="patch-bad", tasks=[task]))

    assert report.task_results[0].status == "failed"
    assert report.run_records[0].failure_type == "patch_apply_error"


def test_patch_submission_parallel_runner_records_split_timings() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = DatasetSpec.model_validate_json(
        (root / "examples/patch_benchmark.json").read_text(encoding="utf-8")
    )

    report = EvaluationOrchestrator().run(dataset, workers=2, task_timeout_sec=10)

    assert report.aggregate_score == 1.0
    assert report.metadata["workers"] == 2
    assert len(report.run_records) == len(dataset.tasks)
    assert {record.pass_fail for record in report.run_records} == {"pass"}
    assert any(record.metrics.patch_apply_time_sec > 0 for record in report.run_records)
    assert any(record.metrics.hidden_test_time_sec > 0 for record in report.run_records)


def test_local_git_pytest_fixture_runs_hidden_tests() -> None:
    dataset = _load_git_pytest_dataset()
    small_dataset = dataset.model_copy(update={"tasks": dataset.tasks[:8]})

    report = EvaluationOrchestrator().run(small_dataset, workers=4, task_timeout_sec=20)

    assert report.aggregate_score == 1.0
    assert len(report.run_records) == 8
    assert {record.pass_fail for record in report.run_records} == {"pass"}
    assert all(record.metadata["checkout_strategy"] == "worktree" for record in report.run_records)
    assert any(record.metrics.repo_checkout_time_sec > 0 for record in report.run_records)
    assert any(record.metrics.git_clone_time_sec > 0 for record in report.run_records)
    assert any(record.metrics.git_checkout_time_sec > 0 for record in report.run_records)
    assert any(record.metrics.pytest_execution_time_sec > 0 for record in report.run_records)
    assert any(
        record.metrics.hidden_test_file_write_time_sec > 0 for record in report.run_records
    )
    assert any(
        detail["kind"] == "pytest"
        for record in report.run_records
        for detail in record.metadata["judge_details"]
    )


def test_local_git_pytest_fixture_supports_clone_checkout_strategy() -> None:
    dataset = _load_git_pytest_dataset()
    small_dataset = dataset.model_copy(update={"tasks": dataset.tasks[:8]})

    report = EvaluationOrchestrator(checkout_strategy="clone").run(
        small_dataset,
        workers=4,
        task_timeout_sec=20,
    )

    assert report.aggregate_score == 1.0
    assert {record.pass_fail for record in report.run_records} == {"pass"}
    assert all(record.metadata["checkout_strategy"] == "clone" for record in report.run_records)
    assert any(record.metrics.git_clone_time_sec > 0 for record in report.run_records)


def test_local_git_pytest_fixture_supports_snapshot_checkout_strategy() -> None:
    dataset = _load_git_pytest_dataset()
    small_dataset = dataset.model_copy(update={"tasks": dataset.tasks[:4]})

    report = EvaluationOrchestrator(checkout_strategy="copytree").run(
        small_dataset,
        workers=2,
        task_timeout_sec=20,
    )

    assert report.aggregate_score == 1.0
    assert {record.pass_fail for record in report.run_records} == {"pass"}
    assert all(record.metadata["checkout_strategy"] == "copytree" for record in report.run_records)
    assert any(
        record.metrics.snapshot_materialize_time_sec > 0 for record in report.run_records
    )
