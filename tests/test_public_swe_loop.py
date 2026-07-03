import json
from pathlib import Path

from sebench_infra.training_loop import (
    ExternalBenchmarkSource,
    ModelPatchAgent,
    StaticPatchClient,
    SWEHarnessRunner,
    SWEIssueInstance,
    apply_unified_diff_to_files,
    swe_instances_to_dataset_version,
    write_swe_predictions,
)
from sebench_infra.training_loop.models import TrainingTaskKind


def test_swe_issue_instance_parses_hf_row_and_task_does_not_leak_gold_patch() -> None:
    source = ExternalBenchmarkSource(
        name="swe-bench",
        dataset="SWE-bench/SWE-bench_Lite",
        split="test",
        url="https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite",
        license="public_dataset_license_check_required",
    )
    row = {
        "instance_id": "demo__repo-123",
        "repo": "demo/repo",
        "base_commit": "abc123",
        "problem_statement": "Fix add so it returns the sum.",
        "patch": "diff --git a/calculator.py b/calculator.py\nSECRET_GOLD_PATCH\n",
        "test_patch": "diff --git a/tests/test_calc.py b/tests/test_calc.py\n",
        "FAIL_TO_PASS": '["tests/test_calc.py::test_add"]',
        "PASS_TO_PASS": ["tests/test_calc.py::test_existing"],
    }

    instance = SWEIssueInstance.from_hf_row(row, source=source)
    task = instance.to_task_spec()
    task_json = task.model_dump_json()

    assert instance.fail_to_pass == ["tests/test_calc.py::test_add"]
    assert "SECRET_GOLD_PATCH" not in task.prompt
    assert "SECRET_GOLD_PATCH" not in task_json
    assert task.fixtures["public_swe"]["has_gold_patch"] is True
    assert task.metadata["gold_patch_available_offline"] is True


def test_swe_issue_to_patch_examples_export_as_training_task_kind() -> None:
    source = ExternalBenchmarkSource(name="swe-smith", dataset="SWE-bench/SWE-smith")
    instance = SWEIssueInstance.from_hf_row(
        {
            "instance_id": "demo__repo-456",
            "repo": "demo/repo",
            "base_commit": "abc123",
            "problem_statement": "Fix subtraction.",
            "patch": "diff --git a/calculator.py b/calculator.py\n@@ -1 +1 @@\n-a\n+b\n",
        },
        source=source,
    )

    dataset = swe_instances_to_dataset_version([instance], seed=13)

    assert len(dataset.examples) == 1
    assert dataset.examples[0].training_task == TrainingTaskKind.ISSUE_TO_PATCH
    assert "Fix subtraction." in dataset.examples[0].input
    assert dataset.metadata["training_task"] == TrainingTaskKind.ISSUE_TO_PATCH.value


def test_model_patch_agent_generates_and_applies_unified_diff() -> None:
    patch = (
        "diff --git a/calculator.py b/calculator.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/calculator.py\n"
        "+++ b/calculator.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def add(a, b):\n"
        "-    return a - b\n"
        "+    return a + b\n"
    )
    source = ExternalBenchmarkSource(name="local", dataset="fixture")
    task = SWEIssueInstance.from_hf_row(
        {
            "instance_id": "demo__repo-789",
            "repo": "demo/repo",
            "base_commit": "abc123",
            "problem_statement": "Fix add.",
        },
        source=source,
    ).to_task_spec()

    generated = ModelPatchAgent(StaticPatchClient(patch)).generate_patch(task)
    ok, files, error = apply_unified_diff_to_files(
        {"calculator.py": "def add(a, b):\n    return a - b\n"},
        generated,
    )

    assert ok, error
    assert files["calculator.py"] == "def add(a, b):\n    return a + b\n"


def test_write_swe_predictions_uses_official_jsonl_shape(tmp_path: Path) -> None:
    out = tmp_path / "predictions.jsonl"

    write_swe_predictions(
        {"demo__repo-1": "diff --git a/a.py b/a.py\n@@ -1 +1 @@\n-a\n+b\n"},
        out,
        model_name_or_path="student-lora",
    )

    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["instance_id"] == "demo__repo-1"
    assert row["model_name_or_path"] == "student-lora"
    assert "model_patch" in row


def test_swe_harness_runner_parses_explicit_results_path(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    write_swe_predictions(
        {
            "demo__repo-1": "diff --git a/a.py b/a.py\n@@ -1 +1 @@\n-a\n+b\n",
            "demo__repo-2": "",
        },
        predictions,
        model_name_or_path="student-lora",
    )
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "resolved_ids": ["demo__repo-1"],
                "unresolved_ids": ["demo__repo-2"],
            }
        ),
        encoding="utf-8",
    )

    report = SWEHarnessRunner().run(
        predictions_path=predictions,
        dataset_name="SWE-bench/SWE-bench_Lite",
        split="test",
        output_dir=tmp_path / "harness",
        results_path=results,
    )

    assert report.aggregate_score == 0.5
    assert [result.status for result in report.task_results] == ["passed", "failed"]
    assert report.metadata["results_path"] == str(results)
