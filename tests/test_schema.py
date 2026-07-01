import pytest
from pydantic import ValidationError

from sebench_infra.benchmark.schemas import (
    CacheHitFlags,
    RunRecord,
    TaskCategory,
    TaskMetrics,
    TaskSpec,
)


def test_task_spec_rejects_path_escape() -> None:
    with pytest.raises(ValidationError):
        TaskSpec(
            task_id="bad",
            title="bad",
            category=TaskCategory.BENCHMARK_AUTHORING,
            prompt="x",
            allowed_paths=["../secret"],
        )


def test_task_spec_adds_default_scoring() -> None:
    task = TaskSpec(
        task_id="ok",
        title="ok",
        category=TaskCategory.BENCHMARK_AUTHORING,
        prompt="x",
    )

    assert task.scoring
    assert task.scoring[0].path == "submission/answer.txt"


def test_run_record_requires_metrics_and_failure_type_for_failures() -> None:
    record = RunRecord(
        run_id="run-001",
        task_id="toy.fail",
        patch_hash="abc",
        pass_fail="fail",
        failure_type="wrong_edit",
        metrics=TaskMetrics(
            wall_time_sec=0.1,
            judge_time_sec=0.02,
            agent_step_latency_sec=0.01,
            cache_hit_flags=CacheHitFlags(repo_snapshot=True),
        ),
    )

    assert record.metrics.cache_hit_flags.repo_snapshot is True
    assert record.failure_type == "wrong_edit"
    assert record.metrics.pytest_execution_time_sec == 0.0
    assert record.metrics.git_clone_time_sec == 0.0
