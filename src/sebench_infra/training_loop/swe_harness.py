from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sebench_infra.benchmark.schemas import (
    BenchmarkEnvironment,
    CacheHitFlags,
    EvaluationReport,
    RunRecord,
    TaskMetrics,
    TaskRunResult,
)
from sebench_infra.orchestrator.records import new_run_id


def write_swe_predictions(
    predictions: dict[str, str],
    path: Path,
    *,
    model_name_or_path: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": patch,
        }
        for instance_id, patch in sorted(predictions.items())
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


class SWEHarnessRunner:
    """Optional adapter around the official SWE-bench evaluation harness."""

    def __init__(self, command: list[str] | None = None, timeout_sec: float = 3600.0) -> None:
        self.command = command
        self.timeout_sec = timeout_sec

    def run(
        self,
        *,
        predictions_path: Path,
        dataset_name: str,
        split: str,
        output_dir: Path,
        results_path: Path | None = None,
    ) -> EvaluationReport:
        if results_path is not None and results_path.exists():
            return report_from_swe_results(
                dataset_id=dataset_name,
                results=_load_swe_results(results_path),
                metadata={
                    "predictions_path": str(predictions_path),
                    "results_path": str(results_path),
                    "split": split,
                },
            )
        if not self.command:
            return _skipped_report(
                dataset_id=dataset_name,
                predictions_path=predictions_path,
                reason="swebench_harness_command_not_configured",
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            *self.command,
            "--predictions_path",
            str(predictions_path),
            "--dataset_name",
            dataset_name,
            "--split",
            split,
            "--run_id",
            datetime.now(UTC).strftime("%Y%m%d%H%M%S"),
        ]
        completed = subprocess.run(
            command,
            cwd=output_dir,
            text=True,
            capture_output=True,
            timeout=self.timeout_sec,
            check=False,
        )
        if completed.returncode != 0:
            return _error_report(
                dataset_id=dataset_name,
                failure_type="swebench_harness_error",
                metadata={
                    "command": command,
                    "stdout": completed.stdout[-4000:],
                    "stderr": completed.stderr[-4000:],
                    "predictions_path": str(predictions_path),
                },
            )
        resolved_results_path = results_path or _find_swe_results_file(output_dir)
        if resolved_results_path is None:
            return _skipped_report(
                dataset_id=dataset_name,
                predictions_path=predictions_path,
                reason="swebench_harness_completed_results_not_found",
                metadata={"command": command, "stdout": completed.stdout[-4000:]},
            )
        return report_from_swe_results(
            dataset_id=dataset_name,
            results=_load_swe_results(resolved_results_path),
            metadata={
                "runner": "SWEHarnessRunner",
                "command": command,
                "stdout": completed.stdout[-4000:],
                "predictions_path": str(predictions_path),
                "results_path": str(resolved_results_path),
                "split": split,
            },
        )


def report_from_swe_results(
    *,
    dataset_id: str,
    results: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> EvaluationReport:
    run_id = new_run_id()
    task_results: list[TaskRunResult] = []
    run_records: list[RunRecord] = []
    for instance_id, value in sorted(results.items()):
        resolved = _resolved(value)
        status = "passed" if resolved else "failed"
        failure_type = None if resolved else "swe_unresolved"
        task_results.append(
            TaskRunResult(
                task_id=instance_id,
                status=status,
                score=1.0 if resolved else 0.0,
                details={"swe_result": value},
                artifacts={},
                patch_hash=instance_id,
                failure_type=failure_type,
            )
        )
        run_records.append(
            _record(
                run_id=run_id,
                task_id=instance_id,
                pass_fail="pass" if resolved else "fail",
                failure_type=failure_type,
                metadata={"swe_result": value},
            )
        )
    aggregate = (
        sum(result.score for result in task_results) / len(task_results)
        if task_results
        else 0.0
    )
    return EvaluationReport(
        run_id=run_id,
        dataset_id=dataset_id,
        task_results=task_results,
        aggregate_score=round(aggregate, 6),
        reward_signal=round(aggregate, 6),
        regression_passed=aggregate >= 0.999,
        metadata={"runner": "SWEHarnessRunner", **(metadata or {})},
        run_records=run_records,
    )


def _load_swe_results(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parsed = _coerce_swe_results(payload)
    if not parsed:
        raise ValueError(f"could not parse SWE-bench results from {path}")
    return parsed


def _find_swe_results_file(output_dir: Path) -> Path | None:
    candidates = sorted(
        (path for path in output_dir.rglob("*.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if _coerce_swe_results(payload):
            return path
    return None


def _coerce_swe_results(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        if isinstance(payload.get("results"), dict):
            return dict(payload["results"])
        if isinstance(payload.get("instances"), list):
            return _coerce_swe_results(payload["instances"])
        if "resolved_ids" in payload or "unresolved_ids" in payload:
            results: dict[str, Any] = {}
            for instance_id in payload.get("resolved_ids", []):
                results[str(instance_id)] = {"resolved": True}
            for instance_id in payload.get("unresolved_ids", []):
                results[str(instance_id)] = {"resolved": False}
            return results
        if "resolved" in payload and isinstance(payload.get("resolved"), list):
            return {str(instance_id): {"resolved": True} for instance_id in payload["resolved"]}
        if "instance_id" in payload:
            return {str(payload["instance_id"]): payload}
        if payload and all(
            isinstance(key, str) and isinstance(value, bool | dict)
            for key, value in payload.items()
        ):
            return dict(payload)
    if isinstance(payload, list):
        results = {}
        for item in payload:
            if isinstance(item, dict) and "instance_id" in item:
                results[str(item["instance_id"])] = item
        return results
    return {}


def _resolved(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        for key in ("resolved", "success", "passed"):
            if key in value:
                return bool(value[key])
    return False


def _skipped_report(
    *,
    dataset_id: str,
    predictions_path: Path,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> EvaluationReport:
    return _error_report(
        dataset_id=dataset_id,
        failure_type=reason,
        metadata={"predictions_path": str(predictions_path), **(metadata or {})},
    )


def _error_report(
    *,
    dataset_id: str,
    failure_type: str,
    metadata: dict[str, Any],
) -> EvaluationReport:
    run_id = new_run_id()
    result = TaskRunResult(
        task_id="swebench_harness",
        status="error",
        score=0.0,
        details=metadata,
        artifacts={},
        patch_hash="swebench-harness",
        failure_type=failure_type,
    )
    return EvaluationReport(
        run_id=run_id,
        dataset_id=dataset_id,
        task_results=[result],
        aggregate_score=0.0,
        reward_signal=0.0,
        regression_passed=False,
        metadata={"runner": "SWEHarnessRunner", "failure_type": failure_type, **metadata},
        run_records=[
            _record(
                run_id=run_id,
                task_id="swebench_harness",
                pass_fail="error",
                failure_type=failure_type,
                metadata=metadata,
            )
        ],
    )


def _record(
    *,
    run_id: str,
    task_id: str,
    pass_fail: str,
    failure_type: str | None,
    metadata: dict[str, Any],
) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        task_id=task_id,
        patch_hash=task_id,
        pass_fail=pass_fail,  # type: ignore[arg-type]
        failure_type=failure_type,
        metrics=TaskMetrics(
            wall_time_sec=0.0,
            judge_time_sec=0.0,
            agent_step_latency_sec=0.0,
            cache_hit_flags=CacheHitFlags(),
        ),
        benchmark_environment=BenchmarkEnvironment(benchmark_backend="swebench_harness"),
        metadata=metadata,
    )
