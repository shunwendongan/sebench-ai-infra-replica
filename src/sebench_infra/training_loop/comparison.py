from __future__ import annotations

from datetime import UTC, datetime

from sebench_infra.benchmark.schemas import EvaluationReport
from sebench_infra.training_loop.models import (
    ModelComparisonReport,
    ModelMetrics,
    ModelRole,
    ModelRunConfig,
    ModelRunResult,
)
from sebench_infra.training_loop.validation import stable_hash


def build_model_run_result(
    config: ModelRunConfig,
    report: EvaluationReport,
    *,
    report_path: str | None = None,
    valid_task_rate: float = 1.0,
    schema_repair_count: int = 0,
    cost_estimate_usd: float = 0.0,
) -> ModelRunResult:
    task_count = len(report.task_results)
    passed = sum(1 for result in report.task_results if result.status == "passed")
    judge_errors = sum(
        1
        for record in report.run_records
        if record.failure_type and "judge" in record.failure_type
    )
    latency = (
        sum(record.metrics.wall_time_sec for record in report.run_records) / len(report.run_records)
        if report.run_records
        else 0.0
    )
    metrics = ModelMetrics(
        task_count=task_count,
        valid_task_rate=valid_task_rate,
        pass_rate=round(passed / task_count, 6) if task_count else 0.0,
        aggregate_score=report.aggregate_score,
        judge_error_rate=round(judge_errors / task_count, 6) if task_count else 0.0,
        latency_sec_avg=round(latency, 6),
        cost_estimate_usd=round(cost_estimate_usd, 6),
        schema_repair_count=schema_repair_count,
    )
    return ModelRunResult(
        config=config,
        metrics=metrics,
        evaluation_report=report,
        report_path=report_path,
        metadata={"source": "evaluation_report"},
    )


def compare_model_runs(
    runs: list[ModelRunResult],
    *,
    dataset_id: str,
    metadata: dict | None = None,
) -> ModelComparisonReport:
    best_role: ModelRole | None = None
    if runs:
        best = max(
            runs,
            key=lambda run: (
                run.metrics.pass_rate,
                run.metrics.aggregate_score,
                -run.metrics.cost_estimate_usd,
                -run.metrics.latency_sec_avg,
            ),
        )
        best_role = best.config.role

    created_at = datetime.now(UTC).isoformat()
    digest = stable_hash(
        "|".join(
            [
                dataset_id,
                created_at,
                *[
                    f"{run.config.role}:{run.config.model}:{run.metrics.pass_rate}"
                    for run in runs
                ],
            ]
        ),
        length=12,
    )
    return ModelComparisonReport(
        comparison_id=f"model-comparison-{digest}",
        dataset_id=dataset_id,
        created_at=created_at,
        runs=runs,
        best_role_by_pass_rate=best_role,
        metadata=metadata or {},
    )


def comparison_to_markdown(report: ModelComparisonReport) -> str:
    lines = [
        "# Student Loop Model Comparison",
        "",
        f"- Dataset: `{report.dataset_id}`",
        f"- Comparison: `{report.comparison_id}`",
        f"- Created at: `{report.created_at}`",
        f"- Best role by pass rate: `{report.best_role_by_pass_rate or 'n/a'}`",
        "",
        (
            "| Role | Provider | Model | Tasks | Valid Task Rate | Pass Rate | Aggregate | "
            "Judge Error | Latency Avg | Cost Est. | Repairs |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report.runs:
        metrics = run.metrics
        lines.append(
            "| "
            f"{run.config.role.value} | "
            f"{run.config.provider.value} | "
            f"`{run.config.model}` | "
            f"{metrics.task_count} | "
            f"{metrics.valid_task_rate:.6f} | "
            f"{metrics.pass_rate:.6f} | "
            f"{metrics.aggregate_score:.6f} | "
            f"{metrics.judge_error_rate:.6f} | "
            f"{metrics.latency_sec_avg:.6f}s | "
            f"${metrics.cost_estimate_usd:.6f} | "
            f"{metrics.schema_repair_count} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Metrics are generated from local/public benchmark reports only.",
            "- GPT-family teacher models are treated as API providers, not trainable weights.",
            "- Student checkpoints should be evaluated through an OpenAI-compatible endpoint.",
        ]
    )
    return "\n".join(lines) + "\n"
