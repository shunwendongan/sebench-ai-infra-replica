#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

NUMERIC_KEYS = [
    "pass_rate",
    "wall_time_p50_sec",
    "wall_time_p95_sec",
    "judge_time_sec_avg",
    "patch_apply_time_sec_avg",
    "hidden_test_time_sec_avg",
    "hidden_test_file_write_time_sec_avg",
    "python_subprocess_startup_time_sec_avg",
    "pytest_process_startup_time_sec_avg",
    "pytest_collection_time_sec_avg",
    "pytest_execution_time_sec_avg",
    "pytest_total_time_sec_avg",
    "judge_cache_lookup_time_sec_avg",
    "repo_checkout_time_sec_avg",
    "git_clone_time_sec_avg",
    "git_checkout_time_sec_avg",
    "snapshot_materialize_time_sec_avg",
    "tempdir_create_time_sec_avg",
    "judge_workspace_cleanup_time_sec_avg",
    "agent_step_latency_p50_sec",
    "agent_step_latency_p95_sec",
    "run_duration_p50_sec",
    "run_duration_p95_sec",
    "run_duration_total_sec",
    "throughput_tasks_per_sec",
    "repo_snapshot_cache_hit_rate",
    "repo_map_cache_hit_rate",
    "llm_prompt_cache_hit_rate",
    "judge_verdict_cache_hit_rate",
    "memory_peak_mb",
    "hidden_judge_timeout_count",
    "task_timeout_count",
    "hard_timeout_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two SE-Bench benchmark reports.")
    parser.add_argument("--before", type=Path, required=True, help="Baseline JSON report.")
    parser.add_argument("--after", type=Path, required=True, help="Optimized JSON report.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/reports/mac_mlx_optimization_delta.md"),
        help="Markdown delta report path.",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def trend(key: str, before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "n/a"
    delta = after - before
    higher_is_better = (
        key.endswith("hit_rate")
        or key == "pass_rate"
        or key == "throughput_tasks_per_sec"
    )
    if abs(delta) < 1e-12:
        return "flat"
    improved = delta > 0 if higher_is_better else delta < 0
    return "improved" if improved else "regressed"


def render(before: dict[str, Any], after: dict[str, Any]) -> str:
    before_metrics = before.get("metrics", {})
    after_metrics = after.get("metrics", {})
    before_backend = before.get("benchmark_backend", "unknown")
    after_backend = after.get("benchmark_backend", "unknown")
    before_dataset = before.get("dataset", {}).get("dataset_id")
    after_dataset = after.get("dataset", {}).get("dataset_id")
    lines = [
        "# Mac MLX Optimization Delta",
        "",
        (
            "This comparison is scoped to local Apple Silicon/MLX engineering validation. "
            "It must not be used as CUDA/GPU-cluster throughput evidence."
        ),
        "",
        "## Run Context",
        "",
        f"- Before: `{before.get('created_at', 'unknown')}` / `{before_backend}`",
        f"- After: `{after.get('created_at', 'unknown')}` / `{after_backend}`",
        f"- Dataset before/after: `{before_dataset}` / `{after_dataset}`",
        "",
        "## Metric Delta",
        "",
        "| Metric | Before | After | Delta | Trend |",
        "|---|---:|---:|---:|---|",
    ]
    for key in NUMERIC_KEYS:
        before_value = before_metrics.get(key)
        after_value = after_metrics.get(key)
        delta = (
            after_value - before_value
            if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float))
            else None
        )
        before_text = fmt(before_value)
        after_text = fmt(after_value)
        delta_text = fmt(delta)
        trend_text = trend(key, before_value, after_value)
        lines.append(
            f"| `{key}` | {before_text} | {after_text} | {delta_text} | {trend_text} |"
        )

    lines.extend(
        [
            "",
            "## Failure Type Delta",
            "",
            "| Failure Type | Before | After |",
            "|---|---:|---:|",
        ]
    )
    before_failures = before_metrics.get("failure_type_counts", {})
    after_failures = after_metrics.get("failure_type_counts", {})
    for failure_type in sorted(set(before_failures) | set(after_failures)):
        before_count = before_failures.get(failure_type, 0)
        after_count = after_failures.get(failure_type, 0)
        lines.append(
            f"| `{failure_type}` | {before_count} | {after_count} |"
        )

    lines.extend(
        [
            "",
            "## Engineering Interpretation",
            "",
            (
                "- If latency improved while pass rate stayed flat, keep the optimization "
                "and inspect cost per pass when real LLM calls are introduced."
            ),
            (
                "- If cache hit rates improved but pass rate regressed, audit context recall "
                "and Judge correctness before accepting the change."
            ),
            (
                "- If failure types shift from system errors to `wrong_edit`, the "
                "infrastructure is likely healthier and the next bottleneck is agent quality."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    before = load(args.before)
    after = load(args.after)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(before, after), encoding="utf-8")
    print(f"wrote comparison report: {args.out}")


if __name__ == "__main__":
    main()
