#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SE-Bench git checkout and hidden pytest diagnostics."
    )
    parser.add_argument(
        "--sweep",
        type=Path,
        required=True,
        help="Worker sweep JSON produced by scripts/run_worker_sweep.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Markdown analysis output path.",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(result: dict[str, Any], key: str) -> float:
    value = result.get("metrics", {}).get(key, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def result_key(result: dict[str, Any]) -> tuple[str, int, int]:
    task_count = result.get("task_count") or 0
    return (str(result.get("checkout_strategy")), int(task_count), int(result["workers"]))


def render(payload: dict[str, Any]) -> str:
    results = sorted(payload["results"], key=result_key)
    successful = [result for result in results if result.get("returncode") == 0]
    best = max(successful, key=lambda result: metric(result, "throughput_tasks_per_sec"))
    lines = [
        "# Git Checkout + Hidden Pytest Diagnostics",
        "",
        "This report is scoped to local Apple Silicon engineering validation. "
        "It is not CUDA/GPU-cluster throughput evidence.",
        "",
        "## Run Context",
        "",
        f"- Manifest: `{payload['manifest']}`",
        f"- Workers: `{payload['workers']}`",
        f"- Task counts: `{payload['task_counts']}`",
        f"- Checkout strategies: `{payload['checkout_strategies']}`",
        f"- Warmup/repeat: `{payload['warmup']}` / `{payload['repeat']}`",
        f"- Pytest diagnostics: `{payload['pytest_diagnostics']}`",
        "",
        "## Best Local Throughput",
        "",
        (
            f"- Strategy: `{best['checkout_strategy']}`; tasks: `{best.get('task_count')}`; "
            f"workers: `{best['workers']}`; throughput: "
            f"`{metric(best, 'throughput_tasks_per_sec'):.6f} tasks/s`."
        ),
        "",
        "## Metrics",
        "",
        "| Strategy | Tasks | Workers | Return | Pass | Throughput | Wall p95 | "
        "Checkout | Clone | Git checkout | Materialize | Pytest startup | "
        "Collection | Pytest exec | Cleanup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| "
            f"{result['checkout_strategy']} | "
            f"{result.get('task_count') or 'all'} | "
            f"{result['workers']} | "
            f"{result['returncode']} | "
            f"{metric(result, 'pass_rate'):.6f} | "
            f"{metric(result, 'throughput_tasks_per_sec'):.6f} | "
            f"{metric(result, 'wall_time_p95_sec'):.6f} | "
            f"{metric(result, 'repo_checkout_time_sec_avg'):.6f} | "
            f"{metric(result, 'git_clone_time_sec_avg'):.6f} | "
            f"{metric(result, 'git_checkout_time_sec_avg'):.6f} | "
            f"{metric(result, 'snapshot_materialize_time_sec_avg'):.6f} | "
            f"{metric(result, 'pytest_process_startup_time_sec_avg'):.6f} | "
            f"{metric(result, 'pytest_collection_time_sec_avg'):.6f} | "
            f"{metric(result, 'pytest_execution_time_sec_avg'):.6f} | "
            f"{metric(result, 'judge_workspace_cleanup_time_sec_avg'):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Checklist",
            "",
            "- If pytest execution dominates, optimize pytest invocation or batch tests by repo.",
            "- If clone dominates, prefer persistent checkout or snapshot materialization.",
            "- If materialization dominates, compare copytree and tar against persistent checkout.",
            "- If throughput drops as workers rise, treat the host as I/O or scheduler saturated.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    payload = load(args.sweep)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(payload), encoding="utf-8")
    print(f"wrote diagnostics report: {args.out}")


if __name__ == "__main__":
    main()
