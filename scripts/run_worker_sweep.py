#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SE-Bench worker-count sweep.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("examples/git_pytest_benchmark.json"),
        help="Benchmark manifest.",
    )
    parser.add_argument(
        "--workers",
        nargs="+",
        type=int,
        default=[64, 128, 256, 1024],
        help="Worker counts to sweep.",
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--task-counts",
        nargs="+",
        type=int,
        default=None,
        help="Optional max task counts to test against a larger manifest.",
    )
    parser.add_argument("--task-timeout-sec", type=float, default=30.0)
    parser.add_argument("--mlx-size", type=int, default=256)
    parser.add_argument("--mlx-iters", type=int, default=1)
    parser.add_argument(
        "--checkout-strategies",
        nargs="+",
        choices=["clone", "worktree", "copytree", "tar"],
        default=["clone"],
        help="Local git checkout strategies to compare.",
    )
    parser.add_argument(
        "--pytest-diagnostics",
        action="store_true",
        help="Run extra python/pytest startup and collection probes.",
    )
    parser.add_argument(
        "--adaptive-worker-cap",
        action="store_true",
        help=(
            "Compatibility flag. Child benchmark caps workers by default unless "
            "--pressure-test is provided."
        ),
    )
    parser.add_argument(
        "--pressure-test",
        action="store_true",
        help="Pass through to child benchmark to disable the default local worker cap.",
    )
    parser.add_argument(
        "--scheduler-policy",
        choices=["fixed", "adaptive"],
        default="fixed",
        help="Pass through child benchmark worker scheduling policy.",
    )
    parser.add_argument(
        "--task-distribution",
        choices=["load", "repo-shard", "repo-shard-worksteal"],
        default="repo-shard-worksteal",
        help="Pass through child benchmark task distribution policy.",
    )
    parser.add_argument(
        "--cache-policy",
        choices=["auto", "process", "shared"],
        default="auto",
        help="Pass through child benchmark checkout cache policy.",
    )
    parser.add_argument(
        "--shared-cache-root",
        default="artifacts/cache/judge_shared",
        help="Pass through child benchmark shared cache root.",
    )
    parser.add_argument(
        "--soft-task-timeout",
        action="store_true",
        help="Pass through child benchmark legacy soft task timeout mode.",
    )
    parser.add_argument(
        "--pytest-timeout-sec",
        type=float,
        default=None,
        help="Override hidden pytest subprocess timeout in child benchmark runs.",
    )
    parser.add_argument(
        "--pytest-plugin-policy",
        choices=["auto", "disabled", "enabled"],
        default="auto",
        help=(
            "Policy for hidden pytest plugin autoload. auto scans repo/test config, "
            "disabled always sets PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, enabled always "
            "allows plugin autoload."
        ),
    )
    plugin_group = parser.add_mutually_exclusive_group()
    plugin_group.add_argument(
        "--disable-pytest-plugin-autoload",
        dest="pytest_plugin_policy",
        action="store_const",
        const="disabled",
        help=(
            "Pass through PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 for hidden pytest subprocesses. "
            "Compatibility alias for --pytest-plugin-policy disabled."
        ),
    )
    plugin_group.add_argument(
        "--enable-pytest-plugin-autoload",
        dest="pytest_plugin_policy",
        action="store_const",
        const="enabled",
        help="Allow hidden pytest subprocesses to autoload third-party plugins.",
    )
    parser.add_argument(
        "--reuse-orchestrator-across-repeats",
        action="store_true",
        help="Pass through to child benchmark to reuse sandbox caches across repeats.",
    )
    parser.add_argument(
        "--command-timeout-sec",
        type=float,
        default=900.0,
        help="Timeout for each child benchmark command.",
    )
    parser.add_argument("--prefix", default="git_pytest_round6")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/reports/git_pytest_round6_worker_sweep.json"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("artifacts/reports/git_pytest_round6_worker_sweep.md"),
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def output_suffix(
    args: argparse.Namespace,
    strategy: str,
    task_count: int | None,
    workers: int,
) -> str:
    if args.checkout_strategies == ["clone"] and task_count is None:
        return f"workers_{workers}"
    task_label = "all" if task_count is None else str(task_count)
    return f"{strategy}_tasks_{task_label}_workers_{workers}"


def run_one(
    args: argparse.Namespace,
    strategy: str,
    task_count: int | None,
    workers: int,
) -> dict[str, Any]:
    suffix = output_suffix(args, strategy, task_count, workers)
    report_path = Path("artifacts/reports") / f"{args.prefix}_{suffix}_report.json"
    summary_path = Path("artifacts/reports") / f"{args.prefix}_{suffix}_summary.md"
    command = [
        sys.executable,
        "scripts/run_mac_mlx_benchmark.py",
        "--manifest",
        str(args.manifest),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--workers",
        str(workers),
        "--checkout-strategy",
        strategy,
        "--task-timeout-sec",
        str(args.task_timeout_sec),
        "--scheduler-policy",
        args.scheduler_policy,
        "--task-distribution",
        args.task_distribution,
        "--cache-policy",
        args.cache_policy,
        "--shared-cache-root",
        args.shared_cache_root,
        "--mlx-size",
        str(args.mlx_size),
        "--mlx-iters",
        str(args.mlx_iters),
        "--skip-torch-mps",
        "--out",
        str(report_path),
        "--summary-out",
        str(summary_path),
    ]
    if task_count is not None:
        command.extend(["--max-tasks", str(task_count)])
    if args.pytest_diagnostics:
        command.append("--pytest-diagnostics")
    if args.adaptive_worker_cap:
        command.append("--adaptive-worker-cap")
    if args.pressure_test:
        command.append("--pressure-test")
    if args.soft_task_timeout:
        command.append("--soft-task-timeout")
    if args.pytest_timeout_sec is not None:
        command.extend(["--pytest-timeout-sec", str(args.pytest_timeout_sec)])
    command.extend(["--pytest-plugin-policy", args.pytest_plugin_policy])
    if args.reuse_orchestrator_across_repeats:
        command.append("--reuse-orchestrator-across-repeats")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=args.command_timeout_sec,
        check=False,
    )
    result: dict[str, Any] = {
        "workers": workers,
        "checkout_strategy": strategy,
        "task_count": task_count,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "report_path": str(report_path),
        "summary_path": str(summary_path),
    }
    if completed.returncode == 0:
        payload = load(ROOT / report_path)
        result["metrics"] = payload["metrics"]
        result["config"] = payload["config"]
        result["dataset"] = payload["dataset"]
    return result


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Worker Sweep",
        "",
        f"- Manifest: `{payload['manifest']}`",
        f"- Workers: `{payload['workers']}`",
        f"- Checkout strategies: `{payload['checkout_strategies']}`",
        f"- Task counts: `{payload['task_counts']}`",
        f"- Warmup/repeat: `{payload['warmup']}` / `{payload['repeat']}`",
        f"- Pytest diagnostics: `{payload['pytest_diagnostics']}`",
        f"- Pressure test: `{payload['pressure_test']}`",
        f"- Scheduler policy: `{payload['scheduler_policy']}`",
        f"- Task distribution: `{payload['task_distribution']}`",
        f"- Cache policy: `{payload['cache_policy']}`",
        f"- Hard task timeout: `{not payload['soft_task_timeout']}`",
        f"- Pytest timeout override: `{payload['pytest_timeout_sec']}`",
        f"- Pytest plugin policy: `{payload['pytest_plugin_policy']}`",
        (
            "- Disable pytest plugin autoload: "
            f"`{payload['disable_pytest_plugin_autoload']}`"
        ),
        (
            "- Reuse orchestrator across repeats: "
            f"`{payload['reuse_orchestrator_across_repeats']}`"
        ),
        "",
        "| Strategy | Tasks | Requested Workers | Effective Workers | Return | Pass Rate | "
        "Hidden Timeouts | Task Timeouts | Run Total (s) | "
        "Throughput tasks/s | Wall p50 (s) | Checkout Avg (s) | "
        "Git Clone Avg (s) | Lock Wait Avg (s) | Materialize Avg (s) | Pytest Exec Avg (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        metrics = result.get("metrics", {})
        config = result.get("config", {})
        lines.append(
            "| "
            f"{result['checkout_strategy']} | "
            f"{result.get('task_count') or 'all'} | "
            f"{result['workers']} | "
            f"{config.get('workers', result['workers'])} | "
            f"{result['returncode']} | "
            f"{metrics.get('pass_rate', 0):.6f} | "
            f"{metrics.get('hidden_judge_timeout_count', 0)} | "
            f"{metrics.get('task_timeout_count', 0)} | "
            f"{metrics.get('run_duration_total_sec', 0):.6f} | "
            f"{metrics.get('throughput_tasks_per_sec', 0):.6f} | "
            f"{metrics.get('wall_time_p50_sec', 0):.6f} | "
            f"{metrics.get('repo_checkout_time_sec_avg', 0):.6f} | "
            f"{metrics.get('git_clone_time_sec_avg', 0):.6f} | "
            f"{metrics.get('cache_lock_wait_time_sec_avg', 0):.6f} | "
            f"{metrics.get('snapshot_materialize_time_sec_avg', 0):.6f} | "
            f"{metrics.get('pytest_execution_time_sec_avg', 0):.6f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    task_counts = args.task_counts or [None]
    results = [
        run_one(args, strategy, task_count, workers)
        for strategy in args.checkout_strategies
        for task_count in task_counts
        for workers in args.workers
    ]
    payload = {
        "manifest": str(args.manifest),
        "workers": args.workers,
        "checkout_strategies": args.checkout_strategies,
        "task_counts": task_counts,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "task_timeout_sec": args.task_timeout_sec,
        "pytest_timeout_sec": args.pytest_timeout_sec,
        "pytest_plugin_policy": args.pytest_plugin_policy,
        "disable_pytest_plugin_autoload": args.pytest_plugin_policy != "enabled",
        "pytest_diagnostics": args.pytest_diagnostics,
        "adaptive_worker_cap": args.adaptive_worker_cap,
        "scheduler_policy": args.scheduler_policy,
        "task_distribution": args.task_distribution,
        "cache_policy": args.cache_policy,
        "shared_cache_root": args.shared_cache_root,
        "pressure_test": args.pressure_test,
        "soft_task_timeout": args.soft_task_timeout,
        "reuse_orchestrator_across_repeats": args.reuse_orchestrator_across_repeats,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(args.summary_out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
