#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sebench_infra.benchmark.schemas import DatasetSpec  # noqa: E402
from sebench_infra.orchestrator import EvaluationOrchestrator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local SE-Bench true-loop toy benchmark on Apple Silicon/MLX."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("examples/toy_benchmark.json"),
        help="Dataset manifest to evaluate.",
    )
    parser.add_argument(
        "--backend",
        default="apple_mlx_local",
        help="Benchmark backend label written to the report.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs not included.")
    parser.add_argument("--repeat", type=int, default=3, help="Measured repeated runs.")
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Limit measured dataset size without editing the manifest.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel local task workers.")
    parser.add_argument(
        "--adaptive-worker-cap",
        action="store_true",
        help=(
            "Compatibility flag. Worker cap is now enabled by default unless "
            "--pressure-test is provided."
        ),
    )
    parser.add_argument(
        "--pressure-test",
        action="store_true",
        help="Disable the default local worker cap for explicit saturation tests.",
    )
    parser.add_argument(
        "--scheduler-policy",
        choices=["fixed", "adaptive"],
        default="fixed",
        help=(
            "Worker scheduling policy. fixed applies the local cap once; adaptive "
            "can reduce workers between repeats when timeout or tail-latency signals "
            "show local saturation."
        ),
    )
    parser.add_argument(
        "--task-distribution",
        choices=["load", "repo-shard", "repo-shard-worksteal"],
        default="repo-shard-worksteal",
        help="Task distribution policy for hard local worker processes.",
    )
    parser.add_argument(
        "--soft-task-timeout",
        action="store_true",
        help=(
            "Use the legacy ThreadPool future timeout. Default local runs use hard "
            "worker-process timeouts when --task-timeout-sec is set."
        ),
    )
    parser.add_argument(
        "--checkout-strategy",
        choices=["clone", "worktree", "copytree", "tar"],
        default="worktree",
        help="Local git checkout materialization strategy for git_repo fixtures.",
    )
    parser.add_argument(
        "--cache-policy",
        choices=["auto", "process", "shared"],
        default="auto",
        help=(
            "Checkout cache policy for local sandboxes. auto uses process-local "
            "cache for stable capped production runs and shared cache for explicit "
            "pressure tests."
        ),
    )
    parser.add_argument(
        "--shared-cache-root",
        default="artifacts/cache/judge_shared",
        help="Relative path for shared checkout cache.",
    )
    parser.add_argument(
        "--pytest-diagnostics",
        action="store_true",
        help="Run extra python/pytest startup and collection probes around hidden pytest.",
    )
    parser.add_argument(
        "--task-timeout-sec",
        type=float,
        default=None,
        help="Per-task timeout. Local runs use a hard worker-process timeout by default.",
    )
    parser.add_argument(
        "--pytest-timeout-sec",
        type=float,
        default=None,
        help="Override hidden pytest subprocess timeout for every pytest judge rule.",
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
            "Set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 for hidden pytest subprocesses. "
            "Compatibility alias for --pytest-plugin-policy disabled."
        ),
    )
    plugin_group.add_argument(
        "--enable-pytest-plugin-autoload",
        dest="pytest_plugin_policy",
        action="store_const",
        const="enabled",
        help=(
            "Allow pytest to autoload third-party plugins. Use this for external "
            "repositories whose hidden tests require pytest plugins. Compatibility "
            "alias for --pytest-plugin-policy enabled."
        ),
    )
    parser.add_argument(
        "--reuse-orchestrator-across-repeats",
        action="store_true",
        help=(
            "Reuse sandbox caches across measured repeats. Default isolates repeats so "
            "judge verdict cache does not inflate throughput comparisons."
        ),
    )
    parser.add_argument("--mlx-size", type=int, default=512, help="MLX matmul sanity size.")
    parser.add_argument("--mlx-iters", type=int, default=3, help="MLX measured iterations.")
    parser.add_argument(
        "--skip-torch-mps",
        action="store_true",
        help="Skip the PyTorch MPS sanity check even if torch is installed.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/reports/toy_baseline_report.json"),
        help="JSON report path.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("artifacts/reports/mac_mlx_baseline_summary.md"),
        help="Markdown summary path.",
    )
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def mean_bool(values: list[bool]) -> float:
    return sum(1 for value in values if value) / len(values) if values else 0.0


def run_dataset(
    dataset: DatasetSpec,
    orchestrator: EvaluationOrchestrator,
    workers: int,
    task_timeout_sec: float | None,
) -> dict[str, Any]:
    start = time.perf_counter()
    report = orchestrator.run(
        dataset,
        workers=workers,
        task_timeout_sec=task_timeout_sec,
    )
    return {
        "duration_sec": time.perf_counter() - start,
        "report": report.model_dump(mode="json"),
    }


def build_orchestrator(args: argparse.Namespace) -> EvaluationOrchestrator:
    return EvaluationOrchestrator(
        runner_mode="local",
        checkout_strategy=args.checkout_strategy,
        pytest_diagnostics=args.pytest_diagnostics,
        pytest_timeout_sec=args.pytest_timeout_sec,
        pytest_plugin_policy=args.pytest_plugin_policy,
        hard_task_timeout=not args.soft_task_timeout,
        cache_policy=args.cache_policy,
        shared_cache_root=args.shared_cache_root,
        task_distribution=args.task_distribution,
    )


def mlx_sanity(size: int, warmup: int, iters: int) -> dict[str, Any]:
    try:
        import mlx.core as mx
        import numpy as np
    except Exception as exc:
        return {"status": "skipped", "reason": repr(exc), "pytorch_mps_used": False}

    rng = np.random.default_rng(42)
    a = mx.array(rng.standard_normal((size, size), dtype=np.float32))
    b = mx.array(rng.standard_normal((size, size), dtype=np.float32))
    mx.eval(a, b)
    for _ in range(warmup):
        c = a @ b
        mx.eval(c)

    times: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        c = a @ b
        mx.eval(c)
        times.append(time.perf_counter() - start)

    mean = statistics.fmean(times)
    return {
        "status": "ok",
        "backend": "mlx",
        "device": str(mx.default_device()),
        "size": size,
        "warmup": warmup,
        "iters": iters,
        "mean_seconds": mean,
        "p50_seconds": percentile(times, 0.50),
        "p95_seconds": percentile(times, 0.95),
        "gflops_mean": (2 * size**3) / mean / 1e9,
        "all_seconds": times,
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "pytorch_mps_used": False,
    }


def torch_mps_sanity(size: int, warmup: int, iters: int, skip: bool) -> dict[str, Any]:
    if skip:
        return {"status": "skipped", "reason": "disabled_by_flag", "pytorch_mps_used": False}
    try:
        import torch
    except Exception as exc:
        return {"status": "skipped", "reason": repr(exc), "pytorch_mps_used": False}

    if not torch.backends.mps.is_available():
        return {
            "status": "skipped",
            "reason": "torch.backends.mps.is_available() is false",
            "torch_version": torch.__version__,
            "pytorch_mps_used": False,
        }

    torch.manual_seed(42)
    device = torch.device("mps")
    a = torch.randn((size, size), dtype=torch.float32, device=device)
    b = torch.randn((size, size), dtype=torch.float32, device=device)
    for _ in range(warmup):
        _ = a @ b
        torch.mps.synchronize()

    times: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        _ = a @ b
        torch.mps.synchronize()
        times.append(time.perf_counter() - start)

    mean = statistics.fmean(times)
    return {
        "status": "ok",
        "backend": "torch_mps",
        "device": "mps",
        "size": size,
        "warmup": warmup,
        "iters": iters,
        "mean_seconds": mean,
        "p50_seconds": percentile(times, 0.50),
        "p95_seconds": percentile(times, 0.95),
        "gflops_mean": (2 * size**3) / mean / 1e9,
        "all_seconds": times,
        "torch_version": torch.__version__,
        "pytorch_mps_used": True,
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    wall_times = [float(r["metrics"]["wall_time_sec"]) for r in records]
    judge_times = [float(r["metrics"]["judge_time_sec"]) for r in records]
    agent_latencies = [float(r["metrics"]["agent_step_latency_sec"]) for r in records]
    patch_apply_times = [float(r["metrics"].get("patch_apply_time_sec", 0.0)) for r in records]
    hidden_test_times = [float(r["metrics"].get("hidden_test_time_sec", 0.0)) for r in records]
    hidden_test_file_write_times = [
        float(r["metrics"].get("hidden_test_file_write_time_sec", 0.0)) for r in records
    ]
    python_subprocess_startup_times = [
        float(r["metrics"].get("python_subprocess_startup_time_sec", 0.0)) for r in records
    ]
    pytest_process_startup_times = [
        float(r["metrics"].get("pytest_process_startup_time_sec", 0.0)) for r in records
    ]
    pytest_collection_times = [
        float(r["metrics"].get("pytest_collection_time_sec", 0.0)) for r in records
    ]
    pytest_execution_times = [
        float(r["metrics"].get("pytest_execution_time_sec", 0.0)) for r in records
    ]
    pytest_total_times = [
        float(r["metrics"].get("pytest_total_time_sec", 0.0)) for r in records
    ]
    judge_cache_lookup_times = [
        float(r["metrics"].get("judge_cache_lookup_time_sec", 0.0)) for r in records
    ]
    repo_checkout_times = [
        float(r["metrics"].get("repo_checkout_time_sec", 0.0)) for r in records
    ]
    git_clone_times = [float(r["metrics"].get("git_clone_time_sec", 0.0)) for r in records]
    git_checkout_times = [
        float(r["metrics"].get("git_checkout_time_sec", 0.0)) for r in records
    ]
    snapshot_materialize_times = [
        float(r["metrics"].get("snapshot_materialize_time_sec", 0.0)) for r in records
    ]
    cache_lock_wait_times = [
        float(r["metrics"].get("cache_lock_wait_time_sec", 0.0)) for r in records
    ]
    tempdir_create_times = [
        float(r["metrics"].get("tempdir_create_time_sec", 0.0)) for r in records
    ]
    judge_workspace_cleanup_times = [
        float(r["metrics"].get("judge_workspace_cleanup_time_sec", 0.0)) for r in records
    ]
    pass_count = sum(1 for r in records if r["pass_fail"] == "pass")
    cache_flags = [r["metrics"]["cache_hit_flags"] for r in records]
    failure_types: dict[str, int] = {}
    pytest_plugin_policy_counts: dict[str, int] = {}
    pytest_plugin_required_count = 0
    pytest_plugin_autoload_disabled_count = 0
    hard_timeout_count = 0
    pytest_subprocess_count = 0
    affinity_keys: set[str] = set()
    repo_shard_steal_count = 0
    worker_repo_switch_count = 0
    previous_worker_affinity: dict[int, str] = {}
    for record in records:
        failure_type = record.get("failure_type") or "none"
        failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
        if record.get("metadata", {}).get("hard_timeout_triggered"):
            hard_timeout_count += 1
        metadata = record.get("metadata", {})
        pytest_subprocess_count += int(record["metrics"].get("pytest_subprocess_count", 0))
        affinity_key = metadata.get("affinity_key")
        if affinity_key:
            affinity_keys.add(str(affinity_key))
        if metadata.get("repo_shard_stolen"):
            repo_shard_steal_count += 1
        worker_id = metadata.get("worker_id")
        if worker_id is not None and affinity_key:
            worker_int = int(worker_id)
            previous = previous_worker_affinity.get(worker_int)
            if previous is not None and previous != affinity_key:
                worker_repo_switch_count += 1
            previous_worker_affinity[worker_int] = str(affinity_key)
        plugin_scan = record.get("metadata", {}).get("pytest_plugin_scan") or {}
        effective_policy = str(plugin_scan.get("effective_policy") or "unknown")
        pytest_plugin_policy_counts[effective_policy] = (
            pytest_plugin_policy_counts.get(effective_policy, 0) + 1
        )
        if plugin_scan.get("plugin_required"):
            pytest_plugin_required_count += 1
        if plugin_scan.get("disable_autoload"):
            pytest_plugin_autoload_disabled_count += 1

    return {
        "task_records": len(records),
        "pass_rate": pass_count / len(records) if records else 0.0,
        "wall_time_p50_sec": percentile(wall_times, 0.50),
        "wall_time_p95_sec": percentile(wall_times, 0.95),
        "judge_time_sec_avg": statistics.fmean(judge_times) if judge_times else 0.0,
        "patch_apply_time_sec_avg": (
            statistics.fmean(patch_apply_times) if patch_apply_times else 0.0
        ),
        "hidden_test_time_sec_avg": (
            statistics.fmean(hidden_test_times) if hidden_test_times else 0.0
        ),
        "hidden_test_file_write_time_sec_avg": (
            statistics.fmean(hidden_test_file_write_times)
            if hidden_test_file_write_times
            else 0.0
        ),
        "python_subprocess_startup_time_sec_avg": (
            statistics.fmean(python_subprocess_startup_times)
            if python_subprocess_startup_times
            else 0.0
        ),
        "pytest_process_startup_time_sec_avg": (
            statistics.fmean(pytest_process_startup_times)
            if pytest_process_startup_times
            else 0.0
        ),
        "pytest_collection_time_sec_avg": (
            statistics.fmean(pytest_collection_times) if pytest_collection_times else 0.0
        ),
        "pytest_execution_time_sec_avg": (
            statistics.fmean(pytest_execution_times) if pytest_execution_times else 0.0
        ),
        "pytest_total_time_sec_avg": (
            statistics.fmean(pytest_total_times) if pytest_total_times else 0.0
        ),
        "judge_cache_lookup_time_sec_avg": (
            statistics.fmean(judge_cache_lookup_times) if judge_cache_lookup_times else 0.0
        ),
        "repo_checkout_time_sec_avg": (
            statistics.fmean(repo_checkout_times) if repo_checkout_times else 0.0
        ),
        "git_clone_time_sec_avg": (
            statistics.fmean(git_clone_times) if git_clone_times else 0.0
        ),
        "git_checkout_time_sec_avg": (
            statistics.fmean(git_checkout_times) if git_checkout_times else 0.0
        ),
        "snapshot_materialize_time_sec_avg": (
            statistics.fmean(snapshot_materialize_times) if snapshot_materialize_times else 0.0
        ),
        "cache_lock_wait_time_sec_avg": (
            statistics.fmean(cache_lock_wait_times) if cache_lock_wait_times else 0.0
        ),
        "tempdir_create_time_sec_avg": (
            statistics.fmean(tempdir_create_times) if tempdir_create_times else 0.0
        ),
        "judge_workspace_cleanup_time_sec_avg": (
            statistics.fmean(judge_workspace_cleanup_times)
            if judge_workspace_cleanup_times
            else 0.0
        ),
        "agent_step_latency_p50_sec": percentile(agent_latencies, 0.50),
        "agent_step_latency_p95_sec": percentile(agent_latencies, 0.95),
        "repo_snapshot_cache_hit_rate": mean_bool([f["repo_snapshot"] for f in cache_flags]),
        "repo_map_cache_hit_rate": mean_bool([f["repo_map"] for f in cache_flags]),
        "llm_prompt_cache_hit_rate": mean_bool([f["llm_prompt"] for f in cache_flags]),
        "judge_verdict_cache_hit_rate": mean_bool([f["judge_verdict"] for f in cache_flags]),
        "memory_peak_mb": current_memory_mb(),
        "cost_per_pass": None,
        "hidden_judge_timeout_count": failure_types.get("hidden_judge_timeout", 0),
        "task_timeout_count": failure_types.get("task_timeout", 0),
        "hard_timeout_count": hard_timeout_count,
        "pytest_subprocess_count": pytest_subprocess_count,
        "pytest_subprocess_per_task": (
            pytest_subprocess_count / len(records) if records else 0.0
        ),
        "affinity_key_count": len(affinity_keys),
        "worker_repo_switch_count": worker_repo_switch_count,
        "repo_shard_steal_count": repo_shard_steal_count,
        "failure_type_counts": failure_types,
        "pytest_plugin_policy_counts": pytest_plugin_policy_counts,
        "pytest_plugin_required_count": pytest_plugin_required_count,
        "pytest_plugin_autoload_disabled_rate": (
            pytest_plugin_autoload_disabled_count / len(records) if records else 0.0
        ),
    }


def add_run_duration_metrics(metrics: dict[str, Any], runs: list[dict[str, Any]]) -> None:
    durations = [float(run["duration_sec"]) for run in runs]
    task_counts = [len(run["report"].get("run_records", [])) for run in runs]
    total_duration = sum(durations)
    total_tasks = sum(task_counts)
    metrics["run_duration_p50_sec"] = percentile(durations, 0.50)
    metrics["run_duration_p95_sec"] = percentile(durations, 0.95)
    metrics["run_duration_total_sec"] = total_duration
    metrics["throughput_tasks_per_sec"] = (
        total_tasks / total_duration if total_duration > 0 else 0.0
    )


def current_memory_mb() -> float:
    # macOS reports ru_maxrss in bytes; Linux reports KiB. The Apple benchmark path
    # is the default here, so prefer the Darwin interpretation.
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def hardware_payload(backend: str, pytorch_mps_used: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "benchmark_backend": backend,
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "pytorch_mps_used": pytorch_mps_used,
    }
    try:
        import mlx.core as mx

        payload["mlx_version"] = getattr(mx, "__version__", "unknown")
    except Exception:
        payload["mlx_version"] = None
    try:
        import torch

        payload["torch_version"] = torch.__version__
    except Exception:
        payload["torch_version"] = None
    return payload


def recommended_worker_cap(pytest_diagnostics: bool) -> int:
    if platform.system() == "Darwin":
        return 32 if pytest_diagnostics else 8
    detected = os.cpu_count() or 1
    return min(detected, 32) if pytest_diagnostics else detected


def effective_worker_count(
    requested_workers: int,
    task_count: int,
    pytest_diagnostics: bool,
    pressure_test: bool,
    adaptive_worker_cap: bool,
    scheduler_policy: str = "fixed",
) -> tuple[int, int | None]:
    requested = max(1, requested_workers)
    if scheduler_policy == "adaptive":
        if pressure_test:
            return min(requested, task_count), None
        cap = recommended_worker_cap(pytest_diagnostics)
        return max(1, min(requested, task_count, cap)), cap
    if pressure_test and not adaptive_worker_cap:
        return requested, None
    cap = recommended_worker_cap(pytest_diagnostics)
    return max(1, min(requested, task_count, cap)), cap


def next_adaptive_worker_count(
    current_workers: int,
    metrics: dict[str, Any],
    *,
    task_timeout_sec: float | None,
) -> tuple[int, str]:
    timeout_count = int(metrics.get("hidden_judge_timeout_count", 0)) + int(
        metrics.get("task_timeout_count", 0)
    )
    pass_rate = float(metrics.get("pass_rate", 0.0))
    wall_p95 = float(metrics.get("wall_time_p95_sec", 0.0))
    pytest_avg = float(metrics.get("pytest_execution_time_sec_avg", 0.0))

    if timeout_count > 0 or pass_rate < 1.0:
        return max(1, current_workers // 2), "timeout_or_pass_rate_backpressure"
    if task_timeout_sec and wall_p95 > task_timeout_sec * 0.75:
        return max(1, current_workers // 2), "wall_p95_backpressure"
    if pytest_avg > 0.75 and current_workers > 8:
        return max(8, current_workers // 2), "pytest_saturation_backpressure"
    return current_workers, "stable"


def task_affinity_key(task: Any) -> str:
    git_repo = task.fixtures.get("git_repo")
    if isinstance(git_repo, dict):
        return str(git_repo.get("repo_id") or git_repo.get("path") or task.task_id)
    toy_repo = task.fixtures.get("toy_repo")
    if isinstance(toy_repo, dict):
        return str(toy_repo.get("repo_id") or task.task_id)
    return task.task_id


def repo_affinity_worker_cap(
    dataset: DatasetSpec,
    workers: int,
    *,
    hard_task_timeout: bool,
    pressure_test: bool,
) -> tuple[int, dict[str, Any] | None]:
    if pressure_test or not hard_task_timeout or workers <= 1:
        return workers, None
    keys = [task_affinity_key(task) for task in dataset.tasks]
    unique_keys = len(set(keys))
    if unique_keys <= 1 or unique_keys >= workers:
        return workers, None
    avg_tasks_per_key = len(keys) / unique_keys
    if avg_tasks_per_key < 4:
        return workers, None
    return max(1, unique_keys), {
        "reason": "repo_affinity_cap",
        "unique_affinity_keys": unique_keys,
        "avg_tasks_per_key": avg_tasks_per_key,
        "uncapped_workers": workers,
    }


def resolve_cache_policy(
    requested_policy: str,
    *,
    hard_task_timeout: bool,
    pressure_test: bool,
    repo_cap_reason: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    if requested_policy != "auto":
        return requested_policy, {"reason": "explicit", "requested": requested_policy}
    if pressure_test:
        return "shared", {"reason": "auto_pressure_test_shared"}
    if hard_task_timeout and repo_cap_reason is not None:
        return "process", {
            "reason": "auto_repo_affinity_process",
            "repo_cap_reason": repo_cap_reason,
        }
    return "process", {"reason": "auto_stable_process"}


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    dataset = payload["dataset"]
    config = payload["config"]
    lines = [
        "# Mac MLX Local Benchmark Summary",
        "",
        f"- Created at: `{payload['created_at']}`",
        f"- Backend: `{payload['benchmark_backend']}`",
        f"- Dataset: `{dataset['dataset_id']}` ({dataset['task_count']} tasks)",
        (
            "- Workers requested/effective/cap/pressure: "
            f"`{config['requested_workers']}` / `{config['workers']}` / "
            f"`{config['worker_cap']}` / `{config['pressure_test']}`"
        ),
        f"- Worker cap reason: `{config['worker_cap_reason']}`",
        f"- Scheduler policy: `{config['scheduler_policy']}`",
        f"- Task distribution: `{config['task_distribution']}`",
        f"- Hard task timeout: `{config['hard_task_timeout']}`",
        f"- Checkout strategy: `{config['checkout_strategy']}`",
        f"- Cache policy: `{config['cache_policy']}`",
        f"- Pytest timeout override: `{config['pytest_timeout_sec']}`",
        f"- Pytest plugin policy: `{config['pytest_plugin_policy']}`",
        (
            "- Disable pytest plugin autoload: "
            f"`{config['disable_pytest_plugin_autoload']}`"
        ),
        f"- Pass rate: `{metrics['pass_rate']:.3f}`",
        (
            "- Wall time p50/p95: "
            f"`{metrics['wall_time_p50_sec']:.6f}s` / "
            f"`{metrics['wall_time_p95_sec']:.6f}s`"
        ),
        f"- Judge avg: `{metrics['judge_time_sec_avg']:.6f}s`",
        f"- Patch apply avg: `{metrics['patch_apply_time_sec_avg']:.6f}s`",
        f"- Hidden test avg: `{metrics['hidden_test_time_sec_avg']:.6f}s`",
        f"- Hidden judge timeout count: `{metrics['hidden_judge_timeout_count']}`",
        f"- Task timeout count: `{metrics['task_timeout_count']}`",
        f"- Hard timeout count: `{metrics['hard_timeout_count']}`",
        (
            "- Hidden write / pytest exec avg: "
            f"`{metrics['hidden_test_file_write_time_sec_avg']:.6f}s` / "
            f"`{metrics['pytest_execution_time_sec_avg']:.6f}s`"
        ),
        (
            "- Python startup / pytest startup / collection avg: "
            f"`{metrics['python_subprocess_startup_time_sec_avg']:.6f}s` / "
            f"`{metrics['pytest_process_startup_time_sec_avg']:.6f}s` / "
            f"`{metrics['pytest_collection_time_sec_avg']:.6f}s`"
        ),
        f"- Repo checkout avg: `{metrics['repo_checkout_time_sec_avg']:.6f}s`",
        (
            "- Git clone / checkout / materialize avg: "
            f"`{metrics['git_clone_time_sec_avg']:.6f}s` / "
            f"`{metrics['git_checkout_time_sec_avg']:.6f}s` / "
            f"`{metrics['snapshot_materialize_time_sec_avg']:.6f}s`"
        ),
        f"- Cache lock wait avg: `{metrics['cache_lock_wait_time_sec_avg']:.6f}s`",
        (
            "- Tempdir create / cleanup avg: "
            f"`{metrics['tempdir_create_time_sec_avg']:.6f}s` / "
            f"`{metrics['judge_workspace_cleanup_time_sec_avg']:.6f}s`"
        ),
        (
            "- Judge cache lookup avg: "
            f"`{metrics['judge_cache_lookup_time_sec_avg']:.6f}s`"
        ),
        (
            "- Agent step p50/p95: "
            f"`{metrics['agent_step_latency_p50_sec']:.6f}s` / "
            f"`{metrics['agent_step_latency_p95_sec']:.6f}s`"
        ),
        (
            "- Snapshot/repo-map/prompt hit: "
            f"`{metrics['repo_snapshot_cache_hit_rate']:.3f}` / "
            f"`{metrics['repo_map_cache_hit_rate']:.3f}` / "
            f"`{metrics['llm_prompt_cache_hit_rate']:.3f}`"
        ),
        f"- Judge verdict cache hit: `{metrics['judge_verdict_cache_hit_rate']:.3f}`",
        (
            "- Affinity keys / worker repo switches / steals: "
            f"`{metrics['affinity_key_count']}` / "
            f"`{metrics['worker_repo_switch_count']}` / "
            f"`{metrics['repo_shard_steal_count']}`"
        ),
        (
            "- Pytest subprocess count / per task: "
            f"`{metrics['pytest_subprocess_count']}` / "
            f"`{metrics['pytest_subprocess_per_task']:.3f}`"
        ),
        f"- Memory peak: `{metrics['memory_peak_mb']:.2f} MB`",
        (
            "- Run duration p50/p95: "
            f"`{metrics['run_duration_p50_sec']:.6f}s` / "
            f"`{metrics['run_duration_p95_sec']:.6f}s`"
        ),
        f"- Throughput: `{metrics['throughput_tasks_per_sec']:.3f} tasks/s`",
        (
            "- Pytest plugin required / autoload disabled rate: "
            f"`{metrics['pytest_plugin_required_count']}` / "
            f"`{metrics['pytest_plugin_autoload_disabled_rate']:.3f}`"
        ),
        f"- MLX sanity: `{payload['mlx_sanity']['status']}`",
        f"- PyTorch MPS sanity: `{payload['torch_mps_sanity']['status']}`",
        "",
        "## Interpretation Boundary",
        "",
        (
            "This report validates the local Apple Silicon/MLX engineering loop. "
            "It does not represent CUDA or GPU-cluster industrial throughput."
        ),
        "",
        "## Failure Types",
        "",
    ]
    for failure_type, count in sorted(metrics["failure_type_counts"].items()):
        lines.append(f"- `{failure_type}`: {count}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.adaptive_worker_cap:
        args.scheduler_policy = "adaptive"
    dataset = DatasetSpec.model_validate_json(args.manifest.read_text(encoding="utf-8"))
    source_task_count = len(dataset.tasks)
    if args.max_tasks is not None:
        dataset = dataset.model_copy(update={"tasks": dataset.tasks[: args.max_tasks]})
    effective_workers, worker_cap = effective_worker_count(
        args.workers,
        len(dataset.tasks),
        args.pytest_diagnostics,
        args.pressure_test,
        args.adaptive_worker_cap,
        args.scheduler_policy,
    )
    hard_task_timeout = not args.soft_task_timeout and args.task_timeout_sec is not None
    effective_workers, repo_cap_reason = repo_affinity_worker_cap(
        dataset,
        effective_workers,
        hard_task_timeout=hard_task_timeout,
        pressure_test=args.pressure_test,
    )
    worker_cap_reason = repo_cap_reason or (
        {"reason": "local_default_cap", "cap": worker_cap}
        if worker_cap is not None
        else {"reason": "pressure_test_or_uncapped"}
    )
    requested_cache_policy = args.cache_policy
    args.cache_policy, cache_policy_reason = resolve_cache_policy(
        requested_cache_policy,
        hard_task_timeout=hard_task_timeout,
        pressure_test=args.pressure_test,
        repo_cap_reason=repo_cap_reason,
    )
    orchestrator = build_orchestrator(args)

    for _ in range(args.warmup):
        run_dataset(dataset, orchestrator, effective_workers, args.task_timeout_sec)

    measured_runs = []
    adaptive_history = []
    current_workers = effective_workers
    for _ in range(args.repeat):
        if not args.reuse_orchestrator_across_repeats:
            orchestrator = build_orchestrator(args)
        run = run_dataset(dataset, orchestrator, current_workers, args.task_timeout_sec)
        measured_runs.append(run)
        run_metrics = summarize_records(run["report"].get("run_records", []))
        next_workers, reason = (
            next_adaptive_worker_count(
                current_workers,
                run_metrics,
                task_timeout_sec=args.task_timeout_sec,
            )
            if args.scheduler_policy == "adaptive"
            else (current_workers, "fixed")
        )
        adaptive_history.append(
            {
                "workers": current_workers,
                "next_workers": next_workers,
                "reason": reason,
                "pass_rate": run_metrics.get("pass_rate", 0.0),
                "hidden_judge_timeout_count": run_metrics.get(
                    "hidden_judge_timeout_count", 0
                ),
                "task_timeout_count": run_metrics.get("task_timeout_count", 0),
                "wall_time_p95_sec": run_metrics.get("wall_time_p95_sec", 0.0),
                "pytest_execution_time_sec_avg": run_metrics.get(
                    "pytest_execution_time_sec_avg", 0.0
                ),
            }
        )
        current_workers = next_workers
    run_records = [
        record
        for run in measured_runs
        for record in run["report"].get("run_records", [])
    ]
    metrics = summarize_records(run_records)
    add_run_duration_metrics(metrics, measured_runs)
    mlx_report = mlx_sanity(args.mlx_size, args.warmup, args.mlx_iters)
    torch_mps_report = torch_mps_sanity(
        args.mlx_size,
        args.warmup,
        args.mlx_iters,
        args.skip_torch_mps,
    )
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark_backend": args.backend,
        "hardware": hardware_payload(
            args.backend,
            bool(torch_mps_report.get("pytorch_mps_used")),
        ),
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "version": dataset.version,
            "task_count": len(dataset.tasks),
            "source_task_count": source_task_count,
        },
        "config": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "workers": effective_workers,
            "requested_workers": args.workers,
            "worker_cap": worker_cap,
            "pressure_test": args.pressure_test,
            "adaptive_worker_cap": args.adaptive_worker_cap,
            "scheduler_policy": args.scheduler_policy,
            "task_distribution": args.task_distribution,
            "worker_cap_reason": worker_cap_reason,
            "requested_cache_policy": requested_cache_policy,
            "cache_policy_reason": cache_policy_reason,
            "adaptive_worker_history": adaptive_history,
            "hard_task_timeout": hard_task_timeout,
            "soft_task_timeout": args.soft_task_timeout,
            "task_timeout_sec": args.task_timeout_sec,
            "pytest_timeout_sec": args.pytest_timeout_sec,
            "mlx_size": args.mlx_size,
            "mlx_iters": args.mlx_iters,
            "max_tasks": args.max_tasks,
            "checkout_strategy": args.checkout_strategy,
            "cache_policy": args.cache_policy,
            "shared_cache_root": args.shared_cache_root,
            "pytest_diagnostics": args.pytest_diagnostics,
            "pytest_plugin_policy": args.pytest_plugin_policy,
            "disable_pytest_plugin_autoload": args.pytest_plugin_policy != "enabled",
            "reuse_orchestrator_across_repeats": args.reuse_orchestrator_across_repeats,
        },
        "metrics": metrics,
        "runs": measured_runs,
        "mlx_sanity": mlx_report,
        "torch_mps_sanity": torch_mps_report,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(args.summary_out, payload)
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
