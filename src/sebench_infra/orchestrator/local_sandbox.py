from __future__ import annotations

import contextlib
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from sebench_infra.benchmark.schemas import TaskSpec
from sebench_infra.orchestrator.agents import build_patch_agent_runner
from sebench_infra.orchestrator.cache import (
    JudgeVerdictCache,
    SnapshotHitTracker,
    _benchmark_environment,
    _hash_artifacts,
    _judge_cache_key,
)
from sebench_infra.orchestrator.judge import (
    LocalJudgeRunner,
    _aggregate_detail_timings,
    _aggregate_pytest_subprocess_count,
    _normalize_pytest_plugin_policy,
)
from sebench_infra.orchestrator.sandbox_types import SandboxResult
from sebench_infra.orchestrator.whitelist import PathWhitelist
from sebench_infra.orchestrator.workspace import (
    CheckoutResult,
    WorkspaceManager,
    _cleanup_hidden_judge_artifacts,
    _clone_git_repo,
    _empty_checkout_timings,
    _string_map,
    _write_files,
)


class LocalSandbox:
    """Local runner with Work/Judge separation and pluggable patch agents."""

    def __init__(
        self,
        checkout_strategy: str = "worktree",
        pytest_diagnostics: bool = False,
        pytest_timeout_sec: float | None = None,
        disable_pytest_plugin_autoload: bool = True,
        pytest_plugin_policy: str | None = None,
        cache_policy: str = "process",
        shared_cache_root: str | Path = "artifacts/cache/judge_shared",
        agent_backend: str = "fixture",
        codex_binary: str = "codex",
        codex_model: str | None = None,
        codex_timeout_sec: float = 300.0,
    ) -> None:
        self.checkout_strategy = checkout_strategy
        self.cache_policy = cache_policy
        self.pytest_diagnostics = pytest_diagnostics
        self.pytest_timeout_sec = pytest_timeout_sec
        self.pytest_plugin_policy = _normalize_pytest_plugin_policy(
            pytest_plugin_policy,
            disable_pytest_plugin_autoload,
        )
        self.disable_pytest_plugin_autoload = self.pytest_plugin_policy != "enabled"
        self.workspace = WorkspaceManager(
            checkout_strategy=checkout_strategy,
            cache_policy=cache_policy,
            shared_cache_root=shared_cache_root,
        )
        self.judge = LocalJudgeRunner(
            pytest_diagnostics=pytest_diagnostics,
            pytest_timeout_sec=pytest_timeout_sec,
            pytest_plugin_policy=self.pytest_plugin_policy,
        )
        self.agent_runner = build_patch_agent_runner(
            agent_backend=agent_backend,
            codex_binary=codex_binary,
            codex_model=codex_model,
            codex_timeout_sec=codex_timeout_sec,
        )
        self.snapshot_hits = SnapshotHitTracker()
        self.judge_cache = JudgeVerdictCache()

    def run_task(self, task: TaskSpec) -> SandboxResult:
        git_repo = task.fixtures.get("git_repo")
        if isinstance(git_repo, dict):
            return self._run_git_task(task, git_repo)

        toy_repo = task.fixtures.get("toy_repo")
        if isinstance(toy_repo, dict):
            return self._run_toy_task(task, toy_repo)

        return self._run_legacy_task(task)

    def _run_legacy_task(self, task: TaskSpec) -> SandboxResult:
        start = time.perf_counter()
        answer = (
            f"Task {task.task_id} is reproducible using public fixtures, typed schemas, "
            "and isolated evaluation contracts."
        )
        artifacts = {"submission/answer.txt": answer}
        whitelist = PathWhitelist(task.allowed_paths)
        safe_artifacts = {
            path: content for path, content in artifacts.items() if whitelist.is_allowed(path)
        }
        patch_hash = _hash_artifacts(safe_artifacts)
        return SandboxResult(
            artifacts=safe_artifacts,
            metadata={
                "local_runner": True,
                "agent_backend": "legacy_demo",
                "patch_hash": patch_hash,
                "wall_time_sec": time.perf_counter() - start,
                "judge_time_sec": 0.0,
                "agent_step_latency_sec": 0.0,
                "cache_hit_flags": {
                    "image": False,
                    "repo_snapshot": False,
                    "repo_map": False,
                    "llm_prompt": True,
                    "judge_verdict": False,
                },
                "benchmark_environment": _benchmark_environment(),
            },
        )

    def _run_toy_task(self, task: TaskSpec, toy_repo: dict[str, Any]) -> SandboxResult:
        start = time.perf_counter()
        repo_key = str(toy_repo.get("repo_id", task.task_id))
        snapshot_hit = self.snapshot_hits.mark_seen(repo_key)

        with tempfile.TemporaryDirectory(prefix=f"sebench-{task.task_id}-") as temp_root:
            root = Path(temp_root)
            work_dir = root / "work"
            judge_dir = root / "judge"
            work_dir.mkdir()
            judge_dir.mkdir()

            initial_files = _string_map(toy_repo.get("files", {}))
            _write_files(work_dir, initial_files)
            _write_files(judge_dir, initial_files)

            agent_result = self.agent_runner.run(
                task=task,
                work_dir=work_dir,
                fixture=toy_repo,
                patch_required=False,
            )
            safe_artifacts = self._safe_artifacts(task, agent_result.artifacts)
            patch_hash = _hash_artifacts(safe_artifacts)
            uses_patch = (
                isinstance(toy_repo.get("agent_patch"), str)
                or "submission/model.patch" in agent_result.artifacts
            )
            plugin_scan = self.judge.resolve_plugin_policy(
                judge_dir,
                toy_repo.get("hidden_judge", []),
            )
            judge_start = time.perf_counter()
            lookup_start = time.perf_counter()
            judge_cache_key = _judge_cache_key(task.task_id, patch_hash, toy_repo)
            cached_verdict = self.judge_cache.get(judge_cache_key, plugin_scan)
            judge_cache_lookup_time = time.perf_counter() - lookup_start
            judge_cache_hit = cached_verdict is not None
            patch_apply_time = 0.0
            hidden_test_time = 0.0
            if cached_verdict is not None:
                judge_passed, failure_type, judge_details = cached_verdict
            else:
                judge_run = self.judge.run(
                    judge_dir=judge_dir,
                    task=task,
                    fixture=toy_repo,
                    safe_artifacts=safe_artifacts,
                    uses_patch=uses_patch,
                    plugin_scan=plugin_scan,
                )
                judge_passed = judge_run.passed
                failure_type = judge_run.failure_type
                judge_details = judge_run.details
                patch_apply_time = judge_run.patch_apply_time_sec
                hidden_test_time = judge_run.hidden_test_time_sec
                self.judge_cache.set(
                    judge_cache_key,
                    plugin_scan,
                    (judge_passed, failure_type, judge_details),
                )
            judge_time = time.perf_counter() - judge_start

        metadata = self._metadata(
            start=start,
            patch_hash=patch_hash,
            judge_passed=judge_passed,
            failure_type=failure_type,
            judge_details=judge_details,
            judge_time=judge_time,
            agent_step_latency=agent_result.latency_sec,
            patch_apply_time=patch_apply_time,
            hidden_test_time=hidden_test_time,
            judge_cache_lookup_time=judge_cache_lookup_time,
            judge_cache_hit=judge_cache_hit,
            snapshot_hit=snapshot_hit,
            plugin_scan=plugin_scan,
            checkout_timings=_empty_checkout_timings(),
            cleanup_time=0.0,
            tempdir_create_time=0.0,
            extra={
                "toy_repo_runner": True,
                **agent_result.metadata,
            },
        )
        return SandboxResult(artifacts=safe_artifacts, metadata=metadata)

    def _run_git_task(self, task: TaskSpec, git_repo: dict[str, Any]) -> SandboxResult:
        start = time.perf_counter()
        repo_path = Path(str(git_repo["path"])).expanduser().resolve()
        base_commit = str(git_repo.get("base_commit", "HEAD"))
        repo_key = f"{repo_path}:{base_commit}"
        snapshot_hit = self.snapshot_hits.mark_seen(repo_key)

        tempdir_create_start = time.perf_counter()
        temp_root = Path(tempfile.mkdtemp(prefix=f"sebench-git-{task.task_id}-"))
        tempdir_create_time = time.perf_counter() - tempdir_create_start
        cleanup_time = 0.0
        checkout: CheckoutResult | None = None
        judge_dir: Path | None = None

        try:
            work_dir = temp_root / "work"
            requested_judge_dir = temp_root / "judge"
            if self.agent_runner.backend == "codex_cli":
                _clone_git_repo(repo_path, work_dir, base_commit)
            else:
                work_dir.mkdir()

            agent_result = self.agent_runner.run(
                task=task,
                work_dir=work_dir,
                fixture=git_repo,
                patch_required=True,
            )
            safe_artifacts = self._safe_artifacts(task, agent_result.artifacts)
            patch_hash = _hash_artifacts(safe_artifacts)
            uses_patch = True

            checkout = self.workspace.prepare_git_checkout(
                repo_path,
                requested_judge_dir,
                base_commit,
            )
            judge_dir = checkout.path

            plugin_scan = self.judge.resolve_plugin_policy(
                judge_dir,
                git_repo.get("hidden_judge", []),
            )
            judge_start = time.perf_counter()
            lookup_start = time.perf_counter()
            judge_cache_key = _judge_cache_key(task.task_id, patch_hash, git_repo)
            cached_verdict = self.judge_cache.get(judge_cache_key, plugin_scan)
            judge_cache_lookup_time = time.perf_counter() - lookup_start
            judge_cache_hit = cached_verdict is not None
            patch_apply_time = 0.0
            hidden_test_time = 0.0
            if cached_verdict is not None:
                judge_passed, failure_type, judge_details = cached_verdict
            else:
                judge_run = self.judge.run(
                    judge_dir=judge_dir,
                    task=task,
                    fixture=git_repo,
                    safe_artifacts=safe_artifacts,
                    uses_patch=uses_patch,
                    plugin_scan=plugin_scan,
                )
                judge_passed = judge_run.passed
                failure_type = judge_run.failure_type
                judge_details = judge_run.details
                patch_apply_time = judge_run.patch_apply_time_sec
                hidden_test_time = judge_run.hidden_test_time_sec
                self.judge_cache.set(
                    judge_cache_key,
                    plugin_scan,
                    (judge_passed, failure_type, judge_details),
                )
            judge_time = time.perf_counter() - judge_start
        finally:
            cleanup_start = time.perf_counter()
            if checkout is not None and self.cache_policy == "shared" and judge_dir is not None:
                with contextlib.suppress(Exception):
                    _cleanup_hidden_judge_artifacts(
                        judge_dir,
                        git_repo.get("hidden_judge", []),
                    )
            shutil.rmtree(temp_root, ignore_errors=True)
            cleanup_time = time.perf_counter() - cleanup_start

        metadata = self._metadata(
            start=start,
            patch_hash=patch_hash,
            judge_passed=judge_passed,
            failure_type=failure_type,
            judge_details=judge_details,
            judge_time=judge_time,
            agent_step_latency=agent_result.latency_sec,
            patch_apply_time=patch_apply_time,
            hidden_test_time=hidden_test_time,
            judge_cache_lookup_time=judge_cache_lookup_time,
            judge_cache_hit=judge_cache_hit,
            snapshot_hit=snapshot_hit,
            plugin_scan=plugin_scan,
            checkout_timings=checkout.timings,
            cleanup_time=cleanup_time,
            tempdir_create_time=tempdir_create_time,
            extra={
                "git_repo_runner": True,
                "checkout_strategy": self.checkout_strategy,
                "cache_policy": self.cache_policy,
                **agent_result.metadata,
            },
        )
        return SandboxResult(artifacts=safe_artifacts, metadata=metadata)

    def _safe_artifacts(self, task: TaskSpec, artifacts: dict[str, str]) -> dict[str, str]:
        whitelist = PathWhitelist(task.allowed_paths)
        return {path: content for path, content in artifacts.items() if whitelist.is_allowed(path)}

    def _metadata(
        self,
        *,
        start: float,
        patch_hash: str,
        judge_passed: bool,
        failure_type: str | None,
        judge_details: list[dict[str, Any]],
        judge_time: float,
        agent_step_latency: float,
        patch_apply_time: float,
        hidden_test_time: float,
        judge_cache_lookup_time: float,
        judge_cache_hit: bool,
        snapshot_hit: bool,
        plugin_scan: dict[str, Any],
        checkout_timings: dict[str, float],
        cleanup_time: float,
        tempdir_create_time: float,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "local_runner": True,
            "pytest_diagnostics": self.pytest_diagnostics,
            "pytest_timeout_sec": self.pytest_timeout_sec,
            "pytest_plugin_policy": self.pytest_plugin_policy,
            "pytest_plugin_scan": plugin_scan,
            "disable_pytest_plugin_autoload": plugin_scan["disable_autoload"],
            "patch_hash": patch_hash,
            "judge_passed": judge_passed,
            "failure_type": None if judge_passed else failure_type,
            "judge_details": judge_details,
            "wall_time_sec": time.perf_counter() - start,
            "judge_time_sec": judge_time,
            "agent_step_latency_sec": agent_step_latency,
            "patch_apply_time_sec": patch_apply_time,
            "hidden_test_time_sec": hidden_test_time,
            **_aggregate_detail_timings(judge_details),
            "judge_cache_lookup_time_sec": judge_cache_lookup_time,
            "repo_checkout_time_sec": checkout_timings["repo_checkout_time_sec"],
            "git_clone_time_sec": checkout_timings["git_clone_time_sec"],
            "git_checkout_time_sec": checkout_timings["git_checkout_time_sec"],
            "snapshot_materialize_time_sec": checkout_timings[
                "snapshot_materialize_time_sec"
            ],
            "cache_lock_wait_time_sec": checkout_timings["cache_lock_wait_time_sec"],
            "tempdir_create_time_sec": tempdir_create_time,
            "judge_workspace_cleanup_time_sec": cleanup_time,
            "pytest_subprocess_count": _aggregate_pytest_subprocess_count(judge_details),
            "memory_peak_mb": None,
            "cache_hit_flags": {
                "image": False,
                "repo_snapshot": snapshot_hit,
                "repo_map": snapshot_hit,
                "llm_prompt": True,
                "judge_verdict": judge_cache_hit,
            },
            "benchmark_environment": _benchmark_environment(),
            **extra,
        }
