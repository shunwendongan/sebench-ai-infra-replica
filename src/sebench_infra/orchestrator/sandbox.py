import contextlib
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from sebench_infra.benchmark.schemas import TaskSpec
from sebench_infra.orchestrator.whitelist import PathWhitelist


@dataclass(frozen=True)
class SandboxResult:
    artifacts: dict[str, str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CheckoutResult:
    path: Path
    timings: dict[str, float]


class LocalSandbox:
    """Local runner with a real toy Work/Judge split.

    Tasks without a ``fixtures.toy_repo`` payload keep the deterministic legacy demo
    behavior. Toy tasks create separate Work and Judge directories: Work produces
    allowlisted artifacts, while Judge applies only those artifacts to a clean copy
    before running hidden rules.
    """

    def __init__(
        self,
        checkout_strategy: str = "worktree",
        pytest_diagnostics: bool = False,
        pytest_timeout_sec: float | None = None,
        disable_pytest_plugin_autoload: bool = True,
        pytest_plugin_policy: str | None = None,
        cache_policy: str = "process",
        shared_cache_root: str | Path = "artifacts/cache/judge_shared",
    ) -> None:
        valid_checkout_strategies = {"clone", "worktree", "copytree", "tar"}
        if checkout_strategy not in valid_checkout_strategies:
            raise ValueError(
                f"checkout_strategy must be one of {sorted(valid_checkout_strategies)}"
            )
        if cache_policy not in {"process", "shared"}:
            raise ValueError("cache_policy must be one of: process, shared")
        self.checkout_strategy = checkout_strategy
        self.cache_policy = cache_policy
        self.shared_cache_root = _resolve_shared_cache_root(shared_cache_root)
        self.pytest_diagnostics = pytest_diagnostics
        self.pytest_timeout_sec = pytest_timeout_sec
        self.pytest_plugin_policy = _normalize_pytest_plugin_policy(
            pytest_plugin_policy,
            disable_pytest_plugin_autoload,
        )
        self.disable_pytest_plugin_autoload = self.pytest_plugin_policy != "enabled"
        self._snapshot_cache: set[str] = set()
        self._judge_cache: dict[str, tuple[bool, str | None, list[dict[str, Any]]]] = {}
        self._cache_root = Path(tempfile.mkdtemp(prefix="sebench-sandbox-cache-"))
        self._snapshot_dirs: dict[str, Path] = {}
        self._tar_paths: dict[str, Path] = {}
        self._persistent_checkouts: dict[str, Path] = {}
        self._workspace_cache_lock = threading.RLock()
        self._judge_cache_lock = threading.Lock()

    def run_task(self, task: TaskSpec) -> SandboxResult:
        git_repo = task.fixtures.get("git_repo")
        if isinstance(git_repo, dict):
            return self._run_git_task(task, git_repo)

        toy_repo = task.fixtures.get("toy_repo")
        if isinstance(toy_repo, dict):
            return self._run_toy_task(task, toy_repo)

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
        with self._workspace_cache_lock:
            snapshot_hit = repo_key in self._snapshot_cache
            self._snapshot_cache.add(repo_key)

        with tempfile.TemporaryDirectory(prefix=f"sebench-{task.task_id}-") as temp_root:
            root = Path(temp_root)
            work_dir = root / "work"
            judge_dir = root / "judge"
            work_dir.mkdir()
            judge_dir.mkdir()

            initial_files = _string_map(toy_repo.get("files", {}))
            agent_files = _string_map(toy_repo.get("agent_files", {}))
            agent_patch = toy_repo.get("agent_patch")
            _write_files(work_dir, initial_files)

            agent_start = time.perf_counter()
            if isinstance(agent_patch, str):
                _write_files(work_dir, {"submission/model.patch": agent_patch})
            else:
                _write_files(work_dir, agent_files)
            agent_step_latency = time.perf_counter() - agent_start

            whitelist = PathWhitelist(task.allowed_paths)
            if isinstance(agent_patch, str):
                candidate_paths = ["submission/model.patch"]
            else:
                candidate_paths = sorted(agent_files)
            candidate_artifacts = {}
            for path in candidate_paths:
                target = work_dir / path
                if target.is_file():
                    candidate_artifacts[path] = target.read_text(encoding="utf-8")
            safe_artifacts = {
                path: content
                for path, content in candidate_artifacts.items()
                if whitelist.is_allowed(path)
            }
            patch_hash = _hash_artifacts(safe_artifacts)

            _write_files(judge_dir, initial_files)
            judge_cache_key = _judge_cache_key(task.task_id, patch_hash, toy_repo)
            plugin_scan = _resolve_pytest_plugin_policy(
                judge_dir,
                toy_repo.get("hidden_judge", []),
                self.pytest_plugin_policy,
            )
            judge_start = time.perf_counter()
            lookup_start = time.perf_counter()
            with self._judge_cache_lock:
                cached_verdict = self._judge_cache.get(
                    _policy_aware_judge_cache_key(judge_cache_key, plugin_scan)
                )
            judge_cache_lookup_time = time.perf_counter() - lookup_start
            judge_cache_hit = cached_verdict is not None
            patch_apply_time = 0.0
            hidden_test_time = 0.0
            if judge_cache_hit:
                judge_passed, failure_type, judge_details = cached_verdict
            else:
                try:
                    judge_details_prefix: list[dict[str, Any]] = []
                    if isinstance(agent_patch, str):
                        patch_text = safe_artifacts.get("submission/model.patch")
                        if patch_text is None:
                            raise ValueError("patch artifact rejected by path whitelist")
                        patch_apply_start = time.perf_counter()
                        touched_paths = _apply_unified_diff(
                            judge_dir,
                            patch_text,
                            _patch_whitelist(task, toy_repo),
                        )
                        patch_apply_time = time.perf_counter() - patch_apply_start
                        judge_details_prefix.append(
                            {
                                "kind": "patch_apply",
                                "passed": True,
                                "paths": sorted(touched_paths),
                                "time_sec": patch_apply_time,
                                "failure_type": None,
                            }
                        )
                    else:
                        _write_files(judge_dir, safe_artifacts)
                    hidden_test_start = time.perf_counter()
                    judge_passed, failure_type, judge_details = _run_hidden_rules(
                        judge_dir,
                        toy_repo.get("hidden_judge", []),
                        pytest_diagnostics=self.pytest_diagnostics,
                        pytest_timeout_sec=self.pytest_timeout_sec,
                        disable_pytest_plugin_autoload=plugin_scan["disable_autoload"],
                        pytest_plugin_scan=plugin_scan,
                    )
                    hidden_test_time = time.perf_counter() - hidden_test_start
                    judge_details = judge_details_prefix + judge_details
                except Exception as exc:
                    judge_passed = False
                    failure_type = "patch_apply_error"
                    judge_details = [
                        {
                            "kind": "patch_apply",
                            "passed": False,
                            "error": repr(exc),
                            "failure_type": failure_type,
                        }
                    ]
                with self._judge_cache_lock:
                    cache_key = _policy_aware_judge_cache_key(judge_cache_key, plugin_scan)
                    self._judge_cache[cache_key] = (judge_passed, failure_type, judge_details)
            judge_time = time.perf_counter() - judge_start

        metadata = {
            "local_runner": True,
            "toy_repo_runner": True,
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
            "repo_checkout_time_sec": 0.0,
            "memory_peak_mb": None,
            "cache_hit_flags": {
                "image": False,
                "repo_snapshot": snapshot_hit,
                "repo_map": snapshot_hit,
                "llm_prompt": True,
                "judge_verdict": judge_cache_hit,
            },
            "benchmark_environment": _benchmark_environment(),
        }
        return SandboxResult(artifacts=safe_artifacts, metadata=metadata)

    def _run_git_task(self, task: TaskSpec, git_repo: dict[str, Any]) -> SandboxResult:
        start = time.perf_counter()
        repo_path = Path(str(git_repo["path"])).expanduser().resolve()
        base_commit = str(git_repo.get("base_commit", "HEAD"))
        repo_key = f"{repo_path}:{base_commit}"
        with self._workspace_cache_lock:
            snapshot_hit = repo_key in self._snapshot_cache
            self._snapshot_cache.add(repo_key)

        tempdir_create_start = time.perf_counter()
        temp_root = Path(tempfile.mkdtemp(prefix=f"sebench-git-{task.task_id}-"))
        tempdir_create_time = time.perf_counter() - tempdir_create_start
        cleanup_time = 0.0
        checkout: CheckoutResult | None = None
        judge_dir: Path | None = None

        try:
            work_dir = temp_root / "work"
            requested_judge_dir = temp_root / "judge"
            work_dir.mkdir()

            agent_patch = str(git_repo.get("agent_patch", ""))
            agent_start = time.perf_counter()
            _write_files(work_dir, {"submission/model.patch": agent_patch})
            agent_step_latency = time.perf_counter() - agent_start

            whitelist = PathWhitelist(task.allowed_paths)
            candidate_artifacts = {
                "submission/model.patch": (work_dir / "submission/model.patch").read_text(
                    encoding="utf-8"
                )
            }
            safe_artifacts = {
                path: content
                for path, content in candidate_artifacts.items()
                if whitelist.is_allowed(path)
            }
            patch_hash = _hash_artifacts(safe_artifacts)

            checkout = self._prepare_git_checkout(
                repo_path,
                requested_judge_dir,
                base_commit,
            )
            judge_dir = checkout.path

            judge_cache_key = _judge_cache_key(task.task_id, patch_hash, git_repo)
            plugin_scan = _resolve_pytest_plugin_policy(
                judge_dir,
                git_repo.get("hidden_judge", []),
                self.pytest_plugin_policy,
            )
            judge_start = time.perf_counter()
            lookup_start = time.perf_counter()
            with self._judge_cache_lock:
                cached_verdict = self._judge_cache.get(
                    _policy_aware_judge_cache_key(judge_cache_key, plugin_scan)
                )
            judge_cache_lookup_time = time.perf_counter() - lookup_start
            judge_cache_hit = cached_verdict is not None
            patch_apply_time = 0.0
            hidden_test_time = 0.0
            if judge_cache_hit:
                judge_passed, failure_type, judge_details = cached_verdict
            else:
                try:
                    patch_text = safe_artifacts.get("submission/model.patch")
                    if patch_text is None:
                        raise ValueError("patch artifact rejected by path whitelist")
                    patch_apply_start = time.perf_counter()
                    touched_paths = _apply_unified_diff(
                        judge_dir,
                        patch_text,
                        _patch_whitelist(task, git_repo),
                    )
                    patch_apply_time = time.perf_counter() - patch_apply_start
                    hidden_test_start = time.perf_counter()
                    judge_passed, failure_type, judge_details = _run_hidden_rules(
                        judge_dir,
                        git_repo.get("hidden_judge", []),
                        pytest_diagnostics=self.pytest_diagnostics,
                        pytest_timeout_sec=self.pytest_timeout_sec,
                        disable_pytest_plugin_autoload=plugin_scan["disable_autoload"],
                        pytest_plugin_scan=plugin_scan,
                    )
                    hidden_test_time = time.perf_counter() - hidden_test_start
                    judge_details = [
                        {
                            "kind": "patch_apply",
                            "passed": True,
                            "paths": sorted(touched_paths),
                            "time_sec": patch_apply_time,
                            "failure_type": None,
                        },
                        *judge_details,
                    ]
                except Exception as exc:
                    judge_passed = False
                    failure_type = "patch_apply_error"
                    judge_details = [
                        {
                            "kind": "patch_apply",
                            "passed": False,
                            "error": repr(exc),
                            "failure_type": failure_type,
                        }
                    ]
                with self._judge_cache_lock:
                    cache_key = _policy_aware_judge_cache_key(judge_cache_key, plugin_scan)
                    self._judge_cache[cache_key] = (judge_passed, failure_type, judge_details)
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

        metadata = {
            "local_runner": True,
            "git_repo_runner": True,
            "checkout_strategy": self.checkout_strategy,
            "cache_policy": self.cache_policy,
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
            "repo_checkout_time_sec": checkout.timings["repo_checkout_time_sec"],
            "git_clone_time_sec": checkout.timings["git_clone_time_sec"],
            "git_checkout_time_sec": checkout.timings["git_checkout_time_sec"],
            "snapshot_materialize_time_sec": checkout.timings[
                "snapshot_materialize_time_sec"
            ],
            "cache_lock_wait_time_sec": checkout.timings["cache_lock_wait_time_sec"],
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
        }
        return SandboxResult(artifacts=safe_artifacts, metadata=metadata)

    def _prepare_git_checkout(
        self,
        repo_path: Path,
        checkout_dir: Path,
        base_commit: str,
    ) -> CheckoutResult:
        if self.checkout_strategy == "clone":
            timings = _clone_git_repo(repo_path, checkout_dir, base_commit)
            return CheckoutResult(path=checkout_dir, timings=timings)
        if self.checkout_strategy == "worktree":
            if self.cache_policy == "shared":
                return self._prepare_shared_persistent_checkout(repo_path, base_commit)
            return self._prepare_persistent_checkout(repo_path, base_commit)
        if self.checkout_strategy == "copytree":
            snapshot_dir, prep_timings = self._ensure_snapshot_dir(repo_path, base_commit)
            materialize_start = time.perf_counter()
            shutil.copytree(snapshot_dir, checkout_dir)
            materialize_time = time.perf_counter() - materialize_start
            timings = _merge_checkout_timings(
                prep_timings,
                {"snapshot_materialize_time_sec": materialize_time},
            )
            return CheckoutResult(path=checkout_dir, timings=timings)
        if self.checkout_strategy == "tar":
            tar_path, prep_timings = self._ensure_tar_snapshot(repo_path, base_commit)
            materialize_start = time.perf_counter()
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(checkout_dir.parent)
            materialize_time = time.perf_counter() - materialize_start
            timings = _merge_checkout_timings(
                prep_timings,
                {"snapshot_materialize_time_sec": materialize_time},
            )
            return CheckoutResult(path=checkout_dir, timings=timings)
        raise AssertionError(f"unhandled checkout strategy: {self.checkout_strategy}")

    def _prepare_persistent_checkout(self, repo_path: Path, base_commit: str) -> CheckoutResult:
        cache_key = _stable_key(f"{repo_path}:{base_commit}:{threading.get_ident()}")
        checkout_dir = self._cache_root / f"worktree-{cache_key}"
        timings = _empty_checkout_timings()
        with self._workspace_cache_lock:
            if cache_key not in self._persistent_checkouts:
                timings = _clone_git_repo(repo_path, checkout_dir, base_commit)
                self._persistent_checkouts[cache_key] = checkout_dir

        reset_start = time.perf_counter()
        _run_git(["git", "reset", "--hard", base_commit], checkout_dir, timeout=10)
        _run_git(["git", "clean", "-fdx"], checkout_dir, timeout=10)
        reset_time = time.perf_counter() - reset_start
        timings = _merge_checkout_timings(timings, {"git_checkout_time_sec": reset_time})
        return CheckoutResult(path=checkout_dir, timings=timings)

    def _prepare_shared_persistent_checkout(
        self,
        repo_path: Path,
        base_commit: str,
    ) -> CheckoutResult:
        cache_key = shared_checkout_cache_key(repo_path, base_commit, self.checkout_strategy)
        mirror_dir = self.shared_cache_root / "mirrors" / f"{cache_key}.git"
        checkout_dir = self.shared_cache_root / "worktrees" / f"{cache_key}-{os.getpid()}"
        lock_path = self.shared_cache_root / "locks" / f"{cache_key}.lock"
        timings = _empty_checkout_timings()

        lock_wait_start = time.perf_counter()
        with _file_lock(lock_path):
            timings["cache_lock_wait_time_sec"] += time.perf_counter() - lock_wait_start
            if not mirror_dir.exists():
                mirror_dir.parent.mkdir(parents=True, exist_ok=True)
                tmp_dir = mirror_dir.with_name(f"{mirror_dir.name}.tmp-{os.getpid()}")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                clone_start = time.perf_counter()
                _run_git(
                    ["git", "clone", "--quiet", "--mirror", str(repo_path), str(tmp_dir)],
                    cwd=None,
                    timeout=20,
                )
                os.replace(tmp_dir, mirror_dir)
                timings["git_clone_time_sec"] += time.perf_counter() - clone_start

        if not (checkout_dir / ".git").exists():
            shutil.rmtree(checkout_dir, ignore_errors=True)
            checkout_dir.parent.mkdir(parents=True, exist_ok=True)
            clone_start = time.perf_counter()
            _run_git(
                ["git", "clone", "--quiet", str(mirror_dir), str(checkout_dir)],
                cwd=None,
                timeout=20,
            )
            timings["git_clone_time_sec"] += time.perf_counter() - clone_start

        reset_start = time.perf_counter()
        _run_git(["git", "reset", "--hard", base_commit], checkout_dir, timeout=10)
        _run_git(["git", "clean", "-fdx"], checkout_dir, timeout=10)
        timings["git_checkout_time_sec"] += time.perf_counter() - reset_start
        timings["repo_checkout_time_sec"] = (
            timings["git_clone_time_sec"]
            + timings["git_checkout_time_sec"]
            + timings["snapshot_materialize_time_sec"]
            + timings["cache_lock_wait_time_sec"]
        )
        return CheckoutResult(path=checkout_dir, timings=timings)

    def _ensure_snapshot_dir(
        self,
        repo_path: Path,
        base_commit: str,
    ) -> tuple[Path, dict[str, float]]:
        cache_key = _stable_key(f"{repo_path}:{base_commit}:snapshot")
        snapshot_dir = self._cache_root / f"snapshot-{cache_key}"
        with self._workspace_cache_lock:
            if cache_key in self._snapshot_dirs:
                return self._snapshot_dirs[cache_key], _empty_checkout_timings()
            timings = _clone_git_repo(repo_path, snapshot_dir, base_commit)
            self._snapshot_dirs[cache_key] = snapshot_dir
            return snapshot_dir, timings

    def _ensure_tar_snapshot(
        self,
        repo_path: Path,
        base_commit: str,
    ) -> tuple[Path, dict[str, float]]:
        cache_key = _stable_key(f"{repo_path}:{base_commit}:tar")
        tar_path = self._cache_root / f"snapshot-{cache_key}.tar"
        with self._workspace_cache_lock:
            if cache_key in self._tar_paths:
                return self._tar_paths[cache_key], _empty_checkout_timings()
            snapshot_dir, timings = self._ensure_snapshot_dir(repo_path, base_commit)
            with tarfile.open(tar_path, "w") as tar:
                tar.add(snapshot_dir, arcname="judge")
            self._tar_paths[cache_key] = tar_path
            return tar_path, timings


class DockerSandbox:
    """Container runner placeholder following the Work/Judge split."""

    def __init__(self, work_image: str, judge_image: str, timeout_seconds: int = 120) -> None:
        self.work_image = work_image
        self.judge_image = judge_image
        self.timeout_seconds = timeout_seconds

    def run_task(self, task: TaskSpec) -> SandboxResult:
        try:
            import docker
        except ImportError as exc:
            raise RuntimeError("Install docker SDK to use docker runner mode") from exc

        client = docker.from_env()
        command = ["python", "-m", "sebench_work", task.model_dump_json()]
        output = client.containers.run(
            self.work_image,
            command=command,
            detach=False,
            remove=True,
            network_disabled=True,
            stdout=True,
            stderr=True,
            mem_limit="1g",
        )
        answer = output.decode("utf-8", errors="replace")
        artifacts = {"submission/answer.txt": answer}
        whitelist = PathWhitelist(task.allowed_paths)
        return SandboxResult(
            artifacts={
                path: content
                for path, content in artifacts.items()
                if whitelist.is_allowed(path)
            },
            metadata={
                "docker_runner": True,
                "judge_image": self.judge_image,
                "patch_hash": _hash_artifacts(artifacts),
            },
        )


def _hash_artifacts(artifacts: dict[str, str]) -> str:
    payload = json.dumps(artifacts, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _judge_cache_key(task_id: str, patch_hash: str, toy_repo: dict[str, Any]) -> str:
    rules = toy_repo.get("hidden_judge", [])
    payload = json.dumps(
        {"task_id": task_id, "patch_hash": patch_hash, "hidden_judge": rules},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _policy_aware_judge_cache_key(base_key: str, plugin_scan: dict[str, Any]) -> str:
    payload = {
        "base_key": base_key,
        "pytest_plugin_policy": plugin_scan.get("policy"),
        "disable_autoload": plugin_scan.get("disable_autoload"),
        "plugin_required": plugin_scan.get("plugin_required"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _benchmark_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "benchmark_backend": "apple_mlx_local" if platform.system() == "Darwin" else "local_toy",
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "pytorch_mps_used": False,
    }
    try:
        import mlx.core as mx

        env["mlx_version"] = getattr(mx, "__version__", "unknown")
    except Exception:
        env["mlx_version"] = None
    try:
        import torch

        env["torch_version"] = torch.__version__
    except Exception:
        env["torch_version"] = None
    return env


def _normalize_pytest_plugin_policy(
    policy: str | None,
    disable_pytest_plugin_autoload: bool,
) -> str:
    if policy is None:
        return "enabled" if not disable_pytest_plugin_autoload else "auto"
    if policy not in {"auto", "disabled", "enabled"}:
        raise ValueError("pytest_plugin_policy must be one of: auto, disabled, enabled")
    return policy


def _resolve_pytest_plugin_policy(
    repo_root: Path,
    rules: Any,
    policy: str,
) -> dict[str, Any]:
    scan = _scan_pytest_plugin_dependencies(repo_root, rules)
    if policy == "disabled":
        disable_autoload = True
        effective_policy = "disabled"
    elif policy == "enabled":
        disable_autoload = False
        effective_policy = "enabled"
    else:
        disable_autoload = not scan["plugin_required"]
        effective_policy = "auto_disabled" if disable_autoload else "auto_enabled"
    return {
        "policy": policy,
        "effective_policy": effective_policy,
        "disable_autoload": disable_autoload,
        "plugin_required": scan["plugin_required"],
        "reasons": scan["reasons"],
        "scanned_files": scan["scanned_files"],
    }


def _scan_pytest_plugin_dependencies(repo_root: Path, rules: Any) -> dict[str, Any]:
    reasons: list[str] = []
    scanned_files: list[str] = []
    for path in _candidate_pytest_config_files(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        scanned_files.append(rel)
        _extend_plugin_reasons(reasons, rel, _read_text_limited(path))
    if isinstance(rules, list):
        for index, rule in enumerate(rules):
            if not isinstance(rule, dict) or rule.get("kind") != "pytest":
                continue
            files = _string_map(rule.get("files", {}))
            for rel, content in files.items():
                label = f"hidden_judge[{index}]:{rel}"
                scanned_files.append(label)
                _extend_plugin_reasons(reasons, label, content)
            args = " ".join(str(arg) for arg in rule.get("args", []))
            if args:
                _extend_plugin_reasons(reasons, f"hidden_judge[{index}]:args", args)
    return {
        "plugin_required": bool(reasons),
        "reasons": sorted(set(reasons)),
        "scanned_files": sorted(set(scanned_files)),
    }


def _candidate_pytest_config_files(repo_root: Path) -> list[Path]:
    names = {"pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini", "setup.py"}
    paths: list[Path] = []
    for path in repo_root.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        name = path.name
        if (
            name in names
            or name == "conftest.py"
            or (name.startswith("requirements") and path.suffix in {".txt", ".in"})
        ):
            paths.append(path)
    return paths


def _read_text_limited(path: Path, limit_bytes: int = 512_000) -> str:
    try:
        payload = path.read_bytes()[:limit_bytes]
    except OSError:
        return ""
    return payload.decode("utf-8", errors="ignore")


def _extend_plugin_reasons(reasons: list[str], source: str, text: str) -> None:
    lowered = text.lower()
    if "pytest_plugins" in lowered:
        reasons.append(f"{source}: pytest_plugins declaration")
    for match in re.finditer(r"(?i)\bpytest[-_][a-z0-9_.-]+", text):
        plugin = match.group(0).lower().replace("_", "-")
        if plugin not in {"pytest", "pytest-disable-plugin-autoload"}:
            reasons.append(f"{source}: dependency {plugin}")
    flag_patterns = {
        "--asyncio-mode": "pytest-asyncio option",
        "pytest.mark.asyncio": "pytest-asyncio mark",
        "--ds": "pytest-django option",
        "django_settings_module": "pytest-django setting",
        "pytest.mark.django_db": "pytest-django mark",
        "--cov": "pytest-cov option",
        "--cov-report": "pytest-cov option",
        "--numprocesses": "pytest-xdist option",
        " -n ": "pytest-xdist short option",
        "--dist": "pytest-xdist option",
        "--benchmark": "pytest-benchmark option",
        "--mypy": "pytest-mypy option",
    }
    padded = f" {lowered} "
    for marker, reason in flag_patterns.items():
        haystack = padded if marker.startswith(" ") else lowered
        if marker in haystack:
            reasons.append(f"{source}: {reason}")


def _string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(path): str(content) for path, content in value.items()}


def _write_files(root: Path, files: dict[str, str]) -> None:
    for rel_path, content in files.items():
        pure = PurePosixPath(rel_path)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"unsafe toy file path: {rel_path}")
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _cleanup_hidden_judge_artifacts(root: Path, rules: Any) -> None:
    if not isinstance(rules, list):
        return
    for rule in rules:
        if not isinstance(rule, dict) or rule.get("kind") != "pytest":
            continue
        for rel_path in _string_map(rule.get("files", {})):
            pure = PurePosixPath(rel_path)
            if pure.is_absolute() or ".." in pure.parts:
                continue
            target = root / rel_path
            with contextlib.suppress(FileNotFoundError):
                target.unlink()
            _remove_empty_parents(target.parent, root)

    shutil.rmtree(root / ".pytest_cache", ignore_errors=True)
    for pycache_dir in list(root.rglob("__pycache__")):
        shutil.rmtree(pycache_dir, ignore_errors=True)


def _remove_empty_parents(path: Path, stop: Path) -> None:
    try:
        current = path.resolve()
        boundary = stop.resolve()
    except FileNotFoundError:
        current = path
        boundary = stop
    while current != boundary and boundary in current.parents:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _empty_checkout_timings() -> dict[str, float]:
    return {
        "repo_checkout_time_sec": 0.0,
        "git_clone_time_sec": 0.0,
        "git_checkout_time_sec": 0.0,
        "snapshot_materialize_time_sec": 0.0,
        "cache_lock_wait_time_sec": 0.0,
    }


def _merge_checkout_timings(
    base: dict[str, float],
    extra: dict[str, float],
) -> dict[str, float]:
    timings = _empty_checkout_timings()
    for source in (base, extra):
        for key, value in source.items():
            timings[key] = timings.get(key, 0.0) + float(value)
    timings["repo_checkout_time_sec"] = (
        timings["git_clone_time_sec"]
        + timings["git_checkout_time_sec"]
        + timings["snapshot_materialize_time_sec"]
        + timings["cache_lock_wait_time_sec"]
    )
    return timings


def _stable_key(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def shared_checkout_cache_key(
    repo_path: Path,
    base_commit: str,
    checkout_strategy: str,
    schema_version: str = "v1",
) -> str:
    return _stable_key(
        f"{schema_version}:{repo_path.expanduser().resolve()}:{base_commit}:{checkout_strategy}"
    )


def _resolve_shared_cache_root(root: str | Path) -> Path:
    path = Path(root)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("shared_cache_root must be a relative path without '..'")
    resolved = Path.cwd() / path
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


@contextlib.contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _clone_git_repo(repo_path: Path, checkout_dir: Path, base_commit: str) -> dict[str, float]:
    clone_start = time.perf_counter()
    _run_git(
        ["git", "clone", "--quiet", "--no-hardlinks", str(repo_path), str(checkout_dir)],
        cwd=None,
        timeout=20,
    )
    clone_time = time.perf_counter() - clone_start
    checkout_start = time.perf_counter()
    _run_git(["git", "checkout", "--quiet", base_commit], checkout_dir, timeout=10)
    checkout_time = time.perf_counter() - checkout_start
    return _merge_checkout_timings(
        {},
        {
            "git_clone_time_sec": clone_time,
            "git_checkout_time_sec": checkout_time,
        },
    )


def _run_git(args: list[str], cwd: Path | None, timeout: float) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        args,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{' '.join(args)} failed: stdout={completed.stdout.strip()} "
            f"stderr={completed.stderr.strip()}"
        )
    return completed


def _patch_whitelist(task: TaskSpec, toy_repo: dict[str, Any]) -> PathWhitelist:
    raw_paths = toy_repo.get("patch_allowed_paths")
    if isinstance(raw_paths, list) and raw_paths:
        return PathWhitelist([str(path) for path in raw_paths])
    return PathWhitelist(task.expected_artifacts)


def _apply_unified_diff(judge_dir: Path, patch_text: str, whitelist: PathWhitelist) -> set[str]:
    touched_paths = _paths_in_unified_diff(patch_text)
    if not touched_paths:
        raise ValueError("patch does not modify any tracked path")
    for path in touched_paths:
        if not whitelist.is_allowed(path):
            raise ValueError(f"patch modifies disallowed path: {path}")

    apply_commands = [
        ["git", "apply", "--check", "--whitespace=nowarn", "-"],
        ["git", "apply", "--whitespace=nowarn", "-"],
    ]
    for args in apply_commands:
        completed = subprocess.run(
            args,
            cwd=judge_dir,
            input=patch_text,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "git apply failed: "
                f"stdout={completed.stdout.strip()} stderr={completed.stderr.strip()}"
            )
    return touched_paths


def _paths_in_unified_diff(patch_text: str) -> set[str]:
    paths: set[str] = set()
    for line in patch_text.splitlines():
        if line.startswith("+++ "):
            path = _normalize_diff_path(line[4:].strip())
        elif line.startswith("--- "):
            path = _normalize_diff_path(line[4:].strip())
        else:
            continue
        if path is not None:
            paths.add(path)
    return paths


def _normalize_diff_path(raw_path: str) -> str | None:
    path = raw_path.split("\t", 1)[0]
    if path == "/dev/null":
        return None
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe patch path: {raw_path}")
    return path


def _run_hidden_rules(
    judge_dir: Path,
    rules: Any,
    pytest_diagnostics: bool = False,
    pytest_timeout_sec: float | None = None,
    disable_pytest_plugin_autoload: bool = False,
    pytest_plugin_scan: dict[str, Any] | None = None,
) -> tuple[bool, str | None, list[dict[str, Any]]]:
    if not isinstance(rules, list):
        return False, "judge_config_error", [{"error": "hidden_judge must be a list"}]

    details: list[dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            return False, "judge_config_error", [{"error": "hidden rule must be an object"}]
        try:
            passed, detail = _run_hidden_rule(
                judge_dir,
                rule,
                pytest_diagnostics=pytest_diagnostics,
                pytest_timeout_sec=pytest_timeout_sec,
                disable_pytest_plugin_autoload=disable_pytest_plugin_autoload,
                pytest_plugin_scan=pytest_plugin_scan,
            )
        except subprocess.TimeoutExpired as exc:
            passed = False
            detail = {
                "kind": rule.get("kind"),
                "error": repr(exc),
                "timeout_sec": (
                    pytest_timeout_sec
                    if rule.get("kind") == "pytest" and pytest_timeout_sec is not None
                    else rule.get("timeout_sec")
                ),
                "failure_type": "hidden_judge_timeout",
            }
        except AssertionError as exc:
            passed = False
            detail = {
                "kind": rule.get("kind"),
                "error": str(exc) or repr(exc),
                "failure_type": "wrong_edit",
            }
        except Exception as exc:
            passed = False
            detail = {
                "kind": rule.get("kind"),
                "error": repr(exc),
                "failure_type": "hidden_judge_error",
            }
        details.append(detail)
        if not passed:
            return False, str(detail.get("failure_type", "hidden_judge_failed")), details
    return True, None, details


def _run_hidden_rule(
    judge_dir: Path,
    rule: dict[str, Any],
    pytest_diagnostics: bool = False,
    pytest_timeout_sec: float | None = None,
    disable_pytest_plugin_autoload: bool = False,
    pytest_plugin_scan: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    kind = rule.get("kind")
    if kind == "python_function":
        rel_path = str(rule["path"])
        module = _load_python_module(judge_dir / rel_path)
        function = getattr(module, str(rule["function"]))
        actual = function(*rule.get("args", []))
        expected = rule.get("expected")
        passed = actual == expected
        return passed, {
            "kind": kind,
            "passed": passed,
            "actual": actual,
            "expected": expected,
            "failure_type": None if passed else "wrong_edit",
        }

    if kind == "cli_stdout_contains":
        command = [str(part) for part in rule["command"]]
        if command and command[0] == "python":
            command[0] = sys.executable
        completed = subprocess.run(
            command,
            cwd=judge_dir,
            text=True,
            capture_output=True,
            timeout=float(rule.get("timeout_sec", 5)),
            check=False,
        )
        expected = str(rule["expected"])
        passed = completed.returncode == 0 and expected in completed.stdout
        return passed, {
            "kind": kind,
            "passed": passed,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "expected": expected,
            "failure_type": None if passed else "wrong_edit",
        }

    if kind == "python_script_stdout_contains":
        rel_path = str(rule["path"])
        args = [str(arg) for arg in rule.get("args", [])]
        expected = str(rule["expected"])
        stdout = _run_python_script_in_process(judge_dir, judge_dir / rel_path, args)
        passed = expected in stdout
        return passed, {
            "kind": kind,
            "passed": passed,
            "stdout": stdout.strip(),
            "expected": expected,
            "failure_type": None if passed else "wrong_edit",
        }

    if kind == "python_inline_tests":
        code = str(rule["code"])
        completed = _run_python_inline_tests(judge_dir, code, float(rule.get("timeout_sec", 5)))
        passed = completed.returncode == 0
        failure_type = None
        if not passed:
            failure_type = (
                "wrong_edit" if "AssertionError" in completed.stderr else "hidden_judge_error"
            )
        return passed, {
            "kind": kind,
            "passed": passed,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "failure_type": failure_type,
        }

    if kind == "pytest":
        test_files = _string_map(rule.get("files", {}))
        file_write_start = time.perf_counter()
        _write_files(judge_dir, test_files)
        file_write_time = time.perf_counter() - file_write_start
        args = _normalized_pytest_args(rule.get("args", ["-q"]))
        timeout_sec = (
            float(pytest_timeout_sec)
            if pytest_timeout_sec is not None
            else float(rule.get("timeout_sec", 10))
        )
        python_startup_time = 0.0
        pytest_startup_time = 0.0
        pytest_collection_time = 0.0
        pytest_subprocess_count = 1
        pytest_env = _pytest_env_overrides(disable_pytest_plugin_autoload)
        if pytest_diagnostics:
            pytest_subprocess_count += 3
            _, python_startup_time = _run_timed_command(
                [sys.executable, "-c", "pass"],
                judge_dir,
                timeout_sec,
            )
            _, pytest_startup_time = _run_timed_command(
                [sys.executable, "-m", "pytest", "--version"],
                judge_dir,
                timeout_sec,
                extra_env=pytest_env,
            )
            _, pytest_collection_time = _run_timed_command(
                [sys.executable, "-m", "pytest", *args, "--collect-only"],
                judge_dir,
                timeout_sec,
                extra_env=pytest_env,
            )
        completed, pytest_execution_time = _run_pytest(
            judge_dir,
            args,
            timeout_sec,
            extra_env=pytest_env,
        )
        passed = completed.returncode == 0
        failure_type = None
        if not passed:
            failure_type = "wrong_edit" if completed.returncode == 1 else "hidden_judge_error"
        pytest_total_time = (
            python_startup_time
            + pytest_startup_time
            + pytest_collection_time
            + pytest_execution_time
        )
        return passed, {
            "kind": kind,
            "passed": passed,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "args": args,
            "timeout_sec": timeout_sec,
            "disable_pytest_plugin_autoload": disable_pytest_plugin_autoload,
            "pytest_plugin_scan": pytest_plugin_scan or {},
            "hidden_test_file_write_time_sec": file_write_time,
            "python_subprocess_startup_time_sec": python_startup_time,
            "pytest_process_startup_time_sec": pytest_startup_time,
            "pytest_collection_time_sec": pytest_collection_time,
            "pytest_execution_time_sec": pytest_execution_time,
            "pytest_total_time_sec": pytest_total_time,
            "pytest_subprocess_count": pytest_subprocess_count,
            "failure_type": failure_type,
        }

    if kind == "json_value":
        payload = json.loads((judge_dir / str(rule["path"])).read_text(encoding="utf-8"))
        actual: Any = payload
        for part in str(rule["json_path"]).split("."):
            actual = actual[part]
        expected = rule.get("expected")
        passed = actual == expected
        return passed, {
            "kind": kind,
            "passed": passed,
            "actual": actual,
            "expected": expected,
            "failure_type": None if passed else "format_error",
        }

    return False, {"kind": kind, "passed": False, "failure_type": "judge_config_error"}


def _load_python_module(path: Path) -> Any:
    module_name = f"sebench_toy_{hashlib.sha1(str(path).encode()).hexdigest()[:10]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    with _python_execution_context(path.parent):
        spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _python_execution_context(root: Path):
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    old_path = sys.path[:]
    try:
        _clear_repo_package_modules(root)
        os.chdir(root)
        sys.path.insert(0, str(root))
        yield
    finally:
        sys.argv = old_argv
        sys.path = old_path
        os.chdir(old_cwd)


def _clear_repo_package_modules(root: Path) -> None:
    package_names = [
        path.name for path in root.iterdir() if path.is_dir() and (path / "__init__.py").is_file()
    ]
    for package_name in package_names:
        for module_name in list(sys.modules):
            if module_name == package_name or module_name.startswith(f"{package_name}."):
                del sys.modules[module_name]


def _run_python_script_in_process(root: Path, path: Path, args: list[str]) -> str:
    buffer = io.StringIO()
    with _python_execution_context(root):
        sys.argv = [str(path), *args]
        code = path.read_text(encoding="utf-8")
        namespace = {"__name__": "__main__", "__file__": str(path)}
        with contextlib.redirect_stdout(buffer):
            exec(compile(code, str(path), "exec"), namespace)
    return buffer.getvalue()


def _run_python_inline_tests(
    root: Path,
    code: str,
    timeout_sec: float,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(root) if not existing_pythonpath else f"{root}{os.pathsep}{existing_pythonpath}"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )


def _normalized_pytest_args(raw_args: Any) -> list[str]:
    args = [str(arg) for arg in raw_args] if isinstance(raw_args, list) else ["-q"]
    if not any(arg in {"-q", "--quiet"} for arg in args):
        args.insert(0, "-q")
    if not any(arg == "--tb" or arg.startswith("--tb=") for arg in args):
        args.append("--tb=short")
    if "--disable-warnings" not in args:
        args.append("--disable-warnings")
    if not any(arg == "--maxfail" or arg.startswith("--maxfail=") or arg == "-x" for arg in args):
        args.append("--maxfail=1")
    return args


def _run_timed_command(
    command: list[str],
    root: Path,
    timeout_sec: float,
    extra_env: dict[str, str | None] | None = None,
) -> tuple[subprocess.CompletedProcess, float]:
    env = os.environ.copy()
    if extra_env:
        for key, value in extra_env.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(root) if not existing_pythonpath else f"{root}{os.pathsep}{existing_pythonpath}"
    )
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )
    return completed, time.perf_counter() - start


def _run_pytest(
    root: Path,
    args: list[str],
    timeout_sec: float,
    extra_env: dict[str, str | None] | None = None,
) -> tuple[subprocess.CompletedProcess, float]:
    return _run_timed_command(
        [sys.executable, "-m", "pytest", *args],
        root,
        timeout_sec,
        extra_env=extra_env,
    )


def _pytest_env_overrides(disable_plugin_autoload: bool) -> dict[str, str | None]:
    if disable_plugin_autoload:
        return {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    return {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": None}


def _aggregate_detail_timings(details: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "hidden_test_file_write_time_sec",
        "python_subprocess_startup_time_sec",
        "pytest_process_startup_time_sec",
        "pytest_collection_time_sec",
        "pytest_execution_time_sec",
        "pytest_total_time_sec",
    ]
    return {
        key: sum(float(detail.get(key, 0.0)) for detail in details if isinstance(detail, dict))
        for key in keys
    }


def _aggregate_pytest_subprocess_count(details: list[dict[str, Any]]) -> int:
    return sum(
        int(detail.get("pytest_subprocess_count", 0))
        for detail in details
        if isinstance(detail, dict)
    )
