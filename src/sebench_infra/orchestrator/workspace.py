from __future__ import annotations

import contextlib
import fcntl
import os
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from sebench_infra.orchestrator.cache import _stable_key


@dataclass(frozen=True)
class CheckoutResult:
    path: Path
    timings: dict[str, float]


class WorkspaceManager:
    """Prepare and clean local judge workspaces for git-backed tasks."""

    def __init__(
        self,
        *,
        checkout_strategy: str = "worktree",
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
        self._cache_root = Path(tempfile.mkdtemp(prefix="sebench-sandbox-cache-"))
        self._snapshot_dirs: dict[str, Path] = {}
        self._tar_paths: dict[str, Path] = {}
        self._persistent_checkouts: dict[str, Path] = {}
        self._lock = threading.RLock()

    def prepare_git_checkout(
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            if cache_key in self._tar_paths:
                return self._tar_paths[cache_key], _empty_checkout_timings()
            snapshot_dir, timings = self._ensure_snapshot_dir(repo_path, base_commit)
            with tarfile.open(tar_path, "w") as tar:
                tar.add(snapshot_dir, arcname="judge")
            self._tar_paths[cache_key] = tar_path
            return tar_path, timings


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
