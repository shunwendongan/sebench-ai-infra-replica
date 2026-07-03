from __future__ import annotations

import hashlib
import json
import platform
import threading
from typing import Any


class SnapshotHitTracker:
    """Thread-safe tracker for per-run repo snapshot/cache hit flags."""

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self._lock = threading.RLock()

    def mark_seen(self, key: str) -> bool:
        with self._lock:
            hit = key in self._seen
            self._seen.add(key)
            return hit


class JudgeVerdictCache:
    """Thread-safe in-process cache for deterministic judge verdicts."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[bool, str | None, list[dict[str, Any]]]] = {}
        self._lock = threading.Lock()

    def get(
        self,
        base_key: str,
        plugin_scan: dict[str, Any],
    ) -> tuple[bool, str | None, list[dict[str, Any]]] | None:
        with self._lock:
            return self._cache.get(_policy_aware_judge_cache_key(base_key, plugin_scan))

    def set(
        self,
        base_key: str,
        plugin_scan: dict[str, Any],
        verdict: tuple[bool, str | None, list[dict[str, Any]]],
    ) -> None:
        with self._lock:
            self._cache[_policy_aware_judge_cache_key(base_key, plugin_scan)] = verdict


def _hash_artifacts(artifacts: dict[str, str]) -> str:
    payload = json.dumps(artifacts, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _judge_cache_key(task_id: str, patch_hash: str, fixture: dict[str, Any]) -> str:
    rules = fixture.get("hidden_judge", [])
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
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


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


def _stable_key(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
