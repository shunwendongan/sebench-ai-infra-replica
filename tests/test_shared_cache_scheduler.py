import subprocess
import sys
from pathlib import Path

import pytest

from sebench_infra.benchmark.schemas import DatasetSpec
from sebench_infra.orchestrator import EvaluationOrchestrator
from sebench_infra.orchestrator.sandbox import LocalSandbox, shared_checkout_cache_key


def _small_git_dataset(task_count: int = 8) -> DatasetSpec:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "examples/git_pytest_benchmark.json"
    if not manifest.exists():
        subprocess.run(
            [
                sys.executable,
                "scripts/build_git_pytest_benchmark.py",
                "--tasks",
                "128",
                "--out",
                str(manifest),
            ],
            cwd=root,
            check=True,
        )
    dataset = DatasetSpec.model_validate_json(manifest.read_text(encoding="utf-8"))
    tasks = []
    for task in dataset.tasks[:task_count]:
        payload = task.model_dump(mode="json")
        git_repo = payload["fixtures"].get("git_repo")
        if isinstance(git_repo, dict):
            repo_path = Path(str(git_repo["path"]))
            if not repo_path.is_absolute():
                git_repo["path"] = str((root / repo_path).resolve())
        tasks.append(type(task).model_validate(payload))
    return dataset.model_copy(update={"tasks": tasks})


def test_shared_checkout_cache_key_is_stable_and_commit_scoped(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    key_a = shared_checkout_cache_key(repo, "commit-a", "worktree")
    key_a_again = shared_checkout_cache_key(repo, "commit-a", "worktree")
    key_b = shared_checkout_cache_key(repo, "commit-b", "worktree")

    assert key_a == key_a_again
    assert key_a != key_b


def test_shared_cache_root_rejects_absolute_and_escaping_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shared_cache_root"):
        LocalSandbox(cache_policy="shared", shared_cache_root=tmp_path)

    with pytest.raises(ValueError, match="shared_cache_root"):
        LocalSandbox(cache_policy="shared", shared_cache_root="../cache")


def test_shared_cache_worktree_passes_and_cleans_hidden_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = _small_git_dataset()

    report = EvaluationOrchestrator(
        checkout_strategy="worktree",
        cache_policy="shared",
        shared_cache_root="shared-cache",
        task_distribution="repo-shard-worksteal",
    ).run(dataset, workers=4, task_timeout_sec=20)

    assert report.aggregate_score == 1.0
    assert {record.pass_fail for record in report.run_records} == {"pass"}
    assert {record.metadata["cache_policy"] for record in report.run_records} == {"shared"}
    assert {
        record.metadata["task_distribution"] for record in report.run_records
    } == {"repo-shard-worksteal"}
    assert all(record.metrics.pytest_subprocess_count == 1 for record in report.run_records)

    shared_root = tmp_path / "shared-cache"
    assert len(list((shared_root / "mirrors").glob("*.git"))) == 4
    assert list((shared_root / "worktrees").glob("*"))
    assert not list((shared_root / "worktrees").rglob("test_hidden_*.py"))


def test_repo_shard_distribution_keeps_same_repo_on_one_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = _small_git_dataset()

    report = EvaluationOrchestrator(
        checkout_strategy="worktree",
        cache_policy="shared",
        shared_cache_root="shared-cache",
        task_distribution="repo-shard-worksteal",
    ).run(dataset, workers=4, task_timeout_sec=20)

    workers_by_affinity: dict[str, set[int]] = {}
    for record in report.run_records:
        affinity_key = str(record.metadata["affinity_key"])
        worker_id = int(record.metadata["worker_id"])
        workers_by_affinity.setdefault(affinity_key, set()).add(worker_id)

    assert set(workers_by_affinity) == {
        "local-calc",
        "local-config",
        "local-stats",
        "local-text",
    }
    assert all(len(worker_ids) == 1 for worker_ids in workers_by_affinity.values())
    assert not any(record.metadata["repo_shard_stolen"] for record in report.run_records)
