from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from sebench_infra.benchmark.regression import RegressionGate
from sebench_infra.benchmark.schemas import (
    BenchmarkEnvironment,
    CacheHitFlags,
    DatasetSpec,
    EvaluationReport,
    RunRecord,
    TaskMetrics,
    TaskRunResult,
    TaskSpec,
)
from sebench_infra.benchmark.scoring import ScoreEngine
from sebench_infra.observability import log_event, observe_run
from sebench_infra.orchestrator.records import new_run_id
from sebench_infra.orchestrator.sandbox import DockerSandbox, LocalSandbox


class EvaluationOrchestrator:
    """Evaluate authored tasks through local or containerized sandboxes."""

    def __init__(
        self,
        runner_mode: str = "local",
        work_image: str = "sebench-work:latest",
        judge_image: str = "sebench-judge:latest",
        timeout_seconds: int = 120,
        checkout_strategy: str = "worktree",
        pytest_diagnostics: bool = False,
        pytest_timeout_sec: float | None = None,
        disable_pytest_plugin_autoload: bool = True,
        pytest_plugin_policy: str | None = None,
        hard_task_timeout: bool = True,
        cache_policy: str = "process",
        shared_cache_root: str = "artifacts/cache/judge_shared",
        task_distribution: str = "load",
    ) -> None:
        self.scorer = ScoreEngine()
        self.regression_gate = RegressionGate()
        self.runner_mode = runner_mode
        self.work_image = work_image
        self.judge_image = judge_image
        self.timeout_seconds = timeout_seconds
        self.checkout_strategy = checkout_strategy
        self.pytest_diagnostics = pytest_diagnostics
        self.pytest_timeout_sec = pytest_timeout_sec
        self.disable_pytest_plugin_autoload = disable_pytest_plugin_autoload
        self.pytest_plugin_policy = pytest_plugin_policy
        self.hard_task_timeout = hard_task_timeout
        if task_distribution not in {"load", "repo-shard", "repo-shard-worksteal"}:
            raise ValueError(
                "task_distribution must be one of: load, repo-shard, repo-shard-worksteal"
            )
        self.cache_policy = cache_policy
        self.shared_cache_root = shared_cache_root
        self.task_distribution = task_distribution
        if runner_mode == "docker":
            self.sandbox = DockerSandbox(work_image, judge_image, timeout_seconds)
        else:
            self.sandbox = LocalSandbox(
                checkout_strategy=checkout_strategy,
                pytest_diagnostics=pytest_diagnostics,
                pytest_timeout_sec=pytest_timeout_sec,
                disable_pytest_plugin_autoload=disable_pytest_plugin_autoload,
                pytest_plugin_policy=pytest_plugin_policy,
                cache_policy=cache_policy,
                shared_cache_root=shared_cache_root,
            )

    def run(
        self,
        dataset: DatasetSpec,
        max_tasks: int | None = None,
        workers: int = 1,
        task_timeout_sec: float | None = None,
    ) -> EvaluationReport:
        run_id = new_run_id()
        tasks = dataset.tasks[:max_tasks] if max_tasks else dataset.tasks
        task_results: list[TaskRunResult] = []
        run_records: list[RunRecord] = []

        with observe_run():
            log_event(
                "evaluation_started",
                run_id=run_id,
                dataset_id=dataset.dataset_id,
                tasks=len(tasks),
                workers=workers,
            )
            workers = max(1, workers)
            if (
                self.runner_mode == "local"
                and self.hard_task_timeout
                and task_timeout_sec is not None
            ):
                evaluated = self._run_local_process_workers(
                    run_id,
                    tasks,
                    workers,
                    task_timeout_sec,
                )
            elif workers == 1:
                evaluated = [self._evaluate_task(run_id, task) for task in tasks]
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(self._evaluate_task, run_id, task) for task in tasks]
                    evaluated = []
                    for future in futures:
                        try:
                            evaluated.append(future.result(timeout=task_timeout_sec))
                        except Exception as exc:
                            task = tasks[len(evaluated)]
                            evaluated.append(self._error_result(run_id, task, exc))

            for result, record in evaluated:
                task_results.append(result)
                run_records.append(record)

        aggregate = (
            sum(result.score for result in task_results) / len(task_results)
            if task_results
            else 0.0
        )
        report = EvaluationReport(
            run_id=run_id,
            dataset_id=dataset.dataset_id,
            task_results=task_results,
            aggregate_score=round(aggregate, 6),
            reward_signal=round(aggregate, 6),
            regression_passed=False,
            metadata={
                "runner": type(self.sandbox).__name__,
                "workers": workers,
                "task_timeout_sec": task_timeout_sec,
                "cache_policy": self.cache_policy,
                "task_distribution": self.task_distribution,
                "hard_task_timeout": bool(
                    self.runner_mode == "local"
                    and self.hard_task_timeout
                    and task_timeout_sec is not None
                ),
            },
            run_records=run_records,
        )
        report.regression_passed = self.regression_gate.check(report)
        log_event(
            "evaluation_finished",
            run_id=run_id,
            aggregate_score=report.aggregate_score,
            regression_passed=report.regression_passed,
        )
        return report

    def _run_local_process_workers(
        self,
        run_id: str,
        tasks: list[TaskSpec],
        workers: int,
        task_timeout_sec: float,
    ) -> list[tuple[TaskRunResult, RunRecord]]:
        scheduler = _LocalProcessScheduler(
            orchestrator=self,
            run_id=run_id,
            tasks=tasks,
            workers=workers,
            task_timeout_sec=task_timeout_sec,
            task_distribution=self.task_distribution,
        )
        return scheduler.run()

    def _evaluate_task(self, run_id: str, task: TaskSpec) -> tuple[TaskRunResult, RunRecord]:
        sandbox_result = self.sandbox.run_task(task)
        metadata = sandbox_result.metadata
        result = self.scorer.score_task(
            task,
            {
                "artifacts": sandbox_result.artifacts,
                "metadata": metadata,
            },
        )
        return result, self._run_record(run_id, task, result, metadata)

    def _error_result(
        self,
        run_id: str,
        task: TaskSpec,
        exc: Exception,
    ) -> tuple[TaskRunResult, RunRecord]:
        metadata = {
            "patch_hash": "error",
            "failure_type": "runner_error",
            "wall_time_sec": 0.0,
            "judge_time_sec": 0.0,
            "agent_step_latency_sec": 0.0,
            "cache_hit_flags": {},
            "benchmark_environment": {},
            "judge_details": [{"kind": "runner_error", "error": repr(exc)}],
        }
        result = TaskRunResult(
            task_id=task.task_id,
            status="error",
            score=0.0,
            details={"error": repr(exc)},
            artifacts={},
            patch_hash="error",
            failure_type="runner_error",
        )
        return result, self._run_record(run_id, task, result, metadata)

    def _timeout_result(
        self,
        run_id: str,
        task: TaskSpec,
        *,
        task_timeout_sec: float,
        elapsed_sec: float,
        killed_pid: int | None,
        kill_time_sec: float,
    ) -> tuple[TaskRunResult, RunRecord]:
        metadata = {
            "patch_hash": "task-timeout",
            "failure_type": "task_timeout",
            "wall_time_sec": elapsed_sec,
            "judge_time_sec": 0.0,
            "agent_step_latency_sec": 0.0,
            "cache_hit_flags": {},
            "benchmark_environment": {},
            "hard_timeout_triggered": True,
            "task_timeout_sec": task_timeout_sec,
            "timeout_elapsed_sec": elapsed_sec,
            "killed_pid": killed_pid,
            "timeout_process_kill_time_sec": kill_time_sec,
            "judge_details": [
                {
                    "kind": "task_timeout",
                    "passed": False,
                    "timeout_sec": task_timeout_sec,
                    "elapsed_sec": elapsed_sec,
                    "killed_pid": killed_pid,
                    "kill_time_sec": kill_time_sec,
                    "failure_type": "task_timeout",
                }
            ],
        }
        result = TaskRunResult(
            task_id=task.task_id,
            status="error",
            score=0.0,
            details={"error": "task exceeded hard timeout", "timeout_sec": task_timeout_sec},
            artifacts={},
            patch_hash="task-timeout",
            failure_type="task_timeout",
        )
        return result, self._run_record(run_id, task, result, metadata)

    def _run_record(
        self,
        run_id: str,
        task: TaskSpec,
        result: TaskRunResult,
        metadata: dict,
    ) -> RunRecord:
        return RunRecord(
            run_id=run_id,
            task_id=task.task_id,
            patch_hash=metadata.get("patch_hash", result.patch_hash or ""),
            pass_fail=(
                "pass"
                if result.status == "passed"
                else "error"
                if result.status == "error"
                else "fail"
            ),
            failure_type=metadata.get("failure_type") or result.failure_type,
            metrics=TaskMetrics(
                wall_time_sec=float(metadata.get("wall_time_sec", 0.0)),
                judge_time_sec=float(metadata.get("judge_time_sec", 0.0)),
                agent_step_latency_sec=float(metadata.get("agent_step_latency_sec", 0.0)),
                patch_apply_time_sec=float(metadata.get("patch_apply_time_sec", 0.0)),
                hidden_test_time_sec=float(metadata.get("hidden_test_time_sec", 0.0)),
                hidden_test_file_write_time_sec=float(
                    metadata.get("hidden_test_file_write_time_sec", 0.0)
                ),
                python_subprocess_startup_time_sec=float(
                    metadata.get("python_subprocess_startup_time_sec", 0.0)
                ),
                pytest_process_startup_time_sec=float(
                    metadata.get("pytest_process_startup_time_sec", 0.0)
                ),
                pytest_collection_time_sec=float(
                    metadata.get("pytest_collection_time_sec", 0.0)
                ),
                pytest_execution_time_sec=float(
                    metadata.get("pytest_execution_time_sec", 0.0)
                ),
                pytest_total_time_sec=float(metadata.get("pytest_total_time_sec", 0.0)),
                judge_cache_lookup_time_sec=float(
                    metadata.get("judge_cache_lookup_time_sec", 0.0)
                ),
                repo_checkout_time_sec=float(metadata.get("repo_checkout_time_sec", 0.0)),
                git_clone_time_sec=float(metadata.get("git_clone_time_sec", 0.0)),
                git_checkout_time_sec=float(metadata.get("git_checkout_time_sec", 0.0)),
                snapshot_materialize_time_sec=float(
                    metadata.get("snapshot_materialize_time_sec", 0.0)
                ),
                cache_lock_wait_time_sec=float(
                    metadata.get("cache_lock_wait_time_sec", 0.0)
                ),
                tempdir_create_time_sec=float(metadata.get("tempdir_create_time_sec", 0.0)),
                judge_workspace_cleanup_time_sec=float(
                    metadata.get("judge_workspace_cleanup_time_sec", 0.0)
                ),
                pytest_subprocess_count=int(metadata.get("pytest_subprocess_count", 0)),
                memory_peak_mb=metadata.get("memory_peak_mb"),
                cache_hit_flags=CacheHitFlags.model_validate(
                    metadata.get("cache_hit_flags", {})
                ),
            ),
            benchmark_environment=BenchmarkEnvironment.model_validate(
                metadata.get("benchmark_environment", {})
            ),
            metadata={
                "runner": type(self.sandbox).__name__,
                "checkout_strategy": metadata.get("checkout_strategy"),
                "cache_policy": metadata.get("cache_policy"),
                "worker_id": metadata.get("worker_id"),
                "affinity_key": metadata.get("affinity_key"),
                "repo_shard_stolen": metadata.get("repo_shard_stolen", False),
                "task_distribution": metadata.get("task_distribution"),
                "pytest_diagnostics": metadata.get("pytest_diagnostics"),
                "pytest_timeout_sec": metadata.get("pytest_timeout_sec"),
                "pytest_plugin_policy": metadata.get("pytest_plugin_policy"),
                "pytest_plugin_scan": metadata.get("pytest_plugin_scan"),
                "disable_pytest_plugin_autoload": metadata.get(
                    "disable_pytest_plugin_autoload"
                ),
                "hard_timeout_triggered": metadata.get("hard_timeout_triggered", False),
                "task_timeout_sec": metadata.get("task_timeout_sec"),
                "timeout_elapsed_sec": metadata.get("timeout_elapsed_sec"),
                "killed_pid": metadata.get("killed_pid"),
                "timeout_process_kill_time_sec": metadata.get(
                    "timeout_process_kill_time_sec"
                ),
                "judge_details": metadata.get("judge_details", []),
            },
        )


@dataclass
class _WorkerState:
    worker_id: int
    process: mp.Process
    input_queue: mp.Queue
    current_index: int | None = None
    current_task: TaskSpec | None = None
    started_at: float | None = None
    current_affinity_key: str | None = None
    current_stolen: bool = False


class _LocalProcessScheduler:
    def __init__(
        self,
        *,
        orchestrator: EvaluationOrchestrator,
        run_id: str,
        tasks: list[TaskSpec],
        workers: int,
        task_timeout_sec: float,
        task_distribution: str,
    ) -> None:
        self.orchestrator = orchestrator
        self.run_id = run_id
        self.tasks = tasks
        self.workers = max(1, workers)
        self.task_timeout_sec = max(0.001, float(task_timeout_sec))
        self.task_distribution = task_distribution
        self.context = mp.get_context("spawn")
        self.output_queue: mp.Queue = self.context.Queue()
        self.worker_config = {
            "checkout_strategy": orchestrator.checkout_strategy,
            "pytest_diagnostics": orchestrator.pytest_diagnostics,
            "pytest_timeout_sec": orchestrator.pytest_timeout_sec,
            "disable_pytest_plugin_autoload": orchestrator.disable_pytest_plugin_autoload,
            "pytest_plugin_policy": orchestrator.pytest_plugin_policy,
            "cache_policy": orchestrator.cache_policy,
            "shared_cache_root": orchestrator.shared_cache_root,
        }
        self.states: list[_WorkerState] = []
        self.pending_by_worker = self._build_pending_queues()
        self.home_worker_ids = {
            index for index, pending in enumerate(self.pending_by_worker) if pending
        }

    def run(self) -> list[tuple[TaskRunResult, RunRecord]]:
        if not self.tasks:
            return []
        self._start_workers(min(self.workers, len(self.tasks)))
        results: list[tuple[TaskRunResult, RunRecord] | None] = [None] * len(self.tasks)
        completed = 0

        try:
            self._fill_idle_workers()
            while completed < len(self.tasks):
                completed += self._collect_ready_results(results)
                timed_out = self._collect_timed_out_results(results)
                completed += timed_out
                if timed_out:
                    self._replace_dead_workers()
                self._fill_idle_workers()
                if completed < len(self.tasks):
                    time.sleep(0.005)
        finally:
            self._stop_workers()

        return [item for item in results if item is not None]

    def _start_workers(self, count: int) -> None:
        for worker_id in range(count):
            self.states.append(self._start_worker(worker_id))

    def _start_worker(self, worker_id: int) -> _WorkerState:
        input_queue: mp.Queue = self.context.Queue()
        process = self.context.Process(
            target=_local_task_worker_loop,
            args=(worker_id, input_queue, self.output_queue, self.worker_config),
        )
        process.start()
        return _WorkerState(worker_id=worker_id, process=process, input_queue=input_queue)

    def _build_pending_queues(self) -> list[list[int]]:
        queues = [[] for _ in range(min(self.workers, len(self.tasks)))]
        if self.task_distribution == "load":
            for index, _task in enumerate(self.tasks):
                queues[index % len(queues)].append(index)
            return queues

        key_to_worker: dict[str, int] = {}
        for index, task in enumerate(self.tasks):
            affinity_key = _task_affinity_key(task)
            if affinity_key not in key_to_worker:
                key_to_worker[affinity_key] = len(key_to_worker) % len(queues)
            queues[key_to_worker[affinity_key]].append(index)
        return queues

    def _next_task_for_worker(self, worker_id: int) -> tuple[int, bool] | None:
        if self.pending_by_worker[worker_id]:
            return self.pending_by_worker[worker_id].pop(0), False
        if self.task_distribution != "repo-shard-worksteal":
            return None
        if worker_id not in self.home_worker_ids:
            return None
        donor_id = max(
            range(len(self.pending_by_worker)),
            key=lambda index: len(self.pending_by_worker[index]),
        )
        if donor_id == worker_id or len(self.pending_by_worker[donor_id]) <= 1:
            return None
        return self.pending_by_worker[donor_id].pop(), True

    def _fill_idle_workers(self) -> None:
        for state in self.states:
            if state.current_task is not None or not state.process.is_alive():
                continue
            next_item = self._next_task_for_worker(state.worker_id)
            if next_item is None:
                continue
            task_index, stolen = next_item
            task = self.tasks[task_index]
            affinity_key = _task_affinity_key(task)
            state.input_queue.put(
                {
                    "task_index": task_index,
                    "task": task.model_dump(mode="json"),
                    "affinity_key": affinity_key,
                    "repo_shard_stolen": stolen,
                }
            )
            state.current_index = task_index
            state.current_task = task
            state.started_at = time.perf_counter()
            state.current_affinity_key = affinity_key
            state.current_stolen = stolen

    def _collect_ready_results(
        self,
        results: list[tuple[TaskRunResult, RunRecord] | None],
    ) -> int:
        count = 0
        while True:
            try:
                message = self.output_queue.get_nowait()
            except queue.Empty:
                return count
            worker_id = int(message["worker_id"])
            task_index = int(message["task_index"])
            state = self.states[worker_id]
            task = self.tasks[task_index]
            if message.get("ok"):
                result = TaskRunResult.model_validate(message["result"])
                metadata = dict(message["metadata"])
                metadata.update(
                    {
                        "worker_id": worker_id,
                        "affinity_key": message.get("affinity_key"),
                        "repo_shard_stolen": bool(message.get("repo_shard_stolen", False)),
                        "task_distribution": self.task_distribution,
                    }
                )
                evaluated = (
                    result,
                    self.orchestrator._run_record(self.run_id, task, result, metadata),
                )
            else:
                error = RuntimeError(str(message.get("error", "worker error")))
                result, record = self.orchestrator._error_result(self.run_id, task, error)
                metadata = dict(record.metadata)
                metadata.update(
                    {
                        "worker_id": worker_id,
                        "affinity_key": message.get("affinity_key"),
                        "repo_shard_stolen": bool(message.get("repo_shard_stolen", False)),
                        "task_distribution": self.task_distribution,
                    }
                )
                evaluated = result, record.model_copy(update={"metadata": metadata})
            results[task_index] = evaluated
            state.current_index = None
            state.current_task = None
            state.started_at = None
            state.current_affinity_key = None
            state.current_stolen = False
            count += 1

    def _collect_timed_out_results(
        self,
        results: list[tuple[TaskRunResult, RunRecord] | None],
    ) -> int:
        now = time.perf_counter()
        count = 0
        for state in self.states:
            if state.current_task is None or state.started_at is None:
                continue
            elapsed = now - state.started_at
            if elapsed <= self.task_timeout_sec:
                continue
            kill_start = time.perf_counter()
            killed_pid = state.process.pid
            _kill_process_group(state.process)
            kill_time = time.perf_counter() - kill_start
            assert state.current_index is not None
            results[state.current_index] = self.orchestrator._timeout_result(
                self.run_id,
                state.current_task,
                task_timeout_sec=self.task_timeout_sec,
                elapsed_sec=elapsed,
                killed_pid=killed_pid,
                kill_time_sec=kill_time,
            )
            result, record = results[state.current_index]
            metadata = dict(record.metadata)
            metadata.update(
                {
                    "worker_id": state.worker_id,
                    "affinity_key": state.current_affinity_key,
                    "repo_shard_stolen": state.current_stolen,
                    "task_distribution": self.task_distribution,
                }
            )
            results[state.current_index] = (
                result,
                record.model_copy(update={"metadata": metadata}),
            )
            state.current_index = None
            state.current_task = None
            state.started_at = None
            state.current_affinity_key = None
            state.current_stolen = False
            count += 1
        return count

    def _replace_dead_workers(self) -> None:
        for index, state in enumerate(self.states):
            if state.process.is_alive():
                continue
            state.process.join(timeout=0.1)
            self.states[index] = self._start_worker(state.worker_id)

    def _stop_workers(self) -> None:
        for state in self.states:
            if state.process.is_alive():
                state.input_queue.put(None)
        for state in self.states:
            state.process.join(timeout=1.0)
            if state.process.is_alive():
                _kill_process_group(state.process)


def _local_task_worker_loop(
    worker_id: int,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    config: dict[str, Any],
) -> None:
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass
    sandbox = LocalSandbox(**config)
    scorer = ScoreEngine()
    while True:
        message = input_queue.get()
        if message is None:
            return
        task_index = int(message["task_index"])
        affinity_key = message.get("affinity_key")
        repo_shard_stolen = bool(message.get("repo_shard_stolen", False))
        try:
            task = TaskSpec.model_validate(message["task"])
            sandbox_result = sandbox.run_task(task)
            result = scorer.score_task(
                task,
                {
                    "artifacts": sandbox_result.artifacts,
                    "metadata": sandbox_result.metadata,
                },
            )
            output_queue.put(
                {
                    "worker_id": worker_id,
                    "task_index": task_index,
                    "ok": True,
                    "result": result.model_dump(mode="json"),
                    "metadata": sandbox_result.metadata,
                    "affinity_key": affinity_key,
                    "repo_shard_stolen": repo_shard_stolen,
                }
            )
        except Exception as exc:
            output_queue.put(
                {
                    "worker_id": worker_id,
                    "task_index": task_index,
                    "ok": False,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "affinity_key": affinity_key,
                    "repo_shard_stolen": repo_shard_stolen,
                }
            )


def _task_affinity_key(task: TaskSpec) -> str:
    git_repo = task.fixtures.get("git_repo")
    if isinstance(git_repo, dict):
        return str(git_repo.get("repo_id") or git_repo.get("path") or task.task_id)
    toy_repo = task.fixtures.get("toy_repo")
    if isinstance(toy_repo, dict):
        return str(toy_repo.get("repo_id") or task.task_id)
    return task.task_id


def _kill_process_group(process: mp.Process) -> None:
    pid = process.pid
    if pid is None:
        return
    if hasattr(os, "killpg"):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            process.kill()
    else:
        process.kill()
    process.join(timeout=1.0)
