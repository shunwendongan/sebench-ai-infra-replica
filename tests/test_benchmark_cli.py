import importlib.util
from pathlib import Path

from sebench_infra.benchmark.schemas import DatasetSpec, TaskCategory, TaskSpec


def load_run_mac_mlx_benchmark_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts/run_mac_mlx_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_mac_mlx_benchmark", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_worker_cap_limits_normal_runs() -> None:
    module = load_run_mac_mlx_benchmark_module()
    expected_cap = module.recommended_worker_cap(pytest_diagnostics=False)

    workers, cap = module.effective_worker_count(
        requested_workers=128,
        task_count=1024,
        pytest_diagnostics=False,
        pressure_test=False,
        adaptive_worker_cap=False,
    )

    assert workers == min(128, 1024, expected_cap)
    assert cap == expected_cap


def test_default_worker_cap_limits_diagnostics_runs_more_strictly() -> None:
    module = load_run_mac_mlx_benchmark_module()

    workers, cap = module.effective_worker_count(
        requested_workers=64,
        task_count=128,
        pytest_diagnostics=True,
        pressure_test=False,
        adaptive_worker_cap=False,
    )

    assert workers <= 32
    assert cap is not None


def test_pressure_test_preserves_requested_workers() -> None:
    module = load_run_mac_mlx_benchmark_module()

    workers, cap = module.effective_worker_count(
        requested_workers=128,
        task_count=1024,
        pytest_diagnostics=False,
        pressure_test=True,
        adaptive_worker_cap=False,
    )

    assert workers == 128
    assert cap is None


def test_adaptive_pressure_test_starts_with_requested_workers() -> None:
    module = load_run_mac_mlx_benchmark_module()

    workers, cap = module.effective_worker_count(
        requested_workers=64,
        task_count=128,
        pytest_diagnostics=False,
        pressure_test=True,
        adaptive_worker_cap=False,
        scheduler_policy="adaptive",
    )

    assert workers == 64
    assert cap is None


def test_adaptive_worker_count_halves_on_timeout_signal() -> None:
    module = load_run_mac_mlx_benchmark_module()

    workers, reason = module.next_adaptive_worker_count(
        32,
        {
            "pass_rate": 1.0,
            "hidden_judge_timeout_count": 1,
            "task_timeout_count": 0,
            "wall_time_p95_sec": 1.0,
            "pytest_execution_time_sec_avg": 0.2,
        },
        task_timeout_sec=30,
    )

    assert workers == 16
    assert reason == "timeout_or_pass_rate_backpressure"


def test_adaptive_worker_count_reduces_on_pytest_saturation() -> None:
    module = load_run_mac_mlx_benchmark_module()

    workers, reason = module.next_adaptive_worker_count(
        16,
        {
            "pass_rate": 1.0,
            "hidden_judge_timeout_count": 0,
            "task_timeout_count": 0,
            "wall_time_p95_sec": 1.0,
            "pytest_execution_time_sec_avg": 0.8,
        },
        task_timeout_sec=30,
    )

    assert workers == 8
    assert reason == "pytest_saturation_backpressure"


def test_repo_affinity_cap_limits_hard_timeout_workers() -> None:
    module = load_run_mac_mlx_benchmark_module()
    tasks = []
    for index in range(16):
        tasks.append(
            TaskSpec(
                task_id=f"task.{index}",
                title="Affinity task",
                category=TaskCategory.CODE_REPAIR,
                prompt="noop",
                fixtures={"git_repo": {"repo_id": f"repo-{index % 4}"}},
            )
        )
    dataset = DatasetSpec(dataset_id="affinity", tasks=tasks)

    workers, reason = module.repo_affinity_worker_cap(
        dataset,
        8,
        hard_task_timeout=True,
        pressure_test=False,
    )

    assert workers == 4
    assert reason["reason"] == "repo_affinity_cap"


def test_auto_cache_policy_uses_process_for_repo_affinity_production() -> None:
    module = load_run_mac_mlx_benchmark_module()

    policy, reason = module.resolve_cache_policy(
        "auto",
        hard_task_timeout=True,
        pressure_test=False,
        repo_cap_reason={"reason": "repo_affinity_cap"},
    )

    assert policy == "process"
    assert reason["reason"] == "auto_repo_affinity_process"


def test_auto_cache_policy_uses_shared_for_pressure_tests() -> None:
    module = load_run_mac_mlx_benchmark_module()

    policy, reason = module.resolve_cache_policy(
        "auto",
        hard_task_timeout=True,
        pressure_test=True,
        repo_cap_reason=None,
    )

    assert policy == "shared"
    assert reason["reason"] == "auto_pressure_test_shared"
