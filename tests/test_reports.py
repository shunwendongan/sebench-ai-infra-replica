import importlib.util
from pathlib import Path


def load_compare_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "compare_benchmark_reports.py"
    spec = importlib.util.spec_from_file_location("compare_benchmark_reports", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_throughput_delta_uses_higher_is_better_trend() -> None:
    compare = load_compare_module()

    assert compare.trend("throughput_tasks_per_sec", 10.0, 12.0) == "improved"
    assert compare.trend("throughput_tasks_per_sec", 12.0, 10.0) == "regressed"
