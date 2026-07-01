#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare NumPy CPU, PyTorch MPS, and MLX GPU on the same matmul workload."
    )
    parser.add_argument("--size", type=int, default=1024, help="Square matrix size N.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=5, help="Measured iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/apple_silicon_benchmark.json"),
        help="JSON output path.",
    )
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(
    name: str,
    device: str,
    size: int,
    warmup: int,
    iters: int,
    times: list[float],
) -> dict:
    mean = statistics.fmean(times)
    flops = 2 * size**3
    return {
        "name": name,
        "device": device,
        "size": size,
        "warmup": warmup,
        "iters": iters,
        "mean_seconds": mean,
        "median_seconds": statistics.median(times),
        "min_seconds": min(times),
        "p90_seconds": percentile(times, 0.90),
        "gflops_mean": flops / mean / 1e9,
        "all_seconds": times,
    }


def bench_numpy(size: int, warmup: int, iters: int, seed: int) -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(seed)
    a = rng.standard_normal((size, size), dtype=np.float32)
    b = rng.standard_normal((size, size), dtype=np.float32)
    for _ in range(warmup):
        _ = a @ b

    times: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        _ = a @ b
        times.append(time.perf_counter() - start)
    result = summarize("numpy_cpu", "CPU", size, warmup, iters, times)
    result["versions"] = {"numpy": np.__version__}
    return result


def bench_torch_mps(size: int, warmup: int, iters: int, seed: int) -> dict[str, Any]:
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError("torch MPS is not available")
    torch.manual_seed(seed)
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
    result = summarize("torch_mps", "mps", size, warmup, iters, times)
    result["versions"] = {"torch": torch.__version__}
    return result


def bench_mlx(size: int, warmup: int, iters: int, seed: int) -> dict[str, Any]:
    import mlx.core as mx
    import numpy as np

    rng = np.random.default_rng(seed)
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
    result = summarize("mlx_gpu", str(mx.default_device()), size, warmup, iters, times)
    result["versions"] = {"mlx": getattr(mx, "__version__", "unknown")}
    return result


def run_backend(name: str, fn, size: int, warmup: int, iters: int, seed: int) -> dict[str, Any]:
    try:
        result = fn(size, warmup, iters, seed)
        result["status"] = "ok"
        return result
    except Exception as exc:
        return {"name": name, "status": "skipped", "reason": repr(exc)}


def main() -> None:
    args = parse_args()
    results = [
        run_backend("numpy_cpu", bench_numpy, args.size, args.warmup, args.iters, args.seed),
        run_backend("torch_mps", bench_torch_mps, args.size, args.warmup, args.iters, args.seed),
        run_backend("mlx_gpu", bench_mlx, args.size, args.warmup, args.iters, args.seed),
    ]
    baseline = next(
        (r for r in results if r.get("name") == "numpy_cpu" and r["status"] == "ok"),
        None,
    )
    if baseline:
        baseline_mean = baseline["mean_seconds"]
        for result in results:
            if result["status"] == "ok":
                result["speedup_vs_numpy"] = baseline_mean / result["mean_seconds"]

    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "workload": {
            "operation": "C = A @ B",
            "dtype": "float32",
            "size": args.size,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
